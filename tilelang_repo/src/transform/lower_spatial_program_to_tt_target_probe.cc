/*!
 * \file lower_spatial_program_to_tt_target_probe.cc
 * \brief Read-only TT target intake probe for Phase B SpatialProgram contracts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"
#include "common/tt_hardware_model.h"

namespace tvm {
namespace tl {

using tvm::GlobalInfo;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
namespace sp = tvm::tl::spatial;
using tvm::tl::str;

namespace {

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return std::nullopt;
}

std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return str(Downcast<String>(value.value()));
  }
  return std::nullopt;
}

bool ContainsKind(const Array<String>& supported_kinds, const std::string& expected) {
  for (const String& supported_kind : supported_kinds) {
    if (str(supported_kind) == expected) {
      return true;
    }
  }
  return false;
}

bool IsNeutralAffinity(const std::string& affinity_kind) {
  static const std::unordered_set<std::string> kNeutralAffinities = {
      "ingress", "compute", "egress", "communication", "phase_boundary"};
  return kNeutralAffinities.count(affinity_kind);
}

bool IsTTLeakingAffinity(const std::string& affinity_kind) {
  return affinity_kind == "brisc" || affinity_kind == "trisc" || affinity_kind == "ncrisc";
}

void ValidateTaskIndexRange(const String& subject_name, const char* subject_kind,
                            const char* key_name, int64_t task_index, int task_count) {
  ICHECK_GE(task_index, 0)
      << "missing spatial contract: " << subject_kind << " " << str(subject_name) << " "
      << key_name << " out of bounds";
  ICHECK_LT(task_index, task_count)
      << "missing spatial contract: " << subject_kind << " " << str(subject_name) << " "
      << key_name << " out of bounds";
}

std::string DeriveOrderingKindForChannel(sp::SpatialChannelKind channel_kind,
                                         sp::SpatialChannelDeliveryKind delivery_kind) {
  switch (channel_kind) {
    case sp::SpatialChannelKind::kCarry:
      return "carry_handoff";
    case sp::SpatialChannelKind::kReduceMerge:
      return "reduction_completion";
    case sp::SpatialChannelKind::kGather:
    case sp::SpatialChannelKind::kScatter:
      return "selection_index_handoff";
    default:
      return delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized
                 ? "phase_boundary_materialization"
                 : "must_happen_before";
  }
}

std::string DeriveMaterializationKindForChannel(sp::SpatialChannelKind channel_kind,
                                                sp::SpatialChannelDeliveryKind delivery_kind) {
  if (delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized) {
    return "phase_boundary_materialization";
  }
  if (channel_kind == sp::SpatialChannelKind::kReduceMerge) {
    return "completion_visibility";
  }
  return "phase_boundary";
}

void ValidateChannelContract(const Channel& channel, const SpatialCapabilityModel& capability_model,
                             int task_count) {
  auto maybe_payload_kind = GetPayloadString(channel->payload, schema_key::kPayloadKind);
  ICHECK(maybe_payload_kind)
      << "missing spatial contract: channel " << str(channel->name) << " "
      << schema_key::kPayloadKind;
  auto maybe_delivery_kind = GetPayloadString(channel->payload, schema_key::kDeliveryKind);
  ICHECK(maybe_delivery_kind)
      << "missing spatial contract: channel " << str(channel->name) << " "
      << schema_key::kDeliveryKind;
  auto maybe_source_task_index = GetPayloadIndex(channel->payload, schema_key::kSourceTaskIndex);
  auto maybe_target_task_index = GetPayloadIndex(channel->payload, schema_key::kTargetTaskIndex);
  ICHECK(maybe_source_task_index && maybe_target_task_index)
      << "missing spatial contract: channel " << str(channel->name) << " source/target task index";
  ValidateTaskIndexRange(channel->name, "channel", schema_key::kSourceTaskIndex,
                         *maybe_source_task_index, task_count);
  ValidateTaskIndexRange(channel->name, "channel", schema_key::kTargetTaskIndex,
                         *maybe_target_task_index, task_count);

  auto parsed_channel_kind = sp::ParseSpatialChannelKind(str(channel->kind));
  auto parsed_payload_kind = sp::ParseSpatialChannelPayloadKind(*maybe_payload_kind);
  auto parsed_delivery_kind = sp::ParseSpatialChannelDeliveryKind(*maybe_delivery_kind);
  ICHECK(parsed_channel_kind)
      << "missing spatial contract: channel " << str(channel->name) << " kind";
  ICHECK(parsed_payload_kind)
      << "missing spatial contract: channel " << str(channel->name) << " payload kind parse";
  ICHECK(parsed_delivery_kind)
      << "missing spatial contract: channel " << str(channel->name) << " delivery kind parse";

  ICHECK(ContainsKind(capability_model->supported_flow_kinds, str(channel->kind)))
      << "missing spatial contract: channel " << str(channel->name) << " unsupported flow kind "
      << str(channel->kind);
  ICHECK(ContainsKind(capability_model->supported_payload_kinds, *maybe_payload_kind))
      << "missing spatial contract: channel " << str(channel->name)
      << " unsupported payload kind " << *maybe_payload_kind;
  ICHECK(ContainsKind(capability_model->supported_delivery_kinds, *maybe_delivery_kind))
      << "missing spatial contract: channel " << str(channel->name)
      << " unsupported delivery kind " << *maybe_delivery_kind;
  if (*maybe_payload_kind == sp::ToString(sp::SpatialChannelPayloadKind::kStateVersion)) {
    auto maybe_state_index = GetPayloadIndex(channel->payload, schema_key::kStateIndex);
    ICHECK(maybe_state_index)
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kStateIndex;
    auto maybe_source_version = GetPayloadString(channel->payload, schema_key::kSourceVersion);
    ICHECK(maybe_source_version && !maybe_source_version->empty())
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kSourceVersion;
    auto maybe_target_version = GetPayloadString(channel->payload, schema_key::kTargetVersion);
    ICHECK(maybe_target_version && !maybe_target_version->empty())
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kTargetVersion;
  }
}

void ValidatePlacementContract(const Placement& placement,
                              const SpatialCapabilityModel& capability_model, int task_count) {
  auto maybe_task_index = GetPayloadIndex(placement->payload, schema_key::kTaskIndex);
  ICHECK(maybe_task_index)
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kTaskIndex;
  ValidateTaskIndexRange(placement->name, "placement", schema_key::kTaskIndex, *maybe_task_index,
                         task_count);
  auto maybe_affinity_kind = GetPayloadString(placement->payload, schema_key::kAffinityKind);
  ICHECK(maybe_affinity_kind)
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kAffinityKind;
  auto maybe_obligation_kind = GetPayloadString(placement->payload, schema_key::kObligationKind);
  ICHECK(maybe_obligation_kind && !maybe_obligation_kind->empty())
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kObligationKind;
  auto maybe_placement_domain =
      GetPayloadString(placement->payload, schema_key::kPlacementDomain);
  ICHECK(maybe_placement_domain && !maybe_placement_domain->empty())
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kPlacementDomain;
  ICHECK(!IsTTLeakingAffinity(*maybe_affinity_kind))
      << "TT-leaking placement affinity: " << *maybe_affinity_kind;
  ICHECK(IsNeutralAffinity(*maybe_affinity_kind))
      << "missing spatial contract: placement " << str(placement->name)
      << " unsupported neutral affinity " << *maybe_affinity_kind;
  ICHECK(*maybe_placement_domain == str(capability_model->placement_domain))
      << "missing spatial contract: placement " << str(placement->name)
      << " placement_domain mismatch " << *maybe_placement_domain;
}

void ValidatePhaseContract(const ProgramPhase& phase) {
  auto maybe_phase_index = GetPayloadIndex(phase->payload, schema_key::kPhaseIndex);
  ICHECK(maybe_phase_index)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kPhaseIndex;
  auto maybe_task_indices = phase->payload.Get(String(schema_key::kTaskIndices));
  ICHECK(maybe_task_indices)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kTaskIndices;
  auto maybe_channel_indices = phase->payload.Get(String(schema_key::kChannelIndices));
  ICHECK(maybe_channel_indices)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kChannelIndices;
  auto maybe_closure_basis = GetPayloadString(phase->payload, schema_key::kClosureBasis);
  ICHECK(maybe_closure_basis && !maybe_closure_basis->empty())
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kClosureBasis;
}

void ValidateSyncEdgeContract(const SyncEdge& edge, int task_count) {
  auto maybe_source_task_index = GetPayloadIndex(edge->payload, schema_key::kSourceTaskIndex);
  auto maybe_target_task_index = GetPayloadIndex(edge->payload, schema_key::kTargetTaskIndex);
  ICHECK(maybe_source_task_index && maybe_target_task_index)
      << "missing spatial contract: sync edge " << str(edge->name) << " source/target task index";
  ValidateTaskIndexRange(edge->name, "sync edge", schema_key::kSourceTaskIndex,
                         *maybe_source_task_index, task_count);
  ValidateTaskIndexRange(edge->name, "sync edge", schema_key::kTargetTaskIndex,
                         *maybe_target_task_index, task_count);
  auto maybe_ordering_kind = GetPayloadString(edge->payload, schema_key::kOrderingKind);
  ICHECK(maybe_ordering_kind && !maybe_ordering_kind->empty())
      << "missing spatial contract: sync edge " << str(edge->name) << " "
      << schema_key::kOrderingKind;
  auto maybe_materialization_kind =
      GetPayloadString(edge->payload, schema_key::kMaterializationKind);
  ICHECK(maybe_materialization_kind && !maybe_materialization_kind->empty())
      << "missing spatial contract: sync edge " << str(edge->name) << " "
      << schema_key::kMaterializationKind;
}

void ValidateSyncEdgeCapability(const SyncEdge& edge,
                                const SpatialCapabilityModel& capability_model) {
  ICHECK(ContainsKind(capability_model->supported_sync_kinds, str(edge->kind)))
      << "missing spatial contract: sync edge " << str(edge->name) << " unsupported sync kind "
      << str(edge->kind);
  auto maybe_ordering_kind = GetPayloadString(edge->payload, schema_key::kOrderingKind);
  ICHECK(maybe_ordering_kind && ContainsKind(capability_model->supported_ordering_kinds,
                                             *maybe_ordering_kind))
      << "missing spatial contract: sync edge " << str(edge->name)
      << " unsupported ordering kind " << (maybe_ordering_kind ? *maybe_ordering_kind : "");
  auto maybe_materialization_kind =
      GetPayloadString(edge->payload, schema_key::kMaterializationKind);
  ICHECK(maybe_materialization_kind &&
         ContainsKind(capability_model->supported_materialization_kinds,
                      *maybe_materialization_kind))
      << "missing spatial contract: sync edge " << str(edge->name)
      << " unsupported materialization kind "
      << (maybe_materialization_kind ? *maybe_materialization_kind : "");
}

void ValidateLayoutContract(const SpatialLayout& layout) {
  auto maybe_domain_index = GetPayloadIndex(layout->payload, schema_key::kDomainIndex);
  ICHECK(maybe_domain_index)
      << "missing spatial contract: layout " << str(layout->name) << " "
      << schema_key::kDomainIndex;
  auto maybe_domain_transform_kind =
      GetPayloadString(layout->payload, schema_key::kDomainTransformKind);
  ICHECK(maybe_domain_transform_kind && !maybe_domain_transform_kind->empty())
      << "missing spatial contract: layout " << str(layout->name) << " "
      << schema_key::kDomainTransformKind;
}

void ValidateWorkPartitionContract(const WorkPartition& partition) {
  auto maybe_domain_index = GetPayloadIndex(partition->payload, schema_key::kDomainIndex);
  ICHECK(maybe_domain_index)
      << "missing spatial contract: work partition " << str(partition->name) << " "
      << schema_key::kDomainIndex;
  auto maybe_partition_family =
      GetPayloadString(partition->payload, schema_key::kPartitionFamily);
  ICHECK(maybe_partition_family && !maybe_partition_family->empty())
      << "missing spatial contract: work partition " << str(partition->name) << " "
      << schema_key::kPartitionFamily;
}

std::vector<int64_t> BuildTaskPhaseIndexByTask(const SpatialProgram& program) {
  std::vector<int64_t> phase_index_by_task(program->tasks.size(), -1);
  for (int task_index = 0; task_index < program->tasks.size(); ++task_index) {
    auto maybe_phase_index =
        GetPayloadIndex(program->tasks[task_index]->payload, schema_key::kPhaseIndex);
    ICHECK(maybe_phase_index)
        << "missing spatial contract: task " << str(program->tasks[task_index]->name) << " "
        << schema_key::kPhaseIndex;
    phase_index_by_task[task_index] = *maybe_phase_index;
  }
  return phase_index_by_task;
}

void ValidateCrossPhaseSyncCoverage(const SpatialProgram& program) {
  if (program->phases.size() <= 1) {
    return;
  }
  const std::vector<int64_t> task_phase_index_by_task = BuildTaskPhaseIndexByTask(program);
  std::unordered_set<std::string> covered_phase_pairs;
  for (const SyncEdge& edge : program->sync_edges) {
    auto maybe_source_task_index = GetPayloadIndex(edge->payload, schema_key::kSourceTaskIndex);
    auto maybe_target_task_index = GetPayloadIndex(edge->payload, schema_key::kTargetTaskIndex);
    auto maybe_ordering_kind = GetPayloadString(edge->payload, schema_key::kOrderingKind);
    auto maybe_materialization_kind =
        GetPayloadString(edge->payload, schema_key::kMaterializationKind);
    ICHECK(maybe_source_task_index && maybe_target_task_index)
        << "missing spatial contract: sync edge " << str(edge->name)
        << " source/target task index";
    const int64_t source_phase_index = task_phase_index_by_task[*maybe_source_task_index];
    const int64_t target_phase_index = task_phase_index_by_task[*maybe_target_task_index];
    if (source_phase_index != target_phase_index) {
      covered_phase_pairs.insert(std::to_string(source_phase_index) + "->" +
                                 std::to_string(target_phase_index) + "|" +
                                 *maybe_ordering_kind + "|" + *maybe_materialization_kind);
    }
  }

  std::unordered_set<std::string> required_phase_pairs;
  for (const Channel& channel : program->channels) {
    auto maybe_source_task_index =
        GetPayloadIndex(channel->payload, schema_key::kSourceTaskIndex);
    auto maybe_target_task_index =
        GetPayloadIndex(channel->payload, schema_key::kTargetTaskIndex);
    auto maybe_payload_kind = GetPayloadString(channel->payload, schema_key::kPayloadKind);
    auto maybe_delivery_kind = GetPayloadString(channel->payload, schema_key::kDeliveryKind);
    ICHECK(maybe_source_task_index && maybe_target_task_index)
        << "missing spatial contract: channel " << str(channel->name)
        << " source/target task index";
    ICHECK(maybe_payload_kind && maybe_delivery_kind)
        << "missing spatial contract: channel " << str(channel->name)
        << " payload/delivery kind";
    const int64_t source_phase_index = task_phase_index_by_task[*maybe_source_task_index];
    const int64_t target_phase_index = task_phase_index_by_task[*maybe_target_task_index];
    if (source_phase_index != target_phase_index) {
      auto parsed_channel_kind = sp::ParseSpatialChannelKind(str(channel->kind));
      auto parsed_delivery_kind = sp::ParseSpatialChannelDeliveryKind(*maybe_delivery_kind);
      ICHECK(parsed_channel_kind && parsed_delivery_kind)
          << "missing spatial contract: channel " << str(channel->name)
          << " flow/delivery kind parse";
      required_phase_pairs.insert(std::to_string(source_phase_index) + "->" +
                                  std::to_string(target_phase_index) + "|" +
                                  DeriveOrderingKindForChannel(*parsed_channel_kind,
                                                               *parsed_delivery_kind) + "|" +
                                  DeriveMaterializationKindForChannel(*parsed_channel_kind,
                                                                      *parsed_delivery_kind));
    }
  }
  for (const std::string& required_pair : required_phase_pairs) {
    ICHECK(covered_phase_pairs.count(required_pair))
        << "missing spatial contract: cross-phase channel coverage requires sync edge";
  }
}

void ValidateResourceIntentCapability(const ResourceIntent& intent,
                                      const SpatialCapabilityModel& capability_model) {
  ICHECK(ContainsKind(capability_model->supported_resource_intent_kinds, str(intent->kind)))
      << "missing spatial contract: resource intent unsupported by capability model "
      << str(intent->kind);
}

std::optional<Target> FindBlackholeTarget(const IRModule& mod) {
  for (const auto& [gvar, base_func] : mod->functions) {
    auto func = base_func.as<tir::PrimFunc>();
    if (!func || !IsBlackholePrimFunc(func.value())) {
      continue;
    }
    auto maybe_target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
    if (maybe_target) {
      return maybe_target.value();
    }
  }
  return std::nullopt;
}

}  // namespace

tvm::transform::Pass LowerSpatialProgramToTTTargetProbe() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = GetModuleTTHardwareModel(mod);
    auto maybe_capability_model = GetModuleSpatialCapabilityModel(mod);
    if (!maybe_hardware_model || !maybe_capability_model) {
      auto maybe_target = FindBlackholeTarget(mod);
      ICHECK(maybe_target)
          << "LowerSpatialProgramToTTTargetProbe requires blackhole target to derive "
             "hardware/capability model";
      if (!maybe_hardware_model) {
        maybe_hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
      }
      if (!maybe_capability_model) {
        maybe_capability_model = DeriveSpatialCapabilityModel(maybe_hardware_model.value());
      }
      mod = mod->ShallowCopy();
      mod->UpdateGlobalInfo(attr::kTLTTHardwareModel,
                            Array<GlobalInfo>{maybe_hardware_model.value()});
      mod->UpdateGlobalInfo(attr::kTLSpatialCapabilityModel,
                            Array<GlobalInfo>{maybe_capability_model.value()});
    }

    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_spatial_program = func.value()->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (!maybe_spatial_program) {
        continue;
      }
      const SpatialProgram& program = maybe_spatial_program.value();
      for (const Channel& channel : program->channels) {
        ValidateChannelContract(channel, maybe_capability_model.value(), program->tasks.size());
      }
      for (const Placement& placement : program->placements) {
        ValidatePlacementContract(placement, maybe_capability_model.value(), program->tasks.size());
      }
      for (const ProgramPhase& phase : program->phases) {
        ValidatePhaseContract(phase);
      }
      for (const SyncEdge& edge : program->sync_edges) {
        ValidateSyncEdgeContract(edge, program->tasks.size());
        ValidateSyncEdgeCapability(edge, maybe_capability_model.value());
      }
      ValidateCrossPhaseSyncCoverage(program);
      for (const SpatialLayout& layout : program->layouts) {
        ValidateLayoutContract(layout);
        ICHECK(ContainsKind(maybe_capability_model.value()->supported_layout_kinds,
                            str(layout->kind)))
            << "missing spatial contract: layout unsupported by capability model "
            << str(layout->kind);
      }
      for (const WorkPartition& partition : program->work_partitions) {
        ValidateWorkPartitionContract(partition);
        ICHECK(ContainsKind(maybe_capability_model.value()->supported_partition_kinds,
                            str(partition->kind)))
            << "missing spatial contract: partition unsupported by capability model "
            << str(partition->kind);
      }
      for (const ResourceIntent& intent : program->resource_intents) {
        ValidateResourceIntentCapability(intent, maybe_capability_model.value());
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.LowerSpatialProgramToTTTargetProbe", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSpatialProgramToTTTargetProbe",
                        LowerSpatialProgramToTTTargetProbe);
}

}  // namespace tl
}  // namespace tvm
