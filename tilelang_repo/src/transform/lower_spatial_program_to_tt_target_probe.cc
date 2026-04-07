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
#include "common/spatial_analysis.h"
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

void ValidateChannelContract(const Channel& channel, const SpatialCapabilityModel& capability_model,
                             int task_count) {
  const std::string payload_kind = str(channel->payload_kind);
  ICHECK(!payload_kind.empty())
      << "missing spatial contract: channel " << str(channel->name) << " "
      << schema_key::kPayloadKind;
  const std::string delivery_kind = str(channel->delivery_kind);
  ICHECK(!delivery_kind.empty())
      << "missing spatial contract: channel " << str(channel->name) << " "
      << schema_key::kDeliveryKind;
  ICHECK(channel->source_task_index >= 0 && channel->target_task_index >= 0)
      << "missing spatial contract: channel " << str(channel->name) << " source/target task index";
  ValidateTaskIndexRange(channel->name, "channel", schema_key::kSourceTaskIndex,
                         channel->source_task_index, task_count);
  ValidateTaskIndexRange(channel->name, "channel", schema_key::kTargetTaskIndex,
                         channel->target_task_index, task_count);

  auto parsed_channel_kind = sp::ParseSpatialChannelKind(str(channel->kind));
  auto parsed_payload_kind = sp::ParseSpatialChannelPayloadKind(payload_kind);
  auto parsed_delivery_kind = sp::ParseSpatialChannelDeliveryKind(delivery_kind);
  ICHECK(parsed_channel_kind)
      << "missing spatial contract: channel " << str(channel->name) << " kind";
  ICHECK(parsed_payload_kind)
      << "missing spatial contract: channel " << str(channel->name) << " payload kind parse";
  ICHECK(parsed_delivery_kind)
      << "missing spatial contract: channel " << str(channel->name) << " delivery kind parse";

  ICHECK(ContainsKind(capability_model->supported_flow_kinds, str(channel->kind)))
      << "missing spatial contract: channel " << str(channel->name) << " unsupported flow kind "
      << str(channel->kind);
  ICHECK(ContainsKind(capability_model->supported_payload_kinds, payload_kind))
      << "missing spatial contract: channel " << str(channel->name)
      << " unsupported payload kind " << payload_kind;
  ICHECK(ContainsKind(capability_model->supported_delivery_kinds, delivery_kind))
      << "missing spatial contract: channel " << str(channel->name)
      << " unsupported delivery kind " << delivery_kind;
  if (payload_kind == sp::ToString(sp::SpatialChannelPayloadKind::kStateVersion)) {
    ICHECK(channel->state_index >= 0)
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kStateIndex;
    ICHECK(!str(channel->source_version).empty())
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kSourceVersion;
    ICHECK(!str(channel->target_version).empty())
        << "missing spatial contract: channel " << str(channel->name) << " "
        << schema_key::kTargetVersion;
  }
}

void ValidatePlacementContract(const Placement& placement,
                              const SpatialCapabilityModel& capability_model, int task_count) {
  ICHECK(placement->task_index >= 0)
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kTaskIndex;
  ValidateTaskIndexRange(placement->name, "placement", schema_key::kTaskIndex, placement->task_index,
                         task_count);
  const std::string affinity_kind = str(placement->affinity_kind);
  ICHECK(!affinity_kind.empty())
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kAffinityKind;
  ICHECK(!str(placement->obligation_kind).empty())
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kObligationKind;
  ICHECK(!str(placement->placement_domain).empty())
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kPlacementDomain;
  ICHECK(!IsTTLeakingAffinity(affinity_kind))
      << "TT-leaking placement affinity: " << affinity_kind;
  ICHECK(IsNeutralAffinity(affinity_kind))
      << "missing spatial contract: placement " << str(placement->name)
      << " unsupported neutral affinity " << affinity_kind;
  ICHECK(str(placement->placement_domain) == str(capability_model->placement_domain))
      << "missing spatial contract: placement " << str(placement->name)
      << " placement_domain mismatch " << str(placement->placement_domain);
}

void ValidatePhaseContract(const ProgramPhase& phase) {
  ICHECK(phase->phase_index >= 0)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kPhaseIndex;
  auto maybe_task_indices = GetPayloadIndices(phase->payload, schema_key::kTaskIndices);
  ICHECK(maybe_task_indices)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kTaskIndices;
  auto maybe_channel_indices = GetPayloadIndices(phase->payload, schema_key::kChannelIndices);
  ICHECK(maybe_channel_indices)
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kChannelIndices;
  ICHECK(!str(phase->closure_basis).empty())
      << "missing spatial contract: phase " << str(phase->name) << " "
      << schema_key::kClosureBasis;
}

void ValidateSyncEdgeContract(const SyncEdge& edge, int task_count) {
  ICHECK(edge->source_task_index >= 0 && edge->target_task_index >= 0)
      << "missing spatial contract: sync edge " << str(edge->name) << " source/target task index";
  ValidateTaskIndexRange(edge->name, "sync edge", schema_key::kSourceTaskIndex,
                         edge->source_task_index, task_count);
  ValidateTaskIndexRange(edge->name, "sync edge", schema_key::kTargetTaskIndex,
                         edge->target_task_index, task_count);
  ICHECK(!str(edge->ordering_kind).empty())
      << "missing spatial contract: sync edge " << str(edge->name) << " "
      << schema_key::kOrderingKind;
  ICHECK(!str(edge->materialization_kind).empty())
      << "missing spatial contract: sync edge " << str(edge->name) << " "
      << schema_key::kMaterializationKind;
}

void ValidateSyncEdgeCapability(const SyncEdge& edge,
                                const SpatialCapabilityModel& capability_model) {
  ICHECK(ContainsKind(capability_model->supported_sync_kinds, str(edge->kind)))
      << "missing spatial contract: sync edge " << str(edge->name) << " unsupported sync kind "
      << str(edge->kind);
  const std::string ordering_kind = str(edge->ordering_kind);
  ICHECK(!ordering_kind.empty() &&
         ContainsKind(capability_model->supported_ordering_kinds, ordering_kind))
      << "missing spatial contract: sync edge " << str(edge->name)
      << " unsupported ordering kind " << ordering_kind;
  const std::string materialization_kind = str(edge->materialization_kind);
  ICHECK(!materialization_kind.empty() &&
         ContainsKind(capability_model->supported_materialization_kinds,
                      materialization_kind))
      << "missing spatial contract: sync edge " << str(edge->name)
      << " unsupported materialization kind " << materialization_kind;
}

void ValidateLayoutContract(const SpatialLayout& layout) {
  ICHECK(layout->domain_index >= 0)
      << "missing spatial contract: layout " << str(layout->name) << " "
      << schema_key::kDomainIndex;
  ICHECK(!str(layout->domain_transform_kind).empty())
      << "missing spatial contract: layout " << str(layout->name) << " "
      << schema_key::kDomainTransformKind;
}

void ValidateWorkPartitionContract(const WorkPartition& partition) {
  ICHECK(partition->domain_index >= 0)
      << "missing spatial contract: work partition " << str(partition->name) << " "
      << schema_key::kDomainIndex;
  ICHECK(!str(partition->partition_family).empty())
      << "missing spatial contract: work partition " << str(partition->name) << " "
      << schema_key::kPartitionFamily;
}

std::vector<int64_t> BuildTaskPhaseIndexByTask(const SpatialProgram& program) {
  std::vector<int64_t> phase_index_by_task(program->tasks.size(), -1);
  for (int task_index = 0; task_index < program->tasks.size(); ++task_index) {
    ICHECK(program->tasks[task_index]->phase_index >= 0)
        << "missing spatial contract: task " << str(program->tasks[task_index]->name) << " "
        << schema_key::kPhaseIndex;
    phase_index_by_task[task_index] = program->tasks[task_index]->phase_index;
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
    ICHECK(edge->source_task_index >= 0 && edge->target_task_index >= 0)
        << "missing spatial contract: sync edge " << str(edge->name)
        << " source/target task index";
    const int64_t source_phase_index = task_phase_index_by_task[edge->source_task_index];
    const int64_t target_phase_index = task_phase_index_by_task[edge->target_task_index];
    if (source_phase_index != target_phase_index) {
      covered_phase_pairs.insert(std::to_string(source_phase_index) + "->" +
                                 std::to_string(target_phase_index) + "|" +
                                 str(edge->ordering_kind) + "|" +
                                 str(edge->materialization_kind));
    }
  }

  std::unordered_set<std::string> required_phase_pairs;
  for (const Channel& channel : program->channels) {
    ICHECK(channel->source_task_index >= 0 && channel->target_task_index >= 0)
        << "missing spatial contract: channel " << str(channel->name)
        << " source/target task index";
    ICHECK(!str(channel->payload_kind).empty() && !str(channel->delivery_kind).empty())
        << "missing spatial contract: channel " << str(channel->name)
        << " payload/delivery kind";
    const int64_t source_phase_index = task_phase_index_by_task[channel->source_task_index];
    const int64_t target_phase_index = task_phase_index_by_task[channel->target_task_index];
    if (source_phase_index != target_phase_index) {
      auto parsed_channel_kind = sp::ParseSpatialChannelKind(str(channel->kind));
      auto parsed_delivery_kind = sp::ParseSpatialChannelDeliveryKind(str(channel->delivery_kind));
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
