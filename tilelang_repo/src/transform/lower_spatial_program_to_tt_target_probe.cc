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

void ValidateChannelContract(const Channel& channel, const SpatialCapabilityModel& capability_model) {
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
}

void ValidatePlacementContract(const Placement& placement) {
  auto maybe_task_index = GetPayloadIndex(placement->payload, schema_key::kTaskIndex);
  ICHECK(maybe_task_index)
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kTaskIndex;
  auto maybe_affinity_kind = GetPayloadString(placement->payload, schema_key::kAffinityKind);
  ICHECK(maybe_affinity_kind)
      << "missing spatial contract: placement " << str(placement->name) << " "
      << schema_key::kAffinityKind;
  ICHECK(!IsTTLeakingAffinity(*maybe_affinity_kind))
      << "TT-leaking placement affinity: " << *maybe_affinity_kind;
  ICHECK(IsNeutralAffinity(*maybe_affinity_kind))
      << "missing spatial contract: placement " << str(placement->name)
      << " unsupported neutral affinity " << *maybe_affinity_kind;
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
        ValidateChannelContract(channel, maybe_capability_model.value());
      }
      for (const Placement& placement : program->placements) {
        ValidatePlacementContract(placement);
      }
      for (const SpatialLayout& layout : program->layouts) {
        ICHECK(ContainsKind(maybe_capability_model.value()->supported_layout_kinds,
                            str(layout->kind)))
            << "missing spatial contract: layout unsupported by capability model "
            << str(layout->kind);
      }
      for (const WorkPartition& partition : program->work_partitions) {
        ICHECK(ContainsKind(maybe_capability_model.value()->supported_partition_kinds,
                            str(partition->kind)))
            << "missing spatial contract: partition unsupported by capability model "
            << str(partition->kind);
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
