/*!
 * \file lower_spatial_program_to_tt_target.cc
 * \brief Materialize TTProgram from frozen SpatialProgram and typed target companion attrs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_program.h"
#include "common/tt_hardware_model.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

using tvm::GlobalInfo;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

tir::PrimFunc StripLegacyTTProjectionAttrs(tir::PrimFunc func) {
  static const char* kLegacyProjectionAttrs[] = {
      "blackhole.segment_plan",
      "blackhole.runtime_args",
      "blackhole.common_runtime_args",
      "blackhole.accessors",
      "blackhole.cb_configs",
      "blackhole.cb_bindings",
      "blackhole.total_l1_bytes",
      "blackhole.num_cbs",
      "blackhole.semaphore_plan",
      "blackhole.core_plan",
      "blackhole.grid_shape",
      "blackhole.grid_x",
      "blackhole.grid_y",
      "blackhole.cores_needed",
      "blackhole.work_per_core",
      "blackhole.core_type",
      "blackhole.gemm_contract",
      "blackhole.compute_contract",
      "blackhole.direct_runtime_unsupported_reasons",
  };
  for (const char* key : kLegacyProjectionAttrs) {
    func = tvm::WithoutAttr(std::move(func), key);
  }
  return func;
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

int64_t GetIntOrDefault(const Map<String, Any>& dict, const char* key, int64_t default_value = -1) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

String GetStringOrDefault(const Map<String, Any>& dict, const char* key, String default_value = String()) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<String>(value.value());
  }
  return default_value;
}

Array<Any> GetArrayOrEmpty(const Map<String, Any>& dict, const char* key) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<Array<Any>>(value.value());
  }
  return Array<Any>();
}

Map<String, Any> GetMapOrEmpty(const Map<String, Any>& dict, const char* key) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<Map<String, Any>>(value.value());
  }
  return Map<String, Any>();
}

bool HasAny(const Map<String, Any>& dict, const char* key) { return dict.Get(String(key)).has_value(); }

Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

Array<TTCBPlan> BuildCBPlans(const tir::PrimFunc& func) {
  auto maybe_cb_plans = func->GetAttr<Array<TTCBPlan>>(attr::kTLTTCBPlans);
  ICHECK(maybe_cb_plans)
      << "LowerSpatialProgramToTTTarget requires " << attr::kTLTTCBPlans
      << " on Blackhole device PrimFunc";
  return maybe_cb_plans.value();
}

Array<TTCoreGroup> BuildCoreGroups(const tir::PrimFunc& func) {
  auto maybe_core_groups = func->GetAttr<Array<TTCoreGroup>>(attr::kTLTTCoreGroups);
  ICHECK(maybe_core_groups)
      << "LowerSpatialProgramToTTTarget requires " << attr::kTLTTCoreGroups
      << " on Blackhole device PrimFunc";
  return maybe_core_groups.value();
}

Array<TTTransportPlan> BuildTransportPlans(const SpatialProgram& program) {
  Array<TTTransportPlan> transport_plans;
  for (const Channel& channel : program->channels) {
    transport_plans.push_back(TTTransportPlan(channel->name, channel->kind,
                                              channel->source_task_index, channel->target_task_index,
                                              channel->payload_kind, channel->delivery_kind,
                                              channel->payload));
  }
  return transport_plans;
}

Array<TTSemaphorePlan> BuildSemaphorePlans(const tir::PrimFunc& func, const SpatialProgram& program) {
  if (auto maybe_semaphore_plans = func->GetAttr<Array<TTSemaphorePlan>>(attr::kTLTTSemaphorePlans)) {
    return maybe_semaphore_plans.value();
  }
  return Array<TTSemaphorePlan>();
}

Array<TTComputeSyncPlan> BuildComputeSyncPlans(const SpatialProgram& program) {
  Array<TTComputeSyncPlan> sync_plans;
  for (const SyncEdge& edge : program->sync_edges) {
    sync_plans.push_back(TTComputeSyncPlan(edge->name, edge->kind, edge->source_task_index,
                                           edge->target_task_index, edge->ordering_kind,
                                           edge->materialization_kind, edge->payload));
  }
  return sync_plans;
}

Array<TTDstLayoutPlan> BuildDstLayoutPlans(const SpatialProgram& program, const Array<TTABIPlan>& abi_plans) {
  Array<TTDstLayoutPlan> dst_layouts;
  std::unordered_set<std::string> seen;
  for (const SpatialLayout& layout : program->layouts) {
    std::string dedupe = str(layout->target_name) + "|" + str(layout->kind) + "|" +
                         str(layout->domain_transform_kind);
    if (!seen.insert(dedupe).second) {
      continue;
    }
    dst_layouts.push_back(TTDstLayoutPlan(layout->name, layout->target_name, layout->kind,
                                          String("dram"), layout->payload));
  }
  for (const TTABIPlan& abi : abi_plans) {
    for (const Any& item : abi->compile_time_arg_specs) {
      Map<String, Any> spec = AsMap(item);
      String buffer = GetStringOrDefault(spec, "buffer");
      String layout = GetStringOrDefault(spec, "layout");
      String memory_space = GetStringOrDefault(spec, "memory_space");
      if (buffer.empty() || layout.empty() || memory_space.empty()) {
        continue;
      }
      std::string dedupe = str(buffer) + "|" + str(layout) + "|" + str(memory_space);
      if (!seen.insert(dedupe).second) {
        continue;
      }
      dst_layouts.push_back(
          TTDstLayoutPlan(String("dst_layout_" + dedupe), buffer, layout, memory_space, spec));
    }
  }
  return dst_layouts;
}

Array<TTABIPlan> BuildABIPlans(const tir::PrimFunc& func, Array<TTKernel>* kernels_out) {
  auto maybe_kernels = func->GetAttr<Array<TTKernel>>(attr::kTLTTKernelSeeds);
  auto maybe_abi_plans = func->GetAttr<Array<TTABIPlan>>(attr::kTLTTABIPlans);
  ICHECK(maybe_kernels)
      << "LowerSpatialProgramToTTTarget requires " << attr::kTLTTKernelSeeds
      << " on Blackhole device PrimFunc";
  ICHECK(maybe_abi_plans)
      << "LowerSpatialProgramToTTTarget requires " << attr::kTLTTABIPlans
      << " on Blackhole device PrimFunc";
  const Array<TTKernel>& kernels = maybe_kernels.value();
  const Array<TTABIPlan>& abi_plans = maybe_abi_plans.value();
  ICHECK_EQ(kernels.size(), abi_plans.size())
      << "TT target typed companion seeds must keep kernel/ABI cardinality aligned";
  *kernels_out = kernels;
  return abi_plans;
}

Array<TTExecutionPlan> BuildExecutionPlans(const SpatialProgram& program, const Array<TTKernel>& kernels) {
  Array<ffi::String> kernel_names;
  for (const TTKernel& kernel : kernels) {
    kernel_names.push_back(kernel->name);
  }
  Array<Integer> phase_indices;
  for (const ProgramPhase& phase : program->phases) {
    phase_indices.push_back(Integer(phase->phase_index));
  }
  Map<String, Any> payload;
  payload.Set("phase_count", Integer(static_cast<int>(program->phases.size())));
  Array<TTExecutionPlan> execution_plans;
  execution_plans.push_back(TTExecutionPlan(String("main_execution"), kernel_names, phase_indices, payload));
  return execution_plans;
}

TTProgram BuildTTProgramForFunc(const tir::PrimFunc& func, const String& entry_name,
                                const SpatialProgram& program,
                                const TTHardwareModel& hardware_model) {
  Array<TTKernel> kernels;
  Array<TTABIPlan> abi_plans = BuildABIPlans(func, &kernels);
  Map<String, Any> payload =
      func->GetAttr<Map<String, Any>>(attr::kTLTTProgramPayload).value_or(Map<String, Any>());
  payload.Set("arch_name", hardware_model->arch_name);
  payload.Set("logical_worker_grid_x", Integer(hardware_model->logical_worker_grid_x));
  payload.Set("logical_worker_grid_y", Integer(hardware_model->logical_worker_grid_y));
  payload.Set("worker_l1_size", Integer(hardware_model->worker_l1_size));
  return TTProgram(entry_name, program->member_func, kernels, BuildCoreGroups(func), BuildCBPlans(func),
                   BuildTransportPlans(program), BuildSemaphorePlans(func, program),
                   BuildComputeSyncPlans(program), BuildDstLayoutPlans(program, abi_plans), abi_plans,
                   BuildExecutionPlans(program, kernels), payload);
}

}  // namespace

tvm::transform::Pass LowerSpatialProgramToTTTarget() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = GetModuleTTHardwareModel(mod);
    if (!maybe_hardware_model) {
      auto maybe_target = FindBlackholeTarget(mod);
      ICHECK(maybe_target) << "LowerSpatialProgramToTTTarget requires blackhole target";
      maybe_hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
      mod = mod->ShallowCopy();
      mod->UpdateGlobalInfo(attr::kTLTTHardwareModel, Array<GlobalInfo>{maybe_hardware_model.value()});
    }
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_spatial_program = func.value()->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (!maybe_spatial_program) {
        continue;
      }
      tir::PrimFunc rewritten = func.value();
      Map<String, Any> attrs;
      if (rewritten->attrs.defined()) {
        attrs = rewritten->attrs->dict;
      }
      attrs.Set(attr::kTLTTProgram,
                BuildTTProgramForFunc(func.value(), gvar->name_hint, maybe_spatial_program.value(),
                                      maybe_hardware_model.value()));
      rewritten.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
      rewritten = StripLegacyTTProjectionAttrs(std::move(rewritten));
      updated->Add(gvar, rewritten, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.LowerSpatialProgramToTTTarget", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSpatialProgramToTTTarget", LowerSpatialProgramToTTTarget);
}

}  // namespace tl
}  // namespace tvm
