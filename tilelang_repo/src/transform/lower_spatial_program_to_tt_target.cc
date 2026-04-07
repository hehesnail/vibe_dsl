/*!
 * \file lower_spatial_program_to_tt_target.cc
 * \brief Materialize TTProgram from frozen SpatialProgram and bridge target attrs.
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
  Array<TTCBPlan> cb_plans;
  auto maybe_cb_configs = func->GetAttr<Array<Any>>("blackhole.cb_configs");
  if (!maybe_cb_configs) {
    return cb_plans;
  }
  for (const Any& item : maybe_cb_configs.value()) {
    Map<String, Any> cb = AsMap(item);
    if (cb.empty()) {
      continue;
    }
    cb_plans.push_back(TTCBPlan(GetStringOrDefault(cb, "name"),
                                GetIntOrDefault(cb, "cb_id"),
                                GetStringOrDefault(cb, "role"),
                                GetIntOrDefault(cb, "num_pages", 0),
                                GetIntOrDefault(cb, "page_size", 0),
                                GetStringOrDefault(cb, "data_format"),
                                cb));
  }
  return cb_plans;
}

Array<TTCoreGroup> BuildCoreGroups(const tir::PrimFunc& func) {
  Array<TTCoreGroup> core_groups;
  auto maybe_core_plan = func->GetAttr<Map<String, Any>>("blackhole.core_plan");
  if (!maybe_core_plan) {
    return core_groups;
  }
  Map<String, Any> core_plan = maybe_core_plan.value();
  core_groups.push_back(TTCoreGroup(String("main_core_group"),
                                    GetIntOrDefault(core_plan, "logical_grid_x", 1),
                                    GetIntOrDefault(core_plan, "logical_grid_y", 1),
                                    GetStringOrDefault(core_plan, "linearization", String("row_major")),
                                    GetArrayOrEmpty(core_plan, "physical_cores"),
                                    GetArrayOrEmpty(core_plan, "work_packets"),
                                    core_plan));
  return core_groups;
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
  Array<TTSemaphorePlan> semaphore_plans;
  auto maybe_semaphore_plan = func->GetAttr<Array<Any>>("blackhole.semaphore_plan");
  if (maybe_semaphore_plan) {
    int edge_index = 0;
    for (const Any& item : maybe_semaphore_plan.value()) {
      Map<String, Any> sem = AsMap(item);
      if (sem.empty()) {
        continue;
      }
      int64_t source_task_index = -1;
      int64_t target_task_index = -1;
      if (edge_index < program->sync_edges.size()) {
        source_task_index = program->sync_edges[edge_index]->source_task_index;
        target_task_index = program->sync_edges[edge_index]->target_task_index;
      }
      semaphore_plans.push_back(TTSemaphorePlan(
          GetStringOrDefault(sem, "name", String("semaphore_" + std::to_string(edge_index))),
          String("barrier"), GetIntOrDefault(sem, "id"), GetIntOrDefault(sem, "initial_value", 0),
          GetStringOrDefault(sem, "core_type", String("worker")), source_task_index,
          target_task_index, GetArrayOrEmpty(sem, "core_ranges"), sem));
      ++edge_index;
    }
  }
  return semaphore_plans;
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

Array<TTABIPlan> BuildABIPlans(const tir::PrimFunc& func, const Array<Any>& segment_plan,
                               Array<TTKernel>* kernels_out) {
  Array<TTABIPlan> abi_plans;
  Array<TTKernel> kernels;
  Array<Any> top_runtime_args =
      func->GetAttr<Array<Any>>("blackhole.runtime_args").value_or(Array<Any>());
  Array<Any> top_common_runtime_args =
      func->GetAttr<Array<Any>>("blackhole.common_runtime_args").value_or(Array<Any>());
  if (!segment_plan.empty()) {
    int index = 0;
    for (const Any& item : segment_plan) {
      Map<String, Any> segment = AsMap(item);
      if (segment.empty()) {
        continue;
      }
      String kernel_name = GetStringOrDefault(segment, "name",
                                              String("kernel_" + std::to_string(index)));
      String kernel_kind = GetStringOrDefault(segment, "kind", String("fused_dataflow"));
      String core_type = GetStringOrDefault(segment, "core_type", String("brisc"));
      Array<Any> runtime_args = GetArrayOrEmpty(segment, "runtime_args");
      Array<Any> common_runtime_args = GetArrayOrEmpty(segment, "common_runtime_args");
      const bool uses_top_level_runtime_args = runtime_args.empty() && !top_runtime_args.empty();
      const bool uses_top_level_common_runtime_args =
          common_runtime_args.empty() && !top_common_runtime_args.empty();
      if (runtime_args.empty()) {
        runtime_args = top_runtime_args;
      }
      if (common_runtime_args.empty()) {
        common_runtime_args = top_common_runtime_args;
      }
      segment.Set("tt_uses_top_level_runtime_args", Bool(uses_top_level_runtime_args));
      segment.Set("tt_uses_top_level_common_runtime_args",
                  Bool(uses_top_level_common_runtime_args));
      abi_plans.push_back(TTABIPlan(
          String("abi_" + std::to_string(index)), kernel_name, runtime_args,
          common_runtime_args,
          GetArrayOrEmpty(segment, "compile_time_arg_specs"), GetArrayOrEmpty(segment, "accessors"),
          GetArrayOrEmpty(segment, "semaphore_bindings"), segment));
      kernels.push_back(TTKernel(kernel_name, kernel_kind, core_type, index, segment));
      ++index;
    }
  } else {
    Map<String, Any> payload;
    payload.Set("runtime_args", func->GetAttr<Array<Any>>("blackhole.runtime_args").value_or(Array<Any>()));
    payload.Set("common_runtime_args",
                func->GetAttr<Array<Any>>("blackhole.common_runtime_args").value_or(Array<Any>()));
    payload.Set("compile_time_arg_specs",
                func->GetAttr<Array<Any>>("blackhole.compile_time_arg_specs").value_or(Array<Any>()));
    payload.Set("accessors", func->GetAttr<Array<Any>>("blackhole.accessors").value_or(Array<Any>()));
    String kernel_name = GetStringOrDefault(payload, "name", String("main"));
    abi_plans.push_back(TTABIPlan(String("abi_0"), kernel_name, GetArrayOrEmpty(payload, "runtime_args"),
                                  GetArrayOrEmpty(payload, "common_runtime_args"),
                                  GetArrayOrEmpty(payload, "compile_time_arg_specs"),
                                  GetArrayOrEmpty(payload, "accessors"), Array<Any>(), payload));
    kernels.push_back(TTKernel(kernel_name, String("fused_dataflow"), String("brisc"), 0, payload));
  }
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
  Array<Any> segment_plan = func->GetAttr<Array<Any>>("blackhole.segment_plan").value_or(Array<Any>());
  Array<TTKernel> kernels;
  Array<TTABIPlan> abi_plans = BuildABIPlans(func, segment_plan, &kernels);
  Map<String, Any> payload;
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
