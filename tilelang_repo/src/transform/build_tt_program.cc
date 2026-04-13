/*!
 * \file build_tt_program.cc
 * \brief Materialize TTProgram from frozen SpatialProgram and planner results.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "assign_blackhole_cores.h"
#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_program.h"
#include "common/tt_hardware_model.h"
#include "common/tt_target_program.h"
#include "lower_blackhole_ops.h"
#include "plan_blackhole_cb.h"

namespace tvm {
namespace tl {

using tvm::GlobalInfo;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

tir::PrimFunc StripTTIntermediateAttrs(tir::PrimFunc func) {
  static const char* kLegacyProjectionAttrs[] = {
      "blackhole.cb_requirements",
  };
  for (const char* key : kLegacyProjectionAttrs) {
    func = tvm::WithoutAttr(std::move(func), key);
  }
  static const char* kIntermediateSeedAttrs[] = {
      attr::kTLTTSemaphorePlans,
  };
  for (const char* key : kIntermediateSeedAttrs) {
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

String GetStringOrDefault(const Map<String, Any>& dict, const char* key, String default_value = String()) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<String>(value.value());
  }
  return default_value;
}

Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

struct MaterializedTTPlanning {
  tir::PrimFunc func;
  Array<TTKernel> kernels;
  Array<TTABIPlan> abi_plans;
  Array<TTCBPlan> cb_plans;
  Array<TTCoreGroup> core_groups;
  Map<String, Any> payload;
};

const char* CBFlowClassToString(CBFlowClass flow_class) {
  switch (flow_class) {
    case CBFlowClass::kStream:
      return "stream";
    case CBFlowClass::kRepublish:
      return "republish";
    case CBFlowClass::kState:
    default:
      return "state";
  }
}

Array<TTCBPlan> BuildCBPlans(const std::vector<CBConfig>& configs) {
  Array<TTCBPlan> tt_cb_plans;
  for (const auto& config : configs) {
    Map<String, Any> cb_attr;
    cb_attr.Set("cb_id", Integer(config.cb_id));
    cb_attr.Set("page_size", Integer(config.page_size));
    cb_attr.Set("num_pages", Integer(config.num_pages));
    if (config.initial_reserve_pages > 0) {
      cb_attr.Set("initial_reserve_pages", Integer(config.initial_reserve_pages));
    }
    cb_attr.Set("flow_class", String(CBFlowClassToString(config.flow_class)));
    if (config.publish_pages_per_event > 0) {
      cb_attr.Set("publish_pages_per_event", Integer(config.publish_pages_per_event));
    }
    if (config.consume_pages_per_event > 0) {
      cb_attr.Set("consume_pages_per_event", Integer(config.consume_pages_per_event));
    }
    cb_attr.Set("total_size_bytes", Integer(config.total_size));
    cb_attr.Set("data_format", String(config.data_format));
    cb_attr.Set("name", String(config.name));
    cb_attr.Set("role", String(config.role));
    cb_attr.Set("lifetime_begin", Integer(config.lifetime_begin));
    cb_attr.Set("lifetime_end", Integer(config.lifetime_end));
    Array<Any> requirement_names;
    Array<Any> requirement_indices;
    for (const auto& req_name : config.requirement_names) {
      requirement_names.push_back(String(req_name));
    }
    for (int req_index : config.requirement_indices) {
      requirement_indices.push_back(Integer(req_index));
    }
    cb_attr.Set("requirement_names", requirement_names);
    cb_attr.Set("requirement_indices", requirement_indices);
    tt_cb_plans.push_back(TTCBPlan(String(config.name), config.cb_id, String(config.role),
                                   config.num_pages, config.page_size, String(config.data_format),
                                   cb_attr));
  }
  return tt_cb_plans;
}

Array<TTCoreGroup> BuildCoreGroups(const PlanTTCoreGroups& planner) {
  const CoreAssignment assignment = planner.GetCoreAssignment();
  Map<String, Any> core_plan;
  core_plan.Set("logical_grid_x", Integer(assignment.grid_x));
  core_plan.Set("logical_grid_y", Integer(assignment.grid_y));
  core_plan.Set("linearization", String("row_major"));

  Array<Any> physical_cores;
  Array<Any> work_packets;
  for (int core_idx = 0; core_idx < assignment.cores_needed; ++core_idx) {
    const CoreCoord coord = planner.GetCoreCoord(core_idx);
    Map<String, Any> core_info;
    core_info.Set("core_x", Integer(coord.x));
    core_info.Set("core_y", Integer(coord.y));
    physical_cores.push_back(core_info);

    const RuntimeArgs runtime_args = planner.GetRuntimeArgs(core_idx);
    Map<String, Any> packet_info;
    packet_info.Set("core_x", Integer(coord.x));
    packet_info.Set("core_y", Integer(coord.y));
    packet_info.Set("work_offset", Integer(runtime_args.work_offset_linear));
    packet_info.Set("work_count", Integer(runtime_args.work_count));
    work_packets.push_back(packet_info);
  }

  core_plan.Set("physical_cores", physical_cores);
  core_plan.Set("work_packets", work_packets);

  Array<TTCoreGroup> tt_core_groups;
  tt_core_groups.push_back(TTCoreGroup(String("main_core_group"), assignment.grid_x,
                                       assignment.grid_y, String("row_major"),
                                       physical_cores, work_packets, core_plan));
  return tt_core_groups;
}

MaterializedTTPlanning MaterializeTTPlanning(tir::PrimFunc func) {
  PlanTTKernelABI kernel_abi_planner;
  func = kernel_abi_planner.Transform(func);

  PlanTTCBAlloc cb_planner;
  func = cb_planner.Transform(func);

  PlanTTCoreGroups core_group_planner;
  func = core_group_planner.Transform(func);

  MaterializedTTPlanning planning;
  planning.func = std::move(func);
  planning.kernels = kernel_abi_planner.GetTTKernels();
  planning.abi_plans = kernel_abi_planner.GetTTABIPlans();
  planning.cb_plans = BuildCBPlans(cb_planner.GetCBConfigs());
  planning.core_groups = BuildCoreGroups(core_group_planner);
  planning.payload = kernel_abi_planner.GetTTProgramPayload();
  return planning;
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

TTProgram BuildTTProgramForFunc(const MaterializedTTPlanning& planning, const String& entry_name,
                                const SpatialProgram& program,
                                const TTHardwareModel& hardware_model) {
  const Array<TTKernel>& kernels = planning.kernels;
  const Array<TTABIPlan>& abi_plans = planning.abi_plans;
  ICHECK(!kernels.empty()) << "BuildTTProgram requires TT kernels from PlanTTKernelABI";
  ICHECK_EQ(kernels.size(), abi_plans.size())
      << "BuildTTProgram requires aligned TT kernel and ABI planning";
  Map<String, Any> payload = planning.payload;
  payload.Set("arch_name", hardware_model->arch_name);
  payload.Set("logical_worker_grid_x", Integer(hardware_model->logical_worker_grid_x));
  payload.Set("logical_worker_grid_y", Integer(hardware_model->logical_worker_grid_y));
  payload.Set("worker_l1_size", Integer(hardware_model->worker_l1_size));
  return TTProgram(entry_name, program->member_func, kernels, planning.core_groups,
                   planning.cb_plans, BuildTransportPlans(program),
                   BuildSemaphorePlans(planning.func, program),
                   BuildComputeSyncPlans(program), BuildDstLayoutPlans(program, abi_plans), abi_plans,
                   BuildExecutionPlans(program, kernels), payload);
}

}  // namespace

tvm::transform::Pass BuildTTProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = GetModuleTTHardwareModel(mod);
    if (!maybe_hardware_model) {
      auto maybe_target = FindBlackholeTarget(mod);
      ICHECK(maybe_target) << "BuildTTProgram requires blackhole target";
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
      MaterializedTTPlanning planning = MaterializeTTPlanning(func.value());
      auto maybe_spatial_program = planning.func->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (!maybe_spatial_program) {
        continue;
      }
      Map<String, Any> attrs;
      if (planning.func->attrs.defined()) {
        attrs = planning.func->attrs->dict;
      }
      attrs.Set(attr::kTLTTProgram, BuildTTProgramForFunc(planning, gvar->name_hint,
                                                          maybe_spatial_program.value(),
                                                          maybe_hardware_model.value()));
      planning.func.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
      planning.func = StripTTIntermediateAttrs(std::move(planning.func));
      updated->Add(gvar, planning.func, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.BuildTTProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BuildTTProgram", BuildTTProgram);
}

}  // namespace tl
}  // namespace tvm
