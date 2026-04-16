/*!
 * \file build_tt_program.cc
 * \brief Materialize TTProgram from SpatialPlan and planner results.
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
#include "common/spatial_plan.h"
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
      attr::kTLBlackholeLogicalBufferTileBridgeSpecs,
      kTLBlackholeLoweringRequirementsSeed,
  };
  for (const char* key : kLegacyProjectionAttrs) {
    func = tvm::WithoutAttr(std::move(func), key);
  }
  static const char* kIntermediateSeedAttrs[] = {attr::kTLTTSemaphorePlans,
                                                 attr::kTLInternalTTBlockPlans,
                                                 attr::kTLInternalTTKernelPlans,
                                                 attr::kTLInternalTTABIPlanSeeds,
                                                 attr::kTLInternalTTSyncPlans,
                                                 attr::kTLInternalTTSemaphorePlans,
                                                 attr::kTLInternalTTComputeSyncPlans,
                                                 attr::kTLInternalTTDstLayoutPlans,
                                                 attr::kTLInternalTTExecutionPlans,
                                                 attr::kTLInternalTTKernels,
                                                 attr::kTLInternalTTABIPlans,
                                                 attr::kTLInternalTTCBPlans,
                                                 attr::kTLInternalTTCoreGroups,
                                                 attr::kTLInternalTTTransportPlans,
                                                 attr::kTLInternalTTProgramPayload};
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

SpatialPlan RequireValidatedSpatialPlan(const tir::PrimFunc& func, const char* pass_name) {
  auto maybe_spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(maybe_spatial_plan)
      << pass_name << " requires tl.spatial_plan; run AnalyzeSpatialStructureFacts, "
      << "BuildSpatialPlanCompanion, and ValidateSpatialPlan before target planning";
  ICHECK(func->GetAttr<Bool>(attr::kTLSpatialPlanValidated, Bool(false)).value())
      << pass_name << " requires validated SpatialPlan; run ValidateSpatialPlan before target planning";
  return maybe_spatial_plan.value();
}

void RequireTTMetalBuiltinSelection(const tir::PrimFunc& func, const char* pass_name) {
  ICHECK(func->GetAttr<Bool>(kTLBlackholeTTMetalBuiltinSelection, Bool(false)).value())
      << pass_name
      << " requires exact TT-Metal builtin selection; run "
         "SelectBlackholeTTMetalBuiltins after SplitBlackholeKernel";
}

Map<String, Any> CopyAttrs(const tir::PrimFunc& func) {
  return func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
}

struct TTProgramSlices {
  String entry_name;
  String member_func;
  Array<TTBlockPlan> block_plans;
  Array<TTKernelPlan> kernel_plans;
  Array<TTTransportPlan> transport_plans;
  Array<TTSyncPlan> sync_plans;
  Array<TTABIPlan> abi_plans;
  Array<TTExecutionPlan> execution_plans;
  Array<TTKernel> kernels;
  Array<TTCoreGroup> core_groups;
  Array<TTCBPlan> cb_plans;
  Array<TTSemaphorePlan> semaphore_plans;
  Array<TTComputeSyncPlan> compute_sync_plans;
  Array<TTDstLayoutPlan> dst_layout_plans;
  Map<String, Any> payload;
};

TTProgramSlices UnpackTTProgram(const TTProgram& program) {
  TTProgramSlices slices;
  slices.entry_name = program->entry_name;
  slices.member_func = program->member_func;
  slices.block_plans = program->block_plans;
  slices.kernel_plans = program->kernel_plans;
  slices.transport_plans = program->transport_plans;
  slices.sync_plans = program->sync_plans;
  slices.abi_plans = program->abi_plans;
  slices.execution_plans = program->execution_plans;
  slices.kernels = program->kernels;
  slices.core_groups = program->core_groups;
  slices.cb_plans = program->cb_plans;
  slices.semaphore_plans = program->semaphore_plans;
  slices.compute_sync_plans = program->compute_sync_plans;
  slices.dst_layout_plans = program->dst_layout_plans;
  slices.payload = program->payload;
  return slices;
}

TTProgram PackTTProgram(TTProgramSlices slices) {
  return TTProgram(std::move(slices.entry_name), std::move(slices.member_func),
                   std::move(slices.block_plans), std::move(slices.kernel_plans),
                   std::move(slices.transport_plans), std::move(slices.sync_plans),
                   std::move(slices.abi_plans), std::move(slices.execution_plans),
                   std::move(slices.kernels), std::move(slices.core_groups),
                   std::move(slices.cb_plans), std::move(slices.semaphore_plans),
                   std::move(slices.compute_sync_plans),
                   std::move(slices.dst_layout_plans), std::move(slices.payload));
}

TTProgramSlices GetOrCreateTTProgramSlices(const tir::PrimFunc& func, const GlobalVar& gvar,
                                           const SpatialPlan& spatial_plan) {
  if (auto maybe_program = func->GetAttr<TTProgram>(attr::kTLTTProgram)) {
    return UnpackTTProgram(maybe_program.value());
  }
  TTProgramSlices slices;
  slices.entry_name = gvar->name_hint;
  slices.member_func = spatial_plan->member_func;
  return slices;
}

TTProgram RequireStagedTTProgram(const tir::PrimFunc& func, const char* consumer,
                                 const char* next_step_guidance) {
  auto maybe_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  ICHECK(maybe_program)
      << consumer << " requires staged tl.tt_program owner truth. " << next_step_guidance;
  return maybe_program.value();
}

tir::PrimFunc WithTTProgramAttr(tir::PrimFunc func, TTProgram program) {
  Map<String, Any> attrs = CopyAttrs(func);
  attrs.Set(attr::kTLTTProgram, std::move(program));
  func.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
  return func;
}

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

Array<TTBlockPlan> BuildBlockPlans(const SpatialPlan& plan, const Array<TTCoreGroup>& core_groups) {
  Array<Integer> task_indices;
  for (size_t i = 0; i < plan->execution_units.size(); ++i) {
    task_indices.push_back(Integer(static_cast<int64_t>(i)));
  }

  Array<TTBlockPlan> block_plans;
  for (size_t i = 0; i < core_groups.size(); ++i) {
    const TTCoreGroup& core_group = core_groups[i];
    Map<String, Any> payload = core_group->payload;
    payload.Set("core_group_name", core_group->name);
    payload.Set("physical_core_count", Integer(static_cast<int64_t>(core_group->physical_cores.size())));
    payload.Set("work_packet_count", Integer(static_cast<int64_t>(core_group->work_packets.size())));
    block_plans.push_back(
        TTBlockPlan(String("block_plan_" + std::to_string(i)), String("core_group"),
                    task_indices, payload));
  }
  return block_plans;
}

Array<TTKernelPlan> BuildKernelPlans(const Array<TTKernel>& kernels) {
  Array<TTKernelPlan> kernel_plans;
  for (const TTKernel& kernel : kernels) {
    Map<String, Any> payload = kernel->payload;
    payload.Set("compat_kernel_name", kernel->name);
    kernel_plans.push_back(TTKernelPlan(kernel->name, kernel->kind, kernel->core_type,
                                        /*block_plan_index=*/0, kernel->abi_plan_index, payload));
  }
  return kernel_plans;
}

String DeriveTransportDeliveryKind(const DataflowEdge& edge) {
  if (edge->crosses_phase) {
    return String("phase_boundary_materialized");
  }
  if (edge->kind == "join") {
    return String("completion_visible");
  }
  return String("ordered");
}

String DeriveSyncOrderingKind(const DataflowEdge& edge) {
  if (edge->crosses_phase) {
    return String("phase_boundary_materialization");
  }
  if (edge->kind == "carry") {
    return String("carry_handoff");
  }
  if (edge->kind == "join") {
    return String("reduction_completion");
  }
  return String("must_happen_before");
}

String DeriveSyncMaterializationKind(const DataflowEdge& edge) {
  if (edge->crosses_phase) {
    return String("phase_boundary_materialization");
  }
  if (edge->kind == "join") {
    return String("completion_visibility");
  }
  return String("phase_boundary");
}

Array<TTTransportPlan> BuildTransportPlans(const SpatialPlan& plan) {
  Array<TTTransportPlan> transport_plans;
  for (const DataflowEdge& edge : plan->dataflow_edges) {
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0) {
      continue;
    }
    Map<String, Any> payload;
    payload.Set("subject", edge->subject);
    payload.Set("boundary_kind", edge->kind);
    transport_plans.push_back(
        TTTransportPlan(edge->name, edge->kind, edge->producer_unit_index,
                        edge->consumer_unit_index, String("tensor"),
                        DeriveTransportDeliveryKind(edge), payload));
  }
  return transport_plans;
}

Array<TTSemaphorePlan> BuildSemaphorePlans(const tir::PrimFunc& func) {
  if (auto maybe_semaphore_plans = func->GetAttr<Array<TTSemaphorePlan>>(attr::kTLTTSemaphorePlans)) {
    return maybe_semaphore_plans.value();
  }
  return Array<TTSemaphorePlan>();
}

Array<TTComputeSyncPlan> BuildComputeSyncPlans(const SpatialPlan& plan) {
  Array<TTComputeSyncPlan> sync_plans;
  for (const DataflowEdge& edge : plan->dataflow_edges) {
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
        edge->producer_unit_index == edge->consumer_unit_index) {
      continue;
    }
    Map<String, Any> payload;
    payload.Set("subject", edge->subject);
    payload.Set("boundary_kind", edge->kind);
    sync_plans.push_back(TTComputeSyncPlan(
        String("sync_" + std::string(edge->name)), edge->kind, edge->producer_unit_index,
        edge->consumer_unit_index, DeriveSyncOrderingKind(edge),
        DeriveSyncMaterializationKind(edge), payload));
  }
  return sync_plans;
}

Array<TTSyncPlan> BuildSyncPlans(const Array<TTComputeSyncPlan>& compute_sync_plans) {
  Array<TTSyncPlan> sync_plans;
  for (const TTComputeSyncPlan& sync : compute_sync_plans) {
    Map<String, Any> payload = sync->payload;
    payload.Set("compat_sync_name", sync->name);
    sync_plans.push_back(TTSyncPlan(sync->name, sync->kind, sync->source_task_index,
                                    sync->target_task_index, sync->ordering_kind,
                                    sync->materialization_kind, payload));
  }
  return sync_plans;
}

Array<TTDstLayoutPlan> BuildDstLayoutPlans(const Array<TTABIPlan>& abi_plans) {
  Array<TTDstLayoutPlan> dst_layouts;
  std::unordered_set<std::string> seen;
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

Array<TTExecutionPlan> BuildExecutionPlans(const SpatialPlan& plan, const Array<TTKernel>& kernels) {
  Array<ffi::String> kernel_names;
  for (const TTKernel& kernel : kernels) {
    kernel_names.push_back(kernel->name);
  }
  std::unordered_set<int> seen;
  Array<Integer> phase_indices;
  for (const PhasePlan& phase : plan->phase_plans) {
    const int phase_index = static_cast<int>(phase->phase_index);
    if (seen.insert(phase_index).second) {
      phase_indices.push_back(Integer(phase_index));
    }
  }
  if (phase_indices.empty()) {
    phase_indices.push_back(Integer(0));
  }
  Map<String, Any> payload;
  payload.Set("phase_count", Integer(static_cast<int>(phase_indices.size())));
  payload.Set("execution_unit_count", Integer(static_cast<int>(plan->execution_units.size())));
  payload.Set("dataflow_edge_count", Integer(static_cast<int>(plan->dataflow_edges.size())));
  payload.Set("closure_count", Integer(static_cast<int>(plan->closures.size())));
  payload.Set("boundary_count", Integer(static_cast<int>(plan->boundaries.size())));
  Array<TTExecutionPlan> execution_plans;
  execution_plans.push_back(TTExecutionPlan(String("main_execution"), kernel_names, phase_indices, payload));
  return execution_plans;
}

}  // namespace

tvm::transform::Pass PlanTTBlocks() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTBlocks");
      PlanTTCoreGroups planner;
      tir::PrimFunc planned = planner.Transform(func.value());
      const Array<TTCoreGroup> core_groups = BuildCoreGroups(planner);
      TTProgramSlices slices = GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.block_plans = BuildBlockPlans(spatial_plan, core_groups);
      slices.core_groups = core_groups;
      planned = WithTTProgramAttr(std::move(planned), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTBlocks", {});
}

tvm::transform::Pass PlanTTCompute() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTCompute");
      RequireTTMetalBuiltinSelection(func.value(), "PlanTTCompute");
      PlanTTKernelABI planner;
      tir::PrimFunc planned = planner.Transform(func.value());
      const Array<TTKernel> kernels = planner.GetTTKernels();
      TTProgramSlices slices = GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.kernel_plans = BuildKernelPlans(kernels);
      slices.kernels = kernels;
      slices.abi_plans = planner.GetTTABIPlans();
      slices.payload = planner.GetTTProgramPayload();
      planned = WithTTProgramAttr(std::move(planned), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTCompute", {});
}

tvm::transform::Pass PlanTTTransport() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTTransport");
      PlanTTCBAlloc planner;
      tir::PrimFunc planned = planner.Transform(func.value());
      TTProgramSlices slices = GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.cb_plans = BuildCBPlans(planner.GetCBConfigs());
      slices.transport_plans = BuildTransportPlans(spatial_plan);
      planned = WithTTProgramAttr(std::move(planned), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTTransport", {});
}

tvm::transform::Pass PlanTTSync() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTSync");
      const Array<TTComputeSyncPlan> compute_sync_plans = BuildComputeSyncPlans(spatial_plan);
      tir::PrimFunc planned = func.value();
      TTProgramSlices slices = GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.sync_plans = BuildSyncPlans(compute_sync_plans);
      slices.compute_sync_plans = compute_sync_plans;
      slices.semaphore_plans = BuildSemaphorePlans(func.value());
      planned = WithTTProgramAttr(std::move(planned), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTSync", {});
}

tvm::transform::Pass PlanTTABI() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      RequireValidatedSpatialPlan(func.value(), "PlanTTABI");
      const TTProgram staged =
          RequireStagedTTProgram(func.value(), "PlanTTABI", "Run PlanTTCompute before PlanTTABI");
      TTProgramSlices slices = UnpackTTProgram(staged);
      ICHECK(!slices.abi_plans.empty())
          << "PlanTTABI requires TTABIPlan owner truth; Run PlanTTCompute before PlanTTABI";
      slices.dst_layout_plans = BuildDstLayoutPlans(slices.abi_plans);
      tir::PrimFunc planned = WithTTProgramAttr(func.value(), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTABI", {});
}

tvm::transform::Pass PlanTTExecution() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = GetModuleTTHardwareModel(mod);
    if (!maybe_hardware_model) {
      auto maybe_target = FindBlackholeTarget(mod);
      ICHECK(maybe_target) << "PlanTTExecution requires blackhole target";
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
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTExecution");
      const TTProgram staged = RequireStagedTTProgram(
          func.value(), "PlanTTExecution", "Run PlanTTCompute before PlanTTExecution");
      TTProgramSlices slices = UnpackTTProgram(staged);
      const Array<TTKernel>& kernels = slices.kernels;
      ICHECK(!kernels.empty())
          << "PlanTTExecution requires TTKernel owner truth; Run PlanTTCompute before PlanTTExecution";
      Map<String, Any> payload = slices.payload;
      payload.Set("arch_name", maybe_hardware_model.value()->arch_name);
      payload.Set("logical_worker_grid_x", Integer(maybe_hardware_model.value()->logical_worker_grid_x));
      payload.Set("logical_worker_grid_y", Integer(maybe_hardware_model.value()->logical_worker_grid_y));
      payload.Set("worker_l1_size", Integer(maybe_hardware_model.value()->worker_l1_size));

      slices.payload = payload;
      slices.execution_plans = BuildExecutionPlans(spatial_plan, kernels);
      tir::PrimFunc planned = WithTTProgramAttr(func.value(), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTExecution", {});
}

tvm::transform::Pass BuildTTProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const TTProgram staged = RequireStagedTTProgram(
          func.value(), "BuildTTProgram",
          "Run PlanTTBlocks, PlanTTCompute, PlanTTTransport, PlanTTSync, PlanTTABI, and PlanTTExecution before BuildTTProgram");
      const TTProgramSlices slices = UnpackTTProgram(staged);

      ICHECK(!slices.block_plans.empty()) << "BuildTTProgram requires TTBlockPlan owner truth";
      ICHECK(!slices.kernel_plans.empty()) << "BuildTTProgram requires TTKernelPlan owner truth";
      ICHECK(!slices.core_groups.empty())
          << "BuildTTProgram requires TTCoreGroup compatibility payloads; run PlanTTBlocks before BuildTTProgram";
      ICHECK(!slices.abi_plans.empty())
          << "BuildTTProgram requires TTABIPlan owner truth; run PlanTTCompute and PlanTTABI before BuildTTProgram";
      ICHECK(!slices.execution_plans.empty())
          << "BuildTTProgram requires TTExecutionPlan owner truth; run PlanTTExecution before BuildTTProgram";
      ICHECK_EQ(slices.block_plans.size(), slices.core_groups.size())
          << "BuildTTProgram requires aligned TTBlockPlan and TTCoreGroup compatibility payloads";
      ICHECK_EQ(slices.kernel_plans.size(), slices.kernels.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTKernel compatibility payloads";
      ICHECK_EQ(slices.kernel_plans.size(), slices.abi_plans.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTABIPlan owner truth";
      ICHECK_EQ(slices.sync_plans.size(), slices.compute_sync_plans.size())
          << "BuildTTProgram requires aligned TTSyncPlan and TTComputeSyncPlan compatibility payloads";

      tir::PrimFunc planned = WithTTProgramAttr(func.value(), staged);
      planned = StripTTIntermediateAttrs(std::move(planned));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.BuildTTProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanTTBlocks", PlanTTBlocks);
  refl::GlobalDef().def("tl.transform.PlanTTCompute", PlanTTCompute);
  refl::GlobalDef().def("tl.transform.PlanTTTransport", PlanTTTransport);
  refl::GlobalDef().def("tl.transform.PlanTTSync", PlanTTSync);
  refl::GlobalDef().def("tl.transform.PlanTTABI", PlanTTABI);
  refl::GlobalDef().def("tl.transform.PlanTTExecution", PlanTTExecution);
  refl::GlobalDef().def("tl.transform.BuildTTProgram", BuildTTProgram);
}

}  // namespace tl
}  // namespace tvm
