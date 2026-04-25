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
#include "common/buffer_tile_bridge_spec_utils.h"
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

struct LogicalTileLayoutInfo {
  Array<PrimExpr> logical_shape;
  Array<PrimExpr> local_shape;
  PrimExpr thread_extent;
  PrimExpr replicate_extent;
  Array<PrimExpr> inverse_logical_index_vars;
  Array<PrimExpr> inverse_logical_index_exprs;
};

std::unordered_map<std::string, LogicalTileLayoutInfo> CollectLogicalTileLayoutsFromBody(
    const tir::Stmt& body) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    std::unordered_map<std::string, LogicalTileLayoutInfo> Collect(const tir::Stmt& stmt) {
      layout_by_buffer_.clear();
      VisitStmt(stmt);
      return layout_by_buffer_;
    }

   private:
    void Record(const tir::Buffer& buffer, const Layout& layout) {
      const std::string scope = buffer.scope();
      if (scope != "local" && scope != "local.fragment" && scope != "blackhole.acc") {
        return;
      }
      auto maybe_spec = TryBuildBufferTileBridgeSpec(buffer, layout);
      if (!maybe_spec) {
        return;
      }
      const Map<String, Any>& spec = maybe_spec.value();
      auto buffer_it = spec.find(String(schema_key::kBuffer));
      if (buffer_it == spec.end()) {
        return;
      }
      const std::string buffer_name = Downcast<String>((*buffer_it).second);
      if (buffer_name.empty() || layout_by_buffer_.count(buffer_name)) {
        return;
      }
      LogicalTileLayoutInfo info;
      if (auto value = spec.Get(String(schema_key::kShape))) {
        info.logical_shape = Downcast<Array<PrimExpr>>(value.value());
      }
      if (auto value = spec.Get(String(schema_key::kLocalShape))) {
        info.local_shape = Downcast<Array<PrimExpr>>(value.value());
      }
      if (auto value = spec.Get(String(schema_key::kThreadExtent))) {
        info.thread_extent = Downcast<PrimExpr>(value.value());
      }
      if (auto value = spec.Get(String(schema_key::kReplicateExtent))) {
        info.replicate_extent = Downcast<PrimExpr>(value.value());
      }
      if (auto value = spec.Get(String(schema_key::kInverseLogicalIndexVars))) {
        info.inverse_logical_index_vars = Downcast<Array<PrimExpr>>(value.value());
      }
      if (auto value = spec.Get(String(schema_key::kInverseLogicalIndexExprs))) {
        info.inverse_logical_index_exprs = Downcast<Array<PrimExpr>>(value.value());
      }
      layout_by_buffer_.emplace(buffer_name, std::move(info));
    }

    void VisitStmt_(const tir::BlockNode* op) final {
      if (op->annotations.count(attr::kLayoutMap)) {
        if (auto layout_map_any = op->annotations.Get(attr::kLayoutMap)) {
          auto layout_map = layout_map_any->as<Map<tir::Buffer, Layout>>();
          if (layout_map && layout_map.value().defined()) {
            for (const auto& [buffer, layout] : layout_map.value()) {
              Record(buffer, layout);
            }
          }
        }
      }
      tir::StmtExprVisitor::VisitStmt_(op);
    }

    std::unordered_map<std::string, LogicalTileLayoutInfo> layout_by_buffer_;
  };
  return Collector().Collect(body);
}

tir::PrimFunc StripTTIntermediateAttrs(tir::PrimFunc func) {
  static const char* kIntermediateSeedAttrs[] = {attr::kTLTTSemaphorePlans};
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

int64_t GetIntegerOrDefault(const Map<String, Any>& dict, const char* key,
                            int64_t default_value = 0) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

SpatialPlan RequireValidatedSpatialPlan(const tir::PrimFunc& func, const char* pass_name) {
  auto maybe_spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(maybe_spatial_plan)
      << pass_name
      << " requires tl.spatial_plan; run BuildSpatialPlan and ValidateSpatialPlan before target planning";
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
  Array<TTMeshPlan> mesh_plans;
  Array<TTBufferDistributionPlan> buffer_distribution_plans;
  Array<TTBlockPlan> block_plans;
  Array<TTKernelPlan> kernel_plans;
  Array<TTComputeOpPlan> compute_op_plans;
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
  Array<TTLiveFormPlan> live_form_plans;
  Array<TTMaterializationPlan> materialization_plans;
  Array<TTConsumerBindingPlan> consumer_binding_plans;
};

TTProgramSlices UnpackTTProgram(const TTProgram& program) {
  TTProgramSlices slices;
  slices.entry_name = program->entry_name;
  slices.member_func = program->member_func;
  slices.mesh_plans = program->mesh_plans;
  slices.buffer_distribution_plans = program->buffer_distribution_plans;
  slices.block_plans = program->block_plans;
  slices.kernel_plans = program->kernel_plans;
  slices.compute_op_plans = program->compute_op_plans;
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
  slices.live_form_plans = program->live_form_plans;
  slices.materialization_plans = program->materialization_plans;
  slices.consumer_binding_plans = program->consumer_binding_plans;
  return slices;
}

TTProgram PackTTProgram(TTProgramSlices slices) {
  return TTProgram(std::move(slices.entry_name), std::move(slices.member_func),
                   std::move(slices.mesh_plans),
                   std::move(slices.buffer_distribution_plans),
                   std::move(slices.block_plans), std::move(slices.kernel_plans),
                   std::move(slices.compute_op_plans),
                   std::move(slices.transport_plans), std::move(slices.sync_plans),
                   std::move(slices.abi_plans), std::move(slices.execution_plans),
                   std::move(slices.kernels), std::move(slices.core_groups),
                   std::move(slices.cb_plans), std::move(slices.semaphore_plans),
                   std::move(slices.compute_sync_plans),
                   std::move(slices.dst_layout_plans), std::move(slices.live_form_plans),
                   std::move(slices.materialization_plans),
                   std::move(slices.consumer_binding_plans));
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
    cb_attr.Set("total_size_bytes", Integer(config.total_size));
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
                                   config.initial_reserve_pages,
                                   String(CBFlowClassToString(config.flow_class)),
                                   config.publish_pages_per_event,
                                   config.consume_pages_per_event, config.lifetime_begin,
                                   config.lifetime_end,
                                   cb_attr));
  }
  return tt_cb_plans;
}

Array<TTMeshPlan> BuildUnitMeshPlans() {
  Map<String, Any> payload;
  payload.Set("coordinate_rank", Integer(2));
  payload.Set("device_count", Integer(1));
  Array<TTMeshPlan> mesh_plans;
  mesh_plans.push_back(TTMeshPlan(String("unit_mesh"), String("unit_mesh"),
                                  Array<Integer>{Integer(1), Integer(1)},
                                  Array<Integer>{Integer(0), Integer(0)},
                                  Array<Integer>{Integer(1), Integer(1)},
                                  String("default_system_mesh"), payload));
  return mesh_plans;
}

Array<TTCoreGroup> BuildCoreGroups(const PlanTTCoreGroups& planner) {
  const CoreAssignment assignment = planner.GetCoreAssignment();

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

  Array<TTCoreGroup> tt_core_groups;
  tt_core_groups.push_back(TTCoreGroup(String("main_core_group"), assignment.grid_x,
                                       assignment.grid_y, String("row_major"),
                                       physical_cores, work_packets));
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
    Map<String, Any> payload;
    payload.Set("logical_grid_x", Integer(core_group->logical_grid_x));
    payload.Set("logical_grid_y", Integer(core_group->logical_grid_y));
    payload.Set("linearization", core_group->linearization);
    payload.Set("physical_cores", core_group->physical_cores);
    payload.Set("work_packets", core_group->work_packets);
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
    Map<String, Any> payload;
    payload.Set("kernel_name", kernel->name);
    kernel_plans.push_back(TTKernelPlan(kernel->name, kernel->kind, kernel->core_type,
                                        /*block_plan_index=*/0, kernel->abi_plan_index, payload));
  }
  return kernel_plans;
}

Array<TTComputeOpPlan> AttachComputeOpKernelPlanIndices(
    const Array<TTComputeOpPlan>& compute_op_plans,
    const Array<TTKernelPlan>& kernel_plans) {
  std::unordered_map<std::string, int64_t> kernel_index_by_name;
  for (int64_t i = 0; i < static_cast<int64_t>(kernel_plans.size()); ++i) {
    kernel_index_by_name.emplace(static_cast<std::string>(kernel_plans[i]->name), i);
  }

  Array<TTComputeOpPlan> updated;
  for (const TTComputeOpPlan& plan : compute_op_plans) {
    auto kernel_index_it = kernel_index_by_name.find(static_cast<std::string>(plan->kernel_name));
    const int64_t kernel_plan_index =
        kernel_index_it == kernel_index_by_name.end() ? -1 : kernel_index_it->second;
    updated.push_back(TTComputeOpPlan(
        plan->name, plan->kernel_name, kernel_plan_index, plan->kind, plan->enabled,
        plan->operand_bindings, plan->problem_shape_axes, plan->problem_shape, plan->tile_shape,
        plan->block_shape, plan->subblock_shape, plan->accumulator_dtype,
        plan->mbarrier_buffer, plan->mbarrier_scope, plan->mbarrier_index_exprs));
  }
  return updated;
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
    sync_plans.push_back(TTComputeSyncPlan(
        String("sync_" + std::string(edge->name)), edge->kind, edge->producer_unit_index,
        edge->consumer_unit_index, DeriveSyncOrderingKind(edge),
        DeriveSyncMaterializationKind(edge)));
  }
  return sync_plans;
}

Array<TTSyncPlan> BuildSyncPlans(const Array<TTComputeSyncPlan>& compute_sync_plans) {
  Array<TTSyncPlan> sync_plans;
  for (const TTComputeSyncPlan& sync : compute_sync_plans) {
    sync_plans.push_back(TTSyncPlan(sync->name, sync->kind, sync->source_task_index,
                                    sync->target_task_index, sync->ordering_kind,
                                    sync->materialization_kind, Map<String, Any>()));
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

String NormalizeMemorySpace(String memory_space) {
  const std::string value = str(memory_space);
  if (value == "dram" || value == "global") {
    return String("DRAM");
  }
  if (value == "l1" || value == "local" || value == "shared" ||
      value.rfind("blackhole.", 0) == 0) {
    return String("L1");
  }
  if (value.empty()) {
    return String("L1");
  }
  return memory_space;
}

String MemorySpaceFromLayoutScope(const String& scope) {
  const std::string value = str(scope);
  if (value == "global") {
    return String("DRAM");
  }
  return String("L1");
}

LogicalTileLayoutInfo LogicalTileLayoutInfoFromLayoutSpec(const LayoutSpec& layout_spec) {
  LogicalTileLayoutInfo info;
  info.logical_shape = layout_spec->logical_shape;
  info.local_shape = layout_spec->local_shape;
  info.thread_extent = layout_spec->thread_extent;
  info.replicate_extent = layout_spec->replicate_extent;
  info.inverse_logical_index_vars = layout_spec->inverse_logical_index_vars;
  info.inverse_logical_index_exprs = layout_spec->inverse_logical_index_exprs;
  return info;
}

Array<TTBufferDistributionPlan> BuildBufferDistributionPlans(
    const SpatialPlan& spatial_plan, const Array<TTDstLayoutPlan>& dst_layout_plans,
    const tir::PrimFunc& func) {
  struct DstLayoutInfo {
    String layout;
    String memory_space;
    int64_t page_size_bytes = 0;
    Map<String, Any> payload;
  };

  std::unordered_map<std::string, DstLayoutInfo> dst_layout_by_buffer;
  for (const TTDstLayoutPlan& dst_layout : dst_layout_plans) {
    DstLayoutInfo info;
    info.layout = dst_layout->layout;
    info.memory_space = NormalizeMemorySpace(dst_layout->memory_space);
    info.payload = dst_layout->payload;
    info.page_size_bytes = GetIntegerOrDefault(dst_layout->payload, "transport_page_size", 0);
    dst_layout_by_buffer.emplace(str(dst_layout->buffer), std::move(info));
  }

  const std::unordered_map<std::string, LogicalTileLayoutInfo> current_layouts_by_buffer =
      CollectLogicalTileLayoutsFromBody(func->body);
  Array<TTBufferDistributionPlan> distribution_plans;
  std::unordered_set<std::string> seen;
  for (const LayoutSpec& layout_spec : spatial_plan->layout_specs) {
    const std::string buffer = str(layout_spec->subject);
    if (buffer.empty() || !seen.insert(buffer).second) {
      continue;
    }
    String layout = String("local");
    String memory_space = MemorySpaceFromLayoutScope(layout_spec->scope);
    int64_t page_size_bytes = 0;
    Map<String, Any> payload;
    payload.Set("spatial_layout", layout_spec->name);
    payload.Set("spatial_distribution_kind", layout_spec->distribution_kind);
    auto dst_it = dst_layout_by_buffer.find(buffer);
    if (dst_it != dst_layout_by_buffer.end()) {
      layout = dst_it->second.layout;
      memory_space = dst_it->second.memory_space;
      page_size_bytes = dst_it->second.page_size_bytes;
      payload.Set("abi_layout", dst_it->second.layout);
      payload.Set("abi_memory_space", dst_it->second.memory_space);
    }
    const String host_visibility =
        str(memory_space) == "DRAM" ? String("host_visible") : String("device_local");
    LogicalTileLayoutInfo layout_info = LogicalTileLayoutInfoFromLayoutSpec(layout_spec);
    auto current_layout_it = current_layouts_by_buffer.find(buffer);
    if (current_layout_it != current_layouts_by_buffer.end() &&
        current_layout_it->second.logical_shape.size() > 0) {
      layout_info = current_layout_it->second;
    }
    distribution_plans.push_back(TTBufferDistributionPlan(
        String("buffer_distribution_" + buffer), String(buffer), String("unit_mesh"),
        /*mesh_plan_index=*/0, String("replicated"), layout, memory_space, page_size_bytes,
        Array<Integer>{}, String("row_major"), host_visibility, layout_info.logical_shape,
        layout_info.local_shape, layout_info.thread_extent, layout_info.replicate_extent,
        layout_info.inverse_logical_index_vars, layout_info.inverse_logical_index_exprs, payload));
  }
  return distribution_plans;
}

Array<TTMaterializationPlan> RemapMaterializationCBRequirementIndices(
    const Array<TTMaterializationPlan>& materialization_plans,
    const Array<TTCBPlan>& cb_plans) {
  std::unordered_map<int64_t, int64_t> cb_plan_index_by_requirement_index;
  for (int64_t cb_plan_index = 0; cb_plan_index < static_cast<int64_t>(cb_plans.size());
       ++cb_plan_index) {
    const TTCBPlan& cb_plan = cb_plans[static_cast<size_t>(cb_plan_index)];
    if (auto requirement_indices = cb_plan->payload.Get("requirement_indices")) {
      for (const Any& index_any : Downcast<Array<Any>>(requirement_indices.value())) {
        cb_plan_index_by_requirement_index[Downcast<Integer>(index_any)->value] =
            cb_plan_index;
      }
    }
  }

  Array<TTMaterializationPlan> remapped;
  for (const TTMaterializationPlan& plan : materialization_plans) {
    Array<Integer> cb_plan_indices;
    for (const Integer& index : plan->required_cb_plan_indices) {
      const int64_t requirement_index = index->value;
      auto it = cb_plan_index_by_requirement_index.find(requirement_index);
      cb_plan_indices.push_back(Integer(it != cb_plan_index_by_requirement_index.end()
                                            ? it->second
                                            : requirement_index));
    }
    remapped.push_back(TTMaterializationPlan(
        plan->name, plan->source_live_form, plan->materialization_boundary,
        plan->materialization_boundary_index, plan->target_buffer, plan->host_buffer,
        plan->target_kernel, plan->materialization_protocol, plan->publication_protocol, cb_plan_indices,
        plan->required_sync_plan_indices, plan->produced_live_form, plan->payload));
  }
  return remapped;
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
      slices.mesh_plans = BuildUnitMeshPlans();
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
      slices.cb_plans = planner.GetStagedCBPlans();
      slices.abi_plans = planner.GetTTABIPlans();
      slices.live_form_plans = planner.GetTTLiveFormPlans();
      slices.materialization_plans = planner.GetTTMaterializationPlans();
      slices.consumer_binding_plans = planner.GetTTConsumerBindingPlans();
      slices.compute_op_plans =
          AttachComputeOpKernelPlanIndices(planner.GetTTComputeOpPlans(), slices.kernel_plans);
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
      slices.materialization_plans =
          RemapMaterializationCBRequirementIndices(slices.materialization_plans, slices.cb_plans);
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
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTABI");
      const TTProgram staged =
          RequireStagedTTProgram(func.value(), "PlanTTABI", "Run PlanTTCompute before PlanTTABI");
      TTProgramSlices slices = UnpackTTProgram(staged);
      ICHECK(!slices.abi_plans.empty())
          << "PlanTTABI requires TTABIPlan owner truth; Run PlanTTCompute before PlanTTABI";
      slices.dst_layout_plans = BuildDstLayoutPlans(slices.abi_plans);
      slices.buffer_distribution_plans =
          BuildBufferDistributionPlans(spatial_plan, slices.dst_layout_plans, func.value());
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

      ICHECK(!slices.mesh_plans.empty()) << "BuildTTProgram requires TTMeshPlan owner truth";
      ICHECK(!slices.buffer_distribution_plans.empty())
          << "BuildTTProgram requires TTBufferDistributionPlan owner truth";
      ICHECK(!slices.block_plans.empty()) << "BuildTTProgram requires TTBlockPlan owner truth";
      ICHECK(!slices.kernel_plans.empty()) << "BuildTTProgram requires TTKernelPlan owner truth";
      ICHECK(!slices.core_groups.empty())
          << "BuildTTProgram requires TTCoreGroup owner truth; run PlanTTBlocks before BuildTTProgram";
      ICHECK(!slices.abi_plans.empty())
          << "BuildTTProgram requires TTABIPlan owner truth; run PlanTTCompute and PlanTTABI before BuildTTProgram";
      ICHECK(!slices.execution_plans.empty())
          << "BuildTTProgram requires TTExecutionPlan owner truth; run PlanTTExecution before BuildTTProgram";
      ICHECK_EQ(slices.block_plans.size(), slices.core_groups.size())
          << "BuildTTProgram requires aligned TTBlockPlan and TTCoreGroup owner truth";
      ICHECK_EQ(slices.kernel_plans.size(), slices.kernels.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTKernel owner truth";
      ICHECK_EQ(slices.kernel_plans.size(), slices.abi_plans.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTABIPlan owner truth";
      ICHECK_EQ(slices.sync_plans.size(), slices.compute_sync_plans.size())
          << "BuildTTProgram requires aligned TTSyncPlan and TTComputeSyncPlan owner truth";

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
