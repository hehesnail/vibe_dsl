/*!
 * \file build_tt_program.cc
 * \brief Materialize TTProgram from SpatialPlan and planner results.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "assign_blackhole_cores.h"
#include "common/blackhole_tile_compute_covering.h"
#include "common/blackhole_tile_compute_dag.h"
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

std::optional<TTHardwareModel> EnsureBlackholeHardwareModel(IRModule* mod) {
  if (auto maybe_hardware_model = GetModuleTTHardwareModel(*mod)) {
    return maybe_hardware_model.value();
  }
  auto maybe_target = FindBlackholeTarget(*mod);
  if (!maybe_target) {
    return std::nullopt;
  }
  TTHardwareModel hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
  *mod = (*mod)->ShallowCopy();
  (*mod)->UpdateGlobalInfo(attr::kTLTTHardwareModel,
                           Array<GlobalInfo>{hardware_model});
  return hardware_model;
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
  Array<TTResourceDemand> resource_demands;
  Array<TTResourcePressureReport> resource_pressure_reports;
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
  slices.resource_demands = program->resource_demands;
  slices.resource_pressure_reports = program->resource_pressure_reports;
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
                   std::move(slices.consumer_binding_plans),
                   std::move(slices.resource_demands),
                   std::move(slices.resource_pressure_reports));
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
    Array<String> requirement_names;
    Array<Integer> requirement_indices;
    for (const auto& req_name : config.requirement_names) {
      requirement_names.push_back(String(req_name));
    }
    for (int req_index : config.requirement_indices) {
      requirement_indices.push_back(Integer(req_index));
    }
    tt_cb_plans.push_back(TTCBPlan(String(config.name), config.cb_id, String(config.role),
                                   config.num_pages, config.page_size, String(config.data_format),
                                   config.initial_reserve_pages,
                                   String(CBFlowClassToString(config.flow_class)),
                                   config.publish_pages_per_event,
                                   config.consume_pages_per_event, config.lifetime_begin,
                                   config.lifetime_end, requirement_names,
                                   requirement_indices));
  }
  return tt_cb_plans;
}

Array<TTMeshPlan> BuildUnitMeshPlans() {
  Array<TTMeshPlan> mesh_plans;
  mesh_plans.push_back(TTMeshPlan(String("unit_mesh"), String("unit_mesh"),
                                  Array<Integer>{Integer(1), Integer(1)},
                                  Array<Integer>{Integer(0), Integer(0)},
                                  Array<Integer>{Integer(1), Integer(1)},
                                  String("default_system_mesh")));
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
    block_plans.push_back(
        TTBlockPlan(String("block_plan_" + std::to_string(i)), String("core_group"),
                    task_indices, core_group->name, static_cast<int64_t>(i)));
  }
  return block_plans;
}

Array<TTKernelPlan> BuildKernelPlans(const Array<TTKernel>& kernels) {
  Array<TTKernelPlan> kernel_plans;
  for (const TTKernel& kernel : kernels) {
    kernel_plans.push_back(TTKernelPlan(kernel->name, kernel->kind, kernel->core_type,
                                        /*block_plan_index=*/0, kernel->abi_plan_index));
  }
  return kernel_plans;
}

bool IsTileComputeDAGOutputRole(const std::string& role) {
  return role == "output" || role == "c";
}

String PrimaryComputeKernelName(const TTProgramSlices& slices) {
  for (const TTKernelPlan& kernel_plan : slices.kernel_plans) {
    if (kernel_plan->kind == "compute") {
      return kernel_plan->name;
    }
  }
  for (const TTKernel& kernel : slices.kernels) {
    if (kernel->kind == "compute") {
      return kernel->name;
    }
  }
  return String("compute");
}

String PrimaryCoreGroupName(const TTProgramSlices& slices) {
  if (!slices.core_groups.empty()) {
    return slices.core_groups[0]->name;
  }
  return String("main_core_group");
}

int64_t PrimaryCoreGroupIndex(const TTProgramSlices& slices) {
  return slices.core_groups.empty() ? -1 : 0;
}

int64_t TotalCBL1Bytes(const Array<TTCBPlan>& cb_plans) {
  int64_t total = 0;
  for (const TTCBPlan& cb : cb_plans) {
    total += cb->num_pages * cb->page_size_bytes;
  }
  return total;
}

Array<TTTileComputeMaterializationDemand> BuildTileComputeMaterializationDemands(
    const BlackholeTileComputeDAG& dag, const String& kernel_name,
    Array<String>* unsupported_reasons) {
  Array<TTTileComputeMaterializationDemand> demands;
  for (const BlackholeTileComputeDAGNode& node : dag.nodes) {
    const BlackholeTileComputeCoveringDecision decision =
        SelectBlackholeTileComputeCovering(node.op_name);
    if (!decision.selected) {
      unsupported_reasons->push_back(String(
          "node " + std::to_string(node.id) + " operation " + node.op_name +
          ": " + decision.reject_reason));
      continue;
    }
    if (decision.materialization_policy == "none") {
      continue;
    }
    const std::string name =
        "tile_compute_materialization_" + std::to_string(node.id);
    demands.push_back(TTTileComputeMaterializationDemand(
        String(name), kernel_name, node.id, String(decision.operation_name),
        String(decision.pattern_name), String(decision.materialization_policy),
        String("selected_pattern:" + decision.pattern_name +
               ";side_effect:" + node.side_effect_class)));
  }
  return demands;
}

Array<TTTileComputeFanoutDemand> BuildTileComputeFanoutDemands(
    const BlackholeTileComputeDAG& dag, const String& kernel_name) {
  std::unordered_map<int64_t, std::vector<const BlackholeTileComputeDAGEdge*>>
      uses_by_producer;
  for (const BlackholeTileComputeDAGEdge& edge : dag.edges) {
    if (edge.producer_node < 0 || IsTileComputeDAGOutputRole(edge.value_role)) {
      continue;
    }
    uses_by_producer[edge.producer_node].push_back(&edge);
  }

  Array<TTTileComputeFanoutDemand> demands;
  std::vector<int64_t> producer_nodes;
  for (const auto& entry : uses_by_producer) {
    producer_nodes.push_back(entry.first);
  }
  std::sort(producer_nodes.begin(), producer_nodes.end());

  for (const int64_t producer_node : producer_nodes) {
    const std::vector<const BlackholeTileComputeDAGEdge*>& uses =
        uses_by_producer.at(producer_node);
    if (uses.size() < 2U || producer_node < 0 ||
        producer_node >= static_cast<int64_t>(dag.nodes.size())) {
      continue;
    }
    const BlackholeTileComputeDAGNode& producer = dag.nodes[producer_node];
    const bool requires_materialization =
        producer.side_effect_class == "tile_regs" ||
        producer.side_effect_class == "dst" ||
        producer.side_effect_class == "pack";
    Array<Integer> consumer_nodes;
    for (const BlackholeTileComputeDAGEdge* use : uses) {
      consumer_nodes.push_back(Integer(use->consumer_node));
    }
    const std::string name =
        "tile_compute_fanout_" + std::to_string(producer_node);
    demands.push_back(TTTileComputeFanoutDemand(
        String(name), kernel_name, producer_node,
        String(producer.op_name), String(uses.front()->value_repr),
        static_cast<int64_t>(uses.size()), consumer_nodes,
        String(requires_materialization
                   ? "materialize_before_cross_event_use"
                   : "share_live_value"),
        String("producer_use_count:" + std::to_string(uses.size()) +
               ";producer_side_effect:" + producer.side_effect_class)));
  }
  return demands;
}

Array<TTTileComputeFanoutDemand> RefreshFanoutDemandKernelNames(
    const Array<TTTileComputeFanoutDemand>& demands, const String& kernel_name) {
  Array<TTTileComputeFanoutDemand> refreshed;
  for (const TTTileComputeFanoutDemand& demand : demands) {
    refreshed.push_back(TTTileComputeFanoutDemand(
        demand->name, kernel_name, demand->producer_node,
        demand->producer_operation, demand->value_repr, demand->use_count,
        demand->consumer_nodes, demand->policy, demand->evidence));
  }
  return refreshed;
}

Array<TTTileComputeMaterializationDemand> RefreshMaterializationDemandKernelNames(
    const Array<TTTileComputeMaterializationDemand>& demands,
    const String& kernel_name) {
  Array<TTTileComputeMaterializationDemand> refreshed;
  for (const TTTileComputeMaterializationDemand& demand : demands) {
    refreshed.push_back(TTTileComputeMaterializationDemand(
        demand->name, kernel_name, demand->node_id, demand->operation_name,
        demand->pattern_name, demand->policy, demand->evidence));
  }
  return refreshed;
}

Array<TTResourceDemand> BuildTileComputeResourceDemands(
    const tir::PrimFunc& func, const TTProgramSlices& slices) {
  const BlackholeTileComputeDAG dag = BuildBlackholeTileComputeDAG(func);
  if (dag.nodes.empty()) {
    return {};
  }

  const String kernel_name = PrimaryComputeKernelName(slices);
  Array<String> unsupported_reasons;
  Array<TTTileComputeFanoutDemand> fanout_demands =
      BuildTileComputeFanoutDemands(dag, kernel_name);
  Array<TTTileComputeMaterializationDemand> materialization_demands =
      BuildTileComputeMaterializationDemands(dag, kernel_name,
                                             &unsupported_reasons);
  if (fanout_demands.empty() && materialization_demands.empty() &&
      unsupported_reasons.empty()) {
    return {};
  }

  Array<TTResourceDemand> demands;
  demands.push_back(TTResourceDemand(
      String("resource_demand_" + static_cast<std::string>(kernel_name)),
      kernel_name, PrimaryCoreGroupName(slices), PrimaryCoreGroupIndex(slices),
      fanout_demands, materialization_demands, unsupported_reasons,
      static_cast<int64_t>(slices.cb_plans.size()), TotalCBL1Bytes(slices.cb_plans),
      static_cast<int64_t>(slices.semaphore_plans.size()),
      static_cast<int64_t>(slices.transport_plans.size())));
  return demands;
}

TTResourceDemand RefreshResourceDemandCounters(const TTResourceDemand& demand,
                                               const TTProgramSlices& slices) {
  const String kernel_name = PrimaryComputeKernelName(slices);
  return TTResourceDemand(
      String("resource_demand_" + static_cast<std::string>(kernel_name)),
      kernel_name, PrimaryCoreGroupName(slices), PrimaryCoreGroupIndex(slices),
      RefreshFanoutDemandKernelNames(demand->tile_compute_fanout_demands,
                                     kernel_name),
      RefreshMaterializationDemandKernelNames(
          demand->tile_compute_materialization_demands, kernel_name),
      demand->tile_compute_unsupported_reasons,
      static_cast<int64_t>(slices.cb_plans.size()), TotalCBL1Bytes(slices.cb_plans),
      static_cast<int64_t>(slices.semaphore_plans.size()),
      static_cast<int64_t>(slices.transport_plans.size()));
}

Array<TTResourceDemand> RefreshResourceDemandCounters(
    const Array<TTResourceDemand>& demands, const TTProgramSlices& slices) {
  Array<TTResourceDemand> refreshed;
  for (const TTResourceDemand& demand : demands) {
    refreshed.push_back(RefreshResourceDemandCounters(demand, slices));
  }
  return refreshed;
}

String CoreGridRequirement(const TTProgramSlices& slices) {
  if (slices.core_groups.empty()) {
    return String("unassigned");
  }
  const TTCoreGroup& core_group = slices.core_groups[0];
  return String("core_group:" + static_cast<std::string>(core_group->name) +
                ";grid:" + std::to_string(core_group->logical_grid_x) + "x" +
                std::to_string(core_group->logical_grid_y));
}

String DRAMViewRequirement(const TTProgramSlices& slices) {
  int64_t dram_buffers = 0;
  for (const TTBufferDistributionPlan& distribution :
       slices.buffer_distribution_plans) {
    if (distribution->memory_space == "DRAM") {
      ++dram_buffers;
    }
  }
  return String("dram_buffer_views:" + std::to_string(dram_buffers));
}

constexpr int64_t kDefaultCBIdLimit = 64;
constexpr int64_t kDefaultWorkerL1BudgetBytes = 1572864;
constexpr int64_t kDefaultL1AlignmentBytes = 32;

int64_t PositiveOrDefault(int64_t value, int64_t default_value) {
  return value > 0 ? value : default_value;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  if (value <= 0) {
    return 0;
  }
  if (alignment <= 1) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

int64_t HardwareCBIdLimit(const std::optional<TTHardwareModel>& maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultCBIdLimit;
  }
  return PositiveOrDefault(maybe_hardware_model.value()->max_cb_count, kDefaultCBIdLimit);
}

int64_t HardwareWorkerL1BudgetBytes(
    const std::optional<TTHardwareModel>& maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultWorkerL1BudgetBytes;
  }
  return PositiveOrDefault(maybe_hardware_model.value()->worker_l1_size,
                           kDefaultWorkerL1BudgetBytes);
}

int64_t HardwareL1AlignmentBytes(const std::optional<TTHardwareModel>& maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultL1AlignmentBytes;
  }
  return PositiveOrDefault(maybe_hardware_model.value()->l1_allocation_alignment_bytes,
                           kDefaultL1AlignmentBytes);
}

int64_t TotalAlignedCBL1Bytes(const Array<TTCBPlan>& cb_plans, int64_t alignment) {
  int64_t bytes = 0;
  for (const TTCBPlan& cb_plan : cb_plans) {
    bytes += AlignUp(cb_plan->num_pages * cb_plan->page_size_bytes, alignment);
  }
  return bytes;
}

bool IsL1MemorySpace(const String& memory_space) {
  return memory_space == "L1" || memory_space == "interleaved_l1";
}

int64_t TotalAlignedL1BufferBytes(
    const Array<TTBufferDistributionPlan>& buffer_distribution_plans,
    int64_t alignment) {
  int64_t bytes = 0;
  for (const TTBufferDistributionPlan& plan : buffer_distribution_plans) {
    if (!IsL1MemorySpace(plan->memory_space) || plan->page_size_bytes <= 0) {
      continue;
    }
    bytes += AlignUp(plan->page_size_bytes, alignment);
  }
  return bytes;
}

Array<TTResourcePressureReport> BuildResourcePressureReports(
    const TTProgramSlices& slices,
    const std::optional<TTHardwareModel>& maybe_hardware_model) {
  const int64_t cb_id_limit = HardwareCBIdLimit(maybe_hardware_model);
  const int64_t worker_l1_budget_bytes =
      HardwareWorkerL1BudgetBytes(maybe_hardware_model);
  const int64_t l1_alignment_bytes = HardwareL1AlignmentBytes(maybe_hardware_model);
  const int64_t per_core_cb_id_pressure = static_cast<int64_t>(slices.cb_plans.size());
  const int64_t per_core_cb_l1_bytes = TotalCBL1Bytes(slices.cb_plans);
  const int64_t per_core_cb_l1_aligned_bytes =
      TotalAlignedCBL1Bytes(slices.cb_plans, l1_alignment_bytes);
  const int64_t l1_alignment_waste_bytes =
      per_core_cb_l1_aligned_bytes - per_core_cb_l1_bytes;
  const int64_t per_core_l1_buffer_bytes =
      TotalAlignedL1BufferBytes(slices.buffer_distribution_plans, l1_alignment_bytes);
  const int64_t max_simultaneous_l1_bytes =
      per_core_cb_l1_aligned_bytes + per_core_l1_buffer_bytes;

  Array<TTResourcePressureReport> reports;
  for (const TTResourceDemand& demand : slices.resource_demands) {
    Array<String> unsupported_reasons = demand->tile_compute_unsupported_reasons;
    if (per_core_cb_id_pressure > cb_id_limit) {
      unsupported_reasons.push_back(
          String("CB id pressure exceeds hardware limit: required " +
                 std::to_string(per_core_cb_id_pressure) + ", limit " +
                 std::to_string(cb_id_limit)));
    }
    if (max_simultaneous_l1_bytes > worker_l1_budget_bytes) {
      unsupported_reasons.push_back(
          String("L1 pressure exceeds worker budget: required " +
                 std::to_string(max_simultaneous_l1_bytes) + ", budget " +
                 std::to_string(worker_l1_budget_bytes)));
    }
    reports.push_back(TTResourcePressureReport(
        String("resource_pressure_" + static_cast<std::string>(demand->kernel_name)),
        demand->kernel_name, demand->core_group, demand->core_group_index,
        demand->tile_compute_unsupported_reasons,
        demand->tile_compute_materialization_demands,
        per_core_cb_id_pressure, per_core_cb_l1_bytes,
        per_core_l1_buffer_bytes, max_simultaneous_l1_bytes,
        cb_id_limit, worker_l1_budget_bytes, l1_alignment_bytes,
        per_core_cb_l1_aligned_bytes, l1_alignment_waste_bytes,
        CoreGridRequirement(slices), DRAMViewRequirement(slices),
        unsupported_reasons));
  }
  return reports;
}

void RefreshResourcePlanningSlices(
    TTProgramSlices* slices,
    std::optional<TTHardwareModel> maybe_hardware_model = std::nullopt) {
  if (slices->resource_demands.empty()) {
    return;
  }
  slices->resource_demands =
      RefreshResourceDemandCounters(slices->resource_demands, *slices);
  slices->resource_pressure_reports =
      BuildResourcePressureReports(*slices, maybe_hardware_model);
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
        plan->name, plan->kernel_name, kernel_plan_index, plan->kind, plan->operation_name,
        plan->enabled, plan->operand_bindings, plan->problem_shape_axes, plan->problem_shape,
        plan->tile_shape, plan->block_shape, plan->subblock_shape, plan->accumulator_dtype,
        plan->mbarrier_buffer, plan->mbarrier_scope, plan->mbarrier_index_exprs,
        plan->tile_compute_dag_node_id, plan->tile_compute_source_emitter,
        plan->tile_compute_materialization_policy, plan->tile_compute_fanout_use_count,
        plan->tile_compute_fanout_policy));
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
    transport_plans.push_back(
        TTTransportPlan(edge->name, edge->kind, edge->producer_unit_index,
                        edge->consumer_unit_index, String("tensor"),
                        DeriveTransportDeliveryKind(edge), edge->subject));
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
                                    sync->materialization_kind));
  }
  return sync_plans;
}

Array<TTDstLayoutPlan> BuildDstLayoutPlans(const Array<TTABIPlan>& abi_plans) {
  Array<TTDstLayoutPlan> dst_layouts;
  std::unordered_set<std::string> seen;
  for (const TTABIPlan& abi : abi_plans) {
    for (const TTCompileTimeArgSpec& spec : abi->compile_time_arg_specs) {
      String buffer = spec->buffer;
      String layout = spec->layout;
      String memory_space = spec->memory_space;
      if (buffer.empty() || layout.empty() || memory_space.empty()) {
        continue;
      }
      std::string dedupe = str(buffer) + "|" + str(layout) + "|" + str(memory_space);
      if (!seen.insert(dedupe).second) {
        continue;
      }
      const int64_t page_size_bytes = spec->transport_page_size;
      dst_layouts.push_back(
          TTDstLayoutPlan(String("dst_layout_" + dedupe), buffer, layout, memory_space,
                          page_size_bytes));
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
  };

  std::unordered_map<std::string, DstLayoutInfo> dst_layout_by_buffer;
  for (const TTDstLayoutPlan& dst_layout : dst_layout_plans) {
    DstLayoutInfo info;
    info.layout = dst_layout->layout;
    info.memory_space = NormalizeMemorySpace(dst_layout->memory_space);
    info.page_size_bytes = dst_layout->page_size_bytes;
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
    String spatial_layout = layout_spec->name;
    String spatial_distribution_kind = layout_spec->distribution_kind;
    String abi_layout;
    String abi_memory_space;
    auto dst_it = dst_layout_by_buffer.find(buffer);
    if (dst_it != dst_layout_by_buffer.end()) {
      layout = dst_it->second.layout;
      memory_space = dst_it->second.memory_space;
      page_size_bytes = dst_it->second.page_size_bytes;
      abi_layout = dst_it->second.layout;
      abi_memory_space = dst_it->second.memory_space;
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
        layout_info.inverse_logical_index_vars, layout_info.inverse_logical_index_exprs,
        spatial_layout, spatial_distribution_kind, abi_layout, abi_memory_space));
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
    for (const Integer& index : cb_plan->requirement_indices) {
      cb_plan_index_by_requirement_index[index->value] = cb_plan_index;
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
        plan->target_kernel, plan->bridge_kind, plan->materialization_kind,
        plan->materialization_protocol, plan->publication_protocol, cb_plan_indices,
        plan->required_sync_plan_indices, plan->produced_live_form));
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
  Array<TTExecutionPlan> execution_plans;
  execution_plans.push_back(TTExecutionPlan(String("main_execution"), kernel_names, phase_indices));
  return execution_plans;
}

}  // namespace

tvm::transform::Pass PlanTTBlocks() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    std::optional<TTHardwareModel> maybe_hardware_model =
        EnsureBlackholeHardwareModel(&mod);
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan = RequireValidatedSpatialPlan(func.value(), "PlanTTBlocks");
      PlanTTCoreGroups planner;
      tir::PrimFunc planned = planner.Transform(func.value(), maybe_hardware_model);
      const Array<TTCoreGroup> core_groups = BuildCoreGroups(planner);
      TTProgramSlices slices = GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.mesh_plans = BuildUnitMeshPlans();
      slices.block_plans = BuildBlockPlans(spatial_plan, core_groups);
      slices.core_groups = core_groups;
      slices.resource_demands = BuildTileComputeResourceDemands(func.value(), slices);
      RefreshResourcePlanningSlices(&slices, maybe_hardware_model);
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
      if (slices.resource_demands.empty()) {
        slices.resource_demands = BuildTileComputeResourceDemands(func.value(), slices);
      }
      RefreshResourcePlanningSlices(&slices);
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
      RefreshResourcePlanningSlices(&slices);
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
      RefreshResourcePlanningSlices(&slices);
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
      RefreshResourcePlanningSlices(&slices);
      tir::PrimFunc planned = WithTTProgramAttr(func.value(), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.PlanTTABI", {});
}

tvm::transform::Pass PlanTTExecution() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = EnsureBlackholeHardwareModel(&mod);
    ICHECK(maybe_hardware_model) << "PlanTTExecution requires blackhole target";
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
      RefreshResourcePlanningSlices(&slices, maybe_hardware_model);
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
      if (!slices.resource_demands.empty()) {
        ICHECK(!slices.resource_pressure_reports.empty())
            << "BuildTTProgram requires ResourcePressureReport for ResourceDemand owner truth";
      }

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
