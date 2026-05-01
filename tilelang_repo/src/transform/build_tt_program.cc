/*!
 * \file build_tt_program.cc
 * \brief Materialize TTProgram from SpatialPlan and planner results.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../tir/builtin_blackhole.h"
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

struct BufferStorageInfo {
  int64_t byte_size = 0;
  int64_t dtype_bytes = 0;
  std::string scope;
  Array<Integer> shape;
};

constexpr int64_t kBlackholeTileElements = 32 * 32;

std::unordered_map<std::string, LogicalTileLayoutInfo>
CollectLogicalTileLayoutsFromBody(const tir::Stmt &body) {
  class Collector final : public tir::StmtExprVisitor {
  public:
    std::unordered_map<std::string, LogicalTileLayoutInfo>
    Collect(const tir::Stmt &stmt) {
      layout_by_buffer_.clear();
      VisitStmt(stmt);
      return layout_by_buffer_;
    }

  private:
    void Record(const tir::Buffer &buffer, const Layout &layout) {
      const std::string scope = buffer.scope();
      if (scope != "local" && scope != "local.fragment" &&
          scope != "blackhole.acc") {
        return;
      }
      auto maybe_spec = TryBuildBufferTileBridgeSpec(buffer, layout);
      if (!maybe_spec) {
        return;
      }
      const Map<String, Any> &spec = maybe_spec.value();
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
        info.inverse_logical_index_vars =
            Downcast<Array<PrimExpr>>(value.value());
      }
      if (auto value =
              spec.Get(String(schema_key::kInverseLogicalIndexExprs))) {
        info.inverse_logical_index_exprs =
            Downcast<Array<PrimExpr>>(value.value());
      }
      layout_by_buffer_.emplace(buffer_name, std::move(info));
    }

    void VisitStmt_(const tir::BlockNode *op) final {
      if (op->annotations.count(attr::kLayoutMap)) {
        if (auto layout_map_any = op->annotations.Get(attr::kLayoutMap)) {
          auto layout_map = layout_map_any->as<Map<tir::Buffer, Layout>>();
          if (layout_map && layout_map.value().defined()) {
            for (const auto &[buffer, layout] : layout_map.value()) {
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

std::optional<int64_t> ConstIntValue(const PrimExpr &expr) {
  if (const auto *int_imm = expr.as<IntImmNode>()) {
    return int_imm->value;
  }
  return std::nullopt;
}

int64_t DTypeStorageBytes(const DataType &dtype) {
  return std::max<int64_t>(1, (static_cast<int64_t>(dtype.bits()) *
                                   static_cast<int64_t>(dtype.lanes()) +
                               7) /
                                  8);
}

int64_t EstimateBufferByteSize(const tir::Buffer &buffer) {
  int64_t elements = 1;
  if (buffer->shape.empty()) {
    return 0;
  }
  for (const PrimExpr &extent : buffer->shape) {
    auto maybe_extent = ConstIntValue(extent);
    if (!maybe_extent || maybe_extent.value() <= 0) {
      return 0;
    }
    elements *= maybe_extent.value();
  }
  return elements * DTypeStorageBytes(buffer->dtype);
}

Array<Integer> ExtractStaticIntegerShape(const ffi::Array<PrimExpr> &shape) {
  Array<Integer> result;
  for (const PrimExpr &extent : shape) {
    auto maybe_extent = ConstIntValue(extent);
    if (!maybe_extent || maybe_extent.value() <= 0) {
      return Array<Integer>();
    }
    result.push_back(Integer(maybe_extent.value()));
  }
  return result;
}

std::unordered_map<std::string, BufferStorageInfo>
CollectBufferStorageInfo(const tir::PrimFunc &func) {
  std::unordered_map<std::string, BufferStorageInfo> info_by_name;

  auto record_buffer = [&](const tir::Buffer &buffer) {
    const std::string name = static_cast<std::string>(buffer->name);
    if (name.empty()) {
      return;
    }
    BufferStorageInfo &info = info_by_name[name];
    info.byte_size = std::max(info.byte_size, EstimateBufferByteSize(buffer));
    info.dtype_bytes = std::max(info.dtype_bytes, DTypeStorageBytes(buffer->dtype));
    if (info.shape.empty()) {
      info.shape = ExtractStaticIntegerShape(buffer->shape);
    }
    if (info.scope.empty()) {
      info.scope = buffer.scope();
    }
  };

  for (const auto &entry : func->buffer_map) {
    record_buffer(entry.second);
  }
  tir::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *store = node.as<tir::BufferStoreNode>()) {
      record_buffer(store->buffer);
    }
    if (const auto *load = node.as<tir::BufferLoadNode>()) {
      record_buffer(load->buffer);
    }
    if (const auto *block = node.as<tir::BlockNode>()) {
      for (const tir::Buffer &buffer : block->alloc_buffers) {
        record_buffer(buffer);
      }
    }
    if (const auto *decl = node.as<tir::DeclBufferNode>()) {
      record_buffer(decl->buffer);
    }
  });
  return info_by_name;
}

std::unordered_map<std::string, std::string>
CollectSourceBufferByMaterializedTarget(const tir::PrimFunc &func,
                                        const Array<TTCBPlan> &cb_plans) {
  std::unordered_map<std::string, std::string> source_by_target;
  std::unordered_set<std::string> ambiguous_targets;
  std::unordered_map<const tir::VarNode *, std::string> buffer_by_data;
  std::unordered_map<int64_t, std::vector<std::string>> cb_targets;

  auto record_buffer_data = [&](const tir::Buffer &buffer) {
    const tir::VarNode *data = BufferDataIdentity(buffer);
    const std::string name = BufferIdentityName(buffer);
    if (data != nullptr && !name.empty()) {
      buffer_by_data[data] = name;
    }
  };

  for (const auto &entry : func->buffer_map) {
    record_buffer_data(entry.second);
  }
  for (const TTCBPlan &cb_plan : cb_plans) {
    for (const String &requirement_name : cb_plan->requirement_names) {
      const std::string target = str(requirement_name);
      if (!target.empty()) {
        cb_targets[cb_plan->cb_id].push_back(target);
      }
    }
  }

  auto bind_source = [&](const std::string &target, const std::string &source) {
    if (target.empty() || source.empty() || target == source) {
      return;
    }
    auto it = source_by_target.find(target);
    if (it == source_by_target.end()) {
      source_by_target.emplace(target, source);
      return;
    }
    if (it->second != source) {
      ambiguous_targets.insert(target);
    }
  };

  tir::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *decl = node.as<tir::DeclBufferNode>()) {
      record_buffer_data(decl->buffer);
    }
    const auto *store = node.as<tir::BufferStoreNode>();
    if (store != nullptr) {
      const auto *load = store->value.as<tir::BufferLoadNode>();
      if (load != nullptr) {
        bind_source(BufferIdentityName(store->buffer),
                    BufferIdentityName(load->buffer));
      }
    }

    const auto *call = node.as<CallNode>();
    if (call == nullptr || call->args.size() < 3 ||
        (!call->op.same_as(tir::builtin::blackhole_read_tile_to_cb()) &&
         !call->op.same_as(tir::builtin::blackhole_read_page_to_cb()) &&
         !call->op.same_as(tir::builtin::blackhole_read_bcast_cols_to_cb()))) {
      return;
    }
    const auto *source_var = call->args[0].as<tir::VarNode>();
    const auto *cb_id = call->args[2].as<IntImmNode>();
    if (source_var == nullptr || cb_id == nullptr) {
      return;
    }
    auto source_it = buffer_by_data.find(source_var);
    auto targets_it = cb_targets.find(cb_id->value);
    if (targets_it == cb_targets.end()) {
      return;
    }
    const std::string source_name = source_it != buffer_by_data.end()
                                        ? source_it->second
                                        : std::string(source_var->name_hint);
    if (source_name.empty()) {
      return;
    }
    for (const std::string &target : targets_it->second) {
      bind_source(target, source_name);
    }
  });
  for (const std::string &target : ambiguous_targets) {
    source_by_target.erase(target);
  }
  return source_by_target;
}

tir::PrimFunc StripTTIntermediateAttrs(tir::PrimFunc func) {
  static const char *kIntermediateSeedAttrs[] = {attr::kTLTTSemaphorePlans};
  for (const char *key : kIntermediateSeedAttrs) {
    func = tvm::WithoutAttr(std::move(func), key);
  }
  return func;
}

std::optional<Target> FindBlackholeTarget(const IRModule &mod) {
  for (const auto &[gvar, base_func] : mod->functions) {
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

std::optional<TTHardwareModel> EnsureBlackholeHardwareModel(IRModule *mod) {
  if (auto maybe_hardware_model = GetModuleTTHardwareModel(*mod)) {
    return maybe_hardware_model.value();
  }
  auto maybe_target = FindBlackholeTarget(*mod);
  if (!maybe_target) {
    return std::nullopt;
  }
  TTHardwareModel hardware_model =
      BuildBlackholeTTHardwareModel(maybe_target.value());
  *mod = (*mod)->ShallowCopy();
  (*mod)->UpdateGlobalInfo(attr::kTLTTHardwareModel,
                           Array<GlobalInfo>{hardware_model});
  return hardware_model;
}

String GetStringOrDefault(const Map<String, Any> &dict, const char *key,
                          String default_value = String()) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<String>(value.value());
  }
  return default_value;
}

int64_t GetIntegerOrDefault(const Map<String, Any> &dict, const char *key,
                            int64_t default_value = 0) {
  if (auto value = dict.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

Map<String, Any> AsMap(const Any &any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

SpatialPlan RequireValidatedSpatialPlan(const tir::PrimFunc &func,
                                        const char *pass_name) {
  auto maybe_spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(maybe_spatial_plan)
      << pass_name
      << " requires tl.spatial_plan; run BuildSpatialPlan and "
         "ValidateSpatialPlan before target planning";
  ICHECK(
      func->GetAttr<Bool>(attr::kTLSpatialPlanValidated, Bool(false)).value())
      << pass_name
      << " requires validated SpatialPlan; run ValidateSpatialPlan before "
         "target planning";
  return maybe_spatial_plan.value();
}

void RequireTTMetalBuiltinSelection(const tir::PrimFunc &func,
                                    const char *pass_name) {
  ICHECK(func->GetAttr<Bool>(kTLBlackholeTTMetalBuiltinSelection, Bool(false))
             .value())
      << pass_name
      << " requires exact TT-Metal builtin selection; run "
         "SelectBlackholeTTMetalBuiltins after SplitBlackholeKernel";
}

Map<String, Any> CopyAttrs(const tir::PrimFunc &func) {
  return func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
}

struct TTProgramSlices {
  String entry_name;
  String member_func;
  Array<TTMeshPlan> mesh_plans;
  Array<TTBufferDistributionPlan> buffer_distribution_plans;
  Array<TTTensorMemoryConfigPlan> tensor_memory_config_plans;
  Array<TTOpShardingContract> op_sharding_contracts;
  Array<TTPlacementResolutionPlan> placement_resolution_plans;
  Array<TTReshardPlan> reshard_plans;
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

TTProgramSlices UnpackTTProgram(const TTProgram &program) {
  TTProgramSlices slices;
  slices.entry_name = program->entry_name;
  slices.member_func = program->member_func;
  slices.mesh_plans = program->mesh_plans;
  slices.buffer_distribution_plans = program->buffer_distribution_plans;
  slices.tensor_memory_config_plans = program->tensor_memory_config_plans;
  slices.op_sharding_contracts = program->op_sharding_contracts;
  slices.placement_resolution_plans = program->placement_resolution_plans;
  slices.reshard_plans = program->reshard_plans;
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
  return TTProgram(
      std::move(slices.entry_name), std::move(slices.member_func),
      std::move(slices.mesh_plans), std::move(slices.buffer_distribution_plans),
      std::move(slices.tensor_memory_config_plans),
      std::move(slices.op_sharding_contracts),
      std::move(slices.placement_resolution_plans),
      std::move(slices.reshard_plans),
      std::move(slices.block_plans), std::move(slices.kernel_plans),
      std::move(slices.compute_op_plans), std::move(slices.transport_plans),
      std::move(slices.sync_plans), std::move(slices.abi_plans),
      std::move(slices.execution_plans), std::move(slices.kernels),
      std::move(slices.core_groups), std::move(slices.cb_plans),
      std::move(slices.semaphore_plans), std::move(slices.compute_sync_plans),
      std::move(slices.dst_layout_plans), std::move(slices.live_form_plans),
      std::move(slices.materialization_plans),
      std::move(slices.consumer_binding_plans),
      std::move(slices.resource_demands),
      std::move(slices.resource_pressure_reports));
}

TTProgramSlices GetOrCreateTTProgramSlices(const tir::PrimFunc &func,
                                           const GlobalVar &gvar,
                                           const SpatialPlan &spatial_plan) {
  if (auto maybe_program = func->GetAttr<TTProgram>(attr::kTLTTProgram)) {
    return UnpackTTProgram(maybe_program.value());
  }
  TTProgramSlices slices;
  slices.entry_name = gvar->name_hint;
  slices.member_func = spatial_plan->member_func;
  return slices;
}

TTProgram RequireStagedTTProgram(const tir::PrimFunc &func,
                                 const char *consumer,
                                 const char *next_step_guidance) {
  auto maybe_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  ICHECK(maybe_program) << consumer
                        << " requires staged tl.tt_program owner truth. "
                        << next_step_guidance;
  return maybe_program.value();
}

tir::PrimFunc WithTTProgramAttr(tir::PrimFunc func, TTProgram program) {
  Map<String, Any> attrs = CopyAttrs(func);
  attrs.Set(attr::kTLTTProgram, std::move(program));
  func.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
  return func;
}

const char *CBFlowClassToString(CBFlowClass flow_class) {
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

Array<TTCBPlan> BuildCBPlans(const std::vector<CBConfig> &configs) {
  Array<TTCBPlan> tt_cb_plans;
  for (const auto &config : configs) {
    Array<String> requirement_names;
    Array<Integer> requirement_indices;
    for (const auto &req_name : config.requirement_names) {
      requirement_names.push_back(String(req_name));
    }
    for (int req_index : config.requirement_indices) {
      requirement_indices.push_back(Integer(req_index));
    }
    tt_cb_plans.push_back(
        TTCBPlan(String(config.name), config.cb_id, String(config.role),
                 config.num_pages, config.page_size, String(config.data_format),
                 config.initial_reserve_pages,
                 String(CBFlowClassToString(config.flow_class)),
                 config.publish_pages_per_event, config.consume_pages_per_event,
                 config.lifetime_begin, config.lifetime_end, requirement_names,
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

Array<TTCoreGroup> BuildCoreGroups(const PlanTTCoreGroups &planner) {
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
  tt_core_groups.push_back(TTCoreGroup(
      String("main_core_group"), assignment.grid_x, assignment.grid_y,
      String("row_major"), physical_cores, work_packets));
  return tt_core_groups;
}

Array<TTBlockPlan> BuildBlockPlans(const SpatialPlan &plan,
                                   const Array<TTCoreGroup> &core_groups) {
  Array<Integer> task_indices;
  for (size_t i = 0; i < plan->execution_units.size(); ++i) {
    task_indices.push_back(Integer(static_cast<int64_t>(i)));
  }

  Array<TTBlockPlan> block_plans;
  for (size_t i = 0; i < core_groups.size(); ++i) {
    const TTCoreGroup &core_group = core_groups[i];
    block_plans.push_back(TTBlockPlan(
        String("block_plan_" + std::to_string(i)), String("core_group"),
        task_indices, core_group->name, static_cast<int64_t>(i)));
  }
  return block_plans;
}

Array<TTKernelPlan> BuildKernelPlans(const Array<TTKernel> &kernels) {
  Array<TTKernelPlan> kernel_plans;
  for (const TTKernel &kernel : kernels) {
    kernel_plans.push_back(
        TTKernelPlan(kernel->name, kernel->kind, kernel->core_type,
                     /*block_plan_index=*/0, kernel->abi_plan_index));
  }
  return kernel_plans;
}

bool IsTileComputeDAGOutputRole(const std::string &role) {
  return role == "output" || role == "c";
}

String PrimaryComputeKernelName(const TTProgramSlices &slices) {
  for (const TTKernelPlan &kernel_plan : slices.kernel_plans) {
    if (kernel_plan->kind == "compute") {
      return kernel_plan->name;
    }
  }
  for (const TTKernel &kernel : slices.kernels) {
    if (kernel->kind == "compute") {
      return kernel->name;
    }
  }
  if (!slices.kernel_plans.empty()) {
    return slices.kernel_plans[0]->name;
  }
  if (!slices.kernels.empty()) {
    return slices.kernels[0]->name;
  }
  return String("compute");
}

String PrimaryCoreGroupName(const TTProgramSlices &slices) {
  if (!slices.core_groups.empty()) {
    return slices.core_groups[0]->name;
  }
  return String("main_core_group");
}

int64_t PrimaryCoreGroupIndex(const TTProgramSlices &slices) {
  return slices.core_groups.empty() ? -1 : 0;
}

int64_t TotalCBL1Bytes(const Array<TTCBPlan> &cb_plans) {
  int64_t total = 0;
  for (const TTCBPlan &cb : cb_plans) {
    total += cb->num_pages * cb->page_size_bytes;
  }
  return total;
}

Array<TTTileComputeMaterializationDemand>
BuildTileComputeMaterializationDemands(const BlackholeTileComputeDAG &dag,
                                       const String &kernel_name,
                                       Array<String> *unsupported_reasons) {
  Array<TTTileComputeMaterializationDemand> demands;
  for (const BlackholeTileComputeDAGNode &node : dag.nodes) {
    const BlackholeTileComputeCoveringDecision decision =
        SelectBlackholeTileComputeCovering(node.op_name);
    if (!decision.selected) {
      unsupported_reasons->push_back(String("node " + std::to_string(node.id) +
                                            " operation " + node.op_name +
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

Array<TTTileComputeFanoutDemand>
BuildTileComputeFanoutDemands(const BlackholeTileComputeDAG &dag,
                              const String &kernel_name) {
  std::unordered_map<int64_t, std::vector<const BlackholeTileComputeDAGEdge *>>
      uses_by_producer;
  for (const BlackholeTileComputeDAGEdge &edge : dag.edges) {
    if (edge.producer_node < 0 || IsTileComputeDAGOutputRole(edge.value_role)) {
      continue;
    }
    uses_by_producer[edge.producer_node].push_back(&edge);
  }

  Array<TTTileComputeFanoutDemand> demands;
  std::vector<int64_t> producer_nodes;
  for (const auto &entry : uses_by_producer) {
    producer_nodes.push_back(entry.first);
  }
  std::sort(producer_nodes.begin(), producer_nodes.end());

  for (const int64_t producer_node : producer_nodes) {
    const std::vector<const BlackholeTileComputeDAGEdge *> &uses =
        uses_by_producer.at(producer_node);
    if (uses.size() < 2U || producer_node < 0 ||
        producer_node >= static_cast<int64_t>(dag.nodes.size())) {
      continue;
    }
    const BlackholeTileComputeDAGNode &producer = dag.nodes[producer_node];
    const bool requires_materialization =
        producer.side_effect_class == "tile_regs" ||
        producer.side_effect_class == "dst" ||
        producer.side_effect_class == "pack";
    Array<Integer> consumer_nodes;
    for (const BlackholeTileComputeDAGEdge *use : uses) {
      consumer_nodes.push_back(Integer(use->consumer_node));
    }
    const std::string name =
        "tile_compute_fanout_" + std::to_string(producer_node);
    demands.push_back(TTTileComputeFanoutDemand(
        String(name), kernel_name, producer_node, String(producer.op_name),
        String(uses.front()->value_repr), static_cast<int64_t>(uses.size()),
        consumer_nodes,
        String(requires_materialization ? "materialize_before_cross_event_use"
                                        : "share_live_value"),
        String("producer_use_count:" + std::to_string(uses.size()) +
               ";producer_side_effect:" + producer.side_effect_class)));
  }
  return demands;
}

Array<TTTileComputeFanoutDemand>
RefreshFanoutDemandKernelNames(const Array<TTTileComputeFanoutDemand> &demands,
                               const String &kernel_name) {
  Array<TTTileComputeFanoutDemand> refreshed;
  for (const TTTileComputeFanoutDemand &demand : demands) {
    refreshed.push_back(TTTileComputeFanoutDemand(
        demand->name, kernel_name, demand->producer_node,
        demand->producer_operation, demand->value_repr, demand->use_count,
        demand->consumer_nodes, demand->policy, demand->evidence));
  }
  return refreshed;
}

Array<TTTileComputeMaterializationDemand>
RefreshMaterializationDemandKernelNames(
    const Array<TTTileComputeMaterializationDemand> &demands,
    const String &kernel_name) {
  Array<TTTileComputeMaterializationDemand> refreshed;
  for (const TTTileComputeMaterializationDemand &demand : demands) {
    refreshed.push_back(TTTileComputeMaterializationDemand(
        demand->name, kernel_name, demand->node_id, demand->operation_name,
        demand->pattern_name, demand->policy, demand->evidence));
  }
  return refreshed;
}

Array<TTResourceDemand>
BuildTileComputeResourceDemands(const tir::PrimFunc &func,
                                const TTProgramSlices &slices) {
  const BlackholeTileComputeDAG dag = BuildBlackholeTileComputeDAG(func);
  if (dag.nodes.empty()) {
    if (slices.cb_plans.empty() && slices.semaphore_plans.empty() &&
        slices.transport_plans.empty()) {
      return {};
    }
    const String kernel_name = PrimaryComputeKernelName(slices);
    Array<TTResourceDemand> demands;
    demands.push_back(TTResourceDemand(
        String("resource_demand_" + static_cast<std::string>(kernel_name)),
        kernel_name, PrimaryCoreGroupName(slices),
        PrimaryCoreGroupIndex(slices), Array<TTTileComputeFanoutDemand>{},
        Array<TTTileComputeMaterializationDemand>{}, Array<String>{},
        static_cast<int64_t>(slices.cb_plans.size()),
        TotalCBL1Bytes(slices.cb_plans),
        static_cast<int64_t>(slices.semaphore_plans.size()),
        static_cast<int64_t>(slices.transport_plans.size())));
    return demands;
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
      static_cast<int64_t>(slices.cb_plans.size()),
      TotalCBL1Bytes(slices.cb_plans),
      static_cast<int64_t>(slices.semaphore_plans.size()),
      static_cast<int64_t>(slices.transport_plans.size())));
  return demands;
}

TTResourceDemand RefreshResourceDemandCounters(const TTResourceDemand &demand,
                                               const TTProgramSlices &slices) {
  const String kernel_name = PrimaryComputeKernelName(slices);
  return TTResourceDemand(
      String("resource_demand_" + static_cast<std::string>(kernel_name)),
      kernel_name, PrimaryCoreGroupName(slices), PrimaryCoreGroupIndex(slices),
      RefreshFanoutDemandKernelNames(demand->tile_compute_fanout_demands,
                                     kernel_name),
      RefreshMaterializationDemandKernelNames(
          demand->tile_compute_materialization_demands, kernel_name),
      demand->tile_compute_unsupported_reasons,
      static_cast<int64_t>(slices.cb_plans.size()),
      TotalCBL1Bytes(slices.cb_plans),
      static_cast<int64_t>(slices.semaphore_plans.size()),
      static_cast<int64_t>(slices.transport_plans.size()));
}

Array<TTResourceDemand>
RefreshResourceDemandCounters(const Array<TTResourceDemand> &demands,
                              const TTProgramSlices &slices) {
  Array<TTResourceDemand> refreshed;
  for (const TTResourceDemand &demand : demands) {
    refreshed.push_back(RefreshResourceDemandCounters(demand, slices));
  }
  return refreshed;
}

String CoreGridRequirement(const TTProgramSlices &slices) {
  if (slices.core_groups.empty()) {
    return String("unassigned");
  }
  const TTCoreGroup &core_group = slices.core_groups[0];
  return String("core_group:" + static_cast<std::string>(core_group->name) +
                ";grid:" + std::to_string(core_group->logical_grid_x) + "x" +
                std::to_string(core_group->logical_grid_y));
}

String DRAMViewRequirement(const TTProgramSlices &slices) {
  int64_t dram_buffers = 0;
  for (const TTBufferDistributionPlan &distribution :
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

int64_t
HardwareCBIdLimit(const std::optional<TTHardwareModel> &maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultCBIdLimit;
  }
  return PositiveOrDefault(maybe_hardware_model.value()->max_cb_count,
                           kDefaultCBIdLimit);
}

int64_t HardwareWorkerL1BudgetBytes(
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultWorkerL1BudgetBytes;
  }
  return PositiveOrDefault(maybe_hardware_model.value()->worker_l1_size,
                           kDefaultWorkerL1BudgetBytes);
}

int64_t HardwareL1AlignmentBytes(
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  if (!maybe_hardware_model) {
    return kDefaultL1AlignmentBytes;
  }
  return PositiveOrDefault(
      maybe_hardware_model.value()->l1_allocation_alignment_bytes,
      kDefaultL1AlignmentBytes);
}

int64_t TotalAlignedCBL1Bytes(const Array<TTCBPlan> &cb_plans,
                              int64_t alignment) {
  int64_t bytes = 0;
  for (const TTCBPlan &cb_plan : cb_plans) {
    bytes += AlignUp(cb_plan->num_pages * cb_plan->page_size_bytes, alignment);
  }
  return bytes;
}

bool IsL1MemorySpace(const String &memory_space) {
  return memory_space == "L1" || memory_space == "interleaved_l1";
}

bool IsCBBackedL1Layout(const String &layout) {
  return layout == "circular_buffer";
}

bool IsSharedSpatialDistributionKind(const String &distribution_kind) {
  return distribution_kind == "shared_visible";
}

int64_t TotalAlignedL1BufferBytes(
    const Array<TTBufferDistributionPlan> &buffer_distribution_plans,
    int64_t alignment) {
  int64_t bytes = 0;
  for (const TTBufferDistributionPlan &plan : buffer_distribution_plans) {
    if (!IsL1MemorySpace(plan->memory_space) ||
        IsCBBackedL1Layout(plan->layout) || plan->page_size_bytes <= 0) {
      continue;
    }
    bytes += AlignUp(plan->page_size_bytes, alignment);
  }
  return bytes;
}

Array<TTResourcePressureReport> BuildResourcePressureReports(
    const TTProgramSlices &slices,
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  const int64_t cb_id_limit = HardwareCBIdLimit(maybe_hardware_model);
  const int64_t worker_l1_budget_bytes =
      HardwareWorkerL1BudgetBytes(maybe_hardware_model);
  const int64_t l1_alignment_bytes =
      HardwareL1AlignmentBytes(maybe_hardware_model);
  const int64_t per_core_cb_id_pressure =
      static_cast<int64_t>(slices.cb_plans.size());
  const int64_t per_core_cb_l1_bytes = TotalCBL1Bytes(slices.cb_plans);
  const int64_t per_core_cb_l1_aligned_bytes =
      TotalAlignedCBL1Bytes(slices.cb_plans, l1_alignment_bytes);
  const int64_t l1_alignment_waste_bytes =
      per_core_cb_l1_aligned_bytes - per_core_cb_l1_bytes;
  const int64_t per_core_l1_buffer_bytes = TotalAlignedL1BufferBytes(
      slices.buffer_distribution_plans, l1_alignment_bytes);
  const int64_t max_simultaneous_l1_bytes =
      per_core_cb_l1_aligned_bytes + per_core_l1_buffer_bytes;

  Array<TTResourcePressureReport> reports;
  for (const TTResourceDemand &demand : slices.resource_demands) {
    Array<String> unsupported_reasons =
        demand->tile_compute_unsupported_reasons;
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
        String("resource_pressure_" +
               static_cast<std::string>(demand->kernel_name)),
        demand->kernel_name, demand->core_group, demand->core_group_index,
        demand->tile_compute_unsupported_reasons,
        demand->tile_compute_materialization_demands, per_core_cb_id_pressure,
        per_core_cb_l1_bytes, per_core_l1_buffer_bytes,
        max_simultaneous_l1_bytes, cb_id_limit, worker_l1_budget_bytes,
        l1_alignment_bytes, per_core_cb_l1_aligned_bytes,
        l1_alignment_waste_bytes, CoreGridRequirement(slices),
        DRAMViewRequirement(slices), unsupported_reasons));
  }
  return reports;
}

void RefreshResourcePlanningSlices(
    TTProgramSlices *slices,
    std::optional<TTHardwareModel> maybe_hardware_model = std::nullopt) {
  if (slices->resource_demands.empty()) {
    return;
  }
  slices->resource_demands =
      RefreshResourceDemandCounters(slices->resource_demands, *slices);
  slices->resource_pressure_reports =
      BuildResourcePressureReports(*slices, maybe_hardware_model);
}

Array<TTComputeOpPlan>
AttachComputeOpKernelPlanIndices(const Array<TTComputeOpPlan> &compute_op_plans,
                                 const Array<TTKernelPlan> &kernel_plans) {
  std::unordered_map<std::string, int64_t> kernel_index_by_name;
  for (int64_t i = 0; i < static_cast<int64_t>(kernel_plans.size()); ++i) {
    kernel_index_by_name.emplace(
        static_cast<std::string>(kernel_plans[i]->name), i);
  }

  Array<TTComputeOpPlan> updated;
  for (const TTComputeOpPlan &plan : compute_op_plans) {
    auto kernel_index_it =
        kernel_index_by_name.find(static_cast<std::string>(plan->kernel_name));
    const int64_t kernel_plan_index =
        kernel_index_it == kernel_index_by_name.end() ? -1
                                                      : kernel_index_it->second;
    updated.push_back(TTComputeOpPlan(
        plan->name, plan->kernel_name, kernel_plan_index, plan->kind,
        plan->operation_name, plan->enabled, plan->operand_bindings,
        plan->problem_shape_axes, plan->problem_shape, plan->tile_shape,
        plan->block_shape, plan->subblock_shape, plan->accumulator_dtype,
        plan->mbarrier_buffer, plan->mbarrier_scope, plan->mbarrier_index_exprs,
        plan->tile_compute_dag_node_id, plan->tile_compute_source_emitter,
        plan->tile_compute_materialization_policy,
        plan->tile_compute_fanout_use_count, plan->tile_compute_fanout_policy));
  }
  return updated;
}

String DeriveTransportDeliveryKind(const DataflowEdge &edge) {
  if (edge->crosses_phase) {
    return String("phase_boundary_materialized");
  }
  if (edge->kind == "join") {
    return String("completion_visible");
  }
  return String("ordered");
}

String DeriveSyncOrderingKind(const DataflowEdge &edge) {
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

String DeriveSyncMaterializationKind(const DataflowEdge &edge) {
  if (edge->crosses_phase) {
    return String("phase_boundary_materialization");
  }
  if (edge->kind == "join") {
    return String("completion_visibility");
  }
  return String("phase_boundary");
}

Array<TTTransportPlan> BuildTransportPlans(const SpatialPlan &plan) {
  Array<TTTransportPlan> transport_plans;
  for (const DataflowEdge &edge : plan->dataflow_edges) {
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

Array<TTSemaphorePlan> BuildSemaphorePlans(const tir::PrimFunc &func) {
  if (auto maybe_semaphore_plans =
          func->GetAttr<Array<TTSemaphorePlan>>(attr::kTLTTSemaphorePlans)) {
    return maybe_semaphore_plans.value();
  }
  return Array<TTSemaphorePlan>();
}

Array<TTComputeSyncPlan> BuildComputeSyncPlans(const SpatialPlan &plan) {
  Array<TTComputeSyncPlan> sync_plans;
  for (const DataflowEdge &edge : plan->dataflow_edges) {
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
        edge->producer_unit_index == edge->consumer_unit_index) {
      continue;
    }
    sync_plans.push_back(TTComputeSyncPlan(
        String("sync_" + std::string(edge->name)), edge->kind,
        edge->producer_unit_index, edge->consumer_unit_index,
        DeriveSyncOrderingKind(edge), DeriveSyncMaterializationKind(edge)));
  }
  return sync_plans;
}

Array<TTSyncPlan>
BuildSyncPlans(const Array<TTComputeSyncPlan> &compute_sync_plans) {
  Array<TTSyncPlan> sync_plans;
  for (const TTComputeSyncPlan &sync : compute_sync_plans) {
    sync_plans.push_back(
        TTSyncPlan(sync->name, sync->kind, sync->source_task_index,
                   sync->target_task_index, sync->ordering_kind,
                   sync->materialization_kind));
  }
  return sync_plans;
}

Array<TTDstLayoutPlan> BuildDstLayoutPlans(const Array<TTABIPlan> &abi_plans) {
  Array<TTDstLayoutPlan> dst_layouts;
  std::unordered_set<std::string> seen;
  for (const TTABIPlan &abi : abi_plans) {
    for (const TTCompileTimeArgSpec &spec : abi->compile_time_arg_specs) {
      String buffer = spec->buffer;
      String layout = spec->layout;
      String memory_space = spec->memory_space;
      if (buffer.empty() || layout.empty() || memory_space.empty()) {
        continue;
      }
      std::string dedupe =
          str(buffer) + "|" + str(layout) + "|" + str(memory_space);
      if (!seen.insert(dedupe).second) {
        continue;
      }
      const int64_t page_size_bytes = spec->transport_page_size;
      dst_layouts.push_back(TTDstLayoutPlan(String("dst_layout_" + dedupe),
                                            buffer, layout, memory_space,
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

String MemorySpaceFromLayoutScope(const String &scope) {
  const std::string value = str(scope);
  if (value == "global") {
    return String("DRAM");
  }
  return String("L1");
}

String LayoutFromLayoutScope(const String &scope) {
  const std::string value = str(scope);
  if (value == "global") {
    return String("interleaved");
  }
  if (value.rfind("blackhole.cb.", 0) == 0) {
    return String("circular_buffer");
  }
  return String("local");
}

int64_t EstimateDRAMPageSizeBytes(const BufferStorageInfo &storage_info) {
  if (storage_info.byte_size <= 0) {
    return 0;
  }
  const int64_t dtype_bytes = PositiveOrDefault(storage_info.dtype_bytes, 1);
  const int64_t tile_bytes = kBlackholeTileElements * dtype_bytes;
  return std::max<int64_t>(
      dtype_bytes, std::min(storage_info.byte_size, tile_bytes));
}

LogicalTileLayoutInfo
LogicalTileLayoutInfoFromLayoutSpec(const LayoutSpec &layout_spec) {
  LogicalTileLayoutInfo info;
  info.logical_shape = layout_spec->logical_shape;
  info.local_shape = layout_spec->local_shape;
  info.thread_extent = layout_spec->thread_extent;
  info.replicate_extent = layout_spec->replicate_extent;
  info.inverse_logical_index_vars = layout_spec->inverse_logical_index_vars;
  info.inverse_logical_index_exprs = layout_spec->inverse_logical_index_exprs;
  return info;
}

String ShardingStrategyForCoreGroup(const TTCoreGroup &core_group) {
  if (core_group->logical_grid_x > 1 && core_group->logical_grid_y > 1) {
    return String("block");
  }
  if (core_group->logical_grid_x > 1) {
    return String("width");
  }
  if (core_group->logical_grid_y > 1) {
    return String("height");
  }
  return String("block");
}

String ShardOrientationForCoreGroup(const TTCoreGroup &core_group) {
  const std::string linearization = str(core_group->linearization);
  if (linearization == "col_major") {
    return String("col_major");
  }
  return String("row_major");
}

Array<Integer> ShardGridShapeForCoreGroup(const TTCoreGroup &core_group) {
  int64_t min_x = std::numeric_limits<int64_t>::max();
  int64_t min_y = std::numeric_limits<int64_t>::max();
  int64_t max_x = -1;
  int64_t max_y = -1;
  for (const Any &item : core_group->physical_cores) {
    Map<String, Any> core = AsMap(item);
    if (core.empty()) {
      continue;
    }
    const int64_t core_x = GetIntegerOrDefault(core, "core_x", -1);
    const int64_t core_y = GetIntegerOrDefault(core, "core_y", -1);
    if (core_x < 0 || core_y < 0) {
      continue;
    }
    min_x = std::min(min_x, core_x);
    min_y = std::min(min_y, core_y);
    max_x = std::max(max_x, core_x);
    max_y = std::max(max_y, core_y);
  }
  if (max_x < 0 || max_y < 0) {
    return Array<Integer>();
  }
  return Array<Integer>{Integer(max_y - min_y + 1), Integer(max_x - min_x + 1)};
}

Array<Integer>
ShardDataShapeForBuffer(const LogicalTileLayoutInfo &layout_info,
                        const std::unordered_map<std::string, BufferStorageInfo>
                            &storage_info_by_buffer,
                        const std::string &buffer) {
  Array<Integer> shape = ExtractStaticIntegerShape(layout_info.logical_shape);
  if (!shape.empty()) {
    return shape;
  }
  auto storage_it = storage_info_by_buffer.find(buffer);
  if (storage_it != storage_info_by_buffer.end()) {
    return storage_it->second.shape;
  }
  return Array<Integer>();
}

Array<TTBufferDistributionPlan>
BuildBufferDistributionPlans(const SpatialPlan &spatial_plan,
                             const TTProgramSlices &slices,
                             const tir::PrimFunc &func) {
  struct DstLayoutInfo {
    String layout;
    String memory_space;
    int64_t page_size_bytes = 0;
  };

  std::unordered_map<std::string, DstLayoutInfo> dst_layout_by_buffer;
  for (const TTDstLayoutPlan &dst_layout : slices.dst_layout_plans) {
    DstLayoutInfo info;
    info.layout = dst_layout->layout;
    info.memory_space = NormalizeMemorySpace(dst_layout->memory_space);
    info.page_size_bytes = dst_layout->page_size_bytes;
    dst_layout_by_buffer.emplace(str(dst_layout->buffer), std::move(info));
  }

  const std::unordered_map<std::string, LogicalTileLayoutInfo>
      current_layouts_by_buffer = CollectLogicalTileLayoutsFromBody(func->body);
  const std::unordered_map<std::string, BufferStorageInfo>
      storage_info_by_buffer = CollectBufferStorageInfo(func);
  const std::unordered_map<std::string, std::string> source_buffer_by_target =
      CollectSourceBufferByMaterializedTarget(func, slices.cb_plans);
  std::unordered_map<std::string, int64_t> cb_page_size_by_buffer;
  for (const TTCBPlan &cb_plan : slices.cb_plans) {
    for (const String &requirement_name : cb_plan->requirement_names) {
      const std::string buffer_name = str(requirement_name);
      int64_t &page_size = cb_page_size_by_buffer[buffer_name];
      page_size = std::max(page_size, cb_plan->page_size_bytes);
    }
  }
  const bool has_core_group = !slices.core_groups.empty();
  Array<TTBufferDistributionPlan> distribution_plans;
  std::unordered_set<std::string> seen;
  for (const LayoutSpec &layout_spec : spatial_plan->layout_specs) {
    const std::string buffer = str(layout_spec->subject);
    if (buffer.empty() || !seen.insert(buffer).second) {
      continue;
    }
    String layout = LayoutFromLayoutScope(layout_spec->scope);
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
    auto storage_it = storage_info_by_buffer.find(buffer);
    if (page_size_bytes == 0 && storage_it != storage_info_by_buffer.end()) {
      if (str(memory_space) == "L1") {
        page_size_bytes = storage_it->second.byte_size;
      } else if (str(memory_space) == "DRAM") {
        page_size_bytes = EstimateDRAMPageSizeBytes(storage_it->second);
      }
    }
    auto cb_page_it = cb_page_size_by_buffer.find(buffer);
    const bool has_cb_page = cb_page_it != cb_page_size_by_buffer.end();
    if (str(memory_space) == "L1" && has_cb_page) {
      page_size_bytes = std::max(page_size_bytes, cb_page_it->second);
    }
    String distribution_kind = String("replicated");
    Array<Integer> shard_shape;
    Array<Integer> shard_grid_shape;
    String sharding_strategy = String("none");
    String shard_orientation = String("row_major");
    String source_buffer;
    String source_region_kind = String("none");
    Array<Integer> source_region_shape;
    String logical_index_mapping = String("none");
    String core_local_address_mapping = String("none");
    String attached_core_group_name;
    int64_t attached_core_group_index = -1;
    const String host_visibility = str(memory_space) == "DRAM"
                                       ? String("host_visible")
                                       : String("device_local");
    LogicalTileLayoutInfo layout_info =
        LogicalTileLayoutInfoFromLayoutSpec(layout_spec);
    auto current_layout_it = current_layouts_by_buffer.find(buffer);
    if (current_layout_it != current_layouts_by_buffer.end() &&
        current_layout_it->second.logical_shape.size() > 0) {
      layout_info = current_layout_it->second;
    }
    if (str(memory_space) == "DRAM" && str(layout) == "interleaved") {
      distribution_kind = String("interleaved");
      logical_index_mapping = String("interleaved_page_index");
    } else if (str(memory_space) == "L1" && has_core_group &&
               (IsSharedSpatialDistributionKind(spatial_distribution_kind) ||
                IsCBBackedL1Layout(layout) || has_cb_page)) {
      const TTCoreGroup &core_group = slices.core_groups[0];
      distribution_kind = String("sharded");
      shard_grid_shape = ShardGridShapeForCoreGroup(core_group);
      shard_shape =
          ShardDataShapeForBuffer(layout_info, storage_info_by_buffer, buffer);
      sharding_strategy = ShardingStrategyForCoreGroup(core_group);
      shard_orientation = ShardOrientationForCoreGroup(core_group);
      auto source_it = source_buffer_by_target.find(buffer);
      if (source_it != source_buffer_by_target.end()) {
        source_buffer = String(source_it->second);
        source_region_kind = String("per_work_tile");
        source_region_shape = shard_shape;
      }
      logical_index_mapping = String("work_packet_row_major");
      core_local_address_mapping = String("l1_shard_linear");
      attached_core_group_name = core_group->name;
      attached_core_group_index = 0;
    }
    distribution_plans.push_back(TTBufferDistributionPlan(
        String("buffer_distribution_" + buffer), String(buffer),
        String("unit_mesh"),
        /*mesh_plan_index=*/0, distribution_kind, layout, memory_space,
        page_size_bytes, shard_shape, shard_grid_shape, sharding_strategy,
        shard_orientation, source_buffer, source_region_kind,
        source_region_shape, logical_index_mapping, core_local_address_mapping,
        host_visibility, attached_core_group_name, attached_core_group_index,
        layout_info.logical_shape, layout_info.local_shape,
        layout_info.thread_extent, layout_info.replicate_extent,
        layout_info.inverse_logical_index_vars,
        layout_info.inverse_logical_index_exprs, spatial_layout,
        spatial_distribution_kind, abi_layout, abi_memory_space));
  }
  return distribution_plans;
}

std::string TTMemoryLayoutFromDistribution(const TTBufferDistributionPlan &plan) {
  const std::string distribution_kind = str(plan->distribution_kind);
  if (distribution_kind == "interleaved" || distribution_kind == "replicated") {
    return "INTERLEAVED";
  }
  if (distribution_kind == "sharded") {
    const std::string strategy = str(plan->sharding_strategy);
    if (strategy == "height") {
      return "HEIGHT_SHARDED";
    }
    if (strategy == "width") {
      return "WIDTH_SHARDED";
    }
    if (strategy == "block") {
      return "BLOCK_SHARDED";
    }
  }
  return "UNSUPPORTED";
}

std::string TensorMemoryConfigOrigin(
    const TTBufferDistributionPlan &distribution,
    const std::unordered_map<std::string, TensorPlacementIntent> &intent_by_subject) {
  if (!distribution->source_buffer.empty() ||
      (str(distribution->source_region_kind) != "" &&
       str(distribution->source_region_kind) != "none")) {
    return "materialization_requirement";
  }
  auto intent_it = intent_by_subject.find(str(distribution->buffer));
  if (intent_it != intent_by_subject.end()) {
    return str(intent_it->second->source);
  }
  return "derived_default";
}

std::string TensorMemoryConfigGridRef(const TTBufferDistributionPlan &distribution) {
  if (!distribution->attached_core_group.empty()) {
    return str(distribution->attached_core_group);
  }
  if (!distribution->mesh_plan.empty()) {
    return str(distribution->mesh_plan);
  }
  return "";
}

Array<TTTensorMemoryConfigPlan> BuildTensorMemoryConfigPlans(
    const SpatialPlan &spatial_plan,
    const Array<TTBufferDistributionPlan> &buffer_distribution_plans) {
  std::unordered_map<std::string, TensorPlacementIntent> intent_by_subject;
  for (const TensorPlacementIntent &intent : spatial_plan->tensor_placement_intents) {
    intent_by_subject.emplace(str(intent->subject), intent);
  }
  Array<TTTensorMemoryConfigPlan> plans;
  for (int64_t index = 0;
       index < static_cast<int64_t>(buffer_distribution_plans.size()); ++index) {
    const TTBufferDistributionPlan &distribution = buffer_distribution_plans[index];
    const std::string buffer = str(distribution->buffer);
    auto intent_it = intent_by_subject.find(buffer);
    Array<PrimExpr> logical_shape = distribution->logical_shape;
    String dtype;
    if (intent_it != intent_by_subject.end() && !intent_it->second->logical_shape.empty()) {
      logical_shape = intent_it->second->logical_shape;
    }
    plans.push_back(TTTensorMemoryConfigPlan(
        String("tensor_memory_config_" + buffer), String(buffer), String(""),
        logical_shape, dtype, String(TTMemoryLayoutFromDistribution(distribution)),
        String(str(distribution->memory_space) == "DRAM" ? "DRAM" : "L1"),
        String(TensorMemoryConfigGridRef(distribution)),
        distribution->shard_grid_shape, distribution->shard_shape,
        distribution->shard_orientation, distribution->sharding_strategy,
        Array<Integer>{}, String(TensorMemoryConfigOrigin(distribution, intent_by_subject)),
        distribution->source_buffer, distribution->name, index,
        str(distribution->memory_space) == "DRAM",
        !distribution->source_buffer.empty()));
  }
  return plans;
}

bool IsOutputOperandRole(const TTComputeOpPlan &compute_op,
                         const TTComputeOperandBindingPlan &binding) {
  const std::string kind = str(compute_op->kind);
  const std::string role = str(binding->role);
  if (kind == "gemm" && role == "c") {
    return true;
  }
  return role == "dst" || role == "out" || role == "output" ||
         role == "result";
}

Array<TTOpShardingContract> BuildOpShardingContracts(
    const Array<TTComputeOpPlan> &compute_op_plans,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans) {
  std::unordered_map<std::string, int64_t> memory_config_index_by_subject;
  for (int64_t index = 0;
       index < static_cast<int64_t>(tensor_memory_config_plans.size()); ++index) {
    const TTTensorMemoryConfigPlan &memory_config =
        tensor_memory_config_plans[index];
    memory_config_index_by_subject.emplace(str(memory_config->subject), index);
  }

  Array<TTOpShardingContract> contracts;
  for (int64_t compute_index = 0;
       compute_index < static_cast<int64_t>(compute_op_plans.size());
       ++compute_index) {
    const TTComputeOpPlan &compute_op = compute_op_plans[compute_index];
    for (const TTComputeOperandBindingPlan &binding :
         compute_op->operand_bindings) {
      const std::string operand_buffer = str(binding->buffer);
      auto memory_config_it =
          memory_config_index_by_subject.find(operand_buffer);
      if (memory_config_it == memory_config_index_by_subject.end()) {
        continue;
      }
      const int64_t memory_config_index = memory_config_it->second;
      const TTTensorMemoryConfigPlan &memory_config =
          tensor_memory_config_plans[static_cast<size_t>(memory_config_index)];
      const bool is_output = IsOutputOperandRole(compute_op, binding);
      Array<String> accepted_memory_layouts;
      accepted_memory_layouts.push_back(memory_config->memory_layout);
      Array<String> accepted_buffer_types;
      accepted_buffer_types.push_back(memory_config->buffer_type);
      Array<String> accepted_sharding_strategies;
      accepted_sharding_strategies.push_back(
          memory_config->shard_distribution_strategy.empty()
              ? String("none")
              : memory_config->shard_distribution_strategy);
      contracts.push_back(TTOpShardingContract(
          String("op_sharding_contract_" + str(compute_op->name) + "_" +
                 str(binding->role)),
          compute_op->name, compute_index, compute_op->operation_name,
          compute_op->kind, binding->role, binding->buffer,
          binding->host_buffer, memory_config->name, memory_config_index,
          accepted_memory_layouts, accepted_buffer_types,
          accepted_sharding_strategies, memory_config->shard_orientation,
          is_output ? String("produces_operand_placement")
                    : String("not_output"),
          /*may_request_input_conversion=*/false,
          /*can_produce_output_placement=*/is_output,
          /*direct_external_write_allowed=*/
          is_output && memory_config->buffer_type == "DRAM" &&
              !binding->host_buffer.empty(),
          String("")));
    }
  }
  return contracts;
}

Array<TTPlacementResolutionPlan> BuildPlacementResolutionPlans(
    const Array<TTOpShardingContract> &op_sharding_contracts,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans) {
  Array<TTPlacementResolutionPlan> resolutions;
  for (int64_t index = 0;
       index < static_cast<int64_t>(op_sharding_contracts.size()); ++index) {
    const TTOpShardingContract &contract = op_sharding_contracts[index];
    const int64_t memory_config_index = contract->memory_config_plan_index;
    ICHECK_GE(memory_config_index, 0)
        << "BuildPlacementResolutionPlans requires resolved memory config "
           "index";
    ICHECK_LT(memory_config_index,
              static_cast<int64_t>(tensor_memory_config_plans.size()))
        << "BuildPlacementResolutionPlans memory config index out of bounds";
    const TTTensorMemoryConfigPlan &memory_config =
        tensor_memory_config_plans[static_cast<size_t>(memory_config_index)];
    resolutions.push_back(TTPlacementResolutionPlan(
        String("placement_resolution_" + str(contract->name)),
        contract->name, index, contract->compute_op_plan,
        contract->compute_op_plan_index, contract->operand_role,
        memory_config->name, memory_config_index, memory_config->memory_layout,
        memory_config->buffer_type, String("selected_existing"),
        /*conversion_required=*/false, String(""), String("")));
  }
  return resolutions;
}

std::string ReshardConversionKind(const TTTensorMemoryConfigPlan &source,
                                  const TTTensorMemoryConfigPlan &target) {
  const std::string source_layout = str(source->memory_layout);
  const std::string target_layout = str(target->memory_layout);
  if (source_layout == "INTERLEAVED" && target_layout != "INTERLEAVED") {
    return "interleaved_to_sharded";
  }
  if (source_layout != "INTERLEAVED" && target_layout == "INTERLEAVED") {
    return "sharded_to_interleaved";
  }
  if (source_layout != target_layout ||
      str(source->shard_distribution_strategy) !=
          str(target->shard_distribution_strategy) ||
      str(source->shard_orientation) != str(target->shard_orientation)) {
    return "reshard";
  }
  return "unsupported";
}

Array<TTReshardPlan> BuildReshardPlans(
    const Array<TTBufferDistributionPlan> &buffer_distribution_plans,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans,
    const Array<TTMaterializationPlan> &materialization_plans) {
  std::unordered_map<std::string, int64_t> memory_config_index_by_subject;
  for (int64_t index = 0;
       index < static_cast<int64_t>(tensor_memory_config_plans.size()); ++index) {
    const TTTensorMemoryConfigPlan &memory_config =
        tensor_memory_config_plans[index];
    memory_config_index_by_subject.emplace(str(memory_config->subject), index);
  }
  std::unordered_map<std::string, int64_t> materialization_index_by_target;
  for (int64_t index = 0;
       index < static_cast<int64_t>(materialization_plans.size()); ++index) {
    const TTMaterializationPlan &materialization = materialization_plans[index];
    if (!materialization->target_buffer.empty()) {
      materialization_index_by_target.emplace(str(materialization->target_buffer),
                                              index);
    }
  }

  Array<TTReshardPlan> reshard_plans;
  for (const TTBufferDistributionPlan &distribution :
       buffer_distribution_plans) {
    if (distribution->source_buffer.empty()) {
      continue;
    }
    const std::string source = str(distribution->source_buffer);
    const std::string target = str(distribution->buffer);
    auto source_it = memory_config_index_by_subject.find(source);
    auto target_it = memory_config_index_by_subject.find(target);
    if (source_it == memory_config_index_by_subject.end() ||
        target_it == memory_config_index_by_subject.end()) {
      continue;
    }
    const int64_t source_index = source_it->second;
    const int64_t target_index = target_it->second;
    const TTTensorMemoryConfigPlan &source_config =
        tensor_memory_config_plans[static_cast<size_t>(source_index)];
    const TTTensorMemoryConfigPlan &target_config =
        tensor_memory_config_plans[static_cast<size_t>(target_index)];
    const std::string conversion_kind =
        ReshardConversionKind(source_config, target_config);
    auto materialization_it = materialization_index_by_target.find(target);
    String materialization_name;
    int64_t materialization_index = -1;
    String materialization_protocol;
    Array<Integer> required_cb_plan_indices;
    Array<Integer> required_sync_plan_indices;
    if (materialization_it != materialization_index_by_target.end()) {
      materialization_index = materialization_it->second;
      const TTMaterializationPlan &materialization =
          materialization_plans[static_cast<size_t>(materialization_index)];
      materialization_name = materialization->name;
      materialization_protocol = materialization->materialization_protocol;
      required_cb_plan_indices = materialization->required_cb_plan_indices;
      required_sync_plan_indices = materialization->required_sync_plan_indices;
    }
    const bool admitted = conversion_kind == "interleaved_to_sharded";
    if (admitted && materialization_protocol.empty()) {
      materialization_protocol = String("staged_copy");
    }
    reshard_plans.push_back(TTReshardPlan(
        String("reshard_" + source + "_to_" + target), String(source),
        String(target), source_config->name, source_index, target_config->name,
        target_index, String(conversion_kind), distribution->source_region_kind,
        distribution->source_region_shape, materialization_name,
        materialization_index, materialization_protocol,
        required_cb_plan_indices, required_sync_plan_indices, String("runtime"),
        String("planner"), admitted ? String("admitted") : String("unsupported"),
        admitted ? String("") : String("unsupported conversion kind")));
  }
  return reshard_plans;
}

Array<TTMaterializationPlan> RemapMaterializationCBRequirementIndices(
    const Array<TTMaterializationPlan> &materialization_plans,
    const Array<TTCBPlan> &cb_plans) {
  std::unordered_map<int64_t, int64_t> cb_plan_index_by_requirement_index;
  for (int64_t cb_plan_index = 0;
       cb_plan_index < static_cast<int64_t>(cb_plans.size()); ++cb_plan_index) {
    const TTCBPlan &cb_plan = cb_plans[static_cast<size_t>(cb_plan_index)];
    for (const Integer &index : cb_plan->requirement_indices) {
      cb_plan_index_by_requirement_index[index->value] = cb_plan_index;
    }
  }

  Array<TTMaterializationPlan> remapped;
  for (const TTMaterializationPlan &plan : materialization_plans) {
    Array<Integer> cb_plan_indices;
    for (const Integer &index : plan->required_cb_plan_indices) {
      const int64_t requirement_index = index->value;
      auto it = cb_plan_index_by_requirement_index.find(requirement_index);
      cb_plan_indices.push_back(Integer(
          it != cb_plan_index_by_requirement_index.end() ? it->second
                                                         : requirement_index));
    }
    remapped.push_back(TTMaterializationPlan(
        plan->name, plan->source_live_form, plan->materialization_boundary,
        plan->materialization_boundary_index, plan->target_buffer,
        plan->host_buffer, plan->target_kernel, plan->bridge_kind,
        plan->materialization_kind, plan->materialization_protocol,
        plan->publication_protocol, cb_plan_indices,
        plan->required_sync_plan_indices, plan->produced_live_form));
  }
  return remapped;
}

Array<TTExecutionPlan> BuildExecutionPlans(const SpatialPlan &plan,
                                           const Array<TTKernel> &kernels) {
  Array<ffi::String> kernel_names;
  for (const TTKernel &kernel : kernels) {
    kernel_names.push_back(kernel->name);
  }
  std::unordered_set<int> seen;
  Array<Integer> phase_indices;
  for (const PhasePlan &phase : plan->phase_plans) {
    const int phase_index = static_cast<int>(phase->phase_index);
    if (seen.insert(phase_index).second) {
      phase_indices.push_back(Integer(phase_index));
    }
  }
  if (phase_indices.empty()) {
    phase_indices.push_back(Integer(0));
  }
  Array<TTExecutionPlan> execution_plans;
  execution_plans.push_back(
      TTExecutionPlan(String("main_execution"), kernel_names, phase_indices));
  return execution_plans;
}

} // namespace

tvm::transform::Pass PlanTTBlocks() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    std::optional<TTHardwareModel> maybe_hardware_model =
        EnsureBlackholeHardwareModel(&mod);
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTBlocks");
      PlanTTCoreGroups planner;
      tir::PrimFunc planned =
          planner.Transform(func.value(), maybe_hardware_model);
      const Array<TTCoreGroup> core_groups = BuildCoreGroups(planner);
      TTProgramSlices slices =
          GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.mesh_plans = BuildUnitMeshPlans();
      slices.block_plans = BuildBlockPlans(spatial_plan, core_groups);
      slices.core_groups = core_groups;
      slices.resource_demands =
          BuildTileComputeResourceDemands(func.value(), slices);
      RefreshResourcePlanningSlices(&slices, maybe_hardware_model);
      planned = WithTTProgramAttr(std::move(planned),
                                  PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTBlocks", {});
}

tvm::transform::Pass PlanTTCompute() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTCompute");
      RequireTTMetalBuiltinSelection(func.value(), "PlanTTCompute");
      PlanTTKernelABI planner;
      tir::PrimFunc planned = planner.Transform(func.value());
      const Array<TTKernel> kernels = planner.GetTTKernels();
      TTProgramSlices slices =
          GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.kernel_plans = BuildKernelPlans(kernels);
      slices.kernels = kernels;
      slices.cb_plans = planner.GetStagedCBPlans();
      slices.abi_plans = planner.GetTTABIPlans();
      slices.live_form_plans = planner.GetTTLiveFormPlans();
      slices.materialization_plans = planner.GetTTMaterializationPlans();
      slices.consumer_binding_plans = planner.GetTTConsumerBindingPlans();
      slices.compute_op_plans = AttachComputeOpKernelPlanIndices(
          planner.GetTTComputeOpPlans(), slices.kernel_plans);
      if (slices.resource_demands.empty()) {
        slices.resource_demands =
            BuildTileComputeResourceDemands(func.value(), slices);
      }
      RefreshResourcePlanningSlices(&slices);
      planned = WithTTProgramAttr(std::move(planned),
                                  PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTCompute", {});
}

tvm::transform::Pass PlanTTTransport() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTTransport");
      PlanTTCBAlloc planner;
      tir::PrimFunc planned = planner.Transform(func.value());
      TTProgramSlices slices =
          GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.cb_plans = BuildCBPlans(planner.GetCBConfigs());
      slices.materialization_plans = RemapMaterializationCBRequirementIndices(
          slices.materialization_plans, slices.cb_plans);
      slices.transport_plans = BuildTransportPlans(spatial_plan);
      if (slices.resource_demands.empty()) {
        slices.resource_demands =
            BuildTileComputeResourceDemands(func.value(), slices);
      }
      RefreshResourcePlanningSlices(&slices);
      planned = WithTTProgramAttr(std::move(planned),
                                  PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTTransport", {});
}

tvm::transform::Pass PlanTTSync() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTSync");
      const Array<TTComputeSyncPlan> compute_sync_plans =
          BuildComputeSyncPlans(spatial_plan);
      tir::PrimFunc planned = func.value();
      TTProgramSlices slices =
          GetOrCreateTTProgramSlices(planned, gvar, spatial_plan);
      slices.sync_plans = BuildSyncPlans(compute_sync_plans);
      slices.compute_sync_plans = compute_sync_plans;
      slices.semaphore_plans = BuildSemaphorePlans(func.value());
      if (slices.resource_demands.empty()) {
        slices.resource_demands =
            BuildTileComputeResourceDemands(func.value(), slices);
      }
      RefreshResourcePlanningSlices(&slices);
      planned = WithTTProgramAttr(std::move(planned),
                                  PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTSync", {});
}

tvm::transform::Pass PlanTTABI() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    std::optional<TTHardwareModel> maybe_hardware_model =
        EnsureBlackholeHardwareModel(&mod);
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTABI");
      const TTProgram staged = RequireStagedTTProgram(
          func.value(), "PlanTTABI", "Run PlanTTCompute before PlanTTABI");
      TTProgramSlices slices = UnpackTTProgram(staged);
      ICHECK(!slices.abi_plans.empty())
          << "PlanTTABI requires TTABIPlan owner truth; Run PlanTTCompute "
             "before PlanTTABI";
      slices.dst_layout_plans = BuildDstLayoutPlans(slices.abi_plans);
      slices.buffer_distribution_plans =
          BuildBufferDistributionPlans(spatial_plan, slices, func.value());
      slices.tensor_memory_config_plans =
          BuildTensorMemoryConfigPlans(spatial_plan, slices.buffer_distribution_plans);
      slices.op_sharding_contracts = BuildOpShardingContracts(
          slices.compute_op_plans, slices.tensor_memory_config_plans);
      slices.placement_resolution_plans = BuildPlacementResolutionPlans(
          slices.op_sharding_contracts, slices.tensor_memory_config_plans);
      slices.reshard_plans =
          BuildReshardPlans(slices.buffer_distribution_plans,
                            slices.tensor_memory_config_plans,
                            slices.materialization_plans);
      RefreshResourcePlanningSlices(&slices, maybe_hardware_model);
      tir::PrimFunc planned =
          WithTTProgramAttr(func.value(), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTABI", {});
}

tvm::transform::Pass PlanTTExecution() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    auto maybe_hardware_model = EnsureBlackholeHardwareModel(&mod);
    ICHECK(maybe_hardware_model) << "PlanTTExecution requires blackhole target";
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const SpatialPlan spatial_plan =
          RequireValidatedSpatialPlan(func.value(), "PlanTTExecution");
      const TTProgram staged =
          RequireStagedTTProgram(func.value(), "PlanTTExecution",
                                 "Run PlanTTCompute before PlanTTExecution");
      TTProgramSlices slices = UnpackTTProgram(staged);
      const Array<TTKernel> &kernels = slices.kernels;
      ICHECK(!kernels.empty())
          << "PlanTTExecution requires TTKernel owner truth; Run PlanTTCompute "
             "before PlanTTExecution";
      slices.execution_plans = BuildExecutionPlans(spatial_plan, kernels);
      RefreshResourcePlanningSlices(&slices, maybe_hardware_model);
      tir::PrimFunc planned =
          WithTTProgramAttr(func.value(), PackTTProgram(std::move(slices)));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.PlanTTExecution", {});
}

tvm::transform::Pass BuildTTProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      const TTProgram staged = RequireStagedTTProgram(
          func.value(), "BuildTTProgram",
          "Run PlanTTBlocks, PlanTTCompute, PlanTTTransport, PlanTTSync, "
          "PlanTTABI, and PlanTTExecution before BuildTTProgram");
      const TTProgramSlices slices = UnpackTTProgram(staged);

      ICHECK(!slices.mesh_plans.empty())
          << "BuildTTProgram requires TTMeshPlan owner truth";
      ICHECK(!slices.buffer_distribution_plans.empty())
          << "BuildTTProgram requires TTBufferDistributionPlan owner truth";
      ICHECK(!slices.tensor_memory_config_plans.empty())
          << "BuildTTProgram requires TTTensorMemoryConfigPlan owner truth";
      if (!slices.compute_op_plans.empty()) {
        ICHECK(!slices.op_sharding_contracts.empty())
            << "BuildTTProgram requires TTOpShardingContract owner truth";
        ICHECK(!slices.placement_resolution_plans.empty())
            << "BuildTTProgram requires TTPlacementResolutionPlan owner truth";
      }
      ICHECK(!slices.block_plans.empty())
          << "BuildTTProgram requires TTBlockPlan owner truth";
      ICHECK(!slices.kernel_plans.empty())
          << "BuildTTProgram requires TTKernelPlan owner truth";
      ICHECK(!slices.core_groups.empty())
          << "BuildTTProgram requires TTCoreGroup owner truth; run "
             "PlanTTBlocks before BuildTTProgram";
      ICHECK(!slices.abi_plans.empty())
          << "BuildTTProgram requires TTABIPlan owner truth; run PlanTTCompute "
             "and PlanTTABI before BuildTTProgram";
      ICHECK(!slices.execution_plans.empty())
          << "BuildTTProgram requires TTExecutionPlan owner truth; run "
             "PlanTTExecution before BuildTTProgram";
      ICHECK_EQ(slices.block_plans.size(), slices.core_groups.size())
          << "BuildTTProgram requires aligned TTBlockPlan and TTCoreGroup "
             "owner truth";
      ICHECK_EQ(slices.kernel_plans.size(), slices.kernels.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTKernel owner "
             "truth";
      ICHECK_EQ(slices.kernel_plans.size(), slices.abi_plans.size())
          << "BuildTTProgram requires aligned TTKernelPlan and TTABIPlan owner "
             "truth";
      ICHECK_EQ(slices.sync_plans.size(), slices.compute_sync_plans.size())
          << "BuildTTProgram requires aligned TTSyncPlan and TTComputeSyncPlan "
             "owner truth";
      if (!slices.resource_demands.empty()) {
        ICHECK(!slices.resource_pressure_reports.empty())
            << "BuildTTProgram requires ResourcePressureReport for "
               "ResourceDemand owner truth";
      }

      tir::PrimFunc planned = WithTTProgramAttr(func.value(), staged);
      planned = StripTTIntermediateAttrs(std::move(planned));
      updated->Add(gvar, planned, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.BuildTTProgram", {});
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

} // namespace tl
} // namespace tvm
