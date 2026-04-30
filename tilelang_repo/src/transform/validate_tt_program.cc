/*!
 * \file validate_tt_program.cc
 * \brief Validate TTProgram invariants for Phase C cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_tile_compute_covering.h"
#include "common/blackhole_tile_compute_legalizer.h"
#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_plan.h"
#include "common/tt_hardware_model.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

namespace {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

int64_t GetIntOrDefault(const Map<String, Any>& map, const char* key, int64_t default_value = -1) {
  if (auto value = map.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

int64_t RequireInt(const Map<String, Any>& map, const char* key, const std::string& context) {
  auto value = map.Get(String(key));
  ICHECK(value.has_value()) << context << " requires " << key;
  return Downcast<Integer>(value.value())->value;
}

std::string CoreCoordKey(int64_t x, int64_t y) {
  return std::to_string(x) + "," + std::to_string(y);
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

std::optional<TTHardwareModel> GetValidationHardwareModel(const IRModule& mod) {
  if (auto maybe_hardware_model = GetModuleTTHardwareModel(mod)) {
    return maybe_hardware_model.value();
  }
  if (auto maybe_target = FindBlackholeTarget(mod)) {
    return BuildBlackholeTTHardwareModel(maybe_target.value());
  }
  return std::nullopt;
}

void ValidatePositiveIntegerArray(const Array<Integer>& values, const std::string& context) {
  ICHECK(!values.empty()) << context << " requires non-empty shape";
  for (const Integer& value : values) {
    ICHECK_GT(value->value, 0) << context << " requires positive dimensions";
  }
}

void ValidateMeshPlan(const TTMeshPlan& mesh_plan) {
  ICHECK(!mesh_plan->name.empty()) << "TTMeshPlan requires name";
  ICHECK(!mesh_plan->mesh_kind.empty()) << "TTMeshPlan requires mesh_kind";
  ValidatePositiveIntegerArray(mesh_plan->mesh_shape, "TTMeshPlan mesh_shape");
  ICHECK_EQ(mesh_plan->device_range_start.size(), mesh_plan->mesh_shape.size())
      << "TTMeshPlan device_range_start rank must match mesh_shape";
  ICHECK_EQ(mesh_plan->device_range_shape.size(), mesh_plan->mesh_shape.size())
      << "TTMeshPlan device_range_shape rank must match mesh_shape";
  for (int i = 0; i < mesh_plan->mesh_shape.size(); ++i) {
    ICHECK_GE(mesh_plan->device_range_start[i]->value, 0)
        << "TTMeshPlan device_range_start requires non-negative coordinates";
    ICHECK_GT(mesh_plan->device_range_shape[i]->value, 0)
        << "TTMeshPlan device_range_shape requires positive dimensions";
    ICHECK_LE(mesh_plan->device_range_start[i]->value + mesh_plan->device_range_shape[i]->value,
              mesh_plan->mesh_shape[i]->value)
        << "TTMeshPlan device range must fit in mesh_shape";
  }
}

void ValidateBufferDistributionPlan(
    const TTBufferDistributionPlan& plan,
    const std::unordered_map<std::string, int64_t>& mesh_index_by_name) {
  ICHECK(!plan->name.empty()) << "TTBufferDistributionPlan requires name";
  ICHECK(!plan->buffer.empty()) << "TTBufferDistributionPlan requires buffer";
  ICHECK(!plan->mesh_plan.empty()) << "TTBufferDistributionPlan requires mesh_plan";
  ICHECK_GE(plan->mesh_plan_index, 0)
      << "TTBufferDistributionPlan requires mesh_plan_index";
  auto mesh_it = mesh_index_by_name.find(static_cast<std::string>(plan->mesh_plan));
  ICHECK(mesh_it != mesh_index_by_name.end())
      << "TTBufferDistributionPlan references unknown mesh_plan " << plan->mesh_plan;
  ICHECK_EQ(plan->mesh_plan_index, mesh_it->second)
      << "TTBufferDistributionPlan mesh_plan_index must match mesh_plan";
  ICHECK(!plan->distribution_kind.empty())
      << "TTBufferDistributionPlan requires distribution_kind";
  const std::string distribution_kind = plan->distribution_kind;
  ICHECK(distribution_kind == "replicated" || distribution_kind == "sharded")
      << "TTBufferDistributionPlan distribution_kind must be replicated or sharded";
  ICHECK(!plan->layout.empty()) << "TTBufferDistributionPlan requires layout";
  ICHECK(!plan->memory_space.empty()) << "TTBufferDistributionPlan requires memory_space";
  const std::string memory_space = plan->memory_space;
  ICHECK(memory_space == "DRAM" || memory_space == "L1")
      << "TTBufferDistributionPlan memory_space must be DRAM or L1";
  ICHECK_GE(plan->page_size_bytes, 0)
      << "TTBufferDistributionPlan requires non-negative page_size_bytes";
  if (distribution_kind == "sharded") {
    ValidatePositiveIntegerArray(plan->shard_shape, "TTBufferDistributionPlan shard_shape");
  }
  ICHECK(!plan->shard_orientation.empty())
      << "TTBufferDistributionPlan requires shard_orientation";
  ICHECK(!plan->host_visibility.empty())
      << "TTBufferDistributionPlan requires host_visibility";
  if (!plan->logical_shape.empty()) {
    ICHECK(!plan->local_shape.empty())
        << "TTBufferDistributionPlan logical_shape requires local_shape";
    ICHECK(plan->thread_extent.defined())
        << "TTBufferDistributionPlan logical_shape requires thread_extent";
    ICHECK(plan->replicate_extent.defined())
        << "TTBufferDistributionPlan logical_shape requires replicate_extent";
    ICHECK(!plan->inverse_logical_index_exprs.empty())
        << "TTBufferDistributionPlan logical_shape requires inverse layout expressions";
    ICHECK(!plan->inverse_logical_index_vars.empty())
        << "TTBufferDistributionPlan logical_shape requires inverse layout variables";
  }
}

void ValidateCoreGroup(const TTCoreGroup& core_group,
                       const std::optional<TTHardwareModel>& maybe_hardware_model) {
  ICHECK_GT(core_group->logical_grid_x, 0) << "TTCoreGroup requires positive logical_grid_x";
  ICHECK_GT(core_group->logical_grid_y, 0) << "TTCoreGroup requires positive logical_grid_y";
  ICHECK(!core_group->physical_cores.empty()) << "TTCoreGroup requires physical_cores";
  ICHECK(!core_group->work_packets.empty()) << "TTCoreGroup requires work_packets";
  int64_t hardware_grid_x = 0;
  int64_t hardware_grid_y = 0;
  int64_t functional_worker_count = 0;
  if (maybe_hardware_model) {
    const TTHardwareModel& hardware_model = maybe_hardware_model.value();
    hardware_grid_x = hardware_model->logical_worker_grid_x;
    hardware_grid_y = hardware_model->logical_worker_grid_y;
    functional_worker_count = hardware_model->functional_worker_count;
    ICHECK_GT(hardware_grid_x, 0)
        << "TTCoreGroup validation requires positive TTHardwareModel logical_worker_grid_x";
    ICHECK_GT(hardware_grid_y, 0)
        << "TTCoreGroup validation requires positive TTHardwareModel logical_worker_grid_y";
    ICHECK_GT(functional_worker_count, 0)
        << "TTCoreGroup validation requires positive TTHardwareModel functional_worker_count";
    ICHECK_LE(static_cast<int64_t>(core_group->physical_cores.size()),
              functional_worker_count)
        << "TTCoreGroup physical_cores exceed hardware functional worker count";
  }

  std::unordered_set<std::string> physical_core_coords;
  for (const Any& item : core_group->physical_cores) {
    Map<String, Any> core = AsMap(item);
    ICHECK(!core.empty()) << "TTCoreGroup physical_core must be a map";
    const int64_t core_x = RequireInt(core, "core_x", "TTCoreGroup physical_core");
    const int64_t core_y = RequireInt(core, "core_y", "TTCoreGroup physical_core");
    ICHECK_GE(core_x, 0) << "TTCoreGroup physical_core requires non-negative core_x";
    ICHECK_GE(core_y, 0) << "TTCoreGroup physical_core requires non-negative core_y";
    if (maybe_hardware_model) {
      ICHECK_LT(core_x, hardware_grid_x)
          << "TTCoreGroup physical_core outside hardware logical worker grid";
      ICHECK_LT(core_y, hardware_grid_y)
          << "TTCoreGroup physical_core outside hardware logical worker grid";
    }
    ICHECK(physical_core_coords.insert(CoreCoordKey(core_x, core_y)).second)
        << "TTCoreGroup duplicate physical_core coordinate";
  }

  for (const Any& item : core_group->work_packets) {
    Map<String, Any> packet = AsMap(item);
    ICHECK(!packet.empty()) << "TTCoreGroup work_packet must be a map";
    const int64_t core_x = RequireInt(packet, "core_x", "TTCoreGroup work_packet");
    const int64_t core_y = RequireInt(packet, "core_y", "TTCoreGroup work_packet");
    ICHECK(physical_core_coords.count(CoreCoordKey(core_x, core_y)))
        << "TTCoreGroup work_packet references core outside physical_cores";
    ICHECK_GE(GetIntOrDefault(packet, "work_offset", -1), 0)
        << "TTCoreGroup work_packet requires non-negative work_offset";
    ICHECK_GT(GetIntOrDefault(packet, "work_count", 0), 0)
        << "TTCoreGroup work_packet requires positive work_count";
  }
}

void ValidateBlockPlan(const TTBlockPlan& block_plan) {
  ICHECK(!block_plan->name.empty()) << "TTBlockPlan requires name";
  ICHECK(!block_plan->placement_kind.empty()) << "TTBlockPlan requires placement_kind";
  ICHECK(!block_plan->task_indices.empty()) << "TTBlockPlan requires task_indices";
}

void ValidateKernelPlan(const TTKernelPlan& kernel_plan, int64_t abi_plan_count,
                        int64_t block_plan_count) {
  ICHECK(!kernel_plan->name.empty()) << "TTKernelPlan requires name";
  ICHECK(!kernel_plan->kind.empty()) << "TTKernelPlan requires kind";
  ICHECK(!kernel_plan->core_type.empty()) << "TTKernelPlan requires core_type";
  ICHECK_GE(kernel_plan->abi_plan_index, 0) << "TTKernelPlan requires abi_plan_index";
  ICHECK_LT(kernel_plan->abi_plan_index, abi_plan_count)
      << "TTKernelPlan abi_plan_index out of bounds";
  ICHECK_GE(kernel_plan->block_plan_index, 0) << "TTKernelPlan requires block_plan_index";
  ICHECK_LT(kernel_plan->block_plan_index, block_plan_count)
      << "TTKernelPlan block_plan_index out of bounds";
}

void ValidateComputeOperandBindingPlan(const TTComputeOperandBindingPlan& binding) {
  ICHECK(!binding->role.empty()) << "TTComputeOperandBindingPlan requires role";
  ICHECK(!binding->buffer.empty()) << "TTComputeOperandBindingPlan requires buffer";
  const std::string role = binding->role;
  ICHECK(role == "a" || role == "b" || role == "c" || role == "input" ||
         role == "lhs" || role == "rhs" || role == "output" || role == "scaler")
      << "TTComputeOperandBindingPlan unsupported role " << binding->role;
  if (!binding->transform_kind.empty()) {
    const std::string transform_kind = binding->transform_kind;
    ICHECK(transform_kind == "identity" || transform_kind == "transpose" ||
           transform_kind == "broadcast" || transform_kind == "cast")
        << "TTComputeOperandBindingPlan unsupported transform_kind "
        << binding->transform_kind;
  }
}

BlackholeTileComputeCoveringDecision RequireSelectedBlackholeTileComputeCoveringForPlan(
    const TTComputeOpPlan& plan, const std::vector<std::string>& operand_roles) {
  const std::string operation_name = plan->operation_name;
  const BlackholeTileComputeCoveringDecision covering =
      SelectBlackholeTileComputeCovering(operation_name);
  ICHECK(covering.selected)
      << "TileCompute covering rejected operation " << operation_name
      << ": " << covering.reject_reason;
  ICHECK_EQ(static_cast<std::string>(plan->kind), covering.result_kind)
      << "TileCompute covering selected result kind " << covering.result_kind
      << " for " << operation_name << ", but TTComputeOpPlan recorded "
      << plan->kind;
  RequireLegalBlackholeTileComputeSelection(covering.result_kind,
                                            covering.operation_name,
                                            operand_roles);
  return covering;
}

std::optional<BlackholeTileComputeCoveringDecision>
FindSelectedBlackholeTileComputeCoveringForSourceEmitter(
    const std::string& source_emitter) {
  for (const BlackholeTileComputePattern& pattern :
       GetBlackholeTileComputePatterns()) {
    if (!pattern.source_emitter ||
        source_emitter != ToString(*pattern.source_emitter)) {
      continue;
    }
    BlackholeTileComputeCoveringDecision covering =
        SelectBlackholeTileComputeCovering(ToString(pattern.operation));
    ICHECK(covering.selected)
        << "TileCompute covering rejected source emitter " << source_emitter
        << " operation " << ToString(pattern.operation)
        << ": " << covering.reject_reason;
    return covering;
  }
  return std::nullopt;
}

void ValidateComputeOpPlan(const TTComputeOpPlan& plan, int64_t kernel_plan_count,
                           const std::unordered_set<std::string>& kernel_names) {
  ICHECK(!plan->name.empty()) << "TTComputeOpPlan requires name";
  ICHECK(!plan->kernel_name.empty()) << "TTComputeOpPlan requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(plan->kernel_name)))
      << "TTComputeOpPlan references unknown kernel " << plan->kernel_name;
  ICHECK_GE(plan->kernel_plan_index, 0) << "TTComputeOpPlan requires kernel_plan_index";
  ICHECK_LT(plan->kernel_plan_index, kernel_plan_count)
      << "TTComputeOpPlan kernel_plan_index out of bounds";
  ICHECK(!plan->kind.empty()) << "TTComputeOpPlan requires kind";
  ICHECK(!plan->operation_name.empty()) << "TTComputeOpPlan requires operation_name";
  const std::string kind = plan->kind;
  ICHECK(kind == "gemm" || kind == "binary" || kind == "unary" || kind == "reduce" ||
         kind == "sfpu" || kind == "pack" || kind == "copy")
      << "TTComputeOpPlan unsupported kind " << plan->kind;
  ICHECK(!plan->operand_bindings.empty())
      << "TTComputeOpPlan requires operand_bindings";
  std::unordered_set<std::string> roles;
  std::vector<std::string> operand_roles;
  for (const TTComputeOperandBindingPlan& binding : plan->operand_bindings) {
    ValidateComputeOperandBindingPlan(binding);
    operand_roles.push_back(static_cast<std::string>(binding->role));
    ICHECK(roles.insert(static_cast<std::string>(binding->role)).second)
        << "TTComputeOpPlan duplicate operand role " << binding->role;
  }
  if (kind == "gemm") {
    for (const char* role : {"a", "b", "c"}) {
      ICHECK(roles.count(role)) << "TTComputeOpPlan GEMM requires operand role " << role;
    }
    for (const TTComputeOperandBindingPlan& binding : plan->operand_bindings) {
      ICHECK(!binding->host_buffer.empty())
          << "TTComputeOpPlan GEMM operand role " << binding->role
          << " requires host_buffer";
    }
    ICHECK_EQ(plan->problem_shape_axes.size(), 3)
        << "TTComputeOpPlan GEMM requires M/N/K problem_shape_axes";
    ICHECK_EQ(plan->problem_shape.size(), 3)
        << "TTComputeOpPlan GEMM requires M/N/K problem_shape";
    ICHECK_EQ(plan->tile_shape.size(), 3) << "TTComputeOpPlan GEMM requires tile_shape";
    ICHECK_EQ(plan->block_shape.size(), 3) << "TTComputeOpPlan GEMM requires block_shape";
    ICHECK_EQ(plan->subblock_shape.size(), 2)
        << "TTComputeOpPlan GEMM requires subblock_shape";
    ValidatePositiveIntegerArray(plan->problem_shape, "TTComputeOpPlan GEMM problem_shape");
    ValidatePositiveIntegerArray(plan->tile_shape, "TTComputeOpPlan GEMM tile_shape");
    ValidatePositiveIntegerArray(plan->block_shape, "TTComputeOpPlan GEMM block_shape");
    ValidatePositiveIntegerArray(plan->subblock_shape, "TTComputeOpPlan GEMM subblock_shape");
    ICHECK(!plan->accumulator_dtype.empty())
        << "TTComputeOpPlan GEMM requires accumulator_dtype";
  }
  RequireSelectedBlackholeTileComputeCoveringForPlan(plan, operand_roles);
  if (plan->tile_compute_dag_node_id >= 0) {
    ICHECK(!plan->tile_compute_source_emitter.empty())
        << "DAG-driven TTComputeOpPlan requires tile_compute_source_emitter";
    const std::string source_emitter = plan->tile_compute_source_emitter;
    const std::optional<BlackholeTileComputeCoveringDecision> source_covering =
        FindSelectedBlackholeTileComputeCoveringForSourceEmitter(source_emitter);
    ICHECK(source_covering)
        << "DAG-driven TTComputeOpPlan references unknown tile_compute_source_emitter "
        << source_emitter;
    ICHECK_EQ(plan->tile_compute_materialization_policy,
              source_covering->materialization_policy)
        << "DAG-driven TTComputeOpPlan materialization policy must match DAG source "
           "covering";
    ICHECK_GE(plan->tile_compute_fanout_use_count, 0)
        << "DAG-driven TTComputeOpPlan requires non-negative fanout use count";
    if (plan->tile_compute_fanout_use_count > 1) {
      const std::string fanout_policy = plan->tile_compute_fanout_policy;
      ICHECK(fanout_policy == "share_live_value" ||
             fanout_policy == "materialize_before_cross_event_use")
          << "DAG-driven TTComputeOpPlan unsupported fanout policy "
          << plan->tile_compute_fanout_policy;
    }
  }
}

void ValidateTileComputeFanoutDemand(
    const TTTileComputeFanoutDemand& demand,
    const std::unordered_set<std::string>& kernel_names) {
  ICHECK(!demand->name.empty()) << "TTTileComputeFanoutDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTTileComputeFanoutDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTTileComputeFanoutDemand references unknown kernel "
      << demand->kernel_name;
  ICHECK_GE(demand->producer_node, 0)
      << "TTTileComputeFanoutDemand requires producer_node";
  ICHECK(!demand->producer_operation.empty())
      << "TTTileComputeFanoutDemand requires producer_operation";
  ICHECK_GT(demand->use_count, 1)
      << "TTTileComputeFanoutDemand requires fanout use_count > 1";
  ICHECK_EQ(demand->consumer_nodes.size(),
            static_cast<size_t>(demand->use_count))
      << "TTTileComputeFanoutDemand consumer_nodes must match use_count";
  const std::string policy = demand->policy;
  ICHECK(policy == "share_live_value" ||
         policy == "materialize_before_cross_event_use")
      << "TTTileComputeFanoutDemand unsupported policy " << demand->policy;
  ICHECK(!demand->evidence.empty())
      << "TTTileComputeFanoutDemand requires evidence";
}

void ValidateTileComputeMaterializationDemand(
    const TTTileComputeMaterializationDemand& demand,
    const std::unordered_set<std::string>& kernel_names) {
  ICHECK(!demand->name.empty())
      << "TTTileComputeMaterializationDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTTileComputeMaterializationDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTTileComputeMaterializationDemand references unknown kernel "
      << demand->kernel_name;
  ICHECK_GE(demand->node_id, 0)
      << "TTTileComputeMaterializationDemand requires node_id";
  ICHECK(!demand->operation_name.empty())
      << "TTTileComputeMaterializationDemand requires operation_name";
  ICHECK(!demand->pattern_name.empty())
      << "TTTileComputeMaterializationDemand requires pattern_name";
  ICHECK(!demand->policy.empty())
      << "TTTileComputeMaterializationDemand requires policy";
  ICHECK(demand->policy != "none")
      << "TTTileComputeMaterializationDemand cannot record policy=none";
  ICHECK(!demand->evidence.empty())
      << "TTTileComputeMaterializationDemand requires evidence";
}

void ValidateResourceDemand(
    const TTResourceDemand& demand,
    const std::unordered_set<std::string>& kernel_names,
    int64_t core_group_count) {
  ICHECK(!demand->name.empty()) << "TTResourceDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTResourceDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTResourceDemand references unknown kernel " << demand->kernel_name;
  ICHECK(!demand->core_group.empty())
      << "TTResourceDemand requires core_group";
  ICHECK_GE(demand->core_group_index, 0)
      << "TTResourceDemand requires core_group_index";
  ICHECK_LT(demand->core_group_index, core_group_count)
      << "TTResourceDemand core_group_index out of bounds";
  ICHECK_GE(demand->cb_requirement_count, 0)
      << "TTResourceDemand requires non-negative cb_requirement_count";
  ICHECK_GE(demand->cb_l1_bytes, 0)
      << "TTResourceDemand requires non-negative cb_l1_bytes";
  ICHECK_GE(demand->semaphore_count, 0)
      << "TTResourceDemand requires non-negative semaphore_count";
  ICHECK_GE(demand->communication_edge_count, 0)
      << "TTResourceDemand requires non-negative communication_edge_count";
  ICHECK(!demand->tile_compute_fanout_demands.empty() ||
         !demand->tile_compute_materialization_demands.empty() ||
         !demand->tile_compute_unsupported_reasons.empty())
      << "TTResourceDemand requires tile-compute demand evidence";
  for (const TTTileComputeFanoutDemand& fanout :
       demand->tile_compute_fanout_demands) {
    ValidateTileComputeFanoutDemand(fanout, kernel_names);
  }
  for (const TTTileComputeMaterializationDemand& materialization :
       demand->tile_compute_materialization_demands) {
    ValidateTileComputeMaterializationDemand(materialization, kernel_names);
  }
  for (const String& reason : demand->tile_compute_unsupported_reasons) {
    ICHECK(!reason.empty())
        << "TTResourceDemand tile_compute_unsupported_reasons cannot be empty";
  }
}

void ValidateResourcePressureReport(
    const TTResourcePressureReport& report,
    const std::unordered_set<std::string>& kernel_names,
    int64_t core_group_count,
    const std::optional<TTHardwareModel>& maybe_hardware_model) {
  ICHECK(!report->name.empty()) << "TTResourcePressureReport requires name";
  ICHECK(!report->kernel_name.empty())
      << "TTResourcePressureReport requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(report->kernel_name)))
      << "TTResourcePressureReport references unknown kernel "
      << report->kernel_name;
  ICHECK(!report->core_group.empty())
      << "TTResourcePressureReport requires core_group";
  ICHECK_GE(report->core_group_index, 0)
      << "TTResourcePressureReport requires core_group_index";
  ICHECK_LT(report->core_group_index, core_group_count)
      << "TTResourcePressureReport core_group_index out of bounds";
  for (const TTTileComputeMaterializationDemand& materialization :
       report->required_materializations) {
    ValidateTileComputeMaterializationDemand(materialization, kernel_names);
  }
  ICHECK_GE(report->per_core_cb_id_pressure, 0)
      << "TTResourcePressureReport requires non-negative per_core_cb_id_pressure";
  ICHECK_GE(report->per_core_cb_l1_bytes, 0)
      << "TTResourcePressureReport requires non-negative per_core_cb_l1_bytes";
  ICHECK_GE(report->per_core_l1_buffer_bytes, 0)
      << "TTResourcePressureReport requires non-negative per_core_l1_buffer_bytes";
  ICHECK_GE(report->max_simultaneous_l1_bytes, 0)
      << "TTResourcePressureReport requires non-negative max_simultaneous_l1_bytes";
  ICHECK_GT(report->cb_id_limit, 0)
      << "TTResourcePressureReport requires positive cb_id_limit";
  ICHECK_GT(report->worker_l1_budget_bytes, 0)
      << "TTResourcePressureReport requires positive worker_l1_budget_bytes";
  ICHECK_GT(report->l1_alignment_bytes, 0)
      << "TTResourcePressureReport requires positive l1_alignment_bytes";
  ICHECK_GE(report->per_core_cb_l1_aligned_bytes, report->per_core_cb_l1_bytes)
      << "TTResourcePressureReport aligned CB L1 bytes must cover raw CB L1 bytes";
  ICHECK_EQ(report->l1_alignment_waste_bytes,
            report->per_core_cb_l1_aligned_bytes - report->per_core_cb_l1_bytes)
      << "TTResourcePressureReport l1_alignment_waste_bytes must equal aligned - raw CB bytes";
  ICHECK_LE(report->per_core_cb_id_pressure, report->cb_id_limit)
      << "ResourcePressureReport CB id pressure exceeds hardware limit: required "
      << report->per_core_cb_id_pressure << ", limit " << report->cb_id_limit;
  ICHECK_LE(report->max_simultaneous_l1_bytes, report->worker_l1_budget_bytes)
      << "ResourcePressureReport L1 pressure exceeds worker budget: required "
      << report->max_simultaneous_l1_bytes << ", budget "
      << report->worker_l1_budget_bytes;
  ICHECK_EQ(report->max_simultaneous_l1_bytes,
            report->per_core_cb_l1_aligned_bytes + report->per_core_l1_buffer_bytes)
      << "TTResourcePressureReport max_simultaneous_l1_bytes must equal aligned CB bytes "
         "plus L1 buffer bytes";
  if (maybe_hardware_model) {
    const TTHardwareModel& hardware_model = maybe_hardware_model.value();
    ICHECK_EQ(report->cb_id_limit, hardware_model->max_cb_count)
        << "TTResourcePressureReport cb_id_limit must match TTHardwareModel";
    ICHECK_EQ(report->worker_l1_budget_bytes, hardware_model->worker_l1_size)
        << "TTResourcePressureReport worker_l1_budget_bytes must match TTHardwareModel";
    ICHECK_EQ(report->l1_alignment_bytes,
              hardware_model->l1_allocation_alignment_bytes)
        << "TTResourcePressureReport l1_alignment_bytes must match TTHardwareModel";
  }
  ICHECK(report->tile_compute_unsupported_reasons.empty())
      << "ResourcePressureReport unsupported tile compute: "
      << report->tile_compute_unsupported_reasons[0];
  ICHECK(report->unsupported_reasons.empty())
      << "ResourcePressureReport unsupported: "
      << report->unsupported_reasons[0];
  ICHECK(!report->core_grid_requirement.empty())
      << "TTResourcePressureReport requires core_grid_requirement";
  ICHECK(!report->dram_view_requirement.empty())
      << "TTResourcePressureReport requires dram_view_requirement";
}

void ValidateSyncPlan(const TTSyncPlan& sync_plan) {
  ICHECK(!sync_plan->name.empty()) << "TTSyncPlan requires name";
  ICHECK(!sync_plan->kind.empty()) << "TTSyncPlan requires kind";
  ICHECK_GE(sync_plan->source_task_index, 0) << "TTSyncPlan requires source_task_index";
  ICHECK_GE(sync_plan->target_task_index, 0) << "TTSyncPlan requires target_task_index";
  ICHECK(!sync_plan->ordering_kind.empty()) << "TTSyncPlan requires ordering_kind";
  ICHECK(!sync_plan->completion_kind.empty()) << "TTSyncPlan requires completion_kind";
}

void ValidateCBPlan(const TTCBPlan& cb_plan) {
  ICHECK(!cb_plan->name.empty()) << "TTCBPlan requires name";
  ICHECK(!cb_plan->resource_class.empty()) << "TTCBPlan requires resource_class";
  ICHECK_GT(cb_plan->num_pages, 0) << "TTCBPlan requires positive num_pages";
  ICHECK_GT(cb_plan->page_size_bytes, 0) << "TTCBPlan requires positive page_size_bytes";
  ICHECK(!cb_plan->data_format.empty()) << "TTCBPlan requires data_format";
  ICHECK_GE(cb_plan->initial_reserve_pages, 0)
      << "TTCBPlan requires non-negative initial_reserve_pages";
  ICHECK(!cb_plan->flow_class.empty()) << "TTCBPlan requires flow_class";
  const std::string flow_class = cb_plan->flow_class;
  ICHECK(flow_class == "state" || flow_class == "stream" || flow_class == "republish")
      << "TTCBPlan flow_class must be one of state/stream/republish";
  ICHECK_GE(cb_plan->publish_pages_per_event, 0)
      << "TTCBPlan requires non-negative publish_pages_per_event";
  ICHECK_GE(cb_plan->consume_pages_per_event, 0)
      << "TTCBPlan requires non-negative consume_pages_per_event";
  if (flow_class == "republish") {
    ICHECK_GT(cb_plan->publish_pages_per_event, 0)
        << "republish TTCBPlan requires positive publish_pages_per_event";
    ICHECK_GT(cb_plan->consume_pages_per_event, 0)
        << "republish TTCBPlan requires positive consume_pages_per_event";
    ICHECK_LE(cb_plan->publish_pages_per_event, cb_plan->num_pages)
        << "republish TTCBPlan publish_pages_per_event must fit in num_pages";
    ICHECK_LE(cb_plan->consume_pages_per_event, cb_plan->num_pages)
        << "republish TTCBPlan consume_pages_per_event must fit in num_pages";
  }
  ICHECK_GE(cb_plan->lifetime_begin, 0) << "TTCBPlan requires non-negative lifetime_begin";
  ICHECK_GE(cb_plan->lifetime_end, cb_plan->lifetime_begin)
      << "TTCBPlan requires lifetime_end >= lifetime_begin";
}

void ValidateAccessor(const TTAccessorSpec& accessor) {
  ICHECK(!accessor->buffer.empty()) << "TTABIPlan accessor requires buffer";
  ICHECK_GT(accessor->compile_time_arg_count, 0)
      << "TTABIPlan accessor requires compile_time_arg_count";
  ICHECK(!accessor->layout.empty()) << "TTABIPlan accessor requires layout";
  ICHECK(!accessor->memory_space.empty()) << "TTABIPlan accessor requires memory_space";
}

void ValidateCompileTimeArgSpec(const TTCompileTimeArgSpec& spec) {
  ICHECK(!spec->kind.empty()) << "TTABIPlan compile_time_arg_spec requires kind";
  ICHECK(!spec->dtype.empty()) << "TTABIPlan compile_time_arg_spec requires dtype";
  ICHECK_GE(spec->offset, 0) << "TTABIPlan compile_time_arg_spec requires offset";
  ICHECK_GE(spec->count, 0) << "TTABIPlan compile_time_arg_spec requires count";
}

void ValidateKernelLeafFields(const TTKernel& kernel) {
  ICHECK(kernel->launch_spec.defined()) << "TTKernel requires launch_spec";
  ICHECK(!kernel->launch_spec->core_type.empty()) << "TTKernel launch_spec requires core_type";

  if (kernel->kind == "compute" || kernel->core_type == "trisc") {
    ICHECK(kernel->compute_config.defined() && !kernel->compute_config->math_fidelity.empty())
        << "TTKernel compute kernels require compute_config";
    ICHECK_GT(kernel->compute_config->k_pack, 0)
        << "TTKernel compute_config requires positive k_pack";
  }
}

void ValidateLiveFormPlans(const TTProgram& program,
                           std::unordered_set<std::string>* live_form_names) {
  for (const TTLiveFormPlan& plan : program->live_form_plans) {
    ICHECK(!plan->name.empty()) << "TTLiveFormPlan requires name";
    ICHECK(!plan->logical_value.empty()) << "TTLiveFormPlan requires logical_value";
    ICHECK(!plan->spatial_live_value.empty())
        << "TTLiveFormPlan requires spatial_live_value";
    ICHECK_GE(plan->spatial_live_value_index, 0)
        << "TTLiveFormPlan requires spatial_live_value_index";
    ICHECK(!plan->producer_kernel.empty()) << "TTLiveFormPlan requires producer_kernel";
    ICHECK(!plan->physical_form.empty()) << "TTLiveFormPlan requires physical_form";
    ICHECK(!plan->execution_topology.empty()) << "TTLiveFormPlan requires execution_topology";
    ICHECK_GT(plan->physical_local_extent, 0)
        << "TTLiveFormPlan requires positive physical_local_extent";
    ICHECK_GT(plan->logical_element_count, 0)
        << "TTLiveFormPlan requires positive logical_element_count";
    ICHECK(live_form_names->insert(plan->name).second)
        << "duplicate TTLiveFormPlan name " << plan->name;
  }
}

void ValidateMaterializationPlans(const TTProgram& program,
                                  const std::unordered_set<std::string>& live_form_names,
                                  int64_t cb_plan_count) {
  for (const TTMaterializationPlan& plan : program->materialization_plans) {
    ICHECK(!plan->name.empty()) << "TTMaterializationPlan requires name";
    ICHECK(!plan->source_live_form.empty())
        << "TTMaterializationPlan requires source_live_form";
    ICHECK(live_form_names.count(plan->source_live_form))
        << "TTMaterializationPlan references unknown source_live_form "
        << plan->source_live_form;
    ICHECK(!plan->materialization_boundary.empty())
        << "TTMaterializationPlan requires materialization_boundary";
    ICHECK_GE(plan->materialization_boundary_index, 0)
        << "TTMaterializationPlan requires materialization_boundary_index";
    ICHECK(!plan->target_buffer.empty()) << "TTMaterializationPlan requires target_buffer";
    ICHECK(!plan->target_kernel.empty()) << "TTMaterializationPlan requires target_kernel";
    ICHECK(!plan->materialization_protocol.empty())
        << "TTMaterializationPlan requires materialization_protocol";
    ICHECK(!plan->publication_protocol.empty())
        << "TTMaterializationPlan requires publication_protocol";
    ICHECK(!plan->produced_live_form.empty())
        << "TTMaterializationPlan requires produced_live_form";
    ICHECK(live_form_names.count(plan->produced_live_form))
        << "TTMaterializationPlan references unknown produced_live_form "
        << plan->produced_live_form;
    if (plan->materialization_protocol == buffer_materialization::kCBRepublish) {
      ICHECK(!plan->required_cb_plan_indices.empty())
          << "TTMaterializationPlan cb_republish requires required_cb_plan_indices";
      ICHECK(plan->publication_protocol == buffer_materialization::kMailboxWritePtr ||
             plan->publication_protocol == buffer_materialization::kPackThreadDirectStore ||
             plan->publication_protocol == buffer_materialization::kPackTile ||
             plan->publication_protocol == buffer_materialization::kTilizeCastFragmentSlice)
          << "TTMaterializationPlan cb_republish has unsupported publication_protocol "
          << plan->publication_protocol;
      if (plan->publication_protocol == buffer_materialization::kPackThreadDirectStore ||
          plan->publication_protocol == buffer_materialization::kPackTile) {
        ICHECK(!plan->host_buffer.empty()) << "TTMaterializationPlan requires host_buffer";
      }
    }
    for (const Integer& index : plan->required_cb_plan_indices) {
      ICHECK_GE(index->value, 0) << "TTMaterializationPlan requires non-negative CB plan index";
      ICHECK_LT(index->value, cb_plan_count)
          << "TTMaterializationPlan required_cb_plan_indices out of bounds";
    }
  }
}

void ValidateConsumerBindingPlans(const TTProgram& program,
                                  const std::unordered_set<std::string>& live_form_names,
                                  int64_t abi_plan_count) {
  for (const TTConsumerBindingPlan& plan : program->consumer_binding_plans) {
    ICHECK(!plan->name.empty()) << "TTConsumerBindingPlan requires name";
    ICHECK(!plan->consumer_kernel.empty()) << "TTConsumerBindingPlan requires consumer_kernel";
    ICHECK(!plan->consumer_op_kind.empty()) << "TTConsumerBindingPlan requires consumer_op_kind";
    ICHECK(!plan->source_live_form.empty())
        << "TTConsumerBindingPlan requires source_live_form";
    ICHECK(live_form_names.count(plan->source_live_form))
        << "TTConsumerBindingPlan references unknown source_live_form "
        << plan->source_live_form;
    ICHECK(!plan->live_value_edge.empty()) << "TTConsumerBindingPlan requires live_value_edge";
    ICHECK_GE(plan->live_value_edge_index, 0)
        << "TTConsumerBindingPlan requires live_value_edge_index";
    if (plan->abi_plan_index >= 0) {
      ICHECK_LT(plan->abi_plan_index, abi_plan_count)
          << "TTConsumerBindingPlan abi_plan_index out of bounds";
    }
    ICHECK(plan->accepts_distributed_slice || plan->requires_full_logical_tile)
        << "TTConsumerBindingPlan must declare whether the consumer accepts a distributed slice "
           "or requires a full logical tile";
  }
}

void ValidateSpatialLiveReferences(const TTProgram& program, const SpatialPlan& spatial_plan) {
  std::unordered_map<std::string, std::string> live_value_name_by_form;
  for (const TTLiveFormPlan& plan : program->live_form_plans) {
    ICHECK_LT(plan->spatial_live_value_index,
              static_cast<int64_t>(spatial_plan->live_values.size()))
        << "TTLiveFormPlan spatial_live_value_index out of bounds";
    const LiveValue& live_value =
        spatial_plan->live_values[static_cast<size_t>(plan->spatial_live_value_index)];
    ICHECK_EQ(plan->spatial_live_value, live_value->name)
        << "TTLiveFormPlan spatial_live_value must match SpatialPlan live_values index";
    ICHECK_EQ(plan->logical_value, live_value->subject)
        << "TTLiveFormPlan logical_value must match SpatialPlan LiveValue subject";
    ICHECK_GE(live_value->version_index, 0)
        << "TTLiveFormPlan requires versioned SpatialPlan LiveValue";
    ICHECK(!live_value->definition_kind.empty())
        << "TTLiveFormPlan requires SpatialPlan LiveValue definition_kind";
    live_value_name_by_form[static_cast<std::string>(plan->name)] =
        static_cast<std::string>(plan->spatial_live_value);
  }

  for (const TTMaterializationPlan& plan : program->materialization_plans) {
    ICHECK_LT(plan->materialization_boundary_index,
              static_cast<int64_t>(spatial_plan->materialization_boundaries.size()))
        << "TTMaterializationPlan materialization_boundary_index out of bounds";
    const MaterializationBoundary& boundary =
        spatial_plan->materialization_boundaries[static_cast<size_t>(
            plan->materialization_boundary_index)];
    ICHECK_EQ(plan->materialization_boundary, boundary->name)
        << "TTMaterializationPlan materialization_boundary must match SpatialPlan index";
    ICHECK(!boundary->event_lifetime_kind.empty())
        << "TTMaterializationPlan requires SpatialPlan MaterializationBoundary lifetime";
    ICHECK_GE(boundary->min_publish_pages, 1)
        << "TTMaterializationPlan requires bounded publish pages";
    auto source_it = live_value_name_by_form.find(static_cast<std::string>(plan->source_live_form));
    ICHECK(source_it != live_value_name_by_form.end())
        << "TTMaterializationPlan source_live_form missing matching TTLiveFormPlan";
    ICHECK_EQ(source_it->second, static_cast<std::string>(boundary->source_live_value))
        << "TTMaterializationPlan source_live_form must refer to boundary source_live_value";
    ICHECK_LT(boundary->target_live_value_index,
              static_cast<int64_t>(spatial_plan->live_values.size()))
        << "MaterializationBoundary target_live_value_index out of bounds";
    const LiveValue& target_live_value =
        spatial_plan->live_values[static_cast<size_t>(boundary->target_live_value_index)];
    ICHECK_EQ(boundary->target_live_value, target_live_value->name)
        << "MaterializationBoundary target_live_value must match SpatialPlan index";
    ICHECK_EQ(plan->target_buffer, target_live_value->subject)
        << "TTMaterializationPlan target_buffer must refer to boundary target_live_value";
  }

  for (const TTConsumerBindingPlan& plan : program->consumer_binding_plans) {
    ICHECK_LT(plan->live_value_edge_index,
              static_cast<int64_t>(spatial_plan->live_value_edges.size()))
        << "TTConsumerBindingPlan live_value_edge_index out of bounds";
    const LiveValueEdge& live_edge =
        spatial_plan->live_value_edges[static_cast<size_t>(plan->live_value_edge_index)];
    ICHECK_EQ(plan->live_value_edge, live_edge->name)
        << "TTConsumerBindingPlan live_value_edge must match SpatialPlan index";
    ICHECK(!live_edge->use_kind.empty())
        << "TTConsumerBindingPlan requires SpatialPlan LiveValueEdge use_kind";
    ICHECK_GE(live_edge->source_version_index, 0)
        << "TTConsumerBindingPlan requires SpatialPlan source version";
    auto source_it = live_value_name_by_form.find(static_cast<std::string>(plan->source_live_form));
    ICHECK(source_it != live_value_name_by_form.end())
        << "TTConsumerBindingPlan source_live_form missing matching TTLiveFormPlan";
    ICHECK_EQ(source_it->second, static_cast<std::string>(live_edge->source_live_value))
        << "TTConsumerBindingPlan source_live_form must refer to edge source_live_value";
  }
}

void CheckTTProgram(const TTProgram& program, const SpatialPlan& spatial_plan,
                    const std::optional<TTHardwareModel>& maybe_hardware_model) {
  ICHECK(!program->entry_name.empty()) << "TTProgram requires entry_name";
  ICHECK(!program->mesh_plans.empty()) << "TTProgram requires at least one TTMeshPlan";
  ICHECK(!program->buffer_distribution_plans.empty())
      << "TTProgram requires at least one TTBufferDistributionPlan";
  ICHECK(!program->block_plans.empty()) << "TTProgram requires at least one TTBlockPlan";
  ICHECK(!program->kernel_plans.empty()) << "TTProgram requires at least one TTKernelPlan";
  ICHECK(!program->kernels.empty()) << "TTProgram requires at least one TTKernel";
  ICHECK(!program->core_groups.empty()) << "TTProgram requires at least one TTCoreGroup";
  ICHECK(!program->abi_plans.empty()) << "TTProgram requires at least one TTABIPlan";
  ICHECK(!program->execution_plans.empty()) << "TTProgram requires at least one TTExecutionPlan";
  ICHECK_EQ(program->block_plans.size(), program->core_groups.size())
      << "TTProgram requires aligned TTBlockPlan and TTCoreGroup owner truth";
  ICHECK_EQ(program->kernel_plans.size(), program->kernels.size())
      << "TTProgram requires aligned TTKernelPlan and TTKernel owner truth";
  ICHECK_EQ(program->sync_plans.size(), program->compute_sync_plans.size())
      << "TTProgram requires aligned TTSyncPlan and TTComputeSyncPlan owner truth";

  std::unordered_map<std::string, int64_t> mesh_index_by_name;
  for (int64_t mesh_index = 0; mesh_index < static_cast<int64_t>(program->mesh_plans.size());
       ++mesh_index) {
    const TTMeshPlan& mesh_plan = program->mesh_plans[mesh_index];
    ValidateMeshPlan(mesh_plan);
    ICHECK(mesh_index_by_name.emplace(static_cast<std::string>(mesh_plan->name), mesh_index).second)
        << "duplicate TTMeshPlan name " << mesh_plan->name;
  }

  std::unordered_set<std::string> spatial_layout_subjects;
  for (const LayoutSpec& layout : spatial_plan->layout_specs) {
    spatial_layout_subjects.insert(static_cast<std::string>(layout->subject));
  }
  std::unordered_set<std::string> distributed_buffers;
  for (const TTBufferDistributionPlan& distribution : program->buffer_distribution_plans) {
    ValidateBufferDistributionPlan(distribution, mesh_index_by_name);
    ICHECK(distributed_buffers.insert(static_cast<std::string>(distribution->buffer)).second)
        << "duplicate TTBufferDistributionPlan buffer " << distribution->buffer;
    ICHECK(spatial_layout_subjects.count(static_cast<std::string>(distribution->buffer)))
        << "TTBufferDistributionPlan buffer must match SpatialPlan LayoutSpec subject "
        << distribution->buffer;
  }

  for (const TTBlockPlan& block_plan : program->block_plans) {
    ValidateBlockPlan(block_plan);
  }
  for (const TTSyncPlan& sync_plan : program->sync_plans) {
    ValidateSyncPlan(sync_plan);
  }

  std::unordered_set<std::string> kernel_names;
  for (const TTKernel& kernel : program->kernels) {
    ICHECK(!kernel->name.empty()) << "TTKernel requires name";
    ICHECK(!kernel->kind.empty()) << "TTKernel requires kind";
    ICHECK(!kernel->core_type.empty()) << "TTKernel requires core_type";
    ICHECK_GE(kernel->abi_plan_index, 0) << "TTKernel requires abi_plan_index";
    ICHECK_LT(kernel->abi_plan_index, static_cast<int64_t>(program->abi_plans.size()))
        << "TTKernel abi_plan_index out of bounds";
    ICHECK(kernel_names.insert(kernel->name).second) << "duplicate TTKernel name " << kernel->name;
    ValidateKernelLeafFields(kernel);
  }
  for (const TTKernelPlan& kernel_plan : program->kernel_plans) {
    ValidateKernelPlan(kernel_plan, static_cast<int64_t>(program->abi_plans.size()),
                       static_cast<int64_t>(program->block_plans.size()));
    ICHECK(kernel_names.count(kernel_plan->name))
        << "TTKernelPlan missing matching TTKernel owner truth: " << kernel_plan->name;
  }
  std::unordered_set<std::string> compute_op_names;
  for (const TTComputeOpPlan& compute_op_plan : program->compute_op_plans) {
    ValidateComputeOpPlan(compute_op_plan,
                          static_cast<int64_t>(program->kernel_plans.size()), kernel_names);
    ICHECK(compute_op_names.insert(static_cast<std::string>(compute_op_plan->name)).second)
        << "duplicate TTComputeOpPlan name " << compute_op_plan->name;
  }

  std::unordered_map<std::string, const TTResourcePressureReportNode*>
      resource_report_by_kernel;
  std::unordered_set<std::string> resource_demand_kernels;
  for (const TTResourcePressureReport& report :
       program->resource_pressure_reports) {
    ValidateResourcePressureReport(
        report, kernel_names, static_cast<int64_t>(program->core_groups.size()),
        maybe_hardware_model);
    const std::string kernel_name = report->kernel_name;
    ICHECK(resource_report_by_kernel.emplace(kernel_name, report.get()).second)
        << "duplicate TTResourcePressureReport for kernel " << report->kernel_name;
  }
  for (const TTResourceDemand& demand : program->resource_demands) {
    ValidateResourceDemand(
        demand, kernel_names, static_cast<int64_t>(program->core_groups.size()));
    ICHECK(resource_demand_kernels
               .insert(static_cast<std::string>(demand->kernel_name))
               .second)
        << "duplicate TTResourceDemand for kernel " << demand->kernel_name;
    auto report_it =
        resource_report_by_kernel.find(static_cast<std::string>(demand->kernel_name));
    ICHECK(report_it != resource_report_by_kernel.end())
        << "TTResourceDemand requires matching ResourcePressureReport for kernel "
        << demand->kernel_name;
    ICHECK_GE(report_it->second->required_materializations.size(),
              demand->tile_compute_materialization_demands.size())
        << "ResourcePressureReport required_materializations must cover "
           "TTResourceDemand tile_compute_materialization_demands";
  }
  for (const auto& entry : resource_report_by_kernel) {
    ICHECK(resource_demand_kernels.count(entry.first))
        << "TTResourcePressureReport requires matching TTResourceDemand for kernel "
        << entry.first;
  }

  for (const TTCoreGroup& core_group : program->core_groups) {
    ValidateCoreGroup(core_group, maybe_hardware_model);
  }

  std::unordered_set<int64_t> cb_ids;
  for (const TTCBPlan& cb : program->cb_plans) {
    ValidateCBPlan(cb);
    ICHECK_GE(cb->cb_id, 0) << "TTCBPlan requires non-negative cb_id";
    ICHECK(cb_ids.insert(cb->cb_id).second) << "duplicate TTCBPlan cb_id " << cb->cb_id;
  }

  std::unordered_set<std::string> live_form_names;
  ValidateLiveFormPlans(program, &live_form_names);
  ValidateMaterializationPlans(program, live_form_names,
                               static_cast<int64_t>(program->cb_plans.size()));

  std::unordered_set<std::string> abi_kernel_names;
  for (const TTABIPlan& abi : program->abi_plans) {
    ICHECK(!abi->kernel_name.empty()) << "TTABIPlan requires kernel_name";
    for (const TTAccessorSpec& accessor : abi->accessors) {
      ValidateAccessor(accessor);
      ICHECK(distributed_buffers.count(static_cast<std::string>(accessor->buffer)))
          << "TTABIPlan accessor buffer requires TTBufferDistributionPlan";
    }
    for (const TTCompileTimeArgSpec& spec : abi->compile_time_arg_specs) {
      ValidateCompileTimeArgSpec(spec);
    }
    abi_kernel_names.insert(abi->kernel_name);
  }
  for (const TTKernel& kernel : program->kernels) {
    ICHECK(abi_kernel_names.count(kernel->name))
        << "TTKernel missing matching TTABIPlan: " << kernel->name;
  }
  ValidateConsumerBindingPlans(program, live_form_names,
                               static_cast<int64_t>(program->abi_plans.size()));
  ValidateSpatialLiveReferences(program, spatial_plan);

  for (const TTTransportPlan& transport : program->transport_plans) {
    ICHECK(!transport->kind.empty()) << "TTTransportPlan requires kind";
    ICHECK(!transport->value_kind.empty()) << "TTTransportPlan requires value_kind";
    ICHECK(!transport->delivery_kind.empty()) << "TTTransportPlan requires delivery_kind";
    ICHECK_GE(transport->source_task_index, 0) << "TTTransportPlan requires source_task_index";
    ICHECK_GE(transport->target_task_index, 0) << "TTTransportPlan requires target_task_index";
  }

  for (const TTSemaphorePlan& semaphore : program->semaphore_plans) {
    ICHECK_GE(semaphore->semaphore_id, 0) << "TTSemaphorePlan requires non-negative semaphore_id";
    ICHECK(!semaphore->kind.empty()) << "TTSemaphorePlan requires kind";
    ICHECK(!semaphore->core_type.empty()) << "TTSemaphorePlan requires core_type";
  }

  for (const TTDstLayoutPlan& layout : program->dst_layout_plans) {
    ICHECK(!layout->buffer.empty()) << "TTDstLayoutPlan requires buffer";
    ICHECK(!layout->layout.empty()) << "TTDstLayoutPlan requires layout";
    ICHECK(!layout->memory_space.empty()) << "TTDstLayoutPlan requires memory_space";
  }

  for (const TTExecutionPlan& execution : program->execution_plans) {
    ICHECK(!execution->kernel_names.empty()) << "TTExecutionPlan requires kernel_names";
    ICHECK(!execution->phase_indices.empty()) << "TTExecutionPlan requires phase_indices";
    for (const tvm::ffi::String& kernel_name : execution->kernel_names) {
      ICHECK(kernel_names.count(kernel_name)) << "TTExecutionPlan references unknown kernel "
                                              << kernel_name;
    }
  }
}

}  // namespace

tvm::transform::Pass ValidateTTProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    const std::optional<TTHardwareModel> maybe_hardware_model =
        GetValidationHardwareModel(mod);
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<TTProgram>(attr::kTLTTProgram);
      if (!maybe_program) {
        continue;
      }
      auto maybe_spatial_plan = func.value()->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
      ICHECK(maybe_spatial_plan)
          << "ValidateTTProgram requires tl.spatial_plan for live-form validation";
      CheckTTProgram(maybe_program.value(), maybe_spatial_plan.value(),
                     maybe_hardware_model);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.ValidateTTProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateTTProgram", ValidateTTProgram);
}

}  // namespace tl
}  // namespace tvm
