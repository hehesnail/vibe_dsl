/*!
 * \file validate_tt_program.cc
 * \brief Validate TTProgram invariants for Phase C cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_tile_compute_covering.h"
#include "common/blackhole_tile_compute_legalizer.h"
#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_plan.h"
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

bool HasKey(const Map<String, Any>& map, const char* key) { return map.Get(String(key)).has_value(); }

int64_t GetIntOrDefault(const Map<String, Any>& map, const char* key, int64_t default_value = -1) {
  if (auto value = map.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
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

void ValidateCoreGroup(const TTCoreGroup& core_group) {
  ICHECK_GT(core_group->logical_grid_x, 0) << "TTCoreGroup requires positive logical_grid_x";
  ICHECK_GT(core_group->logical_grid_y, 0) << "TTCoreGroup requires positive logical_grid_y";
  ICHECK(!core_group->physical_cores.empty()) << "TTCoreGroup requires physical_cores";
  ICHECK(!core_group->work_packets.empty()) << "TTCoreGroup requires work_packets";
  for (const Any& item : core_group->work_packets) {
    Map<String, Any> packet = AsMap(item);
    ICHECK(!packet.empty()) << "TTCoreGroup work_packet must be a map";
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

void RequireSelectedBlackholeTileComputeCoveringForPlan(
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

void CheckTTProgram(const TTProgram& program, const SpatialPlan& spatial_plan) {
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

  for (const TTCoreGroup& core_group : program->core_groups) {
    ValidateCoreGroup(core_group);
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
      CheckTTProgram(maybe_program.value(), maybe_spatial_plan.value());
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
