/*!
 * \file tt_target_program.cc
 * \brief Stage 4 Phase C TT target companion objects.
 */

#include "tt_target_program.h"

namespace tvm {
namespace tl {

namespace {

template <typename NodeT> void RegisterNodeReflection() {
  NodeT::RegisterReflection();
}

} // namespace

void TTMeshPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTMeshPlanNode>()
      .def_ro("name", &TTMeshPlanNode::name)
      .def_ro("mesh_kind", &TTMeshPlanNode::mesh_kind)
      .def_ro("mesh_shape", &TTMeshPlanNode::mesh_shape)
      .def_ro("device_range_start", &TTMeshPlanNode::device_range_start)
      .def_ro("device_range_shape", &TTMeshPlanNode::device_range_shape)
      .def_ro("system_mesh_ref", &TTMeshPlanNode::system_mesh_ref);
}

TTMeshPlan::TTMeshPlan(ffi::String name, ffi::String mesh_kind,
                       ffi::Array<Integer> mesh_shape,
                       ffi::Array<Integer> device_range_start,
                       ffi::Array<Integer> device_range_shape,
                       ffi::String system_mesh_ref) {
  auto n = ffi::make_object<TTMeshPlanNode>();
  n->name = std::move(name);
  n->mesh_kind = std::move(mesh_kind);
  n->mesh_shape = std::move(mesh_shape);
  n->device_range_start = std::move(device_range_start);
  n->device_range_shape = std::move(device_range_shape);
  n->system_mesh_ref = std::move(system_mesh_ref);
  data_ = std::move(n);
}

void TTBufferDistributionPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTBufferDistributionPlanNode>()
      .def_ro("name", &TTBufferDistributionPlanNode::name)
      .def_ro("buffer", &TTBufferDistributionPlanNode::buffer)
      .def_ro("mesh_plan", &TTBufferDistributionPlanNode::mesh_plan)
      .def_ro("mesh_plan_index", &TTBufferDistributionPlanNode::mesh_plan_index)
      .def_ro("distribution_kind",
              &TTBufferDistributionPlanNode::distribution_kind)
      .def_ro("layout", &TTBufferDistributionPlanNode::layout)
      .def_ro("memory_space", &TTBufferDistributionPlanNode::memory_space)
      .def_ro("page_size_bytes", &TTBufferDistributionPlanNode::page_size_bytes)
      .def_ro("shard_shape", &TTBufferDistributionPlanNode::shard_shape)
      .def_ro("shard_grid_shape",
              &TTBufferDistributionPlanNode::shard_grid_shape)
      .def_ro("sharding_strategy",
              &TTBufferDistributionPlanNode::sharding_strategy)
      .def_ro("shard_orientation",
              &TTBufferDistributionPlanNode::shard_orientation)
      .def_ro("source_buffer", &TTBufferDistributionPlanNode::source_buffer)
      .def_ro("source_region_kind",
              &TTBufferDistributionPlanNode::source_region_kind)
      .def_ro("source_region_shape",
              &TTBufferDistributionPlanNode::source_region_shape)
      .def_ro("logical_index_mapping",
              &TTBufferDistributionPlanNode::logical_index_mapping)
      .def_ro("core_local_address_mapping",
              &TTBufferDistributionPlanNode::core_local_address_mapping)
      .def_ro("host_visibility", &TTBufferDistributionPlanNode::host_visibility)
      .def_ro("attached_core_group",
              &TTBufferDistributionPlanNode::attached_core_group)
      .def_ro("attached_core_group_index",
              &TTBufferDistributionPlanNode::attached_core_group_index)
      .def_ro("logical_shape", &TTBufferDistributionPlanNode::logical_shape)
      .def_ro("local_shape", &TTBufferDistributionPlanNode::local_shape)
      .def_ro("thread_extent", &TTBufferDistributionPlanNode::thread_extent)
      .def_ro("replicate_extent",
              &TTBufferDistributionPlanNode::replicate_extent)
      .def_ro("inverse_logical_index_vars",
              &TTBufferDistributionPlanNode::inverse_logical_index_vars)
      .def_ro("inverse_logical_index_exprs",
              &TTBufferDistributionPlanNode::inverse_logical_index_exprs)
      .def_ro("spatial_layout", &TTBufferDistributionPlanNode::spatial_layout)
      .def_ro("spatial_distribution_kind",
              &TTBufferDistributionPlanNode::spatial_distribution_kind)
      .def_ro("abi_layout", &TTBufferDistributionPlanNode::abi_layout)
      .def_ro("abi_memory_space",
              &TTBufferDistributionPlanNode::abi_memory_space);
}

TTBufferDistributionPlan::TTBufferDistributionPlan(
    ffi::String name, ffi::String buffer, ffi::String mesh_plan,
    int64_t mesh_plan_index, ffi::String distribution_kind, ffi::String layout,
    ffi::String memory_space, int64_t page_size_bytes,
    ffi::Array<Integer> shard_shape, ffi::Array<Integer> shard_grid_shape,
    ffi::String sharding_strategy, ffi::String shard_orientation,
    ffi::String source_buffer, ffi::String source_region_kind,
    ffi::Array<Integer> source_region_shape, ffi::String logical_index_mapping,
    ffi::String core_local_address_mapping, ffi::String host_visibility,
    ffi::String attached_core_group, int64_t attached_core_group_index,
    ffi::Array<PrimExpr> logical_shape, ffi::Array<PrimExpr> local_shape,
    PrimExpr thread_extent, PrimExpr replicate_extent,
    ffi::Array<PrimExpr> inverse_logical_index_vars,
    ffi::Array<PrimExpr> inverse_logical_index_exprs,
    ffi::String spatial_layout, ffi::String spatial_distribution_kind,
    ffi::String abi_layout, ffi::String abi_memory_space) {
  auto n = ffi::make_object<TTBufferDistributionPlanNode>();
  n->name = std::move(name);
  n->buffer = std::move(buffer);
  n->mesh_plan = std::move(mesh_plan);
  n->mesh_plan_index = mesh_plan_index;
  n->distribution_kind = std::move(distribution_kind);
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->page_size_bytes = page_size_bytes;
  n->shard_shape = std::move(shard_shape);
  n->shard_grid_shape = std::move(shard_grid_shape);
  n->sharding_strategy = std::move(sharding_strategy);
  n->shard_orientation = std::move(shard_orientation);
  n->source_buffer = std::move(source_buffer);
  n->source_region_kind = std::move(source_region_kind);
  n->source_region_shape = std::move(source_region_shape);
  n->logical_index_mapping = std::move(logical_index_mapping);
  n->core_local_address_mapping = std::move(core_local_address_mapping);
  n->host_visibility = std::move(host_visibility);
  n->attached_core_group = std::move(attached_core_group);
  n->attached_core_group_index = attached_core_group_index;
  n->logical_shape = std::move(logical_shape);
  n->local_shape = std::move(local_shape);
  n->thread_extent = std::move(thread_extent);
  n->replicate_extent = std::move(replicate_extent);
  n->inverse_logical_index_vars = std::move(inverse_logical_index_vars);
  n->inverse_logical_index_exprs = std::move(inverse_logical_index_exprs);
  n->spatial_layout = std::move(spatial_layout);
  n->spatial_distribution_kind = std::move(spatial_distribution_kind);
  n->abi_layout = std::move(abi_layout);
  n->abi_memory_space = std::move(abi_memory_space);
  data_ = std::move(n);
}

void TTTensorMemoryConfigPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTTensorMemoryConfigPlanNode>()
      .def_ro("name", &TTTensorMemoryConfigPlanNode::name)
      .def_ro("subject", &TTTensorMemoryConfigPlanNode::subject)
      .def_ro("value_identity", &TTTensorMemoryConfigPlanNode::value_identity)
      .def_ro("logical_shape", &TTTensorMemoryConfigPlanNode::logical_shape)
      .def_ro("dtype", &TTTensorMemoryConfigPlanNode::dtype)
      .def_ro("memory_layout", &TTTensorMemoryConfigPlanNode::memory_layout)
      .def_ro("buffer_type", &TTTensorMemoryConfigPlanNode::buffer_type)
      .def_ro("grid_ref", &TTTensorMemoryConfigPlanNode::grid_ref)
      .def_ro("shard_grid_shape",
              &TTTensorMemoryConfigPlanNode::shard_grid_shape)
      .def_ro("shard_shape", &TTTensorMemoryConfigPlanNode::shard_shape)
      .def_ro("shard_orientation",
              &TTTensorMemoryConfigPlanNode::shard_orientation)
      .def_ro("shard_distribution_strategy",
              &TTTensorMemoryConfigPlanNode::shard_distribution_strategy)
      .def_ro("page_shape", &TTTensorMemoryConfigPlanNode::page_shape)
      .def_ro("origin", &TTTensorMemoryConfigPlanNode::origin)
      .def_ro("source_buffer", &TTTensorMemoryConfigPlanNode::source_buffer)
      .def_ro("buffer_distribution_plan",
              &TTTensorMemoryConfigPlanNode::buffer_distribution_plan)
      .def_ro("buffer_distribution_plan_index",
              &TTTensorMemoryConfigPlanNode::buffer_distribution_plan_index)
      .def_ro("has_runtime_accessor",
              &TTTensorMemoryConfigPlanNode::has_runtime_accessor)
      .def_ro("requires_materialization",
              &TTTensorMemoryConfigPlanNode::requires_materialization);
}

TTTensorMemoryConfigPlan::TTTensorMemoryConfigPlan(
    ffi::String name, ffi::String subject, ffi::String value_identity,
    ffi::Array<PrimExpr> logical_shape, ffi::String dtype,
    ffi::String memory_layout, ffi::String buffer_type, ffi::String grid_ref,
    ffi::Array<Integer> shard_grid_shape, ffi::Array<Integer> shard_shape,
    ffi::String shard_orientation, ffi::String shard_distribution_strategy,
    ffi::Array<Integer> page_shape, ffi::String origin,
    ffi::String source_buffer, ffi::String buffer_distribution_plan,
    int64_t buffer_distribution_plan_index, bool has_runtime_accessor,
    bool requires_materialization) {
  auto n = ffi::make_object<TTTensorMemoryConfigPlanNode>();
  n->name = std::move(name);
  n->subject = std::move(subject);
  n->value_identity = std::move(value_identity);
  n->logical_shape = std::move(logical_shape);
  n->dtype = std::move(dtype);
  n->memory_layout = std::move(memory_layout);
  n->buffer_type = std::move(buffer_type);
  n->grid_ref = std::move(grid_ref);
  n->shard_grid_shape = std::move(shard_grid_shape);
  n->shard_shape = std::move(shard_shape);
  n->shard_orientation = std::move(shard_orientation);
  n->shard_distribution_strategy = std::move(shard_distribution_strategy);
  n->page_shape = std::move(page_shape);
  n->origin = std::move(origin);
  n->source_buffer = std::move(source_buffer);
  n->buffer_distribution_plan = std::move(buffer_distribution_plan);
  n->buffer_distribution_plan_index = buffer_distribution_plan_index;
  n->has_runtime_accessor = has_runtime_accessor;
  n->requires_materialization = requires_materialization;
  data_ = std::move(n);
}

void TTOpShardingContractNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTOpShardingContractNode>()
      .def_ro("name", &TTOpShardingContractNode::name)
      .def_ro("compute_op_plan", &TTOpShardingContractNode::compute_op_plan)
      .def_ro("compute_op_plan_index",
              &TTOpShardingContractNode::compute_op_plan_index)
      .def_ro("operation_name", &TTOpShardingContractNode::operation_name)
      .def_ro("op_kind", &TTOpShardingContractNode::op_kind)
      .def_ro("operand_role", &TTOpShardingContractNode::operand_role)
      .def_ro("operand_buffer", &TTOpShardingContractNode::operand_buffer)
      .def_ro("operand_host_buffer",
              &TTOpShardingContractNode::operand_host_buffer)
      .def_ro("memory_config_plan",
              &TTOpShardingContractNode::memory_config_plan)
      .def_ro("memory_config_plan_index",
              &TTOpShardingContractNode::memory_config_plan_index)
      .def_ro("accepted_memory_layouts",
              &TTOpShardingContractNode::accepted_memory_layouts)
      .def_ro("accepted_buffer_types",
              &TTOpShardingContractNode::accepted_buffer_types)
      .def_ro("accepted_sharding_strategies",
              &TTOpShardingContractNode::accepted_sharding_strategies)
      .def_ro("required_shard_orientation",
              &TTOpShardingContractNode::required_shard_orientation)
      .def_ro("output_policy", &TTOpShardingContractNode::output_policy)
      .def_ro("may_request_input_conversion",
              &TTOpShardingContractNode::may_request_input_conversion)
      .def_ro("can_produce_output_placement",
              &TTOpShardingContractNode::can_produce_output_placement)
      .def_ro("direct_external_write_allowed",
              &TTOpShardingContractNode::direct_external_write_allowed)
      .def_ro("reject_reason", &TTOpShardingContractNode::reject_reason);
}

TTOpShardingContract::TTOpShardingContract(
    ffi::String name, ffi::String compute_op_plan, int64_t compute_op_plan_index,
    ffi::String operation_name, ffi::String op_kind, ffi::String operand_role,
    ffi::String operand_buffer, ffi::String operand_host_buffer,
    ffi::String memory_config_plan, int64_t memory_config_plan_index,
    ffi::Array<ffi::String> accepted_memory_layouts,
    ffi::Array<ffi::String> accepted_buffer_types,
    ffi::Array<ffi::String> accepted_sharding_strategies,
    ffi::String required_shard_orientation, ffi::String output_policy,
    bool may_request_input_conversion, bool can_produce_output_placement,
    bool direct_external_write_allowed, ffi::String reject_reason) {
  auto n = ffi::make_object<TTOpShardingContractNode>();
  n->name = std::move(name);
  n->compute_op_plan = std::move(compute_op_plan);
  n->compute_op_plan_index = compute_op_plan_index;
  n->operation_name = std::move(operation_name);
  n->op_kind = std::move(op_kind);
  n->operand_role = std::move(operand_role);
  n->operand_buffer = std::move(operand_buffer);
  n->operand_host_buffer = std::move(operand_host_buffer);
  n->memory_config_plan = std::move(memory_config_plan);
  n->memory_config_plan_index = memory_config_plan_index;
  n->accepted_memory_layouts = std::move(accepted_memory_layouts);
  n->accepted_buffer_types = std::move(accepted_buffer_types);
  n->accepted_sharding_strategies = std::move(accepted_sharding_strategies);
  n->required_shard_orientation = std::move(required_shard_orientation);
  n->output_policy = std::move(output_policy);
  n->may_request_input_conversion = may_request_input_conversion;
  n->can_produce_output_placement = can_produce_output_placement;
  n->direct_external_write_allowed = direct_external_write_allowed;
  n->reject_reason = std::move(reject_reason);
  data_ = std::move(n);
}

void TTPlacementResolutionPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTPlacementResolutionPlanNode>()
      .def_ro("name", &TTPlacementResolutionPlanNode::name)
      .def_ro("op_sharding_contract",
              &TTPlacementResolutionPlanNode::op_sharding_contract)
      .def_ro("op_sharding_contract_index",
              &TTPlacementResolutionPlanNode::op_sharding_contract_index)
      .def_ro("consumer_op_plan",
              &TTPlacementResolutionPlanNode::consumer_op_plan)
      .def_ro("consumer_op_plan_index",
              &TTPlacementResolutionPlanNode::consumer_op_plan_index)
      .def_ro("consumer_operand_role",
              &TTPlacementResolutionPlanNode::consumer_operand_role)
      .def_ro("selected_memory_config_plan",
              &TTPlacementResolutionPlanNode::selected_memory_config_plan)
      .def_ro("selected_memory_config_plan_index",
              &TTPlacementResolutionPlanNode::selected_memory_config_plan_index)
      .def_ro("selected_memory_layout",
              &TTPlacementResolutionPlanNode::selected_memory_layout)
      .def_ro("selected_buffer_type",
              &TTPlacementResolutionPlanNode::selected_buffer_type)
      .def_ro("resolution_kind",
              &TTPlacementResolutionPlanNode::resolution_kind)
      .def_ro("conversion_required",
              &TTPlacementResolutionPlanNode::conversion_required)
      .def_ro("conversion_plan",
              &TTPlacementResolutionPlanNode::conversion_plan)
      .def_ro("conflict_reason",
              &TTPlacementResolutionPlanNode::conflict_reason);
}

TTPlacementResolutionPlan::TTPlacementResolutionPlan(
    ffi::String name, ffi::String op_sharding_contract,
    int64_t op_sharding_contract_index, ffi::String consumer_op_plan,
    int64_t consumer_op_plan_index, ffi::String consumer_operand_role,
    ffi::String selected_memory_config_plan,
    int64_t selected_memory_config_plan_index,
    ffi::String selected_memory_layout, ffi::String selected_buffer_type,
    ffi::String resolution_kind, bool conversion_required,
    ffi::String conversion_plan, ffi::String conflict_reason) {
  auto n = ffi::make_object<TTPlacementResolutionPlanNode>();
  n->name = std::move(name);
  n->op_sharding_contract = std::move(op_sharding_contract);
  n->op_sharding_contract_index = op_sharding_contract_index;
  n->consumer_op_plan = std::move(consumer_op_plan);
  n->consumer_op_plan_index = consumer_op_plan_index;
  n->consumer_operand_role = std::move(consumer_operand_role);
  n->selected_memory_config_plan = std::move(selected_memory_config_plan);
  n->selected_memory_config_plan_index = selected_memory_config_plan_index;
  n->selected_memory_layout = std::move(selected_memory_layout);
  n->selected_buffer_type = std::move(selected_buffer_type);
  n->resolution_kind = std::move(resolution_kind);
  n->conversion_required = conversion_required;
  n->conversion_plan = std::move(conversion_plan);
  n->conflict_reason = std::move(conflict_reason);
  data_ = std::move(n);
}

void TTReshardPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTReshardPlanNode>()
      .def_ro("name", &TTReshardPlanNode::name)
      .def_ro("source_value", &TTReshardPlanNode::source_value)
      .def_ro("target_value", &TTReshardPlanNode::target_value)
      .def_ro("source_memory_config_plan",
              &TTReshardPlanNode::source_memory_config_plan)
      .def_ro("source_memory_config_plan_index",
              &TTReshardPlanNode::source_memory_config_plan_index)
      .def_ro("target_memory_config_plan",
              &TTReshardPlanNode::target_memory_config_plan)
      .def_ro("target_memory_config_plan_index",
              &TTReshardPlanNode::target_memory_config_plan_index)
      .def_ro("conversion_kind", &TTReshardPlanNode::conversion_kind)
      .def_ro("source_region_kind", &TTReshardPlanNode::source_region_kind)
      .def_ro("source_region_shape", &TTReshardPlanNode::source_region_shape)
      .def_ro("materialization_plan",
              &TTReshardPlanNode::materialization_plan)
      .def_ro("materialization_plan_index",
              &TTReshardPlanNode::materialization_plan_index)
      .def_ro("materialization_protocol",
              &TTReshardPlanNode::materialization_protocol)
      .def_ro("required_cb_plan_indices",
              &TTReshardPlanNode::required_cb_plan_indices)
      .def_ro("required_sync_plan_indices",
              &TTReshardPlanNode::required_sync_plan_indices)
      .def_ro("scheduling_kind", &TTReshardPlanNode::scheduling_kind)
      .def_ro("inserted_by", &TTReshardPlanNode::inserted_by)
      .def_ro("admission_status", &TTReshardPlanNode::admission_status)
      .def_ro("unsupported_reason", &TTReshardPlanNode::unsupported_reason);
}

TTReshardPlan::TTReshardPlan(
    ffi::String name, ffi::String source_value, ffi::String target_value,
    ffi::String source_memory_config_plan,
    int64_t source_memory_config_plan_index,
    ffi::String target_memory_config_plan,
    int64_t target_memory_config_plan_index, ffi::String conversion_kind,
    ffi::String source_region_kind, ffi::Array<Integer> source_region_shape,
    ffi::String materialization_plan, int64_t materialization_plan_index,
    ffi::String materialization_protocol,
    ffi::Array<Integer> required_cb_plan_indices,
    ffi::Array<Integer> required_sync_plan_indices,
    ffi::String scheduling_kind, ffi::String inserted_by,
    ffi::String admission_status, ffi::String unsupported_reason) {
  auto n = ffi::make_object<TTReshardPlanNode>();
  n->name = std::move(name);
  n->source_value = std::move(source_value);
  n->target_value = std::move(target_value);
  n->source_memory_config_plan = std::move(source_memory_config_plan);
  n->source_memory_config_plan_index = source_memory_config_plan_index;
  n->target_memory_config_plan = std::move(target_memory_config_plan);
  n->target_memory_config_plan_index = target_memory_config_plan_index;
  n->conversion_kind = std::move(conversion_kind);
  n->source_region_kind = std::move(source_region_kind);
  n->source_region_shape = std::move(source_region_shape);
  n->materialization_plan = std::move(materialization_plan);
  n->materialization_plan_index = materialization_plan_index;
  n->materialization_protocol = std::move(materialization_protocol);
  n->required_cb_plan_indices = std::move(required_cb_plan_indices);
  n->required_sync_plan_indices = std::move(required_sync_plan_indices);
  n->scheduling_kind = std::move(scheduling_kind);
  n->inserted_by = std::move(inserted_by);
  n->admission_status = std::move(admission_status);
  n->unsupported_reason = std::move(unsupported_reason);
  data_ = std::move(n);
}

void TTComputeOperandBindingPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeOperandBindingPlanNode>()
      .def_ro("role", &TTComputeOperandBindingPlanNode::role)
      .def_ro("buffer", &TTComputeOperandBindingPlanNode::buffer)
      .def_ro("host_buffer", &TTComputeOperandBindingPlanNode::host_buffer)
      .def_ro("tensor_dtype", &TTComputeOperandBindingPlanNode::tensor_dtype)
      .def_ro("cb_dtype", &TTComputeOperandBindingPlanNode::cb_dtype)
      .def_ro("transform_kind",
              &TTComputeOperandBindingPlanNode::transform_kind);
}

TTComputeOperandBindingPlan::TTComputeOperandBindingPlan(
    ffi::String role, ffi::String buffer, ffi::String host_buffer,
    ffi::String tensor_dtype, ffi::String cb_dtype,
    ffi::String transform_kind) {
  auto n = ffi::make_object<TTComputeOperandBindingPlanNode>();
  n->role = std::move(role);
  n->buffer = std::move(buffer);
  n->host_buffer = std::move(host_buffer);
  n->tensor_dtype = std::move(tensor_dtype);
  n->cb_dtype = std::move(cb_dtype);
  n->transform_kind = std::move(transform_kind);
  data_ = std::move(n);
}

void TTComputeOpPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeOpPlanNode>()
      .def_ro("name", &TTComputeOpPlanNode::name)
      .def_ro("kernel_name", &TTComputeOpPlanNode::kernel_name)
      .def_ro("kernel_plan_index", &TTComputeOpPlanNode::kernel_plan_index)
      .def_ro("kind", &TTComputeOpPlanNode::kind)
      .def_ro("operation_name", &TTComputeOpPlanNode::operation_name)
      .def_ro("enabled", &TTComputeOpPlanNode::enabled)
      .def_ro("operand_bindings", &TTComputeOpPlanNode::operand_bindings)
      .def_ro("problem_shape_axes", &TTComputeOpPlanNode::problem_shape_axes)
      .def_ro("problem_shape", &TTComputeOpPlanNode::problem_shape)
      .def_ro("tile_shape", &TTComputeOpPlanNode::tile_shape)
      .def_ro("block_shape", &TTComputeOpPlanNode::block_shape)
      .def_ro("subblock_shape", &TTComputeOpPlanNode::subblock_shape)
      .def_ro("accumulator_dtype", &TTComputeOpPlanNode::accumulator_dtype)
      .def_ro("mbarrier_buffer", &TTComputeOpPlanNode::mbarrier_buffer)
      .def_ro("mbarrier_scope", &TTComputeOpPlanNode::mbarrier_scope)
      .def_ro("mbarrier_index_exprs",
              &TTComputeOpPlanNode::mbarrier_index_exprs)
      .def_ro("tile_compute_dag_node_id",
              &TTComputeOpPlanNode::tile_compute_dag_node_id)
      .def_ro("tile_compute_source_emitter",
              &TTComputeOpPlanNode::tile_compute_source_emitter)
      .def_ro("tile_compute_materialization_policy",
              &TTComputeOpPlanNode::tile_compute_materialization_policy)
      .def_ro("tile_compute_fanout_use_count",
              &TTComputeOpPlanNode::tile_compute_fanout_use_count)
      .def_ro("tile_compute_fanout_policy",
              &TTComputeOpPlanNode::tile_compute_fanout_policy);
}

TTComputeOpPlan::TTComputeOpPlan(
    ffi::String name, ffi::String kernel_name, int64_t kernel_plan_index,
    ffi::String kind, ffi::String operation_name, bool enabled,
    ffi::Array<TTComputeOperandBindingPlan> operand_bindings,
    ffi::Array<ffi::String> problem_shape_axes,
    ffi::Array<Integer> problem_shape, ffi::Array<Integer> tile_shape,
    ffi::Array<Integer> block_shape, ffi::Array<Integer> subblock_shape,
    ffi::String accumulator_dtype, ffi::String mbarrier_buffer,
    ffi::String mbarrier_scope, ffi::Array<ffi::String> mbarrier_index_exprs,
    int64_t tile_compute_dag_node_id, ffi::String tile_compute_source_emitter,
    ffi::String tile_compute_materialization_policy,
    int64_t tile_compute_fanout_use_count,
    ffi::String tile_compute_fanout_policy) {
  auto n = ffi::make_object<TTComputeOpPlanNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->kernel_plan_index = kernel_plan_index;
  n->kind = std::move(kind);
  n->operation_name = std::move(operation_name);
  n->enabled = enabled;
  n->operand_bindings = std::move(operand_bindings);
  n->problem_shape_axes = std::move(problem_shape_axes);
  n->problem_shape = std::move(problem_shape);
  n->tile_shape = std::move(tile_shape);
  n->block_shape = std::move(block_shape);
  n->subblock_shape = std::move(subblock_shape);
  n->accumulator_dtype = std::move(accumulator_dtype);
  n->mbarrier_buffer = std::move(mbarrier_buffer);
  n->mbarrier_scope = std::move(mbarrier_scope);
  n->mbarrier_index_exprs = std::move(mbarrier_index_exprs);
  n->tile_compute_dag_node_id = tile_compute_dag_node_id;
  n->tile_compute_source_emitter = std::move(tile_compute_source_emitter);
  n->tile_compute_materialization_policy =
      std::move(tile_compute_materialization_policy);
  n->tile_compute_fanout_use_count = tile_compute_fanout_use_count;
  n->tile_compute_fanout_policy = std::move(tile_compute_fanout_policy);
  data_ = std::move(n);
}

void TTTileComputeFanoutDemandNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTTileComputeFanoutDemandNode>()
      .def_ro("name", &TTTileComputeFanoutDemandNode::name)
      .def_ro("kernel_name", &TTTileComputeFanoutDemandNode::kernel_name)
      .def_ro("producer_node", &TTTileComputeFanoutDemandNode::producer_node)
      .def_ro("producer_operation",
              &TTTileComputeFanoutDemandNode::producer_operation)
      .def_ro("value_repr", &TTTileComputeFanoutDemandNode::value_repr)
      .def_ro("use_count", &TTTileComputeFanoutDemandNode::use_count)
      .def_ro("consumer_nodes", &TTTileComputeFanoutDemandNode::consumer_nodes)
      .def_ro("policy", &TTTileComputeFanoutDemandNode::policy)
      .def_ro("evidence", &TTTileComputeFanoutDemandNode::evidence);
}

TTTileComputeFanoutDemand::TTTileComputeFanoutDemand(
    ffi::String name, ffi::String kernel_name, int64_t producer_node,
    ffi::String producer_operation, ffi::String value_repr, int64_t use_count,
    ffi::Array<Integer> consumer_nodes, ffi::String policy,
    ffi::String evidence) {
  auto n = ffi::make_object<TTTileComputeFanoutDemandNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->producer_node = producer_node;
  n->producer_operation = std::move(producer_operation);
  n->value_repr = std::move(value_repr);
  n->use_count = use_count;
  n->consumer_nodes = std::move(consumer_nodes);
  n->policy = std::move(policy);
  n->evidence = std::move(evidence);
  data_ = std::move(n);
}

void TTTileComputeMaterializationDemandNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTTileComputeMaterializationDemandNode>()
      .def_ro("name", &TTTileComputeMaterializationDemandNode::name)
      .def_ro("kernel_name",
              &TTTileComputeMaterializationDemandNode::kernel_name)
      .def_ro("node_id", &TTTileComputeMaterializationDemandNode::node_id)
      .def_ro("operation_name",
              &TTTileComputeMaterializationDemandNode::operation_name)
      .def_ro("pattern_name",
              &TTTileComputeMaterializationDemandNode::pattern_name)
      .def_ro("policy", &TTTileComputeMaterializationDemandNode::policy)
      .def_ro("evidence", &TTTileComputeMaterializationDemandNode::evidence);
}

TTTileComputeMaterializationDemand::TTTileComputeMaterializationDemand(
    ffi::String name, ffi::String kernel_name, int64_t node_id,
    ffi::String operation_name, ffi::String pattern_name, ffi::String policy,
    ffi::String evidence) {
  auto n = ffi::make_object<TTTileComputeMaterializationDemandNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->node_id = node_id;
  n->operation_name = std::move(operation_name);
  n->pattern_name = std::move(pattern_name);
  n->policy = std::move(policy);
  n->evidence = std::move(evidence);
  data_ = std::move(n);
}

void TTResourceDemandNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTResourceDemandNode>()
      .def_ro("name", &TTResourceDemandNode::name)
      .def_ro("kernel_name", &TTResourceDemandNode::kernel_name)
      .def_ro("core_group", &TTResourceDemandNode::core_group)
      .def_ro("core_group_index", &TTResourceDemandNode::core_group_index)
      .def_ro("tile_compute_fanout_demands",
              &TTResourceDemandNode::tile_compute_fanout_demands)
      .def_ro("tile_compute_materialization_demands",
              &TTResourceDemandNode::tile_compute_materialization_demands)
      .def_ro("tile_compute_unsupported_reasons",
              &TTResourceDemandNode::tile_compute_unsupported_reasons)
      .def_ro("cb_requirement_count",
              &TTResourceDemandNode::cb_requirement_count)
      .def_ro("cb_l1_bytes", &TTResourceDemandNode::cb_l1_bytes)
      .def_ro("semaphore_count", &TTResourceDemandNode::semaphore_count)
      .def_ro("communication_edge_count",
              &TTResourceDemandNode::communication_edge_count);
}

TTResourceDemand::TTResourceDemand(
    ffi::String name, ffi::String kernel_name, ffi::String core_group,
    int64_t core_group_index,
    ffi::Array<TTTileComputeFanoutDemand> tile_compute_fanout_demands,
    ffi::Array<TTTileComputeMaterializationDemand>
        tile_compute_materialization_demands,
    ffi::Array<ffi::String> tile_compute_unsupported_reasons,
    int64_t cb_requirement_count, int64_t cb_l1_bytes, int64_t semaphore_count,
    int64_t communication_edge_count) {
  auto n = ffi::make_object<TTResourceDemandNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->core_group = std::move(core_group);
  n->core_group_index = core_group_index;
  n->tile_compute_fanout_demands = std::move(tile_compute_fanout_demands);
  n->tile_compute_materialization_demands =
      std::move(tile_compute_materialization_demands);
  n->tile_compute_unsupported_reasons =
      std::move(tile_compute_unsupported_reasons);
  n->cb_requirement_count = cb_requirement_count;
  n->cb_l1_bytes = cb_l1_bytes;
  n->semaphore_count = semaphore_count;
  n->communication_edge_count = communication_edge_count;
  data_ = std::move(n);
}

void TTResourcePressureReportNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTResourcePressureReportNode>()
      .def_ro("name", &TTResourcePressureReportNode::name)
      .def_ro("kernel_name", &TTResourcePressureReportNode::kernel_name)
      .def_ro("core_group", &TTResourcePressureReportNode::core_group)
      .def_ro("core_group_index",
              &TTResourcePressureReportNode::core_group_index)
      .def_ro("tile_compute_unsupported_reasons",
              &TTResourcePressureReportNode::tile_compute_unsupported_reasons)
      .def_ro("required_materializations",
              &TTResourcePressureReportNode::required_materializations)
      .def_ro("per_core_cb_id_pressure",
              &TTResourcePressureReportNode::per_core_cb_id_pressure)
      .def_ro("per_core_cb_l1_bytes",
              &TTResourcePressureReportNode::per_core_cb_l1_bytes)
      .def_ro("per_core_l1_buffer_bytes",
              &TTResourcePressureReportNode::per_core_l1_buffer_bytes)
      .def_ro("max_simultaneous_l1_bytes",
              &TTResourcePressureReportNode::max_simultaneous_l1_bytes)
      .def_ro("cb_id_limit", &TTResourcePressureReportNode::cb_id_limit)
      .def_ro("worker_l1_budget_bytes",
              &TTResourcePressureReportNode::worker_l1_budget_bytes)
      .def_ro("l1_alignment_bytes",
              &TTResourcePressureReportNode::l1_alignment_bytes)
      .def_ro("per_core_cb_l1_aligned_bytes",
              &TTResourcePressureReportNode::per_core_cb_l1_aligned_bytes)
      .def_ro("l1_alignment_waste_bytes",
              &TTResourcePressureReportNode::l1_alignment_waste_bytes)
      .def_ro("core_grid_requirement",
              &TTResourcePressureReportNode::core_grid_requirement)
      .def_ro("dram_view_requirement",
              &TTResourcePressureReportNode::dram_view_requirement)
      .def_ro("unsupported_reasons",
              &TTResourcePressureReportNode::unsupported_reasons);
}

TTResourcePressureReport::TTResourcePressureReport(
    ffi::String name, ffi::String kernel_name, ffi::String core_group,
    int64_t core_group_index,
    ffi::Array<ffi::String> tile_compute_unsupported_reasons,
    ffi::Array<TTTileComputeMaterializationDemand> required_materializations,
    int64_t per_core_cb_id_pressure, int64_t per_core_cb_l1_bytes,
    int64_t per_core_l1_buffer_bytes, int64_t max_simultaneous_l1_bytes,
    int64_t cb_id_limit, int64_t worker_l1_budget_bytes,
    int64_t l1_alignment_bytes, int64_t per_core_cb_l1_aligned_bytes,
    int64_t l1_alignment_waste_bytes, ffi::String core_grid_requirement,
    ffi::String dram_view_requirement,
    ffi::Array<ffi::String> unsupported_reasons) {
  auto n = ffi::make_object<TTResourcePressureReportNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->core_group = std::move(core_group);
  n->core_group_index = core_group_index;
  n->tile_compute_unsupported_reasons =
      std::move(tile_compute_unsupported_reasons);
  n->required_materializations = std::move(required_materializations);
  n->per_core_cb_id_pressure = per_core_cb_id_pressure;
  n->per_core_cb_l1_bytes = per_core_cb_l1_bytes;
  n->per_core_l1_buffer_bytes = per_core_l1_buffer_bytes;
  n->max_simultaneous_l1_bytes = max_simultaneous_l1_bytes;
  n->cb_id_limit = cb_id_limit;
  n->worker_l1_budget_bytes = worker_l1_budget_bytes;
  n->l1_alignment_bytes = l1_alignment_bytes;
  n->per_core_cb_l1_aligned_bytes = per_core_cb_l1_aligned_bytes;
  n->l1_alignment_waste_bytes = l1_alignment_waste_bytes;
  n->core_grid_requirement = std::move(core_grid_requirement);
  n->dram_view_requirement = std::move(dram_view_requirement);
  n->unsupported_reasons = std::move(unsupported_reasons);
  data_ = std::move(n);
}

void TTBlockPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTBlockPlanNode>()
      .def_ro("name", &TTBlockPlanNode::name)
      .def_ro("placement_kind", &TTBlockPlanNode::placement_kind)
      .def_ro("task_indices", &TTBlockPlanNode::task_indices)
      .def_ro("core_group", &TTBlockPlanNode::core_group)
      .def_ro("core_group_index", &TTBlockPlanNode::core_group_index);
}

TTBlockPlan::TTBlockPlan(ffi::String name, ffi::String placement_kind,
                         ffi::Array<Integer> task_indices,
                         ffi::String core_group, int64_t core_group_index) {
  auto n = ffi::make_object<TTBlockPlanNode>();
  n->name = std::move(name);
  n->placement_kind = std::move(placement_kind);
  n->task_indices = std::move(task_indices);
  n->core_group = std::move(core_group);
  n->core_group_index = core_group_index;
  data_ = std::move(n);
}

void TTKernelPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelPlanNode>()
      .def_ro("name", &TTKernelPlanNode::name)
      .def_ro("kind", &TTKernelPlanNode::kind)
      .def_ro("core_type", &TTKernelPlanNode::core_type)
      .def_ro("block_plan_index", &TTKernelPlanNode::block_plan_index)
      .def_ro("abi_plan_index", &TTKernelPlanNode::abi_plan_index);
}

TTKernelPlan::TTKernelPlan(ffi::String name, ffi::String kind,
                           ffi::String core_type, int64_t block_plan_index,
                           int64_t abi_plan_index) {
  auto n = ffi::make_object<TTKernelPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->core_type = std::move(core_type);
  n->block_plan_index = block_plan_index;
  n->abi_plan_index = abi_plan_index;
  data_ = std::move(n);
}

void TTKernelLaunchSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelLaunchSpecNode>()
      .def_ro("core_type", &TTKernelLaunchSpecNode::core_type)
      .def_ro("processor", &TTKernelLaunchSpecNode::processor)
      .def_ro("noc", &TTKernelLaunchSpecNode::noc);
}

TTKernelLaunchSpec::TTKernelLaunchSpec(ffi::String core_type,
                                       ffi::String processor, ffi::String noc) {
  auto n = ffi::make_object<TTKernelLaunchSpecNode>();
  n->core_type = std::move(core_type);
  n->processor = std::move(processor);
  n->noc = std::move(noc);
  data_ = std::move(n);
}

void TTKernelDefineNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelDefineNode>()
      .def_ro("name", &TTKernelDefineNode::name)
      .def_ro("value", &TTKernelDefineNode::value);
}

TTKernelDefine::TTKernelDefine(ffi::String name, ffi::String value) {
  auto n = ffi::make_object<TTKernelDefineNode>();
  n->name = std::move(name);
  n->value = std::move(value);
  data_ = std::move(n);
}

void TTKernelNamedCompileArgNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelNamedCompileArgNode>()
      .def_ro("name", &TTKernelNamedCompileArgNode::name)
      .def_ro("value", &TTKernelNamedCompileArgNode::value);
}

TTKernelNamedCompileArg::TTKernelNamedCompileArg(ffi::String name,
                                                 int64_t value) {
  auto n = ffi::make_object<TTKernelNamedCompileArgNode>();
  n->name = std::move(name);
  n->value = value;
  data_ = std::move(n);
}

void TTKernelComputeConfigNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelComputeConfigNode>()
      .def_ro("math_fidelity", &TTKernelComputeConfigNode::math_fidelity)
      .def_ro("fp32_dest_acc_en", &TTKernelComputeConfigNode::fp32_dest_acc_en)
      .def_ro("dst_full_sync_en", &TTKernelComputeConfigNode::dst_full_sync_en)
      .def_ro("math_approx_mode", &TTKernelComputeConfigNode::math_approx_mode)
      .def_ro("unpack_to_dest_mode",
              &TTKernelComputeConfigNode::unpack_to_dest_mode)
      .def_ro("bfp8_pack_precise",
              &TTKernelComputeConfigNode::bfp8_pack_precise)
      .def_ro("defines", &TTKernelComputeConfigNode::defines)
      .def_ro("named_compile_args",
              &TTKernelComputeConfigNode::named_compile_args)
      .def_ro("clear_accum", &TTKernelComputeConfigNode::clear_accum)
      .def_ro("k_pack", &TTKernelComputeConfigNode::k_pack)
      .def_ro("wg_wait", &TTKernelComputeConfigNode::wg_wait)
      .def_ro("policy_type", &TTKernelComputeConfigNode::policy_type)
      .def_ro("policy_name", &TTKernelComputeConfigNode::policy_name);
}

TTKernelComputeConfig::TTKernelComputeConfig(
    ffi::String math_fidelity, bool fp32_dest_acc_en, bool dst_full_sync_en,
    bool math_approx_mode, ffi::Array<ffi::String> unpack_to_dest_mode,
    bool bfp8_pack_precise, ffi::Array<TTKernelDefine> defines,
    ffi::Array<TTKernelNamedCompileArg> named_compile_args, bool clear_accum,
    int64_t k_pack, int64_t wg_wait, int64_t policy_type,
    ffi::String policy_name) {
  auto n = ffi::make_object<TTKernelComputeConfigNode>();
  n->math_fidelity = std::move(math_fidelity);
  n->fp32_dest_acc_en = fp32_dest_acc_en;
  n->dst_full_sync_en = dst_full_sync_en;
  n->math_approx_mode = math_approx_mode;
  n->unpack_to_dest_mode = std::move(unpack_to_dest_mode);
  n->bfp8_pack_precise = bfp8_pack_precise;
  n->defines = std::move(defines);
  n->named_compile_args = std::move(named_compile_args);
  n->clear_accum = clear_accum;
  n->k_pack = k_pack;
  n->wg_wait = wg_wait;
  n->policy_type = policy_type;
  n->policy_name = std::move(policy_name);
  data_ = std::move(n);
}

void TTPerWorkArgSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTPerWorkArgSpecNode>()
      .def_ro("arg_kind", &TTPerWorkArgSpecNode::arg_kind)
      .def_ro("arg_identity", &TTPerWorkArgSpecNode::arg_identity)
      .def_ro("buffer", &TTPerWorkArgSpecNode::buffer)
      .def_ro("descriptor_kind", &TTPerWorkArgSpecNode::descriptor_kind)
      .def_ro("value_source", &TTPerWorkArgSpecNode::value_source)
      .def_ro("constant_value", &TTPerWorkArgSpecNode::constant_value)
      .def_ro("access_region", &TTPerWorkArgSpecNode::access_region)
      .def_ro("access_region_index",
              &TTPerWorkArgSpecNode::access_region_index)
      .def_ro("index_buffer", &TTPerWorkArgSpecNode::index_buffer)
      .def_ro("index_value_scale",
              &TTPerWorkArgSpecNode::index_value_scale);
}

TTPerWorkArgSpec::TTPerWorkArgSpec(ffi::String arg_kind,
                                   ffi::String arg_identity, ffi::String buffer,
                                   ffi::String descriptor_kind,
                                   ffi::String value_source,
                                   int64_t constant_value)
    : TTPerWorkArgSpec(std::move(arg_kind), std::move(arg_identity),
                       std::move(buffer), std::move(descriptor_kind),
                       std::move(value_source), constant_value,
                       ffi::String(), -1, ffi::String(), 1) {}

TTPerWorkArgSpec::TTPerWorkArgSpec(ffi::String arg_kind,
                                   ffi::String arg_identity, ffi::String buffer,
                                   ffi::String descriptor_kind,
                                   ffi::String value_source,
                                   int64_t constant_value,
                                   ffi::String access_region,
                                   int64_t access_region_index)
    : TTPerWorkArgSpec(std::move(arg_kind), std::move(arg_identity),
                       std::move(buffer), std::move(descriptor_kind),
                       std::move(value_source), constant_value,
                       std::move(access_region), access_region_index,
                       ffi::String(), 1) {}

TTPerWorkArgSpec::TTPerWorkArgSpec(ffi::String arg_kind,
                                   ffi::String arg_identity, ffi::String buffer,
                                   ffi::String descriptor_kind,
                                   ffi::String value_source,
                                   int64_t constant_value,
                                   ffi::String access_region,
                                   int64_t access_region_index,
                                   ffi::String index_buffer,
                                   int64_t index_value_scale) {
  auto n = ffi::make_object<TTPerWorkArgSpecNode>();
  n->arg_kind = std::move(arg_kind);
  n->arg_identity = std::move(arg_identity);
  n->buffer = std::move(buffer);
  n->descriptor_kind = std::move(descriptor_kind);
  n->value_source = std::move(value_source);
  n->constant_value = constant_value;
  n->access_region = std::move(access_region);
  n->access_region_index = access_region_index;
  n->index_buffer = std::move(index_buffer);
  n->index_value_scale = index_value_scale <= 0 ? 1 : index_value_scale;
  data_ = std::move(n);
}

void TTKernelNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelNode>()
      .def_ro("name", &TTKernelNode::name)
      .def_ro("kind", &TTKernelNode::kind)
      .def_ro("core_type", &TTKernelNode::core_type)
      .def_ro("abi_plan_index", &TTKernelNode::abi_plan_index)
      .def_ro("launch_spec", &TTKernelNode::launch_spec)
      .def_ro("compute_config", &TTKernelNode::compute_config)
      .def_ro("per_work_arg_specs", &TTKernelNode::per_work_arg_specs);
}

TTKernel::TTKernel(ffi::String name, ffi::String kind, ffi::String core_type,
                   int64_t abi_plan_index, TTKernelLaunchSpec launch_spec,
                   TTKernelComputeConfig compute_config,
                   ffi::Array<TTPerWorkArgSpec> per_work_arg_specs) {
  auto n = ffi::make_object<TTKernelNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->core_type = std::move(core_type);
  n->abi_plan_index = abi_plan_index;
  n->launch_spec = std::move(launch_spec);
  n->compute_config = std::move(compute_config);
  n->per_work_arg_specs = std::move(per_work_arg_specs);
  data_ = std::move(n);
}

void TTCoreGroupNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTCoreGroupNode>()
      .def_ro("name", &TTCoreGroupNode::name)
      .def_ro("logical_grid_x", &TTCoreGroupNode::logical_grid_x)
      .def_ro("logical_grid_y", &TTCoreGroupNode::logical_grid_y)
      .def_ro("logical_grid_z", &TTCoreGroupNode::logical_grid_z)
      .def_ro("linearization", &TTCoreGroupNode::linearization)
      .def_ro("physical_cores", &TTCoreGroupNode::physical_cores)
      .def_ro("work_packets", &TTCoreGroupNode::work_packets);
}

TTCoreGroup::TTCoreGroup(ffi::String name, int64_t logical_grid_x,
                         int64_t logical_grid_y, ffi::String linearization,
                         ffi::Array<ffi::Any> physical_cores,
                         ffi::Array<ffi::Any> work_packets,
                         int64_t logical_grid_z) {
  auto n = ffi::make_object<TTCoreGroupNode>();
  n->name = std::move(name);
  n->logical_grid_x = logical_grid_x;
  n->logical_grid_y = logical_grid_y;
  n->logical_grid_z = logical_grid_z;
  n->linearization = std::move(linearization);
  n->physical_cores = std::move(physical_cores);
  n->work_packets = std::move(work_packets);
  data_ = std::move(n);
}

void TTCBPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTCBPlanNode>()
      .def_ro("name", &TTCBPlanNode::name)
      .def_ro("cb_id", &TTCBPlanNode::cb_id)
      .def_ro("resource_class", &TTCBPlanNode::resource_class)
      .def_ro("num_pages", &TTCBPlanNode::num_pages)
      .def_ro("page_size_bytes", &TTCBPlanNode::page_size_bytes)
      .def_ro("data_format", &TTCBPlanNode::data_format)
      .def_ro("initial_reserve_pages", &TTCBPlanNode::initial_reserve_pages)
      .def_ro("flow_class", &TTCBPlanNode::flow_class)
      .def_ro("publish_pages_per_event", &TTCBPlanNode::publish_pages_per_event)
      .def_ro("consume_pages_per_event", &TTCBPlanNode::consume_pages_per_event)
      .def_ro("lifetime_begin", &TTCBPlanNode::lifetime_begin)
      .def_ro("lifetime_end", &TTCBPlanNode::lifetime_end)
      .def_ro("requirement_names", &TTCBPlanNode::requirement_names)
      .def_ro("requirement_indices", &TTCBPlanNode::requirement_indices);
}

TTCBPlan::TTCBPlan(ffi::String name, int64_t cb_id, ffi::String resource_class,
                   int64_t num_pages, int64_t page_size_bytes,
                   ffi::String data_format, int64_t initial_reserve_pages,
                   ffi::String flow_class, int64_t publish_pages_per_event,
                   int64_t consume_pages_per_event, int64_t lifetime_begin,
                   int64_t lifetime_end,
                   ffi::Array<ffi::String> requirement_names,
                   ffi::Array<Integer> requirement_indices) {
  auto n = ffi::make_object<TTCBPlanNode>();
  n->name = std::move(name);
  n->cb_id = cb_id;
  n->resource_class = std::move(resource_class);
  n->num_pages = num_pages;
  n->page_size_bytes = page_size_bytes;
  n->data_format = std::move(data_format);
  n->initial_reserve_pages = initial_reserve_pages;
  n->flow_class = std::move(flow_class);
  n->publish_pages_per_event = publish_pages_per_event;
  n->consume_pages_per_event = consume_pages_per_event;
  n->lifetime_begin = lifetime_begin;
  n->lifetime_end = lifetime_end;
  n->requirement_names = std::move(requirement_names);
  n->requirement_indices = std::move(requirement_indices);
  data_ = std::move(n);
}

void TTTransportPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTTransportPlanNode>()
      .def_ro("name", &TTTransportPlanNode::name)
      .def_ro("kind", &TTTransportPlanNode::kind)
      .def_ro("source_task_index", &TTTransportPlanNode::source_task_index)
      .def_ro("target_task_index", &TTTransportPlanNode::target_task_index)
      .def_ro("value_kind", &TTTransportPlanNode::value_kind)
      .def_ro("delivery_kind", &TTTransportPlanNode::delivery_kind)
      .def_ro("subject", &TTTransportPlanNode::subject);
}

TTTransportPlan::TTTransportPlan(ffi::String name, ffi::String kind,
                                 int64_t source_task_index,
                                 int64_t target_task_index,
                                 ffi::String value_kind,
                                 ffi::String delivery_kind,
                                 ffi::String subject) {
  auto n = ffi::make_object<TTTransportPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->value_kind = std::move(value_kind);
  n->delivery_kind = std::move(delivery_kind);
  n->subject = std::move(subject);
  data_ = std::move(n);
}

void TTSyncPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTSyncPlanNode>()
      .def_ro("name", &TTSyncPlanNode::name)
      .def_ro("kind", &TTSyncPlanNode::kind)
      .def_ro("source_task_index", &TTSyncPlanNode::source_task_index)
      .def_ro("target_task_index", &TTSyncPlanNode::target_task_index)
      .def_ro("ordering_kind", &TTSyncPlanNode::ordering_kind)
      .def_ro("completion_kind", &TTSyncPlanNode::completion_kind);
}

TTSyncPlan::TTSyncPlan(ffi::String name, ffi::String kind,
                       int64_t source_task_index, int64_t target_task_index,
                       ffi::String ordering_kind, ffi::String completion_kind) {
  auto n = ffi::make_object<TTSyncPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->ordering_kind = std::move(ordering_kind);
  n->completion_kind = std::move(completion_kind);
  data_ = std::move(n);
}

void TTSemaphorePlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTSemaphorePlanNode>()
      .def_ro("name", &TTSemaphorePlanNode::name)
      .def_ro("kind", &TTSemaphorePlanNode::kind)
      .def_ro("semaphore_id", &TTSemaphorePlanNode::semaphore_id)
      .def_ro("initial_value", &TTSemaphorePlanNode::initial_value)
      .def_ro("core_type", &TTSemaphorePlanNode::core_type)
      .def_ro("source_task_index", &TTSemaphorePlanNode::source_task_index)
      .def_ro("target_task_index", &TTSemaphorePlanNode::target_task_index)
      .def_ro("core_ranges", &TTSemaphorePlanNode::core_ranges);
}

TTSemaphorePlan::TTSemaphorePlan(ffi::String name, ffi::String kind,
                                 int64_t semaphore_id, int64_t initial_value,
                                 ffi::String core_type,
                                 int64_t source_task_index,
                                 int64_t target_task_index,
                                 ffi::Array<ffi::Any> core_ranges) {
  auto n = ffi::make_object<TTSemaphorePlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->semaphore_id = semaphore_id;
  n->initial_value = initial_value;
  n->core_type = std::move(core_type);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->core_ranges = std::move(core_ranges);
  data_ = std::move(n);
}

void TTComputeSyncPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeSyncPlanNode>()
      .def_ro("name", &TTComputeSyncPlanNode::name)
      .def_ro("kind", &TTComputeSyncPlanNode::kind)
      .def_ro("source_task_index", &TTComputeSyncPlanNode::source_task_index)
      .def_ro("target_task_index", &TTComputeSyncPlanNode::target_task_index)
      .def_ro("ordering_kind", &TTComputeSyncPlanNode::ordering_kind)
      .def_ro("materialization_kind",
              &TTComputeSyncPlanNode::materialization_kind);
}

TTComputeSyncPlan::TTComputeSyncPlan(ffi::String name, ffi::String kind,
                                     int64_t source_task_index,
                                     int64_t target_task_index,
                                     ffi::String ordering_kind,
                                     ffi::String materialization_kind) {
  auto n = ffi::make_object<TTComputeSyncPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->ordering_kind = std::move(ordering_kind);
  n->materialization_kind = std::move(materialization_kind);
  data_ = std::move(n);
}

void TTDstLayoutPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTDstLayoutPlanNode>()
      .def_ro("name", &TTDstLayoutPlanNode::name)
      .def_ro("buffer", &TTDstLayoutPlanNode::buffer)
      .def_ro("layout", &TTDstLayoutPlanNode::layout)
      .def_ro("memory_space", &TTDstLayoutPlanNode::memory_space)
      .def_ro("page_size_bytes", &TTDstLayoutPlanNode::page_size_bytes);
}

TTDstLayoutPlan::TTDstLayoutPlan(ffi::String name, ffi::String buffer,
                                 ffi::String layout, ffi::String memory_space,
                                 int64_t page_size_bytes) {
  auto n = ffi::make_object<TTDstLayoutPlanNode>();
  n->name = std::move(name);
  n->buffer = std::move(buffer);
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->page_size_bytes = page_size_bytes;
  data_ = std::move(n);
}

void TTLiveFormPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTLiveFormPlanNode>()
      .def_ro("name", &TTLiveFormPlanNode::name)
      .def_ro("logical_value", &TTLiveFormPlanNode::logical_value)
      .def_ro("spatial_live_value", &TTLiveFormPlanNode::spatial_live_value)
      .def_ro("spatial_live_value_index",
              &TTLiveFormPlanNode::spatial_live_value_index)
      .def_ro("producer_kernel", &TTLiveFormPlanNode::producer_kernel)
      .def_ro("physical_form", &TTLiveFormPlanNode::physical_form)
      .def_ro("execution_topology", &TTLiveFormPlanNode::execution_topology)
      .def_ro("physical_local_extent",
              &TTLiveFormPlanNode::physical_local_extent)
      .def_ro("logical_element_count",
              &TTLiveFormPlanNode::logical_element_count)
      .def_ro("ownership_kind", &TTLiveFormPlanNode::ownership_kind);
}

TTLiveFormPlan::TTLiveFormPlan(
    ffi::String name, ffi::String logical_value, ffi::String spatial_live_value,
    int64_t spatial_live_value_index, ffi::String producer_kernel,
    ffi::String physical_form, ffi::String execution_topology,
    int64_t physical_local_extent, int64_t logical_element_count,
    ffi::String ownership_kind) {
  auto n = ffi::make_object<TTLiveFormPlanNode>();
  n->name = std::move(name);
  n->logical_value = std::move(logical_value);
  n->spatial_live_value = std::move(spatial_live_value);
  n->spatial_live_value_index = spatial_live_value_index;
  n->producer_kernel = std::move(producer_kernel);
  n->physical_form = std::move(physical_form);
  n->execution_topology = std::move(execution_topology);
  n->physical_local_extent = physical_local_extent;
  n->logical_element_count = logical_element_count;
  n->ownership_kind = std::move(ownership_kind);
  data_ = std::move(n);
}

void TTMaterializationPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTMaterializationPlanNode>()
      .def_ro("name", &TTMaterializationPlanNode::name)
      .def_ro("source_live_form", &TTMaterializationPlanNode::source_live_form)
      .def_ro("materialization_boundary",
              &TTMaterializationPlanNode::materialization_boundary)
      .def_ro("materialization_boundary_index",
              &TTMaterializationPlanNode::materialization_boundary_index)
      .def_ro("target_buffer", &TTMaterializationPlanNode::target_buffer)
      .def_ro("host_buffer", &TTMaterializationPlanNode::host_buffer)
      .def_ro("target_kernel", &TTMaterializationPlanNode::target_kernel)
      .def_ro("bridge_kind", &TTMaterializationPlanNode::bridge_kind)
      .def_ro("materialization_kind",
              &TTMaterializationPlanNode::materialization_kind)
      .def_ro("materialization_protocol",
              &TTMaterializationPlanNode::materialization_protocol)
      .def_ro("publication_protocol",
              &TTMaterializationPlanNode::publication_protocol)
      .def_ro("required_cb_plan_indices",
              &TTMaterializationPlanNode::required_cb_plan_indices)
      .def_ro("required_sync_plan_indices",
              &TTMaterializationPlanNode::required_sync_plan_indices)
      .def_ro("produced_live_form",
              &TTMaterializationPlanNode::produced_live_form);
}

TTMaterializationPlan::TTMaterializationPlan(
    ffi::String name, ffi::String source_live_form,
    ffi::String materialization_boundary,
    int64_t materialization_boundary_index, ffi::String target_buffer,
    ffi::String host_buffer, ffi::String target_kernel, ffi::String bridge_kind,
    ffi::String materialization_kind, ffi::String materialization_protocol,
    ffi::String publication_protocol,
    ffi::Array<Integer> required_cb_plan_indices,
    ffi::Array<Integer> required_sync_plan_indices,
    ffi::String produced_live_form) {
  auto n = ffi::make_object<TTMaterializationPlanNode>();
  n->name = std::move(name);
  n->source_live_form = std::move(source_live_form);
  n->materialization_boundary = std::move(materialization_boundary);
  n->materialization_boundary_index = materialization_boundary_index;
  n->target_buffer = std::move(target_buffer);
  n->host_buffer = std::move(host_buffer);
  n->target_kernel = std::move(target_kernel);
  n->bridge_kind = std::move(bridge_kind);
  n->materialization_kind = std::move(materialization_kind);
  n->materialization_protocol = std::move(materialization_protocol);
  n->publication_protocol = std::move(publication_protocol);
  n->required_cb_plan_indices = std::move(required_cb_plan_indices);
  n->required_sync_plan_indices = std::move(required_sync_plan_indices);
  n->produced_live_form = std::move(produced_live_form);
  data_ = std::move(n);
}

void TTConsumerBindingPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTConsumerBindingPlanNode>()
      .def_ro("name", &TTConsumerBindingPlanNode::name)
      .def_ro("consumer_kernel", &TTConsumerBindingPlanNode::consumer_kernel)
      .def_ro("consumer_op_kind", &TTConsumerBindingPlanNode::consumer_op_kind)
      .def_ro("source_live_form", &TTConsumerBindingPlanNode::source_live_form)
      .def_ro("live_value_edge", &TTConsumerBindingPlanNode::live_value_edge)
      .def_ro("live_value_edge_index",
              &TTConsumerBindingPlanNode::live_value_edge_index)
      .def_ro("accepts_distributed_slice",
              &TTConsumerBindingPlanNode::accepts_distributed_slice)
      .def_ro("requires_full_logical_tile",
              &TTConsumerBindingPlanNode::requires_full_logical_tile)
      .def_ro("abi_plan_index", &TTConsumerBindingPlanNode::abi_plan_index)
      .def_ro("target_buffer", &TTConsumerBindingPlanNode::target_buffer)
      .def_ro("materialization_plan",
              &TTConsumerBindingPlanNode::materialization_plan);
}

TTConsumerBindingPlan::TTConsumerBindingPlan(
    ffi::String name, ffi::String consumer_kernel, ffi::String consumer_op_kind,
    ffi::String source_live_form, ffi::String live_value_edge,
    int64_t live_value_edge_index, bool accepts_distributed_slice,
    bool requires_full_logical_tile, int64_t abi_plan_index,
    ffi::String target_buffer, ffi::String materialization_plan) {
  auto n = ffi::make_object<TTConsumerBindingPlanNode>();
  n->name = std::move(name);
  n->consumer_kernel = std::move(consumer_kernel);
  n->consumer_op_kind = std::move(consumer_op_kind);
  n->source_live_form = std::move(source_live_form);
  n->live_value_edge = std::move(live_value_edge);
  n->live_value_edge_index = live_value_edge_index;
  n->accepts_distributed_slice = accepts_distributed_slice;
  n->requires_full_logical_tile = requires_full_logical_tile;
  n->abi_plan_index = abi_plan_index;
  n->target_buffer = std::move(target_buffer);
  n->materialization_plan = std::move(materialization_plan);
  data_ = std::move(n);
}

void TTExactCBVirtualValueNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExactCBVirtualValueNode>()
      .def_ro("name", &TTExactCBVirtualValueNode::name)
      .def_ro("logical_value", &TTExactCBVirtualValueNode::logical_value)
      .def_ro("live_form", &TTExactCBVirtualValueNode::live_form)
      .def_ro("live_form_index", &TTExactCBVirtualValueNode::live_form_index)
      .def_ro("producer_kernel", &TTExactCBVirtualValueNode::producer_kernel)
      .def_ro("producer_event", &TTExactCBVirtualValueNode::producer_event)
      .def_ro("event_lifetime_kind",
              &TTExactCBVirtualValueNode::event_lifetime_kind)
      .def_ro("loop_role", &TTExactCBVirtualValueNode::loop_role)
      .def_ro("num_pages", &TTExactCBVirtualValueNode::num_pages)
      .def_ro("page_size_bytes", &TTExactCBVirtualValueNode::page_size_bytes)
      .def_ro("data_format", &TTExactCBVirtualValueNode::data_format);
}

TTExactCBVirtualValue::TTExactCBVirtualValue(
    ffi::String name, ffi::String logical_value, ffi::String live_form,
    int64_t live_form_index, ffi::String producer_kernel,
    ffi::String producer_event, ffi::String event_lifetime_kind,
    ffi::String loop_role, int64_t num_pages, int64_t page_size_bytes,
    ffi::String data_format) {
  auto n = ffi::make_object<TTExactCBVirtualValueNode>();
  n->name = std::move(name);
  n->logical_value = std::move(logical_value);
  n->live_form = std::move(live_form);
  n->live_form_index = live_form_index;
  n->producer_kernel = std::move(producer_kernel);
  n->producer_event = std::move(producer_event);
  n->event_lifetime_kind = std::move(event_lifetime_kind);
  n->loop_role = std::move(loop_role);
  n->num_pages = num_pages;
  n->page_size_bytes = page_size_bytes;
  n->data_format = std::move(data_format);
  data_ = std::move(n);
}

void TTExactCBUseEventNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExactCBUseEventNode>()
      .def_ro("name", &TTExactCBUseEventNode::name)
      .def_ro("virtual_value", &TTExactCBUseEventNode::virtual_value)
      .def_ro("virtual_value_index",
              &TTExactCBUseEventNode::virtual_value_index)
      .def_ro("consumer_kernel", &TTExactCBUseEventNode::consumer_kernel)
      .def_ro("consumer_event", &TTExactCBUseEventNode::consumer_event)
      .def_ro("operand_role", &TTExactCBUseEventNode::operand_role)
      .def_ro("program_point", &TTExactCBUseEventNode::program_point)
      .def_ro("requires_full_logical_tile",
              &TTExactCBUseEventNode::requires_full_logical_tile)
      .def_ro("borrow_kind", &TTExactCBUseEventNode::borrow_kind);
}

TTExactCBUseEvent::TTExactCBUseEvent(
    ffi::String name, ffi::String virtual_value, int64_t virtual_value_index,
    ffi::String consumer_kernel, ffi::String consumer_event,
    ffi::String operand_role, int64_t program_point,
    bool requires_full_logical_tile, ffi::String borrow_kind) {
  auto n = ffi::make_object<TTExactCBUseEventNode>();
  n->name = std::move(name);
  n->virtual_value = std::move(virtual_value);
  n->virtual_value_index = virtual_value_index;
  n->consumer_kernel = std::move(consumer_kernel);
  n->consumer_event = std::move(consumer_event);
  n->operand_role = std::move(operand_role);
  n->program_point = program_point;
  n->requires_full_logical_tile = requires_full_logical_tile;
  n->borrow_kind = std::move(borrow_kind);
  data_ = std::move(n);
}

void TTExactCBLiveIntervalNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExactCBLiveIntervalNode>()
      .def_ro("name", &TTExactCBLiveIntervalNode::name)
      .def_ro("virtual_value", &TTExactCBLiveIntervalNode::virtual_value)
      .def_ro("virtual_value_index",
              &TTExactCBLiveIntervalNode::virtual_value_index)
      .def_ro("begin_point", &TTExactCBLiveIntervalNode::begin_point)
      .def_ro("end_point", &TTExactCBLiveIntervalNode::end_point)
      .def_ro("live_in", &TTExactCBLiveIntervalNode::live_in)
      .def_ro("live_out", &TTExactCBLiveIntervalNode::live_out)
      .def_ro("loop_carried", &TTExactCBLiveIntervalNode::loop_carried)
      .def_ro("interference_class",
              &TTExactCBLiveIntervalNode::interference_class);
}

TTExactCBLiveInterval::TTExactCBLiveInterval(
    ffi::String name, ffi::String virtual_value, int64_t virtual_value_index,
    int64_t begin_point, int64_t end_point, bool live_in, bool live_out,
    bool loop_carried, ffi::String interference_class) {
  auto n = ffi::make_object<TTExactCBLiveIntervalNode>();
  n->name = std::move(name);
  n->virtual_value = std::move(virtual_value);
  n->virtual_value_index = virtual_value_index;
  n->begin_point = begin_point;
  n->end_point = end_point;
  n->live_in = live_in;
  n->live_out = live_out;
  n->loop_carried = loop_carried;
  n->interference_class = std::move(interference_class);
  data_ = std::move(n);
}

void TTExactCBAllocationNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExactCBAllocationNode>()
      .def_ro("name", &TTExactCBAllocationNode::name)
      .def_ro("virtual_value", &TTExactCBAllocationNode::virtual_value)
      .def_ro("virtual_value_index",
              &TTExactCBAllocationNode::virtual_value_index)
      .def_ro("cb_plan", &TTExactCBAllocationNode::cb_plan)
      .def_ro("cb_plan_index", &TTExactCBAllocationNode::cb_plan_index)
      .def_ro("physical_cb_id", &TTExactCBAllocationNode::physical_cb_id)
      .def_ro("page_count", &TTExactCBAllocationNode::page_count)
      .def_ro("release_program_point",
              &TTExactCBAllocationNode::release_program_point)
      .def_ro("release_reason", &TTExactCBAllocationNode::release_reason);
}

TTExactCBAllocation::TTExactCBAllocation(
    ffi::String name, ffi::String virtual_value, int64_t virtual_value_index,
    ffi::String cb_plan, int64_t cb_plan_index, int64_t physical_cb_id,
    int64_t page_count, int64_t release_program_point,
    ffi::String release_reason) {
  auto n = ffi::make_object<TTExactCBAllocationNode>();
  n->name = std::move(name);
  n->virtual_value = std::move(virtual_value);
  n->virtual_value_index = virtual_value_index;
  n->cb_plan = std::move(cb_plan);
  n->cb_plan_index = cb_plan_index;
  n->physical_cb_id = physical_cb_id;
  n->page_count = page_count;
  n->release_program_point = release_program_point;
  n->release_reason = std::move(release_reason);
  data_ = std::move(n);
}

void TTExactCBReleaseEventNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExactCBReleaseEventNode>()
      .def_ro("name", &TTExactCBReleaseEventNode::name)
      .def_ro("allocation", &TTExactCBReleaseEventNode::allocation)
      .def_ro("allocation_index",
              &TTExactCBReleaseEventNode::allocation_index)
      .def_ro("cb_plan", &TTExactCBReleaseEventNode::cb_plan)
      .def_ro("cb_plan_index", &TTExactCBReleaseEventNode::cb_plan_index)
      .def_ro("program_point", &TTExactCBReleaseEventNode::program_point)
      .def_ro("page_count", &TTExactCBReleaseEventNode::page_count)
      .def_ro("reason", &TTExactCBReleaseEventNode::reason);
}

TTExactCBReleaseEvent::TTExactCBReleaseEvent(
    ffi::String name, ffi::String allocation, int64_t allocation_index,
    ffi::String cb_plan, int64_t cb_plan_index, int64_t program_point,
    int64_t page_count, ffi::String reason) {
  auto n = ffi::make_object<TTExactCBReleaseEventNode>();
  n->name = std::move(name);
  n->allocation = std::move(allocation);
  n->allocation_index = allocation_index;
  n->cb_plan = std::move(cb_plan);
  n->cb_plan_index = cb_plan_index;
  n->program_point = program_point;
  n->page_count = page_count;
  n->reason = std::move(reason);
  data_ = std::move(n);
}

void TTRuntimeArgSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTRuntimeArgSpecNode>()
      .def_ro("name", &TTRuntimeArgSpecNode::name)
      .def_ro("kind", &TTRuntimeArgSpecNode::kind)
      .def_ro("dtype", &TTRuntimeArgSpecNode::dtype)
      .def_ro("buffer", &TTRuntimeArgSpecNode::buffer)
      .def_ro("identity", &TTRuntimeArgSpecNode::identity)
      .def_ro("core_x", &TTRuntimeArgSpecNode::core_x)
      .def_ro("core_y", &TTRuntimeArgSpecNode::core_y);
}

TTRuntimeArgSpec::TTRuntimeArgSpec(ffi::String name, ffi::String kind,
                                   ffi::String dtype, ffi::String buffer,
                                   ffi::String identity, int64_t core_x,
                                   int64_t core_y) {
  auto n = ffi::make_object<TTRuntimeArgSpecNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->dtype = std::move(dtype);
  n->buffer = std::move(buffer);
  n->identity = std::move(identity);
  n->core_x = core_x;
  n->core_y = core_y;
  data_ = std::move(n);
}

void TTCompileTimeArgSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTCompileTimeArgSpecNode>()
      .def_ro("name", &TTCompileTimeArgSpecNode::name)
      .def_ro("kind", &TTCompileTimeArgSpecNode::kind)
      .def_ro("dtype", &TTCompileTimeArgSpecNode::dtype)
      .def_ro("offset", &TTCompileTimeArgSpecNode::offset)
      .def_ro("count", &TTCompileTimeArgSpecNode::count)
      .def_ro("buffer", &TTCompileTimeArgSpecNode::buffer)
      .def_ro("segment_role", &TTCompileTimeArgSpecNode::segment_role)
      .def_ro("values", &TTCompileTimeArgSpecNode::values)
      .def_ro("args_config_bits", &TTCompileTimeArgSpecNode::args_config_bits)
      .def_ro("transport_page_size",
              &TTCompileTimeArgSpecNode::transport_page_size)
      .def_ro("layout", &TTCompileTimeArgSpecNode::layout)
      .def_ro("memory_space", &TTCompileTimeArgSpecNode::memory_space)
      .def_ro("host_axis_order", &TTCompileTimeArgSpecNode::host_axis_order)
      .def_ro("transpose_2d", &TTCompileTimeArgSpecNode::transpose_2d);
}

TTCompileTimeArgSpec::TTCompileTimeArgSpec(
    ffi::String name, ffi::String kind, ffi::String dtype, int64_t offset,
    int64_t count, ffi::String buffer, ffi::String segment_role,
    ffi::Array<Integer> values, int64_t args_config_bits,
    int64_t transport_page_size, ffi::String layout, ffi::String memory_space,
    ffi::Array<Integer> host_axis_order, bool transpose_2d) {
  auto n = ffi::make_object<TTCompileTimeArgSpecNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->dtype = std::move(dtype);
  n->offset = offset;
  n->count = count;
  n->buffer = std::move(buffer);
  n->segment_role = std::move(segment_role);
  n->values = std::move(values);
  n->args_config_bits = args_config_bits;
  n->transport_page_size = transport_page_size;
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->host_axis_order = std::move(host_axis_order);
  n->transpose_2d = transpose_2d;
  data_ = std::move(n);
}

void TTAccessorSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTAccessorSpecNode>()
      .def_ro("buffer", &TTAccessorSpecNode::buffer)
      .def_ro("compile_time_arg_offset",
              &TTAccessorSpecNode::compile_time_arg_offset)
      .def_ro("compile_time_arg_count",
              &TTAccessorSpecNode::compile_time_arg_count)
      .def_ro("common_runtime_arg_offset",
              &TTAccessorSpecNode::common_runtime_arg_offset)
      .def_ro("common_runtime_arg_count",
              &TTAccessorSpecNode::common_runtime_arg_count)
      .def_ro("args_config_bits", &TTAccessorSpecNode::args_config_bits)
      .def_ro("transport_page_size", &TTAccessorSpecNode::transport_page_size)
      .def_ro("layout", &TTAccessorSpecNode::layout)
      .def_ro("memory_space", &TTAccessorSpecNode::memory_space)
      .def_ro("host_axis_order", &TTAccessorSpecNode::host_axis_order)
      .def_ro("transpose_2d", &TTAccessorSpecNode::transpose_2d);
}

TTAccessorSpec::TTAccessorSpec(
    ffi::String buffer, int64_t compile_time_arg_offset,
    int64_t compile_time_arg_count, int64_t common_runtime_arg_offset,
    int64_t common_runtime_arg_count, int64_t args_config_bits,
    int64_t transport_page_size, ffi::String layout, ffi::String memory_space,
    ffi::Array<Integer> host_axis_order, bool transpose_2d) {
  auto n = ffi::make_object<TTAccessorSpecNode>();
  n->buffer = std::move(buffer);
  n->compile_time_arg_offset = compile_time_arg_offset;
  n->compile_time_arg_count = compile_time_arg_count;
  n->common_runtime_arg_offset = common_runtime_arg_offset;
  n->common_runtime_arg_count = common_runtime_arg_count;
  n->args_config_bits = args_config_bits;
  n->transport_page_size = transport_page_size;
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->host_axis_order = std::move(host_axis_order);
  n->transpose_2d = transpose_2d;
  data_ = std::move(n);
}

void TTSemaphoreBindingSpecNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTSemaphoreBindingSpecNode>()
      .def_ro("name", &TTSemaphoreBindingSpecNode::name)
      .def_ro("semaphore_id", &TTSemaphoreBindingSpecNode::semaphore_id)
      .def_ro("arg_kind", &TTSemaphoreBindingSpecNode::arg_kind);
}

TTSemaphoreBindingSpec::TTSemaphoreBindingSpec(ffi::String name,
                                               int64_t semaphore_id,
                                               ffi::String arg_kind) {
  auto n = ffi::make_object<TTSemaphoreBindingSpecNode>();
  n->name = std::move(name);
  n->semaphore_id = semaphore_id;
  n->arg_kind = std::move(arg_kind);
  data_ = std::move(n);
}

void TTABIPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTABIPlanNode>()
      .def_ro("name", &TTABIPlanNode::name)
      .def_ro("kernel_name", &TTABIPlanNode::kernel_name)
      .def_ro("runtime_args", &TTABIPlanNode::runtime_args)
      .def_ro("common_runtime_args", &TTABIPlanNode::common_runtime_args)
      .def_ro("compile_time_arg_specs", &TTABIPlanNode::compile_time_arg_specs)
      .def_ro("accessors", &TTABIPlanNode::accessors)
      .def_ro("semaphore_bindings", &TTABIPlanNode::semaphore_bindings);
}

TTABIPlan::TTABIPlan(ffi::String name, ffi::String kernel_name,
                     ffi::Array<TTRuntimeArgSpec> runtime_args,
                     ffi::Array<TTRuntimeArgSpec> common_runtime_args,
                     ffi::Array<TTCompileTimeArgSpec> compile_time_arg_specs,
                     ffi::Array<TTAccessorSpec> accessors,
                     ffi::Array<TTSemaphoreBindingSpec> semaphore_bindings) {
  auto n = ffi::make_object<TTABIPlanNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->runtime_args = std::move(runtime_args);
  n->common_runtime_args = std::move(common_runtime_args);
  n->compile_time_arg_specs = std::move(compile_time_arg_specs);
  n->accessors = std::move(accessors);
  n->semaphore_bindings = std::move(semaphore_bindings);
  data_ = std::move(n);
}

void TTExecutionPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExecutionPlanNode>()
      .def_ro("name", &TTExecutionPlanNode::name)
      .def_ro("kernel_names", &TTExecutionPlanNode::kernel_names)
      .def_ro("phase_indices", &TTExecutionPlanNode::phase_indices);
}

TTExecutionPlan::TTExecutionPlan(ffi::String name,
                                 ffi::Array<ffi::String> kernel_names,
                                 ffi::Array<Integer> phase_indices) {
  auto n = ffi::make_object<TTExecutionPlanNode>();
  n->name = std::move(name);
  n->kernel_names = std::move(kernel_names);
  n->phase_indices = std::move(phase_indices);
  data_ = std::move(n);
}

void TTProgramNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTProgramNode>()
      .def_ro("entry_name", &TTProgramNode::entry_name)
      .def_ro("member_func", &TTProgramNode::member_func)
      .def_ro("mesh_plans", &TTProgramNode::mesh_plans)
      .def_ro("buffer_distribution_plans",
              &TTProgramNode::buffer_distribution_plans)
      .def_ro("tensor_memory_config_plans",
              &TTProgramNode::tensor_memory_config_plans)
      .def_ro("op_sharding_contracts",
              &TTProgramNode::op_sharding_contracts)
      .def_ro("placement_resolution_plans",
              &TTProgramNode::placement_resolution_plans)
      .def_ro("reshard_plans", &TTProgramNode::reshard_plans)
      .def_ro("block_plans", &TTProgramNode::block_plans)
      .def_ro("kernel_plans", &TTProgramNode::kernel_plans)
      .def_ro("compute_op_plans", &TTProgramNode::compute_op_plans)
      .def_ro("transport_plans", &TTProgramNode::transport_plans)
      .def_ro("sync_plans", &TTProgramNode::sync_plans)
      .def_ro("abi_plans", &TTProgramNode::abi_plans)
      .def_ro("execution_plans", &TTProgramNode::execution_plans)
      .def_ro("kernels", &TTProgramNode::kernels)
      .def_ro("core_groups", &TTProgramNode::core_groups)
      .def_ro("cb_plans", &TTProgramNode::cb_plans)
      .def_ro("semaphore_plans", &TTProgramNode::semaphore_plans)
      .def_ro("compute_sync_plans", &TTProgramNode::compute_sync_plans)
      .def_ro("dst_layout_plans", &TTProgramNode::dst_layout_plans)
      .def_ro("live_form_plans", &TTProgramNode::live_form_plans)
      .def_ro("materialization_plans", &TTProgramNode::materialization_plans)
      .def_ro("consumer_binding_plans", &TTProgramNode::consumer_binding_plans)
      .def_ro("exact_cb_virtual_values",
              &TTProgramNode::exact_cb_virtual_values)
      .def_ro("exact_cb_use_events", &TTProgramNode::exact_cb_use_events)
      .def_ro("exact_cb_live_intervals",
              &TTProgramNode::exact_cb_live_intervals)
      .def_ro("exact_cb_allocations", &TTProgramNode::exact_cb_allocations)
      .def_ro("exact_cb_release_events",
              &TTProgramNode::exact_cb_release_events)
      .def_ro("resource_demands", &TTProgramNode::resource_demands)
      .def_ro("resource_pressure_reports",
              &TTProgramNode::resource_pressure_reports);
}

TTProgram::TTProgram(
    ffi::String entry_name, ffi::String member_func,
    ffi::Array<TTMeshPlan> mesh_plans,
    ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans,
    ffi::Array<TTTensorMemoryConfigPlan> tensor_memory_config_plans,
    ffi::Array<TTOpShardingContract> op_sharding_contracts,
    ffi::Array<TTPlacementResolutionPlan> placement_resolution_plans,
    ffi::Array<TTReshardPlan> reshard_plans,
    ffi::Array<TTBlockPlan> block_plans, ffi::Array<TTKernelPlan> kernel_plans,
    ffi::Array<TTComputeOpPlan> compute_op_plans,
    ffi::Array<TTTransportPlan> transport_plans,
    ffi::Array<TTSyncPlan> sync_plans, ffi::Array<TTABIPlan> abi_plans,
    ffi::Array<TTExecutionPlan> execution_plans, ffi::Array<TTKernel> kernels,
    ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
    ffi::Array<TTSemaphorePlan> semaphore_plans,
    ffi::Array<TTComputeSyncPlan> compute_sync_plans,
    ffi::Array<TTDstLayoutPlan> dst_layout_plans,
    ffi::Array<TTLiveFormPlan> live_form_plans,
    ffi::Array<TTMaterializationPlan> materialization_plans,
    ffi::Array<TTConsumerBindingPlan> consumer_binding_plans,
    ffi::Array<TTExactCBVirtualValue> exact_cb_virtual_values,
    ffi::Array<TTExactCBUseEvent> exact_cb_use_events,
    ffi::Array<TTExactCBLiveInterval> exact_cb_live_intervals,
    ffi::Array<TTExactCBAllocation> exact_cb_allocations,
    ffi::Array<TTExactCBReleaseEvent> exact_cb_release_events,
    ffi::Array<TTResourceDemand> resource_demands,
    ffi::Array<TTResourcePressureReport> resource_pressure_reports) {
  auto n = ffi::make_object<TTProgramNode>();
  n->entry_name = std::move(entry_name);
  n->member_func = std::move(member_func);
  n->mesh_plans = std::move(mesh_plans);
  n->buffer_distribution_plans = std::move(buffer_distribution_plans);
  n->tensor_memory_config_plans = std::move(tensor_memory_config_plans);
  n->op_sharding_contracts = std::move(op_sharding_contracts);
  n->placement_resolution_plans = std::move(placement_resolution_plans);
  n->reshard_plans = std::move(reshard_plans);
  n->block_plans = std::move(block_plans);
  n->kernel_plans = std::move(kernel_plans);
  n->compute_op_plans = std::move(compute_op_plans);
  n->transport_plans = std::move(transport_plans);
  n->sync_plans = std::move(sync_plans);
  n->abi_plans = std::move(abi_plans);
  n->execution_plans = std::move(execution_plans);
  n->kernels = std::move(kernels);
  n->core_groups = std::move(core_groups);
  n->cb_plans = std::move(cb_plans);
  n->semaphore_plans = std::move(semaphore_plans);
  n->compute_sync_plans = std::move(compute_sync_plans);
  n->dst_layout_plans = std::move(dst_layout_plans);
  n->live_form_plans = std::move(live_form_plans);
  n->materialization_plans = std::move(materialization_plans);
  n->consumer_binding_plans = std::move(consumer_binding_plans);
  n->exact_cb_virtual_values = std::move(exact_cb_virtual_values);
  n->exact_cb_use_events = std::move(exact_cb_use_events);
  n->exact_cb_live_intervals = std::move(exact_cb_live_intervals);
  n->exact_cb_allocations = std::move(exact_cb_allocations);
  n->exact_cb_release_events = std::move(exact_cb_release_events);
  n->resource_demands = std::move(resource_demands);
  n->resource_pressure_reports = std::move(resource_pressure_reports);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNodeReflection<TTMeshPlanNode>();
  RegisterNodeReflection<TTBufferDistributionPlanNode>();
  RegisterNodeReflection<TTTensorMemoryConfigPlanNode>();
  RegisterNodeReflection<TTOpShardingContractNode>();
  RegisterNodeReflection<TTPlacementResolutionPlanNode>();
  RegisterNodeReflection<TTReshardPlanNode>();
  RegisterNodeReflection<TTComputeOperandBindingPlanNode>();
  RegisterNodeReflection<TTComputeOpPlanNode>();
  RegisterNodeReflection<TTTileComputeFanoutDemandNode>();
  RegisterNodeReflection<TTTileComputeMaterializationDemandNode>();
  RegisterNodeReflection<TTResourceDemandNode>();
  RegisterNodeReflection<TTResourcePressureReportNode>();
  RegisterNodeReflection<TTBlockPlanNode>();
  RegisterNodeReflection<TTKernelPlanNode>();
  RegisterNodeReflection<TTKernelLaunchSpecNode>();
  RegisterNodeReflection<TTKernelDefineNode>();
  RegisterNodeReflection<TTKernelNamedCompileArgNode>();
  RegisterNodeReflection<TTKernelComputeConfigNode>();
  RegisterNodeReflection<TTPerWorkArgSpecNode>();
  RegisterNodeReflection<TTKernelNode>();
  RegisterNodeReflection<TTCoreGroupNode>();
  RegisterNodeReflection<TTCBPlanNode>();
  RegisterNodeReflection<TTTransportPlanNode>();
  RegisterNodeReflection<TTSyncPlanNode>();
  RegisterNodeReflection<TTSemaphorePlanNode>();
  RegisterNodeReflection<TTComputeSyncPlanNode>();
  RegisterNodeReflection<TTDstLayoutPlanNode>();
  RegisterNodeReflection<TTLiveFormPlanNode>();
  RegisterNodeReflection<TTMaterializationPlanNode>();
  RegisterNodeReflection<TTConsumerBindingPlanNode>();
  RegisterNodeReflection<TTExactCBVirtualValueNode>();
  RegisterNodeReflection<TTExactCBUseEventNode>();
  RegisterNodeReflection<TTExactCBLiveIntervalNode>();
  RegisterNodeReflection<TTExactCBAllocationNode>();
  RegisterNodeReflection<TTExactCBReleaseEventNode>();
  RegisterNodeReflection<TTRuntimeArgSpecNode>();
  RegisterNodeReflection<TTCompileTimeArgSpecNode>();
  RegisterNodeReflection<TTAccessorSpecNode>();
  RegisterNodeReflection<TTSemaphoreBindingSpecNode>();
  RegisterNodeReflection<TTABIPlanNode>();
  RegisterNodeReflection<TTExecutionPlanNode>();
  RegisterNodeReflection<TTProgramNode>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.TTMeshPlan",
      [](ffi::String name, ffi::String mesh_kind,
         ffi::Array<Integer> mesh_shape, ffi::Array<Integer> device_range_start,
         ffi::Array<Integer> device_range_shape, ffi::String system_mesh_ref) {
        return TTMeshPlan(std::move(name), std::move(mesh_kind),
                          std::move(mesh_shape), std::move(device_range_start),
                          std::move(device_range_shape),
                          std::move(system_mesh_ref));
      });
  refl::GlobalDef().def(
      "tl.TTBufferDistributionPlan",
      [](ffi::String name, ffi::String buffer, ffi::String mesh_plan,
         int64_t mesh_plan_index, ffi::String distribution_kind,
         ffi::String layout, ffi::String memory_space, int64_t page_size_bytes,
         ffi::Array<Integer> shard_shape, ffi::Array<Integer> shard_grid_shape,
         ffi::String sharding_strategy, ffi::String shard_orientation,
         ffi::String source_buffer, ffi::String source_region_kind,
         ffi::Array<Integer> source_region_shape,
         ffi::String logical_index_mapping,
         ffi::String core_local_address_mapping, ffi::String host_visibility,
         ffi::String attached_core_group, int64_t attached_core_group_index,
         ffi::Array<PrimExpr> logical_shape, ffi::Array<PrimExpr> local_shape,
         PrimExpr thread_extent, PrimExpr replicate_extent,
         ffi::Array<PrimExpr> inverse_logical_index_vars,
         ffi::Array<PrimExpr> inverse_logical_index_exprs,
         ffi::String spatial_layout, ffi::String spatial_distribution_kind,
         ffi::String abi_layout, ffi::String abi_memory_space) {
        return TTBufferDistributionPlan(
            std::move(name), std::move(buffer), std::move(mesh_plan),
            mesh_plan_index, std::move(distribution_kind), std::move(layout),
            std::move(memory_space), page_size_bytes, std::move(shard_shape),
            std::move(shard_grid_shape), std::move(sharding_strategy),
            std::move(shard_orientation), std::move(source_buffer),
            std::move(source_region_kind), std::move(source_region_shape),
            std::move(logical_index_mapping),
            std::move(core_local_address_mapping), std::move(host_visibility),
            std::move(attached_core_group), attached_core_group_index,
            std::move(logical_shape), std::move(local_shape),
            std::move(thread_extent), std::move(replicate_extent),
            std::move(inverse_logical_index_vars),
            std::move(inverse_logical_index_exprs), std::move(spatial_layout),
            std::move(spatial_distribution_kind), std::move(abi_layout),
            std::move(abi_memory_space));
      });
	  refl::GlobalDef().def(
	      "tl.TTTensorMemoryConfigPlan",
      [](ffi::String name, ffi::String subject, ffi::String value_identity,
         ffi::Array<PrimExpr> logical_shape, ffi::String dtype,
         ffi::String memory_layout, ffi::String buffer_type,
         ffi::String grid_ref, ffi::Array<Integer> shard_grid_shape,
         ffi::Array<Integer> shard_shape, ffi::String shard_orientation,
         ffi::String shard_distribution_strategy,
         ffi::Array<Integer> page_shape, ffi::String origin,
         ffi::String source_buffer, ffi::String buffer_distribution_plan,
         int64_t buffer_distribution_plan_index, bool has_runtime_accessor,
         bool requires_materialization) {
        return TTTensorMemoryConfigPlan(
            std::move(name), std::move(subject), std::move(value_identity),
            std::move(logical_shape), std::move(dtype),
            std::move(memory_layout), std::move(buffer_type),
            std::move(grid_ref), std::move(shard_grid_shape),
            std::move(shard_shape), std::move(shard_orientation),
            std::move(shard_distribution_strategy), std::move(page_shape),
            std::move(origin), std::move(source_buffer),
            std::move(buffer_distribution_plan), buffer_distribution_plan_index,
	            has_runtime_accessor, requires_materialization);
	      });
  refl::GlobalDef().def(
      "tl.TTOpShardingContract",
      [](ffi::String name, ffi::String compute_op_plan,
         int64_t compute_op_plan_index, ffi::String operation_name,
         ffi::String op_kind, ffi::String operand_role,
         ffi::String operand_buffer, ffi::String operand_host_buffer,
         ffi::String memory_config_plan, int64_t memory_config_plan_index,
         ffi::Array<ffi::String> accepted_memory_layouts,
         ffi::Array<ffi::String> accepted_buffer_types,
         ffi::Array<ffi::String> accepted_sharding_strategies,
         ffi::String required_shard_orientation, ffi::String output_policy,
         bool may_request_input_conversion, bool can_produce_output_placement,
         bool direct_external_write_allowed, ffi::String reject_reason) {
        return TTOpShardingContract(
            std::move(name), std::move(compute_op_plan), compute_op_plan_index,
            std::move(operation_name), std::move(op_kind),
            std::move(operand_role), std::move(operand_buffer),
            std::move(operand_host_buffer), std::move(memory_config_plan),
            memory_config_plan_index, std::move(accepted_memory_layouts),
            std::move(accepted_buffer_types),
            std::move(accepted_sharding_strategies),
            std::move(required_shard_orientation), std::move(output_policy),
            may_request_input_conversion, can_produce_output_placement,
            direct_external_write_allowed, std::move(reject_reason));
      });
  refl::GlobalDef().def(
      "tl.TTPlacementResolutionPlan",
      [](ffi::String name, ffi::String op_sharding_contract,
         int64_t op_sharding_contract_index, ffi::String consumer_op_plan,
         int64_t consumer_op_plan_index, ffi::String consumer_operand_role,
         ffi::String selected_memory_config_plan,
         int64_t selected_memory_config_plan_index,
         ffi::String selected_memory_layout, ffi::String selected_buffer_type,
         ffi::String resolution_kind, bool conversion_required,
         ffi::String conversion_plan, ffi::String conflict_reason) {
        return TTPlacementResolutionPlan(
            std::move(name), std::move(op_sharding_contract),
            op_sharding_contract_index, std::move(consumer_op_plan),
            consumer_op_plan_index, std::move(consumer_operand_role),
            std::move(selected_memory_config_plan),
            selected_memory_config_plan_index,
            std::move(selected_memory_layout), std::move(selected_buffer_type),
            std::move(resolution_kind), conversion_required,
            std::move(conversion_plan), std::move(conflict_reason));
      });
  refl::GlobalDef().def(
      "tl.TTReshardPlan",
      [](ffi::String name, ffi::String source_value, ffi::String target_value,
         ffi::String source_memory_config_plan,
         int64_t source_memory_config_plan_index,
         ffi::String target_memory_config_plan,
         int64_t target_memory_config_plan_index, ffi::String conversion_kind,
         ffi::String source_region_kind,
         ffi::Array<Integer> source_region_shape,
         ffi::String materialization_plan, int64_t materialization_plan_index,
         ffi::String materialization_protocol,
         ffi::Array<Integer> required_cb_plan_indices,
         ffi::Array<Integer> required_sync_plan_indices,
         ffi::String scheduling_kind, ffi::String inserted_by,
         ffi::String admission_status, ffi::String unsupported_reason) {
        return TTReshardPlan(
            std::move(name), std::move(source_value), std::move(target_value),
            std::move(source_memory_config_plan), source_memory_config_plan_index,
            std::move(target_memory_config_plan), target_memory_config_plan_index,
            std::move(conversion_kind), std::move(source_region_kind),
            std::move(source_region_shape), std::move(materialization_plan),
            materialization_plan_index, std::move(materialization_protocol),
            std::move(required_cb_plan_indices),
            std::move(required_sync_plan_indices), std::move(scheduling_kind),
            std::move(inserted_by), std::move(admission_status),
            std::move(unsupported_reason));
      });
	  refl::GlobalDef().def("tl.TTComputeOperandBindingPlan",
                        [](ffi::String role, ffi::String buffer,
                           ffi::String host_buffer, ffi::String tensor_dtype,
                           ffi::String cb_dtype, ffi::String transform_kind) {
                          return TTComputeOperandBindingPlan(
                              std::move(role), std::move(buffer),
                              std::move(host_buffer), std::move(tensor_dtype),
                              std::move(cb_dtype), std::move(transform_kind));
                        });
  refl::GlobalDef().def(
      "tl.TTComputeOpPlan",
      [](ffi::String name, ffi::String kernel_name, int64_t kernel_plan_index,
         ffi::String kind, ffi::String operation_name, bool enabled,
         ffi::Array<TTComputeOperandBindingPlan> operand_bindings,
         ffi::Array<ffi::String> problem_shape_axes,
         ffi::Array<Integer> problem_shape, ffi::Array<Integer> tile_shape,
         ffi::Array<Integer> block_shape, ffi::Array<Integer> subblock_shape,
         ffi::String accumulator_dtype, ffi::String mbarrier_buffer,
         ffi::String mbarrier_scope,
         ffi::Array<ffi::String> mbarrier_index_exprs,
         int64_t tile_compute_dag_node_id,
         ffi::String tile_compute_source_emitter,
         ffi::String tile_compute_materialization_policy,
         int64_t tile_compute_fanout_use_count,
         ffi::String tile_compute_fanout_policy) {
        return TTComputeOpPlan(
            std::move(name), std::move(kernel_name), kernel_plan_index,
            std::move(kind), std::move(operation_name), enabled,
            std::move(operand_bindings), std::move(problem_shape_axes),
            std::move(problem_shape), std::move(tile_shape),
            std::move(block_shape), std::move(subblock_shape),
            std::move(accumulator_dtype), std::move(mbarrier_buffer),
            std::move(mbarrier_scope), std::move(mbarrier_index_exprs),
            tile_compute_dag_node_id, std::move(tile_compute_source_emitter),
            std::move(tile_compute_materialization_policy),
            tile_compute_fanout_use_count,
            std::move(tile_compute_fanout_policy));
      });
  refl::GlobalDef().def(
      "tl.TTTileComputeFanoutDemand",
      [](ffi::String name, ffi::String kernel_name, int64_t producer_node,
         ffi::String producer_operation, ffi::String value_repr,
         int64_t use_count, ffi::Array<Integer> consumer_nodes,
         ffi::String policy, ffi::String evidence) {
        return TTTileComputeFanoutDemand(
            std::move(name), std::move(kernel_name), producer_node,
            std::move(producer_operation), std::move(value_repr), use_count,
            std::move(consumer_nodes), std::move(policy), std::move(evidence));
      });
  refl::GlobalDef().def(
      "tl.TTTileComputeMaterializationDemand",
      [](ffi::String name, ffi::String kernel_name, int64_t node_id,
         ffi::String operation_name, ffi::String pattern_name,
         ffi::String policy, ffi::String evidence) {
        return TTTileComputeMaterializationDemand(
            std::move(name), std::move(kernel_name), node_id,
            std::move(operation_name), std::move(pattern_name),
            std::move(policy), std::move(evidence));
      });
  refl::GlobalDef().def(
      "tl.TTResourceDemand",
      [](ffi::String name, ffi::String kernel_name, ffi::String core_group,
         int64_t core_group_index,
         ffi::Array<TTTileComputeFanoutDemand> tile_compute_fanout_demands,
         ffi::Array<TTTileComputeMaterializationDemand>
             tile_compute_materialization_demands,
         ffi::Array<ffi::String> tile_compute_unsupported_reasons,
         int64_t cb_requirement_count, int64_t cb_l1_bytes,
         int64_t semaphore_count, int64_t communication_edge_count) {
        return TTResourceDemand(
            std::move(name), std::move(kernel_name), std::move(core_group),
            core_group_index, std::move(tile_compute_fanout_demands),
            std::move(tile_compute_materialization_demands),
            std::move(tile_compute_unsupported_reasons), cb_requirement_count,
            cb_l1_bytes, semaphore_count, communication_edge_count);
      });
  refl::GlobalDef().def(
      "tl.TTResourcePressureReport",
      [](ffi::String name, ffi::String kernel_name, ffi::String core_group,
         int64_t core_group_index,
         ffi::Array<ffi::String> tile_compute_unsupported_reasons,
         ffi::Array<TTTileComputeMaterializationDemand>
             required_materializations,
         int64_t per_core_cb_id_pressure, int64_t per_core_cb_l1_bytes,
         int64_t per_core_l1_buffer_bytes, int64_t max_simultaneous_l1_bytes,
         int64_t cb_id_limit, int64_t worker_l1_budget_bytes,
         int64_t l1_alignment_bytes, int64_t per_core_cb_l1_aligned_bytes,
         int64_t l1_alignment_waste_bytes, ffi::String core_grid_requirement,
         ffi::String dram_view_requirement,
         ffi::Array<ffi::String> unsupported_reasons) {
        return TTResourcePressureReport(
            std::move(name), std::move(kernel_name), std::move(core_group),
            core_group_index, std::move(tile_compute_unsupported_reasons),
            std::move(required_materializations), per_core_cb_id_pressure,
            per_core_cb_l1_bytes, per_core_l1_buffer_bytes,
            max_simultaneous_l1_bytes, cb_id_limit, worker_l1_budget_bytes,
            l1_alignment_bytes, per_core_cb_l1_aligned_bytes,
            l1_alignment_waste_bytes, std::move(core_grid_requirement),
            std::move(dram_view_requirement), std::move(unsupported_reasons));
      });
  refl::GlobalDef().def(
      "tl.TTBlockPlan", [](ffi::String name, ffi::String placement_kind,
                           ffi::Array<Integer> task_indices,
                           ffi::String core_group, int64_t core_group_index) {
        return TTBlockPlan(std::move(name), std::move(placement_kind),
                           std::move(task_indices), std::move(core_group),
                           core_group_index);
      });
  refl::GlobalDef().def("tl.TTKernelPlan",
                        [](ffi::String name, ffi::String kind,
                           ffi::String core_type, int64_t block_plan_index,
                           int64_t abi_plan_index) {
                          return TTKernelPlan(std::move(name), std::move(kind),
                                              std::move(core_type),
                                              block_plan_index, abi_plan_index);
                        });
  refl::GlobalDef().def(
      "tl.TTKernelLaunchSpec",
      [](ffi::String core_type, ffi::String processor, ffi::String noc) {
        return TTKernelLaunchSpec(std::move(core_type), std::move(processor),
                                  std::move(noc));
      });
  refl::GlobalDef().def(
      "tl.TTKernelDefine", [](ffi::String name, ffi::String value) {
        return TTKernelDefine(std::move(name), std::move(value));
      });
  refl::GlobalDef().def(
      "tl.TTKernelNamedCompileArg", [](ffi::String name, int64_t value) {
        return TTKernelNamedCompileArg(std::move(name), value);
      });
  refl::GlobalDef().def(
      "tl.TTKernelComputeConfig",
      [](ffi::String math_fidelity, bool fp32_dest_acc_en,
         bool dst_full_sync_en, bool math_approx_mode,
         ffi::Array<ffi::String> unpack_to_dest_mode, bool bfp8_pack_precise,
         ffi::Array<TTKernelDefine> defines,
         ffi::Array<TTKernelNamedCompileArg> named_compile_args,
         bool clear_accum, int64_t k_pack, int64_t wg_wait, int64_t policy_type,
         ffi::String policy_name) {
        return TTKernelComputeConfig(
            std::move(math_fidelity), fp32_dest_acc_en, dst_full_sync_en,
            math_approx_mode, std::move(unpack_to_dest_mode), bfp8_pack_precise,
            std::move(defines), std::move(named_compile_args), clear_accum,
            k_pack, wg_wait, policy_type, std::move(policy_name));
      });
  refl::GlobalDef().def("tl.TTPerWorkArgSpec",
                        [](ffi::String arg_kind, ffi::String arg_identity,
                           ffi::String buffer, ffi::String descriptor_kind,
                           ffi::String value_source, int64_t constant_value) {
                          return TTPerWorkArgSpec(
                              std::move(arg_kind), std::move(arg_identity),
                              std::move(buffer), std::move(descriptor_kind),
                              std::move(value_source), constant_value);
                        });
  refl::GlobalDef().def("tl.TTPerWorkArgSpecWithAccessRegion",
                        [](ffi::String arg_kind, ffi::String arg_identity,
                           ffi::String buffer, ffi::String descriptor_kind,
                           ffi::String value_source, int64_t constant_value,
                           ffi::String access_region,
                           int64_t access_region_index) {
                          return TTPerWorkArgSpec(
                              std::move(arg_kind), std::move(arg_identity),
                              std::move(buffer), std::move(descriptor_kind),
                              std::move(value_source), constant_value,
                              std::move(access_region), access_region_index);
                        });
  refl::GlobalDef().def("tl.TTPerWorkArgSpecWithIndexTable",
                        [](ffi::String arg_kind, ffi::String arg_identity,
                           ffi::String buffer, ffi::String descriptor_kind,
                           ffi::String value_source, int64_t constant_value,
                           ffi::String access_region,
                           int64_t access_region_index,
                           ffi::String index_buffer,
                           int64_t index_value_scale) {
                          return TTPerWorkArgSpec(
                              std::move(arg_kind), std::move(arg_identity),
                              std::move(buffer), std::move(descriptor_kind),
                              std::move(value_source), constant_value,
                              std::move(access_region), access_region_index,
                              std::move(index_buffer), index_value_scale);
                        });
  refl::GlobalDef().def(
      "tl.TTKernel",
      [](ffi::String name, ffi::String kind, ffi::String core_type,
         int64_t abi_plan_index, TTKernelLaunchSpec launch_spec,
         TTKernelComputeConfig compute_config,
         ffi::Array<TTPerWorkArgSpec> per_work_arg_specs) {
        return TTKernel(std::move(name), std::move(kind), std::move(core_type),
                        abi_plan_index, std::move(launch_spec),
                        std::move(compute_config),
                        std::move(per_work_arg_specs));
      });
  refl::GlobalDef().def(
      "tl.TTCoreGroup",
      [](ffi::String name, int64_t logical_grid_x, int64_t logical_grid_y,
         ffi::String linearization, ffi::Array<ffi::Any> physical_cores,
         ffi::Array<ffi::Any> work_packets) {
        return TTCoreGroup(std::move(name), logical_grid_x, logical_grid_y,
                           std::move(linearization), std::move(physical_cores),
                           std::move(work_packets));
      });
  refl::GlobalDef().def(
      "tl.TTCBPlan",
      [](ffi::String name, int64_t cb_id, ffi::String resource_class,
         int64_t num_pages, int64_t page_size_bytes, ffi::String data_format,
         int64_t initial_reserve_pages, ffi::String flow_class,
         int64_t publish_pages_per_event, int64_t consume_pages_per_event,
         int64_t lifetime_begin, int64_t lifetime_end,
         ffi::Array<ffi::String> requirement_names,
         ffi::Array<Integer> requirement_indices) {
        return TTCBPlan(
            std::move(name), cb_id, std::move(resource_class), num_pages,
            page_size_bytes, std::move(data_format), initial_reserve_pages,
            std::move(flow_class), publish_pages_per_event,
            consume_pages_per_event, lifetime_begin, lifetime_end,
            std::move(requirement_names), std::move(requirement_indices));
      });
  refl::GlobalDef().def(
      "tl.TTTransportPlan",
      [](ffi::String name, ffi::String kind, int64_t source_task_index,
         int64_t target_task_index, ffi::String value_kind,
         ffi::String delivery_kind, ffi::String subject) {
        return TTTransportPlan(std::move(name), std::move(kind),
                               source_task_index, target_task_index,
                               std::move(value_kind), std::move(delivery_kind),
                               std::move(subject));
      });
  refl::GlobalDef().def("tl.TTSyncPlan", [](ffi::String name, ffi::String kind,
                                            int64_t source_task_index,
                                            int64_t target_task_index,
                                            ffi::String ordering_kind,
                                            ffi::String completion_kind) {
    return TTSyncPlan(std::move(name), std::move(kind), source_task_index,
                      target_task_index, std::move(ordering_kind),
                      std::move(completion_kind));
  });
  refl::GlobalDef().def(
      "tl.TTSemaphorePlan",
      [](ffi::String name, ffi::String kind, int64_t semaphore_id,
         int64_t initial_value, ffi::String core_type,
         int64_t source_task_index, int64_t target_task_index,
         ffi::Array<ffi::Any> core_ranges) {
        return TTSemaphorePlan(std::move(name), std::move(kind), semaphore_id,
                               initial_value, std::move(core_type),
                               source_task_index, target_task_index,
                               std::move(core_ranges));
      });
  refl::GlobalDef().def(
      "tl.TTComputeSyncPlan",
      [](ffi::String name, ffi::String kind, int64_t source_task_index,
         int64_t target_task_index, ffi::String ordering_kind,
         ffi::String materialization_kind) {
        return TTComputeSyncPlan(std::move(name), std::move(kind),
                                 source_task_index, target_task_index,
                                 std::move(ordering_kind),
                                 std::move(materialization_kind));
      });
  refl::GlobalDef().def(
      "tl.TTDstLayoutPlan",
      [](ffi::String name, ffi::String buffer, ffi::String layout,
         ffi::String memory_space, int64_t page_size_bytes) {
        return TTDstLayoutPlan(std::move(name), std::move(buffer),
                               std::move(layout), std::move(memory_space),
                               page_size_bytes);
      });
  refl::GlobalDef().def(
      "tl.TTLiveFormPlan",
      [](ffi::String name, ffi::String logical_value,
         ffi::String spatial_live_value, int64_t spatial_live_value_index,
         ffi::String producer_kernel, ffi::String physical_form,
         ffi::String execution_topology, int64_t physical_local_extent,
         int64_t logical_element_count, ffi::String ownership_kind) {
        return TTLiveFormPlan(
            std::move(name), std::move(logical_value),
            std::move(spatial_live_value), spatial_live_value_index,
            std::move(producer_kernel), std::move(physical_form),
            std::move(execution_topology), physical_local_extent,
            logical_element_count, std::move(ownership_kind));
      });
  refl::GlobalDef().def(
      "tl.TTMaterializationPlan",
      [](ffi::String name, ffi::String source_live_form,
         ffi::String materialization_boundary,
         int64_t materialization_boundary_index, ffi::String target_buffer,
         ffi::String host_buffer, ffi::String target_kernel,
         ffi::String bridge_kind, ffi::String materialization_kind,
         ffi::String materialization_protocol, ffi::String publication_protocol,
         ffi::Array<Integer> required_cb_plan_indices,
         ffi::Array<Integer> required_sync_plan_indices,
         ffi::String produced_live_form) {
        return TTMaterializationPlan(
            std::move(name), std::move(source_live_form),
            std::move(materialization_boundary), materialization_boundary_index,
            std::move(target_buffer), std::move(host_buffer),
            std::move(target_kernel), std::move(bridge_kind),
            std::move(materialization_kind),
            std::move(materialization_protocol),
            std::move(publication_protocol),
            std::move(required_cb_plan_indices),
            std::move(required_sync_plan_indices),
            std::move(produced_live_form));
      });
  refl::GlobalDef().def(
      "tl.TTConsumerBindingPlan",
      [](ffi::String name, ffi::String consumer_kernel,
         ffi::String consumer_op_kind, ffi::String source_live_form,
         ffi::String live_value_edge, int64_t live_value_edge_index,
         bool accepts_distributed_slice, bool requires_full_logical_tile,
         int64_t abi_plan_index, ffi::String target_buffer,
         ffi::String materialization_plan) {
        return TTConsumerBindingPlan(
            std::move(name), std::move(consumer_kernel),
            std::move(consumer_op_kind), std::move(source_live_form),
            std::move(live_value_edge), live_value_edge_index,
            accepts_distributed_slice, requires_full_logical_tile,
            abi_plan_index, std::move(target_buffer),
            std::move(materialization_plan));
      });
  refl::GlobalDef().def(
      "tl.TTExactCBVirtualValue",
      [](ffi::String name, ffi::String logical_value, ffi::String live_form,
         int64_t live_form_index, ffi::String producer_kernel,
         ffi::String producer_event, ffi::String event_lifetime_kind,
         ffi::String loop_role, int64_t num_pages, int64_t page_size_bytes,
         ffi::String data_format) {
        return TTExactCBVirtualValue(
            std::move(name), std::move(logical_value), std::move(live_form),
            live_form_index, std::move(producer_kernel),
            std::move(producer_event), std::move(event_lifetime_kind),
            std::move(loop_role), num_pages, page_size_bytes,
            std::move(data_format));
      });
  refl::GlobalDef().def(
      "tl.TTExactCBUseEvent",
      [](ffi::String name, ffi::String virtual_value,
         int64_t virtual_value_index, ffi::String consumer_kernel,
         ffi::String consumer_event, ffi::String operand_role,
         int64_t program_point, bool requires_full_logical_tile,
         ffi::String borrow_kind) {
        return TTExactCBUseEvent(
            std::move(name), std::move(virtual_value), virtual_value_index,
            std::move(consumer_kernel), std::move(consumer_event),
            std::move(operand_role), program_point, requires_full_logical_tile,
            std::move(borrow_kind));
      });
  refl::GlobalDef().def(
      "tl.TTExactCBLiveInterval",
      [](ffi::String name, ffi::String virtual_value,
         int64_t virtual_value_index, int64_t begin_point, int64_t end_point,
         bool live_in, bool live_out, bool loop_carried,
         ffi::String interference_class) {
        return TTExactCBLiveInterval(
            std::move(name), std::move(virtual_value), virtual_value_index,
            begin_point, end_point, live_in, live_out, loop_carried,
            std::move(interference_class));
      });
  refl::GlobalDef().def(
      "tl.TTExactCBAllocation",
      [](ffi::String name, ffi::String virtual_value,
         int64_t virtual_value_index, ffi::String cb_plan,
         int64_t cb_plan_index, int64_t physical_cb_id, int64_t page_count,
         int64_t release_program_point, ffi::String release_reason) {
        return TTExactCBAllocation(
            std::move(name), std::move(virtual_value), virtual_value_index,
            std::move(cb_plan), cb_plan_index, physical_cb_id, page_count,
            release_program_point, std::move(release_reason));
      });
  refl::GlobalDef().def(
      "tl.TTExactCBReleaseEvent",
      [](ffi::String name, ffi::String allocation, int64_t allocation_index,
         ffi::String cb_plan, int64_t cb_plan_index, int64_t program_point,
         int64_t page_count, ffi::String reason) {
        return TTExactCBReleaseEvent(
            std::move(name), std::move(allocation), allocation_index,
            std::move(cb_plan), cb_plan_index, program_point, page_count,
            std::move(reason));
      });
  refl::GlobalDef().def(
      "tl.TTRuntimeArgSpec",
      [](ffi::String name, ffi::String kind, ffi::String dtype,
         ffi::String buffer, ffi::String identity, int64_t core_x,
         int64_t core_y) {
        return TTRuntimeArgSpec(std::move(name), std::move(kind),
                                std::move(dtype), std::move(buffer),
                                std::move(identity), core_x, core_y);
      });
  refl::GlobalDef().def(
      "tl.TTCompileTimeArgSpec",
      [](ffi::String name, ffi::String kind, ffi::String dtype, int64_t offset,
         int64_t count, ffi::String buffer, ffi::String segment_role,
         ffi::Array<Integer> values, int64_t args_config_bits,
         int64_t transport_page_size, ffi::String layout,
         ffi::String memory_space, ffi::Array<Integer> host_axis_order,
         bool transpose_2d) {
        return TTCompileTimeArgSpec(
            std::move(name), std::move(kind), std::move(dtype), offset, count,
            std::move(buffer), std::move(segment_role), std::move(values),
            args_config_bits, transport_page_size, std::move(layout),
            std::move(memory_space), std::move(host_axis_order), transpose_2d);
      });
  refl::GlobalDef().def(
      "tl.TTAccessorSpec",
      [](ffi::String buffer, int64_t compile_time_arg_offset,
         int64_t compile_time_arg_count, int64_t common_runtime_arg_offset,
         int64_t common_runtime_arg_count, int64_t args_config_bits,
         int64_t transport_page_size, ffi::String layout,
         ffi::String memory_space, ffi::Array<Integer> host_axis_order,
         bool transpose_2d) {
        return TTAccessorSpec(
            std::move(buffer), compile_time_arg_offset, compile_time_arg_count,
            common_runtime_arg_offset, common_runtime_arg_count,
            args_config_bits, transport_page_size, std::move(layout),
            std::move(memory_space), std::move(host_axis_order), transpose_2d);
      });
  refl::GlobalDef().def(
      "tl.TTSemaphoreBindingSpec",
      [](ffi::String name, int64_t semaphore_id, ffi::String arg_kind) {
        return TTSemaphoreBindingSpec(std::move(name), semaphore_id,
                                      std::move(arg_kind));
      });
  refl::GlobalDef().def(
      "tl.TTABIPlan",
      [](ffi::String name, ffi::String kernel_name,
         ffi::Array<TTRuntimeArgSpec> runtime_args,
         ffi::Array<TTRuntimeArgSpec> common_runtime_args,
         ffi::Array<TTCompileTimeArgSpec> compile_time_arg_specs,
         ffi::Array<TTAccessorSpec> accessors,
         ffi::Array<TTSemaphoreBindingSpec> semaphore_bindings) {
        return TTABIPlan(
            std::move(name), std::move(kernel_name), std::move(runtime_args),
            std::move(common_runtime_args), std::move(compile_time_arg_specs),
            std::move(accessors), std::move(semaphore_bindings));
      });
  refl::GlobalDef().def(
      "tl.TTExecutionPlan",
      [](ffi::String name, ffi::Array<ffi::String> kernel_names,
         ffi::Array<Integer> phase_indices) {
        return TTExecutionPlan(std::move(name), std::move(kernel_names),
                               std::move(phase_indices));
      });
  refl::GlobalDef().def(
	      "tl.TTProgram",
	      [](ffi::String entry_name, ffi::String member_func,
	         ffi::Array<TTMeshPlan> mesh_plans,
	         ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans,
	         ffi::Array<TTTensorMemoryConfigPlan> tensor_memory_config_plans,
	         ffi::Array<TTOpShardingContract> op_sharding_contracts,
	         ffi::Array<TTPlacementResolutionPlan> placement_resolution_plans,
	         ffi::Array<TTReshardPlan> reshard_plans,
	         ffi::Array<TTBlockPlan> block_plans,
         ffi::Array<TTKernelPlan> kernel_plans,
         ffi::Array<TTComputeOpPlan> compute_op_plans,
         ffi::Array<TTTransportPlan> transport_plans,
         ffi::Array<TTSyncPlan> sync_plans, ffi::Array<TTABIPlan> abi_plans,
         ffi::Array<TTExecutionPlan> execution_plans,
         ffi::Array<TTKernel> kernels, ffi::Array<TTCoreGroup> core_groups,
         ffi::Array<TTCBPlan> cb_plans,
         ffi::Array<TTSemaphorePlan> semaphore_plans,
         ffi::Array<TTComputeSyncPlan> compute_sync_plans,
         ffi::Array<TTDstLayoutPlan> dst_layout_plans,
         ffi::Array<TTLiveFormPlan> live_form_plans,
         ffi::Array<TTMaterializationPlan> materialization_plans,
         ffi::Array<TTConsumerBindingPlan> consumer_binding_plans,
         ffi::Array<TTExactCBVirtualValue> exact_cb_virtual_values,
         ffi::Array<TTExactCBUseEvent> exact_cb_use_events,
         ffi::Array<TTExactCBLiveInterval> exact_cb_live_intervals,
         ffi::Array<TTExactCBAllocation> exact_cb_allocations,
         ffi::Array<TTExactCBReleaseEvent> exact_cb_release_events,
         ffi::Array<TTResourceDemand> resource_demands,
         ffi::Array<TTResourcePressureReport> resource_pressure_reports) {
	        return TTProgram(
	            std::move(entry_name), std::move(member_func),
	            std::move(mesh_plans), std::move(buffer_distribution_plans),
	            std::move(tensor_memory_config_plans),
	            std::move(op_sharding_contracts),
	            std::move(placement_resolution_plans),
	            std::move(reshard_plans),
	            std::move(block_plans), std::move(kernel_plans),
            std::move(compute_op_plans), std::move(transport_plans),
            std::move(sync_plans), std::move(abi_plans),
            std::move(execution_plans), std::move(kernels),
            std::move(core_groups), std::move(cb_plans),
            std::move(semaphore_plans), std::move(compute_sync_plans),
            std::move(dst_layout_plans), std::move(live_form_plans),
            std::move(materialization_plans), std::move(consumer_binding_plans),
            std::move(exact_cb_virtual_values),
            std::move(exact_cb_use_events), std::move(exact_cb_live_intervals),
            std::move(exact_cb_allocations), std::move(exact_cb_release_events),
            std::move(resource_demands), std::move(resource_pressure_reports));
      });
}

} // namespace tl
} // namespace tvm
