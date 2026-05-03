/*!
 * \file tt_program_projection.h
 * \brief TTProgram -> ExecutableSpec projection helpers for the canonical
 * writer boundary.
 */

#ifndef TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_
#define TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/function.h>

#include "../transform/common/blackhole_runtime_arg_schema.h"
#include "../transform/common/companion_base.h"
#include "../transform/common/tt_target_program.h"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace tt_program_projection {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace executable_key {
constexpr const char *kSchemaVersion = "schema_version";
constexpr const char *kSource = "source";
constexpr const char *kEntryName = "entry_name";
constexpr const char *kMemberFunc = "member_func";
constexpr const char *kMeshPlans = "mesh_plans";
constexpr const char *kBufferDistributionPlans = "buffer_distribution_plans";
constexpr const char *kTensorMemoryConfigPlans =
    "tensor_memory_config_plans";
constexpr const char *kOpShardingContracts = "op_sharding_contracts";
constexpr const char *kPlacementResolutionPlans =
    "placement_resolution_plans";
constexpr const char *kReshardPlans = "reshard_plans";
constexpr const char *kComputeOpPlans = "compute_op_plans";
constexpr const char *kSegmentPlan = "segment_plan";
constexpr const char *kCBConfigs = "cb_configs";
constexpr const char *kCorePlan = "core_plan";
constexpr const char *kSemaphorePlan = "semaphore_plan";
constexpr const char *kDirectRuntimeUnsupportedReasons =
    "direct_runtime_unsupported_reasons";
constexpr const char *kLiveFormPlans = "live_form_plans";
constexpr const char *kMaterializationPlans = "materialization_plans";
constexpr const char *kConsumerBindingPlans = "consumer_binding_plans";
constexpr const char *kResourcePressureReports = "resource_pressure_reports";
} // namespace executable_key

inline Map<String, Any> AsMap(const Any &any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

inline tvm::ffi::Optional<TTProgram> GetTTProgram(const tir::PrimFunc &func) {
  return func->GetAttr<TTProgram>(attr::kTLTTProgram);
}

inline TTProgram RequireTTProgram(const tir::PrimFunc &func,
                                  const char *consumer) {
  auto maybe_program = GetTTProgram(func);
  ICHECK(maybe_program)
      << consumer << " requires tl.tt_program for executable-writer cutover";
  return maybe_program.value();
}

inline Array<Any> EncodeCBPlans(const Array<TTCBPlan> &cb_plans) {
  Array<Any> encoded;
  for (const TTCBPlan &cb : cb_plans) {
    Map<String, Any> item;
    item.Set("name", cb->name);
    item.Set("cb_id", Integer(cb->cb_id));
    item.Set("role", cb->resource_class);
    item.Set("num_pages", Integer(cb->num_pages));
    item.Set("page_size", Integer(cb->page_size_bytes));
    item.Set("total_size_bytes", Integer(cb->num_pages * cb->page_size_bytes));
    item.Set("data_format", cb->data_format);
    item.Set("initial_reserve_pages", Integer(cb->initial_reserve_pages));
    item.Set("flow_class", cb->flow_class);
    item.Set("publish_pages_per_event", Integer(cb->publish_pages_per_event));
    item.Set("consume_pages_per_event", Integer(cb->consume_pages_per_event));
    item.Set("lifetime_begin", Integer(cb->lifetime_begin));
    item.Set("lifetime_end", Integer(cb->lifetime_end));
    if (!cb->requirement_names.empty()) {
      item.Set("requirement_names", cb->requirement_names);
    }
    if (!cb->requirement_indices.empty()) {
      item.Set("requirement_indices", cb->requirement_indices);
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeMeshPlans(const Array<TTMeshPlan> &mesh_plans) {
  Array<Any> encoded;
  for (const TTMeshPlan &plan : mesh_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("mesh_kind", plan->mesh_kind);
    item.Set("mesh_shape", plan->mesh_shape);
    item.Set("device_range_start", plan->device_range_start);
    item.Set("device_range_shape", plan->device_range_shape);
    item.Set("system_mesh_ref", plan->system_mesh_ref);
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeBufferDistributionPlans(
    const Array<TTBufferDistributionPlan> &buffer_distribution_plans) {
  Array<Any> encoded;
  for (const TTBufferDistributionPlan &plan : buffer_distribution_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("buffer", plan->buffer);
    item.Set("mesh_plan", plan->mesh_plan);
    item.Set("mesh_plan_index", Integer(plan->mesh_plan_index));
    item.Set("distribution_kind", plan->distribution_kind);
    item.Set("layout", plan->layout);
    item.Set("memory_space", plan->memory_space);
    item.Set("page_size_bytes", Integer(plan->page_size_bytes));
    if (!plan->shard_shape.empty()) {
      item.Set("shard_shape", plan->shard_shape);
    }
    if (!plan->shard_grid_shape.empty()) {
      item.Set("shard_grid_shape", plan->shard_grid_shape);
    }
    if (!plan->sharding_strategy.empty()) {
      item.Set("sharding_strategy", plan->sharding_strategy);
    }
    item.Set("shard_orientation", plan->shard_orientation);
    if (!plan->source_buffer.empty()) {
      item.Set("source_buffer", plan->source_buffer);
    }
    if (!plan->source_region_kind.empty()) {
      item.Set("source_region_kind", plan->source_region_kind);
    }
    if (!plan->source_region_shape.empty()) {
      item.Set("source_region_shape", plan->source_region_shape);
    }
    if (!plan->logical_index_mapping.empty()) {
      item.Set("logical_index_mapping", plan->logical_index_mapping);
    }
    if (!plan->core_local_address_mapping.empty()) {
      item.Set("core_local_address_mapping", plan->core_local_address_mapping);
    }
    item.Set("host_visibility", plan->host_visibility);
    if (!plan->attached_core_group.empty()) {
      item.Set("attached_core_group", plan->attached_core_group);
      item.Set("attached_core_group_index",
               Integer(plan->attached_core_group_index));
    }
    if (!plan->logical_shape.empty()) {
      item.Set("logical_shape", plan->logical_shape);
    }
    if (!plan->local_shape.empty()) {
      item.Set("local_shape", plan->local_shape);
    }
    if (plan->thread_extent.defined()) {
      item.Set("thread_extent", plan->thread_extent);
    }
    if (plan->replicate_extent.defined()) {
      item.Set("replicate_extent", plan->replicate_extent);
    }
    if (!plan->inverse_logical_index_vars.empty()) {
      item.Set("inverse_logical_index_vars", plan->inverse_logical_index_vars);
    }
    if (!plan->inverse_logical_index_exprs.empty()) {
      item.Set("inverse_logical_index_exprs",
               plan->inverse_logical_index_exprs);
    }
    if (!plan->spatial_layout.empty()) {
      item.Set("spatial_layout", plan->spatial_layout);
    }
    if (!plan->spatial_distribution_kind.empty()) {
      item.Set("spatial_distribution_kind", plan->spatial_distribution_kind);
    }
    if (!plan->abi_layout.empty()) {
      item.Set("abi_layout", plan->abi_layout);
    }
    if (!plan->abi_memory_space.empty()) {
      item.Set("abi_memory_space", plan->abi_memory_space);
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeTensorMemoryConfigPlans(
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans) {
  Array<Any> encoded;
  for (const TTTensorMemoryConfigPlan &plan : tensor_memory_config_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("subject", plan->subject);
    if (!plan->value_identity.empty()) {
      item.Set("value_identity", plan->value_identity);
    }
    if (!plan->logical_shape.empty()) {
      item.Set("logical_shape", plan->logical_shape);
    }
    if (!plan->dtype.empty()) {
      item.Set("dtype", plan->dtype);
    }
    item.Set("memory_layout", plan->memory_layout);
    item.Set("buffer_type", plan->buffer_type);
    if (!plan->grid_ref.empty()) {
      item.Set("grid_ref", plan->grid_ref);
    }
    if (!plan->shard_grid_shape.empty()) {
      item.Set("shard_grid_shape", plan->shard_grid_shape);
    }
    if (!plan->shard_shape.empty()) {
      item.Set("shard_shape", plan->shard_shape);
    }
    item.Set("shard_orientation", plan->shard_orientation);
    item.Set("shard_distribution_strategy",
             plan->shard_distribution_strategy);
    if (!plan->page_shape.empty()) {
      item.Set("page_shape", plan->page_shape);
    }
    item.Set("origin", plan->origin);
    if (!plan->source_buffer.empty()) {
      item.Set("source_buffer", plan->source_buffer);
    }
    item.Set("buffer_distribution_plan", plan->buffer_distribution_plan);
    item.Set("buffer_distribution_plan_index",
             Integer(plan->buffer_distribution_plan_index));
    item.Set("has_runtime_accessor", Bool(plan->has_runtime_accessor));
    item.Set("requires_materialization",
             Bool(plan->requires_materialization));
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeOpShardingContracts(
    const Array<TTOpShardingContract> &op_sharding_contracts) {
  Array<Any> encoded;
  for (const TTOpShardingContract &contract : op_sharding_contracts) {
    Map<String, Any> item;
    item.Set("name", contract->name);
    item.Set("compute_op_plan", contract->compute_op_plan);
    item.Set("compute_op_plan_index",
             Integer(contract->compute_op_plan_index));
    item.Set("operation_name", contract->operation_name);
    item.Set("op_kind", contract->op_kind);
    item.Set("operand_role", contract->operand_role);
    item.Set("operand_buffer", contract->operand_buffer);
    if (!contract->operand_host_buffer.empty()) {
      item.Set("operand_host_buffer", contract->operand_host_buffer);
    }
    item.Set("memory_config_plan", contract->memory_config_plan);
    item.Set("memory_config_plan_index",
             Integer(contract->memory_config_plan_index));
    item.Set("accepted_memory_layouts", contract->accepted_memory_layouts);
    item.Set("accepted_buffer_types", contract->accepted_buffer_types);
    item.Set("accepted_sharding_strategies",
             contract->accepted_sharding_strategies);
    item.Set("required_shard_orientation",
             contract->required_shard_orientation);
    item.Set("output_policy", contract->output_policy);
    item.Set("may_request_input_conversion",
             Bool(contract->may_request_input_conversion));
    item.Set("can_produce_output_placement",
             Bool(contract->can_produce_output_placement));
    item.Set("direct_external_write_allowed",
             Bool(contract->direct_external_write_allowed));
    if (!contract->reject_reason.empty()) {
      item.Set("reject_reason", contract->reject_reason);
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodePlacementResolutionPlans(
    const Array<TTPlacementResolutionPlan> &placement_resolution_plans) {
  Array<Any> encoded;
  for (const TTPlacementResolutionPlan &plan : placement_resolution_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("op_sharding_contract", plan->op_sharding_contract);
    item.Set("op_sharding_contract_index",
             Integer(plan->op_sharding_contract_index));
    item.Set("consumer_op_plan", plan->consumer_op_plan);
    item.Set("consumer_op_plan_index",
             Integer(plan->consumer_op_plan_index));
    item.Set("consumer_operand_role", plan->consumer_operand_role);
    item.Set("selected_memory_config_plan",
             plan->selected_memory_config_plan);
    item.Set("selected_memory_config_plan_index",
             Integer(plan->selected_memory_config_plan_index));
    item.Set("selected_memory_layout", plan->selected_memory_layout);
    item.Set("selected_buffer_type", plan->selected_buffer_type);
    item.Set("resolution_kind", plan->resolution_kind);
    item.Set("conversion_required", Bool(plan->conversion_required));
    if (!plan->conversion_plan.empty()) {
      item.Set("conversion_plan", plan->conversion_plan);
    }
    if (!plan->conflict_reason.empty()) {
      item.Set("conflict_reason", plan->conflict_reason);
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any>
EncodeReshardPlans(const Array<TTReshardPlan> &reshard_plans) {
  Array<Any> encoded;
  for (const TTReshardPlan &plan : reshard_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("source_value", plan->source_value);
    item.Set("target_value", plan->target_value);
    item.Set("source_memory_config_plan",
             plan->source_memory_config_plan);
    item.Set("source_memory_config_plan_index",
             Integer(plan->source_memory_config_plan_index));
    item.Set("target_memory_config_plan",
             plan->target_memory_config_plan);
    item.Set("target_memory_config_plan_index",
             Integer(plan->target_memory_config_plan_index));
    item.Set("conversion_kind", plan->conversion_kind);
    item.Set("source_region_kind", plan->source_region_kind);
    if (!plan->source_region_shape.empty()) {
      item.Set("source_region_shape", plan->source_region_shape);
    }
    if (!plan->materialization_plan.empty()) {
      item.Set("materialization_plan", plan->materialization_plan);
    }
    item.Set("materialization_plan_index",
             Integer(plan->materialization_plan_index));
    item.Set("materialization_protocol", plan->materialization_protocol);
    item.Set("required_cb_plan_indices", plan->required_cb_plan_indices);
    if (!plan->required_sync_plan_indices.empty()) {
      item.Set("required_sync_plan_indices",
               plan->required_sync_plan_indices);
    }
    item.Set("scheduling_kind", plan->scheduling_kind);
    item.Set("inserted_by", plan->inserted_by);
    item.Set("admission_status", plan->admission_status);
    item.Set("unsupported_reason", plan->unsupported_reason);
    encoded.push_back(item);
  }
  return encoded;
}

inline Map<String, Any>
EncodeComputeOperandBindingPlan(const TTComputeOperandBindingPlan &binding) {
  Map<String, Any> item;
  item.Set("role", binding->role);
  item.Set("buffer", binding->buffer);
  if (!binding->host_buffer.empty()) {
    item.Set("host_buffer", binding->host_buffer);
  }
  if (!binding->tensor_dtype.empty()) {
    item.Set("tensor_dtype", binding->tensor_dtype);
  }
  if (!binding->cb_dtype.empty()) {
    item.Set("cb_dtype", binding->cb_dtype);
  }
  if (!binding->transform_kind.empty()) {
    item.Set("transform_kind", binding->transform_kind);
  }
  return item;
}

inline void SetIntegerShapeField(Map<String, Any> *item, const char *key,
                                 const Array<Integer> &values, size_t index) {
  if (index < values.size()) {
    item->Set(String(key), values[index]);
  }
}

inline Map<String, Any> EncodeComputeOpPlan(const TTComputeOpPlan &plan) {
  Map<String, Any> item;
  item.Set("name", plan->name);
  item.Set("kernel_name", plan->kernel_name);
  item.Set("kernel_plan_index", Integer(plan->kernel_plan_index));
  item.Set("enabled", Bool(plan->enabled));
  item.Set("kind", plan->kind);
  item.Set("operation_name", plan->operation_name);
  Array<Any> operand_bindings;
  for (const TTComputeOperandBindingPlan &binding : plan->operand_bindings) {
    operand_bindings.push_back(EncodeComputeOperandBindingPlan(binding));
  }
  item.Set("operand_bindings", operand_bindings);
  if (!plan->problem_shape_axes.empty()) {
    item.Set("problem_shape_axes", plan->problem_shape_axes);
  }
  if (!plan->problem_shape.empty()) {
    item.Set("problem_shape", plan->problem_shape);
  }
  if (!plan->tile_shape.empty()) {
    item.Set("tile_shape", plan->tile_shape);
  }
  if (!plan->block_shape.empty()) {
    item.Set("block_shape", plan->block_shape);
  }
  if (!plan->subblock_shape.empty()) {
    item.Set("subblock_shape", plan->subblock_shape);
  }
  if (!plan->accumulator_dtype.empty()) {
    item.Set("accumulator_dtype", plan->accumulator_dtype);
  }
  if (!plan->mbarrier_buffer.empty()) {
    item.Set("mbarrier_buffer", plan->mbarrier_buffer);
  }
  if (!plan->mbarrier_scope.empty()) {
    item.Set("mbarrier_scope", plan->mbarrier_scope);
  }
  if (!plan->mbarrier_index_exprs.empty()) {
    item.Set("mbarrier_index_exprs", plan->mbarrier_index_exprs);
  }
  for (const TTComputeOperandBindingPlan &binding : plan->operand_bindings) {
    const std::string role = static_cast<std::string>(binding->role);
    if (role == "a") {
      item.Set("a_buffer", binding->host_buffer);
      item.Set("transpose_A", Bool(binding->transform_kind == "transpose"));
      if (!binding->tensor_dtype.empty()) {
        item.Set("a_tensor_dtype", binding->tensor_dtype);
      }
      if (!binding->cb_dtype.empty()) {
        item.Set("a_cb_dtype", binding->cb_dtype);
      }
    } else if (role == "b") {
      item.Set("b_buffer", binding->host_buffer);
      item.Set("transpose_B", Bool(binding->transform_kind == "transpose"));
      if (!binding->tensor_dtype.empty()) {
        item.Set("b_tensor_dtype", binding->tensor_dtype);
      }
      if (!binding->cb_dtype.empty()) {
        item.Set("b_cb_dtype", binding->cb_dtype);
      }
    } else if (role == "c") {
      item.Set("c_buffer", binding->host_buffer);
      if (!binding->tensor_dtype.empty()) {
        item.Set("c_tensor_dtype", binding->tensor_dtype);
      }
      if (!binding->cb_dtype.empty()) {
        item.Set("c_cb_dtype", binding->cb_dtype);
      }
    }
  }
  if (plan->problem_shape_axes.size() == plan->problem_shape.size()) {
    for (size_t i = 0; i < plan->problem_shape_axes.size(); ++i) {
      const std::string axis =
          static_cast<std::string>(plan->problem_shape_axes[i]);
      if (axis == "M" || axis == "N" || axis == "K") {
        item.Set(String(axis), plan->problem_shape[i]);
      }
    }
  } else if (plan->problem_shape.size() >= 3) {
    item.Set("M", plan->problem_shape[0]);
    item.Set("N", plan->problem_shape[1]);
    item.Set("K", plan->problem_shape[2]);
  }
  SetIntegerShapeField(&item, "Mt", plan->tile_shape, 0);
  SetIntegerShapeField(&item, "Nt", plan->tile_shape, 1);
  SetIntegerShapeField(&item, "Kt", plan->tile_shape, 2);
  SetIntegerShapeField(&item, "block_m_tiles", plan->block_shape, 0);
  SetIntegerShapeField(&item, "block_n_tiles", plan->block_shape, 1);
  SetIntegerShapeField(&item, "block_k_tiles", plan->block_shape, 2);
  SetIntegerShapeField(&item, "subblock_m_tiles", plan->subblock_shape, 0);
  SetIntegerShapeField(&item, "subblock_n_tiles", plan->subblock_shape, 1);
  item.Set("has_mbarrier", Bool(!plan->mbarrier_buffer.empty()));
  item.Set("tile_compute_dag_node_id", Integer(plan->tile_compute_dag_node_id));
  if (!plan->tile_compute_source_emitter.empty()) {
    item.Set("tile_compute_source_emitter", plan->tile_compute_source_emitter);
  }
  if (!plan->tile_compute_materialization_policy.empty()) {
    item.Set("tile_compute_materialization_policy",
             plan->tile_compute_materialization_policy);
  }
  item.Set("tile_compute_fanout_use_count",
           Integer(plan->tile_compute_fanout_use_count));
  if (!plan->tile_compute_fanout_policy.empty()) {
    item.Set("tile_compute_fanout_policy", plan->tile_compute_fanout_policy);
  }
  return item;
}

inline Array<Any>
EncodeComputeOpPlans(const Array<TTComputeOpPlan> &compute_op_plans) {
  Array<Any> encoded;
  for (const TTComputeOpPlan &plan : compute_op_plans) {
    encoded.push_back(EncodeComputeOpPlan(plan));
  }
  return encoded;
}

inline Map<String, Any> EncodeTileComputeMaterializationDemand(
    const TTTileComputeMaterializationDemand &demand) {
  Map<String, Any> item;
  item.Set("name", demand->name);
  item.Set("kernel_name", demand->kernel_name);
  item.Set("node_id", Integer(demand->node_id));
  item.Set("operation_name", demand->operation_name);
  item.Set("pattern_name", demand->pattern_name);
  item.Set("policy", demand->policy);
  item.Set("evidence", demand->evidence);
  return item;
}

inline Array<Any> EncodeTileComputeMaterializationDemands(
    const Array<TTTileComputeMaterializationDemand> &demands) {
  Array<Any> encoded;
  for (const TTTileComputeMaterializationDemand &demand : demands) {
    encoded.push_back(EncodeTileComputeMaterializationDemand(demand));
  }
  return encoded;
}

inline Array<Any>
EncodeResourcePressureReports(const Array<TTResourcePressureReport> &reports) {
  Array<Any> encoded;
  for (const TTResourcePressureReport &report : reports) {
    Map<String, Any> item;
    item.Set("name", report->name);
    item.Set("kernel_name", report->kernel_name);
    item.Set("core_group", report->core_group);
    item.Set("core_group_index", Integer(report->core_group_index));
    item.Set("tile_compute_unsupported_reasons",
             report->tile_compute_unsupported_reasons);
    item.Set("required_materializations",
             EncodeTileComputeMaterializationDemands(
                 report->required_materializations));
    item.Set("per_core_cb_id_pressure",
             Integer(report->per_core_cb_id_pressure));
    item.Set("per_core_cb_l1_bytes", Integer(report->per_core_cb_l1_bytes));
    item.Set("per_core_l1_buffer_bytes",
             Integer(report->per_core_l1_buffer_bytes));
    item.Set("max_simultaneous_l1_bytes",
             Integer(report->max_simultaneous_l1_bytes));
    item.Set("cb_id_limit", Integer(report->cb_id_limit));
    item.Set("worker_l1_budget_bytes", Integer(report->worker_l1_budget_bytes));
    item.Set("l1_alignment_bytes", Integer(report->l1_alignment_bytes));
    item.Set("per_core_cb_l1_aligned_bytes",
             Integer(report->per_core_cb_l1_aligned_bytes));
    item.Set("l1_alignment_waste_bytes",
             Integer(report->l1_alignment_waste_bytes));
    item.Set("core_grid_requirement", report->core_grid_requirement);
    item.Set("dram_view_requirement", report->dram_view_requirement);
    item.Set("unsupported_reasons", report->unsupported_reasons);
    encoded.push_back(item);
  }
  return encoded;
}

inline Map<String, Any> EncodeCoreGroup(const TTCoreGroup &core_group) {
  Map<String, Any> item;
  item.Set("logical_grid_x", Integer(core_group->logical_grid_x));
  item.Set("logical_grid_y", Integer(core_group->logical_grid_y));
  item.Set("logical_grid_z", Integer(core_group->logical_grid_z));
  item.Set("linearization", core_group->linearization);
  item.Set("physical_cores", core_group->physical_cores);
  item.Set("work_packets", core_group->work_packets);
  return item;
}

inline Array<Any>
EncodeSemaphorePlans(const Array<TTSemaphorePlan> &semaphore_plans) {
  Array<Any> encoded;
  for (const TTSemaphorePlan &sem : semaphore_plans) {
    Map<String, Any> item;
    item.Set("id", Integer(sem->semaphore_id));
    item.Set("initial_value", Integer(sem->initial_value));
    item.Set("core_type", sem->core_type);
    if (!sem->core_ranges.empty()) {
      item.Set("core_ranges", sem->core_ranges);
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline bool HasLaunchSpec(const TTKernelLaunchSpec &launch_spec) {
  return launch_spec.defined() && !launch_spec->core_type.empty();
}

inline Map<String, Any>
EncodeKernelLaunchSpec(const TTKernelLaunchSpec &launch_spec) {
  Map<String, Any> encoded;
  if (!HasLaunchSpec(launch_spec)) {
    return encoded;
  }
  encoded.Set("core_type", launch_spec->core_type);
  encoded.Set("processor", launch_spec->processor);
  encoded.Set("noc", launch_spec->noc);
  return encoded;
}

inline bool HasComputeConfig(const TTKernelComputeConfig &compute_config) {
  return compute_config.defined() && !compute_config->math_fidelity.empty();
}

inline Map<String, Any>
EncodeKernelComputeConfig(const TTKernelComputeConfig &compute_config) {
  Map<String, Any> encoded;
  if (!HasComputeConfig(compute_config)) {
    return encoded;
  }
  encoded.Set("math_fidelity", compute_config->math_fidelity);
  encoded.Set("fp32_dest_acc_en", Bool(compute_config->fp32_dest_acc_en));
  encoded.Set("dst_full_sync_en", Bool(compute_config->dst_full_sync_en));
  encoded.Set("math_approx_mode", Bool(compute_config->math_approx_mode));
  encoded.Set("unpack_to_dest_mode", compute_config->unpack_to_dest_mode);
  encoded.Set("bfp8_pack_precise", Bool(compute_config->bfp8_pack_precise));
  Array<Any> defines;
  for (const TTKernelDefine &define : compute_config->defines) {
    Map<String, Any> item;
    item.Set("name", define->name);
    item.Set("value", define->value);
    defines.push_back(item);
  }
  encoded.Set("defines", defines);
  Array<Any> named_compile_args;
  for (const TTKernelNamedCompileArg &arg :
       compute_config->named_compile_args) {
    Map<String, Any> item;
    item.Set("name", arg->name);
    item.Set("value", Integer(arg->value));
    named_compile_args.push_back(item);
  }
  encoded.Set("named_compile_args", named_compile_args);
  encoded.Set("clear_accum", Bool(compute_config->clear_accum));
  encoded.Set("k_pack", Integer(compute_config->k_pack));
  encoded.Set("wg_wait", Integer(compute_config->wg_wait));
  encoded.Set("policy_type", Integer(compute_config->policy_type));
  encoded.Set("policy_name", compute_config->policy_name);
  return encoded;
}

inline Array<Any>
EncodePerWorkArgSpecs(const Array<TTPerWorkArgSpec> &per_work_arg_specs) {
  Array<Any> encoded;
  for (const TTPerWorkArgSpec &spec : per_work_arg_specs) {
    Map<String, Any> item;
    item.Set(::tvm::tl::blackhole_runtime_arg_schema::kArgKind, spec->arg_kind);
    item.Set(::tvm::tl::blackhole_runtime_arg_schema::kArgIdentity,
             spec->arg_identity);
    if (!spec->buffer.empty()) {
      item.Set(::tvm::tl::blackhole_runtime_arg_schema::kBuffer, spec->buffer);
    }
    item.Set(::tvm::tl::blackhole_runtime_arg_schema::kDescriptorKind,
             spec->descriptor_kind);
    item.Set(::tvm::tl::blackhole_runtime_arg_schema::kValueSource,
             spec->value_source);
    if (spec->value_source ==
        ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceConstant) {
      item.Set(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue,
               Integer(spec->constant_value));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any>
EncodeRuntimeArgSpecs(const Array<TTRuntimeArgSpec> &runtime_args) {
  Array<Any> encoded;
  for (const TTRuntimeArgSpec &spec : runtime_args) {
    Map<String, Any> item;
    item.Set("name", spec->name);
    item.Set("kind", spec->kind);
    item.Set("dtype", spec->dtype);
    if (!spec->buffer.empty()) {
      item.Set("buffer", spec->buffer);
    }
    if (!spec->identity.empty()) {
      item.Set("identity", spec->identity);
    }
    if (spec->core_x >= 0) {
      item.Set("core_x", Integer(spec->core_x));
    }
    if (spec->core_y >= 0) {
      item.Set("core_y", Integer(spec->core_y));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeCompileTimeArgSpecs(
    const Array<TTCompileTimeArgSpec> &compile_time_arg_specs) {
  Array<Any> encoded;
  for (const TTCompileTimeArgSpec &spec : compile_time_arg_specs) {
    Map<String, Any> item;
    item.Set("name", spec->name);
    item.Set("kind", spec->kind);
    item.Set("dtype", spec->dtype);
    item.Set("offset", Integer(spec->offset));
    item.Set("count", Integer(spec->count));
    if (!spec->buffer.empty()) {
      item.Set("buffer", spec->buffer);
    }
    if (!spec->segment_role.empty()) {
      item.Set("segment_role", spec->segment_role);
    }
    if (!spec->values.empty()) {
      item.Set("values", spec->values);
    }
    if (spec->args_config_bits != 0) {
      item.Set("args_config_bits", Integer(spec->args_config_bits));
    }
    if (spec->transport_page_size > 0) {
      item.Set("transport_page_size", Integer(spec->transport_page_size));
    }
    if (!spec->layout.empty()) {
      item.Set("layout", spec->layout);
    }
    if (!spec->memory_space.empty()) {
      item.Set("memory_space", spec->memory_space);
    }
    if (!spec->host_axis_order.empty()) {
      item.Set("host_axis_order", spec->host_axis_order);
    }
    if (spec->transpose_2d) {
      item.Set("transpose_2d", Bool(true));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeAccessorSpecs(const Array<TTAccessorSpec> &accessors) {
  Array<Any> encoded;
  for (const TTAccessorSpec &spec : accessors) {
    Map<String, Any> item;
    item.Set("buffer", spec->buffer);
    item.Set("compile_time_arg_offset", Integer(spec->compile_time_arg_offset));
    item.Set("compile_time_arg_count", Integer(spec->compile_time_arg_count));
    item.Set("common_runtime_arg_offset",
             Integer(spec->common_runtime_arg_offset));
    item.Set("common_runtime_arg_count",
             Integer(spec->common_runtime_arg_count));
    item.Set("args_config_bits", Integer(spec->args_config_bits));
    if (spec->transport_page_size > 0) {
      item.Set("transport_page_size", Integer(spec->transport_page_size));
    }
    item.Set("layout", spec->layout);
    item.Set("memory_space", spec->memory_space);
    if (!spec->host_axis_order.empty()) {
      item.Set("host_axis_order", spec->host_axis_order);
    }
    if (spec->transpose_2d) {
      item.Set("transpose_2d", Bool(true));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeSemaphoreBindingSpecs(
    const Array<TTSemaphoreBindingSpec> &semaphore_bindings) {
  Array<Any> encoded;
  for (const TTSemaphoreBindingSpec &spec : semaphore_bindings) {
    Map<String, Any> item;
    item.Set("name", spec->name);
    item.Set("semaphore_id", Integer(spec->semaphore_id));
    item.Set("arg_kind", spec->arg_kind);
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any>
EncodeLiveFormPlans(const Array<TTLiveFormPlan> &live_form_plans) {
  Array<Any> encoded;
  for (const TTLiveFormPlan &plan : live_form_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("logical_value", plan->logical_value);
    item.Set("spatial_live_value", plan->spatial_live_value);
    item.Set("spatial_live_value_index",
             Integer(plan->spatial_live_value_index));
    item.Set("producer_kernel", plan->producer_kernel);
    item.Set("physical_form", plan->physical_form);
    item.Set("execution_topology", plan->execution_topology);
    item.Set("physical_local_extent", Integer(plan->physical_local_extent));
    item.Set("logical_element_count", Integer(plan->logical_element_count));
    item.Set("ownership_kind", plan->ownership_kind);
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeMaterializationPlans(
    const Array<TTMaterializationPlan> &materialization_plans) {
  Array<Any> encoded;
  for (const TTMaterializationPlan &plan : materialization_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("source_live_form", plan->source_live_form);
    item.Set("materialization_boundary", plan->materialization_boundary);
    item.Set("materialization_boundary_index",
             Integer(plan->materialization_boundary_index));
    item.Set("target_buffer", plan->target_buffer);
    item.Set("host_buffer", plan->host_buffer);
    item.Set("target_kernel", plan->target_kernel);
    if (!plan->bridge_kind.empty()) {
      item.Set("bridge_kind", plan->bridge_kind);
    }
    if (!plan->materialization_kind.empty()) {
      item.Set("materialization_kind", plan->materialization_kind);
    }
    item.Set("materialization_protocol", plan->materialization_protocol);
    item.Set("publication_protocol", plan->publication_protocol);
    item.Set("required_cb_plan_indices", plan->required_cb_plan_indices);
    if (!plan->required_sync_plan_indices.empty()) {
      item.Set("required_sync_plan_indices", plan->required_sync_plan_indices);
    }
    item.Set("produced_live_form", plan->produced_live_form);
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeConsumerBindingPlans(
    const Array<TTConsumerBindingPlan> &consumer_binding_plans) {
  Array<Any> encoded;
  for (const TTConsumerBindingPlan &plan : consumer_binding_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("consumer_kernel", plan->consumer_kernel);
    item.Set("consumer_op_kind", plan->consumer_op_kind);
    item.Set("source_live_form", plan->source_live_form);
    if (!plan->target_buffer.empty()) {
      item.Set("target_buffer", plan->target_buffer);
    }
    if (!plan->materialization_plan.empty()) {
      item.Set("materialization_plan", plan->materialization_plan);
    }
    item.Set("live_value_edge", plan->live_value_edge);
    item.Set("live_value_edge_index", Integer(plan->live_value_edge_index));
    item.Set("accepts_distributed_slice",
             Bool(plan->accepts_distributed_slice));
    item.Set("requires_full_logical_tile",
             Bool(plan->requires_full_logical_tile));
    if (plan->abi_plan_index >= 0) {
      item.Set("abi_plan_index", Integer(plan->abi_plan_index));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeSegmentPlan(const TTProgram &program) {
  Array<Any> segments;
  for (const TTKernel &kernel : program->kernels) {
    ICHECK_GE(kernel->abi_plan_index, 0);
    const TTABIPlan &abi =
        program->abi_plans[static_cast<size_t>(kernel->abi_plan_index)];
    Map<String, Any> segment;
    segment.Set("name", kernel->name);
    segment.Set("kind", kernel->kind);
    segment.Set("core_type", kernel->core_type);
    if (HasLaunchSpec(kernel->launch_spec)) {
      segment.Set("launch_spec", EncodeKernelLaunchSpec(kernel->launch_spec));
    }
    if (HasComputeConfig(kernel->compute_config)) {
      segment.Set("compute_config",
                  EncodeKernelComputeConfig(kernel->compute_config));
    }
    if (!kernel->per_work_arg_specs.empty()) {
      segment.Set(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs,
                  EncodePerWorkArgSpecs(kernel->per_work_arg_specs));
    }
    Array<Any> compute_ops;
    for (const TTComputeOpPlan &plan : program->compute_op_plans) {
      if (plan->kernel_name == kernel->name) {
        compute_ops.push_back(EncodeComputeOpPlan(plan));
      }
    }
    if (!compute_ops.empty()) {
      segment.Set("compute_ops", compute_ops);
    }
    if (!abi->runtime_args.empty()) {
      segment.Set("runtime_args", EncodeRuntimeArgSpecs(abi->runtime_args));
    }
    if (!abi->common_runtime_args.empty()) {
      segment.Set("common_runtime_args",
                  EncodeRuntimeArgSpecs(abi->common_runtime_args));
    }
    if (!abi->compile_time_arg_specs.empty()) {
      segment.Set("compile_time_arg_specs",
                  EncodeCompileTimeArgSpecs(abi->compile_time_arg_specs));
    }
    if (!abi->accessors.empty()) {
      segment.Set("accessors", EncodeAccessorSpecs(abi->accessors));
    }
    if (!abi->semaphore_bindings.empty()) {
      segment.Set("semaphore_bindings",
                  EncodeSemaphoreBindingSpecs(abi->semaphore_bindings));
    }
    segments.push_back(segment);
  }
  return segments;
}

inline Array<Any> GetSegmentPlanFromTTProgram(const TTProgram &program) {
  return EncodeSegmentPlan(program);
}

inline Array<Any> GetSegmentPlanFromTTProgram(const tir::PrimFunc &func,
                                              const char *consumer) {
  return EncodeSegmentPlan(RequireTTProgram(func, consumer));
}

inline Array<Any> GetCBConfigsFromTTProgram(const TTProgram &program) {
  return EncodeCBPlans(program->cb_plans);
}

inline Array<Any> GetCBConfigsFromTTProgram(const tir::PrimFunc &func,
                                            const char *consumer) {
  return EncodeCBPlans(RequireTTProgram(func, consumer)->cb_plans);
}

inline Array<Any> GetSemaphorePlanFromTTProgram(const TTProgram &program) {
  return EncodeSemaphorePlans(program->semaphore_plans);
}

inline Array<Any> GetSemaphorePlanFromTTProgram(const tir::PrimFunc &func,
                                                const char *consumer) {
  return EncodeSemaphorePlans(
      RequireTTProgram(func, consumer)->semaphore_plans);
}

inline Map<String, Any> GetCorePlanFromTTProgram(const TTProgram &program) {
  if (!program->core_groups.empty()) {
    return EncodeCoreGroup(program->core_groups[0]);
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetCorePlanFromTTProgram(const tir::PrimFunc &func,
                                                 const char *consumer) {
  return GetCorePlanFromTTProgram(RequireTTProgram(func, consumer));
}

inline void ValidateLiveProjectionEvidence(const TTProgram &program) {
  std::unordered_set<std::string> live_form_names;
  for (const TTLiveFormPlan &plan : program->live_form_plans) {
    ICHECK(!plan->name.empty())
        << "executable projection requires TTLiveFormPlan name";
    ICHECK_GE(plan->spatial_live_value_index, 0)
        << "executable projection requires TTLiveFormPlan "
           "spatial_live_value_index";
    live_form_names.insert(plan->name);
  }
  for (const TTMaterializationPlan &plan : program->materialization_plans) {
    ICHECK(!plan->name.empty())
        << "executable projection requires TTMaterializationPlan name";
    ICHECK(!plan->source_live_form.empty())
        << "executable projection requires TTMaterializationPlan "
           "source_live_form";
    ICHECK(
        live_form_names.count(static_cast<std::string>(plan->source_live_form)))
        << "executable projection requires TTMaterializationPlan "
           "source_live_form to reference "
           "a TTLiveFormPlan";
    ICHECK(!plan->produced_live_form.empty())
        << "executable projection requires TTMaterializationPlan "
           "produced_live_form";
    ICHECK(live_form_names.count(
        static_cast<std::string>(plan->produced_live_form)))
        << "executable projection requires TTMaterializationPlan "
           "produced_live_form to reference "
           "a TTLiveFormPlan";
    ICHECK(!plan->materialization_boundary.empty())
        << "executable projection requires TTMaterializationPlan "
           "materialization_boundary";
    ICHECK_GE(plan->materialization_boundary_index, 0)
        << "executable projection requires TTMaterializationPlan "
           "materialization_boundary_index";
    ICHECK(!plan->target_buffer.empty())
        << "executable projection requires TTMaterializationPlan target_buffer";
    ICHECK(!plan->materialization_protocol.empty())
        << "executable projection requires TTMaterializationPlan "
           "materialization_protocol";
    ICHECK(!plan->publication_protocol.empty())
        << "executable projection requires TTMaterializationPlan "
           "publication_protocol";
  }
  for (const TTConsumerBindingPlan &plan : program->consumer_binding_plans) {
    ICHECK(!plan->name.empty())
        << "executable projection requires TTConsumerBindingPlan name";
    ICHECK(!plan->source_live_form.empty())
        << "executable projection requires TTConsumerBindingPlan "
           "source_live_form";
    ICHECK(
        live_form_names.count(static_cast<std::string>(plan->source_live_form)))
        << "executable projection requires TTConsumerBindingPlan "
           "source_live_form to reference "
           "a TTLiveFormPlan";
    ICHECK(!plan->live_value_edge.empty())
        << "executable projection requires TTConsumerBindingPlan "
           "live_value_edge";
    ICHECK_GE(plan->live_value_edge_index, 0)
        << "executable projection requires TTConsumerBindingPlan "
           "live_value_edge_index";
    ICHECK(plan->accepts_distributed_slice || plan->requires_full_logical_tile)
        << "executable projection requires TTConsumerBindingPlan to declare a "
           "consumer "
           "coverage requirement";
    ICHECK(
        !(plan->accepts_distributed_slice && plan->requires_full_logical_tile))
        << "executable projection requires TTConsumerBindingPlan to choose one "
           "consumer "
           "coverage requirement";
  }
}

inline bool HasPositiveIntegerShape(const Array<Integer> &shape) {
  if (shape.empty()) {
    return false;
  }
  for (const Integer &value : shape) {
    if (value->value <= 0) {
      return false;
    }
  }
  return true;
}

inline bool IsShardedMemoryLayout(const String &memory_layout) {
  const std::string layout = memory_layout;
  return layout == "HEIGHT_SHARDED" || layout == "WIDTH_SHARDED" ||
         layout == "BLOCK_SHARDED" || layout == "ND_SHARDED";
}

inline void ValidatePlacementProjectionEvidence(const TTProgram &program) {
  std::unordered_map<std::string, int64_t> memory_config_index_by_name;
  std::unordered_map<std::string, int64_t> memory_config_index_by_subject;
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->tensor_memory_config_plans.size());
       ++index) {
    const TTTensorMemoryConfigPlan &plan =
        program->tensor_memory_config_plans[static_cast<size_t>(index)];
    ICHECK(!plan->name.empty())
        << "executable projection requires TTTensorMemoryConfigPlan name";
    ICHECK(!plan->subject.empty())
        << "executable projection requires TTTensorMemoryConfigPlan subject";
    ICHECK(!plan->memory_layout.empty())
        << "executable projection requires TTTensorMemoryConfigPlan "
           "memory_layout";
    ICHECK(!plan->buffer_type.empty())
        << "executable projection requires TTTensorMemoryConfigPlan "
           "buffer_type";
    ICHECK(!plan->origin.empty())
        << "executable projection requires TTTensorMemoryConfigPlan origin";
    ICHECK(memory_config_index_by_name
               .emplace(static_cast<std::string>(plan->name), index)
               .second)
        << "executable projection has duplicate TTTensorMemoryConfigPlan name "
        << plan->name;
    ICHECK(
        memory_config_index_by_subject
            .emplace(static_cast<std::string>(plan->subject), index)
            .second)
        << "executable projection has duplicate TTTensorMemoryConfigPlan "
           "subject "
        << plan->subject;
    if (IsShardedMemoryLayout(plan->memory_layout)) {
      ICHECK(!plan->grid_ref.empty())
          << "executable projection sharded TTTensorMemoryConfigPlan "
          << plan->name << " requires grid_ref";
      ICHECK(HasPositiveIntegerShape(plan->shard_grid_shape))
          << "executable projection sharded TTTensorMemoryConfigPlan "
          << plan->name << " requires shard_grid_shape";
      ICHECK(HasPositiveIntegerShape(plan->shard_shape))
          << "executable projection sharded TTTensorMemoryConfigPlan "
          << plan->name << " requires shard_shape";
      ICHECK(plan->shard_orientation == "row_major" ||
             plan->shard_orientation == "col_major")
          << "executable projection sharded TTTensorMemoryConfigPlan "
          << plan->name << " requires row_major or col_major orientation";
      ICHECK(plan->shard_distribution_strategy == "height" ||
             plan->shard_distribution_strategy == "width" ||
             plan->shard_distribution_strategy == "block" ||
             plan->shard_distribution_strategy == "nd")
          << "executable projection sharded TTTensorMemoryConfigPlan "
          << plan->name << " requires sharding strategy";
    }
  }

  std::unordered_set<std::string> materialization_names;
  for (const TTMaterializationPlan &plan : program->materialization_plans) {
    if (!plan->name.empty()) {
      materialization_names.insert(static_cast<std::string>(plan->name));
    }
  }

  for (const TTReshardPlan &plan : program->reshard_plans) {
    ICHECK(!plan->name.empty())
        << "executable projection requires TTReshardPlan name";
    ICHECK(!plan->source_value.empty())
        << "executable projection requires TTReshardPlan source_value";
    ICHECK(!plan->target_value.empty())
        << "executable projection requires TTReshardPlan target_value";
    ICHECK_NE(plan->source_value, plan->target_value)
        << "executable projection TTReshardPlan source_value and "
           "target_value must differ";
    ICHECK_GE(plan->source_memory_config_plan_index, 0)
        << "executable projection TTReshardPlan requires "
           "source_memory_config_plan_index";
    ICHECK_LT(plan->source_memory_config_plan_index,
              static_cast<int64_t>(
                  program->tensor_memory_config_plans.size()))
        << "executable projection TTReshardPlan "
           "source_memory_config_plan_index out of bounds";
    ICHECK_GE(plan->target_memory_config_plan_index, 0)
        << "executable projection TTReshardPlan requires "
           "target_memory_config_plan_index";
    ICHECK_LT(plan->target_memory_config_plan_index,
              static_cast<int64_t>(
                  program->tensor_memory_config_plans.size()))
        << "executable projection TTReshardPlan "
           "target_memory_config_plan_index out of bounds";
    const TTTensorMemoryConfigPlan &source_config =
        program->tensor_memory_config_plans[static_cast<size_t>(
            plan->source_memory_config_plan_index)];
    const TTTensorMemoryConfigPlan &target_config =
        program->tensor_memory_config_plans[static_cast<size_t>(
            plan->target_memory_config_plan_index)];
    ICHECK_EQ(plan->source_memory_config_plan, source_config->name)
        << "executable projection TTReshardPlan "
           "source_memory_config_plan must match indexed "
           "TTTensorMemoryConfigPlan";
    ICHECK_EQ(plan->target_memory_config_plan, target_config->name)
        << "executable projection TTReshardPlan "
           "target_memory_config_plan must match indexed "
           "TTTensorMemoryConfigPlan";
    ICHECK_EQ(plan->source_value, source_config->subject)
        << "executable projection TTReshardPlan source_value must match "
           "source memory config subject";
    ICHECK_EQ(plan->target_value, target_config->subject)
        << "executable projection TTReshardPlan target_value must match "
           "target memory config subject";
    const std::string conversion_kind = plan->conversion_kind;
    ICHECK(conversion_kind == "interleaved_to_sharded" ||
           conversion_kind == "sharded_to_interleaved" ||
           conversion_kind == "reshard" || conversion_kind == "unsupported")
        << "executable projection TTReshardPlan unsupported conversion_kind "
        << plan->conversion_kind;
    if (conversion_kind == "interleaved_to_sharded") {
      ICHECK_EQ(source_config->memory_layout, "INTERLEAVED")
          << "executable projection TTReshardPlan "
             "interleaved_to_sharded requires interleaved source";
      ICHECK_NE(target_config->memory_layout, "INTERLEAVED")
          << "executable projection TTReshardPlan "
             "interleaved_to_sharded requires sharded target";
      ICHECK(!plan->materialization_protocol.empty())
          << "executable projection TTReshardPlan "
             "interleaved_to_sharded requires materialization_protocol";
      ICHECK(!plan->source_region_kind.empty() &&
             plan->source_region_kind != "none")
          << "executable projection TTReshardPlan "
             "interleaved_to_sharded requires source_region_kind";
      ICHECK(HasPositiveIntegerShape(plan->source_region_shape))
          << "executable projection TTReshardPlan "
             "interleaved_to_sharded requires source_region_shape";
    }
    if (plan->materialization_plan_index >= 0) {
      ICHECK_LT(plan->materialization_plan_index,
                static_cast<int64_t>(program->materialization_plans.size()))
          << "executable projection TTReshardPlan "
             "materialization_plan_index out of bounds";
      ICHECK(materialization_names.count(
          static_cast<std::string>(plan->materialization_plan)))
          << "executable projection TTReshardPlan materialization_plan must "
             "reference a TTMaterializationPlan";
    }
    ICHECK(plan->scheduling_kind == "runtime" ||
           plan->scheduling_kind == "load_time" ||
           plan->scheduling_kind == "compile_time")
        << "executable projection TTReshardPlan unsupported scheduling_kind "
        << plan->scheduling_kind;
    ICHECK(plan->inserted_by == "planner" || plan->inserted_by == "user")
        << "executable projection TTReshardPlan unsupported inserted_by "
        << plan->inserted_by;
    ICHECK(plan->admission_status == "admitted" ||
           plan->admission_status == "unsupported")
        << "executable projection TTReshardPlan unsupported admission_status "
        << plan->admission_status;
    if (plan->admission_status == "admitted") {
      ICHECK(plan->unsupported_reason.empty())
          << "executable projection TTReshardPlan admitted conversion cannot "
             "carry unsupported_reason";
    } else {
      ICHECK(!plan->unsupported_reason.empty())
          << "executable projection TTReshardPlan unsupported conversion "
             "requires unsupported_reason";
    }
  }
}

inline Map<String, Any>
MaterializeBlackholeExecutableProjection(const TTProgram &program) {
  ValidateLiveProjectionEvidence(program);
  ValidatePlacementProjectionEvidence(program);

  Map<String, Any> executable;
  executable.Set(String(executable_key::kSchemaVersion), Integer(1));
  executable.Set(String(executable_key::kSource), String(attr::kTLTTProgram));
  executable.Set(String(executable_key::kEntryName), program->entry_name);
  if (!program->member_func.empty()) {
    executable.Set(String(executable_key::kMemberFunc), program->member_func);
  }

  Array<Any> mesh_plans = EncodeMeshPlans(program->mesh_plans);
  if (!mesh_plans.empty()) {
    executable.Set(String(executable_key::kMeshPlans), mesh_plans);
  }

  Array<Any> buffer_distribution_plans =
      EncodeBufferDistributionPlans(program->buffer_distribution_plans);
  if (!buffer_distribution_plans.empty()) {
    executable.Set(String(executable_key::kBufferDistributionPlans),
                   buffer_distribution_plans);
  }

  Array<Any> tensor_memory_config_plans =
      EncodeTensorMemoryConfigPlans(program->tensor_memory_config_plans);
  if (!tensor_memory_config_plans.empty()) {
    executable.Set(String(executable_key::kTensorMemoryConfigPlans),
                   tensor_memory_config_plans);
  }

  Array<Any> op_sharding_contracts =
      EncodeOpShardingContracts(program->op_sharding_contracts);
  if (!op_sharding_contracts.empty()) {
    executable.Set(String(executable_key::kOpShardingContracts),
                   op_sharding_contracts);
  }

  Array<Any> placement_resolution_plans =
      EncodePlacementResolutionPlans(program->placement_resolution_plans);
  if (!placement_resolution_plans.empty()) {
    executable.Set(String(executable_key::kPlacementResolutionPlans),
                   placement_resolution_plans);
  }

  Array<Any> reshard_plans = EncodeReshardPlans(program->reshard_plans);
  if (!reshard_plans.empty()) {
    executable.Set(String(executable_key::kReshardPlans), reshard_plans);
  }

  Array<Any> compute_op_plans = EncodeComputeOpPlans(program->compute_op_plans);
  if (!compute_op_plans.empty()) {
    executable.Set(String(executable_key::kComputeOpPlans), compute_op_plans);
  }

  Array<Any> segment_plan = EncodeSegmentPlan(program);
  if (!segment_plan.empty()) {
    executable.Set(String(executable_key::kSegmentPlan), segment_plan);
  }

  Array<Any> cb_configs = EncodeCBPlans(program->cb_plans);
  if (!cb_configs.empty()) {
    executable.Set(String(executable_key::kCBConfigs), cb_configs);
  }

  Map<String, Any> core_plan = GetCorePlanFromTTProgram(program);
  if (!core_plan.empty()) {
    executable.Set(String(executable_key::kCorePlan), core_plan);
  }

  Array<Any> semaphore_plan = EncodeSemaphorePlans(program->semaphore_plans);
  if (!semaphore_plan.empty()) {
    executable.Set(String(executable_key::kSemaphorePlan), semaphore_plan);
  }

  Array<Any> live_form_plans = EncodeLiveFormPlans(program->live_form_plans);
  if (!live_form_plans.empty()) {
    executable.Set(String(executable_key::kLiveFormPlans), live_form_plans);
  }

  Array<Any> materialization_plans =
      EncodeMaterializationPlans(program->materialization_plans);
  if (!materialization_plans.empty()) {
    executable.Set(String(executable_key::kMaterializationPlans),
                   materialization_plans);
  }

  Array<Any> consumer_binding_plans =
      EncodeConsumerBindingPlans(program->consumer_binding_plans);
  if (!consumer_binding_plans.empty()) {
    executable.Set(String(executable_key::kConsumerBindingPlans),
                   consumer_binding_plans);
  }
  Array<Any> resource_pressure_reports =
      EncodeResourcePressureReports(program->resource_pressure_reports);
  if (!resource_pressure_reports.empty()) {
    executable.Set(String(executable_key::kResourcePressureReports),
                   resource_pressure_reports);
  }
  return executable;
}

inline tir::PrimFunc
MaterializeBlackholeExecutableProjectionAttr(const tir::PrimFunc &func) {
  auto maybe_program = GetTTProgram(func);
  if (!maybe_program) {
    return func;
  }
  return WithAttr(
      std::move(func), attr::kTLBlackholeExecutable,
      MaterializeBlackholeExecutableProjection(maybe_program.value()));
}

inline Map<String, Any>
GetBlackholeExecutableProjection(const tir::PrimFunc &func) {
  return func->GetAttr<Map<String, Any>>(attr::kTLBlackholeExecutable)
      .value_or(Map<String, Any>());
}

inline Map<String, Any>
RequireBlackholeExecutableProjection(const tir::PrimFunc &func,
                                     const char *consumer) {
  Map<String, Any> executable = GetBlackholeExecutableProjection(func);
  ICHECK(!executable.empty())
      << consumer
      << " requires tl.blackhole_executable for executable-writer cutover";
  return executable;
}

inline Array<Any> GetExecutableArrayField(const Map<String, Any> &executable,
                                          const char *key) {
  if (auto value = executable.Get(String(key))) {
    return Downcast<Array<Any>>(value.value());
  }
  return Array<Any>();
}

inline Array<Any> GetExecutableArrayField(const tir::PrimFunc &func,
                                          const char *consumer,
                                          const char *key) {
  return GetExecutableArrayField(
      RequireBlackholeExecutableProjection(func, consumer), key);
}

inline Map<String, Any>
GetExecutableMapField(const Map<String, Any> &executable, const char *key) {
  if (auto value = executable.Get(String(key))) {
    return AsMap(value.value());
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetExecutableMapField(const tir::PrimFunc &func,
                                              const char *consumer,
                                              const char *key) {
  return GetExecutableMapField(
      RequireBlackholeExecutableProjection(func, consumer), key);
}

inline Array<Any> GetSegmentPlanFromExecutable(const tir::PrimFunc &func,
                                               const char *consumer) {
  return GetExecutableArrayField(func, consumer, executable_key::kSegmentPlan);
}

inline Array<Any> GetCBConfigsFromExecutable(const tir::PrimFunc &func,
                                             const char *consumer) {
  return GetExecutableArrayField(func, consumer, executable_key::kCBConfigs);
}

inline Map<String, Any> GetCorePlanFromExecutable(const tir::PrimFunc &func,
                                                  const char *consumer) {
  return GetExecutableMapField(func, consumer, executable_key::kCorePlan);
}

} // namespace tt_program_projection
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_
