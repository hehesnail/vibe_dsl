/*!
 * \file tt_program_projection.h
 * \brief TTProgram -> ExecutableSpec projection helpers for the canonical writer boundary.
 */

#ifndef TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_
#define TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/function.h>

#include <initializer_list>
#include <string>
#include "../transform/common/blackhole_runtime_arg_schema.h"
#include "../transform/common/companion_base.h"
#include "../transform/common/tt_target_program.h"

namespace tvm {
namespace tl {
namespace tt_program_projection {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace executable_key {
constexpr const char* kSchemaVersion = "schema_version";
constexpr const char* kSource = "source";
constexpr const char* kEntryName = "entry_name";
constexpr const char* kMemberFunc = "member_func";
constexpr const char* kMeshPlans = "mesh_plans";
constexpr const char* kBufferDistributionPlans = "buffer_distribution_plans";
constexpr const char* kSegmentPlan = "segment_plan";
constexpr const char* kCBConfigs = "cb_configs";
constexpr const char* kCorePlan = "core_plan";
constexpr const char* kSemaphorePlan = "semaphore_plan";
constexpr const char* kDirectRuntimeUnsupportedReasons = "direct_runtime_unsupported_reasons";
constexpr const char* kLiveFormPlans = "live_form_plans";
constexpr const char* kMaterializationPlans = "materialization_plans";
constexpr const char* kConsumerBindingPlans = "consumer_binding_plans";
constexpr const char* kBufferTileBridgeSpecs = ::tvm::tl::schema_key::kBufferTileBridgeSpecs;
constexpr const char* kUnsupportedComputeOps = "unsupported_compute_ops";
}  // namespace executable_key

inline Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

inline void CopyPayloadAllowlist(Map<String, Any>* dst, const Map<String, Any>& payload,
                                 std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (auto value = payload.Get(String(key))) {
      dst->Set(String(key), value.value());
    }
  }
}

inline tvm::ffi::Optional<TTProgram> GetTTProgram(const tir::PrimFunc& func) {
  return func->GetAttr<TTProgram>(attr::kTLTTProgram);
}

inline TTProgram RequireTTProgram(const tir::PrimFunc& func, const char* consumer) {
  auto maybe_program = GetTTProgram(func);
  ICHECK(maybe_program) << consumer << " requires tl.tt_program for executable-writer cutover";
  return maybe_program.value();
}

inline Array<Any> EncodeCBPlans(const Array<TTCBPlan>& cb_plans) {
  Array<Any> encoded;
  for (const TTCBPlan& cb : cb_plans) {
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
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeMeshPlans(const Array<TTMeshPlan>& mesh_plans) {
  Array<Any> encoded;
  for (const TTMeshPlan& plan : mesh_plans) {
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
    const Array<TTBufferDistributionPlan>& buffer_distribution_plans) {
  Array<Any> encoded;
  for (const TTBufferDistributionPlan& plan : buffer_distribution_plans) {
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
    item.Set("shard_orientation", plan->shard_orientation);
    item.Set("host_visibility", plan->host_visibility);
    encoded.push_back(item);
  }
  return encoded;
}

inline Map<String, Any> EncodeCoreGroup(const TTCoreGroup& core_group) {
  Map<String, Any> item;
  item.Set("logical_grid_x", Integer(core_group->logical_grid_x));
  item.Set("logical_grid_y", Integer(core_group->logical_grid_y));
  item.Set("linearization", core_group->linearization);
  item.Set("physical_cores", core_group->physical_cores);
  item.Set("work_packets", core_group->work_packets);
  return item;
}

inline Array<Any> EncodeSemaphorePlans(const Array<TTSemaphorePlan>& semaphore_plans) {
  Array<Any> encoded;
  for (const TTSemaphorePlan& sem : semaphore_plans) {
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

inline Array<Any> EncodeLiveFormPlans(const Array<TTLiveFormPlan>& live_form_plans) {
  Array<Any> encoded;
  for (const TTLiveFormPlan& plan : live_form_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("logical_value", plan->logical_value);
    item.Set("spatial_live_value", plan->spatial_live_value);
    item.Set("spatial_live_value_index", Integer(plan->spatial_live_value_index));
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
    const Array<TTMaterializationPlan>& materialization_plans) {
  Array<Any> encoded;
  for (const TTMaterializationPlan& plan : materialization_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("source_live_form", plan->source_live_form);
    item.Set("materialization_boundary", plan->materialization_boundary);
    item.Set("materialization_boundary_index", Integer(plan->materialization_boundary_index));
    item.Set("target_buffer", plan->target_buffer);
    item.Set("host_buffer", plan->host_buffer);
    item.Set("target_kernel", plan->target_kernel);
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
    const Array<TTConsumerBindingPlan>& consumer_binding_plans) {
  Array<Any> encoded;
  for (const TTConsumerBindingPlan& plan : consumer_binding_plans) {
    Map<String, Any> item;
    item.Set("name", plan->name);
    item.Set("consumer_kernel", plan->consumer_kernel);
    item.Set("consumer_op_kind", plan->consumer_op_kind);
    item.Set("source_live_form", plan->source_live_form);
    item.Set("live_value_edge", plan->live_value_edge);
    item.Set("live_value_edge_index", Integer(plan->live_value_edge_index));
    item.Set("accepts_distributed_slice", Bool(plan->accepts_distributed_slice));
    item.Set("requires_full_logical_tile", Bool(plan->requires_full_logical_tile));
    if (plan->abi_plan_index >= 0) {
      item.Set("abi_plan_index", Integer(plan->abi_plan_index));
    }
    encoded.push_back(item);
  }
  return encoded;
}

inline Array<Any> EncodeSegmentPlan(const TTProgram& program) {
  Array<Any> segments;
  for (const TTKernel& kernel : program->kernels) {
    ICHECK_GE(kernel->abi_plan_index, 0);
    const TTABIPlan& abi = program->abi_plans[static_cast<size_t>(kernel->abi_plan_index)];
    Map<String, Any> segment;
    segment.Set("name", kernel->name);
    segment.Set("kind", kernel->kind);
    segment.Set("core_type", kernel->core_type);
    CopyPayloadAllowlist(
        &segment, kernel->payload,
        {"launch_spec", "compute_config", "compute_ops",
         ::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs});
    if (!abi->runtime_args.empty()) {
      segment.Set("runtime_args", abi->runtime_args);
    }
    if (!abi->common_runtime_args.empty()) {
      segment.Set("common_runtime_args", abi->common_runtime_args);
    }
    if (!abi->compile_time_arg_specs.empty()) {
      segment.Set("compile_time_arg_specs", abi->compile_time_arg_specs);
    }
    if (!abi->accessors.empty()) {
      segment.Set("accessors", abi->accessors);
    }
    if (!abi->semaphore_bindings.empty()) {
      segment.Set("semaphore_bindings", abi->semaphore_bindings);
    }
    segments.push_back(segment);
  }
  return segments;
}

inline Array<Any> GetSegmentPlanFromTTProgram(const TTProgram& program) {
  return EncodeSegmentPlan(program);
}

inline Array<Any> GetSegmentPlanFromTTProgram(const tir::PrimFunc& func, const char* consumer) {
  return EncodeSegmentPlan(RequireTTProgram(func, consumer));
}

inline Array<Any> GetCBConfigsFromTTProgram(const TTProgram& program) {
  return EncodeCBPlans(program->cb_plans);
}

inline Array<Any> GetCBConfigsFromTTProgram(const tir::PrimFunc& func, const char* consumer) {
  return EncodeCBPlans(RequireTTProgram(func, consumer)->cb_plans);
}

inline Array<Any> GetSemaphorePlanFromTTProgram(const TTProgram& program) {
  return EncodeSemaphorePlans(program->semaphore_plans);
}

inline Array<Any> GetSemaphorePlanFromTTProgram(const tir::PrimFunc& func,
                                                const char* consumer) {
  return EncodeSemaphorePlans(RequireTTProgram(func, consumer)->semaphore_plans);
}

inline Map<String, Any> GetCorePlanFromTTProgram(const TTProgram& program) {
  if (!program->core_groups.empty()) {
    return EncodeCoreGroup(program->core_groups[0]);
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetCorePlanFromTTProgram(const tir::PrimFunc& func, const char* consumer) {
  return GetCorePlanFromTTProgram(RequireTTProgram(func, consumer));
}

inline Array<Any> GetDirectRuntimeUnsupportedReasonsFromTTProgram(const TTProgram& program) {
  if (auto reasons = program->payload.Get("direct_runtime_unsupported_reasons")) {
    return Downcast<Array<Any>>(reasons.value());
  }
  return Array<Any>();
}

inline Array<Any> GetDirectRuntimeUnsupportedReasonsFromTTProgram(const tir::PrimFunc& func,
                                                                  const char* consumer) {
  return GetDirectRuntimeUnsupportedReasonsFromTTProgram(RequireTTProgram(func, consumer));
}

inline Map<String, Any> MaterializeBlackholeExecutableProjection(const TTProgram& program) {
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
    executable.Set(String(executable_key::kMaterializationPlans), materialization_plans);
  }

  Array<Any> consumer_binding_plans =
      EncodeConsumerBindingPlans(program->consumer_binding_plans);
  if (!consumer_binding_plans.empty()) {
    executable.Set(String(executable_key::kConsumerBindingPlans), consumer_binding_plans);
  }

  auto copy_payload_field = [&](const char* key) {
    if (auto value = program->payload.Get(String(key))) {
      executable.Set(String(key), value.value());
    }
  };
  copy_payload_field(executable_key::kDirectRuntimeUnsupportedReasons);
  copy_payload_field(executable_key::kBufferTileBridgeSpecs);
  copy_payload_field(executable_key::kUnsupportedComputeOps);
  return executable;
}

inline tir::PrimFunc MaterializeBlackholeExecutableProjectionAttr(const tir::PrimFunc& func) {
  auto maybe_program = GetTTProgram(func);
  if (!maybe_program) {
    return func;
  }
  return WithAttr(std::move(func), attr::kTLBlackholeExecutable,
                  MaterializeBlackholeExecutableProjection(maybe_program.value()));
}

inline Map<String, Any> GetBlackholeExecutableProjection(const tir::PrimFunc& func) {
  return func->GetAttr<Map<String, Any>>(attr::kTLBlackholeExecutable)
      .value_or(Map<String, Any>());
}

inline Map<String, Any> RequireBlackholeExecutableProjection(const tir::PrimFunc& func,
                                                             const char* consumer) {
  Map<String, Any> executable = GetBlackholeExecutableProjection(func);
  ICHECK(!executable.empty())
      << consumer << " requires tl.blackhole_executable for executable-writer cutover";
  return executable;
}

inline Array<Any> GetExecutableArrayField(const Map<String, Any>& executable, const char* key) {
  if (auto value = executable.Get(String(key))) {
    return Downcast<Array<Any>>(value.value());
  }
  return Array<Any>();
}

inline Array<Any> GetExecutableArrayField(const tir::PrimFunc& func, const char* consumer,
                                          const char* key) {
  return GetExecutableArrayField(RequireBlackholeExecutableProjection(func, consumer), key);
}

inline Map<String, Any> GetExecutableMapField(const Map<String, Any>& executable, const char* key) {
  if (auto value = executable.Get(String(key))) {
    return AsMap(value.value());
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetExecutableMapField(const tir::PrimFunc& func, const char* consumer,
                                              const char* key) {
  return GetExecutableMapField(RequireBlackholeExecutableProjection(func, consumer), key);
}

inline Array<Any> GetSegmentPlanFromExecutable(const tir::PrimFunc& func, const char* consumer) {
  return GetExecutableArrayField(func, consumer, executable_key::kSegmentPlan);
}

inline Array<Any> GetCBConfigsFromExecutable(const tir::PrimFunc& func, const char* consumer) {
  return GetExecutableArrayField(func, consumer, executable_key::kCBConfigs);
}

inline Map<String, Any> GetCorePlanFromExecutable(const tir::PrimFunc& func, const char* consumer) {
  return GetExecutableMapField(func, consumer, executable_key::kCorePlan);
}

}  // namespace tt_program_projection
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_
