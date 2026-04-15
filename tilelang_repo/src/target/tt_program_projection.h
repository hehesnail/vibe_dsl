/*!
 * \file tt_program_projection.h
 * \brief In-memory TTProgram projections for runtime/codegen direct readers.
 */

#ifndef TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_
#define TVM_TL_TARGET_TT_PROGRAM_PROJECTION_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/function.h>

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
constexpr const char* kSegmentPlan = "segment_plan";
constexpr const char* kCBConfigs = "cb_configs";
constexpr const char* kCorePlan = "core_plan";
constexpr const char* kSemaphorePlan = "semaphore_plan";
constexpr const char* kGemmContract = "gemm_contract";
constexpr const char* kComputeContract = "compute_contract";
constexpr const char* kMultiGemmContracts = "multi_gemm_contracts";
constexpr const char* kMultiComputeContracts = "multi_compute_contracts";
constexpr const char* kComputeEpilogueOps = "compute_epilogue_ops";
constexpr const char* kDirectRuntimeUnsupportedReasons = "direct_runtime_unsupported_reasons";
constexpr const char* kBufferTileBridgeSpecs = ::tvm::tl::schema_key::kBufferTileBridgeSpecs;
constexpr const char* kUnsupportedComputeOps = "unsupported_compute_ops";
}  // namespace executable_key

inline Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

inline tvm::ffi::Optional<TTProgram> GetTTProgram(const tir::PrimFunc& func) {
  return func->GetAttr<TTProgram>(attr::kTLTTProgram);
}

inline TTProgram RequireTTProgram(const tir::PrimFunc& func, const char* consumer) {
  auto maybe_program = GetTTProgram(func);
  ICHECK(maybe_program) << consumer << " requires tl.tt_program for target-truth cutover";
  return maybe_program.value();
}

inline Array<Any> EncodeCBPlans(const Array<TTCBPlan>& cb_plans) {
  Array<Any> encoded;
  for (const TTCBPlan& cb : cb_plans) {
    Map<String, Any> item = cb->payload;
    item.Set("name", cb->name);
    item.Set("cb_id", Integer(cb->cb_id));
    item.Set("role", cb->resource_class);
    item.Set("num_pages", Integer(cb->num_pages));
    item.Set("page_size", Integer(cb->page_size_bytes));
    item.Set("data_format", cb->data_format);
    encoded.push_back(item);
  }
  return encoded;
}

inline Map<String, Any> EncodeCoreGroup(const TTCoreGroup& core_group) {
  Map<String, Any> item = core_group->payload;
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
    Map<String, Any> item = sem->payload;
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

inline Array<Any> EncodeSegmentPlan(const TTProgram& program) {
  Array<Any> segments;
  for (const TTKernel& kernel : program->kernels) {
    ICHECK_GE(kernel->abi_plan_index, 0);
    const TTABIPlan& abi = program->abi_plans[static_cast<size_t>(kernel->abi_plan_index)];
    Map<String, Any> segment = kernel->payload;
    segment.Set("name", kernel->name);
    segment.Set("kind", kernel->kind);
    segment.Set("core_type", kernel->core_type);
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
    if (auto v = kernel->payload.Get(
            String(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs))) {
      segment.Set(String(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs),
                  v.value());
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

inline Map<String, Any> GetGemmContractFromTTProgram(const TTProgram& program) {
  if (auto contract = program->payload.Get("gemm_contract")) {
    return AsMap(contract.value());
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetGemmContractFromTTProgram(const tir::PrimFunc& func,
                                                     const char* consumer) {
  return GetGemmContractFromTTProgram(RequireTTProgram(func, consumer));
}

inline Map<String, Any> GetComputeContractFromTTProgram(const TTProgram& program) {
  if (auto contract = program->payload.Get("compute_contract")) {
    return AsMap(contract.value());
  }
  return Map<String, Any>();
}

inline Map<String, Any> GetComputeContractFromTTProgram(const tir::PrimFunc& func,
                                                        const char* consumer) {
  return GetComputeContractFromTTProgram(RequireTTProgram(func, consumer));
}

inline Map<String, Any> MaterializeBlackholeExecutableProjection(const TTProgram& program) {
  Map<String, Any> executable;
  executable.Set(String(executable_key::kSchemaVersion), Integer(1));
  executable.Set(String(executable_key::kSource), String(attr::kTLTTProgram));
  executable.Set(String(executable_key::kEntryName), program->entry_name);
  if (!program->member_func.empty()) {
    executable.Set(String(executable_key::kMemberFunc), program->member_func);
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

  auto copy_payload_field = [&](const char* key) {
    if (auto value = program->payload.Get(String(key))) {
      executable.Set(String(key), value.value());
    }
  };
  copy_payload_field(executable_key::kGemmContract);
  copy_payload_field(executable_key::kComputeContract);
  copy_payload_field(executable_key::kMultiGemmContracts);
  copy_payload_field(executable_key::kMultiComputeContracts);
  copy_payload_field(executable_key::kComputeEpilogueOps);
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
