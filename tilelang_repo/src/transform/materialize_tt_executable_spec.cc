/*!
 * \file materialize_tt_executable_spec.cc
 * \brief Materialize legacy Blackhole attrs from TTProgram as the single writer.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>

#include <string>
#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

Map<String, Any> CopyWithoutLegacyTTAttrs(const tir::PrimFunc& func) {
  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    for (const auto& kv : func->attrs->dict) {
      const std::string key = kv.first;
      if (key == "blackhole.segment_plan" || key == "blackhole.runtime_args" ||
          key == "blackhole.common_runtime_args" || key == "blackhole.accessors" ||
          key == "blackhole.cb_configs" || key == "blackhole.semaphore_plan" ||
          key == "blackhole.core_plan") {
        continue;
      }
      attrs.Set(kv.first, kv.second);
    }
  }
  return attrs;
}

Map<String, Any> AsMap(const Any& any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

bool GetBoolOrDefault(const Map<String, Any>& dict, const char* key, bool default_value = false) {
  if (auto value = dict.Get(key)) {
    return Downcast<Bool>(value.value());
  }
  return default_value;
}

Array<Any> AggregateArgs(const Array<TTABIPlan>& abi_plans, bool common) {
  Array<Any> aggregated;
  std::unordered_set<std::string> seen;
  for (const TTABIPlan& abi : abi_plans) {
    const Array<Any>& args = common ? abi->common_runtime_args : abi->runtime_args;
    for (const Any& item : args) {
      Map<String, Any> arg = AsMap(item);
      if (arg.empty()) {
        continue;
      }
      std::string identity;
      std::string kind;
      if (auto v = arg.Get("identity")) {
        identity = Downcast<String>(v.value());
      }
      if (auto v = arg.Get("kind")) {
        kind = Downcast<String>(v.value());
      }
      std::string dedupe_key = !identity.empty() && !kind.empty() ? identity + ":" + kind : identity;
      if (!dedupe_key.empty() && !seen.insert(dedupe_key).second) {
        continue;
      }
      aggregated.push_back(arg);
    }
  }
  return aggregated;
}

Array<Any> EncodeCBPlans(const Array<TTCBPlan>& cb_plans) {
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

Map<String, Any> EncodeCoreGroup(const TTCoreGroup& core_group) {
  Map<String, Any> item = core_group->payload;
  item.Set("logical_grid_x", Integer(core_group->logical_grid_x));
  item.Set("logical_grid_y", Integer(core_group->logical_grid_y));
  item.Set("linearization", core_group->linearization);
  item.Set("physical_cores", core_group->physical_cores);
  item.Set("work_packets", core_group->work_packets);
  return item;
}

Array<Any> EncodeSemaphorePlans(const Array<TTSemaphorePlan>& semaphore_plans) {
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

Array<Any> EncodeSegmentPlan(const TTProgram& program) {
  Array<Any> segments;
  for (const TTKernel& kernel : program->kernels) {
    ICHECK_GE(kernel->abi_plan_index, 0);
    const TTABIPlan& abi = program->abi_plans[static_cast<size_t>(kernel->abi_plan_index)];
    Map<String, Any> segment = kernel->payload;
    segment.Set("name", kernel->name);
    segment.Set("kind", kernel->kind);
    segment.Set("core_type", kernel->core_type);
    if (!abi->runtime_args.empty() &&
        !GetBoolOrDefault(segment, "tt_uses_top_level_runtime_args", false)) {
      segment.Set("runtime_args", abi->runtime_args);
    }
    if (!abi->common_runtime_args.empty() &&
        !GetBoolOrDefault(segment, "tt_uses_top_level_common_runtime_args", false)) {
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

}  // namespace

tvm::transform::Pass MaterializeTTExecutableSpec() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<TTProgram>(attr::kTLTTProgram);
      if (!maybe_program) {
        continue;
      }
      const TTProgram& program = maybe_program.value();
      Map<String, Any> attrs = CopyWithoutLegacyTTAttrs(func.value());
      Array<Any> segment_plan = EncodeSegmentPlan(program);
      if (!segment_plan.empty()) {
        attrs.Set("blackhole.segment_plan", segment_plan);
      }
      Array<Any> runtime_args = AggregateArgs(program->abi_plans, /*common=*/false);
      if (!runtime_args.empty()) {
        attrs.Set("blackhole.runtime_args", runtime_args);
      }
      Array<Any> common_runtime_args = AggregateArgs(program->abi_plans, /*common=*/true);
      if (!common_runtime_args.empty()) {
        attrs.Set("blackhole.common_runtime_args", common_runtime_args);
      }
      if (!program->cb_plans.empty()) {
        attrs.Set("blackhole.cb_configs", EncodeCBPlans(program->cb_plans));
      }
      if (!program->semaphore_plans.empty()) {
        attrs.Set("blackhole.semaphore_plan", EncodeSemaphorePlans(program->semaphore_plans));
      }
      if (!program->core_groups.empty()) {
        attrs.Set("blackhole.core_plan", EncodeCoreGroup(program->core_groups[0]));
      }
      tir::PrimFunc rewritten = func.value();
      rewritten.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
      updated->Add(gvar, rewritten, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.MaterializeTTExecutableSpec", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeTTExecutableSpec", MaterializeTTExecutableSpec);
}

}  // namespace tl
}  // namespace tvm
