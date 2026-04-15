/*!
 * \file validate_tt_program.cc
 * \brief Validate TTProgram invariants for Phase C cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>
#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
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

void ValidateSyncPlan(const TTSyncPlan& sync_plan) {
  ICHECK(!sync_plan->name.empty()) << "TTSyncPlan requires name";
  ICHECK(!sync_plan->kind.empty()) << "TTSyncPlan requires kind";
  ICHECK_GE(sync_plan->source_task_index, 0) << "TTSyncPlan requires source_task_index";
  ICHECK_GE(sync_plan->target_task_index, 0) << "TTSyncPlan requires target_task_index";
  ICHECK(!sync_plan->ordering_kind.empty()) << "TTSyncPlan requires ordering_kind";
  ICHECK(!sync_plan->completion_kind.empty()) << "TTSyncPlan requires completion_kind";
}

void ValidateAccessor(const Map<String, Any>& accessor) {
  ICHECK(HasKey(accessor, "buffer")) << "TTABIPlan accessor requires buffer";
  ICHECK(HasKey(accessor, "compile_time_arg_offset"))
      << "TTABIPlan accessor requires compile_time_arg_offset";
  ICHECK(HasKey(accessor, "compile_time_arg_count"))
      << "TTABIPlan accessor requires compile_time_arg_count";
  ICHECK(HasKey(accessor, "common_runtime_arg_offset"))
      << "TTABIPlan accessor requires common_runtime_arg_offset";
  ICHECK(HasKey(accessor, "common_runtime_arg_count"))
      << "TTABIPlan accessor requires common_runtime_arg_count";
  ICHECK(HasKey(accessor, "args_config_bits")) << "TTABIPlan accessor requires args_config_bits";
  ICHECK(HasKey(accessor, "layout")) << "TTABIPlan accessor requires layout";
  ICHECK(HasKey(accessor, "memory_space")) << "TTABIPlan accessor requires memory_space";
}

void ValidateCompileTimeArgSpec(const Map<String, Any>& spec) {
  ICHECK(HasKey(spec, "kind")) << "TTABIPlan compile_time_arg_spec requires kind";
  ICHECK(HasKey(spec, "dtype")) << "TTABIPlan compile_time_arg_spec requires dtype";
  ICHECK(HasKey(spec, "offset")) << "TTABIPlan compile_time_arg_spec requires offset";
  ICHECK(HasKey(spec, "count")) << "TTABIPlan compile_time_arg_spec requires count";
}

void ValidateComputeEpilogueOps(const Map<String, Any>& compute, const char* context) {
  if (auto epilogue_any = compute.Get(String("epilogue_ops"))) {
    Array<Any> epilogue_ops = Downcast<Array<Any>>(epilogue_any.value());
    for (const Any& op_any : epilogue_ops) {
      Map<String, Any> op = AsMap(op_any);
      ICHECK(!op.empty()) << context << " epilogue_ops entries must be maps";
      ICHECK(HasKey(op, "kind")) << context << " epilogue_ops requires kind";
    }
  }
}

void ValidateProgramEpilogueOps(const Map<String, Any>& payload, const char* key, const char* context) {
  if (auto epilogue_any = payload.Get(String(key))) {
    Array<Any> epilogue_ops = Downcast<Array<Any>>(epilogue_any.value());
    for (const Any& op_any : epilogue_ops) {
      Map<String, Any> op = AsMap(op_any);
      ICHECK(!op.empty()) << context << " entries must be maps";
      ICHECK(HasKey(op, "kind")) << context << " requires kind";
    }
  }
}

void ValidateBufferTileBridgeSpecs(const Map<String, Any>& payload) {
  if (auto specs_any = payload.Get(String(schema_key::kBufferTileBridgeSpecs))) {
    Array<Any> specs = Downcast<Array<Any>>(specs_any.value());
    for (const Any& spec_any : specs) {
      Map<String, Any> spec = AsMap(spec_any);
      ICHECK(!spec.empty()) << "TTProgram payload buffer_tile_bridge_specs entries must be maps";
      ICHECK(HasKey(spec, "buffer"))
          << "TTProgram payload buffer_tile_bridge_specs requires buffer";
      ICHECK(HasKey(spec, "shape"))
          << "TTProgram payload buffer_tile_bridge_specs requires shape";
      ICHECK(HasKey(spec, "local_shape"))
          << "TTProgram payload buffer_tile_bridge_specs requires local_shape";
    }
  }
}

void ValidateUnsupportedComputeOps(const Map<String, Any>& payload) {
  if (auto ops_any = payload.Get(String("unsupported_compute_ops"))) {
    Array<Any> ops = Downcast<Array<Any>>(ops_any.value());
    for (const Any& op_any : ops) {
      auto op = op_any.as<String>();
      ICHECK(op) << "TTProgram payload unsupported_compute_ops entries must be strings";
      ICHECK(!op.value().empty())
          << "TTProgram payload unsupported_compute_ops entries must be non-empty";
    }
  }
}

void ValidateKernelPayload(const TTKernel& kernel) {
  Map<String, Any> payload = kernel->payload;
  ICHECK(HasKey(payload, "launch_spec"))
      << "TTKernel requires launch_spec in payload for reader-side cutover";
  Map<String, Any> launch_spec = AsMap(payload.Get(String("launch_spec")).value());
  ICHECK(HasKey(launch_spec, "core_type")) << "TTKernel launch_spec requires core_type";
  ICHECK(HasKey(launch_spec, "processor")) << "TTKernel launch_spec requires processor";
  ICHECK(HasKey(launch_spec, "noc")) << "TTKernel launch_spec requires noc";

  if (kernel->kind == "compute" || kernel->core_type == "trisc") {
    ICHECK(HasKey(payload, "compute_config"))
        << "TTKernel compute payload requires compute_config for compute kernels";
    Map<String, Any> compute_config = AsMap(payload.Get(String("compute_config")).value());
    ICHECK(HasKey(compute_config, "math_fidelity"))
        << "TTKernel compute_config requires math_fidelity";
    ICHECK(HasKey(compute_config, "fp32_dest_acc_en"))
        << "TTKernel compute_config requires fp32_dest_acc_en";
    ICHECK(HasKey(compute_config, "clear_accum"))
        << "TTKernel compute_config requires clear_accum";
    ICHECK(HasKey(compute_config, "k_pack")) << "TTKernel compute_config requires k_pack";
  }
}

void ValidateProgramPayload(const TTProgram& program) {
  const Map<String, Any>& payload = program->payload;
  if (auto gemm_any = payload.Get(String("gemm_contract"))) {
    Map<String, Any> gemm = AsMap(gemm_any.value());
    ICHECK(!gemm.empty()) << "TTProgram payload gemm_contract must be a map";
    ICHECK(HasKey(gemm, "a_buffer")) << "TTProgram payload gemm_contract requires a_buffer";
    ICHECK(HasKey(gemm, "b_buffer")) << "TTProgram payload gemm_contract requires b_buffer";
    ICHECK(HasKey(gemm, "c_buffer")) << "TTProgram payload gemm_contract requires c_buffer";
    ICHECK(HasKey(gemm, "M")) << "TTProgram payload gemm_contract requires M";
    ICHECK(HasKey(gemm, "N")) << "TTProgram payload gemm_contract requires N";
    ICHECK(HasKey(gemm, "K")) << "TTProgram payload gemm_contract requires K";
  }
  if (auto compute_any = payload.Get(String("compute_contract"))) {
    Map<String, Any> compute = AsMap(compute_any.value());
    ICHECK(!compute.empty()) << "TTProgram payload compute_contract must be a map";
    ICHECK(HasKey(compute, "enabled")) << "TTProgram payload compute_contract requires enabled";
    ICHECK(HasKey(compute, "kind")) << "TTProgram payload compute_contract requires kind";
    if (GetIntOrDefault(compute, "M", -1) != -1 || HasKey(compute, "a_buffer")) {
      ICHECK(HasKey(compute, "a_buffer")) << "TTProgram payload compute_contract requires a_buffer";
      ICHECK(HasKey(compute, "b_buffer")) << "TTProgram payload compute_contract requires b_buffer";
      ICHECK(HasKey(compute, "c_buffer")) << "TTProgram payload compute_contract requires c_buffer";
      ICHECK(HasKey(compute, "M")) << "TTProgram payload compute_contract requires M";
      ICHECK(HasKey(compute, "N")) << "TTProgram payload compute_contract requires N";
      ICHECK(HasKey(compute, "K")) << "TTProgram payload compute_contract requires K";
      ICHECK(HasKey(compute, "math_fidelity"))
          << "TTProgram payload compute_contract requires math_fidelity";
      ICHECK(HasKey(compute, "fp32_dest_acc_en"))
          << "TTProgram payload compute_contract requires fp32_dest_acc_en";
      ICHECK(HasKey(compute, "clear_accum"))
          << "TTProgram payload compute_contract requires clear_accum";
      ICHECK(HasKey(compute, "k_pack")) << "TTProgram payload compute_contract requires k_pack";
      ValidateComputeEpilogueOps(compute, "TTProgram payload compute_contract");
    }
  }
  if (auto multi_gemm_any = payload.Get(String("multi_gemm_contracts"))) {
    Array<Any> contracts = Downcast<Array<Any>>(multi_gemm_any.value());
    ICHECK(!contracts.empty()) << "TTProgram payload multi_gemm_contracts must be non-empty";
    for (const Any& contract_any : contracts) {
      Map<String, Any> gemm = AsMap(contract_any);
      ICHECK(!gemm.empty()) << "TTProgram payload multi_gemm_contracts entries must be maps";
      ICHECK(HasKey(gemm, "a_buffer"))
          << "TTProgram payload multi_gemm_contracts requires a_buffer";
      ICHECK(HasKey(gemm, "b_buffer"))
          << "TTProgram payload multi_gemm_contracts requires b_buffer";
      ICHECK(HasKey(gemm, "c_buffer"))
          << "TTProgram payload multi_gemm_contracts requires c_buffer";
      ICHECK(HasKey(gemm, "M")) << "TTProgram payload multi_gemm_contracts requires M";
      ICHECK(HasKey(gemm, "N")) << "TTProgram payload multi_gemm_contracts requires N";
      ICHECK(HasKey(gemm, "K")) << "TTProgram payload multi_gemm_contracts requires K";
    }
  }
  if (auto multi_compute_any = payload.Get(String("multi_compute_contracts"))) {
    Array<Any> contracts = Downcast<Array<Any>>(multi_compute_any.value());
    ICHECK(!contracts.empty()) << "TTProgram payload multi_compute_contracts must be non-empty";
    for (const Any& contract_any : contracts) {
      Map<String, Any> compute = AsMap(contract_any);
      ICHECK(!compute.empty())
          << "TTProgram payload multi_compute_contracts entries must be maps";
      ICHECK(HasKey(compute, "enabled"))
          << "TTProgram payload multi_compute_contracts requires enabled";
      ICHECK(HasKey(compute, "kind"))
          << "TTProgram payload multi_compute_contracts requires kind";
      ICHECK(HasKey(compute, "a_buffer"))
          << "TTProgram payload multi_compute_contracts requires a_buffer";
      ICHECK(HasKey(compute, "b_buffer"))
          << "TTProgram payload multi_compute_contracts requires b_buffer";
      ICHECK(HasKey(compute, "c_buffer"))
          << "TTProgram payload multi_compute_contracts requires c_buffer";
      ICHECK(HasKey(compute, "M")) << "TTProgram payload multi_compute_contracts requires M";
      ICHECK(HasKey(compute, "N")) << "TTProgram payload multi_compute_contracts requires N";
      ICHECK(HasKey(compute, "K")) << "TTProgram payload multi_compute_contracts requires K";
      ICHECK(HasKey(compute, "math_fidelity"))
          << "TTProgram payload multi_compute_contracts requires math_fidelity";
      ICHECK(HasKey(compute, "fp32_dest_acc_en"))
          << "TTProgram payload multi_compute_contracts requires fp32_dest_acc_en";
      ICHECK(HasKey(compute, "clear_accum"))
          << "TTProgram payload multi_compute_contracts requires clear_accum";
      ICHECK(HasKey(compute, "k_pack"))
          << "TTProgram payload multi_compute_contracts requires k_pack";
      ValidateComputeEpilogueOps(compute, "TTProgram payload multi_compute_contracts");
    }
  }
  ValidateProgramEpilogueOps(payload, "compute_epilogue_ops", "TTProgram payload compute_epilogue_ops");
  ValidateBufferTileBridgeSpecs(payload);
  ValidateUnsupportedComputeOps(payload);
}

void CheckTTProgram(const TTProgram& program) {
  ICHECK(!program->entry_name.empty()) << "TTProgram requires entry_name";
  ICHECK(!program->block_plans.empty()) << "TTProgram requires at least one TTBlockPlan";
  ICHECK(!program->kernel_plans.empty()) << "TTProgram requires at least one TTKernelPlan";
  ICHECK(!program->kernels.empty()) << "TTProgram requires at least one TTKernel";
  ICHECK(!program->core_groups.empty()) << "TTProgram requires at least one TTCoreGroup";
  ICHECK(!program->abi_plans.empty()) << "TTProgram requires at least one TTABIPlan";
  ICHECK(!program->execution_plans.empty()) << "TTProgram requires at least one TTExecutionPlan";
  ICHECK_EQ(program->block_plans.size(), program->core_groups.size())
      << "TTProgram requires aligned TTBlockPlan and TTCoreGroup compatibility payloads";
  ICHECK_EQ(program->kernel_plans.size(), program->kernels.size())
      << "TTProgram requires aligned TTKernelPlan and TTKernel compatibility payloads";
  ICHECK_EQ(program->sync_plans.size(), program->compute_sync_plans.size())
      << "TTProgram requires aligned TTSyncPlan and TTComputeSyncPlan compatibility payloads";

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
    ValidateKernelPayload(kernel);
  }
  for (const TTKernelPlan& kernel_plan : program->kernel_plans) {
    ValidateKernelPlan(kernel_plan, static_cast<int64_t>(program->abi_plans.size()),
                       static_cast<int64_t>(program->block_plans.size()));
    ICHECK(kernel_names.count(kernel_plan->name))
        << "TTKernelPlan missing matching TTKernel compatibility payload: "
        << kernel_plan->name;
  }

  ValidateProgramPayload(program);

  for (const TTCoreGroup& core_group : program->core_groups) {
    ValidateCoreGroup(core_group);
  }

  std::unordered_set<int64_t> cb_ids;
  for (const TTCBPlan& cb : program->cb_plans) {
    ICHECK_GE(cb->cb_id, 0) << "TTCBPlan requires non-negative cb_id";
    ICHECK(cb_ids.insert(cb->cb_id).second) << "duplicate TTCBPlan cb_id " << cb->cb_id;
  }

  std::unordered_set<std::string> abi_kernel_names;
  for (const TTABIPlan& abi : program->abi_plans) {
    ICHECK(!abi->kernel_name.empty()) << "TTABIPlan requires kernel_name";
    for (const Any& accessor_any : abi->accessors) {
      ValidateAccessor(AsMap(accessor_any));
    }
    for (const Any& spec_any : abi->compile_time_arg_specs) {
      ValidateCompileTimeArgSpec(AsMap(spec_any));
    }
    abi_kernel_names.insert(abi->kernel_name);
  }
  for (const TTKernel& kernel : program->kernels) {
    ICHECK(abi_kernel_names.count(kernel->name))
        << "TTKernel missing matching TTABIPlan: " << kernel->name;
  }

  for (const TTTransportPlan& transport : program->transport_plans) {
    ICHECK(!transport->kind.empty()) << "TTTransportPlan requires kind";
    ICHECK(!transport->payload_kind.empty()) << "TTTransportPlan requires payload_kind";
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
      CheckTTProgram(maybe_program.value());
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
