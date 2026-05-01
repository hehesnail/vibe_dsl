/*!
 * \file select_blackhole_tt_metal_builtins.cc
 * \brief Dedicated exact TT-Metal builtin selection pass.
 */

#include "lower_blackhole_ops.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <utility>

#include "common/blackhole_utils.h"

namespace tvm {
namespace tl {

namespace {

TTProgram WithStagedCBAndComputeOpPlans(const TTProgram& program,
                                        ffi::Array<TTCBPlan> cb_plans,
                                        ffi::Array<TTComputeOpPlan> compute_op_plans) {
  return TTProgram(program->entry_name, program->member_func, program->mesh_plans,
                   program->buffer_distribution_plans,
                   program->tensor_memory_config_plans,
                   program->op_sharding_contracts,
                   program->placement_resolution_plans, program->reshard_plans,
                   program->block_plans,
                   program->kernel_plans, std::move(compute_op_plans),
                   program->transport_plans, program->sync_plans,
                   program->abi_plans, program->execution_plans, program->kernels,
                   program->core_groups, std::move(cb_plans), program->semaphore_plans,
                   program->compute_sync_plans, program->dst_layout_plans,
                   program->live_form_plans, program->materialization_plans,
                   program->consumer_binding_plans, program->resource_demands,
                   program->resource_pressure_reports);
}

}  // namespace

tvm::transform::Pass SelectBlackholeTTMetalBuiltins() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      PlanTTKernelABI selector;
      tir::PrimFunc selected = selector.SelectComputeBuiltins(func.value());
      auto staged_program = selected->GetAttr<TTProgram>(attr::kTLTTProgram);
      ICHECK(staged_program)
          << "SelectBlackholeTTMetalBuiltins requires staged tl.tt_program from PlanTTBlocks";
      selected =
          WithAttr(std::move(selected), attr::kTLTTProgram,
                   WithStagedCBAndComputeOpPlans(staged_program.value(),
                                                 selector.GetStagedCBPlans(),
                                                 selector.GetTTComputeOpPlans()));
      selected = WithAttr(std::move(selected), kTLBlackholeTTMetalBuiltinSelection, Bool(true));
      updated->Add(gvar, selected, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.SelectBlackholeTTMetalBuiltins", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SelectBlackholeTTMetalBuiltins",
                        SelectBlackholeTTMetalBuiltins);
}

}  // namespace tl
}  // namespace tvm
