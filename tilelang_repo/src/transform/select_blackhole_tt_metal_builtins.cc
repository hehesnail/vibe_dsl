/*!
 * \file select_blackhole_tt_metal_builtins.cc
 * \brief Dedicated exact TT-Metal builtin selection pass.
 */

#include "lower_blackhole_ops.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>

#include "common/blackhole_utils.h"

namespace tvm {
namespace tl {

namespace {

const std::unordered_set<std::string>& HelperCompositeBlackholeBuiltinNames() {
  static const auto* names = new std::unordered_set<std::string>{
      "tl.blackhole.copy_tile_from_cb",
      "tl.blackhole.write_local_slice_to_cb",
      "tl.blackhole.write_local_fragment_tile_to_cb",
      "tl.blackhole.write_local_fragment_slice_to_tiled_cb",
      "tl.blackhole.cast_fragment_slice_to_tiled_cb",
      "tl.blackhole.read_cb_front_tile_to_local",
      "tl.blackhole.read_cb_front_tile_to_local_fragment",
      "tl.blackhole.reduce_row",
      "tl.blackhole.mul_row_bcast",
      "tl.blackhole.mul_grouped_row_bcast",
      "tl.blackhole.div_row_bcast",
      "tl.blackhole.div_grouped_row_bcast",
      "tl.blackhole.exp2_row_bcast_affine",
      "tl.blackhole.exp2_grouped_row_bcast_affine",
      "tl.blackhole.scalar_max",
      "tl.blackhole.scalar_exp2_affine",
      "tl.blackhole.binary_max_tile_local",
      "tl.blackhole.reduce_rows_local",
      "tl.blackhole.mul_tiles_bcast_rows_local",
      "tl.blackhole.div_tiles_bcast_rows_local",
      "tl.blackhole.exp_tiles_bcast_rows_affine_local",
      "tl.blackhole.exp_tile_affine_local",
      "tl.blackhole.scalar_fma",
  };
  return *names;
}

TTProgram WithStagedCBPlans(const TTProgram& program, ffi::Array<TTCBPlan> cb_plans) {
  return TTProgram(program->entry_name, program->member_func, program->mesh_plans,
                   program->buffer_distribution_plans, program->block_plans,
                   program->kernel_plans, program->compute_op_plans,
                   program->transport_plans, program->sync_plans,
                   program->abi_plans, program->execution_plans, program->kernels,
                   program->core_groups, std::move(cb_plans), program->semaphore_plans,
                   program->compute_sync_plans, program->dst_layout_plans,
                   program->live_form_plans, program->materialization_plans,
                   program->consumer_binding_plans, program->payload);
}

}  // namespace

bool IsHelperCompositeBlackholeBuiltin(const tvm::Op& op) {
  return HelperCompositeBlackholeBuiltinNames().count(op->name) != 0U;
}

bool UsesHelperCompositeBlackholeBuiltin(const tir::PrimFunc& func) {
  bool found = false;
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (found) {
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call) {
      return;
    }
    if (const auto* ir_op = call->op.as<tvm::OpNode>()) {
      found = HelperCompositeBlackholeBuiltinNames().count(ir_op->name) != 0U;
    }
  });
  return found;
}

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
      ICHECK(!UsesHelperCompositeBlackholeBuiltin(selected))
          << "SelectBlackholeTTMetalBuiltins emitted helper/composite builtin residue";
      auto staged_program = selected->GetAttr<TTProgram>(attr::kTLTTProgram);
      ICHECK(staged_program)
          << "SelectBlackholeTTMetalBuiltins requires staged tl.tt_program from PlanTTBlocks";
      selected =
          WithAttr(std::move(selected), attr::kTLTTProgram,
                   WithStagedCBPlans(staged_program.value(), selector.GetStagedCBPlans()));
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
