/*!
 * \file validate_tt_target_program.cc
 * \brief Validate TTProgram invariants for Phase C cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

namespace {

void ValidateTTProgram(const TTProgram& program) {
  ICHECK(!program->entry_name.empty()) << "TTProgram requires entry_name";
  ICHECK(!program->kernels.empty()) << "TTProgram requires at least one TTKernel";
  ICHECK(!program->core_groups.empty()) << "TTProgram requires at least one TTCoreGroup";
  ICHECK(!program->abi_plans.empty()) << "TTProgram requires at least one TTABIPlan";
  ICHECK(!program->execution_plans.empty()) << "TTProgram requires at least one TTExecutionPlan";

  std::unordered_set<std::string> kernel_names;
  for (const TTKernel& kernel : program->kernels) {
    ICHECK(!kernel->name.empty()) << "TTKernel requires name";
    ICHECK(!kernel->kind.empty()) << "TTKernel requires kind";
    ICHECK(!kernel->core_type.empty()) << "TTKernel requires core_type";
    ICHECK_GE(kernel->abi_plan_index, 0) << "TTKernel requires abi_plan_index";
    ICHECK_LT(kernel->abi_plan_index, static_cast<int64_t>(program->abi_plans.size()))
        << "TTKernel abi_plan_index out of bounds";
    ICHECK(kernel_names.insert(kernel->name).second) << "duplicate TTKernel name " << kernel->name;
  }

  std::unordered_set<int64_t> cb_ids;
  for (const TTCBPlan& cb : program->cb_plans) {
    ICHECK_GE(cb->cb_id, 0) << "TTCBPlan requires non-negative cb_id";
    ICHECK(cb_ids.insert(cb->cb_id).second) << "duplicate TTCBPlan cb_id " << cb->cb_id;
  }

  std::unordered_set<std::string> abi_kernel_names;
  for (const TTABIPlan& abi : program->abi_plans) {
    ICHECK(!abi->kernel_name.empty()) << "TTABIPlan requires kernel_name";
    abi_kernel_names.insert(abi->kernel_name);
  }
  for (const TTKernel& kernel : program->kernels) {
    ICHECK(abi_kernel_names.count(kernel->name))
        << "TTKernel missing matching TTABIPlan: " << kernel->name;
  }

  for (const TTExecutionPlan& execution : program->execution_plans) {
    ICHECK(!execution->kernel_names.empty()) << "TTExecutionPlan requires kernel_names";
    for (const tvm::ffi::String& kernel_name : execution->kernel_names) {
      ICHECK(kernel_names.count(kernel_name)) << "TTExecutionPlan references unknown kernel "
                                              << kernel_name;
    }
  }
}

}  // namespace

tvm::transform::Pass ValidateTTTargetProgram() {
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
      ValidateTTProgram(maybe_program.value());
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.ValidateTTTargetProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateTTTargetProgram", ValidateTTTargetProgram);
}

}  // namespace tl
}  // namespace tvm
