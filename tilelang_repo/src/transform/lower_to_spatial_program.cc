/*!
 * \file lower_to_spatial_program.cc
 * \brief Thin compatibility wrapper for the split Phase B spatial lowering pipeline.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

tvm::transform::Pass AnalyzeSpatialDomainPlan();
tvm::transform::Pass AnalyzeSpatialExecutionPlan();
tvm::transform::Pass MaterializeSpatialProgram();

tvm::transform::Pass LowerToSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    mod = AnalyzeSpatialDomainPlan()(mod);
    mod = AnalyzeSpatialExecutionPlan()(mod);
    mod = MaterializeSpatialProgram()(mod);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.LowerToSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerToSpatialProgram", LowerToSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
