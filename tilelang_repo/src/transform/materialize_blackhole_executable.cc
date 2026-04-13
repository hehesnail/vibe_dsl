/*!
 * \file materialize_blackhole_executable.cc
 * \brief Canonical Blackhole executable writer boundary.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

tvm::transform::Pass MaterializeBlackholeExecutable() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      if (!func.value()->GetAttr<TTProgram>(attr::kTLTTProgram)) {
        continue;
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.MaterializeBlackholeExecutable", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeBlackholeExecutable",
                        MaterializeBlackholeExecutable);
}

}  // namespace tl
}  // namespace tvm
