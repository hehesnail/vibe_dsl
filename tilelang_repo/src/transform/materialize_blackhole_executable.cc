/*!
 * \file materialize_blackhole_executable.cc
 * \brief Canonical Blackhole executable writer boundary.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include "../target/tt_program_projection.h"
#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

tvm::transform::Pass MaterializeBlackholeExecutable() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      tir::PrimFunc rewritten = func.value();
      if (func.value()->GetAttr<TTProgram>(attr::kTLTTProgram)) {
        rewritten =
            tt_program_projection::MaterializeBlackholeExecutableProjectionAttr(func.value());
      } else if (tt_program_projection::GetBlackholeExecutableProjection(func.value()).size() != 0) {
        rewritten = tvm::WithoutAttr(std::move(rewritten), attr::kTLBlackholeExecutable);
      }
      if (!rewritten.same_as(func.value())) {
        updated = updated->ShallowCopy();
        updated->Add(gvar, rewritten, true);
      }
    }
    return updated;
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
