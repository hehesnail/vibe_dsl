/*
 * \file validate_stateful_semantic_ir.cc
 * \brief Minimal structural validation for Stage 4 A1 SemanticProgram.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

namespace {

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

bool IsAllowedLawKind(const ffi::String& kind) {
  static const std::unordered_set<std::string> kKinds = {"map", "reduce", "select", "recurrence"};
  return kKinds.count(kind);
}

bool IsAllowedStateRole(const ffi::String& role) {
  static const std::unordered_set<std::string> kRoles = {
      "carry", "reduction_accumulator", "selection_state", "index_state", "transient"};
  return kRoles.count(role);
}

}  // namespace

tir::transform::Pass ValidateStatefulSemanticIR() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto maybe_program = func->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
    ICHECK(maybe_program) << "ValidateStatefulSemanticIR requires tl.semantic_program";
    const SemanticProgram& program = maybe_program.value();
    ICHECK(!program->domains.empty()) << "SemanticProgram must contain at least one Domain";
    for (const State& state : program->states) {
      ICHECK(state.defined());
      ICHECK(IsAllowedStateRole(state->role))
          << "Unsupported State.role in A2 validator: " << state->role;
    }
    for (const Update& update : program->updates) {
      ICHECK(update.defined());
      ICHECK(update->law.defined()) << "SemanticProgram update must carry UpdateLaw";
      ICHECK(IsAllowedLawKind(update->law->kind))
          << "Unsupported UpdateLaw.kind in A1 validator: " << update->law->kind;
    }
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.ValidateStatefulSemanticIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateStatefulSemanticIR", ValidateStatefulSemanticIR);
}

}  // namespace tl
}  // namespace tvm
