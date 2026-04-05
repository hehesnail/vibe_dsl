/*
 * \file typed_rebind_blackhole_companion_programs.cc
 * \brief Apply a typed rebind contract to Blackhole semantic companion attrs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/structural_hash.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_rebind.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;

tir::transform::Pass TypedRebindBlackholeCompanionPrograms(Map<String, Any> plan) {
  auto fpass = [plan](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto maybe_program = func->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
    auto maybe_witnesses = func->GetAttr<ffi::Array<SemanticWitness>>(attr::kTLSemanticWitnesses);
    auto maybe_structure = func->GetAttr<Map<String, Any>>(attr::kTLSemanticStructure);
    auto maybe_freeze = func->GetAttr<Map<String, Any>>(attr::kTLSemanticHardFreeze);
    ICHECK(maybe_program && maybe_witnesses && maybe_structure && maybe_freeze)
        << "TypedRebindBlackholeCompanionPrograms requires live semantic companion attrs";

    SemanticRebindPlan rebind_plan = DecodeSemanticRebindPlan(plan);
    auto rebound_witnesses = ApplySemanticRebindToWitnesses(maybe_witnesses.value(), rebind_plan);
    auto rebound_program =
        ApplySemanticRebindToProgram(maybe_program.value(), rebound_witnesses, rebind_plan);
    auto rebound_structure = ApplySemanticRebindToStructure(maybe_structure.value(), rebind_plan);

    Map<String, Any> freeze = maybe_freeze.value();
    const std::string previous_hash =
        freeze.find("body_hash") != freeze.end()
            ? static_cast<std::string>(tvm::Downcast<String>(freeze["body_hash"]))
            : std::to_string(tvm::StructuralHash()(func->body));
    int64_t next_epoch = 1;
    if (auto it = freeze.find("rebind_epoch"); it != freeze.end()) {
      next_epoch = tvm::Downcast<Integer>((*it).second)->value + 1;
    }
    freeze.Set("state", String("rebound"));
    freeze.Set("contract_mode", String(ToString(ContractMode::kTypedRebind)));
    freeze.Set("previous_body_hash", String(previous_hash));
    freeze.Set("body_hash", String(std::to_string(tvm::StructuralHash()(func->body))));
    freeze.Set("rebind_epoch", Integer(next_epoch));
    freeze.Set("rebind_scope", String(ToString(rebind_plan.scope)));
    freeze.Set("rebind_trace", rebind_plan.trace);
    freeze.Set("rebind_reason", String(rebind_plan.reason));
    freeze.Set("unsafe_mutation_policy", String("require_rebind_or_invalidate"));

    tir::PrimFunc updated = tvm::WithoutAttr(func, attr::kTLCompanionInvalidationReason);
    Map<String, Any> attrs = updated->attrs.defined() ? updated->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticStructure, rebound_structure);
    attrs.Set(attr::kTLSemanticWitnesses, rebound_witnesses);
    attrs.Set(attr::kTLSemanticProgram, rebound_program);
    attrs.Set(attr::kTLSemanticHardFreeze, freeze);
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.TypedRebindBlackholeCompanionPrograms", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TypedRebindBlackholeCompanionPrograms",
                        TypedRebindBlackholeCompanionPrograms);
}

}  // namespace tl
}  // namespace tvm
