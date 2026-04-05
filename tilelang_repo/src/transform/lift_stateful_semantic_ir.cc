/*
 * \file lift_stateful_semantic_ir.cc
 * \brief Lift the minimal Stage 4 semantic summary into typed SemanticProgram objects.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

Array<String> DowncastStringArray(const Array<Any>& items) {
  Array<String> result;
  for (const Any& item : items) {
    result.push_back(tvm::Downcast<String>(item));
  }
  return result;
}

}  // namespace

tir::transform::Pass LiftStatefulSemanticIR() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto maybe_structure = func->GetAttr<Map<String, Any>>(attr::kTLSemanticStructure);
    if (!maybe_structure) {
      return func;
    }
    Map<String, Any> structure = maybe_structure.value();

    Array<TIRAnchor> anchors{TIRAnchor("source_attr", attr::kTLSemanticStructure)};
    Array<Domain> domains;
    domains.push_back(
        Domain(tvm::Downcast<String>(structure["domain_name"]),
               DowncastStringArray(tvm::Downcast<Array<Any>>(structure["domain_axes"])),
               DowncastStringArray(tvm::Downcast<Array<Any>>(structure["domain_traits"])),
               anchors));

    Array<State> states;
    for (const Any& state_any : tvm::Downcast<Array<Any>>(structure["states"])) {
      auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
      Array<TIRAnchor> state_anchors{
          TIRAnchor("state_source", state_map["role"].cast<String>())};
      states.push_back(State(tvm::Downcast<String>(state_map["name"]),
                             tvm::Downcast<String>(state_map["role"]),
                             tvm::Downcast<String>(state_map["scope"]), state_anchors));
    }

    Array<Update> updates;
    for (const Any& update_any : tvm::Downcast<Array<Any>>(structure["updates"])) {
      auto update_map = tvm::Downcast<Map<String, Any>>(update_any);
      Array<String> source_states;
      if (update_map.count("target_state") && !tvm::Downcast<String>(update_map["target_state"]).empty()) {
        source_states.push_back(tvm::Downcast<String>(update_map["target_state"]));
      }
      Array<String> access_traits;
      if (update_map.count("traits")) {
        access_traits = DowncastStringArray(tvm::Downcast<Array<Any>>(update_map["traits"]));
      }
      Array<AccessMap> access_maps{
          AccessMap("tir_region", {}, access_traits)};
      UpdateLaw law(tvm::Downcast<String>(update_map["kind"]),
                    tvm::Downcast<String>(update_map["target_state"]), source_states,
                    access_maps);
      Array<TIRAnchor> update_anchors{
          TIRAnchor("update_kind", tvm::Downcast<String>(update_map["kind"]))};
      Array<TIRValueBinding> bindings{
          TIRValueBinding("target_state", "state", tvm::Downcast<String>(update_map["target_state"]))};
      updates.push_back(Update(tvm::Downcast<String>(update_map["name"]),
                               tvm::Downcast<String>(update_map["target_state"]), law,
                               update_anchors, bindings));
    }

    Array<SemanticSupplement> supplements;
    if (structure.count("supplements")) {
      for (const Any& supplement_any : tvm::Downcast<Array<Any>>(structure["supplements"])) {
        auto supplement_map = tvm::Downcast<Map<String, Any>>(supplement_any);
        supplements.push_back(
            SemanticSupplement(tvm::Downcast<String>(supplement_map["kind"]),
                               tvm::Downcast<Map<String, Any>>(supplement_map["payload"])));
      }
    }

    Array<String> seeds = DowncastStringArray(tvm::Downcast<Array<Any>>(structure["seeds"]));
    SemanticProgram semantic_program(domains, states, updates, supplements, seeds, anchors);

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticProgram, semantic_program);
    attrs.Set(attr::kTLSemanticHardFreeze,
              Map<String, Any>{{"state", String("lifted_a1")},
                               {"unsafe_mutation_policy",
                                String("invalidate_companion_programs")}});
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.LiftStatefulSemanticIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LiftStatefulSemanticIR", LiftStatefulSemanticIR);
}

}  // namespace tl
}  // namespace tvm
