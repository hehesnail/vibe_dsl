/*
 * \file validate_semantic_refinement.cc
 * \brief Validate that SemanticProgram is a legal abstraction of semantic witnesses.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/structural_hash.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

std::unordered_set<std::string> ToStringSet(const Array<String>& values) {
  std::unordered_set<std::string> result;
  for (const String& value : values) {
    result.insert(value);
  }
  return result;
}

std::unordered_set<std::string> ToStringSet(const Array<Any>& values) {
  std::unordered_set<std::string> result;
  for (const Any& value : values) {
    result.insert(tvm::Downcast<String>(value));
  }
  return result;
}

bool HasBinding(const Update& update, const std::string& kind, const std::string& value_repr) {
  for (const TIRValueBinding& binding : update->bindings) {
    if (std::string(binding->kind) == kind && std::string(binding->value_repr) == value_repr) {
      return true;
    }
  }
  return false;
}

bool IsAllowedSupplementKind(const std::string& kind) {
  static const std::unordered_set<std::string> kAllowed = {
      "state_identity", "access_trait", "update_law_trait", "semantic_boundary"};
  return kAllowed.count(kind);
}

bool IsAllowedSubjectKind(const std::string& kind) {
  static const std::unordered_set<std::string> kAllowed = {
      "domain", "state", "update", "access", "relation", "boundary"};
  return kAllowed.count(kind);
}

bool IsAllowedFactAxis(const std::string& axis) {
  static const std::unordered_set<std::string> kAllowed = {
      "role",          "identity",         "lifetime",      "law_family",
      "source_set",    "ordering",         "boundary",      "indirection",
      "selection_contract",                "distribution_hint",
      "companion",     "derives_index_from", "feeds_update", "carried_from",
      "semantic_boundary",                 "ordered_region"};
  return kAllowed.count(axis);
}

bool IsAllowedContractMode(const std::string& mode) {
  static const std::unordered_set<std::string> kAllowed = {
      "preserve", "typed_rebind", "invalidate"};
  return kAllowed.count(mode);
}

}  // namespace

tir::transform::Pass ValidateSemanticRefinement() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto maybe_program = func->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
    if (!maybe_program) {
      return func;
    }
    auto maybe_witnesses = func->GetAttr<Array<SemanticWitness>>(attr::kTLSemanticWitnesses);
    ICHECK(maybe_witnesses) << "ValidateSemanticRefinement requires tl.semantic_witnesses";
    auto maybe_freeze = func->GetAttr<Map<String, Any>>(attr::kTLSemanticHardFreeze);
    ICHECK(maybe_freeze) << "ValidateSemanticRefinement requires tl.semantic_hard_freeze";
    ICHECK(!func->GetAttr<String>(attr::kTLCompanionInvalidationReason))
        << "ValidateSemanticRefinement requires live companion program; found invalidation reason";

    const SemanticProgram& program = maybe_program.value();
    const Array<SemanticWitness>& witnesses = maybe_witnesses.value();

    const Map<String, Any>& freeze = maybe_freeze.value();
    auto contract_it = freeze.find("contract_mode");
    ICHECK(contract_it != freeze.end())
        << "ValidateSemanticRefinement requires tl.semantic_hard_freeze.contract_mode";
    const std::string contract_mode = tvm::Downcast<String>((*contract_it).second);
    ICHECK(IsAllowedContractMode(contract_mode))
        << "Unsupported semantic companion contract mode: " << contract_mode;
    ICHECK_NE(contract_mode, "invalidate")
        << "Invalidated companion program cannot retain tl.semantic_program";
    if (contract_mode == "typed_rebind") {
      ICHECK(freeze.find("rebind_epoch") != freeze.end())
          << "typed_rebind contract requires rebind_epoch";
    }
    if (auto it = freeze.find("body_hash"); it != freeze.end()) {
      const std::string expected_hash = tvm::Downcast<String>((*it).second);
      const std::string current_hash = std::to_string(tvm::StructuralHash()(func->body));
      ICHECK_EQ(expected_hash, current_hash)
          << "Semantic companion contract violation: PrimFunc body changed after lift without "
             "invalidate/rebind";
    } else {
      ICHECK_NE(contract_mode, "preserve")
          << "preserve contract requires tl.semantic_hard_freeze.body_hash";
    }

    std::unordered_map<std::string, State> states_by_name;
    std::unordered_set<std::string> state_names;
    for (const State& state : program->states) {
      states_by_name.emplace(state->name, state);
      state_names.insert(state->name);
    }

    std::unordered_map<std::string, Update> updates_by_name;
    std::unordered_set<std::string> update_names;
    for (const Update& update : program->updates) {
      updates_by_name.emplace(update->name, update);
      update_names.insert(update->name);
    }

    for (const SemanticSupplement& supplement : program->supplements) {
      ICHECK(IsAllowedSupplementKind(supplement->kind))
          << "Unsupported SemanticSupplement kind in refinement validator: "
          << supplement->kind;
    }

    for (const SemanticWitness& witness : witnesses) {
      const std::string subject_kind = witness->subject_kind;
      const std::string subject_anchor = witness->subject_anchor_id;
      const std::string fact_axis = witness->fact_axis;
      ICHECK(IsAllowedSubjectKind(subject_kind))
          << "Unsupported witness subject kind: " << witness->subject_kind;
      ICHECK(IsAllowedFactAxis(fact_axis))
          << "Unsupported witness fact axis: " << witness->fact_axis;
      ICHECK(!std::string(witness->canonicalization_point).empty())
          << "Semantic witness must carry canonicalization_point";
      ICHECK(!witness->evidence_sources.empty())
          << "Semantic witness must carry at least one evidence source";

      if (subject_kind == "state") {
        ICHECK(states_by_name.count(subject_anchor))
            << "State witness references missing state anchor: " << subject_anchor;
      } else if (subject_kind == "update") {
        ICHECK(updates_by_name.count(subject_anchor))
            << "Update witness references missing update anchor: " << subject_anchor;
      } else if (subject_kind == "relation") {
        if (fact_axis == "derives_index_from") {
          ICHECK(states_by_name.count(subject_anchor))
              << "Relation witness references missing state anchor: " << subject_anchor;
        } else {
          ICHECK(updates_by_name.count(subject_anchor))
              << "Relation witness references missing update anchor: " << subject_anchor;
        }
      }

      for (const String& related_anchor : witness->related_anchor_ids) {
        const std::string related = related_anchor;
        ICHECK(state_names.count(related) || update_names.count(related))
            << "Semantic witness references missing related anchor: " << related;
      }

      if (subject_kind == "state" && fact_axis == "role") {
        auto it = witness->fact_value.find("role");
        ICHECK(it != witness->fact_value.end()) << "state.role witness requires role payload";
        ICHECK_EQ(states_by_name.at(subject_anchor)->role, tvm::Downcast<String>((*it).second))
            << "state.role witness does not match SemanticProgram";
      } else if (subject_kind == "update" && fact_axis == "law_family") {
        auto it = witness->fact_value.find("kind");
        ICHECK(it != witness->fact_value.end()) << "update.law_family witness requires kind payload";
        ICHECK_EQ(updates_by_name.at(subject_anchor)->law->kind,
                  tvm::Downcast<String>((*it).second))
            << "update.law_family witness does not match SemanticProgram";
      } else if (subject_kind == "update" && fact_axis == "source_set") {
        auto it = witness->fact_value.find("sources");
        ICHECK(it != witness->fact_value.end()) << "update.source_set witness requires sources payload";
        auto actual = ToStringSet(updates_by_name.at(subject_anchor)->law->source_states);
        auto expected = ToStringSet(tvm::Downcast<Array<Any>>((*it).second));
        ICHECK(actual == expected) << "update.source_set witness does not match SemanticProgram";
      } else if (subject_kind == "relation" && fact_axis == "companion") {
        const Update& update = updates_by_name.at(subject_anchor);
        ICHECK_EQ(std::string(update->law->kind), "select")
            << "relation.companion requires a select update";
        auto it = witness->fact_value.find("binding_kind");
        std::string binding_kind =
            it != witness->fact_value.end() ? std::string(tvm::Downcast<String>((*it).second))
                                            : "paired_value_state";
        for (const String& related_anchor : witness->related_anchor_ids) {
          ICHECK(HasBinding(update, binding_kind, related_anchor))
              << "relation.companion witness missing update binding for " << related_anchor;
        }
      } else if (subject_kind == "relation" && fact_axis == "carried_from") {
        const Update& update = updates_by_name.at(subject_anchor);
        ICHECK_EQ(std::string(update->law->kind), "recurrence")
            << "relation.carried_from requires a recurrence update";
        auto it = witness->fact_value.find("binding_kind");
        std::string binding_kind =
            it != witness->fact_value.end() ? std::string(tvm::Downcast<String>((*it).second))
                                            : "recurrence_source_state";
        for (const String& related_anchor : witness->related_anchor_ids) {
          ICHECK(HasBinding(update, binding_kind, related_anchor))
              << "relation.carried_from witness missing update binding for " << related_anchor;
        }
      } else if (subject_kind == "relation" && fact_axis == "derives_index_from") {
        ICHECK_EQ(std::string(states_by_name.at(subject_anchor)->role), "index_state")
            << "relation.derives_index_from requires index_state";
      }
    }

    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.ValidateSemanticRefinement", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateSemanticRefinement",
                        ValidateSemanticRefinement);
}

}  // namespace tl
}  // namespace tvm
