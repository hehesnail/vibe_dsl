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
#include "common/semantic_refinement_rules.h"
#include "common/semantic_vocab.h"
#include "common/semantic_witness_decoder.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;

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
    auto contract_mode = DecodeContractMode(freeze);
    ICHECK(contract_mode)
        << "ValidateSemanticRefinement requires tl.semantic_hard_freeze.contract_mode";
    ICHECK(*contract_mode != ContractMode::kInvalidate)
        << "Invalidated companion program cannot retain tl.semantic_program";
    if (ContractModeRequiresRebindEpoch(*contract_mode)) {
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
      ICHECK(!ContractModeRequiresBodyHash(*contract_mode))
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
      ICHECK(ParseSupplementKind(static_cast<std::string>(supplement->kind)))
          << "Unsupported SemanticSupplement kind in refinement validator: "
          << supplement->kind;
    }

    for (const SemanticWitness& witness : witnesses) {
      auto decoded = DecodeSemanticWitness(witness);
      ICHECK(decoded) << "Unsupported semantic witness vocabulary: "
                      << witness->subject_kind << "/" << witness->fact_axis;
      ICHECK(!std::string(witness->canonicalization_point).empty())
          << "Semantic witness must carry canonicalization_point";
      ICHECK(!witness->evidence_sources.empty())
          << "Semantic witness must carry at least one evidence source";

      if (decoded->subject_kind == WitnessSubjectKind::kState) {
        ICHECK(states_by_name.count(decoded->subject_anchor_id))
            << "State witness references missing state anchor: " << decoded->subject_anchor_id;
      } else if (decoded->subject_kind == WitnessSubjectKind::kUpdate) {
        ICHECK(updates_by_name.count(decoded->subject_anchor_id))
            << "Update witness references missing update anchor: " << decoded->subject_anchor_id;
      } else if (decoded->subject_kind == WitnessSubjectKind::kRelation) {
        if (RelationAxisRequiresStateAnchor(decoded->fact_axis)) {
          ICHECK(states_by_name.count(decoded->subject_anchor_id))
              << "Relation witness references missing state anchor: "
              << decoded->subject_anchor_id;
        }
        if (RelationAxisRequiresUpdateAnchor(decoded->fact_axis)) {
          ICHECK(updates_by_name.count(decoded->subject_anchor_id))
              << "Relation witness references missing update anchor: "
              << decoded->subject_anchor_id;
        }
      }

      for (const String& related_anchor : witness->related_anchor_ids) {
        const std::string related = related_anchor;
        ICHECK(state_names.count(related) || update_names.count(related))
            << "Semantic witness references missing related anchor: " << related;
      }

      if (decoded->subject_kind == WitnessSubjectKind::kState &&
          decoded->fact_axis == WitnessFactAxis::kRole) {
        auto role = DecodeWitnessStateRole(witness);
        ICHECK(role) << "state.role witness requires supported role payload";
        ICHECK_EQ(states_by_name.at(decoded->subject_anchor_id)->role, String(ToString(*role)))
            << "state.role witness does not match SemanticProgram";
      } else if (decoded->subject_kind == WitnessSubjectKind::kUpdate &&
                 decoded->fact_axis == WitnessFactAxis::kLawFamily) {
        auto law_kind = DecodeWitnessUpdateLawKind(witness);
        ICHECK(law_kind) << "update.law_family witness requires supported kind payload";
        ICHECK_EQ(updates_by_name.at(decoded->subject_anchor_id)->law->kind,
                  String(ToString(*law_kind)))
            << "update.law_family witness does not match SemanticProgram";
      } else if (decoded->subject_kind == WitnessSubjectKind::kUpdate &&
                 decoded->fact_axis == WitnessFactAxis::kSourceSet) {
        auto it = witness->fact_value.find("sources");
        ICHECK(it != witness->fact_value.end()) << "update.source_set witness requires sources payload";
        auto actual = ToStringSet(updates_by_name.at(decoded->subject_anchor_id)->law->source_states);
        auto expected = ToStringSet(tvm::Downcast<Array<Any>>((*it).second));
        ICHECK(actual == expected) << "update.source_set witness does not match SemanticProgram";
      } else if (decoded->subject_kind == WitnessSubjectKind::kRelation &&
                 decoded->fact_axis == WitnessFactAxis::kCompanion) {
        const Update& update = updates_by_name.at(decoded->subject_anchor_id);
        auto law_kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
        ICHECK(law_kind && RelationAxisCompatibleWithLawKind(decoded->fact_axis, *law_kind))
            << "relation.companion requires a select update";
        BindingKind binding_kind =
            DecodeWitnessBindingKind(witness, "binding_kind")
                .value_or(DefaultBindingKindForRelation(decoded->fact_axis));
        ICHECK(BindingKindCompatibleWithRelation(decoded->fact_axis, binding_kind))
            << "relation.companion uses incompatible binding kind " << ToString(binding_kind);
        for (const String& related_anchor : witness->related_anchor_ids) {
          ICHECK(HasBinding(update, ToString(binding_kind), related_anchor))
              << "relation.companion witness missing update binding for " << related_anchor;
        }
      } else if (decoded->subject_kind == WitnessSubjectKind::kRelation &&
                 decoded->fact_axis == WitnessFactAxis::kCarriedFrom) {
        const Update& update = updates_by_name.at(decoded->subject_anchor_id);
        auto law_kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
        ICHECK(law_kind && RelationAxisCompatibleWithLawKind(decoded->fact_axis, *law_kind))
            << "relation.carried_from requires a recurrence update";
        BindingKind binding_kind =
            DecodeWitnessBindingKind(witness, "binding_kind")
                .value_or(DefaultBindingKindForRelation(decoded->fact_axis));
        ICHECK(BindingKindCompatibleWithRelation(decoded->fact_axis, binding_kind))
            << "relation.carried_from uses incompatible binding kind " << ToString(binding_kind);
        for (const String& related_anchor : witness->related_anchor_ids) {
          ICHECK(HasBinding(update, ToString(binding_kind), related_anchor))
              << "relation.carried_from witness missing update binding for " << related_anchor;
        }
      } else if (decoded->subject_kind == WitnessSubjectKind::kRelation &&
                 decoded->fact_axis == WitnessFactAxis::kDerivesIndexFrom) {
        ICHECK_EQ(std::string(states_by_name.at(decoded->subject_anchor_id)->role),
                  ToString(StateRole::kIndexState))
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
