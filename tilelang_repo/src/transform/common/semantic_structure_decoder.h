/*!
 * \file semantic_structure_decoder.h
 * \brief Decode tl.semantic_structure + tl.semantic_witnesses into local semantic facts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_STRUCTURE_DECODER_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_STRUCTURE_DECODER_H_

#include <tvm/ir/attrs.h>
#include <tvm/tir/function.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "companion_base.h"
#include "semantic_program.h"
#include "semantic_refinement_rules.h"
#include "semantic_state_effect_graph.h"
#include "semantic_witness_decoder.h"

namespace tvm {
namespace tl {
namespace semantic {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

inline Array<String> DowncastSemanticStructureStringArray(const Array<Any>& items) {
  Array<String> result;
  for (const Any& item : items) {
    result.push_back(tvm::Downcast<String>(item));
  }
  return result;
}

inline Array<TIRValueBinding> DowncastSemanticStructureBindingArray(const Array<Any>& items) {
  Array<TIRValueBinding> result;
  for (const Any& item : items) {
    auto binding_map = tvm::Downcast<Map<String, Any>>(item);
    result.push_back(
        TIRValueBinding(tvm::Downcast<String>(binding_map["kind"]),
                        tvm::Downcast<String>(binding_map["symbol"]),
                        tvm::Downcast<String>(binding_map["value_repr"])));
  }
  return result;
}

inline void PushSemanticBindingUnique(Array<TIRValueBinding>* bindings, BindingKind kind,
                                      const std::string& value_repr) {
  for (const TIRValueBinding& binding : *bindings) {
    if (std::string(binding->kind) == ToString(kind) &&
        std::string(binding->value_repr) == value_repr) {
      return;
    }
  }
  bindings->push_back(
      TIRValueBinding(String(ToString(kind)), String("state"), String(value_repr)));
}

inline SemanticProgram DecodeSemanticProgramFromStructure(
    const Map<String, Any>& structure, const Array<SemanticWitness>& witnesses) {
  std::unordered_map<std::string, StateRole> state_role_by_anchor;
  std::unordered_map<std::string, UpdateLawKind> update_kind_by_anchor;
  std::unordered_map<std::string, std::vector<std::string>> update_sources_by_anchor;
  std::unordered_map<std::string, std::vector<std::string>> companion_relations_by_update;
  std::unordered_map<std::string, BindingKind> companion_binding_kind_by_update;
  std::unordered_map<std::string, std::vector<std::string>> carried_relations_by_update;
  std::unordered_map<std::string, BindingKind> carried_binding_kind_by_update;
  for (const SemanticWitness& witness : witnesses) {
    auto decoded = DecodeSemanticWitness(witness);
    if (!decoded) {
      continue;
    }
    if (decoded->subject_kind == WitnessSubjectKind::kState &&
        decoded->fact_axis == WitnessFactAxis::kRole) {
      if (auto payload = DecodeWitnessStateRolePayload(witness)) {
        state_role_by_anchor[decoded->subject_anchor_id] = payload->role;
      }
    } else if (decoded->subject_kind == WitnessSubjectKind::kUpdate &&
               decoded->fact_axis == WitnessFactAxis::kLawFamily) {
      if (auto payload = DecodeWitnessUpdateLawFamilyPayload(witness)) {
        update_kind_by_anchor[decoded->subject_anchor_id] = payload->kind;
      }
    } else if (decoded->subject_kind == WitnessSubjectKind::kUpdate &&
               decoded->fact_axis == WitnessFactAxis::kSourceSet) {
      if (auto payload = DecodeWitnessUpdateSourceSetPayload(witness)) {
        update_sources_by_anchor[decoded->subject_anchor_id] = payload->sources;
      }
    } else if (decoded->subject_kind == WitnessSubjectKind::kRelation &&
               decoded->fact_axis == WitnessFactAxis::kCompanion) {
      companion_relations_by_update[decoded->subject_anchor_id] = {};
      for (const String& related : witness->related_anchor_ids) {
        companion_relations_by_update[decoded->subject_anchor_id].push_back(related);
      }
      BindingKind binding_kind = DefaultBindingKindForRelation(decoded->fact_axis);
      if (auto payload = DecodeWitnessRelationBindingPayload(witness)) {
        binding_kind = payload->binding_kind;
      }
      companion_binding_kind_by_update[decoded->subject_anchor_id] = binding_kind;
    } else if (decoded->subject_kind == WitnessSubjectKind::kRelation &&
               decoded->fact_axis == WitnessFactAxis::kCarriedFrom) {
      carried_relations_by_update[decoded->subject_anchor_id] = {};
      for (const String& related : witness->related_anchor_ids) {
        carried_relations_by_update[decoded->subject_anchor_id].push_back(related);
      }
      BindingKind binding_kind = DefaultBindingKindForRelation(decoded->fact_axis);
      if (auto payload = DecodeWitnessRelationBindingPayload(witness)) {
        binding_kind = payload->binding_kind;
      }
      carried_binding_kind_by_update[decoded->subject_anchor_id] = binding_kind;
    }
  }

  Array<TIRAnchor> anchors{TIRAnchor("source_attr", attr::kTLSemanticStructure)};
  Array<Domain> domains;
  domains.push_back(Domain(
      tvm::Downcast<String>(structure["domain_name"]),
      DowncastSemanticStructureStringArray(tvm::Downcast<Array<Any>>(structure["domain_axes"])),
      DowncastSemanticStructureStringArray(tvm::Downcast<Array<Any>>(structure["domain_traits"])),
      anchors));

  Array<State> states;
  for (const Any& state_any : tvm::Downcast<Array<Any>>(structure["states"])) {
    auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
    String state_name = tvm::Downcast<String>(state_map["name"]);
    String state_role = tvm::Downcast<String>(state_map["role"]);
    if (auto it = state_role_by_anchor.find(std::string(state_name));
        it != state_role_by_anchor.end()) {
      state_role = String(ToString(it->second));
    }
    Array<TIRAnchor> state_anchors{TIRAnchor("state_source", state_role)};
    states.push_back(
        State(state_name, state_role, tvm::Downcast<String>(state_map["scope"]), state_anchors));
  }

  Array<Update> updates;
  for (const Any& update_any : tvm::Downcast<Array<Any>>(structure["updates"])) {
    auto update_map = tvm::Downcast<Map<String, Any>>(update_any);
    String update_name = tvm::Downcast<String>(update_map["name"]);
    String update_kind = tvm::Downcast<String>(update_map["kind"]);
    if (auto it = update_kind_by_anchor.find(std::string(update_name));
        it != update_kind_by_anchor.end()) {
      update_kind = String(ToString(it->second));
    }
    String target_state = tvm::Downcast<String>(update_map["target_state"]);
    Array<String> source_states;
    if (auto it = update_sources_by_anchor.find(std::string(update_name));
        it != update_sources_by_anchor.end()) {
      for (const auto& source : it->second) {
        source_states.push_back(String(source));
      }
    } else if (update_map.count("source_states")) {
      source_states = DowncastSemanticStructureStringArray(
          tvm::Downcast<Array<Any>>(update_map["source_states"]));
    } else if (update_map.count("target_state") && !std::string(target_state).empty()) {
      source_states.push_back(target_state);
    }
    Array<String> access_traits;
    if (update_map.count("traits")) {
      access_traits =
          DowncastSemanticStructureStringArray(tvm::Downcast<Array<Any>>(update_map["traits"]));
    }
    Array<AccessMap> access_maps{AccessMap("tir_region", {}, access_traits)};
    UpdateLaw law(update_kind, target_state, source_states, access_maps);
    Array<TIRAnchor> update_anchors{TIRAnchor("update_kind", update_kind)};
    Array<TIRValueBinding> bindings{
        TIRValueBinding("target_state", "state", target_state)};
    if (update_map.count("bindings")) {
      for (const TIRValueBinding& binding : DowncastSemanticStructureBindingArray(
               tvm::Downcast<Array<Any>>(update_map["bindings"]))) {
        bindings.push_back(binding);
      }
    }
    if (auto it = companion_relations_by_update.find(std::string(update_name));
        it != companion_relations_by_update.end()) {
      const BindingKind binding_kind =
          companion_binding_kind_by_update.count(std::string(update_name))
              ? companion_binding_kind_by_update[std::string(update_name)]
              : BindingKind::kPairedValueState;
      for (const auto& related : it->second) {
        PushSemanticBindingUnique(&bindings, binding_kind, related);
      }
    }
    if (auto it = carried_relations_by_update.find(std::string(update_name));
        it != carried_relations_by_update.end()) {
      const BindingKind binding_kind =
          carried_binding_kind_by_update.count(std::string(update_name))
              ? carried_binding_kind_by_update[std::string(update_name)]
              : BindingKind::kRecurrenceSourceState;
      for (const auto& related : it->second) {
        PushSemanticBindingUnique(&bindings, binding_kind, related);
      }
    }
    updates.push_back(Update(update_name, target_state, law, update_anchors, bindings));
  }

  Array<SemanticSupplement> supplements;
  if (structure.count("supplements")) {
    for (const Any& supplement_any : tvm::Downcast<Array<Any>>(structure["supplements"])) {
      auto supplement_map = tvm::Downcast<Map<String, Any>>(supplement_any);
      supplements.push_back(SemanticSupplement(
          tvm::Downcast<String>(supplement_map["kind"]),
          tvm::Downcast<Map<String, Any>>(supplement_map["payload"])));
    }
  }

  Array<String> seeds =
      DowncastSemanticStructureStringArray(tvm::Downcast<Array<Any>>(structure["seeds"]));
  BuiltStateEffectGraph graph = BuildStateEffectGraph(states, updates, witnesses);
  return SemanticProgram(domains, states, updates, supplements, seeds, anchors, graph.state_versions,
                         graph.state_defs, graph.state_uses, graph.state_joins);
}

inline std::optional<SemanticProgram> MaybeDecodeSemanticProgramFromFunc(const tir::PrimFunc& func) {
  auto maybe_structure = func->GetAttr<Map<String, Any>>(attr::kTLSemanticStructure);
  if (!maybe_structure) {
    return std::nullopt;
  }
  auto maybe_witnesses = func->GetAttr<Array<SemanticWitness>>(attr::kTLSemanticWitnesses);
  return DecodeSemanticProgramFromStructure(maybe_structure.value(),
                                            maybe_witnesses.value_or(Array<SemanticWitness>{}));
}

inline SemanticProgram RequireSemanticProgramFromFunc(const tir::PrimFunc& func,
                                                      const char* consumer_name) {
  auto maybe_program = MaybeDecodeSemanticProgramFromFunc(func);
  ICHECK(maybe_program)
      << consumer_name << " requires tl.semantic_structure; run AnalyzeSemanticStructure first";
  return maybe_program.value();
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_STRUCTURE_DECODER_H_
