/*!
 * \file semantic_rebind.cc
 * \brief Typed rebind helpers for Blackhole semantic companion programs.
 */

#include "semantic_rebind.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <vector>

#include "semantic_refinement_rules.h"
#include "semantic_state_effect_graph.h"

namespace tvm {
namespace tl {
namespace semantic {
namespace {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

std::string RemapName(const std::string& value, const SemanticRebindPlan& plan) {
  if (auto it = plan.state_remap.find(value); it != plan.state_remap.end()) {
    return it->second;
  }
  if (auto it = plan.update_remap.find(value); it != plan.update_remap.end()) {
    return it->second;
  }
  return value;
}

std::unordered_map<std::string, std::string> DecodeRemapEntries(const Map<String, Any>& plan,
                                                                const char* key) {
  std::unordered_map<std::string, std::string> remap;
  auto it = plan.find(String(key));
  if (it == plan.end()) {
    return remap;
  }
  for (const Any& entry_any : tvm::Downcast<Array<Any>>((*it).second)) {
    auto entry = tvm::Downcast<Map<String, Any>>(entry_any);
    remap[entry["old"].cast<String>()] = entry["new"].cast<String>();
  }
  return remap;
}

Array<Any> RewriteStringArray(const Array<Any>& values, const SemanticRebindPlan& plan) {
  Array<Any> rewritten;
  for (const Any& value : values) {
    rewritten.push_back(String(RemapName(value.cast<String>(), plan)));
  }
  return rewritten;
}

Array<String> RewriteStringArray(const Array<String>& values, const SemanticRebindPlan& plan) {
  Array<String> rewritten;
  for (const String& value : values) {
    rewritten.push_back(String(RemapName(value, plan)));
  }
  return rewritten;
}

Array<Any> RewriteStructureStates(const Array<Any>& states, const SemanticRebindPlan& plan) {
  Array<Any> rewritten;
  for (const Any& state_any : states) {
    auto state = tvm::Downcast<Map<String, Any>>(state_any);
    Map<String, Any> entry = state;
    entry.Set("name", String(RemapName(state["name"].cast<String>(), plan)));
    rewritten.push_back(entry);
  }
  return rewritten;
}

Array<Any> RewriteStructureUpdates(const Array<Any>& updates, const SemanticRebindPlan& plan) {
  Array<Any> rewritten;
  for (const Any& update_any : updates) {
    auto update = tvm::Downcast<Map<String, Any>>(update_any);
    Map<String, Any> entry = update;
    entry.Set("name", String(RemapName(update["name"].cast<String>(), plan)));
    entry.Set("target_state", String(RemapName(update["target_state"].cast<String>(), plan)));
    if (update.count("source_states")) {
      entry.Set("source_states",
                RewriteStringArray(tvm::Downcast<Array<Any>>(update["source_states"]), plan));
    }
    if (update.count("bindings")) {
      Array<Any> rewritten_bindings;
      for (const Any& binding_any : tvm::Downcast<Array<Any>>(update["bindings"])) {
        auto binding = tvm::Downcast<Map<String, Any>>(binding_any);
        Map<String, Any> binding_entry = binding;
        binding_entry.Set("value_repr",
                          String(RemapName(binding["value_repr"].cast<String>(), plan)));
        rewritten_bindings.push_back(binding_entry);
      }
      entry.Set("bindings", rewritten_bindings);
    }
    rewritten.push_back(entry);
  }
  return rewritten;
}

}  // namespace

SemanticRebindPlan DecodeSemanticRebindPlan(const Map<String, Any>& plan) {
  auto scope_it = plan.find("rebind_scope");
  ICHECK(scope_it != plan.end()) << "Typed rebind plan requires rebind_scope";
  auto scope = ParseRebindScope((*scope_it).second.cast<String>());
  ICHECK(scope) << "Unsupported rebind_scope in typed rebind plan: "
                << (*scope_it).second.cast<String>();

  SemanticRebindPlan decoded{*scope, "", {}, {}, {}};
  if (auto reason_it = plan.find("reason"); reason_it != plan.end()) {
    decoded.reason = (*reason_it).second.cast<String>();
  }
  if (auto trace_it = plan.find("rebind_trace"); trace_it != plan.end()) {
    decoded.trace = tvm::Downcast<Array<Any>>((*trace_it).second);
  }
  decoded.state_remap = DecodeRemapEntries(plan, "state_remap");
  decoded.update_remap = DecodeRemapEntries(plan, "update_remap");
  if (decoded.trace.empty()) {
    // Sort keys for deterministic trace order across runs.
    std::vector<std::pair<std::string, std::string>> sorted_state(
        decoded.state_remap.begin(), decoded.state_remap.end());
    std::sort(sorted_state.begin(), sorted_state.end());
    for (const auto& kv : sorted_state) {
      decoded.trace.push_back(Map<String, Any>{{"kind", String("state")},
                                               {"old", String(kv.first)},
                                               {"new", String(kv.second)}});
    }
    std::vector<std::pair<std::string, std::string>> sorted_update(
        decoded.update_remap.begin(), decoded.update_remap.end());
    std::sort(sorted_update.begin(), sorted_update.end());
    for (const auto& kv : sorted_update) {
      decoded.trace.push_back(Map<String, Any>{{"kind", String("update")},
                                               {"old", String(kv.first)},
                                               {"new", String(kv.second)}});
    }
  }
  return decoded;
}

Map<String, Any> ApplySemanticRebindToStructure(const Map<String, Any>& structure,
                                                const SemanticRebindPlan& plan) {
  Map<String, Any> rewritten = structure;
  if (structure.count("states")) {
    rewritten.Set("states", RewriteStructureStates(tvm::Downcast<Array<Any>>(structure["states"]), plan));
  }
  if (structure.count("updates")) {
    rewritten.Set("updates",
                  RewriteStructureUpdates(tvm::Downcast<Array<Any>>(structure["updates"]), plan));
  }
  return rewritten;
}

Array<SemanticWitness> ApplySemanticRebindToWitnesses(const Array<SemanticWitness>& witnesses,
                                                      const SemanticRebindPlan& plan) {
  Array<SemanticWitness> rewritten;
  for (const SemanticWitness& witness : witnesses) {
    Array<String> related;
    for (const String& related_anchor : witness->related_anchor_ids) {
      related.push_back(String(RemapName(related_anchor, plan)));
    }
    rewritten.push_back(SemanticWitness(
        witness->subject_kind, String(RemapName(witness->subject_anchor_id, plan)),
        witness->fact_axis, witness->fact_value, related, witness->evidence_sources,
        witness->canonicalization_point));
  }
  return rewritten;
}

SemanticProgram ApplySemanticRebindToProgram(const SemanticProgram& program,
                                             const Array<SemanticWitness>& witnesses,
                                             const SemanticRebindPlan& plan) {
  Array<State> states;
  for (const State& state : program->states) {
    states.push_back(State(String(RemapName(state->name, plan)), state->role, state->storage_scope,
                           state->anchors));
  }

  Array<Update> updates;
  for (const Update& update : program->updates) {
    Array<String> source_states = RewriteStringArray(update->law->source_states, plan);
    Array<TIRValueBinding> bindings;
    for (const TIRValueBinding& binding : update->bindings) {
      bindings.push_back(TIRValueBinding(binding->kind, binding->symbol,
                                         String(RemapName(binding->value_repr, plan))));
    }
    UpdateLaw law(update->law->kind, String(RemapName(update->law->target_state, plan)),
                  source_states, update->law->access_maps);
    updates.push_back(Update(String(RemapName(update->name, plan)),
                             String(RemapName(update->state_name, plan)), law, update->anchors,
                             bindings));
  }

  BuiltStateEffectGraph graph = BuildStateEffectGraph(states, updates, witnesses);
  return SemanticProgram(program->domains, states, updates, program->supplements, program->seeds,
                         program->anchors, graph.state_versions, graph.state_defs,
                         graph.state_uses, graph.state_joins);
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
