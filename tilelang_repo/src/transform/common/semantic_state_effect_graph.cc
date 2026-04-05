/*!
 * \file semantic_state_effect_graph.cc
 * \brief Build/query helpers for SemanticProgram internal state/effect graph.
 */

#include "semantic_state_effect_graph.h"

#include <unordered_map>
#include <unordered_set>

#include "semantic_vocab.h"
#include "semantic_witness_decoder.h"

namespace tvm {
namespace tl {
namespace semantic {
namespace {

using tvm::ffi::Array;
using tvm::ffi::String;

std::string InitialVersionName(const std::string& state_name) {
  return "state_" + state_name + "_v0";
}

std::string UpdateResultVersionName(const std::string& update_name) {
  return "update_" + update_name + "_out";
}

std::string StateDefName(const std::string& version_name) {
  return "def_" + version_name;
}

std::string StateUseName(const std::string& update_name, const std::string& state_name,
                         StateUseKind kind) {
  return "use_" + update_name + "_" + state_name + "_" + ToString(kind);
}

std::string StateJoinName(const std::string& update_name) {
  return "join_" + update_name;
}

std::string StateJoinOutputVersionName(const std::string& update_name) {
  return "join_" + update_name + "_out";
}

bool StateHasRole(const State& state, StateRole role) {
  auto parsed = ParseStateRole(static_cast<std::string>(state->role));
  return parsed && *parsed == role;
}

}  // namespace

BuiltStateEffectGraph BuildStateEffectGraph(const Array<State>& states, const Array<Update>& updates,
                                            const Array<SemanticWitness>& witnesses) {
  BuiltStateEffectGraph graph;
  std::unordered_map<std::string, std::string> current_version_by_state;
  std::unordered_map<std::string, StateRole> state_role_by_name;
  std::unordered_map<std::string, std::vector<std::string>> companion_states_by_update;
  std::unordered_map<std::string, std::vector<std::string>> carried_states_by_update;

  for (const State& state : states) {
    if (auto role = ParseStateRole(static_cast<std::string>(state->role))) {
      state_role_by_name[std::string(state->name)] = *role;
    }
    const std::string initial_version = InitialVersionName(std::string(state->name));
    current_version_by_state[std::string(state->name)] = initial_version;
    graph.state_versions.push_back(StateVersion(
        String(initial_version), state->name, String(""),
        String(ToString(StateVersionKind::kInitial)), {}, {TIRAnchor("version_kind", "initial")}));
    graph.state_defs.push_back(StateDef(
        String(StateDefName(initial_version)), state->name, String(initial_version), String(""),
        String(ToString(StateDefKind::kInitial)), {TIRAnchor("def_kind", "initial")}));
  }

  for (const SemanticWitness& witness : witnesses) {
    auto decoded = DecodeSemanticWitness(witness);
    if (!decoded || decoded->subject_kind != WitnessSubjectKind::kRelation) {
      continue;
    }
    if (decoded->fact_axis == WitnessFactAxis::kCompanion) {
      for (const String& related : witness->related_anchor_ids) {
        companion_states_by_update[decoded->subject_anchor_id].push_back(related);
      }
    } else if (decoded->fact_axis == WitnessFactAxis::kCarriedFrom) {
      for (const String& related : witness->related_anchor_ids) {
        carried_states_by_update[decoded->subject_anchor_id].push_back(related);
      }
    }
  }

  for (const Update& update : updates) {
    const std::string update_name = update->name;
    const std::string target_state = update->state_name;
    std::vector<std::string> source_versions;

    for (const String& source_state : update->law->source_states) {
      const std::string source_name = source_state;
      const std::string version_name = current_version_by_state.count(source_name)
                                           ? current_version_by_state.at(source_name)
                                           : InitialVersionName(source_name);
      graph.state_uses.push_back(StateUse(
          String(StateUseName(update_name, source_name, StateUseKind::kSourceState)),
          update->name, String(source_name), String(version_name),
          String(ToString(StateUseKind::kSourceState)),
          {TIRAnchor("use_kind", "source_state")}));
      source_versions.push_back(version_name);
    }

    if (auto it = companion_states_by_update.find(update_name); it != companion_states_by_update.end()) {
      for (const std::string& companion_state : it->second) {
        const std::string version_name = current_version_by_state.count(companion_state)
                                             ? current_version_by_state.at(companion_state)
                                             : InitialVersionName(companion_state);
        graph.state_uses.push_back(StateUse(
            String(StateUseName(update_name, companion_state, StateUseKind::kCompanionState)),
            update->name, String(companion_state), String(version_name),
            String(ToString(StateUseKind::kCompanionState)),
            {TIRAnchor("use_kind", "companion_state")}));
      }
    }

    if (auto it = carried_states_by_update.find(update_name); it != carried_states_by_update.end()) {
      for (const std::string& carried_state : it->second) {
        const std::string version_name = current_version_by_state.count(carried_state)
                                             ? current_version_by_state.at(carried_state)
                                             : InitialVersionName(carried_state);
        graph.state_uses.push_back(StateUse(
            String(StateUseName(update_name, carried_state, StateUseKind::kCarriedState)),
            update->name, String(carried_state), String(version_name),
            String(ToString(StateUseKind::kCarriedState)),
            {TIRAnchor("use_kind", "carried_state")}));
      }
    }

    Array<String> source_version_array;
    for (const std::string& version_name : source_versions) {
      source_version_array.push_back(String(version_name));
    }
    if (target_state.empty()) {
      continue;
    }
    const std::string output_version = UpdateResultVersionName(update_name);
    graph.state_versions.push_back(StateVersion(
        String(output_version), update->state_name, update->name,
        String(ToString(StateVersionKind::kUpdateResult)), source_version_array,
        {TIRAnchor("version_kind", "update_result")}));
    graph.state_defs.push_back(StateDef(
        String(StateDefName(output_version)), update->state_name, String(output_version),
        update->name, String(ToString(StateDefKind::kUpdateResult)),
        {TIRAnchor("def_kind", "update_result")}));

    const std::string previous_target_version =
        current_version_by_state.count(target_state) ? current_version_by_state.at(target_state)
                                                     : InitialVersionName(target_state);
    current_version_by_state[target_state] = output_version;

    const bool is_carry_target =
        state_role_by_name.count(target_state) &&
        state_role_by_name.at(target_state) == StateRole::kCarry;
    const bool has_carried_relation = carried_states_by_update.count(update_name);
    const auto update_kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
    const bool is_recurrence_update =
        update_kind && *update_kind == UpdateLawKind::kRecurrence;
    if (is_carry_target || has_carried_relation || is_recurrence_update) {
      const std::string join_output = StateJoinOutputVersionName(update_name);
      graph.state_versions.push_back(StateVersion(
          String(join_output), update->state_name, String(""),
          String(ToString(StateVersionKind::kUpdateResult)),
          {String(previous_target_version), String(output_version)},
          {TIRAnchor("version_kind", "join_output")}));
      graph.state_joins.push_back(StateJoin(
          String(StateJoinName(update_name)), update->state_name,
          String(ToString(StateJoinKind::kLoopCarried)),
          {String(previous_target_version), String(output_version)}, String(join_output),
          {TIRAnchor("join_kind", "loop_carried")}));
      current_version_by_state[target_state] = join_output;
    }
  }

  return graph;
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
