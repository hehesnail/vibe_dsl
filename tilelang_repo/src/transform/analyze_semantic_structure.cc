/*
 * \file analyze_semantic_structure.cc
 * \brief Build a minimal semantic-structure summary from existing Blackhole analysis attrs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"
#include "common/semantic_witness_payloads.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using tvm::Integer;
using namespace tvm::tl::semantic;

namespace {

// Strip trailing `_N` suffix added by TIR lowering (e.g. `logits_frag_1` -> `logits_frag`).
// Assumption: lowering only appends `_<digits>` to buffer names.  If a future lowering
// pass uses a different naming convention this function must be updated accordingly.
std::string CanonicalBufferName(const std::string& name) {
  size_t pos = name.size();
  while (pos > 0 && std::isdigit(static_cast<unsigned char>(name[pos - 1]))) {
    --pos;
  }
  if (pos > 0 && pos < name.size() && name[pos - 1] == '_') {
    return name.substr(0, pos - 1);
  }
  return name;
}

bool IsTrackedStateScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

class LocalBufferCollector : public tir::StmtExprVisitor {
 public:
  void VisitStmt_(const tir::BlockNode* op) final {
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      Register(buffer);
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  Array<Any> Encode() const {
    Array<Any> states;
    for (const auto& name : order_) {
      const auto& entry = entries_.at(name);
      Map<String, Any> state;
      state.Set("name", String(name));
      state.Set("role", String(ToString(StateRole::kTransient)));
      state.Set("scope", String(entry.scope));
      states.push_back(state);
    }
    return states;
  }

  bool HasIntegerDType(const std::string& name) const {
    auto it = entries_.find(name);
    return it != entries_.end() && it->second.is_integer;
  }

 private:
  struct StateEntry {
    std::string scope;
    bool is_integer{false};
  };

  void Register(const tir::Buffer& buffer) {
    const std::string scope = buffer.scope();
    if (!IsTrackedStateScope(scope)) {
      return;
    }
    const std::string name = CanonicalBufferName(buffer->name);
    if (entries_.count(name)) {
      return;
    }
    entries_.emplace(name, StateEntry{scope, buffer->dtype.is_int() || buffer->dtype.is_uint()});
    order_.push_back(name);
  }

  std::unordered_map<std::string, StateEntry> entries_;
  std::vector<std::string> order_;
};

void PushStringUnique(Array<Any>* arr, std::unordered_set<std::string>* seen,
                      const std::string& value) {
  if (seen->insert(value).second) {
    arr->push_back(String(value));
  }
}

SemanticWitness MakeWitness(const std::string& subject_kind, const std::string& subject_anchor_id,
                            const std::string& fact_axis, Map<String, Any> fact_value,
                            Array<String> related_anchor_ids,
                            Array<String> evidence_sources) {
  return SemanticWitness(String(subject_kind), String(subject_anchor_id), String(fact_axis),
                         std::move(fact_value), std::move(related_anchor_ids),
                         std::move(evidence_sources),
                         String("analyze_semantic_structure"));
}

}  // namespace

tir::transform::Pass AnalyzeSemanticStructure() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }

    Map<String, Any> structure;
    Array<Any> domain_axes;
    Array<Any> domain_traits;
    std::unordered_set<std::string> seen_traits;

    if (auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
      if (auto axes = work.value().find("axes"); axes != work.value().end()) {
        domain_axes = tvm::Downcast<Array<Any>>((*axes).second);
      }
      if (auto bounds = work.value().find("work_dependent_loop_bounds");
          bounds != work.value().end() &&
          !tvm::Downcast<Array<Any>>((*bounds).second).empty()) {
        PushStringUnique(&domain_traits, &seen_traits, "work_dependent_bounds");
      }
      if (auto derived = work.value().find("derived_index_exprs");
          derived != work.value().end() &&
          !tvm::Downcast<Array<Any>>((*derived).second).empty()) {
        PushStringUnique(&domain_traits, &seen_traits, "derived_indices");
      }
    }

    if (auto pipeline = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
        pipeline && !pipeline.value().empty()) {
      PushStringUnique(&domain_traits, &seen_traits, "pipeline");
    }

    LocalBufferCollector buffer_collector;
    buffer_collector(func->body);

    Array<SemanticWitness> witnesses;
    Array<Any> states;
    std::unordered_map<std::string, int> state_index;
    std::unordered_set<std::string> reduction_targets;
    std::unordered_set<std::string> arg_reduce_targets;
    std::unordered_set<std::string> integer_states;
    std::unordered_set<std::string> loop_carried_states;
    std::unordered_set<std::string> selection_targets;
    std::unordered_map<std::string, Array<Any>> update_sources_by_target;
    std::unordered_map<std::string, std::string> paired_value_state_by_selection_target;
    std::unordered_set<std::string> paired_selection_companions;
    std::unordered_map<std::string, Array<Any>> recurrence_edges_by_target;
    std::unordered_map<std::string, std::string> arg_reduce_target_evidence_sources;
    std::unordered_map<std::string, std::string> loop_carried_state_evidence_sources;
    std::unordered_map<std::string, std::string> selection_target_evidence_sources;
    std::unordered_map<std::string, std::string> update_source_evidence_sources;
    std::unordered_map<std::string, std::string> selection_pair_evidence_sources;
    std::unordered_map<std::string, std::string> recurrence_edge_evidence_sources;
    auto register_state = [&states, &state_index](const std::string& name, const std::string& role,
                                                  const std::string& scope) {
      auto it = state_index.find(name);
      if (it != state_index.end()) {
        auto entry = tvm::Downcast<Map<String, Any>>(states[it->second]);
        entry.Set("role", String(role));
        if (!scope.empty()) {
          entry.Set("scope", String(scope));
        }
        states.Set(it->second, entry);
        return;
      }
      Map<String, Any> entry;
      entry.Set("name", String(name));
      entry.Set("role", String(role));
      entry.Set("scope", String(scope));
      state_index.emplace(name, static_cast<int>(states.size()));
      states.push_back(entry);
    };
    auto register_string_fact = [](std::unordered_set<std::string>* values,
                                   std::unordered_map<std::string, std::string>* evidence_sources,
                                   const std::string& value, const std::string& evidence_source) {
      if (values->insert(value).second) {
        evidence_sources->emplace(value, evidence_source);
      }
    };
    auto register_array_fact = [](std::unordered_map<std::string, Array<Any>>* values,
                                  std::unordered_map<std::string, std::string>* evidence_sources,
                                  const std::string& key, const Array<Any>& value,
                                  const std::string& evidence_source) {
      if (!values->count(key)) {
        values->emplace(key, value);
        evidence_sources->emplace(key, evidence_source);
      }
    };
    auto register_selection_pair = [&](const std::string& companion_target,
                                       const std::string& value_target,
                                       const std::string& evidence_source) {
      if (!paired_value_state_by_selection_target.count(companion_target)) {
        paired_value_state_by_selection_target[companion_target] = value_target;
        paired_selection_companions.insert(companion_target);
        selection_pair_evidence_sources[companion_target] = evidence_source;
      }
    };
    auto ingest_structural_region = [&](const Map<String, Any>& region, bool from_manifest) {
      const std::string selection_target_source =
          from_manifest ? "semantic_manifest" : "selection_targets";
      const std::string update_source_source =
          from_manifest ? "semantic_manifest" : "update_sources";
      const std::string arg_reduce_source =
          from_manifest ? "semantic_manifest" : "fragment_regions";
      const std::string selection_pair_source =
          from_manifest ? "semantic_manifest" : "selection_pairs";
      const std::string recurrence_edge_source =
          from_manifest ? "semantic_manifest" : "recurrence_edges";
      const std::string loop_carried_source =
          from_manifest ? "semantic_manifest" : "loop_carried_state";
      if (region.count("fragment_buffers")) {
        for (const Any& buffer_any : tvm::Downcast<Array<Any>>(region["fragment_buffers"])) {
          auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
          const std::string name = buffer["name"].cast<String>();
          register_state(name, ToString(StateRole::kTransient), buffer["scope"].cast<String>());
          bool is_integer = buffer_collector.HasIntegerDType(name);
          if (auto it = buffer.find("is_integer"); it != buffer.end()) {
            is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
          }
          if (is_integer) {
            integer_states.insert(name);
          }
        }
      }
      if (region.count("loop_carried_state")) {
        for (const Any& carried_any : tvm::Downcast<Array<Any>>(region["loop_carried_state"])) {
          auto carried = tvm::Downcast<Map<String, Any>>(carried_any);
          const std::string name = carried["name"].cast<String>();
          register_string_fact(&loop_carried_states, &loop_carried_state_evidence_sources, name,
                               loop_carried_source);
          register_state(name, ToString(StateRole::kCarry), "");
        }
      }
      if (region.count("selection_targets")) {
        for (const Any& target_any : tvm::Downcast<Array<Any>>(region["selection_targets"])) {
          register_string_fact(&selection_targets, &selection_target_evidence_sources,
                               tvm::Downcast<String>(target_any), selection_target_source);
        }
      }
      if (region.count("update_sources")) {
        for (const Any& source_any : tvm::Downcast<Array<Any>>(region["update_sources"])) {
          auto source_map = tvm::Downcast<Map<String, Any>>(source_any);
          register_array_fact(&update_sources_by_target, &update_source_evidence_sources,
                              source_map["target"].cast<String>(),
                              tvm::Downcast<Array<Any>>(source_map["sources"]),
                              update_source_source);
        }
      }
      if (region.count("arg_reduce_targets")) {
        for (const Any& target_any : tvm::Downcast<Array<Any>>(region["arg_reduce_targets"])) {
          register_string_fact(&arg_reduce_targets, &arg_reduce_target_evidence_sources,
                               tvm::Downcast<String>(target_any), arg_reduce_source);
        }
      }
      if (region.count("selection_pairs")) {
        for (const Any& pair_any : tvm::Downcast<Array<Any>>(region["selection_pairs"])) {
          auto pair_map = tvm::Downcast<Map<String, Any>>(pair_any);
          register_selection_pair(pair_map["companion_target"].cast<String>(),
                                  pair_map["value_target"].cast<String>(),
                                  selection_pair_source);
        }
      }
      if (region.count("recurrence_edges")) {
        for (const Any& edge_any : tvm::Downcast<Array<Any>>(region["recurrence_edges"])) {
          auto edge_map = tvm::Downcast<Map<String, Any>>(edge_any);
          register_array_fact(&recurrence_edges_by_target, &recurrence_edge_evidence_sources,
                              edge_map["target"].cast<String>(),
                              tvm::Downcast<Array<Any>>(edge_map["source_states"]),
                              recurrence_edge_source);
          register_string_fact(&loop_carried_states, &loop_carried_state_evidence_sources,
                               edge_map["target"].cast<String>(), loop_carried_source);
          register_state(edge_map["target"].cast<String>(), ToString(StateRole::kCarry), "");
        }
      }
    };
    auto evidence_source_array = [](const std::string& source) {
      return Array<String>{String(source)};
    };
    auto lookup_evidence_source =
        [](const std::unordered_map<std::string, std::string>& evidence_sources,
           const std::string& key, const char* fallback) {
          auto it = evidence_sources.find(key);
          return it != evidence_sources.end() ? it->second : std::string(fallback);
        };

    if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
      if (auto structural_it = manifest.value().find("structural_regions");
          structural_it != manifest.value().end()) {
        for (const Any& region_any : tvm::Downcast<Array<Any>>((*structural_it).second)) {
          ingest_structural_region(tvm::Downcast<Map<String, Any>>(region_any), true);
        }
      }
    }

    if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
      for (const Any& region_any : regions.value()) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        for (const Any& buffer_any : tvm::Downcast<Array<Any>>(region["fragment_buffers"])) {
          auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
          const std::string name = buffer["name"].cast<String>();
          register_state(name, ToString(StateRole::kTransient), buffer["scope"].cast<String>());
          bool is_integer = buffer_collector.HasIntegerDType(name);
          if (auto it = buffer.find("is_integer"); it != buffer.end()) {
            is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
          }
          if (is_integer) {
            integer_states.insert(name);
          }
        }
        ingest_structural_region(region, false);
        for (const Any& reduction_any : tvm::Downcast<Array<Any>>(region["row_reductions"])) {
          auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
          const std::string target = reduction["target"].cast<String>();
          reduction_targets.insert(target);
          if (buffer_collector.HasIntegerDType(target)) {
            integer_states.insert(target);
          }
          // A reduction target is index_state only if it actually carries index information:
          // either it has integer dtype, or it is an integer arg-reduce target.  Non-integer
          // arg_reduce_targets (e.g. the value component of an arg-reduce pair) remain
          // reduction_accumulator — they participate in arg-reduce but don't carry indices.
          const bool is_index = integer_states.count(target) ||
                                (arg_reduce_targets.count(target) &&
                                 buffer_collector.HasIntegerDType(target));
          const std::string role = is_index ? ToString(StateRole::kIndexState)
                                            : ToString(StateRole::kReductionAccumulator);
          register_state(target, role, "");
        }
      }
    } else {
      for (const Any& state_any : buffer_collector.Encode()) {
        auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
        const std::string name = state_map["name"].cast<String>();
        register_state(name, state_map["role"].cast<String>(), state_map["scope"].cast<String>());
        if (buffer_collector.HasIntegerDType(name)) {
          integer_states.insert(name);
        }
      }
    }

    for (const std::string& carried : loop_carried_states) {
      if (!reduction_targets.count(carried)) {
        register_state(carried, ToString(StateRole::kCarry), "");
      }
    }
    for (const std::string& name : selection_targets) {
      if (paired_selection_companions.count(name)) {
        register_state(name, ToString(StateRole::kIndexState), "");
      } else if (!integer_states.count(name)) {
        register_state(name, ToString(StateRole::kSelectionState), "");
      }
    }

    for (const Any& state_any : states) {
      auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
      auto role = ParseStateRole(state_map["role"].cast<String>());
      ICHECK(role) << "AnalyzeSemanticStructure encountered unsupported state role "
                   << state_map["role"].cast<String>();
      witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kState),
                                      state_map["name"].cast<String>(),
                                      ToString(WitnessFactAxis::kRole),
                                      MakeStateRolePayload(*role), Array<String>{},
                                      Array<String>{String("states")}));
    }

    for (const std::string& target : arg_reduce_targets) {
      // derives_index_from requires index_state; only emit for integer arg-reduce targets.
      if (!integer_states.count(target)) {
        continue;
      }
      const std::string evidence_source =
          lookup_evidence_source(arg_reduce_target_evidence_sources, target, "fragment_regions");
      witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), target,
                                      ToString(WitnessFactAxis::kDerivesIndexFrom),
                                      MakeEmptyPayload(), Array<String>{},
                                      evidence_source_array(evidence_source)));
    }

    Array<Any> updates;
    {
      Map<String, Any> entry;
      entry.Set("name", String("root_map"));
      entry.Set("kind", String(ToString(UpdateLawKind::kMap)));
      // root_map targets the single state only when unambiguous;
      // otherwise leave empty — Phase B decides spatial ownership.
      String root_target("");
      if (states.size() == 1) {
        root_target = tvm::Downcast<Map<String, Any>>(states[0])["name"].cast<String>();
      }
      entry.Set("target_state", root_target);
      updates.push_back(entry);
      witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), "root_map",
                                      ToString(WitnessFactAxis::kLawFamily),
                                      MakeUpdateLawFamilyPayload(UpdateLawKind::kMap),
                                      Array<String>{},
                                      Array<String>{String("semantic_structure")}));
    }
    if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
      for (const Any& region_any : regions.value()) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        for (const Any& reduction_any : tvm::Downcast<Array<Any>>(region["row_reductions"])) {
          auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
          Map<String, Any> entry;
          entry.Set("name", String(std::string("reduce_") + reduction["target"].cast<String>()));
          entry.Set("kind", String(ToString(UpdateLawKind::kReduce)));
          entry.Set("target_state", reduction["target"].cast<String>());
          entry.Set("reduce_kind", reduction["kind"].cast<String>());
          if (auto it = update_sources_by_target.find(reduction["target"].cast<String>());
              it != update_sources_by_target.end()) {
            entry.Set("source_states", it->second);
          }
          updates.push_back(entry);
          const std::string update_name =
              std::string("reduce_") + reduction["target"].cast<String>();
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                          ToString(WitnessFactAxis::kLawFamily),
                                          MakeUpdateLawFamilyPayload(UpdateLawKind::kReduce),
                                            Array<String>{},
                                            Array<String>{String("row_reductions")}));
          if (auto it = update_sources_by_target.find(reduction["target"].cast<String>());
              it != update_sources_by_target.end()) {
            const std::string update_source =
                lookup_evidence_source(update_source_evidence_sources,
                                       reduction["target"].cast<String>(), "update_sources");
            witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                            ToString(WitnessFactAxis::kSourceSet),
                                            MakeUpdateSourceSetPayload(it->second),
                                            Array<String>{},
                                            evidence_source_array(update_source)));
          }
        }
      }
    }

    if (!selection_targets.empty()) {
      for (const std::string& state_name : selection_targets) {
        Map<String, Any> entry;
        entry.Set("name", String(std::string("select_") + state_name));
        entry.Set("kind", String(ToString(UpdateLawKind::kSelect)));
        entry.Set("target_state", String(state_name));
        entry.Set("traits", Array<Any>{String("selected"), String("indexed")});
        if (auto it = update_sources_by_target.find(state_name); it != update_sources_by_target.end()) {
          entry.Set("source_states", it->second);
        }
        if (auto it = paired_value_state_by_selection_target.find(state_name);
            it != paired_value_state_by_selection_target.end()) {
          Array<Any> bindings;
          Map<String, Any> binding;
          binding.Set("kind", String(ToString(BindingKind::kPairedValueState)));
          binding.Set("symbol", String("state"));
          binding.Set("value_repr", String(it->second));
          bindings.push_back(binding);
          entry.Set("bindings", bindings);
        }
        updates.push_back(entry);
        const std::string update_name = std::string("select_") + state_name;
        const std::string selection_target_source =
            lookup_evidence_source(selection_target_evidence_sources, state_name,
                                   "selection_targets");
        witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                        ToString(WitnessFactAxis::kLawFamily),
                                        MakeUpdateLawFamilyPayload(UpdateLawKind::kSelect),
                                        Array<String>{},
                                        evidence_source_array(selection_target_source)));
        if (auto it = update_sources_by_target.find(state_name); it != update_sources_by_target.end()) {
          const std::string update_source =
              lookup_evidence_source(update_source_evidence_sources, state_name, "update_sources");
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                          ToString(WitnessFactAxis::kSourceSet),
                                          MakeUpdateSourceSetPayload(it->second),
                                          Array<String>{},
                                          evidence_source_array(update_source)));
        }
        if (auto it = paired_value_state_by_selection_target.find(state_name);
            it != paired_value_state_by_selection_target.end()) {
          const std::string selection_pair_source =
              lookup_evidence_source(selection_pair_evidence_sources, state_name,
                                     "selection_pairs");
          witnesses.push_back(
              MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                          ToString(WitnessFactAxis::kCompanion),
                          MakeRelationBindingPayload(BindingKind::kPairedValueState),
                          Array<String>{String(it->second)},
                          evidence_source_array(selection_pair_source)));
        }
      }
    }

    if (!loop_carried_states.empty()) {
      for (const std::string& state_name : loop_carried_states) {
        Map<String, Any> entry;
        entry.Set("name", String(std::string("recur_") + state_name));
        entry.Set("kind", String(ToString(UpdateLawKind::kRecurrence)));
        entry.Set("target_state", String(state_name));
        entry.Set("traits", Array<Any>{String("carried"), String("staged")});
        if (auto it = recurrence_edges_by_target.find(state_name);
            it != recurrence_edges_by_target.end()) {
          entry.Set("source_states", it->second);
          Array<Any> bindings;
          for (const Any& source_any : it->second) {
            Map<String, Any> binding;
            binding.Set("kind", String(ToString(BindingKind::kRecurrenceSourceState)));
            binding.Set("symbol", String("state"));
            binding.Set("value_repr", tvm::Downcast<String>(source_any));
            bindings.push_back(binding);
          }
          entry.Set("bindings", bindings);
        } else if (auto it = update_sources_by_target.find(state_name);
                   it != update_sources_by_target.end()) {
          entry.Set("source_states", it->second);
        }
        updates.push_back(entry);
        const std::string update_name = std::string("recur_") + state_name;
        const std::string loop_carried_source =
            lookup_evidence_source(loop_carried_state_evidence_sources, state_name,
                                   "loop_carried_state");
        witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                        ToString(WitnessFactAxis::kLawFamily),
                                        MakeUpdateLawFamilyPayload(UpdateLawKind::kRecurrence),
                                        Array<String>{},
                                        evidence_source_array(loop_carried_source)));
        Map<String, Any> ordering_payload;
        ordering_payload.Set("ordering", String("ordered"));
        const std::string recurrence_edge_source =
            lookup_evidence_source(recurrence_edge_evidence_sources, state_name,
                                   "recurrence_edges");
        witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                        ToString(WitnessFactAxis::kOrdering),
                                        std::move(ordering_payload), Array<String>{},
                                        evidence_source_array(recurrence_edge_source)));
        if (auto it = recurrence_edges_by_target.find(state_name);
            it != recurrence_edges_by_target.end()) {
          Array<String> related_sources;
          for (const Any& source_any : it->second) {
            related_sources.push_back(tvm::Downcast<String>(source_any));
          }
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                                          ToString(WitnessFactAxis::kCarriedFrom),
                                          MakeRelationBindingPayload(
                                              BindingKind::kRecurrenceSourceState),
                                          std::move(related_sources),
                                          evidence_source_array(recurrence_edge_source)));
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                          ToString(WitnessFactAxis::kSourceSet),
                                          MakeUpdateSourceSetPayload(it->second),
                                          Array<String>{},
                                          evidence_source_array(recurrence_edge_source)));
        } else if (auto it = update_sources_by_target.find(state_name);
                   it != update_sources_by_target.end()) {
          const std::string update_source =
              lookup_evidence_source(update_source_evidence_sources, state_name, "update_sources");
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                          ToString(WitnessFactAxis::kSourceSet),
                                          MakeUpdateSourceSetPayload(it->second),
                                          Array<String>{},
                                          evidence_source_array(update_source)));
        }
      }
    }

    Array<Any> seeds;
    std::unordered_set<std::string> seen_seed_markers;
    if (auto semantic_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticSeeds)) {
      if (auto capture = semantic_seeds.value().find("capture_kinds");
          capture != semantic_seeds.value().end()) {
        for (const Any& seed_any : tvm::Downcast<Array<Any>>((*capture).second)) {
          PushStringUnique(&seeds, &seen_seed_markers, tvm::Downcast<String>(seed_any));
        }
      }
    }
    Array<Any> supplements;
    if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
      PushStringUnique(&seeds, &seen_seed_markers, "explicit_op_manifest");

      Array<Any> manifest_op_kinds;
      std::unordered_set<std::string> seen_manifest_op_kinds;
      if (auto op_it = manifest.value().find("operations"); op_it != manifest.value().end()) {
        for (const Any& op_any : tvm::Downcast<Array<Any>>((*op_it).second)) {
          auto op_map = tvm::Downcast<Map<String, Any>>(op_any);
          PushStringUnique(&manifest_op_kinds, &seen_manifest_op_kinds, op_map["kind"].cast<String>());
        }
      }

      int ordered_region_count = 0;
      if (auto region_it = manifest.value().find("ordered_regions");
          region_it != manifest.value().end()) {
        for (const Any& region_any : tvm::Downcast<Array<Any>>((*region_it).second)) {
          auto region = tvm::Downcast<Map<String, Any>>(region_any);
          ++ordered_region_count;
          witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kBoundary),
                                          region["anchor"].cast<String>(),
                                          ToString(WitnessFactAxis::kOrderedRegion),
                                          MakeEmptyPayload(), Array<String>{},
                                          Array<String>{String("semantic_manifest")}));
        }
      }

      Map<String, Any> supplement_payload;
      supplement_payload.Set("source", String("semantic_manifest"));
      supplement_payload.Set("operation_kinds", manifest_op_kinds);
      supplement_payload.Set("ordered_region_count", Integer(ordered_region_count));
      Map<String, Any> supplement;
      supplement.Set("kind", String(ToString(SupplementKind::kSemanticBoundary)));
      supplement.Set("payload", supplement_payload);
      supplements.push_back(supplement);
    }

    structure.Set("domain_name", String("device_program"));
    structure.Set("domain_axes", domain_axes);
    structure.Set("domain_traits", domain_traits);
    structure.Set("states", states);
    structure.Set("updates", updates);
    structure.Set("seeds", seeds);
    structure.Set("supplements", supplements);

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticStructure, structure);
    attrs.Set(attr::kTLSemanticWitnesses, witnesses);
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.AnalyzeSemanticStructure", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSemanticStructure", AnalyzeSemanticStructure);
}

}  // namespace tl
}  // namespace tvm
