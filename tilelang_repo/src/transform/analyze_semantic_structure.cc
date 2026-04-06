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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

Array<String> EvidenceSourceArray(const std::string& source) {
  return Array<String>{String(source)};
}

std::string LookupEvidenceSource(
    const std::unordered_map<std::string, std::string>& evidence_sources,
    const std::string& key, const char* fallback) {
  auto it = evidence_sources.find(key);
  return it != evidence_sources.end() ? it->second : std::string(fallback);
}

// ---------------------------------------------------------------------------
// EvidenceAccumulator — collects merged evidence from manifest + fragment_regions
// ---------------------------------------------------------------------------

struct EvidenceAccumulator {
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

  // Reduction evidence from manifest and/or fragment_regions.
  struct ReductionEntry {
    std::string target;
    std::string kind;           // "max" / "sum"
    std::string evidence_source;
  };
  std::vector<ReductionEntry> reductions;
  std::unordered_set<std::string> seen_reduction_targets;

  // Evidence-source tracking per fact.
  std::unordered_map<std::string, std::string> arg_reduce_target_evidence_sources;
  std::unordered_map<std::string, std::string> loop_carried_state_evidence_sources;
  std::unordered_map<std::string, std::string> selection_target_evidence_sources;
  std::unordered_map<std::string, std::string> update_source_evidence_sources;
  std::unordered_map<std::string, std::string> selection_pair_evidence_sources;
  std::unordered_map<std::string, std::string> recurrence_edge_evidence_sources;
  std::unordered_map<std::string, std::string> reduction_evidence_sources;

  void RegisterState(const std::string& name, const std::string& role,
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
  }

  void RegisterStringFact(std::unordered_set<std::string>* values,
                          std::unordered_map<std::string, std::string>* evidence_sources,
                          const std::string& value, const std::string& evidence_source) {
    if (values->insert(value).second) {
      evidence_sources->emplace(value, evidence_source);
    }
  }

  void RegisterArrayFact(std::unordered_map<std::string, Array<Any>>* values,
                         std::unordered_map<std::string, std::string>* evidence_sources,
                         const std::string& key, const Array<Any>& value,
                         const std::string& evidence_source) {
    if (!values->count(key)) {
      values->emplace(key, value);
      evidence_sources->emplace(key, evidence_source);
    }
  }

  void RegisterSelectionPair(const std::string& companion_target,
                             const std::string& value_target,
                             const std::string& evidence_source) {
    if (!paired_value_state_by_selection_target.count(companion_target)) {
      paired_value_state_by_selection_target[companion_target] = value_target;
      paired_selection_companions.insert(companion_target);
      selection_pair_evidence_sources[companion_target] = evidence_source;
    }
  }

  void RegisterReduction(const std::string& target, const std::string& kind,
                         const std::string& evidence_source) {
    if (seen_reduction_targets.insert(target).second) {
      reductions.push_back({target, kind, evidence_source});
      reduction_evidence_sources[target] = evidence_source;
    }
  }

  // Ingest a structural region from either manifest or fragment_regions.
  void IngestStructuralRegion(const Map<String, Any>& region, bool from_manifest,
                              const LocalBufferCollector& buffer_collector) {
    const std::string source_tag = from_manifest ? "semantic_manifest" : "fragment_regions";
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

    if (region.count(manifest_key::kFragmentBuffers)) {
      for (const Any& buffer_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kFragmentBuffers])) {
        auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
        const std::string name = buffer["name"].cast<String>();
        RegisterState(name, ToString(StateRole::kTransient), buffer["scope"].cast<String>());
        bool is_integer = buffer_collector.HasIntegerDType(name);
        if (auto it = buffer.find("is_integer"); it != buffer.end()) {
          is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
        }
        if (is_integer) {
          integer_states.insert(name);
        }
      }
    }
    if (region.count(manifest_key::kLoopCarriedState)) {
      for (const Any& carried_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kLoopCarriedState])) {
        auto carried = tvm::Downcast<Map<String, Any>>(carried_any);
        const std::string name = carried["name"].cast<String>();
        RegisterStringFact(&loop_carried_states, &loop_carried_state_evidence_sources, name,
                           loop_carried_source);
        RegisterState(name, ToString(StateRole::kCarry), "");
      }
    }
    if (region.count(manifest_key::kSelectionTargets)) {
      for (const Any& target_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kSelectionTargets])) {
        RegisterStringFact(&selection_targets, &selection_target_evidence_sources,
                           tvm::Downcast<String>(target_any), selection_target_source);
      }
    }
    if (region.count(manifest_key::kUpdateSources)) {
      for (const Any& source_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kUpdateSources])) {
        auto source_map = tvm::Downcast<Map<String, Any>>(source_any);
        RegisterArrayFact(&update_sources_by_target, &update_source_evidence_sources,
                          source_map["target"].cast<String>(),
                          tvm::Downcast<Array<Any>>(source_map["sources"]),
                          update_source_source);
      }
    }
    if (region.count(manifest_key::kArgReduceTargets)) {
      for (const Any& target_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kArgReduceTargets])) {
        RegisterStringFact(&arg_reduce_targets, &arg_reduce_target_evidence_sources,
                           tvm::Downcast<String>(target_any), arg_reduce_source);
      }
    }
    if (region.count(manifest_key::kSelectionPairs)) {
      for (const Any& pair_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kSelectionPairs])) {
        auto pair_map = tvm::Downcast<Map<String, Any>>(pair_any);
        RegisterSelectionPair(pair_map["companion_target"].cast<String>(),
                              pair_map["value_target"].cast<String>(),
                              selection_pair_source);
      }
    }
    if (region.count(manifest_key::kRecurrenceEdges)) {
      for (const Any& edge_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kRecurrenceEdges])) {
        auto edge_map = tvm::Downcast<Map<String, Any>>(edge_any);
        RegisterArrayFact(&recurrence_edges_by_target, &recurrence_edge_evidence_sources,
                          edge_map["target"].cast<String>(),
                          tvm::Downcast<Array<Any>>(edge_map["source_states"]),
                          recurrence_edge_source);
        RegisterStringFact(&loop_carried_states, &loop_carried_state_evidence_sources,
                           edge_map["target"].cast<String>(), loop_carried_source);
        RegisterState(edge_map["target"].cast<String>(), ToString(StateRole::kCarry), "");
      }
    }
    // row_reductions — manifest-first: only ingest if not already registered.
    if (region.count(manifest_key::kRowReductions)) {
      for (const Any& reduction_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kRowReductions])) {
        auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
        RegisterReduction(reduction["target"].cast<String>(),
                          reduction["kind"].cast<String>(),
                          source_tag);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Phase 1: Collect domain skeleton
// ---------------------------------------------------------------------------

void CollectDomainSkeleton(const tir::PrimFunc& func,
                           Array<Any>* domain_axes,
                           Array<Any>* domain_traits,
                           std::unordered_set<std::string>* seen_traits) {
  if (auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
    if (auto axes = work.value().find("axes"); axes != work.value().end()) {
      *domain_axes = tvm::Downcast<Array<Any>>((*axes).second);
    }
    if (auto bounds = work.value().find("work_dependent_loop_bounds");
        bounds != work.value().end() &&
        !tvm::Downcast<Array<Any>>((*bounds).second).empty()) {
      PushStringUnique(domain_traits, seen_traits, "work_dependent_bounds");
    }
    if (auto derived = work.value().find("derived_index_exprs");
        derived != work.value().end() &&
        !tvm::Downcast<Array<Any>>((*derived).second).empty()) {
      PushStringUnique(domain_traits, seen_traits, "derived_indices");
    }
  }
  if (auto pipeline = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
      pipeline && !pipeline.value().empty()) {
    PushStringUnique(domain_traits, seen_traits, "pipeline");
  }
}

// ---------------------------------------------------------------------------
// Phase 2: Ingest evidence from manifest and fragment_regions
// ---------------------------------------------------------------------------

void IngestAllEvidence(const tir::PrimFunc& func, EvidenceAccumulator* acc,
                       const LocalBufferCollector& buffer_collector) {
  // Manifest-first: ingest structural regions from manifest before fragment_regions.
  if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
    if (auto structural_it = manifest.value().find(manifest_key::kStructuralRegions);
        structural_it != manifest.value().end()) {
      for (const Any& region_any : tvm::Downcast<Array<Any>>((*structural_it).second)) {
        acc->IngestStructuralRegion(
            tvm::Downcast<Map<String, Any>>(region_any), true, buffer_collector);
      }
    }
  }

  // Fragment regions: fallback for evidence not already present in manifest.
  if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
    for (const Any& region_any : regions.value()) {
      auto region = tvm::Downcast<Map<String, Any>>(region_any);
      // Register fragment buffers for state tracking.
      for (const Any& buffer_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kFragmentBuffers])) {
        auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
        const std::string name = buffer["name"].cast<String>();
        acc->RegisterState(name, ToString(StateRole::kTransient),
                           buffer["scope"].cast<String>());
        bool is_integer = buffer_collector.HasIntegerDType(name);
        if (auto it = buffer.find("is_integer"); it != buffer.end()) {
          is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
        }
        if (is_integer) {
          acc->integer_states.insert(name);
        }
      }
      // Structural evidence (manifest-first dedup handled inside IngestStructuralRegion).
      acc->IngestStructuralRegion(region, false, buffer_collector);
    }
  } else {
    // No fragment regions at all — fall back to buffer collector.
    for (const Any& state_any : buffer_collector.Encode()) {
      auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
      const std::string name = state_map["name"].cast<String>();
      acc->RegisterState(name, state_map["role"].cast<String>(),
                         state_map["scope"].cast<String>());
      if (buffer_collector.HasIntegerDType(name)) {
        acc->integer_states.insert(name);
      }
    }
  }

  // Apply reduction evidence to state roles.
  for (const auto& red : acc->reductions) {
    acc->reduction_targets.insert(red.target);
    if (buffer_collector.HasIntegerDType(red.target)) {
      acc->integer_states.insert(red.target);
    }
    // A reduction target is index_state only if it actually carries index information:
    // either it has integer dtype, or it is an integer arg-reduce target.  Non-integer
    // arg_reduce_targets (e.g. the value component of an arg-reduce pair) remain
    // reduction_accumulator — they participate in arg-reduce but don't carry indices.
    const bool is_index = acc->integer_states.count(red.target) ||
                          (acc->arg_reduce_targets.count(red.target) &&
                           buffer_collector.HasIntegerDType(red.target));
    const std::string role = is_index ? ToString(StateRole::kIndexState)
                                      : ToString(StateRole::kReductionAccumulator);
    acc->RegisterState(red.target, role, "");
  }

  // Refine roles: carry and selection.
  for (const std::string& carried : acc->loop_carried_states) {
    if (!acc->reduction_targets.count(carried)) {
      acc->RegisterState(carried, ToString(StateRole::kCarry), "");
    }
  }
  for (const std::string& name : acc->selection_targets) {
    if (acc->paired_selection_companions.count(name)) {
      acc->RegisterState(name, ToString(StateRole::kIndexState), "");
    } else if (!acc->integer_states.count(name)) {
      acc->RegisterState(name, ToString(StateRole::kSelectionState), "");
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 3: Emit witnesses from accumulated evidence
// ---------------------------------------------------------------------------

void EmitStateRoleWitnesses(const EvidenceAccumulator& acc,
                            Array<SemanticWitness>* witnesses) {
  for (const Any& state_any : acc.states) {
    auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
    auto role = ParseStateRole(state_map["role"].cast<String>());
    ICHECK(role) << "AnalyzeSemanticStructure encountered unsupported state role "
                 << state_map["role"].cast<String>();
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kState),
                                     state_map["name"].cast<String>(),
                                     ToString(WitnessFactAxis::kRole),
                                     MakeStateRolePayload(*role), Array<String>{},
                                     Array<String>{String("states")}));
  }
  for (const std::string& target : acc.arg_reduce_targets) {
    if (!acc.integer_states.count(target)) {
      continue;
    }
    const std::string evidence_source =
        LookupEvidenceSource(acc.arg_reduce_target_evidence_sources, target, "fragment_regions");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), target,
                                     ToString(WitnessFactAxis::kDerivesIndexFrom),
                                     MakeEmptyPayload(), Array<String>{},
                                     EvidenceSourceArray(evidence_source)));
  }
}

void EmitReductionUpdates(const EvidenceAccumulator& acc,
                          Array<Any>* updates,
                          Array<SemanticWitness>* witnesses) {
  for (const auto& red : acc.reductions) {
    Map<String, Any> entry;
    const std::string update_name = std::string("reduce_") + red.target;
    entry.Set("name", String(update_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kReduce)));
    entry.Set("target_state", String(red.target));
    entry.Set("reduce_kind", String(red.kind));
    if (auto it = acc.update_sources_by_target.find(red.target);
        it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    updates->push_back(entry);

    const std::string evidence_source =
        LookupEvidenceSource(acc.reduction_evidence_sources, red.target, "row_reductions");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kReduce),
                                     Array<String>{},
                                     EvidenceSourceArray(evidence_source)));
    if (auto it = acc.update_sources_by_target.find(red.target);
        it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, red.target, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
  }
}

void EmitSelectionUpdates(const EvidenceAccumulator& acc,
                          Array<Any>* updates,
                          Array<SemanticWitness>* witnesses) {
  for (const std::string& state_name : acc.selection_targets) {
    Map<String, Any> entry;
    entry.Set("name", String(std::string("select_") + state_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kSelect)));
    entry.Set("target_state", String(state_name));
    entry.Set("traits", Array<Any>{String("selected"), String("indexed")});
    if (auto it = acc.update_sources_by_target.find(state_name);
        it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    if (auto it = acc.paired_value_state_by_selection_target.find(state_name);
        it != acc.paired_value_state_by_selection_target.end()) {
      Array<Any> bindings;
      Map<String, Any> binding;
      binding.Set("kind", String(ToString(BindingKind::kPairedValueState)));
      binding.Set("symbol", String("state"));
      binding.Set("value_repr", String(it->second));
      bindings.push_back(binding);
      entry.Set("bindings", bindings);
    }
    updates->push_back(entry);
    const std::string update_name = std::string("select_") + state_name;
    const std::string selection_target_source =
        LookupEvidenceSource(acc.selection_target_evidence_sources, state_name,
                             "selection_targets");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kSelect),
                                     Array<String>{},
                                     EvidenceSourceArray(selection_target_source)));
    if (auto it = acc.update_sources_by_target.find(state_name);
        it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, state_name, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
    if (auto it = acc.paired_value_state_by_selection_target.find(state_name);
        it != acc.paired_value_state_by_selection_target.end()) {
      const std::string selection_pair_source =
          LookupEvidenceSource(acc.selection_pair_evidence_sources, state_name,
                               "selection_pairs");
      witnesses->push_back(
          MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                      ToString(WitnessFactAxis::kCompanion),
                      MakeRelationBindingPayload(BindingKind::kPairedValueState),
                      Array<String>{String(it->second)},
                      EvidenceSourceArray(selection_pair_source)));
    }
  }
}

void EmitRecurrenceUpdates(const EvidenceAccumulator& acc,
                           Array<Any>* updates,
                           Array<SemanticWitness>* witnesses) {
  for (const std::string& state_name : acc.loop_carried_states) {
    Map<String, Any> entry;
    entry.Set("name", String(std::string("recur_") + state_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kRecurrence)));
    entry.Set("target_state", String(state_name));
    entry.Set("traits", Array<Any>{String("carried"), String("staged")});
    if (auto it = acc.recurrence_edges_by_target.find(state_name);
        it != acc.recurrence_edges_by_target.end()) {
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
    } else if (auto it = acc.update_sources_by_target.find(state_name);
               it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    updates->push_back(entry);
    const std::string update_name = std::string("recur_") + state_name;
    const std::string loop_carried_source =
        LookupEvidenceSource(acc.loop_carried_state_evidence_sources, state_name,
                             "loop_carried_state");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kRecurrence),
                                     Array<String>{},
                                     EvidenceSourceArray(loop_carried_source)));
    Map<String, Any> ordering_payload;
    ordering_payload.Set("ordering", String("ordered"));
    const std::string recurrence_edge_source =
        LookupEvidenceSource(acc.recurrence_edge_evidence_sources, state_name,
                             "recurrence_edges");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kOrdering),
                                     std::move(ordering_payload), Array<String>{},
                                     EvidenceSourceArray(recurrence_edge_source)));
    if (auto it = acc.recurrence_edges_by_target.find(state_name);
        it != acc.recurrence_edges_by_target.end()) {
      Array<String> related_sources;
      for (const Any& source_any : it->second) {
        related_sources.push_back(tvm::Downcast<String>(source_any));
      }
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                                       ToString(WitnessFactAxis::kCarriedFrom),
                                       MakeRelationBindingPayload(
                                           BindingKind::kRecurrenceSourceState),
                                       std::move(related_sources),
                                       EvidenceSourceArray(recurrence_edge_source)));
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(recurrence_edge_source)));
    } else if (auto it = acc.update_sources_by_target.find(state_name);
               it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, state_name, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 4: Collect seeds and supplements from manifest
// ---------------------------------------------------------------------------

void CollectSeedsAndSupplements(const tir::PrimFunc& func,
                                Array<Any>* seeds,
                                Array<Any>* supplements,
                                Array<SemanticWitness>* witnesses) {
  std::unordered_set<std::string> seen_seed_markers;
  if (auto semantic_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticSeeds)) {
    if (auto capture = semantic_seeds.value().find("capture_kinds");
        capture != semantic_seeds.value().end()) {
      for (const Any& seed_any : tvm::Downcast<Array<Any>>((*capture).second)) {
        PushStringUnique(seeds, &seen_seed_markers, tvm::Downcast<String>(seed_any));
      }
    }
  }
  if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
    PushStringUnique(seeds, &seen_seed_markers, "explicit_op_manifest");

    Array<Any> manifest_op_kinds;
    std::unordered_set<std::string> seen_manifest_op_kinds;
    if (auto op_it = manifest.value().find(manifest_key::kOperations);
        op_it != manifest.value().end()) {
      for (const Any& op_any : tvm::Downcast<Array<Any>>((*op_it).second)) {
        auto op_map = tvm::Downcast<Map<String, Any>>(op_any);
        PushStringUnique(&manifest_op_kinds, &seen_manifest_op_kinds,
                         op_map["kind"].cast<String>());
      }
    }

    int ordered_region_count = 0;
    if (auto region_it = manifest.value().find(manifest_key::kOrderedRegions);
        region_it != manifest.value().end()) {
      for (const Any& region_any : tvm::Downcast<Array<Any>>((*region_it).second)) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        ++ordered_region_count;
        witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kBoundary),
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
    supplements->push_back(supplement);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------

tir::transform::Pass AnalyzeSemanticStructure() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }

    // Phase 1: domain skeleton.
    Array<Any> domain_axes;
    Array<Any> domain_traits;
    std::unordered_set<std::string> seen_traits;
    CollectDomainSkeleton(func, &domain_axes, &domain_traits, &seen_traits);

    // Phase 2: evidence ingestion.
    LocalBufferCollector buffer_collector;
    buffer_collector(func->body);
    EvidenceAccumulator acc;
    IngestAllEvidence(func, &acc, buffer_collector);

    // Phase 3: emit witnesses and updates.
    Array<SemanticWitness> witnesses;
    EmitStateRoleWitnesses(acc, &witnesses);

    Array<Any> updates;
    {
      Map<String, Any> entry;
      entry.Set("name", String("root_map"));
      entry.Set("kind", String(ToString(UpdateLawKind::kMap)));
      String root_target("");
      if (acc.states.size() == 1) {
        root_target = tvm::Downcast<Map<String, Any>>(acc.states[0])["name"].cast<String>();
      }
      entry.Set("target_state", root_target);
      updates.push_back(entry);
      witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), "root_map",
                                      ToString(WitnessFactAxis::kLawFamily),
                                      MakeUpdateLawFamilyPayload(UpdateLawKind::kMap),
                                      Array<String>{},
                                      Array<String>{String("semantic_structure")}));
    }
    EmitReductionUpdates(acc, &updates, &witnesses);
    if (!acc.selection_targets.empty()) {
      EmitSelectionUpdates(acc, &updates, &witnesses);
    }
    if (!acc.loop_carried_states.empty()) {
      EmitRecurrenceUpdates(acc, &updates, &witnesses);
    }

    // Phase 4: seeds and supplements.
    Array<Any> seeds;
    Array<Any> supplements;
    CollectSeedsAndSupplements(func, &seeds, &supplements, &witnesses);

    // Assemble structure.
    Map<String, Any> structure;
    structure.Set("domain_name", String("device_program"));
    structure.Set("domain_axes", domain_axes);
    structure.Set("domain_traits", domain_traits);
    structure.Set("states", acc.states);
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
