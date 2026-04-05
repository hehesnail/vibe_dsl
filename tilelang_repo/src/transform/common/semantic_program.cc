/*!
 * \file semantic_program.cc
 * \brief Minimal Stage 4 semantic IR object constructors and reflection.
 */

#include "semantic_program.h"

namespace tvm {
namespace tl {

TIRAnchor::TIRAnchor(ffi::String kind, ffi::String value_repr) {
  auto n = ffi::make_object<TIRAnchorNode>();
  n->kind = std::move(kind);
  n->value_repr = std::move(value_repr);
  data_ = std::move(n);
}

TIRValueBinding::TIRValueBinding(ffi::String kind, ffi::String symbol, ffi::String value_repr) {
  auto n = ffi::make_object<TIRValueBindingNode>();
  n->kind = std::move(kind);
  n->symbol = std::move(symbol);
  n->value_repr = std::move(value_repr);
  data_ = std::move(n);
}

AccessMap::AccessMap(ffi::String kind, ffi::Array<PrimExpr> indices, ffi::Array<ffi::String> traits) {
  auto n = ffi::make_object<AccessMapNode>();
  n->kind = std::move(kind);
  n->indices = std::move(indices);
  n->traits = std::move(traits);
  data_ = std::move(n);
}

UpdateLaw::UpdateLaw(ffi::String kind, ffi::String target_state,
                     ffi::Array<ffi::String> source_states, ffi::Array<AccessMap> access_maps) {
  auto n = ffi::make_object<UpdateLawNode>();
  n->kind = std::move(kind);
  n->target_state = std::move(target_state);
  n->source_states = std::move(source_states);
  n->access_maps = std::move(access_maps);
  data_ = std::move(n);
}

Domain::Domain(ffi::String name, ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
               ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<DomainNode>();
  n->name = std::move(name);
  n->axes = std::move(axes);
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

State::State(ffi::String name, ffi::String role, ffi::String storage_scope,
             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<StateNode>();
  n->name = std::move(name);
  n->role = std::move(role);
  n->storage_scope = std::move(storage_scope);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

Update::Update(ffi::String name, ffi::String state_name, UpdateLaw law,
               ffi::Array<TIRAnchor> anchors, ffi::Array<TIRValueBinding> bindings) {
  auto n = ffi::make_object<UpdateNode>();
  n->name = std::move(name);
  n->state_name = std::move(state_name);
  n->law = std::move(law);
  n->anchors = std::move(anchors);
  n->bindings = std::move(bindings);
  data_ = std::move(n);
}

SemanticSupplement::SemanticSupplement(ffi::String kind, ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<SemanticSupplementNode>();
  n->kind = std::move(kind);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

SemanticWitness::SemanticWitness(ffi::String subject_kind, ffi::String subject_anchor_id,
                                 ffi::String fact_axis, ffi::Map<ffi::String, ffi::Any> fact_value,
                                 ffi::Array<ffi::String> related_anchor_ids,
                                 ffi::Array<ffi::String> evidence_sources,
                                 ffi::String canonicalization_point) {
  auto n = ffi::make_object<SemanticWitnessNode>();
  n->subject_kind = std::move(subject_kind);
  n->subject_anchor_id = std::move(subject_anchor_id);
  n->fact_axis = std::move(fact_axis);
  n->fact_value = std::move(fact_value);
  n->related_anchor_ids = std::move(related_anchor_ids);
  n->evidence_sources = std::move(evidence_sources);
  n->canonicalization_point = std::move(canonicalization_point);
  data_ = std::move(n);
}

StateVersion::StateVersion(ffi::String name, ffi::String state_name, ffi::String producer_update,
                           ffi::String kind, ffi::Array<ffi::String> source_versions,
                           ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<StateVersionNode>();
  n->name = std::move(name);
  n->state_name = std::move(state_name);
  n->producer_update = std::move(producer_update);
  n->kind = std::move(kind);
  n->source_versions = std::move(source_versions);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

StateDef::StateDef(ffi::String name, ffi::String state_name, ffi::String version_name,
                   ffi::String producer_update, ffi::String kind,
                   ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<StateDefNode>();
  n->name = std::move(name);
  n->state_name = std::move(state_name);
  n->version_name = std::move(version_name);
  n->producer_update = std::move(producer_update);
  n->kind = std::move(kind);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

StateUse::StateUse(ffi::String name, ffi::String consumer_update, ffi::String state_name,
                   ffi::String version_name, ffi::String kind,
                   ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<StateUseNode>();
  n->name = std::move(name);
  n->consumer_update = std::move(consumer_update);
  n->state_name = std::move(state_name);
  n->version_name = std::move(version_name);
  n->kind = std::move(kind);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

StateJoin::StateJoin(ffi::String name, ffi::String state_name, ffi::String kind,
                     ffi::Array<ffi::String> input_versions, ffi::String output_version,
                     ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<StateJoinNode>();
  n->name = std::move(name);
  n->state_name = std::move(state_name);
  n->kind = std::move(kind);
  n->input_versions = std::move(input_versions);
  n->output_version = std::move(output_version);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SemanticProgram::SemanticProgram(ffi::Array<Domain> domains, ffi::Array<State> states,
                                 ffi::Array<Update> updates,
                                 ffi::Array<SemanticSupplement> supplements,
                                 ffi::Array<ffi::String> seeds, ffi::Array<TIRAnchor> anchors,
                                 ffi::Array<StateVersion> state_versions,
                                 ffi::Array<StateDef> state_defs,
                                 ffi::Array<StateUse> state_uses,
                                 ffi::Array<StateJoin> state_joins) {
  auto n = ffi::make_object<SemanticProgramNode>();
  n->domains = std::move(domains);
  n->states = std::move(states);
  n->updates = std::move(updates);
  n->supplements = std::move(supplements);
  n->seeds = std::move(seeds);
  n->anchors = std::move(anchors);
  n->state_versions = std::move(state_versions);
  n->state_defs = std::move(state_defs);
  n->state_uses = std::move(state_uses);
  n->state_joins = std::move(state_joins);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRAnchorNode::RegisterReflection();
  TIRValueBindingNode::RegisterReflection();
  AccessMapNode::RegisterReflection();
  UpdateLawNode::RegisterReflection();
  DomainNode::RegisterReflection();
  StateNode::RegisterReflection();
  UpdateNode::RegisterReflection();
  SemanticSupplementNode::RegisterReflection();
  SemanticWitnessNode::RegisterReflection();
  StateVersionNode::RegisterReflection();
  StateDefNode::RegisterReflection();
  StateUseNode::RegisterReflection();
  StateJoinNode::RegisterReflection();
  SemanticProgramNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TIRAnchor",
                        [](ffi::String kind, ffi::String value_repr) {
                          return TIRAnchor(std::move(kind), std::move(value_repr));
                        });
  refl::GlobalDef().def("tl.TIRValueBinding",
                        [](ffi::String kind, ffi::String symbol, ffi::String value_repr) {
                          return TIRValueBinding(std::move(kind), std::move(symbol),
                                                 std::move(value_repr));
                        });
  refl::GlobalDef().def("tl.AccessMap",
                        [](ffi::String kind, ffi::Array<PrimExpr> indices,
                           ffi::Array<ffi::String> traits) {
                          return AccessMap(std::move(kind), std::move(indices), std::move(traits));
                        });
  refl::GlobalDef().def("tl.UpdateLaw",
                        [](ffi::String kind, ffi::String target_state,
                           ffi::Array<ffi::String> source_states,
                           ffi::Array<AccessMap> access_maps) {
                          return UpdateLaw(std::move(kind), std::move(target_state),
                                           std::move(source_states), std::move(access_maps));
                        });
  refl::GlobalDef().def("tl.Domain",
                        [](ffi::String name, ffi::Array<ffi::String> axes,
                           ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
                          return Domain(std::move(name), std::move(axes), std::move(traits),
                                        std::move(anchors));
                        });
  refl::GlobalDef().def("tl.State",
                        [](ffi::String name, ffi::String role, ffi::String storage_scope,
                           ffi::Array<TIRAnchor> anchors) {
                          return State(std::move(name), std::move(role), std::move(storage_scope),
                                       std::move(anchors));
                        });
  refl::GlobalDef().def("tl.Update",
                        [](ffi::String name, ffi::String state_name, UpdateLaw law,
                           ffi::Array<TIRAnchor> anchors,
                           ffi::Array<TIRValueBinding> bindings) {
                          return Update(std::move(name), std::move(state_name), std::move(law),
                                        std::move(anchors), std::move(bindings));
                        });
  refl::GlobalDef().def("tl.SemanticSupplement",
                        [](ffi::String kind, ffi::Map<ffi::String, ffi::Any> payload) {
                          return SemanticSupplement(std::move(kind), std::move(payload));
                        });
  refl::GlobalDef().def("tl.SemanticWitness",
                        [](ffi::String subject_kind, ffi::String subject_anchor_id,
                           ffi::String fact_axis, ffi::Map<ffi::String, ffi::Any> fact_value,
                           ffi::Array<ffi::String> related_anchor_ids,
                           ffi::Array<ffi::String> evidence_sources,
                           ffi::String canonicalization_point) {
                          return SemanticWitness(std::move(subject_kind),
                                                 std::move(subject_anchor_id),
                                                 std::move(fact_axis), std::move(fact_value),
                                                 std::move(related_anchor_ids),
                                                 std::move(evidence_sources),
                                                 std::move(canonicalization_point));
                        });
  refl::GlobalDef().def("tl.StateVersion",
                        [](ffi::String name, ffi::String state_name, ffi::String producer_update,
                           ffi::String kind, ffi::Array<ffi::String> source_versions,
                           ffi::Array<TIRAnchor> anchors) {
                          return StateVersion(std::move(name), std::move(state_name),
                                              std::move(producer_update), std::move(kind),
                                              std::move(source_versions), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.StateDef",
                        [](ffi::String name, ffi::String state_name, ffi::String version_name,
                           ffi::String producer_update, ffi::String kind,
                           ffi::Array<TIRAnchor> anchors) {
                          return StateDef(std::move(name), std::move(state_name),
                                          std::move(version_name), std::move(producer_update),
                                          std::move(kind), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.StateUse",
                        [](ffi::String name, ffi::String consumer_update, ffi::String state_name,
                           ffi::String version_name, ffi::String kind,
                           ffi::Array<TIRAnchor> anchors) {
                          return StateUse(std::move(name), std::move(consumer_update),
                                          std::move(state_name), std::move(version_name),
                                          std::move(kind), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.StateJoin",
                        [](ffi::String name, ffi::String state_name, ffi::String kind,
                           ffi::Array<ffi::String> input_versions, ffi::String output_version,
                           ffi::Array<TIRAnchor> anchors) {
                          return StateJoin(std::move(name), std::move(state_name),
                                           std::move(kind), std::move(input_versions),
                                           std::move(output_version), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SemanticProgram",
                        [](ffi::Array<Domain> domains, ffi::Array<State> states,
                           ffi::Array<Update> updates,
                           ffi::Array<SemanticSupplement> supplements,
                           ffi::Array<ffi::String> seeds, ffi::Array<TIRAnchor> anchors,
                           ffi::Array<StateVersion> state_versions,
                           ffi::Array<StateDef> state_defs,
                           ffi::Array<StateUse> state_uses,
                           ffi::Array<StateJoin> state_joins) {
                          return SemanticProgram(std::move(domains), std::move(states),
                                                 std::move(updates), std::move(supplements),
                                                 std::move(seeds), std::move(anchors),
                                                 std::move(state_versions), std::move(state_defs),
                                                 std::move(state_uses), std::move(state_joins));
                        });
}

}  // namespace tl
}  // namespace tvm
