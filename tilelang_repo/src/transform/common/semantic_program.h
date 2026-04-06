/*!
 * \file semantic_program.h
 * \brief Stage 4 companion IR guardrail constants and minimal semantic IR objects.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/global_info.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {

namespace attr {
constexpr const char* kTLDevicePrograms = "tl.device_programs";
constexpr const char* kTLSemanticSeeds = "tl.semantic_seeds";
constexpr const char* kTLSemanticManifestSeeds = "tl.semantic_manifest_seeds";
constexpr const char* kTLSemanticManifest = "tl.semantic_manifest";
constexpr const char* kTLSemanticHardFreeze = "tl.semantic_hard_freeze";
constexpr const char* kTLSemanticStructure = "tl.semantic_structure";
constexpr const char* kTLSemanticWitnesses = "tl.semantic_witnesses";
constexpr const char* kTLSemanticProgram = "tl.semantic_program";
constexpr const char* kTLSpatialProgram = "tl.spatial_program";
constexpr const char* kTLTTProgram = "tl.tt_program";
constexpr const char* kTLCompanionInvalidationReason = "tl.companion_invalidation_reason";
}  // namespace attr

// Manifest schema keys — used as Map<String,Any> keys inside kTLSemanticManifest
// and structural_regions.  Centralised here so a typo becomes a compile error.
namespace manifest_key {
constexpr const char* kBuffers = "buffers";
constexpr const char* kOperations = "operations";
constexpr const char* kOrderedRegions = "ordered_regions";
constexpr const char* kAnchors = "anchors";
constexpr const char* kStructuralRegions = "structural_regions";
constexpr const char* kFragmentBuffers = "fragment_buffers";
constexpr const char* kSelectionTargets = "selection_targets";
constexpr const char* kSelectionPairs = "selection_pairs";
constexpr const char* kArgReduceTargets = "arg_reduce_targets";
constexpr const char* kUpdateSources = "update_sources";
constexpr const char* kLoopCarriedState = "loop_carried_state";
constexpr const char* kRecurrenceEdges = "recurrence_edges";
constexpr const char* kRowReductions = "row_reductions";
}  // namespace manifest_key

// Shared object-field keys used inside typed Map<String, Any> payloads.
namespace schema_key {
constexpr const char* kAnchor = "anchor";
constexpr const char* kBuffer = "buffer";
constexpr const char* kBuffers = "buffers";
constexpr const char* kCaptureStage = "capture_stage";
constexpr const char* kCompanionBuffer = "companion_buffer";
constexpr const char* kCompanionTarget = "companion_target";
constexpr const char* kDirection = "direction";
constexpr const char* kDType = "dtype";
constexpr const char* kIsInteger = "is_integer";
constexpr const char* kKind = "kind";
constexpr const char* kMidBuffer = "mid_buffer";
constexpr const char* kMidBufferRef = "mid_buffer_ref";
constexpr const char* kMidShape = "mid_shape";
constexpr const char* kName = "name";
constexpr const char* kOperations = "operations";
constexpr const char* kOrderedRegion = "ordered_region";
constexpr const char* kPayload = "payload";
constexpr const char* kScope = "scope";
constexpr const char* kShape = "shape";
constexpr const char* kSource = "source";
constexpr const char* kSourceBuffers = "source_buffers";
constexpr const char* kSources = "sources";
constexpr const char* kSourceStates = "source_states";
constexpr const char* kSrcBuffer = "src_buffer";
constexpr const char* kSrcBufferRef = "src_buffer_ref";
constexpr const char* kSrcShape = "src_shape";
constexpr const char* kTarget = "target";
constexpr const char* kTargetBuffer = "target_buffer";
constexpr const char* kValue = "value";
constexpr const char* kValueBuffer = "value_buffer";
constexpr const char* kValueTarget = "value_target";
constexpr const char* kDstBuffer = "dst_buffer";
constexpr const char* kDstBufferRef = "dst_buffer_ref";
constexpr const char* kDstShape = "dst_shape";
}  // namespace schema_key

class TIRAnchorNode : public Object {
 public:
  ffi::String kind;
  ffi::String value_repr;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRAnchorNode>()
        .def_ro("kind", &TIRAnchorNode::kind)
        .def_ro("value_repr", &TIRAnchorNode::value_repr);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TIRAnchor", TIRAnchorNode, Object);
};

class TIRAnchor : public ObjectRef {
 public:
  TVM_DLL TIRAnchor(ffi::String kind, ffi::String value_repr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TIRAnchor, ObjectRef, TIRAnchorNode);
};

class TIRValueBindingNode : public Object {
 public:
  ffi::String kind;
  ffi::String symbol;
  ffi::String value_repr;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRValueBindingNode>()
        .def_ro("kind", &TIRValueBindingNode::kind)
        .def_ro("symbol", &TIRValueBindingNode::symbol)
        .def_ro("value_repr", &TIRValueBindingNode::value_repr);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TIRValueBinding", TIRValueBindingNode, Object);
};

class TIRValueBinding : public ObjectRef {
 public:
  TVM_DLL TIRValueBinding(ffi::String kind, ffi::String symbol, ffi::String value_repr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TIRValueBinding, ObjectRef, TIRValueBindingNode);
};

class AccessMapNode : public Object {
 public:
  ffi::String kind;
  ffi::Array<PrimExpr> indices;
  ffi::Array<ffi::String> traits;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AccessMapNode>()
        .def_ro("kind", &AccessMapNode::kind)
        .def_ro("indices", &AccessMapNode::indices)
        .def_ro("traits", &AccessMapNode::traits);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AccessMap", AccessMapNode, Object);
};

class AccessMap : public ObjectRef {
 public:
  TVM_DLL AccessMap(ffi::String kind, ffi::Array<PrimExpr> indices,
                    ffi::Array<ffi::String> traits);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AccessMap, ObjectRef, AccessMapNode);
};

class UpdateLawNode : public Object {
 public:
  ffi::String kind;
  ffi::String target_state;
  ffi::Array<ffi::String> source_states;
  ffi::Array<AccessMap> access_maps;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UpdateLawNode>()
        .def_ro("kind", &UpdateLawNode::kind)
        .def_ro("target_state", &UpdateLawNode::target_state)
        .def_ro("source_states", &UpdateLawNode::source_states)
        .def_ro("access_maps", &UpdateLawNode::access_maps);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.UpdateLaw", UpdateLawNode, Object);
};

class UpdateLaw : public ObjectRef {
 public:
  TVM_DLL UpdateLaw(ffi::String kind, ffi::String target_state,
                    ffi::Array<ffi::String> source_states, ffi::Array<AccessMap> access_maps);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(UpdateLaw, ObjectRef, UpdateLawNode);
};

class DomainNode : public Object {
 public:
  ffi::String name;
  ffi::Array<ffi::String> axes;
  ffi::Array<ffi::String> traits;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DomainNode>()
        .def_ro("name", &DomainNode::name)
        .def_ro("axes", &DomainNode::axes)
        .def_ro("traits", &DomainNode::traits)
        .def_ro("anchors", &DomainNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Domain", DomainNode, Object);
};

class Domain : public ObjectRef {
 public:
  TVM_DLL Domain(ffi::String name, ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                 ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Domain, ObjectRef, DomainNode);
};

class StateNode : public Object {
 public:
  ffi::String name;
  ffi::String role;
  ffi::String storage_scope;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StateNode>()
        .def_ro("name", &StateNode::name)
        .def_ro("role", &StateNode::role)
        .def_ro("storage_scope", &StateNode::storage_scope)
        .def_ro("anchors", &StateNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.State", StateNode, Object);
};

class State : public ObjectRef {
 public:
  TVM_DLL State(ffi::String name, ffi::String role, ffi::String storage_scope,
                ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(State, ObjectRef, StateNode);
};

class UpdateNode : public Object {
 public:
  ffi::String name;
  ffi::String state_name;
  UpdateLaw law;
  ffi::Array<TIRAnchor> anchors;
  ffi::Array<TIRValueBinding> bindings;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UpdateNode>()
        .def_ro("name", &UpdateNode::name)
        .def_ro("state_name", &UpdateNode::state_name)
        .def_ro("law", &UpdateNode::law)
        .def_ro("anchors", &UpdateNode::anchors)
        .def_ro("bindings", &UpdateNode::bindings);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Update", UpdateNode, Object);
};

class Update : public ObjectRef {
 public:
  TVM_DLL Update(ffi::String name, ffi::String state_name, UpdateLaw law,
                 ffi::Array<TIRAnchor> anchors, ffi::Array<TIRValueBinding> bindings);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Update, ObjectRef, UpdateNode);
};

class SemanticSupplementNode : public Object {
 public:
  ffi::String kind;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemanticSupplementNode>()
        .def_ro("kind", &SemanticSupplementNode::kind)
        .def_ro("payload", &SemanticSupplementNode::payload);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SemanticSupplement", SemanticSupplementNode, Object);
};

class SemanticSupplement : public ObjectRef {
 public:
  TVM_DLL SemanticSupplement(ffi::String kind, ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SemanticSupplement, ObjectRef,
                                             SemanticSupplementNode);
};

class SemanticWitnessNode : public Object {
 public:
  ffi::String subject_kind;
  ffi::String subject_anchor_id;
  ffi::String fact_axis;
  ffi::Map<ffi::String, ffi::Any> fact_value;
  ffi::Array<ffi::String> related_anchor_ids;
  ffi::Array<ffi::String> evidence_sources;
  ffi::String canonicalization_point;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemanticWitnessNode>()
        .def_ro("subject_kind", &SemanticWitnessNode::subject_kind)
        .def_ro("subject_anchor_id", &SemanticWitnessNode::subject_anchor_id)
        .def_ro("fact_axis", &SemanticWitnessNode::fact_axis)
        .def_ro("fact_value", &SemanticWitnessNode::fact_value)
        .def_ro("related_anchor_ids", &SemanticWitnessNode::related_anchor_ids)
        .def_ro("evidence_sources", &SemanticWitnessNode::evidence_sources)
        .def_ro("canonicalization_point", &SemanticWitnessNode::canonicalization_point);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SemanticWitness", SemanticWitnessNode, Object);
};

class SemanticWitness : public ObjectRef {
 public:
  TVM_DLL SemanticWitness(ffi::String subject_kind, ffi::String subject_anchor_id,
                          ffi::String fact_axis, ffi::Map<ffi::String, ffi::Any> fact_value,
                          ffi::Array<ffi::String> related_anchor_ids,
                          ffi::Array<ffi::String> evidence_sources,
                          ffi::String canonicalization_point);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SemanticWitness, ObjectRef, SemanticWitnessNode);
};

class StateVersionNode : public Object {
 public:
  ffi::String name;
  ffi::String state_name;
  ffi::String producer_update;
  ffi::String kind;
  ffi::Array<ffi::String> source_versions;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StateVersionNode>()
        .def_ro("name", &StateVersionNode::name)
        .def_ro("state_name", &StateVersionNode::state_name)
        .def_ro("producer_update", &StateVersionNode::producer_update)
        .def_ro("kind", &StateVersionNode::kind)
        .def_ro("source_versions", &StateVersionNode::source_versions)
        .def_ro("anchors", &StateVersionNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.StateVersion", StateVersionNode, Object);
};

class StateVersion : public ObjectRef {
 public:
  TVM_DLL StateVersion(ffi::String name, ffi::String state_name, ffi::String producer_update,
                       ffi::String kind, ffi::Array<ffi::String> source_versions,
                       ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StateVersion, ObjectRef, StateVersionNode);
};

class StateDefNode : public Object {
 public:
  ffi::String name;
  ffi::String state_name;
  ffi::String version_name;
  ffi::String producer_update;
  ffi::String kind;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StateDefNode>()
        .def_ro("name", &StateDefNode::name)
        .def_ro("state_name", &StateDefNode::state_name)
        .def_ro("version_name", &StateDefNode::version_name)
        .def_ro("producer_update", &StateDefNode::producer_update)
        .def_ro("kind", &StateDefNode::kind)
        .def_ro("anchors", &StateDefNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.StateDef", StateDefNode, Object);
};

class StateDef : public ObjectRef {
 public:
  TVM_DLL StateDef(ffi::String name, ffi::String state_name, ffi::String version_name,
                   ffi::String producer_update, ffi::String kind,
                   ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StateDef, ObjectRef, StateDefNode);
};

class StateUseNode : public Object {
 public:
  ffi::String name;
  ffi::String consumer_update;
  ffi::String state_name;
  ffi::String version_name;
  ffi::String kind;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StateUseNode>()
        .def_ro("name", &StateUseNode::name)
        .def_ro("consumer_update", &StateUseNode::consumer_update)
        .def_ro("state_name", &StateUseNode::state_name)
        .def_ro("version_name", &StateUseNode::version_name)
        .def_ro("kind", &StateUseNode::kind)
        .def_ro("anchors", &StateUseNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.StateUse", StateUseNode, Object);
};

class StateUse : public ObjectRef {
 public:
  TVM_DLL StateUse(ffi::String name, ffi::String consumer_update, ffi::String state_name,
                   ffi::String version_name, ffi::String kind, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StateUse, ObjectRef, StateUseNode);
};

class StateJoinNode : public Object {
 public:
  ffi::String name;
  ffi::String state_name;
  ffi::String kind;
  ffi::Array<ffi::String> input_versions;
  ffi::String output_version;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StateJoinNode>()
        .def_ro("name", &StateJoinNode::name)
        .def_ro("state_name", &StateJoinNode::state_name)
        .def_ro("kind", &StateJoinNode::kind)
        .def_ro("input_versions", &StateJoinNode::input_versions)
        .def_ro("output_version", &StateJoinNode::output_version)
        .def_ro("anchors", &StateJoinNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.StateJoin", StateJoinNode, Object);
};

class StateJoin : public ObjectRef {
 public:
  TVM_DLL StateJoin(ffi::String name, ffi::String state_name, ffi::String kind,
                    ffi::Array<ffi::String> input_versions, ffi::String output_version,
                    ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StateJoin, ObjectRef, StateJoinNode);
};

class SemanticProgramNode : public Object {
 public:
  ffi::Array<Domain> domains;
  ffi::Array<State> states;
  ffi::Array<Update> updates;
  ffi::Array<SemanticSupplement> supplements;
  ffi::Array<ffi::String> seeds;
  ffi::Array<TIRAnchor> anchors;
  ffi::Array<StateVersion> state_versions;
  ffi::Array<StateDef> state_defs;
  ffi::Array<StateUse> state_uses;
  ffi::Array<StateJoin> state_joins;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemanticProgramNode>()
        .def_ro("domains", &SemanticProgramNode::domains)
        .def_ro("states", &SemanticProgramNode::states)
        .def_ro("updates", &SemanticProgramNode::updates)
        .def_ro("supplements", &SemanticProgramNode::supplements)
        .def_ro("seeds", &SemanticProgramNode::seeds)
        .def_ro("anchors", &SemanticProgramNode::anchors)
        .def_ro("state_versions", &SemanticProgramNode::state_versions)
        .def_ro("state_defs", &SemanticProgramNode::state_defs)
        .def_ro("state_uses", &SemanticProgramNode::state_uses)
        .def_ro("state_joins", &SemanticProgramNode::state_joins);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SemanticProgram", SemanticProgramNode, Object);
};

class SemanticProgram : public ObjectRef {
 public:
  TVM_DLL SemanticProgram(ffi::Array<Domain> domains, ffi::Array<State> states,
                          ffi::Array<Update> updates,
                          ffi::Array<SemanticSupplement> supplements,
                          ffi::Array<ffi::String> seeds, ffi::Array<TIRAnchor> anchors,
                          ffi::Array<StateVersion> state_versions,
                          ffi::Array<StateDef> state_defs,
                          ffi::Array<StateUse> state_uses,
                          ffi::Array<StateJoin> state_joins);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SemanticProgram, ObjectRef, SemanticProgramNode);
};

class TLDeviceProgramInfoNode : public GlobalInfoNode {
 public:
  ffi::String root_symbol;
  ffi::Array<ffi::String> member_funcs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TLDeviceProgramInfoNode>()
        .def_ro("root_symbol", &TLDeviceProgramInfoNode::root_symbol)
        .def_ro("member_funcs", &TLDeviceProgramInfoNode::member_funcs);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TLDeviceProgramInfo", TLDeviceProgramInfoNode,
                                    GlobalInfoNode);
};

class TLDeviceProgramInfo : public GlobalInfo {
 public:
  TVM_DLL TLDeviceProgramInfo(ffi::String root_symbol, ffi::Array<ffi::String> member_funcs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TLDeviceProgramInfo, GlobalInfo,
                                             TLDeviceProgramInfoNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_
