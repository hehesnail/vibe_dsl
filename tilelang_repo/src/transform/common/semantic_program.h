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
constexpr const char* kTLSemanticHardFreeze = "tl.semantic_hard_freeze";
constexpr const char* kTLSemanticStructure = "tl.semantic_structure";
constexpr const char* kTLSemanticProgram = "tl.semantic_program";
constexpr const char* kTLSpatialProgram = "tl.spatial_program";
constexpr const char* kTLTTProgram = "tl.tt_program";
constexpr const char* kTLCompanionInvalidationReason = "tl.companion_invalidation_reason";
}  // namespace attr

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

class SemanticProgramNode : public Object {
 public:
  ffi::Array<Domain> domains;
  ffi::Array<State> states;
  ffi::Array<Update> updates;
  ffi::Array<SemanticSupplement> supplements;
  ffi::Array<ffi::String> seeds;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemanticProgramNode>()
        .def_ro("domains", &SemanticProgramNode::domains)
        .def_ro("states", &SemanticProgramNode::states)
        .def_ro("updates", &SemanticProgramNode::updates)
        .def_ro("supplements", &SemanticProgramNode::supplements)
        .def_ro("seeds", &SemanticProgramNode::seeds)
        .def_ro("anchors", &SemanticProgramNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SemanticProgram", SemanticProgramNode, Object);
};

class SemanticProgram : public ObjectRef {
 public:
  TVM_DLL SemanticProgram(ffi::Array<Domain> domains, ffi::Array<State> states,
                          ffi::Array<Update> updates,
                          ffi::Array<SemanticSupplement> supplements,
                          ffi::Array<ffi::String> seeds, ffi::Array<TIRAnchor> anchors);
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
