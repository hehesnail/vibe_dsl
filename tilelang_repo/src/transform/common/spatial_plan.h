/*!
 * \file spatial_plan.h
 * \brief Task 1 SpatialPlan companion objects derived from normalized TIR.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_PLAN_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_PLAN_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_info.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

class ExecutionClosureNode : public Object {
 public:
  ffi::String name;
  ffi::String closure_basis;
  ffi::String execution_role;
  ffi::Array<Integer> stmt_indices;
  ffi::Array<ffi::String> read_buffers;
  ffi::Array<ffi::String> write_buffers;
  ffi::Array<ffi::String> cut_frontiers;
  ffi::Array<ffi::String> traits;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecutionClosureNode>()
        .def_ro("name", &ExecutionClosureNode::name)
        .def_ro("closure_basis", &ExecutionClosureNode::closure_basis)
        .def_ro("execution_role", &ExecutionClosureNode::execution_role)
        .def_ro("stmt_indices", &ExecutionClosureNode::stmt_indices)
        .def_ro("read_buffers", &ExecutionClosureNode::read_buffers)
        .def_ro("write_buffers", &ExecutionClosureNode::write_buffers)
        .def_ro("cut_frontiers", &ExecutionClosureNode::cut_frontiers)
        .def_ro("traits", &ExecutionClosureNode::traits)
        .def_ro("anchors", &ExecutionClosureNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ExecutionClosure", ExecutionClosureNode, Object);
};

class ExecutionClosure : public ObjectRef {
 public:
  TVM_DLL ExecutionClosure(ffi::String name, ffi::String closure_basis,
                           ffi::String execution_role, ffi::Array<Integer> stmt_indices,
                           ffi::Array<ffi::String> read_buffers,
                           ffi::Array<ffi::String> write_buffers,
                           ffi::Array<ffi::String> cut_frontiers,
                           ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExecutionClosure, ObjectRef,
                                             ExecutionClosureNode);
};

class ClosureBoundaryNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String source_closure;
  ffi::String target_closure;
  int64_t source_closure_index = -1;
  int64_t target_closure_index = -1;
  ffi::String subject;
  ffi::Array<ffi::String> traits;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ClosureBoundaryNode>()
        .def_ro("name", &ClosureBoundaryNode::name)
        .def_ro("kind", &ClosureBoundaryNode::kind)
        .def_ro("source_closure", &ClosureBoundaryNode::source_closure)
        .def_ro("target_closure", &ClosureBoundaryNode::target_closure)
        .def_ro("source_closure_index", &ClosureBoundaryNode::source_closure_index)
        .def_ro("target_closure_index", &ClosureBoundaryNode::target_closure_index)
        .def_ro("subject", &ClosureBoundaryNode::subject)
        .def_ro("traits", &ClosureBoundaryNode::traits)
        .def_ro("anchors", &ClosureBoundaryNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ClosureBoundary", ClosureBoundaryNode, Object);
};

class ClosureBoundary : public ObjectRef {
 public:
  TVM_DLL ClosureBoundary(ffi::String name, ffi::String kind, ffi::String source_closure,
                          ffi::String target_closure, int64_t source_closure_index,
                          int64_t target_closure_index, ffi::String subject,
                          ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ClosureBoundary, ObjectRef, ClosureBoundaryNode);
};

class ValidatedHintSetNode : public Object {
 public:
  ffi::Array<ffi::String> accepted_hints;
  ffi::Array<ffi::String> rejected_hints;
  ffi::Map<ffi::String, ffi::Any> diagnostics;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ValidatedHintSetNode>()
        .def_ro("accepted_hints", &ValidatedHintSetNode::accepted_hints)
        .def_ro("rejected_hints", &ValidatedHintSetNode::rejected_hints)
        .def_ro("diagnostics", &ValidatedHintSetNode::diagnostics)
        .def_ro("anchors", &ValidatedHintSetNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ValidatedHintSet", ValidatedHintSetNode, Object);
};

class ValidatedHintSet : public ObjectRef {
 public:
  TVM_DLL ValidatedHintSet(ffi::Array<ffi::String> accepted_hints,
                           ffi::Array<ffi::String> rejected_hints,
                           ffi::Map<ffi::String, ffi::Any> diagnostics,
                           ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ValidatedHintSet, ObjectRef, ValidatedHintSetNode);
};

class SpatialStructureFactsNode : public Object {
 public:
  ffi::String member_func;
  ffi::Array<ExecutionClosure> closure_candidates;
  ffi::Array<ClosureBoundary> boundary_candidates;
  ValidatedHintSet validated_hints;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialStructureFactsNode>()
        .def_ro("member_func", &SpatialStructureFactsNode::member_func)
        .def_ro("closure_candidates", &SpatialStructureFactsNode::closure_candidates)
        .def_ro("boundary_candidates", &SpatialStructureFactsNode::boundary_candidates)
        .def_ro("validated_hints", &SpatialStructureFactsNode::validated_hints)
        .def_ro("anchors", &SpatialStructureFactsNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialStructureFacts", SpatialStructureFactsNode, Object);
};

class SpatialStructureFacts : public ObjectRef {
 public:
  TVM_DLL SpatialStructureFacts(ffi::String member_func,
                                ffi::Array<ExecutionClosure> closure_candidates,
                                ffi::Array<ClosureBoundary> boundary_candidates,
                                ValidatedHintSet validated_hints,
                                ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialStructureFacts, ObjectRef,
                                             SpatialStructureFactsNode);
};

class SpatialPlanNode : public Object {
 public:
  ffi::String member_func;
  ffi::Array<ExecutionClosure> closures;
  ffi::Array<ClosureBoundary> boundaries;
  ValidatedHintSet validated_hints;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialPlanNode>()
        .def_ro("member_func", &SpatialPlanNode::member_func)
        .def_ro("closures", &SpatialPlanNode::closures)
        .def_ro("boundaries", &SpatialPlanNode::boundaries)
        .def_ro("validated_hints", &SpatialPlanNode::validated_hints)
        .def_ro("anchors", &SpatialPlanNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialPlan", SpatialPlanNode, Object);
};

class SpatialPlan : public ObjectRef {
 public:
  TVM_DLL SpatialPlan(ffi::String member_func, ffi::Array<ExecutionClosure> closures,
                      ffi::Array<ClosureBoundary> boundaries,
                      ValidatedHintSet validated_hints, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialPlan, ObjectRef, SpatialPlanNode);
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

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_PLAN_H_
