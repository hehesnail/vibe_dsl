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

class ExecutionUnitNode : public Object {
 public:
  ffi::String name;
  ffi::String formation_basis;
  ffi::String unit_role;
  ffi::Array<Integer> stmt_indices;
  ffi::Array<ffi::String> read_buffers;
  ffi::Array<ffi::String> write_buffers;
  ffi::Array<ffi::String> traits;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecutionUnitNode>()
        .def_ro("name", &ExecutionUnitNode::name)
        .def_ro("formation_basis", &ExecutionUnitNode::formation_basis)
        .def_ro("unit_role", &ExecutionUnitNode::unit_role)
        .def_ro("stmt_indices", &ExecutionUnitNode::stmt_indices)
        .def_ro("read_buffers", &ExecutionUnitNode::read_buffers)
        .def_ro("write_buffers", &ExecutionUnitNode::write_buffers)
        .def_ro("traits", &ExecutionUnitNode::traits)
        .def_ro("anchors", &ExecutionUnitNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ExecutionUnit", ExecutionUnitNode, Object);
};

class ExecutionUnit : public ObjectRef {
 public:
  TVM_DLL ExecutionUnit(ffi::String name, ffi::String formation_basis, ffi::String unit_role,
                        ffi::Array<Integer> stmt_indices, ffi::Array<ffi::String> read_buffers,
                        ffi::Array<ffi::String> write_buffers, ffi::Array<ffi::String> traits,
                        ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExecutionUnit, ObjectRef, ExecutionUnitNode);
};

class DataflowEdgeNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String producer_unit;
  ffi::String consumer_unit;
  int64_t producer_unit_index = -1;
  int64_t consumer_unit_index = -1;
  ffi::String subject;
  bool crosses_phase = false;
  ffi::Array<ffi::String> traits;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowEdgeNode>()
        .def_ro("name", &DataflowEdgeNode::name)
        .def_ro("kind", &DataflowEdgeNode::kind)
        .def_ro("producer_unit", &DataflowEdgeNode::producer_unit)
        .def_ro("consumer_unit", &DataflowEdgeNode::consumer_unit)
        .def_ro("producer_unit_index", &DataflowEdgeNode::producer_unit_index)
        .def_ro("consumer_unit_index", &DataflowEdgeNode::consumer_unit_index)
        .def_ro("subject", &DataflowEdgeNode::subject)
        .def_ro("crosses_phase", &DataflowEdgeNode::crosses_phase)
        .def_ro("traits", &DataflowEdgeNode::traits)
        .def_ro("anchors", &DataflowEdgeNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DataflowEdge", DataflowEdgeNode, Object);
};

class DataflowEdge : public ObjectRef {
 public:
  TVM_DLL DataflowEdge(ffi::String name, ffi::String kind, ffi::String producer_unit,
                       ffi::String consumer_unit, int64_t producer_unit_index,
                       int64_t consumer_unit_index, ffi::String subject, bool crosses_phase,
                       ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DataflowEdge, ObjectRef, DataflowEdgeNode);
};

class LayoutSpecNode : public Object {
 public:
  ffi::String name;
  ffi::String subject;
  ffi::String scope;
  ffi::String distribution_kind;
  ffi::Array<ffi::String> unit_names;
  ffi::Array<Integer> unit_indices;
  ffi::Array<ffi::String> virtual_device_axes;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutSpecNode>()
        .def_ro("name", &LayoutSpecNode::name)
        .def_ro("subject", &LayoutSpecNode::subject)
        .def_ro("scope", &LayoutSpecNode::scope)
        .def_ro("distribution_kind", &LayoutSpecNode::distribution_kind)
        .def_ro("unit_names", &LayoutSpecNode::unit_names)
        .def_ro("unit_indices", &LayoutSpecNode::unit_indices)
        .def_ro("virtual_device_axes", &LayoutSpecNode::virtual_device_axes)
        .def_ro("anchors", &LayoutSpecNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.LayoutSpec", LayoutSpecNode, Object);
};

class LayoutSpec : public ObjectRef {
 public:
  TVM_DLL LayoutSpec(ffi::String name, ffi::String subject, ffi::String scope,
                     ffi::String distribution_kind, ffi::Array<ffi::String> unit_names,
                     ffi::Array<Integer> unit_indices,
                     ffi::Array<ffi::String> virtual_device_axes,
                     ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LayoutSpec, ObjectRef, LayoutSpecNode);
};

class PhasePlanNode : public Object {
 public:
  ffi::String name;
  int64_t phase_index = 0;
  ffi::Array<ffi::String> unit_names;
  ffi::Array<Integer> unit_indices;
  ffi::Array<ffi::String> ingress_edge_names;
  ffi::Array<Integer> ingress_edge_indices;
  ffi::Array<ffi::String> egress_edge_names;
  ffi::Array<Integer> egress_edge_indices;
  ffi::Array<ffi::String> boundary_subjects;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PhasePlanNode>()
        .def_ro("name", &PhasePlanNode::name)
        .def_ro("phase_index", &PhasePlanNode::phase_index)
        .def_ro("unit_names", &PhasePlanNode::unit_names)
        .def_ro("unit_indices", &PhasePlanNode::unit_indices)
        .def_ro("ingress_edge_names", &PhasePlanNode::ingress_edge_names)
        .def_ro("ingress_edge_indices", &PhasePlanNode::ingress_edge_indices)
        .def_ro("egress_edge_names", &PhasePlanNode::egress_edge_names)
        .def_ro("egress_edge_indices", &PhasePlanNode::egress_edge_indices)
        .def_ro("boundary_subjects", &PhasePlanNode::boundary_subjects)
        .def_ro("anchors", &PhasePlanNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.PhasePlan", PhasePlanNode, Object);
};

class PhasePlan : public ObjectRef {
 public:
  TVM_DLL PhasePlan(ffi::String name, int64_t phase_index, ffi::Array<ffi::String> unit_names,
                    ffi::Array<Integer> unit_indices,
                    ffi::Array<ffi::String> ingress_edge_names,
                    ffi::Array<Integer> ingress_edge_indices,
                    ffi::Array<ffi::String> egress_edge_names,
                    ffi::Array<Integer> egress_edge_indices,
                    ffi::Array<ffi::String> boundary_subjects,
                    ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PhasePlan, ObjectRef, PhasePlanNode);
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
  ffi::Array<ExecutionUnit> execution_units;
  ffi::Array<DataflowEdge> dataflow_edges;
  ffi::Array<LayoutSpec> layout_specs;
  ffi::Array<PhasePlan> phase_plans;
  ffi::Array<ExecutionClosure> closures;
  ffi::Array<ClosureBoundary> boundaries;
  ValidatedHintSet validated_hints;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialPlanNode>()
        .def_ro("member_func", &SpatialPlanNode::member_func)
        .def_ro("execution_units", &SpatialPlanNode::execution_units)
        .def_ro("dataflow_edges", &SpatialPlanNode::dataflow_edges)
        .def_ro("layout_specs", &SpatialPlanNode::layout_specs)
        .def_ro("phase_plans", &SpatialPlanNode::phase_plans)
        .def_ro("closures", &SpatialPlanNode::closures)
        .def_ro("boundaries", &SpatialPlanNode::boundaries)
        .def_ro("validated_hints", &SpatialPlanNode::validated_hints)
        .def_ro("anchors", &SpatialPlanNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialPlan", SpatialPlanNode, Object);
};

class SpatialPlan : public ObjectRef {
 public:
  TVM_DLL SpatialPlan(ffi::String member_func, ffi::Array<ExecutionUnit> execution_units,
                      ffi::Array<DataflowEdge> dataflow_edges,
                      ffi::Array<LayoutSpec> layout_specs,
                      ffi::Array<PhasePlan> phase_plans,
                      ValidatedHintSet validated_hints,
                      ffi::Array<ExecutionClosure> closures,
                      ffi::Array<ClosureBoundary> boundaries,
                      ffi::Array<TIRAnchor> anchors);
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
