/*!
 * \file spatial_plan.cc
 * \brief Task 1 SpatialPlan companion objects derived from normalized TIR.
 */

#include "spatial_plan.h"

namespace tvm {
namespace tl {

ExecutionClosure::ExecutionClosure(ffi::String name, ffi::String closure_basis,
                                   ffi::String execution_role, ffi::Array<Integer> stmt_indices,
                                   ffi::Array<ffi::String> read_buffers,
                                   ffi::Array<ffi::String> write_buffers,
                                   ffi::Array<ffi::String> cut_frontiers,
                                   ffi::Array<ffi::String> traits,
                                   ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ExecutionClosureNode>();
  n->name = std::move(name);
  n->closure_basis = std::move(closure_basis);
  n->execution_role = std::move(execution_role);
  n->stmt_indices = std::move(stmt_indices);
  n->read_buffers = std::move(read_buffers);
  n->write_buffers = std::move(write_buffers);
  n->cut_frontiers = std::move(cut_frontiers);
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

ClosureBoundary::ClosureBoundary(ffi::String name, ffi::String kind,
                                 ffi::String source_closure, ffi::String target_closure,
                                 int64_t source_closure_index, int64_t target_closure_index,
                                 ffi::String subject, ffi::Array<ffi::String> traits,
                                 ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ClosureBoundaryNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_closure = std::move(source_closure);
  n->target_closure = std::move(target_closure);
  n->source_closure_index = source_closure_index;
  n->target_closure_index = target_closure_index;
  n->subject = std::move(subject);
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

ValidatedHintSet::ValidatedHintSet(ffi::Array<ffi::String> accepted_hints,
                                   ffi::Array<ffi::String> rejected_hints,
                                   ffi::Map<ffi::String, ffi::Any> diagnostics,
                                   ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ValidatedHintSetNode>();
  n->accepted_hints = std::move(accepted_hints);
  n->rejected_hints = std::move(rejected_hints);
  n->diagnostics = std::move(diagnostics);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

ExecutionUnit::ExecutionUnit(ffi::String name, ffi::String formation_basis,
                             ffi::String unit_role, ffi::Array<Integer> stmt_indices,
                             ffi::Array<ffi::String> read_buffers,
                             ffi::Array<ffi::String> write_buffers,
                             ffi::Array<ffi::String> traits,
                             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ExecutionUnitNode>();
  n->name = std::move(name);
  n->formation_basis = std::move(formation_basis);
  n->unit_role = std::move(unit_role);
  n->stmt_indices = std::move(stmt_indices);
  n->read_buffers = std::move(read_buffers);
  n->write_buffers = std::move(write_buffers);
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

DataflowEdge::DataflowEdge(ffi::String name, ffi::String kind, ffi::String producer_unit,
                           ffi::String consumer_unit, int64_t producer_unit_index,
                           int64_t consumer_unit_index, ffi::String subject,
                           bool crosses_phase, ffi::Array<ffi::String> traits,
                           ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<DataflowEdgeNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->producer_unit = std::move(producer_unit);
  n->consumer_unit = std::move(consumer_unit);
  n->producer_unit_index = producer_unit_index;
  n->consumer_unit_index = consumer_unit_index;
  n->subject = std::move(subject);
  n->crosses_phase = crosses_phase;
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

LayoutSpec::LayoutSpec(ffi::String name, ffi::String subject, ffi::String scope,
                       ffi::String distribution_kind, ffi::Array<ffi::String> unit_names,
                       ffi::Array<Integer> unit_indices,
                       ffi::Array<ffi::String> virtual_device_axes,
                       ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<LayoutSpecNode>();
  n->name = std::move(name);
  n->subject = std::move(subject);
  n->scope = std::move(scope);
  n->distribution_kind = std::move(distribution_kind);
  n->unit_names = std::move(unit_names);
  n->unit_indices = std::move(unit_indices);
  n->virtual_device_axes = std::move(virtual_device_axes);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

PhasePlan::PhasePlan(ffi::String name, int64_t phase_index, ffi::Array<ffi::String> unit_names,
                     ffi::Array<Integer> unit_indices,
                     ffi::Array<ffi::String> ingress_edge_names,
                     ffi::Array<Integer> ingress_edge_indices,
                     ffi::Array<ffi::String> egress_edge_names,
                     ffi::Array<Integer> egress_edge_indices,
                     ffi::Array<ffi::String> boundary_subjects,
                     ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<PhasePlanNode>();
  n->name = std::move(name);
  n->phase_index = phase_index;
  n->unit_names = std::move(unit_names);
  n->unit_indices = std::move(unit_indices);
  n->ingress_edge_names = std::move(ingress_edge_names);
  n->ingress_edge_indices = std::move(ingress_edge_indices);
  n->egress_edge_names = std::move(egress_edge_names);
  n->egress_edge_indices = std::move(egress_edge_indices);
  n->boundary_subjects = std::move(boundary_subjects);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialStructureFacts::SpatialStructureFacts(ffi::String member_func,
                                             ffi::Array<ExecutionClosure> closure_candidates,
                                             ffi::Array<ClosureBoundary> boundary_candidates,
                                             ValidatedHintSet validated_hints,
                                             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialStructureFactsNode>();
  n->member_func = std::move(member_func);
  n->closure_candidates = std::move(closure_candidates);
  n->boundary_candidates = std::move(boundary_candidates);
  n->validated_hints = std::move(validated_hints);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialPlan::SpatialPlan(ffi::String member_func, ffi::Array<ExecutionUnit> execution_units,
                         ffi::Array<DataflowEdge> dataflow_edges,
                         ffi::Array<LayoutSpec> layout_specs,
                         ffi::Array<PhasePlan> phase_plans,
                         ValidatedHintSet validated_hints,
                         ffi::Array<ExecutionClosure> closures,
                         ffi::Array<ClosureBoundary> boundaries,
                         ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialPlanNode>();
  n->member_func = std::move(member_func);
  n->execution_units = std::move(execution_units);
  n->dataflow_edges = std::move(dataflow_edges);
  n->layout_specs = std::move(layout_specs);
  n->phase_plans = std::move(phase_plans);
  n->closures = std::move(closures);
  n->boundaries = std::move(boundaries);
  n->validated_hints = std::move(validated_hints);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

TLDeviceProgramInfo::TLDeviceProgramInfo(ffi::String root_symbol,
                                         ffi::Array<ffi::String> member_funcs) {
  auto n = ffi::make_object<TLDeviceProgramInfoNode>();
  n->root_symbol = std::move(root_symbol);
  n->member_funcs = std::move(member_funcs);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ExecutionClosureNode::RegisterReflection();
  ClosureBoundaryNode::RegisterReflection();
  ValidatedHintSetNode::RegisterReflection();
  ExecutionUnitNode::RegisterReflection();
  DataflowEdgeNode::RegisterReflection();
  LayoutSpecNode::RegisterReflection();
  PhasePlanNode::RegisterReflection();
  SpatialStructureFactsNode::RegisterReflection();
  SpatialPlanNode::RegisterReflection();
  TLDeviceProgramInfoNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.ExecutionClosure",
      [](ffi::String name, ffi::String closure_basis, ffi::String execution_role,
         ffi::Array<Integer> stmt_indices, ffi::Array<ffi::String> read_buffers,
         ffi::Array<ffi::String> write_buffers, ffi::Array<ffi::String> cut_frontiers,
         ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
        return ExecutionClosure(std::move(name), std::move(closure_basis),
                                std::move(execution_role), std::move(stmt_indices),
                                std::move(read_buffers), std::move(write_buffers),
                                std::move(cut_frontiers), std::move(traits),
                                std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.ClosureBoundary",
      [](ffi::String name, ffi::String kind, ffi::String source_closure,
         ffi::String target_closure, int64_t source_closure_index, int64_t target_closure_index,
         ffi::String subject, ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
        return ClosureBoundary(std::move(name), std::move(kind), std::move(source_closure),
                               std::move(target_closure), source_closure_index,
                               target_closure_index, std::move(subject), std::move(traits),
                               std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.ValidatedHintSet",
      [](ffi::Array<ffi::String> accepted_hints, ffi::Array<ffi::String> rejected_hints,
         ffi::Map<ffi::String, ffi::Any> diagnostics, ffi::Array<TIRAnchor> anchors) {
        return ValidatedHintSet(std::move(accepted_hints), std::move(rejected_hints),
                                std::move(diagnostics), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.ExecutionUnit",
      [](ffi::String name, ffi::String formation_basis, ffi::String unit_role,
         ffi::Array<Integer> stmt_indices, ffi::Array<ffi::String> read_buffers,
         ffi::Array<ffi::String> write_buffers, ffi::Array<ffi::String> traits,
         ffi::Array<TIRAnchor> anchors) {
        return ExecutionUnit(std::move(name), std::move(formation_basis), std::move(unit_role),
                             std::move(stmt_indices), std::move(read_buffers),
                             std::move(write_buffers), std::move(traits), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.DataflowEdge",
      [](ffi::String name, ffi::String kind, ffi::String producer_unit,
         ffi::String consumer_unit, int64_t producer_unit_index, int64_t consumer_unit_index,
         ffi::String subject, bool crosses_phase, ffi::Array<ffi::String> traits,
         ffi::Array<TIRAnchor> anchors) {
        return DataflowEdge(std::move(name), std::move(kind), std::move(producer_unit),
                            std::move(consumer_unit), producer_unit_index,
                            consumer_unit_index, std::move(subject), crosses_phase,
                            std::move(traits), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.LayoutSpec",
      [](ffi::String name, ffi::String subject, ffi::String scope,
         ffi::String distribution_kind, ffi::Array<ffi::String> unit_names,
         ffi::Array<Integer> unit_indices, ffi::Array<ffi::String> virtual_device_axes,
         ffi::Array<TIRAnchor> anchors) {
        return LayoutSpec(std::move(name), std::move(subject), std::move(scope),
                          std::move(distribution_kind), std::move(unit_names),
                          std::move(unit_indices), std::move(virtual_device_axes),
                          std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.PhasePlan",
      [](ffi::String name, int64_t phase_index, ffi::Array<ffi::String> unit_names,
         ffi::Array<Integer> unit_indices, ffi::Array<ffi::String> ingress_edge_names,
         ffi::Array<Integer> ingress_edge_indices, ffi::Array<ffi::String> egress_edge_names,
         ffi::Array<Integer> egress_edge_indices, ffi::Array<ffi::String> boundary_subjects,
         ffi::Array<TIRAnchor> anchors) {
        return PhasePlan(std::move(name), phase_index, std::move(unit_names),
                         std::move(unit_indices), std::move(ingress_edge_names),
                         std::move(ingress_edge_indices), std::move(egress_edge_names),
                         std::move(egress_edge_indices), std::move(boundary_subjects),
                         std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.SpatialStructureFacts",
      [](ffi::String member_func, ffi::Array<ExecutionClosure> closure_candidates,
         ffi::Array<ClosureBoundary> boundary_candidates, ValidatedHintSet validated_hints,
         ffi::Array<TIRAnchor> anchors) {
        return SpatialStructureFacts(std::move(member_func), std::move(closure_candidates),
                                     std::move(boundary_candidates),
                                     std::move(validated_hints), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.SpatialPlan",
      [](ffi::String member_func, ffi::Array<ExecutionUnit> execution_units,
         ffi::Array<DataflowEdge> dataflow_edges, ffi::Array<LayoutSpec> layout_specs,
         ffi::Array<PhasePlan> phase_plans, ValidatedHintSet validated_hints,
         ffi::Array<ExecutionClosure> closures, ffi::Array<ClosureBoundary> boundaries,
         ffi::Array<TIRAnchor> anchors) {
        return SpatialPlan(std::move(member_func), std::move(execution_units),
                           std::move(dataflow_edges), std::move(layout_specs),
                           std::move(phase_plans), std::move(validated_hints),
                           std::move(closures), std::move(boundaries),
                           std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.TLDeviceProgramInfo",
      [](ffi::String root_symbol, ffi::Array<ffi::String> member_funcs) {
        return TLDeviceProgramInfo(std::move(root_symbol), std::move(member_funcs));
      });
}

}  // namespace tl
}  // namespace tvm
