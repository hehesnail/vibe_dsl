/*!
 * \file spatial_plan.cc
 * \brief Task 1 SpatialPlan objects derived from normalized TIR.
 */

#include "spatial_plan.h"

namespace tvm {
namespace tl {

ExecutionClosure::ExecutionClosure(ffi::String name, ffi::String closure_basis,
                                   ffi::String execution_role, ffi::Array<Integer> stmt_indices,
                                   ffi::Array<ffi::String> read_buffers,
                                   ffi::Array<ffi::String> write_buffers,
                                   ffi::Array<ffi::String> cut_frontiers,
                                   ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
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

ClosureBoundary::ClosureBoundary(ffi::String name, ffi::String kind, ffi::String source_closure,
                                 ffi::String target_closure, int64_t source_closure_index,
                                 int64_t target_closure_index, ffi::String subject,
                                 ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
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

ExecutionUnit::ExecutionUnit(ffi::String name, ffi::String formation_basis, ffi::String unit_role,
                             ffi::Array<Integer> stmt_indices, ffi::Array<ffi::String> read_buffers,
                             ffi::Array<ffi::String> write_buffers, ffi::Array<ffi::String> traits,
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
                           int64_t consumer_unit_index, ffi::String subject, bool crosses_phase,
                           ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
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

DependenceComponent::DependenceComponent(ffi::String name, ffi::String component_kind,
                                         ffi::Array<Integer> unit_indices,
                                         ffi::Array<Integer> edge_indices,
                                         ffi::Array<ffi::String> subjects,
                                         ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<DependenceComponentNode>();
  n->name = std::move(name);
  n->component_kind = std::move(component_kind);
  n->unit_indices = std::move(unit_indices);
  n->edge_indices = std::move(edge_indices);
  n->subjects = std::move(subjects);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

AccessRegion::AccessRegion(ffi::String name, ffi::String subject, ffi::String unit_name,
                           int64_t unit_index, ffi::String access_kind,
                           ffi::String value_kind, int64_t logical_rank,
                           ffi::Array<ffi::String> loop_vars,
                           ffi::Array<PrimExpr> index_exprs,
                           ffi::Array<PrimExpr> lower_bounds, ffi::Array<PrimExpr> extents,
                           ffi::Array<PrimExpr> strides, ffi::String coverage_kind,
                           ffi::String predicate_kind, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<AccessRegionNode>();
  n->name = std::move(name);
  n->subject = std::move(subject);
  n->unit_name = std::move(unit_name);
  n->unit_index = unit_index;
  n->access_kind = std::move(access_kind);
  n->value_kind = std::move(value_kind);
  n->logical_rank = logical_rank;
  n->loop_vars = std::move(loop_vars);
  n->index_exprs = std::move(index_exprs);
  n->lower_bounds = std::move(lower_bounds);
  n->extents = std::move(extents);
  n->strides = std::move(strides);
  n->coverage_kind = std::move(coverage_kind);
  n->predicate_kind = std::move(predicate_kind);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

LayoutSpec::LayoutSpec(ffi::String name, ffi::String subject, ffi::String scope,
                       ffi::String distribution_kind, ffi::Array<ffi::String> unit_names,
                       ffi::Array<Integer> unit_indices,
                       ffi::Array<ffi::String> virtual_device_axes,
                       ffi::Array<PrimExpr> logical_shape, ffi::Array<PrimExpr> local_shape,
                       PrimExpr thread_extent, PrimExpr replicate_extent,
                       ffi::Array<PrimExpr> inverse_logical_index_vars,
                       ffi::Array<PrimExpr> inverse_logical_index_exprs,
                       ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<LayoutSpecNode>();
  n->name = std::move(name);
  n->subject = std::move(subject);
  n->scope = std::move(scope);
  n->distribution_kind = std::move(distribution_kind);
  n->unit_names = std::move(unit_names);
  n->unit_indices = std::move(unit_indices);
  n->virtual_device_axes = std::move(virtual_device_axes);
  n->logical_shape = std::move(logical_shape);
  n->local_shape = std::move(local_shape);
  n->thread_extent = std::move(thread_extent);
  n->replicate_extent = std::move(replicate_extent);
  n->inverse_logical_index_vars = std::move(inverse_logical_index_vars);
  n->inverse_logical_index_exprs = std::move(inverse_logical_index_exprs);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

PhasePlan::PhasePlan(ffi::String name, int64_t phase_index, ffi::Array<ffi::String> unit_names,
                     ffi::Array<Integer> unit_indices, ffi::Array<ffi::String> ingress_edge_names,
                     ffi::Array<Integer> ingress_edge_indices,
                     ffi::Array<ffi::String> egress_edge_names,
                     ffi::Array<Integer> egress_edge_indices,
                     ffi::Array<ffi::String> boundary_subjects, ffi::Array<TIRAnchor> anchors) {
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

LiveValue::LiveValue(ffi::String name, ffi::String subject, ffi::String producer_unit,
                     int64_t producer_unit_index, int64_t version_index,
                     ffi::String definition_kind, int64_t defining_access_region_index,
                     int64_t defining_event_index, ffi::String value_role,
                     ffi::Array<Integer> logical_shape, ffi::String dtype,
                     ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<LiveValueNode>();
  n->name = std::move(name);
  n->subject = std::move(subject);
  n->producer_unit = std::move(producer_unit);
  n->producer_unit_index = producer_unit_index;
  n->version_index = version_index;
  n->definition_kind = std::move(definition_kind);
  n->defining_access_region_index = defining_access_region_index;
  n->defining_event_index = defining_event_index;
  n->value_role = std::move(value_role);
  n->logical_shape = std::move(logical_shape);
  n->dtype = std::move(dtype);
  n->traits = std::move(traits);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

LiveValueEdge::LiveValueEdge(ffi::String name, ffi::String source_live_value,
                             int64_t source_live_value_index, ffi::String dataflow_edge,
                             int64_t dataflow_edge_index, ffi::String producer_unit,
                             ffi::String consumer_unit, int64_t producer_unit_index,
                             int64_t consumer_unit_index, ffi::String relation_kind,
                             ffi::String use_kind, int64_t consumer_access_region_index,
                             int64_t source_version_index, int64_t target_version_index,
                             bool requires_full_logical_value, bool accepts_distributed_slice,
                             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<LiveValueEdgeNode>();
  n->name = std::move(name);
  n->source_live_value = std::move(source_live_value);
  n->source_live_value_index = source_live_value_index;
  n->dataflow_edge = std::move(dataflow_edge);
  n->dataflow_edge_index = dataflow_edge_index;
  n->producer_unit = std::move(producer_unit);
  n->consumer_unit = std::move(consumer_unit);
  n->producer_unit_index = producer_unit_index;
  n->consumer_unit_index = consumer_unit_index;
  n->relation_kind = std::move(relation_kind);
  n->use_kind = std::move(use_kind);
  n->consumer_access_region_index = consumer_access_region_index;
  n->source_version_index = source_version_index;
  n->target_version_index = target_version_index;
  n->requires_full_logical_value = requires_full_logical_value;
  n->accepts_distributed_slice = accepts_distributed_slice;
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

MaterializationBoundary::MaterializationBoundary(
    ffi::String name, ffi::String source_live_value, int64_t source_live_value_index,
    ffi::String target_live_value, int64_t target_live_value_index, ffi::String live_value_edge,
    int64_t live_value_edge_index, ffi::String required_visibility, ffi::String logical_coverage,
    ffi::String phase_relation, int64_t source_access_region_index,
    int64_t target_access_region_index, ffi::String event_lifetime_kind,
    int64_t min_publish_pages, int64_t max_consume_pages, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<MaterializationBoundaryNode>();
  n->name = std::move(name);
  n->source_live_value = std::move(source_live_value);
  n->source_live_value_index = source_live_value_index;
  n->target_live_value = std::move(target_live_value);
  n->target_live_value_index = target_live_value_index;
  n->live_value_edge = std::move(live_value_edge);
  n->live_value_edge_index = live_value_edge_index;
  n->required_visibility = std::move(required_visibility);
  n->logical_coverage = std::move(logical_coverage);
  n->phase_relation = std::move(phase_relation);
  n->source_access_region_index = source_access_region_index;
  n->target_access_region_index = target_access_region_index;
  n->event_lifetime_kind = std::move(event_lifetime_kind);
  n->min_publish_pages = min_publish_pages;
  n->max_consume_pages = max_consume_pages;
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialPlan::SpatialPlan(ffi::String member_func, ffi::Array<ExecutionUnit> execution_units,
                         ffi::Array<AccessRegion> access_regions,
                         ffi::Array<DataflowEdge> dataflow_edges,
                         ffi::Array<DependenceComponent> dependence_components,
                         ffi::Array<LayoutSpec> layout_specs, ffi::Array<PhasePlan> phase_plans,
                         ffi::Array<LiveValue> live_values,
                         ffi::Array<LiveValueEdge> live_value_edges,
                         ffi::Array<MaterializationBoundary> materialization_boundaries,
                         ValidatedHintSet validated_hints, ffi::Array<ExecutionClosure> closures,
                         ffi::Array<ClosureBoundary> boundaries, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialPlanNode>();
  n->member_func = std::move(member_func);
  n->execution_units = std::move(execution_units);
  n->access_regions = std::move(access_regions);
  n->dataflow_edges = std::move(dataflow_edges);
  n->dependence_components = std::move(dependence_components);
  n->layout_specs = std::move(layout_specs);
  n->phase_plans = std::move(phase_plans);
  n->live_values = std::move(live_values);
  n->live_value_edges = std::move(live_value_edges);
  n->materialization_boundaries = std::move(materialization_boundaries);
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
  AccessRegionNode::RegisterReflection();
  DataflowEdgeNode::RegisterReflection();
  DependenceComponentNode::RegisterReflection();
  LayoutSpecNode::RegisterReflection();
  PhasePlanNode::RegisterReflection();
  LiveValueNode::RegisterReflection();
  LiveValueEdgeNode::RegisterReflection();
  MaterializationBoundaryNode::RegisterReflection();
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
                                std::move(cut_frontiers), std::move(traits), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.ClosureBoundary",
      [](ffi::String name, ffi::String kind, ffi::String source_closure, ffi::String target_closure,
         int64_t source_closure_index, int64_t target_closure_index, ffi::String subject,
         ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
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
      "tl.ExecutionUnit", [](ffi::String name, ffi::String formation_basis, ffi::String unit_role,
                             ffi::Array<Integer> stmt_indices, ffi::Array<ffi::String> read_buffers,
                             ffi::Array<ffi::String> write_buffers, ffi::Array<ffi::String> traits,
                             ffi::Array<TIRAnchor> anchors) {
        return ExecutionUnit(std::move(name), std::move(formation_basis), std::move(unit_role),
                             std::move(stmt_indices), std::move(read_buffers),
                             std::move(write_buffers), std::move(traits), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.AccessRegion",
      [](ffi::String name, ffi::String subject, ffi::String unit_name, int64_t unit_index,
         ffi::String access_kind, ffi::String value_kind, int64_t logical_rank,
         ffi::Array<ffi::String> loop_vars, ffi::Array<PrimExpr> index_exprs,
         ffi::Array<PrimExpr> lower_bounds, ffi::Array<PrimExpr> extents,
         ffi::Array<PrimExpr> strides, ffi::String coverage_kind, ffi::String predicate_kind,
         ffi::Array<TIRAnchor> anchors) {
        return AccessRegion(std::move(name), std::move(subject), std::move(unit_name),
                            unit_index, std::move(access_kind), std::move(value_kind),
                            logical_rank, std::move(loop_vars), std::move(index_exprs),
                            std::move(lower_bounds), std::move(extents), std::move(strides),
                            std::move(coverage_kind), std::move(predicate_kind),
                            std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.DataflowEdge",
      [](ffi::String name, ffi::String kind, ffi::String producer_unit, ffi::String consumer_unit,
         int64_t producer_unit_index, int64_t consumer_unit_index, ffi::String subject,
         bool crosses_phase, ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
        return DataflowEdge(std::move(name), std::move(kind), std::move(producer_unit),
                            std::move(consumer_unit), producer_unit_index, consumer_unit_index,
                            std::move(subject), crosses_phase, std::move(traits),
                            std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.DependenceComponent",
      [](ffi::String name, ffi::String component_kind, ffi::Array<Integer> unit_indices,
         ffi::Array<Integer> edge_indices, ffi::Array<ffi::String> subjects,
         ffi::Array<TIRAnchor> anchors) {
        return DependenceComponent(std::move(name), std::move(component_kind),
                                   std::move(unit_indices), std::move(edge_indices),
                                   std::move(subjects), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.LayoutSpec",
      [](ffi::String name, ffi::String subject, ffi::String scope, ffi::String distribution_kind,
         ffi::Array<ffi::String> unit_names, ffi::Array<Integer> unit_indices,
         ffi::Array<ffi::String> virtual_device_axes,
         ffi::Array<PrimExpr> logical_shape, ffi::Array<PrimExpr> local_shape,
         PrimExpr thread_extent, PrimExpr replicate_extent,
         ffi::Array<PrimExpr> inverse_logical_index_vars,
         ffi::Array<PrimExpr> inverse_logical_index_exprs,
         ffi::Array<TIRAnchor> anchors) {
        return LayoutSpec(std::move(name), std::move(subject), std::move(scope),
                          std::move(distribution_kind), std::move(unit_names),
                          std::move(unit_indices), std::move(virtual_device_axes),
                          std::move(logical_shape), std::move(local_shape),
                          std::move(thread_extent), std::move(replicate_extent),
                          std::move(inverse_logical_index_vars),
                          std::move(inverse_logical_index_exprs), std::move(anchors));
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
      "tl.LiveValue",
      [](ffi::String name, ffi::String subject, ffi::String producer_unit,
         int64_t producer_unit_index, int64_t version_index, ffi::String definition_kind,
         int64_t defining_access_region_index, int64_t defining_event_index,
         ffi::String value_role, ffi::Array<Integer> logical_shape, ffi::String dtype,
         ffi::Array<ffi::String> traits, ffi::Array<TIRAnchor> anchors) {
        return LiveValue(std::move(name), std::move(subject), std::move(producer_unit),
                         producer_unit_index, version_index, std::move(definition_kind),
                         defining_access_region_index, defining_event_index,
                         std::move(value_role), std::move(logical_shape), std::move(dtype),
                         std::move(traits), std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.LiveValueEdge",
      [](ffi::String name, ffi::String source_live_value, int64_t source_live_value_index,
         ffi::String dataflow_edge, int64_t dataflow_edge_index, ffi::String producer_unit,
         ffi::String consumer_unit, int64_t producer_unit_index, int64_t consumer_unit_index,
         ffi::String relation_kind, ffi::String use_kind, int64_t consumer_access_region_index,
         int64_t source_version_index, int64_t target_version_index,
         bool requires_full_logical_value, bool accepts_distributed_slice,
         ffi::Array<TIRAnchor> anchors) {
        return LiveValueEdge(std::move(name), std::move(source_live_value), source_live_value_index,
                             std::move(dataflow_edge), dataflow_edge_index,
                             std::move(producer_unit), std::move(consumer_unit),
                             producer_unit_index, consumer_unit_index, std::move(relation_kind),
                             std::move(use_kind), consumer_access_region_index,
                             source_version_index, target_version_index,
                             requires_full_logical_value, accepts_distributed_slice,
                             std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.MaterializationBoundary",
      [](ffi::String name, ffi::String source_live_value, int64_t source_live_value_index,
         ffi::String target_live_value, int64_t target_live_value_index,
         ffi::String live_value_edge, int64_t live_value_edge_index,
         ffi::String required_visibility, ffi::String logical_coverage, ffi::String phase_relation,
         int64_t source_access_region_index, int64_t target_access_region_index,
         ffi::String event_lifetime_kind, int64_t min_publish_pages, int64_t max_consume_pages,
         ffi::Array<TIRAnchor> anchors) {
        return MaterializationBoundary(
            std::move(name), std::move(source_live_value), source_live_value_index,
            std::move(target_live_value), target_live_value_index, std::move(live_value_edge),
            live_value_edge_index, std::move(required_visibility), std::move(logical_coverage),
            std::move(phase_relation), source_access_region_index, target_access_region_index,
            std::move(event_lifetime_kind), min_publish_pages, max_consume_pages,
            std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.SpatialPlan",
      [](ffi::String member_func, ffi::Array<ExecutionUnit> execution_units,
         ffi::Array<AccessRegion> access_regions, ffi::Array<DataflowEdge> dataflow_edges,
         ffi::Array<DependenceComponent> dependence_components,
         ffi::Array<LayoutSpec> layout_specs, ffi::Array<PhasePlan> phase_plans,
         ffi::Array<LiveValue> live_values,
         ffi::Array<LiveValueEdge> live_value_edges,
         ffi::Array<MaterializationBoundary> materialization_boundaries,
         ValidatedHintSet validated_hints, ffi::Array<ExecutionClosure> closures,
         ffi::Array<ClosureBoundary> boundaries, ffi::Array<TIRAnchor> anchors) {
        return SpatialPlan(std::move(member_func), std::move(execution_units),
                           std::move(access_regions), std::move(dataflow_edges),
                           std::move(dependence_components), std::move(layout_specs),
                           std::move(phase_plans), std::move(live_values),
                           std::move(live_value_edges), std::move(materialization_boundaries),
                           std::move(validated_hints), std::move(closures), std::move(boundaries),
                           std::move(anchors));
      });
  refl::GlobalDef().def(
      "tl.TLDeviceProgramInfo", [](ffi::String root_symbol, ffi::Array<ffi::String> member_funcs) {
        return TLDeviceProgramInfo(std::move(root_symbol), std::move(member_funcs));
      });
}

}  // namespace tl
}  // namespace tvm
