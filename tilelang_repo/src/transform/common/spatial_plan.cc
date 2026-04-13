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

SpatialPlan::SpatialPlan(ffi::String member_func, ffi::Array<ExecutionClosure> closures,
                         ffi::Array<ClosureBoundary> boundaries,
                         ValidatedHintSet validated_hints,
                         ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialPlanNode>();
  n->member_func = std::move(member_func);
  n->closures = std::move(closures);
  n->boundaries = std::move(boundaries);
  n->validated_hints = std::move(validated_hints);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ExecutionClosureNode::RegisterReflection();
  ClosureBoundaryNode::RegisterReflection();
  ValidatedHintSetNode::RegisterReflection();
  SpatialStructureFactsNode::RegisterReflection();
  SpatialPlanNode::RegisterReflection();
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
      [](ffi::String member_func, ffi::Array<ExecutionClosure> closures,
         ffi::Array<ClosureBoundary> boundaries, ValidatedHintSet validated_hints,
         ffi::Array<TIRAnchor> anchors) {
        return SpatialPlan(std::move(member_func), std::move(closures),
                           std::move(boundaries), std::move(validated_hints),
                           std::move(anchors));
      });
}

}  // namespace tl
}  // namespace tvm
