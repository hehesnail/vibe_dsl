/*!
 * \file build_spatial_plan_companion.cc
 * \brief Freeze Task 1 SpatialPlan companion from analyzed spatial structure facts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_plan.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

Array<Integer> ToIntegerArray(const std::vector<int>& values) {
  Array<Integer> result;
  for (int value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

Array<String> ToStringArraySorted(const std::unordered_set<std::string>& values) {
  std::vector<std::string> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  return ToStringArray(sorted);
}

Array<String> ExecutionUnitNamesForIndices(const std::vector<int>& unit_indices,
                                           const Array<ExecutionUnit>& execution_units) {
  Array<String> unit_names;
  for (int unit_index : unit_indices) {
    unit_names.push_back(execution_units[unit_index]->name);
  }
  return unit_names;
}

Array<String> DataflowEdgeNamesForIndices(const std::vector<int>& edge_indices,
                                          const Array<DataflowEdge>& dataflow_edges) {
  Array<String> edge_names;
  for (int edge_index : edge_indices) {
    edge_names.push_back(dataflow_edges[edge_index]->name);
  }
  return edge_names;
}

std::vector<int> ComputeExecutionUnitPhases(const Array<ClosureBoundary>& boundaries, int unit_count) {
  std::vector<std::vector<int>> preds(unit_count);
  for (const ClosureBoundary& boundary : boundaries) {
    if (boundary->source_closure_index < 0 || boundary->target_closure_index < 0 ||
        boundary->source_closure_index >= unit_count ||
        boundary->target_closure_index >= unit_count ||
        boundary->source_closure_index == boundary->target_closure_index) {
      continue;
    }
    preds[boundary->target_closure_index].push_back(boundary->source_closure_index);
  }
  std::vector<int> phases(unit_count, 0);
  for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
    for (int pred_index : preds[unit_index]) {
      phases[unit_index] = std::max(phases[unit_index], phases[pred_index] + 1);
    }
  }
  return phases;
}

Array<ExecutionUnit> BuildExecutionUnits(const Array<ExecutionClosure>& closures) {
  Array<ExecutionUnit> execution_units;
  for (const ExecutionClosure& closure : closures) {
    execution_units.push_back(ExecutionUnit(closure->name, closure->closure_basis,
                                            closure->execution_role, closure->stmt_indices,
                                            closure->read_buffers, closure->write_buffers,
                                            closure->traits,
                                            MakeAnchors("execution_unit", str(closure->name))));
  }
  return execution_units;
}

Array<DataflowEdge> BuildDataflowEdges(const Array<ClosureBoundary>& boundaries,
                                       const std::vector<int>& unit_phases) {
  Array<DataflowEdge> dataflow_edges;
  for (const ClosureBoundary& boundary : boundaries) {
    bool crosses_phase = false;
    if (boundary->source_closure_index >= 0 && boundary->target_closure_index >= 0 &&
        boundary->source_closure_index < static_cast<int64_t>(unit_phases.size()) &&
        boundary->target_closure_index < static_cast<int64_t>(unit_phases.size()) &&
        boundary->source_closure_index != boundary->target_closure_index) {
      crosses_phase =
          unit_phases[boundary->source_closure_index] != unit_phases[boundary->target_closure_index];
    }
    dataflow_edges.push_back(DataflowEdge(
        boundary->name, boundary->kind, boundary->source_closure, boundary->target_closure,
        boundary->source_closure_index, boundary->target_closure_index, boundary->subject,
        crosses_phase, boundary->traits, MakeAnchors("dataflow_edge", str(boundary->name))));
  }
  return dataflow_edges;
}

class BufferScopeCollector : public tir::StmtExprVisitor {
 public:
  std::unordered_map<std::string, std::string> Collect(const tir::PrimFunc& func) {
    scope_by_buffer_.clear();
    for (const auto& [_, buffer] : func->buffer_map) {
      Record(buffer);
    }
    VisitStmt(func->body);
    return scope_by_buffer_;
  }

 private:
  void Record(const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty() || scope_by_buffer_.count(name)) {
      return;
    }
    scope_by_buffer_.emplace(name, std::string(buffer.scope()));
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      Record(buffer);
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<std::string, std::string> scope_by_buffer_;
};

std::string DeriveDistributionKind(const std::string& scope) {
  if (scope == "global") {
    return "global_visible";
  }
  if (scope.rfind("shared", 0) == 0) {
    return "shared_visible";
  }
  if (scope.empty()) {
    return "logical_only";
  }
  return "local_visible";
}

Array<LayoutSpec> BuildLayoutSpecs(const tir::PrimFunc& func, const Array<ExecutionUnit>& execution_units) {
  struct LayoutInfo {
    std::string scope;
    std::vector<int> unit_indices;
    std::unordered_set<std::string> unit_names;
  };

  BufferScopeCollector collector;
  std::unordered_map<std::string, std::string> scope_by_buffer = collector.Collect(func);
  std::unordered_map<std::string, LayoutInfo> layout_info_by_subject;

  for (int unit_index = 0; unit_index < static_cast<int>(execution_units.size()); ++unit_index) {
    const ExecutionUnit& unit = execution_units[unit_index];
    auto record_subject = [&](const String& subject) {
      const std::string key = str(subject);
      if (key.empty()) {
        return;
      }
      LayoutInfo& info = layout_info_by_subject[key];
      if (info.scope.empty()) {
        auto scope_it = scope_by_buffer.find(key);
        if (scope_it != scope_by_buffer.end()) {
          info.scope = scope_it->second;
        }
      }
      if (std::find(info.unit_indices.begin(), info.unit_indices.end(), unit_index) ==
          info.unit_indices.end()) {
        info.unit_indices.push_back(unit_index);
      }
      info.unit_names.insert(str(unit->name));
    };
    for (const String& subject : unit->read_buffers) {
      record_subject(subject);
    }
    for (const String& subject : unit->write_buffers) {
      record_subject(subject);
    }
  }

  std::vector<std::string> subjects;
  subjects.reserve(layout_info_by_subject.size());
  for (const auto& [subject, _] : layout_info_by_subject) {
    subjects.push_back(subject);
  }
  std::sort(subjects.begin(), subjects.end());

  Array<LayoutSpec> layout_specs;
  for (const std::string& subject : subjects) {
    LayoutInfo& info = layout_info_by_subject[subject];
    std::sort(info.unit_indices.begin(), info.unit_indices.end());
    layout_specs.push_back(LayoutSpec(
        String("layout_" + subject), String(subject), String(info.scope),
        String(DeriveDistributionKind(info.scope)),
        ExecutionUnitNamesForIndices(info.unit_indices, execution_units),
        ToIntegerArray(info.unit_indices), Array<String>{},
        MakeAnchors("layout_spec", subject)));
  }
  return layout_specs;
}

Array<PhasePlan> BuildPhasePlans(const Array<ExecutionUnit>& execution_units,
                                 const Array<DataflowEdge>& dataflow_edges,
                                 const std::vector<int>& unit_phases) {
  struct PhaseInfo {
    std::unordered_set<std::string> unit_names;
    std::vector<int> unit_indices;
    std::unordered_set<std::string> ingress_edge_names;
    std::vector<int> ingress_edge_indices;
    std::unordered_set<std::string> egress_edge_names;
    std::vector<int> egress_edge_indices;
    std::unordered_set<std::string> boundary_subjects;
  };

  std::unordered_map<int, PhaseInfo> phases;
  for (int unit_index = 0; unit_index < static_cast<int>(execution_units.size()); ++unit_index) {
    const int phase_index = unit_index < static_cast<int>(unit_phases.size()) ? unit_phases[unit_index] : 0;
    PhaseInfo& phase = phases[phase_index];
    phase.unit_names.insert(str(execution_units[unit_index]->name));
    phase.unit_indices.push_back(unit_index);
  }

  for (int edge_index = 0; edge_index < static_cast<int>(dataflow_edges.size()); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
        edge->producer_unit_index >= static_cast<int64_t>(unit_phases.size()) ||
        edge->consumer_unit_index >= static_cast<int64_t>(unit_phases.size()) ||
        !edge->crosses_phase) {
      continue;
    }
    const int producer_phase = unit_phases[edge->producer_unit_index];
    const int consumer_phase = unit_phases[edge->consumer_unit_index];
    PhaseInfo& producer_info = phases[producer_phase];
    producer_info.egress_edge_names.insert(str(edge->name));
    producer_info.egress_edge_indices.push_back(edge_index);
    producer_info.boundary_subjects.insert(str(edge->subject));

    PhaseInfo& consumer_info = phases[consumer_phase];
    consumer_info.ingress_edge_names.insert(str(edge->name));
    consumer_info.ingress_edge_indices.push_back(edge_index);
    consumer_info.boundary_subjects.insert(str(edge->subject));
  }

  std::vector<int> phase_indices;
  phase_indices.reserve(phases.size());
  for (const auto& [phase_index, _] : phases) {
    phase_indices.push_back(phase_index);
  }
  std::sort(phase_indices.begin(), phase_indices.end());

  Array<PhasePlan> phase_plans;
  for (int phase_index : phase_indices) {
    PhaseInfo& phase = phases[phase_index];
    std::sort(phase.unit_indices.begin(), phase.unit_indices.end());
    std::sort(phase.ingress_edge_indices.begin(), phase.ingress_edge_indices.end());
    std::sort(phase.egress_edge_indices.begin(), phase.egress_edge_indices.end());
    phase_plans.push_back(PhasePlan(
        String("phase_" + std::to_string(phase_index)), phase_index,
        ExecutionUnitNamesForIndices(phase.unit_indices, execution_units),
        ToIntegerArray(phase.unit_indices),
        DataflowEdgeNamesForIndices(phase.ingress_edge_indices, dataflow_edges),
        ToIntegerArray(phase.ingress_edge_indices),
        DataflowEdgeNamesForIndices(phase.egress_edge_indices, dataflow_edges),
        ToIntegerArray(phase.egress_edge_indices),
        ToStringArraySorted(phase.boundary_subjects),
        MakeAnchors("phase_plan", std::to_string(phase_index))));
  }
  return phase_plans;
}

}  // namespace

tvm::transform::Pass BuildSpatialPlanCompanion() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_facts =
          func.value()->GetAttr<SpatialStructureFacts>(attr::kTLSpatialStructureFacts);
      if (!maybe_facts) {
        continue;
      }

      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const Array<ExecutionUnit> execution_units =
          BuildExecutionUnits(maybe_facts.value()->closure_candidates);
      const std::vector<int> unit_phases = ComputeExecutionUnitPhases(
          maybe_facts.value()->boundary_candidates, static_cast<int>(execution_units.size()));
      const Array<DataflowEdge> dataflow_edges =
          BuildDataflowEdges(maybe_facts.value()->boundary_candidates, unit_phases);
      const Array<LayoutSpec> layout_specs = BuildLayoutSpecs(func.value(), execution_units);
      const Array<PhasePlan> phase_plans =
          BuildPhasePlans(execution_units, dataflow_edges, unit_phases);

      SpatialPlan plan(String(member_func), execution_units, dataflow_edges, layout_specs,
                       phase_plans, maybe_facts.value()->validated_hints,
                       maybe_facts.value()->closure_candidates,
                       maybe_facts.value()->boundary_candidates,
                       MakeAnchors("spatial_plan", member_func));

      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialPlan, plan);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.transform.BuildSpatialPlanCompanion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BuildSpatialPlanCompanion",
                        BuildSpatialPlanCompanion);
}

}  // namespace tl
}  // namespace tvm
