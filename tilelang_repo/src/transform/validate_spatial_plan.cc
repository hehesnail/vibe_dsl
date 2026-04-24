/*!
 * \file validate_spatial_plan.cc
 * \brief Validate SpatialPlan invariants for Task 1 owner cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_analysis.h"
#include "common/spatial_plan.h"

namespace tvm {
namespace tl {

namespace {

using tvm::Bool;
using tvm::DictAttrs;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

Map<String, Any> CopyAttrs(const tir::PrimFunc& func) {
  return func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
}

bool SameIntegerArray(const Array<Integer>& lhs, const Array<Integer>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i]->value != rhs[i]->value) {
      return false;
    }
  }
  return true;
}

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool ContainsTTNoun(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  const std::string lowered = ToLower(value);
  static const char* kForbidden[] = {
      "tt_",
      "semaphore",
      "runtime_arg",
      "noc",
      "cb_",
      "cb.",
      "core_group",
      "brisc",
      "trisc",
      "ncrisc",
      "ethernet",
  };
  for (const char* token : kForbidden) {
    if (lowered.find(token) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void ValidateNoTTNoun(const std::string& value, const std::string& context) {
  ICHECK(!ContainsTTNoun(value)) << context
                                 << " must not contain TT-specific noun leakage: " << value;
}

void ValidateNoTTNouns(const Array<String>& values, const std::string& context) {
  for (const String& value : values) {
    ValidateNoTTNoun(str(value), context);
  }
}

std::vector<int> BuildPhaseIndexByUnit(const SpatialPlan& plan) {
  std::vector<int> phase_index_by_unit(plan->execution_units.size(), -1);
  std::unordered_set<int64_t> seen_phase_indices;
  for (const PhasePlan& phase : plan->phase_plans) {
    ICHECK(seen_phase_indices.insert(phase->phase_index).second)
        << "PhasePlan phase_index must be unique: " << phase->phase_index;
    ICHECK_EQ(phase->unit_names.size(), phase->unit_indices.size())
        << "PhasePlan " << phase->name << " requires aligned unit_names/unit_indices";
    for (int i = 0; i < phase->unit_indices.size(); ++i) {
      const int64_t unit_index = phase->unit_indices[i]->value;
      ICHECK_GE(unit_index, 0) << "PhasePlan " << phase->name
                               << " requires non-negative unit_index";
      ICHECK_LT(unit_index, static_cast<int64_t>(plan->execution_units.size()))
          << "PhasePlan " << phase->name << " unit_index out of bounds";
      const ExecutionUnit& unit = plan->execution_units[unit_index];
      ICHECK_EQ(str(phase->unit_names[i]), str(unit->name))
          << "PhasePlan " << phase->name << " unit_names must match execution_units";
      ICHECK_EQ(phase_index_by_unit[unit_index], -1)
          << "ExecutionUnit " << unit->name << " must belong to exactly one PhasePlan";
      phase_index_by_unit[unit_index] = static_cast<int>(phase->phase_index);
    }
  }
  for (int unit_index = 0; unit_index < static_cast<int>(phase_index_by_unit.size());
       ++unit_index) {
    ICHECK_NE(phase_index_by_unit[unit_index], -1)
        << "ExecutionUnit " << plan->execution_units[unit_index]->name
        << " is missing PhasePlan membership";
  }
  return phase_index_by_unit;
}

void ValidateExecutionUnits(const SpatialPlan& plan) {
  std::unordered_set<std::string> seen_names;
  for (const ExecutionUnit& unit : plan->execution_units) {
    ICHECK(!unit->name.empty()) << "ExecutionUnit requires name";
    ICHECK(seen_names.insert(str(unit->name)).second)
        << "duplicate ExecutionUnit name " << unit->name;
    ICHECK(!unit->formation_basis.empty())
        << "ExecutionUnit " << unit->name << " requires formation_basis";
    ICHECK(!unit->unit_role.empty()) << "ExecutionUnit " << unit->name << " requires unit_role";
    ICHECK(!unit->stmt_indices.empty())
        << "ExecutionUnit " << unit->name << " requires stmt_indices";
    ValidateNoTTNoun(str(unit->formation_basis), "ExecutionUnit formation_basis");
    ValidateNoTTNoun(str(unit->unit_role), "ExecutionUnit unit_role");
    ValidateNoTTNouns(unit->traits, "ExecutionUnit traits");
  }
}

void ValidateCompatibilityProjection(const SpatialPlan& plan) {
  ICHECK_EQ(plan->execution_units.size(), plan->closures.size())
      << "SpatialPlan compatibility projection requires "
         "execution_units/closures alignment";
  ICHECK_EQ(plan->dataflow_edges.size(), plan->boundaries.size())
      << "SpatialPlan compatibility projection requires "
         "dataflow_edges/boundaries alignment";

  for (int i = 0; i < plan->execution_units.size(); ++i) {
    const ExecutionUnit& unit = plan->execution_units[i];
    const ExecutionClosure& closure = plan->closures[i];
    ICHECK_EQ(str(unit->name), str(closure->name))
        << "SpatialPlan compatibility projection mismatch for ExecutionUnit "
           "name";
    ICHECK_EQ(str(unit->formation_basis), str(closure->closure_basis))
        << "SpatialPlan compatibility projection mismatch for formation_basis";
    ICHECK_EQ(str(unit->unit_role), str(closure->execution_role))
        << "SpatialPlan compatibility projection mismatch for unit_role";
    ICHECK(SameIntegerArray(unit->stmt_indices, closure->stmt_indices))
        << "SpatialPlan compatibility projection mismatch for stmt_indices";
    ICHECK(SameStringArray(unit->read_buffers, closure->read_buffers))
        << "SpatialPlan compatibility projection mismatch for read_buffers";
    ICHECK(SameStringArray(unit->write_buffers, closure->write_buffers))
        << "SpatialPlan compatibility projection mismatch for write_buffers";
    ICHECK(SameStringArray(unit->traits, closure->traits))
        << "SpatialPlan compatibility projection mismatch for traits";
  }

  for (int i = 0; i < plan->dataflow_edges.size(); ++i) {
    const DataflowEdge& edge = plan->dataflow_edges[i];
    const ClosureBoundary& boundary = plan->boundaries[i];
    ICHECK_EQ(str(edge->name), str(boundary->name))
        << "SpatialPlan compatibility projection mismatch for DataflowEdge "
           "name";
    ICHECK_EQ(str(edge->kind), str(boundary->kind))
        << "SpatialPlan compatibility projection mismatch for DataflowEdge "
           "kind";
    ICHECK_EQ(str(edge->producer_unit), str(boundary->source_closure))
        << "SpatialPlan compatibility projection mismatch for producer_unit";
    ICHECK_EQ(str(edge->consumer_unit), str(boundary->target_closure))
        << "SpatialPlan compatibility projection mismatch for consumer_unit";
    ICHECK_EQ(edge->producer_unit_index, boundary->source_closure_index)
        << "SpatialPlan compatibility projection mismatch for "
           "producer_unit_index";
    ICHECK_EQ(edge->consumer_unit_index, boundary->target_closure_index)
        << "SpatialPlan compatibility projection mismatch for "
           "consumer_unit_index";
    ICHECK_EQ(str(edge->subject), str(boundary->subject))
        << "SpatialPlan compatibility projection mismatch for DataflowEdge "
           "subject";
    ICHECK(SameStringArray(edge->traits, boundary->traits))
        << "SpatialPlan compatibility projection mismatch for DataflowEdge "
           "traits";
  }
}

void ValidateDataflowEdges(const SpatialPlan& plan, const std::vector<int>& phase_index_by_unit) {
  std::unordered_set<std::string> seen_names;
  for (const DataflowEdge& edge : plan->dataflow_edges) {
    ICHECK(!edge->name.empty()) << "DataflowEdge requires name";
    ICHECK(seen_names.insert(str(edge->name)).second)
        << "duplicate DataflowEdge name " << edge->name;
    ICHECK(!edge->kind.empty()) << "DataflowEdge " << edge->name << " requires kind";
    ICHECK(!edge->subject.empty()) << "DataflowEdge " << edge->name << " requires subject";
    ValidateNoTTNoun(str(edge->kind), "DataflowEdge kind");
    ValidateNoTTNouns(edge->traits, "DataflowEdge traits");

    ICHECK_GE(edge->producer_unit_index, 0)
        << "DataflowEdge " << edge->name << " requires producer_unit_index";
    ICHECK_GE(edge->consumer_unit_index, 0)
        << "DataflowEdge " << edge->name << " requires consumer_unit_index";
    ICHECK_LT(edge->producer_unit_index, static_cast<int64_t>(plan->execution_units.size()))
        << "DataflowEdge " << edge->name << " producer_unit_index out of bounds";
    ICHECK_LT(edge->consumer_unit_index, static_cast<int64_t>(plan->execution_units.size()))
        << "DataflowEdge " << edge->name << " consumer_unit_index out of bounds";

    const ExecutionUnit& producer = plan->execution_units[edge->producer_unit_index];
    const ExecutionUnit& consumer = plan->execution_units[edge->consumer_unit_index];
    ICHECK_EQ(str(edge->producer_unit), str(producer->name))
        << "DataflowEdge " << edge->name << " producer_unit must match producer_unit_index";
    ICHECK_EQ(str(edge->consumer_unit), str(consumer->name))
        << "DataflowEdge " << edge->name << " consumer_unit must match consumer_unit_index";

    const bool crosses_phase = phase_index_by_unit[edge->producer_unit_index] !=
                               phase_index_by_unit[edge->consumer_unit_index];
    ICHECK_EQ(edge->crosses_phase, crosses_phase)
        << "DataflowEdge " << edge->name << " crosses_phase must match PhasePlan membership";
  }
}

void ValidateLayoutSpecs(const SpatialPlan& plan) {
  std::unordered_map<std::string, std::unordered_set<int>> unit_indices_by_subject;
  for (int unit_index = 0; unit_index < static_cast<int>(plan->execution_units.size());
       ++unit_index) {
    const ExecutionUnit& unit = plan->execution_units[unit_index];
    for (const String& subject : unit->read_buffers) {
      unit_indices_by_subject[str(subject)].insert(unit_index);
    }
    for (const String& subject : unit->write_buffers) {
      unit_indices_by_subject[str(subject)].insert(unit_index);
    }
  }

  std::unordered_set<std::string> seen_names;
  for (const LayoutSpec& layout : plan->layout_specs) {
    ICHECK(!layout->name.empty()) << "LayoutSpec requires name";
    ICHECK(seen_names.insert(str(layout->name)).second)
        << "duplicate LayoutSpec name " << layout->name;
    ICHECK(!layout->subject.empty()) << "LayoutSpec " << layout->name << " requires subject";
    ICHECK(!layout->distribution_kind.empty())
        << "LayoutSpec " << layout->name << " requires distribution_kind";
    ICHECK_EQ(layout->unit_names.size(), layout->unit_indices.size())
        << "LayoutSpec " << layout->name << " requires aligned unit_names/unit_indices";
    ValidateNoTTNoun(str(layout->distribution_kind), "LayoutSpec distribution_kind");

    auto it = unit_indices_by_subject.find(str(layout->subject));
    ICHECK(it != unit_indices_by_subject.end())
        << "LayoutSpec " << layout->name << " subject must be referenced by ExecutionUnit";
    for (int i = 0; i < layout->unit_indices.size(); ++i) {
      const int64_t unit_index = layout->unit_indices[i]->value;
      ICHECK_GE(unit_index, 0) << "LayoutSpec " << layout->name
                               << " requires non-negative unit_index";
      ICHECK_LT(unit_index, static_cast<int64_t>(plan->execution_units.size()))
          << "LayoutSpec " << layout->name << " unit_index out of bounds";
      ICHECK(it->second.count(static_cast<int>(unit_index)))
          << "LayoutSpec " << layout->name << " unit_indices must reference subject users";
      ICHECK_EQ(str(layout->unit_names[i]), str(plan->execution_units[unit_index]->name))
          << "LayoutSpec " << layout->name << " unit_names must match execution_units";
    }
  }
}

void ValidatePhasePlans(const SpatialPlan& plan, const std::vector<int>& phase_index_by_unit) {
  std::unordered_map<std::string, int> edge_index_by_name;
  for (int edge_index = 0; edge_index < plan->dataflow_edges.size(); ++edge_index) {
    edge_index_by_name.emplace(str(plan->dataflow_edges[edge_index]->name), edge_index);
  }

  for (const PhasePlan& phase : plan->phase_plans) {
    std::unordered_set<int> ingress_edge_indices;
    std::unordered_set<int> egress_edge_indices;
    for (int i = 0; i < phase->ingress_edge_indices.size(); ++i) {
      const int edge_index = phase->ingress_edge_indices[i]->value;
      ICHECK_GE(edge_index, 0) << "PhasePlan " << phase->name
                               << " ingress edge index must be non-negative";
      ICHECK_LT(edge_index, plan->dataflow_edges.size())
          << "PhasePlan " << phase->name << " ingress edge index out of bounds";
      const DataflowEdge& edge = plan->dataflow_edges[edge_index];
      ICHECK_EQ(str(phase->ingress_edge_names[i]), str(edge->name))
          << "PhasePlan " << phase->name << " ingress edge names must match dataflow_edges";
      ICHECK(edge->crosses_phase) << "PhasePlan " << phase->name
                                  << " ingress edges must be cross-phase";
      ICHECK_EQ(phase_index_by_unit[edge->consumer_unit_index], phase->phase_index)
          << "PhasePlan " << phase->name << " ingress edges must terminate in this phase";
      ingress_edge_indices.insert(edge_index);
    }
    for (int i = 0; i < phase->egress_edge_indices.size(); ++i) {
      const int edge_index = phase->egress_edge_indices[i]->value;
      ICHECK_GE(edge_index, 0) << "PhasePlan " << phase->name
                               << " egress edge index must be non-negative";
      ICHECK_LT(edge_index, plan->dataflow_edges.size())
          << "PhasePlan " << phase->name << " egress edge index out of bounds";
      const DataflowEdge& edge = plan->dataflow_edges[edge_index];
      ICHECK_EQ(str(phase->egress_edge_names[i]), str(edge->name))
          << "PhasePlan " << phase->name << " egress edge names must match dataflow_edges";
      ICHECK(edge->crosses_phase) << "PhasePlan " << phase->name
                                  << " egress edges must be cross-phase";
      ICHECK_EQ(phase_index_by_unit[edge->producer_unit_index], phase->phase_index)
          << "PhasePlan " << phase->name << " egress edges must originate in this phase";
      egress_edge_indices.insert(edge_index);
    }

    std::unordered_set<std::string> expected_boundary_subjects;
    for (int edge_index : ingress_edge_indices) {
      expected_boundary_subjects.insert(str(plan->dataflow_edges[edge_index]->subject));
    }
    for (int edge_index : egress_edge_indices) {
      expected_boundary_subjects.insert(str(plan->dataflow_edges[edge_index]->subject));
    }
    for (const String& subject : phase->boundary_subjects) {
      ICHECK(expected_boundary_subjects.count(str(subject)))
          << "PhasePlan " << phase->name << " boundary_subjects must come from cross-phase edges";
    }
  }
}

void ValidateLiveValueBoundaryObjects(const SpatialPlan& plan,
                                      const std::vector<int>& phase_index_by_unit) {
  ICHECK_EQ(plan->live_values.size(), plan->dataflow_edges.size())
      << "SpatialPlan live_values must cover every DataflowEdge";
  ICHECK_EQ(plan->live_value_edges.size(), plan->dataflow_edges.size())
      << "SpatialPlan live_value_edges must cover every DataflowEdge";
  ICHECK_EQ(plan->materialization_boundaries.size(), plan->dataflow_edges.size())
      << "SpatialPlan materialization_boundaries must cover every DataflowEdge";

  std::unordered_set<std::string> seen_live_value_names;
  for (int live_value_index = 0; live_value_index < plan->live_values.size(); ++live_value_index) {
    const LiveValue& value = plan->live_values[live_value_index];
    ICHECK(!value->name.empty()) << "LiveValue requires name";
    ICHECK(seen_live_value_names.insert(str(value->name)).second)
        << "duplicate LiveValue name " << value->name;
    ICHECK(!value->subject.empty()) << "LiveValue " << value->name << " requires subject";
    ICHECK(!value->producer_unit.empty())
        << "LiveValue " << value->name << " requires producer_unit";
    ICHECK_GE(value->producer_unit_index, 0)
        << "LiveValue " << value->name << " requires producer_unit_index";
    ICHECK_LT(value->producer_unit_index, static_cast<int64_t>(plan->execution_units.size()))
        << "LiveValue " << value->name << " producer_unit_index out of bounds";
    ICHECK_EQ(str(value->producer_unit),
              str(plan->execution_units[value->producer_unit_index]->name))
        << "LiveValue " << value->name << " producer_unit must match producer_unit_index";
    ICHECK(!value->value_role.empty()) << "LiveValue " << value->name << " requires value_role";
    ValidateNoTTNoun(str(value->value_role), "LiveValue value_role");
    ValidateNoTTNouns(value->traits, "LiveValue traits");
    ICHECK(!value->logical_shape.empty())
        << "LiveValue " << value->name << " requires logical_shape";
    for (const Integer& dim : value->logical_shape) {
      ICHECK_GT(dim->value, 0) << "LiveValue " << value->name
                               << " requires positive logical_shape dimensions";
    }
    ICHECK(!value->dtype.empty()) << "LiveValue " << value->name << " requires dtype";
    ICHECK_NE(str(value->dtype), "unknown")
        << "LiveValue " << value->name << " requires resolved dtype";
  }

  std::unordered_set<std::string> seen_live_edge_names;
  for (int live_edge_index = 0; live_edge_index < plan->live_value_edges.size();
       ++live_edge_index) {
    const LiveValueEdge& live_edge = plan->live_value_edges[live_edge_index];
    ICHECK(!live_edge->name.empty()) << "LiveValueEdge requires name";
    ICHECK(seen_live_edge_names.insert(str(live_edge->name)).second)
        << "duplicate LiveValueEdge name " << live_edge->name;
    ICHECK(!live_edge->source_live_value.empty())
        << "LiveValueEdge " << live_edge->name << " requires source_live_value";
    ICHECK_GE(live_edge->source_live_value_index, 0)
        << "LiveValueEdge " << live_edge->name << " requires source_live_value_index";
    ICHECK_LT(live_edge->source_live_value_index, static_cast<int64_t>(plan->live_values.size()))
        << "LiveValueEdge " << live_edge->name << " source_live_value_index out of bounds";
    const LiveValue& source_live_value = plan->live_values[live_edge->source_live_value_index];
    ICHECK_EQ(str(live_edge->source_live_value), str(source_live_value->name))
        << "LiveValueEdge " << live_edge->name
        << " source_live_value must match source_live_value_index";
    ICHECK(!live_edge->dataflow_edge.empty())
        << "LiveValueEdge " << live_edge->name << " requires dataflow_edge";
    ICHECK_GE(live_edge->dataflow_edge_index, 0)
        << "LiveValueEdge " << live_edge->name << " requires dataflow_edge_index";
    ICHECK_LT(live_edge->dataflow_edge_index, static_cast<int64_t>(plan->dataflow_edges.size()))
        << "LiveValueEdge " << live_edge->name << " dataflow_edge_index out of bounds";
    const DataflowEdge& dataflow_edge = plan->dataflow_edges[live_edge->dataflow_edge_index];
    ICHECK_EQ(str(live_edge->dataflow_edge), str(dataflow_edge->name))
        << "LiveValueEdge " << live_edge->name << " dataflow_edge must match dataflow_edge_index";
    ICHECK_EQ(str(source_live_value->subject), str(dataflow_edge->subject))
        << "LiveValueEdge " << live_edge->name
        << " source_live_value subject must match DataflowEdge subject";
    ICHECK_EQ(str(live_edge->producer_unit), str(dataflow_edge->producer_unit))
        << "LiveValueEdge " << live_edge->name << " producer_unit must match DataflowEdge";
    ICHECK_EQ(str(live_edge->consumer_unit), str(dataflow_edge->consumer_unit))
        << "LiveValueEdge " << live_edge->name << " consumer_unit must match DataflowEdge";
    ICHECK_EQ(live_edge->producer_unit_index, dataflow_edge->producer_unit_index)
        << "LiveValueEdge " << live_edge->name << " producer_unit_index must match DataflowEdge";
    ICHECK_EQ(live_edge->consumer_unit_index, dataflow_edge->consumer_unit_index)
        << "LiveValueEdge " << live_edge->name << " consumer_unit_index must match DataflowEdge";
    ICHECK(!live_edge->relation_kind.empty())
        << "LiveValueEdge " << live_edge->name << " requires relation_kind";
    ICHECK_EQ(str(live_edge->relation_kind), str(dataflow_edge->kind))
        << "LiveValueEdge " << live_edge->name << " relation_kind must match DataflowEdge kind";
    ValidateNoTTNoun(str(live_edge->relation_kind), "LiveValueEdge relation_kind");
    ICHECK(live_edge->requires_full_logical_value || live_edge->accepts_distributed_slice)
        << "LiveValueEdge " << live_edge->name
        << " must either require full logical value or accept distributed slice";
  }

  std::unordered_set<std::string> seen_boundary_names;
  for (const MaterializationBoundary& boundary : plan->materialization_boundaries) {
    ICHECK(!boundary->name.empty()) << "MaterializationBoundary requires name";
    ICHECK(seen_boundary_names.insert(str(boundary->name)).second)
        << "duplicate MaterializationBoundary name " << boundary->name;
    ICHECK(!boundary->source_live_value.empty())
        << "MaterializationBoundary " << boundary->name << " requires source_live_value";
    ICHECK_GE(boundary->source_live_value_index, 0)
        << "MaterializationBoundary " << boundary->name << " requires source_live_value_index";
    ICHECK_LT(boundary->source_live_value_index, static_cast<int64_t>(plan->live_values.size()))
        << "MaterializationBoundary " << boundary->name << " source_live_value_index out of bounds";
    const LiveValue& source_live_value = plan->live_values[boundary->source_live_value_index];
    ICHECK_EQ(str(boundary->source_live_value), str(source_live_value->name))
        << "MaterializationBoundary " << boundary->name
        << " source_live_value must match source_live_value_index";
    ICHECK(!boundary->live_value_edge.empty())
        << "MaterializationBoundary " << boundary->name << " requires live_value_edge";
    ICHECK_GE(boundary->live_value_edge_index, 0)
        << "MaterializationBoundary " << boundary->name << " requires live_value_edge_index";
    ICHECK_LT(boundary->live_value_edge_index, static_cast<int64_t>(plan->live_value_edges.size()))
        << "MaterializationBoundary " << boundary->name << " live_value_edge_index out of bounds";
    const LiveValueEdge& live_edge = plan->live_value_edges[boundary->live_value_edge_index];
    ICHECK_EQ(str(boundary->live_value_edge), str(live_edge->name))
        << "MaterializationBoundary " << boundary->name
        << " live_value_edge must match live_value_edge_index";
    ICHECK_EQ(str(boundary->source_live_value), str(live_edge->source_live_value))
        << "MaterializationBoundary " << boundary->name
        << " source_live_value must match LiveValueEdge source";
    ICHECK_EQ(boundary->source_live_value_index, live_edge->source_live_value_index)
        << "MaterializationBoundary " << boundary->name
        << " source_live_value_index must match LiveValueEdge source";
    const DataflowEdge& dataflow_edge = plan->dataflow_edges[live_edge->dataflow_edge_index];
    const bool crosses_phase = phase_index_by_unit[dataflow_edge->producer_unit_index] !=
                               phase_index_by_unit[dataflow_edge->consumer_unit_index];
    ICHECK(!boundary->required_visibility.empty())
        << "MaterializationBoundary " << boundary->name << " requires required_visibility";
    ICHECK(!boundary->logical_coverage.empty())
        << "MaterializationBoundary " << boundary->name << " requires logical_coverage";
    ICHECK(!boundary->phase_relation.empty())
        << "MaterializationBoundary " << boundary->name << " requires phase_relation";
    ValidateNoTTNoun(str(boundary->required_visibility),
                     "MaterializationBoundary required_visibility");
    ValidateNoTTNoun(str(boundary->logical_coverage), "MaterializationBoundary logical_coverage");
    ValidateNoTTNoun(str(boundary->phase_relation), "MaterializationBoundary phase_relation");
    ICHECK_EQ(str(boundary->phase_relation), crosses_phase ? "cross_phase" : "same_phase")
        << "MaterializationBoundary " << boundary->name
        << " phase_relation must match DataflowEdge phase membership";
    ICHECK_EQ(str(boundary->required_visibility), crosses_phase ? "next_phase" : "same_unit")
        << "MaterializationBoundary " << boundary->name
        << " required_visibility must match DataflowEdge phase membership";
  }
}

void CheckSpatialPlan(const SpatialPlan& plan) {
  ICHECK(!plan->member_func.empty()) << "SpatialPlan requires member_func";
  ICHECK(plan->validated_hints.defined()) << "SpatialPlan requires ValidatedHintSet";
  ValidateExecutionUnits(plan);
  ValidateCompatibilityProjection(plan);
  const std::vector<int> phase_index_by_unit = BuildPhaseIndexByUnit(plan);
  ValidateDataflowEdges(plan, phase_index_by_unit);
  ValidateLayoutSpecs(plan);
  ValidatePhasePlans(plan, phase_index_by_unit);
  ValidateLiveValueBoundaryObjects(plan, phase_index_by_unit);
}

}  // namespace

tvm::transform::Pass ValidateSpatialPlan() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_plan = func.value()->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
      if (!maybe_plan) {
        continue;
      }
      CheckSpatialPlan(maybe_plan.value());
      tir::PrimFunc validated = func.value();
      Map<String, Any> attrs = CopyAttrs(validated);
      attrs.Set(attr::kTLSpatialPlanValidated, Bool(true));
      validated.CopyOnWrite()->attrs = DictAttrs(attrs);
      updated->Add(gvar, validated, true);
    }
    return updated;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.ValidateSpatialPlan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateSpatialPlan", ValidateSpatialPlan);
}

}  // namespace tl
}  // namespace tvm
