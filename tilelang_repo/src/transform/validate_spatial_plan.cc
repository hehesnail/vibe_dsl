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
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_analysis.h"
#include "common/spatial_access_region.h"
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

bool IsOneOf(const std::string& value, std::initializer_list<const char*> allowed) {
  for (const char* item : allowed) {
    if (value == item) {
      return true;
    }
  }
  return false;
}

bool AccessRegionReads(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "read" || access_kind == "read_write" || access_kind == "reduce_read";
}

bool AccessRegionWrites(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "write" || access_kind == "read_write" || access_kind == "reduce_write";
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
  ICHECK_LE(plan->boundaries.size(), plan->dataflow_edges.size())
      << "SpatialPlan compatibility projection requires "
         "boundaries to remain a DataflowEdge prefix";

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

  for (int i = 0; i < plan->boundaries.size(); ++i) {
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

void ValidateDependenceComponents(const SpatialPlan& plan) {
  std::unordered_set<std::string> seen_names;
  for (const DependenceComponent& component : plan->dependence_components) {
    ICHECK(!component->name.empty()) << "DependenceComponent requires name";
    ICHECK(seen_names.insert(str(component->name)).second)
        << "duplicate DependenceComponent name " << component->name;
    ICHECK(!component->component_kind.empty())
        << "DependenceComponent " << component->name << " requires component_kind";
    ICHECK(IsOneOf(str(component->component_kind),
                   {"carry_cycle", "reduction_cycle", "recurrence"}))
        << "DependenceComponent " << component->name << " component_kind has unsupported value "
        << component->component_kind;
    ValidateNoTTNoun(str(component->component_kind), "DependenceComponent component_kind");
    ICHECK(!component->unit_indices.empty())
        << "DependenceComponent " << component->name << " requires unit_indices";
    ICHECK(!component->edge_indices.empty())
        << "DependenceComponent " << component->name << " requires edge_indices";
    ICHECK(!component->subjects.empty())
        << "DependenceComponent " << component->name << " requires subjects";

    std::unordered_set<int64_t> component_units;
    for (const Integer& unit_index_value : component->unit_indices) {
      const int64_t unit_index = unit_index_value->value;
      ICHECK_GE(unit_index, 0)
          << "DependenceComponent " << component->name << " unit_index must be non-negative";
      ICHECK_LT(unit_index, static_cast<int64_t>(plan->execution_units.size()))
          << "DependenceComponent " << component->name << " unit_index out of bounds";
      ICHECK(component_units.insert(unit_index).second)
          << "DependenceComponent " << component->name << " duplicate unit_index";
    }

    std::unordered_set<std::string> component_subjects;
    for (const String& subject : component->subjects) {
      ICHECK(!subject.empty())
          << "DependenceComponent " << component->name << " subject must be non-empty";
      component_subjects.insert(str(subject));
    }

    bool has_carry_edge = false;
    bool has_reduction_edge = false;
    for (const Integer& edge_index_value : component->edge_indices) {
      const int64_t edge_index = edge_index_value->value;
      ICHECK_GE(edge_index, 0)
          << "DependenceComponent " << component->name << " edge_index must be non-negative";
      ICHECK_LT(edge_index, static_cast<int64_t>(plan->dataflow_edges.size()))
          << "DependenceComponent " << component->name << " edge_index out of bounds";
      const DataflowEdge& edge = plan->dataflow_edges[edge_index];
      ICHECK(component_units.count(edge->producer_unit_index))
          << "DependenceComponent " << component->name
          << " edge producer must be in component units";
      ICHECK(component_units.count(edge->consumer_unit_index))
          << "DependenceComponent " << component->name
          << " edge consumer must be in component units";
      ICHECK(component_subjects.count(str(edge->subject)))
          << "DependenceComponent " << component->name
          << " subjects must include every component edge subject";
      has_carry_edge = has_carry_edge || str(edge->kind) == "carry";
      has_reduction_edge = has_reduction_edge || str(edge->kind) == "reduction";
    }
    if (str(component->component_kind) == "carry_cycle") {
      ICHECK(has_carry_edge)
          << "DependenceComponent " << component->name << " carry_cycle requires a carry edge";
    }
    if (str(component->component_kind) == "reduction_cycle") {
      ICHECK(has_reduction_edge)
          << "DependenceComponent " << component->name
          << " reduction_cycle requires a reduction edge";
    }
  }
}

void ValidateAccessRegions(const SpatialPlan& plan) {
  std::unordered_set<std::string> seen_names;
  std::unordered_set<std::string> covered_accesses;
  for (const AccessRegion& region : plan->access_regions) {
    ICHECK(!region->name.empty()) << "AccessRegion requires name";
    ICHECK(seen_names.insert(str(region->name)).second)
        << "duplicate AccessRegion name " << region->name;
    ICHECK(!region->subject.empty()) << "AccessRegion " << region->name << " requires subject";
    ICHECK(!region->unit_name.empty())
        << "AccessRegion " << region->name << " requires unit_name";
    ICHECK_GE(region->unit_index, 0)
        << "AccessRegion " << region->name << " requires unit_index";
    ICHECK_LT(region->unit_index, static_cast<int64_t>(plan->execution_units.size()))
        << "AccessRegion " << region->name << " unit_index out of bounds";
    const ExecutionUnit& unit = plan->execution_units[region->unit_index];
    ICHECK_EQ(str(region->unit_name), str(unit->name))
        << "AccessRegion " << region->name << " unit_name must match unit_index";

    ICHECK(IsOneOf(str(region->access_kind),
                   {"read", "write", "read_write", "reduce_read", "reduce_write"}))
        << "AccessRegion " << region->name << " access_kind has unsupported value "
        << region->access_kind;
    ICHECK(IsOneOf(str(region->value_kind),
                   {"tensor", "tile", "fragment", "accumulator", "scalar"}))
        << "AccessRegion " << region->name << " value_kind has unsupported value "
        << region->value_kind;
    ICHECK(IsOneOf(str(region->coverage_kind),
                   {"full", "slice", "row_slice", "grouped_slice", "scalar"}))
        << "AccessRegion " << region->name << " coverage_kind has unsupported value "
        << region->coverage_kind;
    ICHECK(IsOneOf(str(region->predicate_kind), {"unconditional", "guarded", "unknown"}))
        << "AccessRegion " << region->name << " predicate_kind has unsupported value "
        << region->predicate_kind;
    ValidateNoTTNoun(str(region->access_kind), "AccessRegion access_kind");
    ValidateNoTTNoun(str(region->value_kind), "AccessRegion value_kind");
    ValidateNoTTNoun(str(region->coverage_kind), "AccessRegion coverage_kind");
    ValidateNoTTNoun(str(region->predicate_kind), "AccessRegion predicate_kind");

    ICHECK_GE(region->logical_rank, 0)
        << "AccessRegion " << region->name << " requires non-negative logical_rank";
    ICHECK_EQ(region->logical_rank, static_cast<int64_t>(region->extents.size()))
        << "AccessRegion " << region->name << " logical_rank must match extents";
    ICHECK_EQ(region->lower_bounds.size(), region->extents.size())
        << "AccessRegion " << region->name << " requires lower_bounds/extents alignment";
    ICHECK_EQ(region->strides.size(), region->extents.size())
        << "AccessRegion " << region->name << " requires strides/extents alignment";
    if (str(region->coverage_kind) == "scalar") {
      ICHECK_EQ(region->logical_rank, 0)
          << "AccessRegion " << region->name << " scalar coverage requires rank 0";
    } else {
      ICHECK_GT(region->logical_rank, 0)
          << "AccessRegion " << region->name << " non-scalar coverage requires rank";
    }

    const bool unit_reads_subject =
        std::find_if(unit->read_buffers.begin(), unit->read_buffers.end(), [&](const String& name) {
          return str(name) == str(region->subject);
        }) != unit->read_buffers.end();
    const bool unit_writes_subject =
        std::find_if(unit->write_buffers.begin(), unit->write_buffers.end(),
                     [&](const String& name) { return str(name) == str(region->subject); }) !=
        unit->write_buffers.end();
    if (str(region->access_kind) == "read" || str(region->access_kind) == "reduce_read") {
      ICHECK(unit_reads_subject)
          << "AccessRegion " << region->name << " read subject must be in ExecutionUnit reads";
      covered_accesses.insert(std::to_string(region->unit_index) + "|" + str(region->subject) +
                              "|read");
    } else if (str(region->access_kind) == "write" ||
               str(region->access_kind) == "reduce_write") {
      ICHECK(unit_writes_subject)
          << "AccessRegion " << region->name << " write subject must be in ExecutionUnit writes";
      covered_accesses.insert(std::to_string(region->unit_index) + "|" + str(region->subject) +
                              "|write");
    } else {
      ICHECK(unit_reads_subject && unit_writes_subject)
          << "AccessRegion " << region->name
          << " read_write subject must be in ExecutionUnit reads and writes";
      covered_accesses.insert(std::to_string(region->unit_index) + "|" + str(region->subject) +
                              "|read");
      covered_accesses.insert(std::to_string(region->unit_index) + "|" + str(region->subject) +
                              "|write");
    }
  }

  for (int unit_index = 0; unit_index < plan->execution_units.size(); ++unit_index) {
    const ExecutionUnit& unit = plan->execution_units[unit_index];
    for (const String& subject : unit->read_buffers) {
      const std::string key = std::to_string(unit_index) + "|" + str(subject) + "|read";
      ICHECK(covered_accesses.count(key))
          << "SpatialPlan access_regions must cover ExecutionUnit read " << unit->name << " "
          << subject;
    }
    for (const String& subject : unit->write_buffers) {
      const std::string key = std::to_string(unit_index) + "|" + str(subject) + "|write";
      ICHECK(covered_accesses.count(key))
          << "SpatialPlan access_regions must cover ExecutionUnit write " << unit->name << " "
          << subject;
    }
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
    if (!layout->logical_shape.empty()) {
      ICHECK(!layout->local_shape.empty())
          << "LayoutSpec " << layout->name
          << " requires local_shape when logical_shape is present";
      ICHECK(layout->thread_extent.defined())
          << "LayoutSpec " << layout->name
          << " requires thread_extent when logical_shape is present";
      ICHECK(layout->replicate_extent.defined())
          << "LayoutSpec " << layout->name
          << " requires replicate_extent when logical_shape is present";
      ICHECK(!layout->inverse_logical_index_exprs.empty())
          << "LayoutSpec " << layout->name
          << " requires inverse logical layout expressions";
      ICHECK(!layout->inverse_logical_index_vars.empty())
          << "LayoutSpec " << layout->name
          << " requires inverse logical layout variables";
    }

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

void ValidateTensorPlacementIntents(const SpatialPlan& plan) {
  std::unordered_set<std::string> layout_subjects;
  for (const LayoutSpec& layout : plan->layout_specs) {
    layout_subjects.insert(str(layout->subject));
  }
  std::unordered_set<std::string> seen_names;
  std::unordered_set<std::string> seen_subjects;
  for (const TensorPlacementIntent& intent : plan->tensor_placement_intents) {
    ICHECK(!intent->name.empty()) << "TensorPlacementIntent requires name";
    ICHECK(seen_names.insert(str(intent->name)).second)
        << "duplicate TensorPlacementIntent name " << intent->name;
    ICHECK(!intent->subject.empty())
        << "TensorPlacementIntent " << intent->name << " requires subject";
    ICHECK(layout_subjects.count(str(intent->subject)) != 0U)
        << "TensorPlacementIntent " << intent->name
        << " subject must match a SpatialPlan LayoutSpec";
    ICHECK(seen_subjects.insert(str(intent->subject)).second)
        << "duplicate TensorPlacementIntent subject " << intent->subject;
    const std::string source = str(intent->source);
    ICHECK(source == "user" || source == "op_contract" ||
           source == "derived_default" || source == "materialization_requirement")
        << "TensorPlacementIntent " << intent->name << " has invalid source "
        << intent->source;
    const std::string memory_space = str(intent->memory_space_class);
    ICHECK(memory_space == "DRAM" || memory_space == "L1" || memory_space == "either")
        << "TensorPlacementIntent " << intent->name
        << " has invalid memory_space_class " << intent->memory_space_class;
    const std::string strategy = str(intent->strategy_class);
    ICHECK(strategy == "interleaved" || strategy == "height_sharded" ||
           strategy == "width_sharded" || strategy == "block_sharded" ||
           strategy == "nd_sharded")
        << "TensorPlacementIntent " << intent->name << " has invalid strategy_class "
        << intent->strategy_class;
    ICHECK_GE(intent->logical_rank, 0)
        << "TensorPlacementIntent " << intent->name << " requires non-negative rank";
    ICHECK_EQ(intent->logical_rank, static_cast<int64_t>(intent->logical_shape.size()))
        << "TensorPlacementIntent " << intent->name
        << " logical_rank must match logical_shape";
    if (strategy != "interleaved") {
      ICHECK(!intent->shard_shape.empty())
          << "TensorPlacementIntent " << intent->name
          << " sharded placement requires shard_shape";
      ICHECK(!intent->shard_grid_shape.empty())
          << "TensorPlacementIntent " << intent->name
          << " sharded placement requires shard_grid_shape";
      ICHECK(str(intent->shard_orientation) == "row_major" ||
             str(intent->shard_orientation) == "col_major")
          << "TensorPlacementIntent " << intent->name
          << " has invalid shard_orientation " << intent->shard_orientation;
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
  if (!plan->dataflow_edges.empty()) {
    ICHECK(!plan->live_values.empty())
        << "SpatialPlan live_values must cover DataflowEdge producer values";
  }
  ICHECK_EQ(plan->live_value_edges.size(), plan->dataflow_edges.size())
      << "SpatialPlan live_value_edges must cover every DataflowEdge";
  ICHECK_EQ(plan->materialization_boundaries.size(), plan->dataflow_edges.size())
      << "SpatialPlan materialization_boundaries must cover every DataflowEdge";

  std::unordered_set<std::string> seen_live_value_names;
  std::unordered_set<std::string> seen_live_value_producer_subjects;
  std::unordered_set<std::string> seen_live_value_subject_versions;
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
    const std::string producer_subject_key =
        std::to_string(value->producer_unit_index) + "|" + str(value->subject);
    ICHECK(seen_live_value_producer_subjects.insert(producer_subject_key).second)
        << "duplicate LiveValue producer/subject " << value->producer_unit << " "
        << value->subject;
    ICHECK_GE(value->version_index, 0)
        << "LiveValue " << value->name << " requires non-negative version_index";
    const std::string subject_version_key = str(value->subject) + "|" +
                                            std::to_string(value->version_index);
    ICHECK(seen_live_value_subject_versions.insert(subject_version_key).second)
        << "duplicate LiveValue subject/version " << value->subject << " v"
        << value->version_index;
    ICHECK(!value->definition_kind.empty())
        << "LiveValue " << value->name << " requires definition_kind";
    ICHECK(IsOneOf(str(value->definition_kind),
                   {"external_input", "compute_write", "materialization_write", "phi",
                    "host_output"}))
        << "LiveValue " << value->name << " definition_kind has unsupported value "
        << value->definition_kind;
    ValidateNoTTNoun(str(value->definition_kind), "LiveValue definition_kind");
    ICHECK_GE(value->defining_access_region_index, -1)
        << "LiveValue " << value->name << " defining_access_region_index must be >= -1";
    if (value->defining_access_region_index >= 0) {
      ICHECK_LT(value->defining_access_region_index,
                static_cast<int64_t>(plan->access_regions.size()))
          << "LiveValue " << value->name << " defining_access_region_index out of bounds";
      const AccessRegion& region = plan->access_regions[value->defining_access_region_index];
      ICHECK_EQ(str(region->subject), str(value->subject))
          << "LiveValue " << value->name << " defining access subject must match";
      ICHECK_EQ(region->unit_index, value->producer_unit_index)
          << "LiveValue " << value->name << " defining access unit must match producer";
      ICHECK(AccessRegionWrites(region))
          << "LiveValue " << value->name << " defining access must be a write";
    }
    ICHECK_GE(value->defining_event_index, -1)
        << "LiveValue " << value->name << " defining_event_index must be >= -1";
    if (value->defining_event_index >= 0) {
      ICHECK_LT(value->defining_event_index, static_cast<int64_t>(plan->dataflow_edges.size()))
          << "LiveValue " << value->name << " defining_event_index out of bounds";
    }
    ICHECK(!value->value_role.empty()) << "LiveValue " << value->name << " requires value_role";
    ICHECK(IsOneOf(str(value->value_role),
                   {"fragment", "accumulator", "cast_source", "publish_source",
                    "consumer_input"}))
        << "LiveValue " << value->name << " value_role has unsupported value "
        << value->value_role;
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
    ICHECK(IsOneOf(str(live_edge->relation_kind), {"carry", "flow", "join", "materialize"}))
        << "LiveValueEdge " << live_edge->name << " relation_kind has unsupported value "
        << live_edge->relation_kind;
    ValidateNoTTNoun(str(live_edge->relation_kind), "LiveValueEdge relation_kind");
    ICHECK(!live_edge->use_kind.empty())
        << "LiveValueEdge " << live_edge->name << " requires use_kind";
    ICHECK(IsOneOf(str(live_edge->use_kind),
                   {"compute_consume", "materialization_consume", "transport_consume",
                    "host_output_consume"}))
        << "LiveValueEdge " << live_edge->name << " use_kind has unsupported value "
        << live_edge->use_kind;
    ValidateNoTTNoun(str(live_edge->use_kind), "LiveValueEdge use_kind");
    ICHECK_EQ(live_edge->source_version_index, source_live_value->version_index)
        << "LiveValueEdge " << live_edge->name
        << " source_version_index must match source LiveValue";
    ICHECK_GE(live_edge->target_version_index, 0)
        << "LiveValueEdge " << live_edge->name << " requires target_version_index";
    ICHECK_GE(live_edge->consumer_access_region_index, -1)
        << "LiveValueEdge " << live_edge->name
        << " consumer_access_region_index must be >= -1";
    if (live_edge->consumer_access_region_index >= 0) {
      ICHECK_LT(live_edge->consumer_access_region_index,
                static_cast<int64_t>(plan->access_regions.size()))
          << "LiveValueEdge " << live_edge->name
          << " consumer_access_region_index out of bounds";
      const AccessRegion& region = plan->access_regions[live_edge->consumer_access_region_index];
      ICHECK_EQ(str(region->subject), str(dataflow_edge->subject))
          << "LiveValueEdge " << live_edge->name << " consumer access subject must match";
      ICHECK_EQ(region->unit_index, live_edge->consumer_unit_index)
          << "LiveValueEdge " << live_edge->name << " consumer access unit must match";
      ICHECK(AccessRegionReads(region))
          << "LiveValueEdge " << live_edge->name << " consumer access must be a read";
    }
    if (live_edge->accepts_distributed_slice) {
      ICHECK_GE(live_edge->consumer_access_region_index, 0)
          << "LiveValueEdge " << live_edge->name
          << " distributed slice consumer requires AccessRegion evidence";
      ICHECK_GE(source_live_value->defining_access_region_index, 0)
          << "LiveValueEdge " << live_edge->name
          << " distributed slice producer requires AccessRegion evidence";
      const AccessRegion& producer_region =
          plan->access_regions[source_live_value->defining_access_region_index];
      const AccessRegion& consumer_region =
          plan->access_regions[live_edge->consumer_access_region_index];
      ICHECK(IsSliceCompatible(producer_region, consumer_region))
          << "LiveValueEdge " << live_edge->name
          << " distributed slice requires compatible producer/consumer AccessRegion";
    }
    ICHECK(live_edge->requires_full_logical_value || live_edge->accepts_distributed_slice)
        << "LiveValueEdge " << live_edge->name
        << " must either require full logical value or accept distributed slice";
    ICHECK(!(live_edge->requires_full_logical_value && live_edge->accepts_distributed_slice))
        << "LiveValueEdge " << live_edge->name
        << " must not require a full logical value and accept a distributed slice";
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
    ICHECK(!boundary->target_live_value.empty())
        << "MaterializationBoundary " << boundary->name << " requires target_live_value";
    ICHECK_GE(boundary->target_live_value_index, 0)
        << "MaterializationBoundary " << boundary->name << " requires target_live_value_index";
    ICHECK_LT(boundary->target_live_value_index, static_cast<int64_t>(plan->live_values.size()))
        << "MaterializationBoundary " << boundary->name << " target_live_value_index out of bounds";
    const LiveValue& target_live_value = plan->live_values[boundary->target_live_value_index];
    ICHECK_EQ(str(boundary->target_live_value), str(target_live_value->name))
        << "MaterializationBoundary " << boundary->name
        << " target_live_value must match target_live_value_index";
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
    if (str(live_edge->relation_kind) == "materialize") {
      ICHECK_EQ(target_live_value->producer_unit_index, live_edge->consumer_unit_index)
          << "MaterializationBoundary " << boundary->name
          << " target_live_value producer must match materialize consumer unit";
    }
    const DataflowEdge& dataflow_edge = plan->dataflow_edges[live_edge->dataflow_edge_index];
    ICHECK_EQ(live_edge->target_version_index, target_live_value->version_index)
        << "MaterializationBoundary " << boundary->name
        << " target_version_index must match target LiveValue";
    const bool crosses_phase = phase_index_by_unit[dataflow_edge->producer_unit_index] !=
                               phase_index_by_unit[dataflow_edge->consumer_unit_index];
    ICHECK(!boundary->required_visibility.empty())
        << "MaterializationBoundary " << boundary->name << " requires required_visibility";
    ICHECK(!boundary->logical_coverage.empty())
        << "MaterializationBoundary " << boundary->name << " requires logical_coverage";
    ICHECK(!boundary->phase_relation.empty())
        << "MaterializationBoundary " << boundary->name << " requires phase_relation";
    ICHECK(IsOneOf(str(boundary->required_visibility),
                   {"same_unit", "next_phase", "host_visible_output"}))
        << "MaterializationBoundary " << boundary->name
        << " required_visibility has unsupported value " << boundary->required_visibility;
    ICHECK(IsOneOf(str(boundary->logical_coverage),
                   {"full_logical_value", "distributed_slice", "row_slice", "grouped_slice"}))
        << "MaterializationBoundary " << boundary->name
        << " logical_coverage has unsupported value " << boundary->logical_coverage;
    ICHECK(IsOneOf(str(boundary->phase_relation), {"same_phase", "cross_phase"}))
        << "MaterializationBoundary " << boundary->name
        << " phase_relation has unsupported value " << boundary->phase_relation;
    ICHECK_GE(boundary->source_access_region_index, -1)
        << "MaterializationBoundary " << boundary->name
        << " source_access_region_index must be >= -1";
    ICHECK_GE(boundary->target_access_region_index, -1)
        << "MaterializationBoundary " << boundary->name
        << " target_access_region_index must be >= -1";
    if (boundary->source_access_region_index >= 0) {
      ICHECK_LT(boundary->source_access_region_index,
                static_cast<int64_t>(plan->access_regions.size()))
          << "MaterializationBoundary " << boundary->name
          << " source_access_region_index out of bounds";
      ICHECK_EQ(str(plan->access_regions[boundary->source_access_region_index]->subject),
                str(source_live_value->subject))
          << "MaterializationBoundary " << boundary->name
          << " source access subject must match source LiveValue";
    }
    if (boundary->target_access_region_index >= 0) {
      ICHECK_LT(boundary->target_access_region_index,
                static_cast<int64_t>(plan->access_regions.size()))
          << "MaterializationBoundary " << boundary->name
          << " target_access_region_index out of bounds";
      ICHECK_EQ(str(plan->access_regions[boundary->target_access_region_index]->subject),
                str(target_live_value->subject))
          << "MaterializationBoundary " << boundary->name
          << " target access subject must match target LiveValue";
    }
    ICHECK(!boundary->event_lifetime_kind.empty())
        << "MaterializationBoundary " << boundary->name << " requires event_lifetime_kind";
    ICHECK(IsOneOf(str(boundary->event_lifetime_kind),
                   {"single_event", "multi_event", "loop_carried"}))
        << "MaterializationBoundary " << boundary->name
        << " event_lifetime_kind has unsupported value " << boundary->event_lifetime_kind;
    ICHECK_GE(boundary->min_publish_pages, 1)
        << "MaterializationBoundary " << boundary->name << " requires min_publish_pages >= 1";
    ICHECK_GE(boundary->max_consume_pages, boundary->min_publish_pages)
        << "MaterializationBoundary " << boundary->name
        << " requires max_consume_pages >= min_publish_pages";
    if (str(boundary->event_lifetime_kind) == "loop_carried") {
      bool has_component_evidence = false;
      for (const DependenceComponent& component : plan->dependence_components) {
        const std::string component_kind = str(component->component_kind);
        if (component_kind != "carry_cycle" && component_kind != "reduction_cycle" &&
            component_kind != "recurrence") {
          continue;
        }
        for (const Integer& component_edge_index : component->edge_indices) {
          if (component_edge_index->value == live_edge->dataflow_edge_index) {
            has_component_evidence = true;
            break;
          }
        }
        if (has_component_evidence) {
          break;
        }
      }
      ICHECK(has_component_evidence)
          << "MaterializationBoundary " << boundary->name
          << " loop_carried lifetime requires DependenceComponent evidence";
    }
    ValidateNoTTNoun(str(boundary->required_visibility),
                     "MaterializationBoundary required_visibility");
    ValidateNoTTNoun(str(boundary->logical_coverage), "MaterializationBoundary logical_coverage");
    ValidateNoTTNoun(str(boundary->phase_relation), "MaterializationBoundary phase_relation");
    ValidateNoTTNoun(str(boundary->event_lifetime_kind),
                     "MaterializationBoundary event_lifetime_kind");
    if (live_edge->requires_full_logical_value) {
      ICHECK_EQ(str(boundary->logical_coverage), "full_logical_value")
          << "MaterializationBoundary " << boundary->name
          << " full-value consumer requires full_logical_value coverage";
    }
    if (str(boundary->logical_coverage) != "full_logical_value") {
      ICHECK(live_edge->accepts_distributed_slice)
          << "MaterializationBoundary " << boundary->name
          << " partial coverage requires a slice-capable LiveValueEdge";
    }
    ICHECK_EQ(str(boundary->phase_relation), crosses_phase ? "cross_phase" : "same_phase")
        << "MaterializationBoundary " << boundary->name
        << " phase_relation must match DataflowEdge phase membership";
    ICHECK_EQ(str(boundary->required_visibility), crosses_phase ? "next_phase" : "same_unit")
        << "MaterializationBoundary " << boundary->name
        << " required_visibility must match DataflowEdge phase membership";
    const std::string expected_lifetime =
        str(dataflow_edge->kind) == "carry" ? "loop_carried" : (crosses_phase ? "multi_event"
                                                                              : "single_event");
    ICHECK_EQ(str(boundary->event_lifetime_kind), expected_lifetime)
        << "MaterializationBoundary " << boundary->name
        << " event_lifetime_kind must match dataflow edge lifetime";
  }
}

void CheckSpatialPlan(const SpatialPlan& plan) {
  ICHECK(!plan->member_func.empty()) << "SpatialPlan requires member_func";
  ICHECK(plan->validated_hints.defined()) << "SpatialPlan requires ValidatedHintSet";
  ValidateExecutionUnits(plan);
  ValidateCompatibilityProjection(plan);
  const std::vector<int> phase_index_by_unit = BuildPhaseIndexByUnit(plan);
  ValidateAccessRegions(plan);
  ValidateDataflowEdges(plan, phase_index_by_unit);
  ValidateDependenceComponents(plan);
  ValidateLayoutSpecs(plan);
  ValidateTensorPlacementIntents(plan);
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
