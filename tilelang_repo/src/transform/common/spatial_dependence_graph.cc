/*!
 * \file spatial_dependence_graph.cc
 * \brief Graph algorithms for SpatialPlan dataflow dependencies.
 */

#include "spatial_dependence_graph.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "blackhole_utils.h"
#include "spatial_analysis.h"

namespace tvm {
namespace tl {

namespace {

using tvm::Integer;
using tvm::ffi::Array;
using tvm::ffi::String;

Array<Integer> ToIntegerArray(const std::vector<int>& values) {
  Array<Integer> result;
  result.reserve(values.size());
  for (int value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

bool IsCycleEdgeKind(const std::string& kind) {
  return kind == "carry" || kind == "reduction";
}

bool AccessesRead(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "read" || access_kind == "read_write" || access_kind == "reduce_read";
}

bool AccessesWrite(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "write" || access_kind == "read_write" || access_kind == "reduce_write";
}

bool IsCarryEdge(const DataflowEdge& edge) { return str(edge->kind) == "carry"; }

bool IsReductionEdge(const DataflowEdge& edge) { return str(edge->kind) == "reduction"; }

void AppendUnique(std::vector<std::string>* values, const std::string& value) {
  ICHECK(values != nullptr);
  if (value.empty()) {
    return;
  }
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

std::vector<std::vector<int>> ComputeStronglyConnectedComponents(
    const std::vector<std::vector<int>>& adjacency) {
  const int vertex_count = static_cast<int>(adjacency.size());
  std::vector<int> index_by_vertex(vertex_count, -1);
  std::vector<int> lowlink(vertex_count, -1);
  std::vector<int> stack;
  std::vector<bool> on_stack(vertex_count, false);
  std::vector<std::vector<int>> components;
  int next_index = 0;

  std::function<void(int)> visit = [&](int vertex) {
    index_by_vertex[vertex] = next_index;
    lowlink[vertex] = next_index;
    ++next_index;
    stack.push_back(vertex);
    on_stack[vertex] = true;

    for (int successor : adjacency[vertex]) {
      if (index_by_vertex[successor] == -1) {
        visit(successor);
        lowlink[vertex] = std::min(lowlink[vertex], lowlink[successor]);
      } else if (on_stack[successor]) {
        lowlink[vertex] = std::min(lowlink[vertex], index_by_vertex[successor]);
      }
    }

    if (lowlink[vertex] != index_by_vertex[vertex]) {
      return;
    }

    std::vector<int> component;
    while (true) {
      const int member = stack.back();
      stack.pop_back();
      on_stack[member] = false;
      component.push_back(member);
      if (member == vertex) {
        break;
      }
    }
    std::sort(component.begin(), component.end());
    components.push_back(std::move(component));
  };

  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    if (index_by_vertex[vertex] == -1) {
      visit(vertex);
    }
  }
  std::sort(components.begin(), components.end(),
            [](const std::vector<int>& lhs, const std::vector<int>& rhs) {
              return lhs.front() < rhs.front();
            });
  return components;
}

bool ContainsUnit(const std::vector<int>& component, int unit_index) {
  return std::binary_search(component.begin(), component.end(), unit_index);
}

}  // namespace

Array<ClosureBoundary> BuildClosureBoundariesFromAccessRegions(
    const Array<ExecutionUnit>& execution_units, const Array<AccessRegion>& access_regions) {
  const int unit_count = static_cast<int>(execution_units.size());
  std::vector<std::vector<std::string>> reads_by_unit(unit_count);
  std::vector<std::vector<std::string>> writes_by_unit(unit_count);

  for (const AccessRegion& region : access_regions) {
    if (!region.defined() || region->unit_index < 0 || region->unit_index >= unit_count ||
        region->subject.empty()) {
      continue;
    }
    const int unit_index = static_cast<int>(region->unit_index);
    const std::string subject = str(region->subject);
    if (AccessesRead(region)) {
      AppendUnique(&reads_by_unit[unit_index], subject);
    }
    if (AccessesWrite(region)) {
      AppendUnique(&writes_by_unit[unit_index], subject);
    }
  }

  Array<ClosureBoundary> boundaries;
  std::unordered_set<std::string> emitted;
  std::unordered_map<std::string, std::vector<int>> producer_indices_by_subject;

  for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
    const ExecutionUnit& unit = execution_units[unit_index];
    std::unordered_set<std::string> read_set(reads_by_unit[unit_index].begin(),
                                             reads_by_unit[unit_index].end());

    for (const std::string& subject : writes_by_unit[unit_index]) {
      if (!read_set.count(subject)) {
        continue;
      }
      const std::string key =
          "carry|" + subject + "|" + std::to_string(unit_index) + "|" +
          std::to_string(unit_index);
      if (!emitted.insert(key).second) {
        continue;
      }
      boundaries.push_back(ClosureBoundary(
          String("carry_" + subject + "_" + std::to_string(unit_index)), String("carry"),
          unit->name, unit->name, unit_index, unit_index, String(subject),
          MakeTraits({"self_edge"}), MakeAnchors("closure_boundary", key)));
    }

    for (const std::string& subject : reads_by_unit[unit_index]) {
      auto it = producer_indices_by_subject.find(subject);
      if (it == producer_indices_by_subject.end() || it->second.empty()) {
        continue;
      }
      const std::vector<int>& producers = it->second;
      const bool is_join = producers.size() > 1;
      const std::string boundary_kind = is_join ? "join" : "flow";
      for (int producer_index : producers) {
        const std::string key = boundary_kind + "|" + subject + "|" +
                                std::to_string(producer_index) + "|" +
                                std::to_string(unit_index);
        if (!emitted.insert(key).second) {
          continue;
        }
        boundaries.push_back(ClosureBoundary(
            String(boundary_kind + "_" + subject + "_" + std::to_string(producer_index) + "_" +
                   std::to_string(unit_index)),
            String(boundary_kind), execution_units[producer_index]->name, unit->name,
            producer_index, unit_index, String(subject),
            is_join ? MakeTraits({"multi_producer"}) : Array<String>{},
            MakeAnchors("closure_boundary", key)));
      }
    }

    for (const std::string& subject : writes_by_unit[unit_index]) {
      producer_indices_by_subject[subject].push_back(unit_index);
    }
  }

  return boundaries;
}

SpatialLocalValueDependenceEdges BuildLocalValueDependenceEdges(
    const Array<ExecutionUnit>& execution_units,
    const std::vector<SpatialLocalValueFlowEvidence>& local_value_flows) {
  SpatialLocalValueDependenceEdges result;
  std::unordered_set<std::string> emitted;
  for (const SpatialLocalValueFlowEvidence& flow : local_value_flows) {
    if (flow.source_subject.empty() || flow.target_subject.empty() ||
        flow.source_subject == flow.target_subject || flow.unit_index < 0 ||
        flow.unit_index >= static_cast<int64_t>(execution_units.size())) {
      continue;
    }
    const std::string key = flow.source_subject + "|" + flow.target_subject + "|" +
                            std::to_string(flow.unit_index);
    if (!emitted.insert(key).second) {
      continue;
    }
    const ExecutionUnit& unit = execution_units[flow.unit_index];
    const std::string name = "materialize_" + flow.source_subject + "_to_" +
                             flow.target_subject + "_" + std::to_string(flow.unit_index);
    std::vector<std::string> traits{"same_unit"};
    if (flow.accepts_distributed_slice) {
      AppendUnique(&traits, "distributed_slice_consumer");
    }
    result.dataflow_edges.push_back(DataflowEdge(
        String(name), String("materialize"), unit->name, unit->name, flow.unit_index,
        flow.unit_index, String(flow.source_subject), false, ToStringArray(traits),
        MakeAnchors("dataflow_edge", name)));
    result.target_subject_by_edge.emplace(name, flow.target_subject);
  }
  return result;
}

Array<DependenceComponent> BuildDependenceComponents(
    const Array<ExecutionUnit>& execution_units, const Array<DataflowEdge>& dataflow_edges) {
  const int unit_count = static_cast<int>(execution_units.size());
  std::vector<std::unordered_set<int>> adjacency_sets(unit_count);
  std::vector<bool> has_cycle_self_edge(unit_count, false);

  for (const DataflowEdge& edge : dataflow_edges) {
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
        edge->producer_unit_index >= unit_count || edge->consumer_unit_index >= unit_count) {
      continue;
    }
    const int producer = static_cast<int>(edge->producer_unit_index);
    const int consumer = static_cast<int>(edge->consumer_unit_index);
    if (producer == consumer) {
      if (IsCycleEdgeKind(str(edge->kind))) {
        has_cycle_self_edge[producer] = true;
      }
      continue;
    }
    adjacency_sets[producer].insert(consumer);
  }

  std::vector<std::vector<int>> adjacency(unit_count);
  for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
    adjacency[unit_index].assign(adjacency_sets[unit_index].begin(),
                                 adjacency_sets[unit_index].end());
    std::sort(adjacency[unit_index].begin(), adjacency[unit_index].end());
  }

  Array<DependenceComponent> dependence_components;
  for (const std::vector<int>& component : ComputeStronglyConnectedComponents(adjacency)) {
    bool is_recurrent_component = component.size() > 1;
    if (component.size() == 1 && has_cycle_self_edge[component.front()]) {
      is_recurrent_component = true;
    }
    if (!is_recurrent_component) {
      continue;
    }

    std::vector<int> edge_indices;
    std::vector<std::string> subjects;
    bool has_carry = false;
    bool has_reduction = false;
    for (int edge_index = 0; edge_index < static_cast<int>(dataflow_edges.size());
         ++edge_index) {
      const DataflowEdge& edge = dataflow_edges[edge_index];
      if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
          edge->producer_unit_index >= unit_count || edge->consumer_unit_index >= unit_count) {
        continue;
      }
      const int producer = static_cast<int>(edge->producer_unit_index);
      const int consumer = static_cast<int>(edge->consumer_unit_index);
      if (!ContainsUnit(component, producer) || !ContainsUnit(component, consumer)) {
        continue;
      }
      if (component.size() == 1 && producer == consumer && !IsCycleEdgeKind(str(edge->kind))) {
        continue;
      }

      edge_indices.push_back(edge_index);
      AppendUnique(&subjects, str(edge->subject));
      has_carry = has_carry || IsCarryEdge(edge);
      has_reduction = has_reduction || IsReductionEdge(edge);
    }
    if (edge_indices.empty()) {
      continue;
    }
    std::sort(subjects.begin(), subjects.end());

    const std::string component_kind =
        has_reduction ? "reduction_cycle" : (has_carry ? "carry_cycle" : "recurrence");
    const std::string name = "dependence_component_" + component_kind + "_" +
                             std::to_string(dependence_components.size());
    dependence_components.push_back(DependenceComponent(
        String(name), String(component_kind), ToIntegerArray(component),
        ToIntegerArray(edge_indices), ToStringArray(subjects),
        MakeAnchors("dependence_component", name)));
  }

  return dependence_components;
}

}  // namespace tl
}  // namespace tvm
