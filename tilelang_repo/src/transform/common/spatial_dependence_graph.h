/*!
 * \file spatial_dependence_graph.h
 * \brief Graph algorithms for SpatialPlan dataflow dependencies.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_DEPENDENCE_GRAPH_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_DEPENDENCE_GRAPH_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spatial_plan.h"

namespace tvm {
namespace tl {

struct SpatialLocalValueFlowEvidence {
  std::string source_subject;
  std::string target_subject;
  int64_t unit_index{-1};
  bool accepts_distributed_slice{false};
};

struct SpatialLocalValueDependenceEdges {
  ffi::Array<DataflowEdge> dataflow_edges;
  std::unordered_map<std::string, std::string> target_subject_by_edge;
};

ffi::Array<ClosureBoundary> BuildClosureBoundariesFromAccessRegions(
    const ffi::Array<ExecutionUnit>& execution_units,
    const ffi::Array<AccessRegion>& access_regions);

SpatialLocalValueDependenceEdges BuildLocalValueDependenceEdges(
    const ffi::Array<ExecutionUnit>& execution_units,
    const std::vector<SpatialLocalValueFlowEvidence>& local_value_flows);

ffi::Array<DependenceComponent> BuildDependenceComponents(
    const ffi::Array<ExecutionUnit>& execution_units,
    const ffi::Array<DataflowEdge>& dataflow_edges);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_DEPENDENCE_GRAPH_H_
