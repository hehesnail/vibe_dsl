/*!
 * \file tt_live_form_solver.cc
 * \brief Typed live-form transfer decisions for TTProgram planning.
 */

#include "tt_live_form_solver.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "companion_base.h"

namespace tvm {
namespace tl {

namespace {

constexpr const char* kThreadDistributedTopology = "thread_distributed";
constexpr const char* kThreadDistributedSlice = "thread_distributed_slice";
constexpr const char* kCBMaterializedTile = "cb_materialized_tile";
constexpr const char* kProducerThreadLaneOwnership = "producer_thread_lane";
constexpr const char* kMaterializedCBPagesOwnership = "materialized_cb_pages";

enum class TTLiveFormLatticeKind {
  kBottom,
  kFragment,
  kAccumulator,
  kExactCBSingleEvent,
  kExactCBMultiEvent,
  kHostVisible,
  kConflict,
  kUnsupported,
};

struct TTLiveFormWorkItem {
  int64_t boundary_index{-1};
};

TTLiveFormLatticeKind JoinLiveFormState(TTLiveFormLatticeKind lhs,
                                        TTLiveFormLatticeKind rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == TTLiveFormLatticeKind::kBottom) {
    return rhs;
  }
  if (rhs == TTLiveFormLatticeKind::kBottom) {
    return lhs;
  }
  if (lhs == TTLiveFormLatticeKind::kUnsupported ||
      rhs == TTLiveFormLatticeKind::kUnsupported) {
    return TTLiveFormLatticeKind::kUnsupported;
  }
  if (lhs == TTLiveFormLatticeKind::kConflict || rhs == TTLiveFormLatticeKind::kConflict) {
    return TTLiveFormLatticeKind::kConflict;
  }
  const bool lhs_exact = lhs == TTLiveFormLatticeKind::kExactCBSingleEvent ||
                         lhs == TTLiveFormLatticeKind::kExactCBMultiEvent;
  const bool rhs_exact = rhs == TTLiveFormLatticeKind::kExactCBSingleEvent ||
                         rhs == TTLiveFormLatticeKind::kExactCBMultiEvent;
  if (lhs_exact && rhs_exact) {
    return TTLiveFormLatticeKind::kExactCBMultiEvent;
  }
  return TTLiveFormLatticeKind::kConflict;
}

TTLiveFormLatticeKind ApplyBoundaryTransfer(const TTLiveFormBoundaryRequest& boundary,
                                            TTLiveFormLatticeKind source_state) {
  if (source_state == TTLiveFormLatticeKind::kUnsupported ||
      source_state == TTLiveFormLatticeKind::kConflict) {
    return source_state;
  }
  if (boundary.min_publish_pages < 1 ||
      boundary.max_consume_pages < boundary.min_publish_pages) {
    return TTLiveFormLatticeKind::kUnsupported;
  }
  if (boundary.logical_coverage != "distributed_slice" &&
      boundary.logical_coverage != "full_logical_value") {
    return TTLiveFormLatticeKind::kUnsupported;
  }
  if (boundary.event_lifetime_kind == "single_event") {
    return TTLiveFormLatticeKind::kExactCBSingleEvent;
  }
  if (boundary.event_lifetime_kind == "multi_event" ||
      boundary.event_lifetime_kind == "loop_carried") {
    return TTLiveFormLatticeKind::kExactCBMultiEvent;
  }
  return TTLiveFormLatticeKind::kUnsupported;
}

std::string OwnershipKindForState(TTLiveFormLatticeKind state) {
  switch (state) {
    case TTLiveFormLatticeKind::kExactCBSingleEvent:
      return "materialized_cb_pages_single_event";
    case TTLiveFormLatticeKind::kExactCBMultiEvent:
      return "materialized_cb_pages_multi_event";
    default:
      return kMaterializedCBPagesOwnership;
  }
}

std::vector<TTLiveFormBoundaryRequest> BuildBoundaryGraph(
    const TTLiveFormSolverRequest& request) {
  if (!request.validated_live_boundaries.empty()) {
    return request.validated_live_boundaries;
  }
  return {TTLiveFormBoundaryRequest{"selected_boundary",
                                    request.selected_boundary_index,
                                    request.source_spatial_live_value,
                                    request.source_spatial_live_value_index,
                                    request.target_spatial_live_value,
                                    request.target_spatial_live_value_index,
                                    request.boundary_event_lifetime_kind,
                                    request.boundary_logical_coverage,
                                    request.min_publish_pages,
                                    request.max_consume_pages}};
}

const TTLiveFormBoundaryRequest& SelectBoundary(
    const std::vector<TTLiveFormBoundaryRequest>& boundaries,
    const TTLiveFormSolverRequest& request) {
  for (const TTLiveFormBoundaryRequest& boundary : boundaries) {
    if (request.selected_boundary_index >= 0 &&
        boundary.index == request.selected_boundary_index) {
      return boundary;
    }
  }
  for (const TTLiveFormBoundaryRequest& boundary : boundaries) {
    if (boundary.source_spatial_live_value_index == request.source_spatial_live_value_index &&
        boundary.target_spatial_live_value_index == request.target_spatial_live_value_index) {
      return boundary;
    }
  }
  ICHECK(!boundaries.empty()) << "TT live-form solver requires a boundary graph";
  return boundaries.front();
}

TTLiveFormLatticeKind SolveBoundaryGraph(
    const std::vector<TTLiveFormBoundaryRequest>& boundaries,
    const TTLiveFormBoundaryRequest& selected_boundary) {
  std::unordered_map<int64_t, TTLiveFormLatticeKind> state_by_live_value;
  std::unordered_map<int64_t, std::vector<size_t>> outgoing_by_source;
  std::vector<TTLiveFormWorkItem> worklist;
  for (size_t i = 0; i < boundaries.size(); ++i) {
    const TTLiveFormBoundaryRequest& boundary = boundaries[i];
    outgoing_by_source[boundary.source_spatial_live_value_index].push_back(i);
    worklist.push_back(TTLiveFormWorkItem{boundary.index});
  }
  state_by_live_value[selected_boundary.source_spatial_live_value_index] =
      TTLiveFormLatticeKind::kFragment;

  auto state_or_bottom = [&](int64_t live_value_index) {
    auto it = state_by_live_value.find(live_value_index);
    return it == state_by_live_value.end() ? TTLiveFormLatticeKind::kBottom : it->second;
  };
  auto push_outgoing = [&](int64_t live_value_index) {
    auto it = outgoing_by_source.find(live_value_index);
    if (it == outgoing_by_source.end()) {
      return;
    }
    for (size_t boundary_position : it->second) {
      worklist.push_back(TTLiveFormWorkItem{boundaries[boundary_position].index});
    }
  };
  auto find_boundary = [&](int64_t boundary_index) -> const TTLiveFormBoundaryRequest* {
    for (const TTLiveFormBoundaryRequest& boundary : boundaries) {
      if (boundary.index == boundary_index) {
        return &boundary;
      }
    }
    return nullptr;
  };

  size_t cursor = 0;
  while (cursor < worklist.size()) {
    const TTLiveFormBoundaryRequest* boundary = find_boundary(worklist[cursor++].boundary_index);
    if (boundary == nullptr) {
      continue;
    }
    const TTLiveFormLatticeKind source_state =
        state_or_bottom(boundary->source_spatial_live_value_index);
    if (source_state == TTLiveFormLatticeKind::kBottom) {
      continue;
    }
    if (boundary->source_spatial_live_value_index ==
        boundary->target_spatial_live_value_index) {
      continue;
    }
    const TTLiveFormLatticeKind transfer = ApplyBoundaryTransfer(*boundary, source_state);
    const TTLiveFormLatticeKind old_target =
        state_or_bottom(boundary->target_spatial_live_value_index);
    const TTLiveFormLatticeKind joined = JoinLiveFormState(old_target, transfer);
    if (joined != old_target) {
      state_by_live_value[boundary->target_spatial_live_value_index] = joined;
      push_outgoing(boundary->target_spatial_live_value_index);
    }
  }
  TTLiveFormLatticeKind selected_state =
      state_or_bottom(selected_boundary.target_spatial_live_value_index);
  if (selected_state == TTLiveFormLatticeKind::kBottom) {
    selected_state =
        ApplyBoundaryTransfer(selected_boundary, TTLiveFormLatticeKind::kFragment);
  }
  return selected_state;
}

}  // namespace

TTLiveFormSolverResult SolveFragmentCastLiveFormTransition(
    const TTLiveFormSolverRequest& request) {
  ICHECK(!request.source_logical_value.empty())
      << "TT live-form solver requires source logical value";
  ICHECK(!request.target_logical_value.empty())
      << "TT live-form solver requires target logical value";
  ICHECK(!request.source_spatial_live_value.empty())
      << "TT live-form solver requires source SpatialPlan LiveValue";
  ICHECK_GE(request.source_spatial_live_value_index, 0)
      << "TT live-form solver requires source SpatialPlan LiveValue index";
  ICHECK(!request.target_spatial_live_value.empty())
      << "TT live-form solver requires target SpatialPlan LiveValue";
  ICHECK_GE(request.target_spatial_live_value_index, 0)
      << "TT live-form solver requires target SpatialPlan LiveValue index";
  ICHECK_GT(request.logical_element_count, 0)
      << "TT live-form solver requires positive logical element count";
  ICHECK(!request.boundary_event_lifetime_kind.empty())
      << "TT live-form solver requires boundary event lifetime";
  ICHECK(!request.boundary_logical_coverage.empty())
      << "TT live-form solver requires boundary logical coverage";
  ICHECK(request.boundary_logical_coverage == "distributed_slice" ||
         request.boundary_logical_coverage == "full_logical_value")
      << "TT live-form solver unsupported boundary logical coverage "
      << request.boundary_logical_coverage;
  ICHECK_GE(request.min_publish_pages, 1)
      << "TT live-form solver requires bounded publish pages";
  ICHECK_GE(request.max_consume_pages, request.min_publish_pages)
      << "TT live-form solver requires consume pages to cover publish pages";

  const std::vector<TTLiveFormBoundaryRequest> boundaries = BuildBoundaryGraph(request);
  const TTLiveFormBoundaryRequest& selected_boundary = SelectBoundary(boundaries, request);
  ICHECK_EQ(selected_boundary.source_spatial_live_value_index,
            request.source_spatial_live_value_index)
      << "TT live-form solver selected boundary source must match request";
  ICHECK_EQ(selected_boundary.target_spatial_live_value_index,
            request.target_spatial_live_value_index)
      << "TT live-form solver selected boundary target must match request";
  const TTLiveFormLatticeKind target_state =
      SolveBoundaryGraph(boundaries, selected_boundary);
  ICHECK(target_state != TTLiveFormLatticeKind::kUnsupported &&
         target_state != TTLiveFormLatticeKind::kConflict)
      << "TT live-form solver rejected selected boundary " << selected_boundary.name;

  TTLiveFormSolverResult result;
  result.source_value = TTLiveFormValueDecision{
      request.source_logical_value,
      request.source_spatial_live_value,
      request.source_spatial_live_value_index,
      kThreadDistributedSlice,
      kThreadDistributedTopology,
      request.source_local_extent,
      request.logical_element_count,
      kProducerThreadLaneOwnership,
  };
  result.target_value = TTLiveFormValueDecision{
      request.target_logical_value,
      request.target_spatial_live_value,
      request.target_spatial_live_value_index,
      kCBMaterializedTile,
      kThreadDistributedTopology,
      request.target_local_extent,
      request.logical_element_count,
      OwnershipKindForState(target_state),
  };
  result.materialization = TTLiveFormMaterializationDecision{
      "live_form_" + request.source_logical_value,
      "live_form_" + request.target_logical_value,
      request.target_logical_value,
      request.bridge_kind,
      request.materialization_kind,
      buffer_materialization::kCBRepublish,
      request.publication_protocol,
  };
  const bool accepts_distributed_slice = request.boundary_logical_coverage == "distributed_slice";
  result.consumer = TTLiveFormConsumerDecision{
      accepts_distributed_slice,
      !accepts_distributed_slice,
  };
  return result;
}

}  // namespace tl
}  // namespace tvm
