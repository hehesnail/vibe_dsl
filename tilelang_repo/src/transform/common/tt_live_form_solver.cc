/*!
 * \file tt_live_form_solver.cc
 * \brief Typed live-form transfer decisions for TTProgram planning.
 */

#include "tt_live_form_solver.h"

#include <tvm/runtime/logging.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

namespace {

constexpr const char* kThreadDistributedTopology = "thread_distributed";
constexpr const char* kThreadDistributedSlice = "thread_distributed_slice";
constexpr const char* kCBMaterializedTile = "cb_materialized_tile";
constexpr const char* kProducerThreadLaneOwnership = "producer_thread_lane";
constexpr const char* kMaterializedCBPagesOwnership = "materialized_cb_pages";

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
  ICHECK_GE(request.min_publish_pages, 1)
      << "TT live-form solver requires bounded publish pages";
  ICHECK_GE(request.max_consume_pages, request.min_publish_pages)
      << "TT live-form solver requires consume pages to cover publish pages";

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
      kMaterializedCBPagesOwnership,
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
  result.consumer = TTLiveFormConsumerDecision{/*accepts_distributed_slice=*/true,
                                              /*requires_full_logical_tile=*/false};
  return result;
}

}  // namespace tl
}  // namespace tvm
