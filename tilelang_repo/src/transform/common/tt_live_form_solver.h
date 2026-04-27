/*!
 * \file tt_live_form_solver.h
 * \brief Typed live-form transfer decisions for TTProgram planning.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_TT_LIVE_FORM_SOLVER_H_
#define TVM_TL_TRANSFORM_COMMON_TT_LIVE_FORM_SOLVER_H_

#include <cstdint>
#include <string>

namespace tvm {
namespace tl {

struct TTLiveFormValueDecision {
  std::string logical_value;
  std::string spatial_live_value;
  int64_t spatial_live_value_index{-1};
  std::string physical_form;
  std::string execution_topology;
  int64_t physical_local_extent{0};
  int64_t logical_element_count{0};
  std::string ownership_kind;
};

struct TTLiveFormMaterializationDecision {
  std::string source_live_form;
  std::string produced_live_form;
  std::string target_buffer;
  std::string bridge_kind;
  std::string materialization_kind;
  std::string materialization_protocol;
  std::string publication_protocol;
};

struct TTLiveFormConsumerDecision {
  bool accepts_distributed_slice{false};
  bool requires_full_logical_tile{true};
};

struct TTLiveFormSolverRequest {
  std::string source_logical_value;
  std::string target_logical_value;
  std::string source_spatial_live_value;
  int64_t source_spatial_live_value_index{-1};
  std::string target_spatial_live_value;
  int64_t target_spatial_live_value_index{-1};
  int64_t source_local_extent{0};
  int64_t target_local_extent{0};
  int64_t logical_element_count{0};
  std::string boundary_event_lifetime_kind;
  int64_t min_publish_pages{0};
  int64_t max_consume_pages{0};
  std::string bridge_kind;
  std::string materialization_kind;
  std::string publication_protocol;
};

struct TTLiveFormSolverResult {
  TTLiveFormValueDecision source_value;
  TTLiveFormValueDecision target_value;
  TTLiveFormMaterializationDecision materialization;
  TTLiveFormConsumerDecision consumer;
};

TTLiveFormSolverResult SolveFragmentCastLiveFormTransition(
    const TTLiveFormSolverRequest& request);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_TT_LIVE_FORM_SOLVER_H_
