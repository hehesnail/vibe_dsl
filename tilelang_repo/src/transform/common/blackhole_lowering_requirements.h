/*!
 * \file blackhole_lowering_requirements.h
 * \brief Derive typed Blackhole lowering support facts directly from SpatialPlan and current TIR.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_

#include <tvm/tir/function.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../blackhole_cb_common.h"
#include "spatial_plan.h"

namespace tvm {
namespace tl {

enum class BlackholeBufferFlowEventKind {
  kWrite,
  kComputeConsume,
  kTransportConsume,
  kReference,
};

struct BlackholeBufferFlowEvent {
  int order_index = -1;
  BlackholeBufferFlowEventKind kind = BlackholeBufferFlowEventKind::kReference;
};

struct BlackholeBufferFlowFact {
  std::string buffer;
  std::string scope;
  CBFlowClass flow_class = CBFlowClass::kState;
  int publish_pages_per_event = 0;
  int consume_pages_per_event = 0;
  std::vector<BlackholeBufferFlowEvent> events;
};

struct BlackholeBufferMaterializationFact {
  std::string kind;
  std::string target_buffer;
  std::string scope;
  std::string materialization_kind;
  std::string bridge_kind;
  std::string value_role;
  std::string merge_kind;
  std::string execution_protocol;
  std::string result_live_form;
  std::string source_buffer;
  int64_t logical_row_width = -1;
  int64_t logical_element_count = -1;
  std::string spatial_materialization_boundary;
  int64_t spatial_materialization_boundary_index = -1;
  std::string spatial_live_value_edge;
  int64_t spatial_live_value_edge_index = -1;
  std::string source_live_value;
  int64_t source_live_value_index = -1;
  std::string target_live_value;
  int64_t target_live_value_index = -1;
};

struct BlackholeLoweringSupportFacts {
  std::vector<BlackholeBufferMaterializationFact> buffer_materialization_facts;
  std::vector<BlackholeBufferFlowFact> buffer_flow_facts;
};

TVM_DLL BlackholeLoweringSupportFacts CollectBlackholeLoweringSupportFacts(
    const tvm::tir::PrimFunc& func, const SpatialPlan& plan);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
