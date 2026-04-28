/*!
 * \file blackhole_tile_compute_covering.h
 * \brief Local Blackhole tile compute pattern covering selection.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_COVERING_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_COVERING_H_

#include "blackhole_tile_compute_patterns.h"

#include <tvm/ffi/container/map.h>
#include <tvm/tir/function.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

struct BlackholeTileComputeCoveringDecision {
  bool selected{false};
  std::string selection_kind;
  std::string pattern_name;
  std::string operation_name;
  std::string result_kind;
  std::vector<std::string> operand_roles;
  std::string selected_output;
  std::optional<BlackholeTileComputeSourceEmitterKind> source_emitter;
  std::string materialization_policy;
  int64_t cost{0};
  std::string reject_reason;
};

BlackholeTileComputeCoveringDecision SelectBlackholeTileComputeCovering(
    const std::string& operation_name);

ffi::Map<ffi::String, ffi::Any> EncodeBlackholeTileComputeCoveringDecision(
    const BlackholeTileComputeCoveringDecision& decision);

ffi::Map<ffi::String, ffi::Any> SelectBlackholeTileComputeDAGCoveringDiagnostic(
    const tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_COVERING_H_
