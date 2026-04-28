/*!
 * \file blackhole_tile_compute_legalizer.h
 * \brief Blackhole tile compute legalizer scaffold.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_LEGALIZER_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_LEGALIZER_H_

#include "tt_target_program.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {

enum class BlackholeTileLegalizationAction {
  kLegal,
  kLower,
  kSplit,
  kPromoteDType,
  kMaterialize,
  kReject,
};

struct BlackholeTileLegalizationDiagnostic {
  BlackholeTileLegalizationAction action{BlackholeTileLegalizationAction::kReject};
  std::string operation_name;
  std::string reason;

  bool IsLegal() const { return action == BlackholeTileLegalizationAction::kLegal; }
};

BlackholeTileLegalizationDiagnostic LegalizeBlackholeTileComputeSelection(
    const std::string& kind, const std::string& operation_name,
    const std::vector<std::string>& operand_roles);

BlackholeTileLegalizationDiagnostic LegalizeBlackholeTileComputePlan(
    const TTComputeOpPlan& plan);

void RequireLegalBlackholeTileComputeSelection(
    const std::string& kind, const std::string& operation_name,
    const std::vector<std::string>& operand_roles);

void RequireLegalBlackholeTileComputePlan(const TTComputeOpPlan& plan);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_LEGALIZER_H_
