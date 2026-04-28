/*!
 * \file blackhole_tile_compute_legalizer.cc
 * \brief Blackhole tile compute legalizer scaffold.
 */

#include "blackhole_tile_compute_legalizer.h"

#include "blackhole_tile_compute_patterns.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace {

std::string JoinRoles(const std::vector<std::string>& roles) {
  std::ostringstream os;
  for (size_t i = 0; i < roles.size(); ++i) {
    if (i != 0U) {
      os << ", ";
    }
    os << roles[i];
  }
  return os.str();
}

BlackholeTileLegalizationDiagnostic Legal(
    const std::string& operation_name) {
  return BlackholeTileLegalizationDiagnostic{
      BlackholeTileLegalizationAction::kLegal,
      operation_name,
      "",
  };
}

BlackholeTileLegalizationDiagnostic Reject(
    const std::string& operation_name, const std::string& reason) {
  return BlackholeTileLegalizationDiagnostic{
      BlackholeTileLegalizationAction::kReject,
      operation_name,
      reason,
  };
}

bool HasRole(const std::unordered_set<std::string>& roles, const std::string& role) {
  return roles.find(role) != roles.end();
}

}  // namespace

BlackholeTileLegalizationDiagnostic LegalizeBlackholeTileComputeSelection(
    const std::string& kind, const std::string& operation_name,
    const std::vector<std::string>& operand_roles) {
  const BlackholeTileComputePattern* pattern =
      FindBlackholeTileComputePattern(operation_name);
  if (pattern == nullptr) {
    return Reject(operation_name, "no leaf pattern covers operation");
  }
  if (pattern->result_kind != kind) {
    return Reject(operation_name, "kind " + kind + " does not match pattern result kind " +
                                      pattern->result_kind);
  }
  std::unordered_set<std::string> role_set;
  for (const std::string& role : operand_roles) {
    if (!role_set.insert(role).second) {
      return Reject(operation_name, "duplicate operand role " + role);
    }
  }
  for (const std::string& required_role : pattern->operand_roles) {
    if (!HasRole(role_set, required_role)) {
      return Reject(operation_name,
                    "missing operand role " + required_role + " in [" +
                        JoinRoles(operand_roles) + "]");
    }
  }
  return Legal(operation_name);
}

BlackholeTileLegalizationDiagnostic LegalizeBlackholeTileComputePlan(
    const TTComputeOpPlan& plan) {
  std::vector<std::string> roles;
  for (const TTComputeOperandBindingPlan& binding : plan->operand_bindings) {
    roles.push_back(static_cast<std::string>(binding->role));
  }
  return LegalizeBlackholeTileComputeSelection(
      static_cast<std::string>(plan->kind),
      static_cast<std::string>(plan->operation_name),
      roles);
}

void RequireLegalBlackholeTileComputeSelection(
    const std::string& kind, const std::string& operation_name,
    const std::vector<std::string>& operand_roles) {
  const BlackholeTileLegalizationDiagnostic diagnostic =
      LegalizeBlackholeTileComputeSelection(kind, operation_name, operand_roles);
  ICHECK(diagnostic.IsLegal())
      << "TileCompute legalizer rejected operation " << operation_name
      << ": " << diagnostic.reason;
}

void RequireLegalBlackholeTileComputePlan(const TTComputeOpPlan& plan) {
  const BlackholeTileLegalizationDiagnostic diagnostic =
      LegalizeBlackholeTileComputePlan(plan);
  ICHECK(diagnostic.IsLegal())
      << "TileCompute legalizer rejected operation " << plan->operation_name
      << ": " << diagnostic.reason;
}

}  // namespace tl
}  // namespace tvm
