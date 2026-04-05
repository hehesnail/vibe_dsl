/*!
 * \file semantic_refinement_rules.cc
 * \brief Centralized legality rules for typed semantic refinement.
 */

#include "semantic_refinement_rules.h"

#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {
namespace semantic {

bool RelationAxisRequiresStateAnchor(WitnessFactAxis axis) {
  return axis == WitnessFactAxis::kDerivesIndexFrom;
}

bool RelationAxisRequiresUpdateAnchor(WitnessFactAxis axis) {
  return axis == WitnessFactAxis::kCompanion || axis == WitnessFactAxis::kCarriedFrom;
}

bool RelationAxisCompatibleWithLawKind(WitnessFactAxis axis, UpdateLawKind law_kind) {
  switch (axis) {
    case WitnessFactAxis::kCompanion:
      return law_kind == UpdateLawKind::kSelect;
    case WitnessFactAxis::kCarriedFrom:
      return law_kind == UpdateLawKind::kRecurrence;
    default:
      return true;
  }
}

BindingKind DefaultBindingKindForRelation(WitnessFactAxis axis) {
  switch (axis) {
    case WitnessFactAxis::kCompanion:
      return BindingKind::kPairedValueState;
    case WitnessFactAxis::kCarriedFrom:
      return BindingKind::kRecurrenceSourceState;
    default:
      LOG(FATAL) << "No default binding kind for relation axis " << ToString(axis);
      return BindingKind::kTargetState;
  }
}

bool BindingKindCompatibleWithRelation(WitnessFactAxis axis, BindingKind binding_kind) {
  switch (axis) {
    case WitnessFactAxis::kCompanion:
      return binding_kind == BindingKind::kPairedValueState;
    case WitnessFactAxis::kCarriedFrom:
      return binding_kind == BindingKind::kRecurrenceSourceState;
    default:
      return true;
  }
}

bool ContractModeRequiresBodyHash(ContractMode mode) {
  return mode == ContractMode::kPreserve;
}

bool ContractModeRequiresRebindEpoch(ContractMode mode) {
  return mode == ContractMode::kTypedRebind;
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
