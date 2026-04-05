/*!
 * \file semantic_refinement_rules.h
 * \brief Centralized legality rules for typed semantic refinement.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_REFINEMENT_RULES_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_REFINEMENT_RULES_H_

#include "semantic_vocab.h"

namespace tvm {
namespace tl {
namespace semantic {

TVM_DLL bool RelationAxisRequiresStateAnchor(WitnessFactAxis axis);
TVM_DLL bool RelationAxisRequiresUpdateAnchor(WitnessFactAxis axis);
TVM_DLL bool RelationAxisCompatibleWithLawKind(WitnessFactAxis axis, UpdateLawKind law_kind);
TVM_DLL BindingKind DefaultBindingKindForRelation(WitnessFactAxis axis);
TVM_DLL bool BindingKindCompatibleWithRelation(WitnessFactAxis axis, BindingKind binding_kind);
TVM_DLL bool ContractModeRequiresBodyHash(ContractMode mode);
TVM_DLL bool ContractModeRequiresRebindEpoch(ContractMode mode);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_REFINEMENT_RULES_H_
