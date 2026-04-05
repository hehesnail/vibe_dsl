/*!
 * \file semantic_vocab.h
 * \brief Closed semantic vocabulary for Phase A witness/core contracts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_VOCAB_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_VOCAB_H_

#include <tvm/runtime/base.h>

#include <optional>
#include <string>

namespace tvm {
namespace tl {
namespace semantic {

enum class WitnessSubjectKind {
  kDomain,
  kState,
  kUpdate,
  kAccess,
  kRelation,
  kBoundary,
};

enum class WitnessFactAxis {
  kRole,
  kIdentity,
  kLifetime,
  kLawFamily,
  kSourceSet,
  kOrdering,
  kBoundary,
  kIndirection,
  kSelectionContract,
  kDistributionHint,
  kCompanion,
  kDerivesIndexFrom,
  kFeedsUpdate,
  kCarriedFrom,
  kSemanticBoundary,
  kOrderedRegion,
};

enum class StateRole {
  kCarry,
  kReductionAccumulator,
  kSelectionState,
  kIndexState,
  kTransient,
};

enum class UpdateLawKind {
  kMap,
  kReduce,
  kSelect,
  kRecurrence,
};

enum class SupplementKind {
  kStateIdentity,
  kAccessTrait,
  kUpdateLawTrait,
  kSemanticBoundary,
};

enum class ContractMode {
  kPreserve,
  kTypedRebind,
  kInvalidate,
};

enum class BindingKind {
  kTargetState,
  kPairedValueState,
  kRecurrenceSourceState,
};

TVM_DLL std::optional<WitnessSubjectKind> ParseWitnessSubjectKind(const std::string& value);
TVM_DLL std::optional<WitnessFactAxis> ParseWitnessFactAxis(const std::string& value);
TVM_DLL std::optional<StateRole> ParseStateRole(const std::string& value);
TVM_DLL std::optional<UpdateLawKind> ParseUpdateLawKind(const std::string& value);
TVM_DLL std::optional<SupplementKind> ParseSupplementKind(const std::string& value);
TVM_DLL std::optional<ContractMode> ParseContractMode(const std::string& value);
TVM_DLL std::optional<BindingKind> ParseBindingKind(const std::string& value);

TVM_DLL const char* ToString(WitnessSubjectKind kind);
TVM_DLL const char* ToString(WitnessFactAxis axis);
TVM_DLL const char* ToString(StateRole role);
TVM_DLL const char* ToString(UpdateLawKind kind);
TVM_DLL const char* ToString(SupplementKind kind);
TVM_DLL const char* ToString(ContractMode mode);
TVM_DLL const char* ToString(BindingKind kind);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_VOCAB_H_
