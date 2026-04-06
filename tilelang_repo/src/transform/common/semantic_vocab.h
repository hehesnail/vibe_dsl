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
  kFragmentLoweringStructure,
  kPipelineStructure,
  kWorkDecompositionStructure,
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

enum class StateVersionKind {
  kInitial,
  kUpdateResult,
};

enum class StateDefKind {
  kInitial,
  kUpdateResult,
};

enum class StateUseKind {
  kSourceState,
  kCompanionState,
  kCarriedState,
};

enum class StateJoinKind {
  kLoopCarried,
  kOrderedUpdate,
};

enum class RebindScope {
  kBodyHashRefresh,
  kAnchorRemap,
};

// ---------------------------------------------------------------------------
// Spatial IR closed vocabularies (Phase B)
// ---------------------------------------------------------------------------

enum class SpatialTaskKind {
  kTransfer,
  kCompute,
  kCollective,
  kControl,
};

enum class SpatialChannelKind {
  kTensorFlow,
  kStateFlow,
  kPhaseBoundary,
};

enum class SpatialLayoutKind {
  kRegular,
  kIndexed,
};

enum class SpatialPartitionKind {
  kBlocked,
  kReplicated,
};

enum class SpatialPlacementKind {
  kExecution,
};

enum class SpatialSyncKind {
  kDependency,
  kCompletion,
  kBarrier,
};

enum class SpatialResourceIntentKind {
  kBuffer,
  kStateResidency,
  kSynchronizationSupport,
  kPhaseBoundaryMaterialization,
  kLoweringSupport,
};

TVM_DLL std::optional<WitnessSubjectKind> ParseWitnessSubjectKind(const std::string& value);
TVM_DLL std::optional<WitnessFactAxis> ParseWitnessFactAxis(const std::string& value);
TVM_DLL std::optional<StateRole> ParseStateRole(const std::string& value);
TVM_DLL std::optional<UpdateLawKind> ParseUpdateLawKind(const std::string& value);
TVM_DLL std::optional<SupplementKind> ParseSupplementKind(const std::string& value);
TVM_DLL std::optional<ContractMode> ParseContractMode(const std::string& value);
TVM_DLL std::optional<BindingKind> ParseBindingKind(const std::string& value);
TVM_DLL std::optional<StateVersionKind> ParseStateVersionKind(const std::string& value);
TVM_DLL std::optional<StateDefKind> ParseStateDefKind(const std::string& value);
TVM_DLL std::optional<StateUseKind> ParseStateUseKind(const std::string& value);
TVM_DLL std::optional<StateJoinKind> ParseStateJoinKind(const std::string& value);
TVM_DLL std::optional<RebindScope> ParseRebindScope(const std::string& value);

TVM_DLL std::optional<SpatialTaskKind> ParseSpatialTaskKind(const std::string& value);
TVM_DLL std::optional<SpatialChannelKind> ParseSpatialChannelKind(const std::string& value);
TVM_DLL std::optional<SpatialLayoutKind> ParseSpatialLayoutKind(const std::string& value);
TVM_DLL std::optional<SpatialPartitionKind> ParseSpatialPartitionKind(const std::string& value);
TVM_DLL std::optional<SpatialPlacementKind> ParseSpatialPlacementKind(const std::string& value);
TVM_DLL std::optional<SpatialSyncKind> ParseSpatialSyncKind(const std::string& value);
TVM_DLL std::optional<SpatialResourceIntentKind> ParseSpatialResourceIntentKind(const std::string& value);

TVM_DLL const char* ToString(WitnessSubjectKind kind);
TVM_DLL const char* ToString(WitnessFactAxis axis);
TVM_DLL const char* ToString(StateRole role);
TVM_DLL const char* ToString(UpdateLawKind kind);
TVM_DLL const char* ToString(SupplementKind kind);
TVM_DLL const char* ToString(ContractMode mode);
TVM_DLL const char* ToString(BindingKind kind);
TVM_DLL const char* ToString(StateVersionKind kind);
TVM_DLL const char* ToString(StateDefKind kind);
TVM_DLL const char* ToString(StateUseKind kind);
TVM_DLL const char* ToString(StateJoinKind kind);
TVM_DLL const char* ToString(RebindScope scope);

TVM_DLL const char* ToString(SpatialTaskKind kind);
TVM_DLL const char* ToString(SpatialChannelKind kind);
TVM_DLL const char* ToString(SpatialLayoutKind kind);
TVM_DLL const char* ToString(SpatialPartitionKind kind);
TVM_DLL const char* ToString(SpatialPlacementKind kind);
TVM_DLL const char* ToString(SpatialSyncKind kind);
TVM_DLL const char* ToString(SpatialResourceIntentKind kind);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_VOCAB_H_
