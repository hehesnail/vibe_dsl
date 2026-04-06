/*!
 * \file semantic_vocab.cc
 * \brief Closed semantic vocabulary parse/print helpers and FFI normalizers.
 */

#include "semantic_vocab.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <utility>

namespace tvm {
namespace tl {
namespace semantic {
namespace {

template <typename EnumT>
ffi::String NormalizeEnumString(const ffi::String& value, std::optional<EnumT> (*parse_fn)(const std::string&),
                                const char* (*print_fn)(EnumT), const char* label) {
  auto parsed = parse_fn(static_cast<std::string>(value));
  ICHECK(parsed) << "Unsupported semantic " << label << ": " << value;
  return ffi::String(print_fn(*parsed));
}

}  // namespace

std::optional<WitnessSubjectKind> ParseWitnessSubjectKind(const std::string& value) {
  if (value == "domain") return WitnessSubjectKind::kDomain;
  if (value == "state") return WitnessSubjectKind::kState;
  if (value == "update") return WitnessSubjectKind::kUpdate;
  if (value == "access") return WitnessSubjectKind::kAccess;
  if (value == "relation") return WitnessSubjectKind::kRelation;
  if (value == "boundary") return WitnessSubjectKind::kBoundary;
  return std::nullopt;
}

std::optional<WitnessFactAxis> ParseWitnessFactAxis(const std::string& value) {
  if (value == "role") return WitnessFactAxis::kRole;
  if (value == "identity") return WitnessFactAxis::kIdentity;
  if (value == "lifetime") return WitnessFactAxis::kLifetime;
  if (value == "law_family") return WitnessFactAxis::kLawFamily;
  if (value == "source_set") return WitnessFactAxis::kSourceSet;
  if (value == "ordering") return WitnessFactAxis::kOrdering;
  if (value == "boundary") return WitnessFactAxis::kBoundary;
  if (value == "indirection") return WitnessFactAxis::kIndirection;
  if (value == "selection_contract") return WitnessFactAxis::kSelectionContract;
  if (value == "distribution_hint") return WitnessFactAxis::kDistributionHint;
  if (value == "companion") return WitnessFactAxis::kCompanion;
  if (value == "derives_index_from") return WitnessFactAxis::kDerivesIndexFrom;
  if (value == "feeds_update") return WitnessFactAxis::kFeedsUpdate;
  if (value == "carried_from") return WitnessFactAxis::kCarriedFrom;
  if (value == "semantic_boundary") return WitnessFactAxis::kSemanticBoundary;
  if (value == "ordered_region") return WitnessFactAxis::kOrderedRegion;
  return std::nullopt;
}

std::optional<StateRole> ParseStateRole(const std::string& value) {
  if (value == "carry") return StateRole::kCarry;
  if (value == "reduction_accumulator") return StateRole::kReductionAccumulator;
  if (value == "selection_state") return StateRole::kSelectionState;
  if (value == "index_state") return StateRole::kIndexState;
  if (value == "transient") return StateRole::kTransient;
  return std::nullopt;
}

std::optional<UpdateLawKind> ParseUpdateLawKind(const std::string& value) {
  if (value == "map") return UpdateLawKind::kMap;
  if (value == "reduce") return UpdateLawKind::kReduce;
  if (value == "select") return UpdateLawKind::kSelect;
  if (value == "recurrence") return UpdateLawKind::kRecurrence;
  return std::nullopt;
}

std::optional<SupplementKind> ParseSupplementKind(const std::string& value) {
  if (value == "state_identity") return SupplementKind::kStateIdentity;
  if (value == "access_trait") return SupplementKind::kAccessTrait;
  if (value == "update_law_trait") return SupplementKind::kUpdateLawTrait;
  if (value == "semantic_boundary") return SupplementKind::kSemanticBoundary;
  if (value == "fragment_lowering_structure") return SupplementKind::kFragmentLoweringStructure;
  if (value == "pipeline_structure") return SupplementKind::kPipelineStructure;
  if (value == "work_decomposition_structure") return SupplementKind::kWorkDecompositionStructure;
  return std::nullopt;
}

std::optional<ContractMode> ParseContractMode(const std::string& value) {
  if (value == "preserve") return ContractMode::kPreserve;
  if (value == "typed_rebind") return ContractMode::kTypedRebind;
  if (value == "invalidate") return ContractMode::kInvalidate;
  return std::nullopt;
}

std::optional<BindingKind> ParseBindingKind(const std::string& value) {
  if (value == "target_state") return BindingKind::kTargetState;
  if (value == "paired_value_state") return BindingKind::kPairedValueState;
  if (value == "recurrence_source_state") return BindingKind::kRecurrenceSourceState;
  return std::nullopt;
}

std::optional<StateVersionKind> ParseStateVersionKind(const std::string& value) {
  if (value == "initial") return StateVersionKind::kInitial;
  if (value == "update_result") return StateVersionKind::kUpdateResult;
  return std::nullopt;
}

std::optional<StateDefKind> ParseStateDefKind(const std::string& value) {
  if (value == "initial") return StateDefKind::kInitial;
  if (value == "update_result") return StateDefKind::kUpdateResult;
  return std::nullopt;
}

std::optional<StateUseKind> ParseStateUseKind(const std::string& value) {
  if (value == "source_state") return StateUseKind::kSourceState;
  if (value == "companion_state") return StateUseKind::kCompanionState;
  if (value == "carried_state") return StateUseKind::kCarriedState;
  return std::nullopt;
}

std::optional<StateJoinKind> ParseStateJoinKind(const std::string& value) {
  if (value == "loop_carried") return StateJoinKind::kLoopCarried;
  if (value == "ordered_update") return StateJoinKind::kOrderedUpdate;
  return std::nullopt;
}

std::optional<RebindScope> ParseRebindScope(const std::string& value) {
  if (value == "body_hash_refresh") return RebindScope::kBodyHashRefresh;
  if (value == "anchor_remap") return RebindScope::kAnchorRemap;
  return std::nullopt;
}

const char* ToString(WitnessSubjectKind kind) {
  switch (kind) {
    case WitnessSubjectKind::kDomain:
      return "domain";
    case WitnessSubjectKind::kState:
      return "state";
    case WitnessSubjectKind::kUpdate:
      return "update";
    case WitnessSubjectKind::kAccess:
      return "access";
    case WitnessSubjectKind::kRelation:
      return "relation";
    case WitnessSubjectKind::kBoundary:
      return "boundary";
  }
  LOG(FATAL) << "Unknown WitnessSubjectKind";
  return "unknown";
}

const char* ToString(WitnessFactAxis axis) {
  switch (axis) {
    case WitnessFactAxis::kRole:
      return "role";
    case WitnessFactAxis::kIdentity:
      return "identity";
    case WitnessFactAxis::kLifetime:
      return "lifetime";
    case WitnessFactAxis::kLawFamily:
      return "law_family";
    case WitnessFactAxis::kSourceSet:
      return "source_set";
    case WitnessFactAxis::kOrdering:
      return "ordering";
    case WitnessFactAxis::kBoundary:
      return "boundary";
    case WitnessFactAxis::kIndirection:
      return "indirection";
    case WitnessFactAxis::kSelectionContract:
      return "selection_contract";
    case WitnessFactAxis::kDistributionHint:
      return "distribution_hint";
    case WitnessFactAxis::kCompanion:
      return "companion";
    case WitnessFactAxis::kDerivesIndexFrom:
      return "derives_index_from";
    case WitnessFactAxis::kFeedsUpdate:
      return "feeds_update";
    case WitnessFactAxis::kCarriedFrom:
      return "carried_from";
    case WitnessFactAxis::kSemanticBoundary:
      return "semantic_boundary";
    case WitnessFactAxis::kOrderedRegion:
      return "ordered_region";
  }
  LOG(FATAL) << "Unknown WitnessFactAxis";
  return "unknown";
}

const char* ToString(StateRole role) {
  switch (role) {
    case StateRole::kCarry:
      return "carry";
    case StateRole::kReductionAccumulator:
      return "reduction_accumulator";
    case StateRole::kSelectionState:
      return "selection_state";
    case StateRole::kIndexState:
      return "index_state";
    case StateRole::kTransient:
      return "transient";
  }
  LOG(FATAL) << "Unknown StateRole";
  return "unknown";
}

const char* ToString(UpdateLawKind kind) {
  switch (kind) {
    case UpdateLawKind::kMap:
      return "map";
    case UpdateLawKind::kReduce:
      return "reduce";
    case UpdateLawKind::kSelect:
      return "select";
    case UpdateLawKind::kRecurrence:
      return "recurrence";
  }
  LOG(FATAL) << "Unknown UpdateLawKind";
  return "unknown";
}

const char* ToString(SupplementKind kind) {
  switch (kind) {
    case SupplementKind::kStateIdentity:
      return "state_identity";
    case SupplementKind::kAccessTrait:
      return "access_trait";
    case SupplementKind::kUpdateLawTrait:
      return "update_law_trait";
    case SupplementKind::kSemanticBoundary:
      return "semantic_boundary";
    case SupplementKind::kFragmentLoweringStructure:
      return "fragment_lowering_structure";
    case SupplementKind::kPipelineStructure:
      return "pipeline_structure";
    case SupplementKind::kWorkDecompositionStructure:
      return "work_decomposition_structure";
  }
  LOG(FATAL) << "Unknown SupplementKind";
  return "unknown";
}

const char* ToString(ContractMode mode) {
  switch (mode) {
    case ContractMode::kPreserve:
      return "preserve";
    case ContractMode::kTypedRebind:
      return "typed_rebind";
    case ContractMode::kInvalidate:
      return "invalidate";
  }
  LOG(FATAL) << "Unknown ContractMode";
  return "unknown";
}

const char* ToString(BindingKind kind) {
  switch (kind) {
    case BindingKind::kTargetState:
      return "target_state";
    case BindingKind::kPairedValueState:
      return "paired_value_state";
    case BindingKind::kRecurrenceSourceState:
      return "recurrence_source_state";
  }
  LOG(FATAL) << "Unknown BindingKind";
  return "unknown";
}

const char* ToString(StateVersionKind kind) {
  switch (kind) {
    case StateVersionKind::kInitial:
      return "initial";
    case StateVersionKind::kUpdateResult:
      return "update_result";
  }
  LOG(FATAL) << "Unknown StateVersionKind";
  return "unknown";
}

const char* ToString(StateDefKind kind) {
  switch (kind) {
    case StateDefKind::kInitial:
      return "initial";
    case StateDefKind::kUpdateResult:
      return "update_result";
  }
  LOG(FATAL) << "Unknown StateDefKind";
  return "unknown";
}

const char* ToString(StateUseKind kind) {
  switch (kind) {
    case StateUseKind::kSourceState:
      return "source_state";
    case StateUseKind::kCompanionState:
      return "companion_state";
    case StateUseKind::kCarriedState:
      return "carried_state";
  }
  LOG(FATAL) << "Unknown StateUseKind";
  return "unknown";
}

const char* ToString(StateJoinKind kind) {
  switch (kind) {
    case StateJoinKind::kLoopCarried:
      return "loop_carried";
    case StateJoinKind::kOrderedUpdate:
      return "ordered_update";
  }
  LOG(FATAL) << "Unknown StateJoinKind";
  return "unknown";
}

const char* ToString(RebindScope scope) {
  switch (scope) {
    case RebindScope::kBodyHashRefresh:
      return "body_hash_refresh";
    case RebindScope::kAnchorRemap:
      return "anchor_remap";
  }
  LOG(FATAL) << "Unknown RebindScope";
  return "unknown";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.SemanticVocabNormalizeBindingKind", [](ffi::String value) {
    return NormalizeEnumString(value, ParseBindingKind, ToString, "binding kind");
  });
  refl::GlobalDef().def("tl.SemanticVocabNormalizeContractMode", [](ffi::String value) {
    return NormalizeEnumString(value, ParseContractMode, ToString, "contract mode");
  });
  refl::GlobalDef().def("tl.SemanticVocabNormalizeWitnessSubjectKind", [](ffi::String value) {
    return NormalizeEnumString(value, ParseWitnessSubjectKind, ToString, "witness subject kind");
  });
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
