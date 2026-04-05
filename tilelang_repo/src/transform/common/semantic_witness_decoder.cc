/*!
 * \file semantic_witness_decoder.cc
 * \brief Typed decoder for raw SemanticWitness payloads.
 */

#include "semantic_witness_decoder.h"

namespace tvm {
namespace tl {
namespace semantic {

std::optional<DecodedSemanticWitness> DecodeSemanticWitness(const SemanticWitness& witness) {
  auto subject_kind = ParseWitnessSubjectKind(static_cast<std::string>(witness->subject_kind));
  auto fact_axis = ParseWitnessFactAxis(static_cast<std::string>(witness->fact_axis));
  if (!subject_kind || !fact_axis) {
    return std::nullopt;
  }
  return DecodedSemanticWitness{*subject_kind,
                                *fact_axis,
                                static_cast<std::string>(witness->subject_anchor_id),
                                witness->fact_value,
                                witness->related_anchor_ids,
                                witness->evidence_sources,
                                static_cast<std::string>(witness->canonicalization_point)};
}

std::optional<StateRolePayload> DecodeWitnessStateRolePayload(const SemanticWitness& witness) {
  return DecodeStateRolePayload(witness->fact_value);
}

std::optional<UpdateLawFamilyPayload> DecodeWitnessUpdateLawFamilyPayload(
    const SemanticWitness& witness) {
  return DecodeUpdateLawFamilyPayload(witness->fact_value);
}

std::optional<UpdateSourceSetPayload> DecodeWitnessUpdateSourceSetPayload(
    const SemanticWitness& witness) {
  return DecodeUpdateSourceSetPayload(witness->fact_value);
}

std::optional<RelationBindingPayload> DecodeWitnessRelationBindingPayload(
    const SemanticWitness& witness) {
  return DecodeRelationBindingPayload(witness->fact_value);
}

std::optional<StateRole> DecodeWitnessStateRole(const SemanticWitness& witness) {
  auto payload = DecodeWitnessStateRolePayload(witness);
  return payload ? std::optional<StateRole>(payload->role) : std::nullopt;
}

std::optional<UpdateLawKind> DecodeWitnessUpdateLawKind(const SemanticWitness& witness) {
  auto payload = DecodeWitnessUpdateLawFamilyPayload(witness);
  return payload ? std::optional<UpdateLawKind>(payload->kind) : std::nullopt;
}

std::optional<ContractMode> DecodeContractMode(const ffi::Map<ffi::String, ffi::Any>& freeze) {
  auto it = freeze.find("contract_mode");
  if (it == freeze.end()) {
    return std::nullopt;
  }
  return ParseContractMode(static_cast<std::string>(tvm::Downcast<ffi::String>((*it).second)));
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
