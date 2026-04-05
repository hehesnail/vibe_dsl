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

std::optional<StateRole> DecodeWitnessStateRole(const SemanticWitness& witness) {
  auto it = witness->fact_value.find("role");
  if (it == witness->fact_value.end()) {
    return std::nullopt;
  }
  return ParseStateRole(static_cast<std::string>(tvm::Downcast<ffi::String>((*it).second)));
}

std::optional<UpdateLawKind> DecodeWitnessUpdateLawKind(const SemanticWitness& witness) {
  auto it = witness->fact_value.find("kind");
  if (it == witness->fact_value.end()) {
    return std::nullopt;
  }
  return ParseUpdateLawKind(static_cast<std::string>(tvm::Downcast<ffi::String>((*it).second)));
}

std::optional<BindingKind> DecodeWitnessBindingKind(const SemanticWitness& witness,
                                                    const char* payload_key) {
  auto it = witness->fact_value.find(payload_key);
  if (it == witness->fact_value.end()) {
    return std::nullopt;
  }
  return ParseBindingKind(static_cast<std::string>(tvm::Downcast<ffi::String>((*it).second)));
}

std::optional<ContractMode> DecodeContractMode(const ffi::Map<ffi::String, ffi::Any>& freeze) {
  auto it = freeze.find("contract_mode");
  if (it == freeze.end()) {
    return std::nullopt;
  }
  return ParseContractMode(static_cast<std::string>(tvm::Downcast<ffi::String>((*it).second)));
}

std::vector<std::string> DecodeWitnessStringPayloadArray(const SemanticWitness& witness,
                                                         const char* payload_key) {
  std::vector<std::string> result;
  auto it = witness->fact_value.find(payload_key);
  if (it == witness->fact_value.end()) {
    return result;
  }
  for (const ffi::Any& value : tvm::Downcast<ffi::Array<ffi::Any>>((*it).second)) {
    result.push_back(tvm::Downcast<ffi::String>(value));
  }
  return result;
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
