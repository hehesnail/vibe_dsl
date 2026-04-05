/*!
 * \file semantic_witness_decoder.h
 * \brief Typed decoder for raw SemanticWitness payloads.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_DECODER_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_DECODER_H_

#include <tvm/ffi/reflection/registry.h>

#include <optional>
#include <string>
#include <vector>

#include "semantic_program.h"
#include "semantic_vocab.h"

namespace tvm {
namespace tl {
namespace semantic {

struct DecodedSemanticWitness {
  WitnessSubjectKind subject_kind;
  WitnessFactAxis fact_axis;
  std::string subject_anchor_id;
  ffi::Map<ffi::String, ffi::Any> fact_value;
  ffi::Array<ffi::String> related_anchor_ids;
  ffi::Array<ffi::String> evidence_sources;
  std::string canonicalization_point;
};

TVM_DLL std::optional<DecodedSemanticWitness> DecodeSemanticWitness(const SemanticWitness& witness);
TVM_DLL std::optional<StateRole> DecodeWitnessStateRole(const SemanticWitness& witness);
TVM_DLL std::optional<UpdateLawKind> DecodeWitnessUpdateLawKind(const SemanticWitness& witness);
TVM_DLL std::optional<BindingKind> DecodeWitnessBindingKind(const SemanticWitness& witness,
                                                            const char* payload_key);
TVM_DLL std::optional<ContractMode> DecodeContractMode(const ffi::Map<ffi::String, ffi::Any>& freeze);
TVM_DLL std::vector<std::string> DecodeWitnessStringPayloadArray(const SemanticWitness& witness,
                                                                 const char* payload_key);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_DECODER_H_
