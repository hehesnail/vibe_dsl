/*!
 * \file semantic_witness_payloads.h
 * \brief Typed builder/decoder helpers for canonical SemanticWitness payload families.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_PAYLOADS_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_PAYLOADS_H_

#include <optional>
#include <string>
#include <vector>

#include "semantic_program.h"
#include "semantic_vocab.h"

namespace tvm {
namespace tl {
namespace semantic {

struct StateRolePayload {
  StateRole role;
};

struct UpdateLawFamilyPayload {
  UpdateLawKind kind;
};

struct UpdateSourceSetPayload {
  std::vector<std::string> sources;
};

struct RelationBindingPayload {
  BindingKind binding_kind;
};

TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeEmptyPayload();
TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeStateRolePayload(StateRole role);
TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeUpdateLawFamilyPayload(UpdateLawKind kind);
TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeUpdateSourceSetPayload(
    const ffi::Array<ffi::String>& sources);
TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeUpdateSourceSetPayload(
    const ffi::Array<ffi::Any>& sources);
TVM_DLL ffi::Map<ffi::String, ffi::Any> MakeRelationBindingPayload(BindingKind binding_kind);

TVM_DLL ffi::Map<ffi::String, ffi::Any> NormalizeStateRolePayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL ffi::Map<ffi::String, ffi::Any> NormalizeUpdateLawFamilyPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL ffi::Map<ffi::String, ffi::Any> NormalizeUpdateSourceSetPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL ffi::Map<ffi::String, ffi::Any> NormalizeRelationBindingPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);

TVM_DLL std::optional<StateRolePayload> DecodeStateRolePayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL std::optional<UpdateLawFamilyPayload> DecodeUpdateLawFamilyPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL std::optional<UpdateSourceSetPayload> DecodeUpdateSourceSetPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);
TVM_DLL std::optional<RelationBindingPayload> DecodeRelationBindingPayload(
    const ffi::Map<ffi::String, ffi::Any>& payload);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_WITNESS_PAYLOADS_H_
