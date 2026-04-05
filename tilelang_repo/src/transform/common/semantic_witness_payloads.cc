/*!
 * \file semantic_witness_payloads.cc
 * \brief Typed builder/decoder helpers for canonical SemanticWitness payload families.
 */

#include "semantic_witness_payloads.h"

#include <tvm/runtime/logging.h>

#include <initializer_list>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace semantic {
namespace {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

bool CheckExactPayloadKeys(const Map<String, Any>& payload,
                           std::initializer_list<const char*> required_keys) {
  std::unordered_set<std::string> allowed;
  for (const char* key : required_keys) {
    allowed.insert(key);
    if (payload.find(String(key)) == payload.end()) {
      return false;
    }
  }
  if (payload.size() != allowed.size()) {
    return false;
  }
  for (const auto& kv : payload) {
    if (!allowed.count(static_cast<std::string>(kv.first))) {
      return false;
    }
  }
  return true;
}

void RequireExactPayloadKeys(const Map<String, Any>& payload,
                             std::initializer_list<const char*> required_keys,
                             const char* payload_label) {
  std::unordered_set<std::string> allowed;
  for (const char* key : required_keys) {
    allowed.insert(key);
    ICHECK(payload.find(String(key)) != payload.end())
        << payload_label << " payload requires key `" << key << "`";
  }
  ICHECK_EQ(payload.size(), allowed.size())
      << payload_label << " payload must only contain canonical keys";
  for (const auto& kv : payload) {
    ICHECK(allowed.count(static_cast<std::string>(kv.first)))
        << payload_label << " payload uses unsupported key `" << kv.first << "`";
  }
}

Array<String> NormalizeStringArrayValue(const Any& value, const char* payload_label,
                                        const char* field_key) {
  if (auto typed = value.as<Array<String>>()) {
    return *typed;
  }
  if (auto generic = value.as<Array<Any>>()) {
    Array<String> normalized;
    for (const Any& item : *generic) {
      normalized.push_back(item.cast<String>());
    }
    return normalized;
  }
  ICHECK(false) << payload_label << " payload field `" << field_key << "` must be an array";
  return {};
}

}  // namespace

Map<String, Any> MakeEmptyPayload() { return {}; }

Map<String, Any> MakeStateRolePayload(StateRole role) {
  Map<String, Any> payload;
  payload.Set("role", String(ToString(role)));
  return payload;
}

Map<String, Any> MakeUpdateLawFamilyPayload(UpdateLawKind kind) {
  Map<String, Any> payload;
  payload.Set("kind", String(ToString(kind)));
  return payload;
}

Map<String, Any> MakeUpdateSourceSetPayload(const Array<String>& sources) {
  Map<String, Any> payload;
  payload.Set("sources", sources);
  return payload;
}

Map<String, Any> MakeUpdateSourceSetPayload(const Array<Any>& sources) {
  Array<String> normalized;
  for (const Any& source : sources) {
    normalized.push_back(source.cast<String>());
  }
  return MakeUpdateSourceSetPayload(normalized);
}

Map<String, Any> MakeRelationBindingPayload(BindingKind binding_kind) {
  Map<String, Any> payload;
  payload.Set("binding_kind", String(ToString(binding_kind)));
  return payload;
}

Map<String, Any> NormalizeStateRolePayload(const Map<String, Any>& payload) {
  auto decoded = DecodeStateRolePayload(payload);
  ICHECK(decoded) << "Invalid state.role payload";
  return MakeStateRolePayload(decoded->role);
}

Map<String, Any> NormalizeUpdateLawFamilyPayload(const Map<String, Any>& payload) {
  auto decoded = DecodeUpdateLawFamilyPayload(payload);
  ICHECK(decoded) << "Invalid update.law_family payload";
  return MakeUpdateLawFamilyPayload(decoded->kind);
}

Map<String, Any> NormalizeUpdateSourceSetPayload(const Map<String, Any>& payload) {
  auto decoded = DecodeUpdateSourceSetPayload(payload);
  ICHECK(decoded) << "Invalid update.source_set payload";
  Array<String> normalized;
  for (const std::string& source : decoded->sources) {
    normalized.push_back(String(source));
  }
  return MakeUpdateSourceSetPayload(normalized);
}

Map<String, Any> NormalizeRelationBindingPayload(const Map<String, Any>& payload) {
  auto decoded = DecodeRelationBindingPayload(payload);
  ICHECK(decoded) << "Invalid relation binding payload";
  return MakeRelationBindingPayload(decoded->binding_kind);
}

std::optional<StateRolePayload> DecodeStateRolePayload(const Map<String, Any>& payload) {
  if (!CheckExactPayloadKeys(payload, {"role"})) {
    return std::nullopt;
  }
  auto role_any = payload.find(String("role"));
  if (role_any == payload.end()) return std::nullopt;
  auto role = ParseStateRole((*role_any).second.cast<String>());
  if (!role) {
    return std::nullopt;
  }
  return StateRolePayload{*role};
}

std::optional<UpdateLawFamilyPayload> DecodeUpdateLawFamilyPayload(
    const Map<String, Any>& payload) {
  if (!CheckExactPayloadKeys(payload, {"kind"})) {
    return std::nullopt;
  }
  auto kind_any = payload.find(String("kind"));
  if (kind_any == payload.end()) return std::nullopt;
  auto kind = ParseUpdateLawKind((*kind_any).second.cast<String>());
  if (!kind) {
    return std::nullopt;
  }
  return UpdateLawFamilyPayload{*kind};
}

std::optional<UpdateSourceSetPayload> DecodeUpdateSourceSetPayload(
    const Map<String, Any>& payload) {
  if (!CheckExactPayloadKeys(payload, {"sources"})) {
    return std::nullopt;
  }
  auto sources_any = payload.find(String("sources"));
  if (sources_any == payload.end()) return std::nullopt;
  Array<String> normalized =
      NormalizeStringArrayValue((*sources_any).second, "update.source_set", "sources");
  UpdateSourceSetPayload result;
  for (const String& source : normalized) {
    result.sources.push_back(source);
  }
  return result;
}

std::optional<RelationBindingPayload> DecodeRelationBindingPayload(
    const Map<String, Any>& payload) {
  if (!CheckExactPayloadKeys(payload, {"binding_kind"})) {
    return std::nullopt;
  }
  auto binding_any = payload.find(String("binding_kind"));
  if (binding_any == payload.end()) return std::nullopt;
  auto binding_kind = ParseBindingKind((*binding_any).second.cast<String>());
  if (!binding_kind) {
    return std::nullopt;
  }
  return RelationBindingPayload{*binding_kind};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.SemanticPayloadNormalizeStateRole",
                        [](Map<String, Any> payload) { return NormalizeStateRolePayload(payload); });
  refl::GlobalDef().def("tl.SemanticPayloadNormalizeUpdateLawFamily", [](Map<String, Any> payload) {
    return NormalizeUpdateLawFamilyPayload(payload);
  });
  refl::GlobalDef().def("tl.SemanticPayloadNormalizeSourceSet",
                        [](Map<String, Any> payload) { return NormalizeUpdateSourceSetPayload(payload); });
  refl::GlobalDef().def("tl.SemanticPayloadNormalizeRelationBinding", [](Map<String, Any> payload) {
    return NormalizeRelationBindingPayload(payload);
  });
}

}  // namespace semantic
}  // namespace tl
}  // namespace tvm
