/*!
 * \file spatial_analysis.cc
 * \brief Shared helpers for spatial companion analysis.
 */

#include "spatial_analysis.h"

namespace tvm {
namespace tl {

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return std::nullopt;
}

std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return Downcast<String>(value.value());
  }
  return std::nullopt;
}

std::optional<std::vector<int64_t>> GetPayloadIndices(const Map<String, Any>& payload,
                                                      const char* key) {
  if (auto value = payload.Get(String(key))) {
    std::vector<int64_t> result;
    for (const Any& item : Downcast<Array<Any>>(value.value())) {
      result.push_back(Downcast<Integer>(item)->value);
    }
    return result;
  }
  return std::nullopt;
}

Array<TIRAnchor> MakeAnchors(const std::string& kind, const std::string& value) {
  return Array<TIRAnchor>{TIRAnchor(String(kind), String(value))};
}

std::string GetMemberFuncName(const GlobalVar& gvar, const tir::PrimFunc& func) {
  return func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
}

bool ContainsKind(const Array<String>& supported_kinds, const std::string& expected) {
  for (const String& supported_kind : supported_kinds) {
    if (supported_kind == expected) {
      return true;
    }
  }
  return false;
}

Array<String> ToStringArray(const std::vector<std::string>& values) {
  Array<String> result;
  for (const auto& value : values) {
    result.push_back(String(value));
  }
  return result;
}

Array<String> MakeTraits(std::initializer_list<const char*> values) {
  Array<String> result;
  for (const char* value : values) {
    result.push_back(String(value));
  }
  return result;
}

bool HasTrait(const Array<String>& traits, const char* trait) {
  for (const String& current : traits) {
    if (current == trait) {
      return true;
    }
  }
  return false;
}

bool SameStringArray(const Array<String>& lhs, const Array<String>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

bool SameIntegerAnyArray(const Array<Any>& lhs, const Array<Any>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (Downcast<Integer>(lhs[i])->value != Downcast<Integer>(rhs[i])->value) {
      return false;
    }
  }
  return true;
}

}  // namespace tl
}  // namespace tvm
