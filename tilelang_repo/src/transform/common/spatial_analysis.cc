/*!
 * \file spatial_analysis.cc
 * \brief Shared helpers for spatial companion analysis.
 */

#include "spatial_analysis.h"

namespace tvm {
namespace tl {

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
