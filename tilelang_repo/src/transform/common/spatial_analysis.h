/*!
 * \file spatial_analysis.h
 * \brief Shared helpers for spatial companion analysis.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_

#include <optional>
#include <string>
#include <vector>

#include <tvm/ir/module.h>
#include <tvm/tir/function.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key);
std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key);
std::optional<std::vector<int64_t>> GetPayloadIndices(const Map<String, Any>& payload,
                                                      const char* key);

Array<TIRAnchor> MakeAnchors(const std::string& kind, const std::string& value);
std::string GetMemberFuncName(const GlobalVar& gvar, const tir::PrimFunc& func);
bool ContainsKind(const Array<String>& supported_kinds, const std::string& expected);

Array<String> ToStringArray(const std::vector<std::string>& values);
Array<String> MakeTraits(std::initializer_list<const char*> values);
bool HasTrait(const Array<String>& traits, const char* trait);

bool SameStringArray(const Array<String>& lhs, const Array<String>& rhs);
bool SameIntegerAnyArray(const Array<Any>& lhs, const Array<Any>& rhs);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_
