/*!
 * \file blackhole_utils.h
 * \brief Shared utilities for Blackhole transform passes.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <string>

namespace tvm {
namespace tl {

/*! \brief Convert ffi::String to std::string without static_cast noise. */
inline std::string str(const ffi::String& s) { return static_cast<std::string>(s); }

inline bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

inline const tir::VarNode* BufferDataIdentity(const tir::Buffer& buffer) {
  return buffer.defined() && buffer->data.defined() ? buffer->data.get() : nullptr;
}

inline bool SameBufferIdentity(const tir::Buffer& lhs, const tir::Buffer& rhs) {
  return lhs.same_as(rhs) ||
         (BufferDataIdentity(lhs) != nullptr && BufferDataIdentity(lhs) == BufferDataIdentity(rhs));
}

inline std::string BufferIdentityName(const tir::Buffer& buffer) {
  if (!buffer.defined()) {
    return "";
  }
  if (buffer->data.defined() && !std::string(buffer->data->name_hint).empty()) {
    return buffer->data->name_hint;
  }
  if (!std::string(buffer->name).empty()) {
    return buffer->name;
  }
  return "";
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
