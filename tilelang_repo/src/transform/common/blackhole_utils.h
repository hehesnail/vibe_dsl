/*!
 * \file blackhole_utils.h
 * \brief Shared utilities for Blackhole transform passes.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

inline bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
