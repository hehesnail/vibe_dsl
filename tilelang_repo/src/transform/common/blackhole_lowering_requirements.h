/*!
 * \file blackhole_lowering_requirements.h
 * \brief Derive Blackhole lowering requirements directly from SpatialPlan and analysis attrs.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_

#include <tvm/tir/function.h>

#include "spatial_plan.h"

namespace tvm {
namespace tl {

TVM_DLL tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> BuildBlackholeLoweringRequirements(
    const tvm::tir::PrimFunc& func, const SpatialPlan& plan);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
