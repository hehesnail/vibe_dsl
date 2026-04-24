/*!
 * \file blackhole_lowering_requirements.h
 * \brief Derive typed Blackhole lowering support facts directly from SpatialPlan and current TIR.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_

#include <tvm/tir/function.h>

#include "spatial_plan.h"

namespace tvm {
namespace tl {

struct BlackholeLoweringSupportFacts {
  tvm::ffi::Array<tvm::ffi::Any> buffer_materialization_contracts;
  tvm::ffi::Array<tvm::ffi::Any> buffer_flow_contracts;
};

TVM_DLL BlackholeLoweringSupportFacts CollectBlackholeLoweringSupportFacts(
    const tvm::tir::PrimFunc& func, const SpatialPlan& plan);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_LOWERING_REQUIREMENTS_H_
