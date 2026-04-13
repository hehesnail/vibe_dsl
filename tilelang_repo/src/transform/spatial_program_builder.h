/*!
 * \file spatial_program_builder.h
 * \brief Shared SpatialProgram builder entry points used by split Phase B passes.
 */

#ifndef TVM_TL_TRANSFORM_SPATIAL_PROGRAM_BUILDER_H_
#define TVM_TL_TRANSFORM_SPATIAL_PROGRAM_BUILDER_H_

#include <tvm/tir/function.h>

#include "common/spatial_plan.h"
#include "common/spatial_program.h"

namespace tvm {
namespace tl {

TVM_DLL SpatialExecutionPlan BuildSpatialExecutionPlanForFunc(
    const std::string& member_func, const SpatialPlan& plan, const tir::PrimFunc& func,
    const SpatialCapabilityModel& capability_model);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_SPATIAL_PROGRAM_BUILDER_H_
