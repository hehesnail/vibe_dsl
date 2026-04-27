/*!
 * \file spatial_access_region.h
 * \brief Affine-lite AccessRegion helper queries for SpatialPlan construction.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_ACCESS_REGION_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_ACCESS_REGION_H_

#include "spatial_plan.h"

namespace tvm {
namespace tl {

bool SameSubject(const AccessRegion& lhs, const AccessRegion& rhs);
bool IsFullLogicalValue(const AccessRegion& region);
bool IsSliceCompatible(const AccessRegion& producer, const AccessRegion& consumer);
PrimExpr RegionElementCount(const AccessRegion& region);
ffi::Array<PrimExpr> LinearizedIndex(const AccessRegion& region);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_ACCESS_REGION_H_
