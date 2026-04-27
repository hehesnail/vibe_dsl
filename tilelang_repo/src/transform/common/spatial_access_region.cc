/*!
 * \file spatial_access_region.cc
 * \brief Affine-lite AccessRegion helper queries for SpatialPlan construction.
 */

#include "spatial_access_region.h"

#include "blackhole_utils.h"

namespace tvm {
namespace tl {

bool SameSubject(const AccessRegion& lhs, const AccessRegion& rhs) {
  return lhs.defined() && rhs.defined() && str(lhs->subject) == str(rhs->subject);
}

bool IsFullLogicalValue(const AccessRegion& region) {
  return region.defined() && str(region->coverage_kind) == "full";
}

bool IsSliceCompatible(const AccessRegion& producer, const AccessRegion& consumer) {
  if (!SameSubject(producer, consumer)) {
    return false;
  }
  if (IsFullLogicalValue(producer)) {
    return true;
  }
  return str(producer->coverage_kind) == str(consumer->coverage_kind) &&
         producer->logical_rank == consumer->logical_rank;
}

PrimExpr RegionElementCount(const AccessRegion& region) {
  if (!region.defined()) {
    return PrimExpr();
  }
  PrimExpr count = IntImm(DataType::Int(64), 1);
  for (const PrimExpr& extent : region->extents) {
    count = count * extent;
  }
  return count;
}

ffi::Array<PrimExpr> LinearizedIndex(const AccessRegion& region) {
  if (!region.defined()) {
    return ffi::Array<PrimExpr>{};
  }
  return region->index_exprs;
}

}  // namespace tl
}  // namespace tvm
