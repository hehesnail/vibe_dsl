/*!
 * \file fragment_region_analysis.h
 * \brief Shared Blackhole fragment-region analysis entrypoint.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_FRAGMENT_REGION_ANALYSIS_H_
#define TVM_TL_TRANSFORM_COMMON_FRAGMENT_REGION_ANALYSIS_H_

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

ffi::Map<ffi::String, ffi::Any> AnalyzeBlackholeFragmentRegionEvidence(
    const tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_FRAGMENT_REGION_ANALYSIS_H_
