/*!
 * \file analyze_blackhole_pipeline_stages.h
 * \brief Pipeline-stage evidence helpers for Blackhole planning.
 */

#ifndef TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_PIPELINE_STAGES_H_
#define TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_PIPELINE_STAGES_H_

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

TVM_DLL tvm::ffi::Array<tvm::ffi::Any> AnalyzeBlackholePipelineStageEvidence(
    const tvm::tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_PIPELINE_STAGES_H_
