/*!
 * \file analyze_blackhole_work_decomposition.h
 * \brief Work-decomposition evidence helpers for Blackhole planning.
 */

#ifndef TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_WORK_DECOMPOSITION_H_
#define TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_WORK_DECOMPOSITION_H_

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

TVM_DLL tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>
AnalyzeBlackholeWorkDecompositionEvidence(const tvm::tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_ANALYZE_BLACKHOLE_WORK_DECOMPOSITION_H_
