/*!
 * \file blackhole_tile_compute_dag.h
 * \brief Pass-local Blackhole tile compute DAG diagnostics.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

ffi::Map<ffi::String, ffi::Any> BuildBlackholeTileComputeDAGDiagnostic(
    const tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_
