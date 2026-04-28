/*!
 * \file blackhole_tile_compute_dag.h
 * \brief Pass-local Blackhole tile compute DAG diagnostics.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/function.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

struct BlackholeTileComputeDAGEdge {
  int64_t id{-1};
  int64_t producer_node{-1};
  int64_t consumer_node{-1};
  std::string value_role;
  std::string value_repr;
  std::string value_key;
  const Object* value_identity{nullptr};
  bool requires_materialization{false};
};

struct BlackholeTileComputeDAGNode {
  int64_t id{-1};
  std::string op_kind;
  std::string op_name;
  std::string side_effect_class;
  std::string token_input;
  std::string token_output;
};

struct BlackholeTileComputeDAG {
  std::vector<BlackholeTileComputeDAGNode> nodes;
  std::vector<BlackholeTileComputeDAGEdge> edges;
};

BlackholeTileComputeDAG BuildBlackholeTileComputeDAG(const tir::PrimFunc& func);

ffi::Map<ffi::String, ffi::Any> BuildBlackholeTileComputeDAGDiagnostic(
    const tir::PrimFunc& func);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_DAG_H_
