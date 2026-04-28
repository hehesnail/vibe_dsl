/*!
 * \file blackhole_tile_compute_patterns.cc
 * \brief Blackhole tile compute leaf pattern schema.
 */

#include "blackhole_tile_compute_patterns.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <algorithm>

namespace tvm {
namespace tl {
namespace {

ffi::Array<ffi::String> EncodeStringVector(const std::vector<std::string>& values) {
  ffi::Array<ffi::String> encoded;
  for (const std::string& value : values) {
    encoded.push_back(ffi::String(value));
  }
  return encoded;
}

ffi::Map<ffi::String, ffi::Any> EncodePattern(const BlackholeTileComputePattern& pattern) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("name"), ffi::String(pattern.name));
  encoded.Set(ffi::String("root_op_name"), ffi::String(pattern.root_op_name));
  encoded.Set(ffi::String("result_kind"), ffi::String(pattern.result_kind));
  encoded.Set(ffi::String("operation_name"), ffi::String(pattern.operation_name));
  encoded.Set(ffi::String("operand_roles"), EncodeStringVector(pattern.operand_roles));
  encoded.Set(ffi::String("required_input_forms"),
              EncodeStringVector(pattern.required_input_forms));
  encoded.Set(ffi::String("produced_form"), ffi::String(pattern.produced_form));
  encoded.Set(ffi::String("side_effect_class"), ffi::String(pattern.side_effect_class));
  encoded.Set(ffi::String("base_cost"), Integer(pattern.base_cost));
  encoded.Set(ffi::String("selected_output"), ffi::String("tt_compute_op_plan"));
  return encoded;
}

}  // namespace

const std::vector<BlackholeTileComputePattern>& GetBlackholeTileComputePatterns() {
  static const std::vector<BlackholeTileComputePattern> patterns = {
      {"fill_fragment_pattern", "fill_tile", "unary", "fill_tile",
       {"output"}, {}, "fragment", "dst", 1},
      {"copy_tile_pattern", "copy_tile", "copy", "copy_tile",
       {"input", "output"}, {"fragment_or_exact_cb"}, "fragment_or_exact_cb", "dst", 1},
      {"typecast_tile_pattern", "typecast_tile", "unary", "typecast_tile",
       {"input", "output"}, {"fragment"}, "fragment", "dst", 1},
      {"binary_max_tile_pattern", "binary_max_tile", "binary", "binary_max_tile",
       {"lhs", "rhs", "output"}, {"exact_cb", "exact_cb"}, "exact_cb", "tile_regs", 2},
      {"add_tiles_pattern", "add_tiles", "binary", "add_tiles",
       {"lhs", "rhs", "output"}, {"exact_cb", "exact_cb"}, "exact_cb", "tile_regs", 2},
      {"mul_tiles_pattern", "mul_tiles", "binary", "mul_tiles",
       {"lhs", "rhs", "output"}, {"exact_cb", "exact_cb"}, "exact_cb", "tile_regs", 2},
      {"mul_tiles_bcast_cols_pattern", "mul_tiles_bcast_cols", "binary",
       "mul_tiles_bcast_cols", {"lhs", "rhs", "output"},
       {"exact_cb", "broadcast_exact_cb"}, "exact_cb", "tile_regs", 2},
      {"add_tiles_bcast_cols_pattern", "add_tiles_bcast_cols", "binary",
       "add_tiles_bcast_cols", {"lhs", "rhs", "output"},
       {"exact_cb", "broadcast_exact_cb"}, "exact_cb", "tile_regs", 2},
      {"exp2_tile_pattern", "exp2_tile", "unary", "exp2_tile",
       {"input", "output"}, {"exact_cb"}, "exact_cb", "tile_regs", 2},
      {"recip_tile_pattern", "recip_tile", "unary", "recip_tile",
       {"input", "output"}, {"exact_cb"}, "exact_cb", "tile_regs", 2},
      {"reduce_tile_pattern", "reduce_tile", "reduce", "reduce_tile",
       {"input", "scaler", "output"}, {"exact_cb", "exact_cb"}, "exact_cb",
       "tile_regs", 3},
      {"pack_tile_pattern", "pack_tile", "pack", "pack_tile",
       {"input", "output"}, {"fragment"}, "exact_cb", "pack", 1},
      {"matmul_tiles_pattern", "gemm", "gemm", "matmul_tiles",
       {"a", "b", "c"}, {"exact_cb", "exact_cb"}, "accumulator", "dst", 4},
  };
  return patterns;
}

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    const std::string& operation_name) {
  const std::vector<BlackholeTileComputePattern>& patterns =
      GetBlackholeTileComputePatterns();
  auto it = std::find_if(patterns.begin(), patterns.end(),
                         [&](const BlackholeTileComputePattern& pattern) {
                           return pattern.operation_name == operation_name;
                         });
  return it == patterns.end() ? nullptr : &(*it);
}

bool IsKnownBlackholeTileComputeOperation(const std::string& operation_name) {
  return FindBlackholeTileComputePattern(operation_name) != nullptr;
}

ffi::Array<ffi::Map<ffi::String, ffi::Any>> EncodeBlackholeTileComputePatternTable() {
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> encoded;
  for (const BlackholeTileComputePattern& pattern : GetBlackholeTileComputePatterns()) {
    encoded.push_back(EncodePattern(pattern));
  }
  return encoded;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.BlackholeTileComputePatternTable",
                        []() { return EncodeBlackholeTileComputePatternTable(); });
}

}  // namespace tl
}  // namespace tvm
