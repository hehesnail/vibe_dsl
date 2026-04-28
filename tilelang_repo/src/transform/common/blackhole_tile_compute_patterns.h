/*!
 * \file blackhole_tile_compute_patterns.h
 * \brief Blackhole tile compute leaf pattern schema.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/object.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

struct BlackholeTileComputePattern {
  std::string name;
  std::string root_op_name;
  std::string result_kind;
  std::string operation_name;
  std::vector<std::string> operand_roles;
  std::vector<std::string> required_input_forms;
  std::string produced_form;
  std::string side_effect_class;
  std::string source_emitter;
  int64_t base_cost{1};
};

const std::vector<BlackholeTileComputePattern>& GetBlackholeTileComputePatterns();

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    const std::string& operation_name);

bool IsKnownBlackholeTileComputeOperation(const std::string& operation_name);

ffi::Array<ffi::Map<ffi::String, ffi::Any>> EncodeBlackholeTileComputePatternTable();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_
