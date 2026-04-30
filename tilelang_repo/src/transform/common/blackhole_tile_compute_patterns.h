/*!
 * \file blackhole_tile_compute_patterns.h
 * \brief Blackhole tile compute leaf pattern schema.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/object.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

enum class BlackholeTileComputeResultKind {
  kUnary,
  kBinary,
  kCopy,
  kReduce,
  kPack,
  kGemm,
};

enum class BlackholeTileComputeOperation {
  kFillTile,
  kCopyTile,
  kTypecastTile,
  kBinaryMaxTile,
  kAddTiles,
  kMulTiles,
  kMulTilesBcastCols,
  kAddTilesBcastCols,
  kExp2Tile,
  kRecipTile,
  kReduceTile,
  kPackTile,
  kMatmulTiles,
};

enum class BlackholeTileComputeOperandRole {
  kInput,
  kOutput,
  kLhs,
  kRhs,
  kA,
  kB,
  kC,
  kScaler,
};

enum class BlackholeTileComputeValueForm {
  kFragment,
  kFragmentOrExactCB,
  kExactCB,
  kBroadcastExactCB,
  kAccumulator,
};

enum class BlackholeTileComputeSideEffectClass {
  kDst,
  kFragment,
  kTileRegs,
  kPack,
};

enum class BlackholeTileComputeSourceEmitterKind {
  kFillFragment,
  kCopyTile,
  kTypecastTile,
  kBinaryMaxTile,
  kAddTiles,
  kMulTiles,
  kMulTilesBcastCols,
  kAddTilesBcastCols,
  kExp2Tile,
  kRecipTile,
  kReduceTile,
};

enum class BlackholeTileComputeSourceEmitterCategory {
  kNone,
  kCustom,
  kBinary,
  kBroadcastColsBinary,
  kUnary,
};

struct BlackholeTileComputeCallOperand {
  BlackholeTileComputeOperandRole role;
  size_t arg_index{0};
};

struct BlackholeTileComputePattern {
  const char* name;
  const char* root_op_name;
  BlackholeTileComputeResultKind result_kind;
  BlackholeTileComputeOperation operation;
  std::vector<BlackholeTileComputeOperandRole> operand_roles;
  std::vector<BlackholeTileComputeValueForm> required_input_forms;
  BlackholeTileComputeValueForm produced_form;
  BlackholeTileComputeSideEffectClass side_effect_class;
  std::optional<BlackholeTileComputeSourceEmitterKind> source_emitter;
  BlackholeTileComputeSourceEmitterCategory source_emitter_category{
      BlackholeTileComputeSourceEmitterCategory::kNone};
  const char* source_init_builtin{nullptr};
  const char* source_tile_builtin{nullptr};
  std::vector<BlackholeTileComputeCallOperand> blackhole_compute_operands;
  std::vector<BlackholeTileComputeCallOperand> generic_tile_op_operands;
  int64_t base_cost{1};
};

const char* ToString(BlackholeTileComputeResultKind kind);
const char* ToString(BlackholeTileComputeOperation operation);
const char* ToString(BlackholeTileComputeOperandRole role);
const char* ToString(BlackholeTileComputeValueForm form);
const char* ToString(BlackholeTileComputeSideEffectClass side_effect_class);
const char* ToString(BlackholeTileComputeSourceEmitterKind source_emitter);
const char* ToString(
    BlackholeTileComputeSourceEmitterCategory source_emitter_category);

std::optional<BlackholeTileComputeOperation> ParseBlackholeTileComputeOperation(
    const std::string& operation_name);

std::vector<std::string> BlackholeTileComputeOperandRoleNames(
    const std::vector<BlackholeTileComputeOperandRole>& roles);

const std::vector<BlackholeTileComputePattern>& GetBlackholeTileComputePatterns();

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    BlackholeTileComputeOperation operation);

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    const std::string& operation_name);

bool IsKnownBlackholeTileComputeOperation(const std::string& operation_name);

ffi::Array<ffi::Map<ffi::String, ffi::Any>> EncodeBlackholeTileComputePatternTable();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_TILE_COMPUTE_PATTERNS_H_
