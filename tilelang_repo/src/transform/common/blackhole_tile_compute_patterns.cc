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

using Form = BlackholeTileComputeValueForm;
using Op = BlackholeTileComputeOperation;
using Result = BlackholeTileComputeResultKind;
using Role = BlackholeTileComputeOperandRole;
using SideEffect = BlackholeTileComputeSideEffectClass;
using SourceEmitter = BlackholeTileComputeSourceEmitterKind;

template <typename Enum>
struct EnumStringEntry {
  Enum value;
  const char* name;
};

template <typename Enum, size_t N>
const char* FindEnumName(Enum value, const EnumStringEntry<Enum> (&entries)[N]) {
  for (const EnumStringEntry<Enum>& entry : entries) {
    if (entry.value == value) {
      return entry.name;
    }
  }
  return "";
}

constexpr EnumStringEntry<Result> kResultKindNames[] = {
    {Result::kUnary, "unary"},
    {Result::kBinary, "binary"},
    {Result::kCopy, "copy"},
    {Result::kReduce, "reduce"},
    {Result::kPack, "pack"},
    {Result::kGemm, "gemm"},
};

constexpr EnumStringEntry<Op> kOperationNames[] = {
    {Op::kFillTile, "fill_tile"},
    {Op::kCopyTile, "copy_tile"},
    {Op::kTypecastTile, "typecast_tile"},
    {Op::kBinaryMaxTile, "binary_max_tile"},
    {Op::kAddTiles, "add_tiles"},
    {Op::kMulTiles, "mul_tiles"},
    {Op::kMulTilesBcastCols, "mul_tiles_bcast_cols"},
    {Op::kAddTilesBcastCols, "add_tiles_bcast_cols"},
    {Op::kExp2Tile, "exp2_tile"},
    {Op::kRecipTile, "recip_tile"},
    {Op::kReduceTile, "reduce_tile"},
    {Op::kPackTile, "pack_tile"},
    {Op::kMatmulTiles, "matmul_tiles"},
};

constexpr EnumStringEntry<Role> kOperandRoleNames[] = {
    {Role::kInput, "input"},
    {Role::kOutput, "output"},
    {Role::kLhs, "lhs"},
    {Role::kRhs, "rhs"},
    {Role::kA, "a"},
    {Role::kB, "b"},
    {Role::kC, "c"},
    {Role::kScaler, "scaler"},
};

constexpr EnumStringEntry<Form> kValueFormNames[] = {
    {Form::kFragment, "fragment"},
    {Form::kFragmentOrExactCB, "fragment_or_exact_cb"},
    {Form::kExactCB, "exact_cb"},
    {Form::kBroadcastExactCB, "broadcast_exact_cb"},
    {Form::kAccumulator, "accumulator"},
};

constexpr EnumStringEntry<SideEffect> kSideEffectClassNames[] = {
    {SideEffect::kDst, "dst"},
    {SideEffect::kFragment, "fragment"},
    {SideEffect::kTileRegs, "tile_regs"},
    {SideEffect::kPack, "pack"},
};

constexpr EnumStringEntry<SourceEmitter> kSourceEmitterNames[] = {
    {SourceEmitter::kFillFragment, "fill_fragment"},
    {SourceEmitter::kCopyTile, "copy_tile"},
    {SourceEmitter::kTypecastTile, "typecast_tile"},
    {SourceEmitter::kBinaryMaxTile, "binary_max_tile"},
    {SourceEmitter::kAddTiles, "add_tiles"},
    {SourceEmitter::kMulTiles, "mul_tiles"},
    {SourceEmitter::kMulTilesBcastCols, "mul_tiles_bcast_cols"},
    {SourceEmitter::kExp2Tile, "exp2_tile"},
    {SourceEmitter::kReduceTile, "reduce_tile"},
};

template <typename Enum>
std::vector<std::string> EnumNames(const std::vector<Enum>& values) {
  std::vector<std::string> names;
  names.reserve(values.size());
  for (Enum value : values) {
    names.push_back(ToString(value));
  }
  return names;
}

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
  encoded.Set(ffi::String("result_kind"), ffi::String(ToString(pattern.result_kind)));
  encoded.Set(ffi::String("operation_name"), ffi::String(ToString(pattern.operation)));
  encoded.Set(ffi::String("operand_roles"),
              EncodeStringVector(EnumNames(pattern.operand_roles)));
  encoded.Set(ffi::String("required_input_forms"),
              EncodeStringVector(EnumNames(pattern.required_input_forms)));
  encoded.Set(ffi::String("produced_form"), ffi::String(ToString(pattern.produced_form)));
  encoded.Set(ffi::String("side_effect_class"),
              ffi::String(ToString(pattern.side_effect_class)));
  encoded.Set(ffi::String("source_emitter"),
              ffi::String(pattern.source_emitter ? ToString(*pattern.source_emitter) : ""));
  encoded.Set(ffi::String("base_cost"), Integer(pattern.base_cost));
  encoded.Set(ffi::String("selected_output"), ffi::String("tt_compute_op_plan"));
  return encoded;
}

}  // namespace

const char* ToString(BlackholeTileComputeResultKind kind) {
  return FindEnumName(kind, kResultKindNames);
}

const char* ToString(BlackholeTileComputeOperation operation) {
  return FindEnumName(operation, kOperationNames);
}

const char* ToString(BlackholeTileComputeOperandRole role) {
  return FindEnumName(role, kOperandRoleNames);
}

const char* ToString(BlackholeTileComputeValueForm form) {
  return FindEnumName(form, kValueFormNames);
}

const char* ToString(BlackholeTileComputeSideEffectClass side_effect_class) {
  return FindEnumName(side_effect_class, kSideEffectClassNames);
}

const char* ToString(BlackholeTileComputeSourceEmitterKind source_emitter) {
  return FindEnumName(source_emitter, kSourceEmitterNames);
}

std::optional<BlackholeTileComputeOperation> ParseBlackholeTileComputeOperation(
    const std::string& operation_name) {
  for (const EnumStringEntry<Op>& entry : kOperationNames) {
    if (operation_name == entry.name) {
      return entry.value;
    }
  }
  return std::nullopt;
}

std::vector<std::string> BlackholeTileComputeOperandRoleNames(
    const std::vector<BlackholeTileComputeOperandRole>& roles) {
  return EnumNames(roles);
}

const std::vector<BlackholeTileComputePattern>& GetBlackholeTileComputePatterns() {
  static const std::vector<BlackholeTileComputePattern> patterns = {
      {"fill_fragment_pattern", "fill_tile", Result::kUnary, Op::kFillTile,
       {Role::kOutput}, {}, Form::kFragment, SideEffect::kDst,
       SourceEmitter::kFillFragment, {{Role::kOutput, 1}}, {}, 1},
      {"copy_tile_pattern", "copy_tile", Result::kCopy, Op::kCopyTile,
       {Role::kInput, Role::kOutput}, {Form::kFragmentOrExactCB},
       Form::kFragmentOrExactCB, SideEffect::kDst, SourceEmitter::kCopyTile,
       {{Role::kInput, 1}, {Role::kOutput, 2}}, {}, 1},
      {"typecast_tile_pattern", "typecast_tile", Result::kUnary, Op::kTypecastTile,
       {Role::kInput, Role::kOutput}, {Form::kFragment}, Form::kFragment,
       SideEffect::kDst, SourceEmitter::kTypecastTile,
       {{Role::kInput, 1}, {Role::kOutput, 2}}, {}, 1},
      {"binary_max_tile_pattern", "binary_max_tile", Result::kBinary, Op::kBinaryMaxTile,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kBinaryMaxTile,
       {{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}, {}, 2},
      {"add_tiles_pattern", "add_tiles", Result::kBinary, Op::kAddTiles,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kAddTiles,
       {{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}, {}, 2},
      {"mul_tiles_pattern", "mul_tiles", Result::kBinary, Op::kMulTiles,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kMulTiles,
       {{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}, {}, 2},
      {"mul_tiles_bcast_cols_pattern", "mul_tiles_bcast_cols", Result::kBinary,
       Op::kMulTilesBcastCols, {Role::kLhs, Role::kRhs, Role::kOutput},
       {Form::kExactCB, Form::kBroadcastExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, SourceEmitter::kMulTilesBcastCols,
       {{Role::kLhs, 2}, {Role::kRhs, 3}, {Role::kOutput, 2}}, {}, 2},
      {"add_tiles_bcast_cols_pattern", "add_tiles_bcast_cols", Result::kBinary,
       Op::kAddTilesBcastCols, {Role::kLhs, Role::kRhs, Role::kOutput},
       {Form::kExactCB, Form::kBroadcastExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, std::nullopt,
       {{Role::kLhs, 2}, {Role::kRhs, 3}, {Role::kOutput, 2}}, {}, 2},
      {"exp2_tile_pattern", "exp2_tile", Result::kUnary, Op::kExp2Tile,
       {Role::kInput, Role::kOutput}, {Form::kExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, SourceEmitter::kExp2Tile,
       {{Role::kOutput, 2}, {Role::kLhs, 3}, {Role::kRhs, 4}}, {}, 2},
      {"recip_tile_pattern", "recip_tile", Result::kUnary, Op::kRecipTile,
       {Role::kInput, Role::kOutput}, {Form::kExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, std::nullopt,
       {{Role::kInput, 2}, {Role::kOutput, 2}}, {}, 2},
      {"reduce_tile_pattern", "reduce_tile", Result::kReduce, Op::kReduceTile,
       {Role::kInput, Role::kScaler, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kReduceTile,
       {}, {{Role::kInput, 0}, {Role::kOutput, 1}}, 3},
      {"pack_tile_pattern", "pack_tile", Result::kPack, Op::kPackTile,
       {Role::kInput, Role::kOutput}, {Form::kFragment}, Form::kExactCB,
       SideEffect::kPack, std::nullopt,
       {{Role::kInput, 1}, {Role::kOutput, 2}}, {}, 1},
      {"matmul_tiles_pattern", "gemm", Result::kGemm, Op::kMatmulTiles,
       {Role::kA, Role::kB, Role::kC}, {Form::kExactCB, Form::kExactCB},
       Form::kAccumulator, SideEffect::kDst, std::nullopt,
       {}, {{Role::kA, 0}, {Role::kB, 1}, {Role::kC, 2}}, 4},
  };
  return patterns;
}

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    BlackholeTileComputeOperation operation) {
  const std::vector<BlackholeTileComputePattern>& patterns =
      GetBlackholeTileComputePatterns();
  auto it = std::find_if(patterns.begin(), patterns.end(),
                         [&](const BlackholeTileComputePattern& pattern) {
                           return pattern.operation == operation;
                         });
  return it == patterns.end() ? nullptr : &(*it);
}

const BlackholeTileComputePattern* FindBlackholeTileComputePattern(
    const std::string& operation_name) {
  const std::optional<BlackholeTileComputeOperation> operation =
      ParseBlackholeTileComputeOperation(operation_name);
  return operation ? FindBlackholeTileComputePattern(*operation) : nullptr;
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
