/*!
 * \file blackhole_tile_compute_patterns.cc
 * \brief Blackhole tile compute leaf pattern schema.
 */

#include "blackhole_tile_compute_patterns.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <algorithm>
#include <initializer_list>

namespace tvm {
namespace tl {
namespace {

using Form = BlackholeTileComputeValueForm;
using Op = BlackholeTileComputeOperation;
using Result = BlackholeTileComputeResultKind;
using Role = BlackholeTileComputeOperandRole;
using SideEffect = BlackholeTileComputeSideEffectClass;
using SourceEmitter = BlackholeTileComputeSourceEmitterKind;

std::vector<BlackholeTileComputeCallOperand> Args(
    std::initializer_list<BlackholeTileComputeCallOperand> args) {
  return std::vector<BlackholeTileComputeCallOperand>(args);
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
              EncodeStringVector(BlackholeTileComputeOperandRoleNames(pattern.operand_roles)));
  encoded.Set(ffi::String("required_input_forms"),
              EncodeStringVector([&]() {
                std::vector<std::string> forms;
                for (BlackholeTileComputeValueForm form : pattern.required_input_forms) {
                  forms.push_back(ToString(form));
                }
                return forms;
              }()));
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
  switch (kind) {
    case BlackholeTileComputeResultKind::kUnary:
      return "unary";
    case BlackholeTileComputeResultKind::kBinary:
      return "binary";
    case BlackholeTileComputeResultKind::kCopy:
      return "copy";
    case BlackholeTileComputeResultKind::kReduce:
      return "reduce";
    case BlackholeTileComputeResultKind::kPack:
      return "pack";
    case BlackholeTileComputeResultKind::kGemm:
      return "gemm";
  }
  return "";
}

const char* ToString(BlackholeTileComputeOperation operation) {
  switch (operation) {
    case BlackholeTileComputeOperation::kFillTile:
      return "fill_tile";
    case BlackholeTileComputeOperation::kCopyTile:
      return "copy_tile";
    case BlackholeTileComputeOperation::kTypecastTile:
      return "typecast_tile";
    case BlackholeTileComputeOperation::kBinaryMaxTile:
      return "binary_max_tile";
    case BlackholeTileComputeOperation::kAddTiles:
      return "add_tiles";
    case BlackholeTileComputeOperation::kMulTiles:
      return "mul_tiles";
    case BlackholeTileComputeOperation::kMulTilesBcastCols:
      return "mul_tiles_bcast_cols";
    case BlackholeTileComputeOperation::kAddTilesBcastCols:
      return "add_tiles_bcast_cols";
    case BlackholeTileComputeOperation::kExp2Tile:
      return "exp2_tile";
    case BlackholeTileComputeOperation::kRecipTile:
      return "recip_tile";
    case BlackholeTileComputeOperation::kReduceTile:
      return "reduce_tile";
    case BlackholeTileComputeOperation::kPackTile:
      return "pack_tile";
    case BlackholeTileComputeOperation::kMatmulTiles:
      return "matmul_tiles";
  }
  return "";
}

const char* ToString(BlackholeTileComputeOperandRole role) {
  switch (role) {
    case BlackholeTileComputeOperandRole::kInput:
      return "input";
    case BlackholeTileComputeOperandRole::kOutput:
      return "output";
    case BlackholeTileComputeOperandRole::kLhs:
      return "lhs";
    case BlackholeTileComputeOperandRole::kRhs:
      return "rhs";
    case BlackholeTileComputeOperandRole::kA:
      return "a";
    case BlackholeTileComputeOperandRole::kB:
      return "b";
    case BlackholeTileComputeOperandRole::kC:
      return "c";
    case BlackholeTileComputeOperandRole::kScaler:
      return "scaler";
  }
  return "";
}

const char* ToString(BlackholeTileComputeValueForm form) {
  switch (form) {
    case BlackholeTileComputeValueForm::kFragment:
      return "fragment";
    case BlackholeTileComputeValueForm::kFragmentOrExactCB:
      return "fragment_or_exact_cb";
    case BlackholeTileComputeValueForm::kExactCB:
      return "exact_cb";
    case BlackholeTileComputeValueForm::kBroadcastExactCB:
      return "broadcast_exact_cb";
    case BlackholeTileComputeValueForm::kAccumulator:
      return "accumulator";
  }
  return "";
}

const char* ToString(BlackholeTileComputeSideEffectClass side_effect_class) {
  switch (side_effect_class) {
    case BlackholeTileComputeSideEffectClass::kDst:
      return "dst";
    case BlackholeTileComputeSideEffectClass::kFragment:
      return "fragment";
    case BlackholeTileComputeSideEffectClass::kTileRegs:
      return "tile_regs";
    case BlackholeTileComputeSideEffectClass::kPack:
      return "pack";
  }
  return "";
}

const char* ToString(BlackholeTileComputeSourceEmitterKind source_emitter) {
  switch (source_emitter) {
    case BlackholeTileComputeSourceEmitterKind::kFillFragment:
      return "fill_fragment";
    case BlackholeTileComputeSourceEmitterKind::kCopyTile:
      return "copy_tile";
    case BlackholeTileComputeSourceEmitterKind::kTypecastTile:
      return "typecast_tile";
    case BlackholeTileComputeSourceEmitterKind::kBinaryMaxTile:
      return "binary_max_tile";
    case BlackholeTileComputeSourceEmitterKind::kAddTiles:
      return "add_tiles";
    case BlackholeTileComputeSourceEmitterKind::kMulTiles:
      return "mul_tiles";
    case BlackholeTileComputeSourceEmitterKind::kMulTilesBcastCols:
      return "mul_tiles_bcast_cols";
    case BlackholeTileComputeSourceEmitterKind::kExp2Tile:
      return "exp2_tile";
    case BlackholeTileComputeSourceEmitterKind::kReduceTile:
      return "reduce_tile";
  }
  return "";
}

std::optional<BlackholeTileComputeOperation> ParseBlackholeTileComputeOperation(
    const std::string& operation_name) {
  for (const BlackholeTileComputePattern& pattern : GetBlackholeTileComputePatterns()) {
    if (operation_name == ToString(pattern.operation)) {
      return pattern.operation;
    }
  }
  return std::nullopt;
}

std::vector<std::string> BlackholeTileComputeOperandRoleNames(
    const std::vector<BlackholeTileComputeOperandRole>& roles) {
  std::vector<std::string> names;
  names.reserve(roles.size());
  for (BlackholeTileComputeOperandRole role : roles) {
    names.push_back(ToString(role));
  }
  return names;
}

const std::vector<BlackholeTileComputePattern>& GetBlackholeTileComputePatterns() {
  static const std::vector<BlackholeTileComputePattern> patterns = {
      {"fill_fragment_pattern", "fill_tile", Result::kUnary, Op::kFillTile,
       {Role::kOutput}, {}, Form::kFragment, SideEffect::kDst,
       SourceEmitter::kFillFragment, Args({{Role::kOutput, 1}}), {}, 1},
      {"copy_tile_pattern", "copy_tile", Result::kCopy, Op::kCopyTile,
       {Role::kInput, Role::kOutput}, {Form::kFragmentOrExactCB},
       Form::kFragmentOrExactCB, SideEffect::kDst, SourceEmitter::kCopyTile,
       Args({{Role::kInput, 1}, {Role::kOutput, 2}}), {}, 1},
      {"typecast_tile_pattern", "typecast_tile", Result::kUnary, Op::kTypecastTile,
       {Role::kInput, Role::kOutput}, {Form::kFragment}, Form::kFragment,
       SideEffect::kDst, SourceEmitter::kTypecastTile,
       Args({{Role::kInput, 1}, {Role::kOutput, 2}}), {}, 1},
      {"binary_max_tile_pattern", "binary_max_tile", Result::kBinary, Op::kBinaryMaxTile,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kBinaryMaxTile,
       Args({{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}), {}, 2},
      {"add_tiles_pattern", "add_tiles", Result::kBinary, Op::kAddTiles,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kAddTiles,
       Args({{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}), {}, 2},
      {"mul_tiles_pattern", "mul_tiles", Result::kBinary, Op::kMulTiles,
       {Role::kLhs, Role::kRhs, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kMulTiles,
       Args({{Role::kLhs, 1}, {Role::kRhs, 2}, {Role::kOutput, 1}}), {}, 2},
      {"mul_tiles_bcast_cols_pattern", "mul_tiles_bcast_cols", Result::kBinary,
       Op::kMulTilesBcastCols, {Role::kLhs, Role::kRhs, Role::kOutput},
       {Form::kExactCB, Form::kBroadcastExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, SourceEmitter::kMulTilesBcastCols,
       Args({{Role::kLhs, 2}, {Role::kRhs, 3}, {Role::kOutput, 2}}), {}, 2},
      {"add_tiles_bcast_cols_pattern", "add_tiles_bcast_cols", Result::kBinary,
       Op::kAddTilesBcastCols, {Role::kLhs, Role::kRhs, Role::kOutput},
       {Form::kExactCB, Form::kBroadcastExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, std::nullopt,
       Args({{Role::kLhs, 2}, {Role::kRhs, 3}, {Role::kOutput, 2}}), {}, 2},
      {"exp2_tile_pattern", "exp2_tile", Result::kUnary, Op::kExp2Tile,
       {Role::kInput, Role::kOutput}, {Form::kExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, SourceEmitter::kExp2Tile,
       Args({{Role::kOutput, 2}, {Role::kLhs, 3}, {Role::kRhs, 4}}), {}, 2},
      {"recip_tile_pattern", "recip_tile", Result::kUnary, Op::kRecipTile,
       {Role::kInput, Role::kOutput}, {Form::kExactCB}, Form::kExactCB,
       SideEffect::kTileRegs, std::nullopt,
       Args({{Role::kInput, 2}, {Role::kOutput, 2}}), {}, 2},
      {"reduce_tile_pattern", "reduce_tile", Result::kReduce, Op::kReduceTile,
       {Role::kInput, Role::kScaler, Role::kOutput}, {Form::kExactCB, Form::kExactCB},
       Form::kExactCB, SideEffect::kTileRegs, SourceEmitter::kReduceTile,
       {}, Args({{Role::kInput, 0}, {Role::kOutput, 1}}), 3},
      {"pack_tile_pattern", "pack_tile", Result::kPack, Op::kPackTile,
       {Role::kInput, Role::kOutput}, {Form::kFragment}, Form::kExactCB,
       SideEffect::kPack, std::nullopt,
       Args({{Role::kInput, 1}, {Role::kOutput, 2}}), {}, 1},
      {"matmul_tiles_pattern", "gemm", Result::kGemm, Op::kMatmulTiles,
       {Role::kA, Role::kB, Role::kC}, {Form::kExactCB, Form::kExactCB},
       Form::kAccumulator, SideEffect::kDst, std::nullopt,
       {}, Args({{Role::kA, 0}, {Role::kB, 1}, {Role::kC, 2}}), 4},
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
