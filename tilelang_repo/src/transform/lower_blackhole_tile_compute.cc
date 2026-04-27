/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_blackhole_tile_compute.cc
 * \brief Explicit Blackhole tile compute selection and leaf TT-Metal sequences.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_utils.h"

#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::Buffer;
using tir::Call;
using tir::CallNode;
using tir::Evaluate;
using tir::SeqStmt;
using tir::Stmt;
using tir::builtin::blackhole_add_bcast_cols_init_short;
using tir::builtin::blackhole_add_tiles;
using tir::builtin::blackhole_add_tiles_bcast_cols;
using tir::builtin::blackhole_add_tiles_init;
using tir::builtin::blackhole_binary_max_tile;
using tir::builtin::blackhole_binary_max_tile_init;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_copy_tile_to_dst_init_short;
using tir::builtin::blackhole_exp2_tile;
using tir::builtin::blackhole_exp2_tile_init;
using tir::builtin::blackhole_fill_fragment;
using tir::builtin::blackhole_mul_bcast_cols_init_short;
using tir::builtin::blackhole_mul_tiles;
using tir::builtin::blackhole_mul_tiles_bcast_cols;
using tir::builtin::blackhole_mul_tiles_init;
using tir::builtin::blackhole_pack_reconfig_data_format;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_recip_tile;
using tir::builtin::blackhole_recip_tile_init;
using tir::builtin::blackhole_reconfig_data_format;
using tir::builtin::blackhole_reduce_init;
using tir::builtin::blackhole_reduce_tile;
using tir::builtin::blackhole_reduce_uninit;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_tile_regs_wait;
using tvm::Integer;

namespace {

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Void(), op, args));
}

PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

int64_t ComputeStaticElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
}

int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  ICHECK_GE(value, 0);
  return static_cast<int>((value + divisor - 1) / divisor);
}

bool IsFloatImmValue(const PrimExpr& expr, double expected) {
  if (const auto* imm = expr.as<FloatImmNode>()) {
    return imm->value == expected;
  }
  if (const auto* cast = expr.as<CastNode>()) {
    return IsFloatImmValue(cast->value, expected);
  }
  return false;
}

bool IsInfinityExpr(const PrimExpr& expr) {
  const auto* call = expr.as<CallNode>();
  if (!call || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  Op op = Downcast<Op>(call->op);
  return op->name == "tir.infinity" || op->name == "tl.infinity";
}

bool IsZeroValue(const PrimExpr& expr) {
  return tir::is_zero(expr) || IsFloatImmValue(expr, 0.0);
}

bool IsNegInfValue(const PrimExpr& expr) {
  if (const auto* cast = expr.as<CastNode>()) {
    return IsNegInfValue(cast->value);
  }
  if (const auto* mul = expr.as<MulNode>()) {
    return ((IsFloatImmValue(mul->a, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->a)) &&
            IsFloatImmValue(mul->b, -1.0)) ||
           ((IsFloatImmValue(mul->b, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->b)) &&
            IsFloatImmValue(mul->a, -1.0));
  }
  return IsFloatImmValue(expr, -std::numeric_limits<double>::infinity());
}

std::string GetStorageScope(const Buffer& buffer) {
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

int64_t ProductIntegerArrayField(const Map<String, Any>& map, const char* key,
                                 int64_t default_value = 0) {
  auto it = map.find(String(key));
  if (it == map.end()) {
    return default_value;
  }
  int64_t product = 1;
  for (const Integer& dim : Downcast<Array<Integer>>((*it).second)) {
    if (dim->value <= 0) {
      return default_value;
    }
    product *= dim->value;
  }
  return product;
}

class ExactTileComputeEmitter {
 public:
  explicit ExactTileComputeEmitter(std::vector<Stmt>* stmts) : stmts_(stmts) {
    ICHECK(stmts_ != nullptr);
  }

  void Append(const Op& op, const std::vector<PrimExpr>& args) {
    stmts_->push_back(MakeBlackholeCall(op, args));
  }

  void Reserve(int cb_id, int num_tiles) {
    Append(blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void Wait(int cb_id, int num_tiles) {
    Append(blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void Pop(int cb_id, int num_tiles) {
    Append(blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void PopIfOwned(int cb_id, int num_tiles, bool borrowed_live) {
    if (!borrowed_live) {
      Pop(cb_id, num_tiles);
    }
  }

  void Push(int cb_id, int num_tiles) {
    Append(blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void ReconfigDataFormat(int lhs_cb_id, int rhs_cb_id) {
    Append(blackhole_reconfig_data_format(), {IntImm32(lhs_cb_id), IntImm32(rhs_cb_id)});
  }

  void PackTile(int out_cb_id, int out_tile) {
    Append(blackhole_pack_reconfig_data_format(), {IntImm32(out_cb_id)});
    Append(blackhole_pack_tile(), {IntImm32(0), IntImm32(out_cb_id), IntImm32(out_tile)});
  }

  template <typename EmitBody>
  void EmitPackedTile(int out_cb_id, int out_tile, EmitBody emit_body) {
    Append(blackhole_tile_regs_acquire(), {});
    emit_body(*this);
    Append(blackhole_tile_regs_commit(), {});
    Append(blackhole_tile_regs_wait(), {});
    PackTile(out_cb_id, out_tile);
    Append(blackhole_tile_regs_release(), {});
  }

 private:
  std::vector<Stmt>* stmts_;
};

}  // namespace

bool PlanTTKernelABI::MatchExplicitTileReduce(const CallNode* op,
                                              RowReductionMatch* match) const {
  if (!op || !match || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != "tl.tileop.reduce") {
    return false;
  }
  ICHECK_GE(op->args.size(), 5U)
      << "tl.tileop.reduce must carry src, dst, reduce_type, dim, and clear";

  const Buffer logical_src = NormalizeToBufferRegion(op->args[0])->buffer;
  const Buffer logical_dst = NormalizeToBufferRegion(op->args[1])->buffer;
  const Buffer src = ResolvePhysicalComputeBuffer(logical_src);
  const Buffer dst = ResolvePhysicalComputeBuffer(logical_dst);

  const auto* kind_imm = op->args[2].as<StringImmNode>();
  ICHECK(kind_imm) << "tl.tileop.reduce requires string reduce_type";
  const std::string kind = kind_imm->value;
  ICHECK(kind == "sum" || kind == "max")
      << "Blackhole explicit tile reduce currently supports sum/max only, got "
      << kind;

  const auto* dim_imm = op->args[3].as<IntImmNode>();
  ICHECK(dim_imm) << "tl.tileop.reduce requires static dim";
  const std::vector<int64_t> src_shape = GetLogicalBufferShape(src);
  ICHECK(!src_shape.empty())
      << "Blackhole explicit tile reduce requires a static logical source shape for "
      << BufferIdentityName(src);
  int64_t dim = dim_imm->value;
  if (dim < 0) {
    dim += static_cast<int64_t>(src_shape.size());
  }
  ICHECK_GE(dim, 0) << "Blackhole explicit tile reduce dim out of range: " << dim_imm->value;
  ICHECK_LT(dim, static_cast<int64_t>(src_shape.size()))
      << "Blackhole explicit tile reduce dim " << dim << " out of rank "
      << src_shape.size();
  ICHECK_EQ(dim, static_cast<int64_t>(src_shape.size() - 1))
      << "Blackhole explicit tile reduce Phase A supports row reductions over "
      << "the innermost logical axis only";

  const auto clear_bool = op->args[4].as<Bool>();
  ICHECK(clear_bool) << "tl.tileop.reduce requires static clear bool";
  const bool clear = clear_bool.value()->value;

  const std::vector<int64_t> dst_shape = GetLogicalBufferShape(dst);
  const int64_t dst_elements =
      dst_shape.empty() ? GetLogicalBufferElementCount(dst)
                        : ComputeStaticElementCount(dst_shape);
  match->src = src;
  match->dst = dst;
  match->num_elements = IntImm32(static_cast<int>(std::max<int64_t>(1, dst_elements)));
  match->row_width = IntImm32(static_cast<int>(std::max<int64_t>(1, src_shape[dim])));
  match->kind = kind;
  match->grouped = dst_elements > 1;
  match->clear = clear;
  match->accumulate_existing = !clear;
  return true;
}

bool PlanTTKernelABI::MatchExplicitTileTypecast(const CallNode* op,
                                                FragmentCastMatch* match) const {
  if (!op || !match || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName || op->args.size() < 4U) {
    return false;
  }
  const auto* operation_imm = op->args[0].as<StringImmNode>();
  if (!operation_imm ||
      operation_imm->value != blackhole_tile_compute_schema::kTypecastTile ||
      !IsBufferLikeExpr(op->args[1]) || !IsBufferLikeExpr(op->args[2])) {
    return false;
  }
  match->src = ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[1])->buffer);
  match->dst = NormalizeToBufferRegion(op->args[2])->buffer;
  match->src_offset = IntImm(DataType::Int(32), 0);
  match->dst_offset = IntImm(DataType::Int(32), 0);
  match->num_elements = op->args[3];
  match->row_width = PrimExpr();
  const auto [logical_src_rows, logical_src_cols] = GetLogicalMatrixShape(match->src);
  if (logical_src_rows > 0 && logical_src_cols > 0) {
    match->row_width = IntImm(DataType::Int(32), static_cast<int>(logical_src_cols));
    match->num_elements =
        IntImm(DataType::Int(32), static_cast<int>(logical_src_rows * logical_src_cols));
  }
  return true;
}

Stmt PlanTTKernelABI::LowerExplicitBlackholeTileCompute(const CallNode* op) {
  if (!op || !op->op->IsInstance<OpNode>()) {
    return Stmt();
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName ||
      op->args.empty()) {
    return Stmt();
  }
  const auto* operation_imm = op->args[0].as<StringImmNode>();
  ICHECK(operation_imm) << "tl.tileop.blackhole_compute requires string operation name";
  const std::string operation = operation_imm->value;
  auto buffer_arg = [&](size_t index) -> Buffer {
    ICHECK_LT(index, op->args.size())
        << "tl.tileop.blackhole_compute missing buffer argument " << index
        << " for operation " << operation;
    return ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[index])->buffer);
  };
  auto prim_arg = [&](size_t index) -> PrimExpr {
    ICHECK_LT(index, op->args.size())
        << "tl.tileop.blackhole_compute missing scalar argument " << index
        << " for operation " << operation;
    return op->args[index];
  };
  auto string_arg = [&](size_t index) -> std::string {
    const auto* imm = prim_arg(index).as<StringImmNode>();
    ICHECK(imm) << "tl.tileop.blackhole_compute expects string argument " << index
                << " for operation " << operation;
    return imm->value;
  };

  if (operation == blackhole_tile_compute_schema::kFillTile) {
    return GenerateFillTileSequence(buffer_arg(1), prim_arg(2), prim_arg(3));
  }
  if (operation == blackhole_tile_compute_schema::kCopyTile) {
    return GenerateCopyTileSequence(buffer_arg(1), buffer_arg(2), prim_arg(3));
  }
  if (operation == blackhole_tile_compute_schema::kTypecastTile) {
    FragmentCastMatch match;
    ICHECK(MatchExplicitTileTypecast(op, &match))
        << "tl.tileop.blackhole_compute typecast_tile requires source and destination regions";
    const int cast_order_index =
        ResolveCurrentBufferTransferOrder(match.src, match.dst,
                                          current_lowering_order_index_);
    const bool publish_cb =
        !select_compute_builtins_only_ &&
        ShouldPublishBufferResult(match.dst, cast_order_index);
    std::vector<Stmt> prefix;
    std::vector<Stmt> suffix;
    if (publish_cb) {
      ValidatePublishedBufferSourceEdge(match.src, match.dst);
      const bool source_has_live_cb =
          buffer_live_form_cb_by_buffer_identity_.count(BufferIdentityName(match.src)) != 0U;
      const bool dst_has_republish_fact = FindBufferMaterializationFact(match.dst) != nullptr;
      if (!(source_has_live_cb && dst_has_republish_fact)) {
        AppendPublishedBufferSourceMaterialization(match.src, cast_order_index, &prefix, &suffix);
      }
    }
    prefix.push_back(GenerateFragmentCastSequence(match, publish_cb, cast_order_index));
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return SeqStmt::Flatten(prefix);
  }
  if (operation == blackhole_tile_compute_schema::kBinaryMaxTile) {
    return GenerateBinaryMaxTileSequence(buffer_arg(1), buffer_arg(2));
  }
  if (operation == blackhole_tile_compute_schema::kMulTiles ||
      operation == blackhole_tile_compute_schema::kAddTiles) {
    return GenerateBinaryTileSequence(buffer_arg(1), buffer_arg(2), operation);
  }
  if (operation == blackhole_tile_compute_schema::kMulTilesBcastCols) {
    return GenerateMulTilesBcastColsSequence(string_arg(1), buffer_arg(2), buffer_arg(3),
                                             prim_arg(4), prim_arg(5));
  }
  if (operation == blackhole_tile_compute_schema::kExp2Tile) {
    const std::string mode = string_arg(1);
    if (mode == "bcast_cols") {
      return GenerateExp2TileLeafDAGSequence(mode, buffer_arg(2), buffer_arg(3), buffer_arg(4),
                                             prim_arg(5), prim_arg(6), prim_arg(7), prim_arg(8));
    }
    if (mode == "binary") {
      return GenerateExp2TileLeafDAGSequence(mode, buffer_arg(2), buffer_arg(3), buffer_arg(4),
                                             prim_arg(5), prim_arg(6), PrimExpr(), PrimExpr());
    }
    ICHECK(false) << "Unsupported Blackhole exp2_tile mode: " << mode;
  }
  ICHECK(false) << "Unsupported Blackhole tile compute operation: " << operation;
  return Stmt();
}

Stmt PlanTTKernelABI::GenerateRowReductionSequence(const RowReductionMatch& match) {
  PrimExpr accumulator_fill_value;
  const bool accumulator_is_identity_fill =
      match.accumulate_existing && TryGetLastFragmentFillValue(match.dst, &accumulator_fill_value) &&
      ((match.kind == "sum" && IsZeroValue(accumulator_fill_value)) ||
       (match.kind == "max" && IsNegInfValue(accumulator_fill_value)));
  const bool accumulate_existing = match.accumulate_existing && !accumulator_is_identity_fill;
  ExactTiledCBValue src_in = CreateRowReductionInputCBValue(match.src);
  ExactTiledCBValue out;
  ExactTiledCBValue reduced;
  if (accumulate_existing) {
    reduced = CreateEmptyExactTiledCBValue(match.dst, "reduce_partial");
    out = reduced.num_tiles == 1 ? reduced : CreateEmptyExactTiledCBValue(match.dst, "reduce_out");
  } else {
    out = CreateEmptyExactTiledCBValue(match.dst, "reduce_out");
    reduced = out;
  }
  const bool reuse_reduced_as_output =
      accumulate_existing && out.cb_id == reduced.cb_id;
  ExactTiledCBValue dst_in;
  if (accumulate_existing) {
    dst_in = CreateExactInputCBValue(match.dst, "reduce_accum");
    ICHECK_EQ(dst_in.num_tiles, reduced.num_tiles)
        << "Blackhole accumulating row reduction requires accumulator and reduction tile counts "
           "to match";
    ICHECK_EQ(out.num_tiles, reduced.num_tiles)
        << "Blackhole accumulating row reduction requires output and reduction tile counts to "
           "match";
  }
  ExactTiledCBValue scaler = CreateReduceScalerExactTiledCBValue();
  RecordExactComputeOpPlan("reduce", "reduce_tile",
                           {{"input", match.src, "identity"},
                            {"scaler", scaler.buffer, "identity"},
                            {"output", accumulate_existing ? reduced.buffer : match.dst, "identity"}});
  if (accumulate_existing) {
    RecordExactComputeOpPlan("binary", match.kind == "sum" ? "add_tiles" : "binary_max_tile",
                             {{"lhs", match.dst, "identity"},
                              {"rhs", reduced.buffer, "identity"},
                              {"output", match.dst, "identity"}});
  }

  const Buffer scaler_local = scaler.buffer;
  const Stmt scaler_publish =
      PublishConstantToExactTiledCB(scaler_local, make_const(match.src->dtype, 1.0), scaler);

  const int tiles_per_reduction = std::max(1, src_in.num_tiles / std::max(1, out.num_tiles));
  std::vector<Stmt> stmts;
  if (Stmt publish_src = PublishExactInputToTiledCB(match.src, &src_in); publish_src.defined()) {
    stmts.push_back(publish_src);
  }
  if (accumulate_existing) {
    if (Stmt publish_dst = PublishExactInputToTiledCB(match.dst, &dst_in);
        publish_dst.defined()) {
      stmts.push_back(publish_dst);
    }
  }
  stmts.push_back(scaler_publish);
  if (accumulate_existing) {
    MarkExactCBValuesOverlap({src_in.cb_id, dst_in.cb_id, scaler.cb_id, reduced.cb_id,
                              out.cb_id});
  } else {
    MarkExactCBValuesOverlap({src_in.cb_id, scaler.cb_id, out.cb_id});
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(reduced.cb_id), IntImm32(reduced.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scaler.cb_id), IntImm32(scaler.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id)}));
  for (int out_tile = 0; out_tile < reduced.num_tiles; ++out_tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_reduce_init(),
        {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(reduced.cb_id),
         StringImm(match.kind), StringImm("row")}));
    for (int tile = 0; tile < tiles_per_reduction; ++tile) {
      const int src_tile = out_tile * tiles_per_reduction + tile;
      stmts.push_back(MakeBlackholeCall(
          blackhole_reduce_tile(),
          {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(src_tile), IntImm32(0),
           IntImm32(0), StringImm(match.kind), StringImm("row")}));
    }
    stmts.push_back(
        MakeBlackholeCall(blackhole_reduce_uninit(), {StringImm(match.kind), StringImm("row")}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(reduced.cb_id)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_pack_tile(), {IntImm32(0), IntImm32(reduced.cb_id), IntImm32(out_tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  if (!src_in.borrowed_live) {
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scaler.cb_id), IntImm32(scaler.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(reduced.cb_id), IntImm32(reduced.num_tiles)}));
  if (accumulate_existing) {
    if (!reuse_reduced_as_output) {
      stmts.push_back(
          MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
    }
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(), {IntImm32(reduced.cb_id), IntImm32(reduced.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                      {IntImm32(dst_in.cb_id), IntImm32(reduced.cb_id)}));
    if (match.kind == "sum") {
      stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                        {IntImm32(dst_in.cb_id), IntImm32(reduced.cb_id)}));
    }
    for (int tile = 0; tile < out.num_tiles; ++tile) {
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
      if (match.kind == "sum") {
        stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                          {IntImm32(dst_in.cb_id), IntImm32(reduced.cb_id),
                                           IntImm32(tile), IntImm32(tile), IntImm32(0)}));
      } else {
        stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(),
                                          {IntImm32(dst_in.cb_id)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_copy_tile(), {IntImm32(dst_in.cb_id), IntImm32(tile), IntImm32(0)}));
        stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(),
                                          {IntImm32(reduced.cb_id)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_copy_tile(), {IntImm32(reduced.cb_id), IntImm32(tile), IntImm32(1)}));
        stmts.push_back(MakeBlackholeCall(blackhole_binary_max_tile_init(), {}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_binary_max_tile(),
            {IntImm32(0), IntImm32(1), IntImm32(0), StringImm("C")}));
      }
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
      if (reuse_reduced_as_output) {
        ICHECK_EQ(out.num_tiles, 1)
            << "Blackhole in-place accumulating row reduction currently requires one output tile";
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_pop_front(), {IntImm32(reduced.cb_id), IntImm32(reduced.num_tiles)}));
        stmts.push_back(
            MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
      }
      stmts.push_back(
          MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
    }
    if (!dst_in.borrowed_live) {
      stmts.push_back(
          MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
    }
    if (!reuse_reduced_as_output) {
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(reduced.cb_id), IntImm32(reduced.num_tiles)}));
    }
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  }
  RecordExactOutputLiveForm(match.dst, out);

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(scaler_local, body);
  body = tir::Allocate(scaler_local->data, scaler_local->dtype, scaler_local->shape, Bool(1), body);
  body = AttachExactOutputLiveFormMarker(match.dst, out, body);
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateFillTileSequence(const Buffer& dst, const PrimExpr& value,
                                               const PrimExpr& requested_num_elements) {
  const FutureBufferUses future_uses =
      ClassifyFutureBufferUses(dst, current_lowering_order_index_);
  if (GetStorageScope(dst) == "local.fragment" &&
      !future_uses.has_compute_consume && !future_uses.has_transport_consume &&
      !future_uses.has_reference && FindBufferMaterializationFact(dst) == nullptr) {
    return Evaluate(IntImm32(0));
  }

  PrimExpr num_elements = requested_num_elements;
  int64_t physical_local_extent = 0;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(dst)) {
    physical_local_extent = ProductIntegerArrayField(*spec, schema_key::kLocalShape, int64_t{0});
  }
  if (physical_local_extent > 0) {
    num_elements = IntImm(DataType::Int(32), static_cast<int>(physical_local_extent));
  }
  const int64_t logical_extent = GetLogicalBufferElementCount(dst);
  if (physical_local_extent <= 0 && logical_extent > 1) {
    bool should_promote_extent = !num_elements.defined();
    if (const auto* int_imm = num_elements.as<IntImmNode>()) {
      should_promote_extent = int_imm->value < logical_extent;
    }
    if (should_promote_extent) {
      num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }

  ClearSelectedSourceLiveProducer(dst);
  ClearTiledCBLiveFormAliases(dst);
  for (const std::string& identity : CollectBufferFlowIdentities(dst)) {
    last_fragment_fill_value_by_buffer_identity_[identity] = value;
  }
  if (const VarNode* data = BufferDataIdentity(dst)) {
    last_fragment_fill_value_by_data_[data] = value;
  }
  return MaybeWrapComputeSegment(MakeBlackholeCall(
      tir::builtin::blackhole_fill_fragment(), {dst->data, num_elements, value}));
}

Stmt PlanTTKernelABI::GenerateCopyTileSequence(const Buffer& src, const Buffer& dst,
                                               const PrimExpr& num_elements) {
  ExactTiledCBValue live_value;
  if (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
      TryCreateLiveExactTiledCBValue(src, &live_value)) {
    RecordTiledCBLiveFormAliases(dst, live_value.cb_id);
    InvalidateLastFragmentFillValue(dst);
    return Evaluate(IntImm32(0));
  }

  PrimExpr fill_value;
  if (TryGetLastFragmentFillValue(src, &fill_value)) {
    const std::string dst_name = BufferIdentityName(dst);
    if (!dst_name.empty()) {
      ClearTiledCBLiveFormIdentity(dst_name);
      last_fragment_fill_value_by_buffer_identity_[dst_name] = fill_value;
    }
    if (const VarNode* data = BufferDataIdentity(dst)) {
      last_fragment_fill_value_by_data_[data] = fill_value;
    }
    return Evaluate(IntImm32(0));
  }

  FragmentCastMatch cast_match;
  cast_match.dst = dst;
  cast_match.src = src;
  cast_match.dst_offset = IntImm(DataType::Int(32), 0);
  cast_match.src_offset = IntImm(DataType::Int(32), 0);
  cast_match.num_elements = num_elements;
  if (!cast_match.num_elements.defined()) {
    const int64_t logical_extent = GetLogicalVectorLength(dst);
    if (logical_extent > 1) {
      cast_match.num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }
  return GenerateFragmentCastSequence(cast_match, /*publish_cb=*/false);
}

Stmt PlanTTKernelABI::GenerateBinaryMaxTileSequence(const Buffer& dst, const Buffer& rhs) {
  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, "binary_max_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, "binary_max_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(dst, "binary_max_out");
  RecordExactComputeOpPlan("binary", "binary_max_tile",
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs, "identity"},
                            {"output", dst, "identity"}});

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(lhs_in.cb_id, lhs_in.num_tiles);
  emit.Wait(rhs_in.cb_id, rhs_in.num_tiles);
  emit.ReconfigDataFormat(lhs_in.cb_id, rhs_in.cb_id);
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    emit.EmitPackedTile(out.cb_id, tile, [&](ExactTileComputeEmitter& tile_emit) {
      tile_emit.Append(blackhole_copy_tile_to_dst_init_short(), {IntImm32(lhs_in.cb_id)});
      tile_emit.Append(blackhole_copy_tile(),
                       {IntImm32(lhs_in.cb_id), IntImm32(tile), IntImm32(0)});
      tile_emit.Append(blackhole_copy_tile_to_dst_init_short(), {IntImm32(rhs_in.cb_id)});
      tile_emit.Append(blackhole_copy_tile(),
                       {IntImm32(rhs_in.cb_id), IntImm32(tile), IntImm32(1)});
      tile_emit.Append(blackhole_binary_max_tile_init(), {});
      tile_emit.Append(blackhole_binary_max_tile(),
                       {IntImm32(0), IntImm32(1), IntImm32(0), StringImm("C")});
    });
  }
  emit.PopIfOwned(lhs_in.cb_id, lhs_in.num_tiles, lhs_in.borrowed_live);
  emit.PopIfOwned(rhs_in.cb_id, rhs_in.num_tiles, rhs_in.borrowed_live);
  emit.Push(out.cb_id, out.num_tiles);
  RecordExactOutputLiveForm(dst, out);
  Stmt body = AttachExactOutputLiveFormMarker(dst, out, SeqStmt::Flatten(stmts));
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateBinaryTileSequence(const Buffer& dst,
                                                  const Buffer& rhs,
                                                  const std::string& operation_name) {
  ICHECK(operation_name == blackhole_tile_compute_schema::kMulTiles ||
         operation_name == blackhole_tile_compute_schema::kAddTiles)
      << "Unsupported explicit binary tile operation " << operation_name;
  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, operation_name + "_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, operation_name + "_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(dst, operation_name + "_out");
  RecordExactComputeOpPlan("binary", operation_name,
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs, "identity"},
                            {"output", dst, "identity"}});

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(lhs_in.cb_id, lhs_in.num_tiles);
  emit.Wait(rhs_in.cb_id, rhs_in.num_tiles);
  emit.ReconfigDataFormat(lhs_in.cb_id, rhs_in.cb_id);
  if (operation_name == blackhole_tile_compute_schema::kMulTiles) {
    emit.Append(blackhole_mul_tiles_init(), {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id)});
  } else {
    emit.Append(blackhole_add_tiles_init(), {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id)});
  }
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    emit.EmitPackedTile(out.cb_id, tile, [&](ExactTileComputeEmitter& tile_emit) {
      if (operation_name == blackhole_tile_compute_schema::kMulTiles) {
        tile_emit.Append(blackhole_mul_tiles(),
                         {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id),
                          IntImm32(tile), IntImm32(tile), IntImm32(0)});
      } else {
        tile_emit.Append(blackhole_add_tiles(),
                         {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id),
                          IntImm32(tile), IntImm32(tile), IntImm32(0)});
      }
    });
  }
  emit.PopIfOwned(lhs_in.cb_id, lhs_in.num_tiles, lhs_in.borrowed_live);
  emit.PopIfOwned(rhs_in.cb_id, rhs_in.num_tiles, rhs_in.borrowed_live);
  emit.Push(out.cb_id, out.num_tiles);
  RecordExactOutputLiveForm(dst, out);
  Stmt body = AttachExactOutputLiveFormMarker(dst, out, SeqStmt::Flatten(stmts));
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateMulTilesBcastColsSequence(
    const std::string& kind, const Buffer& dst, const Buffer& rhs,
    const PrimExpr& num_elements, const PrimExpr& row_width) {
  (void)num_elements;
  (void)row_width;
  ICHECK(kind == "mul" || kind == "div")
      << "mul_tiles_bcast_cols lowering supports mul/div forms only, got " << kind;

  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, "mul_bcast_cols_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, "mul_bcast_cols_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(dst, "mul_bcast_cols_out");
  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(dst);
  const int tiles_per_row =
      logical_rows > 0 && logical_cols > 0
          ? std::max(1, CeilDivToInt(logical_cols, kBlackholeTileCols))
          : std::max(1, lhs_in.num_tiles / std::max(1, rhs_in.num_tiles));

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }

  ExactTiledCBValue* rhs_operand = &rhs_in;
  ExactTiledCBValue rhs_full;
  ExactTiledCBValue rhs_full_ones;
  ExactTiledCBValue reciprocal;
  if (kind == "div") {
    rhs_full = CreateEmptyExactTiledCBValue(dst, "mul_bcast_cols_rhs_full");
    rhs_full_ones = CreateConstantExactTiledCBValue(dst->dtype, "mul_bcast_cols_one");
    reciprocal = CreateEmptyExactTiledCBValue(dst, "recip_tile_out");
    MarkExactCBValuesOverlap({rhs_in.cb_id, rhs_full_ones.cb_id, rhs_full.cb_id,
                              reciprocal.cb_id});
    RecordExactComputeOpPlan("binary", "mul_tiles_bcast_cols",
                             {{"lhs", rhs_full_ones.buffer, "identity"},
                              {"rhs", rhs, "broadcast"},
                              {"output", rhs_full.buffer, "identity"}});
    RecordExactComputeOpPlan("unary", "recip_tile",
                             {{"input", rhs_full.buffer, "identity"},
                              {"output", reciprocal.buffer, "identity"}});

    stmts.push_back(PublishConstantToExactTiledCB(
        rhs_full_ones.buffer, make_const(dst->dtype, 1.0), rhs_full_ones));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(rhs_full.cb_id),
                                       IntImm32(rhs_full.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(rhs_full_ones.cb_id),
                                       IntImm32(rhs_full_ones.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                      {IntImm32(rhs_full_ones.cb_id),
                                       IntImm32(rhs_in.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_bcast_cols_init_short(),
                                      {IntImm32(rhs_full_ones.cb_id),
                                       IntImm32(rhs_in.cb_id)}));
    for (int tile = 0; tile < rhs_full.num_tiles; ++tile) {
      const int rhs_tile = tile / tiles_per_row;
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_mul_tiles_bcast_cols(),
          {IntImm32(rhs_full_ones.cb_id), IntImm32(rhs_in.cb_id), IntImm32(0),
           IntImm32(rhs_tile), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                        {IntImm32(rhs_full.cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(rhs_full.cb_id),
                                         IntImm32(tile)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(rhs_full_ones.cb_id),
                                       IntImm32(rhs_full_ones.num_tiles)}));
    if (!rhs_in.borrowed_live) {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                        {IntImm32(rhs_in.cb_id),
                                         IntImm32(rhs_in.num_tiles)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(rhs_full.cb_id),
                                       IntImm32(rhs_full.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(reciprocal.cb_id),
                                       IntImm32(reciprocal.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(rhs_full.cb_id),
                                       IntImm32(rhs_full.num_tiles)}));
    for (int tile = 0; tile < reciprocal.num_tiles; ++tile) {
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(),
                                        {IntImm32(rhs_full.cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                        {IntImm32(rhs_full.cb_id), IntImm32(tile),
                                         IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_recip_tile_init(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_recip_tile(), {IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                        {IntImm32(reciprocal.cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(reciprocal.cb_id),
                                         IntImm32(tile)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(rhs_full.cb_id),
                                       IntImm32(rhs_full.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(reciprocal.cb_id),
                                       IntImm32(reciprocal.num_tiles)}));
    rhs_operand = &reciprocal;
  }

  RecordExactComputeOpPlan("binary", kind == "div" ? "mul_tiles" : "mul_tiles_bcast_cols",
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs_operand->buffer,
                             kind == "div" ? "identity" : "broadcast"},
                            {"output", dst, "identity"}});
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_operand->cb_id, out.cb_id});
  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(rhs_operand->cb_id),
                                     IntImm32(rhs_operand->num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(lhs_in.cb_id),
                                     IntImm32(rhs_operand->cb_id)}));
  if (kind == "div") {
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                      {IntImm32(lhs_in.cb_id),
                                       IntImm32(rhs_operand->cb_id)}));
  } else {
    stmts.push_back(MakeBlackholeCall(blackhole_mul_bcast_cols_init_short(),
                                      {IntImm32(lhs_in.cb_id),
                                       IntImm32(rhs_operand->cb_id)}));
  }
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    const int rhs_tile = tile / tiles_per_row;
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    if (kind == "div") {
      stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                        {IntImm32(lhs_in.cb_id),
                                         IntImm32(rhs_operand->cb_id),
                                         IntImm32(tile), IntImm32(tile), IntImm32(0)}));
    } else {
      stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_bcast_cols(),
                                        {IntImm32(lhs_in.cb_id),
                                         IntImm32(rhs_operand->cb_id),
                                         IntImm32(tile), IntImm32(rhs_tile), IntImm32(0)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                      {IntImm32(out.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  if (!lhs_in.borrowed_live) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  }
  if (kind == "mul") {
    if (!rhs_in.borrowed_live) {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                        {IntImm32(rhs_in.cb_id),
                                         IntImm32(rhs_in.num_tiles)}));
    }
  } else {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(reciprocal.cb_id),
                                       IntImm32(reciprocal.num_tiles)}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  RecordExactOutputLiveForm(dst, out);
  Stmt body = AttachExactOutputLiveFormMarker(dst, out, SeqStmt::Flatten(stmts));
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateExp2TileLeafDAGSequence(
    const std::string& mode, const Buffer& dst, const Buffer& lhs, const Buffer& rhs,
    const PrimExpr& lhs_scale_value, const PrimExpr& rhs_scale_value,
    const PrimExpr& row_width, const PrimExpr& num_elements) {
  (void)row_width;
  (void)num_elements;
  const bool use_bcast_cols = mode == "bcast_cols";
  ICHECK(use_bcast_cols || mode == "binary")
      << "Unsupported exp2_tile leaf DAG mode: " << mode;

  ExactTiledCBValue lhs_in = CreateExactInputCBValue(lhs, "exp2_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, "exp2_rhs");
  ExactTiledCBValue lhs_scale =
      CreateConstantExactTiledCBValue(lhs->dtype, "exp2_lhs_scale");
  ExactTiledCBValue rhs_scale =
      CreateConstantExactTiledCBValue(rhs->dtype, "exp2_rhs_scale");
  ExactTiledCBValue scaled_lhs = CreateEmptyExactTiledCBValue(dst, "exp2_scaled_lhs");
  ExactTiledCBValue scaled_rhs =
      CreateEmptyExactTiledCBValue(use_bcast_cols ? rhs : dst, "exp2_scaled_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(dst, "exp2_out");

  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", lhs, "identity"},
                            {"rhs", lhs_scale.buffer, "identity"},
                            {"output", scaled_lhs.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", rhs, "identity"},
                            {"rhs", rhs_scale.buffer, "identity"},
                            {"output", scaled_rhs.buffer, "identity"}});
  RecordExactComputeOpPlan("binary",
                           use_bcast_cols ? "add_tiles_bcast_cols" : "add_tiles",
                           {{"lhs", scaled_lhs.buffer, "identity"},
                            {"rhs", scaled_rhs.buffer,
                             use_bcast_cols ? "broadcast" : "identity"},
                            {"output", out.buffer, "identity"}});
  RecordExactComputeOpPlan("unary", "exp2_tile",
                           {{"input", out.buffer, "identity"},
                            {"output", dst, "identity"}});

  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(dst);
  const int tiles_per_row =
      logical_rows > 0 && logical_cols > 0
          ? std::max(1, CeilDivToInt(logical_cols, kBlackholeTileCols))
          : std::max(1, lhs_in.num_tiles / std::max(1, rhs_in.num_tiles));

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(lhs, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }
  stmts.push_back(PublishConstantToExactTiledCB(lhs_scale.buffer, lhs_scale_value, lhs_scale));
  stmts.push_back(PublishConstantToExactTiledCB(rhs_scale.buffer, rhs_scale_value, rhs_scale));
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, lhs_scale.cb_id, rhs_scale.cb_id,
                            scaled_lhs.cb_id, scaled_rhs.cb_id, out.cb_id});

  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(scaled_lhs.cb_id),
                                     IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(lhs_scale.cb_id),
                                     IntImm32(lhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(lhs_in.cb_id),
                                     IntImm32(lhs_scale.cb_id)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(lhs_in.cb_id),
                                     IntImm32(lhs_scale.cb_id)}));
  for (int tile = 0; tile < scaled_lhs.num_tiles; ++tile) {
    const int lhs_tile = lhs_in.num_tiles == 1 ? 0 : tile % lhs_in.num_tiles;
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(lhs_in.cb_id),
                                       IntImm32(lhs_scale.cb_id),
                                       IntImm32(lhs_tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                      {IntImm32(scaled_lhs.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(scaled_lhs.cb_id),
                                       IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  if (!lhs_in.borrowed_live) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(lhs_scale.cb_id),
                                     IntImm32(lhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(scaled_lhs.cb_id),
                                     IntImm32(scaled_lhs.num_tiles)}));

  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(scaled_rhs.cb_id),
                                     IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(rhs_scale.cb_id),
                                     IntImm32(rhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(rhs_in.cb_id),
                                     IntImm32(rhs_scale.cb_id)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(rhs_in.cb_id),
                                     IntImm32(rhs_scale.cb_id)}));
  for (int tile = 0; tile < scaled_rhs.num_tiles; ++tile) {
    const int rhs_tile = rhs_in.num_tiles == 1 ? 0 : tile % rhs_in.num_tiles;
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(rhs_in.cb_id),
                                       IntImm32(rhs_scale.cb_id),
                                       IntImm32(rhs_tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                      {IntImm32(scaled_rhs.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(scaled_rhs.cb_id),
                                       IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  if (!rhs_in.borrowed_live) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(rhs_scale.cb_id),
                                     IntImm32(rhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(scaled_rhs.cb_id),
                                     IntImm32(scaled_rhs.num_tiles)}));

  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(scaled_lhs.cb_id),
                                     IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(scaled_rhs.cb_id),
                                     IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(scaled_lhs.cb_id),
                                     IntImm32(scaled_rhs.cb_id)}));
  if (use_bcast_cols) {
    stmts.push_back(MakeBlackholeCall(blackhole_add_bcast_cols_init_short(),
                                      {IntImm32(scaled_lhs.cb_id),
                                       IntImm32(scaled_rhs.cb_id)}));
  } else {
    stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                      {IntImm32(scaled_lhs.cb_id),
                                       IntImm32(scaled_rhs.cb_id)}));
  }
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    const int lhs_tile = scaled_lhs.num_tiles == 1 ? 0 : tile % scaled_lhs.num_tiles;
    const int rhs_tile =
        use_bcast_cols ? tile / tiles_per_row
                       : (scaled_rhs.num_tiles == 1 ? 0 : tile % scaled_rhs.num_tiles);
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    if (use_bcast_cols) {
      stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_bcast_cols(),
                                        {IntImm32(scaled_lhs.cb_id),
                                         IntImm32(scaled_rhs.cb_id),
                                         IntImm32(lhs_tile), IntImm32(rhs_tile),
                                         IntImm32(0)}));
    } else {
      stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                        {IntImm32(scaled_lhs.cb_id),
                                         IntImm32(scaled_rhs.cb_id),
                                         IntImm32(lhs_tile), IntImm32(rhs_tile),
                                         IntImm32(0)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile_init(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile(), {IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                      {IntImm32(out.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(scaled_lhs.cb_id),
                                     IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(scaled_rhs.cb_id),
                                     IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  RecordExactOutputLiveForm(dst, out);

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(lhs_scale.buffer, body);
  body = tir::DeclBuffer(rhs_scale.buffer, body);
  body = tir::Allocate(rhs_scale.buffer->data, rhs_scale.buffer->dtype,
                       rhs_scale.buffer->shape, Bool(1), body);
  body = tir::Allocate(lhs_scale.buffer->data, lhs_scale.buffer->dtype,
                       lhs_scale.buffer->shape, Bool(1), body);
  body = AttachExactOutputLiveFormMarker(dst, out, body);
  return MaybeWrapComputeSegment(body);
}

}  // namespace tl
}  // namespace tvm
