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
 * \file blackhole_tile_compute_normalizer.cc
 * \brief Normalize Blackhole compute-region scalar loops into semantic leaf tile ops.
 */

#include "blackhole_tile_compute_normalizer.h"

#include "blackhole_utils.h"
#include "../../op/region.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/utils.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimExpr IntImm32(int value) { return IntImm(DataType::Int(32), value); }

bool IsBlackholeComputeBuffer(const Buffer& buffer) {
  if (!buffer.defined()) {
    return false;
  }
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" ||
         scope == "blackhole.acc";
}

bool IsOneDimensionalBuffer(const Buffer& buffer) {
  return buffer.defined() && buffer->shape.size() == 1U;
}

bool IsLiteralScalarValue(const PrimExpr& expr) {
  bool has_buffer_access = false;
  PostOrderVisit(expr, [&](const ObjectRef& node) {
    has_buffer_access = has_buffer_access || node.as<BufferLoadNode>() != nullptr;
  });
  return !has_buffer_access;
}

bool SameBufferStorage(const Buffer& lhs, const Buffer& rhs) {
  return SameBufferIdentity(lhs, rhs);
}

bool ProvenEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
  if (StructuralEqual()(lhs, rhs)) {
    return true;
  }
  arith::Analyzer analyzer;
  return tir::is_zero(analyzer.Simplify(lhs - rhs));
}

PrimExpr MakeFullRegionExpr(const Buffer& buffer, int access_mask) {
  Array<PrimExpr> indices;
  Array<PrimExpr> args;
  for (const PrimExpr& shape : buffer->shape) {
    indices.push_back(IntImm(shape.dtype(), 0));
  }
  args.push_back(BufferLoad(buffer, indices));
  args.push_back(IntImm32(access_mask));
  for (const PrimExpr& shape : buffer->shape) {
    args.push_back(shape);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

bool MatchLoad(const PrimExpr& expr, Buffer* buffer, PrimExpr* index) {
  const auto* load = expr.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1U ||
      !IsBlackholeComputeBuffer(load->buffer)) {
    return false;
  }
  if (buffer != nullptr) {
    *buffer = load->buffer;
  }
  if (index != nullptr) {
    *index = load->indices[0];
  }
  return true;
}

bool MatchLoadFromBuffer(const PrimExpr& expr, const Buffer& buffer,
                         const PrimExpr& expected_index) {
  Buffer load_buffer;
  PrimExpr load_index;
  if (!MatchLoad(expr, &load_buffer, &load_index) ||
      !SameBufferStorage(load_buffer, buffer)) {
    return false;
  }
  return ProvenEqual(load_index, expected_index);
}

bool MatchScaledLoad(const PrimExpr& expr, Buffer* buffer, PrimExpr* scale) {
  if (MatchLoad(expr, buffer, nullptr)) {
    *scale = make_const(expr.dtype(), 1.0);
    return true;
  }
  const auto* mul = expr.as<MulNode>();
  if (!mul) {
    return false;
  }
  if (MatchLoad(mul->a, buffer, nullptr) && IsLiteralScalarValue(mul->b)) {
    *scale = mul->b;
    return true;
  }
  if (MatchLoad(mul->b, buffer, nullptr) && IsLiteralScalarValue(mul->a)) {
    *scale = mul->a;
    return true;
  }
  return false;
}

bool MatchExp2Call(const PrimExpr& expr, PrimExpr* arg) {
  const auto* call = expr.as<CallNode>();
  if (!call || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op op = Downcast<Op>(call->op);
  if (op->name == "tir.exp2" && call->args.size() == 1U) {
    *arg = call->args[0];
    return true;
  }
  const bool is_extern_exp2 =
      (op.same_as(tir::builtin::call_pure_extern()) ||
       op.same_as(tir::builtin::call_extern())) &&
      call->args.size() == 2U;
  if (!is_extern_exp2) {
    return false;
  }
  const auto* callee = call->args[0].as<StringImmNode>();
  if (!callee) {
    return false;
  }
  const std::string name = callee->value;
  if (name != "exp2" && name != "exp2f" && name != "exp2l" &&
      name != "__exp2f") {
    return false;
  }
  *arg = call->args[1];
  return true;
}

PrimExpr BufferElementCount(const Buffer& buffer, const PrimExpr& fallback) {
  if (!buffer.defined() || buffer->shape.empty()) {
    return fallback;
  }
  arith::Analyzer analyzer;
  PrimExpr product = IntImm32(1);
  for (const PrimExpr& extent : buffer->shape) {
    product = analyzer.Simplify(product * extent);
  }
  return product;
}

Buffer MakeBlackholeTempBufferLike(const Buffer& buffer, const char* suffix,
                                   int* temp_index) {
  ICHECK(buffer.defined());
  ICHECK(temp_index != nullptr);
  const std::string base = BufferIdentityName(buffer).empty()
                               ? std::string("blackhole_tile_tmp")
                               : BufferIdentityName(buffer);
  const std::string name =
      base + suffix + "_" + std::to_string((*temp_index)++);
  return tir::decl_buffer(buffer->shape, buffer->dtype, name,
                          std::string(buffer.scope()));
}

Stmt MakeBlackholeTileComputeCall(const char* operation,
                                  Array<PrimExpr> payload) {
  Array<PrimExpr> args;
  args.push_back(StringImm(operation));
  for (const PrimExpr& arg : payload) {
    args.push_back(arg);
  }
  return Evaluate(Call(DataType::Handle(),
                       Op::Get(blackhole_tile_compute_schema::kOpName), args));
}

Stmt WrapBlackholeTempBuffers(Stmt body, const std::vector<Buffer>& buffers) {
  for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
    body = DeclBuffer(*it, body);
    body = Allocate((*it)->data, (*it)->dtype, (*it)->shape, Bool(1), body);
  }
  return body;
}

struct TileComputeRewriteContext {
  const BufferStoreNode* store{nullptr};
  PrimExpr linear_index;
  PrimExpr num_elements;
};

enum class TileComputeLeafCallKind {
  kUnary,
  kFill,
  kInplaceBinary,
  kBroadcastColsBinary,
};

struct TileComputeLeafCall {
  TileComputeLeafCallKind kind;
  const char* operation{nullptr};
  Buffer src;
  Buffer dst;
  Buffer rhs;
  PrimExpr value;
  PrimExpr num_elements;
  PrimExpr row_width;
};

struct TileComputeRewriteMatch {
  std::vector<TileComputeLeafCall> leaf_calls;
  std::vector<Buffer> temp_buffers;

  void AddUnary(const char* operation, const Buffer& src, const Buffer& dst,
                const PrimExpr& num_elements) {
    leaf_calls.push_back({TileComputeLeafCallKind::kUnary, operation, src, dst,
                          Buffer(), PrimExpr(), num_elements, PrimExpr()});
  }

  void AddFill(const Buffer& dst, const PrimExpr& value,
               const PrimExpr& num_elements) {
    leaf_calls.push_back({TileComputeLeafCallKind::kFill,
                          blackhole_tile_compute_schema::kFillTile, Buffer(),
                          dst, Buffer(), value, num_elements, PrimExpr()});
  }

  void AddInplaceBinary(const char* operation, const Buffer& dst,
                        const Buffer& rhs, const PrimExpr& num_elements) {
    leaf_calls.push_back({TileComputeLeafCallKind::kInplaceBinary, operation,
                          Buffer(), dst, rhs, PrimExpr(), num_elements,
                          PrimExpr()});
  }

  void AddBroadcastColsBinary(const char* operation, const Buffer& dst,
                              const Buffer& rhs,
                              const PrimExpr& num_elements,
                              const PrimExpr& row_width) {
    leaf_calls.push_back({TileComputeLeafCallKind::kBroadcastColsBinary,
                          operation, Buffer(), dst, rhs, PrimExpr(),
                          num_elements, row_width});
  }
};

class TileComputeIRBuilder {
 public:
  explicit TileComputeIRBuilder(int* temp_index) : temp_index_(temp_index) {}

  Buffer TempLike(const Buffer& buffer, const char* suffix) const {
    return MakeBlackholeTempBufferLike(buffer, suffix, temp_index_);
  }

  Stmt Render(const TileComputeRewriteMatch& match) const {
    std::vector<Stmt> stmts;
    stmts.reserve(match.leaf_calls.size());
    for (const TileComputeLeafCall& leaf_call : match.leaf_calls) {
      stmts.push_back(RenderLeafCall(leaf_call));
    }
    return WrapBlackholeTempBuffers(SeqStmt::Flatten(stmts),
                                    match.temp_buffers);
  }

 private:
  Stmt RenderLeafCall(const TileComputeLeafCall& leaf_call) const {
    switch (leaf_call.kind) {
      case TileComputeLeafCallKind::kUnary:
        return MakeBlackholeTileComputeCall(
            leaf_call.operation,
            {MakeFullRegionExpr(leaf_call.src, 1),
             MakeFullRegionExpr(leaf_call.dst, 2), leaf_call.num_elements});
      case TileComputeLeafCallKind::kFill:
        return MakeBlackholeTileComputeCall(
            blackhole_tile_compute_schema::kFillTile,
            {MakeFullRegionExpr(leaf_call.dst, 2), leaf_call.value,
             leaf_call.num_elements});
      case TileComputeLeafCallKind::kInplaceBinary:
        return MakeBlackholeTileComputeCall(
            leaf_call.operation,
            {MakeFullRegionExpr(leaf_call.dst, 3),
             MakeFullRegionExpr(leaf_call.rhs, 1), leaf_call.num_elements});
      case TileComputeLeafCallKind::kBroadcastColsBinary:
        return MakeBlackholeTileComputeCall(
            leaf_call.operation,
            {MakeFullRegionExpr(leaf_call.dst, 3),
             MakeFullRegionExpr(leaf_call.rhs, 1), leaf_call.num_elements,
             leaf_call.row_width});
    }
    ICHECK(false) << "unknown Blackhole tile compute leaf call kind";
    return Stmt();
  }

  int* temp_index_;
};

using TileComputeRewriteMatchFn =
    bool (*)(const TileComputeRewriteContext& ctx,
             TileComputeRewriteMatch* match,
             TileComputeIRBuilder* builder);

struct TileComputeRewriteRule {
  const char* name;
  int benefit;
  TileComputeRewriteMatchFn match;
};

bool MatchTypecastTileRule(const TileComputeRewriteContext& ctx,
                           TileComputeRewriteMatch* match,
                           TileComputeIRBuilder* builder) {
  (void)builder;
  Buffer src;
  PrimExpr src_index;
  const auto* cast = ctx.store->value.as<CastNode>();
  if (!cast || !MatchLoad(cast->value, &src, &src_index) ||
      !ProvenEqual(src_index, ctx.linear_index)) {
    return false;
  }
  match->AddUnary(blackhole_tile_compute_schema::kTypecastTile, src,
                  ctx.store->buffer, ctx.num_elements);
  return true;
}

bool MatchCopyTileRule(const TileComputeRewriteContext& ctx,
                       TileComputeRewriteMatch* match,
                       TileComputeIRBuilder* builder) {
  (void)builder;
  Buffer src;
  PrimExpr src_index;
  if (!MatchLoad(ctx.store->value, &src, &src_index) ||
      SameBufferStorage(src, ctx.store->buffer) ||
      !ProvenEqual(src_index, ctx.linear_index)) {
    return false;
  }
  match->AddUnary(blackhole_tile_compute_schema::kCopyTile, src,
                  ctx.store->buffer, ctx.num_elements);
  return true;
}

bool MatchBinaryMaxTileRule(const TileComputeRewriteContext& ctx,
                            TileComputeRewriteMatch* match,
                            TileComputeIRBuilder* builder) {
  (void)builder;
  auto match_ordered_max = [&](const PrimExpr& lhs,
                               const PrimExpr& rhs) -> bool {
    Buffer rhs_buffer;
    if (!MatchLoadFromBuffer(lhs, ctx.store->buffer, ctx.linear_index) ||
        !MatchLoad(rhs, &rhs_buffer, nullptr)) {
      return false;
    }
    match->AddInplaceBinary(blackhole_tile_compute_schema::kBinaryMaxTile,
                            ctx.store->buffer, rhs_buffer,
                            ctx.num_elements);
    return true;
  };
  const auto* max = ctx.store->value.as<MaxNode>();
  return max && (match_ordered_max(max->a, max->b) ||
                 match_ordered_max(max->b, max->a));
}

bool MatchFmaTileRule(const TileComputeRewriteContext& ctx,
                      TileComputeRewriteMatch* match,
                      TileComputeIRBuilder* builder) {
  (void)builder;
  auto match_ordered_fma = [&](const PrimExpr& mul_expr,
                               const PrimExpr& add_expr) -> bool {
    const auto* mul = mul_expr.as<MulNode>();
    if (!mul) {
      return false;
    }
    Buffer other_mul;
    Buffer add_buffer;
    if (MatchLoadFromBuffer(mul->a, ctx.store->buffer, ctx.linear_index) &&
        MatchLoad(mul->b, &other_mul, nullptr) &&
        MatchLoad(add_expr, &add_buffer, nullptr)) {
      match->AddInplaceBinary(blackhole_tile_compute_schema::kMulTiles,
                              ctx.store->buffer, other_mul,
                              ctx.num_elements);
      match->AddInplaceBinary(blackhole_tile_compute_schema::kAddTiles,
                              ctx.store->buffer, add_buffer,
                              ctx.num_elements);
      return true;
    }
    if (MatchLoadFromBuffer(mul->b, ctx.store->buffer, ctx.linear_index) &&
        MatchLoad(mul->a, &other_mul, nullptr) &&
        MatchLoad(add_expr, &add_buffer, nullptr)) {
      match->AddInplaceBinary(blackhole_tile_compute_schema::kMulTiles,
                              ctx.store->buffer, other_mul,
                              ctx.num_elements);
      match->AddInplaceBinary(blackhole_tile_compute_schema::kAddTiles,
                              ctx.store->buffer, add_buffer,
                              ctx.num_elements);
      return true;
    }
    return false;
  };
  const auto* add = ctx.store->value.as<AddNode>();
  return add && (match_ordered_fma(add->a, add->b) ||
                 match_ordered_fma(add->b, add->a));
}

bool MatchExp2ScaledDifferenceRule(const TileComputeRewriteContext& ctx,
                                   TileComputeRewriteMatch* match,
                                   TileComputeIRBuilder* builder) {
  PrimExpr exp2_arg;
  if (!MatchExp2Call(ctx.store->value, &exp2_arg)) {
    return false;
  }
  const auto* sub = exp2_arg.as<SubNode>();
  if (!sub) {
    return false;
  }
  Buffer lhs;
  Buffer rhs;
  PrimExpr lhs_scale;
  PrimExpr rhs_scale;
  if (!MatchScaledLoad(sub->a, &lhs, &lhs_scale) ||
      !MatchScaledLoad(sub->b, &rhs, &rhs_scale)) {
    return false;
  }

  arith::Analyzer analyzer;
  const bool use_bcast_cols = SameBufferStorage(lhs, ctx.store->buffer);
  PrimExpr row_width =
      IsOneDimensionalBuffer(ctx.store->buffer) ? ctx.num_elements : IntImm32(1);
  Buffer lhs_scale_buffer =
      builder->TempLike(ctx.store->buffer, "_lhs_scale");
  match->temp_buffers.push_back(lhs_scale_buffer);
  if (!SameBufferStorage(lhs, ctx.store->buffer)) {
    match->AddUnary(blackhole_tile_compute_schema::kCopyTile, lhs,
                    ctx.store->buffer, ctx.num_elements);
  }
  match->AddFill(lhs_scale_buffer, lhs_scale,
                 BufferElementCount(lhs_scale_buffer, ctx.num_elements));
  match->AddInplaceBinary(blackhole_tile_compute_schema::kMulTiles,
                          ctx.store->buffer, lhs_scale_buffer,
                          ctx.num_elements);

  Buffer scaled_rhs = builder->TempLike(rhs, "_scaled_rhs");
  Buffer rhs_scale_buffer = builder->TempLike(rhs, "_rhs_scale");
  match->temp_buffers.push_back(scaled_rhs);
  match->temp_buffers.push_back(rhs_scale_buffer);
  const PrimExpr rhs_elements = BufferElementCount(rhs, ctx.num_elements);
  match->AddUnary(blackhole_tile_compute_schema::kCopyTile, rhs, scaled_rhs,
                  rhs_elements);
  match->AddFill(rhs_scale_buffer, analyzer.Simplify(-rhs_scale),
                 BufferElementCount(rhs_scale_buffer, rhs_elements));
  match->AddInplaceBinary(blackhole_tile_compute_schema::kMulTiles,
                          scaled_rhs, rhs_scale_buffer, rhs_elements);
  if (use_bcast_cols) {
    match->AddBroadcastColsBinary(
        blackhole_tile_compute_schema::kAddTilesBcastCols,
        ctx.store->buffer, scaled_rhs, ctx.num_elements, row_width);
  } else {
    match->AddInplaceBinary(blackhole_tile_compute_schema::kAddTiles,
                            ctx.store->buffer, scaled_rhs,
                            ctx.num_elements);
  }
  match->AddUnary(blackhole_tile_compute_schema::kExp2Tile, ctx.store->buffer,
                  ctx.store->buffer, ctx.num_elements);
  return true;
}

bool MatchRowBroadcastMulRule(const TileComputeRewriteContext& ctx,
                              TileComputeRewriteMatch* match,
                              TileComputeIRBuilder* builder) {
  (void)builder;
  auto match_ordered_broadcast = [&](const PrimExpr& self_expr,
                                     const PrimExpr& scalar_expr) -> bool {
    Buffer scalar;
    if (!MatchLoadFromBuffer(self_expr, ctx.store->buffer, ctx.linear_index) ||
        !MatchLoad(scalar_expr, &scalar, nullptr) ||
        SameBufferStorage(scalar, ctx.store->buffer)) {
      return false;
    }
    match->AddBroadcastColsBinary(
        blackhole_tile_compute_schema::kMulTilesBcastCols,
        ctx.store->buffer, scalar, ctx.num_elements, ctx.num_elements);
    return true;
  };
  const auto* mul = ctx.store->value.as<MulNode>();
  return mul && (match_ordered_broadcast(mul->a, mul->b) ||
                 match_ordered_broadcast(mul->b, mul->a));
}

bool MatchRowBroadcastDivRule(const TileComputeRewriteContext& ctx,
                              TileComputeRewriteMatch* match,
                              TileComputeIRBuilder* builder) {
  const auto* div = ctx.store->value.as<DivNode>();
  Buffer scalar;
  if (!div ||
      !MatchLoadFromBuffer(div->a, ctx.store->buffer, ctx.linear_index) ||
      !MatchLoad(div->b, &scalar, nullptr) ||
      SameBufferStorage(scalar, ctx.store->buffer)) {
    return false;
  }
  Buffer reciprocal = builder->TempLike(scalar, "_recip");
  match->temp_buffers.push_back(reciprocal);
  const PrimExpr scalar_elements = BufferElementCount(scalar, IntImm32(1));
  match->AddUnary(blackhole_tile_compute_schema::kRecipTile, scalar,
                  reciprocal, scalar_elements);
  match->AddBroadcastColsBinary(
      blackhole_tile_compute_schema::kMulTilesBcastCols,
      ctx.store->buffer, reciprocal, ctx.num_elements, ctx.num_elements);
  return true;
}

bool MatchFillTileRule(const TileComputeRewriteContext& ctx,
                       TileComputeRewriteMatch* match,
                       TileComputeIRBuilder* builder) {
  (void)builder;
  if (!IsLiteralScalarValue(ctx.store->value)) {
    return false;
  }
  match->AddFill(ctx.store->buffer, ctx.store->value, ctx.num_elements);
  return true;
}

const std::vector<TileComputeRewriteRule>&
GetBlackholeTileComputeRewriteRules() {
  static const std::vector<TileComputeRewriteRule> rules = {
      {"typecast_tile", 100, &MatchTypecastTileRule},
      {"copy_tile", 90, &MatchCopyTileRule},
      {"binary_max_tile", 80, &MatchBinaryMaxTileRule},
      {"fma_leaf_sequence", 70, &MatchFmaTileRule},
      {"exp2_leaf_sequence", 60, &MatchExp2ScaledDifferenceRule},
      {"row_broadcast_mul_leaf", 50, &MatchRowBroadcastMulRule},
      {"row_broadcast_div_leaf_sequence", 40, &MatchRowBroadcastDivRule},
      {"fill_tile", 10, &MatchFillTileRule},
  };
  return rules;
}

Stmt ApplyBlackholeTileComputeRewriteRules(const BufferStoreNode* store,
                                           const PrimExpr& linear_index,
                                           const PrimExpr& num_elements,
                                           int* temp_index) {
  if (!store || !IsBlackholeComputeBuffer(store->buffer) ||
      store->indices.size() != 1U ||
      !ProvenEqual(store->indices[0], linear_index)) {
    return Stmt();
  }
  TileComputeIRBuilder builder(temp_index);
  const TileComputeRewriteContext ctx{store, linear_index, num_elements};
  int previous_benefit = 1000000000;
  for (const TileComputeRewriteRule& rule :
       GetBlackholeTileComputeRewriteRules()) {
    ICHECK(rule.name != nullptr);
    ICHECK_LE(rule.benefit, previous_benefit)
        << "Blackhole tile compute rewrite rules must be ordered by "
           "non-increasing local benefit";
    previous_benefit = rule.benefit;
    TileComputeRewriteMatch match;
    if (rule.match(ctx, &match, &builder)) {
      return builder.Render(match);
    }
  }
  return Stmt();
}

class BlackholeTileComputeNormalizer : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    if (!IsBlackholePrimFunc(f)) {
      return f;
    }
    PrimFuncNode* fptr = f.CopyOnWrite();
    BlackholeTileComputeNormalizer normalizer;
    fptr->body = normalizer.VisitStmt(f->body);
    return f;
  }

 private:
  int temp_index_{0};

  Stmt VisitStmt_(const ForNode* op) final {
    if (Stmt normalized = NormalizeBlackholeTileComputeLoop(op, &temp_index_);
        normalized.defined()) {
      return normalized;
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

}  // namespace

Stmt NormalizeBlackholeTileComputeLoop(const ForNode* op, int* temp_index) {
  if (const auto* store = op->body.as<BufferStoreNode>()) {
    if (store->indices.size() == 1U) {
      return ApplyBlackholeTileComputeRewriteRules(store, store->indices[0],
                                                   op->extent, temp_index);
    }
  }
  const auto* inner_loop = op->body.as<ForNode>();
  const auto* inner_store =
      inner_loop ? inner_loop->body.as<BufferStoreNode>() : nullptr;
  if (inner_loop && inner_store && inner_store->indices.size() == 1U) {
    arith::Analyzer analyzer;
    return ApplyBlackholeTileComputeRewriteRules(
        inner_store, inner_store->indices[0],
        analyzer.Simplify(op->extent * inner_loop->extent), temp_index);
  }
  return Stmt();
}

namespace transform {

using namespace tir::transform;

tvm::transform::Pass NormalizeBlackholeTileCompute() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return BlackholeTileComputeNormalizer::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeBlackholeTileCompute", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeBlackholeTileCompute",
                        NormalizeBlackholeTileCompute);
}

}  // namespace transform

}  // namespace tl
}  // namespace tvm
