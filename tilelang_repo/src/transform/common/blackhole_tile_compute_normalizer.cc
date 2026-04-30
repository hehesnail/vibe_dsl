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

Stmt MakeUnaryTileCall(const char* operation, const Buffer& src,
                       const Buffer& dst, const PrimExpr& num_elements) {
  return MakeBlackholeTileComputeCall(
      operation,
      {MakeFullRegionExpr(src, 1), MakeFullRegionExpr(dst, 2), num_elements});
}

Stmt MakeInplaceBinaryTileCall(const char* operation, const Buffer& dst,
                               const Buffer& rhs,
                               const PrimExpr& num_elements) {
  return MakeBlackholeTileComputeCall(
      operation,
      {MakeFullRegionExpr(dst, 3), MakeFullRegionExpr(rhs, 1), num_elements});
}

Stmt MakeCopyTileCall(const Buffer& src, const Buffer& dst,
                      const PrimExpr& num_elements) {
  return MakeUnaryTileCall(blackhole_tile_compute_schema::kCopyTile, src, dst,
                           num_elements);
}

Stmt MakeFillTileCall(const Buffer& dst, const PrimExpr& value,
                      const PrimExpr& num_elements) {
  return MakeBlackholeTileComputeCall(
      blackhole_tile_compute_schema::kFillTile,
      {MakeFullRegionExpr(dst, 2), value, num_elements});
}

Stmt MakeMulTilesCall(const Buffer& dst, const Buffer& rhs,
                      const PrimExpr& num_elements) {
  return MakeInplaceBinaryTileCall(blackhole_tile_compute_schema::kMulTiles,
                                   dst, rhs, num_elements);
}

Stmt MakeAddTilesCall(const Buffer& dst, const Buffer& rhs,
                      const PrimExpr& num_elements) {
  return MakeInplaceBinaryTileCall(blackhole_tile_compute_schema::kAddTiles,
                                   dst, rhs, num_elements);
}

Stmt MakeBroadcastColsBinaryCall(const char* operation, const Buffer& dst,
                                 const Buffer& rhs,
                                 const PrimExpr& num_elements,
                                 const PrimExpr& row_width) {
  return MakeBlackholeTileComputeCall(
      operation, {MakeFullRegionExpr(dst, 3), MakeFullRegionExpr(rhs, 1),
                  num_elements, row_width});
}

Stmt MakeExp2TileCall(const Buffer& src, const Buffer& dst,
                      const PrimExpr& num_elements) {
  return MakeUnaryTileCall(blackhole_tile_compute_schema::kExp2Tile, src, dst,
                           num_elements);
}

Stmt MakeRecipTileCall(const Buffer& src, const Buffer& dst,
                       const PrimExpr& num_elements) {
  return MakeUnaryTileCall(blackhole_tile_compute_schema::kRecipTile, src, dst,
                           num_elements);
}

Stmt MakeTypecastTileCall(const Buffer& src, const Buffer& dst,
                          const PrimExpr& num_elements) {
  return MakeUnaryTileCall(blackhole_tile_compute_schema::kTypecastTile, src,
                           dst, num_elements);
}

Stmt MakeBinaryMaxTileCall(const Buffer& dst, const Buffer& rhs,
                           const PrimExpr& num_elements) {
  return MakeInplaceBinaryTileCall(
      blackhole_tile_compute_schema::kBinaryMaxTile, dst, rhs, num_elements);
}

Stmt TryNormalizeBlackholeTileComputeStore(const BufferStoreNode* store,
                                           const PrimExpr& linear_index,
                                           const PrimExpr& num_elements,
                                           int* temp_index) {
  if (!store || !IsBlackholeComputeBuffer(store->buffer) ||
      store->indices.size() != 1U ||
      !ProvenEqual(store->indices[0], linear_index)) {
    return Stmt();
  }

  Buffer src;
  PrimExpr src_index;
  if (const auto* cast = store->value.as<CastNode>()) {
    if (MatchLoad(cast->value, &src, &src_index) &&
        ProvenEqual(src_index, linear_index)) {
      return MakeTypecastTileCall(src, store->buffer, num_elements);
    }
  }

  if (MatchLoad(store->value, &src, &src_index) &&
      !SameBufferStorage(src, store->buffer) &&
      ProvenEqual(src_index, linear_index)) {
    return MakeCopyTileCall(src, store->buffer, num_elements);
  }

  auto try_match_max = [&](const PrimExpr& lhs,
                           const PrimExpr& rhs) -> Stmt {
    Buffer rhs_buffer;
    if (!MatchLoadFromBuffer(lhs, store->buffer, linear_index) ||
        !MatchLoad(rhs, &rhs_buffer, nullptr)) {
      return Stmt();
    }
    return MakeBinaryMaxTileCall(store->buffer, rhs_buffer, num_elements);
  };
  if (const auto* max = store->value.as<MaxNode>()) {
    if (Stmt matched = try_match_max(max->a, max->b); matched.defined()) {
      return matched;
    }
    if (Stmt matched = try_match_max(max->b, max->a); matched.defined()) {
      return matched;
    }
  }

  auto try_match_fma = [&](const PrimExpr& mul_expr,
                           const PrimExpr& add_expr) -> Stmt {
    const auto* mul = mul_expr.as<MulNode>();
    if (!mul) {
      return Stmt();
    }
    auto emit_fma = [&](const Buffer& other_mul,
                        const Buffer& add_buffer) -> Stmt {
      std::vector<Stmt> stmts{
          MakeMulTilesCall(store->buffer, other_mul, num_elements),
          MakeAddTilesCall(store->buffer, add_buffer, num_elements)};
      return SeqStmt::Flatten(stmts);
    };
    Buffer other_mul;
    Buffer add_buffer;
    if (MatchLoadFromBuffer(mul->a, store->buffer, linear_index) &&
        MatchLoad(mul->b, &other_mul, nullptr) &&
        MatchLoad(add_expr, &add_buffer, nullptr)) {
      return emit_fma(other_mul, add_buffer);
    }
    if (MatchLoadFromBuffer(mul->b, store->buffer, linear_index) &&
        MatchLoad(mul->a, &other_mul, nullptr) &&
        MatchLoad(add_expr, &add_buffer, nullptr)) {
      return emit_fma(other_mul, add_buffer);
    }
    return Stmt();
  };
  if (const auto* add = store->value.as<AddNode>()) {
    if (Stmt matched = try_match_fma(add->a, add->b); matched.defined()) {
      return matched;
    }
    if (Stmt matched = try_match_fma(add->b, add->a); matched.defined()) {
      return matched;
    }
  }

  PrimExpr exp2_arg;
  if (MatchExp2Call(store->value, &exp2_arg)) {
    if (const auto* sub = exp2_arg.as<SubNode>()) {
      Buffer lhs;
      Buffer rhs;
      PrimExpr lhs_scale;
      PrimExpr rhs_scale;
      if (MatchScaledLoad(sub->a, &lhs, &lhs_scale) &&
          MatchScaledLoad(sub->b, &rhs, &rhs_scale)) {
        arith::Analyzer analyzer;
        const bool use_bcast_cols = SameBufferStorage(lhs, store->buffer);
        PrimExpr row_width =
            IsOneDimensionalBuffer(store->buffer) ? num_elements : IntImm32(1);
        std::vector<Buffer> temps;
        std::vector<Stmt> stmts;
        Buffer lhs_scale_buffer =
            MakeBlackholeTempBufferLike(store->buffer, "_lhs_scale", temp_index);
        temps.push_back(lhs_scale_buffer);
        if (!SameBufferStorage(lhs, store->buffer)) {
          stmts.push_back(MakeCopyTileCall(lhs, store->buffer, num_elements));
        }
        stmts.push_back(MakeFillTileCall(lhs_scale_buffer, lhs_scale,
                                         BufferElementCount(lhs_scale_buffer,
                                                            num_elements)));
        stmts.push_back(MakeMulTilesCall(store->buffer, lhs_scale_buffer,
                                         num_elements));

        Buffer scaled_rhs =
            MakeBlackholeTempBufferLike(rhs, "_scaled_rhs", temp_index);
        Buffer rhs_scale_buffer =
            MakeBlackholeTempBufferLike(rhs, "_rhs_scale", temp_index);
        temps.push_back(scaled_rhs);
        temps.push_back(rhs_scale_buffer);
        const PrimExpr rhs_elements = BufferElementCount(rhs, num_elements);
        stmts.push_back(MakeCopyTileCall(rhs, scaled_rhs, rhs_elements));
        stmts.push_back(MakeFillTileCall(rhs_scale_buffer,
                                         analyzer.Simplify(-rhs_scale),
                                         BufferElementCount(rhs_scale_buffer,
                                                            rhs_elements)));
        stmts.push_back(MakeMulTilesCall(scaled_rhs, rhs_scale_buffer,
                                         rhs_elements));
        if (use_bcast_cols) {
          stmts.push_back(MakeBroadcastColsBinaryCall(
              blackhole_tile_compute_schema::kAddTilesBcastCols,
              store->buffer, scaled_rhs, num_elements, row_width));
        } else {
          stmts.push_back(MakeAddTilesCall(store->buffer, scaled_rhs,
                                           num_elements));
        }
        stmts.push_back(
            MakeExp2TileCall(store->buffer, store->buffer, num_elements));
        return WrapBlackholeTempBuffers(SeqStmt::Flatten(stmts), temps);
      }
    }
  }

  auto try_match_row_broadcast = [&](const PrimExpr& self_expr,
                                     const PrimExpr& scalar_expr,
                                     const char* operation) -> Stmt {
    Buffer scalar;
    if (!MatchLoadFromBuffer(self_expr, store->buffer, linear_index) ||
        !MatchLoad(scalar_expr, &scalar, nullptr) ||
        SameBufferStorage(scalar, store->buffer)) {
      return Stmt();
    }
    return MakeBroadcastColsBinaryCall(operation, store->buffer, scalar,
                                       num_elements, num_elements);
  };
  if (const auto* mul = store->value.as<MulNode>()) {
    if (Stmt matched = try_match_row_broadcast(
            mul->a, mul->b, blackhole_tile_compute_schema::kMulTilesBcastCols);
        matched.defined()) {
      return matched;
    }
    if (Stmt matched = try_match_row_broadcast(
            mul->b, mul->a, blackhole_tile_compute_schema::kMulTilesBcastCols);
        matched.defined()) {
      return matched;
    }
  }
  if (const auto* div = store->value.as<DivNode>()) {
    Buffer scalar;
    if (MatchLoadFromBuffer(div->a, store->buffer, linear_index) &&
        MatchLoad(div->b, &scalar, nullptr) &&
        !SameBufferStorage(scalar, store->buffer)) {
      std::vector<Buffer> temps;
      Buffer reciprocal =
          MakeBlackholeTempBufferLike(scalar, "_recip", temp_index);
      temps.push_back(reciprocal);
      const PrimExpr scalar_elements = BufferElementCount(scalar, IntImm32(1));
      std::vector<Stmt> stmts{
          MakeRecipTileCall(scalar, reciprocal, scalar_elements),
          MakeBroadcastColsBinaryCall(
              blackhole_tile_compute_schema::kMulTilesBcastCols,
              store->buffer, reciprocal, num_elements, num_elements)};
      return WrapBlackholeTempBuffers(SeqStmt::Flatten(stmts), temps);
    }
  }

  if (IsLiteralScalarValue(store->value)) {
    return MakeFillTileCall(store->buffer, store->value, num_elements);
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
      return TryNormalizeBlackholeTileComputeStore(store, store->indices[0],
                                                  op->extent, temp_index);
    }
  }
  const auto* inner_loop = op->body.as<ForNode>();
  const auto* inner_store =
      inner_loop ? inner_loop->body.as<BufferStoreNode>() : nullptr;
  if (inner_loop && inner_store && inner_store->indices.size() == 1U) {
    arith::Analyzer analyzer;
    return TryNormalizeBlackholeTileComputeStore(
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
