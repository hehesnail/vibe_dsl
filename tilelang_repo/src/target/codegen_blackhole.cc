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
 * \file codegen_blackhole.cc
 * \brief Generate TT-Metal code for Blackhole backend.
 */

#include "codegen_blackhole.h"

#include <tvm/arith/analyzer.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../layout/layout.h"
#include "../transform/common/blackhole_ir_attrs.h"
#include "../transform/common/blackhole_runtime_arg_schema.h"
#include "../tir/builtin_blackhole.h"
#include "tt_program_projection.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

namespace tvm {
namespace tl {

namespace {

bool IsBufferAddressRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr32" || kind == "input_buffer_addr" ||
         kind == "output_buffer_addr32" || kind == "output_buffer_addr";
}

std::optional<int64_t> TryEvalStaticInt(PrimExpr expr) {
  arith::Analyzer analyzer;
  expr = analyzer.Simplify(expr);
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

bool IsRank1RowVectorLogicalTileLayout(const tvm::ffi::Array<PrimExpr>& logical_shape,
                                       const tvm::ffi::Array<PrimExpr>& local_shape) {
  if (logical_shape.size() < 2U || local_shape.size() != 1U) {
    return false;
  }
  std::optional<int64_t> rows = TryEvalStaticInt(logical_shape[logical_shape.size() - 2]);
  std::optional<int64_t> cols = TryEvalStaticInt(logical_shape[logical_shape.size() - 1]);
  std::optional<int64_t> local_extent = TryEvalStaticInt(local_shape[0]);
  return rows.has_value() && cols.has_value() && local_extent.has_value() &&
         rows.value() > 0 && rows.value() <= 32 && cols.value() == 32 &&
         local_extent.value() == rows.value();
}

std::optional<bool> TryEvalStaticBool(PrimExpr expr) {
  arith::Analyzer analyzer;
  expr = analyzer.Simplify(expr);
  if (tir::is_zero(expr)) {
    return false;
  }
  if (tir::is_one(expr)) {
    return true;
  }
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value != 0;
  }
  if (const auto* not_op = expr.as<tir::NotNode>()) {
    if (std::optional<bool> value = TryEvalStaticBool(not_op->a)) {
      return !value.value();
    }
    return std::nullopt;
  }
  if (const auto* and_op = expr.as<tir::AndNode>()) {
    std::optional<bool> lhs = TryEvalStaticBool(and_op->a);
    if (lhs && !lhs.value()) {
      return false;
    }
    std::optional<bool> rhs = TryEvalStaticBool(and_op->b);
    if (rhs && !rhs.value()) {
      return false;
    }
    if (lhs && rhs) {
      return lhs.value() && rhs.value();
    }
    return std::nullopt;
  }
  if (const auto* or_op = expr.as<tir::OrNode>()) {
    std::optional<bool> lhs = TryEvalStaticBool(or_op->a);
    if (lhs && lhs.value()) {
      return true;
    }
    std::optional<bool> rhs = TryEvalStaticBool(or_op->b);
    if (rhs && rhs.value()) {
      return true;
    }
    if (lhs && rhs) {
      return lhs.value() || rhs.value();
    }
    return std::nullopt;
  }
  auto compare_ints = [](const PrimExpr& lhs_expr, const PrimExpr& rhs_expr,
                         auto&& predicate) -> std::optional<bool> {
    std::optional<int64_t> lhs = TryEvalStaticInt(lhs_expr);
    std::optional<int64_t> rhs = TryEvalStaticInt(rhs_expr);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return predicate(lhs.value(), rhs.value());
  };
  if (const auto* eq = expr.as<tir::EQNode>()) {
    return compare_ints(eq->a, eq->b, [](int64_t lhs, int64_t rhs) { return lhs == rhs; });
  }
  if (const auto* ne = expr.as<tir::NENode>()) {
    return compare_ints(ne->a, ne->b, [](int64_t lhs, int64_t rhs) { return lhs != rhs; });
  }
  if (const auto* lt = expr.as<tir::LTNode>()) {
    return compare_ints(lt->a, lt->b, [](int64_t lhs, int64_t rhs) { return lhs < rhs; });
  }
  if (const auto* le = expr.as<tir::LENode>()) {
    return compare_ints(le->a, le->b, [](int64_t lhs, int64_t rhs) { return lhs <= rhs; });
  }
  if (const auto* gt = expr.as<tir::GTNode>()) {
    return compare_ints(gt->a, gt->b, [](int64_t lhs, int64_t rhs) { return lhs > rhs; });
  }
  if (const auto* ge = expr.as<tir::GENode>()) {
    return compare_ints(ge->a, ge->b, [](int64_t lhs, int64_t rhs) { return lhs >= rhs; });
  }
  return std::nullopt;
}

std::string RequireStringImm(const tvm::PrimExpr& expr, const char* op_name,
                             const char* arg_name) {
  const auto* value = expr.as<tvm::tir::StringImmNode>();
  ICHECK(value) << op_name << " expects " << arg_name << " to be a string literal";
  return value->value;
}

const char* ReduceKindToTTMetal(const std::string& reduce_kind, const char* op_name) {
  if (reduce_kind == "sum") {
    return "PoolType::SUM";
  }
  if (reduce_kind == "max") {
    return "PoolType::MAX";
  }
  ICHECK(false) << op_name << " got unsupported reduce kind " << reduce_kind;
  return "";
}

const char* ReduceDimToTTMetal(const std::string& reduce_dim, const char* op_name) {
  if (reduce_dim == "row") {
    return "ReduceDim::REDUCE_ROW";
  }
  if (reduce_dim == "col") {
    return "ReduceDim::REDUCE_COL";
  }
  ICHECK(false) << op_name << " got unsupported reduce dim " << reduce_dim;
  return "";
}

const tvm::tir::VarNode* AsHandleVar(const tvm::PrimExpr& expr) {
  if (const auto* var = expr.as<tvm::tir::VarNode>()) {
    return var;
  }
  return nullptr;
}

std::optional<std::string> BlackholeBuiltinName(const tvm::tir::CallNode* op) {
  if (op == nullptr || !op->op->IsInstance<tvm::OpNode>()) {
    return std::nullopt;
  }
  tvm::Op call_op = Downcast<tvm::Op>(op->op);
  const std::string op_name = call_op->name;
  const std::string prefix = "tl.blackhole.";
  if (op_name.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  return op_name.substr(prefix.length());
}

std::optional<int> CBRequirementIndexAnnotation(const tvm::tir::AllocateNode* op) {
  if (op == nullptr || !op->annotations.defined()) {
    return std::nullopt;
  }
  if (auto value =
          op->annotations.Get(tvm::ffi::String(blackhole_ir_attrs::kCBRequirementIndex))) {
    return Downcast<tvm::Integer>(value.value()).IntValue();
  }
  return std::nullopt;
}

bool IsTRISCOnlyBlackholeBuiltin(const std::string& builtin_name) {
  static const std::unordered_set<std::string> kTRISCOnlyBuiltins = {
      "mm_init",
      "reconfig_data_format",
      "mm_init_short",
      "mm_init_short_with_dt",
      "matmul_tiles",
      "tile_regs_acquire",
      "tile_regs_commit",
      "tile_regs_wait",
      "tile_regs_release",
      "pack_tile",
      "pack_reconfig_data_format",
      "copy_tile_to_dst_init_short",
      "copy_tile_to_dst_init_short_with_dt",
      "copy_tile",
      "binary_op_init_common",
      "unary_op_init_common",
      "add_tiles_init",
      "add_tiles",
      "sub_tiles_init",
      "sub_tiles",
      "add_bcast_rows_init_short",
      "add_bcast_cols_init_short",
      "add_tiles_bcast_rows",
      "add_tiles_bcast_cols",
      "mul_tiles_init",
      "mul_tiles",
      "mul_bcast_rows_init_short",
      "mul_bcast_cols_init_short",
      "mul_tiles_bcast_rows",
      "mul_tiles_bcast_cols",
      "reduce_init",
      "reduce_tile",
      "reduce_uninit",
      "binary_max_tile_init",
      "binary_max_tile",
      "div_binary_tile_init",
      "div_binary_tile",
      "exp_tile_init",
      "exp_tile",
      "exp2_tile_init",
      "exp2_tile",
      "recip_tile_init",
      "recip_tile",
      "fill_fragment",
      "add_fragment",
      "add_fragment_from_cb_front",
      "pack_untilize_slice",
      "pack_untilize_tile",
      "tilize_local_fragment_slice",
      "tilize_cast_fragment_slice",
      "pack_fill_fragment_to_tiled_cb",
      "generate_reduce_scaler_to_cb",
      "untilize_cb_front_tile",
      "untilize_cb_front_tile_fragment",
      "cast_fragment_slice",
  };
  return kTRISCOnlyBuiltins.count(builtin_name) != 0;
}

bool IsDataMovementOnlyBlackholeBuiltin(const std::string& builtin_name) {
  static const std::unordered_set<std::string> kDataMovementOnlyBuiltins = {
      "noc_async_read",
      "noc_async_write",
      "noc_async_read_barrier",
      "noc_async_write_barrier",
      "read_tile_to_cb",
      "read_page_to_cb",
      "read_bcast_cols_to_cb",
      "copy_cb_page",
      "write_tile_from_cb",
      "write_page_from_cb",
      "zero_cb_page",
      "guard_mask_to_cb",
  };
  return kDataMovementOnlyBuiltins.count(builtin_name) != 0;
}

bool EmitsBlackholeBuiltinForCore(const std::string& builtin_name,
                                  CodeGenBlackhole::CoreType core_type) {
  if (core_type != CodeGenBlackhole::CoreType::kTRISC &&
      IsTRISCOnlyBlackholeBuiltin(builtin_name)) {
    return false;
  }
  if (core_type == CodeGenBlackhole::CoreType::kTRISC &&
      IsDataMovementOnlyBlackholeBuiltin(builtin_name)) {
    return false;
  }
  return true;
}

bool SameCodegenStorageVar(const tvm::tir::VarNode* lhs,
                           const tvm::tir::VarNode* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs == nullptr || rhs == nullptr || lhs->name_hint != rhs->name_hint ||
      lhs->dtype != rhs->dtype) {
    return false;
  }
  const bool lhs_is_ptr = lhs->type_annotation.as<tvm::PointerTypeNode>() != nullptr;
  const bool rhs_is_ptr = rhs->type_annotation.as<tvm::PointerTypeNode>() != nullptr;
  if (lhs_is_ptr != rhs_is_ptr) {
    return false;
  }
  if (!lhs_is_ptr) {
    return true;
  }
  return tvm::tir::GetPtrStorageScope(GetRef<tvm::tir::Var>(lhs)) ==
         tvm::tir::GetPtrStorageScope(GetRef<tvm::tir::Var>(rhs));
}

bool StmtUsesVar(const tvm::tir::Stmt& stmt, const tvm::tir::VarNode* target) {
  class Visitor final : public tvm::tir::StmtExprVisitor {
   public:
    explicit Visitor(const tvm::tir::VarNode* target) : target_(target) {}

    bool Check(const tvm::tir::Stmt& stmt) {
      VisitStmt(stmt);
      return found_;
    }

   private:
    void VisitStmt(const tvm::tir::Stmt& stmt) final {
      if (found_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitStmt(stmt);
    }

    void VisitExpr(const tvm::PrimExpr& expr) final {
      if (found_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitExpr(expr);
    }

    void VisitExpr_(const tvm::tir::VarNode* op) final {
      if (SameCodegenStorageVar(op, target_)) {
        found_ = true;
      }
    }

    const tvm::tir::VarNode* target_;
    bool found_{false};
  };

  return target != nullptr && stmt.defined() && Visitor(target).Check(stmt);
}

bool ExprUsesVar(const tvm::PrimExpr& expr, const tvm::tir::VarNode* target) {
  bool found = false;
  tvm::tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (found) {
      return;
    }
    if (const auto* var = node.as<tvm::tir::VarNode>()) {
      found = SameCodegenStorageVar(var, target);
    }
  });
  return found;
}

bool IsNoOpStmt(const tvm::tir::Stmt& stmt) {
  if (!stmt.defined()) {
    return true;
  }
  if (const auto* seq = stmt.as<tvm::tir::SeqStmtNode>()) {
    return std::all_of(seq->seq.begin(), seq->seq.end(), [](const tvm::tir::Stmt& child) {
      return IsNoOpStmt(child);
    });
  }
  if (const auto* eval = stmt.as<tvm::tir::EvaluateNode>()) {
    return tir::is_zero(eval->value);
  }
  if (const auto* if_then_else = stmt.as<tvm::tir::IfThenElseNode>()) {
    return IsNoOpStmt(if_then_else->then_case) &&
           (!if_then_else->else_case.defined() ||
            IsNoOpStmt(if_then_else->else_case.value()));
  }
  if (const auto* let = stmt.as<tvm::tir::LetStmtNode>()) {
    return IsNoOpStmt(let->body);
  }
  if (const auto* attr = stmt.as<tvm::tir::AttrStmtNode>()) {
    return IsNoOpStmt(attr->body);
  }
  if (const auto* decl = stmt.as<tvm::tir::DeclBufferNode>()) {
    return IsNoOpStmt(decl->body);
  }
  if (const auto* alloc = stmt.as<tvm::tir::AllocateNode>()) {
    return IsNoOpStmt(alloc->body);
  }
  return false;
}

bool IsCBPopFrontOnlyStmt(const tvm::tir::Stmt& stmt) {
  if (const auto* eval = stmt.as<tvm::tir::EvaluateNode>()) {
    const auto* call = eval->value.as<tvm::tir::CallNode>();
    auto builtin_name = BlackholeBuiltinName(call);
    return builtin_name.has_value() && builtin_name.value() == "cb_pop_front";
  }
  if (const auto* seq = stmt.as<tvm::tir::SeqStmtNode>()) {
    return std::all_of(seq->seq.begin(), seq->seq.end(), [](const tvm::tir::Stmt& child) {
      return IsCBPopFrontOnlyStmt(child);
    });
  }
  return false;
}

bool BlackholeBuiltinNeedsThreadIndexForEmission(const std::string& builtin_name) {
  static const std::unordered_set<std::string> kThreadIndexedBridgeBuiltins = {
      "tilize_local_fragment_slice",
      "tilize_cast_fragment_slice",
      "untilize_cb_front_tile_fragment",
  };
  return kThreadIndexedBridgeBuiltins.count(builtin_name) != 0;
}

bool IsThreadSurvivorPopGuard(const tvm::tir::IfThenElseNode* op,
                              const tvm::tir::VarNode* thread_var,
                              const tvm::PrimExpr& thread_extent) {
  (void)thread_extent;
  return op != nullptr && !op->else_case.defined() && IsCBPopFrontOnlyStmt(op->then_case) &&
         ExprUsesVar(op->condition, thread_var);
}

bool ThreadUsesOnlySurvivorPopGuards(const tvm::tir::Stmt& stmt,
                                     const tvm::tir::VarNode* thread_var,
                                     const tvm::PrimExpr& thread_extent) {
  class Visitor final : public tvm::tir::StmtExprVisitor {
   public:
    Visitor(const tvm::tir::VarNode* thread_var, tvm::PrimExpr thread_extent)
        : thread_var_(thread_var), thread_extent_(std::move(thread_extent)) {}

    bool Check(const tvm::tir::Stmt& stmt) {
      VisitStmt(stmt);
      return !has_disallowed_use_;
    }

   private:
    void VisitStmt(const tvm::tir::Stmt& stmt) final {
      if (has_disallowed_use_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitStmt(stmt);
    }

    void VisitExpr(const tvm::PrimExpr& expr) final {
      if (has_disallowed_use_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitExpr(expr);
    }

    void VisitStmt_(const tvm::tir::IfThenElseNode* op) final {
      if (IsNoOpStmt(op->then_case) &&
          (!op->else_case.defined() || IsNoOpStmt(op->else_case.value()))) {
        return;
      }
      if (IsThreadSurvivorPopGuard(op, thread_var_, thread_extent_)) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const tvm::tir::CallNode* op) final {
      auto builtin_name = BlackholeBuiltinName(op);
      if (builtin_name.has_value() &&
          BlackholeBuiltinNeedsThreadIndexForEmission(builtin_name.value())) {
        has_disallowed_use_ = true;
        return;
      }
      tvm::tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const tvm::tir::VarNode* op) final {
      if (SameCodegenStorageVar(op, thread_var_)) {
        has_disallowed_use_ = true;
      }
    }

    const tvm::tir::VarNode* thread_var_;
    tvm::PrimExpr thread_extent_;
    bool has_disallowed_use_{false};
  };

  return thread_var != nullptr && stmt.defined() &&
         Visitor(thread_var, thread_extent).Check(stmt);
}

tvm::tir::Stmt UnwrapThreadSurvivorPopGuards(const tvm::tir::Stmt& stmt,
                                             const tvm::tir::VarNode* thread_var,
                                             const tvm::PrimExpr& thread_extent) {
  class Rewriter final : public tvm::tir::StmtExprMutator {
   public:
    Rewriter(const tvm::tir::VarNode* thread_var, tvm::PrimExpr thread_extent)
        : thread_var_(thread_var), thread_extent_(std::move(thread_extent)) {}

    tvm::tir::Stmt Rewrite(const tvm::tir::Stmt& stmt) { return VisitStmt(stmt); }

   private:
    tvm::tir::Stmt VisitStmt_(const tvm::tir::IfThenElseNode* op) final {
      if (IsThreadSurvivorPopGuard(op, thread_var_, thread_extent_)) {
        return VisitStmt(op->then_case);
      }
      return tvm::tir::StmtExprMutator::VisitStmt_(op);
    }

    const tvm::tir::VarNode* thread_var_;
    tvm::PrimExpr thread_extent_;
  };

  return Rewriter(thread_var, thread_extent).Rewrite(stmt);
}

std::optional<const tvm::tir::VarNode*> FragmentFillDataVar(const tvm::tir::CallNode* call) {
  auto builtin_name = BlackholeBuiltinName(call);
  if (!builtin_name.has_value() || builtin_name.value() != "fill_fragment" ||
      call->args.empty()) {
    return std::nullopt;
  }
  return AsHandleVar(call->args[0]);
}

std::unordered_set<const tvm::tir::VarNode*> CollectDeadFragmentFillDataVars(
    const tvm::tir::Stmt& stmt) {
  class Visitor final : public tvm::tir::StmtExprVisitor {
   public:
    std::unordered_set<const tvm::tir::VarNode*> Collect(const tvm::tir::Stmt& stmt) {
      VisitStmt(stmt);
      std::unordered_set<const tvm::tir::VarNode*> dead;
      for (const tvm::tir::VarNode* var : fill_data_vars_) {
        if (live_data_vars_.count(var) == 0U) {
          dead.insert(var);
        }
      }
      return dead;
    }

   private:
    void MarkLive(const tvm::tir::VarNode* var) {
      if (var != nullptr && fill_data_vars_.count(var) != 0U) {
        live_data_vars_.insert(var);
      }
    }

    void VisitStmt_(const tvm::tir::AllocateNode* op) final {
      VisitExpr(op->condition);
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::AttrStmtNode* op) final {
      VisitExpr(op->value);
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::DeclBufferNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tvm::tir::BufferStoreNode* op) final {
      VisitExpr(op->value);
      for (const tvm::PrimExpr& index : op->indices) {
        VisitExpr(index);
      }
    }

    void VisitExpr_(const tvm::tir::BufferLoadNode* op) final {
      if (op->buffer.defined()) {
        MarkLive(op->buffer->data.get());
      }
      tvm::tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const tvm::tir::CallNode* op) final {
      auto builtin_name = BlackholeBuiltinName(op);
      if (builtin_name.has_value()) {
        if (builtin_name.value() == "fill_fragment") {
          if (std::optional<const tvm::tir::VarNode*> data = FragmentFillDataVar(op)) {
            fill_data_vars_.insert(data.value());
          }
          for (size_t i = 1; i < op->args.size(); ++i) {
            VisitExpr(op->args[i]);
          }
          return;
        }
        if (builtin_name.value() == "pack_fill_fragment_to_tiled_cb") {
          for (size_t i = 1; i < op->args.size(); ++i) {
            VisitExpr(op->args[i]);
          }
          return;
        }
      }
      tvm::tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const tvm::tir::VarNode* op) final { MarkLive(op); }

    std::unordered_set<const tvm::tir::VarNode*> fill_data_vars_;
    std::unordered_set<const tvm::tir::VarNode*> live_data_vars_;
  };

  return stmt.defined() ? Visitor().Collect(stmt)
                        : std::unordered_set<const tvm::tir::VarNode*>();
}

bool StmtUsesVarInEmittedBody(const tvm::tir::Stmt& stmt,
                              const tvm::tir::VarNode* target,
                              CodeGenBlackhole::CoreType core_type) {
  class Visitor final : public tvm::tir::StmtExprVisitor {
   public:
    Visitor(const tvm::tir::VarNode* target, CodeGenBlackhole::CoreType core_type)
        : target_(target), core_type_(core_type) {}

    bool Check(const tvm::tir::Stmt& stmt) {
      VisitStmt(stmt);
      return found_;
    }

   private:
    void VisitStmt(const tvm::tir::Stmt& stmt) final {
      if (found_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitStmt(stmt);
    }

    void VisitExpr(const tvm::PrimExpr& expr) final {
      if (found_) {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitExpr(expr);
    }

    void VisitStmt_(const tvm::tir::AllocateNode* op) final {
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::DeclBufferNode* op) final {
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::ForNode* op) final {
      VisitExpr(op->min);
      VisitExpr(op->extent);
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::LetStmtNode* op) final {
      VisitExpr(op->value);
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::AttrStmtNode* op) final {
      VisitExpr(op->value);
      VisitStmt(op->body);
    }

    void VisitStmt_(const tvm::tir::BufferStoreNode* op) final {
      if (core_type_ != CodeGenBlackhole::CoreType::kTRISC && op->buffer.defined() &&
          std::string(op->buffer.scope()) == "blackhole.acc") {
        return;
      }
      tvm::tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const tvm::tir::EvaluateNode* op) final {
      if (const auto* call = op->value.as<tvm::tir::CallNode>()) {
        auto builtin_name = BlackholeBuiltinName(call);
        if (builtin_name.has_value() &&
            !EmitsBlackholeBuiltinForCore(builtin_name.value(), core_type_)) {
          return;
        }
      }
      VisitExpr(op->value);
    }

    void VisitExpr_(const tvm::tir::CallNode* op) final {
      auto builtin_name = BlackholeBuiltinName(op);
      if (builtin_name.has_value() &&
          !EmitsBlackholeBuiltinForCore(builtin_name.value(), core_type_)) {
        return;
      }
      if (builtin_name.has_value() &&
          BlackholeBuiltinNeedsThreadIndexForEmission(builtin_name.value())) {
        found_ = true;
        return;
      }
      tvm::tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const tvm::tir::LetNode* op) final {
      VisitExpr(op->value);
      VisitExpr(op->body);
    }

    void VisitExpr_(const tvm::tir::VarNode* op) final {
      if (SameCodegenStorageVar(op, target_)) {
        found_ = true;
      }
    }

    const tvm::tir::VarNode* target_;
    CodeGenBlackhole::CoreType core_type_;
    bool found_{false};
  };

  return target != nullptr && stmt.defined() && Visitor(target, core_type).Check(stmt);
}

std::vector<tvm::tir::Stmt> FlattenTopLevelSeq(const tvm::tir::Stmt& stmt) {
  if (const auto* seq = stmt.as<tvm::tir::SeqStmtNode>()) {
    return std::vector<tvm::tir::Stmt>(seq->seq.begin(), seq->seq.end());
  }
  return {stmt};
}

bool ExtractThreadScopedCBStaging(const tvm::tir::Stmt& stmt,
                                  const tvm::tir::VarNode* thread_var,
                                  CodeGenBlackhole::CoreType core_type,
                                  tvm::tir::Stmt* once_prefix,
                                  tvm::tir::Stmt* threaded_body,
                                  tvm::tir::Stmt* once_suffix) {
  auto is_blackhole_builtin = [](const tvm::tir::CallNode* call, const tvm::Op& builtin,
                                 const char* op_name) {
    if (!call) {
      return false;
    }
    if (call->op.same_as(builtin)) {
      return true;
    }
    if (const auto* op = call->op.as<tvm::OpNode>()) {
      return op->name == op_name;
    }
    return false;
  };

  tvm::tir::Stmt current = stmt;
  while (true) {
    if (const auto* attr = current.as<tvm::tir::AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto* let = current.as<tvm::tir::LetStmtNode>()) {
      current = let->body;
      continue;
    }
    if (const auto* decl = current.as<tvm::tir::DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<tvm::tir::AllocateNode>()) {
      current = alloc->body;
      continue;
    }
    break;
  }

  const auto* seq = current.as<tvm::tir::SeqStmtNode>();
  if (!seq || seq->seq.size() < 3) {
    return false;
  }
  const auto* reserve_eval = seq->seq.front().as<tvm::tir::EvaluateNode>();
  const auto* push_eval = seq->seq.back().as<tvm::tir::EvaluateNode>();
  if (!reserve_eval || !push_eval) {
    return false;
  }
  const auto* reserve_call = reserve_eval->value.as<tvm::tir::CallNode>();
  const auto* push_call = push_eval->value.as<tvm::tir::CallNode>();
  if (!is_blackhole_builtin(reserve_call, tir::builtin::blackhole_cb_reserve_back(),
                            "tl.blackhole.cb_reserve_back") ||
      !is_blackhole_builtin(push_call, tir::builtin::blackhole_cb_push_back(),
                            "tl.blackhole.cb_push_back")) {
    return false;
  }

  std::vector<tvm::tir::Stmt> middle(seq->seq.begin() + 1, seq->seq.end() - 1);
  if (middle.empty()) {
    return false;
  }
  tvm::tir::Stmt middle_stmt =
      middle.size() == 1 ? middle.front() : tvm::tir::SeqStmt::Flatten(middle);
  const bool uses_thread_var = StmtUsesVarInEmittedBody(middle_stmt, thread_var, core_type);
  if (!uses_thread_var) {
    return false;
  }

  *once_prefix = seq->seq.front();
  *threaded_body = middle_stmt;
  *once_suffix = seq->seq.back();
  return true;
}

struct ThreadEmissionPiece {
  tvm::tir::Stmt stmt;
  bool uses_thread_var{false};
};

std::vector<ThreadEmissionPiece> BuildThreadEmissionPieces(const tvm::tir::Stmt& stmt,
                                                           const tvm::tir::VarNode* thread_var,
                                                           CodeGenBlackhole::CoreType core_type) {
  auto add_piece = [](std::vector<ThreadEmissionPiece>* pieces, const tvm::tir::Stmt& piece,
                      bool uses_thread_var) {
    if (!piece.defined()) {
      return;
    }
    if (IsNoOpStmt(piece)) {
      return;
    }
    if (const auto* seq = piece.as<tvm::tir::SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return;
      }
    }
    pieces->push_back(ThreadEmissionPiece{piece, uses_thread_var});
  };

  std::vector<ThreadEmissionPiece> pieces;
  for (const auto& top_level_stmt : FlattenTopLevelSeq(stmt)) {
    tvm::tir::Stmt once_prefix;
    tvm::tir::Stmt threaded_body;
    tvm::tir::Stmt once_suffix;
    if (ExtractThreadScopedCBStaging(top_level_stmt, thread_var, core_type, &once_prefix,
                                     &threaded_body, &once_suffix)) {
      add_piece(&pieces, once_prefix, /*uses_thread_var=*/false);
      add_piece(&pieces, threaded_body, /*uses_thread_var=*/true);
      add_piece(&pieces, once_suffix, /*uses_thread_var=*/false);
      continue;
    }

    const bool uses_thread_var = StmtUsesVarInEmittedBody(top_level_stmt, thread_var, core_type);
    add_piece(&pieces, top_level_stmt, uses_thread_var);
  }
  return pieces;
}

ffi::Array<ffi::Any> AggregateSegmentRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  ffi::Array<ffi::Any> aggregated;
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (segment_plan.empty()) {
    return aggregated;
  }

  std::unordered_set<std::string> seen_runtime_args;
  for (const auto& item : segment_plan) {
    auto segment = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (segment.empty()) {
      continue;
    }
    auto runtime_args_it = segment.Get("runtime_args");
    if (!runtime_args_it.has_value()) {
      continue;
    }
    for (const auto& arg_item : Downcast<tvm::ffi::Array<tvm::ffi::Any>>(runtime_args_it.value())) {
      auto arg = arg_item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      std::string identity;
      std::string kind;
      if (auto v = arg.Get("identity")) {
        identity = Downcast<tvm::ffi::String>(v.value());
      }
      if (auto v = arg.Get("kind")) {
        kind = Downcast<tvm::ffi::String>(v.value());
      }
      std::string dedupe_key =
          !identity.empty() && !kind.empty() ? identity + ":" + kind : identity;
      if (!dedupe_key.empty() && !seen_runtime_args.insert(dedupe_key).second) {
        continue;
      }
      aggregated.push_back(arg);
    }
  }
  return aggregated;
}

ffi::Array<ffi::Any> GetRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  return AggregateSegmentRuntimeArgsForCodegen(f);
}

ffi::Array<ffi::Any> GetPerWorkArgSpecsForCodegen(const tvm::tir::PrimFunc& f) {
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (segment_plan.size() != 1) {
    return ffi::Array<ffi::Any>();
  }
  auto segment = segment_plan[0].as<ffi::Map<ffi::String, ffi::Any>>().value_or(
      ffi::Map<ffi::String, ffi::Any>());
  if (auto v = segment.Get(
          ffi::String(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs))) {
    return Downcast<ffi::Array<ffi::Any>>(v.value());
  }
  return ffi::Array<ffi::Any>();
}

ffi::Array<ffi::Any> GetCBConfigsForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCBConfigsFromExecutable(f, "Blackhole codegen");
}

std::string MapGetString(const ffi::Map<ffi::String, ffi::Any>& map,
                         const char* key) {
  if (auto value = map.Get(ffi::String(key))) {
    return Downcast<ffi::String>(value.value());
  }
  return "";
}

int64_t MapGetInt(const ffi::Map<ffi::String, ffi::Any>& map,
                  const char* key, int64_t default_value = 0) {
  if (auto value = map.Get(ffi::String(key))) {
    return Downcast<Integer>(value.value()).IntValue();
  }
  return default_value;
}

ffi::Array<ffi::Any> MapGetArray(const ffi::Map<ffi::String, ffi::Any>& map,
                                 const char* key) {
  if (auto value = map.Get(ffi::String(key))) {
    return Downcast<ffi::Array<ffi::Any>>(value.value());
  }
  return ffi::Array<ffi::Any>();
}

ffi::Map<ffi::String, ffi::Any> GetCorePlanForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCorePlanFromExecutable(f, "Blackhole codegen");
}

std::string GetCoreTypeForCodegen(const tvm::tir::PrimFunc& f) {
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (!segment_plan.empty()) {
    auto segment = segment_plan[0].as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (auto value = segment.Get("core_type")) {
      return Downcast<ffi::String>(value.value());
    }
  }
  return "";
}

bool HasRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  return !GetRuntimeArgsForCodegen(f).empty();
}

}  // namespace

CodeGenBlackhole::CodeGenBlackhole()
    : headers_emitted_(false),
      core_type_(CoreType::kBRISC),
      need_dataflow_api_h_(false),
      need_compute_api_h_(false) {}

void CodeGenBlackhole::Init(bool output_ssa, bool emit_asserts,
                            bool emit_fwd_func_decl, std::string target_str,
                            const std::unordered_set<std::string> &devices) {
  CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl,
                     target_str, devices);

  // Reset state for new CodeGen instance
  headers_emitted_ = false;
  core_type_ = CoreType::kBRISC;
  need_dataflow_api_h_ = false;
  need_compute_api_h_ = false;
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_identity_.clear();
  runtime_arg_vars_by_name_.clear();
  per_work_arg_bindings_by_identity_.clear();
  per_work_arg_bindings_.clear();
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_data_format_by_id_.clear();
  cb_id_by_requirement_index_.clear();
  cb_num_pages_by_requirement_index_.clear();
  cb_initial_reserve_pages_by_requirement_index_.clear();
  local_non_input_cb_ids_.clear();
  emitted_cb_front_pages_.clear();
  emitted_cb_consumed_front_pages_.clear();
  active_cb_allocation_reserved_pages_.clear();
  dead_fragment_fill_data_vars_.clear();
  active_scalar_reduction_.reset();
  scalar_reduction_counter_ = 0;
  tile_regs_scope_active_ = false;
  thread_idx_x_expr_.clear();
  logical_grid_x_ = 1;
  logical_grid_y_ = 1;
  logical_grid_z_ = 1;
  linearization_ = "row_major";
}

std::string CodeGenBlackhole::GetKernelCode() const {
  // Return the kernel code with TT-Metal headers but without TVM-specific headers
  // decl_stream now contains TT-Metal headers (dataflow_api.h, etc.)
  // stream contains the actual kernel implementation
  const std::string body = stream.str();
  std::ostringstream kernel_code;
  kernel_code << decl_stream.str();
  if (body.find("tilelang_cb_write_ptr_bytes_direct(") != std::string::npos) {
    kernel_code << "ALWI uint32_t tilelang_cb_write_ptr_bytes_direct(uint32_t cb_id) {\n";
    kernel_code << "#ifdef COMPILE_FOR_TRISC\n";
    kernel_code << "  uint32_t address = 0;\n";
    kernel_code << "  PACK({\n";
    kernel_code << "    address = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;\n";
    kernel_code << "    mailbox_write(ckernel::ThreadId::MathThreadId, address);\n";
    kernel_code << "    mailbox_write(ckernel::ThreadId::UnpackThreadId, address);\n";
    kernel_code << "  })\n";
    kernel_code << "  MATH(address = mailbox_read(ckernel::ThreadId::PackThreadId);)\n";
    kernel_code << "  UNPACK(address = mailbox_read(ckernel::ThreadId::PackThreadId);)\n";
    kernel_code << "  return address;\n";
    kernel_code << "#else\n";
    kernel_code << "  return experimental::CircularBuffer(cb_id).get_write_ptr();\n";
    kernel_code << "#endif\n";
    kernel_code << "}\n";
  }
  kernel_code << body;
  return kernel_code.str();
}

void CodeGenBlackhole::AddFunction(const tvm::GlobalVar &gvar,
                                   const tvm::tir::PrimFunc &f) {
  // Emit TT-Metal headers for kernel code (per-instance, not static)
  if (!headers_emitted_) {
    // Clear decl_stream to remove TVM headers added by CodeGenCHost::Init
    decl_stream.str("");
    decl_stream.clear();

    decl_stream << "// TT-Metal kernel generated by TileLang\n";
    decl_stream << "#include <cstdint>\n";
    decl_stream << "#include <cmath>\n";
    decl_stream << "#include <limits>\n";
    decl_stream << "\n";
    decl_stream << "template <typename To, typename From>\n";
    decl_stream << "static inline To tilelang_bit_cast(From value) {\n";
    decl_stream << "  static_assert(sizeof(To) == sizeof(From), \"tilelang_bit_cast requires equal-sized types\");\n";
    decl_stream << "  To out;\n";
    decl_stream << "  __builtin_memcpy(&out, &value, sizeof(To));\n";
    decl_stream << "  return out;\n";
    decl_stream << "}\n";
    decl_stream << "\n";

    // Detect core type from function attributes (IR-driven, not function name)
    std::string core_type_str = GetCoreTypeForCodegen(f);
    if (core_type_str == "brisc") {
      core_type_ = CoreType::kBRISC;
    } else if (core_type_str == "ncrisc") {
      core_type_ = CoreType::kNCRISC;
    } else if (core_type_str == "trisc") {
      core_type_ = CoreType::kTRISC;
    } else {
      ICHECK(false) << "Blackhole codegen requires executable segment core_type, got '"
                    << core_type_str << "'";
    }

    // Include appropriate API header based on core type
    switch (core_type_) {
      case CoreType::kBRISC:
      case CoreType::kNCRISC:
        decl_stream << "// DataMovement kernel API (BRISC/NCRISC)\n";
        decl_stream << "#include \"api/dataflow/dataflow_api.h\"\n";
        decl_stream << "#include \"experimental/circular_buffer.h\"\n";
        decl_stream << "#include \"experimental/tensor.h\"\n";
        break;
      case CoreType::kTRISC:
        decl_stream << "// Compute kernel API (TRISC)\n";
        decl_stream << "#ifndef REDUCE_OP\n";
        decl_stream << "#define REDUCE_OP PoolType::SUM\n";
        decl_stream << "#endif\n";
        decl_stream << "#ifndef REDUCE_DIM\n";
        decl_stream << "#define REDUCE_DIM ReduceDim::REDUCE_ROW\n";
        decl_stream << "#endif\n";
        decl_stream << "#include \"api/compute/pack.h\"\n";
        decl_stream << "#include \"api/compute/reconfig_data_format.h\"\n";
        decl_stream << "#include \"api/compute/tile_move_copy.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_binary.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/eltwise_unary.h\"\n";
        decl_stream << "#include \"api/compute/bcast.h\"\n";
        decl_stream << "#include \"api/compute/binary_max_min.h\"\n";
        decl_stream << "#include \"api/compute/reduce.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/fill.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/recip.h\"\n";
        decl_stream << "#include \"api/compute/compute_kernel_api.h\"\n";
        decl_stream << "#include \"api/compute/matmul.h\"\n";
        decl_stream << "#include \"api/debug/waypoint.h\"\n";
        decl_stream << "#include \"experimental/circular_buffer.h\"\n";
        decl_stream << "#include \"hostdevcommon/kernel_structs.h\"\n";
	        decl_stream << "using half = _Float16;\n";
	        decl_stream << "static constexpr float inff = std::numeric_limits<float>::infinity();\n";
	        decl_stream << "ALWI uint32_t tilelang_bitcast_float_to_u32(float value) {\n";
	        decl_stream << "  return tilelang_bit_cast<uint32_t>(value);\n";
	        decl_stream << "}\n";
        decl_stream << "ALWI uint16_t tilelang_float_to_half_bits(float value) {\n";
        decl_stream << "  const uint32_t bits = tilelang_bitcast_float_to_u32(value);\n";
        decl_stream << "  const uint32_t sign = (bits >> 16) & 0x8000u;\n";
        decl_stream << "  const uint32_t exponent = (bits >> 23) & 0xffu;\n";
        decl_stream << "  uint32_t mantissa = bits & 0x7fffffu;\n";
        decl_stream << "  if (exponent == 0xffu) {\n";
        decl_stream << "    if (mantissa == 0u) {\n";
        decl_stream << "      return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "    }\n";
        decl_stream << "    mantissa >>= 13;\n";
        decl_stream << "    return static_cast<uint16_t>(sign | 0x7c00u | mantissa | (mantissa == 0u));\n";
        decl_stream << "  }\n";
        decl_stream << "  int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;\n";
        decl_stream << "  if (half_exponent >= 31) {\n";
        decl_stream << "    return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "  }\n";
        decl_stream << "  if (half_exponent <= 0) {\n";
        decl_stream << "    if (half_exponent < -10) {\n";
        decl_stream << "      return static_cast<uint16_t>(sign);\n";
        decl_stream << "    }\n";
        decl_stream << "    mantissa |= 0x800000u;\n";
        decl_stream << "    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);\n";
        decl_stream << "    uint32_t half_mantissa = mantissa >> shift;\n";
        decl_stream << "    const uint32_t round_bit = 1u << (shift - 1);\n";
        decl_stream << "    const uint32_t remainder = mantissa & (round_bit - 1u);\n";
        decl_stream << "    const bool round_up = (mantissa & round_bit) != 0u && (remainder != 0u || (half_mantissa & 1u) != 0u);\n";
        decl_stream << "    if (round_up) {\n";
        decl_stream << "      ++half_mantissa;\n";
        decl_stream << "    }\n";
        decl_stream << "    return static_cast<uint16_t>(sign | half_mantissa);\n";
        decl_stream << "  }\n";
        decl_stream << "  uint32_t half_mantissa = mantissa >> 13;\n";
        decl_stream << "  const uint32_t remainder = mantissa & 0x1fffu;\n";
        decl_stream << "  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0u)) {\n";
        decl_stream << "    ++half_mantissa;\n";
        decl_stream << "    if (half_mantissa == 0x400u) {\n";
        decl_stream << "      half_mantissa = 0u;\n";
        decl_stream << "      ++half_exponent;\n";
        decl_stream << "      if (half_exponent >= 31) {\n";
        decl_stream << "        return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | (half_mantissa & 0x3ffu));\n";
	        decl_stream << "}\n";
	        decl_stream << "ALWI uint16_t tilelang_float_to_bfloat_bits(float value) {\n";
	        decl_stream << "  const uint32_t bits = tilelang_bit_cast<uint32_t>(value);\n";
	        decl_stream << "  const uint32_t lsb = (bits >> 16) & 1u;\n";
	        decl_stream << "  const uint32_t rounding_bias = 0x7fffu + lsb;\n";
	        decl_stream << "  return static_cast<uint16_t>((bits + rounding_bias) >> 16);\n";
	        decl_stream << "}\n";
	        decl_stream << "ALWI float tilelang_bfloat_bits_to_float(uint16_t value) {\n";
	        decl_stream << "  return tilelang_bit_cast<float>(static_cast<uint32_t>(value) << 16);\n";
	        decl_stream << "}\n";
        decl_stream << "ALWI float tilelang_fast_exp2f(float x) {\n";
        decl_stream << "  if (x <= -126.0f) { return 0.0f; }\n";
        decl_stream << "  if (x >= 126.0f) { x = 126.0f; }\n";
        decl_stream << "  int ipart = static_cast<int>(x);\n";
        decl_stream << "  if (static_cast<float>(ipart) > x) { --ipart; }\n";
        decl_stream << "  const float fpart = x - static_cast<float>(ipart);\n";
        decl_stream << "  const float poly = 1.0f + fpart * (0.69314718f + fpart * (0.24022651f + fpart * (0.05550411f + fpart * 0.00961813f)));\n";
	        decl_stream << "  const uint32_t exponent_bits = static_cast<uint32_t>(ipart + 127) << 23;\n";
	        decl_stream << "  return tilelang_bit_cast<float>(exponent_bits) * poly;\n";
	        decl_stream << "}\n";
        decl_stream << "template <typename T>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_fill_fragment(T* dst, uint32_t num_elements, T value) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = value; }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_add_fragment(DstT* dst, const SrcT* src, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = static_cast<DstT>(dst[i] + static_cast<DstT>(src[i])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_cast_fragment_slice(DstT* dst, const SrcT* src, uint32_t dst_offset, uint32_t src_offset, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[dst_offset + i] = static_cast<DstT>(src[src_offset + i]); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_tilize_fragment_tile_nfaces(const BitsT* src, BitsT* dst) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  uint32_t dst_index = 0;\n";
        decl_stream << "  for (uint32_t face_y = 0; face_y < kTileRows / kFaceRows; ++face_y) {\n";
        decl_stream << "    for (uint32_t face_x = 0; face_x < kTileCols / kFaceCols; ++face_x) {\n";
        decl_stream << "      for (uint32_t row = 0; row < kFaceRows; ++row) {\n";
        decl_stream << "        const BitsT* src_row = src + (face_y * kFaceRows + row) * kTileCols + face_x * kFaceCols;\n";
        decl_stream << "        for (uint32_t col = 0; col < kFaceCols; ++col) {\n";
        decl_stream << "          dst[dst_index++] = src_row[col];\n";
        decl_stream << "        }\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_tilize_fragment_slice_nfaces(const BitsT* src, BitsT* dst, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  const uint32_t tiles_per_row = row_width / kTileCols;\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) {\n";
        decl_stream << "    const uint32_t logical_index = dst_offset_elements + i;\n";
        decl_stream << "    const uint32_t global_row = logical_index / row_width;\n";
        decl_stream << "    const uint32_t global_col = logical_index % row_width;\n";
        decl_stream << "    const uint32_t tile_row = global_row / kTileRows;\n";
        decl_stream << "    const uint32_t tile_col = global_col / kTileCols;\n";
        decl_stream << "    const uint32_t row_in_tile = global_row % kTileRows;\n";
        decl_stream << "    const uint32_t col_in_tile = global_col % kTileCols;\n";
        decl_stream << "    const uint32_t face_row = row_in_tile / kFaceRows;\n";
        decl_stream << "    const uint32_t face_col = col_in_tile / kFaceCols;\n";
        decl_stream << "    const uint32_t row_in_face = row_in_tile % kFaceRows;\n";
        decl_stream << "    const uint32_t col_in_face = col_in_tile % kFaceCols;\n";
        decl_stream << "    const uint32_t tile_index = tile_row * tiles_per_row + tile_col;\n";
        decl_stream << "    const uint32_t tiled_index = tile_index * 1024u + face_row * (kFaceRows * kTileCols) + face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face;\n";
        decl_stream << "    dst[tiled_index] = src[i];\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_fill_tiled_cb_slice_nfaces(BitsT* dst, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width, BitsT value) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  const uint32_t tiles_per_row = row_width / kTileCols;\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) {\n";
        decl_stream << "    const uint32_t logical_index = dst_offset_elements + i;\n";
        decl_stream << "    const uint32_t global_row = logical_index / row_width;\n";
        decl_stream << "    const uint32_t global_col = logical_index % row_width;\n";
        decl_stream << "    const uint32_t tile_row = global_row / kTileRows;\n";
        decl_stream << "    const uint32_t tile_col = global_col / kTileCols;\n";
        decl_stream << "    const uint32_t row_in_tile = global_row % kTileRows;\n";
        decl_stream << "    const uint32_t col_in_tile = global_col % kTileCols;\n";
        decl_stream << "    const uint32_t face_row = row_in_tile / kFaceRows;\n";
        decl_stream << "    const uint32_t face_col = col_in_tile / kFaceCols;\n";
        decl_stream << "    const uint32_t row_in_face = row_in_tile % kFaceRows;\n";
        decl_stream << "    const uint32_t col_in_face = col_in_tile % kFaceCols;\n";
        decl_stream << "    const uint32_t tile_index = tile_row * tiles_per_row + tile_col;\n";
        decl_stream << "    const uint32_t tiled_index = tile_index * 1024u + face_row * (kFaceRows * kTileCols) + face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face;\n";
        decl_stream << "    dst[tiled_index] = value;\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "ALWI void tilelang_pack_fill_bfloat16_tiled_cb(uint32_t cb_id, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width, float value) {\n";
        decl_stream << "  (void)dst_offset_elements; (void)row_width;\n";
        decl_stream << "  const uint32_t num_tiles = (num_elements + 1023u) / 1024u;\n";
        decl_stream << "  fill_tile_init();\n";
        decl_stream << "  for (uint32_t tile = 0; tile < num_tiles; ++tile) {\n";
        decl_stream << "    tile_regs_acquire();\n";
        decl_stream << "    fill_tile(0, value);\n";
        decl_stream << "    tile_regs_commit();\n";
        decl_stream << "    tile_regs_wait();\n";
        decl_stream << "    pack_reconfig_data_format(cb_id);\n";
        decl_stream << "    pack_tile<true>(0, cb_id, tile);\n";
        decl_stream << "    tile_regs_release();\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "ALWI void tilelang_pack_fill_float32_tiled_cb(uint32_t cb_id, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width, float value) {\n";
        decl_stream << "  (void)dst_offset_elements; (void)row_width;\n";
        decl_stream << "  const uint32_t num_tiles = (num_elements + 1023u) / 1024u;\n";
        decl_stream << "  fill_tile_init();\n";
        decl_stream << "  for (uint32_t tile = 0; tile < num_tiles; ++tile) {\n";
        decl_stream << "    tile_regs_acquire();\n";
        decl_stream << "    fill_tile(0, value);\n";
        decl_stream << "    tile_regs_commit();\n";
        decl_stream << "    tile_regs_wait();\n";
        decl_stream << "    pack_reconfig_data_format(cb_id);\n";
        decl_stream << "    pack_tile<true>(0, cb_id, tile);\n";
        decl_stream << "    tile_regs_release();\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_untilize_fragment_tile_nfaces(const BitsT* src, BitsT* dst) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  uint32_t src_index = 0;\n";
        decl_stream << "  for (uint32_t face_y = 0; face_y < kTileRows / kFaceRows; ++face_y) {\n";
        decl_stream << "    for (uint32_t face_x = 0; face_x < kTileCols / kFaceCols; ++face_x) {\n";
        decl_stream << "      for (uint32_t row = 0; row < kFaceRows; ++row) {\n";
        decl_stream << "        BitsT* dst_row = dst + (face_y * kFaceRows + row) * kTileCols + face_x * kFaceCols;\n";
        decl_stream << "        for (uint32_t col = 0; col < kFaceCols; ++col) {\n";
        decl_stream << "          dst_row[col] = src[src_index++];\n";
        decl_stream << "        }\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        break;
      default:
        ICHECK(false) << "Blackhole codegen reached unknown core_type enum";
        break;
    }
    decl_stream << "\n";
    headers_emitted_ = true;
  }

  // Generate TT-Metal kernel_main function using IR visitor
  GenerateGenericKernelMain(f, gvar->name_hint);
}

void CodeGenBlackhole::GenerateGenericKernelMain(const tvm::tir::PrimFunc &f,
                                                  const std::string &func_name) {
  // Add function name as comment
  stream << "// Kernel: " << func_name << "\n";

  // Generate kernel_main entry point (TT-Metal convention)
  stream << "void kernel_main() {\n";

  // Generate argument loading code
  // TT-Metal kernels use get_arg_val<uint32_t>(arg_index) to read arguments
  stream << "  // Load kernel arguments from runtime\n";
  LoadCorePlan(f);
  LoadLogicalTileLayouts(f);
  LoadAccessorOffsets(f);
  LoadCBConfigMetadata(f);
  if (HasRuntimeArgsForCodegen(f)) {
    EmitRuntimeArgLoads(f);
    if (EmitTypedReductionRegionIfSupported(f)) {
      stream << "}\n\n";
      return;
    }
  } else if (EmitTypedReductionRegionIfSupported(f)) {
    stream << "}\n\n";
    return;
  }
  dead_fragment_fill_data_vars_ = CollectDeadFragmentFillDataVars(f->body);
  this->VisitStmt(f->body);
  stream << "}\n\n";
}

bool CodeGenBlackhole::EmitTypedReductionRegionIfSupported(const tvm::tir::PrimFunc& f) {
  if (core_type_ != CoreType::kTRISC) {
    return false;
  }

  struct ReductionOpRecord {
    std::string input_buffer;
    std::string output_buffer;
    std::string host_buffer;
    std::string accumulator_dtype;
    std::string reduction_kind;
    std::string reduction_dim;
    int repeat_extent = 1;
    std::vector<int64_t> input_cb_requirement_indices;
    std::vector<int64_t> output_cb_requirement_indices;
  };

  std::vector<ReductionOpRecord> reduction_records;
  for (const ffi::Any& segment_any :
       tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen")) {
    auto segment = segment_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (segment.empty() || MapGetString(segment, "kind") != "compute") {
      continue;
    }
    for (const ffi::Any& op_any : MapGetArray(segment, "compute_ops")) {
      auto op = op_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (op.empty() || MapGetString(op, "kind") != "reduce" ||
          MapGetString(op, "operation_name") != "reduce_tile") {
        continue;
      }
      ReductionOpRecord info;
      info.accumulator_dtype = MapGetString(op, "accumulator_dtype");
      info.reduction_kind = MapGetString(op, "reduction_kind");
      info.reduction_dim = MapGetString(op, "reduction_dim");
      info.repeat_extent = static_cast<int>(MapGetInt(op, "repeat_extent", 1));
      for (const ffi::Any& binding_any : MapGetArray(op, "operand_bindings")) {
        auto binding = binding_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
            ffi::Map<ffi::String, ffi::Any>());
        std::vector<int64_t> cb_requirement_indices;
        for (const ffi::Any& index_any : MapGetArray(binding, "cb_requirement_indices")) {
          cb_requirement_indices.push_back(Downcast<Integer>(index_any).IntValue());
        }
        const std::string role = MapGetString(binding, "role");
        if (role == "input") {
          info.input_buffer = MapGetString(binding, "buffer");
          info.input_cb_requirement_indices = std::move(cb_requirement_indices);
        } else if (role == "output") {
          info.output_buffer = MapGetString(binding, "buffer");
          info.host_buffer = MapGetString(binding, "host_buffer");
          info.output_cb_requirement_indices = std::move(cb_requirement_indices);
        }
      }
      reduction_records.push_back(info);
    }
  }
  if (reduction_records.empty()) {
    return false;
  }
  std::string reduction_kind;
  std::string reduction_dim;
  int repeat_extent = 0;
  for (const ReductionOpRecord& info : reduction_records) {
    if (info.reduction_kind.empty() || info.reduction_dim.empty() ||
        info.repeat_extent <= 0) {
      return false;
    }
    if (reduction_kind.empty()) {
      reduction_kind = info.reduction_kind;
      reduction_dim = info.reduction_dim;
      repeat_extent = info.repeat_extent;
      continue;
    }
    if (reduction_kind != info.reduction_kind ||
        reduction_dim != info.reduction_dim ||
        repeat_extent != info.repeat_extent) {
      return false;
    }
  }
  if (reduction_kind != "max" ||
      (reduction_dim != "row" && reduction_dim != "col")) {
    return false;
  }

  auto is_reduced_value_dtype = [](const std::string& dtype) {
    return dtype == "Float32" || dtype == "BFloat16" || dtype == "Float16_b" ||
           dtype == "Float16";
  };
  const ReductionOpRecord* data_record = nullptr;
  for (const ReductionOpRecord& info : reduction_records) {
    if (is_reduced_value_dtype(info.accumulator_dtype) && !info.host_buffer.empty() &&
        !info.input_buffer.empty()) {
      data_record = &info;
      break;
    }
  }
  if (data_record == nullptr || data_record->output_buffer.empty()) {
    return false;
  }

  auto layout_it = logical_tile_layout_bindings_by_buffer_name_.find(data_record->input_buffer);
  if (layout_it == logical_tile_layout_bindings_by_buffer_name_.end() ||
      layout_it->second.logical_shape.size() < 2U) {
    return false;
  }
  const LogicalTileLayoutBinding& input_layout = layout_it->second;
  const auto* rows_imm = input_layout.logical_shape[0].as<IntImmNode>();
  const auto* cols_imm = input_layout.logical_shape[1].as<IntImmNode>();
  if (rows_imm == nullptr || cols_imm == nullptr ||
      rows_imm->value <= 0 || cols_imm->value <= 0) {
    return false;
  }
  const int rows = static_cast<int>(rows_imm->value);
  const int cols = static_cast<int>(cols_imm->value);
  const bool reduce_over_cols = reduction_dim == "row";
  const int output_extent = reduce_over_cols ? rows : cols;
  const int reduction_extent = reduce_over_cols ? cols : rows;
  if (output_extent <= 0 || reduction_extent <= 0 ||
      output_extent > 256 || reduction_extent > 1024) {
    return false;
  }
  const int input_tile_rows = (rows + 31) / 32;
  const int input_tiles_per_row = (cols + 31) / 32;

  struct CBInfo {
    int id = -1;
    int num_pages = 0;
    int page_size = 0;
    std::string data_format;
  };
  auto read_cb_info = [&](const ffi::Map<ffi::String, ffi::Any>& cb) -> CBInfo {
    CBInfo result;
    result.id = static_cast<int>(MapGetInt(cb, "cb_id", -1));
    result.num_pages = static_cast<int>(MapGetInt(cb, "num_pages", 0));
    result.page_size = static_cast<int>(MapGetInt(cb, "page_size", 0));
    result.data_format = MapGetString(cb, "data_format");
    return result;
  };
  auto cb_requirement_indices =
      [&](const ffi::Map<ffi::String, ffi::Any>& cb) -> std::vector<int64_t> {
    std::vector<int64_t> indices;
    for (const ffi::Any& index_any : MapGetArray(cb, "requirement_indices")) {
      indices.push_back(Downcast<Integer>(index_any).IntValue());
    }
    return indices;
  };
  auto find_cb_by_requirement_index = [&](int64_t requirement_index) -> CBInfo {
    CBInfo result;
    int matches = 0;
    for (const ffi::Any& cb_any : GetCBConfigsForCodegen(f)) {
      auto cb = cb_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (cb.empty()) {
        continue;
      }
      const std::vector<int64_t> indices = cb_requirement_indices(cb);
      if (std::find(indices.begin(), indices.end(), requirement_index) ==
          indices.end()) {
        continue;
      }
      result = read_cb_info(cb);
      ++matches;
    }
    return matches == 1 ? result : CBInfo();
  };
  auto find_cb_for_binding_requirements =
      [&](const std::vector<int64_t>& requirement_indices) -> CBInfo {
    CBInfo result;
    bool found = false;
    for (int64_t requirement_index : requirement_indices) {
      CBInfo candidate = find_cb_by_requirement_index(requirement_index);
      if (candidate.id < 0) {
        return CBInfo();
      }
      if (found && candidate.id != result.id) {
        return CBInfo();
      }
      result = candidate;
      found = true;
    }
    return found ? result : CBInfo();
  };

  const CBInfo input_cb =
      find_cb_for_binding_requirements(data_record->input_cb_requirement_indices);
  if (input_cb.id < 0 || input_cb.num_pages <= 0 ||
      input_cb.num_pages < input_tile_rows * input_tiles_per_row) {
    return false;
  }
  if (input_cb.data_format != "Float32" && input_cb.data_format != "Float16_b") {
    return false;
  }

  auto host_element_size_bytes = [&](const std::string& host_buffer,
                                     int default_value) -> int {
    for (const ffi::Any& plan_any : tt_program_projection::GetExecutableArrayField(
             f, "Blackhole codegen",
             tt_program_projection::executable_key::kBufferDistributionPlans)) {
      auto plan = plan_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (!plan.empty() && MapGetString(plan, "buffer") == host_buffer) {
        return static_cast<int>(MapGetInt(plan, "page_size_bytes", default_value));
      }
    }
    return default_value;
  };

  enum class ReductionProjection {
    kReducedValue,
    kCoordinate,
  };
  struct ReductionChannel {
    ReductionProjection projection{ReductionProjection::kReducedValue};
    std::string accumulator_dtype;
    std::string host_buffer;
    CBInfo output_cb;
    int element_size_bytes{0};
    std::string storage_var;
  };

  std::vector<ReductionChannel> channels;
  channels.reserve(reduction_records.size());
  for (const ReductionOpRecord& info : reduction_records) {
    if (info.output_buffer.empty() || info.host_buffer.empty()) {
      return false;
    }
    if (is_reduced_value_dtype(info.accumulator_dtype) &&
        info.input_buffer != data_record->input_buffer) {
      return false;
    }

    ReductionChannel channel;
    channel.accumulator_dtype = info.accumulator_dtype;
    channel.host_buffer = info.host_buffer;
    channel.output_cb = find_cb_for_binding_requirements(info.output_cb_requirement_indices);
    if (channel.output_cb.id < 0 || channel.output_cb.num_pages <= 0) {
      return false;
    }
    if (is_reduced_value_dtype(info.accumulator_dtype)) {
      channel.projection = ReductionProjection::kReducedValue;
      channel.element_size_bytes = host_element_size_bytes(info.host_buffer, 4);
      if (channel.element_size_bytes != 2 && channel.element_size_bytes != 4) {
        return false;
      }
      if (channel.element_size_bytes == 2 &&
          channel.output_cb.data_format != "Float16_b" &&
          channel.output_cb.data_format != "Float16") {
        return false;
      }
    } else if (info.accumulator_dtype == "Int32") {
      channel.projection = ReductionProjection::kCoordinate;
      channel.element_size_bytes = host_element_size_bytes(info.host_buffer, 4);
      if (channel.element_size_bytes != 4 || channel.output_cb.data_format != "Int32") {
        return false;
      }
    } else {
      return false;
    }
    channels.push_back(std::move(channel));
  }
  if (channels.empty()) {
    return false;
  }

  if (repeat_extent <= 0 || repeat_extent > 32) {
    return false;
  }
  int duplicate_groups = std::max(1, output_extent / 16);
  if (input_layout.thread_extent.defined()) {
    if (const auto* thread_extent = input_layout.thread_extent.as<IntImmNode>()) {
      for (const ReductionChannel& channel : channels) {
        const int writer_event_elements = channel.element_size_bytes == 2 ? 16 : 32;
        duplicate_groups = std::max(
            duplicate_groups,
            std::max(1, static_cast<int>(thread_extent->value / writer_event_elements)));
      }
    }
  }

  for (size_t i = 0; i < channels.size(); ++i) {
    channels[i].storage_var = "__tl_reduction_accum_" + std::to_string(i);
  }
  std::string history_var = "__tl_reduction_coord_history";
  bool needs_separate_history = true;
  for (const ReductionChannel& channel : channels) {
    if (channel.projection == ReductionProjection::kCoordinate) {
      history_var = channel.storage_var;
      needs_separate_history = false;
      break;
    }
  }

  stream << "\n// Typed repeated reductions lowered from executable compute records.\n";
  stream << "cb_wait_front(" << input_cb.id << ", " << input_cb.num_pages << ");\n";
  if (needs_separate_history) {
    stream << "int32_t " << history_var << "[" << output_extent * repeat_extent << "];\n";
  }
  for (const ReductionChannel& channel : channels) {
    if (channel.projection == ReductionProjection::kReducedValue) {
      stream << "float " << channel.storage_var << "[" << output_extent * repeat_extent << "];\n";
    } else {
      stream << "int32_t " << channel.storage_var << "[" << output_extent * repeat_extent << "];\n";
    }
  }
  if (input_cb.data_format == "Float32") {
    stream << "const float* __tl_region_input_tiles[" << input_cb.num_pages << "];\n";
    stream << "{ experimental::CircularBuffer __tl_region_input_cb(" << input_cb.id << "); "
           << "for (uint32_t __tl_tile = 0; __tl_tile < " << input_cb.num_pages
           << "; ++__tl_tile) { __tl_region_input_tiles[__tl_tile] = "
           << "reinterpret_cast<const float*>(__tl_region_input_cb.get_tile_address(__tl_tile)); } }\n";
  } else {
    stream << "const uint16_t* __tl_region_input_tiles[" << input_cb.num_pages << "];\n";
    stream << "{ experimental::CircularBuffer __tl_region_input_cb(" << input_cb.id << "); "
           << "for (uint32_t __tl_tile = 0; __tl_tile < " << input_cb.num_pages
           << "; ++__tl_tile) { __tl_region_input_tiles[__tl_tile] = "
           << "reinterpret_cast<const uint16_t*>(__tl_region_input_cb.get_tile_address(__tl_tile)); } }\n";
  }
  stream << "MATH({\n";
  stream << "  constexpr bool kReduceOverCols = " << (reduce_over_cols ? "true" : "false") << ";\n";
  stream << "  constexpr uint32_t kLogicalRows = " << rows << ";\n";
  stream << "  constexpr uint32_t kLogicalCols = " << cols << ";\n";
  stream << "  constexpr uint32_t kOutputExtent = " << output_extent << ";\n";
  stream << "  constexpr uint32_t kReductionExtent = " << reduction_extent << ";\n";
  stream << "  constexpr uint32_t kRepeatExtent = " << repeat_extent << ";\n";
  stream << "  constexpr uint32_t kTilesPerRow = " << input_tiles_per_row << ";\n";
  stream << "  constexpr uint32_t kFaceRows = 16;\n";
  stream << "  constexpr uint32_t kFaceCols = 16;\n";
  stream << "  for (uint32_t out_coord = 0; out_coord < kOutputExtent; ++out_coord) {\n";
  stream << "    for (uint32_t repeat = 0; repeat < kRepeatExtent; ++repeat) {\n";
  stream << "      float best = -std::numeric_limits<float>::infinity();\n";
  stream << "      int32_t best_coord = -1;\n";
  stream << "      for (uint32_t reduce_coord = 0; reduce_coord < kReductionExtent; ++reduce_coord) {\n";
  stream << "        bool already_emitted = false;\n";
  stream << "        for (uint32_t prev = 0; prev < repeat; ++prev) { "
         << "already_emitted = already_emitted || "
         << "(" << history_var << "[out_coord * kRepeatExtent + prev] == "
         << "static_cast<int32_t>(reduce_coord)); }\n";
  stream << "        if (already_emitted) { continue; }\n";
  stream << "        const uint32_t logical_row = kReduceOverCols ? out_coord : reduce_coord;\n";
  stream << "        const uint32_t logical_col = kReduceOverCols ? reduce_coord : out_coord;\n";
  stream << "        if (logical_row >= kLogicalRows || logical_col >= kLogicalCols) { continue; }\n";
  stream << "        const uint32_t tile_index = (logical_row / 32u) * kTilesPerRow + "
         << "(logical_col / 32u);\n";
  stream << "        const uint32_t row_in_tile = logical_row % 32u;\n";
  stream << "        const uint32_t col_in_tile = logical_col % 32u;\n";
  stream << "        const uint32_t face_row = row_in_tile / kFaceRows;\n";
  stream << "        const uint32_t face_col = col_in_tile / kFaceCols;\n";
  stream << "        const uint32_t row_in_face = row_in_tile % kFaceRows;\n";
  stream << "        const uint32_t col_in_face = col_in_tile % kFaceCols;\n";
  stream << "        const uint32_t offset = face_row * (kFaceRows * 32u) + "
         << "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face;\n";
  if (input_cb.data_format == "Float32") {
    stream << "        const float value = __tl_region_input_tiles[tile_index][offset];\n";
  } else {
    stream << "        const uint16_t bits = __tl_region_input_tiles[tile_index][offset];\n";
    stream << "        const float value = tilelang_bit_cast<float>(static_cast<uint32_t>(bits) << 16);\n";
  }
  stream << "        if (value > best || (value == best && "
         << "static_cast<int32_t>(reduce_coord) > best_coord)) {\n";
  stream << "          best = value;\n";
  stream << "          best_coord = static_cast<int32_t>(reduce_coord);\n";
  stream << "        }\n";
  stream << "      }\n";
  if (needs_separate_history) {
    stream << "      " << history_var << "[out_coord * kRepeatExtent + repeat] = "
           << "best_coord;\n";
  }
  for (const ReductionChannel& channel : channels) {
    if (channel.projection == ReductionProjection::kReducedValue) {
      stream << "      " << channel.storage_var
             << "[out_coord * kRepeatExtent + repeat] = best;\n";
    } else {
      stream << "      " << channel.storage_var
             << "[out_coord * kRepeatExtent + repeat] = best_coord;\n";
    }
  }
  stream << "    }\n";
  stream << "  }\n";
  stream << "})\n";

  for (int repeat = 0; repeat < repeat_extent; ++repeat) {
    for (int group = 0; group < duplicate_groups; ++group) {
      stream << "{\n";
      for (const ReductionChannel& channel : channels) {
        stream << "cb_reserve_back(" << channel.output_cb.id << ", 1);\n";
      }
      for (size_t i = 0; i < channels.size(); ++i) {
        const ReductionChannel& channel = channels[i];
        const char* pointer_type =
            channel.projection == ReductionProjection::kCoordinate
                ? "int32_t"
                : (channel.element_size_bytes == 2 ? "uint16_t" : "float");
        stream << pointer_type << "* __tl_reduction_out_" << i
               << " = reinterpret_cast<" << pointer_type
               << "*>(tilelang_cb_write_ptr_bytes_direct(" << channel.output_cb.id << "));\n";
      }
      stream << "MATH({ for (uint32_t out_coord = 0; out_coord < " << output_extent
             << "; ++out_coord) { ";
      for (size_t i = 0; i < channels.size(); ++i) {
        const ReductionChannel& channel = channels[i];
        const std::string offset =
            "out_coord * " + std::to_string(repeat_extent) + " + " +
            std::to_string(repeat);
        if (channel.projection == ReductionProjection::kReducedValue) {
          if (channel.element_size_bytes == 2) {
            const char* cast_helper = channel.output_cb.data_format == "Float16"
                                          ? "tilelang_float_to_half_bits"
                                          : "tilelang_float_to_bfloat_bits";
            stream << "__tl_reduction_out_" << i << "[out_coord] = "
                   << cast_helper << "(" << channel.storage_var << "[" << offset << "]); ";
          } else {
            stream << "__tl_reduction_out_" << i << "[out_coord] = "
                   << channel.storage_var << "[" << offset << "]; ";
          }
        } else {
          stream << "__tl_reduction_out_" << i << "[out_coord] = "
                 << channel.storage_var << "[" << offset << "]; ";
        }
      }
      stream << "} mailbox_write(ckernel::ThreadId::PackThreadId, 1); })\n";
      stream << "PACK({ volatile uint32_t __tl_done = "
             << "mailbox_read(ckernel::ThreadId::MathThreadId); (void)__tl_done; })\n";
      for (const ReductionChannel& channel : channels) {
        stream << "cb_push_back(" << channel.output_cb.id << ", 1);\n";
      }
      stream << "}\n";
    }
  }
  stream << "cb_pop_front(" << input_cb.id << ", " << input_cb.num_pages << ");\n";
  return true;
}

void CodeGenBlackhole::LoadCorePlan(const tvm::tir::PrimFunc &f) {
  logical_grid_x_ = 1;
  logical_grid_y_ = 1;
  logical_grid_z_ = 1;
  linearization_ = "row_major";

  auto core_plan = GetCorePlanForCodegen(f);
  if (core_plan.empty()) {
    return;
  }

  if (auto v = core_plan.Get("logical_grid_x")) {
    logical_grid_x_ = Downcast<tvm::Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_x")) {
    logical_grid_x_ = Downcast<tvm::Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("logical_grid_y")) {
    logical_grid_y_ = Downcast<tvm::Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_y")) {
    logical_grid_y_ = Downcast<tvm::Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("logical_grid_z")) {
    logical_grid_z_ = Downcast<tvm::Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_z")) {
    logical_grid_z_ = Downcast<tvm::Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("linearization")) {
    linearization_ = Downcast<tvm::ffi::String>(v.value());
  }
}

void CodeGenBlackhole::LoadLogicalTileLayouts(const tvm::tir::PrimFunc& f) {
  logical_tile_layout_bindings_by_buffer_name_.clear();
  auto ingest_spec = [&](const ffi::Map<ffi::String, ffi::Any>& spec) {
    auto maybe_buffer = spec.Get(ffi::String(schema_key::kBuffer));
    if (!maybe_buffer) {
      return;
    }
    LogicalTileLayoutBinding binding;
    binding.buffer_name = Downcast<ffi::String>(maybe_buffer.value());
    if (auto v = spec.Get(ffi::String("logical_shape"))) {
      binding.logical_shape = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String("local_shape"))) {
      binding.local_shape = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kInverseLogicalIndexVars))) {
      binding.inverse_logical_index_vars = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kInverseLogicalIndexExprs))) {
      binding.inverse_logical_index_exprs = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kThreadExtent))) {
      binding.thread_extent = Downcast<tvm::PrimExpr>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kReplicateExtent))) {
      binding.replicate_extent = Downcast<tvm::PrimExpr>(v.value());
    }
    if (binding.buffer_name.empty()) {
      return;
    }
    auto [it, inserted] =
        logical_tile_layout_bindings_by_buffer_name_.emplace(binding.buffer_name, binding);
    if (!inserted) {
      ICHECK(StructuralEqual()(it->second.logical_shape, binding.logical_shape))
          << "Blackhole codegen requires a single logical bridge shape per buffer; "
          << binding.buffer_name;
      ICHECK(StructuralEqual()(it->second.local_shape, binding.local_shape))
          << "Blackhole codegen requires a single local bridge shape per buffer; "
          << binding.buffer_name;
    }
  };

  for (const ffi::Any& item_any : tt_program_projection::GetExecutableArrayField(
           f, "Blackhole codegen", tt_program_projection::executable_key::kBufferDistributionPlans)) {
    auto item = item_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (item.empty() || !item.Get("logical_shape")) {
      continue;
    }
    ingest_spec(item);
  }
}

void CodeGenBlackhole::LoadAccessorOffsets(const tvm::tir::PrimFunc& f) {
  accessor_compile_time_offset_by_buffer_.clear();
  for (const ffi::Any& segment_any : tt_program_projection::GetExecutableArrayField(
           f, "Blackhole codegen", tt_program_projection::executable_key::kSegmentPlan)) {
    auto segment = segment_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (segment.empty()) {
      continue;
    }
    Array<ffi::Any> accessors;
    if (auto value = segment.Get(ffi::String("accessors"))) {
      accessors = Downcast<Array<ffi::Any>>(value.value());
    }
    for (const ffi::Any& accessor_any : accessors) {
      auto accessor = accessor_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (accessor.empty()) {
        continue;
      }
      std::string buffer;
      if (auto value = accessor.Get(ffi::String("buffer"))) {
        buffer = Downcast<ffi::String>(value.value());
      }
      if (buffer.empty()) {
        continue;
      }
      int64_t offset = -1;
      if (auto value = accessor.Get(ffi::String("compile_time_arg_offset"))) {
        offset = Downcast<Integer>(value.value()).IntValue();
      }
      ICHECK_GE(offset, 0)
          << "Blackhole codegen requires accessor compile_time_arg_offset for buffer "
          << buffer;
      auto [it, inserted] =
          accessor_compile_time_offset_by_buffer_.emplace(buffer, static_cast<int>(offset));
      ICHECK(inserted || it->second == offset)
          << "Blackhole codegen cannot disambiguate multiple compile-time accessor offsets for "
          << "buffer " << buffer << ": " << it->second << " vs " << offset;
    }
  }
}

const CodeGenBlackhole::LogicalTileLayoutBinding* CodeGenBlackhole::FindLogicalTileLayoutBinding(
    const tvm::tir::VarNode* var) const {
  if (var == nullptr) {
    return nullptr;
  }
  auto it = logical_tile_layout_bindings_by_buffer_name_.find(var->name_hint);
  if (it == logical_tile_layout_bindings_by_buffer_name_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool CodeGenBlackhole::LogicalTileLayoutRequiresGenericBridge(
    const LogicalTileLayoutBinding& binding) const {
  return !binding.inverse_logical_index_exprs.empty() && !binding.local_shape.empty();
}

void CodeGenBlackhole::LoadCBConfigMetadata(const tvm::tir::PrimFunc &f) {
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_data_format_by_id_.clear();
  cb_id_by_requirement_index_.clear();
  cb_num_pages_by_requirement_index_.clear();
  cb_initial_reserve_pages_by_requirement_index_.clear();
  local_non_input_cb_ids_.clear();
  emitted_cb_front_pages_.clear();
  emitted_cb_consumed_front_pages_.clear();
  active_cb_allocation_reserved_pages_.clear();
  auto cb_configs = GetCBConfigsForCodegen(f);
  if (!cb_configs.empty()) {
    for (const auto &item : cb_configs) {
      auto cb_info = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (cb_info.empty()) {
        continue;
      }
      int cb_id = -1;
      int page_size = 0;
      int num_pages = 1;
      int initial_reserve_pages = 0;
      if (auto v = cb_info.Get("cb_id")) {
        cb_id = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("page_size")) {
        page_size = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("num_pages")) {
        num_pages = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("initial_reserve_pages")) {
        initial_reserve_pages = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (cb_id >= 0) {
        cb_page_size_by_id_[cb_id] = page_size;
        cb_num_pages_by_id_[cb_id] = std::max(1, num_pages);
        cb_data_format_by_id_[cb_id] = MapGetString(cb_info, "data_format");
        const std::string role = MapGetString(cb_info, "role");
        if (role != "input") {
          local_non_input_cb_ids_.insert(cb_id);
        }
        if (auto requirement_indices = cb_info.Get("requirement_indices")) {
          for (const auto& requirement_index_any :
               Downcast<tvm::ffi::Array<tvm::ffi::Any>>(requirement_indices.value())) {
            const int requirement_index =
                Downcast<tvm::Integer>(requirement_index_any).IntValue();
            cb_id_by_requirement_index_[requirement_index] = cb_id;
            cb_num_pages_by_requirement_index_[requirement_index] = std::max(1, num_pages);
            if (initial_reserve_pages > 0) {
              cb_initial_reserve_pages_by_requirement_index_[requirement_index] =
                  std::max(1, initial_reserve_pages);
            }
          }
        }
      }
    }
  }
}

void CodeGenBlackhole::MaybeEmitConsumedCBPopBeforeReserve(int cb_id) {
  if (!local_non_input_cb_ids_.count(cb_id)) {
    return;
  }
  const int front_pages = emitted_cb_front_pages_.count(cb_id)
                              ? emitted_cb_front_pages_.at(cb_id)
                              : 0;
  const int consumed_pages = emitted_cb_consumed_front_pages_.count(cb_id)
                                 ? emitted_cb_consumed_front_pages_.at(cb_id)
                                 : 0;
  const int pop_pages = std::min(front_pages, consumed_pages);
  if (pop_pages <= 0) {
    return;
  }
  PrintIndent();
  stream << "cb_pop_front(" << cb_id << ", " << pop_pages << ");\n";
  emitted_cb_front_pages_[cb_id] = std::max(0, front_pages - pop_pages);
  emitted_cb_consumed_front_pages_[cb_id] =
      std::max(0, consumed_pages - pop_pages);
}

void CodeGenBlackhole::RecordEmittedCBQueueEvent(const std::string& kind,
                                                 int cb_id, int pages) {
  if (cb_id < 0 || pages <= 0 || !local_non_input_cb_ids_.count(cb_id)) {
    return;
  }
  if (kind == "push_back") {
    emitted_cb_front_pages_[cb_id] += pages;
    return;
  }
  if (kind == "wait_front") {
    const int front_pages = emitted_cb_front_pages_.count(cb_id)
                                ? emitted_cb_front_pages_.at(cb_id)
                                : 0;
    if (front_pages > 0) {
      emitted_cb_consumed_front_pages_[cb_id] = std::max(
          emitted_cb_consumed_front_pages_[cb_id],
          std::min(front_pages, pages));
    }
    return;
  }
  if (kind == "pop_front") {
    emitted_cb_front_pages_[cb_id] =
        std::max(0, emitted_cb_front_pages_[cb_id] - pages);
    emitted_cb_consumed_front_pages_[cb_id] =
        std::max(0, emitted_cb_consumed_front_pages_[cb_id] - pages);
  }
}

void CodeGenBlackhole::EmitRuntimeArgLoads(const tvm::tir::PrimFunc &f) {
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_identity_.clear();
  runtime_arg_vars_by_name_.clear();
  per_work_arg_bindings_by_identity_.clear();
  per_work_arg_bindings_.clear();
  LoadCBConfigMetadata(f);

  ffi::Array<ffi::Any> runtime_args = GetRuntimeArgsForCodegen(f);
  ffi::Array<ffi::Any> per_work_arg_specs = GetPerWorkArgSpecsForCodegen(f);
  for (const auto& item : per_work_arg_specs) {
    auto spec = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (spec.empty()) {
      continue;
    }
    std::string arg_kind;
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgKind)) {
      arg_kind = Downcast<tvm::ffi::String>(v.value());
    }
    PerWorkArgSpecBinding binding;
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgIdentity)) {
      binding.arg_identity = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kBuffer)) {
      binding.buffer = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kValueSource)) {
      binding.value_source = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue)) {
      binding.constant_value = Downcast<tvm::Integer>(v.value()).IntValue();
    }
    ICHECK(!binding.arg_identity.empty())
        << "Blackhole codegen requires per-work binding arg_identity";
    ICHECK(!binding.value_source.empty())
        << "Blackhole codegen requires per-work value_source for " << binding.arg_identity;
    per_work_arg_bindings_by_identity_[binding.arg_identity] = binding;
    per_work_arg_bindings_.push_back(std::move(binding));
  }
  ICHECK(!runtime_args.empty())
      << "Blackhole codegen requires executable kernel runtime args";
  if (logical_grid_x_ > 1 || logical_grid_y_ > 1 || logical_grid_z_ > 1) {
    for (const auto& item : runtime_args) {
      auto arg = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      std::string arg_kind;
      if (auto v = arg.Get("kind")) {
        arg_kind = Downcast<tvm::ffi::String>(v.value());
      }
      std::string arg_identity;
      if (auto v = arg.Get("identity")) {
        arg_identity = Downcast<tvm::ffi::String>(v.value());
      }
      bool requires_explicit_per_work_binding = false;
      if (auto v = arg.Get(
              ::tvm::tl::blackhole_runtime_arg_schema::kRequiresPerWorkBinding)) {
        requires_explicit_per_work_binding = Downcast<tvm::Bool>(v.value());
      }
      if (per_work_arg_bindings_by_identity_.count(arg_identity) != 0U) {
        ICHECK(requires_explicit_per_work_binding)
            << "Blackhole codegen requires runtime arg '" << arg_kind
            << "' identity '" << arg_identity
            << "' to declare requires_per_work_binding when a per-work "
            << "binding uses that identity";
      }
      if (!requires_explicit_per_work_binding) {
        continue;
      }
      ICHECK(!arg_identity.empty())
          << "Blackhole codegen requires runtime arg identity before per-work binding for "
          << arg_kind;
      ICHECK(per_work_arg_bindings_by_identity_.count(arg_identity))
          << "Blackhole codegen requires explicit per-work arg binding for runtime arg kind '"
          << arg_kind << "' identity '" << arg_identity
          << "' on multi-work kernels; codegen must not recover block/tile semantics "
          << "from work_linear_id or implicit runtime-arg inference";
    }
  }

  std::unordered_map<std::string, const tvm::tir::VarNode *> buffer_vars_by_name;
  auto record_handle_dtype = [&](const tvm::tir::VarNode* var,
                                 std::optional<DataType> dtype = std::nullopt) {
    if (var == nullptr) {
      return;
    }
    if (dtype.has_value()) {
      handle_data_type_[var] = dtype.value();
      return;
    }
    if (const auto* ptr = var->type_annotation.as<PointerTypeNode>()) {
      if (const auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        handle_data_type_[var] = prim->dtype;
      }
    }
  };
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      buffer_vars_by_name[param->name_hint] = param.get();
      record_handle_dtype(param.get());
    }
  }
  for (const auto &kv : f->buffer_map) {
    const auto &buffer = kv.second;
    buffer_vars_by_name[buffer->name] = buffer->data.get();
    record_handle_dtype(buffer->data.get(), buffer->dtype);
  }
  int arg_idx = 0;
  for (const auto &item : runtime_args) {
    auto arg_info = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (arg_info.empty()) {
      continue;
    }

    std::string arg_name = "arg" + std::to_string(arg_idx);
    std::string arg_kind;
    if (auto v = arg_info.Get("name")) {
      arg_name = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = arg_info.Get("kind")) {
      arg_kind = Downcast<tvm::ffi::String>(v.value());
    }

    stream << "  uint32_t " << arg_name << " = get_arg_val<uint32_t>(" << arg_idx << ");\n";
    runtime_arg_vars_by_name_[arg_name] = arg_name;
    if (auto v = arg_info.Get("identity")) {
      const std::string arg_identity = Downcast<tvm::ffi::String>(v.value());
      if (!arg_identity.empty() && !runtime_arg_vars_by_identity_.count(arg_identity)) {
        runtime_arg_vars_by_identity_[arg_identity] = arg_name;
      }
    }

    if (IsBufferAddressRuntimeArgKind(arg_kind)) {
      auto buffer_it = arg_info.Get("buffer");
      ICHECK(buffer_it.has_value())
          << "Blackhole codegen requires explicit buffer binding for runtime arg "
          << arg_name << " kind=" << arg_kind;
      const std::string bound_buffer_name = Downcast<tvm::ffi::String>(buffer_it.value());
      ICHECK(!bound_buffer_name.empty())
          << "Blackhole codegen requires non-empty buffer binding for runtime arg "
          << arg_name << " kind=" << arg_kind;
      auto [name_binding_it, name_inserted] =
          buffer_runtime_arg_map_by_name_.emplace(bound_buffer_name, arg_name);
      ICHECK(name_inserted || name_binding_it->second == arg_name)
          << "Blackhole codegen buffer " << bound_buffer_name
          << " has conflicting runtime arg name bindings " << name_binding_it->second
          << " and " << arg_name;
      auto var_it = buffer_vars_by_name.find(bound_buffer_name);
      if (var_it != buffer_vars_by_name.end()) {
        auto [var_binding_it, var_inserted] =
            buffer_runtime_arg_map_.emplace(var_it->second, arg_name);
        ICHECK(var_inserted || var_binding_it->second == arg_name)
            << "Blackhole codegen buffer " << bound_buffer_name
            << " has conflicting runtime arg bindings " << var_binding_it->second
            << " and " << arg_name;
      }
    }
    ++arg_idx;
  }
  stream << "\n";

  if (!cb_num_pages_by_id_.empty()) {
    stream << "\n";
  }
}

std::string CodeGenBlackhole::GetRuntimeArgVarForBuffer(
    const tvm::PrimExpr &buffer_expr, const char* preferred_kind) const {
  const auto *buffer_var = buffer_expr.as<tvm::tir::VarNode>();
  ICHECK(buffer_var) << "Expected buffer data var in runtime-arg-backed Blackhole builtin";
  auto it = buffer_runtime_arg_map_.find(buffer_var);
  if (it != buffer_runtime_arg_map_.end()) {
    return it->second;
  }
  auto by_name = buffer_runtime_arg_map_by_name_.find(buffer_var->name_hint);
  if (by_name != buffer_runtime_arg_map_by_name_.end()) {
    return by_name->second;
  }

  std::ostringstream available_names;
  bool first = true;
  for (const auto& kv : runtime_arg_vars_by_name_) {
    if (!first) {
      available_names << ", ";
    }
    available_names << kv.first;
    first = false;
  }
  std::ostringstream bound_buffers;
  first = true;
  for (const auto& kv : buffer_runtime_arg_map_by_name_) {
    if (!first) {
      bound_buffers << ", ";
    }
    bound_buffers << kv.first << "->" << kv.second;
    first = false;
  }
  ICHECK(false) << "Missing runtime arg binding for buffer var: " << buffer_var->name_hint
                << ", preferred_kind=" << (preferred_kind ? preferred_kind : "<none>")
                << ", available arg vars=[" << available_names.str() << "]"
                << ", bound buffers=[" << bound_buffers.str() << "]";
  return "";
}

std::optional<DataType> CodeGenBlackhole::TryResolveHandleDataType(
    const tvm::tir::VarNode* var) const {
  if (!var) {
    return std::nullopt;
  }
  if (auto it = handle_data_type_.find(var); it != handle_data_type_.end()) {
    return it->second;
  }
  if (const auto* ptr = var->type_annotation.as<PointerTypeNode>()) {
    if (const auto* prim = ptr->element_type.as<PrimTypeNode>()) {
      return prim->dtype;
    }
  }
  return std::nullopt;
}

DataType CodeGenBlackhole::ResolveHandleDataType(const tvm::tir::VarNode* var, const char* op_name,
                                                 const char* role) const {
  auto maybe_dtype = TryResolveHandleDataType(var);
  ICHECK(maybe_dtype.has_value()) << "Missing " << role << " handle dtype for " << op_name;
  return maybe_dtype.value();
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::VarNode* op, std::ostream& os) {
  if (auto it = var_idmap_.find(op); it != var_idmap_.end()) {
    os << it->second;
    return;
  }
  for (const auto& [known_var, known_name] : var_idmap_) {
    if (SameCodegenStorageVar(op, known_var)) {
      os << known_name;
      return;
    }
  }
  CodeGenC::VisitExpr_(op, os);
}

int CodeGenBlackhole::ResolveCBId(const tvm::PrimExpr &expr) const {
  const auto *cb_id_imm = expr.as<tvm::tir::IntImmNode>();
  ICHECK(cb_id_imm) << "Blackhole CB operations currently expect constant cb_id";
  const int cb_id = static_cast<int>(cb_id_imm->value);
  ICHECK_GE(cb_id, 0) << "Blackhole codegen expects final cb_id, but saw placeholder " << cb_id;
  return cb_id;
}

void CodeGenBlackhole::PrintResolvedCBId(const tvm::PrimExpr &expr, std::ostream &os) const {
  os << ResolveCBId(expr);
}

void CodeGenBlackhole::PrintPackReconfigDataFormatForCB(int cb_id, std::ostream& os) {
  need_compute_api_h_ = true;
  os << "pack_reconfig_data_format<true>(" << cb_id << ")";
}

int CodeGenBlackhole::GetCBPageSize(int cb_id) const {
  auto it = cb_page_size_by_id_.find(cb_id);
  ICHECK(it != cb_page_size_by_id_.end()) << "Missing CB page size for cb_id=" << cb_id;
  return it->second;
}

int CodeGenBlackhole::GetCBNumPages(int cb_id) const {
  auto it = cb_num_pages_by_id_.find(cb_id);
  ICHECK(it != cb_num_pages_by_id_.end()) << "Missing CB num_pages for cb_id=" << cb_id;
  return it->second;
}

std::string CodeGenBlackhole::GetCBDataFormat(int cb_id) const {
  auto it = cb_data_format_by_id_.find(cb_id);
  ICHECK(it != cb_data_format_by_id_.end()) << "Missing CB data_format for cb_id=" << cb_id;
  return it->second;
}

std::string CodeGenBlackhole::GetCBHeadVar(int cb_id) const {
  return "cb_head_" + std::to_string(cb_id);
}

std::string CodeGenBlackhole::GetCBTailVar(int cb_id) const {
  return "cb_tail_" + std::to_string(cb_id);
}

void CodeGenBlackhole::RegisterActiveCBWritePtrBinding(int cb_id, const std::string& var_name,
                                                       const std::string& type_name) {
  auto& bindings = active_cb_write_ptr_bindings_[cb_id];
  auto it = std::find_if(bindings.begin(), bindings.end(),
                         [&](const ActiveCBWritePtrBinding& binding) {
                           return binding.var_name == var_name;
                         });
  if (it == bindings.end()) {
    bindings.push_back(ActiveCBWritePtrBinding{var_name, type_name});
    return;
  }
  it->type_name = type_name;
}

void CodeGenBlackhole::UnregisterActiveCBWritePtrBinding(int cb_id,
                                                         const std::string& var_name) {
  auto it = active_cb_write_ptr_bindings_.find(cb_id);
  if (it == active_cb_write_ptr_bindings_.end()) {
    return;
  }
  auto& bindings = it->second;
  bindings.erase(std::remove_if(bindings.begin(), bindings.end(),
                                [&](const ActiveCBWritePtrBinding& binding) {
                                  return binding.var_name == var_name;
                                }),
                 bindings.end());
  if (bindings.empty()) {
    active_cb_write_ptr_bindings_.erase(it);
  }
}

void CodeGenBlackhole::EmitActiveCBWritePtrRefreshes(int cb_id) {
  auto it = active_cb_write_ptr_bindings_.find(cb_id);
  if (it == active_cb_write_ptr_bindings_.end()) {
    return;
  }
  for (const ActiveCBWritePtrBinding& binding : it->second) {
    PrintIndent();
    stream << binding.var_name << " = reinterpret_cast<" << binding.type_name
           << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << "));\n";
  }
}

// ============================================================================
// Visitor Implementation for TT-Metal Builtin Calls
// ============================================================================

void CodeGenBlackhole::VisitExpr_(const tvm::tir::CallNode *op,
                                  std::ostream &os) {
  if (op->op->IsInstance<OpNode>()) {
    Op call_op = Downcast<Op>(op->op);
    if (call_op->name == "tl.infinity") {
      std::ostringstream dtype_os;
      PrintType(op->dtype, dtype_os);
      os << "static_cast<" << dtype_os.str() << ">(1.0f / 0.0f)";
      return;
    }
    if (call_op->name == "tir.exp2") {
      std::ostringstream dtype_os;
      PrintType(op->dtype, dtype_os);
      os << "static_cast<" << dtype_os.str() << ">(tilelang_fast_exp2f(static_cast<float>(";
      PrintExpr(op->args[0], os);
      os << ")))";
      return;
    }
    if ((call_op->name == "tir.call_pure_extern" || call_op->name == "tir.call_extern") &&
        op->args.size() >= 2) {
      if (const auto* callee = op->args[0].as<tvm::tir::StringImmNode>()) {
        const std::string callee_name = callee->value;
        if (callee_name == "exp2f" || callee_name == "exp2") {
          std::ostringstream dtype_os;
          PrintType(op->dtype, dtype_os);
          os << "static_cast<" << dtype_os.str() << ">(tilelang_fast_exp2f(static_cast<float>(";
          PrintExpr(op->args[1], os);
          os << ")))";
          return;
        }
      }
    }
  }
  // Try to handle TT-Metal builtin calls
  if (HandleBlackholeBuiltin(op, os)) {
    return;
  }
  // Fall back to parent class for other calls
  CodeGenCHost::VisitExpr_(op, os);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::EvaluateNode *op) {
  // Handle TT-Metal builtin calls in Evaluate statements
  if (const auto *call = op->value.as<tvm::tir::CallNode>()) {
    std::ostringstream os;
    if (HandleBlackholeBuiltin(call, os)) {
      bool is_cb_reserve_back = call->op.same_as(tir::builtin::blackhole_cb_reserve_back());
      bool is_cb_push_back = call->op.same_as(tir::builtin::blackhole_cb_push_back());
      bool is_cb_wait_front = call->op.same_as(tir::builtin::blackhole_cb_wait_front());
      bool is_cb_pop_front = call->op.same_as(tir::builtin::blackhole_cb_pop_front());
      if (!is_cb_reserve_back && !is_cb_push_back &&
          !is_cb_wait_front && !is_cb_pop_front) {
        if (const auto* builtin = call->op.as<OpNode>()) {
          is_cb_reserve_back = builtin->name == "tl.blackhole.cb_reserve_back";
          is_cb_push_back = builtin->name == "tl.blackhole.cb_push_back";
          is_cb_wait_front = builtin->name == "tl.blackhole.cb_wait_front";
          is_cb_pop_front = builtin->name == "tl.blackhole.cb_pop_front";
        }
      }
      if (is_cb_reserve_back) {
        const int cb_id = ResolveCBId(call->args[0]);
        MaybeEmitConsumedCBPopBeforeReserve(cb_id);
        const auto* pages = call->args[1].as<IntImmNode>();
        auto reserved_it = active_cb_allocation_reserved_pages_.find(cb_id);
        if (pages != nullptr && reserved_it != active_cb_allocation_reserved_pages_.end() &&
            reserved_it->second >= pages->value) {
          return;
        }
      }
      // This is a Blackhole builtin - print it as a statement
      PrintIndent();
      stream << os.str() << ";\n";
      if ((is_cb_reserve_back || is_cb_push_back ||
           is_cb_wait_front || is_cb_pop_front) &&
          call->args.size() >= 2U) {
        const int cb_id = ResolveCBId(call->args[0]);
        const auto* pages = call->args[1].as<IntImmNode>();
        if (pages != nullptr && pages->value > 0) {
          if (is_cb_push_back) {
            RecordEmittedCBQueueEvent("push_back", cb_id,
                                      static_cast<int>(pages->value));
          } else if (is_cb_wait_front) {
            RecordEmittedCBQueueEvent("wait_front", cb_id,
                                      static_cast<int>(pages->value));
          } else if (is_cb_pop_front) {
            RecordEmittedCBQueueEvent("pop_front", cb_id,
                                      static_cast<int>(pages->value));
          }
        }
      }
      if (is_cb_push_back) {
        const int cb_id = ResolveCBId(call->args[0]);
        const auto* pages = call->args[1].as<IntImmNode>();
        auto reserved_it = active_cb_allocation_reserved_pages_.find(cb_id);
        if (pages != nullptr && pages->value > 0) {
          if (reserved_it != active_cb_allocation_reserved_pages_.end()) {
            reserved_it->second = std::max<int64_t>(0, reserved_it->second - pages->value);
            if (reserved_it->second == 0) {
              active_cb_allocation_reserved_pages_.erase(reserved_it);
            }
          }
        }
      }
      return;
    }
  }
  // Fall back to grandparent class (tvm::codegen::CodeGenC) for non-builtin expressions
  // We need to call the grandparent directly since CodeGenCHost doesn't override VisitStmt_ for EvaluateNode
  tvm::codegen::CodeGenC::VisitStmt_(op);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::ForNode *op) {
  if (core_type_ != CoreType::kTRISC && op->kind == tvm::tir::ForKind::kParallel) {
    const auto* extent = op->extent.as<tvm::tir::IntImmNode>();
    if (extent != nullptr && extent->value > 0 && !StmtUsesVar(op->body, op->loop_var.get())) {
      this->PrintStmt(op->body);
      return;
    }
  }

  std::string begin_str = PrintExpr(op->min);
  PrimExpr end = tvm::tir::is_zero(op->min)
                     ? op->extent
                     : arith::Analyzer().Simplify(op->min + op->extent);
  std::string end_str = PrintExpr(end);
  std::string step_str = op->step.has_value() ? PrintExpr(*op->step) : "";
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << begin_str << "; " << vid << " < " << end_str << "; ";
  if (step_str.empty()) {
    stream << "++" << vid;
  } else {
    stream << vid << " += " << step_str;
  }
  stream << ") {\n";

  std::optional<std::string> prev_var_id;
  if (auto it = var_idmap_.find(op->loop_var.get()); it != var_idmap_.end()) {
    prev_var_id = it->second;
  }
  var_idmap_[op->loop_var.get()] = vid;

  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);

  if (prev_var_id) {
    var_idmap_[op->loop_var.get()] = *prev_var_id;
  } else {
    var_idmap_.erase(op->loop_var.get());
  }

  PrintIndent();
  stream << "}\n";
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::IfThenElseNode *op) {
  if (IsNoOpStmt(op->then_case) &&
      (!op->else_case.defined() || IsNoOpStmt(op->else_case.value()))) {
    return;
  }
  arith::Analyzer analyzer;
  PrimExpr condition = analyzer.Simplify(op->condition);
  std::optional<bool> static_condition = TryEvalStaticBool(condition);
  if (static_condition && !static_condition.value()) {
    if (op->else_case.defined()) {
      this->PrintStmt(op->else_case.value());
    }
    return;
  }
  if (static_condition && static_condition.value()) {
    this->PrintStmt(op->then_case);
    return;
  }
  tvm::codegen::CodeGenC::VisitStmt_(op);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::AllocateNode *op) {
  std::string scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  RegisterHandleType(op->buffer_var.get(), op->dtype);

  const bool runtime_managed_storage =
      scope == "shared" || scope == "shared.dyn" || scope == "shared.barrier" ||
      scope.rfind("blackhole.cb", 0) == 0;
  const bool compute_local_fragment_storage =
      scope == "blackhole.acc" && core_type_ == CoreType::kTRISC;
  const std::optional<int> cb_requirement_index = CBRequirementIndexAnnotation(op);
  const bool cb_backed_accumulator =
      compute_local_fragment_storage && cb_requirement_index.has_value();

  if (runtime_managed_storage || (scope == "blackhole.acc" && !compute_local_fragment_storage)) {
    // Blackhole shared / CB allocations are runtime/device-managed
    // resources, not C arrays inside the generated kernel body.  The
    // blackhole.acc scope only materializes inside TRISC kernels, where it can
    // be either CB-backed accumulator storage or ordinary compute-local stack
    // storage depending on whether TT planning assigned a CB requirement.
    this->PrintStmt(op->body);
    return;
  }

  ICHECK(!tvm::tir::is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (cb_backed_accumulator) {
    const int requirement_index = *cb_requirement_index;
    auto cb_it = cb_id_by_requirement_index_.find(requirement_index);
    ICHECK(cb_it != cb_id_by_requirement_index_.end())
        << "Blackhole codegen requires a physical CB id for requirement index "
        << requirement_index;
    const int cb_id = cb_it->second;
    const int num_pages = cb_num_pages_by_requirement_index_.count(requirement_index)
                              ? cb_num_pages_by_requirement_index_.at(requirement_index)
                              : GetCBNumPages(cb_id);
    const int64_t dtype_bytes =
        std::max<int64_t>(1, (static_cast<int64_t>(op->dtype.bits()) *
                              static_cast<int64_t>(op->dtype.lanes()) + 7) / 8);
    const int64_t allocation_bytes =
        static_cast<int64_t>(op->ConstantAllocationSize()) * dtype_bytes;
    const int page_size = GetCBPageSize(cb_id);
    const int allocation_pages = std::max<int>(
        1, static_cast<int>((allocation_bytes + page_size - 1) / page_size));
    ICHECK_LE(allocation_pages, num_pages)
        << "Blackhole CB-backed allocation for requirement index " << requirement_index
        << " needs " << allocation_pages << " pages but CB " << cb_id << " has "
        << num_pages;
    auto reserve_it = cb_initial_reserve_pages_by_requirement_index_.find(requirement_index);
    ICHECK(reserve_it != cb_initial_reserve_pages_by_requirement_index_.end())
        << "Blackhole CB-backed allocation for requirement index " << requirement_index
        << " requires initial reserve pages";
    const int initial_reserve_pages =
        reserve_it->second;
    ICHECK_LE(allocation_pages, initial_reserve_pages)
        << "Blackhole CB-backed allocation for requirement index " << requirement_index
        << " needs " << allocation_pages << " pages but only reserves "
        << initial_reserve_pages;

    std::ostringstream dtype_os;
    PrintType(op->dtype, dtype_os);

    PrintIndent();
    stream << "cb_reserve_back(" << cb_id << ", " << initial_reserve_pages << ");\n";
    const int64_t reserved_pages_before_allocation =
        active_cb_allocation_reserved_pages_.count(cb_id)
            ? active_cb_allocation_reserved_pages_.at(cb_id)
            : int64_t{0};
    active_cb_allocation_reserved_pages_[cb_id] =
        reserved_pages_before_allocation + initial_reserve_pages;
    PrintIndent();
    stream << dtype_os.str() << "* " << vid << " = reinterpret_cast<" << dtype_os.str()
           << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << "));\n";

    std::optional<std::string> prev_var_id;
    if (auto it = var_idmap_.find(op->buffer_var.get()); it != var_idmap_.end()) {
      prev_var_id = it->second;
    }
    var_idmap_[op->buffer_var.get()] = vid;
    RegisterActiveCBWritePtrBinding(cb_id, vid, dtype_os.str());
    this->PrintStmt(op->body);
    UnregisterActiveCBWritePtrBinding(cb_id, vid);
    const int64_t reserved_pages_after_body =
        active_cb_allocation_reserved_pages_.count(cb_id)
            ? active_cb_allocation_reserved_pages_.at(cb_id)
            : int64_t{0};
    const int64_t unreleased_allocation_pages =
        std::max<int64_t>(0, reserved_pages_after_body - reserved_pages_before_allocation);
    if (unreleased_allocation_pages > 0) {
      PrintIndent();
      stream << "cb_push_back(" << cb_id << ", " << unreleased_allocation_pages << ");\n";
      PrintIndent();
      stream << "cb_pop_front(" << cb_id << ", " << unreleased_allocation_pages << ");\n";
    }
    if (reserved_pages_before_allocation > 0) {
      active_cb_allocation_reserved_pages_[cb_id] = reserved_pages_before_allocation;
    } else {
      active_cb_allocation_reserved_pages_.erase(cb_id);
    }
    if (prev_var_id) {
      var_idmap_[op->buffer_var.get()] = *prev_var_id;
    } else {
      var_idmap_.erase(op->buffer_var.get());
    }
    return;
  }

  PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  PrintStorageScope(scope, stream);
  PrintType(op->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  std::optional<std::string> prev_var_id;
  if (auto it = var_idmap_.find(op->buffer_var.get()); it != var_idmap_.end()) {
    prev_var_id = it->second;
  }
  var_idmap_[op->buffer_var.get()] = vid;
  this->PrintStmt(op->body);
  if (prev_var_id) {
    var_idmap_[op->buffer_var.get()] = *prev_var_id;
  } else {
    var_idmap_.erase(op->buffer_var.get());
  }
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::BufferStoreNode* op) {
  if (core_type_ != CoreType::kTRISC && op->buffer.defined() &&
      std::string(op->buffer.scope()) == "blackhole.acc") {
    return;
  }
  tvm::codegen::CodeGenC::VisitStmt_(op);
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::FloorDivNode *op,
                                   std::ostream &os) {
  // FloorDiv is not implemented in base CodeGenC
  // For Blackhole, we can implement it as regular division for positive integers
  // Or use a more complex expression: ((a >= 0 ? a : a - b + 1) / b)
  // For simplicity, we use regular division assuming positive values
  // TODO: Add proper floor div handling for negative values if needed
  os << "(";
  VisitExpr(op->a, os);
  os << " / ";
  VisitExpr(op->b, os);
  os << ")";
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::FloorModNode *op,
                                   std::ostream &os) {
  // FloorMod is not implemented in base CodeGenC
  // For Blackhole, implement as regular modulo for positive integers
  // TODO: Add proper floor mod handling for negative values if needed
  os << "(";
  VisitExpr(op->a, os);
  os << " % ";
  VisitExpr(op->b, os);
  os << ")";
}

void CodeGenBlackhole::BindThreadIndex(const tvm::tir::IterVar &iv) {
  // For Blackhole, we need to handle thread/block indices differently than CUDA
  // Blackhole uses a different parallelism model based on Tensix cores

  if (var_idmap_.count(iv->var.get())) {
    return;
  }

  std::string thread_tag = iv->thread_tag;
  auto runtime_arg_for_binding = [&](const PerWorkArgSpecBinding& binding)
      -> std::optional<std::string> {
    auto it = runtime_arg_vars_by_identity_.find(binding.arg_identity);
    if (it == runtime_arg_vars_by_identity_.end()) {
      return std::nullopt;
    }
    return it->second;
  };
  const bool row_major_grid = linearization_ == "row_major" && logical_grid_x_ > 0;
  auto resolve_explicit_axis = [&](bool want_x) -> std::optional<std::string> {
    for (const auto& binding : per_work_arg_bindings_) {
      auto arg_var = runtime_arg_for_binding(binding);
      if (!arg_var.has_value()) {
        continue;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
        if (want_x) {
          if (row_major_grid) {
            return "(" + arg_var.value() + " % " + std::to_string(logical_grid_x_) + ")";
          }
          return arg_var;
        }
        if (row_major_grid) {
          std::string y_expr =
              "(" + arg_var.value() + " / " + std::to_string(logical_grid_x_) + ")";
          if (logical_grid_z_ > 1 && logical_grid_y_ > 0) {
            y_expr = "(" + y_expr + " % " + std::to_string(logical_grid_y_) + ")";
          }
          return y_expr;
        }
        return std::string("0 /* explicit_linear_work_binding_y */");
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockXYLinear) {
        if (want_x) {
          if (row_major_grid) {
            return "(" + arg_var.value() + " % " + std::to_string(logical_grid_x_) + ")";
          }
          return arg_var;
        }
        if (row_major_grid) {
          std::string y_expr =
              "(" + arg_var.value() + " / " + std::to_string(logical_grid_x_) + ")";
          if (logical_grid_y_ > 0) {
            y_expr = "(" + y_expr + " % " + std::to_string(logical_grid_y_) + ")";
          }
          return y_expr;
        }
        return std::string("0 /* explicit_xy_work_binding_y */");
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockYXLinear) {
        if (want_x) {
          if (logical_grid_y_ > 0) {
            return "(" + arg_var.value() + " / " + std::to_string(logical_grid_y_) + ")";
          }
          return arg_var;
        }
        if (logical_grid_y_ > 0) {
          return "(" + arg_var.value() + " % " + std::to_string(logical_grid_y_) + ")";
        }
        return std::string("0 /* explicit_yx_work_binding_y */");
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockX) {
        if (want_x) {
          return arg_var;
        }
        continue;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockY) {
        if (!want_x) {
          return arg_var;
        }
        continue;
      }
    }
    return std::nullopt;
  };
  const auto explicit_block_x = resolve_explicit_axis(/*want_x=*/true);
  const auto explicit_block_y = resolve_explicit_axis(/*want_x=*/false);
  auto resolve_explicit_z = [&]() -> std::optional<std::string> {
    for (const auto& binding : per_work_arg_bindings_) {
      auto arg_var = runtime_arg_for_binding(binding);
      if (!arg_var.has_value()) {
        continue;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockZ) {
        return arg_var;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
        if (row_major_grid && logical_grid_z_ > 1) {
          const int xy_work = std::max(1, logical_grid_x_ * logical_grid_y_);
          return "(" + arg_var.value() + " / " + std::to_string(xy_work) + ")";
        }
        continue;
      }
    }
    return std::nullopt;
  };
  const auto explicit_block_z = resolve_explicit_z();
  const bool has_explicit_work_binding =
      explicit_block_x.has_value() || explicit_block_y.has_value() ||
      explicit_block_z.has_value();

  // Map CUDA-style thread indices to Blackhole concepts
  // For staged single-core execution, block coordinates must come from the
  // strongest explicit work contract available. If the ABI already carries a
  // buffer-specific runtime binding, consume that descriptor directly.
  if (thread_tag == "blockIdx.x") {
    if (explicit_block_x.has_value()) {
      var_idmap_[iv->var.get()] = explicit_block_x.value();
    } else if (has_explicit_work_binding) {
      var_idmap_[iv->var.get()] = "0 /* explicit_work_binding_x */";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_x */";
    }
  } else if (thread_tag == "blockIdx.y") {
    if (explicit_block_y.has_value()) {
      var_idmap_[iv->var.get()] = explicit_block_y.value();
    } else if (has_explicit_work_binding) {
      var_idmap_[iv->var.get()] = "0 /* explicit_work_binding_y */";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_y */";
    }
  } else if (thread_tag == "blockIdx.z") {
    if (explicit_block_z.has_value()) {
      var_idmap_[iv->var.get()] = explicit_block_z.value();
    } else if (logical_grid_z_ > 1) {
      std::optional<std::string> linear_work_arg;
      for (const auto& binding : per_work_arg_bindings_) {
        if (binding.value_source !=
            ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
          continue;
        }
        linear_work_arg = runtime_arg_for_binding(binding);
        if (linear_work_arg.has_value()) {
          break;
        }
      }
      if (linear_work_arg.has_value() && row_major_grid) {
        const int xy_work = std::max(1, logical_grid_x_ * logical_grid_y_);
        var_idmap_[iv->var.get()] =
            "(" + linear_work_arg.value() + " / " + std::to_string(xy_work) + ")";
      } else {
        var_idmap_[iv->var.get()] = "0 /* core_z */";
      }
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_z */";
    }
  } else if (thread_tag == "threadIdx.x") {
    // For Blackhole, threadIdx.x could map to worker threads within a core
    // For now, use the variable name directly
    var_idmap_[iv->var.get()] = iv->var->name_hint;
    thread_idx_x_expr_ = iv->var->name_hint;
  } else if (thread_tag == "threadIdx.y") {
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  } else if (thread_tag == "threadIdx.z") {
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  } else {
    // Unknown thread tag - use the variable name
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  }
}

void CodeGenBlackhole::PrintStorageScope(const std::string &scope,
                                          std::ostream &os) {
  // Blackhole uses different memory model than CUDA
  // - "global" -> DRAM (no keyword needed)
  // - "shared" / "shared.dyn" -> Circular Buffer (CB) - handled separately
  // - "blackhole.cb.*" -> runtime/device-managed resource
  // - "blackhole.acc" -> compute-local stack storage emitted in TRISC kernels
  // - "local" -> Local registers (no keyword needed)
  // - "warp" / "warp::sync" -> Not applicable for Blackhole

  if (scope == "shared" || scope == "shared.dyn" ||
      scope == "shared.barrier" ||
      scope.rfind("blackhole.cb", 0) == 0) {
    // For Blackhole, shared memory is allocated as Circular Buffers
    // and emitted outside the generated C body.
    os << "/* blackhole managed resource */ ";
  } else if (scope == "local") {
    // Local scope doesn't need a qualifier in C++
    // Variables are local by default
  } else if (scope == "global") {
    // Global memory - no qualifier needed
  } else if (scope.find("warp") == 0) {
    // Warp scope not applicable for Blackhole
    // Blackhole doesn't have warps like CUDA
  } else {
    // Unknown scope - add a comment
    os << "/* scope: " << scope << " */ ";
  }
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::AttrStmtNode *op) {
  // Handle Blackhole-specific attribute statements
  // For TT-Metal kernels, we handle specific attr_keys differently

  if (op->attr_key == tir::attr::thread_extent) {
    // For thread_extent, we need to bind the thread index variable
    // This is similar to CUDA but maps to Blackhole core/thread model
    auto iv = Downcast<tvm::tir::IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      const std::string thread_tag = iv->thread_tag;
      const bool is_thread_idx = thread_tag.rfind("threadIdx.", 0) == 0;
      if (is_thread_idx) {
        const bool thread_var_used =
            StmtUsesVarInEmittedBody(op->body, iv->var.get(), core_type_);
        std::optional<std::string> prev_var_id;
        if (auto it = var_idmap_.find(iv->var.get()); it != var_idmap_.end()) {
          prev_var_id = it->second;
        }
        auto restore_thread_var = [&]() {
          if (prev_var_id) {
            var_idmap_[iv->var.get()] = *prev_var_id;
          } else {
            var_idmap_.erase(iv->var.get());
          }
        };
        std::vector<std::pair<const tvm::tir::VarNode*, std::optional<std::string>>>
            nested_thread_prev_ids;
        auto restore_nested_thread_vars = [&]() {
          for (auto it = nested_thread_prev_ids.rbegin(); it != nested_thread_prev_ids.rend();
               ++it) {
            if (it->second) {
              var_idmap_[it->first] = *(it->second);
            } else {
              var_idmap_.erase(it->first);
            }
          }
        };
        auto emit_with_thread_binding = [&](const std::string& binding,
                                            const tvm::tir::Stmt& stmt) {
          const bool binds_thread_idx_x = thread_tag == "threadIdx.x";
          const std::string previous_thread_idx_x_expr = thread_idx_x_expr_;
          var_idmap_[iv->var.get()] = binding;
          if (binds_thread_idx_x) {
            thread_idx_x_expr_ = binding;
          }
          this->VisitStmt(stmt);
          if (binds_thread_idx_x) {
            thread_idx_x_expr_ = previous_thread_idx_x_expr;
          }
        };
        tvm::tir::Stmt partition_body = op->body;
        while (const auto* nested_attr = partition_body.as<tvm::tir::AttrStmtNode>()) {
          if (nested_attr->attr_key != tir::attr::thread_extent) {
            break;
          }
          auto nested_iv = Downcast<tvm::tir::IterVar>(nested_attr->node);
          const std::string nested_tag = nested_iv->thread_tag;
          const bool nested_is_unit_thread =
              nested_tag.rfind("threadIdx.", 0) == 0 && tir::is_one(nested_attr->value);
          if (!nested_is_unit_thread) {
            break;
          }
          std::optional<std::string> nested_prev_var_id;
          if (auto it = var_idmap_.find(nested_iv->var.get()); it != var_idmap_.end()) {
            nested_prev_var_id = it->second;
          }
          nested_thread_prev_ids.push_back({nested_iv->var.get(), nested_prev_var_id});
          var_idmap_[nested_iv->var.get()] = "0";
          partition_body = nested_attr->body;
        }
        if (!thread_var_used || tir::is_one(op->value)) {
          emit_with_thread_binding("0", partition_body);
          restore_nested_thread_vars();
          restore_thread_var();
          return;
        } else if (ThreadUsesOnlySurvivorPopGuards(partition_body, iv->var.get(), op->value)) {
          arith::Analyzer analyzer;
          const PrimExpr survivor_index =
              analyzer.Simplify(op->value - IntImm(iv->var.dtype(), 1));
          const tvm::tir::Stmt survivor_body =
              UnwrapThreadSurvivorPopGuards(partition_body, iv->var.get(), op->value);
          emit_with_thread_binding(PrintExpr(survivor_index), survivor_body);
          restore_nested_thread_vars();
          restore_thread_var();
          return;
        } else {
          auto emit_thread_loop = [&](const std::vector<tvm::tir::Stmt>& loop_body_stmts) {
            if (loop_body_stmts.empty()) {
              return;
            }
            std::ostringstream dtype_os;
            PrintType(iv->var.dtype(), dtype_os);
            const std::string loop_var = iv->var->name_hint;
            const bool binds_thread_idx_x = thread_tag == "threadIdx.x";
            const std::string previous_thread_idx_x_expr = thread_idx_x_expr_;
            var_idmap_[iv->var.get()] = loop_var;
            if (binds_thread_idx_x) {
              thread_idx_x_expr_ = loop_var;
            }
            PrintIndent();
            stream << "for (" << dtype_os.str() << " " << loop_var << " = 0; " << loop_var
                   << " < ";
            PrintExpr(op->value, stream);
            stream << "; ++" << loop_var << ") {\n";
            int scope_id = BeginScope();
            tvm::tir::Stmt loop_body =
                loop_body_stmts.size() == 1 ? loop_body_stmts.front()
                                            : tvm::tir::SeqStmt::Flatten(loop_body_stmts);
            this->VisitStmt(loop_body);
            if (binds_thread_idx_x) {
              thread_idx_x_expr_ = previous_thread_idx_x_expr;
            }
            EndScope(scope_id);
            PrintIndent();
            stream << "}\n";
          };

          auto emit_pieces = [&](const std::vector<ThreadEmissionPiece>& pieces) {
            std::vector<tvm::tir::Stmt> pending_threaded_stmts;
            for (const auto& piece : pieces) {
              if (piece.uses_thread_var) {
                pending_threaded_stmts.push_back(piece.stmt);
                continue;
              }
              emit_thread_loop(pending_threaded_stmts);
              pending_threaded_stmts.clear();
              emit_with_thread_binding("0", piece.stmt);
            }
            emit_thread_loop(pending_threaded_stmts);
          };

          std::function<void(const tvm::tir::Stmt&)> emit_split_stmt;
          emit_split_stmt = [&](const tvm::tir::Stmt& stmt) {
            if (const auto* seq = stmt.as<tvm::tir::SeqStmtNode>()) {
              for (const tvm::tir::Stmt& child : seq->seq) {
                emit_split_stmt(child);
              }
              return;
            }
            if (const auto* attr = stmt.as<tvm::tir::AttrStmtNode>()) {
              emit_split_stmt(attr->body);
              return;
            }
            if (const auto* decl = stmt.as<tvm::tir::DeclBufferNode>()) {
              emit_split_stmt(decl->body);
              return;
            }
            if (const auto* for_node = stmt.as<tvm::tir::ForNode>()) {
              std::string begin_str = PrintExpr(for_node->min);
              PrimExpr end = tvm::tir::is_zero(for_node->min)
                                 ? for_node->extent
                                 : arith::Analyzer().Simplify(for_node->min + for_node->extent);
              std::string end_str = PrintExpr(end);
              std::string step_str =
                  for_node->step.has_value() ? PrintExpr(*for_node->step) : "";
              PrintIndent();
              std::string vid = AllocVarID(for_node->loop_var.get());
              stream << "for (";
              PrintType(for_node->loop_var.dtype(), stream);
              stream << ' ' << vid << " = " << begin_str << "; " << vid << " < " << end_str
                     << "; ";
              if (step_str.empty()) {
                stream << "++" << vid;
              } else {
                stream << vid << " += " << step_str;
              }
              stream << ") {\n";

              std::optional<std::string> prev_var_id;
              if (auto it = var_idmap_.find(for_node->loop_var.get()); it != var_idmap_.end()) {
                prev_var_id = it->second;
              }
              var_idmap_[for_node->loop_var.get()] = vid;

              int for_scope = BeginScope();
              emit_split_stmt(for_node->body);
              this->EndScope(for_scope);

              if (prev_var_id) {
                var_idmap_[for_node->loop_var.get()] = *prev_var_id;
              } else {
                var_idmap_.erase(for_node->loop_var.get());
              }

              PrintIndent();
              stream << "}\n";
              return;
            }
            if (const auto* alloc = stmt.as<tvm::tir::AllocateNode>()) {
              std::string scope = GetPtrStorageScope(alloc->buffer_var);
              alloc_storage_scope_[alloc->buffer_var.get()] = scope;
              RegisterHandleType(alloc->buffer_var.get(), alloc->dtype);

              const bool runtime_managed_storage =
                  scope == "shared" || scope == "shared.dyn" ||
                  scope == "shared.barrier" || scope.rfind("blackhole.cb", 0) == 0;
              const bool compute_local_fragment_storage =
                  scope == "blackhole.acc" && core_type_ == CoreType::kTRISC;
              const std::optional<int> cb_requirement_index =
                  CBRequirementIndexAnnotation(alloc);
              const bool cb_backed_accumulator =
                  compute_local_fragment_storage && cb_requirement_index.has_value();
              if (runtime_managed_storage ||
                  (scope == "blackhole.acc" && !compute_local_fragment_storage) ||
                  cb_backed_accumulator) {
                emit_with_thread_binding("0", stmt);
                return;
              }

              ICHECK(!tvm::tir::is_zero(alloc->condition));
              const size_t constant_size = alloc->ConstantAllocationSize();
              ICHECK_GT(constant_size, 0)
                  << "Can only handle constant size stack allocation for now";
              std::string vid = AllocVarID(alloc->buffer_var.get());
              PrintIndent();
              PrintStorageScope(scope, stream);
              PrintType(alloc->dtype, stream);
              stream << ' ' << vid << '[' << constant_size << "];\n";

              std::optional<std::string> prev_var_id;
              if (auto it = var_idmap_.find(alloc->buffer_var.get());
                  it != var_idmap_.end()) {
                prev_var_id = it->second;
              }
              var_idmap_[alloc->buffer_var.get()] = vid;
              emit_split_stmt(alloc->body);
              if (prev_var_id) {
                var_idmap_[alloc->buffer_var.get()] = *prev_var_id;
              } else {
                var_idmap_.erase(alloc->buffer_var.get());
              }
              return;
            }

            const std::vector<ThreadEmissionPiece> pieces =
                BuildThreadEmissionPieces(stmt, iv->var.get(), core_type_);
            const bool has_threaded_piece =
                std::any_of(pieces.begin(), pieces.end(), [](const ThreadEmissionPiece& piece) {
                  return piece.uses_thread_var;
                });
            if (!has_threaded_piece) {
              emit_with_thread_binding("0", stmt);
              return;
            }
            emit_pieces(pieces);
          };

          emit_split_stmt(partition_body);
          restore_nested_thread_vars();
          restore_thread_var();
          return;
        }
      }
    }
    if (!var_idmap_.count(iv->var.get())) {
      BindThreadIndex(iv);
    }
    this->VisitStmt(op->body);
  } else if (op->attr_key == tir::attr::virtual_thread ||
             op->attr_key == tir::attr::coproc_scope ||
             op->attr_key == tir::attr::coproc_uop_scope) {
    // For virtual_thread and coproc attributes, just visit the body
    // These are CUDA-specific constructs that don't directly apply to Blackhole
    this->VisitStmt(op->body);
  } else if (op->attr_key == tir::attr::realize_scope ||
             op->attr_key == tir::attr::storage_alignment) {
    // Storage scope/alignment annotations - just visit the body
    // The Blackhole CB (circular buffer) system handles this differently
    this->VisitStmt(op->body);
  } else if (op->attr_key == "pragma_unroll") {
    // Unroll pragma - just visit the body
    // Blackhole compiler handles unrolling via TT-Metal
    this->VisitStmt(op->body);
  } else if (op->attr_key == "pragma") {
    // Generic pragma - skip for now
    this->VisitStmt(op->body);
  } else {
    // For all other attributes, fall back to parent class
    CodeGenCHost::VisitStmt_(op);
  }
}

bool CodeGenBlackhole::HandleBlackholeBuiltin(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  auto maybe_builtin_name = BlackholeBuiltinName(op);
  if (!maybe_builtin_name.has_value()) return false;
  const std::string builtin_name = maybe_builtin_name.value();
  const std::string op_name = "tl.blackhole." + builtin_name;
  auto skip_current_segment_builtin = [&]() {
    os << "/* skipped " << op_name << " outside its TT-Metal segment */";
  };
  if (core_type_ != CoreType::kTRISC &&
      IsTRISCOnlyBlackholeBuiltin(builtin_name)) {
    skip_current_segment_builtin();
    return true;
  }
  if (core_type_ == CoreType::kTRISC &&
      IsDataMovementOnlyBlackholeBuiltin(builtin_name)) {
    skip_current_segment_builtin();
    return true;
  }

  // Handle each builtin type
  if (builtin_name == "cb_reserve_back") {
    PrintCBReserveBack(op, os);
    return true;
  } else if (builtin_name == "cb_push_back") {
    PrintCBPushBack(op, os);
    return true;
  } else if (builtin_name == "cb_wait_front") {
    PrintCBWaitFront(op, os);
    return true;
  } else if (builtin_name == "cb_pop_front") {
    PrintCBPopFront(op, os);
    return true;
  } else if (builtin_name == "noc_async_read") {
    PrintNOCAsyncRead(op, os);
    return true;
  } else if (builtin_name == "noc_async_write") {
    PrintNOCAsyncWrite(op, os);
    return true;
  } else if (builtin_name == "noc_async_read_barrier") {
    PrintNOCReadBarrier(os);
    return true;
  } else if (builtin_name == "noc_async_write_barrier") {
    PrintNOCWriteBarrier(os);
    return true;
  } else if (builtin_name == "read_tile_to_cb") {
    PrintReadTileToCB(op, os);
    return true;
  } else if (builtin_name == "read_page_to_cb") {
    PrintReadPageToCB(op, os);
    return true;
  } else if (builtin_name == "read_bcast_cols_to_cb") {
    PrintReadBcastColsToCB(op, os);
    return true;
  } else if (builtin_name == "copy_cb_page") {
    PrintCopyCBPage(op, os);
    return true;
  } else if (builtin_name == "write_tile_from_cb") {
    PrintWriteTileFromCB(op, os);
    return true;
  } else if (builtin_name == "write_page_from_cb") {
    PrintWritePageFromCB(op, os);
    return true;
  } else if (builtin_name == "zero_cb_page") {
    PrintZeroCBPage(op, os);
    return true;
  } else if (builtin_name == "guard_mask_to_cb") {
    PrintGuardMaskToCB(op, os);
    return true;
  } else if (builtin_name == "get_semaphore") {
    PrintGetSemaphore(op, os);
    return true;
  } else if (builtin_name == "runtime_arg_u32") {
    PrintRuntimeArgU32(op, os);
    return true;
  } else if (builtin_name == "semaphore_wait") {
    PrintSemaphoreWait(op, os);
    return true;
  } else if (builtin_name == "semaphore_set") {
    PrintSemaphoreSet(op, os);
    return true;
  } else if (builtin_name == "semaphore_inc_remote") {
    PrintSemaphoreIncRemote(op, os);
    return true;
  } else if (builtin_name == "semaphore_set_remote") {
    PrintSemaphoreSetRemote(op, os);
    return true;
  } else if (builtin_name == "mm_init") {
    PrintMMInit(op, os);
    return true;
  } else if (builtin_name == "reconfig_data_format") {
    PrintReconfigDataFormat(op, os);
    return true;
  } else if (builtin_name == "mm_init_short") {
    PrintMMInitShort(op, os);
    return true;
  } else if (builtin_name == "mm_init_short_with_dt") {
    PrintMMInitShortWithDT(op, os);
    return true;
  } else if (builtin_name == "matmul_tiles") {
    PrintMatmulTiles(op, os);
    return true;
  } else if (builtin_name == "tile_regs_acquire") {
    PrintTileRegsAcquire(os);
    return true;
  } else if (builtin_name == "tile_regs_commit") {
    PrintTileRegsCommit(os);
    return true;
  } else if (builtin_name == "tile_regs_wait") {
    PrintTileRegsWait(os);
    return true;
  } else if (builtin_name == "tile_regs_release") {
    PrintTileRegsRelease(os);
    return true;
  } else if (builtin_name == "pack_tile") {
    PrintPackTile(op, os);
    return true;
  } else if (builtin_name == "pack_reconfig_data_format") {
    PrintPackReconfigDataFormat(op, os);
    return true;
  } else if (builtin_name == "copy_tile_to_dst_init_short") {
    PrintCopyTileToDstInitShort(op, os);
    return true;
  } else if (builtin_name == "copy_tile_to_dst_init_short_with_dt") {
    PrintCopyTileToDstInitShortWithDT(op, os);
    return true;
  } else if (builtin_name == "copy_tile") {
    PrintCopyTile(op, os);
    return true;
  } else if (builtin_name == "binary_op_init_common") {
    PrintBinaryOpInitCommon(op, os);
    return true;
  } else if (builtin_name == "unary_op_init_common") {
    PrintUnaryOpInitCommon(op, os);
    return true;
  } else if (builtin_name == "add_tiles_init") {
    PrintAddTilesInit(op, os);
    return true;
  } else if (builtin_name == "add_tiles") {
    PrintAddTiles(op, os);
    return true;
  } else if (builtin_name == "sub_tiles_init") {
    PrintSubTilesInit(op, os);
    return true;
  } else if (builtin_name == "sub_tiles") {
    PrintSubTiles(op, os);
    return true;
  } else if (builtin_name == "add_bcast_rows_init_short") {
    PrintAddBcastRowsInitShort(op, os);
    return true;
  } else if (builtin_name == "add_bcast_cols_init_short") {
    PrintAddBcastColsInitShort(op, os);
    return true;
  } else if (builtin_name == "add_tiles_bcast_rows") {
    PrintAddTilesBcastRows(op, os);
    return true;
  } else if (builtin_name == "add_tiles_bcast_cols") {
    PrintAddTilesBcastCols(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_init") {
    PrintMulTilesInit(op, os);
    return true;
  } else if (builtin_name == "mul_tiles") {
    PrintMulTiles(op, os);
    return true;
  } else if (builtin_name == "mul_bcast_rows_init_short") {
    PrintMulBcastRowsInitShort(op, os);
    return true;
  } else if (builtin_name == "mul_bcast_cols_init_short") {
    PrintMulBcastColsInitShort(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_bcast_rows") {
    PrintMulTilesBcastRows(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_bcast_cols") {
    PrintMulTilesBcastCols(op, os);
    return true;
  } else if (builtin_name == "reduce_init") {
    PrintReduceInit(op, os);
    return true;
  } else if (builtin_name == "reduce_tile") {
    PrintReduceTile(op, os);
    return true;
  } else if (builtin_name == "reduce_uninit") {
    PrintReduceUninit(op, os);
    return true;
  } else if (builtin_name == "binary_max_tile_init") {
    PrintBinaryMaxTileInit(op, os);
    return true;
  } else if (builtin_name == "binary_max_tile") {
    PrintBinaryMaxTile(op, os);
    return true;
  } else if (builtin_name == "div_binary_tile_init") {
    PrintDivBinaryTileInit(op, os);
    return true;
  } else if (builtin_name == "div_binary_tile") {
    PrintDivBinaryTile(op, os);
    return true;
  } else if (builtin_name == "exp_tile_init") {
    PrintExpTileInit(op, os);
    return true;
  } else if (builtin_name == "exp_tile") {
    PrintExpTile(op, os);
    return true;
  } else if (builtin_name == "exp2_tile_init") {
    PrintExp2TileInit(op, os);
    return true;
  } else if (builtin_name == "exp2_tile") {
    PrintExp2Tile(op, os);
    return true;
  } else if (builtin_name == "recip_tile_init") {
    PrintRecipTileInit(op, os);
    return true;
  } else if (builtin_name == "recip_tile") {
    PrintRecipTile(op, os);
    return true;
  } else if (builtin_name == "fill_fragment") {
    PrintFillFragment(op, os);
    return true;
  } else if (builtin_name == "add_fragment") {
    PrintAddFragment(op, os);
    return true;
  } else if (builtin_name == "add_fragment_from_cb_front") {
    PrintAddFragmentFromCBFront(op, os);
    return true;
  } else if (builtin_name == "pack_untilize_slice") {
    PrintPackUntilizeSlice(op, os);
    return true;
  } else if (builtin_name == "pack_untilize_tile") {
    PrintPackUntilizeTile(op, os);
    return true;
  } else if (builtin_name == "tilize_local_fragment_slice") {
    PrintTilizeLocalFragmentSlice(op, os);
    return true;
  } else if (builtin_name == "tilize_cast_fragment_slice") {
    PrintTilizeCastFragmentSlice(op, os);
    return true;
  } else if (builtin_name == "pack_fill_fragment_to_tiled_cb") {
    PrintPackFillFragmentToTiledCB(op, os);
    return true;
  } else if (builtin_name == "generate_reduce_scaler_to_cb") {
    PrintGenerateReduceScalerToCB(op, os);
    return true;
  } else if (builtin_name == "untilize_cb_front_tile") {
    PrintUntilizeCBFrontTile(op, os);
    return true;
  } else if (builtin_name == "untilize_cb_front_tile_fragment") {
    PrintUntilizeCBFrontTileFragment(op, os);
    return true;
  } else if (builtin_name == "cast_fragment_slice") {
    PrintCastFragmentSlice(op, os);
    return true;
  }

  return false;
}

// ============================================================================
// TT-Metal Builtin Print Functions
// ============================================================================

void CodeGenBlackhole::PrintCBReserveBack(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_reserve_back(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPushBack(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "cb_push_back(";
  os << cb_id;
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintCBWaitFront(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_wait_front(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPopFront(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "cb_pop_front(";
  os << cb_id;
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncRead(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncWrite(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCReadBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read_barrier()";
}

void CodeGenBlackhole::PrintNOCWriteBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write_barrier()";
}

namespace {

int ResolveCompileTimeAccessorOffset(const tvm::tir::CallNode* op,
                                     int arg_index,
                                     const char* builtin_name) {
  const auto* accessor_offset = op->args[arg_index].as<tvm::tir::IntImmNode>();
  ICHECK(accessor_offset)
      << "Blackhole codegen currently supports only compile-time-only accessor slots; "
      << builtin_name << " expects constant accessor compile-time offset";
  return static_cast<int>(accessor_offset->value);
}

void EmitTensorAccessorGenerator(std::ostream& os,
                                 const char* prefix,
                                 int accessor_offset,
                                 const std::string& addr_var,
                                 const std::string& size_expr = "") {
  os << "; constexpr auto " << prefix << "_accessor_args = TensorAccessorArgs<"
     << accessor_offset << ">(); const auto " << prefix << "_gen = TensorAccessor("
     << prefix << "_accessor_args, " << addr_var;
  if (!size_expr.empty()) {
    os << ", " << size_expr;
  }
  os << "); ";
}

}  // namespace

int CodeGenBlackhole::ResolveAccessorOffsetForBuffer(const tvm::PrimExpr& buffer_expr,
                                                     int tir_accessor_arg_index,
                                                     const tvm::tir::CallNode* op,
                                                     const char* builtin_name) const {
  const int tir_offset =
      ResolveCompileTimeAccessorOffset(op, tir_accessor_arg_index, builtin_name);
  const auto* buffer_var = buffer_expr.as<tvm::tir::VarNode>();
  if (buffer_var == nullptr) {
    return tir_offset;
  }
  auto it = accessor_compile_time_offset_by_buffer_.find(buffer_var->name_hint);
  if (it == accessor_compile_time_offset_by_buffer_.end()) {
    return tir_offset;
  }
  return it->second;
}

void CodeGenBlackhole::PrintReadTileToCB(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveAccessorOffsetForBuffer(op->args[0], /*tir_accessor_arg_index=*/4, op,
                                     "tl.blackhole.read_tile_to_cb");
  os << "{ ";
  os << "const uint32_t tile_index = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t tile_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ")";
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var, "tile_bytes");
  os << "noc_async_read_tile(tile_index, src_gen, cb_l1_addr); ";
  os << "noc_async_read_barrier(); }";
}

void CodeGenBlackhole::PrintWriteTileFromCB(const tvm::tir::CallNode *op,
                                            std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string dst_addr_var = GetRuntimeArgVarForBuffer(op->args[1], "output_buffer_addr");
  const int cb_id = ResolveCBId(op->args[0]);
  const int accessor_offset =
      ResolveAccessorOffsetForBuffer(op->args[1], /*tir_accessor_arg_index=*/4, op,
                                     "tl.blackhole.write_tile_from_cb");
  os << "{ ";
  os << "const uint32_t tile_index = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t tile_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ")";
  EmitTensorAccessorGenerator(os, "dst", accessor_offset, dst_addr_var, "tile_bytes");
  os << "noc_async_write_tile(tile_index, dst_gen, cb_l1_addr); ";
  os << "noc_async_write_barrier(); }";
}

void CodeGenBlackhole::PrintReadPageToCB(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveAccessorOffsetForBuffer(op->args[0], /*tir_accessor_arg_index=*/4, op,
                                     "tl.blackhole.read_page_to_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ") + ";
  PrintExpr(op->args[5], os);
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var);
  os << "const uint64_t src_noc_addr = src_gen.get_noc_addr(page_id); ";
  os << "noc_async_read(src_noc_addr, cb_l1_addr, page_bytes); }";
}

void CodeGenBlackhole::PrintReadBcastColsToCB(const tvm::tir::CallNode *op,
                                              std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveAccessorOffsetForBuffer(op->args[0], /*tir_accessor_arg_index=*/4, op,
                                     "tl.blackhole.read_bcast_cols_to_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t vector_len = ";
  PrintExpr(op->args[5], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << "); ";
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var);
  os << "const uint64_t src_noc_addr = src_gen.get_noc_addr(page_id); "
        "volatile uint16_t* dst_bits = reinterpret_cast<volatile uint16_t*>(cb_l1_addr); "
        "const uint32_t scratch_byte_offset = 2048u - page_bytes; "
        "const uint32_t scratch_l1_addr = cb_l1_addr + scratch_byte_offset; "
        "noc_async_read(src_noc_addr, scratch_l1_addr, page_bytes); "
        "noc_async_read_barrier(); "
        "const uint32_t scratch_element_offset = scratch_byte_offset / 2u; "
        "const uint32_t page_elements = page_bytes / 2u; "
        "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
        "constexpr uint32_t kTileCols = 32; "
        "const uint32_t rows = vector_len < 32u ? vector_len : 32u; "
        "for (uint32_t i = 0; i < 1024u; ++i) { "
        "if (i < scratch_element_offset || i >= scratch_element_offset + page_elements) { "
        "dst_bits[i] = 0; } } "
        "for (uint32_t row = 0; row < rows; ++row) { "
        "if (row >= page_elements) { continue; } "
        "const uint32_t row_in_tile = row; "
        "const uint32_t face_row = row_in_tile / kFaceRows; "
        "const uint32_t row_in_face = row_in_tile % kFaceRows; "
        "const uint32_t dst_element = "
        "face_row * (kFaceRows * kTileCols) + row_in_face * kFaceCols; "
        "dst_bits[dst_element] = dst_bits[scratch_element_offset + row]; "
        "} "
        "for (uint32_t i = 0; i < page_elements; ++i) { "
        "dst_bits[scratch_element_offset + i] = 0; } }";
}

void CodeGenBlackhole::PrintCopyCBPage(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  ICHECK_EQ(op->args.size(), 5)
      << "tl.blackhole.copy_cb_page expects 5 arguments";
  need_dataflow_api_h_ = true;
  const int src_cb_id = ResolveCBId(op->args[0]);
  const int dst_cb_id = ResolveCBId(op->args[1]);
  os << "{ const uint32_t page_bytes = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_l1_addr = get_read_ptr(" << src_cb_id << ") + ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t dst_l1_addr = get_write_ptr(" << dst_cb_id << ") + ";
  PrintExpr(op->args[4], os);
  os << "; const volatile uint8_t* src_bytes = "
        "reinterpret_cast<const volatile uint8_t*>(src_l1_addr); "
        "volatile tt_l1_ptr uint8_t* dst_bytes = "
        "reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_l1_addr); "
        "for (uint32_t i = 0; i < page_bytes; ++i) { "
        "dst_bytes[i] = src_bytes[i]; } }";
}

void CodeGenBlackhole::PrintWritePageFromCB(const tvm::tir::CallNode *op,
                                            std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string dst_addr_var = GetRuntimeArgVarForBuffer(op->args[1], "output_buffer_addr");
  const int cb_id = ResolveCBId(op->args[0]);
  const int accessor_offset =
      ResolveAccessorOffsetForBuffer(op->args[1], /*tir_accessor_arg_index=*/4, op,
                                     "tl.blackhole.write_page_from_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ") + ";
  PrintExpr(op->args[5], os);
  EmitTensorAccessorGenerator(os, "dst", accessor_offset, dst_addr_var);
  os << "const uint64_t dst_noc_addr = dst_gen.get_noc_addr(page_id); ";
  os << "if (page_bytes <= 8u) { "
        "const uint32_t scratch_l1_addr = "
        "noc_get_interim_inline_value_addr(noc_index, dst_noc_addr); "
        "volatile uint8_t* scratch_bytes = reinterpret_cast<volatile uint8_t*>(scratch_l1_addr); "
        "const volatile uint8_t* src_bytes = reinterpret_cast<const volatile uint8_t*>(cb_l1_addr); "
        "for (uint32_t i = 0; i < page_bytes; ++i) { scratch_bytes[i] = src_bytes[i]; } "
        "noc_async_write(scratch_l1_addr, dst_noc_addr, page_bytes); "
        "noc_async_write_barrier(); "
        "} else { "
        "noc_async_write(cb_l1_addr, dst_noc_addr, page_bytes); "
        "} }";
}

void CodeGenBlackhole::PrintZeroCBPage(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  ICHECK_EQ(op->args.size(), 3)
      << "tl.blackhole.zero_cb_page expects 3 arguments";
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "{ const uint32_t page_bytes = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ") + ";
  PrintExpr(op->args[2], os);
  os << "; volatile tt_l1_ptr uint32_t* dst_words = "
        "reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_addr); "
        "for (uint32_t i = 0; i < page_bytes / sizeof(uint32_t); ++i) { "
        "dst_words[i] = 0u; } }";
}

void CodeGenBlackhole::PrintGuardMaskToCB(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  ICHECK_EQ(op->args.size(), 4)
      << "tl.blackhole.guard_mask_to_cb expects 4 arguments";
  const auto* page_bytes = op->args[3].as<tvm::tir::IntImmNode>();
  ICHECK(page_bytes != nullptr && page_bytes->value == 2048)
      << "tl.blackhole.guard_mask_to_cb currently admits one bf16 tiled page";
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "{ const uint32_t bound_value = static_cast<uint32_t>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t base_value = static_cast<uint32_t>(";
  PrintExpr(op->args[2], os);
  os << "); const uint32_t valid_cols = "
        "(bound_value <= base_value) ? 0u : "
        "(((bound_value - base_value) >= 32u) ? 32u : (bound_value - base_value)); "
        "volatile tt_l1_ptr uint16_t* dst_u16 = "
        "reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(" << cb_id << ")); "
        "volatile tt_l1_ptr uint32_t* dst_u32 = "
        "reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(" << cb_id << ")); "
        "constexpr uint32_t kTileWords = 2048u / sizeof(uint32_t); "
        "constexpr uint32_t kNegInfPair = 0xFF80FF80u; "
        "constexpr uint16_t kNegInfBf16 = 0xFF80u; "
        "if (valid_cols == 0u) { "
        "for (uint32_t i = 0; i < kTileWords; ++i) { dst_u32[i] = kNegInfPair; } "
        "} else { "
        "for (uint32_t i = 0; i < kTileWords; ++i) { dst_u32[i] = 0u; } "
        "if (valid_cols < 32u) { "
        "const uint32_t cur_pos_in_tile = valid_cols - 1u; "
        "const uint32_t face_start = (cur_pos_in_tile < 15u) ? 0u : 1u; "
        "const uint32_t fill_pos_in_face = (cur_pos_in_tile + 1u) % 16u; "
        "if (face_start == 0u) { "
        "for (uint32_t face = 1u; face < 4u; face += 2u) { "
        "const uint32_t face_word = face << 7; "
        "for (uint32_t j = 0; j < 128u; ++j) { dst_u32[face_word + j] = kNegInfPair; } "
        "} } "
        "const bool fill_odd = (fill_pos_in_face % 2u) == 1u; "
        "const uint32_t fill_word_col = (fill_pos_in_face + 1u) >> 1; "
        "for (uint32_t face = face_start; face < 4u; face += 2u) { "
        "const uint32_t face_u16 = face << 8; "
        "const uint32_t face_u32 = face << 7; "
        "for (uint32_t row = 0; row < 16u; ++row) { "
        "if (fill_odd) { dst_u16[face_u16 + fill_pos_in_face + 16u * row] = kNegInfBf16; } "
        "for (uint32_t col_word = fill_word_col; col_word < 8u; ++col_word) { "
        "dst_u32[face_u32 + col_word + 8u * row] = kNegInfPair; "
        "} } } } } }";
}

void CodeGenBlackhole::PrintMMInit(const tvm::tir::CallNode *op,
                                   std::ostream &os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 3U || op->args.size() == 4U)
      << "tl.blackhole.mm_init expects 3 or 4 arguments";
  os << "mm_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  if (op->args.size() == 4U) {
    os << ", ";
    PrintExpr(op->args[3], os);
  }
  os << ")";
}

void CodeGenBlackhole::PrintReconfigDataFormat(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.reconfig_data_format expects 2 arguments";
  os << "reconfig_data_format(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMMInitShort(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 2U || op->args.size() == 3U)
      << "tl.blackhole.mm_init_short expects 2 or 3 arguments";
  os << "mm_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  if (op->args.size() == 3U) {
    os << ", ";
    PrintExpr(op->args[2], os);
  }
  os << ")";
}

void CodeGenBlackhole::PrintMMInitShortWithDT(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 3U || op->args.size() == 4U)
      << "tl.blackhole.mm_init_short_with_dt expects 3 or 4 arguments";
  os << "mm_init_short_with_dt(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  if (op->args.size() == 4U) {
    os << ", ";
    PrintExpr(op->args[3], os);
  }
  os << ")";
}

void CodeGenBlackhole::PrintMatmulTiles(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_compute_api_h_ = true;
  os << "matmul_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);  // in0_tile_index
  os << ", ";
  PrintExpr(op->args[3], os);  // in1_tile_index
  os << ", ";
  PrintExpr(op->args[4], os);  // dst_tile_index
  os << ")";
}

bool CodeGenBlackhole::TryStartScalarReduction(const tvm::tir::CallNode* op,
                                               const std::string& reduce_kind,
                                               const std::string& reduce_dim,
                                               std::ostream& os) {
  if (core_type_ != CoreType::kTRISC || reduce_kind != "max" || reduce_dim != "row") {
    return false;
  }
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.reduce_init expects 5 arguments";
  const int input_cb = ResolveCBId(op->args[0]);
  const int output_cb = ResolveCBId(op->args[2]);
  if (GetCBDataFormat(input_cb) != "Int32" || GetCBDataFormat(output_cb) != "Int32") {
    return false;
  }
  const int output_page_size = GetCBPageSize(output_cb);
  const int source_tiles = GetCBNumPages(input_cb);
  if (output_page_size <= 0 || output_page_size % static_cast<int>(sizeof(int32_t)) != 0 ||
      source_tiles <= 0) {
    return false;
  }
  const int output_extent = output_page_size / static_cast<int>(sizeof(int32_t));
  const int output_blocks = (output_extent + 31) / 32;
  if (output_extent <= 0 || output_blocks <= 0 || source_tiles % output_blocks != 0) {
    return false;
  }
  const int tiles_per_output_block = source_tiles / output_blocks;
  if (tiles_per_output_block <= 0) {
    return false;
  }

  ICHECK(!active_scalar_reduction_.has_value())
      << "Nested scalar reduction lowering is not supported";
  ScalarReductionContext context;
  context.input_cb = input_cb;
  context.output_cb = output_cb;
  context.output_extent = output_extent;
  context.tiles_per_output_block = tiles_per_output_block;
  context.accumulator_var =
      "__tl_scalar_reduce_" + std::to_string(scalar_reduction_counter_++);
  active_scalar_reduction_ = context;

  os << "int32_t " << context.accumulator_var << "[" << output_extent << "];\n";
  os << "MATH({ for (uint32_t __tl_out = 0; __tl_out < " << output_extent
     << "u; ++__tl_out) { " << context.accumulator_var
     << "[__tl_out] = std::numeric_limits<int32_t>::min(); } })\n";
  return true;
}

bool CodeGenBlackhole::IsActiveScalarReductionInput(int cb_id) const {
  return active_scalar_reduction_.has_value() &&
         active_scalar_reduction_->input_cb == cb_id;
}

bool CodeGenBlackhole::IsActiveScalarReductionOutput(int cb_id) const {
  return active_scalar_reduction_.has_value() &&
         active_scalar_reduction_->output_cb == cb_id;
}

void CodeGenBlackhole::EmitScalarReductionTile(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  ICHECK(active_scalar_reduction_.has_value())
      << "Scalar reduction tile emission requires an active context";
  const ScalarReductionContext& context = active_scalar_reduction_.value();
  os << "do {\n";
  os << "  const uint32_t " << context.accumulator_var << "_tile = ";
  PrintExpr(op->args[2], os);
  os << ";\n";
  os << "  const int32_t* " << context.accumulator_var
     << "_src = reinterpret_cast<const int32_t*>(experimental::CircularBuffer("
     << context.input_cb << ").get_tile_address(" << context.accumulator_var << "_tile));\n";
  os << "  MATH({\n";
  os << "    constexpr uint32_t kOutputExtent = " << context.output_extent << "u;\n";
  os << "    constexpr uint32_t kTilesPerOutputBlock = " << context.tiles_per_output_block
     << "u;\n";
  os << "    constexpr uint32_t kFaceRows = 16u;\n";
  os << "    constexpr uint32_t kFaceCols = 16u;\n";
  os << "    const uint32_t output_base = (" << context.accumulator_var
     << "_tile / kTilesPerOutputBlock) * 32u;\n";
  os << "    for (uint32_t output_in_tile = 0; output_in_tile < 32u; ++output_in_tile) {\n";
  os << "      const uint32_t out_coord = output_base + output_in_tile;\n";
  os << "      if (out_coord >= kOutputExtent) { continue; }\n";
  os << "      for (uint32_t col_in_tile = 0; col_in_tile < 32u; ++col_in_tile) {\n";
  os << "        const uint32_t face_row = output_in_tile / kFaceRows;\n";
  os << "        const uint32_t face_col = col_in_tile / kFaceCols;\n";
  os << "        const uint32_t row_in_face = output_in_tile % kFaceRows;\n";
  os << "        const uint32_t col_in_face = col_in_tile % kFaceCols;\n";
  os << "        const uint32_t offset = face_row * (kFaceRows * 32u) + "
     << "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face;\n";
  os << "        const int32_t value = " << context.accumulator_var << "_src[offset];\n";
  os << "        if (value > " << context.accumulator_var << "[out_coord]) { "
     << context.accumulator_var << "[out_coord] = value; }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  })\n";
  os << "} while (0)";
}

void CodeGenBlackhole::EmitScalarReductionPack(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  (void)op;
  ICHECK(active_scalar_reduction_.has_value())
      << "Scalar reduction pack emission requires an active context";
  const ScalarReductionContext& context = active_scalar_reduction_.value();
  os << "do {\n";
  os << "  int32_t* " << context.accumulator_var
     << "_out = reinterpret_cast<int32_t*>(tilelang_cb_write_ptr_bytes_direct("
     << context.output_cb << "));\n";
  os << "  MATH({ for (uint32_t __tl_out = 0; __tl_out < " << context.output_extent
     << "u; ++__tl_out) { " << context.accumulator_var << "_out[__tl_out] = "
     << context.accumulator_var
     << "[__tl_out]; } mailbox_write(ckernel::ThreadId::PackThreadId, 1); })\n";
  os << "  PACK({ volatile uint32_t __tl_done = mailbox_read(ckernel::ThreadId::MathThreadId); "
     << "(void)__tl_done; })\n";
  os << "} while (0)";
}

void CodeGenBlackhole::PrintTileRegsAcquire(std::ostream &os) {
  need_compute_api_h_ = true;
  tile_regs_scope_active_ = true;
  os << "tile_regs_acquire()";
}

void CodeGenBlackhole::PrintTileRegsCommit(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_commit()";
}

void CodeGenBlackhole::PrintTileRegsWait(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_wait()";
}

void CodeGenBlackhole::PrintTileRegsRelease(std::ostream &os) {
  need_compute_api_h_ = true;
  if (active_scalar_reduction_.has_value() && !tile_regs_scope_active_) {
    os << "(void)0";
    return;
  }
  tile_regs_scope_active_ = false;
  os << "tile_regs_release()";
}

void CodeGenBlackhole::PrintPackTile(const tvm::tir::CallNode *op,
                                     std::ostream &os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.pack_tile expects 2 or 3 arguments";
  const int output_cb = ResolveCBId(op->args[1]);
  if (IsActiveScalarReductionOutput(output_cb)) {
    EmitScalarReductionPack(op, os);
    return;
  }
  os << "pack_tile";
  if (op->args.size() == 3) {
    os << "<true>";
  }
  os << "(";
  PrintExpr(op->args[0], os);  // src_tile_index
  os << ", ";
  os << output_cb;
  if (op->args.size() == 3) {
    os << ", ";
    PrintExpr(op->args[2], os);  // dst_tile_index
  }
  os << ")";
}

void CodeGenBlackhole::PrintPackReconfigDataFormat(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  ICHECK_EQ(op->args.size(), 1)
      << "tl.blackhole.pack_reconfig_data_format expects 1 argument";
  const int cb_id = ResolveCBId(op->args[0]);
  if (IsActiveScalarReductionOutput(cb_id)) {
    os << "(void)0";
    return;
  }
  PrintPackReconfigDataFormatForCB(cb_id, os);
}

void CodeGenBlackhole::PrintCopyTileToDstInitShort(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.copy_tile_to_dst_init_short expects 1 argument";
  os << "copy_tile_to_dst_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintCopyTileToDstInitShortWithDT(const tvm::tir::CallNode* op,
                                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.copy_tile_to_dst_init_short_with_dt expects 2 arguments";
  os << "copy_tile_to_dst_init_short_with_dt(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintCopyTile(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3) << "tl.blackhole.copy_tile expects 3 arguments";
  os << "copy_tile(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintBinaryOpInitCommon(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3)
      << "tl.blackhole.binary_op_init_common expects 3 arguments";
  const int input_cb = ResolveCBId(op->args[0]);
  const int output_cb = ResolveCBId(op->args[2]);
  if (GetCBDataFormat(input_cb) == "Int32" && GetCBDataFormat(output_cb) == "Int32") {
    os << "(void)0";
    return;
  }
  os << "binary_op_init_common(";
  os << input_cb;
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  os << output_cb;
  os << ")";
}

void CodeGenBlackhole::PrintUnaryOpInitCommon(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.unary_op_init_common expects 2 arguments";
  os << "unary_op_init_common(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.add_tiles_init expects 2 or 3 arguments";
  os << "add_tiles_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  if (op->args.size() == 3) {
    os << ", ";
    PrintExpr(op->args[2], os);
  }
  os << ")";
}

void CodeGenBlackhole::PrintAddTiles(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles expects 5 arguments";
  os << "add_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintSubTilesInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.sub_tiles_init expects 2 arguments";
  os << "sub_tiles_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintSubTiles(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.sub_tiles expects 5 arguments";
  os << "sub_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddBcastRowsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.add_bcast_rows_init_short expects 2 arguments";
  os << "add_bcast_rows_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddBcastColsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.add_bcast_cols_init_short expects 2 arguments";
  os << "add_bcast_cols_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesBcastRows(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles_bcast_rows expects 5 arguments";
  os << "add_tiles_bcast_rows(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesBcastCols(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles_bcast_cols expects 5 arguments";
  os << "add_tiles_bcast_cols(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.mul_tiles_init expects 2 arguments";
  os << "mul_tiles_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTiles(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles expects 5 arguments";
  os << "mul_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulBcastRowsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.mul_bcast_rows_init_short expects 2 arguments";
  os << "mul_bcast_rows_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulBcastColsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.mul_bcast_cols_init_short expects 2 arguments";
  os << "mul_bcast_cols_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesBcastRows(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles_bcast_rows expects 5 arguments";
  os << "mul_tiles_bcast_rows(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesBcastCols(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles_bcast_cols expects 5 arguments";
  os << "mul_tiles_bcast<BroadcastType::COL>(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceInit(const tvm::tir::CallNode* op,
                                       std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.reduce_init expects 5 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[3], "tl.blackhole.reduce_init",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[4], "tl.blackhole.reduce_init",
                                                  "reduce_dim");
  if (TryStartScalarReduction(op, reduce_kind, reduce_dim, os)) {
    return;
  }
  os << "reduce_init<" << ReduceKindToTTMetal(reduce_kind, "tl.blackhole.reduce_init") << ", "
     << ReduceDimToTTMetal(reduce_dim, "tl.blackhole.reduce_init") << ">(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceTile(const tvm::tir::CallNode* op,
                                       std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 7) << "tl.blackhole.reduce_tile expects 7 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[5], "tl.blackhole.reduce_tile",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[6], "tl.blackhole.reduce_tile",
                                                  "reduce_dim");
  const int input_cb = ResolveCBId(op->args[0]);
  if (reduce_kind == "max" && reduce_dim == "row" &&
      IsActiveScalarReductionInput(input_cb)) {
    EmitScalarReductionTile(op, os);
    return;
  }
  os << "reduce_tile<" << ReduceKindToTTMetal(reduce_kind, "tl.blackhole.reduce_tile") << ", "
     << ReduceDimToTTMetal(reduce_dim, "tl.blackhole.reduce_tile") << ">(";
  os << input_cb;
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceUninit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.reduce_uninit expects 2 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[0], "tl.blackhole.reduce_uninit",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[1], "tl.blackhole.reduce_uninit",
                                                  "reduce_dim");
  if (active_scalar_reduction_.has_value() &&
      reduce_kind == "max" && reduce_dim == "row") {
    active_scalar_reduction_.reset();
    os << "(void)0";
    return;
  }
  os << "reduce_uninit<false>()";
}

void CodeGenBlackhole::PrintBinaryMaxTileInit(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "binary_max_tile_init()";
}

void CodeGenBlackhole::PrintBinaryMaxTile(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 3 || op->args.size() == 4)
      << "tl.blackhole.binary_max_tile expects 3 or 4 arguments";
  os << "binary_max_tile(";
  PrintExpr(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  if (op->args.size() == 4) {
    const auto* mode = op->args[3].as<StringImmNode>();
    ICHECK(mode != nullptr)
        << "tl.blackhole.binary_max_tile vector_mode must be a string literal";
    os << ", (int)VectorMode::" << mode->value;
  }
  os << ")";
}

void CodeGenBlackhole::PrintDivBinaryTileInit(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "div_binary_tile_init()";
}

void CodeGenBlackhole::PrintDivBinaryTile(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3) << "tl.blackhole.div_binary_tile expects 3 arguments";
  os << "div_binary_tile(";
  PrintExpr(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintExpTileInit(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "exp_tile_init()";
}

void CodeGenBlackhole::PrintExpTile(const tvm::tir::CallNode* op,
                                    std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.exp_tile expects 1 argument";
  os << "exp_tile(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintExp2TileInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "exp2_tile_init()";
}

void CodeGenBlackhole::PrintExp2Tile(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.exp2_tile expects 1 argument";
  os << "exp2_tile(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintRecipTileInit(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.empty() || op->args.size() == 1)
      << "tl.blackhole.recip_tile_init expects 0 or 1 arguments";
  os << "recip_tile_init";
  if (op->args.size() == 1) {
    const auto* legacy = op->args[0].as<IntImmNode>();
    ICHECK(legacy != nullptr && (legacy->value == 0 || legacy->value == 1))
        << "tl.blackhole.recip_tile_init legacy_compat must be literal 0 or 1";
    os << "<" << (legacy->value != 0 ? "true" : "false") << ">";
  }
  os << "()";
}

void CodeGenBlackhole::PrintRecipTile(const tvm::tir::CallNode* op,
                                      std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 1 || op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.recip_tile expects 1, 2, or 3 arguments";
  os << "recip_tile";
  if (op->args.size() == 3) {
    const auto* legacy = op->args[2].as<IntImmNode>();
    ICHECK(legacy != nullptr && (legacy->value == 0 || legacy->value == 1))
        << "tl.blackhole.recip_tile legacy_compat must be literal 0 or 1";
    os << "<" << (legacy->value != 0 ? "true" : "false") << ">";
  }
  os << "(";
  PrintExpr(op->args[0], os);
  if (op->args.size() >= 2) {
    const auto* mode = op->args[1].as<StringImmNode>();
    ICHECK(mode != nullptr)
        << "tl.blackhole.recip_tile vector_mode must be a string literal";
    os << ", (int)VectorMode::" << mode->value;
  }
  os << ")";
}

void CodeGenBlackhole::PrintFillFragment(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var) << "tl.blackhole.fill_fragment expects a direct destination handle var";
  for (const tvm::tir::VarNode* dead_var : dead_fragment_fill_data_vars_) {
    if (SameCodegenStorageVar(dst_var, dead_var)) {
      os << "(void)0";
      return;
    }
  }
  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.fill_fragment", "destination");

  std::ostringstream dtype_os;
  PrintType(dst_dtype, dtype_os);

  os << "MATH({ " << dtype_os.str() << "* dst = reinterpret_cast<" << dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[1], os);
  os << "; const " << dtype_os.str() << " value = static_cast<" << dtype_os.str() << ">(";
  PrintExpr(op->args[2], os);
  os << "); tilelang_fill_fragment(dst, num_elements, value); })";
}

void CodeGenBlackhole::PrintAddFragment(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.add_fragment expects direct source/destination handle vars";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.add_fragment", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.add_fragment", "source");

  std::ostringstream dst_dtype_os;
  std::ostringstream src_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  PrintType(src_dtype, src_dtype_os);

  os << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_add_fragment(dst, src, num_elements); })";
}

void CodeGenBlackhole::PrintAddFragmentFromCBFront(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var) << "tl.blackhole.add_fragment_from_cb_front expects a direct destination handle var";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.add_fragment_from_cb_front", "destination");

  std::ostringstream dst_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  const int cb_id = ResolveCBId(op->args[1]);
  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id
     << "); const " << dst_dtype_os.str() << "* src = reinterpret_cast<const "
     << dst_dtype_os.str() << "*>(cb_front_" << cb_id << ".get_tile_address(0)); "
     << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_add_fragment(dst, src, num_elements); }) }";
}

void CodeGenBlackhole::PrintPackUntilizeSlice(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var) << "tl.blackhole.pack_untilize_slice expects a direct source handle var";
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.pack_untilize_slice", "source");

  std::ostringstream src_dtype_os;
  PrintType(src_dtype, src_dtype_os);

  const int cb_id = ResolveCBId(op->args[1]);
  const PrimExpr src_offset = op->args.size() >= 5 ? op->args[4] : IntImm(DataType::Int(32), 0);
  const bool raw_16bit_float_copy = src_dtype.is_float16() || src_dtype.is_bfloat16();
  if (raw_16bit_float_copy) {
    os << "{ const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); uint16_t* dst_bits = reinterpret_cast<uint16_t*>(tilelang_cb_write_ptr_bytes_direct("
       << cb_id << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(src_offset, os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[3], os);
    os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset_elements + i] = src_bits[src_offset_elements + i]; } }) }";
    return;
  }

  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << src_dtype_os.str() << "* dst = reinterpret_cast<" << src_dtype_os.str()
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(src_offset, os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[3], os);
  os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
     << "dst[dst_offset_elements + i] = src[src_offset_elements + i]; } }) }";
}

void CodeGenBlackhole::PrintPackUntilizeTile(const tvm::tir::CallNode* op,
                                             std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var)
      << "tl.blackhole.pack_untilize_tile expects a direct source handle var";
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.pack_untilize_tile", "source");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = src_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.pack_untilize_tile requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";

  os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_tile_index = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; MATH({ tilelang_tilize_fragment_tile_nfaces<" << bits_type << ">(src_bits + src_offset_elements, "
     << "dst_bits + dst_tile_index * 1024u); }) }";
}

void CodeGenBlackhole::PrintTilizeLocalFragmentSlice(const tvm::tir::CallNode* op,
                                                     std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var)
      << "tl.blackhole.tilize_local_fragment_slice expects a direct source handle var";
  const DataType src_dtype = ResolveHandleDataType(
      src_var, "tl.blackhole.tilize_local_fragment_slice", "source");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = src_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.tilize_local_fragment_slice requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";
  const PrimExpr src_offset = op->args.size() >= 6 ? op->args[5] : IntImm(DataType::Int(32), 0);
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(src_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic fragment->tiled CB bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic fragment->tiled CB bridge requires inverse logical index "
           "expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic fragment->tiled CB bridge requires at least two inverse "
             "layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr =
        tir::Substitute(binding->inverse_logical_index_exprs[
                            binding->inverse_logical_index_exprs.size() >= 2 ? 1 : 0],
                        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic fragment->tiled CB bridge requires trailing inverse "
             "logical indices to be zero for "
          << binding->buffer_name;
    }
    const bool rank1_row_vector_layout =
        IsRank1RowVectorLogicalTileLayout(binding->logical_shape, binding->local_shape);
    PrimExpr emitted_logical_row_expr = logical_row_expr;
    PrimExpr emitted_logical_col_expr = logical_col_expr;
    if (rank1_row_vector_layout) {
      emitted_logical_row_expr = local_index_var;
      emitted_logical_col_expr = IntImm(DataType::Int(32), 0);
    }
    os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
    PrintExpr(op->args[0], os);
    os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
       << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
       << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t row_width = ";
    PrintExpr(op->args[4], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(src_offset, os);
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "const uint32_t tiles_per_row = row_width / kTileCols; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(emitted_logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(emitted_logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * row_width + logical_col; "
          "";
    if (rank1_row_vector_layout) {
      os << "if (thread_idx_x != 0) { continue; } ";
    }
    os << ""
          "if (logical_col >= row_width) { continue; } "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + num_elements) { continue; } "
          "const uint32_t tile_row = logical_row / kTileRows; "
          "const uint32_t tile_col = logical_col / kTileCols; "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
          "const uint32_t tiled_index = tile_index * 1024u + "
          "face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
          "dst_bits[tiled_index] = src_bits[src_offset_elements + __tl_local_i]; } }) }";
    return;
  }

  os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[4], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(src_offset, os);
  os << "; MATH({ tilelang_tilize_fragment_slice_nfaces<" << bits_type
     << ">(src_bits + src_offset_elements, dst_bits, dst_offset_elements, num_elements, row_width); }) }";
}

void CodeGenBlackhole::PrintTilizeCastFragmentSlice(const tvm::tir::CallNode* op,
                                                    std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var)
      << "tl.blackhole.tilize_cast_fragment_slice expects a direct destination handle var";
  ICHECK(src_var)
      << "tl.blackhole.tilize_cast_fragment_slice expects a direct source handle var";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.tilize_cast_fragment_slice", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.tilize_cast_fragment_slice", "source");
  const int cb_id = ResolveCBId(op->args[2]);

  std::ostringstream src_dtype_os;
  PrintType(src_dtype, src_dtype_os);

  std::string dst_bits_type;
  auto make_convert_expr = [&](const std::string& index_name) {
    const std::string src_expr = "static_cast<float>(src[src_offset_elements + " + index_name + "])";
    if (dst_dtype.is_bfloat16()) {
      return "tilelang_float_to_bfloat_bits(" + src_expr + ")";
    }
    if (dst_dtype.is_float16()) {
      return "tilelang_float_to_half_bits(" + src_expr + ")";
    }
    if (dst_dtype.is_float() && dst_dtype.bits() == 32) {
      return "tilelang_bitcast_float_to_u32(" + src_expr + ")";
    }
    return std::string();
  };
  if (dst_dtype.is_bfloat16()) {
    dst_bits_type = "uint16_t";
  } else if (dst_dtype.is_float16()) {
    dst_bits_type = "uint16_t";
  } else if (dst_dtype.is_float() && dst_dtype.bits() == 32) {
    dst_bits_type = "uint32_t";
  } else {
    ICHECK(false)
        << "tl.blackhole.tilize_cast_fragment_slice currently supports only float16, "
           "bfloat16, or float32 destination dtypes";
  }
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(src_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic cast-fragment->tiled CB bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic cast-fragment->tiled CB bridge requires inverse logical "
           "index expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic cast-fragment->tiled CB bridge requires at least two "
             "inverse layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr =
        tir::Substitute(binding->inverse_logical_index_exprs[
                            binding->inverse_logical_index_exprs.size() >= 2 ? 1 : 0],
                        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic cast-fragment->tiled CB bridge requires trailing "
             "inverse logical indices to be zero for "
          << binding->buffer_name;
    }
    const bool rank1_row_vector_layout =
        IsRank1RowVectorLogicalTileLayout(binding->logical_shape, binding->local_shape);
    PrimExpr emitted_logical_row_expr = logical_row_expr;
    PrimExpr emitted_logical_col_expr = logical_col_expr;
    if (rank1_row_vector_layout) {
      emitted_logical_row_expr = local_index_var;
      emitted_logical_col_expr = IntImm(DataType::Int(32), 0);
    }
    os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
       << src_dtype_os.str() << "*>(";
    PrintExpr(op->args[1], os);
    os << "); " << dst_bits_type << "* dst_bits = reinterpret_cast<" << dst_bits_type
       << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
       << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(op->args[4], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[5], os);
    os << "; const uint32_t row_width = ";
    PrintExpr(op->args[6], os);
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "const uint32_t tiles_per_row = row_width / kTileCols; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(emitted_logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(emitted_logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * row_width + logical_col; "
          "";
    if (rank1_row_vector_layout) {
      os << "if (thread_idx_x != 0) { continue; } ";
    }
    os << ""
          "if (logical_col >= row_width) { continue; } "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + num_elements) { continue; } "
          "const uint32_t tile_row = logical_row / kTileRows; "
          "const uint32_t tile_col = logical_col / kTileCols; "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
          "const uint32_t tiled_index = tile_index * 1024u + "
          "face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
          "dst_bits[tiled_index] = ";
    os << make_convert_expr("__tl_local_i");
    os << "; } }) }";
    return;
  }

  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); " << dst_bits_type << "* dst_bits = reinterpret_cast<" << dst_bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
     << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(op->args[4], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[5], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[6], os);
  os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
        "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
        "const uint32_t tiles_per_row = row_width / kTileCols; "
        "for (uint32_t i = 0; i < num_elements; ++i) { "
        "const uint32_t logical_index = dst_offset_elements + i; "
        "const uint32_t global_row = logical_index / row_width; "
        "const uint32_t global_col = logical_index % row_width; "
        "const uint32_t tile_row = global_row / kTileRows; "
        "const uint32_t tile_col = global_col / kTileCols; "
        "const uint32_t row_in_tile = global_row % kTileRows; "
        "const uint32_t col_in_tile = global_col % kTileCols; "
        "const uint32_t face_row = row_in_tile / kFaceRows; "
        "const uint32_t face_col = col_in_tile / kFaceCols; "
        "const uint32_t row_in_face = row_in_tile % kFaceRows; "
        "const uint32_t col_in_face = col_in_tile % kFaceCols; "
        "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
        "const uint32_t tiled_index = tile_index * 1024u + face_row * (kFaceRows * kTileCols) + "
        "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
        "dst_bits[tiled_index] = ";
  os << make_convert_expr("i");
  os << "; } }) }";
}

void CodeGenBlackhole::PrintPackFillFragmentToTiledCB(const tvm::tir::CallNode* op,
                                                      std::ostream& os) {
  ICHECK_EQ(op->args.size(), 6)
      << "tl.blackhole.pack_fill_fragment_to_tiled_cb expects 6 arguments";
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.pack_fill_fragment_to_tiled_cb expects a direct destination handle var";
  const DataType dst_dtype = ResolveHandleDataType(
      dst_var, "tl.blackhole.pack_fill_fragment_to_tiled_cb", "destination");
  const int cb_id = ResolveCBId(op->args[1]);
  if (dst_dtype.is_int() || dst_dtype.is_uint()) {
    std::ostringstream dtype_os;
    PrintType(dst_dtype, dtype_os);
    os << "{ (void)(";
    PrintExpr(op->args[0], os);
    os << "); " << dtype_os.str() << "* dst = reinterpret_cast<" << dtype_os.str()
       << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
       << ")); tilelang_fill_tiled_cb_slice_nfaces<" << dtype_os.str()
       << ">(dst, static_cast<uint32_t>(";
    PrintExpr(op->args[2], os);
    os << "), static_cast<uint32_t>(";
    PrintExpr(op->args[3], os);
    os << "), static_cast<uint32_t>(";
    PrintExpr(op->args[4], os);
    os << "), static_cast<" << dtype_os.str() << ">(";
    PrintExpr(op->args[5], os);
    os << ")); }";
    return;
  }
  if (!dst_dtype.is_bfloat16() && !(dst_dtype.is_float() && dst_dtype.bits() == 32)) {
    ICHECK(false) << "tl.blackhole.pack_fill_fragment_to_tiled_cb currently admits bf16 or "
                     "float32 publication";
  }
  const std::string cb_data_format = GetCBDataFormat(cb_id);
  const bool write_bfloat16 =
      cb_data_format == "Float16_b" || cb_data_format == "Float16";
  const bool write_float32 = cb_data_format == "Float32";
  ICHECK(write_bfloat16 || write_float32)
      << "tl.blackhole.pack_fill_fragment_to_tiled_cb currently admits bf16 or float32 CB "
         "formats, saw "
      << cb_data_format << " for cb_id=" << cb_id;
  os << "{ (void)(";
  PrintExpr(op->args[2], os);
  os << "); (void)(";
  PrintExpr(op->args[4], os);
  os << "); const uint32_t num_tiles = (static_cast<uint32_t>(";
  PrintExpr(op->args[3], os);
  os << ") + 1023u) / 1024u; ";
  if (write_bfloat16) {
    os << "const uint16_t fill_bits = tilelang_float_to_bfloat_bits(static_cast<float>(";
    PrintExpr(op->args[5], os);
    os << ")); volatile tt_l1_ptr uint16_t* dst_bits = "
          "reinterpret_cast<volatile tt_l1_ptr uint16_t*>("
          "tilelang_cb_write_ptr_bytes_direct("
       << cb_id
       << ")); for (uint32_t tile = 0; tile < num_tiles; ++tile) { "
          "const uint32_t tile_base = tile * 1024u; "
          "for (uint32_t i = 0; i < 1024u; ++i) { "
          "dst_bits[tile_base + i] = fill_bits; } } }";
  } else {
    os << "const uint32_t fill_bits = tilelang_bit_cast<uint32_t>(static_cast<float>(";
    PrintExpr(op->args[5], os);
    os << ")); volatile tt_l1_ptr uint32_t* dst_bits = "
          "reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
          "tilelang_cb_write_ptr_bytes_direct("
       << cb_id
       << ")); for (uint32_t tile = 0; tile < num_tiles; ++tile) { "
          "const uint32_t tile_base = tile * 1024u; "
          "for (uint32_t i = 0; i < 1024u; ++i) { "
          "dst_bits[tile_base + i] = fill_bits; } } }";
  }
}

void CodeGenBlackhole::PrintGenerateReduceScalerToCB(const tvm::tir::CallNode* op,
                                                     std::ostream& os) {
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.generate_reduce_scaler_to_cb expects 2 arguments";
  const int cb_id = ResolveCBId(op->args[0]);
  os << "{ volatile tt_l1_ptr uint32_t* dst_words = "
        "reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilelang_cb_write_ptr_bytes_direct("
     << cb_id << ")); "
        "for (uint32_t i = 0; i < 512u; ++i) { dst_words[i] = 0u; } "
        "const uint32_t scaler = static_cast<uint32_t>(";
  PrintExpr(op->args[1], os);
  os << "); if (scaler != 0u) { "
        "for (uint32_t face = 0; face < 4u; ++face) { "
        "const uint32_t face_word = face << 7; "
        "for (uint32_t col_pair = 0; col_pair < 8u; ++col_pair) { "
        "dst_words[face_word + col_pair] = scaler; } } } }";
}

void CodeGenBlackhole::PrintUntilizeCBFrontTile(const tvm::tir::CallNode* op,
                                                std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.untilize_cb_front_tile expects a direct destination handle var";
  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.untilize_cb_front_tile", "destination");

  std::ostringstream dst_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  const int cb_id = ResolveCBId(op->args[1]);
  const bool raw_16bit_float_copy = dst_dtype.is_float16() || dst_dtype.is_bfloat16();

  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); ";
  if (raw_16bit_float_copy) {
    os << "const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(cb_front_" << cb_id
       << ".get_tile_address(";
    PrintExpr(op->args[2], os);
    os << ")); uint16_t* dst_bits = reinterpret_cast<uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[4], os);
    os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset_elements + i] = src_bits[i]; } }) }";
    return;
  }

  os << "const " << dst_dtype_os.str() << "* src = reinterpret_cast<const "
     << dst_dtype_os.str() << "*>(cb_front_" << cb_id << ".get_tile_address(";
  PrintExpr(op->args[2], os);
  os << ")); " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[4], os);
  os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
     << "dst[dst_offset_elements + i] = src[i]; } }) }";
}

void CodeGenBlackhole::PrintUntilizeCBFrontTileFragment(const tvm::tir::CallNode* op,
                                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.untilize_cb_front_tile_fragment expects a direct destination handle var";
  const DataType dst_dtype = ResolveHandleDataType(
      dst_var, "tl.blackhole.untilize_cb_front_tile_fragment", "destination");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = dst_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.untilize_cb_front_tile_fragment requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";
  const std::string cb_data_format = GetCBDataFormat(cb_id);
  const bool cb_bfloat16_to_float32 =
      dst_dtype.is_float() && dst_dtype.bits() == 32 && cb_data_format == "Float16_b";
  const char* src_bits_type = cb_bfloat16_to_float32 ? "uint16_t" : bits_type;
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(dst_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic tiled CB->fragment bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic tiled CB->fragment bridge requires inverse logical index "
           "expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic tiled CB->fragment bridge requires at least two inverse "
             "layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr =
        tir::Substitute(binding->inverse_logical_index_exprs[
                            binding->inverse_logical_index_exprs.size() >= 2 ? 1 : 0],
                        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic tiled CB->fragment bridge requires trailing inverse "
             "logical indices to be zero for "
          << binding->buffer_name;
    }
    const bool rank1_row_vector_layout =
        IsRank1RowVectorLogicalTileLayout(binding->logical_shape, binding->local_shape);
    PrimExpr emitted_logical_row_expr = logical_row_expr;
    PrimExpr emitted_logical_col_expr = logical_col_expr;
    if (rank1_row_vector_layout) {
      emitted_logical_row_expr = local_index_var;
      emitted_logical_col_expr = IntImm(DataType::Int(32), 0);
    }
    os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); const "
       << src_bits_type << "* src_bits = reinterpret_cast<const " << src_bits_type << "*>(cb_front_"
       << cb_id << ".get_tile_address(";
    PrintExpr(op->args[2], os);
    os << ")); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type << "*>(";
    PrintExpr(op->args[0], os);
    os << "); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t logical_row_width = ";
    if (binding->logical_shape.size() >= 2) {
      PrintExpr(binding->logical_shape[1], os);
    } else {
      os << "32u";
    }
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(emitted_logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(emitted_logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * logical_row_width + logical_col; "
          "";
    if (rank1_row_vector_layout) {
      os << "if (thread_idx_x != 0) { continue; } ";
    }
    os << ""
          "if (logical_col >= logical_row_width) { continue; } "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + 1024u) { continue; } "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tiled_index = face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; ";
    if (cb_bfloat16_to_float32) {
      os << "reinterpret_cast<float*>(dst_bits)[__tl_local_i] = "
            "tilelang_bfloat_bits_to_float(src_bits[tiled_index]);";
    } else {
      os << "dst_bits[__tl_local_i] = src_bits[tiled_index];";
    }
    os << " } }) }";
    return;
  }

  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); const "
     << src_bits_type << "* src_bits = reinterpret_cast<const " << src_bits_type << "*>(cb_front_"
     << cb_id << ".get_tile_address(";
  PrintExpr(op->args[2], os);
  os << ")); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  if (cb_bfloat16_to_float32) {
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "uint32_t src_index = 0; float* dst = reinterpret_cast<float*>(dst_bits) + "
          "dst_offset_elements; for (uint32_t face_y = 0; face_y < kTileRows / kFaceRows; "
          "++face_y) { for (uint32_t face_x = 0; face_x < kTileCols / kFaceCols; ++face_x) { "
          "for (uint32_t row = 0; row < kFaceRows; ++row) { "
          "float* dst_row = dst + (face_y * kFaceRows + row) * kTileCols + face_x * kFaceCols; "
          "for (uint32_t col = 0; col < kFaceCols; ++col) { "
          "dst_row[col] = tilelang_bfloat_bits_to_float(src_bits[src_index++]); } } } } }) }";
  } else {
    os << "; MATH({ tilelang_untilize_fragment_tile_nfaces<" << bits_type
       << ">(src_bits, dst_bits + dst_offset_elements); }) }";
  }
}

void CodeGenBlackhole::PrintCastFragmentSlice(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.cast_fragment_slice expects direct source/destination handle vars";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.cast_fragment_slice", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.cast_fragment_slice", "source");

  std::ostringstream dst_dtype_os;
  std::ostringstream src_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  PrintType(src_dtype, src_dtype_os);
  const bool fp32_to_16bit_float_cast =
      (dst_dtype.is_float16() || dst_dtype.is_bfloat16()) && src_dtype.is_float() &&
      src_dtype.bits() == 32;
  if (fp32_to_16bit_float_cast) {
    const char* cast_bits_helper = dst_dtype.is_bfloat16() ? "tilelang_float_to_bfloat_bits"
                                                           : "tilelang_float_to_half_bits";
    os << "MATH({ uint16_t* dst_bits = reinterpret_cast<uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); const float* src = reinterpret_cast<const float*>(";
    PrintExpr(op->args[1], os);
    os << "); const uint32_t dst_offset = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t src_offset = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[4], os);
    os << "; for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset + i] = " << cast_bits_helper
       << "(src[src_offset + i]); } })";
    return;
  }

  os << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t dst_offset = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[4], os);
  os << "; tilelang_cast_fragment_slice(dst, src, dst_offset, src_offset, num_elements); })";
}

void CodeGenBlackhole::PrintKernelAttributes() {
  // Print kernel-specific attributes for TT-Metal
  // This is a placeholder for future kernel attribute emission
}

void CodeGenBlackhole::PrintCBDeclare(const std::string &name,
                                      tvm::DataType dtype, int num_pages,
                                      int page_size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// CB declaration: " << name << "\n";
  PrintIndent();
  stream << "// TODO: Implement CB allocation\n";
}

void CodeGenBlackhole::PrintCBWaitFront(const std::string &name,
                                        int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_wait_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPopFront(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_pop_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBReserveBack(const std::string &name,
                                          int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_reserve_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPushBack(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_push_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintNOCRead(const std::string &src_addr,
                                    const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC read: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWrite(const std::string &src_addr,
                                     const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC write: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWait() {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "noc_async_read_barrier();\n";
}

void CodeGenBlackhole::PrintGetSemaphore(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "get_semaphore(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintRuntimeArgU32(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  const auto* arg_name = op->args[0].as<tvm::tir::StringImmNode>();
  ICHECK(arg_name) << "tl.blackhole.runtime_arg_u32 expects a string literal name";
  auto it = runtime_arg_vars_by_name_.find(arg_name->value);
  ICHECK(it != runtime_arg_vars_by_name_.end())
      << "Missing runtime arg binding for name: " << arg_name->value;
  if (op->dtype.is_int() && op->dtype.bits() == 32) {
    os << "static_cast<int32_t>(" << it->second << ")";
    return;
  }
  os << it->second;
}

void CodeGenBlackhole::PrintSemaphoreWait(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreSet(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreIncRemote(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_inc(get_noc_addr(";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[3], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreSetRemote(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_set_remote(";
  PrintExpr(op->args[0], os);
  os << ", get_noc_addr(";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << "))";
}

}  // namespace tl
}  // namespace tvm
