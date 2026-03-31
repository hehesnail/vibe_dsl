/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file analyze_blackhole_work_decomposition.cc
 * \brief Analyze split-after Blackhole work decomposition and emit a unified IR attr.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/structural_hash.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using tir::AttrStmtNode;
using tir::ForNode;
using tir::IterVar;
using tir::PrimFunc;
using tir::StmtExprVisitor;
using tvm::DictAttrs;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

std::string NormalizeLaunchAxisName(const std::string& name) {
  if (name == "blockIdx.x" || name == "bx") {
    return "bx";
  }
  if (name == "blockIdx.y" || name == "by") {
    return "by";
  }
  if (name == "blockIdx.z" || name == "bz") {
    return "bz";
  }
  return "";
}

class WorkDecompositionAnalyzer : public StmtExprVisitor {
 public:
  void Analyze(const PrimFunc& func) { VisitStmt(func->body); }

  bool HasWorkDecomposition() const {
    return !axes_.empty() || !derived_index_exprs_.empty() || !work_dependent_loop_bounds_.empty();
  }

  Map<String, Any> Encode() const {
    Map<String, Any> work_info;

    Array<Any> axes;
    for (const auto& axis : axes_) {
      axes.push_back(String(axis));
    }
    work_info.Set("axes", axes);

    Array<Any> derived_index_exprs;
    for (const auto& expr : derived_index_exprs_) {
      Map<String, Any> entry;
      entry.Set("expr", expr);
      derived_index_exprs.push_back(entry);
    }
    work_info.Set("derived_index_exprs", derived_index_exprs);

    Array<Any> work_dependent_loop_bounds;
    for (const auto& loop_bound_info : work_dependent_loop_bounds_) {
      Map<String, Any> loop_bound;
      loop_bound.Set("loop_var", String(loop_bound_info.loop_var_name));
      loop_bound.Set("min", loop_bound_info.min);
      loop_bound.Set("extent", loop_bound_info.extent);
      work_dependent_loop_bounds.push_back(loop_bound);
    }
    work_info.Set("work_dependent_loop_bounds", work_dependent_loop_bounds);

    return work_info;
  }

 private:
  struct LoopBoundInfo {
    std::string loop_var_name;
    PrimExpr min;
    PrimExpr extent;
  };

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      std::string axis = NormalizeLaunchAxisName(iv->var->name_hint);
      if (!axis.empty()) {
        if (seen_axes_.insert(axis).second) {
          axes_.push_back(axis);
        }
        launch_vars_.emplace(iv->var.get(), axis);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    MaybeRecordDerivedExpr(op->min);
    MaybeRecordDerivedExpr(op->extent);

    if (ExprUsesLaunchVar(op->min) || ExprUsesLaunchVar(op->extent)) {
      std::string loop_key = op->loop_var->name_hint + "|" + StructuralKey(op->min) + "|" +
                             StructuralKey(op->extent);
      if (seen_loop_bounds_.insert(loop_key).second) {
        work_dependent_loop_bounds_.push_back(
            {op->loop_var->name_hint, op->min, op->extent});
      }
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr(const PrimExpr& expr) final {
    MaybeRecordDerivedExpr(expr);
    StmtExprVisitor::VisitExpr(expr);
  }

  bool ExprUsesLaunchVar(const PrimExpr& expr) const {
    bool uses_launch_var = false;
    tir::PostOrderVisit(expr, [&uses_launch_var, this](const ObjectRef& node) {
      if (const auto* var = node.as<tir::VarNode>()) {
        uses_launch_var |= launch_vars_.count(var);
      }
    });
    return uses_launch_var;
  }

  bool IsTrivialLaunchExpr(const PrimExpr& expr) const {
    if (const auto* var = expr.as<tir::VarNode>()) {
      return launch_vars_.count(var);
    }
    return expr.as<IntImmNode>() != nullptr;
  }

  void MaybeRecordDerivedExpr(const PrimExpr& expr) {
    if (!expr.defined() || !ExprUsesLaunchVar(expr) || IsTrivialLaunchExpr(expr)) {
      return;
    }
    std::string expr_key = StructuralKey(expr);
    if (seen_derived_exprs_.insert(expr_key).second) {
      derived_index_exprs_.push_back(expr);
    }
  }

  std::string StructuralKey(const PrimExpr& expr) const {
    return std::to_string(StructuralHash()(expr));
  }

  std::unordered_map<const tir::VarNode*, std::string> launch_vars_;
  std::unordered_set<std::string> seen_axes_;
  std::vector<std::string> axes_;
  std::unordered_set<std::string> seen_derived_exprs_;
  std::vector<PrimExpr> derived_index_exprs_;
  std::unordered_set<std::string> seen_loop_bounds_;
  std::vector<LoopBoundInfo> work_dependent_loop_bounds_;
};

}  // namespace

tir::transform::Pass AnalyzeBlackholeWorkDecompositionPass() {
  auto fpass = [](PrimFunc func, IRModule, tir::transform::PassContext) -> PrimFunc {
    WorkDecompositionAnalyzer analyzer;
    analyzer.Analyze(func);
    if (!analyzer.HasWorkDecomposition()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.work_decomposition", analyzer.Encode());

    PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AnalyzeBlackholeWorkDecomposition", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeBlackholeWorkDecomposition",
                        AnalyzeBlackholeWorkDecompositionPass);
}

}  // namespace tl
}  // namespace tvm
