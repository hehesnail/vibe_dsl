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
 * \file analyze_blackhole_fragment_regions.cc
 * \brief Analyze split-after Blackhole fragment compute regions and emit a structured IR attr.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using tir::BlockNode;
using tir::Buffer;
using tir::BufferLoad;
using tir::BufferLoadNode;
using tir::BufferStoreNode;
using tir::CallNode;
using tir::ForNode;
using tir::PrimFunc;
using tir::StmtExprVisitor;
using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

std::string ExprToString(const PrimExpr& expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

struct BufferInfo {
  Buffer buffer;
  std::string scope;
};

bool IsFragmentLikeScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment";
}

bool ExprUsesFloorDivLikeIndex(const PrimExpr& expr) {
  bool found = false;
  tir::PostOrderVisit(expr, [&found](const ObjectRef& node) {
    if (node.as<tir::FloorDivNode>() || node.as<tir::FloorModNode>()) {
      found = true;
    }
  });
  return found;
}

bool IsLoadFromBuffer(const PrimExpr& expr, const std::string& buffer_name) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return load->buffer->name == buffer_name;
  }
  return false;
}

bool IsLoadFromNonFragmentLocal(const PrimExpr& expr,
                                const std::unordered_map<std::string, BufferInfo>& fragment_buffers) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return load->buffer.scope() == "local" && !fragment_buffers.count(load->buffer->name);
  }
  return false;
}

class FragmentRegionAnalyzer final : public StmtExprVisitor {
 public:
  void Analyze(const PrimFunc& func) { VisitStmt(func->body); }

  bool HasRegion() const { return !fragment_buffers_.empty() || !seen_ops_.empty(); }

  Map<String, Any> EncodeSingleRegion() const {
    Map<String, Any> region;

    Array<Any> fragment_buffers;
    for (const auto& name : fragment_buffer_order_) {
      const auto& info = fragment_buffers_.at(name);
      Map<String, Any> entry;
      entry.Set("name", String(name));
      entry.Set("scope", String(info.scope));
      fragment_buffers.push_back(entry);
    }
    region.Set("fragment_buffers", fragment_buffers);

    Array<Any> ops;
    for (const auto& op_name : op_order_) {
      ops.push_back(String(op_name));
    }
    region.Set("ops", ops);

    Array<Any> row_reductions;
    for (const auto& target : row_reduction_targets_) {
      Map<String, Any> entry;
      entry.Set("target", String(target.first));
      entry.Set("kind", String(target.second));
      row_reductions.push_back(entry);
    }
    region.Set("row_reductions", row_reductions);

    Array<Any> row_broadcasts;
    for (const auto& source : row_broadcast_sources_) {
      Map<String, Any> entry;
      entry.Set("source", String(source));
      row_broadcasts.push_back(entry);
    }
    region.Set("row_broadcasts", row_broadcasts);

    Array<Any> loop_carried_state;
    for (const auto& name : loop_carried_order_) {
      Map<String, Any> entry;
      entry.Set("name", String(name));
      loop_carried_state.push_back(entry);
    }
    region.Set("loop_carried_state", loop_carried_state);

    return region;
  }

 private:
  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "tilelang_root") {
      for (const Buffer& buffer : op->alloc_buffers) {
        const std::string scope = buffer.scope();
        if (!IsFragmentLikeScope(scope)) {
          continue;
        }
        const std::string name = buffer->name;
        if (fragment_buffers_.emplace(name, BufferInfo{buffer, scope}).second) {
          fragment_buffer_order_.push_back(name);
        }
      }

      if (const auto* seq = op->body.as<tir::SeqStmtNode>()) {
        bool seen_pipeline_loop = false;
        for (const auto& stmt : seq->seq) {
          if (!seen_pipeline_loop && stmt.as<ForNode>()) {
            seen_pipeline_loop = true;
            inside_pipeline_loop_ = true;
            VisitStmt(stmt);
            inside_pipeline_loop_ = false;
            continue;
          }
          if (!seen_pipeline_loop) {
            pre_loop_stmt_ = true;
            VisitStmt(stmt);
            pre_loop_stmt_ = false;
          } else {
            post_loop_stmt_ = true;
            VisitStmt(stmt);
            post_loop_stmt_ = false;
          }
        }
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    const bool prev_inside = inside_pipeline_loop_;
    inside_pipeline_loop_ = true;
    StmtExprVisitor::VisitStmt_(op);
    inside_pipeline_loop_ = prev_inside;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    const std::string target_name = op->buffer->name;
    if (fragment_buffers_.count(target_name)) {
      if (pre_loop_stmt_) {
        pre_loop_writes_.insert(target_name);
      }
      if (inside_pipeline_loop_) {
        in_loop_writes_.insert(target_name);
      }
      if (post_loop_stmt_) {
        post_loop_writes_.insert(target_name);
      }

      DetectOpsAndRelationships(target_name, op->value, op->indices);
    } else if (op->buffer.scope() == "local") {
      DetectTempReductionBuffer(target_name, op->value);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const std::string source_name = op->buffer->name;
    if (fragment_buffers_.count(source_name)) {
      if (pre_loop_stmt_) {
        pre_loop_reads_.insert(source_name);
      }
      if (inside_pipeline_loop_) {
        in_loop_reads_.insert(source_name);
      }
      if (post_loop_stmt_) {
        post_loop_reads_.insert(source_name);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (const auto* op_node = op->op.as<OpNode>()) {
      const std::string op_name = op_node->name;
      if (op_name == "tl.tileop.gemm_py") {
        AddOp("gemm");
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void DetectTempReductionBuffer(const std::string& target_name, const PrimExpr& value) {
    bool has_allreduce_max = false;
    bool has_allreduce_sum = false;
    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          if (op_node->name == "tir.call_extern" && !call->args.empty()) {
            const std::string callee = ExprToString(GetRef<PrimExpr>(call));
            has_allreduce_max |= callee.find("AllReduce<tl::MaxOp") != std::string::npos;
            has_allreduce_sum |= callee.find("AllReduce<tl::SumOp") != std::string::npos;
          }
        }
      }
    });
    if (has_allreduce_max) {
      temp_reduction_buffers_[target_name] = "max";
    } else if (has_allreduce_sum) {
      temp_reduction_buffers_[target_name] = "sum";
    }
  }

  void DetectOpsAndRelationships(const std::string& target_name, const PrimExpr& value,
                                 const Array<PrimExpr>& store_indices) {
    bool saw_pointwise = false;
    bool saw_floor_div_broadcast = false;
    std::unordered_set<std::string> local_sources;
    bool has_allreduce_max = false;
    bool has_allreduce_sum = false;
    bool has_self_max_with_temp = false;
    std::string temp_reduction_kind;

    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          const std::string op_name = op_node->name;
          if (op_name == "tir.call_extern" && !call->args.empty()) {
            const std::string callee = ExprToString(GetRef<PrimExpr>(call));
            has_allreduce_max |= callee.find("AllReduce<tl::MaxOp") != std::string::npos;
            has_allreduce_sum |= callee.find("AllReduce<tl::SumOp") != std::string::npos;
          }
          if (op_name == "tir.exp2" || op_name == "tir.max" || op_name == "tir.multiply" ||
                     op_name == "tir.add" || op_name == "tir.divide" ||
                     op_name == "tir.if_then_else") {
            saw_pointwise = true;
          }
          if (op_name == "tir.max") {
            if (call->args.size() == 2) {
              const bool lhs_self_rhs_temp =
                  IsLoadFromBuffer(call->args[0], target_name) &&
                  IsLoadFromNonFragmentLocal(call->args[1], fragment_buffers_);
              const bool rhs_self_lhs_temp =
                  IsLoadFromBuffer(call->args[1], target_name) &&
                  IsLoadFromNonFragmentLocal(call->args[0], fragment_buffers_);
              has_self_max_with_temp |= lhs_self_rhs_temp || rhs_self_lhs_temp;
            }
            bool has_self_source = false;
            bool has_temp_source = false;
            for (const PrimExpr& arg : call->args) {
              tir::PostOrderVisit(arg, [&](const ObjectRef& arg_node) {
                if (const auto* load = arg_node.as<BufferLoadNode>()) {
                  const std::string source_name = load->buffer->name;
                  if (source_name == target_name) {
                    has_self_source = true;
                  } else if (!fragment_buffers_.count(source_name) &&
                             load->buffer.scope() == "local") {
                    has_temp_source = true;
                  }
                }
              });
            }
            has_self_max_with_temp |= has_self_source && has_temp_source;
          }
        }
      } else if (const auto* load = node.as<BufferLoadNode>()) {
        const std::string source_name = load->buffer->name;
        if (source_name == target_name || !fragment_buffers_.count(source_name)) {
          if (const auto it = temp_reduction_buffers_.find(source_name);
              it != temp_reduction_buffers_.end()) {
            temp_reduction_kind = it->second;
          }
          return;
        }
        local_sources.insert(source_name);
        for (const PrimExpr& index : load->indices) {
          if (ExprUsesFloorDivLikeIndex(index)) {
            saw_floor_div_broadcast = true;
          }
        }
      }
    });

    for (const PrimExpr& index : store_indices) {
      if (ExprUsesFloorDivLikeIndex(index)) {
        saw_floor_div_broadcast = true;
      }
    }

    if (saw_pointwise) {
      AddOp("pointwise_chain");
    }

    if (has_allreduce_max || has_self_max_with_temp || temp_reduction_kind == "max") {
      AddOp("row_reduction");
      row_reduction_targets_.push_back({target_name, "max"});
    }
    if (has_allreduce_sum || temp_reduction_kind == "sum") {
      AddOp("row_reduction");
      row_reduction_targets_.push_back({target_name, "sum"});
    }

    if (!local_sources.empty() && saw_floor_div_broadcast) {
      AddOp("row_broadcast");
      for (const auto& source_name : local_sources) {
        if (seen_row_broadcast_sources_.insert(source_name).second) {
          row_broadcast_sources_.push_back(source_name);
        }
      }
    }
  }

  void AddOp(const std::string& op_name) {
    if (seen_ops_.insert(op_name).second) {
      op_order_.push_back(op_name);
    }
  }

  std::unordered_map<std::string, BufferInfo> fragment_buffers_;
  std::vector<std::string> fragment_buffer_order_;

  std::unordered_set<std::string> seen_ops_;
  std::vector<std::string> op_order_;

  std::vector<std::pair<std::string, std::string>> row_reduction_targets_;
  std::unordered_set<std::string> seen_row_broadcast_sources_;
  std::vector<std::string> row_broadcast_sources_;

  bool pre_loop_stmt_ = false;
  bool inside_pipeline_loop_ = false;
  bool post_loop_stmt_ = false;

  std::unordered_set<std::string> pre_loop_writes_;
  std::unordered_set<std::string> in_loop_writes_;
  std::unordered_set<std::string> post_loop_writes_;
  std::unordered_set<std::string> pre_loop_reads_;
  std::unordered_set<std::string> in_loop_reads_;
  std::unordered_set<std::string> post_loop_reads_;
  std::unordered_map<std::string, std::string> temp_reduction_buffers_;

  mutable std::vector<std::string> loop_carried_order_;

  void FinalizeLoopCarriedState() const {
    if (!loop_carried_order_.empty()) {
      return;
    }
    for (const auto& name : fragment_buffer_order_) {
      const bool carried = in_loop_writes_.count(name) &&
                           (pre_loop_writes_.count(name) || in_loop_reads_.count(name) ||
                            post_loop_reads_.count(name));
      if (carried) {
        loop_carried_order_.push_back(name);
      }
    }
  }

 public:
  Map<String, Any> Encode() const {
    FinalizeLoopCarriedState();
    Map<String, Any> region = EncodeSingleRegion();
    Map<String, Any> result;
    Array<Any> regions;
    regions.push_back(region);
    result.Set("regions", regions);
    return result;
  }
};

}  // namespace

tir::transform::Pass AnalyzeBlackholeFragmentRegionsPass() {
  auto fpass = [](PrimFunc func, IRModule, tir::transform::PassContext) -> PrimFunc {
    FragmentRegionAnalyzer analyzer;
    analyzer.Analyze(func);
    if (!analyzer.HasRegion()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.fragment_regions", analyzer.Encode()["regions"]);
    PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AnalyzeBlackholeFragmentRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeBlackholeFragmentRegions",
                        AnalyzeBlackholeFragmentRegionsPass);
}

}  // namespace tl
}  // namespace tvm
