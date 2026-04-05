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
#include <cctype>
#include <cstring>
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
using tir::CastNode;
using tir::DivNode;
using tir::ForNode;
using tir::MaxNode;
using tir::MulNode;
using tir::PrimFunc;
using tir::Stmt;
using tir::StmtExprVisitor;
using tir::AddNode;
using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::GetRef;
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

int64_t StaticNumElements(const Buffer& buffer) {
  int64_t num_elements = 1;
  for (const PrimExpr& dim : buffer->shape) {
    if (const auto* int_imm = dim.as<tir::IntImmNode>()) {
      num_elements *= int_imm->value;
      continue;
    }
    return -1;
  }
  return num_elements;
}

bool IsFragmentLikeScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

std::string CanonicalBufferName(const std::string& name) {
  size_t pos = name.size();
  while (pos > 0 && std::isdigit(static_cast<unsigned char>(name[pos - 1]))) {
    --pos;
  }
  if (pos > 0 && pos < name.size() && name[pos - 1] == '_') {
    return name.substr(0, pos - 1);
  }
  return name;
}

bool ExprUsesFloorDivLikeIndex(const PrimExpr& expr) {
  bool found = false;
  tir::PostOrderVisit(expr, [&found](const ObjectRef& node) {
    if (node.as<tir::FloorDivNode>() || node.as<tir::FloorModNode>()) {
      found = true;
      return;
    }
    if (const auto* call = node.as<CallNode>()) {
      if (const auto* op_node = call->op.as<OpNode>()) {
        if (op_node->name == "tir.shift_right") {
          found = true;
        }
      }
    }
  });
  return found;
}

bool IsLoadFromBuffer(const PrimExpr& expr, const std::string& buffer_name) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return CanonicalBufferName(load->buffer->name) == buffer_name;
  }
  return false;
}

bool IsLoadFromNonFragmentLocal(const PrimExpr& expr,
                                const std::unordered_map<std::string, BufferInfo>& fragment_buffers) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    const std::string canonical_name = CanonicalBufferName(load->buffer->name);
    return load->buffer.scope() == "local" && !fragment_buffers.count(canonical_name);
  }
  return false;
}

bool IsLoadFromFragmentBuffer(const PrimExpr& expr,
                              const std::unordered_map<std::string, BufferInfo>& fragment_buffers) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return fragment_buffers.count(CanonicalBufferName(load->buffer->name));
  }
  return false;
}

class FragmentRegionAnalyzer final : public StmtExprVisitor {
 public:
  void Analyze(const PrimFunc& func) { AnalyzeTopLevel(func->body); }

  bool HasRegion() const { return !fragment_buffers_.empty() || !seen_ops_.empty(); }

  Map<String, Any> EncodeSingleRegion() const {
    Map<String, Any> region;

    Array<Any> fragment_buffers;
    for (const auto& name : fragment_buffer_order_) {
      const auto& info = fragment_buffers_.at(name);
      Map<String, Any> entry;
      entry.Set("name", String(name));
      entry.Set("scope", String(info.scope));
      entry.Set("is_integer", Integer(info.buffer->dtype.is_int() || info.buffer->dtype.is_uint()));
      fragment_buffers.push_back(entry);
    }
    region.Set("fragment_buffers", fragment_buffers);

    Array<Any> ops;
    for (const auto& op_name : op_order_) {
      ops.push_back(String(op_name));
    }
    region.Set("ops", ops);

    Array<Any> pointwise_ops;
    for (const auto& op_name : pointwise_op_order_) {
      pointwise_ops.push_back(String(op_name));
    }
    region.Set("pointwise_ops", pointwise_ops);

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

    Array<Any> selection_targets;
    for (const auto& target : selection_target_order_) {
      selection_targets.push_back(String(target));
    }
    region.Set("selection_targets", selection_targets);

    Array<Any> update_sources;
    for (const auto& target : update_source_target_order_) {
      Map<String, Any> entry;
      entry.Set("target", String(target));
      Array<Any> sources;
      for (const auto& source : update_source_order_.at(target)) {
        sources.push_back(String(source));
      }
      entry.Set("sources", sources);
      update_sources.push_back(entry);
    }
    region.Set("update_sources", update_sources);

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
  void RegisterFragmentBuffer(const Buffer& buffer, bool allow_plain_local = false) {
    const std::string scope = buffer.scope();
    if (!IsFragmentLikeScope(scope)) {
      return;
    }
    const std::string name = CanonicalBufferName(buffer->name);
    if (scope == "local" && !allow_plain_local) {
      static constexpr const char* kClearSuffix = "_clear";
      const size_t suffix_len = std::strlen(kClearSuffix);
      if (name.size() > suffix_len &&
          name.compare(name.size() - suffix_len, suffix_len, kClearSuffix) == 0) {
        return;
      }
    }
    if (fragment_buffers_.emplace(name, BufferInfo{buffer, scope}).second) {
      fragment_buffer_order_.push_back(name);
    }
  }

  void AnalyzeTopLevel(const Stmt& body) {
    if (const auto* block = body.as<BlockNode>()) {
      if (block->name_hint == "tilelang_root") {
        for (const Buffer& buffer : block->alloc_buffers) {
          RegisterFragmentBuffer(buffer, true);
        }
        AnalyzeTopLevel(block->body);
        return;
      }
    }
    if (const auto* seq = body.as<tir::SeqStmtNode>()) {
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
    VisitStmt(body);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "tilelang_root") {
      for (const Buffer& buffer : op->alloc_buffers) {
        RegisterFragmentBuffer(buffer);
      }
      AnalyzeTopLevel(op->body);
      return;
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
    RegisterFragmentBuffer(op->buffer);
    const std::string target_name = CanonicalBufferName(op->buffer->name);
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
    RegisterFragmentBuffer(op->buffer);
    const std::string source_name = CanonicalBufferName(op->buffer->name);
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
    bool has_local_max_reduction = false;
    bool has_local_sum_reduction = false;
    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          const std::string op_name = op_node->name;
          if (op_node->name == "tir.call_extern" && !call->args.empty()) {
            const std::string callee = ExprToString(GetRef<PrimExpr>(call));
            has_allreduce_max |= callee.find("AllReduce<tl::MaxOp") != std::string::npos;
            has_allreduce_sum |= callee.find("AllReduce<tl::SumOp") != std::string::npos;
          }
          if (call->args.size() == 2) {
            const bool lhs_self = IsLoadFromBuffer(call->args[0], target_name);
            const bool rhs_self = IsLoadFromBuffer(call->args[1], target_name);
            const bool lhs_fragment = IsLoadFromFragmentBuffer(call->args[0], fragment_buffers_);
            const bool rhs_fragment = IsLoadFromFragmentBuffer(call->args[1], fragment_buffers_);
            if (op_name == "tir.max") {
              has_local_max_reduction |= (lhs_self && rhs_fragment) || (rhs_self && lhs_fragment);
            } else if (op_name == "tir.add") {
              has_local_sum_reduction |= (lhs_self && rhs_fragment) || (rhs_self && lhs_fragment);
            }
          }
        }
      }
    });
    if (has_allreduce_max) {
      temp_reduction_buffers_[target_name] = "max";
    } else if (has_allreduce_sum) {
      temp_reduction_buffers_[target_name] = "sum";
    } else if (has_local_max_reduction) {
      temp_reduction_buffers_[target_name] = "max";
    } else if (has_local_sum_reduction) {
      temp_reduction_buffers_[target_name] = "sum";
    }
  }

  void DetectOpsAndRelationships(const std::string& target_name, const PrimExpr& value,
                                 const Array<PrimExpr>& store_indices) {
    bool saw_pointwise = false;
    bool saw_floor_div_broadcast = false;
    bool saw_rank_broadcast = false;
    bool saw_scalar_fragment_broadcast = false;
    bool saw_direct_fragment_max_reduction = false;
    bool saw_direct_fragment_sum_reduction = false;
    std::unordered_set<std::string> local_sources;
    bool has_allreduce_max = false;
    bool has_allreduce_sum = false;
    bool has_self_max_with_temp = false;
    std::string temp_reduction_kind;
    const int64_t target_elements = StaticNumElements(fragment_buffers_.at(target_name).buffer);

    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          const std::string op_name = op_node->name;
          if (op_name == "tir.call_extern" && !call->args.empty()) {
            const std::string callee = ExprToString(GetRef<PrimExpr>(call));
            has_allreduce_max |= callee.find("AllReduce<tl::MaxOp") != std::string::npos;
            has_allreduce_sum |= callee.find("AllReduce<tl::SumOp") != std::string::npos;
          }
          if (op_name == "tir.exp2" || op_name == "tir.if_then_else") {
            saw_pointwise = true;
            AddPointwiseOp(op_name == "tir.exp2" ? "exp2" : "if_then_else");
            if (op_name == "tir.if_then_else") {
              AddSelectionTarget(target_name);
            }
          }
        }
      } else if (node.as<CastNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("cast");
      } else if (const auto* max = node.as<MaxNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("max");
        const bool lhs_self_rhs_temp =
            IsLoadFromBuffer(max->a, target_name) &&
            IsLoadFromNonFragmentLocal(max->b, fragment_buffers_);
        const bool rhs_self_lhs_temp =
            IsLoadFromBuffer(max->b, target_name) &&
            IsLoadFromNonFragmentLocal(max->a, fragment_buffers_);
        has_self_max_with_temp |= lhs_self_rhs_temp || rhs_self_lhs_temp;

        const bool lhs_self_rhs_fragment =
            IsLoadFromBuffer(max->a, target_name) && IsLoadFromFragmentBuffer(max->b, fragment_buffers_);
        const bool rhs_self_lhs_fragment =
            IsLoadFromBuffer(max->b, target_name) && IsLoadFromFragmentBuffer(max->a, fragment_buffers_);
        if (target_elements == 1) {
          saw_direct_fragment_max_reduction |= lhs_self_rhs_fragment || rhs_self_lhs_fragment;
        }
      } else if (const auto* add = node.as<AddNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("add");
        if (target_elements == 1) {
          const bool lhs_self_rhs_fragment =
              IsLoadFromBuffer(add->a, target_name) &&
              IsLoadFromFragmentBuffer(add->b, fragment_buffers_);
          const bool rhs_self_lhs_fragment =
              IsLoadFromBuffer(add->b, target_name) &&
              IsLoadFromFragmentBuffer(add->a, fragment_buffers_);
          saw_direct_fragment_sum_reduction |= lhs_self_rhs_fragment || rhs_self_lhs_fragment;
        }
      } else if (node.as<MulNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("mul");
      } else if (node.as<DivNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("div");
      } else if (const auto* load = node.as<BufferLoadNode>()) {
        const std::string source_name = CanonicalBufferName(load->buffer->name);
        if (source_name == target_name || !fragment_buffers_.count(source_name)) {
          if (const auto it = temp_reduction_buffers_.find(source_name);
              it != temp_reduction_buffers_.end()) {
            temp_reduction_kind = it->second;
          }
          return;
        }
        local_sources.insert(source_name);
        if (load->indices.size() < store_indices.size()) {
          saw_rank_broadcast = true;
        }
        const int64_t source_elements = StaticNumElements(fragment_buffers_.at(source_name).buffer);
        if (source_elements == 1 && target_elements > 1) {
          saw_scalar_fragment_broadcast = true;
        }
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

    if (value.as<IntImmNode>() || value.as<FloatImmNode>()) {
      saw_pointwise = true;
      AddPointwiseOp("fill");
    }

    if (saw_pointwise) {
      AddOp("pointwise_chain");
    }

    if (has_allreduce_max || has_self_max_with_temp || temp_reduction_kind == "max" ||
        saw_direct_fragment_max_reduction) {
      AddOp("row_reduction");
      AddRowReduction(target_name, "max");
    }
    if (has_allreduce_sum || temp_reduction_kind == "sum" || saw_direct_fragment_sum_reduction) {
      AddOp("row_reduction");
      AddRowReduction(target_name, "sum");
    }

    if (!local_sources.empty() &&
        (saw_floor_div_broadcast || saw_rank_broadcast || saw_scalar_fragment_broadcast)) {
      AddOp("row_broadcast");
      for (const auto& source_name : local_sources) {
        if (seen_row_broadcast_sources_.insert(source_name).second) {
          row_broadcast_sources_.push_back(source_name);
        }
      }
    }

    if (!local_sources.empty()) {
      AddUpdateSources(target_name, local_sources);
    }
  }

  void AddOp(const std::string& op_name) {
    if (seen_ops_.insert(op_name).second) {
      op_order_.push_back(op_name);
    }
  }

  void AddPointwiseOp(const std::string& op_name) {
    if (seen_pointwise_ops_.insert(op_name).second) {
      pointwise_op_order_.push_back(op_name);
    }
  }

  std::string CanonicalReductionTarget(const std::string& target_name) const {
    if (fragment_buffers_.count(target_name)) {
      return target_name;
    }
    static constexpr const char* kClearSuffix = "_clear";
    const size_t suffix_len = std::strlen(kClearSuffix);
    if (target_name.size() > suffix_len &&
        target_name.compare(target_name.size() - suffix_len, suffix_len, kClearSuffix) == 0) {
      const std::string base_name = target_name.substr(0, target_name.size() - suffix_len);
      if (fragment_buffers_.count(base_name)) {
        return base_name;
      }
    }
    return target_name;
  }

  void AddRowReduction(const std::string& target_name, const std::string& kind) {
    const std::string canonical_target = CanonicalReductionTarget(target_name);
    const std::string key = canonical_target + ":" + kind;
    if (seen_row_reductions_.insert(key).second) {
      row_reduction_targets_.push_back({canonical_target, kind});
    }
  }

  void AddSelectionTarget(const std::string& target_name) {
    const std::string canonical_target = CanonicalReductionTarget(target_name);
    if (seen_selection_targets_.insert(canonical_target).second) {
      selection_target_order_.push_back(canonical_target);
    }
  }

  void AddUpdateSources(const std::string& target_name,
                        const std::unordered_set<std::string>& source_names) {
    const std::string canonical_target = CanonicalReductionTarget(target_name);
    if (update_sources_.find(canonical_target) == update_sources_.end()) {
      update_source_target_order_.push_back(canonical_target);
    }
    auto& seen = update_sources_[canonical_target];
    auto& order = update_source_order_[canonical_target];
    for (const auto& source_name : source_names) {
      if (seen.insert(source_name).second) {
        order.push_back(source_name);
      }
    }
  }

  std::unordered_map<std::string, BufferInfo> fragment_buffers_;
  std::vector<std::string> fragment_buffer_order_;

  std::unordered_set<std::string> seen_ops_;
  std::vector<std::string> op_order_;
  std::unordered_set<std::string> seen_pointwise_ops_;
  std::vector<std::string> pointwise_op_order_;

  std::vector<std::pair<std::string, std::string>> row_reduction_targets_;
  std::unordered_set<std::string> seen_row_reductions_;
  std::unordered_set<std::string> seen_row_broadcast_sources_;
  std::vector<std::string> row_broadcast_sources_;
  std::unordered_set<std::string> seen_selection_targets_;
  std::vector<std::string> selection_target_order_;
  std::unordered_map<std::string, std::unordered_set<std::string>> update_sources_;
  std::unordered_map<std::string, std::vector<std::string>> update_source_order_;
  std::vector<std::string> update_source_target_order_;

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
