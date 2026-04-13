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
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../layout/layout.h"
#include "common/blackhole_utils.h"
#include "common/fragment_layout_contract_utils.h"
#include "common/fragment_region_analysis.h"

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

struct BufferInfo {
  Buffer buffer;
  std::string scope;
};

const tir::VarNode* BufferKey(const Buffer& buffer) { return BufferDataIdentity(buffer); }

std::string BufferName(const Buffer& buffer) { return BufferIdentityName(buffer); }

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

bool IsStageLocalScope(const std::string& scope) {
  return scope.rfind("shared", 0) == 0 || scope.rfind("blackhole.cb.", 0) == 0;
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

bool IsLoadFromBuffer(const PrimExpr& expr, const Buffer& buffer) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return SameBufferIdentity(load->buffer, buffer);
  }
  return false;
}

bool IsLoadFromNonFragmentLocal(const PrimExpr& expr,
                                const std::unordered_map<const tir::VarNode*, BufferInfo>&
                                    fragment_buffers) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    const auto* key = BufferKey(load->buffer);
    return load->buffer.scope() == "local" && key != nullptr && !fragment_buffers.count(key);
  }
  return false;
}

bool IsLoadFromFragmentBuffer(const PrimExpr& expr,
                              const std::unordered_map<const tir::VarNode*, BufferInfo>&
                                  fragment_buffers) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    const auto* key = BufferKey(load->buffer);
    return key != nullptr && fragment_buffers.count(key);
  }
  return false;
}

class FragmentRegionAnalyzer final : public StmtExprVisitor {
 public:
  void Analyze(const PrimFunc& func) {
    AnalyzeTopLevel(func->body);
  }

  bool HasRegion() const { return !fragment_buffer_order_.empty() || !seen_ops_.empty(); }

  Map<String, Any> EncodeSingleRegion() const {
    MaterializeFragmentLayoutContractsFromRegionEvidence();
    Map<String, Any> region;

    Array<Any> fragment_buffers;
    for (const auto* key : fragment_buffer_order_) {
      auto info_it = fragment_buffers_.find(key);
      if (info_it == fragment_buffers_.end()) {
        continue;
      }
      const auto& info = info_it->second;
      Map<String, Any> entry;
      entry.Set(schema_key::kName, String(BufferName(info.buffer)));
      entry.Set(schema_key::kBuffer, info.buffer);
      entry.Set(schema_key::kScope, String(info.scope));
      entry.Set(schema_key::kIsInteger,
                Integer(info.buffer->dtype.is_int() || info.buffer->dtype.is_uint()));
      fragment_buffers.push_back(entry);
    }
    region.Set(manifest_key::kFragmentBuffers, fragment_buffers);

    Array<Any> fragment_layout_contracts;
    for (const auto* key : fragment_buffer_order_) {
      auto contract_it = fragment_layout_contracts_.find(key);
      if (contract_it == fragment_layout_contracts_.end()) {
        continue;
      }
      fragment_layout_contracts.push_back(contract_it->second);
    }
    if (!fragment_layout_contracts.empty()) {
      region.Set(String(schema_key::kFragmentLayoutContracts), fragment_layout_contracts);
    }

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
    for (const auto* target_key : row_reduction_targets_) {
      auto buffer_it = fragment_buffers_.find(target_key);
      if (buffer_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& buffer = buffer_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kTarget, String(BufferName(buffer)));
      entry.Set(schema_key::kTargetBuffer, buffer);
      if (auto kind_it = row_reduction_kind_.find(target_key);
          kind_it != row_reduction_kind_.end() && !kind_it->second.empty()) {
        entry.Set(schema_key::kKind, String(kind_it->second));
      }
      row_reductions.push_back(entry);
    }
    region.Set(manifest_key::kRowReductions, row_reductions);

    Array<Any> arg_reduce_targets;
    for (const auto* key : BuildArgReduceTargets()) {
      auto buffer_it = fragment_buffers_.find(key);
      if (buffer_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& buffer = buffer_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kName, String(BufferName(buffer)));
      entry.Set(schema_key::kBuffer, buffer);
      arg_reduce_targets.push_back(entry);
    }
    region.Set(manifest_key::kArgReduceTargets, arg_reduce_targets);

    Array<Any> row_broadcasts;
    for (const auto* key : row_broadcast_sources_) {
      auto buffer_it = fragment_buffers_.find(key);
      if (buffer_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& buffer = buffer_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kSource, String(BufferName(buffer)));
      entry.Set(schema_key::kBuffer, buffer);
      row_broadcasts.push_back(entry);
    }
    region.Set("row_broadcasts", row_broadcasts);

    Array<Any> selection_targets;
    for (const auto* key : selection_target_order_) {
      auto buffer_it = fragment_buffers_.find(key);
      if (buffer_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& buffer = buffer_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kName, String(BufferName(buffer)));
      entry.Set(schema_key::kBuffer, buffer);
      selection_targets.push_back(entry);
    }
    region.Set(manifest_key::kSelectionTargets, selection_targets);

    Array<Any> selection_pairs;
    for (const auto& pair : BuildSelectionPairs()) {
      auto value_it = fragment_buffers_.find(pair.value_target);
      auto companion_it = fragment_buffers_.find(pair.companion_target);
      if (value_it == fragment_buffers_.end() || companion_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& value_buffer = value_it->second.buffer;
      const Buffer& companion_buffer = companion_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kValueTarget, String(BufferName(value_buffer)));
      entry.Set(schema_key::kValueBuffer, value_buffer);
      entry.Set(schema_key::kCompanionTarget, String(BufferName(companion_buffer)));
      entry.Set(schema_key::kCompanionBuffer, companion_buffer);
      Array<Any> sources;
      Array<Any> source_buffers;
      for (const auto& source : pair.shared_sources) {
        auto source_it = fragment_buffers_.find(source);
        if (source_it == fragment_buffers_.end()) {
          continue;
        }
        const Buffer& source_buffer = source_it->second.buffer;
        sources.push_back(String(BufferName(source_buffer)));
        source_buffers.push_back(source_buffer);
      }
      entry.Set(schema_key::kSourceStates, sources);
      entry.Set(schema_key::kSourceBuffers, source_buffers);
      selection_pairs.push_back(entry);
    }
    region.Set(manifest_key::kSelectionPairs, selection_pairs);

    Array<Any> update_sources;
    for (const auto* target : update_source_target_order_) {
      auto target_it = fragment_buffers_.find(target);
      if (target_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& target_buffer = target_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kTarget, String(BufferName(target_buffer)));
      entry.Set(schema_key::kTargetBuffer, target_buffer);
      Array<Any> sources;
      Array<Any> source_buffers;
      for (const auto* source : update_source_order_.at(target)) {
        auto source_it = fragment_buffers_.find(source);
        if (source_it == fragment_buffers_.end()) {
          continue;
        }
        const Buffer& source_buffer = source_it->second.buffer;
        sources.push_back(String(BufferName(source_buffer)));
        source_buffers.push_back(source_buffer);
      }
      entry.Set(schema_key::kSources, sources);
      entry.Set(schema_key::kSourceBuffers, source_buffers);
      update_sources.push_back(entry);
    }
    region.Set(manifest_key::kUpdateSources, update_sources);

    Array<Any> loop_carried_state;
    for (const auto* key : loop_carried_order_) {
      auto buffer_it = fragment_buffers_.find(key);
      if (buffer_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& buffer = buffer_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kName, String(BufferName(buffer)));
      entry.Set(schema_key::kBuffer, buffer);
      loop_carried_state.push_back(entry);
    }
    region.Set(manifest_key::kLoopCarriedState, loop_carried_state);

    Array<Any> recurrence_edges;
    for (const auto* target : loop_carried_order_) {
      auto it = update_source_order_.find(target);
      if (it == update_source_order_.end() || it->second.empty()) {
        continue;
      }
      auto target_it = fragment_buffers_.find(target);
      if (target_it == fragment_buffers_.end()) {
        continue;
      }
      const Buffer& target_buffer = target_it->second.buffer;
      Map<String, Any> entry;
      entry.Set(schema_key::kTarget, String(BufferName(target_buffer)));
      entry.Set(schema_key::kTargetBuffer, target_buffer);
      Array<Any> sources;
      Array<Any> source_buffers;
      for (const auto* source : it->second) {
        auto source_it = fragment_buffers_.find(source);
        if (source_it == fragment_buffers_.end()) {
          continue;
        }
        const Buffer& source_buffer = source_it->second.buffer;
        sources.push_back(String(BufferName(source_buffer)));
        source_buffers.push_back(source_buffer);
      }
      entry.Set(schema_key::kSourceStates, sources);
      entry.Set(schema_key::kSourceBuffers, source_buffers);
      recurrence_edges.push_back(entry);
    }
    region.Set(manifest_key::kRecurrenceEdges, recurrence_edges);

    return region;
  }

 private:
  void RegisterFragmentBuffer(const Buffer& buffer, bool allow_plain_local = false) {
    (void)allow_plain_local;
    const std::string scope = buffer.scope();
    if (!IsFragmentLikeScope(scope)) {
      return;
    }
    const auto* key = BufferKey(buffer);
    if (key == nullptr) {
      return;
    }
    if (temp_reduction_buffers_.count(key)) {
      return;
    }
    if (fragment_buffers_.emplace(key, BufferInfo{buffer, scope}).second) {
      fragment_buffer_order_.push_back(key);
    }
  }

  void UnregisterFragmentBuffer(const tir::VarNode* key) {
    if (key == nullptr) {
      return;
    }
    fragment_buffers_.erase(key);
    fragment_buffer_order_.erase(
        std::remove(fragment_buffer_order_.begin(), fragment_buffer_order_.end(), key),
        fragment_buffer_order_.end());
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
        if (!seen_pipeline_loop) {
          if (const auto* loop = stmt.as<ForNode>()) {
            std::optional<int64_t> maybe_num_stages = GetNumStages(loop);
            if (!maybe_num_stages.has_value()) {
              maybe_num_stages = InferStageCountFromStmt(stmt);
            }
            if (maybe_num_stages.has_value()) {
              seen_pipeline_loop = true;
              inside_pipeline_loop_ = true;
              VisitStmt(stmt);
              inside_pipeline_loop_ = false;
              continue;
            }
          }
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
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map_any = op->annotations.Get(attr::kLayoutMap);
      if (layout_map_any) {
        auto layout_map = layout_map_any->as<Map<Buffer, Layout>>();
        if (layout_map && layout_map.value().defined()) {
          for (const auto& [buffer, layout] : layout_map.value()) {
            const std::string scope = buffer.scope();
            if (scope == "local" || scope == "local.fragment" || scope == "blackhole.acc") {
              if (const auto* key = BufferKey(buffer); key != nullptr) {
                layout_fragment_buffers_.insert(key);
                if (auto contract = TryBuildFragmentLayoutContract(buffer, layout)) {
                  fragment_layout_contracts_[key] = contract.value();
                }
              }
            }
          }
        }
      }
    }
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
    if (!inside_pipeline_loop_) {
      std::optional<int64_t> maybe_num_stages = GetNumStages(op);
      if (!maybe_num_stages.has_value()) {
        maybe_num_stages = InferStageCountFromStmt(GetRef<Stmt>(op));
      }
      if (maybe_num_stages.has_value()) {
        inside_pipeline_loop_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    inside_pipeline_loop_ = prev_inside;
  }

  std::optional<int64_t> GetNumStages(const ForNode* loop) const {
    if (!loop->annotations.defined()) {
      return std::nullopt;
    }
    for (const char* key : {"num_stages", "tl_pipelined_num_stages"}) {
      if (auto value = loop->annotations.Get(key)) {
        if (const auto* imm = value.value().as<tir::IntImmNode>()) {
          return imm->value;
        }
      }
    }
    return std::nullopt;
  }

  std::optional<int64_t> InferStageCountFromStmt(const Stmt& stmt) const {
    std::optional<int64_t> inferred;
    tir::PostOrderVisit(stmt, [&inferred](const ObjectRef& node) {
      auto update_from_buffer = [&inferred](const Buffer& buffer) {
        const std::string scope = buffer.scope();
        if (!IsStageLocalScope(scope) || buffer->shape.size() < 3) {
          return;
        }
        if (const auto* imm = buffer->shape[0].as<tir::IntImmNode>()) {
          if (imm->value > 0) {
            inferred = imm->value;
          }
        }
      };
      if (const auto* store = node.as<BufferStoreNode>()) {
        update_from_buffer(store->buffer);
      } else if (const auto* load = node.as<BufferLoadNode>()) {
        update_from_buffer(load->buffer);
      }
    });
    return inferred;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    const auto* target_key = BufferKey(op->buffer);
    if (op->buffer.scope() == "local" && target_key != nullptr &&
        !layout_fragment_buffers_.count(target_key) &&
        DetectTempReductionBuffer(op->buffer, op->value)) {
      UnregisterFragmentBuffer(BufferKey(op->buffer));
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    RegisterFragmentBuffer(op->buffer);
    if (target_key != nullptr && fragment_buffers_.count(target_key)) {
      if (pre_loop_stmt_) {
        pre_loop_writes_.insert(target_key);
      }
      if (inside_pipeline_loop_) {
        in_loop_writes_.insert(target_key);
      }
      if (post_loop_stmt_) {
        post_loop_writes_.insert(target_key);
      }

      DetectOpsAndRelationships(op->buffer, op->value, op->indices);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    RegisterFragmentBuffer(op->buffer);
    const auto* source_key = BufferKey(op->buffer);
    if (source_key != nullptr && fragment_buffers_.count(source_key)) {
      if (pre_loop_stmt_) {
        pre_loop_reads_.insert(source_key);
      }
      if (inside_pipeline_loop_) {
        in_loop_reads_.insert(source_key);
      }
      if (post_loop_stmt_) {
        post_loop_reads_.insert(source_key);
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

  bool DetectTempReductionBuffer(const Buffer& target_buffer, const PrimExpr& value) {
    const auto* target_key = BufferKey(target_buffer);
    if (target_key == nullptr) {
      return false;
    }
    bool has_local_max_reduction = false;
    bool has_local_sum_reduction = false;
    auto mark_local_reduction = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                                    const char* kind) {
      const bool lhs_self = IsLoadFromBuffer(lhs, target_buffer);
      const bool rhs_self = IsLoadFromBuffer(rhs, target_buffer);
      const bool lhs_fragment = IsLoadFromFragmentBuffer(lhs, fragment_buffers_);
      const bool rhs_fragment = IsLoadFromFragmentBuffer(rhs, fragment_buffers_);
      if (!((lhs_self && rhs_fragment) || (rhs_self && lhs_fragment))) {
        return;
      }
      if (std::string(kind) == "max") {
        has_local_max_reduction = true;
      } else if (std::string(kind) == "sum") {
        has_local_sum_reduction = true;
      }
    };
    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          const std::string op_name = op_node->name;
          if (call->args.size() == 2) {
            if (op_name == "tir.max") {
              mark_local_reduction(call->args[0], call->args[1], "max");
            } else if (op_name == "tir.add") {
              mark_local_reduction(call->args[0], call->args[1], "sum");
            }
          }
        }
      } else if (const auto* max = node.as<MaxNode>()) {
        mark_local_reduction(max->a, max->b, "max");
      } else if (const auto* add = node.as<AddNode>()) {
        mark_local_reduction(add->a, add->b, "sum");
      }
    });
    if (has_local_max_reduction) {
      temp_reduction_buffers_[target_key] = "max";
      return true;
    }
    if (has_local_sum_reduction) {
      temp_reduction_buffers_[target_key] = "sum";
      return true;
    }
    return false;
  }

  void DetectOpsAndRelationships(const Buffer& target_buffer, const PrimExpr& value,
                                 const Array<PrimExpr>& store_indices) {
    const auto* target_key = BufferKey(target_buffer);
    if (target_key == nullptr) {
      return;
    }
    bool saw_pointwise = false;
    bool saw_floor_div_broadcast = false;
    bool saw_rank_broadcast = false;
    bool saw_scalar_fragment_broadcast = false;
    bool saw_direct_fragment_max_reduction = false;
    bool saw_direct_fragment_sum_reduction = false;
    std::unordered_set<const tir::VarNode*> local_sources;
    bool has_self_max_with_temp = false;
    std::string temp_reduction_kind;
    const int64_t target_elements = StaticNumElements(fragment_buffers_.at(target_key).buffer);
    auto is_self_reduce_from_larger_fragment = [&](const PrimExpr& self_expr,
                                                   const PrimExpr& fragment_expr) {
      const auto* fragment_load = fragment_expr.as<BufferLoadNode>();
      if (!IsLoadFromBuffer(self_expr, target_buffer) || fragment_load == nullptr) {
        return false;
      }
      const auto* source_key = BufferKey(fragment_load->buffer);
      if (source_key == nullptr || !fragment_buffers_.count(source_key)) {
        return false;
      }
      const int64_t source_elements = StaticNumElements(fragment_buffers_.at(source_key).buffer);
      return source_elements > target_elements;
    };

    tir::PostOrderVisit(value, [&](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          const std::string op_name = op_node->name;
          if (op_name == "tir.exp2" || op_name == "tir.if_then_else") {
            saw_pointwise = true;
            AddPointwiseOp(op_name == "tir.exp2" ? "exp2" : "if_then_else");
            if (op_name == "tir.if_then_else") {
              AddSelectionTarget(target_key);
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
            IsLoadFromBuffer(max->a, target_buffer) &&
            IsLoadFromNonFragmentLocal(max->b, fragment_buffers_);
        const bool rhs_self_lhs_temp =
            IsLoadFromBuffer(max->b, target_buffer) &&
            IsLoadFromNonFragmentLocal(max->a, fragment_buffers_);
        has_self_max_with_temp |= lhs_self_rhs_temp || rhs_self_lhs_temp;

        const bool lhs_self_rhs_fragment =
            IsLoadFromBuffer(max->a, target_buffer) &&
            IsLoadFromFragmentBuffer(max->b, fragment_buffers_);
        const bool rhs_self_lhs_fragment =
            IsLoadFromBuffer(max->b, target_buffer) &&
            IsLoadFromFragmentBuffer(max->a, fragment_buffers_);
        saw_direct_fragment_max_reduction |=
            is_self_reduce_from_larger_fragment(max->a, max->b) ||
            is_self_reduce_from_larger_fragment(max->b, max->a) ||
            ((target_elements == 1) && (lhs_self_rhs_fragment || rhs_self_lhs_fragment));
      } else if (const auto* add = node.as<AddNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("add");
        saw_direct_fragment_sum_reduction |=
            is_self_reduce_from_larger_fragment(add->a, add->b) ||
            is_self_reduce_from_larger_fragment(add->b, add->a);
      } else if (node.as<MulNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("mul");
      } else if (node.as<DivNode>()) {
        saw_pointwise = true;
        AddPointwiseOp("div");
      } else if (const auto* load = node.as<BufferLoadNode>()) {
        const auto* source_key = BufferKey(load->buffer);
        if (source_key == nullptr || source_key == target_key || !fragment_buffers_.count(source_key)) {
          if (const auto it = temp_reduction_buffers_.find(source_key);
              it != temp_reduction_buffers_.end()) {
            temp_reduction_kind = it->second;
          }
          return;
        }
        local_sources.insert(source_key);
        if (load->indices.size() < store_indices.size()) {
          saw_rank_broadcast = true;
        }
        const int64_t source_elements = StaticNumElements(fragment_buffers_.at(source_key).buffer);
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

    if (has_self_max_with_temp || temp_reduction_kind == "max" || saw_direct_fragment_max_reduction ||
        temp_reduction_kind == "sum" || saw_direct_fragment_sum_reduction) {
      AddOp("row_reduction");
      const std::string reduction_kind =
          (has_self_max_with_temp || temp_reduction_kind == "max" ||
           saw_direct_fragment_max_reduction)
              ? "max"
              : "sum";
      AddRowReduction(target_key, reduction_kind);
      for (const auto* source_key : local_sources) {
        if (!fragment_buffers_.count(source_key)) {
          continue;
        }
        const int64_t source_elements = StaticNumElements(fragment_buffers_.at(source_key).buffer);
        if (source_elements > target_elements) {
          AddRowReductionSource(source_key);
        }
      }
    }

    if (!local_sources.empty() &&
        (saw_floor_div_broadcast || saw_rank_broadcast || saw_scalar_fragment_broadcast)) {
      AddOp("row_broadcast");
      AddRowBroadcastDestination(target_key);
      for (const auto* source_key : local_sources) {
        if (seen_row_broadcast_sources_.insert(source_key).second) {
          row_broadcast_sources_.push_back(source_key);
        }
      }
    }

    if (!local_sources.empty()) {
      AddUpdateSources(target_key, local_sources);
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

  void AddRowReduction(const tir::VarNode* target_key, const std::string& kind) {
    if (seen_row_reductions_.insert(target_key).second) {
      row_reduction_targets_.push_back(target_key);
    }
    if (auto it = row_reduction_kind_.find(target_key);
        it == row_reduction_kind_.end() || it->second.empty()) {
      row_reduction_kind_[target_key] = kind;
    }
  }

  void AddSelectionTarget(const tir::VarNode* target_key) {
    if (seen_selection_targets_.insert(target_key).second) {
      selection_target_order_.push_back(target_key);
    }
  }

  void AddRowReductionSource(const tir::VarNode* source_key) {
    if (seen_row_reduction_sources_.insert(source_key).second) {
      row_reduction_sources_.push_back(source_key);
    }
  }

  void AddRowBroadcastDestination(const tir::VarNode* target_key) {
    if (seen_row_broadcast_destinations_.insert(target_key).second) {
      row_broadcast_destinations_.push_back(target_key);
    }
  }

  void AddUpdateSources(const tir::VarNode* target_key,
                        const std::unordered_set<const tir::VarNode*>& source_keys) {
    if (update_sources_.find(target_key) == update_sources_.end()) {
      update_source_target_order_.push_back(target_key);
    }
    auto& seen = update_sources_[target_key];
    auto& order = update_source_order_[target_key];
    for (const auto* source_key : source_keys) {
      if (seen.insert(source_key).second) {
        order.push_back(source_key);
      }
    }
  }

  struct SelectionPair {
    const tir::VarNode* value_target{nullptr};
    const tir::VarNode* companion_target{nullptr};
    std::vector<const tir::VarNode*> shared_sources;
  };

  std::vector<SelectionPair> BuildSelectionPairs() const {
    std::vector<SelectionPair> pairs;
    for (const auto* companion_target : selection_target_order_) {
      auto companion_it = update_source_order_.find(companion_target);
      if (companion_it == update_source_order_.end()) {
        continue;
      }
      int best_overlap = 0;
      const tir::VarNode* best_value_target = nullptr;
      std::vector<const tir::VarNode*> best_shared_sources;
      bool ambiguous = false;
      for (const auto* value_target : row_reduction_targets_) {
        if (value_target == companion_target) {
          continue;
        }
        auto value_it = update_source_order_.find(value_target);
        if (value_it == update_source_order_.end()) {
          continue;
        }
        std::vector<const tir::VarNode*> shared_sources;
        for (const auto* source : companion_it->second) {
          for (const auto* candidate_source : value_it->second) {
            if (source == candidate_source) {
              shared_sources.push_back(source);
              break;
            }
          }
        }
        const int overlap = static_cast<int>(shared_sources.size());
        if (overlap == 0) {
          continue;
        }
        if (overlap > best_overlap) {
          best_overlap = overlap;
          best_value_target = value_target;
          best_shared_sources = std::move(shared_sources);
          ambiguous = false;
        } else if (overlap == best_overlap) {
          ambiguous = true;
        }
      }
      if (best_overlap > 0 && !ambiguous) {
        pairs.push_back(SelectionPair{best_value_target, companion_target, best_shared_sources});
      }
    }
    return pairs;
  }

  std::vector<const tir::VarNode*> BuildArgReduceTargets() const {
    std::vector<const tir::VarNode*> targets;
    std::unordered_set<const tir::VarNode*> selection_like_sources(selection_target_order_.begin(),
                                                                   selection_target_order_.end());
    for (const auto& pair : BuildSelectionPairs()) {
      selection_like_sources.insert(pair.companion_target);
    }
    for (const auto* target : row_reduction_targets_) {
      auto it = update_source_order_.find(target);
      if (it == update_source_order_.end()) {
        continue;
      }
      for (const auto* source : it->second) {
        if (selection_like_sources.count(source)) {
          targets.push_back(target);
          break;
        }
      }
    }
    return targets;
  }

  void MaterializeFragmentLayoutContractsFromRegionEvidence() const {
    auto set_contract = [&](const tir::VarNode* key, const char* distribution_kind) {
      if (key == nullptr || fragment_layout_contracts_.count(key) || !fragment_buffers_.count(key)) {
        return;
      }
      const Buffer& buffer = fragment_buffers_.at(key).buffer;
      const std::string buffer_name = BufferName(buffer);
      const std::string scope = buffer.scope();
      if (buffer_name.empty() || !IsFragmentLikeScope(scope)) {
        return;
      }
      Map<String, Any> contract;
      contract.Set(String(schema_key::kBuffer), String(buffer_name));
      contract.Set(String(schema_key::kScope), String(scope));
      contract.Set(String(schema_key::kShape), EncodeFragmentContractShape(buffer->shape));
      contract.Set(String(schema_key::kDistributionKind), String(distribution_kind));
      contract.Set(String(schema_key::kStorageTopologyKind), String(fragment_layout::kLinear));
      fragment_layout_contracts_[key] = contract;
    };

    for (const auto* key : row_reduction_targets_) {
      set_contract(key, fragment_layout::kRowState);
    }
    for (const auto* key : row_broadcast_sources_) {
      set_contract(key, fragment_layout::kRowState);
    }
    for (const auto* key : row_reduction_sources_) {
      set_contract(key, fragment_layout::kGroupedRows);
    }
    for (const auto* key : row_broadcast_destinations_) {
      set_contract(key, fragment_layout::kGroupedRows);
    }
  }

  std::unordered_map<const tir::VarNode*, BufferInfo> fragment_buffers_;
  std::vector<const tir::VarNode*> fragment_buffer_order_;
  std::unordered_set<const tir::VarNode*> layout_fragment_buffers_;
  mutable std::unordered_map<const tir::VarNode*, Map<String, Any>> fragment_layout_contracts_;

  std::unordered_set<std::string> seen_ops_;
  std::vector<std::string> op_order_;
  std::unordered_set<std::string> seen_pointwise_ops_;
  std::vector<std::string> pointwise_op_order_;

  std::vector<const tir::VarNode*> row_reduction_targets_;
  std::unordered_set<const tir::VarNode*> seen_row_reductions_;
  std::unordered_map<const tir::VarNode*, std::string> row_reduction_kind_;
  std::vector<const tir::VarNode*> row_reduction_sources_;
  std::unordered_set<const tir::VarNode*> seen_row_reduction_sources_;
  std::unordered_set<const tir::VarNode*> seen_row_broadcast_sources_;
  std::vector<const tir::VarNode*> row_broadcast_sources_;
  std::unordered_set<const tir::VarNode*> seen_row_broadcast_destinations_;
  std::vector<const tir::VarNode*> row_broadcast_destinations_;
  std::unordered_set<const tir::VarNode*> seen_selection_targets_;
  std::vector<const tir::VarNode*> selection_target_order_;
  std::unordered_map<const tir::VarNode*, std::unordered_set<const tir::VarNode*>> update_sources_;
  std::unordered_map<const tir::VarNode*, std::vector<const tir::VarNode*>> update_source_order_;
  std::vector<const tir::VarNode*> update_source_target_order_;

  bool pre_loop_stmt_ = false;
  bool inside_pipeline_loop_ = false;
  bool post_loop_stmt_ = false;

  std::unordered_set<const tir::VarNode*> pre_loop_writes_;
  std::unordered_set<const tir::VarNode*> in_loop_writes_;
  std::unordered_set<const tir::VarNode*> post_loop_writes_;
  std::unordered_set<const tir::VarNode*> pre_loop_reads_;
  std::unordered_set<const tir::VarNode*> in_loop_reads_;
  std::unordered_set<const tir::VarNode*> post_loop_reads_;
  std::unordered_map<const tir::VarNode*, std::string> temp_reduction_buffers_;

  mutable std::vector<const tir::VarNode*> loop_carried_order_;

  void FinalizeLoopCarriedState() const {
    if (!loop_carried_order_.empty()) {
      return;
    }
    for (const auto* key : fragment_buffer_order_) {
      const bool carried = in_loop_writes_.count(key) &&
                           (pre_loop_writes_.count(key) || in_loop_reads_.count(key) ||
                            post_loop_reads_.count(key));
      if (carried) {
        loop_carried_order_.push_back(key);
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

Map<String, Any> AnalyzeBlackholeFragmentRegionEvidence(const PrimFunc& func) {
  FragmentRegionAnalyzer analyzer;
  analyzer.Analyze(func);
  if (!analyzer.HasRegion()) {
    return {};
  }
  return analyzer.Encode();
}

tir::transform::Pass AnalyzeBlackholeFragmentRegionsPass() {
  auto fpass = [](PrimFunc func, IRModule, tir::transform::PassContext) -> PrimFunc {
    Map<String, Any> encoded = AnalyzeBlackholeFragmentRegionEvidence(func);
    if (encoded.empty()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.fragment_regions", encoded["regions"]);
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
