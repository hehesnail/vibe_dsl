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
 * \file analyze_blackhole_pipeline_stages.cc
 * \brief Analyze split-after Blackhole pipelined loop structure and emit a structured IR attr.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using tir::BlockNode;
using tir::Buffer;
using tir::BufferLoadNode;
using tir::BufferStoreNode;
using tir::ForNode;
using tir::PrimFunc;
using tir::StmtExprVisitor;
using tvm::DictAttrs;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

struct BufferInfo {
  Buffer buffer;
  std::string scope;
};

bool IsTrackedLocalScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment";
}

bool IsSharedScope(const std::string& scope) { return scope.rfind("shared", 0) == 0; }

class PipelineStageAnalyzer final : public StmtExprVisitor {
 public:
  void Analyze(const PrimFunc& func) { VisitStmt(func->body); }

  bool HasPipelineStages() const { return !stages_.empty(); }

  Array<Any> Encode() const {
    Array<Any> stages;
    for (const auto& stage : stages_) {
      Map<String, Any> stage_info;
      stage_info.Set("loop_var", String(stage.loop_var));
      stage_info.Set("num_stages", Integer(stage.num_stages));

      Array<Any> stage_local_buffers;
      for (const auto& name : stage.stage_local_buffer_names) {
        const auto& info = shared_buffers_.at(name);
        Map<String, Any> entry;
        entry.Set("name", String(name));
        entry.Set("scope", String(info.scope));
        stage_local_buffers.push_back(entry);
      }
      stage_info.Set("stage_local_buffers", stage_local_buffers);

      Array<Any> loop_carried_state;
      for (const auto& name : stage.loop_carried_state_names) {
        Map<String, Any> entry;
        entry.Set("name", String(name));
        loop_carried_state.push_back(entry);
      }
      stage_info.Set("loop_carried_state", loop_carried_state);
      stages.push_back(stage_info);
    }
    return stages;
  }

 private:
  struct StageInfo {
    std::string loop_var;
    int64_t num_stages{0};
    std::vector<std::string> stage_local_buffer_names;
    std::vector<std::string> loop_carried_state_names;
  };

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "tilelang_root") {
      for (const Buffer& buffer : op->alloc_buffers) {
        const std::string scope = buffer.scope();
        const std::string name = buffer->name;
        if (IsSharedScope(scope)) {
          shared_buffers_.emplace(name, BufferInfo{buffer, scope});
        } else if (IsTrackedLocalScope(scope)) {
          tracked_local_buffers_.emplace(name, BufferInfo{buffer, scope});
        }
      }

      if (const auto* seq = op->body.as<tir::SeqStmtNode>()) {
        bool saw_pipeline_loop = false;
        for (const auto& stmt : seq->seq) {
          if (!saw_pipeline_loop) {
            if (const auto* loop = stmt.as<ForNode>()) {
              if (auto maybe_num_stages = GetNumStages(loop)) {
                saw_pipeline_loop = true;
                current_stage_ = StageInfo{loop->loop_var->name_hint, maybe_num_stages.value(), {}, {}};
                pre_loop_stmt_ = false;
                inside_pipeline_loop_ = true;
                VisitStmt(stmt);
                inside_pipeline_loop_ = false;
                FinalizeCurrentStage();
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
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    const bool prev_inside = inside_pipeline_loop_;
    if (current_stage_.has_value() && op->loop_var->name_hint == current_stage_->loop_var) {
      inside_pipeline_loop_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
    inside_pipeline_loop_ = prev_inside;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    const std::string name = op->buffer->name;
    if (tracked_local_buffers_.count(name)) {
      if (pre_loop_stmt_) {
        pre_loop_writes_.insert(name);
      }
      if (inside_pipeline_loop_) {
        in_loop_writes_.insert(name);
      }
    }
    if (inside_pipeline_loop_ && current_stage_.has_value() && shared_buffers_.count(name) &&
        seen_stage_local_buffers_.insert(name).second) {
      current_stage_->stage_local_buffer_names.push_back(name);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const std::string name = op->buffer->name;
    if (tracked_local_buffers_.count(name)) {
      if (inside_pipeline_loop_) {
        in_loop_reads_.insert(name);
      }
      if (post_loop_stmt_) {
        post_loop_reads_.insert(name);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::optional<int64_t> GetNumStages(const ForNode* loop) const {
    if (!loop->annotations.defined()) {
      return std::nullopt;
    }
    if (auto value = loop->annotations.Get("num_stages")) {
      if (const auto* imm = value.value().as<IntImmNode>()) {
        return imm->value;
      }
    }
    return std::nullopt;
  }

  void FinalizeCurrentStage() {
    ICHECK(current_stage_.has_value());
    for (const auto& kv : tracked_local_buffers_) {
      const std::string& name = kv.first;
      const bool carried =
          in_loop_writes_.count(name) &&
          (pre_loop_writes_.count(name) || in_loop_reads_.count(name) || post_loop_reads_.count(name));
      if (carried) {
        current_stage_->loop_carried_state_names.push_back(name);
      }
    }
    stages_.push_back(*current_stage_);
    current_stage_.reset();
    seen_stage_local_buffers_.clear();
    pre_loop_writes_.clear();
    in_loop_writes_.clear();
    in_loop_reads_.clear();
    post_loop_reads_.clear();
  }

  std::unordered_map<std::string, BufferInfo> tracked_local_buffers_;
  std::unordered_map<std::string, BufferInfo> shared_buffers_;

  bool pre_loop_stmt_ = false;
  bool inside_pipeline_loop_ = false;
  bool post_loop_stmt_ = false;

  std::unordered_set<std::string> pre_loop_writes_;
  std::unordered_set<std::string> in_loop_writes_;
  std::unordered_set<std::string> in_loop_reads_;
  std::unordered_set<std::string> post_loop_reads_;
  std::unordered_set<std::string> seen_stage_local_buffers_;

  std::optional<StageInfo> current_stage_;
  std::vector<StageInfo> stages_;
};

}  // namespace

tir::transform::Pass AnalyzeBlackholePipelineStagesPass() {
  auto fpass = [](PrimFunc func, IRModule, tir::transform::PassContext) -> PrimFunc {
    PipelineStageAnalyzer analyzer;
    analyzer.Analyze(func);
    if (!analyzer.HasPipelineStages()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.pipeline_stages", analyzer.Encode());

    PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AnalyzeBlackholePipelineStages", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeBlackholePipelineStages",
                        AnalyzeBlackholePipelineStagesPass);
}

}  // namespace tl
}  // namespace tvm
