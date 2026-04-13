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
 * \file split_blackhole_kernel.cc
 * \brief SplitBlackholeKernels: annotate statements with segment kind
 *        for 3-kernel (reader/compute/writer) GEMM.
 *
 * Only activates when the function body contains a compute op.
 * Pure-copy functions are returned unchanged.
 *
 * Annotation format per annotated stmt:
 *   AttrStmt(node=StringImm("blackhole.segment_kind"),
 *            attr_key="blackhole.segment_kind",
 *            value=StringImm("reader"|"compute"|"writer"),
 *            body=original_stmt)
 *
 * Classification of top-level statements:
 *   ForNode  with copy_semantics.direction == "dram_to_cb"        -> reader
 *   ForNode  with copy_semantics.kind == "fused_staged_copy"       -> reader
 *   EvaluateNode  with tl.tileop.gemm_py op                        -> compute
 *   ForNode  with copy_semantics.direction == "cb_to_dram"         -> writer
 *   AllocateNode / anything else                                    -> passthrough
 */

#include "split_blackhole_kernel.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <string>
#include <vector>

#include "../../3rdparty/tvm/src/runtime/thread_storage_scope.h"
#include "common/blackhole_utils.h"
#include "common/blackhole_runtime_arg_schema.h"
#include "common/companion_base.h"

namespace tvm {
namespace tl {

using tir::AttrStmt;
using tir::EvaluateNode;
using tir::ForNode;
using tir::PrimFunc;
using tir::SeqStmt;
using tir::Stmt;
using tir::StmtVisitor;
using tir::StringImm;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

// Attr key produced by AnnotateBlackholeCopySemantics
static constexpr const char* kCopySemantics = "blackhole.copy_semantics";

// Attr key we emit
static constexpr const char* kSegmentKind = "blackhole.segment_kind";

static std::string MakeBlackholeRuntimeArgIdentity(const std::string& kind, const std::string& name,
                                                   const std::string& buffer_name = "") {
  if (!buffer_name.empty()) {
    return kind + ":" + buffer_name;
  }
  return !kind.empty() ? kind : name;
}

static Map<String, Any> MakePerWorkArgSpec(const std::string& arg_kind,
                                           const std::string& value_kind,
                                           const std::string& buffer_name = "",
                                           uint32_t constant_value = 0) {
  Map<String, Any> spec;
  spec.Set(String(blackhole_runtime_arg_schema::kArgKind), String(arg_kind));
  spec.Set(String(blackhole_runtime_arg_schema::kArgIdentity),
           String(MakeBlackholeRuntimeArgIdentity(arg_kind, arg_kind, buffer_name)));
  if (!buffer_name.empty()) {
    spec.Set(String(blackhole_runtime_arg_schema::kBuffer), String(buffer_name));
  }
  spec.Set(String(blackhole_runtime_arg_schema::kValueKind), String(value_kind));
  if (value_kind == blackhole_runtime_arg_schema::kValueConstant) {
    spec.Set(String(blackhole_runtime_arg_schema::kConstantValue),
             Integer(static_cast<int64_t>(constant_value)));
  }
  return spec;
}

// ------------------------------------------------------------------
// Helpers to read annotation fields from a Map<String, Any>
// ------------------------------------------------------------------

static std::string GetStringField(const Map<String, Any>& ann, const std::string& key,
                                  const std::string& def = "") {
  auto opt = ann.Get(String(key));
  if (!opt.has_value()) return def;
  auto str_opt = opt.value().try_cast<String>();
  return str_opt.has_value() ? std::string(str_opt.value()) : def;
}

static std::string GetBufferNameField(const Map<String, Any>& ann, const char* name_key,
                                      const char* ref_key) {
  if (auto opt = ann.Get(String(ref_key)); opt.has_value()) {
    auto buffer = opt.value().try_cast<tir::Buffer>();
    if (buffer.has_value()) {
      return BufferIdentityName(buffer.value());
    }
  }
  return GetStringField(ann, name_key);
}

static std::string GetStorageScope(const tir::Buffer& buffer) {
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

static bool IsDRAMScope(const std::string& scope) {
  return scope.empty() || scope == "global";
}

static bool IsAccumulatorLikeScope(const std::string& scope) {
  if (scope.rfind("local", 0) == 0) return true;
  auto s = runtime::StorageScope::Create(scope);
  return s.rank == runtime::StorageRank::kBlackholeAccumulator;
}

static bool FindWriterOutputBuffer(const Stmt& stmt, std::string* output_buf_name_out) {
  struct WriterCopyFinder : tir::StmtVisitor {
    std::string output_buf_name;
    bool found{false};

    void VisitStmt_(const tir::BufferStoreNode* op) final {
      if (found) return;
      const auto* load = op->value.as<tir::BufferLoadNode>();
      if (!load) return;
      std::string dst_scope = GetStorageScope(op->buffer);
      std::string src_scope = GetStorageScope(load->buffer);
      if (IsDRAMScope(dst_scope) && IsAccumulatorLikeScope(src_scope)) {
        output_buf_name = op->buffer->name;
        found = true;
      }
    }
  };

  WriterCopyFinder finder;
  finder(stmt);
  if (!finder.found) {
    return false;
  }
  if (output_buf_name_out) {
    *output_buf_name_out = finder.output_buf_name;
  }
  return true;
}

// ------------------------------------------------------------------
// Check whether a Stmt (or any descendant) contains a gemm_py call.
// We only need to detect its presence to decide whether to activate.
// ------------------------------------------------------------------

class ComputeOpFinder : public StmtVisitor {
 public:
  bool found = false;

  void VisitStmt_(const EvaluateNode* op) final {
    if (found) return;
    if (const auto* call = op->value.as<tir::CallNode>()) {
      if (const auto* op_node = call->op.as<OpNode>()) {
        if (op_node->name == "tl.tileop.gemm_py") {
          found = true;
          return;
        }
      }
    }
  }
};

static bool HasComputeOp(const Stmt& body) {
  ComputeOpFinder finder;
  finder(body);
  return finder.found;
}

// ------------------------------------------------------------------
// Collect info about compute ops so the annotator can distinguish segment kinds.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Classify a single top-level statement and return its segment kind.
// Returns "" for passthrough (no annotation).
//
// past_compute: true if a compute stmt has already been seen in this function.
//   Used to classify copies that appear after gemm as "writer" even when their
//   copy direction isn't recognized as cb_to_dram (e.g. local.fragment → global).
// ------------------------------------------------------------------

static std::string ClassifyStmt(const Stmt& stmt,
                                 std::string* input_buf_name_out,
                                 std::string* output_buf_name_out,
                                 bool past_compute) {
  if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
    return ClassifyStmt(attr->body, input_buf_name_out, output_buf_name_out, past_compute);
  }

  if (const auto* for_node = stmt.as<ForNode>()) {
    // Check for blackhole.copy_semantics annotation
    auto ann = for_node->annotations.Get(String(kCopySemantics));
    if (ann.has_value()) {
      Map<String, Any> ann_map =
          ann.value().as<Map<String, Any>>().value_or(Map<String, Any>());
      if (!ann_map.empty()) {
        std::string direction = GetStringField(ann_map, schema_key::kDirection);
        std::string kind      = GetStringField(ann_map, schema_key::kKind);
        std::string dst_scope = GetStringField(ann_map, "dst_scope");

        if (direction == "dram_to_cb") {
          // reader: captures DRAM source buffer name
          if (input_buf_name_out) {
            *input_buf_name_out =
                GetBufferNameField(ann_map, schema_key::kSrcBuffer, schema_key::kSrcBufferRef);
          }
          return "reader";
        }
        if (direction == "cb_to_dram") {
          // writer: captures DRAM destination buffer name
          if (output_buf_name_out) {
            *output_buf_name_out =
                GetBufferNameField(ann_map, schema_key::kDstBuffer, schema_key::kDstBufferRef);
          }
          return "writer";
        }
        if (kind == "fused_staged_copy") {
          // treat as reader (dram→cb side) for now
          if (input_buf_name_out) {
            *input_buf_name_out =
                GetBufferNameField(ann_map, schema_key::kSrcBuffer, schema_key::kSrcBufferRef);
          }
          return "reader";
        }
        // After a compute op, any copy with a global/DRAM destination is the writer.
        // This handles local.fragment → global (GEMM output copy).
        if (past_compute && (dst_scope.empty() || dst_scope == "global")) {
          if (output_buf_name_out) {
            *output_buf_name_out =
                GetBufferNameField(ann_map, schema_key::kDstBuffer, schema_key::kDstBufferRef);
          }
          return "writer";
        }
      }
    }
    if (past_compute && FindWriterOutputBuffer(stmt, output_buf_name_out)) {
      return "writer";
    }
    return "";
  }

  if (past_compute && FindWriterOutputBuffer(stmt, output_buf_name_out)) {
    return "writer";
  }

  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<tir::CallNode>()) {
      if (const auto* op_node = call->op.as<OpNode>()) {
        if (op_node->name == "tl.tileop.gemm_py") {
          return "compute";
        }
      }
    }
    return "";
  }

  return "";  // AllocateNode and anything else: passthrough
}

// ------------------------------------------------------------------
// Build the canonical reader/compute/writer segment payloads for GEMM.
// ------------------------------------------------------------------

static void StoreGemmSegmentPlan(PrimFunc& func,
                                 const std::vector<std::string>& input_buf_names,
                                 const std::string& output_buf_name) {
  (void)func;
  (void)input_buf_names;
  (void)output_buf_name;
}

// ------------------------------------------------------------------
// StmtMutator that annotates stmts inside the innermost SeqStmt
// that contains classifiable ops (copies + compute).
// ------------------------------------------------------------------

class SegmentAnnotator : public tir::StmtMutator {
 public:
  std::vector<std::string> input_buf_names;
  std::string output_buf_name;
  bool found_and_annotated = false;

  Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
    // Check if this SeqStmt contains classifiable stmts
    bool has_classify = false;
    for (const Stmt& s : op->seq) {
      std::string dummy_in, dummy_out;
      if (!ClassifyStmt(s, &dummy_in, &dummy_out, false).empty()) {
        has_classify = true;
        break;
      }
    }

    if (!has_classify) {
      // Recurse into children to find a deeper SeqStmt
      return StmtMutator::VisitStmt_(op);
    }

    // Annotate this SeqStmt's children
    found_and_annotated = true;
    bool past_compute = false;
    Array<Stmt> new_seq;
    new_seq.reserve(op->seq.size());
    for (const Stmt& stmt : op->seq) {
      std::string input_name, output_name;
      std::string kind = ClassifyStmt(stmt, &input_name, &output_name, past_compute);

      if (kind == "compute") past_compute = true;

      if (kind.empty()) {
        new_seq.push_back(stmt);
        continue;
      }

      if (!input_name.empty()) input_buf_names.push_back(input_name);
      if (!output_name.empty()) output_buf_name = output_name;

      new_seq.push_back(
          AttrStmt(StringImm(kSegmentKind), kSegmentKind,
                   StringImm(kind), stmt));
    }
    return SeqStmt(new_seq);
  }
};

// ------------------------------------------------------------------
// Main transform
// ------------------------------------------------------------------

static PrimFunc TransformFunc(const PrimFunc& func) {
  // 1. Check if this function has any compute op at all
  if (!HasComputeOp(func->body)) {
    return func;  // pure-copy path, leave unchanged
  }

  // 2. Annotate stmts using the mutator (finds the right SeqStmt level)
  SegmentAnnotator annotator;
  Stmt new_body = annotator(func->body);

  if (!annotator.found_and_annotated) {
    return func;  // Nothing to annotate
  }

  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = new_body;

  if (annotator.output_buf_name.empty()) {
    FindWriterOutputBuffer(new_body, &annotator.output_buf_name);
  }

  // 3. Write segment plan
  StoreGemmSegmentPlan(new_func, annotator.input_buf_names, annotator.output_buf_name);

  return new_func;
}

// ------------------------------------------------------------------
// Pass registration
// ------------------------------------------------------------------

tir::transform::Pass SplitBlackholeKernelPass() {
  auto fpass = [](PrimFunc func, IRModule /*m*/,
                  tir::transform::PassContext /*ctx*/) -> PrimFunc {
    return TransformFunc(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0,
                                             "tl.transform.SplitBlackholeKernel", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SplitBlackholeKernel", SplitBlackholeKernelPass);
}

}  // namespace tl
}  // namespace tvm
