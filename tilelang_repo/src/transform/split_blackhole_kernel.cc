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
 *        and emit blackhole.segment_plan for 3-kernel (reader/compute/writer) GEMM.
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
// Collect info about compute ops (for segment plan)
// ------------------------------------------------------------------

struct GemmInfo {
  std::string a_buf_name;
  std::string b_buf_name;
  std::string c_buf_name;
};

class GemmInfoCollector : public StmtVisitor {
 public:
  GemmInfo info;
  bool found = false;

  void VisitStmt_(const EvaluateNode* op) final {
    if (found) return;
    if (const auto* call = op->value.as<tir::CallNode>()) {
      if (const auto* op_node = call->op.as<OpNode>()) {
        if (op_node->name != "tl.tileop.gemm_py") return;
      } else {
        return;
      }
      // args[0]=A, [1]=B, [2]=C (BufferVar or Cast wrapping BufferVar)
      const auto& args = call->args;
      if (args.size() >= 3) {
        auto extract_name = [](const PrimExpr& e) -> std::string {
          // BufferVar node: tir::Var whose type_annotation is a buffer
          // In TIR the region arg is typically a tir::Var
          if (const auto* var = e.as<tir::VarNode>()) {
            return std::string(var->name_hint);
          }
          // May be wrapped in a call (buffer region pointer)
          if (const auto* call = e.as<tir::CallNode>()) {
            if (!call->args.empty()) {
              if (const auto* var = call->args[0].as<tir::VarNode>()) {
                return std::string(var->name_hint);
              }
            }
          }
          return "";
        };
        info.a_buf_name = extract_name(args[0]);
        info.b_buf_name = extract_name(args[1]);
        info.c_buf_name = extract_name(args[2]);
      }
      found = true;
    }
  }
};

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
        std::string direction = GetStringField(ann_map, "direction");
        std::string kind      = GetStringField(ann_map, "kind");
        std::string dst_scope = GetStringField(ann_map, "dst_scope");

        if (direction == "dram_to_cb") {
          // reader: captures DRAM source buffer name
          if (input_buf_name_out) {
            *input_buf_name_out = GetStringField(ann_map, "src_buffer");
          }
          return "reader";
        }
        if (direction == "cb_to_dram") {
          // writer: captures DRAM destination buffer name
          if (output_buf_name_out) {
            *output_buf_name_out = GetStringField(ann_map, "dst_buffer");
          }
          return "writer";
        }
        if (kind == "fused_staged_copy") {
          // treat as reader (dram→cb side) for now
          if (input_buf_name_out) {
            *input_buf_name_out = GetStringField(ann_map, "src_buffer");
          }
          return "reader";
        }
        // After a compute op, any copy with a global/DRAM destination is the writer.
        // This handles local.fragment → global (GEMM output copy).
        if (past_compute && (dst_scope.empty() || dst_scope == "global")) {
          if (output_buf_name_out) {
            *output_buf_name_out = GetStringField(ann_map, "dst_buffer");
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
// Build and store the 3-kernel segment_plan attribute
// ------------------------------------------------------------------

static void StoreGemmSegmentPlan(PrimFunc& func,
                                  const std::vector<std::string>& input_buf_names,
                                  const std::string& output_buf_name) {
  auto make_arg = [](const std::string& name, const char* kind,
                     const std::string& buffer_name = "") -> Map<String, Any> {
    Map<String, Any> arg;
    arg.Set("name", String(name));
    arg.Set("kind", String(kind));
    arg.Set("dtype", String("uint32"));
    if (!buffer_name.empty()) {
      arg.Set("buffer", String(buffer_name));
    }
    return arg;
  };

  // Reader kernel (BRISC/RISCV_0)
  Map<String, Any> reader;
  reader.Set("name", String("reader"));
  reader.Set("kind", String("reader"));
  reader.Set("core_type", String("brisc"));
  Array<Any> reader_args;
  for (const auto& buf_name : input_buf_names) {
    if (!buf_name.empty()) {
      reader_args.push_back(make_arg(buf_name + "_addr", "input_buffer_addr32", buf_name));
    }
  }
  reader_args.push_back(make_arg("work_linear_id", "work_linear_id"));
  reader_args.push_back(make_arg("a_tile_start_id", "a_tile_start_id"));
  reader_args.push_back(make_arg("a_tile_num_tiles", "a_tile_num_tiles"));
  reader_args.push_back(make_arg("a_tile_stride", "a_tile_stride"));
  reader_args.push_back(make_arg("b_tile_start_id", "b_tile_start_id"));
  reader_args.push_back(make_arg("b_tile_num_tiles", "b_tile_num_tiles"));
  reader_args.push_back(make_arg("b_tile_stride", "b_tile_stride"));
  reader_args.push_back(make_arg("k_tile_start_id", "k_tile_start_id"));
  reader_args.push_back(make_arg("num_k_tiles", "num_k_tiles"));
  reader.Set("runtime_args", reader_args);

  // Compute kernel (TRISC)
  Map<String, Any> compute;
  compute.Set("name", String("compute"));
  compute.Set("kind", String("compute"));
  compute.Set("core_type", String("trisc"));
  Array<Any> compute_args;
  compute_args.push_back(make_arg("k_tile_start_id", "k_tile_start_id"));
  compute_args.push_back(make_arg("num_k_tiles", "num_k_tiles"));
  compute.Set("runtime_args", compute_args);

  // Writer kernel (NCRISC/RISCV_1)
  Map<String, Any> writer;
  writer.Set("name", String("writer"));
  writer.Set("kind", String("writer"));
  writer.Set("core_type", String("ncrisc"));
  Array<Any> writer_args;
  if (!output_buf_name.empty()) {
    writer_args.push_back(make_arg(output_buf_name + "_addr", "output_buffer_addr32",
                                   output_buf_name));
  }
  writer_args.push_back(make_arg("work_linear_id", "work_linear_id"));
  writer_args.push_back(make_arg("output_tile_start_id", "output_tile_start_id"));
  writer_args.push_back(make_arg("output_tile_num_tiles", "output_tile_num_tiles"));
  writer_args.push_back(make_arg("output_tile_stride", "output_tile_stride"));
  writer.Set("runtime_args", writer_args);

  Array<Any> kernels;
  kernels.push_back(reader);
  kernels.push_back(compute);
  kernels.push_back(writer);

  Map<String, Any> attrs;
  if (func->attrs.defined()) attrs = func->attrs->dict;
  attrs.Set("blackhole.segment_plan", kernels);
  func.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
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

  GemmInfoCollector gemm_info_collector;
  gemm_info_collector(func->body);
  GemmInfo gemm_info = gemm_info_collector.info;
  if (gemm_info.a_buf_name.empty() && !annotator.input_buf_names.empty()) {
    gemm_info.a_buf_name = annotator.input_buf_names[0];
  }
  if (gemm_info.b_buf_name.empty() && annotator.input_buf_names.size() > 1) {
    gemm_info.b_buf_name = annotator.input_buf_names[1];
  }
  if (gemm_info.c_buf_name.empty()) {
    gemm_info.c_buf_name = annotator.output_buf_name;
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
