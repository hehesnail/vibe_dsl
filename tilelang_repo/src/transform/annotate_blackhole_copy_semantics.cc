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
 * \file annotate_blackhole_copy_semantics.cc
 * \brief Annotate copy For loops with blackhole.copy_semantics before SplitHostDevice.
 *
 * Runs in the split-before phase (after LowerTileOp, before AnnotateDeviceRegions).
 * Finds BufferStore(BufferLoad) copy loop patterns and wraps the outermost
 * copy-containing For loop with:
 *
 *   AttrStmt("blackhole.copy_semantics", StringImm("<kind>:<dir>:<src>:<dst>:<dtype>"), ...)
 *
 * This gives LowerBlackholeOps stable metadata without requiring pattern-matching
 * on the loop body, which is fragile after FlattenBuffer / VectorizeLoop / StorageRewrite.
 *
 * Annotation string format:
 *   staged_copy:<direction>:<src_buf>:<dst_buf>:<dtype>
 *   fused_staged_copy:dram_to_cb_to_dram:<src_dram>:<mid_shared>:<dst_dram>:<dtype>
 *
 * where <direction> is one of: dram_to_cb, cb_to_dram, dram_to_dram, cb_to_cb
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using tvm::ffi::GetRef;
using tvm::Integer;

// Attr key used to mark copy-containing For loops
static constexpr const char* kBlackholeCopySemantics = "blackhole.copy_semantics";

// Storage scope helpers (same logic as LowerBlackholeOps)
static bool IsDramScope(const std::string& scope) {
  return scope.empty() || scope == "global";
}

static bool IsCBScope(const std::string& scope) {
  return scope == "shared" || scope == "shared.dyn" ||
         (!scope.empty() && scope.rfind("shared", 0) == 0);
}

static std::string GetStorageScopeStr(const Buffer& buf) {
  ffi::String s = buf.scope();
  return std::string(s);
}

static std::string DataTypeStr(const DataType& dt) {
  if (dt.is_float()) {
    if (dt.bits() == 16) return "float16";
    if (dt.bits() == 32) return "float32";
    if (dt.bits() == 8)  return "float8";
  } else if (dt.is_int()) {
    if (dt.bits() == 32) return "int32";
    if (dt.bits() == 16) return "int16";
    if (dt.bits() == 8)  return "int8";
  } else if (dt.is_uint()) {
    if (dt.bits() == 32) return "uint32";
    if (dt.bits() == 16) return "uint16";
  }
  return "unknown";
}

static bool IsCopyOp(const BufferStoreNode* op) {
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    return !op->buffer.same_as(load->buffer);
  }
  return false;
}

struct CopyStoreInfo {
  const BufferStoreNode* store;
  std::string direction;  // "dram_to_cb", "cb_to_dram", "dram_to_dram", "cb_to_cb"
};

static std::string GetCopyDirection(const BufferStoreNode* op) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return "unknown";

  std::string src_scope = GetStorageScopeStr(load->buffer);
  std::string dst_scope = GetStorageScopeStr(op->buffer);

  if (IsDramScope(src_scope) && IsCBScope(dst_scope)) return "dram_to_cb";
  if (IsCBScope(src_scope) && IsDramScope(dst_scope)) return "cb_to_dram";
  if (IsDramScope(src_scope) && IsDramScope(dst_scope)) return "dram_to_dram";
  if (IsCBScope(src_scope) && IsCBScope(dst_scope)) return "cb_to_cb";
  return "unknown";
}

/*!
 * \brief Collect all copy BufferStore nodes reachable from stmt.
 * Does not recurse across annotated AttrStmt boundaries (to avoid re-processing).
 */
static void CollectCopyStores(const Stmt& stmt, std::vector<CopyStoreInfo>* out) {
  if (const auto* store = stmt.as<BufferStoreNode>()) {
    if (IsCopyOp(store)) {
      CopyStoreInfo info;
      info.store = store;
      info.direction = GetCopyDirection(store);
      out->push_back(info);
    }
    return;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    // Don't recurse into already-annotated regions
    if (attr->attr_key == kBlackholeCopySemantics) return;
    CollectCopyStores(attr->body, out);
    return;
  }
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    for (const auto& child : seq->seq) {
      CollectCopyStores(child, out);
    }
    return;
  }
  if (const auto* f = stmt.as<ForNode>()) {
    CollectCopyStores(f->body, out);
    return;
  }
  if (const auto* alloc = stmt.as<AllocateNode>()) {
    CollectCopyStores(alloc->body, out);
    return;
  }
  if (const auto* let = stmt.as<LetStmtNode>()) {
    CollectCopyStores(let->body, out);
    return;
  }
  // IfThenElse, Block, etc. — recurse into both branches if needed
  if (const auto* ite = stmt.as<IfThenElseNode>()) {
    CollectCopyStores(ite->then_case, out);
    if (ite->else_case.defined()) CollectCopyStores(ite->else_case.value(), out);
    return;
  }
}

static std::string BuildAnnotationStr(const std::vector<CopyStoreInfo>& copies) {
  // Find dram_to_cb and cb_to_dram copies
  const BufferStoreNode* dram_to_cb_store = nullptr;
  const BufferStoreNode* cb_to_dram_store = nullptr;

  for (const auto& ci : copies) {
    if (ci.direction == "dram_to_cb" && !dram_to_cb_store) {
      dram_to_cb_store = ci.store;
    } else if (ci.direction == "cb_to_dram" && !cb_to_dram_store) {
      cb_to_dram_store = ci.store;
    }
  }

  // Fused: both directions present, sharing a shared buffer
  if (dram_to_cb_store && cb_to_dram_store) {
    const auto* dram_load = dram_to_cb_store->value.as<BufferLoadNode>();
    const auto* cb_load   = cb_to_dram_store->value.as<BufferLoadNode>();
    if (dram_load && cb_load) {
      std::string src_dram  = std::string(dram_load->buffer->name);
      std::string mid_shared = std::string(dram_to_cb_store->buffer->name);
      std::string dst_dram  = std::string(cb_to_dram_store->buffer->name);
      std::string dtype     = DataTypeStr(dram_to_cb_store->buffer->dtype);
      return "fused_staged_copy:dram_to_cb_to_dram:" +
             src_dram + ":" + mid_shared + ":" + dst_dram + ":" + dtype;
    }
  }

  // Single dram_to_cb
  if (dram_to_cb_store) {
    const auto* load = dram_to_cb_store->value.as<BufferLoadNode>();
    if (load) {
      return "staged_copy:dram_to_cb:" +
             std::string(load->buffer->name) + ":" +
             std::string(dram_to_cb_store->buffer->name) + ":" +
             DataTypeStr(dram_to_cb_store->buffer->dtype);
    }
  }

  // Single cb_to_dram
  if (cb_to_dram_store) {
    const auto* load = cb_to_dram_store->value.as<BufferLoadNode>();
    if (load) {
      return "staged_copy:cb_to_dram:" +
             std::string(load->buffer->name) + ":" +
             std::string(cb_to_dram_store->buffer->name) + ":" +
             DataTypeStr(cb_to_dram_store->buffer->dtype);
    }
  }

  // Other kinds (dram_to_dram, cb_to_cb) — use first copy found
  if (!copies.empty()) {
    const auto* store = copies[0].store;
    const auto* load  = store->value.as<BufferLoadNode>();
    if (load) {
      return "staged_copy:" + copies[0].direction + ":" +
             std::string(load->buffer->name) + ":" +
             std::string(store->buffer->name) + ":" +
             DataTypeStr(store->buffer->dtype);
    }
  }

  return "unknown_copy:unknown";
}

/*!
 * \brief StmtMutator that wraps copy-containing For loops with
 *        AttrStmt("blackhole.copy_semantics", ...).
 *
 * When a For loop is found to contain at least one copy BufferStore
 * (BufferStore where value is a BufferLoad of a different buffer),
 * it is wrapped with an annotation AttrStmt.  The For loop itself
 * is NOT mutated — the annotation is purely additive metadata.
 */
class BlackholeCopyAnnotator : public StmtMutator {
 public:
  PrimFunc Transform(const PrimFunc& func) {
    Stmt new_body = VisitStmt(func->body);
    if (new_body.same_as(func->body)) return func;
    PrimFunc new_func = func;
    new_func.CopyOnWrite()->body = new_body;
    return new_func;
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    // Collect copy stores in the entire sub-tree of this For loop
    std::vector<CopyStoreInfo> copies;
    CollectCopyStores(GetRef<Stmt>(op), &copies);

    if (copies.empty()) {
      // No copies here — recurse normally so inner For loops can be annotated
      return StmtMutator::VisitStmt_(op);
    }

    std::string ann_str = BuildAnnotationStr(copies);

    // Wrap this For loop with the annotation.
    // Do NOT recurse into this For loop's body — the annotation marks the
    // entire loop as a copy unit, which is what LowerBlackholeOps expects.
    return AttrStmt(Integer(0), kBlackholeCopySemantics,
                    StringImm(ann_str), GetRef<Stmt>(op));
  }
};

// Pass entry point
static PrimFunc AnnotateBlackholeCopySemanticsFunc(PrimFunc func, IRModule /*m*/,
                                                   tir::transform::PassContext /*ctx*/) {
  return BlackholeCopyAnnotator().Transform(func);
}

tir::transform::Pass AnnotateBlackholeCopySemanticsPass() {
  return tir::transform::CreatePrimFuncPass(
      AnnotateBlackholeCopySemanticsFunc, 0,
      "tl.transform.AnnotateBlackholeCopySemantics", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateBlackholeCopySemantics",
                        AnnotateBlackholeCopySemanticsPass);
}

}  // namespace tl
}  // namespace tvm
