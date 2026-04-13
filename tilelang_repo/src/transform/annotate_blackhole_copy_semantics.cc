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
 * copy-containing For loop by attaching a structured loop annotation:
 *
 *   For(..., annotations={"blackhole.copy_semantics": Map<String, Any>{...}})
 *
 * This gives PlanTTKernelABI stable metadata without requiring pattern-matching
 * on the loop body, which is fragile after FlattenBuffer / VectorizeLoop / StorageRewrite.
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

#include "common/companion_base.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::ffi::GetRef;
using tvm::Integer;
using tvm::ffi::Array;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;

// Attr key used to mark copy-containing For loops
static constexpr const char* kBlackholeCopySemantics = "blackhole.copy_semantics";

// Storage scope helpers (same logic as PlanTTKernelABI)
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
  if (dt.is_bfloat16()) return "bfloat16";
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

static Array<Integer> ExtractStaticShape(const Buffer& buf) {
  Array<Integer> shape;
  for (const PrimExpr& dim : buf->shape) {
    if (const auto* imm = dim.as<IntImmNode>()) {
      shape.push_back(Integer(imm->value));
    }
  }
  return shape;
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

static Map<String, Any> BuildAnnotation(const std::vector<CopyStoreInfo>& copies) {
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
      Map<String, Any> ann;
      ann.Set(schema_key::kKind, String("fused_staged_copy"));
      ann.Set(schema_key::kDirection, String("dram_to_cb_to_dram"));
      ann.Set(schema_key::kSrcBuffer, String(dram_load->buffer->name));
      ann.Set(schema_key::kSrcBufferRef, dram_load->buffer);
      ann.Set(schema_key::kDstBuffer, String(cb_to_dram_store->buffer->name));
      ann.Set(schema_key::kDstBufferRef, cb_to_dram_store->buffer);
      ann.Set(schema_key::kMidBuffer, String(dram_to_cb_store->buffer->name));
      ann.Set(schema_key::kMidBufferRef, dram_to_cb_store->buffer);
      ann.Set("src_scope", String(GetStorageScopeStr(dram_load->buffer)));
      ann.Set("dst_scope", String(GetStorageScopeStr(cb_to_dram_store->buffer)));
      ann.Set(schema_key::kDType, String(DataTypeStr(dram_to_cb_store->buffer->dtype)));
      ann.Set(schema_key::kSrcShape, ExtractStaticShape(dram_load->buffer));
      ann.Set(schema_key::kDstShape, ExtractStaticShape(cb_to_dram_store->buffer));
      ann.Set(schema_key::kMidShape, ExtractStaticShape(dram_to_cb_store->buffer));
      return ann;
    }
  }

  // Single dram_to_cb
  if (dram_to_cb_store) {
    const auto* load = dram_to_cb_store->value.as<BufferLoadNode>();
    if (load) {
      Map<String, Any> ann;
      ann.Set(schema_key::kKind, String("staged_copy"));
      ann.Set(schema_key::kDirection, String("dram_to_cb"));
      ann.Set(schema_key::kSrcBuffer, String(load->buffer->name));
      ann.Set(schema_key::kSrcBufferRef, load->buffer);
      ann.Set(schema_key::kDstBuffer, String(dram_to_cb_store->buffer->name));
      ann.Set(schema_key::kDstBufferRef, dram_to_cb_store->buffer);
      ann.Set("src_scope", String(GetStorageScopeStr(load->buffer)));
      ann.Set("dst_scope", String(GetStorageScopeStr(dram_to_cb_store->buffer)));
      ann.Set(schema_key::kDType, String(DataTypeStr(dram_to_cb_store->buffer->dtype)));
      ann.Set(schema_key::kSrcShape, ExtractStaticShape(load->buffer));
      ann.Set(schema_key::kDstShape, ExtractStaticShape(dram_to_cb_store->buffer));
      ann.Set(schema_key::kMidShape, ExtractStaticShape(dram_to_cb_store->buffer));
      return ann;
    }
  }

  // Single cb_to_dram
  if (cb_to_dram_store) {
    const auto* load = cb_to_dram_store->value.as<BufferLoadNode>();
    if (load) {
      Map<String, Any> ann;
      ann.Set(schema_key::kKind, String("staged_copy"));
      ann.Set(schema_key::kDirection, String("cb_to_dram"));
      ann.Set(schema_key::kSrcBuffer, String(load->buffer->name));
      ann.Set(schema_key::kSrcBufferRef, load->buffer);
      ann.Set(schema_key::kDstBuffer, String(cb_to_dram_store->buffer->name));
      ann.Set(schema_key::kDstBufferRef, cb_to_dram_store->buffer);
      ann.Set("src_scope", String(GetStorageScopeStr(load->buffer)));
      ann.Set("dst_scope", String(GetStorageScopeStr(cb_to_dram_store->buffer)));
      ann.Set(schema_key::kDType, String(DataTypeStr(cb_to_dram_store->buffer->dtype)));
      ann.Set(schema_key::kSrcShape, ExtractStaticShape(load->buffer));
      ann.Set(schema_key::kDstShape, ExtractStaticShape(cb_to_dram_store->buffer));
      ann.Set(schema_key::kMidShape, ExtractStaticShape(load->buffer));
      return ann;
    }
  }

  // Other kinds (dram_to_dram, cb_to_cb) — use first copy found
  if (!copies.empty()) {
    const auto* store = copies[0].store;
    const auto* load  = store->value.as<BufferLoadNode>();
    if (load) {
      Map<String, Any> ann;
      ann.Set(schema_key::kKind, String("staged_copy"));
      ann.Set(schema_key::kDirection, String(copies[0].direction));
      ann.Set(schema_key::kSrcBuffer, String(load->buffer->name));
      ann.Set(schema_key::kSrcBufferRef, load->buffer);
      ann.Set(schema_key::kDstBuffer, String(store->buffer->name));
      ann.Set(schema_key::kDstBufferRef, store->buffer);
      ann.Set("src_scope", String(GetStorageScopeStr(load->buffer)));
      ann.Set("dst_scope", String(GetStorageScopeStr(store->buffer)));
      ann.Set(schema_key::kDType, String(DataTypeStr(store->buffer->dtype)));
      ann.Set(schema_key::kSrcShape, ExtractStaticShape(load->buffer));
      ann.Set(schema_key::kDstShape, ExtractStaticShape(store->buffer));
      return ann;
    }
  }

  Map<String, Any> ann;
  ann.Set(schema_key::kKind, String("unknown_copy"));
  ann.Set(schema_key::kDirection, String("unknown"));
  return ann;
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

    For annotated_loop = GetRef<For>(op);
    auto* n = annotated_loop.CopyOnWrite();
    n->annotations = op->annotations;
    n->annotations.Set(String(kBlackholeCopySemantics), BuildAnnotation(copies));
    return annotated_loop;
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
