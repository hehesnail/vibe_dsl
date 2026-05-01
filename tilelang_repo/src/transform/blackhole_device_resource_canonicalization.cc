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
 * \file blackhole_device_resource_canonicalization.cc
 * \brief Canonicalize Blackhole device-private resources before SplitHostDevice.
 *
 * TIR's StorageScope conflates storage hierarchy with resource semantics.
 * GPU has a 1:1 mapping, but Blackhole breaks this assumption:
 *   - CB (Circular Buffer) is in L1 but is a FIFO queue, not contiguous memory
 *   - Dst accumulator is register-backed, not addressable memory
 *
 * This pass:
 *   1. Classifies ALL device-private buffers anywhere in the IR:
 *        shared.dyn / shared  →  blackhole.cb.{input|output|intermed}
 *        local.fragment / local(with fragment layout evidence)  →  blackhole.acc
 *   2. Rewrites buffer scope (StorageRank) to the new types
 *   3. Relocates device-private buffers that live ABOVE the thread_extent boundary
 *      (either as explicit AllocateNode stmts or in Block.alloc_buffers) INTO the
 *      thread_extent body, so that SplitHostDevice's VarUseDefAnalyzer sees them
 *      as defined (not free vars → not ABI params)
 *   4. Rewrites attached typed projections to match canonicalized resource scopes
 *
 * Must run before AnnotateDeviceRegions while the current TIR still exposes
 * copy/dataflow structure directly.
 *
 * With corrected scopes, generic passes naturally skip Blackhole resources:
 *   MergeSharedMemoryAllocations: checks rank == kShared  → misses kBlackholeCB
 *   LowerDeviceKernelLaunch:      checks rank == kShared  → misses kBlackholeCB
 *   SplitHostDevice:              VarUseDefAnalyzer sees Allocate inside device body
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../target/tt_program_projection.h"
#include "../layout/layout.h"
#include "common/tt_target_program.h"
#include "common/companion_base.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::ffi::Array;
using tvm::ffi::Any;
using tvm::ffi::GetRef;
using tvm::ffi::Map;
using tvm::ffi::String;
using tvm::Integer;
using runtime::StorageRank;
using runtime::StorageScope;

// ---------------------------------------------------------------------------
// Helper: get the scope string for a buffer_var
// ---------------------------------------------------------------------------
static std::string GetScope(const Var& v) {
  return std::string(GetPtrStorageScope(v));
}

static std::string GetStorageScopeStr(const Buffer& buf) {
  ffi::String s = buf.scope();
  return std::string(s);
}

static bool IsDramScope(const std::string& scope) {
  return scope.empty() || scope == "global";
}

// ---------------------------------------------------------------------------
// Helper: is this scope a CB candidate?
//   shared / shared.dyn / shared.*  →  CB candidate
// ---------------------------------------------------------------------------
static bool IsCBScope(const std::string& scope) {
  return !scope.empty() && scope.rfind("shared", 0) == 0;
}

static bool IsFragmentScope(const std::string& scope) {
  return scope == "local.fragment";
}

// ---------------------------------------------------------------------------
// Helper: create a new Var with a different pointer storage scope
// ---------------------------------------------------------------------------
static Var RemapVarScope(const Var& old_var, const std::string& new_scope) {
  const auto* ptr = old_var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr) << "buffer_var must have PointerType, got: " << old_var;
  return Var(old_var->name_hint, PointerType(ptr->element_type, new_scope));
}

// ---------------------------------------------------------------------------
// Helper: create a new Buffer with a different data var
// ---------------------------------------------------------------------------
static Buffer RemapBufferData(const Buffer& buf, const Var& new_data) {
  return Buffer(new_data, buf->dtype, buf->shape, buf->strides, buf->elem_offset,
                buf->name, buf->data_alignment, buf->offset_factor, buf->buffer_type,
                buf->axis_separators);
}

static bool IsCopyOp(const BufferStoreNode* op) {
  const auto* load = op->value.as<BufferLoadNode>();
  return load && !op->buffer.same_as(load->buffer);
}

enum class CopyDirection {
  kUnknown,
  kDramToCB,
  kCBToDram,
  kDramToDram,
  kCBToCB,
};

static CopyDirection GetCopyDirection(const BufferStoreNode* op) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return CopyDirection::kUnknown;
  }

  const std::string src_scope = GetStorageScopeStr(load->buffer);
  const std::string dst_scope = GetStorageScopeStr(op->buffer);
  if (IsDramScope(src_scope) && IsCBScope(dst_scope)) {
    return CopyDirection::kDramToCB;
  }
  if (IsCBScope(src_scope) && IsDramScope(dst_scope)) {
    return CopyDirection::kCBToDram;
  }
  if (IsDramScope(src_scope) && IsDramScope(dst_scope)) {
    return CopyDirection::kDramToDram;
  }
  if (IsCBScope(src_scope) && IsCBScope(dst_scope)) {
    return CopyDirection::kCBToCB;
  }
  return CopyDirection::kUnknown;
}

// ===========================================================================
// Phase 1: classify all device-private buffers
// ===========================================================================
struct ResourceInfo {
  std::string new_scope;   // e.g. "blackhole.cb.input", "blackhole.acc"
  std::string role;        // "input", "output", "intermed", "accumulator"
  std::string cls;         // "cb" or "accumulator"
};

class BlackholeResourceClassifier : public StmtExprVisitor {
 public:
  // Final results: buffer var name → resource info
  std::unordered_map<std::string, ResourceInfo> resource_map;
  void Run(const PrimFunc& func) {
    // Scan 1: collect copy direction info
    VisitStmt(func->body);
    // Scan 2: revisit nodes with full context to assign scopes and fragment-layout evidence
    ClassifyAllocates(func->body);
  }

 private:
  std::unordered_set<std::string> layout_fragment_names_;
  // Direct current-stage copy recovery:
  std::unordered_set<std::string> cb_input_names_;    // appear as shared dst in dram_to_cb
  std::unordered_set<std::string> cb_output_names_;   // appear as shared src in cb_to_dram

  void VisitStmt_(const ForNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (IsCopyOp(op)) {
      const auto* load = op->value.as<BufferLoadNode>();
      ICHECK(load != nullptr);
      switch (GetCopyDirection(op)) {
        case CopyDirection::kDramToCB:
          cb_input_names_.insert(op->buffer->name);
          break;
        case CopyDirection::kCBToDram:
          cb_output_names_.insert(load->buffer->name);
          break;
        default:
          break;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void CollectLayoutFragments(const BlockNode* op) {
    if (!op->annotations.count(attr::kLayoutMap)) return;
    auto layout_map_any = op->annotations.Get(attr::kLayoutMap);
    if (!layout_map_any) return;
    auto layout_map = layout_map_any->as<Map<Buffer, Layout>>();
    if (!layout_map || !layout_map.value().defined()) return;
    for (const auto& [buffer, _layout] : layout_map.value()) {
      const Layout& layout = _layout;
      std::string scope = GetScope(buffer->data);
      if (scope == "local" || scope == "local.fragment") {
        std::string name = std::string(buffer->name);
        layout_fragment_names_.insert(name);
        resource_map[name] = {"blackhole.acc", "accumulator", "accumulator"};
      }
    }
  }

  // Walk the IR to find all device-private buffer allocations.
  // Does NOT stop at thread_extent — we classify buffers anywhere in the tree
  // (A_shared/B_shared may be inside thread_extent, C_local may be in alloc_buffers).
  void ClassifyAllocates(const Stmt& stmt) {
    if (const auto* alloc = stmt.as<AllocateNode>()) {
      std::string scope = GetScope(alloc->buffer_var);
      std::string name = alloc->buffer_var->name_hint;
      ClassifyBuffer(name, scope);
      ClassifyAllocates(alloc->body);
    } else if (const auto* decl = stmt.as<DeclBufferNode>()) {
      ClassifyAllocates(decl->body);
    } else if (const auto* br = stmt.as<BlockRealizeNode>()) {
      // Check alloc_buffers: these may be device-private but outside thread_extent
      CollectLayoutFragments(br->block.get());
      for (const auto& buf : br->block->alloc_buffers) {
        std::string scope = GetScope(buf->data);
        ClassifyBuffer(std::string(buf->name), scope);
      }
      ClassifyAllocates(br->block->body);
    } else if (const auto* block = stmt.as<BlockNode>()) {
      CollectLayoutFragments(block);
      for (const auto& buf : block->alloc_buffers) {
        std::string scope = GetScope(buf->data);
        ClassifyBuffer(std::string(buf->name), scope);
      }
      ClassifyAllocates(block->body);
    } else if (const auto* attr = stmt.as<AttrStmtNode>()) {
      // Continue into thread_extent body — A_shared/B_shared may be inside
      ClassifyAllocates(attr->body);
    } else if (const auto* seq = stmt.as<SeqStmtNode>()) {
      for (const auto& s : seq->seq) ClassifyAllocates(s);
    }
  }

  void ClassifyBuffer(const std::string& name, const std::string& scope) {
    if (scope == "local.fragment") {
      // Always an accumulator
      resource_map[name] = {"blackhole.acc", "accumulator", "accumulator"};
    } else if (scope == "local") {
      // Promote local buffers to accumulator only when the current TIR still
      // carries explicit fragment-layout evidence for that logical subject.
      if (layout_fragment_names_.count(name)) {
        resource_map[name] = {"blackhole.acc", "accumulator", "accumulator"};
      }
      // else: keep as local — not a Blackhole device-private resource we reclassify
    } else if (IsCBScope(scope)) {
      // Determine CB role from direct copy/dataflow recovery on the current TIR
      std::string cb_scope, role;
      if (cb_input_names_.count(name)) {
        cb_scope = "blackhole.cb.input";
        role = "input";
      } else if (cb_output_names_.count(name)) {
        cb_scope = "blackhole.cb.output";
        role = "output";
      } else {
        cb_scope = "blackhole.cb";
        role = "intermed";
      }
      resource_map[name] = {cb_scope, role, "cb"};
    }
  }
};

// ===========================================================================
// Phase 2: rewrite scopes + relocate allocations inside thread_extent
// ===========================================================================

// Represents a single wrapper node to be re-applied inside thread_extent body.
struct WrapperInfo {
  enum Kind {
    kAllocate,          // Explicit AllocateNode stripped from above thread_extent
    kDeclBuffer,        // Explicit DeclBufferNode stripped from above thread_extent
    kAllocateFromBuffer // Synthesized from Block.alloc_buffers (generates Allocate+DeclBuffer)
  } kind;

  // kAllocate fields:
  Var new_var;
  DataType dtype;
  Array<PrimExpr> extents;
  PrimExpr condition;
  Map<String, Any> annotations;

  // kDeclBuffer / kAllocateFromBuffer fields:
  Buffer new_buf;

  // Apply: wrap |inner| with this allocation node
  Stmt Apply(Stmt inner) const {
    if (kind == kAllocate) {
      return Allocate(new_var, dtype, extents, condition, inner, annotations);
    } else if (kind == kDeclBuffer) {
      return DeclBuffer(new_buf, inner);
    } else {
      // kAllocateFromBuffer: synthesize both Allocate and DeclBuffer
      Stmt with_decl = DeclBuffer(new_buf, inner);
      return Allocate(new_var, dtype, extents, Bool(1), with_decl);
    }
  }
};

class BlackholeResourceCanonicalizer : public StmtExprMutator {
 public:
  explicit BlackholeResourceCanonicalizer(
      const std::unordered_map<std::string, ResourceInfo>& resource_map)
      : resource_map_(resource_map) {}

  PrimFunc Transform(PrimFunc func) {
    Stmt new_body = VisitStmt(func->body);
    auto n = func.CopyOnWrite();
    n->body = new_body;
    // Rebuild lowered resource metadata directly on the current explicit attrs.
    n->attrs = RewriteLoweredAttrs(func->attrs);
    return func;
  }

 private:
  std::unordered_map<std::string, ResourceInfo> resource_map_;
  // var remapping: old buffer_var → new buffer_var (by identity)
  std::unordered_map<const VarNode*, Var> var_remap_;
  // Resource-name aliases are used only after the Buffer itself has been
  // classified as a Blackhole resource. Generic Var rewriting remains identity-keyed.
  std::unordered_map<std::string, Var> canonical_var_by_resource_name_;
  std::unordered_set<std::string> ambiguous_resource_names_;
  // buffer remapping cache: old Buffer → new Buffer (same object across all call sites)
  // Critical for PlanTTKernelABI.buffer_to_cb_ deduplication which uses Buffer pointer equality.
  std::unordered_map<const BufferNode*, Buffer> buf_remap_;
  // collected wrappers stripped from above thread_extent, to inject inside it
  std::vector<WrapperInfo> wrappers_;
  // true once we have entered (and are wrapping) the outermost thread_extent
  bool wrapped_{false};

  const ResourceInfo* GetInfo(const std::string& name) const {
    auto it = resource_map_.find(name);
    if (it == resource_map_.end()) return nullptr;
    return &it->second;
  }

  std::string CanonicalScopeForBuffer(const std::string& name) const {
    if (const auto* info = GetInfo(name)) {
      return info->new_scope;
    }
    return "";
  }

  Map<String, Any> RewriteScopedRecord(const Map<String, Any>& record,
                                       const char* name_key) const {
    auto maybe_name = record.Get(String(name_key));
    if (!maybe_name.has_value()) {
      return record;
    }
    auto name = maybe_name.value().as<String>();
    if (!name.has_value()) {
      return record;
    }
    const std::string scope =
        CanonicalScopeForBuffer(static_cast<std::string>(name.value()));
    if (scope.empty()) {
      return record;
    }
    Map<String, Any> rewritten = record;
    rewritten.Set(String(schema_key::kScope), String(scope));
    return rewritten;
  }

  Array<Any> RewriteScopedRecordArray(const Array<Any>& records,
                                      const char* name_key) const {
    Array<Any> rewritten;
    for (const Any& record_any : records) {
      auto record = record_any.as<Map<String, Any>>();
      if (!record.has_value()) {
        rewritten.push_back(record_any);
        continue;
      }
      rewritten.push_back(RewriteScopedRecord(record.value(), name_key));
    }
    return rewritten;
  }

  TTProgram RewriteTTProgram(const TTProgram& program) const {
    return TTProgram(program->entry_name, program->member_func, program->mesh_plans,
                     program->buffer_distribution_plans,
                     program->tensor_memory_config_plans,
                     program->op_sharding_contracts,
                     program->placement_resolution_plans, program->reshard_plans,
                     program->block_plans,
                     program->kernel_plans, program->compute_op_plans,
                     program->transport_plans, program->sync_plans,
                     program->abi_plans, program->execution_plans, program->kernels,
                     program->core_groups, program->cb_plans, program->semaphore_plans,
                     program->compute_sync_plans, program->dst_layout_plans,
                     program->live_form_plans, program->materialization_plans,
                     program->consumer_binding_plans, program->resource_demands,
                     program->resource_pressure_reports);
  }

  const ResourceInfo* GetOrInferInfo(const std::string& name,
                                     const std::string& scope) {
    if (const auto* info = GetInfo(name)) return info;

    if (IsCBScope(scope)) {
      resource_map_[name] = {"blackhole.cb", "intermed", "cb"};
      return &resource_map_.at(name);
    }
    if (scope == "local.fragment") {
      resource_map_[name] = {"blackhole.acc", "accumulator", "accumulator"};
      return &resource_map_.at(name);
    }
    return nullptr;
  }

  Var GetNewVar(const Var& old_var) {
    auto it = var_remap_.find(old_var.get());
    if (it != var_remap_.end()) return it->second;
    return old_var;
  }

  void RecordCanonicalResourceVar(const std::string& name, const Var& new_var) {
    if (ambiguous_resource_names_.count(name)) {
      return;
    }
    auto [it, inserted] = canonical_var_by_resource_name_.emplace(name, new_var);
    if (!inserted && !it->second.same_as(new_var)) {
      canonical_var_by_resource_name_.erase(name);
      ambiguous_resource_names_.insert(name);
    }
  }

  Buffer GetNewBuffer(const Buffer& buf) {
    auto it = buf_remap_.find(buf.get());
    if (it != buf_remap_.end()) return it->second;
    Var new_data = GetNewVar(buf->data);
    if (new_data.same_as(buf->data) &&
        GetOrInferInfo(std::string(buf->name), buf.scope()) != nullptr &&
        !ambiguous_resource_names_.count(std::string(buf->name))) {
      auto alias_it = canonical_var_by_resource_name_.find(std::string(buf->name));
      if (alias_it != canonical_var_by_resource_name_.end()) {
        new_data = alias_it->second;
      }
    }
    if (new_data.same_as(buf->data)) return buf;
    Buffer new_buf = RemapBufferData(buf, new_data);
    buf_remap_[buf.get()] = new_buf;
    return new_buf;
  }

  // ---- Handle Block.alloc_buffers (e.g. tilelang_root block wrapping the kernel) ----
  // Strip device-private buffers from alloc_buffers; they will be relocated
  // inside the thread_extent body via wrappers_.
  //
  // Pattern from flatten_buffer.cc:
  //   1. Get Block ObjectRef
  //   2. Modify alloc_buffers (strip device-private) via CopyOnWrite
  //   3. Delegate recursive visit of body/reads/writes to base class
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = ffi::GetRef<Block>(op);

    // Build stripped alloc_buffers; pre-populate var_remap_ so the body visit
    // (done by the base class below) can remap var references.
    Array<Buffer> new_alloc_buffers;
    for (const auto& buf : op->alloc_buffers) {
      std::string name = std::string(buf->name);
        const ResourceInfo* info = GetOrInferInfo(name, buf.scope());
        if (info != nullptr) {
          Var new_var = RemapVarScope(buf->data, info->new_scope);
          var_remap_[buf->data.get()] = new_var;
          RecordCanonicalResourceVar(name, new_var);
          Buffer new_buf = RemapBufferData(buf, new_var);
        WrapperInfo wi;
        wi.kind = WrapperInfo::kAllocateFromBuffer;
        wi.new_var = new_var;
        wi.dtype = buf->dtype;
        wi.extents = buf->shape;
        wi.new_buf = new_buf;
        wrappers_.push_back(std::move(wi));
        // Stripped: don't add to new_alloc_buffers
      } else {
        new_alloc_buffers.push_back(buf);
      }
    }

    if (!new_alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = new_alloc_buffers;
    }

    // Delegate recursive visit (body, reads, writes, iter_vars, ...) to base class.
    // Our overrides (VisitExpr_(VarNode), VisitStmt_(AllocateNode), etc.) will fire
    // through virtual dispatch as the base class walks the block's fields.
    return StmtExprMutator::VisitStmt_(block.get());
  }

  // ---- Strip device-private AllocateNode above thread_extent ----
  // Inside thread_extent (wrapped_ == true): just remap scope in-place.
  Stmt VisitStmt_(const AllocateNode* op) final {
    std::string name = op->buffer_var->name_hint;
    const ResourceInfo* info = GetOrInferInfo(name, GetScope(op->buffer_var));
    if (info != nullptr) {
      Var new_var = RemapVarScope(op->buffer_var, info->new_scope);
      var_remap_[op->buffer_var.get()] = new_var;
      RecordCanonicalResourceVar(name, new_var);

      if (!wrapped_) {
        // Above thread_extent: strip this node, collect for relocation
        WrapperInfo wi;
        wi.kind = WrapperInfo::kAllocate;
        wi.new_var = new_var;
        wi.dtype = op->dtype;
        wi.extents = op->extents;
        wi.condition = op->condition;
        wi.annotations = op->annotations;
        wrappers_.push_back(std::move(wi));
        return VisitStmt(op->body);
      } else {
        // Inside thread_extent: keep the Allocate but remap buffer_var scope
        Stmt new_body = VisitStmt(op->body);
        return Allocate(new_var, op->dtype, op->extents, op->condition, new_body,
                        op->annotations);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // ---- Strip device-private DeclBuffer above thread_extent ----
  // Inside thread_extent: update buffer data var in-place.
  Stmt VisitStmt_(const DeclBufferNode* op) final {
    std::string name = op->buffer->name;
    if (GetOrInferInfo(name, op->buffer.scope()) != nullptr) {
      Buffer new_buf = GetNewBuffer(op->buffer);
      if (!wrapped_) {
        // Above thread_extent: strip this node, collect for relocation
        WrapperInfo wi;
        wi.kind = WrapperInfo::kDeclBuffer;
        wi.new_buf = new_buf;
        wrappers_.push_back(std::move(wi));
        return VisitStmt(op->body);
      } else {
        // Inside thread_extent: keep DeclBuffer with updated buffer
        return DeclBuffer(new_buf, VisitStmt(op->body));
      }
    }
    // Non-device-private: still update buffer data var if it was remapped
    Buffer new_buf = GetNewBuffer(op->buffer);
    if (!new_buf.same_as(op->buffer)) {
      return DeclBuffer(new_buf, VisitStmt(op->body));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // ---- At outermost thread_extent: inject collected wrappers INSIDE its body ----
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent && !wrapped_) {
      wrapped_ = true;
      // Visit the device body first (scope remap of A_shared/B_shared happens here)
      Stmt new_body = VisitStmt(op->body);
      // Inject device-private allocations at the top of the device body
      if (!wrappers_.empty()) {
        for (int i = static_cast<int>(wrappers_.size()) - 1; i >= 0; --i) {
          new_body = wrappers_[i].Apply(new_body);
        }
      }
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // ---- Remap Var references ----
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) return it->second;
    return GetRef<Var>(op);
  }

  // ---- Update Buffer in BufferLoad ----
  // MUST NOT use: auto node = Downcast<BufferLoad>(base).CopyOnWrite()
  // The temporary Downcast<BufferLoad>(base) owns the CopyOnWrite'd copy; when it goes
  // out of scope (at the semicolon), the copy is freed and node becomes dangling.
  // Instead: keep the named ObjectRef alive, then CopyOnWrite through it.
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    Buffer new_buf = GetNewBuffer(op->buffer);
    // Visit indices first (base class handles that).
    // Re-use the base class's index-visiting result, then swap the buffer.
    PrimExpr base = StmtExprMutator::VisitExpr_(op);
    if (new_buf.same_as(op->buffer)) return base;
    // base may be a new BufferLoad (if indices changed) or the original.
    // Extract visited indices from base, then construct with new buffer.
    const auto* bl = base.as<BufferLoadNode>();
    ICHECK(bl) << "Expected BufferLoad after base visit";
    return BufferLoad(new_buf, bl->indices);
  }

  // ---- Update Buffer in BufferStore ----
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Buffer new_buf = GetNewBuffer(op->buffer);
    Stmt base = StmtExprMutator::VisitStmt_(op);
    if (new_buf.same_as(op->buffer)) return base;
    const auto* bs = base.as<BufferStoreNode>();
    ICHECK(bs) << "Expected BufferStore after base visit";
    return BufferStore(new_buf, bs->value, bs->indices);
  }

  // ---- Rebuild lowered attrs from the canonicalized resource state ----
  DictAttrs RewriteLoweredAttrs(const DictAttrs& old_attrs) const {
    Map<String, Any> new_attrs;
    if (old_attrs.defined()) {
      for (const auto& [k, v] : old_attrs->dict) {
        if (k == String(attr::kTLTTProgram)) {
          auto program = v.as<TTProgram>();
          new_attrs.Set(k, program.has_value() ? Any(RewriteTTProgram(program.value())) : v);
          continue;
        }
        if (k == String(attr::kTLBlackholeExecutable)) {
          continue;
        }
        new_attrs.Set(k, v);
      }
    }
    if (auto program_any = new_attrs.Get(String(attr::kTLTTProgram))) {
      auto program = program_any.value().as<TTProgram>();
      if (program.has_value()) {
        new_attrs.Set(attr::kTLBlackholeExecutable,
                      tt_program_projection::MaterializeBlackholeExecutableProjection(
                          program.value()));
      }
    }
    return DictAttrs(new_attrs);
  }
};

// ===========================================================================
// Pass entry point
// ===========================================================================

namespace transform {
using namespace tir::transform;

Pass BlackholeDeviceResourceCanonicalization() {
  auto pass_func = [](PrimFunc f, const IRModule& /*m*/, const PassContext& /*ctx*/) -> PrimFunc {
    // Phase 1: classify device-private resources
    // (This pass is only inserted into the Blackhole pipeline branch in phase.py,
    //  so no target attr check is needed here.)
    BlackholeResourceClassifier classifier;
    classifier.Run(f);

    // If nothing device-private found, skip
    if (classifier.resource_map.empty()) return f;

    // Phase 2: rewrite scopes and relocate allocations
    BlackholeResourceCanonicalizer canonicalizer(classifier.resource_map);
    return canonicalizer.Transform(f);
  };

  return CreatePrimFuncPass(pass_func, 0,
                            "tl.BlackholeDeviceResourceCanonicalization", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BlackholeDeviceResourceCanonicalization",
                        BlackholeDeviceResourceCanonicalization);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
