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
 * \file lower_blackhole_ops.cc
 * \brief Implementation of LowerBlackholeOps pass.
 *
 * Transforms TileLang high-level operations (T.copy, T.gemm, T.clear)
 * into TT-Metal builtin sequences.
 */

#include "lower_blackhole_ops.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::PrimFunc;
using tir::PrimFuncNode;
using tir::Stmt;
using tir::StmtExprMutator;
using tir::CallNode;
using tir::BufferStoreNode;
using tir::BufferLoadNode;
using tir::EvaluateNode;
using tir::Call;
using tir::Evaluate;
using tir::SeqStmt;
using tir::LetStmt;
using tir::Var;
using tir::Buffer;
using tir::builtin::blackhole_mm_init;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_matmul_tiles;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_noc_async_read;
using tir::builtin::blackhole_noc_async_write;
using tir::builtin::blackhole_noc_async_read_barrier;
using tir::builtin::blackhole_noc_async_write_barrier;
using tvm::Integer;
using tvm::DataType;
using tvm::IntImm;
using tvm::DictAttrs;
using tvm::ffi::GetRef;
using tvm::ffi::String;
using ffi::String;
using tvm::ffi::Map;
using tvm::ffi::Array;
using tvm::ffi::Any;

// Helper to create a call to TT-Metal builtin
static Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

// Helper to create IntImm(32) expression
static PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

// Helper to get storage scope from buffer
static std::string GetStorageScope(const Buffer& buffer) {
  // Use the scope() method which returns ffi::String
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

LowerBlackholeOps::LowerBlackholeOps() : next_cb_id_(0) {
  // Initialize CB allocation tracking
}

PrimFunc LowerBlackholeOps::Transform(const PrimFunc& func) {
  current_func_ = func;
  cb_requirements_.clear();
  next_cb_id_ = 0;

  // Transform the function body
  Stmt body = VisitStmt(func->body);

  // Create new function with transformed body
  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = body;

  // Store CB requirements in function attributes for PlanBlackholeCB
  StoreCBRequirements(new_func);

  return new_func;
}

// Get CB configuration from function attributes
LowerBlackholeOps::CBConfig LowerBlackholeOps::GetCBConfig() const {
  CBConfig config;

  // Try to get CB configuration from function attributes
  if (auto cb_in0 = current_func_->GetAttr<Integer>("tl_cb_in0")) {
    config.in0_id = cb_in0.value()->value;
  }
  if (auto cb_in1 = current_func_->GetAttr<Integer>("tl_cb_in1")) {
    config.in1_id = cb_in1.value()->value;
  }
  if (auto cb_out = current_func_->GetAttr<Integer>("tl_cb_out")) {
    config.out_id = cb_out.value()->value;
  }
  if (auto k_tiles = current_func_->GetAttr<Integer>("tl_k_tiles")) {
    config.num_k_tiles = k_tiles.value()->value;
  }

  return config;
}

// Allocate a CB ID for a buffer
int LowerBlackholeOps::AllocateCBId(const Buffer& buffer, CBType type) {
  // Check if this buffer already has a CB assigned
  auto it = buffer_to_cb_.find(buffer);
  if (it != buffer_to_cb_.end()) {
    return it->second;
  }

  // Allocate new CB ID based on type
  int cb_id;
  switch (type) {
    case CBType::kInput:
      cb_id = next_input_cb_++;
      break;
    case CBType::kOutput:
      cb_id = next_output_cb_++;
      break;
    default:
      cb_id = next_intermediate_cb_++;
      break;
  }

  buffer_to_cb_[buffer] = cb_id;

  // Record CB requirement
  CBRequirement req;
  req.name = buffer->name;
  req.type = type;

  // Calculate page size from buffer shape
  int64_t total_elements = 1;
  for (const auto& shape_dim : buffer->shape) {
    if (const auto* int_imm = shape_dim.as<IntImmNode>()) {
      total_elements *= int_imm->value;
    }
  }
  req.page_size = static_cast<int>(total_elements * buffer->dtype.bytes());
  req.num_pages = 2;  // Default double buffering

  // Determine data format
  if (buffer->dtype.is_float()) {
    if (buffer->dtype.bits() == 16) {
      req.data_format = "Float16";
    } else if (buffer->dtype.bits() == 32) {
      req.data_format = "Float32";
    } else if (buffer->dtype.bits() == 8) {
      req.data_format = "Bfp8";
    }
  } else if (buffer->dtype.is_int()) {
    if (buffer->dtype.bits() == 32) {
      req.data_format = "Int32";
    } else if (buffer->dtype.bits() == 16) {
      req.data_format = "Int16";
    }
  }

  cb_requirements_.push_back(req);

  return cb_id;
}

// Store CB requirements in function attributes
void LowerBlackholeOps::StoreCBRequirements(PrimFunc& func) {
  if (cb_requirements_.empty()) {
    return;
  }

  // Get existing attributes
  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  // Build CB requirements array
  Array<Any> cb_reqs;
  for (const auto& req : cb_requirements_) {
    Map<String, Any> req_map;
    req_map.Set("name", String(req.name));
    req_map.Set("type", String(req.type == CBType::kInput ? "input" :
                               req.type == CBType::kOutput ? "output" : "intermediate"));
    req_map.Set("page_size", Integer(req.page_size));
    req_map.Set("num_pages", Integer(req.num_pages));
    req_map.Set("data_format", String(req.data_format));

    cb_reqs.push_back(req_map);
  }

  attrs.Set("blackhole.cb_requirements", cb_reqs);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

// Detect matmul operation using Op comparison
bool LowerBlackholeOps::IsMatmulCall(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);

  // Direct Op comparison instead of string matching
  static const Op& tl_matmul = Op::Get("tl.matmul");
  static const Op& tl_gemm = Op::Get("tl.gemm");

  return call_op.same_as(tl_matmul) || call_op.same_as(tl_gemm);
}

// Detect clear operation using Op comparison
bool LowerBlackholeOps::IsClearOperation(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);

  // Direct Op comparison
  static const Op& tl_clear = Op::Get("tl.clear");

  return call_op.same_as(tl_clear);
}

// Detect copy operation using buffer scopes
bool LowerBlackholeOps::IsCopyOperation(const BufferStoreNode* op) const {
  // Check if this is a BufferStore where value is a BufferLoad from another buffer
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    return !op->buffer.same_as(load->buffer);
  }
  return false;
}

// Determine copy direction
CopyDirection LowerBlackholeOps::GetCopyDirection(const BufferStoreNode* op) const {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return CopyDirection::kUnknown;

  std::string dst_scope = GetStorageScope(op->buffer);
  std::string src_scope = GetStorageScope(load->buffer);

  // Helper to check if scope indicates CB (shared memory)
  auto isCBScope = [](const std::string& scope) {
    return scope == "shared" || scope == "shared.dyn" || scope.find("shared") == 0;
  };

  // Helper to check if scope indicates DRAM (global memory)
  auto isDRAMScope = [](const std::string& scope) {
    return scope.empty() || scope == "global";
  };

  // DRAM -> CB (global -> shared)
  if (isDRAMScope(src_scope) && isCBScope(dst_scope)) {
    return CopyDirection::kDramToCB;
  }

  // CB -> DRAM (shared -> global)
  if (isCBScope(src_scope) && isDRAMScope(dst_scope)) {
    return CopyDirection::kCBToDram;
  }

  // CB -> CB (shared -> shared)
  if (isCBScope(src_scope) && isCBScope(dst_scope)) {
    return CopyDirection::kCBToCB;
  }

  return CopyDirection::kUnknown;
}

Stmt LowerBlackholeOps::GenerateMatmulSequence(const CallNode* op) {
  CBConfig cb_config = GetCBConfig();

  std::vector<Stmt> stmts;

  // 1. Initialize MM engine
  stmts.push_back(MakeBlackholeCall(
      blackhole_mm_init(),
      {IntImm32(cb_config.in0_id), IntImm32(cb_config.in1_id), IntImm32(cb_config.out_id)}));

  // 2. Acquire tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));

  // 3. Generate K-tile loop
  for (int kt = 0; kt < cb_config.num_k_tiles; ++kt) {
    // Wait for input tiles
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(),
        {IntImm32(cb_config.in0_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(),
        {IntImm32(cb_config.in1_id), IntImm32(1)}));

    // Perform matmul
    stmts.push_back(MakeBlackholeCall(
        blackhole_matmul_tiles(),
        {IntImm32(cb_config.in0_id), IntImm32(cb_config.in1_id),
         IntImm32(0), IntImm32(0), IntImm32(0)}));

    // Pop input tiles
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(cb_config.in0_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(cb_config.in1_id), IntImm32(1)}));
  }

  // 4-5. Commit and wait
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));

  // 6-8. Pack and push output
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_reserve_back(),
      {IntImm32(cb_config.out_id), IntImm32(1)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_pack_tile(),
      {IntImm32(0), IntImm32(cb_config.out_id)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_push_back(),
      {IntImm32(cb_config.out_id), IntImm32(1)}));

  // 9. Release tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));

  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateCopySequence(const BufferStoreNode* op) {
  CopyDirection direction = GetCopyDirection(op);

  if (direction == CopyDirection::kUnknown) {
    LOG(WARNING) << "LowerBlackholeOps: Unknown copy direction, falling back";
    return StmtExprMutator::VisitStmt_(op);
  }

  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return StmtExprMutator::VisitStmt_(op);

  std::vector<Stmt> stmts;

  switch (direction) {
    case CopyDirection::kDramToCB: {
      // DRAM -> CB (Reader pattern)
      int cb_id = AllocateCBId(op->buffer, CBType::kInput);
      int tile_size = 2048;  // Default tile size, should be calculated from buffer

      // cb_reserve_back(cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));

      // Note: The actual noc_async_read call requires runtime address calculation
      // This is handled by CodeGen which generates:
      // noc_async_read(get_noc_addr(tile_idx, addr_gen), get_write_ptr(cb_id), tile_size)

      // For now, we emit a builtin call that CodeGen will handle
      // The DRAM address is passed as a runtime argument
      // Buffer destination is the CB

      // noc_async_read_barrier()
      stmts.push_back(MakeBlackholeCall(blackhole_noc_async_read_barrier(), {}));

      // cb_push_back(cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      break;
    }

    case CopyDirection::kCBToDram: {
      // CB -> DRAM (Writer pattern)
      int cb_id = AllocateCBId(load->buffer, CBType::kOutput);

      // cb_wait_front(cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));

      // Note: CodeGen generates:
      // noc_async_write(get_read_ptr(cb_id), get_noc_addr(tile_idx, addr_gen), tile_size)

      // noc_async_write_barrier()
      stmts.push_back(MakeBlackholeCall(blackhole_noc_async_write_barrier(), {}));

      // cb_pop_front(cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      break;
    }

    case CopyDirection::kCBToCB: {
      // CB -> CB (local copy)
      int src_cb_id = AllocateCBId(load->buffer, CBType::kIntermediate);
      int dst_cb_id = AllocateCBId(op->buffer, CBType::kIntermediate);

      // cb_wait_front(src_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(src_cb_id), IntImm32(1)}));

      // cb_reserve_back(dst_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(dst_cb_id), IntImm32(1)}));

      // Note: local copy would use memcpy or similar
      // For now, just pop and push markers

      // cb_push_back(dst_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(dst_cb_id), IntImm32(1)}));

      // cb_pop_front(src_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(src_cb_id), IntImm32(1)}));
      break;
    }

    default:
      return StmtExprMutator::VisitStmt_(op);
  }

  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateClearSequence(const CallNode* op) {
  // Clear operation: tile_regs_acquire() to zero DST registers
  // In full implementation, would also zero-fill
  return MakeBlackholeCall(blackhole_tile_regs_acquire(), {});
}

// StmtExprMutator overrides
// Note: We only override specific node types and return the original node
// for unmatched patterns to avoid deep recursion that causes stack overflow.
Stmt LowerBlackholeOps::VisitStmt_(const EvaluateNode* op) {
  if (const auto* call = op->value.as<CallNode>()) {
    if (IsMatmulCall(call)) {
      return GenerateMatmulSequence(call);
    }
    if (IsClearOperation(call)) {
      return GenerateClearSequence(call);
    }
  }
  // Return original statement without recursion to avoid stack overflow
  // The parent class's VisitStmt_ would recursively visit child nodes,
  // which can cause deep recursion for deeply nested IR trees.
  return GetRef<Stmt>(op);
}

Stmt LowerBlackholeOps::VisitStmt_(const BufferStoreNode* op) {
  if (IsCopyOperation(op)) {
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

// Modern TVM pass registration using CreatePrimFuncPass
tir::transform::Pass LowerBlackholeOpsPass() {
  auto fpass = [](PrimFunc func, IRModule m, tir::transform::PassContext ctx) -> PrimFunc {
    return LowerBlackholeOps().Transform(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.LowerBlackholeOps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.LowerBlackholeOps", LowerBlackholeOpsPass);
}

}  // namespace tl
}  // namespace tvm
