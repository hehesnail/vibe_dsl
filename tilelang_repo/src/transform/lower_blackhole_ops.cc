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
using tvm::Integer;
using tvm::DataType;
using tvm::IntImm;

// Helper to create a call to TT-Metal builtin
static Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

static PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

LowerBlackholeOps::LowerBlackholeOps() = default;

PrimFunc LowerBlackholeOps::Transform(const PrimFunc& func) {
  current_func_ = func;

  // Get CB configuration from function attributes
  CBConfig cb_config = GetCBConfig();

  // Transform the function body
  Stmt body = VisitStmt(func->body);

  // Create new function with transformed body
  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = body;

  return new_func;
}

LowerBlackholeOps::CBConfig LowerBlackholeOps::GetCBConfig() const {
  CBConfig config;

  // Try to get CB configuration from function attributes using GetAttr
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

bool LowerBlackholeOps::IsMatmulCall(const CallNode* op) const {
  // Check if this is a TileLang matmul call
  // Pattern: T.gemm(A_shared, B_shared, C_local)
  // In TIR, this appears as a Call to a specific intrinsic

  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;

  // Check for TileLang matmul intrinsic name patterns
  return (op_name.find("gemm") != std::string::npos ||
          op_name.find("matmul") != std::string::npos ||
          op_name == "tl.matmul");
}

bool LowerBlackholeOps::IsCopyOperation(const BufferStoreNode* op) const {
  // Detect copy operations: typically a store from one buffer to another
  // This is a heuristic - in practice, we'd check for specific patterns

  // For now, identify BufferStore where value is a BufferLoad from another buffer
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    // If storing from one buffer to another, it's likely a copy
    return !op->buffer.same_as(load->buffer);
  }
  return false;
}

bool LowerBlackholeOps::IsClearOperation(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;

  return (op_name.find("clear") != std::string::npos ||
          op_name == "tl.clear");
}

Stmt LowerBlackholeOps::GenerateMatmulSequence(const CallNode* op) {
  CBConfig cb_config = GetCBConfig();

  // Build the matmul sequence:
  // 1. mm_init(in0_cb, in1_cb, out_cb)
  // 2. tile_regs_acquire()
  // 3. For each K tile:
  //    - cb_wait_front(in0_cb, 1)
  //    - cb_wait_front(in1_cb, 1)
  //    - matmul_tiles(in0_cb, in1_cb, 0, 0, 0)
  //    - cb_pop_front(in0_cb, 1)
  //    - cb_pop_front(in1_cb, 1)
  // 4. tile_regs_commit()
  // 5. tile_regs_wait()
  // 6. cb_reserve_back(out_cb, 1)
  // 7. pack_tile(0, out_cb)
  // 8. cb_push_back(out_cb, 1)
  // 9. tile_regs_release()

  std::vector<Stmt> stmts;

  // 1. Initialize MM engine
  stmts.push_back(MakeBlackholeCall(
      blackhole_mm_init(),
      {IntImm32(cb_config.in0_id), IntImm32(cb_config.in1_id), IntImm32(cb_config.out_id)}));

  // 2. Acquire tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));

  // 3. Generate K-tile loop if needed
  // For simplicity, we'll unroll for now with the configured number of K tiles
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
  // For copy operations, we need to determine:
  // - Source: DRAM or CB
  // - Destination: CB or DRAM
  // For now, assume DRAM -> CB (reader) or CB -> DRAM (writer)

  // This is a simplified implementation
  // In practice, we'd need to analyze the buffer scopes

  // For demonstration, generate a simple noc_async_read sequence
  std::vector<Stmt> stmts;

  // cb_reserve_back(cb_id, num_tiles)
  // noc_async_read(src_dram_addr, cb_addr, size)
  // noc_async_read_barrier()
  // cb_push_back(cb_id, num_tiles)

  // Placeholder implementation
  return VisitStmt_(op);
}

Stmt LowerBlackholeOps::GenerateClearSequence(const CallNode* op) {
  // Clear operation: tile_regs_acquire() zeros the DST registers
  return MakeBlackholeCall(blackhole_tile_regs_acquire(), {});
}

// StmtExprMutator overrides
Stmt LowerBlackholeOps::VisitStmt_(const EvaluateNode* op) {
  if (const auto* call = op->value.as<CallNode>()) {
    if (IsMatmulCall(call)) {
      return GenerateMatmulSequence(call);
    }
    if (IsClearOperation(call)) {
      return GenerateClearSequence(call);
    }
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt LowerBlackholeOps::VisitStmt_(const BufferStoreNode* op) {
  if (IsCopyOperation(op)) {
    return GenerateCopySequence(op);
  }
  return StmtExprMutator::VisitStmt_(op);
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
