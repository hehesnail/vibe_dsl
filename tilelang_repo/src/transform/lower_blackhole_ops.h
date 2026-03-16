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
 * \file lower_blackhole_ops.h
 * \brief Lower TileLang ops to TT-Metal specific ops for Blackhole.
 */
#ifndef TL_TRANSFORM_LOWER_BLACKHOLE_OPS_H_
#define TL_TRANSFORM_LOWER_BLACKHOLE_OPS_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief Lower TileLang operations to TT-Metal specific builtin calls.
 *
 * This pass transforms:
 * - T.gemm() -> cb_wait_front + matmul_tiles + cb_push_back sequence
 * - T.copy() -> cb_reserve_back + noc_async_read/write + cb_push_back
 * - T.clear() -> tile_regs_acquire (implicitly zeros DST)
 *
 * The output TIR contains only TT-Metal specific builtin calls that
 * can be directly translated to C++ code by CodeGenBlackhole.
 */
class LowerBlackholeOps : public tvm::tir::StmtExprMutator {
 public:
  /*!\brief Constructor */
  LowerBlackholeOps();

  /*!
   * \brief Transform a PrimFunc to TT-Metal specific form.
   * \param func The input PrimFunc
   * \return The transformed PrimFunc
   */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

 private:
  // Detect specific TileLang operations
  bool IsMatmulCall(const tvm::tir::CallNode* op) const;
  bool IsCopyOperation(const tvm::tir::BufferStoreNode* op) const;
  bool IsClearOperation(const tvm::tir::CallNode* op) const;

  // Generate TT-Metal sequences
  tvm::tir::Stmt GenerateMatmulSequence(const tvm::tir::CallNode* op);
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op);
  tvm::tir::Stmt GenerateClearSequence(const tvm::tir::CallNode* op);

  // Helper to create builtin call
  tvm::tir::Stmt MakeTTMetalCall(const std::string& builtin_name,
                                  const std::vector<tvm::PrimExpr>& args);

  // StmtExprMutator overrides
  tvm::tir::Stmt VisitStmt_(const tvm::tir::EvaluateNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::BufferStoreNode* op) override;

  // Extract CB configuration from function attributes
  struct CBConfig {
    int in0_id = 0;
    int in1_id = 1;
    int out_id = 16;
    int num_k_tiles = 1;
  };
  CBConfig GetCBConfig() const;

  // Current function context
  mutable tvm::tir::PrimFunc current_func_;
};

/*!
 * \brief Create the LowerBlackholeOps pass.
 * \return The pass object
 */
tvm::tir::transform::Pass LowerBlackholeOpsPass();

}  // namespace tl
}  // namespace tvm

#endif  // TL_TRANSFORM_LOWER_BLACKHOLE_OPS_H_
