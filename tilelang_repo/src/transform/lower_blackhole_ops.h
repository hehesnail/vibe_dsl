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
 * \brief Lower TileLang high-level ops to TT-Metal builtins for Blackhole backend
 */

#ifndef TVM_TL_LOWER_BLACKHOLE_OPS_H_
#define TVM_TL_LOWER_BLACKHOLE_OPS_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief CB type for allocation tracking
 */
enum class CBType {
  kInput,        // CB 0-15: Input buffers (Reader -> Compute)
  kOutput,       // CB 16-31: Output buffers (Compute -> Writer)
  kIntermediate  // CB 32-63: Intermediate buffers
};

/*!
 * \brief Copy direction classification
 */
enum class CopyDirection {
  kDramToCB,     // DRAM -> CB (Reader)
  kCBToDram,     // CB -> DRAM (Writer)
  kCBToCB,       // CB -> CB (local copy)
  kDramToDram,   // DRAM -> DRAM (Stage 2 copy pass integration path)
  kUnknown
};

/*!
 * \brief CB requirement description for PlanBlackholeCB
 */
struct CBRequirement {
  std::string name;        // Buffer name
  CBType type;             // CB classification
  int page_size;           // Size of each page in bytes
  int num_pages;           // Number of pages
  std::string data_format; // Data format string

  CBRequirement()
      : type(CBType::kIntermediate), page_size(2048), num_pages(2), data_format("Float16") {}
};

/*!
 * \brief LowerBlackholeOps Pass
 *
 * This pass transforms TileLang high-level operations:
 * - T.copy(A, B) -> CB reserve + NOC read/write + push/pop sequence
 * - T.gemm(A, B, C) -> MM init + tile_regs_acquire + matmul_tiles + pack_tile
 * - T.clear(C) -> tile_regs_acquire (zero DST)
 *
 * Also records CB requirements in function attributes for PlanBlackholeCB.
 */
class LowerBlackholeOps : public tvm::tir::StmtExprMutator {
 public:
  LowerBlackholeOps();

  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

 private:
  /*! \brief CB configuration from function attributes */
  struct CBConfig {
    int in0_id = 0;
    int in1_id = 1;
    int out_id = 16;
    int num_k_tiles = 1;
  };

  /*! \brief Get CB configuration from function attributes */
  CBConfig GetCBConfig() const;

  /*! \brief Allocate a CB ID for a buffer */
  int AllocateCBId(const tvm::tir::Buffer& buffer, CBType type);

  /*! \brief Store CB requirements in function attributes */
  void StoreCBRequirements(tvm::tir::PrimFunc& func);

  /*! \brief Store minimal target mode inferred from lowering */
  void StoreTargetMode(tvm::tir::PrimFunc& func);

  /*! \brief Store runtime argument schema inferred during lowering */
  void StoreRuntimeArgs(tvm::tir::PrimFunc& func);

  /*! \brief Store minimal segment/kernel plan inferred during lowering */
  void StoreSegmentPlan(tvm::tir::PrimFunc& func);

  /*! \brief Detect matmul call using Op comparison (not string matching) */
  bool IsMatmulCall(const tvm::tir::CallNode* op) const;

  /*! \brief Detect clear operation using Op comparison */
  bool IsClearOperation(const tvm::tir::CallNode* op) const;

  /*! \brief Detect copy operation by buffer pattern */
  bool IsCopyOperation(const tvm::tir::BufferStoreNode* op) const;

  /*! \brief Determine copy direction using buffer scopes */
  CopyDirection GetCopyDirection(const tvm::tir::BufferStoreNode* op) const;

  /*! \brief Record Stage 2 copy requirements for a DRAM -> DRAM copy */
  void RecordDramToDramCopy(const tvm::tir::BufferStoreNode* op);

  /*! \brief Estimate a copy tile page size for a buffer */
  int EstimateCopyPageSize(const tvm::tir::Buffer& buffer) const;

  /*! \brief Generate matmul builtin sequence */
  tvm::tir::Stmt GenerateMatmulSequence(const tvm::tir::CallNode* op);

  /*! \brief Generate copy builtin sequence (DRAM->CB, CB->DRAM, CB->CB) */
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op);

  /*! \brief Generate clear builtin sequence */
  tvm::tir::Stmt GenerateClearSequence(const tvm::tir::CallNode* op);

  // StmtExprMutator overrides
  tvm::tir::Stmt VisitStmt_(const tvm::tir::EvaluateNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::BufferStoreNode* op) override;

  tvm::tir::PrimFunc current_func_;
  std::map<tvm::tir::Buffer, int, std::less<>> buffer_to_cb_;
  std::vector<CBRequirement> cb_requirements_;
  bool saw_copy_op_ = false;
  bool saw_matmul_op_ = false;
  bool needs_copy_runtime_args_ = false;
  std::string copy_input_buffer_name_;
  std::string copy_output_buffer_name_;

  // CB allocation counters
  int next_input_cb_ = 0;        // Start at 0
  int next_output_cb_ = 16;      // Start at 16
  int next_intermediate_cb_ = 32; // Start at 32
  int next_cb_id_;
};

/*!
 * \brief Create the LowerBlackholeOps pass
 * \return The pass function
 */
tvm::tir::transform::Pass LowerBlackholeOpsPass();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LOWER_BLACKHOLE_OPS_H_
