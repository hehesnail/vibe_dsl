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
 * \file codegen_blackhole.h
 * \brief Generate TT-Metal code for Blackhole backend.
 */
#ifndef TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
#define TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_

#include <string>
#include <unordered_set>

#include "codegen_c_host.h"

namespace tvm {
namespace tl {

/*!
 * \brief CodeGen for Blackhole (TT-Metal) backend.
 *
 * Generates C++ code for Tenstorrent Blackhole architecture.
 * Supports BRISC, TRISC, and NCRISC core types.
 */
class CodeGenBlackhole : public CodeGenCHost {
 public:
  CodeGenBlackhole();

  // Note: Parent class Init is not virtual, so we just shadow it
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tir::PrimFunc &f) override;

  // Generate TT-Metal kernel_main entry point
  void GenerateKernelMain(const tvm::GlobalVar &gvar,
                          const tvm::tir::PrimFunc &f);

  // Override visitor to handle TT-Metal builtin calls
  void VisitExpr_(const tvm::tir::CallNode *op,
                  std::ostream &os) override;

  // Override to handle FloorDiv/FloorMod (not implemented in base CodeGenC)
  void VisitExpr_(const tvm::tir::FloorDivNode *op,
                  std::ostream &os) override;
  void VisitExpr_(const tvm::tir::FloorModNode *op,
                  std::ostream &os) override;

  // Override AttrStmt visitor to handle CUDA-specific attributes
  void VisitStmt_(const tvm::tir::AttrStmtNode *op) override;

  // Override storage scope printing for Blackhole memory types
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) override;

  // Override thread index binding for Blackhole
  void BindThreadIndex(const tvm::tir::IterVar &iv) override;

  // Note: PrintFuncPrefix and PrintType are final in parent class,
  // so we don't override them here. Blackhole-specific handling
  // is done through visitor and AddFunction.

  // Blackhole core type enumeration
  enum class CoreType {
    kBRISC,   // Broadcast RISC - control core
    kTRISC,   // Tensix RISC - compute core
    kNCRISC,  // NOC RISC - data movement core
    kUnknown
  };

  // Set the core type for code generation
  void SetCoreType(CoreType core_type) { core_type_ = core_type; }

  // Get the core type
  CoreType GetCoreType() const { return core_type_; }

  // Note: Visitor methods (VisitExpr_, VisitStmt_) are marked 'final' in the
  // CodeGenCHost/CodeGenC parent classes and cannot be overridden further.
  // Blackhole-specific IR handling should be performed via preprocessing passes
  // that transform the TIR before it reaches the CodeGen stage.

 protected:
  // Print TT-Metal specific function attributes
  void PrintKernelAttributes();

  // Handle TT-Metal builtin calls (from VisitExpr_)
  bool HandleBlackholeBuiltin(const tvm::tir::CallNode *op, std::ostream &os);

  // Print CB (Circular Buffer) operations
  void PrintCBReserveBack(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBPushBack(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBWaitFront(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBPopFront(const tvm::tir::CallNode *op, std::ostream &os);

  // Print NOC operations
  void PrintNOCAsyncRead(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintNOCAsyncWrite(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintNOCReadBarrier(std::ostream &os);
  void PrintNOCWriteBarrier(std::ostream &os);

  // Print compute operations
  void PrintMMInit(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMatmulTiles(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintTileRegsAcquire(std::ostream &os);
  void PrintTileRegsCommit(std::ostream &os);
  void PrintTileRegsWait(std::ostream &os);
  void PrintTileRegsRelease(std::ostream &os);
  void PrintPackTile(const tvm::tir::CallNode *op, std::ostream &os);

  // Legacy: Print CB operations (old interface)
  void PrintCBDeclare(const std::string &name, tvm::DataType dtype,
                      int num_pages, int page_size);
  void PrintCBWaitFront(const std::string &name, int num_tiles);
  void PrintCBPopFront(const std::string &name, int num_tiles);
  void PrintCBReserveBack(const std::string &name, int num_tiles);
  void PrintCBPushBack(const std::string &name, int num_tiles);

  // Legacy: Print NOC operations (old interface)
  void PrintNOCRead(const std::string &src_addr, const std::string &dst_addr,
                    int size);
  void PrintNOCWrite(const std::string &src_addr, const std::string &dst_addr,
                     int size);
  void PrintNOCWait();

  // Print semaphore operations
  void PrintSemInit(int sem_id, int value);
  void PrintSemWait(int sem_id, int value);
  void PrintSemPost(int sem_id);

 public:
  // ==========================================================================
  // Copy Kernel Generation (Phase 1)
  // ==========================================================================

  /*!\brief Generate a simple copy kernel for single-core execution
   *
   * Generates both reader and writer kernels that run sequentially
   * on the same core, using a circular buffer for data transfer.
   *
   * \param func_name Name of the generated kernel
   * \param src_buf Source buffer name
   * \param dst_buf Destination buffer name
   * \param num_tiles Number of tiles to copy
   * \param tile_size_bytes Size of each tile in bytes
   * \return Generated C++ code string
   */
  std::string GenerateSimpleCopyKernel(const std::string& func_name,
                                       const std::string& src_buf,
                                       const std::string& dst_buf,
                                       int num_tiles,
                                       int tile_size_bytes = 2048);

  /*!\brief Generate reader kernel (BRISC)
   *
   * Reads data from DRAM into circular buffer.
   */
  std::string GenerateReaderKernel(const std::string& func_name,
                                   const std::string& src_buf,
                                   int cb_id,
                                   int num_tiles,
                                   int tile_size_bytes);

  /*!\brief Generate writer kernel (NCRISC)
   *
   * Writes data from circular buffer to DRAM.
   */
  std::string GenerateWriterKernel(const std::string& func_name,
                                   const std::string& dst_buf,
                                   int cb_id,
                                   int num_tiles,
                                   int tile_size_bytes);

 private:
  // Current core type being generated
  CoreType core_type_{CoreType::kUnknown};

  // TT-Metal specific state
  bool need_tt_metal_h_{false};
  bool need_dataflow_api_h_{false};
  bool need_compute_api_h_{false};

  // Whether to emit kernel entry point wrapper
  bool emit_kernel_wrapper_{true};

  // Track declared CBs
  std::unordered_set<std::string> declared_cbs_;

  // Default L1 memory alignment
  static constexpr int kL1Alignment = 16;
};

} // namespace tl
} // namespace tvm

#endif // TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
