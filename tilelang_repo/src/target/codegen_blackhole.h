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

  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices) override;

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tir::PrimFunc &f) override;

  // Override codegen methods for Blackhole-specific handling
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(tvm::DataType t, std::ostream &os) final;

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

  // Visitor overrides for Blackhole-specific intrinsics
  void VisitExpr_(const tvm::tir::CallNode *op, std::ostream &os) override;
  void VisitStmt_(const tvm::tir::AttrStmtNode *op) override;
  void VisitStmt_(const tvm::tir::ForNode *op) override;

 protected:
  // Print TT-Metal specific function attributes
  void PrintKernelAttributes();

  // Print CB (Circular Buffer) operations
  void PrintCBDeclare(const std::string &name, tvm::DataType dtype,
                      int num_pages, int page_size);
  void PrintCBWaitFront(const std::string &name, int num_tiles);
  void PrintCBPopFront(const std::string &name, int num_tiles);
  void PrintCBReserveBack(const std::string &name, int num_tiles);
  void PrintCBPushBack(const std::string &name, int num_tiles);

  // Print NOC operations
  void PrintNOCRead(const std::string &src_addr, const std::string &dst_addr,
                    int size);
  void PrintNOCWrite(const std::string &src_addr, const std::string &dst_addr,
                     int size);
  void PrintNOCWait();

  // Print semaphore operations
  void PrintSemInit(int sem_id, int value);
  void PrintSemWait(int sem_id, int value);
  void PrintSemPost(int sem_id);

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
