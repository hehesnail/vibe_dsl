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
 * \file codegen_blackhole.cc
 * \brief Generate TT-Metal code for Blackhole backend.
 */

#include "codegen_blackhole.h"

#include <sstream>
#include <string>

#include "tvm/tir/builtin.h"
#include "tvm/tir/op.h"
#include "tvm/tir/transform.h"

namespace tvm {
namespace tl {

CodeGenBlackhole::CodeGenBlackhole() = default;

void CodeGenBlackhole::Init(bool output_ssa, bool emit_asserts,
                            bool emit_fwd_func_decl, std::string target_str,
                            const std::unordered_set<std::string> &devices) {
  CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl,
                     target_str, devices);

  // Set default core type to TRISC (compute core)
  core_type_ = CoreType::kTRISC;
}

void CodeGenBlackhole::AddFunction(const tvm::GlobalVar &gvar,
                                   const tvm::tir::PrimFunc &f) {
  // Emit TT-Metal headers if needed
  if (need_tt_metal_h_) {
    decl_stream << "#include <cstdint>\n";
    decl_stream << "#include <cmath>\n";
    decl_stream << "\n";
  }
  if (need_dataflow_api_h_) {
    decl_stream << "// DataMovement kernel API (BRISC/NCRISC)\n";
    decl_stream << "#include \"dataflow_api.h\"\n";
    decl_stream << "\n";
  }
  if (need_compute_api_h_) {
    decl_stream << "// Compute kernel API (TRISC)\n";
    decl_stream << "#include \"compute_kernel_api.h\"\n";
    decl_stream << "\n";
  }

  // Detect core type from function attributes
  // Note: Using tvm::attr::kGlobalSymbol to detect kernel type
  auto global_symbol = f->GetAttr<tvm::ffi::String>(tvm::attr::kGlobalSymbol);
  if (global_symbol) {
    std::string symbol = global_symbol.value();
    // Determine core type from symbol name or other attributes
    // This is a heuristic - adjust based on naming convention
    if (symbol.find("_brisc") != std::string::npos) {
      core_type_ = CoreType::kBRISC;
    } else if (symbol.find("_ncrisc") != std::string::npos) {
      core_type_ = CoreType::kNCRISC;
    } else if (symbol.find("_trisc") != std::string::npos) {
      core_type_ = CoreType::kTRISC;
    }
  }

  // Call parent AddFunction
  CodeGenCHost::AddFunction(gvar, f);
}

// Note: PrintFuncPrefix, PrintType, and VisitExpr_ are final in parent class
// and cannot be overridden. Blackhole-specific handling is done through
// AddFunction and VisitStmt_ methods, or by preprocessing the IR.

// Note: VisitStmt_ for AttrStmtNode is final in parent class, so we cannot
// override it here. CB allocation handling should be done through IR passes
// that transform the IR before codegen, or through other mechanisms.

// Note: VisitStmt_ for ForNode is final in parent class, so we cannot override.
// Loop handling customization should be done via IR passes or other mechanisms.

void CodeGenBlackhole::PrintKernelAttributes() {
  // Print kernel-specific attributes for TT-Metal
  // This is a placeholder for future kernel attribute emission
}

void CodeGenBlackhole::PrintCBDeclare(const std::string &name,
                                      tvm::DataType dtype, int num_pages,
                                      int page_size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// CB declaration: " << name << "\n";
  PrintIndent();
  stream << "// TODO: Implement CB allocation\n";
}

void CodeGenBlackhole::PrintCBWaitFront(const std::string &name,
                                        int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_wait_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPopFront(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_pop_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBReserveBack(const std::string &name,
                                          int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_reserve_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPushBack(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_push_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintNOCRead(const std::string &src_addr,
                                    const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC read: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWrite(const std::string &src_addr,
                                     const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC write: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWait() {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "noc_async_read_barrier();\n";
}

void CodeGenBlackhole::PrintSemInit(int sem_id, int value) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// Semaphore init: " << sem_id << " = " << value << "\n";
}

void CodeGenBlackhole::PrintSemWait(int sem_id, int value) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// Semaphore wait: " << sem_id << " == " << value << "\n";
}

void CodeGenBlackhole::PrintSemPost(int sem_id) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// Semaphore post: " << sem_id << "\n";
}

}  // namespace tl
}  // namespace tvm
