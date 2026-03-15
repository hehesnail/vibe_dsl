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
    stream << "#include <cstdint>\n";
    stream << "#include <cmath>\n";
    stream << "\n";
  }
  if (need_dataflow_api_h_) {
    stream << "// DataMovement kernel API (BRISC/NCRISC)\n";
    stream << "#include \"dataflow_api.h\"\n";
    stream << "\n";
  }
  if (need_compute_api_h_) {
    stream << "// Compute kernel API (TRISC)\n";
    stream << "#include \"compute_kernel_api.h\"\n";
    stream << "\n";
  }

  // Detect core type from function attributes
  auto core_type_attr = f->attrs.GetAttr<tvm::String>(tvm::attr::kKernel);
  if (core_type_attr.defined()) {
    std::string kernel_type = core_type_attr.value();
    if (kernel_type == "brisc") {
      core_type_ = CoreType::kBRISC;
    } else if (kernel_type == "trisc") {
      core_type_ = CoreType::kTRISC;
    } else if (kernel_type == "ncrisc") {
      core_type_ = CoreType::kNCRISC;
    }
  }

  // Call parent AddFunction
  CodeGenCHost::AddFunction(gvar, f);
}

void CodeGenBlackhole::PrintFuncPrefix(std::ostream &os) {
  // TT-Metal kernels don't need special prefixes
  // They are compiled as RISC-V ELF by the TT-Metal build system
  os << "void";
}

void CodeGenBlackhole::PrintType(tvm::DataType t, std::ostream &os) {
  // Handle TT-Metal specific vector types if needed
  // Default to parent implementation
  CodeGenCHost::PrintType(t, os);
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::CallNode *op,
                                  std::ostream &os) {
  // Handle TT-Metal specific intrinsics
  if (op->op.same_as(tvm::tir::builtin::call_extern) ||
      op->op.same_as(tvm::tir::builtin::call_pure_extern)) {
    std::string func_name = op->args[0].as<tvm::tir::StringImmNode>()->value;

    // Handle CB operations
    if (func_name == "tt_cb_wait_front") {
      ICHECK_EQ(op->args.size(), 3);
      std::string cb_name = op->args[1].as<tvm::tir::StringImmNode>()->value;
      int num_tiles = op->args[2].as<tvm::tir::IntImmNode>()->value;
      PrintCBWaitFront(cb_name, num_tiles);
      return;
    } else if (func_name == "tt_cb_pop_front") {
      ICHECK_EQ(op->args.size(), 3);
      std::string cb_name = op->args[1].as<tvm::tir::StringImmNode>()->value;
      int num_tiles = op->args[2].as<tvm::tir::IntImmNode>()->value;
      PrintCBPopFront(cb_name, num_tiles);
      return;
    } else if (func_name == "tt_cb_reserve_back") {
      ICHECK_EQ(op->args.size(), 3);
      std::string cb_name = op->args[1].as<tvm::tir::StringImmNode>()->value;
      int num_tiles = op->args[2].as<tvm::tir::IntImmNode>()->value;
      PrintCBReserveBack(cb_name, num_tiles);
      return;
    } else if (func_name == "tt_cb_push_back") {
      ICHECK_EQ(op->args.size(), 3);
      std::string cb_name = op->args[1].as<tvm::tir::StringImmNode>()->value;
      int num_tiles = op->args[2].as<tvm::tir::IntImmNode>()->value;
      PrintCBPushBack(cb_name, num_tiles);
      return;
    }

    // Handle NOC operations
    if (func_name == "tt_noc_read") {
      // TODO: Implement NOC read
      need_dataflow_api_h_ = true;
    } else if (func_name == "tt_noc_write") {
      // TODO: Implement NOC write
      need_dataflow_api_h_ = true;
    }
  }

  // Fall back to parent implementation
  CodeGenCHost::VisitExpr_(op, os);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::AttrStmtNode *op) {
  // Handle Blackhole-specific attributes
  if (op->attr_key == tvm::attr::kKernel) {
    // Kernel type attribute already processed in AddFunction
    this->PrintStmt(op->body);
    return;
  }

  // Handle CB allocation attributes
  if (op->attr_key == "tt_cb") {
    // CB allocation directive
    this->PrintStmt(op->body);
    return;
  }

  // Fall back to parent implementation
  CodeGenCHost::VisitStmt_(op);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::ForNode *op) {
  // For Blackhole, we use the standard loop generation
  // TT-Metal compiler will handle loop unrolling if needed
  CodeGenCHost::VisitStmt_(op);
}

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
