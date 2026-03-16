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

#include "../tir/builtin_blackhole.h"
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

// ============================================================================
// Visitor Implementation for TT-Metal Builtin Calls
// ============================================================================

void CodeGenBlackhole::VisitExpr_(const tvm::tir::CallNode *op,
                                  std::ostream &os) {
  // Try to handle TT-Metal builtin calls
  if (HandleBlackholeBuiltin(op, os)) {
    return;
  }
  // Fall back to parent class for other calls
  CodeGenCHost::VisitExpr_(op, os);
}

bool CodeGenBlackhole::HandleBlackholeBuiltin(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;

  // Check for TT-Metal builtin prefix
  const std::string prefix = "tl.blackhole.";
  if (op_name.find(prefix) != 0) return false;

  std::string builtin_name = op_name.substr(prefix.length());

  // Handle each builtin type
  if (builtin_name == "cb_reserve_back") {
    PrintCBReserveBack(op, os);
    return true;
  } else if (builtin_name == "cb_push_back") {
    PrintCBPushBack(op, os);
    return true;
  } else if (builtin_name == "cb_wait_front") {
    PrintCBWaitFront(op, os);
    return true;
  } else if (builtin_name == "cb_pop_front") {
    PrintCBPopFront(op, os);
    return true;
  } else if (builtin_name == "noc_async_read") {
    PrintNOCAsyncRead(op, os);
    return true;
  } else if (builtin_name == "noc_async_write") {
    PrintNOCAsyncWrite(op, os);
    return true;
  } else if (builtin_name == "noc_async_read_barrier") {
    PrintNOCReadBarrier(os);
    return true;
  } else if (builtin_name == "noc_async_write_barrier") {
    PrintNOCWriteBarrier(os);
    return true;
  } else if (builtin_name == "mm_init") {
    PrintMMInit(op, os);
    return true;
  } else if (builtin_name == "matmul_tiles") {
    PrintMatmulTiles(op, os);
    return true;
  } else if (builtin_name == "tile_regs_acquire") {
    PrintTileRegsAcquire(os);
    return true;
  } else if (builtin_name == "tile_regs_commit") {
    PrintTileRegsCommit(os);
    return true;
  } else if (builtin_name == "tile_regs_wait") {
    PrintTileRegsWait(os);
    return true;
  } else if (builtin_name == "tile_regs_release") {
    PrintTileRegsRelease(os);
    return true;
  } else if (builtin_name == "pack_tile") {
    PrintPackTile(op, os);
    return true;
  }

  return false;
}

// ============================================================================
// TT-Metal Builtin Print Functions
// ============================================================================

void CodeGenBlackhole::PrintCBReserveBack(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_reserve_back(";
  PrintExpr(op->args[0], os);  // cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPushBack(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_push_back(";
  PrintExpr(op->args[0], os);  // cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBWaitFront(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_wait_front(";
  PrintExpr(op->args[0], os);  // cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPopFront(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_pop_front(";
  PrintExpr(op->args[0], os);  // cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncRead(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncWrite(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCReadBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read_barrier()";
}

void CodeGenBlackhole::PrintNOCWriteBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write_barrier()";
}

void CodeGenBlackhole::PrintMMInit(const tvm::tir::CallNode *op,
                                   std::ostream &os) {
  need_compute_api_h_ = true;
  os << "mm_init(";
  PrintExpr(op->args[0], os);  // in0_cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // in1_cb_id
  os << ", ";
  PrintExpr(op->args[2], os);  // out_cb_id
  os << ")";
}

void CodeGenBlackhole::PrintMatmulTiles(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_compute_api_h_ = true;
  os << "matmul_tiles(";
  PrintExpr(op->args[0], os);  // in0_cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // in1_cb_id
  os << ", ";
  PrintExpr(op->args[2], os);  // in0_tile_index
  os << ", ";
  PrintExpr(op->args[3], os);  // in1_tile_index
  os << ", ";
  PrintExpr(op->args[4], os);  // dst_tile_index
  os << ")";
}

void CodeGenBlackhole::PrintTileRegsAcquire(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_acquire()";
}

void CodeGenBlackhole::PrintTileRegsCommit(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_commit()";
}

void CodeGenBlackhole::PrintTileRegsWait(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_wait()";
}

void CodeGenBlackhole::PrintTileRegsRelease(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_release()";
}

void CodeGenBlackhole::PrintPackTile(const tvm::tir::CallNode *op,
                                     std::ostream &os) {
  need_compute_api_h_ = true;
  os << "pack_tile(";
  PrintExpr(op->args[0], os);  // src_tile_index
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_cb_id
  os << ")";
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

// ============================================================================
// Copy Kernel Generation (Phase 1)
// ============================================================================

std::string CodeGenBlackhole::GenerateSimpleCopyKernel(
    const std::string& func_name, const std::string& src_buf,
    const std::string& dst_buf, int num_tiles, int tile_size_bytes) {
  std::ostringstream os;

  // Header
  os << "// Simple Copy Kernel for Blackhole\n";
  os << "// Generated by TileLang CodeGenBlackhole\n";
  os << "// Function: " << func_name << "\n";
  os << "// Operation: Copy " << num_tiles << " tiles (" << tile_size_bytes
     << " bytes each)\n";
  os << "\n";

  // Include dataflow API
  os << "#include \"dataflow_api.h\"\n\n";

  // Reader kernel
  os << "// Reader Kernel - BRISC\n";
  os << GenerateReaderKernel(func_name + "_reader", src_buf, 0, num_tiles,
                             tile_size_bytes);
  os << "\n";

  // Writer kernel
  os << "// Writer Kernel - NCRISC\n";
  os << GenerateWriterKernel(func_name + "_writer", dst_buf, 0, num_tiles,
                             tile_size_bytes);
  os << "\n";

  // Combined kernel (sequential execution) - TT-Sim compatible version
  os << "// Combined Kernel - Single Core Sequential Execution\n";
  os << "// TT-Sim compatible: uses InterleavedAddrGen for address translation\n";
  os << "void " << func_name << "(uint64_t src_dram_addr, uint64_t dst_dram_addr) {\n";
  os << "  // Execute reader then writer sequentially on same core\n";
  os << "  // CB synchronization ensures data consistency\n";
  os << "  \n";
  os << "  constexpr uint32_t tile_size = " << tile_size_bytes << ";\n";
  os << "  constexpr uint32_t num_tiles = " << num_tiles << ";\n";
  os << "  \n";
  os << "  // Create address generators for TT-Sim compatibility\n";
  os << "  InterleavedAddrGen<true> src_gen = {\n";
  os << "    .bank_base_address = (uint32_t)src_dram_addr,\n";
  os << "    .page_size = tile_size\n";
  os << "  };\n";
  os << "  InterleavedAddrGen<true> dst_gen = {\n";
  os << "    .bank_base_address = (uint32_t)dst_dram_addr,\n";
  os << "    .page_size = tile_size\n";
  os << "  };\n";
  os << "  \n";
  os << "  // Reader: DRAM -> CB\n";
  os << "  for (uint32_t i = 0; i < num_tiles; i++) {\n";
  os << "    cb_reserve_back(0, 1);\n";
  os << "    uint32_t write_ptr = get_write_ptr(0);\n";
  os << "    uint64_t src_noc_addr = get_noc_addr(i, src_gen);\n";
  os << "    noc_async_read(src_noc_addr, write_ptr, tile_size);\n";
  os << "    noc_async_read_barrier();\n";
  os << "    cb_push_back(0, 1);\n";
  os << "  }\n";
  os << "  \n";
  os << "  // Writer: CB -> DRAM\n";
  os << "  for (uint32_t i = 0; i < num_tiles; i++) {\n";
  os << "    cb_wait_front(0, 1);\n";
  os << "    uint32_t read_ptr = get_read_ptr(0);\n";
  os << "    uint64_t dst_noc_addr = get_noc_addr(i, dst_gen);\n";
  os << "    noc_async_write(read_ptr, dst_noc_addr, tile_size);\n";
  os << "    noc_async_write_barrier();\n";
  os << "    cb_pop_front(0, 1);\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

std::string CodeGenBlackhole::GenerateReaderKernel(const std::string& func_name,
                                                   const std::string& src_buf,
                                                   int cb_id, int num_tiles,
                                                   int tile_size_bytes) {
  std::ostringstream os;

  os << "void " << func_name << "() {\n";
  os << "  // Runtime arguments\n";
  os << "  uint64_t src_dram_addr = get_arg_val<uint32_t>(0);\n";
  os << "  uint64_t src_dram_addr_hi = get_arg_val<uint32_t>(1);\n";
  os << "  src_dram_addr |= (src_dram_addr_hi << 32);\n";
  os << "  uint32_t num_tiles = get_arg_val<uint32_t>(2);\n";
  os << "\n";
  os << "  // CB configuration\n";
  os << "  constexpr uint32_t cb_id = " << cb_id << ";\n";
  os << "  constexpr uint32_t tile_size = " << tile_size_bytes << ";\n";
  os << "\n";
  os << "  // Read loop\n";
  os << "  for (uint32_t i = 0; i < num_tiles; i++) {\n";
  os << "    cb_reserve_back(cb_id, 1);\n";
  os << "    uint32_t write_ptr = get_write_ptr(cb_id);\n";
  os << "    uint64_t src_addr = src_dram_addr + i * tile_size;\n";
  os << "    noc_async_read(src_addr, write_ptr, tile_size);\n";
  os << "    noc_async_read_barrier();\n";
  os << "    cb_push_back(cb_id, 1);\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

std::string CodeGenBlackhole::GenerateWriterKernel(const std::string& func_name,
                                                   const std::string& dst_buf,
                                                   int cb_id, int num_tiles,
                                                   int tile_size_bytes) {
  std::ostringstream os;

  os << "void " << func_name << "() {\n";
  os << "  // Runtime arguments\n";
  os << "  uint64_t dst_dram_addr = get_arg_val<uint32_t>(0);\n";
  os << "  uint64_t dst_dram_addr_hi = get_arg_val<uint32_t>(1);\n";
  os << "  dst_dram_addr |= (dst_dram_addr_hi << 32);\n";
  os << "  uint32_t num_tiles = get_arg_val<uint32_t>(2);\n";
  os << "\n";
  os << "  // CB configuration\n";
  os << "  constexpr uint32_t cb_id = " << cb_id << ";\n";
  os << "  constexpr uint32_t tile_size = " << tile_size_bytes << ";\n";
  os << "\n";
  os << "  // Write loop\n";
  os << "  for (uint32_t i = 0; i < num_tiles; i++) {\n";
  os << "    cb_wait_front(cb_id, 1);\n";
  os << "    uint32_t read_ptr = get_read_ptr(cb_id);\n";
  os << "    uint64_t dst_addr = dst_dram_addr + i * tile_size;\n";
  os << "    noc_async_write(read_ptr, dst_addr, tile_size);\n";
  os << "    noc_async_write_barrier();\n";
  os << "    cb_pop_front(cb_id, 1);\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

}  // namespace tl
}  // namespace tvm
