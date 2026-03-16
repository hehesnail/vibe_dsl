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
 * \file test_codegen_blackhole_gemm.cc
 * \brief Test Blackhole CodeGen for GEMM operations.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <string>

#include "tilelang_repo/src/target/codegen_blackhole.h"
#include "tilelang_repo/src/tir/builtin_blackhole.h"

namespace tvm {
namespace tl {
namespace testing {

using namespace tvm::tir;
using namespace tvm::builtin;

/*!
 * \brief Create a simple GEMM TIR function for testing.
 *
 * This creates a compute kernel that performs C = A @ B using
 * TT-Metal matmul_tiles API.
 *
 * \return A PrimFunc representing the GEMM compute kernel.
 */
PrimFunc CreateGemmComputeFunc() {
  // Define buffer shapes and types
  DataType fp16 = DataType::Float(16);
  DataType fp32 = DataType::Float(32);
  DataType int32 = DataType::Int(32);

  // Create buffer variables (simulate CBs)
  Var buf_a("buf_a", PointerType(PrimType(fp16), "global"));
  Var buf_b("buf_b", PointerType(PrimType(fp16), "global"));
  Var buf_c("buf_c", PointerType(PrimType(fp32), "global"));

  // Create buffers
  Buffer A = BufferDecl(buf_a, {32, 32}, fp16, {}, "A");
  Buffer B = BufferDecl(buf_b, {32, 32}, fp16, {}, "B");
  Buffer C = BufferDecl(buf_c, {32, 32}, fp32, {}, "C");

  // Build the compute sequence:
  // mm_init(0, 1, 16)
  // tile_regs_acquire()
  // cb_wait_front(0, 1)
  // cb_wait_front(1, 1)
  // matmul_tiles(0, 1, 0, 0, 0)
  // cb_pop_front(0, 1)
  // cb_pop_front(1, 1)
  // tile_regs_commit()
  // tile_regs_wait()
  // cb_reserve_back(16, 1)
  // pack_tile(0, 16)
  // cb_push_back(16, 1)
  // tile_regs_release()

  std::vector<Stmt> stmts;
  IntImm cb_0(int32, 0);
  IntImm cb_1(int32, 1);
  IntImm cb_out(int32, 16);
  IntImm tile_idx(int32, 0);
  IntImm num_tiles(int32, 1);

  // 1. mm_init(0, 1, 16)
  stmts.push_back(Evaluate(Call(int32, blackhole_mm_init(),
                                {cb_0, cb_1, cb_out})));

  // 2. tile_regs_acquire()
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_acquire(), {})));

  // 3. cb_wait_front(0, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_wait_front(),
                                {cb_0, num_tiles})));

  // 4. cb_wait_front(1, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_wait_front(),
                                {cb_1, num_tiles})));

  // 5. matmul_tiles(0, 1, 0, 0, 0)
  stmts.push_back(Evaluate(Call(int32, blackhole_matmul_tiles(),
                                {cb_0, cb_1, tile_idx, tile_idx, tile_idx})));

  // 6. cb_pop_front(0, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_pop_front(),
                                {cb_0, num_tiles})));

  // 7. cb_pop_front(1, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_pop_front(),
                                {cb_1, num_tiles})));

  // 8. tile_regs_commit()
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_commit(), {})));

  // 9. tile_regs_wait()
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_wait(), {})));

  // 10. cb_reserve_back(16, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_reserve_back(),
                                {cb_out, num_tiles})));

  // 11. pack_tile(0, 16)
  stmts.push_back(Evaluate(Call(int32, blackhole_pack_tile(),
                                {tile_idx, cb_out})));

  // 12. cb_push_back(16, 1)
  stmts.push_back(Evaluate(Call(int32, blackhole_cb_push_back(),
                                {cb_out, num_tiles})));

  // 13. tile_regs_release()
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_release(), {})));

  // Create the function body
  Stmt body = SeqStmt::Flatten(stmts);

  // Create the PrimFunc with global symbol attribute
  PrimFunc func = PrimFunc({buf_a, buf_b, buf_c}, body, VoidType(), {});
  func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol,
                  String("gemm_compute_kernel"));
  func = WithAttr(std::move(func), tvm::attr::kCallingConv,
                  Integer(CallingConv::kDeviceKernelLaunch));

  return func;
}

/*!
 * \brief Test that CodeGenBlackhole generates correct TT-Metal code for GEMM.
 */
TEST(CodeGenBlackholeGEMM, BasicMatmulTiles) {
  // Create a GEMM compute function
  PrimFunc func = CreateGemmComputeFunc();

  // Create CodeGenBlackhole instance
  CodeGenBlackhole cg;
  std::unordered_set<std::string> devices;
  devices.insert("blackhole");
  cg.Init(false, false, true, "blackhole", devices);

  // Generate code
  GlobalVar gvar("gemm_compute_kernel");
  cg.AddFunction(gvar, func);
  std::string code = cg.Finish();

  // Verify the generated code contains expected TT-Metal API calls
  std::cout << "=== Generated TT-Metal Code ===" << std::endl;
  std::cout << code << std::endl;
  std::cout << "=============================" << std::endl;

  // Check for required includes
  EXPECT_NE(code.find("#include"), std::string::npos)
      << "Generated code should have includes";

  // Check for TT-Metal compute API calls
  EXPECT_NE(code.find("mm_init"), std::string::npos)
      << "Generated code should call mm_init";
  EXPECT_NE(code.find("tile_regs_acquire"), std::string::npos)
      << "Generated code should call tile_regs_acquire";
  EXPECT_NE(code.find("tile_regs_commit"), std::string::npos)
      << "Generated code should call tile_regs_commit";
  EXPECT_NE(code.find("tile_regs_wait"), std::string::npos)
      << "Generated code should call tile_regs_wait";
  EXPECT_NE(code.find("tile_regs_release"), std::string::npos)
      << "Generated code should call tile_regs_release";
  EXPECT_NE(code.find("pack_tile"), std::string::npos)
      << "Generated code should call pack_tile";

  // Check for matmul_tiles call
  EXPECT_NE(code.find("matmul_tiles"), std::string::npos)
      << "Generated code should call matmul_tiles";

  // Check for CB operations
  EXPECT_NE(code.find("cb_wait_front"), std::string::npos)
      << "Generated code should call cb_wait_front";
  EXPECT_NE(code.find("cb_pop_front"), std::string::npos)
      << "Generated code should call cb_pop_front";
  EXPECT_NE(code.find("cb_reserve_back"), std::string::npos)
      << "Generated code should call cb_reserve_back";
  EXPECT_NE(code.find("cb_push_back"), std::string::npos)
      << "Generated code should call cb_push_back";
}

/*!
 * \brief Test that CodeGenBlackhole handles multi-tile accumulation.
 */
TEST(CodeGenBlackholeGEMM, MultiTileAccumulate) {
  // For now, just verify the basic structure works
  // Full multi-tile test would require loop structures

  DataType int32 = DataType::Int(32);

  // Create a loop over K tiles
  Var kt("kt", int32);
  IntImm zero(int32, 0);
  IntImm one(int32, 1);
  IntImm four(int32, 4);  // 4 K tiles
  IntImm cb_0(int32, 0);
  IntImm cb_1(int32, 1);

  // Build loop body with matmul_tiles
  std::vector<Stmt> loop_body;
  loop_body.push_back(Evaluate(Call(int32, blackhole_cb_wait_front(),
                                    {cb_0, one})));
  loop_body.push_back(Evaluate(Call(int32, blackhole_cb_wait_front(),
                                    {cb_1, one})));
  loop_body.push_back(Evaluate(Call(int32, blackhole_matmul_tiles(),
                                    {cb_0, cb_1, zero, zero, zero})));
  loop_body.push_back(Evaluate(Call(int32, blackhole_cb_pop_front(),
                                    {cb_0, one})));
  loop_body.push_back(Evaluate(Call(int32, blackhole_cb_pop_front(),
                                    {cb_1, one})));

  // Create for loop: for (kt = 0; kt < 4; kt++)
  Stmt body = For(kt, zero, four, ForKind::kSerial,
                  SeqStmt::Flatten(loop_body));

  // Add init and cleanup
  std::vector<Stmt> stmts;
  stmts.push_back(Evaluate(Call(int32, blackhole_mm_init(),
                                {cb_0, cb_1, IntImm(int32, 16)})));
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_acquire(), {})));
  stmts.push_back(body);
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_commit(), {})));
  stmts.push_back(Evaluate(Call(int32, blackhole_tile_regs_wait(), {})));

  PrimFunc func = PrimFunc({}, SeqStmt::Flatten(stmts), VoidType(), {});
  func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol,
                  String("gemm_accumulate_kernel"));

  // Generate code
  CodeGenBlackhole cg;
  std::unordered_set<std::string> devices;
  devices.insert("blackhole");
  cg.Init(false, false, true, "blackhole", devices);

  GlobalVar gvar("gemm_accumulate_kernel");
  cg.AddFunction(gvar, func);
  std::string code = cg.Finish();

  std::cout << "=== Multi-Tile Accumulation Code ===" << std::endl;
  std::cout << code << std::endl;
  std::cout << "===================================" << std::endl;

  // Verify the loop structure is present
  EXPECT_NE(code.find("for"), std::string::npos)
      << "Generated code should have a for loop for K tiles";
}

/*!
 * \brief Test that generated code has proper kernel_main structure.
 */
TEST(CodeGenBlackholeGEMM, KernelMainStructure) {
  PrimFunc func = CreateGemmComputeFunc();

  CodeGenBlackhole cg;
  std::unordered_set<std::string> devices;
  devices.insert("blackhole");
  cg.Init(false, false, true, "blackhole", devices);

  GlobalVar gvar("kernel_main");
  cg.AddFunction(gvar, func);
  std::string code = cg.Finish();

  // Check for function structure
  EXPECT_NE(code.find("void"), std::string::npos)
      << "Generated code should have void return type";
  EXPECT_NE(code.find("kernel_main"), std::string::npos)
      << "Generated code should have kernel_main function";
}

}  // namespace testing
}  // namespace tl
}  // namespace tvm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
