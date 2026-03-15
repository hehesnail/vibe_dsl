/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file split_blackhole_kernel.h
 * \brief Split unified PrimFunc into Reader/Compute/Writer kernels
 */

#ifndef TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
#define TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Result of splitting a kernel into R/C/W components
 */
struct SplitResult {
  PrimFunc reader_func;   // Data movement: DRAM -> CB
  PrimFunc compute_func;  // Computation: CB -> CB (e.g., GEMM)
  PrimFunc writer_func;   // Data movement: CB -> DRAM

  bool HasReader() const { return reader_func.defined(); }
  bool HasCompute() const { return compute_func.defined(); }
  bool HasWriter() const { return writer_func.defined(); }
};

/*!
 * \brief SplitBlackholeKernel Pass
 *
 * This pass splits a unified TIR PrimFunc into three separate kernels
 * for Blackhole architecture:
 * 1. Reader Kernel: Handles data movement from DRAM to Circular Buffer
 * 2. Compute Kernel: Handles computation on data in CBs
 * 3. Writer Kernel: Handles data movement from CB to DRAM
 *
 * Each kernel runs on different RISC-V cores (BRISC/TRISC/NCRISC).
 */
class SplitBlackholeKernel : public StmtExprMutator {
 public:
  /*! \brief Main entry point */
  SplitResult Transform(const PrimFunc& func);

  /*! \brief Generate Reader kernel from original function */
  PrimFunc GenerateReaderKernel(const PrimFunc& func);

  /*! \brief Generate Compute kernel from original function */
  PrimFunc GenerateComputeKernel(const PrimFunc& func);

  /*! \brief Generate Writer kernel from original function */
  PrimFunc GenerateWriterKernel(const PrimFunc& func);

  /*! \brief Check if a function contains CB synchronization of given type */
  bool ContainsCBSync(const PrimFunc& func, const std::string& sync_type);

 private:
  /*! \brief Analyze data flow to identify read/compute/write regions */
  struct DataFlowAnalysis {
    std::vector<Buffer> input_buffers;
    std::vector<Buffer> output_buffers;
    std::vector<Buffer> intermediate_buffers;
    bool has_compute;
  };

  DataFlowAnalysis AnalyzeDataFlow(const PrimFunc& func);

  /*! \brief Insert CB synchronization primitives */
  Stmt InsertCBSync(const Stmt& stmt, const std::string& kernel_type);

  /*! \brief Extract statements related to specific buffer types */
  std::vector<Stmt> ExtractStatements(const Stmt& stmt,
                                       const std::vector<Buffer>& buffers);
};

/*!
 * \brief Create the SplitBlackholeKernel pass
 * \return The pass function
 */
tvm::tir::transform::Pass SplitBlackholeKernelPass();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
