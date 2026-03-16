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
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief Result of splitting a kernel into R/C/W components
 */
struct SplitResult {
  tvm::tir::PrimFunc reader_func;   // Data movement: DRAM -> CB
  tvm::tir::PrimFunc compute_func;  // Computation: CB -> CB (e.g., GEMM)
  tvm::tir::PrimFunc writer_func;   // Data movement: CB -> DRAM

  bool HasReader() const { return reader_func.defined(); }
  bool HasCompute() const { return compute_func.defined(); }
  bool HasWriter() const { return writer_func.defined(); }
};

/*!
 * \brief SplitBlackholeKernel Pass
 *
 * This pass splits a unified TIR PrimFunc into three separate kernels
 * for Blackhole architecture.
 */
class SplitBlackholeKernel : public tvm::tir::StmtExprMutator {
 public:
  /*! \brief Main entry point */
  SplitResult Transform(const tvm::tir::PrimFunc& func);

  /*! \brief Generate Reader kernel from original function */
  tvm::tir::PrimFunc GenerateReaderKernel(const tvm::tir::PrimFunc& func);

  /*! \brief Generate Compute kernel from original function */
  tvm::tir::PrimFunc GenerateComputeKernel(const tvm::tir::PrimFunc& func);

  /*! \brief Generate Writer kernel from original function */
  tvm::tir::PrimFunc GenerateWriterKernel(const tvm::tir::PrimFunc& func);
};

/*!
 * \brief Create the SplitBlackholeKernel pass
 * \return The pass function
 */
tir::transform::Pass SplitBlackholeKernelPass();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
