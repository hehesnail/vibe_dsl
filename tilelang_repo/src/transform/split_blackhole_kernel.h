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
 * \brief SplitBlackholeKernels pass: annotate statements with segment kind
 *        and emit blackhole.segment_plan for 3-kernel GEMM.
 */

#ifndef TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
#define TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief Create the SplitBlackholeKernels pass.
 *
 * Scans each device PrimFunc for compute ops (tl.tileop.gemm_py).
 * If found, wraps each top-level statement with:
 *   AttrStmt("blackhole.segment_kind", "reader"|"compute"|"writer", stmt)
 * and writes blackhole.segment_plan (3-kernel schema) to the function attrs.
 *
 * Pure-copy functions (no compute op) are left unchanged; they continue
 * through the existing fused_dataflow single-kernel path.
 */
tir::transform::Pass SplitBlackholeKernelPass();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
