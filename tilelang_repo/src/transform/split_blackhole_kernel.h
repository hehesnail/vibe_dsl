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
 * \brief Historical Blackhole Phase-B normalization hook.
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
 * Task 4 removed cross-pass segment-kind markers. The pass name
 * remains as a stable phase hook, but segment truth is now constructed inside
 * TT planning rather than emitted onto the TIR body here.
 */
tir::transform::Pass SplitBlackholeKernelPass();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_SPLIT_BLACKHOLE_KERNEL_H_
