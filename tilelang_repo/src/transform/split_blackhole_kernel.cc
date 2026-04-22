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
 * \file split_blackhole_kernel.cc
 * \brief Historical Phase-B normalization hook for Blackhole kernels.
 *
 * Task 4 moved reader/compute/writer owner truth into the TTProgram /
 * ExecutableSpec explicit kernel records. This pass intentionally stops
 * emitting cross-pass blackhole.segment_kind markers.
 */

#include "split_blackhole_kernel.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

tir::transform::Pass SplitBlackholeKernelPass() {
  auto fpass = [](tir::PrimFunc func, IRModule /*m*/,
                  tir::transform::PassContext /*ctx*/) -> tir::PrimFunc {
    return func;
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0,
                                            "tl.transform.SplitBlackholeKernel", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SplitBlackholeKernel", SplitBlackholeKernelPass);
}

}  // namespace tl
}  // namespace tvm
