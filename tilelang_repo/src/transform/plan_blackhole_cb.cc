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
 * \file plan_blackhole_cb.cc
 * \brief Plan Circular Buffer (CB) allocation for Blackhole backend
 */

#include "plan_blackhole_cb.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using tir::PrimFunc;
using tir::PrimFuncNode;
using tvm::DataType;

// Main entry point - simplified implementation
PrimFunc PlanBlackholeCB::Transform(const PrimFunc& func) {
  // For now, just return the function as-is
  // Full implementation will analyze alloc_shared and plan CB allocation
  return func;
}

// Validate CB allocation constraints
bool PlanBlackholeCB::Validate() const {
  // Simplified validation - always return true for now
  return true;
}

// Calculate page size for a tile
int PlanBlackholeCB::CalculatePageSize(int rows, int cols, int dtype_size) {
  return rows * cols * dtype_size;
}

// Modern TVM pass registration
tir::transform::Pass PlanBlackholeCBPass() {
  auto fpass = [](PrimFunc func, IRModule m, tir::transform::PassContext ctx) -> PrimFunc {
    return PlanBlackholeCB().Transform(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.PlanBlackholeCB", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.PlanBlackholeCB", PlanBlackholeCBPass);
}

}  // namespace tl
}  // namespace tvm
