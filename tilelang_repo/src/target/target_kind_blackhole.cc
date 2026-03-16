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
 * \file target_kind_blackhole.cc
 * \brief Register Blackhole (Tenstorrent) target kind.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target_kind.h>

namespace tvm {

/*!
 * \brief Blackhole target kind registration
 *
 * Registers Blackhole (Tenstorrent) as a target in TVM.
 * Uses kDLCPU as the underlying device type since Blackhole
 * is programmed as a co-processor via TT-Metal API.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Register Blackhole target kind with appropriate attributes
  // Using the TVM target registration mechanism
  // Note: This complements the existing target.build.tilelang_blackhole registration
  refl::GlobalDef()
      .def("target.build.blackhole", [](ffi::Module mod, ffi::Any target) -> ffi::Module {
        // Delegate to tilelang_blackhole build function
        // This is called when target "blackhole" is used
        auto build_fn = ffi::GetGlobalFunc("target.build.tilelang_blackhole");
        if (build_fn.has_value()) {
          return build_fn.value()(mod, target).operator ffi::Module();
        }
        LOG(FATAL) << "Blackhole target build function not available. "
                   << "Make sure TileLang is built with Blackhole support.";
        return ffi::Module();
      });
}

}  // namespace tvm
