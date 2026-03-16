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
 * \file rt_mod_blackhole.cc
 * \brief Blackhole (TT-Metal) runtime module.
 */

#include <tvm/runtime/device_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ir/transform.h>
#include <tvm/target/codegen.h>

#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unistd.h>

#include "codegen_blackhole.h"
#include "blackhole_module.h"

namespace tvm {
namespace runtime {

using tvm::ffi::Map;
using tvm::ffi::String;

/*!
 * \brief Blackhole device API
 *
 * Implements TVM DeviceAPI interface for TT-Metal/Blackhole backend.
 * Uses TT-Metal C++ API to interact with the device.
 */
class BlackholeDeviceAPI final : public DeviceAPI {
 public:
  // Device query methods
  void SetDevice(Device dev) final {
    // TODO: Implement device selection
    // For now, Blackhole only supports single device
  }

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    switch (kind) {
      case kExist:
        // Blackhole device exists if TT-Metal is available
        rv->operator=(true);
        break;
      case kMaxThreadsPerBlock:
        // Each core supports multiple threads
        rv->operator=(1);
        break;
      case kWarpSize:
        // Not applicable for Blackhole
        rv->operator=(1);
        break;
      case kMaxSharedMemoryPerBlock:
        // L1 memory per core: 1.5 MB
        rv->operator=(1572864);
        break;
      case kComputeVersion:
        // Return Blackhole architecture version
        rv->operator=("blackhole");
        break;
      case kDeviceName:
        rv->operator=("blackhole");
        break;
      case kMaxClockRate:
        // Clock rate in kHz
        rv->operator=(1000000);  // 1 GHz
        break;
      case kMultiProcessorCount:
        // Number of compute cores (Tensix)
        rv->operator=(140);
        break;
      case kMaxThreadDimensions:
        rv->operator=(3);
        break;
      case kMaxRegistersPerBlock:
        // Not applicable
        rv->operator=(0);
        break;
      case kGcnArch:
        rv->operator=("blackhole");
        break;
      case kApiVersion:
        rv->operator=(0);
        break;
      default:
        break;
    }
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    // TODO: Implement memory allocation using TT-Metal
    // For now, use standard allocation
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    // TODO: Implement memory deallocation using TT-Metal
    free(ptr);
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return AllocDataSpace(dev, size, kL1Alignment, type_hint);
  }

  void FreeWorkspace(Device dev, void* ptr) final {
    FreeDataSpace(dev, ptr);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    // TODO: Implement stream synchronization
    // For synchronous execution, this is a no-op
  }

  TVMStreamHandle CreateStream(Device dev) final {
    // Blackhole uses synchronous execution
    return nullptr;
  }

  void FreeStream(Device dev, TVMStreamHandle stream) final {
    // No-op for synchronous execution
  }

  // Global singleton accessor
  static BlackholeDeviceAPI* Global();

 private:
  // L1 memory alignment requirement
  static constexpr size_t kL1Alignment = 16;
};

BlackholeDeviceAPI* BlackholeDeviceAPI::Global() {
  static BlackholeDeviceAPI* inst = new BlackholeDeviceAPI();
  return inst;
}

// Device API registration
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("device_api.blackhole", []() -> void* {
    return static_cast<void*>(BlackholeDeviceAPI::Global());
  });
}

}  // namespace runtime

namespace codegen {

using namespace tvm::runtime;

// Register Blackhole target kind
TVM_REGISTER_TARGET_KIND("blackhole", kDLExtDev)
    .add_attr_option<int64_t>("max_shared_memory_per_block", 1572864)  // 1.5 MB L1
    .add_attr_option<int64_t>("num_cores", 140)  // 14x10 Tensix cores
    .add_attr_option<int64_t>("num_cbs", 64)     // 64 circular buffers per core
    .set_default_keys({"blackhole"});

/*!
 * \brief Extract CB configuration from PrimFunc attrs
 * \param f The PrimFunc
 * \return Vector of CB configurations
 */
static std::vector<CBConfig> ExtractCBConfig(const tir::PrimFunc& f) {
  std::vector<CBConfig> cb_configs;

  auto cb_attr = f->GetAttr<ffi::Map<ffi::String, ffi::ObjectRef>>("tl.blackhole_cb_config");
  if (!cb_attr) {
    // Use default CB config for simple kernels
    cb_configs.push_back({0, 1, 2048, "float16"});  // cb_id=0, 1 page, 2KB page size
    return cb_configs;
  }

  for (const auto& kv : cb_attr.value()) {
    CBConfig config;
    config.cb_id = std::stoi(static_cast<std::string>(kv.first));
    auto cb_info = Downcast<ffi::Map<ffi::String, ffi::ObjectRef>>(kv.second);

    if (auto num_pages = cb_info.Get("num_pages")) {
      config.num_pages = Downcast<Integer>(num_pages.value()).IntValue();
    }
    if (auto page_size = cb_info.Get("page_size")) {
      config.page_size = Downcast<Integer>(page_size.value()).IntValue();
    }
    if (auto data_format = cb_info.Get("data_format")) {
      // The value is a TIR StringImm node, not ffi::String
      config.data_format = Downcast<StringImm>(data_format.value())->value;
    }

    cb_configs.push_back(config);
  }

  return cb_configs;
}

/*!
 * \brief Extract function information for Blackhole backend
 * \param mod The IR module
 * \return Map from function name to BlackholeFunctionInfo
 */
static std::unordered_map<std::string, BlackholeFunctionInfo> ExtractBlackholeFuncInfo(
    const IRModule& mod) {
  std::unordered_map<std::string, BlackholeFunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    BlackholeFunctionInfo info;

    // Extract argument types and buffer flags
    for (size_t i = 0; i < f->params.size(); ++i) {
      DLDataType dtype = f->params[i]->dtype;
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.code == kDLBool) {
        dtype.code = kDLInt;
        dtype.bits = 32;
      }
      info.arg_types.push_back(dtype);

      // Check if this is a buffer argument (handle type)
      info.is_buffer_arg.push_back(dtype.code == kDLOpaqueHandle);
    }

    // Extract CB configuration
    info.cb_configs = ExtractCBConfig(f);

    // Check for multi-kernel configuration (R/C/W split)
    auto kernel_split = f->GetAttr<ffi::Map<ffi::String, ffi::ObjectRef>>("tl.blackhole_kernel_split");
    if (kernel_split) {
      auto reader_opt = kernel_split.value().Get("reader");
      auto compute_opt = kernel_split.value().Get("compute");
      auto writer_opt = kernel_split.value().Get("writer");
      info.has_reader = reader_opt.has_value();
      info.has_compute = compute_opt.has_value();
      info.has_writer = writer_opt.has_value();
    } else {
      // Single kernel mode
      info.has_writer = true;
    }

    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    if (global_symbol) {
      fmap[static_cast<std::string>(global_symbol.value())] = info;
    }
  }
  return fmap;
}

/*!
 * \brief Build function for Blackhole target
 * \param mod The IR module containing PrimFuncs
 * \param target The target device
 * \return A Blackhole runtime module
 *
 * This function generates TT-Metal C++ code from TIR using CodeGenBlackhole
 * and creates a BlackholeModule that can execute the kernels.
 */
ffi::Module BuildTileLangBlackhole(IRModule mod, Target target) {
  LOG(INFO) << "BuildTileLangBlackhole: Generating TT-Metal code for Blackhole target";

  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  devices.insert("blackhole");

  tl::CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // Create temporary directory for kernel files
  std::string kernel_dir = "/tmp/tilelang_blackhole/" + std::to_string(getpid());
  std::filesystem::create_directories(kernel_dir);

  // Process all functions in the module
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);

    // Check calling convention
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (calling_conv == CallingConv::kDeviceKernelLaunch) {
      // Device kernel - generate TT-Metal kernel code
      cg.AddFunction(gvar, f);
    } else {
      // Host function - also add it
      cg.AddFunction(gvar, f);
    }
  }

  std::string code = cg.Finish();

  // Extract function info for BlackholeModule
  auto func_info_map = ExtractBlackholeFuncInfo(mod);

  // Store kernel code in function info
  for (auto& kv : func_info_map) {
    kv.second.kernel_code = code;
  }

  LOG(INFO) << "BuildTileLangBlackhole: Generated " << code.size()
            << " bytes of TT-Metal code for " << func_info_map.size() << " functions";

  // Create BlackholeModule
  return BlackholeModuleCreate(std::move(func_info_map), kernel_dir);
}

/*!
 * \brief Build function for Blackhole target without host code
 * \param mod The IR module containing PrimFuncs
 * \param target The target device
 * \return A Blackhole runtime module with only device code
 *
 * This generates only the device kernels without host wrapper code.
 */
ffi::Module BuildTileLangBlackholeWithoutHost(IRModule mod, Target target) {
  LOG(INFO) << "BuildTileLangBlackholeWithoutHost: Generating device code only";

  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = false;

  std::unordered_set<std::string> devices;
  devices.insert("blackhole");

  tl::CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // Create temporary directory for kernel files
  std::string kernel_dir = "/tmp/tilelang_blackhole/" + std::to_string(getpid());
  std::filesystem::create_directories(kernel_dir);

  // Process only device kernel functions
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);

    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (calling_conv == CallingConv::kDeviceKernelLaunch) {
      cg.AddFunction(gvar, f);
    }
  }

  std::string code = cg.Finish();

  // Extract function info for BlackholeModule
  auto func_info_map = ExtractBlackholeFuncInfo(mod);

  // Store kernel code in function info
  for (auto& kv : func_info_map) {
    kv.second.kernel_code = code;
  }

  return BlackholeModuleCreate(std::move(func_info_map), kernel_dir);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_blackhole", BuildTileLangBlackhole)
      .def("target.build.tilelang_blackhole_without_host", BuildTileLangBlackholeWithoutHost);
}

} // namespace codegen
} // namespace tvm
