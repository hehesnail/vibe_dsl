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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Blackhole device API
 *
 * Implements TVM DeviceAPI interface for TT-Metal/Blackhole backend.
 * Uses TT-Metal C++ API to interact with the device.
 */
class BlackholeDeviceAPI final : public DeviceAPI {
 public:
  // Device query methods
  void SetDevice(TVMContext ctx) final {
    // TODO: Implement device selection
    // For now, Blackhole only supports single device
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind,
               TVMValue* rv) final {
    switch (kind) {
      case kExist:
        // Blackhole device exists if TT-Metal is available
        rv->v_int = 1;
        break;
      case kMaxThreadsPerBlock:
        // Each core supports multiple threads
        rv->v_int = 1;
        break;
      case kWarpSize:
        // Not applicable for Blackhole
        rv->v_int = 1;
        break;
      case kMaxSharedMemoryPerBlock:
        // L1 memory per core: 1.5 MB
        rv->v_int = 1572864;
        break;
      case kComputeVersion:
        // Return Blackhole architecture version
        rv->v_str = const_cast<char*>("blackhole");
        break;
      case kDeviceName:
        rv->v_str = const_cast<char*>("blackhole");
        break;
      case kMaxClockRate:
        // Clock rate in kHz
        rv->v_int = 1000000;  // 1 GHz
        break;
      case kMultiProcessorCount:
        // Number of compute cores (Tensix)
        rv->v_int = 140;
        break;
      case kMaxThreadDimensions:
        rv->v_int = 3;
        break;
      case kMaxRegistersPerBlock:
        // Not applicable
        rv->v_int = 0;
        break;
      case kGcnArch:
        rv->v_str = const_cast<char*>("blackhole");
        break;
      case kApiVersion:
        rv->v_int = 0;
        break;
      default:
        break;
    }
  }

  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
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

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    // TODO: Implement memory deallocation using TT-Metal
    free(ptr);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    return AllocDataSpace(ctx, size, kL1Alignment, type_hint);
  }

  void FreeWorkspace(TVMContext ctx, void* ptr) final {
    FreeDataSpace(ctx, ptr);
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to,
                      size_t to_offset, size_t size, TVMContext ctx_from,
                      TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    // TODO: Implement data transfer using TT-Metal NOC
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset, size);
  }

  void StreamSynchronize(TVMContext ctx, TVMStreamHandle stream) final {
    // TODO: Implement stream synchronization
    // For synchronous execution, this is a no-op
  }

  void* CreateStream(TVMContext ctx) final {
    // Blackhole uses synchronous execution
    return nullptr;
  }

  void FreeStream(TVMContext ctx, TVMStreamHandle stream) final {
    // No-op for synchronous execution
  }

  void StreamWaitEvent(TVMContext ctx, TVMStreamHandle stream,
                       TVMEventHandle event) final {
    // TODO: Implement event synchronization
  }

  TVMEventHandle CreateEvent(TVMContext ctx) final {
    return nullptr;
  }

  void FreeEvent(TVMContext ctx, TVMEventHandle event) final {
    // No-op
  }

  void EventRecord(TVMEventHandle event, TVMStreamHandle stream) final {
    // TODO: Implement event recording
  }

  void EventSynchronize(TVMEventHandle event) final {
    // TODO: Implement event synchronization
  }

  bool EventQuery(TVMEventHandle event) final {
    // TODO: Implement event query
    return true;
  }

  void* GetNativeContext(TVMContext ctx) final {
    return nullptr;
  }

  void ReuseAllocator(TVMContext ctx, std::function<void*(size_t, size_t, DLDataType)> alloc,
                      std::function<void(void*)> free) final {
    // TODO: Implement custom allocator
  }

 private:
  // L1 memory alignment requirement
  static constexpr size_t kL1Alignment = 16;
};

/*!
 * \brief Blackhole module
 *
 * Wraps compiled TT-Metal kernels and provides TVM runtime interface.
 */
class BlackholeModuleNode final : public runtime::ModuleNode {
 public:
  BlackholeModuleNode(std::string data, std::string fmt,
                      std::unordered_map<std::string, FunctionInfo> fmap,
                      std::string source)
      : data_(std::move(data)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  const char* type_key() const final { return "blackhole"; }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // TODO: Implement kernel launching
    // For now, return empty function
    return PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      LOG(WARNING) << "Blackhole kernel execution not yet implemented";
    });
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    // Save the compiled kernel code
    std::string fmt = GetFormat(format, fmt_);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cc") {
      // Save C/C++ source code
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, source_);
    } else {
      // Save binary data
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(source_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return source_;
    if (fmt_ == "c" && format == "cc") return source_;
    return "";
  }

  // Initialize TT-Metal device
  void InitDevice() {
    // TODO: Initialize TT-Metal device
    // This should be called before any kernel execution
  }

 private:
  // Compiled kernel data
  std::string data_;
  // Format (c, cc, bin, etc.)
  std::string fmt_;
  // Function map
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // Source code
  std::string source_;
};

// Module creation function
Module BlackholeModuleCreate(std::string data, std::string fmt,
                             std::unordered_map<std::string, FunctionInfo> fmap,
                             std::string source) {
  auto n = make_object<BlackholeModuleNode>(std::move(data), std::move(fmt),
                                            std::move(fmap), std::move(source));
  return Module(n);
}

// Device API registration
TVM_REGISTER_GLOBAL("device_api.blackhole")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      DeviceAPI* ptr = BlackholeDeviceAPI::Global();
      *rv = static_cast<void*>(ptr);
    });

// Module creation registration
TVM_REGISTER_GLOBAL("runtime.module.loadfile_blackhole")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      // TODO: Implement module loading
    });

TVM_REGISTER_GLOBAL("runtime.module.loadfile_c")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      // For C source files that may contain Blackhole code
      // TODO: Implement C source loading
    });

// Blackhole device context
static DeviceAPI* BlackholeDeviceAPI::Global() {
  static BlackholeDeviceAPI* inst = new BlackholeDeviceAPI();
  return inst;
}

}  // namespace runtime
}  // namespace tvm
