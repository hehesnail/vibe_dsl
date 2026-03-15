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

#include <cstring>
#include <memory>
#include <cstring>
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
}  // namespace tvm
