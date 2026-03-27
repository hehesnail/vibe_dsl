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
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <unistd.h>

#include "codegen_blackhole.h"
#include "blackhole_module.h"
#include "../tir/builtin_blackhole.h"

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
 * \brief Extract CB configuration from PrimFunc attrs.
 * \param f The PrimFunc
 * \return Vector of CB configurations
 */
static std::vector<CBConfig> ExtractCBConfig(const tir::PrimFunc& f) {
  std::vector<CBConfig> cb_configs;

  auto cb_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.cb_configs");
  if (!cb_attr) {
    // Use default CB config for simple kernels
    cb_configs.push_back({0, "default_cb", "intermediate", 1, 2048, "Float16_b"});
    return cb_configs;
  }

  for (const auto& item : cb_attr.value()) {
    CBConfig config;
    auto cb_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (cb_info.empty()) continue;

    if (auto cb_id = cb_info.Get("cb_id")) {
      config.cb_id = Downcast<Integer>(cb_id.value()).IntValue();
    }
    if (auto name = cb_info.Get("name")) {
      config.name = Downcast<String>(name.value());
    }
    if (auto role = cb_info.Get("role")) {
      config.role = Downcast<String>(role.value());
    }
    if (auto num_pages = cb_info.Get("num_pages")) {
      config.num_pages = Downcast<Integer>(num_pages.value()).IntValue();
    }
    if (auto page_size = cb_info.Get("page_size")) {
      config.page_size_bytes = Downcast<Integer>(page_size.value()).IntValue();
    }
    if (auto data_format = cb_info.Get("data_format")) {
      config.data_format = Downcast<String>(data_format.value());
    }

    if (config.name.empty()) {
      config.name = "cb_" + std::to_string(config.cb_id);
    }
    if (config.role.empty()) {
      config.role = "intermediate";
    }
    if (config.data_format.empty()) {
      config.data_format = "Float16_b";
    }

    cb_configs.push_back(config);
  }

  return cb_configs;
}

static CorePlan ExtractCorePlan(const tir::PrimFunc& f) {
  CorePlan plan;
  auto core_plan_attr = f->GetAttr<ffi::Map<ffi::String, ffi::Any>>("blackhole.core_plan");
  if (!core_plan_attr) {
    return plan;
  }

  const auto& core_plan = core_plan_attr.value();
  if (auto v = core_plan.Get("logical_grid_x")) {
    plan.logical_grid_x = Downcast<Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_x")) {
    plan.logical_grid_x = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("logical_grid_y")) {
    plan.logical_grid_y = Downcast<Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_y")) {
    plan.logical_grid_y = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("linearization")) {
    plan.linearization = Downcast<String>(v.value());
  }

  if (auto v = core_plan.Get("physical_cores")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto core_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (core_info.empty()) {
        continue;
      }
      PhysicalCore core;
      if (auto x = core_info.Get("core_x")) {
        core.core_x = Downcast<Integer>(x.value()).IntValue();
      }
      if (auto y = core_info.Get("core_y")) {
        core.core_y = Downcast<Integer>(y.value()).IntValue();
      }
      plan.physical_cores.push_back(core);
    }
  }

  if (auto v = core_plan.Get("work_packets")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto packet_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (packet_info.empty()) {
        continue;
      }
      WorkPacket packet;
      if (auto x = packet_info.Get("core_x")) {
        packet.core_x = Downcast<Integer>(x.value()).IntValue();
      }
      if (auto y = packet_info.Get("core_y")) {
        packet.core_y = Downcast<Integer>(y.value()).IntValue();
      }
      if (auto offset = packet_info.Get("work_offset")) {
        packet.work_offset = Downcast<Integer>(offset.value()).IntValue();
      }
      if (auto count = packet_info.Get("work_count")) {
        packet.work_count = Downcast<Integer>(count.value()).IntValue();
      }
      plan.work_packets.push_back(packet);
    }
  }

  if (plan.physical_cores.empty()) {
    plan.physical_cores.push_back(PhysicalCore{});
  }
  if (plan.work_packets.empty()) {
    plan.work_packets.push_back(WorkPacket{
        0,
        0,
        0,
        std::max<uint32_t>(1, plan.logical_grid_x * plan.logical_grid_y),
    });
  }
  return plan;
}

static GemmContractSpec ExtractGemmContract(const tir::PrimFunc& f) {
  GemmContractSpec contract;
  auto gemm_attr = f->GetAttr<ffi::Map<ffi::String, ffi::Any>>("blackhole.gemm_contract");
  if (!gemm_attr) {
    return contract;
  }

  const auto& attrs = gemm_attr.value();
  if (auto v = attrs.Get("a_buffer")) {
    contract.a_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_buffer")) {
    contract.b_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_buffer")) {
    contract.c_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("M")) {
    contract.M = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("N")) {
    contract.N = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("K")) {
    contract.K = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("transpose_A")) {
    contract.transpose_A = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("transpose_B")) {
    contract.transpose_B = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("a_tensor_dtype")) {
    contract.a_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_tensor_dtype")) {
    contract.b_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_tensor_dtype")) {
    contract.c_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("a_cb_dtype")) {
    contract.a_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_cb_dtype")) {
    contract.b_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_cb_dtype")) {
    contract.c_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("accumulator_dtype")) {
    contract.accumulator_dtype = Downcast<String>(v.value());
  }

  contract.enabled = !contract.a_buffer.empty() && !contract.b_buffer.empty() &&
                     !contract.c_buffer.empty() && contract.M > 0 && contract.N > 0 &&
                     contract.K > 0;
  return contract;
}

static ComputeContractSpec ComputeContractFromLegacyGemm(const GemmContractSpec& gemm) {
  ComputeContractSpec contract;
  if (!gemm.enabled) {
    return contract;
  }

  constexpr uint32_t kBlackholeTileRows = 32;
  constexpr uint32_t kBlackholeTileCols = 32;
  contract.enabled = true;
  contract.kind = "gemm";
  contract.a_buffer = gemm.a_buffer;
  contract.b_buffer = gemm.b_buffer;
  contract.c_buffer = gemm.c_buffer;
  contract.M = gemm.M;
  contract.N = gemm.N;
  contract.K = gemm.K;
  contract.Mt = gemm.M / kBlackholeTileRows;
  contract.Nt = gemm.N / kBlackholeTileCols;
  contract.Kt = gemm.K / kBlackholeTileCols;
  contract.block_m_tiles = contract.Mt;
  contract.block_n_tiles = contract.Nt;
  contract.block_k_tiles = contract.Kt;
  contract.subblock_m_tiles = contract.Mt;
  contract.subblock_n_tiles = contract.Nt;
  contract.transpose_A = gemm.transpose_A;
  contract.transpose_B = gemm.transpose_B;
  contract.a_tensor_dtype = gemm.a_tensor_dtype;
  contract.b_tensor_dtype = gemm.b_tensor_dtype;
  contract.c_tensor_dtype = gemm.c_tensor_dtype;
  contract.a_cb_dtype = gemm.a_cb_dtype;
  contract.b_cb_dtype = gemm.b_cb_dtype;
  contract.c_cb_dtype = gemm.c_cb_dtype;
  contract.accumulator_dtype = gemm.accumulator_dtype;
  contract.math_fidelity = "HiFi4";
  contract.fp32_dest_acc_en = true;
  contract.math_approx_mode = false;
  contract.clear_accum = false;
  contract.k_pack = 1;
  contract.wg_wait = 0;
  contract.policy_type = 0;
  contract.policy_name = "Square";
  contract.has_mbarrier = false;
  return contract;
}

static ComputeContractSpec ExtractComputeContract(const tir::PrimFunc& f,
                                                  const GemmContractSpec& gemm_contract) {
  ComputeContractSpec contract;
  auto compute_attr = f->GetAttr<ffi::Map<ffi::String, ffi::Any>>("blackhole.compute_contract");
  if (!compute_attr) {
    return ComputeContractFromLegacyGemm(gemm_contract);
  }

  const auto& attrs = compute_attr.value();
  if (auto v = attrs.Get("enabled")) {
    contract.enabled = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("kind")) {
    contract.kind = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("a_buffer")) {
    contract.a_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_buffer")) {
    contract.b_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_buffer")) {
    contract.c_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("M")) {
    contract.M = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("N")) {
    contract.N = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("K")) {
    contract.K = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("Mt")) {
    contract.Mt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("Nt")) {
    contract.Nt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("Kt")) {
    contract.Kt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("block_m_tiles")) {
    contract.block_m_tiles = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("block_n_tiles")) {
    contract.block_n_tiles = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("block_k_tiles")) {
    contract.block_k_tiles = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("subblock_m_tiles")) {
    contract.subblock_m_tiles = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("subblock_n_tiles")) {
    contract.subblock_n_tiles = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("transpose_A")) {
    contract.transpose_A = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("transpose_B")) {
    contract.transpose_B = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("a_tensor_dtype")) {
    contract.a_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_tensor_dtype")) {
    contract.b_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_tensor_dtype")) {
    contract.c_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("a_cb_dtype")) {
    contract.a_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("b_cb_dtype")) {
    contract.b_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_cb_dtype")) {
    contract.c_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("accumulator_dtype")) {
    contract.accumulator_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("math_fidelity")) {
    contract.math_fidelity = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("fp32_dest_acc_en")) {
    contract.fp32_dest_acc_en = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("math_approx_mode")) {
    contract.math_approx_mode = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("clear_accum")) {
    contract.clear_accum = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("k_pack")) {
    contract.k_pack = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("wg_wait")) {
    contract.wg_wait = static_cast<int32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("policy_type")) {
    contract.policy_type = static_cast<int32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = attrs.Get("policy_name")) {
    contract.policy_name = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("has_mbarrier")) {
    contract.has_mbarrier = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("mbarrier_buffer")) {
    contract.mbarrier_buffer = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("mbarrier_scope")) {
    contract.mbarrier_scope = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("mbarrier_index_exprs")) {
    for (const auto& expr : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      contract.mbarrier_index_exprs.push_back(Downcast<String>(expr));
    }
  }
  if (auto v = attrs.Get("unpack_to_dest_mode")) {
    for (const auto& mode : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      contract.unpack_to_dest_mode.push_back(Downcast<String>(mode));
    }
  }

  contract.enabled = contract.enabled || (!contract.kind.empty() && contract.kind == "gemm" &&
                                          !contract.a_buffer.empty() &&
                                          !contract.b_buffer.empty() &&
                                          !contract.c_buffer.empty() && contract.M > 0 &&
                                          contract.N > 0 && contract.K > 0);
  if (contract.enabled && contract.kind == "gemm") {
    if (contract.Mt == 0 && contract.M > 0) contract.Mt = contract.M / 32;
    if (contract.Nt == 0 && contract.N > 0) contract.Nt = contract.N / 32;
    if (contract.Kt == 0 && contract.K > 0) contract.Kt = contract.K / 32;
    if (contract.block_m_tiles == 0) contract.block_m_tiles = contract.Mt;
    if (contract.block_n_tiles == 0) contract.block_n_tiles = contract.Nt;
    if (contract.block_k_tiles == 0) contract.block_k_tiles = contract.Kt;
    if (contract.subblock_m_tiles == 0) contract.subblock_m_tiles = contract.Mt;
    if (contract.subblock_n_tiles == 0) contract.subblock_n_tiles = contract.Nt;
    if (contract.math_fidelity.empty()) contract.math_fidelity = "HiFi4";
  }
  return contract;
}

static bool ExtractComputeConfig(const ffi::Map<ffi::String, ffi::Any>& spec_info,
                                 KernelComputeConfigSpec* compute_config) {
  if (spec_info.empty()) {
    return false;
  }
  if (auto v = spec_info.Get("math_fidelity")) {
    compute_config->math_fidelity = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("fp32_dest_acc_en")) {
    compute_config->fp32_dest_acc_en = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("math_approx_mode")) {
    compute_config->math_approx_mode = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("clear_accum")) {
    compute_config->clear_accum = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("k_pack")) {
    compute_config->k_pack = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("wg_wait")) {
    compute_config->wg_wait = static_cast<int32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("policy_type")) {
    compute_config->policy_type = static_cast<int32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("policy_name")) {
    compute_config->policy_name = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("unpack_to_dest_mode")) {
    for (const auto& mode : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      compute_config->unpack_to_dest_mode.push_back(Downcast<String>(mode));
    }
  }
  return !compute_config->math_fidelity.empty() || compute_config->fp32_dest_acc_en ||
         compute_config->math_approx_mode || compute_config->clear_accum ||
         compute_config->k_pack != 1 || compute_config->wg_wait != 0 ||
         !compute_config->unpack_to_dest_mode.empty();
}

static std::vector<KernelArgSpec> MakeDefaultCopyRuntimeArgs() {
  return {
      {"input0", "input_buffer_addr32", "uint32", ""},
      {"output0", "output_buffer_addr32", "uint32", ""},
      {"work_linear_id", "work_linear_id", "uint32", ""},
      {"a_tile_start_id", "a_tile_start_id", "uint32", ""},
      {"a_tile_num_tiles", "a_tile_num_tiles", "uint32", ""},
      {"a_tile_stride", "a_tile_stride", "uint32", ""},
      {"output_tile_start_id", "output_tile_start_id", "uint32", ""},
      {"output_tile_num_tiles", "output_tile_num_tiles", "uint32", ""},
      {"output_tile_stride", "output_tile_stride", "uint32", ""},
  };
}

static bool HasCopyBuiltins(const tir::PrimFunc& f) {
  bool found = false;
  tir::PostOrderVisit(f->body, [&](const ObjectRef& node) {
    if (found) {
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call) {
      return;
    }
    found = call->op.same_as(tir::builtin::blackhole_read_tile_to_cb()) ||
            call->op.same_as(tir::builtin::blackhole_write_tile_from_cb());
  });
  return found;
}

static std::vector<KernelArgSpec> ExtractRuntimeArgsFromArray(const ffi::Array<ffi::Any>& items) {
  std::vector<KernelArgSpec> runtime_args;
  for (const auto& item : items) {
    auto arg_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (arg_info.empty()) continue;

    KernelArgSpec arg;
    if (auto v = arg_info.Get("name")) {
      arg.name = Downcast<String>(v.value());
    }
    if (auto v = arg_info.Get("kind")) {
      arg.kind = Downcast<String>(v.value());
    }
    if (auto v = arg_info.Get("dtype")) {
      arg.dtype = Downcast<String>(v.value());
    }
    if (auto v = arg_info.Get("buffer")) {
      arg.buffer = Downcast<String>(v.value());
    }
    if (!arg.kind.empty()) {
      runtime_args.push_back(std::move(arg));
    }
  }
  return runtime_args;
}

static std::vector<AccessorSpec> ExtractAccessorsFromArray(const ffi::Array<ffi::Any>& items) {
  std::vector<AccessorSpec> accessors;
  for (const auto& item : items) {
    auto accessor_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (accessor_info.empty()) continue;

    AccessorSpec accessor;
    if (auto v = accessor_info.Get("buffer")) {
      accessor.buffer = Downcast<String>(v.value());
    }
    if (auto v = accessor_info.Get("slot")) {
      accessor.slot = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("compile_time_arg_offset")) {
      accessor.compile_time_arg_offset =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("compile_time_arg_count")) {
      accessor.compile_time_arg_count =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("common_runtime_arg_offset")) {
      accessor.common_runtime_arg_offset =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("common_runtime_arg_count")) {
      accessor.common_runtime_arg_count =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("args_config_bits")) {
      accessor.args_config_bits =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("layout")) {
      accessor.layout = Downcast<String>(v.value());
    }
    if (auto v = accessor_info.Get("memory_space")) {
      accessor.memory_space = Downcast<String>(v.value());
    }
    if (!accessor.buffer.empty()) {
      if (accessor.compile_time_arg_offset == 0 && accessor.slot != 0) {
        accessor.compile_time_arg_offset = accessor.slot;
      }
      accessor.slot = accessor.compile_time_arg_offset;
      if (accessor.compile_time_arg_count == 0) {
        accessor.compile_time_arg_count =
            accessor.layout == "interleaved" ? 2U : 0U;
      }
      accessors.push_back(std::move(accessor));
    }
  }
  return accessors;
}

static std::vector<CompileTimeArgSpec> ExtractCompileTimeArgSpecsFromArray(
    const ffi::Array<ffi::Any>& items) {
  std::vector<CompileTimeArgSpec> compile_time_arg_specs;
  for (const auto& item : items) {
    auto spec_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (spec_info.empty()) {
      continue;
    }

    CompileTimeArgSpec spec;
    if (auto v = spec_info.Get("name")) {
      spec.name = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("kind")) {
      spec.kind = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("dtype")) {
      spec.dtype = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("offset")) {
      spec.offset = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = spec_info.Get("count")) {
      spec.count = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = spec_info.Get("buffer")) {
      spec.buffer = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("segment_role")) {
      spec.segment_role = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("values")) {
      for (const auto& value : Downcast<ffi::Array<ffi::Any>>(v.value())) {
        spec.values.push_back(static_cast<uint32_t>(Downcast<Integer>(value).IntValue()));
      }
    }
    if (auto v = spec_info.Get("layout")) {
      spec.layout = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get("memory_space")) {
      spec.memory_space = Downcast<String>(v.value());
    }
    if (spec.count == 0) {
      spec.count = static_cast<uint32_t>(spec.values.size());
    }

    if (!spec.kind.empty()) {
      compile_time_arg_specs.push_back(std::move(spec));
    }
  }
  return compile_time_arg_specs;
}

static bool ExtractLaunchSpec(const ffi::Map<ffi::String, ffi::Any>& spec_info,
                              KernelLaunchSpec* launch_spec) {
  if (spec_info.empty()) {
    return false;
  }
  if (auto v = spec_info.Get("core_type")) {
    launch_spec->core_type = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("processor")) {
    launch_spec->processor = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("noc")) {
    launch_spec->noc = Downcast<String>(v.value());
  }
  return !launch_spec->core_type.empty() || !launch_spec->processor.empty() ||
         !launch_spec->noc.empty();
}

static bool IsUnifiedWorkDescriptorKind(const std::string& kind) {
  return kind == "work_linear_id" || kind == "a_tile_start_id" ||
         kind == "a_tile_num_tiles" || kind == "a_tile_stride" ||
         kind == "b_tile_start_id" || kind == "b_tile_num_tiles" ||
         kind == "b_tile_stride" || kind == "output_tile_start_id" ||
         kind == "output_tile_num_tiles" || kind == "output_tile_stride" ||
         kind == "k_tile_start_id" || kind == "num_k_tiles";
}

static std::vector<KernelArgSpec> ExtractRuntimeArgs(const tir::PrimFunc& f) {
  if (auto segment_plan_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.segment_plan")) {
    std::vector<KernelArgSpec> aggregated;
    std::unordered_set<std::string> seen;
    for (const auto& item : segment_plan_attr.value()) {
      auto segment = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (segment.empty()) {
        continue;
      }
      auto runtime_args_it = segment.Get("runtime_args");
      if (!runtime_args_it.has_value()) {
        continue;
      }
      std::vector<KernelArgSpec> segment_args =
          ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(runtime_args_it.value()));
      for (const auto& arg : segment_args) {
        const bool is_buffer_arg =
            arg.kind == "input_buffer_addr32" || arg.kind == "input_buffer_addr" ||
            arg.kind == "output_buffer_addr32" || arg.kind == "output_buffer_addr";
        const std::string dedupe_key = is_buffer_arg
                                           ? (arg.kind + ":" + arg.buffer)
                                           : (IsUnifiedWorkDescriptorKind(arg.kind)
                                                  ? arg.kind
                                                  : (arg.kind + ":" + arg.name));
        if ((is_buffer_arg && arg.buffer.empty()) || seen.count(dedupe_key)) {
          continue;
        }
        aggregated.push_back(arg);
        seen.insert(dedupe_key);
      }
    }
    if (!aggregated.empty()) {
      return aggregated;
    }
  }

  auto runtime_args_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.runtime_args");
  if (!runtime_args_attr) {
    return HasCopyBuiltins(f) ? MakeDefaultCopyRuntimeArgs() : std::vector<KernelArgSpec>{};
  }

  std::vector<KernelArgSpec> runtime_args = ExtractRuntimeArgsFromArray(runtime_args_attr.value());

  if (runtime_args.empty()) {
    return HasCopyBuiltins(f) ? MakeDefaultCopyRuntimeArgs() : std::vector<KernelArgSpec>{};
  }
  return runtime_args;
}

static std::vector<KernelArgSpec> ExtractCommonRuntimeArgs(const tir::PrimFunc& f) {
  if (auto segment_plan_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.segment_plan")) {
    std::vector<KernelArgSpec> aggregated;
    std::unordered_set<std::string> seen;
    for (const auto& item : segment_plan_attr.value()) {
      auto segment = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (segment.empty()) {
        continue;
      }
      auto common_runtime_args_it = segment.Get("common_runtime_args");
      if (!common_runtime_args_it.has_value()) {
        continue;
      }
      std::vector<KernelArgSpec> segment_args = ExtractRuntimeArgsFromArray(
          Downcast<ffi::Array<ffi::Any>>(common_runtime_args_it.value()));
      for (const auto& arg : segment_args) {
        const std::string dedupe_key =
            arg.kind.empty() ? arg.name : (arg.kind + ":" + arg.name);
        if (arg.kind.empty() || seen.count(dedupe_key)) {
          continue;
        }
        aggregated.push_back(arg);
        seen.insert(dedupe_key);
      }
    }
    if (!aggregated.empty()) {
      return aggregated;
    }
  }

  auto runtime_args_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.common_runtime_args");
  if (!runtime_args_attr) {
    return {};
  }
  return ExtractRuntimeArgsFromArray(runtime_args_attr.value());
}

struct SegmentInfo {
  std::string name;
  std::string kind;
  std::string core_type;
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs;
  bool has_launch_spec = false;
  KernelLaunchSpec launch_spec;
  bool has_compute_config = false;
  KernelComputeConfigSpec compute_config;
  std::vector<AccessorSpec> accessors;
};

static std::vector<SegmentInfo> ExtractSegmentPlan(const tir::PrimFunc& f, ExecutableSpec* spec) {
  std::vector<SegmentInfo> segments_out;
  auto segment_plan_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.segment_plan");
  if (!segment_plan_attr) {
    return segments_out;
  }

  const auto& segments = segment_plan_attr.value();
  if (segments.empty()) {
    return segments_out;
  }

  for (const auto& item : segments) {
    auto segment = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (segment.empty()) {
      continue;
    }

    SegmentInfo info;
    if (auto v = segment.Get("name")) {
      info.name = Downcast<String>(v.value());
    }
    if (auto v = segment.Get("kind")) {
      info.kind = Downcast<String>(v.value());
    }
    if (auto v = segment.Get("core_type")) {
      info.core_type = Downcast<String>(v.value());
    }
    if (auto v = segment.Get("runtime_args")) {
      info.runtime_args = ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("common_runtime_args")) {
      info.common_runtime_args =
          ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("accessors")) {
      info.accessors = ExtractAccessorsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("compile_time_arg_specs")) {
      info.compile_time_arg_specs =
          ExtractCompileTimeArgSpecsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("launch_spec")) {
      auto launch_spec = v.value().as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      info.has_launch_spec = ExtractLaunchSpec(launch_spec, &info.launch_spec);
    }
    if (auto v = segment.Get("compute_config")) {
      auto compute_config = v.value().as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      info.has_compute_config = ExtractComputeConfig(compute_config, &info.compute_config);
    }

    if (segments_out.empty()) {
      if (!info.kind.empty()) {
        spec->default_kernel_kind = info.kind;
      }
      if (!info.core_type.empty()) {
        spec->default_kernel_core_type = info.core_type;
      }
    }

    segments_out.push_back(std::move(info));
  }
  return segments_out;
}

static std::string GetPrimFuncName(const GlobalVar& gvar, const tir::PrimFunc& f) {
  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  return global_symbol ? static_cast<std::string>(global_symbol.value())
                       : static_cast<std::string>(gvar->name_hint);
}

static bool IsBlackholeDeviceKernel(const tir::PrimFunc& f) {
  auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
  if (calling_conv.defined()) {
    return calling_conv == CallingConv::kDeviceKernelLaunch;
  }
  return static_cast<bool>(f->GetAttr<ffi::Array<ffi::Any>>("blackhole.segment_plan")) ||
         static_cast<bool>(f->GetAttr<ffi::Map<ffi::String, ffi::Any>>("blackhole.core_plan"));
}

static bool IsBlackholeHostEntry(const tir::PrimFunc& f) {
  auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
  return calling_conv.defined() && calling_conv == CallingConv::kCPackedFunc;
}

static ExecutableSpec ExtractExecutableSpecFromDeviceFunc(const tir::PrimFunc& f,
                                                          const std::string& entry_name) {
  ExecutableSpec spec;
  spec.entry_name = entry_name;

  for (size_t i = 0; i < f->params.size(); ++i) {
    DLDataType dtype = f->params[i]->dtype;
    if (dtype.code == kDLBool) {
      dtype.code = kDLInt;
      dtype.bits = 32;
    }
    spec.tvm_arg_names.push_back(f->params[i]->name_hint);
    spec.tvm_arg_types.push_back(dtype);
    spec.tvm_is_buffer_arg.push_back(dtype.code == kDLOpaqueHandle);
  }

  spec.cb_configs = ExtractCBConfig(f);
  spec.core_plan = ExtractCorePlan(f);
  spec.runtime_args = ExtractRuntimeArgs(f);
  spec.gemm_contract = ExtractGemmContract(f);
  spec.compute_contract = ExtractComputeContract(f, spec.gemm_contract);
  ExtractSegmentPlan(f, &spec);
  return spec;
}

class SegmentBodyExtractor final : public tir::StmtMutator {
 public:
  explicit SegmentBodyExtractor(std::string segment_kind)
      : segment_kind_(std::move(segment_kind)) {}

  Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
    bool has_segment_markers = false;
    ffi::Array<Stmt> seq;
    seq.reserve(op->seq.size());

    for (const Stmt& stmt : op->seq) {
      if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
        if (attr->attr_key == "blackhole.segment_kind") {
          has_segment_markers = true;
          if (const auto* kind = attr->value.as<StringImmNode>()) {
            if (kind->value == segment_kind_) {
              seq.push_back(this->VisitStmt(attr->body));
            }
          }
          continue;
        }
      }
      seq.push_back(this->VisitStmt(stmt));
    }

    if (!has_segment_markers) {
      return tir::StmtMutator::VisitStmt_(op);
    }
    if (seq.empty()) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (seq.size() == 1) {
      return seq[0];
    }
    return tir::SeqStmt(seq);
  }

 private:
  std::string segment_kind_;
};

static ffi::Array<ffi::Any> EncodeRuntimeArgs(const std::vector<KernelArgSpec>& runtime_args) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& arg : runtime_args) {
    ffi::Map<ffi::String, ffi::Any> arg_info;
    arg_info.Set("name", ffi::String(arg.name));
    arg_info.Set("kind", ffi::String(arg.kind));
    arg_info.Set("dtype", ffi::String(arg.dtype));
    if (!arg.buffer.empty()) {
      arg_info.Set("buffer", ffi::String(arg.buffer));
    }
    encoded.push_back(arg_info);
  }
  return encoded;
}

static ffi::Array<ffi::Any> EncodeAccessors(const std::vector<AccessorSpec>& accessors) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& accessor : accessors) {
    ffi::Map<ffi::String, ffi::Any> accessor_info;
    accessor_info.Set("buffer", ffi::String(accessor.buffer));
    accessor_info.Set("slot", Integer(static_cast<int>(accessor.slot)));
    accessor_info.Set("compile_time_arg_offset",
                      Integer(static_cast<int>(accessor.compile_time_arg_offset)));
    accessor_info.Set("compile_time_arg_count",
                      Integer(static_cast<int>(accessor.compile_time_arg_count)));
    accessor_info.Set("common_runtime_arg_offset",
                      Integer(static_cast<int>(accessor.common_runtime_arg_offset)));
    accessor_info.Set("common_runtime_arg_count",
                      Integer(static_cast<int>(accessor.common_runtime_arg_count)));
    accessor_info.Set("args_config_bits", Integer(static_cast<int>(accessor.args_config_bits)));
    accessor_info.Set("layout", ffi::String(accessor.layout));
    accessor_info.Set("memory_space", ffi::String(accessor.memory_space));
    encoded.push_back(accessor_info);
  }
  return encoded;
}

static tir::PrimFunc MakeSegmentPrimFunc(const tir::PrimFunc& f, const SegmentInfo& segment) {
  SegmentBodyExtractor extractor(segment.kind);
  tir::PrimFunc segment_func = f;
  segment_func.CopyOnWrite()->body = extractor(f->body);

  ffi::Map<ffi::String, ffi::Any> attrs;
  if (f->attrs.defined()) {
    attrs = f->attrs->dict;
  }
  attrs.Set("blackhole.core_type", ffi::String(segment.core_type));
  if (!segment.runtime_args.empty()) {
    attrs.Set("blackhole.runtime_args", EncodeRuntimeArgs(segment.runtime_args));
  }
  if (!segment.common_runtime_args.empty()) {
    attrs.Set("blackhole.common_runtime_args", EncodeRuntimeArgs(segment.common_runtime_args));
  }
  if (!segment.accessors.empty()) {
    attrs.Set("blackhole.accessors", EncodeAccessors(segment.accessors));
  }
  segment_func.CopyOnWrite()->attrs = tvm::DictAttrs(attrs);
  return segment_func;
}

static std::string EmitKernelSourceForPrimFunc(const tir::PrimFunc& f,
                                               const std::string& func_name,
                                               Target target,
                                               bool kernel_code_only) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = !kernel_code_only;
  std::unordered_set<std::string> devices = {"blackhole"};

  tl::CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.AddFunction(GlobalVar(func_name), f);
  return kernel_code_only ? cg.GetKernelCode() : cg.Finish();
}

static void PopulateKernelSpecsForDeviceFunc(const tir::PrimFunc& f,
                                             const std::string& func_name,
                                             Target target,
                                             bool kernel_code_only,
                                             const std::string& legacy_code,
                                             ExecutableSpec* spec) {
  spec->kernels.clear();
  std::vector<SegmentInfo> segments = ExtractSegmentPlan(f, spec);
  const bool use_legacy_single_kernel_path = segments.empty();
  if (use_legacy_single_kernel_path) {
    KernelSpec kernel;
    kernel.name = func_name;
    kernel.kind = spec->default_kernel_kind;
    kernel.core_type = spec->default_kernel_core_type;
    kernel.runtime_args = spec->runtime_args;
    kernel.common_runtime_args = ExtractCommonRuntimeArgs(f);
    kernel.source_code = legacy_code;
    ICHECK(!kernel.source_code.empty())
        << "Blackhole build produced no kernel source and no copy fallback was applicable for "
        << func_name;
    spec->kernels.push_back(std::move(kernel));
    return;
  }

  for (const SegmentInfo& segment : segments) {
    tir::PrimFunc segment_func = MakeSegmentPrimFunc(f, segment);
    KernelSpec kernel;
    kernel.name = func_name + "_" + (segment.name.empty() ? segment.kind : segment.name);
    kernel.kind = segment.kind;
    kernel.core_type = segment.core_type;
    kernel.runtime_args = segment.runtime_args.empty() ? spec->runtime_args : segment.runtime_args;
    kernel.common_runtime_args = segment.common_runtime_args;
    kernel.compile_time_arg_specs = segment.compile_time_arg_specs;
    kernel.has_launch_spec = segment.has_launch_spec;
    if (segment.has_launch_spec) {
      kernel.launch_spec = segment.launch_spec;
    }
    kernel.has_compute_config = segment.has_compute_config;
    if (segment.has_compute_config) {
      kernel.compute_config = segment.compute_config;
    }
    kernel.accessors = segment.accessors;
    kernel.source_code = EmitKernelSourceForPrimFunc(segment_func, kernel.name, target,
                                                     kernel_code_only);
    ICHECK(!kernel.source_code.empty())
        << "Blackhole build produced no kernel source for segment "
        << segment.kind << " of " << func_name;
    spec->kernels.push_back(std::move(kernel));
  }
}

static std::string FindLaunchedKernelSymbol(
    const tir::PrimFunc& f,
    const std::unordered_set<std::string>& device_kernel_symbols) {
  std::string kernel_symbol;
  tir::PostOrderVisit(f->body, [&](const ObjectRef& node) {
    if (!kernel_symbol.empty()) {
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call || !call->op.same_as(tir::builtin::tvm_call_packed()) ||
        call->args.empty()) {
      return;
    }
    if (const auto* callee = call->args[0].as<StringImmNode>()) {
      std::string name = callee->value;
      if (device_kernel_symbols.count(name)) {
        kernel_symbol = std::move(name);
      }
    }
  });
  return kernel_symbol;
}

/*!
 * \brief Extract executable specs for the Blackhole backend.
 * \param mod The IR module
 * \return Map from function name to ExecutableSpec
 */
static std::unordered_map<std::string, ExecutableSpec> ExtractBlackholeFuncInfo(
    const IRModule& mod) {
  std::unordered_map<std::string, ExecutableSpec> fmap;
  std::unordered_map<std::string, ExecutableSpec> device_specs;
  std::unordered_map<std::string, tir::PrimFunc> host_entries;
  std::unordered_set<std::string> device_kernel_symbols;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    std::string func_name = GetPrimFuncName(gvar, f);

    if (IsBlackholeDeviceKernel(f)) {
      device_kernel_symbols.insert(func_name);
      device_specs.emplace(func_name, ExtractExecutableSpecFromDeviceFunc(f, func_name));
    } else if (IsBlackholeHostEntry(f)) {
      host_entries.emplace(func_name, f);
    }
  }

  for (auto& kv : device_specs) {
    fmap.emplace(kv.first, kv.second);
  }

  for (const auto& kv : host_entries) {
    const std::string launched_kernel =
        FindLaunchedKernelSymbol(kv.second, device_kernel_symbols);
    if (launched_kernel.empty()) {
      continue;
    }
    auto it = device_specs.find(launched_kernel);
    if (it == device_specs.end()) {
      continue;
    }
    ExecutableSpec host_spec = it->second;
    host_spec.entry_name = kv.first;
    fmap[kv.first] = std::move(host_spec);
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

  auto func_info_map = ExtractBlackholeFuncInfo(mod);
  std::unordered_map<std::string, tir::PrimFunc> device_funcs;
  std::unordered_set<std::string> legacy_single_kernel_funcs;
  std::unordered_map<std::string, std::string> host_to_device;
  std::unordered_set<std::string> device_kernel_symbols;

  // Create temporary directory for kernel files
  std::string kernel_dir = "/tmp/tilelang_blackhole/" + std::to_string(getpid());
  std::filesystem::create_directories(kernel_dir);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (IsBlackholeDeviceKernel(f)) {
      device_kernel_symbols.insert(func_name);
      device_funcs.emplace(func_name, f);
      ExecutableSpec probe_spec;
      auto segments = ExtractSegmentPlan(f, &probe_spec);
      if (segments.empty() || (segments.size() == 1 && probe_spec.default_kernel_kind == "fused_dataflow")) {
        legacy_single_kernel_funcs.insert(func_name);
      }
    }
  }
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    if (IsBlackholeHostEntry(f)) {
      const std::string host_name = GetPrimFuncName(gvar, f);
      const std::string launched_kernel = FindLaunchedKernelSymbol(f, device_kernel_symbols);
      if (!launched_kernel.empty()) {
        host_to_device.emplace(host_name, launched_kernel);
      }
    }
  }

  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;
  std::unordered_set<std::string> devices = {"blackhole"};
  tl::CodeGenBlackhole legacy_cg;
  legacy_cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  bool has_legacy_funcs = false;
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (legacy_single_kernel_funcs.count(func_name)) {
      legacy_cg.AddFunction(gvar, f);
      has_legacy_funcs = true;
    }
  }
  const std::string legacy_code = has_legacy_funcs ? legacy_cg.Finish() : std::string();

  for (auto& kv : device_funcs) {
    auto spec_it = func_info_map.find(kv.first);
    if (spec_it == func_info_map.end()) {
      continue;
    }
    const std::string& source = legacy_single_kernel_funcs.count(kv.first) ? legacy_code : std::string();
    PopulateKernelSpecsForDeviceFunc(kv.second, kv.first, target, /*kernel_code_only=*/false,
                                     source,
                                     &spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
    }
  }

  size_t total_code_size = 0;
  for (const auto& kv : func_info_map) {
    for (const auto& kernel : kv.second.kernels) {
      total_code_size += kernel.source_code.size();
    }
  }

  LOG(INFO) << "BuildTileLangBlackhole: Generated " << total_code_size
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

  auto func_info_map = ExtractBlackholeFuncInfo(mod);
  std::unordered_map<std::string, tir::PrimFunc> device_funcs;
  std::unordered_set<std::string> legacy_single_kernel_funcs;
  std::unordered_map<std::string, std::string> host_to_device;
  std::unordered_set<std::string> device_kernel_symbols;

  // Create temporary directory for kernel files
  std::string kernel_dir = "/tmp/tilelang_blackhole/" + std::to_string(getpid());
  std::filesystem::create_directories(kernel_dir);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (IsBlackholeDeviceKernel(f)) {
      device_kernel_symbols.insert(func_name);
      device_funcs.emplace(func_name, f);
      ExecutableSpec probe_spec;
      auto segments = ExtractSegmentPlan(f, &probe_spec);
      if (segments.empty() || (segments.size() == 1 && probe_spec.default_kernel_kind == "fused_dataflow")) {
        legacy_single_kernel_funcs.insert(func_name);
      }
    }
  }
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    if (IsBlackholeHostEntry(f)) {
      const std::string host_name = GetPrimFuncName(gvar, f);
      const std::string launched_kernel = FindLaunchedKernelSymbol(f, device_kernel_symbols);
      if (!launched_kernel.empty()) {
        host_to_device.emplace(host_name, launched_kernel);
      }
    }
  }
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = false;
  std::unordered_set<std::string> devices = {"blackhole"};
  tl::CodeGenBlackhole legacy_cg;
  legacy_cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  bool has_legacy_funcs = false;
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (legacy_single_kernel_funcs.count(func_name)) {
      legacy_cg.AddFunction(gvar, f);
      has_legacy_funcs = true;
    }
  }
  const std::string legacy_kernel_code =
      has_legacy_funcs ? legacy_cg.GetKernelCode() : std::string();

  for (auto& kv : device_funcs) {
    auto spec_it = func_info_map.find(kv.first);
    if (spec_it == func_info_map.end()) {
      continue;
    }
    const std::string& source =
        legacy_single_kernel_funcs.count(kv.first) ? legacy_kernel_code : std::string();
    PopulateKernelSpecsForDeviceFunc(kv.second, kv.first, target, /*kernel_code_only=*/true,
                                     source,
                                     &spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
    }
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
