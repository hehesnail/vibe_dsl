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
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <unistd.h>

#include "codegen_blackhole.h"
#include "blackhole_module.h"
#include "tt_program_projection.h"
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
        rv->operator=(110);
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
    .add_attr_option<int64_t>("num_cores", 110)  // 11x10 logical worker cores
    .add_attr_option<int64_t>("logical_worker_grid_x", 11)
    .add_attr_option<int64_t>("logical_worker_grid_y", 10)
    .add_attr_option<int64_t>("num_cbs", 64)     // 64 circular buffers per core
    .set_default_keys({"blackhole"});

/*!
 * \brief Extract CB configuration from PrimFunc attrs.
 * \param f The PrimFunc
 * \return Vector of CB configurations
 */
static std::vector<CBConfig> ExtractCBConfig(const tir::PrimFunc& f) {
  std::vector<CBConfig> cb_configs;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto cb_attr = executable.Get(String(tl::tt_program_projection::executable_key::kCBConfigs))
                     ? Downcast<ffi::Array<ffi::Any>>(executable.Get(
                           String(tl::tt_program_projection::executable_key::kCBConfigs))
                                                           .value())
                     : ffi::Array<ffi::Any>();
  if (cb_attr.empty()) {
    // Use default CB config for simple kernels
    CBConfig config;
    config.cb_id = 0;
    config.name = "default_cb";
    config.role = "intermediate";
    config.num_pages = 1;
    config.page_size_bytes = 2048;
    config.data_format = "Float16_b";
    cb_configs.push_back(std::move(config));
    return cb_configs;
  }

  for (const auto& item : cb_attr) {
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
    if (auto initial_reserve_pages = cb_info.Get("initial_reserve_pages")) {
      config.initial_reserve_pages =
          Downcast<Integer>(initial_reserve_pages.value()).IntValue();
    }
    if (auto flow_class = cb_info.Get("flow_class")) {
      config.flow_class = Downcast<String>(flow_class.value());
    }
    if (auto publish_pages = cb_info.Get("publish_pages_per_event")) {
      config.publish_pages_per_event = Downcast<Integer>(publish_pages.value()).IntValue();
    }
    if (auto consume_pages = cb_info.Get("consume_pages_per_event")) {
      config.consume_pages_per_event = Downcast<Integer>(consume_pages.value()).IntValue();
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
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto core_plan =
      executable.Get(String(tl::tt_program_projection::executable_key::kCorePlan))
          ? executable.Get(String(tl::tt_program_projection::executable_key::kCorePlan))
                .value()
                .as<ffi::Map<ffi::String, ffi::Any>>()
                .value_or(ffi::Map<ffi::String, ffi::Any>())
          : ffi::Map<ffi::String, ffi::Any>();
  if (core_plan.empty()) {
    return plan;
  }

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
  return plan;
}

static std::vector<SemaphoreSpec> ExtractSemaphorePlan(const tir::PrimFunc& f) {
  std::vector<SemaphoreSpec> semaphores;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto semaphore_attr =
      executable.Get(String(tl::tt_program_projection::executable_key::kSemaphorePlan))
          ? Downcast<ffi::Array<ffi::Any>>(
                executable.Get(String(tl::tt_program_projection::executable_key::kSemaphorePlan))
                    .value())
          : ffi::Array<ffi::Any>();
  if (semaphore_attr.empty()) {
    return semaphores;
  }

  for (const auto& item : semaphore_attr) {
    auto semaphore_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (semaphore_info.empty()) {
      continue;
    }

    SemaphoreSpec spec;
    if (auto v = semaphore_info.Get("id")) {
      spec.id = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = semaphore_info.Get("initial_value")) {
      spec.initial_value = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = semaphore_info.Get("core_type")) {
      spec.core_type = Downcast<String>(v.value());
    }
    if (auto v = semaphore_info.Get("core_ranges")) {
      for (const auto& range_any : Downcast<ffi::Array<ffi::Any>>(v.value())) {
        auto range_info = range_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
            ffi::Map<ffi::String, ffi::Any>());
        if (range_info.empty()) {
          continue;
        }

        auto parse_coord = [](const ffi::Optional<ffi::Any>& coord_any) {
          PhysicalCore coord;
          if (!coord_any.has_value()) {
            return coord;
          }
          auto coord_info = coord_any.value().as<ffi::Map<ffi::String, ffi::Any>>().value_or(
              ffi::Map<ffi::String, ffi::Any>());
          if (auto x = coord_info.Get("core_x")) {
            coord.core_x = static_cast<uint32_t>(Downcast<Integer>(x.value()).IntValue());
          }
          if (auto y = coord_info.Get("core_y")) {
            coord.core_y = static_cast<uint32_t>(Downcast<Integer>(y.value()).IntValue());
          }
          return coord;
        };

        CoreRangeSpec range_spec;
        range_spec.start = parse_coord(range_info.Get("start"));
        range_spec.end = parse_coord(range_info.Get("end"));
        spec.core_ranges.push_back(std::move(range_spec));
      }
    }
    semaphores.push_back(std::move(spec));
  }

  return semaphores;
}

static GemmContractSpec ParseGemmContract(const ffi::Map<ffi::String, ffi::Any>& attrs) {
  GemmContractSpec contract;
  if (attrs.empty()) {
    return contract;
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

static GemmContractSpec ExtractGemmContract(const tir::PrimFunc& f) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto attrs =
      executable.Get(String(tl::tt_program_projection::executable_key::kGemmContract))
          ? executable.Get(String(tl::tt_program_projection::executable_key::kGemmContract))
                .value()
                .as<ffi::Map<ffi::String, ffi::Any>>()
                .value_or(ffi::Map<ffi::String, ffi::Any>())
          : ffi::Map<ffi::String, ffi::Any>();
  return ParseGemmContract(attrs);
}

static std::vector<GemmContractSpec> ExtractMultiGemmContracts(const tir::PrimFunc& f) {
  std::vector<GemmContractSpec> contracts;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto maybe_items =
      executable.Get(String(tl::tt_program_projection::executable_key::kMultiGemmContracts));
  if (!maybe_items) {
    return contracts;
  }
  for (const auto& item : Downcast<ffi::Array<ffi::Any>>(maybe_items.value())) {
    auto attrs =
        item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(ffi::Map<ffi::String, ffi::Any>());
    GemmContractSpec contract = ParseGemmContract(attrs);
    if (contract.enabled) {
      contracts.push_back(std::move(contract));
    }
  }
  return contracts;
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
  contract.dst_full_sync_en = false;
  contract.math_approx_mode = false;
  contract.bfp8_pack_precise = false;
  contract.clear_accum = false;
  contract.k_pack = 1;
  contract.wg_wait = 0;
  contract.policy_type = 0;
  contract.policy_name = "Square";
  contract.defines = {};
  contract.named_compile_args = {};
  contract.has_mbarrier = false;
  return contract;
}

static std::vector<ComputeContractSpec::EpilogueOpSpec> ParseComputeEpilogueOps(
    const ffi::Map<ffi::String, ffi::Any>& attrs, const char* key = "epilogue_ops") {
  std::vector<ComputeContractSpec::EpilogueOpSpec> epilogue_ops;
  if (auto v = attrs.Get(key)) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto op =
          item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(ffi::Map<ffi::String, ffi::Any>());
      if (op.empty()) {
        continue;
      }
      ComputeContractSpec::EpilogueOpSpec entry;
      if (auto value = op.Get("kind")) entry.kind = Downcast<String>(value.value());
      if (auto value = op.Get("dst_buffer")) entry.dst_buffer = Downcast<String>(value.value());
      if (auto value = op.Get("src_buffer")) entry.src_buffer = Downcast<String>(value.value());
      if (auto value = op.Get("scalar_buffer")) {
        entry.scalar_buffer = Downcast<String>(value.value());
      }
      if (auto value = op.Get("lhs_buffer")) entry.lhs_buffer = Downcast<String>(value.value());
      if (auto value = op.Get("rhs_buffer")) entry.rhs_buffer = Downcast<String>(value.value());
      if (auto value = op.Get("add_buffer")) entry.add_buffer = Downcast<String>(value.value());
      if (auto value = op.Get("reduce_kind")) entry.reduce_kind = Downcast<String>(value.value());
      if (auto value = op.Get("num_elements_expr")) {
        entry.num_elements_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("row_width_expr")) {
        entry.row_width_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("dst_offset_expr")) {
        entry.dst_offset_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("src_offset_expr")) {
        entry.src_offset_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("dst_scale_expr")) {
        entry.dst_scale_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("scalar_scale_expr")) {
        entry.scalar_scale_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("lhs_scale_expr")) {
        entry.lhs_scale_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("rhs_scale_expr")) {
        entry.rhs_scale_expr = Downcast<String>(value.value());
      }
      if (auto value = op.Get("grouped")) entry.grouped = Downcast<Bool>(value.value());
      if (auto value = op.Get("clear")) entry.clear = Downcast<Bool>(value.value());
      if (auto value = op.Get("publish_cb")) entry.publish_cb = Downcast<Bool>(value.value());
      if (auto value = op.Get("buffer_materialization_contract")) {
        auto contract =
            value.value()
                .as<ffi::Map<ffi::String, ffi::Any>>()
                .value_or(ffi::Map<ffi::String, ffi::Any>());
        if (auto field = contract.Get("kind")) {
          entry.buffer_materialization_contract.kind = Downcast<String>(field.value());
        }
        if (auto field = contract.Get("target_buffer")) {
          entry.buffer_materialization_contract.target_buffer =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("scope")) {
          entry.buffer_materialization_contract.scope = Downcast<String>(field.value());
        }
        if (auto field = contract.Get("materialization_kind")) {
          entry.buffer_materialization_contract.materialization_kind =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("bridge_kind")) {
          entry.buffer_materialization_contract.bridge_kind =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("value_role")) {
          entry.buffer_materialization_contract.value_role = Downcast<String>(field.value());
        }
        if (auto field = contract.Get("merge_kind")) {
          entry.buffer_materialization_contract.merge_kind = Downcast<String>(field.value());
        }
        if (auto field = contract.Get("execution_protocol")) {
          entry.buffer_materialization_contract.execution_protocol =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("result_live_form")) {
          entry.buffer_materialization_contract.result_live_form =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("source_buffer")) {
          entry.buffer_materialization_contract.source_buffer =
              Downcast<String>(field.value());
        }
        if (auto field = contract.Get("logical_row_width")) {
          entry.buffer_materialization_contract.logical_row_width =
              static_cast<int>(Downcast<Integer>(field.value())->value);
        }
        if (auto field = contract.Get("logical_element_count")) {
          entry.buffer_materialization_contract.logical_element_count =
              static_cast<int>(Downcast<Integer>(field.value())->value);
        }
      }
      if (!entry.kind.empty()) {
        epilogue_ops.push_back(std::move(entry));
      }
    }
  }
  return epilogue_ops;
}

static ComputeContractSpec ParseComputeContract(const ffi::Map<ffi::String, ffi::Any>& attrs) {
  ComputeContractSpec contract;
  if (attrs.empty()) {
    return contract;
  }
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
  if (auto v = attrs.Get("dst_full_sync_en")) {
    contract.dst_full_sync_en = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("math_approx_mode")) {
    contract.math_approx_mode = Downcast<Bool>(v.value());
  }
  if (auto v = attrs.Get("bfp8_pack_precise")) {
    contract.bfp8_pack_precise = Downcast<Bool>(v.value());
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
  if (auto v = attrs.Get("defines")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto define = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (define.empty()) {
        continue;
      }
      KernelDefineSpec entry;
      if (auto name = define.Get("name")) {
        entry.name = Downcast<String>(name.value());
      }
      if (auto value = define.Get("value")) {
        entry.value = Downcast<String>(value.value());
      }
      if (!entry.name.empty()) {
        contract.defines.push_back(std::move(entry));
      }
    }
  }
  if (auto v = attrs.Get("named_compile_args")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto arg = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      NamedCompileArgSpec entry;
      if (auto name = arg.Get("name")) {
        entry.name = Downcast<String>(name.value());
      }
      if (auto value = arg.Get("value")) {
        entry.value = static_cast<uint32_t>(Downcast<Integer>(value.value())->value);
      }
      if (!entry.name.empty()) {
        contract.named_compile_args.push_back(std::move(entry));
      }
    }
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
  contract.epilogue_ops = ParseComputeEpilogueOps(attrs);

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

static ComputeContractSpec ExtractComputeContract(const tir::PrimFunc& f,
                                                  const GemmContractSpec& gemm_contract) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto attrs =
      executable.Get(String(tl::tt_program_projection::executable_key::kComputeContract))
          ? executable.Get(String(tl::tt_program_projection::executable_key::kComputeContract))
                .value()
                .as<ffi::Map<ffi::String, ffi::Any>>()
                .value_or(ffi::Map<ffi::String, ffi::Any>())
          : ffi::Map<ffi::String, ffi::Any>();
  if (attrs.empty()) {
    return ComputeContractFromLegacyGemm(gemm_contract);
  }
  return ParseComputeContract(attrs);
}

static std::vector<ComputeContractSpec> ExtractMultiComputeContracts(const tir::PrimFunc& f) {
  std::vector<ComputeContractSpec> contracts;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto maybe_items =
      executable.Get(String(tl::tt_program_projection::executable_key::kMultiComputeContracts));
  if (!maybe_items) {
    return contracts;
  }
  for (const auto& item : Downcast<ffi::Array<ffi::Any>>(maybe_items.value())) {
    auto attrs =
        item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(ffi::Map<ffi::String, ffi::Any>());
    ComputeContractSpec contract = ParseComputeContract(attrs);
    if (contract.enabled) {
      contracts.push_back(std::move(contract));
    }
  }
  return contracts;
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
  if (auto v = spec_info.Get("dst_full_sync_en")) {
    compute_config->dst_full_sync_en = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("math_approx_mode")) {
    compute_config->math_approx_mode = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("bfp8_pack_precise")) {
    compute_config->bfp8_pack_precise = Downcast<Bool>(v.value());
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
  if (auto v = spec_info.Get("defines")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto define = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (define.empty()) {
        continue;
      }
      KernelDefineSpec entry;
      if (auto name = define.Get("name")) {
        entry.name = Downcast<String>(name.value());
      }
      if (auto value = define.Get("value")) {
        entry.value = Downcast<String>(value.value());
      }
      if (!entry.name.empty()) {
        compute_config->defines.push_back(std::move(entry));
      }
    }
  }
  if (auto v = spec_info.Get("named_compile_args")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto arg = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
          ffi::Map<ffi::String, ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      NamedCompileArgSpec entry;
      if (auto name = arg.Get("name")) {
        entry.name = Downcast<String>(name.value());
      }
      if (auto value = arg.Get("value")) {
        entry.value = static_cast<uint32_t>(Downcast<Integer>(value.value())->value);
      }
      if (!entry.name.empty()) {
        compute_config->named_compile_args.push_back(std::move(entry));
      }
    }
  }
  if (auto v = spec_info.Get("unpack_to_dest_mode")) {
    for (const auto& mode : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      compute_config->unpack_to_dest_mode.push_back(Downcast<String>(mode));
    }
  }
  return !compute_config->math_fidelity.empty() || compute_config->fp32_dest_acc_en ||
         compute_config->dst_full_sync_en || compute_config->math_approx_mode ||
         compute_config->bfp8_pack_precise || compute_config->clear_accum ||
         compute_config->k_pack != 1 || compute_config->wg_wait != 0 ||
         !compute_config->defines.empty() || !compute_config->named_compile_args.empty() ||
         !compute_config->unpack_to_dest_mode.empty();
}

static KernelComputeConfigSpec ComputeConfigFromContract(const ComputeContractSpec& contract) {
  KernelComputeConfigSpec compute_config;
  compute_config.math_fidelity = contract.math_fidelity;
  compute_config.fp32_dest_acc_en = contract.fp32_dest_acc_en;
  compute_config.dst_full_sync_en = contract.dst_full_sync_en;
  compute_config.math_approx_mode = contract.math_approx_mode;
  compute_config.unpack_to_dest_mode = contract.unpack_to_dest_mode;
  compute_config.bfp8_pack_precise = contract.bfp8_pack_precise;
  compute_config.clear_accum = contract.clear_accum;
  compute_config.k_pack = contract.k_pack;
  compute_config.wg_wait = contract.wg_wait;
  compute_config.policy_type = contract.policy_type;
  compute_config.policy_name = contract.policy_name;
  compute_config.defines = contract.defines;
  compute_config.named_compile_args = contract.named_compile_args;
  return compute_config;
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
    if (auto v = arg_info.Get("identity")) {
      arg.identity = Downcast<String>(v.value());
    }
    if (auto v = arg_info.Get("core_x")) {
      arg.core_x = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      arg.has_core_coord = true;
    }
    if (auto v = arg_info.Get("core_y")) {
      arg.core_y = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      arg.has_core_coord = true;
    }
    if (!arg.kind.empty()) {
      ICHECK(!arg.identity.empty())
          << "Blackhole runtime/common-runtime arg '" << arg.name << "' kind '" << arg.kind
          << "' is missing explicit identity";
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
    if (auto v = accessor_info.Get("transport_page_size")) {
      accessor.transport_page_size_bytes =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = accessor_info.Get("layout")) {
      accessor.layout = Downcast<String>(v.value());
    }
    if (auto v = accessor_info.Get("memory_space")) {
      accessor.memory_space = Downcast<String>(v.value());
    }
    if (auto v = accessor_info.Get("host_axis_order")) {
      for (const auto& axis : Downcast<ffi::Array<ffi::Any>>(v.value())) {
        accessor.host_axis_order.push_back(Downcast<Integer>(axis).IntValue());
      }
    }
    if (auto v = accessor_info.Get("transpose_2d")) {
      accessor.transpose_2d = Downcast<Bool>(v.value());
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

static std::vector<SemaphoreBindingSpec> ExtractSemaphoreBindingsFromArray(
    const ffi::Array<ffi::Any>& items) {
  std::vector<SemaphoreBindingSpec> bindings;
  for (const auto& item : items) {
    auto binding_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (binding_info.empty()) {
      continue;
    }

    SemaphoreBindingSpec binding;
    if (auto v = binding_info.Get("name")) {
      binding.name = Downcast<String>(v.value());
    }
    if (auto v = binding_info.Get("semaphore_id")) {
      binding.semaphore_id = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = binding_info.Get("arg_kind")) {
      binding.arg_kind = Downcast<String>(v.value());
    }
    if (!binding.name.empty()) {
      bindings.push_back(std::move(binding));
    }
  }
  return bindings;
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
    if (auto v = spec_info.Get("args_config_bits")) {
      spec.args_config_bits =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (auto v = spec_info.Get("transport_page_size")) {
      spec.transport_page_size_bytes =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
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

static std::vector<PerWorkArgSpec> ExtractPerWorkArgSpecsFromArray(
    const ffi::Array<ffi::Any>& items) {
  std::vector<PerWorkArgSpec> per_work_arg_specs;
  for (const auto& item : items) {
    auto spec_info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (spec_info.empty()) {
      continue;
    }

    PerWorkArgSpec spec;
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgKind)) {
      spec.arg_kind = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgIdentity)) {
      spec.arg_identity = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kBuffer)) {
      spec.buffer = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kValueKind)) {
      spec.value_kind = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue)) {
      spec.constant_value = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    if (!spec.arg_kind.empty() && !spec.value_kind.empty()) {
      per_work_arg_specs.push_back(std::move(spec));
    }
  }
  return per_work_arg_specs;
}

static std::vector<KernelArgSpec> AggregateSegmentRuntimeArgs(
    const ffi::Array<ffi::Any>& segment_plan, const char* field_name) {
  std::vector<KernelArgSpec> aggregated;
  std::unordered_set<std::string> seen;
  for (const auto& item : segment_plan) {
    auto segment = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (segment.empty()) {
      continue;
    }
    auto args_it = segment.Get(field_name);
    if (!args_it.has_value()) {
      continue;
    }
    std::vector<KernelArgSpec> segment_args =
        ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(args_it.value()));
    for (const auto& arg : segment_args) {
      const std::string dedupe_key = arg.identity + ":" + arg.kind;
      if (arg.kind.empty() || seen.count(dedupe_key)) {
        continue;
      }
      aggregated.push_back(arg);
      seen.insert(dedupe_key);
    }
  }
  return aggregated;
}

static std::vector<PerWorkArgSpec> AggregateSegmentPerWorkArgSpecs(
    const ffi::Array<ffi::Any>& segment_plan) {
  std::vector<PerWorkArgSpec> aggregated;
  std::unordered_set<std::string> seen;
  for (const auto& item : segment_plan) {
    auto segment = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (segment.empty()) {
      continue;
    }
    auto specs_it = segment.Get(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs);
    if (!specs_it.has_value()) {
      continue;
    }
    std::vector<PerWorkArgSpec> segment_specs =
        ExtractPerWorkArgSpecsFromArray(Downcast<ffi::Array<ffi::Any>>(specs_it.value()));
    for (const auto& spec : segment_specs) {
      const std::string dedupe_key =
          !spec.arg_identity.empty() ? spec.arg_identity : spec.arg_kind;
      if (dedupe_key.empty() || seen.count(dedupe_key)) {
        continue;
      }
      aggregated.push_back(spec);
      seen.insert(dedupe_key);
    }
  }
  return aggregated;
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

static std::vector<KernelArgSpec> ExtractRuntimeArgs(const tir::PrimFunc& f) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto segment_plan =
      executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
          ? Downcast<ffi::Array<ffi::Any>>(
                executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
                    .value())
          : ffi::Array<ffi::Any>();
  return AggregateSegmentRuntimeArgs(segment_plan, "runtime_args");
}

static std::vector<KernelArgSpec> ExtractCommonRuntimeArgs(const tir::PrimFunc& f) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto segment_plan =
      executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
          ? Downcast<ffi::Array<ffi::Any>>(
                executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
                    .value())
          : ffi::Array<ffi::Any>();
  return AggregateSegmentRuntimeArgs(segment_plan, "common_runtime_args");
}

static std::vector<std::string> ExtractDirectRuntimeUnsupportedReasons(const tir::PrimFunc& f) {
  std::vector<std::string> reasons;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto reason_items = executable.Get(
                          String(tl::tt_program_projection::executable_key::
                                     kDirectRuntimeUnsupportedReasons))
                          ? Downcast<ffi::Array<ffi::Any>>(
                                executable.Get(String(
                                                  tl::tt_program_projection::executable_key::
                                                      kDirectRuntimeUnsupportedReasons))
                                    .value())
                          : ffi::Array<ffi::Any>();
  for (const auto& item : reason_items) {
    if (auto reason = item.as<ffi::String>()) {
      reasons.emplace_back(reason.value());
    }
  }
  return reasons;
}

static std::vector<PerWorkArgSpec> ExtractPerWorkArgSpecs(const tir::PrimFunc& f) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto segment_plan =
      executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
          ? Downcast<ffi::Array<ffi::Any>>(
                executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
                    .value())
          : ffi::Array<ffi::Any>();
  return AggregateSegmentPerWorkArgSpecs(segment_plan);
}

struct StaticBufferInfo {
  std::vector<int64_t> shape;
  DLDataType dtype{};
};

static bool TryExtractStaticBufferInfo(const tir::Buffer& buffer, StaticBufferInfo* info) {
  if (!buffer.defined() || buffer->shape.empty()) {
    return false;
  }
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr& extent : buffer->shape) {
    const auto* int_imm = extent.as<IntImmNode>();
    if (int_imm == nullptr || int_imm->value <= 0) {
      return false;
    }
    shape.push_back(int_imm->value);
  }
  info->shape = std::move(shape);
  info->dtype = buffer->dtype;
  return true;
}

static void RecordStaticBufferInfo(std::unordered_map<std::string, StaticBufferInfo>* by_name,
                                   const std::unordered_set<std::string>& target_buffers,
                                   const tir::Buffer& buffer) {
  if (!buffer.defined() || buffer->name.empty()) {
    return;
  }
  if (!target_buffers.empty() && !target_buffers.count(buffer->name)) {
    return;
  }
  StaticBufferInfo info;
  if (!TryExtractStaticBufferInfo(buffer, &info)) {
    return;
  }
  auto [it, inserted] = by_name->emplace(buffer->name, std::move(info));
  if (inserted) {
    return;
  }
  if (it->second.shape != info.shape || it->second.dtype.code != info.dtype.code ||
      it->second.dtype.bits != info.dtype.bits || it->second.dtype.lanes != info.dtype.lanes) {
    return;
  }
}

static std::unordered_map<std::string, StaticBufferInfo> CollectStaticBufferInfo(
    const tir::PrimFunc& f, const std::unordered_set<std::string>& target_buffers) {
  std::unordered_map<std::string, StaticBufferInfo> by_name;
  for (const auto& kv : f->buffer_map) {
    RecordStaticBufferInfo(&by_name, target_buffers, kv.second);
  }
  tir::PostOrderVisit(f->body, [&](const ObjectRef& node) {
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      RecordStaticBufferInfo(&by_name, target_buffers, load->buffer);
      return;
    }
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      RecordStaticBufferInfo(&by_name, target_buffers, store->buffer);
    }
  });
  return by_name;
}

static int64_t ShapeProduct(const std::vector<int64_t>& shape, size_t begin, size_t end) {
  int64_t product = 1;
  for (size_t i = begin; i < end; ++i) {
    product *= shape[i];
  }
  return product;
}

static std::vector<int64_t> MakeIdentityAxisOrder(size_t ndim) {
  std::vector<int64_t> axis_order;
  axis_order.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    axis_order.push_back(static_cast<int64_t>(i));
  }
  return axis_order;
}

static int AxisOrderDisplacementScore(const std::vector<int64_t>& axis_order) {
  int score = 0;
  for (size_t i = 0; i < axis_order.size(); ++i) {
    score += std::abs(static_cast<int>(axis_order[i]) - static_cast<int>(i));
  }
  return score;
}

static uint32_t GetTotalLogicalWorkItems(const CorePlan& core_plan) {
  uint32_t total = 0;
  for (const auto& packet : core_plan.work_packets) {
    total += packet.work_count;
  }
  return std::max<uint32_t>(1, total);
}

static std::unordered_set<std::string> CollectMaterializedBufferNames(const ExecutableSpec& spec) {
  std::unordered_set<std::string> buffers;
  auto record = [&](const std::string& buffer) {
    if (!buffer.empty()) {
      buffers.insert(buffer);
    }
  };
  for (const auto& kernel : spec.kernels) {
    for (const auto& accessor : kernel.accessors) {
      record(accessor.buffer);
    }
    for (const auto& compile_time_arg_spec : kernel.compile_time_arg_specs) {
      if (!compile_time_arg_spec.layout.empty() && !compile_time_arg_spec.memory_space.empty()) {
        record(compile_time_arg_spec.buffer);
      }
    }
  }
  return buffers;
}

static std::vector<int64_t> InferStaticWorkMajorAxisOrder(
    const std::vector<int64_t>& shape, uint32_t total_work_items, uint32_t tile_rows) {
  const std::vector<int64_t> identity = MakeIdentityAxisOrder(shape.size());
  if (shape.size() <= 2 || total_work_items <= 1) {
    return identity;
  }

  const int64_t total_rows = ShapeProduct(shape, 0, shape.size() - 1);
  if (total_rows <= 0 || total_rows % total_work_items != 0) {
    return identity;
  }
  const int64_t rows_per_work_item = total_rows / total_work_items;
  if (rows_per_work_item <= 0 || rows_per_work_item % tile_rows != 0) {
    return identity;
  }

  std::vector<int64_t> row_axes;
  row_axes.reserve(shape.size() - 1);
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    row_axes.push_back(static_cast<int64_t>(i));
  }

  std::vector<int64_t> best_axis_order;
  int best_score = std::numeric_limits<int>::max();
  do {
    int64_t leading_product = 1;
    for (size_t split = 1; split <= row_axes.size(); ++split) {
      leading_product *= shape[static_cast<size_t>(row_axes[split - 1])];
      if (leading_product > total_work_items) {
        break;
      }
      if (leading_product != total_work_items) {
        continue;
      }
      int64_t trailing_product = 1;
      for (size_t i = split; i < row_axes.size(); ++i) {
        trailing_product *= shape[static_cast<size_t>(row_axes[i])];
      }
      if (trailing_product != rows_per_work_item) {
        continue;
      }
      std::vector<int64_t> candidate = row_axes;
      candidate.push_back(static_cast<int64_t>(shape.size() - 1));
      const int score = AxisOrderDisplacementScore(candidate);
      if (score < best_score) {
        best_score = score;
        best_axis_order = std::move(candidate);
      }
    }
  } while (std::next_permutation(row_axes.begin(), row_axes.end()));

  return best_axis_order.empty() ? identity : best_axis_order;
}

struct SegmentInfo {
  std::string name;
  std::string kind;
  std::string core_type;
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs;
  std::vector<PerWorkArgSpec> per_work_arg_specs;
  bool has_launch_spec = false;
  KernelLaunchSpec launch_spec;
  bool has_compute_config = false;
  KernelComputeConfigSpec compute_config;
  std::vector<AccessorSpec> accessors;
  std::vector<SemaphoreBindingSpec> semaphore_bindings;
  std::vector<RemoteCoreDescriptorSpec> remote_core_descriptors;
};

static void PopulateBufferMaterializationSpecs(
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name,
    ExecutableSpec* spec);

static std::vector<RemoteCoreDescriptorSpec> ExtractRemoteCoreDescriptors(
    const std::vector<KernelArgSpec>& runtime_args) {
  std::unordered_map<std::string, RemoteCoreDescriptorSpec> descriptors;
  for (const auto& arg : runtime_args) {
    if (arg.kind != "logical_core_noc_x" && arg.kind != "logical_core_noc_y") {
      continue;
    }
    ICHECK(!arg.identity.empty())
        << "Blackhole remote core descriptor extraction requires identity for runtime arg "
        << arg.name << " kind=" << arg.kind;
    ICHECK(arg.has_core_coord)
        << "Blackhole remote core descriptor extraction requires core_x/core_y for runtime arg "
        << arg.name << " kind=" << arg.kind;
    auto [it, inserted] =
        descriptors.emplace(arg.identity, RemoteCoreDescriptorSpec{arg.identity, arg.core_x, arg.core_y});
    if (!inserted) {
      ICHECK_EQ(it->second.core_x, arg.core_x)
          << "Blackhole remote core descriptor " << arg.identity
          << " must use one logical core";
      ICHECK_EQ(it->second.core_y, arg.core_y)
          << "Blackhole remote core descriptor " << arg.identity
          << " must use one logical core";
    }
  }

  std::vector<RemoteCoreDescriptorSpec> ordered;
  ordered.reserve(descriptors.size());
  for (auto& entry : descriptors) {
    ordered.push_back(std::move(entry.second));
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const RemoteCoreDescriptorSpec& a, const RemoteCoreDescriptorSpec& b) {
              return a.identity < b.identity;
            });
  return ordered;
}

static std::vector<SegmentInfo> ExtractSegmentPlan(const tir::PrimFunc& f, ExecutableSpec* spec) {
  std::vector<SegmentInfo> segments_out;
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(
      f, "Blackhole executable spec extraction");
  auto segments =
      executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
          ? Downcast<ffi::Array<ffi::Any>>(
                executable.Get(String(tl::tt_program_projection::executable_key::kSegmentPlan))
                    .value())
          : ffi::Array<ffi::Any>();
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
      info.remote_core_descriptors = ExtractRemoteCoreDescriptors(info.runtime_args);
    }
    if (auto v = segment.Get("common_runtime_args")) {
      info.common_runtime_args =
          ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("accessors")) {
      info.accessors = ExtractAccessorsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("semaphore_bindings")) {
      info.semaphore_bindings =
          ExtractSemaphoreBindingsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get("compile_time_arg_specs")) {
      info.compile_time_arg_specs =
          ExtractCompileTimeArgSpecsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = segment.Get(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs)) {
      info.per_work_arg_specs =
          ExtractPerWorkArgSpecsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
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
  return !tl::tt_program_projection::GetBlackholeExecutableProjection(f).empty();
}

static bool IsBlackholeHostEntry(const tir::PrimFunc& f) {
  auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
  return calling_conv.defined() && calling_conv == CallingConv::kCPackedFunc;
}

static void ValidateExtractedCorePlan(const CorePlan& core_plan, const std::string& entry_name) {
  ICHECK(!core_plan.work_packets.empty())
      << "Blackhole planner/runtime contract requires non-empty core_plan.work_packets for "
      << entry_name;
  uint64_t total_work_items = 0;
  for (const auto& packet : core_plan.work_packets) {
    ICHECK_GT(packet.work_count, 0U)
        << "Blackhole planner/runtime contract requires positive work_count in core_plan."
           "work_packets for "
        << entry_name;
    total_work_items += packet.work_count;
  }
  ICHECK_GT(total_work_items, 0U)
      << "Blackhole planner/runtime contract requires at least one logical work item for "
      << entry_name;
}

static ExecutableSpec ExtractExecutableSpecFromDeviceFunc(const tir::PrimFunc& f,
                                                          const std::string& entry_name) {
  auto unsupported_ops = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kUnsupportedComputeOps);
  if (!unsupported_ops.empty()) {
    std::ostringstream os;
    for (int i = 0; i < unsupported_ops.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << Downcast<String>(unsupported_ops[i]);
    }
    ICHECK(false) << "Blackhole compute subset lowering is not implemented for ops ["
                  << os.str() << "]";
  }

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
  ValidateExtractedCorePlan(spec.core_plan, entry_name);
  spec.semaphores = ExtractSemaphorePlan(f);
  spec.runtime_args = ExtractRuntimeArgs(f);
  spec.common_runtime_args = ExtractCommonRuntimeArgs(f);
  spec.per_work_arg_specs = ExtractPerWorkArgSpecs(f);
  spec.gemm_contract = ExtractGemmContract(f);
  spec.compute_contract = ExtractComputeContract(f, spec.gemm_contract);
  spec.multi_gemm_contracts = ExtractMultiGemmContracts(f);
  spec.multi_compute_contracts = ExtractMultiComputeContracts(f);
  spec.direct_runtime_unsupported_reasons = ExtractDirectRuntimeUnsupportedReasons(f);
  ExtractSegmentPlan(f, &spec);
  return spec;
}

class SegmentBodyExtractor final : public tir::StmtMutator {
 public:
  SegmentBodyExtractor(std::string segment_kind, bool retain_unmarked_stmts)
      : segment_kind_(std::move(segment_kind)),
        retain_unmarked_stmts_(retain_unmarked_stmts) {}

  static bool IsNoOp(const Stmt& stmt) {
    if (!stmt.defined()) {
      return true;
    }
    if (const auto* eval = stmt.as<tir::EvaluateNode>()) {
      if (const auto* imm = eval->value.as<IntImmNode>()) {
        return imm->value == 0;
      }
    }
    return false;
  }

  Stmt VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == "blackhole.segment_kind") {
      if (const auto* kind = op->value.as<StringImmNode>()) {
        if (kind->value == segment_kind_) {
          return this->VisitStmt(op->body);
        }
      }
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    return tir::StmtMutator::VisitStmt_(op);
  }

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
      if (retain_unmarked_stmts_) {
        seq.push_back(this->VisitStmt(stmt));
      }
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

  Stmt VisitStmt_(const tir::AllocateNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    if (IsNoOp(body)) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (!tir::UsesVar(body, [buffer_var = op->buffer_var.get()](const tir::VarNode* var) {
          return var == buffer_var;
        })) {
      return body;
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return tir::Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body,
                         op->annotations, op->span);
  }

  Stmt VisitStmt_(const tir::ForNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    if (IsNoOp(body)) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return tir::For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                    op->annotations, std::nullopt, op->span);
  }

  Stmt VisitStmt_(const tir::IfThenElseNode* op) final {
    Stmt then_case = this->VisitStmt(op->then_case);
    Stmt else_case = op->else_case.defined() ? this->VisitStmt(op->else_case.value()) : Stmt();
    const bool then_is_noop = IsNoOp(then_case);
    const bool else_is_noop = !else_case.defined() || IsNoOp(else_case);
    if (then_is_noop && else_is_noop) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (then_case.same_as(op->then_case) &&
        ((!op->else_case.defined() && !else_case.defined()) ||
         (op->else_case.defined() && else_case.same_as(op->else_case.value())))) {
      return GetRef<Stmt>(op);
    }
    return tir::IfThenElse(op->condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const tir::LetStmtNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    if (IsNoOp(body)) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return tir::LetStmt(op->var, op->value, body, op->span);
  }

 private:
  std::string segment_kind_;
  bool retain_unmarked_stmts_{false};
};

static ffi::Array<ffi::Any> EncodeRuntimeArgs(const std::vector<KernelArgSpec>& runtime_args) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& arg : runtime_args) {
    ffi::Map<ffi::String, ffi::Any> arg_info;
    arg_info.Set("name", ffi::String(arg.name));
    arg_info.Set("kind", ffi::String(arg.kind));
    arg_info.Set("dtype", ffi::String(arg.dtype));
    arg_info.Set("identity", ffi::String(arg.identity));
    if (!arg.buffer.empty()) {
      arg_info.Set("buffer", ffi::String(arg.buffer));
    }
    if (arg.has_core_coord) {
      arg_info.Set("core_x", Integer(static_cast<int>(arg.core_x)));
      arg_info.Set("core_y", Integer(static_cast<int>(arg.core_y)));
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
    if (accessor.transport_page_size_bytes != 0) {
      accessor_info.Set("transport_page_size",
                        Integer(static_cast<int>(accessor.transport_page_size_bytes)));
    }
    accessor_info.Set("layout", ffi::String(accessor.layout));
    accessor_info.Set("memory_space", ffi::String(accessor.memory_space));
    if (!accessor.host_axis_order.empty()) {
      ffi::Array<ffi::Any> axis_order;
      for (int64_t axis : accessor.host_axis_order) {
        axis_order.push_back(Integer(axis));
      }
      accessor_info.Set("host_axis_order", axis_order);
    }
    if (accessor.transpose_2d) {
      accessor_info.Set("transpose_2d", Bool(true));
    }
    encoded.push_back(accessor_info);
  }
  return encoded;
}

static ffi::Array<ffi::Any> EncodeCompileTimeArgSpecs(
    const std::vector<CompileTimeArgSpec>& compile_time_arg_specs) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& spec : compile_time_arg_specs) {
    ffi::Map<ffi::String, ffi::Any> spec_info;
    spec_info.Set("name", ffi::String(spec.name));
    spec_info.Set("kind", ffi::String(spec.kind));
    spec_info.Set("dtype", ffi::String(spec.dtype));
    spec_info.Set("offset", Integer(static_cast<int>(spec.offset)));
    spec_info.Set("count", Integer(static_cast<int>(spec.count)));
    if (!spec.buffer.empty()) {
      spec_info.Set("buffer", ffi::String(spec.buffer));
    }
    if (!spec.segment_role.empty()) {
      spec_info.Set("segment_role", ffi::String(spec.segment_role));
    }
    if (!spec.values.empty()) {
      ffi::Array<ffi::Any> values;
      for (uint32_t value : spec.values) {
        values.push_back(Integer(static_cast<int>(value)));
      }
      spec_info.Set("values", values);
    }
    if (spec.args_config_bits != 0) {
      spec_info.Set("args_config_bits", Integer(static_cast<int>(spec.args_config_bits)));
    }
    if (spec.transport_page_size_bytes != 0) {
      spec_info.Set("transport_page_size",
                    Integer(static_cast<int>(spec.transport_page_size_bytes)));
    }
    if (!spec.layout.empty()) {
      spec_info.Set("layout", ffi::String(spec.layout));
    }
    if (!spec.memory_space.empty()) {
      spec_info.Set("memory_space", ffi::String(spec.memory_space));
    }
    encoded.push_back(spec_info);
  }
  return encoded;
}

static ffi::Array<ffi::Any> EncodePerWorkArgSpecs(
    const std::vector<PerWorkArgSpec>& per_work_arg_specs) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& spec : per_work_arg_specs) {
    ffi::Map<ffi::String, ffi::Any> spec_info;
    spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kArgKind, ffi::String(spec.arg_kind));
    if (!spec.arg_identity.empty()) {
      spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kArgIdentity, ffi::String(spec.arg_identity));
    }
    if (!spec.buffer.empty()) {
      spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kBuffer, ffi::String(spec.buffer));
    }
    spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kValueKind, ffi::String(spec.value_kind));
    if (spec.value_kind == ::tvm::tl::blackhole_runtime_arg_schema::kValueConstant) {
      spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue,
                    Integer(static_cast<int>(spec.constant_value)));
    }
    encoded.push_back(spec_info);
  }
  return encoded;
}

static ffi::Array<ffi::Any> EncodeSemaphoreBindings(
    const std::vector<SemaphoreBindingSpec>& bindings) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& binding : bindings) {
    ffi::Map<ffi::String, ffi::Any> binding_info;
    binding_info.Set("name", ffi::String(binding.name));
    binding_info.Set("semaphore_id", Integer(static_cast<int>(binding.semaphore_id)));
    binding_info.Set("arg_kind", ffi::String(binding.arg_kind));
    encoded.push_back(binding_info);
  }
  return encoded;
}

static bool IsInputBufferArgKind(const std::string& kind) {
  return kind == "input_buffer_addr32" || kind == "input_buffer_addr";
}

static bool IsOutputBufferArgKind(const std::string& kind) {
  return kind == "output_buffer_addr32" || kind == "output_buffer_addr";
}

static std::string NormalizeBufferBindingName(std::string name) {
  constexpr const char* kHandleSuffix = "_handle";
  if (name.size() > std::strlen(kHandleSuffix) &&
      name.compare(name.size() - std::strlen(kHandleSuffix), std::strlen(kHandleSuffix),
                   kHandleSuffix) == 0) {
    name.resize(name.size() - std::strlen(kHandleSuffix));
  }
  return name;
}

static std::string ResolveBufferRole(const ExecutableSpec& spec, const std::string& buffer_name) {
  auto check_args = [&](const std::vector<KernelArgSpec>& args, bool output) {
    return std::any_of(args.begin(), args.end(), [&](const KernelArgSpec& arg) {
      if (arg.buffer != buffer_name) {
        return false;
      }
      return output ? IsOutputBufferArgKind(arg.kind) : IsInputBufferArgKind(arg.kind);
    });
  };

  if (check_args(spec.runtime_args, /*output=*/true)) {
    return "output";
  }
  if (check_args(spec.runtime_args, /*output=*/false)) {
    return "input";
  }
  for (const auto& kernel : spec.kernels) {
    if (check_args(kernel.runtime_args, /*output=*/true) ||
        check_args(kernel.common_runtime_args, /*output=*/true)) {
      return "output";
    }
    if (check_args(kernel.runtime_args, /*output=*/false) ||
        check_args(kernel.common_runtime_args, /*output=*/false)) {
      return "input";
    }
  }
  return "";
}

static uint32_t ChooseBufferMaterializationPageSize(const ExecutableSpec& spec,
                                                    const std::string& buffer_name) {
  uint32_t inferred_page_size = 0;
  auto record_page_size = [&](uint32_t candidate_page_size, const char* source_name) {
    if (candidate_page_size == 0) {
      return;
    }
    if (inferred_page_size == 0) {
      inferred_page_size = candidate_page_size;
      return;
    }
    ICHECK_EQ(inferred_page_size, candidate_page_size)
        << "Blackhole buffer materialization requires a single transport page size per buffer; "
        << buffer_name << " used both " << inferred_page_size << " and " << candidate_page_size
        << " from " << source_name;
  };
  for (const auto& kernel : spec.kernels) {
    for (const auto& accessor : kernel.accessors) {
      if (accessor.buffer != buffer_name) {
        continue;
      }
      record_page_size(accessor.transport_page_size_bytes, "accessor");
    }
    for (const auto& compile_time_arg_spec : kernel.compile_time_arg_specs) {
      if (compile_time_arg_spec.buffer != buffer_name) {
        continue;
      }
      record_page_size(compile_time_arg_spec.transport_page_size_bytes,
                       "compile_time_arg_spec");
    }
  }
  if (inferred_page_size != 0) {
    return inferred_page_size;
  }

  const std::string role = ResolveBufferRole(spec, buffer_name);
  for (const auto& cb : spec.cb_configs) {
    if (cb.role == role) {
      return cb.page_size_bytes;
    }
  }
  if (!spec.cb_configs.empty()) {
    return spec.cb_configs.front().page_size_bytes;
  }
  return 2048;
}

static bool KernelArgsContainKind(const std::vector<KernelArgSpec>& args,
                                  std::string_view kind) {
  return std::any_of(args.begin(), args.end(), [&](const KernelArgSpec& arg) {
    return arg.kind == kind;
  });
}

static bool PerWorkArgSpecsContainKind(const std::vector<PerWorkArgSpec>& specs,
                                       std::string_view kind) {
  return std::any_of(specs.begin(), specs.end(), [&](const PerWorkArgSpec& spec) {
    return spec.arg_kind == kind;
  });
}

static bool RuntimeArgKindRequiresExplicitPerWorkBinding(std::string_view kind) {
  return kind == "a_tile_start_id" || kind == "a_tile_num_tiles" ||
         kind == "a_tile_stride" || kind == "b_tile_start_id" ||
         kind == "b_tile_num_tiles" || kind == "b_tile_stride" ||
         kind == "output_tile_start_id" || kind == "output_tile_num_tiles" ||
         kind == "output_tile_stride" || kind == "k_tile_start_id" ||
         kind == "num_k_tiles";
}

static void ValidateKernelExplicitPerWorkBindingSchema(const CorePlan& core_plan,
                                                       const KernelSpec& kernel,
                                                       const std::string& func_name) {
  if (GetTotalLogicalWorkItems(core_plan) <= 1) {
    return;
  }
  for (const auto& arg : kernel.runtime_args) {
    if (!RuntimeArgKindRequiresExplicitPerWorkBinding(arg.kind)) {
      continue;
    }
    ICHECK(PerWorkArgSpecsContainKind(kernel.per_work_arg_specs, arg.kind))
        << "Blackhole build requires explicit per-work arg binding for runtime arg kind '"
        << arg.kind << "' on kernel " << kernel.name << " of " << func_name
        << "; runtime/codegen must not recover block/tile semantics from work_linear_id or "
        << "top-level TTProgram payload fallback";
  }
}

static void ValidateKernelRuntimeArgSchema(const KernelSpec& kernel,
                                           const std::string& func_name) {
  const bool is_copy_or_dataflow_kernel =
      kernel.kind == "fused_dataflow" || kernel.kind == "reader" || kernel.kind == "writer" ||
      !kernel.accessors.empty();
  if (!is_copy_or_dataflow_kernel) {
    return;
  }
  ICHECK(!kernel.runtime_args.empty())
      << "Blackhole runtime arg schema is required for copy/dataflow kernels; "
      << "TTProgram segment runtime args are missing for kernel " << kernel.name
      << " of " << func_name;
}

static const KernelArgSpec* FindKernelArgSpecByName(const KernelSpec& kernel,
                                                    const std::string& name) {
  auto find_in_args = [&](const std::vector<KernelArgSpec>& args) -> const KernelArgSpec* {
    for (const auto& arg : args) {
      if (arg.name == name) {
        return &arg;
      }
    }
    return nullptr;
  };
  if (const auto* arg = find_in_args(kernel.runtime_args)) {
    return arg;
  }
  return find_in_args(kernel.common_runtime_args);
}

static const SemaphoreBindingSpec* FindSemaphoreBindingSpec(const KernelSpec& kernel,
                                                            const std::string& name,
                                                            const std::string& arg_kind) {
  for (const auto& binding : kernel.semaphore_bindings) {
    if (binding.name == name && binding.arg_kind == arg_kind) {
      return &binding;
    }
  }
  return nullptr;
}

static bool KernelHasRemoteCoreDescriptorIdentity(const KernelSpec& kernel,
                                                  const std::string& identity) {
  return std::any_of(kernel.remote_core_descriptors.begin(), kernel.remote_core_descriptors.end(),
                     [&](const RemoteCoreDescriptorSpec& descriptor) {
                       return descriptor.identity == identity;
                     });
}

class CommunicationBuiltinSchemaValidator final : public tir::StmtExprVisitor {
 public:
  CommunicationBuiltinSchemaValidator(const tir::PrimFunc& func, const KernelSpec& kernel,
                                      const std::unordered_set<uint32_t>& planned_semaphore_ids)
      : func_name_(kernel.name),
        kernel_(kernel),
        planned_semaphore_ids_(planned_semaphore_ids) {
    VisitStmt(func->body);
    if (saw_semaphore_builtin_ && !saw_semaphore_source_) {
      ICHECK(false)
          << "Blackhole communication protocol for kernel " << func_name_
          << " requires semaphore builtins to consume explicit owner truth via "
             "get_semaphore(planned_id) or get_semaphore(runtime_arg_u32(bound_semaphore)); "
             "raw semaphore address recovery is not admitted";
    }
  }

 private:
  using tir::StmtExprVisitor::VisitExpr_;
  using tir::StmtExprVisitor::VisitStmt_;

  const tir::CallNode* ResolveCall(const PrimExpr& expr) const {
    PrimExpr resolved = ResolveExpr(expr);
    return resolved.as<tir::CallNode>();
  }

  PrimExpr ResolveExpr(PrimExpr expr) const {
    for (int depth = 0; depth < 16; ++depth) {
      if (const auto* var = expr.as<tir::VarNode>()) {
        auto it = let_bindings_.find(var);
        if (it == let_bindings_.end()) {
          return expr;
        }
        expr = it->second;
        continue;
      }
      if (const auto* cast = expr.as<tir::CastNode>()) {
        expr = cast->value;
        continue;
      }
      return expr;
    }
    return expr;
  }

  static const tir::StringImmNode* ResolveRuntimeArgNameLiteral(const tir::CallNode* call) {
    if (call == nullptr || !call->op.defined()) {
      return nullptr;
    }
    const auto* op = call->op.as<OpNode>();
    if (op == nullptr || op->name != "tl.blackhole.runtime_arg_u32" || call->args.size() != 1) {
      return nullptr;
    }
    return call->args[0].as<tir::StringImmNode>();
  }

  void ValidateGetSemaphoreCall(const tir::CallNode* call) {
    ICHECK_EQ(call->args.size(), 1U)
        << "tl.blackhole.get_semaphore expects exactly one argument in kernel " << func_name_;
    const PrimExpr semaphore_expr = ResolveExpr(call->args[0]);
    if (const auto* imm = semaphore_expr.as<tir::IntImmNode>()) {
      const uint32_t semaphore_id = static_cast<uint32_t>(imm->value);
      ICHECK(planned_semaphore_ids_.count(semaphore_id))
          << "Blackhole communication protocol for kernel " << func_name_
          << " requires get_semaphore(" << semaphore_id
          << ") to reference a planned semaphore id";
      saw_semaphore_source_ = true;
      return;
    }

    const auto* runtime_arg_call = ResolveCall(semaphore_expr);
    const auto* arg_name = ResolveRuntimeArgNameLiteral(runtime_arg_call);
    ICHECK(arg_name != nullptr)
        << "Blackhole communication protocol for kernel " << func_name_
        << " requires get_semaphore to consume either a literal planned semaphore id or "
           "runtime_arg_u32(bound_semaphore)";
    const auto* arg_spec = FindKernelArgSpecByName(kernel_, arg_name->value);
    ICHECK(arg_spec != nullptr)
        << "Blackhole communication protocol for kernel " << func_name_
        << " references unknown runtime arg " << arg_name->value << " in get_semaphore";
    ICHECK(arg_spec->kind == "semaphore_id_u32")
        << "Blackhole communication protocol for kernel " << func_name_
        << " requires get_semaphore(runtime_arg_u32(" << arg_name->value
        << ")) to bind a semaphore_id_u32 runtime arg";
    const auto* binding = FindSemaphoreBindingSpec(kernel_, arg_spec->name, arg_spec->kind);
    ICHECK(binding != nullptr)
        << "Blackhole communication protocol for kernel " << func_name_
        << " requires semaphore_id_u32 runtime arg " << arg_spec->name
        << " to carry an explicit semaphore binding";
    ICHECK(planned_semaphore_ids_.count(binding->semaphore_id))
        << "Blackhole communication protocol for kernel " << func_name_
        << " requires bound semaphore " << binding->name
        << " to reference a planned semaphore id";
    saw_semaphore_source_ = true;
  }

  void ValidateRemoteCoordPair(const tir::CallNode* call, size_t x_index, size_t y_index,
                               const char* builtin_name) {
    ICHECK_GT(call->args.size(), std::max(x_index, y_index))
        << "Blackhole communication protocol for kernel " << func_name_ << " expects "
        << builtin_name << " to carry remote_noc_x/y coordinates";
    const auto* x_call = ResolveCall(call->args[x_index]);
    const auto* y_call = ResolveCall(call->args[y_index]);
    const auto* x_name = ResolveRuntimeArgNameLiteral(x_call);
    const auto* y_name = ResolveRuntimeArgNameLiteral(y_call);
    ICHECK(x_name != nullptr && y_name != nullptr)
        << "Blackhole communication routing for kernel " << func_name_ << " requires "
        << builtin_name
        << " remote coordinates to come from logical_core_noc runtime args, not literal or "
           "body-recovered coordinates";
    const auto* x_arg = FindKernelArgSpecByName(kernel_, x_name->value);
    const auto* y_arg = FindKernelArgSpecByName(kernel_, y_name->value);
    ICHECK(x_arg != nullptr && y_arg != nullptr)
        << "Blackhole communication routing for kernel " << func_name_
        << " references unknown logical_core_noc runtime arg in " << builtin_name;
    ICHECK(x_arg->kind == "logical_core_noc_x")
        << "Blackhole communication routing for kernel " << func_name_ << " requires "
        << builtin_name << " x-coordinate to bind logical_core_noc_x";
    ICHECK(y_arg->kind == "logical_core_noc_y")
        << "Blackhole communication routing for kernel " << func_name_ << " requires "
        << builtin_name << " y-coordinate to bind logical_core_noc_y";
    ICHECK(!x_arg->identity.empty() && x_arg->identity == y_arg->identity)
        << "Blackhole communication routing for kernel " << func_name_ << " requires "
        << builtin_name << " logical_core_noc_x/y to reference one remote endpoint identity";
    ICHECK(KernelHasRemoteCoreDescriptorIdentity(kernel_, x_arg->identity))
        << "Blackhole communication routing for kernel " << func_name_
        << " requires remote endpoint " << x_arg->identity
        << " to be materialized in KernelSpec.remote_core_descriptors";
  }

  void VisitStmt_(const tir::LetStmtNode* op) final {
    VisitExpr(op->value);
    auto it = let_bindings_.find(op->var.get());
    const bool had_binding = it != let_bindings_.end();
    PrimExpr old_binding;
    if (had_binding) {
      old_binding = it->second;
    }
    let_bindings_[op->var.get()] = op->value;
    VisitStmt(op->body);
    if (had_binding) {
      let_bindings_[op->var.get()] = old_binding;
    } else {
      let_bindings_.erase(op->var.get());
    }
  }

  void VisitExpr_(const tir::LetNode* op) final {
    VisitExpr(op->value);
    auto it = let_bindings_.find(op->var.get());
    const bool had_binding = it != let_bindings_.end();
    PrimExpr old_binding;
    if (had_binding) {
      old_binding = it->second;
    }
    let_bindings_[op->var.get()] = op->value;
    VisitExpr(op->body);
    if (had_binding) {
      let_bindings_[op->var.get()] = old_binding;
    } else {
      let_bindings_.erase(op->var.get());
    }
  }

  void VisitExpr_(const tir::CallNode* op) final {
    const auto* builtin = op->op.as<OpNode>();
    if (builtin != nullptr) {
      const std::string& builtin_name = builtin->name;
      if (builtin_name == "tl.blackhole.get_semaphore") {
        saw_semaphore_builtin_ = true;
        ValidateGetSemaphoreCall(op);
      } else if (builtin_name == "tl.blackhole.semaphore_wait" ||
                 builtin_name == "tl.blackhole.semaphore_set") {
        saw_semaphore_builtin_ = true;
      } else if (builtin_name == "tl.blackhole.semaphore_inc_remote") {
        saw_semaphore_builtin_ = true;
        ValidateRemoteCoordPair(op, 1, 2, "semaphore_inc_remote");
      } else if (builtin_name == "tl.blackhole.semaphore_set_remote") {
        saw_semaphore_builtin_ = true;
        ValidateRemoteCoordPair(op, 1, 2, "semaphore_set_remote");
      }
    }
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  std::string func_name_;
  const KernelSpec& kernel_;
  const std::unordered_set<uint32_t>& planned_semaphore_ids_;
  std::unordered_map<const tir::VarNode*, PrimExpr> let_bindings_;
  bool saw_semaphore_builtin_{false};
  bool saw_semaphore_source_{false};
};

static void ValidateKernelCommunicationProtocolSchema(const tir::PrimFunc& func,
                                                      const KernelSpec& kernel,
                                                      const ExecutableSpec& spec) {
  std::unordered_set<uint32_t> planned_semaphore_ids;
  for (const auto& semaphore : spec.semaphores) {
    planned_semaphore_ids.insert(semaphore.id);
  }
  CommunicationBuiltinSchemaValidator validator(func, kernel, planned_semaphore_ids);
  (void)validator;
}

static bool MaterializationNeedsExplicitPerWorkAccessDescriptor(
    const BufferMaterializationSpec& materialization,
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name) {
  auto it = buffer_info_by_name.find(materialization.buffer);
  if (it != buffer_info_by_name.end() && it->second.shape.size() > 2) {
    return true;
  }
  if (materialization.host_axis_order.size() > 2) {
    return true;
  }
  if (materialization.transpose_2d && materialization.host_axis_order.size() > 2) {
    return true;
  }
  return false;
}

static void AppendDirectRuntimeUnsupportedReason(ExecutableSpec* spec,
                                                 const std::string& reason) {
  ICHECK(spec != nullptr);
  if (std::find(spec->direct_runtime_unsupported_reasons.begin(),
                spec->direct_runtime_unsupported_reasons.end(),
                reason) != spec->direct_runtime_unsupported_reasons.end()) {
    return;
  }
  spec->direct_runtime_unsupported_reasons.push_back(reason);
}

static void EnforceExplicitPerWorkAccessDescriptorGate(
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name,
    ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  const uint32_t total_logical_work_items = GetTotalLogicalWorkItems(spec->core_plan);
  if (total_logical_work_items <= 1) {
    return;
  }

  const bool has_multidim_materialized_buffer = std::any_of(
      spec->buffer_materializations.begin(), spec->buffer_materializations.end(),
      [&](const BufferMaterializationSpec& materialization) {
        return MaterializationNeedsExplicitPerWorkAccessDescriptor(materialization,
                                                                   buffer_info_by_name);
      });
  if (!has_multidim_materialized_buffer) {
    return;
  }

  auto kernel_is_missing_explicit_access_descriptor =
      [](const std::vector<KernelArgSpec>& runtime_args,
         const std::vector<PerWorkArgSpec>& per_work_arg_specs) {
        const bool has_reader_tile_coords =
            KernelArgsContainKind(runtime_args, "a_tile_start_id") &&
            KernelArgsContainKind(runtime_args, "b_tile_start_id");
        const bool has_writer_tile_coord =
            KernelArgsContainKind(runtime_args, "output_tile_start_id");
        if (!(has_reader_tile_coords || has_writer_tile_coord)) {
          return false;
        }

        for (const auto& arg : runtime_args) {
          if (arg.kind == "a_tile_start_id" || arg.kind == "a_tile_num_tiles" ||
              arg.kind == "a_tile_stride" || arg.kind == "b_tile_start_id" ||
              arg.kind == "b_tile_num_tiles" || arg.kind == "b_tile_stride" ||
              arg.kind == "output_tile_start_id" || arg.kind == "output_tile_num_tiles" ||
              arg.kind == "output_tile_stride" || arg.kind == "k_tile_start_id" ||
              arg.kind == "num_k_tiles") {
            if (!PerWorkArgSpecsContainKind(per_work_arg_specs, arg.kind)) {
              return true;
            }
          }
        }
        return false;
      };

  bool missing_explicit_descriptor = false;
  if (!spec->kernels.empty()) {
    for (const auto& kernel : spec->kernels) {
      if (kernel_is_missing_explicit_access_descriptor(kernel.runtime_args,
                                                       kernel.per_work_arg_specs)) {
        missing_explicit_descriptor = true;
        break;
      }
    }
  } else {
    missing_explicit_descriptor = kernel_is_missing_explicit_access_descriptor(
        spec->runtime_args, spec->per_work_arg_specs);
  }

  if (!missing_explicit_descriptor) {
    return;
  }

  AppendDirectRuntimeUnsupportedReason(
      spec,
      "missing explicit per-work access descriptor; direct runtime must not "
      "reconstruct tile access from work_linear_id when total logical work "
      "items > 1 for materialized buffers with rank > 2");
}

static void EnforceTypedDstCbAccumulationGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  (void)spec;
}

static void EnforceExplicitBufferRoleSchemaGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  size_t n_buffer_args = 0;
  std::unordered_set<std::string> expected_buffer_names;
  for (size_t i = 0; i < spec->tvm_is_buffer_arg.size(); ++i) {
    if (!spec->tvm_is_buffer_arg[i]) {
      continue;
    }
    ++n_buffer_args;
    if (i < spec->tvm_arg_names.size() && !spec->tvm_arg_names[i].empty()) {
      expected_buffer_names.insert(NormalizeBufferBindingName(spec->tvm_arg_names[i]));
    }
  }
  if (n_buffer_args == 0) {
    return;
  }

  bool missing_buffer_name = false;
  std::unordered_set<std::string> bound_buffer_names;
  auto record_args = [&](const std::vector<KernelArgSpec>& args) {
    for (const auto& arg : args) {
      if (!IsInputBufferArgKind(arg.kind) && !IsOutputBufferArgKind(arg.kind)) {
        continue;
      }
      if (arg.buffer.empty()) {
        missing_buffer_name = true;
        continue;
      }
      bound_buffer_names.insert(NormalizeBufferBindingName(arg.buffer));
    }
  };
  record_args(spec->runtime_args);
  record_args(spec->common_runtime_args);

  if (missing_buffer_name || bound_buffer_names.empty()) {
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "missing explicit buffer role schema; direct runtime requires named "
        "input/output buffer bindings and must not recover output positionally");
    return;
  }

  for (const auto& buffer_name : expected_buffer_names) {
    if (!bound_buffer_names.count(buffer_name)) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "missing explicit buffer role schema; direct runtime requires named "
          "input/output buffer bindings and must not recover output positionally");
      return;
    }
  }
}

static std::vector<int64_t> ChooseBufferMaterializationAxisOrder(
    const ExecutableSpec& spec, const BufferMaterializationSpec& materialization,
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name) {
  auto info_it = buffer_info_by_name.find(materialization.buffer);
  if (info_it == buffer_info_by_name.end()) {
    return {};
  }
  const StaticBufferInfo& info = info_it->second;
  if (materialization.layout != "interleaved" || materialization.memory_space != "dram" ||
      info.shape.size() < 2 || info.dtype.lanes != 1 || info.dtype.bits == 0) {
    return {};
  }

  constexpr uint32_t kBlackholeTileCols = 32;
  const uint32_t element_size_bytes = static_cast<uint32_t>((info.dtype.bits + 7) / 8);
  if (element_size_bytes == 0 || materialization.transport_page_size_bytes == 0 ||
      materialization.transport_page_size_bytes % element_size_bytes != 0) {
    return {};
  }

  const uint32_t tile_elements = materialization.transport_page_size_bytes / element_size_bytes;
  if (tile_elements == 0 || tile_elements % kBlackholeTileCols != 0) {
    return {};
  }
  const uint32_t tile_rows = tile_elements / kBlackholeTileCols;
  const int64_t total_rows = ShapeProduct(info.shape, 0, info.shape.size() - 1);
  const int64_t cols = info.shape.back();
  if (tile_rows == 0 || total_rows <= 0 || cols <= 0 || cols % kBlackholeTileCols != 0 ||
      total_rows % tile_rows != 0) {
    return {};
  }
  return InferStaticWorkMajorAxisOrder(info.shape, GetTotalLogicalWorkItems(spec.core_plan),
                                       tile_rows);
}

static void PopulateBufferMaterializationSpecs(
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name,
    ExecutableSpec* spec) {
  std::unordered_map<std::string, BufferMaterializationSpec> by_buffer;
  std::vector<std::string> order;

  auto register_buffer = [&](const std::string& buffer_name, const std::string& layout,
                             const std::string& memory_space,
                             const std::vector<int64_t>& host_axis_order = std::vector<int64_t>{},
                             bool transpose_2d = false) {
    if (buffer_name.empty()) {
      return;
    }
    auto [it, inserted] = by_buffer.emplace(buffer_name, BufferMaterializationSpec{});
    auto& materialization = it->second;
    if (inserted) {
      materialization.buffer = buffer_name;
      materialization.materialization_kind = "replicated";
      materialization.layout = layout;
      materialization.memory_space = memory_space;
      materialization.host_axis_order = host_axis_order;
      materialization.transpose_2d = transpose_2d;
      order.push_back(buffer_name);
      return;
    }
    ICHECK_EQ(materialization.layout, layout)
        << "Blackhole buffer materialization requires a single layout per buffer; "
        << buffer_name << " used both " << materialization.layout << " and " << layout;
    ICHECK_EQ(materialization.memory_space, memory_space)
        << "Blackhole buffer materialization requires a single memory_space per buffer; "
        << buffer_name << " used both " << materialization.memory_space << " and " << memory_space;
    if (!host_axis_order.empty()) {
      ICHECK(materialization.host_axis_order.empty() ||
             materialization.host_axis_order == host_axis_order)
          << "Blackhole buffer materialization requires a single host_axis_order per buffer; "
          << buffer_name;
      if (materialization.host_axis_order.empty()) {
        materialization.host_axis_order = host_axis_order;
      }
    }
    if (transpose_2d) {
      materialization.transpose_2d = true;
    }
  };

  for (const auto& kernel : spec->kernels) {
    for (const auto& accessor : kernel.accessors) {
      if (accessor.buffer.empty()) {
        continue;
      }
      register_buffer(accessor.buffer, accessor.layout, accessor.memory_space,
                      accessor.host_axis_order, accessor.transpose_2d);
    }
    for (const auto& compile_time_arg_spec : kernel.compile_time_arg_specs) {
      if (compile_time_arg_spec.buffer.empty() || compile_time_arg_spec.layout.empty() ||
          compile_time_arg_spec.memory_space.empty()) {
        continue;
      }
      register_buffer(compile_time_arg_spec.buffer, compile_time_arg_spec.layout,
                      compile_time_arg_spec.memory_space);
    }
  }

  spec->buffer_materializations.clear();
  spec->buffer_materializations.reserve(order.size());
  for (const auto& buffer_name : order) {
    auto materialization = by_buffer.at(buffer_name);
    materialization.transport_page_size_bytes =
        ChooseBufferMaterializationPageSize(*spec, buffer_name);
    if (materialization.host_axis_order.empty()) {
      materialization.host_axis_order =
          ChooseBufferMaterializationAxisOrder(*spec, materialization, buffer_info_by_name);
    }
    spec->buffer_materializations.push_back(std::move(materialization));
  }
}

static tir::PrimFunc MakeSegmentPrimFunc(const tir::PrimFunc& f, const SegmentInfo& segment) {
  const bool retain_unmarked_stmts =
      segment.kind == "compute" || segment.core_type == "trisc";
  SegmentBodyExtractor extractor(segment.kind, retain_unmarked_stmts);
  tir::PrimFunc segment_func = f;
  segment_func.CopyOnWrite()->body = extractor(f->body);

  const ffi::Map<ffi::String, ffi::Any> original_executable =
      tl::tt_program_projection::RequireBlackholeExecutableProjection(
          f, "Blackhole segment materialization");
  const ffi::String kernel_name =
      segment.name.empty() ? ffi::String(segment.kind) : ffi::String(segment.name);
  const ffi::String kernel_kind =
      segment.kind.empty() ? ffi::String("fused_dataflow") : ffi::String(segment.kind);
  const ffi::String kernel_core_type =
      segment.core_type.empty() ? ffi::String("brisc") : ffi::String(segment.core_type);
  const ffi::Array<ffi::Any> encoded_runtime_args = EncodeRuntimeArgs(segment.runtime_args);
  const ffi::Array<ffi::Any> encoded_common_runtime_args =
      EncodeRuntimeArgs(segment.common_runtime_args);
  const ffi::Array<ffi::Any> encoded_accessors = EncodeAccessors(segment.accessors);
  const ffi::Array<ffi::Any> encoded_compile_time_arg_specs =
      EncodeCompileTimeArgSpecs(segment.compile_time_arg_specs);
  const ffi::Array<ffi::Any> encoded_per_work_arg_specs =
      EncodePerWorkArgSpecs(segment.per_work_arg_specs);
  const ffi::Array<ffi::Any> encoded_semaphore_bindings =
      EncodeSemaphoreBindings(segment.semaphore_bindings);

  ffi::Map<ffi::String, ffi::Any> encoded_segment;
  encoded_segment.Set("name", kernel_name);
  encoded_segment.Set("kind", kernel_kind);
  encoded_segment.Set("core_type", kernel_core_type);
  if (!encoded_runtime_args.empty()) {
    encoded_segment.Set("runtime_args", encoded_runtime_args);
  }
  if (!encoded_common_runtime_args.empty()) {
    encoded_segment.Set("common_runtime_args", encoded_common_runtime_args);
  }
  if (!encoded_accessors.empty()) {
    encoded_segment.Set("accessors", encoded_accessors);
  }
  if (!encoded_compile_time_arg_specs.empty()) {
    encoded_segment.Set("compile_time_arg_specs", encoded_compile_time_arg_specs);
  }
  if (!encoded_per_work_arg_specs.empty()) {
    encoded_segment.Set(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs,
                        encoded_per_work_arg_specs);
  }
  if (!encoded_semaphore_bindings.empty()) {
    encoded_segment.Set("semaphore_bindings", encoded_semaphore_bindings);
  }
  if (segment.has_launch_spec) {
    ffi::Map<ffi::String, ffi::Any> launch_spec;
    launch_spec.Set("core_type", ffi::String(segment.launch_spec.core_type));
    launch_spec.Set("processor", ffi::String(segment.launch_spec.processor));
    launch_spec.Set("noc", ffi::String(segment.launch_spec.noc));
    encoded_segment.Set("launch_spec", launch_spec);
  }
  if (segment.has_compute_config) {
    ffi::Map<ffi::String, ffi::Any> compute_config;
    compute_config.Set("math_fidelity", ffi::String(segment.compute_config.math_fidelity));
    compute_config.Set("fp32_dest_acc_en", Bool(segment.compute_config.fp32_dest_acc_en));
    compute_config.Set("dst_full_sync_en", Bool(segment.compute_config.dst_full_sync_en));
    compute_config.Set("math_approx_mode", Bool(segment.compute_config.math_approx_mode));
    compute_config.Set("bfp8_pack_precise", Bool(segment.compute_config.bfp8_pack_precise));
    compute_config.Set("clear_accum", Bool(segment.compute_config.clear_accum));
    compute_config.Set("k_pack", Integer(static_cast<int>(segment.compute_config.k_pack)));
    compute_config.Set("wg_wait", Integer(segment.compute_config.wg_wait));
    compute_config.Set("policy_type", Integer(segment.compute_config.policy_type));
    compute_config.Set("policy_name", ffi::String(segment.compute_config.policy_name));
    encoded_segment.Set("compute_config", compute_config);
  }

  ffi::Map<ffi::String, ffi::Any> segment_executable;
  for (const auto& kv : original_executable) {
    if (kv.first ==
        ffi::String(tl::tt_program_projection::executable_key::kSegmentPlan)) {
      continue;
    }
    segment_executable.Set(kv.first, kv.second);
  }
  ffi::Array<ffi::Any> segment_plan;
  segment_plan.push_back(encoded_segment);
  segment_executable.Set(
      ffi::String(tl::tt_program_projection::executable_key::kSegmentPlan), segment_plan);

  ffi::Map<ffi::String, ffi::Any> attrs;
  static const std::unordered_set<std::string> kSyntheticProjectionAttrs = {
      tvm::tl::attr::kTLBlackholeExecutable,
  };
  if (f->attrs.defined()) {
    for (const auto& kv : f->attrs->dict) {
      if (kSyntheticProjectionAttrs.count(static_cast<std::string>(kv.first))) {
        continue;
      }
      attrs.Set(kv.first, kv.second);
    }
  }
  attrs.Set(tvm::tl::attr::kTLBlackholeExecutable, segment_executable);
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
                                             ExecutableSpec* spec) {
  spec->kernels.clear();
  std::vector<SegmentInfo> segments = ExtractSegmentPlan(f, spec);
  ICHECK(!segments.empty())
      << "Blackhole build requires non-empty TTProgram segment truth on device PrimFunc "
      << func_name;

  for (const SegmentInfo& segment : segments) {
    const bool whole_kernel_fused_dataflow =
        segments.size() == 1 && (segment.kind.empty() || segment.kind == "fused_dataflow");
    tir::PrimFunc segment_func =
        whole_kernel_fused_dataflow ? f : MakeSegmentPrimFunc(f, segment);
    KernelSpec kernel;
    kernel.name = func_name + "_" + (segment.name.empty() ? segment.kind : segment.name);
    kernel.kind = segment.kind;
    kernel.core_type = segment.core_type;
    kernel.runtime_args = ExtractRuntimeArgs(segment_func);
    kernel.common_runtime_args = ExtractCommonRuntimeArgs(segment_func);
    kernel.per_work_arg_specs = segment.per_work_arg_specs;
    kernel.remote_core_descriptors = segment.remote_core_descriptors.empty()
                                         ? ExtractRemoteCoreDescriptors(kernel.runtime_args)
                                         : segment.remote_core_descriptors;
    kernel.compile_time_arg_specs = segment.compile_time_arg_specs;
    kernel.has_launch_spec = segment.has_launch_spec;
    if (segment.has_launch_spec) {
      kernel.launch_spec = segment.launch_spec;
    }
    if ((kernel.core_type == "trisc" || kernel.kind == "compute") && spec->compute_contract.enabled &&
        spec->compute_contract.kind == "gemm") {
      kernel.has_compute_config = true;
      kernel.compute_config = ComputeConfigFromContract(spec->compute_contract);
    } else {
      kernel.has_compute_config = segment.has_compute_config;
      if (segment.has_compute_config) {
        kernel.compute_config = segment.compute_config;
      }
    }
    kernel.accessors = segment.accessors;
    kernel.semaphore_bindings = segment.semaphore_bindings;
    ValidateKernelRuntimeArgSchema(kernel, func_name);
    ValidateKernelExplicitPerWorkBindingSchema(spec->core_plan, kernel, func_name);
    ValidateKernelCommunicationProtocolSchema(segment_func, kernel, *spec);
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
    auto prim_target = f->GetAttr<Target>(tvm::attr::kTarget);
    const bool is_blackhole_prim_func =
        prim_target && prim_target.value()->kind->name == "blackhole";
    if (is_blackhole_prim_func && !IsBlackholeHostEntry(f)) {
      if (tl::tt_program_projection::GetBlackholeExecutableProjection(f).empty()) {
        ICHECK(false) << "Blackhole build requires tl.blackhole_executable on device PrimFunc "
                      << gvar->name_hint;
      }
    }
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (IsBlackholeDeviceKernel(f)) {
      device_kernel_symbols.insert(func_name);
      device_funcs.emplace(func_name, f);
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

  for (auto& kv : device_funcs) {
    auto spec_it = func_info_map.find(kv.first);
    if (spec_it == func_info_map.end()) {
      continue;
    }
    PopulateKernelSpecsForDeviceFunc(kv.second, kv.first, target, /*kernel_code_only=*/false,
                                     &spec_it->second);
    const auto buffer_info =
        CollectStaticBufferInfo(kv.second, CollectMaterializedBufferNames(spec_it->second));
    PopulateBufferMaterializationSpecs(buffer_info, &spec_it->second);
    EnforceExplicitPerWorkAccessDescriptorGate(buffer_info, &spec_it->second);
    EnforceTypedDstCbAccumulationGate(&spec_it->second);
    EnforceExplicitBufferRoleSchemaGate(&spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
      host_it->second.buffer_materializations = device_it->second.buffer_materializations;
      host_it->second.per_work_arg_specs = device_it->second.per_work_arg_specs;
      host_it->second.direct_runtime_unsupported_reasons =
          device_it->second.direct_runtime_unsupported_reasons;
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
    auto prim_target = f->GetAttr<Target>(tvm::attr::kTarget);
    const bool is_blackhole_prim_func =
        prim_target && prim_target.value()->kind->name == "blackhole";
    if (is_blackhole_prim_func && !IsBlackholeHostEntry(f)) {
      if (tl::tt_program_projection::GetBlackholeExecutableProjection(f).empty()) {
        ICHECK(false) << "Blackhole build requires tl.blackhole_executable on device PrimFunc "
                      << gvar->name_hint;
      }
    }
    const std::string func_name = GetPrimFuncName(gvar, f);
    if (IsBlackholeDeviceKernel(f)) {
      device_kernel_symbols.insert(func_name);
      device_funcs.emplace(func_name, f);
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
  for (auto& kv : device_funcs) {
    auto spec_it = func_info_map.find(kv.first);
    if (spec_it == func_info_map.end()) {
      continue;
    }
    PopulateKernelSpecsForDeviceFunc(kv.second, kv.first, target, /*kernel_code_only=*/true,
                                     &spec_it->second);
    const auto buffer_info =
        CollectStaticBufferInfo(kv.second, CollectMaterializedBufferNames(spec_it->second));
    PopulateBufferMaterializationSpecs(buffer_info, &spec_it->second);
    EnforceExplicitPerWorkAccessDescriptorGate(buffer_info, &spec_it->second);
    EnforceTypedDstCbAccumulationGate(&spec_it->second);
    EnforceExplicitBufferRoleSchemaGate(&spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
      host_it->second.buffer_materializations = device_it->second.buffer_materializations;
      host_it->second.per_work_arg_specs = device_it->second.per_work_arg_specs;
      host_it->second.direct_runtime_unsupported_reasons =
          device_it->second.direct_runtime_unsupported_reasons;
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
