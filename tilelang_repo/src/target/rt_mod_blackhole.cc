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
#include <cctype>
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
#include "../transform/common/companion_base.h"
#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace runtime {

using tvm::ffi::Map;
using tvm::ffi::String;
namespace buffer_materialization = tvm::tl::buffer_materialization;

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
    .add_attr_option<int64_t>("max_cb_count", 64)
    .add_attr_option<int64_t>("num_cbs", 64)     // 64 circular buffers per core
    .set_default_keys({"blackhole"});

static ffi::Map<ffi::String, ffi::Any> RequireMap(const ffi::Any& any,
                                                  const std::string& context) {
  auto map = any.as<ffi::Map<ffi::String, ffi::Any>>();
  ICHECK(map.has_value() && map.value().defined() && !map.value().empty())
      << context << " must be a non-empty map";
  return map.value();
}

static ffi::Array<ffi::Any> RequireExecutableArrayField(const tir::PrimFunc& f,
                                                        const char* consumer,
                                                        const char* key) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(f, consumer);
  auto value = executable.Get(String(key));
  ICHECK(value.has_value()) << consumer << " requires executable array field " << key;
  return Downcast<ffi::Array<ffi::Any>>(value.value());
}

static ffi::Map<ffi::String, ffi::Any> RequireExecutableMapField(const tir::PrimFunc& f,
                                                                 const char* consumer,
                                                                 const char* key) {
  auto executable = tl::tt_program_projection::RequireBlackholeExecutableProjection(f, consumer);
  auto value = executable.Get(String(key));
  ICHECK(value.has_value()) << consumer << " requires executable map field " << key;
  return RequireMap(value.value(), std::string(consumer) + "." + key);
}

/*!
 * \brief Extract CB configuration from PrimFunc attrs.
 * \param f The PrimFunc
 * \return Vector of CB configurations
 */
static std::vector<CBConfig> ExtractCBConfig(const tir::PrimFunc& f) {
  std::vector<CBConfig> cb_configs;
  auto cb_attr = RequireExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kCBConfigs);
  if (cb_attr.empty()) {
    return cb_configs;
  }

  for (const auto& item : cb_attr) {
    CBConfig config;
    auto cb_info = RequireMap(item, "Blackhole executable CB config");

    bool has_cb_id = false;
    bool has_num_pages = false;
    bool has_page_size = false;
    if (auto cb_id = cb_info.Get("cb_id")) {
      config.cb_id = Downcast<Integer>(cb_id.value()).IntValue();
      has_cb_id = true;
    }
    if (auto name = cb_info.Get("name")) {
      config.name = Downcast<String>(name.value());
    }
    if (auto role = cb_info.Get("role")) {
      config.role = Downcast<String>(role.value());
    }
    if (auto num_pages = cb_info.Get("num_pages")) {
      config.num_pages = Downcast<Integer>(num_pages.value()).IntValue();
      has_num_pages = true;
    }
    if (auto page_size = cb_info.Get("page_size")) {
      config.page_size_bytes = Downcast<Integer>(page_size.value()).IntValue();
      has_page_size = true;
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
    if (auto requirement_names = cb_info.Get("requirement_names")) {
      for (const auto& name : Downcast<ffi::Array<ffi::Any>>(requirement_names.value())) {
        config.requirement_names.push_back(Downcast<String>(name));
      }
    }
    if (auto requirement_indices = cb_info.Get("requirement_indices")) {
      for (const auto& index : Downcast<ffi::Array<ffi::Any>>(requirement_indices.value())) {
        config.requirement_indices.push_back(Downcast<Integer>(index).IntValue());
      }
    }

    ICHECK(has_cb_id) << "Blackhole executable CB config requires explicit cb_id";
    ICHECK(!config.name.empty())
        << "Blackhole executable CB config requires explicit name for cb_id=" << config.cb_id;
    ICHECK(!config.role.empty())
        << "Blackhole executable CB config requires explicit role for " << config.name;
    ICHECK(has_num_pages && config.num_pages > 0)
        << "Blackhole executable CB config requires explicit num_pages for " << config.name;
    ICHECK(has_page_size && config.page_size_bytes > 0)
        << "Blackhole executable CB config requires explicit page_size for " << config.name;
    ICHECK(!config.data_format.empty())
        << "Blackhole executable CB config requires explicit data_format for " << config.name;

    cb_configs.push_back(config);
  }

  return cb_configs;
}

static CorePlan ExtractCorePlan(const tir::PrimFunc& f) {
  CorePlan plan;
  auto core_plan = RequireExecutableMapField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kCorePlan);

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
      auto core_info = RequireMap(item, "Blackhole executable core_plan.physical_cores item");
      PhysicalCore core;
      auto x = core_info.Get("core_x");
      auto y = core_info.Get("core_y");
      ICHECK(x.has_value() && y.has_value())
          << "Blackhole executable core_plan.physical_cores item requires core_x/core_y";
      core.core_x = Downcast<Integer>(x.value()).IntValue();
      core.core_y = Downcast<Integer>(y.value()).IntValue();
      plan.physical_cores.push_back(core);
    }
  }

  if (auto v = core_plan.Get("work_packets")) {
    for (const auto& item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto packet_info = RequireMap(item, "Blackhole executable core_plan.work_packets item");
      WorkPacket packet;
      auto x = packet_info.Get("core_x");
      auto y = packet_info.Get("core_y");
      auto offset = packet_info.Get("work_offset");
      auto count = packet_info.Get("work_count");
      ICHECK(x.has_value() && y.has_value())
          << "Blackhole executable core_plan.work_packets item requires core_x/core_y";
      ICHECK(offset.has_value())
          << "Blackhole executable core_plan.work_packets item requires work_offset";
      ICHECK(count.has_value())
          << "Blackhole executable core_plan.work_packets item requires work_count";
      packet.core_x = Downcast<Integer>(x.value()).IntValue();
      packet.core_y = Downcast<Integer>(y.value()).IntValue();
      packet.work_offset = Downcast<Integer>(offset.value()).IntValue();
      packet.work_count = Downcast<Integer>(count.value()).IntValue();
      plan.work_packets.push_back(packet);
    }
  }

  ICHECK(!plan.physical_cores.empty())
      << "Blackhole executable core_plan requires physical_cores";
  ICHECK(!plan.work_packets.empty())
      << "Blackhole executable core_plan requires work_packets";
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
    auto semaphore_info = RequireMap(item, "Blackhole executable semaphore_plan item");

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
        auto range_info = RequireMap(range_any, "Blackhole executable semaphore core_range");

        auto parse_coord = [](const ffi::Optional<ffi::Any>& coord_any) {
          PhysicalCore coord;
          ICHECK(coord_any.has_value())
              << "Blackhole executable semaphore core_range requires start/end";
          auto coord_info =
              RequireMap(coord_any.value(), "Blackhole executable semaphore core_range coord");
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
      auto define = RequireMap(item, "Blackhole executable compute_config.define item");
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
      auto arg = RequireMap(item, "Blackhole executable compute_config.named_compile_args item");
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

static bool ExtractComputeOp(const ffi::Map<ffi::String, ffi::Any>& spec_info,
                             KernelComputeOpSpec* compute_op) {
  if (spec_info.empty()) {
    return false;
  }
  if (auto v = spec_info.Get("enabled")) {
    compute_op->enabled = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("kind")) {
    compute_op->kind = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("operation_name")) {
    compute_op->operation_name = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("a_buffer")) {
    compute_op->a_buffer = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("b_buffer")) {
    compute_op->b_buffer = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("c_buffer")) {
    compute_op->c_buffer = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("operand_bindings")) {
    for (const auto& binding_item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      auto binding_info =
          RequireMap(binding_item, "Blackhole executable compute_op.operand_bindings item");
      ComputeOperandBindingSpec binding;
      if (auto role = binding_info.Get("role")) {
        binding.role = Downcast<String>(role.value());
      }
      if (auto buffer = binding_info.Get("buffer")) {
        binding.buffer = Downcast<String>(buffer.value());
      }
      if (auto host_buffer = binding_info.Get("host_buffer")) {
        binding.host_buffer = Downcast<String>(host_buffer.value());
      }
      ICHECK(!binding.role.empty()) << "Blackhole compute operand binding requires role";
      ICHECK(!binding.buffer.empty())
          << "Blackhole compute operand binding for role " << binding.role
          << " requires device buffer";
      if (compute_op->kind == "gemm") {
        ICHECK(!binding.host_buffer.empty())
            << "Blackhole GEMM compute operand binding for role " << binding.role
            << " requires explicit host_buffer";
      }
      compute_op->operand_bindings.push_back(std::move(binding));
    }
  }
  if (auto v = spec_info.Get("M")) {
    compute_op->M = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("N")) {
    compute_op->N = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("K")) {
    compute_op->K = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("Mt")) {
    compute_op->Mt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("Nt")) {
    compute_op->Nt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("Kt")) {
    compute_op->Kt = static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("block_m_tiles")) {
    compute_op->block_m_tiles =
        static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("block_n_tiles")) {
    compute_op->block_n_tiles =
        static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("block_k_tiles")) {
    compute_op->block_k_tiles =
        static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("subblock_m_tiles")) {
    compute_op->subblock_m_tiles =
        static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("subblock_n_tiles")) {
    compute_op->subblock_n_tiles =
        static_cast<uint32_t>(Downcast<Integer>(v.value())->value);
  }
  if (auto v = spec_info.Get("transpose_A")) {
    compute_op->transpose_A = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("transpose_B")) {
    compute_op->transpose_B = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("a_tensor_dtype")) {
    compute_op->a_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("b_tensor_dtype")) {
    compute_op->b_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("c_tensor_dtype")) {
    compute_op->c_tensor_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("a_cb_dtype")) {
    compute_op->a_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("b_cb_dtype")) {
    compute_op->b_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("c_cb_dtype")) {
    compute_op->c_cb_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("accumulator_dtype")) {
    compute_op->accumulator_dtype = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("has_mbarrier")) {
    compute_op->has_mbarrier = Downcast<Bool>(v.value());
  }
  if (auto v = spec_info.Get("mbarrier_buffer")) {
    compute_op->mbarrier_buffer = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("mbarrier_scope")) {
    compute_op->mbarrier_scope = Downcast<String>(v.value());
  }
  if (auto v = spec_info.Get("mbarrier_index_exprs")) {
    for (const auto& expr : Downcast<ffi::Array<ffi::Any>>(v.value())) {
      compute_op->mbarrier_index_exprs.push_back(Downcast<String>(expr));
    }
  }
  if (compute_op->enabled && compute_op->kind == "gemm") {
    ICHECK_GT(compute_op->Mt, 0U) << "Blackhole GEMM compute_op requires explicit Mt";
    ICHECK_GT(compute_op->Nt, 0U) << "Blackhole GEMM compute_op requires explicit Nt";
    ICHECK_GT(compute_op->Kt, 0U) << "Blackhole GEMM compute_op requires explicit Kt";
    ICHECK_GT(compute_op->block_m_tiles, 0U)
        << "Blackhole GEMM compute_op requires explicit block_m_tiles";
    ICHECK_GT(compute_op->block_n_tiles, 0U)
        << "Blackhole GEMM compute_op requires explicit block_n_tiles";
    ICHECK_GT(compute_op->block_k_tiles, 0U)
        << "Blackhole GEMM compute_op requires explicit block_k_tiles";
    ICHECK_GT(compute_op->subblock_m_tiles, 0U)
        << "Blackhole GEMM compute_op requires explicit subblock_m_tiles";
    ICHECK_GT(compute_op->subblock_n_tiles, 0U)
        << "Blackhole GEMM compute_op requires explicit subblock_n_tiles";
  }
  return compute_op->enabled && !compute_op->kind.empty();
}

static ffi::Map<ffi::String, ffi::Any> EncodeKernelComputeOp(
    const KernelComputeOpSpec& compute_op) {
  ffi::Map<ffi::String, ffi::Any> item;
  item.Set("enabled", Bool(compute_op.enabled));
  item.Set("kind", ffi::String(compute_op.kind));
  item.Set("operation_name", ffi::String(compute_op.operation_name));
  item.Set("a_buffer", ffi::String(compute_op.a_buffer));
  item.Set("b_buffer", ffi::String(compute_op.b_buffer));
  item.Set("c_buffer", ffi::String(compute_op.c_buffer));
  ffi::Array<ffi::Any> operand_bindings;
  for (const auto& binding : compute_op.operand_bindings) {
    ffi::Map<ffi::String, ffi::Any> encoded_binding;
    encoded_binding.Set("role", ffi::String(binding.role));
    encoded_binding.Set("buffer", ffi::String(binding.buffer));
    if (!binding.host_buffer.empty()) {
      encoded_binding.Set("host_buffer", ffi::String(binding.host_buffer));
    }
    operand_bindings.push_back(encoded_binding);
  }
  item.Set("operand_bindings", operand_bindings);
  item.Set("M", Integer(static_cast<int>(compute_op.M)));
  item.Set("N", Integer(static_cast<int>(compute_op.N)));
  item.Set("K", Integer(static_cast<int>(compute_op.K)));
  item.Set("Mt", Integer(static_cast<int>(compute_op.Mt)));
  item.Set("Nt", Integer(static_cast<int>(compute_op.Nt)));
  item.Set("Kt", Integer(static_cast<int>(compute_op.Kt)));
  item.Set("block_m_tiles", Integer(static_cast<int>(compute_op.block_m_tiles)));
  item.Set("block_n_tiles", Integer(static_cast<int>(compute_op.block_n_tiles)));
  item.Set("block_k_tiles", Integer(static_cast<int>(compute_op.block_k_tiles)));
  item.Set("subblock_m_tiles", Integer(static_cast<int>(compute_op.subblock_m_tiles)));
  item.Set("subblock_n_tiles", Integer(static_cast<int>(compute_op.subblock_n_tiles)));
  item.Set("transpose_A", Bool(compute_op.transpose_A));
  item.Set("transpose_B", Bool(compute_op.transpose_B));
  item.Set("a_tensor_dtype", ffi::String(compute_op.a_tensor_dtype));
  item.Set("b_tensor_dtype", ffi::String(compute_op.b_tensor_dtype));
  item.Set("c_tensor_dtype", ffi::String(compute_op.c_tensor_dtype));
  item.Set("a_cb_dtype", ffi::String(compute_op.a_cb_dtype));
  item.Set("b_cb_dtype", ffi::String(compute_op.b_cb_dtype));
  item.Set("c_cb_dtype", ffi::String(compute_op.c_cb_dtype));
  item.Set("accumulator_dtype", ffi::String(compute_op.accumulator_dtype));
  item.Set("has_mbarrier", Bool(compute_op.has_mbarrier));
  item.Set("mbarrier_buffer", ffi::String(compute_op.mbarrier_buffer));
  item.Set("mbarrier_scope", ffi::String(compute_op.mbarrier_scope));
  ffi::Array<ffi::Any> mbarrier_index_exprs;
  for (const auto& expr : compute_op.mbarrier_index_exprs) {
    mbarrier_index_exprs.push_back(ffi::String(expr));
  }
  item.Set("mbarrier_index_exprs", mbarrier_index_exprs);
  return item;
}

static std::vector<KernelArgSpec> ExtractRuntimeArgsFromArray(const ffi::Array<ffi::Any>& items) {
  std::vector<KernelArgSpec> runtime_args;
  for (const auto& item : items) {
    auto arg_info = RequireMap(item, "Blackhole executable runtime arg item");

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
    ICHECK(!arg.kind.empty()) << "Blackhole runtime/common-runtime arg requires kind";
    ICHECK(!arg.identity.empty())
        << "Blackhole runtime/common-runtime arg '" << arg.name << "' kind '" << arg.kind
        << "' is missing explicit identity";
    runtime_args.push_back(std::move(arg));
  }
  return runtime_args;
}

static std::vector<AccessorSpec> ExtractAccessorsFromArray(const ffi::Array<ffi::Any>& items) {
  std::vector<AccessorSpec> accessors;
  for (const auto& item : items) {
    auto accessor_info = RequireMap(item, "Blackhole executable accessor item");

    AccessorSpec accessor;
    if (auto v = accessor_info.Get("buffer")) {
      accessor.buffer = Downcast<String>(v.value());
    }
    bool has_compile_time_arg_offset = false;
    bool has_compile_time_arg_count = false;
    bool has_common_runtime_arg_offset = false;
    bool has_common_runtime_arg_count = false;
    bool has_args_config_bits = false;
    if (auto v = accessor_info.Get("compile_time_arg_offset")) {
      accessor.compile_time_arg_offset =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      has_compile_time_arg_offset = true;
    }
    if (auto v = accessor_info.Get("compile_time_arg_count")) {
      accessor.compile_time_arg_count =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      has_compile_time_arg_count = true;
    }
    if (auto v = accessor_info.Get("common_runtime_arg_offset")) {
      accessor.common_runtime_arg_offset =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      has_common_runtime_arg_offset = true;
    }
    if (auto v = accessor_info.Get("common_runtime_arg_count")) {
      accessor.common_runtime_arg_count =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      has_common_runtime_arg_count = true;
    }
    if (auto v = accessor_info.Get("args_config_bits")) {
      accessor.args_config_bits =
          static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
      has_args_config_bits = true;
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
      ICHECK(has_compile_time_arg_offset)
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit compile_time_arg_offset";
      ICHECK(has_compile_time_arg_count)
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit compile_time_arg_count";
      ICHECK(has_common_runtime_arg_offset)
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit common_runtime_arg_offset";
      ICHECK(has_common_runtime_arg_count)
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit common_runtime_arg_count";
      ICHECK(has_args_config_bits)
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit args_config_bits";
      ICHECK(!accessor.layout.empty())
          << "Blackhole accessor for buffer " << accessor.buffer << " requires explicit layout";
      ICHECK(!accessor.memory_space.empty())
          << "Blackhole accessor for buffer " << accessor.buffer
          << " requires explicit memory_space";
      accessors.push_back(std::move(accessor));
    }
  }
  return accessors;
}

static std::vector<SemaphoreBindingSpec> ExtractSemaphoreBindingsFromArray(
    const ffi::Array<ffi::Any>& items) {
  std::vector<SemaphoreBindingSpec> bindings;
  for (const auto& item : items) {
    auto binding_info = RequireMap(item, "Blackhole executable semaphore binding item");

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
    auto spec_info = RequireMap(item, "Blackhole executable compile_time_arg_spec item");

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
    if (auto v = spec_info.Get("host_axis_order")) {
      for (const auto& axis : Downcast<ffi::Array<ffi::Any>>(v.value())) {
        spec.host_axis_order.push_back(Downcast<Integer>(axis).IntValue());
      }
    }
    if (auto v = spec_info.Get("transpose_2d")) {
      spec.transpose_2d = Downcast<Bool>(v.value());
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
    auto spec_info = RequireMap(item, "Blackhole executable per_work_arg_spec item");

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
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kDescriptorKind)) {
      spec.descriptor_kind = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kValueSource)) {
      spec.value_source = Downcast<String>(v.value());
    }
    if (auto v = spec_info.Get(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue)) {
      spec.constant_value = static_cast<uint32_t>(Downcast<Integer>(v.value()).IntValue());
    }
    ICHECK(!spec.arg_identity.empty())
        << "Blackhole per-work descriptor requires explicit arg_identity";
    ICHECK(!spec.descriptor_kind.empty())
        << "Blackhole per-work descriptor for " << spec.arg_identity
        << " requires descriptor_kind";
    ICHECK(!spec.value_source.empty())
        << "Blackhole per-work descriptor for " << spec.arg_identity
        << " requires value_source";
    per_work_arg_specs.push_back(std::move(spec));
  }
  return per_work_arg_specs;
}

static std::vector<KernelArgSpec> AggregateSegmentRuntimeArgs(
    const ffi::Array<ffi::Any>& segment_plan, const char* field_name) {
  std::vector<KernelArgSpec> aggregated;
  std::unordered_set<std::string> seen;
  for (const auto& item : segment_plan) {
    auto segment = RequireMap(item, "Blackhole executable segment_plan item");
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
    auto segment = RequireMap(item, "Blackhole executable segment_plan item");
    auto specs_it = segment.Get(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs);
    if (!specs_it.has_value()) {
      continue;
    }
    std::vector<PerWorkArgSpec> segment_specs =
        ExtractPerWorkArgSpecsFromArray(Downcast<ffi::Array<ffi::Any>>(specs_it.value()));
    for (const auto& spec : segment_specs) {
      const std::string dedupe_key = spec.arg_identity;
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
  auto segment_plan = RequireExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kSegmentPlan);
  return AggregateSegmentRuntimeArgs(segment_plan, "runtime_args");
}

static std::vector<KernelArgSpec> ExtractCommonRuntimeArgs(const tir::PrimFunc& f) {
  auto segment_plan = RequireExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kSegmentPlan);
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

static std::vector<int64_t> ExtractIntegerVector(const ffi::Map<ffi::String, ffi::Any>& item,
                                                 const char* key) {
  std::vector<int64_t> values;
  if (auto field = item.Get(key)) {
    for (const Integer& value : Downcast<ffi::Array<Integer>>(field.value())) {
      values.push_back(value->value);
    }
  }
  return values;
}

static bool HasPositiveIntegerShape(const std::vector<int64_t>& shape) {
  return !shape.empty() &&
         std::all_of(shape.begin(), shape.end(),
                     [](int64_t value) { return value > 0; });
}

static bool HasShardedSourceBinding(const BufferDistributionSpec& plan) {
  return !plan.source_buffer.empty() ||
         (!plan.source_region_kind.empty() && plan.source_region_kind != "none") ||
         !plan.source_region_shape.empty();
}

static std::vector<BufferDistributionSpec> ExtractBufferDistributionPlans(
    const tir::PrimFunc& f) {
  std::vector<BufferDistributionSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kBufferDistributionPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable buffer_distribution_plans item");
    BufferDistributionSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("buffer")) plan.buffer = Downcast<String>(value.value());
    if (auto value = item.Get("mesh_plan")) plan.mesh_plan = Downcast<String>(value.value());
    if (auto value = item.Get("mesh_plan_index")) {
      plan.mesh_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("distribution_kind")) {
      plan.distribution_kind = Downcast<String>(value.value());
    }
    if (auto value = item.Get("layout")) plan.layout = Downcast<String>(value.value());
    if (auto value = item.Get("memory_space")) {
      plan.memory_space = Downcast<String>(value.value());
    }
    bool has_page_size_bytes = false;
    if (auto value = item.Get("page_size_bytes")) {
      plan.page_size_bytes = static_cast<uint32_t>(Downcast<Integer>(value.value())->value);
      has_page_size_bytes = true;
    }
    plan.shard_shape = ExtractIntegerVector(item, "shard_shape");
    plan.shard_grid_shape = ExtractIntegerVector(item, "shard_grid_shape");
    if (auto value = item.Get("sharding_strategy")) {
      plan.sharding_strategy = Downcast<String>(value.value());
    }
    if (auto value = item.Get("shard_orientation")) {
      plan.shard_orientation = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_buffer")) {
      plan.source_buffer = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_region_kind")) {
      plan.source_region_kind = Downcast<String>(value.value());
    }
    plan.source_region_shape = ExtractIntegerVector(item, "source_region_shape");
    if (auto value = item.Get("logical_index_mapping")) {
      plan.logical_index_mapping = Downcast<String>(value.value());
    }
    if (auto value = item.Get("core_local_address_mapping")) {
      plan.core_local_address_mapping = Downcast<String>(value.value());
    }
    if (auto value = item.Get("host_visibility")) {
      plan.host_visibility = Downcast<String>(value.value());
    }
    if (auto value = item.Get("attached_core_group")) {
      plan.attached_core_group = Downcast<String>(value.value());
    }
    if (auto value = item.Get("attached_core_group_index")) {
      plan.attached_core_group_index = Downcast<Integer>(value.value())->value;
    }

    ICHECK(!plan.name.empty())
        << "Blackhole executable buffer_distribution_plans item requires name";
    ICHECK(!plan.buffer.empty())
        << "Blackhole executable buffer_distribution_plans item requires buffer";
    ICHECK(!plan.mesh_plan.empty())
        << "Blackhole executable buffer_distribution_plans item requires mesh_plan";
    ICHECK_GE(plan.mesh_plan_index, 0)
        << "Blackhole executable buffer_distribution_plans item requires mesh_plan_index";
    ICHECK(!plan.distribution_kind.empty())
        << "Blackhole executable buffer_distribution_plans item requires distribution_kind";
    ICHECK(plan.distribution_kind == "interleaved" || plan.distribution_kind == "sharded" ||
           plan.distribution_kind == "replicated")
        << "Blackhole executable buffer_distribution_plans item for " << plan.buffer
        << " has unsupported distribution_kind " << plan.distribution_kind;
    ICHECK(!plan.layout.empty())
        << "Blackhole executable buffer_distribution_plans item requires layout";
    ICHECK(!plan.memory_space.empty())
        << "Blackhole executable buffer_distribution_plans item requires memory_space";
    const bool requires_page_size_bytes =
        plan.distribution_kind == "interleaved" || plan.distribution_kind == "sharded";
    ICHECK(!requires_page_size_bytes ||
           (has_page_size_bytes && plan.page_size_bytes > 0))
        << "Blackhole executable buffer_distribution_plans item for " << plan.buffer
        << " requires positive page_size_bytes";
    ICHECK(!plan.logical_index_mapping.empty())
        << "Blackhole executable buffer_distribution_plans item for " << plan.buffer
        << " requires logical_index_mapping";
    ICHECK(!plan.host_visibility.empty())
        << "Blackhole executable buffer_distribution_plans item for " << plan.buffer
        << " requires host_visibility";
    if (plan.distribution_kind == "sharded") {
      ICHECK(plan.memory_space == "L1")
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " must be in L1";
      ICHECK(plan.sharding_strategy == "height" ||
             plan.sharding_strategy == "width" ||
             plan.sharding_strategy == "block")
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer
          << " requires sharding_strategy height, width, or block";
      ICHECK(plan.shard_orientation == "row_major" ||
             plan.shard_orientation == "col_major")
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer
          << " requires shard_orientation row_major or col_major";
      ICHECK(!plan.shard_shape.empty())
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " requires shard_shape";
      ICHECK(!plan.shard_grid_shape.empty())
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " requires shard_grid_shape";
      const bool has_source_binding = HasShardedSourceBinding(plan);
      if (has_source_binding) {
        ICHECK(!plan.source_buffer.empty())
            << "Blackhole executable buffer_distribution_plans sharded buffer "
            << plan.buffer << " source binding requires source_buffer";
        ICHECK(!plan.source_region_kind.empty() && plan.source_region_kind != "none")
            << "Blackhole executable buffer_distribution_plans sharded buffer "
            << plan.buffer << " source binding requires source_region_kind";
        ICHECK(HasPositiveIntegerShape(plan.source_region_shape))
            << "Blackhole executable buffer_distribution_plans sharded buffer "
            << plan.buffer << " source binding requires positive source_region_shape";
      } else {
        ICHECK(plan.source_region_kind.empty() || plan.source_region_kind == "none")
            << "Blackhole executable buffer_distribution_plans pure-local sharded buffer "
            << plan.buffer << " cannot carry source_region_kind without source_buffer";
        ICHECK(plan.source_region_shape.empty())
            << "Blackhole executable buffer_distribution_plans pure-local sharded buffer "
            << plan.buffer << " cannot carry source_region_shape without source_buffer";
      }
      ICHECK(!plan.core_local_address_mapping.empty() &&
             plan.core_local_address_mapping != "none")
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " requires core_local_address_mapping";
      ICHECK(!plan.attached_core_group.empty())
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " requires attached_core_group";
      ICHECK_GE(plan.attached_core_group_index, 0)
          << "Blackhole executable buffer_distribution_plans sharded buffer "
          << plan.buffer << " requires attached_core_group_index";
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

static std::vector<TensorMemoryConfigSpec> ExtractTensorMemoryConfigPlans(
    const tir::PrimFunc& f) {
  std::vector<TensorMemoryConfigSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kTensorMemoryConfigPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable tensor_memory_config_plans item");
    TensorMemoryConfigSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("subject")) plan.subject = Downcast<String>(value.value());
    if (auto value = item.Get("value_identity")) {
      plan.value_identity = Downcast<String>(value.value());
    }
    plan.logical_shape = ExtractIntegerVector(item, "logical_shape");
    if (auto value = item.Get("dtype")) plan.dtype = Downcast<String>(value.value());
    if (auto value = item.Get("memory_layout")) {
      plan.memory_layout = Downcast<String>(value.value());
    }
    if (auto value = item.Get("buffer_type")) {
      plan.buffer_type = Downcast<String>(value.value());
    }
    if (auto value = item.Get("grid_ref")) plan.grid_ref = Downcast<String>(value.value());
    plan.shard_grid_shape = ExtractIntegerVector(item, "shard_grid_shape");
    plan.shard_shape = ExtractIntegerVector(item, "shard_shape");
    if (auto value = item.Get("shard_orientation")) {
      plan.shard_orientation = Downcast<String>(value.value());
    }
    if (auto value = item.Get("shard_distribution_strategy")) {
      plan.shard_distribution_strategy = Downcast<String>(value.value());
    }
    plan.page_shape = ExtractIntegerVector(item, "page_shape");
    if (auto value = item.Get("origin")) plan.origin = Downcast<String>(value.value());
    if (auto value = item.Get("source_buffer")) {
      plan.source_buffer = Downcast<String>(value.value());
    }
    if (auto value = item.Get("buffer_distribution_plan")) {
      plan.buffer_distribution_plan = Downcast<String>(value.value());
    }
    if (auto value = item.Get("buffer_distribution_plan_index")) {
      plan.buffer_distribution_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("has_runtime_accessor")) {
      plan.has_runtime_accessor = Downcast<Bool>(value.value());
    }
    if (auto value = item.Get("requires_materialization")) {
      plan.requires_materialization = Downcast<Bool>(value.value());
    }

    ICHECK(!plan.name.empty())
        << "Blackhole executable tensor_memory_config_plans item requires name";
    ICHECK(!plan.subject.empty())
        << "Blackhole executable tensor_memory_config_plans item requires subject";
    ICHECK(!plan.memory_layout.empty())
        << "Blackhole executable tensor_memory_config_plans item for "
        << plan.subject << " requires memory_layout";
    ICHECK(!plan.buffer_type.empty())
        << "Blackhole executable tensor_memory_config_plans item for "
        << plan.subject << " requires buffer_type";
    ICHECK(!plan.origin.empty())
        << "Blackhole executable tensor_memory_config_plans item for "
        << plan.subject << " requires origin";
    if (plan.memory_layout != "INTERLEAVED") {
      ICHECK(!plan.grid_ref.empty())
          << "Blackhole executable tensor_memory_config_plans sharded item for "
          << plan.subject << " requires grid_ref";
      ICHECK(HasPositiveIntegerShape(plan.shard_grid_shape))
          << "Blackhole executable tensor_memory_config_plans sharded item for "
          << plan.subject << " requires positive shard_grid_shape";
      ICHECK(HasPositiveIntegerShape(plan.shard_shape))
          << "Blackhole executable tensor_memory_config_plans sharded item for "
          << plan.subject << " requires positive shard_shape";
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

static std::vector<ReshardPlanSpec> ExtractReshardPlans(const tir::PrimFunc& f) {
  std::vector<ReshardPlanSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kReshardPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable reshard_plans item");
    ReshardPlanSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("source_value")) {
      plan.source_value = Downcast<String>(value.value());
    }
    if (auto value = item.Get("target_value")) {
      plan.target_value = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_memory_config_plan")) {
      plan.source_memory_config_plan = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_memory_config_plan_index")) {
      plan.source_memory_config_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("target_memory_config_plan")) {
      plan.target_memory_config_plan = Downcast<String>(value.value());
    }
    if (auto value = item.Get("target_memory_config_plan_index")) {
      plan.target_memory_config_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("conversion_kind")) {
      plan.conversion_kind = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_region_kind")) {
      plan.source_region_kind = Downcast<String>(value.value());
    }
    plan.source_region_shape = ExtractIntegerVector(item, "source_region_shape");
    if (auto value = item.Get("materialization_plan")) {
      plan.materialization_plan = Downcast<String>(value.value());
    }
    if (auto value = item.Get("materialization_plan_index")) {
      plan.materialization_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("materialization_protocol")) {
      plan.materialization_protocol = Downcast<String>(value.value());
    }
    plan.required_cb_plan_indices = ExtractIntegerVector(item, "required_cb_plan_indices");
    plan.required_sync_plan_indices = ExtractIntegerVector(item, "required_sync_plan_indices");
    if (auto value = item.Get("scheduling_kind")) {
      plan.scheduling_kind = Downcast<String>(value.value());
    }
    if (auto value = item.Get("inserted_by")) {
      plan.inserted_by = Downcast<String>(value.value());
    }
    if (auto value = item.Get("admission_status")) {
      plan.admission_status = Downcast<String>(value.value());
    }
    if (auto value = item.Get("unsupported_reason")) {
      plan.unsupported_reason = Downcast<String>(value.value());
    }

    ICHECK(!plan.name.empty())
        << "Blackhole executable reshard_plans item requires name";
    ICHECK(!plan.source_value.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires source_value";
    ICHECK(!plan.target_value.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires target_value";
    ICHECK_GE(plan.source_memory_config_plan_index, 0)
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires source_memory_config_plan_index";
    ICHECK_GE(plan.target_memory_config_plan_index, 0)
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires target_memory_config_plan_index";
    ICHECK(!plan.conversion_kind.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires conversion_kind";
    ICHECK(!plan.scheduling_kind.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires scheduling_kind";
    ICHECK(!plan.inserted_by.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires inserted_by";
    ICHECK(!plan.admission_status.empty())
        << "Blackhole executable reshard_plans item " << plan.name
        << " requires admission_status";
    if (plan.conversion_kind == "interleaved_to_sharded") {
      ICHECK(!plan.materialization_protocol.empty())
          << "Blackhole executable reshard_plans item " << plan.name
          << " requires materialization_protocol";
      ICHECK(!plan.source_region_kind.empty() && plan.source_region_kind != "none")
          << "Blackhole executable reshard_plans item " << plan.name
          << " requires source_region_kind";
      ICHECK(HasPositiveIntegerShape(plan.source_region_shape))
          << "Blackhole executable reshard_plans item " << plan.name
          << " requires positive source_region_shape";
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

static std::vector<LiveFormPlanSpec> ExtractLiveFormPlans(const tir::PrimFunc& f) {
  std::vector<LiveFormPlanSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kLiveFormPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable live_form_plans item");
    LiveFormPlanSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("logical_value")) plan.logical_value = Downcast<String>(value.value());
    if (auto value = item.Get("spatial_live_value")) {
      plan.spatial_live_value = Downcast<String>(value.value());
    }
    if (auto value = item.Get("spatial_live_value_index")) {
      plan.spatial_live_value_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("producer_kernel")) {
      plan.producer_kernel = Downcast<String>(value.value());
    }
    if (auto value = item.Get("physical_form")) plan.physical_form = Downcast<String>(value.value());
    if (auto value = item.Get("execution_topology")) {
      plan.execution_topology = Downcast<String>(value.value());
    }
    if (auto value = item.Get("physical_local_extent")) {
      plan.physical_local_extent = static_cast<uint32_t>(Downcast<Integer>(value.value())->value);
    }
    if (auto value = item.Get("logical_element_count")) {
      plan.logical_element_count = static_cast<uint32_t>(Downcast<Integer>(value.value())->value);
    }
    if (auto value = item.Get("ownership_kind")) {
      plan.ownership_kind = Downcast<String>(value.value());
    }
    if (!plan.name.empty()) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

static std::vector<MaterializationPlanSpec> ExtractMaterializationPlans(const tir::PrimFunc& f) {
  std::vector<MaterializationPlanSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kMaterializationPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable materialization_plans item");
    MaterializationPlanSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("source_live_form")) {
      plan.source_live_form = Downcast<String>(value.value());
    }
    if (auto value = item.Get("materialization_boundary")) {
      plan.materialization_boundary = Downcast<String>(value.value());
    }
    if (auto value = item.Get("materialization_boundary_index")) {
      plan.materialization_boundary_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("target_buffer")) plan.target_buffer = Downcast<String>(value.value());
    if (auto value = item.Get("host_buffer")) plan.host_buffer = Downcast<String>(value.value());
    if (auto value = item.Get("target_kernel")) plan.target_kernel = Downcast<String>(value.value());
    if (auto value = item.Get("bridge_kind")) plan.bridge_kind = Downcast<String>(value.value());
    if (auto value = item.Get("materialization_kind")) {
      plan.materialization_kind = Downcast<String>(value.value());
    }
    if (auto value = item.Get("materialization_protocol")) {
      plan.materialization_protocol = Downcast<String>(value.value());
    }
    if (auto value = item.Get("publication_protocol")) {
      plan.publication_protocol = Downcast<String>(value.value());
    }
    plan.required_cb_plan_indices = ExtractIntegerVector(item, "required_cb_plan_indices");
    plan.required_sync_plan_indices = ExtractIntegerVector(item, "required_sync_plan_indices");
    if (auto value = item.Get("produced_live_form")) {
      plan.produced_live_form = Downcast<String>(value.value());
    }
    if (!plan.name.empty()) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

static std::vector<ConsumerBindingPlanSpec> ExtractConsumerBindingPlans(const tir::PrimFunc& f) {
  std::vector<ConsumerBindingPlanSpec> plans;
  auto items = tl::tt_program_projection::GetExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kConsumerBindingPlans);
  for (const auto& item_any : items) {
    auto item = RequireMap(item_any, "Blackhole executable consumer_binding_plans item");
    ConsumerBindingPlanSpec plan;
    if (auto value = item.Get("name")) plan.name = Downcast<String>(value.value());
    if (auto value = item.Get("consumer_kernel")) {
      plan.consumer_kernel = Downcast<String>(value.value());
    }
    if (auto value = item.Get("consumer_op_kind")) {
      plan.consumer_op_kind = Downcast<String>(value.value());
    }
    if (auto value = item.Get("source_live_form")) {
      plan.source_live_form = Downcast<String>(value.value());
    }
    if (auto value = item.Get("target_buffer")) {
      plan.target_buffer = Downcast<String>(value.value());
    }
    if (auto value = item.Get("materialization_plan")) {
      plan.materialization_plan = Downcast<String>(value.value());
    }
    if (auto value = item.Get("live_value_edge")) {
      plan.live_value_edge = Downcast<String>(value.value());
    }
    if (auto value = item.Get("live_value_edge_index")) {
      plan.live_value_edge_index = Downcast<Integer>(value.value())->value;
    }
    if (auto value = item.Get("accepts_distributed_slice")) {
      plan.accepts_distributed_slice = Downcast<Bool>(value.value());
    }
    if (auto value = item.Get("requires_full_logical_tile")) {
      plan.requires_full_logical_tile = Downcast<Bool>(value.value());
    }
    if (auto value = item.Get("abi_plan_index")) {
      plan.abi_plan_index = Downcast<Integer>(value.value())->value;
    }
    if (!plan.name.empty()) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

static std::vector<PerWorkArgSpec> ExtractPerWorkArgSpecs(const tir::PrimFunc& f) {
  auto segment_plan = RequireExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kSegmentPlan);
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
  std::vector<KernelComputeOpSpec> compute_ops;
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
  auto segments = RequireExecutableArrayField(
      f, "Blackhole executable spec extraction",
      tl::tt_program_projection::executable_key::kSegmentPlan);
  if (segments.empty()) {
    return segments_out;
  }

  for (const auto& item : segments) {
    auto segment = RequireMap(item, "Blackhole executable segment_plan item");

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
      auto launch_spec = RequireMap(v.value(), "Blackhole executable segment launch_spec");
      info.has_launch_spec = ExtractLaunchSpec(launch_spec, &info.launch_spec);
    }
    if (auto v = segment.Get("compute_config")) {
      auto compute_config = RequireMap(v.value(), "Blackhole executable segment compute_config");
      info.has_compute_config = ExtractComputeConfig(compute_config, &info.compute_config);
    }
    if (auto v = segment.Get("compute_ops")) {
      for (const auto& op_item : Downcast<ffi::Array<ffi::Any>>(v.value())) {
        auto compute_op = RequireMap(op_item, "Blackhole executable segment compute_op item");
        KernelComputeOpSpec op;
        if (ExtractComputeOp(compute_op, &op)) {
          info.compute_ops.push_back(std::move(op));
        }
      }
    }

    ICHECK(!info.name.empty()) << "Blackhole executable segment requires explicit name";
    ICHECK(!info.kind.empty()) << "Blackhole executable segment " << info.name
                              << " requires explicit kind";
    ICHECK(!info.core_type.empty()) << "Blackhole executable segment " << info.name
                                   << " requires explicit core_type";

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
  spec.buffer_distribution_plans = ExtractBufferDistributionPlans(f);
  spec.tensor_memory_config_plans = ExtractTensorMemoryConfigPlans(f);
  spec.reshard_plans = ExtractReshardPlans(f);
  spec.runtime_args = ExtractRuntimeArgs(f);
  spec.common_runtime_args = ExtractCommonRuntimeArgs(f);
  spec.per_work_arg_specs = ExtractPerWorkArgSpecs(f);
  spec.live_form_plans = ExtractLiveFormPlans(f);
  spec.materialization_plans = ExtractMaterializationPlans(f);
  spec.consumer_binding_plans = ExtractConsumerBindingPlans(f);
  spec.direct_runtime_unsupported_reasons = ExtractDirectRuntimeUnsupportedReasons(f);
  ExtractSegmentPlan(f, &spec);
  return spec;
}

class SegmentBodyExtractor final : public tir::StmtMutator {
 public:
  SegmentBodyExtractor(std::string segment_kind, bool retain_unmarked_stmts)
      : segment_kind_(std::move(segment_kind)),
        retain_unmarked_stmts_(retain_unmarked_stmts) {}

  enum class DetectedKind {
    kNone,
    kReader,
    kCompute,
    kWriter,
  };

  struct ChildSegmentInfo {
    DetectedKind kind = DetectedKind::kNone;
    bool has_blackhole_builtin = false;
    bool has_reader_anchor = false;
    bool has_compute_anchor = false;
    bool has_writer_anchor = false;
    bool has_cb_reserve = false;
    bool has_cb_push = false;
    bool has_cb_wait = false;
    bool has_cb_pop = false;

    bool HasSegmentAnchor() const {
      return has_reader_anchor || has_compute_anchor || has_writer_anchor;
    }

    bool Contains(DetectedKind requested) const {
      switch (requested) {
        case DetectedKind::kReader:
          return has_reader_anchor;
        case DetectedKind::kCompute:
          return has_compute_anchor;
        case DetectedKind::kWriter:
          return has_writer_anchor;
        case DetectedKind::kNone:
          return false;
      }
      return false;
    }
  };

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
    Stmt body = this->VisitStmt(op->body);
    if (IsNoOp(body)) {
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return tir::AttrStmt(op->node, op->attr_key, op->value, body);
  }

  Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
    std::vector<ChildSegmentInfo> child_info;
    child_info.reserve(op->seq.size());
    bool has_segment_anchors = false;
    bool has_ambiguous_blackhole_stmt = false;
    for (const Stmt& stmt : op->seq) {
      ChildSegmentInfo info = DetectChildSegmentInfo(stmt);
      has_segment_anchors = has_segment_anchors || info.HasSegmentAnchor();
      has_ambiguous_blackhole_stmt =
          has_ambiguous_blackhole_stmt ||
          (!info.HasSegmentAnchor() && info.has_blackhole_builtin);
      child_info.push_back(info);
    }

    if (!has_segment_anchors) {
      if (retain_unmarked_stmts_ ||
          (has_ambiguous_blackhole_stmt && forced_segment_kind_ == RequestedSegmentKind())) {
        return tir::StmtMutator::VisitStmt_(op);
      }
      return tir::Evaluate(IntImm(DataType::Int(32), 0));
    }

    if (has_ambiguous_blackhole_stmt) {
      for (size_t i = 0; i < child_info.size(); ++i) {
        if (child_info[i].kind != DetectedKind::kNone || !child_info[i].has_blackhole_builtin) {
          continue;
        }
        child_info[i].kind = InferAmbiguousSegmentKind(child_info, i);
      }
    }

    ffi::Array<Stmt> seq;
    seq.reserve(op->seq.size());
    for (size_t i = 0; i < op->seq.size(); ++i) {
      const bool keep_segment_stmt =
          child_info[i].Contains(RequestedSegmentKind()) ||
          child_info[i].kind == RequestedSegmentKind() ||
          (child_info[i].kind == DetectedKind::kNone && retain_unmarked_stmts_);
      if (!keep_segment_stmt) {
        continue;
      }
      const DetectedKind previous_forced_segment_kind = forced_segment_kind_;
      if (child_info[i].Contains(RequestedSegmentKind()) ||
          child_info[i].kind == RequestedSegmentKind()) {
        forced_segment_kind_ = RequestedSegmentKind();
      }
      Stmt rewritten = this->VisitStmt(op->seq[i]);
      forced_segment_kind_ = previous_forced_segment_kind;
      if (!IsNoOp(rewritten)) {
        seq.push_back(rewritten);
      }
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
  static bool IsReaderAnchor(const tir::CallNode* op) {
    return op->op.same_as(tir::builtin::blackhole_read_tile_to_cb()) ||
           op->op.same_as(tir::builtin::blackhole_read_page_to_cb()) ||
           op->op.same_as(tir::builtin::blackhole_read_bcast_cols_to_cb());
  }

  static bool IsWriterAnchor(const tir::CallNode* op) {
    return op->op.same_as(tir::builtin::blackhole_write_tile_from_cb()) ||
           op->op.same_as(tir::builtin::blackhole_write_page_from_cb());
  }

  static bool IsBlackholeBuiltin(const tir::CallNode* op) {
    const auto* op_node = op->op.as<OpNode>();
    return op_node != nullptr && std::string(op_node->name).rfind("tl.blackhole.", 0) == 0;
  }

  static bool IsComputeAnchor(const tir::CallNode* op) {
    if (!IsBlackholeBuiltin(op) || IsReaderAnchor(op) || IsWriterAnchor(op)) {
      return false;
    }
    return !op->op.same_as(tir::builtin::blackhole_cb_reserve_back()) &&
           !op->op.same_as(tir::builtin::blackhole_cb_push_back()) &&
           !op->op.same_as(tir::builtin::blackhole_cb_wait_front()) &&
           !op->op.same_as(tir::builtin::blackhole_cb_pop_front()) &&
           !op->op.same_as(tir::builtin::blackhole_noc_async_read()) &&
           !op->op.same_as(tir::builtin::blackhole_noc_async_write()) &&
           !op->op.same_as(tir::builtin::blackhole_noc_async_read_barrier()) &&
           !op->op.same_as(tir::builtin::blackhole_noc_async_write_barrier()) &&
           !op->op.same_as(tir::builtin::blackhole_get_semaphore()) &&
           !op->op.same_as(tir::builtin::blackhole_runtime_arg_u32()) &&
           !op->op.same_as(tir::builtin::blackhole_semaphore_wait()) &&
           !op->op.same_as(tir::builtin::blackhole_semaphore_set()) &&
           !op->op.same_as(tir::builtin::blackhole_semaphore_inc_remote()) &&
           !op->op.same_as(tir::builtin::blackhole_semaphore_set_remote());
  }

  static ChildSegmentInfo DetectChildSegmentInfo(const Stmt& stmt) {
    ChildSegmentInfo info;
    tir::PostOrderVisit(stmt, [&](const tvm::runtime::ObjectRef& node) {
      const auto* op = node.as<tir::CallNode>();
      if (!op || !SegmentBodyExtractor::IsBlackholeBuiltin(op)) {
        return;
      }
      info.has_blackhole_builtin = true;
      if (SegmentBodyExtractor::IsReaderAnchor(op)) {
        info.has_reader_anchor = true;
      } else if (SegmentBodyExtractor::IsWriterAnchor(op)) {
        info.has_writer_anchor = true;
      } else if (SegmentBodyExtractor::IsComputeAnchor(op)) {
        info.has_compute_anchor = true;
      }
      if (op->op.same_as(tir::builtin::blackhole_cb_reserve_back())) {
        info.has_cb_reserve = true;
      } else if (op->op.same_as(tir::builtin::blackhole_cb_push_back())) {
        info.has_cb_push = true;
      } else if (op->op.same_as(tir::builtin::blackhole_cb_wait_front())) {
        info.has_cb_wait = true;
      } else if (op->op.same_as(tir::builtin::blackhole_cb_pop_front())) {
        info.has_cb_pop = true;
      }
    });
    const int segment_kinds = static_cast<int>(info.has_reader_anchor) +
                              static_cast<int>(info.has_compute_anchor) +
                              static_cast<int>(info.has_writer_anchor);
    if (segment_kinds == 1) {
      if (info.has_reader_anchor) {
        info.kind = DetectedKind::kReader;
      } else if (info.has_compute_anchor) {
        info.kind = DetectedKind::kCompute;
      } else if (info.has_writer_anchor) {
        info.kind = DetectedKind::kWriter;
      }
    }
    return info;
  }

  static DetectedKind InferNextSegmentKind(const std::vector<ChildSegmentInfo>& info,
                                           size_t index) {
    for (size_t i = index + 1; i < info.size(); ++i) {
      if (info[i].kind != DetectedKind::kNone) {
        return info[i].kind;
      }
    }
    return DetectedKind::kNone;
  }

  static DetectedKind InferPreviousSegmentKind(const std::vector<ChildSegmentInfo>& info,
                                               size_t index) {
    for (size_t i = index; i > 0; --i) {
      if (info[i - 1].kind != DetectedKind::kNone) {
        return info[i - 1].kind;
      }
    }
    return DetectedKind::kNone;
  }

  static DetectedKind InferAmbiguousSegmentKind(const std::vector<ChildSegmentInfo>& info,
                                                size_t index) {
    const ChildSegmentInfo& current = info[index];
    if (current.has_cb_reserve || current.has_cb_wait) {
      DetectedKind next = InferNextSegmentKind(info, index);
      if (next != DetectedKind::kNone) {
        return next;
      }
    }
    if (current.has_cb_push || current.has_cb_pop) {
      DetectedKind previous = InferPreviousSegmentKind(info, index);
      if (previous != DetectedKind::kNone) {
        return previous;
      }
    }
    return InferNeighborSegmentKind(info, index);
  }

  static DetectedKind InferNeighborSegmentKind(const std::vector<ChildSegmentInfo>& info,
                                               size_t index) {
    DetectedKind previous = InferPreviousSegmentKind(info, index);
    if (previous != DetectedKind::kNone) {
      return previous;
    }
    DetectedKind next = InferNextSegmentKind(info, index);
    if (next != DetectedKind::kNone) {
      return next;
    }
    return DetectedKind::kNone;
  }

  DetectedKind RequestedSegmentKind() const {
    if (segment_kind_ == "reader") {
      return DetectedKind::kReader;
    }
    if (segment_kind_ == "compute") {
      return DetectedKind::kCompute;
    }
    if (segment_kind_ == "writer") {
      return DetectedKind::kWriter;
    }
    return DetectedKind::kNone;
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
  DetectedKind forced_segment_kind_{DetectedKind::kNone};
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
    if (!spec.host_axis_order.empty()) {
      ffi::Array<ffi::Any> axis_order;
      for (int64_t axis : spec.host_axis_order) {
        axis_order.push_back(Integer(axis));
      }
      spec_info.Set("host_axis_order", axis_order);
    }
    if (spec.transpose_2d) {
      spec_info.Set("transpose_2d", Bool(true));
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
    if (!spec.descriptor_kind.empty()) {
      spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kDescriptorKind,
                    ffi::String(spec.descriptor_kind));
    }
    if (!spec.value_source.empty()) {
      spec_info.Set(::tvm::tl::blackhole_runtime_arg_schema::kValueSource,
                    ffi::String(spec.value_source));
    }
    if (spec.value_source ==
        ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceConstant) {
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

  ICHECK_NE(inferred_page_size, 0)
      << "Blackhole buffer materialization requires explicit transport_page_size for buffer "
      << buffer_name;
  return inferred_page_size;
}

static bool KernelArgsContainKind(const std::vector<KernelArgSpec>& args,
                                  std::string_view kind) {
  return std::any_of(args.begin(), args.end(), [&](const KernelArgSpec& arg) {
    return arg.kind == kind;
  });
}

static bool PerWorkArgSpecsContainArgIdentity(const std::vector<PerWorkArgSpec>& specs,
                                              std::string_view identity) {
  return std::any_of(specs.begin(), specs.end(), [&](const PerWorkArgSpec& spec) {
    return spec.arg_identity == identity;
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
    ICHECK(!arg.identity.empty())
        << "Blackhole build requires runtime arg identity before per-work descriptor binding for "
        << arg.name << " kind=" << arg.kind << " on kernel " << kernel.name << " of "
        << func_name;
    ICHECK(PerWorkArgSpecsContainArgIdentity(kernel.per_work_arg_specs, arg.identity))
        << "Blackhole build requires explicit per-work arg binding for runtime arg kind '"
        << arg.kind << "' identity '" << arg.identity << "' on kernel " << kernel.name
        << " of " << func_name
        << "; runtime/codegen must not recover block/tile semantics from work_linear_id or "
        << "implicit runtime-arg inference";
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
      << "executable segment runtime args are missing for kernel " << kernel.name
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
            if (arg.identity.empty() ||
                !PerWorkArgSpecsContainArgIdentity(per_work_arg_specs, arg.identity)) {
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

static bool IsDirectRuntimeAdmittedPublicationProtocol(const std::string& publication_protocol) {
  return publication_protocol == buffer_materialization::kPackThreadDirectStore ||
         publication_protocol == buffer_materialization::kPackTile ||
         publication_protocol == buffer_materialization::kTilizeCastFragmentSlice;
}

static bool IsDirectRuntimeAdmittedHostPublicationProtocol(
    const std::string& publication_protocol) {
  return publication_protocol == buffer_materialization::kPackThreadDirectStore ||
         publication_protocol == buffer_materialization::kPackTile;
}

static void EnforceTypedDstCbAccumulationGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  for (const auto& materialization : spec->buffer_materializations) {
    if (materialization.materialization_protocol != buffer_materialization::kCBRepublish) {
      continue;
    }
    if (materialization.execution_topology_kind != "thread_distributed") {
      continue;
    }
    if (IsDirectRuntimeAdmittedHostPublicationProtocol(materialization.publication_protocol)) {
      continue;
    }
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "thread-distributed cb_republish materialization is not admitted by direct runtime; "
        "requires a non-mailbox materialization protocol for compute-thread CB publication");
    return;
  }

  std::unordered_map<std::string, const LiveFormPlanSpec*> live_form_by_name;
  for (const auto& live_form : spec->live_form_plans) {
    live_form_by_name.emplace(live_form.name, &live_form);
  }
  for (const auto& plan : spec->materialization_plans) {
    if (plan.materialization_protocol != buffer_materialization::kCBRepublish ||
        IsDirectRuntimeAdmittedPublicationProtocol(plan.publication_protocol)) {
      continue;
    }
    auto live_form_it = live_form_by_name.find(plan.produced_live_form);
    if (live_form_it == live_form_by_name.end() ||
        live_form_it->second->execution_topology != "thread_distributed") {
      continue;
    }
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "thread-distributed cb_republish materialization is not admitted by direct runtime; "
        "requires a non-mailbox materialization protocol for compute-thread CB publication");
    return;
  }
}

static bool SpecHasRuntimeArgKind(const ExecutableSpec& spec, std::string_view kind) {
  if (KernelArgsContainKind(spec.runtime_args, kind) ||
      KernelArgsContainKind(spec.common_runtime_args, kind)) {
    return true;
  }
  for (const auto& kernel : spec.kernels) {
    if (KernelArgsContainKind(kernel.runtime_args, kind) ||
        KernelArgsContainKind(kernel.common_runtime_args, kind)) {
      return true;
    }
  }
  return false;
}

static bool SpecHasThreadDistributedCastRepublishPlan(const ExecutableSpec& spec) {
  std::unordered_map<std::string, const LiveFormPlanSpec*> live_form_by_name;
  for (const auto& live_form : spec.live_form_plans) {
    live_form_by_name.emplace(live_form.name, &live_form);
  }
  for (const auto& plan : spec.materialization_plans) {
    if (plan.materialization_protocol != buffer_materialization::kCBRepublish ||
        plan.publication_protocol != buffer_materialization::kTilizeCastFragmentSlice) {
      continue;
    }
    auto live_form_it = live_form_by_name.find(plan.produced_live_form);
    if (live_form_it == live_form_by_name.end() ||
        live_form_it->second->execution_topology == "thread_distributed") {
      return true;
    }
  }
  return false;
}

static void EnforceExactLiveFormMultiPageRepublishGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  if (!SpecHasThreadDistributedCastRepublishPlan(*spec)) {
    return;
  }
  if (!SpecHasRuntimeArgKind(*spec, "num_k_tiles")) {
    return;
  }
  for (const CBConfig& cb : spec->cb_configs) {
    if (cb.role != "input" || cb.flow_class != "republish" || cb.num_pages <= 1) {
      continue;
    }
    if (cb.publish_pages_per_event == 1 && cb.consume_pages_per_event == 1) {
      continue;
    }
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "multi-page exact CB-republish live-form direct runtime is not admitted; "
        "requires typed one-page producer and consumer event windows");
    return;
  }
}

static void EnforceMultiBlockExactCBRepublishGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  if (!SpecHasThreadDistributedCastRepublishPlan(*spec) ||
      !SpecHasRuntimeArgKind(*spec, "num_k_tiles")) {
    return;
  }
  for (const CBConfig& cb : spec->cb_configs) {
    if (cb.role == "input" && cb.flow_class == "republish" && cb.num_pages > 1) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "multi-block exact CB-republish flash-attention direct runtime correctness is not "
          "admitted; compile/source/spec lowering is supported, but runtime execution needs "
          "the multi-block online-softmax live-form contract to be admitted separately");
      return;
    }
  }
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
      expected_buffer_names.insert(spec->tvm_arg_names[i]);
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
      bound_buffer_names.insert(arg.buffer);
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

static std::unordered_set<std::string> CollectRuntimeBoundBufferNames(
    const ExecutableSpec& spec) {
  std::unordered_set<std::string> names;
  auto record_args = [&](const std::vector<KernelArgSpec>& args) {
    for (const auto& arg : args) {
      if ((IsInputBufferArgKind(arg.kind) || IsOutputBufferArgKind(arg.kind)) &&
          !arg.buffer.empty()) {
        names.insert(arg.buffer);
      }
    }
  };
  record_args(spec.runtime_args);
  record_args(spec.common_runtime_args);
  for (const auto& kernel : spec.kernels) {
    record_args(kernel.runtime_args);
    record_args(kernel.common_runtime_args);
  }
  return names;
}

static std::unordered_set<std::string> CollectRuntimeOutputBufferNames(
    const ExecutableSpec& spec) {
  std::unordered_set<std::string> names;
  auto record_args = [&](const std::vector<KernelArgSpec>& args) {
    for (const auto& arg : args) {
      if (IsOutputBufferArgKind(arg.kind) && !arg.buffer.empty()) {
        names.insert(arg.buffer);
      }
    }
  };
  record_args(spec.runtime_args);
  record_args(spec.common_runtime_args);
  for (const auto& kernel : spec.kernels) {
    record_args(kernel.runtime_args);
    record_args(kernel.common_runtime_args);
  }
  return names;
}

static bool IsComputeOutputBindingRole(const std::string& role) {
  return role == "output" || role == "out" || role == "dst" ||
         role == "result";
}

static std::unordered_set<std::string> CollectComputeInputBufferNames(
    const ExecutableSpec& spec) {
  std::unordered_set<std::string> names;
  for (const KernelSpec& kernel : spec.kernels) {
    for (const KernelComputeOpSpec& compute_op : kernel.compute_ops) {
      if (!compute_op.enabled) {
        continue;
      }
      for (const auto& binding : compute_op.operand_bindings) {
        if (!binding.buffer.empty() && !IsComputeOutputBindingRole(binding.role)) {
          names.insert(binding.buffer);
        }
      }
    }
  }
  return names;
}

static void EnforceStandalonePacrLeafSimulatorGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  bool has_reduce = false;
  bool has_fill_typecast_publish = false;
  bool has_gemm = false;
  const std::unordered_set<std::string> runtime_outputs =
      CollectRuntimeOutputBufferNames(*spec);
  const std::unordered_set<std::string> compute_inputs =
      CollectComputeInputBufferNames(*spec);
  for (const KernelSpec& kernel : spec->kernels) {
    for (const KernelComputeOpSpec& compute_op : kernel.compute_ops) {
      if (!compute_op.enabled) {
        continue;
      }
      has_gemm = has_gemm || compute_op.kind == "gemm";
      bool produces_runtime_output = false;
      bool produces_terminal_compute_value = false;
      for (const auto& binding : compute_op.operand_bindings) {
        if (!IsComputeOutputBindingRole(binding.role) || binding.buffer.empty()) {
          continue;
        }
        if (runtime_outputs.count(binding.buffer) != 0U) {
          produces_runtime_output = true;
        }
        if (compute_inputs.count(binding.buffer) == 0U) {
          produces_terminal_compute_value = true;
        }
      }
      const bool terminal_leaf_publish =
          produces_runtime_output || produces_terminal_compute_value;
      has_reduce =
          has_reduce || ((compute_op.kind == "reduce" ||
                          compute_op.operation_name == "reduce_tile") &&
                         terminal_leaf_publish);
      has_fill_typecast_publish =
          has_fill_typecast_publish ||
          ((compute_op.operation_name == "fill_tile" ||
            compute_op.operation_name == "typecast_tile") &&
           terminal_leaf_publish);
    }
  }
  if (has_reduce && !has_gemm) {
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "standalone reduce_tile leaf direct runtime is gated: TT-Sim reports "
        "UnimplementedFunctionality tensix_execute_pacr count=1 for row-reduce pack");
  }
  if (has_fill_typecast_publish && !has_gemm) {
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "standalone fill/typecast publish direct runtime is gated: TT-Sim reports "
        "tensix_execute_pacr unsupported for compute-only pack publish");
  }
}

static const BufferDistributionSpec* FindBufferDistributionSpec(
    const ExecutableSpec& spec, const std::string& buffer_name) {
  auto it = std::find_if(
      spec.buffer_distribution_plans.begin(), spec.buffer_distribution_plans.end(),
      [&](const BufferDistributionSpec& plan) { return plan.buffer == buffer_name; });
  return it == spec.buffer_distribution_plans.end() ? nullptr : &(*it);
}

static std::string NormalizeMemorySpace(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static bool HasPositiveShape(const std::vector<int64_t>& shape) {
  return !shape.empty() &&
         std::all_of(shape.begin(), shape.end(), [](int64_t value) { return value > 0; });
}

static bool IsAdmittedRuntimeBufferDistribution(const BufferDistributionSpec& distribution,
                                                std::string* reject_reason) {
  const std::string memory_space = NormalizeMemorySpace(distribution.memory_space);
  if (distribution.distribution_kind == "interleaved" &&
      (distribution.layout == "interleaved" ||
       distribution.layout == "page_indexed") &&
      memory_space == "dram") {
    if (distribution.logical_index_mapping != "interleaved_page_index") {
      if (reject_reason != nullptr) {
        *reject_reason = "lacks interleaved_page_index address mapping";
      }
      return false;
    }
    return true;
  }
  if (distribution.distribution_kind == "sharded" &&
      distribution.layout == "sharded" && memory_space == "l1") {
    const bool valid_strategy = distribution.sharding_strategy == "height" ||
                                distribution.sharding_strategy == "width" ||
                                distribution.sharding_strategy == "block";
    const bool valid_orientation = distribution.shard_orientation == "row_major" ||
                                   distribution.shard_orientation == "col_major";
    if (!valid_strategy || !valid_orientation ||
        !HasPositiveShape(distribution.shard_grid_shape) ||
        !HasPositiveShape(distribution.shard_shape) ||
        distribution.logical_index_mapping != "work_packet_row_major" ||
        distribution.core_local_address_mapping != "l1_shard_linear" ||
        distribution.attached_core_group.empty() ||
        distribution.attached_core_group_index < 0) {
      if (reject_reason != nullptr) {
        *reject_reason =
            "lacks sharded L1 strategy/orientation/shape/core mapping fields";
      }
      return false;
    }
    return true;
  }
  if (reject_reason != nullptr) {
    *reject_reason = "is " + distribution.distribution_kind + "/" + distribution.layout +
                     "/" + memory_space;
  }
  return false;
}

static void EnforceBufferDistributionAddressContractGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  const std::unordered_set<std::string> runtime_buffers =
      CollectRuntimeBoundBufferNames(*spec);
  if (runtime_buffers.empty()) {
    return;
  }
  if (spec->buffer_distribution_plans.empty()) {
    AppendDirectRuntimeUnsupportedReason(
        spec,
        "missing buffer distribution address contract; direct runtime requires "
        "ExecutableSpec.buffer_distribution_plans for runtime buffer address materialization");
    return;
  }

  for (const auto& buffer_name : runtime_buffers) {
    const BufferDistributionSpec* distribution =
        FindBufferDistributionSpec(*spec, buffer_name);
    if (distribution == nullptr) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "missing buffer distribution address contract for runtime buffer " +
              buffer_name);
      return;
    }
    std::string reject_reason;
    if (!IsAdmittedRuntimeBufferDistribution(*distribution, &reject_reason)) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "buffer distribution for runtime buffer " + buffer_name + " " +
              reject_reason +
              "; direct runtime admits interleaved DRAM or static sharded L1 runtime buffers");
      return;
    }
    if (distribution->page_size_bytes == 0) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "buffer distribution for runtime buffer " + buffer_name +
              " requires positive page_size_bytes");
      return;
    }
    auto materialization_it = std::find_if(
        spec->buffer_materializations.begin(), spec->buffer_materializations.end(),
        [&](const BufferMaterializationSpec& materialization) {
          return materialization.buffer == buffer_name;
        });
    if (materialization_it != spec->buffer_materializations.end() &&
        materialization_it->transport_page_size_bytes != distribution->page_size_bytes) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "buffer distribution for runtime buffer " + buffer_name +
              " has page_size_bytes inconsistent with buffer materialization transport_page_size");
      return;
    }
  }

  for (const auto& plan : spec->buffer_distribution_plans) {
    if (plan.distribution_kind != "sharded") {
      continue;
    }
    if (!HasShardedSourceBinding(plan)) {
      continue;
    }
    if (plan.source_buffer.empty() || plan.source_region_kind != "per_work_tile" ||
        plan.source_region_shape.empty() ||
        plan.logical_index_mapping != "work_packet_row_major" ||
        plan.core_local_address_mapping != "l1_shard_linear" ||
        plan.attached_core_group.empty() || plan.attached_core_group_index < 0) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "sharded L1 buffer distribution for " + plan.buffer +
              " is not admitted by direct runtime; expected source_buffer, "
              "per_work_tile source_region_shape, work_packet_row_major, and "
              "l1_shard_linear typed address mapping");
      return;
    }
    if (FindBufferDistributionSpec(*spec, plan.source_buffer) == nullptr) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "sharded L1 buffer distribution for " + plan.buffer +
              " references missing source_buffer distribution " + plan.source_buffer);
      return;
    }
  }
}

static const ReshardPlanSpec* FindReshardPlanSpec(
    const ExecutableSpec& spec, const std::string& source_value,
    const std::string& target_value) {
  auto it = std::find_if(
      spec.reshard_plans.begin(), spec.reshard_plans.end(),
      [&](const ReshardPlanSpec& plan) {
        return plan.source_value == source_value && plan.target_value == target_value;
      });
  return it == spec.reshard_plans.end() ? nullptr : &(*it);
}

static void EnforceProjectedReshardAdmissionGate(ExecutableSpec* spec) {
  ICHECK(spec != nullptr);
  for (const auto& distribution : spec->buffer_distribution_plans) {
    if (distribution.distribution_kind != "sharded" ||
        !HasShardedSourceBinding(distribution)) {
      continue;
    }
    const ReshardPlanSpec* reshard =
        FindReshardPlanSpec(*spec, distribution.source_buffer, distribution.buffer);
    if (reshard == nullptr) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "missing projected reshard conversion for " + distribution.source_buffer +
              " -> " + distribution.buffer +
              "; direct runtime consumes TTReshardPlan records and must not infer "
              "conversion from buffer distribution source bindings");
      return;
    }
  }

  for (const auto& reshard : spec->reshard_plans) {
    if (reshard.admission_status != "admitted") {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "projected reshard conversion " + reshard.name + " is not admitted: " +
              (reshard.unsupported_reason.empty() ? "missing unsupported_reason"
                                                  : reshard.unsupported_reason));
      return;
    }
    if (reshard.conversion_kind != "interleaved_to_sharded") {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "projected reshard conversion " + reshard.name + " kind " +
              reshard.conversion_kind +
              " is not admitted by direct runtime; current admitted conversion is "
              "interleaved_to_sharded staged copy");
      return;
    }
    if (reshard.materialization_protocol != "staged_copy") {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "projected reshard conversion " + reshard.name +
              " requires unsupported materialization protocol " +
              reshard.materialization_protocol);
      return;
    }
    if (reshard.source_region_kind != "per_work_tile" ||
        !HasPositiveIntegerShape(reshard.source_region_shape)) {
      AppendDirectRuntimeUnsupportedReason(
          spec,
          "projected reshard conversion " + reshard.name +
              " is missing admitted per_work_tile source-region evidence");
      return;
    }
  }
}

static bool BufferMaterializationRequiresExplicitHostAxisOrder(
    const BufferMaterializationSpec& materialization) {
  return materialization.layout == "interleaved" && materialization.memory_space == "dram" &&
         materialization.transport_page_size_bytes > 0;
}

static void PopulateBufferMaterializationSpecs(
    const std::unordered_map<std::string, StaticBufferInfo>& buffer_info_by_name,
    ExecutableSpec* spec) {
  (void)buffer_info_by_name;
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
                      compile_time_arg_spec.memory_space, compile_time_arg_spec.host_axis_order,
                      compile_time_arg_spec.transpose_2d);
    }
  }

  std::unordered_map<std::string, const LiveFormPlanSpec*> live_form_by_name;
  for (const auto& live_form : spec->live_form_plans) {
    live_form_by_name.emplace(live_form.name, &live_form);
  }

  for (const auto& plan : spec->materialization_plans) {
    if (plan.host_buffer.empty()) {
      ICHECK(plan.publication_protocol != buffer_materialization::kPackThreadDirectStore &&
             plan.publication_protocol != buffer_materialization::kPackTile)
          << "Blackhole buffer materialization plan requires explicit host_buffer for target "
          << plan.target_buffer;
      continue;
    }
    auto materialization_it = by_buffer.find(plan.host_buffer);
    ICHECK(materialization_it != by_buffer.end())
        << "Blackhole buffer materialization plan host_buffer " << plan.host_buffer
        << " must reference a registered host buffer materialization";
    auto live_form_it = live_form_by_name.find(plan.produced_live_form);
    ICHECK(live_form_it != live_form_by_name.end())
        << "Blackhole buffer materialization plan references unknown produced_live_form "
        << plan.produced_live_form;
    BufferMaterializationSpec& materialization = materialization_it->second;
    const LiveFormPlanSpec& live_form = *live_form_it->second;
    materialization.live_form_kind = live_form.physical_form;
    materialization.execution_topology_kind = live_form.execution_topology;
    materialization.physical_local_extent = live_form.physical_local_extent;
    materialization.logical_element_count = live_form.logical_element_count;
    materialization.producer_kernel = live_form.producer_kernel.empty()
                                          ? plan.target_kernel
                                          : live_form.producer_kernel;
    materialization.materialization_protocol = plan.materialization_protocol;
    materialization.publication_protocol = plan.publication_protocol;
  }

  spec->buffer_materializations.clear();
  spec->buffer_materializations.reserve(order.size());
  for (const auto& buffer_name : order) {
    auto materialization = by_buffer.at(buffer_name);
    materialization.transport_page_size_bytes =
        ChooseBufferMaterializationPageSize(*spec, buffer_name);
    if (materialization.host_axis_order.empty()) {
      ICHECK(!BufferMaterializationRequiresExplicitHostAxisOrder(materialization))
          << "Blackhole buffer materialization requires explicit host_axis_order for buffer "
          << materialization.buffer;
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
  ICHECK(!segment.kind.empty())
      << "Blackhole segment materialization requires explicit segment kind";
  ICHECK(!segment.core_type.empty())
      << "Blackhole segment materialization requires explicit segment core_type";
  const ffi::String kernel_kind = ffi::String(segment.kind);
  const ffi::String kernel_core_type = ffi::String(segment.core_type);
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
  if (!segment.compute_ops.empty()) {
    ffi::Array<ffi::Any> compute_ops;
    for (const auto& compute_op : segment.compute_ops) {
      compute_ops.push_back(EncodeKernelComputeOp(compute_op));
    }
    encoded_segment.Set("compute_ops", compute_ops);
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
      << "Blackhole build requires non-empty executable segment truth on device PrimFunc "
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
    kernel.has_compute_config = segment.has_compute_config;
    if (segment.has_compute_config) {
      kernel.compute_config = segment.compute_config;
    }
    kernel.compute_ops = segment.compute_ops;
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
    EnforceBufferDistributionAddressContractGate(&spec_it->second);
    EnforceProjectedReshardAdmissionGate(&spec_it->second);
    EnforceExplicitPerWorkAccessDescriptorGate(buffer_info, &spec_it->second);
    EnforceTypedDstCbAccumulationGate(&spec_it->second);
    EnforceExactLiveFormMultiPageRepublishGate(&spec_it->second);
    EnforceExplicitBufferRoleSchemaGate(&spec_it->second);
    EnforceStandalonePacrLeafSimulatorGate(&spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
      host_it->second.buffer_distribution_plans =
          device_it->second.buffer_distribution_plans;
      host_it->second.tensor_memory_config_plans =
          device_it->second.tensor_memory_config_plans;
      host_it->second.reshard_plans = device_it->second.reshard_plans;
      host_it->second.buffer_materializations = device_it->second.buffer_materializations;
      host_it->second.live_form_plans = device_it->second.live_form_plans;
      host_it->second.materialization_plans = device_it->second.materialization_plans;
      host_it->second.consumer_binding_plans = device_it->second.consumer_binding_plans;
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
    EnforceBufferDistributionAddressContractGate(&spec_it->second);
    EnforceProjectedReshardAdmissionGate(&spec_it->second);
    EnforceExplicitPerWorkAccessDescriptorGate(buffer_info, &spec_it->second);
    EnforceTypedDstCbAccumulationGate(&spec_it->second);
    EnforceExactLiveFormMultiPageRepublishGate(&spec_it->second);
    EnforceExplicitBufferRoleSchemaGate(&spec_it->second);
    EnforceStandalonePacrLeafSimulatorGate(&spec_it->second);
  }
  for (const auto& kv : host_to_device) {
    auto host_it = func_info_map.find(kv.first);
    auto device_it = func_info_map.find(kv.second);
    if (host_it != func_info_map.end() && device_it != func_info_map.end()) {
      host_it->second.kernels = device_it->second.kernels;
      host_it->second.buffer_distribution_plans =
          device_it->second.buffer_distribution_plans;
      host_it->second.tensor_memory_config_plans =
          device_it->second.tensor_memory_config_plans;
      host_it->second.reshard_plans = device_it->second.reshard_plans;
      host_it->second.buffer_materializations = device_it->second.buffer_materializations;
      host_it->second.live_form_plans = device_it->second.live_form_plans;
      host_it->second.materialization_plans = device_it->second.materialization_plans;
      host_it->second.consumer_binding_plans = device_it->second.consumer_binding_plans;
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
