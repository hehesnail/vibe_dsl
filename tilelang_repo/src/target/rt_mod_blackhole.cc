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
#include <sstream>
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
  if (auto v = core_plan.Get("grid_x")) {
    plan.grid_x = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("grid_y")) {
    plan.grid_y = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("cores_needed")) {
    plan.cores_needed = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("work_per_core")) {
    plan.work_per_core = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("core_grid_x")) {
    plan.core_grid_x = Downcast<Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("core_grid_y")) {
    plan.core_grid_y = Downcast<Integer>(v.value()).IntValue();
  }
  return plan;
}

static std::vector<KernelArgSpec> MakeDefaultRuntimeArgs(const ExecutableSpec& spec) {
  std::vector<KernelArgSpec> runtime_args;

  if (spec.target_mode == "single_core_copy") {
    runtime_args.push_back({"input0", "input_buffer_addr32", "uint32"});
    runtime_args.push_back({"output0", "output_buffer_addr32", "uint32"});
    runtime_args.push_back({"num_tiles", "tile_count", "uint32"});
    runtime_args.push_back({"scratch_l1", "scratch_l1_buffer_addr32", "uint32"});
  }

  return runtime_args;
}

static std::vector<KernelArgSpec> ExtractRuntimeArgs(const tir::PrimFunc& f,
                                                     const ExecutableSpec& spec) {
  std::vector<KernelArgSpec> runtime_args;
  auto runtime_args_attr = f->GetAttr<ffi::Array<ffi::Any>>("blackhole.runtime_args");
  if (!runtime_args_attr) {
    return MakeDefaultRuntimeArgs(spec);
  }

  for (const auto& item : runtime_args_attr.value()) {
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
    if (!arg.kind.empty()) {
      runtime_args.push_back(std::move(arg));
    }
  }

  if (runtime_args.empty()) {
    return MakeDefaultRuntimeArgs(spec);
  }
  return runtime_args;
}

static uint32_t GetCopyTileSizeBytes(const ExecutableSpec& spec) {
  for (const auto& cb : spec.cb_configs) {
    if (cb.role == "input" || cb.role == "output") {
      return cb.page_size_bytes;
    }
  }
  return 2048;
}

static std::string EmitSingleCoreCopyKernelSource(const ExecutableSpec& spec) {
  const uint32_t tile_size = GetCopyTileSizeBytes(spec);
  std::ostringstream os;
  os << "// TT-Metal single-core copy kernel generated by TileLang\n";
  os << "#include <cstdint>\n\n";
  os << "void kernel_main() {\n";
  os << "  uint32_t src_dram_addr = get_arg_val<uint32_t>(0);\n";
  os << "  uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);\n";
  os << "  uint32_t num_tiles = get_arg_val<uint32_t>(2);\n";
  os << "  uint32_t scratch_l1_addr = get_arg_val<uint32_t>(3);\n\n";
  os << "  constexpr uint32_t TILE_SIZE = " << tile_size << ";\n";
  os << "  InterleavedAddrGen<true> src_gen = {\n";
  os << "      .bank_base_address = src_dram_addr,\n";
  os << "      .page_size = TILE_SIZE};\n";
  os << "  InterleavedAddrGen<true> dst_gen = {\n";
  os << "      .bank_base_address = dst_dram_addr,\n";
  os << "      .page_size = TILE_SIZE};\n\n";
  os << "  for (uint32_t i = 0; i < num_tiles; ++i) {\n";
  os << "    uint64_t src_noc_addr = get_noc_addr(i, src_gen);\n";
  os << "    noc_async_read(src_noc_addr, scratch_l1_addr, TILE_SIZE);\n";
  os << "    noc_async_read_barrier();\n";
  os << "    uint64_t dst_noc_addr = get_noc_addr(i, dst_gen);\n";
  os << "    noc_async_write(scratch_l1_addr, dst_noc_addr, TILE_SIZE);\n";
  os << "    noc_async_write_barrier();\n";
  os << "  }\n";
  os << "}\n";
  return os.str();
}

/*!
 * \brief Extract executable specs for the Blackhole backend.
 * \param mod The IR module
 * \return Map from function name to ExecutableSpec
 */
static std::unordered_map<std::string, ExecutableSpec> ExtractBlackholeFuncInfo(
    const IRModule& mod) {
  std::unordered_map<std::string, ExecutableSpec> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    ExecutableSpec spec;

    // Extract argument types and buffer flags
    for (size_t i = 0; i < f->params.size(); ++i) {
      DLDataType dtype = f->params[i]->dtype;
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.code == kDLBool) {
        dtype.code = kDLInt;
        dtype.bits = 32;
      }
      spec.tvm_arg_types.push_back(dtype);

      // Check if this is a buffer argument (handle type)
      spec.tvm_is_buffer_arg.push_back(dtype.code == kDLOpaqueHandle);
    }

    // Extract CB configuration
    spec.cb_configs = ExtractCBConfig(f);
    spec.core_plan = ExtractCorePlan(f);
    spec.target_mode = f->GetAttr<ffi::String>("blackhole.target_mode")
                           .value_or(ffi::String("single_core_copy"));
    spec.runtime_args = ExtractRuntimeArgs(f, spec);

    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    if (global_symbol) {
      spec.entry_name = static_cast<std::string>(global_symbol.value());
      fmap[spec.entry_name] = spec;
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

  auto func_info_map = ExtractBlackholeFuncInfo(mod);

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

  // Process non-copy functions through generic codegen.
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    std::string func_name = global_symbol ? static_cast<std::string>(global_symbol.value())
                                          : static_cast<std::string>(gvar->name_hint);
    auto fit = func_info_map.find(func_name);
    if (fit != func_info_map.end() && fit->second.target_mode == "single_core_copy") {
      continue;
    }

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

  // Attach generated source to the Stage 0 executable spec.
  for (auto& kv : func_info_map) {
    KernelSpec kernel;
    kernel.name = kv.first;
    kernel.kind = "fused_dataflow";
    kernel.core_type = "brisc";
    kernel.source_code = kv.second.target_mode == "single_core_copy"
                             ? EmitSingleCoreCopyKernelSource(kv.second)
                             : code;
    kernel.runtime_args = kv.second.runtime_args.empty() ? MakeDefaultRuntimeArgs(kv.second)
                                                         : kv.second.runtime_args;
    kv.second.kernels.push_back(std::move(kernel));
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

  auto func_info_map = ExtractBlackholeFuncInfo(mod);

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

  // Process only non-copy device kernel functions through generic codegen.
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    std::string func_name = global_symbol ? static_cast<std::string>(global_symbol.value())
                                          : static_cast<std::string>(gvar->name_hint);
    auto fit = func_info_map.find(func_name);
    if (fit != func_info_map.end() && fit->second.target_mode == "single_core_copy") {
      continue;
    }

    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // Process device kernels (kDeviceKernelLaunch) OR functions without calling_conv set
    // (which is the case for Blackhole device functions from tilelang.transform flow)
    if (!calling_conv.defined() || calling_conv == CallingConv::kDeviceKernelLaunch) {
      cg.AddFunction(gvar, f);
    }
  }

  // Get pure kernel code (without TVM headers) for TT-Metal compilation
  std::string kernel_code = cg.GetKernelCode();

  // Store pure kernel code in the executable spec.
  for (auto& kv : func_info_map) {
    KernelSpec kernel;
    kernel.name = kv.first;
    kernel.kind = "fused_dataflow";
    kernel.core_type = "brisc";
    kernel.source_code = kv.second.target_mode == "single_core_copy"
                             ? EmitSingleCoreCopyKernelSource(kv.second)
                             : kernel_code;
    kernel.runtime_args = kv.second.runtime_args.empty() ? MakeDefaultRuntimeArgs(kv.second)
                                                         : kv.second.runtime_args;
    kv.second.kernels.push_back(std::move(kernel));
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
