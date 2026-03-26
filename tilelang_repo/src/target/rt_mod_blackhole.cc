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
  if (auto v = attrs.Get("ab_dtype")) {
    contract.ab_dtype = Downcast<String>(v.value());
  }
  if (auto v = attrs.Get("c_dtype")) {
    contract.c_dtype = Downcast<String>(v.value());
  }

  contract.enabled = !contract.a_buffer.empty() && !contract.b_buffer.empty() &&
                     !contract.c_buffer.empty() && contract.M > 0 && contract.N > 0 &&
                     contract.K > 0;
  return contract;
}

static std::vector<KernelArgSpec> MakeDefaultCopyRuntimeArgs() {
  return {
      {"input0", "input_buffer_addr32", "uint32", ""},
      {"output0", "output_buffer_addr32", "uint32", ""},
      {"current_work_linear_id", "current_work_linear_id", "uint32", ""},
      {"num_tiles", "tile_count", "uint32", ""},
      {"scratch_l1", "scratch_l1_buffer_addr32", "uint32", ""},
  };
}

static bool HasCopyRuntimeArgSchema(const std::vector<KernelArgSpec>& runtime_args) {
  if (runtime_args.size() != 5) {
    return false;
  }
  return runtime_args[0].kind == "input_buffer_addr32" &&
         runtime_args[1].kind == "output_buffer_addr32" &&
         runtime_args[2].kind == "current_work_linear_id" &&
         runtime_args[3].kind == "tile_count" &&
         runtime_args[4].kind == "scratch_l1_buffer_addr32";
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
        if (!is_buffer_arg || arg.buffer.empty() || seen.count(arg.buffer)) {
          continue;
        }
        aggregated.push_back(arg);
        seen.insert(arg.buffer);
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

struct SegmentInfo {
  std::string name;
  std::string kind;
  std::string core_type;
  std::vector<KernelArgSpec> runtime_args;
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

static tir::PrimFunc MakeSegmentPrimFunc(const tir::PrimFunc& f, const SegmentInfo& segment) {
  SegmentBodyExtractor extractor(segment.kind);
  tir::PrimFunc segment_func = f;
  segment_func.CopyOnWrite()->body = extractor(f->body);

  ffi::Map<ffi::String, ffi::Any> attrs;
  if (f->attrs.defined()) {
    attrs = f->attrs->dict;
  }
  attrs.Set("blackhole.core_type", ffi::String(segment.core_type));
  attrs.Set("blackhole.runtime_args", EncodeRuntimeArgs(segment.runtime_args));
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
  const bool use_legacy_single_kernel_path =
      segments.empty() ||
      (segments.size() == 1 && segments[0].kind == "fused_dataflow");
  if (use_legacy_single_kernel_path) {
    KernelSpec kernel;
    kernel.name = func_name;
    kernel.kind = spec->default_kernel_kind;
    kernel.core_type = spec->default_kernel_core_type;
    kernel.runtime_args = spec->runtime_args;
    kernel.source_code =
        (!legacy_code.empty()) ? legacy_code
                             : (HasCopyRuntimeArgSchema(kernel.runtime_args)
                                    ? EmitSingleCoreCopyKernelSource(*spec)
                                    : std::string());
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
    kernel.runtime_args = segment.runtime_args;
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
