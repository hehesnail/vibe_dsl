/*!
 * \file target/blackhole_module.cc
 * \brief Unified Blackhole module implementation
 *
 * This file provides the direct TT-Metal execution path for Blackhole kernels.
 */

#include "blackhole_module.h"

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <unordered_map>

#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"

#ifdef TILELANG_BLACKHOLE_DIRECT
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tilize_utils.hpp>
#endif

namespace tvm {
namespace runtime {

static std::string NormalizeBlackholeKernelSource(std::string source) {
  const std::string old_compute_include = "#include \"compute_kernel_api.h\"";
  const std::string new_compute_include = "#include \"api/compute/compute_kernel_api.h\"";
  size_t pos = source.find(old_compute_include);
  if (pos != std::string::npos) {
    source.replace(pos, old_compute_include.size(), new_compute_include);
  }
  return source;
}

// Forward declaration
class BlackholeWrappedFunc;

// ============================================================================
// BlackholeWrappedFunc declaration
// ============================================================================

class BlackholeWrappedFunc {
 public:
  void Init(BlackholeModuleNode* m, ObjectPtr<Object> sptr,
            const std::string& func_name,
            const ExecutableSpec& info) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    info_ = info;
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const;

 private:
  BlackholeModuleNode* m_;
  ObjectPtr<Object> sptr_;
  std::string func_name_;
  ExecutableSpec info_;
};

// Argument extraction helpers
uint32_t ExtractScalar(const ffi::AnyView& arg, DLDataType dtype) {
  if (dtype.code == kDLInt) {
    auto opt_i32 = arg.try_cast<int32_t>();
    if (opt_i32.has_value()) {
      return static_cast<uint32_t>(opt_i32.value());
    }
    auto opt_i64 = arg.try_cast<int64_t>();
    if (opt_i64.has_value()) {
      return static_cast<uint32_t>(opt_i64.value());
    }
  }
  if (dtype.code == kDLUInt) {
    auto opt_u32 = arg.try_cast<uint32_t>();
    if (opt_u32.has_value()) {
      return opt_u32.value();
    }
    auto opt_u64 = arg.try_cast<uint64_t>();
    if (opt_u64.has_value()) {
      return static_cast<uint32_t>(opt_u64.value());
    }
  }
  if (dtype.code == kDLFloat) {
    float f = 0.0f;
    auto opt_f = arg.try_cast<float>();
    if (opt_f.has_value()) {
      f = opt_f.value();
    } else {
      auto opt_d = arg.try_cast<double>();
      if (opt_d.has_value()) {
        f = static_cast<float>(opt_d.value());
      }
    }
    return *reinterpret_cast<uint32_t*>(&f);
  }
  LOG(FATAL) << "Cannot extract scalar of type code " << dtype.code;
  return 0;
}

DLTensor* ExtractTensorArg(const ffi::AnyView& arg, void* void_arg) {
  auto opt_tensor = arg.try_cast<DLTensor*>();
  if (opt_tensor.has_value()) {
    return opt_tensor.value();
  }
  if (void_arg != nullptr) {
    DLTensor* tensor = *reinterpret_cast<DLTensor**>(void_arg);
    if (tensor != nullptr) {
      return tensor;
    }
  }
  LOG(FATAL) << "Cannot extract DLTensor* from packed argument";
  return nullptr;
}

// ============================================================================
// Direct TT-Metal path helpers (only when linked against TT-Metal)
// ============================================================================

#ifdef TILELANG_BLACKHOLE_DIRECT

using namespace tt::tt_metal;

template <typename T>
static std::vector<T> TransposeRowMajor2D(const T* src, uint32_t rows, uint32_t cols) {
  std::vector<T> out(static_cast<size_t>(rows) * cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
    }
  }
  return out;
}

static bool IsTwoDimTensor(const DLTensor* tensor) {
  return tensor != nullptr && tensor->ndim == 2;
}

static std::pair<uint32_t, uint32_t> GetTensorShape2D(const DLTensor* tensor) {
  ICHECK(IsTwoDimTensor(tensor));
  return {static_cast<uint32_t>(tensor->shape[0]), static_cast<uint32_t>(tensor->shape[1])};
}

static void ValidateGemmInputShape(const ExecutableSpec& spec,
                                   const RuntimeTensorBinding& binding,
                                   uint32_t rows,
                                   uint32_t cols) {
  const auto& gemm = spec.gemm_contract;
  const uint32_t logical_grid_x = std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
  const uint32_t logical_grid_y = std::max<uint32_t>(1, spec.core_plan.logical_grid_y);

  if (binding.name == gemm.a_buffer) {
    const uint32_t expected_rows = gemm.transpose_A ? gemm.K * logical_grid_y
                                                    : gemm.M * logical_grid_y;
    const uint32_t expected_cols = gemm.transpose_A ? gemm.M : gemm.K;
    ICHECK(rows == expected_rows && cols == expected_cols)
        << "Unexpected A tensor shape for GEMM direct path: got (" << rows << ", " << cols
        << "), expected (" << expected_rows << ", " << expected_cols
        << ") for logical grid " << logical_grid_y << "x" << logical_grid_x;
    return;
  }

  if (binding.name == gemm.b_buffer) {
    const uint32_t expected_rows = gemm.transpose_B ? gemm.N * logical_grid_x
                                                    : gemm.K * logical_grid_x;
    const uint32_t expected_cols = gemm.transpose_B ? gemm.K : gemm.N;
    ICHECK(rows == expected_rows && cols == expected_cols)
        << "Unexpected B tensor shape for GEMM direct path: got (" << rows << ", " << cols
        << "), expected (" << expected_rows << ", " << expected_cols
        << ") for transpose_B=" << gemm.transpose_B
        << " and logical grid " << logical_grid_x << "x" << logical_grid_y;
    return;
  }
}

static void ValidateGemmOutputShape(const ExecutableSpec& spec,
                                    uint32_t rows,
                                    uint32_t cols) {
  const auto& gemm = spec.gemm_contract;
  const uint32_t logical_grid_x = std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
  const uint32_t logical_grid_y = std::max<uint32_t>(1, spec.core_plan.logical_grid_y);
  const uint32_t expected_rows = gemm.M * logical_grid_y;
  const uint32_t expected_cols = gemm.N * logical_grid_x;
  ICHECK(rows == expected_rows && cols == expected_cols)
      << "Unexpected C tensor shape for GEMM direct path: got (" << rows << ", " << cols
      << "), expected (" << expected_rows << ", " << expected_cols
      << ") for logical grid " << logical_grid_x << "x" << logical_grid_y;
}

static std::vector<uint8_t> BuildInputTransferData(const ExecutableSpec& spec,
                                                   const RuntimeTensorBinding& binding) {
  const DLTensor* tensor = binding.tensor;
  ICHECK(tensor != nullptr);
  const size_t tensor_size = GetDataSize(*tensor);

  if (!spec.gemm_contract.enabled || !IsTwoDimTensor(tensor)) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), tensor->data, tensor_size);
    return raw;
  }

  const auto& gemm = spec.gemm_contract;
  if (binding.name != gemm.a_buffer && binding.name != gemm.b_buffer) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), tensor->data, tensor_size);
    return raw;
  }

  ICHECK(tensor->dtype.bits == 16)
      << "Only 16-bit GEMM inputs are currently supported for Blackhole host tilize";
  const auto* raw = static_cast<const uint16_t*>(tensor->data);
  const auto [rows, cols] = GetTensorShape2D(tensor);
  ValidateGemmInputShape(spec, binding, rows, cols);

  std::vector<uint16_t> tiled;
  if (binding.name == gemm.b_buffer && gemm.transpose_B) {
    tiled = tilize_nfaces(TransposeRowMajor2D(raw, rows, cols), cols, rows);
  } else {
    std::vector<uint16_t> row_major(raw, raw + static_cast<size_t>(rows) * cols);
    tiled = tilize_nfaces(row_major, rows, cols);
  }

  std::vector<uint8_t> bytes(tiled.size() * sizeof(uint16_t));
  std::memcpy(bytes.data(), tiled.data(), bytes.size());
  return bytes;
}

static void CopyOutputFromDeviceBuffer(const ExecutableSpec& spec,
                                       const RuntimeTensorBinding& binding,
                                       const std::vector<uint8_t>& output_data) {
  ICHECK(binding.tensor != nullptr);
  const size_t tensor_size = GetDataSize(*binding.tensor);
  ICHECK(output_data.size() >= tensor_size)
      << "Output data size mismatch for " << binding.name;

  if (!spec.gemm_contract.enabled || binding.name != spec.gemm_contract.c_buffer ||
      !IsTwoDimTensor(binding.tensor)) {
    std::memcpy(binding.tensor->data, output_data.data(), tensor_size);
    return;
  }

  ICHECK(binding.tensor->dtype.code == kDLFloat && binding.tensor->dtype.bits == 32)
      << "Only float32 GEMM outputs are currently supported for Blackhole host untilize";
  const auto [rows, cols] = GetTensorShape2D(binding.tensor);
  ValidateGemmOutputShape(spec, rows, cols);
  const size_t numel = static_cast<size_t>(rows) * cols;
  ICHECK(output_data.size() == numel * sizeof(float))
      << "Unexpected GEMM output buffer size for " << binding.name << ": got "
      << output_data.size() << " bytes, expected " << (numel * sizeof(float));
  const auto* tiled = reinterpret_cast<const float*>(output_data.data());
  std::vector<float> tiled_vec(tiled, tiled + numel);
  std::vector<float> row_major = untilize_nfaces(tiled_vec, rows, cols);
  std::memcpy(binding.tensor->data, row_major.data(), row_major.size() * sizeof(float));
}

static tt::DataFormat ParseDataFormat(const std::string& value) {
  if (value == "Float16" || value == "Float16_b") return tt::DataFormat::Float16_b;
  if (value == "Float32") return tt::DataFormat::Float32;
  if (value == "UInt16") return tt::DataFormat::UInt16;
  if (value == "UInt32") return tt::DataFormat::UInt32;
  LOG(FATAL) << "Unsupported data format: " << value;
  return tt::DataFormat::Float16_b;
}

static uint32_t ChoosePageSize(const ExecutableSpec& spec, const std::string& role) {
  for (const auto& cb : spec.cb_configs) {
    if (cb.role == role) return cb.page_size_bytes;
  }
  if (!spec.cb_configs.empty()) return spec.cb_configs.front().page_size_bytes;
  return 2048;
}

static uint32_t GetRuntimeNumKTiles(const ExecutableSpec& spec) {
  if (spec.gemm_contract.enabled) {
    constexpr uint32_t kBlackholeTileCols = 32;
    ICHECK_EQ(spec.gemm_contract.K % kBlackholeTileCols, 0)
        << "Blackhole GEMM direct path requires K to be 32-tile aligned";
    return std::max<uint32_t>(1, spec.gemm_contract.K / kBlackholeTileCols);
  }
  return 0;
}

static void CreateCircularBuffersFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const ExecutableSpec& spec) {
  for (const auto& cb : spec.cb_configs) {
    uint32_t total_size = cb.num_pages * cb.page_size_bytes;
    CircularBufferConfig cb_config(
        total_size,
        {{static_cast<uint8_t>(cb.cb_id), ParseDataFormat(cb.data_format)}});
    cb_config.set_page_size(static_cast<uint8_t>(cb.cb_id), cb.page_size_bytes);
    CreateCircularBuffer(program, core_spec, cb_config);
  }
}

static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel, const std::string& kernel_path) {
  if (kernel.core_type == "trisc" || kernel.kind == "compute") {
    return CreateKernel(
        program,
        kernel_path,
        core_spec,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = kernel.compile_time_args});
  }

  DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
  NOC noc = NOC::RISCV_0_default;
  if (kernel.core_type == "ncrisc") {
    processor = DataMovementProcessor::RISCV_1;
    noc = NOC::RISCV_1_default;
  }

  return CreateKernel(
      program,
      kernel_path,
      core_spec,
      DataMovementConfig{
          .processor = processor,
          .noc = noc,
          .compile_args = kernel.compile_time_args});
}

struct RuntimeBufferBinding {
  std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
  size_t size_bytes{0};
  bool is_output{false};
};

static const RuntimeBufferBinding& ResolveRuntimeBufferBinding(
    const KernelArgSpec& arg_spec,
    bool expect_output,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::vector<std::string>& ordered_names) {
  if (!arg_spec.buffer.empty()) {
    auto it = buffer_bindings.find(arg_spec.buffer);
    ICHECK(it != buffer_bindings.end())
        << "Missing runtime buffer binding for " << arg_spec.buffer;
    ICHECK(it->second.is_output == expect_output)
        << "Runtime buffer role mismatch for " << arg_spec.buffer
        << ": expected output=" << expect_output;
    return it->second;
  }

  ICHECK(!ordered_names.empty())
      << "No runtime buffer binding available for arg kind " << arg_spec.kind;
  auto it = buffer_bindings.find(ordered_names.front());
  ICHECK(it != buffer_bindings.end())
      << "Missing fallback runtime buffer binding for " << ordered_names.front();
  return it->second;
}

static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<uint32_t>& scalar_args) {
  std::vector<uint32_t> args;
  size_t scalar_index = 0;

  for (const auto& arg_spec : kernel.runtime_args) {
    if (arg_spec.kind == "input_buffer_addr") {
      const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/false,
                                                        buffer_bindings, input_names);
      const uint64_t src_addr = binding.mesh_buffer->address();
      args.push_back(static_cast<uint32_t>(src_addr & 0xFFFFFFFF));
      args.push_back(static_cast<uint32_t>(src_addr >> 32));
    } else if (arg_spec.kind == "input_buffer_addr32") {
      const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/false,
                                                        buffer_bindings, input_names);
      const uint64_t src_addr = binding.mesh_buffer->address();
      args.push_back(static_cast<uint32_t>(src_addr));
    } else if (arg_spec.kind == "output_buffer_addr") {
      const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/true,
                                                        buffer_bindings, output_names);
      const uint64_t dst_addr = binding.mesh_buffer->address();
      args.push_back(static_cast<uint32_t>(dst_addr & 0xFFFFFFFF));
      args.push_back(static_cast<uint32_t>(dst_addr >> 32));
    } else if (arg_spec.kind == "output_buffer_addr32") {
      const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/true,
                                                        buffer_bindings, output_names);
      const uint64_t dst_addr = binding.mesh_buffer->address();
      args.push_back(static_cast<uint32_t>(dst_addr));
    } else if (arg_spec.kind == "tile_count") {
      const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/false,
                                                        buffer_bindings, input_names);
      const uint32_t tile_size = ChoosePageSize(spec, "input");
      args.push_back(tile_size == 0 ? 0
                                    : static_cast<uint32_t>(binding.size_bytes / tile_size));
    } else if (arg_spec.kind == "num_k_tiles") {
      uint32_t num_k_tiles = GetRuntimeNumKTiles(spec);
      if (num_k_tiles == 0) {
        const auto& binding = ResolveRuntimeBufferBinding(arg_spec, /*expect_output=*/false,
                                                          buffer_bindings, input_names);
        const uint32_t tile_size = ChoosePageSize(spec, "input");
        num_k_tiles =
            tile_size == 0 ? 0 : static_cast<uint32_t>(binding.size_bytes / tile_size);
      }
      args.push_back(num_k_tiles);
    } else if (arg_spec.kind == "current_work_linear_id") {
      args.push_back(current_work_linear_id);
    } else if (arg_spec.kind == "scalar_u32") {
      ICHECK(scalar_index < scalar_args.size())
          << "Spec requested more scalar args than provided";
      args.push_back(scalar_args[scalar_index++]);
    } else {
      LOG(FATAL) << "Unsupported runtime arg kind: " << arg_spec.kind;
    }
  }

  return args;
}

#endif  // TILELANG_BLACKHOLE_DIRECT

// ============================================================================
// BlackholeModuleNode implementation
// ============================================================================

BlackholeModuleNode::BlackholeModuleNode(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir)
    : fmap_(std::move(fmap)),
      kernel_dir_(std::move(kernel_dir)) {
}


ffi::Optional<ffi::Function> BlackholeModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);

  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return ffi::Function();
  }

  const ExecutableSpec& info = it->second;
  BlackholeWrappedFunc f;
  f.Init(this, sptr_to_self, name, info);

  std::vector<FunctionInfo::ArgExtraTags> arg_extra_tags;
  return PackFuncVoidAddr(f, info.tvm_arg_types, arg_extra_tags);
}

void BlackholeModuleNode::WriteToFile(const ffi::String& file_name,
                                      const ffi::String& format) const {
  LOG(WARNING) << "BlackholeModule WriteToFile not yet implemented";
}

ffi::Bytes BlackholeModuleNode::SaveToBytes() const {
  LOG(WARNING) << "BlackholeModule SaveToBytes not yet implemented";
  return ffi::Bytes("");
}

ffi::String BlackholeModuleNode::InspectSource(const ffi::String& format) const {
  LOG(INFO) << "BlackholeModuleNode::InspectSource called";
  auto it = fmap_.find("default");
  if (it != fmap_.end()) {
    const auto& spec = it->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Found 'default' function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  if (!fmap_.empty()) {
    const auto& spec = fmap_.begin()->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Using first function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  LOG(WARNING) << "No functions found in fmap_";
  return ffi::String("");
}

// ============================================================================
// Unique temp-directory helper (used by both execution paths)
// ============================================================================

static std::string MakeUniqueTempDir(const std::string& prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  std::filesystem::path dir = std::filesystem::temp_directory_path() /
                              (prefix + std::to_string(getpid()) + "_" + std::to_string(id));
  std::filesystem::create_directories(dir);
  return dir.string();
}

// ============================================================================
// Direct TT-Metal execution path
// ============================================================================

void BlackholeModuleNode::ExecuteDirect(
    const std::string& func_name,
    const std::vector<RuntimeTensorBinding>& buffer_args,
    const std::vector<uint32_t>& scalar_args,
    const std::vector<std::string>& output_names) {
#ifdef TILELANG_BLACKHOLE_DIRECT
  using namespace tt::tt_metal;

  auto fit = fmap_.find(func_name);
  if (fit == fmap_.end()) {
    LOG(FATAL) << "Function not found: " << func_name;
  }
  const ExecutableSpec& spec = fit->second;
  if (spec.kernels.empty()) {
    LOG(FATAL) << "ExecutableSpec has no kernels for function: " << func_name;
  }

  // Use role-aware page size for DRAM buffers so runtime allocation matches spec roles.
  uint32_t input_page_size = ChoosePageSize(spec, "input");
  uint32_t output_page_size = ChoosePageSize(spec, "output");

  // Build work items: pair each logical work_id with its assigned physical core.
  // Each WorkPacket entry owns a slice of the logical work range on one core.
  struct WorkItem {
    uint32_t work_id;
    CoreCoord core;
  };
  std::vector<WorkItem> work_items;
  for (const auto& packet : spec.core_plan.work_packets) {
    CoreCoord packet_core{packet.core_x, packet.core_y};
    for (uint32_t i = 0; i < packet.work_count; ++i) {
      work_items.push_back({packet.work_offset + i, packet_core});
    }
  }
  if (work_items.empty()) {
    // Fallback: derive core from physical_cores if available, else use mapping default {1,2}.
    CoreCoord fallback = spec.core_plan.physical_cores.empty()
        ? CoreCoord{1, 2}
        : CoreCoord{spec.core_plan.physical_cores[0].core_x,
                    spec.core_plan.physical_cores[0].core_y};
    work_items.push_back({0, fallback});
  }

  std::vector<CoreCoord> launch_cores;
  launch_cores.reserve(work_items.size());
  for (const auto& item : work_items) {
    launch_cores.push_back(item.core);
  }
  std::sort(launch_cores.begin(), launch_cores.end());
  launch_cores.erase(std::unique(launch_cores.begin(), launch_cores.end()), launch_cores.end());
  ICHECK(!launch_cores.empty()) << "No launch cores resolved for direct execution";
  ICHECK(launch_cores.size() == work_items.size())
      << "Blackhole direct runtime supports only one logical work item per physical core in a "
         "single launch. Function "
      << func_name << " maps " << work_items.size() << " logical work items onto "
      << launch_cores.size() << " unique cores; oversubscribed direct launch is not supported.";
  const CoreRangeSet launch_core_ranges(launch_cores);

  // Keep direct execution hermetic per call. Reusing a persistent MeshDevice across
  // multiple Python direct-call tests can leave simulator/device state behind and
  // cause cross-test contamination across cases.
  LOG(INFO) << "Initializing Blackhole TT-Metal device...";
  std::shared_ptr<distributed::MeshDevice> mesh_device;
  try {
    mesh_device = distributed::MeshDevice::create_unit_mesh(0);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Blackhole device: " << e.what();
  }
  ICHECK(mesh_device != nullptr);
  LOG(INFO) << "Blackhole device initialized successfully";

  distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

  std::unordered_map<std::string, RuntimeBufferBinding> runtime_buffers;
  std::vector<std::string> input_names;
  std::vector<std::string> ordered_output_names;
  for (const auto& binding : buffer_args) {
    ICHECK(binding.tensor != nullptr) << "Null tensor passed to Blackhole direct path";
    const size_t tensor_size = GetDataSize(*binding.tensor);
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = binding.is_output ? output_page_size : input_page_size,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = tensor_size};
    auto mesh_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    runtime_buffers.emplace(binding.name, RuntimeBufferBinding{
                                              .mesh_buffer = mesh_buffer,
                                              .size_bytes = tensor_size,
                                              .is_output = binding.is_output,
                                          });
    if (binding.is_output) {
      ordered_output_names.push_back(binding.name);
    } else {
      input_names.push_back(binding.name);
      std::vector<uint8_t> input_data = BuildInputTransferData(spec, binding);
      EnqueueWriteMeshBuffer(cq, mesh_buffer, input_data, /*blocking=*/true);
    }
  }
  if (!output_names.empty()) {
    ordered_output_names = output_names;
  }

  // Write kernel source files to temp directory
  std::string tmp_dir = MakeUniqueTempDir("tilelang_bh_direct_");

  std::vector<std::string> kernel_paths;
  kernel_paths.reserve(spec.kernels.size());
  for (size_t i = 0; i < spec.kernels.size(); ++i) {
    const auto& kernel = spec.kernels[i];
    std::string kernel_path = tmp_dir + "/" + func_name + "_" + std::to_string(i) + "_" +
                              kernel.kind + ".cpp";
    std::ofstream ofs(kernel_path);
    if (!ofs) {
      LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
    }
    ofs << NormalizeBlackholeKernelSource(kernel.source_code);
    kernel_paths.push_back(kernel_path);
  }

  LOG(INFO) << "Direct path: executing " << work_items.size()
            << " logical work items across " << launch_cores.size()
            << " launch cores for " << func_name;

  Program program = CreateProgram();

  // Materialize shared CBs and kernels once for the full launch core set.
  CreateCircularBuffersFromSpec(program, launch_core_ranges, spec);

  std::vector<KernelHandle> kernels;
  kernels.reserve(spec.kernels.size());
  for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    const auto& kernel_spec = spec.kernels[ki];
    LOG(INFO) << "Direct path: create kernel[" << ki << "] kind=" << kernel_spec.kind
              << " core_type=" << kernel_spec.core_type;
    kernels.push_back(CreateKernelFromSpec(program, launch_core_ranges, kernel_spec,
                                           kernel_paths[ki]));
  }

  // Keep runtime args per core/work-item so each logical work item sees its own ID.
  for (const auto& item : work_items) {
    LOG(INFO) << "Direct path: configure work_id=" << item.work_id
              << " core=(" << item.core.x << "," << item.core.y << ")";
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
      const auto& kernel_spec = spec.kernels[ki];
      auto runtime_args = BuildRuntimeArgsFromSpec(
          kernel_spec, spec, item.work_id, runtime_buffers, input_names, ordered_output_names,
          scalar_args);

      LOG(INFO) << "Direct path: set runtime args kernel[" << ki
                << "] core=(" << item.core.x << "," << item.core.y
                << ") count=" << runtime_args.size();
      SetRuntimeArgs(program, kernels[ki], item.core, runtime_args);
    }
  }

  // Execute the full program once across the multi-core launch set.
  distributed::MeshWorkload workload;
  distributed::MeshCoordinateRange device_range(mesh_device->shape());
  workload.add_program(device_range, std::move(program));
  LOG(INFO) << "Direct path: enqueue multi-core workload for " << func_name;
  distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

  // Read back results
  for (const auto& binding : buffer_args) {
    if (!binding.is_output) {
      continue;
    }
    auto it = runtime_buffers.find(binding.name);
    ICHECK(it != runtime_buffers.end())
        << "Missing runtime output binding for " << binding.name;
    std::vector<uint8_t> output_data;
    distributed::EnqueueReadMeshBuffer(cq, output_data, it->second.mesh_buffer,
                                       /*blocking=*/true);
    CopyOutputFromDeviceBuffer(spec, binding, output_data);
  }

  // Cleanup kernel temp files
  std::filesystem::remove_all(tmp_dir);

  LOG(INFO) << "Direct path execution completed for " << func_name;

#else
  LOG(FATAL) << "Direct TT-Metal path not available. "
             << "Rebuild with TILELANG_BLACKHOLE_DIRECT=ON.";
#endif  // TILELANG_BLACKHOLE_DIRECT
}

// ============================================================================
// Execution dispatch
// ============================================================================

void BlackholeWrappedFunc::operator()(ffi::PackedArgs args, ffi::Any* rv,
                                       void** void_args) const {
  // Classify buffer args as input or output using runtime_args kind info.
  // runtime_args lists buffer-role entries in the same order they appear in
  // tvm_is_buffer_arg, interleaved with non-buffer entries that are skipped.
  std::vector<bool> buffer_is_output;
  for (const auto& arg : info_.runtime_args) {
    if (arg.kind == "input_buffer_addr32" || arg.kind == "input_buffer_addr") {
      buffer_is_output.push_back(false);
    } else if (arg.kind == "output_buffer_addr32" || arg.kind == "output_buffer_addr") {
      buffer_is_output.push_back(true);
    }
  }
  // Fallback: if runtime_args carries no buffer kind info, treat last buffer as output.
  const bool use_position_fallback = buffer_is_output.empty();
  size_t n_buffer_args = 0;
  for (bool is_buf : info_.tvm_is_buffer_arg) {
    if (is_buf) ++n_buffer_args;
  }

  // Collect arguments
  std::vector<RuntimeTensorBinding> buffer_args;
  std::vector<uint32_t> scalars;
  std::vector<std::string> output_names;

  size_t buf_idx = 0;
  for (size_t i = 0; i < info_.tvm_arg_types.size(); ++i) {
    if (info_.tvm_is_buffer_arg[i]) {
      DLTensor* tensor = ExtractTensorArg(args[i], void_args != nullptr ? void_args[i] : nullptr);
      bool is_out = use_position_fallback
          ? (buf_idx == n_buffer_args - 1)
          : (buf_idx < buffer_is_output.size() && buffer_is_output[buf_idx]);
      const std::string buffer_name =
          (i < info_.tvm_arg_names.size() && !info_.tvm_arg_names[i].empty())
              ? info_.tvm_arg_names[i]
              : ("arg" + std::to_string(i));
      buffer_args.push_back(RuntimeTensorBinding{buffer_name, tensor, is_out});
      if (is_out) {
        output_names.push_back(buffer_name);
      }
      ++buf_idx;
    } else {
      ffi::AnyView arg = args[i];
      uint32_t val = ExtractScalar(arg, info_.tvm_arg_types[i]);
      scalars.push_back(val);
    }
  }

  m_->ExecuteDirect(func_name_, buffer_args, scalars, output_names);
}

// ============================================================================
// Module creation and registration
// ============================================================================

ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir) {
  auto n = ffi::make_object<BlackholeModuleNode>(std::move(fmap), std::move(kernel_dir));
  return ffi::Module(std::move(n));
}

ffi::Module BlackholeModuleLoadFromBytes(const ffi::Bytes& bytes) {
  LOG(FATAL) << "BlackholeModule LoadFromBytes not yet implemented";
  __builtin_unreachable();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.module.loadbinary_blackhole", BlackholeModuleLoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm
