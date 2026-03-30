/*!
 * \file target/blackhole_module.cc
 * \brief Unified Blackhole module implementation
 *
 * This file provides the direct TT-Metal execution path for Blackhole kernels.
 */

#include "blackhole_module.h"

#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>

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
#include <tt-metalium/tensor_accessor_args.hpp>
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

static std::string EncodeExecutableSpecMetadata(const ExecutableSpec& spec) {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  spec.Save(&writer);
  return os.str();
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

static std::string DLTensorDataFormat(const DLTensor& tensor) {
  const DLDataType& dtype = tensor.dtype;
  if (dtype.code == kDLBfloat && dtype.bits == 16) return "Float16_b";
  if (dtype.code == kDLFloat && dtype.bits == 16) return "Float16";
  if (dtype.code == kDLFloat && dtype.bits == 32) return "Float32";
  if (dtype.code == kDLUInt && dtype.bits == 16) return "UInt16";
  if (dtype.code == kDLUInt && dtype.bits == 32) return "UInt32";
  if (dtype.code == kDLInt && dtype.bits == 16) return "Int16";
  if (dtype.code == kDLInt && dtype.bits == 32) return "Int32";
  return "unknown";
}

static void ValidateGemmTensorDType(const RuntimeTensorBinding& binding,
                                    const std::string& expected_dtype) {
  ICHECK(binding.tensor != nullptr);
  const std::string actual_dtype = DLTensorDataFormat(*binding.tensor);
  ICHECK_EQ(actual_dtype, expected_dtype)
      << "Unexpected tensor dtype for GEMM binding " << binding.name << ": got " << actual_dtype
      << ", expected " << expected_dtype;
}

static ComputeContractSpec GetComputeContract(const ExecutableSpec& spec);

static void ValidateComputeContractDirectRuntimeConstraints(const ExecutableSpec& spec) {
  const auto contract = GetComputeContract(spec);
  if (!contract.enabled || contract.kind != "gemm") {
    return;
  }
  ICHECK(!contract.has_mbarrier)
      << "Blackhole direct runtime does not yet support GEMM compute_contract.mbarrier bindings";
}

static ComputeContractSpec GetComputeContract(const ExecutableSpec& spec) {
  if (spec.compute_contract.enabled) {
    return spec.compute_contract;
  }
  if (!spec.gemm_contract.enabled) {
    return ComputeContractSpec();
  }

  constexpr uint32_t kBlackholeTileRows = 32;
  constexpr uint32_t kBlackholeTileCols = 32;
  ComputeContractSpec contract;
  contract.enabled = true;
  contract.kind = "gemm";
  contract.a_buffer = spec.gemm_contract.a_buffer;
  contract.b_buffer = spec.gemm_contract.b_buffer;
  contract.c_buffer = spec.gemm_contract.c_buffer;
  contract.M = spec.gemm_contract.M;
  contract.N = spec.gemm_contract.N;
  contract.K = spec.gemm_contract.K;
  contract.Mt = spec.gemm_contract.M / kBlackholeTileRows;
  contract.Nt = spec.gemm_contract.N / kBlackholeTileCols;
  contract.Kt = spec.gemm_contract.K / kBlackholeTileCols;
  contract.transpose_A = spec.gemm_contract.transpose_A;
  contract.transpose_B = spec.gemm_contract.transpose_B;
  contract.a_tensor_dtype = spec.gemm_contract.a_tensor_dtype;
  contract.b_tensor_dtype = spec.gemm_contract.b_tensor_dtype;
  contract.c_tensor_dtype = spec.gemm_contract.c_tensor_dtype;
  contract.a_cb_dtype = spec.gemm_contract.a_cb_dtype;
  contract.b_cb_dtype = spec.gemm_contract.b_cb_dtype;
  contract.c_cb_dtype = spec.gemm_contract.c_cb_dtype;
  contract.accumulator_dtype = spec.gemm_contract.accumulator_dtype;
  return contract;
}

static void ValidateGemmInputShape(const ExecutableSpec& spec,
                                   const RuntimeTensorBinding& binding,
                                   uint32_t rows,
                                   uint32_t cols) {
  const auto gemm = GetComputeContract(spec);
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
  const auto gemm = GetComputeContract(spec);
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
  const auto gemm = GetComputeContract(spec);

  if (!gemm.enabled || gemm.kind != "gemm" || !IsTwoDimTensor(tensor)) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), tensor->data, tensor_size);
    return raw;
  }

  if (binding.name != gemm.a_buffer && binding.name != gemm.b_buffer) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), tensor->data, tensor_size);
    return raw;
  }

  const std::string expected_tensor_dtype =
      binding.name == gemm.a_buffer ? gemm.a_tensor_dtype : gemm.b_tensor_dtype;
  const std::string expected_cb_dtype =
      binding.name == gemm.a_buffer ? gemm.a_cb_dtype : gemm.b_cb_dtype;
  ValidateGemmTensorDType(binding, expected_tensor_dtype);
  ICHECK_EQ(expected_tensor_dtype, expected_cb_dtype)
      << "Blackhole direct GEMM currently requires identical tensor and CB dtype for "
      << binding.name;
  ICHECK_EQ(expected_tensor_dtype, "Float16_b")
      << "Blackhole direct GEMM currently supports only bfloat16 inputs, but " << binding.name
      << " requested " << expected_tensor_dtype;
  const auto* raw = static_cast<const uint16_t*>(tensor->data);
  const auto [rows, cols] = GetTensorShape2D(tensor);
  ValidateGemmInputShape(spec, binding, rows, cols);

  std::vector<uint16_t> tiled;
  if ((binding.name == gemm.a_buffer && gemm.transpose_A) ||
      (binding.name == gemm.b_buffer && gemm.transpose_B)) {
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
  const auto gemm = GetComputeContract(spec);

  if (!gemm.enabled || gemm.kind != "gemm" || binding.name != gemm.c_buffer ||
      !IsTwoDimTensor(binding.tensor)) {
    std::memcpy(binding.tensor->data, output_data.data(), tensor_size);
    return;
  }

  ValidateGemmTensorDType(binding, gemm.c_tensor_dtype);
  ICHECK_EQ(gemm.c_cb_dtype, gemm.accumulator_dtype)
      << "Blackhole direct GEMM currently requires identical output CB and accumulator dtypes";
  ICHECK_EQ(gemm.c_tensor_dtype, "Float32")
      << "Blackhole direct GEMM currently supports only float32 outputs, but "
      << gemm.c_buffer << " requested " << gemm.c_tensor_dtype;
  ICHECK_EQ(gemm.accumulator_dtype, "Float32")
      << "Blackhole direct GEMM currently supports only float32 accumulators, but requested "
      << gemm.accumulator_dtype;
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

static const BufferMaterializationSpec& ResolveBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name) {
  auto it = std::find_if(spec.buffer_materializations.begin(), spec.buffer_materializations.end(),
                         [&](const BufferMaterializationSpec& materialization) {
                           return materialization.buffer == buffer_name;
                         });
  ICHECK(it != spec.buffer_materializations.end())
      << "Missing Blackhole buffer materialization spec for buffer " << buffer_name;
  ICHECK_EQ(it->materialization_kind, "replicated")
      << "Unsupported Blackhole buffer materialization kind for " << buffer_name << ": "
      << it->materialization_kind;
  ICHECK_EQ(it->memory_space, "dram")
      << "Unsupported Blackhole buffer memory_space for " << buffer_name << ": "
      << it->memory_space;
  ICHECK_GT(it->transport_page_size_bytes, 0U)
      << "Blackhole buffer materialization requires transport_page_size for " << buffer_name;
  return *it;
}

static void ValidateSemaphoreCoreType(const std::string& core_type) {
  ICHECK(core_type.empty() || core_type == "worker")
      << "Unsupported Blackhole semaphore core_type: " << core_type;
}

static CoreRangeSet BuildSemaphoreCoreRangeSet(const SemaphoreSpec& semaphore) {
  std::vector<CoreRange> core_ranges;
  core_ranges.reserve(semaphore.core_ranges.size());
  for (const auto& range : semaphore.core_ranges) {
    const CoreCoord start{range.start.core_x, range.start.core_y};
    const CoreCoord end{range.end.core_x, range.end.core_y};
    core_ranges.emplace_back(start, end);
  }
  return CoreRangeSet(std::move(core_ranges));
}

static std::unordered_map<uint32_t, uint32_t> CreateSemaphoresFromSpec(Program& program,
                                                                       const ExecutableSpec& spec) {
  std::unordered_map<uint32_t, uint32_t> semaphore_ids;
  for (const auto& semaphore : spec.semaphores) {
    ICHECK(!semaphore.core_ranges.empty())
        << "Blackhole semaphore_plan entry id=" << semaphore.id
        << " must define at least one core range";
    ValidateSemaphoreCoreType(semaphore.core_type);
    const CoreRangeSet core_ranges = BuildSemaphoreCoreRangeSet(semaphore);
    uint32_t created_id = CreateSemaphore(program, core_ranges, semaphore.initial_value);
    ICHECK_EQ(created_id, semaphore.id)
        << "Blackhole semaphore_plan id mismatch: requested " << semaphore.id
        << ", TT-Metal allocated " << created_id;
    semaphore_ids.emplace(semaphore.id, created_id);
  }
  return semaphore_ids;
}

static uint32_t GetRuntimeNumKTiles(const ExecutableSpec& spec) {
  const auto gemm = GetComputeContract(spec);
  if (gemm.enabled && gemm.kind == "gemm") {
    ICHECK_GT(gemm.Kt, 0U)
        << "Blackhole GEMM direct path requires compute_contract.Kt to be populated";
    return std::max<uint32_t>(1, gemm.Kt);
  }
  return 0;
}

static uint32_t GetRuntimeLogicalGridX(const ExecutableSpec& spec) {
  return std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
}

static uint32_t GetRuntimeLogicalNTiles(const ExecutableSpec& spec) {
  const auto gemm = GetComputeContract(spec);
  ICHECK(gemm.enabled && gemm.kind == "gemm")
      << "logical_n_tiles is only defined for GEMM kernels in Blackhole direct runtime";
  ICHECK_GT(gemm.Nt, 0U)
      << "Blackhole GEMM direct path requires compute_contract.Nt to be populated";
  const uint32_t local_n_tiles = std::max<uint32_t>(1, gemm.Nt);
  const uint32_t logical_grid_x = GetRuntimeLogicalGridX(spec);
  return local_n_tiles * logical_grid_x;
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

struct RuntimeBufferBinding {
  std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
  size_t size_bytes{0};
  bool is_output{false};
};

static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::string& kernel_path);
static std::vector<uint32_t> BuildCommonRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names);
static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<uint32_t>& scalar_args);

struct DirectWorkItem {
  uint32_t work_id;
  CoreCoord core;
};

struct DirectRuntimeBufferState {
  std::unordered_map<std::string, RuntimeBufferBinding> runtime_buffers;
  std::vector<std::string> input_names;
  std::vector<std::string> ordered_output_names;
};

static std::vector<DirectWorkItem> BuildDirectWorkItems(const ExecutableSpec& spec,
                                                        const std::string& func_name) {
  std::vector<DirectWorkItem> work_items;
  for (const auto& packet : spec.core_plan.work_packets) {
    CoreCoord packet_core{packet.core_x, packet.core_y};
    for (uint32_t i = 0; i < packet.work_count; ++i) {
      work_items.push_back({packet.work_offset + i, packet_core});
    }
  }
  ICHECK(!work_items.empty())
      << "Blackhole planner/runtime contract requires non-empty work_items derived from "
         "core_plan.work_packets for "
      << func_name;
  return work_items;
}

static std::vector<CoreCoord> BuildDirectLaunchCores(const std::vector<DirectWorkItem>& work_items,
                                                     const std::string& func_name) {
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
  return launch_cores;
}

static DirectRuntimeBufferState MaterializeRuntimeBuffers(
    distributed::MeshCommandQueue& cq,
    distributed::MeshDevice* mesh_device,
    const ExecutableSpec& spec,
    const std::vector<RuntimeTensorBinding>& buffer_args,
    const std::vector<std::string>& output_names) {
  DirectRuntimeBufferState state;
  for (const auto& binding : buffer_args) {
    ICHECK(binding.tensor != nullptr) << "Null tensor passed to Blackhole direct path";
    const size_t tensor_size = GetDataSize(*binding.tensor);
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = materialization.transport_page_size_bytes,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = tensor_size};
    auto mesh_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device);
    state.runtime_buffers.emplace(binding.name, RuntimeBufferBinding{
                                                .mesh_buffer = mesh_buffer,
                                                .size_bytes = tensor_size,
                                                .is_output = binding.is_output,
                                            });
    if (binding.is_output) {
      state.ordered_output_names.push_back(binding.name);
    } else {
      state.input_names.push_back(binding.name);
    }
    std::vector<uint8_t> initial_data = BuildInputTransferData(spec, binding);
    EnqueueWriteMeshBuffer(cq, mesh_buffer, initial_data, /*blocking=*/true);
  }
  if (!output_names.empty()) {
    state.ordered_output_names = output_names;
  }
  return state;
}

static std::vector<std::string> WriteKernelSourceFiles(const ExecutableSpec& spec,
                                                       const std::string& func_name,
                                                       const std::string& tmp_dir) {
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
  return kernel_paths;
}

static std::vector<KernelHandle> CreateProgramKernelsFromSpec(
    Program& program,
    const CoreRangeSet& launch_core_ranges,
    const ExecutableSpec& spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& runtime_buffers,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& ordered_output_names,
    const std::vector<std::string>& kernel_paths) {
  std::vector<KernelHandle> kernels;
  kernels.reserve(spec.kernels.size());
  for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    const auto& kernel_spec = spec.kernels[ki];
    LOG(INFO) << "Direct path: create kernel[" << ki << "] kind=" << kernel_spec.kind
              << " core_type=" << kernel_spec.core_type;
    kernels.push_back(CreateKernelFromSpec(program, launch_core_ranges, kernel_spec,
                                           runtime_buffers, kernel_paths[ki]));
    const auto common_runtime_args = BuildCommonRuntimeArgsFromSpec(
        kernel_spec, runtime_buffers, semaphore_ids, input_names, ordered_output_names);
    if (!common_runtime_args.empty()) {
      LOG(INFO) << "Direct path: set common runtime args kernel[" << ki
                << "] count=" << common_runtime_args.size();
      SetCommonRuntimeArgs(program, kernels.back(), common_runtime_args);
    }
  }
  return kernels;
}

static void ApplyWorkItemRuntimeArgs(
    Program& program,
    const ExecutableSpec& spec,
    const std::vector<KernelHandle>& kernels,
    const std::vector<DirectWorkItem>& work_items,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& runtime_buffers,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& ordered_output_names,
    const std::vector<uint32_t>& scalar_args) {
  for (const auto& item : work_items) {
    LOG(INFO) << "Direct path: configure work_id=" << item.work_id
              << " core=(" << item.core.x << "," << item.core.y << ")";
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
      const auto& kernel_spec = spec.kernels[ki];
      auto runtime_args = BuildRuntimeArgsFromSpec(
          kernel_spec, spec, item.work_id, device, runtime_buffers, semaphore_ids, input_names,
          ordered_output_names, scalar_args);

      LOG(INFO) << "Direct path: set runtime args kernel[" << ki
                << "] core=(" << item.core.x << "," << item.core.y
                << ") count=" << runtime_args.size();
      SetRuntimeArgs(program, kernels[ki], item.core, runtime_args);
    }
  }
}

static std::vector<uint32_t> BuildKernelCompileTimeArgs(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings);

static DataMovementProcessor ParseDataMovementProcessor(const std::string& processor) {
  if (processor == "riscv_0") {
    return DataMovementProcessor::RISCV_0;
  }
  if (processor == "riscv_1") {
    return DataMovementProcessor::RISCV_1;
  }
  LOG(FATAL) << "Unsupported Blackhole launch_spec processor: " << processor;
}

static MathFidelity ParseMathFidelity(const std::string& math_fidelity) {
  if (math_fidelity == "LoFi") {
    return MathFidelity::LoFi;
  }
  if (math_fidelity == "HiFi2") {
    return MathFidelity::HiFi2;
  }
  if (math_fidelity == "HiFi3") {
    return MathFidelity::HiFi3;
  }
  if (math_fidelity.empty() || math_fidelity == "HiFi4") {
    return MathFidelity::HiFi4;
  }
  LOG(FATAL) << "Unsupported Blackhole compute math_fidelity: " << math_fidelity;
}

static UnpackToDestMode ParseUnpackToDestMode(const std::string& mode) {
  if (mode == "Default") {
    return UnpackToDestMode::Default;
  }
  if (mode == "UnpackToDestFp32") {
    return UnpackToDestMode::UnpackToDestFp32;
  }
  LOG(FATAL) << "Unsupported Blackhole unpack_to_dest_mode: " << mode;
}

static NOC ParseNoc(const std::string& noc) {
  if (noc == "riscv_0_default") {
    return NOC::RISCV_0_default;
  }
  if (noc == "riscv_1_default") {
    return NOC::RISCV_1_default;
  }
  LOG(FATAL) << "Unsupported Blackhole launch_spec noc: " << noc;
}

static void AppendCompileTimeArgValues(const CompileTimeArgSpec& spec,
                                       std::vector<uint32_t>* compile_time_args) {
  const size_t before = compile_time_args->size();
  for (uint32_t value : spec.values) {
    compile_time_args->push_back(value);
  }
  const uint32_t emitted_count =
      static_cast<uint32_t>(compile_time_args->size() - before);
  ICHECK_GT(emitted_count, 0U)
      << "Blackhole compile-time ABI kind " << spec.kind
      << " did not materialize any values";
  if (spec.count != 0U) {
    ICHECK_EQ(spec.count, emitted_count)
        << "Blackhole compile-time ABI kind " << spec.kind
        << " has count mismatch: expected " << spec.count
        << ", materialized " << emitted_count;
  }
}

static void AppendAccessorCompileTimeArgs(const CompileTimeArgSpec& spec,
                                          const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
                                          std::vector<uint32_t>* compile_time_args) {
  const std::string buffer_name = !spec.buffer.empty() ? spec.buffer : spec.name;
  ICHECK(!buffer_name.empty())
      << "Blackhole interleaved accessor compile-time ABI entry is missing a buffer name";
  auto it = buffer_bindings.find(buffer_name);
  ICHECK(it != buffer_bindings.end())
      << "Missing runtime buffer binding for accessor buffer " << buffer_name;
  const auto args_config_bits = static_cast<tensor_accessor::ArgsConfig::Underlying>(
      spec.args_config_bits);
  const tensor_accessor::ArgsConfig args_config(args_config_bits);
  ICHECK((args_config & tensor_accessor::ArgConfig::Runtime).raw() == 0)
      << "Blackhole direct runtime does not yet support accessor common runtime args";

  const size_t before = compile_time_args->size();
  TensorAccessorArgs(*(it->second.mesh_buffer), args_config).append_to(*compile_time_args);
  const uint32_t emitted_count =
      static_cast<uint32_t>(compile_time_args->size() - before);
  ICHECK_EQ(emitted_count, 2U)
      << "Blackhole interleaved accessor compile-time ABI for buffer " << buffer_name
      << " must materialize exactly two uint32 values";
  if (spec.count != 0U) {
    ICHECK_EQ(spec.count, emitted_count)
        << "Blackhole interleaved accessor compile-time ABI count mismatch for buffer "
        << buffer_name;
  }
}

static bool IsSupportedCommonRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr" || kind == "input_buffer_addr32" ||
         kind == "output_buffer_addr" || kind == "output_buffer_addr32" ||
         kind == "semaphore_id_u32";
}

static void ValidateKernelDirectRuntimeSchema(const KernelSpec& kernel) {
  for (const auto& arg_spec : kernel.common_runtime_args) {
    ICHECK(IsSupportedCommonRuntimeArgKind(arg_spec.kind))
        << "Blackhole direct runtime only supports shared common runtime args for "
           "buffer addresses and semaphores; unsupported common runtime arg kind: "
        << arg_spec.kind;
  }

  for (const auto& accessor : kernel.accessors) {
    ICHECK_EQ(accessor.layout, "interleaved")
        << "Blackhole direct runtime currently supports only interleaved accessors";
    ICHECK_EQ(accessor.memory_space, "dram")
        << "Blackhole direct runtime currently supports only DRAM accessors";
    ICHECK_EQ(accessor.common_runtime_arg_count, 0U)
        << "Blackhole direct runtime currently supports only interleaved accessors without common runtime args";
    ICHECK_EQ(accessor.args_config_bits, 2U)
        << "Blackhole direct runtime expects interleaved DRAM accessor args_config_bits == 2";
  }

  for (const auto& spec : kernel.compile_time_arg_specs) {
    if (spec.kind != "interleaved_accessor_cta") {
      continue;
    }
    ICHECK_EQ(spec.layout, "interleaved")
        << "Blackhole direct runtime currently supports only interleaved accessors";
    ICHECK_EQ(spec.memory_space, "dram")
        << "Blackhole direct runtime currently supports only DRAM accessors";
    ICHECK_EQ(spec.args_config_bits, 2U)
        << "Blackhole direct runtime expects interleaved DRAM accessor args_config_bits == 2";
  }
}

static void ValidateKernelDirectRuntimeConstraints(const KernelSpec& kernel) {
  if (kernel.has_launch_spec && !kernel.launch_spec.core_type.empty()) {
    ICHECK_EQ(kernel.launch_spec.core_type, kernel.core_type)
        << "Blackhole launch_spec.core_type mismatch for kernel " << kernel.name
        << ": launch_spec.core_type=" << kernel.launch_spec.core_type
        << ", kernel.core_type=" << kernel.core_type;
  }
  ValidateKernelDirectRuntimeSchema(kernel);
}

static std::vector<uint32_t> BuildKernelCompileTimeArgsFromSchema(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  std::vector<uint32_t> compile_time_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs = kernel.compile_time_arg_specs;
  std::sort(compile_time_arg_specs.begin(), compile_time_arg_specs.end(),
            [](const CompileTimeArgSpec& a, const CompileTimeArgSpec& b) {
              if (a.offset != b.offset) {
                return a.offset < b.offset;
              }
              return a.name < b.name;
            });

  uint32_t expected_offset = 0;
  for (const auto& spec : compile_time_arg_specs) {
    ICHECK_EQ(spec.offset, expected_offset)
        << "Blackhole compile-time ABI offset mismatch for " << spec.name
        << ": got " << spec.offset << ", expected " << expected_offset;

    if (spec.kind == "interleaved_accessor_cta") {
      AppendAccessorCompileTimeArgs(spec, buffer_bindings, &compile_time_args);
    } else if (spec.kind == "gemm_shape" || spec.kind == "gemm_transpose_flags" ||
               spec.kind == "gemm_block_shape" || spec.kind == "gemm_subblock_shape" ||
               spec.kind == "gemm_clear_accum" || spec.kind == "gemm_k_pack" ||
               spec.kind == "gemm_wg_wait" || spec.kind == "gemm_policy" ||
               spec.kind == "literal_u32") {
      AppendCompileTimeArgValues(spec, &compile_time_args);
    } else {
      LOG(FATAL) << "Unsupported Blackhole compile-time ABI kind: " << spec.kind;
    }

    expected_offset = static_cast<uint32_t>(compile_time_args.size());
  }

  return compile_time_args;
}

static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::string& kernel_path) {
  ValidateKernelDirectRuntimeConstraints(kernel);
  const std::vector<uint32_t> compile_time_args =
      BuildKernelCompileTimeArgs(kernel, buffer_bindings);
  const std::string core_type = kernel.core_type;
  if (core_type == "trisc" || kernel.kind == "compute") {
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool dst_full_sync_en = false;
    bool math_approx_mode = false;
    bool bfp8_pack_precise = false;
    if (kernel.has_compute_config) {
      math_fidelity = ParseMathFidelity(kernel.compute_config.math_fidelity);
      fp32_dest_acc_en = kernel.compute_config.fp32_dest_acc_en;
      dst_full_sync_en = kernel.compute_config.dst_full_sync_en;
      math_approx_mode = kernel.compute_config.math_approx_mode;
      bfp8_pack_precise = kernel.compute_config.bfp8_pack_precise;
      for (const auto& define : kernel.compute_config.defines) {
        defines.emplace(define.name, define.value);
      }
      for (const auto& arg : kernel.compute_config.named_compile_args) {
        named_compile_args.emplace(arg.name, arg.value);
      }
      for (const auto& mode : kernel.compute_config.unpack_to_dest_mode) {
        unpack_to_dest_mode.push_back(ParseUnpackToDestMode(mode));
      }
    }
    return CreateKernel(
        program,
        kernel_path,
        core_spec,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compile_time_args,
            .defines = defines,
            .named_compile_args = named_compile_args});
  }

  DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
  NOC noc = NOC::RISCV_0_default;
  if (kernel.has_launch_spec) {
    if (!kernel.launch_spec.processor.empty()) {
      processor = ParseDataMovementProcessor(kernel.launch_spec.processor);
    } else if (core_type == "ncrisc") {
      processor = DataMovementProcessor::RISCV_1;
    }
    if (!kernel.launch_spec.noc.empty()) {
      noc = ParseNoc(kernel.launch_spec.noc);
    } else if (core_type == "ncrisc") {
      noc = NOC::RISCV_1_default;
    }
  } else if (core_type == "ncrisc") {
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
          .compile_args = compile_time_args});
}

static std::vector<uint32_t> BuildKernelCompileTimeArgs(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  ValidateKernelDirectRuntimeSchema(kernel);

  if (!kernel.compile_time_arg_specs.empty()) {
    return BuildKernelCompileTimeArgsFromSchema(kernel, buffer_bindings);
  }

  std::vector<uint32_t> compile_time_args = kernel.compile_time_args;
  if (kernel.accessors.empty()) {
    return compile_time_args;
  }

  std::vector<AccessorSpec> accessors = kernel.accessors;
  std::sort(accessors.begin(), accessors.end(),
            [](const AccessorSpec& a, const AccessorSpec& b) {
              return a.compile_time_arg_offset < b.compile_time_arg_offset;
            });

  uint32_t expected_slot = static_cast<uint32_t>(compile_time_args.size());
  for (const auto& accessor : accessors) {
    ICHECK_EQ(accessor.layout, "interleaved")
        << "Blackhole direct runtime currently supports only interleaved accessors";
    ICHECK_EQ(accessor.memory_space, "dram")
        << "Blackhole direct runtime currently supports only DRAM accessors";
    ICHECK_EQ(accessor.common_runtime_arg_count, 0U)
        << "Blackhole direct runtime currently supports only interleaved accessors without common runtime args";
    ICHECK_EQ(accessor.args_config_bits, 2U)
        << "Blackhole direct runtime expects interleaved DRAM accessor args_config_bits == 2";
    ICHECK_EQ(accessor.compile_time_arg_count, 2U)
        << "Blackhole direct runtime currently supports only interleaved accessors with two compile-time args";
    ICHECK_EQ(accessor.compile_time_arg_offset, expected_slot)
        << "Accessor compile-time offset mismatch for buffer " << accessor.buffer
        << ": got " << accessor.compile_time_arg_offset << ", expected " << expected_slot;
    auto it = buffer_bindings.find(accessor.buffer);
    ICHECK(it != buffer_bindings.end())
        << "Missing runtime buffer binding for accessor buffer " << accessor.buffer;
    TensorAccessorArgs(*(it->second.mesh_buffer), tensor_accessor::ArgsConfig(2))
        .append_to(compile_time_args);
    expected_slot += accessor.compile_time_arg_count;
  }
  return compile_time_args;
}

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

static uint32_t ResolveRuntimeSemaphoreId(
    const KernelSpec& kernel,
    const KernelArgSpec& arg_spec,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  auto binding_it = std::find_if(
      kernel.semaphore_bindings.begin(), kernel.semaphore_bindings.end(),
      [&](const SemaphoreBindingSpec& binding) {
        return binding.name == arg_spec.name && binding.arg_kind == arg_spec.kind;
      });
  ICHECK(binding_it != kernel.semaphore_bindings.end())
      << "Blackhole runtime arg " << arg_spec.name << " kind=" << arg_spec.kind
      << " is missing a matching semaphore binding";
  auto semaphore_it = semaphore_ids.find(binding_it->semaphore_id);
  ICHECK(semaphore_it != semaphore_ids.end())
      << "Blackhole kernel semaphore binding " << binding_it->name
      << " references missing planned semaphore id " << binding_it->semaphore_id;
  return semaphore_it->second;
}

static std::vector<uint32_t> BuildCommonRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
  std::vector<uint32_t> args;
  for (const auto& arg_spec : kernel.common_runtime_args) {
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
    } else if (arg_spec.kind == "semaphore_id_u32") {
      args.push_back(ResolveRuntimeSemaphoreId(kernel, arg_spec, semaphore_ids));
    } else {
      LOG(FATAL) << "Unsupported common runtime arg kind: " << arg_spec.kind;
    }
  }
  return args;
}

static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<uint32_t>& scalar_args) {
  std::vector<uint32_t> args;
  size_t scalar_index = 0;
  const uint32_t num_k_tiles = GetRuntimeNumKTiles(spec);
  const uint32_t logical_grid_x = GetRuntimeLogicalGridX(spec);
  const uint32_t work_linear_id = current_work_linear_id;
  const uint32_t bx = logical_grid_x == 0 ? 0 : (work_linear_id % logical_grid_x);
  const uint32_t by = logical_grid_x == 0 ? 0 : (work_linear_id / logical_grid_x);
  const auto kernel_requests_kind = [&](const char* kind) {
    return std::any_of(kernel.runtime_args.begin(), kernel.runtime_args.end(),
                       [&](const KernelArgSpec& runtime_arg) {
                         return runtime_arg.kind == kind;
                       });
  };

  const auto compute_contract = GetComputeContract(spec);
  const bool has_gemm_compute_contract =
      compute_contract.enabled && compute_contract.kind == "gemm";

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
    } else if (arg_spec.kind == "work_linear_id" || arg_spec.kind == "current_work_linear_id") {
      args.push_back(work_linear_id);
    } else if (arg_spec.kind == "a_tile_start_id") {
      args.push_back(has_gemm_compute_contract && kernel.kind == "reader" ? by : work_linear_id);
    } else if (arg_spec.kind == "a_tile_num_tiles") {
      ICHECK(kernel_requests_kind("a_tile_start_id"))
          << "a_tile_num_tiles requires a_tile_start_id in Blackhole direct runtime";
      args.push_back(has_gemm_compute_contract && kernel.kind == "reader" ? num_k_tiles : 1);
    } else if (arg_spec.kind == "a_tile_stride") {
      ICHECK(kernel_requests_kind("a_tile_start_id"))
          << "a_tile_stride requires a_tile_start_id in Blackhole direct runtime";
      args.push_back(1);
    } else if (arg_spec.kind == "b_tile_start_id") {
      ICHECK(has_gemm_compute_contract && kernel.kind == "reader")
          << "b_tile_start_id is only supported for GEMM reader kernels in Blackhole direct runtime";
      args.push_back(bx);
    } else if (arg_spec.kind == "b_tile_num_tiles") {
      ICHECK(kernel_requests_kind("b_tile_start_id"))
          << "b_tile_num_tiles requires b_tile_start_id in Blackhole direct runtime";
      ICHECK(has_gemm_compute_contract && kernel.kind == "reader")
          << "b_tile_num_tiles is only supported for GEMM reader kernels in Blackhole direct runtime";
      args.push_back(num_k_tiles);
    } else if (arg_spec.kind == "b_tile_stride") {
      ICHECK(kernel_requests_kind("b_tile_start_id"))
          << "b_tile_stride requires b_tile_start_id in Blackhole direct runtime";
      ICHECK(has_gemm_compute_contract && kernel.kind == "reader")
          << "b_tile_stride is only supported for GEMM reader kernels in Blackhole direct runtime";
      args.push_back(GetRuntimeLogicalNTiles(spec));
    } else if (arg_spec.kind == "output_tile_start_id") {
      args.push_back(work_linear_id);
    } else if (arg_spec.kind == "output_tile_num_tiles") {
      ICHECK(kernel_requests_kind("output_tile_start_id"))
          << "output_tile_num_tiles requires output_tile_start_id in Blackhole direct runtime";
      args.push_back(1);
    } else if (arg_spec.kind == "output_tile_stride") {
      ICHECK(kernel_requests_kind("output_tile_start_id"))
          << "output_tile_stride requires output_tile_start_id in Blackhole direct runtime";
      args.push_back(1);
    } else if (arg_spec.kind == "k_tile_start_id") {
      ICHECK(has_gemm_compute_contract)
          << "k_tile_start_id is only supported for GEMM kernels in Blackhole direct runtime";
      ICHECK(kernel_requests_kind("num_k_tiles"))
          << "k_tile_start_id requires num_k_tiles in Blackhole direct runtime";
      args.push_back(0);
    } else if (arg_spec.kind == "num_k_tiles") {
      ICHECK_GT(num_k_tiles, 0)
          << "num_k_tiles requested by runtime schema, but direct runtime could not derive a "
             "supported value from ExecutableSpec";
      args.push_back(num_k_tiles);
    } else if (arg_spec.kind == "scalar_u32") {
      ICHECK(scalar_index < scalar_args.size())
          << "Spec requested more scalar args than provided";
      args.push_back(scalar_args[scalar_index++]);
    } else if (arg_spec.kind == "logical_core_noc_x") {
      ICHECK(arg_spec.has_core_coord)
          << "logical_core_noc_x requires core_x/core_y in the runtime arg schema";
      const CoreCoord noc_core =
          device.worker_core_from_logical_core(CoreCoord{arg_spec.core_x, arg_spec.core_y});
      args.push_back(static_cast<uint32_t>(noc_core.x));
    } else if (arg_spec.kind == "logical_core_noc_y") {
      ICHECK(arg_spec.has_core_coord)
          << "logical_core_noc_y requires core_x/core_y in the runtime arg schema";
      const CoreCoord noc_core =
          device.worker_core_from_logical_core(CoreCoord{arg_spec.core_x, arg_spec.core_y});
      args.push_back(static_cast<uint32_t>(noc_core.y));
    } else if (arg_spec.kind == "semaphore_id_u32") {
      args.push_back(ResolveRuntimeSemaphoreId(kernel, arg_spec, semaphore_ids));
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

static void ValidateExecutableSpecCorePlan(const std::string& func_name,
                                           const ExecutableSpec& spec) {
  const auto& core_plan = spec.core_plan;
  ICHECK(!core_plan.work_packets.empty())
      << "Blackhole planner/runtime contract requires non-empty core_plan.work_packets for "
      << func_name;
  uint64_t total_work_items = 0;
  for (const auto& packet : core_plan.work_packets) {
    ICHECK_GT(packet.work_count, 0U)
        << "Blackhole planner/runtime contract requires positive work_count in core_plan."
           "work_packets for "
        << func_name;
    total_work_items += packet.work_count;
  }
  ICHECK_GT(total_work_items, 0U)
      << "Blackhole planner/runtime contract requires at least one logical work item for "
      << func_name;
}

BlackholeModuleNode::BlackholeModuleNode(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir)
    : fmap_(std::move(fmap)),
      kernel_dir_(std::move(kernel_dir)) {
  for (const auto& entry : fmap_) {
    ValidateExecutableSpecCorePlan(entry.first, entry.second);
  }
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

ffi::Optional<ffi::String> BlackholeModuleNode::GetFunctionMetadata(const ffi::String& name) {
  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return std::nullopt;
  }
  return ffi::String(EncodeExecutableSpecMetadata(it->second));
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
  ValidateComputeContractDirectRuntimeConstraints(spec);
  for (const auto& kernel_spec : spec.kernels) {
    ValidateKernelDirectRuntimeConstraints(kernel_spec);
  }

  const std::vector<DirectWorkItem> work_items = BuildDirectWorkItems(spec, func_name);
  const std::vector<CoreCoord> launch_cores = BuildDirectLaunchCores(work_items, func_name);
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

  DirectRuntimeBufferState runtime_buffer_state =
      MaterializeRuntimeBuffers(cq, mesh_device.get(), spec, buffer_args, output_names);

  // Write kernel source files to temp directory
  std::string tmp_dir = MakeUniqueTempDir("tilelang_bh_direct_");
  std::vector<std::string> kernel_paths = WriteKernelSourceFiles(spec, func_name, tmp_dir);

  LOG(INFO) << "Direct path: executing " << work_items.size()
            << " logical work items across " << launch_cores.size()
            << " launch cores for " << func_name;

  Program program = CreateProgram();
  const std::unordered_map<uint32_t, uint32_t> semaphore_ids =
      CreateSemaphoresFromSpec(program, spec);

  // Materialize shared CBs and kernels once for the full launch core set.
  CreateCircularBuffersFromSpec(program, launch_core_ranges, spec);

  std::vector<KernelHandle> kernels = CreateProgramKernelsFromSpec(
      program, launch_core_ranges, spec, runtime_buffer_state.runtime_buffers, semaphore_ids,
      runtime_buffer_state.input_names, runtime_buffer_state.ordered_output_names, kernel_paths);

  // Keep runtime args per core/work-item so each logical work item sees its own ID.
  ApplyWorkItemRuntimeArgs(program, spec, kernels, work_items, *mesh_device,
                           runtime_buffer_state.runtime_buffers, semaphore_ids,
                           runtime_buffer_state.input_names,
                           runtime_buffer_state.ordered_output_names, scalar_args);

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
    auto it = runtime_buffer_state.runtime_buffers.find(binding.name);
    ICHECK(it != runtime_buffer_state.runtime_buffers.end())
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
