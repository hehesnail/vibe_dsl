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
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

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

static bool BlackholeDebugIOEnabled() {
  const char* value = std::getenv("TILELANG_BLACKHOLE_DEBUG_IO");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static float BFloat16BitsToFloat(uint16_t bits) {
  union {
    uint32_t i;
    float f;
  } value{static_cast<uint32_t>(bits) << 16};
  return value.f;
}

template <typename T>
static std::string PreviewTensorValues(const std::vector<T>& values, size_t limit = 8) {
  std::ostringstream os;
  os << "[";
  const size_t count = std::min(limit, values.size());
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      os << ", ";
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
      os << BFloat16BitsToFloat(values[i]);
    } else {
      os << values[i];
    }
  }
  if (values.size() > count) {
    os << ", ...";
  }
  os << "]";
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

static std::vector<const KernelComputeOpSpec*> CollectGemmComputeOps(const ExecutableSpec& spec) {
  std::vector<const KernelComputeOpSpec*> gemm_ops;
  for (const auto& kernel : spec.kernels) {
    for (const auto& compute_op : kernel.compute_ops) {
      if (!compute_op.enabled || compute_op.kind != "gemm") {
        continue;
      }
      gemm_ops.push_back(&compute_op);
    }
  }
  return gemm_ops;
}

static KernelComputeOpSpec GetPrimaryGemmCompute(const ExecutableSpec& spec) {
  const auto gemm_ops = CollectGemmComputeOps(spec);
  if (gemm_ops.empty()) {
    return KernelComputeOpSpec();
  }
  return *gemm_ops.front();
}

static void ValidateGemmComputeOpsShareDirectRuntimeWorkDecomposition(
    const ExecutableSpec& spec) {
  const auto gemm_ops = CollectGemmComputeOps(spec);
  if (gemm_ops.size() <= 1) {
    return;
  }

  const KernelComputeOpSpec& reference = *gemm_ops.front();
  for (size_t i = 1; i < gemm_ops.size(); ++i) {
    const KernelComputeOpSpec& gemm = *gemm_ops[i];
    ICHECK_EQ(gemm.M, reference.M)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on M";
    ICHECK_EQ(gemm.N, reference.N)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on N";
    ICHECK_EQ(gemm.K, reference.K)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on K";
    ICHECK_EQ(gemm.Mt, reference.Mt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Mt";
    ICHECK_EQ(gemm.Nt, reference.Nt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Nt";
    ICHECK_EQ(gemm.Kt, reference.Kt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Kt";
  }
}

static void ValidateGemmComputeDirectRuntimeConstraints(const ExecutableSpec& spec) {
  ValidateGemmComputeOpsShareDirectRuntimeWorkDecomposition(spec);
  for (const KernelComputeOpSpec* gemm_op : CollectGemmComputeOps(spec)) {
    ICHECK(!gemm_op->has_mbarrier)
        << "Blackhole direct runtime does not yet support GEMM compute_ops.mbarrier bindings";
  }
}

static void ValidateGemmInputShape(const ExecutableSpec& spec,
                                   const RuntimeTensorBinding& binding,
                                   uint32_t rows,
                                   uint32_t cols) {
  const auto gemm = GetPrimaryGemmCompute(spec);
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
  const auto gemm = GetPrimaryGemmCompute(spec);
  const uint32_t logical_grid_x = std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
  const uint32_t logical_grid_y = std::max<uint32_t>(1, spec.core_plan.logical_grid_y);
  const uint32_t expected_rows = gemm.M * logical_grid_y;
  const uint32_t expected_cols = gemm.N * logical_grid_x;
  ICHECK(rows == expected_rows && cols == expected_cols)
      << "Unexpected C tensor shape for GEMM direct path: got (" << rows << ", " << cols
      << "), expected (" << expected_rows << ", " << expected_cols
      << ") for logical grid " << logical_grid_x << "x" << logical_grid_y;
}

static const BufferMaterializationSpec& ResolveBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name);
static const BufferMaterializationSpec* FindBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name);

template <typename T>
static const T* GetTensorData(const DLTensor* tensor) {
  const uint8_t* base = static_cast<const uint8_t*>(tensor->data);
  return reinterpret_cast<const T*>(base + tensor->byte_offset);
}

template <typename T>
static T* GetTensorData(DLTensor* tensor) {
  uint8_t* base = static_cast<uint8_t*>(tensor->data);
  return reinterpret_cast<T*>(base + tensor->byte_offset);
}

static bool HasCompactRowMajorLayout(const DLTensor* tensor) {
  if (tensor == nullptr || tensor->ndim <= 0) {
    return false;
  }
  if (tensor->strides == nullptr) {
    return true;
  }
  int64_t expected_stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    if (tensor->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= tensor->shape[i];
  }
  return true;
}

static std::vector<int64_t> GetTensorShape(const DLTensor* tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor->ndim);
  for (int i = 0; i < tensor->ndim; ++i) {
    shape.push_back(tensor->shape[i]);
  }
  return shape;
}

static int64_t ShapeProduct(const std::vector<int64_t>& shape, size_t begin, size_t end) {
  int64_t product = 1;
  for (size_t i = begin; i < end; ++i) {
    product *= shape[i];
  }
  return product;
}

static std::vector<int64_t> InvertAxisOrder(const std::vector<int64_t>& axis_order) {
  std::vector<int64_t> inverse(axis_order.size(), -1);
  for (size_t i = 0; i < axis_order.size(); ++i) {
    inverse[static_cast<size_t>(axis_order[i])] = static_cast<int64_t>(i);
  }
  return inverse;
}

static std::vector<int64_t> PermuteShape(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& axis_order) {
  std::vector<int64_t> permuted_shape;
  permuted_shape.reserve(axis_order.size());
  for (int64_t axis : axis_order) {
    permuted_shape.push_back(shape[static_cast<size_t>(axis)]);
  }
  return permuted_shape;
}

static std::vector<int64_t> ComputeRowMajorStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = running;
    running *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

static bool IsValidAxisOrder(const std::vector<int64_t>& axis_order, int ndim) {
  if (static_cast<int>(axis_order.size()) != ndim) {
    return false;
  }
  std::vector<bool> seen(static_cast<size_t>(ndim), false);
  for (int64_t axis : axis_order) {
    if (axis < 0 || axis >= ndim || seen[static_cast<size_t>(axis)]) {
      return false;
    }
    seen[static_cast<size_t>(axis)] = true;
  }
  return true;
}

template <typename T>
static std::vector<T> PermuteContiguousTensorAxes(const T* src,
                                                  const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& axis_order) {
  const std::vector<int64_t> permuted_shape = PermuteShape(shape, axis_order);
  const std::vector<int64_t> input_strides = ComputeRowMajorStrides(shape);
  const std::vector<int64_t> output_strides = ComputeRowMajorStrides(permuted_shape);
  const size_t numel = static_cast<size_t>(ShapeProduct(shape, 0, shape.size()));
  std::vector<T> output(numel);
  for (size_t out_linear = 0; out_linear < numel; ++out_linear) {
    size_t remainder = out_linear;
    int64_t input_linear = 0;
    for (size_t i = 0; i < axis_order.size(); ++i) {
      const int64_t stride = output_strides[i];
      const int64_t coord =
          stride == 0 ? 0 : static_cast<int64_t>(remainder / static_cast<size_t>(stride));
      remainder %= static_cast<size_t>(stride);
      input_linear += coord * input_strides[static_cast<size_t>(axis_order[i])];
    }
    output[out_linear] = src[static_cast<size_t>(input_linear)];
  }
  return output;
}

struct InterleavedTilePlan {
  bool enabled = false;
  uint32_t tile_rows = 0;
  uint32_t tile_cols = 0;
  std::vector<int64_t> axis_order;
  bool transpose_2d = false;
};

static InterleavedTilePlan BuildInterleavedTilePlan(const ExecutableSpec& spec,
                                                    const BufferMaterializationSpec& materialization,
                                                    const DLTensor* tensor) {
  (void)spec;
  InterleavedTilePlan plan;
  if (tensor == nullptr || tensor->ndim < 2 || !HasCompactRowMajorLayout(tensor)) {
    return plan;
  }
  if (tensor->dtype.lanes != 1 || tensor->dtype.bits == 0) {
    return plan;
  }

  const uint32_t element_size_bytes = static_cast<uint32_t>((tensor->dtype.bits + 7) / 8);
  if (element_size_bytes == 0 || materialization.transport_page_size_bytes == 0 ||
      materialization.transport_page_size_bytes % element_size_bytes != 0) {
    return plan;
  }

  constexpr uint32_t kBlackholeTileCols = 32;
  const uint32_t tile_elements = materialization.transport_page_size_bytes / element_size_bytes;
  const uint32_t tile_rows = tile_elements / kBlackholeTileCols;
  if (tile_elements == 0 || tile_elements % kBlackholeTileCols != 0) {
    return plan;
  }

  if (!materialization.host_axis_order.empty()) {
    ICHECK(IsValidAxisOrder(materialization.host_axis_order, tensor->ndim))
        << "Invalid Blackhole host_axis_order materialization contract for buffer "
        << materialization.buffer;
    plan.axis_order = materialization.host_axis_order;
  } else {
    ICHECK(!materialization.host_axis_order.empty())
        << "Blackhole runtime requires explicit host_axis_order materialization contract for buffer "
        << materialization.buffer;
  }
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  int64_t total_rows = ShapeProduct(device_shape, 0, device_shape.size() - 1);
  int64_t cols = device_shape.back();
  if (materialization.transpose_2d) {
    std::swap(total_rows, cols);
  }
  if (tile_rows == 0 || cols <= 0 || cols % kBlackholeTileCols != 0 || total_rows <= 0 ||
      total_rows % tile_rows != 0) {
    return plan;
  }

  plan.enabled = true;
  plan.tile_rows = tile_rows;
  plan.tile_cols = kBlackholeTileCols;
  plan.transpose_2d = materialization.transpose_2d;
  return plan;
}

template <typename T>
static std::vector<uint8_t> BuildInterleavedTiledTransferData(const DLTensor* tensor,
                                                              const InterleavedTilePlan& plan) {
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<T> permuted = PermuteContiguousTensorAxes(
      GetTensorData<T>(tensor), host_shape, plan.axis_order);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  uint32_t rows = static_cast<uint32_t>(ShapeProduct(device_shape, 0, device_shape.size() - 1));
  uint32_t cols = static_cast<uint32_t>(device_shape.back());
  std::vector<T> row_major = permuted;
  if (plan.transpose_2d) {
    row_major = TransposeRowMajor2D(permuted.data(), rows, cols);
    std::swap(rows, cols);
  }
  std::vector<T> tiled = tilize_nfaces(row_major, rows, cols);
  std::vector<uint8_t> bytes(tiled.size() * sizeof(T));
  std::memcpy(bytes.data(), tiled.data(), bytes.size());
  return bytes;
}

template <typename T>
static void CopyInterleavedTiledOutputToTensor(DLTensor* tensor,
                                               const InterleavedTilePlan& plan,
                                               const std::vector<uint8_t>& output_data) {
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  uint32_t rows = static_cast<uint32_t>(ShapeProduct(device_shape, 0, device_shape.size() - 1));
  uint32_t cols = static_cast<uint32_t>(device_shape.back());
  if (plan.transpose_2d) {
    std::swap(rows, cols);
  }
  const size_t numel = static_cast<size_t>(ShapeProduct(device_shape, 0, device_shape.size()));
  ICHECK_EQ(output_data.size(), numel * sizeof(T))
      << "Unexpected interleaved tiled output buffer size: got " << output_data.size()
      << " bytes, expected " << (numel * sizeof(T)) << " bytes";
  const auto* tiled = reinterpret_cast<const T*>(output_data.data());
  std::vector<T> tiled_vec(tiled, tiled + numel);
  if (BlackholeDebugIOEnabled()) {
    LOG(INFO) << "Blackhole debug output: tiled preview=" << PreviewTensorValues(tiled_vec);
  }
  std::vector<T> device_row_major = untilize_nfaces(tiled_vec, rows, cols);
  if (plan.transpose_2d) {
    device_row_major = TransposeRowMajor2D(device_row_major.data(), rows, cols);
  }
  const std::vector<int64_t> inverse_axis_order = InvertAxisOrder(plan.axis_order);
  std::vector<T> host_row_major =
      PermuteContiguousTensorAxes(device_row_major.data(), device_shape, inverse_axis_order);
  if (BlackholeDebugIOEnabled()) {
    LOG(INFO) << "Blackhole debug output: host preview=" << PreviewTensorValues(host_row_major);
  }
  std::memcpy(GetTensorData<T>(tensor), host_row_major.data(), host_row_major.size() * sizeof(T));
}

static std::vector<uint8_t> BuildInputTransferData(const ExecutableSpec& spec,
                                                   const RuntimeTensorBinding& binding) {
  const DLTensor* tensor = binding.tensor;
  ICHECK(tensor != nullptr);
  const size_t tensor_size = GetDataSize(*tensor);
  const auto gemm = GetPrimaryGemmCompute(spec);

  if (!gemm.enabled || gemm.kind != "gemm" || !IsTwoDimTensor(tensor)) {
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    const InterleavedTilePlan tile_plan =
        BuildInterleavedTilePlan(spec, materialization, tensor);
    if (tile_plan.enabled) {
      if (tensor->dtype.bits == 16) {
        auto tiled = BuildInterleavedTiledTransferData<uint16_t>(tensor, tile_plan);
        if (BlackholeDebugIOEnabled()) {
          const auto* values = reinterpret_cast<const uint16_t*>(tiled.data());
          std::vector<uint16_t> preview(values, values + tiled.size() / sizeof(uint16_t));
          LOG(INFO) << "Blackhole debug input " << binding.name
                    << ": tiled preview=" << PreviewTensorValues(preview);
        }
        return tiled;
      }
      if (tensor->dtype.bits == 32) {
        auto tiled = BuildInterleavedTiledTransferData<uint32_t>(tensor, tile_plan);
        if (BlackholeDebugIOEnabled()) {
          const auto* values = reinterpret_cast<const uint32_t*>(tiled.data());
          std::vector<uint32_t> preview(values, values + tiled.size() / sizeof(uint32_t));
          LOG(INFO) << "Blackhole debug input " << binding.name
                    << ": tiled preview=" << PreviewTensorValues(preview);
        }
        return tiled;
      }
    }
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), GetTensorData<uint8_t>(tensor), tensor_size);
    return raw;
  }

  if (binding.name != gemm.a_buffer && binding.name != gemm.b_buffer) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), GetTensorData<uint8_t>(tensor), tensor_size);
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
  const auto* raw = GetTensorData<uint16_t>(tensor);
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
  const auto gemm = GetPrimaryGemmCompute(spec);
  const BufferMaterializationSpec* materialized_output =
      FindBufferMaterializationSpec(spec, binding.name);

  if (!gemm.enabled || gemm.kind != "gemm" || binding.name != gemm.c_buffer ||
      (materialized_output != nullptr && !materialized_output->live_form_kind.empty()) ||
      !IsTwoDimTensor(binding.tensor)) {
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    const InterleavedTilePlan tile_plan =
        BuildInterleavedTilePlan(spec, materialization, binding.tensor);
    if (tile_plan.enabled) {
      if (binding.tensor->dtype.bits == 16) {
        CopyInterleavedTiledOutputToTensor<uint16_t>(binding.tensor, tile_plan, output_data);
        return;
      }
      if (binding.tensor->dtype.bits == 32) {
        CopyInterleavedTiledOutputToTensor<uint32_t>(binding.tensor, tile_plan, output_data);
        return;
      }
    }
    std::memcpy(GetTensorData<uint8_t>(binding.tensor), output_data.data(), tensor_size);
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
  std::memcpy(GetTensorData<float>(binding.tensor), row_major.data(),
              row_major.size() * sizeof(float));
}

static tt::DataFormat ParseDataFormat(const std::string& value) {
  if (value == "Float16") return tt::DataFormat::Float16;
  if (value == "Float16_b") return tt::DataFormat::Float16_b;
  if (value == "Float32") return tt::DataFormat::Float32;
  if (value == "UInt16") return tt::DataFormat::UInt16;
  if (value == "UInt32") return tt::DataFormat::UInt32;
  LOG(FATAL) << "Unsupported data format: " << value;
  return tt::DataFormat::Float16_b;
}

static const BufferMaterializationSpec& ResolveBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name) {
  const BufferMaterializationSpec* materialization =
      FindBufferMaterializationSpec(spec, buffer_name);
  ICHECK(materialization != nullptr)
      << "Missing Blackhole buffer materialization spec for buffer " << buffer_name;
  ICHECK_EQ(materialization->materialization_kind, "replicated")
      << "Unsupported Blackhole buffer materialization kind for " << buffer_name << ": "
      << materialization->materialization_kind;
  ICHECK_EQ(materialization->memory_space, "dram")
      << "Unsupported Blackhole buffer memory_space for " << buffer_name << ": "
      << materialization->memory_space;
  ICHECK_GT(materialization->transport_page_size_bytes, 0U)
      << "Blackhole buffer materialization requires transport_page_size for " << buffer_name;
  return *materialization;
}

static const BufferMaterializationSpec* FindBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name) {
  auto it = std::find_if(spec.buffer_materializations.begin(), spec.buffer_materializations.end(),
                         [&](const BufferMaterializationSpec& materialization) {
                           return materialization.buffer == buffer_name;
                         });
  if (it == spec.buffer_materializations.end()) {
    return nullptr;
  }
  return &*it;
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
  const auto gemm = GetPrimaryGemmCompute(spec);
  if (gemm.enabled && gemm.kind == "gemm") {
    ICHECK_GT(gemm.Kt, 0U)
        << "Blackhole GEMM direct path requires compute_ops GEMM Kt to be populated";
    return std::max<uint32_t>(1, gemm.Kt);
  }
  return 0;
}

static uint32_t GetRuntimeLogicalGridX(const ExecutableSpec& spec) {
  return std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
}

static uint32_t GetRuntimeLogicalNTiles(const ExecutableSpec& spec) {
  const auto gemm = GetPrimaryGemmCompute(spec);
  ICHECK(gemm.enabled && gemm.kind == "gemm")
      << "logical_n_tiles is only defined for GEMM kernels in Blackhole direct runtime";
  ICHECK_GT(gemm.Nt, 0U)
      << "Blackhole GEMM direct path requires compute_ops GEMM Nt to be populated";
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
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids);
static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<uint32_t>& scalar_args);

struct DirectWorkItem {
  uint32_t work_id;
  CoreCoord core;
};

struct DirectLaunchWave {
  std::vector<DirectWorkItem> work_items;
  std::vector<CoreCoord> launch_cores;
};

struct DirectRuntimeBufferState {
  std::unordered_map<std::string, RuntimeBufferBinding> runtime_buffers;
  std::vector<std::string> input_names;
  std::vector<std::string> ordered_output_names;
};

struct SynchronizationRuntimeContext {
  const IDevice* device{nullptr};
  const std::unordered_map<uint32_t, uint32_t>* semaphore_ids{nullptr};
};

static uint64_t EncodeDirectLaunchCoreKey(const CoreCoord& core) {
  return (static_cast<uint64_t>(core.x) << 32) | static_cast<uint64_t>(core.y);
}

static bool HasDirectLaunchSynchronizationContract(const ExecutableSpec& spec) {
  if (!spec.semaphores.empty()) {
    return true;
  }
  return std::any_of(spec.kernels.begin(), spec.kernels.end(), [](const KernelSpec& kernel) {
    return !kernel.semaphore_bindings.empty() || !kernel.remote_core_descriptors.empty();
  });
}

static std::vector<DirectLaunchWave> BuildDirectLaunchWaves(const ExecutableSpec& spec,
                                                            const std::string& func_name) {
  const auto& work_packets = spec.core_plan.work_packets;
  uint64_t total_work_items = 0;
  uint32_t max_work_count = 0;
  for (const auto& packet : work_packets) {
    total_work_items += packet.work_count;
    max_work_count = std::max(max_work_count, packet.work_count);
  }
  ICHECK_GT(total_work_items, 0U)
      << "Blackhole planner/runtime contract requires at least one logical work item for "
      << func_name;
  ICHECK_GT(max_work_count, 0U)
      << "Blackhole planner/runtime contract requires positive work_count in core_plan.work_packets "
         "for "
      << func_name;

  const bool oversubscribed = total_work_items > work_packets.size();
  if (oversubscribed) {
    ICHECK(!HasDirectLaunchSynchronizationContract(spec))
        << "Blackhole direct runtime oversubscribed launch currently wave-schedules logical work "
           "items across repeated program launches. This is only supported when the executable "
           "has no explicit semaphore or remote-core synchronization contract. Function "
        << func_name << " maps " << total_work_items << " logical work items onto "
        << work_packets.size() << " physical-core packets.";
  }

  std::vector<DirectLaunchWave> waves;
  waves.reserve(max_work_count);
  for (uint32_t wave_index = 0; wave_index < max_work_count; ++wave_index) {
    DirectLaunchWave wave;
    wave.work_items.reserve(work_packets.size());
    wave.launch_cores.reserve(work_packets.size());
    std::unordered_set<uint64_t> seen_cores;
    seen_cores.reserve(work_packets.size());
    for (const auto& packet : work_packets) {
      if (wave_index >= packet.work_count) {
        continue;
      }
      const CoreCoord packet_core{packet.core_x, packet.core_y};
      ICHECK(seen_cores.insert(EncodeDirectLaunchCoreKey(packet_core)).second)
          << "Blackhole planner/runtime contract requires each launch wave to map at most one "
             "logical work item to a physical core. Function "
          << func_name << " duplicates core (" << packet_core.x << "," << packet_core.y
          << ") within wave " << wave_index;
      wave.work_items.push_back({packet.work_offset + wave_index, packet_core});
      wave.launch_cores.push_back(packet_core);
    }
    if (!wave.work_items.empty()) {
      waves.push_back(std::move(wave));
    }
  }

  ICHECK(!waves.empty()) << "No launch waves resolved for direct execution";
  return waves;
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
    const std::vector<std::string>& kernel_paths) {
  std::vector<KernelHandle> kernels;
  kernels.reserve(spec.kernels.size());
  for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    const auto& kernel_spec = spec.kernels[ki];
    LOG(INFO) << "Direct path: create kernel[" << ki << "] kind=" << kernel_spec.kind
              << " core_type=" << kernel_spec.core_type;
    kernels.push_back(CreateKernelFromSpec(program, launch_core_ranges, kernel_spec,
                                           runtime_buffers, kernel_paths[ki]));
    const auto common_runtime_args =
        BuildCommonRuntimeArgsFromSpec(kernel_spec, runtime_buffers, semaphore_ids);
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
    const std::vector<uint32_t>& scalar_args) {
  for (const auto& item : work_items) {
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
      const auto& kernel_spec = spec.kernels[ki];
      auto runtime_args = BuildRuntimeArgsFromSpec(
          kernel_spec, spec, item.work_id, device, runtime_buffers, semaphore_ids, scalar_args);
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

static void ValidateDirectRuntimeAccessorSpec(const std::string& buffer_name,
                                              const std::string& layout,
                                              const std::string& memory_space,
                                              uint32_t common_runtime_arg_count,
                                              uint32_t args_config_bits) {
  ICHECK_EQ(layout, "interleaved")
      << "Blackhole direct runtime currently supports only interleaved accessors";
  ICHECK_EQ(memory_space, "dram")
      << "Blackhole direct runtime currently supports only DRAM accessors";
  ICHECK_EQ(common_runtime_arg_count, 0U)
      << "Blackhole direct runtime currently supports only interleaved accessors without common runtime args";
  ICHECK_EQ(args_config_bits, 2U)
      << "Blackhole direct runtime expects interleaved DRAM accessor args_config_bits == 2";
}

static void AppendInterleavedAccessorCompileTimeArgs(
    const std::string& buffer_name,
    uint32_t expected_count,
    uint32_t args_config_bits,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* compile_time_args) {
  ICHECK(!buffer_name.empty())
      << "Blackhole interleaved accessor compile-time ABI entry is missing a buffer name";
  ValidateDirectRuntimeAccessorSpec(buffer_name, "interleaved", "dram",
                                    /*common_runtime_arg_count=*/0, args_config_bits);
  auto it = buffer_bindings.find(buffer_name);
  ICHECK(it != buffer_bindings.end())
      << "Missing runtime buffer binding for accessor buffer " << buffer_name;
  const auto underlying_args_config_bits = static_cast<tensor_accessor::ArgsConfig::Underlying>(
      args_config_bits);
  const tensor_accessor::ArgsConfig args_config(underlying_args_config_bits);
  ICHECK((args_config & tensor_accessor::ArgConfig::Runtime).raw() == 0)
      << "Blackhole direct runtime does not yet support accessor common runtime args";

  const size_t before = compile_time_args->size();
  TensorAccessorArgs(*(it->second.mesh_buffer), args_config).append_to(*compile_time_args);
  const uint32_t emitted_count =
      static_cast<uint32_t>(compile_time_args->size() - before);
  ICHECK_EQ(emitted_count, 2U)
      << "Blackhole interleaved accessor compile-time ABI for buffer " << buffer_name
      << " must materialize exactly two uint32 values";
  if (expected_count != 0U) {
    ICHECK_EQ(expected_count, emitted_count)
        << "Blackhole interleaved accessor compile-time ABI count mismatch for buffer "
        << buffer_name;
  }
}

static void AppendAccessorCompileTimeArgs(
    const CompileTimeArgSpec& spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* compile_time_args) {
  ICHECK(!spec.buffer.empty())
      << "Blackhole interleaved accessor compile-time ABI requires explicit buffer binding for "
      << spec.name;
  AppendInterleavedAccessorCompileTimeArgs(spec.buffer, spec.count, spec.args_config_bits,
                                           buffer_bindings, compile_time_args);
}

static bool IsSupportedCommonRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr" || kind == "input_buffer_addr32" ||
         kind == "output_buffer_addr" || kind == "output_buffer_addr32" ||
         kind == "semaphore_id_u32";
}

static bool IsLogicalCoreNocRuntimeArgKind(const std::string& kind) {
  return kind == "logical_core_noc_x" || kind == "logical_core_noc_y";
}

static const RemoteCoreDescriptorSpec* ResolveRemoteCoreDescriptorSpec(
    const KernelSpec& kernel, const KernelArgSpec& arg_spec) {
  ICHECK(!arg_spec.identity.empty())
      << "Blackhole synchronization schema requires identity for runtime arg " << arg_spec.name
      << " kind=" << arg_spec.kind;
  const RemoteCoreDescriptorSpec* matched = nullptr;
  for (const auto& descriptor : kernel.remote_core_descriptors) {
    if (descriptor.identity != arg_spec.identity) {
      continue;
    }
    matched = &descriptor;
    break;
  }
  ICHECK(matched != nullptr)
      << "Blackhole synchronization schema requires a matching remote core descriptor for runtime arg "
      << arg_spec.name << " identity=" << arg_spec.identity;
  return matched;
}

static const SemaphoreBindingSpec* ResolveSemaphoreBindingSpec(const KernelSpec& kernel,
                                                              const KernelArgSpec& arg_spec) {
  const SemaphoreBindingSpec* matched = nullptr;
  for (const auto& binding : kernel.semaphore_bindings) {
    if (binding.name != arg_spec.name || binding.arg_kind != arg_spec.kind) {
      continue;
    }
    ICHECK(matched == nullptr)
        << "Blackhole synchronization schema requires a unique semaphore binding for runtime arg "
        << arg_spec.name << " kind=" << arg_spec.kind;
    matched = &binding;
  }
  ICHECK(matched != nullptr)
      << "Blackhole synchronization schema requires a matching semaphore binding for runtime arg "
      << arg_spec.name << " kind=" << arg_spec.kind;
  return matched;
}

static void ValidateLogicalCoreNocRuntimeArgs(const KernelSpec& kernel) {
  struct LogicalCorePairState {
    const KernelArgSpec* x{nullptr};
    const KernelArgSpec* y{nullptr};
  };

  std::unordered_map<std::string, LogicalCorePairState> pair_by_identity;
  for (const auto& arg_spec : kernel.runtime_args) {
    if (!IsLogicalCoreNocRuntimeArgKind(arg_spec.kind)) {
      continue;
    }
    ICHECK(!arg_spec.identity.empty())
        << "Blackhole synchronization schema requires identity for runtime arg " << arg_spec.name
        << " kind=" << arg_spec.kind;
    ICHECK(arg_spec.has_core_coord)
        << "Blackhole synchronization schema requires core_x/core_y for runtime arg "
        << arg_spec.name << " kind=" << arg_spec.kind;
    auto& pair = pair_by_identity[arg_spec.identity];
    if (arg_spec.kind == "logical_core_noc_x") {
      ICHECK(pair.x == nullptr)
          << "Blackhole synchronization core descriptor " << arg_spec.identity
          << " cannot define logical_core_noc_x more than once";
      pair.x = &arg_spec;
    } else {
      ICHECK(pair.y == nullptr)
          << "Blackhole synchronization core descriptor " << arg_spec.identity
          << " cannot define logical_core_noc_y more than once";
      pair.y = &arg_spec;
    }
  }

  for (const auto& entry : pair_by_identity) {
    const auto& pair = entry.second;
    ICHECK(pair.x != nullptr && pair.y != nullptr)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must define both logical_core_noc_x and logical_core_noc_y";
    ICHECK_EQ(pair.x->core_x, pair.y->core_x)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must use one logical core for logical_core_noc_x/y";
    ICHECK_EQ(pair.x->core_y, pair.y->core_y)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must use one logical core for logical_core_noc_x/y";

    auto descriptor_it =
        std::find_if(kernel.remote_core_descriptors.begin(), kernel.remote_core_descriptors.end(),
                     [&](const RemoteCoreDescriptorSpec& descriptor) {
                       return descriptor.identity == entry.first;
                     });
    ICHECK(descriptor_it != kernel.remote_core_descriptors.end())
        << "Blackhole synchronization core descriptor " << entry.first
        << " must be materialized into KernelSpec.remote_core_descriptors";
    ICHECK_EQ(descriptor_it->core_x, pair.x->core_x)
        << "Blackhole synchronization core descriptor " << entry.first
        << " core_x mismatch between runtime args and KernelSpec.remote_core_descriptors";
    ICHECK_EQ(descriptor_it->core_y, pair.x->core_y)
        << "Blackhole synchronization core descriptor " << entry.first
        << " core_y mismatch between runtime args and KernelSpec.remote_core_descriptors";
  }
}

static void ValidateKernelSynchronizationSchema(
    const KernelSpec& kernel, const std::unordered_set<uint32_t>& planned_semaphore_ids) {
  ValidateLogicalCoreNocRuntimeArgs(kernel);

  auto validate_semaphore_runtime_arg = [&](const KernelArgSpec& arg_spec) {
    if (arg_spec.kind != "semaphore_id_u32") {
      return;
    }
    const auto* binding = ResolveSemaphoreBindingSpec(kernel, arg_spec);
    ICHECK(planned_semaphore_ids.count(binding->semaphore_id))
        << "Blackhole synchronization schema requires semaphore binding " << binding->name
        << " to reference a planned semaphore id; missing id " << binding->semaphore_id;
  };

  for (const auto& arg_spec : kernel.common_runtime_args) {
    validate_semaphore_runtime_arg(arg_spec);
  }
  for (const auto& arg_spec : kernel.runtime_args) {
    validate_semaphore_runtime_arg(arg_spec);
  }
}

static void ValidateKernelDirectRuntimeSchema(const KernelSpec& kernel) {
  for (const auto& arg_spec : kernel.common_runtime_args) {
    ICHECK(IsSupportedCommonRuntimeArgKind(arg_spec.kind))
        << "Blackhole direct runtime only supports shared common runtime args for "
           "buffer addresses and semaphores; unsupported common runtime arg kind: "
        << arg_spec.kind;
  }

  for (const auto& accessor : kernel.accessors) {
    ValidateDirectRuntimeAccessorSpec(accessor.buffer, accessor.layout, accessor.memory_space,
                                      accessor.common_runtime_arg_count,
                                      accessor.args_config_bits);
  }

  for (const auto& spec : kernel.compile_time_arg_specs) {
    if (spec.kind != "interleaved_accessor_cta") {
      continue;
    }
    ICHECK(!spec.buffer.empty())
        << "Blackhole direct runtime requires explicit buffer binding for compile-time ABI "
        << spec.name;
    ValidateDirectRuntimeAccessorSpec(spec.buffer, spec.layout, spec.memory_space,
                                      /*common_runtime_arg_count=*/0, spec.args_config_bits);
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

static void ValidateExecutableSpecSynchronizationSchema(const std::string& func_name,
                                                        const ExecutableSpec& spec) {
  std::unordered_set<uint32_t> planned_semaphore_ids;
  for (const auto& semaphore : spec.semaphores) {
    planned_semaphore_ids.insert(semaphore.id);
  }
  for (const auto& kernel : spec.kernels) {
    ValidateKernelSynchronizationSchema(kernel, planned_semaphore_ids);
  }
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
    ValidateDirectRuntimeAccessorSpec(accessor.buffer, accessor.layout, accessor.memory_space,
                                      accessor.common_runtime_arg_count,
                                      accessor.args_config_bits);
    ICHECK_EQ(accessor.compile_time_arg_count, 2U)
        << "Blackhole direct runtime currently supports only interleaved accessors with two compile-time args";
    ICHECK_EQ(accessor.compile_time_arg_offset, expected_slot)
        << "Accessor compile-time offset mismatch for buffer " << accessor.buffer
        << ": got " << accessor.compile_time_arg_offset << ", expected " << expected_slot;
    AppendInterleavedAccessorCompileTimeArgs(accessor.buffer, accessor.compile_time_arg_count,
                                             accessor.args_config_bits, buffer_bindings,
                                             &compile_time_args);
    expected_slot += accessor.compile_time_arg_count;
  }
  return compile_time_args;
}

static const RuntimeBufferBinding& ResolveRuntimeBufferBinding(
    const KernelArgSpec& arg_spec,
    bool expect_output,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  ICHECK(!arg_spec.buffer.empty())
      << "Direct runtime requires explicit buffer binding for arg kind " << arg_spec.kind;
  auto it = buffer_bindings.find(arg_spec.buffer);
  ICHECK(it != buffer_bindings.end())
      << "Missing runtime buffer binding for " << arg_spec.buffer;
  ICHECK(it->second.is_output == expect_output)
      << "Runtime buffer role mismatch for " << arg_spec.buffer
      << ": expected output=" << expect_output;
  return it->second;
}

static uint32_t ResolveRuntimeSemaphoreId(
    const KernelSpec& kernel,
    const KernelArgSpec& arg_spec,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  const auto* binding = ResolveSemaphoreBindingSpec(kernel, arg_spec);
  auto semaphore_it = semaphore_ids.find(binding->semaphore_id);
  ICHECK(semaphore_it != semaphore_ids.end())
      << "Blackhole kernel semaphore binding " << binding->name
      << " references missing planned semaphore id " << binding->semaphore_id;
  return semaphore_it->second;
}

static CoreCoord ResolveLogicalCoreNocCoord(const KernelArgSpec& arg_spec,
                                            const KernelSpec& kernel,
                                            const IDevice& device) {
  const auto* descriptor = ResolveRemoteCoreDescriptorSpec(kernel, arg_spec);
  return device.worker_core_from_logical_core(
      CoreCoord{descriptor->core_x, descriptor->core_y});
}

static SynchronizationRuntimeContext BuildSynchronizationRuntimeContext(
    const IDevice& device, const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  return SynchronizationRuntimeContext{
      .device = &device,
      .semaphore_ids = &semaphore_ids,
  };
}

static SynchronizationRuntimeContext BuildCommonSynchronizationRuntimeContext(
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  return SynchronizationRuntimeContext{
      .device = nullptr,
      .semaphore_ids = &semaphore_ids,
  };
}

static bool TryAppendSynchronizationRuntimeArg(const KernelSpec& kernel,
                                               const KernelArgSpec& arg_spec,
                                               const SynchronizationRuntimeContext& sync_context,
                                               std::vector<uint32_t>* args) {
  if (arg_spec.kind == "semaphore_id_u32") {
    ICHECK(sync_context.semaphore_ids != nullptr)
        << "Blackhole synchronization runtime context is missing semaphore ids";
    args->push_back(ResolveRuntimeSemaphoreId(kernel, arg_spec, *sync_context.semaphore_ids));
    return true;
  }
  if (arg_spec.kind == "logical_core_noc_x") {
    ICHECK(sync_context.device != nullptr)
        << "Blackhole synchronization runtime context is missing device";
    const CoreCoord noc_core = ResolveLogicalCoreNocCoord(arg_spec, kernel, *sync_context.device);
    args->push_back(static_cast<uint32_t>(noc_core.x));
    return true;
  }
  if (arg_spec.kind == "logical_core_noc_y") {
    ICHECK(sync_context.device != nullptr)
        << "Blackhole synchronization runtime context is missing device";
    const CoreCoord noc_core = ResolveLogicalCoreNocCoord(arg_spec, kernel, *sync_context.device);
    args->push_back(static_cast<uint32_t>(noc_core.y));
    return true;
  }
  return false;
}

static void AppendRuntimeBufferAddressArg(
    const KernelArgSpec& arg_spec,
    bool expect_output,
    bool use_32bit_addr,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* args) {
  const auto& binding =
      ResolveRuntimeBufferBinding(arg_spec, expect_output, buffer_bindings);
  const uint64_t addr = binding.mesh_buffer->address();
  args->push_back(static_cast<uint32_t>(addr & 0xFFFFFFFF));
  if (!use_32bit_addr) {
    args->push_back(static_cast<uint32_t>(addr >> 32));
  }
}

static bool TryAppendSharedRuntimeArg(
    const KernelArgSpec& arg_spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* args) {
  if (arg_spec.kind == "input_buffer_addr") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/false, /*use_32bit_addr=*/false,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "input_buffer_addr32") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/false, /*use_32bit_addr=*/true,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "output_buffer_addr") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/true, /*use_32bit_addr=*/false,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "output_buffer_addr32") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/true, /*use_32bit_addr=*/true,
                                  buffer_bindings, args);
    return true;
  }
  return false;
}

struct DirectRuntimeWorkContext {
  uint32_t num_k_tiles = 0;
  uint32_t logical_grid_x = 0;
  uint32_t logical_n_tiles = 0;
  uint32_t work_linear_id = 0;
  uint32_t bx = 0;
  uint32_t by = 0;
  bool has_gemm_compute_op = false;
};

static DirectRuntimeWorkContext BuildDirectRuntimeWorkContext(const KernelSpec& kernel,
                                                             const ExecutableSpec& spec,
                                                             uint32_t current_work_linear_id) {
  (void)kernel;
  DirectRuntimeWorkContext context;
  context.logical_grid_x = GetRuntimeLogicalGridX(spec);
  context.work_linear_id = current_work_linear_id;
  context.bx = context.logical_grid_x == 0 ? 0 : (context.work_linear_id % context.logical_grid_x);
  context.by = context.logical_grid_x == 0 ? 0 : (context.work_linear_id / context.logical_grid_x);
  const auto gemm_op = GetPrimaryGemmCompute(spec);
  context.has_gemm_compute_op = gemm_op.enabled && gemm_op.kind == "gemm";
  if (context.has_gemm_compute_op) {
    context.num_k_tiles = GetRuntimeNumKTiles(spec);
    context.logical_n_tiles = GetRuntimeLogicalNTiles(spec);
  }
  return context;
}

static const PerWorkArgSpec* FindPerWorkArgSpec(const std::vector<PerWorkArgSpec>& per_work_arg_specs,
                                                const KernelArgSpec& arg_spec) {
  ICHECK(!arg_spec.identity.empty())
      << "Blackhole direct runtime per-work binding requires runtime arg identity for "
      << arg_spec.name << " kind=" << arg_spec.kind;
  auto it = std::find_if(per_work_arg_specs.begin(), per_work_arg_specs.end(),
                         [&](const PerWorkArgSpec& spec) {
                           return spec.arg_identity == arg_spec.identity;
                         });
  return it == per_work_arg_specs.end() ? nullptr : &(*it);
}

static uint32_t EvaluatePerWorkArgSpec(const PerWorkArgSpec& spec,
                                       const DirectRuntimeWorkContext& context) {
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
    return context.work_linear_id;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockX) {
    return context.bx;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockY) {
    return context.by;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles) {
    ICHECK(context.has_gemm_compute_op)
        << "Blackhole direct runtime per_work_arg_spec requires GEMM compute_op for "
        << spec.arg_identity;
    return context.num_k_tiles;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceComputeLogicalNTiles) {
    ICHECK(context.has_gemm_compute_op)
        << "Blackhole direct runtime per_work_arg_spec requires GEMM compute_op for "
        << spec.arg_identity;
    return context.logical_n_tiles;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceConstant) {
    return spec.constant_value;
  }
  LOG(FATAL) << "Unsupported Blackhole per_work_arg_spec value_source " << spec.value_source
             << " for arg " << spec.arg_identity;
  return 0;
}

static bool TryAppendPerWorkRuntimeArg(const KernelSpec& kernel,
                                       const KernelArgSpec& arg_spec,
                                       const std::vector<PerWorkArgSpec>& per_work_arg_specs,
                                       const DirectRuntimeWorkContext& context,
                                       size_t* scalar_index,
                                       const std::vector<uint32_t>& scalar_args,
                                       std::vector<uint32_t>* args) {
  (void)kernel;

  if (arg_spec.kind == "work_linear_id" || arg_spec.kind == "current_work_linear_id") {
    args->push_back(context.work_linear_id);
    return true;
  }
  if (const PerWorkArgSpec* spec = FindPerWorkArgSpec(per_work_arg_specs, arg_spec)) {
    args->push_back(EvaluatePerWorkArgSpec(*spec, context));
    return true;
  }
  if (arg_spec.kind == "scalar_u32") {
    ICHECK(*scalar_index < scalar_args.size())
        << "Spec requested more scalar args than provided";
    args->push_back(scalar_args[(*scalar_index)++]);
    return true;
  }
  return false;
}

static std::vector<uint32_t> BuildCommonRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  std::vector<uint32_t> args;
  const auto sync_context = BuildCommonSynchronizationRuntimeContext(semaphore_ids);
  for (const auto& arg_spec : kernel.common_runtime_args) {
    if (TryAppendSynchronizationRuntimeArg(kernel, arg_spec, sync_context, &args)) {
      continue;
    }
    if (!TryAppendSharedRuntimeArg(arg_spec, buffer_bindings, &args)) {
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
    const std::vector<uint32_t>& scalar_args) {
  std::vector<uint32_t> args;
  size_t scalar_index = 0;
  const DirectRuntimeWorkContext context =
      BuildDirectRuntimeWorkContext(kernel, spec, current_work_linear_id);
  const auto sync_context = BuildSynchronizationRuntimeContext(device, semaphore_ids);

  for (const auto& arg_spec : kernel.runtime_args) {
    if (TryAppendSynchronizationRuntimeArg(kernel, arg_spec, sync_context, &args)) {
      continue;
    }
    if (TryAppendSharedRuntimeArg(arg_spec, buffer_bindings, &args)) {
      continue;
    }
    if (!TryAppendPerWorkRuntimeArg(kernel, arg_spec, kernel.per_work_arg_specs, context,
                                    &scalar_index, scalar_args,
                                    &args)) {
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
    ValidateExecutableSpecSynchronizationSchema(entry.first, entry.second);
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
  if (!spec.direct_runtime_unsupported_reasons.empty()) {
    std::ostringstream os;
    for (size_t i = 0; i < spec.direct_runtime_unsupported_reasons.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << spec.direct_runtime_unsupported_reasons[i];
    }
    LOG(FATAL) << "Blackhole direct runtime is not supported for " << func_name << ": "
               << os.str();
  }
  ValidateGemmComputeDirectRuntimeConstraints(spec);
  for (const auto& kernel_spec : spec.kernels) {
    ValidateKernelDirectRuntimeConstraints(kernel_spec);
  }

  const std::vector<DirectLaunchWave> launch_waves = BuildDirectLaunchWaves(spec, func_name);
  uint64_t total_work_items = 0;
  for (const auto& wave : launch_waves) {
    total_work_items += wave.work_items.size();
  }

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

  LOG(INFO) << "Direct path: executing " << total_work_items << " logical work items across "
            << launch_waves.size() << " launch wave(s) for " << func_name;

  for (size_t wave_index = 0; wave_index < launch_waves.size(); ++wave_index) {
    const auto& launch_wave = launch_waves[wave_index];
    const CoreRangeSet launch_core_ranges(launch_wave.launch_cores);
    LOG(INFO) << "Direct path: launch wave " << (wave_index + 1) << "/" << launch_waves.size()
              << " with " << launch_wave.work_items.size() << " logical work items across "
              << launch_wave.launch_cores.size() << " launch cores for " << func_name;

    Program program = CreateProgram();
    const std::unordered_map<uint32_t, uint32_t> semaphore_ids =
        CreateSemaphoresFromSpec(program, spec);

    CreateCircularBuffersFromSpec(program, launch_core_ranges, spec);

    std::vector<KernelHandle> kernels = CreateProgramKernelsFromSpec(
        program, launch_core_ranges, spec, runtime_buffer_state.runtime_buffers, semaphore_ids,
        kernel_paths);

    ApplyWorkItemRuntimeArgs(program, spec, kernels, launch_wave.work_items, *mesh_device,
                             runtime_buffer_state.runtime_buffers, semaphore_ids, scalar_args);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    LOG(INFO) << "Direct path: enqueue multi-core workload for " << func_name << " wave "
              << (wave_index + 1) << "/" << launch_waves.size();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
  }

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
  // Direct runtime requires explicit schema-derived name->role bindings.
  std::unordered_map<std::string, bool> buffer_is_output_by_name;
  auto append_buffer_contract = [&](const std::vector<KernelArgSpec>& runtime_args) {
    for (const auto& arg : runtime_args) {
      const bool is_input = arg.kind == "input_buffer_addr32" || arg.kind == "input_buffer_addr";
      const bool is_output = arg.kind == "output_buffer_addr32" || arg.kind == "output_buffer_addr";
      if (!is_input && !is_output) {
        continue;
      }
      ICHECK(!arg.buffer.empty())
          << "Blackhole direct runtime requires explicit buffer role schema for arg "
          << arg.name << " kind=" << arg.kind;
      auto [it, inserted] = buffer_is_output_by_name.emplace(arg.buffer, is_output);
      ICHECK(inserted || it->second == is_output)
          << "Blackhole direct runtime buffer role mismatch for " << arg.buffer;
      if (is_input) {
        ICHECK(!is_output);
      }
    }
  };
  append_buffer_contract(info_.runtime_args);
  append_buffer_contract(info_.common_runtime_args);
  ICHECK(!buffer_is_output_by_name.empty())
      << "Blackhole direct runtime requires explicit buffer role schema";

  // Collect arguments
  std::vector<RuntimeTensorBinding> buffer_args;
  std::vector<uint32_t> scalars;
  std::vector<std::string> output_names;

  for (size_t i = 0; i < info_.tvm_arg_types.size(); ++i) {
    if (info_.tvm_is_buffer_arg[i]) {
      DLTensor* tensor = ExtractTensorArg(args[i], void_args != nullptr ? void_args[i] : nullptr);
      ICHECK_LT(i, info_.tvm_arg_names.size())
          << "Blackhole direct runtime requires formal buffer identity for arg index " << i;
      const std::string buffer_name = info_.tvm_arg_names[i];
      ICHECK(!buffer_name.empty())
          << "Blackhole direct runtime requires formal buffer identity for arg index " << i;
      auto role_it = buffer_is_output_by_name.find(buffer_name);
      ICHECK(role_it != buffer_is_output_by_name.end())
          << "Blackhole direct runtime requires explicit buffer role binding for "
          << buffer_name;
      bool is_out = role_it->second;
      buffer_args.push_back(RuntimeTensorBinding{buffer_name, tensor, is_out});
      if (is_out) {
        output_names.push_back(buffer_name);
      }
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
