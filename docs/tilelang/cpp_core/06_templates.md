# TileLang Template System

## Overview

TileLang's template system provides hardware-specific CUDA/HIP/C++ implementations for core GPU operations. These templates are included during kernel compilation and provide optimized implementations of matrix multiplication, memory copy, reduction, and synchronization primitives.

## Template Directory Structure

```
/root/dev/vibe_dsl/tilelang/src/tl_templates/
├── cpp/                    # C++ CPU templates
│   ├── common.h           # Common CPU definitions
│   └── gemm.h             # CPU GEMM templates
├── cpu/                    # CPU-specific templates
│   ├── common.h
│   └── gemm.h
├── cuda/                   # CUDA GPU templates
│   ├── common.h           # Core CUDA definitions
│   ├── atomic.h           # Atomic operations
│   ├── barrier.h          # Synchronization barriers
│   ├── cluster.h          # Thread cluster operations
│   ├── copy.h             # Memory copy intrinsics
│   ├── copy_sm90.h        # Hopper-specific copy
│   ├── copy_sm100.h       # Blackwell-specific copy
│   ├── gemm.h             # GEMM dispatch header
│   ├── gemm_sm70.h        # Volta MMA
│   ├── gemm_sm80.h        # Ampere MMA
│   ├── gemm_sm90.h        # Hopper WGMMA
│   ├── gemm_sm100.h       # Blackwell TCgen05
│   ├── gemm_sm120.h       # Future arch
│   ├── gemm_mma.h         # MMA primitive wrapper
│   ├── reduce.h           # Reduction primitives
│   ├── intrin.h           # Warp-level intrinsics
│   └── instruction/       # Low-level PTX instructions
│       ├── mma.h
│       ├── mma_sm70.h
│       ├── wgmma.h
│       └── tcgen05mma.h
└── hip/                    # AMD HIP templates
    ├── common.h
    ├── atomic.h
    ├── copy.h
    └── gemm.h
```

## Common CUDA Template (`common.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/common.h`, this is the foundational header providing:

### Type Definitions (lines 16-34)

```cpp
using cutlass::bfloat16_t;
using cutlass::half_t;
using cutlass::tfloat32_t;
using int4_t = int4;

// Fast math functions from CUTLASS
#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos
#define htanh cutlass::fast_tanh
```

### Device Function Macros (lines 36-38)

```cpp
#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH
```

### Error Checking Macros (lines 40-58)

```cpp
#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    cudaError_t __err = (stmt);                                                \
    if (__err != cudaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, cudaGetErrorName(__err), cudaGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)
```

### Data Type Enum (lines 261-285)

```cpp
enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat8_e4m3 = 10,
  kFloat8_e5m2 = 11,
  kFloat16 = 12,
  kBFloat16 = 13,
  kFloat16x2 = 14,
  kFloat32 = 15,
  kTensorFloat32 = 16,
  kFloat64 = 17,
  // ... bit types
};
```

### GMMA Descriptor (lines 287-346)

The WGMMA (Warp Group Matrix Multiply Accumulate) descriptor for Hopper GPUs:

```cpp
union GmmaDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  struct {
    uint16_t start_address_ : 14, : 2;        // bit [0,14)
    uint16_t leading_byte_offset_ : 14, : 2;  // bit [16,30)
    uint16_t stride_byte_offset_ : 14, : 2;   // bit [32,46)
    uint8_t : 1, base_offset_ : 3, : 4;       // bit [49,52)
    uint8_t : 6, layout_type_ : 2;            // bit [62,64)
  } bitfield;

  template <typename T>
  CUTE_HOST_DEVICE constexpr GmmaDescriptor operator+(const T &offset) const {
    GmmaDescriptor ret;
    ret.reg32_[0] = reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = reg32_[1];
    return ret;
  }
};
```

### TCgen05 Descriptor (lines 348-410)

For Blackwell (SM100+) Tensor Core operations:

```cpp
union Tcgen05SMemDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];

  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, version_ : 2;
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;
    uint8_t : 5, layout_type_ : 3;
  } bitfield;

  struct {
    uint32_t lo;
    uint32_t hi;
  } words;
};
```

### Warp Shuffle Specializations (lines 679-773)

Optimized warp shuffle for 16-bit types:

```cpp
// Generic passthrough
template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

// Specialization for cutlass::half_t - avoids FP32 conversion
template <>
TL_DEVICE half_t shfl_xor_sync(unsigned mask, half_t val, int laneMask) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_xor_sync(mask, raw32, laneMask);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<half_t &>(ret16);
}
```

## Memory Copy Templates (`copy.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/copy.h`.

### Async Copy Primitives (lines 16-79)

```cpp
TL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> TL_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}
```

### Global-to-Shared Copy (lines 29-52)

```cpp
template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;"
        ::"r"(addr), "l"((void const *)(global_ptr)), "n"(N));
  } else {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;"
        ::"r"(addr), "l"((void const *)(global_ptr)), "n"(N));
  }
}
```

### Global Memory Load Intrinsics (lines 82-169)

Template-specialized loads with predication:

```cpp
// Primary template declaration
template <typename AccessType, int LoadBytes> struct global_load;

// ldg32: Load 32 bits
template <typename AccessType> struct global_load<AccessType, 4> {
  TL_DEVICE global_load(AccessType &D, void const *ptr, bool pred_guard) {
    unsigned &data = reinterpret_cast<unsigned &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %2, 0;\n"
                 "  mov.b32 %0, %3;\n"
                 "  @p ld.global.u32 %0, [%1];\n"
                 "}\n"
                 : "=r"(data)
                 : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

// ldg128: Load 128 bits
template <typename AccessType> struct global_load<AccessType, 16> {
  TL_DEVICE global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 "}\n"
                 : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                 : "l"(ptr), "r"((int)pred_guard), ...);
  }
};
```

## GEMM Templates

### Architecture Dispatch (`gemm.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/gemm.h`:

```cpp
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1200))
#include "gemm_sm120.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1000))
#include "gemm_sm100.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "gemm_sm90.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 890))
#include "gemm_sm89.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
#include "gemm_sm80.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 700))
#include "gemm_sm70.h"
#endif
```

### SM80/Ampere GEMM (`gemm_sm80.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/gemm_sm80.h`:

```cpp
#include "gemm_mma.h"
namespace tl {
using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;
} // namespace tl
```

### MMA Instruction Wrapper (`gemm_mma.h`)

Provides unified interface for MMA operations across architectures:

```cpp
namespace tl_mma {
// Register-Shared mode (operands in registers and shared memory)
template <typename... Args>
TL_DEVICE void gemm_rs(Args... args) {
  // Architecture-specific implementation
}

// Shared-Register mode
template <typename... Args>
TL_DEVICE void gemm_sr(Args... args) { }

// Shared-Shared mode
template <typename... Args>
TL_DEVICE void gemm_ss(Args... args) { }
} // namespace tl_mma
```

### WGMMA for Hopper (`instruction/wgmma.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/instruction/wgmma.h`:

```cpp
// WGMMA (Warp Group Matrix Multiply Accumulate) for SM90+
// Operates on 128x128 tiles with tensor cores

template <int M, int N, int K, typename AType, typename BType, typename CType>
TL_DEVICE void wgmma_mma_sync(CType *d, AType *a, BType *b) {
  // Uses cute::GMMA or raw PTX depending on configuration
  // Supports various swizzle modes (128B, 64B, 32B)
}
```

## Warp-Level Intrinsics (`intrin.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/intrin.h`:

### Thread Indexing (lines 26-60)

```cpp
namespace detail {
TL_DEVICE int linear_thread_idx_in_block() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_group_idx(int warp_size = detail::default_warp_size(),
                                  int warps_per_group = detail::default_warps_per_group()) {
  int threads_per_group = warp_size * warps_per_group;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}
```

### Warp Group Synchronization (lines 62-132)

```cpp
#if __CUDA_ARCH_LIST__ >= 900
TL_DEVICE void warpgroup_arrive() { cute::warpgroup_arrive(); }
TL_DEVICE void warpgroup_commit_batch() { cute::warpgroup_commit_batch(); }

template <int NumMma> TL_DEVICE void warpgroup_wait() {
  cute::warpgroup_wait<NumMma>();
}

TL_DEVICE void warpgroup_fence_operand(uint32_t *regs, int count) {
  #pragma unroll
  for (int i = 0; i < count; ++i) {
    cute::warpgroup_fence_operand(regs[i]);
  }
}

// Register allocation for warp specialization
template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
#endif
```

### Thread Election (lines 88-123)

```cpp
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    // Elect one thread in entire block
    return cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync();
  } else if constexpr (thread_extent == 32) {
    // Elect one thread per warp
    return cute::elect_one_sync();
  }
  // General case: elect one thread per thread_extent group
  return __shfl_sync(0xffffffff,
                     (threadIdx.x / 32) % (thread_extent / 32),
                     0) == 0 && cute::elect_one_sync();
}
```

## Barrier Templates (`barrier.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/barrier.h`:

```cpp
// mbarrier for TMA synchronization on Hopper
TL_DEVICE void mbarrier_init(void *barrier, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
               ::"r"(smem_ptr_to_uint(barrier)), "r"(count));
}

TL_DEVICE void mbarrier_arrive(void *barrier) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
               ::"r"(smem_ptr_to_uint(barrier)));
}

TL_DEVICE void mbarrier_wait(void *barrier, uint32_t phase) {
  asm volatile("mbarrier.try_wait.parity.shared.b64 _, [%0], %1;\n"
               ::"r"(smem_ptr_to_uint(barrier)), "r"(phase));
}
```

## Reduction Templates (`reduce.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/reduce.h`:

```cpp
namespace tl {

// Warp-level reduction
template <typename T, typename Op>
TL_DEVICE T warp_reduce(T val, Op op) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block-level reduction using shared memory
template <typename T, typename Op>
TL_DEVICE T block_reduce(T val, Op op, void *shared_mem) {
  // Warp-level reduce
  val = warp_reduce(val, op);

  // Store to shared memory
  if (get_lane_idx() == 0) {
    ((T*)shared_mem)[get_warp_idx()] = val;
  }
  __syncthreads();

  // Final reduction by warp 0
  if (get_warp_idx() == 0) {
    val = (get_lane_idx() < num_warps) ? ((T*)shared_mem)[get_lane_idx()] : T(0);
    val = warp_reduce(val, op);
  }
  return val;
}

} // namespace tl
```

## Atomic Operations (`atomic.h`)

Located at `/root/dev/vibe_dsl/tilelang/src/tl_templates/cuda/atomic.h`:

```cpp
namespace tl {

// Atomic add for various types
template <typename T>
TL_DEVICE T atomic_add(T *addr, T val);

template <>
TL_DEVICE float atomic_add(float *addr, float val) {
  return atomicAdd(addr, val);
}

template <>
TL_DEVICE half_t atomic_add(half_t *addr, half_t val) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  return atomicAdd(reinterpret_cast<__half*>(addr), val.to_half());
  #else
  // Emulation for older architectures
  #endif
}

// Atomic max/min
template <typename T>
TL_DEVICE T atomic_max(T *addr, T val) {
  return atomicMax(addr, val);
}

} // namespace tl
```

## Template Compilation Integration

The templates are integrated into the compilation process in `/root/dev/vibe_dsl/tilelang/tilelang/engine/lower.py:72-76`:

```python
options = [
    "-std=c++17",
    "-I" + TILELANG_TEMPLATE_PATH,  # Includes src/tl_templates
    "-I" + CUTLASS_INCLUDE_DIR,
]
```

## Summary

The TileLang template system provides:

1. **Architecture Abstraction**: Automatic selection of appropriate implementations based on `__CUDA_ARCH_LIST__`
2. **Optimized Primitives**: Highly tuned implementations of copy, GEMM, reduction, and synchronization
3. **Type Safety**: C++ templates with specializations for different data types
4. **PTX Integration**: Direct inline assembly for performance-critical operations
5. **Multi-Backend Support**: CUDA, HIP, and CPU backends with unified interfaces

These templates are the foundation for generating efficient GPU kernels across different hardware generations from Volta (SM70) to Blackwell (SM100+) and beyond.
