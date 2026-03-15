# Phase 3: GEMM 支持 - 矩阵乘法实现

## 任务目标

实现 TileLang T.gemm() 算子在 Blackhole 后端的完整支持，包括：
1. **Compute Kernel CodeGen**: 生成调用 TT-Metal LLK matmul_tiles 的计算代码
2. **数据流支持**: 结合 Phase 2 的 R/C/W 拆分，实现完整的 GEMM 数据流
3. **TT-Sim 验证**: 在仿真器上验证小尺寸和中等尺寸 GEMM 的正确性
4. **多核并行**: 利用 140 cores 实现矩阵并行计算

## 基本信息

- **任务ID**: phase3_gemm
- **所属阶段**: Phase 3
- **前置任务**: phase2_split_blackhole_kernel, phase2_plan_blackhole_cb, phase2_assign_blackhole_cores
- **负责人**: -
- **状态**: 设计中

---

## 目标

实现完整的 GEMM (General Matrix Multiply) 支持，使以下 TileLang 代码可以编译到 Blackhole：

```python
@T.prim_func
def gemm_kernel(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), "float16")
        B_shared = T.alloc_shared((block_K, block_N), "float16")
        C_local = T.alloc_fragment((block_M, block_N), "float32")

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by*block_M, k*block_K], A_shared)
            T.copy(B[k*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])
```

---

## 设计概要

### 输入

经过 Phase 2 Passes 处理后的 PrimFunc：
- `reader_func`: 从 DRAM 读取 A、B 矩阵到 CB
- `compute_func`: 在 CB 上执行 matmul_tiles 计算
- `writer_func`: 将 C 矩阵结果写回 DRAM

### 输出

TT-Metal 风格的三个 kernel：
- **Reader Kernel (BRISC)**: `noc_async_read` A、B 到 CB
- **Compute Kernel (TRISC)**: `matmul_tiles` 计算
- **Writer Kernel (NCRISC/BRISC)**: `noc_async_write` C 到 DRAM

### 核心逻辑

```
TileLang DSL
    ↓ LowerAndLegalize
TIR with T.gemm()
    ↓ Phase 2 Passes
├─ AssignBlackholeCores (140 cores)
├─ PlanBlackholeCB (64 CBs, 1.5MB)
└─ SplitBlackholeKernel (R/C/W)
    ↓ Phase 3 CodeGen
codegen_blackhole.cc
    ├─ GenerateReaderKernel() → A, B DRAM → CB
    ├─ GenerateComputeKernel() → matmul_tiles(CB → CB/Local)
    └─ GenerateWriterKernel() → C CB → DRAM
    ↓ Runtime
TT-Sim / Blackhole Hardware
```

---

## 技术调研

### TT-Metal LLK matmul_tiles API

参考 `tt_metal/third_party/tt_llk/` 和官方示例：

```cpp
// Compute Kernel (TRISC - UNPACK/MATH/PACK)
#include "compute_kernel_api/matmul.h"

void kernel_main() {
    // Runtime args
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // CB IDs
    constexpr uint32_t cb_id_in0 = 0;  // A matrix
    constexpr uint32_t cb_id_in1 = 1;  // B matrix
    constexpr uint32_t cb_id_out = 2;  // C matrix

    // Tile dimensions
    constexpr uint32_t Mt = 4;  // M tiles
    constexpr uint32_t Kt = 4;  // K tiles
    constexpr uint32_t Nt = 4;  // N tiles

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for input tiles
        cb_wait_front(cb_id_in0, Kt);
        cb_wait_front(cb_id_in1, Kt);

        // Reserve output space
        cb_reserve_back(cb_id_out, 1);

        // Matrix multiply: C = A @ B
        matmul_tiles(
            cb_id_in0,      // A CB
            cb_id_in1,      // B CB
            cb_id_out,      // C CB
            0,              // A tile index start
            0,              // B tile index start
            0,              // C tile index
            Kt,             // number of K tiles to accumulate
            Mt, Nt          // output tile dimensions
        );

        // Release input tiles
        cb_pop_front(cb_id_in1, Kt);
        cb_pop_front(cb_id_in0, Kt);

        // Push output tile
        cb_push_back(cb_id_out, 1);
    }
}
```

### matmul_tiles 参数分析

| 参数 | 说明 | Blackhole 约束 |
|------|------|----------------|
| `in0_cb` | A 矩阵 CB | FP16/BF16 tile |
| `in1_cb` | B 矩阵 CB | FP16/BF16 tile |
| `out_cb` | C 矩阵 CB | FP32 tile (累加) |
| `in0_tile_start` | A 起始 tile | 0 ~ Kt-1 |
| `in1_tile_start` | B 起始 tile | 0 ~ Kt-1 |
| `out_tile` | C 输出 tile | 0 |
| `num_tiles` | K 维度 tile 数 | 受 L1 限制 |
| `Mt, Nt` | 输出 tile 维度 | 通常 32x32 |

### Compute Engine 架构

Blackhole TRISC 包含三个 RISC-V 核心：
- **UNPACK**: 从 CB 读取数据到 SrcA/SrcB 寄存器
- **MATH**: 执行矩阵乘法运算 (FPU/SFPU)
- **PACK**: 将结果写回 CB

`matmul_tiles` 是高层 API，内部协调三个核心。

---

## 实现方案

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A: 直接生成 matmul_tiles 调用 | 简洁，复用 LLK | 需要理解内部细节 | ✅ 选择 |
| B: 手写汇编指令 | 极致性能 | 复杂度高，难维护 | - |
| C: 使用 TT-Metal OpLibrary | 更高级抽象 | 灵活性差 | - |

### 方案 A 详细设计

#### 1. CodeGen 扩展

在 `codegen_blackhole.h` 中添加 Compute Kernel 生成：

```cpp
class CodeGenBlackhole : public CodeGenCHost {
 public:
  // Phase 1: Copy kernels
  std::string GenerateReaderKernel(...);
  std::string GenerateWriterKernel(...);

  // Phase 3: Compute kernel
  std::string GenerateComputeKernel(const PrimFunc& func);

 private:
  // GEMM specific
  void PrintMatmulTiles(const std::string& cb_a, const std::string& cb_b,
                        const std::string& cb_c, int num_k_tiles);
  void PrintCBWaitFront(const std::string& cb_name, int num_tiles);
  void PrintCBPopFront(const std::string& cb_name, int num_tiles);
};
```

#### 2. Compute Kernel 生成模板

```cpp
std::string CodeGenBlackhole::GenerateComputeKernel(
    const PrimFunc& func,
    const CBConfig& cb_a,
    const CBConfig& cb_b,
    const CBConfig& cb_c) {
  std::ostringstream os;

  // Header
  os << "#include \"compute_kernel_api/matmul.h\"\n";
  os << "#include \"compute_kernel_api/tile_move_copy.h\"\n\n";

  // Kernel main
  os << "void kernel_main() {\n";
  os << "  // Runtime args\n";
  os << "  uint32_t num_blocks = get_arg_val<uint32_t>(0);\n";
  os << "  uint32_t K_tiles = get_arg_val<uint32_t>(1);\n\n";

  // CB configuration
  os << "  constexpr uint32_t cb_in0 = " << cb_a.id << ";\n";
  os << "  constexpr uint32_t cb_in1 = " << cb_b.id << ";\n";
  os << "  constexpr uint32_t cb_out = " << cb_c.id << ";\n\n";

  // Main loop
  os << "  for (uint32_t block = 0; block < num_blocks; block++) {\n";
  os << "    // Wait for input tiles\n";
  os << "    cb_wait_front(cb_in0, K_tiles);\n";
  os << "    cb_wait_front(cb_in1, K_tiles);\n";
  os << "    cb_reserve_back(cb_out, 1);\n\n";

  // Matmul
  os << "    // Compute: C += A @ B\n";
  os << "    matmul_tiles(cb_in0, cb_in1, cb_out,\n";
  os << "                 0, 0, 0,\n";
  os << "                 K_tiles, " << cb_a.Mt << ", " << cb_b.Nt << ");\n\n";

  // Cleanup
  os << "    cb_pop_front(cb_in0, K_tiles);\n";
  os << "    cb_pop_front(cb_in1, K_tiles);\n";
  os << "    cb_push_back(cb_out, 1);\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}
```

#### 3. CB 分配策略

对于 GEMM，需要 3 个 CB：

```
CB 0: A matrix tiles (FP16/BF16)
CB 1: B matrix tiles (FP16/BF16)
CB 2: C matrix tiles (FP32)

Memory layout per core:
- A: block_M x block_K x 2 bytes (FP16)
- B: block_K x block_N x 2 bytes (FP16)
- C: block_M x block_N x 4 bytes (FP32)
- Total < 1.5MB per core
```

#### 4. 完整的 R/C/W Kernel 流程

```
Reader (BRISC):
  for k in 0..K_tiles:
    cb_reserve_back(cb0, 1)
    noc_async_read(A_dram + k*tile_size, cb0_addr, tile_size)
    cb_push_back(cb0, 1)

    cb_reserve_back(cb1, 1)
    noc_async_read(B_dram + k*tile_size, cb1_addr, tile_size)
    cb_push_back(cb1, 1)

Compute (TRISC):
  cb_wait_front(cb0, K_tiles)
  cb_wait_front(cb1, K_tiles)
  cb_reserve_back(cb2, 1)
  matmul_tiles(cb0, cb1, cb2, 0, 0, 0, K_tiles, Mt, Nt)
  cb_pop_front(cb0, K_tiles)
  cb_pop_front(cb1, K_tiles)
  cb_push_back(cb2, 1)

Writer (NCRISC/BRISC):
  cb_wait_front(cb2, 1)
  noc_async_write(cb2_addr, C_dram, tile_size)
  cb_pop_front(cb2, 1)
```

---

## 测试计划

### 1. 单元测试

**文件**: `tests/target/test_codegen_blackhole_gemm.cc`

```cpp
TEST(CodeGenBlackholeGEMM, SimpleMatmulTiles) {
  // 32x32x32 GEMM (1 tile each)
  // Input: A[32,32] x B[32,32] -> C[32,32]
  // Verify generated code contains matmul_tiles
}

TEST(CodeGenBlackholeGEMM, MultiTileAccumulate) {
  // 64x64x64 GEMM (2x2x2 tiles)
  // Verify K-dimension accumulation
}

TEST(CodeGenBlackholeGEMM, DifferentDtypes) {
  // FP16 x FP16 -> FP32
  // BF16 x BF16 -> FP32
}
```

### 2. TT-Sim 验证

**测试尺寸**:
- 小尺寸: 32x32x32 (1 tile)
- 中尺寸: 128x128x128 (4x4x4 tiles)
- 大尺寸: 256x256x256 (8x8x8 tiles)

**验证内容**:
- 结果与 CPU reference 对比
- 所有元素误差 < 1e-3 (FP16) / < 1e-6 (FP32)

### 3. 多核并行测试

**测试配置**:
- 2x2 cores (4 cores)
- 14x10 cores (all 140 cores)
- 验证分块计算结果拼接正确

---

## 实施步骤

### Step 1: TT-Metal LLK 调研 (1 天)
- [ ] 阅读 `tt_llk/blackhole/llk_io.h`
- [ ] 阅读 `tt_llk/blackhole/llk_math_matmul.h`
- [ ] 运行官方 matmul 示例

### Step 2: Compute Kernel CodeGen (2 天)
- [ ] 扩展 `codegen_blackhole.h` 接口
- [ ] 实现 `GenerateComputeKernel()`
- [ ] 实现 `PrintMatmulTiles()`
- [ ] 单元测试

### Step 3: CB 分配集成 (1 天)
- [ ] 更新 `PlanBlackholeCB` 支持 3 CB 场景
- [ ] 验证 L1 内存约束

### Step 4: TT-Sim 验证 (2 天)
- [ ] 创建 `phase3_gemm_ttsim_test.cpp`
- [ ] 小尺寸验证
- [ ] 中尺寸验证

### Step 5: 多核并行 (2 天)
- [ ] 集成 `AssignBlackholeCores`
- [ ] 分块 GEMM 测试

---

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| LLK API 理解偏差 | 中 | 高 | 先跑通官方示例，再移植 |
| TT-Sim 不支持 matmul | 中 | 高 | 准备纯 CPU 验证方案 |
| FP32 累加精度问题 | 低 | 中 | 与 CPU reference 对比 |
| L1 内存不足 | 低 | 中 | 动态 tile size 调整 |

---

## 参考

- TT-Metal LLK: `tt_metal/third_party/tt_llk/blackhole/`
- Matmul 示例: `tt_metal/programming_examples/matmul/`
- Compute API: `tt_metal/hw/inc/compute_kernel_api/`
- Phase 2: `phase2_split_blackhole_kernel.md`

---

## 开发记录

### 2026-03-16

- **完成**: Phase 3 设计文档创建
- **设计**: 确定方案 A (直接生成 matmul_tiles)
- **计划**: 开始 LLK 调研

### 2026-03-16 (调研完成)

- **完成**: TT-Metal LLK 调研
- **发现**: 关键 API 和官方示例

**关键 API 文档** (`tt_metal/hw/inc/api/compute/matmul.h`):
```cpp
// Initialization - 必须在 matmul_tiles 之前调用
void mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id,
             const uint32_t transpose = 0);

// Tile-wise matrix multiplication
void matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id,
                  uint32_t in0_tile_index, uint32_t in1_tile_index,
                  uint32_t dst_tile_index);

// Block-wise matrix multiplication (for larger tiles)
void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                  uint32_t in0_tile_index, uint32_t in1_tile_index,
                  uint32_t idst, const uint32_t transpose,
                  uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim);
```

**官方示例** (`matmul/matmul_single_core/`):
- **Reader**: 使用 `noc_async_read_tile` 读取 A、B 矩阵到 CB
- **Compute**: 使用 `mm_init` + `matmul_tiles` + `pack_tile`
- **Writer**: 使用 `noc_async_write_tile` 写入结果

**关键发现**:
1. `matmul_tiles` 自动累加到 DST 寄存器 (DST += A*B)
2. 必须先调用 `tile_regs_acquire()` 清零 DST 寄存器
3. 计算完成后调用 `pack_tile(0, cb_out)` 将结果写入输出 CB
4. CB 索引: in0=0, in1=1, out=16 (官方推荐)

**参考文件**:
- `matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp`
- `matmul_single_core/kernels/compute/mm.cpp`
- `hw/inc/api/compute/matmul.h`

