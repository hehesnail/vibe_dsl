# TileLang Tenstorrent 后端扩展 - 实施方案（Blackhole 专用版）

## 设计原则

**核心目标**：保持 TileLang DSL 与现有 CUDA/HIP 模式完全一致，所有 Tenstorrent 特定差异下沉到 Lowering Pipeline。

**架构限定**：本设计专注于 **Blackhole** 架构（14x10 Tensix cores, 1.5MB L1, 64 CBs）

```python
# 同一套 DSL 代码，编译到不同硬件
@T.prim_func
def gemm_kernel(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_K, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by*block_M, k*block_K], A_shared)
            T.copy(B[k*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])

# target = "cuda"       → CUDA kernel
# target = "blackhole"  → Blackhole R/C/W kernels (14x10 grid, 140 cores)
```

---

## 核心映射关系

### 1. Blackhole 硬件规格

| 参数 | Blackhole 值 | 说明 |
|------|-------------|------|
| **L1 SRAM** | 1.5 MB (1,572,864 bytes) | 每 Tensix Core |
| **Tensix Grid** | 14 x 10 = 140 cores | 可计算核心 |
| **Physical Grid** | 17 x 12 = 204 cores | 包含以太网/DMA 核心 |
| **CB 数量** | 64 (CB 0-63) | 比 Wormhole 的 32 多一倍 |
| **DRAM** | 32 GB (8 banks x 4 GB) | GDDR6 |
| **NOC** | 2 | Network on Chip |

### 2. 任务切分映射

| TileLang DSL | GPU (HIP/CUDA) | Blackhole |
|--------------|----------------|-----------|
| `T.Kernel(grid_x, grid_y, threads=...)` | Grid/Block 硬件调度 | **14x10 Core Grid 软件切分** |
| `bx, by` (block indices) | `blockIdx.x/y` | **映射到 Core (X,Y) ∈ [0,13]x[0,9]** |
| 单核数据量 | 自动分配 | **通过 `split_work_to_cores` 计算** |
| 运行时参数 | 自动传递 | **显式 `SetRuntimeArgs` 设置 work_offset** |

**关键差异**：
- GPU：硬件 GigaThread Engine 自动调度 blocks 到 SMs
- Blackhole：软件通过 `split_work_to_cores` 显式切分到 140 cores，每个 Core 运行相同 kernel 但处理不同数据范围

### 3. 内存映射

| TileLang DSL | GPU (HIP/CUDA) | Blackhole |
|--------------|----------------|-----------|
| `T.alloc_shared(...)` | `__shared__` memory | **Circular Buffer (CB)** |
| 内存大小 | 用户代码决定 | **自动计算 tile_size × num_pages** |
| 超出处理 | 编译/运行时报错 | **Compile-time 静态验证 < 1.5MB** |
| 复用策略 | `MergeSharedMemoryAllocations` pass | **流式处理 (streaming) + 双缓冲** |
| 数据搬运 | 隐式/异步拷贝 | **显式 NoC (noc_async_read/write)** |
| 同步机制 | `__syncthreads()` | **CB push/pop + barrier** |
| **最大 CB 数** | N/A | **64 (Blackhole) vs 32 (Wormhole)** |

---

## 文件结构（修正后）

### C++ Transform Passes（`src/transform/`）

IR 变换层，处理 Blackhole 特定的 lowering：

```
src/transform/
├── assign_blackhole_cores.cc/h           # [新增] Grid 到 Core (X,Y) 映射
│   # 输入: T.Kernel(grid_x, grid_y)
│   # 输出: 每个 core 的 work_offset, work_per_core
│   # Blackhole: 14x10 grid
│
├── plan_blackhole_cb.cc/h                # [新增] CB 分配规划
│   # 输入: alloc_shared 的 shape/dtype
│   # 输出: CB ID (0-63), num_pages, page_size
│   # 验证: 总 CB size < 1.5MB, CB 数 <= 64
│
├── split_blackhole_kernel.cc/h           # [新增] 内核拆分
│   # 输入: unified PrimFunc (T.copy, T.gemm, T.copy)
│   # 输出: reader_func, compute_func, writer_func
│   # 操作: 自动插入 CB 同步 (reserve/push/wait/pop)
│
└── lower_blackhole_noc.cc/h              # [新增] NoC 地址 lower
    # 输入: DRAM buffer access
    # 输出: get_noc_addr(x, y, addr) 计算
```

### C++ Code Generation（`src/target/`）

代码生成层，将 TIR 转换为 TT-Metal C++ kernel：

```
src/target/
├── codegen_blackhole.h                   # [新增] 代码生成器头文件
├── codegen_blackhole.cc                  # [新增] 代码生成器实现
│   # 继承 CodeGenC
│   # GenerateReaderKernel() → dataflow_api.h 风格 (BRISC)
│   # GenerateComputeKernel() → ckernel_api.h 风格 (TRISC)
│   # GenerateWriterKernel() → dataflow_api.h 风格 (NCRISC)
│
├── rt_mod_blackhole.cc                   # [新增] 运行时模块
│   # BuildTileLangBlackhole()
│   # BuildTileLangBlackholeWithoutCompile()
│   # 链接 libtt_metal.so
│   # TVM_FFI_STATIC_INIT_BLOCK 注册
│
└── intrin_rule_blackhole.cc              # [可选] intrinsic lowering
```

### Device Templates（`src/tl_templates/blackhole/`）

TT-Metal Blackhole 特定的 device 代码模板：

```
src/tl_templates/blackhole/
├── common.h                              # 通用定义、数据类型
│   # Blackhole: 64 CBs, 1.5MB L1
├── cb.h                                  # CB 操作封装
│   # cb_reserve_back(), cb_push_back()
│   # cb_wait_front(), cb_pop_front()
│   # get_read_ptr(), get_write_ptr()
│   # Blackhole 支持 64 CBs (vs Wormhole 32)
│
├── noc.h                                 # NoC 操作封装
│   # noc_async_read(), noc_async_write()
│   # noc_async_read_barrier()
│   # get_noc_addr(x, y, addr)
│
├── dataflow.h                            # Reader/Writer 辅助
├── compute.h                             # Compute kernel 辅助
│   # matmul_tiles, add_tiles, etc.
└── sfpi.h                                # SFPU 指令封装
    # Blackhole 增强指令: float_to_int16
```

### Python 层（`tilelang/`）

```
tilelang/
├── contrib/
│   └── tt_metal_build.py                 # [新增] TT-Metal 编译器封装
│       # 仅用于 Host 端代码生成（如需要）
│
├── engine/
│   └── lower.py                          # [修改] 添加 blackhole 分支
│       # elif target.kind.name == "blackhole":
│       #   device_mod = AssignBlackholeCores()(device_mod)
│       #   device_mod = PlanBlackholeCB()(device_mod)
│       #   device_mod = SplitBlackholeKernel()(device_mod)
│       #   device_mod = target.build.tilelang_blackhole(device_mod, target)
│
├── jit/
│   ├── execution_backend.py              # [修改] 添加 blackhole 支持
│   └── adapter/wrapper.py                # [修改] 添加 TLBlackholeSourceWrapper
│
├── carver/arch/
│   └── blackhole.py                      # [新增] BlackholeArch 类
│       # L1 size: 1.5MB
│       # CB count: 64
│       # Core grid: 14x10
│       # DRAM: 32GB
│
└── utils/
    └── target.py                         # [修改] 添加 "blackhole" target
```

### 测试文件（`tests/blackhole/`）

```
tests/
├── transform/
│   ├── test_split_blackhole_kernel.cc    # Split pass 单元测试
│   ├── test_plan_blackhole_cb.cc         # CB 规划测试
│   └── test_assign_blackhole_cores.cc    # Core 分配测试
│
├── target/
│   ├── test_codegen_blackhole.cc         # CodeGen 测试
│   └── test_rt_mod_blackhole.cc          # Runtime 测试
│
└── blackhole/                            # 端到端集成测试
    ├── test_blackhole_copy.py            # Copy 算子测试
    └── test_blackhole_gemm.py            # GEMM 算子测试
```

---

## Lowering Pipeline 详细流程

### Blackhole 特定 Passes

```
TileLang DSL (统一代码)
    ↓
[Phase 0: TileLang 环境准备]
    - 编译测试 tilelang_repo 基础功能
    - 验证 CUDA/HIP 后端正常（如有 GPU）
    ↓
[Phase 1: TT-Metal 环境准备]
    - 编译 tt_metal_repo → libtt_metal.so
    - 编译 tt-sim → libttsim_bh.so
    - 配置 TileLang + Blackhole CMake
    ↓
[Phase 2: LowerAndLegalize]  ← 复用现有 passes
    - LowerTileOp (T.copy/gemm → TIR ops)
    - LegalizeLoop, LegalizeVectorizedLoop
    ↓
[Blackhole-specific passes]  ← 新增
    │
    ├─ AssignBlackholeCores
    │  输入: T.Kernel(grid_x, grid_y)
    │  逻辑: grid_x * grid_y work items 分配到 14x10 core grid
    │  输出: 每个 core 的 CoreCoord{x,y}, work_offset, work_per_core
    │
    ├─ PlanBlackholeCB
    │  输入: alloc_shared(shape, dtype), num_stages
    │  逻辑: page_size = 32*32*sizeof(dtype), num_pages = num_stages
    │        cb_id = 0,1,2... (max 63 for Blackhole)
    │  验证: sum(cb_size) < 1.5MB, num_cbs <= 64
    │  输出: CB 配置存入 function attributes
    │
    └─ SplitBlackholeKernel
       输入: unified PrimFunc
       分析: 识别 T.copy(DRAM→CB) / T.compute / T.copy(CB→DRAM)
       插入 CB 同步:
         - reader: cb_reserve_back() / noc_async_read() / cb_push_back()
         - compute: cb_wait_front() / compute / cb_pop_front() / cb_reserve_back() / cb_push_back()
         - writer: cb_wait_front() / noc_async_write() / cb_pop_front()
       输出: reader_func, compute_func, writer_func (三个 PrimFunc)
       Blackhole: NCRISC 无 16KB IRAM 限制 (vs Wormhole)
    ↓
[Phase 3: OptimizeForTarget]  ← 复用部分 passes
    - VectorizeLoop
    - StorageRewrite
    ↓
codegen_blackhole.cc
    ├─ GenerateReaderKernel(reader_func, cb_config)   → BRISC kernel
    ├─ GenerateComputeKernel(compute_func, input_cbs, output_cbs) → TRISC kernel
    └─ GenerateWriterKernel(writer_func, cb_config)   → NCRISC kernel
    ↓
rt_mod_blackhole.cc
    ├─ 链接 libtt_metal.so
    ├─ LazyInitialize(): CreateDevice(), CreateProgram()
    ├─ CreateCircularBuffer(program, all_cores, cb_config)
    ├─ CreateKernelFromString(program, reader_code, all_cores, BRISC)
    ├─ CreateKernelFromString(program, compute_code, all_cores, TRISC)
    ├─ CreateKernelFromString(program, writer_code, all_cores, NCRISC)
    ├─ SetRuntimeArgs(program, kernel_id, core, {work_offset, work_per_core, ...})
    └─ EnqueueProgram(cq, program, blocking), Finish(cq)
```

---

## 实现阶段规划

### Phase 0: 环境准备（1 周）

**目标**：准备 TileLang、TT-Metal、TT-Sim 编译环境

**任务**：
1. [ ] **TileLang 环境准备**
   - 编译 tilelang_repo 基础功能
   - 运行基础测试验证环境正常

2. [ ] **TT-Metal 编译**
   ```bash
   cd tt_metal_repo
   ./build_metal.sh --build-shared-libs --enable-blackhole
   ```

3. [ ] **TT-Sim 编译**
   ```bash
   git clone https://github.com/tenstorrent/tt-sim.git
   cd tt-sim && mkdir build && cd build
   cmake .. -DARCH=blackhole && make
   ```

4. [ ] **TileLang + Blackhole 配置**
   - CMake 配置 TT-Metal 路径
   - 验证编译链接 `libtt_metal.so`

**验证标准**：
- [ ] `test_tilelang_common` 通过
- [ ] `libtt_metal.so` 编译成功
- [ ] `libttsim_bh.so` 编译成功

---

### Phase 1: 基础框架 - 单核 Copy（3 周）

**目标**：实现单核 (0,0) 的代码生成，在 TT-Sim 上验证

**任务**：
1. [ ] `src/target/codegen_blackhole.cc`
   - 实现 `GenerateReaderKernel()`
   - 生成 TT-Metal C++ 代码（dataflow_api.h 风格）

2. [ ] `src/target/rt_mod_blackhole.cc`
   - 实现 `BuildTileLangBlackhole()`
   - 链接 `libtt_metal.so`
   - 延迟编译（LazyInitialize）

3. [ ] `CMakeLists.txt` 更新
   - 添加 Blackhole 后端编译选项

**测试**：
- [ ] `test_codegen_blackhole.cc`：验证生成代码语法正确
- [ ] `test_rt_mod_blackhole.cc`：验证 Runtime Module 创建
- [ ] `test_blackhole_simple_copy.py`：端到端 Copy 在 TT-Sim 通过

**验证标准**：
- 生成代码包含 `kernel_main()`, `noc_async_read()`, `cb_push_back()`
- TT-Sim 运行结果与 CPU 参考一致

---

### Phase 2: R/C/W 拆分与多核（4 周）

**目标**：实现 Kernel 拆分和 140 核并行

**任务**：
1. [ ] `src/transform/split_blackhole_kernel.cc`
   - unified → reader/compute/writer 拆分
   - 插入 CB 同步操作

2. [ ] `src/transform/plan_blackhole_cb.cc`
   - CB 分配（0-63，最多 64 个）
   - 验证总大小 < 1.5MB

3. [ ] `src/transform/assign_blackhole_cores.cc`
   - 14x10 grid 映射
   - work_offset 计算

4. [ ] `tilelang/carver/arch/blackhole.py`
   - BlackholeArch 类
   - L1=1.5MB, CB=64, Grid=14x10

**测试**：
- [ ] `test_split_blackhole_kernel.cc`：拆分逻辑正确
- [ ] `test_plan_blackhole_cb.cc`：CB 分配正确
- [ ] `test_assign_blackhole_cores.cc`：Core 分配正确
- [ ] `test_blackhole_multicore_copy.py`：140 核并行正确

**验证标准**：
- 140 cores 并行 Copy 结果正确
- CB 同步无死锁

---

### Phase 3: GEMM 支持（3 周）

**目标**：支持矩阵乘法

**任务**：
1. [ ] `T.gemm` → `matmul_tiles` 映射
   - Compute kernel 生成

2. [ ] SFPU 指令支持
   - `add_tiles`, `mul_tiles` 等

3. [ ] 双缓冲优化
   - `num_stages=2` 支持

**测试**：
- [ ] `test_blackhole_simple_gemm.py`：GEMM 结果与 numpy 一致

**验证标准**：
- 32x32x32 GEMM 误差 < 1%

---

### Phase 4: 优化与完善（2 周）

**目标**：性能优化和工程完善

**任务**：
1. [ ] 自动 tile size 选择
2. [ ] L1 内存优化建议
3. [ ] 编译错误处理完善
4. [ ] CI/CD 配置

**测试**：
- [ ] 回归测试：所有之前测试仍通过
- [ ] 性能测试：与理论峰值对比

---

## 测试策略

### 测试分层

```
┌─────────────────────────────────────────────────────────┐
│  集成测试 (pytest)                                       │
│  tests/blackhole/test_blackhole_*.py                    │
│  - 端到端测试，在 TT-Sim 上运行                          │
├─────────────────────────────────────────────────────────┤
│  Runtime 测试 (gtest)                                    │
│  tests/target/test_rt_mod_blackhole.cc                  │
│  - Runtime Module 功能测试                               │
├─────────────────────────────────────────────────────────┤
│  CodeGen 测试 (gtest)                                    │
│  tests/target/test_codegen_blackhole.cc                 │
│  - 生成代码语法检查                                      │
├─────────────────────────────────────────────────────────┤
│  Pass 单元测试 (gtest)                                   │
│  tests/transform/test_split_blackhole_kernel.cc         │
│  tests/transform/test_plan_blackhole_cb.cc              │
│  tests/transform/test_assign_blackhole_cores.cc         │
│  - Transform pass 逻辑验证                               │
└─────────────────────────────────────────────────────────┘
```

### 每阶段测试检查清单

| 阶段 | 关键测试 | 通过标准 |
|------|----------|----------|
| Phase 0 | `test_tilelang_common` | TileLang 基础功能正常 |
| Phase 1 | `test_codegen_blackhole`, `test_blackhole_simple_copy` | 单核 Copy 在 TT-Sim 通过 |
| Phase 2 | `test_split_*`, `test_plan_*`, `test_assign_*`, `test_multicore_copy` | 140 核并行正确 |
| Phase 3 | `test_blackhole_simple_gemm` | GEMM 结果正确 |
| Phase 4 | 回归测试、性能测试 | 优化不破坏功能 |

---

## CI/CD 集成

```yaml
# .github/workflows/blackhole_test.yaml
name: Blackhole Tests

on: [push, pull_request]

env:
  TT_METAL_HOME: /opt/tt-metal
  TT_SIM_HOME: /opt/tt-sim
  TT_METAL_SIMULATOR: /opt/tt-sim/build/libttsim_bh.so
  TT_METAL_SLOW_DISPATCH_MODE: 1

jobs:
  # Phase 0: 环境准备测试
  setup:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Cache TT-Metal Build
        uses: actions/cache@v3
        with:
          path: /opt/tt-metal
          key: tt-metal-${{ hashFiles('tt_metal_repo/**') }}

      - name: Build TT-Metal
        if: steps.cache.outputs.cache-hit != 'true'
        run: |
          cd tt_metal_repo
          ./build_metal.sh --build-shared-libs --enable-blackhole

      - name: Download TT-Sim
        run: |
          wget -q https://github.com/tenstorrent/tt-sim/releases/latest/download/libttsim_bh.so
          mkdir -p /opt/tt-sim/build
          mv libttsim_bh.so /opt/tt-sim/build/

  # Phase 1-3: 单元测试和集成测试
  test:
    needs: setup
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Build TileLang
        run: |
          mkdir build && cd build
          cmake .. \
            -DTT_METAL_HOME=/opt/tt-metal \
            -DTT_SIM_HOME=/opt/tt-sim \
            -DTILELANG_ENABLE_BLACKHOLE=ON \
            -DTILELANG_USE_TT_SIM=ON
          make -j$(nproc)

      # Pass 单元测试
      - name: Test Transform Passes
        run: |
          ./build/tests/test_split_blackhole_kernel
          ./build/tests/test_plan_blackhole_cb
          ./build/tests/test_assign_blackhole_cores

      # CodeGen 测试
      - name: Test CodeGen
        run: |
          ./build/tests/test_codegen_blackhole

      # Runtime 测试
      - name: Test Runtime
        run: |
          ./build/tests/test_rt_mod_blackhole

      # 端到端集成测试
      - name: Test Integration
        env:
          TT_METAL_SIMULATOR: /opt/tt-sim/build/libttsim_bh.so
          TT_METAL_SLOW_DISPATCH_MODE: 1
        run: |
          pytest tests/blackhole/ -v --tb=short

  # 性能测试（可选）
  benchmark:
    needs: test
    runs-on: [self-hosted, blackhole]  # 真实硬件
    if: github.event_name == 'pull_request'
    steps:
      - name: Run Benchmarks
        run: |
          pytest tests/blackhole/benchmark/ -v
```

---

## 关键设计决策总结

### 为什么只支持 Blackhole？

1. **CB 数量差异**：Blackhole 64 vs Wormhole 32，影响 CB 分配策略
2. **NCRISC IRAM**：Wormhole 16KB 限制使 Writer kernel 复杂化
3. **资源集中**：先做好 Blackhole，后续再抽象支持其他架构

### 为什么用动态链接库而非 Python 包？

1. **性能**：C++ 直接调用避免 Python 开销
2. **一致性**：与 CUDA/HIP 后端架构一致
3. **调试**：更容易跟踪 TT-Metal 内部错误

### 为什么延迟编译？

1. **缓存复用**：TT-Metal 的 `JitBuildCache` 自动管理
2. **快速迭代**：CodeGen 错误不阻塞编译
3. **与 CUDA 模式一致**：Lazy kernel loading

---

## 附录：参考文件路径

### TileLang 参考
```
tilelang_repo/src/target/codegen_cuda.h         # CodeGen 模式参考
tilelang_repo/src/target/rt_mod_cuda.cc         # Runtime 模式参考
tilelang_repo/tilelang/engine/phase.py          # Lowering pipeline
tilelang_repo/tilelang/carver/arch/cuda.py      # Arch 检测模式
```

### TT-Metal 参考
```
tt_metal_repo/tt_metal/api/tt-metalium/         # Public API
├── host_api.hpp                                # CreateDevice, CreateProgram
├── kernel.hpp                                  # CreateKernelFromString
└── circular_buffer.hpp                         # CreateCircularBuffer

tt_metal_repo/tt_metal/core_descriptors/
└── blackhole_140_arch.yaml                     # 140 cores 配置

tt_metal_repo/tt_metal/third_party/umd/device/api/umd/device/arch/
└── blackhole_implementation.hpp                # 硬件参数定义
```

### 硬件规格来源
```
tt_metal_repo/tt_metal/hw/inc/internal/tt-1xx/blackhole/
├── tensix_types.h                              # 数据格式定义
├── noc/noc_parameters.h                        # NOC 参数
└── dev_mem_map.h                               # 内存映射
```
