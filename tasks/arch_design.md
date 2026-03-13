# TileLang Tenstorrent 后端扩展 - 实施方案（最终版）

## 设计原则

**核心目标**：保持 TileLang DSL 与现有 CUDA/HIP 模式完全一致，所有 Tenstorrent 特定差异下沉到 Lowering Pipeline。

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
# target = "tenstorrent" → Tenstorrent R/C/W kernels
```

---

## 核心映射关系

### 1. 任务切分映射

| TileLang DSL | GPU (HIP/CUDA) | Tenstorrent |
|--------------|----------------|-------------|
| `T.Kernel(grid_x, grid_y, threads=...)` | Grid/Block 硬件调度 | **Core Grid 软件切分** |
| `bx, by` (block indices) | `blockIdx.x/y` | **映射到 Core (X,Y) 坐标** |
| 单核数据量 | 自动分配 | **通过 `split_work_to_cores` 计算** |
| 运行时参数 | 自动传递 | **显式 `SetRuntimeArgs` 设置 work_offset** |

**关键差异**：
- GPU：硬件 GigaThread Engine 自动调度 blocks 到 SMs
- Tenstorrent：软件通过 `split_work_to_cores` 显式切分，每个 Core 运行相同 kernel 但处理不同数据范围

### 2. 内存映射

| TileLang DSL | GPU (HIP/CUDA) | Tenstorrent |
|--------------|----------------|-------------|
| `T.alloc_shared(...)` | `__shared__` memory | **Circular Buffer (CB)** |
| 内存大小 | 用户代码决定 | **自动计算 tile_size × num_pages** |
| 超出处理 | 编译/运行时报错 | **Compile-time 静态验证 < 1.5MB** |
| 复用策略 | `MergeSharedMemoryAllocations` pass | **流式处理 (streaming) + 双缓冲** |
| 数据搬运 | 隐式/异步拷贝 | **显式 NoC (noc_async_read/write)** |
| 同步机制 | `__syncthreads()` | **CB push/pop + barrier** |

---

## 文件结构（修正后）

### C++ Transform Passes（`src/transform/`）

IR 变换层，处理 Tenstorrent 特定的 lowering：

```
src/transform/
├── assign_tenstorrent_cores.cc/h         # [新增] Grid 到 Core (X,Y) 映射
│   # 输入: T.Kernel(grid_x, grid_y)
│   # 输出: 每个 core 的 work_offset, work_per_core
│
├── plan_tenstorrent_cb.cc/h              # [新增] CB 分配规划
│   # 输入: alloc_shared 的 shape/dtype
│   # 输出: CB ID, num_pages, page_size
│   # 验证: 总 CB size < 1.5MB (Blackhole)
│
├── split_tenstorrent_kernel.cc/h         # [新增] 内核拆分
│   # 输入: unified PrimFunc (T.copy, T.gemm, T.copy)
│   # 输出: reader_func, compute_func, writer_func
│   # 操作: 自动插入 CB 同步 (reserve/push/wait/pop)
│
└── lower_tenstorrent_noc.cc/h            # [新增] NoC 地址 lower
    # 输入: DRAM buffer access
    # 输出: get_noc_addr(x, y, addr) 计算
```

### C++ Code Generation（`src/target/`）

代码生成层，将 TIR 转换为 TT-Metal C++ kernel：

```
src/target/
├── codegen_tenstorrent.h                 # [新增] 代码生成器头文件
├── codegen_tenstorrent.cc                # [新增] 代码生成器实现
│   # 继承 CodeGenC
│   # GenerateReaderKernel() → dataflow_api.h 风格
│   # GenerateComputeKernel() → ckernel_api.h 风格
│   # GenerateWriterKernel() → dataflow_api.h 风格
│
├── rt_mod_tenstorrent.cc                 # [新增] 运行时模块
│   # BuildTileLangTenstorrent()
│   # BuildTileLangTenstorrentWithoutCompile()
│   # TVM_FFI_STATIC_INIT_BLOCK 注册
│
└── intrin_rule_tenstorrent.cc            # [可选] intrinsic lowering
```

### Device Templates（`src/tl_templates/tenstorrent/`）

TT-Metal 特定的 device 代码模板：

```
src/tl_templates/tenstorrent/
├── common.h                              # 通用定义、数据类型
├── cb.h                                  # CB 操作封装
│   # cb_reserve_back(), cb_push_back()
│   # cb_wait_front(), cb_pop_front()
│   # get_read_ptr(), get_write_ptr()
│
├── noc.h                                 # NoC 操作封装
│   # noc_async_read(), noc_async_write()
│   # noc_async_read_barrier()
│   # get_noc_addr(x, y, addr)
│
├── dataflow.h                            # Reader/Writer 辅助
├── compute.h                             # Compute kernel 辅助
└── sfpi.h                                # SFPU 指令封装
```

### Python 层（`tilelang/`）

```
tilelang/
├── contrib/
│   └── tt_metal_build.py                 # [新增] TT-Metal 编译器封装
│       # compile_tenstorrent() 调用 tt-metal 编译系统
│
├── engine/
│   └── lower.py                          # [修改] 添加 tenstorrent 分支
│       # elif target.kind.name == "tenstorrent":
│       #   device_mod = AssignTenstorrentCores()(device_mod)
│       #   device_mod = PlanTenstorrentCB()(device_mod)
│       #   device_mod = SplitTenstorrentKernel()(device_mod)
│       #   device_mod = target.build.tilelang_tenstorrent(device_mod, target)
│
├── jit/
│   ├── execution_backend.py              # [修改] 添加 tenstorrent 支持
│   └── adapter/wrapper.py                # [修改] 添加 TLTenstorrentSourceWrapper
│
├── carver/arch/
│   └── tenstorrent.py                    # [新增] TenstorrentArch 类
│       # Blackhole/Wormhole 检测
│       # L1 size, CB count, Core grid 等信息
│
└── utils/
    └── target.py                         # [修改] 添加 "tenstorrent" target
```

---

## Lowering Pipeline 详细流程

### Tenstorrent 特定 Passes

```
TileLang DSL (统一代码)
    ↓
[Phase 1: LowerAndLegalize]  ← 复用现有 passes
    - LowerTileOp (T.copy/gemm → TIR ops)
    ↓
[Tenstorrent-specific passes]  ← [新增]
    │
    ├─ AssignTenstorrentCores
    │  输入: T.Kernel(grid_x, grid_y)
    │  逻辑: split_work_to_cores(grid_x * grid_y, core_grid)
    │  输出: 为每个 core 生成 work_offset, work_per_core
    │  修改: 添加 block_idx → core_x/core_y 映射
    │
    ├─ PlanTenstorrentCB
    │  输入: alloc_shared(shape, dtype), num_stages
    │  逻辑: tile_size = 32*32*sizeof(dtype)
    │        num_pages = num_stages (双缓冲=2)
    │        cb_id = 自动分配 (0, 1, 2...)
    │  验证: sum(cb_size) < 1.5MB
    │  输出: CB 配置存入 function attributes
    │
    └─ SplitTenstorrentKernel
       输入: unified PrimFunc
       分析:
         - T.copy(DRAM→shared) → reader_func
         - T.gemm/T.compute → compute_func
         - T.copy(shared→DRAM) → writer_func
       插入同步:
         - reader: cb_reserve_back() / noc_async_read() / cb_push_back()
         - compute: cb_wait_front() / compute / cb_pop_front() / cb_push_back()
         - writer: cb_wait_front() / noc_async_write() / cb_pop_front()
       输出: 三个独立的 PrimFunc
    ↓
[Phase 2: OptimizeForTarget]  ← 复用部分 passes
    - 适配 ThreadSync 为 CB barrier
    ↓
codegen_tenstorrent.cc
    ├─ GenerateReaderKernel(reader_func, cb_config)
    ├─ GenerateComputeKernel(compute_func, input_cbs, output_cbs)
    └─ GenerateWriterKernel(writer_func, cb_config)
    ↓
Runtime (Host)
    ├─ split_work_to_cores(total_work, core_grid)
    ├─ CreateCircularBuffer(program, all_cores, cb_config)
    ├─ CreateKernel(program, "reader.cpp", all_cores)
    ├─ CreateKernel(program, "compute.cpp", all_cores)
    ├─ CreateKernel(program, "writer.cpp", all_cores)
    ├─ SetRuntimeArgs(program, kernel_id, core, {work_offset, work_per_core, ...})
    └─ EnqueueMeshWorkload(cq, workload)
```

---

## 实现阶段规划

### Phase 1: 基础框架 - 单核支持（3 周）

**目标**：实现单核 (0,0) 的代码生成，验证基础流程

**任务**：
1. [ ] `src/transform/split_tenstorrent_kernel.cc`
   - 实现 unified → R/C/W 拆分
   - 单核：固定 CoreCoord{0, 0}

2. [ ] `src/transform/plan_tenstorrent_cb.cc`
   - 从 alloc_shared 推导 CB 配置
   - 单核：简化验证

3. [ ] `src/target/codegen_tenstorrent.cc`
   - 生成 reader.cpp, compute.cpp, writer.cpp
   - 生成 TT-Metal 风格 kernel_main()

4. [ ] `src/target/rt_mod_tenstorrent.cc`
   - WithoutCompile 版本

5. [ ] `CMakeLists.txt` 更新

**验证**：生成简单 copy 算子的三个 kernel 代码

### Phase 2: 多核任务切分（2 周）

**目标**：实现 `T.Kernel` 到多 Core 的映射

**任务**：
1. [ ] `src/transform/assign_tenstorrent_cores.cc`
   - 实现 block_idx → core(X,Y) 映射
   - 生成 work_offset, work_per_core

2. [ ] `tilelang/carver/arch/tenstorrent.py`
   - BlackholeArch 类
   - core_grid, L1 size 等硬件信息

3. [ ] `tilelang/contrib/tt_metal_build.py`
   - Host 端代码生成
   - split_work_to_cores 调用

**验证**：多核并行 copy 算子

### Phase 3: TT-Metal 集成（3 周）

**目标**：端到端编译和执行

**任务**：
1. [ ] 研究 TT-Metal JIT Build API
2. [ ] 实现 `CompileTenstorrentKernels()`
3. [ ] 完整 Runtime 模块
4. [ ] Python 层集成

**验证**：`target = "tenstorrent"` 端到端运行

### Phase 4: 计算操作（2 周）

**目标**：支持 GEMM

**任务**：
1. [ ] SFPU 指令支持
2. [ ] `T.gemm` → `matmul_tiles` 映射
3. [ ] 双缓冲优化

### Phase 5: 优化（2 周）

**目标**：性能优化

**任务**：
1. [ ] 自动 tile size 选择
2. [ ] L1 内存优化建议
3. [ ] Wormhole 支持

---

## 关键设计确认

### 1. DSL 层 100% 兼容

- 用户代码无需修改即可编译到 Tenstorrent
- `T.Kernel`, `T.alloc_shared`, `T.copy`, `T.gemm` 等行为一致
- 差异完全下沉到 lowering pipeline

### 2. 核心映射

```
T.Kernel(grid_x, grid_y, threads=128) as (bx, by)
         ↓  lowering
Core (X,Y) = MapBlockToCore(bx, by, core_grid)
work_offset = GetWorkOffset(bx, by)
work_per_core = GetWorkPerCore(bx, by)
```

### 3. 内存管理

```
T.alloc_shared((M, N), dtype)
         ↓  lowering
CB ID = AutoAssignCB()
page_size = 32 * 32 * sizeof(dtype)
num_pages = num_stages  // 双缓冲
```

### 4. 同步机制

自动生成，用户不可见：
- Reader: `cb_reserve_back()` → `noc_async_read()` → `cb_push_back()`
- Compute: `cb_wait_front()` → compute → `cb_pop_front()` → `cb_push_back()`
- Writer: `cb_wait_front()` → `noc_async_write()` → `cb_pop_front()`

---

## 下一步行动

1. **开始 Phase 1**：实现 `split_tenstorrent_kernel.cc`
2. **准备环境**：确认 TT-Metal 头文件路径
3. **创建测试**：单核 copy 算子作为首个验证目标
