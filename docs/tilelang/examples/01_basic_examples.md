# TileLang 基础示例详解

本文档详细分析 TileLang 项目中的 GEMM (General Matrix Multiply) 示例代码，帮助理解 TileLang DSL 的核心概念和编程模式。

## 目录

1. [GEMM 基础示例](#gemm-基础示例)
2. [GEMM 调度与优化](#gemm-调度与优化)
3. [GEMM 持久化内核](#gemm-持久化内核)
4. [GEMM 自动调优](#gemm-自动调优)
5. [FP8 GEMM 示例](#fp8-gemm-示例)
6. [性能优化技巧总结](#性能优化技巧总结)

---

## GEMM 基础示例

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/gemm/example_gemm.py`

### 示例概述

这是最基本的 GEMM 实现，展示了 TileLang 的核心编程模式：
- 使用 `@tilelang.jit` 装饰器进行 JIT 编译
- 使用 `T.prim_func` 定义原始函数
- 使用 `T.Kernel` 定义 CUDA 线程块网格
- 使用 `T.alloc_shared` 和 `T.alloc_fragment` 分配不同层级的内存

### 关键代码详解

#### 1. 内核定义与装饰器

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
```

**关键点：**
- `@tilelang.jit(out_idx=[-1])`: JIT 编译装饰器，`out_idx=[-1]` 表示最后一个参数是输出
- `@T.prim_func`: 标记这是一个原始函数，会被编译为 CUDA 内核
- 类型注解 `T.Tensor((M, K), dtype)` 定义张量形状和数据类型

#### 2. 线程块网格配置

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
```

**关键点：**
- `T.ceildiv(N, block_N)`: 计算 X 维度的线程块数量
- `T.ceildiv(M, block_M)`: 计算 Y 维度的线程块数量
- `threads=128`: 每个线程块包含 128 个线程
- `(bx, by)`: 线程块索引

#### 3. 内存分配

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
```

**关键点：**
- `T.alloc_shared`: 分配共享内存（Shared Memory），用于存储从全局内存加载的数据块
- `T.alloc_fragment`: 分配寄存器片段（Register Fragment），用于存储局部计算结果
- 数据类型区分：`dtype` 用于输入，`accum_dtype` 用于累加（通常精度更高）

#### 4. 流水线计算

```python
T.clear(C_local)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

**关键点：**
- `T.clear(C_local)`: 清零累加器
- `T.Pipelined`: 启用流水线执行，`num_stages=3` 表示三级流水线
- `T.copy`: 从全局内存复制到共享内存
- `T.gemm`: 执行矩阵乘累加操作

#### 5. 结果写回

```python
T.copy(C_local, C[by * block_M, bx * block_N])
```

---

## GEMM 调度与优化

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/gemm/example_gemm_schedule.py`

### 示例概述

此示例展示了更高级的调度技术，包括：
- 使用 `T.use_swizzle` 启用光栅化以优化 L2 缓存局部性
- 使用 `T.Parallel` 进行并行数据拷贝

### 关键代码详解

#### 1. 启用 Swizzle（光栅化）

```python
# Enable rasterization for better L2 Cache Locality
T.use_swizzle(panel_size=10)
```

**作用：** `T.use_swizzle` 启用基于 swizzle 的光栅化，改善 L2 缓存局部性，减少缓存抖动。

#### 2. 并行拷贝

```python
# Instead of using
# T.copy(B[k * block_K, bx * block_N], B_shared)
# we can also use Parallel to auto map the thread
# bindings and vectorize the copy operation.
for k, j in T.Parallel(block_K, block_N):
    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
```

**关键点：**
- `T.Parallel`: 并行循环，自动映射线程绑定并可能向量化拷贝操作
- 相比 `T.copy`，`T.Parallel` 提供了更细粒度的控制

---

## GEMM 持久化内核

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/gemm/example_gemm_persistent.py`

### 示例概述

持久化内核（Persistent Kernel）是一种优化技术，通过让 SM（Streaming Multiprocessor）持续工作来减少调度开销。本示例对比了非持久化和持久化两种实现方式。

### 关键代码详解

#### 1. 非持久化内核

```python
with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
```

传统方式：每个线程块处理一个输出 tile，处理完成后退出。

#### 2. 手动持久化内核

```python
sm_num = driver.get_num_sms()
m_blocks = T.ceildiv(M, block_M)
n_blocks = T.ceildiv(N, block_N)
waves = T.ceildiv(m_blocks * n_blocks, sm_num)
group_size = 8

with T.Kernel(sm_num, threads=threads) as (block_id):
    for w in T.serial(waves):
        tile_id = sm_num * w + block_id
        bx = (tile_id // group_size) % m_blocks
        by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
```

**关键点：**
- `driver.get_num_sms()`: 获取 GPU 的 SM 数量
- `waves`: 计算需要多少轮才能处理完所有 tile
- 手动计算每个线程块应该处理的 tile 坐标

#### 3. 使用 Persistent 原语

```python
for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
```

**关键点：**
- `T.Persistent`: TileLang 提供的高级原语，自动处理持久化调度
- 参数：网格形状、SM 数量、当前块 ID

---

## GEMM 自动调优

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/gemm/example_gemm_autotune.py`

### 示例概述

此示例展示了如何使用 TileLang 的自动调优功能来搜索最优的 GEMM 配置。

### 关键代码详解

#### 1. 配置空间定义

```python
def get_configs(M, N, K, with_roller=False, topk=20):
    if with_roller:
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        carve_template = MatmulTemplate(
            M=M, N=N, K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)
        roller_hints = carve_template.recommend_hints(topk=topk)
        # ... 从 hints 生成配置
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        # ... 笛卡尔积生成配置
```

**关键点：**
- `with_roller=True`: 使用 BitBLAS Roller 生成设备感知的配置建议
- `MatmulTemplate`: 矩阵乘法模板
- 手动配置：通过笛卡尔积遍历所有可能的配置组合

#### 2. 自动调优执行

```python
autotuner = (
    AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K, with_roller))
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        backend=profile_backend,
    )
)
return autotuner.run(warmup=3, rep=20)
```

**关键点：**
- `AutoTuner.from_kernel`: 从内核函数创建调优器
- `set_compile_args`: 设置编译参数
- `set_profile_args`: 设置性能分析参数
- `autotuner.run`: 执行搜索，返回最优配置

#### 3. 启发式配置

```python
def get_heuristic_config() -> dict:
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version in {80}:  # Ampere
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 2, ...}
    elif sm_version in {90}:  # Hopper
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3, ...}
```

---

## FP8 GEMM 示例

### 文件位置
- `/root/dev/vibe_dsl/tilelang/examples/gemm_fp8/example_tilelang_gemm_fp8.py`
- `/root/dev/vibe_dsl/tilelang/examples/gemm_fp8/example_tilelang_gemm_fp8_intrinsic.py`
- `/root/dev/vibe_dsl/tilelang/examples/gemm_fp8/example_tilelang_gemm_fp8_2xAcc.py`

### 示例概述

FP8 (8-bit Floating Point) GEMM 是低精度计算的重要应用场景，特别适用于深度学习推理。TileLang 支持 E4M3 和 E5M2 两种 FP8 格式。

### 关键代码详解

#### 1. 基础 FP8 GEMM

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # ...
```

**关键点：**
- 支持的数据类型：`T.float8_e4m3fn`, `T.float8_e4m3fnuz`, `T.float8_e5m2`, `T.float8_e5m2fnuz`
- 累加仍使用高精度（`T.float32`）以保证数值稳定性

#### 2. 2x 累加优化

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    # for fp8 gemm, do one promote after 4 wgmma inst, i.e. block_K = 128.
    # if block_K < 128, promote after 128/block_K iters.
    # if block_K > 128, promote after every iter.
    update_interval = 128 // block_K if block_K < 128 else 1

    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
    C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

    for k in T.Pipelined(K_iters, num_stages=3):
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        # Promote to enable 2xAcc
        if (k + 1) % update_interval == 0:
            for i, j in T.Parallel(block_M, block_N):
                C_local_accum[i, j] += C_local[i, j]
            T.clear(C_local)
```

**关键点：**
- `update_interval`: 控制累加更新的间隔，用于启用 2x 累加优化
- 定期将 `C_local` 的结果累加到 `C_local_accum` 并清零，避免数值溢出

#### 3. 使用 Tensor Core Intrinsics

```python
mma_emitter = TensorCoreIntrinEmitter(
    a_dtype=in_dtype,
    b_dtype=in_dtype,
    accum_dtype=accum_dtype,
    a_transposed=False,
    b_transposed=True,
    block_row_warps=block_row_warps,
    block_col_warps=block_col_warps,
    warp_row_tiles=warp_row_tiles,
    warp_col_tiles=warp_col_tiles,
    chunk=chunk,
)

# 在循环中使用
for ki in T.serial(0, (block_K // micro_size_k)):
    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
    mma_emitter.mma(A_local, B_local, C_local)
```

**关键点：**
- `TensorCoreIntrinEmitter`: 自动生成 Tensor Core 指令（如 `ldmatrix`, `mma`, `stmatrix`）
- 支持 AMD GPU 的 `MatrixCoreIntrinEmitter`（使用 `mfma` 指令）

---

## 性能优化技巧总结

### 1. 内存层级优化

| 技术 | API | 作用 |
|------|-----|------|
| 共享内存分配 | `T.alloc_shared` | 减少全局内存访问延迟 |
| 寄存器片段 | `T.alloc_fragment` | 存储局部累加结果 |
| 本地内存 | `T.alloc_local` | 细粒度 warp 级数据存储 |

### 2. 数据移动优化

| 技术 | API | 作用 |
|------|-----|------|
| 流水线加载 | `T.Pipelined` | 重叠数据拷贝与计算 |
| 并行拷贝 | `T.Parallel` | 向量化数据加载 |
| 直接拷贝 | `T.copy` | 全局内存到共享内存 |

### 3. 计算优化

| 技术 | API | 作用 |
|------|-----|------|
| Swizzle/Rasterization | `T.use_swizzle` | 改善 L2 缓存局部性 |
| 布局注解 | `T.annotate_layout` | 自定义内存布局避免 bank 冲突 |
| GEMM 策略 | `T.GemmWarpPolicy` | 控制 warp 级并行策略 |

### 4. 高级优化

| 技术 | 适用场景 |
|------|----------|
| 持久化内核 | 大规模矩阵，减少调度开销 |
| 自动调优 | 搜索最优 tile 大小和流水线深度 |
| Tensor Core Intrinsics | 需要极致性能，手动控制 MMA 指令 |
| FP8 低精度 | 推理场景，提高吞吐降低显存 |

### 5. 代码示例索引

| 文件 | 主要特性 |
|------|----------|
| `example_gemm.py:5` | 基础 GEMM |
| `example_gemm_schedule.py:19` | Swizzle 优化 |
| `example_gemm_persistent.py:35` | 持久化内核 |
| `example_gemm_autotune.py:22` | 自动调优 |
| `example_gemm_intrinsics.py:26` | Tensor Core 指令 |
| `example_tilelang_gemm_fp8.py:14` | FP8 基础 |
| `example_tilelang_gemm_fp8_2xAcc.py:8` | 2x 累加优化 |
| `example_tilelang_gemm_fp8_intrinsic.py:30` | FP8 + Intrinsics |
