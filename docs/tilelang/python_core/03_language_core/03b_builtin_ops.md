# TileLang 内置操作模块 (builtin.py)

## 模块概述

`tilelang/language/builtin.py` 是 TileLang 语言的核心模块，提供了对底层 GPU 硬件特性的直接访问。该模块封装了 CUDA/PTX 指令、线程管理、同步原语、Tensor Core 操作等底层功能，使开发者能够编写高性能的 GPU 内核。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/builtin.py`

**主要功能分类**:
1. 线程/块索引操作
2. 内存屏障与同步原语
3. Tensor Core 操作 (WGMMA/MMA)
4. 全局内存访问操作
5. 寄存器管理
6. 辅助工具函数

---

## 线程/块索引操作

### Warp 和 Lane 索引

#### `get_lane_idx(warp_size=None)`

获取当前线程在其 warp 中的 lane 索引。

```python
def get_lane_idx(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr
```

**参数**:
- `warp_size`: 逻辑 warp 大小，NVIDIA 默认为 32，AMD 默认为 64

**返回值**: 当前线程的 lane 索引 (int32)

**示例**:
```python
lane = T.get_lane_idx()          # 使用默认 warp 大小
lane = T.get_lane_idx(64)        # 显式指定 warp 大小为 64
```

**实现**: 调用 `tl::get_lane_idx(warp_size)`，通过线性线程 ID 计算 lane 索引。

**源码**: `tilelang/language/builtin.py:485-509`

---

#### `get_warp_idx_sync(warp_size=None)`

获取当前 warp 的索引（假设 warp 内线程已收敛）。

```python
def get_warp_idx_sync(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr
```

**参数**:
- `warp_size`: 逻辑 warp 大小

**返回值**: 当前 warp 的索引 (int32)

**实现**: 调用 `tl::get_warp_idx_sync(warp_size)`，将块线性线程 ID 除以 warp 大小。

**源码**: `tilelang/language/builtin.py:512-535`

---

#### `get_warp_idx(warp_size=None)`

获取当前 warp 的索引（无需 warp 收敛）。

```python
def get_warp_idx(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr
```

**源码**: `tilelang/language/builtin.py:538-561`

---

#### `get_warp_group_idx(warp_size=None, warps_per_group=None)`

获取当前 warp group 的索引。

```python
def get_warp_group_idx(
    warp_size: int | PrimExpr | None = None,
    warps_per_group: int | PrimExpr | None = None,
) -> PrimExpr
```

**参数**:
- `warp_size`: 逻辑 warp 大小
- `warps_per_group`: 每个 warp group 的 warp 数量，NVIDIA 默认为 4

**示例**:
```python
group = T.get_warp_group_idx()
group = T.get_warp_group_idx(32, 6)  # 将 6 个 warp 视为一个 group
```

**源码**: `tilelang/language/builtin.py:564-597`

---

#### `shuffle_elect(thread_extent)`

在逻辑线程组中选举一个 leader lane。

```python
def shuffle_elect(thread_extent: int) -> PrimExpr
```

**参数**:
- `thread_extent`: 线程组的大小（线程数）。传入 0 表示在整个线程块中选举一个 lane

**返回值**: 布尔值，表示当前线程是否为 elected leader

**示例**:
```python
is_leader = T.shuffle_elect(64)
T.if_then_else(is_leader, do_leader_work(), T.evaluate(0))
```

**实现**: 调用 `tl::tl_shuffle_elect<thread_extent>()`，使用 `cutlass::canonical_warp_idx_sync()` 和 `cute::elect_one_sync()` 或 `__shfl_sync` 实现。

**源码**: `tilelang/language/builtin.py:600-621`

---

## 同步原语

### 线程同步

#### `sync_threads(barrier_id=None, arrive_count=None)`

同步块内所有线程。

```python
def sync_threads(barrier_id: int = None, arrive_count: int = None)
```

**参数**:
- `barrier_id`: 屏障 ID（可选）
- `arrive_count`: 到达计数（可选）

**实现**: 调用 `tir.tvm_storage_sync` intrinsic。

**源码**: `tilelang/language/builtin.py:867-874`

---

#### `sync_warp(mask=None)`

同步 warp 内所有线程。

```python
def sync_warp(mask: int = None)
```

**参数**:
- `mask`: 线程掩码（可选）

**源码**: `tilelang/language/builtin.py:877-881`

---

#### `sync_global()`

同步整个网格的所有线程。

```python
def sync_global()
```

**实现**: 使用线程绑定和块范围信息调用 `tir.tvm_storage_sync`。

**源码**: `tilelang/language/builtin.py:891-897`

---

#### `sync_grid()`

同步网格内所有线程。

```python
def sync_grid()
```

**源码**: `tilelang/language/builtin.py:900-902`

---

### Shuffle 操作

#### `shfl_xor(value, offset)`

执行 XOR 偏移的 shuffle 操作。

```python
def shfl_xor(
    value: int | PrimExpr | tir.Call,
    offset: int | PrimExpr | tir.Call
)
```

**参数**:
- `value`: 要 shuffle 的值
- `offset`: shuffle 偏移量

**实现**:
- AMD: 调用 `__shfl_xor`
- NVIDIA: 调用 `__shfl_xor_sync`

**源码**: `tilelang/language/builtin.py:824-838`

---

#### `shfl_down(value, offset)`

执行向下偏移的 shuffle 操作。

```python
def shfl_down(
    value: int | PrimExpr | tir.Call,
    offset: int | PrimExpr | tir.Call
)
```

**源码**: `tilelang/language/builtin.py:841-851`

---

#### `shfl_up(value, offset)`

执行向上偏移的 shuffle 操作。

```python
def shfl_up(
    value: int | PrimExpr | tir.Call,
    offset: int | PrimExpr | tir.Call
)
```

**源码**: `tilelang/language/builtin.py:854-864`

---

#### `shfl_sync(mask, value, srcLane, width=None)`

从同一 warp 中的指定线程接收数据。

```python
def shfl_sync(
    mask: int,
    value: int | PrimExpr,
    srcLane: int,
    width: int = None
)
```

**源码**: `tilelang/language/builtin.py:884-888`

---

### 内存屏障 (MBarrier)

#### `mbarrier_wait_parity(mbarrier, parity)`

等待内存屏障的奇偶条件。

```python
def mbarrier_wait_parity(
    mbarrier: BarrierType,
    parity: int | Var
)
```

**参数**:
- `mbarrier`: 内存屏障对象
- `parity`: 要等待的奇偶值（0 或 1）

**示例**:
```python
mbar = T.alloc_barrier(1)
T.mbarrier_wait_parity(mbar, 0)

# 在流水线内核中的常见用法
mbars = T.alloc_barrier([128] * n)
for ko in range(num_stages):
    # 生产者等待消费者完成前一次迭代
    T.mbarrier_wait_parity(mbars[1], ko ^ 1)
    T.copy(A_global, A_shared)
    T.mbarrier_arrive(mbars[0])

    # 消费者等待生产者数据
    T.mbarrier_wait_parity(mbars[0], ko)
    T.gemm(A_shared, B_shared, C_local)
    T.mbarrier_arrive(mbars[1])
```

**源码**: `tilelang/language/builtin.py:379-417`

---

#### `mbarrier_arrive(mbarrier, cta_id=None)`

到达内存屏障。

```python
def mbarrier_arrive(
    mbarrier: BarrierType,
    cta_id: int | Var | None = None
)
```

**参数**:
- `mbarrier`: 内存屏障对象
- `cta_id`: 集群中的对等 CTA 排名（仅对集群屏障有效）

**源码**: `tilelang/language/builtin.py:420-435`

---

#### `mbarrier_expect_tx(mbarrier, tx)`

设置内存屏障的期望事务计数。

```python
def mbarrier_expect_tx(
    mbarrier: BarrierType,
    tx: int
)
```

**参数**:
- `mbarrier`: 内存屏障对象
- `tx`: 期望的事务计数

**源码**: `tilelang/language/builtin.py:438-451`

---

#### `barrier_wait(mbarrier, parity)` / `barrier_arrive(mbarrier)`

`mbarrier_wait_parity` 和 `mbarrier_arrive` 的语法糖。

**源码**: `tilelang/language/builtin.py:799-821`

---

#### `cp_async_barrier_noinc(barrier)`

执行 PTX 异步复制屏障（`cp.async.mbarrier.arrive.noinc`）。

```python
def cp_async_barrier_noinc(barrier: BarrierType)
```

**源码**: `tilelang/language/builtin.py:996-999`

---

#### `ptx_arrive_cluster_barrier(mbarrier, cta_id)`

在集群中的共享屏障到达。

```python
def ptx_arrive_cluster_barrier(
    mbarrier: BarrierType,
    cta_id: int | Var
)
```

**源码**: `tilelang/language/builtin.py:367-376`

---

## Tensor Core 操作

### WGMMA (Warp Group Matrix Multiply-Accumulate)

WGMMA 是 NVIDIA Hopper 架构引入的 warp group 级矩阵乘累加指令。

#### `warpgroup_arrive()`

信号表示 warp group 已准备好执行后续的 WGMMA 操作。

```python
def warpgroup_arrive()
```

**源码**: `tilelang/language/builtin.py:454-460`

---

#### `warpgroup_commit_batch()`

提交当前的 WGMMA 批次。

```python
def warpgroup_commit_batch()
```

**源码**: `tilelang/language/builtin.py:463-469`

---

#### `warpgroup_wait(num_mma)`

等待指定的 WGMMA 批次完成。

```python
def warpgroup_wait(num_mma: int)
```

**参数**:
- `num_mma`: 要等待的 WGMMA 批次标识符

**源码**: `tilelang/language/builtin.py:472-482`

---

#### `wait_wgmma(id)`

等待 WGMMA 操作完成。

```python
def wait_wgmma(id: int)
```

**源码**: `tilelang/language/builtin.py:786-796`

---

#### `warpgroup_fence_operand(buffer_or_ptr, offset=0, num_regs=None, dtype=None)`

在目标累加器寄存器上插入 warp group 屏障。

```python
def warpgroup_fence_operand(
    buffer_or_ptr: BufferLikeType | PrimExpr,
    offset: int | PrimExpr = 0,
    num_regs: int | PrimExpr | None = None,
    dtype: DType | None = None,
)
```

**参数**:
- `buffer_or_ptr`: 表示累加器片段的缓冲区，或指针表达式
- `offset`: 从累加器片段起始位置的元素偏移
- `num_regs`: 要屏障的 32 位寄存器数量
- `dtype`: 累加器元素的数据类型字符串

**用途**: 防止 NVCC 将累加器片段的使用下沉到对应的 WGMMA 操作之后。

**源码**: `tilelang/language/builtin.py:624-783`

---

### SM70 MMA (Volta Tensor Core)

#### `ptx_mma_sm70(...)`

SM70 (Volta) 架构的 PTX Tensor Core MMA 指令。

```python
def ptx_mma_sm70(
    shape: str,
    A_layout: str,
    B_layout: str,
    A_dtype: str,
    B_dtype: str,
    C_dtype: str,
    multiplicand_a: Var,
    a_index: Expr,
    multiplicand_b: Var,
    b_index: Expr,
    accumulator: Var,
    c_index: Expr,
)
```

**参数**:
- `shape`: MMA 片段形状（如 `"m16n16k4"`）
- `A_layout`/`B_layout`: 乘数片段 A/B 的布局（"row" 或 "col"）
- `A_dtype`/`B_dtype`/`C_dtype`: 数据类型（如 "fp16"、"fp32"）
- `multiplicand_a`/`multiplicand_b`/`accumulator`: 变量
- `a_index`/`b_index`/`c_index`: 索引

**支持**: m16n16k4 形状，FP16 输入，FP16/FP32 累加。

**源码**: `tilelang/language/builtin.py:1015-1111`

---

### TCGEN05 (Blackwell)

#### `tcgen05_mma_arrive(mbar)`

信号 UMMA (TCGEN05) 屏障到达。

```python
def tcgen05_mma_arrive(mbar: tir.Buffer | BufferLoad | PrimExpr)
```

**参数**:
- `mbar`: 共享内存中的 mbarrier 对象或其地址

**源码**: `tilelang/language/builtin.py:1002-1012`

---

## 内存访问操作

### 只读数据缓存加载 (LDG)

#### `__ldg(load_or_buf, index=None)`

通过 CUDA 只读数据缓存显式加载。

```python
def __ldg(
    load_or_buf: BufferLoad | tir.Buffer,
    index: PrimExpr | int | None = None
) -> PrimExpr
```

**参数**:
- `load_or_buf`: `BufferLoad`（如 `x[i]`）或 `Buffer`
- `index`: 传递 `Buffer` 时的可选索引

**返回值**: 加载的值

**示例**:
```python
val = T.__ldg(x[i])      # 使用 BufferLoad
val = T.__ldg(buf, 0)    # 使用 Buffer 和索引
```

**源码**: `tilelang/language/builtin.py:51-77`

---

### 显式 PTX 全局内存加载

#### `ldg32(src, pred=None)`

从全局内存加载 32 位（4 字节）。

```python
def ldg32(
    src: BufferLikeType,
    pred: PrimExpr = None
) -> PrimExpr
```

**参数**:
- `src`: `Buffer`、`BufferRegion` 或 `BufferLoad`
- `pred`: 可选的谓词条件，为 False 时跳过加载

**返回值**: 加载的 32 位值 (uint32)

**示例**:
```python
val = T.ldg32(x[i])
val = T.ldg32(x[i:i+2])       # 加载 2 x fp16
val = T.ldg32(x[i], pred=i < N)  # 谓词加载
```

**源码**: `tilelang/language/builtin.py:1114-1137`

---

#### `ldg64(src, pred=None)`

从全局内存加载 64 位（8 字节）。

```python
def ldg64(
    src: BufferLikeType,
    pred: PrimExpr = None
) -> PrimExpr
```

**返回值**: 加载的 64 位值 (uint32x2)

**源码**: `tilelang/language/builtin.py:1140-1163`

---

#### `ldg128(src, pred=None)`

从全局内存加载 128 位（16 字节）。

```python
def ldg128(
    src: BufferLikeType,
    pred: PrimExpr = None
) -> PrimExpr
```

**返回值**: 加载的 128 位值 (uint32x4)

**源码**: `tilelang/language/builtin.py:1166-1189`

---

#### `ldg256(src, pred=None)`

从全局内存加载 256 位（32 字节）。

```python
def ldg256(
    src: BufferLikeType,
    pred: PrimExpr = None
) -> PrimExpr
```

**返回值**: 加载的 256 位值 (uint32x8)

**源码**: `tilelang/language/builtin.py:1192-1215`

---

### 显式 PTX 全局内存存储

#### `stg32(dst, value, pred=None)`

向全局内存存储 32 位（4 字节）。

```python
def stg32(
    dst: BufferLikeType,
    value: PrimExpr,
    pred: PrimExpr = None
)
```

**参数**:
- `dst`: 目标位置的 `Buffer`、`BufferRegion` 或 `BufferLoad`
- `value`: 要存储的 32 位值
- `pred`: 可选的谓词条件

**源码**: `tilelang/language/builtin.py:1218-1238`

---

#### `stg64(dst, value, pred=None)`

向全局内存存储 64 位（8 字节）。

```python
def stg64(
    dst: BufferLikeType,
    value: PrimExpr,
    pred: PrimExpr = None
)
```

**源码**: `tilelang/language/builtin.py:1241-1261`

---

#### `stg128(dst, value, pred=None)`

向全局内存存储 128 位（16 字节）。

```python
def stg128(
    dst: BufferLikeType,
    value: PrimExpr,
    pred: PrimExpr = None
)
```

**源码**: `tilelang/language/builtin.py:1264-1284`

---

#### `stg256(dst, value, pred=None)`

向全局内存存储 256 位（32 字节）。

```python
def stg256(
    dst: BufferLikeType,
    value: PrimExpr,
    pred: PrimExpr = None
)
```

**源码**: `tilelang/language/builtin.py:1287-1307`

---

## 寄存器管理

### 最大寄存器数设置

#### `set_max_nreg(reg_count, is_inc)`

设置要使用的最大寄存器数量。

```python
def set_max_nreg(reg_count: int, is_inc: int)
```

**参数**:
- `reg_count`: 要分配的寄存器数量
- `is_inc`: 0 表示递减，1 表示递增

**参考**: [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)

**源码**: `tilelang/language/builtin.py:319-334`

---

#### `inc_max_nreg(reg_count)` / `dec_max_nreg(reg_count)`

递增/递减最大寄存器数量。

```python
def inc_max_nreg(reg_count: int)
def dec_max_nreg(reg_count: int)
```

**源码**: `tilelang/language/builtin.py:337-344`

---

#### `annotate_producer_reg_dealloc(reg_count=24)`

标注生产者寄存器释放。

```python
def annotate_producer_reg_dealloc(reg_count: int = 24)
```

**源码**: `tilelang/language/builtin.py:347-349`

---

#### `annotate_consumer_reg_alloc(reg_count=240)`

标注消费者寄存器分配。

```python
def annotate_consumer_reg_alloc(reg_count: int = 240)
```

**源码**: `tilelang/language/builtin.py:352-354`

---

#### `no_set_max_nreg()` / `disable_warp_group_reg_alloc()`

禁用最大寄存器限制设置。

```python
def no_set_max_nreg()
def disable_warp_group_reg_alloc()
```

**源码**: `tilelang/language/builtin.py:357-364`

---

## 内存描述符操作

### TMA (Tensor Memory Access)

#### `create_tma_descriptor(*args)`

创建 TMA 描述符。

```python
def create_tma_descriptor(*args)
```

**源码**: `tilelang/language/builtin.py:255-264`

---

#### `tma_load(*args)`

执行 TMA 加载操作。

```python
def tma_load(*args)
```

**源码**: `tilelang/language/builtin.py:271-280`

---

#### `tma_store_arrive(*args)` / `tma_store_wait(*args)`

TMA 存储到达/等待操作。

```python
def tma_store_arrive(*args)
def tma_store_wait(*args)
```

**源码**: `tilelang/language/builtin.py:295-316`

---

#### `fence_proxy_async(*args)`

创建异步代理操作的屏障。

```python
def fence_proxy_async(*args)
```

**源码**: `tilelang/language/builtin.py:283-292`

---

### WGMMA/TCGEN05 描述符

#### `initialize_wgmma_descriptor(...)`

初始化 WGMMA/UTCMMA 共享内存描述符。

```python
def initialize_wgmma_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    layout_type_: int = 0,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
) -> PrimExpr
```

**参数**:
- `descriptor`: 描述符缓冲区（必须是 1D 大小为 1 的缓冲区）
- `start_address`: 起始地址
- `layout_type_`: 布局类型
- `leading_byte_offset`: 前导字节偏移
- `stride_byte_offset`: 步幅字节偏移

**源码**: `tilelang/language/builtin.py:905-932`

---

#### `initialize_tcgen05_descriptor(...)`

初始化 TCGEN05 共享内存描述符。

```python
def initialize_tcgen05_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    leading_is_absolute: bool = False,
    swizzle_mode: int = 0,
) -> PrimExpr
```

**参数**:
- `descriptor`: 描述符缓冲区
- `start_address`: 起始地址
- `leading_byte_offset`: 前导字节偏移
- `stride_byte_offset`: 步幅字节偏移
- `base_offset`: 基础偏移
- `leading_is_absolute`: 前导偏移是否为绝对值
- `swizzle_mode`: 交换模式

**源码**: `tilelang/language/builtin.py:935-966`

---

#### `increase_descriptor_offset(descriptor, offset)`

增加内存描述符的偏移量。

```python
def increase_descriptor_offset(
    descriptor: PrimExpr,
    offset: PrimExpr
) -> PrimExpr
```

**源码**: `tilelang/language/builtin.py:969-988`

---

## 辅助工具函数

#### `access_ptr(base, access_type="r", *extents, offset=0, extent=None, ignore_last_ndim=0)`

从类缓冲区基位置创建 TileLang `tl.access_ptr`。

```python
def access_ptr(
    base: BufferLikeType,
    access_type: str | int = "r",
    *extents: PrimExpr | int | tuple[PrimExpr | int, ...] | list[PrimExpr | int],
    offset: PrimExpr | int = 0,
    extent: PrimExpr | int | None = None,
    ignore_last_ndim: int = 0,
) -> PrimExpr
```

**参数**:
- `base`: 基位置（`BufferLoad`、`BufferRegion`、`Buffer` 或绑定的 `Var`）
- `access_type`: 访问掩码（"r"=读, "w"=写, "rw"=读写）
- `*extents`: 每轴范围
- `offset`: 基位置的额外元素偏移
- `extent`: 显式的 1D 范围覆盖
- `ignore_last_ndim`: 忽略最后 N 维

**返回值**: handle 类型的 `tir.Call`

**示例**:
```python
T.access_ptr(A[i], "r")           # 范围默认为 1
T.access_ptr(A[i], "r", 16)       # 范围=16
T.access_ptr(A[i, j], "r", m, n)  # 范围=m*n
```

**源码**: `tilelang/language/builtin.py:80-252`

---

#### `loop_break()`

跳出最内层循环。

```python
def loop_break()
```

**源码**: `tilelang/language/builtin.py:991-993`

---

## 内部辅助函数

### `_normalize_index_arg(value)`

规范化 warp 大小参数，统一接受 Python int 和 PrimExpr 值。

```python
def _normalize_index_arg(value: int | PrimExpr | None) -> PrimExpr | None
```

**源码**: `tilelang/language/builtin.py:18-29`

---

### `_mbar_to_buffer_load(mbar)`

将内存屏障转换为缓冲区加载。

```python
def _mbar_to_buffer_load(mbar: BarrierType) -> BufferLoad
```

**源码**: `tilelang/language/builtin.py:32-48`

---

## 架构支持说明

| 功能 | NVIDIA (CUDA) | AMD (HIP) | 说明 |
|------|---------------|-----------|------|
| `shfl_xor` | `__shfl_xor_sync` | `__shfl_xor` | 自动检测平台 |
| `shfl_down` | `__shfl_down_sync` | `__shfl_down` | 自动检测平台 |
| `shfl_up` | `__shfl_up_sync` | `__shfl_up` | 自动检测平台 |
| WGMMA | Hopper+ | 不支持 | 需要 SM90+ |
| TCGEN05 | Blackwell | 不支持 | 需要 SM100+ |
| MMA SM70 | Volta+ | 不支持 | Tensor Core |

模块通过 `_IS_HIP_AVAILABLE` 标志自动检测 AMD HIP 支持，并相应地调整底层调用。

---

## 总结

`builtin.py` 模块是 TileLang 与底层 GPU 硬件交互的关键桥梁，提供了：

1. **细粒度的线程控制**: Warp/lane 索引、shuffle 操作
2. **灵活的同步机制**: 从 warp 级到网格级的多级同步
3. **先进的 Tensor Core 支持**: WGMMA、TCGEN05、传统 MMA
4. **高效的内存访问**: 显式 PTX 加载/存储指令
5. **寄存器管理**: 动态调整寄存器分配
6. **内存描述符**: TMA 和 WGMMA 描述符管理

这些底层原语使 TileLang 能够生成高度优化的 GPU 代码，同时保持 Python 级别的编程便利性。
