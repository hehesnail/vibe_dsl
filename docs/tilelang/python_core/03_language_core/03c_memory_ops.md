# TileLang 内存操作模块

本文档详细分析 TileLang 的内存操作模块，包括内存分配、拷贝操作和填充操作。

## 模块概述

TileLang 的内存操作模块位于 `tilelang/language/` 目录下，包含三个核心文件：

| 文件 | 功能 | 大小 |
|------|------|------|
| `allocate.py` | 内存分配接口 | 13KB |
| `copy_op.py` | 拷贝操作接口 | 9KB |
| `fill_op.py` | 填充操作接口 | 3KB |

这些模块为 TileLang 程序提供了完整的内存管理能力，支持不同内存层级的分配和操作。

---

## 内存分配机制 (allocate.py)

内存分配模块提供了多种内存分配函数，用于在不同内存作用域中分配缓冲区。

### 核心分配函数

#### 1. Shared Memory 分配 - `alloc_shared`

```python
def alloc_shared(shape: ShapeType, dtype: DType, scope="shared.dyn") -> Buffer
```

**功能**：分配用于线程间通信的共享内存缓冲区。

**参数**：
- `shape`: 缓冲区形状元组
- `dtype`: 数据类型（如 'float32', 'int32'）
- `scope`: 内存作用域，默认为 `"shared.dyn"`

**特殊处理**：
- 当 `dtype` 为 `"bool"` 时，自动将作用域切换为 `"shared"`，因为 TileLang 的合并 shared memory pass 目前无法处理 bool 类型（`allocate.py:44-47`）

**源码引用**：`tilelang/language/allocate.py:33-48`

#### 2. Local Memory 分配 - `alloc_local`

```python
def alloc_local(shape: ShapeType, dtype: DType, scope="local") -> Buffer
```

**功能**：分配线程私有的本地内存缓冲区。

**参数**：
- `shape`: 缓冲区形状元组
- `dtype`: 数据类型
- `scope`: 内存作用域，默认为 `"local"`

**源码引用**：`tilelang/language/allocate.py:51-62`

#### 3. Fragment Memory 分配 - `alloc_fragment`

```python
def alloc_fragment(shape: ShapeType, dtype: DType, scope="local.fragment") -> Buffer
```

**功能**：分配用于专用操作（如 Tensor Core 运算）的片段内存缓冲区。

**参数**：
- `shape`: 缓冲区形状元组
- `dtype`: 数据类型
- `scope`: 内存作用域，默认为 `"local.fragment"`

**源码引用**：`tilelang/language/allocate.py:65-76`

#### 4. 变量分配 - `alloc_var`

```python
def alloc_var(dtype: DType, *args, scope: str = "local.var", init: PrimExpr | int | float | None = None) -> Buffer
```

**功能**：分配单元素变量缓冲区，支持初始化值。

**参数**：
- `dtype`: 数据类型
- `*args`: 可选位置参数（可以是初始化值或作用域）
- `scope`: 内存作用域，默认为 `"local.var"`
- `init`: 可选的初始化值

**使用示例**：
```python
a = T.alloc_var('int32', 1)                          # 初始值为 1 的变量
a = T.alloc_var('int32', 'local.var')                # 指定作用域
a = T.alloc_var('int32', 1, 'local.var')             # 初始值和作用域
a = T.alloc_var('int32', init=1)                     # 关键字参数初始化
```

**初始化机制**：
- 如果提供了初始化值，通过 `block_attr` 设置 `"tl.local_var_init"` 属性（`allocate.py:138`）
- 支持整数、浮点数、IntImm、FloatImm 类型的初始化值

**源码引用**：`tilelang/language/allocate.py:87-141`

### 同步原语分配

#### 5. Barrier 分配 - `alloc_barrier`

```python
def alloc_barrier(arrive_count: int | list[int]) -> Buffer
```

**功能**：分配屏障缓冲区，用于线程同步。

**参数**：
- `arrive_count`: 需要到达屏障的线程数，可以是单个整数或整数列表

**实现细节**：
- 使用 `scope="shared.barrier"` 分配缓冲区
- 通过 `block_attr` 设置 `"barrier_init"` 属性进行初始化（`allocate.py:167`）
- 使用 `IntImm` 表达式转换为 C++ pass 可消费的格式

**源码引用**：`tilelang/language/allocate.py:144-169`

#### 6. Cluster Barrier 分配 - `alloc_cluster_barrier`

```python
def alloc_cluster_barrier(arrive_count: int | list[int]) -> Buffer
```

**功能**：分配集群级屏障缓冲区，用于多线程块集群同步。

**作用域**：`shared.cluster_barrier`

**源码引用**：`tilelang/language/allocate.py:172-192`

### 高级内存分配

#### 7. Tensor Memory (TMEM) 分配 - `alloc_tmem`

```python
def alloc_tmem(shape: ShapeType, dtype: DType) -> Buffer
```

**功能**：为第5代 Tensor Core 操作（如 TCGEN5.MMA）分配 TMEM 缓冲区。

**TMEM 特性**：
- Hopper GPU 引入的专用片上内存
- 组织为 512 列 x 128 行（lane）的 2D 数组
- 每个单元 32 位
- 分配单位为列，每列的所有 lane 一起分配

**约束条件**：
- 分配的列数必须是 2 的幂且至少为 32
- 分配是动态的，必须显式释放
- 分配和释放必须由同一个 warp 执行
- 基地址存储在 shared memory 中

**源码引用**：`tilelang/language/allocate.py:195-221`

#### 8. Reducer 分配 - `alloc_reducer`

```python
def alloc_reducer(shape: ShapeType, dtype: DType, op: ReducerOp = "sum", replication=None) -> Buffer
```

**功能**：分配用于归约操作的缓冲区。

**参数**：
- `shape`: 缓冲区形状
- `dtype`: 数据类型
- `op`: 归约操作类型（"sum", "max", "min"）
- `replication`: 复制策略（"all" 或 "none"）

**使用要求**：
- `op="sum"` 要求使用 `reducer[...] += ...`
- `op="max"` 要求使用 `reducer[...] = T.max(reducer[...], ...)`
- `op="min"` 要求使用 `reducer[...] = T.min(reducer[...], ...)`
- 必须先使用 `T.fill` 进行初始化才能开始归约
- 必须使用 `T.finalize_reducer` 获取最终结果

**初始化要求**：
- `sum`: 必须填充 0
- `min`: 使用 `T.max_value` 作为初始值
- `max`: 使用 `T.min_value` 作为初始值

**源码引用**：`tilelang/language/allocate.py:227-260`

### Descriptor 分配

#### 9. WGMMA Descriptor 分配 - `alloc_wgmma_desc`

```python
def alloc_wgmma_desc(dtype: DType = _dtypes.uint64) -> Buffer
```

**功能**：为 WGMMA（Warp Group Matrix Multiply Accumulate）分配描述符缓冲区。

**作用域**：`local.descriptor.wgmma`

**源码引用**：`tilelang/language/allocate.py:285-286`

#### 10. TCGEN5 Descriptor 分配

```python
def alloc_tcgen05_smem_desc(dtype: DType = _dtypes.uint64) -> Buffer
def alloc_tcgen05_instruction_desc(dtype: DType = _dtypes.uint32) -> Buffer
def alloc_tcgen05_instr_desc(dtype: DType = _dtypes.uint32) -> Buffer  # 别名
```

**功能**：为 TCGEN5.MMA 操作分配描述符缓冲区。

**作用域**：
- `local.descriptor.tcgen05_smem`
- `local.descriptor.tcgen05_instr`

**源码引用**：`tilelang/language/allocate.py:289-299`

### 输出张量分配

#### 11. Empty 张量分配 - `empty`

```python
def empty(*shape, dtype: DType = _dtypes.float32) -> Tensor
```

**功能**：创建输出张量描述符，用于 Eager 模式构建器。

**返回值**：`OutTensor` 对象

**源码引用**：`tilelang/language/allocate.py:306-314`

---

## 拷贝操作 (copy_op.py)

拷贝操作模块提供了多种数据拷贝原语，支持不同内存层级之间的数据传输。

### 核心拷贝函数

#### 1. 通用拷贝 - `copy`

```python
def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt
```

**功能**：在内存区域之间拷贝数据，是 TileLang 最主要的拷贝原语。

**参数**：
- `src`: 源内存区域（Buffer/BufferLoad/BufferRegion）
- `dst`: 目标内存区域（Buffer/BufferLoad/BufferRegion）
- `coalesced_width`: 合并内存访问宽度（可选）
- `disable_tma`: 是否禁用 TMA 加速，默认为 False
- `eviction_policy`: 缓存驱逐策略（"evict_normal", "evict_first", "evict_last"）
- `annotations`: 额外的注解字典
- `loop_layout`: 并行循环布局提示（仅用于 SIMT 拷贝）

**范围处理规则**（`copy_op.py:82-98`）：
1. 支持 `Buffer`、`BufferRegion`、`BufferLoad` 作为源或目标
2. 范围推导：`Buffer -> shape`，`BufferRegion -> [r.extent]`，`BufferLoad -> 推断的范围`
3. 正常情况下要求两侧范围相同
4. 优化：如果两侧都是标量 `BufferLoad`，直接降级为 `BufferStore`
5. 语法糖：支持传递缓冲区头地址表示整个缓冲区

**驱逐策略映射**（`copy_op.py:113-114`）：
- `"evict_normal"` -> 0
- `"evict_first"` -> 1
- `"evict_last"` -> 2

**返回值**：TVM intrinsic call，操作类型为 `"tl.tileop.copy"`

**源码引用**：`tilelang/language/copy_op.py:51-120`

#### 2. 异步拷贝 - `async_copy`

```python
def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt
```

**功能**：显式异步 Global -> Shared 拷贝原语，通过 `cp.async` 降级。

**特点**：
- 后端强制执行 `cp.async` 约束
- 生成 `ptx_cp_async(...)` + `ptx_commit_group()`
- **不自动插入等待**，同步是显式的

**返回值**：TVM intrinsic call，操作类型为 `"tl.tileop.async_copy"`

**源码引用**：`tilelang/language/copy_op.py:123-164`

#### 3. 2D 卷积 im2col 变换 - `c2d_im2col`

```python
def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tir.PrimExpr
```

**功能**：执行 2D 卷积的 im2col 变换，将图像数据转换为列格式。

**参数**：
- `img`: 输入图像缓冲区
- `col`: 输出列缓冲区
- `nhw_step`: Batch 和空间维度的步长
- `c_step`: 通道维度的步长
- `kernel`: 卷积核大小
- `stride`: 卷积步长
- `dilation`: 扩张率
- `pad`: 填充大小
- `eviction_policy`: 缓存驱逐策略

**返回值**：TVM intrinsic call，操作类型为 `"tl.tileop.c2d_im2col"`

**源码引用**：`tilelang/language/copy_op.py:167-211`

### 内部辅助函数

#### `_normalize_copy_regions`

```python
def _normalize_copy_regions(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[tir.BufferRegion | tir.BufferLoad | tir.Buffer, ...]
```

**功能**：规范化拷贝操作的源和目标区域。

**处理逻辑**（`copy_op.py:14-48`）：
1. 如果两侧都是 Buffer，验证形状相等
2. 获取两侧的范围（extent）
3. 检测标量 BufferLoad 情况
4. 如果两侧都是标量 BufferLoad，直接返回用于降级为 BufferStore
5. 断言至少一侧有范围信息
6. 对齐和广播范围（从右侧/尾部对齐）
7. 使用 `to_buffer_region` 转换为 BufferRegion

**源码引用**：`tilelang/language/copy_op.py:14-48`

---

## 填充操作 (fill_op.py)

填充操作模块提供了缓冲区初始化和清零功能。

### 核心填充函数

#### 1. 填充指定值 - `fill`

```python
def fill(buffer: BufferLikeType, value: tir.PrimExpr) -> tir.PrimExpr
```

**功能**：用指定值填充缓冲区或缓冲区区域。

**参数**：
- `buffer`: TVM 缓冲区或缓冲区区域
- `value`: 填充值

**范围推导逻辑**（`fill_op.py:20-36`）：
1. 如果 buffer 是带有 let 值的 Var，解析为实际对象
2. `Buffer`: 使用 `buffer.shape` 作为范围
3. `BufferRegion`: 使用 `region` 中每个维度的 `extent`
4. `BufferLoad`: 尝试从 load 获取区域，失败则使用 `[1] * len(indices)`

**返回值**：TVM intrinsic call，操作类型为 `"tl.tileop.fill"`

**源码引用**：`tilelang/language/fill_op.py:10-37`

#### 2. 清零操作 - `clear`

```python
def clear(buffer: BufferLikeType) -> tir.PrimExpr
```

**功能**：将缓冲区清零（填充为 0）。

**参数**：
- `buffer`: TVM 缓冲区或包含缓冲区区域的变量

**处理逻辑**（`fill_op.py:52-63`）：
1. 如果 buffer 是带有 let 值的 Var，获取实际的缓冲区区域
2. 根据区域类型调用 `fill` 函数
3. 直接调用 `fill(buffer, 0)`

**异常**：如果缓冲区变量包含无效的缓冲区区域，抛出 `ValueError`

**源码引用**：`tilelang/language/fill_op.py:40-63`

---

## Shared Memory 管理

### 内存作用域体系

TileLang 支持以下内存作用域：

| 作用域 | 用途 | 分配函数 |
|--------|------|----------|
| `shared.dyn` | 动态共享内存（默认） | `alloc_shared` |
| `shared` | 静态共享内存 | `alloc_shared` (bool 类型) |
| `shared.barrier` | 屏障同步 | `alloc_barrier` |
| `shared.cluster_barrier` | 集群屏障同步 | `alloc_cluster_barrier` |
| `shared.tmem` | Tensor Memory | `alloc_tmem` |
| `local` | 线程本地内存 | `alloc_local` |
| `local.fragment` | 片段内存（Tensor Core） | `alloc_fragment` |
| `local.var` | 单元素变量 | `alloc_var` |
| `local.descriptor.*` | 描述符内存 | `alloc_descriptor` |

### Shared Memory 合并优化

TileLang 的 `merge smem pass` 会尝试合并共享内存分配以优化内存使用：

- **Bool 类型特殊处理**：由于合并 pass 目前不支持 bool 类型，当分配 bool 类型的 shared memory 时，自动切换到 `"shared"` 作用域（`allocate.py:44-47`）

### 内存分配最佳实践

1. **Shared Memory 分配**：
   - 优先使用 `alloc_shared` 进行线程间数据共享
   - 注意 bool 类型的特殊作用域处理

2. **Local Memory 分配**：
   - 使用 `alloc_local` 存储线程私有数据
   - 使用 `alloc_fragment` 进行 Tensor Core 运算

3. **变量初始化**：
   - 使用 `alloc_var` 的 `init` 参数进行显式初始化
   - 注意初始化值的类型匹配

4. **同步原语**：
   - 使用 `alloc_barrier` 进行线程块内同步
   - 使用 `alloc_cluster_barrier` 进行集群级同步

5. **TMEM 使用**：
   - 确保列数是 2 的幂且 >= 32
   - 分配和释放必须在同一个 warp 中
   - 注意 TMEM 的生命周期管理

---

## 代码引用汇总

### allocate.py
- `alloc_shared`: `tilelang/language/allocate.py:33-48`
- `alloc_local`: `tilelang/language/allocate.py:51-62`
- `alloc_fragment`: `tilelang/language/allocate.py:65-76`
- `alloc_var`: `tilelang/language/allocate.py:87-141`
- `alloc_barrier`: `tilelang/language/allocate.py:144-169`
- `alloc_cluster_barrier`: `tilelang/language/allocate.py:172-192`
- `alloc_tmem`: `tilelang/language/allocate.py:195-221`
- `alloc_reducer`: `tilelang/language/allocate.py:227-260`
- `alloc_descriptor`: `tilelang/language/allocate.py:266-282`
- `empty`: `tilelang/language/allocate.py:306-314`

### copy_op.py
- `_normalize_copy_regions`: `tilelang/language/copy_op.py:14-48`
- `copy`: `tilelang/language/copy_op.py:51-120`
- `async_copy`: `tilelang/language/copy_op.py:123-164`
- `c2d_im2col`: `tilelang/language/copy_op.py:167-211`

### fill_op.py
- `fill`: `tilelang/language/fill_op.py:10-37`
- `clear`: `tilelang/language/fill_op.py:40-63`
