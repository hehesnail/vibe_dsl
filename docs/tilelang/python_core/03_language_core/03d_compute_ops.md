# TileLang 计算操作模块

本文档详细分析 TileLang 的计算操作模块，包括 GEMM 操作、归约操作、原子操作和数学内联函数。

## 模块概述

计算操作模块位于 `tilelang/language/` 目录下，包含四个核心文件：

| 文件 | 功能 | 大小 |
|------|------|------|
| `gemm_op.py` | 通用矩阵乘法 (GEMM) 操作 | 13KB |
| `reduce_op.py` | 归约操作（sum、max、min 等） | 18KB |
| `atomic.py` | 原子操作（atomic_add、atomic_max 等） | 21KB |
| `math_intrinsics.py` | 数学内联函数（log、exp、IEEE 运算等） | 10KB |

这些模块提供了 TileLang DSL 的核心计算能力，支持从底层原子操作到高层矩阵运算的完整计算原语。

---

## GEMM 操作 (gemm_op.py)

### 核心实现

GEMM 操作的共享实现函数 `_gemm_impl` 位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:22`，是所有 GEMM 变体的基础：

```python
def _gemm_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `A` | `BufferLikeType` | - | 输入矩阵 A（支持 Buffer、BufferLoad、BufferRegion 或 Var） |
| `B` | `BufferLikeType` | - | 输入矩阵 B |
| `C` | `BufferLikeType` | - | 输出矩阵 C |
| `transpose_A` | `bool` | `False` | 是否转置矩阵 A |
| `transpose_B` | `bool` | `False` | 是否转置矩阵 B |
| `policy` | `GemmWarpPolicy` | `Square` | Warp 分区策略 |
| `clear_accum` | `bool` | `False` | 是否清空累加器 |
| `k_pack` | `int` | `1` | 打包矩阵核心数量（仅 ROCm） |
| `wg_wait` | `int` | `0` | 等待的 Warpgroup MMA 批次标识符 |
| `mbar` | `BarrierType | None` | `None` | Blackwell 架构的 Mbarrier |

### 形状验证逻辑

`_gemm_impl` 函数在 `/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:70-87` 执行严格的形状验证：

1. **输出矩阵 C 必须是 2D 张量**
2. **输入矩阵 A/B 必须是 2D 或更高维张量**，且高维部分必须为 1
3. **K 维度一致性检查**：验证 A 的 K 维与 B 的 K 维相等
4. **偏移量验证**：确保 A 和 B 第一维的偏移量为 0

### 公开 API

#### `gemm()` - 主 GEMM 函数

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:196`，根据环境配置自动选择实现版本：

```python
def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr
```

内部通过 `_env.use_gemm_v1()` 决定使用 `gemm_v1` 或 `gemm_v2`。

#### `gemm_v1()` - 标准实现

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:139`，使用操作码 `"tl.tileop.gemm"`。

#### `gemm_v2()` - 快速编译版本

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:168`，使用操作码 `"tl.tileop.gemm_py"`，标记为实验性，用于加速编译。

### 参数合法性处理

`_gemm_impl` 中的 `legalize_arguments` 函数（`/root/dev/vibe_dsl/tilelang/tilelang/language/gemm_op.py:40-51`）处理 let-bound 变量：

```python
def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
    if isinstance(arg, tir.Var) and T.has_let_value(arg):
        return T.get_let_value(arg).buffer
    return arg
```

---

## 归约操作 (reduce_op.py)

### 归约类型定义

归约类型由 `ReduceKind` 字面量定义（`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:21`）：

```python
ReduceKind = Literal["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"]
```

| 类型 | 说明 |
|------|------|
| `sum` | 求和归约 |
| `abssum` | 绝对值求和 |
| `max` | 最大值归约 |
| `absmax` | 绝对值最大值归约 |
| `min` | 最小值归约 |
| `bitand` | 按位与归约 |
| `bitor` | 按位或归约 |
| `bitxor` | 按位异或归约 |

### 核心归约函数

#### `reduce()` - 通用归约

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:25`，是宏实现的归约操作：

```python
def reduce(
    buffer: tir.Buffer,
    out: tir.Buffer,
    reduce_type: ReduceKind,
    dim: int,
    clear: bool
) -> None
```

**缓冲区作用域处理逻辑**（`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:46-110`）：

| 输入作用域 | 输出作用域 | 处理策略 |
|-----------|-----------|---------|
| shared | shared | 分配 fragment 中间缓冲区，先 copy 到 fragment，再归约，最后 copy 回 shared |
| shared | fragment | 分配输入 fragment，copy 后直接在 fragment 上归约 |
| fragment | shared | 分配输出 fragment，归约后 copy 到 shared |
| fragment | fragment | 直接在 fragment 上归约 |

### 便捷归约函数

#### `reduce_max()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:115`

```python
def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True) -> None
```

当 `clear=True` 时，输出缓冲区会先初始化为 `-inf`。

#### `reduce_min()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:136`

```python
def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True) -> None
```

当 `clear=True` 时，输出缓冲区会先初始化为 `inf`。

#### `reduce_sum()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:152`

```python
def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True) -> None
```

**重要说明**（`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:162-168`）：
当 `clear=True` 时，`reduce_sum` 不会直接在输出缓冲区上计算。因为在 warp 归约期间，同一个值会被累加多次（warp 中的线程数）。因此实现步骤为：
1. 创建与输出相同形状和类型的临时缓冲区
2. 将输出复制到临时缓冲区
3. 调用 reduce_sum 处理临时缓冲区和输出
4. 将临时缓冲区加到输出上

#### `reduce_abssum()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:177`

```python
def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1) -> None
```

#### `reduce_absmax()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:192`

```python
def reduce_absmax(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True) -> None
```

#### 位运算归约
- `reduce_bitand()` - 按位与归约 (`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:207`)
- `reduce_bitor()` - 按位或归约 (`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:222`)
- `reduce_bitxor()` - 按位异或归约 (`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:237`)

### 累积和 (Cumsum)

#### `cumsum()` - 主累积和函数

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:292`：

```python
def cumsum(
    src: BufferLikeType,
    dst: BufferLikeType | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> tir.PrimExpr | None
```

**参数说明**：
- `src`: 源缓冲区（支持 Buffer、BufferRegion、BufferLoad）
- `dst`: 目标缓冲区，若为 `None` 则原地操作
- `dim`: 归约维度，支持负索引（Python 风格）
- `reverse`: 是否反向累积

**示例用法**（`/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:305-336`）：

```python
# 1D 包含扫描
T.cumsum(src=A_shared, dst=A_shared, dim=0)

# 2D 前缀和（沿最后一维，反向）
T.cumsum(src=tile, dim=1, reverse=True)

# 对缓冲区区域操作
T.cumsum(InputG_fragment[i * chunk_size:(i + 1) * chunk_size], dim=0)
```

#### `cumsum_fragment()` - Fragment 累积和宏

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:252`，处理 fragment 缓冲区的累积和：

```python
@macro
def cumsum_fragment(
    src: BufferLikeType,
    dst: BufferLikeType,
    dim: int,
    reverse: bool,
) -> None
```

处理流程：
1. 分配 shared memory 缓冲区
2. 将 src copy 到 shared memory
3. 执行累积和操作
4. 将结果 copy 到 dst

### Warp 级归约

#### `warp_reduce_sum()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:395`

```python
def warp_reduce_sum(value: tir.PrimExpr) -> tir.PrimExpr
```

使用 shuffle 操作在 warp 内对所有线程的寄存器值进行求和归约。归约后，warp 中的所有线程都将获得相同的求和结果。

#### `warp_reduce_max()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:411`

```python
def warp_reduce_max(value: tir.PrimExpr) -> tir.PrimExpr
```

#### `warp_reduce_min()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:427`

```python
def warp_reduce_min(value: tir.PrimExpr) -> tir.PrimExpr
```

#### `warp_reduce_bitand()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:443`

```python
def warp_reduce_bitand(value: tir.PrimExpr) -> tir.PrimExpr
```

#### `warp_reduce_bitor()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:459`

```python
def warp_reduce_bitor(value: tir.PrimExpr) -> tir.PrimExpr
```

### Reducer 终结

#### `finalize_reducer()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/reduce_op.py:375`

```python
def finalize_reducer(reducer: tir.Buffer) -> tir.PrimExpr
```

发出 `tl.tileop.finalize_reducer` 内联调用，用于终结给定的 reducer 缓冲区。

---

## 原子操作 (atomic.py)

### 内存序定义

内存序映射表位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:11-18`：

```python
_MEMORY_ORDER_ID_MAP = {
    "relaxed": 0,
    "consume": 1,
    "acquire": 2,
    "release": 3,
    "acq_rel": 4,
    "seq_cst": 5,
}
```

### 原子最大值

#### `atomic_max()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:21`

```python
def atomic_max(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False
) -> PrimExpr
```

**两种执行路径**：

1. **标量路径**（`/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:64-75`）：
   - 当 `dst` 和 `value` 都没有暴露 extents 时使用
   - 使用 `tl.atomic_max_elem_op` 内联函数
   - 支持 `memory_order` 参数
   - 支持 `return_prev` 返回前值

2. **Tile 区域路径**（`/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:77-100`）：
   - 用于 Buffer/BufferRegion/BufferLoad 输入
   - 使用 `tl.tileop.atomicmax` 内联函数
   - 支持张量到张量的原子操作
   - 要求输入输出形状结构相等

### 原子最小值

#### `atomic_min()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:103`

```python
def atomic_min(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False
) -> PrimExpr
```

实现逻辑与 `atomic_max` 类似，使用 `tl.atomic_min_elem_op` 和 `tl.tileop.atomicmin`。

### 原子加法

#### `atomic_add()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:185`

```python
def atomic_add(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False,
    use_tma: bool = False
) -> PrimExpr
```

**额外参数**：
- `use_tma`: 是否使用 TMA (cp.reduce) 执行原子加法，仅 sm90+ 可用

**标量路径**（`/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:234-248`）：
- 使用 `tl.atomic_add_elem_op` 或 `tl.atomic_add_ret_elem_op`（当 `return_prev=True`）

**Tile 区域路径**（`/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:250-278`）：
- 使用 `tl.tileop.atomicadd`
- 支持 `use_tma` 和 `memory_order` 注解

### 向量原子加法

#### `atomic_addx2()` - 双宽度原子加法
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:281`

```python
def atomic_addx2(
    dst: Buffer,
    value: PrimExpr,
    return_prev: bool = False
) -> PrimExpr
```

支持 FP16 对和 BF16 向量化原子加法（CUDA Arch > 750）。

#### `atomic_addx4()` - 四宽度原子加法
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:319`

```python
def atomic_addx4(
    dst: Buffer,
    value: PrimExpr,
    return_prev: bool = False
) -> PrimExpr
```

支持 float4 向量化原子加法（CUDA Arch >= 900）。

### 原子加载与存储

#### `atomic_load()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:357`

```python
def atomic_load(src: Buffer, memory_order: str = "seq_cst") -> PrimExpr
```

使用指定内存序从缓冲区原子加载值，返回加载的 `PrimExpr`。

使用 `tl.atomic_load_elem_op` 内联函数。

#### `atomic_store()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/atomic.py:401`

```python
def atomic_store(
    dst: Buffer,
    src: PrimExpr,
    memory_order: str = "seq_cst"
) -> PrimExpr
```

使用指定内存序将值原子存储到缓冲区。

使用 `tl.atomic_store_elem_op` 内联函数。

---

## 数学内联函数 (math_intrinsics.py)

### 快速数学函数

所有快速数学函数都使用 `tl.__<op_name>` 内联操作码。

#### 对数函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `__log(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:15` | 自然对数 log(x) |
| `__log2(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:32` | 以 2 为底的对数 |
| `__log10(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:49` | 以 10 为底的对数 |

#### 三角函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `__tan(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:66` | 正切函数 |
| `__cos(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:83` | 余弦函数 |
| `__sin(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:100` | 正弦函数 |

#### 指数函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `__exp10(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:117` | 10 的 x 次方 |
| `__exp(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:134` | 2 的 x 次方 |

### IEEE 合规运算

所有 IEEE 运算支持四种舍入模式：
- `rn`: 向最近舍入（默认）
- `rz`: 向零舍入
- `ru`: 向正无穷舍入
- `rd`: 向负无穷舍入

舍入模式验证函数 `_validate_rounding_mode` 位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:7`。

#### 基本运算

| 函数 | 位置 | 说明 |
|------|------|------|
| `ieee_add(x, y, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:152` | IEEE 合规加法 |
| `ieee_sub(x, y, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:178` | IEEE 合规减法 |
| `ieee_mul(x, y, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:202` | IEEE 合规乘法 |

#### 融合乘加

#### `ieee_fmaf()`
位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:226`

```python
def ieee_fmaf(
    x: PrimExpr,
    y: PrimExpr,
    z: PrimExpr,
    rounding_mode: str = "rn"
) -> PrimExpr
```

计算 `x * y + z`，使用 IEEE 合规的融合乘加运算。

#### 倒数与平方根

| 函数 | 位置 | 说明 |
|------|------|------|
| `ieee_frcp(x, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:253` | 倒数 1/x |
| `ieee_fsqrt(x, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:274` | 平方根 sqrt(x) |
| `ieee_frsqrt(x)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:295` | 倒数平方根 1/sqrt(x)，仅支持向最近舍入 |
| `ieee_fdiv(x, y, rounding_mode="rn")` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:312` | 除法 x/y |

### 打包 FP32x2 运算

这些函数操作 `float32x2` 数据类型，在支持的 NVIDIA 架构上可降级为 PTX 指令。

#### 参数验证

`_validate_f32x2_math_args` 函数（`/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:336`）验证所有参数：
- 必须是 `PrimExpr` 类型
- 必须是 `float32x2` 数据类型

#### 运算函数

| 函数 | 位置 | PTX 指令 | 说明 |
|------|------|---------|------|
| `fadd2(x, y)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:344` | `add.rn.f32x2` | 打包加法 |
| `fmul2(x, y)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:368` | `mul.rn.f32x2` | 打包乘法 |
| `fma2(x, y, z)` | `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:380` | `fma.rn.f32x2` | 打包融合乘加 |

#### 向后兼容别名

位于 `/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:393-403`：
- `fadd_f32x2(x, y)` -> `fadd2(x, y)`
- `fmul_f32x2(x, y)` -> `fmul2(x, y)`
- `fma_f32x2(x, y, z)` -> `fma2(x, y, z)`

### 导出列表

`__all__` 定义了模块的公开接口（`/root/dev/vibe_dsl/tilelang/tilelang/language/math_intrinsics.py:406-426`）：

```python
__all__ = [
    "__log", "__log2", "__log10",
    "__tan", "__cos", "__sin",
    "__exp10", "__exp",
    "ieee_add", "ieee_sub", "ieee_mul", "ieee_fmaf",
    "ieee_frcp", "ieee_fsqrt", "ieee_frsqrt", "ieee_fdiv",
    "fadd2", "fmul2", "fma2",
]
```

---

## 架构与设计模式

### 1. 双重路径模式

原子操作和 GEMM 操作都实现了双重路径：
- **标量/元素路径**：用于单个元素的细粒度操作
- **Tile 区域路径**：用于张量块的粗粒度操作

### 2. 宏实现模式

归约操作使用 `@macro` 装饰器实现，允许在编译时展开复杂的缓冲区管理逻辑。

### 3. 内联函数封装

所有计算操作最终都封装为 TVM TIR 内联函数调用 (`tir.call_intrin`)，通过操作码字符串标识具体操作。

### 4. 类型安全设计

- 使用 Python 类型注解（`BufferLikeType`、`PrimExpr`、`ReduceKind` 等）
- 运行时类型验证（如 `_validate_f32x2_math_args`）
- 形状和维度验证（如 `_legalize_dim`）

### 5. 内存序支持

原子操作支持 C++11 风格的内存序语义，通过 `_MEMORY_ORDER_ID_MAP` 映射到内部标识符。

---

## 使用示例

### GEMM 示例

```python
import tilelang.language as T

@T.prim_func
def gemm_kernel(A: T.Buffer((1024, 1024), "float32"),
                B: T.Buffer((1024, 1024), "float32"),
                C: T.Buffer((1024, 1024), "float32")):
    # 分配共享内存
    A_shared = T.alloc_shared((128, 128), "float32")
    B_shared = T.alloc_shared((128, 128), "float32")
    C_fragment = T.alloc_fragment((128, 128), "float32")

    # 拷贝数据
    T.copy(A[0:128, 0:128], A_shared)
    T.copy(B[0:128, 0:128], B_shared)

    # 执行 GEMM
    T.gemm(A_shared, B_shared, C_fragment, transpose_B=True)

    # 拷贝结果
    T.copy(C_fragment, C[0:128, 0:128])
```

### 归约示例

```python
import tilelang.language as T

@T.prim_func
def reduce_kernel(Input: T.Buffer((128, 64), "float32"),
                  Output: T.Buffer((128,), "float32")):
    # 分配缓冲区
    Input_shared = T.alloc_shared((128, 64), "float32")
    Output_shared = T.alloc_shared((128,), "float32")

    # 拷贝输入
    T.copy(Input, Input_shared)

    # 沿维度 1 求和归约
    T.reduce_sum(Input_shared, Output_shared, dim=1, clear=True)

    # 拷贝输出
    T.copy(Output_shared, Output)
```

### 原子操作示例

```python
import tilelang.language as T

@T.prim_func
def atomic_kernel(Input: T.Buffer((1024,), "float32"),
                  Output: T.Buffer((1,), "float32")):
    # 并行求和
    for i in T.thread_binding(1024, "threadIdx.x"):
        T.atomic_add(Output, Input[i])
```

### 数学函数示例

```python
import tilelang.language as T
from tilelang.language.math_intrinsics import __exp, __log, ieee_fmaf

@T.prim_func
def math_kernel(Input: T.Buffer((256,), "float32"),
                Output: T.Buffer((256,), "float32")):
    for i in T.thread_binding(256, "threadIdx.x"):
        # 快速指数
        exp_val = __exp(Input[i])

        # 快速对数
        log_val = __log(Input[i])

        # IEEE 融合乘加
        result = ieee_fmaf(exp_val, log_val, 1.0, rounding_mode="rn")
        Output[i] = result
```
