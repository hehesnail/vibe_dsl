# TileLang 控制流模块

本文档详细分析 TileLang 的控制流模块，包括循环管理、条件执行、分支与跳转等核心功能。

## 模块概述

TileLang 的控制流模块主要分布在以下文件中：

- `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py` - 循环管理（Parallel, Pipelined, Serial, Unroll, Vectorized 等）
- `/root/dev/vibe_dsl/tilelang/tilelang/language/logical.py` - 逻辑操作（any_of, all_of）
- `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py` - 条件分支（If, Then, Else, where）
- `/root/dev/vibe_dsl/tilelang/tilelang/language/customize.py` - 循环控制（loop_break）

## 循环管理详解

### 1. Parallel - 并行循环

`Parallel` 用于构建嵌套并行 for 循环，适用于创建元素级张量表达式。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:13`

```python
def Parallel(
    *extents: int | tir.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    prefer_async: bool | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `extents` | `int \| tir.PrimExpr` | 迭代范围 |
| `coalesced_width` | `int \| None` | 并行循环的合并宽度 |
| `loop_layout` | `Fragment \| None` | 并行循环嵌套的布局注解，作为 `"parallel_loop_layout"` 附加到最外层循环 |
| `prefer_async` | `bool \| None` | PTX 异步复制重写提示，设置为 `True` 时在此并行循环子树中请求 cp.async 注入 |
| `annotations` | `dict[str, Any] \| None` | 用户提供的循环注解 |

**布局约束**:

- 每个并行循环在布局推断后必须由布局注解覆盖
- 对于嵌套并行循环，注解必须位于最外层循环；内层并行循环不得携带布局注解
- 对于深度为 `k` 的嵌套，布局必须满足 `InputDim == k`
- 违规（最外层循环缺少注解、内层循环有注解或 `InputDim` 不匹配）会导致编译错误

**使用示例**:

```python
import tilelang.language as T

# 简单的并行循环
with T.Parallel(128) as i:
    C[i] = A[i] + B[i]

# 嵌套并行循环
with T.Parallel(16, 16) as (i, j):
    C[i, j] = A[i, j] + B[i, j]

# 带布局注解的并行循环
with T.Parallel(16, 16, loop_layout=layout) as (i, j):
    C[i, j] = A[i, j] + B[i, j]
```

### 2. Pipelined - 流水线循环

`Pipelined` 用于构建流水线化的 for 循环，支持软件流水线优化。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:112`

```python
def Pipelined(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    num_stages: int = 0,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    sync: list[list[int]] | None = None,
    group: list[list[int]] | None = None,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | `tir.PrimExpr` | 迭代最小值 |
| `stop` | `tir.PrimExpr \| None` | 迭代最大值 |
| `num_stages` | `int` | 流水线生产者和消费者之间使用的最大缓冲数，为 0 时禁用流水线 |
| `order` | `list[int] \| None` | 语句执行顺序 |
| `stage` | `list[int] \| None` | 语句的阶段分配 |
| `sync` | `list[list[int]] \| None` | 同步点配置 |
| `group` | `list[list[int]] \| None` | 分组配置 |

**使用示例**:

```python
import tilelang.language as T

# 基本的流水线循环
for ko in T.Pipelined(0, K // block_K, num_stages=3):
    # 阶段 0: 加载 A 和 B
    T.copy(A[...], A_shared[...])
    T.copy(B[...], B_shared[...])
    # 阶段 1: 执行矩阵乘法
    T.gemm(A_shared, B_shared, C_local)
```

### 3. Serial - 串行循环

`Serial` 用于构建串行 for 循环，按顺序执行迭代。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:152`

```python
def serial(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    step: tir.PrimExpr | None = None,
    *,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | `tir.PrimExpr` | 迭代起始值 |
| `stop` | `tir.PrimExpr \| None` | 迭代结束值 |
| `step` | `tir.PrimExpr \| None` | 迭代步长 |
| `annotations` | `dict[str, Any] \| None` | 可选的循环注解 |

**别名**: `Serial` 是 `serial` 的大写别名（`loop.py:261`）

**使用示例**:

```python
import tilelang.language as T

# 基本串行循环
for i in T.Serial(0, 128):
    C[i] = A[i] + B[i]

# 带步长的串行循环
for i in T.Serial(0, 128, 2):
    C[i] = A[i] + B[i]

# 简写形式
for i in T.Serial(128):
    C[i] = A[i] + B[i]
```

### 4. Unroll - 循环展开

`Unroll` 用于构建展开的 for 循环，编译器会将循环体复制多次。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:193`

```python
def unroll(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    step: tir.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | `tir.PrimExpr` | 迭代起始值 |
| `stop` | `tir.PrimExpr \| None` | 迭代结束值 |
| `step` | `tir.PrimExpr \| None` | 迭代步长 |
| `explicit` | `bool` | 是否显式展开循环 |
| `unroll_factor` | `int \| None` | 循环展开因子 |
| `annotations` | `dict[str, Any] \| None` | 可选的循环注解 |

**注意**: 当 `unroll_factor` 不为 None 时，`pragma_unroll_explicit` 必须为 True。

**别名**: `Unroll` 是 `unroll` 的大写别名（`loop.py:273`）

**使用示例**:

```python
import tilelang.language as T

# 基本展开循环
for i in T.Unroll(0, 8):
    C[i] = A[i] + B[i]

# 显式展开
for i in T.Unroll(0, 16, explicit=True):
    C[i] = A[i] + B[i]

# 指定展开因子
for i in T.Unroll(0, 32, unroll_factor=4):
    C[i] = A[i] + B[i]
```

### 5. Vectorized - 向量化循环

`Vectorized` 用于构建向量化的 for 循环，启用 SIMD 指令优化。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:287`

```python
def vectorized(
    start: tir.PrimExpr,
    stop: tir.PrimExpr | None = None,
    *,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | `tir.PrimExpr` | 迭代起始值 |
| `stop` | `tir.PrimExpr \| None` | 迭代结束值 |
| `annotations` | `dict[str, Any] \| None` | 可选的循环注解 |

**别名**: `Vectorized` 是 `vectorized` 的大写别名（`loop.py:314`）

**使用示例**:

```python
import tilelang.language as T

# 向量化循环
for i in T.Vectorized(0, 4):
    C[i] = A[i] + B[i]
```

### 6. Persistent - 持久化循环

`Persistent` 用于构建持久化的 for 循环，适用于特定的 GPU 内核调度场景。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py:90`

```python
def Persistent(
    domain: list[tir.PrimExpr],
    wave_size: tir.PrimExpr,
    index: tir.PrimExpr,
    group_size: tir.PrimExpr | int | None = 8,
) -> frame.ForFrame:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `domain` | `list[tir.PrimExpr]` | 主导列表 |
| `wave_size` | `tir.PrimExpr` | Wave 大小 |
| `index` | `tir.PrimExpr` | 一个 wave 中的 tile 索引 |
| `group_size` | `tir.PrimExpr \| int \| None` | 组大小，默认为 8 |

## 条件执行

### 1. If-Then-Else 语句

TileLang 支持标准的 if-then-else 条件分支。

**源码位置**:
- `If`: `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py:1095`
- `Then`: `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py:1114`
- `Else`: `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py:1125`

```python
def If(condition: PrimExpr) -> frame.IfFrame:
def Then() -> frame.ThenFrame:
def Else() -> frame.ElseFrame:
```

**使用示例**:

```python
import tilelang.language as T

with T.If(condition):
    with T.Then():
        # 条件为真时执行
        C[i] = A[i] + B[i]
    with T.Else():
        # 条件为假时执行
        C[i] = 0
```

### 2. where - 块谓词

`where` 用于为代码块设置谓词条件，类似于守卫条件。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py:374`

```python
def where(predicate: Union[PrimExpr, int]) -> None:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `predicate` | `PrimExpr \| int` | 谓词条件，只能是 0 或 1 |

**使用示例**:

```python
import tilelang.language as T

T.where(i < 128)
C[i] = A[i] + B[i]
```

### 3. if_then_else - 条件表达式

`if_then_else` 是一个函数式条件表达式，根据条件返回两个值之一。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/tir/op.py:3048`

```python
def if_then_else(cond, t, f, span=None):
```

**重要区别**: 与 `Select` 不同，`if_then_else` 不会执行不满足条件的分支，可用于防止越界访问。但如果向量中某些通道的条件不同，`if_then_else` 无法被向量化。

**使用示例**:

```python
import tilelang.language as T

# 条件表达式
val = T.if_then_else(i < N, A[i], 0)
C[i] = val
```

## 逻辑操作

TileLang 提供了对缓冲区元素进行逻辑判断的函数。

### 1. any_of - 任意元素为真

检查缓冲区中是否有任意元素为真。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/logical.py:12`

```python
def any_of(buffer: BufferLikeType) -> tir.PrimExpr:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `buffer` | `BufferLikeType` | TVM buffer 或 buffer region |

**返回值**: 执行 `any` 操作的 TVM intrinsic 调用

**使用示例**:

```python
import tilelang.language as T

# 检查是否有任何元素满足条件
has_positive = T.any_of(A > 0)
```

### 2. all_of - 所有元素为真

检查缓冲区中是否所有元素都为真。

**源码位置**: `/root/dev/vibe_dsl/tilelang/tilelang/language/logical.py:51`

```python
def all_of(buffer: BufferLikeType) -> tir.PrimExpr:
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `buffer` | `BufferLikeType` | TVM buffer 或 buffer region |

**返回值**: 执行 `all` 操作的 TVM intrinsic 调用

**使用示例**:

```python
import tilelang.language as T

# 检查是否所有元素都满足条件
all_positive = T.all_of(A > 0)
```

## 分支与跳转

### loop_break - 跳出循环

`loop_break` 用于跳出当前循环。

**源码位置**:
- `/root/dev/vibe_dsl/tilelang/tilelang/language/customize.py:76`
- `/root/dev/vibe_dsl/tilelang/tilelang/language/builtin.py:991`

```python
def loop_break() -> PrimExpr:
```

**返回值**: 对 `tl.loop_break` intrinsic 的调用

**使用示例**:

```python
import tilelang.language as T

for i in T.Serial(0, 128):
    with T.If(A[i] == 0):
        with T.Then():
            T.loop_break()  # 遇到 0 时跳出循环
```

## 循环类型对比

| 循环类型 | 执行方式 | 适用场景 | 性能特点 |
|----------|----------|----------|----------|
| `Parallel` | 并行执行 | 元素级操作、张量计算 | 最大化并行度 |
| `Pipelined` | 流水线执行 | 数据加载与计算重叠 | 隐藏内存延迟 |
| `Serial` | 串行执行 | 顺序依赖操作 | 简单直接 |
| `Unroll` | 展开执行 | 小循环、减少分支开销 | 增加代码大小 |
| `Vectorized` | SIMD 执行 | 数据并行操作 | 利用向量指令 |
| `Persistent` | 持久化调度 | 特定 GPU 内核场景 | 优化线程利用率 |

## 最佳实践

1. **并行循环**: 使用 `Parallel` 进行独立的元素级计算，最大化 GPU 并行性
2. **流水线优化**: 使用 `Pipelined` 将数据加载与计算重叠，隐藏内存访问延迟
3. **循环展开**: 对于小循环使用 `Unroll` 减少循环开销，但要注意代码膨胀
4. **向量化**: 使用 `Vectorized` 启用 SIMD 指令，提高数据并行操作的效率
5. **条件执行**: 使用 `where` 设置块级谓词，`if_then_else` 用于表达式级条件
6. **循环控制**: 使用 `loop_break` 提前退出循环，但注意控制流复杂性

## 相关文件

- `/root/dev/vibe_dsl/tilelang/tilelang/language/loop.py` - 循环管理实现
- `/root/dev/vibe_dsl/tilelang/tilelang/language/logical.py` - 逻辑操作实现
- `/root/dev/vibe_dsl/tilelang/tilelang/language/ast/ir.py` - 条件分支实现
- `/root/dev/vibe_dsl/tilelang/tilelang/language/customize.py` - 循环控制实现
- `/root/dev/vibe_dsl/tilelang/tilelang/language/builtin.py` - 内置函数实现
- `/root/dev/vibe_dsl/tilelang/tilelang/language/tir/op.py` - 操作符实现
