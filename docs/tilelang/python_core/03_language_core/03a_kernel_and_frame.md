# TileLang DSL 语言核心 - Kernel 与 Frame 模块

本文档深入分析 TileLang DSL 的核心模块，包括 Kernel 内核定义、Frame 栈帧管理以及装饰器机制。

## 目录

1. [模块概述](#模块概述)
2. [KernelLaunchFrame 详解](#kernellaunchframe-详解)
3. [Kernel 函数实现](#kernel-函数实现)
4. [LetFrame 与变量绑定](#letframe-与变量绑定)
5. [装饰器机制](#装饰器机制)

---

## 模块概述

TileLang 的 DSL 语言核心由三个主要文件组成：

| 文件 | 大小 | 功能 |
|------|------|------|
| `tilelang/language/kernel.py` | 13KB | 内核启动框架、GPU/CPU Kernel 定义、线程/块绑定管理 |
| `tilelang/language/frame.py` | 6KB | LetFrame 实现、变量绑定栈管理 |
| `tilelang/language/customize.py` | 3KB | 张量操作工具函数（reshape, view, clamp, dp4a 等） |

### 核心架构关系

```
T.Kernel()                    # kernel.py:229
    ↓
KernelLaunchFrame (TIRFrame)  # kernel.py:96
    ↓
_ffi_api.KernelLaunch()       # C++ FFI 调用
    ↓
TIR Block/Thread Bindings     # blockIdx/threadIdx
```

---

## KernelLaunchFrame 详解

### 类定义与继承

`KernelLaunchFrame` 继承自 TVM 的 `TIRFrame`，是 TileLang 内核执行的核心上下文管理器。

```python
@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """
```

**源码位置**: `tilelang/language/kernel.py:95-227`

### 核心功能

#### 1. 上下文管理 (`__enter__` / `__exit__`)

```python
def __enter__(self) -> Var | list[Var]:
    """
    Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
    Returns one Var if we detect exactly 5 frames (meaning there is a single
    block dimension), or a list of Vars otherwise.
    """
    super().__enter__()
    _get_current_stack().push(self)

    last_block_frame = self.frames[-1]
    assert isinstance(last_block_frame, BlockFrame)

    maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)

    if maybe_cpu:
        # CPU kernel frame, return a list of for frame items.
        return _normalize_bindings([frame.vars[0] for frame in self.frames[0:-1]])
    else:
        # Otherwise, return a list of iter_var.var objects (excluding the last 4 frames).
        # As 4 frames for threadIdx.x, threadIdx.y, threadIdx.z and block frame with attributes
        return _normalize_bindings([frame.iter_var.var for frame in self.frames[0:-4]])
```

**源码位置**: `tilelang/language/kernel.py:102-123`

**关键逻辑**:
- GPU Kernel: 排除最后 4 个帧（threadIdx.x, threadIdx.y, threadIdx.z, block frame）
- CPU Kernel: 排除最后 1 个帧（block frame），返回 for 循环变量

#### 2. 线程绑定查询

```python
def get_thread_binding(self, dim: int = 0) -> Var:
    """
    Returns the thread binding for the given dimension.
    dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
    """
    return self.frames[-4 + dim].iter_var.var

def get_thread_bindings(self) -> list[Var]:
    """Returns the thread binding for the given dimension."""
    return [frame.iter_var.var for frame in self.frames[-4:-1]]
```

**源码位置**: `tilelang/language/kernel.py:171-183`

#### 3. 块绑定查询

```python
def get_block_binding(self, dim: int = 0) -> Var:
    """
    Returns the block binding for the given dimension.
    dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
    """
    return self.frames[dim].iter_var.var

def get_block_bindings(self) -> list[Var]:
    """Returns all three block bindings."""
    return [frame.iter_var.var for frame in self.frames[0:-4]]
```

**源码位置**: `tilelang/language/kernel.py:194-205`

#### 4. 线程/块数量查询

```python
def get_thread_extent(self, dim: int) -> int:
    """Returns the thread extent for the given dimension."""
    iter_var = self.frames[-4 + dim].iter_var
    return int(iter_var.dom.extent)

def get_block_extent(self, dim: int) -> int:
    """Returns the block extent for the given dimension."""
    iter_var = self.frames[dim].iter_var
    return int(iter_var.dom.extent)

def get_num_threads(self) -> int:
    """Returns the total number of threads."""
    num_threads: int = 1
    for thread_dim in range(3):
        num_threads *= self.get_thread_extent(thread_dim)
    return num_threads
```

**源码位置**: `tilelang/language/kernel.py:143-192`

### 栈管理

`KernelLaunchFrame` 使用线程本地存储管理栈：

```python
# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local = threading.local()

def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "kernel_launch_frame_stack"):
        _local.kernel_launch_frame_stack = FrameStack()
    return _local.kernel_launch_frame_stack
```

**源码位置**: `tilelang/language/kernel.py:73-81`

### FrameStack 实现

```python
class FrameStack:
    """
    A simple stack-like wrapper around a deque that provides
    push, pop, and top methods for convenience.
    """

    def __init__(self):
        self._stack = deque()

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        raise IndexError(f"{self.__class__.__name__} is empty")

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")
```

**源码位置**: `tilelang/language/kernel.py:26-71`

---

## Kernel 函数实现

### 函数签名

```python
def Kernel(
    *blocks: int | tir.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
```

**源码位置**: `tilelang/language/kernel.py:229-235`

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `*blocks` | `int \| tir.PrimExpr` | 块维度 (gridDim.x/y/z)，支持 1-3 维 |
| `threads` | `int \| list[int] \| tuple \| None` | 线程维度 (blockDim.x/y/z)，默认 128 |
| `cluster_dims` | `int \| tuple[int, int, int] \| list[int] \| None` | SM90+ 集群启动维度 |
| `is_cpu` | `bool` | 是否为 CPU 内核（不绑定 threadIdx/blockIdx） |
| `prelude` | `str \| None` | C 代码前缀，注入到生成代码之前 |

### 使用示例

```python
# 1D CUDA kernel
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    # bx is the blockIdx.x binding
    ...

# 2D grid with 2D thread block
with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
    tx, ty = T.get_thread_bindings()
    ...

# CPU kernel
with T.Kernel(loop_extent, is_cpu=True) as (i,):
    ...
```

### 实现逻辑

```python
def Kernel(*blocks, threads=None, cluster_dims=None, is_cpu=False, prelude=None):
    # 检查是否在 JIT/prim_func 上下文中
    from tilelang.language.eager.builder import Builder
    if Builder.current() is None:
        raise JITNoBuilderError("T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context.")

    attrs: dict = {}

    # 默认线程数
    if not is_cpu and threads is None:
        threads = 128

    # 标准化线程维度为 [x, y, z]
    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))

    # CPU 内核标记
    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    # C 代码前缀
    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    # 集群维度处理 (SM90+)
    if cluster_dims is not None:
        if isinstance(cluster_dims, (list, tuple)):
            cluster_dims = list(cluster_dims) + [1] * (3 - len(cluster_dims))
        elif isinstance(cluster_dims, int):
            cluster_dims = [cluster_dims, 1, 1]

        if cluster_dims != [1, 1, 1]:
            attrs["cluster_dims"] = cluster_dims

    return _ffi_api.KernelLaunch(blocks, threads, attrs)
```

**源码位置**: `tilelang/language/kernel.py:289-329`

### 辅助函数

TileLang 提供了一系列便捷的绑定查询函数：

```python
def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_binding(dim)

def get_thread_bindings() -> list[Var]:
    """Returns all three thread bindings."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_bindings()

def get_block_binding(dim: int = 0) -> Var:
    """Returns the block binding for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_binding(dim)

def get_block_bindings() -> list[Var]:
    """Returns all three block bindings."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_bindings()
```

**源码位置**: `tilelang/language/kernel.py:332-353`

### Var 类扩展

为了支持单维度内核绑定的可迭代解包，TileLang 对 TVM 的 `Var` 类进行了扩展：

```python
# Ensure single-dimension kernel bindings can be unpacked like iterables.
# especially for issue https://github.com/tile-ai/tilelang/issues/830
if not hasattr(Var, "__iter__"):
    def _var_iter(self):
        yield self
    Var.__iter__ = _var_iter

if not hasattr(Var, "__len__"):
    Var.__len__ = lambda self: 1
```

**源码位置**: `tilelang/language/kernel.py:15-23`

这使得用户可以统一使用以下两种语法：
```python
with T.Kernel(...) as pid:      # 单维度
with T.Kernel(...) as (pid,):   # 解包语法
```

---

## LetFrame 与变量绑定

### 类定义

`LetFrame` 继承自 TVM 的 `TIRFrame`，用于管理 let 绑定的变量作用域。

```python
@_register_object("script.ir_builder.tir.LetFrame")
class LetFrame(TIRFrame):
    """A TIR frame for let bindings that manages variable scope and value tracking.

    This frame type extends TIRFrame to provide variable binding functionality and
    maintains a global stack of active bindings.
    """
```

**源码位置**: `tilelang/language/frame.py:110-116`

### FrameStack 与变量映射

与 `KernelLaunchFrame` 不同，`LetFrame` 的栈管理器维护了一个变量到值的映射：

```python
class FrameStack:
    """A stack-like container for managing TIR frame objects and their variable bindings."""

    def __init__(self):
        self._stack = deque()
        self._var_value_map = {}

    def push(self, item):
        self._stack.append(item)
        if hasattr(item, "var") and hasattr(item, "value"):
            self._var_value_map[item.var] = item.value

    def pop(self):
        if self._stack:
            item = self._stack.pop()
            if hasattr(item, "var"):
                self._var_value_map.pop(item.var, None)
            return item
        raise IndexError(f"{self.__class__.__name__} is empty")

    def get_value(self, var):
        return self._var_value_map.get(var)

    def has_value(self, var):
        return var in self._var_value_map
```

**源码位置**: `tilelang/language/frame.py:13-97`

### BufferLoad 处理

`LetFrame` 在 `__enter__` 中处理特殊的 BufferLoad 情况：

```python
def __enter__(self) -> Var:
    super().__enter__()
    if isinstance(self.value, BufferLoad):
        indices = self.value.indices
        is_block_load = False
        for index in indices[:-1]:
            if DataType(index.dtype).lanes > 1:
                is_block_load = True
                break
        if is_block_load:
            self.value = BufferRegion(self.value.buffer, [Range(x.base, x.lanes) for x in indices])

    _get_let_stack().push(self)
    return self.var
```

**源码位置**: `tilelang/language/frame.py:118-136`

这段代码检测是否为块加载（block load），如果是，则将 `BufferLoad` 转换为 `BufferRegion`。

### 辅助函数

```python
def has_let_value(var: Var) -> bool:
    """Check if a variable has a binding in the current let frame stack."""
    return _get_let_stack().has_value(var)

def get_let_value(var: Var) -> PrimExpr | None:
    """Get the value bound to a variable in the current let frame stack."""
    return _get_let_stack().get_value(var)
```

**源码位置**: `tilelang/language/frame.py:188-209`

---

## 装饰器机制

### customize.py 模块概述

`customize.py` 提供了一系列常用的张量编程操作，这些操作直接暴露在 TileLang 语言表面。

**源码位置**: `tilelang/language/customize.py`

### 核心函数

#### 1. dp4a - 4元素点积累加

```python
def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern(
        "handle",
        "DP4A",
        T.access_ptr(A, "r"),
        T.access_ptr(B, "r"),
        T.access_ptr(C, "rw"),
    )
```

**源码位置**: `tilelang/language/customize.py:11-28`

#### 2. clamp - 数值钳制

```python
def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]"""
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst
```

**源码位置**: `tilelang/language/customize.py:31-44`

#### 3. reshape - 缓冲区重塑

```python
def reshape(src: Buffer, shape: ShapeType) -> Buffer:
    """Reshapes the input buffer to the specified shape."""
    assert prim_expr_equal(bits_product(shape, src.dtype), bits_product(src.shape, src.dtype)), (
        f"T.reshape/view shape check failed. src {src} src.shape: {src.shape}, src.dtype: {src.dtype}, target shape: {shape}, target dtype: {src.dtype}"
    )
    return T.Tensor(shape, src.dtype, src.data)
```

**源码位置**: `tilelang/language/customize.py:47-60`

#### 4. view - 张量视图

```python
def view(src: Buffer, shape: ShapeType | None = None, dtype: DType | None = None) -> Buffer:
    """Return a Tensor view of the input buffer with an optional new shape and dtype.

    If `shape` is None the source buffer's shape is used; if `dtype` is None the source buffer's dtype is used. The returned buffer shares the same underlying data as `src` (no copy).
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    assert prim_expr_equal(bits_product(shape, dtype), bits_product(src.shape, src.dtype)), "T.reshape/view shape check failed."
    return T.Tensor(shape, dtype, src.data)
```

**源码位置**: `tilelang/language/customize.py:63-73`

#### 5. loop_break - 循环中断

```python
def loop_break() -> PrimExpr:
    """Break out of the current loop."""
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))
```

**源码位置**: `tilelang/language/customize.py:76-82`

### 原子操作导入

```python
from .atomic import (
    atomic_max, atomic_min, atomic_add, atomic_addx2, atomic_addx4,
    atomic_load, atomic_store
)
```

**源码位置**: `tilelang/language/customize.py:8`

---

## 总结

### 关键设计模式

1. **上下文管理器模式**: `KernelLaunchFrame` 和 `LetFrame` 都使用 `__enter__`/`__exit__` 管理作用域
2. **线程本地存储**: 使用 `threading.local()` 避免跨线程干扰
3. **栈结构管理**: 使用 `deque` 实现高效的栈操作
4. **FFI 桥接**: 通过 `_ffi_api` 调用底层 C++ 实现

### 文件引用汇总

| 功能 | 文件路径 | 行号范围 |
|------|----------|----------|
| KernelLaunchFrame 类 | `tilelang/language/kernel.py` | 95-227 |
| Kernel 函数 | `tilelang/language/kernel.py` | 229-329 |
| FrameStack (Kernel) | `tilelang/language/kernel.py` | 26-71 |
| LetFrame 类 | `tilelang/language/frame.py` | 110-186 |
| FrameStack (Let) | `tilelang/language/frame.py` | 13-97 |
| 工具函数 | `tilelang/language/customize.py` | 1-83 |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    TileLang DSL 语言层                       │
├─────────────────────────────────────────────────────────────┤
│  T.Kernel()        T.get_thread_binding()    T.reshape()   │
│       │                   │                       │         │
│       ▼                   ▼                       ▼         │
│  KernelLaunchFrame    KernelLaunchFrame       customize.py  │
│  (kernel.py:96)       .Current()              (view, clamp) │
│       │                   │                                 │
│       ▼                   ▼                                 │
│  _ffi_api.KernelLaunch  _get_current_stack()               │
│       │                   │                                 │
│       ▼                   ▼                                 │
│  ┌──────────────────────────────────────┐                  │
│  │      Thread Local FrameStack         │                  │
│  │  (threading.local() isolation)       │                  │
│  └──────────────────────────────────────┘                  │
│       │                                                     │
│       ▼                                                     │
│  TVM TIR Block/Thread Bindings                              │
└─────────────────────────────────────────────────────────────┘
```
