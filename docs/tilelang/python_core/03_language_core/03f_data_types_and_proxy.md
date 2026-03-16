# TileLang 数据类型与代理模块

## 模块概述

TileLang 的数据类型系统 (`dtypes.py`) 和代理模块 (`proxy.py`) 构成了语言核心的基础组件，负责：

1. **数据类型系统** - 提供丰富的数值类型支持，包括标量类型、向量类型和低精度浮点类型
2. **代理对象系统** - 提供 Buffer/Tensor 的抽象代理，支持不同内存作用域的缓冲区创建

这两个模块协同工作，为 TileLang 的 DSL 提供了类型安全和内存管理的基础能力。

---

## 数据类型系统详解

### 核心类型定义

TileLang 的数据类型系统基于 TVM 的 `DataType` 构建，通过类型别名和扩展提供了丰富的类型支持。

**文件**: `/root/dev/vibe_dsl/tilelang/tilelang/language/dtypes.py`

#### 基础类型别名

```python
# 代码位置: dtypes.py:413-420
bool = dtype("bool")
short = dtype("int16")
int = dtype("int32")
uint = dtype("uint32")
long = dtype("int64")
half = dtype("float16")
float = dtype("float32")
double = dtype("float64")
```

#### 整数类型

**有符号整数** (`dtypes.py:421-449`):
- 标量: `int4`, `int8`, `int16`, `int32`, `int64`
- 向量类型 (x2, x4, x8, x16, x32, x64): 如 `int8x2`, `int32x4`, `int64x8`

**无符号整数** (`dtypes.py:450-477`):
- 标量: `uint8`, `uint16`, `uint32`, `uint64`
- 向量类型: 如 `uint8x2`, `uint32x4`, `uint64x16`

#### 浮点类型

**标准浮点** (`dtypes.py:478-498`):
- `float16` / `half` - 半精度浮点
- `float32` / `float` - 单精度浮点
- `float64` / `double` - 双精度浮点
- `bfloat16` - Brain Floating Point 16

**8位浮点类型** (`dtypes.py:499-547`):
- `float8_e3m4` - 3位指数，4位尾数
- `float8_e4m3` - 4位指数，3位尾数
- `float8_e4m3b11fnuz` - 带偏置的变体
- `float8_e4m3fn` - NVIDIA 标准 FP8 E4M3
- `float8_e4m3fnuz` - AMD 变体
- `float8_e5m2` - NVIDIA 标准 FP8 E5M2
- `float8_e5m2fnuz` - AMD 变体
- `float8_e8m0fnu` - 8位指数，0位尾数

**6位浮点类型** (`dtypes.py:555-568`):
- `float6_e2m3fn` - 2位指数，3位尾数
- `float6_e3m2fn` - 3位指数，2位尾数

**4位浮点类型** (`dtypes.py:569-575`):
- `float4_e2m1fn` - 2位指数，1位尾数

所有浮点类型都支持向量变体 (x2, x4, x8, x16, x32, x64)。

### 类型转换映射

#### Python 类型映射 (`dtypes.py:26-30`)

```python
_PYTHON_DTYPE_TO_STR = {
    bool: "bool",
    int: "int32",
    float: "float32",
}
```

#### NumPy 类型映射 (`dtypes.py:32-52`)

```python
_NUMPY_DTYPE_TO_STR = {
    np.bool_: "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    # ... 更多映射
}
```

#### PyTorch 类型映射 (`dtypes.py:54-94`)

```python
_TORCH_DTYPE_TO_STR = {
    torch.bool: "bool",
    torch.int8: "int8",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    # ... 更多映射
}
```

### dtype 类扩展

TileLang 通过猴子补丁为 TVM 的 `DataType` 类添加了多个方法：

#### `__call__` 方法 (`dtypes.py:138-164`)

允许使用 dtype 作为构造函数创建常量：

```python
def __dtype_call__(self: dtype, expr=None, is_size_var: bool = False) -> tir.Var:
    if isinstance(expr, int_):
        return tvm.tir.const(expr, dtype=self)
    # 处理其他类型...
    return call(expr, is_size_var)

# 使用示例
float32(3.14)  # 创建 float32 常量
```

#### `as_torch` 方法 (`dtypes.py:167-211`)

将 TileLang dtype 转换为 PyTorch dtype：

```python
def __dtype_as_torch__(self: dtype) -> torch.dtype:
    dtype_str = str(self)
    if dtype_str == "float8_e4m3":
        # 根据后端选择 CUDA 或 HIP 变体
        if torch.version.hip is not None:
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn
    # ... 其他转换逻辑
```

#### `bytes` 属性 (`dtypes.py:227-229`)

返回类型的字节大小：

```python
def __dtype_bytes__(self: dtype) -> int:
    return self.itemsize
```

### 类型创建与转换

#### `get_tvm_dtype` 函数 (`dtypes.py:238-241`)

统一入口，将任意类型转换为 TVM dtype：

```python
def get_tvm_dtype(value: AnyDType) -> dtype:
    if isinstance(value, (dtype, ir.Type)):
        return value
    return dtype(value)
```

#### `__dtype_new__` 构造函数 (`dtypes.py:217-224`)

支持多种输入类型的统一构造：

```python
def __dtype_new__(cls, value: AnyDType) -> dtype:
    if isinstance(value, str):
        return __orig_dtype_new(cls, _CANONICAL_TO_DISPLAY_STR.get(value, value))
    elif value in _DTYPE_TO_STR:
        return __orig_dtype_new(cls, _DTYPE_TO_STR[value])
    else:
        raise TypeError(f"Invalid DataType {value}")
```

---

## 代理对象详解

### 模块概述

代理模块 (`proxy.py`) 提供了 Buffer/Tensor 的抽象代理，用于在 TileLang DSL 中创建和管理内存缓冲区。

**文件**: `/root/dev/vibe_dsl/tilelang/tilelang/language/proxy.py`

### BufferProxy 类

`BufferProxy` 是传统的缓冲区代理类，现已标记为弃用。

**代码位置**: `proxy.py:18-71`

```python
class BufferProxy:
    """Buffer proxy class for constructing tir buffer."""

    @deprecated("T.Buffer(...)", "T.Tensor(...)")
    def __call__(self, shape, dtype=_dtypes.float32, data=None, ...):
        return buffer(shape, dtype=dtype, ...)

    @deprecated("T.Buffer[...]", "T.Tensor(...)")
    def __getitem__(self, keys) -> tir.Buffer:
        # 支持 T.Buffer[shape, dtype] 语法
        ...

    def from_ptr(self, pointer_var: Var, shape, dtype="float32", strides=None) -> tir.Buffer:
        """从指针创建缓冲区"""
        return match_buffer(pointer_var, shape, dtype=dtype, strides=strides)
```

### BaseTensorProxy 类

所有 Tensor 代理类的基类，提供可配置的默认参数。

**代码位置**: `proxy.py:83-148`

```python
class BaseTensorProxy:
    """Base proxy class for tensor types with configurable defaults."""

    default_scope = "global"
    default_align = 0
    default_offset_factor = 0

    def __call__(self, shape, dtype="float32", data=None, strides=None,
                 elem_offset=None, scope=None, align=None, ...):
        # 使用类默认值
        scope = scope or self.default_scope
        align = align or self.default_align
        return buffer(shape, dtype=dtype, scope=scope, ...)

    def __getitem__(self, keys) -> tir.Buffer:
        # 支持 T.Tensor[shape, dtype] 语法
        ...

    def from_ptr(self, pointer_var, shape, dtype="float32", strides=None):
        dtype = _normalize_tensor_dtype(dtype)
        return match_buffer(pointer_var, shape, dtype=dtype, strides=strides)
```

### 特化的 Tensor 代理类

#### TensorProxy (`proxy.py:150-168`)

默认的全局内存张量代理，自动计算连续存储的 strides：

```python
class TensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers.
    The tensor should be by default contiguous.
    """

    @staticmethod
    def _construct_strides(shape: tuple[Any]):
        s, strides = 1, [1]
        for dim in shape[:0:-1]:
            s *= dim
            strides.append(s)
        return tuple(reversed(strides))

    def __call__(self, shape, dtype="float32", data=None, scope=None):
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        return super().__call__(shape, dtype=dtype,
                               strides=TensorProxy._construct_strides(shape),
                               data=data, scope=scope)
```

#### StridedTensorProxy (`proxy.py:171-180`)

支持非连续存储的张量代理：

```python
class StridedTensorProxy(BaseTensorProxy):
    """Tensor proxy with stride information required."""

    def __call__(self, shape, strides, dtype="float32", scope=None):
        if len(shape) != len(strides):
            raise ValueError("Invalid shape/strides' dimensions")
        return super().__call__(shape, dtype=dtype, strides=strides, scope=scope)
```

#### FragmentBufferProxy (`proxy.py:183-190`)

用于 Tensor Core 操作的片段内存：

```python
class FragmentBufferProxy(BaseTensorProxy):
    """Proxy class for fragment memory buffers.
    Typically used in GPU tensor core operations.
    """
    default_scope = "local.fragment"
```

#### SharedBufferProxy (`proxy.py:193-200`)

用于 GPU 共享内存：

```python
class SharedBufferProxy(BaseTensorProxy):
    """Proxy class for shared memory buffers.
    Commonly used in GPU shared memory operations.
    """
    default_scope = "shared.dyn"
```

#### LocalBufferProxy (`proxy.py:203-210`)

用于 GPU 本地/寄存器内存：

```python
class LocalBufferProxy(BaseTensorProxy):
    """Proxy class for local memory buffers.
    Typically used for temporary computations in GPU kernels.
    """
    default_scope = "local"
```

### 内存作用域对照表

| 代理类 | 默认作用域 | 用途 |
|--------|-----------|------|
| `TensorProxy` | `global` | 全局内存，连续存储 |
| `StridedTensorProxy` | `global` | 全局内存，支持 strides |
| `FragmentBufferProxy` | `local.fragment` | Tensor Core 片段内存 |
| `SharedBufferProxy` | `shared.dyn` | GPU 共享内存 |
| `LocalBufferProxy` | `local` | GPU 本地/寄存器内存 |

### 指针与动态 Tensor 创建

#### `ptr` 函数 (`proxy.py:269-288`)

创建表示指针的 TIR 变量：

```python
def ptr(dtype: DType | None = None, storage_scope: str = "global",
        *, is_size_var: bool = False) -> Var:
    """Create a TIR var that represents a pointer."""
    return handle(dtype=dtype, storage_scope=storage_scope, is_size_var=is_size_var)
```

#### `make_tensor` 函数 (`proxy.py:329-341`)

从指针创建 Tensor：

```python
def make_tensor(ptr: Var | PrimExpr, shape, dtype="float32", strides=None) -> tir.Buffer:
    dtype = _normalize_tensor_dtype(dtype)
    if isinstance(ptr, Var):
        return Tensor.from_ptr(ptr, shape, dtype, strides)
    # 从地址表达式创建
    return make_tensor_from_addr(ptr, shape, dtype=dtype, strides=strides, ...)
```

#### `make_tensor_from_addr` 函数 (`proxy.py:310-326`)

从地址表达式创建 Tensor：

```python
def make_tensor_from_addr(addr: PrimExpr, shape, dtype="float32",
                          strides=None, storage_scope="global") -> tir.Buffer:
    pointer_var = _materialize_pointer_from_addr(addr, dtype, storage_scope)
    return buffer(shape, dtype=dtype, data=pointer_var, strides=strides, scope=storage_scope)
```

---

## 类型推断机制

### 多源类型统一

TileLang 通过 `_DTYPE_TO_STR` 字典 (`dtypes.py:110`) 统一处理来自不同来源的类型：

```python
_DTYPE_TO_STR = {**_PYTHON_DTYPE_TO_STR, **_NUMPY_DTYPE_TO_STR, **_TORCH_DTYPE_TO_STR}
```

这允许以下等价的类型指定方式：

```python
# 字符串
T.Tensor((128, 128), "float32")

# Python 类型
T.Tensor((128, 128), float)  # 映射到 float32

# NumPy 类型
T.Tensor((128, 128), np.float32)

# PyTorch 类型
T.Tensor((128, 128), torch.float32)

# TileLang dtype
T.Tensor((128, 128), T.float32)
```

### 指针类型规范化

`_normalize_tensor_dtype` 函数 (`proxy.py:74-80`) 处理指针类型的特殊情况：

```python
def _normalize_tensor_dtype(dtype: DType) -> DType:
    # T.ptr 是前端标记，运行时 ABI 使用 int64 存储
    if dtype is ptr:
        return _dtypes.int64
    return dtype
```

### 类型检查与 TYPE_CHECKING

两个模块都使用 `TYPE_CHECKING` 模式提供类型提示：

**dtypes.py** (`dtypes.py:12-21`, `dtypes.py:244-411`):
```python
if TYPE_CHECKING:
    class dtype(Generic[_T]):
        @property
        def bits(self) -> int: ...
        @property
        def bytes(self) -> int: ...
        def as_torch(self) -> torch.dtype: ...
else:
    dtype = tvm.DataType
```

**proxy.py** (`proxy.py:217-266`):
```python
if TYPE_CHECKING:
    class BaseTensor:
        def __getitem__(self, key) -> Any: ...
        def __setitem__(self, key, value) -> None: ...
        @classmethod
        def from_ptr(cls, pointer_var, shape, dtype, strides): ...
else:
    Tensor = TensorProxy()
    # ...
```

---

## 代码引用汇总

### dtypes.py 关键位置

| 功能 | 行号 |
|------|------|
| 类型别名定义 (bool, int, float等) | 413-420 |
| 整数类型定义 | 421-449 |
| 无符号整数类型定义 | 450-477 |
| 浮点类型定义 | 478-498 |
| FP8 类型定义 | 499-547 |
| FP6 类型定义 | 555-568 |
| FP4 类型定义 | 569-575 |
| Python 类型映射 | 26-30 |
| NumPy 类型映射 | 32-52 |
| PyTorch 类型映射 | 54-94 |
| `__dtype_call__` 方法 | 138-164 |
| `__dtype_as_torch__` 方法 | 167-211 |
| `__dtype_new__` 构造函数 | 217-224 |
| `get_tvm_dtype` 函数 | 238-241 |
| 所有类型名称集合 `_all_dtypes` | 578-743 |

### proxy.py 关键位置

| 功能 | 行号 |
|------|------|
| `BufferProxy` 类 | 18-71 |
| `BaseTensorProxy` 类 | 83-148 |
| `TensorProxy` 类 | 150-168 |
| `StridedTensorProxy` 类 | 171-180 |
| `FragmentBufferProxy` 类 | 183-190 |
| `SharedBufferProxy` 类 | 193-200 |
| `LocalBufferProxy` 类 | 203-210 |
| `ptr` 函数 | 269-288 |
| `make_tensor` 函数 | 329-341 |
| `make_tensor_from_addr` 函数 | 310-326 |
| `_normalize_tensor_dtype` 函数 | 74-80 |
| 类型检查定义 | 217-266 |

---

## 使用示例

### 数据类型使用

```python
import tilelang as T

# 创建常量
val = T.float32(3.14)
idx = T.int32(42)

# 类型转换到 PyTorch
torch_dtype = T.float16.as_torch()  # torch.float16

# 获取类型大小
size = T.float32.bytes  # 4
```

### Tensor 代理使用

```python
import tilelang as T

# 全局内存 Tensor
A = T.Tensor((128, 128), "float32")

# 共享内存 Buffer
B = T.SharedBuffer((128, 128), "float16")

# 本地内存 Buffer
C = T.LocalBuffer((32,), "float32")

# Tensor Core 片段内存
D = T.FragmentBuffer((16, 16), "float16")

# 从指针创建 Tensor
ptr = T.ptr("float32", "global")
E = T.Tensor.from_ptr(ptr, (128, 128), "float32")

# 动态创建 Tensor
addr = T.get_element_address(base, offset)
F = T.make_tensor(addr, (64, 64), "float32")
```
