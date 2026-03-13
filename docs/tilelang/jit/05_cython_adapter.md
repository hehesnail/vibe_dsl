# TileLang JIT Cython Adapter

## 概述

Cython Adapter 是 TileLang JIT 系统的高性能适配器，利用 Cython 编译生成高效的 C/C++ 扩展模块。该适配器通过 `tilelang_cython_wrapper` 模块提供低延迟的 kernel 调用接口，适用于对性能要求极高的场景。

**源码文件**: `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/cython/adapter.py`

## 架构设计

### 类层次结构

```
BaseKernelAdapter (base.py)
    └── CythonKernelAdapter (cython/adapter.py)
```

### 核心职责

1. **Cython 集成**: 通过 `tilelang_cython_wrapper` 调用 Cython 编译的 wrapper
2. **静态信息提取**: 预处理 buffer 的形状、步长、数据类型等静态信息
3. **多目标支持**: 支持 CUDA、HIP、CPU、Metal 等多种后端
4. **张量验证**: 可选的输入张量属性验证
5. **低延迟调用**: 通过 Cython 直接调用底层库，减少 Python 开销

## Cython Kernel Adapter

### 类定义

**`CythonKernelAdapter`** (cython/adapter.py:38):

```python
class CythonKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using cython.

    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """
```

### 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `target` | str \| Target | 目标平台 |
| `ir_module` | tvm.IRModule | TVM IR 模块 |
| `lib` | ctypes.CDLL | 编译后的库句柄 |
| `dynamic_symbolic_map` | dict | 动态符号映射 |
| `ptr_map` | dict | 指针参数映射 |
| `buffer_dtype_map` | dict | buffer 数据类型映射 |
| `static_shape_map` | dict | 静态形状信息 |
| `static_strides_map` | dict | 静态步长信息 |
| `static_contiguous_list` | list | 连续 buffer 列表 |
| `buffer_device_map` | dict | buffer 设备映射 |

### 初始化流程

**`__init__` 方法** (cython/adapter.py:75-150):

```python
def __init__(
    self,
    params: list[KernelParam],
    result_idx: list[int],
    target: str | Target,
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    host_mod: tvm.IRModule | None = None,
    device_mod: tvm.IRModule | None = None,
    device_kernel_source: str | None = None,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
):
```

初始化步骤:
1. 存储参数和输出索引
2. 处理 IR 模块
3. 处理动态符号映射 (`_process_dynamic_symbolic`)
4. 处理 buffer 数据类型 (`_process_buffer_dtype`)
5. 处理指针参数映射 (`_process_ptr_map`)
6. 处理 buffer 设备 (`_process_buffer_device`)
7. 处理静态 buffer 信息 (`_process_static_buffer_infos`)
8. 创建 `TLWrapper` 和 `LibraryGenerator`
9. 编译库并加载
10. 初始化 Cython wrapper

### 静态信息处理

#### Buffer 数据类型处理

**`_process_buffer_dtype` 方法** (cython/adapter.py:241-255):

```python
def _process_buffer_dtype(self) -> dict[tir.Var, tuple[int, torch.dtype]]:
    """Extract information about buffer dtypes from the TIR function.

    Maps buffer variables to their corresponding dtypes.
    """
    func = self.prim_func
    params = func.params
    buffer_map = func.buffer_map
    buffer_dtype_map = {}
    for i, param in enumerate(params):
        if param in buffer_map:
            buffer = buffer_map[param]
            name, dtype = buffer.name, buffer.dtype
            buffer_dtype_map[name] = (i, map_torch_type(dtype))
    return buffer_dtype_map
```

#### 指针参数处理

**`_process_ptr_map` 方法** (cython/adapter.py:257-269):

```python
def _process_ptr_map(self) -> dict[int, str]:
    """Extract information about pointer arguments from the TIR function.

    Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    for runtime shape resolution.
    """
    func = self.prim_func
    params = func.params
    ptr_map = {}
    for i, param in enumerate(params):
        if param.dtype == "handle":
            ptr_map[i] = param.name
    return ptr_map
```

#### 静态 Buffer 信息

**`_process_static_buffer_infos` 方法** (cython/adapter.py:271-306):

```python
def _process_static_buffer_infos(
    self,
) -> tuple[dict, dict, list]:
    """Extract information about static shapes from the TIR function.

    Maps buffer variables to their corresponding static shapes.
    """
```

处理内容包括:
- 静态形状: `[(dim_index, value), ...]`
- 静态步长: `[(dim_index, value), ...]`
- 连续性检测: 检查 buffer 是否为连续存储

### 设备处理

**`_process_buffer_device` 方法** (cython/adapter.py:308-332):

```python
def _process_buffer_device(self) -> dict[tir.Var, tuple[int, torch.device]]:
    """Extract information about buffer devices from the TIR function.

    Maps buffer variables to their corresponding devices.
    """
    func = self.prim_func
    params = func.params
    buffer_map = func.buffer_map
    buffer_device_map = {}
    device = None
    if is_cuda_target(self.target) or is_hip_target(self.target):
        device = torch.device("cuda")
    elif is_cpu_target(self.target):
        device = torch.device("cpu")
    elif is_metal_target(self.target):
        device = torch.device("mps")
    # ...
```

### Cython Wrapper 集成

**初始化 Cython Wrapper** (cython/adapter.py:142-149):

```python
self.cython_wrapper = CythonKernelWrapper(self.result_idx, self.params, self.lib)
self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
self.cython_wrapper.set_static_shape_map(self.static_shape_map)
self.cython_wrapper.set_static_strides_map(self.static_strides_map)
self.cython_wrapper.set_static_contiguous_list(self.static_contiguous_list)
self.cython_wrapper.set_buffer_device_map(self.buffer_device_map)
self.cython_wrapper.set_ptr_map(self.ptr_map)
```

### Kernel 执行

**`_forward_from_prebuild_lib` 方法** (cython/adapter.py:334-341):

```python
def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
    """Low-level function to call the compiled CUDA kernel.

    Converts PyTorch tensor pointers to C void pointers for ctypes interface.
    """
    ctypes_args = [ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args]
    ctypes_args.append(ctypes.c_void_p(stream))
    self.lib.call(*ctypes_args)
```

**`_convert_torch_func` 方法** (cython/adapter.py:343-356):

```python
def _convert_torch_func(self) -> Callable:
    """Returns a PyTorch-compatible function wrapper for the kernel."""

    def lambda_forward(*args, stream: int = -1, skip_tensor_validation: bool = False):
        """
        Args:
            args: List of input tensors
            stream: CUDA stream ID, default to -1, will use the current stream if not specified
            skip_tensor_validation: Whether to skip tensor attributes validation which
                includes shape, dtype, device, etc.
        """
        return self.cython_wrapper.forward([*args], stream=stream, skip_tensor_validation=skip_tensor_validation)

    return lambda_forward
```

### 从数据库加载

**`from_database` 类方法** (cython/adapter.py:152-214):

```python
@classmethod
def from_database(
    cls,
    params: list[TensorType],
    result_idx: list[int],
    target: str,
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    host_kernel_source: str,
    device_kernel_source: str,
    kernel_lib_path: str,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
):
```

支持从预编译的库路径加载适配器，跳过编译步骤。

## CythonKernelWrapper

`CythonKernelWrapper` 是 Cython 模块 `tilelang_cython_wrapper` 提供的类，它:

1. **高性能张量处理**: 直接访问 PyTorch 张量的底层数据
2. **动态形状解析**: 根据运行时输入计算输出形状
3. **张量验证**: 检查形状、数据类型、设备等属性
4. **自动设备管理**: 推断和使用正确的 CUDA 设备

### 接口方法

```python
# 设置各种映射
set_dynamic_symbolic_map(self.dynamic_symbolic_map)
set_buffer_dtype_map(self.buffer_dtype_map)
set_static_shape_map(self.static_shape_map)
set_static_strides_map(self.static_strides_map)
set_static_contiguous_list(self.static_contiguous_list)
set_buffer_device_map(self.buffer_device_map)
set_ptr_map(self.ptr_map)

# 执行 kernel
forward([*args], stream=stream, skip_tensor_validation=False)
```

## 多目标支持

Cython Adapter 支持多种目标平台:

| 目标 | 检测函数 | 设备 |
|------|----------|------|
| CUDA | `is_cuda_target()` | `torch.device("cuda")` |
| HIP | `is_hip_target()` | `torch.device("cuda")` |
| CPU | `is_cpu_target()` | `torch.device("cpu")` |
| Metal | `is_metal_target()` | `torch.device("mps")` |

## 属性访问

### 源代码路径

```python
@property
def srcpath(self):
    """Returns the source path of the compiled library."""
    return self.lib_generator.srcpath

@property
def libpath(self):
    """Returns the path to the compiled library."""
    return self.lib_generator.libpath

@property
def lib_code(self):
    """Returns the code of the compiled library."""
    return self.lib_generator.lib_code
```

### 动态形状检测

```python
@property
def is_dynamic(self):
    """Indicates whether the kernel handles dynamic shapes."""
    return self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0
```

## 使用示例

```python
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang.engine.param import KernelParam

# 创建适配器
adapter = CythonKernelAdapter(
    params=[KernelParam(...), ...],
    result_idx=[-1],
    target="cuda",
    func_or_mod=ir_module,
    device_kernel_source=cuda_source,
    verbose=True
)

# 执行 kernel (使用当前 stream)
output = adapter(input_tensor1, input_tensor2)

# 执行 kernel (指定 stream，跳过验证)
output = adapter(input_tensor1, input_tensor2, stream=stream_id, skip_tensor_validation=True)

# 检查是否为动态形状
if adapter.is_dynamic:
    print("Kernel handles dynamic shapes")

# 获取编译后的库路径
print(f"Library path: {adapter.libpath}")
```

## 与其他 Adapter 对比

| 特性 | Cython Adapter | TVM FFI Adapter | NVRTC Adapter | CuTeDSL Adapter |
|------|----------------|-----------------|---------------|-----------------|
| 包装层 | Cython | TVM Runtime | Python | C++ + Python |
| 编译时机 | 构建时 | 运行时 | 运行时 | 运行时 |
| 启动延迟 | 极低 | 低 | 中等 | 低 |
| 多目标 | 是 | 是 | CUDA only | CUDA only |
| 张量验证 | 支持 | 支持 | 支持 | 支持 |
| 静态信息 | 预处理 | 运行时 | 运行时 | 运行时 |

## 总结

Cython Adapter 是 TileLang 的高性能适配器，它:

1. **Cython 加速**: 通过 Cython wrapper 最小化 Python 调用开销
2. **静态预处理**: 提前提取形状、步长等静态信息
3. **多目标支持**: 统一接口支持 CUDA、HIP、CPU、Metal
4. **灵活验证**: 支持跳过张量验证以获取极致性能
5. **缓存友好**: 支持从预编译库加载
