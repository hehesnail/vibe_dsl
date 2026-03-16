# TileLang JIT TVM FFI Adapter

## 概述

TVM FFI Adapter 是 TileLang JIT 系统的核心适配器之一，负责将 TVM 编译的 kernel 与 PyTorch 张量进行桥接。该适配器通过 TVM 的 FFI (Foreign Function Interface) 机制，实现 TIR (Tensor IR) 函数到可执行 kernel 的转换。

**源码文件**: `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/tvm_ffi.py`

## 架构设计

### 类层次结构

```
BaseKernelAdapter (base.py)
    └── TVMFFIKernelAdapter (tvm_ffi.py)
```

### 核心职责

1. **TVM Runtime 集成**: 通过 `tvm.runtime.Executable` 加载和执行编译后的 kernel
2. **PyTorch 张量适配**: 将 PyTorch 张量转换为 TVM 可识别的数据指针
3. **动态形状解析**: 支持运行时动态形状的计算和输出张量分配
4. **CUDA Stream/设备管理**: 捕获当前 PyTorch CUDA 流和设备上下文

## 核心实现

### TVMFFIKernelAdapter 类

```python
class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors."""
```

**关键属性** (tvm_ffi.py:46-63):

| 属性 | 类型 | 说明 |
|------|------|------|
| `target` | str \| Target | 目标平台 (如 'cuda') |
| `ir_module` | tvm.IRModule | TVM IR 模块 |
| `host_kernel_source` | str | Host 端 kernel 源代码 |
| `device_kernel_source` | str | Device 端 kernel 源代码 |
| `executable` | tvm.runtime.Executable | TVM 可执行对象 |
| `dynamic_symbolic_map` | dict | 动态符号到 (类型, buffer索引, 维度) 的映射 |

### 初始化流程

**`__init__` 方法** (tvm_ffi.py:66-111):

```python
def __init__(
    self,
    params: list[KernelParam],
    result_idx: list[int],
    target: str | Target,
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    host_mod: tvm.IRModule | None = None,
    device_mod: tvm.IRModule | None = None,
    rt_mod: tvm.runtime.Module | None = None,
    host_kernel_source: str | None = None,
    device_kernel_source: str | None = None,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
):
```

初始化流程:
1. 存储 kernel 参数和输出索引
2. 转换 `tir.PrimFunc` 为 `tvm.IRModule`
3. 标准化目标平台
4. 处理动态符号映射 (`_process_dynamic_symbolic`)
5. 调用 `_post_init()` 完成最终初始化

### 动态符号处理

**`_process_dynamic_symbolic` 方法** (tvm_ffi.py:113-139):

该方法提取 TIR 函数中的动态形状信息，建立符号变量到运行时位置的映射:

```python
def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
    """Extract information about dynamic shapes from the TIR function.

    Maps symbolic variables to their corresponding (id, buffer_index, dimension)
    for runtime shape resolution.
    id represents shape or stride, 0 represents shape, 1 represents stride
    """
```

映射结构:
- `id=0`: 形状变量 (shape var) -> (0, buffer_param_index, dim_index)
- `id=1`: 步长变量 (stride var) -> (1, buffer_param_index, stride_index)
- `id=2`: 标量参数 (scalar param) -> (2, param_index, -1)

### Torch 函数转换

**`_convert_torch_func` 方法** (tvm_ffi.py:141-251):

这是核心的执行转换方法，生成一个 PyTorch 兼容的可调用函数:

```python
def _convert_torch_func(self) -> Callable[..., Any]:
    # Capture thunks that reflect Torch's current stream and device.
    current_device_functor = self.get_current_device_functor()

    # Convert TVM types to native Python types during initialization
    param_dtypes = [param.torch_dtype() for param in self.params]
    param_shapes = [...]  # 转换形状为原生 Python 类型

    def func(*inputs: torch.Tensor | Any):
        # 1. 验证输入数量
        # 2. 解析输出设备
        # 3. 准备输入/输出张量
        # 4. 处理动态形状
        # 5. 调用 executable
        # 6. 返回结果

    return func
```

执行流程:
1. **输入验证**: 检查输入张量数量是否匹配期望值 (tvm_ffi.py:196-199)
2. **输出张量分配**: 根据动态形状计算输出张量大小 (tvm_ffi.py:210-242)
3. **动态形状解析**: 从输入张量中提取动态维度值 (tvm_ffi.py:215-227)
4. **Kernel 执行**: 通过 `executable(*tensor_list)` 调用 TVM runtime (tvm_ffi.py:244)
5. **结果返回**: 根据 `result_idx` 返回单个或多个输出张量 (tvm_ffi.py:247-249)

### 从数据库加载

**`from_database` 类方法** (tvm_ffi.py:253-288):

支持从预编译的 kernel 库加载适配器，用于缓存场景:

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

该方法通过 `runtime.load_module(kernel_lib_path)` 直接加载预编译的模块。

## 源代码获取

### 获取 Host 源代码

```python
def get_host_source(self):
    """Returns the source code of the host module."""
    if self.host_kernel_source is not None:
        return self.host_kernel_source
    return self.rt_mod.inspect_source()
```

### 获取 Device 源代码

```python
def get_device_source(self):
    """Returns the source code of the device module."""
    if self.device_kernel_source is not None:
        return self.device_kernel_source
    return self.rt_mod.imports[0].inspect_source()
```

## 平台特定处理

### macOS 支持

```python
COMPILE_ARGS = {}

if sys.platform == "darwin":
    from torch.utils import cpp_extension
    COMPILE_ARGS["options"] = ["-x", "objective-c++", "-g", "-std=gnu++17"] + \
                              ["-I" + i for i in cpp_extension.include_paths()]
```

## 与 BaseKernelAdapter 的关系

TVMFFIKernelAdapter 继承自 `BaseKernelAdapter`，复用以下功能:

- `_legalize_result_idx`: 规范化输出索引 (base.py:20-40)
- `_post_init`: 初始化后处理，设置 `self.func` (base.py:95-96)
- `get_current_stream_functor`: 获取当前 CUDA 流 (base.py:47-66)
- `get_current_device_functor`: 获取当前设备 (base.py:68-84)

## 使用示例

```python
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter
from tilelang.engine.param import KernelParam

# 创建适配器
adapter = TVMFFIKernelAdapter(
    params=[KernelParam(...), ...],
    result_idx=[-1],  # 最后一个参数是输出
    target="cuda",
    func_or_mod=ir_module,
    verbose=True
)

# 执行 kernel
output = adapter(input_tensor1, input_tensor2)
```

## 总结

TVM FFI Adapter 是 TileLang JIT 系统的基础适配器，它:

1. **直接集成 TVM Runtime**: 利用 TVM 成熟的编译和执行基础设施
2. **无缝衔接 PyTorch**: 自动处理张量类型转换和 CUDA 上下文
3. **支持动态形状**: 在运行时解析和计算动态维度
4. **支持缓存加载**: 可以从预编译库快速恢复执行环境
