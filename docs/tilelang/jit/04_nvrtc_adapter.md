# TileLang JIT NVRTC Adapter

## 概述

NVRTC Adapter 是 TileLang JIT 系统的运行时编译适配器，利用 NVIDIA Runtime Compilation (NVRTC) 库在运行时编译 CUDA C++ 代码。该适配器通过生成 Python 包装代码，实现零 C++ 依赖的 kernel 启动机制。

**源码文件**:
- `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/nvrtc/adapter.py`
- `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/nvrtc/wrapper.py`

## 架构设计

### 类层次结构

```
BaseKernelAdapter (base.py)
    └── NVRTCKernelAdapter (nvrtc/adapter.py)

TLCUDASourceWrapper (wrapper.py)
    └── TLNVRTCSourceWrapper (nvrtc/wrapper.py)
```

### 核心职责

1. **运行时编译**: 使用 NVRTC 在运行时编译 CUDA C++ 源代码
2. **Python 包装生成**: 生成纯 Python 的 kernel 启动代码
3. **TMA 描述符管理**: 通过 `cuda.bindings.driver` 初始化 TMA 描述符
4. **L2 缓存策略**: 支持持久化 L2 缓存配置
5. **零 C++ 依赖**: 完全使用 Python 和 CUDA Driver API

## NVRTC Kernel Adapter

### 类定义

**`NVRTCKernelAdapter`** (nvrtc/adapter.py:26):

```python
class NVRTCKernelAdapter(BaseKernelAdapter):
    """Adapter for NVRTC backend with runtime CUDA compilation."""
    pymodule = None
    kernels = {}
```

### 初始化流程

**`__init__` 方法** (nvrtc/adapter.py:30-96):

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
1. 检查 NVRTC 可用性 (`check_nvrtc_available()`)
2. 处理 IR 模块和参数
3. 缓存参数类型和形状信息
4. 处理动态符号映射
5. 创建 `TLPyWrapper` 包装器
6. 生成 host 函数和函数名
7. 创建 `NVRTCLibraryGenerator` 库生成器
8. 编译库并加载 kernel
9. 通过 `cuda.cuLibraryGetKernel` 获取 kernel 句柄

### Kernel 句柄获取

```python
# 从编译后的库中获取 kernel 句柄
culib = self.lib_generator.culib
for name in self.function_names:
    result, self.kernels[name] = cuda.cuLibraryGetKernel(culib, bytes(name, "utf-8"))
    assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to get kernel: {name}"
```

### 动态符号处理

**`_process_dynamic_symbolic` 方法** (nvrtc/adapter.py:157-182):

```python
def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
    """Extract information about dynamic shapes from the TIR function.

    Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
    for runtime shape resolution.
    """
```

### Kernel 执行

**`_forward_from_prebuild_lib` 方法** (nvrtc/adapter.py:208-210):

```python
def _forward_from_prebuild_lib(self, *args, stream: int | None = None):
    """Low-level function to call the compiled CUDA kernel."""
    return self.pymodule.call(self.kernels, *args, stream=stream)
```

**`_wrap_forward_from_prebuild_lib` 方法** (nvrtc/adapter.py:212-270):

处理:
1. 输入验证
2. 输出张量分配
3. 动态形状解析
4. CUDA 流管理

## NVRTC Source Wrapper

### 类定义

**`TLNVRTCSourceWrapper`** (nvrtc/wrapper.py:214):

```python
class TLNVRTCSourceWrapper(TLCUDASourceWrapper):
    """NVRTC backend wrapper: generates Python kernel launch code.

    Core responsibility: transform TVM IRModule into executable Python function
    that initializes resources (TMA descriptors, L2 cache) and launches kernels
    via CUDA Driver API.

    Why Python generation instead of C++:
        NVRTC workflow requires runtime compilation, Python is the natural host.
        Using cuda.bindings.driver eliminates C++ wrapper complexity.
    """
```

### Python 类型映射

**`_TYPE_MAP`** (nvrtc/wrapper.py:230-247):

```python
_TYPE_MAP: ClassVar[dict[str, str]] = {
    "float32": "ctypes.c_float",
    "float16": "ctypes.c_uint16",
    "bfloat16": "ctypes.c_uint16",
    "float8_e4m3": "ctypes.c_uint8",
    "float64": "ctypes.c_double",
    "int64": "ctypes.c_int64",
    "int32": "ctypes.c_int32",
    "bool": "ctypes.c_bool",
    "int8": "ctypes.c_int8",
    # ...
}
```

### 核心模板

#### Host 函数模板

**`PREDEF_HOST_FUNC_PY`** (nvrtc/wrapper.py:27-54):

```python
PREDEF_HOST_FUNC_PY = """
from cuda.bindings.driver import (
    CUtensorMapDataType, CUtensorMapInterleave, ...
    cuLaunchKernelEx, ...
)
import ctypes

_function_names = {}

def call({}):
    {}
"""
```

#### TMA 描述符初始化

**`TMA_DESC_INIT_FUNC_PY`** (nvrtc/wrapper.py:56-85):

```python
TMA_DESC_INIT_FUNC_PY = """
    {0}_type = CUtensorMapDataType({1})
    {0}_tensorRank = {2}
    {0}_globalAddress = {3}.data_ptr()
    {0}_globalDim = [{4}]
    {0}_globalStride = [{5}][1:]
    {0}_boxDim = [{6}]
    ...
    res, {0} = cuTensorMapEncodeTiled(...)
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to initialize the TMA descriptor {0}: {{res}}")
"""
```

#### Kernel 启动

**`KERNEL_LAUNCH_FUNC_PY`** (nvrtc/wrapper.py:184-211):

```python
KERNEL_LAUNCH_FUNC_PY = """
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        {7}, kernels["{0}"], CUdevice({10})
    )[0]

    config = CUlaunchConfig()
    config.gridDimX = {1}
    config.gridDimY = {2}
    config.gridDimZ = {3}
    config.blockDimX = {4}
    config.blockDimY = {5}
    config.blockDimZ = {6}
    config.sharedMemBytes = {7}
    config.hStream = stream

    arg_values = {8}
    arg_types = {9}

    res = cuLaunchKernelEx(config, kernels["{0}"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel {0}: {{res}}")
"""
```

### 两阶段代码生成

**`create_dispatch_func` 方法** (nvrtc/wrapper.py:291-443):

采用两阶段设计避免 TMA 描述符重复创建:

```python
def create_dispatch_func(self, code, function_informations):
    """Generate Python dispatch function that launches multiple CUDA kernels.

    Why two-pass design:
        Pass 1: Collect TMA descriptors from all kernels into shared dicts
        Pass 2: Generate code - descriptors first (deduplicated), then launches

        Single-pass would create duplicate descriptors for each kernel.
        Dict naturally deduplicates by descriptor name.
    """
```

**第一阶段**: 收集所有 kernel 的 TMA 描述符
```python
# First pass: collect all TMA descriptors from all kernels to avoid duplication
kernel_info_list = []
for function_name, function_info in function_informations.items():
    # 解析函数声明
    # 提取调用参数
    # 存储 kernel 信息
    kernel_info_list.append({...})
```

**第二阶段**: 生成代码
```python
# Generate TMA descriptor initialization code once for all kernels
kernel_launch_code += self.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)

# Second pass: generate kernel launch code for each kernel
for kernel_info in kernel_info_list:
    # 生成 L2 持久化映射初始化
    # 生成 PDL 同步代码
    # 生成 kernel 启动代码
```

### L2 持久化缓存

**`L2_PERSISTENT_MAP_CREATE_HANDLE_PY`** (nvrtc/wrapper.py:124-140):

```python
L2_PERSISTENT_MAP_CREATE_HANDLE_PY = """
    from cuda.bindings.driver import (
        CUstreamAttrValue, CUstreamAttrID, CUlimit, CUaccessProperty,
        cuCtxGetLimit, cuCtxSetLimit, cuStreamSetAttribute, cuCtxResetPersistingL2Cache,
    )

    stream_attribute = CUstreamAttrValue()
    res, init_persisting_l2_cache_size = cuCtxGetLimit(CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE)
"""
```

**`generate_l2_persistent_map` 方法** (nvrtc/wrapper.py:445-473):

```python
def generate_l2_persistent_map(self, function_name: str) -> str:
    """Generate Python code to configure L2 cache persistence for a kernel.

    L2 persistence pins frequently-accessed data in L2 cache to reduce
    memory bandwidth. Requires explicit setup via CUDA stream attributes.
    """
```

### PDL 同步

**`PDL_SYNC_PY`** (nvrtc/wrapper.py:174-182):

```python
PDL_SYNC_PY = """
    num_attrs = 1
    attrs = [CUlaunchAttribute()]
    attrs[0].id = CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attrs[0].value.programmaticStreamSerializationAllowed = 1

    config.numAttrs = num_attrs
    config.attrs = attrs
"""
```

### 参数转换

**`transform_nvrtc_arg`** (nvrtc/wrapper.py:373-376):

```python
def transform_nvrtc_arg(name: str, arg_type: str):
    if arg_type == "ctypes.c_void_p":
        return (f"{name}.data_ptr()", arg_type)
    return (name, arg_type)
```

将张量参数转换为 `data_ptr()` 调用，获取底层内存指针。

## 数据流

```
TVM IRModule
    ↓
TLNVRTCSourceWrapper
    ↓
收集 kernel 元数据 (grid/block, smem, params)
    ↓
两阶段代码生成
    ├── 收集 TMA 描述符
    └── 生成启动代码
    ↓
Python 包装代码
    ↓
NVRTC 编译
    ↓
cuLibraryGetKernel 获取句柄
    ↓
cuLaunchKernelEx 执行
```

## 使用示例

```python
from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
from tilelang.engine.param import KernelParam

# 创建适配器
adapter = NVRTCKernelAdapter(
    params=[KernelParam(...), ...],
    result_idx=[-1],
    target="cuda",
    func_or_mod=ir_module,
    device_kernel_source=cuda_source_code,
    verbose=True
)

# 执行 kernel
output = adapter(input_tensor1, input_tensor2)
```

## 与 CuTeDSL Adapter 对比

| 特性 | NVRTC Adapter | CuTeDSL Adapter |
|------|---------------|-----------------|
| 编译时机 | 运行时 | 运行时 |
| 包装代码 | Python | C++ + Python |
| TMA 处理 | `cuda.bindings.driver` | C++ Driver API |
| L2 缓存 | 支持 | 支持 |
| 依赖 | `cuda.bindings` | CuTeDSL 库 |
| 启动开销 | 中等 | 低 |

## 总结

NVRTC Adapter 是 TileLang 的轻量级运行时编译适配器，它:

1. **纯 Python 包装**: 使用 `cuda.bindings.driver` 避免 C++ 编译
2. **两阶段生成**: 有效避免 TMA 描述符重复创建
3. **完整 CUDA 特性**: 支持 L2 持久化、PDL 同步等高级特性
4. **零 C++ 依赖**: 简化部署，降低环境要求
5. **运行时灵活**: 支持动态代码生成和编译
