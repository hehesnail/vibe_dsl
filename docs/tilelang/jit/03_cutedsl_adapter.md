# TileLang JIT CuTeDSL Adapter

## 概述

CuTeDSL Adapter 是 TileLang JIT 系统的高级适配器，专为 NVIDIA CuTeDSL (CUDA Templates for Deep Learning) 后端设计。该适配器通过生成 C++ launcher 代码和 Python 包装器，实现高性能 CUDA kernel 的编译和执行。

**源码文件**:
- `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/cutedsl/adapter.py`
- `/root/dev/vibe_dsl/tilelang/tilelang/jit/adapter/cutedsl/wrapper.py`

## 架构设计

### 类层次结构

```
BaseKernelAdapter (base.py)
    └── CuTeDSLKernelAdapter (cutedsl/adapter.py)

TLCUDASourceWrapper (wrapper.py)
    └── TLCuTeDSLSourceWrapper (cutedsl/wrapper.py)
```

### 核心职责

1. **CuTeDSL 集成**: 与 CuTeDSL 编译器交互，生成 cubin 文件
2. **C++ Launcher 生成**: 自动生成高效的 C++ kernel 启动代码
3. **TMA 描述符管理**: 处理 Tensor Memory Accelerator 描述符的初始化和传递
4. **多 GPU 支持**: 通过 CUDA Driver API 管理多设备上下文
5. **缓存集成**: 支持 cubin 文件的生成、存储和重用

## CuTeDSL Kernel Adapter

### 类定义

**`CuTeDSLKernelAdapter`** (cutedsl/adapter.py:22):

```python
class CuTeDSLKernelAdapter(BaseKernelAdapter):
    """Adapter for CuTeDSL backend with C++ launcher generation."""
    pymodule = None
```

### 初始化流程

**`__init__` 方法** (cutedsl/adapter.py:25-98):

```python
def __init__(
    self,
    params: list[KernelParam],
    result_idx: list[int],
    target: str | Target,
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    host_mod: tvm.IRModule | None = None,
    device_mod: tvm.IRModule | None = None,
    host_kernel_source: str | None = None,
    device_kernel_source: str | None = None,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | None = None,
):
```

初始化步骤:
1. 检查 CuTeDSL 可用性 (`check_cutedsl_available()`)
2. 处理 IR 模块和参数
3. 创建 `TLPyWrapper` 包装器
4. 调用 `wrapper.wrap()` 生成 host 函数和函数名
5. 创建 `CuTeDSLLibraryGenerator` 库生成器
6. 编译和加载库

### 动态符号处理

**`_process_dynamic_symbolic` 方法** (cutedsl/adapter.py:157-208):

```python
def _process_dynamic_symbolic(self) -> tuple[dict[tir.Var, tuple[int, int, int]], list[tir.Var]]:
    """Extract information about dynamic symbols from the TIR function.

    Returns:
        (dynamic_symbolic_map, dynamic_symbolic_order)

    Mapping encodes:
        - id=0: shape var -> (0, buffer_param_index, dim_index)
        - id=1: stride var -> (1, buffer_param_index, stride_index)
    """
```

该方法返回两个值:
- `dynamic_symbolic_map`: 符号变量到位置信息的映射
- `dynamic_symbolic_order`: 符号变量的有序列表，确保参数顺序一致

### Kernel 执行

**`_forward_from_prebuild_lib` 方法** (cutedsl/adapter.py:233-246):

```python
def _forward_from_prebuild_lib(self, *args, stream: int | None = None, device_id: int = 0):
    """Low-level function to call the compiled CUDA kernel."""
    result = self.pymodule.call(*args, stream=stream, device_id=device_id)
    self._save_cubin_to_cache_if_needed()
    return result
```

**`_wrap_forward_from_prebuild_lib` 方法** (cutedsl/adapter.py:288-381):

这是高层包装方法，处理:
1. 输入验证
2. 输出张量分配
3. 动态形状解析 (支持 shape 和 stride 变量)
4. CUDA 流管理
5. 多 GPU 设备 ID 提取

### 缓存支持

**`_save_cubin_to_cache_if_needed` 方法** (cutedsl/adapter.py:248-286):

```python
def _save_cubin_to_cache_if_needed(self):
    """Save cubin to cache directory after first execution."""
    if getattr(self, "_cubin_saved_to_cache", False):
        return
    # Copy cubin from temp directory to cache directory
```

### 资源清理

**`cleanup` 方法** (cutedsl/adapter.py:416-424):

```python
def cleanup(self):
    """Explicitly cleanup this adapter's CUDA resources."""
    self._cleanup_module(self.pymodule)
```

通过 `weakref.finalize` 在对象回收时自动清理 CUDA 模块和上下文。

## CuTeDSL Source Wrapper

### 类定义

**`TLCuTeDSLSourceWrapper`** (cutedsl/wrapper.py:672):

```python
class TLCuTeDSLSourceWrapper(TLCUDASourceWrapper):
    """Wrapper class for TileLang CuTe DSL backend with C++ launcher.

    Generates optimized C++ launcher code that:
    - Loads cubin via CUDA Driver API
    - Passes TMA descriptors by value (host-side, no device copy)
    - Launches kernels with minimal Python overhead
    - Supports both single and multiple kernel scenarios
    """
```

### C++ Launcher 模板

#### TMA 描述符初始化

**`CPP_TMA_DESC_INIT_TEMPLATE`** (cutedsl/wrapper.py:34-62):

```cpp
CUresult tma_init(CUtensorMap* tma_descs, {func_args}) {
  // Initialize {num_descs} TMA descriptor(s)
  CUresult result;

  // Descriptor {desc_idx}: {desc_name}
  uint64_t globalDim[{rank}] = {{...}};
  uint64_t globalStrides[{stride_rank}] = {{...}};
  uint32_t boxDim[{rank}] = {{...}};

  result = cuTensorMapEncodeTiled(
      &tma_descs[{desc_idx}], ...
  );
  // ...
}
```

#### Kernel 启动

**`CPP_KERNEL_LAUNCH_TEMPLATE`** (cutedsl/wrapper.py:152-178):

```cpp
// Launch kernel {kernel_idx}: {kernel_name}
{
    auto kernels_it = g_device_kernels.find(device_id);
    const std::vector<CUfunction>& kernels = kernels_it->second;

    void* args[] = {{...}};
    result = cuLaunchKernel(
        kernels[{kernel_idx}],
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        smem_size, stream, args, nullptr
    );
}
```

### 代码生成流程

#### 1. 收集函数参数

**`_collect_function_args` 方法** (cutedsl/wrapper.py:792-819):

```python
def _collect_function_args(self) -> tuple[list[dict], list[str]]:
    """Collect all function arguments from primary function."""
    function_args = []
    buffer_args = []

    for param in self.prim_func.params:
        if param in self.prim_func.buffer_map:
            buffer = self.prim_func.buffer_map[param]
            function_args.append({"name": buffer.data.name, "type": "buffer"})
            buffer_args.append(buffer.data.name)
        elif isinstance(param, tvm.tir.Var):
            function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})

    # Add dynamic symbols
    for dyn_sym in self.get_dynamic_symbolic_set(self.prim_func):
        function_args.append({"name": dyn_sym_name, "type": ...})

    return function_args, buffer_args
```

#### 2. 生成 C++ Launcher

**`_generate_cpp_launcher` 方法** (cutedsl/wrapper.py:1096-1149):

生成完整的 C++ launcher 代码，包括:
- 多设备模块和 kernel 存储
- TMA 描述符初始化函数
- Kernel 初始化代码
- Kernel 启动代码

#### 3. 生成 Python 包装器

**`_generate_python_wrapper` 方法** (cutedsl/wrapper.py:1252-1271):

生成 Python 端的包装代码，包括:
- Cubin 生成逻辑 (首次调用时)
- C++ launcher 加载
- Kernel 调用入口

### 多 GPU 支持

CuTeDSL wrapper 通过以下机制支持多 GPU:

1. **设备级模块存储** (cutedsl/wrapper.py:228-230):
```cpp
static std::unordered_map<int, CUmodule> g_device_modules;
static std::unordered_map<int, std::vector<CUfunction>> g_device_kernels;
static std::unordered_map<int, CUcontext> g_device_contexts;
```

2. **设备上下文管理** (cutedsl/wrapper.py:329-399):
```cpp
static CUresult tilelang_init_cuda_module(const std::string& cubin_path, int device_id) {
    // 1. 获取设备句柄
    // 2. Retain primary context
    // 3. 设置当前上下文
    // 4. 加载 cubin 模块
}
```

3. **Kernel 查找** (cutedsl/wrapper.py:234-323):
```cpp
CUresult find_kernel_by_pattern(CUmodule module, const char* pattern, CUfunction* out_func)
```

支持通过模式匹配查找 kernel，优先选择基础名称而非 `_N` 变体。

### TMA 描述符处理

**`_generate_tma_desc_init` 方法** (cutedsl/wrapper.py:934-975):

生成单个 TMA 描述符的初始化代码，支持:
- 标准 Tiled 描述符 (`cuTensorMapEncodeTiled`)
- Im2col 描述符 (`cuTensorMapEncodeIm2col`)

**关键特性**:
- TMA 描述符存储在 host 内存的栈数组中
- 通过 `__grid_constant__` 传递给 kernel
- `cuLaunchKernel` 自动将 128 字节的 `CUtensorMap` 复制到 kernel 参数空间

### 协作组支持

**`CPP_COOPERATIVE_KERNEL_LAUNCH_TEMPLATE`** (cutedsl/wrapper.py:182-207):

```cpp
result = cuLaunchCooperativeKernel(
    kernels[{kernel_idx}],
    grid_x, grid_y, grid_z,
    block_x, block_y, block_z,
    smem_size, stream, args
);
```

当 `use_cooperative_groups` 属性为 True 时使用，支持 grid 级同步。

## 数据类型映射

### Python/CuTeDSL 类型映射

| TVM 类型 | CuTeDSL 类型 |
|----------|--------------|
| float32 | cutlass.Float32 |
| float16 | cutlass.Float16 |
| bfloat16 | cutlass.BFloat16 |
| int32 | cutlass.Int32 |
| int8 | cutlass.Int8 |
| bool | cutlass.Uint8 |

### C++ 类型映射

| CuTeDSL 类型 | C++ 类型 |
|--------------|----------|
| cutlass.Float32 | float |
| cutlass.Int32 | int32_t |
| cutlass.Int64 | int64_t |
| cutlass.Uint8 | uint8_t |

## 使用示例

```python
from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
from tilelang.engine.param import KernelParam

# 创建适配器
adapter = CuTeDSLKernelAdapter(
    params=[KernelParam(...), ...],
    result_idx=[-1],
    target="cuda",
    func_or_mod=ir_module,
    device_kernel_source=cuda_source,
    verbose=True
)

# 执行 kernel (自动处理设备选择)
output = adapter(input_tensor1, input_tensor2)

# 显式清理资源
adapter.cleanup()
```

## 总结

CuTeDSL Adapter 是 TileLang 的高性能后端适配器，它:

1. **自动生成 C++ Launcher**: 消除 Python 调用开销
2. **原生 TMA 支持**: 高效处理 Tensor Memory Accelerator 描述符
3. **多 GPU 就绪**: 通过 CUDA Driver API 管理多设备上下文
4. **缓存友好**: 支持 cubin 文件的生成和重用
5. **协作组支持**: 支持 grid 级同步操作
