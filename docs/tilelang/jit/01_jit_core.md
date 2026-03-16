# TileLang JIT 编译系统深度解析

本文档详细分析 TileLang 项目的 JIT (Just-In-Time) 编译系统架构与实现。

## 1. JIT 架构概述

TileLang 的 JIT 编译系统采用分层架构设计，核心组件分布在以下模块：

```
tilelang/jit/
├── __init__.py          # JIT 入口，提供 jit 装饰器和 compile 函数
├── kernel.py            # JITKernel 类，内核编译与执行封装
├── execution_backend.py # 执行后端解析与验证
├── param.py             # 类型参数定义
├── exceptions.py        # 异常处理
└── adapter/             # 各类执行后端适配器
    ├── base.py          # BaseKernelAdapter 抽象基类
    ├── tvm_ffi.py       # TVM FFI 适配器
    ├── cython/          # Cython 后端
    ├── nvrtc/           # NVRTC 后端
    ├── torch/           # PyTorch/Metal 后端
    └── cutedsl/         # CuTe DSL 后端
```

### 1.1 核心数据流

```
用户代码 (Python)
    ↓
@tilelang.jit 装饰器 (JITImpl)
    ↓
PrimFunc (TIR) 生成
    ↓
cached() 缓存检查
    ↓
JITKernel 编译
    ↓
Adapter 创建与封装
    ↓
PyTorch 可调用函数
```

### 1.2 双模式执行架构

JIT 系统支持两种执行模式（在 `tilelang/jit/__init__.py:192-237` 中定义）：

**Lazy 模式**：
- 被装饰函数显式返回 PrimFunc
- 调用 JIT 包装器返回编译后的内核对象，可单独调用
- 适用于需要检查或重用内核对象的场景

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), ...):
        ...
    return kernel  # 显式返回 PrimFunc

kernel = matmul(1024, 1024, 1024, 128, 128, 32)  # 返回 kernel
result = kernel(a, b)  # 单独执行
```

**Eager 模式**：
- 使用 DSL builder 模式，通过张量类型注解定义计算
- 调用 JIT 包装器立即编译并执行内核，直接返回结果
- 适用于快速原型开发

```python
@tilelang.jit
def gemm(A, B, C, block_M: int = 64):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]  # 张量形状通过注解指定
    B: T.Tensor[[K, N], dtype]
    C: T.Tensor[[M, N], dtype]
    with T.Kernel(...):
        ...

gemm(A, B, C)  # 编译并立即执行
```

模式通过 `_infer_jit_mode` 方法（`tilelang/jit/__init__.py:304-317`）自动推断，也可通过 `mode` 参数显式指定。

## 2. 内核编译流程

### 2.1 编译入口

编译流程从 `compile()` 函数（`tilelang/jit/__init__.py:47-107`）开始：

```python
def compile(
    func: PrimFunc = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel
```

该函数将编译委托给 `cached()` 函数（`tilelang/cache/__init__.py:30-86`），实现缓存机制。

### 2.2 缓存机制

缓存系统由 `KernelCache` 类（`tilelang/cache/kernel_cache.py:27-545`）实现，采用三级缓存策略：

1. **内存缓存**：进程内字典缓存，避免重复编译
2. **磁盘缓存**：持久化到文件系统，跨进程共享
3. **数据库缓存**：支持从数据库加载预编译内核

缓存键生成（`tilelang/cache/kernel_cache.py:127-168`）：
```python
def _generate_key(
    self,
    func: Callable,
    out_idx: list[int],
    execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
    args=None,
    target: str | Target = "auto",
    target_host: str | Target = None,
    pass_configs: dict = None,
    compile_flags: list[str] | str | None = None,
) -> str:
    func_binary = func.script(show_meta=True).encode()
    key_data = {
        "func": sha256(func_binary).hexdigest(),  # TIR 脚本哈希
        "out_idx": tuple(out_idx),
        "args_repr": tuple(repr(arg) for arg in args),
        "target": str(target),
        "target_host": str(target_host),
        "execution_backend": execution_backend,
        "pass_configs": pass_configs,
        "compile_flags": compile_flags,
        **self._get_base_key(),  # 包含版本信息
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return sha256(key_string.encode()).hexdigest()
```

### 2.3 JITKernel 编译流程

`JITKernel` 类（`tilelang/jit/kernel.py:38-783`）是编译流程的核心，其初始化过程：

```python
def __init__(
    self,
    func: PrimFunc = None,
    out_idx: list[int] | int = None,
    execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
    target: str | Target = "auto",
    target_host: str | Target = None,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    from_database: bool = False,
    compile_flags: list[str] | None = None,
):
```

编译流程在 `_compile_and_create_adapter` 方法（`tilelang/jit/kernel.py:209-336`）中实现：

1. **目标平台确定**：通过 `determine_target()` 解析目标平台
2. **Pass 配置处理**：合并编译标志到 pass_configs
3. **TVM 编译**：调用 `tilelang.lower()` 进行代码生成
4. **适配器创建**：根据执行后端创建对应的 Adapter

```python
def _compile_and_create_adapter(self, tilelang_func: PrimFunc, out_idx: list[int]) -> BaseKernelAdapter:
    # 1. 确定编译参数
    verbose = self.verbose
    target = self.target
    target_host = self.target_host
    execution_backend = self.execution_backend
    pass_configs = dict(self.pass_configs) if self.pass_configs else {}
    compile_flags = self.compile_flags

    # 2. 配置代码生成选项
    enable_host_codegen = execution_backend == "tvm_ffi"
    enable_device_compile = execution_backend == "tvm_ffi"

    # 3. TVM 编译流程
    with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), self.target:
        artifact = tilelang.lower(
            tilelang_func,
            target=target,
            target_host=target_host,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
        )

    self.artifact = artifact

    # 4. 根据执行后端创建适配器
    if execution_backend == "tvm_ffi":
        adapter = TVMFFIKernelAdapter(...)
    elif execution_backend == "cython":
        adapter = CythonKernelAdapter(...)
    elif execution_backend == "nvrtc":
        adapter = NVRTCKernelAdapter(...)
    elif execution_backend == "torch":
        adapter = MetalKernelAdapter(...)
    elif execution_backend == "cutedsl":
        adapter = CuTeDSLKernelAdapter(...)

    return adapter
```

### 2.4 编译产物 (CompiledArtifact)

编译产物由 `CompiledArtifact` 数据类（`tilelang/engine/param.py:153-164`）表示：

```python
@dataclass
class CompiledArtifact:
    host_mod: tvm.IRModule      # Host 端 TVM IR 模块
    device_mod: tvm.IRModule    # Device 端 TVM IR 模块
    params: list[KernelParam]   # 内核参数列表
    kernel_source: str          # 生成的内核源代码
    rt_mod: tvm.runtime.Module | None = None  # 运行时模块
```

## 3. 执行后端机制

### 3.1 后端架构

执行后端系统采用适配器模式，基类 `BaseKernelAdapter`（`tilelang/jit/adapter/base.py:11-97`）定义统一接口：

```python
class BaseKernelAdapter(ABC):
    func: Callable | None = None

    def __init__(self, mod, params: list[KernelParam], result_idx: list[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self._post_init()

    @abstractmethod
    def _convert_torch_func(self) -> callable:
        """子类必须实现：转换为 PyTorch 可调用函数"""
        pass
```

### 3.2 后端类型与选择

支持的执行后端（`tilelang/jit/execution_backend.py:8-106`）：

| 后端 | 目标平台 | 特点 |
|------|----------|------|
| `tvm_ffi` | CUDA, Metal, CPU | 默认后端，使用 TVM FFI 和 DLPack |
| `cython` | CUDA, HIP, CPU | Cython 包装，低延迟调用 |
| `nvrtc` | CUDA | NVRTC 运行时编译 |
| `torch` | Metal | PyTorch Metal 后端 |
| `cutedsl` | CUDA | CuTe DSL 代码生成 |

后端解析逻辑（`tilelang/jit/execution_backend.py:66-106`）：

```python
def resolve_execution_backend(requested: str | None, target: Target) -> str:
    req = _canon_backend(requested)
    allowed_all = allowed_backends_for_target(target, include_unavailable=True)
    allowed_avail = allowed_backends_for_target(target, include_unavailable=False)

    # 自动选择默认后端
    if req in (None, "auto"):
        if is_cutedsl_target(target):
            return "cutedsl"
        kind = _target_kind(target)
        if kind == "cuda" or kind == "metal":
            choice = "tvm_ffi"
        else:
            choice = "cython"
        if choice not in allowed_avail and allowed_avail:
            choice = allowed_avail[0]
        return choice

    # 验证后端有效性
    if req not in allowed_all:
        raise ValueError(f"Invalid execution backend '{requested}' for target '{_target_kind(target)}'")

    return req
```

### 3.3 TVM FFI 适配器

`TVMFFIKernelAdapter`（`tilelang/jit/adapter/tvm_ffi.py:34-313`）是最常用的适配器，实现细节：

**动态形状处理**（`tilelang/jit/adapter/tvm_ffi.py:113-139`）：
```python
def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
    """提取动态形状信息，将符号变量映射到 (id, buffer_index, dimension)"""
    func = self.prim_func
    params = func.params
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}

    # 处理标量参数
    for i, param in enumerate(params):
        if isinstance(param, tir.Var) and (param not in dynamic_symbolic_map):
            dynamic_symbolic_map[param] = (2, i, -1)

    # 处理张量形状
    for i, param in enumerate(params):
        if param in buffer_map:
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var):
                    dynamic_symbolic_map[shape] = (0, i, j)

    # 处理张量步长
    for i, param in enumerate(params):
        if param in buffer_map:
            buffer = buffer_map[param]
            for j, stride in enumerate(buffer.strides):
                if isinstance(stride, tir.Var):
                    dynamic_symbolic_map[stride] = (1, i, j)
    return dynamic_symbolic_map
```

**PyTorch 函数转换**（`tilelang/jit/adapter/tvm_ffi.py:141-251`）：
```python
def _convert_torch_func(self) -> Callable[..., Any]:
    # 捕获当前流和设备
    current_device_functor = self.get_current_device_functor()

    # 转换参数类型
    param_dtypes = [param.torch_dtype() for param in self.params]
    param_shapes = [...]  # 处理形状

    # 创建可执行对象
    if self.executable is None:
        self.executable = runtime.Executable(self.rt_mod)

    dynamic_symbolic_map = self._process_dynamic_symbolic()
    executable = self.executable

    def func(*inputs: torch.Tensor | Any):
        # 验证输入数量
        expected_inputs = len(self.params) - len(self.result_idx)
        if len(inputs) != expected_inputs:
            raise ValueError(f"Kernel expected {expected_inputs} inputs")

        # 准备输入输出张量
        tensor_list = []
        for i in range(len(self.params)):
            if i in self.result_idx:
                # 创建输出张量
                dtype = param_dtypes[i]
                shape = [...]  # 解析动态形状
                tensor = torch.empty(*shape, dtype=dtype, device=out_device)
            else:
                tensor = inputs[ins_idx]
                ins_idx += 1
            tensor_list.append(tensor)

        # 执行内核
        executable(*tensor_list)

        # 返回结果
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        return [tensor_list[i] for i in self.result_idx]

    return func
```

### 3.4 流和设备同步

`BaseKernelAdapter` 提供 PyTorch 流/设备同步机制（`tilelang/jit/adapter/base.py:47-84`）：

```python
@staticmethod
def get_current_stream_functor() -> Callable[[], int]:
    """返回获取 PyTorch 当前 CUDA 流指针的可调用对象"""
    if torch.cuda.is_available():
        try:
            torch.cuda._lazy_init()
            current_device = torch._C._cuda_getDevice
            get_stream = torch._C._cuda_getCurrentRawStream
            return lambda: get_stream(current_device())
        except Exception:
            return lambda: int(torch.cuda.current_stream().cuda_stream)
    return lambda: 0  # CPU 回退

@staticmethod
def get_current_device_functor() -> Callable[[], torch.device]:
    """返回获取 PyTorch 当前设备的可调用对象"""
    if torch.cuda.is_available():
        try:
            torch.cuda._lazy_init()
            current_device = torch._C._cuda_getDevice
            return lambda: torch.device("cuda", current_device())
        except Exception:
            return lambda: torch.device("cuda", torch.cuda.current_device())
    return lambda: torch.device("cpu")
```

## 4. 参数处理逻辑

### 4.1 内核参数 (KernelParam)

`KernelParam` 类（`tilelang/engine/param.py:12-151`）封装内核参数信息：

```python
@dataclass
class KernelParam:
    dtype: tvm.DataType           # TVM 数据类型
    shape: list[int | Var]        # 形状维度（支持动态变量）

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        """从 TVM Buffer 创建参数"""
        dtype = buffer.dtype
        shape = []
        for s in buffer.shape:
            if isinstance(s, IntImm):
                shape.append(s.value)
            elif isinstance(s, (Var, PrimExpr)):
                shape.append(s)
        return cls(dtype, shape)

    @classmethod
    def from_var(cls, var: Var):
        """从 TVM 变量创建标量参数"""
        return cls(var.dtype, [])

    def torch_dtype(self) -> torch.dtype:
        """转换为 PyTorch 数据类型"""
        return T.dtype(self.dtype).as_torch()
```

### 4.2 输出索引处理

`result_idx`（即 `out_idx`）处理逻辑（`tilelang/jit/adapter/base.py:20-40`）：

```python
def _legalize_result_idx(self, result_idx: list[int] | None) -> list[int]:
    params = self.params
    if result_idx is None:
        result_idx = []
    elif isinstance(result_idx, int):
        # 支持负索引
        if result_idx < 0:
            result_idx = len(params) + result_idx
        result_idx = [result_idx]
    elif isinstance(result_idx, list):
        for i, idx in enumerate(result_idx):
            if idx < 0:
                result_idx[i] = len(params) + idx
    return result_idx
```

### 4.3 JITImpl 参数解析

`JITImpl` 类（`tilelang/jit/__init__.py:192-457`）处理装饰器参数：

```python
@dataclass
class JITImpl(Generic[_P, _KP, _T, _Ret]):
    out_idx: list[int] | int | None
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None
    target: str | Target | None
    target_host: str | Target | None
    verbose: bool | None
    pass_configs: dict[str, Any] | None
    debug_root_path: str | None
    compile_flags: list[str] | str | None
    func_source: str
    signature: inspect.Signature
    mode: Literal["auto", "lazy", "eager"]
    func: JITFunc[_KP, _T]
```

缓存键解析（`tilelang/jit/__init__.py:409-415`）：
```python
def parse_cache_key(self, *args: _P.args, **kwargs: _P.kwargs):
    tune_params = kwargs.pop("__tune_params", {})
    key_args_tuple = args
    key_kwargs_tuple = tuple(sorted(kwargs.items()))
    tuned_key_kwargs_tuple = tuple(sorted(tune_params.items()))
    key = (key_args_tuple, key_kwargs_tuple, tuned_key_kwargs_tuple)
    return key
```

## 5. 异常处理

JIT 系统定义了专门的异常类（`tilelang/jit/exceptions.py:1-24`）：

```python
class JITNoBuilderError(Exception):
    """
    当 JIT 操作需要 Builder 但不存在时抛出。
    在 eager 模式下，TileLang 直接构造 AST 而没有显式的 prim_func，
    因此必须有可用的 Builder。
    """
    pass


class EagerJITBuildError(Exception):
    """
    构建 TileLang eager JIT 内核时出错抛出。
    表示 eager 风格内核构造过程中出现问题。
    """
    pass
```

## 6. 并行编译

`par_compile()` 函数（`tilelang/jit/__init__.py:110-189`）支持多线程并行编译：

```python
def par_compile(
    funcs: Iterable[PrimFunc[_KP, _T]],
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", ...] | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    num_workers: int | None = None,
    ignore_error: bool = False,
) -> list[JITKernel[_KP, _T]]:
    with concurrent.futures.ThreadPoolExecutor(num_workers, "tl-par-comp") as executor:
        futures = []
        future_map = {}
        for i, func in enumerate(funcs):
            future = executor.submit(compile, func=func, ...)
            future_map[future] = i
            futures.append(future)

        results = [... for _ in futures]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx = future_map[future]
            if ignore_error:
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Error compiling function at index {idx}: {e}")
                    results[idx] = None
            else:
                results[idx] = future.result()
        return results
```

## 7. 环境变量配置

JIT 系统支持以下环境变量（在 `tilelang/cache/__init__.py:44-51` 和 `tilelang/env.py` 中处理）：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TILELANG_TARGET` | 默认编译目标 | "auto" |
| `TILELANG_EXECUTION_BACKEND` | 默认执行后端 | "auto" |
| `TILELANG_VERBOSE` | 启用详细输出 | False |
| `TILELANG_CACHE_DIR` | 缓存目录 | ~/.cache/tilelang |
| `TILELANG_CLEAR_CACHE` | 启动时清除缓存 | False |

## 8. 代码引用索引

### 8.1 核心文件

- `tilelang/jit/__init__.py:47` - `compile()` 函数入口
- `tilelang/jit/__init__.py:110` - `par_compile()` 并行编译
- `tilelang/jit/__init__.py:192` - `JITImpl` 类定义
- `tilelang/jit/__init__.py:480` - `jit()` 装饰器
- `tilelang/jit/kernel.py:38` - `JITKernel` 类定义
- `tilelang/jit/kernel.py:209` - `_compile_and_create_adapter()` 编译核心
- `tilelang/jit/execution_backend.py:26` - `allowed_backends_for_target()` 后端选择
- `tilelang/jit/execution_backend.py:66` - `resolve_execution_backend()` 后端解析
- `tilelang/jit/adapter/base.py:11` - `BaseKernelAdapter` 基类
- `tilelang/jit/adapter/tvm_ffi.py:34` - `TVMFFIKernelAdapter` 实现
- `tilelang/cache/__init__.py:30` - `cached()` 缓存入口
- `tilelang/cache/kernel_cache.py:27` - `KernelCache` 缓存实现
- `tilelang/engine/param.py:12` - `KernelParam` 参数类
- `tilelang/engine/param.py:153` - `CompiledArtifact` 编译产物

### 8.2 关键方法

- `tilelang/jit/__init__.py:304` - `_infer_jit_mode()` 模式推断
- `tilelang/jit/__init__.py:409` - `parse_cache_key()` 缓存键解析
- `tilelang/jit/adapter/base.py:20` - `_legalize_result_idx()` 输出索引规范化
- `tilelang/jit/adapter/base.py:47` - `get_current_stream_functor()` 流获取
- `tilelang/jit/adapter/tvm_ffi.py:113` - `_process_dynamic_symbolic()` 动态形状处理
- `tilelang/jit/adapter/tvm_ffi.py:141` - `_convert_torch_func()` PyTorch 函数转换
- `tilelang/cache/kernel_cache.py:127` - `_generate_key()` 缓存键生成
- `tilelang/cache/kernel_cache.py:170` - `cached()` 缓存查询

## 9. 总结

TileLang 的 JIT 编译系统是一个设计精良的分层架构：

1. **用户层**：通过 `@tilelang.jit` 装饰器提供简洁的 API
2. **编译层**：`JITKernel` 类协调 TVM 编译流程
3. **缓存层**：三级缓存机制优化编译性能
4. **适配层**：多种执行后端适配不同硬件平台
5. **运行时层**：通过 DLPack 实现与 PyTorch 的无缝集成

该系统支持动态形状、自动设备/流同步、并行编译等高级特性，为深度学习算子开发提供了高效的编译基础设施。
