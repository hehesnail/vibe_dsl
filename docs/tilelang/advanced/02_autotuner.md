# TileLang AutoTuner 模块详解

## 模块概述

AutoTuner 是 TileLang 的自动调优模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/` 目录。该模块通过搜索不同的编译配置来找到最优的内核性能参数。

目录结构：
```
tilelang/autotuner/
├── __init__.py      # 模块导出
├── tuner.py         # 核心调优逻辑
├── param.py         # 参数定义与缓存
└── capture.py       # 输入捕获工具
```

核心功能：
- 配置空间搜索
- 并行编译与评测
- 结果缓存（内存+磁盘）
- 参考程序正确性验证

## 核心类详解

### 1. CompileArgs

```python
@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner."""
    out_idx: list[int] | int | None = None
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"] = "auto"
    target: Literal["auto", "cuda", "hip"] = "auto"
    target_host: str | Target = None
    verbose: bool = False
    pass_configs: dict[str, Any] | None = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py:39-67`

编译参数数据类，支持哈希用于缓存键生成。

### 2. ProfileArgs

```python
@dataclass(frozen=True)
class ProfileArgs:
    """Profile arguments for the auto-tuner."""
    warmup: int = 25
    rep: int = 100
    timeout: int = 30
    backend: Literal["event", "cupti", "cudagraph"] = "event"
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py:82-133`

性能分析参数，控制评测方式和正确性验证标准。

### 3. AutotuneResult

```python
@dataclass(frozen=True)
class AutotuneResult:
    """Results from auto-tuning process."""
    latency: float | None = None
    config: dict | None = None
    ref_latency: float | None = None
    libcode: str | None = None
    func: Callable | None = None
    kernel: Callable | None = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py:135-478`

调优结果数据类，包含：
- 最优延迟 (`latency`)
- 最优配置 (`config`)
- 参考延迟 (`ref_latency`)
- 生成的内核代码 (`libcode`)
- 编译后的函数和内核

**磁盘缓存方法**:
- `save_to_disk()`: 保存结果到磁盘
- `load_from_disk()`: 从磁盘加载结果
- `_save_kernel_to_disk()`: 保存内核文件
- `_load_kernel_from_disk()`: 加载内核文件

### 4. AutoTuner

```python
class AutoTuner:
    """Auto-tuner for tilelang programs."""
    compile_args = CompileArgs()
    profile_args = ProfileArgs()
    _memory_cache = {}  # In-memory cache dictionary
    cache_dir: Path = Path(env.TILELANG_CACHE_DIR) / "autotuner"
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:117-638`

核心调优类，主要方法：

#### set_compile_args()

```python
def set_compile_args(
    self,
    out_idx: list[int] | int | None = None,
    target: Literal["auto", "cuda", "hip", "metal"] | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"] | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:158-210`

设置编译参数，支持环境变量默认值：
- `TILELANG_TARGET`: 默认编译目标
- `TILELANG_EXECUTION_BACKEND`: 默认执行后端
- `TILELANG_VERBOSE`: 是否启用详细输出

#### set_profile_args()

```python
def set_profile_args(
    self,
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 30,
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable = None,
    supply_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable = None,
    cache_input_tensors: bool = False,
    backend: Literal["event", "cupti", "cudagraph"] = "event",
)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:212-275`

设置性能分析参数。

#### run()

```python
def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30) -> AutotuneResult:
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:315-629`

执行自动调优的核心方法，流程：

1. **缓存检查**: 先检查内存缓存，再检查磁盘缓存
2. **配置生成**: 处理配置列表，提取函数参数
3. **并行编译**: 使用 `ThreadPoolExecutor` 并发编译配置
4. **性能评测**: 对每个编译结果进行基准测试
5. **结果保存**: 保存最优结果到缓存

### 5. AutoTuneImpl

```python
@dataclass
class AutoTuneImpl(Generic[_P, _T]):
    jit_impl: JITImpl
    warmup: int = 25
    rep: int = 100
    timeout: int = 100
    configs: dict | Callable = None
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:644-737`

JIT 实现的自动调优包装器，作为装饰器使用。

### 6. CaptureStack / AutotuneInputsCapture

```python
class CaptureStack:
    """Thread-local stack for capturing auto-tune inputs."""
    def push(self, item): ...
    def pop(self): ...
    def top(self): ...

class AutotuneInputsCapture:
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/capture.py:10-127`

输入捕获机制，支持上下文管理器：

```python
with set_autotune_inputs(a, b, c):
    result = kernel(...)
```

## 实现逻辑分析

### 调优流程

```
┌─────────────────┐
│   检查缓存       │
│ (内存 → 磁盘)   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  生成配置列表    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  并行编译配置    │
│ (ThreadPool)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  性能评测        │
│ (带超时控制)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  选择最优配置    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  保存结果到缓存  │
└─────────────────┘
```

### 并行编译策略

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:536-571`

```python
# 计算工作进程数
available_cpu_count = get_available_cpu_count()
cpu_utilizations = float(env.TILELANG_AUTO_TUNING_CPU_UTILITIES)
cpu_counts = int(env.TILELANG_AUTO_TUNING_CPU_COUNTS)
max_cpu_count = int(env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT)

# 创建线程池
pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

# 提交编译任务
for i, config_arg in enumerate(config_args):
    future = pool.submit(compile_func, **config_arg)
    futures.append(future)
```

### 缓存机制

**内存缓存**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:134`
```python
_memory_cache = {}  # 全局内存缓存字典
```

**磁盘缓存结构**:
```
TILELANG_CACHE_DIR/autotuner/
└── {hash_key}/
    ├── best_config.json      # 最优配置
    ├── function.pkl          # 序列化函数
    ├── out_idx.json          # 输出索引
    ├── latency.json          # 延迟数据
    ├── device_kernel.cu      # 设备内核代码
    ├── host_kernel.cu        # 主机内核代码
    ├── kernel_lib.so         # 内核库
    └── params.pkl            # 内核参数
```

### 超时控制

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py:46-63`

```python
def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)
    return result
```

## 使用示例

### 装饰器方式

```python
import tilelang as tl
from tilelang.autotuner import autotune

@tl.jit
@autotune(
    configs=[
        {"block_M": 128, "block_N": 128, "num_stages": 2},
        {"block_M": 256, "block_N": 128, "num_stages": 3},
        {"block_M": 128, "block_N": 256, "num_stages": 2},
    ],
    warmup=25,
    rep=100,
    ref_prog=reference_gemm,  # 参考实现用于正确性验证
)
def gemm_kernel(A, B, C, block_M, block_N, num_stages):
    # 内核实现
    ...

# 使用调优后的内核
kernel = gemm_kernel(A, B, C)
```

### 程序化方式

```python
from tilelang.autotuner import AutoTuner

def kernel_func(block_M, block_N, num_stages):
    # 返回 PrimFunc
    return prim_func

tuner = AutoTuner(kernel_func, configs=[
    {"block_M": 128, "block_N": 128},
    {"block_M": 256, "block_N": 128},
])

result = tuner.set_compile_args(
    out_idx=[2],
    target="cuda",
).set_profile_args(
    ref_prog=reference_func,
    rtol=1e-2,
).run()

print(f"Best latency: {result.latency}")
print(f"Best config: {result.config}")
```

### 输入捕获

```python
from tilelang.autotuner import set_autotune_inputs

# 方式1：直接传递张量
with set_autotune_inputs(input_a, input_b):
    kernel = gemm_kernel(input_a, input_b)

# 方式2：传递列表
with set_autotune_inputs([input_a, input_b]):
    kernel = gemm_kernel(input_a, input_b)
```

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| CompileArgs 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py` | 39-80 |
| ProfileArgs 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py` | 82-133 |
| AutotuneResult 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/param.py` | 135-478 |
| AutoTuner 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py` | 117-638 |
| AutoTuneImpl 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py` | 644-737 |
| autotune 装饰器 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/tuner.py` | 739-834 |
| CaptureStack 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/capture.py` | 10-79 |
| set_autotune_inputs | `/root/dev/vibe_dsl/tilelang/tilelang/autotuner/capture.py` | 100-118 |
