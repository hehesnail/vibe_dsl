# TileLang Profiler 模块详解

## 模块概述

Profiler 是 TileLang 的性能分析和基准测试模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/profiler/` 目录。该模块提供了 GPU 内核的精确计时和正确性验证功能。

目录结构：
```
tilelang/profiler/
├── __init__.py      # 模块导出，Profiler 类定义
└── bench.py         # 底层基准测试实现
```

核心功能：
- 高精度 GPU 内核计时（CUDA Events / CUPTI / CUDA Graph）
- L2 缓存管理确保一致测量
- 正确性验证（与参考实现比较）
- 支持多种张量供应方式

## 核心类详解

### 1. Profiler

```python
@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations."""
    params: list[KernelParam]
    result_idx: list[int]
    supply_type: TensorSupplyType
    adapter: BaseKernelAdapter | None = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py:20-279`

主要方法：

#### _get_inputs()

```python
def _get_inputs(self, with_output=False, dynamic_symbolic_constraints: dict[str, int] | None = None):
    """Generate input tensors based on supply type."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py:61-69`

根据 `supply_type` 生成输入张量，支持动态形状约束。

#### assert_allclose()

```python
def assert_allclose(
    self,
    reference_program: Callable,
    input_tensors: list[torch.Tensor] | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    max_mismatched_ratio=0.01,
):
    """Validates kernel output against a reference implementation."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py:103-163`

验证内核输出与参考实现的一致性：
- 生成或使用提供的输入张量
- 执行参考程序和 TileLang 内核
- 使用 `torch_assert_close` 比较结果
- 支持 Float8 类型转换

#### assert_consistent()

```python
def assert_consistent(self, repeat=10):
    """Checks for kernel consistency across multiple runs."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py:196-213`

检查内核多次执行的一致性，用于检测竞态条件。

#### do_bench()

```python
def do_bench(
    self,
    func: Callable | None = None,
    warmup: int = 25,
    rep: int = 100,
    n_warmup: int = 0,
    n_repeat: int = 0,
    input_tensors: list[torch.Tensor] = None,
    backend: Literal["event", "cupti", "cudagraph"] = "event",
    quantiles: list[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    dynamic_symbolic_constraints: dict[str, int] | None = None,
) -> float:
    """Benchmarks the execution time of a given function."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py:221-270`

执行基准测试，支持动态形状约束。

### 2. do_bench (底层函数)

```python
def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti", "cudagraph"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> float | list[float]:
    """Benchmark the runtime of a PyTorch function with L2 cache management."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:64-138`

底层基准测试函数，提供三种后端支持。

## 实现逻辑分析

### 基准测试流程

```
┌─────────────────────┐
│  初始函数调用 + 同步  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  创建 L2 缓存刷新缓冲区│
│  (256MB int32/int8) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  估算内核运行时间    │
│  (5 次迭代平均)     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  计算 warmup/repeat │
│  迭代次数            │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Warmup 阶段        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  基准测试阶段        │
│  (根据 backend 选择) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  结果聚合            │
│  (mean/median/quantiles)
└─────────────────────┘
```

### L2 缓存刷新

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:103-107`

```python
# Create L2 cache flush buffer (256 MB)
# Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
cache_size = int(256e6 // 4) if fast_flush else int(256e6)
cache_dtype = torch.int if fast_flush else torch.int8
cache = torch.empty(cache_size, dtype=cache_dtype, device="cuda")

# 在每次迭代前刷新缓存
cache.zero_()
```

### 后端实现

#### 1. CUDA Events 后端

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:140-173`

```python
def _bench_with_cuda_events(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    for i in range(n_repeat):
        cache.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
```

使用 CUDA 事件进行精确计时，是最常用的后端。

#### 2. CUPTI 后端

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:175-207`

```python
def _bench_with_cupti(fn: Callable, cache: torch.Tensor, n_repeat: int) -> float:
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    )

    with profiler:
        for _ in range(2):
            for _ in range(n_repeat):
                cache.zero_()
                fn()
            profiler.step()

    # 计算平均内核时间，排除缓存清除开销
    total_cuda_time = 0.0
    excluded_time = 0.0
    for event in profiler.key_averages():
        total_cuda_time += event.self_device_time_total
```

使用 PyTorch Profiler 获取详细的内核级计时信息。

#### 3. CUDA Graph 后端

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:209-258`

```python
def _bench_with_cudagraph(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    with torch.cuda.stream(torch.cuda.Stream()):
        # 构建包含 n_repeat 次展开的 CUDA Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()

    # 多次重放 Graph 测量时间
    for i in range(n_retries):
        cache.zero_()
        start_events[i].record()
        g.replay()
        end_events[i].record()
```

使用 CUDA Graph 消除主机端开销，提供最精确的测量。

### 抑制 stdout/stderr

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py:12-57`

```python
class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr output."""
    def __enter__(self):
        # 保存原始文件描述符
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        # 重定向到 null 设备
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
```

用于在 CUPTI 分析时抑制输出。

## 使用示例

### 基本基准测试

```python
import tilelang as tl
from tilelang.profiler import do_bench

# 定义测试函数
def benchmark_func():
    # 执行内核
    kernel(input_a, input_b, output)

# 使用 CUDA Events 后端
latency = do_bench(
    benchmark_func,
    warmup=25,      # 25ms warmup
    rep=100,        # 100ms 测量
    backend="event",
    return_mode="mean"
)
print(f"Average latency: {latency} ms")

# 获取中位数和 95% 分位数
quantiles = do_bench(
    benchmark_func,
    warmup=25,
    rep=100,
    quantiles=[0.5, 0.95],
    backend="event"
)
```

### 使用 Profiler 类

```python
from tilelang.profiler import Profiler
from tilelang.engine.param import KernelParam

# 定义内核参数
params = [
    KernelParam(dtype="float16", shape=[1024, 1024]),
    KernelParam(dtype="float16", shape=[1024, 1024]),
    KernelParam(dtype="float32", shape=[1024, 1024]),
]

# 创建 Profiler
profiler = Profiler(
    params=params,
    result_idx=[2],
    supply_type=tl.TensorSupplyType.Random,
)

# 附加适配器
profiler.with_default_adapter(kernel_adapter)

# 正确性验证
def reference_gemm(a, b):
    return torch.matmul(a, b)

profiler.assert_allclose(
    reference_gemm,
    atol=1e-2,
    rtol=1e-2,
)

# 性能测试
latency = profiler.do_bench(
    warmup=25,
    rep=100,
    backend="event",
)
```

### 动态形状支持

```python
# 使用动态形状约束
latency = profiler.do_bench(
    dynamic_symbolic_constraints={"m": 2048, "n": 1024, "k": 512},
    warmup=25,
    rep=100,
)
```

### 一致性检查

```python
# 检查内核是否产生一致结果
profiler.assert_consistent(repeat=10)
```

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| Profiler 类 | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py` | 20-279 |
| assert_allclose | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py` | 103-163 |
| assert_consistent | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py` | 196-213 |
| do_bench (Profiler) | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/__init__.py` | 221-270 |
| do_bench (底层) | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py` | 64-138 |
| _bench_with_cuda_events | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py` | 140-173 |
| _bench_with_cupti | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py` | 175-207 |
| _bench_with_cudagraph | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py` | 209-258 |
| suppress_stdout_stderr | `/root/dev/vibe_dsl/tilelang/tilelang/profiler/bench.py` | 12-57 |

## 性能优化建议

1. **选择合适的后端**:
   - `event`: 通用场景，开销适中
   - `cudagraph`: 需要最精确测量时使用
   - `cupti`: 需要详细内核分析时使用

2. **调整 warmup/rep**:
   - 对于快速内核，增加 rep 次数
   - 对于慢速内核，可以减少 warmup

3. **使用 L2 缓存刷新**:
   - 启用 `fast_flush=True` 使用更快的 int32 刷新
   - 禁用可以测量更真实的缓存性能

4. **正确性验证**:
   - 开发阶段启用 `assert_allclose`
   - 生产环境可以跳过以节省时间
