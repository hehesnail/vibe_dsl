# TileLang 测试框架详解

## 概述

TileLang 的测试框架位于 `tilelang/testing/` 目录，提供了全面的测试基础设施，包括硬件需求标记、性能回归测试、随机种子控制等功能。该框架基于 pytest 构建，与 TVM 测试框架紧密集成。

## 文件结构

```
tilelang/testing/
├── __init__.py              # 测试框架主入口
├── perf_regression.py       # 性能回归测试工具
└── python/                  # 测试用例目录
    ├── amd/                 # AMD GPU 测试
    ├── analysis/            # 分析工具测试
    ├── arith/               # 算术运算测试
    ├── autotune/            # 自动调优测试
    ├── cache/               # 缓存机制测试
    ├── carver/              # Carver 模板测试
    ├── components/          # 组件测试
    ├── cpu/                 # CPU 后端测试
    ├── cuda/                # CUDA 特定测试
    ├── debug/               # 调试功能测试
    ├── fastmath/            # 快速数学运算测试
    ├── issue/               # 回归问题测试
    ├── jit/                 # JIT 编译测试
    ├── kernel/              # 核心算子测试
    └── language/            # 语言特性测试
```

## 核心模块详解

### 1. 测试框架主入口 (`__init__.py`)

#### 1.1 主要导出接口

```python
# tilelang/testing/__init__.py:13-23
__all__ = [
    "requires_package",           # 标记需要特定包
    "requires_cuda",              # 标记需要 CUDA
    "requires_metal",             # 标记需要 Metal
    "requires_rocm",              # 标记需要 ROCm
    "requires_llvm",              # 标记需要 LLVM
    "main",                       # 测试主入口
    "requires_cuda_compute_version",  # CUDA 计算能力版本检查
    "process_func",
    "regression",
] + [f"requires_cuda_compute_version_{op}" for op in ("ge", "gt", "le", "lt", "eq")]
```

#### 1.2 测试主入口

```python
# tilelang/testing/__init__.py:27-29
def main():
    """pytest.main() 包装器，允许运行单个测试文件"""
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))
```

使用示例：

```python
# 在测试文件末尾添加
if __name__ == "__main__":
    tilelang.testing.main()
```

#### 1.3 随机种子控制

```python
# tilelang/testing/__init__.py:32-37
def set_random_seed(seed: int = 42) -> None:
    """设置所有随机数生成器的种子，确保测试结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

#### 1.4 CUDA 计算能力版本检查

```python
# tilelang/testing/__init__.py:40-104
def requires_cuda_compute_version(major_version, minor_version=0, mode="ge"):
    """
    标记测试需要特定的 CUDA 计算架构

    Parameters
    ----------
    major_version: int
        主版本号
    minor_version: int
        次版本号
    mode: str
        比较模式："ge"(>=), "gt"(>), "le"(<=), "lt"(<), "eq"(==)
    """
    min_version = (major_version, minor_version)
    try:
        arch = nvcc.get_target_compute_version()
        compute_version = nvcc.parse_compute_version(arch)
    except ValueError:
        compute_version = (0, 0)

    def compare(compute_version, min_version, mode) -> bool:
        if mode == "ge":
            return compute_version >= min_version
        elif mode == "gt":
            return compute_version > min_version
        elif mode == "le":
            return compute_version <= min_version
        elif mode == "lt":
            return compute_version < min_version
        elif mode == "eq":
            return compute_version == min_version
        else:
            raise ValueError(f"Invalid mode: {mode}")

    requires = [
        pytest.mark.skipif(
            not compare(compute_version, min_version, mode),
            reason=f"Requires CUDA compute {mode} {min_version_str}, but have {compute_version_str}",
        ),
        *requires_cuda.marks(),
    ]

    def inner(func):
        return _compose([func], requires)
    return inner
```

便捷函数：

```python
# tilelang/testing/__init__.py:107-124
def requires_cuda_compute_version_ge(major_version, minor_version=0):
    """要求 CUDA 计算能力 >= 指定版本"""
    return requires_cuda_compute_version(major_version, minor_version, mode="ge")

def requires_cuda_compute_version_gt(major_version, minor_version=0):
    """要求 CUDA 计算能力 > 指定版本"""
    return requires_cuda_compute_version(major_version, minor_version, mode="gt")

def requires_cuda_compute_version_eq(major_version, minor_version=0):
    """要求 CUDA 计算能力 == 指定版本"""
    return requires_cuda_compute_version(major_version, minor_version, mode="eq")

def requires_cuda_compute_version_lt(major_version, minor_version=0):
    """要求 CUDA 计算能力 < 指定版本"""
    return requires_cuda_compute_version(major_version, minor_version, mode="lt")

def requires_cuda_compute_version_le(major_version, minor_version=0):
    """要求 CUDA 计算能力 <= 指定版本"""
    return requires_cuda_compute_version(major_version, minor_version, mode="le")
```

使用示例：

```python
import tilelang.testing

# 要求 Ampere 架构 (SM80+)
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_ampere_feature():
    pass

# 要求 Hopper 架构 (SM90)
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_hopper_feature():
    pass
```

### 2. 性能回归测试 (`perf_regression.py`)

#### 2.1 数据结构

```python
# tilelang/testing/perf_regression.py:14-18
@dataclass(frozen=True)
class PerfResult:
    name: str
    latency: float

_RESULTS: list[PerfResult] = []
_MAX_RETRY_NUM = 5
_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="
```

#### 2.2 执行单个性能测试

```python
# tilelang/testing/perf_regression.py:51-68
def process_func(func: Callable[..., float], name: str | None = None, /, **kwargs: Any) -> None:
    """
    执行单个性能函数并记录延迟

    `func` 应该返回一个正的延迟标量（秒或毫秒）
    """
    result_name = getattr(func, "__module__", "<unknown>") if name is None else name
    if result_name.startswith("regression_"):
        result_name = result_name[len("regression_"):]
    latency = float(func(**kwargs))
    _iter = 0
    while latency <= 0.0 and _iter < _MAX_RETRY_NUM:
        latency = float(func(**kwargs))
        _iter += 1
    if latency <= 0.0:
        warnings.warn(f"{result_name} has latency {latency} <= 0. Please verify the profiling results.", RuntimeWarning, 1)
        return
    _RESULTS.append(PerfResult(name=result_name, latency=latency))
```

#### 2.3 批量回归测试

```python
# tilelang/testing/perf_regression.py:71-109
def regression(prefixes: Sequence[str] = ("regression_",), verbose: bool = True) -> None:
    """
    运行调用模块中的入口点并打印 markdown 表格

    这是许多示例脚本调用的函数
    """
    caller_globals = inspect.currentframe().f_back.f_globals

    _reset_results()
    functions: list[tuple[str, Callable[[], Any]]] = []
    for k, v in list(caller_globals.items()):
        if not callable(v):
            continue
        if any(k.startswith(p) for p in prefixes):
            functions.append((k, v))

    sorted_functions = sorted(functions, key=lambda kv: kv[0])
    total = len(sorted_functions)

    for idx, (name, fn) in enumerate(sorted_functions, 1):
        if verbose:
            display_name = name[len("regression_"):] if name.startswith("regression_") else name
            print(f"  ├─ [{idx}/{total}] {display_name}", end="", flush=True)
        start_time = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev_level = logging.root.level
            logging.disable(logging.WARNING)
            try:
                fn()
            finally:
                logging.disable(logging.NOTSET)
                logging.root.setLevel(prev_level)
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f" ({elapsed:.2f}s)", flush=True)

    _emit_results()
```

#### 2.4 结果输出

```python
# tilelang/testing/perf_regression.py:27-44
def _emit_results() -> None:
    """
    为父收集器输出结果

    默认输出历史文本格式。设置 `TL_PERF_REGRESSION_FORMAT=json`
    输出单个 JSON 标记行，这对基准代码的额外打印更健壮
    """
    fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
    if fmt == "json":
        print(_RESULTS_JSON_PREFIX + json.dumps(_results_to_jsonable(), separators=(",", ":")))
        return
    # 回退（人类可读）：每行一个结果
    for r in _RESULTS:
        print(f"{r.name}: {r.latency}")
```

### 3. Pytest 配置 (`examples/conftest.py`)

```python
# examples/conftest.py:27-57
# CuTeDSL 后端：自动标记已知的失败/不支持的测试
CUTEDSL_KNOWN_FAILURES = {
    # 未实现的稀疏操作：tl.tl_gemm_sp
    "sparse_tensorcore/test_example_sparse_tensorcore.py::test_tilelang_example_sparse_tensorcore",
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp",
    # 不稳定 - 单独运行时通过，并行执行时失败
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",
}

def pytest_collection_modifyitems(config, items):
    """当 TILELANG_TARGET=cutedsl 时，自动标记已知的不良测试"""
    if os.environ.get("TILELANG_TARGET") != "cutedsl":
        return

    for item in items:
        nid = item.nodeid
        if _match_any(nid, CUTEDSL_KNOWN_FAILURES):
            item.add_marker(
                pytest.mark.xfail(
                    reason="CuTeDSL: known limitation (unimplemented op or flaky)",
                    strict=False,
                )
            )

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """确保至少收集到一个测试。如果所有测试都被跳过则报错"""
    known_types = {"failed", "passed", "skipped", "deselected", "xfailed", "xpassed", "warnings", "error"}
    if sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"})) == 0:
        terminalreporter.write_sep("!", f"Error: No tests were collected.")
        pytest.exit("No tests were collected.", returncode=5)
```

## 测试用例组织

### 1. 内核测试 (`testing/python/kernel/`)

核心算子测试，包括：

```python
# testing/python/kernel/test_tilelang_kernel_gemm.py 示例
import tilelang.testing

def test_gemm_f16_f16_f16():
    """测试 FP16 GEMM"""
    # ... 测试代码 ...
    pass

@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_gemm_f16_f16_f16_ampere():
    """测试 Ampere 架构特定的 FP16 GEMM"""
    # ... 测试代码 ...
    pass
```

### 2. 语言特性测试 (`testing/python/language/`)

测试 TileLang DSL 的各种语言特性：

```python
# testing/python/language/test_tilelang_language_copy.py
def test_copy_shared_to_local():
    """测试共享内存到局部内存的拷贝"""
    pass

def test_copy_global_to_shared():
    """测试全局内存到共享内存的拷贝"""
    pass
```

### 3. JIT 编译测试 (`testing/python/jit/`)

测试 JIT 编译器的各种功能：

```python
# testing/python/jit/test_tilelang_jit_gemm.py
def test_jit_gemm_basic():
    """测试基本 GEMM JIT 编译"""
    pass

def test_jit_gemm_autotune():
    """测试带自动调优的 GEMM JIT 编译"""
    pass
```

### 4. 问题回归测试 (`testing/python/issue/`)

针对特定 GitHub Issue 的回归测试：

```python
# testing/python/issue/test_tilelang_issue_1001.py
def test_issue_1001_regression():
    """回归测试：Issue #1001 的修复"""
    pass
```

## 使用示例

### 1. 基本测试

```python
import tilelang
import tilelang.testing

def test_basic_gemm():
    """基本 GEMM 测试"""
    @tilelang.jit
    def gemm(A, B, C):
        # ... 实现 ...
        pass

    # 运行测试
    # ...

if __name__ == "__main__":
    tilelang.testing.main()
```

### 2. 性能回归测试

```python
# examples/gemm/regression_example_gemm.py
import tilelang.testing

def run_regression_perf(M=4096, N=4096, K=4096):
    """运行性能回归测试"""
    kernel = matmul(M, N, K, "float16", "float16", "float32")
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")

if __name__ == "__main__":
    tilelang.testing.regression()
```

### 3. 条件测试

```python
import tilelang.testing
import pytest

# 需要特定包
@tilelang.testing.requires_package("bitblas")
def test_with_bitblas():
    pass

# 需要 CUDA
@tilelang.testing.requires_cuda
def test_cuda_only():
    pass

# 需要特定 CUDA 版本
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_ampere_only():
    pass

# 使用 pytest 标记
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_conditional():
    pass
```

### 4. 设置随机种子

```python
import tilelang.testing

def test_with_seed():
    """使用固定随机种子的测试"""
    tilelang.testing.set_random_seed(42)
    # ... 测试代码 ...
```

## 运行测试

### 1. 运行所有测试

```bash
# 在 tilelang 目录下
pytest testing/python/
```

### 2. 运行特定测试文件

```bash
pytest testing/python/kernel/test_tilelang_kernel_gemm.py
```

### 3. 运行特定测试函数

```bash
pytest testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16_f16_f16
```

### 4. 运行示例回归测试

```bash
# 运行单个示例的回归测试
python examples/gemm/regression_example_gemm.py

# 以 JSON 格式输出结果
TL_PERF_REGRESSION_FORMAT=json python examples/gemm/regression_example_gemm.py
```

### 5. 使用特定目标后端

```bash
# 使用 CuTeDSL 后端
TILELANG_TARGET=cutedsl pytest testing/python/
```

## 最佳实践

1. **使用适当的装饰器**：根据测试需求使用 `requires_cuda`、`requires_package` 等装饰器
2. **设置随机种子**：在需要可复现性的测试中调用 `set_random_seed()`
3. **编写回归测试**：为性能敏感的算子编写 `regression_` 前缀的函数
4. **组织测试文件**：按功能模块组织测试文件，保持清晰的目录结构
5. **添加问题回归测试**：为修复的 bug 添加专门的回归测试
