# TileLang Analysis 模块详解

## 模块概述

Analysis 是 TileLang 的 IR 分析和检查模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/analysis/` 目录。该模块提供了一系列编译器 Pass，用于分析和验证 TileLang 程序的结构正确性。

目录结构：
```
tilelang/analysis/
├── __init__.py              # 模块导出
├── ast_printer.py          # AST 可视化打印
├── fragment_loop_checker.py # Fragment 循环检查
├── layout_visual.py        # 布局可视化
└── nested_loop_checker.py  # 嵌套循环检查
```

核心功能：
- AST 结构可视化
- Fragment 缓冲区访问检查
- 嵌套循环模式验证
- 布局推断可视化

## 核心类详解

### 1. ASTPrinter

```python
def ASTPrinter():
    """
    A visitor pass that renders the TileLang AST hierarchy in a visual tree format.
    """
    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        print(f"PrimFunc(params={func.params}, ...)")
        visitor = _ASTPrintVisitor()
        visitor.print_stmt_brief(func.body, func_body_prefix)
        visitor.indent.append(_normal_indent)
        visitor.visit_stmt(func.body)
        return func
    return prim_func_pass(pass_fn, opt_level=0)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/ast_printer.py:109-132`

AST 可视化打印器，以树形结构显示 TensorIR 的层次结构。

#### _ASTPrintVisitor

```python
@tir.functor.visitor
class _ASTPrintVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.indent: list[str] = []
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/ast_printer.py:21-106`

AST 打印访问器，特点：
- 使用 Unicode 连接线显示树形结构（`├──`, `└──`）
- 限制每行输出长度（默认 140 字符）
- 特殊处理 SeqStmt 和 IfThenElse 节点

**输出示例**:
```
PrimFunc(params=[A, B], ret_type=None, ...)
└── body=Block
    ├── name_hint: "root"
    ├── reads: [...]
    ├── writes: [...]
    └── body(Stmt): For
        ├── loop_var: i
        ├── min: 0
        ├── extent: 1024
        └── body(Stmt): Block
```

### 2. FragmentLoopChecker

```python
def FragmentLoopChecker():
    """
    When using T.Parallel over a local/fragment buffer, there are several restrictions
    to ensure that the parallelization is valid.
    """
    def pass_fn(func: PrimFunc, mod, ctx):
        _FragmentLoopCheckVisitor().visit_stmt(func.body)
        return func
    return prim_func_pass(pass_fn, opt_level=0)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/fragment_loop_checker.py:96-111`

Fragment 循环检查器，验证并行循环中对 Fragment 缓冲区的访问是否合法。

#### _FragmentLoopCheckVisitor

```python
@tir.functor.visitor
class _FragmentLoopCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.loop_stack = []
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/fragment_loop_checker.py:47-93`

检查规则：
1. **符号范围检查**: T.Parallel 循环的范围不能是符号变量
2. **索引使用检查**: 符号范围的循环变量不能用于索引 Fragment 缓冲区

**错误示例**:
```python
# 非法：符号范围的并行循环变量用于索引 Fragment
for i in T.Parallel(M):  # M 是符号变量
    fragment[i] = ...     # 错误！
```

### 3. NestedLoopChecker

```python
def NestedLoopChecker():
    """
    User-friendly pass which identifies any invalid nested-loop pattern.
    """
    def pass_fn(func: PrimFunc, mod, ctx):
        _NestedLoopCheckVisitor().visit_stmt(func.body)
        return func
    return prim_func_pass(pass_fn, opt_level=0)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/nested_loop_checker.py:61-119`

嵌套循环检查器，验证 TileLang 中四种循环类型的嵌套规则：
- `T.serial`: 串行循环
- `T.Parallel`: 并行循环
- `T.Pipelined`: 流水线循环
- `T.Persistent`: 持久化循环

#### _NestedLoopCheckVisitor

```python
@tir.functor.visitor
class _NestedLoopCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.in_parallel_context = False
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/nested_loop_checker.py:24-58`

**检查规则**:

| 规则 | 说明 | 示例 |
|------|------|------|
| Rule 1 | T.serial 可以嵌套在任何循环中 | 无限制 |
| Rule 2 | 连续的 T.Parallel 允许嵌套 | `for i in T.Parallel: for j in T.Parallel` |
| Rule 2a | 非连续 T.Parallel 禁止嵌套 | `for i in T.Parallel: stmt; for j in T.Parallel` |
| Rule 2b | T.Parallel 内禁止 TileOp | `for i in T.Parallel: T.copy(A, B)` |
| Rule 3 | T.Pipelined 不能在 T.Parallel 内 | `for i in T.Parallel: for j in T.Pipelined` |

### 4. LayoutVisual

```python
def LayoutVisual(formats: str = ""):
    """
    User-friendly pass which visualizes fragment layouts inferred during compilation.
    """
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor(formats=formats).visit_stmt(func.body)
        return func
    return prim_func_pass(pass_fn, opt_level=0)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/layout_visual.py:81-86`

布局可视化 Pass，显示 Fragment 布局的推断结果。

#### _LayoutVisualVisitor

```python
@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
    def __init__(self, formats: list[str] = ""):
        super().__init__()
        self.layout_found = []
        self.processed_layouts = set()
        self.formats_list = [f for f in formats if f != "txt"]
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/analysis/layout_visual.py:32-76`

功能：
- 从 Block 注解中提取 `layout_map`
- 打印 Fragment 布局信息（形状、线程映射、索引映射）
- 生成可视化图表（PDF/PNG/SVG）

**输出格式**:
```
A inferenced layout:
  Shape: (128, 128) -> (128, 128)
  Thread: lambda i, j: (i * 16 + j) % 256
  Index:  lambda i, j: (i, j)
```

## 实现逻辑分析

### Pass 注册机制

所有分析 Pass 都使用 TVM 的 `prim_func_pass` 注册：

```python
from tvm.tir.transform import prim_func_pass

def MyChecker():
    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        # 执行检查
        _Visitor().visit_stmt(func.body)
        return func
    return prim_func_pass(pass_fn, opt_level=0)
```

### Visitor 模式

所有检查器都继承自 `PyStmtExprVisitor`：

```python
@tir.functor.visitor
class _CheckVisitor(PyStmtExprVisitor):
    def visit_for_(self, op: For) -> None:
        # 处理 For 循环节点
        self.visit_stmt(op.body)

    def visit_block_(self, op: tir.Block) -> None:
        # 处理 Block 节点
        pass
```

### 嵌套循环检查算法

```
遍历 AST:
  遇到 T.Parallel:
    如果已在 parallel 上下文中:
      检查是否为连续嵌套
      如果不是，报错
    设置 in_parallel_context = True
    继续遍历子节点
    恢复 in_parallel_context = False

  遇到 T.Pipelined:
    如果 in_parallel_context:
      报错（Pipelined 不能在 Parallel 内）

  遇到 TileOp (T.copy, T.gemm 等):
    如果 in_parallel_context:
      报错（TileOp 不能在 Parallel 内）
```

## 使用示例

### AST 打印

```python
import tilelang as tl
from tilelang.analysis import ASTPrinter

# 定义 kernel
@T.prim_func
def my_kernel(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32")):
    for i in T.Parallel(1024):
        B[i] = A[i] * 2.0

# 应用 ASTPrinter Pass
sch = tl.Schedule(my_kernel)
sch.apply(ASTPrinter())
```

### 嵌套循环检查

```python
from tilelang.analysis import NestedLoopChecker

# 错误的 kernel：嵌套 Parallel
@T.prim_func
def bad_kernel(A: T.Buffer((1024, 1024), "float32")):
    for i in T.Parallel(1024):
        # 中间有语句，不是连续嵌套
        x = A[i, 0]
        for j in T.Parallel(1024):  # 错误！
            A[i, j] = x

# 检查会抛出 ValueError
sch = tl.Schedule(bad_kernel)
sch.apply(NestedLoopChecker())
# ValueError: [Tilelang Semantic Check] Nested parallel loops are not allowed.
```

### Fragment 循环检查

```python
from tilelang.analysis import FragmentLoopChecker

# 错误的 kernel：符号范围用于 Fragment 索引
@T.prim_func
def bad_fragment_kernel(A: T.Buffer((1024,), "float32"), M: T.int32):
    fragment = T.alloc_fragment((1024,), "float32")
    for i in T.Parallel(M):  # M 是符号变量
        fragment[i] = A[i]  # 错误！

# 检查会抛出 ValueError
sch.apply(FragmentLoopChecker())
# ValueError: Loop variable i in a T.Parallel loop with symbolic range...
```

### 布局可视化

```python
from tilelang.analysis import LayoutVisual

# 启用布局可视化
import os
os.environ["TL_ENABLE_LAYOUT_VISUALIZATION"] = "png,svg"

# 编译 kernel 时会自动打印和保存布局
@T.prim_func
def layout_kernel(...):
    # 使用 Fragment
    fragment = T.alloc_fragment((128, 128), "float16")
    ...

# 应用可视化 Pass
sch.apply(LayoutVisual(formats="png,svg"))
```

## 配置选项

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TL_ENABLE_LAYOUT_VISUALIZATION` | 启用布局可视化 | "" (禁用) |
| | 可选值: "png", "pdf", "svg", "all", 或组合如 "png,svg" | |

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| ASTPrinter | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/ast_printer.py` | 109-132 |
| _ASTPrintVisitor | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/ast_printer.py` | 21-106 |
| FragmentLoopChecker | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/fragment_loop_checker.py` | 96-111 |
| _FragmentLoopCheckVisitor | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/fragment_loop_checker.py` | 47-93 |
| NestedLoopChecker | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/nested_loop_checker.py` | 61-119 |
| _NestedLoopCheckVisitor | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/nested_loop_checker.py` | 24-58 |
| LayoutVisual | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/layout_visual.py` | 81-86 |
| _LayoutVisualVisitor | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/layout_visual.py` | 32-76 |
| print_fragment_format | `/root/dev/vibe_dsl/tilelang/tilelang/analysis/layout_visual.py` | 9-29 |

## 调试建议

1. **使用 ASTPrinter**: 当不确定 IR 结构时，使用 ASTPrinter 查看完整的 AST 树
2. **启用布局可视化**: 设置 `TL_ENABLE_LAYOUT_VISUALIZATION=png` 查看 Fragment 布局
3. **检查循环嵌套**: 遇到并行相关错误时，检查是否违反了嵌套规则
4. **Fragment 使用**: 确保 Fragment 索引不使用符号范围的循环变量
