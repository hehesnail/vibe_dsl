# TileLang AST、Parser 与 Eager 模式详解

## 1. 概述

TileLang 的语言核心由三个关键组件构成：AST（抽象语法树）、Parser（解析器）和 Eager 模式执行引擎。本文档深入分析这些组件的实现原理和交互机制。

### 1.1 架构概览

```
Python Source Code
       |
       v
+------------------+
|   DSLMutator     |  (tilelang/language/eager/ast.py:261)
|  (AST Transform) |
+--------+---------+
         |
         v
+------------------+
|   IRGenerator    |  (tilelang/language/eager/ast.py:630)
|  (Code Generation)|
+--------+---------+
         |
         v
+------------------+
|  Builder Mode    |  (tilelang/language/eager/builder.py)
|  (TIR Building)  |
+--------+---------+
         |
         v
+------------------+
|   TVM TIR        |
|  (Lower & Run)   |
+------------------+
```

## 2. AST IR 定义详解

### 2.1 核心 IR 文件结构

TileLang 的 AST IR 定义位于 `tilelang/language/ast/ir.py`，这是一个基于 TVM IRBuilder 的扩展系统。

#### 2.1.1 主要组件分类

**类型别名与基础定义** (`tilelang/language/ast/ir.py:42-50`):
```python
# 核心类型别名
Buffer = tir.Buffer
Var = tir.Var
IterVar = tir.IterVar
PrimExpr = tir.PrimExpr
```

**Buffer 创建函数** (`tilelang/language/ast/ir.py:53-116`):
```python
def buffer(
    shape: ShapeType,
    dtype: DType = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = 0,
    offset_factor: int = 0,
    buffer_type: str = "",
    axis_separators: List[int] = None,
) -> Buffer:
```

**指针类型支持** (`tilelang/language/ast/ir.py:119-141`):
```python
def ptr(dtype, storage_scope="global"):
    """Create a pointer type with specified dtype and storage scope."""
```

### 2.2 循环构造器

TileLang 提供多种循环类型，通过 `tilelang/language/ast/ir.py` 实现：

| 函数 | 行号 | 用途 |
|------|------|------|
| `serial` | 144 | 串行循环 |
| `parallel` | 171 | 并行循环 |
| `vectorized` | 198 | 向量化循环 |
| `unroll` | 225 | 循环展开 |
| `thread_binding` | 252 | 线程绑定 |

**示例 - Thread Binding 实现** (`tilelang/language/ast/ir.py:252-281`):
```python
def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    annotations: Mapping[str, Any] = None,
) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ir.thread_binding(start, stop, thread, annotations=annotations)
```

### 2.3 Buffer 操作

**Buffer 声明与初始化** (`tilelang/language/ast/ir.py:284-343`):
```python
def alloc_buffer(
    shape: ShapeType,
    dtype: DType = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = 0,
    offset_factor: int = 0,
    buffer_type: str = "",
    axis_separators: List[int] = None,
) -> Buffer:
```

**Buffer 加载与存储** (`tilelang/language/ast/ir.py:346-396`):
```python
def buffer_load(buf: Buffer, indices: List[PrimExpr], predicate=True) -> PrimExpr:
    """Load from buffer at specified indices."""

def buffer_store(
    buf: Buffer,
    value: PrimExpr,
    indices: List[PrimExpr],
    predicate: Union[PrimExpr, bool] = True,
) -> None:
    """Store value to buffer at specified indices."""
```

### 2.4 数学与位运算操作

TileLang 在 `tilelang/language/ast/ir.py:399-1027` 定义了丰富的运算操作：

**算术运算**: `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `ceil`, `cos`, `cosh`, `exp`, `exp2`, `exp10`, `floor`, `log`, `log1p`, `log2`, `log10`, `pow`, `round`, `rsqrt`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`

**位运算**: `bitwise_and`, `bitwise_not`, `bitwise_or`, `bitwise_xor`, `clz`, `popcount`, `shift_left`, `shift_right`

**比较与选择**: `max`, `min`, `if_then_else`, `likely`

### 2.5 特殊内建函数

**PTX/CUDA 特定操作** (`tilelang/language/ast/ir.py:1030-1150`):
```python
# 异步拷贝与屏障
ptx_cp_async
ptx_cp_async_bulk
ptx_wait_group
ptx_commit_group
ptx_cp_async_barrier

# MMA/WGMMA 操作
ptx_mma
ptx_mma_sp
ptx_wgmma_ss
ptx_wgmma_rs
ptx_ldmatrix

# 屏障操作
ptx_init_barrier_thread_count
ptx_fence_barrier_init
ptx_arrive_barrier
ptx_arrive_barrier_expect_tx
ptx_wait_barrier
create_barriers
```

**Tensor Core 操作** (`tilelang/language/ast/ir.py:1153-1190`):
```python
tvm_load_matrix_sync
tvm_mma_sync
tvm_bmma_sync
tvm_fill_fragment
tvm_store_matrix_sync
tvm_mfma
tvm_mfma_store
tvm_rdna_wmma
tvm_rdna_wmma_store
```

## 3. Parser 解析器实现

### 3.1 解析器架构

TileLang 的解析器基于 TVM Script Parser 构建，位于 `tilelang/language/parser/` 目录。

```
tilelang/language/parser/
├── __init__.py      # 模块导出
├── entry.py         # 入口装饰器 (prim_func, macro)
├── operation.py     # 操作注册
└── parser.py        # 核心解析逻辑
```

### 3.2 入口装饰器

**prim_func 装饰器** (`tilelang/language/parser/entry.py:37-86`):
```python
def prim_func(
    func: Optional[Callable] = None,
    private: bool = False,
    check_well_formed=True
) -> Union[PrimFunc, Callable]:
    """解析 TIR prim func 的装饰器。

    Parameters
    ----------
    func : Callable
        待解析的函数
    private : bool
        是否为私有函数（无全局符号）

    Returns
    -------
    PrimFunc : 解析后的 TIR 原函数
    """
    def decorator_wrapper(func):
        if not inspect.isfunction(func):
            raise TypeError(f"Expect a function, but got: {func}")
        if utils.is_defined_in_class(outer_stack, func):
            return func
        f = parse(func, utils.inspect_function_capture(func),
                  check_well_formed=check_well_formed)
        setattr(f, "__name__", func.__name__)
        return f
```

**Macro 装饰器** (`tilelang/language/parser/entry.py:105-156`):
```python
def macro(*args, hygienic: bool = True) -> Callable:
    """宏定义装饰器。

    Parameters
    ----------
    hygienic: bool
        是否为卫生宏。卫生宏在定义处解析符号，
        非卫生宏在使用处解析符号。
    """
```

### 3.3 核心解析逻辑

解析器使用 `@dispatch.register` 机制注册不同 AST 节点的处理方法。

**For 循环解析** (`tilelang/language/parser/parser.py:177-199`):
```python
@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.frame.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of: range, T.serial, T.grid, ..."
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            self.eval_assign(target=node.target, source=iters,
                           bind_value=bind_for_value)
            self.visit_body(node.body)
```

**赋值语句解析** (`tilelang/language/parser/parser.py:219-267`):
```python
@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    if isinstance(lhs, doc.Subscript):
        # Buffer store operation
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)
```

**函数定义解析** (`tilelang/language/parser/parser.py:369-417`):
```python
@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    privacy = find_decorator_annotation(node, "private", default=False)
    with self.var_table.with_frame():
        with T.prim_func(is_private=privacy):
            T.func_name(node.name)
            for arg in node.args.args:
                if arg.annotation is None:
                    self.report_error(arg, "Type annotation required")
                ann = self.eval_expr(arg.annotation)
                param = T.arg(arg.arg, ann)
                self.var_table.add(arg.arg, param)
            self.visit_body(node.body)
```

### 3.4 值绑定机制

解析器提供三种值绑定方法：

**With 语句绑定** (`tilelang/language/parser/parser.py:42-76`):
```python
def bind_with_value(self: Parser, node: doc.expr,
                    var_name: str, value: Any) -> Any:
    """绑定 with 语句中的值，如 `with T.grid(...) as i, j, k`。"""
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, (Buffer, Var)):
        IRBuilder.name(var_name, value)
        return value
```

**For 语句绑定** (`tilelang/language/parser/parser.py:78-112`):
```python
def bind_for_value(self: Parser, node: doc.expr,
                   var_name: str, value: Any) -> Any:
    """绑定 for 语句中的迭代变量。"""
    if isinstance(value, Var):
        IRBuilder.name(var_name, value)
        return value
```

**赋值语句绑定** (`tilelang/language/parser/parser.py:114-160`):
```python
def bind_assign_value(self: Parser, node: doc.expr,
                      var_name: str, value: Any) -> Any:
    """绑定赋值语句中的值。"""
    if isinstance(value, T.meta_var):
        return value.value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        IRBuilder.name(var_name, res)
        return res
```

### 3.5 操作注册

**表达式操作注册** (`tilelang/language/parser/operation.py:29-155`):
```python
def _register_expr_op(ty: type):
    """注册表达式操作，包括算术、比较、逻辑运算。"""
    def _and(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if DataType(a.dtype).lanes > 1 or DataType(b.dtype).lanes > 1:
            return a & b
        else:
            return tir.And(a, b)

    def _auto_broadcast(a, b, op):
        """自动广播处理不同形状的运算数。"""
        if DataType(a.dtype).lanes == DataType(b.dtype).lanes:
            return op(a, b)
        elif DataType(a.dtype).lanes == 1:
            broadcast_a = tir.Broadcast(a, DataType(b.dtype).lanes)
            return op(broadcast_a, b)
```

## 4. TIR 转换机制

### 4.1 TIR 模块结构

```
tilelang/language/tir/
├── __init__.py    # 导出 prim_func 和 IR 操作
├── entry.py       # TIR 入口函数
├── ir.py          # TIR IR 构造器
└── op.py          # TIR 操作包装
```

### 4.2 TIR 入口函数

**prim_func 入口** (`tilelang/language/tir/entry.py:10-62`):
```python
def prim_func(
    func: Callable | None = None,
    private: bool = False,
    check_well_formed: bool = False
) -> PrimFunc | Callable:
    """TIR prim_func 解析入口。

    与 parser/entry.py 的区别：
    - 此版本使用 check_well_formed=False 作为默认
    - 用于 TileLang 内部 TIR 代码生成
    """
    def decorator_wrapper(func):
        f = parse(func, utils.inspect_function_capture(func),
                  check_well_formed=check_well_formed)
        setattr(f, "__name__", func.__name__)
        return f
```

**Macro 入口** (`tilelang/language/tir/entry.py:64-118`):
```python
def macro(*args, hygienic: bool = True) -> Callable:
    """TIR 宏定义入口。"""
    def _decorator(func: Callable) -> _tir_entry.TIRMacro:
        source, closure_vars = scan_macro(func, utils.inspect_function_capture(func))
        obj = _tir_entry.TIRMacro(source, closure_vars, func, hygienic)
        obj.__name__ = func.__name__
        return obj
```

### 4.3 TIR IR 构造器

**循环构造器** (`tilelang/language/tir/ir.py:9-148`):
```python
def serial(start: PrimExpr, stop: PrimExpr = None,
           *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """串行循环构造器。"""
    return _ir.serial(start=start, stop=stop, annotations=annotations)

def parallel(start: PrimExpr, stop: PrimExpr = None,
             *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """并行循环构造器。"""
    return _ir.parallel(start=start, stop=stop, annotations=annotations)

def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    *,
    annotations: dict[str, Any] = None,
) -> frame.ForFrame:
    """线程绑定循环构造器。"""
    return _ir.thread_binding(start=start, stop=stop,
                              thread=thread, annotations=annotations)

def grid(*extents: PrimExpr) -> frame.ForFrame:
    """多维网格循环构造器。"""
    return _ir.grid(*extents)
```

**Unroll 循环的特殊处理** (`tilelang/language/tir/ir.py:75-102`):
```python
def unroll(start: PrimExpr, stop: PrimExpr = None,
           *, annotations: dict[str, Any] = None) -> frame.ForFrame:
    """循环展开构造器。

    默认添加 pragma_unroll_explicit=False 注解，
    确保编译器自动处理循环展开。
    """
    if annotations is None:
        annotations = {"pragma_unroll_explicit": False}
    else:
        annotations = dict(annotations)
        annotations.setdefault("pragma_unroll_explicit", False)
    return _ir.unroll(start=start, stop=stop, annotations=annotations)
```

### 4.4 TIR 操作包装

**操作包装器模式** (`tilelang/language/tir/ir.py:151-169`):
```python
def _op_wrapper(func):
    """包装 TIR 操作，移除 dtype 参数。"""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)
    return wrapped

# 数学操作
abs = _op_wrapper(_tir_op.abs)
exp = _op_wrapper(_tir_op.exp)
sqrt = _op_wrapper(_tir_op.sqrt)
# ... 更多操作
```

**带 dtype 转发包装器** (`tilelang/language/tir/ir.py:151-159`):
```python
def _dtype_forward(func):
    """转发 dtype 参数到第一个位置。"""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)
    return wrapped

# 外部调用操作
call_extern = _dtype_forward(_tir_op.call_extern)
call_intrin = _dtype_forward(_tir_op.call_intrin)
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
```

### 4.5 扩展操作

`tilelang/language/tir/op.py` 提供了 TileLang 特定的操作扩展：

**调用操作** (`tilelang/language/tir/op.py:14-83`):
```python
def call_packed(*args, span=None):
    """调用外部打包函数。"""
    return _tvm_op.call_packed(*args, span=span)

def call_cpacked(*args, span=None):
    """调用 C 打包函数。"""
    return _tvm_op.call_cpacked(*args, span=span)
```

**PTX 操作** (`tilelang/language/tir/op.py:86-150`):
```python
def ptx_wait_group(group: PrimExpr, span=None):
    """PTX wait_group 指令。"""
    return _tvm_op.ptx_wait_group(group, span)

def ptx_commit_group(span=None):
    """PTX commit_group 指令。"""
    return _tvm_op.ptx_commit_group(span)

def ptx_cp_async_barrier(barrier_id: PrimExpr, span=None):
    """PTX 异步拷贝屏障。"""
    return _tvm_op.ptx_cp_async_barrier(barrier_id, span)
```

**Tensor Core 操作** (`tilelang/language/tir/op.py:153-280`):
```python
def tvm_load_matrix_sync(fragment, m, n, k, source, layout, ...):
    """加载矩阵到 Tensor Core fragment。"""
    return _tvm_op.tvm_load_matrix_sync(...)

def tvm_mma_sync(fragment_d, fragment_a, fragment_b, fragment_c, ...):
    """执行矩阵乘累加操作。"""
    return _tvm_op.tvm_mma_sync(...)
```

## 5. Eager 模式详解

### 5.1 Eager 模式架构

Eager 模式允许 TileLang 代码在 Python 中直接执行，无需显式编译。这是通过 AST 转换和即时代码生成实现的。

```
Python Function
      |
      v
+-------------+
|   mutate    |  (tilelang/language/eager/ast.py:652)
|  (AST Xform)|
+------+------+
       |
       v
+-------------+
| IRGenerator |  (tilelang/language/eager/ast.py:630)
|  (Wrapper)  |
+------+------+
       |
       v
+-------------+
|   Builder   |  (tilelang/language/eager/builder.py)
| (TIR Build) |
+------+------+
       |
       v
+-------------+
|   JIT Run   |
+-------------+
```

### 5.2 Eager AST 核心组件

**BaseBuilder 基类** (`tilelang/language/eager/ast.py:174-252`):
```python
class BaseBuilder:
    """Eager 模式构建器基类。

    提供上下文管理、变量绑定、运算求值等基础功能。
    """
    empty = _empty

    def ctx_if(self, cond) -> Iterable[_T]:
        """If 语句上下文。"""
        yield cond

    def ctx_then(self, val: _T) -> Iterable[None]:
        """Then 分支上下文。"""
        if val:
            yield

    def ctx_else(self, val: _T) -> Iterable[None]:
        """Else 分支上下文。"""
        if not val:
            yield

    def bind(self, name: str, value: Any, annot: Any = empty) -> Any:
        """绑定变量名到值。"""
        return value

    def aug_assign(self, op: Operator, target: Any,
                   aug_value: Any, name: str | None = None) -> Any:
        """执行增强赋值操作。"""
        return eval_op(op, target, aug_value)

    def boolop(self, op: BoolOp, left: Any,
               right: Callable[[], Any] | None = None) -> Any:
        """执行布尔运算。"""
        if op == "And":
            return left and right()
        if op == "Or":
            return left or right()
        if op == "Not":
            return not left
```

**DSLMutator AST 转换器** (`tilelang/language/eager/ast.py:261-625`):
```python
class DSLMutator(ast.NodeTransformer):
    """DSL AST 转换器。

    将 Python AST 转换为使用 BaseBuilder 的代码。
    """
    def __init__(self, nonlocals: dict[str, Any],
                 globals: dict[str, Any], filename: str):
        self.tmp_counter = 0
        self.nonlocals = nonlocals
        self.globals = globals
        self.extra_type_hints: dict[str, Any] = {}
        self.filename = filename

    def visit_If(self, node: ast.If):
        """转换 if 语句为上下文管理器形式。"""
        node = self.generic_visit(node)
        br = self.get_tmp()
        if len(node.orelse) == 0:
            return quote(
                f"for {br} in __tb.ctx_if(cond):\n"
                f"  for _ in __tb.ctx_then({br}):\n    pass\n",
                cond=node.test, passes=[node.body], span=node
            )
        return quote(
            f"for {br} in __tb.ctx_if(cond):\n"
            f"  for _ in __tb.ctx_then({br}):\n    pass\n"
            f"  for _ in __tb.ctx_else({br}):\n    pass\n",
            cond=node.test, passes=[node.body, node.orelse], span=node
        )

    def visit_For(self, node: ast.For):
        """转换 for 循环。"""
        node = self.generic_visit(node)
        tmp = self.get_tmp()
        var = ast.Name(tmp, ctx=ast.Load())
        ast_set_span(var, ast_get_span(node.target))
        stmts = self._emit_assign_target(node.target, var)
        return quote(
            f"for {tmp} in __tb.ctx_for(range):\n  pass\n",
            target=node.target, range=node.iter,
            passes=[stmts + node.body], span=node
        )

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST]:
        """转换赋值语句，支持元组解包。"""
        node = self.generic_visit(node)
        rval = node.value
        if len(node.targets) == 1:
            return self._emit_assign_target(node.targets[0], rval)
        else:
            # 多目标赋值处理
            tmp_name = self.get_tmp()
            ...
```

**Quote 工具函数** (`tilelang/language/eager/ast.py:63-81`):
```python
def quote(expr: str, *, passes: list[Any] | None = None,
          span=None, **kws) -> list[ast.AST]:
    """将字符串表达式转换为 AST 节点。

    支持占位符替换和 Pass 节点注入。
    """
    tree = ast.parse(expr)
    if isinstance(span, ast.AST):
        span = ast_get_span(span)
    tree = QuoteVisitor(kws, passes, span).visit(tree)
    return tree.body

def quote1(expr: str, *, passes: list[Any] | None = None,
           span=None, **kws) -> ast.AST:
    """quote 的单语句版本。"""
    res = quote(expr, passes=passes, span=span, **kws)
    assert len(res) == 1
    return res[0]

def quote_expr(expr: str, **kws) -> ast.expr:
    """quote 的表达式版本。"""
    res = quote1(expr, **kws)
    assert isinstance(res, ast.Expr)
    return res.value
```

**IRGenerator 数据类** (`tilelang/language/eager/ast.py:630-635`):
```python
@dataclass
class IRGenerator(Generic[_P, _T]):
    """IR 生成器包装类。

    包含转换后的生成函数、源代码和额外类型提示。
    """
    gen: Callable[[BaseBuilder], Callable[_P, _T]]
    source: str
    extra_type_hints: dict[str, Any] = field(default_factory=dict)
```

**mutate 入口函数** (`tilelang/language/eager/ast.py:652-706`):
```python
def mutate(func: Callable[_P, _T]) -> IRGenerator[_P, _T]:
    """将 Python 函数转换为 IR 生成器。

    执行 AST 分析和转换，创建可用于代码生成的 IRGenerator。

    Args:
        func: 待转换的 Python 函数

    Returns:
        IRGenerator: 包含转换后函数的生成器实例
    """
    tree = utils.get_ast(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    nonlocals = utils.get_func_nonlocals(func)

    # DSLMutator 生成名为 `make_closure` 的函数
    # 接受所有 nonlocal 名称，返回转换后的函数
    mut = DSLMutator(nonlocals, func.__globals__, Path(filename).name)
    tree = mut.visit(tree)
    make_closure = utils.get_compiled_object(
        tree, "make_closure", filename, func.__globals__
    )
    fn = make_closure(**nonlocals)
    return IRGenerator(gen=fn, source=ast.unparse(tree),
                       extra_type_hints=mut.extra_type_hints)
```

### 5.3 Builder 模式实现

**Builder 类** (`tilelang/language/eager/builder.py:39-1500+`):
```python
class Builder(BaseBuilder):
    """TileLang Eager 模式构建器。

    继承 BaseBuilder，实现具体的 TIR 代码生成逻辑。
    """
    def __init__(self, frame: KernelLaunchFrame = None,
                 params: list[str] = None):
        self.params = params
        self.frame = frame
        self.ib = IRBuilder()  # TVM IRBuilder
        self.kernel = frame.kernel if frame else None
        self._func = None
        self._launch_param_concrete_values: dict[str, Any] = {}
        self._cached_for_op_const_values: dict[str, Any] = {}
        self._cached_for_op_param_values: dict[str, Any] = {}
        self._cached_for_op_var_values: dict[str, Any] = {}
        self._cached_thread_extent: dict[str, tir.PrimExpr] = {}
        self._cached_thread_extent_list: list[tuple[str, tir.PrimExpr]] = []
```

**表达式解包** (`tilelang/language/eager/builder.py:39-75`):
```python
def unwrap_expr(expr) -> PrimExpr | int | float:
    """解包表达式为 PrimExpr 或标量值。

    处理 meta_var、Ref 类型和 Var 类型。
    """
    if isinstance(expr, tir.meta_var):
        expr = expr.value
    elif isinstance(expr, Ref):
        return expr.load()
    elif is_var(expr):
        expr = tir.BufferLoad(expr, indices=[0])
    return expr
```

**变量与引用管理** (`tilelang/language/eager/builder.py:78-200`):
```python
class Ref:
    """可变引用类型，用于 Eager 模式中的变量。

    提供 load/store 语义，支持延迟求值。
    """
    def __init__(self, builder: Builder, name: str,
                 var: Var | Buffer, annot: Any = None):
        self.builder = builder
        self.name = name
        self.var = var
        self.annot = annot

    def load(self):
        """加载引用值。"""
        if isinstance(self.var, Var):
            return self.var
        elif isinstance(self.var, Buffer):
            return tir.BufferLoad(self.var, indices=[0])
        raise ValueError(f"Unsupported var type: {type(self.var)}")

    def store(self, value):
        """存储值到引用。"""
        value = self.builder._unwrap_expr(value)
        if isinstance(self.var, Var):
            self.builder.emit(tir.LetStmt(self.var, value, tir.Evaluate(0)))
        elif isinstance(self.var, Buffer):
            self.builder.emit(tir.BufferStore(self.var, value, indices=[0]))
```

**变量绑定** (`tilelang/language/eager/builder.py:203-280`):
```python
def bind(self, name: str, value: Any, annot: Any = empty) -> Any:
    """绑定变量名到值，创建 Ref 或返回具体值。

    根据注解类型决定如何绑定：
    - Tensor/StridedTensor 注解: 创建 Buffer
    - 标量注解: 创建 Var
    - 无注解: 直接返回值
    """
    value = self.unwrap_value(value)

    # 处理指针类型注解
    if annot is not self.empty and annot is not None:
        from tilelang.language.proxy import TensorProxy, StridedTensorProxy, ptr
        if annot is ptr or isinstance(annot, (TensorProxy, StridedTensorProxy)):
            return self._bind_ptr(name, value, annot)

    # 处理标量类型注解
    if annot is not self.empty and annot is not None:
        if isinstance(annot, dt.dtype):
            return self._bind_scalar(name, value, annot)

    # 无注解或已解析值
    if isinstance(value, (int, float, bool, str, PrimExpr)):
        return value
    if isinstance(value, Ref):
        return value.load()
    return value
```

**标量变量绑定** (`tilelang/language/eager/builder.py:283-340`):
```python
def _bind_scalar(self, name: str, value: Any, dtype: dt.dtype):
    """绑定标量变量。

    创建 TIR Var 和可选的初始化语句。
    """
    var = tir.Var(name, dtype.tvm_dtype)
    ref = Ref(self, name, var, dtype)

    if value is not self.empty and value is not None:
        value = unwrap_expr(value)
        if not isinstance(value, PrimExpr):
            value = tvm.tir.const(value, dtype.tvm_dtype)
        self.emit(tir.LetStmt(var, value, tir.Evaluate(0)))

    return ref
```

**指针/Buffer 绑定** (`tilelang/language/eager/builder.py:343-450`):
```python
def _bind_ptr(self, name: str, value: Any, annot: Any):
    """绑定指针/Buffer 变量。

    处理 Tensor、StridedTensor 和原始指针类型。
    """
    if isinstance(value, Buffer):
        # 已有 Buffer，直接包装
        return BufferRef(self, name, value, annot)

    if isinstance(value, np.ndarray):
        # NumPy 数组，创建对应 Buffer
        shape = value.shape
        dtype = dt.from_numpy_dtype(value.dtype)
        buffer = self._create_buffer(name, shape, dtype, value)
        return BufferRef(self, name, buffer, annot)

    if isinstance(value, (int, PrimExpr)):
        # 指针地址值
        return self._bind_raw_ptr(name, value, annot)
```

**线程范围管理** (`tilelang/language/eager/builder.py:453-550`):
```python
def thread_binding(self, thread: str, extent: PrimExpr):
    """绑定线程轴到范围。

    缓存线程范围用于后续代码生成。
    """
    if thread in self._cached_thread_extent:
        return self._cached_thread_extent[thread]

    var = tir.Var(thread, "int32")
    self._cached_thread_extent[thread] = var
    self._cached_thread_extent_list.append((thread, var))
    return var

def get_thread_extent(self, thread: str) -> tir.PrimExpr:
    """获取线程轴的范围。"""
    return self._cached_thread_extent.get(thread)
```

**For 循环构建** (`tilelang/language/eager/builder.py:553-700`):
```python
def ctx_for(self, range: Iterable[Any]) -> Iterable[Any]:
    """For 循环上下文。

    支持多种迭代类型：
    - range: 标准范围迭代
    - T.serial: 串行循环
    - T.parallel: 并行循环
    - T.thread_binding: 线程绑定
    """
    if isinstance(range, TIter):
        # 处理 TileLang 迭代器
        return self._handle_titer(range)

    if hasattr(range, '__iter__'):
        # 标准迭代
        for item in range:
            yield item
```

**If/Then/Else 构建** (`tilelang/language/eager/builder.py:703-800`):
```python
def ctx_if(self, cond) -> Iterable[bool]:
    """If 条件上下文。

    生成 TIR IfThenElse 节点。
    """
    cond = unwrap_expr(cond)
    self._if_cond = cond
    yield cond

def ctx_then(self, val: bool) -> Iterable[None]:
    """Then 分支上下文。"""
    if val:
        with self.ib.if_scope(self._if_cond):
            yield

def ctx_else(self, val: bool) -> Iterable[None]:
    """Else 分支上下文。"""
    if not val:
        with self.ib.else_scope():
            yield
```

**增强赋值** (`tilelang/language/eager/builder.py:803-900`):
```python
def aug_assign(self, op: Operator, target: Any,
               aug_value: Any, name: str | None = None) -> Any:
    """执行增强赋值 (+=, -=, *= 等)。

    生成对应的 TIR 运算和存储语句。
    """
    target_val = unwrap_expr(target)
    aug_val = unwrap_expr(aug_value)

    # 执行运算
    result = eval_op(op, target_val, aug_val)

    # 存储结果
    if name:
        var = tir.Var(name, result.dtype)
        self.emit(tir.LetStmt(var, result, tir.Evaluate(0)))
        return var
    return result
```

**Buffer 操作** (`tilelang/language/eager/builder.py:903-1100`):
```python
def assign_slice(self, lval: Any, sl: slice, value: Any, annot: Any = empty):
    """切片赋值操作。

    支持多维切片和步长。
    """
    if isinstance(lval, BufferRef):
        # Buffer 切片赋值
        indices = self._parse_slice(sl)
        value = unwrap_expr(value)
        self.emit(tir.BufferStore(lval.buffer, value, indices))
    elif isinstance(lval, Ref):
        # 标量引用赋值
        lval.store(value)
```

**函数定义与参数** (`tilelang/language/eager/builder.py:1103-1250`):
```python
def arg(self, name: str, value: Any):
    """处理函数参数。

    在函数定义时绑定参数。
    """
    if self.params is None or name not in self.params:
        return value

    # 创建参数变量
    param_var = tir.Var(name, "int32")
    self._launch_param_concrete_values[name] = value
    return param_var

def ret(self, value: Any) -> Any:
    """返回值处理。"""
    value = unwrap_expr(value)
    self.emit(tir.Evaluate(tir.ret(value)))
    return value
```

**Kernel 上下文处理** (`tilelang/language/eager/builder.py:1253-1400`):
```python
def skip_kernel_ctx(self) -> bool:
    """检查是否跳过 Kernel 上下文。

    用于嵌套 Kernel 检测。
    """
    return self.frame is not None

def ctx_with(self, ctx: AbstractContextManager[Any]) -> AbstractContextManager[Any]:
    """With 语句上下文处理。

    支持 Kernel、Block 等上下文管理器。
    """
    if isinstance(ctx, KernelLaunchFrame):
        # Kernel 上下文特殊处理
        return ctx
    return ctx
```

**文件位置追踪** (`tilelang/language/eager/builder.py:1403-1450`):
```python
def set_fileline(self, filename: str, lineno: int, func_name: str):
    """设置当前代码位置信息。

    用于调试和错误报告。
    """
    self._current_filename = filename
    self._current_lineno = lineno
    self._current_func_name = func_name
```

**prim_func 装饰器** (`tilelang/language/eager/builder.py:1453-1500+`):
```python
def prim_func(
    func: Callable | None = None,
    *,
    is_kernel: bool = False,
    target: str = "cuda",
    target_host: str = "llvm",
    execution_backend: str = "dlpack",
    verbose: bool = False,
    skip_check: bool = False,
) -> JITFunc | Callable:
    """Eager 模式 prim_func 装饰器。

    将 Python 函数编译为可执行的 TIR 函数。

    Parameters
    ----------
    func : Callable
        待装饰的函数
    is_kernel : bool
        是否为 Kernel 函数（GPU）
    target : str
        目标平台
    execution_backend : str
        执行后端 (dlpack, torch, numpy)

    Returns
    -------
    JITFunc : 可执行的 JIT 函数
    """
    def decorator(func):
        # 1. 检查是否已包含内部 prim_func
        if has_internal_prim_func(func):
            return func

        # 2. 执行 AST 转换
        generator = mutate(func)

        # 3. 创建 JITFunc
        jit_func = JITFunc(
            generator=generator,
            is_kernel=is_kernel,
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            skip_check=skip_check,
        )
        return jit_func
```

### 5.4 JIT 执行

**JITFunc 类** (`tilelang/language/eager/builder.py`):
```python
class JITFunc:
    """JIT 编译函数包装器。

    管理代码生成、编译缓存和执行。
    """
    def __init__(self, generator: IRGenerator, ...):
        self.generator = generator
        self.is_kernel = is_kernel
        self.target = target
        self._compiled_func = None
        self._cache_key = None

    def __call__(self, *args, **kwargs):
        """执行 JIT 函数。

        1. 检查缓存
        2. 生成 TIR
        3. 编译
        4. 执行
        """
        # 构建缓存键
        cache_key = self._make_cache_key(args, kwargs)

        if cache_key != self._cache_key:
            # 需要重新编译
            self._compile(args, kwargs)
            self._cache_key = cache_key

        # 执行编译后的函数
        return self._compiled_func(*args, **kwargs)

    def _compile(self, args, kwargs):
        """编译函数到可执行代码。"""
        # 1. 创建 Builder
        builder = Builder()

        # 2. 生成 TIR
        inner_func = self.generator.gen(builder)
        inner_func(builder)

        # 3. 获取生成的 TIR 函数
        tir_func = builder.get()

        # 4. 编译
        self._compiled_func = compile(tir_func, self.target)
```

### 5.5 工具函数

**AST 获取** (`tilelang/language/eager/utils.py:58-66`):
```python
def get_ast(func: Callable):
    """获取函数的 AST。

    保留原始缩进和行号信息。
    """
    _, start = inspect.getsourcelines(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    source = inspect.getsource(func)
    source = _remove_leading_ident(source)
    source = "\n" * (start - 1) + source
    tree = ast.parse(source, filename=filename)
    return tree
```

**Nonlocal 变量获取** (`tilelang/language/eager/utils.py:34-56`):
```python
def get_func_nonlocals(func):
    """获取函数的 nonlocal 变量。

    修改版 inspect.getclosurevars。
    """
    code = func.__code__
    nonlocal_vars = {}
    if func.__closure__ is not None:
        for var, cell in zip(code.co_freevars, func.__closure__):
            try:
                nonlocal_vars[var] = cell.cell_contents
            except ValueError:
                pass
    return nonlocal_vars
```

**代码编译** (`tilelang/language/eager/utils.py:71-86`):
```python
def get_compiled_object(
    source: str | ast.AST,
    name: str,
    filename: str = None,
    globals: dict[str, Any] = None
):
    """编译源代码或 AST 为可执行对象。

    支持磁盘缓存以加速重复编译。
    """
    if isinstance(source, ast.AST):
        ast.fix_missing_locations(source)
        compiled = compile(source, filename, "exec")
    else:
        compiled = disk_compile(source, name)

    locs = {}
    exec(compiled, globals, locs)
    return locs[name]
```

**Stride 构造** (`tilelang/language/eager/utils.py:88-99`):
```python
def construct_strides(
    shape: tuple[Any, ...],
    allow_prim_expr: bool = True
) -> tuple[Any, ...]:
    """构造行优先 strides。

    从 shape 计算默认 strides，支持 PrimExpr。
    """
    strides = []
    stride = 1
    for s in shape[::-1]:
        strides.append(stride)
        stride *= s
        if not allow_prim_expr and isinstance(stride, tir.PrimExpr):
            raise ValueError("Cannot construct strides with PrimExpr")
    return tuple(reversed(strides))
```

## 6. 工作流程示例

### 6.1 Eager 模式执行流程

```python
from tilelang import language as T

@T.prim_func(is_kernel=True)
def my_kernel(A: T.Buffer((1024,), "float32"),
              B: T.Buffer((1024,), "float32")):
    # 1. AST 转换阶段
    # DSLMutator 将上述代码转换为：
    # def make_closure():
    #     def my_kernel(__tb):
    #         A = __tb.arg("A", A)
    #         B = __tb.arg("B", B)
    #         ...

    # 2. Builder 执行阶段
    for i in T.serial(1024):
        # 转换为：
        # for i in __tb.ctx_for(T.serial(1024)):
        B[i] = A[i] * 2.0
        # 转换为：
        # __tb.assign_slice(B, i, A[i] * 2.0)

    # 3. TIR 生成阶段
    # Builder 生成对应的 TIR 语句：
    # for i in tir.serial(0, 1024):
    #     tir.BufferStore(B, tir.Mul(tir.BufferLoad(A, [i]), 2.0), [i])
```

### 6.2 Parser 模式执行流程

```python
from tilelang.language import parser as T

@T.prim_func
def my_func(A: T.Buffer((1024,), "float32")):
    # 1. 解析阶段
    # Parser 遍历 AST 节点：
    # - visit_function_def: 创建 PrimFunc 框架
    # - 处理参数 A，创建 Buffer

    for i in T.serial(1024):
        # 2. visit_for 处理
        # - 评估 T.serial(1024) 得到 ForFrame
        # - 绑定迭代变量 i

        A[i] = 0.0
        # 3. visit_assign 处理
        # - 评估 rhs: 0.0
        # - 创建 BufferStore 语句
```

## 7. 关键设计决策

### 7.1 双模式设计

TileLang 提供两种代码生成模式：

1. **Parser 模式** (`tilelang/language/parser/`):
   - 基于 TVM Script Parser
   - 静态解析，适合库代码
   - 完整的类型检查

2. **Eager 模式** (`tilelang/language/eager/`):
   - 基于 AST 转换
   - 支持动态值和即时执行
   - 适合交互式开发和调试

### 7.2 AST 转换策略

Eager 模式使用 `DSLMutator` 将 Python 控制流转换为上下文管理器模式：

| Python 结构 | 转换后 |
|------------|--------|
| `if cond:` | `for br in __tb.ctx_if(cond):` |
| `for i in range:` | `for tmp in __tb.ctx_for(range):` |
| `a += b` | `__tb.aug_assign("Add", a, b)` |
| `x, y = a, b` | 两阶段绑定（解包 + 赋值） |

### 7.3 类型系统

TileLang 在 Python 类型之上构建了分层类型系统：

1. **Python 类型**: `int`, `float`, `bool`
2. **TIR 类型**: `PrimExpr`, `Var`, `Buffer`
3. **TileLang 类型**: `dtype`, `TensorProxy`, `StridedTensorProxy`
4. **引用类型**: `Ref`, `BufferRef`

### 7.4 缓存策略

Eager 模式实现多级缓存：

1. **AST 缓存**: `IRGenerator` 缓存转换后的生成器
2. **编译缓存**: `JITFunc` 基于参数类型缓存编译结果
3. **磁盘缓存**: `disk_compile` 缓存编译后的 Python 代码

## 8. 文件索引

| 文件路径 | 行数 | 主要内容 |
|---------|------|----------|
| `tilelang/language/ast/ir.py` | ~1500 | AST IR 定义、Buffer 操作、数学运算 |
| `tilelang/language/parser/parser.py` | ~586 | 核心解析逻辑、AST 节点处理 |
| `tilelang/language/parser/entry.py` | ~214 | prim_func/macro 装饰器 |
| `tilelang/language/parser/operation.py` | ~155 | 表达式操作注册 |
| `tilelang/language/tir/ir.py` | ~307 | TIR 循环构造器、操作包装 |
| `tilelang/language/tir/op.py` | ~1500+ | TIR 操作扩展 |
| `tilelang/language/tir/entry.py` | ~118 | TIR 入口函数 |
| `tilelang/language/eager/ast.py` | ~706 | DSLMutator、IRGenerator |
| `tilelang/language/eager/builder.py` | ~1500+ | Builder 类、JITFunc、prim_func |
| `tilelang/language/eager/utils.py` | ~99 | AST 工具、编译工具 |
