# TileLang AutoDD 模块详解

## 模块概述

AutoDD（Automatic Delta Debugging）是 TileLang 的自动调试模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py`。该模块实现了基于 AST（抽象语法树）的代码简化技术，用于自动缩小导致特定错误的代码范围。

核心功能：
- 基于概率的 Delta Debugging 算法（PDD）
- AST 模式匹配与重写
- 并行任务管理
- 代码执行验证

## 核心类详解

### 1. ASTRewrite (抽象基类)

```python
class ASTRewrite(ABC):
    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def match(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> bool: ...

    @abstractmethod
    def rewrite(self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool) -> "ast.AST | list[ast.AST] | None": ...
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:40-51`

所有 AST 重写规则的基类，定义了匹配和重写接口。

### 2. GeneralRemove

```python
@dataclass
class GeneralRemove(ASTRewrite):
    name: str
    target_type: type[ast.AST]
    inside_list: bool = True
    replace_with: "ast.AST | list[ast.AST] | None" = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:54-71`

通用移除规则，用于移除特定类型的 AST 节点。

### 3. ASTPatRewrite

```python
@dataclass
class ASTPatRewrite(ASTRewrite):
    name: str
    match_pat: ASTPat
    rewrite_pat: ASTPat
    checker: "Callable[[dict[str, ast.AST]], bool] | dict[str, Callable[[ast.AST], bool]] | None" = None
    derived: "dict[str, Callable[[dict[str, ast.AST]], ast.AST]] | None" = None
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:279-336`

基于模式的 AST 重写类，支持从代码字符串创建模式：

```python
# 示例：创建 for 循环绑定到 0 的重写规则
for_bind_0 = ASTPatRewrite.from_code(
    name="for-bind-0",
    kind="stmt",
    match="for VARS in EXPR: BODY",
    rewrite="VARS = ZEROS\nBODY",
    placeholders={"VARS", "EXPR", "BODY", "ZEROS"},
    derived={
        "ZEROS": lambda ph: expr_to_zeros(ph["VARS"]),
    },
)
```

### 4. PDD (Probabilistic Delta Debugging)

```python
class PDD:
    def __init__(self, all_labels: list[int], init_proba: float = 0.93):
        self.all_labels = all_labels
        self.probas = {label: init_proba for label in all_labels}
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:466-518`

概率性 Delta Debugging 核心类，实现算法：

1. **初始化**: 为每个重写标签设置初始概率（默认 0.93）
2. **生成任务**: 根据概率选择要应用的重写规则
3. **更新概率**: 根据执行结果更新概率
   - 如果产生"有趣"结果（保留目标错误），相关规则概率设为 1.0
   - 否则降低概率

### 5. ASTPDD

```python
class ASTPDD(TaskManager, PDD):
    def __init__(self, tree: ast.AST, rewrites: list[ASTRewrite], init_proba: float = 0.93):
        self.tree, _, total_rewrites = attach_rewrites(tree, rewrites)
        all_labels = [i for i in range(total_rewrites)]
        super().__init__(all_labels, init_proba)
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:532-563`

结合 AST 重写和 PDD 的任务管理器。

### 6. ParTaskManager

```python
@dataclass
class ParTaskManager:
    err_msg: str
    text: str
    output_file: Path
    timeout: int = 60
    num_workers: int = 1
    backend: JobBackend = "runner"
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:884-1001`

并行任务管理器，支持多工作进程并发执行代码验证。

## 实现逻辑分析

### Delta Debugging 流程

1. **代码解析**: 使用 Python `ast` 模块解析源代码
2. **重写规则附加**: `attach_rewrites()` 遍历 AST，为每个节点附加匹配的重写规则
3. **任务生成**: `PDD.generator()` 根据概率生成候选任务
4. **并行执行**: `ParTaskManager` 使用多进程执行代码验证
5. **结果反馈**: 根据执行结果更新概率，迭代优化

### 重写规则分类

**快速简化器** (`/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:1091-1097`):
```python
fast_reducers = [
    if_remover_1,    # 移除 if 条件，保留 body
    if_remover_2,    # 移除 if-else 中的条件，保留 if body
    if_remover_3,    # 移除 if-else 中的条件，保留 else body
    for_bind_0,      # for 循环变量绑定为 0
    GeneralRemove("stmt-remover", ast.stmt, replace_with=ast.Pass()),
]
```

**规范化器** (`/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:1100-1103`):
```python
canonicalizers = [
    with_bind_0,     # with 语句变量绑定
    AttachFullFuncArgs(),  # 附加完整函数参数
]
```

**简化器** (`/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:1106-1112`):
```python
simplifiers = [
    assign_rhs_1,    # 赋值右侧替换为 1
    CallFwdArg1(),   # 函数调用保留第一个参数
    BinOpFwdArg("left"),   # 二元操作保留左操作数
    BinOpFwdArg("right"),  # 二元操作保留右操作数
]
```

### AsyncPythonRunner

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py:681-765`

异步 Python 代码执行器，特点：
- 使用独立进程执行代码（避免污染主进程）
- 支持超时控制
- 捕获 stdout/stderr
- 进程生命周期管理

## 使用示例

### 命令行使用

```bash
python -m tilelang.autodd source.py \
    --err-msg "TargetError" \
    -o output.py \
    --backend runner \
    --timeout 60 \
    -j 4
```

### 程序化使用

```python
import ast
from tilelang.autodd import ASTPDD, GeneralRemove

# 解析源代码
source = """
def buggy_func():
    x = 1
    y = 2  # 可能可以移除
    return x + y
"""
tree = ast.parse(source)

# 定义重写规则
rewrites = [
    GeneralRemove("stmt-remover", ast.Assign, replace_with=ast.Pass())
]

# 创建 PDD 实例
pdd = ASTPDD(tree, rewrites)

# 生成简化任务
for task in pdd.task_generator():
    print(task.source)
    # 验证简化后的代码是否仍产生目标错误
    # ...
```

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| ASTRewrite 基类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 40-51 |
| GeneralRemove 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 54-71 |
| ASTPatRewrite 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 279-336 |
| PDD 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 466-518 |
| ASTPDD 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 532-563 |
| ParTaskManager 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 884-1001 |
| AsyncPythonRunner 类 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 681-765 |
| attach_rewrites 函数 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 394-398 |
| apply_rewrites 函数 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 442-445 |
| main 函数 | `/root/dev/vibe_dsl/tilelang/tilelang/autodd.py` | 1012-1138 |
