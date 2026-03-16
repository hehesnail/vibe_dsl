# TileLang Carver 模块详解

## 模块概述

Carver 是 TileLang 的高级分析和调度生成模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/carver/` 目录。该模块提供了针对 GPU 架构的自动调度生成和优化功能，特别针对矩阵乘法（GEMM）和 Tensor Core 进行了深度优化。

目录结构：
```
tilelang/carver/
├── __init__.py              # 模块导出
├── analysis.py              # TIR 块和循环分析
├── utils.py                 # 工具函数
├── matmul_analysis.py       # 矩阵乘法分析
├── common_schedules.py      # 通用调度策略
├── arch/                    # 架构支持
│   ├── __init__.py
│   ├── arch_base.py         # 基础架构类
│   ├── cuda.py              # CUDA 架构
│   ├── cdna.py              # AMD CDNA 架构
│   ├── cpu.py               # CPU 架构
│   ├── metal.py             # Metal 架构
│   └── driver/              # 驱动支持
├── roller/                  # Roller 调度生成
│   ├── __init__.py
│   ├── node.py              # 计算图节点
│   ├── hint.py              # 调度提示
│   ├── bestfit.py           # 最佳适配算法
│   ├── rasterization.py     # 栅格化
│   ├── policy/              # 调度策略
│   │   ├── common.py
│   │   ├── default.py
│   │   └── tensorcore.py
│   └── shape_inference/     # 形状推断
│       ├── common.py
│       └── tir.py
└── template/                # 内核模板
    ├── __init__.py
    ├── base.py              # 基础模板
    ├── matmul.py            # 矩阵乘法模板
    ├── gemv.py              # 矩阵向量乘法
    ├── conv.py              # 卷积模板
    ├── elementwise.py       # 逐元素操作
    ├── general_reduce.py    # 归约操作
    └── flashattention.py    # Flash Attention
```

核心功能：
- TIR 块和迭代变量分析
- GPU 架构特性抽象
- Tensor Core 自动检测和调度
- 矩阵乘法模式识别和优化
- 调度提示生成（Roller）

## 核心类详解

### 1. IterInfo

```python
class IterInfo:
    """Information about a loop/iter var."""
    kind: Literal["S", "R", "O"]  # Spatial, Reduction, Other
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: tir.schedule.LoopRV
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py:13-43`

表示循环/迭代变量的信息：
- `kind`: 迭代类型（S=空间，R=归约，O=其他）
- `var`: 循环变量
- `dom`: 迭代域
- `loop_rv`: Schedule 中的循环引用

### 2. BlockInfo

```python
class BlockInfo:
    """Information about a TIR block."""
    name: str
    iters: list[IterInfo]
    block_rv: tir.schedule.BlockRV
    _reduction_block: bool
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py:46-116`

TIR 块信息类，主要方法：
- `dom()`: 返回迭代域列表
- `dom_kind()`: 返回迭代类型字符串（如 "SSSR"）
- `is_injective()`: 是否为单射（所有迭代都是空间的）
- `is_elementwise()`: 是否为逐元素操作
- `is_reduction()`: 是否为归约块

### 3. normalize_prim_func

```python
def normalize_prim_func(sch: tir.Schedule) -> list[BlockInfo] | None:
    """Normalize the primfunc to normal form."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py:122-155`

将 PrimFunc 规范化为标准形式，返回 BlockInfo 列表。

### 4. IterKind / IterTrait (矩阵乘法分析)

```python
class IterKind(Enum):
    kIter_S = 0  # 空间轴（batch）
    kIter_I = 1  # I 轴（M 维度）
    kIter_J = 2  # J 轴（N 维度）
    kIter_K = 3  # K 轴（归约维度）
    kIter_T = 4  # 平凡轴（extent=1）

@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:186-208`

用于识别 GEMM 模式的迭代类型分类。

### 5. detect_iter_traits

```python
def detect_iter_traits(block: tir.Block) -> tuple[list[IterTrait]] | None:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:234-305`

检测块是否匹配 GEMM 模式，返回 A、B、C 的迭代特征。

### 6. get_index_map

```python
def get_index_map(block: tir.Block, layout: list[str] | None = None) -> tuple[tir.IndexMap, ...] | None:
    """Get index maps for the block."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:308-403`

为 GEMM 块生成索引映射，支持布局指定：
- `layout[0]`: A 矩阵布局（'n'=normal, 't'=transposed, 'a'=auto）
- `layout[1]`: B 矩阵布局
- `layout[2]`: C 矩阵布局

### 7. get_tensorized_func_and_tags

```python
def get_tensorized_func_and_tags(
    func: tir.PrimFunc,
    target: Target,
    layout: list[str] | None = None,
    skip_normalize: bool = False,
    allow_gemv: bool = False,
) -> tuple[tir.PrimFunc, dict[str, list[int] | int]]:
    """Transform function to matmul if necessary and detect tensorcore tags."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:510-665`

核心函数，执行以下步骤：
1. 检测是否可以使用 Tensor Core
2. 将函数规范化为矩阵乘法形式
3. 分析 Tensor Core 配置标签

**返回的标签**:
```python
{
    "tensorcore_config": [out_axis - 2, out_axis - 1],  # 使用 Tensor Core 的轴
    "pipeline_stage": 1 or 2,  # 流水线阶段数
    "use_async_copy": True or False,  # 是否使用异步拷贝
    "intrin_info": {
        "in_dtype": "float16",
        "out_dtype": "float32",
        "trans_b": True or False,
    }
}
```

### 8. normalize_to_matmul

```python
def normalize_to_matmul(sch: tir.Schedule, main_block: BlockRV, layout: list[str] | None = None) -> tir.Schedule | None:
    """Normalize tensor functions to C[S, I, J] += A[S, I, K] * B[S, J, K]."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:485-507`

将张量函数规范化为标准 GEMM 形式：
1. 重新索引输入缓冲区
2. 应用布局变换
3. 变换块布局

### 9. get_roller_hints_from_func

```python
def get_roller_hints_from_func(
    func_or_module: tir.PrimFunc | IRModule,
    arch: TileDevice,
    topk: int = 10,
    tensorcore_only: bool = False,
    allow_gemv: bool = False
) -> list[Hint] | None:
    """Generate Roller hints from a PrimFunc."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/utils.py:29-65`

从 PrimFunc 生成 Roller 调度提示：
1. 检测 Tensor Core 适用性
2. 选择策略（TensorCorePolicy 或 DefaultPolicy）
3. 生成 top-k 调度配置

### 10. try_inline / try_inline_contiguous_spatial

```python
def try_inline(sch: tir.Schedule, blocks: list[BlockInfo]) -> list[BlockInfo]:
    """Try to inline as many blocks as possible."""

def try_inline_contiguous_spatial(sch: tir.Schedule, block_infos: list[BlockInfo]) -> list[BlockInfo]:
    """Try to inline contiguous spatial blocks."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/common_schedules.py:91-164`

块内联优化：
- `try_inline`: 尝试 compute_inline 和 reverse_compute_inline
- `try_inline_contiguous_spatial`: 仅内联连续的空间块

## 实现逻辑分析

### GEMM 模式识别流程

```
输入 TIR Block
      │
      ▼
┌─────────────────────┐
│ detect_iter_traits  │
│ - 检查 reads=2, writes=1
│ - 识别 I, J, K 轴    │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 检查是否为 GEMM 模式 │
│ - 需要 I, J, K 三种轴
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ make_iter_fusion    │
│ _index_map          │
│ - 创建索引映射       │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ normalize_to_matmul │
│ - reindex           │
│ - transform_layout  │
│ - transform_block_layout
└─────────────────────┘
```

### Tensor Core 检测逻辑

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:543-632`

```python
def analysis_tensorcore_tags(sch: tir.Schedule, block: BlockRV, target: Target):
    # 1. 检查 SM 版本
    if check_sm_version(target.arch) < 70:
        return False

    # 2. 检查数据类型支持
    in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
    if not is_tensorcore_supported_precision(in_dtype, out_dtype, arch):
        return None

    # 3. 分析 Tensor Core 配置
    tags["tensorcore_config"] = [out_axis - 2, out_axis - 1]

    # 4. 分析流水线阶段
    if target.kind.name == "cuda" and check_sm_version(target.arch) in {80, 90}:
        tags["pipeline_stage"] = 2
        tags["use_async_copy"] = True

    # 5. 分析内联信息
    tags["intrin_info"] = {
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
        "trans_b": check_last_trait(block_stmt.reads[1].region),
    }
```

### 布局传播

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:781-829`

```python
def layout_propagate_chain(sch, start_block, start_buffer, end_block, index_map):
    """Propagate layout transformation through a chain of blocks."""
    while True:
        producers = sch.get_producers(block)
        if len(producers) == 0:
            break
        for producer in producers:
            # 反向传播索引映射
            tmp_index_map = IndexMap(write_indices, read_indices, None)
            tmp_index_map = tmp_index_map.non_surjective_inverse(write.buffer.shape)[0]
            # 组合映射
            final_indices = index_map.map_indices(tmp_index_map.map_indices(write_indices))
            index_map = IndexMap(write_indices, final_indices, None)
```

### LDMATRIX 布局映射

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py:668-721`

```python
def get_propagate_map(trans: bool = True, dtype="float16", matrix_name="A", index_dtype="int32"):
    """Get index map for LDMATRIX layout propagation."""
    # 根据数据类型选择布局
    if dtype in ["bfloat16", "float16"]:
        ldmatrix_layout = ldmatrix_32x8_to_shared_16x16_layout
        ldmatrix_layout_trans = ldmatrix_trans_32x8_to_shared_16x16_layout
    elif dtype in ["int8", "float8_e4m3", "float8_e5m2"]:
        if matrix_name == "A" and trans is False:
            ldmatrix_layout = ldmatrix_32x16_to_shared_16x32_layout_a
        elif matrix_name == "B" and trans is True:
            ldmatrix_layout = ldmatrix_32x16_to_shared_16x32_layout_b
```

## 使用示例

### 分析 TIR 函数

```python
from tilelang.carver import normalize_prim_func, BlockInfo, IterInfo
from tilelang.carver.analysis import get_reduction_blocks

# 创建 Schedule
sch = tir.Schedule(my_prim_func)
root_block = sch.get_block("root")
blocks = sch.get_child_blocks(root_block)

# 规范化并获取 BlockInfo
block_infos = normalize_prim_func(sch)
for info in block_infos:
    print(f"Block: {info.name}, Kind: {info.dom_kind()}, Domain: {info.dom()}")

# 获取归约块
reduction_blocks = get_reduction_blocks(sch, blocks)
```

### 检测 GEMM 模式

```python
from tilelang.carver.matmul_analysis import (
    detect_iter_traits,
    get_index_map,
    get_tensorized_func_and_tags,
    normalize_to_matmul,
)

# 检测迭代特征
traits = detect_iter_traits(block_stmt)
if traits:
    A_traits, B_traits, C_traits, block_traits = traits
    print("Detected GEMM pattern!")

# 获取索引映射
index_maps = get_index_map(block_stmt, layout=["n", "t", "n"])
if index_maps:
    matmul_map, A_map, B_map, C_map = index_maps

# 检测 Tensor Core 支持
tensorized_func, tags = get_tensorized_func_and_tags(
    func, target, layout=["a", "a", "a"]
)
if tags:
    print(f"Tensor Core config: {tags['tensorcore_config']}")
    print(f"Pipeline stages: {tags['pipeline_stage']}")
```

### 获取 Roller 提示

```python
from tilelang.carver import get_roller_hints_from_func
from tilelang.carver.arch import CUDA

# 创建架构对象
arch = CUDA(target)

# 生成调度提示
hints = get_roller_hints_from_func(
    func,
    arch=arch,
    topk=10,
    tensorcore_only=True,
    allow_gemv=False,
)

for hint in hints:
    print(f"Config: {hint.config}, Estimated latency: {hint.latency}")
```

### 块内联优化

```python
from tilelang.carver import try_inline, try_inline_contiguous_spatial
from tilelang.carver.common_schedules import get_output_blocks

# 获取块信息
block_infos = normalize_prim_func(sch)

# 尝试内联所有可能的块
remaining = try_inline(sch, block_infos)

# 或仅内联连续的空间块
remaining = try_inline_contiguous_spatial(sch, block_infos)

# 获取输出块
output_blocks = get_output_blocks(sch, block_infos)
```

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| IterInfo 类 | `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py` | 13-43 |
| BlockInfo 类 | `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py` | 46-116 |
| normalize_prim_func | `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py` | 122-155 |
| get_reduction_blocks | `/root/dev/vibe_dsl/tilelang/tilelang/carver/analysis.py` | 260-281 |
| IterKind / IterTrait | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 186-208 |
| detect_iter_traits | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 234-305 |
| get_index_map | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 308-403 |
| normalize_to_matmul | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 485-507 |
| get_tensorized_func_and_tags | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 510-665 |
| get_propagate_map | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 668-721 |
| layout_propagate_chain | `/root/dev/vibe_dsl/tilelang/tilelang/carver/matmul_analysis.py` | 781-829 |
| try_inline | `/root/dev/vibe_dsl/tilelang/tilelang/carver/common_schedules.py` | 91-126 |
| try_inline_contiguous_spatial | `/root/dev/vibe_dsl/tilelang/tilelang/carver/common_schedules.py` | 129-164 |
| get_roller_hints_from_func | `/root/dev/vibe_dsl/tilelang/tilelang/carver/utils.py` | 29-65 |
| get_rasterization_code | `/root/dev/vibe_dsl/tilelang/tilelang/carver/utils.py` | 14-26 |
