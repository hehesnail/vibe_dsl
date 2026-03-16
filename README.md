# TileLang Blackhole 后端扩展项目

为 TileLang DSL 添加 Tenstorrent Blackhole 硬件后端支持，基于 TT-Metal 底层框架实现编译器后端。

## 架构概览

```
TileLang DSL (Python)
       ↓ LowerAndLegalize
TIR (TVM IR)
       ↓ Blackhole Passes (AssignCores/PlanCB/SplitKernel)
TT-Metal C++ (BRISC/TRISC/NCRISC)
       ↓ JIT Build (libtt_metal.so)
RISC-V ELF
       ↓ Runtime
Blackhole Hardware (140 cores, 14x10 grid)
```

## 核心设计原则

- **Blackhole 专用**：专注 14x10 grid, 64 CBs, 1.5MB L1 架构
- **DSL 零修改**：同一套 TileLang 代码编译到不同后端
- **动态链接**：通过 libtt_metal.so 调用 TT-Metal API
- **TT-Sim 验证**：无硬件条件下使用仿真器验证

```python
# 同一套 DSL 代码，编译到不同硬件
@T.prim_func
def gemm_kernel(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32)) as (bx, by):
        A_shared = T.alloc_shared((32, 32), "float16")
        B_shared = T.alloc_shared((32, 32), "float16")
        C_local = T.alloc_fragment((32, 32), "float32")

        T.copy(A[by*32:(by+1)*32, :], A_shared)
        T.copy(B[:, bx*32:(bx+1)*32], B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*32:(by+1)*32, bx*32:(bx+1)*32])

# target = "cuda"      → CUDA kernel
# target = "blackhole" → Blackhole R/C/W kernels
```

## 核心映射关系

| TileLang DSL | GPU (CUDA) | Blackhole |
|--------------|------------|-----------|
| `T.Kernel(grid_x, grid_y)` | Grid/Block 硬件调度 | **14x10 Core Grid 软件切分** |
| `bx, by` | `blockIdx.x/y` | **Core (X,Y) ∈ [0,13]x[0,9]** |
| `T.alloc_shared` | `__shared__` | **Circular Buffer (CB 0-63)** |
| `T.copy` | 隐式/异步拷贝 | **显式 NoC 读写** |
| 同步 | `__syncthreads()` | **CB push/pop** |

## 项目结构

```
.
├── CLAUDE.md              # 工作流程入口（Agent 自动加载）
├── docs/                  # 技术文档
│   ├── tilelang/          # TileLang 架构分析
│   └── tt_metal/          # TT-Metal 源码分析
├── memory/                # 知识管理
│   ├── general_dev.md     # 开发经验总结
│   └── bugs.md            # 问题与解决方案
├── tasks/                 # 任务管理
│   ├── arch_design.md     # 总体架构设计
│   ├── progress.md        # 任务状态看板
│   └── dev_design/        # 各任务详细设计
├── tilelang_repo/         # TileLang 源码（⚠️ 自行维护的修改版）
│                           # 用于开发 Blackhole 后端，不向原仓库提交
└── tt_metal_repo/         # TT-Metal 源码（⚠️ 自行维护的修改版）
                            # 编译生成 libtt_metal.so，不向原仓库提交
```

## 开发阶段

| 阶段 | 目标 | 关键产出 |
|------|------|----------|
| Phase 0 | 环境准备 | TileLang/TT-Metal/TT-Sim 编译完成 |
| Phase 1 | CodeGen 框架 | 单核 Copy 在 TT-Sim 运行 |
| Phase 2 | R/C/W 拆分 | 140 核并行 Copy |
| Phase 3 | GEMM 支持 | 矩阵乘法正确性验证 |
| Phase 4 | 性能优化 | 自动 tile size、内存优化 |

## 快速开始

1. **查阅当前任务**：阅读 [`tasks/progress.md`](tasks/progress.md)
2. **了解架构**：阅读 [`tasks/arch_design.md`](tasks/arch_design.md)
3. **领取任务**：按工作流程（`CLAUDE.md`）推进开发

## 关键文档

- [架构设计](tasks/arch_design.md) - Lowering Pipeline、硬件映射、测试策略
- [开发进度](tasks/progress.md) - 当前阶段与任务清单
- [TT-Metal 分析](docs/tt_metal/source_analysis/) - JIT 编译、内存模型、硬件规格

## 技术栈

- **编译器**：TVM / TileLang / MLIR
- **运行时**：TT-Metal (libtt_metal.so)
- **仿真**：TT-Sim (libttsim_bh.so)
- **测试**：gtest (C++) / pytest (Python)
- **CI/CD**：GitHub Actions
