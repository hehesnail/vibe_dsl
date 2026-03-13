# TileLang Tenstorrent 后端扩展项目

为 TileLang DSL 添加 Tenstorrent 硬件后端支持，基于 TT-Metal 底层框架实现编译器后端。

## 架构概览

```
TileLang DSL (Python)
       ↓
TIR (TVM IR)
       ↓
TT-Metal CodeGen → TT-Metal Runtime
       ↓
Tenstorrent Hardware (Grayskull/Wormhole/Blackhole)
```

## 核心设计原则

**DSL 100% 兼容**：同一套 TileLang 代码可编译到不同硬件，差异完全下沉到 Lowering Pipeline。

```python
# 同一套 DSL 代码，编译到不同硬件
@T.prim_func
def gemm_kernel(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_K, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by*block_M, k*block_K], A_shared)
            T.copy(B[k*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])

# target = "cuda"       → CUDA kernel
# target = "tenstorrent" → Tenstorrent R/C/W kernels
```

## 核心映射关系

| TileLang DSL | GPU (CUDA/HIP) | Tenstorrent |
|--------------|----------------|-------------|
| `T.Kernel(grid_x, grid_y)` | Grid/Block 硬件调度 | **Core Grid 软件切分** |
| `bx, by` (block indices) | `blockIdx.x/y` | **Core (X,Y) 坐标** |
| `T.alloc_shared(...)` | `__shared__` memory | **Circular Buffer (CB)** |
| `T.copy` | 隐式/异步拷贝 | **显式 NoC 读写** |
| 同步机制 | `__syncthreads()` | **CB push/pop + barrier** |

## 开发阶段

### Phase 1: 基础框架 - 单核支持
- [ ] Kernel 拆分：`unified → Reader/Compute/Writer`
- [ ] CB 分配规划
- [ ] 代码生成器实现
- [ ] Runtime 模块

### Phase 2: 多核任务切分
- [ ] Grid 到 Core (X,Y) 映射
- [ ] 硬件信息抽象 (Blackhole/Wormhole)
- [ ] Host 端代码生成

### Phase 3: TT-Metal 集成
- [ ] JIT 编译系统对接
- [ ] Python 层集成
- [ ] 端到端执行验证

### Phase 4-5: 计算支持与优化
- [ ] GEMM / SFPU 指令
- [ ] 自动 tile size 选择
- [ ] 性能优化

## 项目结构

```
.
├── docs/
│   ├── tilelang/          # TileLang 架构文档
│   └── tt_metal/          # TT-Metal 源码分析
├── memory/
│   ├── bugs.md            # 问题与解决方案
│   └── general_dev.md     # 开发模式与最佳实践
├── tasks/
│   ├── arch_design.md     # 详细架构设计
│   └── progress.md        # 开发进度跟踪
├── tilelang_repo/         # TileLang 源码
└── tt_metal_repo/         # TT-Metal 源码
```

## 关键文档

- [详细架构设计](tasks/arch_design.md) - Lowering Pipeline、文件结构、实现规划
- [开发进度](tasks/progress.md) - 当前阶段与任务清单
- [TileLang 后端架构](docs/tilelang/cpp_core/04_target_backends.md) - 现有后端参考
- [TT-Metal 源码分析](docs/tt_metal/source_analysis/) - JIT 编译、内存模型

## 开发规范

1. **渐进式实现**：先跑通端到端，再优化性能
2. **复用现有基础设施**：基于 TVM 代码生成框架
3. **知识管理**：开发前查阅 `memory/`，遇到问题时记录到 `memory/bugs.md`
4. **代码风格**：遵循 TVM/TileLang 命名规范

## 快速开始

查阅 `tasks/progress.md` 了解当前未完成任务，按阶段顺序推进开发。
