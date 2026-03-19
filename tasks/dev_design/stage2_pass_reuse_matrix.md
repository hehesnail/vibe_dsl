# Stage 2 Pass 复用矩阵设计

## 目标

- 按 TileLang / CUDA 正式调用链重新定义 Blackhole 的 pass 接入方式
- 不再从旧 Blackhole 实现反推“还剩哪些 pass 没补”
- 明确：
  - 哪些 pass 直接复用
  - 哪些只复用职责，不直接复用实现
  - 哪些不进入 Blackhole 主线
  - Blackhole 新增接入点分别位于哪个阶段

## 核心结论

### 1. Blackhole 的 pass 主线必须按三层组织

当前正确主线应收敛为：

```text
DSL
  -> LowerAndLegalize
  -> split 前 Blackhole 语义规划
  -> AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch
  -> split 后 Blackhole 正式 plan 提取
  -> rt_mod_blackhole / BlackholeModule direct host path
```

### 2. 不能再把“memory plan / execution plan”混成一个晚期 matcher 问题

Blackhole 需要显式区分：

- split 前语义规划：
  - 保留 tile/dataflow/shared/block 语义
- split 后正式 plan 提取：
  - `runtime_args`
  - `cb_requirements`
  - `cb_configs`
  - `core_plan`

### 3. `LowerTileOp` 和 split 后 Blackhole passes 是两个不同职责区

- `LowerTileOp`：
  split 前语义接入点
- `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores`：
  split 后正式 plan 提取层

## Pass 复用矩阵

### A. 前端合法化 / 语义归一化

这层全部继续直接复用：

| pass | 目标状态 | 说明 |
|------|----------|------|
| `BindTarget` | `reuse` | 通用 target 绑定 |
| `AddWrapperForSingleBufStore` | `reuse` | 前端形状归一化 |
| `LegalizeNegativeIndex` | `reuse` | 通用索引合法化 |
| `VerifyParallelLoop` | `reuse` | 通用语义检查 |
| `InjectAssumes` | `reuse` | 参数/边界假设 |
| `Simplify` | `reuse` | 通用规范化 |
| `LayoutReducer` | `reuse` | 布局推导前置 |
| `LayoutInference` | `reuse` | tile/dataflow 布局基础 |
| `LowerL2Persistent` | `reuse` | 与 Blackhole 主问题正交 |
| `DecoupleTypeCast` | `reuse` | 表达式合法化 |
| `LegalizeVectorizedLoop` | `reuse` | 通用 loop 合法化 |
| `LegalizeSafeMemoryAccess` | `reuse` | 边界安全检查 |
| `LowerAccessPtr` | `reuse` | 指针 metadata 到标准 TIR |
| `HoistNonRestrictParams` | `reuse` | host/device 参数基础 |

### B. split 前 Blackhole 语义规划

| pass / 层 | 目标状态 | 说明 |
|-----------|----------|------|
| `LowerTileOp` | `reuse_with_blackhole_branch` | Blackhole 最关键接入点；负责保留 copy/gemm 的可规划语义 |
| `IfStmtBinding` | `reuse_or_branch` | 保留其规划职责，按需 target-aware |
| `PlanAndUpdateBufferAllocationLocation` | `reuse_or_branch` | 保留其 memory/dataflow planning 职责 |
| `PipelinePlanning` | `reuse_or_branch` | 未来 copy/gemm dataflow 规划的主要承载点 |
| `InjectSoftwarePipeline` | `reuse_or_branch` | 保留 pipeline planning 职责，不要求照搬 CUDA 实现 |
| `LowerOpaqueBlock` | `reuse` | 结构规范化 |
| `VerifyMemory` | `reuse` | 通用内存合法性检查 |
| `AnnotateEntryFunc` | `reuse` | entry 语义统一 |

这一层的输出不是正式 host/runtime 协议，而是：

- `Blackhole-preserving TIR`

### C. 通用中后段规范化 / 优化

这层不是先验“全复用”，而是按 split 前语义是否还能保真来逐项接回：

| pass | 目标状态 | 说明 |
|------|----------|------|
| `NarrowDataType` | `reuse_if_safe` | 通用 dtype 规范化 |
| `FlattenBuffer` | `reuse_if_safe` | 需以 copy/gemm 语义保真为前提 |
| `ConfigIndexBitwidth` | `reuse_if_safe` | index 统一 |
| `VectorizeLoop` | `reuse_if_safe` | 是否启用可 target-gate |
| `StorageRewrite` | `reuse_if_safe` | 当前最容易打断 copy 识别 |
| `LoopUnswitching` | `reuse_if_safe` | 通用 loop 优化 |
| `UnrollLoop` | `reuse_if_safe` | 通用 loop 展开 |
| `RenormalizeSplitPattern` | `reuse_if_safe` | 规范化循环分裂模式 |
| `RemoveNoOp` | `reuse_if_safe` | 通用清理 |
| `HoistIfThenElse` | `reuse_if_safe` | 控制流规范化 |

接回原则：

- 不以兼容旧 Blackhole 实现为准
- 只要 split 前语义规划仍能保真，就应接回

### D. host/device 与调用约束

这层全部作为正式主线复用：

| pass | 目标状态 | 说明 |
|------|----------|------|
| `AnnotateDeviceRegions` | `reuse` | host/device 边界识别 |
| `SplitHostDevice` | `reuse` | 不能再绕开 |
| `AnnotateReadOnlyParams` | `reuse` | 参数约束与 ABI 一致性 |
| `MakePackedAPI` | `reuse` | 正式 host callable 入口 |
| `LowerDeviceKernelLaunch` | `reuse` | 正式 launch ABI 与 `thread_extent` 来源 |

### E. split 后 Blackhole 正式 plan 提取

这层是 Blackhole 真正新增的正式主线，不属于 CUDA 直接复用部分。

| pass / 层 | 目标状态 | 说明 |
|-----------|----------|------|
| `LowerBlackholeOps` | `keep_and_refocus` | split 后 requirement extraction：`segment_plan` / `runtime_args` / `cb_requirements` |
| `PlanBlackholeCB` | `keep_and_expand` | split 后 memory planner：生成正式 `cb_configs` |
| `AssignBlackholeCores` | `keep_and_expand` | split 后 execution planner：生成正式 `core_plan` |

### F. 不进入 Blackhole 主线的 CUDA/Hopper 专属实现

| pass | 目标状态 | 说明 |
|------|----------|------|
| `WarpSpecialized` | `do_not_reuse_impl` | CUDA warp 专属 |
| `InjectTmaBarrier` | `do_not_reuse_impl` | Hopper/TMA 专属 |
| `MultiVersionBuffer` | `do_not_reuse_impl` | 与当前 Blackhole 执行模型不对齐 |
| `OptimizeCPAsyncSync` | `do_not_reuse_impl` | CUDA cp.async 专属 |
| `RewriteWgmmaSync` | `do_not_reuse_impl` | WGMMA 专属 |
| `LowerThreadAllreduce` | `do_not_reuse_impl` | 线程归约模型不对齐 |
| `LowerLDGSTG` | `do_not_reuse_impl` | Nvidia load/store 降法 |
| `LowerHopperIntrin` | `do_not_reuse_impl` | Hopper 指令专属 |
| `InjectFenceProxy` | `do_not_reuse_impl` | async proxy 专属 |
| `AnnotateWarpGroupRegAlloc` | `do_not_reuse_impl` | warp-group reg alloc 专属 |
| `MarkCudaSyncCalls` | `do_not_reuse_impl` | CUDA PDL 专属 |
| `PersistThreadblock` | `do_not_reuse_impl` | GPU persistent block 专属 |

## Blackhole 新增接入点

### 接入点 1: split 前语义规划

位置：

- `LowerTileOp` 的 Blackhole-aware branch

职责：

- 保留 copy/gemm/tile/shared/pipeline 语义
- 输出 `Blackhole-preserving TIR`

### 接入点 2: split 后 requirement extraction

位置：

- `LowerBlackholeOps`

职责：

- 提取：
  - `blackhole.segment_plan`
  - `blackhole.runtime_args`
  - `blackhole.cb_requirements`

### 接入点 3: split 后 memory planning

位置：

- `PlanBlackholeCB`

职责：

- 生成正式 `blackhole.cb_configs`
- 做 `cb_id` deterministic 分配、生命周期复用、1.5MB hard check

### 接入点 4: split 后 execution planning

位置：

- `AssignBlackholeCores`

职责：

- 生成正式 `blackhole.core_plan`
- 补足 logical grid / work packets / physical core mapping

### 接入点 5: host-side materialization

位置：

- `BlackholeModule`

职责：

- 直接 materialize TT-Metal host objects
- 不再通过 external runner 作为主路径

## 基于矩阵的改造顺序

### 第一步：固定三层模型

- split 前语义规划
- split 后正式 plan 提取
- host-side materialization

### 第二步：把差异收缩到四个接入点

- `LowerTileOp`
- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`

### 第三步：用 copy 打通首条正式主链

copy 的正式完成标准不是“能跑”，而是：

- 经过正式 TIR / host-device 主链
- 经过 split 后 Blackhole 正式 plan 提取
- 通过 `BlackholeModule` direct host path 执行

### 第四步：在不破坏 copy 主链的前提下接回通用 pass

- `FlattenBuffer`
- `VectorizeLoop`
- `StorageRewrite`

### 第五步：在同一结构上接入 GEMM

- 不再新增 runtime-only 或 runner-only 路径

## 验证方式

### 文档一致性

以下文档必须保持一致：

- `final_blackhole_backend_redesign.md`
- `stage2_single_core_pass_integration.md`
- `stage2_blackhole_logical_block_launch_plan.md`
- `stage2_concrete_dev_task_plan.md`
- `progress.md`

### 结构验证

- split 前仍有 `Blackhole-preserving TIR`
- split 后稳定产出：
  - `blackhole.segment_plan`
  - `blackhole.runtime_args`
  - `blackhole.cb_requirements`
  - `blackhole.cb_configs`
  - `blackhole.core_plan`

### 正式执行验证

正式执行只看：

- TileLang 暴露的正式 host callable
- `BlackholeModule` direct host path

external runner 不再作为正式阶段完成标准
