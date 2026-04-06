# Stage 4 Phase B: Spatial Program IR

## 基本信息

- **文档角色**: `Phase B` 实施与设计边界文档
- **当前状态**: 当前主实施阶段；`2026-04-06` 已完成首轮落地：
  `SpatialProgram / ProgramPhase`、copy/GEMM fast-path、`flash-attn` multi-phase gate、
  以及 `LowerBlackholeOps` 对 spatial summary 的最小接线均已进入主链
- **上游输入**: 冻结后的 `SemanticProgram`
- **下游输出**: 冻结后的 `SpatialProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. Phase B 的职责

`Phase B` 只负责一件事：

- 把已经冻结的 algorithmic truth 组织成稳定的 spatial/dataflow program structure

它回答的问题是：

- 哪些 `Update` 应该组织成哪些 `Task`
- 哪些 `State` 之间要形成哪些 `Channel`
- 哪些 `Layout / WorkPartition / SyncEdge / Placement / ResourceIntent` 必须显式存在
- 多 `T.Kernel` / 多 device member 程序的 `ProgramPhase` 边界在哪里

它不负责：

- 再做一次 semantic recovery
- 发明 TT resource / CB / semaphore / ABI
- 从 raw TIR 或 late builtin 重新猜 `carry / selection / recurrence / source_states`

## 2. Core Design Boundary

### 2.1 核心对象

`Phase B` 的长期 core object set 是：

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

### 2.2 ProgramPhase 的稳定宿主

`ProgramPhase` 的跨函数真相固定挂在：

- `IRModule.global_infos["tl.device_programs"]`

而：

- `PrimFunc.attrs["tl.spatial_program"]`

只保留 member-local spatial truth。

因此：

- 单 `PrimFunc` 程序只是 `member_funcs.size() == 1` 的退化情况
- 任何 cross-function 的 phase order、shared buffer、global sync 都不能由单个 `PrimFunc` 自己发明

### 2.3 小闭集 family

`Phase B` 仍然遵守 small-closed family 设计：

- `Task.kind`
  - `transfer`
  - `compute`
  - `collective`
  - `control`
- `Layout.kind`
  - `regular`
  - `packed`
  - `indexed`
- `WorkPartition.kind`
  - `replicated`
  - `blocked`
  - `indexed`
  - `filtered`
- `Placement.kind`
  - `execution`
  - `communication`
  - `phase_boundary`
- `SyncEdge.kind`
  - `dependency`
  - `barrier`
  - `completion`
- `ResourceIntent.kind`
  - `buffer`
  - `state_residency`
  - `synchronization_support`
  - `phase_boundary_materialization`

更细差异通过 bindings 与 typed traits 表达，不通过 workload noun 扩 schema。

### 2.4 Legality 与 Policy 的边界

`Phase B` 必须明确区分：

- **analysis / legality facts**
  - 哪些 cut 是必须的
  - 哪些 state flow 必须显式化
  - 哪些 phase boundary 必须物化
  - 哪些 layout / partition family 是被语义强制的
- **policy decisions**
  - 合法空间内的 fusion / split 偏好
  - canonical fast-path 选择
  - placement / reuse / sync 粒度偏好

规则是：

- analysis 决定 legality
- policy 只在合法空间内选择

### 2.5 不能回退的约束

`Phase B` 不允许：

- 直接消费 raw fragment attrs 当 semantic truth
- 通过名字匹配恢复 spatial 边界
- 把 `Task:TTKernel = 1:1` 固化成默认心智模型
- 用 TT resource 名词污染 `Task / Channel / Layout / WorkPartition`

## 3. Semantic To Spatial Contract

`Phase B` 只允许从下列上游真源读取必须保留的算法约束：

- `SemanticProgram`
- internal state/effect graph
- companion lifecycle contract

这也意味着：

- `Phase B` 不直接读取 `tl.semantic_manifest`
- `Phase B` 不直接读取 `blackhole.fragment_regions`
- manifest / fragment evidence 必须先在 `Phase A` 被归约成冻结后的 semantic truth

主要投影关系是：

- `Update` -> 一个或多个 `Task`
- `State` -> `Channel` / `SyncEdge` / `ResourceIntent`
- `Domain` -> `Layout` / `WorkPartition` 的候选空间
- `AccessMap` -> gather / scatter / paged / routed 等空间边界
- `UpdateLaw` -> ordered update / carry / merge / completion requirement

如果 `Phase B` 发现缺失 truth，处理原则只能是：

1. 若该 truth 本质上属于 `Phase A`，回去补 witness/core/validator
2. 若该 truth 本质上属于 spatial organization，本层补 object / legality / policy
3. 不允许直接加 matcher 绕过分层

## 4. 当前实施重点

当前 `Phase B` 的实施重点是：

1. 引入 `SpatialProgram` 与 `ProgramPhase`
2. 建立 simple-workload canonical fast-path
3. 跑通至少一个 non-trivial multi-phase spatial gate
4. 让 `LowerBlackholeOps` 不再继续承担 task/channel/layout/sync 的 monolithic 黑洞职责

## 5. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## 6. Task 3: Stage 3 - Phase B Spatial Program IR

**Files:**
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/collect_device_programs.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [x] **Step 1: Introduce `SpatialProgram` and `ProgramPhase`**

Required objects:

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

Rules:

- module-scope `ProgramPhase` truth lives in `tl.device_programs`
- member-local truth lives in `PrimFunc.attrs["tl.spatial_program"]`
- simple workload gets canonical fast-path

- [x] **Step 2: Add simple-workload fast-path for copy / GEMM**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'copy or gemm or fast_path' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:

- copy / GEMM 不需要进入重 candidate synthesis
- trivial workload 仍能快速构造 canonical `SpatialProgram`

- [x] **Step 3: Add one non-trivial multi-phase spatial gate**

Recommended first gate: `flash-attn`

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'flash_attention or multi_phase' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- 至少一个 case 证明下游不会退化回 `Task:TTKernel = 1:1`
- `ProgramPhase` / `Channel` / phase-boundary materialization 在结构测试里可见

- [x] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [x] **Step 5: Stage 3 exit gate**

Only proceed when:

- `SpatialProgram` 能消费冻结后的 semantic truth
- simple-workload fast-path 稳定
- 至少一个 non-trivial multi-phase spatial gate 通过

### 6.1 2026-04-06 实施结果

- `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入 Blackhole 主线，位置在
  `ValidateSemanticRefinement` 之后、`LowerBlackholeOps` 之前
- `tl.device_programs` 现在会聚合 `ProgramPhase` truth；对 pre-`SplitHostDevice`
  的单 `PrimFunc` 退化场景，registry/validator 已支持 root-symbol fallback
- copy canonical fast-path：
  单 `transfer` task + 单 phase + 单 channel
- GEMM canonical fast-path：
  `reader / compute / writer` 三 task + 三 channel
- `flash-attn` 首个 non-trivial gate 已通过：
  `phase0_compute(reduce_*) -> phase1_stateful(recur_*)`
- 为保持 `Phase B` 只消费冻结后的 semantic truth，split device `main_kernel`
  路径缺失的 `row_reduction.kind` 已回补到 Phase A evidence，而不是让 `Phase B`
  直接回退消费 `fragment_regions`
- `LowerBlackholeOps` 已开始显式读取 `tl.spatial_program`，并把
  `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
  写入 `blackhole.lowering_requirements`

### 6.2 本轮验证

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```
