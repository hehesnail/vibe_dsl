# Stage 4 Phase B: Spatial Program IR

## 基本信息

- **文档角色**: `Phase B` 实施与设计边界文档
- **当前状态**: 当前主实施阶段；`2026-04-06` 已完成首轮落地：
  `SpatialProgram / ProgramPhase`、copy/GEMM fast-path、`flash-attn` multi-phase gate、
  以及 `LowerBlackholeOps` 对 spatial summary 的最小接线均已进入主链。
  但这仍是 **Phase B first landing**，还不是可直接进入 `Phase C` 的最终退出状态；
  在 `SpatialProgram` 成为唯一可信 spatial truth owner 之前，仍需完成本页第 7 节定义的
  hardening gates。
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

### 6.3 2026-04-06 Hardening Slice: Semantic-Domain-First Spatial Scaffolding

本轮 `Phase B` hardening 已落地一个 truth-source purity 子项：

- `LowerToSpatialProgram` 构造 `Layout / WorkPartition` 时，已改为优先读取
  `SemanticProgram.domains[*].axes / traits`
- `blackhole.work_decomposition` 仍允许作为过渡回退，但不再是 primary truth source

这一步当前覆盖到：

- `layout.axes`
- `work_partition.axes`
- `layout.kind == indexed` 的 `derived_indices` 判定

对应测试：

- 删除 `blackhole.work_decomposition` 后，`SpatialProgram` 仍能从
  `SemanticProgram.domain` 恢复正确的 axes / indexed layout

## 7. Hardening Gates Before Phase C

`Phase B` 的目标不是“已经有一套 `SpatialProgram` 对象”，而是：

- 让 `Spatial Program IR` 真正成为 `SemanticProgram -> TTProgram` 之间唯一可信的
  spatial truth owner

只有满足下面这些 gate，`Phase C` 才允许开始做正式 cutover。

### 7.1 Truth-Source Purity

`LowerToSpatialProgram` 必须完成真源纯化：

- `Phase B` 只能消费冻结后的 `SemanticProgram`
- `blackhole.work_decomposition`
- `blackhole.segment_plan`
- `blackhole.pipeline_stages`
- `blackhole.fragment_regions`

这些 attr 都不能再作为 spatial truth source。

允许的过渡状态只有两种：

1. 这些信息已经被 `Phase A` 归约成 semantic truth
2. 这些信息只作为 lowering compatibility summary 存在，不再参与 `SpatialProgram`
   构造决策

当前代码仍未完全达标：

- `Layout / WorkPartition` 仍直接从 `blackhole.work_decomposition` 恢复
- GEMM fast-path 仍直接依赖 `blackhole.segment_plan`

### 7.2 Schema Strengthening

当前 `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
已经 object 化，但仍偏向 `name / kind / traits` summary。

进入 `Phase C` 前，至少要补到足以稳定承载下面这些 spatial-owned truth：

- task ownership
- channel payload / source-state / versioned-state flow
- layout / work-partition 的结构化依据
- phase-boundary materialization contract
- cross-member phase ordering
- synchronization semantics

原则是：

- 不能让 `Phase C` 再回头猜 task graph / state flow / phase boundary
- 不能把这些 truth 继续散落在 legacy `blackhole.*` attr 里

### 7.3 Legality Must Be Explicit

`ValidateSpatialProgram` 当前只做结构健全性检查，还不等于 spatial legality validator。

进入 `Phase C` 前，validator 至少要能 fail-fast 检出：

- phase order / phase boundary 不一致
- channel source-target-state 绑定不一致
- layout / partition 与 semantic domain 不一致
- multi-phase state materialization 缺失
- module-scope `tl.device_programs` 与 member-local `tl.spatial_program`
  的 cross-function truth 不一致

规则仍然是：

- analysis 决定 legality
- policy 只在合法空间内选择

### 7.4 Lowering Consumer Cutover

`LowerBlackholeOps` 在 `Phase B` 完成前必须收窄成 spatial consumer，而不是继续承担
残余 spatial recovery。

目标状态：

- `LowerBlackholeOps` 只读取 `SpatialProgram` 提供的 lowering contract
- 对 task / channel / layout / sync / phase-boundary 的判断不再回头读取
  `fragment_regions` / `work_decomposition` / `pipeline_stages`

当前仍未完全达标：

- lowering requirements 仍同时读取 `blackhole.work_decomposition`
- lowering-facing fragment summary 仍依赖 `blackhole.fragment_regions`
- pipeline 相关 summary 仍直接来自 `blackhole.pipeline_stages`

这类 mixed ownership 在 `Phase C` 之前必须继续收紧。

### 7.5 Family Coverage Gates

当前 `copy / GEMM / flash-attn` 只能证明首轮 spatial cut 可行，不能证明设计已经足够 general。

在进入 `Phase C` 前，`Phase B` 至少需要补齐：

1. 一个 `selection / indexing` family 的 spatial gate
   - 推荐第一项：`topk`
2. 一个非 attention 的更复杂 family gate
   - 推荐从下面三者中至少打一项：
   - `routed / grouped dispatch`
   - `paged / indexed sparse decode`
   - `chunk recurrence / scan`

要求不是“所有 runtime correctness 一次做完”，而是：

- compile-path 上能证明 `SemanticProgram -> SpatialProgram` 的 object boundary 成立
- spatial object set 不会因新 family 立刻退化回 workload-specific matcher

### 7.6 Module-Scope Program Truth

`tl.device_programs` 不能只停留在“聚合 phase 列表”的最小实现。

进入 `Phase C` 前，它必须稳定承载 cross-function spatial truth，至少包括：

- phase order
- member_func ownership
- cross-member phase-boundary truth

单 `PrimFunc` 程序仍只是退化情况，不应反向成为默认心智模型。

## 8. Current Gap Inventory

基于当前实现状态，`Phase B` 的主要缺口可以收成下面四类：

1. spatial builder 仍存在 legacy attr truth leakage
2. object schema 已有骨架，但还不够强到支撑 `Phase C` 只读消费
3. validator 仍偏结构检查，缺少真正的 legality contract
4. family coverage 仍不足以支撑“足够 general”的宣称

因此当前状态应表述为：

- `Phase B` 已完成首轮落地并验证主链可行
- `Phase B` 尚未完成 hardening，不应直接表述为已具备 `Phase C` cutover 前提
