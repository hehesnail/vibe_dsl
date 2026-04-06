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

### 6.4 2026-04-06 Hardening Slice: Validator / Consumer / Family Gate

本轮继续补了三个直接面向 `Phase C` 的 hardening 子项：

- `ValidateSpatialProgram` 不再只看 object 是否“能串起来”，还会校验
  `SpatialLayout / WorkPartition` 与 `SemanticProgram.domain` 的一致性
  - `layout.axes` 必须和 semantic domain axes 一致
  - `work_partition.axes` 必须和 semantic domain axes 一致
  - `layout.kind == indexed` 必须和 semantic domain 的 `derived_indices` trait 对齐
- `LowerBlackholeOps` 的 `work_axes / derived_index_expr_count` 已改为优先从
  `tl.spatial_program` 恢复
  - `blackhole.work_decomposition` 仍保留 compatibility fallback
  - `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
    继续只从 `tl.spatial_program` 恢复
- `Phase B` family gate 已补到 `topk / selection`
  - `SpatialProgram` 现在有 compile-path 测试覆盖 `select + recurrence` workload family

对应测试：

- 伪造 semantic-domain 不一致的 `SpatialProgram` 时，`ValidateSpatialProgram`
  会 fail-fast
- 删除 `blackhole.work_decomposition` 后，`LowerBlackholeOps` 仍能从
  `tl.spatial_program` 导出 `work_axes / derived_index_expr_count`
- `example_topk` 已纳入 `Phase B` transform-level coverage，验证 selection/indexing
  family 不会立刻退化回 workload-specific matcher

### 6.5 2026-04-06 Hardening Slice: GEMM Builder Purity / Fragment Fallback

本轮继续补了两个直接指向 `truth-source purity` 和 `consumer cutover` 的缺口：

- `LowerToSpatialProgram` 的 GEMM fast-path 已不再读取 `blackhole.segment_plan`
  - `reader / compute / writer` task graph 现在直接从
    `SplitBlackholeKernel` 留在 IR body 上的 `blackhole.segment_kind`
    annotation 恢复
  - `segment_plan` 可以继续作为后续 lowering / runtime compatibility attr 存在，
    但不再参与 `SpatialProgram` 构造
- `LowerBlackholeOps` 的 fragment lowering requirements 已不再把
  `blackhole.fragment_regions` 当成唯一来源
  - 当 `fragment_regions` 存在时，仍优先消费这份 lowering-facing compatibility summary
  - 当 `fragment_regions` 缺失时，会退回到
    `SemanticProgram + residual body scan`
    恢复最小 fragment contract：
    `fragment_op_kinds / pointwise_op_kinds / row_reduction_targets /
    row_broadcast_sources / fragment_loop_carried_state`

对应测试：

- 删除 `blackhole.segment_plan` 后，GEMM 仍能恢复 `reader / compute / writer`
  spatial fast-path
- 删除 `blackhole.fragment_regions` 后，`LowerBlackholeOps` 仍能恢复
  `flash-attn` 所需的 fragment lowering requirements

### 6.6 2026-04-06 Hardening Slice: Stronger Phase / Registry Legality

本轮继续把 `ValidateSpatialProgram` 从“最小 semantic-domain legality”往真正的
spatial legality validator 再推进一格：

- `Placement` 现在必须引用已知 task，且 `member_func` 必须和当前
  `SpatialProgram.member_func` 一致
- `SyncEdge` 现在必须引用已知 task
- phase 不再只检查“引用的 channel 名存在”
  - phase 引用的 channel，其 `target_task` 必须属于该 phase
  - multi-phase 程序里，非首 phase 不能失去 channel contract
- `tl.device_programs` 不再只比较聚合出来的 phase 数量
  - 现在会逐项核对 phase signature：`name / task_names / channel_names`

对应测试：

- 伪造一个下游 phase 失去 channel contract 的 `SpatialProgram`，validator 会 fail-fast
- 伪造一个 member-local phase 名和 `tl.device_programs` 聚合 truth 不一致的
  `SpatialProgram`，validator 会 fail-fast

### 6.7 2026-04-06 Hardening Slice: Pipeline Contract Migration

本轮把 `pipeline_stages` 从 lowering-facing legacy attr 往 typed companion contract
再推进一层，目标是让 `LowerBlackholeOps` 不再把
`blackhole.pipeline_stages` 当 primary source：

- `AnalyzeSemanticStructure` 现在会把 `blackhole.pipeline_stages`
  收成 `SemanticProgram.supplements[*]`
  - kind: `pipeline_structure`
  - payload: `pipeline_stages[*].loop_var / num_stages / stage_local_buffers /
    loop_carried_state`
- `ResourceIntent` schema 已补上 typed `payload`
- `LowerToSpatialProgram` 现在会把 semantic supplement 投影成
  `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract)`
- `ValidateSpatialProgram` 现在会校验：
  - pipeline contract 必须真的携带 `pipeline_stages`
  - 每个 stage entry 必须携带 `loop_var / num_stages`
  - 语义侧要求 pipeline contract 时，`SpatialProgram` 不能缺失该 resource intent
- `LowerBlackholeOps` 现在优先从 `tl.spatial_program.resource_intents[*].payload`
  恢复 `pipeline_stage_counts / pipeline_loop_vars`
  - `blackhole.pipeline_stages` 退回 compatibility fallback
  - body `num_stages` annotation 退回最后 fallback

对应测试：

- `SpatialProgram` 必须显式暴露 pipeline contract resource intent
- 删掉 pipeline contract 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.pipeline_stages` 后，`LowerBlackholeOps` 仍能恢复
  `pipeline_stage_counts / pipeline_loop_vars`

### 6.8 2026-04-06 Hardening Slice: Work-Dependent Bound Contract Migration

本轮继续收 `blackhole.work_decomposition` 的 residual truth，但只针对它还在
`LowerBlackholeOps` 里承担唯一价值的那一项：

- `work_dependent_loop_bounds`

具体迁移如下：

- `AnalyzeSemanticStructure` 现在会把
  `blackhole.work_decomposition.work_dependent_loop_bounds`
  收成 `SemanticProgram.supplements[*]`
  - kind: `work_decomposition_structure`
  - payload: `work_dependent_loop_bounds[*]`
- `WorkPartition` schema 已补上 typed `payload`
- `LowerToSpatialProgram` 现在会把这份 truth 投影到
  `SpatialProgram.work_partitions[*].payload.work_dependent_loop_bounds`
- `ValidateSpatialProgram` 现在会校验：
  - semantic domain 带 `work_dependent_bounds` trait 时，
    `SpatialProgram` 不能丢失对应的 `WorkPartition` payload
  - payload 不能是空的
- `LowerBlackholeOps` 现在优先从 `WorkPartition.payload`
  恢复 `work_dependent_loop_bound_count`
  - `blackhole.work_decomposition` 只退回 compatibility fallback

对应测试：

- causal `flash-attn` 的 `SpatialProgram` 必须显式投影
  `work_dependent_loop_bounds`
- 删掉 `WorkPartition` payload 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.work_decomposition` 后，`LowerBlackholeOps` 仍能恢复
  `work_dependent_loop_bound_count`

### 6.9 2026-04-06 Hardening Slice: Fragment Contract Migration

本轮继续收 `blackhole.fragment_regions` 的 lowering-facing residual truth，但仍保持
它在 `Phase A` 里的 compatibility 身份，不把删除 attr 本身当目标。目标是：

- 让 `SpatialProgram` 开始显式持有 `fragment` lowering contract
- 让 `LowerBlackholeOps` 不再把 `blackhole.fragment_regions` 当 primary input

具体迁移如下：

- `AnalyzeSemanticStructure` 现在会把
  `blackhole.fragment_regions` 里的 lowering-facing summary 收成
  `SemanticProgram.supplements[*]`
  - kind: `fragment_lowering_structure`
  - payload:
    `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
    pointwise_op_kinds / fragment_loop_carried_state`
- `LowerToSpatialProgram` 现在会把这份 truth 投影成
  `ResourceIntent(kind=lowering_support, traits+=fragment_contract)`
- `ValidateSpatialProgram` 现在会校验：
  - semantic 侧要求 fragment contract 时，`SpatialProgram` 不能缺失该 resource intent
  - contract payload 必须显式携带 `fragment_op_kinds`
  - `pointwise_chain` / `row_broadcast` 不能丢失其从属 payload
- `LowerBlackholeOps` 现在优先从
  `tl.spatial_program.resource_intents[*].payload`
  恢复 fragment lowering requirements
  - `blackhole.fragment_regions` 退回 compatibility fallback
  - `SemanticProgram + residual body scan` 仍保留作无 attr 时的最后 fallback

对应测试：

- `flash-attn` 的 `SpatialProgram` 必须显式投影 fragment contract resource intent
- 删掉 fragment contract 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.fragment_regions` 后，`LowerBlackholeOps` 仍能恢复
  `fragment_op_kinds / pointwise_op_kinds`

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

- `Layout / WorkPartition` 已切到 semantic-domain-first，
  但 `blackhole.work_decomposition` 仍保留 compatibility fallback
- `pipeline_stages` / `fragment_regions` 仍保留 lowering-facing compatibility 角色，
  但都已开始迁入 typed contract，尚未完全删掉 fallback

本轮新增的明确 cutover 设计是：

- `blackhole.pipeline_stages` 不能被“直接删掉”
- `AnalyzeSemanticStructure` 必须先把 pipeline stage truth 收成
  `SemanticProgram.supplements[*]`
- `LowerToSpatialProgram` 再把这份 truth 投影成 `SpatialProgram.resource_intents[*]`
  里的 typed pipeline contract
- 只有在 `LowerBlackholeOps` 能从 `tl.spatial_program` 直接恢复
  `pipeline_stage_counts / pipeline_loop_vars` 之后，
  `blackhole.pipeline_stages` 才能降成纯 compatibility fallback
- `blackhole.fragment_regions` 也不能被“直接删掉”
- `AnalyzeSemanticStructure` 必须先把 fragment lowering truth 收成
  `SemanticProgram.supplements[*]`
- `LowerToSpatialProgram` 再把这份 truth 投影成
  `SpatialProgram.resource_intents[*]` 里的 typed fragment contract
- 只有在 `LowerBlackholeOps` 能从 `tl.spatial_program` 直接恢复
  `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
  pointwise_op_kinds / fragment_loop_carried_state` 之后，
  `blackhole.fragment_regions` 才能继续往纯 compatibility fallback 收敛

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

本轮优先补强的 schema 子项是：

- `ResourceIntent` 需要从纯 `name / kind / target / traits` summary
  升到可携带 typed payload 的 contract node
- pipeline legality 相关的 spatial-owned truth
  （`loop_var / num_stages / stage_local_buffers / loop_carried_state`）
  先挂在 `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract)` 上
- fragment legality / lowering 相关的 spatial-owned truth
  （`fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
  pointwise_op_kinds / fragment_loop_carried_state`）
  先挂在 `ResourceIntent(kind=lowering_support, traits+=fragment_contract)` 上
- 这不是最终 schema 终点，但它必须足以支撑
  `LowerBlackholeOps` 不再把 `blackhole.pipeline_stages / fragment_regions`
  当 primary source

### 7.3 Legality Must Be Explicit

`ValidateSpatialProgram` 当前已经具备最小 semantic-domain legality gate，但还不等于完整的
spatial legality validator。

进入 `Phase C` 前，validator 至少要能 fail-fast 检出：

- phase order / phase boundary 不一致
- channel source-target-state 绑定不一致
- layout / partition 与 semantic domain 不一致
- multi-phase state materialization 缺失
- module-scope `tl.device_programs` 与 member-local `tl.spatial_program`
  的 cross-function truth 不一致

本轮已新增的显式 contract：

- phase-channel contract 必须真正落到 owning phase
- downstream multi-phase phase 不能没有 channel contract
- module-scope aggregated phase truth 不能只在 phase 数量上“凑巧一致”
- pipeline domain program 不能丢失 pipeline contract resource intent

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

- `work_axes / derived_index_expr_count` 已改为优先读取 `tl.spatial_program`
- `work_dependent_loop_bound_count` 已改为优先读取
  `WorkPartition.payload.work_dependent_loop_bounds`
- lowering-facing fragment summary 已有 `SemanticProgram + residual body scan`
  fallback，但 `blackhole.fragment_regions` 仍保留 compatibility path
- pipeline legality 已切到 `tl.spatial_program` 优先读取，
  但 `blackhole.pipeline_stages` / body-annotation 仍保留 compatibility fallback

这类 mixed ownership 在 `Phase C` 之前必须继续收紧。

### 7.5 Family Coverage Gates

当前 `copy / GEMM / flash-attn / topk` 只能证明首轮 spatial cut 可行，不能证明设计已经足够 general。

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

## 7.7 2026-04-06 Hardening Execution Order

`Phase B` 的剩余问题必须按 fixed order 收口，不能按“哪个 attr 看起来更顺手”零散修补。

本轮 hardening 的执行顺序固定为：

1. 先把 `ValidateSpatialProgram` 从结构检查升级到最小 legality gate
   - phase / task / channel / layout / work-partition 的 cross-reference 必须一致
   - multi-phase 程序除了 `phase_boundary_materialization` 外，还必须证明 phase 间真的有
     channel 或 boundary state contract
2. 再把 `LowerBlackholeOps` 收窄成 spatial consumer
   - `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
     只允许从 `tl.spatial_program` 恢复
   - `work_axes / derived_index_count` 应优先从 `SpatialProgram.layouts /
     work_partitions` 恢复，不再把 `blackhole.work_decomposition` 当 primary source
3. 最后扩 family gate
   - 至少补一个非 `copy / GEMM / flash-attn` 的 compile-path spatial case，确认当前 schema
     不会一离开这三类 workload 就退化

这轮 hardening 的明确边界：

- `segment_plan` 已不再参与 `SpatialProgram` builder
- `work_decomposition` / `fragment_regions` / `pipeline_stages` 仍可保留 compatibility path
- 但不能让这些 legacy attrs 继续外溢成 validator 或 `LowerBlackholeOps`
  的 primary truth source

## 8. Current Gap Inventory

基于当前实现状态，`Phase B` 的主要缺口可以收成下面四类：

1. spatial builder 仍存在 legacy attr truth leakage
2. object schema 已有骨架，但还不够强到支撑 `Phase C` 只读消费
3. validator 仍偏结构检查，缺少真正的 legality contract
4. family coverage 仍不足以支撑“足够 general”的宣称

因此当前状态应表述为：

- `Phase B` 已完成首轮落地并验证主链可行
- `Phase B` 尚未完成 hardening，不应直接表述为已具备 `Phase C` cutover 前提
