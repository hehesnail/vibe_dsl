# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-06`
- **总阶段**: Stage 4
- **当前主线**: layered IR architecture transition
  - `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- copy / GEMM 当前支持面必须保持不回退
- `SplitBlackholeKernel` 已接入主线：
  - 纯 copy 走 `fused_dataflow` 单 kernel
  - GEMM 走 `reader / compute / writer` 3-kernel
- direct runtime 当前正式支持面：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 阶段状态

- **Stage 0**: 已完成
  - `tl.device_programs`
  - `tl.semantic_seeds`
  - hard-freeze / invalidation 护栏
- **Phase A**: 已完成
  - `Stateful Semantic IR` 已落地并完成 Phase B 前 hardening
  - 工程边界文档：`tasks/dev_design/stage4_phase_a_semantic_ir.md`
  - 理论并行文档：`tasks/dev_design/stage4_phase_a_formalization_note.md`
  - `stage4_semantic_manifest` `Phase 1-2` 已落地：
    `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`
  - `AnalyzeSemanticStructure` 已改成对 manifest structural evidence 的 manifest-first 消费，
    `fragment_regions` 退化为 compatibility fallback
- **Phase B**: 已完成 compile-path hardening
  - `SpatialProgram / ProgramPhase / Task / Channel / Layout / WorkPartition /
    Placement / SyncEdge / ResourceIntent` 已落地
  - `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入 Blackhole 主线
  - compile-path hardening 已收口：
    - `LowerToSpatialProgram` 不再把 `work_decomposition / segment_plan`
      当 spatial builder truth source
    - `LowerBlackholeOps` 已硬要求 `tl.spatial_program`，不再回读
      `work_decomposition / fragment_regions / pipeline_stages`
    - representative family gate 已覆盖
      `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`
  - `Phase B` 当前应视为已完成 `SemanticProgram -> SpatialProgram`
    compile-path cutover；后续 schema / validator 的继续加固将由 `Phase C` translator
    真实需求驱动，而不再作为继续停在 `Phase B` 的 blocker
- **Phase C**: 当前主实施阶段
  - `TT Target IR` 已定义，下一步开始 `Spatial Program IR -> TT Target IR`
    的单一真源 cutover

## 当前主 blocker

当前 blocker 已经收敛到：

- `Spatial Program IR -> TT Target IR` 的单一真源切换尚未开始落地
- `TTProgram / MaterializeTTExecutableSpec` 仍不存在，当前 target/runtime contract
  还停留在 `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole`
  的旧主链上

这也是当前 `blackhole.acc` correctness payoff 仍未完全兑现的根因：主 blocker
已经转成 spatial / target 层的 contract materialization，而不是继续补 semantic matcher
或继续在 `Phase B` 保留 legacy attr fallback。

## 下一步

1. 执行 `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
   - 落地 `TTProgram`
   - 让 `MaterializeTTExecutableSpec` 成为唯一 target materialization
   - 删除 compatibility writer / reader / fallback
2. 用 `TTProgram` translator 反推 `SpatialProgram` 还缺的 contract
   - 继续按 typed schema 扩 `Task / Channel / Placement / ResourceIntent`
   - 继续收紧 `ValidateSpatialProgram`，但只补真实 translator 需要的 legality
3. 在新 target 主链上承接 `flash-attn` correctness payoff 与更宽 family expansion

## 当前代码事实

- Blackhole 设备侧 pass 主线当前为：
  `LowerDeviceStorageAccessInfo`
  -> `AugmentSemanticManifest`
  -> `LowerIntrin`
  -> `Simplify`
  -> `HoistBroadcastValues`
  -> `SplitBlackholeKernel`
  -> `AnalyzeBlackholeWorkDecomposition`
  -> `AnalyzeBlackholeFragmentRegions`
  -> `AnalyzeBlackholePipelineStages`
  -> `AnalyzeSemanticStructure`
  -> `LiftStatefulSemanticIR`
  -> `ValidateStatefulSemanticIR`
  -> `ValidateSemanticRefinement`
  -> `LowerToSpatialProgram`
  -> `ValidateSpatialProgram`
  -> `LowerBlackholeOps`
  -> `PlanBlackholeCB`
  -> `AssignBlackholeCores`
- `Phase A` 当前已具备：
  - semantic witness algebra
  - typed vocab / decoder / payload / rule modules
  - internal state/effect graph normalization
  - `preserve / typed_rebind / invalidate` lifecycle contract
  - `TypedRebindBlackholeCompanionPrograms`
  - `tl.semantic_manifest_seeds / tl.semantic_manifest`
  - `AnalyzeSemanticStructure` 对 manifest 的 seed / witness / supplement integration
  - manifest-backed structural evidence：
    `fragment_buffers / selection_targets / selection_pairs / arg_reduce_targets /
    update_sources / loop_carried_state / recurrence_edges / row_reductions`
  - manifest / structural nested-field key 已集中到
    `manifest_key::` / `schema_key::` 命名空间（`semantic_program.h`）
  - structural evidence / copy semantics / direct-runtime binding 当前已切到 handle-first：
    - `structural_regions[*]` 与 `blackhole.fragment_regions[*]` 对 state/edge/buffer
      同时保留 display name 和 typed `Buffer` handle
    - `AnalyzeSemanticStructure` manifest-first 消费 typed handle，旧字符串只保留兼容回退
    - `SplitBlackholeKernel` / `LowerBlackholeOps` 对 copy/runtime/accessor 绑定优先按
      `Buffer.data` identity 与 typed handle 恢复，不再信任 annotation name string
  - `AnalyzeBlackholeFragmentRegions` 当前对 plain-local fragment state 与 temp reduction scratch
    的区分优先依赖 `layout_map` 与 store/dataflow 结构，不再靠 `_clear` 命名约定
  - `row_reductions` 的 semantic `reduce_kind` truth source 当前已固定到 manifest explicit-op payload：
    - `tl.semantic_manifest.structural_regions[*].row_reductions[*].kind` 由 explicit `reduce`
      payload 的 `reduce_kind` 生成
    - split device / `main_kernel` 路径的 `blackhole.fragment_regions[*].row_reductions[*].kind`
      也已补齐为 IR-structural truth，避免 `Phase B` 因 kind 缺失退化成单一 stateful phase
  - `Phase B` 当前已具备：
    - typed `SpatialProgram` / `ProgramPhase` object set
    - module-scope `ProgramPhase` aggregation 写回 `tl.device_programs`
    - unsplit single-`PrimFunc` 退化场景对 `tl.device_programs.root_symbol` 的 registry fallback
    - `Layout / WorkPartition` 的 spatial scaffolding 已切到 semantic-domain-first：
      `LowerToSpatialProgram` 只读取 `SemanticProgram.domains[*].axes / traits`
    - `ValidateSpatialProgram` 已从纯结构检查升级到最小 semantic-domain legality gate：
      `layout / work_partition` 必须和 `SemanticProgram.domain` 的 axes 对齐，
      `layout.kind == indexed` 必须和 `derived_indices` trait 对齐
    - `LowerBlackholeOps` 的 `work_axes / derived_index_expr_count`
      已切到 spatial-program-first：
      优先读取 `tl.spatial_program.layouts / work_partitions`，`blackhole.work_decomposition`
      只保留 compatibility fallback
    - GEMM fast-path 的 `reader / compute / writer` task graph
      已切到 `blackhole.segment_kind` IR-structural truth：
      `LowerToSpatialProgram` 不再把 `blackhole.segment_plan` 当 spatial builder 输入
    - `LowerBlackholeOps` lowering requirements 中的
      `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
    - `LowerBlackholeOps` 的 fragment lowering requirements
      已切到 spatial-program-first：
      `AnalyzeSemanticStructure` 会把 `blackhole.fragment_regions`
      收成 `fragment_lowering_structure` semantic supplement，
      `LowerToSpatialProgram` 再把它投影成
      `ResourceIntent(kind=lowering_support, traits+=fragment_contract)`
    - `LowerBlackholeOps` 现在优先从 fragment contract resource intent payload
      恢复 `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
      pointwise_op_kinds / fragment_loop_carried_state`
    - `fragment_regions` 缺失时，仍可从 `SemanticProgram + residual body scan`
      恢复最小 fragment contract
    - `ValidateSpatialProgram` 已继续补强 phase/channel/module-scope legality：
      phase channel 必须真正落到 owning phase，非首 multi-phase phase 不能失去
      channel contract，`tl.device_programs` 也不再只比较 phase 数量
    - `ValidateSpatialProgram` 现在也会把 semantic statefulness 显式收成最小 legality：
      stateful semantic states 必须 materialize `state_residency`，
      multi-phase spatial program 还必须为每个 stateful semantic state
      提供 `phase_boundary_materialization`
    - `ValidateSpatialProgram` 现在也会显式校验 fragment contract：
      fragment program 不能缺失 fragment resource intent，
      contract payload 必须显式携带 `fragment_op_kinds`
    - pipeline stage truth 已开始迁离 legacy attr：
      `AnalyzeSemanticStructure` 会把 `blackhole.pipeline_stages` 收成
      `pipeline_structure` semantic supplement，
      `LowerToSpatialProgram` 再把它投影成
      `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract)`
    - `LowerBlackholeOps` 的 pipeline legality input
      已切到 spatial-program-first：
      优先读取 pipeline contract resource intent payload，
      `blackhole.pipeline_stages` 和 body `num_stages` annotation
      只保留 compatibility fallback
    - `ValidateSpatialProgram` 现在会显式校验 pipeline contract：
      pipeline program 不能缺失 pipeline resource intent，
      contract payload 也必须携带 `pipeline_stages[*].loop_var / num_stages`
    - `work_dependent_loop_bounds` 也已开始迁离 `blackhole.work_decomposition`：
      `AnalyzeSemanticStructure` 会把它收成
      `work_decomposition_structure` semantic supplement，
      `LowerToSpatialProgram` 再把它投影成
      `WorkPartition.payload.work_dependent_loop_bounds`
    - `LowerBlackholeOps` 现在优先从 `WorkPartition.payload`
      恢复 `work_dependent_loop_bound_count`，
      `blackhole.work_decomposition` 只剩 compatibility fallback
    - `ValidateSpatialProgram` 现在也会显式校验
      `work_dependent_bounds` domain 不能丢失 `WorkPartition` payload
    - `BlackholeDeviceResourceCanonicalization` 现在也有
      Blackhole-only 的 IR-structural fallback：
      `shared* -> blackhole.cb`、`local.fragment -> blackhole.acc`
      不再只依赖 `blackhole.resource_plan` 的显式条目
    - `LowerBlackholeOps` 的 lowering-requirements consumer path
      已完成 hard cutover：
      现在显式要求 `tl.spatial_program`，并且
      `work_axes / derived_index_expr_count / work_dependent_loop_bound_count /
      spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states /
      pipeline_stage_counts / pipeline_loop_vars / fragment_*`
      都只从 `SpatialProgram` 恢复
    - target/transform tests 里所有直接验证 `LowerBlackholeOps` 的路径
      也已切回真实主线：
      `SplitBlackholeKernel -> Analyze* -> AnalyzeSemanticStructure ->
      LiftStatefulSemanticIR -> Validate* -> LowerToSpatialProgram ->
      ValidateSpatialProgram -> LowerBlackholeOps`
    - generic spatial builder 已不再对 `root_map` 之类的 update name 做协议分支；
      generic path 按 semantic `Update` object 自身建 task，更新名只作为 object identity
    - transform-level family gate 已补到
      `topk / selection / chunk recurrence / routed dispatch / paged decode`，
      `Phase B` 当前覆盖
      `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`
  - `Phase B` 当前的边界已经收口：
    - compile-path cutover 已完成
    - 当前不再把 residual `Phase B` fallback cleanup 当主 blocker
    - `SpatialProgram` 更强的 schema / validator 继续增强会跟随 `Phase C`
      translator 的真实 contract 需求推进
    - 这不等于 `Phase C` cutover proof，也不等于 runtime correctness 全完成
  - `blackhole.fragment_regions` 不再是 semantic truth 输入；
    semantic 侧所有 evidence 已切换为 manifest-first 消费；
    `fragment_regions` 当前只保留 `LowerBlackholeOps` lowering-facing compatibility 职责，
    已不再是其唯一 fragment truth source
- `Phase A` 当前工程状态已经收口：
  - 不再依赖名字匹配恢复语义
  - 不再把 formal proof 草稿混在实现文档里
  - 当前只作为 `Phase B` 的输入边界与实现参考保留
  - `stage4_phase_a_semantic_ir.md` 已补充基于真实 `example_topk` IR/attrs 的 `topk / selection`
    端到端 walkthrough，并补了 LLM/MoE 业务语境、源 DSL、算法语义、关键记号说明、TIR 片段与逐段解析，用于说明
    `Phase A` 的 pass-by-pass 状态、核心思想与设计动机
  - `LowerAndLegalize` 已在 `LowerTileOp` 前插入 `CollectSemanticManifestSeeds`
  - `OptimizeForTarget` 已在 `SplitHostDevice` 后插入 `ProjectSemanticManifest`

## 最新验证摘要

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `40 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -q`
  - `7 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q`
  - `28 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `41 passed, 10 skipped, 1 xfailed`
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `26 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `27 passed`

说明：

- `Phase A` compile-path 和 semantic gate 当前稳定
- `flash-attn` correctness 仍不应被写成已完成稳定基线

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_semantic_manifest.md`
- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`
- `tasks/dev_design/stage4_stage0_guardrails.md`
- `tasks/dev_design/README.md`

## 说明

- `progress.md` 现在只保留相对稳定的阶段状态、主 blocker、当前下一步和验证摘要。
- 详细逐步实现记录、历史 checklist 和理论证明草稿分别留在阶段文档、git history 与 formalization note 中。
- `2026-04-06` 已完成一轮活动文档状态审计：
  `README.md`、`AGENTS.md`、`CLAUDE.md`、active stage docs 已统一到
  `Phase A` 完成、`Phase B` compile-path hardening 收口、`Phase C` 成为当前主线、
  以及 manifest-first 与 `fragment_regions` 残余职责的当前口径。
- `stage4_semantic_manifest.md` 当前作为 `Phase A` 信息源重构文档保留：
  - 它的定位是把 `Phase A` 需要的 explicit-op evidence 从 late matcher 前移到真实销毁边界：
    - `LowerTileOp` 边界的 early capture
    - `SplitHostDevice` 之后的显式 projection
    - device-side pre-`LowerIntrin` 的 residual augment
  - semantic recovery 主体仍然只在 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`
  - 当前 `Phase 1-2` 已实施；`selection / arg-reduce / recurrence` structural evidence
    已前移到 manifest 并改成 manifest-first 消费
  - `fragment_regions` 当前仍保留 `row_reductions` 与 residual compatibility 职责
  - `structural_regions` / `fragment_regions` 当前的 state/edge entry 已扩成
    `name + typed Buffer handle` 双通道 schema；后续 consumer 新逻辑默认应吃 handle-first
  - `fragment_regions` 的剩余存在理由已经收敛：
    - semantic 侧：`reduce_*` update 的 residual evidence
    - lowering 侧：`LowerBlackholeOps` 当前的 fragment subset / legality summary
  - 后续删除顺序应是：
    先迁走 `row_reductions` 的 semantic-owned 部分，再给 lowering 建独立 summary，
    最后删除 `fragment_regions`
  - 它不是新的跨层真源，也不改变 `Phase B / C` 只消费冻结后 companion IR 的边界
- `final_blackhole_backend_redesign.md` 已在 `2026-04-06` 按当前执行状态刷新：
  - 已收成轻量总体设计文档
  - `Phase A/B/C` 细节默认下沉到对应 stage 文档
