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
- **Phase B**: 当前主实施阶段
  - 目标是把冻结后的 semantic truth 组织成 `SpatialProgram`
- **Phase C**: 已定义，待 Phase B 后推进

## 当前主 blocker

当前 blocker 已经不是 `Phase A` 语义恢复本身，而是：

- 还没有完成 `Spatial Program IR -> TT Target IR` 的单一真源切换

这也是当前 `blackhole.acc` correctness payoff 仍未完全兑现的根因：问题已经转成
spatial / target 层的 truth ownership，而不是继续补 semantic matcher。

## 下一步

1. 执行 `tasks/dev_design/stage4_phase_b_spatial_ir.md`
   - 引入 `SpatialProgram / ProgramPhase / Task / Channel / Layout / WorkPartition`
   - 建立 `Phase A -> Phase B` 的稳定消费边界
2. 执行 `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
   - 完成 TT target cutover
   - 删除 compatibility writer / reader / fallback
   - 承接 `flash-attn` correctness payoff 与更宽 family expansion

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
    update_sources / loop_carried_state / recurrence_edges`
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
  - `38 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `source scripts/setup_tt_sim.sh && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

说明：

- `Phase A` compile-path 和 semantic gate 当前稳定
- `flash-attn` correctness 仍不应被写成已完成稳定基线

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_semantic_manifest.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`
- `tasks/dev_design/stage4_stage0_guardrails.md`
- `tasks/dev_design/README.md`

## 说明

- `progress.md` 现在只保留相对稳定的阶段状态、主 blocker、当前下一步和验证摘要。
- 详细逐步实现记录、历史 checklist 和理论证明草稿分别留在阶段文档、git history 与 formalization note 中。
- `stage4_semantic_manifest.md` 当前作为 `Phase A` 信息源重构文档保留：
  - 它的定位是把 `Phase A` 需要的 explicit-op evidence 从 late matcher 前移到真实销毁边界：
    - `LowerTileOp` 边界的 early capture
    - `SplitHostDevice` 之后的显式 projection
    - device-side pre-`LowerIntrin` 的 residual augment
  - semantic recovery 主体仍然只在 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`
  - 当前 `Phase 1-2` 已实施；`selection / arg-reduce / recurrence` structural evidence
    已前移到 manifest 并改成 manifest-first 消费
  - `fragment_regions` 当前仍保留 `row_reductions` 与 residual compatibility 职责
  - 它不是新的跨层真源，也不改变 `Phase B / C` 只消费冻结后 companion IR 的边界
- `final_blackhole_backend_redesign.md` 已在 `2026-04-06` 按当前执行状态刷新：
  - 已收成轻量总体设计文档
  - `Phase A/B/C` 细节默认下沉到对应 stage 文档
