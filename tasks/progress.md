# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部是历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-05`
- **阶段**: Stage 4（总阶段）— layered IR architecture transition for complex workload families
- **当前定义**:
  - Stage 4 现在是当前 Blackhole 架构迁移的总包含阶段：
    `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
  - `Stage 0 / 1 / 2 / 3 / 4 / 5 / 6` 是 Stage 4 内部执行子阶段，不是新的顶层大阶段
  - Stage 4 当前按阶段文档直接执行，不再保留额外的总 plan 文档

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- copy / GEMM current support surface 是当前必须保持的稳定基线
- `SplitBlackholeKernel` 已接入当前设备侧 pass 主线：
  - 纯 copy 走 `fused_dataflow` 单 kernel
  - GEMM 走 3-kernel（reader / compute / writer）
- direct runtime 当前正式支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- `flash-attn` 当前已完成 analysis 与 compile-path 最小打通，但 correctness 仍未作为稳定基线通过

## 当前主 blocker

- Phase A2 现在也已落地，Semantic 层当前已明确承接：
  - wider `AccessMap.traits`
  - `UpdateLaw.kind == select / recurrence`
  - typed `SemanticSupplement`
  - workload-agnostic semantic state roles
- Phase A 当前已按设计边界完成：
  - semantic recovery 不再依赖名字匹配
  - `selection_state` 当前消费 `fragment_regions[*].selection_targets`
  - `selection` 的 value/index companion pairing 当前消费
    `fragment_regions[*].selection_pairs`，并下沉为 `paired_value_state`
  - selection/index family 的 arg-reduction target 当前消费
    `fragment_regions[*].arg_reduce_targets`
  - `recurrence` 当前消费 `fragment_regions[*].recurrence_edges`，并下沉为
    `recurrence_source_state`
  - `UpdateLaw.source_states` 当前消费 `fragment_regions[*].update_sources`
  - `stage4_phase_a_semantic_ir.md` 当前已补齐 post-A2 formal proof framing：
    - `Phase A` 的弱证明目标
    - canonical evidence / abstract domain / `α` / `γ` 的最小定义
    - `typed witness family` 的最小闭集设计
    - `ValidateSemanticRefinement` 的职责与 required checks
  - `Phase A` 的 generic witness / refinement / invalidation contract 现已代码落地：
    - `tl.semantic_witnesses` 已作为 compiler-internal generic witness algebra 接入
    - `AnalyzeSemanticStructure` 现在先投影 witness，再由 `LiftStatefulSemanticIR` 投影到 semantic core
    - `ValidateSemanticRefinement` 已接入 Python API 与 Blackhole 主编译链
    - `InvalidateBlackholeCompanionPrograms` 现在会整体清除
      `tl.semantic_structure / tl.semantic_witnesses / tl.semantic_program / tl.spatial_program / tl.tt_program`
      并把 hard-freeze contract 标记为 `invalidate`
- 当前 layered IR 迁移的直接动机仍然是 `blackhole.acc` 混合语义问题：
  - 一部分 lowering 仍把它当 TT compute-side tile scratch / matmul destination
  - 另一部分 helper 仍把它当线性 fragment scratch 数组
- 这个问题不再被定义为“只修 `flash-attn`”：
  - 它是把 `domain / state / update`、`task / channel / layout / sync`、`TT resource / ABI`
    从同一层里拆开的架构性问题
- 因此当前 blocker 不是单个测试名，而是：
  - 还没有完成 `Spatial Program IR -> TT Target IR` 的单一真源切换

## 下一步

1. 执行 [stage4_phase_b_spatial_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_b_spatial_ir.md)
   - 一等化 `ProgramPhase / Task / Channel / Layout / WorkPartition`
2. 执行 [stage4_phase_c_tt_target_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_c_tt_target_ir.md)
   - 完成 TT target cutover、compatibility deletion、`flash-attn` correctness payoff、以及 family expansion

## 当前代码事实

- Stage 4 Stage 0 guardrails 已落地：
  - `IRModule.global_infos["tl.device_programs"]` 已在 `SplitHostDevice` 前建立
  - `PrimFunc.attrs["tl.semantic_seeds"]` 已作为 pre-lift typed input 通道接入
  - `tl.semantic_hard_freeze` / `tl.companion_invalidation_reason` 已定义最小 A1 hard-freeze contract
  - `CollectDevicePrograms`、`ProjectSemanticSeeds`、`InvalidateBlackholeCompanionPrograms` 已接入主线与 Python API
- 当前 Blackhole 设备侧 pass 主线：
  `LowerDeviceStorageAccessInfo`
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
- Phase A1 当前已落地的最小语义对象：
  - `SemanticProgram`
  - `Domain`
  - `State`
  - `Update`
  - `AccessMap`
  - `UpdateLaw`
  - `TIRAnchor`
  - `TIRValueBinding`
- Phase A2 当前已落地的扩展：
  - `SemanticSupplement`
  - abstract semantic roles:
    - `carry`
    - `reduction_accumulator`
    - `selection_state`
    - `index_state`
    - `transient`
  - `UpdateLaw.kind == select / recurrence`
  - `flash-attn / topk / chunk recurrence` 的 workload-agnostic semantic gate
  - `fragment_regions[*].selection_targets`
  - `fragment_regions[*].selection_pairs`
  - `fragment_regions[*].arg_reduce_targets`
  - `fragment_regions[*].recurrence_edges`
  - `fragment_regions[*].update_sources`
- TT-Sim 当前正式入口是顶层 `scripts/setup_tt_sim.sh`，并且必须和后续测试命令在同一个 shell 中执行

## 最近验证

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'device_program_registry or semantic_seeds or hard_freeze' -q`
  - `3 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `20 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'witness or refinement or invalidation_contract' -q`
  - `5 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'recovers_index_state_from_integer_ir_not_names' -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'recovers_index_state_from_integer_ir_not_names or chunk_recurrence_semantic_program_lifts_recurrence_updates' -q`
  - `2 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'selection_pairing_is_recovered_from_compute_pattern' -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'selection_pairing_recovers_index_role_without_integer_hints' -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'topk_fragment_analysis_recovers_arg_reduce_targets' -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'chunk_recurrence_edges_are_recovered_from_compute_pattern' -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'copy or gemm or flash_attention' -q`
  - `4 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'topk or selection or recurrence' -q`
  - `4 passed`
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

## 当前活动文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_stage0_guardrails.md`
- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/README.md`

## 说明

- `progress.md` 现在只保留相对稳定的阶段状态、稳定基线、blocker 和下一步。
- 容易快速过期的逐测试结果与临时执行现场，不再写成长期状态事实；需要看即时验证，去看当前提交、命令记录或新一次验证结果。
- 旧的 runtime 架构说明、旧单层 implementation plan、以及过去阶段的详细执行记录都已移入 `tasks/dev_design/archive/`。
