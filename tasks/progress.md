# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部是历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-06`
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
  - `Phase A` 当前还进一步收正了 stringly-typed 内部协议：
    - 新增 `semantic_vocab / semantic_witness_decoder / semantic_refinement_rules`
    - string 现在主要保留在 FFI/attr 边界
    - `LiftStatefulSemanticIR` 与 `ValidateSemanticRefinement` 内部已改为 typed vocab + centralized rule table
  - `Phase A` 当前又把高频 witness payload 收成了集中 typed family：
    - 新增 `semantic_witness_payloads`
    - `AnalyzeSemanticStructure` 现在通过 centralized payload builder 发射 canonical payload
    - `LiftStatefulSemanticIR` 与 `ValidateSemanticRefinement` 不再直接按 key 拆
      `fact_value`
    - `relation.derives_index_from` 已收成 empty payload，不再保留冗余
      `"kind": "index_derivation"` 协议
  - 进入 `Phase B` 前要求补齐的 3 个理论层机制现在也已代码落地：
    - companion lifecycle 现在有 machine-checkable
      `preserve / typed_rebind / invalidate` contract
    - `ValidateSemanticRefinement` 现已覆盖 witness coverage、typed rebind legality、
      graph consistency、以及 source/companion/carried relation 与 graph fact 的一致性
    - `SemanticProgram` 现已持有 internal state/effect graph：
      `StateVersion / StateDef / StateUse / StateJoin`
    - `TypedRebindBlackholeCompanionPrograms` 已作为 audited-safe rebind 入口接入 Python API
      与 C++ 主线
  - `Phase A` 的 research-grade formalization backlog 也已写入
    `stage4_phase_a_semantic_ir.md`：
    - 把 academic 目标收成“sound abstraction over canonical evidence domain”
    - 明确了 concrete semantics / abstract semantics / `alpha-gamma` / pass contract /
      translation validation 的研究任务
    - 明确这条 formalization 轨道是 **Phase B 并行研究项**，不是当前工程 blocker
  - `Phase A` 文档现在还进一步补了 repo-driven 理论定义：
    - 从本仓库的三类 truth 混层问题，推导出 `Phase A` 为什么必须是 abstraction layer
    - 明确了 witness/core/validator、state/effect graph、以及
      `preserve / typed_rebind / invalidate` 的结构必要性
    - 明确了 `Phase A -> Phase B` 的关系是
      `refinement by organization`，而不是再做一次 semantic recovery
  - `Phase A` 文档现已再往前补成 theorem/obligation checklist：
    - 固定了 `Evidence Domain / Abstract Domain / alpha / refinement checker`
    - 列出了 `Lift Soundness / Graph Soundness / Contract Preservation /
      Typed Rebind Preservation / Invalidation Safety / Phase A->B Refinement /
      Rejection Discipline`
    - 把下一步最值钱的 research deliverable 收敛为：
      formal semantics note、obligation matrix、`Phase B` refinement validator、
      small mechanized core
  - `Phase A` 文档当前又把 `formal semantics note` 的 repo-specific skeleton 补出来了：
    - 先固定 `Scope / Concrete Objects / Abstract Objects / alpha / R / Soundness Theorems`
    - 再固定 `Failure Modes / Bridge to Phase B / Minimal Mechanization Plan`
    - 这样后续 research track 可以直接展开，不需要再从抽象概念重新组织
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
  - Phase A closing hardening：
    - `typed_rebind` contract
    - stronger `ValidateSemanticRefinement`
    - internal state/effect graph normalization
- TT-Sim 当前正式入口是顶层 `scripts/setup_tt_sim.sh`，并且必须和后续测试命令在同一个 shell 中执行

## 最近验证

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'device_program_registry or semantic_seeds or hard_freeze' -q`
  - `3 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `28 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'state_effect_graph or typed_rebind or missing_loop_carried_join' -q`
  - `4 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'semantic_vocab_normalizes or semantic_vocab_rejects' -q`
  - `2 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'semantic_payload' -q`
  - `2 passed`
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
