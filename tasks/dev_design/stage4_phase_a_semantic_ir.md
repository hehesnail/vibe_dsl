# Stage 4 Phase A: Stateful Semantic IR

## 基本信息

- **文档角色**: `Phase A` 工程实现与边界文档
- **当前状态**: `2026-04-06` 按设计边界已完成；当前作为实现参考与 `Phase B` 输入边界保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **并行理论文档**: `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 1. Phase A 的职责

`Phase A` 只负责一件事：

- 冻结 workload 的 **algorithmic truth**

它不负责：

- `ProgramPhase / Task / Channel / Layout / WorkPartition`
- `TT resource / transport / ABI`
- 为后段保留“再猜一次上层语义”的空间

因此，`Phase A` 的长期语义边界是：

- `SemanticProgram`
- `Domain`
- `State`
- `Update`
- `UpdateLaw`
- `AccessMap`
- `SemanticSupplement`

以及少量固定语义轴：

- state role
- update-law kind
- access trait

`Phase A` 的设计纪律仍然是：

1. 不允许名字匹配恢复语义
2. 不允许 workload-specific noun 进入长期协议
3. analysis evidence 必须能归约到 semantic core
4. 不能归约的 truth 必须二选一：
   - 若它是跨 family 复用的基础语义轴，则扩 core
   - 否则进入 `Phase B` 或 `Phase C`

像下面这些对象，只能作为 pre-lift evidence，不能成长为第二套 schema：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `update_sources`
- `recurrence_edges`

## 2. 当前已落地的退出状态

### 2.1 A1 最小语义层

`Phase A1` 已完成并稳定接入主链：

- 最小对象集：
  - `SemanticProgram`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `AccessMap`
  - `TIRAnchor`
  - `TIRValueBinding`
- 主链：
  - `AnalyzeSemanticStructure`
  - `LiftStatefulSemanticIR`
  - `ValidateStatefulSemanticIR`
- 当前主设备链中，semantic lift 已位于 `LowerBlackholeOps` 之前

### 2.2 A2 语义扩面

`Phase A2` 已完成，当前语义层已明确承接：

- workload-agnostic state roles：
  - `carry`
  - `reduction_accumulator`
  - `selection_state`
  - `index_state`
  - `transient`
- wider `UpdateLaw.kind`：
  - `map`
  - `reduce`
  - `select`
  - `recurrence`
- wider `AccessMap.traits`
- typed `SemanticSupplement`

当前 `flash-attn / topk / chunk recurrence` 只作为 validation family：

- 不进入 schema 命名空间
- 不以 workload noun 参与协议分派

### 2.3 进入 Phase B 前的收口加固

`Phase A` 为避免继续膨胀，已补齐以下机制：

- generic witness algebra：
  - `tl.semantic_witnesses`
- typed vocabulary / decoder / rule table：
  - `semantic_vocab`
  - `semantic_witness_decoder`
  - `semantic_refinement_rules`
- typed payload family：
  - `semantic_witness_payloads`
- stronger refinement validation：
  - `ValidateSemanticRefinement`
- internal state/effect normalization：
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- companion lifecycle contract：
  - `preserve`
  - `typed_rebind`
  - `invalidate`
- audited-safe rebind pass：
  - `TypedRebindBlackholeCompanionPrograms`

## 3. 当前代码落点

核心代码面现在集中在：

- `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- `tilelang_repo/src/transform/validate_semantic_refinement.cc`
- `tilelang_repo/src/transform/typed_rebind_blackhole_companion_programs.cc`
- `tilelang_repo/src/transform/common/semantic_program.h`
- `tilelang_repo/src/transform/common/semantic_vocab.h`
- `tilelang_repo/src/transform/common/semantic_witness_decoder.h`
- `tilelang_repo/src/transform/common/semantic_witness_payloads.h`
- `tilelang_repo/src/transform/common/semantic_refinement_rules.h`
- `tilelang_repo/src/transform/common/semantic_state_effect_graph.h`

当前 Blackhole 设备侧 pass 主线中与 `Phase A` 直接相关的部分是：

```text
AnalyzeBlackholeWorkDecomposition
-> AnalyzeBlackholeFragmentRegions
-> AnalyzeBlackholePipelineStages
-> AnalyzeSemanticStructure
-> LiftStatefulSemanticIR
-> ValidateStatefulSemanticIR
-> ValidateSemanticRefinement
-> LowerBlackholeOps
```

`Phase A` 当前使用并维护的 companion attrs 包括：

- `tl.semantic_seeds`
- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`
- `tl.semantic_hard_freeze`
- `tl.companion_invalidation_reason`

## 4. 当前验证摘要

截至 `2026-04-06`，当前稳定验证快照是：

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `28 passed`
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

这些验证说明：

- `Phase A` 当前 compile-path 与 semantic gate 是稳定的
- `copy / GEMM / current flash-attn compile-path` 没有因 semantic layer 回退

但这不等价于：

- `flash-attn` correctness 已成为稳定基线

`blackhole.acc` 的最终 correctness payoff 已经是 `Phase B / C` 单一真源切换问题，不再是
`Phase A` 语义恢复本身的 blocker。

## 5. 现在如何使用这份文档

这份文档现在只承担三件事：

1. 说明 `Phase A` 的工程边界
2. 说明当前代码里已经落地了哪些 semantic objects / contracts / validators
3. 为 `Phase B` 提供明确输入约束

它不再承担：

- 逐步实施 checklist
- 逐次测试流水账
- formal proof 草稿

这些内容已经分别分流到：

- git history
- `tasks/progress.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 6. 给 Phase B 的交接约束

`Phase B` 应只消费冻结后的 semantic truth：

- `SemanticProgram`
- internal state/effect graph
- lifecycle contract

`Phase B` 不应再直接依赖：

- raw fragment attrs
- ad hoc relation attrs
- 任何名字匹配或 late semantic guessing

如果 `Phase B` 发现缺失的 truth 仍必须回到 semantic recovery 层补充，处理原则仍然是：

1. 先判断该 truth 是否真属于 `Phase A`
2. 若属于 `Phase A`，优先补 witness/core/validator，不补 matcher
3. 若不属于 `Phase A`，直接进入 `Spatial Program IR` 或 `TT Target IR`

## 7. Parallel Theory Track

并行的理论化 / 证明化内容已单独放到：

- [stage4_phase_a_formalization_note.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_formalization_note.md)

那里专门承接：

- canonical evidence / abstract domain 的 formalization
- `E / A / alpha / R`
- theorem / obligation checklist
- `Phase A -> Phase B` refinement interface

这条线是并行 research track，不阻塞 `Phase B` 工程实现。
