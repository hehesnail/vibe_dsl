# Stage 4 Phase A: Stateful Semantic IR

## 基本信息

- **文档角色**: `Phase A` 工程边界与已落地实现文档
- **当前状态**: `2026-04-07` 已完成；当前作为 `Phase B` 输入边界与实现参考保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **信息源边界**: `tasks/dev_design/stage4_semantic_manifest.md`
- **并行理论文档**: `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 1. 作用域

`Phase A` 只负责一件事：

- 冻结 workload 的 algorithmic truth

它必须回答：

- 程序有哪些 `Domain / State / Update`
- 每个 `State` 的语义角色是什么
- 每个 `Update` 属于哪类 `UpdateLaw`
- `AccessMap`、state version、def-use-join 关系如何固定

它不负责：

- `Task / Channel / Layout / WorkPartition / ProgramPhase`
- TT resource / transport / sync / ABI
- 为后段保留“再猜一次 algorithmic truth”的空间

## 2. 必须交付的 Semantic Contract

### 2.1 Core Objects

`Phase A` 的长期 core object set 只保留：

- `SemanticProgram`
- `Domain`
- `State`
- `Update`
- `UpdateLaw`
- `AccessMap`
- `SemanticSupplement`

### 2.2 固定语义轴

当前稳定保留的 workload-agnostic 语义轴包括：

- state role：
  - `carry`
  - `reduction_accumulator`
  - `selection_state`
  - `index_state`
  - `transient`
- `UpdateLaw.kind`：
  - `map`
  - `reduce`
  - `select`
  - `recurrence`
- access trait

这些轴可以扩，但扩展规则固定：

- 若是跨 family 复用的基础语义轴，才允许进入 semantic core
- 若属于 task/layout/sync/placement/target 组织问题，必须下沉到 `Phase B / C`
- workload noun 不允许进入长期协议

### 2.3 内部 normalization contract

`SemanticProgram` 之外，`Phase A` 还必须冻结：

- `StateVersion`
- `StateDef`
- `StateUse`
- `StateJoin`

这些对象不是装饰信息，而是 semantic truth 的 normalized state/effect skeleton。

### 2.4 Companion lifecycle contract

semantic lift 之后，companion truth 只能处于三种生命周期：

- `preserve`
- `typed_rebind`
- `invalidate`

任何 post-lift pass 如果不能满足这三类之一，就不能继续沿用旧 semantic companion。

## 3. 基本纪律

- 不允许名字匹配恢复语义
- 不允许 workload-specific noun 进入 semantic schema
- analysis evidence 必须能归约到 semantic core
- pre-lift evidence 不能成长为第二套长期 schema
- `Phase B` 只能消费冻结后的 semantic truth，不能重新读 raw evidence 发明语义

像下面这些对象只能作为 pre-lift evidence 或 manifest payload，
不能上升成语义层长期对象：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `update_sources`
- `recurrence_edges`

## 4. 当前已落地的部分

### 4.1 主链对象与 validator

- `AnalyzeSemanticStructure`
- `LiftStatefulSemanticIR`
- `ValidateStatefulSemanticIR`
- `ValidateSemanticRefinement`

### 4.2 witness / decoder / refinement 基础设施

- `tl.semantic_witnesses`
- `semantic_vocab`
- `semantic_witness_decoder`
- `semantic_witness_payloads`
- `semantic_refinement_rules`

### 4.3 state/effect 与 rebind 基础设施

- `semantic_state_effect_graph`
- `TypedRebindBlackholeCompanionPrograms`
- `body_hash` 校验
- rebind 时重建 state/effect graph，而不是沿用旧 graph

### 4.4 语义边界收口

- `AnalyzeSemanticStructure` 已采用 manifest-first 消费
- `blackhole.fragment_regions` 不再是 semantic truth owner
- semantic main path 已不再依赖 `CanonicalBufferName` 这类 suffix 归一化逻辑
- state identity 当前统一优先使用 `Buffer.data` / typed handle
- `flash-attn / topk / chunk recurrence` 当前只作为 validation family，
  不进入 schema 命名空间

## 5. 当前信任边界

`Phase A` 的正确性建立在上游 evidence 足够完整之上。
它当前依赖的主要输入面是：

- `AnalyzeBlackholeWorkDecomposition`
  - domain skeleton
- `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`
  - explicit-op evidence 与 manifest-backed structural evidence
- `AnalyzeBlackholePipelineStages`
  - pipeline trait
- `AnalyzeBlackholeFragmentRegions`
  - 当前只剩 lowering compatibility / residual fallback

`Phase A` 当前能保证的是：

- 给定上游 evidence，lift 和 validation 是自洽的

`Phase A` 当前不能保证的是：

- 当上游 evidence 本身遗漏或错误时，自动发明出正确语义

因此扩展新 family 时，先看 evidence source 是否正确收集，再看 semantic core 是否足够。

## 6. 当前代码落点

核心代码面集中在：

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

与 `Phase A` 直接相关的当前设备侧主链是：

```text
AugmentSemanticManifest
  -> LowerIntrin
  -> Simplify
  -> HoistBroadcastValues
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> ValidateSemanticRefinement
  -> LowerBlackholeOps
```

## 7. 当前 companion attrs

`Phase A` 当前维护的 companion attrs 包括：

- `tl.semantic_seeds`
- `tl.semantic_manifest_seeds`
- `tl.semantic_manifest`
- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`
- `tl.semantic_hard_freeze`
- `tl.companion_invalidation_reason`

## 8. 给 `Phase B` 的交接约束

`Phase B` 应只消费下面这些冻结后的 semantic truth：

- `SemanticProgram`
- normalized state/effect graph
- lifecycle contract

`Phase B` 不应直接依赖：

- raw fragment attrs
- ad hoc relation attrs
- 名字匹配
- late semantic guessing

如果 `Phase B` 发现缺 truth，处理原则固定为：

1. 先判断它是否真属于 `Phase A`
2. 若属于 `Phase A`，补 witness / core / validator，不补 matcher
3. 若不属于 `Phase A`，直接下沉到 `Spatial Program IR` 或 `TT Target IR`

## 9. 使用方式

这份文档现在只承担四件事：

1. 定义 `Phase A` 的工程边界
2. 说明当前已经落地的 semantic objects / validators
3. 明确当前信任边界与 evidence 前提
4. 为 `Phase B` 提供正式输入约束

它不再承担：

- 长例子 walkthrough
- 逐步实施 checklist
- 验证数字快照
- formal proof 草稿

这些内容分别以下沉到：

- git history
- `tasks/progress.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 10. 稳定结论

- `Phase A` 当前 compile-path 与 semantic gate 已稳定
- `SemanticProgram` 可以作为 `Phase B` 的正式输入边界
- `flash-attn` 的最终 correctness payoff 不属于 `Phase A`；
  它已经转成 `Phase B / C` 的单一真源切换与 target materialization 问题
