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

## 4. 一个端到端例子：`topk / selection`

下面用当前已经有回归覆盖的 `topk / selection` 路径说明 `Phase A` 的核心思想。这个例子比
`flash-attn` 更适合讲设计，因为它同时暴露了两件事：

- `best_value` 这类 value state 的聚合语义
- `best_slot` 这类 companion/index state 的派生语义

而且当前测试已经明确覆盖了两种关键场景：

- 可以从真实 `topk` 例子中恢复 `select` update、`selection_state`、`index_state`
- 即便去掉变量名提示，仍然可以从 IR 结构和 `selection_pairs` 恢复 `index_state`

对应测试见：

- `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
  - `test_topk_semantic_program_lifts_select_updates_and_selection_roles`
  - `test_topk_semantic_program_recovers_index_state_from_integer_ir_not_names`
  - `test_selection_pairing_recovers_index_role_without_integer_hints`
  - `test_selection_pairing_is_recovered_from_compute_pattern`

### 4.1 输入阶段：TIR 里已经有什么

以 `topk` 为例，在进入 `Phase A` 之前，TIR 中已经存在下面这些可验证结构：

- fragment buffers：
  - `score_fragment`
  - `carry_slots`
  - `best_value`
  - `best_slot`
- row reduction：
  - `best_value` 上的 `max`
- selection companion 计算：
  - `best_slot = if_then_else(..., carry_slots[i], carry_slots[j])`
- loop-carried / fragment-local 的 buffer 生命周期

这一步最关键的事实是：

- 这里还没有 `State.role = index_state`
- 也还没有 `UpdateLaw.kind = select`

也就是说，输入侧只有 **结构证据**，还没有冻结后的算法语义。

### 4.2 `AnalyzeBlackholeWorkDecomposition`

这个 pass 先产出：

- `blackhole.work_decomposition`
- domain axes
- work-dependent bounds
- derived index exprs

对 `topk / selection` 来说，这一步给 `Phase A` 提供的是 **domain skeleton**，而不是 select 语义本身。

设计上要先有这一步，是因为：

- `SemanticProgram.Domain` 不能凭空生成，它必须锚定真实 work decomposition
- domain truth 和 state/update truth 必须分开
- 如果在这一步就开始判断 `best_slot` 是什么语义，等于把 work analysis 和 semantic recovery 重新混层

所以这个 pass 之后的状态可以概括成：

- 已知道程序工作域长什么样
- 还不知道哪些 state 是 `selection_state` / `index_state`

### 4.3 `AnalyzeBlackholeFragmentRegions`

这是 `topk / selection` 例子里最关键的 **evidence collector**。它基于 fragment buffer、scope、row
reduction、`if_then_else`/`max` 计算形态和 def-use，产出：

- `fragment_buffers`
- `row_reductions`
- `arg_reduce_targets`
- `selection_targets`
- `selection_pairs`
- `update_sources`
- `loop_carried_state`
- `recurrence_edges`

对 `topk` 而言，最重要的具体结果是：

- `best_value` 会进入 `row_reductions`
- `max_idx` 一类目标会进入 `arg_reduce_targets`
- `best_slot` 会进入 `selection_targets`
- `best_value <-> best_slot` 的 companion 关系会进入 `selection_pairs`
- `best_slot` 的 source set 会显式记录 `score_fragment`、`carry_slots`

设计上把这一步停在 “evidence” 而不是直接生成 `SemanticProgram`，是为了避免两个问题：

- fragment analysis 变成第二个 semantic IR
- 后面没有办法检查 semantic lift 是不是伪造了事实

所以这个 pass 之后的状态是：

- 已有足够强的结构证据说明 “这里存在 selection / companion / source-flow”
- 但这些证据还没有被宣称成最终语义真相

### 4.4 `AnalyzeBlackholePipelineStages`

这个 pass 为函数补上：

- `blackhole.pipeline_stages`

在 `topk` 这个例子里，它通常不会新增 selection-specific truth；但它仍然是 `Phase A` 之前必须完成的
canonicalization point，因为 `AnalyzeSemanticStructure` 会把它折成 domain trait，比如 `pipeline`。

设计上保留这一步的原因是：

- pipeline/staging 是程序组织事实，不该在 semantic lift 之后再猜
- `Phase A` 需要在稳定的 pre-lift 程序形态上工作

所以这一步之后可以理解成：

- semantic evidence 的宿主形态已经稳定
- 后面的 semantic pass 不需要再直接理解 pipeline lowering 细节

### 4.5 `AnalyzeSemanticStructure`

这一层才开始把异质 evidence 收成统一协议，但它做的仍然不是“直接发明 semantic core”，而是：

- 读取 `work_decomposition`、`fragment_regions`、`pipeline_stages`
- 生成统一的 `tl.semantic_structure`
- 生成 generic `tl.semantic_witnesses`

对 `topk / selection`，它会把上面的 fragment evidence 规范化成下面这些 witness 轴：

- `("state", "role")`
- `("update", "law_family")`
- `("update", "source_set")`
- `("relation", "companion")`
- `("relation", "derives_index_from")`

这正是测试 `test_topk_semantic_witnesses_expose_generic_fact_axes` 在检查的内容。

对这个例子来说，最关键的归一化结果是：

- `best_value` 会被归成 `reduction_accumulator`
- `best_slot` 会被归成 `index_state` 或 `selection_state`
- `select_best_slot` 会被归成 `UpdateLaw.kind = select`
- `best_value` 作为 paired value state 不再停留在 `selection_pairs` 这种 ad hoc attr 里，而会变成 relation witness

为什么要专门有 witness 层，而不是直接 lift？

- 因为 fragment analysis、seed、pipeline、future evidence family 本来就是异构来源
- 如果没有 witness 层，semantic lift 会重新变成 case-by-case matcher
- 有了 witness 层之后，`Phase A` 的问题才被收正成：
  - 先收集 canonical evidence
  - 再把 evidence 归一化成闭集 fact axes
  - 最后再投影到 small semantic core

### 4.6 `LiftStatefulSemanticIR`

这个 pass 才把 witness 投影成真正冻结后的 `SemanticProgram`。在 `topk / selection` 例子里，结果会收成：

- `State`
  - `best_value : reduction_accumulator`
  - `best_slot : index_state`
  - `carry_slots : carry` 或 source-side carried state
- `Update`
  - `reduce_best_value : reduce`
  - `select_best_slot : select`
- `UpdateLaw.source_states`
  - 来自 `update_sources`
- `Update.bindings`
  - `paired_value_state = best_value`
- internal graph
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`

这里的设计重点不是“把 topk 识别出来”，而是：

- 用很小的 semantic core 表达所有真正需要冻结的算法事实
- companion/source/carry/order 这些关系进入 core object 与 normalized graph
- 不再把 `selection_pairs / arg_reduce_targets / update_sources` 当长期协议保存

所以这一步之后，`Phase A` 的输出就已经变成：

- 一个可以被 `Phase B` 直接消费的 semantic abstraction
- 而不是一堆还要后段继续解释的 analysis attrs

### 4.7 `ValidateStatefulSemanticIR`

这个 pass 负责最小 structural closure。对 `topk / selection` 例子，它主要保证：

- `tl.semantic_program` 存在
- `Domain` 非空
- `State.role` / `UpdateLaw.kind` 属于受支持闭集
- update 引用的 target state 存在
- state/effect graph 的 version / def / use / join 引用一致

设计上保留这一层，是因为 `SemanticProgram` 必须先是一个自洽语言对象，后面才谈得上 refinement。

### 4.8 `ValidateSemanticRefinement`

这一层才真正检查：

- witness 有没有被 semantic core 正确消费
- contract 是不是还处于 live/preserve/rebind 合法状态
- graph 关系是不是和 witness 声明一致

对 `topk / selection` 来说，它会检查：

- `state.role` witness 是否真的对应到 `best_value` / `best_slot` 的 role
- `update.law_family` witness 是否真的落成 `reduce` / `select`
- `update.source_set` witness 是否真的变成 `UpdateLaw.source_states` 与对应 `StateUse`
- `relation.companion` witness 是否真的变成 `select` update 上的 `paired_value_state` binding
- `relation.derives_index_from` witness 是否真的只指向 `index_state`

这一层的意义非常直接：

- `SemanticProgram` 不是 analysis pass 自己说了算
- `Phase A` 的输出必须能够被 witness 反向核对
- 如果后续 pass 改坏了 body 或 companion contract，就必须 `typed_rebind` 或 `invalidate`

### 4.9 这个例子真正说明了什么

`topk / selection` 这个例子说明的不是：

- `Phase A` 很擅长识别 `topk`

而是：

1. `Phase A` 先收集 canonical evidence，而不是直接猜语义
2. 这些 evidence 会被归一化成 workload-agnostic witness axes
3. semantic lift 只负责把 witness 投影成小闭集 core
4. validator 负责保证 semantic core 不是无根据的自我声明

因此 `Phase A` 的核心设计思想可以压成一句话：

> 先把结构证据收正，再把证据抽象成统一 witness，最后把 witness 冻结成小闭集 semantic truth。

这也是为什么当前 `Phase A` 能同时承接 `topk / selection`、`flash-attn`、`chunk recurrence`，但协议里又不需要
引入这些 workload noun。

## 5. 当前验证摘要

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

## 6. 现在如何使用这份文档

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

## 7. 给 Phase B 的交接约束

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

## 8. Parallel Theory Track

并行的理论化 / 证明化内容已单独放到：

- [stage4_phase_a_formalization_note.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_formalization_note.md)

那里专门承接：

- canonical evidence / abstract domain 的 formalization
- `E / A / alpha / R`
- theorem / obligation checklist
- `Phase A -> Phase B` refinement interface

这条线是并行 research track，不阻塞 `Phase B` 工程实现。
