# Stage 4 Phase A: Stateful Semantic IR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不回退 copy / GEMM / current compile-path 的前提下，建立 `SemanticProgram` 为中心的最小语义层，并扩到能显式承接 `flash-attn` 的 carry/update 语义和至少一个 non-attention gate。

**Architecture:** 先做 `Phase A1` 的最小语义 core 和 hard freeze，再做 `Phase A2` 的 wider `AccessMap / UpdateLaw`、typed supplement 和 non-attention semantic gate。`Phase A` 的职责是冻结算法语义真相，而不是提前发明 Spatial/TT target policy。

**Tech Stack:** TileLang transform passes, TVM Object system, pytest

## Proof Obligation

`Phase A` 必须满足一个比“测试过了”更硬的约束：

- `SemanticProgram` 的正式 vocabulary 必须是小闭集
- analysis attrs 里的 relation 只能作为 evidence，不能偷偷长成第二套 semantic schema

因此，当前 `Phase A` 对外承诺的长期 semantic core 只包括：

- `Domain`
- `State`
- `Update`
- `UpdateLaw`
- `AccessMap`

以及少量固定枚举轴：

- state role
- update-law kind
- access trait

像下面这些东西：

- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`
- `update_sources`

只能被当作 pre-lift analysis evidence。它们的合法性取决于一件事：

- 是否能归约到已有 semantic core 字段

当前允许的归约目标只有：

- `State.role`
- `UpdateLaw.kind`
- `UpdateLaw.source_states`
- `AccessMap.traits`
- `Update.bindings`

如果某个新 relation/evidence 不能归约到这些 core 字段之一，就不允许把它继续留在 `Phase A`；必须二选一：

1. 证明 semantic core 少了一个真正基础、跨 family 复用的语义轴，然后扩 core
2. 承认该信息不属于 `Phase A`，转入 `Phase B` 或 `Phase C`

这条约束的目的不是禁止 analysis evidence，而是防止 `Phase A` 退化成无限增长的 relation vocabulary。

---

## Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## Task 1: Stage 1 - Phase A1 Minimal Semantic IR

**Files:**
- Create: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- Create: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- Create: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Status: `2026-04-05` 已落地

- [x] **Step 1: Add the minimal semantic object set**

Required objects:

- `SemanticProgram`
- `Domain`
- `State`
- `Update`
- `AccessMap`
- `UpdateLaw` with stable `kind`
- `TIRAnchor`
- `TIRValueBinding`

A1 explicit boundary:

- `MapLaw` / `ReduceLaw` fully modeled
- `SelectLaw` / `RecurrenceLaw` allowed as `kind` shell only if needed by validator
- no rebind-aware safe-pass contract yet

- [x] **Step 2: Lift and validate copy / GEMM / flash-attn subset**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'copy or gemm or flash_attention' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- `tl.semantic_program` 在目标 `PrimFunc` 上可见
- copy / GEMM 不丢已有 compile-path 能力
- `flash-attn` subset 至少能稳定 lift 出 `Domain / State / UpdateLaw.kind`

- [x] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [x] **Step 4: Stage 1 exit gate**

Only proceed when:

- A1 minimal object set 已稳定
- `ValidateStatefulSemanticIR` 能拦住结构不一致输入
- copy / GEMM / current `flash-attn` compile-path 零回归

Implemented note:

- `SemanticProgram / Domain / State / Update / AccessMap / UpdateLaw / TIRAnchor / TIRValueBinding`
  已作为最小 typed object set 接入
- A1 当前通过 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> ValidateStatefulSemanticIR`
  从现有 `blackhole.work_decomposition / fragment_regions / pipeline_stages / semantic_seeds`
  构建最小语义层
- Blackhole 主设备链当前已在 `tilelang/engine/lower.py` 中接入 A1 semantic lift
- 当前验证：
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -q`
    - `7 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'copy or gemm or flash_attention' -q`
    - `4 passed`
  - `pytest testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
    - `25 passed`
  - shared zero-regression baseline：
    - `test_blackhole_copy_pipeline.py -q`
      - `40 passed, 10 skipped, 1 xfailed`
    - `test_blackhole_copy_runtime.py -q` under `scripts/setup_tt_sim.sh`
      - `12 passed`
    - `test_blackhole_gemm.py -q`
      - `24 passed, 11 skipped`
    - `test_blackhole_tvm_ffi_export.py -q`
      - `1 passed`

## Task 2: Stage 2 - Phase A2 Semantic Expansion

**Files:**
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- Modify: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Status: `2026-04-05` 已落地

Design note:

- A2 继续遵守“语义层只保留抽象 role / trait，不允许 workload-specific 名字进入协议”的总原则
- `flash-attn` / `topk` / `chunk recurrence` 只是 validation family，不是 schema family
- A2 的 state role 应保持 small-closed、非 workload-specific；当前扩面目标：
  - `carry`
  - `reduction_accumulator`
  - `selection_state`
  - `index_state`
  - `transient`
- A2 的 law 扩面目标：
  - 在 A1 的 `map / reduce` 之外，补 `select / recurrence`
- `SemanticSupplement` 只允许承载 typed semantic recovery 缺口，不能退化成 workload noun bag
- 对 `flash-attn`，A2 要求语义层能区分：
  - algorithmic carry / reduction-update state
  - transient compute scratch / matmul destination hint
  但这种区分必须通过抽象 role/trait 表达，不能把 `scores_max / logsum / acc_s_cast` 等具体命名写成长期协议
- A2 semantic recovery 必须基于 IR 结构与 typed analysis attrs；如果当前 attrs 不足以稳定恢复角色，就先扩 attrs/schema，不能回退到名字匹配

- [x] **Step 1: Expand semantic schema beyond A1**

Required additions:

- fuller `AccessMap.traits`
- `SelectLaw`
- `RecurrenceLaw`
- typed `SemanticSupplement`
- clearer `AtomicEffect -> Update` recovery boundary

- [x] **Step 2: Make `flash-attn` carry / stats state explicit in semantic layer**

This stage must separate:

- algorithmic carry / reduction-update state
- TT compute scratch / matmul destination state

This is the first stage allowed to directly attack the root cause behind the current `blackhole.acc` correctness mismatch.

- [x] **Step 3: Add one non-attention semantic gate**

Recommended gates for A2:

1. `topk`
2. chunk recurrence

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'topk or selection or recurrence' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- `topk` 稳定 lift 出 `UpdateLaw.kind == select`
- chunk recurrence 稳定 lift 出 `UpdateLaw.kind == recurrence`
- `flash-attn` 的 stats/carry/update 不再依赖名字匹配；若结构信号不足，必须显式扩 attrs/schema
- `flash-attn` pipeline 断言能看见 algorithmic state 与 transient scratch 的语义分离

- [x] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [x] **Step 5: Stage 2 exit gate**

Only proceed when:

- `flash-attn` semantic root cause 已在 semantic 层有显式对象表达
- 至少一个 non-attention semantic gate 通过
- shared zero-regression baseline 全绿

Implemented note:

- `SemanticProgram` 现在额外承载 typed `SemanticSupplement`，A2 当前已把
  `AccessMap.traits`、`UpdateLaw.kind == select / recurrence` 和 supplement payload
  接进 typed semantic object set
- `AnalyzeSemanticStructure` 当前会从现有
  `fragment_regions / work_decomposition / pipeline_stages / semantic_seeds`
  恢复抽象 state role：
  - `carry`
  - `reduction_accumulator`
  - `selection_state`
  - `index_state`
  - `transient`
- A2 明确保持 workload-agnostic schema：
  - `flash-attn / topk / chunk recurrence` 只作为 validation family
  - schema 本身不引入 workload-specific noun bag
- `AnalyzeBlackholeFragmentRegions` 当前还会显式导出 `selection_targets`
  这类局部计算关系事实；`AnalyzeSemanticStructure` 现在消费这些 typed targets，
  而不是再用全局 `if_then_else` / `row_broadcast` 命中去晚期猜 `selection_state`
- `AnalyzeBlackholeFragmentRegions` 现在也会显式导出 `selection_pairs`：
  - `value_target`
  - `companion_target`
  - `source_states`
  这份 pairing 当前会被 `AnalyzeSemanticStructure` 下沉到对应 `select` update 的
  typed binding（当前为 `paired_value_state`），避免继续在 semantic lift 末端猜
  “哪个 value state 和哪个 index/companion state 属于同一次 selection”
- `AnalyzeBlackholeFragmentRegions` 现在也会显式导出 `arg_reduce_targets`：
  - row-reduction target whose source comes from a selection companion/value flow
  这份 typed relation 当前优先用于恢复 `index_state`，
  不再把 selection/index family 的角色判定建立在 integer hint 上
- `AnalyzeBlackholeFragmentRegions` 现在也会显式导出 `update_sources`：
  - `target -> source_states`
  - `LiftStatefulSemanticIR` 优先把这份 typed 关系写进 `UpdateLaw.source_states`
  - `select / recurrence / reduce` 不再默认把 `target_state` 自己回填成唯一 source
- `AnalyzeBlackholeFragmentRegions` 现在也会显式导出 `recurrence_edges`：
  - `target`
  - `source_states`
  这份 carried-update edge 当前会被 `AnalyzeSemanticStructure` 写进对应
  `recurrence` update 的 typed binding（当前为 `recurrence_source_state`），
  避免 recurrence 继续只剩“carried role + source_states 列表”而没有显式 edge 事实
- `recurrence` 当前也不再依赖 `gemm + loop_carried_state` 的组合 heuristic；
  A2 当前至少已收正到直接基于 loop-carried 结构恢复 recurrence update
- `flash-attn` 当前已在 semantic layer 和 pipeline gate 上稳定看到：
  - algorithmic carry / reduction state
  - transient compute scratch
  的抽象角色分离
- 以当前总设计对 Phase A 的边界看，A2 exit gate 已满足：
  - workload-agnostic semantic schema 已稳定
  - `selection / recurrence / source-state` 的关键计算关系已显式进入 typed analysis attrs
  - shared zero-regression baseline 全绿
- 当前验证：
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -q`
    - `15 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'recovers_index_state_from_integer_ir_not_names' -q`
    - `1 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'recovers_index_state_from_integer_ir_not_names or chunk_recurrence_semantic_program_lifts_recurrence_updates' -q`
    - `2 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'selection_pairing_is_recovered_from_compute_pattern' -q`
    - `1 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'selection_pairing_recovers_index_role_without_integer_hints' -q`
    - `1 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'topk_fragment_analysis_recovers_arg_reduce_targets' -q`
    - `1 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'chunk_recurrence_edges_are_recovered_from_compute_pattern' -q`
    - `1 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'topk or selection or recurrence' -q`
    - `4 passed`（含新的 no-name-hint regression）
  - `pytest testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
    - `26 passed`
  - shared zero-regression baseline：
    - `test_blackhole_copy_pipeline.py -q`
      - `40 passed, 10 skipped, 1 xfailed`
    - `test_blackhole_copy_runtime.py -q` under `scripts/setup_tt_sim.sh`
      - `12 passed`
    - `test_blackhole_gemm.py -q`
      - `24 passed, 11 skipped`
    - `test_blackhole_tvm_ffi_export.py -q`
      - `1 passed`

## Post-A2 Formal Proof Framing

当前 `Phase A` 的实现状态，已经足以支持“设计方向正确、schema 边界合理、compile gate 通过”
这一级别的结论；但还不足以支持“`Phase A` 已被理论证明正确”。

`Phase A` 后续若要从“方向正确”升级为“可证明的抽象系统”，证明目标必须收成下面这个弱命题：

- 对一个受限的 workload class，只要其关键算法语义能由 `Phase A` 的 canonical evidence family
  表达，并且这些 evidence 可归约到当前 semantic core，
  则 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR` 产出的 `SemanticProgram`
  是这组 evidence 的 sound abstraction

不允许追求的强命题：

- 从任意 lower 后 TIR 自动恢复任意复杂 workload 的全部非平凡语义性质

### Minimal Proof Objects

若要证明上面的弱命题，`Phase A` 至少需要补齐以下 formal objects：

1. **Concrete domain**
   - 不是“任意 TIR”
   - 而是 semantic lift canonicalization 点之后、post-lift hard freeze 之前的
     canonical evidence 集合：
     - pre-lift TIR structural facts
     - `blackhole.fragment_regions`
     - `blackhole.work_decomposition`
     - `blackhole.pipeline_stages`
     - `tl.semantic_seeds`
     - 后续允许补入的 early semantic capture signal
2. **Abstract domain**
   - `SemanticProgram`
   - `Domain / State / Update / AccessMap / UpdateLaw`
   - 少量固定 role / law / trait 轴
   - 受约束的 typed `SemanticSupplement`
3. **Concretization `γ`**
   - `γ(program)` 给出与该 `SemanticProgram` 一致的 concrete evidence 集
4. **Abstraction `α`**
   - `α(evidence)` 给出该 evidence 的最小 sound `SemanticProgram`
   - 当前实现中的 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR`
     本质上就是 `α` 的工程实现候选

### Required Proof Obligations

在这套 framing 下，`Phase A` 的证明责任至少包括：

1. **Canonicality**
   - semantic lift 只消费固定 canonicalization 点上的 evidence
   - lift 之后若发生 unsafe TIR mutation，companion IR 必须整体失效
2. **Evidence Closure**
   - evidence 必须属于小闭集 witness family
   - 不允许长期停留在开放 `Map<String, Any>` relation bag 形态
3. **Evidence-to-Core Reducibility**
   - 每个 witness 都必须归约到：
     - `State.role`
     - `UpdateLaw.kind`
     - `UpdateLaw.source_states`
     - `AccessMap.traits`
     - `Update.bindings`
   - 否则要么扩 semantic core，要么分流到 `Phase B / C`
4. **Abstraction Soundness**
   - 对所有合法 evidence `e`，必须满足 `e ∈ γ(α(e))`
   - 也就是 lift 结果不能引入与 evidence 不一致的语义事实
5. **Supplement Discipline**
   - `SemanticSupplement` 只能裁决：
     - `state_identity`
     - `access_trait`
     - `update_law_trait`
     - `semantic_boundary`
   - 不能表达 workload noun、算法模板、spatial truth、target truth
6. **Preserve / Rebind / Invalidate Contract**
   - semantic lift 之后的 pass 必须显式属于三类之一：
     - preserve companion abstraction
     - 通过 typed rebind 维护 companion abstraction
     - invalidate companion abstraction
7. **Phase-B Refinement Readiness**
   - `SpatialProgram` 只能消费冻结后的 semantic truth
   - 不允许在 `Phase B / C` 重新猜 `carry / recurrence / selection / source_states`

### Main Gaps In The Current Implementation

按上面的 proof obligation 看，当前代码还缺下面这些关键构件：

1. **缺 concrete semantics**
   - 当前 evidence 还只是 attrs / analysis result / early signal 的工程容器
   - 还没有被定义成“它们分别代表什么语义事实”
2. **缺 abstract semantics**
   - `Domain / State / Update / AccessMap / UpdateLaw` 已有 schema
   - 但 `UpdateLaw` rich payload、`AccessMap.traits`、`SemanticSupplement`
     还没有 formal meaning
3. **缺 typed witness closure**
   - 当前
     - `selection_pairs`
     - `arg_reduce_targets`
     - `recurrence_edges`
     - `update_sources`
     仍主要以开放 relation attr 形态存在
   - 长期需要收成闭集 typed witness family，而不是开放 schema
4. **缺内部 state/effect graph**
   - 当前设计已提出 `StateVersion / StateDef / StateUse / StateJoin`
   - 这类近似 `MemorySSA` 的 internal normalization graph
     不是可选优化，而是证明 carry / ordered update / re-lift correctness 的必要支撑
5. **缺 stronger validator**
   - 当前 `ValidateStatefulSemanticIR` 还只是最小结构验证
   - 还没有检查 lifted `SemanticProgram` 是否真是 evidence 的合法 refinement
6. **缺 pass-level semantic preservation contract**
   - 当前已具备 hard-freeze 和 invalidation 方向
   - 但还没有形成 machine-checkable 的 preserve / rebind / invalidate 三分合同

### Design Consequence

因此，`Phase A` 的后续理论收口不应继续靠“多补几个 regression case”完成，而应优先实现：

1. `Phase A` canonical evidence semantics
2. typed witness family
3. stronger refinement validator
4. pass-level preserve / rebind / invalidate contract

只有把这四项补齐，`Phase A` 才能从“当前已完成的工程语义层”进一步升级成
“有界、可验证、可维护的 semantic abstraction layer”。

## Generic Witness Algebra

上面的 proof framing 只定义了“evidence 必须闭集化”，但还没有把 witness system 本身写成
一开始就具备通用性的 schema。`Phase A` 若要避免再次走向 case-by-case 膨胀，witness 不能按
当前 workload family 或当前 relation 名字切类；它必须从一开始就是**正交的事实代数**。

### Witness Design Goal

witness 不是新的 semantic core；它们是：

- pre-lift canonical evidence 的 typed 载体
- `AnalyzeSemanticStructure` 的正式输入
- `LiftStatefulSemanticIR` 的投影来源
- `ValidateSemanticRefinement` 的核对对象

因此 witness algebra 必须满足：

1. 小闭集
2. 每个 witness 都绑定稳定结构锚点
3. witness 只表达 semantic fact，不表达 workload noun
4. witness 只沿少数正交 axis 扩展，而不是沿 family 扩展
5. witness 不允许直接表达 spatial / TT target 事实

### Witness Normal Form

所有 witness 都应统一具备下面的最小公共字段：

- `subject_kind`
- `subject_anchor_id`
- `fact_axis`
- `fact_value`
- `related_anchor_ids`
- `evidence_sources`
- `canonicalization_point`

其中：

- `subject_kind`
  - `domain`
  - `state`
  - `update`
  - `access`
  - `relation`
  - `boundary`
- `subject_anchor_id`
  - 必须指向 pre-lift 已稳定存在的结构锚点
- `fact_axis`
  - 只能从预定义 axis 集合中选择
- `fact_value`
  - 只能是该 axis 允许的 typed payload
- `related_anchor_ids`
  - 用于伴随关系、carried dependency、derived-index dependency 等跨对象事实
- `evidence_sources`
  - 只记录该 witness 来自哪些 analysis pass / seed channel
- `canonicalization_point`
  - 记录该 witness 绑定在哪个 pass window 上有效

### Closed Witness Axes

第一版 witness algebra 不按 workload 切类，而按 subject-kind x fact-axis 切分。
推荐第一版固定下面这些 axis。

1. **State axes**
   - `role`
   - `identity`
   - `lifetime`
2. **Update axes**
   - `law_family`
   - `source_set`
   - `ordering`
   - `boundary`
3. **Access axes**
   - `indirection`
   - `selection_contract`
   - `distribution_hint`
4. **Relation axes**
   - `companion`
   - `derives_index_from`
   - `feeds_update`
   - `carried_from`
5. **Boundary axes**
   - `semantic_boundary`
   - `ordered_region`

这组 axis 的含义是：

- 新 workload 进入时，优先复用现有 axis 组合
- 只有当某个新事实既跨 family 复用、又无法归约到现有 axis 时，才允许扩 axis
- 不允许为 `topk`、`paged decode`、`fusedmoe`、`flash-attn` 各自新增 witness 类

### Witness-to-Core Projection

对每个 axis，都必须写死唯一合法的 core projection：

| subject kind | fact axis | legal projection |
|--------------|-----------|------------------|
| `state` | `role` | `State.role` |
| `state` | `identity` | `State.role` 或 `SemanticSupplement(state_identity)` |
| `update` | `law_family` | `UpdateLaw.kind` |
| `update` | `source_set` | `UpdateLaw.source_states` |
| `update` | `ordering` | `UpdateLaw` typed trait / `SemanticSupplement(update_law_trait)` |
| `update` | `boundary` | `SemanticSupplement(semantic_boundary)` |
| `access` | `indirection` | `AccessMap.traits` |
| `access` | `selection_contract` | `AccessMap.traits` / `Update.bindings` |
| `relation` | `companion` | `Update.bindings` |
| `relation` | `derives_index_from` | `State.role = index_state` |
| `relation` | `feeds_update` | `UpdateLaw.source_states` |
| `relation` | `carried_from` | `UpdateLaw.kind = recurrence` + `UpdateLaw.source_states` + `Update.bindings` |
| `boundary` | `semantic_boundary` | `SemanticSupplement(semantic_boundary)` |

这条表的意义是：

- witness 不是自由解释的提示
- witness 必须有唯一合法投影
- witness 一旦无法找到合法投影，就说明 schema 设计有洞，或该事实本该分流到 `Phase B / C`

### Compatibility Mapping From Current A2 Evidence

当前 A2 已落地的开放 relation attr 只作为 compatibility producer 保留，并映射到通用 witness algebra：

- `selection_targets`
  - `state.role = selection_state`
  - 必要时补 `access.selection_contract`
- `selection_pairs`
  - `relation.companion`
  - `relation.feeds_update`
- `arg_reduce_targets`
  - `relation.derives_index_from`
  - `state.role = index_state`
- `update_sources`
  - `update.source_set`
- `recurrence_edges`
  - `relation.carried_from`
  - `update.ordering`

因此 cutover 的目标不是“新增一批 selection/recurrence witness class”，而是：

- 让这些开放 attrs 先投影到统一 witness algebra
- 再由统一 witness algebra 投影到 semantic core

一旦 witness producer 稳定接管，同名开放 attrs 不再允许承担 semantic 真源入口。

## Stronger Refinement Validator

当前 `ValidateStatefulSemanticIR` 仍然只做最小结构检查。要把 `Phase A` 提升到“可验证的抽象层”，
必须新增一层更强的 refinement validator。

### Validator Goal

refinement validator 不是再做一遍 recovery；它只回答一件事：

- 当前 `SemanticProgram` 是否真是当前 canonical witness set 的合法抽象

因此它检查的不是“有没有猜对 workload”，而是：

- semantic core 是否解释了 witness
- supplement 是否只补裁决、不补结构
- pass 之后 companion state 是否仍处在 preserve / rebind / invalidate 合同内

### Required Checks

第一版 refinement validator 至少应覆盖：

1. **Witness Coverage**
   - 每个 witness 都必须：
     - 被某个 core 字段消费
     - 或被显式拒绝并给出 invalidation / unsupported reason
   - 不允许 orphan witness
2. **Projection Consistency**
   - `state.role`
     - 对应 `State.role` 必须一致
   - `update.law_family`
     - 对应 `UpdateLaw.kind` 必须一致
   - `update.source_set`
     - 对应 `UpdateLaw.source_states` 必须与 witness payload 一致
   - `relation.companion`
     - 对应 update 必须属于允许伴随关系的 family，且对应 binding 必须存在
   - `relation.carried_from`
     - 对应 update 必须是 `recurrence`，且 carried source binding 必须存在
   - `relation.derives_index_from`
     - 对应 state 必须是 `index_state`
3. **Supplement Legality**
   - `SemanticSupplement.kind` 只能来自允许集合
   - supplement target 必须指向已有 anchor
   - supplement 不能重述已可由 witness 唯一恢复的结构事实
4. **Role/Law Compatibility**
   - `carry` state 不允许只关联纯 `map` update 而无 `relation.carried_from` /
     `update.ordering` 事实
   - `index_state` 不允许由纯 transient fragment state 无 `relation.derives_index_from`
     或裁决 supplement 地推导得到
   - `selection_state` / `index_state` / `reduction_accumulator`
     的角色组合必须与 `update.law_family` 和 `access.selection_contract` 相容
5. **Anchor/Biding Integrity**
   - `Update.bindings` / `SemanticSupplement.target_anchor_id`
     不能引用缺失 anchor
   - witness target、update anchor、state anchor 必须在同一 canonicalization epoch 中有效
6. **Pass-Window Legality**
   - 若 `tl.semantic_hard_freeze` 之后发生 unsafe mutation 且未 rebind
     companion program 必须已经失效
   - 若 companion program 仍存在，则必须证明当前 pass window 属于 preserve 或 typed rebind

### Validator Decomposition

为了避免一个 monolithic validator 重新变成黑盒，建议拆成三层：

1. **`ValidateSemanticWitnesses`**
   - witness 自身是否 well-formed
2. **`ValidateStatefulSemanticIR`**
   - `SemanticProgram` 结构是否合法
3. **`ValidateSemanticRefinement`**
   - witness 与 `SemanticProgram` 是否满足 refinement contract

其中：

- 当前已有的 `ValidateStatefulSemanticIR` 可继续保留为最小结构 gate
- 新的 `ValidateSemanticRefinement` 负责真正的 abstraction soundness 检查

### Exit-Gate Consequence

一旦进入这一步，`Phase A` 的“完成”标准也应相应升级：

- 不再只是 compile regression 全绿
- 而是：
  - witness producer 稳定
  - witness-to-core projection 有固定 contract
  - refinement validator 能拦住 witness/core mismatch

只有到这一步，`Phase A` 才能真正摆脱“workload 增加时继续补 relation + regression case”的
被动演化模式。

## Task 3: Stage 2.5 - Generic Witness Algebra And Refinement Contract

**Files:**
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- Modify: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- Create: `tilelang_repo/src/transform/validate_semantic_refinement.cc`
- Modify: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tasks/progress.md`
- Status: `2026-04-05` 已实现并验证

- [x] **Step 1: Add failing tests for generic witness production and refinement checks**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'witness or refinement' -q
```

Expected:

- 新测试先 fail
- fail 原因是：
  - 尚无 typed generic witness object
  - 尚无 `ValidateSemanticRefinement`
  - 尚无 preserve / rebind / invalidate 的 machine-checkable enforcement

- [x] **Step 2: Introduce generic witness objects and compatibility projection**

Deliver:

- compiler-internal generic witness algebra
- 当前开放 attrs 先投影成 witness，再由 witness 投影到 semantic core
- 不新增 workload-shaped witness class

- [x] **Step 3: Add generic refinement validation**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'witness_projection or refinement_mismatch' -q
```

Expected:

- validator 能发现 witness/core mismatch
- validator 能拒绝 orphan witness
- validator 能拒绝非法 supplement 重述结构事实

- [x] **Step 4: Mechanize preserve / rebind / invalidate contract**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'hard_freeze or invalidation or rebind_contract' -q
```

Expected:

- unsafe mutation 后 companion IR 整体失效
- 未声明 preserve / typed rebind 的 pass 不能静默保留 semantic companion

- [x] **Step 5: Re-run semantic and baseline verification**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- witness/refinement contract 不回退当前 A1/A2 gate
- compile-path baseline 继续全绿

## Task 4: Stage 2.6 - Typed Semantic Vocabulary And Modular Rule Tables

当前 `Generic Witness Algebra` 已经把 workload-specific relation bag 收敛成闭 vocabulary，
但实现仍然大量依赖：

- `ffi::String`
- `Map<String, Any>`
- pass 内部散落的字符串比较

这在“避免名字匹配猜语义”这个层面已经足够，但在“模块化、可演化、可防膨胀”的层面还不够硬。

因此下一步收口目标不是继续加字符串 allow-list，而是把 `Phase A` 进一步拆成三层：

1. **Semantic Vocab**
   - 统一定义：
     - `WitnessSubjectKind`
     - `WitnessFactAxis`
     - `StateRole`
     - `UpdateLawKind`
     - `SupplementKind`
     - `ContractMode`
     - `BindingKind`
   - FFI / attr 边界仍然以 string form 存在
   - pass 内部一律先 decode 成 typed enum，再做逻辑
2. **Semantic Witness Decoder**
   - 把 `SemanticWitness` 从 raw string payload 解析成内部 typed view
   - `LiftStatefulSemanticIR` 与 `ValidateSemanticRefinement` 不再自己手写
     `"state" && "role"` 这类分派
3. **Semantic Refinement Rules**
   - 把 relation compatibility、contract legality、binding compatibility
     从 validator/pass 内部抽成集中规则模块

### Design Constraints

- 不新增 workload-shaped witness class
- 不回退到名字匹配
- string 只允许存在于：
  - attr key
  - FFI reflection / serialization 边界
  - debug / error message
- 语义判断、规则分派、contract legality 必须使用 typed vocabulary

### Intended Code Shape

- `semantic_program.h/.cc`
  - 保留 FFI object 定义
- `semantic_vocab.h/.cc`
  - 定义 enum class + parse/print helper
- `semantic_witness_decoder.h/.cc`
  - 定义 typed witness view 与 payload decoder
- `semantic_refinement_rules.h/.cc`
  - 定义 relation/update/binding/contract 的合法性规则
- `AnalyzeSemanticStructure`
  - 产出 raw witness 时使用 centralized vocab printer
- `LiftStatefulSemanticIR`
  - 只消费 typed decoder
- `ValidateStatefulSemanticIR` / `ValidateSemanticRefinement`
  - 只消费 typed vocab 与 centralized rule table

Status:

- `2026-04-05` 已实现并验证

Delivered:

- `semantic_vocab.h/.cc`
  - closed enum vocabulary
  - parse / print helper
  - FFI normalization hooks
- `semantic_witness_decoder.h/.cc`
  - raw witness -> typed witness view
  - centralized payload decoder
- `semantic_refinement_rules.h/.cc`
  - relation / binding / contract legality rule table
- `LiftStatefulSemanticIR`
  - 不再直接在 pass 内散落 `"state" && "role"` 风格字符串分派
- `ValidateStatefulSemanticIR` / `ValidateSemanticRefinement`
  - 改为 typed vocab + centralized rule table

Verification:

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'semantic_vocab_normalizes or semantic_vocab_rejects' -q`
  - `2 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `22 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

Verification:

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'witness or refinement or invalidation_contract' -q`
  - `5 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `20 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

## Task 5: Stage 2.7 - Typed Witness Payload Families

`Stage 2.6` 已经把 closed vocabulary、typed decoder 和 centralized rule table 收正了，
但 `SemanticWitness.fact_value` 的高频 payload 仍然主要表现为 pass 内部手拼的
`Map<String, Any>`：

- `state.role -> {"role": ...}`
- `update.law_family -> {"kind": ...}`
- `update.source_set -> {"sources": [...]}`
- `relation.{companion,carried_from} -> {"binding_kind": ...}`

这会带来两个长期问题：

1. payload shape 仍然由各个 pass 自己记忆和手工维护
2. “边界 string / 内部 typed” 还没有贯彻到 payload 层

因此下一步不是继续在 `LiftStatefulSemanticIR` 和 validator 里手写 payload key，而是把
payload 也拆成正式层次：

1. **Typed Payload Builders**
   - `AnalyzeSemanticStructure` 不再手工拼 `Map<String, Any>`
   - 统一通过 payload builder 产出 canonical payload
2. **Typed Payload Decoders**
   - `LiftStatefulSemanticIR` 与 `ValidateSemanticRefinement`
     不再自己按 key 拆 `fact_value`
   - 一律先 decode 成 typed payload view
3. **Payload Normalization Hooks**
   - Python / FFI 边界允许通过集中 normalize/build helper 构造 payload
   - 保持 attr/serialization 边界的 canonical form

### Design Constraints

- 不新增 workload-shaped payload class
- 不把 payload decoder 再散回各个 pass
- payload schema 必须和 `WitnessFactAxis` 对齐
- 若某个 axis 不需要额外值，允许空 payload；不强制保留冗余 `"kind"` 协议
- payload legality 必须由集中模块判定，而不是各 pass 自己猜 key/shape

### Intended Code Shape

- `semantic_witness_payloads.h/.cc`
  - 定义 typed payload family
  - builder / decoder / normalize helper
- `AnalyzeSemanticStructure`
  - 通过 payload builder 发射 canonical witness payload
- `semantic_witness_decoder.h/.cc`
  - 复用 typed payload decoder，而不是自己拆 `fact_value`
- `LiftStatefulSemanticIR`
  - 只消费 typed payload view
- `ValidateSemanticRefinement`
  - 只消费 typed payload view，并验证 payload shape 与 refinement 一致

### Expected First Payload Families

- `StateRolePayload`
- `UpdateLawFamilyPayload`
- `UpdateSourceSetPayload`
- `RelationBindingPayload`
- 必要时允许 `EmptyPayload`

### Expected Verification

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'semantic_payload' -q`
  - 验证 payload builder / normalize hook / rejection behavior
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - witness / lift / refinement 主回归保持通过
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`

Status:

- `2026-04-05` 已实现并验证

Delivered:

- `semantic_witness_payloads.h/.cc`
  - centralized typed payload builder / decoder / normalize helper
- `AnalyzeSemanticStructure`
  - 使用 payload builder 发射 canonical witness payload
- `semantic_witness_decoder.h/.cc`
  - 复用 typed payload decoder，而不是手工按 key 拆 `fact_value`
- `LiftStatefulSemanticIR`
  - 只消费 typed payload view
- `ValidateSemanticRefinement`
  - 改为验证 typed payload shape 与 semantic refinement 一致
- `relation.derives_index_from`
  - 改为 empty payload；移除冗余 `"kind": "index_derivation"` 字段

Verification:

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'semantic_payload' -q`
  - `2 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `24 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

## Task 6: Phase A Closing Hardening Before Phase B

`Phase A` 当前已经满足工程 exit gate，但在进入 `Phase B` 之前，还必须把下面三类
“理论层机制”补成代码里的正式 contract；否则 `Spatial Program IR` 一旦开始引入更多
restructure / companion consumer，`Phase A` 很容易重新退化成“回归 case 绿，但语义真源
边界不够硬”的状态。

这一步不扩 semantic vocabulary，不扩 workload family 支持面，只补三类高杠杆机制：

1. **Machine-checkable preserve / typed_rebind / invalidate contract**
2. **Stronger refinement validator**
3. **Internal state/effect graph**

### Design Goal

进入 `Phase B` 之前，`Phase A` 至少应从“工程上可用的 semantic recovery layer”升级到：

- companion IR 生命周期有正式合同
- witness -> semantic core 的投影有更强的 refinement check
- carry / ordered update / source-set / companion relation 不再只靠扁平对象和局部 relation，
  而有统一的 internal graph normalization skeleton

### Scope Boundaries

这一步**不做**：

- 新 workload family 扩面
- 新 semantic core noun
- `Phase B / C` object 设计
- 把 `Phase A` 直接升级成完整形式化证明系统

这一步**要做**：

1. `typed_rebind` 不再只停留在 freeze schema 中的枚举值
2. `ValidateSemanticRefinement` 继续从“payload/value 一致性校验”升级到：
   - witness coverage
   - graph consistency
   - stronger role/law compatibility
   - typed rebind legality
3. `SemanticProgram` 内部新增 state/effect normalization graph：
   - `StateVersion`
   - `StateDef`
   - `StateUse`
   - `StateJoin`

### Intended Code Shape

- `semantic_program.h/.cc`
  - 新增 `StateVersion / StateDef / StateUse / StateJoin`
  - `SemanticProgram` 持有上述 graph objects
- `semantic_rebind.*`
  - 集中定义 typed rebind plan / helper / apply logic
- `LiftStatefulSemanticIR`
  - 在 semantic core 投影完成后，统一构建 internal state/effect graph
- `ValidateStatefulSemanticIR`
  - 覆盖 graph well-formedness
- `ValidateSemanticRefinement`
  - 校验 witness/core/graph/refreeze contract 一致性
- `TypedRebindBlackholeCompanionPrograms`
  - 作为 audited-safe pass contract 的第一版正式入口

### Required Semantic Contracts

1. **Preserve**
   - `body_hash` 不变
   - graph / witness / semantic core 全部保持一致
2. **Typed Rebind**
   - 允许 `body_hash` 变化
   - 必须显式携带 `rebind_epoch`
   - 必须显式记录 rebind 作用域 / remap 结果
   - rebind 后 witness/core/graph 必须重新自洽
3. **Invalidate**
   - companion attrs 必须整体删除
   - 不允许留下 stale `tl.semantic_program`

### Required Validator Strengthening

在现有 validator 基础上，至少补下面几类检查：

1. **Witness Coverage**
   - 每个 witness 必须被某个 semantic field 或 graph fact 消费
2. **Role/Law Compatibility**
   - `carry` state 必须进入 graph join
   - `index_state` 必须有合法 derivation witness
   - `selection` / `companion` / `carried_from` 必须与 update family 相容
3. **Graph Consistency**
   - `StateDef` / `StateUse` / `StateJoin` 不能引用缺失 state / update / version
   - `UpdateLaw.source_states` 必须和 `StateUse` source set 一致
   - carried / ordered update 必须在 graph 中留下 join 或 effect edge
4. **Typed Rebind Legality**
   - `typed_rebind` 必须有 epoch
   - 必须有 before/after body hash 与 rebind trace
   - typed rebind 后 graph / witness / semantic core 不允许出现 orphan anchor

### Expected Exit State

完成后，`Phase A` 的“完成”定义升级为：

- semantic layer 不只是 compile-path 回归全绿
- companion 生命周期可检查
- rebind 不再是 schema 占位
- validator 能拦住 witness/core/graph mismatch
- Phase B 只消费冻结后的 semantic truth，不再倒逼 Phase A 回头补 matcher

### Implemented Status

当前这 3 项已经按上面的边界落地：

- `SemanticProgram` 已新增：
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- `LiftStatefulSemanticIR` 现在会在 semantic core 投影后统一构建 internal state/effect graph
  - graph 只覆盖 **stateful** update；像 copy path 这类没有显式 semantic state 的 target-less
    `map` update，不强行进入 graph
- `ValidateStatefulSemanticIR` 现在除 role/law 之外，还检查 graph well-formedness：
  - version/def/use/join 引用闭包
  - update target state closure
  - graph kind legality
- `ValidateSemanticRefinement` 现在已增强为：
  - witness coverage 必须完整
  - `update.source_set` 必须对应 `StateUse(source_state)`
  - `relation.companion` / `relation.carried_from` 必须对应 binding + `StateUse`
  - `carry` state 与 `recurrence` update 必须对应 `loop_carried` `StateJoin`
  - `typed_rebind` 必须显式携带
    `previous_body_hash / rebind_epoch / rebind_scope / rebind_trace`
- `TypedRebindBlackholeCompanionPrograms` 已实现为正式 pass：
  - 支持 audited-safe `body_hash_refresh`
  - 支持 state/update remap 计划
  - 会重写 structure / witnesses / semantic program，并刷新 freeze contract

### Implementation Notes

- `loop_carried StateJoin` 的生成不能只看 `carry role` 或显式 `carried_from` relation；
  对 `UpdateLaw.kind == recurrence` 的 ordered update，也必须生成 join。否则 topk / selection
  这类 synthetic recurrence gate 会在 refinement 里误报缺失。
- internal state/effect graph 不应覆盖所有 update，而应只覆盖能绑定到 semantic state 的 update。
  否则像 copy pipeline 里 `target_state == ""` 的 root `map` update 会制造伪造的 orphan version。

### Final Verification

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'state_effect_graph or typed_rebind or missing_loop_carried_join' -q`
  - `4 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `28 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

## Task 7: Research-Guided Phase A Formalization Backlog

`Phase A` 到这里已经按项目的工程和理论边界完成，可以进入 `Phase B`。但如果后续想把它
进一步升级成**学术意义上可证明的 semantic abstraction layer**，接下来的工作不应该是
“再补更多 workload case”，而应是把现有设计 formalize 成可证明对象。

这条线不是当前工程 blocker；它是与 `Phase B / C` 并行的 research track。它的目标不是证明
“任意 lower 后 TIR 都能被自动恢复语义”，而是证明更弱、也更正确的命题：

- 对一个明确限定的 canonical evidence domain，如果某个 workload family 的关键语义能被
  该 evidence domain 观测并归约到 `SemanticProgram` core，那么
  `AnalyzeSemanticStructure + LiftStatefulSemanticIR` 产出的 `SemanticProgram`
  是这个 evidence domain 的 sound abstraction。

### What “Academic Completion” Means Here

对本仓库，学术意义上的 `Phase A` 完成，不是“测试足够多”，而是至少要能正式表述并 defend：

1. `Phase A` 的 concrete semantics 是什么
2. `SemanticProgram` 的 abstract semantics 是什么
3. `alpha / gamma` 或等价 refinement relation 是什么
4. `preserve / typed_rebind / invalidate` 分别保持什么语义合同
5. `Phase A -> Phase B` 的 cutover 为什么是 refinement-preserving

### Research Workstream A: Canonical Evidence Collecting Semantics

第一步不要直接去 formalize 全部 lower 后 TIR，而是 formalize `Phase A` 实际消费的
canonical evidence domain：

- `tl.semantic_witnesses`
- `fragment_regions[*].selection_targets`
- `fragment_regions[*].selection_pairs`
- `fragment_regions[*].arg_reduce_targets`
- `fragment_regions[*].recurrence_edges`
- `fragment_regions[*].update_sources`
- pre-lift semantic seeds 与 hard-freeze contract

这里建议采用 **collecting semantics -> abstraction** 的口径，而不是“直接对 pass 代码做证明”。
更具体地说，应定义：

- 哪些 evidence object 属于合法观测集
- 每种 evidence 代表哪类可观测语义事实
- 哪些事实属于 algorithmic state / update law / carry boundary / selection companion
- 哪些事实不属于 `Phase A`，而应留给 `Phase B / C`

这条路线直接对应抽象解释框架，而不是 ad hoc matcher 叙事。

Repo mapping:

- `AnalyzeSemanticStructure`
- `semantic_witness_payloads`
- `semantic_witness_decoder`

### Research Workstream B: Abstract Semantics of SemanticProgram

第二步 formalize `SemanticProgram` 自身，而不是只把它当序列化对象：

- `Domain / State / Update / UpdateLaw`
- `StateVersion / StateDef / StateUse / StateJoin`
- `SemanticSupplement`
- hard-freeze contract

这里最重要的是把当前工程对象解释成**抽象语义事实**：

- `State.role` 不只是枚举值，而是 abstract state class
- `UpdateLaw.kind` 不只是 tag，而是 abstract transformer family
- `StateJoin(loop_carried)` 不只是 graph edge，而是 recurrence/order fact
- `typed_rebind` 不只是 attr rewrite，而是“保持抽象语义、刷新 anchor/body hash”的受限变换

Repo mapping:

- `semantic_program.h/.cc`
- `semantic_state_effect_graph.*`
- `semantic_rebind.*`

### Research Workstream C: Alpha/Gamma or Refinement Relation

第三步把 `Phase A` 的弱证明目标写成正式关系：

- `alpha : Evidence -> SemanticProgram`
- `gamma : SemanticProgram -> Set[Evidence]`

或者，如果更适合工程实现，也可以写成 forward simulation / refinement relation，而不强求
完整 Galois connection。关键是要明确：

- `LiftStatefulSemanticIR` 实现的是哪个抽象函数
- `ValidateSemanticRefinement` 在检查哪个 soundness obligation
- 哪些 witness/core/graph mismatch 属于“抽象不 sound”
- 哪些 workload/evidence 组合应被 reject，而不是被错误 lift

对当前仓库，最现实的 theorem statement 应该是：

- `ValidateSemanticRefinement(alpha(e))` 成立时，`alpha(e)` 不伪造与 `e` 矛盾的语义事实

而不是：

- `alpha(e)` 是所有程序语义的最精确恢复

### Research Workstream D: Pass Contracts as Semantic Preservation Obligations

第四步把现有 `preserve / typed_rebind / invalidate` 从工程合同升级成正式 proof obligation。

建议的命题拆分：

1. **Preserve theorem**
   - body hash、witness、semantic core、state/effect graph 都保持不变
2. **Typed rebind theorem**
   - 允许 body/anchor 变化
   - 但要求 remap + trace + refreshed graph 后，抽象语义与 rebind 前等价
3. **Invalidate safety theorem**
   - 一旦 companion 无法证明保持语义，就必须整体失效，不能留下 stale truth

这部分不需要一开始就做全编译器证明；更现实的做法是：

- 先对 audited pass family 建立局部 preservation obligation
- 非 audited pass 继续 fail-closed 到 `invalidate`

Repo mapping:

- `TypedRebindBlackholeCompanionPrograms`
- `InvalidateBlackholeCompanionPrograms`
- `ValidateSemanticRefinement`

### Research Workstream E: Phase A -> Phase B Translation Validation

第五步不要一开始就追求“整条编译器全证明”，而是优先补
`SemanticProgram -> SpatialProgram` 的 translation validation。

对本项目，这条线尤其重要，因为真正的当前 blocker 已经是 `Phase B / C`：

- `Phase A` 负责冻结算法语义真相
- `Phase B` 负责 program/task/channel/layout/workpartition
- 如果 `Phase B` 不能被检查为 semantic-refining，那么 `Phase A` 的 formalization 价值会被削弱

因此，更现实的 research path 是：

1. 先让 `Phase A` 有 formal semantics
2. 再给 `Phase B` 写一个 executable refinement validator
3. 最后才考虑整条编译链的更强 theorem

### Research Workstream F: Effect/Resource Typing for Blackhole-Specific Ambiguity

`blackhole.acc` 的历史问题说明，后续 formalization 不能只盯 state/update，还必须补
effect/resource typing 视角。否则很多“算法语义 vs compute scratch / transport resource”的
混淆，最终都会在 `Phase B / C` 重新出现。

对本仓库，最值得 formalize 的 effect/resource 轴是：

- algorithmic state
- transient scratch
- index companion / selection companion
- ordered recurrence / carry boundary
- target/runtime resource boundary

这条线不要求把所有 TT resource 直接拉回 `Phase A`，但要求 `Phase A` 明确知道哪些东西
不是它的职责，并把 effect/resource 边界留给 `Phase B / C`。

### Recommended Order

如果真要沿 academic 路线继续推进，建议顺序是：

1. formalize canonical evidence semantics
2. formalize `SemanticProgram` abstract semantics
3. 写出 `alpha / gamma` 或 refinement relation
4. 把 `ValidateSemanticRefinement` 显式对应到 theorem obligations
5. 做 `Phase A -> Phase B` translation validation
6. 最后再考虑 proof assistant mechanization

### Mechanization Recommendation

如果将来要做 mechanized proof，最合适的切入点不是直接 mechanize 全部 TIR，而是：

- 先 mechanize `tl.semantic_witnesses`
- 再 mechanize `SemanticProgram` 小闭集 core
- 再 mechanize `ValidateSemanticRefinement` 所对应的 soundness lemma

proof assistant 选择上，`Rocq/Coq` 与现有 CompCert 生态最接近，适合较严肃的 compiler proof；
如果目标更偏快速实验，也可以先用 Lean/Isabelle 做较小核心的试验性 formalization。

### Primary References

- Cousot and Cousot, *Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints*, POPL 1977  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml
- Cousot, *Types as Abstract Interpretations*, POPL 1997  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL97.shtml
- Cousot and Cousot, *Abstract Interpretation Algorithms and Proof Methods*, POPL 2014  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL14.shtml
- Darais and Van Horn, *Constructive Galois Connections: Taming Proofs of Static Analyses*, ICFP 2016  
  https://arxiv.org/abs/1507.03559
- Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, 2020  
  https://arxiv.org/abs/2002.11054
- Wang et al., *Homeostasis: A Compiler Correctness Framework for Self-Regulating Program Analysis*, 2021  
  https://arxiv.org/abs/2106.01768
- Lopes et al., *Alive2: Bounded Translation Validation for LLVM*, PLDI 2021  
  https://pldi21.sigplan.org/details/pldi-2021-papers/5/Alive2-Bounded-Translation-Validation-for-LLVM
- Leroy, *Formal Verification of a Realistic Compiler*, CACM 2009  
  https://www.cs.cmu.edu/~15811/papers/compcert.pdf
- Lucassen and Gifford, *Polymorphic Effect Systems*, 1988  
  https://research.ibm.com/publications/polymorphic-effect-systems
- Tofte and Talpin, *Region-Based Memory Management*, 1997  
  https://www.sciencedirect.com/science/article/pii/S0890540196926139

## Task 8: Repo-Driven Theoretical Definition of Phase A and Its Boundary to Phase B

这一节不再从“论文有哪些概念”出发，而是直接从当前仓库已经暴露出来的问题反推
`Phase A` 为什么必须长成现在这套结构。

### The Problem That Forces Phase A to Exist

Blackhole 当前真正的问题不是“少一个 kernel emitter”，而是三类 truth 在旧链路里长期混杂：

1. **algorithmic truth**
   - carry
   - selection companion
   - index flow
   - ordered recurrence
   - value/update dependency
2. **spatial program truth**
   - task
   - channel
   - layout
   - work partition
   - phase boundary
3. **target/runtime truth**
   - TT resource
   - transport / sync
   - ABI / common-runtime / per-work bindings

当这三类 truth 被放在同一层时，后段就会被迫做两种错误事情：

- 从已经被压碎的 lower 后 IR 里反向猜算法语义
- 把本应属于 task/layout/resource 的事实重新塞回语义层

`blackhole.acc` 的历史问题正是这个混层的具体表现：同一组 IR 对象同时承载了
algorithmic state、compute scratch 和 target-local resource intent。

所以 layered IR 不是偏好，而是由问题结构直接推出的：

- `Phase A` 冻结 algorithmic truth
- `Phase B` 一等化 spatial program truth
- `Phase C` 物化 target/runtime truth

### Why Phase A Must Be an Abstraction Layer

一旦接受上面的分层，`Phase A` 就不能再被理解成“更聪明的 recovery pass”。

原因很简单：

- lower 后 IR 已经不会稳定保留全部前端算法语义
- 但 `Phase A` 又必须产出一个对后续 compiler pass 可消费、可冻结、可验证的真相层

因此，`Phase A` 唯一合理的定义是：

- 它不是 concrete semantics 本身
- 它是对 canonical evidence domain 的**有界抽象**

对本仓库，这个定义落成下面这条链：

- `AnalyzeSemanticStructure`
  - 负责从可验证的 IR/analysis facts 里收集 canonical evidence
- `tl.semantic_witnesses`
  - 作为 compiler-internal evidence carrier
- `LiftStatefulSemanticIR`
  - 把 evidence 投影成 `SemanticProgram`
- `ValidateSemanticRefinement`
  - 检查这个投影没有伪造与 evidence 矛盾的语义事实

这意味着 `Phase A` 的职责不是“恢复全部真实程序意义”，而是：

- 用最小 semantic core 冻结**足够支撑后续编译**的算法语义真相

### Why Phase A Must Stay Small and Workload-Agnostic

如果 `Phase A` 试图直接吸收 workload noun，它很快就会退化成无限膨胀的 matcher registry。

因此，当前设计必须坚持：

- `State.role` 是抽象语义角色，而不是 workload 类型名
- `UpdateLaw.kind` 是抽象变换族，而不是某个 kernel family 标签
- witness family 表示的是 evidence axis，而不是“当前例子里长什么样”

这就是为什么下面这些概念适合进入 `Phase A`：

- `carry`
- `reduction_accumulator`
- `selection_state`
- `index_state`
- `select`
- `recurrence`
- `source_set`
- `companion`
- `carried_from`

而这些概念不应直接成为 `Phase A` vocabulary：

- `flash_attn_state`
- `topk_selector`
- `paged_decode_route`
- `fusedmoe_dispatch`

前者是可跨 family 复用的 semantic axis，后者是 workload-shaped noun。

### Why Witnesses and Validators Are Structurally Necessary

从仓库约束出发，`Phase A` 还有一条硬要求：

- 不允许基于名字匹配恢复语义
- 不允许把关键信息继续留给后段猜

这直接推出 witness/core/validator 三层结构，而不是纯 attrs 拼接：

1. **witness layer**
   - 保存 canonical evidence
   - 作为“为什么能得出这个语义结论”的可检查依据
2. **semantic core**
   - 提供小闭集、typed、workload-agnostic 的 compiler truth
3. **refinement validator**
   - 保证 witness -> core 的投影是 sound 的

如果缺其中任一层，`Phase A` 都会退化：

- 没有 witness：semantic core 只是 pass 的臆断结果
- 没有 core：analysis attrs 永远无法成为稳定 compiler contract
- 没有 validator：semantic layer 只是“看起来合理”的缓存，而不是可检验真相

### Why Internal State/Effect Graph Is Part of the Semantics

只靠 `State` 与 `Update` 还不足以表达本仓库真正关心的结构事实：

- 哪个 version 从哪里来
- 哪个 update 实际消费了哪个 state
- 哪个 carried/ordered relation 真的成立
- 哪个 recurrence boundary 是真实 join，而不是局部 hints

因此，`StateVersion / StateDef / StateUse / StateJoin` 不是实现细节，而是
`Phase A` 抽象语义的一部分。它们的作用是把：

- flat object summary

收成：

- normalized state/effect skeleton

这一步对 selection / recurrence / carry 尤其重要，因为这些 family 的关键真相不是“有几个对象”，
而是“这些对象之间怎样以 version/use/join 方式连接”。

### Why Preserve / Typed Rebind / Invalidate Is Not Optional

一旦 `Phase A` 被定义为冻结后的 compiler truth，它就不能容忍后续 pass 在对象 identity 变化后
继续把旧 truth 当真。

于是又直接推出 companion lifecycle contract：

- **preserve**
  - 当 body、anchors、graph 都保持一致时，semantic truth 可直接沿用
- **typed_rebind**
  - 当结构变化可被受控 remap 解释时，允许刷新 anchor/body hash，并重建 graph
- **invalidate**
  - 当无法证明语义保持时，必须整体失效

这不是工程偏好，而是 semantic layer 想长期存在的必要条件。否则 `Phase A` 只是一次性 analysis
结果，而不是后续 `Phase B` 可以信赖的冻结真相。

### The Necessary Boundary Between Phase A and Phase B

从上面的定义可以直接推出 `Phase B` 的边界：

- `Phase A` 冻结的是 algorithmic truth
- `Phase B` 负责把这份 truth 投影成 spatial program structure

因此，`Phase B` 不应该重新决定：

- 哪些是 carry state
- 哪些 update 属于 recurrence/select
- 哪些 companion/source relations 成立

`Phase B` 只应该决定：

- 这些 semantic facts 应该如何被组织成
  - `ProgramPhase`
  - `Task`
  - `Channel`
  - `Layout`
  - `WorkPartition`
  - `SyncEdge`

所以 `Phase A -> Phase B` 的正确关系不是“Phase B 再恢复一遍更具体的语义”，而是：

- `Phase B` 对 `Phase A` 做 **refinement by organization**

也就是说，`Phase B` 增加的是 spatial structure，不是新的 algorithmic truth。

### Theorem-Shaped Statement for This Repo

按当前仓库的真实边界，最值得追求的命题不是一个过强的“万能恢复器 theorem”，而是：

- 对属于 canonical evidence domain 的程序，如果 `ValidateSemanticRefinement` 通过，
  那么 `SemanticProgram` 中的 state/update/law/source/companion/carry facts
  都是对 evidence 的 sound abstraction
- 对 audited-safe 的 preserve / typed_rebind pass family，
  semantic truth 要么保持不变，要么被受控地重绑，要么被整体失效
- `Phase B` 只能组织这些已冻结的 semantic facts，不能重新发明或修正它们

这就是本仓库里 `Phase A` 最自然、也最有价值的理论化方向。

## Task 9: Theorem and Obligation Checklist for Phase A

如果要把上面的 repo-driven 定义继续收成更正式的研究说明，最合适的下一层不是直接写
proof assistant 代码，而是先把 theorem、precondition、proof obligation 和可执行检查点
写成固定 checklist。

这一节的目标是把当前 `Phase A` 从：

- “理论方向已经讲清”

推进到：

- “已经知道下一步具体要证明什么，以及哪些现有 pass/validator 在承接这些命题”

### Core Mathematical Objects

对本仓库，`Phase A` 至少需要固定下面 4 类对象：

1. **Canonical Evidence Domain `E`**
   - 由 `tl.semantic_witnesses`、selected fragment attrs、semantic seed、hard-freeze precondition
     组成
2. **Abstract Semantic Domain `A`**
   - 由 `SemanticProgram` core 与 state/effect graph 组成
3. **Abstraction Function `alpha : E -> A`**
   - 由 `AnalyzeSemanticStructure + LiftStatefulSemanticIR` 实现
4. **Refinement Checker `R(e, a)`**
   - 由 `ValidateSemanticRefinement` 的 obligation family 近似实现

在当前仓库语境里，不必强求一开始就把 `gamma` 写成完整集合语义；更现实的落点是：

- 先写出 `R(e, alpha(e))`
- 让 validator 与 theorem checklist 对齐

### Theorem T1: Evidence Well-Formedness

**命题**

- 进入 `LiftStatefulSemanticIR` 的 canonical evidence 必须满足 vocabulary closure、payload closure、
  anchor closure 和 canonicalization-point precondition。

**为什么它重要**

- 如果 `E` 自身不封闭，后续所有 abstraction theorem 都会失去对象边界。

**当前对应实现**

- `semantic_vocab`
- `semantic_witness_decoder`
- `semantic_witness_payloads`

**还缺什么**

- 把“哪些 evidence source 算 canonical”写成更正式的 precondition，而不是只在 pass 顺序里隐含

### Theorem T2: Lift Soundness

**命题**

- 对任意满足 precondition 的 evidence `e`，若 `a = alpha(e)`，则 `a` 中的
  `State.role / UpdateLaw.kind / source_set / companion / carried_from / loop_carried`
  不会与 `e` 矛盾。

**当前仓库版本的可接受表述**

- `ValidateSemanticRefinement(alpha(e))` 通过时，`alpha(e)` 是 `e` 的 sound abstraction

**为什么它重要**

- 这是 `Phase A` 能否被称为 semantic layer 的最低门槛。

**当前对应实现**

- `LiftStatefulSemanticIR`
- `ValidateSemanticRefinement`

**还缺什么**

- 把“sound abstraction”的语义对象写成正式定义，而不是只用 validator 行为去代替定义

### Theorem T3: Graph Soundness

**命题**

- `StateVersion / StateDef / StateUse / StateJoin` 必须是 `SemanticProgram` 中
  state/update facts 的一致 normalization，而不是额外发明的新真相。

**更具体地说**

- graph 不能引入 core/witness 中不存在的 stateful fact
- graph 必须为 carried/order/source/companion facts 提供一致的 normalization skeleton
- target-less update 不应被伪造为 stateful version producer

**为什么它重要**

- 如果 graph 自己会发明语义，那它就不是 normalization layer，而成了第二 semantic channel。

**当前对应实现**

- `semantic_state_effect_graph`
- `ValidateStatefulSemanticIR`
- `ValidateSemanticRefinement`

**还缺什么**

- 一份 graph-as-normalization 的正式语义说明

### Theorem T4: Contract Preservation

**命题**

- audited-safe `preserve` pass family 不改变 `Phase A` 抽象语义。

**当前仓库版本的实际含义**

- body hash、witness、core、graph 都保持一致
- 不允许 silent drift

**为什么它重要**

- 没有这个 theorem，semantic freeze 只是“建议”，不是 contract。

**当前对应实现**

- `tl.semantic_hard_freeze`
- `ValidateSemanticRefinement`

**还缺什么**

- audited-safe preserve pass family 的显式白名单与逐 pass obligation

### Theorem T5: Typed Rebind Preservation

**命题**

- 对允许的 rebind scope，只要 remap/trace/pre-post hash/graph refresh 满足 contract，
  rebind 前后的抽象语义应视为等价。

**为什么它重要**

- 否则 `typed_rebind` 只是“比较安全的重写器”，而不是有语义保证的 contract mode。

**当前对应实现**

- `TypedRebindBlackholeCompanionPrograms`
- `semantic_rebind`
- `ValidateSemanticRefinement`

**还缺什么**

- 把“语义等价”具体化成 state/update/law/graph 级别的 preservation relation
- 给不同 `rebind_scope` 写出更细的 obligation 表

### Theorem T6: Invalidation Safety

**命题**

- 当 preserve/rebind obligation 无法被证明时，整体 invalidation 是 sound 的 fail-closed 行为。

**为什么它重要**

- 这保证编译器在不能维护 semantic truth 时，至少不会继续传播 stale truth。

**当前对应实现**

- `InvalidateBlackholeCompanionPrograms`

**还缺什么**

- 更明确地区分“precision loss”与“soundness risk”，避免把所有困难都粗暴归为 invalidate

### Theorem T7: Phase A to Phase B Refinement

**命题**

- `Phase B` 生成的 `SpatialProgram` 必须 refinement-preserve `Phase A` 已冻结的 algorithmic truth。

**当前仓库里的正确解释**

- `Phase B` 只能组织 semantic facts
- 不能重命名事实的意义
- 不能引入与 `Phase A` 矛盾的新 algorithmic truth

**为什么它重要**

- 否则 `Phase A` 的 formalization 只是局部漂亮，到了 `Phase B` 仍可能重新混层。

**当前对应实现**

- 还没有代码实现；这是 `Phase B` 最值得最早补的 validator 方向

**还缺什么**

- `SpatialProgram` 的 formal object model
- `SemanticProgram -> SpatialProgram` 的 executable refinement checker

### Theorem T8: Rejection Discipline

**命题**

- 对不属于 canonical evidence domain，或不能 soundly 归约到 current semantic core 的 workload/evidence，
  compiler 必须 reject / invalidate / defer，而不是猜。

**为什么它重要**

- 没有 rejection discipline，`Phase A` 最终一定会重新滑回 heuristic recovery。

**当前对应实现**

- 部分体现在 validator / invalidate contract 中

**还缺什么**

- 一份明确的 “reject rather than guess” 入口清单
- 把 rejection 分成：
  - evidence 不足
  - core 不足
  - 属于 Phase B/C 的 truth

### Research Deliverables Worth Pursuing Next

如果真要把 `Phase A` 继续往 research artifact 推，最值得追加的 deliverable 是：

1. **Formal semantics note**
   - 写清 `E / A / alpha / refinement relation`
2. **Obligation matrix**
   - 每个 theorem 对应哪些 pass、哪些 validator、哪些 fail-closed path
3. **Phase B refinement validator**
   - 先做 executable version，不急着 mechanize
4. **Small mechanized core**
   - 只 mechanize witness/core/validator 的最小闭集，而不是整个 compiler

### Minimal “Paper Honest” Claim

如果今天就必须对外做一个学术上诚实的表述，最合理的版本应该是：

- 我们没有证明一个万能语义恢复器
- 我们定义并实现了一个针对 canonical evidence domain 的 bounded semantic abstraction layer
- 它带有 typed witness、normalized state/effect graph、以及可执行的 refinement contract
- 它为后续 `SpatialProgram` refinement 提供了冻结的 algorithmic truth boundary

这个表述既不夸大，也不会低估当前设计真正已经达到的高度。

## Task 10: Formal Semantics Note Skeleton for This Repo

为了避免后续 research track 再次退回“概念很多但对象不固定”，这里直接给出一版适合本仓库的
`formal semantics note` 骨架。它不是第二份总设计，而是将来如果要写技术报告、research memo
或 mechanization note，最自然的展开顺序。

### Section 1: Scope and Claim

先把 claim 收窄到本仓库真正能 defend 的版本：

- 研究对象不是任意 lower 后 TIR 的全语义恢复
- 研究对象是 `Phase A` 消费的 canonical evidence domain
- 目标不是 completeness
- 目标是 bounded semantic abstraction 的 soundness

建议固定成下面这个 claim：

- 对满足 canonical evidence precondition 的程序，`Phase A` 产生的 `SemanticProgram`
  是该 evidence domain 上的 sound abstraction；后续 audited preserve/rebind pass 不破坏该抽象，
  `Phase B` 只能对其做 organization-preserving refinement。

### Section 2: Concrete Objects

这一节只定义本仓库真正拿来做 `Phase A` 输入的对象，不扩到全部 compiler IR。

建议固定：

- `E_seed`
  - pre-lift semantic seeds
- `E_frag`
  - selected fragment attrs
- `E_w`
  - `tl.semantic_witnesses`
- `E_freeze_pre`
  - lift 之前可要求的 freeze/canonicalization precondition

合起来得到：

- `E = (E_seed, E_frag, E_w, E_freeze_pre)`

关键点：

- `E` 是 collecting domain
- `E` 只包含 `Phase A` 实际依赖的事实
- 不把 `Phase B/C` 才需要的 spatial/target truth 混进来

### Section 3: Abstract Objects

这一节定义 `A`，也就是 `Phase A` 的抽象语义对象。

建议固定：

- `A_core`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `SemanticSupplement`
- `A_graph`
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- `A_contract`
  - `preserve`
  - `typed_rebind`
  - `invalidate`

合起来得到：

- `A = (A_core, A_graph, A_contract)`

关键点：

- `A_graph` 是 normalization，不是第二 semantic channel
- `A_contract` 是 companion truth 的生命周期语义，而不是纯工程元数据

### Section 4: Semantic Meaning of Each Abstract Component

这里不写实现细节，而写语义解释。

至少要明确：

- `State.role`
  - 表示哪类 algorithmic state
- `UpdateLaw.kind`
  - 表示哪类 abstract update family
- `source_set`
  - 表示该 update 语义上依赖哪些 state
- `companion`
  - 表示 selection/value-index 之类的 companion relation
- `carried_from`
  - 表示 recurrence/carry 依赖
- `StateJoin(loop_carried)`
  - 表示 ordered recurrence boundary

这一节的目标不是列举字段，而是回答：

- “这个对象在抽象语义里到底是什么意思”

### Section 5: Abstraction and Validation

这一节定义：

- `alpha : E -> A`
- `R(e, a)`

对本仓库的最现实写法是：

- `alpha` 由 `AnalyzeSemanticStructure + LiftStatefulSemanticIR` 实现
- `R(e, a)` 由 `ValidateSemanticRefinement` 近似实现

建议明确两条关系：

1. `alpha` 负责从 evidence 生成 semantic abstraction
2. `R` 负责检查生成结果是否满足 soundness obligation

这样后面不管是否引入完整 `gamma`，至少对象关系已经定了。

### Section 6: Soundness Theorems

这一节直接复用 `Task 9` 的 theorem checklist，但改成更像 report 的组织：

1. Evidence well-formedness
2. Lift soundness
3. Graph soundness
4. Preserve theorem
5. Typed rebind theorem
6. Invalidation safety
7. `Phase A -> Phase B` refinement theorem
8. Rejection discipline

每个 theorem 都应写：

- statement
- precondition
- checked by
- still missing

### Section 7: Failure Modes and Explicit Non-Goals

这一节很重要，因为它决定论文/技术报告是否诚实。

建议明确写出：

- 本工作不证明 arbitrary workload completeness
- 本工作不证明从任意 lower 后 IR 自动恢复所有非平凡语义
- 本工作不保证没有 precision loss
- 本工作允许 reject / invalidate / defer
- 本工作当前不证明 `Phase C` target materialization correctness

同时列出真实 failure mode：

- evidence 不足
- current semantic core 不足
- truth 属于 `Phase B/C`
- companion contract 被破坏

### Section 8: Bridge to Phase B

这一节只讲一个核心句子：

- `Phase B` 不是 semantic recovery layer，而是 organization-preserving refinement layer

然后展开成：

- `Phase B` 允许增加 task/channel/layout/workpartition structure
- `Phase B` 不允许修改 `Phase A` 冻结的 algorithmic truth
- `Phase B` 最需要的是 executable refinement validator，而不是新的 heuristic matcher

这会把 `Phase A` formalization 和主线工程推进真正接起来，而不是停在 `Phase A` 自我完结。

### Section 9: Minimal Mechanization Plan

如果将来真要 mechanize，这一节应该非常克制。

推荐最小顺序：

1. mechanize witness vocabulary and payload family
2. mechanize `SemanticProgram` core
3. mechanize state/effect graph normalization invariant
4. mechanize `ValidateSemanticRefinement` 对应的核心 lemma

不要直接 mechanize：

- 全部 TIR
- 全部 Phase B/C
- 全编译器执行语义

因为那会把研究重心从“证明 `Phase A` 是什么”转成“和整个 compiler 规模作战”。

### Immediate Next Research Artifacts

如果要真正开始沿这条线做事，建议顺序是：

1. 把上面的 skeleton 复制成单独 `formal semantics note`
2. 为 `Task 9` 的每个 theorem 做一页 obligation matrix
3. 在 `Phase B` 开始前，先定义 `SemanticProgram -> SpatialProgram` 的 refinement interface
4. 只有当这三步稳定后，再考虑 mechanization

## Task 11: Semi-Formal Definitions of E, A, alpha, and R

为了让上面的 `formal semantics note` skeleton 不是只有目录，这里先给出一版适合本仓库的
半正式定义。目标不是一次把它写成可 mechanize 数学对象，而是把对象边界固定到足够明确，
使后续 theorem/validator/Phase B refinement 都能引用同一套定义。

### Definition D1: Canonical Evidence Domain `E`

对本仓库，`Phase A` 的 concrete 输入不定义为“任意 lower 后 TIR”，而定义为一个有限 evidence tuple：

- `E = (E_seed, E_frag, E_w, E_pre)`

其中：

- `E_seed`
  - pre-lift semantic seed 集合
  - 例如：tile op family、reduction/selection/recurrence skeleton、device-program membership
- `E_frag`
  - selected fragment analysis attrs
  - 例如：
    - `selection_targets`
    - `selection_pairs`
    - `arg_reduce_targets`
    - `recurrence_edges`
    - `update_sources`
- `E_w`
  - canonicalized witness multiset，也就是 `tl.semantic_witnesses`
- `E_pre`
  - 使 lift 合法的 precondition
  - 例如：
    - 已经过了允许的 canonicalization point
    - witness vocabulary/payload/anchor closure 成立
    - companion 没被 invalidated

这个定义的关键点是：

- `E` 是 **Phase A actually consumes**
- `E` 不是全部 IR 执行语义
- `E` 是可观察、可枚举、可 fail-fast 的 collecting domain

### Definition D2: Abstract Semantic Domain `A`

`Phase A` 的输出定义为：

- `A = (A_core, A_graph, A_contract)`

其中：

- `A_core`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `SemanticSupplement`
- `A_graph`
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- `A_contract`
  - `preserve`
  - `typed_rebind`
  - `invalidate`

这里有两个刻意的限制：

- `A_graph` 只允许表达 core 的 normalization，不允许引入第二 semantic truth
- `A_contract` 是抽象 truth 的 lifecycle 语义，不只是实现元数据

### Definition D3: State and Update Meaning

为了避免 `A_core` 继续被当成“带枚举的对象树”，这里把最关键的两个对象先定成语义解释：

- `State(name, role, scope, anchors)`
  - 表示一个 algorithmic state class 的抽象代表
  - `role` 限定它在算法语义中的职责，而不是 workload noun
- `Update(name, target_state, law, bindings)`
  - 表示作用在一个 algorithmic state 上的抽象变换实例
  - `law.kind` 表示该变换属于哪一类抽象 update family

于是：

- `carry` 表示跨 step / loop boundary 保持的 state class
- `selection_state` 表示 selection/value flow 的抽象 state
- `index_state` 表示 selection companion / index flow 的抽象 state
- `recurrence` 表示带 ordered carry 结构的 abstract update family
- `select` 表示带 value/index companion 关系的 abstract update family

### Definition D4: Graph Meaning

`A_graph` 的对象语义约束为：

- `StateVersion`
  - 一个 state 的抽象版本点
- `StateDef`
  - 一个 version 的定义来源
- `StateUse`
  - 一个 update 对某个 state version 的抽象使用
- `StateJoin`
  - 对 carried / ordered boundary 的抽象 join fact

更强的一句定义是：

- `A_graph` 不是额外真相，而是把 `A_core` 中已经存在的 state/update 事实
  规范化为 version/def/use/join skeleton

因此：

- 如果 graph 引入 core/witness 中不存在的 stateful fact，则 graph 不 sound
- 如果 graph 漏掉 `Phase A` 明确承诺的 carried/order/source/companion fact，则 graph 不 complete
  于当前 abstraction contract

### Definition D5: Abstraction Function `alpha`

对本仓库，`alpha` 不定义成一个单独纯函数实现文件，而定义成 pass composition：

- `alpha = BuildWitnesses ; LiftSemanticCore ; NormalizeStateEffect`

具体映射到实现：

- `AnalyzeSemanticStructure`
  - 负责从 seeds/fragment attrs 中生成 canonical witness input
- `LiftStatefulSemanticIR`
  - 负责 witness -> core projection
- `semantic_state_effect_graph`
  - 负责 core -> normalized graph projection

所以更具体地说：

- `alpha(E)` 的结果就是当前 `tl.semantic_program`

但这里有个重要限制：

- `alpha` 只对满足 `E_pre` 的 evidence 定义
- 不满足 precondition 的输入，不要求 `alpha` 返回有意义 abstraction；应 reject / invalidate / defer

### Definition D6: Refinement Predicate `R(E, A)`

`R(E, A)` 定义为：

- `A` 不伪造与 `E` 矛盾的 semantic facts
- `A` 的 graph 是 `A_core` 的合法 normalization
- `A_contract` 与当前 companion lifecycle 一致

在当前仓库里，`R` 由多个 obligation 组成，而不是单一 bool 定义：

- role/law compatibility
- source/companion/carried relation consistency
- loop-carried join consistency
- graph closure
- hard-freeze contract legality
- typed rebind legality
- witness coverage

当前实现上，`ValidateSemanticRefinement` 是 `R` 的 executable approximation。

### Definition D7: Soundness Target

基于上面的对象，当前仓库最值得追求的 soundness target 是：

- 对任意满足 precondition 的 `E`，若 `A = alpha(E)` 且 `R(E, A)` 成立，
  则 `A` 中的 semantic facts 可以被后续 compiler 当作冻结后的 algorithmic truth 使用

注意这里故意没有承诺：

- `A` 是最精确 abstraction
- `A` 覆盖任意未来 workload
- `A` 保留全部 lower 前程序意义

这个 target 和当前工程边界是一致的，也更适合作为后续 theorem 的母命题。

### Definition D8: Phase A to Phase B Interface

给 `Phase B` 的输入接口可以直接写成：

- `I_B = (A_core, A_graph, A_contract)`

也就是说，`Phase B` 的输入不是 raw witness，也不是 fragment attrs，而是：

- 已冻结的 algorithmic truth
- 以及它的 normalized state/effect skeleton

因此 `Phase B` 的 refinement obligation 可以先写成：

- 对任意 `SpatialProgram S`，若 `LowerToSpatialProgram(A) = S`，则 `S` 只能组织 `A` 的 truth，
  不能修改 `A` 已承诺的 semantic meaning

这就是 `Phase A -> Phase B` interface 的半正式版本。

### What This Buys Us

把 `E`、`A`、`alpha`、`R` 固定下来以后，后面的工作会突然清楚很多：

- `Task 9` 的 theorem checklist 都有了共同对象
- `ValidateSemanticRefinement` 可以继续按 `R` 的 obligation family 去扩
- `Phase B` 可以围绕 `I_B` 去设计 refinement validator
- 后续若做 mechanization，也知道应该 mechanize 哪些最小对象，而不是盲目扩到全编译器
