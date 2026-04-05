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
