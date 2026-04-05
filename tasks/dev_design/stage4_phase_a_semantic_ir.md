# Stage 4 Phase A: Stateful Semantic IR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不回退 copy / GEMM / current compile-path 的前提下，建立 `SemanticProgram` 为中心的最小语义层，并扩到能显式承接 `flash-attn` 的 carry/update 语义和至少一个 non-attention gate。

**Architecture:** 先做 `Phase A1` 的最小语义 core 和 hard freeze，再做 `Phase A2` 的 wider `AccessMap / UpdateLaw`、typed supplement 和 non-attention semantic gate。`Phase A` 的职责是冻结算法语义真相，而不是提前发明 Spatial/TT target policy。

**Tech Stack:** TileLang transform passes, TVM Object system, pytest

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
- 当前为避免名字匹配，`AnalyzeBlackholeFragmentRegions` 已向 `fragment_buffers`
  补充 typed `is_integer` 结构信号，`AnalyzeSemanticStructure` 用这份 typed attr
  恢复 `index_state`
- `flash-attn` 当前已在 semantic layer 和 pipeline gate 上稳定看到：
  - algorithmic carry / reduction state
  - transient compute scratch
  的抽象角色分离
- 当前仍有未完成项：
  - A2 schema 已落地
  - 但 semantic recovery 精度仍在继续收紧，尤其是 `selection_state` 这类较粗角色的结构化区分还未完成
- 当前验证：
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -q`
    - `11 passed`
  - `pytest testing/python/transform/test_blackhole_semantic_ir.py -k 'recovers_index_state_from_integer_ir_not_names' -q`
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
