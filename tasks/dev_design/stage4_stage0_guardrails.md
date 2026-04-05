# Stage 4 Stage 0: P0 Guardrails And Cutover Gates

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 layered IR 正式落地前，先把单一真源 cutover、semantic lift 边界、device-program registry、early semantic capture 和 deletion gates 固定下来。

**Architecture:** `Stage 0` 不直接解决 `flash-attn` correctness，本阶段只负责建立迁移护栏，避免后续 Phase A/B/C 一边新增 companion IR 一边继续让 legacy attrs 长成第二真源。

**Tech Stack:** TileLang transform passes, TVM PrimFunc / IRModule attrs/global_infos, pytest

---

## Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## Task 0: Stage 0 - P0 Guardrails And Cutover Gates

**Files:**
- Create: `tilelang_repo/src/transform/collect_device_programs.cc`
- Create: `tilelang_repo/src/transform/project_semantic_seeds.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tasks/progress.md`

- [x] **Step 1: Establish the migration guardrails**

Deliver:

- `IRModule.global_infos["tl.device_programs"]` 作为唯一 module-scope device-program registry
- `PrimFunc.attrs["tl.semantic_seeds"]` 作为 pre-lift typed input 通道
- A1 `post-lift hard freeze` 规则
- unsafe TIR mutation 导致 `tl.semantic_program / tl.spatial_program / tl.tt_program` 整体失效的 contract
- compatibility deletion gates checklist

Implemented:

- `tilelang_repo/src/transform/common/semantic_program.h`
  - 定义 `tl.device_programs`、`tl.semantic_seeds`、`tl.semantic_hard_freeze`
  - 定义 `tl.semantic_program` / `tl.spatial_program` / `tl.tt_program`
  - 定义 `tl.companion_invalidation_reason`
- `tilelang_repo/src/transform/collect_device_programs.cc`
  - 在 `SplitHostDevice` 前为 Blackhole `PrimFunc` 建立 module-scope device-program registry
- `tilelang_repo/src/transform/project_semantic_seeds.cc`
  - 投影最小 pre-lift semantic seeds
  - 定义 hard-freeze state=`pre_lift_seeded`
  - 定义 unsafe mutation 触发 companion IR 整体失效的 contract
- `tilelang_repo/tilelang/engine/phase.py`
  - 在 Blackhole 主链 `SplitHostDevice` 前接入
    `AnnotateDeviceRegions -> ProjectSemanticSeeds -> CollectDevicePrograms`
- `tilelang_repo/tilelang/transform/__init__.py`
  - 暴露 `CollectDevicePrograms` / `ProjectSemanticSeeds` / `InvalidateBlackholeCompanionPrograms`

- [x] **Step 2: Add Stage 0 structure tests**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'device_program_registry or semantic_seeds or hard_freeze' -q
```

Expected:

- `tl.device_programs` 能被查询
- `tl.semantic_seeds` 在 semantic lift 前可见
- lift 之后 unsafe mutator 会显式拒绝或触发整体失效

Status:

- `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
  - `test_device_program_registry_is_collected_before_split_host_device`
  - `test_semantic_seeds_are_projected_before_semantic_lift`
  - `test_hard_freeze_invalidates_companion_programs_after_unsafe_mutation`
- 验证结果：`3 passed`

- [x] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

Status:

- `test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

- [x] **Step 4: Stage 0 exit gate**

Only proceed when:

- `CollectDevicePrograms` 已在 `SplitHostDevice` 前建立 registry
- `ProgramPhase` 的稳定宿主不再依赖单 `PrimFunc.attrs`
- `SemanticSeed` 输入通道和 deletion gates 已有结构测试
- shared zero-regression baseline 全绿

Exit status:

- 本阶段的 migration guardrails 已在当前 Blackhole 主链落地
- 本阶段不负责 semantic role recovery 精度；后续如果 Phase A 语义分类需要更多结构信号，应扩 attrs/schema，不在 Stage 0 回退到名字匹配
- Phase A 可以开始承接 `SemanticProgram` typed companion IR 本体
