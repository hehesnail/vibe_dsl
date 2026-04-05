# Stage 4 Phase C: TT Target IR And New-Mainline Cutover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `SpatialProgram` 映射到 `TTProgram`，由 `MaterializeTTExecutableSpec` 接管物化，清除 compatibility fallback，并在新主链下继续做 family expansion。

**Architecture:** `Phase C1` 先建立最小 TT target object 集并完成 copy/GEMM cutover，`Phase C2` 再按 deletion gates 清 compatibility 路径并解决 `flash-attn` correctness，最后在新主链下扩到更多 workload family。

**Tech Stack:** TileLang transform passes, Blackhole codegen/runtime, TT-Sim, pytest

---

## Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## TT-Sim Runtime Gate

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
```

## Task 4: Stage 4 - Phase C1 TT Target IR And Materialization Cutover

**Files:**
- Create: `tilelang_repo/src/transform/common/tt_target_program.h`
- Create: `tilelang_repo/src/transform/common/tt_target_program.cc`
- Create: `tilelang_repo/src/transform/lower_spatial_program_to_tt_target.cc`
- Create: `tilelang_repo/src/transform/validate_tt_target_program.cc`
- Create: `tilelang_repo/src/transform/materialize_tt_executable_spec.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Introduce minimal TT target object set**

Required objects:

- `TTProgram`
- `TTKernel`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTHardwareModelStub`

- [ ] **Step 2: Move copy / GEMM materialization to `MaterializeTTExecutableSpec`**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py -k 'copy or gemm or materialize' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
```

Expected:

- `ExecutableSpec` / `KernelSpec` 只从 TT target truth 物化
- `blackhole.runtime_args` / `common_runtime_args` / `cb_configs` / `core_plan` 只作为 compatibility projection 存在
- copy / GEMM runtime 不回退

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 4: Stage 4 exit gate**

Only proceed when:

- `TTABIPlan` 已拥有 compile-time / common-runtime / per-work 三层 ABI
- `TTTransportPlan` / `TTHardwareModelStub` 已接入合法性检查
- copy / GEMM spec/runtime 已经由 `MaterializeTTExecutableSpec` 接管

## Task 5: Stage 5 - Phase C2 Compatibility Deletion And Flash-Attn Correctness

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`

- [ ] **Step 1: Delete compatibility writers and fallback readers by gate**

Delete only after corresponding TT stable fields exist:

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.common_runtime_args`
- `blackhole.accessors`
- `blackhole.cb_configs`
- `blackhole.semaphore_plan`
- `blackhole.core_plan`

- [ ] **Step 2: Re-run flash-attn pipeline gate on the new mainline**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- `flash-attn` compile-path 不再依赖 late target-specific semantic guessing
- `blackhole.acc` 不再同时承载 algorithm state 和 TT scratch 两类真语义

- [ ] **Step 3: Run direct runtime correctness gate in TT-Sim**

Run:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
```

Expected:

- `mha` direct runtime correctness pass
- `gqa` direct runtime correctness pass
- 不再出现当前 `blackhole.acc` 混合语义导致的 mismatch

- [ ] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 5: Stage 5 exit gate**

Only proceed when:

- `flash-attn` MHA/GQA direct runtime correctness 通过
- 基线 copy / GEMM / export / pipeline 全绿
- compatibility writer / reader / fallback 已按 deletion gates 收掉

## Task 6: Stage 6 - Family Expansion Under The New Mainline

**Files:**
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py`
- Create or modify consumer-specific tests alongside selected workloads
- Modify: `tasks/progress.md`

- [ ] **Step 1: Add the first non-attention family to the new path**

Recommended order:

1. `topk`
2. paged decode
3. `fusedmoe`
4. chunk recurrence

- [ ] **Step 2: Add compile gates before runtime gates**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py -q
```

Expected:

- 每个 family 先在 semantic / spatial / TT 层有稳定结构 gate
- 只有进入正式 direct runtime 支持面后，才加 TT-Sim runtime gate

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 4: Stage 6 exit gate**

Only proceed when:

- 新增 family 不引入 case-by-case matcher
- 每个 family 都先通过 compile gate，再决定是否进入 runtime gate
