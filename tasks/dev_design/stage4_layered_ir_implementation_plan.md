# Layered IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `final_blackhole_backend_redesign.md` 中已经定稿的 layered IR 架构拆成可执行实施阶段，在不回退 copy / GEMM / current direct host path 的前提下，完成 `Stateful Semantic IR -> Spatial Program IR -> TT Target IR -> MaterializeTTExecutableSpec` 的主链迁移。

**Architecture:** 先做 `P0` 护栏，把 single-source cutover、semantic lift canonicalization、device-program registry、early semantic capture、compatibility deletion gate 收紧成机器可执行的边界；然后按 `Phase A -> Phase B -> Phase C` 顺序逐层新增 companion IR，并要求每一层都先通过自己的 validator 和结构测试，再允许进入下一层。consumer 推进顺序以 `flash-attn` 为第一优先级，但每一层都必须至少有一个 non-attention gate，防止实现重新退化成 attention-only。

**Tech Stack:** TileLang DSL, TVM PrimFunc / TIR / Object system, TileLang transform passes, Blackhole codegen/runtime, pytest, CMake, TT-Sim, TT-Metal direct runtime

---

## Scope

本计划覆盖：

- `P0` 实施前置收口项
- `Phase A1 / A2`: `SemanticProgram`
- `Phase B`: `SpatialProgram`
- `Phase C1 / C2`: `TTProgram` + `MaterializeTTExecutableSpec`
- `flash-attn` correctness payoff
- 新分层下的 workload family 扩面顺序

本计划不覆盖：

- 第二份总体设计文档
- 第二条正式执行路径
- 重新引入 legacy external runner
- 任何基于单个 kernel 名字或单个样例形态的 matcher 协议

## Stage Summary

| Stage | 目标 | 主要产出 | 通过标准 |
|------|------|----------|----------|
| 0 | 锁定迁移护栏 | `SemanticSeed` 通道、`CollectDevicePrograms`、deletion gates、A1 hard freeze | 新结构测试通过，copy / GEMM / export 基线零回归 |
| 1 | Phase A1 最小语义层 | `SemanticProgram` 最小 object 集 + `AnalyzeSemanticStructure` + `Lift/Validate` | semantic 结构测试通过；copy / GEMM / flash-attn compile-path 零回归 |
| 2 | Phase A2 语义扩面 | `SelectLaw` / `RecurrenceLaw` / `SemanticSupplement` / wider `AccessMap` | `flash-attn` carry/update 语义显式化；至少一个 non-attention semantic gate |
| 3 | Phase B 空间层 | `SpatialProgram`、`ProgramPhase`、simple-workload fast-path | 至少一个 non-trivial multi-phase spatial case；不退化回 `Task:TTKernel = 1:1` |
| 4 | Phase C1 目标层与物化切换 | `TTProgram`、`TTHardwareModel` stub、`MaterializeTTExecutableSpec` 接管 copy/GEMM | copy / GEMM spec/runtime 只从 TT 层物化，现有 runtime 回归通过 |
| 5 | Phase C2 cutover + correctness payoff | 删除 compatibility writer / reader / fallback，拿下 `flash-attn` direct runtime correctness | `flash-attn` MHA/GQA TT-Sim correctness 通过，基线不回退 |
| 6 | workload family 扩面 | `topk` / paged decode / fusedmoe / chunk recurrence 接入新主链 | 每个 family 至少先有 compile gate，进入正式执行面后再加 runtime gate |

## File Map

### New transform / companion IR files

- Create: `tilelang_repo/src/transform/common/semantic_program.h`
  - `SemanticProgram / Domain / State / Update / AccessMap / UpdateLaw / TIRAnchor / TIRValueBinding / SemanticSeed / DeviceProgramInfo`
- Create: `tilelang_repo/src/transform/common/semantic_program.cc`
  - object registration, printer, debug dump helpers
- Create: `tilelang_repo/src/transform/collect_device_programs.cc`
  - pre-`SplitHostDevice` registry，稳定写入 `IRModule.global_infos["tl.device_programs"]`
- Create: `tilelang_repo/src/transform/project_semantic_seeds.cc`
  - 从 `LowerTileOp` 后的稳定信号投影 `tl.semantic_seeds`
- Create: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
  - 汇总现有 `AnalyzeBlackhole*` 结果和 early seed，构建 semantic recovery 输入
- Create: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
  - 生成 `PrimFunc.attrs["tl.semantic_program"]`
- Create: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
  - semantic legality / consistency checks
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
  - semantic -> spatial projection，含 simple-workload fast-path
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
  - spatial legality / phase boundary validation
- Create: `tilelang_repo/src/transform/lower_spatial_program_to_tt_target.cc`
  - spatial -> TT mapping
- Create: `tilelang_repo/src/transform/validate_tt_target_program.cc`
  - TT legality / hardware-model validation
- Create: `tilelang_repo/src/transform/materialize_tt_executable_spec.cc`
  - `TTProgram` 到 `ExecutableSpec` / `blackhole.*` compatibility attrs 的唯一 writer
- Create: `tilelang_repo/src/transform/common/tt_target_program.h`
  - `TTProgram / TTKernel / TTABIPlan / TTTransportPlan / TTCBPlan / TTSemaphorePlan / TTDstLayoutPlan / TTExecutionPlan / TTHardwareModelStub`
- Create: `tilelang_repo/src/transform/common/tt_target_program.cc`
  - target companion object registration 与 debug printer

### New tests

- Create: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
  - Stage 0 / 1 / 2 的 semantic 结构和 validator tests
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
  - Stage 3 的 `ProgramPhase / Task / Channel / Layout / WorkPartition` tests
- Create: `tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py`
  - Stage 4 / 5 的 `TTProgram / TTHardwareModel / MaterializeTTExecutableSpec` tests

### Existing files to modify

- Modify: `tilelang_repo/CMakeLists.txt`
  - 接入新增 transform / companion object 源文件
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
  - 暴露新 pass
- Modify: `tilelang_repo/tilelang/engine/phase.py`
  - 插入 `CollectDevicePrograms -> ProjectSemanticSeeds -> AnalyzeSemanticStructure -> LiftToStatefulSemanticIR -> ValidateStatefulSemanticIR -> LowerToSpatialProgram -> ValidateSpatialProgram -> LowerSpatialProgramToTTTarget -> ValidateTTTargetProgram -> MaterializeTTExecutableSpec`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
  - 调整 Blackhole mainline 的 device-side 入口与 compatibility path
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
  - 变成 semantic recovery helper，而不再直接充当最终协议 writer
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
  - 输出 `Update / AccessMap / fragment region` 相关事实
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
  - 输出 semantic ordered-update / carry 边界与 spatial phase hint
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  - 从“混合 recovery + target lowering”收缩为消费 semantic/spatial/TT truth 的 lowering helper
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
  - 从直接 writer 改为 `TTCBPlan` 构造 / 校验 helper
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
  - 从直接 writer 改为 `TTExecutionPlan` 构造 / 校验 helper
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - 只消费 `MaterializeTTExecutableSpec` 的稳定物化结果
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
  - 删除 compatibility aggregation / fallback，保留 direct host materialization
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
  - 不再做 target-side semantic guessing，只消费 TT target truth
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  - 接入 Stage 0-4 的 spec / compatibility deletion gate 回归
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  - 接入 Stage 0-4 的 multi-kernel / ABI / TTProgram gate 回归
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  - 接入 Stage 1-5 的 semantic / spatial / TT compile-path 回归
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  - Stage 5 correctness gate
- Modify: `tasks/progress.md`
  - 同步阶段状态和下一步

## Global Acceptance Criteria

以下命令构成所有阶段共享的零回归基线。每进入下一阶段前都要重跑：

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

进入 direct runtime / TT-Sim gate 时，必须使用固定环境入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
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

- [ ] **Step 1: Establish the migration guardrails**

Deliver:

- `IRModule.global_infos["tl.device_programs"]` 作为唯一 module-scope device-program registry
- `PrimFunc.attrs["tl.semantic_seeds"]` 作为 pre-lift typed input 通道
- A1 `post-lift hard freeze` 规则
- unsafe TIR mutation 导致 `tl.semantic_program / tl.spatial_program / tl.tt_program` 整体失效的 contract
- compatibility deletion gates checklist

- [ ] **Step 2: Add Stage 0 structure tests**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'device_program_registry or semantic_seeds or hard_freeze' -q
```

Expected:

- `tl.device_programs` 能被查询
- `tl.semantic_seeds` 在 semantic lift 前可见
- lift 之后 unsafe mutator 会显式拒绝或触发整体失效

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the global acceptance commands.

- [ ] **Step 4: Stage 0 exit gate**

Only proceed when:

- `CollectDevicePrograms` 已在 `SplitHostDevice` 前建立 registry
- `ProgramPhase` 的稳定宿主不再依赖单 `PrimFunc.attrs`
- `SemanticSeed` 输入通道和 deletion gates 已有结构测试
- shared zero-regression baseline 全绿

## Task 1: Stage 1 - Phase A1 Minimal Semantic IR

**Files:**
- Create: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- Create: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- Create: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Add the minimal semantic object set**

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

- [ ] **Step 2: Lift and validate copy / GEMM / flash-attn subset**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'copy or gemm or flash_attention' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- `tl.semantic_program` 在目标 `PrimFunc` 上可见
- copy / GEMM 不丢已有 compile-path 能力
- `flash-attn` subset 至少能稳定 lift 出 `Domain / State / UpdateLaw.kind`

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the global acceptance commands.

- [ ] **Step 4: Stage 1 exit gate**

Only proceed when:

- A1 minimal object set 已稳定
- `ValidateStatefulSemanticIR` 能拦住结构不一致输入
- copy / GEMM / current `flash-attn` compile-path 零回归

## Task 2: Stage 2 - Phase A2 Semantic Expansion

**Files:**
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- Modify: `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Expand semantic schema beyond A1**

Required additions:

- fuller `AccessMap.traits`
- `SelectLaw`
- `RecurrenceLaw`
- typed `SemanticSupplement`
- clearer `AtomicEffect -> Update` recovery boundary

- [ ] **Step 2: Make `flash-attn` carry / stats state explicit in semantic layer**

This stage must separate:

- algorithmic carry / reduction-update state
- TT compute scratch / matmul destination state

This is the first stage allowed to directly attack the root cause behind the current `blackhole.acc` correctness mismatch.

- [ ] **Step 3: Add one non-attention semantic gate**

Recommended first gate: `topk`

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k 'topk or selection or recurrence' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- 至少一个 non-attention case 稳定 lift 出正确的 `UpdateLaw.kind`
- `flash-attn` 的 stats/carry/update 不再依赖 raw TIR 晚期猜测

- [ ] **Step 4: Re-run shared zero-regression baseline**

Run the global acceptance commands.

- [ ] **Step 5: Stage 2 exit gate**

Only proceed when:

- `flash-attn` semantic root cause 已在 semantic 层有显式对象表达
- 至少一个 non-attention semantic gate 通过
- shared zero-regression baseline 全绿

## Task 3: Stage 3 - Phase B Spatial Program IR

**Files:**
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Introduce `SpatialProgram` and `ProgramPhase`**

Required objects:

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

Rules:

- module-scope `ProgramPhase` truth lives in `tl.device_programs`
- member-local truth lives in `PrimFunc.attrs["tl.spatial_program"]`
- simple workload gets canonical fast-path

- [ ] **Step 2: Add simple-workload fast-path for copy / GEMM**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'copy or gemm or fast_path' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:

- copy / GEMM 不需要进入重 candidate synthesis
- trivial workload 仍能快速构造 canonical `SpatialProgram`

- [ ] **Step 3: Add one non-trivial multi-phase spatial gate**

Recommended first gate: `flash-attn`

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'flash_attention or multi_phase' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- 至少一个 case 证明下游不会退化回 `Task:TTKernel = 1:1`
- `ProgramPhase` / `Channel` / phase-boundary materialization 在结构测试里可见

- [ ] **Step 4: Re-run shared zero-regression baseline**

Run the global acceptance commands.

- [ ] **Step 5: Stage 3 exit gate**

Only proceed when:

- `SpatialProgram` 能消费冻结后的 semantic truth
- simple-workload fast-path 稳定
- 至少一个 non-trivial multi-phase spatial gate 通过

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

Run the global acceptance commands.

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

Run the global acceptance commands.

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

Run the global acceptance commands.

- [ ] **Step 4: Stage 6 exit gate**

Only proceed when:

- 新增 family 不引入 case-by-case matcher
- 每个 family 都先通过 compile gate，再决定是否进入 runtime gate

## Execution Order

严格执行顺序：

1. Stage 0
2. Stage 1
3. Stage 2
4. Stage 3
5. Stage 4
6. Stage 5
7. Stage 6

不允许跳过：

- Stage 0 的迁移护栏
- Stage 3 的 non-trivial spatial gate
- Stage 4 的 `MaterializeTTExecutableSpec` cutover
- Stage 5 的 TT-Sim correctness gate

## Review Checklist

执行每个阶段前，都要先问四件事：

1. 这一步是在新增单一真源，还是在偷偷扩 compatibility attrs？
2. 这一步所需信息是否已经在 IR / companion IR 里显式存在？如果没有，先扩 schema，不要让后段猜。
3. 这一步的 gate 是 compile-structure、spec-materialization，还是 TT-Sim runtime？不要混淆层次。
4. 这一步是否至少保留一个 non-attention 视角，防止实现退化成 attention-only？
