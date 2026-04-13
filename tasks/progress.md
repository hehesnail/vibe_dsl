# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 根因诊断统一维护在 `tasks/dev_design/task0_ir_layering_root_cause.md`
> companion/pass 细节统一维护在 `tasks/dev_design/task1_spatial_plan_companion.md`
> `Task 2` 的 `TTProgram companion` cutover 统一维护在
> `tasks/dev_design/task2_ttprogram_companion_cutover.md`
> `Task 3` 的 runtime gate 与支持面边界统一维护在
> `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

## 当前阶段

- **日期**: `2026-04-13`
- **总阶段**: Stage 4
- **目标主线**: `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
- **当前代码基线**: `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
- **阶段状态**:
  - `Task 1` 已落地到当前 active Blackhole compile path
  - 当前执行基线仍由旧 `Phase C` consumer 链承接
  - 当前工作从 architecture cut-in 转入 `Task 2` cutover

## 当前任务链

- `Task 0`: 文档与 owner 链收口
  - 状态：已完成
- `Task 1`: `SpatialPlan companion` cut-in
  - 状态：已完成
  - 目标：
    `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
- `Task 2`: `TTProgram companion` cutover
  - 状态：未开始
  - 目标：
    `PlanTTBlocks -> PlanTTTransport -> PlanTTSync -> PlanTTABI ->
    PlanTTExecution -> MaterializeBlackholeExecutable`
- `Task 3`: 旧 recovery 链退场与 workload 回归
  - 状态：未开始
  - 目标：
    退场旧 recovery 主链，并让
    `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
    转到新主链

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 已恢复
- `TTProgram` translator / validator / materializer
  已进入当前正式主链；runtime/codegen 已切到 `TTProgram` direct reader
- `per_work_arg_specs` 已收成 kernel-local `TTKernel / ExecutableSpec` truth；
  synthetic segment codegen、direct runtime arg materialization
  与 direct runtime launch 已统一消费同一套 `value_kind` contract
- admitted support surface 仍然偏窄：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- Blackhole runtime / direct-runtime 正式 correctness baseline
  统一使用 `bf16` 输入；
  `fp16` 不再作为当前 TT-Sim 上的正式 runtime 基线

## 当前 blocker

- 第一 blocker 已经不是“再补一个 runtime protocol”，
  而是把 target owner 链从旧 recovery 主链切到 `TTProgram companion` 主链
- `SpatialPlan companion` 已经 cut-in，
  但当前 active consumer 仍然是
  `semantic_manifest / SemanticProgram / 旧 SpatialProgram /
  matcher-driven LowerBlackholeOps`
  这条旧路线
- 详细根因、旧链问题域和切入层次判断，
  统一见 `tasks/dev_design/task0_ir_layering_root_cause.md`
- `flash-attn` 的 direct-runtime correctness payoff、
  wider support surface 与 target-side gate，
  统一见 `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

## 当前未完成项

- 完成 `Task 2`
  - `PlanTTBlocks`
  - `PlanTTTransport`
  - `PlanTTSync`
  - `PlanTTABI`
  - `PlanTTExecution`
  - `MaterializeBlackholeExecutable`
- 完成 `Task 3`
  - 退场旧 recovery 主链
  - 让 `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
    在新主链下重新承接
  - 兑现更宽 copy/dataflow/sync 支持面
- 在 cutover 完成前，保持 copy / GEMM / export 当前正式支持面不回退

## 当前边界

- oversubscribed direct runtime 目前不是通用同步执行模型；
  一旦 executable 带显式 `TTSemaphorePlan`、
  `semaphore_bindings` 或 `remote_core_descriptors`，
  仍应 fail-fast
- `flash-attn` direct runtime 当前不是 admitted support surface；
  对缺失显式 per-work contract 或 typed fragment materialization/merge protocol
  的 kernel，继续通过 `direct_runtime_unsupported_reasons` skip / fail-fast
- TT-Sim `fp16` 路径当前仍可能命中 simulator fatal taxonomy；
  该路径按 simulator capability boundary 处理，不作为当前 correctness gate
- `TT_METAL_WATCHER=10` 调试 multicore direct path 时，
  watcher 线程仍可能自己抛错或卡在 dump；
  正式 baseline 应在标准 watcher-off 环境下判断

## 下一步

1. 实现 `TTProgram companion` cutover：
   `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI /
   PlanTTExecution / MaterializeBlackholeExecutable`
2. 让 `Task / Channel` 在新主链里只保留 derived-view 角色，
   不再继续扩张旧 `SpatialProgram` truth
3. 在同一轮 cutover 中退场旧 recovery 主链，
   然后把 workload family 重新接到新主链

## 最新验证摘要

- `tilelang_repo/build` fresh rebuild 通过
- `Task 1` 结构回归：
  - `test_blackhole_spatial_ir.py -k 'task1 or spatial_passes_are_registered'`
    -> `4 passed`
  - `test_blackhole_semantic_ir.py -k hard_freeze_invalidates_companion_programs_after_unsafe_mutation`
    -> `1 passed`
- 旧稳定面 smoke：
  - `test_blackhole_spatial_ir.py -k 'spatial_pass_pipeline_materializes or copy_spatial_program_uses_single_transfer_fast_path or gemm_spatial_program_uses_reader_compute_writer_fast_path'`
    -> `3 passed`
  - `test_blackhole_semantic_ir.py -k 'copy_semantic_program_lifts_minimal_domain_and_map_update or gemm_semantic_program_lifts_fragment_state_and_map_update or hard_freeze_invalidates_companion_programs_after_unsafe_mutation'`
    -> `3 passed`
- `tilelang.lower(..., target="blackhole", enable_device_compile=False)` smoke：
  staged copy / GEMM 均能经过 `blackhole_codegen` 主链并同时产出
  `tl.spatial_structure_facts + tl.spatial_plan + tl.spatial_program`
- 当前 dirty 主线上，完整
  `test_blackhole_spatial_ir.py + test_blackhole_semantic_ir.py`
  仍有既有失败，集中在 flash/topk/fusedmoe/chunk recurrence 旧链分析；
  本轮未把这批已有失败当成 `Task 1` 关闭条件
