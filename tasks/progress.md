# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 根因诊断统一维护在 `tasks/dev_design/ir_layering_root_cause_and_direction.md`
> companion/pass 细节统一维护在 `tasks/dev_design/spatial_dataflow_program_model.md`
> `Phase C` 当前 TT baseline、runtime gate 与支持面边界统一维护在
> `tasks/dev_design/stage4_phase_c_tt_target_ir.md`

## 当前阶段

- **日期**: `2026-04-13`
- **总阶段**: Stage 4
- **目标主线**: `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
- **当前代码基线**: `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
- **阶段状态**:
  - 当前执行基线仍是旧 `Phase C` 主链
  - companion-based 总设计与任务链已锁定
  - 当前工作转入 architecture cutover

## 当前任务链

- `Task 0`: 文档与 owner 链收口
  - 状态：已完成
- `Task 1`: `SpatialPlan companion` cut-in
  - 状态：未开始
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
  而是把 active owner 链从旧 recovery 主链切到 companion 主链
- 当前需要切掉的不是单个 matcher，
  而是整条
  `semantic_manifest / SemanticProgram / 旧 SpatialProgram /
  matcher-driven LowerBlackholeOps`
  主路线
- 详细根因、旧链问题域和切入层次判断，
  统一见 `tasks/dev_design/ir_layering_root_cause_and_direction.md`
- `flash-attn` 的 direct-runtime correctness payoff、
  wider support surface 与 target-side gate，
  统一见 `tasks/dev_design/stage4_phase_c_tt_target_ir.md`

## 当前未完成项

- 完成 `Task 1`
  - `AnalyzeSpatialStructureFacts`
  - `BuildSpatialPlanCompanion`
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

1. 在 `Simplify` 后实现 `AnalyzeSpatialStructureFacts`
2. 实现 `BuildSpatialPlanCompanion`，把 `Task / Channel` 退回 derived view
3. 实现 `TTProgram companion` cutover：
   `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI /
   PlanTTExecution / MaterializeBlackholeExecutable`
4. 在同一轮 cutover 中退场旧 recovery 主链，
   然后把 workload family 重新接到新主链

## 最新验证摘要

- `tilelang_repo/build` fresh rebuild 通过
- 所有 runtime 检查均在标准 TT-Sim 环境入口下完成
- `flash-attn` regression：
  `test_blackhole_flash_attention_pipeline.py` -> `62 passed`
  `test_blackhole_flash_attention_runtime.py` -> `9 passed, 5 skipped`
- copy regression：
  `test_blackhole_copy_pipeline.py` -> `43 passed, 10 skipped, 1 xfailed`
  `test_blackhole_copy_runtime.py` -> `12 passed`
- copy baseline：
  `test_blackhole_copy_pipeline.py` -> `52 passed, 1 skipped, 1 xfailed`
  `test_blackhole_copy_runtime.py` -> `12 passed`
