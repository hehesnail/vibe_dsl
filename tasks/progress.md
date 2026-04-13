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
  - `Task 2` 已落地到当前 active Blackhole compile path
  - 当前执行基线已固定为 canonical `TTProgram` bundle +
    `MaterializeBlackholeExecutable` writer boundary
  - 当前工作从 `Task 2` cutover 转入 `Task 3` runtime/workload payoff
  - `Task 3` 已进入旧链删除批次：
    projection bridge、fragment-layout side-channel 已删除，
    `BuildTTProgram` 末端也开始剥离中间 `tl.tt_*` seed attrs
  - canonical `LowerToBlackholeTTProgram` 已不再显式串
    `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores`
    这条旧 pass 链；剩余 planning 兼容逻辑已内收到 `BuildTTProgram`

## 当前任务链

- `Task 0`: 文档与 owner 链收口
  - 状态：已完成
- `Task 1`: `SpatialPlan companion` cut-in
  - 状态：已完成
  - 目标：
    `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
- `Task 2`: `TTProgram companion` cutover
  - 状态：已完成
  - 目标：
    `PlanTTBlocks -> PlanTTTransport -> PlanTTSync -> PlanTTABI ->
    PlanTTExecution -> MaterializeBlackholeExecutable`
- `Task 3`: 旧 recovery 链退场与 workload 回归
  - 状态：进行中
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
- canonical Python/engine bundle 已固定为：
  - transform alias:
    `BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable`
  - engine bundle:
    `LowerToBlackholePhaseB -> LowerToBlackholeTTProgram -> LowerToBlackholeExecutable`
- 旧 `LowerSpatialProgramToTTTarget / ValidateTTTargetProgram /
  MaterializeTTExecutableSpec`
  仍保留为 compatibility shell，不再承担当前入口语义
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

- 第一 blocker 已经从 `Task 2` owner cutover 切换到 `Task 3`
  runtime/workload payoff
- 当前主要压力点不再是
  `TTProgram companion` 是否进入 active path，
  而是 `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
  在新主链上的 correctness payoff 与 admitted support surface
- 旧链清理当前已完成外层桥接删除：
  final Phase C 输出不再暴露
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
  tl.tt_core_groups / tl.tt_program_payload`
  这组中间 seed attrs
- 当前剩余的结构性 blocker 是
  `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores`
  不再作为 active path 上的显式阶段，
  但仍在 `BuildTTProgram` 内部承担 seed bridge owner 责任，
  尚未被真实 `PlanTTBlocks / PlanTTTransport / PlanTTSync /
  PlanTTABI / PlanTTExecution` 取代
- 详细根因、旧链问题域和切入层次判断，
  统一见 `tasks/dev_design/task0_ir_layering_root_cause.md`
- `flash-attn` 的 direct-runtime correctness payoff、
  wider support surface 与 target-side gate，
  统一见 `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
- 当前不把下面这些当 blocker：
  - non-Blackhole backend 的统一收口
  - repo-wide frontend 统一
  - public Python `transform` API 改名

## 当前优先级

1. **P0: `Task 3A` runtime gate + `flash-attn` payoff**
  - 在新主链上兑现 admitted subset correctness
  - 继续把旧 seed bridge owner 从 canonical pipeline 中清掉
2. **P1: `Task 3B` wider family / support surface**
  - `topk -> fusedmoe -> paged decode -> chunk recurrence`
  - wider copy/dataflow
   - wider sync 最后进入 admitted surface

## 当前未完成项

- 完成 `Task 3`
  - 退场旧 recovery 主链
  - 让 `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
    在新主链下重新承接
  - 兑现更宽 copy/dataflow/sync 支持面
- 用真实 `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI /
  PlanTTExecution` 取代当前内部 seed bridge
- 在 `Task 3` 收口前，保持 copy / GEMM / export 当前正式支持面不回退

## 当前边界

- oversubscribed direct runtime 目前不是通用同步执行模型；
  一旦 executable 带显式 `TTSemaphorePlan`、
  `semaphore_bindings` 或 `remote_core_descriptors`，
  仍应 fail-fast
- `flash-attn` direct runtime 当前不是 admitted support surface；
  对缺失显式 per-work contract 或 typed fragment materialization/merge protocol
  的 kernel，继续通过 `direct_runtime_unsupported_reasons` skip / fail-fast
- grouped-row / fragment-layout contract 仍是 `flash-attn` 当前主要缺口；
  相关 probe 已不再把未 admitted 的前提固化成稳定绿测
- TT-Sim `fp16` 路径当前仍可能命中 simulator fatal taxonomy；
  该路径按 simulator capability boundary 处理，不作为当前 correctness gate
- `TT_METAL_WATCHER=10` 调试 multicore direct path 时，
  watcher 线程仍可能自己抛错或卡在 dump；
  正式 baseline 应在标准 watcher-off 环境下判断

## 下一步

1. 做 `Task 3A`
  - runtime gate
  - `flash-attn`
  - `PlanTT*` owner pass 替换 seed bridge
2. 再做 `Task 3B`
  - wider family / support surface

## 最新验证摘要

- `tilelang_repo/build` fresh rebuild 通过
- 旧链删除本轮验证：
  - `test_blackhole_tt_target_probe.py`
    -> `20 passed`
  - `test_blackhole_copy_pipeline.py -k 'blackhole_codegen_only or build_reads_tt_program_without_legacy_projection_attrs or buffer_materialization_specs_are_exposed'`
    -> `3 passed`
  - `test_blackhole_gemm.py -k 'gemm_contract or executable_spec or blackhole_codegen'`
    -> `1 passed`
- `Task 2` canonical bundle / compatibility shell 回归：
  - `test_blackhole_tt_target_probe.py -k tt_target_probe_pass_is_registered`
    -> `1 passed`
  - `test_blackhole_copy_pipeline.py -k 'blackhole_codegen_only or build_reads_tt_program_without_legacy_projection_attrs or buffer_materialization_specs_are_exposed'`
    -> `3 passed`
  - `test_blackhole_gemm.py -k 'split_kernel_gemm_segment_plan or gemm_contract or executable_spec or blackhole_codegen'`
    -> `2 passed`
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
- `Task 3` 当前残留：
  - `flash-attn` GQA executable-spec / codegen probe 仍会命中
    grouped `reduce_row` 需要 `grouped_rows` fragment layout contract，
    当前 `acc_s` 仍拿到 `thread_distributed`；
    该问题归属 `Task 3` flash-attn payoff，不作为 `Task 2` 关闭条件
- 当前 dirty 主线上，完整
  `test_blackhole_spatial_ir.py + test_blackhole_semantic_ir.py`
  仍有既有失败，集中在 flash/topk/fusedmoe/chunk recurrence 旧链分析；
  本轮未把这批已有失败当成 `Task 1` 关闭条件
