# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 根因诊断统一维护在 `tasks/dev_design/task0_ir_layering_root_cause.md`
> companion/pass 细节统一维护在 `tasks/dev_design/task1_spatial_plan_companion.md`
> `Task 2` 的 `TTProgram companion` cutover 统一维护在
> `tasks/dev_design/task2_ttprogram_companion_cutover.md`
> `Task 3` 的 runtime gate 与支持面边界统一维护在
> `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

## 当前阶段

- **日期**: `2026-04-14`
- **总阶段**: Stage 4
- **目标主线**: `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
- **当前代码基线**:
  `Normalized Tile TIR -> SpatialPlan companion + Blackhole analysis facts ->
  Spatial Program IR -> TTProgram companion -> ExecutableSpec`
- **当前目标基线**:
  `Normalized Tile TIR -> SpatialPlan companion + Blackhole analysis facts ->
  Spatial Program IR -> TTProgram companion -> ExecutableSpec`
  （剩余工作集中在 runtime gate / workload payoff）
- **阶段状态**:
  - `Task 1` 已落地到当前 active Blackhole compile path
  - `Task 2` 已落地到当前 active Blackhole compile path
  - 当前执行基线已固定为 canonical `TTProgram` bundle +
    `MaterializeBlackholeExecutable` writer boundary
  - 当前工作从 `Task 2` cutover 转入 `Task 3`
    runtime gate / workload payoff
  - `Task 3` 已进入旧链删除批次：
    projection bridge、fragment-layout side-channel 已删除，
    `BuildTTProgram` 也已不再物化/读取中间 `tl.tt_*` seed attrs
  - canonical `LowerToBlackholeTTProgram` 已不再显式串
    任何 legacy pass 名字；剩余 planning 兼容逻辑已内收到
    `BuildTTProgram` 内部的
    `PlanTTKernelABI -> PlanTTCBAlloc -> PlanTTCoreGroups`
    helper chain
  - 旧 pass 的 public/test surface 已继续收口：
    `tilelang.transform` Python wrapper、
    FFI global registration、
    `lower_blackhole_ops_through_phase_b` /
    typed-seed 测试 helper
    与纯旧链回归 case 已删除
  - `Task 3A` 的 persistent/public 删除批次已完成：
    `tl.semantic_*` attrs、semantic pass public surface、
    Python wrapper、FFI registration、死代码与纯旧链测试已删除
  - active path 当前直接消费
    `Normalized Tile TIR + SpatialPlan companion +
    blackhole.work_decomposition / blackhole.compute_regions /
    blackhole.pipeline_stages`
    这组当前 owner truth；
    不再保留独立 semantic companion / semantic witness 通道
  - `AnalyzeBlackholeComputeRegions` 的 pass / 文件名切换已完成；
    旧 compute-region 命名与对应 helper 文件已删除
  - `BlackholeDeviceResourceCanonicalization` 当前同时 canonicalize
    TIR body 与
    `blackhole.lowering_requirements / blackhole.compute_regions /
    tl.spatial_program / tl.tt_program`
    中的 buffer `scope`；
    不再允许 body truth 与 companion contract 漂移

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
  兼容 shell 已删除，不再保留公开入口
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
  runtime gate / wider support surface
- `Task 3A` 的 persistent semantic layer 与 `tl.semantic_*`
  主协议已从 active path 退场；
  当前主要压力点回到 `Task 3B / 3C`
  的 runtime gate 与 wider workload payoff
- 旧链清理当前已完成外层桥接删除：
  final Phase C 输出不再暴露
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
  tl.tt_core_groups / tl.tt_program_payload`
  这组中间 seed attrs
- `SplitBlackholeKernel` 已退回成纯 IR annotation pass；
  不再写 `blackhole.segment_plan`
- `PlanTTKernelABI` 已不再综合
  `blackhole.segment_plan / runtime_args / gemm_contract /
  compute_contract`
  这组 compatibility attr；
  target truth 只在 pass 内部聚合为 `TTKernel / TTABIPlan / TTProgram payload`
- 过渡 probe / test surface 已继续删除；
  不再保留 `LowerSpatialProgramToTTTargetProbe`
  这类非正式入口
- 当前剩余的结构性 blocker 是
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
  这组内部 helper chain
  仍在 `BuildTTProgram` 内承担 planning owner 责任；
  legacy pass 名字、公开入口和测试入口已删除，
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

1. **P0: `Task 3B` runtime gate + `flash-attn` payoff**
  - 在新主链上兑现 admitted subset correctness
  - 继续把剩余 internal helper bridge owner 从 canonical pipeline 中清掉
2. **P1: `Task 3C` wider family / support surface**
  - `topk -> fusedmoe -> paged decode -> chunk recurrence`
  - wider copy/dataflow
   - wider sync 最后进入 admitted surface

## 当前未完成项

- 完成 `Task 3`
  - 让 `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
    在新主链下重新承接
  - 兑现更宽 copy/dataflow/sync 支持面
- 用真实 `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI /
  PlanTTExecution` 取代当前 `BuildTTProgram` 内部
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
  helper bridge
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

1. 做 `Task 3B`
  - runtime gate
  - `flash-attn`
  - `PlanTT*` owner pass 替换 seed bridge
2. 再做 `Task 3C`
  - wider family / support surface

## 最新验证摘要

- `2026-04-14` old-chain cleanup verification
- 旧 semantic public surface / attr grep：
  - `rg -n "tl\.semantic_|AnalyzeSemanticStructure|ProjectSemanticSeeds|ProjectSemanticManifest|AugmentSemanticManifest|CollectSemanticManifestSeeds" tilelang_repo/src tilelang_repo/tilelang -g '!tilelang_repo/build/**'`
    -> 无匹配
  - `rg -n "semantic_program|semantic_structure_decoder|semantic_witness|semantic_vocab|semantic_refinement_rules|semantic_state_effect_graph" tilelang_repo/src tilelang_repo/tilelang -g '!tilelang_repo/build/**'`
    -> 无匹配
- 构建：
  - `cmake -S tilelang_repo -B tilelang_repo/build`
  - `cmake --build tilelang_repo/build -j32`
    -> `Built target tilelang`
- 定向 pytest：
  - `python -m pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q`
    -> `41 passed`
  - `python -m pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q -k 'omits_legacy_semantic_attrs or attaches_multi_phase_spatial_program'`
    -> `2 passed, 60 deselected`
- 旧链收口确认：
  - semantic pass / wrapper / dead code 已删除：
    `ProjectSemanticSeeds / CollectSemanticManifestSeeds /
    ProjectSemanticManifest / AugmentSemanticManifest /
    AnalyzeSemanticStructure`
  - `LowerSpatialProgramToTTTargetProbe`
    与 `tl.semantic_*` 相关测试/helper 已删除
  - active path 与 `src/` / `tilelang/` 范围内不再存在
    `tl.semantic_*` 主协议或对应 decoder / witness / vocab /
    refinement / state-effect graph 实现
- `Task 3` 当前残留：
  - `BuildTTProgram` 内部仍直接调用
    `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
    helper chain；当前已把 legacy pass 名字、compatibility attr、
    public/test/FFI surface 清掉，但真正的 `PlanTT*` owner 替换
    仍属于后续 `Task 3`
  - `flash-attn` / wider workload payoff 仍未关闭；
    grouped-row fragment contract 与更宽 sync/dataflow 支持面
    继续归属后续 `Task 3A/3B`
- dirty 主线上完整
  `test_blackhole_spatial_ir.py + test_blackhole_semantic_ir.py`
  仍有既有失败，集中在 flash/topk/fusedmoe/chunk recurrence
  相关旧链分析/WIP；本轮未把这批无关失败当成旧链删除回退理由
