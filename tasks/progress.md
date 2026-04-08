# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-08`
- **总阶段**: Stage 4
- **当前主线**: `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- copy / GEMM 当前支持面不回退
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- direct runtime 当前正式支持面：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 阶段状态

- **Stage 0**: 已完成
- **Phase A**: 已完成
- **Phase B**: 已完成
  - 已完成：execution-bearing `SpatialProgram` contract、stronger validator、
    TT probe intake 与 shared zero-regression baseline
  - 已完成后续代码重构：
    `AnalyzeSpatialDomainPlan -> AnalyzeSpatialExecutionPlan -> MaterializeSpatialProgram`
    已拆出，`LowerToSpatialProgram` 退化为兼容 wrapper
  - 已完成审计收口：
    shared helper 去重、typed field 上提、index canonical linkage、
    validator 收窄、fast path contract 共构、capability model 前置发布
- **Phase C**: 正式 cutover 主链已接入
  - 已完成：read-only translator demand probe、hardware intake、
    `TTProgram` core object set、`LowerSpatialProgramToTTTarget`、
    `ValidateTTTargetProgram`、`MaterializeTTExecutableSpec`
  - 当前保留：legacy target attrs reader 仍作为 projection consumer 存在，
    但 steady-state writer 已切到 `TTProgram -> MaterializeTTExecutableSpec`
  - 当前 bridge 实态：
    `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores`
    仍写 legacy bridge attrs，`LowerSpatialProgramToTTTarget`
    仍从这些 attrs 构造 `TTProgram`；
    当前 immediate gate 是 reader-side cutover，不是先删 producer-side bridge 输入

## `Phase B` 收尾结果

- `Spatial*` object/vocab/shared key 已从 semantic infra 拆出
- `AnalyzeSpatialDomainPlan -> AnalyzeSpatialExecutionPlan -> MaterializeSpatialProgram
  -> ValidateSpatialProgram` 已接入主线，`LowerToSpatialProgram`
  仅保留兼容 wrapper，`LowerBlackholeOps` 已硬要求 `tl.spatial_program`
- `tl.tt_hardware_model` / `tl.spatial_capability_model` 已发布为 module-scope global info
- `LowerToSpatialProgram` 已消费来自 SoC descriptor 的最小 capability snapshot
- `Channel.kind + payload_kind + delivery_kind` 与 `placement.affinity_kind`
  已收成当前 probe intake 所需的最小 contract
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且不会恢复 non-TT-specific spatial semantics
- `ValidateSpatialProgram` 已收正为 coherence / completeness / semantic alignment /
  ordering legality gate
- generic builder 已把 `Task` formation、flow shaping、domain realization、
  phase / ordering synthesis 收进稳定主链
- `SpatialDomainPlan` / `SpatialExecutionPlan` 已成为 `Phase B` 内部 typed 中间契约，
  不再把 domain/layout 与 task/channel/phase 收敛在单个 lowering 入口
- `phase_boundary_materialization` 已收窄为真实跨 phase state handoff，
  不再把“任意后续 phase 读取的 state”过度记为边界物化

结论：

- `SpatialProgram` 已达到 execution-bearing spatial program 的完成判定
- `Phase C` 现在可以把 `SpatialProgram` 当成单一上游真源继续 cutover

补充：

- `Phase B` 已完成一轮完成后设计审计；当前不重新打开 `Phase B`，
  而是带着 object-boundary 风险清单进入 `Phase C`
- `tasks/dev_design/phase_b_code_audit.md` 中点名的结构性问题已按当前
  `Phase C` 前置范围收口：monolith 已拆分、关键 payload truth 已提升为
  typed field、name 退回 display/debug、consumer 主链接走 typed/index truth
- 当前最值得在 `Phase C` 中继续验证的边界包括：
  - `Placement` 是否会被消费成真实 target mapping constraint
  - `SpatialCapabilityModel` 的 quantitative hardware fields 是否会进入 planning / mapping
  - `ResourceIntent` 是否能继续保持 small-closed kind discipline

## 当前 blocker

- legacy target attr reader / fallback 删除尚未完全收口；
  `rt_mod_blackhole` / `codegen_blackhole`
  仍消费 `MaterializeTTExecutableSpec` 反写的 projection
- 当前 reader-side gate 的直接范围是：
  - `rt_mod_blackhole` 仍直接解码
    `blackhole.segment_plan / runtime_args / common_runtime_args /
    accessors / cb_configs / semaphore_plan / core_plan`
  - `codegen_blackhole` 仍直接消费
    `blackhole.segment_plan / runtime_args / cb_configs / core_plan`
- `TTProgram` 已具备
  `TTABIPlan / TTCBPlan / TTCoreGroup / TTSemaphorePlan / TTExecutionPlan`
  这些 direct-read 所需的主要 typed object，
  但 runtime/codegen 侧仍未接入 direct reader
- `ValidateTTTargetProgram` 当前主要覆盖结构完整性与 linkage；
  reader-side cutover 之后还需要更强的 ABI / accessor / launch
  validator gate
- `flash-attn` 的 `blackhole.acc` correctness payoff 仍归属 `Phase C2`

## 独立已知问题

- `TT_METAL_WATCHER=10` 调试 multicore GEMM direct path 时，`tt_metal`
  的 watcher 线程可能在 `WatcherServer::Impl::poll_watcher_data()` 里抛
  `std::runtime_error` 并触发 `SIGABRT`，或在 `TT_METAL_WATCHER_TEST_MODE=1`
  下停在 `Dump #2`
- 当前结论是 watcher-side bug，不是 `BlackholeModule` direct runtime 主链回归；
  direct runtime baseline 需在 `TT_METAL_WATCHER` unset 的正式环境下判断

## 下一步

1. 在 `rt_mod_blackhole` / `codegen_blackhole`
   引入共享的 `TTProgram` direct reader / decoder，
   先完成 reader-side single-truth cutover
2. 把 transform / target regression 的主断言面从
   `blackhole.*` projection 迁到 `tl.tt_program` 或最终 `ExecutableSpec`，
   然后删除 reader-side fallback
3. reader-side gate 收口后，再继续 translator 输入侧的 producer-side bridge attr 清理
4. 在新主链上继续扩大 direct host path/runtime gate 覆盖，
   并把 `flash-attn` 的 `blackhole.acc` correctness payoff 继续放在 `Phase C2`

## 当前主设备链

```text
LowerDeviceStorageAccessInfo
  -> AugmentSemanticManifest
  -> LowerIntrin
  -> Simplify
  -> HoistBroadcastValues
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> ValidateSemanticRefinement
  -> AnalyzeSpatialDomainPlan
  -> AnalyzeSpatialExecutionPlan
  -> MaterializeSpatialProgram
  -> ValidateSpatialProgram
  -> LowerBlackholeOps
  -> PlanBlackholeCB
  -> AssignBlackholeCores
  -> LowerSpatialProgramToTTTarget
  -> ValidateTTTargetProgram
  -> MaterializeTTExecutableSpec
```

## 最新验证摘要

- build:
  - `cmake -S tilelang_repo -B tilelang_repo/build`
  - `cmake --build tilelang_repo/build -j32`
- transform:
  - `test_blackhole_tt_target_probe.py -q`: `19 passed`
- target:
  - `test_blackhole_copy_pipeline.py test_blackhole_gemm.py test_blackhole_tvm_ffi_export.py test_blackhole_flash_attention_pipeline.py -x`:
    `96 passed, 21 skipped, 1 xfailed`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`:
    `12 passed`

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/README.md`
