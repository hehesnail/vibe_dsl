# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-07`
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
- **Phase C**: 已定义；准备轨已完成
  - 已完成：read-only translator demand probe、hardware intake
  - 未开始：正式 `TTProgram / MaterializeTTExecutableSpec` cutover

## `Phase B` 收尾结果

- `Spatial*` object/vocab/shared key 已从 semantic infra 拆出
- `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入主线，
  `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
- `tl.tt_hardware_model` / `tl.spatial_capability_model` 已发布为 module-scope global info
- `LowerToSpatialProgram` 已消费来自 SoC descriptor 的最小 capability snapshot
- `Channel.kind + payload_kind + delivery_kind` 与 `placement.affinity_kind`
  已收成当前 probe intake 所需的最小 contract
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且不会恢复 non-TT-specific spatial semantics
- `ValidateSpatialProgram` 已收正为 coherence / completeness / semantic alignment /
  ordering legality gate
- generic builder 已把 `Task` formation、flow shaping、domain realization、
  phase / ordering synthesis 收进稳定主链
- `phase_boundary_materialization` 已收窄为真实跨 phase state handoff，
  不再把“任意后续 phase 读取的 state”过度记为边界物化

结论：

- `SpatialProgram` 已达到 execution-bearing spatial program 的完成判定
- `Phase C` 现在可以把 `SpatialProgram` 当成单一上游真源继续 cutover

## 当前 blocker

- `TTProgram / MaterializeTTExecutableSpec` 仍不存在
- target/runtime 正式 cutover 仍主要停留在
  `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole`
  的旧主链
- `flash-attn` 的 `blackhole.acc` correctness payoff 仍归属 `Phase C2`

## 独立已知问题

- `TT_METAL_WATCHER=10` 调试 multicore GEMM direct path 时，`tt_metal`
  的 watcher 线程可能在 `WatcherServer::Impl::poll_watcher_data()` 里抛
  `std::runtime_error` 并触发 `SIGABRT`，或在 `TT_METAL_WATCHER_TEST_MODE=1`
  下停在 `Dump #2`
- 当前结论是 watcher-side bug，不是 `BlackholeModule` direct runtime 主链回归；
  direct runtime baseline 需在 `TT_METAL_WATCHER` unset 的正式环境下判断

## 下一步

1. 启动正式 `Phase C` cutover：
   `SpatialProgram -> TT Target IR -> TTProgram / MaterializeTTExecutableSpec`
2. 在 cutover 期间保持当前 direct host path 和 `tvm_ffi` export 支持面不回退
3. 把 `flash-attn` 的 `blackhole.acc` correctness payoff 继续放在 `Phase C2`

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
  -> LowerToSpatialProgram
  -> ValidateSpatialProgram
  -> LowerBlackholeOps
  -> PlanBlackholeCB
  -> AssignBlackholeCores
```

## 最新验证摘要

- build:
  - `cmake --build tilelang_repo/build -j32`
- transform:
  - `test_blackhole_spatial_ir.py -q`: `68 passed`
  - `test_blackhole_tt_target_probe.py -q`: `17 passed`
- target:
  - `test_blackhole_copy_pipeline.py -q`: `50 passed, 1 skipped, 1 xfailed`
  - `test_blackhole_copy_runtime.py -q`: `12 passed`
  - `test_blackhole_gemm.py -q`: `38 passed`
  - `test_blackhole_tvm_ffi_export.py -q`: `1 passed`
  - `test_blackhole_flash_attention_pipeline.py -q`: `27 passed`

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/README.md`
