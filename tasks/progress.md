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
- **Phase B**: 仍在进行中
  - 已完成：boundary cleanup、capability intake、probe、最小 contract hardening
  - 未完成：spatial synthesis algorithm 本体与 stronger execution-bearing contract
- **Phase C**: 已定义；当前只有准备轨落地
  - 已完成：read-only translator demand probe、hardware intake
  - 未开始：正式 `TTProgram / MaterializeTTExecutableSpec` cutover

## `Phase B` 当前已完成

- `Spatial*` object/vocab/shared key 已从 semantic infra 拆出
- `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入主线，
  `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
- `tl.tt_hardware_model` / `tl.spatial_capability_model` 已发布为 module-scope global info
- `LowerToSpatialProgram` 已消费来自 SoC descriptor 的最小 capability snapshot
- `Channel.kind + payload_kind + delivery_kind` 与 `placement.affinity_kind`
  已收成当前 probe intake 所需的最小 contract
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且不会恢复 non-TT-specific spatial semantics
- `ValidateSpatialProgram` 已收正为 coherence / completeness gate

## `Phase B` 当前未完成

- `Task` formation / abstract execution role
- `Flow shaping`
- `Domain realization`
- `Phase / ordering synthesis`
- `ValidateSpatialProgram` 对 stronger contract 的 fail-fast 校验

结论：

- 当前 `SpatialProgram` 只达到 read-only probe intake 的最小上游 contract
- `Phase B` 整体不能判定为完成

## 当前 blocker

- 主 blocker 仍先落在剩余 `Phase B`
- `TTProgram / MaterializeTTExecutableSpec` 仍不存在
- target/runtime 仍主要停留在
  `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole`
  的旧主链
- `flash-attn` 的 `blackhole.acc` correctness payoff 仍归属 `Phase C2`

## 下一步

1. 完成 `Phase B` 正文定义的 spatial synthesis algorithm
2. 补强 `Task / Layout / WorkPartition / ProgramPhase / SyncEdge` 的 stronger contract
3. 补强 `ValidateSpatialProgram`
4. 只有在 `Phase B` 整体完成后，再启动正式 `TTProgram / MaterializeTTExecutableSpec`
   cutover

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

- transform:
  - `test_blackhole_semantic_ir.py`: `40 passed`
  - `test_blackhole_flash_attention_analysis.py`: `7 passed`
  - `test_blackhole_spatial_ir.py`: `44 passed`
  - `test_blackhole_tt_target_probe.py`: `6 passed`
- target:
  - `test_blackhole_copy_pipeline.py -q`: `41 passed, 10 skipped, 1 xfailed`
  - `test_blackhole_copy_runtime.py -q`: `12 passed`
  - `test_blackhole_gemm.py -q`: `26 passed, 11 skipped`
  - `test_blackhole_tvm_ffi_export.py -q`: `1 passed`
  - `test_blackhole_flash_attention_pipeline.py -q`: `27 passed`

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/README.md`
