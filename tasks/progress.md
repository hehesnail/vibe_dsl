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
  - `SpatialProgram` 已成为 execution-bearing 上游真源
  - 交接边界、审计结论与完成后 contract 统一维护在
    `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- **Phase C**: 进行中
  - 当前已完成：`TTProgram` cutover 子线
    - `TTProgram` translator / validator / materializer 已进入正式主链
    - runtime/codegen 已切到 `TTProgram` direct reader，
      reader-side deletion gate 已收口
    - regression 主断言面与 producer-side translator 输入
      已切到 typed companion truth
    - shared zero-regression baseline 与当前 `Phase C2` runtime gate 持续通过
  - 仍未完成：
    - `flash-attn` `Phase C2` runtime / correctness payoff
    - `topk / fusedmoe / paged decode / chunk recurrence`
      等 family 在新主链下的统一承接
    - 更宽 copy/dataflow 支持面
    - 更宽 synchronization 支持面
    - `Placement / SpatialCapabilityModel / payload-backed node schema`
      的剩余 typed uplift 与真实 consumer 验证

## 当前 blocker

`Phase C` 当前已经不再卡在 `TTProgram` cutover，
而是卡在剩余支持面还没有兑现：

- 为 `flash-attn` multi-GEMM compute kernel 补真正的 direct runtime /
  correctness contract，而不是长期停留在 unsupported gate
- 在当前稳定主链上继续承接
  `topk / fusedmoe / paged decode / chunk recurrence`
  等 family
- 继续扩大 copy/dataflow 与 synchronization 支持面
- 继续把 `Placement / SpatialCapabilityModel / payload-backed node schema`
  的剩余边界收成长期 typed contract

## 独立已知问题

- `TT_METAL_WATCHER=10` 调试 multicore GEMM direct path 时，`tt_metal`
  的 watcher 线程可能在 `WatcherServer::Impl::poll_watcher_data()` 里抛
  `std::runtime_error` 并触发 `SIGABRT`，或在 `TT_METAL_WATCHER_TEST_MODE=1`
  下停在 `Dump #2`
- 当前结论是 watcher-side bug，不是 `BlackholeModule` direct runtime 主链回归；
  direct runtime baseline 需在 `TT_METAL_WATCHER` unset 的正式环境下判断

## 下一步

1. 为 `flash-attn` multi-GEMM compute kernel 设计正式 target contract /
   runtime schema，把当前 explicit unsupported gate 推进成真正可执行的主链
2. 在当前 `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
   主链上继续接 wider family：
   `topk / fusedmoe / paged decode / chunk recurrence`
3. 继续扩大 copy/dataflow 与 synchronization 支持面
4. 把 `Placement / SpatialCapabilityModel / payload-backed node schema`
   中仍未完全站稳的边界继续做 typed uplift 和真实 consumer 验证

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
  - `cmake --build tilelang_repo/build -j32`
- transform:
  - `test_blackhole_tt_target_probe.py -q`: `25 passed`
- target:
  - `test_blackhole_copy_pipeline.py test_blackhole_gemm.py test_blackhole_tvm_ffi_export.py test_blackhole_flash_attention_pipeline.py test_blackhole_tt_target_probe.py -q`:
    `129 passed, 21 skipped, 1 xfailed`
- runtime:
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest testing/python/target/blackhole/test_blackhole_copy_runtime.py testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -q`:
    `13 passed, 2 skipped`
