# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-09`
- **总阶段**: Stage 4
- **当前主线**: `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- copy / GEMM 当前支持面不回退
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- direct runtime 当前正式支持面：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range；
    当 `core_plan.work_packets` oversubscribe physical cores 时，
    direct runtime 会按 packet truth 做 multi-wave launch scheduling，
    前提是 executable 没有显式 `semaphore / remote-core` synchronization contract
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
    - `flash-attn` direct runtime 已补上 `K` staged-copy transpose truth：
      `transpose_2d` 现在会从 accessor/materialization schema
      进入 host tilize / readback materialization，small bf16 MHA TT-Sim
      direct runtime 已和 reference 数值对齐
    - GEMM direct runtime 已补上 oversubscribed `work_packets` host scheduling：
      runtime 不再把 packet 扁平成单波次 unique-core launch；
      无显式同步 contract 的 oversubscribed case 会按 packet truth 分 wave 执行，
      `352x352x128` oversubscribed regression 与 `512x512x512 bf16`
      direct runtime 数值对齐都已通过
  - 仍未完成：
    - `flash-attn` `Phase C2` 的 wider runtime / correctness payoff：
      当前只验证了 small bf16 MHA 子集；更宽 `MHA / GQA`、更大 shape
      与 TT-Sim `float16` 路径仍未完成
    - `topk / fusedmoe / paged decode / chunk recurrence`
      等 family 在新主链下的统一承接
    - 更宽 copy/dataflow 支持面
    - 更宽 synchronization 支持面
    - `Placement / SpatialCapabilityModel / payload-backed node schema`
      的剩余 typed uplift 与真实 consumer 验证

## 当前 blocker

`Phase C` 当前已经不再卡在 `TTProgram` cutover，
也不再卡在 `flash-attn` direct runtime 的 `K` transpose 丢失；
当前 blocker 是剩余支持面还没有兑现：

- 把 `flash-attn` 当前 small bf16 correctness milestone
  扩成更宽 `MHA / GQA` runtime / correctness 支持面，
  并把剩余 multi-GEMM compute contract 继续收成长期 typed truth
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

1. 在当前已打通的小 bf16 MHA correctness 基线上，
   继续扩大 `flash-attn` `Phase C2` 支持面：
   收完整 multi-GEMM target contract，补更宽 `MHA / GQA` 与大 shape case，
   并和 TT-Sim `float16` 能力边界分开判断
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
- runtime:
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python && export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib && export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k 'small_bf16_compute_source_keeps_acc_s_cast_cb_pages_consistent or small_bf16_metadata_marks_k_materialization_as_transposed'`:
    `2 passed, 46 deselected`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python && export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib && export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k 'gemm_basic or oversubscribed_work_packets'`:
    `2 passed, 38 deselected`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python && export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib && export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'small_bf16_forward_direct_runtime'`:
    `1 passed, 6 deselected`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python && export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib && export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH && cd /root/dev/vibe_dsl/tilelang_repo && timeout 300s python -u - <<'PY'` 直接运行 `multicore_gemm_kernel(M=512, N=512, K=512)` 的 `bf16` direct runtime 数值对比：
    `max_abs=4.58e-05, mean_abs=2.72e-06, allclose=True`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python && export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib && export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH && cd /root/dev/vibe_dsl/tilelang_repo && pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha_forward_direct_runtime' -vv -s`:
    失败于 `UntestedFunctionality: tensix_execute_unpacr: fp16`
  - 上述较大 `float16` MHA 失败
    这属于 simulator 能力边界，不能和本次 `K` transpose correctness 修复混为一谈
