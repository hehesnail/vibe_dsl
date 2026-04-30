# TileLang Blackhole Backend Progress

> 当前 HEAD 看板只放状态、blocker、下一步和最近验证。
> 设计合同不要写在这里；设计依据看
> `tasks/dev_design/final_blackhole_backend_redesign.md`
> 和对应任务级设计文档。

## Status

- Date: `2026-04-30`
- Active lane: `Hardware-model-backed core and buffer placement`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current HEAD

- Legacy external runner / `build_blackhole/` 已删除。
- Blackhole 正式执行路径是进程内 `BlackholeModule` direct host path；
  `tilelang.compile(..., execution_backend="tvm_ffi")`
  Blackhole wrapper/export path 可用。
- Broad legacy protocols 已退出 active chain：
  `compute_contract`,
  `gemm_contract`,
  `multi_*_contracts`,
  top-level `TTProgram.payload`,
  bridge attrs,
  lowering facts contract maps,
  compute-op seed maps,
  leaf name/default fallbacks。
- Tile compute truth 保持 TT-Metal leaf API 粒度。
  已删除 late scalar-loop matcher / generate family。
  Blackhole scalar-loop normalization 已从通用
  `lower_tile_op.cc`
  抽到独立 common normalizer；
  source projection 已从
  `PlanTTKernelABI`
  大接口面收窄到
  `BlackholeTileComputeSourceProjection`。
  当前实现继续收缩重复 lowering mechanics：
  normalizer 用 unary / binary leaf builders 生成同形 calls；
  source projection 用 binary / broadcast-cols / unary category emitters；
  row-reduction source emission 复用
  `ExactTileComputeEmitter`
  的 CB / tile-register / pack sequence。
- Algorithmic generalization foundation 已存在并在 admitted live-form /
  materialization 决策中使用：
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  first TT live-form solver。
  这些不是 compute expression lowering 或全局 resource allocator。
- `TileComputeDAG`
  只允许作为 pass-local explicit-leaf graph legalization /
  covering model。
  它不能做 composite lowering、
  resource allocation、
  core placement、
  NoC scheduling
  或跨阶段 payload。
- Known composite pseudo-leaf source payload 已清理：
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  改为 normalized explicit leaf sequence；
  `mul_tiles_bcast_cols("div", ...)`
  改为
  `recip_tile + mul_tiles_bcast_cols`。
  Source hooks 现在只能投影一个 selected semantic TT-Metal leaf op。
- First typed resource-pressure surface 已存在：
  `TTResourceDemand`
  /
  `TTResourcePressureReport`
  进入 `TTProgram`，
  被 `ValidateTTProgram` 消费，
  并投影到 `ExecutableSpec`。
- CB / L1 admission 已接入第一版 hardware facts：
  `TTHardwareModel.max_cb_count`,
  worker L1 budget,
  L1 alignment,
  aligned CB bytes,
  allocator-managed L1 pressure。
  `ValidateTTProgram`
  会在 source / runtime emission 前 fail closed。

## Current Blocker

Core placement 和 buffer distribution 仍然过粗：

- `PlanTTCoreGroups`
  仍主要走 hard-coded logical grid path。
- `TTBufferDistributionPlan`
  仍主要是
  `unit_mesh`
  /
  `replicated`。
- Wider runtime admission
  需要 core groups 和 buffer placement
  先消费
  `TTHardwareModel`
  facts，
  并在 source / runtime emission 前给出 typed reject。

## Next Task Order

1. Upgrade core and buffer placement:
   use `TTHardwareModel`
   for worker grid / worker count / L1 / DRAM facts,
   produce safe logical-coordinate core groups,
   and expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`.
2. Resume wider runtime admission:
   multi-block flash-attn direct runtime,
   wider exact-CB events,
   mesh / distributed runtime,
   later NoC / multicast / scheduling optimization.

## Support Boundary

- Direct runtime admitted subset:
  copy equal source/dest range with stride 1;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM accessor with `common_runtime_arg_count = 0`;
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths.
- Flash-attn admitted direct-runtime subset:
  small single-work-item and 32x32 MHA / GQA bf16.
- Flash-attn compile/source/spec stable but runtime-gated subset:
  seq64 / multi-K-step MHA and GQA.
- Not admitted:
  multi-block flash-attn direct-runtime correctness,
  larger multi-page exact-CB publish/consume events,
  full multi-device / sharded / fabric collective runtime.

## Latest Verification

Latest code implementation batch:
current HEAD after
Blackhole tile-compute lowering duplicate-logic cleanup.

Verification for this batch:

- `cmake --build build -j32`
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k 'single_blackhole_tile_compute_normalizer_surface or source_projection_is_not_declared or tile_compute or builtin_selector'`
  (`29 passed, 51 deselected`)
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py`
  (`80 passed`)
- `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k 'leaf_compute_ops or optimized_path_lowers_acc_o_broadcast_updates or optimized_path_lowers_exp2_to_leaf_tile_ops or projects_non_gemm_exact_compute_ops'`
  (`3 passed, 64 deselected`)
- `git diff --check`
  (`passed`)
- Cleanup scan found no active
  `GetBlackholeTileComputeStringArg`,
  old composite generator,
  or string-mode composite payload residue under
  `tilelang_repo/src/transform`.
- Boundary scan asserts
  `lower_tile_op.cc`
  no longer defines Blackhole leaf-call builders /
  normalizer helpers,
  `lower_blackhole_ops.h`
  no longer declares tile-compute source emitter hook methods,
  source projection no longer carries per-leaf add/mul/exp2/recip emit
  methods,
  and `lower_blackhole_tile_compute.cc`
  no longer hand-writes CB / tile-register / pack calls outside
  `ExactTileComputeEmitter`.

Latest doc cleanup verification:

- Core active docs were reduced back to role-specific contracts:
  `final_blackhole_backend_redesign.md`,
  `README.md`,
  `progress.md`,
  `2026-04-28-blackhole-algorithmic-generalization.md`,
  `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`,
  `2026-04-29-blackhole-resource-planning-roadmap.md`,
  and
  `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md`.
- `rg`
  stale-log scan over those docs found no lingering
  phase-log /
  stale production-completion wording.
- `git diff --check`
  passed.
- Docs-only cleanup;
  no build or pytest was required for this change.
