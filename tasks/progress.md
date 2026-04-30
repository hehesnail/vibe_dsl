# TileLang Blackhole Backend Progress

> 当前 HEAD 看板只放状态、blocker、下一步和最近验证摘要。
> 长期设计合同看 `tasks/dev_design/` 下的入口文档。

## Status

- Date: `2026-04-30`
- Active lane: `Hardware-model-backed buffer placement`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current HEAD

- Blackhole 正式执行路径是进程内 `BlackholeModule` direct host path；
  `execution_backend="tvm_ffi"` 的 wrapper/export path 可用。
- Legacy external runner / `build_blackhole/` 已删除。
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
  Composite pseudo-leaf payload 已清理；
  `TileComputeDAG`
  只作为 pass-local explicit-leaf legalization /
  covering input。
- Algorithmic foundation 已存在并有 active consumers：
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  TT live-form solver。
  它们不是 compute expression lowerer 或全局 resource allocator。
- `TTResourceDemand`
  /
  `TTResourcePressureReport`
  已进入 `TTProgram`，
  并由 `ValidateTTProgram`
  消费。
- CB / L1 admission 已使用
  `TTHardwareModel`
  的 CB count、
  worker L1 budget
  和 alignment facts。
- Core groups 已开始消费
  `TTHardwareModel`
  的 worker grid /
  functional worker count；
  validators 会拒绝 out-of-grid core、
  duplicate core
  和非法 work packet。
- Blackhole C++ audit 第一批已收口：
  scalar bitcast、
  DLTensor compact-layout gate、
  leaf-reader fail-closed、
  resource Var alias、
  module bytes serialization/export-load
  均有回归覆盖。

## Current Blocker

`TTBufferDistributionPlan`
仍主要停留在
`unit_mesh` /
`replicated`。

Wider runtime admission 还需要 buffer placement
继续消费 `TTHardwareModel`
的 L1 / DRAM facts，
并在 source / runtime emission 前给出 typed reject。

## Next Task Order

1. Upgrade buffer placement:
   expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`,
   consume L1 / DRAM hardware facts,
   and validate placement before emission.
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

Latest docs-only cleanup:
active design docs were reduced to role-specific contracts;
role-drift scan and `git diff --check` passed.
No build or pytest was required for docs-only changes.

Latest implementation batch:
Blackhole modern C++ audit and hardware-model-backed core-group repair.

Verified:

- C++ build passed.
- Full `test_blackhole_spatial_ir.py` passed.
- Full Blackhole copy runtime suite passed under TT-Sim.
- Full Blackhole GEMM suite passed under TT-Sim.
- Blackhole export/load,
  leaf-reader fail-closed,
  compile-time ABI,
  and non-compact DLTensor gates passed.
- `git diff --check` passed.
