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
- 非 flash-attn 计算面没有删除：
  typed leaf compute surface 仍覆盖
  matmul / copy / unary / binary / broadcast /
  reduce / pack / typecast 等 TT-Metal leaf families。
  Runtime admitted subset 比 typed compute surface 窄，
  以本文件的 support boundary 为准。
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
2. Tighten buffer address ABI:
   make interleaved / sharded / page-indexed buffer address parameters,
   compile-time args,
   runtime args,
   per-work descriptors,
   and typed rejects explicit.
3. Keep admission levels separate for every new subset:
   compile,
   source/spec,
   direct runtime,
   TT-Sim bf16 correctness,
   and typed unsupported reason.
4. Admit first-subset workloads in this order:
   non-flash leaf compute / GEMM variants,
   standalone `topk`,
   exact-CB / materialization / partial-combine primitive
   via multi-block flash-attn,
   grouped / ragged work packets,
   pre-grouped MoE / `fusedmoe`,
   sparse / ragged attention,
   paged GQA decode,
   paged MLA decode,
   then chunk recurrence / scan.
5. Pull forward only the P3 primitives required by the current first subset.
6. Defer production distributed variants until mesh / sharding / CCL /
   NoC / multicast / global scheduling plans are typed and validated.
   Full MoE and full paged decode are not admitted by their first subsets.

## Support Boundary

- Direct runtime admitted subset:
  copy equal source/dest range with stride 1;
  selected interleaved stick / page-shaped copy cases with explicit
  alignment gates;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM buffer address schema with no common runtime args
  (`common_runtime_arg_count = 0`);
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths.
- Typed compute surface broader than direct runtime:
  matmul / copy / unary / binary / broadcast /
  reduce / pack / typecast leaf families must get
  workload-specific admission and correctness gates
  before being counted as runtime support.
- Workload backlog broader than flash-attn:
  `topk`, MoE / `fusedmoe`, paged attention /
  paged decode / MLA decode paged,
  grouped / ragged / sparse attention,
  and chunk recurrence / scan are not admitted yet;
  they require explicit subset definitions and regressions
  before being counted as supported.
- Workload dependency split:
  first single-device subsets are allowed to proceed after typed placement
  and subset-specific admission proof;
  full MoE, full paged attention / decode,
  distributed sparse attention,
  and production flash decode require later P3 runtime features.
  The detailed matrix lives in
  `tasks/dev_design/2026-04-29-blackhole-resource-planning-roadmap.md`.
- Flash-attn admitted direct-runtime subset:
  small single-work-item and 32x32 MHA / GQA bf16.
- Flash-attn compile/source/spec stable but runtime-gated subset:
  seq64 / multi-K-step MHA and GQA.
- Not admitted:
  multi-block flash-attn direct-runtime correctness,
  larger multi-page exact-CB publish/consume events,
  full multi-device / sharded / fabric collective runtime.

## Latest Verification

Latest docs-only planning update:
remaining workload order was re-split around buffer placement,
buffer address ABI,
admission-level verification,
first-subset workload admission,
and later production distributed runtime.
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
