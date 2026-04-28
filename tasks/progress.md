# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-29`
- Active lane:
  `Algorithmic generalization Phase E follow-up:
  resource-planning alignment`
- Current state:
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  and the first TT live-form solver are established foundations and already
  affect admitted live-form / materialization decisions.
  They are not a global resource allocator.
  `TileComputeDAG`
  is constrained to pass-local tile-compute legalization / covering;
  it must not become a resource allocation,
  core placement,
  NoC scheduling,
  or cross-pass payload surface.
  Resource allocation today is split across
  `PlanTTCBAlloc`,
  `PlanTTCoreGroups`,
  `TTBufferDistributionPlan`,
  `TTHardwareModel`,
  and leaf runtime admission.
  CB planning is useful but partial;
  core placement and buffer distribution remain basic and need hardware-model
  backed planning before wider runtime admission resumes.
- Current blocker:
  resource-planning boundaries must be recorded and enforced before expanding
  multi-block flash-attn direct-runtime admission or adding more DAG-covering
  production hooks.
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Completed Baseline

- Legacy external runner / `build_blackhole/`: deleted.
- Blackhole formal execution path:
  in-process `BlackholeModule` direct host path.
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  Blackhole wrapper/export path: restored.
- Broad legacy protocol deletion:
  `compute_contract`,
  `gemm_contract`,
  `multi_*_contracts`,
  top-level `TTProgram.payload`,
  bridge attrs,
  lowering facts contract maps,
  compute-op seed maps,
  and leaf name/default fallbacks are out of the active chain.
- Flash-attn support:
  P2.1 exact row-reduction source-live truth,
  P2.2 small / 32x32 bf16 direct-runtime admission,
  and P2.3 seq64 exact-CB compile/source/spec admission are complete.
- Tile-compute preservation:
  downstream scalar-loop matcher / generate families are deleted;
  compute truth is preserved at TT-Metal leaf API granularity.
- Algorithmic generalization foundation:
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  first TT live-form solver,
  and admitted-surface decision-use cutover are present.
- Tile compute legalizer / DAG covering foundation:
  local `TileComputeDAG` and pattern / legalizer scaffolding exist for the
  admitted compute surface.
  Production use remains valid only when the selected covering changes typed
  plans / unsupported diagnostics or deletes old per-op branch mechanics.

## Support Boundary

- Direct runtime admitted subset:
  copy equal source/dest range with stride 1;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM accessor with `common_runtime_arg_count = 0`;
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths already covered by the typed gate.
- Flash-attn admitted direct-runtime subset:
  small single-work-item and 32x32 MHA / GQA bf16.
- Flash-attn compile/source/spec stable but runtime-gated subset:
  seq64 / multi-K-step MHA and GQA.
- Not admitted:
  multi-block flash-attn direct-runtime correctness,
  larger stage2/block64 multi-page exact-CB publish/consume events,
  full multi-device / sharded / fabric collective runtime.

## Open Debt

- `TileComputeDAG`
  production boundary must stay pass-local and compute-selection-only.
  Any cursor / `try_*` / fallback hook that does not change typed plans,
  diagnostics,
  or delete old branch mechanics should be simplified or removed.
- Resource pressure has no first-class typed report yet.
  The next surface is
  `ResourceDemand`
  /
  `ResourcePressureReport`
  derived from `TTProgram` / `ExecutableSpec`.
- CB allocation needs arch-aware limits and live-interval allocation.
  Blackhole / Wormhole CB limits must come from target / hardware-model facts,
  not stale fixed constants.
- L1 admission currently checks only coarse pressure.
  It needs explicit CB bytes,
  allocator-managed L1 buffer pressure,
  worker L1 budget,
  lock-step / alignment estimates,
  and memory-report validation hooks.
- Core placement still relies on a hard-coded grid path.
  It must consume `TTHardwareModel`
  and produce safe logical-coordinate core groups /
  `CoreRangeSet`-compatible plans.
- Buffer distribution remains mostly unit mesh / replicated.
  It needs explicit interleaved DRAM,
  interleaved L1,
  sharded L1,
  host-visible,
  device-local,
  page-size,
  shard-shape,
  and core-range decisions.
- Multi-block flash-attn direct-runtime correctness remains runtime-gated
  behind typed unsupported-reason metadata.
- Wider exact-CB multi-page publish/consume events remain outside the admitted
  direct-runtime support surface.

## Next Task Order

1. Constrain and clean `TileComputeDAG`
   production use:
   keep it pass-local,
   remove over-complex hooks that do not pay rent,
   and make source emission a projection of selected typed plans.
2. Add typed resource pressure:
   derive
   `ResourceDemand`
   /
   `ResourcePressureReport`
   from validated `TTProgram`
   and `ExecutableSpec`,
   and wire it to validators / typed unsupported reasons.
3. Upgrade CB / L1 admission:
   arch-aware CB limits,
   live-interval CB ID allocation,
   per-core L1 pressure,
   lock-step / alignment estimates,
   and memory-report validation where available.
4. Upgrade core and buffer placement:
   use `TTHardwareModel`
   for worker grid / L1 / DRAM facts,
   produce safe logical-coordinate core groups,
   and expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`.
5. Resume wider runtime admission:
   re-admit multi-block flash-attn direct runtime,
   then wider exact-CB events,
   mesh / distributed runtime,
   and later NoC / multicast / scheduling optimization.

## Latest Verification

- Documentation-only update:
  recorded resource-planning roadmap in
  `tasks/dev_design/2026-04-29-blackhole-resource-planning-roadmap.md`,
  updated the active design index,
  and compacted this progress board around the revised task order.
- No build or runtime test was required for this documentation change.
