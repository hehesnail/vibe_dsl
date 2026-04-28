# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-29`
- Active lane:
  `TileComputeDAG-backed ResourceDemand / ResourcePressureReport typed surface`
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
  The first follow-up cleanup is complete:
  production code no longer persists a function-level
  `TileComputeDAG`
  covering cache,
  no public production API exposes a durable DAG covering object,
  and explicit tile-compute source emission stays on selected leaf pattern
  covering.
  This cleanup did not make
  `TileComputeDAG`
  a production decision input yet.
  Its current DAG-wide fanout /
  materialization reasoning remains diagnostic foundation until the next typed
  resource-pressure task consumes it.
  CB planning is useful but partial;
  core placement and buffer distribution remain basic and need hardware-model
  backed planning before wider runtime admission resumes.
- Current blocker:
  resource pressure has no first-class typed report yet;
  `TileComputeDAG`
  fanout /
  materialization decisions do not yet change production validation or
  admission results;
  wider runtime admission remains blocked until resource pressure can fail
  closed before source / runtime emission.
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
  DAG-wide covering remains foundation only until its fanout /
  materialization decisions feed typed resource demand,
  typed unsupported diagnostics,
  or delete old per-op branch mechanics.
- TileComputeDAG production-boundary cleanup:
  production code does not persist DAG covering decisions in
  `PlanTTKernelABI`,
  the covering header exposes only leaf covering decisions and diagnostic FFI,
  and static tests guard against reintroducing a production DAG cache.

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

- Resource pressure has no first-class typed report yet.
  The next surface is
  `ResourceDemand`
  /
  `ResourcePressureReport`
  derived from `TTProgram` / `ExecutableSpec`,
  with pass-local `TileComputeDAG` fanout /
  materialization decisions as the first production input.
  If those DAG decisions cannot change validator /
  admission outcomes,
  the DAG surface must be downgraded to diagnostic-only or deleted rather than
  kept as an active production item.
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

1. Add TileComputeDAG-backed typed resource pressure:
   derive
   `ResourceDemand`
   /
   `ResourcePressureReport`
   from validated `TTProgram`
   and `ExecutableSpec`,
   consume pass-local `TileComputeDAG`
   fanout /
   materialization decisions as the first workload-derived demand source,
   and wire the result to validators / typed unsupported reasons.
   This is the production-use gate for the DAG;
   source emitters and runtime readers must still consume only typed plans.
2. Upgrade CB / L1 admission:
   arch-aware CB limits,
   live-interval CB ID allocation,
   per-core L1 pressure,
   lock-step / alignment estimates,
   and memory-report validation where available.
3. Upgrade core and buffer placement:
   use `TTHardwareModel`
   for worker grid / L1 / DRAM facts,
   produce safe logical-coordinate core groups,
   and expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`.
4. Resume wider runtime admission:
   re-admit multi-block flash-attn direct runtime,
   then wider exact-CB events,
   mesh / distributed runtime,
   and later NoC / multicast / scheduling optimization.

## Latest Verification

- Current documentation alignment:
  merged the
  `TileComputeDAG`
  production-use gate into the next
  `ResourceDemand`
  /
  `ResourcePressureReport`
  task,
  because the DAG is not complete while it only feeds diagnostics.
- Previous TileComputeDAG production-boundary cleanup:
  added static regression tests that reject a production DAG covering cache,
  reject public DAG covering production APIs,
  and keep explicit source / GEMM plan recording on leaf covering decisions.
- Verification:
  documentation-only adjustment;
  no build was required for this progress / design realignment.
  Previous code verification remains:
  `cd tilelang_repo && cmake --build build -j32`
  rebuilt `libtilelang.so`
  successfully.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -k 'tile_compute_production_path_uses_covering_selection or tile_compute_production_path_does_not_persist_dag_covering_cache or tile_compute_covering_header_does_not_expose_dag_covering_as_production_api or tile_compute_explicit_source_path_uses_leaf_covering_without_dag_cache or tile_compute_gemm_plan_construction_uses_leaf_covering_decision' -q`
  passed with `5 passed, 64 deselected`.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -q`
  passed with `69 passed`.
