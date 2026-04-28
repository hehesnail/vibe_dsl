# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-29`
- Active lane:
  `CB / L1 resource admission upgrade`
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
  `TileComputeDAG`
  now pays rent on the production path through typed
  `TTResourceDemand`
  /
  `TTResourcePressureReport`:
  full pre-selection DAG fanout,
  materialization,
  and unsupported covering reasons are captured before builtin selection,
  carried as typed TTProgram fields,
  validated by `ValidateTTProgram`,
  and projected into `ExecutableSpec`.
  CB planning is useful but partial;
  core placement and buffer distribution remain basic and need hardware-model
  backed planning before wider runtime admission resumes.
- Current blocker:
  CB / L1 admission is still coarse:
  CB ID pressure and CB L1 bytes are reported,
  but CB limits are not yet arch-aware,
  CB reuse is not a live-interval allocator,
  allocator-managed L1 buffer pressure is not included,
  and worker L1 budget is not enforced through the resource pressure report;
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
  DAG-wide fanout /
  materialization /
  unsupported-reason decisions now feed typed resource demand and validator /
  executable projection,
  while source emitters and runtime readers still consume only typed TT plans.
- TileComputeDAG production-boundary cleanup:
  production code does not persist DAG covering decisions in
  `PlanTTKernelABI`,
  the covering header exposes only leaf covering decisions and diagnostic FFI,
  and static tests guard against reintroducing a production DAG cache.
- DAG-backed typed resource pressure:
  `TTResourceDemand`
  and `TTResourcePressureReport`
  are first-class TTProgram fields;
  the full pre-selection `TileComputeDAG`
  feeds typed fanout /
  materialization /
  unsupported-reason demand,
  validators consume the reports,
  and executable projection carries
  `resource_pressure_reports`.

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

1. Upgrade CB / L1 admission:
   arch-aware CB limits,
   live-interval CB ID allocation,
   per-core L1 pressure,
   lock-step / alignment estimates,
   and memory-report validation where available.
2. Upgrade core and buffer placement:
   use `TTHardwareModel`
   for worker grid / L1 / DRAM facts,
   produce safe logical-coordinate core groups,
   and expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`.
3. Resume wider runtime admission:
   re-admit multi-block flash-attn direct runtime,
   then wider exact-CB events,
   mesh / distributed runtime,
   and later NoC / multicast / scheduling optimization.

## Latest Verification

- Current implementation:
  added typed
  `TTTileComputeFanoutDemand`,
  `TTTileComputeMaterializationDemand`,
  `TTResourceDemand`,
  and `TTResourcePressureReport`;
  captured the full pre-selection
  `TileComputeDAG`
  demand at `TTProgram` level;
  refreshed resource counters through later TTProgram planning phases;
  made `ValidateTTProgram`
  require matching pressure reports and reject typed unsupported reasons;
  and projected
  `resource_pressure_reports`
  into the executable spec.
- Verification:
  `cd tilelang_repo && cmake --build build -j32`
  passed.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -k 'validate_tt_program_consumes_typed_resource_pressure_report' -q`
  passed with `1 passed, 71 deselected`.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -q`
  passed with `72 passed`.
  `cd tilelang_repo && python -m pytest testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_executable_projection_has_no_plan_local_payload_records testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py::test_flash_attention_forward_tt_target_emits_typed_tt_program_without_payload -q`
  passed with `2 passed`.
