# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-29`
- Active lane:
  `Core / buffer placement hardware-model upgrade`
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
  It also now drives the explicit tile-compute source lowering path:
  `PlanTTKernelABI`
  builds a pass-local DAG lower plan,
  consumes selected source emitters from that plan instead of reselecting
  on the source path,
  and records DAG node /
  source emitter /
  materialization /
  fanout decisions on DAG-driven
  `TTComputeOpPlan`
  entries and executable projection.
  CB / L1 resource admission now uses hardware-model-backed CB count,
  worker L1 budget,
  and L1 alignment facts in both `PlanTTCBAlloc`
  and `TTResourcePressureReport`.
  `ValidateTTProgram`
  rejects over-CB and over-L1 reports before source / runtime emission,
  and executable projection carries the admission facts.
  Core placement and buffer distribution remain basic and need hardware-model
  backed planning before wider runtime admission resumes.
- Current blocker:
  core / buffer placement is still coarse:
  `PlanTTCoreGroups`
  still routes through a hard-coded grid path,
  and `TTBufferDistributionPlan`
  remains mostly `unit_mesh` / `replicated`.
  Wider runtime admission remains blocked until core groups and buffer
  placement consume `TTHardwareModel`
  facts and can fail closed before source / runtime emission.
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
  and source emitters consume a pass-local DAG lower plan before recording
  typed TT plans.
- TileComputeDAG production-boundary cleanup:
  production code does not persist a durable function-level DAG covering cache;
  `PlanTTKernelABI`
  only owns per-run pass-local DAG lowering decisions,
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
- Hardware-backed CB / L1 resource admission:
  `TTHardwareModel`
  carries CB count and L1 alignment facts;
  `PlanTTCBAlloc`
  uses target-derived CB and worker-L1 limits;
  `TTResourcePressureReport`
  records hardware limits,
  aligned CB bytes,
  alignment waste,
  allocator-managed L1 buffer pressure,
  and max simultaneous L1 pressure;
  `ValidateTTProgram`
  rejects CB and L1 over-pressure.
- DAG-driven tile-compute lower plan:
  explicit tile-compute source emission consumes pass-local
  `TileComputeDAG`
  lowering decisions,
  not a second source-path operation selector;
  DAG-driven exact compute lower plans carry
  node id,
  source emitter,
  materialization policy,
  fanout count,
  and fanout policy through
  `TTComputeOpPlan`
  and `ExecutableSpec`.

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

- CB / L1 admission still has future precision work:
  reserved / precolored CB class modeling can be made more explicit,
  L1 buffer sizing is still a conservative plan-level estimate,
  and runtime memory-report hooks should be wired when TT-Sim / TT-Metal
  exposes a stable report interface.
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

1. Upgrade core and buffer placement:
   use `TTHardwareModel`
   for worker grid / L1 / DRAM facts,
   produce safe logical-coordinate core groups,
   and expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`.
2. Resume wider runtime admission:
   re-admit multi-block flash-attn direct runtime,
   then wider exact-CB events,
   mesh / distributed runtime,
   and later NoC / multicast / scheduling optimization.

## Latest Verification

- Current implementation:
  routed `TileComputeDAG`
  covering decisions into the real source-lowering and typed lower-plan path.
  `PlanTTKernelABI`
  now loads a pass-local DAG lower plan,
  `LowerExplicitTileComputeCall`
  consumes that selected covering,
  exact compute plans record DAG node /
  source emitter /
  materialization /
  fanout metadata,
  `ValidateTTProgram`
  validates both the emitted leaf op and the source DAG covering,
  and executable projection carries the fields.
- Verification:
  `cd tilelang_repo && cmake --build build -j32`
  passed.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -k 'tile_compute_explicit_source_path_uses_leaf_covering_without_dag_cache or tile_compute_dag_decisions_drive_typed_compute_lower_plan or executable_projection_carries_dag_driven_compute_lower_plan' -q`
  passed with `3 passed, 74 deselected`.
  `cd tilelang_repo && python -m pytest testing/python/transform/test_blackhole_spatial_ir.py -q`
  passed with `77 passed`.
  `cd tilelang_repo && python -m pytest testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_executable_projection_has_no_plan_local_payload_records testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py::test_flash_attention_forward_tt_target_emits_typed_tt_program_without_payload -q`
  passed with `2 passed`.
