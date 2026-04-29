# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-29`
- Active lane:
  `Tile-compute explicit leaf normalization boundary repair`
- Current state:
  Repo-level design discipline now has a hardware-codegen usefulness gate:
  new algorithmic structures,
  typed fields,
  validators,
  or DAG machinery count as mainline progress only when they make a
  DSL-authored kernel lower to real TT-Metal hardware code more reliably or
  more efficiently by changing leaf normalization,
  legality,
  typed plans,
  resource plans,
  admission diagnostics,
  or by deleting old side channels.
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  and the first TT live-form solver are established foundations and already
  affect admitted live-form / materialization decisions.
  They are not a global resource allocator.
  `TileComputeDAG`
  is constrained to pass-local explicit-leaf graph legalization /
  covering.
  It has no mainline value merely because it exists or checks op names;
  it must drive leaf-graph fanout,
  materialization,
  physical-form,
  resource-demand,
  typed reject decisions,
  or delete old per-op branches.
  It must not become composite lowering,
  resource allocation,
  core placement,
  NoC scheduling,
  or a cross-pass payload surface.
  Resource allocation today is split across
  `PlanTTCBAlloc`,
  `PlanTTCoreGroups`,
  `TTBufferDistributionPlan`,
  `TTHardwareModel`,
  and leaf runtime admission.
  The first follow-up cleanup narrowed the API boundary:
  production code no longer persists a function-level
  `TileComputeDAG`
  covering cache,
  no public production API exposes a durable DAG covering object,
  and explicit tile-compute source emission stays on selected leaf pattern
  covering.
  It did not complete production source lowering,
  because selected hooks can still carry composite pseudo-leaf payloads.
  A first typed surface exists through
  `TTResourceDemand`
  /
  `TTResourcePressureReport`:
  full pre-selection DAG fanout,
  materialization,
  and unsupported covering reasons are captured before builtin selection,
  carried as typed TTProgram fields,
  validated by `ValidateTTProgram`,
  and projected into `ExecutableSpec`.
  This does not by itself prove the production DAG is justified:
  after composite pseudo-leaf cleanup,
  the DAG must still show real leaf-graph decisions that change typed plans,
  validators,
  resource admission,
  or old branch deletion.
  The
  `2026-04-29`
  boundary review found that the source-lowering use went too far:
  `PlanTTKernelABI`
  currently lets selected source hooks legitimize composite pseudo-leaf
  payloads such as
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  and
  `mul_tiles_bcast_cols("div", ...)`.
  That is not an accepted design boundary.
  The repair target is explicit
  `Normalized Tile TIR`
  leaf-sequence normalization before DAG construction;
  `TileComputeDAG`
  then covers only one semantic TT-Metal leaf per source node.
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
  tile-compute source lowering has composite pseudo-leaf residue.
  The known violations are the
  `exp2_tile`
  affine payload path
  and
  `mul_tiles_bcast_cols("div", ...)`.
  These must be deleted and replaced by explicit leaf TIR normalization before
  wider core / buffer placement work resumes.
  After that,
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
  and selected admitted-surface decision-use cutover are present for
  live-form /
  materialization decisions.
  They are not a compute expression lowering solution and not a global
  resource allocator.
- Tile compute legalizer / DAG covering foundation:
  local `TileComputeDAG` and pattern / legalizer scaffolding exist for the
  admitted compute surface.
  DAG-wide fanout /
  materialization /
  unsupported-reason decisions now feed typed resource demand and validator /
  executable projection,
  but source hooks must still be repaired so each source node maps to one
  semantic leaf plan and no composite payload is accepted.
- TileComputeDAG production-boundary cleanup:
  production code does not persist a durable function-level DAG covering cache;
  `PlanTTKernelABI`
  only owns per-run pass-local DAG lowering decisions,
  the covering header exposes only leaf covering decisions and diagnostic FFI,
  and static tests guard against reintroducing a production DAG cache.
- Typed resource pressure surface:
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
  This is a typed surface, not final proof that DAG belongs on the production
  path; that proof still depends on post-cleanup leaf-graph decisions changing
  real lowering/resource/admission behavior.
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
- Tile-compute lower-plan residue:
  current repo HEAD has the pass-local lower-plan mechanics,
  but the production boundary is not clean while source hooks can expand
  composite pseudo-leaf payloads.
  Completion now requires explicit leaf TIR normalization and one DAG source
  node to one semantic `TTComputeOpPlan.operation_name` validation.
  If the repaired DAG only wraps already-selected leaves without changing
  fanout,
  materialization,
  resource-demand,
  typed rejects,
  or deleting old branches,
  it must be downgraded to debug /
  validation support instead of production machinery.

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

- Composite pseudo-leaf cleanup:
  remove
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  and
  `mul_tiles_bcast_cols("div", ...)`
  source payloads;
  express those TIR computations as explicit TT-Metal leaf op sequences in
  `Normalized Tile TIR`;
  tighten pattern schema and validators so DAG-driven source hooks are
  one-to-one with semantic leaf plans.
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

1. Repair tile-compute explicit leaf normalization:
   delete composite pseudo-leaf source payloads,
   implement explicit leaf TIR decomposition for admitted TIR expressions,
   and enforce one DAG source node to one semantic leaf plan.
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
  doc-only synchronization after the hardware-codegen usefulness review.
  The docs now classify
  `AccessRegion` /
  `LiveValueSSA`
  as dataflow /
  liveness /
  materialization substrate,
  not compute expression lowering.
  `TileComputeDAG`
  is now documented as useful only if post-cleanup explicit leaf graph
  decisions affect fanout,
  materialization,
  physical form,
  resource demand,
  typed rejects,
  or old branch deletion.
  Existing
  `PlanTTKernelABI`
  lower-plan mechanics still exist in code,
  but they are not production-complete while composite pseudo-leaf source
  payloads remain.
- Verification:
  docs-only change;
  no build or pytest was run.
