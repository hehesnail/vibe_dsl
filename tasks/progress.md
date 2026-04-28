# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-28`
- Active lane: `Algorithmic generalization Phase E: Decision-Use Cutover`
- Current state:
  `AccessRegion`,
  `DependenceComponent`,
  `LiveValueSSA`,
  and the first TT live-form solver are implemented as foundations.
  The first decision-use slice is on the active chain:
  access-region compatibility,
  recurrence evidence,
  live-value boundary indices,
  and boundary logical coverage now affect legality / query /
  solver output.
- Current blocker:
  none for tile-compute preservation.
  Multi-block flash-attn direct-runtime correctness remains outside the
  admitted runtime surface and fails closed through
  `multi-block exact CB-republish flash-attention direct runtime correctness`.
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
- Post-preservation pass shrink:
  tile compute,
  exact-CB helpers,
  materialization,
  ABI/accessor planning,
  live-form/state bookkeeping,
  staged transport,
  matmul lowering,
  and lower-tile-op normalizer dedup are split / cleaned up.
- Algorithmic generalization:
  Phase A `AccessRegion`,
  Phase B graph-backed `SpatialPlan` dependence,
  Phase C `LiveValueSSA`,
  and Phase D first TT live-form solver are complete.

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

- Finish Phase E by turning algorithmic evidence into broad active-chain
  decisions, not just structure / dumps / validators.
- Replace remaining subject-pair live-value lookup mechanics with indexed
  `LiveValueEdge` / `MaterializationBoundary` queries.
- Expand the live-form solver into a graph-wide worklist/lattice solver over
  validated live edges.
- Audit TTProgram / ExecutableSpec projection admission so unsupported
  diagnostics are typed and early.
- Keep `TileComputeDAG` / legalizer / covering out of production migration
  until Phase E decision-use gates pay rent on the active chain.

## Next Task Order

1. Finish `Algorithmic generalization Phase E: Decision-Use Cutover`.
2. Start `Tile compute legalizer / DAG covering Phase A-B`
   only after Phase E usage-complete checks are in place.
3. Migrate legalizer / DAG covering to production and delete old per-op
   branch mechanics for the admitted compute surface.
4. Re-admit multi-block flash-attn direct runtime through typed
   `TTProgram -> ExecutableSpec` state and TT-Sim bf16 correctness.
5. Add wider exact-CB event admission for stage2/block64 shapes.
6. Expand mesh/distributed runtime admission through typed schema.
7. Expand flash-attn wider-shape runtime admission ladder.

## Latest Verification

- `tasks/progress.md` compacted back to a status board:
  current state,
  blocker,
  completed baseline,
  support boundary,
  open debt,
  next task order,
  and latest verification only.
- Historical document-sync audit details were removed from this file.
- `git diff --check`
  -> passed.
- stale-current-state scan for old flash-attn lane wording,
  old live-form blocker wording,
  old follow-up phase labels,
  current cleanup wording,
  and root design-directory scope contradictions
  -> no conflicting active-doc hits.
- background process scan:
  no lingering `pytest` / `cmake --build` / `ninja` process.
