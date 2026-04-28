# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 细节设计、审计过程和历史流水账不要写在这里。

## Status

- Date: `2026-04-28`
- Active lane: `Tile compute legalizer / DAG covering Phase E cleanup`
- Current state:
  `AccessRegion`,
  `DependenceComponent`,
  `LiveValueSSA`,
  and the TT live-form solver now satisfy
  `Algorithmic generalization Phase E: Decision-Use Cutover`
  for the admitted compute surface.
  On the active chain,
  access-region compatibility,
  recurrence evidence,
  indexed live-value / materialization-boundary queries,
  graph-worklist live-form solving,
  and projection admission validation now affect legality /
  query /
  typed plans /
  unsupported diagnostics on the active chain.
  Tile compute legalizer / DAG covering Phase C-D is complete for the
  admitted compute surface.
  The pass-local
  `TileComputeDAG`
  now has a typed C++ builder with producer-use edges connected by IR object
  identity,
  and
  `SelectBlackholeTileComputeDAGCoveringDiagnostic`
  runs local DAG covering in dependence order.
  The covering diagnostic reports selected patterns,
  source-emitter hooks,
  total cost,
  fanout decisions,
  materialization policy decisions,
  and a fail-closed stale-fallback policy.
  `TTComputeOpPlan`
  recording,
  GEMM `matmul_tiles`
  plan construction,
  explicit
  `tl.tileop.blackhole_compute`
  source dispatch,
  and
  `ValidateTTProgram`
  now select a covering pattern before accepting the operation.
  The selected pattern now carries a
  `source_emitter`
  hook used by explicit source dispatch;
  the old operation-name dispatch chain and add/mul builtin-selection branch
  have been removed from the covered source path.
  The selected path still reuses existing low-level emitter functions;
  deleting the remaining low-level per-op emitter mechanics is the separate
  tile-compute covering Phase E cleanup.
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
  Phase D first TT live-form solver,
  and Phase E decision-use cutover are complete.
- Tile compute legalizer / DAG covering:
  Phase A read-only `TileComputeDAG` diagnostic /
  pattern schema and Phase B legalizer scaffolding are complete.
  The legalizer validates current admitted compute plans and rejects
  unsupported synthetic operation names before `TTProgram` validation
  can pass.
  Phase C-D production migration is complete:
  local covering selection now gates typed compute-plan recording,
  explicit tile-compute source dispatch,
  and
  `TTProgram`
  validation for the migrated leaf-op family.
  Pattern metadata now owns source-emitter hook selection,
  including split hooks for
  `add_tiles`
  and
  `mul_tiles`;
  DAG covering now emits selected pattern IDs/costs plus fanout and
  materialization policy decisions.

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

- `TileComputeDAG` / legalizer / covering Phase E cleanup remains:
  remove or collapse remaining low-level per-op emitter mechanics now that
  pattern covering owns source-emitter selection and DAG covering exposes
  fanout/materialization policy decisions.
- Multi-block flash-attn direct-runtime correctness remains runtime-gated
  behind typed unsupported-reason metadata.
- Wider exact-CB multi-page publish/consume events remain outside the admitted
  direct-runtime support surface.

## Next Task Order

1. Continue `Tile compute legalizer / DAG covering Phase E`
   by deleting or collapsing remaining low-level per-op emitter mechanics
   for the admitted compute surface.
2. Re-admit multi-block flash-attn direct runtime through typed
   `TTProgram -> ExecutableSpec` state and TT-Sim bf16 correctness.
3. Add wider exact-CB event admission for stage2/block64 shapes.
4. Expand mesh/distributed runtime admission through typed schema.
5. Expand flash-attn wider-shape runtime admission ladder.

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
- `Algorithmic generalization Phase E: Decision-Use Cutover`
  completed for the admitted compute surface:
  wider subject live-value maps were removed from TT planning owner truth,
  materialization planning now queries indexed
  `MaterializationBoundary` records,
  the live-form solver runs over a validated boundary graph with a
  worklist/lattice surface,
  and executable projection rejects missing live edge /
  boundary evidence before leaf encoding.
- `Tile compute legalizer / DAG covering Phase A-B`
  completed as foundation:
  `TileComputeDAG` read-only diagnostic covers explicit reduce,
  GEMM,
  and flash-attn leaf compute calls;
  the pattern table covers current TT-Metal leaf operation names;
  and the legalizer is wired into current compute-plan recording plus
  `ValidateTTProgram`.
- `Tile compute legalizer / DAG covering Phase C-D`
  production migration completed for the admitted compute surface:
  `SelectBlackholeTileComputeCovering`
  gates `TTComputeOpPlan`
  recording,
  GEMM `matmul_tiles`
  plan construction,
  explicit blackhole tile-compute source dispatch,
  and `ValidateTTProgram`.
  The selected pattern now carries
  `source_emitter`
  metadata,
  explicit source dispatch uses that hook,
  and the covered source path no longer has the old operation-name dispatch
  chain or add/mul operation-name builtin-selection branch.
  `TileComputeDAG`
  now exposes typed pass-local producer-use edges,
  and DAG covering reports selected patterns,
  local-DP cost,
  fanout decisions,
  materialization policies,
  and stale-fallback rejection.
- `cmake --build tilelang_repo/build -j32`
  -> passed.
- `pytest -q tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'tile_compute_covering_rejects_composite or tile_compute_dag_covering_selects or tile_compute_dag_covering_reports'`
  -> 3 passed, 56 deselected.
- `pytest -q tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
  -> 59 passed.
- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  -> 163 passed, 25 skipped, 1 xfailed.
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
