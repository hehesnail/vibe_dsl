# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 不写长期设计细节。

## Status

- Date: `2026-04-28`
- Active lane: `Blackhole algorithmic generalization implementation queue`
- Current item:
  2026-04-28 follow-up task order refreshed after algorithmic
  generalization design and architecture review.  The design is split into
  `2026-04-28-blackhole-algorithmic-generalization.md`
  for `AccessRegion`, graph-backed `SpatialPlan` dependence, and
  `LiveValueSSA`, plus
  `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
  for `TileComputeDAG`, legalization, and TT-Metal leaf pattern covering.
  The next implementation unit is
  `Algorithmic generalization Phase A: AccessRegion foundation`.
- Blocker:
  No blocker remains for Blackhole tile-compute preservation itself.
  Active lowering no longer recovers row-reduction / broadcast /
  exp2-affine / scalar-fma / fill / copy / typecast compute semantics
  from downstream scalar-loop matcher families.  Multi-block flash-attn
  direct-runtime correctness remains outside the admitted runtime surface
  and fails closed through the direct-runtime unsupported-reason gate.
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Completed Baseline

- `P0.1 contract-family public surface`: completed
- `P0.2 compute operand binding`: completed
- `P0.3 host wrapper / codegen buffer binding`: completed
- `P0.4 typed per-work descriptor`: completed
- `P0.5 materialization host/layout binding`: completed
- `P0.6 projection payload seed cleanup`: completed
- `P1 runtime/codegen backend decoupling design`: completed
- `P1 SpatialPlan live/materialization refinement`: completed
- `P1 TTProgram mesh/buffer distribution schema`: completed
- `P1 compute-kind extension`: completed
- `P1 TTKernel compute_ops payload removal`: completed
- `P1 bridge attr/payload deletion`: completed
- `P1 contract-family fallback deletion`: completed
- `P1 kernel/core/compute/sync payload surface deletion`: completed
- `P1 top-level TTProgram payload deletion`: completed
- `P1 plan-local TT*Plan payload deletion`: completed
- `P1 lowering facts contract-map cleanup`: completed
- `P1 compute-op seed map cleanup`: completed
- `P1 TTProgram kernel leaf map schema cleanup`: completed
- `P1 TTProgram ABI arg/accessor map schema cleanup`: completed
- `P1 leaf reader name/default cleanup`: completed
- `P1/P2 non-GEMM exact compute op typed expansion`: completed
- `P2.1 flash-attn exact row-reduction source-live truth`: completed
- `P2.2 flash-attn bf16 direct-runtime admission`: completed
- `P2.3 seq64 exact CB-republish compile/source/spec admission`: completed
- `Blackhole tile-compute preservation`: completed
- `Post-preservation explicit tile-compute lowering split`: completed
- `Post-preservation exact tiled-CB helper split`: completed
- `Post-preservation fragment/local materialization split`: completed
- `Post-preservation ABI/accessor planning split`: completed
- `Post-preservation live-form/state bookkeeping split`: completed
- `Post-preservation staged transport split`: completed
- `Post-preservation matmul lowering split`: completed
- `Post-preservation lower_tile_op Blackhole normalizer dedup`: completed
- `Blackhole algorithmic generalization design`: completed
- `Blackhole tile compute legalizer / DAG covering design`: completed
- `Blackhole algorithmic design architecture review`: completed
- `Blackhole follow-up task queue refresh`: completed

## Current Support Boundary

- Blackhole formal execution path is the in-process `BlackholeModule`
  direct host path; legacy external runner / `build_blackhole/` are not
  supported paths.
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  wrapper/export path is restored for Blackhole.
- Direct runtime is an admitted leaf backend subset, not the codegen/export
  capability boundary.
- Current admitted direct-runtime subset:
  copy equal source/dest range with stride 1;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM accessor with `common_runtime_arg_count = 0`;
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths already covered by the typed gate.
- `TTMeshPlan` / `TTBufferDistributionPlan`
  schema exists and is validated.
  Runtime execution remains unit mesh / replicated `MeshBuffer` subset;
  full multi-device / sharded / fabric collective runtime is not admitted.
- Flash-attn compile/source/spec baseline is stable.
  The admitted bf16 direct-runtime subset covers:
  small single-work-item flash-attn,
  and 32x32 MHA / GQA.
  The path uses typed exact CB live-form,
  BF16 exact CB physical storage,
  bcast-cols row scalar semantics,
  non-mailbox `copy_tile` / `pack_tile` publication,
  per-event one-page exact CB republish,
  and ABI order `Q, K, V, Output`.
  Seq64 / multi-K-step MHA and GQA compile/source/spec lowering is stable,
  but direct-runtime correctness is not admitted; the runtime gate reports
  `multi-block exact CB-republish flash-attention direct runtime correctness`
  for that shape family.
  Larger stage2/block64 shapes that require a multi-page exact CB
  publish/consume event still fail closed with
  `multi-page exact CB-republish live-form`.

## Open Debt

- `lower_blackhole_ops.cc` is no longer the mixed owner for tile compute,
  exact CB helpers, fragment/local materialization, ABI/accessor planning,
  staged transport emission, live-form/state bookkeeping, or matmul lowering.
  It is now the pass driver plus shared CB requirement / logical-shape /
  validator / visitor orchestration surface.
- `lower_tile_op.cc` no longer has duplicate Blackhole tile compute
  normalization implementations in `LowerTileOpPass` and
  `BlackholeTileComputeNormalizer`.  Future additions to this normalization
  surface should extend the shared helper and continue emitting explicit
  `tl.tileop.blackhole_compute`, not downstream scalar-loop recovery.
- Multi-block flash-attn direct-runtime correctness needs a separate
  online-softmax live-form admission.  Do not reopen it by bypassing the
  runtime gate; admit it only after typed source-live-form and event
  lifetime evidence is verified under TT-Sim.
- Larger flash-attn stage2/block64 shapes still need a wider exact-CB
  live-form admission for single events that publish/consume multiple pages.
  That expansion must continue through typed
  `TTProgram -> ExecutableSpec` records, not runtime-only source patching.
- Full mesh/distributed runtime support remains future work.
  The current schema can express the direction; runtime admission must expand
  through typed `TTProgram -> ExecutableSpec` records, not runtime-only patching.
- The next refactor lane is now ordered by representation dependency:
  affine-lite `AccessRegion`, graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`, TT live-form solver, tile compute legalizer / DAG
  covering, and only then wider runtime admission.  These are
  design-complete but not implemented.

## Next Task Order

1. `Algorithmic generalization Phase A: AccessRegion foundation`
   - Add typed `AccessRegion` / affine-lite access analysis and validator
     coverage.
   - Keep it inside the existing
     `Normalized Tile TIR -> SpatialPlan` boundary.
   - First merge unit should be schema + builder + validator + structure
     tests only; no TTProgram or source-generation behavior change.

2. `Algorithmic generalization Phase B: graph-backed SpatialPlan dependence`
   - Build graph-backed `SpatialPlan` dependence edges from
     `AccessRegion` and tile op dataflow evidence.
   - Replace ad hoc local value-flow edge construction with graph builder
     output and SCC-backed recurrence diagnostics.

3. `Algorithmic generalization Phase C: LiveValueSSA`
   - Version logical values through `LiveValueSSA`,
     materialization boundaries, phi/join records, and source/target version
     references.
   - Delete any pass-local fallback as soon as the corresponding typed
     version field feeds downstream planning.

4. `Algorithmic generalization Phase D: TT live-form solver`
   - Add the live-form lattice solver and project solver decisions into
     `TTLiveFormPlan`, `TTMaterializationPlan`, and
     `TTConsumerBindingPlan`.
   - Replace exact-CB/source-live-form pass-local maps where the value
     survives across events or phases.

5. `Tile compute legalizer / DAG covering Phase A-B`
   - Add `TileComputeDAG` read-only dump, pattern schema, and legalizer
     diagnostics for current admitted leaf ops.
   - Keep existing source emission active until the legalizer accepts the
     same admitted compute set.

6. `Tile compute legalizer / DAG covering Phase C-E`
   - Add `TileComputeDAG`, legalization actions, TT-Metal leaf pattern
     covering, fanout/materialization-aware covering, and old branch
     deletion for current admitted compute ops.
   - Keep `TTComputeOpPlan.operation_name` at leaf API granularity.
   - Do not let covering output become a pass-to-pass DAG payload or
     source-string protocol.

7. `Multi-block flash-attn direct-runtime admission`
   - Re-admit seq64 / multi-K-step direct runtime only after the
     online-softmax live-form contract is represented and verified through
     typed `TTProgram -> ExecutableSpec` state.
   - Correctness gate remains TT-Sim bf16; compile/source/spec stability
     alone is not runtime admission.

8. `Post-P2 flash-attn wider exact-CB event support`
   - Admit larger stage2/block64 shapes only after the exact CB
     multi-page event contract is represented and validated through
     `TTProgram -> ExecutableSpec`.
   - Treat this as event-lifetime admission, not a source emitter patch.

9. `P3 mesh/distributed runtime expansion`
   - Treat this as a deferred runtime admission lane.
   - Reuse `TTMeshPlan` / `TTBufferDistributionPlan` schema.
   - Add real sharded / multi-device / fabric semantics only through typed
     schema and validator extensions.

10. Post-P2 flash-attn wider-shape support
   - Expand workload scale only after the multi-K-step and multi-page
     event contracts above are admitted.
   - Do not jump straight to 4096.  The intended admission ladder is:
     1. Same tile geometry as P2.3, more K steps:
        MHA/GQA bf16
        `(batch=1, heads=4, seq_len=128 and 256, dim=32,
          block_M=32, block_N=32, num_stages=1, threads=128)`.
     2. Grow tile footprint before full GPU parity:
        MHA bf16
        `(batch=1, heads=4, seq_len=128 and 256, dim=64,
          block_M=64, block_N=64, num_stages=1, threads=128)`,
        then GQA bf16 with matching `dim=64, block_M=64, block_N=64`.
     3. Existing CUDA regression correctness scale:
        MHA forward BSHD
        `(batch=1, heads=32, seq_len=256, dim=128,
          block_M=128, block_N=128, num_stages=1, threads=128)`
        and GQA forward BSHD
        `(batch=1, heads=16, seq_len=1024, dim=128, groups=16,
          block_M=64, block_N=64, num_stages=2, threads=128)`.
   - Stretch/perf parity remains the example default/regression scale,
     not the first post-P2 correctness gate:
     MHA forward BSHD
     `(batch=8, heads=32, seq_len=4096, dim=128,
       block_M=128, block_N=128)`
     and GQA
     `(batch=1, heads=64, seq_len=4096, dim=128, groups=16,
       block_M=128, block_N=128)`.
   - Do not treat seq64 as the wider-shape target; it is the next
     multi-K-step direct-runtime admission gate.
   - Keep helper/composite exact-op names out of
     `TTComputeOpPlan.operation_name`,
     `ExecutableSpec.compute_ops`,
     source/codegen protocol,
     and runtime support surfaces.
     The durable compute granularity remains TT-Metal builtin granularity
     (`mul_tiles`, `add_tiles`, `*_bcast_cols`, `exp2_tile`, `pack_tile`,
     etc.).

## Latest Verification

Blackhole follow-up task queue refresh:

- docs updated:
  `tasks/progress.md`,
  `tasks/dev_design/README.md`,
  `memory/general_dev.md`,
  `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md`
  and
  `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
- queue policy:
  implementation order now follows representation dependency:
  `AccessRegion -> SpatialPlan dependence graph -> LiveValueSSA ->
  TT live-form solver -> TileComputeDAG/legalizer/covering ->
  runtime admission -> mesh/distributed expansion`
- placeholder scan:
  `rg -n "TBD|TODO|fill in|later|open question|Open Question" tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
  -> no matches
- `git diff --check`
  -> clean
- background process scan:
  no lingering `pytest` / `cmake --build` / `ninja` process
