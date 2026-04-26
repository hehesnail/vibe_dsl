# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 不写长期设计细节。

## Status

- Date: `2026-04-26`
- Active lane: `P2 flash-attn direct runtime admission`
- Current item:
  `P2.1 complete source-live-form truth for flash-attn exact row-reduction inputs`
- Blocker:
  targeted small bf16 flash-attn source no longer emits mailbox-backed
  `tilelang_get_cb_write_ptr_bytes` /
  `CircularBuffer::get_tile_address`,
  but direct-runtime admission remains gated.
  A gate-open TT-Sim probe reaches execution and fails at
  `UnsupportedFunctionality: tensix_execute_gmpool: src_b_val=0x0 must be 1.0f`
  because the first exact row-reduction still publishes a synthetic zero tile
  instead of consuming the upstream matmul-produced CB-live source.
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
- Flash-attn compile/source/spec baseline is stable;
  flash-attn direct runtime correctness is not admitted.

## Open Debt

- `P2 flash-attn direct runtime admission`: blocked.
- `cast_fragment_slice_to_tiled_cb`
  has explicit typed materialization metadata and a non-mailbox generated
  publication shape, but remains outside the direct-runtime admitted set until
  source-live-form truth is complete enough for correctness.
- The current blocker is no longer the single
  `acc_s -> acc_s_cast` materialization plan or mailbox address exchange.
  Admission now requires the exact row-reduction source to consume the upstream
  matmul CB-live value through typed live-form state instead of a stale
  fragment-fill fallback.
- Full mesh/distributed runtime support remains future work.
  The current schema can express the direction; runtime admission must expand
  through typed `TTProgram -> ExecutableSpec` records, not runtime-only patching.

## Next Task Order

1. `P2.1 source-live-form truth for exact row-reduction inputs`
   - Extend the current `SpatialPlan` live-value /
     materialization-boundary representation as needed so the row-reduction
     source is tied to the upstream matmul-produced logical value.
   - Carry that truth through `TTProgram` physical live-form /
     consumer-binding records and `ExecutableSpec` materialization records.
   - Keep runtime/codegen as leaf readers only; no name matching,
     builtin-sequence recovery, or workload-only path.
   - Done when the targeted small bf16 flash-attn source no longer uses stale
     synthetic fill as the first row-reduction source, and the existing
     metadata/source/gate tests still pass.

2. `P2.2 flash-attn bf16 direct-runtime admission`
   - Re-open the current fail-closed direct-runtime gate only for the proven
     typed subset.
   - Run the formal TT-Sim bf16 correctness gate for the admitted subset.
   - Keep unsupported reasons queryable for live-in / materialization shapes
     still outside the admitted subset.

3. `P2.3 live-form support-surface expansion`
   - Generalize beyond the first flash-attn shape only after the source-live
     and materialization contracts are explicit and validated.
   - Add regression tests at projection, source/codegen, and direct-runtime
     gate boundaries.

4. `P3 mesh/distributed runtime expansion`
   - Treat this as a later runtime admission lane.
   - Reuse `TTMeshPlan` / `TTBufferDistributionPlan` schema.
   - Add real sharded / multi-device / fabric semantics only through typed
     schema and validator extensions.

## Latest Verification

P2 flash-attn admission probe:

- `cmake --build build -j32`
- small bf16 MHA metadata probe: unsupported reason remains queryable for
  `thread-distributed cb_republish materialization`
- temporary gate-bypass probe: internal `acc_s_cast` materialization has empty
  `host_buffer`; after bypassing the host-buffer check for probe only, TT-Sim
  fails at `UnimplementedFunctionality: t_tile_mmio_wr32`
- probe changes were reverted and `cmake --build build -j32` rebuilt the clean
  gated path

P2 flash-attn typed materialization gate restore:

- `cmake --build build -j32`
- flash-attn typed gate regression: `8 passed`
- flash-attn source/codegen targeted regression: `5 passed`
- TT-Sim flash-attn gate check: `1 passed, 1 skipped`
- flash-attn runtime metadata file under TT-Sim env: `9 passed, 5 skipped`
- flash-attn pipeline: `66 passed`

P2 flash-attn non-mailbox publication checkpoint:

- `cmake --build build -j32`
- targeted flash-attn metadata/source/direct-runtime gate:
  `7 passed, 1 skipped`
- full flash-attn runtime metadata file under TT-Sim env:
  `10 passed, 5 skipped`
- exploratory gate-open probe reached TT-Sim execution and failed at
  `UnsupportedFunctionality: tensix_execute_gmpool: src_b_val=0x0 must be 1.0f`;
  source inspection showed the first exact row-reduction still materializes its
  source from a synthetic zero fill rather than the upstream matmul CB-live
  value. The direct runtime gate is therefore intentionally still closed.
