# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 不写长期设计细节。

## Status

- Date: `2026-04-27`
- Active lane: `Blackhole tile-compute preservation design`
- Current item:
  `2026-04-27 Blackhole tile-compute preservation implemented; active code path uses explicit tile compute semantics and TT-Metal leaf compute ops`
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

- `lower_blackhole_ops.cc` still owns several adjacent responsibilities
  (tile compute selection, exact-CB materialization planning, ABI planning,
  and source-emission support).  The active scalar-loop compute recovery
  families are deleted, but file split / responsibility shrink remains
  future cleanup.
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

## Next Task Order

1. `Post-preservation file split / responsibility shrink`
   - Split the remaining `lower_blackhole_ops.cc` responsibilities around
     explicit tile compute selection, exact-CB materialization planning,
     ABI planning, and source-emission support.
   - Keep `TTComputeOpPlan.operation_name` and `KernelSpec.compute_ops`
     at TT-Metal leaf API granularity.

2. `Multi-block flash-attn direct-runtime admission`
   - Re-admit seq64 / multi-K-step direct runtime only after the
     online-softmax live-form contract is represented and verified through
     typed `TTProgram -> ExecutableSpec` state.

3. `P3 mesh/distributed runtime expansion`
   - Treat this as a later runtime admission lane.
   - Reuse `TTMeshPlan` / `TTBufferDistributionPlan` schema.
   - Add real sharded / multi-device / fabric semantics only through typed
     schema and validator extensions.

4. Post-P2 flash-attn wider-shape support
   - Admit larger stage2/block64 shapes only after the exact CB
     multi-page event contract is represented and validated through
     `TTProgram -> ExecutableSpec`.
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

Blackhole tile-compute preservation:

- `cmake --build build -j32`
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_copy_pipeline.py testing/python/target/blackhole/test_blackhole_gemm.py`
  -> `197 passed, 25 skipped, 1 xfailed`
- TT-Sim:
  `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  -> `15 passed, 5 skipped`
- TT-Sim:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  -> `13 passed`
- cleanup scan:
  `rg -n "GenerateScalar|GenerateRowBroadcast|GenerateExp2RowBroadcast|GenerateExplicit|ScalarFma|ScalarExp2|Exp2RowBroadcast|RowBroadcastMatch|ScalarMaxMatch|ScalarFragmentCopyMatch|FragmentFillMatch|MatchScalar|MatchGrouped|MatchDirectRow|MatchDirectFragmentFill|MatchScalarFragmentFillStore|scalar_exp2|scalar_fma|exp2_affine|row_bcast|row_broadcast_affine|scalar_affine|RejectLegacyScalar" tilelang_repo -S`
  -> no matches
