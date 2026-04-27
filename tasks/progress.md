# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、阻塞、下一步和最近验证；
> 不写长期设计细节。

## Status

- Date: `2026-04-27`
- Active lane: `P2 flash-attn direct runtime admission`
- Current item:
  `P2 complete; next lane is P3 mesh/distributed runtime expansion`
- Blocker:
  No P2 blocker remains for the admitted bf16 flash-attn direct-runtime
  surface.  Larger stage2/block64 flash-attn shapes that require a
  multi-page publish/consume event remain an explicit post-P2 support-surface
  backlog, not a seq64 P2.3 blocker.
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
- `P2.3 seq64 exact CB-republish live-form admission`: completed

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
  The admitted bf16 direct-runtime subset now covers:
  small single-work-item flash-attn,
  32x32 MHA / GQA,
  and seq64 / multi-K-step MHA / GQA shapes.
  The path uses typed exact CB live-form,
  BF16 exact CB physical storage,
  bcast-cols row scalar semantics,
  non-mailbox `copy_tile` / `pack_tile` publication,
  per-event one-page exact CB republish,
  and ABI order `Q, K, V, Output`.
  Larger stage2/block64 shapes that require a multi-page exact CB
  publish/consume event still fail closed with
  `multi-page exact CB-republish live-form`.

## Open Debt

- Larger flash-attn stage2/block64 shapes still need a wider exact-CB
  live-form admission for single events that publish/consume multiple pages.
  That expansion must continue through typed
  `TTProgram -> ExecutableSpec` records, not runtime-only source patching.
- Full mesh/distributed runtime support remains future work.
  The current schema can express the direction; runtime admission must expand
  through typed `TTProgram -> ExecutableSpec` records, not runtime-only patching.

## Next Task Order

1. `P3 mesh/distributed runtime expansion`
   - Treat this as a later runtime admission lane.
   - Reuse `TTMeshPlan` / `TTBufferDistributionPlan` schema.
   - Add real sharded / multi-device / fabric semantics only through typed
     schema and validator extensions.

2. Post-P2 flash-attn wider-shape support
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
   - Do not treat seq64 as the wider-shape target; it is only the P2.3
     multi-K-step admission smoke gate.
   - Keep helper/composite exact-op names out of
     `TTComputeOpPlan.operation_name`,
     `ExecutableSpec.compute_ops`,
     source/codegen protocol,
     and runtime support surfaces.
     The durable compute granularity remains TT-Metal builtin granularity
     (`mul_tiles`, `add_tiles`, `*_bcast_cols`, `exp2_tile`, `pack_tile`,
     etc.).

## Latest Verification

P2.3 flash-attn multi-K exact CB-republish closeout:

- `cmake --build build -j32`
- flash-attn pipeline:
  `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  -> `67 passed`
- flash-attn runtime metadata/source/direct-runtime file under TT-Sim env:
  `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  -> `20 passed`
- copy direct-runtime file under TT-Sim env:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  -> `13 passed`
- GEMM direct-runtime guard cases under TT-Sim env:
  richer compute config,
  clear-accum-false cast consumer,
  fragment-fill cast publish,
  transpose-A typed compute schema
  -> `4 passed`
- Production source cleanup scan:
  no `tl.blackhole.*` helper/composite builtin names,
  no `HelperCompositeBlackholeBuiltin`,
  no debug pack hang hooks,
  and no `live_reload_cast` residue under `tilelang_repo/src`.

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

P2.1 flash-attn exact row-reduction source-live truth:

- `cmake --build build -j32`
- new regression:
  `pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py::test_blackhole_flash_attention_first_row_reduction_consumes_matmul_live_form -q`
  -> `1 passed`
- full P2.1 runtime metadata/source/gate file:
  `pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -q`
  -> `11 passed, 5 skipped`
- exploratory full flash-attn pipeline suite was not used as the P2.1 gate:
  it currently contains pre-non-mailbox source expectations such as requiring
  `tilelang_get_cb_write_ptr_bytes` to appear, while the current P2 source gate
  requires that mailbox path to be absent. Representative large-shape pipeline
  tests also still fail on the existing optimized-path L1/admission surface.
  Those belong to later pipeline cleanup / `P2.3` support-surface expansion,
  not to the completed small bf16 row-reduction source-live fix.

P2.2 flash-attn bf16 direct-runtime admission:

- `cmake --build build -j32`
- flash-attn runtime metadata/source/direct-runtime file under TT-Sim env:
  `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  -> `14 passed, 5 skipped`
- flash-attn pipeline file under TT-Sim env:
  `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  -> `66 passed`
- GEMM direct-runtime guard cases under TT-Sim env:
  richer compute config, clear-accum-false cast consumer,
  fragment-fill cast publish, transpose-A/B typed compute schema
  -> `4 passed`
- copy direct-runtime file under TT-Sim env:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  -> `13 passed`
- The admitted subset is bf16 single-page exact CB-republish.
  Seq64 / multi-K-step cases remain intentionally skipped by queryable
  `multi-page exact CB-republish live-form` unsupported reasons.
