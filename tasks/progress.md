# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录当前状态、阻塞、下一步和最近验证；不写设计细节。

## Status

- Date: `2026-04-26`
- Active lane: `P2 flash-attn direct runtime admission`
- Current item: `blocked on replacing mailbox-backed local-fragment scratch staging with typed non-mailbox live-form publication`
- Blocker: `flash-attn compute source still emits mailbox-backed tilelang_get_cb_write_ptr_bytes / CircularBuffer::get_tile_address for local-fragment <-> CB scratch staging; gate-bypass probe reaches TT-Sim t_tile_mmio_wr32 before correctness`
- Main chain: `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Priority

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
- `P2 flash-attn direct runtime admission`: blocked

## Open Debt

- Direct runtime remains an admitted leaf backend subset, not the codegen/export capability boundary.
- Flash-attn compile/source/spec baseline is stable; direct runtime correctness is not admitted.
- `cast_fragment_slice_to_tiled_cb` is now explicit typed materialization metadata, but remains a
  direct-runtime unsupported gate until a non-mailbox, TT compute-linkable CB publication path exists.
- The current blocker is wider than the single `acc_s -> acc_s_cast` materialization plan:
  exact-op scratch staging still uses local fragment to CB and CB to local transfers backed by
  mailbox address exchange. Admission requires typed internal staging/live-form plans plus a
  PACK/DST-linkable publication path, not a runtime gate relaxation.

## Latest Verification

P2 flash-attn admission probe:

- `cmake --build build -j32`
- small bf16 MHA metadata probe: unsupported reason remains queryable for
  `thread-distributed cb_republish materialization`
- temporary gate-bypass probe: internal `acc_s_cast` materialization has empty
  `host_buffer`; after bypassing the host-buffer check for probe only, TT-Sim
  fails at `UnimplementedFunctionality: t_tile_mmio_wr32`
- probe changes were reverted and `cmake --build build -j32` rebuilt the clean gated path

P2 flash-attn typed materialization gate restore:

- `cmake --build build -j32`
- flash-attn typed gate regression: `8 passed`
- flash-attn source/codegen targeted regression: `5 passed`
- TT-Sim flash-attn gate check: `1 passed, 1 skipped`
- flash-attn runtime metadata file under TT-Sim env: `9 passed, 5 skipped`
- flash-attn pipeline: `66 passed`
