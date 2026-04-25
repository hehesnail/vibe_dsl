# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录当前状态、阻塞、下一步和最近验证；不写设计细节。

## Status

- Date: `2026-04-25`
- Active lane: `P1 redundant protocol cleanup`
- Current item: `P1 remaining protocol-residue cleanup`
- Blocker: none
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
- `P1 lowering facts contract-map cleanup`: next
- `P1 leaf reader name/default cleanup`: next
- `P1/P2 workload payoff`: queued

## Open Debt

- Direct runtime remains an admitted leaf backend subset, not the codegen/export capability boundary.
- Non-GEMM exact compute builtins compile through kernel source; per-op typed expansion is next payoff work.
- Remaining analysis-fact maps and leaf name/default readers should shrink into typed fields.
- Flash-attn compile/source/spec baseline is stable; direct runtime correctness is not admitted.

## Latest Verification

P1 redundant protocol cleanup:

- `cmake --build build -j32`
- Blackhole schema/pipeline sweep: `142 passed, 10 skipped, 1 xfailed, 4 warnings`
- TT-Sim copy runtime: `13 passed`
- TT-Sim GEMM: `59 passed`
