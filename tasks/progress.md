# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录当前状态、阻塞、下一步和最近验证；不写设计细节。

## Status

- Date: `2026-04-25`
- Active lane: `P1 compatibility fallback deletion`
- Current item: `P1/P2 workload payoff`
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
- `P1/P2 workload payoff`: next

## Open Debt

- Direct runtime remains an admitted leaf backend subset, not the codegen/export capability boundary.
- Non-GEMM exact compute builtins compile through kernel source; per-op typed expansion is next payoff work.
- Flash-attn compile/source/spec baseline is stable; direct runtime correctness is not admitted.

## Latest Verification

P1 compatibility fallback deletion:

- `cmake --build build -j32`
- Blackhole schema/runtime/pipeline sweep: `222 passed, 1 skipped, 1 xfailed, 4 warnings`
