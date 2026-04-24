# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录当前状态、阻塞、下一步和最近验证；不写设计细节。

## Status

- Date: `2026-04-24`
- Active lane: `SpatialPlan live/materialization refinement`
- Current item: `P1 SpatialPlan live/materialization refinement`
- Blocker: none
- Main chain: `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Priority

- `P0.1 contract-family public surface`: completed
- `P0.2 compute operand binding`: completed
- `P0.3 host wrapper / codegen buffer binding`: completed
- `P0.4 typed per-work descriptor`: completed
- `P0.5 materialization host/layout binding`: completed
- `P0.6 projection payload seed cleanup`: completed
- `P1 SpatialPlan live/materialization refinement`: next
- `P1 compute-kind extension`: after P0
- `P1/P2 workload payoff`: after P0/P1 gates

## Open Debt

- `tl.blackhole_logical_buffer_tile_bridge_specs` remains the only narrow bridge attr.
- Flash-attn compile/source/spec baseline is stable; direct runtime correctness is not admitted.

## Latest Verification

P0.6 projection payload seed cleanup:

- `cmake --build build -j32`
- P0.6 projection poison regression: `1 passed`
- copy/GEMM/flash pipeline: `158 passed, 26 skipped, 1 xfailed`
- spatial/export: `21 passed`
- copy/flash runtime under TT-Sim: `22 passed, 5 skipped`
- TT-Sim selected direct-runtime cases: `5 passed`
