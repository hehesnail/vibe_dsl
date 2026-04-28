# Blackhole LowerTileOp Normalizer Dedup

## Goal

Status: completed.

Before this cleanup,
`lower_tile_op.cc` had two implementations of the same
Blackhole tile-compute normalization:

- `LowerTileOpPass`
  normalizes scalar loops while generic tile ops are lowered.
- `BlackholeTileComputeNormalizer`
  normalizes the same scalar loop shapes when the standalone
  `tl.NormalizeBlackholeTileCompute` pass runs.

This task removed the duplicate implementation surface and keeps one
pass-local normalizer helper shared by both callers.

This is an implementation cleanup inside `Normalized Tile TIR`.
It does not add a new IR layer, pass protocol, or cross-stage side channel.

## Contract

- The output remains explicit `tl.tileop.blackhole_compute` calls.
- Operation names stay at TT-Metal leaf API granularity:
  `fill_tile`, `copy_tile`, `typecast_tile`, `binary_max_tile`,
  `mul_tiles`, `add_tiles`, `mul_tiles_bcast_cols`, and `exp2_tile`.
- The helper may use local structural matching over current TIR as a
  normalization mechanic, but it cannot become a downstream semantic
  recovery path.
- `LowerTileOpPass` and `BlackholeTileComputeNormalizer` may differ only
  in when they invoke the helper:
  `LowerTileOpPass` gates it on the active target,
  while `BlackholeTileComputeNormalizer` gates the whole pass on the
  function target.
- No old composite matcher/generate family is reintroduced.

## Implementation Boundary

The shared helper owns:

- construction of `tl.tileop.blackhole_compute`
- one-dimensional local/fragment store normalization
- simple nested-loop normalization for full tile regions
- local expression patterns already admitted by the preservation lane

It does not own:

- generic tile op lowering
- TT builtin selection
- `SpatialPlan`, `TTProgram`, or `ExecutableSpec` construction
- runtime/codegen support admission

## Validation

- Static regression that `lower_tile_op.cc` has a single Blackhole tile
  compute normalizer surface.
- Existing Blackhole frontend normalization tests for flash-attn leaf
  compute operations.
- `cmake --build build -j32`.
- Blackhole transform/target regression tests.
- Cleanup scan for deleted composite matcher/generate names.
