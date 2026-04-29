# Blackhole LowerTileOp Normalizer Dedup

## Goal

Status: completed as a deduplication cleanup,
but reopened for boundary correction after the
`2026-04-29`
tile-compute review.

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
The helper is allowed to normalize TIR expressions into explicit leaf
tile-compute TIR statements.
It is not allowed to encode a composite expression as a single
leaf-looking
`tl.tileop.blackhole_compute`
payload and leave the real decomposition to
`PlanTTKernelABI`
or `TileComputeDAG`.

## Contract

- The output remains explicit `tl.tileop.blackhole_compute` calls,
  one call per TT-Metal semantic leaf op.
- Operation names stay at TT-Metal leaf API granularity:
  `fill_tile`, `copy_tile`, `typecast_tile`, `binary_max_tile`,
  `mul_tiles`, `add_tiles`, `mul_tiles_bcast_cols`, and `exp2_tile`.
- Leaf source-call schemas must expose the real leaf operands.
  Binary leaf calls use explicit
  `lhs`,
  `rhs`,
  and
  `output`
  roles;
  unary leaf calls use explicit
  `input`
  and
  `output`
  roles.
  A payload field such as
  `mode`
  or
  `kind`
  may select a leaf API variant only if it does not change the semantic op
  family or cause multi-op expansion.
- The helper may use local structural matching over current TIR as a
  normalization mechanic, but it cannot become a downstream semantic
  recovery path.
  For expressions such as
  `exp2(lhs * s0 - rhs * s1)`
  or row-broadcast division,
  the helper must emit an explicit sequence of leaf TIR statements
  with logical temps when required.
  It must not emit
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  or
  `mul_tiles_bcast_cols("div", ...)`.
- `LowerTileOpPass` and `BlackholeTileComputeNormalizer` may differ only
  in when they invoke the helper:
  `LowerTileOpPass` gates it on the active target,
  while `BlackholeTileComputeNormalizer` gates the whole pass on the
  function target.
- No old composite matcher/generate family is reintroduced.
- Copy insertion is not a default decomposition step.
  Semantic copy belongs in explicit TIR when the source program asks for it;
  preservation copies must be justified by def-use /
  liveness;
  physical form copies belong to live-form /
  materialization planning.

## Implementation Boundary

The shared helper owns:

- construction of `tl.tileop.blackhole_compute`
- one-dimensional local/fragment store normalization
- simple nested-loop normalization for full tile regions
- local expression patterns already admitted by the preservation lane
- bottom-up expression decomposition into explicit leaf tile-compute
  statements when the TT-Metal leaf set can express the value

It does not own:

- generic tile op lowering
- TT builtin selection
- `SpatialPlan`, `TTProgram`, or `ExecutableSpec` construction
- source emission
- CB allocation,
  tile-register protocol,
  or physical materialization
- runtime/codegen support admission

## Boundary Correction (`2026-04-29`)

The previous implementation accidentally treated the current in-place
binary shorthand
(`dst = op(dst, rhs)`)
as if it were the semantic leaf contract.
That is wrong.
The logical leaf contract is three-address for binary ops
and two-address for unary ops;
the source emitter may choose an in-place physical realization later,
but the TIR leaf statement must not collapse input and output identities
unless the source semantics actually do so.

The repair task for this document is therefore:

- remove composite `exp2_tile` payload normalization
- remove `mul_tiles_bcast_cols("div", ...)`
- make admitted composite TIR expressions lower to explicit leaf TIR
  sequences with logical temps
- make any unsupported diagnostic use the correct category:
  `lowering_missing`,
  `backend_op_missing`,
  `admission_blocked`,
  or true semantic unsupported after TT-Metal primitive coverage audit

## Validation

- Static regression that `lower_tile_op.cc` has a single Blackhole tile
  compute normalizer surface.
- Existing Blackhole frontend normalization tests for flash-attn leaf
  compute operations.
- `cmake --build build -j32`.
- Blackhole transform/target regression tests.
- Cleanup scan for deleted composite matcher/generate names.
