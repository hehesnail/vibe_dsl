# Blackhole LowerTileOp Normalizer Boundary

## Role

This document defines the Blackhole tile-compute normalization boundary inside
`Normalized Tile TIR`.
It is a task-level implementation contract, not a history log.

## Goal

Blackhole tile-compute normalization is a `Normalized Tile TIR` admission
step that runs before generic `LowerTileOp`.

The normalizer rewrites admitted frontend fragment/local compute loops into
explicit
`tl.tileop.blackhole_compute`
leaf statements.
It does not create a new IR layer or cross-stage protocol.

Implementation boundary:
the Blackhole-specific normalizer lives outside
`lower_tile_op.cc`.
`LowerTileOpPass`
must preserve existing
`tl.tileop.blackhole_compute`
leaf calls for Blackhole and must not call scalar-loop normalization helpers.
After the normalizer runs, any remaining admitted compute-buffer scalar loop
that is not represented as a tile op / explicit Blackhole leaf call is rejected
before generic `LowerTileOp` can lower it into target-ambiguous scalar code.

## Contract

- Output is explicit tile-compute TIR before `LowerTileOp`.
- One emitted call represents one TT-Metal semantic leaf op.
- The implementation is a bounded normalizer, not a pattern engine.
  It may dispatch on the current TIR store value root and use local helper
  predicates, but it must not grow an open-ended rule registry, benefit table,
  or workload-pattern catalog.
- New workload coverage should normally preserve or introduce explicit
  leaf-level TIR earlier.
  Adding another scalarized source idiom to this normalizer is only valid when
  the source idiom is already part of the admitted scalar-loop residue and can
  be lowered immediately to TT-Metal leaves.
- A local normalization result may carry operands, scalar parameters, logical
  temp requests, and the ordered leaf calls to render.
  It must not carry production operation names such as
  `exp2_affine`,
  `row_div`,
  `softmax`,
  or any other composite semantic family.
- Operation names stay at TT-Metal leaf API granularity:
  `fill_tile`,
  `copy_tile`,
  `typecast_tile`,
  `binary_max_tile`,
  `add_tiles`,
  `mul_tiles`,
  `add_tiles_bcast_cols`,
  `mul_tiles_bcast_cols`,
  `exp2_tile`,
  `recip_tile`,
  `reduce_tile`,
  `pack_tile`.
- Binary leaf calls expose
  `lhs`,
  `rhs`,
  and
  `output`
  roles.
- Unary leaf calls expose
  `input`
  and
  `output`
  roles.
- Same-shaped leaf calls should be built by shared unary / binary helpers.
  The normalizer must not carry one hand-written call builder per builtin
  when only the operation name differs.
- Operation-changing `mode` /
  `kind`
  payloads are forbidden.
- `LowerTileOpPass`
  is not a second normalization site.
  Its Blackhole responsibility is to keep
  `tl.gemm_py`,
  `tl.tileop.reduce`,
  and already-normalized
  `tl.tileop.blackhole_compute`
  calls visible to the Blackhole IR chain.
- A validation pass must run after normalization and before `LowerTileOp`.
  It rejects residual fragment/local compute loops whose destination is a
  Blackhole compute buffer and whose value still depends on compute-buffer
  loads or literal compute values.

## Composite Expression Rule

If the current TIR contains a composite expression expressible by admitted
TT-Metal leaf ops,
the normalizer must emit an explicit leaf statement sequence before DAG
construction.

Forbidden outputs:

- `exp2_tile(mode, lhs, rhs, scale, ...)`
- `mul_tiles_bcast_cols("div", ...)`

Required examples:

- `exp2(lhs * s0 - rhs * s1)`
  becomes explicit
  `copy_tile` /
  `fill_tile` /
  `mul_tiles` /
  `add_tiles` or `add_tiles_bcast_cols` /
  `exp2_tile`
  leaf statements with logical temps as needed.
- division by a scalar-load operand becomes
  `recip_tile`
  plus
  `mul_tiles_bcast_cols`.

## Copy / Materialization Boundary

The normalizer may introduce logical temps when required to express the
current TIR value as leaf statements.

It must not introduce physical-form copies as a workaround.
Physical form and event-lifetime materialization belong to live-form /
materialization planning.

## Non-Goals

- No source emission.
- No CB allocation.
- No tile-register protocol.
- No runtime admission.
- No downstream semantic recovery.
- No source-hook expression decomposition.

## Validation

Required checks:

- there is a single shared Blackhole tile-compute normalizer surface
- `lower_tile_op.cc`
  does not call the normalizer and does not define it
- `LowerAndLegalize`
  runs
  `NormalizeBlackholeTileCompute`
  and
  `ValidateBlackholeTileComputeNormalized`
  before
  `LowerTileOp`
- `LowerTileOp`
  preserves existing
  `tl.tileop.blackhole_compute`
  calls for Blackhole
- the normalizer has one loop-normalization implementation surface,
  without a pure forwarding `TryNormalizeBlackholeTileComputeLoop`
  wrapper
- the normalizer has one store-normalization dispatch,
  no open-ended rule registry / benefit table,
  and no large `TryNormalizeBlackholeTileComputeStore`
  compatibility shell
- frontend normalization emits only admitted leaf operation names
- composite payload strings are absent from leaf-looking calls
- division-by-scalar-load normalization produces `recip_tile`
- residual unsupported compute loops fail before `LowerTileOp`
- Blackhole transform tests cover the normalized source surface
