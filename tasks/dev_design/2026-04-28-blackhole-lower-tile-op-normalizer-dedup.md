# Blackhole LowerTileOp Normalizer Boundary

## Role

This document defines the Blackhole tile-compute normalization boundary inside
`Normalized Tile TIR`.
It is a task-level implementation contract, not a history log.

## Goal

Keep one shared Blackhole tile-compute normalizer used by:

- `LowerTileOpPass`
- standalone `tl.NormalizeBlackholeTileCompute`

The helper normalizes current TIR into explicit
`tl.tileop.blackhole_compute`
leaf statements.
It does not create a new IR layer or cross-stage protocol.

Implementation boundary:
the Blackhole-specific normalizer lives outside
`lower_tile_op.cc`.
`LowerTileOpPass`
may call a narrow normalization function,
but `lower_tile_op.cc`
must not own Blackhole leaf-call builders,
logical-temp construction,
or composite-expression decomposition.

## Contract

- Output is explicit tile-compute TIR.
- One emitted call represents one TT-Metal semantic leaf op.
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
- Operation-changing `mode` /
  `kind`
  payloads are forbidden.

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
- row-broadcast division becomes
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
  calls the normalizer but does not define it
- frontend normalization emits only admitted leaf operation names
- composite payload strings are absent from leaf-looking calls
- row-division normalization produces `recip_tile`
- Blackhole transform tests cover the normalized source surface
