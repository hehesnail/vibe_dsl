# Blackhole Tile Compute Preservation Contract

## Role

This document defines how Blackhole tile-compute semantics must be preserved
or normalized in `Normalized Tile TIR`.
It is a task-level contract, not a completion log.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

Blackhole compute semantics must remain at TT-Metal tile API granularity
before `SpatialPlan`, `TTProgram`, and `ExecutableSpec` consume them.

The backend must not destroy tile compute semantics through generic scalar
lowering and then recover them later through workload-shaped matchers.

## Leaf Granularity

Production compute op names are TT-Metal semantic leaf ops, for example:

- `matmul_tiles`
- `reduce_tile`
- `fill_tile`
- `copy_tile`
- `pack_tile`
- `typecast_tile`
- `binary_max_tile`
- `add_tiles`
- `mul_tiles`
- `add_tiles_bcast_cols`
- `mul_tiles_bcast_cols`
- `exp2_tile`
- `recip_tile`

The admitted set may grow with TT-Metal leaf API coverage.
It must not grow by workload names.

## Forbidden Operation Names

Composite or workload-shaped names are not production compute op names:

- `softmax`
- `exp2_affine`
- `row_broadcast_exp2_affine`
- `scalar_exp2_affine`
- `row_reduction`
- GEMM epilogue helper names

They cannot enter:

- `TTComputeOpPlan.operation_name`
- `KernelSpec.compute_ops`
- source/codegen leaf protocol
- runtime admission surface

## Composite Expression Rule

If a current TIR expression is expressible by multiple TT-Metal leaf ops,
the output must be multiple explicit leaf statements in
`Normalized Tile TIR`.

Forbidden leaf-looking payloads:

- `exp2_tile(mode, lhs, rhs, scale, ...)`
- `mul_tiles_bcast_cols("div", ...)`

Required shape:

- `exp2(lhs * s0 - rhs * s1)`
  becomes explicit
  `copy_tile` /
  `fill_tile` /
  `mul_tiles` /
  `add_tiles` or `add_tiles_bcast_cols` /
  `exp2_tile`
- division by a scalar-load operand becomes
  `recip_tile`
  plus
  `mul_tiles_bcast_cols`

Logical temps may be introduced in `Normalized Tile TIR`.
Physical CB / tile-register / publication decisions remain live-form and
source-emission concerns.

## Layer Contract

### Normalized Tile TIR

Owns tile compute preservation and explicit leaf normalization.

### SpatialPlan

Observes execution-unit, dataflow, live-value, and materialization-boundary
relations.
It does not own TT leaf names.

### TTProgram

Selects explicit tile semantics into typed TT leaf plans.

### ExecutableSpec

Consumes projected leaf records.
It does not recover compute semantics.

## Validation

Required checks:

- production compute op names are leaf names
- composite helper names are absent from typed compute plans
- leaf-looking calls do not carry operation-changing mode/kind strings
- validators reject plan/source mismatches
- source/runtime consumers do not contain scalar-loop recovery paths

Runtime tests are required only when the changed surface reaches admitted
backend execution.
