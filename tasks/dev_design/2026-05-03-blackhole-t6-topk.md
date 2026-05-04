# Blackhole T6 topk Design

## Role

This document defines the active task-level boundary for T6 `topk`.
It is not a second overall design document.  The durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status remains in `tasks/progress.md`.

## Goal

T6 admits standalone value/index selection as a real Blackhole direct-runtime
path.  The result must be a typed compiler contract, not source-name recovery
or an external runner path.

The owner truth is:

- normalized Tile TIR structure for the selection operation, axis, `k`, input
  dtype, output value dtype, and index dtype;
- `SpatialPlan` live values and layout records for the input, value output,
  and index output;
- `TTProgram` leaf/source/spec records for the selected implementation,
  including value/index ABI and buffer bindings;
- `ExecutableSpec` records consumed by codegen and `BlackholeModule`.

## Contract

An admitted `topk` form must explicitly represent:

- input tensor identity, shape, dtype, selected axis, and `k`;
- value output identity, shape, dtype, and memory placement;
- index output identity, shape, `int32` dtype, and memory placement;
- ordering/tie behavior for the admitted subset, or a typed unsupported
  reason when deterministic behavior is not represented;
- source/spec records sufficient for direct runtime to execute without
  argument-order, buffer-name, or source-text inference.

Unsupported axis, dtype, shape, placement, tie-breaking, or index-layout
forms must fail before source/runtime guessing.

## Non-Goals

- No fused MoE, grouped/ragged, paged, or distributed `topk` claim.
- No implicit index dtype conversion outside a typed output contract.
- No source-name or output-position matching to recover value/index roles.
- No legacy external runner path.

## Validation

T6 validation must cover:

- projection tests proving value and index outputs are represented in typed
  IR/source/spec records;
- direct-runtime correctness for standalone row-wise fp32 `topk` values with
  exact `int32` indices, using a multi-work shape such as `M=320`, `N=128`,
  `k=6`, `axis=1`, and `blk_m=64`, with non-tie input data unless deterministic
  tie behavior is explicitly represented;
- direct-runtime correctness for the admitted bf16 value surface with exact
  `int32` indices and `M > blk_m`, comparing values with bf16-appropriate
  tolerance and indices exactly;
- typed rejects for unsupported axes, dtypes, placements, tie behavior, and
  layout combinations;
- source/spec tests proving the generated path consumes the typed records.

The positive runtime cases must execute through `BlackholeModule` with the
repository TT-Sim setup.  Validator-only, source-only, or schema-only coverage
cannot complete T6, and negative typed rejects cannot substitute for admitted
value/index correctness.
