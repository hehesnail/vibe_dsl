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

T6 is not a request to add a frontend `topk` operator or a Blackhole-private
selection semantic object.  The compiler must keep the authored computation as
ordinary Tile TIR and lower that structure through the normal
`Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec` chain.

The owner truth is:

- existing normalized Tile TIR structure for the authored selection
  algorithm: value reductions, typed local/index buffers, comparisons,
  predicates, masks, loops, and global value/index stores;
- `SpatialPlan` live values and layout records for the input, value output,
  and index output;
- `TTProgram` leaf/source/spec records for the selected implementation,
  including value/index ABI and buffer bindings;
- `ExecutableSpec` records consumed by codegen and `BlackholeModule`.

T6 does not introduce a frontend `T.topk` API, a Blackhole-specific
`tl.blackhole.topk` op, a top-level `TTSelectionPlan`, or
`ExecutableSpec.selection_plans`.
The frontend input shape for this task is the existing TileLang style used by
`examples/topk/example_topk.py`: normal TileLang/TIR loops and tile operators
such as `T.copy`, `T.fill`, `T.reduce_max`, `T.if_then_else`, local fragment /
index buffers, and explicit value/index output stores.

The compiler may admit this structure only from IR structure, types, regions,
and dataflow.  It must not rely on buffer names, output argument positions,
source text, or a new Blackhole-only pseudo operator.

The active runtime question is therefore a value/index lowering question, not
a frontend topk-op question.  Flash attention already proves that floating row
`reduce_tile` can be admitted inside the existing Blackhole direct path when it
is part of the GEMM/softmax compute chain, and standalone bf16 row reduce is
also admissible once the emitted scaler fill/pack sequence initializes
PACK/UNPACK format state correctly.

The T6 implementation admits the existing TIR value/index selection shape by
recognizing the typed value reduce plus typed `int32` index reduce records and
emitting one backend value/index scan over the reader-materialized input CB.
That scan publishes the value and index output CB pages consumed by the normal
writer path.  This replaces the unsupported standalone
`Int32 reduce_tile<MAX, REDUCE_ROW>` execution shape without adding a frontend
topk op, `TTSelectionPlan`, `selection_plans`, or a source-name side channel.

## Contract

An admitted value/index selection TIR form must explicitly represent:

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
- No frontend or Blackhole-private topk pseudo-op.
- No top-level selection plan family parallel to `SpatialPlan`,
  `TTProgram`, or `ExecutableSpec`.
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

## Current Implementation Notes

- The admitted positive subset is row-wise value/index selection from the
  existing Tile TIR structure, with fp32 or bf16 values and exact `int32`
  indices.
- Compute consumes the input through the typed reader CB materialization; it
  does not read raw host argument pointers.
- Tie behavior matches the authored TIR: the mask/index reduce shape selects
  the highest column index for equal values.
- Output publication follows the existing writer event protocol.  bf16 value
  pages require the writer's 16-row event grouping, while fp32 uses the
  32-row grouping.
- Small scalar output-page writes stage through TT-Metal's per-NOC inline L1
  scratch before `noc_async_write`; source-CB tail scratch and stack scratch
  are not valid for this path.
