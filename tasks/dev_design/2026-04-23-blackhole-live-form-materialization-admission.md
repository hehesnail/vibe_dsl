# Blackhole Live-Form / Materialization Admission Contract

## Role

This document defines the live-form and materialization support-surface
contract for Blackhole.
It is not a status log and not a direct-runtime-only design.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

Represent and validate this question explicitly:

> What physical live form currently holds a logical value, and what
> materialization is required before a consumer may read it?

The answer must flow through:

```text
SpatialPlan logical live values
  -> TTProgram live-form / materialization / consumer-binding plans
  -> ExecutableSpec leaf records
  -> backend admission
```

## Non-Goals

- No legacy merge/live-form bridge.
- No runtime-only patch surface.
- No mailbox-backed publication as a hidden fallback.
- No source-emitter reconstruction from builtin sequences, buffer names, or
  `SeqStmt` shape.
- No direct-runtime admitted subset defining what `TTProgram` may express.

## Layer Contract

### SpatialPlan

Owns target-independent logical value relations:

- logical value identity
- producer / consumer relation
- coverage and shape evidence
- same-phase vs cross-phase relation
- materialization boundary
- whether a consumer requires a full logical value

### TTProgram

Owns TT physical live-form decisions:

- `TTLiveFormPlan`
- `TTMaterializationPlan`
- `TTConsumerBindingPlan`
- materialization protocol
- publication protocol
- required CB / sync plan references
- typed unsupported reasons

### ExecutableSpec

Owns leaf projection and backend admission:

- projected materialization records
- projected consumer bindings
- buffer / CB / accessor / runtime-arg schema
- direct-runtime unsupported reasons

It must not recover logical producer-consumer truth.

## Admitted Protocol Classes

Current protocol classes are defined by explicit typed records, not by source
text:

- direct cast consumer
- `fragment_fill -> cast -> publish`
- `pack_thread_direct_store`
- `pack_tile`
- exact CB republish when event lifetime, source live form, and consumer
  binding are all proven
- device tiled partial combine when the producer and consumer CB windows are
  typed as single-event exact-CB pages and the lowered artifact remains
  direct-runtime admitted

Adding a protocol requires:

1. logical value evidence in `SpatialPlan`
2. typed live-form / materialization / consumer-binding plans in `TTProgram`
3. executable projection
4. validator checks
5. backend admission or typed reject

## Admission Boundary

Compile/source/spec completeness and direct-runtime admission are separate.

`ExecutableSpec`
may be schema-complete and codegen/exportable even when direct runtime rejects
the case.

Direct-runtime rejection must use typed, queryable reasons such as:

- materialization protocol not admitted
- exact-CB event lifetime not admitted
- multi-block runtime path not admitted
- missing full-logical-tile proof
- simulator capability boundary

The current positive direct-runtime subset includes seq64 bf16 flash-attn MHA:
the same lowered artifact must show no exact-CB unsupported reasons,
single-page publish/consume event windows, `acc_s -> acc_s_cast`
`cb_republish` through `tilize_cast_fragment_slice`, tiled device partial
combine, and host-reference correctness.

Current standalone row `reduce_tile` is schema/source-complete through the
full-tile reduce CB to rank-1 writer path: the writer consumes the compute
published CB, writes scalar pages with tiled-row L1 offsets, and owns one
final barrier/pop. Its direct-runtime reject is therefore only the current
TT-Sim `tensix_execute_pacr: count=1` simulator capability boundary.

## Forbidden Regression

Do not reintroduce:

- logical live-form truth stored only in names or helper maps
- `direct_runtime_unsupported_reasons` as owner truth
- source hooks that choose a producer by source shape
- mailbox publication as an admitted compute-side CB protocol without
  TT-Metal linkability proof
- stale exact-output live-form aliases after tiled live-form invalidation
- late accumulator-merge discard paths that try to recover exact-CB queue
  state after a consumer has already consumed the live form

## Validation

Required validation covers:

- logical live-value references are present and consistent
- materialization plans reference valid live forms and boundaries
- consumer bindings match admitted source live forms
- publication protocols match projected CB / sync plans
- exact-CB consumers release borrowed live pages at the consumption point when
  no later consumer can legally read that live form; later producers must not
  repair stale pages with a separate semantic discard path
- row-reduce exact-CB results remain in their typed live form when the next
  consumer before the next write is compute or transport; local untilize is
  reserved for true reference consumers
- direct runtime rejects unsupported surfaces before execution

Runtime tests are support-surface gates only after the typed plans and
projection are valid.
