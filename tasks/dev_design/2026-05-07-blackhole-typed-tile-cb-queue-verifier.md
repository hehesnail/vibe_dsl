# Blackhole Typed Tile-CB Queue Verifier

## Role

This document defines the task-level design for consolidating tile-value and
CB queue correctness rules in the Blackhole `TTProgram -> ExecutableSpec`
boundary.

It is not a second overall design document and not a fifth IR layer.  The
durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status lives in `tasks/progress.md`.

## Problem

T7 seq64, T9.2 paged GQA, and T9.3 paged MLA direct-runtime correctness exposed
one shared backend issue:

- tile values have producer versions, but parts of lowering still query them
  through local buffer aliases;
- exact-CB release and physical CB reuse are partly decided by local helpers;
- physical CBs are FIFO queues, but allocation and source emission previously
  had only scattered local checks for `wait_front`, `reserve_back`, and
  `pop_front`;
- tensor dtype and CB storage dtype can differ, but not every consistency
  check is centralized.

The bug symptoms were workload-specific, but the missing invariant is generic:
once logical tile values are realized as TT CB pages, the backend must validate
latest producer identity, use/release order, physical queue visibility, and
storage format before leaf projection or runtime execution.

## Goal

Add a TTProgram-owned typed verifier for the admitted tile-CB surface:

```text
TTProgram typed tile/exact-CB records
  + physical CB plan
  + projected kernel CB event bodies
  -> typed tile-CB event trace
  -> verifier
  -> validated ExecutableSpec / typed admission reject
```

The first cutover covers the current admitted T7/T9 exact-CB and compute-local
CB queue paths.  It must keep those workloads as witnesses, not protocol
owners.

## Non-Goals

- No new `TileCBIR`, fifth representation layer, payload, or helper bag.
- No `SpatialPlan` rewrite in this task.
- No T9.4 sparse attention, T9.5 scan, T9.6 multi-block flash, or T10
  distributed expansion.
- No workload-specific verifier branches for GQA, MLA, or flash-attn.
- No source-text recovery as owner truth.  Projected leaf CB queue calls are
  represented directly in `KernelSpec.queue_events`; source text may remain
  only as a regression witness, not as an admission extractor.

## Representation Boundary

### TTProgram

Owns the facts being verified:

- `TTExactCBVirtualValue`
- `TTExactCBUseEvent`
- `TTExactCBLiveInterval`
- `TTExactCBAllocation`
- `TTExactCBReleaseEvent`
- `TTCBPlan`
- compute operand CB requirement bindings
- projected executable kernel records that contain physical leaf CB queue calls

The verifier may build pass-local indexing structures and an event trace, but
those structures are derived analysis.  They are not persisted as public
protocol unless a later task promotes them into explicit `TTProgram` fields.

### ExecutableSpec

Consumes only validated projection.

If the verifier finds a contradiction, admission must fail before runtime can
recover behavior from names, source text, or observations.  The physical queue
checker consumes `KernelSpec.queue_events`, which are projected from leaf CB
queue calls after physical CB allocation/remapping.

## Verifier Invariants

### Latest Producer

For a consumer of a logical tile value, the selected exact-CB virtual value
must be the latest dominating producer visible at that program point.

When an exact output live form and an older buffer live alias both describe the
same logical buffer, the exact output live form is the producer owner truth.
Older aliases may be used only when no newer exact output exists.

### Exact-CB Release

For every exact-CB allocation:

- the referenced virtual value, interval, and CB plan must exist;
- the release event must point back to the allocation;
- release must occur at or after the last recorded use;
- release must not occur before the allocation interval begins;
- release reason must be one of the typed lifecycle reasons admitted by the
  exact-CB design.

For loop-carried exact-CB values, live-in/live-out evidence remains required by
the existing exact-CB validator.

### Physical CB Queue

For each kernel and physical CB ID, replay the projected queue events:

- `reserve_back(cb, pages)` requires `visible_front + reserved_back + pages`
  to fit `TTCBPlan.num_pages`;
- `push_back(cb, pages)` requires a matching outstanding reservation and makes
  pages visible on the front;
- `wait_front(cb, pages)` requires visible front pages unless the CB has an
  explicit producer outside the current compute kernel, such as an input
  queue or a projected non-compute kernel `push_back`;
- `pop_front(cb, pages)` requires visible front pages and reduces visibility;
- all page counts must be positive.

This check validates logical-to-physical CB reuse as a FIFO observation
problem, not just as an ID coloring problem.

Writer-visible output CBs are not eligible for generic front-retention
rewrites.  A writer `write_tile_from_cb` consumes the current FIFO front page;
delaying the corresponding `pop_front` without a page-offset protocol makes
later writes reread the same page.  Structured writer `queue_events` must
therefore preserve output wait/pop page balance for the admitted tile writer
surface.

### Storage Format

For every exact-CB allocation and consumer binding:

- page count and page size must be positive;
- CB data format must match the virtual value data format unless the producing
  materialization explicitly records a typecast protocol;
- GEMM tensor output dtype and compute CB storage dtype remain distinct facts;
- a compute op operand bound to a CB requirement must resolve to exactly one
  physical `TTCBPlan`.

## Integration Plan

1. Add structure tests that mutate projected metadata and prove the verifier
   rejects:
   - premature exact-CB release before a later use;
   - unknown or ambiguous CB requirement bindings;
   - compute-local `wait_front` or `pop_front` beyond visible pages;
   - reserve beyond physical CB capacity;
   - exact-CB data-format mismatch.
2. Implement private C++ validation in the `TTProgram -> ExecutableSpec`
   boundary:
   - `ValidateTTProgram` validates exact-CB producer, release, storage, and
     CB-requirement ownership invariants.
   - `KernelSpec.queue_events` carries structured physical queue events for
     each projected kernel; the executable admission gate replays those records
     instead of parsing generated source.
3. Run validation after CB allocation/remapping and before runtime execution.
4. Remove or demote any existing local source checks that duplicate owner truth
   for the covered paths.  Python source checks may remain as regression
   witnesses, but they cannot be the verifier input.
5. Keep T7 seq64, T9.2 full paged GQA, and T9.3 full paged MLA as positive
   bf16 direct-runtime correctness gates.

## Validation

Required:

- C++ build: `cmake --build build -j32`.
- Focused structural pytest selectors proving typed rejects.
- Focused TT-Sim direct-runtime correctness selectors covering:
  - T7 seq64 MHA exact-CB partial combine;
  - T9 page-addressed QK page1;
  - T9 page-addressed AV page1;
  - T9.2 full paged GQA decode;
  - T9.3 dual-score MLA GEMM;
  - T9.3 full paged MLA decode;
  - T9.1 grouped GEMM.

## Completion Criteria

This task is complete only when:

- verifier failures occur before source/runtime recovery;
- current T7/T9 positive runtime gates still pass;
- the verifier covers latest-producer, release, queue, and storage-format
  invariants for the admitted surface;
- the source-regex executable queue extractor is deleted and admission consumes
  structured physical queue-event records from `KernelSpec`;
- docs, progress, and memory reflect the new boundary;
- no new side channel, payload, or workload-specific schema is introduced.
