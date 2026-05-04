# Blackhole Exact-CB Liveness And Allocation Design

## Role

This document defines the task-level design for exact-CB lifecycle analysis
and CB resource allocation in the Blackhole backend.

It is not a second overall design document.
The durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status lives in `tasks/progress.md`.

## Problem

T7 proved the first exact-CB materialization surface for seq64 bf16
flash-attn and GEMM post-merge consumers.
Larger flash-attn shapes exposed a different boundary problem:
the backend still treats exact-CB resident tiles as emitter-local state.

The failing pattern is not that a top-k, reduce, or flash-attn operator is
missing.
It is that exact-CB pages have no first-class lifecycle:

- producer and consumer helpers choose CBs through local maps;
- loop-carried values are recovered through buffer identities and stack state;
- `cb_pop_front` is emitted by local consumers instead of by a global last-use
  decision;
- a value live across a loop boundary can be popped at loop exit or replaced by
  a partial local fragment before its true final consumer.

For flash-attn, `acc_o` is a loop-carried full-tile value.
The dynamic loop's initial value, body value, backedge update, and loop-exit
value must be represented as one lifecycle problem.
They cannot be repaired by adding more completed-state maps in the source
emitter.

## Compiler Prior Art

The design follows the classic register-allocation shape:

- liveness analysis computes live-in, live-out, dead, and kill information;
- live intervals model where a virtual value is live;
- allocation maps virtual resources to finite physical resources;
- release/spill/materialization is inserted from the allocation decision.

Reference surfaces:

- LLVM codegen live variables, live intervals, PHI handling, and register
  allocation:
  `https://llvm.org/docs/CodeGenerator.html`
- Chaitin graph-coloring register allocation:
  `https://research.ibm.com/publications/register-allocation-via-coloring`
- Poletto / Sarkar linear-scan register allocation:
  `https://vsarkar.rice.edu/research/publications/public-register-allocation/`
- MLIR `scf.for` loop-carried `iter_args` / `scf.yield` semantics:
  `https://mlir.llvm.org/docs/Dialects/SCFDialect/`
- MLIR ownership-based deallocation using liveness and retained values:
  `https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/`

The analogy is direct but not literal:
physical CB IDs are the finite resource,
exact-CB resident tile pages are virtual values,
and `cb_pop_front` is the release event derived from last use.

## Goal

Represent and validate exact-CB lifecycle and resource allocation explicitly:

```text
SpatialPlan logical live values / carry edges
  -> TTProgram exact-CB virtual values and live intervals
  -> TTProgram physical CB allocation and release events
  -> ExecutableSpec projected CB / materialization / consumer records
  -> source/runtime consumes the projection
```

The implementation must make larger flash-attn cases a witness of the generic
lifecycle model, not a workload-shaped protocol.

## Non-Goals

- No frontend flash-attn, top-k, reduce, or loop-carried state operator.
- No new long-lived IR layer outside the existing chain.
- No `selection plan`, workload metadata registry, source-hook side channel,
  or direct-runtime-only patch surface.
- No physical L1 address allocator; TT-Metal still owns physical memory
  allocation.
- No global graph-coloring allocator in the first implementation unless
  linear scan proves insufficient.
- No semantic recovery from buffer names, source text, argument positions,
  generated source, or runtime observation.

## Chosen Shape

### Rejected: Emitter Map Repair

Continuing with maps such as
`buffer_live_form_cb_by_buffer_identity_`,
`exact_output_live_form_cb_by_buffer_identity_`,
or loop-carried completed-state maps is rejected as the architectural path.

Those maps may remain temporarily as implementation mechanics while the new
plan is introduced, but they cannot decide cross-boundary value lifetime,
physical CB assignment, or release timing.

### Deferred: Full Graph Coloring

Graph coloring is a valid later allocator when CB pressure becomes complex.
It should not be the first implementation because current Blackhole compute
segments are mostly structured and ordered.

### Selected: TTProgram Linear-Scan Exact-CB Allocator

The first implementation uses a TTProgram-owned liveness model and a
linear-scan allocator:

- assign program points to ordered leaf compute / materialization /
  transport events;
- build virtual exact-CB values from typed live-form and materialization
  evidence;
- compute live intervals, including structured loop live-in, backedge, and
  loop-exit values;
- allocate physical CB plans from target hardware facts and existing reserved
  CB classes;
- insert release events at computed last uses;
- reject or split/materialize when resource pressure cannot be admitted.

This is a TT resource-planning slice, not a new semantic layer.

## Layer Contract

### Normalized Tile TIR

Owns:

- explicit TT-Metal leaf compute semantics;
- loop, predicate, and access structure;
- buffer load/store and tile op semantics;
- enough structure to derive producer, consumer, and loop-carried flow.

Does not own:

- physical CB IDs;
- CB page lifetime;
- `cb_pop_front` placement;
- exact-CB resource allocation.

### SpatialPlan

Owns target-independent lifecycle evidence:

- `LogicalLiveValue` producer and consumer references;
- `DataflowEdge` entries with `carry` / recurrence edges;
- materialization boundaries;
- phase and ordering evidence;
- full-tile vs slice coverage.

Loop-carried values must be represented as dataflow/carry facts.
For a structured loop, the plan must distinguish:

- initial incoming value;
- loop body argument value;
- backedge yielded value;
- loop-exit value consumed after the loop.

If current TIR cannot prove those relationships, the backend must reject or
extend the upstream representation.
It must not let TT planning infer them from names such as `acc_o`.

### TTProgram

Owns exact-CB physical lifecycle and allocation.

The design introduces a typed TTProgram slice, conceptually:

```text
TTExactCBLifecyclePlan
```

The concrete implementation may split this into focused objects, but the
fields must remain typed and projected through the normal TTProgram contract.

Required concepts:

- virtual exact-CB value:
  logical value reference, producer event, shape, dtype, page requirement,
  event-lifetime class, and loop role;
- use event:
  consumer event, operand role, program point, full-tile/slice requirement,
  and whether it consumes or borrows the page;
- live interval:
  begin point, end point, live-in/live-out evidence, loop-carried flags, and
  interference class;
- allocation:
  virtual value or coalesced value group, selected `TTCBPlan`, physical CB ID,
  page window, and admitted event shape;
- release event:
  program point, `TTCBPlan`, page count, and reason
  (`last_use`, `loop_backedge_transfer`, `materialization_split`, or
  `typed_reject_boundary`).

`TTCBPlan.lifetime_begin` and `TTCBPlan.lifetime_end`
must become allocator outputs.
They must not be filled from source emission order guesses.

Existing `TTLiveFormPlan`,
`TTMaterializationPlan`,
`TTConsumerBindingPlan`,
`TTCBPlan`,
`TTResourceDemand`,
and `TTResourcePressureReport`
remain the durable public surfaces.
The exact-CB lifecycle objects feed those surfaces and validators.

### ExecutableSpec

Owns projection and admission only.

It may contain projected lifecycle/release records if source/runtime needs
them directly, but it must not recompute lifecycle.

Leaf consumers must reject missing lifecycle or release records for any
exact-CB value whose lifetime crosses a materialization, transport, or loop
boundary.

## Loop-Carried Semantics

Loop-carried exact-CB values are modeled like SSA loop arguments.

For flash-attn `acc_o`, the required abstract shape is:

```text
initial zero tile
  -> loop body live-in value
  -> per-iteration update
  -> backedge yielded value
  -> loop-exit value
  -> final consumer after the loop
```

The allocator may coalesce the initial, backedge, and exit virtual values into
one physical CB if their intervals and event windows permit it.
That is an allocation decision, not a semantic assumption.

Hard requirements:

- the backedge value is live across the next iteration's body input;
- the loop-exit value remains live until the final after-loop consumer;
- no loop-exit `cb_pop_front` is emitted before a later consumer;
- no after-loop consumer may be satisfied from a partial local fragment when
  the consumer requires the full logical tile;
- final local materialization is legal only if the liveness plan inserts it as
  a split/materialization event with full coverage evidence.

Implementation note from the larger-shape flash checkpoint:

- a clear-accum=false GEMM may reload a loop-carried local accumulator without
  a materialization fact only when loop-carried evidence comes from TIR
  read-before-write / exact-CB leaf accesses and the destination has a full
  static local state shape matching the GEMM tile set;
- source codegen may CB-back a `blackhole.acc` local allocation only when the
  TTProgram CB plan projects explicit `initial_reserve_pages`.  A metadata-only
  CB config is not permission to turn a local accumulator into a CB write
  pointer, because physical CB reuse is an allocation decision.

## Algorithm

### 1. Build Event Order

Create a deterministic event stream for each compute kernel:

- reader publications;
- explicit tile-compute leaf ops;
- materialization publications;
- transport consumers;
- writer consumers;
- loop entry, loop backedge, and loop exit boundaries.

Each event receives a stable program point.
Structured loops keep nested program points so loop live-in/live-out and
backedge relations are explicit.

### 2. Build Def-Use

For every admitted exact-CB live form:

- create a virtual value at the producer event;
- attach consumer use events through typed live-value / consumer-binding
  evidence;
- attach materialization and transport events through
  `TTMaterializationPlan` and `TTConsumerBindingPlan`;
- attach loop-carried values through `SpatialPlan` carry edges and loop
  boundary evidence.

The def-use builder may use pass-local maps to index objects.
Those maps are invalidatable analysis mechanics only.
They are not projected owner truth.

### 3. Compute Liveness

Compute live-in/live-out and live intervals over the event stream.

For structured loops:

- seed the loop header with live-in carried values;
- add backedge uses from yielded values to the next iteration's loop argument;
- add loop-exit values when a value is consumed after the loop;
- close loop-defined values that have outside uses with an explicit exit
  virtual value.

The implementation may start with structured-loop rules rather than a full CFG,
but the observable contract must match CFG liveness:
a value is live until all reachable uses have been satisfied.

### 4. Allocate Physical CBs

Use linear scan over intervals for the first implementation:

- active intervals hold currently allocated CBs;
- expired intervals release their CBs;
- overlapping intervals cannot share the same physical CB unless they are an
  explicitly coalesced same-value group;
- CB class, page count, page size, data format, and event shape must match
  before reuse;
- reserved reader/writer/intermediate CB ranges remain target facts.

If no CB is available:

- split a live range when a legal exact-CB materialization boundary exists;
- materialize to another admitted live form when a typed protocol exists;
- otherwise emit a typed `admission_blocked` reason such as
  `exact_cb_resource_pressure` or `exact_cb_lifetime_unallocatable`.

### 5. Emit Release Events

`cb_pop_front` is emitted only from allocator release events.

Source helpers may still render the TT-Metal call sequence, but they must not
decide last use locally.
`ReleaseExactInputAfterUse`-style helper logic is an implementation debt after
this plan lands.

### 6. Validate

Validators must prove:

- every exact-CB consumer references a live virtual value;
- every live interval has a producer and at least one valid end condition;
- every allocated virtual value references a valid `TTCBPlan`;
- no two interfering intervals share a physical CB;
- release events occur after the last use and before resource reuse;
- loop-carried values have initial, backedge, and exit evidence when consumed
  across the loop boundary;
- `TTResourcePressureReport` accounts for max simultaneous CB ID pressure and
  CB-backed L1 bytes from the allocated intervals.

## Existing-Debt Disposition

The following implementation surfaces become deletion targets:

- source-emitter owner truth in
  `buffer_live_form_cb_by_buffer_identity_`;
- exact-output owner truth in
  `exact_output_live_form_cb_by_buffer_identity_`
  and related order maps;
- loop-carried owner truth in
  `active_loop_carried_buffer_identity_stack_`,
  `completed_loop_carried_buffer_identities_`,
  and `loop_carried_live_form_cb_by_buffer_identity_`;
- local release decisions in `ReleaseExactInputAfterUse`;
- source scans that suppress or insert `cb_pop_front` based on immediate
  statement shape;
- final local-fragment fallbacks for full-tile after-loop consumers.

They may survive temporarily only as cursors for rendering a validated plan.
They must not remain as public or cross-stage protocol.

## Cutover Rule

The implementation must cut over the active exact-CB path in one coherent
change.

It is not acceptable to land the liveness/allocation plan while leaving the
old source-emitter lifecycle path as a parallel fallback.
For exact-CB consumers covered by the new plan:

- physical CB choice comes from the allocation plan;
- `cb_wait_front`, `cb_push_back`, and `cb_pop_front` event ordering comes from
  projected lifecycle/release events;
- source helpers only render the selected event sequence;
- old map-based owner truth must be deleted or demoted to private cursors whose
  values are checked against the plan;
- any uncovered exact-CB shape must fail closed before source/runtime instead
  of falling back to the old path.

The active-chain completion point is the removal of the old path for the
covered exact-CB surface, not merely the presence of a new plan object.

## Unsupported Taxonomy

Failures must be typed:

- `lowering_missing`:
  the TIR does not expose explicit leaf or carry evidence needed to build
  virtual values;
- `backend_op_missing`:
  TT-Metal can express the primitive, but Blackhole lacks the selected
  materialization or source projection;
- `admission_blocked`:
  liveness is known, but exact-CB lifetime, event shape, or resource pressure
  is not admitted;
- simulator capability boundary:
  the generated and admitted artifact hits a known TT-Sim unsupported path.

Plain `unsupported` and source/runtime best effort are forbidden.

## Validation Plan

### Structure

- Add TTProgram construction tests for exact-CB lifecycle objects.
- Add validator rejects for missing producer, missing use, missing loop
  backedge, premature release, and interfering intervals sharing a CB.
- Add projection tests proving lifecycle-driven `TTCBPlan` lifetime fields and
  release events are present.

### Source

Generated compute source for flash-attn seq128, seq256, and seq512 must prove:

- the loop-carried `acc_o` value is initialized as an exact-CB value;
- the dynamic loop consumes and yields the loop-carried exact-CB value;
- there is no immediate loop-exit pop while the exit value has later uses;
- the final after-loop consumer reads the loop-exit exact-CB value;
- no full-tile consumer is satisfied by a partial local fragment such as a
  direct `reinterpret_cast` of the local `acc_o` fragment.

These are source/spec guards.
They do not replace runtime correctness.

### Runtime

Direct-runtime correctness must use the repository TT-Sim setup and bf16
baseline.

Required positive gates:

- existing seq64 bf16 flash-attn MHA exact-CB partial-combine regression;
- seq128 bf16 MHA forward direct runtime;
- seq256 bf16 MHA forward direct runtime;
- seq512 bf16 MHA forward direct runtime, unless TT-Sim exposes a typed
  simulator capability boundary after source/spec admission is proven.

Required negative gates:

- constructed over-pressure exact-CB intervals reject before source/runtime;
- loop-carried value without exit evidence rejects before source/runtime;
- full-tile consumer with only slice/local-fragment coverage rejects before
  source/runtime.

2026-05-05 checkpoint status:

- TTProgram and ExecutableSpec now carry exact-CB virtual values, use events,
  live intervals, allocations, and release events for the covered flash
  loop-carried surface.
- The old loop-carried cb/buffer twin maps and completed-state recovery set
  were removed from active source lowering.  Loop-carried source rendering now
  uses one `LoopCarriedExactCBState` cursor, and every write to that cursor
  goes through the lifecycle-record helper.
- Seq128, seq256, and seq512 bf16 MHA source/spec gates prove the loop-carried
  exact-CB records for `acc_o` and the matching allocation/release records.
  Current TT-Sim direct runtime is fail-closed with a typed simulator reason
  when loop-carried input exact-CB backedge release would hit
  `tensix_execute_pacr: count=1`.
- Seq64 remains the positive direct-runtime correctness gate; the simulator
  gate is intentionally scoped to loop-carried input exact-CB backedge release,
  so accumulator-only loop state is not rejected.
- Borrowed exact-input last-use release decisions are driven by the
  allocator/release surface instead of local source helper decisions.
- 2026-05-05 follow-up: borrowed exact-input last-use rendering no longer
  calls `ShouldReleaseBorrowedExactInputAfterUse` or passes a local
  `should_release` boolean into release-event recording.  The source renderer
  consumes an optional typed release event produced through a release-policy
  helper.  Temporary owned exact inputs can still render a local consume pop
  when no cross-boundary live value exists; cross-boundary exact-CB
  materialization pops are not allowed to use that fallback.
- 2026-05-05 follow-up: validator rejects loop-carried exact-CB values whose
  live interval lacks live-in/live-out evidence, and rejects overlapping
  exact-CB virtual intervals that share one physical CB.  The interference
  gate exposed a real positive-path bug: virtual intervals inherited merged
  CB-requirement lifetime begin points, so later exact-CB versions on the same
  requirement looked live from program point 0.  The interval builder now uses
  producer/use evidence for virtual-value begin/end, and `PlanTTCBAlloc`
  incorporates exact-CB interval bounds into requirement lifetime before
  assigning physical CB IDs.
- 2026-05-05 completion follow-up: exact-CB materialization with
  `pop_front=true` now requires a typed `TTExactCBReleaseEvent` and fails
  closed if release lookup cannot resolve the materialized logical live value.
  Loop-carried exact-CB output materialization binds the destination buffer's
  logical identity before release lookup, so it does not fall back to an
  ephemeral local buffer identity.  `ValidateTTProgram` also rejects a
  full-logical-tile consumer bound to a `thread_distributed_slice` live form.
- 2026-05-05 completion verification: the T7.5 selector reported
  `10 passed, 3 skipped`.  The three skips remain the typed TT-Sim
  `tensix_execute_pacr: count=1` capability boundary for seq128/256/512 after
  source/spec admission; seq64 remains the positive direct-runtime correctness
  gate.

## Completion Criteria

This design is implemented only when:

- exact-CB lifecycle is represented in typed TTProgram fields or objects;
- `TTCBPlan` lifetime and release behavior come from liveness/allocation;
- source emission consumes lifecycle/release records instead of deciding
  ownership from local maps;
- the old map-based exact-CB lifecycle/release path is deleted from the
  covered active chain rather than retained as a fallback;
- validators reject inconsistent lifecycle and resource allocation;
- seq64, seq128, seq256, and seq512 bf16 flash-attn gates either pass direct
  runtime or fail from a typed simulator capability boundary after source/spec
  admission;
- current workload cases are witnesses, not protocol definitions;
- docs, progress, and memory reflect the new boundary.
