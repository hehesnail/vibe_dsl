# Task 1: SpatialPlan Representation Contract

## Role

This document defines the durable `SpatialPlan` representation contract.
The historical filename contains `companion`, but no new companion layer is
being introduced.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

`SpatialPlan`
is the target-independent virtual spatial/dataflow program derived from
`Normalized Tile TIR`.

It answers:

- which anchored sub-TIR regions are stable execution units
- how units are connected by dataflow, carry, reduction, broadcast, or join
  edges
- what virtual layout/distribution relationships exist
- where phase and materialization boundaries occur
- which logical live values are produced and consumed
- which validated hints may be consumed by target planning

It does not answer:

- TT builtin family selection
- CB / semaphore / runtime-arg allocation
- core placement
- transport / sync / launch realization
- executable leaf projection

## Inputs

Allowed inputs:

- current `Normalized Tile TIR`
- anchored sub-TIR structure
- access-region and dependence evidence derived from current IR
- validated hints

Forbidden inputs:

- legacy transition attrs or helper bridges
- public analysis wrappers or pass-to-pass facts bags
- TT-specific runtime/codegen conclusions
- buffer or variable names as semantic roles

If current TIR evidence is insufficient, extend the IR/DSL or reject.
Do not create a replacement side channel.

## Representation Objects

### ExecutionUnit

Represents an anchored sub-TIR execution unit.

Required meaning:

- stable identity and anchors
- unit role
- covered TIR region
- locality / carry / aggregation obligation

It does not duplicate tile op bodies or access expressions.

### DataflowEdge

Represents a relationship between execution units.

Required meaning:

- producer and consumer
- edge kind:
  `flow`,
  `carry`,
  `broadcast`,
  `reduction`,
  `join`
- subject identity
- phase relationship
- anchors / evidence references

It does not decide CB, NoC, semaphore, or delivery kind.

### LayoutSpec

Represents target-independent virtual layout and distribution.

Required meaning:

- logical tensor / buffer identity
- shard / replicate / distribute relationship
- virtual device-axis binding
- logical mesh-axis relationship
- collective intent such as reduce, gather, shard, or replicate

It does not encode TT-Metal device coordinates or physical buffers.

### PhasePlan

Represents virtual ordering and visibility boundaries.

Required meaning:

- phase identity
- phase membership
- cross-phase edges
- materialization boundaries
- ordering evidence

It does not encode TT launch order or runtime synchronization primitives.

### LogicalLiveValue

Represents logical producer/consumer value flow.

Required meaning:

- logical value identity
- producer and consumer references
- coverage / shape / dtype evidence
- full-value vs slice requirements
- same-phase or cross-phase relation

It does not choose physical live form, CB IDs, or source emission.

### ValidatedHintSet

Represents hints proven against current IR.
Hints that cannot be validated do not enter `SpatialPlan`.

## Validation Contract

`ValidateSpatialPlan`
must reject:

- duplicate or missing execution-unit identities
- edges whose endpoints do not exist
- phase membership inconsistencies
- layout specs without logical subject identity
- live values without producer/consumer evidence
- hints not validated against current IR
- any legacy bridge/payload field used as owner truth

## Exit Invariant

After `SpatialPlan`,
all target-independent dataflow and lifecycle facts needed by TT planning are
typed and validated.

Target realization must not recover those facts from names, payloads, or
runtime/codegen behavior.
