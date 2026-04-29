# Blackhole Algorithmic Generalization Contract

## Role

This document defines the algorithmic-analysis structures that are allowed on
the Blackhole main chain.
It is a task-level contract, not a progress log and not a second overall
design.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current implementation status:
`tasks/progress.md`.

## Goal

Replace workload-shaped local heuristics with reusable typed evidence that
drives real lowering decisions:

- access legality
- dependence / recurrence legality
- live-value reaching-definition queries
- live-form and materialization decisions
- typed plan projection
- typed unsupported diagnostics

The structures are useful only when their outputs are consumed by
`SpatialPlan`,
`TTProgram`,
`ExecutableSpec`,
validators,
or admission diagnostics.

## Non-Goals

- Not a compute expression lowering system.
- Not a global resource allocator.
- Not a core placer.
- Not a NoC scheduler.
- Not a persistent side-channel between passes.
- Not a proof that a new graph exists because a dump/test can print it.

## Structures

### AccessRegion

Represents affine-lite access evidence derived from current IR:

- buffer identity
- region shape / rank
- full vs slice coverage
- axis and stride evidence when statically available
- subject identity for producer/consumer compatibility

Allowed uses:

- validate access compatibility
- decide full-tile vs slice admission
- support materialization coverage checks
- explain unsupported access cases

Forbidden uses:

- recover semantic roles from buffer names
- invent missing access semantics downstream
- persist as a cross-stage bag outside explicit plans

### SpatialPlan Dependence Graph

Represents target-independent dependence between spatial/dataflow entities:

- producer / consumer edges
- carry / recurrence edges
- reduction / broadcast / join relationships
- materialization boundaries
- indexed references to access-region and live-value evidence

Allowed uses:

- validate loop-carried materialization lifetime
- gate recurrence-sensitive live-form decisions
- provide indexed evidence to TT planning

Forbidden uses:

- replace `SpatialPlan` with a new hidden graph IR
- let a later pass rebuild dataflow from names or statement order

### LiveValueSSA

Represents logical live-value versions and uses:

- producer version
- consumer uses
- edge / boundary indices
- loop-carried and materialization relationships

Allowed uses:

- reaching-definition queries
- source-live-form selection
- consumer binding
- materialization admission

Forbidden uses:

- global lifetime allocation
- CB ID allocation
- physical buffer placement
- source-emitter fallback

### TT Live-Form Solver

Consumes validated live-value and materialization-boundary evidence to choose
physical live forms for the admitted subset.

Allowed outputs:

- `TTLiveFormPlan`
- `TTMaterializationPlan`
- `TTConsumerBindingPlan`
- typed admission diagnostics

Forbidden outputs:

- untyped payloads
- hidden source-emitter instructions
- runtime-only recovery rules

## Active-Use Contract

Algorithmic structures must pay rent.

For each structure, at least one active consumer must make a different
typed decision because of it:

| Structure | Required active use |
| --- | --- |
| `AccessRegion` | access compatibility, coverage, axis, or materialization legality |
| dependence graph | recurrence / loop-carried / dataflow legality |
| `LiveValueSSA` | source-live reaching-def or consumer binding |
| live-form solver | physical live form, materialization protocol, or typed reject |

If a structure is only dumped, pretty-printed, or checked for shape, it is
foundation/debug work, not task completion.

## Relation To TileComputeDAG

`TileComputeDAG`
may consume this evidence for explicit leaf graph covering.
It must not use these structures to justify composite source-hook lowering.

Correct relation:

```text
Normalized Tile TIR explicit leaf nodes
  + AccessRegion / Dependence / LiveValueSSA evidence
  -> pass-local TileComputeDAG covering
  -> typed plans / resource demand / typed diagnostics
```

Incorrect relation:

```text
scalar/composite expression
  -> TileComputeDAG or source hook
  -> hidden multi-op semantic lowering
```

## Resource-Planning Boundary

These structures provide semantic evidence for resource planning.
They do not allocate resources.

Resource allocation and admission belong to:

- `TTHardwareModel`
- core group plans
- buffer distribution plans
- CB plans
- `TTResourceDemand`
- `TTResourcePressureReport`
- `ExecutableSpec` backend admission

If a resource decision needs lifecycle evidence,
it should consume indexed typed evidence from the appropriate plan.
It should not make `LiveValueSSA` or `TileComputeDAG`
the owner of physical allocation.

## Completion Criteria

This lane is complete only when:

- each structure has a real active-chain consumer
- validators reject missing or inconsistent evidence
- source/runtime emission no longer recovers the same semantics through a
  fallback matcher
- unsupported diagnostics distinguish missing lowering, missing backend op,
  and admission failure
- current workload cases are witnesses, not protocol definitions

## Validation

Required validation should cover:

- structure dumps for focused debugging
- validator failures for missing indexed evidence
- planner tests showing selected plans consume the evidence
- runtime/source tests only after typed plans and admission gates are already
  in place

Doc-only or dump-only checks cannot close a production phase.
