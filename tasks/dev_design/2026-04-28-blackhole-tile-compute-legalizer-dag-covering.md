# Blackhole Tile Compute Legalizer And DAG Covering

## Role

This document defines the tile-compute legalizer and covering boundary.
It is a task-level design contract, not a phase log.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current implementation status:
`tasks/progress.md`.

## Goal

Make TT-Metal leaf compute selection extensible without reintroducing
workload-shaped matchers or composite pseudo-ops.

Pipeline:

```text
Normalized Tile TIR explicit leaf tile compute
  -> pass-local TileComputeDAG
  -> legalization
  -> target leaf pattern covering
  -> typed plans / resource demand / source projection
```

The design is justified only when it:

- selects typed leaf plans
- produces typed unsupported diagnostics before source/runtime emission
- feeds materialization / fanout / resource demand decisions
- or deletes old per-op branch/fallback mechanics

## Non-Goals

- No new long-lived IR layer.
- No composite operation names such as `softmax`, `exp2_affine`, or
  `row_broadcast_exp2_affine`.
- No leaf-looking composite payloads such as
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  or
  `mul_tiles_bcast_cols("div", ...)`.
- No expression decomposition inside source hooks.
- No resource allocation, core placement, NoC scheduling, or global lifecycle
  planning.
- No DAG production claim based only on dumps or diagnostics.

## Input Contract

Input must already be explicit leaf compute semantics in
`Normalized Tile TIR`:

- preserved `tl.tileop.*`
- explicit `tl.tileop.blackhole_compute`
- access-region evidence
- live-value / dependence evidence
- validated planning context

If a source expression requires multiple TT-Metal leaf ops, the normalizer
must emit multiple explicit leaf statements before DAG construction.

## Output Contract

Durable outputs must be typed plan fields, not a persisted DAG:

- `TTComputeOpPlan`
- `TTCBPlan`
- `TTLiveFormPlan`
- `TTMaterializationPlan`
- `TTConsumerBindingPlan`
- `TTResourceDemand`
- `TTResourcePressureReport`
- typed unsupported diagnostics

The DAG itself remains pass-local.

## Leaf Operation Set

Current admitted leaf names are TT-Metal API granularity:

- `fill_tile`
- `copy_tile`
- `typecast_tile`
- `binary_max_tile`
- `add_tiles`
- `mul_tiles`
- `add_tiles_bcast_cols`
- `mul_tiles_bcast_cols`
- `exp2_tile`
- `recip_tile`
- `reduce_tile`
- `pack_tile`
- `matmul_tiles`

Adding a new leaf op requires:

1. pattern schema entry
2. legality rule
3. operand role schema
4. source projection hook only if standalone source emission is admitted
5. validator / planner tests

## TileComputeDAG Contract

`TileComputeDAG`
models explicit leaf-level compute dependencies.

Nodes carry:

- source identity
- leaf operation name
- dtype / shape evidence when known
- side-effect class
- operand roles
- materialization / fanout metadata

Edges carry:

- producer / consumer relation
- operand role
- live-value / access-region evidence when available
- materialization requirement when selected

Allowed decisions:

- legal / reject
- choose target leaf pattern
- choose share vs materialize for explicit leaf graph fanout
- report resource-demand consequences

Forbidden decisions:

- split one source node into several semantic leaf ops
- recover compute family from source text or names
- allocate physical CB IDs
- choose core placement or NoC schedule

## Pattern Schema

Each leaf pattern defines:

- `name`
- `root_op_name`
- result kind
- TT-Metal leaf `operation_name`
- operand roles
- required input forms
- produced form
- side-effect class
- source emitter kind, if admitted
- cost metadata

Pattern operands are semantic leaf operands.
They must not include operation-changing `mode` or `kind` strings.

## Legalization

Legalization classifies each explicit node as:

- `Legal`
- `Lower`
- `Split`
- `PromoteDType`
- `Materialize`
- `Reject`

In the current contract,
`Lower` / `Split`
may only target explicit TIR leaf normalization.
They cannot be implemented as source-emitter expansion.

Rejects must carry typed reasons:

- unsupported shape / axis / dtype
- missing access or live-form proof
- event lifetime not admitted
- backend op missing
- resource admission blocked

## Source Projection

Source hooks are projection metadata for selected leaf patterns.
They should be implemented behind a narrow source-projection helper,
not as a long public/protected method family on
`PlanTTKernelABI`.

They may:

- emit the TT-Metal init/body/pack micro-sequence for one selected leaf op
- record the matching typed compute plan
- publish / consume already selected live-form values

They may not:

- perform expression decomposition
- choose a different operation family
- emit multiple semantic `TTComputeOpPlan.operation_name` entries for one DAG
  source node
- infer semantics from source text

## Resource Interaction

The DAG may feed typed resource demand when the decision comes from explicit
leaf graph fanout or materialization.

It must not own resource allocation.

Resource planning consumes DAG-derived demand through:

- `TTResourceDemand`
- `TTResourcePressureReport`
- validators
- executable projection

## Completion Criteria

This lane is production-useful only when:

- every admitted source node maps to one semantic leaf op
- no leaf-looking source call carries composite semantics
- pattern covering changes typed plans, typed diagnostics, resource demand,
  or deletes old branch paths
- validators reject pattern / plan mismatches
- adding a new leaf op is localized to pattern + legality + source hook +
  tests
- diagnostic-only DAG code is not cited as production completion

## Validation

Required tests:

- pattern table covers admitted leaf op names
- composite helper names do not appear as `operation_name`
- composite payload strings do not appear behind leaf-looking calls
- selected patterns match `TTComputeOpPlan.operation_name`
- source hooks are registered only for admitted standalone source paths
- validators fail closed on unsupported or inconsistent patterns

Runtime tests should be added only after the typed plan and admission path for
that leaf surface is already in place.
