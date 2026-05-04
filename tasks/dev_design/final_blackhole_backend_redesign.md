# TileLang Blackhole Backend Redesign

## Role

This is the only overall design document for the Blackhole backend.
It defines durable architecture contracts, not repo-history logs.

Execution status, blockers, and latest verification live in
`tasks/progress.md`.
Task-level details live in the task design documents listed by
`tasks/dev_design/README.md`.

## Core Conclusion

The Blackhole backend must be organized around one explicit IR chain:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

The root problem was not a shortage of clever late matchers.
It was that virtual spatial/dataflow semantics and TT target-realization
semantics were repeatedly carried through attrs, bags, helper wrappers,
payloads, naming conventions, or runtime fallback.

The end state is:

- program semantics live in the current explicit representation layer
- analyses are pass-local, invalidatable, and recomputable
- cross-stage truth is represented by typed IR fields or explicit objects
- validators fail closed before source / runtime emission
- runtime and codegen consume leaf projections; they do not recover planner
  semantics

## Hard Rules

### IR-First Discipline

- Anything needed by downstream legality, planning, source emission,
  codegen, or runtime admission must be explicit in the current IR layer or
  the next lowered representation.
- Pass-local maps, sets, helper structs, wrappers, and cursors may exist only
  as implementation mechanics.
  They cannot become cross-stage protocol.
- If a distinction cannot be recomputed from the current IR after analysis
  invalidation, it must be represented explicitly.
- Pass names are implementation details.
  Long-term design boundaries are representation layers and typed objects.

### Hardware-Codegen Usefulness Gate

A new object, algorithm, typed field, validator, or DAG mechanism counts as
mainline progress only if it makes a DSL-authored kernel lower to real
TT-Metal hardware code more reliably or more efficiently by doing at least
one of the following:

- changes leaf normalization
- changes legality or admission
- changes typed plans
- changes resource plans
- improves typed unsupported diagnostics before source/runtime emission
- deletes an old matcher, payload, fallback, or side channel

Dumping, checking names, projecting unused metadata, or adding a structure
that is only “future useful” is foundation work, not completion.

### Compute Granularity

Blackhole compute semantics are represented at TT-Metal leaf API granularity:

- `matmul_tiles`
- `reduce_tile`
- unary leaf ops such as `exp2_tile`, `recip_tile`, `typecast_tile`
- binary leaf ops such as `add_tiles`, `mul_tiles`, `binary_max_tile`
- broadcast leaf ops such as `add_tiles_bcast_cols`,
  `mul_tiles_bcast_cols`
- copy / pack / tilize / untilize leaf ops

Composite helpers such as
`softmax`,
`exp2_affine`,
or row-broadcast affine variants are not production compute op names.
They cannot enter
`TTComputeOpPlan.operation_name`
or
`KernelSpec.compute_ops`.

### Composite Lowering Boundary

Leaf-looking payloads must not hide composite semantics.
Forbidden examples:

- `exp2_tile(mode, lhs, rhs, scale, ...)`
- `mul_tiles_bcast_cols("div", ...)`

If a TIR expression is expressible through TT-Metal leaves, it must be
normalized into explicit leaf tile-compute statements in
`Normalized Tile TIR`
before DAG construction or TT source emission.

`TileComputeDAG`
may cover explicit leaf nodes.
It must not become a composite expression lowerer.
One DAG source node maps to one semantic TT-Metal leaf op.

### Unsupported Taxonomy

Fail-closed diagnostics must preserve the reason category:

- `lowering_missing`:
  TIR semantics are expressible, but the normalizer does not yet produce the
  needed explicit leaf sequence.
- `backend_op_missing`:
  TT-Metal can express the primitive, but the Blackhole wrapper / planner /
  codegen does not expose it.
- `admission_blocked`:
  the op exists, but resource, layout, sync, event lifetime, core placement,
  or runtime support is not proven safe.
- semantic unsupported:
  only after TT-Metal primitive coverage and legal-composition audit show no
  valid expression.

Plain `unsupported` is not a license to skip leaf op or normalizer work.

## Layer Contracts

### Normalized Tile TIR

Owns:

- algorithmic compute semantics
- tile op and explicit `tl.tileop.blackhole_compute` leaf calls
- buffer load/store semantics
- access regions, predicates, loops, and local dataflow structure
- expression normalization into explicit TT-Metal leaf tile-compute
  statements when the leaf set can express the value

Does not own:

- TT core placement
- CB / semaphore / runtime arg allocation
- target launch order
- source emission
- runtime admission

Exit invariant:
admitted tile compute is explicit at TT-Metal leaf granularity.

### SpatialPlan

Owns target-independent virtual spatial/dataflow semantics:

- execution units
- dataflow / carry / reduction / broadcast / join edges
- logical live values
- materialization boundaries
- access-region evidence
- target-independent phase and ordering evidence
- validated hints

Does not own:

- TT builtin family realization
- CB IDs or semaphore IDs
- TT core placement
- runtime args
- executable layout

Exit invariant:
all target-independent dataflow and lifecycle facts needed by TT planning are
typed and validated.

### TTProgram

Owns TT-specific physical realization:

- hardware model facts used by planning
- mesh / submesh / device placement
- core groups and worker placement
- buffer distribution and memory placement class
- kernel roles and block plans
- compute op plans
- CB, semaphore, transport, sync, ABI, runtime-arg plans
- resource demand and resource pressure reports
- launch / execution plans

Does not own:

- target-independent dataflow semantics
- runtime/backend-specific fallback recovery
- executable encoding details

Exit invariant:
all source/codegen/runtime-visible TT decisions are explicit, typed, and
validated.

### ExecutableSpec

Owns leaf projection and backend admission:

- executable schema and entry identity
- projected kernel / segment records
- projected buffers, CBs, semaphores, runtime args, accessors
- backend admission results
- runtime-module build inputs for `BlackholeModule` or codegen/export

Does not own:

- target planning
- compute legality
- resource allocation
- semantic recovery from source text or builtin sequences

Exit invariant:
leaf consumers can either execute/build the spec directly or fail closed with
a typed reason.

## Validator Discipline

Every representation has a validator.

Validators must reject:

- missing typed owner-truth fields
- stale bridge attrs or payload-derived semantics
- operation names outside the admitted leaf set
- source hooks that project more than one semantic leaf op for one source node
- resource reports that do not match resource demand
- CB / L1 / core / buffer pressure that exceeds admitted hardware facts
- runtime/codegen records that require planner semantics not present in
  `ExecutableSpec`

Validators must not:

- infer semantics from names
- accept a bag / payload because current code happens to produce it
- let source or runtime rebuild a missing plan
- silently downgrade missing proof into a best-effort path

## Fake Protocol Disposition

The following surfaces are historical debt unless explicitly reintroduced as
typed fields in the correct layer:

- top-level `TTProgram.payload`
- `compute_contract`, `gemm_contract`, `multi_*_contracts`
- bridge attrs used as cross-stage truth
- map/any lowering facts contracts
- compute-op seed maps
- leaf name/default fallback
- legacy external runner path
- runtime/codegen semantic recovery from work ids, buffer names, or builtin
  sequences

If an old surface still exists in implementation, the design stance is
`wrong now, delete or rewrite into explicit IR`.
Survival in current code is not design legitimacy.

## TileComputeDAG Boundary

`TileComputeDAG`
is allowed only as a pass-local selection model over explicit leaf tile
compute nodes.

It may:

- legalize explicit leaf graph nodes
- choose target leaf patterns
- reason about fanout and materialization for explicit leaf values
- feed typed `TTComputeOpPlan`,
  `TTMaterializationPlan`,
  `TTConsumerBindingPlan`,
  `TTResourceDemand`,
  `TTResourcePressureReport`,
  or typed unsupported diagnostics
- delete old per-op branches

It may not:

- persist as a durable IR layer
- own composite expression lowering
- allocate CB IDs
- place cores
- schedule NoC traffic
- carry cross-pass payload truth
- justify itself through diagnostic dumps alone

## Resource Planning Boundary

Resource planning belongs to `TTProgram` and `ExecutableSpec` admission,
not to `TileComputeDAG`.

The resource-planning surface is:

- `TTHardwareModel`
- core group plans
- buffer distribution plans
- CB plans
- semaphore / sync / transport plans
- `TTResourceDemand`
- `TTResourcePressureReport`

CB / L1 planning should fail closed using hardware facts before source or
runtime emission.
Core placement and buffer distribution must likewise become hardware-model
backed before wider workload / runtime admission expands.

TT-Metal remains responsible for physical allocator address assignment.
The compiler plans placement classes, pressure, and legality.

## Runtime Boundary

Direct runtime is one `ExecutableSpec` leaf backend.
It is not the capability ceiling of `TTProgram` or codegen/export.

Admitted direct-runtime subsets may be narrow.
That narrowness must surface as backend admission, not as upstream IR shape.

Long-term codegen/export targets TT-Metal
`Program / MeshWorkload / MeshBuffer`
style explicit program construction.

## Workload Admission Boundary

`flash-attn` is a stress witness, not the only workload family.

Non-flash compute remains in scope:

- copy / layout movement
- GEMM variants
- standalone unary / binary / broadcast / reduce / pack / typecast leaf
  compute workloads
- `topk` / selection / indexing
- MoE / `fusedmoe` routed or segmented dispatch
- paged attention / paged decode / MLA decode with paged KV access
- grouped / ragged / sparse attention variants
- stateful reduction-update and chunk recurrence / scan families

A workload is admitted only when typed plans,
backend admission,
and correctness verification match its risk.
Compile/source/spec success alone is not direct-runtime correctness.
Family names such as MoE, paged attention, or flash decode are not admission
units.
Admission is by explicit subset and capability proof.

## Operating Direction

The operating order is:

1. Keep the four-layer IR chain as the only semantic path.
2. Preserve compute at TT-Metal leaf granularity in `Normalized Tile TIR`.
3. Keep algorithmic structures only where they drive typed decisions:
   access legality,
   dependence,
   live-form,
   materialization,
   resource demand,
   resource pressure,
   typed rejects.
4. Keep `TileComputeDAG`
   pass-local and production-useful only for explicit leaf graph covering.
5. Upgrade core placement and buffer distribution using `TTHardwareModel`.
6. Re-enter wider workload / runtime admission after resource evidence is
   explicit.

## Completion Criteria

A task is complete only when:

- the relevant representation owns the semantics it needs to own
- source/runtime no longer recover missing planner truth
- validators reject inconsistent or missing evidence
- tests or compile/runtime checks match the task risk
- docs and `tasks/progress.md` reflect the current execution boundary without
  duplicating history
- any stable lesson belongs in `memory/`, not in this overall design
