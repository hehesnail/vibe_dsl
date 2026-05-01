# Task 2: TTProgram Representation Contract

## Role

This document defines the durable `TTProgram` representation contract.
The historical filename contains `companion`, but no new companion layer is
being introduced.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

`TTProgram`
is the TT-specific target realization of validated spatial/dataflow semantics.

It answers:

- which TT hardware facts planning uses
- which mesh / device / submesh placement is selected
- how logical work maps to core groups and work packets
- how buffers are distributed and placed
- which kernels, blocks, compute ops, transports, syncs, ABI records, and
  execution plans exist
- what resource demand exists and whether the target admits it

It does not answer:

- target-independent dataflow semantics
- semantic recovery from scalar loops, source text, names, or runtime behavior
- executable encoding details
- backend admission beyond typed plan validity and resource pressure

## Inputs

Allowed inputs:

- validated `SpatialPlan`
- anchored sub-TIR
- explicit tile-compute semantics already present in `Normalized Tile TIR`
- `TTHardwareModel`
- validated target hints

Forbidden inputs:

- public `AnalyzeBlackhole*` wrappers
- lowering facts bags or contract maps
- compute-op seed maps
- bridge attrs
- runtime/codegen/executable fallback conclusions
- buffer names as semantic roles

If TT planning lacks proof, return a typed reject or extend the upstream
representation.

## Representation Slices

### TTHardwareModel

Represents target facts used by planning and validation:

- worker grid
- functional worker count
- worker L1 budget
- DRAM view facts
- CB count and alignment facts

Planning must not hard-code facts that are available from this model.

### TTMeshPlan

Represents physical mesh and device-coordinate coverage:

- mesh identity and shape
- device range / submesh membership
- device coordinate mapping
- workload coverage

### TTCoreGroup

Represents worker placement and logical work packets:

- logical work grid requirement from the DSL kernel domain
- physical worker coordinates admitted by the hardware model
- deterministic work packets that map one or more logical work items onto a
  physical worker
- linearization policy

Physical cores must be inside the hardware worker grid.
Logical work may exceed physical workers only through explicit work packets.
In that case the physical worker runs a temporal loop over its packet,
recomputing logical block coordinates from the work-linear id.
Per-worker L1 / CB scratch is resident on the physical worker and reused for
each temporal work item; it is not multiplied by the total logical block
count.

### TTBufferDistributionPlan

Represents buffer distribution and placement class:

- distribution kind
- memory space
- layout
- page size
- sharding strategy when sharded
- shard-grid shape when sharded
- real per-core data shard shape when sharded
- logical-index to core-local address mapping when sharded or per-work
  indexed
- source buffer / source region binding when an L1 view materializes data
  from a DRAM/global tensor
- host visibility
- attached core range when relevant

This is the current active expansion point.

`shard_shape`
means the per-core tensor data shape in the durable contract.
It must not be used as a synonym for the physical or logical core-grid shape.
Core-grid attachment belongs to the attached core group and shard-grid fields.
`sharding_strategy` follows TT-Metal memory-layout strategy semantics:
`height`, `width`, or `block`.
`shard_orientation` follows TT-Metal `ShardOrientation` semantics:
`row_major` or `col_major`.
These two fields must not be conflated.

For TT-Metal compatibility,
the canonical sharded-memory shape is:

- tensor memory layout:
  height / width / block sharded
- shard spec:
  core grid,
  per-core shard shape,
  row-major or column-major core traversal
- buffer type:
  usually L1 for current admitted direct-runtime forms

`source_buffer`,
`source_region_kind`,
and `source_region_shape`
are not TT-Metal `ShardSpec` fields.
They are TileLang runtime materialization ABI fields that explain how a
resident L1 working view is filled from a DRAM/global source.

GPU-style
`alloc_shared`
shapes in the frontend are interpreted as per-work-item local scratch shapes
for TT planning.
The backend may validate them against worker L1 / CB capacity and report
underutilization, but it must not silently enlarge the scratch shape to fill
Blackhole L1.
Any TT-specific retile or work-coarsening choice must be represented as an
explicit planning decision before source / runtime emission.

### TTComputeOpPlan

Represents TT-Metal leaf compute realization:

- kernel association
- operation name at TT-Metal leaf granularity
- operand bindings
- tile/problem/block/subblock shape
- accumulator dtype
- optional materialization/fanout metadata

It must not record composite operation names.

### TTCB / Transport / Sync / ABI / Execution Plans

Represent the remaining TT-specific physical realization:

- CB pages, roles, flow class, publish/consume event shape
- transport/accessor descriptors
- semaphore and sync plans
- compile-time/runtime ABI schema
- launch and execution ordering

## Validation Contract

`ValidateTTProgram`
must reject:

- missing required typed slices
- compute op names outside admitted leaf set
- stale bridge/payload/contract fields
- resource reports inconsistent with resource demand
- CB / L1 / core pressure beyond `TTHardwareModel`
- core coordinates outside the hardware worker grid
- buffer distributions that are incomplete or unsupported

## Exit Invariant

After `TTProgram`,
all source/codegen/runtime-visible TT decisions are explicit, typed, and
validated.

`ExecutableSpec` must be a projection of these decisions, not a recovery pass.
