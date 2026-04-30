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

- logical grid requirement
- physical worker coordinates
- deterministic work packets
- linearization policy

Physical cores must be inside the hardware worker grid.
Logical work may exceed physical workers only through explicit work packets.

### TTBufferDistributionPlan

Represents buffer distribution and placement class:

- distribution kind
- memory space
- layout
- page size
- shard shape and orientation when sharded
- host visibility
- attached core range when relevant

This is the current active expansion point.

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
