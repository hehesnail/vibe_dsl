# Blackhole Resource Planning Roadmap

## Role

This document defines the resource-planning direction after the algorithmic
generalization and tile-compute covering review.
It is not an overall design document and not a status log.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Scope

Resource planning belongs to `TTProgram` and `ExecutableSpec` admission.
It must not be pushed into `TileComputeDAG`,
source hooks,
runtime fallback,
or diagnostic-only reports.

Allowed owner-truth surfaces:

- `TTHardwareModel`
- core group plans
- buffer distribution plans
- CB plans
- semaphore / sync / transport plans
- `TTResourceDemand`
- `TTResourcePressureReport`
- executable admission records

## Non-Goals

- No new IR layer.
- No persisted `TileComputeDAG` payload.
- No physical address allocator; TT-Metal owns allocation.
- No NoC / multicast / global scheduling before typed core and buffer
  placement exist.
- No direct-runtime-specific patch surface.

## Inputs

Resource planning may consume:

- validated `SpatialPlan`
- validated `TTProgram` plans
- `TTHardwareModel`
- explicit tile-compute fanout / materialization demand
- buffer distribution requirements
- communication edges when represented explicitly

It may not consume:

- buffer names as semantic roles
- source text
- runtime fallback observations as owner truth
- diagnostic DAG dumps as planning input

## Direction 1: Keep `TileComputeDAG` Constrained

`TileComputeDAG`
may feed resource demand only through explicit leaf-graph facts:

- fanout
- share vs materialize decisions
- unsupported materialization reasons

It cannot allocate resources or place work.
If a DAG decision does not change typed plans, resource demand,
validators, or typed diagnostics, it is debug infrastructure.

## Direction 2: Resource Demand And Pressure Reports

`TTResourceDemand`
records what the program needs.
`TTResourcePressureReport`
records whether the target admits it.

Required report categories:

- tile-compute fanout and materialization demand
- CB requirements and CB ID pressure
- CB L1 bytes
- allocator-managed L1 buffer bytes
- max simultaneous L1 pressure
- semaphore pressure
- core-grid requirement
- buffer-distribution requirement
- typed unsupported reasons

Reports must drive validators or typed admission diagnostics.
A dump-only report is not completion.

## Direction 3: CB And L1 Admission

CB planning should use live intervals and target hardware facts:

- target CB count
- reserved / conventional CB classes
- page size and page count
- data format and flow class
- publish / consume event shape
- initial reserve semantics

L1 planning should be pressure admission first:

- CB bytes per core
- allocator-managed L1 buffer bytes
- worker L1 budget
- alignment waste estimate
- max simultaneous pressure

This layer rejects unsafe pressure.
It does not assign physical L1 addresses.

## Direction 4: Core And Buffer Placement

The next active mainline task is to make placement hardware-model backed.

Core placement must consume:

- available worker grid
- worker count under harvesting
- worker L1 size
- DRAM view count when relevant

Initial policy should be conservative:

- logical coordinates
- deterministic row-major work packets
- rectangular core ranges when useful
- do not materialize physical cores outside the hardware worker grid
- allow logical work items to exceed physical worker count only through
  explicit deterministic work packets on admitted workers
- typed reject when requested physical workers or coordinates exceed
  `TTHardwareModel`

`TTBufferDistributionPlan`
must grow beyond
`unit_mesh`
/
`replicated`
to represent:

- interleaved DRAM
- interleaved L1
- height / width / block sharded L1
- host-visible vs device-local placement
- page size
- shard shape
- attached core range

## Direction 5: NoC / Multicast / Scheduling Later

Only after typed core and buffer placement exist should the backend add:

- communication-weighted placement
- multicast grouping
- NoC0 / NoC1 role-aware scoring
- bounded FIFO / exact-CB event sizing
- list scheduling or graph partitioning

Adding these before typed placement exists would recreate the current
over-complexity problem under a new name.

## Current Order

1. Keep explicit leaf tile-compute and DAG covering boundaries clean.
2. Use typed resource demand / pressure reports as admission surfaces.
3. Keep CB / L1 admission hardware-backed.
4. Keep core groups hardware-model-backed and replace unit buffer placement
   with explicit buffer distribution.
5. Re-enter wider runtime admission:
   multi-block flash-attn,
   wider exact-CB events,
   mesh / distributed runtime,
   later NoC / multicast work.

## Completion Criteria

This roadmap is implemented only when:

- resource truth is represented in typed `TTProgram` / `ExecutableSpec`
  records
- validators consume the reports and fail closed
- CB / L1 checks use hardware-model facts
- core groups are derived from hardware facts, not hard-coded constants,
  and validators reject out-of-grid physical cores
- buffer distribution can express the placement choices needed by admitted
  workloads
- no resource truth is carried by bags, payloads, source hooks, or runtime
  fallback
