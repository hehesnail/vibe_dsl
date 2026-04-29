# Blackhole Resource Planning Roadmap

## Role

This is a task-level design note for TT resource planning after the
algorithmic-generalization and tile-compute-covering review.

It is not a new overall design document.
The only long-term chain remains:

```text
Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec
```

This document records how the next resource-planning work should be arranged
without turning `TileComputeDAG` into a new scheduler,
global dataflow engine,
or side-channel protocol.

The first resource-planning task is explicitly
`TileComputeDAG`-backed.
The DAG is not considered production-complete while it only powers diagnostic
FFI and tests.
It must either feed typed resource demand /
typed unsupported reasons,
or be downgraded to diagnostic-only support surface.

## Current State

The current implementation has useful pieces, but they do not yet form a
mature hardware-aware resource planner.

- `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  and the live-form solver are the right semantic foundations.
  They should keep driving legality / query / typed plan decisions,
  but they are not a global memory allocator.
- `TileComputeDAG`
  is valid only as a pass-local compute-selection model.
  Its durable output must be typed plans and typed unsupported reasons,
  not a persisted DAG payload or a replacement scheduler.
  It also must not become a composite expression lowerer:
  input nodes must already be explicit TT-Metal leaf compute semantics in
  `Normalized Tile TIR`.
  Source hooks selected by DAG covering are leaf projections,
  not permission to expand one source node into several semantic compute ops.
  At current repo HEAD,
  DAG-wide fanout /
  materialization /
  unsupported covering reasoning is projected into
  `ResourceDemand`
  /
  `ResourcePressureReport`,
  and those reports are validated and projected into the executable spec.
  The DAG remains pass-local; the durable production surface is the typed
  report, not the DAG itself.
- CB allocation is currently the most concrete resource work.
  `PlanTTCBAlloc`
  already reasons about staged CB requirements,
  use intervals,
  CB IDs,
  auto-pops,
  and L1 bytes,
  but it still needs arch-aware limits and clearer pressure diagnostics.
- Core placement and buffer distribution are still basic.
  `PlanTTCoreGroups`
  uses a hard-coded logical worker grid,
  while `TTHardwareModel`
  already carries device-derived grid / worker / L1 / DRAM facts.
  `TTBufferDistributionPlan`
  is mostly `unit_mesh` + `replicated`,
  not a sharded / bank-aware placement result.
- Direct runtime admission remains a leaf backend subset.
  It must not define the capability of `TTProgram`
  or the shape of the resource planner.

## Research Inputs

The algorithmic analogies that fit this backend are narrow:

- CB ID allocation resembles register allocation.
  Chaitin-style interference coloring explains the correctness model:
  simultaneously live resources cannot share an ID.
  Poletto/Sarkar-style linear scan is the better first implementation shape:
  it is simpler, fast, and works directly from live intervals.
- L1/SRAM planning resembles scratchpad-memory pressure analysis,
  not a replacement for TT-Metal's allocator.
  TT-Metal already owns physical addresses,
  lock-step allocation,
  alignment,
  allocator-managed L1 buffers,
  and memory reports.
- Core placement and NoC-aware optimization resemble static task-graph
  partitioning / scheduling with communication weights.
  That belongs after core groups,
  buffer distribution,
  and resource pressure are represented explicitly.
- SDF-style static buffering is relevant for future bounded FIFO /
  publish-consume event sizing,
  but only after exact-CB event lifetime is explicit in typed plans.

## Direction Arrangement

The five directions are not a linear checklist.
They form a dependency-shaped roadmap.

### Direction 1: Constrain `TileComputeDAG`

Scope:

- pass-local compute legalization and covering only
- input from explicit preserved leaf tile compute semantics,
  `AccessRegion`,
  `LiveValueSSA`,
  and validated planning context
- output to typed
  `TTComputeOpPlan`,
  `TTCBPlan`,
  `TTLiveFormPlan`,
  `TTMaterializationPlan`,
  `TTConsumerBindingPlan`,
  or typed unsupported diagnostics

Non-scope:

- no global resource allocation
- no core placement
- no NoC scheduling
- no persisted DAG payload
- no composite-to-leaf lowering
- no leaf-looking composite payloads such as
  `exp2_tile(mode, ...)`
  or
  `mul_tiles_bcast_cols("div", ...)`
- no source-emitter branch maze that simply wraps the old per-op logic

Immediate requirement:

- before expanding covering,
  audit the current production hook surface and delete or simplify any
  cursor / `try_*` / fallback mechanics that do not change typed plans,
  diagnostics,
  or remove an old branch family.

Status:

- complete for the current production boundary.
  `PlanTTKernelABI`
  does not persist DAG covering decisions,
  the covering header does not expose a production DAG covering object,
  and explicit source emission remains a selected leaf-pattern projection.
  Further DAG work must continue to satisfy the same boundary tests.

### Direction 2: Add DAG-Backed `ResourceDemand` / `ResourcePressureReport`

This is the bridge between semantic planning and hardware resource admission.
It should be derived from validated `TTProgram`
and projected `ExecutableSpec`,
not carried as a side bag.
Its first production input is the pass-local
`TileComputeDAG`
covering result:
fanout,
live-share vs materialize decisions,
and unsupported materialization reasons.
This folds the correct part of Route A into the resource-pressure lane instead
of treating `TileComputeDAG` as a standalone production task.

First typed surface:

```text
ResourceDemand
  kernel
  core_group
  tile_compute_fanout_demands
  tile_compute_materialization_demands
  cb_requirements
  l1_buffer_requirements
  semaphore_requirements
  buffer_distribution_requirements
  communication_edges

ResourcePressureReport
  tile_compute_unsupported_reasons
  required_materializations
  per_core_cb_id_pressure
  per_core_cb_l1_bytes
  per_core_l1_buffer_bytes
  max_simultaneous_l1_bytes
  core_grid_requirement
  dram_view_requirement
  unsupported_reasons
```

Pay-rent rule:

- the report must drive validators or typed unsupported reasons;
  a dump-only report is foundation work, not completion.
- `TileComputeDAG`
  pays rent only when its fanout /
  materialization decisions change this report,
  a validator decision,
  or a typed unsupported reason.
  If a DAG decision is not consumed here,
  it must remain diagnostic-only and cannot be cited as production progress.

Status:

- complete for the first typed surface.
  `TTProgram`
  now carries first-class
  `TTResourceDemand`
  and `TTResourcePressureReport`
  fields.
  `PlanTTBlocks`
  captures the full pre-selection
  `TileComputeDAG`
  demand so fanout is not lost after builtin selection;
  later TTProgram planning phases refresh kernel,
  core,
  CB,
  semaphore,
  transport,
  and distribution counters without rebuilding the DAG from a reduced IR.
  `ValidateTTProgram`
  consumes the typed reports,
  requires matching demand/report entries,
  rejects typed resource-pressure unsupported reasons,
  and
  `MaterializeBlackholeExecutable`
  projects
  `resource_pressure_reports`.
  The remaining work is Direction 3:
  make the reported CB and L1 pressure hardware-aware and admission-relevant.

### Direction 3: Upgrade CB And L1 Admission

CB allocation should become an arch-aware live-interval allocator.

Required changes:

- get CB limit from target / hardware model / TT-Metal arch facts,
  not a stale fixed constant
- allocate architectural CB IDs with linear scan over live intervals
- model reserved / precolored ranges for conventional input,
  output,
  intermediate,
  and remote / state CB classes
- score or reject incompatible reuse by role,
  page size,
  page count,
  data format,
  flow class,
  publish / consume event shape,
  and initial reserve semantics
- emit pressure diagnostics before source emission

L1 admission should stay at pressure / overlap level first.
It should not assign physical addresses.

Required L1 checks:

- CB bytes per core range
- program-local CB space vs allocator-managed L1 buffer pressure
- worker L1 budget from `TTHardwareModel`
- TT-Metal lock-step / alignment waste estimate when the layout is known
- memory-report hooks for runtime validation when TT-Sim / TT-Metal exposes them

Status:

- complete for the first production admission gate.
  `TTHardwareModel`
  now carries `max_cb_count`
  and `l1_allocation_alignment_bytes`;
  `PlanTTCBAlloc`
  validates CB count and CB L1 usage against target-derived hardware facts;
  `TTResourcePressureReport`
  records CB limit,
  worker L1 budget,
  L1 alignment,
  raw and aligned CB bytes,
  L1 alignment waste,
  allocator-managed L1 buffer pressure,
  and max simultaneous L1 pressure;
  `ValidateTTProgram`
  rejects CB-id and L1 over-pressure before source / runtime emission.
  The remaining precision work is runtime memory-report cross-checking and
  richer reserved / precolored CB class modeling when a workload requires it.

### Direction 4: Make Core And Buffer Placement Hardware-Aware

Core placement must stop treating the Blackhole grid as a fixed source-level
constant.

Required first step:

- route `PlanTTCoreGroups`
  through `TTHardwareModel`
  for logical worker grid,
  functional worker count,
  worker L1 size,
  and DRAM view count

Placement should initially remain conservative:

- logical coordinates for safety under harvesting
- rectangular `CoreRange` / `CoreRangeSet`
  when the program model benefits from group APIs
- deterministic row-major work packets as the fallback policy
- typed unsupported reason when requested work exceeds available workers

`TTBufferDistributionPlan`
should then grow from `unit_mesh` / `replicated`
to explicit placement choices:

- interleaved DRAM
- interleaved L1
- height / width / block sharded L1
- host-visible vs device-local
- page size and shard shape
- core range attachment

This still should not become a physical address allocator.
TT-Metal owns the allocator.

### Direction 5: Add NoC / Multicast / Scheduling Optimization Later

Only after Directions 2-4 have typed evidence should the compiler add
communication-aware placement or scheduling.

Future inputs:

- `SpatialPlan.DataflowEdge`
  and `DependenceComponent`
  for producer / consumer / recurrence structure
- `ResourceDemand.communication_edges`
  for data movement amount and direction
- hardware topology / coordinate model
  for logical vs translated vs NoC proximity
- `TTBufferDistributionPlan`
  for interleaved / sharded / core-local data placement

Future algorithms:

- communication-weighted core-range selection
- multicast sender / receiver grouping
- NoC0 / NoC1 role-aware scoring
- bounded FIFO / exact-CB event sizing
- list scheduling or local graph partitioning for high-pressure kernels

This is explicitly not the first implementation step.
Adding it before CB/L1/core/buffer typed evidence exists would recreate the
same over-complexity problem under a different name.

## Integration With Current Backlog

The revised order is:

1. Add DAG-backed resource pressure reporting from existing typed
   `TTProgram`
   and
   `ExecutableSpec`
   records.
   Use pass-local
   `TileComputeDAG`
   fanout /
   materialization decisions as the first production input,
   and make validators / typed unsupported reasons consume the result.
2. Upgrade CB allocation and L1 admission using arch-aware limits and
   live-interval allocation.
   This is complete for the first production gate.
3. Replace hard-coded core grid and unit buffer placement with
   hardware-model-backed core groups and explicit buffer distribution choices.
4. Re-enter wider runtime admission:
   multi-block flash-attn,
   multi-page exact-CB events,
   mesh / distributed runtime,
   and later NoC / multicast optimization.

This means multi-block flash-attn direct-runtime admission should not be used
as the place to invent a resource planner.
It is the workload witness after the resource-planning surface is explicit.

## Completion Criteria

This roadmap is implemented only when:

- every resource-planning structure can state how it makes a DSL-authored
  kernel lower to real TT-Metal hardware code more reliably or more
  efficiently, by changing typed plans, validator decisions, admission
  diagnostics, or deleting old side channels
- `TileComputeDAG`
  remains pass-local and no downstream phase consumes it as owner truth
- `TileComputeDAG`
  fanout /
  materialization decisions feed
  `ResourceDemand`
  /
  `ResourcePressureReport`
  or typed unsupported reasons;
  diagnostic-only DAG covering is not production completion
- resource pressure changes validator decisions or typed unsupported reasons
- CB allocation uses arch-aware limits and live intervals
- L1 admission checks are visible before source / runtime emission
- core groups are derived from hardware model facts rather than hard-coded grid
- buffer distribution plans can express the memory placement choices needed by
  current admitted workloads
- no new bag / payload / helper wrapper carries resource truth across stages
