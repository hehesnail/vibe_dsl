# Blackhole Backend Architecture Design Overview

## Purpose

This document explains the architecture design of the TileLang Blackhole
backend.  It is meant for engineering orientation and for producing an
external architecture illustration from natural language.

The authoritative design remains
`tasks/dev_design/final_blackhole_backend_redesign.md`.  This document
intentionally describes architecture and lowering contracts only.

## 1. Engineering Design Objective

This project is building a real compiler backend from TileLang's normalized
TIR into Tenstorrent Blackhole / TT-Metal execution.  The design goal is not
to make isolated workload cases pass by adding late source matchers.  The goal
is to define a stable lowering contract that can carry complex frontend
programs into hardware execution without losing semantic ownership between
passes.

The backend is organized around this representation chain:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
  -> codegen/runtime
  -> TT-Metal hardware
```

The core engineering problem is that Blackhole execution is explicit and
protocol-heavy.  Correct codegen needs to know which logical values are live,
where data is materialized, which worker cores execute which work packets,
which circular buffers carry which tiles, when CB pages are produced and
consumed, which runtime arguments are per-work dynamic values, which
semaphores or remote endpoints synchronize producers and consumers, and which
resource constraints admit the program.

Those facts cannot be reconstructed safely from final source text, final TIR
shape, variable names, argument positions, or neighboring builtin calls.  A
frontend workload may contain nested loops, predicates, ragged bounds,
page-table lookups, loop-carried values, fused compute, split reductions, or
sharded placement.  If those facts are not represented explicitly before
runtime/codegen, the backend becomes a collection of fragile reverse
engineering heuristics.

The architecture therefore follows an IR-first rule:

- algorithmic semantics live in `Normalized Tile TIR`
- target-independent dataflow and lifecycle semantics live in `SpatialPlan`
- Blackhole hardware execution decisions live in `TTProgram`
- leaf runtime/codegen inputs live in `ExecutableSpec`
- validators reject missing contracts before source or runtime guessing

This is the practical purpose of the IR layers.  Each layer freezes a
different kind of information, establishes invariants, and prevents later
stages from recovering semantics through side channels.

## 2. Hardware Architecture And IR Fit

### Hardware Dataflow Model

TT-Metal exposes Blackhole as a dataflow-oriented execution target, not as a
single GPU-style SPMD kernel.

Host code builds a TT-Metal `Program`, allocates buffers, creates circular
buffers, creates kernels, sets compile-time and runtime arguments, and assigns
work to physical worker cores.  Each worker core has data movement and compute
roles.  Reader kernels move data between DRAM, NoC, L1, and circular buffers.
Compute kernels operate on tiles already visible in CBs using TT-Metal leaf
tile APIs such as `matmul_tiles`, unary SFPU ops, binary tile ops, copy,
pack, tilize, and untilize.  Writer kernels consume output CBs and write
results to DRAM or another resident buffer.

Circular buffers are L1-backed FIFOs.  Their correctness depends on physical
CB IDs, page sizes, page counts, data formats, reserve/push/wait/pop event
ordering, and producer/consumer visibility.  Multi-core execution adds a
separate work-mapping problem: logical work can exceed resident worker count,
so physical cores need explicit work packets and per-core runtime arguments.
Sharded L1 buffers, interleaved DRAM buffers, page-addressed DRAM transport,
semaphores, NoC routes, multicast, and remote endpoints all have different
hardware-facing contracts.

This hardware model directly motivates the IR split.

### Normalized Tile TIR

`Normalized Tile TIR` is the last target-independent program representation
that still looks like the authored algorithm.

It owns:

- loops, predicates, and block domains
- buffer loads and stores
- access expressions and local dataflow
- explicit tile-level compute intent
- normalized TT-Metal leaf tile compute when the primitive is expressible

It answers:

```text
What does the program compute, and which logical elements does it read/write?
```

It deliberately does not answer:

```text
Which core, which CB ID, which semaphore, which launch order, or which runtime
ABI slot?
```

The layer exists to preserve algorithmic meaning before Blackhole execution
choices are frozen.  It also prevents a common failure mode: destroying tile
compute into scalar loops before the backend has proven the corresponding
TT-Metal leaf operations.

### SpatialPlan

`SpatialPlan` is the virtual spatial/dataflow program derived from normalized
TIR.

It owns:

- execution units
- access regions
- closure boundaries
- dataflow, carry, reduction, broadcast, and join edges
- dependence components
- phase plans and ordering evidence
- logical live values and live-value edges
- materialization boundaries
- target-independent layout evidence and validated hints

It answers:

```text
Which logical values flow between which execution units, through which access
regions, across which phases and materialization boundaries?
```

It deliberately does not answer:

```text
Which TT builtin family, physical core, CB ID, semaphore ID, runtime arg, or
executable layout?
```

This layer is needed because complex workloads are not just flat compute.
Paged attention, ragged copies, grouped feeds, recurrent state, split blocks,
and reductions all need logical value flow and lifecycle evidence before the
backend can choose a hardware protocol.  Without `SpatialPlan`, lower stages
would have to rediscover virtual dataflow from source shape or names.

### TTProgram

`TTProgram` is the Blackhole target execution contract.  It is the main
hardware-facing IR layer.

It owns:

- Blackhole hardware model facts used by planning
- mesh and device placement
- physical core groups and logical work packets
- block plans and kernel plans
- reader / compute / writer kernel records
- TT-Metal leaf compute op plans
- live-form, materialization, and consumer-binding plans
- buffer distribution and tensor memory config plans
- sharding, placement resolution, and reshard records
- CB plans, physical CB allocation, exact-CB lifecycle, and queue events
- transport, accessor, sync, semaphore, and remote endpoint plans
- compile-time args, runtime args, common runtime args, and per-work ABI
- execution / launch plans
- resource demand, resource pressure, and typed unsupported reasons

It answers:

```text
How will this spatial/dataflow program execute on Blackhole hardware?
```

It deliberately does not answer:

```text
What is the original target-independent algorithm, or how should a runtime
repair missing planning facts?
```

This is the PTX-like layer in the design: not source text, not a helper bag,
and not a final runtime object.  It is a stable target execution contract that
source generation, runtime, and hardware admission can trust.

### ExecutableSpec

`ExecutableSpec` is the leaf projection consumed by source generation,
direct runtime, module serialization, and TT-Sim validation.

It owns:

- executable schema version and entry identity
- projected kernel and segment records
- source materialization inputs
- CB configs with physical CB IDs
- core plans and work-packet records
- buffer distribution and tensor memory config specs
- runtime args, common runtime args, and per-work arg specs
- accessors and formal buffer identities
- compute op records
- semaphore bindings and remote core descriptors
- physical CB queue events
- admission reasons and runtime-module build inputs

It answers:

```text
What exact typed manifest do codegen and runtime consume?
```

It deliberately does not answer:

```text
How should the target be planned, how should resources be allocated, or how
should missing semantics be recovered from source?
```

This layer exists to make leaf consumers boring.  Runtime/codegen may validate
the manifest and reject unsupported cases, but they should not become
planners.

## 3. Lowering And Execution Path

The current implementation realizes the design through a sequence of passes.
Passes are implementation vehicles; the representation layers above are the
stable architecture.

### Frontend And Normalized Tile TIR

TileLang frontend lowering produces TIR with TileLang block structure, buffer
accesses, loop domains, predicates, and tile-level compute intent.

Before Blackhole target planning, generic and Blackhole-specific
normalization make device-private resources and tile compute explicit.
`NormalizeBlackholeTileCompute` rewrites admissible local tile compute into
explicit Blackhole leaf tile-compute calls.  `ValidateBlackholeTileComputeNormalized`
rejects compute-buffer scalar residue that cannot be represented as stable
TT-Metal leaf compute.

The output of this stage is normalized TIR that still describes the program,
not the hardware schedule.

### SpatialPlan Construction

`BuildSpatialPlan` reads the normalized TIR and derives the virtual dataflow
program.  It collects executable statements and closure candidates, builds
execution units, derives access regions, records dataflow and local value
flows, computes phase plans, derives live values and live-value edges, and
places materialization boundaries.

`ValidateSpatialPlan` checks that this virtual program is internally
consistent before target planning consumes it.

`SplitBlackholeKernel` remains as a historical normalization hook in the
pipeline.  It must not be treated as a long-term protocol owner.  Segment
truth belongs to `TTProgram` kernel records and their executable projection.

### TTProgram Target Planning

The TT planning belt incrementally fills `TTProgram`.

`PlanTTBlocks` anchors Blackhole hardware model facts, mesh plans, core
groups, block plans, work packets, and resource demand.

`SelectBlackholeTTMetalBuiltins` chooses the exact TT-Metal leaf builtin
surface that later compute planning may use.

`PlanTTCompute` creates the compute-facing target contract: typed kernel
records, kernel plans, staged CB requirements, ABI plans, live-form plans,
materialization plans, consumer bindings, exact-CB virtual values, use
events, live intervals, allocations, release events, and TT-Metal compute op
plans.

`PlanTTTransport` turns staged CB requirements into physical transport.  It
allocates physical CB IDs, rewrites requirement indices to `cb_id`s, refreshes
`TTKernel.queue_events`, remaps materialization / exact-CB / compute operand
references through the physical CB plans, and builds transport plans from
`SpatialPlan`.

`PlanTTSync` adds compute synchronization, sync plans, and semaphore plans.

`PlanTTABI` adds the runtime-facing data ABI: destination layout plans,
buffer distribution plans, tensor memory config plans, accessor specs,
sharding contracts, placement resolution, and reshard plans.  This is where
logical buffer access becomes a hardware addressability contract.

`PlanTTExecution` adds launch and execution plans from the validated spatial
program and kernel records.

`BuildTTProgram` seals the target contract.  It requires the staged slices to
exist and line up, then strips temporary intermediate attrs so downstream
consumers see `TTProgram` as the owner truth.

`ValidateTTProgram` checks the target execution contract against
`SpatialPlan` and the Blackhole hardware model.  Missing typed owner-truth,
invalid compute op names, impossible resource pressure, inconsistent CB
lifecycle, and unsupported placement must fail here rather than in runtime.

### Executable Projection

`MaterializeBlackholeExecutable` projects validated `TTProgram` into the
`tl.blackhole_executable` attribute.  The projection serializes the target
contract into leaf-readable schema: mesh plans, buffer distributions, tensor
memory configs, placement records, compute ops, segment records, CB configs,
core plan, semaphore plan, live/materialization/exact-CB records, queue
events, runtime args, accessors, and resource reports.

This pass is not a planner.  It should only project existing typed owner truth
or fail.

### Codegen, Runtime, And Hardware Execution

Source generation consumes `ExecutableSpec` / `KernelSpec` records to emit
TT-Metal reader, compute, and writer kernel source.  It should read projected
kernel bodies, compile-time args, runtime arg schema, accessors, compute ops,
CB configs, and queue events.  It should not rescan final TIR or generated
source to infer segment membership or CB order.

`BlackholeModule` consumes the same executable projection for direct runtime.
It validates executable schema, materializes host/runtime buffer contracts,
builds TT-Metal program objects, creates circular buffers, creates kernels,
sets compile-time and runtime args, assigns per-core work packets, and admits
or rejects the executable before launch.

On hardware or TT-Sim, the projected protocol becomes concrete dataflow:
reader kernels move DRAM or sharded/L1 resident data into CB pages, compute
kernels wait on input CBs and publish output CBs, writer kernels flush output
CBs, semaphores and remote descriptors synchronize cross-core communication,
and runtime admission ensures that resource and schema constraints were
explicit before execution.

## 4. Showcase: Paged/Ragged Attention Decode

This showcase demonstrates how one complex workload carries important
information through all layers without becoming a special frontend opcode or a
runtime recovery path.

### Normalized Tile TIR Information

The authored program contains ordinary TIR evidence:

- buffers for Q, paged K cache, paged V cache, page table, cache sequence
  lengths, and output
- logical loops over batch, query head, page step, row, and tile dimensions
- `PageTable[sequence, page]` loads that select K/V cache pages
- `CacheSeqLens[sequence]` loads that guard valid rows inside a page
- predicates that zero-fill or skip invalid page rows
- explicit leaf tile compute for score GEMM, max/logsum update,
  exponentiation/scaling, value accumulation, and output update
- buffer stores for the final output tile

At this layer, page selection is still an index expression, raggedness is
still a predicate, and attention math is still explicit tile-level compute.

### SpatialPlan Information

`SpatialPlan` turns the same TIR into virtual dataflow:

- execution units for page readers, score/softmax compute, value accumulation,
  and output writing
- access regions for Q, page table, cache length, K cache pages, V cache
  pages, and output
- dataflow edges from page selection to K/V materialization
- predicate evidence for valid-row bounds
- live values for K tiles, V tiles, score tiles, running max, logsum,
  partial output, and final output
- materialization boundaries where DRAM-backed K/V page data must become
  compute-compatible tile live forms
- carry edges for online softmax state across page steps

There is still no physical CB ID, no core coordinate, and no TT-Metal runtime
arg slot.  The layer only states the logical dataflow and lifecycle.

### TTProgram Information

`TTProgram` turns that virtual program into Blackhole execution protocol:

- core groups and work packets for logical sequence/head/page work
- reader, compute, and writer kernel records
- per-work `value_expr` bindings for page starts and valid-row bounds
- buffer distribution records for interleaved DRAM page-addressed K/V cache
  and output buffers
- accessor and transport plans for page-addressed DRAM reads
- live-form and materialization plans that turn K/V pages into
  compute-compatible CB-backed tile streams
- CB plans for Q, K, V, score/state, partial output, and final output live
  forms
- physical CB IDs and exact-CB virtual values
- exact-CB use events, live intervals, allocations, release events, and
  structured `TTKernel.queue_events`
- TT-Metal leaf compute op plans for matmul, reductions, unary/binary tile
  ops, and pack/copy operations
- runtime/common/per-work ABI records consumed by reader, compute, and writer
  kernels
- resource pressure records for CB count, L1 bytes, core usage, and admitted
  synchronization

At this layer, the important hardware protocol is explicit.  A later consumer
does not need to guess that a page-table load controls K/V addressability or
that a CB page belongs to a loop-carried softmax state.

### ExecutableSpec Information

`ExecutableSpec` projects the planned protocol into the leaf manifest:

- kernel records for reader, compute, and writer
- runtime args and per-work arg specs for page starts, row bounds, work
  coordinates, and formal buffer addresses
- accessors for page-addressed K/V reads and output writes
- CB configs with physical `cb_id`, page size, page count, role, format, and
  requirement ownership
- physical `KernelSpec.queue_events` for reserve/push/wait/pop replay
- compute op records for codegen/runtime admission
- materialization records for K/V page tiles and output live forms
- buffer distribution specs for interleaved DRAM and any resident L1 views
- semaphore or remote descriptor records if the chosen plan needs them
- typed admission reasons if the backend cannot execute the requested shape

Runtime and codegen consume this manifest directly.  They do not reload
`PageTable`, classify `CacheSeqLens` by name, infer CB roles from suffixes,
or scan generated source to reconstruct queue ordering.

### Hardware Execution View

The executable becomes a TT-Metal dataflow program:

- host runtime creates buffers, CBs, kernels, and per-core runtime args
- reader kernels evaluate projected per-work bindings and read selected K/V
  pages into input CBs
- compute kernels wait on Q/K/V CBs, run the explicit TT-Metal leaf tile
  sequence for score, online-softmax, and value accumulation, then publish
  output CB pages
- writer kernels consume output CB pages and write final tiles
- queue events, semaphores, and resource admission are checked from the
  projected records, not inferred from emitted source

This is the intended pattern for complex workloads: carry generic TIR
evidence forward into explicit spatial and target records, then hand a typed
manifest to leaf consumers.
