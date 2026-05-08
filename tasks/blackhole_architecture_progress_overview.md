# Blackhole Architecture And Progress Overview

## Purpose

This document is a natural-language brief for explaining the TileLang
Blackhole backend architecture and for generating an external architecture
illustration.

The authoritative design is
`tasks/dev_design/final_blackhole_backend_redesign.md`.  The live execution
board is `tasks/progress.md`.  This file is a compact explanation, not a
second design contract and not a progress log.

## Image Brief

The picture should show two aligned stories.

The lower story is the hardware dataflow: host buffers in DRAM, work assigned
to Blackhole worker cores, data-movement kernels pulling tiles through NoC
into L1 circular buffers, compute kernels consuming and producing CB pages,
writer kernels flushing results, and synchronization/admission gates around
CB queues, semaphores, remote endpoints, and runtime args.

The upper story is the compiler contract that makes that hardware dataflow
deterministic:

`Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec -> Blackhole runtime/codegen`.

Draw the four representation layers as the stable architecture.  Draw pass
names as smaller conveyors or gates between layers.  The pass names matter
for current implementation orientation, but the durable design is the IR
contract, not any individual pass.

The most important visual contrast is between the bright typed path and the
faded retired recovery paths: source scans, final TIR body scans, name or
argument-position inference, payloads, facts bags, workload-shaped schemas,
debug-name CB lookup, and runtime fallback defaults.

## Hardware Dataflow Starting Point

Blackhole execution is not GPU-style "one SPMD kernel owns everything."
The TT-Metal programming model exposes a small dataflow machine per worker
core:

- Host code creates a `Program`, buffers, circular buffers, kernels, compile
  args, runtime args, and per-core work assignment.
- Reader/data-movement kernels run on data-movement processors and move
  tiles between DRAM, NoC, L1, and circular buffers.
- Compute kernels run on compute processors and operate on tiles already
  visible in CBs.  Leaf compute such as `matmul_tiles`, unary SFPU ops,
  binary ops, pack/unpack, and tile copies are expressed at TT-Metal tile
  API granularity.
- Writer/data-movement kernels consume output CBs and write results back to
  DRAM or another resident buffer.
- Circular buffers are L1-backed FIFOs.  Correctness depends on exact page
  size, page count, data format, physical CB ID, reserve/push/wait/pop order,
  and producer/consumer visibility.
- Multi-core execution is a mapping problem: logical work items may be split
  over physical cores through work packets, while each resident core reuses
  its own L1 and CB resources across its assigned temporal work.
- Sharded L1 buffers and interleaved DRAM buffers have different address
  contracts.  Sharding reduces or removes NoC movement only when core grid,
  shard shape, orientation, layout strategy, and source/materialization
  boundaries are explicit.
- More advanced paths add semaphores, remote core descriptors, multicast/NoC
  routes, and ordered partial reductions.  Those are execution protocol, not
  comments on generated source text.

This is why late recovery fails.  A source emitter can see a `cb_wait_front`
or a buffer name, but it cannot safely reconstruct which logical value is
alive, which page-table expression selected the source row, whether a CB page
belongs to a loop-carried value, whether a physical CB ID was reused after a
release, or which remote endpoint owns a semaphore.  Those facts must already
be represented before runtime/codegen consumes the program.

## Why The IR Layers Exist

Each layer pays for itself by freezing the right information at the right
time and leaving later stages with less guessing to do.

`Normalized Tile TIR` exists to keep algorithmic meaning before target
execution choices are made.  It can cover loops, predicates, buffer
loads/stores, access expressions, local dataflow, and TT-Metal leaf tile
compute normalization.  It should answer "what does the program compute and
which logical elements does it touch?"  It should not answer "which CB ID,
which core, which semaphore, or which runtime ABI slot?"

`SpatialPlan` exists because complex workloads need a target-independent
dataflow model before hardware placement.  It can cover execution units,
access regions, closure boundaries, dataflow/carry/reduction/broadcast/join
edges, dependence components, phase plans, live values, live-value edges,
materialization boundaries, layout evidence, and validated hints.  It should
answer "which logical values flow between which units, under which access
regions and phase boundaries?"  It should not decide TT builtins, CB IDs,
runtime args, or launch order.

`TTProgram` exists because TT-Metal hardware execution needs a stable target
contract before source and runtime.  It can cover hardware model facts,
mesh/core groups, work packets, block plans, reader/compute/writer kernel
records, compute op plans, live forms, materialization, consumer bindings,
physical CB allocation, exact-CB lifecycle, queue events, transport plans,
buffer distributions, tensor memory configs, sharding and placement records,
semaphores, remote sync endpoints, runtime/common/per-work ABI, execution
plans, and resource pressure.  It should answer "how will this spatial
program execute on Blackhole?"  This is the PTX-like target execution
contract for this backend.

`ExecutableSpec` exists to make leaf consumers boring.  It can cover schema
version, entry identity, kernel/segment records, source materialization
inputs, projected CB configs, core plans, buffer distribution records,
accessors, compute op records, runtime args, common runtime args, per-work
args, semaphore bindings, remote core descriptors, physical queue events,
admission reasons, and serialized module inputs.  It should answer "what
exact typed manifest does runtime/codegen consume?"  It should not plan,
repair, or rediscover missing semantics.

## What The Layers Can Cover

For image generation, the useful mental model is a set of hard hardware
questions flowing across the layers.

Work partitioning starts as logical loops and kernel domains in
`Normalized Tile TIR`, becomes execution units and phases in `SpatialPlan`,
then becomes physical core groups and work packets in `TTProgram`, and is
finally emitted as core plans and per-core runtime args in `ExecutableSpec`.

Data access starts as buffer loads, stores, predicates, and index expressions
in `Normalized Tile TIR`, becomes `AccessRegion` evidence in `SpatialPlan`,
then becomes buffer distribution, accessor, transport, materialization, and
per-work `value_expr` ABI records in `TTProgram`, and is finally consumed as
runtime args, accessors, and buffer distribution specs in `ExecutableSpec`.

Compute starts as explicit tile-level algorithmic operations in
`Normalized Tile TIR`, becomes logical producer/consumer live values in
`SpatialPlan`, then becomes TT-Metal leaf `TTComputeOpPlan` records in
`TTProgram`, and is projected as `KernelSpec.compute_ops` for codegen and
runtime admission.

FIFO/lifetime correctness starts as value flow and materialization need in
`SpatialPlan`, becomes exact-CB virtual values, use events, live intervals,
allocations, release events, physical `TTCBPlan` records, and
`TTKernel.queue_events` in `TTProgram`, then becomes physical CB configs and
`KernelSpec.queue_events` in `ExecutableSpec`.

Synchronization starts as ordering and cross-phase evidence in `SpatialPlan`,
becomes semaphore plans, compute sync plans, remote core descriptors, and
execution plans in `TTProgram`, then becomes leaf semaphore bindings and
remote endpoint records in `ExecutableSpec`.

Resource legality starts as tile compute and materialization demand, becomes
CB count, CB-backed L1 bytes, allocator-managed L1 bytes, core-grid pressure,
semaphore pressure, buffer distribution pressure, and typed unsupported
reasons in `TTProgram`, then becomes executable admission rather than a
runtime surprise.

This coverage is the value of the IR chain.  It lets new workload surfaces be
expressed as ordinary TIR evidence plus typed target records, instead of
adding a new source matcher or workload-specific runtime schema each time.

## Example Transformation: Paged/Ragged Attention

Use this as the main showcase in the diagram.  It is specific enough to show
why the layers matter, but it is still an example of generic evidence flowing
through the backend rather than a special frontend op.

In `Normalized Tile TIR`, a paged/ragged decode tile is just ordinary program
structure: page-table loads select K/V cache pages, cache-length loads guard
valid rows, loops describe the static page steps, predicates zero or skip
invalid rows, and explicit leaf tile ops describe score computation,
online-softmax partial combine, and output accumulation.

`SpatialPlan` turns that into target-independent evidence.  Page-table and
cache-length reads become access regions and value-flow evidence, not a
`paged_decode` opcode.  Reader, compute, and writer regions become execution
units.  The K/V cache page materializations, score tiles, softmax state,
partial output, and final output become live values and dataflow edges with
phase/materialization boundaries.

`TTProgram` turns the same evidence into Blackhole execution protocol.
It assigns physical work packets and kernels, chooses reader/compute/writer
roles, records per-work `value_expr` bindings for page starts and valid-row
bounds, describes K/V page-addressed DRAM transport, materializes
compute-compatible live forms through CBs, assigns physical CB IDs, records
exact-CB publish/consume/release lifetimes, emits queue events, attaches
compute op plans for the TT-Metal leaf tile sequence, and records runtime
ABI/accessor/buffer-distribution facts.

`ExecutableSpec` is then a typed manifest.  Runtime/codegen see kernel
records, runtime args, per-work specs, accessors, CB configs, physical queue
events, compute ops, materialization records, and buffer distribution specs.
They do not reload `PageTable`, guess from `CacheSeqLens` names, infer CB
roles from suffixes, or scan generated source to rebuild queue order.

The same shape explains the active T9.6 boundary.  Multi-block flash decode
is harder because split blocks introduce more exact-CB publish/consume and
partial-combine obligations.  The desired solution is not a special
multi-block emitter; it is to make those extra lifetimes and combine records
first-class in `TTProgram` and projected into `ExecutableSpec`.

## Current Pass Flow

The current implementation realizes the architecture with this pass belt.
The pass belt should be visible in the diagram, but visually smaller than the
IR layers.

Before the Blackhole-specific chain, generic lowering canonicalizes
device-private resources and normalizes Blackhole tile compute.
`NormalizeBlackholeTileCompute` and
`ValidateBlackholeTileComputeNormalized` make local tile compute explicit and
reject scalar compute-buffer residue that cannot be represented as a stable
leaf compute primitive.

`BuildSpatialPlan` derives execution units, access regions, dataflow,
phases, live values, and materialization boundaries from normalized TIR.
`ValidateSpatialPlan` is the first fail-closed gate.  `SplitBlackholeKernel`
is now only a historical normalization hook; it no longer emits
segment-kind markers as protocol.

The `TTProgram` belt fills target realization in stages.
`PlanTTBlocks` anchors hardware model, mesh, core groups, block plans, and
resource demand.  `SelectBlackholeTTMetalBuiltins` chooses the admitted leaf
builtin surface.  `PlanTTCompute` creates typed kernel records, staged CB
requirements, ABI plans, live-form/materialization records, exact-CB
lifecycle records, and compute op plans.

`PlanTTTransport` turns staged CB requirements into physical CB allocation,
rewrites requirement indices to physical `cb_id`s, refreshes
`TTKernel.queue_events`, remaps exact-CB/materialization/compute operand
references to physical CB plans, and builds transport plans from
`SpatialPlan`.

`PlanTTSync` adds compute synchronization and semaphore plans.  `PlanTTABI`
adds destination layout, buffer distribution, tensor memory config, accessor
specs, sharding contracts, placement resolution, and reshard plans.
`PlanTTExecution` adds launch/execution plans.  `BuildTTProgram` seals the
target contract and strips temporary intermediate attrs.  `ValidateTTProgram`
checks that the target execution contract is consistent with `SpatialPlan`
and the hardware model.

`MaterializeBlackholeExecutable` projects validated `TTProgram` into
`tl.blackhole_executable`.  `BlackholeModule`, source generation, direct
runtime, serialization, and TT-Sim gates consume that executable projection
as leaf readers.

## Visual Emphasis

Use one large center object for `TTProgram`.  It should look like the hardware
execution contract: core/work mapping, reader/compute/writer kernels, CB
queues, buffer distributions, semaphores, remote descriptors, runtime ABI,
and resource pressure all meet there.

Use `SpatialPlan` as the bridge from algorithm to target planning.  It should
look graph-like: execution units, access regions, and live-value edges.

Use `ExecutableSpec` as a typed manifest, not as a planner.  It should be
drawn next to runtime/codegen as the thing they read.

Progress should be a small legend only:

- Completed: foundation, P0 target execution contract hardening, and T8
  irregular/indexed access for the current admitted surface.
- Active: P1 / T9 workload expansion, with T9.6 multi-block flash decode as
  the current boundary.
- Queued: P2 / T10 distributed production, including multi-device placement,
  collectives, NoC/multicast/global scheduling, and production reducers.

## Short Caption

TileLang Blackhole is being organized around the hardware dataflow that
TT-Metal actually exposes: DRAM/L1 buffers, circular-buffer FIFOs,
reader/compute/writer kernels, per-core work packets, runtime ABI,
semaphores, and resource admission.  `Normalized Tile TIR` preserves program
meaning, `SpatialPlan` captures virtual dataflow, `TTProgram` captures
Blackhole execution protocol, and `ExecutableSpec` is the typed manifest
runtime/codegen consume.  The design removes late semantic recovery and makes
complex workloads advance by adding explicit IR evidence and target records,
not by adding new source matchers or runtime guesses.
