# Blackhole Architecture And Progress Overview

## Purpose

This document is a natural-language brief for explaining the current
TileLang Blackhole backend architecture and for generating an external
architecture/progress illustration.

The authoritative design is
`tasks/dev_design/final_blackhole_backend_redesign.md`.  The live execution
board is `tasks/progress.md`.  This file should not become another design
contract or progress log.

## Image Brief

Draw the backend as a compiler pipeline that has moved from fragile late
recovery to an explicit IR-first execution contract.

The main path should read as:

`Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec -> Blackhole runtime/codegen`.

The middle of the image should not be empty.  Between the layer boxes, show a
thin "pass belt" that explains how the implementation moves facts forward:
spatial analysis builds and validates `SpatialPlan`; TT planning passes fill
`TTProgram`; one projection pass materializes `ExecutableSpec`; leaf runtime
and codegen only consume that projection.

The visual contrast should be clear: the bright main path carries typed owner
truth, while faded retired side channels sit below it as discarded routes:
generated source scans, final TIR body scans, name/argument-position recovery,
payloads, fact bags, helper maps, workload-shaped schemas, and fallback
defaults.

## Architecture Story

The original failure mode was not a single missing case.  It was that target
execution semantics escaped the IR chain.  Some cases worked because later
passes or runtime code could rediscover enough information from names, builtin
neighborhoods, argument order, helper attrs, or source text.  That breaks down
as soon as frontend workloads become irregular, fused, paged, split, or
loop-carried.

The current architecture treats the compiler as a sequence of explicit
contracts.

`Normalized Tile TIR` is still the authored program shape.  It owns loops,
predicates, buffer reads and writes, explicit tile operations, access
expressions, and the normalized TT-Metal leaf compute surface when the compute
can legally be represented.

`SpatialPlan` is the target-independent virtual spatial/dataflow program.  It
owns execution units, access regions, closure boundaries, dataflow edges,
dependence components, phase plans, live values, live-value edges,
materialization boundaries, layout evidence, and tensor placement intent.

`TTProgram` is the Blackhole target execution contract.  It owns the hardware
realization: mesh and core groups, block plans, kernel records, compute op
plans, live-form and materialization plans, physical CB allocation, exact-CB
lifecycle, transport plans, semaphore and remote sync plans, buffer
distribution, tensor memory config, sharding and placement records, runtime
ABI, per-work ABI, launch/execution plans, resource pressure, and typed kernel
queue events.

`ExecutableSpec` is the leaf projection and admission schema.  It is produced
from `TTProgram` once, then read by source generation, direct runtime, module
serialization, and TT-Sim validation.  If a required executable record is
missing, the backend should reject the program with a typed reason instead of
rebuilding planner semantics from source or final TIR.

## Current Pass Flow

The pass names are implementation details, not durable architecture
boundaries.  They are still important for the diagram because they explain how
the current code gets from one representation to the next.

Before the Blackhole-specific chain, the generic TileLang lowering path
canonicalizes device-private resources and normalizes Blackhole tile compute.
`NormalizeBlackholeTileCompute` and
`ValidateBlackholeTileComputeNormalized` make local tile compute explicit and
reject scalar compute-buffer residue that cannot be represented as a stable
leaf compute primitive.

The `SpatialPlan` section starts with `BuildSpatialPlan`.  That pass walks the
current normalized TIR, collects executable statements and closure candidates,
derives access regions and dataflow, computes phase and materialization
boundaries, and attaches `tl.spatial_plan`.  `ValidateSpatialPlan` is the
first fail-closed gate: downstream TT planning should see a validated spatial
program, not a bundle of opportunistic analysis facts.  `SplitBlackholeKernel`
remains as a historical normalization hook; it no longer emits segment-kind
markers as cross-pass protocol.

The `TTProgram` section is a staged target-planning belt.  `PlanTTBlocks`
anchors hardware model, mesh, core-group, block-plan, and resource-demand
facts.  `SelectBlackholeTTMetalBuiltins` chooses the exact TT-Metal builtin
surface before compute planning.  `PlanTTCompute` creates typed kernel records
and compute-facing owner truth: kernel plans, kernel bodies, staged CB
requirements, ABI plans, live-form records, materialization records, consumer
bindings, exact-CB virtual values, exact-CB use events, live intervals,
allocations, release events, and compute op plans.

`PlanTTTransport` then turns staged CB requirements into physical CB
allocation and transport truth.  It rewrites CB requirement indices to final
physical `cb_id`s, refreshes the typed `TTKernel.queue_events`, remaps
materialization / exact-CB / compute operand references to the physical CB
plans, and builds transport plans from `SpatialPlan`.  This is the stage that
prevents runtime/codegen from needing to infer queue behavior from generated
source text.

`PlanTTSync` adds compute synchronization and semaphore plans.  `PlanTTABI`
adds the runtime-facing data ABI: destination layout, buffer distribution,
tensor memory config, accessor specs, sharding contracts, placement
resolution, and reshard plans.  `PlanTTExecution` adds execution and wave
launch plans.  `BuildTTProgram` is the sealing step: it requires the staged
slices to exist and line up, then strips temporary intermediate attrs so the
remaining public target contract is `TTProgram`.  `ValidateTTProgram` checks
that the target execution contract is internally consistent with
`SpatialPlan` and the Blackhole hardware model.

The executable section is intentionally narrower.  `MaterializeBlackholeExecutable`
projects `TTProgram` into the `tl.blackhole_executable` map with schema
version, source marker, mesh plans, buffer distributions, tensor memory
configs, placement records, compute ops, segment records, CB configs, core
plan, semaphore plan, live/materialization/exact-CB records, and resource
reports.  Runtime and codegen leaf readers use `ExecutableSpec` / `KernelSpec`
records such as runtime args, common runtime args, per-work args, accessors,
semaphore bindings, remote core descriptors, compute ops, launch specs, and
queue events.

At the far right, `BlackholeModule`, code generation, direct host runtime, and
TT-Sim correctness gates are consumers.  They may validate executable schema
and reject unsupported forms, but they should not become planners.

## What The Diagram Should Emphasize

Make the four representation layers visually larger than the pass names.  The
passes should look like conveyors, gates, or annotations between layers.  The
reason is architectural: passes can be renamed, split, or merged, but the
stable contracts are `SpatialPlan`, `TTProgram`, and `ExecutableSpec`.

Show validators as gates on the main path:
`ValidateBlackholeTileComputeNormalized`, `ValidateSpatialPlan`,
`ValidateTTProgram`, and executable admission checks.  The gates should imply
fail-closed behavior: missing target facts are errors, not invitations for
runtime recovery.

Show `TTProgram` as the heaviest middle object.  It is the PTX-like target
execution contract for this backend: not final source text, but the stable
description of how the program should execute on Blackhole.

Show `ExecutableSpec` as a projection, not a second planner.  It should look
like a typed manifest handed to runtime/codegen.

## Progress Summary

Progress should be a small legend, not the body of the diagram.

Completed: foundation work, P0 target execution contract hardening, and T8
irregular/indexed access are complete for the current admitted surface.  These
lanes removed the major source/body/name recovery paths and moved covered
facts into typed `TTProgram -> ExecutableSpec` records.

Active: P1 / T9 workload-first expansion is in progress.  The current active
boundary is T9.6 multi-block flash decode, where split blocks need explicit
exact-CB publish/consume and partial-combine contracts.

Queued: P2 / T10 distributed production remains future work: typed
multi-device placement, collectives, NoC/multicast/global scheduling, and
production partial-K reduction.

Do not draw the progress state as a long checklist of every admitted case.
The diagram should show direction and maturity: solid completed base lanes,
one highlighted active T9.6 lane, and one queued distributed-production lane.

## Short Caption

TileLang Blackhole is converging on an explicit compiler contract.  Normalized
Tile TIR captures program structure, `SpatialPlan` captures virtual dataflow,
`TTProgram` captures Blackhole execution protocol, and `ExecutableSpec` feeds
runtime/codegen.  The middle passes are now supposed to move typed facts
forward and validate them, not leave holes for leaf recovery.  Current progress
has completed the foundation, P0 contract hardening, and T8 irregular access;
the active frontier is T9.6 multi-block flash decode.
