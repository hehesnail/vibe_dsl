# Blackhole TTProgram Target Execution Contract

## Role

This document defines the active architecture task for hardening `TTProgram`
into the Blackhole target-facing execution contract.

It is not a second overall design document and not a fifth IR layer.  The
durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status lives in `tasks/progress.md`.

## Problem

Current admitted T7/T8/T9 workloads show the same failure mode in different
forms: when `TTProgram` does not own a target execution fact, runtime/codegen
reconstructs it from source text, segment bodies, builtin ordering, argument
names, helper maps, or workload-shaped branches.

That makes individual cases pass while the backend remains brittle for more
complex workload structure.

The fix is not a new IR layer.  The fix is to make `TTProgram` the stable
target execution contract and keep `ExecutableSpec` as a leaf projection of
that contract.

## Goal

For every target execution fact consumed by source/codegen/runtime:

- owner truth is represented by typed `TTProgram` objects or fields;
- `ExecutableSpec` serializes only validated projection;
- leaf readers consume projected records and fail closed if records are
  missing or inconsistent;
- old source/body/name/runtime recovery paths are deleted.

## Non-Goals

- No new PTX-like fifth IR object.
- No workload-specific schema for flash, GQA, MLA, scan, sparse, routed, or
  grouped GEMM.
- No parallel runtime/codegen path.
- No compatibility wrapper around retired payload, facts bag, bridge attr, or
  source scanner surfaces.

## Owned Surfaces

The hardening task covers target-facing execution protocol surfaces that are
already visible to leaf consumers:

- physical CB identity and FIFO queue events;
- exact-CB value lifecycle, allocation, release, and latest-producer rules;
- semaphore and remote synchronization endpoints;
- kernel/segment identity, core type, launch, and execution ordering;
- runtime/common/per-work ABI bindings;
- resource pressure and backend admission reasons;
- buffer distribution, materialization, accessor, and transport records.

## Current Slices

### P0.1 CB Queue Event Ownership

Status: complete.

`TTProgram -> ExecutableSpec` projection now derives structured physical
`KernelSpec.queue_events` from `TTKernel` leaf CB calls and `TTCBPlan`
requirement-index ownership.  Runtime parses that projected array only.  The
old runtime source/body queue-event scanner is deleted.

### P0.2 Remaining Semantic Recovery Audit

Status: next.

Audit runtime/codegen/executable readers for target execution facts still
recovered from:

- final function body scans;
- generated source text;
- runtime-arg names or positions;
- workload-shaped branches;
- pass-local helper maps that survive beyond one pass;
- fallback defaults when typed fields are absent.

Each finding must either be deleted as stale or promoted to typed
`TTProgram` / `ExecutableSpec` owner truth before the parent task can close.

### P0.3 Execution Event / Admission Spine

Status: queued.

Where multiple leaf readers need the same execution ordering or admission
fact, centralize it as `TTProgram` owner truth and project it once.  Do not
let runtime, source emission, and Python metadata each reconstruct the same
decision.

## Completion Criteria

This parent task is complete only when:

- all covered leaf consumers get target execution facts from `ExecutableSpec`;
- `ExecutableSpec` fields are projected from typed `TTProgram` owner truth;
- runtime/codegen no longer scan source text, final bodies, names, or builtin
  neighborhoods to recover execution semantics;
- deleted recovery surfaces have structure tests or source guards;
- admitted T7/T8/T9 direct-runtime correctness still passes under the current
  bf16 TT-Sim baseline;
- docs, progress, and memory clearly distinguish completed slices from the
  unfinished parent task.
