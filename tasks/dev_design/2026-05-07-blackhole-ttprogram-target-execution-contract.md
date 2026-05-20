# Blackhole TTProgram Target Execution Contract

## Role

This document defines the standing architecture contract for `TTProgram` as
the Blackhole target-facing execution contract.

It is not a second overall design document and not a fifth IR layer.  The
durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status and next work live in `tasks/progress.md`.

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

## Execution Spine

Where multiple leaf readers need the same execution ordering or admission
fact, `TTProgram` owns it and `ExecutableSpec` projects it once.  Runtime,
source emission, codegen, and Python metadata must not reconstruct the same
decision independently.

The standing spine is:

- physical CB FIFO events:
  typed `TTKernel.queue_events` + `TTCBPlan` requirement ownership ->
  `KernelSpec.queue_events`;
- exact-CB lifecycle and storage legality:
  `TTExactCB*` records plus `TTCBPlan` ->
  `ValidateTTProgram` and executable queue/lifecycle gates;
- semaphore endpoints:
  `TTSemaphorePlan` plus `TTSemaphoreBindingSpec` ->
  `ExecutableSpec.semaphores` and `KernelSpec.semaphore_bindings`;
- remote synchronization endpoints:
  `TTRemoteCoreDescriptorSpec` plus logical-core ABI references ->
  `KernelSpec.remote_core_descriptors`;
- kernel launch association:
  explicit `tl.launched_kernel_symbols` IR attr ->
  host/device executable association;
- runtime/common/per-work ABI:
  `TTABIPlan` and `TTPerWorkArgSpec` records ->
  executable runtime arg and per-work schemas;
- buffer/materialization/resource admission:
  `TTBufferDistributionPlan`, `TTTensorMemoryConfigPlan`,
  materialization/liveness records, and `TTResourcePressureReport` ->
  executable distribution, materialization, and typed admission gates.

Backend-local simulator capability gates may still append typed
`direct_runtime_unsupported_reasons` during executable build.  Those gates are
leaf admission results over projected records, not planner semantic recovery.

## Retired Recovery Surfaces

The following surfaces are no longer protocol.  Reintroduction requires a new
typed owner field/object and a source guard or validator:

- generated source text or final-body scans for queue events, buffer
  bindings, reduction signatures, materialization shapes, or host launch
  association;
- `TTKernel.body` parsing at the `TTProgram -> ExecutableSpec` projection
  boundary for physical CB FIFO events;
- runtime-arg names or positions as endpoint, buffer, or per-work value owner
  truth;
- workload-shaped branches in leaf readers;
- pass-local helper maps that survive as cross-stage protocol;
- fallback defaults when a typed field is absent.

## Regression Criteria

This contract stays satisfied only while:

- covered leaf consumers get target execution facts from `ExecutableSpec`;
- `ExecutableSpec` fields are projected from typed `TTProgram` owner truth;
- deleted recovery surfaces have structure tests, source guards, or validators;
- stale simulator-fatal admission gates are not retained after an admitted
  bf16 direct-runtime path has positive TT-Sim correctness coverage;
- admitted runtime tests assert empty `direct_runtime_unsupported_reasons`
  before execution instead of converting those reasons into pytest skips;
- positive Blackhole lowering/runtime tests must not catch lowering failures
  and skip with `not yet` messages; unsupported non-contract inputs should be
  explicit negative fail-closed tests;
- admitted T7/T8/T9 direct-runtime correctness remains green under the current
  bf16 TT-Sim baseline;
- docs, progress, and memory do not preserve a second, contradictory
  execution contract.
