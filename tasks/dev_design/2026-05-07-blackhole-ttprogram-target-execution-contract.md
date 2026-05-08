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
`KernelSpec.queue_events` from typed `TTKernel.queue_events` and `TTCBPlan`
requirement-index ownership.  `TTKernel.body` remains the materialized leaf
body used by source emission and local allocation-time rewrites, but it is not
the projection owner truth.  Runtime parses the projected array only.  The old
runtime/source/body queue-event scanners are deleted.

### P0.2 Remaining Semantic Recovery Audit

Status: complete.

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

Completed slice:

- Codegen runtime buffer address binding no longer scans the final TIR body to
  rediscover buffers used by `tl.blackhole.*_to_cb` / `*_from_cb` builtins.
  Packed-entrypoint buffer address args are bound from projected
  `ExecutableSpec.runtime_args[].buffer` records.  Codegen may still add a
  pointer-keyed fast binding when the function signature or `buffer_map`
  directly exposes the same buffer handle, but that is no longer a recovery
  requirement.
- Host launch to device kernel association no longer scans the packed host
  TIR body for `tvm_call_packed` string callees.  `LowerDeviceKernelLaunch`
  records launched kernel symbols in the explicit
  `tl.launched_kernel_symbols` IR attr, and Blackhole runtime consumes that
  attr to copy projected device `ExecutableSpec` records onto the host entry.
  Multiple Blackhole device launches from one host entry fail closed because
  the current runtime module contract admits a single host-to-device
  association.
- Runtime buffer materialization and multidimensional per-work descriptor
  admission no longer scan device `PrimFunc` bodies or buffer maps to rebuild
  static buffer shape/dtype facts.  They consume projected
  `ExecutableSpec.tensor_memory_config_plans[*].logical_shape` records,
  requiring conflicts to fail closed at the executable schema boundary.
- Reduction-region source emission no longer scans the final TIR body to
  recover reduction kind, reduction dimension, or repeated tile extent.
  `TTComputeOpPlan` owns `reduction_kind`, `reduction_dim`, and
  `repeat_extent` for reduce ops, `ValidateTTProgram` checks them, and the
  executable `compute_ops` projection is the only source consumed by
  Blackhole codegen/runtime serialization.
- Guarded row-page copy and guard-mask source CBs now keep physical FIFO
  publish and consume events at one active thread grain.  Source generation
  no longer lets every thread lane reserve/push or wait/pop the same per-work
  CB event when the executable contract only contains one physical CB page
  event.
- Remote synchronization endpoints no longer use `logical_core_noc_x/y`
  runtime-arg pairs as projection owner truth.  `TTABIPlan` carries explicit
  `TTRemoteCoreDescriptorSpec` records, `ValidateTTProgram` requires any
  `logical_core_noc_*` ABI arg to reference one of those descriptors, and
  `TTProgram -> ExecutableSpec` projection serializes
  `KernelSpec.remote_core_descriptors` only from the explicit descriptor
  records.  The remaining segment-body walk in the Blackhole build path is
  validation-only: it checks that semaphore/remote builtins consume the
  projected schema and rejects literal or body-recovered endpoints.
- `blackhole.segment_kind` is no longer an active lowering protocol, including
  as a pass-local marker exception.  Segment identity is recorded while
  lowering builds concrete reader/compute/writer leaves, then materialized as
  staged `TTKernel` bodies on `TTProgram` and projected once into
  `ExecutableSpec` segment records.  The final TIR and active lowering source
  are guarded against reintroducing the marker string.

### P0.3 Execution Event / Admission Spine

Status: complete.

Where multiple leaf readers need the same execution ordering or admission
fact, centralize it as `TTProgram` owner truth and project it once.  Do not
let runtime, source emission, and Python metadata each reconstruct the same
decision.

The current P0 spine is:

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
