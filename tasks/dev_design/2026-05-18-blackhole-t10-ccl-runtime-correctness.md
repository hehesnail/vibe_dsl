# Blackhole T10 Single-Card Multi-Tile CCL Semantics

## Role

This document defines the P2/T10 collective-communication single-card
multi-tile value and local-runtime slice after typed mesh placement.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Scope Change

As of 2026-05-18, T10 completion is scoped to single-card multi-tile
semantics.  Multi-device fabric execution remains recorded as an external
blocker, but it is no longer the active completion gate for T10.1-T10.3 in
this checkout.

This scope proves the logical value behavior of the CCL operations over
tile-aligned bf16 tensors on one host/card-equivalent execution context, and
executes local multi-tile equivalents through `BlackholeModule` on TT-Sim.  It
does not claim fabric route correctness, Ethernet TXQ support, remote
semaphore behavior, or real multi-device launch ordering.

## Goal

Define and verify a minimal, typed, single-card multi-tile semantics path for:

- all-gather;
- reduce-scatter;
- all-to-all.

Completion requires positive bf16 numerical correctness for all three logical
collectives on multi-tile shapes, compared with host references.  A
contract-only fail-closed slice is not a completion point for this task.

## Ordered Work

This task is ordered as a single-card value-semantics gate.  Typed contracts,
projection, validators, and unsupported diagnostics are prerequisites inside
the gate; they do not close the task without positive tensor-value checks.

1. `T10.1` Add the typed CCL owner truth needed by runtime:
   `TTCollectivePlan -> ExecutableSpec.collective_plans`, validators, and
   direct-runtime admission reasons.
2. `T10.2` Keep route / synchronization / launch-order semantics single-card
   and local.  No remote NoC, multicast fabric route, or global cross-device
   scheduling record is required for this scoped completion.  Non-unit mesh
   and fabric-backed CCL records remain fail-closed or externally blocked.
3. `T10.3` Broaden the scoped value gate to cover all three operation kinds
   on multi-tile bf16 tensors with host-reference comparisons and
   single-card `BlackholeModule` local-runtime equivalents.
4. Keep malformed or unsupported CCL variants fail-closed with typed
   diagnostics before source/runtime guessing.

## Representation Contract

`TTCollectivePlan` is the durable CCL owner record.

Required fields:

- `name`
- `operation_kind`: `all_gather`, `reduce_scatter`, or `all_to_all`
- `mesh_plan`
- `mesh_plan_index`
- `source_buffer`
- `target_buffer`
- `source_buffer_distribution`
- `source_buffer_distribution_index`
- `target_buffer_distribution`
- `target_buffer_distribution_index`
- `collective_axis`
- `tensor_axis`
- `split_axis`
- `concat_axis`
- `participant_count`
- `topology`
- `reduce_op`
- `input_shape`
- `output_shape`
- `required_semaphore_plan_indices`
- `required_sync_plan_indices`
- `admission_status`
- `unsupported_reason`

Shape fields describe the logical collective tensor shape at the contract
boundary.  They are not a runtime fallback for buffer addressability; runtime
addressability remains owned by `TTBufferDistributionPlan` and projected
accessor/materialization records.

For all-gather, `output_shape[tensor_axis]` must equal
`input_shape[tensor_axis] * participant_count`.

For reduce-scatter, `input_shape[tensor_axis]` must equal
`output_shape[tensor_axis] * participant_count`, and `reduce_op` must be
present.

For all-to-all, `split_axis` and `concat_axis` must be present, the input shape
must be divisible on the split axis, the output shape must be divisible on the
concat axis, and total element count must be preserved.

## Runtime Admission

`ExecutableSpec.collective_plans` is the canonical leaf projection of
`TTCollectivePlan`.

Direct runtime may execute only admitted CCL records whose mesh, buffer
distribution, route, sync, and launch-order records are complete for the
selected scope.  Under the current single-card multi-tile scope, fabric
`TTCollectivePlan` records remain unsupported/fail-closed; the positive runtime
gate is a local multi-tile equivalent that verifies the same all-gather,
reduce-scatter, and all-to-all value equations through `BlackholeModule` on a
unit mesh.  Runtime must not silently drop a collective, create a unit-mesh
fallback for a multi-device request, or reconstruct collective behavior from
generated source, buffer names, argument positions, or runtime observation.

## Validation Contract

`ValidateTTProgram` and leaf validation must reject:

- missing collective names or unsupported operation kinds;
- missing or mismatched mesh references;
- participant counts smaller than two;
- missing source or target buffers;
- missing or mismatched buffer-distribution references;
- collective and buffer-distribution mesh mismatches;
- invalid tensor/split/concat axes;
- shape rank mismatches or non-positive shape dimensions;
- all-gather, reduce-scatter, or all-to-all shape equations that do not hold;
- reduce-scatter without `reduce_op`;
- admitted records that still carry `unsupported_reason`;
- executable CCL records without complete route / synchronization evidence for
  their admitted scope;
- unsupported records without `unsupported_reason`;
- stale attempts to recover CCL semantics outside typed records.

## Verification

This slice is verified by:

- `scripts/probe_single_card_multitile_ccl_semantics.py`, proving bf16
  all-gather, reduce-scatter, and all-to-all value semantics over tile-aligned
  `8x8` multi-tile shapes;
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_t10_single_card_multitile_ccl_runtime.py`,
  proving local multi-tile all-gather / reduce-scatter / all-to-all
  equivalents through `BlackholeModule` on the repository TT-Sim bf16 direct
  path with `8x8x2` logical tile work items per collective;
- structure tests proving `TTCollectivePlan` exists on `TTProgram`;
- negative validator tests for malformed collective records;
- executable projection tests proving `collective_plans` reaches
  `ExecutableSpec`;
- direct-runtime tests proving unsupported CCL variants fail closed with typed
  diagnostics;
- a recorded external blocker for multi-device fabric CCL
  `eth_txq_cmd=0x2`, without treating that external blocker as part of the
  single-card completion gate;
- compile gate: `cmake --build build -j32`.

## Completion Criteria

This task is complete only when:

- all three collective operation kinds have typed owner truth from
  `TTProgram -> ExecutableSpec`;
- all-gather, reduce-scatter, and all-to-all pass the single-card multi-tile
  bf16 numerical semantics probe against host references;
- all-gather, reduce-scatter, and all-to-all pass the single-card multi-tile
  `BlackholeModule` local-runtime equivalent probe against host references;
- unsupported or malformed CCL forms fail closed with typed diagnostics;
- docs and `tasks/progress.md` state any remaining wider distributed work,
  such as generalized NoC scheduling or production partial-K reduction,
  without treating those as part of the CCL correctness closure.
