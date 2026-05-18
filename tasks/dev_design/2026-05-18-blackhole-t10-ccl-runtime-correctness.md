# Blackhole T10 CCL Runtime Correctness

## Role

This document defines the active P2/T10 collective-communication runtime
correctness slice after typed mesh placement.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

Admit a minimal, typed, distributed CCL runtime path for:

- all-gather;
- reduce-scatter;
- all-to-all.

Completion requires positive bf16 numerical correctness through
`BlackholeModule` under the repository TT-Sim setup, compared with host
references.  A contract-only fail-closed slice is not a completion point for
this task.

## Ordered Work

This task is ordered as a single correctness gate.  Typed contracts,
projection, validators, and unsupported diagnostics are prerequisites inside
the gate; they do not close the task without positive tensor-value checks.

1. `T10.1a` Prove the runtime support boundary:
   TT-Sim can create the required multi-device mesh shape and the selected
   TileLang/TT-Metal runtime route can launch a distributed collective without
   using the legacy external runner.  In the current local setup, the known
   blocker is the TT-Sim fabric fatal during CCL command handling; a real
   multi-device Blackhole target is an equivalent unblocker for this step.
2. `T10.1b` Add the typed CCL owner truth needed by runtime:
   `TTCollectivePlan -> ExecutableSpec.collective_plans`, validators, and
   direct-runtime admission reasons.
3. `T10.1c` Add the minimum route / synchronization / launch-order records
   required by the three admitted CCL operations.  These records are scoped to
   the CCL positive path; wider NoC / multicast / global scheduling
   generalization remains the next task after the correctness gate is green.
4. `T10.1d` Execute all-gather, reduce-scatter, and all-to-all through
   `BlackholeModule` on TT-Sim with bf16 inputs and compare against host
   references.
5. `T10.1e` Keep malformed or unsupported CCL variants fail-closed with typed
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
distribution, route, sync, and launch-order records are complete.  It must not
silently drop a collective, create a unit-mesh fallback, or reconstruct
collective behavior from generated source, buffer names, argument positions,
or runtime observation.

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
- executable CCL records without complete route / synchronization evidence;
- unsupported records without `unsupported_reason`;
- stale attempts to recover CCL semantics outside typed records.

## Verification

This slice is verified by:

- a TT-Sim environment probe proving the required multi-device CCL route is
  locally available, or a recorded blocker if it is not;
- structure tests proving `TTCollectivePlan` exists on `TTProgram`;
- negative validator tests for malformed collective records;
- executable projection tests proving `collective_plans` reaches
  `ExecutableSpec`;
- direct-runtime tests proving unsupported CCL variants fail closed with typed
  diagnostics;
- positive direct-runtime tests for all-gather, reduce-scatter, and all-to-all
  through `BlackholeModule` under TT-Sim using bf16 inputs and host-reference
  comparisons;
- compile gate: `cmake --build build -j32`.

## Completion Criteria

This task is complete only when:

- all three collective operation kinds have typed owner truth from
  `TTProgram -> ExecutableSpec`;
- direct runtime admits the supported CCL records from executable facts, not
  from source/runtime recovery;
- all-gather, reduce-scatter, and all-to-all pass TT-Sim bf16 numerical
  correctness against host references through `BlackholeModule`;
- unsupported or malformed CCL forms fail closed with typed diagnostics;
- docs and `tasks/progress.md` state any remaining wider distributed work,
  such as generalized NoC scheduling or production partial-K reduction,
  without treating those as part of the CCL correctness closure.
