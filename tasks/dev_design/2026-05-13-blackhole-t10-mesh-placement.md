# Blackhole T10 Mesh / Multi-Device Placement

## Role

This document defines the first P2/T10 production-distributed slice.
It is not a new overall design document.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

Represent mesh and device placement as typed target-realization truth before
any distributed runtime movement is admitted.

This slice answers:

- which TT system mesh exists for the program;
- which device-coordinate range the current executable is placed on;
- which core group and buffer distribution records attach to that mesh;
- whether direct runtime can admit that placement or must fail closed.

## Non-Goals

- No CCL contract.
- No NoC / multicast / global scheduling claim.
- No remote route, remote semaphore, or cross-device producer/consumer timing.
- No production partial-K reducer protocol.
- No runtime fallback that treats a multi-device placement as unit mesh.

## Representation Contract

`TTMeshPlan` is the durable mesh owner record for this slice.
`PlanTTBlocks` derives the first mesh plan from the Blackhole target hardware
model; target attrs such as `mesh_shape_x/y` and
`device_range_start/shape_x/y` are input facts, not a separate runtime
protocol.

Required fields:

- `name`
- `mesh_kind`
- `mesh_shape`
- `device_range_start`
- `device_range_shape`
- `system_mesh_ref`

`TTCoreGroup` records must reference the selected mesh plan by name and index,
and carry the same device range used for resident workers.

`TTBufferDistributionPlan` records must keep referencing the selected mesh
plan by name and index.

`ExecutableSpec` must project mesh plans and the core-plan mesh binding before
leaf validation or direct-runtime admission.

## Direct-Runtime Admission

The current `BlackholeModule` direct path admits only unit mesh execution.

If `ExecutableSpec.mesh_plans` is anything other than the unit mesh accepted
by the current direct path, direct runtime must append a typed unsupported
reason before execution.  It must not silently create a unit mesh, widen
`MeshCoordinateRange` from runtime observation, or drop the projected placement
record.

Runtime correctness for this slice is therefore:

- unit-mesh positive paths continue to run through `BlackholeModule`;
- non-unit / multi-device mesh placement compiles to typed records and fails
  closed at direct-runtime admission until CCL / NoC / distributed movement are
  typed.

## Validation Contract

Validators must reject:

- missing mesh plans;
- duplicate mesh-plan names;
- invalid mesh shape or device range;
- core groups without a mesh binding;
- core groups whose mesh binding does not match a `TTMeshPlan`;
- buffer distributions whose `mesh_plan_index` does not match `mesh_plan`;
- executable core plans or buffer distributions that require missing mesh
  records.

## Completion Criteria

This slice is complete only when:

- `PlanTTBlocks` derives mesh placement from target facts instead of always
  constructing `unit_mesh`;
- `TTProgram` core groups and buffer distributions carry the selected mesh;
- `ExecutableSpec` projects and validates mesh records;
- direct runtime admits unit mesh and rejects multi-device mesh with a typed
  reason;
- structure, projection, and direct-runtime tests cover the admitted and
  rejected surfaces.
