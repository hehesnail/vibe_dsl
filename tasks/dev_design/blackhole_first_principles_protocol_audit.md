# Blackhole First-Principles Protocol Audit

## Role

This document is a compact audit of fake or legacy protocol surfaces.
It is historical guidance, not the active progress board.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Audit Rule

A cross-stage protocol is valid only if it is represented as typed owner truth
in the correct layer:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Anything else is a temporary implementation mechanic at best.

## Forbidden Legacy Surfaces

The following surfaces must not be reintroduced as public protocol:

| Legacy surface | Correct owner |
| --- | --- |
| top-level `TTProgram.payload` | typed `TTProgram` fields |
| `compute_contract`, `gemm_contract`, `multi_*_contracts` | `TTComputeOpPlan` and `KernelSpec.compute_ops` |
| `tl.blackhole_logical_buffer_tile_bridge_specs` | `SpatialPlan` live/layout/materialization objects and projected typed plans |
| lowering facts `Map<String, Any>` bags | pass-local typed structs or explicit IR fields |
| compute-op seed maps | direct construction of typed compute plans |
| `blackhole.copy_semantics` | typed transport / executable records |
| `blackhole.segment_kind` after final projection | pass-local lowering mechanics only |
| leaf name/default fallback | required typed leaf records |
| legacy external runner | in-process `BlackholeModule` direct host path |
| runtime/codegen recovery from source text, work IDs, or buffer names | typed `ExecutableSpec` |

## Layer-Specific Checks

### Normalized Tile TIR

- TT-Metal compute semantics are explicit leaf statements.
- Composite semantics are not hidden behind leaf-looking payloads.
- Names are not used to recover semantic roles.

### SpatialPlan

- target-independent dataflow, layout, phase, and live-value facts are typed.
- bridge attrs and helper facts are not consumed as owner truth.

### TTProgram

- TT-specific realization uses typed slices:
  hardware model,
  mesh,
  core groups,
  buffer distribution,
  compute,
  CB,
  transport,
  sync,
  ABI,
  execution,
  resource demand/pressure.
- Planning does not read runtime/codegen fallback observations.

### ExecutableSpec

- leaf readers require typed projected fields.
- missing maps/arrays fail closed.
- runtime args, accessors, CBs, semaphores, cores, and compute ops are explicit.
- module serialization is real when advertised.

## Regression Scan Targets

Useful source scans:

- legacy contract names:
  `compute_contract|gemm_contract|multi_.*_contract`
- old bridge attr:
  `tl.blackhole_logical_buffer_tile_bridge_specs`
- empty leaf-reader fallback:
  `value_or\\(ffi::Map<ffi::String, ffi::Any>\\(\\)\\)`
- default physical core fallback:
  `PhysicalCore\\{\\}`
- composite payload strings:
  `exp2_tile\\(mode|mul_tiles_bcast_cols\\(\"div\"`
- scalar-loop recovery helpers:
  `GenerateScalar|GenerateExplicit|RejectLegacyScalar`

The exact command lines belong in task logs or final reports, not in this
active audit document.
