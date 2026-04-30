# Blackhole Post-Preservation Pass Shrink

## Role

This document defines implementation responsibility boundaries after tile
compute truth has been preserved in `Normalized Tile TIR`.
It is not a history log.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

Keep Blackhole lowering files organized by responsibility without introducing
new IR layers or side channels.

The long-term chain remains:

```text
Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec
```

## Split Rules

- Split by representation boundary or implementation responsibility:
  selection,
  planning,
  source emission,
  materialization,
  ABI,
  transport,
  state.
- File split must not create a new protocol object.
- Durable truth stays in typed
  `SpatialPlan`,
  `TTProgram`,
  or `ExecutableSpec`
  state.
- Helper classes may reduce boilerplate, but must not own cross-stage
  semantics.

## Forbidden Regression

Do not recreate:

- `GenerateScalar*`
- `GenerateExplicit*`
- `RejectLegacyScalar*`
- scalar-loop recovery matchers
- composite protocol names such as `scalar_exp2_affine`
- leaf-looking composite payloads
- public hook tables that duplicate pattern-schema metadata

## Current Responsibility Map

The intended ownership split is:

- `lower_blackhole_tile_compute.cc`:
  explicit preserved tile-compute source lowering
- `lower_blackhole_exact_cb.cc`:
  exact tiled-CB live-form helpers
- `lower_blackhole_materialization.cc`:
  fragment/cast/local-to-CB materialization mechanics
- `lower_blackhole_abi.cc`:
  segment/accessor/runtime-arg/kernel/ABI plan synthesis
- `lower_blackhole_state.cc`:
  pass-local live-form alias and materialization state
- `lower_blackhole_transport.cc`:
  staged copy and page/tile transport source emission
- `lower_blackhole_matmul.cc`:
  GEMM extraction and matmul source emission
- `lower_blackhole_ops.cc`:
  orchestration, not semantic recovery

## Validation

For touched surfaces:

- build the C++ target
- run focused Blackhole transform/pipeline tests
- run TT-Sim runtime tests when runtime behavior changes
- scan for deleted matcher/generator names and composite payload strings

Docs should record the contract, not per-batch command output.
