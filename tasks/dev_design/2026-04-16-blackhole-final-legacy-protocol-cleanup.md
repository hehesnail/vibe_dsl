# Blackhole Final Legacy Protocol Cleanup Plan

> Split from the former monolithic cleanup note into one overview plus task-by-task execution docs.

## Goal

Remove the remaining legacy Blackhole protocol surfaces without replacing them with another protocol layer.

This cleanup is complete only when all of the following are true:

- Blackhole compute/data-movement builtins are locked to exact TT-Metal kernel API granularity.
- builtin lowering happens in a normal mutating selection pass over current IR, not through helper bags or workload-specific special cases.
- helper/composite builtins that do not correspond one-to-one with TT-Metal APIs are deleted.
- `blackhole.copy_semantics` and `blackhole.segment_kind` are gone.
- public `AnalyzeBlackhole*` wrappers and internal evidence bags are gone.
- `blackhole.lowering_requirements` is no longer used as a planning/semantic carrier.

## Compiler Design Baseline

The cleanup must follow normal compiler pass discipline.

1. The current IR/object at each stage is transformed into the next stage. It is not a read-only thing we keep re-analyzing forever.
2. Long-lived semantics may live only on the explicit representation layers:
   - `Normalized Tile TIR`
   - `SpatialPlan`
   - `TTProgram`
   - `ExecutableSpec`
3. A cleanup pass should normally:
   - walk the current IR or current representation object
   - match the local pattern it cares about
   - immediately rewrite IR or write directly into the representation object it is constructing
4. A match result is not a protocol.
5. Pass-local helper structs/classes are fine when they stay inside one `.cc` and only support that pass's visitor/mutator logic.
6. What is not allowed:
   - a new attr bag
   - a new public analysis wrapper
   - a new helper layer between representations
   - another stringly-typed `kind` / `role` / `direction` vocabulary
   - a workload-specific lowering path such as a dedicated flash-attention builtin matcher

This is the same discipline already used by the repo's GPU passes such as:

- [lower_hopper_intrin.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_hopper_intrin.cc)
- [lower_ldg_stg.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_ldg_stg.cc)
- [wgmma_sync_rewriter.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/wgmma_sync_rewriter.cc)
- [annotate_warp_group_reg_alloc.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/annotate_warp_group_reg_alloc.cc)

Blackhole cleanup should look the same: visitor/matcher/mutator over current IR, direct rewrite, no extra semantic baggage.

## One Narrow Exception

`tl.blackhole_logical_buffer_tile_bridge_specs` is the only allowed narrow bridge attr during this cleanup, and only because the optimized/helper entry still needs one leaf-local hand-off from pre-opt logical buffer/tile identity to the optimized device function.

Even that attr is tightly constrained:

- it carries only logical buffer/tile bridge specs
- it is leaf-local, not planning representation
- it must not grow into a replacement for `blackhole.compute_regions`

## Dependency Order

1. Lock the exact TT-Metal builtin surface and move lowering to a dedicated builtin-selection pass.
2. Replace compute-region bag extraction with direct logical bridge capture.
3. Delete public and internal legacy analysis bags.
4. Replace `blackhole.copy_semantics` with direct IR/dataflow recovery in consumers.
5. Replace `blackhole.segment_kind` with direct kernel-kind construction in planner/projection/runtime.
6. Finish docs, verification, and cleanup scans.

## File Structure

**New files**

- `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
- `tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc`

**Modified files across the cleanup**

- `tilelang_repo/tilelang/engine/lower.py`
- `tilelang_repo/tilelang/engine/phase.py`
- `tilelang_repo/tilelang/transform/__init__.py`
- `tilelang_repo/src/tir/builtin_blackhole.h`
- `tilelang_repo/src/tir/builtin_blackhole.cc`
- `tilelang_repo/src/transform/build_tt_program.cc`
- `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.h`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/materialize_blackhole_executable.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/testing/python/target/blackhole/common.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- `tasks/dev_design/README.md`
- `tasks/dev_design/blackhole_first_principles_protocol_audit.md`
- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`

**Deleted files**

- `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

## Responsibility Split

- `select_blackhole_tt_metal_builtins.cc`:
  match primitive IR idioms and rewrite them into exact TT-Metal builtin sequences. This is the only place that should decide which exact builtin sequence a local IR pattern becomes.
- `builtin_blackhole.{h,cc}`:
  define the exact builtin surface only. No composite/helper pseudo-ops.
- `capture_blackhole_logical_bridge_specs.cc`:
  walk current IR directly and attach only the narrow logical bridge attr when still needed by the optimized/helper path.
- `split_blackhole_kernel.cc` and `blackhole_device_resource_canonicalization.cc`:
  recover copy/dataflow meaning directly from current IR plus `SpatialPlan`.
- `build_tt_program.cc` / `lower_blackhole_ops.cc`:
  construct `TTProgram` kernel/transport/ABI representation directly from current IR plus current representation objects. No `segment_kind` replacement layer.
- `tt_program_projection.h`, `materialize_blackhole_executable.cc`, `codegen_blackhole.cc`, and `rt_mod_blackhole.cc`:
  consume projected executable/kernel records directly from `TTProgram` / `ExecutableSpec`, not from legacy attrs.

## Split Docs

- [Task 0: Lock exact TT-Metal builtin surface and add dedicated builtin selection](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md)
- [Task 1: Replace compute-region bags with direct logical bridge capture](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md)
- [Task 2: Remove public and internal legacy analysis bags](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md)
- [Task 3: Replace `blackhole.copy_semantics` with direct IR/dataflow recovery](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md)
- [Task 4: Replace `blackhole.segment_kind` with direct kernel-kind construction](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md)
- [Task 5: Final cleanup, documentation, and verification](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task5.md)

## Completion Cross-Check

- builtin selection is a normal IR mutator over exact TT-Metal shims
- no helper/composite compute builtins remain
- no `compute_epilogue_ops` or similar workload-side payload survives
- no public `AnalyzeBlackhole*` wrapper remains
- no internal evidence bag remains on the active chain
- `blackhole.copy_semantics` is gone
- `blackhole.segment_kind` is gone
- the only surviving bridge attr, if still needed, is `tl.blackhole_logical_buffer_tile_bridge_specs`
