# Blackhole Final Legacy Protocol Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining legacy Blackhole protocol surfaces by establishing a complete TT-Metal builtin basis with explicit legality contracts, deleting every composite/helper builtin that does not map one-to-one to TT-Metal kernel APIs, replacing `blackhole.copy_semantics` and `blackhole.segment_kind` with typed owner truth, and demoting `AnalyzeBlackhole*` wrappers out of the canonical transform API.

**Architecture:** Do the cleanup in dependency order. First audit the full Blackhole target builtin surface against TT-Metal's real public kernel APIs across data movement, synchronization, and compute, then replace the current mixed semantic/helper surface with a closed set of exact TT-Metal kernel-API shims. The builtin layer is not an operator-fusion layer: legal IR patterns should lower into exact TT-Metal instruction sequences, and fusion should come only from SRAM/CB/DST residency plus schedule, not from inventing composite builtins. Any higher-level rowwise compute/dataflow shape emitted by the DSL should be matched only as a lowering-local instruction-selection window over existing `TTProgram` / kernel-body truth and immediately lowered into exact TT-Metal op sequences or rejected as unsupported; do not introduce a new composite compute protocol or owner layer. Helper operations that describe local fragment materialization, tiled-slice reshaping, or bridge staging belong in pre-builtin lowering/materialization layers, not in the target builtin basis. Then introduce a narrow pre-opt bridge capture so engine code no longer needs the public `AnalyzeBlackholeComputeRegions` wrapper, migrate analysis regression coverage onto canonical `SpatialPlan` / `TTProgram` / `ExecutableSpec` surfaces and delete the public wrappers, replace `copy_semantics` consumers with one shared typed copy-role analysis, and replace `segment_kind`-based body slicing with a shared typed segment-slice analysis consumed by planner and runtime.

**Tech Stack:** TileLang C++ transform/runtime passes, TVM TIR `PrimFunc` attrs and typed objects, Python engine helpers, `pytest`, `cmake --build`.

---

## File Structure

**New files**

- Create: `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
- Create: `tilelang_repo/src/transform/common/blackhole_copy_role_analysis.h`
- Create: `tilelang_repo/src/transform/common/blackhole_copy_role_analysis.cc`
- Create: `tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.h`
- Create: `tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.cc`

**Modified files**

- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- Modify: `tilelang_repo/src/transform/build_tt_program.cc`
- Modify: `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- Modify: `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/tt_program_projection.h`
- Modify: `tilelang_repo/testing/python/target/blackhole/common.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/blackhole_first_principles_protocol_audit.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

**Deleted files**

- Delete: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

### Responsibility Split

- `capture_blackhole_logical_bridge_specs.cc`: a narrow internal pass that emits only `tl.blackhole_logical_buffer_tile_bridge_specs`; this replaces engine/test dependence on the public compute-region transform wrapper.
- `builtin_blackhole.{h,cc}`, `lower_blackhole_ops.cc`, and `codegen_blackhole.cc`: define the complete TT-Metal builtin basis for Blackhole across transport/sync/compute, delete helper/composite builtins that are not exact TT-Metal APIs, attach legality contracts to the exact shim surface, and instruction-select existing kernel-body truth into those shims.
- `blackhole_copy_role_analysis.{h,cc}`: one shared analysis that derives copy direction / buffer-role facts from `Normalized Tile TIR` + `SpatialPlan`, so planner and resource canonicalization stop reading `blackhole.copy_semantics`.
- `blackhole_segment_slice_analysis.{h,cc}`: one shared analysis that derives ordered reader/compute/writer slices directly from TIR structure and typed copy-role facts, so planner/runtime stop scanning `AttrStmt("blackhole.segment_kind")`.
- `tt_program_projection.h` and `rt_mod_blackhole.cc`: projection/runtime readers updated to consume typed segment truth from `TTProgram` / `ExecutableSpec` instead of legacy TIR attrs.
- Python tests under `testing/python/target/blackhole/`: canonical-path regressions for `TTProgram` payload, executable projection, and helper behavior.
- `test_blackhole_spatial_ir.py`: phase-bundle regressions that assert legacy attrs stay out of the mainline.

### Dependency Order

1. TT-Metal builtin-basis audit, legality matrix, and exact-shim lowering cutover.
2. Narrow bridge capture helper.
3. Public `AnalyzeBlackhole*` wrapper removal and coverage migration.
4. Typed copy-role analysis replacing `blackhole.copy_semantics`.
5. Typed segment-slice analysis replacing `blackhole.segment_kind`.
6. Docs, verification, final cleanup.

### Task 0: Establish Complete TT-Metal Builtin Basis And Legality Contracts

**Files:**
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/tt_program_projection.h`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tasks/dev_design/blackhole_first_principles_protocol_audit.md`

- [ ] **Step 1: Write the failing API-granularity regression**

Add transform-surface regressions for forbidden helper/composite builtins and canonical transport/compute regressions that assert the remaining target surface stays at TT-Metal API granularity:

```python
def test_blackhole_target_builtin_surface_stays_at_tt_metal_api_granularity():
    script = _prepare_blackhole_phase_b_module(flash_attention_kernel())["main"].script()
    assert "tl.blackhole.reduce_row" not in script
    assert "tl.blackhole.mul_row_bcast" not in script
    assert "tl.blackhole.mul_grouped_row_bcast" not in script
    assert "tl.blackhole.div_row_bcast" not in script
    assert "tl.blackhole.div_grouped_row_bcast" not in script
    assert "tl.blackhole.exp2_row_bcast_affine" not in script
    assert "tl.blackhole.exp2_grouped_row_bcast_affine" not in script
    assert "tl.blackhole.scalar_max" not in script
    assert "tl.blackhole.scalar_exp2_affine" not in script
    assert "tl.blackhole.write_local_fragment_tile_to_cb" not in script
    assert "tl.blackhole.write_local_fragment_slice_to_tiled_cb" not in script
    assert "tl.blackhole.read_cb_front_tile_to_local_fragment" not in script


def test_flash_attention_tt_program_does_not_gain_non_tt_metal_compute_protocol():
    lowered = _lower_flash_attention_to_tt_target()
    tt_program = require_tt_program(lowered["main"])
    payload = dict(tt_program.payload)
    assert "compute_epilogue_ops" not in payload
    assert "tl.blackhole.mul_grouped_row_bcast" not in lowered["main"].script()


def test_copy_lowers_to_transport_tt_metal_shims_without_compute_reload_aliases():
    copy_script = _lower_copy_to_tt_target()["main"].script()
    assert "tl.blackhole.noc_async_read" in copy_script
    assert "tl.blackhole.noc_async_write" in copy_script
    assert "tl.blackhole.cb_wait_front" in copy_script
    assert "tl.blackhole.read_tile_to_cb" not in copy_script
    assert "tl.blackhole.read_page_to_cb" not in copy_script
    assert "tl.blackhole.write_tile_from_cb" not in copy_script
    assert "tl.blackhole.write_page_from_cb" not in copy_script
    assert "tl.blackhole.runtime_arg_u32" not in copy_script
    assert "tl.blackhole.copy_tile_to_dst_init_short" not in copy_script
    assert "tl.blackhole.copy_tile_to_dst_init_short_with_dt" not in copy_script
    assert "tl.blackhole.copy_tile_from_cb" not in copy_script
```

- [ ] **Step 2: Run the targeted regressions and verify failure**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k tt_metal_api_granularity
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k transport_tt_metal_shims_without_compute_reload_aliases
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k non_tt_metal_compute_protocol
```

Expected: FAIL because the canonical target path still mixes exact TT-Metal shims with non-ISA helper/composite builtins.

- [ ] **Step 3: Audit the current builtin surface and split exact TT-Metal shims from helper/composite residue**

Build a basis table from TT-Metal public kernel APIs and turn the current `builtin_blackhole.{h,cc}` surface into an explicit 4-bucket inventory. The implementation should not move forward until every current builtin is classified as one of: keep-as-exact, split/demote helper residue, delete composite/helper, or add missing exact shim.

Bucket A: existing builtins that are already close to exact TT-Metal kernel APIs and can remain in the target builtin basis (subject to legality contracts and minor naming/signature cleanup only):

```cpp
// transport / sync
blackhole_cb_reserve_back
blackhole_cb_push_back
blackhole_cb_wait_front
blackhole_cb_pop_front
blackhole_noc_async_read
blackhole_noc_async_write
blackhole_noc_async_read_barrier
blackhole_noc_async_write_barrier
blackhole_get_semaphore
blackhole_semaphore_wait
blackhole_semaphore_set
blackhole_semaphore_inc_remote
blackhole_semaphore_set_remote

// compute
blackhole_mm_init
blackhole_mm_init_short
blackhole_mm_init_short_with_dt
blackhole_reconfig_data_format
blackhole_matmul_tiles
blackhole_tile_regs_acquire
blackhole_tile_regs_commit
blackhole_tile_regs_wait
blackhole_tile_regs_release
blackhole_pack_tile
blackhole_pack_reconfig_data_format
blackhole_copy_tile_to_dst_init_short
blackhole_copy_tile_to_dst_init_short_with_dt
blackhole_add_tiles_init
blackhole_add_tiles
```

The `copy_tile_to_dst_init_short{,_with_dt}` family is a compute-side DST reload API, not the generic TileLang broad-copy surface. It may remain only as part of a compute-local reload sequence; broad copy kernels should continue to lower through transport/sync shims (`noc_async_*`, `cb_*`) instead of this family.

Bucket B: current surface that is not an exact builtin layer and must be split into exact TT-Metal API shims plus surrounding ABI / accessor / CB-pointer plumbing in earlier lowering. These may survive only after being renamed/reworked into exact TT-Metal calls:

```cpp
blackhole_read_tile_to_cb
blackhole_read_page_to_cb
blackhole_write_tile_from_cb
blackhole_write_page_from_cb
blackhole_runtime_arg_u32
blackhole_copy_tile_from_cb
```

These are not exact today because they currently encode extra work such as compile-time accessor lookup, CB read/write pointer acquisition, implicit barriers, or runtime-arg name resolution inside codegen instead of representing a single TT-Metal API call. `blackhole_copy_tile_from_cb` is especially problematic because it aliases a compute-side DST reload primitive under a generic copy/helper name; it should be removed from the target builtin surface rather than treated as part of the broad copy vocabulary.

Bucket C: current helper/composite surface that must be deleted from `builtin_blackhole.h`, `builtin_blackhole.cc`, target lowering, and `codegen_blackhole.cc` because it does not map one-to-one to TT-Metal APIs:

```cpp
// local-fragment / materialization helpers
blackhole_write_local_slice_to_cb
blackhole_write_local_fragment_tile_to_cb
blackhole_write_local_fragment_slice_to_tiled_cb
blackhole_cast_fragment_slice_to_tiled_cb
blackhole_read_cb_front_tile_to_local
blackhole_read_cb_front_tile_to_local_fragment

// composite compute helpers
blackhole_reduce_row
blackhole_mul_row_bcast
blackhole_mul_grouped_row_bcast
blackhole_div_row_bcast
blackhole_div_grouped_row_bcast
blackhole_scalar_fma
blackhole_exp2_row_bcast_affine
blackhole_exp2_grouped_row_bcast_affine
blackhole_scalar_exp2_affine
blackhole_scalar_max

// local-fragment convenience helpers
blackhole_fill_fragment
blackhole_add_fragment
blackhole_add_fragment_from_cb_front
blackhole_cast_fragment_slice
```

Bucket D: missing exact TT-Metal shims that should be added so the builtin basis actually spans the admitted TT-Metal instruction/API surface instead of forcing future composite builtins to reappear:

```cpp
// broadcast / eltwise binary
blackhole_add_bcast_rows_init_short
blackhole_add_bcast_cols_init_short
blackhole_add_bcast_scalar_init_short
blackhole_add_tiles_bcast_rows
blackhole_add_tiles_bcast_cols
blackhole_add_tiles_bcast_scalar
blackhole_mul_bcast_rows_init_short
blackhole_mul_bcast_cols_init_short
blackhole_mul_tiles_bcast_scalar_init_short
blackhole_mul_tiles_bcast_rows
blackhole_mul_tiles_bcast_cols
blackhole_mul_tiles_bcast_scalar

// reduction
blackhole_reduce_init
blackhole_reduce_tile
blackhole_reduce_uninit

// SFPU / DST-resident unary-binary
blackhole_binary_max_tile
blackhole_binary_max_tile_init
blackhole_div_binary_tile
blackhole_div_binary_tile_init
blackhole_exp_tile
blackhole_exp_tile_init
blackhole_exp2_tile
blackhole_exp2_tile_init
blackhole_recip_tile
blackhole_recip_tile_init
```

Treat the TT-Metal `copy_tile` family as a separate compute-local tile-move concern, not as the representation of broad TileLang copy. The broad copy/data-movement path should remain entirely on the transport/sync side (`noc_async_*`, `cb_*`, barriers, pack/unpack scheduling). If the exact TT-Metal `copy_tile` API is needed in the builtin layer for a legal compute-local DST reload sequence, introduce it only under that compute-local contract next to `copy_tile_to_dst_init_short{,_with_dt}`; do not let it leak back into broad copy tests, naming, or protocol vocabulary.

Delete Bucket C from `builtin_blackhole.h` and `builtin_blackhole.cc`:

```cpp
TVM_DLL const Op& blackhole_reduce_row();
TVM_DLL const Op& blackhole_mul_row_bcast();
TVM_DLL const Op& blackhole_mul_grouped_row_bcast();
TVM_DLL const Op& blackhole_div_row_bcast();
TVM_DLL const Op& blackhole_div_grouped_row_bcast();
TVM_DLL const Op& blackhole_exp2_row_bcast_affine();
TVM_DLL const Op& blackhole_exp2_grouped_row_bcast_affine();
TVM_DLL const Op& blackhole_scalar_max();
TVM_DLL const Op& blackhole_scalar_exp2_affine();
TVM_DLL const Op& blackhole_scalar_fma();
TVM_DLL const Op& blackhole_write_local_slice_to_cb();
TVM_DLL const Op& blackhole_write_local_fragment_tile_to_cb();
TVM_DLL const Op& blackhole_write_local_fragment_slice_to_tiled_cb();
TVM_DLL const Op& blackhole_cast_fragment_slice_to_tiled_cb();
TVM_DLL const Op& blackhole_read_cb_front_tile_to_local();
TVM_DLL const Op& blackhole_read_cb_front_tile_to_local_fragment();
```

Bucket C local-fragment/materialization helpers must be re-expressed in the storage/layout/materialization pipeline before builtin selection. Bucket B transport/ABI helpers must be split so builtin selection only sees exact TT-Metal transport/compute calls plus separately-owned ABI/runtime plumbing.

In `codegen_blackhole.cc`, delete the helper code paths that only exist to realize the removed non-ISA surface:

```cpp
tilelang_scalar_max(...)
tilelang_reduce_row_sum(...)
tilelang_reduce_row_max(...)
tilelang_mul_row_bcast(...)
tilelang_mul_grouped_row_bcast(...)
tilelang_div_row_bcast(...)
tilelang_div_grouped_row_bcast(...)
tilelang_scalar_fma(...)
tilelang_exp2_row_bcast_affine(...)
tilelang_exp2_grouped_row_bcast_affine(...)
tilelang_scalar_exp2_affine(...)
```

- [ ] **Step 4: Add explicit legality contracts for every exact TT-Metal shim**

Introduce one legality matrix used by lowering and validation. Representative rules:

```cpp
// examples
RequireInterleavedDramAccessor(blackhole_noc_async_read);
RequireInterleavedDramAccessor(blackhole_noc_async_write);
RequireCommonRuntimeArgCountZero(current_direct_runtime_surface);

RequireInitPair(blackhole_reduce_tile, blackhole_reduce_init, blackhole_reduce_uninit);
RequireCbOperandAndDstRegs(blackhole_reduce_tile);
RequireInitPair(blackhole_add_tiles_bcast_rows, blackhole_add_bcast_rows_init_short);
RequireBroadcastShapeAndLayout(blackhole_add_tiles_bcast_rows, BroadcastType::ROW);
RequireInitPair(blackhole_mul_tiles_bcast_rows, blackhole_mul_bcast_rows_init_short);
RequireBroadcastShapeAndLayout(blackhole_mul_tiles_bcast_rows, BroadcastType::ROW);
RequireInitPair(blackhole_exp_tile, blackhole_exp_tile_init);
RequireDstTileResidencyAndInit(blackhole_exp_tile);
RequireInitPair(blackhole_exp2_tile, blackhole_exp2_tile_init);
RequireDstTileResidencyAndInit(blackhole_exp2_tile);
RequireInitPair(blackhole_recip_tile, blackhole_recip_tile_init);
RequireDstTileResidencyAndInit(blackhole_recip_tile);
RequireInitPair(blackhole_binary_max_tile, blackhole_binary_max_tile_init);
RequireBinarySfpuOperandsInDst(blackhole_binary_max_tile);
RequireInitPair(blackhole_div_binary_tile, blackhole_div_binary_tile_init);
RequireBinarySfpuOperandsInDst(blackhole_div_binary_tile);
```

By the time builtin selection starts, any surviving local-fragment helper or composite semantic op is a hard error:

```cpp
if (ContainsNonTTMetalBuiltinCandidate(body)) {
  LOG(FATAL) << "Blackhole lowering reached builtin selection with non-TT-Metal helper/composite ops";
}
```

- [ ] **Step 5: Instruction-select exact TT-Metal transport and compute sequences directly from existing kernel-body truth**

Do not introduce a new semantic op or `compute_epilogue_ops` field. Match legal instruction-selection windows from existing kernel-body truth during lowering, then immediately emit exact TT-Metal API shims. For the current flash-attn admitted surface, one legal window lowers into the following exact sequence:

```cpp
auto windows = MatchTTMetalComputeLoweringWindows(func, bridge_specs, spatial_plan);
for (const auto& window : windows) {
  selected_ops.push_back(MakeReduceInit(/*pool=*/"max", /*dim=*/"row", window.max_reduce));
  selected_ops.push_back(MakeReduceTile(/*pool=*/"max", /*dim=*/"row", window.max_reduce));
  selected_ops.push_back(MakeReduceUninit(window.max_reduce));
  selected_ops.push_back(MakeBinaryMaxTileInit(window.max_merge));
  selected_ops.push_back(MakeBinaryMaxTile(window.max_merge));
  selected_ops.push_back(MakeAddBcastRowsInitShort(window.sub_max));
  selected_ops.push_back(MakeAddTilesBcastRows(window.sub_max));
  selected_ops.push_back(MakeExp2TileInit(window.exp_stage));
  selected_ops.push_back(MakeExp2Tile(window.exp_stage));
  selected_ops.push_back(MakeReduceInit(/*pool=*/"sum", /*dim=*/"row", window.sum_reduce));
  selected_ops.push_back(MakeReduceTile(/*pool=*/"sum", /*dim=*/"row", window.sum_reduce));
  selected_ops.push_back(MakeReduceUninit(window.sum_reduce));
  selected_ops.push_back(MakeRecipTileInit(window.norm_recip));
  selected_ops.push_back(MakeRecipTile(window.norm_recip));
  selected_ops.push_back(MakeMulBcastRowsInitShort(window.apply_norm));
  selected_ops.push_back(MakeMulTilesBcastRows(window.apply_norm));
}
```

Likewise, lower broad copy / move / stage patterns directly into exact transport/sync shims:

```cpp
EmitCbReserveBack(...);
EmitNocAsyncRead(...);
EmitNocAsyncReadBarrier(...);
EmitCbPushBack(...);
EmitCbWaitFront(...);
EmitPackTile(...);
EmitNocAsyncWrite(...);
EmitNocAsyncWriteBarrier(...);
EmitCbPopFront(...);
```

If a compute kernel still needs a DST reload sequence (for example partial-result reload inside GEMM), treat that as a separate compute-local lowering concern and do not describe it or test it as the generic TileLang copy surface.

That compute-local path must also carry its own legality contract: prior `cb_wait_front`, acquired DST/register state, and exact tile-move sequencing. It is not interchangeable with the transport-side notion of copy/move/stage.

The window objects are lowering-local analysis results, not new IR/protocol surfaces.

At the end of lowering, the executable/runtime surface should contain only exact TT-Metal shims or fail-fast:

```cpp
if (ContainsNonTTMetalBuiltinCandidate(lowered_body)) {
  LOG(FATAL) << "Blackhole lowering left non-TT-Metal helper/composite builtins in the target surface";
}
```

- [ ] **Step 6: Re-run the canonical-path tests and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/tir/builtin_blackhole.h \
        tilelang_repo/src/tir/builtin_blackhole.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/src/target/tt_program_projection.h \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
        tasks/dev_design/blackhole_first_principles_protocol_audit.md
git commit -m "blackhole: align target builtins with tt-metal api"
```

### Task 1: Replace Public Compute-Region Wrapper Usage With Narrow Bridge Capture

**Files:**
- Create: `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/common.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Write the failing regression for optimized helper bridge capture**

```python
def test_flash_attention_optimized_helper_path_keeps_logical_bridge_specs():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]

    payload = dict(require_tt_program(lowered).payload)
    buffers = {str(spec["buffer"]) for spec in payload["buffer_tile_bridge_specs"]}
    assert {"acc_s", "acc_o", "scores_max"}.issubset(buffers)
```

- [ ] **Step 2: Run the targeted flash-attn test and verify the current helper path fails without public region-pass fallback**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k optimized_helper_path_keeps_logical_bridge_specs
```

Expected: FAIL in `PlanTTKernelABI` or missing `buffer_tile_bridge_specs`.

- [ ] **Step 3: Implement a narrow internal bridge-capture pass**

Add a pass that reuses `AnalyzeBlackholeComputeRegionEvidence(func)` and writes only `tl.blackhole_logical_buffer_tile_bridge_specs`:

```cpp
PrimFunc CaptureLogicalBridgeSpecs(const PrimFunc& func) {
  Map<String, Any> facts = AnalyzeBlackholeComputeRegionEvidence(func);
  Array<Any> specs;
  if (auto maybe_regions = facts.Get(String("regions"))) {
    for (const Any& region_any : Downcast<Array<Any>>(maybe_regions.value())) {
      Map<String, Any> region = Downcast<Map<String, Any>>(region_any);
      if (auto maybe_specs = region.Get(String(schema_key::kBufferTileBridgeSpecs))) {
        for (const Any& spec_any : Downcast<Array<Any>>(maybe_specs.value())) {
          specs.push_back(spec_any);
        }
      }
    }
  }
  if (specs.empty()) {
    return func;
  }
  return WithAttr(std::move(func), attr::kTLBlackholeLogicalBufferTileBridgeSpecs, specs);
}
```

Use it from Python instead of the public transform wrapper:

```python
with target:
    mod = LowerAndLegalize(mod, target)
    if target.kind.name == "blackhole":
        bridge_capture_mod = tilelang.transform.CaptureBlackholeLogicalBridgeSpecs()(
            LowerToBlackholePhaseB(mod)
        )
```

- [ ] **Step 4: Update the test helpers to use the narrow capture path**

Keep the helper behavior aligned with `lower()`:

```python
def lower_blackhole_to_tt_target(mod):
    source_mod = tilelang.transform.CaptureBlackholeLogicalBridgeSpecs()(
        LowerToBlackholePhaseB(mod)
    )
    mod = _align_blackhole_device_symbol(source_mod, mod)
    return tilelang.engine.phase.LowerToBlackholeTTProgram(mod)
```

- [ ] **Step 5: Re-run the targeted flash-attn helper tests and commit**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k 'optimized_helper_path_keeps_logical_bridge_specs or gqa_'
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc \
        tilelang_repo/tilelang/engine/lower.py \
        tilelang_repo/testing/python/target/blackhole/common.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "blackhole: capture logical bridge specs without public region pass"
```

### Task 2: Remove Public `AnalyzeBlackhole*` Transform Wrappers And Move Coverage To Canonical Surfaces

**Files:**
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Delete: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Add the public-surface regression**

Replace direct analysis-wrapper expectations with canonical-surface checks:

```python
def test_blackhole_transform_no_longer_exports_legacy_analysis_wrappers():
    for name in (
        "AnalyzeBlackholeWorkDecomposition",
        "AnalyzeBlackholeComputeRegions",
        "AnalyzeBlackholePipelineStages",
    ):
        assert not hasattr(tilelang.transform, name)
```

Also move one unique content regression into canonical TTProgram payload coverage:

```python
def test_flash_attention_tt_program_payload_exposes_bridge_buffers_without_legacy_region_attr():
    lowered = _lower_flash_attention_to_tt_target()
    payload = dict(require_tt_program(lowered).payload)
    assert "buffer_tile_bridge_specs" in payload
    assert lowered.attrs.get("blackhole.compute_regions") is None
```

- [ ] **Step 2: Run the wrapper-surface test and verify it fails**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k no_longer_exports_legacy_analysis_wrappers
```

Expected: FAIL because `tilelang.transform` still exports the three wrapper names.

- [ ] **Step 3: Remove the public wrapper exports and FFI registrations while preserving internal helper functions**

Python side:

```python
# delete these from tilelang/transform/__init__.py
def AnalyzeBlackholeWorkDecomposition(): ...
def AnalyzeBlackholeComputeRegions(): ...
def AnalyzeBlackholePipelineStages(): ...
```

C++ side, keep helper functions but remove the registered transform passes:

```cpp
// Keep:
Map<String, Any> AnalyzeBlackholeWorkDecompositionEvidence(const PrimFunc& func);
Map<String, Any> AnalyzeBlackholePipelineStageEvidence(const PrimFunc& func);

// Remove:
tir::transform::Pass AnalyzeBlackholeWorkDecompositionPass() { ... }
TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tl.transform.AnalyzeBlackholeWorkDecomposition", ...);
}
```

- [ ] **Step 4: Delete the old direct-analysis test file and keep the unique coverage in canonical-path files**

Delete:

```text
tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
```

Move the useful checks into `test_blackhole_spatial_ir.py` and `test_blackhole_flash_attention_pipeline.py`:

```python
assert lowered.attrs.get("blackhole.work_decomposition") is None
assert lowered.attrs.get("blackhole.compute_regions") is None
assert lowered.attrs.get("blackhole.pipeline_stages") is None
```

- [ ] **Step 5: Run the transform/flash-attn tests and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/tilelang/transform/__init__.py \
        tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc \
        tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc \
        tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git rm tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: remove public legacy analysis wrappers"
```

### Task 3: Replace `blackhole.copy_semantics` With Typed Copy-Role Analysis

**Files:**
- Create: `tilelang_repo/src/transform/common/blackhole_copy_role_analysis.h`
- Create: `tilelang_repo/src/transform/common/blackhole_copy_role_analysis.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`

- [ ] **Step 1: Write the failing copy-role regressions**

Add one planner-facing and one resource-facing test:

```python
def test_blackhole_copy_planning_does_not_require_copy_semantics_annotations():
    mod = _lower_copy_module_to_device_tir()
    func = _strip_loop_annotation(mod["main"], "blackhole.copy_semantics")
    lowered = lower_blackhole_to_tt_target(tvm.IRModule({"main": func}))
    tt_program = require_tt_program(lowered["main"])
    assert any(str(kernel.kind) == "fused_dataflow" for kernel in tt_program.kernels)


def test_blackhole_resource_canonicalization_uses_typed_copy_roles():
    mod = _lower_copy_module_to_device_tir()
    func = _strip_loop_annotation(mod["main"], "blackhole.copy_semantics")
    canonical = tilelang.transform.BlackholeDeviceResourceCanonicalization()(
        tvm.IRModule({"main": func})
    )["main"]
    assert "blackhole.cb.input" in canonical.script()
```

- [ ] **Step 2: Run the copy tests and verify failure**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k 'does_not_require_copy_semantics_annotations or uses_typed_copy_roles'
```

Expected: FAIL because current code still reads `blackhole.copy_semantics`.

- [ ] **Step 3: Add a shared typed copy-role analysis**

Introduce one shared helper consumed by both `SplitBlackholeKernel` and `BlackholeDeviceResourceCanonicalization`:

```cpp
struct BlackholeCopyRoleFact {
  std::string src_buffer;
  std::string dst_buffer;
  std::string mid_buffer;
  std::string direction;
  bool is_fused_staged_copy{false};
};

std::vector<BlackholeCopyRoleFact> AnalyzeBlackholeCopyRoles(
    const tir::PrimFunc& func,
    const SpatialPlan& spatial_plan);
```

Use `SpatialPlan.DataflowEdge` and stable buffer identities instead of loop annotations.

- [ ] **Step 4: Switch the two remaining consumers and remove the pipeline pass**

In `split_blackhole_kernel.cc`:

```cpp
const auto copy_roles = AnalyzeBlackholeCopyRoles(func, spatial_plan);
if (role.direction == "dram_to_cb") return "reader";
if (role.direction == "cb_to_dram") return "writer";
if (role.is_fused_staged_copy) return "reader";
```

In `blackhole_device_resource_canonicalization.cc`:

```cpp
const auto copy_roles = AnalyzeBlackholeCopyRoles(func, spatial_plan);
for (const auto& role : copy_roles) {
  if (role.direction == "dram_to_cb") cb_input_names_.insert(role.dst_buffer);
  if (role.direction == "cb_to_dram") cb_output_names_.insert(role.src_buffer);
}
```

Delete the copy-semantics prepass from `OptimizeForTarget`:

```python
# remove:
mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
```

- [ ] **Step 5: Run the copy suite and commit**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k phase_b
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/common/blackhole_copy_role_analysis.h \
        tilelang_repo/src/transform/common/blackhole_copy_role_analysis.cc \
        tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc \
        tilelang_repo/tilelang/engine/phase.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
git commit -m "blackhole: derive copy roles from typed dataflow truth"
```

### Task 4: Replace `blackhole.segment_kind` With Typed Segment-Slice Analysis

**Files:**
- Create: `tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.h`
- Create: `tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/tt_program_projection.h`
- Modify: `tilelang_repo/src/transform/materialize_blackhole_executable.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Write the failing segment-truth regressions**

Add one phase-bundle regression and one runtime-facing regression:

```python
def test_blackhole_phase_b_does_not_publish_segment_kind_attrs():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    assert "blackhole.segment_kind" not in mod["main"].script()


def test_blackhole_gemm_runtime_keeps_three_kernels_without_segment_kind_attr():
    artifact = _lower_gemm_artifact()
    assert "blackhole.segment_kind" not in artifact.device_mod["main_kernel"].script()
    spec = _extract_blackhole_executable_spec(artifact)
    assert [str(kernel["kind"]) for kernel in spec["kernels"]] == ["reader", "compute", "writer"]
```

- [ ] **Step 2: Run the GEMM/spatial tests and verify failure**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k segment_kind
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k three_kernels_without_segment_kind_attr
```

Expected: FAIL because current planner/runtime still scan `AttrStmt("blackhole.segment_kind")`.

- [ ] **Step 3: Add one shared segment-slice analysis used by planner and runtime**

Introduce a typed slice helper:

```cpp
struct BlackholeSegmentSlice {
  std::string kind;
  Array<tir::Stmt> stmts;
};

Array<BlackholeSegmentSlice> AnalyzeBlackholeSegmentSlices(
    const tir::PrimFunc& func,
    const SpatialPlan& spatial_plan,
    const std::vector<BlackholeCopyRoleFact>& copy_roles);
```

This helper should classify reader / compute / writer slices from TIR structure plus typed copy-role facts, without emitting any `AttrStmt`.

- [ ] **Step 4: Switch planner/runtime to the typed slice helper and stop emitting `segment_kind`**

In `lower_blackhole_ops.cc`, replace `CollectSegmentKindsFromBody` / `current_segment_kind_` scanning:

```cpp
const auto segment_slices = AnalyzeBlackholeSegmentSlices(func, spatial_plan, copy_roles);
for (const auto& slice : segment_slices) {
  Map<String, Any> kernel;
  kernel.Set("name", String(slice.kind));
  kernel.Set("kind", String(slice.kind));
  kernel.Set("core_type", String(CoreTypeForSegmentKind(slice.kind)));
  kernels.push_back(kernel);
}
```

In `tt_program_projection.h`, project typed segment-local leaf bodies or named segment slices into `tl.blackhole_executable`:

```cpp
executable.Set(String(executable_key::kSegmentBodies), EncodeSegmentBodies(program));
```

In `rt_mod_blackhole.cc`, replace `SegmentBodyExtractor` with executable projection reads:

```cpp
Array<Any> segment_bodies = GetExecutableArrayField(f, "Blackhole build", executable_key::kSegmentBodies);
for (const Any& body_any : segment_bodies) {
  Map<String, Any> body = Downcast<Map<String, Any>>(body_any);
  // consume projected segment body directly
}
```

Then remove the `AttrStmt("blackhole.segment_kind", ...)` emission path from `SplitBlackholeKernel`.

- [ ] **Step 5: Run the GEMM/flash-attn suites and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.h \
        tilelang_repo/src/transform/common/blackhole_segment_slice_analysis.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/src/target/tt_program_projection.h \
        tilelang_repo/src/transform/materialize_blackhole_executable.cc \
        tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: replace segment attr slicing with executable truth"
```

### Task 5: Final Cleanup, Documentation, And Full Verification

**Files:**
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/blackhole_first_principles_protocol_audit.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Update progress and protocol-audit docs to reflect the true end state**

Record the final state explicitly:

```md
- `blackhole.copy_semantics`: deleted from active path and runtime/codegen readers
- `blackhole.segment_kind`: deleted from active path and runtime/codegen readers
- `AnalyzeBlackhole*`: no longer exported from `tilelang.transform`
```

- [ ] **Step 2: Add one reusable memory note for each cleanup**

Example `memory/general_dev.md` addition:

```md
- copy-direction and segment-role truth should be derived once from typed owner layers and shared by planner/runtime; loop annotations are not an acceptable long-term bridge.
```

Example `memory/bugs.md` addition:

```md
#### deleting legacy attrs must happen after consumer replacement
- symptom: tests stay green in one path but runtime still scans a removed attr
- fix: replace all readers first, then delete emission, then delete tests/docs
```

- [ ] **Step 3: Run the full Blackhole verification set**

Run:

```bash
cmake --build build -j32
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

- [ ] **Step 4: Confirm the worktree is clean and no long-running commands remain**

Run:

```bash
ps -ef | rg 'pytest|cmake --build|ctest'
git status -sb
```

Expected: only the `ps` / `rg` command itself shows up, then `git status -sb` shows a clean branch after commit/push.

- [ ] **Step 5: Final commit and push**

```bash
git add tasks/progress.md \
        tasks/dev_design/blackhole_first_principles_protocol_audit.md \
        memory/general_dev.md \
        memory/bugs.md
git commit -m "blackhole: finish final legacy protocol cleanup"
git push
```

## Self-Review

### 1. Spec coverage

- Remaining legacy attr readers:
  - Covered by Task 3 (`blackhole.copy_semantics`)
  - Covered by Task 4 (`blackhole.segment_kind`)
- Remaining public analysis wrappers:
  - Covered by Task 1 and Task 2
- Docs/progress/memory sync:
  - Covered by Task 5

### 2. Placeholder scan

- No `TODO` / `TBD` markers remain.
- Every task names exact files.
- Every code-changing task includes concrete test code, implementation skeletons, commands, and commit messages.

### 3. Type consistency

- Narrow bridge capture uses `tl.blackhole_logical_buffer_tile_bridge_specs` consistently.
- Copy cleanup routes ownership through `SpatialPlan` / `TTProgram`, not back through attrs.
- Segment cleanup routes ownership through `TTProgram` / `ExecutableSpec`, not back through `AttrStmt`.
