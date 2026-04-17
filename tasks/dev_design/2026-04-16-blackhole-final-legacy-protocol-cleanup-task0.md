# Task 0: Lock Exact TT-Metal Builtin Surface And Add Dedicated Builtin Selection

## Scope

This task does three things:

1. lock `builtin_blackhole.*` to exact TT-Metal kernel API granularity;
2. define one builtin legality-contract registry shared by selection and validation;
3. implement one normal mutating builtin-selection pass that rewrites primitive IR idioms into legal exact TT-Metal builtin sequences.

This task does **not** authorize another semantic layer. In particular it must not introduce:

- helper/composite builtins such as `blackhole_reduce_row`, `blackhole_mul_row_bcast`, `blackhole_exp2_row_bcast_affine`, `blackhole_scalar_max`, or `blackhole_copy_tile_from_cb`
- workload-specific lowering entrypoints such as `TryLowerRowwiseFlashAttnRegion`
- reusable fact bags such as `ComputeLoweringFacts`
- matcher-window protocols such as `MatchTTMetalComputeLoweringWindows`
- payload side channels such as `compute_epilogue_ops`

A rowwise softmax epilogue is not a builtin. It remains ordinary IR that selects into an exact TT-Metal sequence such as `binary_max_tile`, `exp_tile`, `reduce_tile`, `recip_tile`, `mul_tiles_bcast_rows`, `add_tiles_bcast_rows`, and `pack_tile`, subject to legality.

## Status (`2026-04-17`)

This selector-forwarding slice is landed in repo HEAD.

- `SelectBlackholeTTMetalBuiltins` is now on the active chain between
  `PlanTTBlocks` and `PlanTTCompute`.
- compute-side anchored IR idioms are selected before planner helper lowering;
  `PlanTTCompute` fail-closes on
  `tl.blackhole_tt_metal_builtin_selection`.
- helper/composite builtin residue is no longer admitted on the active IR
  surface; tests and `ValidateTTProgram` reject it by exact op name.
- local pseudo compute builtins such as
  `reduce_rows_local`,
  `mul_tiles_bcast_rows_local`,
  `div_tiles_bcast_rows_local`,
  `exp_tiles_bcast_rows_affine_local`,
  `exp_tile_affine_local`,
  and
  `scalar_fma`
  are deleted from the builtin/codegen surface rather than merely fail-closed.
  The active sequences now lower directly to exact TT-Metal ops such as
  `reduce_init/reduce_tile/reduce_uninit`,
  `mul_tiles_init/mul_tiles`,
  `binary_max_tile`,
  `exp2_tile`,
  and
  `recip_tile`.
- `compute_epilogue_ops` is removed from
  `TTProgram.payload`,
  executable projection,
  codegen,
  runtime,
  and tests.
- selector-created exact temporary CB requirements are now persisted through
  `blackhole.cb_requirements`
  and reloaded by
  `PlanTTCompute`,
  so `PlanTTCBAlloc` does not see dangling requirement indices after
  selector-forwarding rewrites.
- current residue kept intentionally narrow:
  `tl.blackhole_lowering_requirements_seed`
  carries only
  `buffer_materialization_contracts`
  and
  `buffer_tile_bridge_specs`
  so selector-forwarding can preserve stable bridge/materialization facts
  across the IR rewrite.
  It is stripped before final TTProgram materialization and is not a new
  protocol.
- one remaining representation-cutover nuance is unchanged:
  CB/materialization-sensitive bridge publication still touches
  `PlanTTKernelABI`
  because CB requirement indices are still planned there.
  That residue is for
  `Cleanup Task 1 / Task 2`,
  not a reason to reopen helper/composite builtin selection.

## Files

- Create: `tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/build_tt_program.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/tt_program_projection.h`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tasks/dev_design/blackhole_first_principles_protocol_audit.md`

## Execution Slices

1. write regressions that forbid fake helper/composite builtins and payload residue
2. audit the current builtin surface against real TT-Metal kernel APIs
3. delete fake helper/composite builtins and keep only exact TT-Metal shims
4. define one legality-contract registry used by selector and validator
5. implement the dedicated builtin-selection mutator
6. cut lowering/codegen/validation over to the selector and fail closed on residue
7. rerun the focused suites and commit

- [x] **Step 1: Write the failing API-granularity regressions**

Add regressions that lock the surface to exact TT-Metal API granularity:

- transform-surface checks that helper/composite builtins never appear:
  - `blackhole_reduce_row`
  - `blackhole_mul_row_bcast`
  - `blackhole_mul_grouped_row_bcast`
  - `blackhole_div_row_bcast`
  - `blackhole_div_grouped_row_bcast`
  - `blackhole_exp2_row_bcast_affine`
  - `blackhole_exp2_grouped_row_bcast_affine`
  - `blackhole_scalar_max`
  - `blackhole_scalar_exp2_affine`
  - `blackhole_copy_tile_from_cb`
  - local-fragment bridge helpers
- executable/payload checks that no workload-side compute protocol appears, for example no `compute_epilogue_ops`
- copy-path checks that data movement lowers to exact TT-Metal transport/data-movement APIs rather than broad pseudo-copy helpers

- [x] **Step 2: Run the targeted regressions and verify failure**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k tt_metal_api_granularity
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k transport_tt_metal_api_granularity
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k no_compute_epilogue_payload
```

Historical red gate:
FAIL because the old path mixed exact TT-Metal shims with helper/composite
builtins and payload residue.

Current repo HEAD status:
these selectors/payload checks are now part of the green verification baseline.

- [x] **Step 3: Audit the builtin surface against real TT-Metal APIs and delete fake helpers**

Build the exact inventory from TT-Metal public kernel APIs and examples in `tt_metal_repo`. Keep only exact shims such as:

```text
cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front
noc_async_read / noc_async_write / noc_async_read_barrier / noc_async_write_barrier
get_semaphore / semaphore_wait / semaphore_set / semaphore_inc_remote / semaphore_set_remote
tile_regs_acquire / tile_regs_commit / tile_regs_wait / tile_regs_release
pack_tile / pack_reconfig_data_format
copy_tile_to_dst_init_short / copy_tile_to_dst_init_short_with_dt
mm_init / mm_init_short / mm_init_short_with_dt / reconfig_data_format / matmul_tiles
add_bcast_rows_init_short / add_tiles_bcast_rows
mul_bcast_rows_init_short / mul_tiles_bcast_rows
reduce_init / reduce_tile / reduce_uninit
binary_max_tile_init / binary_max_tile
div_binary_tile_init / div_binary_tile
exp_tile_init / exp_tile
exp2_tile_init / exp2_tile
recip_tile_init / recip_tile
```

Also keep any additional exact TT-Metal API variants verified by the audit, for example row/col/scalar broadcast variants or exact dtype/reconfig variants that already exist upstream.

Delete fake helper/composite builtins such as:

```text
blackhole_reduce_row
blackhole_mul_row_bcast
blackhole_mul_grouped_row_bcast
blackhole_div_row_bcast
blackhole_div_grouped_row_bcast
blackhole_exp2_row_bcast_affine
blackhole_exp2_grouped_row_bcast_affine
blackhole_scalar_max
blackhole_scalar_exp2_affine
blackhole_copy_tile_from_cb
blackhole_write_local_slice_to_cb
blackhole_write_local_fragment_tile_to_cb
blackhole_write_local_fragment_slice_to_tiled_cb
blackhole_cast_fragment_slice_to_tiled_cb
blackhole_read_cb_front_tile_to_local
blackhole_read_cb_front_tile_to_local_fragment
```

If a current helper hides a sequence like "init + op + pack" or "reserve + noc + barrier + push", split it into the exact TT-Metal operations instead of renaming the helper.

Current repo HEAD note:
old helper C++ wrapper entrypoints may still exist as compatibility aliases, but
the active IR surface and validator surface are the canonical exact op names
rather than the old helper/composite builtin names.

- [x] **Step 4: Define one legality-contract registry shared by selection and validation**

This registry is not a new pass and not a new protocol layer. It is only the shared legality surface for exact builtin emission and exact builtin validation.

It should answer questions like:

- which init op pairs with which exact compute op
- which exact ops require DST ownership first
- which operands must already reside in CB or DST
- which broadcast/reduce/vector-mode signatures are legal
- which data-format reconfig steps must precede a given exact op
- which barriers must complete outstanding transport ops

The registry should be keyed by exact TT-Metal builtin op and consumed in two places:

1. the selector, while it is building a legal exact sequence
2. `ValidateTTProgram`, while it is replaying the exact selected sequence

Representative rule classes:

1. init / uninit pairing
2. DST/tile-register lifecycle
3. operand residency
4. layout / broadcast / vector-mode requirements
5. data-format / reconfiguration requirements
6. CB protocol requirements
7. transport ordering / barrier completion

Representative rules:

```cpp
RequireInitPair(blackhole_reduce_tile, blackhole_reduce_init, blackhole_reduce_uninit);
RequireDstOwnership(blackhole_reduce_tile);
RequireCbOperands(blackhole_reduce_tile);
RequireReduceSignature(blackhole_reduce_tile, /* pool, dim, scaler, vector mode */);

RequireInitPair(blackhole_add_tiles_bcast_rows, blackhole_add_bcast_rows_init_short);
RequireCbOperands(blackhole_add_tiles_bcast_rows);
RequireBroadcastLayout(blackhole_add_tiles_bcast_rows, BroadcastType::kRows);

RequireInitPair(blackhole_mul_tiles_bcast_rows, blackhole_mul_bcast_rows_init_short);
RequireCbOperands(blackhole_mul_tiles_bcast_rows);
RequireBroadcastLayout(blackhole_mul_tiles_bcast_rows, BroadcastType::kRows);

RequireInitPair(blackhole_binary_max_tile, blackhole_binary_max_tile_init);
RequireDstOperands(blackhole_binary_max_tile);

RequireInitPair(blackhole_exp_tile, blackhole_exp_tile_init);
RequireDstOperands(blackhole_exp_tile);

RequireBarrierAfter(blackhole_noc_async_read, blackhole_noc_async_read_barrier);
RequireBarrierAfter(blackhole_noc_async_write, blackhole_noc_async_write_barrier);
```

The selector may keep a pass-local emission state while traversing current IR:

```cpp
struct TTMetalEmissionState {
  bool dst_owned = false;
  Optional<ExactInitKind> active_init;
  Optional<DataFormatMode> current_format;
  ...
};
```

That state is strictly local to the pass. It is not another protocol layer.

Do **not** mix current direct-runtime admission limits into this registry. Runtime admission stays downstream, after `ExecutableSpec` projection.

Current repo HEAD note:
this slice landed the shared legality surface needed for selector stamp /
residue rejection / exact builtin validation without introducing a new
representation layer.
Deeper CB/materialization cleanup stays with the later cutover tasks.

- [x] **Step 5: Implement the dedicated builtin-selection pass**

Implement a normal mutating pass over current TIR, following the same shape already used by the repo's GPU passes:

```cpp
class SelectBlackholeTTMetalBuiltins : public tir::StmtExprMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final;
  Stmt VisitStmt_(const BufferStoreNode* op) final;
  PrimExpr VisitExpr_(const CallNode* op) final;
};
```

Use local helpers in the same `.cc` only:

- `Match*` helpers that inspect current IR
- `TrySelect*` helpers that emit exact TT-Metal sequences using the legality registry
- tiny local structs if they help one matcher or one rewrite

Do not add public headers or public helper passes for these matchers.

The right shape is:

```cpp
if (auto match = MatchRowReduce(op)) {
  if (auto seq = TrySelectRowReduce(*match, state_, registry_)) {
    state_ = seq->next_state;
    return EmitSelectedSequence(seq->stmts);
  }
}
```

The wrong shape is:

- create `ComputeLoweringFacts`
- attach a "softmax epilogue" payload
- introduce `MatchTTMetalComputeLoweringWindows`
- add a dedicated `TryLowerRowwiseFlashAttnRegion`

- [x] **Step 6: Cut lowering, projection, and validation over to exact builtin selection**

After the selector exists:

- `lower_blackhole_ops.cc` must stop inventing composite Blackhole builtins
- `build_tt_program.cc` must consume the exact selected builtin surface
- `tt_program_projection.h` / `codegen_blackhole.cc` / `rt_mod_blackhole.cc` must stop depending on composite compute payloads such as `compute_epilogue_ops`
- `ValidateTTProgram` must validate the exact builtin sequence using the same legality-contract registry

Fail closed on any surviving residue:

```cpp
ICHECK(!UsesHelperCompositeBlackholeBuiltin(func))
    << "helper/composite builtin residue must be removed before TTProgram validation";
```

- [x] **Step 7: Re-run the focused suites and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k tt_metal_api_granularity
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/tir/builtin_blackhole.h \
        tilelang_repo/src/tir/builtin_blackhole.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.h \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/build_tt_program.cc \
        tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/src/target/tt_program_projection.h \
        tilelang_repo/tilelang/engine/phase.py \
        tilelang_repo/tilelang/transform/__init__.py \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
        tasks/dev_design/blackhole_first_principles_protocol_audit.md
git commit -m "blackhole: lock exact tt-metal builtin surface"
```
