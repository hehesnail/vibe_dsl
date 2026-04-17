# Task 5: Final Cleanup, Documentation, And Full Verification

## Scope

Finish the cleanup by synchronizing the docs, running the full verification set, scanning for forbidden residue, confirming there are no leftover long-running commands, then committing and pushing the final state.

## Files

- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/README.md`
- Modify: `tasks/dev_design/blackhole_first_principles_protocol_audit.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

## Execution Slices

1. update progress/design docs to match the real end state
2. record the stable pass-design lessons in memory
3. run grep-based cleanup scans
4. run the full build/test set
5. confirm no long-running commands remain
6. final commit and push

- [ ] **Step 1: Update progress/design docs to match the real end state**

Record the final state explicitly:

```md
- Blackhole builtin lowering now selects exact TT-Metal builtin sequences from current IR.
- No helper/composite compute builtins remain in `builtin_blackhole.*`.
- No workload-specific compute payload such as `compute_epilogue_ops` remains.
- `blackhole.copy_semantics` is deleted from the active chain.
- `blackhole.segment_kind` is deleted from the active chain.
- `AnalyzeBlackhole*` is no longer exported from `tilelang.transform`.
- No internal evidence bag remains on the active chain.
- `tl.blackhole_logical_buffer_tile_bridge_specs` is the only surviving narrow bridge attr, if still needed.
```

- [ ] **Step 2: Record the stable pass-design lessons in memory**

Add to `memory/general_dev.md`:

```md
- In this pipeline the current IR/object is always being transformed into the next stage; it is not a read-only analysis substrate.
- If a pass can recover what it needs from current IR or current representation objects, recover it locally in that pass instead of minting a bag or attr.
- Pass-local matcher/visitor structs are fine only when they stay inside one `.cc` and immediately feed the rewrite or representation construction in that file.
- Blackhole builtins must stay at exact TT-Metal kernel API granularity; fusion comes from residency and schedule, not from helper/composite builtins.
```

Add to `memory/bugs.md`:

```md
#### deleting legacy attrs requires reader replacement first
- symptom: one path looks green but runtime/codegen still scan a removed attr
- fix: replace every reader, then delete emission, then delete docs/tests
```

- [ ] **Step 3: Run grep-based cleanup scans**

Run:

```bash
rg -n "AnalyzeBlackhole|ComputeLoweringFacts|MatchTTMetalComputeLoweringWindows|TryLowerRowwiseFlashAttnRegion|blackhole\\.copy_semantics|blackhole\\.segment_kind|blackhole\\.lowering_requirements" tilelang_repo/src tilelang_repo/tilelang
rg -n "blackhole_(reduce_row|mul_row_bcast|mul_grouped_row_bcast|div_row_bcast|div_grouped_row_bcast|exp2_row_bcast_affine|exp2_grouped_row_bcast_affine|scalar_max|scalar_exp2_affine|copy_tile_from_cb)" tilelang_repo/src
```

Expected: no active-chain hits beyond tests/docs that are explicitly asserting deletion.

- [ ] **Step 4: Run the full Blackhole verification set**

Run:

```bash
cmake --build build -j32
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

- [ ] **Step 5: Confirm no long-running commands remain**

Run:

```bash
ps -ef | rg 'pytest|cmake --build|ctest'
git status -sb
```

Expected:

- only the `ps` / `rg` command itself is visible
- `git status -sb` is clean after the final commit/push

- [ ] **Step 6: Final commit and push**

```bash
git add tasks/progress.md \
        tasks/dev_design/README.md \
        tasks/dev_design/blackhole_first_principles_protocol_audit.md \
        memory/general_dev.md \
        memory/bugs.md
git commit -m "blackhole: finish legacy protocol cleanup"
git push
```

## Self-Review

### End-State Checklist

- builtin selection is a normal mutating pass over current IR
- exact TT-Metal shims are the only remaining compute/data-movement builtins
- no semantic bag remains between `Normalized Tile TIR`, `SpatialPlan`, `TTProgram`, and `ExecutableSpec`
- runtime/codegen read projected executable/kernel records directly

### Forbidden Residue Checklist

- no `AnalyzeBlackhole*`
- no `ComputeLoweringFacts`
- no `MatchTTMetalComputeLoweringWindows`
- no `TryLowerRowwiseFlashAttnRegion`
- no `blackhole.copy_semantics`
- no `blackhole.segment_kind`
- no helper/composite builtin residue
