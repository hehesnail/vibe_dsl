# Stage 2D GEMM Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GEMM direct path numerical error by making `transpose_B` and host tilize/untilize part of the formal Blackhole runtime contract, then clean up stale artifacts and continue schema quality work.

**Architecture:** Investigation showed the lowered TIR already carries the required CB synchronization. The actual correctness gap is later: `transpose_B` was dropped before runtime/spec materialization, and `BlackholeModule` uploaded row-major host tensors directly instead of TT-Metal tiled data. The fix path is therefore `LowerBlackholeOps -> rt_mod_blackhole -> ExecutableSpec -> BlackholeModule` with an explicit minimal GEMM contract plus host-side transpose/tilize/untilize.

**Tech Stack:** TileLang TIR passes, Blackhole target codegen/runtime, TT-Metal direct host path, pytest, TT-Sim.

> **2026-03-26 实施修正**：进一步调试后确认“缺少同步原语”也不是最终根因。当前实际完成的修复是：
> 先确认 lowered TIR/segment source 的同步成立 → 追到 host/runtime layout contract →
> 引入 `blackhole.gemm_contract` → 在 direct path 做 transpose/tilize/untilize → 再清理遗留与做 schema 收正。

---

### Task 0: Dump and Diff Generated Kernel Sources

**Files:**
- Read: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Reference: `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/`

- [ ] **Step 1: Add a test or debug print that dumps generated reader/compute/writer kernel source**

Add a helper in `test_blackhole_gemm.py` (or use existing pipeline test) that prints the three generated kernel sources to stdout or a temp file.

Expected: Can see the exact code that TT-Metal will execute for each of the three kernels.

- [ ] **Step 2: Manually compare against TT-Metal matmul_single_core reference**

Compare generated reader vs `reader_single_core_mm.cpp`:
- Does reader do `cb_reserve_back` before `get_write_ptr`? (expected: NO — this is the bug)
- Does reader do `cb_push_back` after `noc_async_read_barrier`? (expected: NO — this is the bug)

Compare generated writer vs `writer_single_core_mm.cpp`:
- Does writer do `cb_wait_front` before `get_read_ptr`? (expected: NO)
- Does writer do `cb_pop_front` after `noc_async_write_barrier`? (expected: NO)

Compare generated compute vs `mm.cpp`:
- Does compute have correct `mm_init`, `cb_wait_front`, `matmul_tiles`, `pack_tile` sequence? (expected: YES, this part looks correct from LowerBlackholeOps)

- [ ] **Step 3: Document all observed semantic divergences**

List every divergence found. Expected divergences:
1. Missing CB synchronization in reader/writer (confirmed from code review)
2. Possibly: tile indexing formula differences
3. Possibly: transpose handling differences
4. Possibly: output dtype differences

---

### Task 1: Fix CB Synchronization Protocol

**Files:**
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`

This is a ~10-line change that directly fixes the numerical error root cause.

- [ ] **Step 1: Add `cb_reserve_back` / `cb_push_back` to `PrintReadTileToCB`**

Change from:
```c++
os << "{ ";
os << "const uint32_t tile_index = ...";
os << "; const uint32_t tile_bytes = ...";
os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ")";
os << "; InterleavedAddrGen<true> src_gen = ...";
os << "noc_async_read_tile(tile_index, src_gen, cb_l1_addr); ";
os << "noc_async_read_barrier(); }";
```

To:
```c++
os << "{ ";
os << "cb_reserve_back(" << cb_id << ", 1); ";
os << "const uint32_t tile_index = ...";
os << "; const uint32_t tile_bytes = ...";
os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ")";
os << "; InterleavedAddrGen<true> src_gen = ...";
os << "noc_async_read_tile(tile_index, src_gen, cb_l1_addr); ";
os << "noc_async_read_barrier(); ";
os << "cb_push_back(" << cb_id << ", 1); }";
```

- [ ] **Step 2: Add `cb_wait_front` / `cb_pop_front` to `PrintWriteTileFromCB`**

Change from:
```c++
os << "{ ";
os << "const uint32_t tile_index = ...";
os << "; const uint32_t tile_bytes = ...";
os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ")";
os << "; InterleavedAddrGen<true> dst_gen = ...";
os << "noc_async_write_tile(tile_index, dst_gen, cb_l1_addr); ";
os << "noc_async_write_barrier(); }";
```

To:
```c++
os << "{ ";
os << "cb_wait_front(" << cb_id << ", 1); ";
os << "const uint32_t tile_index = ...";
os << "; const uint32_t tile_bytes = ...";
os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ")";
os << "; InterleavedAddrGen<true> dst_gen = ...";
os << "noc_async_write_tile(tile_index, dst_gen, cb_l1_addr); ";
os << "noc_async_write_barrier(); ";
os << "cb_pop_front(" << cb_id << ", 1); }";
```

- [ ] **Step 3: Rebuild and run copy pipeline tests (must not regress)**

Run: `cd tilelang_repo/build && cmake --build . -j32`
Run: `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
Expected: PASS (copy path should also benefit from correct CB sync, or at minimum not break).

- [ ] **Step 4: Run GEMM test to verify numerical improvement**

Run: `source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k basic -s`
Expected: max_diff should drop dramatically from 37.24. If now correct, GEMM bug is confirmed as CB sync issue.

- [ ] **Step 5: Commit the CB sync fix**

```bash
git add tilelang_repo/src/target/codegen_blackhole.cc
git commit -m "blackhole: add CB reserve/push/wait/pop sync to reader/writer codegen"
```

---

### Task 2: Fix Remaining Numerical Divergences (if any)

**Files:**
- Possibly: `tilelang_repo/src/target/codegen_blackhole.cc`
- Possibly: `tilelang_repo/src/transform/lower_blackhole_ops.cc`

Only proceed with this task if Task 1 Step 4 shows residual numerical error.

- [ ] **Step 1: Check transpose_B handling**

Verify: does the test use transpose_B? Does `LowerBlackholeOps::GenerateMatmulSequence` handle it?
If the test B matrix needs transpose and codegen doesn't handle it, fix.

- [ ] **Step 2: Check output dtype alignment**

Verify: does the test use same dtype for accumulator and output? If not, check `pack_tile` output format.

- [ ] **Step 3: Check tile index formula**

Compare generated A/B/C tile indexing with TT-Metal reference:
- A: `mt * Kt + kt`
- B: `kt * Nt + nt`
- C: `mt * Nt + nt`

- [ ] **Step 4: Fix and re-verify**

Apply any needed fixes, rebuild, re-run GEMM test.

- [ ] **Step 5: Commit if changes were needed**

```bash
git add tilelang_repo/src/target/codegen_blackhole.cc tilelang_repo/src/transform/lower_blackhole_ops.cc
git commit -m "blackhole: fix GEMM residual numerical divergences"
```

---

### Task 3: Clean Up scratch_l1 and Stale Artifacts

**Files:**
- Possibly modify: `tilelang_repo/src/target/blackhole_module.cc`
- Possibly modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Delete: `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`

- [ ] **Step 1: Verify scratch_l1 is dead code**

First prove where scratch is still referenced:
Run: `rg -n "scratch_l1" tilelang_repo/src/target/blackhole_module.cc tilelang_repo/src/target/rt_mod_blackhole.cc tilelang_repo/src/target/codegen_blackhole.cc tilelang_repo/testing/python/target/blackhole`

Then do a negative check by temporarily making scratch allocation/building fail fast in `blackhole_module.cc`, for example:

```c++
if (arg_spec.kind == "scratch_l1_buffer_addr32") {
  LOG(FATAL) << "scratch_l1_buffer_addr32 should be dead on the main copy/GEMM path";
}
```

Run: `cd tilelang_repo/build && cmake --build . -j32`
Run: `source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
Run: `source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k basic`

Expected: main-path copy runtime and GEMM basic still pass without touching the fatal path. If they hit the fatal path, scratch is not dead yet and must not be removed.

After the negative check, remove the temporary fatal guard in the same working change before proceeding.

- [ ] **Step 2: Remove scratch_l1 from runtime schema (if confirmed dead)**

Remove `scratch_l1_buffer_addr32` from:
- `rt_mod_blackhole.cc` runtime arg spec emission
- `blackhole_module.cc` `KernelNeedsScratchL1`, scratch buffer allocation, and `BuildRuntimeArgsFromSpec`

- [ ] **Step 3: Delete tilelang_gemm_test**

Run: `rm -rf tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test`
Run: `rg -n "tilelang_gemm_test" /root/dev/vibe_dsl`
Expected: No remaining references (or only historical docs).

- [ ] **Step 4: Run full regression**

Run: `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
Run: `source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
Expected: All PASS.

- [ ] **Step 5: Commit cleanup**

```bash
git add tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tt_metal_repo/tt_metal/programming_examples
git commit -m "blackhole: remove scratch_l1 dead code and stale gemm bring-up example"
```

---

### Task 4: Schema Formalization (Protocol Quality, Post-Correctness)

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`

This task improves protocol quality AFTER numerical correctness is achieved. It is NOT a prerequisite for GEMM working.

- [ ] **Step 1: Add Mt/Kt/Nt/transpose_B to split-after GEMM attrs**

In `LowerBlackholeOps`, emit formal GEMM ABI fields:
```c++
gemm_attrs.Set("Mt", Integer(Mt));
gemm_attrs.Set("Kt", Integer(Kt));
gemm_attrs.Set("Nt", Integer(Nt));
gemm_attrs.Set("transpose_B", Bool(transpose_b));
```

- [ ] **Step 2: Add output dtype layering**

```c++
gemm_attrs.Set("accumulator_dtype", String(accumulator_dtype));
gemm_attrs.Set("transport_dtype", String(transport_dtype));
gemm_attrs.Set("final_tensor_dtype", String(final_tensor_dtype));
```

- [ ] **Step 3: Thread through SplitBlackholeKernel**

Ensure reader/compute/writer segments preserve the new attrs.

- [ ] **Step 4: Run tests and commit**

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git add tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc
git commit -m "blackhole: formalize GEMM ABI and dtype schema in split-after attrs"
```

---

### Task 5: Sync Progress and Memory After Verification

**Files:**
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Update stage status in `tasks/progress.md`**

Record:
- CB synchronization fix landed; GEMM numerical correctness achieved (or remaining gaps)
- scratch_l1 removed from main path
- Schema formalization landed
- Current next step

- [ ] **Step 2: Add stable lessons to `memory/general_dev.md`**

Record the reusable principle:
```text
TT-Metal CB data path requires both address sharing (get_write_ptr/get_read_ptr) AND synchronization
(cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front). Using real CB addresses without sync
primitives causes data races — data arrives at correct location but producers/consumers don't
coordinate on readiness.
```

- [ ] **Step 3: Record remaining limitations in `memory/bugs.md`**

If tilize/untilize, accessor schema, or other contract gaps remain, record them.

- [ ] **Step 4: Commit the state sync**

```bash
git add tasks/progress.md memory/general_dev.md memory/bugs.md
git commit -m "docs: sync stage2d GEMM contract progress and lessons"
```
