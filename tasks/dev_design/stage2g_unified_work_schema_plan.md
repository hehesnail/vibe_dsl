# Richer Work Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thin Blackhole runtime work ABI with a richer work descriptor schema that can actually express current copy and GEMM work ranges, and make `ExecutableSpec -> BlackholeModule` consume it directly.

**Architecture:** Keep the existing `KernelArgSpec` container and extend it with richer role-explicit work-descriptor kinds instead of introducing a new descriptor object in this round. Update split-after passes, runtime-spec extraction, codegen arg binding, and direct runtime arg building together so copy and GEMM can describe per-buffer `start/count/stride` plus logical work identity, rather than relying on `current_work_linear_id`, `tile_count`, or a single shared reader start field.

**Tech Stack:** TileLang/TVM C++, Blackhole runtime adapter, pytest

---

### Task 1: Write Failing Rich-Schema Tests

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Update copy runtime-arg expectations to richer schema**

Expected copy runtime-arg kinds:

```python
[
    "input_buffer_addr32",
    "output_buffer_addr32",
    "work_linear_id",
    "a_tile_start_id",
    "a_tile_num_tiles",
    "a_tile_stride",
    "output_tile_start_id",
    "output_tile_num_tiles",
    "output_tile_stride",
]
```

- [ ] **Step 2: Update GEMM segment-plan expectations to A/B-separated schema**

Reader expected kinds:

```python
[
    "input_buffer_addr32",
    "input_buffer_addr32",
    "work_linear_id",
    "a_tile_start_id",
    "a_tile_num_tiles",
    "a_tile_stride",
    "b_tile_start_id",
    "b_tile_num_tiles",
    "b_tile_stride",
    "k_tile_start_id",
    "num_k_tiles",
]
```

Compute expected kinds:

```python
["k_tile_start_id", "num_k_tiles"]
```

Writer expected kinds:

```python
[
    "work_linear_id",
    "output_tile_start_id",
    "output_tile_num_tiles",
    "output_tile_stride",
]
```

- [ ] **Step 3: Add fail-fast coverage**

Add tests that verify:

- copy/codegen/runtime reject divergent source/dest range or stride ≠ 1
- direct runtime reject unsupported richer schema combinations rather than silently defaulting

- [ ] **Step 4: Run targeted tests and verify RED**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k "copy_pass_attrs or copy_codegen_uses_runtime_schema" -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -k "fail or reject" -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "split_kernel_gemm_segment_plan" -v
```

Expected: FAIL because implementation still emits the weaker schema and lacks the new fail-fast coverage.

### Task 2: Replace Copy Runtime Schema With Explicit Equal-Range Contract

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`

- [ ] **Step 1: Replace default copy runtime-arg kinds**

Target shape:

```cpp
{"work_linear_id", "work_linear_id", "uint32", ""}
{"a_tile_start_id", "a_tile_start_id", "uint32", ""}
{"a_tile_num_tiles", "a_tile_num_tiles", "uint32", ""}
{"a_tile_stride", "a_tile_stride", "uint32", ""}
{"output_tile_start_id", "output_tile_start_id", "uint32", ""}
{"output_tile_num_tiles", "output_tile_num_tiles", "uint32", ""}
{"output_tile_stride", "output_tile_stride", "uint32", ""}
```

- [ ] **Step 2: Encode current supported copy contract explicitly**

Current formal support remains:

```cpp
a_tile_start_id == output_tile_start_id
a_tile_num_tiles == output_tile_num_tiles
a_tile_stride == 1
output_tile_stride == 1
```

Unsupported divergence must fail loudly in runtime/codegen paths.

- [ ] **Step 3: Run targeted copy tests and verify GREEN**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k "copy_pass_attrs or copy_codegen_uses_runtime_schema" -v
```

Expected: PASS

### Task 3: Replace GEMM Segment Schema With A/B Separation

**Files:**
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`

- [ ] **Step 1: Update GEMM reader runtime args**

Target reader schema:

```cpp
work_linear_id
a_tile_start_id
a_tile_num_tiles
a_tile_stride
b_tile_start_id
b_tile_num_tiles
b_tile_stride
k_tile_start_id
num_k_tiles
```

- [ ] **Step 2: Update compute and writer runtime args**

Compute:

```cpp
k_tile_start_id
num_k_tiles
```

Writer:

```cpp
work_linear_id
output_tile_start_id
output_tile_num_tiles
output_tile_stride
```

- [ ] **Step 3: Run targeted GEMM plan test and verify GREEN**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "split_kernel_gemm_segment_plan" -v
```

Expected: PASS

### Task 4: Teach Codegen and Direct Runtime the Richer Schema

**Files:**
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`

- [ ] **Step 1: Update codegen work-id binding**

`BindThreadIndex` may still reconstruct logical block indices from a work-id-like field, but it must not silently treat divergent source/dest starts as interchangeable. Current supported rule:

```cpp
prefer work_linear_id
else only allow copy fallback when a_tile_start_id == output_tile_start_id and both strides == 1
otherwise fail loudly
```

- [ ] **Step 2: Build richer runtime args explicitly**

For copy, current supported values:

```cpp
work_linear_id = current_work_linear_id
a_tile_start_id = current_work_linear_id
a_tile_num_tiles = 1
a_tile_stride = 1
output_tile_start_id = current_work_linear_id
output_tile_num_tiles = 1
output_tile_stride = 1
```

For GEMM reader, derive per-buffer ranges from logical `bx/by` reconstructed from `work_linear_id`:

```cpp
bx = work_linear_id % logical_grid_x
by = work_linear_id / logical_grid_x
a_tile_start_id = by
a_tile_num_tiles = num_k_tiles
a_tile_stride = 1
b_tile_start_id = bx
b_tile_num_tiles = num_k_tiles
b_tile_stride = logical_n_tiles
k_tile_start_id = 0
num_k_tiles = GetRuntimeNumKTiles(spec)
```

Writer uses:

```cpp
output_tile_start_id = work_linear_id
output_tile_num_tiles = 1
output_tile_stride = 1
```

Any unsupported richer combination must `ICHECK` instead of silently defaulting.

- [ ] **Step 3: Run targeted tests**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k "copy_codegen_uses_runtime_schema" -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "split_kernel_gemm_segment_plan or multicore_gemm" -v
```

Expected: PASS or environment-dependent SKIP only

### Task 5: Full Regression and Docs Sync

**Files:**
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Run full regression**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -v
```

- [ ] **Step 2: Sync docs and reusable learnings**

Record that P3 is partially formalized at the richer-schema level, with copy currently restricted to equal source/dest range and accessor schema still open.

- [ ] **Step 3: Commit**

```bash
git add tasks/dev_design/stage2g_unified_work_schema.md \
        tasks/dev_design/stage2g_unified_work_schema_plan.md \
        tasks/progress.md \
        tasks/dev_design/stage2d_ttmetal_contract_audit.md \
        tasks/dev_design/final_blackhole_backend_redesign.md \
        memory/general_dev.md \
        memory/bugs.md \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: formalize richer runtime work schema"
```
