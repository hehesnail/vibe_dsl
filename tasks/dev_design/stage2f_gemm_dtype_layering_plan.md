# GEMM Dtype Layering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize GEMM dtype layering in Blackhole `gemm_contract` and make `ExecutableSpec -> BlackholeModule` consume explicit dtype schema instead of implicit bf16/fp32 assumptions.

**Architecture:** Extend the split-after GEMM contract schema first, keep runtime support limited to the currently validated bf16/fp32 combination, and make unsupported combinations fail explicitly. Do not mix accessor/work-schema changes into this task.

**Tech Stack:** TileLang/TVM C++, Blackhole runtime adapter, pytest

---

### Task 1: Update Design Docs

**Files:**
- Modify: `tasks/dev_design/stage3_multicore_design.md`
- Create: `tasks/dev_design/stage2f_gemm_dtype_layering.md`

- [ ] **Step 1: Sync Stage 3 doc**

Update the Stage 3 note so `tvm_ffi` wrapper/export is recorded as an independently fixed follow-up, not an unresolved blocker.

- [ ] **Step 2: Add P0 dtype-layering design**

Write the scope, protocol changes, validation plan, and non-goals in `tasks/dev_design/stage2f_gemm_dtype_layering.md`.

### Task 2: Write Failing GEMM Contract Tests

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add attr-materialization assertions for layered dtype fields**

Add assertions for:

```python
assert str(contract["a_tensor_dtype"]) == "Float16_b"
assert str(contract["b_tensor_dtype"]) == "Float16_b"
assert str(contract["c_tensor_dtype"]) == "Float32"
assert str(contract["a_cb_dtype"]) == "Float16_b"
assert str(contract["b_cb_dtype"]) == "Float16_b"
assert str(contract["c_cb_dtype"]) == "Float32"
assert str(contract["accumulator_dtype"]) == "Float32"
```

- [ ] **Step 2: Add `ExecutableSpec` extraction test**

Add a test that lowers GEMM, exports through Blackhole runtime module creation, and inspects the extracted `ExecutableSpec`-backed source/metadata path enough to assert the same dtype fields were preserved into runtime-visible contract state.

- [ ] **Step 3: Run targeted GEMM tests and verify RED**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "contract_attr_is_materialized or executable_spec" -v
```

Expected: FAIL because the new dtype fields are not present yet.

### Task 3: Implement Layered Dtype Contract

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.h`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`

- [ ] **Step 1: Extend `blackhole.gemm_contract` emission**

Replace the old `ab_dtype` / `c_dtype` emission with explicit layered fields derived from current buffers:

```cpp
gemm_contract.Set("a_tensor_dtype", String(DataTypeToDataFormat(gemm_ab_dtype_)));
gemm_contract.Set("b_tensor_dtype", String(DataTypeToDataFormat(gemm_ab_dtype_)));
gemm_contract.Set("c_tensor_dtype", String(DataTypeToDataFormat(gemm_c_dtype_)));
gemm_contract.Set("a_cb_dtype", String(DataTypeToDataFormat(gemm_ab_dtype_)));
gemm_contract.Set("b_cb_dtype", String(DataTypeToDataFormat(gemm_ab_dtype_)));
gemm_contract.Set("c_cb_dtype", String(DataTypeToDataFormat(gemm_c_dtype_)));
gemm_contract.Set("accumulator_dtype", String(DataTypeToDataFormat(gemm_c_dtype_)));
```

- [ ] **Step 2: Extend `GemmContractSpec` and extraction**

Add the corresponding fields to `GemmContractSpec` and teach `ExtractGemmContract` to populate them.

- [ ] **Step 3: Keep `enabled` logic stable**

Do not make enablement depend on the new dtype strings; keep it keyed on buffer names and M/N/K so old valid kernels still materialize a contract.

### Task 4: Make Direct Runtime Consume the New Schema

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`

- [ ] **Step 1: Add explicit dtype validators**

Add helpers that compare `DLTensor.dtype` against the string-valued contract fields for GEMM inputs/outputs.

- [ ] **Step 2: Replace hard-coded assumptions with contract checks**

Update host tilize/untilize path so it:

```cpp
ICHECK_EQ(GetDLTensorDataFormat(*tensor), expected_contract_dtype);
ICHECK_EQ(gemm.a_cb_dtype, gemm.a_tensor_dtype);
ICHECK_EQ(gemm.b_cb_dtype, gemm.b_tensor_dtype);
ICHECK_EQ(gemm.c_cb_dtype, gemm.accumulator_dtype);
```

and then preserves the current bf16/fp32-only runtime support with explicit fatal checks when the contract requests a different combination.

- [ ] **Step 3: Run targeted GEMM tests and verify GREEN**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "contract_attr_is_materialized or executable_spec" -v
```

Expected: PASS

### Task 5: Full Verification and Sync

**Files:**
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Run regression verification**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -v
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -v
```

Expected: GEMM suite passes/skips as environment allows, `tvm_ffi` export passes, copy pipeline remains green.

- [ ] **Step 2: Sync progress and reusable learnings**

Update `tasks/progress.md` to record P0 dtype layering status, and update `memory/` only if the implementation reveals a reusable lesson or bug pattern.

- [ ] **Step 3: Commit**

```bash
git add tasks/dev_design/stage3_multicore_design.md \
        tasks/dev_design/stage2f_gemm_dtype_layering.md \
        tasks/dev_design/stage2f_gemm_dtype_layering_plan.md \
        tasks/progress.md \
        memory/general_dev.md \
        memory/bugs.md \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/target/blackhole_module.h \
        tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: formalize gemm dtype layering"
```
