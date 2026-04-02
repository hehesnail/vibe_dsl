# Stage 3 GEMM Multi-Core Direct Hang Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the formal `BlackholeModule` direct host path run the `64x64x128`, `2x2`, `transpose_B=True` multi-core GEMM without hanging and with correct numerics.

**Architecture:** This plan deliberately ignores the outer `tilelang.compile(..., execution_backend="tvm_ffi")` export/runtime wrapper bug and stays on the formal direct host path: `lower(..., target="blackhole") -> artifact.codegen_mod["main"] -> BlackholeModule::ExecuteDirect()`. The implementation first hardens host-side GEMM input/output transfer against single-core shape assumptions, then narrows the device-side hang by dumping and validating the multi-core reader/compute/writer contract, and finally fixes the smallest direct-path root cause that remains.

**Tech Stack:** TileLang lowering, Blackhole target runtime/materialization, TT-Metal direct execution, TT-Sim, pytest.

---

## File Structure

- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  - Keep the multi-core GEMM direct-call regression as the red/green gate.
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
  - Fix host-side GEMM transfer/readback assumptions and, if needed, the direct launch behavior.
- Possibly modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - Only if the direct-path investigation proves a segment/runtime schema mismatch on the Blackhole main path.
- Modify: `tasks/progress.md`
  - Record the actual Stage 3 state and the separate `tilelang.compile/tvm_ffi` blocker.
- Modify: `tasks/dev_design/stage3_multicore_design.md`
  - Sync implementation notes if the direct-path blocker resolution changes Stage 3 status.
- Modify: `memory/bugs.md`
  - Record the independent wrapper/export blocker if still present after this task.

---

### Task 1: Lock the Direct-Path GEMM Failure as a Reproducible Test

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Keep the direct-path multicore GEMM regression test as the canonical red test**

The file should contain exactly one Stage 3 multi-core GEMM regression on the formal path:

```python
def test_blackhole_gemm_multicore_direct_call():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n, k = 64, 64, 128
    a_torch = torch.randn(m, k, dtype=torch.bfloat16)
    b_torch = torch.randn(n, k, dtype=torch.bfloat16)
    c_output = torch.zeros(m, n, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    with Target("blackhole"):
        artifact = lower(multicore_gemm_kernel(M=m, N=n, K=k), target=Target("blackhole"))

    device_main = {str(gv): f for gv, f in artifact.device_mod.functions.items()}['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]
    assert int(core_plan["logical_grid_x"]) == 2
    assert int(core_plan["logical_grid_y"]) == 2
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1]

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(c_output, c_ref, atol=2e-1, rtol=2e-1)
```

- [ ] **Step 2: Run the red test and record the current failure**

Run:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_multicore_direct_call -vv
```

Expected: FAIL. At the time this plan was written, the first confirmed red failure on the formal direct path was:

```text
Unexpected GEMM tensor shape for A: got (64, 128), expected (32, 128)
```

If the failure evolves after earlier edits, record the new failing symptom before changing production code.

- [ ] **Step 3: Commit the test-only state if the test needed tightening**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "test: lock multicore GEMM direct-path regression"
```

If no test-only change was needed, skip this commit step.

---

### Task 2: Remove Single-Core Shape Assumptions from Host GEMM Transfer

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Fix `BuildInputTransferData()` to use runtime tensor shapes for whole-tensor tilize**

Replace the single-core-shape assumptions with runtime tensor dimensions.

Expected shape handling:

```c++
const uint32_t rows = static_cast<uint32_t>(tensor->shape[0]);
const uint32_t cols = static_cast<uint32_t>(tensor->shape[1]);

if (binding.name == gemm.b_buffer && gemm.transpose_B) {
  tiled = tilize_nfaces(TransposeRowMajor2D(raw, rows, cols), cols, rows);
} else {
  std::vector<uint16_t> row_major(raw, raw + static_cast<size_t>(rows) * cols);
  tiled = tilize_nfaces(row_major, rows, cols);
}
```

Do not leave checks that force A to equal `gemm.M x gemm.K` or B to equal `gemm.N x gemm.K` for the whole runtime tensor when the direct path is launching a tiled multi-core grid.

- [ ] **Step 2: Fix `CopyOutputFromDeviceBuffer()` to untilize using runtime output tensor shape**

Use the bound output tensor shape instead of `gemm.M`/`gemm.N` from the single-core contract.

```c++
const uint32_t rows = static_cast<uint32_t>(binding.tensor->shape[0]);
const uint32_t cols = static_cast<uint32_t>(binding.tensor->shape[1]);
const size_t numel = static_cast<size_t>(rows) * cols;
std::vector<float> row_major = untilize_nfaces(tiled_vec, rows, cols);
```

- [ ] **Step 3: Rebuild before rerunning the test**

Run:

```bash
cmake --build tilelang_repo/build -j32
```

Expected: build succeeds so the red test actually loads the new `libtilelang.so`.

- [ ] **Step 4: Re-run the red test and capture the next symptom**

Run:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_multicore_direct_call -vv
```

Expected:
- If the test passes, stop here and move to Task 4.
- If the shape failure is gone but the test now hangs or fails later in execution, that is the intended signal to continue into Task 3.

- [ ] **Step 5: Commit the host-transfer fix**

```bash
git add tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: use whole-tensor GEMM transfer shapes on direct path"
```

---

### Task 3: Investigate and Fix the Remaining Direct-Path Device Hang

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Possibly modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Dump the multi-core direct-path evidence before changing code**

Use the existing red test kernel to inspect:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_multicore_direct_call -vv -s
```

And inspect the generated plan/source if needed:

```bash
python - <<'PY'
from tvm.target import Target
from tilelang.engine.lower import lower
from tilelang_repo.testing.python.target.blackhole.test_blackhole_gemm import multicore_gemm_kernel

with Target("blackhole"):
    artifact = lower(multicore_gemm_kernel(), target=Target("blackhole"))

device_main = {str(gv): f for gv, f in artifact.device_mod.functions.items()}['I.GlobalVar("main_kernel")']
print(device_main.attrs["blackhole.core_plan"])
print(device_main.script())
PY
```

Expected evidence to compare:
- `core_plan.work_packets` exactly match `[(offset,count)=(0,1),(1,1),(2,1),(3,1)]`
- reader/writer builtins still use `bx/by`-derived tile indices
- no oversubscription is present

- [ ] **Step 2: Form one concrete hypothesis from the evidence**

Write down a single root-cause hypothesis in your working notes before editing code. Valid examples:

- “reader/writer registration is correct, but the launch path is using a different core set than the segment expects”
- “one GEMM segment still behaves as if `current_work_linear_id` is always 0”
- “the multi-core direct path deadlocks because the three segments are not registered over the same core range”

Do not change code until the hypothesis points to one specific layer.

- [ ] **Step 3: Make the smallest code change that tests that hypothesis**

Examples of acceptable minimal fixes:

```c++
// Example only — use the real hypothesis you proved.
ICHECK(reader_launch_cores == compute_launch_cores && compute_launch_cores == writer_launch_cores)
    << "GEMM multi-core direct path requires all segments to share the same launch core set";
```

or:

```c++
// Example only — if the evidence proves a segment/runtime mismatch.
for (const auto& item : work_items) {
  auto runtime_args = BuildRuntimeArgsFromSpec(..., item.work_id, ...);
  SetRuntimeArgs(program, kernels[ki], item.core, runtime_args);
}
```

The change must be driven by the evidence from Step 1, not by guesswork.

- [ ] **Step 4: Rebuild and rerun the direct-path GEMM regression**

Run:

```bash
cmake --build tilelang_repo/build -j32
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_multicore_direct_call -vv
```

Expected: PASS without hanging and with numerics inside tolerance.

- [ ] **Step 5: Commit the direct-path hang fix**

```bash
git add tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: fix multicore GEMM direct-path hang"
```

Only include `rt_mod_blackhole.cc` if it was actually changed.

---

### Task 4: Final Verification and Documentation Sync

**Files:**
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/stage3_multicore_design.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Run the full verification set**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
cmake --build tilelang_repo/build -j32
```

Expected at minimum:
- copy pipeline remains green
- copy runtime remains green
- `test_blackhole_gemm_basic` remains green
- `test_blackhole_gemm_multicore_direct_call` is green

- [ ] **Step 2: Record the separate `tilelang.compile/tvm_ffi` wrapper blocker**

Update `memory/bugs.md` with a reusable bug entry if still true:

- Blackhole `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` export/runtime path generates invalid host shim C (`kernel_error_code = ;`)
- minimal single-core copy also reproduces it
- this is outside the formal direct host path and remains unresolved

- [ ] **Step 3: Update progress and Stage 3 status truthfully**

Update:

- `tasks/progress.md`
  - record the actual Stage 3 status after the direct-path GEMM result
  - include current test counts
  - record that the compile/runtime wrapper is a separate blocker if still unresolved
- `tasks/dev_design/stage3_multicore_design.md`
  - update status/notes if GEMM multi-core direct path is now fixed

- [ ] **Step 4: Commit docs and push**

```bash
git add tasks/progress.md \
        tasks/dev_design/stage3_multicore_design.md \
        memory/bugs.md
git commit -m "docs: record multicore GEMM direct-path status"
git push
```

- [ ] **Step 5: Clear execution state before reporting completion**

Run:

```bash
ps -eo pid,etimes,cmd | rg 'pytest|cmake --build|ninja|git push' || true
```

Expected: no residual background test/build/push processes remain.
