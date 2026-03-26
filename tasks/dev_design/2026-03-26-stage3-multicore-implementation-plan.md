# Stage 3 Multi-Core Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement true multi-core runtime launch for Blackhole copy and GEMM by distributing work across multiple physical cores inside one TT-Metal `Program`.

**Architecture:** Stage 3 deliberately keeps lowering/codegen unchanged. `AssignBlackholeCores` must stop collapsing all logical work onto one core, and `BlackholeModule` must materialize one multi-core `Program` with per-core runtime args. Copy should validate the existing `blockIdx -> current_work_linear_id` chain; GEMM should validate the same chain through a DSL kernel that uses `bx/by` to express per-core tile offsets.

**Tech Stack:** TileLang TIR passes, Blackhole runtime/materialization, TT-Metal multi-core host APIs, TT-Sim, pytest.

---

## File Structure

- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
  - Remove the Stage 2D single-core distribution assumption.
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
  - Convert `ExecuteDirect()` from per-work-item `Program` creation to one `Program` + `CoreRangeSet`.
- Modify: `tilelang_repo/src/target/blackhole_module.h`
  - Keep helper declarations aligned with the multi-core materialization path.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  - Strengthen planner assertions for multi-core `core_plan`.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - Add TT-Sim true-E2E multi-core copy coverage.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  - Add a DSL multi-core GEMM kernel and TT-Sim runtime coverage.
- Modify: `tasks/progress.md`
  - Advance Stage 3 status and refresh test results.
- Modify: `tasks/dev_design/stage3_multicore_design.md`
  - Sync implementation notes if they diverge.
- Possibly modify: `memory/general_dev.md`
  - Record any stable multi-core host/runtime lessons discovered during implementation.
- Possibly modify: `memory/bugs.md`
  - Record reusable bugs only if a real issue is found.

---

### Task 1: Prove Multi-Core Core Planning in the Pass Layer

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`

- [ ] **Step 1: Add a failing planner test for `grid_x=2, grid_y=3`**

Add or extend a test in `test_blackhole_copy_pipeline.py` so it asserts that a grid-indexed staged copy produces six `work_packets` and six `physical_cores`, not one. Use the existing style that inspects split-after attrs.

```python
def test_blackhole_core_plan_uses_multiple_physical_cores_for_grid_copy():
    func = grid_indexed_staged_copy_kernel(M=96, N=64, grid_x=2, grid_y=3)
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    mod = tilelang.transform.AssignBlackholeCores()(mod)

    device_func = next(
        gv for gv, f in mod.functions.items()
        if isinstance(f, tvm.tir.PrimFunc) and int(f.attrs["calling_conv"]) == 2
    )
    core_plan = mod[device_func].attrs["blackhole.core_plan"]
    assert len(core_plan["work_packets"]) == 6
    assert len(core_plan["physical_cores"]) == 6
```

- [ ] **Step 2: Run the new planner test and confirm it fails on current code**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k multiple_physical_cores -q
```

Expected: FAIL because `AssignBlackholeCores` still hard-codes one physical core.

- [ ] **Step 3: Implement the minimal `AssignBlackholeCores` fix**

Update `CalculateWorkDistribution()` in `assign_blackhole_cores.cc` so `cores_needed` is no longer fixed at 1.

```c++
void AssignBlackholeCores::CalculateWorkDistribution(CoreAssignment& assignment) {
  const int total_work = std::max(1, assignment.grid_x * assignment.grid_y);
  const int available_cores = kBlackholeGridX * kBlackholeGridY;
  assignment.cores_needed = std::min(total_work, available_cores);
  assignment.work_per_core = (total_work + assignment.cores_needed - 1) /
                             assignment.cores_needed;
}
```

- [ ] **Step 4: Re-run the planner test and the existing copy pipeline suite**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
```

Expected: PASS with no regression in the single-core planner assertions.

- [ ] **Step 5: Commit the pass-layer multi-core planning change**

```bash
git add tilelang_repo/src/transform/assign_blackhole_cores.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
git commit -m "blackhole: enable multi-core work distribution in core planning"
```

---

### Task 2: Convert `BlackholeModule` to One Multi-Core Program

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.h`

- [ ] **Step 1: Add a failing runtime test that requires multi-core launch**

Extend `test_blackhole_copy_runtime.py` with a grid-indexed staged copy runtime case that uses `grid_x=2, grid_y=3` and compares against the input tensor. Reuse TT-Sim setup conventions already present in the file.

```python
def test_blackhole_copy_runtime_multicore_grid():
    func = grid_indexed_staged_copy_kernel(M=96, N=64, grid_x=2, grid_y=3)
    A = torch.randn((96, 64), dtype=torch.float16)
    rt_mod = tilelang.compile(func, out_idx=[1], target="blackhole")
    out = rt_mod(A)
    torch.testing.assert_close(out, A, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run the new runtime test and confirm current behavior is still serialized**

Run:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -k multicore_grid -q
```

Expected: either FAIL or still run through the old serialized path. Record the exact failure before changing runtime materialization.

- [ ] **Step 3: Refactor `BlackholeModule` helper signatures to accept a core variant**

Change the helper declarations/definitions so CB and kernel creation accept the same core-spec variant TT-Metal already supports.

```c++
using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

static void CreateCircularBuffersFromSpec(
    Program& program, const CoreSpec& core_spec, const ExecutableSpec& spec);

static KernelHandle CreateKernelFromSpec(
    Program& program, const CoreSpec& core_spec,
    const KernelSpec& kernel, const std::string& kernel_path);
```

- [ ] **Step 4: Refactor `ExecuteDirect()` to materialize one `Program`**

Move `Program`, CB creation, and kernel creation outside the work-item loop. Build a `CoreRangeSet` from all assigned cores, keep per-core `SetRuntimeArgs()` inside the loop, and enqueue exactly once.

```c++
std::set<CoreCoord> all_cores;
for (const auto& item : work_items) {
  all_cores.insert(item.core);
}

std::set<CoreRange> core_ranges;
for (const auto& core : all_cores) {
  core_ranges.insert(CoreRange(core, core));
}
CoreRangeSet core_spec(core_ranges);

Program program = CreateProgram();
CreateCircularBuffersFromSpec(program, core_spec, spec);

std::vector<KernelHandle> kernels;
for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
  kernels.push_back(CreateKernelFromSpec(
      program, core_spec, spec.kernels[ki], kernel_paths[ki]));
}

for (const auto& item : work_items) {
  for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    auto args = BuildRuntimeArgsFromSpec(
        spec.kernels[ki], spec, item.work_id, runtime_ctx);
    SetRuntimeArgs(program, kernels[ki], item.core, args);
  }
}
```

- [ ] **Step 5: Rebuild and run copy runtime validation**

Run:

```bash
cmake --build tilelang_repo/build -j32
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py
```

Expected: existing copy runtime tests still pass, including the new multi-core case.

- [ ] **Step 6: Commit the multi-core runtime materialization change**

```bash
git add tilelang_repo/src/target/blackhole_module.cc \
        tilelang_repo/src/target/blackhole_module.h \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py
git commit -m "blackhole: launch direct runtime with one multi-core program"
```

---

### Task 3: Validate GEMM on Multiple Cores Through DSL `bx/by`

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add a DSL multi-core GEMM kernel factory and a failing TT-Sim test**

Add a test kernel that uses `with T.Kernel(Nt, Mt) as (bx, by)` and slices A/B/C using `bx/by`. Keep `transpose_B=True` so the existing Stage 2D direct-path GEMM contract remains exercised.

```python
def grid_gemm_kernel(M=64, N=64, K=128, block_M=32, block_N=32):
    Mt = M // block_M
    Nt = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(Nt, Mt) as (bx, by):
            A_shared = T.alloc_shared((block_M, K), "bfloat16")
            B_shared = T.alloc_shared((block_N, K), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.copy(A[by * block_M:(by + 1) * block_M, 0:K], A_shared)
            T.copy(B[bx * block_N:(bx + 1) * block_N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[
                by * block_M:(by + 1) * block_M,
                bx * block_N:(bx + 1) * block_N
            ])

    return main

def test_blackhole_gemm_multicore_basic():
    A = torch.randn((64, 128), dtype=torch.bfloat16)
    B = torch.randn((64, 128), dtype=torch.bfloat16)
    ref = torch.matmul(A.float(), B.float().transpose(0, 1))
    rt_mod = tilelang.compile(grid_gemm_kernel(), out_idx=[2], target="blackhole")
    out = rt_mod(A, B)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: Run the new GEMM test before changing anything else**

Run:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k multicore_basic -q
```

Expected: this should fail until the multi-core host launch path is fully working.

- [ ] **Step 3: Run the full GEMM suite after the runtime refactor**

Run:

```bash
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
```

Expected: the new multi-core case passes and the existing single-core Stage 2D tests do not regress.

- [ ] **Step 4: Commit the multi-core GEMM coverage**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: add multi-core GEMM TT-Sim coverage"
```

---

### Task 4: Final Regression, Documentation, and Completion

**Files:**
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/stage3_multicore_design.md`
- Possibly modify: `memory/general_dev.md`
- Possibly modify: `memory/bugs.md`

- [ ] **Step 1: Run the full Stage 3 verification set**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py
source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
cmake --build tilelang_repo/build -j32
```

Expected:
- copy pipeline PASS
- copy runtime PASS
- GEMM PASS
- rebuild succeeds

- [ ] **Step 2: Update progress and design docs to reflect Stage 3 results**

Update:

- `tasks/progress.md`
  - mark Stage 3 implemented if verification passes
  - refresh latest test results
  - record whether remaining TT-Metal contract work is still P0/P3-P5 only
- `tasks/dev_design/stage3_multicore_design.md`
  - change status from `待实施` to implemented if appropriate
- `memory/general_dev.md`
  - add one stable lesson only if the runtime multi-core launch reveals a reusable pattern
- `memory/bugs.md`
  - add a bug entry only if a real reusable issue was found and fixed

- [ ] **Step 3: Commit documentation sync**

```bash
git add tasks/progress.md \
        tasks/dev_design/stage3_multicore_design.md \
        memory/general_dev.md \
        memory/bugs.md
git commit -m "docs: record stage3 multicore runtime completion"
```

- [ ] **Step 4: Push and clear execution state**

Run:

```bash
git push
ps -eo pid,etimes,cmd | rg 'pytest|cmake --build|ninja|git push' || true
```

Expected: push succeeds and no residual background test/build/push processes remain before reporting completion.
