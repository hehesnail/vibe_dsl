# Task 4: Replace `blackhole.segment_kind` With Direct Kernel-Kind Construction

## Scope

Replace `blackhole.segment_kind`-based slicing with direct kernel-kind construction while building `TTProgram`, then project that truth forward through `ExecutableSpec`.

In current code terms, the canonical truth lives on `TTKernelPlan.kind` / `TTKernel.kind` and the projected executable `segment_plan` records. The semantic meaning is still reader / compute / writer kernel role, but the task must use the actual owner objects that already exist in this repo.

This task must not introduce:

- a shared segment-slice analysis
- a replacement attr
- a new executable view type
- another pass that re-classifies kernels after planning

`SplitBlackholeKernel` may continue to do structural split/normalization, but it must stop publishing `blackhole.segment_kind`.

## Files

- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/build_tt_program.cc`
- Modify: `tilelang_repo/src/transform/materialize_blackhole_executable.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/target/tt_program_projection.h`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

## Execution Slices

1. add the failing segment-truth regressions
2. confirm planner/runtime still scan `segment_kind`
3. assign kernel kind directly while building `TTProgram`
4. project kernel records directly into executable `segment_plan`
5. switch runtime/codegen readers to projected kernel records
6. stop emitting and reading `blackhole.segment_kind`
7. rerun the GEMM/flash-attn suites and commit

- [ ] **Step 1: Write the failing segment-truth regressions**

Add one phase-bundle regression and one runtime-facing regression:

```python
def test_blackhole_phase_b_does_not_publish_segment_kind_attrs():
    ...

def test_blackhole_gemm_runtime_keeps_three_kernels_without_segment_kind_attr():
    ...
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k segment_kind
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k three_kernels_without_segment_kind_attr
```

Expected: FAIL because current planner/runtime still scan `AttrStmt("blackhole.segment_kind")`.

- [ ] **Step 3: Assign kernel kind directly while building `TTProgram`**

Assign reader / compute / writer kind at the point where the planner is already creating the kernel objects:

```cpp
TTKernel kernel = TTKernel(name, /*kind=*/InferKernelKind(region, spatial_plan, transport_plan),
                           /*core_type=*/InferCoreType(...),
                           abi_plan_index, payload);
program->kernels.push_back(kernel);
```

If the implementation prefers `TTKernelPlan` first and `TTKernel` later, the same rule applies: assign the kernel kind there, not through a side attr.

Any tiny matcher/visitor needed to recognize the current region stays local to the planner `.cc` and immediately feeds object construction.

- [ ] **Step 4: Project kernel records directly into executable `segment_plan`**

Use the existing projection path in [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h):

```cpp
Array<Any> segment_plan = tt_program_projection::GetSegmentPlanFromTTProgram(program);
executable.Set(String(executable_key::kSegmentPlan), segment_plan);
```

If kernel payload needs additional body/ABI fields, add them directly to the projected segment record here. Do not create another intermediate projection layer.

- [ ] **Step 5: Switch runtime/codegen readers to projected kernel records**

Replace attr-based readers and post-hoc extractors with direct reads of projected executable records:

```cpp
auto executable =
    tt_program_projection::RequireBlackholeExecutableProjection(f, "Blackhole runtime");
Array<Any> segment_plan =
    tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole runtime");
```

This cutover should delete logic such as:

- `SegmentBodyExtractor`
- `AttrStmt("blackhole.segment_kind", ...)` scanning
- any runtime-side kernel re-classification pass

Runtime/codegen should consume the kernel records that planning already projected.

- [ ] **Step 6: Stop emitting and reading `blackhole.segment_kind`**

Remove the emission path from `split_blackhole_kernel.cc` and remove every reader from:

- `lower_blackhole_ops.cc`
- `materialize_blackhole_executable.cc`
- `codegen_blackhole.cc`
- `rt_mod_blackhole.cc`

- [ ] **Step 7: Re-run the GEMM/flash-attn suites and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/build_tt_program.cc \
        tilelang_repo/src/transform/materialize_blackhole_executable.cc \
        tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/src/target/tt_program_projection.h \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: replace segment attrs with direct kernel kinds"
```
