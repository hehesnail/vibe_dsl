# Phase B Spatial Program IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Status:** Implemented on `2026-04-06`. The landed slice includes typed `SpatialProgram` / `ProgramPhase`, copy/GEMM fast-paths, the `flash-attn` multi-phase gate, module-scope phase aggregation in `tl.device_programs`, and minimal `LowerBlackholeOps` spatial-summary consumption. Validation evidence is recorded in `tasks/progress.md`.

**Goal:** Land the first end-to-end `Phase B` slice by introducing typed `SpatialProgram` / `ProgramPhase`, validating copy and GEMM fast-path construction, and proving one non-trivial flash-attn multi-phase spatial gate on the main Blackhole pipeline.

**Architecture:** Extend the existing companion IR object set with a typed spatial layer that hangs member-local truth on `PrimFunc.attrs["tl.spatial_program"]` and cross-function phase truth on `tl.device_programs`. A new lowering pass will consume frozen `SemanticProgram` plus stable spatial analysis attrs, produce canonical fast-path spatial programs for copy/GEMM, synthesize a multi-phase flash-attn spatial structure, and feed a validator plus minimal `LowerBlackholeOps` consumption without attempting the Phase C TT cutover yet.

**Tech Stack:** C++ TVM/TIR transform passes, TVM FFI reflection objects, Python transform wrappers, pytest transform/target tests.

---

### Task 1: Add failing Phase B structure tests

**Files:**
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Write failing transform tests for copy/GEMM/flash-attn**

```python
def test_copy_spatial_program_uses_single_transfer_fast_path():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel)
    program = mod["main"].attrs["tl.spatial_program"]
    assert [task.kind for task in program.tasks] == ["transfer"]


def test_gemm_spatial_program_uses_reader_compute_writer_fast_path():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel)
    program = mod["main"].attrs["tl.spatial_program"]
    assert [task.name for task in program.tasks] == ["reader", "compute", "writer"]


def test_flash_attention_spatial_program_exposes_multi_phase_channels():
    mod = _prepare_blackhole_phase_b_module(_lower_flash_attention_example(mha_example))
    program = mod["main_kernel"].attrs["tl.spatial_program"]
    assert len(program.phases) >= 2
    assert len(program.channels) >= 1
```

- [ ] **Step 2: Run transform tests to verify they fail**

Run: `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q`
Expected: FAIL because `LowerToSpatialProgram` / `ValidateSpatialProgram` do not exist and `tl.spatial_program` is missing.

- [ ] **Step 3: Write failing target-side assertions for pipeline presence**

```python
def test_blackhole_gemm_pipeline_attaches_spatial_program():
    artifact = _build_blackhole_gemm_artifact()
    assert "tl.spatial_program" in artifact.device_mod["main"].attrs


def test_flash_attention_pipeline_attaches_spatial_program():
    artifact = _build_flash_attention_artifact()
    assert "tl.spatial_program" in artifact.device_mod["main_kernel"].attrs
```

- [ ] **Step 4: Run the focused target tests to verify they fail**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k spatial_program -q`
Expected: FAIL with missing `tl.spatial_program`.

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k spatial_program -q`
Expected: FAIL with missing `tl.spatial_program`.

### Task 2: Extend the companion IR object model with typed spatial objects

**Files:**
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/collect_device_programs.cc`

- [ ] **Step 1: Add typed object declarations for the spatial layer**

```cpp
class TaskNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String phase_name;
  ffi::Array<ffi::String> update_names;
  ffi::Array<TIRAnchor> anchors;
};
```

- [ ] **Step 2: Add constructors, reflection, and FFI registration**

```cpp
Task::Task(ffi::String name, ffi::String kind, ffi::String phase_name,
           ffi::Array<ffi::String> update_names, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<TaskNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->phase_name = std::move(phase_name);
  n->update_names = std::move(update_names);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}
```

- [ ] **Step 3: Extend `TLDeviceProgramInfo` with phase ownership**

```cpp
class TLDeviceProgramInfoNode : public GlobalInfoNode {
 public:
  ffi::String root_symbol;
  ffi::Array<ffi::String> member_funcs;
  ffi::Array<ProgramPhase> phases;
};
```

- [ ] **Step 4: Keep Stage 0 behavior stable**

Run: `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -k device_program_registry_is_collected_before_split_host_device -q`
Expected: PASS with the existing assertions on `root_symbol` and `member_funcs`.

### Task 3: Implement `LowerToSpatialProgram` and `ValidateSpatialProgram`

**Files:**
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/CMakeLists.txt`

- [ ] **Step 1: Add a failing pass-registration check in Python tests**

```python
def test_spatial_passes_are_registered():
    assert hasattr(tilelang.transform, "LowerToSpatialProgram")
    assert hasattr(tilelang.transform, "ValidateSpatialProgram")
```

- [ ] **Step 2: Run the single registration test to verify it fails**

Run: `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k registered -q`
Expected: FAIL because the wrappers are missing.

- [ ] **Step 3: Implement the lowering pass with two paths**

```cpp
if (IsSimpleCopyFastPath(program, func)) {
  return BuildCopySpatialProgram(program, func);
}
if (IsSimpleGemmFastPath(program, func)) {
  return BuildGemmSpatialProgram(program, func);
}
return BuildGeneralSpatialProgram(program, func, work_info, pipeline_stages);
```

- [ ] **Step 4: Implement the validator with hard checks**

```cpp
ICHECK(!program->tasks.empty()) << "ValidateSpatialProgram requires at least one task";
ICHECK(!program->phases.empty()) << "ValidateSpatialProgram requires at least one phase";
ICHECK_EQ(program->member_func, global_symbol);
```

- [ ] **Step 5: Export Python wrappers and compile the new files**

```python
def LowerToSpatialProgram():
    return tvm.ffi.get_global_func("tl.transform.LowerToSpatialProgram")()


def ValidateSpatialProgram():
    return tvm.ffi.get_global_func("tl.transform.ValidateSpatialProgram")()
```

- [ ] **Step 6: Run the transform test subset**

Run: `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'copy or gemm or flash_attention or registered' -q`
Expected: PASS.

### Task 4: Wire SpatialProgram into the Blackhole lowering pipeline

**Files:**
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`

- [ ] **Step 1: Insert the new passes after semantic validation**

```python
device_mod = tilelang.transform.LiftStatefulSemanticIR()(device_mod)
device_mod = tilelang.transform.ValidateStatefulSemanticIR()(device_mod)
device_mod = tilelang.transform.ValidateSemanticRefinement()(device_mod)
device_mod = tilelang.transform.LowerToSpatialProgram()(device_mod)
device_mod = tilelang.transform.ValidateSpatialProgram()(device_mod)
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)
```

- [ ] **Step 2: Make `LowerBlackholeOps` consume spatial truth before raw fallback**

```cpp
if (auto spatial_program = func->GetAttr<SpatialProgram>(attr::kTLSpatialProgram)) {
  lowering_requirements = BuildLoweringRequirementsFromSpatialProgram(spatial_program.value());
}
if (lowering_requirements.empty()) {
  lowering_requirements = BuildLoweringRequirementsFromAnalysis(func);
}
```

- [ ] **Step 3: Re-run target-side focused tests**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k spatial_program -q`
Expected: PASS.

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k spatial_program -q`
Expected: PASS.

### Task 5: Regressions, docs, and branch completion

**Files:**
- Modify: `tasks/progress.md`
- Modify: `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md` (only if a reusable bug pattern is discovered)

- [ ] **Step 1: Run the shared Phase B transform and target checks**

Run: `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q`
Expected: PASS.

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
Expected: PASS.

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
Expected: PASS.

- [ ] **Step 2: Re-run the zero-regression baseline**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
Expected: PASS / existing skips / xfails only.

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
Expected: PASS.

- [ ] **Step 3: Update progress and design docs to reflect the landed Phase B slice**

```md
- `SpatialProgram` / `ProgramPhase` typed companion objects are now materialized and validated.
- copy / GEMM fast-paths no longer require ad-hoc spatial reconstruction inside `LowerBlackholeOps`.
- flash-attn now has a checked multi-phase spatial gate.
```

- [ ] **Step 4: Commit and push the completed slice**

```bash
git add docs/superpowers/plans/2026-04-06-phase-b-spatial-program-ir.md \
        tilelang_repo/src/transform/common/semantic_program.h \
        tilelang_repo/src/transform/common/semantic_program.cc \
        tilelang_repo/src/transform/collect_device_programs.cc \
        tilelang_repo/src/transform/lower_to_spatial_program.cc \
        tilelang_repo/src/transform/validate_spatial_program.cc \
        tilelang_repo/tilelang/transform/__init__.py \
        tilelang_repo/tilelang/engine/lower.py \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
        tasks/progress.md tasks/dev_design/stage4_phase_b_spatial_ir.md memory/general_dev.md memory/bugs.md
git commit -m "feat: land phase b spatial program ir slice"
git push
```
