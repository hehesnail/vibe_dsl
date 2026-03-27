# Accessor Common Runtime Args Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize accessor compile-time/common-runtime schema for Blackhole copy and GEMM kernels, while keeping direct runtime execution support limited to interleaved accessors and fail-fast for richer layouts.

**Architecture:** Extend the existing `accessors` path instead of creating a second schema. `LowerBlackholeOps` will emit full accessor descriptors plus kernel-level `common_runtime_args`; `rt_mod_blackhole` will extract them into `KernelSpec`; `BlackholeModule` will consume the richer schema but explicitly reject non-interleaved or CRTA-bearing execution for now.

**Tech Stack:** C++, TVM TIR attrs, TileLang Blackhole runtime module, pytest

---

### Task 1: Expand Schema And Extraction

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.h`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Write the failing structure tests**

```python
accessors = segment_plan[0]["accessors"]
assert int(accessors[0]["compile_time_arg_offset"]) == 0
assert int(accessors[0]["compile_time_arg_count"]) == 2
assert int(accessors[0]["common_runtime_arg_offset"]) == 0
assert int(accessors[0]["common_runtime_arg_count"]) == 0
assert "args_config_bits" in accessors[0]
assert "common_runtime_args" in segment_plan[0]
assert len(segment_plan[0]["common_runtime_args"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_pass_attrs -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_contract_attr_is_materialized -q
```

Expected: FAIL because accessors only expose `slot/layout/memory_space` and segments have no `common_runtime_args`.

- [ ] **Step 3: Extend runtime-facing structs**

```cpp
struct AccessorSpec {
  std::string buffer;
  uint32_t compile_time_arg_offset = 0;
  uint32_t compile_time_arg_count = 0;
  uint32_t common_runtime_arg_offset = 0;
  uint32_t common_runtime_arg_count = 0;
  uint32_t args_config_bits = 0;
  std::string layout;
  std::string memory_space;
};

struct KernelArgSpec {
  std::string name;
  std::string kind;
  std::string dtype;
  std::string buffer;
};

struct KernelSpec {
  ...
  std::vector<AccessorSpec> accessors;
  std::vector<KernelArgSpec> common_runtime_args;
};
```

- [ ] **Step 4: Teach `rt_mod_blackhole` to extract and encode the richer schema**

```cpp
if (auto v = accessor_info.Get("compile_time_arg_offset")) {
  accessor.compile_time_arg_offset = ...
}
if (auto v = accessor_info.Get("common_runtime_arg_count")) {
  accessor.common_runtime_arg_count = ...
}
if (auto v = segment.Get("common_runtime_args")) {
  info.common_runtime_args =
      ExtractRuntimeArgsFromArray(Downcast<ffi::Array<ffi::Any>>(v.value()));
}
```

- [ ] **Step 5: Run targeted tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_pass_attrs -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_contract_attr_is_materialized -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tilelang_repo/src/target/blackhole_module.h tilelang_repo/src/target/rt_mod_blackhole.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: extend accessor schema extraction"
```

### Task 2: Emit Richer Accessor Schema From Lowering

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Write the failing lowering tests for `common_runtime_args`**

```python
segment_plan = func.attrs["blackhole.segment_plan"]
reader = next(item for item in segment_plan if str(item["kind"]) == "reader")
assert "common_runtime_args" in reader
assert list(reader["common_runtime_args"]) == []
assert int(reader["accessors"][0]["compile_time_arg_offset"]) == 0
assert int(reader["accessors"][1]["compile_time_arg_offset"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_pass_attrs -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_multicore_gemm_lowering_respects_transposed_b_layout -q
```

Expected: FAIL because lowering still emits the legacy accessor maps.

- [ ] **Step 3: Replace `slot`-only descriptors with full accessor metadata**

```cpp
struct AccessorDescriptor {
  std::string segment_kind;
  std::string buffer_name;
  int compile_time_arg_offset = 0;
  int compile_time_arg_count = 2;
  int common_runtime_arg_offset = 0;
  int common_runtime_arg_count = 0;
  int args_config_bits = 1;  // IsDram interleaved
  std::string layout = "interleaved";
  std::string memory_space = "dram";
};
```

- [ ] **Step 4: Emit empty `common_runtime_args` arrays for current interleaved segments**

```cpp
if (!common_runtime_args.empty()) {
  segment.Set("common_runtime_args", common_runtime_args);
} else {
  segment.Set("common_runtime_args", Array<Any>{});
}
```

- [ ] **Step 5: Keep builtin accessor offsets wired to compile-time offsets**

```cpp
const int accessor_offset = GetReadAccessorOffset(...);
MakeBlackholeCall(blackhole_read_tile_to_cb(), {..., IntImm32(accessor_offset)});
```

- [ ] **Step 6: Run lowering/codegen regression**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected: copy suite `.....x..............`, gemm suite `.....ss`.

- [ ] **Step 7: Commit**

```bash
git add tilelang_repo/src/transform/lower_blackhole_ops.h tilelang_repo/src/transform/lower_blackhole_ops.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: emit richer accessor lowering schema"
```

### Task 3: Fail-Fast Runtime Support For Unsupported Accessor Execution

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add a focused fail-fast test**

```python
with pytest.raises(tvm.error.InternalError, match="common runtime args|interleaved"):
    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
```

Use a rewritten device mod / spec fixture so one kernel advertises either:

```python
{"layout": "sharded", "common_runtime_arg_count": 4}
```

or:

```python
segment["common_runtime_args"] = [{"name": "rank", "kind": "accessor_common_u32", "dtype": "uint32"}]
```

- [ ] **Step 2: Run the fail-fast test to verify it fails for the wrong reason**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_direct_runtime_rejects_common_runtime_accessor_schema -q
```

Expected: FAIL because runtime currently ignores the richer schema.

- [ ] **Step 3: Add explicit runtime guards in `BuildKernelCompileTimeArgs`**

```cpp
ICHECK_EQ(accessor.layout, "interleaved")
    << "Blackhole direct runtime currently supports only interleaved accessors";
ICHECK_EQ(accessor.common_runtime_arg_count, 0U)
    << "Blackhole direct runtime does not yet support accessor common runtime args";
ICHECK(kernel.common_runtime_args.empty())
    << "Blackhole direct runtime does not yet materialize kernel common runtime args";
```

- [ ] **Step 4: Keep codegen tied to compile-time offsets only**

```cpp
const auto* accessor_slot = op->args[4].as<tvm::tir::IntImmNode>();
ICHECK(accessor_slot);
os << "TensorAccessorArgs<" << accessor_slot->value << ">()";
```

- [ ] **Step 5: Run full targeted verification**

Run:

```bash
cmake --build tilelang_repo/build -j32
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:
- build exits `0`
- copy suite `.....x..............`
- gemm suite `.....ss`

- [ ] **Step 6: Sync docs and commit**

```bash
git add tasks/progress.md tasks/dev_design/stage2h_accessor_schema.md tasks/dev_design/stage2d_ttmetal_contract_audit.md tilelang_repo/src/target/blackhole_module.cc tilelang_repo/src/target/codegen_blackhole.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: reject unsupported accessor runtime schema"
```

## Self-Review

- Spec coverage: Task 1 covers `KernelSpec/ExecutableSpec` schema expansion, Task 2 covers lowering/schema emission, Task 3 covers runtime consumption and fail-fast behavior. No spec requirement is left without an owning task.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; all tasks include concrete file paths, commands, and expected outcomes.
- Type consistency: plan consistently uses `AccessorSpec`, `common_runtime_args`, `compile_time_arg_offset`, and `common_runtime_arg_count`.
