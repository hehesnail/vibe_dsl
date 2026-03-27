# Compile-Time ABI Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize Blackhole kernel compile-time ABI as explicit schema in `KernelSpec`/`ExecutableSpec`, while keeping the current direct runtime execution surface unchanged.

**Architecture:** Extend the existing Blackhole split-after contract instead of creating a parallel ABI path. `LowerBlackholeOps`/segment attrs will emit compile-time ABI descriptors and launch metadata, `rt_mod_blackhole` will extract them into new runtime structs, and `BlackholeModule` will materialize compile args from schema rather than anonymous arrays. Unsupported compile-time ABI kinds will fail fast.

**Tech Stack:** C++, TVM TIR attrs, Blackhole direct runtime, TT-Metal host `CreateKernel`, pytest

---

## File Map

- Modify: `tilelang_repo/src/target/blackhole_module.h`
  Responsibility: add `CompileTimeArgSpec` and `KernelLaunchSpec`, then wire them into `KernelSpec`.
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  Responsibility: extract/encode compile-time ABI schema and launch metadata from TIR attrs into `KernelSpec`/`ExecutableSpec`.
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
  Responsibility: materialize compile args from `compile_time_arg_specs`, consume `launch_spec`, and fail fast for unsupported kinds.
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  Responsibility: emit compile-time ABI descriptors and launch metadata for the current copy/GEMM direct path.
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
  Responsibility: declare lowering helpers/descriptor structs for compile-time ABI schema emission.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  Responsibility: assert copy kernel spec carries compile-time ABI schema and runtime rejects unknown kinds.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  Responsibility: assert GEMM kernel spec carries `gemm_shape` / `gemm_transpose_flags`.
- Modify: `tasks/dev_design/stage2i_compile_time_abi_schema.md`
  Responsibility: flip status from design to implemented and sync any implementation constraints.
- Modify: `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  Responsibility: update P3/P0 wording after compile-time ABI schema lands.
- Modify: `tasks/progress.md`
  Responsibility: record current status, regressions, and next steps truthfully.

---

### Task 1: Add Failing Contract Tests For Compile-Time ABI Schema

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add copy pipeline assertions for compile-time ABI schema**

```python
def test_blackhole_copy_compile_time_abi_is_materialized():
    mod = _build_copy_runtime_module()
    spec = _extract_blackhole_spec(mod)
    kernel = spec["kernels"][0]

    compile_specs = list(kernel["compile_time_arg_specs"])
    assert len(compile_specs) == 2
    assert compile_specs[0]["kind"] == "interleaved_accessor_cta"
    assert int(compile_specs[0]["offset"]) == 0
    assert int(compile_specs[0]["count"]) == 2
    assert compile_specs[0]["buffer"] == "input0"
    assert compile_specs[1]["kind"] == "interleaved_accessor_cta"
    assert int(compile_specs[1]["offset"]) == 2
    assert int(compile_specs[1]["count"]) == 2
    assert compile_specs[1]["buffer"] == "output0"

    launch_spec = kernel["launch_spec"]
    assert launch_spec["core_type"] == "brisc"
    assert launch_spec["processor"] == "riscv_0"
    assert launch_spec["noc"] == "riscv_0_default"
```

- [ ] **Step 2: Run the copy schema test and confirm it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_compile_time_abi_is_materialized -q
```

Expected: FAIL because `compile_time_arg_specs` / `launch_spec` do not exist in the extracted kernel spec yet.

- [ ] **Step 3: Add GEMM contract assertions for shape and transpose compile-time ABI**

```python
def test_blackhole_gemm_compile_time_abi_is_materialized():
    mod = _build_blackhole_gemm_runtime_module()
    spec = _extract_blackhole_spec(mod)

    reader = next(kernel for kernel in spec["kernels"] if kernel["kind"] == "reader")
    compute = next(kernel for kernel in spec["kernels"] if kernel["kind"] == "compute")

    reader_specs = list(reader["compile_time_arg_specs"])
    assert [item["kind"] for item in reader_specs[:2]] == [
        "interleaved_accessor_cta",
        "interleaved_accessor_cta",
    ]

    compute_specs = list(compute["compile_time_arg_specs"])
    assert any(item["kind"] == "gemm_shape" for item in compute_specs)
    assert any(item["kind"] == "gemm_transpose_flags" for item in compute_specs)

    gemm_shape = next(item for item in compute_specs if item["kind"] == "gemm_shape")
    assert list(gemm_shape["values"]) == [1, 1, 1]

    gemm_transpose = next(item for item in compute_specs if item["kind"] == "gemm_transpose_flags")
    assert list(gemm_transpose["values"]) == [0, 1]
```

- [ ] **Step 4: Run the GEMM schema test and confirm it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compile_time_abi_is_materialized -q
```

Expected: FAIL because compute kernels do not expose any compile-time ABI schema yet.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "test: add compile-time abi schema coverage"
```

---

### Task 2: Extend Runtime Structs And Extraction For Compile-Time ABI Schema

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.h`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add new runtime structs in `blackhole_module.h`**

```cpp
struct CompileTimeArgSpec {
  std::string name;
  std::string kind;
  std::string dtype;
  uint32_t offset = 0;
  uint32_t count = 0;
  std::string buffer;
  std::string segment_role;
  std::vector<uint32_t> values;
  std::string layout;
  std::string memory_space;
};

struct KernelLaunchSpec {
  std::string core_type;
  std::string processor;
  std::string noc;
};

struct KernelSpec {
  std::string name;
  std::string kind;
  std::string core_type;
  std::string source_code;
  std::vector<uint32_t> compile_time_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs;
  KernelLaunchSpec launch_spec;
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<AccessorSpec> accessors;
};
```

- [ ] **Step 2: Add extraction helpers in `rt_mod_blackhole.cc`**

```cpp
static std::vector<uint32_t> ExtractU32Values(const ffi::Array<ffi::Any>& items) {
  std::vector<uint32_t> values;
  for (const auto& item : items) {
    values.push_back(static_cast<uint32_t>(Downcast<Integer>(item).IntValue()));
  }
  return values;
}

static std::vector<CompileTimeArgSpec> ExtractCompileTimeArgSpecs(
    const ffi::Array<ffi::Any>& items) {
  std::vector<CompileTimeArgSpec> specs;
  for (const auto& item : items) {
    auto info = item.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (info.empty()) continue;
    CompileTimeArgSpec spec;
    if (auto v = info.Get("name")) spec.name = Downcast<String>(v.value());
    if (auto v = info.Get("kind")) spec.kind = Downcast<String>(v.value());
    if (auto v = info.Get("dtype")) spec.dtype = Downcast<String>(v.value());
    if (auto v = info.Get("offset")) spec.offset = Downcast<Integer>(v.value()).IntValue();
    if (auto v = info.Get("count")) spec.count = Downcast<Integer>(v.value()).IntValue();
    if (auto v = info.Get("buffer")) spec.buffer = Downcast<String>(v.value());
    if (auto v = info.Get("segment_role")) spec.segment_role = Downcast<String>(v.value());
    if (auto v = info.Get("values")) {
      spec.values = ExtractU32Values(Downcast<ffi::Array<ffi::Any>>(v.value()));
    }
    if (auto v = info.Get("layout")) spec.layout = Downcast<String>(v.value());
    if (auto v = info.Get("memory_space")) spec.memory_space = Downcast<String>(v.value());
    specs.push_back(std::move(spec));
  }
  return specs;
}
```

- [ ] **Step 3: Add launch-spec extraction and kernel wiring**

```cpp
static KernelLaunchSpec ExtractLaunchSpec(const ffi::Map<ffi::String, ffi::Any>& segment,
                                          const std::string& fallback_core_type) {
  KernelLaunchSpec launch;
  launch.core_type = fallback_core_type;
  if (auto v = segment.Get("launch_spec")) {
    auto info = Downcast<ffi::Map<ffi::String, ffi::Any>>(v.value());
    if (auto x = info.Get("core_type")) launch.core_type = Downcast<String>(x.value());
    if (auto x = info.Get("processor")) launch.processor = Downcast<String>(x.value());
    if (auto x = info.Get("noc")) launch.noc = Downcast<String>(x.value());
  }
  return launch;
}

// In ExtractSegmentInfos / kernel extraction:
if (auto v = segment.Get("compile_time_arg_specs")) {
  info.compile_time_arg_specs =
      ExtractCompileTimeArgSpecs(Downcast<ffi::Array<ffi::Any>>(v.value()));
}
info.launch_spec = ExtractLaunchSpec(segment, info.core_type);
kernel.compile_time_arg_specs = segment.compile_time_arg_specs;
kernel.launch_spec = segment.launch_spec;
```

- [ ] **Step 4: Add encode helpers so extracted specs survive module serialization**

```cpp
static ffi::Array<ffi::Any> EncodeCompileTimeArgSpecs(
    const std::vector<CompileTimeArgSpec>& specs) {
  ffi::Array<ffi::Any> encoded;
  for (const auto& spec : specs) {
    ffi::Map<ffi::String, ffi::Any> info;
    info.Set("name", ffi::String(spec.name));
    info.Set("kind", ffi::String(spec.kind));
    info.Set("dtype", ffi::String(spec.dtype));
    info.Set("offset", Integer(static_cast<int>(spec.offset)));
    info.Set("count", Integer(static_cast<int>(spec.count)));
    if (!spec.buffer.empty()) info.Set("buffer", ffi::String(spec.buffer));
    if (!spec.segment_role.empty()) info.Set("segment_role", ffi::String(spec.segment_role));
    if (!spec.values.empty()) {
      ffi::Array<ffi::Any> values;
      for (uint32_t v : spec.values) values.push_back(Integer(static_cast<int>(v)));
      info.Set("values", values);
    }
    if (!spec.layout.empty()) info.Set("layout", ffi::String(spec.layout));
    if (!spec.memory_space.empty()) info.Set("memory_space", ffi::String(spec.memory_space));
    encoded.push_back(info);
  }
  return encoded;
}
```

- [ ] **Step 5: Run the two schema tests and make sure they now reach lowering gaps instead of extraction gaps**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_compile_time_abi_is_materialized -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compile_time_abi_is_materialized -q
```

Expected: still FAIL, but now because lowering does not emit `compile_time_arg_specs` / `launch_spec` into segment attrs yet.

- [ ] **Step 6: Commit the extraction layer**

```bash
git add tilelang_repo/src/target/blackhole_module.h tilelang_repo/src/target/rt_mod_blackhole.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: extract compile-time abi schema"
```

---

### Task 3: Emit Compile-Time ABI Schema In Lowering

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add lowering-side descriptor structs and helpers**

```cpp
struct CompileTimeArgDescriptor {
  std::string name;
  std::string kind;
  std::string dtype;
  int offset = 0;
  int count = 0;
  std::string buffer;
  std::string segment_role;
  std::vector<int> values;
  std::string layout;
  std::string memory_space;
};

struct LaunchDescriptor {
  std::string core_type;
  std::string processor;
  std::string noc;
};

ffi::Array<ffi::Any> EncodeCompileTimeArgDescriptors(
    const std::vector<CompileTimeArgDescriptor>& descriptors) const;
ffi::Map<ffi::String, ffi::Any> EncodeLaunchDescriptor(const LaunchDescriptor& launch) const;
```

- [ ] **Step 2: Emit copy accessor compile-time ABI descriptors**

```cpp
std::vector<CompileTimeArgDescriptor> descriptors;
descriptors.push_back({"input0_accessor", "interleaved_accessor_cta", "uint32", 0, 2,
                       source_buffer_name, "fused_dataflow", {}, "interleaved", "dram"});
descriptors.push_back({"output0_accessor", "interleaved_accessor_cta", "uint32", 2, 2,
                       dest_buffer_name, "fused_dataflow", {}, "interleaved", "dram"});
segment.Set("compile_time_arg_specs", EncodeCompileTimeArgDescriptors(descriptors));
segment.Set("launch_spec", EncodeLaunchDescriptor({"brisc", "riscv_0", "riscv_0_default"}));
```

- [ ] **Step 3: Emit GEMM reader/writer accessor descriptors and compute shape descriptors**

```cpp
reader_segment.Set("compile_time_arg_specs", EncodeCompileTimeArgDescriptors({
    {"a_accessor", "interleaved_accessor_cta", "uint32", 0, 2, a_buffer_name, "reader", {},
     "interleaved", "dram"},
    {"b_accessor", "interleaved_accessor_cta", "uint32", 2, 2, b_buffer_name, "reader", {},
     "interleaved", "dram"},
}));
reader_segment.Set("launch_spec", EncodeLaunchDescriptor({"brisc", "riscv_0", "riscv_0_default"}));

compute_segment.Set("compile_time_arg_specs", EncodeCompileTimeArgDescriptors({
    {"gemm_shape", "gemm_shape", "uint32", 0, 3, "", "compute", {Mt, Kt, Nt}, "", ""},
    {"gemm_transpose", "gemm_transpose_flags", "uint32", 3, 2, "", "compute",
     {transpose_a ? 1 : 0, transpose_b ? 1 : 0}, "", ""},
}));
compute_segment.Set("launch_spec", EncodeLaunchDescriptor({"trisc", "", ""}));

writer_segment.Set("compile_time_arg_specs", EncodeCompileTimeArgDescriptors({
    {"output_accessor", "interleaved_accessor_cta", "uint32", 0, 2, c_buffer_name, "writer", {},
     "interleaved", "dram"},
}));
writer_segment.Set("launch_spec", EncodeLaunchDescriptor({"ncrisc", "riscv_1", "riscv_1_default"}));
```

- [ ] **Step 4: Keep `compile_time_args` consistent as temporary compatibility data**

```cpp
// Keep existing numeric vectors populated for now.
// The schema becomes the formal source; old vectors remain for compatibility paths.
segment.Set("compile_time_args", existing_compile_time_args);
segment.Set("compile_time_arg_specs", EncodeCompileTimeArgDescriptors(descriptors));
```

- [ ] **Step 5: Run the schema tests and confirm they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_compile_time_abi_is_materialized -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compile_time_abi_is_materialized -q
```

Expected: PASS.

- [ ] **Step 6: Commit the lowering emission**

```bash
git add tilelang_repo/src/transform/lower_blackhole_ops.h tilelang_repo/src/transform/lower_blackhole_ops.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: emit compile-time abi schema"
```

---

### Task 4: Switch Direct Runtime To Schema-Driven Compile Arg Materialization

**Files:**
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add a failing test for unknown compile-time ABI kind**

```python
def test_blackhole_direct_runtime_rejects_unknown_compile_time_abi():
    mod = _build_copy_runtime_module()
    spec = _extract_blackhole_spec(mod)
    kernel = dict(spec["kernels"][0])
    kernel["compile_time_arg_specs"] = [
        {"name": "mystery", "kind": "mystery_kind", "dtype": "uint32", "offset": 0, "count": 1, "values": [7]}
    ]
    kernel["compile_time_args"] = []
    spec["kernels"] = [kernel]

    with pytest.raises(tvm.TVMError, match="unsupported compile-time ABI kind"):
        _run_blackhole_spec_direct(spec)
```

- [ ] **Step 2: Run the fail-fast test and confirm it fails for the wrong reason**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_direct_runtime_rejects_unknown_compile_time_abi -q
```

Expected: FAIL because `BlackholeModule` still ignores `compile_time_arg_specs`.

- [ ] **Step 3: Replace `BuildKernelCompileTimeArgs` with schema-driven assembly**

```cpp
static std::vector<uint32_t> BuildKernelCompileTimeArgs(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  if (kernel.compile_time_arg_specs.empty()) {
    return AppendInterleavedAccessors(kernel.compile_time_args, kernel.accessors, buffer_bindings);
  }

  std::vector<CompileTimeArgSpec> specs = kernel.compile_time_arg_specs;
  std::sort(specs.begin(), specs.end(),
            [](const CompileTimeArgSpec& a, const CompileTimeArgSpec& b) { return a.offset < b.offset; });

  std::vector<uint32_t> compile_args;
  for (const auto& spec : specs) {
    if (spec.kind == "literal_u32" || spec.kind == "gemm_shape" ||
        spec.kind == "gemm_transpose_flags") {
      ICHECK_EQ(spec.values.size(), spec.count);
      compile_args.insert(compile_args.end(), spec.values.begin(), spec.values.end());
    } else if (spec.kind == "interleaved_accessor_cta") {
      ICHECK_EQ(spec.count, 2U);
      auto it = buffer_bindings.find(spec.buffer);
      ICHECK(it != buffer_bindings.end()) << "Missing runtime buffer binding for " << spec.buffer;
      TensorAccessorArgs(*(it->second.mesh_buffer)).append_to(compile_args);
    } else {
      ICHECK(false) << "Blackhole direct runtime encountered unsupported compile-time ABI kind: "
                    << spec.kind;
    }
  }
  return compile_args;
}
```

- [ ] **Step 4: Drive `CreateKernel` from `launch_spec` when present**

```cpp
const KernelLaunchSpec& launch = kernel.launch_spec;
const std::string& core_type = launch.core_type.empty() ? kernel.core_type : launch.core_type;

if (core_type == "trisc" || kernel.kind == "compute") {
  return CreateKernel(program, kernel_path, core_spec,
                      ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                    .fp32_dest_acc_en = true,
                                    .math_approx_mode = false,
                                    .compile_args = compile_time_args});
}

DataMovementProcessor processor = launch.processor == "riscv_1" ? DataMovementProcessor::RISCV_1
                                                                : DataMovementProcessor::RISCV_0;
NOC noc = launch.noc == "riscv_1_default" ? NOC::RISCV_1_default : NOC::RISCV_0_default;
```

- [ ] **Step 5: Run targeted tests and confirm they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_direct_runtime_rejects_unknown_compile_time_abi -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_compile_time_abi_is_materialized -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compile_time_abi_is_materialized -q
```

Expected: PASS.

- [ ] **Step 6: Commit the runtime materialization switch**

```bash
git add tilelang_repo/src/target/blackhole_module.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: materialize compile-time abi from schema"
```

---

### Task 5: Full Regression, Docs, And Final Integration

**Files:**
- Modify: `tasks/dev_design/stage2i_compile_time_abi_schema.md`
- Modify: `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
- Modify: `tasks/progress.md`

- [ ] **Step 1: Run build and regression suite**

Run:

```bash
cmake --build tilelang_repo/build -j32
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:

```text
Build succeeds
copy pipeline: all current passing tests remain passing
gemm: all current passing tests remain passing
```

- [ ] **Step 2: Update design docs to reflect implementation**

```markdown
- `stage2i_compile_time_abi_schema.md`
  - 状态改为 `✅ 已实现`
  - 记录当前正式支持的 compile-time ABI kinds
- `stage2d_ttmetal_contract_audit.md`
  - 把“compile-time ABI 也没有正式化”改成“主路径已 formalize 到 accessor CTA + GEMM shape/transpose + launch schema，更多 ABI 仍未做”
```

- [ ] **Step 3: Update progress with truthful P3 status**

```markdown
- 当前状态新增：compile-time ABI schema 已进入 segment plan / KernelSpec / direct runtime
- P3 备注更新为：work schema、accessor/common-runtime schema、compile-time ABI schema 已 formalize；更宽的 execution surface 仍未做
- 最新回归结果写入 fresh numbers
```

- [ ] **Step 4: Confirm no background processes remain**

Run:

```bash
ps -ef | rg "pytest|ctest|cmake --build|ninja|tt-sim|tt-metal|blackhole" -n
```

Expected: only the `rg` process itself or no matching long-running jobs.

- [ ] **Step 5: Commit and push final integration**

```bash
git add tasks/dev_design/stage2i_compile_time_abi_schema.md tasks/dev_design/stage2d_ttmetal_contract_audit.md tasks/progress.md
git add tilelang_repo/src/target/blackhole_module.h tilelang_repo/src/target/rt_mod_blackhole.cc tilelang_repo/src/target/blackhole_module.cc
git add tilelang_repo/src/transform/lower_blackhole_ops.h tilelang_repo/src/transform/lower_blackhole_ops.cc
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: formalize compile-time abi schema"
git push
```

---

## Self-Review

- Spec coverage:
  - `CompileTimeArgSpec` / `KernelLaunchSpec` introduction: Task 2
  - Lowering emission of schema: Task 3
  - `BlackholeModule` schema-driven materialization and fail-fast: Task 4
  - Regression + doc/progress sync: Task 5
  - No spec requirement is left without an owning task.
- Placeholder scan:
  - Removed generic “update tests/docs” wording; each task names files, commands, and expected behavior.
- Type consistency:
  - Plan consistently uses `CompileTimeArgSpec`, `KernelLaunchSpec`, `compile_time_arg_specs`, and `launch_spec`.
  - The runtime fail-fast message consistently refers to “unsupported compile-time ABI kind”.
