# Flash-Attn TT-Metal-First Compute Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Blackhole GEMM and flash-attn compute path around explicit TT-Metal-aligned `transport / tile_scratch / stats_state` semantics, removing the legacy linear fragment compute path while preserving already-brought-up copy/GEMM/flash-attn cases on the new mainline.

**Architecture:** First extend the IR type system with `blackhole.stats` and add a split-after compute-state analysis pass so state kind becomes explicit instead of inferred from late TIR shape. Then canonicalize compute regions, split transport lowering from compute lowering, upgrade planner/spec/codegen/runtime to consume class-aware CB metadata, and finally tighten tests so the new path is the only supported path. Keep `ExecutableSpec -> BlackholeModule` as the only execution path and add no attention-specific schema.

**Tech Stack:** TileLang TIR passes, TVM runtime storage scopes, Blackhole builtins/codegen/runtime, TT-Metal direct runtime, pytest, CMake, git

---

## Current Constraints

- Current worktree is dirty. Execute this plan in an isolated worktree or commit only the files listed per task.
- The current authoritative spec is `tasks/dev_design/2026-04-01-flash-attn-ttmetal-first-compute-redesign.md`.
- Blackhole direct runtime remains the only execution path; do not reintroduce legacy runner behavior.
- `tilelang_repo/CMakeLists.txt` enumerates Blackhole transform sources explicitly, so any new `src/transform/*.cc` file must also be added there.

## File Map

- Modify: `tilelang_repo/3rdparty/tvm/src/runtime/thread_storage_scope.h`
  - Add `kBlackholeStats` storage rank and parse/print support for `blackhole.stats`.
- Modify: `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
  - Canonicalize stats buffers into `blackhole.stats`.
- Modify: `tilelang_repo/src/transform/split_host_device.cc`
  - Treat `blackhole.stats` as a device-side storage scope.
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
  - Preserve `blackhole.stats` in device-kernel extraction.
- Modify: `tilelang_repo/src/transform/lower_device_storage_access_info.cc`
  - Skip storage access rewriting for `kBlackholeStats` the same way it already skips Blackhole CB/acc resources.
- Create: `tilelang_repo/src/transform/analyze_blackhole_compute_state.cc`
  - Infer split-after compute state kinds and store `blackhole.compute_states`.
- Create: `tilelang_repo/src/transform/canonicalize_blackhole_compute_regions.cc`
  - Mark or rewrite compute regions into canonical `matmul_output_tile`, `stats_reduce_row`, `stats_update`, `stats_apply_to_tile`, and `tile_publish` forms.
- Create: `tilelang_repo/src/transform/lower_blackhole_transport_ops.h`
  - Shared declarations for transport-only lowering.
- Create: `tilelang_repo/src/transform/lower_blackhole_transport_ops.cc`
  - Extract the DRAM/CB/CB transport path out of `LowerBlackholeOps`.
- Create: `tilelang_repo/src/transform/lower_blackhole_compute_ops.h`
  - Shared declarations for compute-only lowering.
- Create: `tilelang_repo/src/transform/lower_blackhole_compute_ops.cc`
  - Lower canonical compute regions into tile/stats builtins and emit class-aware CB requirements.
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
  - Shrink to a compatibility wrapper and shared helpers only.
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  - Replace monolith behavior with a wrapper pass that applies transport lowering then compute lowering.
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.h`
  - Add `resource_class` to CB requirements/configs.
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
  - Allocate `transport`, `tile_scratch`, and `stats_scratch` separately and rewrite new builtin CB arguments.
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
  - Declare new tile/stats builtins.
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
  - Register new builtins with correct effect kinds.
- Modify: `tilelang_repo/src/target/blackhole_module.h`
  - Add `resource_class` to runtime-facing `CBConfig`.
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - Extract/encode `resource_class` and any new compile-time arg specs.
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
  - Stop emitting linear `reinterpret_cast<float*>`/`half*` paths for `blackhole.acc` and `blackhole.stats`; emit only explicit tile/stats builtins.
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
  - Validate class-aware CB config and keep runtime materialization algorithm-agnostic.
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
  - Expose `AnalyzeBlackholeComputeState`, `CanonicalizeBlackholeComputeRegions`, `LowerBlackholeTransportOps`, and `LowerBlackholeComputeOps`.
- Modify: `tilelang_repo/tilelang/engine/lower.py`
  - Wire the new passes into the Blackhole lowering pipeline in the designed order.
- Modify: `tilelang_repo/CMakeLists.txt`
  - Add the new transform source files to the Blackhole source list.
- Create: `tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py`
  - Verify state kind inference and `blackhole.stats` exposure.
- Create: `tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py`
  - Verify canonical compute region attrs on flash-attn/GEMM.
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`
  - Check new compute-state and compute-region attrs.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  - Assert multi-tile output lowering and class-aware CB configs.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  - Assert no legacy linear fragment path remains and that stats/tile builtins drive flash-attn lowering.
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  - Retain direct runtime correctness gates against the new path.
- Modify: `tasks/dev_design/stage4_flash_attention_forward_subset.md`
  - Sync design status after implementation.
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
  - Sync overall architecture status after implementation.
- Modify: `tasks/progress.md`
  - Update stage status and next steps.
- Modify: `memory/general_dev.md`
  - Capture reusable lessons from state typing / canonicalization / class-aware planner work.

### Task 1: Add `blackhole.stats` Storage Rank And Compute-State Analysis

**Files:**
- Modify: `tilelang_repo/3rdparty/tvm/src/runtime/thread_storage_scope.h`
- Modify: `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- Modify: `tilelang_repo/src/transform/split_host_device.cc`
- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/transform/lower_device_storage_access_info.cc`
- Create: `tilelang_repo/src/transform/analyze_blackhole_compute_state.cc`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/CMakeLists.txt`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py`

- [ ] **Step 1: Write the failing compute-state analysis test**

```python
import sys
from pathlib import Path

import tvm
from tvm.target import Target

import tilelang


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example


def _device_primfunc():
    target = Target("blackhole")
    with target:
        mod = tilelang.lower(
            mha_example.flashattn.jit_impl.get_tir(
                1, 32, 128, 128, False, block_M=128, block_N=128, num_stages=1, threads=128
            ),
            target=target,
        )
    return mod["main"]


def test_blackhole_compute_state_analysis_exposes_tile_and_stats_kinds():
    func = _device_primfunc()
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.AnalyzeBlackholeComputeState()(mod)
    analyzed = mod["main"]
    states = analyzed.attrs["blackhole.compute_states"]

    by_name = {str(item["name"]): str(item["kind"]) for item in states}
    assert by_name["acc_s"] == "tile_scratch"
    assert by_name["acc_o"] == "tile_scratch"
    assert by_name["scores_max"] == "stats_state"
    assert by_name["logsum"] == "stats_state"
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py -k tile_and_stats
```

Expected:

```text
FAIL ... AttributeError or KeyError for AnalyzeBlackholeComputeState / blackhole.compute_states
```

- [ ] **Step 3: Extend the runtime storage scope type system with `blackhole.stats`**

```cpp
// tilelang_repo/3rdparty/tvm/src/runtime/thread_storage_scope.h
enum class StorageRank {
  kGlobal = 0,
  kShared = 1,
  kWarp = 2,
  kLocal = 3,
  kBlackholeCB = 13,
  kBlackholeAccumulator = 14,
  kBlackholeStats = 15,
};

inline std::string StorageScope::to_string() const {
  switch (rank) {
    case StorageRank::kBlackholeStats:
      return "blackhole.stats" + tag;
    default:
      break;
  }
  /* keep existing cases unchanged */
}

static StorageScope StorageScope::Create(const std::string& s) {
  StorageScope r;
  if (s.compare(0, 15, "blackhole.stats") == 0) {
    r.rank = StorageRank::kBlackholeStats;
    r.tag = s.substr(15, std::string::npos);
  } else if (s.compare(0, 13, "blackhole.acc") == 0) {
    r.rank = StorageRank::kBlackholeAccumulator;
    r.tag = s.substr(13, std::string::npos);
  } else if (s.compare(0, 12, "blackhole.cb") == 0) {
    r.rank = StorageRank::kBlackholeCB;
    r.tag = s.substr(12, std::string::npos);
  } else {
    /* keep existing parser branches unchanged */
  }
  return r;
}
```

- [ ] **Step 4: Canonicalize stats buffers and add the analysis pass**

```cpp
// tilelang_repo/src/transform/analyze_blackhole_compute_state.cc
namespace tvm::tl {

namespace {

String InferComputeStateKind(const tir::Buffer& buffer) {
  const std::string scope = GetPtrStorageScope(buffer->data);
  if (scope == "blackhole.acc") return String("tile_scratch");
  if (scope == "blackhole.stats") return String("stats_state");
  auto parsed = runtime::StorageScope::Create(scope);
  if (parsed.rank == runtime::StorageRank::kBlackholeCB) return String("transport_cb");
  return String("plain_local");
}

}  // namespace

tir::transform::Pass AnalyzeBlackholeComputeStatePass() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) {
    Array<ffi::Map<ffi::String, ffi::Any>> encoded;
    std::unordered_set<const tir::VarNode*> seen;
    tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
      if (const auto* decl = node.as<tir::DeclBufferNode>()) {
        if (!seen.insert(decl->buffer->data.get()).second) return;
        ffi::Map<ffi::String, ffi::Any> entry;
        entry.Set("name", decl->buffer->name);
        entry.Set("scope", GetPtrStorageScope(decl->buffer->data));
        entry.Set("kind", InferComputeStateKind(decl->buffer));
        encoded.push_back(entry);
      }
    });
    auto attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.compute_states", encoded);
    func.CopyOnWrite()->attrs = DictAttrs(attrs);
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AnalyzeBlackholeComputeState", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tl.transform.AnalyzeBlackholeComputeState", AnalyzeBlackholeComputeStatePass);
}

}  // namespace tvm::tl
```

- [ ] **Step 5: Teach canonicalization/split passes about the new storage scope**

```cpp
// tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc
resource_map[name] = {"blackhole.stats", "stats", "stats"};

// tilelang_repo/src/transform/split_host_device.cc
return scope.rank == StorageRank::kBlackholeCB ||
       scope.rank == StorageRank::kBlackholeAccumulator ||
       scope.rank == StorageRank::kBlackholeStats;

// tilelang_repo/src/transform/lower_device_storage_access_info.cc
if (scope.rank == StorageRank::kBlackholeCB ||
    scope.rank == StorageRank::kBlackholeAccumulator ||
    scope.rank == StorageRank::kBlackholeStats) {
  return StmtExprMutator::VisitStmt_(op);
}
```

- [ ] **Step 6: Run the new transform test and make sure it passes**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py -k tile_and_stats
```

Expected:

```text
1 passed
```

- [ ] **Step 7: Commit the state-typing slice**

```bash
git add \
  tilelang_repo/3rdparty/tvm/src/runtime/thread_storage_scope.h \
  tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc \
  tilelang_repo/src/transform/split_host_device.cc \
  tilelang_repo/src/transform/split_blackhole_kernel.cc \
  tilelang_repo/src/transform/lower_device_storage_access_info.cc \
  tilelang_repo/src/transform/analyze_blackhole_compute_state.cc \
  tilelang_repo/tilelang/transform/__init__.py \
  tilelang_repo/CMakeLists.txt \
  tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py
git commit -m "blackhole: add stats storage rank and compute-state analysis"
```

### Task 2: Canonicalize Compute Regions And Wire The New Analysis Pipeline

**Files:**
- Create: `tilelang_repo/src/transform/canonicalize_blackhole_compute_regions.cc`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/CMakeLists.txt`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Write the failing canonicalization test**

```python
import sys
from pathlib import Path

import tvm
from tvm.target import Target

import tilelang


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example


def test_blackhole_compute_regions_mark_flash_attn_stats_and_tile_regions():
    target = Target("blackhole")
    with target:
        mod = tilelang.lower(
            mha_example.flashattn.jit_impl.get_tir(
                1, 32, 128, 128, False, block_M=128, block_N=128, num_stages=1, threads=128
            ),
            target=target,
        )
    mod = tilelang.transform.AnalyzeBlackholeComputeState()(mod)
    mod = tilelang.transform.CanonicalizeBlackholeComputeRegions()(mod)
    func = mod["main"]
    regions = func.attrs["blackhole.compute_regions"]
    kinds = {str(item["kind"]) for item in regions}
    assert "matmul_output_tile" in kinds
    assert "stats_reduce_row" in kinds
    assert "stats_apply_to_tile" in kinds
```

- [ ] **Step 2: Run the canonicalization test and verify it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py -k flash_attn_stats_and_tile_regions
```

Expected:

```text
FAIL ... missing tl.transform.CanonicalizeBlackholeComputeRegions or blackhole.compute_regions
```

- [ ] **Step 3: Add the canonicalization pass and stable region kinds**

```cpp
// tilelang_repo/src/transform/canonicalize_blackhole_compute_regions.cc
namespace tvm::tl {

tir::transform::Pass CanonicalizeBlackholeComputeRegionsPass() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) {
    Array<ffi::Map<ffi::String, ffi::Any>> regions;
    auto record_region = [&](String name, String kind, String output_kind) {
      ffi::Map<ffi::String, ffi::Any> entry;
      entry.Set("name", name);
      entry.Set("kind", kind);
      entry.Set("output_kind", output_kind);
      regions.push_back(entry);
    };

    tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
      if (const auto* call = node.as<tir::CallNode>()) {
        if (call->op.same_as(tir::builtin::call_extern()) && call->args.size() > 0) {
          return;
        }
        if (IsMatmulCall(call)) {
          record_region("gemm_output", "matmul_output_tile", "tile_scratch");
        }
      }
      if (const auto* store = node.as<tir::BufferStoreNode>()) {
        const std::string scope = GetPtrStorageScope(store->buffer->data);
        if (scope == "blackhole.stats") {
          record_region(store->buffer->name, "stats_update", "stats_state");
        }
      }
    });

    auto attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.compute_regions", regions);
    func.CopyOnWrite()->attrs = DictAttrs(attrs);
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.CanonicalizeBlackholeComputeRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def(
      "tl.transform.CanonicalizeBlackholeComputeRegions",
      CanonicalizeBlackholeComputeRegionsPass);
}

}  // namespace tvm::tl
```

- [ ] **Step 4: Wire the pass into the Python transform API and Blackhole pipeline**

```python
# tilelang_repo/tilelang/transform/__init__.py
def AnalyzeBlackholeComputeState():
    return _ffi_api.AnalyzeBlackholeComputeState()


def CanonicalizeBlackholeComputeRegions():
    return _ffi_api.CanonicalizeBlackholeComputeRegions()

# tilelang_repo/tilelang/engine/lower.py
device_mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(device_mod)
device_mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(device_mod)
device_mod = tilelang.transform.AnalyzeBlackholePipelineStages()(device_mod)
device_mod = tilelang.transform.AnalyzeBlackholeComputeState()(device_mod)
device_mod = tilelang.transform.CanonicalizeBlackholeComputeRegions()(device_mod)
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)
device_mod = tilelang.transform.PlanBlackholeCB()(device_mod)
```

- [ ] **Step 5: Run transform tests and make sure canonicalization is visible**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py \
  tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k "compute_regions or flash"
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit the canonicalization slice**

```bash
git add \
  tilelang_repo/src/transform/canonicalize_blackhole_compute_regions.cc \
  tilelang_repo/tilelang/transform/__init__.py \
  tilelang_repo/tilelang/engine/lower.py \
  tilelang_repo/CMakeLists.txt \
  tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py \
  tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: canonicalize compute regions before lowering"
```

### Task 3: Split Transport Lowering From Compute Lowering And Add Tile/Stats Builtins

**Files:**
- Create: `tilelang_repo/src/transform/lower_blackhole_transport_ops.h`
- Create: `tilelang_repo/src/transform/lower_blackhole_transport_ops.cc`
- Create: `tilelang_repo/src/transform/lower_blackhole_compute_ops.h`
- Create: `tilelang_repo/src/transform/lower_blackhole_compute_ops.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/CMakeLists.txt`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Add a failing GEMM regression for multi-tile output lowering**

```python
def test_blackhole_gemm_multi_tile_output_emits_one_pack_per_output_tile():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact = lower(
        gemm_jit_impl.get_tir(M=128, N=128, K=128, trans_A=False, trans_B=False),
        target=Target("blackhole"),
    )
    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    source = str(compute["source_code"])
    cb_ids = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    assert source.count(f"pack_tile(0, {cb_ids['C_shared']}") == 16
    assert source.count(f"cb_push_back({cb_ids['C_shared']}, 1);") == 16
```

- [ ] **Step 2: Add a failing flash-attn regression that rejects the legacy linear fragment path**

```python
def test_flash_attention_compute_source_uses_stats_and_tile_builtins_not_linear_fragment_helpers():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact = lower(
        mha_example.flashattn.jit_impl.get_tir(
            1, 32, 128, 128, False, block_M=128, block_N=128, num_stages=1, threads=128
        ),
        target=Target("blackhole"),
    )
    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    source = str(compute["source_code"])

    assert "tilelang_reduce_row_" not in source
    assert "reinterpret_cast<float*>(acc_s)" not in source
    assert "reinterpret_cast<float*>(scores_max)" not in source
    assert "tl.blackhole.stats_reduce_row" in str(artifact.mod["main"])
    assert "tl.blackhole.tile_apply_stats_bcast" in str(artifact.mod["main"])
```

- [ ] **Step 3: Run the new lowering regressions and confirm they fail**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k multi_tile_output_emits_one_pack_per_output_tile \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k stats_and_tile_builtins_not_linear_fragment_helpers
```

Expected:

```text
2 failed
```

- [ ] **Step 4: Register explicit tile/stats builtins**

```cpp
// tilelang_repo/src/tir/builtin_blackhole.h
TVM_DLL const Op& blackhole_stats_reduce_row();
TVM_DLL const Op& blackhole_stats_binary();
TVM_DLL const Op& blackhole_stats_exp_affine();
TVM_DLL const Op& blackhole_tile_apply_stats_bcast();
TVM_DLL const Op& blackhole_tile_cast_to_cb();

// tilelang_repo/src/tir/builtin_blackhole.cc
TIR_DEFINE_BUILTIN(stats_reduce_row)
TIR_DEFINE_BUILTIN(stats_binary)
TIR_DEFINE_BUILTIN(stats_exp_affine)
TIR_DEFINE_BUILTIN(tile_apply_stats_bcast)
TIR_DEFINE_BUILTIN(tile_cast_to_cb)

TVM_REGISTER_OP("tl.blackhole.stats_reduce_row")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.blackhole.stats_binary")
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.blackhole.stats_exp_affine")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.blackhole.tile_apply_stats_bcast")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.blackhole.tile_cast_to_cb")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

- [ ] **Step 5: Split the monolith and teach compute lowering to emit one pack per local output tile**

```cpp
// tilelang_repo/src/transform/lower_blackhole_compute_ops.cc
for (int mt = 0; mt < local_mt_; ++mt) {
  for (int nt = 0; nt < local_nt_; ++nt) {
    const int dst_index = mt * local_nt_ + nt;
    stmts.push_back(MakeBlackholeCall(blackhole_mm_init(), {
        IntImm32(in0_id), IntImm32(in1_id), IntImm32(out_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    for (int kt = 0; kt < local_kt_; ++kt) {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(in0_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(in1_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_matmul_tiles(),
          {IntImm32(in0_id), IntImm32(in1_id), IntImm32(0), IntImm32(0), IntImm32(dst_index)}));
      stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(in0_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(in1_id), IntImm32(1)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(dst_index), IntImm32(out_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
}
```

- [ ] **Step 6: Lower canonical stats regions to the new builtin family and keep `LowerBlackholeOps` only as a wrapper**

```cpp
// tilelang_repo/src/transform/lower_blackhole_ops.cc
tir::transform::Pass LowerBlackholeOpsPass() {
  auto transport = LowerBlackholeTransportOpsPass();
  auto compute = LowerBlackholeComputeOpsPass();
  auto fpass = [transport, compute](tir::PrimFunc func, IRModule m, tir::transform::PassContext ctx) {
    IRModule one({{GlobalVar("main"), func}});
    one = transport(one);
    one = compute(one);
    return Downcast<tir::PrimFunc>(one->Lookup("main"));
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.LowerBlackholeOps", {});
}
```

- [ ] **Step 7: Run GEMM and flash-attn pipeline regressions and make sure they pass**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "multi_tile_output_emits_one_pack_per_output_tile or accumulator_scope_canonicalized" \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "stats_and_tile_builtins_not_linear_fragment_helpers or does_not_collapse_multi_tile_gemm_outputs_to_one_pack"
```

Expected:

```text
4 passed
```

- [ ] **Step 8: Commit the split-lowering slice**

```bash
git add \
  tilelang_repo/src/transform/lower_blackhole_transport_ops.h \
  tilelang_repo/src/transform/lower_blackhole_transport_ops.cc \
  tilelang_repo/src/transform/lower_blackhole_compute_ops.h \
  tilelang_repo/src/transform/lower_blackhole_compute_ops.cc \
  tilelang_repo/src/transform/lower_blackhole_ops.h \
  tilelang_repo/src/transform/lower_blackhole_ops.cc \
  tilelang_repo/src/tir/builtin_blackhole.h \
  tilelang_repo/src/tir/builtin_blackhole.cc \
  tilelang_repo/tilelang/transform/__init__.py \
  tilelang_repo/tilelang/engine/lower.py \
  tilelang_repo/CMakeLists.txt \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "blackhole: split transport and compute lowering"
```

### Task 4: Upgrade Planner, ExecutableSpec, Codegen, And Direct Runtime To Resource Classes

**Files:**
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.h`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.h`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`

- [ ] **Step 1: Add a failing planner/spec regression for CB resource classes**

```python
def test_blackhole_flash_attention_cb_configs_expose_resource_classes():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact = lower(
        mha_example.flashattn.jit_impl.get_tir(
            1, 32, 128, 128, False, block_M=128, block_N=128, num_stages=1, threads=128
        ),
        target=Target("blackhole"),
    )
    spec = _extract_blackhole_executable_spec(artifact)
    classes = {str(cb["name"]): str(cb["resource_class"]) for cb in spec["cb_configs"]}
    assert classes["Q_shared"] == "transport"
    assert classes["acc_s"] == "tile_scratch"
    assert classes["scores_max"] == "stats_scratch"
```

- [ ] **Step 2: Add a failing codegen regression proving the legacy linear pointer path is gone**

```python
def test_flash_attention_compute_source_has_no_linear_blackhole_acc_or_stats_pointer_materialization():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact = lower(
        mha_example.flashattn.jit_impl.get_tir(
            1, 32, 128, 128, False, block_M=128, block_N=128, num_stages=1, threads=128
        ),
        target=Target("blackhole"),
    )
    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    source = str(compute["source_code"])

    assert "tilelang_get_cb_write_ptr_bytes" not in source
    assert "reinterpret_cast<float*>(scores_max)" not in source
    assert "reinterpret_cast<float*>(acc_s)" not in source
```

- [ ] **Step 3: Run the new planner/codegen regressions and verify they fail**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "resource_classes or no_linear_blackhole_acc_or_stats_pointer_materialization"
```

Expected:

```text
2 failed
```

- [ ] **Step 4: Add `resource_class` to planner output and runtime-facing CB config**

```cpp
// tilelang_repo/src/transform/plan_blackhole_cb.h
struct CBRequirement {
  int requirement_index = -1;
  std::string name;
  std::string role;
  std::string resource_class;
  int num_pages = 1;
  int page_size = 2048;
  std::string data_format = "Float16_b";
};

// tilelang_repo/src/target/blackhole_module.h
struct CBConfig {
  uint32_t cb_id;
  std::string name;
  std::string role;
  std::string resource_class;
  uint32_t num_pages;
  uint32_t page_size_bytes;
  std::string data_format;
};
```

- [ ] **Step 5: Extract `resource_class` in `rt_mod_blackhole` and validate class-aware constraints in direct runtime**

```cpp
// tilelang_repo/src/target/rt_mod_blackhole.cc
if (auto resource_class = cb_info.Get("resource_class")) {
  config.resource_class = Downcast<String>(resource_class.value());
}
if (config.resource_class.empty()) {
  config.resource_class = "transport";
}

// tilelang_repo/src/target/blackhole_module.cc
for (const auto& cb : spec.cb_configs) {
  if (cb.resource_class == "stats_scratch") {
    ICHECK_NE(cb.role, "output") << "stats scratch cannot materialize as writer output";
  }
}
```

- [ ] **Step 6: Remove the legacy linear pointer code path from `codegen_blackhole`**

```cpp
// tilelang_repo/src/target/codegen_blackhole.cc
const bool runtime_managed_storage =
    scope == "shared" || scope == "shared.dyn" || scope == "shared.barrier" ||
    scope.rfind("blackhole.cb", 0) == 0 || scope == "blackhole.acc" || scope == "blackhole.stats";

if (runtime_managed_storage) {
  this->PrintStmt(op->body);
  return;
}

if (scope == "blackhole.acc" || scope == "blackhole.stats") {
  ICHECK(false) << "Explicit tile/stats builtins must materialize Blackhole compute state; "
                << "linear pointer emission is unsupported";
}
```

- [ ] **Step 7: Run planner/spec/codegen regressions and gated runtime tests**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "cb_ids_are_rewritten_by_planner or compile_time_abi" \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "resource_classes or no_linear_blackhole_acc_or_stats_pointer_materialization" \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
```

Expected:

```text
planner/spec tests pass; flash-attn runtime is either PASS on configured TT-Metal env or SKIPPED with explicit environment gate
```

- [ ] **Step 8: Commit the planner/spec/codegen/runtime slice**

```bash
git add \
  tilelang_repo/src/transform/plan_blackhole_cb.h \
  tilelang_repo/src/transform/plan_blackhole_cb.cc \
  tilelang_repo/src/target/blackhole_module.h \
  tilelang_repo/src/target/rt_mod_blackhole.cc \
  tilelang_repo/src/target/codegen_blackhole.cc \
  tilelang_repo/src/target/blackhole_module.cc \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
git commit -m "blackhole: add class-aware compute planner and runtime schema"
```

### Task 5: Full Verification, Documentation Sync, And Knowledge Capture

**Files:**
- Modify: `tasks/dev_design/stage4_flash_attention_forward_subset.md`
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`

- [ ] **Step 1: Run the transform and codegen regression suites**

Run:

```bash
pytest -q \
  tilelang_repo/testing/python/transform/test_blackhole_compute_state_analysis.py \
  tilelang_repo/testing/python/transform/test_blackhole_compute_region_canonicalization.py \
  tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
```

Expected:

```text
All targeted transform/pipeline/GEMM tests pass
```

- [ ] **Step 2: Run direct runtime smoke tests with explicit environment reporting**

Run:

```bash
pytest -q -rs \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py \
  tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
```

Expected:

```text
Configured TT-Metal env: PASS
Unconfigured env: SKIPPED with explicit reason such as TT_METAL_RUNTIME_ROOT not set
```

- [ ] **Step 3: Update the active design and progress docs to reflect the new mainline**

```markdown
<!-- tasks/dev_design/stage4_flash_attention_forward_subset.md -->
- `blackhole.acc` 已收正为 tile scratch only
- `blackhole.stats` 已成为 statistics state 的正式语义
- flash-attn compute 主链已不再依赖 legacy linear fragment helper

<!-- tasks/dev_design/final_blackhole_backend_redesign.md -->
- split 后 Blackhole compute lowering 已正式分层为 compute-state analysis -> compute region canonicalization -> transport lowering -> compute lowering
- `cb_configs.resource_class` 已进入 spec/runtime 主链

<!-- tasks/progress.md -->
- 当前 blocker 从 compute 语义定义转到更宽 stats/tile coverage 或 runtime correctness residuals（按实际结果填写）
```

- [ ] **Step 4: Capture reusable engineering lessons**

```markdown
<!-- memory/general_dev.md -->
- 当 target compute 模型区分 transport/tile/stats 三类状态时，必须优先扩 storage scope 和 split-after analysis，不要继续让 codegen 从 late TIR 形态猜语义。
- 多 tile matmul lowering 里 local `Mt/Nt/Kt` 必须来自 split-after compute contract，不能让 emitter 默认单 tile。
```

- [ ] **Step 5: Verify git status is clean except for intentional docs updates**

Run:

```bash
git status --short
```

Expected:

```text
Only the documentation files listed in this task remain modified before the final docs commit
```

- [ ] **Step 6: Commit the docs/verification slice**

```bash
git add \
  tasks/dev_design/stage4_flash_attention_forward_subset.md \
  tasks/dev_design/final_blackhole_backend_redesign.md \
  tasks/progress.md \
  memory/general_dev.md
git commit -m "docs: sync blackhole compute redesign status"
```

- [ ] **Step 7: Push the branch and record the verification commands in the PR body or handoff**

Run:

```bash
git push
```

Expected:

```text
Current branch pushed successfully
```

## Self-Review Checklist

- Spec coverage:
  - State typing: Task 1
  - Canonical compute regions: Task 2
  - Split transport/compute lowering and multi-tile matmul output: Task 3
  - Class-aware planner/spec/codegen/runtime: Task 4
  - Verification and docs sync: Task 5
- Placeholder scan:
  - No unfinished markers or “repeat the previous task” shortcuts remain in this plan.
- Type consistency:
  - This plan consistently uses `blackhole.compute_states`, `blackhole.compute_regions`, `blackhole.stats`, and `cb_configs.resource_class` as the canonical names.
