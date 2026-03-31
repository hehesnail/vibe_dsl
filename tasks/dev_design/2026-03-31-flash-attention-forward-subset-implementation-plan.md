# Flash-Attention Forward Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Blackhole 主链在不改 example 的前提下，具备承接 `example_mha_fwd_bshd.py` / `example_gqa_fwd_bshd.py` 前向完整语义所需的通用 analysis 与 lowering 骨架。

**Architecture:** 先补 split 前后通用 analysis pass，分别覆盖 work decomposition、fragment compute region、pipelined staging；再扩 `LowerBlackholeOps` 消费这些分析结果并产出最小冻结结论；最后只在 codegen/runtime 侧消费这些冻结结论，不新增 attention 专属旁路或 host-side 算法解释逻辑。

**Tech Stack:** TileLang TIR passes, TVM PrimFunc attrs, Blackhole lowering/codegen/runtime, pytest, CMake

## Current Status

- ✅ Task 1 completed: analysis test夹具已建好
- ✅ Task 2 completed: `AnalyzeBlackholeWorkDecomposition` 已落地并输出结构化 `blackhole.work_decomposition`
- ✅ Task 3 completed: `AnalyzeBlackholeFragmentRegions` 已覆盖 split-after 与优化后 device IR 上的 MHA/GQA `gemm + row_reduction + row_broadcast + pointwise_chain` 形态，并额外产出细粒度 `pointwise_ops`（`fill/exp2/cast/max/add/mul/div/if_then_else` 等）；`test_blackhole_flash_attention_analysis.py` 当前 `7 passed`
- ✅ Task 4 completed: `AnalyzeBlackholePipelineStages` 已落地并进入主链
- 🔄 Current focus: Task 5 / Task 6。`LowerBlackholeOps` 已开始消费 analysis 并产出通用 `blackhole.lowering_requirements` IR attrs；当前显式 fail-fast 已迁移到 `rt_mod_blackhole` 的 build-time gate，且普通 pointwise op 已不再占据主 blocker
- ✅ Latest narrowing: flash-attn forward 的 `row_reduction` 现已在 `LowerBlackholeOps` 中正式 lower 到 `tl.blackhole.reduce_row` builtin，并覆盖 split-after 与 post-`OptimizeForTarget` 两条真实 device IR 形态；full `lower()` 的 build-time gate 因此已收窄到只剩 `row_broadcast`
- ✅ Latest narrowing: `row_broadcast` 的 MHA optimized-path 形态已进一步进入真实 TIR lowering。除 `acc_o *= scores_scale[0]` / `acc_o /= logsum[0] -> tl.blackhole.mul_row_bcast/div_row_bcast`、`logsum = logsum * scores_scale + scores_sum -> tl.blackhole.scalar_fma` 外，`acc_s[i] = exp2(acc_s[i] * scale - scores_max[0] * scale)` 也已 lower 成 `tl.blackhole.exp2_row_bcast_affine`，`scores_scale[0] = exp2(scores_max_prev[0] * scale - scores_max[0] * scale)` 已 lower 成 `tl.blackhole.scalar_exp2_affine`
- 🔄 Next focus: 继续把剩余 pointwise 子集（当前显式 blocker 为 `fill / max / add / cast`）从 TIR 正式 lower 成 Blackhole compute 子集，而不是回头继续加 schema/gate
- 🔄 Task 6 也已开始：第一条 generic fragment-pipeline legality 已落地，当前 `num_stages > 2` 会在主链上显式失败；full `lower()` 的 GQA `num_stages=4` 现在已能稳定命中这条 legality，不再先被 `RegionOp` staged/shared view canonicalization 内部错误打断

---

## File Map

- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  - 消费新 analysis 结果，产出最小冻结结论与 legality
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - 只提取必须冻结进 `ExecutableSpec` 的结果
- Modify: `tilelang_repo/src/target/blackhole_module.h`
  - 只在必须时新增最小 runtime/materialization schema
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
  - 消费新的冻结结论并对未支持面 fail-fast
- Create: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
  - work decomposition analysis pass
- Create: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
  - fragment compute region analysis pass
- Create: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
  - pipeline stage analysis pass
- Modify: `tilelang_repo/src/transform/Makefile/CMakeLists or pass registration files as needed`
  - 接入 pass 管线与注册
- Test: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`
  - analysis / attrs / canonical form 回归
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  - lowering / spec / legality 回归
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  - direct runtime / fail-fast / full-env gate
- Test: `tilelang_repo/examples/flash_attention/test_example_flash_attention.py`
  - 最终 example 回归入口
- Modify: `tasks/dev_design/stage4_flash_attention_forward_subset.md`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

### Task 1: Build Analysis Test Fixtures

**Files:**
- Create: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`
- Reference: `tilelang_repo/examples/flash_attention/example_mha_fwd_bshd.py`
- Reference: `tilelang_repo/examples/flash_attention/example_gqa_fwd_bshd.py`

- [ ] **Step 1: Write the failing analysis tests**

```python
import tilelang
import tilelang.language as T
from tilelang.examples.flash_attention import example_mha_fwd_bshd, example_gqa_fwd_bshd


def _lower_to_blackhole_ir(program):
    mod = tilelang.lower(program, target="blackhole")
    return mod


def test_mha_forward_exposes_work_decomposition_attrs():
    mod = _lower_to_blackhole_ir(
        example_mha_fwd_bshd.flashattn(
            batch=1, heads=32, seq_len=256, dim=128, is_causal=False,
            block_M=128, block_N=128, num_stages=1, threads=128
        )
    )
    body = str(mod["main"])
    assert "blackhole.work_decomposition" in body


def test_gqa_forward_exposes_fragment_region_attrs():
    mod = _lower_to_blackhole_ir(
        example_gqa_fwd_bshd.flashattn(
            batch=1, heads=16, seq_len=1024, dim=128, is_causal=False,
            groups=16, block_M=64, block_N=64, num_stages=2, threads=128
        )
    )
    body = str(mod["main"])
    assert "blackhole.fragment_regions" in body


def test_forward_pipeline_exposes_stage_attrs():
    mod = _lower_to_blackhole_ir(
        example_mha_fwd_bshd.flashattn(
            batch=1, heads=32, seq_len=256, dim=128, is_causal=True,
            block_M=128, block_N=128, num_stages=1, threads=128
        )
    )
    body = str(mod["main"])
    assert "blackhole.pipeline_stages" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
```

Expected:

```text
FAIL ... missing blackhole.work_decomposition / fragment_regions / pipeline_stages attrs
```

- [ ] **Step 3: Commit**

```bash
git add tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "test: add flash attention analysis coverage"
```

### Task 2: Add Work Decomposition Analysis

**Files:**
- Create: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Test: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Implement failing-pass registration skeleton**

```cpp
namespace tvm {
namespace tl {

tir::transform::Pass AnalyzeBlackholeWorkDecompositionPass() {
  auto fpass = [](tir::PrimFunc func, IRModule m, tir::transform::PassContext ctx) {
    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    ICHECK(false) << "AnalyzeBlackholeWorkDecompositionPass not implemented";
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AnalyzeBlackholeWorkDecomposition", {});
}

}  // namespace tl
}  // namespace tvm
```

- [ ] **Step 2: Run one targeted test to verify the new failure point**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k work_decomposition
```

Expected:

```text
FAIL ... AnalyzeBlackholeWorkDecompositionPass not implemented
```

- [ ] **Step 3: Implement minimal analysis**

```cpp
Map<String, Any> work_info;
work_info.Set("axes", Array<String>{String("bx"), String("by"), String("bz")});
work_info.Set("derived_index_exprs", derived_exprs);
work_info.Set("work_dependent_loop_bounds", loop_bounds);
attrs.Set("blackhole.work_decomposition", work_info);
func.CopyOnWrite()->attrs = DictAttrs(attrs);
```

- [ ] **Step 4: Wire the pass before `LowerBlackholeOps`**

```cpp
passes.push_back(tl::transform::AnalyzeBlackholeWorkDecomposition());
passes.push_back(tl::transform::LowerBlackholeOps());
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k work_decomposition
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```bash
git add tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: add work decomposition analysis"
```

### Task 3: Add Fragment Region Analysis ✅

**Files:**
- Create: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Test: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [x] **Step 1: Write a region-shape expectation into the test**

```python
def test_fragment_regions_capture_reduction_and_broadcast_roles():
    mod = _lower_to_blackhole_ir(
        example_mha_fwd_bshd.flashattn(
            batch=1, heads=32, seq_len=256, dim=128, is_causal=False,
            block_M=128, block_N=128, num_stages=1, threads=128
        )
    )
    body = str(mod["main"])
    assert "row_reduction" in body
    assert "row_broadcast" in body
```

- [x] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k fragment_regions
```

Expected:

```text
FAIL ... row_reduction / row_broadcast not found
```

- [x] **Step 3: Implement fragment region analysis skeleton**

```cpp
Map<String, Any> region;
region.Set("fragment_buffers", fragment_buffers);
region.Set("ops", Array<String>{String("gemm"), String("row_reduction"), String("row_broadcast")});
region.Set("loop_carried_state", loop_carried_state);
attrs.Set("blackhole.fragment_regions", Array<Any>{region});
func.CopyOnWrite()->attrs = DictAttrs(attrs);
```

- [x] **Step 4: Run targeted tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k fragment_regions
```

Expected:

```text
1 passed
```

Latest result:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: add fragment region analysis"
```

### Task 4: Add Pipeline Stage Analysis ✅

**Files:**
- Create: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Test: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Extend tests with stage assertions**

```python
def test_pipeline_stages_capture_num_stages_and_loop_carried_state():
    mod = _lower_to_blackhole_ir(
        example_gqa_fwd_bshd.flashattn(
            batch=1, heads=16, seq_len=1024, dim=128, is_causal=False,
            groups=16, block_M=64, block_N=64, num_stages=2, threads=128
        )
    )
    body = str(mod["main"])
    assert "num_stages" in body
    assert "loop_carried_state" in body
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k pipeline_stages
```

Expected:

```text
FAIL ... num_stages / loop_carried_state not found
```

- [ ] **Step 3: Implement pipeline stage analysis**

```cpp
Map<String, Any> stage_info;
stage_info.Set("num_stages", Integer(num_stages));
stage_info.Set("stage_local_buffers", stage_local_buffers);
stage_info.Set("loop_carried_state", loop_carried_state);
attrs.Set("blackhole.pipeline_stages", Array<Any>{stage_info});
func.CopyOnWrite()->attrs = DictAttrs(attrs);
```

**已实现状态（2026-03-31）**

- `AnalyzeBlackholeFragmentRegions` 已实现并接入 `SplitBlackholeKernel` 后的主链
- `AnalyzeBlackholePipelineStages` 已实现并接入 `SplitBlackholeKernel` 后的主链
- `test_blackhole_flash_attention_analysis.py` 当前为 `3 passed`
- Blackhole `lower()` 主链入口已收正：`is_device_call()` 现在会把 Blackhole entry `PrimFunc` 当作 device 输入，避免在 `SplitBlackholeKernel` 前把 `main` 过滤掉
- `AnalyzeBlackholeFragmentRegions` 已收紧为 region-local pointwise 检测，不再把普通索引算术误记成 `pointwise_chain`
- flash-attention forward 当前 target-level 红灯已前移到 `LowerBlackholeOps` 的显式 fragment-subset fail-fast，`test_blackhole_flash_attention_pipeline.py` 当前为 `1 passed`

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k pipeline_stages
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: add pipeline stage analysis"
```

### Task 5: Consume Analysis in `LowerBlackholeOps`

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.h`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [x] **Step 1: Write failing pipeline/spec tests**

```python
def test_flash_attention_forward_lowers_without_attention_specific_schema():
    kernel = tilelang.compile(
        example_mha_fwd_bshd.flashattn(
            batch=1, heads=32, seq_len=256, dim=128, is_causal=False,
            block_M=128, block_N=128, num_stages=1, threads=128
        ),
        target="blackhole",
        execution_backend="tvm_ffi",
    )
    metadata = kernel.get_metadata()
    assert "flash_attention_plan" not in metadata
    assert "attention_work_contract" not in metadata
    assert "kernels" in metadata
```

- [x] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k without_attention_specific_schema
```

Expected:

```text
FAIL ... unsupported flash-attention forward fragment region
```

- [x] **Step 3: Implement minimal lowering consumption**

```cpp
if (auto fragment_regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
  ValidateSupportedFragmentRegionSubset(fragment_regions.value());
  EmitFragmentRegionLoweringRequirements(fragment_regions.value(), &attrs);
}
if (auto work_info = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
  EmitPerWorkRuntimeRequirements(work_info.value(), &attrs);
}
if (auto pipeline_info = func->GetAttr<Array<Any>>("blackhole.pipeline_stages")) {
  EmitPipelineStageRequirements(pipeline_info.value(), &attrs);
}
```

- [x] **Step 4: Keep `ExecutableSpec` minimal**

```cpp
// Only freeze final runtime/materialization facts.
// Do not encode raw fragment_regions or pipeline_stages into ExecutableSpec.
```

- [x] **Step 5: Run targeted tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k without_attention_specific_schema
```

Expected:

```text
1 passed
```

Latest result:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/src/target/blackhole_module.h \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "blackhole: lower flash attention forward subset through main path"
```

### Task 6: Add Legality and Fail-Fast Gates

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [x] **Step 1: Add failing legality tests**

```python
def test_flash_attention_forward_rejects_unsupported_stage_config():
    with pytest.raises(tvm.TVMError, match="Blackhole flash-attention forward legality"):
        tilelang.compile(
            example_gqa_fwd_bshd.flashattn(
                batch=1, heads=16, seq_len=1024, dim=128, is_causal=False,
                groups=16, block_M=256, block_N=256, num_stages=4, threads=512
            ),
            target="blackhole",
            execution_backend="tvm_ffi",
        )
```

- [x] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k legality
```

Expected:

```text
FAIL ... missing legality guard
```

- [x] **Step 3: Implement legality gates**

```cpp
ICHECK(IsSupportedFlashAttentionForwardShape(config))
    << "Blackhole flash-attention forward legality: unsupported block/thread/stage combination";
ICHECK(IsSupportedFragmentReductionShape(region))
    << "Blackhole flash-attention forward legality: unsupported fragment reduction shape";
```

- [x] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k legality
```

Expected:

```text
1 passed
```

Latest result:

```text
1 passed
```

Note:

- 当前 legality 回归已经覆盖 full `lower()` 的 GQA `num_stages=4` 主链
- 更后续的 widening work 不再是“先修更早的 `RegionOp`/canonicalization 内部错误”，而是继续把 generic fragment/pipeline requirements 收成更宽的 executable legality / lowering 结论

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "blackhole: gate flash attention forward legality"
```

### Task 7: Example-Level Runtime Validation

**Files:**
- Create: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
- Modify: `tilelang_repo/examples/flash_attention/test_example_flash_attention.py`
- Test: `tilelang_repo/examples/flash_attention/example_mha_fwd_bshd.py`
- Test: `tilelang_repo/examples/flash_attention/example_gqa_fwd_bshd.py`

- [ ] **Step 1: Add example-level failing tests**

```python
def test_blackhole_example_mha_fwd_bshd():
    kernel = example_mha_fwd_bshd.flashattn(
        batch=1, heads=32, seq_len=256, dim=128, is_causal=False,
        block_M=128, block_N=128, num_stages=1, threads=128
    )
    profiler = kernel.get_profiler(target="blackhole", execution_backend="tvm_ffi")
    profiler.assert_allclose(partial(example_mha_fwd_bshd.ref_program, is_causal=False), rtol=0.01, atol=0.01)
```

- [ ] **Step 2: Run the targeted example tests to verify current failure**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k "mha or gqa"
```

Expected:

```text
FAIL or SKIP with clear unsupported-subset reason
```

- [ ] **Step 3: Implement minimal runtime/codegen support to make supported examples pass**

```cpp
// No new algorithm reconstruction in runtime.
// Consume finalized lowering facts only.
```

- [ ] **Step 4: Run build and focused regressions**

Run:

```bash
cmake --build tilelang_repo/build -j32
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
```

Expected:

```text
All targeted tests pass in supported environment; unsupported full-env prerequisites remain explicit skips only
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py \
        tilelang_repo/examples/flash_attention/test_example_flash_attention.py
git commit -m "blackhole: validate flash attention forward examples"
```

### Task 8: Documentation and Wrap-Up

**Files:**
- Modify: `tasks/dev_design/stage4_flash_attention_forward_subset.md`
- Modify: `tasks/dev_design/README.md`
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Update docs with implemented scope and remaining gaps**

```markdown
- implemented: work decomposition / fragment region / pipeline stage analysis
- implemented: Blackhole forward fragment compute subset
- remaining: varlen / backward / wider fragment ops
```

- [ ] **Step 2: Run final verification**

Run:

```bash
git status --short
cmake --build tilelang_repo/build -j32
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
pgrep -af "pytest|ctest|cmake --build" || true
```

Expected:

```text
working tree clean after commit; no lingering background test/build processes
```

- [ ] **Step 3: Commit**

```bash
git add tasks/dev_design/stage4_flash_attention_forward_subset.md \
        tasks/dev_design/README.md \
        tasks/dev_design/final_blackhole_backend_redesign.md \
        tasks/progress.md \
        memory/general_dev.md \
        memory/bugs.md
git commit -m "docs: finalize flash attention forward subset status"
git push
```

## Self-Review

- 规格覆盖：
  - 通用 analysis pass：Task 2/3/4
  - `LowerBlackholeOps` 消费 analysis：Task 5
  - legality：Task 6
  - example-level correctness：Task 7
  - 文档与经验：Task 8
- 无 `TODO/TBD/implement later` 占位
- 计划中的类型与命名保持一致：
  - `work_decomposition`
  - `fragment_regions`
  - `pipeline_stages`
  - 不引入 `flash_attention_plan` 或 `attention_work_contract`
