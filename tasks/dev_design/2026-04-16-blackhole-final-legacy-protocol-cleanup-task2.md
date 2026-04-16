# Task 2: Remove Public And Internal Legacy Analysis Bags

## Scope

Delete the public `AnalyzeBlackhole*` transform exports, delete the corresponding internal evidence bags, and remove `blackhole.lowering_requirements` from the active planning chain.

After this task, any pass that needs information must do one of two things:

1. read the current `PrimFunc` / `SpatialPlan` / `TTProgram` directly and immediately rewrite or build from that truth; or
2. if the result must survive, write it straight into the canonical owner object or the final executable projection being constructed.

What it may **not** do:

- call another helper pass to pre-digest semantics into a bag
- create a replacement `Map<String, Any>` contract
- export another analysis wrapper through `tilelang.transform`

If `blackhole_lowering_requirements.cc` survives after this task, it may only contain leaf projection or leaf validation helpers. It must no longer be an internal semantic carrier for planning.

## Files

- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Delete: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Delete: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Delete: `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- Modify: `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Delete: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

## Execution Slices

1. add the public-surface regression
2. confirm the wrapper exports still exist
3. remove the public wrapper exports
4. migrate remaining consumers off evidence bags and `blackhole.lowering_requirements`
5. delete the registered passes, evidence helpers, and obsolete tests
6. rerun the transform/flash-attn suites and commit

- [ ] **Step 1: Add the public-surface regression**

Replace wrapper-surface expectations with canonical-surface checks:

```python
def test_blackhole_transform_no_longer_exports_legacy_analysis_wrappers():
    for name in (
        "AnalyzeBlackholeWorkDecomposition",
        "AnalyzeBlackholeComputeRegions",
        "AnalyzeBlackholePipelineStages",
    ):
        assert not hasattr(tilelang.transform, name)
```

Move unique content checks into canonical-path coverage:

```python
def test_flash_attention_tt_program_payload_exposes_bridge_buffers_without_legacy_attrs():
    lowered = _lower_flash_attention_to_tt_target()
    payload = dict(require_tt_program(lowered).payload)
    assert "buffer_tile_bridge_specs" in payload
    assert lowered.attrs.get("blackhole.work_decomposition") is None
    assert lowered.attrs.get("blackhole.compute_regions") is None
    assert lowered.attrs.get("blackhole.pipeline_stages") is None
```

- [ ] **Step 2: Run the wrapper-surface test and verify failure**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k no_longer_exports_legacy_analysis_wrappers
```

Expected: FAIL because `tilelang.transform` still exports the wrapper names.

- [ ] **Step 3: Remove the public wrapper exports**

Delete from `tilelang/transform/__init__.py`:

```python
def AnalyzeBlackholeWorkDecomposition(): ...
def AnalyzeBlackholeComputeRegions(): ...
def AnalyzeBlackholePipelineStages(): ...
```

Do not replace them with new public wrappers.

- [ ] **Step 4: Migrate remaining consumers off evidence bags and `blackhole.lowering_requirements`**

Replace bag readers with direct current-stage reads implemented inside the consuming pass or inside tiny local helpers in the same `.cc`:

```cpp
// bad:
Map<String, Any> work = AnalyzeBlackholeWorkDecompositionEvidence(func);
Array<Any> stages = AnalyzeBlackholePipelineStageEvidence(func);
Map<String, Any> reqs = GetBlackholeLoweringRequirements(func);

// good:
AppendKernelAccessorsFromCurrentIR(func, spatial_plan, current_kernel);
AppendOrderingFromCurrentSpatialPlan(spatial_plan, current_program);
AppendLeafProjectionFromTTProgram(program, executable);
```

The exact helper names are local implementation detail. The contract is:

- the helper stays in the same consuming file
- it writes directly into the owner object or projection under construction
- it does not return a new semantic bag

After this step, no active-chain reader may depend on:

- `AnalyzeBlackholeWorkDecompositionEvidence`
- `AnalyzeBlackholeComputeRegionEvidence`
- `AnalyzeBlackholePipelineStageEvidence`
- `blackhole.lowering_requirements` as a planning/semantic bag

- [ ] **Step 5: Delete the registered passes, evidence helpers, and obsolete tests**

Delete the registered transform passes, FFI exports, and evidence helper entrypoints:

```cpp
Map<String, Any> AnalyzeBlackholeWorkDecompositionEvidence(const PrimFunc& func);
Map<String, Any> AnalyzeBlackholeComputeRegionEvidence(const PrimFunc& func);
Array<Any> AnalyzeBlackholePipelineStageEvidence(const PrimFunc& func);
tir::transform::Pass AnalyzeBlackholeWorkDecompositionPass() { ... }
```

Delete:

```text
tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
```

If any useful assertions remain in that file, move them into:

- `testing/python/transform/test_blackhole_spatial_ir.py`
- `testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 6: Re-run the transform/flash-attn suites and commit**

Run:

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/tilelang/transform/__init__.py \
        tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc \
        tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git rm tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc \
       tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc \
       tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc \
       tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "blackhole: delete legacy analysis bags"
```
