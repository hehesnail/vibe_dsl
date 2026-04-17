# Task 1: Replace Compute-Region Bags With Direct Logical Bridge Capture

## Scope

Introduce one narrow mutating pass that walks the current `PrimFunc` and attaches only `tl.blackhole_logical_buffer_tile_bridge_specs` when the optimized/helper entry still needs pre-opt logical buffer/tile correspondence.

This pass is not a replacement analysis layer.

- it reads current IR directly
- it may consult current representation objects directly
- it mutates the `PrimFunc` once by attaching one narrow attr
- it must not publish another analysis wrapper, evidence bag, or reusable semantic object

This attr remains temporary and leaf-local. It is not planning representation and must not grow into a replacement for `blackhole.compute_regions`.

## Files

- Create: `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/common.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

## Execution Slices

1. add the failing bridge-capture regression
2. confirm the current helper path still depends on compute-region bags
3. implement direct bridge capture as a local visitor inside the new pass
4. attach only the narrow bridge attr
5. cut remaining bridge consumers over to the narrow attr or direct local IR walks
6. thread the pass into the Blackhole lowering path
7. rerun the flash-attn helper tests and commit

- [ ] **Step 1: Write the failing regression for optimized helper bridge capture**

Add a regression that asserts the optimized helper path preserves the logical bridge specs without requiring `blackhole.compute_regions`:

```python
def test_flash_attention_optimized_helper_path_keeps_logical_bridge_specs():
    lowered = _run_flash_attention_tt_target_after_optimize(... )["main"]
    payload = dict(require_tt_program(lowered).payload)
    buffers = {str(spec["buffer"]) for spec in payload["buffer_tile_bridge_specs"]}
    assert {"acc_s", "acc_o", "scores_max"}.issubset(buffers)
    assert lowered.attrs.get("blackhole.compute_regions") is None
```

- [ ] **Step 2: Run the targeted flash-attn test and verify failure**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k optimized_helper_path_keeps_logical_bridge_specs
```

Expected: FAIL because the helper path still depends on compute-region bag extraction or drops the bridge specs.

- [ ] **Step 3: Implement direct bridge capture inside the new pass**

Implement the capture logic as a visitor local to `capture_blackhole_logical_bridge_specs.cc`, for example:

```cpp
class LogicalBridgeCapture final : public tir::StmtExprVisitor {
 public:
  explicit LogicalBridgeCapture(const Optional<SpatialPlan>& spatial_plan);
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;
  Array<PrimExpr> EncodeSpecs() const;
};
```

The visitor may keep local `(Buffer, shape, scope)` tuples while it walks the current IR, but that local state must die with the pass. It must not become:

- a new attr bag
- a new exported analysis pass
- a new helper layer

- [ ] **Step 4: Attach only the narrow bridge attr**

Use the local visitor inside the pass and attach only `tl.blackhole_logical_buffer_tile_bridge_specs`:

```cpp
PrimFunc CaptureLogicalBridgeSpecs(const PrimFunc& func) {
  LogicalBridgeCapture capture(GetOptionalSpatialPlan(func));
  capture(func->body);
  if (capture.EncodeSpecs().empty()) {
    return func;
  }
  return WithAttr(std::move(func),
                  attr::kTLBlackholeLogicalBufferTileBridgeSpecs,
                  capture.EncodeSpecs());
}
```

No other bridge payload is allowed here.

- [ ] **Step 5: Cut remaining bridge consumers over to the narrow attr or direct local IR walks**

Remove remaining direct dependencies on compute-region evidence for bridge/buffer recovery:

```cpp
// remove:
Map<String, Any> compute_region_evidence = AnalyzeBlackholeComputeRegionEvidence(func);

// replace with:
Optional<ObjectRef> bridge_specs =
    func->GetAttr(attr::kTLBlackholeLogicalBufferTileBridgeSpecs);
```

If a consumer can recover what it needs directly from current IR, do that instead of reading even the narrow attr.

This cutover applies to current helper-path readers in:

- `lower_blackhole_ops.cc`
- `blackhole_lowering_requirements.cc`

- [ ] **Step 6: Thread the pass into the Blackhole lowering path**

Use the pass in the Blackhole lowering pipeline:

```python
with target:
    mod = LowerAndLegalize(mod, target)
    if target.kind.name == "blackhole":
        mod = tilelang.transform.CaptureBlackholeLogicalBridgeSpecs()(
            LowerToBlackholePhaseB(mod)
        )
```

Keep the pass close to the point where the helper/optimized entry still needs this leaf bridge. Do not let it drift upward into the planning stages.

- [ ] **Step 7: Re-run the flash-attn helper tests and commit**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k 'optimized_helper_path_keeps_logical_bridge_specs or gqa_'
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc \
        tilelang_repo/tilelang/engine/lower.py \
        tilelang_repo/tilelang/engine/phase.py \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc \
        tilelang_repo/testing/python/target/blackhole/common.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "blackhole: replace compute-region bags with direct bridge capture"
```
