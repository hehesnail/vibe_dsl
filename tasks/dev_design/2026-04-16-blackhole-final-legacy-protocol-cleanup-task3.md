# Task 3: Replace `blackhole.copy_semantics` With Direct IR/Dataflow Recovery

## Scope

Replace `blackhole.copy_semantics` annotations with direct recovery from current `Normalized Tile TIR` plus `SpatialPlan.DataflowEdge` inside the consuming passes.

This task is intentionally small in abstraction surface:

- no shared copy-analysis layer
- no shared copy-direction attr
- no exported copy matcher
- no new helper vocabulary beyond what already exists on IR and representation objects

If two consumers need nearly the same structural walk, duplication is preferable to exporting another semantic layer. If a tiny shared helper is truly necessary, it may only factor traversal mechanics and must stay internal.

## Files

- Modify: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- Modify: `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`

## Execution Slices

1. add the failing copy-semantics regressions
2. confirm current code still reads `blackhole.copy_semantics`
3. implement direct IR/dataflow matching inside the consumers
4. switch `SplitBlackholeKernel` to direct recovery
5. switch resource canonicalization and delete the old prepass
6. rerun the copy suite and commit

- [ ] **Step 1: Write the failing copy-semantics regressions**

Add one planner-facing and one resource-facing regression:

```python
def test_blackhole_copy_planning_does_not_require_copy_semantics_annotations():
    ...

def test_blackhole_resource_canonicalization_recovers_copy_meaning_from_ir():
    ...
```

Both tests should strip `blackhole.copy_semantics` and still expect correct behavior.

- [ ] **Step 2: Run the copy tests and verify failure**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k 'does_not_require_copy_semantics_annotations or recovers_copy_meaning_from_ir'
```

Expected: FAIL because current code still reads the legacy attr.

- [ ] **Step 3: Implement direct IR/dataflow matching inside the consumers**

Recover copy meaning by matching current IR directly against `SpatialPlan.DataflowEdge`.

One acceptable shape is a tiny file-local helper in anonymous namespace:

```cpp
struct MatchedCopyTransfer {
  Buffer src;
  Buffer dst;
  const DataflowEdge* edge;
  bool fused_stage_copy = false;
};

std::optional<MatchedCopyTransfer> MatchCopyTransfer(
    const BufferStoreNode* store, const SpatialPlan& spatial_plan);
```

The important constraints are:

- use actual buffer identity and `SpatialPlan.DataflowEdge`
- do not use names
- do not re-publish a `CopyDirection` contract
- keep the helper local to the consuming `.cc`

- [ ] **Step 4: Switch `SplitBlackholeKernel` to direct recovery**

Use the local matcher directly in `split_blackhole_kernel.cc`:

```cpp
if (auto match = MatchCopyTransfer(store, spatial_plan)) {
  if (IsReaderTransport(*match->edge) || match->fused_stage_copy) {
    current_kernel_kind_ = "reader";
  }
  if (IsWriterTransport(*match->edge)) {
    current_kernel_kind_ = "writer";
  }
}
```

This logic should drive the current rewrite/planner decision immediately. It must not materialize a new cross-pass semantic object.

- [ ] **Step 5: Switch resource canonicalization and delete the old prepass**

Use the same structural recovery discipline in `blackhole_device_resource_canonicalization.cc`:

```cpp
if (auto match = MatchCopyTransfer(store, spatial_plan)) {
  if (IsReaderTransport(*match->edge)) cb_inputs_.insert(match->dst);
  if (IsWriterTransport(*match->edge)) cb_outputs_.insert(match->src);
}
```

Delete the old prepass from the pipeline:

```python
mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
```

- [ ] **Step 6: Re-run the copy suite and commit**

Run:

```bash
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k phase_b
```

Expected: PASS

Commit:

```bash
git add tilelang_repo/src/transform/split_blackhole_kernel.cc \
        tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc \
        tilelang_repo/tilelang/engine/phase.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
git commit -m "blackhole: recover copy meaning from ir and spatial dataflow"
```
