# TileLang Blackhole GEMM TT-Sim Note

## Status

`tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/` has been removed.

It was a bring-up example that uploaded row-major host buffers directly, bypassed the formal
host layout contract, and no longer reflects the Blackhole direct-path execution model.

## Why It Was Removed

- It duplicated the main GEMM validation path with a separate hand-written TT-Metal example.
- Its host-side data movement contract diverged from the real runtime path.
- Keeping it in-tree risked treating a stale bring-up sample as the canonical reference.

## Current Validation Path

Use the Python direct-path tests instead:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py
```

These tests exercise the maintained TileLang -> Blackhole direct host path and are the current
source of truth for TT-Sim validation in this repository.
