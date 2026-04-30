# TileLang Blackhole GEMM TT-Sim Note

## Status

`tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/` has been removed.

It was a bring-up example that uploaded row-major host buffers directly, bypassed the formal
host layout contract, and no longer reflects the Blackhole direct-path execution model.

## Why It Was Removed

- It duplicated the main GEMM validation path with a separate hand-written TT-Metal example.
- Its host-side data movement contract diverged from the real runtime path.
- Keeping it in-tree risked treating a stale bring-up sample as the canonical reference.

## Maintained Validation Path

Use the maintained Python direct-path tests selected by
`tasks/progress.md`. For TT-Sim runtime tests, source the environment
in the same shell as the test command:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q <selected blackhole pytest target>
```

This note records why the old hand-written TT-Metal example was removed;
it is not a task-level test matrix.
