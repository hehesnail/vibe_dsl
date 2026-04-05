# Stage 4 Phase B: Spatial Program IR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把冻结后的语义真相投影成 `SpatialProgram`，一等化 task/channel/layout/sync/work partition，并证明下游不会退化回 `Task:TTKernel = 1:1`。

**Architecture:** `Phase B` 的重点是明确 `ProgramPhase` 的 module-scope ownership、simple-workload fast-path 和 non-trivial multi-phase gate。它不负责 target 物化，只负责把 semantic truth 转成稳定的空间程序结构。

**Tech Stack:** TileLang transform passes, TVM Object system, pytest

---

## Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## Task 3: Stage 3 - Phase B Spatial Program IR

**Files:**
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/tilelang/engine/phase.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Introduce `SpatialProgram` and `ProgramPhase`**

Required objects:

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

Rules:

- module-scope `ProgramPhase` truth lives in `tl.device_programs`
- member-local truth lives in `PrimFunc.attrs["tl.spatial_program"]`
- simple workload gets canonical fast-path

- [ ] **Step 2: Add simple-workload fast-path for copy / GEMM**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'copy or gemm or fast_path' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:

- copy / GEMM 不需要进入重 candidate synthesis
- trivial workload 仍能快速构造 canonical `SpatialProgram`

- [ ] **Step 3: Add one non-trivial multi-phase spatial gate**

Recommended first gate: `flash-attn`

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'flash_attention or multi_phase' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- 至少一个 case 证明下游不会退化回 `Task:TTKernel = 1:1`
- `ProgramPhase` / `Channel` / phase-boundary materialization 在结构测试里可见

- [ ] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 5: Stage 3 exit gate**

Only proceed when:

- `SpatialProgram` 能消费冻结后的 semantic truth
- simple-workload fast-path 稳定
- 至少一个 non-trivial multi-phase spatial gate 通过
