# Spatial Strong Contract And Phase C Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen `SpatialProgram` from name-linked object graph into an index-linked contract layer that can serve as a real `Phase C` input boundary.

**Architecture:** Keep `SemanticProgram -> SpatialProgram -> TTProgram` as the only mainline. First strengthen `Task / Channel / Placement / SyncEdge / ProgramPhase` with explicit linkage contracts, then make `ValidateSpatialProgram` consume those contracts as primary truth, and finally use the resulting stronger spatial boundary to start the `Phase C` translator cutover instead of letting downstream passes keep guessing.

**Tech Stack:** C++ TVM/TIR transform passes, TVM FFI reflection objects, Python transform/target tests, Blackhole direct-runtime regression suite.

---

### Task 1: Add failing tests for index-linked spatial contracts

**Files:**
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`

- [ ] **Step 1: Write failing contract-projection tests**
- [ ] **Step 2: Run focused tests to verify they fail before implementation**

### Task 2: Strengthen `SpatialProgram` linkage schema

**Files:**
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/lower_to_spatial_program.cc`

- [ ] **Step 1: Add explicit payload contracts for `Task / Channel / Placement / SyncEdge / ProgramPhase`**
- [ ] **Step 2: Project stable phase/task/channel/state indices in copy/GEMM/generic builders**
- [ ] **Step 3: Re-run focused transform tests**

### Task 3: Make `ValidateSpatialProgram` index-first

**Files:**
- Modify: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`

- [ ] **Step 1: Require linkage payloads even when display-name fields are present**
- [ ] **Step 2: Validate phase/task/channel/state ownership through indices**
- [ ] **Step 3: Re-run full spatial transform tests**

### Task 4: Bridge the stronger spatial boundary into downstream cutover work

**Files:**
- Modify: `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- Modify: `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`

- [ ] **Step 1: Record the new strong-contract boundary in Phase B/Phase C docs**
- [ ] **Step 2: Use the strengthened schema as the required input boundary for TT translator work**
- [ ] **Step 3: Re-run target/runtime regression baseline, commit, and push**
