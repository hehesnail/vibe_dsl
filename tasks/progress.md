# TileLang Blackhole Backend Progress

> 设计依据只看 `tasks/dev_design/final_blackhole_backend_redesign.md`。
> 本文件只记录 repo HEAD 当前状态、blocker、下一步和最近验证。

## Current Status

- Date: `2026-04-24`
- Active lane: `public specialization residue cleanup`
- Current blocker: none after P0.3; next active risk is P0.4 per-work descriptor still using arg-kind owner truth
- Active item: `P0.4 typed per-work descriptor`
- Broad state:
  - Main layered chain is fixed as `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  - Cleanup task0-task5 broad legacy protocol convergence is complete
  - Support-surface / workload payoff lane remains open, but public specialization residues stay ahead of new workload expansion

## Priority Queue

1. `P0.1 contract-family public surface`
   - Status: `completed`
   - Result: top-level contract-family projection / `ExecutableSpec` / runtime metadata surface removed
   - Verification: build, GEMM, flash pipeline/runtime, spatial/copy/export, selected TT-Sim GEMM

2. `P0.2 compute operand binding`
   - Status: `completed`
   - Result: GEMM A/B/C binding moved from reader/writer runtime arg order to typed `KernelSpec.compute_ops[].operand_bindings`
   - Verification: build, GEMM, spatial/copy/flash/export, selected TT-Sim typed-compute cases

3. `P0.3 host wrapper / codegen buffer binding`
   - Status: `completed`
   - Result: host wrapper, codegen, and direct-runtime accessor ABI now require exact formal buffer identity plus explicit `buffer` role schema
   - Verification: build, copy pipeline, GEMM, spatial/flash/export, selected TT-Sim copy/common-buffer/GEMM

4. `P0.4 typed per-work descriptor`
   - Status: `active`
   - Goal: stop treating `a_tile_start_id` / `b_tile_start_id` / `output_tile_start_id` / `gemm_num_k_tiles` arg-kind names as long-term owner truth
   - Replacement: typed value expressions tied to core plan, compute op dims, and access pattern
   - Gate: runtime/codegen must stop deriving block axes from arg-kind priority

5. `P0.5 materialization host/layout binding`
   - Status: `pending`
   - Goal: remove `_local` suffix, single-output fallback, and shape heuristics from materialization host/layout binding
   - Replacement: explicit `TTMaterializationPlan` / `ExecutableSpec` host binding and layout/axis truth

6. `P0.6 projection payload seed cleanup`
   - Status: `pending`
   - Goal: projection encoders stop seeding executable records from typed-node `payload`
   - Replacement: fresh typed projection maps plus explicit diagnostic allowlist

7. `P1 SpatialPlan live/materialization refinement`
   - Status: `pending after P0`
   - Focus: recurrence, reduction row state, non-zero live-in merge, and wider consumer binding

8. `P1 compute-kind extension`
   - Status: `pending after P0`
   - Focus: add non-GEMM TT-Metal compute instructions as generic `KernelSpec.compute_ops[].kind` entries

9. `P1/P2 workload payoff`
   - Status: `pending after P0/P1 gates`
   - Focus: materialization admission expansion, narrow bridge deletion, then flash-attn direct runtime

## Remaining Debt

- `tl.blackhole_logical_buffer_tile_bridge_specs` remains the only narrow bridge attr
- `per_work_arg_specs` still exposes arg-kind-named descriptors until P0.4 completes
- Materialization host/layout binding still has suffix / fallback / heuristic residue until P0.5 completes
- Some projection paths still use typed-node `payload` as construction seed until P0.6 completes
- Flash-attn compile/source/spec baseline is stable; direct runtime correctness is not yet admitted support surface

## Current Baseline

- Active pass/phase implementation:
  `BuildSpatialPlan -> ValidateSpatialPlan -> SplitBlackholeKernel -> CaptureBlackholeLogicalBridgeSpecs -> PlanTTBlocks -> SelectBlackholeTTMetalBuiltins -> PlanTTCompute/PlanTTTransport/PlanTTSync/PlanTTABI/PlanTTExecution -> BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable`
- Direct runtime admitted support:
  copy equal range stride-1; GEMM A/B-separated reader + writer output; interleaved DRAM accessors with no common runtime accessor args; non-oversubscribed explicit semaphore / remote endpoint subset; admitted bf16 materialization paths documented in the design docs
- Latest P0.3 verification:
  - `cmake --build build -j32`
  - `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  - `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
  - `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py testing/python/transform/test_blackhole_spatial_ir.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  - TT-Sim selected: `test_blackhole_module_direct_call`, `test_blackhole_copy_direct_runtime_materializes_shared_common_runtime_buffer_args`, `test_blackhole_gemm_direct_runtime_uses_typed_compute_ops_without_contract_family`
