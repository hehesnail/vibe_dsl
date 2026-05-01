# TileLang Blackhole Backend Progress

> 这是当前 checkout 的执行看板。
> 长期合同看 `tasks/dev_design/`。
> 本文件只回答：现在做哪一项、下一项被什么挡住、
> 当前任务需要知道的边界、最近跑过什么验证。
> 不维护按 HEAD 实时更新的实现库存或历史流水。

## Status

- Date: `2026-05-02`
- Active task: `T2 Leaf compute / GEMM variants`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Active Boundary

The current admitted direct-runtime surface is still T1's buffer-address ABI:
interleaved DRAM runtime buffers, staged-copy resident L1 / CB-backed views,
and the admitted 64B page-indexed copy path.

Full tensor/value sharding is a design lane, not current implementation.
Sharded GEMM/layout claims in T2 and production sharding claims in T7 must
wait for the DSL placement surface, tensor memory-config plans, op sharding
contracts, placement conflict handling, and explicit reshard plans.

## Completed Baseline: T1 Buffer Address ABI

T1 is complete for the current admitted direct-runtime surface.
The active non-negotiable boundaries carried into T2 are:

- Do not silently retile GPU-style `alloc_shared((tile_m, tile_n))` to fill
  Blackhole L1. Treat it as per-worker, per-work-item scratch unless an
  explicit retile/work-coarsening plan changes the logical work mapping and
  source-region mapping together.
- Keep TileLang logical work grid, physical worker group, temporal work
  packets, and per-worker L1 / CB scratch reuse separate.
- Runtime must execute typed contracts. It must not infer source regions,
  shard ownership, page metadata, or buffer roles from names, suffixes,
  argument order, or layout strings.

## Active Task: T2 Leaf Compute / GEMM variants

### Problem

The address ABI is now wired into the admitted direct-runtime path.
The next blocker is leaf compute / GEMM coverage:
more leaf families and GEMM layout variants must either run with direct
correctness or fail closed from typed leaf / layout contracts.

### Completion Standard

T2 is complete only when unary / binary / broadcast / reduction / pack /
typecast leaf families and the required GEMM layout variants have:

- explicit leaf/source/spec contracts,
- direct-runtime correctness where admitted,
- typed unsupported reasons where not admitted,
- no fallback to composite source matchers, names, or payloads.

## Required Verification

每个 active implementation task 都使用这张验收表。

| 层级 | 要求 |
| --- | --- |
| Compile | C++ build succeeds with `cmake --build build -j32`. |
| Structure | TIR / `TTProgram` / executable projection tests prove the typed fields exist and old fallbacks are absent. |
| Source/spec | Materialized executable schema contains the real address contract used by the source/runtime path. |
| Direct runtime | The admitted path runs through `BlackholeModule`, not an external runner. |
| TT-Sim correctness | Runtime correctness uses the repository TT-Sim setup and bf16 baseline. |
| Unsupported reason | Unsupported forms fail closed with typed diagnostics before source/runtime guessing. |

## Task Queue

当前 active task 是 T2。

| 任务 | 目标 | 依赖 | 完成目标 |
| --- | --- | --- | --- |
| T1 Buffer address ABI 接入执行路径 | Make sharded L1 and page-indexed address ABI real execution contracts. | Current typed placement fields. | Complete. |
| T2 Leaf compute / GEMM variants | Admit non-flash leaf compute and GEMM layout variants. | T1 complete. | Direct correctness or typed reject for each admitted leaf family / layout. |
| T3 `topk` | Admit standalone value/index selection. | T1 and required leaf reductions. | Value and `int32` index correctness, not compile-only. |
| T4 Exact-CB / materialization primitives | Repair wider publish/consume, partial combine, and source-live-form materialization. | T1. | Multi-kernel intermediate correctness and typed materialization rejects. |
| T5 Grouped / ragged work packets | Represent group/ragged metadata as typed planning input. | T1 and relevant per-work descriptors. | Missing/inconsistent group metadata rejects before source/runtime emission. |
| T6 Workload first paths | Bring up pre-grouped MoE, sparse/ragged attention, paged GQA decode, paged MLA decode, and chunk recurrence in that order. | T1-T5 as needed by each workload. | Each workload has a stated first path with correctness proof and unsupported-form rejects. |
| T7 Distributed production variants | Add mesh/sharding/CCL/NoC/multicast/global scheduling support. | Stable first paths and typed distributed plans. | Production distributed paths have typed placement, communication, and correctness gates. |

Sharded GEMM/layout claims inside T2 and production sharding claims inside T7
must additionally depend on the tensor/value sharding and explicit reshard
design lane:

- DSL / user placement intent capture,
- `TTTensorMemoryConfigPlan`,
- `TTOpShardingContract`,
- placement conflict validation,
- `TTReshardPlan` or typed rejects.

## Support Boundary

- Admitted direct-runtime forms remain limited to the T1 surface plus current
  GEMM A/B-separated reader and writer ranges.
- `sharded_accessor_cta` and `page_indexed_accessor_cta` are typed but not
  admitted as external runtime accessors.
- Workload backlog stays ordered by the task queue: `topk`, then grouped /
  ragged / materialization work, then workload-first paths.

## Latest Verification

Latest implementation batch:
TT-Metal sharding contract alignment audit.

Verified:

- `cmake --build /root/dev/vibe_dsl/tilelang_repo/build -- -j32`
- Red/green structure selectors:
  `pytest -q testing/python/transform/test_blackhole_spatial_ir.py::test_plan_tt_abi_uses_hardware_backed_buffer_distribution testing/python/transform/test_blackhole_spatial_ir.py::test_validate_tt_program_rejects_incomplete_sharded_address_abi`
  (failed before the fix because `shard_orientation` was `block` and
  invalid strategy/orientation values were not rejected; passes after the
  fix with `2 passed`)
- TT-Sim direct-runtime selectors via `scripts/setup_tt_sim.sh`:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_page_indexed_copy_consumes_address_contract`
  (`2 passed`)

Observed boundary:

- 32B bf16 sub-tile page transport is not admitted.
  TT-Sim rejects the NOC read/write with an address-alignment mismatch.
  The admitted page-indexed sample remains the 64B page path.
