# TileLang Blackhole Backend Progress

> 这是当前 checkout 的执行看板。
> 长期合同看 `tasks/dev_design/`。
> 本文件只回答：当前什么是真的、现在做哪一项、下一项被什么挡住、
> 最近跑过什么验证。

## Status

- Date: `2026-05-01`
- Active task: `T2 Leaf compute / GEMM variants`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current HEAD

当前代码状态：

- `BlackholeModule` in-process direct host path is the formal execution path.
  The `execution_backend="tvm_ffi"` wrapper/export path is available.
- Legacy external runner / `build_blackhole/` is gone.
- Legacy protocol families are out of the active chain:
  `compute_contract`,
  `gemm_contract`,
  `multi_*_contracts`,
  top-level `TTProgram.payload`,
  bridge attrs,
  lowering facts contract maps,
  compute-op seed maps,
  and leaf name/default fallbacks.
- `TTResourceDemand` and `TTResourcePressureReport` are in `TTProgram` and
  are consumed by `ValidateTTProgram`.
- Core groups consume `TTHardwareModel` worker-grid facts; validators reject
  out-of-grid cores, duplicate cores, and invalid work packets.
- `TTBufferDistributionPlan` can represent hardware-backed placement:
  interleaved DRAM,
  attached-core sharded L1 for shared / CB-backed views,
  and device-local replicated placement for ordinary per-worker local L1.
- Sharded L1 placement has typed address ABI fields:
  `shard_grid_shape`,
  `sharding_strategy`,
  real per-core `shard_shape`,
  `source_buffer`,
  `source_region_kind`,
  `source_region_shape`,
  `logical_index_mapping`,
  and `core_local_address_mapping`.
  Pure local sharded scratch does not fabricate source-region binding.
- `TTBufferDistributionPlan` is now projected into `ExecutableSpec` as the
  runtime-visible buffer address contract.
  `BlackholeModule` metadata / serialization preserves it, leaf readers
  validate it, and direct-runtime admission consumes it before execution.
- Sharded L1 is admitted for the staged-copy resident L1 / CB-backed path:
  the executable carries DRAM `source_buffer`,
  `per_work_tile` source region,
  `work_packet_row_major` logical mapping,
  and `l1_shard_linear` core-local mapping;
  TT-Sim bf16 correctness passes.
- Page-indexed 64B stick/page copy is admitted through typed
  `transport_page_size`,
  executable `page_size_bytes`,
  and `interleaved_page_index` mapping;
  direct runtime correctness passes.
  32B bf16 sub-tile page transport remains outside the admitted boundary
  because TT-Sim rejects the NOC read/write alignment.
- Accessor compile-time ABI kinds are typed:
  `interleaved_accessor_cta`,
  `sharded_accessor_cta`,
  and `page_indexed_accessor_cta`.
- Direct runtime still admits external runtime buffers as interleaved DRAM
  accessor execution.
  `sharded_accessor_cta` and `page_indexed_accessor_cta` remain fail-closed
  for external compile-time accessor materialization until a later task
  gives them a direct TT-Metal accessor ABI.

## Completed Task: T1 Buffer Address ABI 接入执行路径

### Result

T1 is complete for the current admitted direct-runtime surface.

| 交付项 | 状态 |
| --- | --- |
| T1.1 Executable address contract | Done. `ExecutableSpec` carries `buffer_distribution_plans`; metadata, serialization, and leaf validation preserve required address fields. Missing sharded `source_buffer` rejects during build. |
| T1.2 Sharded L1 execution sample | Done. Grid-indexed staged copy uses DRAM source -> resident per-worker L1 / CB-backed `A_shared`; direct runtime consumes the typed sharded distribution and passes TT-Sim bf16 correctness. |
| T1.3 Page-indexed execution sample | Done for admitted 64B page transport. Stick/page copy uses explicit `transport_page_size`, executable `page_size_bytes`, and `interleaved_page_index`; direct runtime correctness passes. 32B bf16 sub-tile page transport is a recorded alignment boundary, not silently admitted. |
| T1.4 Unsupported-form rejects | Done. Runtime-bound buffers with non-interleaved / replicated distribution fail from typed distribution fields; non-admitted compile-time accessor kinds remain typed rejects. |

### Non-Negotiable Boundaries

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

T1 已完成。
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

## Support Boundary

- Current direct-runtime admitted execution:
  interleaved DRAM buffer address schema with no common runtime args,
  copy equal source/destination range with stride 1,
  selected interleaved stick/page-shaped copy cases,
  staged-copy sharded L1 / CB-backed resident view with typed source region,
  GEMM A/B-separated reader range plus writer output range,
  non-oversubscribed explicit semaphore / remote-endpoint path,
  and admitted bf16 live-form paths.
- Current typed-but-not-yet-executed external accessor forms:
  `sharded_accessor_cta` and `page_indexed_accessor_cta`.
  They are not T1 blockers because the admitted sharded/page-indexed address
  contracts now execute through buffer distribution + interleaved external
  DRAM accessors.
- Current workload backlog:
  `topk`,
  MoE / `fusedmoe`,
  paged attention / paged decode / MLA decode,
  grouped / ragged / sparse attention,
  and chunk recurrence / scan.

## Latest Verification

Latest implementation batch:
T1 buffer address ABI execution integration.

Verified:

- `cmake --build /root/dev/vibe_dsl/tilelang_repo/build -- -j32`
- TT-Sim env via `scripts/setup_tt_sim.sh`,
  then
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py testing/python/target/blackhole/test_blackhole_copy_runtime.py testing/python/transform/test_blackhole_spatial_ir.py`
  (`171 passed, 1 skipped, 1 xfailed, 4 warnings`)
- Targeted red/green T1 selectors:
  missing executable sharded source-buffer reject,
  unadmitted buffer-distribution typed reject,
  sharded L1 bf16 direct runtime,
  and page-indexed 64B direct runtime
  (`4 passed`)

Observed boundary:

- 32B bf16 sub-tile page transport is not admitted.
  TT-Sim rejects the NOC read/write with an address-alignment mismatch.
  The admitted page-indexed sample remains the 64B page path.
