# TileLang Blackhole Backend Progress

> 这是当前 checkout 的执行看板。
> 长期合同看 `tasks/dev_design/`。
> 本文件只回答：当前什么是真的、现在做哪一项、下一项被什么挡住、
> 最近跑过什么验证。

## Status

- Date: `2026-05-01`
- Active task: `T1 Buffer address ABI 接入执行路径`
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
- Accessor compile-time ABI kinds are typed:
  `interleaved_accessor_cta`,
  `sharded_accessor_cta`,
  and `page_indexed_accessor_cta`.
- Direct runtime still admits only interleaved DRAM accessor execution.
  Sharded and page-indexed forms are represented, but they are not yet wired
  into real direct-runtime execution.

## Active Task: T1 Buffer Address ABI 接入执行路径

### Problem

typed ABI 字段已经存在，但 sharded 和 page-indexed address 形态还没有接入
source/spec/direct-runtime 的真实执行路径。

这一项不能靠“所有 non-interleaved 形态都 typed reject”收口。
它必须证明新的 address ABI 被真实 consumer 使用。

### Completion Standard

T1 只有在下面四个交付项全部完成后才算完成。

| 交付项 | 目标 | 做法 | 完成标准 |
| --- | --- | --- | --- |
| T1.1 Executable address contract | `ExecutableSpec` / leaf reader carries the address ABI needed by runtime. | Project the relevant `TTBufferDistributionPlan`, `TTAccessorSpec`, and per-work descriptor fields into the executable schema. Remove any source/runtime fallback that reconstructs this from names or layout strings. | Projection tests show the executable schema contains the typed address contract and rejects missing fields. |
| T1.2 Sharded L1 execution sample | At least one sharded L1 path actually runs. | Use a small staged-copy style program with DRAM/global source, resident per-worker L1 shard, work-packet mapping, and core-local address mapping. Generate source/spec/direct-runtime args from typed plans. | TT-Sim bf16 correctness passes for that sharded L1 path. |
| T1.3 Page-indexed execution sample | At least one page-indexed path actually runs. | Use a minimal page-indexed read/copy path with explicit page metadata and per-work address descriptor. Generate source/spec/direct-runtime args from typed plans. | TT-Sim bf16 correctness passes for that page-indexed path. |
| T1.4 Unsupported-form rejects | Remaining unsupported forms fail for the right reason. | For sharded/page-indexed/shared forms outside the admitted execution samples, inspect typed ABI fields and produce precise unsupported reasons. | Negative tests fail closed before source/runtime guessing. |

### Implementation Order

1. Write failing tests for T1.1 projection and missing-field rejects.
2. Implement executable address contract projection and validation.
3. Write failing TT-Sim or direct-runtime tests for the sharded L1 execution sample.
4. Implement sharded L1 source/spec/direct-runtime consumption.
5. Write failing TT-Sim or direct-runtime tests for the page-indexed execution sample.
6. Implement page-indexed source/spec/direct-runtime consumption.
7. Add negative tests for unsupported non-interleaved forms.
8. Run the verification set listed under "Required Verification".
9. Update this file, relevant design docs, and `memory/`.
10. Commit and push.

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

当前只有 T1 是 active task。
在 sharded 和 page-indexed 都有通过 source/spec/direct-runtime 与 TT-Sim
correctness 的执行样例之前，不推进后续任务。

| 任务 | 目标 | 依赖 | 完成目标 |
| --- | --- | --- | --- |
| T1 Buffer address ABI 接入执行路径 | Make sharded L1 and page-indexed address ABI real execution contracts. | Current typed placement fields. | T1.1-T1.4 all complete. |
| T2 Leaf compute / GEMM variants | Admit non-flash leaf compute and GEMM layout variants. | T1. | Direct correctness or typed reject for each admitted leaf family / layout. |
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
  GEMM A/B-separated reader range plus writer output range,
  non-oversubscribed explicit semaphore / remote-endpoint path,
  and admitted bf16 live-form paths.
- Current typed-but-not-yet-executed address forms:
  sharded L1 and page-indexed accessors.
  These are the active T1 target.
- Current workload backlog:
  `topk`,
  MoE / `fusedmoe`,
  paged attention / paged decode / MLA decode,
  grouped / ragged / sparse attention,
  and chunk recurrence / scan.

## Latest Verification

Latest implementation batch:
typed sharded L1 buffer address ABI split and fail-closed validator/projection
gates.

Verified:

- `cmake --build build -j32`
- `pytest -q tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
  (`90 passed`)
- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  (`165 passed, 25 skipped, 1 xfailed, 4 warnings`)
- TT-Sim env via `scripts/setup_tt_sim.sh`,
  then
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  (`16 passed`)
- TT-Sim env via `scripts/setup_tt_sim.sh`,
  then copy-pipeline direct-runtime ABI path
  (`9 passed`)
