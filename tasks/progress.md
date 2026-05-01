# TileLang Blackhole Backend Progress

> 这是当前 checkout 的执行看板。
> 长期合同看 `tasks/dev_design/`。
> 本文件只回答：现在做哪一项、下一项被什么挡住、
> 当前任务需要知道的边界、最近跑过什么验证。
> 不维护按 HEAD 实时更新的实现库存或历史流水。

## Status

- Date: `2026-05-02`
- Active task: `T2 Leaf compute / GEMM baseline`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Active Boundary

The current admitted direct-runtime surface is still T1's buffer-address ABI:
interleaved DRAM runtime buffers, staged-copy resident L1 / CB-backed views,
and the admitted 64B page-indexed copy path.

Full tensor/value sharding is a design lane, not current implementation.
It is now queued as the next major task after the current T2 baseline.
Sharded GEMM/layout claims and production sharding claims must wait for the
DSL placement surface, tensor memory-config plans, op sharding contracts,
placement conflict handling, and explicit reshard plans.
External `sharded_accessor_cta` / `page_indexed_accessor_cta` runtime
admission is a separate T4 task after the sharding IR exists.

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

## Active Task: T2 Leaf Compute / GEMM Baseline

### Problem

The address ABI is now wired into the admitted direct-runtime path.
The next blocker is leaf compute / GEMM coverage for current admitted
placements:
more leaf families and non-sharded GEMM layout variants must either run with
direct correctness or fail closed from typed leaf / layout contracts.

### Completion Standard

T2 is complete only when unary / binary / broadcast / reduction / pack /
typecast leaf families and the current-placement GEMM baseline have:

- explicit leaf/source/spec contracts,
- direct-runtime correctness where admitted,
- typed unsupported reasons where not admitted,
- no fallback to composite source matchers, names, or payloads.

T2 does not claim sharded GEMM/layout support.
That belongs after T3 tensor/value sharding and explicit reshard.

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
T3 is the tensor/value sharding task; it starts after T2 baseline is complete
and before any sharded GEMM/layout claim.
T4 owns the currently typed-but-not-admitted external accessor ABI forms.

| 任务 | 目标 | 依赖 | 完成目标 |
| --- | --- | --- | --- |
| T1 Buffer address ABI 接入执行路径 | Make sharded L1 and page-indexed address ABI real execution contracts. | Current typed placement fields. | Complete. |
| T2 Leaf compute / GEMM baseline | Admit non-flash leaf compute and current-placement GEMM layout baseline. | T1 complete. | Direct correctness or typed reject for each admitted leaf family / current-placement layout. |
| T3 Tensor/value sharding and explicit reshard | Make TTNN-style user placement intent, op placement contracts, placement conflict handling, and reshard plans first-class in the IR chain. | T2 baseline complete; design in `2026-05-02-blackhole-tensor-sharding-and-reshard.md`. | `T.MemoryConfig` / `T.annotate_memory_config` lower to `TensorPlacementIntent`; `TTTensorMemoryConfigPlan`, `TTOpShardingContract`, conflict rejects, `TTReshardPlan`, and executable projection exist with tests. |
| T4 External accessor/runtime ABI expansion | Admit or precisely reject external `sharded_accessor_cta` and `page_indexed_accessor_cta` runtime/codegen forms. | T1 address ABI and T3 placement/conversion projection. | External sharded/page-indexed accessors have direct TT-Metal ABI records and runtime/codegen admission, or fail from explicit executable accessor records. |
| T5 Sharded GEMM / layout variants | Admit GEMM/layout variants that depend on real tensor sharding, including explicit retile/work-coarsening when a layout changes logical work mapping. | T3 complete; T4 when external sharded/page-indexed accessors are required. | Sharded GEMM/layout correctness where admitted; typed rejects for unsupported placement/conversion/retile combinations. |
| T6 `topk` | Admit standalone value/index selection. | T2 leaf reductions. | Value and `int32` index correctness, not compile-only. |
| T7 Exact-CB / materialization primitives | Repair wider publish/consume, partial combine, source-live-form materialization, and multi-block flash-attn / flash-decode exact-CB correctness. | T1 and relevant T3 materialization rules when sharded values are involved. | Multi-kernel intermediate correctness and typed materialization rejects. |
| T8 Grouped / ragged work packets | Represent group/ragged metadata as typed planning input. | T1 and relevant per-work descriptors. | Missing/inconsistent group/ragged metadata rejects before source/runtime emission. |
| T9 Workload first paths | Bring up pre-grouped MoE, sparse/ragged attention, paged GQA decode, paged MLA decode, chunk recurrence, and multi-block flash decode first paths. | Prior tasks as needed by each workload. | Each workload has a stated first path with correctness proof and unsupported-form rejects. |
| T10 Distributed production variants | Add mesh/sharding/CCL/NoC/multicast/global scheduling support. | Stable first paths and typed distributed plans, including T3 sharding/reshard. | Production distributed paths have typed placement, communication, and correctness gates. |

## Scope Breakdown

Top-level tasks are review / planning boundaries.
Large tasks must land through these smaller checkpoints.

### T2 Leaf Compute / GEMM Baseline

- T2.1 Leaf contract matrix:
  admitted TT-Metal leaf names, operand/result schemas, source/spec records,
  and typed unsupported categories.
- T2.2 Elementwise / pack / typecast families:
  unary, binary, broadcast, pack, and typecast direct correctness or typed
  rejects.
- T2.3 Reduction family:
  reduce leaf contracts, access/live-form evidence, and typed rejects for
  unsupported axes or shapes.
- T2.4 Current-placement GEMM baseline:
  non-sharded GEMM layout variants that use the existing admitted placement
  surface; no sharded GEMM claim.

### T3 Tensor/Value Sharding And Explicit Reshard

- T3.1 DSL placement surface:
  `T.MemoryConfig`, `T.ShardSpec`, `T.NDShardSpec`,
  `T.annotate_memory_config`, constructor sugar, and frontend validation.
- T3.2 `SpatialPlan.TensorPlacementIntent`:
  lower user configs and explicit defaults into target-independent placement
  intent; prove `scope`, `T.Kernel`, and `T.annotate_layout` are not sharding
  APIs.
- T3.3 `TTTensorMemoryConfigPlan`:
  mirror TTNN `MemoryConfig + ShardSpec / NdShardSpec` and validate
  consistency with low-level buffer distribution.
- T3.4 `TTOpShardingContract` and placement conflict rejects:
  op input/output placement contracts plus deterministic producer/consumer
  conflict diagnostics.
- T3.5 `TTReshardPlan` and executable projection:
  explicit conversion records, first admitted conversion path, and
  runtime/codegen fail-closed consumption.

### T4 External Accessor / Runtime ABI Expansion

- T4.1 Executable accessor schema:
  project enough typed records for external sharded and page-indexed accessors.
- T4.2 `sharded_accessor_cta` admission:
  direct TT-Metal accessor ABI or precise typed reject.
- T4.3 `page_indexed_accessor_cta` admission:
  page metadata ABI, runtime/codegen consumption, and precise typed reject for
  unadmitted page shapes.

### T5 Sharded GEMM / Layout Variants

- T5.1 Sharded GEMM placement contracts:
  admitted input/output memory configs and typed rejects for unsupported
  layouts.
- T5.2 Retile / work-coarsening plan:
  explicit logical-work mapping changes and source-region/address mapping
  changes when a layout requires them.
- T5.3 First sharded GEMM correctness:
  direct correctness for the first admitted sharded layout variant.

### T7 Exact-CB / Materialization Primitives

- T7.1 Source-live-form materialization:
  full logical value proof, materialized view records, and consumer binding.
- T7.2 Exact-CB publish / consume:
  CB event lifetime, producer/consumer synchronization, and typed rejects.
- T7.3 Partial combine and multi-block flash:
  partial-output / logsum combine plus multi-block flash-attn / flash-decode
  exact-CB correctness.

### T8 Grouped / Ragged Work Packets

- T8.1 Group metadata schema:
  group sizes, group offsets, padded offsets, and group-to-work mapping as
  typed planning inputs.
- T8.2 Ragged range descriptors:
  row counts, cache sequence lengths, and per-work indexed ranges with shape
  validation.
- T8.3 Sparse/block index metadata:
  sparse block indices and block-table evidence without name-based recovery.
- T8.4 Work-packet validation:
  reject missing or inconsistent grouped/ragged metadata before source or
  runtime emission.

### T9 Workload First Paths

Each workload is a separate first-path checkpoint:

- T9.1 pre-grouped MoE / routed grouped GEMM
- T9.2 paged GQA decode
- T9.3 paged MLA decode
- T9.4 sparse / ragged attention
- T9.5 chunk recurrence / scan
- T9.6 multi-block flash decode

### T10 Distributed Production Variants

- T10.1 mesh / multi-device placement records
- T10.2 CCL contracts:
  all-gather, reduce-scatter, all-to-all, and collective admission
- T10.3 NoC / multicast / global scheduling plans
- T10.4 distributed workload correctness and typed production rejects

## Support Boundary

- Admitted direct-runtime forms remain limited to the T1 surface plus current
  GEMM A/B-separated reader and writer ranges.
- `sharded_accessor_cta` and `page_indexed_accessor_cta` are typed but not
  admitted as external runtime accessors; T4 owns that gap.
- Workload backlog stays ordered by the task queue: `topk`, then
  materialization, grouped / ragged work, and workload-first paths.

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
