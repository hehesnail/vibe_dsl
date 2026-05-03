# TileLang Blackhole Backend Progress

> 这是当前 checkout 的执行看板。
> 长期合同看 `tasks/dev_design/`。
> 本文件只回答：现在做哪一项、下一项被什么挡住、
> 当前任务需要知道的边界、最近跑过什么验证。
> 不维护按 HEAD 实时更新的实现库存或历史流水。

## Status

- Date: `2026-05-04`
- Active task: `T6 topk`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Active Boundary

The current admitted direct-runtime surface is T1's buffer-address ABI, the
completed T2 current-placement compute baseline, T3's first explicit
placement / reshard projection surface, T4's external accessor ABI
expansion, and T5's hardened admitted static sharded-L1 GEMM layout:
interleaved DRAM runtime buffers, staged-copy resident L1 / CB-backed views,
the admitted 64B page-indexed copy path, static external sharded L1 accessors,
admitted standalone leaf compute families, current-placement GEMM direct
correctness, static external sharded-L1 GEMM direct correctness for the first
admitted bf16 layouts including 2x2 multi-core sharded execution and all
external bf16 input/output tensors, explicit
`T.MemoryConfig` / `T.annotate_memory_config` placement intent,
`TTTensorMemoryConfigPlan`, `TTOpShardingContract`,
`TTPlacementResolutionPlan`, and `TTReshardPlan` projection for the current
interleaved-DRAM to resident-L1 staged-copy conversion.

T5 does not claim implicit retile/work-coarsening, production DRAM-sharded
weights, N-D production cases, mesh/CCL/NoC, or distributed production
variants.  Those remain later work and must consume the projected
placement/conversion/accessor records instead of inferring sharding or page
metadata.

The T3 hardening gate for the admitted `interleaved_to_sharded`
staged-copy conversion now covers large and oversubscribed copy shapes,
explicit user placement of the resident L1 view, multiple independent
reshard records, non-zero tile offsets, sharded resident-L1 elementwise
compute chains, mixed elementwise-plus-reduce compute chains, corrupted
executable placement / reshard records, and serialization preservation. It
does not expand T3 into external accessors, sharded GEMM, or production
DRAM-sharded weights.

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

## Completed Baseline: T2 Leaf Compute / GEMM Baseline

T2 is complete for the current-placement surface.
Unary, binary, broadcast-cols, reduction, fill/typecast publish, and the
current-placement GEMM baseline now project typed leaf/source/spec contracts.
Admitted leaf and GEMM forms run through `BlackholeModule`; simulator-limited
standalone reduction and standalone fill/typecast publish fail closed with
typed direct-runtime unsupported reasons. Standalone reduction remains gated
until the TT-Sim `tensix_execute_pacr: count=1` row-reduce PACR boundary is
resolved. The full-tile reduce CB to rank-1 output writer binding is now
covered by typed lowering/source checks.

T2 does not claim tensor/value sharding, explicit reshard, sharded GEMM, or
external sharded/page-indexed runtime accessor admission.

## Completed Baseline: T3 Tensor/Value Sharding And Explicit Reshard

T3 is complete for the first explicit placement and reshard surface.
TileLang now exposes `T.MemoryConfig`, `T.ShardSpec`, `T.NDShardSpec`,
`T.CoreGrid`, convenience constructors, and `T.annotate_memory_config`.
User/default global placement lowers into `SpatialPlan.TensorPlacementIntent`.
`TTProgram` carries `TTTensorMemoryConfigPlan`,
`TTOpShardingContract`, `TTPlacementResolutionPlan`, and `TTReshardPlan`;
validators reject placement conflicts and incomplete conversion records.

`ExecutableSpec` projects tensor memory config and reshard records, and
`BlackholeModule` metadata / serialization / direct-runtime admission consume
those records. The first admitted conversion class is the existing
interleaved-DRAM to resident-L1 staged-copy path, represented explicitly as
`interleaved_to_sharded` with `materialization_protocol = staged_copy`.

T3 does not admit external `sharded_accessor_cta` /
`page_indexed_accessor_cta` runtime accessors or sharded GEMM/layout
variants; those remain T4/T5.

## Completed Baseline: T4 External Accessor / Runtime ABI Expansion

T4 is complete for the admitted external accessor surface.
`TTBufferDistributionPlan`, `TTTensorMemoryConfigPlan`, `TTABIPlan`, and
`ExecutableSpec` now project enough records for:

- interleaved DRAM accessors,
- 64B page-indexed DRAM transport accessors,
- static sharded L1 external accessors backed by `TensorAccessorArgs`.

Direct runtime/codegen consume typed accessor offsets/counts and buffer
distribution records.  Missing page metadata, unsupported common-runtime
accessor metadata, and incomplete sharded distribution records fail closed
from executable records before source/runtime guessing.

T4 does not claim dynamic/common-runtime accessor metadata, production
DRAM-sharded weights, N-D sharding, or GEMM/layout variants beyond the
current-placement baseline.

## Completed Baseline: T5 Sharded GEMM / Layout Variants

T5 is complete for the first static external sharded-L1 GEMM layout.
External `A`, `B`, and `C` tensors can carry explicit block-sharded L1
placement intent when the shard grid is covered by the kernel work mapping.
The GEMM source/spec/direct-runtime path consumes T4 `TTABIPlan` /
`ExecutableSpec` sharded accessor records and direct runtime executes the
admitted single-core bf16-input / fp32-output case plus the 2x2 multi-core
bf16-input / fp32-output and all external bf16 cases through
`BlackholeModule`.

Unsupported external sharded-L1 GEMM layouts that require a logical work
mapping change now fail closed from typed records: a runtime-visible sharded
accessor whose `shard_grid_shape` is not covered by the attached
`TTCoreGroup.work_packets` is rejected in `ValidateTTProgram` with an
explicit retile/work-coarsening diagnostic.

T5 does not claim implicit retile/work-coarsening, dynamic/common-runtime
sharded accessors, production DRAM-sharded weights, N-D sharding, or
distributed GEMM variants.

## Active Task: T6 topk

### Problem

Standalone value/index selection is not yet an admitted Blackhole direct
runtime path. T6 must represent `topk` value and index selection with typed
leaf/source/spec contracts and reject unsupported axis, dtype, shape, or
index-layout combinations before source/runtime guessing.  Task design:
`tasks/dev_design/2026-05-03-blackhole-t6-topk.md`.

### Completion Standard

T6 is complete only when:

- admitted value and `int32` index outputs are represented in typed IR/source
  contracts,
- direct runtime correctness covers the first admitted bf16/fp32 input
  surface,
- unsupported `topk` axes, index dtypes, tie-breaking assumptions, and layout
  combinations fail closed with typed diagnostics,
- no external runner or source-name recovery is introduced.

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

## Recent Verification

2026-05-04 UTC T5 multi-core/all-bf16 sharded GEMM hardening:

- `cmake --build build -- -j32` passed.
- TT-Sim T5/T4 targeted selector:
  `test_blackhole_t5_external_sharded_l1_gemm_projects_accessor_contracts`,
  `test_blackhole_t5_external_sharded_l1_gemm_rejects_unmapped_shard_grid`,
  `test_blackhole_t5_external_sharded_l1_gemm_direct_runtime_bf16`,
  `test_blackhole_t5_multicore_external_sharded_l1_gemm_direct_runtime_bf16`,
  `test_blackhole_t5_multicore_external_sharded_l1_gemm_direct_runtime_all_bf16`,
  and
  `test_blackhole_t4_external_sharded_l1_accessor_projects_from_memory_config`
  passed: `6 passed`.
- GEMM non-direct regression subset:
  `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k 'not direct_runtime and not direct_call and not gemm_basic and not multicore' --tb=short`
  passed: `45 passed, 2 skipped, 17 deselected`.
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py`
  passed: `104 passed`.

2026-05-03 UTC T5 static external sharded-L1 GEMM:

- `cmake --build build -- -j32` passed.
- TT-Sim direct-runtime T5 trio:
  `test_blackhole_t5_external_sharded_l1_gemm_projects_accessor_contracts`,
  `test_blackhole_t5_external_sharded_l1_gemm_rejects_unmapped_shard_grid`,
  and `test_blackhole_t5_external_sharded_l1_gemm_direct_runtime_bf16`
  passed.
- GEMM non-direct regression subset:
  `45 passed, 2 skipped, 15 deselected`.
- `test_blackhole_spatial_ir.py`: `104 passed`.
- T4 sharded accessor projection smoke passed.

2026-05-03 UTC T4 external accessor/runtime ABI expansion:

- `cmake --build build -- -j32` passed.
- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`:
  `57 passed, 10 skipped, 1 xfailed`.
- TT-Sim direct runtime:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`:
  `34 passed`.
- Spatial / TTProgram / executable projection:
  `pytest -q tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`:
  `104 passed`.
- Adjacent GEMM schema/source regression without direct-runtime execution:
  `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k 'not direct_runtime and not direct_call and not gemm_basic and not multicore' --tb=short`:
  `43 passed, 2 skipped, 14 deselected`.
- `timeout 240s pytest -q testing/python/target/blackhole/test_blackhole_gemm.py --tb=short`
  timed out in direct-runtime GEMM execution; no residual pytest/TT-Sim
  process remained. This is tracked as an adjacent TT-Sim/GEMM runtime
  boundary, not as a T4 external accessor blocker.

2026-05-03 UTC flash-attn exact-CB lifetime regression fix:

- `cmake --build build -- -j32` passed.
- Targeted flash-attn source lifetime gates:
  `test_flash_attention_small_compute_source_respects_cb_capacity_on_reuse`,
  `test_flash_attention_seq64_bf16_compute_source_keeps_cb_events_queue_consistent`,
  `test_flash_attention_seq64_bf16_compute_source_releases_qk_scores_before_next_scores_publish`,
  `test_flash_attention_seq64_bf16_compute_source_uses_static_tile_zero_for_single_page_cb_inputs`,
  and `test_flash_attention_seq64_bf16_pv_merge_consumes_scaled_acc_o_live_form`
  all passed.
- seq64 direct runtime recovery:
  `test_blackhole_flash_attention_seq64_mha_bf16_forward_direct_runtime` and
  `test_blackhole_flash_attention_seq64_gqa_bf16_forward_direct_runtime`
  both passed.
- Full flash-attn runtime + pipeline:
  `pytest -q -vv testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py --tb=short`
  passed: `95 passed`.
- Adjacent compute runtime regression checks:
  `test_blackhole_t3_compute_runtime.py` passed: `8 passed`;
  `test_blackhole_leaf_compute_runtime.py` passed: `16 passed, 2 skipped`.

2026-05-02 T3 runtime hardening and adjacent Blackhole regression:

- `cmake --build build -- -j32` passed.
- T3 hardening selector in
  `testing/python/target/blackhole/test_blackhole_copy_runtime.py`:
  `13 passed`.
- `pytest -q -vv testing/python/target/blackhole --tb=short`:
  `253 passed, 9 skipped, 1 xfailed`.
- `pytest -q -vv testing/python/transform/test_blackhole_spatial_ir.py --tb=short`:
  `101 passed`.

2026-05-03 UTC T3 sharded compute runtime hardening:

- `cmake --build build -- -j32` passed after final cleanup.
- `pytest -q -vv testing/python/target/blackhole/test_blackhole_t3_compute_runtime.py --tb=short`:
  `7 passed`.
- `pytest -q -vv testing/python/target/blackhole/test_blackhole_leaf_compute_runtime.py --tb=short`:
  `16 passed, 2 skipped`.
- `pytest -q -vv testing/python/target/blackhole/test_blackhole_copy_runtime.py --tb=short`:
  `31 passed`.
- `pytest -q -vv testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py --tb=short`:
  `15 passed, 5 skipped`.
- Remaining target files
  `test_blackhole_gemm.py`, `test_blackhole_copy_pipeline.py`,
  `test_blackhole_flash_attention_pipeline.py`, and
  `test_blackhole_tvm_ffi_export.py`:
  `191 passed, 2 skipped, 1 xfailed`.
- Post-cleanup smoke:
  `test_blackhole_t3_compute_runtime.py` stayed `7 passed`, and
  `test_blackhole_flash_attention_runtime.py::test_blackhole_flash_attention_single_work_item_metadata_drops_contract_family`
  passed.
- `git diff --check` passed.

## Task Queue

当前 active task 是 T6。
T1, T2, T3, T4, and T5 are complete for their stated boundaries.
T6 now owns standalone value/index selection.

| 任务 | 目标 | 依赖 | 完成目标 |
| --- | --- | --- | --- |
| T1 Buffer address ABI 接入执行路径 | Make sharded L1 and page-indexed address ABI real execution contracts. | Current typed placement fields. | Complete. |
| T2 Leaf compute / GEMM baseline | Admit non-flash leaf compute and current-placement GEMM layout baseline. | T1 complete. | Complete. |
| T3 Tensor/value sharding and explicit reshard | Make TTNN-style user placement intent, op placement contracts, placement conflict handling, and reshard plans first-class in the IR chain. | T2 baseline complete; design in `2026-05-02-blackhole-tensor-sharding-and-reshard.md`. | Complete. |
| T4 External accessor/runtime ABI expansion | Admit or precisely reject external `sharded_accessor_cta` and `page_indexed_accessor_cta` runtime/codegen forms. | T1 address ABI and T3 placement/conversion projection. | Complete. |
| T5 Sharded GEMM / layout variants | Admit GEMM/layout variants that depend on real tensor sharding, including explicit retile/work-coarsening when a layout changes logical work mapping. | T3 and T4 complete. | Complete. |
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

- T5.1 Sharded GEMM placement contracts (complete):
  admitted input/output memory configs and typed rejects for unsupported
  layouts.
- T5.2 Retile / work-coarsening plan (complete for admitted surface):
  explicit logical-work mapping changes are not implicit; external sharded
  accessors whose shard grid is not covered by work packets now reject with a
  typed retile/work-coarsening diagnostic.
- T5.3 First sharded GEMM correctness (complete):
  direct correctness for the first admitted sharded layout variant, including
  single-core, 2x2 multi-core, bf16-input / fp32-output, and all external
  bf16 input/output coverage.

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

- Admitted direct-runtime forms remain limited to the T1 surface, admitted T2
  standalone leaf cases, and current GEMM A/B-separated reader and writer
  ranges.
- `sharded_accessor_cta` and `page_indexed_accessor_cta` are typed but not
  admitted as external runtime accessors; T4 owns that gap.
- Workload backlog stays ordered by the task queue: `topk`, then
  materialization, grouped / ragged work, and workload-first paths.

## Latest Verification

Latest implementation batch:
T3 tensor/value sharding and explicit reshard.

Verified:

- `cmake --build /root/dev/vibe_dsl/tilelang_repo/build -- -j32`
- Structure / planner / executable projection:
  `pytest -q testing/python/transform/test_blackhole_spatial_ir.py`
  (`101 passed`)
- Source/spec and runtime-module metadata:
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  (`56 passed, 10 skipped, 1 xfailed`)
- Leaf / GEMM compute schema regression:
  `pytest -q`
  `testing/python/target/blackhole/test_blackhole_leaf_compute_runtime.py::test_blackhole_standalone_leaf_compute_projects_typed_runtime_contracts`
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_kernel_projects_typed_compute_ops_schema`
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compute_ops_carry_typed_operand_bindings`
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_spec_uses_typed_compute_ops_without_legacy_payload`
  (`11 passed`)
- TT-Sim direct staged-copy selector via `scripts/setup_tt_sim.sh`:
  `pytest -q -vv -x testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`
  (`1 passed`)

Observed boundary:

- T3's first admitted conversion is limited to interleaved DRAM source to
  resident L1 sharded view through the existing staged-copy direct-runtime
  path. Other reshard kinds remain typed unsupported until admitted by later
  tasks.
- Broadcast-cols rank-1 RHS materialization must be reader-produced as a
  full-tile CB page. Scalar NOC reads into first-column tile positions are
  not an admitted runtime path.
