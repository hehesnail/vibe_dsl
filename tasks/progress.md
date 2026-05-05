# TileLang Blackhole Backend Progress

> 当前 checkout 的执行看板。
> 长期架构合同看 `tasks/dev_design/`。
> 本文件只保留当前状态、active task、后续 gate、最近验证摘要。

## Status

- Date: `2026-05-05`
- Active task: `T9 Workload first paths`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Task | State | Current boundary |
| --- | --- | --- |
| T1 Buffer address ABI | Complete | Runtime consumes typed buffer/address records for interleaved DRAM, staged-copy resident L1 views, and the admitted 64B page-indexed copy path. |
| T2 Leaf compute / GEMM baseline | Complete | Admitted non-flash leaf families and current-placement GEMM run through `BlackholeModule` or fail closed with typed reasons. |
| T3 Tensor/value sharding and explicit reshard | Complete | `T.MemoryConfig`, placement intents, tensor memory-config plans, op sharding contracts, placement resolution, and first `interleaved_to_sharded` staged-copy conversion are typed and projected. |
| T4 External accessor / runtime ABI | Complete | External interleaved, 64B page-indexed DRAM, and static sharded-L1 accessors are executable records consumed by source/runtime; unsupported dynamic/common-runtime forms reject from typed records. |
| T5 Sharded GEMM / layout variants | Complete | First static external sharded-L1 GEMM layouts pass direct runtime, including single-core, 2x2 multi-core, 110-core many-core all-bf16, and first K-dimension partial-sum correctness path. |
| T6 `topk` | Complete | Existing-TIR row-wise value/index selection runs through direct runtime for fp32 and bf16 values with exact `int32` indices, without a frontend topk op or selection plan. |
| T7 Exact-CB / materialization primitives | Complete | Exact-CB materialization is admitted through typed live-form/materialization/consumer-binding records, including GEMM post-merge `pack_tile`, source-live `cb_republish`, and seq64 bf16 flash-attn exact-CB partial-combine direct runtime correctness. |
| T7.5 Exact-CB liveness / allocation cutover | Complete | Covered exact-CB resident tiles use typed TTProgram/ExecutableSpec lifecycle, allocation, and release records; old loop-carried owner maps, materialization-pop fallback, and full-tile/slice ambiguity are fail-closed or deleted from the active path. |
| T8 Irregular work domains / indexed access | Implementation | Grid-indexed, one-dimensional and launch-axis two-dimensional table-indexed tile descriptors, scaled contiguous indexed-block copies, sparse multi-entry indexed traversal with per-entry ragged bounds, predicate-derived ragged row/page bounds, copy-shaped paged `cache_seqlens`, per-page valid-row tables, single/multiple non-uniform segmented row ranges, and the T9.1 segmented A grouped-GEMM feed carry TIR-derived evidence through TT per-work descriptors and direct runtime; broader irregular forms remain open. |
| T9 Workload first paths | Implementation | T9.1 pre-grouped MoE/routed grouped GEMM is admitted through ordinary TIR-derived segment-row descriptors, scratch-CB tiled A materialization, and bf16 direct-runtime correctness; T9.2-T9.6 remain queued. |
| T10 Distributed production variants | Queued | Mesh, CCL, NoC/multicast/global scheduling, distributed workload correctness, and production partial-K reduction protocol. |

## Active Boundary Notes

- Runtime/codegen must consume `ExecutableSpec` leaf records; no source-name,
  argument-position, accessor-string, or runtime observation recovery.
- `T.Kernel` describes logical work items.  Tensor sharding comes from
  explicit placement intent and resolved memory-config plans.
- T5 K-sharded GEMM currently proves correctness with blocking logical-z waves
  plus a runtime-issued device tile-add reduction.  It is not the production
  single-launch or fused-launch semaphore/NoC partial-reduce protocol.
  T10.5 owns replacing that path and deleting or folding the temporary special
  route.
- For T6-T10, validators and projection tests are support evidence only.
  An admitted positive path must execute through `BlackholeModule` under the
  repository TT-Sim setup and compare device output against a host reference.
- Larger flash-attn shapes exposed the exact-CB resident lifecycle/resource
  allocation boundary that T7.5 cut over for the covered surface.
  The design contract remains
  `tasks/dev_design/2026-05-05-blackhole-exact-cb-liveness-allocation.md`.
  Covered exact-CB paths must use TTProgram lifecycle/allocation records for
  physical CB choice and release events; old source-emitter map/fallback
  lifecycle routes are not compatibility paths.

## Completed Task: T6 `topk`

T6 admits standalone value/index selection as a real Blackhole direct-runtime
path.  Task design:
`tasks/dev_design/2026-05-03-blackhole-t6-topk.md`.

T6 completed the admitted existing-TIR row-wise subset:

- the frontend shape remains ordinary Tile TIR: `T.copy`, `T.fill`,
  `T.reduce_max`, `T.if_then_else`, local value/index buffers, and explicit
  global stores;
- codegen lowers the typed value/index selection records to one backend scan
  over the reader-materialized input CB and publishes typed value/index output
  CB pages for the normal writer;
- fp32 values plus exact `int32` indices pass direct runtime for
  `M=320`, `N=128`, `k=6`, `axis=1`, `blk_m=64`;
- bf16 values plus exact `int32` indices pass direct runtime with `M > blk_m`;
- no `T.topk`, `tl.blackhole.topk`, `TTSelectionPlan`, `selection_plans`,
  external runner, source-name recovery, or raw compute-side host pointer path
  was introduced.

Unsupported axis/layout/generalized value-index variants remain outside the
admitted T6 subset and must fail closed through the existing typed legality
surface until a later task broadens that subset.

## Completed Task: T7 Exact-CB / Materialization

T7 admits exact-CB materialization as a typed backend contract instead of a
source-name or runtime fallback surface.

The completed subset covers:

- `TTLiveFormPlan`, `TTMaterializationPlan`, and `TTConsumerBindingPlan`
  records for source live forms, materialized CB tile forms, publication
  protocol, and consumer binding;
- GEMM post-merge cast-consumer exact-CB `cb_republish` / `pack_tile`
  correctness through direct runtime;
- seq64 bf16 flash-attn MHA direct runtime where the same lowered artifact
  proves no exact-CB unsupported reasons, one-page publish/consume event
  windows, `acc_s -> acc_s_cast` `cb_republish` via
  `tilize_cast_fragment_slice`, device-side tiled partial combine, and host
  reference correctness;
- typed reject boundaries for unsupported materialization/event forms before
  runtime execution.

No frontend materialization op, alternate runtime combiner, mailbox fallback,
legacy payload, or source-name semantic recovery is part of the T7 contract.

Wider workload surfaces that reuse these primitives remain in later lanes:
T8 derives irregular/ragged/indexed work from TIR, T9 admits workload-first
paths, and T10 owns production distributed / fused partial-reduction protocols.

## Completed Task: T7.5 Exact-CB Liveness / Allocation Cutover

T7.5 moved the covered flash exact-CB resident surface from emitter-local
lifetime repair to typed TTProgram / ExecutableSpec lifecycle records.

The completed subset covers:

- exact-CB virtual values, use events, live intervals, physical CB allocation,
  and release events for the covered loop-carried flash surface;
- source rendering of `cb_wait_front`, `cb_push_back`, and `cb_pop_front`
  through lifecycle/release records instead of loop-carried cb/buffer twin maps
  or completed-state recovery;
- borrowed exact-input last-use rendering through the release-policy helper,
  without the old local `ShouldRelease...` path;
- materialization with `pop_front=true` failing closed unless a typed
  `TTExactCBReleaseEvent` exists;
- validator rejects for missing loop-carried exit evidence, overlapping
  exact-CB intervals sharing a physical CB, and full-logical-tile consumers
  bound to `thread_distributed_slice` live forms;
- seq64 bf16 flash-attn exact-CB partial-combine direct runtime correctness,
  plus seq128/256/512 source/spec admission that skips only on the typed
  TT-Sim `tensix_execute_pacr: count=1` capability boundary.

T8 owns deriving irregular/ragged/indexed work domains from TIR and making that
evidence drive source/runtime addressing.  T9/T10 own workload-first and
distributed production variants.

## Required Verification

Every active implementation task uses this acceptance table.

| Level | Requirement |
| --- | --- |
| Compile | C++ build succeeds with `cmake --build build -j32`. |
| Structure | TIR / `SpatialPlan` / `TTProgram` / executable projection tests prove typed fields exist and old fallbacks are absent. |
| Source/spec | Materialized executable schema contains the records consumed by source/runtime. |
| Direct runtime | The admitted path runs through `BlackholeModule`, not an external runner. |
| TT-Sim correctness | Runtime correctness uses the repository TT-Sim setup and bf16 baseline when tensor values are involved. |
| Unsupported reason | Unsupported forms fail closed with typed diagnostics before source/runtime guessing. |

## Remaining Runtime Correctness Gates

### T8 Irregular Work / Indexed Access

- Indexed block/page traversal beyond the admitted launch-axis table-backed,
  contiguous scaled-block, sparse multi-entry, and sparse+ragged cases, where
  `BufferLoad` / `BufferStore` indices use operands such as broader sparse
  `block_indices`.
- Broader ragged token/page forms beyond the admitted one-dimensional
  row-count, per-entry sparse row-bound, copy-shaped paged `cache_seqlens`,
  and per-page valid-row predicate slices.
- Broader segmented/grouped workload use beyond the admitted single- and
  two-range row-segment copy and T9.1 grouped-GEMM feed.
- In every case, the derived evidence must drive source/runtime addressing.
  Projection-only tests do not complete T8.

### T9 Workload First Paths

Each checkpoint needs its own direct-runtime correctness proof:

- T9.2 paged GQA decode:
  bf16 page/block-table KV reads with ragged `cache_seqlens`, more than one
  page, and the admitted partial combine path.
- T9.3 paged MLA decode:
  bf16 paged latent / KV access through the admitted page-table and ragged
  bound surface.
- T9.4 sparse / ragged attention:
  bf16 indexed sparse-block traversal plus ragged valid lengths.
- T9.5 chunk recurrence / scan:
  multi-chunk loop-carried state and device state-buffer lifetime.
- T9.6 multi-block flash decode:
  bf16 multi-block split with exact-CB publish/consume and partial combine.

### T10 Distributed Production

- T10.1 mesh / multi-device placement:
  admitted mesh or multi-device runtime movement and computation across more
  than one device when the simulator/target supports it.
- T10.2 CCL contracts:
  all-gather, reduce-scatter, and all-to-all correctness over at least two
  logical shards/devices for every admitted collective contract.
- T10.3 NoC / multicast / global scheduling:
  multi-core producer/consumer correctness through the admitted semaphore,
  remote route, or multicast protocol.
- T10.4 distributed workload correctness:
  at least one T9 first path in its admitted distributed form end to end.
- T10.5 K-sharded GEMM production partial reduce:
  replace the current blocking z-wave tile-add path with typed reducer records
  and run a many-core bf16 case such as
  `M=320`, `N=352`, `K>=512`, `logical_grid=11x10x2` or larger.

## Recent Verification

2026-05-05 UTC T9.1 pre-grouped GEMM checkpoint:

- `cmake --build build -j32` passed.
- The grouped GEMM frontend remains ordinary Tile TIR: `GroupOffsets[g]`
  drives A row start, `GroupSizes[g]` drives the guarded A copy predicate,
  W is selected by the same logical group axis, and no frontend grouped-GEMM
  op or registry was added.
- The reader executable contains A `segment_row_start` and
  `segment_row_count` descriptors with `value_source=index_table`,
  `index_buffer=GroupOffsets/GroupSizes`, and
  `index_table_index_sources=[logical_block_x]`.
- Source contains no raw `GroupOffsets` / `GroupSizes` loads and no
  `a_segment_row_start / 32` tile addressing.  A materialization uses a
  page-indexed 64-byte row-tile accessor, a scratch CB, `copy_cb_page`, and
  zero-fill for invalid rows before publishing tiled CB pages to GEMM.
- Direct runtime accepts the segmented A backing tensor as raw row-major,
  decodes grouped output as `(groups * tile_m, tile_n)`, and compares bf16
  grouped GEMM output against a host reference with non-uniform group sizes
  `[17, 0, 32]` and offsets `[3, 40, 12]`.
- Focused selectors
  `test_blackhole_t9_grouped_gemm_projects_segmented_a_descriptors` and
  `test_blackhole_t9_grouped_gemm_direct_runtime_bf16` reported `2 passed`.
- Adjacent T8 indexed/ragged/segmented/page selectors over
  `test_blackhole_copy_pipeline.py` and `test_blackhole_copy_runtime.py`
  reported `32 passed`.
- GEMM schema/source selector
  `test_blackhole_gemm.py -k 'not direct_runtime and not direct_call and not gemm_basic and not multicore'`
  reported all selected tests passed with one expected skip.

2026-05-05 UTC T8 per-page valid-row checkpoint:

- `cmake --build build -j32` passed.
- Minimal paged valid-row copy now admits `PageTable[bx, by]` for source page
  selection and `PageValidRows[bx, by]` for per-page row bounds.
- The A tile-start descriptor and A valid-row descriptor both carry
  `value_source=index_table`, `index_table_shape=[2,3]`, and
  `index_table_index_sources=[logical_block_x,logical_block_y]`.
- Source consumes `a_tile_start_id` and `a_valid_rows`, emits no raw
  `PageTable` / `PageValidRows` loads, and does not need
  `a_ragged_page_index` because the predicate is already page-local
  `row < valid_rows`.
- Direct runtime correctness passed for nontrivial page order and per-page
  bounds including 32, 24, 17, 5, 1, and 0 valid rows.
- Focused selectors
  `test_blackhole_paged_valid_rows_copy_uses_per_page_row_bounds`
  and
  `test_blackhole_module_direct_call_paged_valid_rows_copy_uses_per_page_bounds`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including the per-page
  valid-row case reported `28 passed`.

2026-05-05 UTC T8 sparse multi-entry ragged checkpoint:

- `cmake --build build -j32` passed.
- Minimal three-entry sparse ragged copy now admits
  `BlockIndices[bx, 0/1/2]` and `ValidRows[bx, 0/1/2]` in one logical work
  item.  No new frontend sparse op or registry was introduced.
- Source consumes `a_tile_start_id`, `a_tile_start_id_1`,
  `a_tile_start_id_2`, `a_valid_rows`, `a_valid_rows_1`, and
  `a_valid_rows_2`, with no raw `BlockIndices` / `ValidRows` loads.
- Each descriptor carries `value_source=index_table`,
  `index_table_shape=[3,3]`, and a per-entry literal source
  `constant:0`, `constant:1`, or `constant:2`; direct runtime evaluates each
  per-work arg from that typed descriptor.
- Direct runtime correctness passed for non-contiguous bf16 tiles with
  per-entry row counts including full rows, partial rows, one row, and zero
  rows.
- Focused selectors
  `test_blackhole_sparse_3tile_ragged_copy_scales_per_entry_descriptors`
  and
  `test_blackhole_module_direct_call_sparse_3tile_ragged_copy_uses_per_entry_bounds`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including the three-entry
  sparse ragged case reported `26 passed`.

2026-05-05 UTC T8 multi-segment row-range checkpoint:

- `cmake --build build -j32` passed.
- Minimal two-range segmented row copy now admits two independent
  `SegmentOffsets[bx, 0/1]` starts and two independent
  `SegmentCounts[bx, 0/1]` bounds in one logical work item.
- Source consumes `a_segment_row_start`, `a_segment_row_start_1`,
  `a_segment_row_count`, and `a_segment_row_count_1`, with no raw
  `SegmentOffsets` / `SegmentCounts` loads and no A-side tile-start fallback.
- Row-page source rendering now derives the source page id from the zeroed TIR
  row expression, so non-32-aligned row starts use
  `segment_row_start + page_row` rather than `segment_row_start / 32`.
- Direct runtime correctness passed for offsets
  `[[3,40],[12,55],[1,70]]` and counts `[[11,0],[32,7],[19,24]]`,
  including a zero-row second segment and full/partial rows.
- Focused selectors
  `test_blackhole_segmented_row_copy_uses_segment_index_table_descriptors`,
  `test_blackhole_two_segment_row_copy_uses_per_range_descriptors`,
  `test_blackhole_module_direct_call_segmented_row_copy_uses_start_and_count_tables`,
  and
  `test_blackhole_module_direct_call_two_segment_row_copy_uses_per_range_tables`
  reported `4 passed`.
- T8 indexed/ragged/segmented aggregate selectors including grid-indexed,
  one- and two-dimensional block-indexed, scaled block, sparse, sparse+ragged,
  ragged row, segmented row, two-range segmented row, and paged cache-length
  cases reported `24 passed`.

2026-05-05 UTC T8 sparse indexed plus per-entry ragged checkpoint:

- `cmake --build build -j32` passed.
- The table-backed per-work runtime-arg seed is now descriptor-kind generic for
  the admitted indexed copy family: tile-start and valid-row descriptors both
  preserve their own `index_table_shape` and `index_table_index_sources`.
- Minimal same-work sparse ragged copy now admits `BlockIndices[bx, 0/1]` and
  `ValidRows[bx, 0/1]` together.  Source consumes `a_tile_start_id`,
  `a_tile_start_id_1`, `a_valid_rows`, and `a_valid_rows_1`, with no raw
  table loads.
- Direct runtime evaluates each per-work arg from its descriptor and proves
  bf16 correctness for non-contiguous tiles with independent valid-row counts,
  including zero-row and partial-row entries.
- Focused selectors
  `test_blackhole_sparse_2tile_ragged_copy_uses_per_entry_valid_rows`
  and
  `test_blackhole_module_direct_call_sparse_2tile_ragged_copy_uses_per_entry_bounds`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including both sparse cases
  reported `20 passed`.

2026-05-05 UTC T8 sparse two-entry indexed traversal checkpoint:

- `cmake --build build -j32` passed.
- Minimal same-work sparse indexed copy now admits two independent table-derived
  A tile starts.  `BlockIndices[bx, 0]` lowers to `a_tile_start_id` and
  `BlockIndices[bx, 1]` lowers to `a_tile_start_id_1`.
- Each A tile-start descriptor carries `value_source=index_table`,
  `index_buffer=BlockIndices`, `index_table_shape=[3,2]`, and distinct
  `index_table_index_sources` ending in `constant:0` / `constant:1`.
- Source consumes the projected runtime args and emits no raw `BlockIndices`
  load.  Direct runtime evaluates the two per-work args from the typed
  descriptors and copies non-contiguous bf16 tiles into compact output order.
- Focused selectors
  `test_blackhole_sparse_2tile_copy_uses_two_index_table_tile_start_descriptors`
  and
  `test_blackhole_module_direct_call_sparse_2tile_copy_uses_two_table_entries`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including the sparse case
  reported `18 passed`.

2026-05-05 UTC T8 scaled indexed-block checkpoint:

- `cmake --build build -j32` passed.
- Minimal two-tile block-indexed staged copy now admits a table value as a
  contiguous block id rather than a single tile id.  The A tile-start
  descriptor carries `value_source=index_table`, `index_buffer=BlockIndices`,
  and `index_value_scale=2`.
- Direct runtime scales the table value to a tile start before passing
  `a_tile_start_id`; source lowering consumes that runtime arg as the base
  tile id and emits the second subtile as `a_tile_start_id + 1`, not
  `a_tile_start_id * 2 + 1`.
- Focused selectors
  `test_blackhole_block_indexed_2tile_copy_scales_index_table_descriptor`
  and
  `test_blackhole_module_direct_call_block_indexed_2tile_copy_uses_scaled_table`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including the scaled-block
  and paged cases reported `16 passed`.

2026-05-05 UTC T8 paged ragged cache-length checkpoint:

- `cmake --build build -j32` passed.
- Minimal `PageTable[bx, by]` plus `CacheSeqLens[bx]` staged copy now lowers
  the A tile-start descriptor from the page table with
  `index_table_shape=[2,3]` and
  `index_table_index_sources=[logical_block_x,logical_block_y]`.
- The cache-length table lowers to A `valid_rows` with
  `value_source=index_table`, `index_buffer=CacheSeqLens`, and
  `index_table_index_sources=[logical_block_x]`.
- The launch page axis lowers to A `ragged_page_index` with
  `value_source=logical_block_y`; source consumes
  `a_tile_start_id`, `a_valid_rows`, and `a_ragged_page_index`, and emits no
  raw `PageTable` / `CacheSeqLens` loads.
- The reader uses row-page transport and zero-fills invalid rows according to
  `logical_block_y * 32 + page_row < cache_len`; the direct-runtime case
  passed for a nontrivial 2x3 page table and cache lengths `[45,70]`.
- Focused selectors
  `test_blackhole_paged_cache_len_copy_uses_page_table_and_ragged_page_descriptors`
  and
  `test_blackhole_module_direct_call_paged_cache_len_copy_uses_page_table_and_seq_lens`
  reported `2 passed`.
- T8 indexed/ragged/segmented aggregate selectors including the paged case
  reported `14 passed`.

2026-05-05 UTC T8 launch-axis indexed-table checkpoint:

- `cmake --build build -j32` passed.
- Minimal `BlockIndices[bx, by]` staged copy now lowers to a table-backed
  A tile-start `TTPerWorkArgSpec` with `value_source=index_table`,
  `index_buffer=BlockIndices`, `index_table_shape=[2,3]`, and
  `index_table_index_sources=[logical_block_x,logical_block_y]`, with the
  descriptor still linked to SpatialPlan `AccessRegion` evidence.
- Device source consumes `a_tile_start_id` as a projected runtime arg and
  does not emit a raw `BlockIndices` load.
- Direct runtime evaluates the index-table element from the projected shape
  and launch-axis sources instead of flattening by `work_linear_id`; the
  non-symmetric `2x3` table case passed through `BlackholeModule`.
- `BlackholeModule` binary serialization preserves the new table addressing
  fields; the loaded module passed the same direct-runtime correctness check.
- T8 indexed/ragged/segmented aggregate selectors passed:
  `test_blackhole_grid_indexed_copy_per_work_specs_expose_typed_descriptors`,
  `test_blackhole_block_indexed_copy_per_work_spec_uses_index_table_descriptor`,
  `test_blackhole_block_indexed_2d_copy_per_work_spec_carries_table_addressing`,
  `test_blackhole_ragged_row_copy_uses_valid_rows_index_table_descriptor`,
  `test_blackhole_segmented_row_copy_uses_segment_index_table_descriptors`,
  `test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`,
  `test_blackhole_module_direct_call_block_indexed_copy_uses_index_table`,
  `test_blackhole_module_direct_call_block_indexed_2d_copy_uses_table_addressing`,
  `test_blackhole_serialized_block_indexed_2d_copy_preserves_table_addressing`,
  `test_blackhole_module_direct_call_block_indexed_copy_rejects_out_of_range_index_table`,
  `test_blackhole_module_direct_call_ragged_row_copy_uses_row_count_predicate`,
  and
  `test_blackhole_module_direct_call_segmented_row_copy_uses_start_and_count_tables`
  reported `12 passed`.

2026-05-05 UTC T8 segmented row-segment checkpoint:

- `cmake --build build -j32` passed.
- Minimal non-uniform segmented row copy now lowers `SegmentOffsets[bx]` to
  `descriptor_kind=segment_row_start` and `SegmentCounts[bx]` to
  `descriptor_kind=segment_row_count`, both with `value_source=index_table`
  and access-region evidence linked to A reads.
- Device source consumes `a_segment_row_start` and `a_segment_row_count`;
  it does not emit raw source reads from `SegmentOffsets` / `SegmentCounts`
  and does not synthesize the old A-side `a_tile_start_id` /
  tile-count / tile-stride descriptor for the segmented input path.
- The segmented reader uses 64-byte page-indexed bf16 row pages at
  `segment_row_start + page_row`; invalid rows are zero-filled into CB pages,
  and the writer emits a compact per-work output block.
- Direct runtime correctness passed for non-32-aligned starts
  `[3,37,48]`, counts `[11,0,29]`, and compact output blocks.
- T8 indexed/ragged/segmented aggregate selectors passed:
  `test_blackhole_grid_indexed_copy_per_work_specs_expose_typed_descriptors`,
  `test_blackhole_block_indexed_copy_per_work_spec_uses_index_table_descriptor`,
  `test_blackhole_ragged_row_copy_uses_valid_rows_index_table_descriptor`,
  `test_blackhole_segmented_row_copy_uses_segment_index_table_descriptors`,
  `test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`,
  `test_blackhole_module_direct_call_block_indexed_copy_uses_index_table`,
  `test_blackhole_module_direct_call_block_indexed_copy_rejects_out_of_range_index_table`,
  `test_blackhole_module_direct_call_ragged_row_copy_uses_row_count_predicate`,
  and
  `test_blackhole_module_direct_call_segmented_row_copy_uses_start_and_count_tables`
  reported `9 passed`.

2026-05-05 UTC T8 ragged row-bound checkpoint:

- `cmake --build build -j32` passed.
- Minimal `RowCounts[bx]` staged copy now lowers to a table-backed
  `TTPerWorkArgSpec`: `descriptor_kind=valid_rows`,
  `value_source=index_table`, `index_buffer=RowCounts`, with the descriptor
  linked to A read `AccessRegion` evidence.
- Device source consumes `a_valid_rows` as a projected runtime arg, reads valid
  bf16 row pages, zero-fills invalid row pages in CB, and writes all 32 rows
  back so TIR `if_then_else(load, 0)` semantics do not depend on output
  initialization.
- Direct runtime now treats 64-byte bf16 page-indexed row/stick pages as
  row-major host pages; nfaces host tilization is restricted to complete
  32x32 tile pages.
- Selectors passed:
  `test_blackhole_ragged_row_copy_uses_valid_rows_index_table_descriptor`,
  `test_blackhole_module_direct_call_ragged_row_copy_uses_row_count_predicate`,
  `test_blackhole_module_direct_call_stick_copy`,
  and
  `test_blackhole_module_direct_call_page_indexed_copy_consumes_address_contract`
  reported `4 passed`.
- T8 indexed/ragged aggregate selectors passed:
  `test_blackhole_grid_indexed_copy_per_work_specs_expose_typed_descriptors`,
  `test_blackhole_block_indexed_copy_per_work_spec_uses_index_table_descriptor`,
  `test_blackhole_ragged_row_copy_uses_valid_rows_index_table_descriptor`,
  `test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`,
  `test_blackhole_module_direct_call_block_indexed_copy_uses_index_table`,
  `test_blackhole_module_direct_call_block_indexed_copy_rejects_out_of_range_index_table`,
  and
  `test_blackhole_module_direct_call_ragged_row_copy_uses_row_count_predicate`
  reported `7 passed`.
- This checkpoint does not complete T8.  Broader indexed traversal and broader
  ragged token/page forms remain open.

2026-05-05 UTC T8 table-indexed per-work checkpoint:

- `cmake --build build -j32` passed.
- Minimal `BlockIndices[bx]` staged copy now lowers to a table-backed
  `TTPerWorkArgSpec`: `value_source=index_table`,
  `index_buffer=BlockIndices`, `index_value_scale=1`, with the A tile-start
  descriptor pointing back to SpatialPlan `AccessRegion` evidence.
- Device source consumes `runtime_arg_u32("a_tile_start_id")` and no longer
  emits a raw source-time `BufferLoad` from `BlockIndices`.
- Executable materialization registers the index table as a page-indexed DRAM
  input buffer with 4-byte pages, and direct runtime evaluates per-work tile
  starts from host-side table data.
- Direct runtime now rejects out-of-range table tile starts from typed
  materialization page bounds instead of relying on source-side guard recovery.
- Copy structure/runtime selectors passed:
  `test_blackhole_grid_indexed_copy_per_work_specs_expose_typed_descriptors`,
  `test_blackhole_block_indexed_copy_per_work_spec_uses_index_table_descriptor`,
  `test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`,
  `test_blackhole_module_direct_call_block_indexed_copy_uses_index_table`,
  and
  `test_blackhole_module_direct_call_block_indexed_copy_rejects_out_of_range_index_table`
  reported `5 passed`.
- Flash extended-sequence gate
  `test_blackhole_flash_attention_extended_seq_metadata_carries_loop_carried_exact_cb`
  passed for `seq_len=128,256,512`; the corresponding direct-runtime selector
  still reports the typed TT-Sim boundary and skipped for those three shapes.
- Regression selectors
  `test_flash_attention_segment_writer_block_indices_follow_per_work_value_source`
  and `testing/python/target/blackhole/test_blackhole_topk_runtime.py`
  reported `5 passed`.
- This checkpoint did not complete T8.  Segmented/grouped dispatch and ragged
  predicate-derived bounds remained open at that time.

2026-05-05 UTC T8 / larger-flash regression checkpoint:

- `cmake --build build -j32` passed.
- Larger flash shapes no longer fail source/spec admission on constant
  full-tile `AccessRegion` evidence or clear-accum=false accumulator reload:
  rank-aligned `index_exprs` are treated as indexed evidence only when the
  access expression contains an actual index variable, and loop-carried local
  accumulator reload is admitted only from typed loop-carried evidence plus a
  full static local state shape.
- Source codegen now backs `blackhole.acc` stack allocations with a CB only
  when TTProgram projected explicit `initial_reserve_pages`; metadata-only
  CB configs no longer turn local accumulators such as `acc_s` / `acc_o` into
  shared physical CB write pointers.
- The original GQA larger-shape regression
  `testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py::test_flash_attention_segment_writer_block_indices_follow_per_work_value_source`
  passed.
- Flash larger-shape metadata gate
  `testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py::test_blackhole_flash_attention_extended_seq_metadata_carries_loop_carried_exact_cb`
  passed for `seq_len=128,256,512`.
- The corresponding direct-runtime selector for `seq_len=128,256,512` still
  skips at the typed TT-Sim boundary:
  `loop-carried exact-CB backedge direct runtime is gated:
  TT-Sim reports tensix_execute_pacr: count=1`.
- Flash source/lifecycle selector covering accumulator merge, queue
  consistency, PV merge, output staging, and multiphase CB layout reported
  `8 passed`.
- Seq64 bf16 MHA flash direct-runtime correctness
  `test_blackhole_flash_attention_seq64_mha_bf16_forward_direct_runtime`
  passed.
- T8 grid-indexed structure/pipeline/runtime selector reported `9 passed`.
- `git diff --check` passed.

2026-05-05 UTC T8 irregular/indexed first-slice checkpoint:

- `cmake --build build -j32` passed.
- `AccessRegion` now records concrete `BufferLoad` / `BufferStore`
  `index_exprs`, participating loop/launch vars, and guarded/unconditional
  predicate kind for the admitted grid-indexed slice.
- `TTPerWorkArgSpec` / executable per-work descriptors now carry
  `access_region` and `access_region_index` evidence for tile descriptors
  derived from indexed access.
- Structure/pipeline/runtime selector passed:
  `testing/python/transform/test_blackhole_spatial_ir.py::test_algorithmic_access_region_covers_copy_unit_reads_and_writes`,
  `test_t8_spatial_plan_records_grid_indexed_access_exprs`,
  `test_t8_validate_spatial_plan_rejects_slice_region_without_index_exprs`,
  `test_t8_tt_per_work_descriptors_reference_spatial_access_region_evidence`,
  `testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_grid_indexed_copy_per_work_specs_expose_typed_descriptors`,
  `test_blackhole_grid_indexed_copy_build_rejects_top_level_per_work_payload_fallback`,
  `test_blackhole_grid_indexed_copy_rejects_per_work_arg_kind_fallback_without_identity`,
  `testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch`,
  and
  `test_blackhole_module_direct_call_grid_indexed_copy_worker_semaphore_handshake`
  reported `9 passed`.
- This checkpoint does not complete T8.  Segmented/grouped dispatch, ragged
  bounds, and table-indexed block traversal still need source/runtime gates.

2026-05-05 UTC T7.5 exact-CB liveness/allocation completion:

- `cmake --build build -j32` passed.
- Old loop-carried source fallback symbols were removed from active source:
  `loop_carried_live_form_cb_by_buffer_identity_`,
  `loop_carried_live_form_buffer_by_buffer_identity_`, and
  `completed_loop_carried_buffer_identities_` no longer appear under
  `tilelang_repo/src/transform`, `tilelang_repo/src/target`, or tests.
- Structure/runtime selector passed:
  `testing/python/transform/test_blackhole_spatial_ir.py::test_exact_cb_release_source_does_not_keep_local_last_use_fallback`,
  `testing/python/transform/test_blackhole_spatial_ir.py::test_exact_cb_materialization_pop_requires_typed_release_event`,
  `testing/python/transform/test_blackhole_spatial_ir.py::test_validate_tt_program_consumes_exact_cb_lifecycle_records`,
  `testing/python/transform/test_blackhole_spatial_ir.py::test_validate_tt_program_rejects_loop_carried_exact_cb_without_exit_evidence`,
  `testing/python/transform/test_blackhole_spatial_ir.py::test_validate_tt_program_rejects_interfering_exact_cb_intervals_sharing_cb`,
  `testing/python/transform/test_blackhole_spatial_ir.py::test_validate_tt_program_rejects_full_tile_consumer_bound_to_slice_live_form`,
  `testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py::test_blackhole_t7_seq64_mha_bf16_exact_cb_partial_combine_direct_runtime`,
  `test_blackhole_flash_attention_extended_seq_metadata_carries_loop_carried_exact_cb`,
  and
  `test_blackhole_flash_attention_extended_seq_mha_bf16_forward_direct_runtime`
  reported `10 passed, 3 skipped`.
- The physical-CB interference gate first exposed a real positive-path bug:
  exact-CB virtual intervals were using merged requirement lifetime as their
  begin point, so later versions on the same requirement appeared live from
  program point 0.  The interval builder now uses producer/use evidence, and
  `PlanTTCBAlloc` also feeds exact-CB intervals into requirement lifetime
  before physical CB assignment.
- The three skips are the typed simulator boundary for seq128/256/512:
  loop-carried input exact-CB backedge release is admitted in source/spec, but
  current TT-Sim reports `tensix_execute_pacr: count=1` for the compute-side
  pack path.  Seq64 remains a positive direct-runtime correctness gate.
- Materialization `pop_front=true` now requires a typed
  `TTExactCBReleaseEvent`; the old local
  `blackhole_cb_pop_front(cb_value.cb_id, cb_value.num_tiles)` fallback is
  absent from `lower_blackhole_exact_cb.cc`.
- Full-logical-tile consumer bindings now reject
  `thread_distributed_slice` live forms in `ValidateTTProgram`.
- `git diff --check` passed.

2026-05-04 UTC T6/T7 completion verification:

- `cmake --build build -j32` passed.
- Leaf compute runtime regression passed:
  `testing/python/target/blackhole/test_blackhole_leaf_compute_runtime.py`
  reported `17 passed, 1 skipped`.
- T6 topk runtime plus T7 flash exact-CB/partial-combine selectors passed:
  `testing/python/target/blackhole/test_blackhole_topk_runtime.py`,
  `test_blackhole_flash_attention_seq64_bf16_metadata_admits_multi_block_direct_runtime_contract`,
  `test_blackhole_flash_attention_seq64_mha_bf16_forward_direct_runtime`,
  `test_blackhole_flash_attention_seq64_gqa_bf16_forward_direct_runtime`,
  `test_blackhole_t7_seq64_mha_bf16_exact_cb_partial_combine_direct_runtime`,
  `test_flash_attention_seq64_bf16_compute_source_accumulates_clear_accum_false_gemm_via_tiled_merge_protocol`,
  `test_flash_attention_seq64_bf16_compute_source_keeps_cb_events_queue_consistent`,
  and
  `test_flash_attention_seq64_bf16_pv_merge_consumes_scaled_acc_o_live_form`
  reported `11 passed`.
- T7 exact-CB / GEMM regression subset passed:
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_uses_pack_tile_materialization`,
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_t7_exact_cb_gemm_post_merge_cast_consumer_direct_runtime`,
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_without_zero_preclear_keeps_materialization_gate`,
  and
  `testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_direct_runtime_preserves_clear_accum_false_fragment_for_cast_consumer`
  reported `4 passed`.
- `git diff --check` passed.

2026-05-04 UTC entrypoint / main-design documentation cleanup:

- `git diff --check` passed.
- No build or TT-Sim runtime tests were run because this batch only edits
  top-level entrypoint, README, design, progress, and memory documentation.

2026-05-04 UTC documentation cleanup:

- `git diff --check` passed.
- No build or TT-Sim runtime tests were run because this batch only edits
  design/progress/memory documentation.

2026-05-04 UTC T5 K-dimension sharded GEMM direct-runtime correctness:

- `cmake --build tilelang_repo/build -- -j32` passed.
- K-sharded direct-runtime selector passed:
  `test_blackhole_t5_external_k_sharded_l1_gemm_direct_runtime_partial_sum_bf16`
  and
  `test_blackhole_t5_manycore_external_k_sharded_l1_gemm_direct_runtime_partial_sum_bf16`.
- T4/T5 targeted selector passed: `9 passed`.
- GEMM non-direct regression subset passed:
  `46 passed, 1 skipped, 20 deselected`.
- Spatial IR regression passed:
  `pytest -q testing/python/transform/test_blackhole_spatial_ir.py --tb=short`
  reported `104 passed`.
- The many-core K-sharded gate covers
  `M=320`, `N=352`, `K=512`, `logical_grid=11x10x2`,
  110 physical worker cores, 220 logical work items, width-sharded A/B K
  placement, block-sharded C placement, and device-side fp32 partial-C
  reduction before host readback.
