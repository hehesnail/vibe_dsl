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
| T6 `topk` | Runtime complete / cleanup required | Existing-TIR row-wise value/index selection runs through direct runtime for fp32 and bf16 values with exact `int32` indices, without a frontend topk op or selection plan. The backend still uses a dedicated row-rank scan emitter; final cleanup is generic typed compute-region/reduction lowering. |
| T7 Exact-CB / materialization primitives | Complete | Exact-CB materialization is admitted through typed live-form/materialization/consumer-binding records, including GEMM post-merge `pack_tile`, source-live `cb_republish`, and seq64 bf16 flash-attn exact-CB partial-combine direct runtime correctness. |
| T7.5 Exact-CB liveness / allocation cutover | Complete | Covered exact-CB resident tiles use typed TTProgram/ExecutableSpec lifecycle, allocation, and release records; old loop-carried owner maps, materialization-pop fallback, and full-tile/slice ambiguity are fail-closed or deleted from the active path. |
| T8 Irregular work domains / indexed access | Implementation / cleanup required | Grid-indexed, table-indexed, sparse, ragged, paged, segmented, and T9.1 segmented-A grouped GEMM surfaces execute through direct runtime. Indexed/ragged truth is owned by `AccessRegion` plus typed per-work bindings; public per-work schema no longer carries `index_table_*`, workload-shaped row names, or topk/selection fields. |
| T9 Workload first paths | Implementation | T9.1 pre-grouped MoE/routed grouped GEMM and T9.3 paged MLA decode have bf16 direct-runtime correctness through ordinary TIR-derived indexed/ragged bindings plus typed materialization/lifecycle records. T9.2 paged GQA projection is admitted but the latest full runtime run hits the typed PACR simulator boundary; T9.4-T9.6 remain queued. |
| T10 Distributed production variants | Queued | Mesh, CCL, NoC/multicast/global scheduling, distributed workload correctness, and production partial-K reduction protocol. |

## Active Boundary Notes

- Runtime/codegen must consume `ExecutableSpec` leaf records; no source-name,
  argument-position, accessor-string, or runtime observation recovery.
- Architecture audit `2026-05-05`: completed-task status must distinguish
  runtime coverage from final architecture cleanliness.  T6 still has a
  dedicated `CodeGenBlackhole::TryEmitRowRankReductionScanKernel` path for
  value/index row selection.  It is not a frontend `topk` op or selection
  plan, but it is still a case-shaped backend emitter and must be cleaned up
  into a generic typed compute-region/reduction lowering.
- IR-first audit `2026-05-05`: do not add workload-shaped schema such as
  topk/selection/index-table side channels.  Current T8 cleanup moved sparse
  indexed truth back to `SpatialPlan`: same-subject indexed reads keep
  distinct `AccessRegion.index_exprs`, and per-work binding selects
  the matching region by structural IR expression.
- Per-work runtime values that depend on a dynamic TIR expression use
  `value_source=value_expr`; the serialized TIR expression carries the
  `BufferLoad` and launch-axis variables needed by runtime.
  `index_buffer`, `index_value_scale`, `index_table_shape`,
  `index_table_index_sources`, and `value_source=index_table` are not public
  TTProgram / ExecutableSpec / runtime schema.
- Per-work runtime values that depend on typed compute/work context, such as
  GEMM K-tile count, N-tile stride, or logical-z K offset, also use
  `value_source=value_expr`.  Public schema must not grow
  `compute_op_reduction_extent`, `compute_op_output_x_extent`, or
  `logical_block_z_offset` value-source enums.
- Public per-work schema no longer carries binding-kind subroles such as
  `row_start`, `row_count`, `page_index`, or legacy `descriptor_kind`.
  Cross-stage records carry only `arg_kind`, `arg_identity`, `buffer`,
  `value_source`, optional `value_expr`, and optional `AccessRegion` evidence;
  leaf readers interpret those generic values locally.
- Guarded T8 access evidence is `AccessRegion` owner truth: guarded regions
  carry concrete boolean `predicate_exprs`, and `ValidateSpatialPlan` rejects
  guarded regions without them.  This is a generic IR invariant, not a
  ragged/segmented/paged schema branch.
- Direct runtime no longer uses `work_linear_id` or table-shape metadata as
  the evaluator for table-backed per-work values.  Old ABI branches that
  rebuilt row-count / row-start bindings from only a table-buffer name were
  removed.
- TT lowering no longer keeps pass-local `IndexTableAddressing`,
  `index_buffer`, or `index_value_scale` helper state for per-work value
  binding.  Runtime-arg dedup uses structural `value_expr` equality plus
  `AccessRegion.index_exprs`; compute-segment admission uses the pass-local
  `include_in_compute_segment` bit, not runtime-arg name matching.
- `T.Kernel` describes logical work items.  Tensor sharding comes from
  explicit placement intent and resolved memory-config plans.
- T5 K-sharded GEMM currently proves correctness with blocking logical-z waves
  plus a runtime-issued device tile-add reduction.  T10.5 owns replacing that
  path with typed production reducer records.
- For T6-T10, validators and projection tests are support evidence only.
  An admitted positive path must execute through `BlackholeModule` under the
  repository TT-Sim setup and compare device output against a host reference.
- Larger flash-attn shapes exposed the exact-CB resident
  lifecycle/resource-allocation boundary that T7.5 cut over for the covered
  surface.  Covered exact-CB paths must use TTProgram lifecycle/allocation
  records for physical CB choice and release events.

## Runtime-Complete Task: T6 `topk`

T6 admits standalone value/index selection as a real Blackhole direct-runtime
path.  Task design:
`tasks/dev_design/2026-05-03-blackhole-t6-topk.md`.

The admitted frontend shape remains ordinary Tile TIR: `T.copy`, `T.fill`,
`T.reduce_max`, `T.if_then_else`, local value/index buffers, and explicit
global stores.  No `T.topk`, `tl.blackhole.topk`, `TTSelectionPlan`,
`selection_plans`, external runner, source-name recovery, or raw compute-side
host pointer path is part of the contract.

The architecture cleanup is still open: the active code path is a dedicated
`TryEmitRowRankReductionScanKernel` emitter keyed to value/index row
selection records and generated output-CB naming.  That historical runtime
bring-up artifact must move into a generic typed compute-region / reduction
lowering before T6 is architecturally clean.

Unsupported axis/layout/generalized value-index variants remain outside the
admitted T6 subset and must fail closed through typed legality diagnostics.

## Completed Task: T7 Exact-CB / Materialization

T7 admits exact-CB materialization as a typed backend contract instead of a
source-name or runtime fallback surface.

The completed subset covers `TTLiveFormPlan`, `TTMaterializationPlan`, and
`TTConsumerBindingPlan` records for source live forms, materialized CB tile
forms, publication protocol, and consumer binding; GEMM post-merge
cast-consumer exact-CB `cb_republish` / `pack_tile` correctness; seq64 bf16
flash-attn MHA direct runtime where typed exact-CB partial combine compares
against host reference; and typed reject boundaries for unsupported
materialization/event forms before runtime execution.

No frontend materialization op, alternate runtime combiner, mailbox fallback,
legacy payload, or source-name semantic recovery is part of the T7 contract.

## Completed Task: T7.5 Exact-CB Liveness / Allocation Cutover

T7.5 moved the covered flash exact-CB resident surface from emitter-local
lifetime repair to typed TTProgram / ExecutableSpec lifecycle records.

The completed subset covers exact-CB virtual values, use events, live
intervals, physical CB allocation, and release events for the covered
loop-carried flash surface; source rendering of `cb_wait_front`,
`cb_push_back`, and `cb_pop_front` through lifecycle/release records; validator
rejects for missing loop-carried exit evidence, overlapping exact-CB intervals
sharing a physical CB, and full-logical-tile consumers bound to
`thread_distributed_slice` live forms; seq64 bf16 flash-attn exact-CB
partial-combine direct runtime correctness; and seq128/256/512 source/spec
admission that skips only on the typed TT-Sim `tensix_execute_pacr: count=1`
capability boundary.

T8 owns deriving irregular/ragged/indexed work domains from TIR and making
that evidence drive source/runtime addressing.  T9/T10 own workload-first and
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
  contiguous scaled-block, sparse multi-entry, and sparse+ragged cases.
- Broader ragged token/page forms beyond the admitted one-dimensional
  predicate-bound, per-entry sparse predicate-bound, copy-shaped paged cache
  length, and per-page predicate-bound slices.
- Broader segmented/grouped workload use beyond the admitted single- and
  two-range row-segment copy and T9.1 grouped-GEMM feed.
- In every case, the derived evidence must drive source/runtime addressing.
  Projection-only tests do not complete T8.

### T9 Workload First Paths

Each checkpoint needs its own direct-runtime correctness proof:

- T9.2 paged GQA decode:
  keep page-table / cache-length binding projection on the generic
  `value_expr` path and track the current full-runtime
  `tensix_execute_pacr: intermediate_format=0 late_from_format=5` simulator
  boundary separately from schema cleanup.
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

2026-05-05 UTC compute-shaped per-work value-source cleanup checkpoint:

- Removed public `TTPerWorkArgSpec.value_source` enums for
  `compute_op_reduction_extent`, `compute_op_output_x_extent`, and
  `logical_block_z_offset`.  GEMM tile-count/stride/K-start bindings now use
  ordinary `value_source=value_expr` records whose expressions are evaluated
  under the direct-runtime work/typed-compute context.
- Public-schema guard now rejects these compute-shaped value-source strings
  alongside the earlier index-table, row/page, and selection/topk fields.
- Source audit found no remaining hits for the removed compute-shaped
  value-source constants or strings under `tilelang_repo/src`.
- `cmake --build build -j32` passed.
- Focused structural/projection selector covering public schema guard,
  T9.1 grouped-GEMM binding projection, and flash executable-spec
  projection reported `4 passed`.
- Follow-up fixed the direct-runtime buffer-role admission gate for
  compute-context-only `value_expr`s: expressions such as `num_k_tiles` and
  `logical_n_tiles` no longer have to reference a buffer to be valid, while
  `BufferLoad`-using value expressions still contribute their referenced
  formal input buffers.
- Fresh TT-Sim selectors reported T9.1 grouped-GEMM projection/runtime and
  baseline external-sharded GEMM direct runtime `3 passed`; the public schema
  guard plus flash executable-spec projection reported `3 passed`.

2026-05-05 UTC IR-first per-work schema cleanup checkpoint:

- Public per-work schema no longer exports `index_table`, `index_buffer`,
  `index_value_scale`, `index_table_shape`, `index_table_index_sources`,
  `valid_rows`, `ragged_page_index`, `segment_row_start`,
  `segment_row_count`, or selection-plan fields.
- Dynamic table-derived per-work values use `value_source=value_expr`;
  runtime discovers required input buffers from serialized TIR `BufferLoad`
  nodes.  Bound/base/page-axis values are ordinary `per_work_value*`
  bindings, not row/page/tagged public schema.
- The former pass-local `IndexTableAddressing` / `index_buffer` /
  `index_value_scale` helper path in TT lowering was deleted; there is no
  second table-addressing evaluator behind the public schema.  The remaining
  compute-segment bound-value routing is a pass-local flag, not a runtime-arg
  name predicate.
- Public per-work runtime arg identities for dynamic base/bound/axis values
  use generic `per_work_value`, `per_work_value_1`, ... identities.  Workload
  or row/page shaped identities are not schema.
- T6 emitted source symbols and markers no longer expose topk/selection
  protocol names.  The dedicated row-rank backend scan remains explicit
  cleanup debt until generic typed compute-region lowering replaces it.
- `cmake --build build -j32` passed.
- Focused structural/projection selector covering schema whitelist,
  value-expression bindings, indexed/ragged/segmented/paged projection,
  grouped-GEMM bindings, and paged GQA/MLA binding projection reported
  `18 passed`.
- TT-Sim runtime selectors reported T8 indexed/ragged/segmented/paged copy
  `12 passed`, the tile-start out-of-range rejection selector `1 passed`,
  T9.3 paged MLA selectors `2 passed`, and extended flash
  `3 passed, 3 skipped`.
- The latest T9.2 paged GQA direct-runtime selector reached TT-Sim and failed
  at the typed PACR simulator boundary
  `tensix_execute_pacr: intermediate_format=0 late_from_format=5`; this is
  tracked as remaining T9.2 runtime work, not as a row/page schema fallback.
- `git diff --check` passed, and source audit found no public schema/source
  hits for removed `index_table*` constants, helpers, or value sources.
- Follow-up guard cleanup removed stale test assertions that still treated
  `descriptor_kind`, `row_start`, `row_count`, and `page_index` as acceptable
  per-work descriptor vocabulary.  The focused schema/projection guard
  selector now reports `6 passed`.
- Follow-up side-channel audit removed unused `companion_base.h`
  `selection_targets` / `selection_pairs` manifest keys and stale
  `buffer_*_contracts` schema constants.  The guard now scans that header
  with the Blackhole lowering-support files so these deleted side-channel
  names cannot silently re-enter.
- Remaining similar-looking code is classified separately: T6 still has the
  documented dedicated row-rank backend scan emitter; `guard_mask_to_cb`
  is a generic internal leaf builtin for guard mask materialization, replacing
  the former row-bound public builtin name; and `bound/base value table
  buffer` names are pass-local mechanics that must not become public
  TTProgram / ExecutableSpec schema.  The next schema-risk audit target is
  consumption-side semantic recovery through `arg_kind` / value-expr buffer
  materialization inference, not another workload-shaped schema extension.

2026-05-05 UTC guard-mask leaf cleanup checkpoint:

- Renamed the public/internal `row_bound_mask_to_cb` builtin surface to the
  generic `guard_mask_to_cb` leaf.  Arguments are now `bound_value` and
  `base_value`; old `valid_rows` / `page_base` wording is absent from the
  audited source path.
- Renamed the corresponding local matcher/transport mechanics from
  `RowBoundMask*` / `row_bound*` / `row_page_base*` to guard/base-page terms.
  This is still a leaf materialization primitive, not a paged/ragged schema
  branch.
- `cmake --build build -j32` passed.
- Focused TT-Sim selector covering public schema guard, paged value-expr
  projection, guard-mask flash projection, paged-valid-rows runtime, and
  sparse-ragged runtime reported `5 passed`.

2026-05-05 UTC T9.2 paged GQA decode checkpoint:

- The paged GQA frontend remains ordinary Tile TIR: page-table loads drive K/V
  cache page selection, cache-length loads drive guarded page-local row
  validity, and the attention update reuses the existing flash
  partial-combine sequence.
- The current full T9.2 bf16 direct-runtime selector reaches the admitted
  paged GQA source/spec path but hits the TT-Sim PACR capability boundary
  `tensix_execute_pacr: intermediate_format=0 late_from_format=5` before host
  comparison.
- Larger flash shape coverage for `seq_len=128,256,512` reported
  `3 passed, 3 skipped`; the skips are the typed TT-Sim
  `tensix_execute_pacr: count=1` capability boundary, not an untyped backend
  fallback.

2026-05-05 UTC T7.5 exact-CB lifecycle checkpoint:

- `cmake --build build -j32` passed.
- Focused lifecycle/source/spec/runtime selectors covering exact-CB allocation,
  release events, full-tile consumer rejects, seq64 direct runtime, and
  seq128/256/512 source/spec admission reported `10 passed, 3 skipped`.
- The three skips are the typed simulator boundary for seq128/256/512; seq64
  remains the positive direct-runtime correctness gate.
