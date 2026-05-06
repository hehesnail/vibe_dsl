# TileLang Blackhole Backend Progress

> 当前 checkout 的执行看板。
> 长期架构合同看 `tasks/dev_design/`。
> 本文件只保留当前状态、active task、后续 gate、最近验证摘要。

## Status

- Date: `2026-05-06`
- Active task: `T9 Workload first paths`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Task | State | Current boundary |
| --- | --- | --- |
| T1 Buffer address ABI | Complete | Runtime consumes typed buffer/address records for interleaved DRAM, staged-copy resident L1 views, and the admitted 64B page-addressed copy path. |
| T2 Leaf compute / GEMM baseline | Complete | Admitted non-flash leaf families and current-placement GEMM run through `BlackholeModule` or fail closed with typed reasons. |
| T3 Tensor/value sharding and explicit reshard | Complete | `T.MemoryConfig`, placement intents, tensor memory-config plans, op sharding contracts, placement resolution, and first `interleaved_to_sharded` staged-copy conversion are typed and projected. |
| T4 External accessor / runtime ABI | Complete | External interleaved DRAM, 64B page-addressed interleaved DRAM, and static sharded-L1 accessors are executable records consumed by source/runtime; unsupported dynamic/common-runtime forms reject from typed records. |
| T5 Sharded GEMM / layout variants | Complete | First static external sharded-L1 GEMM layouts pass direct runtime, including single-core, 2x2 multi-core, 110-core many-core all-bf16, and first K-dimension partial-sum correctness path. |
| T6 `topk` | Runtime complete / cleanup required | Existing-TIR row-wise value/index selection runs through direct runtime for fp32 and bf16 values with exact `int32` indices, without a frontend topk op or selection plan. The backend still uses a limited typed compute-region repeated-reduction source path; final cleanup is generic typed compute-region/reduction lowering. |
| T7 Exact-CB / materialization primitives | Complete | Exact-CB materialization is admitted through typed live-form/materialization/consumer-binding records, including GEMM post-merge `pack_tile`, source-live `cb_republish`, and seq64 bf16 flash-attn exact-CB partial-combine direct runtime correctness. |
| T7.5 Exact-CB liveness / allocation cutover | Complete | Covered exact-CB resident tiles use typed TTProgram/ExecutableSpec lifecycle, allocation, and release records; old loop-carried owner maps, materialization-pop fallback, and full-tile/slice ambiguity are fail-closed or deleted from the active path. |
| T8 Irregular work domains / indexed access | Implementation / cleanup required | Grid-indexed, table-indexed, sparse, ragged, paged, segmented, and T9.1 segmented-A grouped GEMM surfaces execute through direct runtime. Indexed/ragged truth is owned by `AccessRegion` plus typed per-work bindings; public per-work schema no longer carries `index_table_*`, workload-shaped row names, or topk/selection fields. |
| T9 Workload first paths | Implementation | T9.1 pre-grouped MoE/routed grouped GEMM and T9.3 paged MLA decode have bf16 direct-runtime correctness through ordinary TIR-derived indexed/ragged bindings plus typed materialization/lifecycle records. T9.2 paged GQA projection is admitted but the latest full runtime run hits the typed PACR simulator boundary; T9.4-T9.6 remain queued. |
| T10 Distributed production variants | Queued | Mesh, CCL, NoC/multicast/global scheduling, distributed workload correctness, and production partial-K reduction protocol. |

## Active Boundary Notes

- Runtime/codegen must consume `ExecutableSpec` leaf records; no source-name,
  argument-position, accessor-string, or runtime observation recovery.
- Architecture audit `2026-05-06`: completed-task status must distinguish
  runtime coverage from final architecture cleanliness.  T6 now routes its
  value/index selection through
  `CodeGenBlackhole::TryEmitTypedComputeRegionKernel`, consuming typed
  `reduce_tile` compute records, explicit compute-operand
  `cb_requirement_indices`, and generic segment bodies; input/output CBs are
  no longer recovered from requirement names, `<buffer>_reduce_out` suffixes,
  output data format, or value/index channel names.  It is not a frontend
  `topk` op or selection plan, but it is still a limited repeated-reduction
  backend source path and must be cleaned up into generic typed
  compute-region / reduction lowering.
- IR-first audit `2026-05-05`: do not add workload-shaped schema such as
  topk/selection/index-table side channels.  Current T8 cleanup moved sparse
  indexed truth back to `SpatialPlan`: same-subject indexed reads keep
  distinct `AccessRegion.index_exprs`, and per-work binding selects
  the matching region by structural IR expression.
- Per-work runtime values that depend on a dynamic TIR expression use
  `value_source=value_expr`; the serialized TIR expression carries the
  `BufferLoad`s needed by runtime, while launch-axis dependencies are
  normalized into explicit `tl.blackhole.runtime_arg_u32(...)` calls before
  projection.
  `index_buffer`, `index_value_scale`, `index_table_shape`,
  `index_table_index_sources`, and `value_source=index_table` are not public
  TTProgram / ExecutableSpec / runtime schema.
- Per-work runtime values that depend on typed compute/work context, such as
  GEMM K-tile count, N-tile stride, or logical-z K offset, also use
  `value_source=value_expr`.  ABI lowering derives compute constants from
  typed GEMM records and uses explicit logical-block runtime-arg calls for
  launch axes.  Public schema must not grow
  `compute_op_reduction_extent`, `compute_op_output_x_extent`, or
  `logical_block_z_offset` value-source enums.
- Public per-work schema no longer carries binding-kind subroles such as
  `row_start`, `row_count`, `page_index`, or legacy `descriptor_kind`.
  Cross-stage records carry only `arg_kind`, `arg_identity`, `buffer`,
  `value_source`, optional `value_expr`, and optional `AccessRegion` evidence;
  leaf readers interpret those generic values locally.  Runtime and lowering no
  longer classify value-expr bindings by `per_work_value[_N]` arg-name prefixes.
- Guarded T8 access evidence is `AccessRegion` owner truth: guarded regions
  carry concrete boolean `predicate_exprs`, and `ValidateSpatialPlan` rejects
  guarded regions without them.  This is a generic IR invariant, not a
  ragged/segmented/paged schema branch.
- Direct runtime no longer uses `work_linear_id` or table-shape metadata as
  the evaluator for table-backed per-work values.  Old ABI branches that
  rebuilt row-count / row-start bindings from only a table-buffer name were
  removed.
- Remote synchronization endpoints are explicit executable segment records:
  `logical_core_noc_x/y` runtime args bind ABI values, but leaf extraction
  must consume `remote_core_descriptors` and must not reconstruct endpoint
  objects from the arg pair.
- TT lowering no longer keeps pass-local `IndexTableAddressing`,
  `index_buffer`, or `index_value_scale` helper state for per-work value
  binding.  Runtime-arg dedup uses structural `value_expr` equality plus
  `AccessRegion.index_exprs`; compute-segment admission uses the pass-local
  `include_in_compute_segment` bit, not runtime-arg name matching.
- Tile-compute pass-local diagnostics use covering vocabulary
  (`covering_kind`, `covered_patterns`, `covered_pattern:*`) rather than
  `selection_*` or `selected_pattern:*`; this remains a local covering
  diagnostic, not a new semantic plan family.
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

The architecture cleanup is still open: the active code path is a limited
`TryEmitTypedComputeRegionKernel` source path keyed to typed repeated
`reduce_tile` records and explicit operand-to-CB requirement links.  That
historical runtime bring-up artifact must move into a generic typed
compute-region / reduction lowering before T6 is architecturally clean.

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

2026-05-06 UTC page-addressed layout cleanup checkpoint:

- Removed `page_indexed` as an active layout / accessor protocol.  Page
  transport now stays on `layout = interleaved` with positive
  `transport_page_size` and
  `logical_index_mapping = interleaved_page_index`.
- Deleted the leftover value-expr/accessor-backed buffer collector that only
  existed to decide whether to rewrite a buffer distribution to the old
  `page_indexed` layout.
- Runtime materialization admission now distinguishes full-tile host
  tilization from raw sub-tile row/stick pages using static buffer shape,
  dtype, and page size, not a special layout name.
- `cmake --build build -j32` passed.
- Focused projection/schema selectors covering block-indexed, 2D indexed,
  ragged, segmented, and stick page-addressed transport reported `8 passed`.
- Direct-runtime selectors reported the page-addressed stick copy `1 passed`,
  missing-page-metadata typed reject `1 passed`, and T9 page-addressed QK/AV
  flash micro paths `4 passed`.

2026-05-06 UTC IR-first CB/name/payload cleanup checkpoint:

- Audited the same family as `page_value_arg_name`, `row_start`, `row_count`,
  `page_index`, and index-table schema.  Additional active risks were
  CB `requirement_names`, GEMM positional tail payload, and unused
  `GetPayload*` Map helpers.
- Removed public/debug `requirement_names` from `TTCBPlan`,
  `ExecutableSpec.cb_configs`, `BlackholeModule` serialization, projection,
  and target tests.  CB-backed `blackhole.acc` allocations now carry explicit
  `tl.blackhole.cb_requirement_index`; codegen consumes requirement indices
  and executable physical `cb_id`s only.
- Moved Blackhole GEMM compute config from frontend positional tail args to
  explicit TIR call annotations consumed by matmul lowering.  `defines` and
  named compile args remain TT-Metal compile configuration, not a new
  semantic side channel.
- Deleted the unused `spatial_analysis` payload-map helper surface so future
  passes cannot reuse it as a cross-stage semantic bag.
- `cmake --build build -j32` passed.
- Focused source/schema guards reported `5 passed`; TT-Sim structural/target
  selectors covering GEMM config, typed operand bindings, copy attrs, and T6
  CB operand links reported `9 passed`.
- Direct-runtime correctness passed after the cleanup: GEMM richer compute
  config `1 passed`; T6 topk fp32 single-work, fp32 multi-work, and bf16
  values with int32 indices `3 passed`.

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

2026-05-06 UTC direct-runtime value_expr name-recovery cleanup checkpoint:

- Removed direct-runtime `value_expr` evaluation by naked `Var.name_hint`.
  `blackhole_module.cc` no longer recognizes `bx/by/bz`, `num_k_tiles`, or
  `logical_n_tiles` by source name.
- `PlanTTKernelABI` now normalizes block-index variables inside per-work
  `value_expr`s into explicit `tl.blackhole.runtime_arg_u32(...)` calls before
  `ExecutableSpec` projection.  The pass-local block-index source analysis is
  retained until ABI projection, but no new cross-stage side channel or schema
  field is introduced.
- GEMM K-tile counts and N-tile strides are folded from typed GEMM/core-grid
  records into ordinary `uint32` `value_expr` constants; logical-z K offsets
  are expressed as an explicit logical-block-z runtime-arg call times that
  typed K-tile count.
- Segment runtime args for generic per-work values are now attached from the
  segment body's actual `runtime_arg_u32` uses, so writer/compute segments do
  not rely on a segment-kind whitelist and do not miss body-retained dynamic
  values.
- Source guard now rejects `Var.name_hint` and hardcoded work/compute variable
  names in `blackhole_module.cc`; grouped-GEMM projection asserts dynamic
  value expressions contain explicit runtime-arg calls and no non-handle
  `tir.Var` nodes.
- `cmake --build tilelang_repo/build -j32` passed.
- Focused structural/projection selector covering the guard, grouped-GEMM
  per-work binding projection, indexed copy value_expr, and ragged value_expr
  reported `4 passed`.
- Runtime note: after the cleanup, the previous `value_expr` fatal is gone in
  metadata/projection checks.  A minimal grouped-GEMM TT-Sim direct-runtime
  probe reached enqueue and timed out after `180s`, so direct GEMM correctness
  was not used as completion evidence for this cleanup checkpoint.

2026-05-06 UTC remote-core descriptor owner-truth checkpoint:

- `TTProgram -> ExecutableSpec` segment projection now materializes
  `remote_core_descriptors` from typed logical-core NOC ABI records, checking
  x/y pairing, identity, coordinate consistency, and duplicate components
  before leaf extraction.
- `rt_mod_blackhole.cc` no longer derives `KernelSpec.remote_core_descriptors`
  from `logical_core_noc_x/y` runtime args.  It only reads the explicit
  executable segment field; `BlackholeModule` validation rejects logical-core
  runtime args when the descriptor record is missing.
- The source guard now scans the effective `rt_mod_blackhole.cc` entry for
  the removed extraction fallbacks; the earlier duplicate dict-key guard bug
  was fixed so these snippets cannot be masked by a later entry for the same
  file.
- `cmake --build tilelang_repo/build -j32` passed.
- Focused structural selectors covering the source guard, missing-descriptor
  rejection, unpaired logical-core NOC rejection, and descriptor materialized
  projection reported `4 passed`.
- The worker semaphore producer/consumer direct-runtime selector reported
  `1 passed`, proving the explicit descriptor still drives TT-Sim execution.

2026-05-06 UTC transport accessor segment resolver checkpoint:

- `lower_blackhole_transport.cc` no longer registers accessors with hardcoded
  `"fused_dataflow"` literals.  Transport emission resolves the segment via
  `ResolveAccessorSegmentKind(...)`; the string remains an ABI/kernel kind,
  not an emitter-owned schema branch.
- The `DramToDram` transition path keeps segment selection separate from
  accessor direction: the read accessor slot is allocated as `DramToCB`, and
  the write accessor slot as `CBToDram`, so non-fused reader/writer segments
  do not collapse to slot 0.
- The public source guard now rejects direct
  `GetReadAccessorSlot("fused_dataflow"`,
  `GetWriteAccessorSlot("fused_dataflow"`, and
  `RegisterAccessor("fused_dataflow"` snippets in
  `lower_blackhole_transport.cc`.
- `cmake --build tilelang_repo/build -j32` passed.
- Focused selectors covering the source guard, compile-time ABI projection,
  block-indexed and ragged per-work projection, explicit remote-core
  projection, block-indexed runtime, ragged runtime, and page-addressed
  accessor runtime reported `9 passed`.
- Remaining same-family implementation risks are not public-schema fields but
  still need cleanup: the pass-local bound/base/extent value routing state in
  `PlanTTKernelABI`, and the limited T6 typed compute-region repeated-reduction
  source path until generic typed reduce/scan lowering replaces it.

2026-05-06 UTC generic per-work binding analysis checkpoint:

- `PlanTTKernelABI` no longer carries the pass-local
  `needs_*_value_arg_`, `*_table_buffer_name_`,
  `*_shared_buffer_names_`, `bound_value_runtime_arg_*`, or
  `runtime_arg_tile_start_scale_*` state used to classify ragged / segmented /
  paged value bindings.
- The lowering pass now resolves active per-work bindings from the current TIR
  expression (`tl.blackhole.runtime_arg_u32(...)` or the active `Let` var)
  back to the generic `IndexedPerWorkRuntimeArg` record.  Transport decisions
  such as guarded page copies and base+extent materialization are derived from
  whether the current index/predicate expressions use generic per-work values,
  not from workload-shaped side flags.
- Tile-origin normalization is keyed by the generic
  `value_usage=buffer_tile_origin` record instead of separate runtime-arg
  name / var scale maps.
- The public source guard now rejects the removed bound/base/dynamic member
  names in `lower_blackhole_ops.h`, `lower_blackhole_ops.cc`,
  `lower_blackhole_transport.cc`, and `lower_blackhole_abi.cc`.
- `cmake --build tilelang_repo/build -j32` passed.
- Focused selectors covering the source guard; block-indexed, 2D indexed,
  sparse-ragged, ragged-row, two-segment, and paged-valid-row projection; and
  the matching TT-Sim direct-runtime cases reported `15 passed`.
- Remaining same-family implementation risk after this checkpoint is the T6
  limited typed compute-region repeated-reduction source path until generic
  typed reduce/scan lowering replaces it.

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
  protocol names.  The limited typed compute-region repeated-reduction source
  path remains explicit cleanup debt until generic typed compute-region
  lowering replaces it.
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
  documented limited repeated row-reduce backend scan emitter;
  `guard_mask_to_cb` is a generic internal leaf builtin for guard mask
  materialization, replacing the former row-bound public builtin name; and
  `bound/base value table buffer` names are pass-local mechanics that must
  not become public TTProgram / ExecutableSpec schema.  The next schema-risk
  audit target is consumption-side semantic recovery through `arg_kind` /
  value-expr buffer materialization inference, not another workload-shaped
  schema extension.

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

2026-05-05 UTC value-expr materialization owner-truth checkpoint:

- Removed runtime-side value-expr buffer materialization inference from the
  leaf reader.  `PopulateBufferMaterializationSpecs` no longer hardcodes
  a runtime-invented DRAM layout, and page-size selection no longer has a
  `value_expr_buffer_load` fallback.
- `TTBufferDistributionPlan` owns value-expr input-buffer distribution for the
  generic host-side table case: buffers reached by
  `TTPerWorkArgSpec.value_expr` `BufferLoad` nodes, and not already backed by
  TT accessor device access, are projected as interleaved DRAM with explicit
  page size and `logical_index_mapping = interleaved_page_index`.  Runtime
  materialization consumes that explicit plan and fails closed if a referenced
  value-expr buffer has no distribution record.
- The public guard scans `rt_mod_blackhole.cc` for the removed fallback
  strings, and the block-indexed projection test now asserts
  `buffer_distribution_plans["BlockIndices"].layout == "interleaved"` plus
  `logical_index_mapping = interleaved_page_index` so owner truth is tested
  before runtime materialization.
- `cmake --build build -j32` passed.
- Focused TT-Sim selector covering the source guard, block-indexed and 2D
  table value-expr projection, sparse/ragged/segmented/paged value-expr
  projection, and sparse-ragged plus paged-valid-rows direct runtime reported
  `9 passed`.
- Remaining similar schema-risk surfaces from the audit after this checkpoint:
  segment body extraction has since moved out of `rt_mod_blackhole.cc` and
  into the TTProgram / ExecutableSpec segment record path;
  pass-local `BlackholeLoweringSupportFacts` remain separate derived analysis
  and are not being treated as a public schema surface.

2026-05-06 UTC codegen logical-z owner-truth checkpoint:

- Removed the `CodeGenBlackhole` `runtime_arg_vars_by_kind_` map and the
  unused `GetRuntimeArgVarByKind` helper.  Codegen no longer recovers
  `blockIdx.z` by looking up `k_tile_start_id` and `num_k_tiles` runtime-arg
  kinds and dividing them.
- Segmented kernels with `logical_grid_z > 1` now carry an explicit generic
  `logical_block_z` per-work runtime arg whose `TTPerWorkArgSpec` uses
  `value_source=logical_block_z`.  `BindThreadIndex(blockIdx.z)` consumes that
  explicit per-work binding, or falls back only to a generic
  `work_linear_id` binding when present.
- The public source guard now rejects the deleted `runtime_arg_vars_by_kind_`
  k-tile lookup snippets in `codegen_blackhole.cc`.
- `cmake --build build -j32` passed.
- Focused TT-Sim selectors reported the source guard `1 passed` and the
  logical-z/K-sharded plus grouped-GEMM coverage `3 passed`, including the
  manycore K-sharded bf16 partial-sum runtime path.

2026-05-06 UTC exact-CB typed gate checkpoint:

- Removed the `rt_mod_blackhole.cc` `SpecHasRuntimeArgKind` helper and the
  exact-CB admission checks that used `num_k_tiles` runtime-arg presence as a
  proxy for GEMM/workload structure.
- Exact-CB multi-page and multi-block gates now consume existing typed
  `KernelComputeOpSpec` records and require an enabled GEMM compute op before
  applying the GEMM-specific runtime boundary.
- The public source guard now rejects `SpecHasRuntimeArgKind` in
  `rt_mod_blackhole.cc`, and the flash metadata test accepts explicit
  `logical_block_yx_linear` per-work binding sources instead of requiring a
  stale `work_linear_id` projection.
- `cmake --build build -j32` passed.
- Focused TT-Sim selectors covering the source guard, exact-CB flash
  executable metadata gate, multi-work flash per-work metadata, and seq64
  multi-block runtime admission reported `9 passed`.

2026-05-06 UTC per-work tile-origin usage checkpoint:

- Removed `blackhole_module.cc::IsTileStartRuntimeArgKind`; direct runtime no
  longer classifies tile-origin bounds by `a_tile_start_id` /
  `b_tile_start_id` / `output_tile_start_id` string families.
- `TTPerWorkArgSpec` / `ExecutableSpec` now carry generic
  `value_usage=buffer_tile_origin` only for per-work values that are consumed
  as the associated buffer's tile origin.  Row-bound values remain ordinary
  `value_expr` bindings without `value_usage`.
- Direct runtime validates only `buffer_tile_origin` values against the target
  buffer materialization page count, preserving out-of-range table fail-closed
  behavior without misclassifying ragged/paged row bounds.
- `cmake --build build -j32` passed.
- Focused TT-Sim selectors covering source/schema guards, block-indexed
  projection, paged cache-length/page-valid row projection, block-indexed
  direct runtime, out-of-range index-table rejection, 2D indexed runtime,
  ragged row-count runtime, and paged row-bound runtime reported `11 passed`.

2026-05-06 UTC segment body owner-truth checkpoint:

- `TTKernel` now carries an optional segment body TIR field, projected into
  the executable segment plan with a shared segment-key constant.
- `PlanTTKernelABI` derives segment bodies while `blackhole.segment_kind` is
  still pass-local lowering evidence, strips those markers, and records the
  resulting body on each kernel.  Final leaf readers consume that body
  directly.
- Removed `rt_mod_blackhole.cc::SegmentBodyExtractor` and its leaf-time
  `blackhole.segment_kind` / neighbor-inference recovery path.  The public
  source guard now rejects those snippets in `rt_mod_blackhole.cc`.
- Fixed the generic segment-body extractor so unmarked executable leaf
  statements are not copied into every segment body, and made retained
  serial-loop input CB pops explicitly compute-segment statements at their
  generation point.
- `cmake --build build -j32` passed.
- Focused selectors covering leaf/source guards, public schema guard, copy
  executable projection/build/direct-runtime schema, GEMM segment body
  disjointness, and GEMM compile-time ABI projection reported `8 passed`.
- `test_blackhole_gemm_basic` timed out after `300s` in the current TT-Sim
  run; baseline GEMM direct runtime timeout is tracked as a runtime
  verification limitation for this checkpoint rather than proof of the
  segment-body change.

2026-05-06 UTC T6 IR-first codegen cleanup checkpoint:

- Audited the user-called-out schema family beyond `page_value_arg_name`:
  public schema/source guards now cover `topk` / selection surfaces,
  index-table-shaped fields, row/page-shaped workload fields, and stale
  compute-shaped value sources.  The active T6 debt is no longer public
  schema, but the limited typed compute-region repeated-reduction source path
  remains cleanup debt until generic compute-region reduction lowering
  replaces it.
- Removed the row-rank named codegen surface from T6.  The remaining limited
  path now enters through `TryEmitTypedComputeRegionKernel`, consumes typed
  `reduce_tile` compute records, and emits neutral `__tl_reduce_*` /
  `kReduceIterations` source names instead of `__tl_rank_*` / `kRankExtent`.
- Fixed a generic codegen/runtime mismatch exposed by T6: codegen now resolves
  CB operation operands through `ExecutableSpec.cb_configs[*].requirement_indices`
  to the physical `cb_id`, and structural tests reject unresolved requirement
  indices in kernel source.
- Fixed the enqueue hang exposed by TT-Sim: thread-lane codegen now analyzes
  use through the current core's emitted body, so loop-invariant publish bodies
  scalarize without being polluted by compute-local stores that are skipped on
  data-movement cores.  Source guards assert the reader publishes exactly the
  input CB page count instead of serializing the same CB event under the
  `threadIdx.x` lane loop.
- `cmake --build build -j32` passed.
- Focused structural selectors reported `4 passed`: T6 source/projection
  guard, no topk/selection named protocol surface, T8 descriptor fallback
  guard, and public schema field guard.
- TT-Sim direct-runtime selectors passed after the fix:
  T6 bf16 values + int32 indices `1 passed`; T6 fp32 single-work `1 passed`;
  T6 fp32 multi-work `1 passed`.

2026-05-06 UTC T6 compute-region source cleanup checkpoint:

- Tightened the T6 source guard so the Blackhole codegen source/header cannot
  reintroduce the old `TryEmitTypedRowReduceScanKernel` entry or local
  value/index-shaped roles such as `value_reduce`, `index_reduce`,
  `value_cb`, and `index_cb`.
- Moved the remaining limited T6 source path behind
  `CodeGenBlackhole::TryEmitTypedComputeRegionKernel`.  The path now consumes
  typed `reduce_tile` records and names its paired outputs as primary/ordinal
  channels in generated source rather than value/index-specific local roles.
- This is still not final architecture completion: the repeated-reduction
  source path remains a limited backend projection until generic typed
  compute-region / reduction lowering can replace it.
- `cmake --build build -j32` passed.
- Focused structural/source selectors reported `3 passed`: T6 source/spec
  projection, T6 no topk/selection/value-index-local codegen surface, and
  public schema field guard.
- TT-Sim direct-runtime selectors passed after the cleanup:
  T6 fp32 single-work `1 passed`; T6 fp32 multi-work `1 passed`; T6 bf16
  values + int32 indices `1 passed`.

2026-05-06 UTC T6 output-CB name-protocol cleanup checkpoint:

- Added `_reduce_out` to the T6 source guard after auditing the remaining
  user-called-out side-channel family.  The guard now fails on output-CB suffix
  matching in addition to topk/selection and value/index-local codegen names.
- Removed the remaining `output_buffer + "_reduce_out"` lookup from
  `CodeGenBlackhole::TryEmitTypedComputeRegionKernel`.  The limited path now
  reads input CBs by exact typed requirement identity and selects paired output
  CBs from executable `cb_configs` role/data-format channel properties, not
  buffer-name suffixes.
- Removed the remaining generic per-work value name-prefix classifiers:
  lowering now treats any resolved active per-work binding as the semantic
  source, and direct runtime range-shape validation counts `value_source =
  value_expr` bindings instead of checking `per_work_value[_N]` arg kinds.
- This closes the immediate name-protocol leak but not the final architecture
  debt: the limited repeated-reduction source path should still be replaced by
  generic typed compute-region / reduction lowering with explicit
  compute-operand-to-CB links.
- `cmake --build build -j32` passed.
- Focused structural/projection selectors reported `6 passed`: T6 no named
  protocol surface, per-work value-expr/source guard, public schema field guard,
  explicit per-work binding guard, segmented-row value-expr projection, and T9
  grouped-GEMM segmented-A projection.
- TT-Sim direct-runtime selectors reported `3 passed`:
  T6 fp32 single-work, T6 fp32 multi-work, and T6 bf16 values + int32 indices.
- TT-Sim per-work direct-runtime selectors reported `2 passed`: segmented row
  copy start/count tables and T9 grouped GEMM bf16.

2026-05-06 UTC T6 compute-operand CB-link cleanup checkpoint:

- Audited the remaining T6 typed compute-region path after removing
  `_reduce_out` name lookup.  The active residue was not another public
  `topk` / selection schema, but codegen still recovered CBs by requirement
  names and selected the paired output CBs by data-format channel.
- Added explicit `TTComputeOperandBindingPlan.cb_requirement_indices` and
  projected it through `TTProgram -> ExecutableSpec -> BlackholeModule`.
  The field is generic compute-operand-to-CB evidence; it does not encode
  topk, row rank, index table, page value, or value/index-local roles.
- Final TTProgram transport attachment rewrites non-output compute operands to
  the boundary exact-CB allocation requirement indices when an exact resident
  value is present.  Operator-internal CBs remain represented by exact-CB
  lifecycle/allocation records rather than being treated as operand-boundary
  inputs.
- Deleted the T6 codegen helpers that matched CBs by requirement name or output
  data format: `find_cb_by_requirement_name`, `cb_has_exact_requirement_name`,
  `find_unique_output_cb_by_channel`, and `candidate_is_ordinal`.
- Added structural coverage proving T6 reduce operand bindings carry valid
  CB requirement indices and that the primary reduce input resolves to the CB
  the reader actually publishes.  This guards the cb18-vs-cb21 hang where
  compute waited on an internal `reduce_src` CB while the reader published the
  boundary input CB.
- `cmake --build build -j32` passed.
- Focused structural selectors reported `5 passed`: T6 operand-CB link,
  T6 contract projection, T6 no named protocol surface, standalone reduce
  writer consumption, and GEMM typed operand bindings.
- TT-Sim T6 direct-runtime selectors reported `3 passed`: fp32 single-work,
  fp32 multi-work, and bf16 values with exact int32 indices.
- Flash source/spec selectors were rerun and still fail before this final
  operand-CB attachment at the existing exact-CB `acc_o` materialization gate.
  They are not counted as passing evidence for this checkpoint.

2026-05-06 UTC tile-compute covering diagnostic terminology checkpoint:

- Removed `selection_kind`, `selection_status`, `selection_order`,
  `selected_patterns`, `selected_pattern:*`, and `local_dag_dp` from
  tile-compute covering diagnostic output and source evidence.  The FFI
  diagnostics now expose `covering_kind`, `covering_status`, `covering_order`,
  `covered_patterns`, and `covered_pattern:*`.
- Renamed the pass-local C++ field from `selection_kind` to `covering_kind`.
  This does not introduce a new IR layer; the durable outputs remain typed
  `TTComputeOpPlan`, materialization/fanout demands, and validators.
- Retargeted the composite-operation validator test to a small elementwise
  leaf TTProgram so it validates the composite-op reject contract without
  depending on flash exact-CB materialization lowering.
- `cmake --build build -j32` passed.
- Focused tile-compute structural selectors reported `12 passed`.

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
