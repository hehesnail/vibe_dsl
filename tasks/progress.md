# TileLang Blackhole Backend Progress

> Current checkout execution board.
> Durable architecture contracts live in `tasks/dev_design/`.
> This file tracks current state, active boundaries, next tasks, and the
> current verification baseline.  It is not a checkpoint log.

## Status

- Date: `2026-05-07`
- Active lane: `T9 Workload first paths`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Task | State | Current boundary |
| --- | --- | --- |
| T1 Buffer address ABI | Complete | Runtime consumes typed interleaved DRAM, page-addressed interleaved DRAM, and staged-copy resident L1 records. |
| T2 Leaf compute / GEMM baseline | Complete | Admitted non-flash leaf families and current-placement GEMM run through `BlackholeModule` or fail closed with typed reasons. |
| T3 Tensor/value sharding and explicit reshard | Complete | `T.MemoryConfig`, placement intents, tensor memory-config plans, op sharding contracts, placement resolution, and first `interleaved_to_sharded` conversion are typed and projected. |
| T4 External accessor / runtime ABI | Complete | External `interleaved_accessor_cta` and `sharded_accessor_cta` records cover interleaved DRAM, page-addressed interleaved DRAM, and static sharded L1. |
| T5 Sharded GEMM / layout variants | Complete | Static external sharded-L1 GEMM is correct for single-core, 2x2, 110-core many-core, all-bf16, and current K-sharded partial-sum paths. |
| T6 `topk` | Complete | Existing TIR value/index selection runs through direct runtime for fp32 and bf16 values with exact `int32` indices.  The old limited typed compute-region emitter is deleted; codegen consumes executable reduction records through a `reduce_dim`-parameterized channel lowering and CB requirement mappings without topk/selection schema or raw host-pointer fallback. |
| T7 Exact-CB / materialization primitives | Complete | Exact-CB materialization, publication, consumer binding, GEMM post-merge materialization, and seq64 bf16 flash-attn exact-CB partial combine pass `BlackholeModule` TT-Sim correctness. |
| T7.5 Exact-CB liveness / allocation cutover | Complete | Covered exact-CB resident tiles use typed lifecycle, allocation, release events, latest-producer validation, storage-format validation, and fail-closed loop-carried/full-tile gates. |
| T8 Irregular work domains / indexed access | Runtime surface admitted / cleanup pending | Indexed, sparse, ragged, paged, segmented, and T9.1 grouped-GEMM feed paths execute through generic `AccessRegion` + `value_expr` bindings.  The fused-dataflow ABI no longer classifies runtime args by `per_work_value*` identity prefixes; remaining work is broader shape coverage and continued removal of consumption-side recovery. |
| T9 Workload first paths | In progress | T9.1 pre-grouped MoE/routed GEMM, T9.2 full paged GQA decode, T9.3 dual-score MLA GEMM, T9.3 full paged MLA decode, and T9.4 sparse/ragged GQA decode have bf16 direct-runtime correctness.  T9.5-T9.6 are queued. |
| T10 Distributed production variants | Queued | Mesh placement, CCL, NoC/multicast/global scheduling, distributed workload correctness, and production partial-K reduction remain future TT target-realization work. |

## Current Protocol Snapshot

- Runtime/codegen consume `ExecutableSpec` records.  They must not recover
  semantics from source names, argument positions, generated source, runtime
  observation, or neighboring builtins.
- Public per-work schema is generic:
  `arg_kind`, `arg_identity`, `buffer`, `value_source`, optional
  `value_expr`, optional `value_usage`, and optional `AccessRegion`
  evidence.  Workload-shaped fields such as `index_table_*`, `row_start`,
  `row_count`, `page_index`, `descriptor_kind`, or topk/selection fields are
  not schema.
- Dynamic table/work-context values use `value_source=value_expr`.  Launch
  axes are normalized to explicit `tl.blackhole.runtime_arg_u32(...)` calls
  before projection; direct runtime does not interpret naked `Var.name_hint`.
- Buffer distribution layout is physical: `interleaved` or `sharded`.
  Page-addressed DRAM transport is ordinary interleaved DRAM with positive
  page-size fields and
  `logical_index_mapping = interleaved_linear_page`.
  There is no `page_indexed` layout, `page_indexed_accessor_cta`, or public
  page-index subrole.
- `transport_page_size` is a leaf transport/materialization byte-size record.
  Owner truth for buffer addressability remains the typed buffer distribution
  and CB plan; it must not become a second semantic source.
- CB identity crosses the `TTProgram -> ExecutableSpec` boundary by numeric
  requirement indices and executable physical `cb_id`s.  `requirement_names`
  and CB-name suffixes are not protocol.
- Segment bodies are projected records.  Final leaf readers must consume
  those records and must not scan `blackhole.segment_kind` or infer segment
  membership from the final function body.
- Remote synchronization endpoints are explicit descriptor records.
  `logical_core_noc_x/y` runtime args bind ABI values, not endpoint owner
  truth.
- Any future fuse-like behavior must be expressed as a generic pass over IR
  constraints and typed records.  Do not add workload-specific fused-op
  schema or per-case lowering branches.
- Exact-CB and physical CB queue correctness are admission checks, not
  workload skips.  `ValidateTTProgram` owns latest exact-CB producer,
  release-reason, storage-format, page-size, and unique CB-requirement-owner
  checks.  `KernelSpec.queue_events` now carries structured physical CB queue
  events, and the executable queue gate replays those records rather than
  parsing generated source text.
- T8 value-expression bindings suppress fused-dataflow default tile-origin
  runtime args through projected `TTPerWorkArgSpec` evidence and
  non-synthesized arg kinds, not by classifying runtime arg identities such as
  `per_work_value*`.
- Current simulator gates must also be typed by `ExecutableSpec` facts.  The
  old T7/T9 `t_tile_mmio_wr32` classifier is gone; remaining PACR gates are
  limited to proven simulator capability boundaries such as compute-only
  terminal publish and loop-carried input exact-CB backedge publish.

## Next Work Queue

### P1: T8 Cleanup And Runtime Breadth

- Broaden indexed / ragged / segmented / paged shapes only when the evidence
  comes from TIR access expressions, predicates, loop domains, and
  `AccessRegion` records.
- Audit consumption-side code for remaining semantic recovery through
  `arg_kind`, runtime-arg names, transport helper state, or value-expression
  materialization inference.
- Keep page-addressed transport generic: builtin page IDs and local variables
  are allowed, but no public row/page/index-table subroles may re-enter.
- Projection-only tests do not complete T8 extensions; each admitted positive
  form needs a `BlackholeModule` TT-Sim correctness gate.

### P2: T9 Workload-First Paths

- T9.2 paged GQA decode:
  keep the admitted source/spec and direct-runtime correctness on generic
  page-table and ragged `value_expr` bindings; broaden shapes only through
  TIR-derived indexed/ragged evidence.
- T9.3 paged MLA decode:
  keep the admitted page-table/ragged bindings, retained latent-KV lifetime,
  dual-score GEMM correctness, and full paged MLA decode direct-runtime
  correctness.  Broader MLA variants must keep the additive score chain
  generic and typed rather than adding a workload-specific side path.
- T9.5 chunk recurrence / scan:
  represent multi-chunk loop-carried state and device state-buffer lifetime
  through typed lifecycle/allocation records before runtime execution.
- T9.6 multi-block flash decode:
  admit bf16 split blocks with exact-CB publish/consume and partial combine.

### P3: T10 Distributed Production

- Add typed mesh / multi-device placement before distributed runtime movement.
- Add CCL contracts for all-gather, reduce-scatter, and all-to-all.
- Add NoC / multicast / global scheduling records for remote routes,
  semaphores, and producer/consumer timing.
- Replace the current K-sharded GEMM blocking z-wave tile-add path with a
  typed production reducer protocol: reducer ownership, partial-C scratch
  placement and lifetime, semaphore IDs, remote NOC routes, transport choice,
  accumulation order, and final writer timing.

## Verification Baseline

Every active implementation task uses these gates:

| Level | Requirement |
| --- | --- |
| Compile | `cmake --build build -j32` succeeds. |
| Structure | TIR / `SpatialPlan` / `TTProgram` / `ExecutableSpec` tests prove required typed records exist and deleted schema stays absent. |
| Source/spec | Materialized executable schema contains the records consumed by source/runtime. |
| Direct runtime | Admitted positive paths run through `BlackholeModule`, not an external runner. |
| TT-Sim correctness | Runtime correctness uses the repository TT-Sim setup and bf16 baseline when tensor values are involved. |
| Unsupported reason | Unsupported forms fail closed with typed diagnostics before source/runtime guessing. |

Current verified baseline:

- T8 consumer-side cleanup: `cmake --build build -j32` passed; source/spec
  selectors covering deleted prefix classification, explicit value-expression
  suppression, indexed/ragged/paged/segmented projection, and structured
  2-tile output-CB queue order reported `14 passed`; TT-Sim direct-runtime
  selectors covering indexed, sparse, ragged, segmented, and paged copies
  reported `11 passed`.
- Latest schema cleanup: `cmake --build build -j32` passed; focused selectors
  covering the source guard, block-indexed projection, stick page-addressed
  projection, direct runtime, and missing-page-metadata typed reject reported
  `5 passed`.
- Page-addressed cleanup baseline: source/schema selectors covering
  block-indexed, 2D indexed, ragged, segmented, and stick page-addressed
  transport reported `8 passed`; direct-runtime selectors covered stick copy,
  missing-page-metadata typed reject, and T9 page-addressed QK/AV flash micro
  paths.
- T6 completion baseline: `cmake --build build -j32` passed; focused source /
  schema selectors covering typed compute records, CB operand links, deleted
  limited emitter, row-specific codegen guard, and raw-pointer absence
  reported `4 passed`; direct-runtime TT-Sim gates pass for fp32 single-work,
  fp32 multi-work, and bf16 values with exact `int32` indices.
- Typed tile-CB verifier baseline: `cmake --build build -j32` passed;
  `test_blackhole_typed_tile_cb_queue_verifier.py` reported `9 passed`,
  covering duplicate CB-requirement ownership, exact-CB data-format mismatch,
  invalid release reasons, structured `KernelSpec.queue_events`, invalid
  constant physical queue events, and writer output-CB wait/pop balance.
- Leaf simulator-gate cleanup: the unused multi-block exact-CB republish
  runtime gate was deleted, and the old standalone fill/typecast PACR branch
  was replaced by a generic compute-only terminal-publish simulator gate.
  The focused typecast-publish selector reported `1 passed, 1 skipped`.
- T7/T9 current runtime baseline: focused TT-Sim runtime selectors reported
  `16 passed`, covering T3 sharded elementwise/reduce mix, T3 serialized
  reshard and page-addressed copy, T7 seq64 MHA exact-CB partial combine,
  T9 page-addressed QK page1, T9 page-addressed AV page1, T9.2 full paged
  GQA decode, T9.3 dual-score MLA GEMM, T9.3 full paged MLA decode, and
  T9.1 grouped GEMM.  Extended seq still carries the narrower loop-carried
  exact-CB PACR typed simulator reason.
- T9.4 sparse/ragged GQA baseline: source/spec projection selector covering
  `BlockIndices` sparse-block bindings, `ValidRows` per-entry ragged bounds,
  raw-table-load absence, and compute-compatible K/V CBs reported `1 passed`;
  bf16 TT-Sim direct-runtime correctness selector reported `1 passed`.

Detailed historical checkpoint logs belong in git history and `memory/`, not
in this file.
