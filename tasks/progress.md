# TileLang Blackhole Backend Progress

> Current checkout execution board.
> Durable architecture contracts live in `tasks/dev_design/`.
> This file tracks current state, active boundaries, next tasks, and the
> current verification baseline.  It is not a checkpoint log.

## Status

- Date: `2026-05-07`
- Active lane: `P0 TTProgram target execution contract hardening`
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
| P0 TTProgram target execution contract hardening | In progress | Parent architecture task: make `TTProgram` the stable target-facing execution contract and delete runtime/codegen semantic recovery.  P0.1 CB queue event ownership is complete.  P0.2 completed slices: codegen runtime buffer address binding consumes projected `ExecutableSpec.runtime_args[].buffer`; host launch association consumes explicit `tl.launched_kernel_symbols`; runtime materialization/admission shape checks consume projected `ExecutableSpec.tensor_memory_config_plans[*].logical_shape`; reduction-region codegen consumes typed `TTComputeOpPlan` reduction facts instead of scanning final bodies. |
| T8 Irregular work domains / indexed access | Complete | Indexed, sparse, ragged, paged, segmented, and T9.1 grouped-GEMM feed paths execute through generic `AccessRegion` + `value_expr` bindings.  Buffer-bound per-work specs carry explicit `AccessRegion` evidence, indexed lookups fail closed on missing structural matches, and broadened segmented/paged/ragged/indexed copy shapes pass direct-runtime gates. |
| T9 Workload first paths | In progress | T9.1 pre-grouped MoE/routed GEMM, T9.2 full paged GQA decode, T9.3 dual-score MLA GEMM, T9.3 full paged MLA decode, T9.4 sparse/ragged GQA decode, and T9.5 chunk recurrence / scan have bf16 direct-runtime correctness.  T9.6 is queued. |
| T10 Distributed production variants | Queued | Mesh placement, CCL, NoC/multicast/global scheduling, distributed workload correctness, and production partial-K reduction remain future TT target-realization work. |

## Current Protocol Snapshot

- Runtime/codegen consume `ExecutableSpec` records.  They must not recover
  semantics from source names, argument positions, generated source, runtime
  observation, or neighboring builtins.
- Public per-work schema is generic:
  `arg_kind`, `arg_identity`, `buffer`, `value_source`, optional
  `value_expr`, and optional `value_usage`.  Buffer-bound indexed/guarded
  bindings must carry explicit `AccessRegion` evidence.  Workload-shaped
  fields such as `index_table_*`, `row_start`,
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
  events projected at the `TTProgram -> ExecutableSpec` boundary, and the
  executable queue gate replays those records rather than parsing generated
  source text or rescanning segment-body TIR.
- T8 value-expression bindings suppress fused-dataflow default tile-origin
  runtime args through projected `TTPerWorkArgSpec` evidence and
  non-synthesized arg kinds, not by classifying runtime arg identities such as
  `per_work_value*`.
- T8 buffer-bound per-work bindings are valid only when they reference explicit
  SpatialPlan `AccessRegion` evidence.  ABI lowering may select evidence from
  current TIR-derived structural indices, but consumers must not reattach it
  from names, arg kinds, helper state, or first same-buffer fallbacks.
- Current simulator gates must also be typed by `ExecutableSpec` facts.  The
  old T7/T9 `t_tile_mmio_wr32` classifier is gone; remaining PACR gates are
  limited to proven simulator capability boundaries such as compute-only
  terminal publish and loop-carried input exact-CB backedge publish.

## Next Work Queue

### P0: TTProgram Target Execution Contract Hardening

- P0.1 CB queue event ownership:
  complete; projected `KernelSpec.queue_events` replaced runtime body/source
  recovery.
- P0.2 remaining semantic recovery audit:
  in progress.  Completed slices: codegen runtime buffer address binding now
  consumes projected `ExecutableSpec.runtime_args[].buffer` records and no
  longer scans final TIR bodies for `tl.blackhole.*_to_cb` / `*_from_cb`
  handle recovery; host launch association now consumes explicit
  `tl.launched_kernel_symbols` attr instead of scanning packed host TIR
  `tvm_call_packed` callees; runtime buffer materialization and
  multidimensional per-work descriptor admission now consume projected
  `ExecutableSpec.tensor_memory_config_plans[*].logical_shape` records instead
  of scanning device `PrimFunc` bodies or buffer maps for static buffer facts;
  reduction-region codegen now consumes typed `TTComputeOpPlan` /
  `ExecutableSpec.compute_ops` reduction records instead of scanning final TIR
  bodies for kind, dimension, or repeat extent.
  Continue inspecting runtime/codegen/executable readers for execution facts
  recovered from source text, final body scans, runtime-arg names/positions,
  workload branches, helper maps, or fallback defaults; promote real owner
  truth into `TTProgram` / `ExecutableSpec` or delete stale paths.
- P0.3 execution event / admission spine:
  centralize shared ordering/admission facts as typed `TTProgram` owner truth
  and project them once for leaf consumers.

### P1: T9 Workload-First Paths

- T9.6 multi-block flash decode:
  admit bf16 split blocks with exact-CB publish/consume and partial combine.

### P2: T10 Distributed Production

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

Current active baseline:

- Compile: `cmake --build build -j32`.
- Protocol/source guards:
  typed tile-CB queue verifier, TTProgram execution-contract source guards,
  completed T8 indexed/ragged/paged/segmented projection selectors, T9
  workload projection selectors, and deleted-schema guards.
- Direct-runtime correctness:
  active admitted T7/T8/T9 positive paths run through `BlackholeModule` with
  the repository TT-Sim bf16 baseline where tensor values are involved.
- Typed unsupported coverage:
  malformed schema, missing page/address metadata, invalid exact-CB lifecycle,
  and current simulator capability boundaries fail closed before source or
  runtime guessing.
- Current known simulator boundary:
  the first T9.5 three-chunk recurrence slice is admitted through typed
  ping-pong state CBs plus a separate writer publication CB.  Broader dynamic
  or extended loop-carried exact-CB recurrence remains future work and must
  still fail closed until it has equivalent typed lifecycle/runtime evidence.
- Current known non-T8 source/spec failure:
  the full copy-pipeline suite still reaches the flash bridge granularity gate
  at `merge_fragment_tiles` destination `acc_o`; that is a T9 exact-CB /
  materialization follow-up, not part of the completed T8 copy cleanup gates.

Historical checkpoint logs, exact selector counts, and patch notes belong in
git history and `memory/`, not in this file.
