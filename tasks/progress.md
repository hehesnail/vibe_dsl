# TileLang Blackhole Backend Progress

> Current checkout execution board.
> Durable architecture contracts live in `tasks/dev_design/`.
> This file tracks current state, active boundaries, next tasks, and the
> current verification baseline.  It is not a checkpoint log.

## Status

- Date: `2026-05-08`
- Active lane: `P1 T9.6 multi-block flash decode`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Lane | State | Current boundary |
| --- | --- | --- |
| Foundation `T1-T7.5` | Complete | Buffer ABI, leaf compute/GEMM, sharding/materialization, exact-CB lifecycle, and admitted non-workload direct-runtime paths use typed `TTProgram -> ExecutableSpec` records or fail closed. |
| `P0` target execution contract | Complete | Covered execution facts are owned by `TTProgram` typed fields/objects and projected once to `ExecutableSpec`; leaf consumers reject source/body/name recovery. |
| `T8` irregular/indexed access | Complete | Indexed, sparse, ragged, paged, segmented, and grouped-feed paths use generic `AccessRegion` plus `value_expr` evidence. |
| `P1 / T9` workload-first paths | In progress | T9.1-T9.5 are admitted on current bf16 direct-runtime surfaces.  Active boundary is T9.6 multi-block flash decode. |
| `P2 / T10` distributed production | Queued | Mesh placement, CCL, NoC/multicast/global scheduling, and production partial-K reduction remain future TT target-realization work. |

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
  those records and must not scan final TIR or infer segment membership from
  builtin neighborhoods.  `blackhole.segment_kind` is not an active lowering
  protocol; active lowering source is guarded against reintroducing it.
- Remote synchronization endpoints are explicit `TTRemoteCoreDescriptorSpec`
  / `KernelSpec.remote_core_descriptors` records.  `logical_core_noc_x/y`
  runtime args bind ABI values and must reference a matching descriptor; they
  are not endpoint owner truth and projection must not reconstruct descriptors
  from runtime-arg pairs.
- Any future fuse-like behavior must be expressed as a generic pass over IR
  constraints and typed records.  Do not add workload-specific fused-op
  schema or per-case lowering branches.
- Exact-CB and physical CB queue correctness are admission checks, not
  workload skips.  `ValidateTTProgram` owns latest exact-CB producer,
  release-reason, storage-format, page-size, and unique CB-requirement-owner
  checks.  `TTKernel.queue_events` is the TTProgram-owned queue-event contract;
  `KernelSpec.queue_events` carries the structured physical projection at the
  `TTProgram -> ExecutableSpec` boundary, and the executable queue gate replays
  those records rather than parsing generated source text or rescanning
  segment-body TIR.
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

### Active

- T9.6 multi-block flash decode:
  admit bf16 split blocks with exact-CB publish/consume and partial combine.

### Queued

- Add typed mesh / multi-device placement before distributed runtime movement.
- Add CCL contracts for all-gather, reduce-scatter, and all-to-all.
- Add NoC / multicast / global scheduling records for remote routes,
  semaphores, and producer/consumer timing.
- Replace the current K-sharded GEMM blocking z-wave tile-add path with a
  typed production reducer protocol: reducer ownership, partial-C scratch
  placement and lifetime, semaphore IDs, remote NOC routes, transport choice,
  accumulation order, and final writer timing.

### Standing Guardrails

- Do not reopen retired body/source/name recovery surfaces.
- New Blackhole execution facts must enter the explicit
  `TTProgram -> ExecutableSpec` contract or fail closed.
- Workload labels are witnesses only; generic IR evidence remains the owner
  truth.

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

Current baseline:

- Compile: `cmake --build build -j32`.
- Protocol/source guards:
  typed tile-CB queue verifier, TTProgram execution-contract source guards,
  T8/T9 projection selectors, and deleted-schema guards.
- Direct-runtime correctness:
  admitted T7/T8/T9 positive paths run through `BlackholeModule` with the
  repository TT-Sim bf16 baseline where tensor values are involved.
- Typed unsupported coverage:
  malformed schema, missing page/address metadata, invalid exact-CB lifecycle,
  and current simulator capability boundaries fail closed before source or
  runtime guessing.

Historical checkpoint logs, exact selector counts, and patch notes belong in
git history and `memory/`, not in this file.
