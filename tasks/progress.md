# TileLang Blackhole Backend Progress

> Current checkout execution board.
> Durable architecture contracts live in `tasks/dev_design/`.
> This file tracks current state, active boundaries, next tasks, and the
> current verification baseline.  It is not a checkpoint log.

## Status

- Date: `2026-05-18`
- Active lane: `P2 / T10 complete`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Lane | State | Current boundary |
| --- | --- | --- |
| Foundation `T1-T7.5` | Complete | Buffer ABI, leaf compute/GEMM, sharding/materialization, exact-CB lifecycle, and admitted non-workload direct-runtime paths use typed `TTProgram -> ExecutableSpec` records or fail closed. |
| `P0` target execution contract | Complete | Covered execution facts are owned by `TTProgram` typed fields/objects and projected once to `ExecutableSpec`; leaf consumers reject source/body/name recovery. |
| `T8` irregular/indexed access | Complete | Indexed, sparse, ragged, paged, segmented, and grouped-feed paths use generic `AccessRegion` plus `value_expr` evidence. |
| `P1 / T9` workload-first paths | Complete | T9.1-T9.6 are admitted on current bf16 direct-runtime surfaces, including grouped GEMM, paged decode, sparse/ragged attention, chunk scan, and split-block flash decode. |
| `P2 / T10` collectives and reducer protocol | Complete under current local scope | Mesh / multi-device placement is typed through `TTProgram -> ExecutableSpec` and direct runtime currently fails closed for non-unit mesh placements.  Per user scope change on 2026-05-18, T10.1-T10.3 completion is defined as single-card multi-tile bf16 value semantics and local `BlackholeModule` equivalents for all-gather, reduce-scatter, and all-to-all plus typed owner truth and fail-closed unsupported forms.  T10.4 now moves partial-K GEMM reducer ownership into `TTReducerPlan -> ExecutableSpec.reducer_plans`; the direct runtime consumes the typed plan for scratch placement/lifetime, local route, transport choice, accumulation order, and final-writer timing.  Multi-device fabric CCL remains an external blocker, not the active completion gate. |

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
- Mesh placement is explicit `TTMeshPlan` owner truth.  `TTCoreGroup`,
  `TTBufferDistributionPlan`, and `ExecutableSpec.core_plan` must bind the
  selected mesh by name and index plus device range.  Current direct runtime
  admits unit mesh only; non-unit mesh placements fail closed with a typed
  unsupported reason before runtime creates a unit mesh.
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
- Exact-CB live-form planning consumes indexed `SpatialPlan`
  `MaterializationBoundary` evidence.  A boundary's source and target live
  values may have different physical forms; TTProgram live-form plans must use
  the live-form solver decision for the selected side, not a subject-level
  cache or a default CB-materialized form.
- Exact-output live-CB evidence is ordered by lowering program point.  A later
  marker cannot satisfy an earlier consumer, while an explicit
  CB-to-local untilize materialization can establish typed local loop-carried
  state for a following `clear_accum=false` GEMM.
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
  limited to proven simulator capability boundaries, and admitted T9.5/T9.6
  paths publish through typed exact/live CB records rather than PACR skips.

## Next Work Queue

Completed ordering follows the 2026-05-18 scope change: T10.1-T10.3 are
judged by single-card multi-tile collective value semantics plus local
`BlackholeModule` runtime equivalents, and T10.4 is judged by typed
partial-K reducer ownership plus bf16 direct-runtime correctness.  A typed
contract, projection, validator, or fail-closed diagnostic is supporting
evidence, not a standalone completion target.

### External Multi-Device Blocker

- Multi-device fabric CCL remains blocked in the current local TT-Sim
  environment:
  - base `scripts/setup_tt_sim.sh` exposes a `1x1` system mesh;
  - adding
    `TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal_repo/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml`
    lets TT-Sim open a `1x2` Blackhole mesh, and `ttnn.get_num_devices()`
    reports `2`;
  - the TTNN CCL runtime probe with `ttnn.FabricConfig.FABRIC_1D` reaches
    Fabric initialization on both simulated devices before the first
    all-gather step, then fails with
    `UnimplementedFunctionality: eth_txq_regs_wr32: eth_txq_cmd=0x2`;
  - `scripts/probe_tt_sim_ccl.sh` is the reusable probe for this boundary:
    exit `0` means the minimal `1x2` bf16 all-gather, reduce-scatter, and
    all-to-all runtime routes are numerically correct; the current expected
    local result is
    `probe_status=fabric_ccl_unsupported`;
  - a temporary probe against upstream TT-Sim `v1.6.1` on `2026-05-18`
    reaches the same `eth_txq_cmd=0x2` fatal, so the current blocker is not
    removed by swapping from the local `v1.4.x`-era simulator binary to the
    latest published Blackhole TT-Sim release;
  - running the same all-gather smoke without fabric config is not a fallback:
    it fails with `Trying to get un-initialized fabric context`;
  - `TTSIM_SEMIHOSTING=1` does not change that fabric fatal.

  Until that simulator boundary changes, multi-device fabric CCL cannot be
  claimed.  This blocker is out of the current single-card multi-tile T10
  completion scope.  External handoff details are captured in
  `tasks/blockers/2026-05-18-ttsim-ccl-eth-txq.md`.

### Ordered Queue

1. `T10.1` Single-card multi-tile CCL value semantics: all-gather,
   reduce-scatter, and all-to-all have typed owner truth through
   `TTCollectivePlan -> ExecutableSpec.collective_plans`, validators, typed
   admission diagnostics, a bf16 multi-tile host-reference value probe, and
   local `BlackholeModule` multi-tile equivalents.  Status: complete under the
   scoped single-card definition.
2. `T10.2` Scoped local scheduling stance: no remote NoC, multicast fabric
   route, or global cross-device scheduling record is required for the
   single-card completion scope; non-unit mesh and fabric-backed records stay
   fail-closed and the fabric blocker is documented externally.  Status:
   complete under the scoped single-card definition.
3. `T10.3` Broadened scoped coverage: the single-card probe covers all three
   logical collective operation kinds on multi-tile bf16 shapes with
   host-reference comparisons and direct-runtime local equivalents.  Status:
   complete under the scoped single-card definition.
4. `T10.4` Partial-K GEMM reducer protocol: K-sharded GEMM now materializes
   `TTReducerPlan` records and projects them to
   `ExecutableSpec.reducer_plans`.  The admitted local direct-runtime
   protocol owns reducer target buffer, partial-C scratch buffer name,
   placement, lifetime, local same-device route, `device_tile_add`
   transport, ascending producer accumulation order, and producer-0 final
   writer timing.  Missing admitted reducer plans fail closed before direct
   runtime execution.  Status: complete for the current single-card/local
   direct-runtime protocol.

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
  T8/T9 projection selectors, T10.1b `TTCollectivePlan` projection/validator
  guards, and deleted-schema guards.
- Single-card T10 value semantics:
  `scripts/probe_single_card_multitile_ccl_semantics.py` checks bf16
  all-gather, reduce-scatter, and all-to-all over tile-aligned `8x8`
  multi-tile shapes against host references; pytest coverage lives in
  `tilelang_repo/testing/python/transform/test_blackhole_single_card_multitile_ccl_semantics.py`.
- Single-card T10 local runtime:
  `tilelang_repo/testing/python/target/blackhole/test_blackhole_t10_single_card_multitile_ccl_runtime.py`
  runs local multi-tile all-gather / reduce-scatter / all-to-all equivalents
  through `BlackholeModule` on the repository TT-Sim bf16 direct path with
  `8x8x4` logical tile work items per collective.
- T10.4 partial-K reducer runtime:
  `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  checks that K-sharded GEMM projects exactly one `reducer_plans` record for
  both small `2x2x2` and many-core `11x10x2` logical grids, runs both bf16
  direct-runtime paths through TT-Sim against torch references, and verifies
  that deleting the admitted reducer plan produces the typed unsupported
  reason
  `K-sharded GEMM requires an admitted TTReducerPlan partial_k_sum record`.
- Direct-runtime correctness:
  admitted T7/T8/T9 positive paths run through `BlackholeModule` with the
  repository TT-Sim bf16 baseline where tensor values are involved, including
  the small bf16 flash-attn path and the seq64 MHA exact-CB partial-combine
  path.
- Typed unsupported coverage:
  malformed schema, missing page/address metadata, invalid exact-CB lifecycle,
  non-unit mesh placement, and current simulator capability boundaries
  fail closed before source or runtime guessing.

Historical checkpoint logs, exact selector counts, and patch notes belong in
git history and `memory/`, not in this file.
