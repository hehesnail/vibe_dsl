# TileLang Blackhole Backend Progress

> Current checkout execution board.
> Durable architecture contracts live in `tasks/dev_design/`.
> This file tracks current state, active boundaries, next tasks, and the
> current verification baseline.  It is not a checkpoint log.

## Status

- Date: `2026-05-20`
- Active lane: `P2 / T10 complete`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current Board

| Lane | State | Current boundary |
| --- | --- | --- |
| Foundation `T1-T7.5` | Complete | Buffer ABI, leaf compute/GEMM, sharding/materialization, exact-CB lifecycle, and admitted non-workload direct-runtime paths use typed `TTProgram -> ExecutableSpec` records or fail closed. |
| `P0` target execution contract | Complete | Covered execution facts are owned by `TTProgram` typed fields/objects and projected once to `ExecutableSpec`; leaf consumers reject source/body/name recovery. |
| `T8` irregular/indexed access | Complete | Indexed, sparse, ragged, paged, segmented, and grouped-feed paths use generic `AccessRegion` plus `value_expr` evidence. |
| `P1 / T9` workload-first paths | Complete for typed workload contracts and admitted bf16 runtime | T9.1-T9.6 typed/source coverage remains complete and admitted bf16 direct-runtime paths include grouped/page-addressed GEMM-style routes, chunk scan, seq32/64/128/256/512 MHA, GQA, paged GQA, sparse/ragged GQA, paged MLA dual-score/decode, and split-block flash decode.  These admitted flash paths now assert empty `direct_runtime_unsupported_reasons` and run positive TT-Sim value comparisons instead of using PACR/materialization fail-closed gates. |
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
  old T7/T9 `t_tile_mmio_wr32` classifier is gone; PACR gates are not allowed
  for currently admitted bf16 flash-attention paths.  Seq32/64/128/256/512
  MHA, GQA, paged GQA, sparse/ragged GQA, paged MLA dual-score/decode, and
  split-block flash decode all publish through typed exact/live CB records
  and run as positive direct-runtime correctness paths.
- Direct-runtime correctness admission is part of the contract.  Runtime tests
  must not rely on broad relative tolerances to hide wrong values.  The
  current audited unsupported boundaries are non-unit mesh / fabric paths,
  malformed schemas, explicit buffer/access metadata gaps, TT-Sim fp16
  capability gaps, and automatic temporal lowering for a single monolithic
  large-K GEMM whose K extent exceeds the current safe matmul window.  The
  former `tilize_cast_fragment_slice`, compute-only terminal publish, and T9.6
  split-block flash blockers are no longer current admitted-path boundaries.
  Previously audited wrong-value paths for GEMM `transpose_A`, multi-tile
  per-work tile compute, existing-TIR TopK repeated row-reduction, standalone
  leaf compute copy/reduce/typecast-publish, `fragment_fill_cast_publish`, and
  admitted bf16 flash attention now run as positive runtime cases instead of
  publishing unsupported reasons.  Copy/direct transport runtime checks are
  exact-value gates, including page-addressed stick, ragged/segmented/indexed
  copy, worker-semaphore copy, and projected T3 reshard copies.
- Current partial-K GEMM reducer admission supports temporal output waves in
  the single-card direct runtime.  Producer shards run in z order; in-physical
  output waves use the typed `device_tile_add` reducer, and later temporal
  output waves are accumulated by a host-mediated float32 page add from the
  typed final/scratch buffers.  Large logical output grids keep the logical
  work grid separate from the resident sharded L1 grid: the resident grid must
  fit core/L1-bank limits, while larger `shard_shape` values can cover
  multiple logical output tiles per resident shard.  The `13x10x4` and
  `20x20x4` bf16 cases now check these paths through TT-Sim instead of
  publishing a fail-closed temporal reason or returning silent wrong values.
  Shape-general large MNK correctness is not defined by making the logical
  output grid huge.  The admitted large-shape path keeps the logical/core
  grid bounded by available cores, lets each work item cover multiple output
  tiles through core-internal M/N loops, and tiles each K shard internally
  when the full shard is larger than the working CB window.  The verified
  interleaved DRAM output/scratch path covers
  `M=N=512,K=2048,k_shards=4` with a `4x4x4` logical/core grid; each core
  writes `4x4` output tiles and each producer shard runs two `k_tile=256`
  chunks before the runtime reduces full-output scratch pages into final C.

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
   runtime execution.  Partial-K output grids that require multiple temporal
   waves per producer are supported for the current single-card direct runtime:
   the first wave uses the device tile-add reducer and later temporal waves
   use typed final/scratch buffer pages for host-mediated float32 accumulation.
   Larger logical grids are admitted by capping the resident sharded L1 grid
   to the physical core/bank budget, as in the verified `20x20x4` case with
   C resident grid `10x10` and `shard_shape=(64,64)`.
   Larger MNK shapes that cannot keep full C resident in sharded L1 use an
   admitted interleaved DRAM output/scratch reducer while keeping the
   logical/core grid bounded.  The verified core-tiled large-MNK guards cover
   `M=N=512,K=2048,k_shards=4` on a `4x4x4` logical/core grid and
   `M=640,N=704,K=2048,k_shards=4` on a full-core `11x10x4` logical/core
   grid using all `110` Blackhole compute cores; each full-core worker owns
   `2x2` output tiles and each K shard is computed by two internal
   `k_tile=256` GEMM chunks before the direct runtime reduces the full
   scratch C buffer into final C.  These core-tiled large-MNK guards use a
   strict absolute bf16-vs-fp32-reference gate of `atol=0.1,rtol=0.0`;
   the latest full-core run measured max abs diff `0.083786`, mean abs diff
   `0.010080`, p99 abs diff `0.037431`, and p999 abs diff `0.051130`.
   The compute-side CB lifetime now pops input pages per consume instead of
   retaining them across serial loops, so nested core-internal M/N loops
   cannot replay stale A/B tiles.  Core-internal `clear_accum=false` K chunks
   preserve Float32 accumulator live-form CBs and reload the previous partial
   into DST for the final continuation path, so partial C is not silently
   downcast to bf16 before reducer accumulation.  This support surface assumes
   the large-K producer shard has been lowered into explicit `k_tile=256`
   chunks; automatic temporal lowering for a single monolithic `T.gemm` whose
   K extent exceeds the current safe matmul-tile window remains a separate
   support boundary.
   Status: complete for the current single-card/local direct-runtime protocol
   subset.

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
  It also checks temporal-wave runtime correctness: `13x10x4` partial-K
  output grids cover `130` logical output tiles over `110` physical launch
  cores, and `20x20x4` covers `400` logical output tiles with a C resident
  grid of `10x10` / `64x64` shards.  Both compare the full bf16
  direct-runtime result against torch, so later-wave tiles are no longer
  allowed to return wrong values.  It also checks core-tiled large-MNK bf16
  cases with admitted interleaved DRAM output/scratch reducers.  The
  `M=N=512,K=2048,k_shards=4` case keeps the logical/core grid at `4x4x4`,
  covers `16x16` output tiles by assigning `4x4` tiles to each core, tiles
  each K shard as two `k_tile=256` chunks, and compares the full direct
  runtime result against torch.  The full-core
  `M=640,N=704,K=2048,k_shards=4` case uses an `11x10x4` logical/core grid,
  verifies `110` unique physical cores and `110` work packets cover `440`
  logical producer work items, assigns `2x2` output tiles to each core, and
  compares the full direct runtime result against torch with the strict
  absolute gate `atol=0.1,rtol=0.0`; the measured full-core distribution is
  max abs diff `0.083786`, mean abs diff `0.010080`, p99 abs diff
  `0.037431`, and p999 abs diff `0.051130`.  The same selector includes a
  `clear_accum=false` precision regression guard that runs two `k_tile=256`
  chunks for `M=N=64,K=512`, asserts both GEMM ops keep Float32 C tensor and
  CB dtypes, and compares TT-Sim output with the torch bf16/fp32 reference.
- Direct-runtime correctness:
  admitted T7/T8/T9 positive paths run through `BlackholeModule` with the
  repository TT-Sim bf16 baseline where tensor values are involved.  Current
  admitted non-GEMM runtime gates use measured absolute tolerances: T9
  page-addressed QK/AV and seq64 QK GEMM-style routes run with
  `atol=1e-4,rtol=0.0` after measuring max abs diff at or below
  `3.815e-6`; T9 paged MLA dual-score runs with `atol=2e-2,rtol=0.0`
  after measuring max abs diff `0.010296`; T3 admitted tile-compute chains
  and T9.5 chunk scan run with `atol=2e-2,rtol=0.0`; T3 reduce-mix runs
  with `atol=1e-3,rtol=0.0`.  Standalone leaf binary/unary/broadcast
  runtime cases now use `atol=2e-2,rtol=0.0`; standalone bf16 row-reduction
  uses `atol=8e-2,rtol=0.0` after measuring max abs diff `0.0625` against
  the torch bf16 row-sum reference.  Single-card T10 local CCL
  all-gather / reduce-scatter / all-to-all now use exact `atol=0,rtol=0`
  comparisons.  TopK value/index selection and copy-runtime selectors now also
  use exact comparisons; no Blackhole runtime correctness test keeps a
  non-zero relative tolerance.
- Flash-attention runtime correctness:
  `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  covers the admitted bf16 flash family through `BlackholeModule`: small
  seq32 H1, MHA H4 seq32/64/128/256/512, GQA seq64, exact-CB partial combine,
  T9 paged GQA, sparse/ragged GQA, paged MLA dual-score/decode, and
  split-block decode.  Admitted paths assert empty
  `direct_runtime_unsupported_reasons` and compare against torch/host
  references with absolute-only gates.  Latest measured precision for the
  main flash sweep was: seq32 H1 max/mean/p99
  `0.011719/0.001298/0.005859`, MHA seq32
  `0.015625/0.001396/0.007812`, MHA seq64
  `0.015625/0.001132/0.005859`, MHA seq128
  `0.019531/0.000990/0.003906`, MHA seq256
  `0.027344/0.000817/0.003906`, MHA seq512
  `0.013672/0.000740/0.002930`, GQA seq64
  `0.011719/0.001212/0.005859`, T9 paged GQA
  `0.015625/0.001124/0.005859`, sparse/ragged GQA
  `0.011719/0.001273/0.005859`, paged MLA dual-score
  `0.010296/0.001575/0.007178`, paged MLA decode
  `0.015625/0.001191/0.005859`, and split-block decode
  `0.009766/0.000993/0.005859`.
- Typed unsupported coverage:
  malformed schema, missing page/address metadata, invalid exact-CB lifecycle,
  non-unit mesh placement, and current simulator capability boundaries
  fail closed before source or runtime guessing.  Current runtime-correctness
  audit coverage no longer treats `tilize_cast_fragment_slice` PACR or T9.6
  split-block materialization as admitted bf16 flash boundaries; those paths
  are covered by positive TT-Sim runtime checks, and the stale
  `tilize_cast_fragment_slice` PACR admission gate has a source guard against
  reintroduction.  These paths run along with GEMM
  `transpose_A`, T3 multi-tile per-work tile compute, TopK repeated
  row-reduction value/index selection, and standalone leaf compute.

Historical checkpoint logs, exact selector counts, and patch notes belong in
git history and `memory/`, not in this file.
