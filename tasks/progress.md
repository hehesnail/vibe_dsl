# TileLang Blackhole Backend Progress

> 当前 checkout 的执行看板。
> 长期架构合同看 `tasks/dev_design/`。
> 本文件只保留当前状态、active task、后续 gate、最近验证摘要。

## Status

- Date: `2026-05-04`
- Active task: `T6 topk`
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
| T6 `topk` | Active | Standalone value/index selection with typed value output and `int32` index output contracts. |
| T7 Exact-CB / materialization primitives | Queued | Wider source-live-form materialization, exact-CB publish/consume, partial combine, and multi-block flash exact-CB correctness. |
| T8 Irregular work domains / indexed access | Queued | TIR-derived segmented, ragged, and indexed addressing evidence; no workload metadata registry. |
| T9 Workload first paths | Queued | Workload checkpoints decomposed into admitted primitive surfaces with direct-runtime correctness. |
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

## Active Task: T6 `topk`

T6 admits standalone value/index selection as a real Blackhole direct-runtime
path.  Task design:
`tasks/dev_design/2026-05-03-blackhole-t6-topk.md`.

T6 is complete only when:

- value output and `int32` index output are represented in typed IR/source/spec
  contracts;
- standalone row-wise fp32 `topk` values plus exact `int32` indices run through
  direct runtime for a multi-work case such as
  `M=320`, `N=128`, `k=6`, `axis=1`, `blk_m=64`;
- the admitted bf16 value surface plus exact `int32` indices also runs through
  direct runtime with `M > blk_m`;
- unsupported axes, index dtypes, tie behavior, and layout combinations fail
  closed with typed diagnostics;
- no external runner or source-name recovery is introduced.

Next implementation step:
add the typed value/index projection and direct-runtime tests first, then land
the source/runtime implementation against those tests.

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

### T6 `topk`

- fp32 values + exact `int32` indices, multi-work row-wise case:
  compare values and indices with `torch.topk`.
- bf16 values + exact `int32` indices, `M > blk_m`:
  compare values with bf16-appropriate tolerance and indices exactly.
- Inputs should avoid ties until deterministic tie behavior is represented.
  Typed rejects for unsupported forms do not count as positive correctness.

### T7 Exact-CB / Materialization

- bf16 source-live-form materialization where one kernel publishes an
  intermediate value and a later kernel consumes the materialized form.
- At least one admitted exact-CB publish/consume multi-kernel path through
  direct runtime.  Unsupported multi-page events must reject with typed
  reasons, but the admitted subset still needs positive runtime correctness.
- Device-produced partial combine over two or more partial values, compared
  with a host reference.
- bf16 multi-block flash-attn or flash-decode exact-CB case exercising
  producer/consumer lifetime and combine behavior.

### T8 Irregular Work / Indexed Access

- Segmented or grouped dispatch from TIR loop/predicate/address structure with
  non-uniform groups and operands such as `group_sizes` / `group_offsets`.
- Ragged bounds from TIR predicates and operands such as `cache_seqlens`,
  proving invalid rows/tokens are skipped.
- Indexed block traversal where `BufferLoad` / `BufferStore` indices use an
  operand such as `block_indices`.
- In every case, the derived evidence must drive source/runtime addressing.
  Projection-only tests do not complete T8.

### T9 Workload First Paths

Each checkpoint needs its own direct-runtime correctness proof:

- T9.1 pre-grouped MoE / routed grouped GEMM:
  bf16 grouped GEMM with explicit non-uniform token ranges.
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
