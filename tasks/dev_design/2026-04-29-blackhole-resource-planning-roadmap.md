# Blackhole Resource Planning Roadmap

## Role

This document defines the resource-planning direction after the algorithmic
generalization and tile-compute covering review.
It is not an overall design document and not a status log.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Scope

Resource planning belongs to `TTProgram` and `ExecutableSpec` admission.
It must not be pushed into `TileComputeDAG`,
source hooks,
runtime fallback,
or diagnostic-only reports.

Allowed owner-truth surfaces:

- `TTHardwareModel`
- core group plans
- buffer distribution plans
- CB plans
- semaphore / sync / transport plans
- `TTResourceDemand`
- `TTResourcePressureReport`
- executable admission records

## Non-Goals

- No new IR layer.
- No persisted `TileComputeDAG` payload.
- No physical address allocator; TT-Metal owns allocation.
- No NoC / multicast / global scheduling before typed core and buffer
  placement exist.
- No direct-runtime-specific patch surface.

## Inputs

Resource planning may consume:

- validated `SpatialPlan`
- validated `TTProgram` plans
- `TTHardwareModel`
- explicit tile-compute fanout / materialization demand
- buffer distribution requirements
- communication edges when represented explicitly

It may not consume:

- buffer names as semantic roles
- source text
- runtime fallback observations as owner truth
- diagnostic DAG dumps as planning input

## Direction 1: Keep `TileComputeDAG` Constrained

`TileComputeDAG`
may feed resource demand only through explicit leaf-graph facts:

- fanout
- share vs materialize decisions
- unsupported materialization reasons

It cannot allocate resources or place work.
If a DAG decision does not change typed plans, resource demand,
validators, or typed diagnostics, it is debug infrastructure.

## Direction 2: Resource Demand And Pressure Reports

`TTResourceDemand`
records what the program needs.
`TTResourcePressureReport`
records whether the target admits it.

Required report categories:

- tile-compute fanout and materialization demand
- CB requirements and CB ID pressure
- CB L1 bytes
- allocator-managed L1 buffer bytes
- max simultaneous L1 pressure
- semaphore pressure
- core-grid requirement
- buffer-distribution requirement
- typed unsupported reasons

Reports must drive validators or typed admission diagnostics.
A dump-only report is not completion.

## Direction 3: CB And L1 Admission

CB planning should use live intervals and target hardware facts:

- target CB count
- reserved / conventional CB classes
- page size and page count
- data format and flow class
- publish / consume event shape
- initial reserve semantics

L1 planning should be pressure admission first:

- CB bytes per core
- allocator-managed L1 buffer bytes
- worker L1 budget
- alignment waste estimate
- max simultaneous pressure

This layer rejects unsafe pressure.
It does not assign physical L1 addresses.

CB sizing is op and live-form dependent.
`page_size`
comes from the data movement or compute protocol for one page / tile of the
value being published or consumed.
`num_pages`
comes from the protocol depth:
double buffering,
exact tiled publication,
producer / consumer event shape,
or materialization lifetime.
The durable separation is:

- op lowering / materialization logic creates `CBRequirement`
- `TTCBPlan` owns the typed CB page count, page size, flow class, and
  publish / consume event shape
- `TTResourceDemand` and `TTResourcePressureReport` aggregate CB count and
  CB-backed L1 bytes for admission

Buffer placement and address ABI must consume these typed CB plans.
They must not rederive CB depth from buffer names, source hooks, or runtime
reader defaults.

## Direction 4: Core And Buffer Placement

Core placement and buffer placement are separate responsibilities.
Core groups must remain hardware-model backed;
the current active expansion point is buffer distribution.

Core placement must consume:

- available worker grid
- worker count under harvesting
- worker L1 size
- DRAM view count when relevant

Initial policy should be conservative:

- logical coordinates
- deterministic row-major work packets
- rectangular core ranges when useful
- do not materialize physical cores outside the hardware worker grid
- allow logical work items to exceed physical worker count only through
  explicit deterministic work packets on admitted workers
- typed reject when requested physical workers or coordinates exceed
  `TTHardwareModel`

The frontend kernel grid and TT worker grid are different axes:

- `T.Kernel(grid_x, grid_y)` describes logical work items.
- `TTCoreGroup.physical_cores` describes resident Blackhole worker cores.
- `TTCoreGroup.work_packets` describes the temporal mapping from logical
  work ids to each resident worker.
- L1 / CB scratch belongs to the resident worker and is reused across the
  worker's temporal packet.

Therefore a logical grid larger than the physical core count is legal only
when the executable address ABI can derive per-work source / destination
regions from
`work_offset`,
`work_count`,
`logical_grid_x`,
and the linearization policy.
It is not legal to allocate one independent L1 / CB instance per logical
block when those blocks exceed resident workers.

`TTBufferDistributionPlan`
must grow beyond
`unit_mesh`
/
`replicated`
to represent:

- interleaved DRAM
- interleaved L1
- height / width / block sharded L1
- host-visible vs device-local placement
- page size
- shard shape
- attached core range

The conservative baseline is:

- DRAM buffers with ABI interleaved layout become interleaved DRAM
  placement.
- Shared or CB-backed L1 buffers become attached-core sharded L1
  placement.
- Per-worker local L1 state stays device-local replicated placement until
  the buffer address ABI can express a stronger per-work indexed contract.
- CB-backed buffers use CB page facts for placement and are not counted again
  as allocator-managed L1 buffers.
- L1 logical byte sizes are admitted against aligned hardware budget; the
  validator should not require every logical local byte size to already be an
  allocation-aligned page.

The sharded plan marks the L1-side working view / materialization, not the
DRAM global tensor itself.
The buffer address ABI must keep the resident core grid separate from the
per-core tensor shard shape used by TT-Metal / TTNN sharded memory configs.
It uses explicit
`shard_grid_shape`,
`sharding_strategy`,
per-core `shard_shape`,
source buffer / source region binding,
logical-index to core-local address mapping,
and DRAM-source region to L1-shard copy/address mapping before sharded
placement is admitted for runtime emission.

The first typed form of that split is:

- `shard_grid_shape`: the attached physical core-grid shape for the resident
  L1 view.
- `sharding_strategy`: height / width / block strategy for assigning the
  logical shard to that grid.
- `shard_orientation`: row-major / column-major traversal of the shard grid,
  matching TT-Metal `ShardOrientation`.
- `shard_shape`: the real per-core tensor data shape of the resident L1 view.
- `source_buffer`: the DRAM / global buffer that materializes the L1 view when
  the view is copied from global memory.
- `source_region_kind`: the address-region class, initially
  `per_work_tile` for per-work TileLang shared tiles.
- `source_region_shape`: the data shape copied from the source buffer into
  one resident L1 shard for one logical work item.
- `logical_index_mapping`: how a logical work id is mapped before addressing,
  initially `work_packet_row_major`.
- `core_local_address_mapping`: how the logical shard is addressed inside the
  resident worker-local view, initially `l1_shard_linear`.

Validators must reject sharded L1 plans that omit placement and address
mapping fields.
Validators must also reject strategy / orientation conflation:
`block`,
`height`,
and `width`
are sharding strategies, while
`row_major`
and `col_major`
are shard orientations.
Source-region binding fields are all-or-none:
if a sharded L1 view is materialized from a DRAM / global source, it must carry
`source_buffer`,
`source_region_kind`,
and `source_region_shape`;
pure worker-local sharded scratch must not fabricate a source binding.
Direct runtime may still reject sharded/page-indexed forms, but the reject must
come from these typed fields rather than source or runtime reconstruction.

For TileLang programs written in a GPU style,
`alloc_shared((tile_m, tile_n))`
is treated as the per-worker, per-work-item L1 / CB-backed scratch footprint.
On Blackhole this footprint may be much smaller than the available worker L1.
That is a performance opportunity, not a correctness license to mutate the
buffer shape during placement.
The legal first step is to preserve the frontend shape, validate it against
L1 / CB capacity, and surface an underutilization diagnostic if useful.
Using more L1 requires an explicit TT-specific retile or work-coarsening plan
that changes the logical work mapping and source-region / address mapping
together.

## Direction 5: Workload Admission Before Distributed Expansion

Only after typed core and buffer placement exist should the backend expand
wider workload / runtime admission.

The workload backlog is not one flat stage.
Each family must name two things separately:

- a first single-device path that can be proven by current
  `TTProgram` / `ExecutableSpec` contracts plus typed resource admission
- a production distributed path that waits for mesh, sharding, CCL,
  NoC, multicast, or global scheduling support

| Family | Repo evidence | First single-device path | Later production requirements |
| --- | --- | --- | --- |
| Non-flash leaf compute and GEMM baseline | Current Blackhole copy / GEMM tests plus TT-Metal leaf API surface | Standalone unary / binary / broadcast / reduce / pack / typecast leaf families and current-placement GEMM layout variants with direct correctness gates; existing stick / page-shaped copy tests remain baseline coverage, not a new top-level task | Sharded layouts, wider multi-core placement, and non-replicated buffer distributions |
| K-sharded GEMM partial reduction | T5 K-dimension sharded GEMM direct-runtime tests and TT-Metal semaphore / NoC primitives | Current direct-runtime path with `logical_grid_z`, per-K-shard waves, partial-C scratch, and runtime-issued device tile-add reduction | Production single-launch or fused-launch reducer protocol with typed reducer ownership, partial-C scratch placement, semaphore ids, remote NOC routes, transport choice, accumulation order, and final writer timing |
| External accessor / runtime ABI | Existing executable accessor kinds `sharded_accessor_cta` and `page_indexed_accessor_cta` | Admit or precisely reject external sharded/page-indexed runtime/codegen accessors from executable records | Wider direct runtime coverage for sharded tensors, paged tensors, and page-table driven workloads |
| `topk` / selection / indexing | `tilelang_repo/examples/topk/example_topk.py` | Single-device `reduce_max` selection with `int32` index outputs and correctness checks | Use as MoE gating input and routing metadata producer |
| MoE / `fusedmoe` | `tilelang_repo/examples/fusedmoe/example_fusedmoe_tilelang.py`; `tt_metal_repo/models/demos/deepseek_v3/tt/moe.py`; `tt_metal_repo/models/demos/deepseek_v3/tt/experts.py` | Shared expert plus pre-packed routed grouped GEMM where `group_sizes`, `group_offsets`, `group_padded_offsets`, `group_idx_for_bx`, routed weights, and token grouping are explicit inputs; no claim of full MoE until gating and combine are admitted | In-pipeline topk, token packing, scatter / gather / combine, expert sharding, all-to-all dispatch / combine, reduce-scatter, all-gather, mesh memory placement |
| Paged attention / paged decode / MLA decode | `tilelang_repo/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py`; `tilelang_repo/examples/deepseek_mla/example_mla_decode_paged.py`; `tt_metal_repo/models/tt_transformers/tt/attention.py` | Single-device page-table or block-table indexed KV read with `cache_seqlens`, optional fixed `num_split`, explicit partial-output / logsum combine, and exact intermediate lifetime | Paged KV cache update / fill, sharded KV cache, multi-device all-gather / all-reduce, split reduction scheduling, NoC-aware max-cores-per-head policy |
| Grouped / ragged / sparse attention | `tilelang_repo/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_varlen_indice.py`; sparse GQA paged examples | Single-device `block_indices` / ragged `cache_seqlens` sparse read plus partial combine | Sharded sparse blocks, cross-core load balancing, distributed sparse scheduling |
| Chunk recurrence / scan | `tilelang_repo/examples/linear_attention/example_mamba_chunk_scan.py`; `tilelang_repo/examples/gdn/example_chunk_o.py`; `tilelang_repo/examples/kda/chunk_o.py` | Single-device chunk state, loop-carried state, and local state-buffer lifetime admission | Cross-core chunk scheduling, distributed state placement, communication-aware state handoff |
| Multi-block flash-attn / flash decode | Current flash-attn Blackhole tests; `tt_metal_repo/tech_reports/FlashAttention/FlashDecode.md` | Existing small and 32x32 bf16 direct runtime path; next is multi-block exact-CB publish / consume correctness | Multi-core split reduction, semaphore-backed remote writes, NoC traffic control, distributed decode scaling |

This means T9 workload bring-up is not blocked on all of T10.
It is blocked on the specific earlier primitives each first path actually
needs, such as T3 sharding / reshard, T4 external accessors, T7
materialization, or T8 irregular work-domain / indexed-access descriptors.
Those primitives must be pulled forward explicitly instead of hidden under a
generic "later runtime" bucket.

## Execution Order

The active board is `tasks/progress.md`.
This roadmap keeps the stable order and exit criteria.

| Task | Goal | Depends On | Exit Criteria |
| --- | --- | --- | --- |
| T1 Buffer address ABI execution integration | Make sharded L1 and page-indexed address ABI real execution contracts, not metadata-only records. | Hardware-backed buffer placement and typed sharded fields. | Complete: sharded L1 staged-copy path materializes from typed plans into source/spec/direct-runtime and passes TT-Sim bf16 correctness; page-indexed 64B page path materializes from typed page metadata and passes direct-runtime correctness; non-admitted 32B bf16 sub-tile page transport and external sharded/page-indexed accessor kinds fail closed from typed fields. |
| T2 Leaf compute / GEMM baseline | Admit non-flash leaf compute and current-placement GEMM layout baseline. | T1. | Unary / binary / broadcast / reduce / pack / typecast, current-placement GEMM variants, and admitted layout movement have direct correctness gates or typed rejects. |
| T3 Tensor/value sharding and explicit reshard | Make TTNN-style user placement intent, op placement contracts, placement conflict handling, and reshard plans first-class in the IR chain. | T2 baseline. | DSL placement intent, tensor memory-config plans, op contracts, conflict rejects, reshard plans, and executable projection are typed and tested. |
| T4 External accessor / runtime ABI expansion | Admit or precisely reject external `sharded_accessor_cta` and `page_indexed_accessor_cta` runtime/codegen forms. | T1 address ABI and T3 placement/conversion projection. | External sharded/page-indexed accessors have direct TT-Metal ABI records and runtime/codegen admission, or fail from explicit executable accessor records. |
| T5 Sharded GEMM / layout variants | Admit GEMM/layout variants that depend on real tensor sharding, including explicit retile/work-coarsening when a layout changes logical work mapping. | T3, and T4 when external sharded/page-indexed accessors are required. | Sharded GEMM/layout correctness where admitted; typed rejects for unsupported placement/conversion/retile combinations. |
| T6 Selection / index base | Bring up standalone `topk` with `int32` index outputs. | T2 leaf reductions. | Value and index correctness are proven; compile-only is not enough. |
| T7 Exact-CB / materialization primitives | Repair wider exact-CB publish/consume, partial combine, source-live-form materialization, and multi-block flash-attn / flash-decode exact-CB correctness. | T1 and relevant T3 materialization rules when sharded values are involved. | Multi-kernel intermediate correctness is covered and missing materialization protocol fails before source/runtime emission. |
| T8 Irregular work domains / indexed access | Represent segmented/ragged ranges, indexed block traversal, and grouped dispatch as explicit work-domain and access descriptors derived from IR operands. | T1 and relevant per-work descriptors. | Missing or inconsistent irregular-domain evidence is rejected before source/runtime emission; no workload-specific metadata registry. |
| T9 Workload first paths | Bring up pre-grouped MoE, sparse/ragged attention, paged GQA decode, paged MLA decode, chunk recurrence, and multi-block flash decode first paths. | Prior tasks as needed by each workload. | Each workload has a stated first path, correctness proof, and typed rejects for unadmitted forms. |
| T10 Production distributed variants | Add mesh/sharding/CCL/NoC/multicast/global scheduling support, including production K-sharded GEMM partial-reduce protocol. | Stable first paths and typed distributed plans, including T3 sharding/reshard. | Distributed paths have typed placement, communication, admission, and correctness gates. |

Do not advance workload admission that depends on external sharded or
page-indexed accessors past T4 until those accessor forms have
source/spec/direct-runtime admission or precise typed rejects.

## Planning Order

1. Keep explicit leaf tile-compute and DAG covering boundaries clean.
2. Use typed resource demand / pressure reports as admission surfaces.
3. Keep CB / L1 admission hardware-backed.
4. Keep core groups hardware-model-backed and replace unit buffer placement
   with explicit buffer distribution.
5. Follow `tasks/progress.md` for first-path workload admission order.
6. Pull forward only the typed primitives required by the current first path.
7. Defer production distributed variants until mesh / sharding / CCL /
   NoC / multicast / global scheduling plans are typed and validated.
8. Do not upgrade K-sharded GEMM from blocking z-wave device reduction to a
   production single-launch or fused-launch partial reduce until reducer
   ownership, partial scratch placement, semaphore ids, remote NOC routes,
   transport choice, accumulation order, and final writer timing are explicit
   `TTProgram` / `ExecutableSpec` records.
9. After that production protocol is admitted, delete or fold the current
   runtime-issued partial-K tile-add reduction path into the typed protocol
   implementation.  It must not remain as a second special-case execution path.

## Completion Criteria

This roadmap is implemented only when:

- resource truth is represented in typed `TTProgram` / `ExecutableSpec`
  records
- validators consume the reports and fail closed
- CB / L1 checks use hardware-model facts
- core groups are derived from hardware facts, not hard-coded constants,
  and validators reject out-of-grid physical cores
- buffer distribution can express the placement choices needed by admitted
  workloads
- workload admission records distinguish first single-device paths from
  production distributed paths
- no resource truth is carried by bags, payloads, source hooks, or runtime
  fallback
