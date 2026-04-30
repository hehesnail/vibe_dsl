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

## Direction 5: Workload Slices Before Full Runtime Expansion

Only after typed core and buffer placement exist should the backend expand
wider workload / runtime admission.

The workload backlog is not one flat stage.
Each family must be split into:

- a first admitted single-device subset that can be proven by current
  `TTProgram` / `ExecutableSpec` contracts plus typed resource admission
- a production distributed subset that waits for mesh, sharding, CCL,
  NoC, multicast, or global scheduling support

| Family | Repo evidence | First admitted subset | Later production requirements |
| --- | --- | --- | --- |
| Non-flash leaf compute and GEMM variants | Current Blackhole copy / GEMM tests plus TT-Metal leaf API surface | Standalone unary / binary / broadcast / reduce / pack / typecast leaf families and GEMM layout variants with direct correctness gates; existing stick / page-shaped copy tests remain baseline coverage, not a new top-level task | Sharded layouts, wider multi-core placement, and non-replicated buffer distributions |
| `topk` / selection / indexing | `tilelang_repo/examples/topk/example_topk.py` | Single-device `reduce_max` selection with `int32` index outputs and correctness checks | Use as MoE gating input and routing metadata producer |
| MoE / `fusedmoe` | `tilelang_repo/examples/fusedmoe/example_fusedmoe_tilelang.py`; `tt_metal_repo/models/demos/deepseek_v3/tt/moe.py`; `tt_metal_repo/models/demos/deepseek_v3/tt/experts.py` | Shared expert plus pre-packed routed grouped GEMM where `group_sizes`, `group_offsets`, `group_padded_offsets`, `group_idx_for_bx`, routed weights, and token grouping are explicit inputs; no claim of full MoE until gating and combine are admitted | In-pipeline topk, token packing, scatter / gather / combine, expert sharding, all-to-all dispatch / combine, reduce-scatter, all-gather, mesh memory placement |
| Paged attention / paged decode / MLA decode | `tilelang_repo/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py`; `tilelang_repo/examples/deepseek_mla/example_mla_decode_paged.py`; `tt_metal_repo/models/tt_transformers/tt/attention.py` | Single-device page-table or block-table indexed KV read with `cache_seqlens`, optional fixed `num_split`, explicit partial-output / logsum combine, and exact intermediate lifetime | Paged KV cache update / fill, sharded KV cache, multi-device all-gather / all-reduce, split reduction scheduling, NoC-aware max-cores-per-head policy |
| Grouped / ragged / sparse attention | `tilelang_repo/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_varlen_indice.py`; sparse GQA paged examples | Single-device `block_indices` / ragged `cache_seqlens` sparse read plus partial combine | Sharded sparse blocks, cross-core load balancing, distributed sparse scheduling |
| Chunk recurrence / scan | `tilelang_repo/examples/linear_attention/example_mamba_chunk_scan.py`; `tilelang_repo/examples/gdn/example_chunk_o.py`; `tilelang_repo/examples/kda/chunk_o.py` | Single-device chunk state, loop-carried state, and local state-buffer lifetime admission | Cross-core chunk scheduling, distributed state placement, communication-aware state handoff |
| Multi-block flash-attn / flash decode | Current flash-attn Blackhole tests; `tt_metal_repo/tech_reports/FlashAttention/FlashDecode.md` | Existing small and 32x32 bf16 direct runtime subset; next is multi-block exact-CB publish / consume correctness | Multi-core split reduction, semaphore-backed remote writes, NoC traffic control, distributed decode scaling |

This means P2 workload bring-up is not blocked on all of P3.
It is blocked on the specific P3 primitives each first subset actually needs.
Those primitives must be pulled forward explicitly instead of hidden under a
generic "later runtime" bucket.

## Execution Order

1. Placement / resource baseline:
   finish hardware-model-backed `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`,
   keep `TTResourceDemand` / `TTResourcePressureReport`
   as validator-consumed admission surfaces.
   Exit when validators reject unsupported memory space,
   distribution kind,
   page size,
   shard shape,
   and attached-core requirements before source / runtime emission.
2. Buffer address ABI gate:
   make interleaved,
   sharded,
   page-shaped,
   and per-work indexed buffer address parameters explicit through
   compile-time args,
   runtime args,
   and per-work descriptors.
   Exit when supported forms materialize from typed plans,
   and unsupported sharded,
   page-indexed,
   or shared runtime address-argument forms fail
   closed with typed reasons instead of being reconstructed by source or
   runtime readers.
3. Verification and admission gate:
   keep compile,
   source/spec projection,
   direct runtime,
   TT-Sim bf16 correctness,
   and typed unsupported reasons separate.
   Exit when every new subset has tests for the strongest admitted path and
   fail-closed tests for the paths that remain unsupported.
4. Leaf compute / GEMM correctness pack:
   admit non-flash standalone leaf compute workloads and GEMM layout variants
   before structured workload families.
   Existing interleaved stick / page-shaped copy direct-runtime tests remain
   the layout-movement baseline.
   Exit when unary / binary / broadcast / reduce / pack / typecast,
   GEMM variants,
   and any newly admitted layout movement have direct correctness gates or
   typed admission rejects.
5. Selection / index base:
   bring up standalone `topk` with `int32` index outputs.
   Exit when the backend proves value and index correctness instead of only
   compiling the selection kernel.
6. Exact intermediate event / materialization base:
   repair wider exact-CB publish / consume,
   partial-output combine,
   and source-live-form materialization for multi-kernel intermediates.
   Multi-block flash-attn is the closest current witness,
   but this step is a runtime primitive gate for MoE,
   paged,
   and sparse workloads,
   not a flash-only milestone.
7. Grouped / ragged work-packet base:
   represent group,
   block,
   ragged row count,
   and per-work indexed ranges as typed planning inputs.
   Exit when missing or inconsistent group / ragged metadata is rejected
   before source / runtime emission.
8. Pre-grouped MoE / `fusedmoe`:
   admit shared expert and pre-packed routed grouped GEMM with explicit
   grouping tensors.
   This does not count as full MoE.
   Exit when missing or inconsistent group metadata is a typed reject and
   local routed output correctness is covered.
9. Sparse / ragged attention first subset:
   admit single-device sparse GQA style workloads with `block_indices`,
   ragged `cache_seqlens`,
   and explicit split combine.
   Exit when index-driven sparse reads and partial combine have direct
   correctness coverage.
10. Paged GQA decode first subset:
   admit single-device page-table or block-table indexed KV reads with
   `cache_seqlens`,
   fixed `num_split` first,
   then dynamic `num_split`.
   Exit when paged GQA decode has typed admission and direct correctness
   coverage for the admitted shapes.
11. Paged MLA decode first subset:
   reuse the paged decode gate for MLA's split Q / Q_pe / KV / K_pe access
   pattern and larger intermediate pressure.
   Exit when paged MLA decode has typed admission and direct correctness
   coverage for the admitted shapes.
12. Chunk recurrence / scan first subset:
   admit single-device chunk state and loop-carried state lifetimes.
   Exit when state materialization and carry lifetimes are explicit in
   typed plans and covered by correctness tests.
13. Production distributed variants:
   only after the first subsets above are stable,
   expand mesh / submesh placement,
   sharded L1 / DRAM buffers,
   sharded buffer address materialization,
   common runtime args,
   remote endpoints,
   semaphores,
   full MoE routing,
   token packing,
   scatter / gather / combine,
   all-to-all,
   reduce-scatter,
   all-gather,
   sharded paged KV cache update / fill,
   distributed sparse attention,
   communication-weighted placement,
   multicast,
   NoC0 / NoC1 scoring,
   bounded FIFO / exact-CB event sizing,
   and list scheduling or graph partitioning.

Adding these before typed placement exists would recreate the current
over-complexity problem under a new name.

## Planning Order

1. Keep explicit leaf tile-compute and DAG covering boundaries clean.
2. Use typed resource demand / pressure reports as admission surfaces.
3. Keep CB / L1 admission hardware-backed.
4. Keep core groups hardware-model-backed and replace unit buffer placement
   with explicit buffer distribution.
5. Follow the execution order above for first-subset workload admission.
6. Pull forward only the P3 primitives required by the current first subset.
7. Defer production distributed variants until mesh / sharding / CCL /
   NoC / multicast / global scheduling plans are typed and validated.

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
- workload admission records distinguish first single-device subsets from
  production distributed subsets
- no resource truth is carried by bags, payloads, source hooks, or runtime
  fallback
