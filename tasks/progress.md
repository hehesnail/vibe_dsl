# TileLang Blackhole Backend Progress

> 当前 HEAD 看板只放状态、blocker、下一步和最近验证摘要。
> 长期设计合同看 `tasks/dev_design/` 下的入口文档。

## Status

- Date: `2026-05-01`
- Active lane: `Buffer address ABI gate`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current HEAD

- Blackhole 正式执行路径是进程内 `BlackholeModule` direct host path；
  `execution_backend="tvm_ffi"` 的 wrapper/export path 可用。
- Legacy external runner / `build_blackhole/` 已删除。
- Broad legacy protocols 已退出 active chain：
  `compute_contract`,
  `gemm_contract`,
  `multi_*_contracts`,
  top-level `TTProgram.payload`,
  bridge attrs,
  lowering facts contract maps,
  compute-op seed maps,
  leaf name/default fallbacks。
- Tile compute truth 保持 TT-Metal leaf API 粒度。
  Composite pseudo-leaf payload 已清理；
  `TileComputeDAG`
  只作为 pass-local explicit-leaf legalization /
  covering input。
- 非 flash-attn 计算面没有删除：
  typed leaf compute surface 仍覆盖
  matmul / copy / unary / binary / broadcast /
  reduce / pack / typecast 等 TT-Metal leaf families。
  Runtime admitted subset 比 typed compute surface 窄，
  以本文件的 support boundary 为准。
- Algorithmic foundation 已存在并有 active consumers：
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  TT live-form solver。
  它们不是 compute expression lowerer 或全局 resource allocator。
- `TTResourceDemand`
  /
  `TTResourcePressureReport`
  已进入 `TTProgram`，
  并由 `ValidateTTProgram`
  消费。
- CB / L1 admission 已使用
  `TTHardwareModel`
  的 CB count、
  worker L1 budget
  和 alignment facts。
- Core groups 已开始消费
  `TTHardwareModel`
  的 worker grid /
  functional worker count；
  validators 会拒绝 out-of-grid core、
  duplicate core
  和非法 work packet。
- `TTBufferDistributionPlan`
  已从假
  `unit_mesh` /
  `replicated`
  baseline 扩到硬件模型约束的 placement：
  DRAM ABI layout 产出 interleaved placement；
  shared / CB-backed L1 产出 attached-core sharded placement；
  普通 per-worker local L1 保持 device-local replicated placement。
  Page size 来自 ABI layout、TIR buffer storage 和 CB plan，
  projection 会携带 attached core group/index。
- `ValidateTTProgram`
  已消费 buffer distribution placement：
  unsupported memory space / distribution kind、
  invalid page size、
  shard shape、
  attached-core reference/index、
  DRAM view size
  和 L1 worker budget
  会在 source / runtime emission 前 fail closed。
- Blackhole C++ audit 第一批已收口：
  scalar bitcast、
  DLTensor compact-layout gate、
  leaf-reader fail-closed、
  resource Var alias、
  module bytes serialization/export-load
  均有回归覆盖。

## Current Blocker

Buffer placement baseline 已进入 typed `TTProgram`，
但 buffer address ABI 仍不够显式。

Source / runtime reader 仍需要 typed interleaved、
sharded、
page-indexed
和 per-work buffer address 参数，
不能在 emission / runtime reader 里从名字、
布局字符串
或旧 runtime args 重新猜。

当前 sharded placement 仍是 coarse marker，
而且标记对象是 L1-side working view / materialization，
不是 DRAM 上的 global tensor 本体：
现有 `TTBufferDistributionPlan.shard_shape`
实际装的是 attached core-group grid shape，
不是 TT-Metal / TTNN 意义上的 per-core data shard shape。
下一步必须把 L1 sharded view 与其 source buffer / source region
显式关联，并拆出
`shard_grid_shape`
/
`sharding_strategy`
/
真实 `shard_shape`
/
logical-index 到 core-local address mapping
/
DRAM-source region 到 L1 shard 的 copy/address mapping，
再让 source / runtime emission 消费。

Buffer address ABI 还必须把 TileLang/GPU 风格的逻辑 work grid
和 Blackhole 物理 worker grid 分开：
`T.Kernel(grid_x, grid_y)` 是 logical work item 域，
`TTCoreGroup.physical_cores` 是实际常驻 worker，
`work_packets` 是 logical work id 到 worker 的 temporal 映射。
当 logical block 数超过 physical core 数时，
每个 worker 在自己的 packet 上循环执行，
并复用同一份 per-worker L1 / CB scratch。
source / runtime reader 不能按 logical block 数复制 L1 / CB allocation，
也不能在 reader 里重新猜每个 work id 对应的 DRAM source region。

GPU-style
`alloc_shared((tile_m, tile_n))`
目前应被解释为 per-worker、per-work-item 的 L1 / CB scratch shape。
这个 shape 对 Blackhole L1 来说可能偏小，
但 baseline correctness 必须尊重前端形状并做 capacity/admission 检查。
填满更多 L1 是显式 TT retile / work-coarsening 任务，
不能由 buffer placement 或 address ABI 静默改大。

## Next Task Order

1. Tighten buffer address ABI:
   make interleaved / sharded / page-indexed buffer address parameters,
   compile-time args,
   runtime args,
   per-work descriptors,
   and typed rejects explicit.
   Preserve the separation between logical work grid,
   physical core group,
   temporal work packets,
   and per-worker L1 / CB scratch reuse.
   For sharded placement, split the current coarse grid marker from the
   real per-core shard data shape and strategy, and bind the L1 sharded
   view to its DRAM/global source region before emitting addresses.
   Do not silently retile GPU-style shared buffers to fill Blackhole L1;
   report underutilization or add an explicit retile/work-coarsening plan
   before changing those shapes.
2. Keep admission levels separate for every new subset:
   compile,
   source/spec,
   direct runtime,
   TT-Sim bf16 correctness,
   and typed unsupported reason.
3. Admit first-subset workloads in this order:
   non-flash leaf compute / GEMM variants,
   standalone `topk`,
   exact-CB / materialization / partial-combine primitive
   via multi-block flash-attn,
   grouped / ragged work packets,
   pre-grouped MoE / `fusedmoe`,
   sparse / ragged attention,
   paged GQA decode,
   paged MLA decode,
   then chunk recurrence / scan.
4. Pull forward only the P3 primitives required by the current first subset.
5. Defer production distributed variants until mesh / sharding / CCL /
   NoC / multicast / global scheduling plans are typed and validated.
   Full MoE and full paged decode are not admitted by their first subsets.

## Support Boundary

- Direct runtime admitted subset:
  copy equal source/dest range with stride 1;
  selected interleaved stick / page-shaped copy cases with explicit
  alignment gates;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM buffer address schema with no common runtime args
  (`common_runtime_arg_count = 0`);
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths.
- Typed compute surface broader than direct runtime:
  matmul / copy / unary / binary / broadcast /
  reduce / pack / typecast leaf families must get
  workload-specific admission and correctness gates
  before being counted as runtime support.
- Workload backlog broader than flash-attn:
  `topk`, MoE / `fusedmoe`, paged attention /
  paged decode / MLA decode paged,
  grouped / ragged / sparse attention,
  and chunk recurrence / scan are not admitted yet;
  they require explicit subset definitions and regressions
  before being counted as supported.
- Workload dependency split:
  first single-device subsets are allowed to proceed after typed placement
  and subset-specific admission proof;
  full MoE, full paged attention / decode,
  distributed sparse attention,
  and production flash decode require later P3 runtime features.
  The detailed matrix lives in
  `tasks/dev_design/2026-04-29-blackhole-resource-planning-roadmap.md`.
- Flash-attn admitted direct-runtime subset:
  small single-work-item and 32x32 MHA / GQA bf16.
- Flash-attn compile/source/spec stable but runtime-gated subset:
  seq64 / multi-K-step MHA and GQA.
- Not admitted:
  multi-block flash-attn direct-runtime correctness,
  larger multi-page exact-CB publish/consume events,
  full multi-device / sharded / fabric collective runtime.

## Latest Verification

Latest implementation batch:
hardware-model-backed buffer distribution placement baseline.

Verified:

- `cmake --build build -j32`
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py`
  (`89 passed`)
- `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  (`52 passed, 10 skipped, 1 xfailed`)
- `pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_gemm.py`
  (`111 passed, 15 skipped`)
- TT-Sim env via `scripts/setup_tt_sim.sh`,
  then
  `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  (`16 passed`)
