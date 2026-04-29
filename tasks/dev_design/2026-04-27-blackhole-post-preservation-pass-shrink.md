# Blackhole Post-Preservation Pass Shrink

## Goal

`Blackhole tile-compute preservation`
已经把 compute truth 收回到
`Normalized Tile TIR`
里的显式 tile compute 语义，
并把 downstream scalar-loop matcher /
generate family 删除。

本 cleanup lane 的目标是继续缩小
`lower_blackhole_ops.cc`
这个膨胀实现文件的责任面。
这不是新 IR 层，也不是新的长期协议对象；
长期主链仍然只有：

```text
Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec
```

## Contract

- 拆分按表示层和实现责任做：
  selection / planning / emission
  可以拆文件或拆 pass，
  但不能引入新的 side-channel。
- `TTComputeOpPlan.operation_name`
  和 `KernelSpec.compute_ops`
  继续只记录 TT-Metal leaf API 粒度：
  `mul_tiles`,
  `add_tiles`,
  `*_bcast_cols`,
  `exp2_tile`,
  `pack_tile`
  等。
- helper class / template / macro
  只能减少 leaf emission boilerplate；
  不能重新产生
  `scalar_exp2_affine`,
  `row_broadcast_affine`,
  `GenerateScalar*`
  这类 composite protocol surface。
  也不能把 composite semantics
  藏进 leaf-looking payload
  或 hook：
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  /
  `mul_tiles_bcast_cols("div", ...)`
  与旧 composite helper 名
  属于同一类设计违规。
- 文件拆分不能改变 validator 纪律：
  跨阶段 truth 必须继续通过 typed
  `SpatialPlan` /
  `TTProgram` /
  `ExecutableSpec`
  state 表达。

## Completed Splits

本 lane 只做 implementation responsibility split，
不引入新的 IR 层或跨 pass 语义通道。
`lower_blackhole_ops.cc`
从约 7.9k 行收缩到约 2.4k 行；
保留 pass driver、
CB requirement bookkeeping、
logical shape loading、
generic validators
和 visitor orchestration。

已拆出的责任文件：

- `tilelang_repo/src/transform/lower_blackhole_tile_compute.cc`
  - owns explicit preserved tile compute lowering
  - owns TT-Metal leaf compute sequence emission for
    fill / copy / row-reduce / binary /
    broadcast-cols / exp2 exact tile sequences
  - keeps the public pass entry and class state in
    `PlanTTKernelABI`
    without adding a new cross-pass protocol

同时加入
`ExactTileComputeEmitter`
作为 pass-local helper，
集中 reserve / wait / pop / push /
pack tile 这些重复 leaf emission mechanics。

第二刀拆出：

- `tilelang_repo/src/transform/lower_blackhole_exact_cb.cc`
  - owns exact tiled-CB live-form helpers
  - owns local fragment
    publish / materialize
    helpers for exact tile compute consumers
  - keeps lifetime and alias truth in
    existing `PlanTTKernelABI`
    typed state,
    without creating a new cross-pass protocol

- `tilelang_repo/src/transform/lower_blackhole_materialization.cc`
  - owns fragment cast and local-to-CB
    materialization sequence planning
  - owns the pass-local structural matching
    for those explicit materialization loops
  - records typed live-form/materialization plans
    through existing `PlanTTKernelABI` state

- `tilelang_repo/src/transform/lower_blackhole_abi.cc`
  - owns segment plan storage,
    accessor descriptor encoding,
    runtime/per-work arg spec synthesis,
    `TTKernel` seed construction,
    `TTABIPlan` seed construction,
    and exact compute op plan finalization
  - keeps `TTComputeOpPlan.operation_name`
    at TT-Metal leaf API granularity

- `tilelang_repo/src/transform/lower_blackhole_state.cc`
  - owns SpatialPlan live-value/materialization
    references,
    tiled-CB live-form alias state,
    materialization plan host-buffer finalization,
    physical compute buffer binding,
    and buffer-flow future-use queries
  - keeps this as pass-local mutable state;
    durable cross-stage truth remains typed
    `SpatialPlan` / `TTProgram` state

- `tilelang_repo/src/transform/lower_blackhole_transport.cc`
  - owns staged copy shape inference,
    accessor slot selection,
    DRAM/CB copy direction handling,
    page/tile transport source emission,
    and fused staged copy sequence emission

- `tilelang_repo/src/transform/lower_blackhole_matmul.cc`
  - owns GEMM extraction,
    matmul builtin sequence emission,
    accumulator partial reload,
    post-merge cast publication,
    and merge/reload helper sequences

The old downstream matcher / generate family remains deleted.
No `GenerateScalar*`,
`GenerateExplicit*`,
`RejectLegacyScalar*`,
or composite helper op protocol is retained
as a compatibility shell.

## Other Pass Audit

按文件规模和职责混杂度扫过
`tilelang_repo/src/transform` 后，
第一批后续 cleanup 已完成：

- `lower_tile_op.cc`
  - 同时包含 generic tile op lowering
    和 Blackhole tile compute normalization
  - 已将原来两份相近的
    `MakeBlackholeTileComputeCall`
    /
    store normalization logic
    分别在 `LowerTileOpPass`
    和 `BlackholeTileComputeNormalizer`
    中的实现收束成单一共享 helper
  - 继续产出显式
    `tl.tileop.blackhole_compute`
    而不是下游 matcher

`storage_rewrite.cc`
和 `thread_storage_sync.cc`
也较重，
但它们更偏 target-independent
storage/sync rewrite；
不应在本 Blackhole cleanup lane
里顺手改动。

## Validation

本 lane 的每批拆分至少需要：

- `cmake --build build -j32`
- Blackhole compile/pipeline regression tests
- 对实际触达 runtime/source 行为的改动，
  继续用 TT-Sim bf16 runtime tests
  作为 admitted support surface gate
- cleanup scan 确认旧 composite matcher /
  generate names 没有回流
