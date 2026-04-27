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
- 文件拆分不能改变 validator 纪律：
  跨阶段 truth 必须继续通过 typed
  `SpatialPlan` /
  `TTProgram` /
  `ExecutableSpec`
  state 表达。

## Current Slice

第一刀拆出：

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

## Remaining Split Candidates

`lower_blackhole_ops.cc`
仍然承担多块相邻职责。
后续应按下面顺序继续缩小：

1. exact-CB live-form /
   materialization helpers
2. ABI / accessor descriptor encoding
3. staged copy / transport source emission
4. matmul partial reload / post-merge publish support

这些都只是 implementation split。
任何 split 都不能把旧 matcher /
generate family 作为 compatibility shell
留在 active chain。

## Other Pass Audit

按文件规模和职责混杂度扫过
`tilelang_repo/src/transform` 后，
最明显的后续候选是：

- `lower_tile_op.cc`
  - 同时包含 generic tile op lowering
    和 Blackhole tile compute normalization
  - 当前有两份相近的
    `MakeBlackholeTileComputeCall`
    /
    store normalization logic
    分别在 `LowerTileOpPass`
    和 `BlackholeTileComputeNormalizer`
  - 后续 cleanup 应把 Blackhole-specific
    normalization 收束成单一实现面，
    继续产出显式
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
