# Blackhole Tile Compute Preservation Design

## 基本信息

- **文档 ID**:
  `2026-04-27-blackhole-tile-compute-preservation`
- **日期**: `2026-04-27`
- **状态**: 当前活动设计合同
- **父设计**:
  `tasks/dev_design/final_blackhole_backend_redesign.md`
- **定位**:
  定义 Blackhole tile compute 语义在
  `Normalized Tile TIR`
  中被保留 / 规范化的边界，
  并给出删除 P2.2/P2.3 late TIR idiom recovery
  的迁移路线

## 1. 设计结论

Blackhole 的 compute ops 集合必须按
TT-Metal compute API 粒度定义。

这不是 reduce 专项问题。
凡是 TT-Metal 以 tile compute API
直接表达的语义，
都不应先被 generic scalar TIR lowering
展开成局部 loop / scalar expression，
再在 `lower_blackhole_ops.cc`
里靠 workload-motivated idiom matcher
恢复。

长期边界固定为：

```text
TileLang source / DSL
  -> Normalized Tile TIR with preserved tile compute semantics
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

其中：

- `Normalized Tile TIR`
  是 tile compute 语义 owner truth
- `SpatialPlan`
  只观察 execution-unit /
  dataflow /
  live-value /
  materialization boundary
  关系，
  不拥有 TT-Metal builtin 名
- `TTProgram`
  负责把已显式存在的 tile compute 语义
  选成 TT-Metal leaf API
  和 typed `TTComputeOpPlan`
- `ExecutableSpec`
  只消费 leaf projection；
  不能恢复 compute 语义

P2.2/P2.3 中存在的
row-reduction /
row-broadcast /
exp2 affine /
scalar max /
scalar fma /
copy /
fill /
cast
late matcher
只允许作为已经识别出的过渡债务。
它们不能继续扩展成新的公共协议面。

## 2. 适用范围

本设计覆盖 Blackhole 上所有
TT-Metal compute API 粒度的 tile semantics，
集合来源应是 TT-Metal compute API
的中心化 leaf table
（当前参考
`tt_metal_repo/tt_metal/hw/inc/api/compute/`），
而不是 flash-attn 当前用到的局部白名单。

当前已知 leaf API 家族包括但不限于：

- matmul:
  `matmul_tiles`
- reduction:
  `reduce_tile`
- binary tile ops:
  `add_tiles`、
  `mul_tiles`、
  `sub_tiles`、
  `binary_max_tile`
- broadcast binary tile ops:
  `add_tiles_bcast_rows`、
  `add_tiles_bcast_cols`、
  `add_tiles_bcast_scalar`、
  `mul_tiles_bcast_rows`、
  `mul_tiles_bcast_cols`、
  `mul_tiles_bcast_scalar`、
  `sub_tiles_bcast_cols`、
  `sub_tiles_bcast_scalar`
- unary tile ops:
  `exp_tile`、
  `exp2_tile`、
  `recip_tile`、
  `sqrt_tile`、
  `abs_tile`、
  `typecast_tile`、
  `fill_tile`
- tile movement / register publication:
  `copy_tile`、
  `pack_tile`
- layout movement APIs
  whose primary contract is tilize / untilize:
  `tilize_block`、
  `fast_tilize_block`、
  `untilize_block`、
  `pack_untilize_block`、
  `pack_untilize_dest`

这个集合按 TT-Metal API 能力增长而增长，
不按 flash-attn、
softmax、
GEMM epilogue
等 workload 名字增长。

不属于生产 compute op 粒度的名字：

- `softmax`
- `exp2_affine`
- `row_broadcast_exp2_affine`
- `scalar_exp2_affine`
- `row_reduction`
  这类 workload/composite helper 名
- `*_init` /
  `*_uninit` /
  `reconfig_data_format`
  等 init / format protocol
- `cb_wait` /
  `cb_pop` /
  `cb_reserve` /
  `cb_push`
  等 transport / queue protocol

composite helper 可以作为调试描述出现，
不能进入
`TTComputeOpPlan.operation_name`、
`KernelSpec.compute_ops`、
source/codegen leaf protocol
或 runtime admission surface。

## 3. 根因

当前 generic `LowerTileOp`
在 Blackhole exact builtin 选择之前运行。
对 GPU / SIMT target，
把 reduce、
broadcast、
elementwise
展开成 thread-level scalar loop
是合理的：
这些 target 的 reduce / elementwise
没有对应的单条 tile hardware compute API，
后续 codegen 面向 SIMT instruction stream。

但 Blackhole 是 tile-based compute target。
TT-Metal 已经把很多语义表达成
tile compute API。
一旦这些语义被提前展开成 scalar loop，
后段只能重新从
`BufferStore(BufferLoad(...))`
形态中猜回：

- 哪个 region 是 tile operand
- 哪个 axis 是 broadcast /
  reduction axis
- 哪个 scalar expression
  应该拆成哪些 TT-Metal tile API
- 哪个 local fragment
  实际需要 CB-live materialization

这就是 P2.2/P2.3
`lower_blackhole_ops.cc`
膨胀的直接原因。
它同时承担了：

- TIR idiom matching
- compute DAG recovery
- exact CB materialization planning
- `TTComputeOpPlan` recording
- source emission support

这个职责组合违反 IR-first 边界。
正确修法不是继续增加 matcher，
而是让 tile compute semantics
在被 scalar expansion 破坏前，
就以显式表示留在
`Normalized Tile TIR`
中。

## 4. 表示合同

### 4.1 `Normalized Tile TIR`

`Normalized Tile TIR`
必须保留或规范化以下 compute 语义：

- tile op kind:
  matmul / reduce / unary / binary / broadcast /
  copy / pack / fill / cast
- operand roles:
  `src`、`lhs`、`rhs`、`dst`、
  `acc`、`scalar`、
  或对应 API 需要的 typed role
- operand buffer region:
  producer / consumer /
  live form /
  tile coordinate /
  region shape
- axis semantics:
  reduction axis、
  broadcast axis、
  tile row/col direction
- dtype semantics:
  logical dtype、
  physical CB dtype、
  accumulator dtype
- operation expression:
  leaf TT-Metal API 可直接表示的 op，
  或可在当前层 deterministic decomposition
  成 leaf API op 的 local expression tree

这个表示可以落成现有 `TileOperator`
保留策略，
也可以落成带 typed attrs 的 TIR call / node。
选择哪种编码是实现细节；
必须满足的长期属性是：

- 跨阶段语义存在于当前 IR
- analysis 可失效、可重算
- downstream 不需要从 scalar loop 中恢复 compute intent
- validator 能证明 unsupported case fail closed

### 4.2 Leaf API 选择

TT-Metal leaf API 选择只允许消费
显式 tile compute 语义。

例如表达式：

```text
exp2(acc * scale - row_max * scale)
```

不能作为生产 builtin
`exp2_affine`
流到后段。
它应在 tile compute normalization
中分解为 leaf op DAG：

```text
mul_tiles / mul_tiles_bcast_cols
add_tiles / add_tiles_bcast_cols
exp2_tile
```

分解规则以 IR 结构、类型、region、axis
为依据，
不能以 workload 名字、
buffer 名字、
或某个 flash-attn case 的局部顺序为依据。

### 4.3 Materialization 分离

compute op 只回答
“执行哪一个 TT-Metal tile compute API 语义”。

它不回答：

- CB 分配
- tile register 生命周期
- `cb_wait` /
  `cb_pop` /
  `cb_reserve` /
  `cb_push`
- page count /
  exact CB republish event
- mailbox /
  semaphore /
  runtime arg

这些仍属于
`SpatialPlan`
live/materialization boundary
和 `TTProgram -> ExecutableSpec`
resource / transport / ABI
realization。
`copy_tile` /
`pack_tile`
可以作为 tile compute leaf op，
tilize / untilize
这类 layout movement API
也必须以 leaf API 粒度表达，
但它们的 CB / layout protocol
不能反向写入
compute semantics。

## 5. Pipeline 变化

当前目标 pipeline：

```text
TileLang source / DSL
  -> layout inference / layout reducer
  -> preserve or normalize Blackhole tile compute semantics
  -> LowerTileOp for non-Blackhole or scalar-only residue
  -> BuildSpatialPlan
  -> ValidateSpatialPlan
  -> SplitBlackholeKernel
  -> PlanTTBlocks
  -> SelectBlackholeTTMetalBuiltins
  -> PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

可接受的实现方式有两类：

1. 让 `LowerTileOp`
   对 Blackhole target
   preserve 所有可映射到 TT-Metal API
   的 tile compute operators，
   类似当前 `GemmPyNode`
   preservation。
2. 在 `LowerTileOp`
   destructive scalar expansion 前
   增加 Blackhole tile compute normalization，
   把可映射语义显式化，
   并只让无法映射的 residue
   继续走 scalar lowering。

这两个方式都必须满足同一 postcondition：

> Blackhole builtin selection 之前，
> TT-Metal API 粒度的 tile compute 语义
> 仍可从当前 IR 结构和类型稳定读出。

## 6. 层责任

### 6.1 `LowerTileOp` / tile op lowering

Blackhole 分支职责：

- preserve 可映射到 TT-Metal compute API
  的 tile operator
- 或把 local tile expression
  normalize 成 explicit tile compute call
- 对不支持的 tile compute form
  fail closed 或保留为明确 unsupported residue
- 不生成 workload-specific helper builtin

CUDA / HIP / LLVM SIMT target
仍可按原有策略 scalar expand。
这是 target programming model 的差异，
不是语义分叉。

### 6.2 Tile compute normalization

normalization 的粒度是
单个 tile / region assignment
或单个 explicit tile operator，
不是整个 workload。

它可以做：

- expression tree decomposition
- axis inference
- dtype / accumulator normalization
- operand role assignment
- leaf op legality check

它不能做：

- 根据 kernel 名 /
  buffer 名 /
  source variable 名判断语义
- 生成 `softmax` /
  `exp2_affine`
  等 composite builtin
- 同时规划 CB protocol /
  semaphore /
  runtime arg
- 为了某个 flash-attn shape
  固化 loop 顺序或 page 数

### 6.3 `SelectBlackholeTTMetalBuiltins`

selector 职责：

- 从 explicit tile compute semantics
  选择 TT-Metal leaf API name
- 填写 typed `TTComputeOpPlan`
- 保证 `operation_name`
  为 TT-Metal API 粒度
- 对 unsupported combination
  提供 fail-closed diagnostic

selector 不再承担：

- post-scalar TIR idiom recovery
- composite op synthesis
- workload-specific sequence matching

### 6.4 `TTProgram` / `ExecutableSpec`

`TTProgram`
只消费 selected tile compute ops
和 `SpatialPlan`
提供的 dataflow/materialization truth。

`ExecutableSpec`
只做 leaf projection。
runtime / codegen
只能读取：

- `KernelSpec.compute_ops`
- typed resource / transport /
  ABI / live-form /
  materialization plans

它们不能把 source fragment
或 scalar expression
再解析成新的 compute plan。

## 7. 迁移路线

### Phase A: reduce preservation

- 让 Blackhole reduce
  在 `LowerTileOp`
  destructive scalar expansion 前
  保留为 explicit tile compute semantics
- `SelectBlackholeTTMetalBuiltins`
  从 explicit reduce semantics
  生成 `reduce_tile`
  plan
- 删除或禁用
  `lower_blackhole_ops.cc`
  中对应 row-reduction scalar-loop matcher
- 保持 P2.1 / P2.2 / P2.3
  flash-attn runtime tests 绿

当前实现状态
（`2026-04-27`）：

- Blackhole `LowerTileOp`
  已 preserve `ReduceOpNode`
  为 `tl.tileop.reduce`
  而不是 scalar expand
- `ReduceOpNode`
  已提供 operator-level
  `GetDataflowAccessInfo()`，
  因此 `SpatialPlan`
  可直接从 preserved op
  看到 source consume /
  destination produce；
  `clear=false`
  reduce 额外表达
  destination consume
- `SelectBlackholeTTMetalBuiltins`
  已从 explicit `tl.tileop.reduce`
  生成 `reduce_tile`
  typed `TTComputeOpPlan`
  和 leaf builtin sequence
- row-reduction scalar-loop matcher
  已从 active lowering path 删除；
  剩余 matcher 只在 residual scan
  中作为 fail-closed diagnostic
  判断 post-scalar reduction residue
- explicit reduce lowering
  保留 accumulator fill/live truth
  到 generator 内消费，
  避免 `clear=false`
  reduction 退回 forbidden
  direct CB interface
  materialization

### Phase B: unary / binary / broadcast preservation

- 添加 generic tile unary /
  binary /
  broadcast representation
- 覆盖 `add_tiles`、
  `mul_tiles`、
  `binary_max_tile`、
  `*_bcast_rows`、
  `*_bcast_cols`、
  `exp_tile`、
  `exp2_tile`、
  `recip_tile`
- 把 P2.2/P2.3
  exp2 affine /
  scalar fma /
  scalar max /
  row broadcast
  的 late recovery
  改成 upstream decomposition
  到 leaf tile API DAG
- 删除 helper/composite builtin 名
  进入 production plan 的路径

### Phase C: copy / pack / materialization boundary cleanup

- 明确 `copy_tile` /
  `pack_tile`
  是 tile register / CB publication
  leaf compute action
- 将 exact CB live-form /
  page event /
  publication protocol
  继续保留在 materialization plans
- 删除 compute matcher
  对 materialization truth
  的隐式推导

### Phase D: validator and file split

- validator reject
  unsupported post-scalar fragment compute residue
- validator reject
  composite `operation_name`
- 将 `lower_blackhole_ops.cc`
  中剩余职责拆回：
  tile compute selection /
  materialization planning /
  ABI planning /
  source emission support
  各自明确边界

## 8. 验证方式

结构验证：

- explicit reduce survives Blackhole frontend lowering
- explicit unary / binary / broadcast tile compute
  survives Blackhole frontend lowering
- unsupported tile expression
  fail closed before TTProgram materialization
- `TTComputeOpPlan.operation_name`
  只出现 TT-Metal API 粒度名字

回归验证：

- flash-attn pipeline tests
- flash-attn bf16 TT-Sim direct runtime tests
- copy runtime tests
- GEMM guard tests
- existing TTProgram / ExecutableSpec validator tests

清理扫描：

- no production
  `exp2_affine` /
  `row_broadcast_exp2_affine` /
  `scalar_exp2_affine`
  builtin name
- no production
  `softmax`
  compute op name
- no new cross-pass
  bag /
  payload /
  helper wrapper
  carrying compute semantics

## 9. 完成判据

本 lane 完成时必须同时满足：

- Blackhole TT-Metal API 粒度 tile compute semantics
  在 scalar expansion 前保留或规范化
- `TTComputeOpPlan`
  只从 explicit tile compute semantics
  产生
- `operation_name`
  只使用 TT-Metal leaf API 粒度
- P2.2/P2.3 late scalar-loop idiom matcher
  被删除或降级为 unreachable debug guard
- runtime / codegen
  不从 scalar expression
  恢复 compute semantics
- 受影响设计文档、
  progress、
  memory
  与代码事实同步
