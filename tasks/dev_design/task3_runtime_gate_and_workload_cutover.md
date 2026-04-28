# Task 3: ExecutableSpec / Leaf Reader Cutover

## 基本信息

- **文档角色**: `ExecutableSpec / leaf reader` 合同文档
- **任务链位置**:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 文件名里的 `runtime_gate_and_workload_cutover`
  只是历史索引，
  不是新的 IR 层命名
- 本文档定义
  `ExecutableSpec`
  的长期 leaf projection /
  runtime-module build contract，
  以及它与 archived cleanup task3/task4/task5
  的历史边界关系
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

## 1. 目标

`ExecutableSpec`
不是新的 planning 层，
也不是已删除的 top-level
`TTProgram.payload`
在 leaf 侧的再包装。

它的语义固定为：

> **唯一的 leaf execution projection / runtime-module build contract**

它负责回答：

- 哪个 `TTProgram`
  被冻结成当前 leaf executable
- 哪些 kernel / segment /
  runtime-arg / accessor /
  CB / semaphore / core records
  已经是 leaf 可直接消费的显式结果
- 当前 execution backend
  是否接受这个 executable
- build / codegen / runtime /
  `BlackholeModule`
  如何据此物化最终
  `artifact.rt_mod`

它不负责：

- target planning
- compute builtin legality
- transport / sync / ABI
  的上游语义决定
- runtime/codegen 侧
  重新恢复 planner 语义
- 用当前 admitted support subset
  反向塑形
  `TTProgram`
  或更早层表示

补充说明：

- archived cleanup task3/task4/task5
  只是把历史 repo state
  收回到这个合同上的执行记录
- 它们不是当前活动路线图，
  也不是新的表示层，
  也不能反向改写
  `ExecutableSpec`
  的长期语义

## 2. 合法输入与禁止输入

leaf writer /
leaf reader
只允许建立在：

- validated `TTProgram`
- `TTProgram`
  上可直接投影出的显式对象
- leaf-local 的
  schema validation /
  backend admission /
  runtime-module materialization

这里要特别写清楚：

- `ExecutableSpec`
  只能是
  `TTProgram`
  的 direct projection
- `MaterializeBlackholeExecutable`
  是唯一 canonical writer
- build / codegen / runtime /
  `BlackholeModule`
  只能读取
  `tl.blackhole_executable`
  或其解析后的
  `ExecutableSpec`

不允许回升为 leaf 输入的东西：

- `blackhole.copy_semantics`
- `blackhole.segment_kind`
- `blackhole.lowering_requirements`
- helper bag /
  seed /
  payload fallback
- 从 `work_linear_id`
  或 builtin 序列
  反推 planning 语义
- runtime / codegen /
  build 自己补出来的
  kernel family /
  copy direction /
  buffer role /
  ABI meaning

如果当前 leaf 输入
证据不足，
结论只能是：

- 回到 `TTProgram`
  或更早层
  补显式表示
- 扩 projection validator
- 在 backend admission
  处显式 reject / unsupported

不能新增
leaf-time matcher /
late fallback /
replacement helper carrier。

## 3. `ExecutableSpec` 的显式 leaf truth

`ExecutableSpec`
的长期 owner truth
只能写成 leaf 可直接消费的
显式 projection 结果。

### 3.1 executable identity

它应编码：

- executable schema version
- executable source
- entry identity
- member function identity

这里的 `source`
只能表示：

- 当前 executable
  直接来源于
  `tl.tt_program`

它不是：

- planner provenance bag
- cleanup residue 来源说明

### 3.2 segment / kernel realization

它应编码：

- segment identity
- segment kind
- core type
- runtime args
- common runtime args
- compile-time arg specs
- accessors
- semaphore bindings
- per-work arg specs
- formal buffer identities
  and their exact runtime/common-runtime
  `buffer`
  role bindings
- typed compute operation records
  such as
  `KernelSpec.compute_ops`
  entries

这里的 `segment_plan`
已经是 leaf truth，
不是给 runtime
再去切 kernel body
的提示词。

formal buffer identity
必须由
`ExecutableSpec`
显式携带并与
runtime arg /
common runtime arg /
compile-time accessor ABI
中的
`buffer`
字段 exact match。
leaf reader /
codegen /
`BlackholeModule`
不能用
PackedArgs
位置、
`_handle`
suffix、
runtime arg kind
或
`spec.name`
fallback
恢复 buffer 身份。

per-work runtime descriptors
必须显式携带
`arg_identity`、
`descriptor_kind`
和
`value_source`。
leaf reader /
runtime /
codegen
只能用
runtime arg identity
绑定目标参数，
并用
typed value source
解释 work value；
不能把
`a_tile_start_id` /
`b_tile_start_id` /
`output_tile_start_id`
这类 arg-kind 名字
当作 block axis /
tile descriptor
owner truth。

### 3.3 execution resources

它应编码：

- CB config
- core plan
- semaphore plan
- resource pressure summary derived from typed
  `TTProgram`
  records
- execution-time 可直接读取的
  launch/resource records

这些事实已经处在
TT-specific 显式程序构造边界，
对齐的是 leaf build /
runtime / codegen 消费面，
不是上游 planner 再补一层的理由。

Resource pressure at this layer is an admission view:

- per-core CB ID pressure
- per-core CB L1 bytes
- allocator-managed L1 buffer pressure
- worker L1 budget
- core grid availability
- buffer distribution support

It may reject the current backend with a typed unsupported reason.
It must not assign TT-Metal physical addresses,
and it must not rebuild resource semantics from source strings,
CB names,
or runtime arg order.

### 3.4 backend admission metadata

它应编码：

- 当前 executable
  对某个 execution backend
  的 admitted / unsupported 信息
- fail-closed 的直接拒绝理由

其中最典型的就是：

- `direct_runtime_unsupported_reasons`

这类字段只允许承担：

- backend admission
- workload gate
- leaf execution diagnostic

它们不能承担：

- `TTProgram`
  语义 owner truth
- builtin legality
- 上游 planner 决策

### 3.5 最终交付边界

`ExecutableSpec`
不是最终交付 artifact 本身，
但最终交付必须站在它之上。

长期交付链固定为：

```text
tl.tt_program
  -> tl.blackhole_executable
  -> ExecutableSpec
  -> artifact.rt_mod
```

JIT / export /
execution backend adapter
最终只能消费：

- `artifact.rt_mod`
- 或 runtime-module
  等价交付物

不能回读：

- `TTProgram`
- legacy attrs
- helper residue

## 4. Writer / Reader / Validator 纪律

### 4.1 `MaterializeBlackholeExecutable` 是唯一 writer

长期 writer 纪律固定为：

- 有 `tl.tt_program`
  就投影
  `tl.blackhole_executable`
- 没有 `tl.tt_program`
  就不能继续保留
  stale executable attr

因此：

- executable projection
  只能由 canonical writer 生成
- 不允许新增第二条 writer path
- 不允许测试或 runtime
  把 leaf helper 重写成新的 owner truth

### 4.2 build / codegen / runtime / `BlackholeModule` 只读 executable projection

长期 reader 纪律固定为：

- build 只读
  `tl.blackhole_executable`
  或解析后的
  `ExecutableSpec`
- codegen 只读
  projected segment /
  CB / core /
  unsupported-op /
  runtime-arg 记录
- runtime /
  `BlackholeModule`
  只读 executable projection
  与 `ExecutableSpec`

明确禁止：

- 直接回读
  `tl.tt_program`
- 直接回读
  `blackhole.copy_semantics`
  / `blackhole.segment_kind`
  / `blackhole.lowering_requirements`
- 直接从 top-level payload
  恢复 work decomposition /
  block meaning /
  tile meaning

### 4.3 `ValidateExecutableSpecProjection`

leaf validator
只负责：

- 检查 projection
  只来源于
  `TTProgram`
- 检查 executable schema
  completeness / consistency
- 检查 readers
  不再自补上游 planning 语义

它不负责：

- 重做 `ValidateTTProgram`
- 重新裁定
  TT builtin legality
- 把 backend 当前 unsupported
  升格成上游表示层限制

### 4.4 Runtime / Backend Decoupling Plan

本轮 TT-Metal runtime model
复查后的结论是：
runtime 需要拆成三个独立边界，
不能再把 direct runtime
当前 admitted subset
当作 codegen 能力边界。

#### 4.4.1 TT-Metal program emission capability

这是 codegen / export
的目标边界。

它只要求
`ExecutableSpec`
足够显式，
能够 materialize：

- `Program`
- `MeshWorkload`
- `MeshDevice` /
  device range
- replicated /
  sharded
  `MeshBuffer`
- kernel source /
  compile-time args /
  runtime args
- circular buffer /
  semaphore /
  global semaphore
- fabric /
  multicast /
  collective transport
  所需的 leaf records

这条能力的 hard gate
是：

- executable schema completeness
- TT-Metal API materialization
  所需字段完整
- codegen/export emitter
  对对应 leaf records
  有实现

它不是：

- direct runtime
  是否已 admitted
- TT-Sim
  当前是否能执行该路径

#### 4.4.2 Direct runtime backend admission

direct runtime
只是 `BlackholeModule`
进程内执行 backend。

当前 admitted subset
可以继续保持保守：

- unit mesh
- replicated runtime buffers
- interleaved DRAM accessors
- copy /
  GEMM /
  已 admission 的
  live-form materialization

但这些限制只能写入：

- backend admission metadata
- `direct_runtime_unsupported_reasons`
- direct-runtime test matrix

它们不能写入：

- `SpatialPlan`
  合法性
- `TTProgram`
  target realization
  合法性
- codegen/export
  hard gate

#### 4.4.3 Runtime implementation sequence

后续 runtime 改动按下面顺序推进：

1. **Backend metadata split**
   - 将 direct runtime
     unsupported reason
     保持为 backend-local
     diagnostic
   - 确认 codegen/export
     不读取该字段决定
     TT-Metal program
     是否可生成
2. **TT-Metal emitter surface**
   - 让 leaf emitter
     面向
     `Program / MeshWorkload / MeshBuffer`
     schema
   - sharded /
     replicated /
     device-range /
     fabric transport
     都必须来自
     `TTProgram`
     和 `ExecutableSpec`
     typed records
3. **Direct runtime generalization**
   - direct runtime
     逐步从
     hard-coded
     `create_unit_mesh(0)`
     /
     replicated buffer
     中剥离出
     backend config
   - 每扩大一个 subset，
     先扩 typed schema
     和 validator，
     再扩 backend admission
     和 execution
4. **Verification split**
   - compile /
     projection /
     codegen /
     export
     验证 schema 和 emitter
   - direct runtime
     只验证 admitted subset
   - multi-device /
     mesh /
     fabric /
     CCL
     验证作为独立 backend
     或硬件/TT-Metal
     integration lane

禁止新增：

- 第二套 legacy runner
- runtime-only matcher
- 从 kernel 名 /
  buffer 名 /
  runtime arg 顺序
  恢复 mesh /
  sharding /
  collective
  语义的 fallback

## 5. Wrong-Now Residue 与 Cleanup Debt

下面这些东西
在历史 repo HEAD
里曾经必须明确写成
**wrong now, delete later**
或 **leaf compatibility debt**，
不能写成
`ExecutableSpec`
的长期 owner truth。
`2026-04-25`
compatibility fallback
收束后，
这些旧面已从 active chain 删除；
后续不能按兼容名义恢复。

### 5.1 `buffer_tile_bridge_specs` 不是新层

repo HEAD
已删除：

- `buffer_tile_bridge_specs`
- `tl.blackhole_logical_buffer_tile_bridge_specs`

logical tile layout
只允许由
`SpatialPlan.LayoutSpec`
和
`TTBufferDistributionPlan`
typed fields
承载，
再投影到
`ExecutableSpec.buffer_distribution_plans`。

它们不是：

- `ExecutableSpec`
  的长期字段族
- TT-Metal runtime contract
- codegen/runtime
  再造出来的新表示层

### 5.2 `compute_contract` / `gemm_contract` family 已删除

repo HEAD
已删除：

- `compute_contract`
- `multi_compute_contracts`
- `gemm_contract`
- `multi_gemm_contracts`
- `compute_epilogue_ops`

这些字段不再出现在
`TTProgram.payload`
生成 /
验证 /
测试、
`MaterializeBlackholeExecutable`
projection、
`ExecutableSpec`
结构、
runtime JSON
或 function metadata
公共 schema
中。

compute truth
现在只能从
`TTComputeOpPlan`
投影到
`KernelSpec.compute_ops`。
runtime /
`BlackholeModule`
只读
`ExecutableSpec`
typed compute /
kernel /
materialization
records；
缺字段时 fail-close，
不再从旧 contract family
推断。
当前 typed replacement
是
  `TTProgram.TTComputeOpPlan`
  到 executable
  `KernelSpec.compute_ops`
  的 projection；
  GEMM 只是其中
  `kind=gemm`
  entry，
  不是 compute kernel
  的通用字段名。
  exact builtin identity
  由
  `operation_name`
  字段承载；
  `kind`
  只表示
  GEMM /
  binary /
  unary /
  reduce
  等 compute family。
  non-GEMM
  internal operand binding
  可以没有
  host runtime buffer；
  只有 GEMM direct-runtime
  admission 继续要求
  host buffer
  显式存在。
  direct runtime
  已改从该 entry
  读取 GEMM
  shape /
  dtype /
  transpose /
  work-decomposition /
  typed operand binding /
  unsupported mbarrier gate
  truth；
  P0.2
  已新增
  `operand_bindings`
  数组，
  entry
  显式携带
  role、
  compute-side
  buffer
  与 host
  runtime
  buffer；
  producer
  不再从 reader/writer
  runtime arg
  顺序恢复
  A/B/C
  语义。
  后续 TT-Metal
  eltwise /
  reduction /
  SFPU /
  pack-materialization
  等 compute instruction
  也必须在同一
  `compute_ops`
  schema 下增加 typed
  `kind`
  entry，
  不允许新增并行 GEMM-only
  public field
  或从 builtin 序列 /
  buffer 名恢复语义

它们不是：

- `ExecutableSpec`
  的长期 planning truth
- task3 已完成的证明
- `TTProgram`
  仍可继续依赖 payload family
  的理由

### 5.3 `direct_runtime_unsupported_reasons` 只能停在 leaf admission

这个字段当前是
合法的 leaf concern，
但合同必须写死：

- 它只能表达
  当前 executable
  对 direct runtime
  的 unsupported 理由
- 它可以作为
  queryable runtime gate
- 它可以 fail-closed
  拒绝执行

它不能表达：

- `TTProgram`
  是否合法
- builtin selection
  是否正确
- 更早层表示
  是否该改

cleanup 之后的
support-surface admission lane：
direct cast consumer、
`fragment_fill -> cast -> publish`
和 flash-attn direct runtime
support 工作
必须继续服从这个边界。

任务级设计见
`2026-04-23-blackhole-live-form-materialization-admission.md`：

- unsupported reason
  只能表示当前 executable
  尚未 admitted
- 真正需要跨阶段保留的
  live-form /
  materialization
  distinction
  必须进入
  `SpatialPlan`
  / `TTProgram`
  / `ExecutableSpec`
  显式对象
- runtime/codegen
  不允许靠
  `SeqStmt`
  /
  builtin 序列
  /
  buffer 名
  补回 producer-consumer 语义

### 5.4 `segment_plan` 是 projection truth，`blackhole.segment_kind` 只允许 pass-local

`segment_plan`
属于 executable projection
的一等 leaf 字段。

repo HEAD
不允许 runtime /
codegen /
leaf reader
继续读取：

- `blackhole.segment_kind`

`blackhole.segment_kind`
只允许作为
`lower_blackhole_ops.cc`
内部 pass-local mechanics，
并且必须在 final IR /
leaf reader 前剥离。

如果后续又需要区分
kernel kind /
segment kind，
只能使用
`TTKernelPlan.kind`
和 executable projection
里的 typed `segment_plan.kind`。

它不是：

- `ExecutableSpec`
  需要 source-level marker
  才能成立的证据
- TT-Metal target model
  对 source marker
  的要求

### 5.5 防御性 fallback 不是长期 reader 合法性

当前实现不再保留
`TTProgram payload fallback`
报错文案，
也不允许从 top-level payload
或 `work_linear_id`
恢复 planning 语义。
防御性检查只能要求 typed 字段存在；
不能把旧 fallback
重新写成 reader 合同。

截至 `2026-04-25`，
leaf reader / codegen
也不再保留下面这些默认恢复面：

- 缺 `cb_configs`
  时生成 `default_cb`
- compute operand
  缺 `host_buffer`
  时用 device buffer 代替
- accessor
  从旧 `slot`
  兼容键恢复
  `compile_time_arg_offset`
- GEMM compute op
  缺 tile/block/subblock 字段时从
  `M/N/K`
  反推
- segment
  缺 `kind/core_type`
  时补
  `fused_dataflow/brisc`
- codegen
  在没有 segment
  `core_type`
  时默认走 dataflow header
- per-work descriptor
  同时携带旧
  `value_kind`
  和新
  `value_source`

这些字段现在必须由
typed `TTProgram`
或 executable projection
显式给出；
缺失时 fail-close。

### 5.6 public specialization audit verdict

本轮审查结论是：
当前 residue
不只是 compute 指令的问题，
而是所有跨
`TTProgram -> ExecutableSpec -> runtime / codegen`
边界的
workload-named /
order-based /
payload-seeded
public surface
都要按同一标准删除。

当前明确属于 wrong-now
的 surface：

- P0.1
  已删除
  `gemm_contract`
  / `compute_contract`
  / `multi_*_contracts`
  / `compute_epilogue_ops`
  在 projection、
  `ExecutableSpec`
  和 runtime metadata
  中的 public field
  形态；
  P0.6
  后续已删除
  `TTProgram.payload`
  seed
  读取 /
  写入
- P0.2
  已把 GEMM
  `KernelSpec.compute_ops`
  entry
  的 operand truth
  收进 typed
  `operand_bindings`；
  剩余 P0
  surface
  不能再新增 reader/writer
  runtime arg order
  恢复路径
- host wrapper
  和 codegen
  不能用 PackedArgs
  buffer 位置、
  handle suffix、
  runtime arg kind
  fallback
  绑定 buffer；
  这些绑定必须来自
  `ExecutableSpec`
  中的 typed buffer identity /
  role records，
  缺失时 fail-close
- `per_work_arg_specs`
  不能长期用
  `a_tile_start_id`
  / `b_tile_start_id`
  / `output_tile_start_id`
  /
  `gemm_num_k_tiles`
  这类 workload arg name
  表达 work descriptor；
  它需要收敛成引用
  core plan、
  compute op dims
  或 access pattern
  的 typed value expression
- materialization
  host buffer
  和 axis order
  不能由
  `_local`
  suffix、
  single-output fallback
  或 leaf-side shape heuristic
  恢复；
  `TTMaterializationPlan`
  /
  `ExecutableSpec`
  必须显式携带
  host binding
  和 layout/axis truth
- host-visible
  `TTMaterializationPlan`
  必须携带
  `host_buffer`
  并投影到
  `ExecutableSpec`；
  direct-runtime admitted
  `pack_thread_direct_store`
  /
  `pack_tile`
  publication
  缺少该字段时必须 fail-close。
  空
  `host_buffer`
  只允许表示内部 materialization，
  不能作为 leaf 侧恢复 host buffer
  的入口。
- `transport_page_size`、
  `host_axis_order`
  和
  `transpose_2d`
  是 accessor /
  compile-time ABI schema
  的显式 truth；
  runtime /
  `BlackholeModule`
  不能从 CB role、
  first-CB、
  固定 2048 page、
  tensor shape
  或 work-item 数量
  推回这些值。
  缺失时应在 leaf reader /
  runtime gate
  处 fail-close。
- projection encoder
  不能再以各类
  typed TTProgram node
  的
  `payload`
  为 seed
  构造 executable map；
  否则新旧语义会绕过
  validator
  混入 leaf schema

对应测试也要同步收口：
测试可以断言
`kind=gemm`
作为
`compute_ops`
variant
存在，
但不能继续要求
contract-family top-level field、
GEMM-specific per-work value kind
或 name/order fallback
作为长期绿测条件。

## 6. Workload Gate 与 Runtime Admission

leaf execution gate
只允许做两类事：

- 判断当前
  `ExecutableSpec -> execution backend`
  组合是否被支持
- 对不支持的组合
  fail-fast /
  fail-closed

它不允许做：

- 重新规划 kernel
- 重新解释 copy / gemm /
  transport / sync 语义
- 让 leaf runtime 限制
  倒逼上游表示层收缩

### 6.1 admitted support surface 只属于 leaf concern

当前 direct runtime
的 admitted support surface，
只属于 leaf execution concern。

codegen / export
不是 direct runtime
的 admitted subset；
它们的 gate
是 `ExecutableSpec`
schema completeness、
TT-Metal program materialization
所需字段完整，
以及 emitter 是否实现。

例如：

- copy direct runtime
  的 admitted subset
- GEMM direct runtime
  的 admitted subset
- `mbarrier`
  等 runtime-side unsupported

这些都只能停在：

- executable admission
- runtime gate
- direct execution constraint

不能升级成：

- `TTProgram`
  的长期语义定义
- task1/task2
  的完成门槛
- codegen/export
  的通用能力上限

### 6.2 workload payoff 不等于 cleanup completion

workload payoff
只能在 leaf boundary
已经收紧之后恢复，
但它不是 task3
唯一 completion truth。

必须明确区分：

- compile / projection /
  codegen / export
  hard gate
- admitted copy / GEMM
  direct-runtime hard gate
- unsupported workload
  的 queryable gate
- hardware /
  mesh /
  fabric
  integration gate
- TT-Sim
  simulator capability boundary

因此：

- `flash-attn`
  direct runtime
  不是当前 mainline hard gate
- unsupported workload
  不能反过来重写
  `ExecutableSpec`
  合同
- workload breadth
  不能拿来掩盖
  reader boundary
  还没收紧
- direct runtime
  不支持 distributed /
  mesh /
  fabric
  时，
  只能说明该 backend
  未 admission，
  不能说明
  TT-Metal codegen/export
  不该表达这些路径

## 7. 与 Archived Cleanup Task 的关系

### 7.1 cleanup task3

cleanup task3
的历史收口是删除：

- `blackhole.copy_semantics`
  及其 compiler-side consumers

它必须同时保持：

- target / codegen / build / runtime
  继续站在
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  projection boundary

它不能把：

- copy annotation
- 新的 copy helper carrier
- target-visible copy protocol

重新引回 leaf 边界。

### 7.2 cleanup task4

cleanup task4
的历史收口是：

- `segment_plan.kind`
  与 source-level marker
  之间不能再有 leaf-reader
  依赖

所以当前合同固定为：

- `blackhole.segment_kind`
  只允许作为
  `lower_blackhole_ops.cc`
  内部 pass-local mechanics
- final IR /
  `ExecutableSpec` /
  leaf reader
  前必须剥离
- 若 leaf reader
  重新依赖 `segment_kind`，
  应视为 regression

### 7.3 cleanup task5

cleanup task5
只记录最终 convergence /
verification / residue scan /
交付 gate 的历史执行，
不是新的语义 owner。

因此：

- task3 文档
  必须先写清楚
  真正的 leaf boundary
- archive 里的 task5
  只能作为完成期验证索引，
  不能作为当前保留
  leaf compatibility residue
  的授权

## 8. Completion Contract

`Task 3`
只有在下面这些条件同时满足后才算完成：

补充硬约束：

- `Task 3`
  默认按终态实现，
  不接受
  “先引入 `ExecutableSpec`
   或新 leaf projection，
   再把旧 leaf reader /
   helper bag /
   payload fallback
   留到后面删除”
  的过渡式收口
- 一旦声明
  `Task 3`
  完成，
  旧 leaf reader /
  helper bag /
  payload fallback /
  planner residue
  就不能再以
  compatibility shell
  身份留在 active path

1. `MaterializeBlackholeExecutable`
   成为唯一 executable writer
2. build / codegen / runtime /
   `BlackholeModule`
   只读
   `tl.blackhole_executable`
   / `ExecutableSpec`
3. `ExecutableSpec`
   只作为
   `TTProgram`
   的 direct projection 存在
4. leaf readers
   不再直接读取
   `blackhole.copy_semantics`
   / `blackhole.segment_kind`
   / `blackhole.lowering_requirements`
   / helper bag
5. `buffer_tile_bridge_specs`
   与 contract-family residue
   已退出 active chain；
   top-level
   `TTProgram.payload`
   已删除，
   后续不能重新引入 fallback 链
6. backend admission
   只停在 leaf execution gate，
   不再反向塑形
   `TTProgram`
   或更早层语义
7. direct runtime
   unsupported
   不再阻断
   schema-complete
   TT-Metal codegen/export
8. JIT / export /
   execution backend adapter
   的最终交付
   站在
   `artifact.rt_mod`
   或等价 runtime artifact
   上，而不是回读 planner residue
