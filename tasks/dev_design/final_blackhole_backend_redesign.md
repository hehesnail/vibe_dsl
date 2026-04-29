# TileLang Blackhole Backend Redesign

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-16`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 只保留长期架构、层间边界、显式表示规则、validator 纪律和当前 rewrite 方向

## 1. 设计结论

Blackhole 当前的根本问题不是“还差几个 builtin”或者“还差几个 contract”，
而是 **中间的 virtual spatial/dataflow layer 没有真正立起来**。

当前代码名义上是：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

但现实里：

- `SpatialPlan` 过薄，只承接一层 legacy 兼容标签
- target-independent 的 virtual spatial/dataflow 语义没有被对象化
- 后段只能补出
  legacy transition attrs / helper bridge / payload bag
  这类影子协议

因此长期路线固定为：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

但这里的 `SpatialPlan`
不再表示“薄兼容壳 + 后段自己猜”，
而是表示
**target-independent 的 virtual spatial/dataflow program**。

`BlackholeModule` / build / codegen / runtime /
`artifact.rt_mod`
都只属于
`ExecutableSpec`
之后的 leaf consumer / delivery 边界，
不再并入长期 layered IR 主链。

当前 leaf compute schema
的收口方向是
generic
`KernelSpec.compute_ops`
typed 数组：
单个 TT-Metal compute instruction family
以
`kind`
区分，
GEMM 只是
`kind=gemm`
entry，
不是所有 compute kernel
的长期字段名或强绑定。
runtime / codegen
只能消费这些 typed entry
和对应 validator
已经接受的字段，
不能从旧
`compute_contract` /
`gemm_contract`
payload family
或 builtin 序列
补回 compute truth。

`2026-04-27`
tile compute preservation
边界再固定一条：
Blackhole 的 compute ops
集合按 TT-Metal API 粒度定义。
凡是 TT-Metal 以 tile compute API
直接表达的语义
（matmul / reduce / unary / binary /
broadcast / copy / pack /
tilize / untilize 等），
必须在
`Normalized Tile TIR`
中保留或规范化，
不能先被 generic scalar lowering
展开后再由
`lower_blackhole_ops.cc`
late TIR idiom matcher
恢复。
`softmax` /
`exp2_affine` /
`row_broadcast_exp2_affine`
这类 workload/composite helper
不是生产 compute op 粒度，
不能进入
`TTComputeOpPlan.operation_name`
或
`KernelSpec.compute_ops`。

`2026-04-29`
tile-compute boundary review
再收紧一条：
删除 composite helper 名字不等于删除 composite 语义。
不允许把
`exp2_affine`
/
row-broadcast division
等复合表达式换皮塞进
`exp2_tile(mode, lhs, rhs, scale, ...)`
或
`mul_tiles_bcast_cols("div", ...)`
这类 leaf-looking payload。
如果 TIR 中已经包含完整计算语义，
正确做法是在
`Normalized Tile TIR`
边界内把表达式规范化为显式 leaf tile compute
statement 序列，
必要时引入显式 logical temp
和由 liveness /
materialization proof
驱动的 copy/materialization。
`TileComputeDAG`
只覆盖这些已经显式存在的 leaf compute nodes；
它不能成为 composite expression lowering owner，
也不能用 source emitter
在 source lowering 阶段生成新的 semantic leaf plan。
每个 DAG source node
必须对应一个 TT-Metal semantic leaf op；
如果某个 source hook
或 projection field
不能满足这个一对一合同，
就必须删除或降为非语义 debug trace。

unsupported / fail-closed 也必须分类：
TIR 语义可表达但 normalizer 未覆盖是
`lowering_missing`；
TT-Metal 有能力但当前 Blackhole builtin /
codegen /
planner 未接是
`backend_op_missing`；
op 已接但资源 /
layout /
sync /
multi-core 条件未证明安全是
`admission_blocked`；
只有经过 TT-Metal primitive coverage
和合法组合审计后仍无法表达，
才是 semantic unsupported。
`unsupported` 不能作为逃避补 leaf op
或补 normalizer 的理由。

`2026-04-25`
compatibility fallback
收束后，
repo HEAD
的 compute leaf truth
已经只从
typed
`TTComputeOpPlan`
/
`KernelSpec.compute_ops`
进入 runtime /
codegen；
旧
`compute_contract` /
`gemm_contract` /
`multi_*_contracts`
不再出现在
`TTProgram.payload -> ExecutableSpec -> runtime`
链路中。
同日后续清理已删除 top-level
`TTProgram.payload`
字段本身；
leaf admission / diagnostic
不能再挂回
`TTProgram`
公共 bag。

`2026-04-24`
TT-Metal runtime model
复查后再固定一条边界：

- direct runtime
  只是
  `ExecutableSpec`
  的一个 leaf execution backend
- 当前 direct runtime
  的 unit-mesh /
  replicated-buffer /
  copy-GEMM admitted subset
  不能作为 codegen /
  export /
  `TTProgram`
  的能力上限
- codegen / export
  的长期目标是
  TT-Metal
  `Program / MeshWorkload / MeshBuffer`
  显式程序模型；
  direct runtime
  只对其中已 admission 的 subset
  允许执行，
  对未 admission 的组合
  fail-closed diagnostic

`2026-04-29`
TT resource-planning
复查后再固定一条边界：

- `TileComputeDAG`
  只能是
  显式 leaf
  `Normalized Tile TIR`
  到 typed compute plans
  之间的 pass-local compute legalization /
  covering model，
  不能从 scalar expression
  恢复 composite semantics，
  不能扩成全局 resource allocator /
  core placer /
  NoC scheduler /
  lifecycle engine
- logical lifecycle
  由
  `SpatialPlan.LiveValue`
  /
  `LiveValueEdge`
  /
  `MaterializationBoundary`
  和 target
  `TTLiveFormPlan`
  /
  `TTMaterializationPlan`
  表达；
  CB ID、
  L1 pressure、
  core placement、
  buffer distribution
  属于
  `TTProgram`
  target realization
  和
  `ExecutableSpec`
  admission，
  不能倒灌回
  `TileComputeDAG`
  或 runtime fallback
- 第一版 resource planning
  应以 typed
  `ResourceDemand`
  /
  `ResourcePressureReport`
  这类从
  `TTProgram`
  /
  `ExecutableSpec`
  可重建的 admission view
  为入口，
  先服务 validator /
  unsupported diagnostic；
  它们不是第五个 IR 层，
  也不能成为新的 bag / payload
- `TileComputeDAG`
  的 DAG-wide fanout /
  materialization reasoning
  只有在投影到
  `ResourceDemand`
  /
  `ResourcePressureReport`
  /
  typed unsupported diagnostic
  并改变 validator /
  admission 结果时才算 production use；
  diagnostic FFI /
  tests
  只能算 foundation，
  不能算完成
- CB allocation
  采用 arch-aware live-interval /
  linear-scan 方向；
  L1/SRAM
  先做 pressure admission，
  不替代 TT-Metal allocator
  的 physical address assignment
- core /
  buffer placement
  应消费
  `TTHardwareModel`
  和显式
  `TTCoreGroup`
  /
  `TTBufferDistributionPlan`
  事实；
  NoC /
  multicast /
  scheduling optimization
  只能在这些 typed resource facts
  稳定后进入

## 2. 第一性原理

对 spatial/dataflow target，
一个算子最终只会落成三类事实：

1. **访存**
   - 从哪里读
   - 搬到哪里
   - 怎么写回
   - 是否跨 core / multicast / gather / remote write
2. **计算**
   - 在 tile 上执行什么 compute builtin
   - operand/result 与 tile-reg / pack / reduction 的关系
3. **通信**
   - 哪些 core 之间交换数据
   - 走哪条 NoC / multicast / remote path
   - 谁等谁、何时可见
   - 哪些 barrier / semaphore / completion / topology 语义需要冻结

编译链必须围绕这三类事实组织，
而不是围绕历史补丁名词组织。

### 2.1 IR-first 编译器纪律

当前链上唯一允许跨阶段承载语义的东西，
只有显式表示层本身：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

这意味着：

- analysis 只能是从当前层派生出的、可失效可重建的临时结果，
  不是协议面
- pass 只能读取当前层，
  并直接 rewrite 成同层或下一层显式表示；
  如果当前层信息不够，就先扩这一层表示
- 任何 helper bag / payload / wrapper / late matcher
  都不能成为长期语义通道
- 任何需要跨阶段保留、被下游依赖、
  且不能在 analysis 失效后由当前层重新推出的内容，
  都说明当前显式表示还不够，
  必须补到 IR 本身，
  而不是继续堆旁路协议

### 2.2 Hardware-Codegen Usefulness Gate

后续任何新设计对象、
算法结构、
pass、
typed field
或 validator
都必须先回答一条问题：

> 它是否让 DSL 写出来的 kernel
> 更可靠或更高效地 lower 到真实 TT-Metal 硬件代码？

这条是主线完成门槛，
不是事后解释。
有效进展必须至少改变下面之一：

- `Normalized Tile TIR`
  的显式 leaf compute /
  access /
  dataflow 表达
- legality /
  covering /
  typed
  `TTComputeOpPlan`
  决策
- live value /
  materialization /
  consumer binding
  的 typed plan
- CB /
  L1 /
  core /
  buffer /
  semaphore
  resource plan
- validator /
  admission
  的 fail-closed diagnostic
- 或删除旧 matcher、
  helper fact、
  payload、
  fallback、
  side-channel

只构造对象、
dump、
shape-only validator、
metadata projection、
测试覆盖、
或引入 paper-like algorithm name，
都只能算 foundation work。
如果一个结构当前只能说明“未来可能有用”，
它只能作为 future candidate 记录，
不能作为 active-chain completion。

## 3. 层间边界

### 3.1 `Normalized Tile TIR`

唯一语义本体。

继续持有：

- tile op
- loop/domain
- predicate
- `BufferLoad / BufferStore`
- address expr
- region/subscript
- loop-carried / dataflow structure
- tile-op 参数

对 Blackhole target，
TT-Metal API 粒度的 tile compute semantics
也必须在这一层保留或规范化：
matmul、
reduce、
unary、
binary、
broadcast、
copy、
pack、
tilize、
untilize
等 leaf API 语义
不能先展开成 scalar loop /
local expression
后再由后段按 workload idiom
恢复。
如果某个 tile expression
需要拆成多个 TT-Metal leaf API，
这个 decomposition
也属于当前层的结构化 normalization，
不是
`TTProgram`
或 runtime/codegen
的 semantic recovery。

只要信息还能由 TIR 稳定表达，
就不允许为下游再造一份旁路语义表示。

### 3.2 `SpatialPlan`

`SpatialPlan` 的新语义是：
**virtual spatial/dataflow program**。

它负责回答：

- 哪些 anchored sub-TIR 构成稳定执行单元
- 单元之间有哪些显式数据流 / carry / reduction / broadcast 关系
- virtual layout / sharding / distribution 语义是什么
- virtual phase / ordering / materialization boundary 是什么
- logical live value
  的 producer / consumer /
  carry / reduction / broadcast
  关系是什么
- logical value
  在跨 unit / phase 被消费前
  需要什么 materialization boundary
- 哪些 hint 经 validate 后进入 planner

长期显式表示对象：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`
- `LiveValue`
- `LiveValueEdge`
- `MaterializationBoundary`

兼容视图：

- legacy compatibility projection

只能作为调试或过渡 projection，
不再承载长期语义。

`SpatialPlan` 不负责：

- TT builtin family
- CB / semaphore / runtime arg
- target block placement
- executable leaf materialization

### 3.3 `TTProgram`

`TTProgram` 是唯一 target realization 表示。

它负责回答：

- mesh / device range /
  submesh / fabric node
  的 physical placement
- buffer distribution
  是 replicated /
  sharded /
  device-local
  还是后续 mesh/fabric
  可见形态
- block / core placement
- kernel family / role
- transport / routing / delivery
- sync / completion / ordering
- ABI / runtime args / accessor binding
- execution / launch order / waves
- physical live form /
  materialization protocol /
  consumer binding

长期显式表示对象：

- `TTMeshPlan`
- `TTBufferDistributionPlan`
- `TTBlockPlan`
- `TTKernelPlan`
- `TTComputeOpPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTLiveFormPlan`
- `TTMaterializationPlan`
- `TTConsumerBindingPlan`

当前代码里的：

- `TTKernel`

当前 P1 实现状态：

- `TTMeshPlan`
  已作为 unit mesh
  owner-truth slice
  进入 `TTProgram`
  和 executable projection
- `TTBufferDistributionPlan`
  已覆盖
  `SpatialPlan.LayoutSpec`
  subject，
  记录 replicated distribution /
  DRAM-L1 memory space /
  ABI accessor coverage
- `TTComputeOpPlan`
  已作为 compute-kind
  owner-truth slice
  进入 `TTProgram`
  和 executable projection；
  当前生成覆盖 GEMM /
  multi-GEMM，
  以及 flash-attn
  compile path 中已选择的
  non-GEMM exact compute builtins。
  `kind`
  表示
  GEMM /
  binary /
  unary /
  reduce
  等 compute family；
  `operation_name`
  表示具体 exact builtin。
  GEMM 以
  `kind=gemm`
  entry 承载 operand binding /
  M-N-K shape /
  tile-block shape /
  accumulator dtype /
  mbarrier binding
- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`

都只允许作为兼容载体或 realization detail 继续存在，
不能替代上面这组长期显式表示边界。

如果当前实现里仍保留：

- `BuildTTProgram`
- `PlanTTBlocks / PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution`

它们也只能是
current `TTProgram`
slice 的构造 /
聚合 /
finalize /
cleanup mechanics，
不是新的长期层边界。

`2026-04-25`
repo HEAD
已删除
`buffer_tile_bridge_specs`
bridge、
contract-family
payload
和
`TTKernel` /
`TTCoreGroup` /
`TTComputeOpPlan` /
operand binding /
`TTComputeSyncPlan`
上的 leaf fallback payload。
top-level
`TTProgram.payload`
也已删除。
`TTMeshPlan` /
`TTBufferDistributionPlan` /
`TTBlockPlan` /
`TTKernelPlan` /
`TTCBPlan` /
`TTTransportPlan` /
`TTSyncPlan` /
`TTSemaphorePlan` /
`TTDstLayoutPlan` /
`TTLiveFormPlan` /
`TTMaterializationPlan` /
`TTConsumerBindingPlan` /
`TTABIPlan` /
`TTExecutionPlan`
已不再暴露 plan-local
`payload`；
仍需保留的信息必须进入 typed fields
或 leaf projection。
同日后续清理已将
`TTKernel.launch_spec` /
`TTKernel.compute_config` /
`TTKernel.per_work_arg_specs`
从 public map/Any 字段
收成 typed
`TTKernelLaunchSpec` /
`TTKernelComputeConfig` /
`TTPerWorkArgSpec`；
leaf `ExecutableSpec`
仍由 projection
编码成 segment map，
但 `TTProgram`
公共 schema
不再用这三个 leaf map
承载 kernel 语义。
`TTABIPlan.runtime_args` /
`common_runtime_args` /
`compile_time_arg_specs` /
`accessors` /
`semaphore_bindings`
也已收成 typed
`TTRuntimeArgSpec` /
`TTCompileTimeArgSpec` /
`TTAccessorSpec` /
`TTSemaphoreBindingSpec`
数组；
ABI leaf map
只在
`ExecutableSpec`
projection boundary
出现。

### 3.4 `ExecutableSpec`

`ExecutableSpec`
是唯一的
leaf execution projection /
runtime-module build contract。

它只负责：

- 冻结 validated `TTProgram`
- 由 canonical writer
  `MaterializeBlackholeExecutable`
  投影
  `tl.blackhole_executable`
  与 `ExecutableSpec`
- 暴露 leaf 可直接消费的
  segment / kernel /
  runtime-arg /
  accessor /
  CB /
  semaphore /
  core /
  execution /
  live-form /
  materialization /
  consumer-binding records
- 基于显式 leaf projection
  做 execution backend 选择 /
  admission gate /
  runtime-module materialization
- 基于同一 projection
  支撑多个 leaf consumer：
  direct runtime、
  TT-Metal codegen/export、
  后续硬件/mesh/fabric
  execution adapter

不再承担：

- target planning
- semantic recovery
- builtin guessing
- fake protocol 补洞

额外边界固定为：

- build / codegen / runtime /
  `BlackholeModule`
  只允许读取
  `tl.blackhole_executable`
  或解析后的
  `ExecutableSpec`
  projection，
  不允许回读
  `TTProgram`
  或 legacy attrs
- `segment_plan.kind`
  与其他 projected leaf records
  是当前 leaf truth；
  如果未来
  `blackhole.segment_kind`
  之类 source marker，
  重新进入 final IR /
  `ExecutableSpec` /
  leaf reader，
  它只能视为 regression，
  不是 `ExecutableSpec`
  成立的前提
- 当前 direct runtime admission
  与 codegen/export gate
  必须拆开：
  direct runtime
  的 admitted support surface
  只属于 leaf execution concern；
  codegen/export
  的 gate
  是 schema completeness /
  emitter capability /
  TT-Metal program legality
- 它可以拒绝某个
  `ExecutableSpec -> execution backend`
  组合
- `direct_runtime_unsupported_reasons`
  这类字段
  也只允许承担
  leaf admission /
  workload gate /
  diagnostic
- 但它**不能**反向收窄
  `TTProgram` 显式语义、
  TT builtin basis、
  或
  `ValidateTTProgram`
  的 legality 边界
- direct runtime
  当前如果仍固定
  `MeshDevice::create_unit_mesh(0)`、
  replicated `MeshBuffer`
  和 interleaved DRAM accessor，
  这些只是该 backend
  的 admission rule，
  不是 TT-Metal program
  emission contract
- `compute_contract` /
  `gemm_contract` /
  `multi_*_contracts`
  这类 contract-family
  字段已退出
  `TTProgram.payload ->
   ExecutableSpec ->
   runtime`
  fallback 链。
  后续新增 compute ABI /
  config /
  epilogue /
  materialization
  事实只能进入
  `TTProgram`
  typed slices
  和
  `ExecutableSpec`
  typed leaf schema
- cleanup 之后的 support-surface
  admission lane：
  direct cast /
  `fragment_fill -> cast -> publish`
  /
  flash-attn direct runtime
  support 工作只允许作为
  live-form /
  materialization admission
  问题推进；
  它的任务级设计见
  `2026-04-23-blackhole-live-form-materialization-admission.md`
- 该 support lane
  不新增 IR 层，
  不允许 runtime/codegen
  通过
  `SeqStmt`
  /
  builtin 序列
  /
  buffer 名
  恢复 producer-consumer 语义；
  logical live value
  必须先进入 `SpatialPlan`
  的
  `LiveValue` /
  `LiveValueEdge` /
  `MaterializationBoundary`，
  materialization boundary
  必须能引用 source 和 target
  logical live value，
  TT physical live form
  必须进入 `TTProgram`，
  leaf materialization
  只能由 `ExecutableSpec`
  冻结后供 runtime/codegen 消费

长期交付链固定为：

```text
tl.tt_program
  -> tl.blackhole_executable
  -> ExecutableSpec
  -> artifact.rt_mod
```

## 4. TT builtin mapping 边界

这次重写最关键的约束只有一条：

> **TT builtin mapping 必须发生在 anchored sub-TIR 仍保留 tile-op、layout、真实 load/store 语义的边界。**

下面提到的 `PlanTTTransport / PlanTTCompute / PlanTTSync / PlanTTABI / PlanTTExecution`
只是当前代码里的构造步骤，
不是新的长期语义层。

长期契约只有一条：

- compute / transport / sync / ABI / execution
  这些 target 语义
  必须落在 `TTProgram`
  的显式 plan 对象里
- planner pass
  只是把当前 IR
  rewrite / lower
  成这些对象的实现手段

具体分成三类 `TTProgram` 语义 slice，
当前分别由对应 planner 构造：

### 4.1 `PlanTTTransport`

输入：

- `BufferLoad / BufferStore`
- address expr
- region
- `DataflowEdge`
- `LayoutSpec`
- `TTBlockPlan`

输出：

- `TTTransportPlan`
  中的 transport / route / delivery / endpoint 语义
- `TTABIPlan`
  中的 accessor / page / runtime-arg / buffer-binding 语义
- `TTExecutionPlan`
  中需要和 transport 对齐的 delivery / launch 约束

### 4.2 `PlanTTCompute`

输入：

- tile op
- layout
- operand/result region
- `ExecutionUnit`

输出：

- `TTKernelPlan`
  中的 exact TT-Metal compute builtin 选择
  - `matmul`
  - `eltwise`
  - `reduce`
  - `sfpu`
  - `copy / pack / untilize`
- `TTComputeOpPlan`
  中的 operand/result binding
- `TTComputeOpPlan`
  中的 tile register / pack-unpack / accumulation / reduction protocol

当前生成覆盖
GEMM / multi-GEMM，
以及 flash-attn compile path
已选择的 non-GEMM
exact compute builtins。
后续新增 compute instruction
必须继续进入
`TTComputeOpPlan.kind`
/
`operation_name`
/
operand binding，
不能只留在 lowered source。

### 4.3 `PlanTTSync`

只负责把 communication 的 completion slice
写入 `TTSyncPlan` / `TTExecutionPlan`：

- ordering
- completion
- barrier / semaphore / global-semaphore relation

补充：

- `route / multicast / remote write / topology`
  属于 `TTTransportPlan + TTExecutionPlan`
- `PlanTTSync`
  不再兼职恢复 compute 或 transport 语义

### 4.4 Pass 实现纪律

这条链上的实现形态也固定为：

- pass 只能在**当前显式表示层**
  （`Normalized Tile TIR` / `SpatialPlan` / `TTProgram`）
  上工作
- 默认实现形态是
  `visitor / matcher / mutator / builder`
  驱动的直接构造或直接改写
- 如果实现上需要一个前置结构遍历，
  它也应留在同一个 `.cc`
  中作为局部 mechanics

明确禁止把这些局部结果升格成：

- 新的 attr bag
- 新的 public analysis wrapper
- 新的 pass-to-pass 语义层
- 新的 runtime/codegen late matcher 恢复层

换句话说：

- `SpatialPlan`
  应由对当前 `Normalized Tile TIR`
  的结构遍历直接构造
- `TTProgram`
  应由对当前 `SpatialPlan`
  和 anchored sub-TIR
  的 planner pass
  直接构造
- `ExecutableSpec`
  应由对当前 `TTProgram`
  的 direct projection 得到

而不是：

```text
current explicit IR
  -> analysis cache / helper bag
  -> another bridge protocol
  -> next explicit IR
```

## 5. Validator 纪律

layered IR 的价值只在于每层都显式承诺：

- 它拥有哪类语义
- 它不拥有哪类语义
- 它与下一层是什么 refinement 关系

因此 validator 是主链对象，不是补丁。

长期 validator set：

- `ValidateSpatialPlan`
  - 检查 execution-unit coverage
  - 检查 dataflow edge endpoint completeness
  - 检查 phase ordering / layout consistency
  - 检查没有 TT noun 泄漏到 `SpatialPlan`
- `ValidateTTProgram`
  - 检查 target 表示 completeness / consistency
  - 检查 exact TT-Metal builtin / transport / sync legality
  - 检查 transport / sync / ABI / execution 闭合
  - 禁止 payload bag 回升为主协议
- `ValidateExecutableSpecProjection`
  - 检查 leaf projection 只来源于 `TTProgram`
  - 禁止 runtime/codegen 自己再补上游 planning 语义
- `ValidateExecutionBackendAdmission`
  - 只在 leaf 边界检查
    某个 backend
    是否接受当前 `ExecutableSpec`
  - backend-specific unsupported
    必须在这里 fail-fast
  - 不能把 backend 当前 support subset
    提升成
    `TTProgram`
    或 builtin legality
  - direct runtime
    和 codegen/export
    必须作为不同 backend /
    consumer
    分别 admission；
    direct runtime
    unsupported
    不能阻断
    schema-complete 的
    TT-Metal codegen/export

fail-closed 纪律固定为：

- 缺 evidence 就 reject / unsupported
- 不再用名字匹配、位置假设、临时分支去补语义
- backend-specific unsupported
  只能停在 leaf execution gate，
  不能回流成上游表示层 / legality 约束

当前 public protocol surface
审查再固定一个更具体的规则：

- 任何被
  projection /
  `ExecutableSpec` parser /
  runtime /
  codegen /
  export wrapper
  读取的 key，
  都已经是 public protocol；
  不能因为它暂存在
  `payload`
  或 helper map
  里就豁免 validator
  和删除标准
- generic schema
  可以用
  `kind`
  variant
  承载 family-specific fields；
  但 workload /
  instruction family
  noun
  不能升级成 top-level
  public field。
  `KernelSpec.compute_ops`
  是合法方向；
  `gemm_contract`
  / `compute_contract`
  / `multi_*_contracts`
  这类 family field
  已从 active chain 删除，
  后续不能作为 compatibility debt
  或测试辅助面重新出现
- buffer /
  operand /
  work-item /
  materialization
  binding
  的 owner truth
  必须是 typed field
  或可由当前 IR 结构稳定推出；
  runtime arg order、
  handle suffix、
  `_local`
  suffix、
  single-output fallback、
  arg-kind priority
  只能作为当前 wrong-now
  residue 定位线索，
  不能继续作为 leaf reader
  合法推断规则
- projection encoder
  必须从 typed fields
  构造 fresh executable maps；
  不能以
  `payload`
  为 seed
  再覆盖 typed 字段。
  需要保留的 diagnostic /
  admission field
  必须显式 allowlist，
  并由 validator
  确认不会成为第二真源

## 6. Fake Protocol 去留规则

审计表中列出的
legacy transition attrs /
helper bridge /
payload bag /
planning seed
都不是长期显式语义。

其中必须明确写死：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  和
  `buffer_tile_bridge_specs`
  已删除。
  logical tile layout
  truth
  只允许进入
  `SpatialPlan.LayoutSpec`
  和
  `TTBufferDistributionPlan`
  的 typed fields，
  再投影到
  `ExecutableSpec`
  的
  `buffer_distribution_plans`
- `blackhole.lowering_requirements` /
  `tl.blackhole_lowering_requirements_seed` /
  `blackhole.cb_requirements`
  这类 broad planning debt
  已退出 repo HEAD active chain；
  后续不能重新引入，
  也不能继续被文档表述成
  `TTProgram`
  的合法输入边界
- `BlackholeLoweringSupportFacts`
  这类 pass 内 analysis 结果
  也不能再伪装成
  `Array<Any>` /
  `Map<String, Any>`
  contract family。
  允许存在的形态只能是
  从当前 TIR /
  `SpatialPlan`
  派生、
  不进入 IR 公共面的
  typed C++ analysis structs；
  如果事实需要跨过当前 lowering
  阶段继续保留，
  必须进入
  `SpatialPlan` /
  `TTProgram`
  typed 表示
- GEMM compute op
  planning 也不能先落成
  pass-local
  `Map<String, Any>`
  seed
  再回填
  `TTComputeOpPlan`。
  `TTComputeOpPlan`
  的输入应来自 typed
  compute-op fact /
  当前 TIR analysis，
  map seed 不能作为旧
  contract-family
  的替身继续存在
- top-level `TTProgram.payload`
  已删除；
  后续不能以
  compatibility fallback
  名义重新引入 planning source

它们的处理纪律固定为：

1. 不扩
2. 不升格
3. 不再写成长期协议
4. 只能被新的显式表示层替换

具体 disposition 见：
`tasks/dev_design/blackhole_first_principles_protocol_audit.md`

## 7. 后段实现审判规则

当前后段实现即使“能跑”，
也不能先被固定成上游设计前提。

判断顺序固定为：

1. 先看表示层边界
2. 再看当前实现是否越权持有上游语义
3. 只有在语义明确属于上游显式表示层时，
   才允许回头补上游 IR / builder logic / validator

明确禁止：

- 先把当前后段实现当成协议
- 再要求 `SpatialPlan`
  去补它需要的语义
- 把“现在能跑”
  当成表示层边界正确的证据
- 把 direct runtime 当前 admitted support surface
  回写成
  TT builtin mapping、
  `TTProgram`
  或
  validator
  的上游合法性边界

如果当前后段实现和第一性原理表示层边界冲突，
优先改后段实现，
而不是回退设计去迎合它。

## 8. 当前 rewrite 方向

当前 rewrite 的长期 closure set
不再围绕“补 fake attr”，
而是围绕下面 4 个表示层边界判断：

1. **中间层重建**
   - 把 `SpatialPlan`
     从薄兼容层
     重写成 virtual spatial/dataflow program
2. **target 表示收口**
   - 把 `TTProgram`
     收正成唯一 physical realization 表示
3. **leaf reader 收口**
   - 让 build / codegen / runtime /
     `BlackholeModule`
     只读 executable projection /
     `ExecutableSpec`
   - leaf reader
     不能再用默认 CB、
     segment kind/core_type 默认、
     accessor `slot`
     兼容键、
     compute operand `host_buffer`
     fallback、
     GEMM tile 维度反推
     或 per-work `value_kind`
     这类旧面恢复语义；
     缺 typed 字段必须 fail-close
   - 将 direct runtime admission
     与 TT-Metal codegen/export
     capability 分离；
     不再让 unit-mesh
     direct path
     决定 broader
     Program / MeshWorkload
     emission
4. **legacy protocol deletion**
   - fake protocol /
     helper residue /
     side channel /
     late matcher family
     只能作为历史 debt
     被删除或审计，
     不能定义新的长期协议

历史架构收口曾按
`Task 1 -> Task 2 -> Task 3 -> Legacy Protocol Deletion`
推进；
当前这些 cleanup workstream
已归档为完成期历史记录。
repo HEAD 的当前任务队列、
blocker
和下一步统一只看
`tasks/progress.md`。

补充：

- `buffer effect / use-role`
- `liveness`

都只是
`Task 1: SpatialPlan Representation Cutover`
里的 preparatory substeps，
不再单独充当顶层 roadmap。

`materialization / source-live-form`
在 cleanup 之后已经重新收束为
support-surface admission lane；
当前只通过
`2026-04-23-blackhole-live-form-materialization-admission.md`
和 `tasks/progress.md`
跟踪，
不再作为单独顶层路线，
也不允许回退到 runtime-only matcher。

当前活动设计 / 约束文档按下面顺序理解：

1. `task0_ir_layering_root_cause.md`
   - 固定根因和显式表示层边界判断
2. `task1_spatial_plan_companion.md`
   - 固定 `Task 1: SpatialPlan Representation Cutover`
   - 文件名里的 `companion`
     只是历史索引；
     架构目标仍是
     `SpatialPlan`
     的直接构造与显式表示
3. `task2_ttprogram_companion_cutover.md`
   - 固定 `Task 2: TTProgram Representation Cutover`
   - 文件名里的 `companion`
     同样只作索引，
     不引入新的 IR 层
4. `task3_runtime_gate_and_workload_cutover.md`
   - 固定 `Task 3: ExecutableSpec / Leaf Reader Cutover`
5. `2026-04-23-blackhole-live-form-materialization-admission.md`
   - 固定 cleanup 之后的
     support-surface admission lane
   - 只定义 live-form /
     materialization /
     runtime admission
     的显式表示边界
6. `2026-04-27-blackhole-tile-compute-preservation.md`
   - 固定 TT-Metal API 粒度
     tile compute semantics
     必须在
     `Normalized Tile TIR`
     中保留 / 规范化
   - downstream scalar-loop
     matcher family
     不能再作为 active compute truth
7. `2026-04-27-blackhole-post-preservation-pass-shrink.md`
   和
   `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md`
   - 固定 tile-compute preservation
     之后的 implementation split /
     helper reuse /
     heavy-pass cleanup
     边界
8. `2026-04-28-blackhole-algorithmic-generalization.md`
   - 固定
     `AccessRegion` /
     graph-backed dependence /
     `LiveValueSSA` /
     TT live-form solver /
     Phase E decision-use cutover
     的算法化重构合同
   - 这些结构的主线价值是 dataflow /
     liveness /
     materialization /
     resource-planning substrate；
     不能把它们写成 compute expression lowering
     的替代方案
9. `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
   - 固定
     `TileComputeDAG` /
     legalizer /
     covering
     的生产迁移 gate
   - `TileComputeDAG`
     只有作为 explicit leaf graph
     上的 pass-local legality /
     fanout /
     materialization /
     resource-demand input
     才有主线价值；
     它不能成为 composite expression lowering owner
  - `2026-04-29`
    boundary repair
    已删除 known composite pseudo-leaf source payload，
    并在
    `Normalized Tile TIR`
    中显式 leaf 化
    `exp2`
    affine
    和 row-broadcast division
    路径；
    后续 DAG production 价值仍按 typed plan /
    resource-demand /
    validator /
    old-branch deletion 是否被真实驱动来判断
10. `blackhole_first_principles_protocol_audit.md`
   - 固定 legacy /
     fake protocol
     的迁移落点、
     validator 责任和删除纪律

归档的
`archive/2026-04-16-blackhole-final-legacy-protocol-cleanup*.md`
只作已完成 cleanup 的历史记录和审计参考；
它们不再是当前活动设计入口，
也不能被用来保留旧 wrapper /
facts /
bag /
payload /
matcher
兼容面。

## 9. 完成判定

第一性原理目标完成，
必须同时满足下面 5 条：

1. **mapping 边界正确**
   - TT builtin mapping
     发生在 anchored sub-TIR 仍保留
     `tile-op / layout / load-store 语义`
     的边界
2. **三类 target 语义都落在 `TTProgram` 的显式 slice**
   - compute:
     `TTKernelPlan`
   - memory-access:
     `TTTransportPlan + TTABIPlan`
   - communication:
     `TTTransportPlan + TTSyncPlan + TTExecutionPlan`
3. **长期语义承载位置收紧**
   - 算法/访存语义只在 `Normalized Tile TIR`
   - virtual spatial/dataflow 语义只在 `SpatialPlan`
   - physical target 语义只在 `TTProgram`
   - `ExecutableSpec`
     只做 leaf projection /
     materialization
4. **leaf writer / reader 边界收紧**
   - `MaterializeBlackholeExecutable`
     是唯一 canonical writer
   - build / codegen / runtime /
     `BlackholeModule`
     只读 executable projection /
     `ExecutableSpec`
   - wrong-now residue
     只允许留在 cleanup /
     audit / progress
     的 debt 表述里，
     不能再回升成长期协议
5. **execution gate 不反向塑形**
   - backend-specific support subset
     只在
     `ValidateExecutionBackendAdmission`
     / leaf execution
     处生效
   - 不再让 runtime 当前限制
     倒逼
     TT builtin basis /
     `TTProgram`
     / upstream legality
   - direct runtime
     当前不支持 mesh /
     sharded buffer /
     fabric collective
     时，
     只能表现为该 backend
     unsupported；
     不能阻止
     `TTProgram`
     和 `ExecutableSpec`
     表达这些 TT-Metal
     target facts
