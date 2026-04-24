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
  以及它与 cleanup task3/task4/task5 的关系
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

## 1. 目标

`ExecutableSpec`
不是新的 planning 层，
也不是 `TTProgram.payload`
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

- cleanup task3/task4/task5
  只是把 repo HEAD
  收回到这个合同上的执行切片
- 它们不是新的表示层，
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
- typed compute operation records
  such as
  `KernelSpec.compute_ops`
  entries

这里的 `segment_plan`
已经是 leaf truth，
不是给 runtime
再去切 kernel body
的提示词。

### 3.3 execution resources

它应编码：

- CB config
- core plan
- semaphore plan
- execution-time 可直接读取的
  launch/resource records

这些事实已经处在
TT-specific 显式程序构造边界，
对齐的是 leaf build /
runtime / codegen 消费面，
不是上游 planner 再补一层的理由。

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

## 5. Wrong-Now Residue 与 Cleanup Debt

下面这些东西
必须明确写成
**wrong now, delete later**
或 **leaf compatibility debt**，
不能写成
`ExecutableSpec`
的长期 owner truth。

### 5.1 `buffer_tile_bridge_specs` 不是新层

如果 repo HEAD
里仍保留：

- `buffer_tile_bridge_specs`
- `tl.blackhole_logical_buffer_tile_bridge_specs`

它们的正确口径只能是：

- task1 残留的 narrow cleanup exception
- task3 仍活着的
  leaf compatibility residue
- projection schema
  上待删的 bridge debt

它们不是：

- `ExecutableSpec`
  的长期字段族
- TT-Metal runtime contract
- codegen/runtime
  再造出来的新表示层

### 5.2 `compute_contract` / `gemm_contract` family 只是 leaf compatibility debt

如果 repo HEAD
里仍保留：

- `compute_contract`
- `multi_compute_contracts`
- `gemm_contract`
- `multi_gemm_contracts`
- `compute_epilogue_ops`

它们的正确口径只能是：

- current leaf/runtime compatibility surface
- admitted direct-runtime gate
- 尚未删完的
  contract-family residue

这里必须特别写清楚：

- `compute_contract <- gemm_contract`
  fallback
  如果还活着，
  它只是 live compatibility debt
- `compute_contract`
  缺席时
  从 `multi_compute_contracts`
  或 `gemm_contract`
  恢复语义，
  也只是 live compatibility debt
- `MaterializeBlackholeExecutable`
  如果仍直接从
  `TTProgram.payload`
  复制这些字段，
  这只是 forced leaf debt
  的当前实现形态，
  不是 canonical writer
  的长期合同
- runtime /
  `BlackholeModule`
  如果仍通过
  `compute_contract <- multi_compute_contracts <- gemm_contract`
  选择 compute truth，
  这只能是
  wrong-now fallback；
  required end-state
  是 runtime
  只读
  `ExecutableSpec`
  typed compute /
  kernel /
  materialization
  records，
  缺字段时 fail-close，
  不再从旧 contract family
  推断
- 当前第一轮 typed replacement
  是
  executable
  `KernelSpec.compute_ops`
  数组；
  GEMM 只是其中
  `kind=gemm`
  entry，
  不是 compute kernel
  的通用字段名。
  direct runtime
  已改从该 entry
  读取 GEMM
  shape /
  dtype /
  transpose /
  work-decomposition /
  unsupported mbarrier gate
  truth；
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

cleanup task5 之后重新打开的
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

### 5.4 `segment_plan` 是 projection truth，`blackhole.segment_kind` 是 task4 debt

`segment_plan`
属于 executable projection
的一等 leaf 字段。

但如果 repo HEAD
里 runtime 仍保留：

- `SegmentBodyExtractor`
- `blackhole.segment_kind`
  body slicing

正确口径只能是：

- task4 尚未删完的
  leaf-local residue
- 当前实现为了切 body
  仍保留的 wrong-now path

它不是：

- `ExecutableSpec`
  需要 source-level marker
  才能成立的证据
- TT-Metal target model
  对 source marker
  的要求

### 5.5 防御性 fallback 不是长期 reader 合法性

当前实现里如果还存在：

- `TTProgram payload fallback`
  报错文案
- 对 top-level payload
  或 `work_linear_id`
  恢复 planning 语义的
  防御性 reject

这只能说明：

- repo HEAD
  还在防守旧路径
- leaf readers
  已经不再把那条路径
  当成合法 owner truth

不能反过来写成：

- 旧 fallback
  仍是允许存在的 reader 合同

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

- `gemm_contract`
  / `compute_contract`
  / `multi_*_contracts`
  仍在 projection、
  `ExecutableSpec`
  和测试里作为 public field；
  它们必须被
  typed compute-op /
  materialization /
  ABI schema
  完整替换后删除
- `KernelSpec.compute_ops`
  是正确方向，
  但 GEMM entry
  的 operand truth
  不能继续从 reader/writer
  runtime arg order
  恢复；
  operand binding
  必须来自 compute op
  或显式 ABI 绑定
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

当前 direct runtime /
codegen 的 admitted support surface，
只属于 leaf execution concern。

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
- TT-Sim `fp16`
  simulator capability boundary

因此：

- `flash-attn`
  direct runtime
  不是当前 cleanup hard gate
- unsupported workload
  不能反过来重写
  `ExecutableSpec`
  合同
- workload breadth
  不能拿来掩盖
  reader boundary
  还没收紧

## 7. 与 Cleanup Task 的关系

### 7.1 cleanup task3

cleanup task3
负责删除：

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
负责删除：

- planner / runtime
  仍残留的
  `blackhole.segment_kind`
  marker path
- `segment_plan.kind`
  与 source-level marker
  之间的错位 residue

所以 task3 文档里必须承认：

- `segment_kind`
  仍可能是 live residue
- 但它不属于
  `ExecutableSpec`
  的 owner truth

### 7.3 cleanup task5

cleanup task5
负责最终 convergence /
verification / residue scan /
交付 gate，
不是新的语义 owner。

因此：

- task3 文档
  必须先写清楚
  真正的 leaf boundary
- task5
  只负责证明
  当前实现
  是否已经收回到这条边界

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
   只被明确记录为 debt，
   不再被文档合法化成长期字段边界；
   `compute_contract` /
   `gemm_contract` /
   `multi_*_contracts`
   退出
   `TTProgram.payload -> ExecutableSpec -> runtime`
   fallback 链之前，
   leaf contract-family deletion
   不能被视为完成
6. backend admission
   只停在 leaf execution gate，
   不再反向塑形
   `TTProgram`
   或更早层语义
7. JIT / export /
   execution backend adapter
   的最终交付
   站在
   `artifact.rt_mod`
   或等价 runtime artifact
   上，而不是回读 planner residue
