# Blackhole Live-Form / Materialization Admission Design

## 1. 定位

- **日期**: `2026-04-23`
- **状态**:
  live-form /
  materialization
  owner truth
  已落到
  `TTProgram -> ExecutableSpec`；
  direct runtime admission
  仍等待非 mailbox
  materialization protocol
- **上级设计**: `final_blackhole_backend_redesign.md`
- **适用范围**:
  - direct cast consumer
  - `fragment_fill -> cast -> publish`
  - 后续 flash-attn direct runtime admission

本文不是第二份总体设计文档。
它只把 cleanup task5 之后重新打开的 support 工作
收束到当前长期主链：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

## 2. Review Charter

### 2.1 Review scope

重新定义 direct cast consumer、
`fragment_fill -> cast -> publish`
和 flash-attn direct runtime
的 support 工作边界，
确保后续实现只补显式表示和 verifier，
不恢复 legacy merge/live-form bridge。

### 2.2 Primary questions

1. direct cast consumer 需要的
   source live form
   应由哪一层表示拥有？
2. thread-distributed fragment
   的 logical layout
   和 physical live form
   如何分层表达？
3. runtime/codegen
   何时可以 admitted，
   何时必须保留 queryable unsupported gate？
4. flash-attn direct runtime
   应作为设计驱动，
   还是作为后续 integration payoff？

### 2.3 Non-goals

- 不扩大 legacy external runner /
  legacy merge bridge /
  late matcher 路径
- 不把 flash-attn direct runtime
  提前升级为当前 hard gate
- 不把 `direct_runtime_unsupported_reasons`
  或 payload 字段
  写成 owner truth
- 不新增与
  `SpatialPlan -> TTProgram -> ExecutableSpec`
  并行的 support IR

## 3. 现状判断

cleanup task5 之后，
legacy protocol deletion
已经收口。
剩下的问题不是“旧协议还没删干净”，
而是当前 admitted runtime surface
仍缺少一类显式语义：

> **logical value 当前以什么 live form 存在，
> 以及 consumer 读取它之前是否已经完成 materialization。**

当前 build/source baseline
能够稳定表达若干 fragment / cast / publish
关系，
但 runtime 真执行卡住的根因是：

- logical layout truth
  不等于 physical live-form truth
- thread-distributed fragment
  在 device side
  往往只是 per-lane physical slice，
  不是 full logical fragment
- 后段如果继续从
  builtin 序列、
  `SeqStmt`
  形态、
  buffer 名、
  或旧 merge fallback
  恢复 producer-consumer 语义，
  就会重新违反 IR-first 纪律

因此 support lane 的第一目标
不是让某个 runtime 测试先绿，
而是补上 explicit live-form /
materialization contract。

### 3.1 当前实现快照

本轮实现已完成 owner-truth 部分：

- `TTProgram`
  新增 typed
  `TTLiveFormPlan` /
  `TTMaterializationPlan` /
  `TTConsumerBindingPlan`
  slices，
  由 planner 写入，
  由
  `ValidateTTProgram`
  fail-close
  校验
- `MaterializeBlackholeExecutable`
  将这些 slices
  投影成
  `ExecutableSpec`
  / runtime metadata
  中的
  `live_form_plans` /
  `materialization_plans` /
  `consumer_binding_plans`
- `BufferMaterializationSpec`
  已扩展
  `live_form_kind` /
  `execution_topology_kind` /
  `physical_local_extent` /
  `logical_element_count` /
  `producer_kernel` /
  `materialization_protocol`
- `fragment_fill -> cast -> publish`
  和 GEMM post-merge cast consumer
  均能显式表达
  `thread_distributed_slice`
  经
  `cb_republish`
  materialize 成
  `cb_materialized_tile`

本轮实现没有把
runtime/codegen
改成按 kernel shape /
builtin sequence /
buffer name
恢复语义。

runtime admission 的实际边界也已明确：
当前 `cb_republish`
仍依赖 mailbox-style
device-side CB write pointer transfer。
在 TT-Sim hard execution 中，
该路径会命中
`UnimplementedFunctionality: t_tile_mmio_wr32`。
尝试在 device helper 中直接读取
local CB interface
又会在 TRISC1 链接阶段暴露
`undefined reference to cb_interface`。
因此当前 direct runtime
不应再 hard-skip 或 runtime-only patch，
而应从
`ExecutableSpec`
metadata
产生 queryable
unsupported reason：

```text
thread-distributed cb_republish materialization is not admitted by direct runtime; requires a non-mailbox materialization protocol for compute-thread CB publication
```

后续 admission
必须补非 mailbox
compute-thread CB publication /
materialization protocol，
而不是在 leaf reader
重建 producer-consumer graph。

## 4. 方案取舍

### 4.1 推荐方案：显式 live-form / materialization owner truth

把 producer value
的 live form
和 consumer materialization requirement
写入当前显式表示链。

- `SpatialPlan`
  负责 target-independent
  的 logical value flow /
  producer-consumer edge /
  materialization boundary
- `TTProgram`
  负责 TT-specific
  physical live form /
  per-lane distribution /
  CB republish /
  cast-consumer binding /
  sync 与 ABI 资源
- `ExecutableSpec`
  冻结 runtime/codegen
  leaf reader 可直接消费的
  materialization spec
- runtime/codegen
  只消费 `ExecutableSpec`，
  不做 semantic recovery

这是唯一符合总设计的方案。

### 4.2 拒绝方案：runtime-only patch

直接在 runtime/codegen
按 kernel shape、
builtin 序列、
buffer 名、
或特定 op pattern
修 direct cast / publish
输出，
短期可能让一个 case 通过，
但会把 semantic recovery
放回 leaf reader。

该方案违反：

- IR-first
- leaf reader 只消费 executable projection
- 不基于名字匹配恢复语义
- support surface 不能反向定义
  `TTProgram`
  legality

不采用。

### 4.3 拒绝方案：flash-attn-first

把 flash-attn direct runtime
作为下一阶段主驱动，
会把多个尚未显式化的问题
混成一个 workload failure：

- live form
- materialization
- reduction / broadcast / recurrence
- transport ordering
- CB lifetime
- simulator capability boundary

flash-attn 应作为
live-form/materialization admission
之后的 integration payoff，
不是当前设计驱动。

## 5. 新显式语义对象

### 5.1 `SpatialPlan` 侧：logical live-value boundary

`SpatialPlan`
需要能表达 target-independent 的
logical live-value 关系。

实现可以按现有类型命名落地，
但必须提供以下等价显式对象或字段：

- `LiveValue`
  - logical buffer / value identity
  - producer execution unit
  - logical shape / element type
  - value role:
    `fragment`,
    `accumulator`,
    `cast_source`,
    `publish_source`,
    `consumer_input`
- `MaterializationBoundary`
  - source `LiveValue`
  - target logical consumer value
  - required visibility:
    `same_unit`,
    `next_phase`,
    `published_to_cb`,
    `host_visible_output`
  - logical coverage:
    full tile,
    partial tile,
    row slice,
    grouped slice
- `LiveValueEdge`
  - producer-consumer relation
  - recurrence / carry classification
  - reduction / broadcast classification
  - whether consumer requires logical full value
    or may consume distributed slice

这些对象只回答
logical program meaning。
它们不包含 CB id、
semaphore id、
runtime arg slot、
TT core coordinate。

### 5.2 `TTProgram` 侧：physical live form

`TTProgram`
需要把上面的 logical relation
实现成 TT-specific physical protocol。

实现可以按现有 `TTProgram`
slice 风格命名，
但必须提供以下等价 typed plan
对象或字段：

- `TTLiveFormPlan`
  - logical value identity
  - producer kernel / task index
  - physical form:
    `full_logical_tile`,
    `thread_distributed_slice`,
    `cb_materialized_tile`,
    `host_buffer_tile`
  - execution topology:
    `single_lane`,
    `thread_distributed`,
    `core_distributed`
  - physical local extent
  - logical element count
  - lane / core ownership rule
- `TTMaterializationPlan`
  - source `TTLiveFormPlan`
  - target consumer kernel / task index
  - materialization protocol:
    `none`,
    `direct_register_use`,
    `cb_republish`,
    `untilize_pack_publish`,
    `host_writeback`
  - required CB plan indices
  - required sync plan indices
  - produced live form after materialization
- `TTConsumerBindingPlan`
  - consumer builtin / op role
  - source live form consumed
  - whether consumer accepts distributed slice
  - whether consumer requires full logical tile
  - ABI/runtime-arg binding if it crosses kernel boundary

这些计划属于 `TTProgram`
physical realization，
由 target planner 构造并由
`ValidateTTProgram`
检查。

### 5.3 `ExecutableSpec` 侧：leaf materialization spec

`ExecutableSpec`
只暴露 leaf reader
所需的冻结结果。

实现必须扩展到以下等价 leaf
projection 字段：

- `BufferMaterializationSpec`
  增加 live-form 字段：
  - `live_form_kind`
  - `execution_topology_kind`
  - `physical_local_extent`
  - `logical_element_count`
  - `producer_kernel`
  - `materialization_protocol`
- compute epilogue op
  只引用 materialization spec id /
  source live-form id，
  不携带新的 owner truth
- `direct_runtime_unsupported_reasons`
  继续只作为 admission diagnostic，
  不参与 semantic construction

runtime/codegen
不得从 TIR body /
builtin sequence /
buffer name
恢复这些字段。
字段缺失时应 fail-close
或保留 explicit unsupported reason。

## 6. Validator 合同

### 6.1 `ValidateSpatialPlan`

必须检查：

- 每个跨 execution-unit consumer
  都有可追踪的 `LiveValueEdge`
- 需要 full logical value 的 consumer
  不得只连到未 materialized 的 distributed slice
- recurrence / carry /
  reduction / broadcast
  分类不得由名字推导
- materialization boundary
  的 logical coverage
  与 producer/consumer region 一致

### 6.2 `ValidateTTProgram`

必须检查：

- 每个 `TTConsumerBindingPlan`
  都引用一个存在的 `TTLiveFormPlan`
- consumer 若要求 full logical tile，
  source live form
  必须是 full logical tile
  或存在 preceding `TTMaterializationPlan`
- `thread_distributed_slice`
  必须声明
  execution topology
  和 physical local extent
- `cb_republish`
  必须引用 typed `TTCBPlan`
  和必要 sync plan
- `TTMaterializationPlan`
  不得通过 payload-only 字段
  承载必需协议

### 6.3 `ExecutableSpec` materialization

必须检查：

- leaf spec 中的
  `live_form_kind`
  / `materialization_protocol`
  与 source `TTProgram`
  一致
- admitted direct runtime
  只能接受 runtime/codegen
  已实现的 protocol
- 未实现 protocol
  只进入
  `direct_runtime_unsupported_reasons`
  diagnostic，
  不绕过 validator

## 7. Workload admission 顺序

### 7.1 第一阶段：`fragment_fill -> cast -> publish`

这是最小 admission case。

目标：

- 证明 `thread_distributed_slice`
  和 full logical tile
  的边界被显式表达
- 证明 cast consumer
  读取前的 materialization protocol
  不是 leaf reader 猜出来的
- 使 runtime gate
  从 hard skip /
  source-only assertion
  转成
  `ExecutableSpec`
  驱动的 queryable
  materialization gate；
  后续非 mailbox protocol
  落地后，
  再晋级为 admitted bf16 TT-Sim test

不允许：

- 按 buffer 名或单个 kernel shape 特判
- 在 runtime 里重建 producer-consumer graph

### 7.2 第二阶段：direct cast consumer

目标：

- 支持 GEMM / accumulator
  producer 之后的 cast consumer
- 只有真实 live-in /
  recurrence consumer
  才保留 merge/materialization path
- fresh / preclear-only path
  继续保持 `clear_accum=true`
  direct path

当前验收：

- build/source contract gate
  仍通过
- typed live-form /
  materialization projection
  能说明
  cast consumer
  消费的是哪一种 live form
- direct runtime
  在缺少 admitted protocol
  时给出 explicit
  unsupported reason，
  而不是重新走旧 merge bridge

最终 admission
仍要求：

- bf16 TT-Sim direct runtime
  admitted
- `direct_runtime_unsupported_reasons`
  不再包含该 admitted shape

### 7.3 第三阶段：flash-attn direct runtime

flash-attn
只在前两阶段完成后
作为 integration payoff。

进入条件：

- fragment/cast/publish
  materialization 已 admitted
- direct cast consumer
  已 admitted
- multi-op compute kernel
  所需 reduction /
  broadcast /
  recurrence /
  transport ordering
  都已站在显式
  `SpatialPlan`
  / `TTProgram`
  objects 上

flash-attn 不作为
live-form/materialization
设计正确性的第一证明。

## 8. 实现边界

允许的实现方向：

- 扩 `SpatialPlan`
  或其当前等价显式对象
  以表达 logical live-value relation
- 扩 `TTProgram`
  typed plan 对象
  以表达 physical live form
  和 materialization protocol
- 扩 `MaterializeBlackholeExecutable`
  projection
  以冻结 leaf materialization spec
- 扩 validator
  fail-close
  检查上述 invariants
- runtime/codegen
  只消费新的 `ExecutableSpec`
  字段

禁止的实现方向：

- 新增 `blackhole.*`
  broad attr bag
- 恢复 `tl.internal_tt_*`
  或 legacy resource plan
- 把 required fields
  放进 untyped payload
  当 owner truth
- runtime/codegen
  通过名字、
  builtin 顺序、
  SeqStmt 形态
  推断 live form
- 为 flash-attn
  写 workload-only support path

## 9. 完成判据

本轮 owner-truth closeout
必须同时满足：

1. live-form /
   materialization distinction
   已进入显式 representation
2. 对应 validator
   fail-close
   拒绝缺字段或不一致 protocol
3. `fragment_fill -> cast -> publish`
   从 build/source-only
   晋级到 typed projection +
   explicit materialization
   runtime gate
4. direct cast consumer
   从 build/source-only
   晋级到 typed projection +
   explicit materialization
   runtime gate
5. runtime/codegen
   没有新增 semantic recovery
   或 workload 特判
6. progress / memory /
   task design
   同步记录 flash-attn
   是否仍留在 payoff lane

完整 runtime admission
还必须继续满足：

- 非 mailbox
  compute-thread CB publication /
  materialization protocol
  落地
- `fragment_fill -> cast -> publish`
  和 direct cast consumer
  晋级为 admitted bf16
  TT-Sim direct-runtime
  correctness gate
- 对应
  `direct_runtime_unsupported_reasons`
  从 admitted shape
  中移除

交付完成还需要：

- 相关 build / compile / projection tests
- TT-Sim bf16 admitted runtime tests
  或 explicit unsupported gate
  regression，
  取决于当前 protocol
  是否已进入 admitted support surface
- source scan
  确认没有新增 legacy attr /
  bag /
  wrapper surface
- `git commit`
  和
  `git push`
