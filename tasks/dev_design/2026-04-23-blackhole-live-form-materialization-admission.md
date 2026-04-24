# Blackhole Live-Form / Materialization Admission Design

## 1. 定位

- **日期**: `2026-04-23`
- **状态**:
  live-form /
  materialization
  的 TT physical /
  leaf owner truth
  已在 admitted case
  落到
  `TTProgram -> ExecutableSpec`；
  `SpatialPlan`
  侧的 logical live-value /
  materialization boundary
  仍是 required end-state，
  不能被
  TTProgram-only recovery
  代替；
  direct runtime admission
  已有
  `pack_thread_direct_store`
  与
  `pack_tile`
  两类非 mailbox
  publication protocol；
  更宽 live-in /
  workload payoff
  继续按显式 IR
  扩支持面
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
而是 admitted runtime surface
必须继续围绕一类显式语义
扩展：

> **logical value 当前以什么 live form 存在，
> 以及 consumer 读取它之前是否已经完成 materialization。**

早期 build/source baseline
能够稳定表达若干 fragment / cast / publish
关系，
但 runtime 真执行曾卡住的根因是：

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

本轮实现只完成当前 admitted cases
的 TT physical /
leaf owner-truth
部分：

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
  `materialization_protocol` /
  `publication_protocol`
- `fragment_fill -> cast -> publish`
  和 GEMM post-merge cast consumer
  均能显式表达
  `thread_distributed_slice`
  经
  `cb_republish`
  materialize 成
  `cb_materialized_tile`

这不等于
`SpatialPlan`
logical live-value
表示已经完成。
当前 admitted case
仍主要由当前 TIR
结构事实 /
typed tileop contract /
pass-local analysis
进入
`TTProgram`
typed slice；
后续一旦 support surface
需要跨 execution-unit /
phase 保留 logical live-value
distinction，
必须先补
`SpatialPlan`
的一等
`LiveValue` /
`LiveValueEdge` /
`MaterializationBoundary`
对象和
`ValidateSpatialPlan`
检查，
不能继续让
`PlanTT*`
或 leaf reader
自己恢复 logical producer-consumer
关系。

本轮实现没有把
runtime/codegen
改成按 kernel shape /
builtin sequence /
buffer name
恢复语义。

runtime admission 的实际边界也已明确。
早期 `cb_republish`
只会降成 mailbox-style
device-side CB write pointer transfer；
在 TT-Sim hard execution 中，
该路径会命中
`UnimplementedFunctionality: t_tile_mmio_wr32`。
尝试在 device helper 中直接读取
local CB interface
又会在 TRISC1 链接阶段暴露
`undefined reference to cb_interface`。
因此 direct runtime admission
不能靠 hard-skip 或 runtime-only patch，
必须从
`ExecutableSpec`
metadata
区分 queryable
unsupported reason
和已实现的非 mailbox
publication protocol。

当前 admission 的协议分层是：

- `materialization_protocol`
  继续表示 consumer-visible 结果；
  `cb_republish`
  表示 produced live form
  是 CB 中可消费的 logical tile
- `publication_protocol`
  表示 device-side 如何把 producer live form
  写入该 CB
- admitted direct runtime
  只接受非 mailbox
  publication protocol；
  mailbox write-pointer transfer
  必须保留 explicit unsupported reason

当前已 admitted 的非 mailbox
publication protocols 有两类：

- `pack_thread_direct_store`
  用于 planner 已证明
  producer value 是常量 full-tile fill /
  cast publish
  的 case：
  PACK thread 在
  `cb_reserve_back`
  之后直接写 reserved CB page，
  再用
  `cb_push_back`
  发布可见性
- `pack_tile`
  用于 GEMM post-merge
  direct cast consumer
  的 zero-preclear
  full-tile case：
  merge 侧从 partials CB
  copy 到 DST register，
  再把同一个 DST tile
  pack 到 materialized
  bf16 target CB

两者都不是 leaf reader
按 kernel shape 猜出来的 fallback；
必须由
`TTMaterializationPlan`
投影到
`ExecutableSpec`
后才能 admitted。
该证明必须来自当前 IR
中的 producer fact
和 typed materialization contract；
如果同一 source buffer
在 fill 之后又被 matmul /
merge /
add /
reduction /
scalar update /
cast 等 producer 写入，
constant-fill fact
必须失效。
GEMM post-merge
`pack_tile`
admission
还必须看到当前 IR
zero-preclear fact，
不能穿过非零 live-in merge
继续作为
publication admission
依据。

非 constant、
非 DST-register-backed
或缺少 zero-live-in proof
的 arbitrary local slice
仍不得靠 mailbox helper admitted。
host output copy
在读取 materialized bf16 output
时也必须消费
`BufferMaterializationSpec.live_form_kind`，
不能把外部 output
机械按 GEMM accumulator
`float32`
dtype 校验。

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

当前实现如果尚未提供这些对象，
该缺口必须按
`missing explicit representation`
处理：

- 当前 admitted case
  只能依赖可由当前 TIR
  和 typed tileop contract
  直接重算的局部事实
- 任何不能在 analysis 失效后
  从当前 TIR
  稳定重算、
  且下游 admission /
  ABI /
  materialization
  仍需要的 distinction，
  必须先进入
  `SpatialPlan`
- 不允许把
  `buffer_materialization_contracts`
  /
  flow contracts
  /
  body-order matcher
  写成替代
  logical live-value
  IR

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
  - publication protocol:
    `mailbox_write_ptr`,
    `pack_thread_direct_store`,
    `pack_tile`,
    `none`
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
  - `publication_protocol`
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
- `cb_republish`
  必须声明
  `publication_protocol`
  且 direct runtime admission
  只能接受已实现的非 mailbox protocol
- `TTMaterializationPlan`
  不得通过 payload-only 字段
  承载必需协议

### 6.3 `ExecutableSpec` materialization

必须检查：

- leaf spec 中的
  `live_form_kind`
  / `materialization_protocol`
  / `publication_protocol`
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
- 使 `cb_republish`
  携带
  `pack_thread_direct_store`
  publication protocol，
  并晋级为 admitted
  bf16 TT-Sim direct runtime test

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
  对 zero-preclear
  full-tile
  GEMM post-merge
  cast consumer
  使用
  `publication_protocol=pack_tile`
  admitted
- `direct_runtime_unsupported_reasons`
  不再包含该 admitted shape
- 无 zero-preclear /
  非零 live-in
  仍保留 explicit
  unsupported reason，
  而不是重新走旧 merge bridge

### 7.3 第三阶段：flash-attn direct runtime

flash-attn
只在前两阶段完成后
作为 integration payoff。

进入条件：

- fragment/cast/publish
  materialization 已 admitted
- direct cast consumer
  当前 zero-preclear
  full-tile shape
  已 admitted；
  更宽 live-in
  组合仍需显式协议后
  再进入 admitted surface
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

本轮 TT physical /
leaf closeout
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
  的当前 admitted shapes
  晋级为 admitted bf16
  TT-Sim direct-runtime
  correctness gate
- 对应
  `direct_runtime_unsupported_reasons`
  从 admitted shape
  中移除
- 未 covered 的 live-in /
  fragment materialization
  组合继续 queryable
  unsupported，
  不静默错跑

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
