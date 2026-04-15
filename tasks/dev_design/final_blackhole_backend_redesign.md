# TileLang Blackhole Backend Redesign

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-16`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 只保留长期架构、层间边界、真源规则、validator 纪律和当前 rewrite 方向

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
- target-independent 的 virtual spatial/dataflow truth 没有被对象化
- 后段只能补出
  legacy transition attrs / helper bridge / payload bag
  这类影子协议

因此长期路线固定为：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec / BlackholeModule
```

但这里的 `SpatialPlan`
不再表示“薄兼容壳 + 后段自己猜”，
而是表示
**target-independent 的 virtual spatial/dataflow program**。

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
   - 哪些 barrier / semaphore / completion / topology truth 需要冻结

编译链必须围绕这三类事实组织，
而不是围绕历史补丁名词组织。

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

只要信息还能由 TIR 稳定表达，
就不允许复制到 companion。

### 3.2 `SpatialPlan`

`SpatialPlan` 的新语义是：
**virtual spatial/dataflow program**。

它负责回答：

- 哪些 anchored sub-TIR 构成稳定执行单元
- 单元之间有哪些显式数据流 / carry / reduction / broadcast 关系
- virtual layout / sharding / distribution truth 是什么
- virtual phase / ordering / materialization boundary 是什么
- 哪些 hint 经 validate 后进入 planner

长期 primary owner object set：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`

兼容视图：

- legacy compatibility projection

只能作为调试或过渡 projection，
不再是长期 primary owner truth。

`SpatialPlan` 不负责：

- TT builtin family
- CB / semaphore / runtime arg
- target block placement
- executable leaf materialization

### 3.3 `TTProgram`

`TTProgram` 是唯一 target realization truth。

它负责回答：

- block / core placement
- kernel family / role
- transport / routing / delivery
- sync / completion / ordering
- ABI / runtime args / accessor binding
- execution / launch order / waves

长期 primary owner object set：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

当前代码里的：

- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`

都只允许作为兼容载体或 realization detail 继续存在，
不能替代上面这组长期 owner 边界。

### 3.4 `ExecutableSpec / runtime / codegen`

只负责：

- 冻结 `TTProgram`
- 投影 `ExecutableSpec`
- build / codegen / runtime / `BlackholeModule` 执行

不再承担：

- target planning
- semantic recovery
- builtin guessing
- fake protocol 补洞

## 4. TT builtin mapping 边界

这次重写最关键的约束只有一条：

> **TT builtin mapping 必须发生在 anchored sub-TIR 仍保留 tile-op、layout、真实 load/store truth 的边界。**

具体分成三类 planning：

### 4.1 `PlanTTTransport`

输入：

- `BufferLoad / BufferStore`
- address expr
- region
- `DataflowEdge`
- `LayoutSpec`
- `TTBlockPlan`

输出：

- `TensorAccessor / Buffer / CB / NoC / multicast / remote endpoint`
  这组 transport truth
- reader / writer kernel 需要消费的
  address / page / runtime-arg-carried accessor truth
- communication 里的 route / topology / delivery truth

### 4.2 `PlanTTCompute`

输入：

- tile op
- layout
- operand/result region
- `ExecutionUnit`

输出：

- TT-Metal compute family
  - `matmul`
  - `eltwise`
  - `reduce`
  - `sfpu`
  - `copy / pack / untilize`
- operand/result binding
- tile register / pack-unpack / accumulation / reduction protocol

### 4.3 `PlanTTSync`

只负责 communication 的 completion slice：

- ordering
- completion
- barrier / semaphore / global-semaphore relation

补充：

- `route / multicast / remote write / topology`
  属于 `PlanTTTransport + PlanTTExecution`
- `PlanTTSync`
  不再兼职恢复 compute 或 transport 语义

## 5. Validator 纪律

layered IR 的价值只在于每层都显式承诺：

- 它拥有哪类 truth
- 它不拥有哪类 truth
- 它与下一层是什么 refinement 关系

因此 validator 是主链对象，不是补丁。

长期 validator set：

- `ValidateSpatialPlan`
  - 检查 execution-unit coverage
  - 检查 dataflow edge endpoint completeness
  - 检查 phase ordering / layout consistency
  - 检查没有 TT noun 泄漏到 `SpatialPlan`
- `ValidateTTProgram`
  - 检查 target owner object completeness / consistency
  - 检查 transport / sync / ABI / execution 闭合
  - 禁止 payload bag 回升为主协议
- `ValidateExecutableSpecProjection`
  - 检查 leaf projection 只来源于 `TTProgram`
  - 禁止 runtime/codegen 自己再补 planning truth

fail-closed 纪律固定为：

- 缺 evidence 就 reject / unsupported
- 不再用名字匹配、位置假设、临时分支去补语义

## 6. Fake Protocol 去留规则

审计表中列出的
legacy transition attrs / helper bridge / payload bag
都不是长期 owner truth。

它们的处理纪律固定为：

1. 不扩
2. 不升格
3. 不再写成长期协议
4. 只能被新 owner object 替换

具体 disposition 见：
`tasks/dev_design/blackhole_first_principles_protocol_audit.md`

## 7. 当前 rewrite 方向

当前 rewrite 不再围绕“补 fake attr”推进，
而是围绕下面 4 个 closure set 推进：

1. **中间层重建**
   - 把 `SpatialPlan`
     从薄兼容层
     重写成 virtual spatial/dataflow program
2. **target owner 收口**
   - 把 `TTProgram`
     收正成唯一 physical realization truth
3. **leaf reader 收口**
   - 让 build / codegen / runtime
     只读 `TTProgram / ExecutableSpec`
4. **legacy protocol 退场**
   - 删除 fake protocol 和 late matcher owner residue

当前实现顺序固定为：

1. **先做 `SpatialPlan owner cutover`**
   - 把 virtual spatial/dataflow truth
     真的对象化
2. **再做 `TTProgram owner cutover`**
   - 把 target realization truth
     从 helper residue 里收回来
3. **再做 `ExecutableSpec / leaf reader cutover`**
   - 让 leaf 只读
     `TTProgram / ExecutableSpec`
4. **最后删 legacy protocol**
   - 只在新 owner truth
     稳定后退场

补充：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

都只是
`SpatialPlan owner cutover`
里的 preparatory substeps，
不再单独充当顶层 roadmap

## 8. 完成判定

第一性原理目标完成，
必须同时满足下面 4 条：

1. **mapping 边界正确**
   - TT builtin mapping
     发生在 anchored sub-TIR 仍保留
     `tile-op / layout / load-store truth`
     的边界
2. **三类语义面各有 owner**
   - compute: `PlanTTCompute`
   - memory-access: `PlanTTTransport + PlanTTABI`
   - communication: `PlanTTTransport + PlanTTSync + PlanTTExecution`
3. **真源位置收紧**
   - 算法/访存 truth 只在 `Normalized Tile TIR`
   - virtual spatial/dataflow truth 只在 `SpatialPlan`
   - physical target truth 只在 `TTProgram`
   - `ExecutableSpec` 只做 leaf materialization
4. **后段不再补语义**
   - runtime / codegen / build
     不再做 late recovery
   - fake protocol / payload bag / matcher
     不再承担 owner 职责
