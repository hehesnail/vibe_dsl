# Task 0: Root Cause and Rewrite Direction

## 基本信息

- **文档角色**: 当前 Blackhole layered IR 根因诊断文档
- **当前状态**: `2026-04-16` 活动设计文档
- **任务链位置**: `Task 0`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 根因结论

当前 Blackhole 后端的问题不是“后段 matcher 不够聪明”，
而是下面三件事同时发生：

1. **target builtin 选择放得太晚**
   - tile op、layout、真实 `load/store`
     已经被打碎
2. **`SpatialPlan` 没有真正立起来**
   - 当前 `SpatialPlan`
     只承接一层 legacy 兼容标签
   - virtual spatial/dataflow truth 没有对象化
3. **后段被迫长出影子 IR**
   - 一串 legacy transition attrs / helper bridge / payload bag

一句话：

> **当前真正缺的不是更多后段 contract，而是 `Normalized Tile TIR` 和 `TTProgram` 之间的 virtual spatial/dataflow layer。**

## 2. 研究结论怎么落到仓库里

这次重设计不是拍脑袋换名词，
而是直接受下面几组论文约束：

### 2.1 `Dato / Revet / SPADA`

共同结论：

- task / stream / layout / routing / async ordering
  不是 emitter 尾部细节
- virtual dataflow program
  必须先显式化
- backend 不应反向恢复前段语义

对仓库的直接含义：

- `SpatialPlan`
  必须承接 virtual spatial/dataflow truth
- 审计表列出的 fake/legacy protocol
  都不是长期协议

### 2.2 `TL / T2S`

共同结论：

- “算什么”
  和
  “怎么 spatially organize / map / realize”
  必须拆开
- tile-level truth
  不能等打平后再恢复

对仓库的直接含义：

- `Normalized Tile TIR`
  持有算法与访存真相
- `SpatialPlan`
  持有 virtual mapping / layout / phase / dataflow truth
- `TTProgram`
  承接 TT-specific physical realization

### 2.3 `MLIR / ScaleHLS / K-CIRCT / Alive2 / Abstract Interpretation / CGC`

共同结论：

- 多层 IR 的意义在于
  不同层解决不同合法性和优化问题
- 每层都必须有显式 contract
- validator 是主链对象
- 缺证据时必须 fail-closed

对仓库的直接含义：

- `ValidateSpatialPlan`
  必须独立存在
- `ValidateTTProgram`
  不能继续只看 payload bag
- build/codegen/runtime
  不能再补 planning truth

## 3. 当前代码现实为什么一定会膨胀

一旦中间 spatial/dataflow owner layer 不存在，
系统就只能做四种事：

1. 看 loop 形状
2. 看 buffer 读写
3. 看零散 attr
4. 看局部 matcher

于是自然长出这些东西：

- 审计表列出的
  legacy transition attrs / helper wrapper / payload bag

这些对象的共同问题不是“名字不好”，
而是它们都在
**替代本该在中间层显式冻结的 truth**。

## 4. 正确的 owner 边界

### 4.1 `Normalized Tile TIR`

owner：

- tile op
- `BufferLoad / BufferStore`
- address expr
- region / predicate / loop/domain
- loop-carried / dataflow structure

### 4.2 `SpatialPlan`

owner：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`

不再只是：

- legacy compatibility projection

这两个旧对象可以保留为兼容视图，
但不再代表长期主语义。

### 4.3 `TTProgram`

owner：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

### 4.4 `ExecutableSpec`

只做 leaf projection，
不再是第二真源。

## 5. 对当前代码的整改方向

整改方向固定为：

1. **重写 `SpatialPlan`**
   - 从薄兼容层
     改成 virtual spatial/dataflow program
2. **让 `TTProgram` 收回 target owner**
   - 后段停止依赖 fake protocol
3. **让 leaf readers 只读 canonical owner truth**
   - build / codegen / runtime
     停止读 legacy gate attrs
4. **把 legacy attr 当债，不当资产**
   - 只允许 shrink / delete
   - 不允许再升格

## 6. 当前判断

当前代码现实不是：

- `SpatialPlan` 已经站稳
- 只是 `TTProgram` 还不够厚

而是：

- `SpatialPlan` 过薄
- `TTProgram` 又被迫承担恢复和 bridge 职责
- leaf readers 还在消费 fake protocol

所以这轮 rewrite 的正确起点不是
“继续补后段 case”，
而是：

> **先把中间层立起来，再把 target 层和 leaf 层收回各自边界。**

补充纪律：

- 不能先把当前后段实现固定成前提，
  再要求前面去适配
- 后段如果在 owner 边界上越权，
  应优先改后段，
  不是先给前面补 attr / 补 matcher / 补过渡 truth
- 如果当前证据不足，
  应优先补上游 IR / owner object / builder logic / validator，
  不是先长一层 facts bag
