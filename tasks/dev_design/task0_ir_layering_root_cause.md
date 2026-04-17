# Task 0: Root Cause And Rewrite Direction

## 基本信息

- **文档角色**: 当前 Blackhole layered IR 根因诊断文档
- **当前状态**: 活动设计基线
- **任务链位置**: `Task 0`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 本文档定义根因和 rewrite 方向
- 它不是当前 repo HEAD 状态文档
- 当前实现状态统一以 `tasks/progress.md` 为准

## 1. 根因结论

当前 Blackhole 后端的问题不是“后段 matcher 不够聪明”，
而是下面三件事同时发生：

1. **target builtin 选择放得太晚**
   - tile op、layout、真实 `BufferLoad / BufferStore`
     的语义边界已经被打碎
2. **`SpatialPlan` 这层显式表示没有真正立起来**
   - 当前中间层过薄，
     无法稳定承接
     virtual spatial/dataflow 语义
3. **下游阶段被迫恢复语义**
   - 于是长出
     legacy transition attrs / helper bridge / payload bag
     这类影子协议

一句话：

> **当前真正缺的不是更多后段 contract，而是 `Normalized Tile TIR` 和 `TTProgram` 之间缺失了一层稳定的 virtual spatial/dataflow 表示。**

## 2. 这件事在分层 IR 上到底意味着什么

### 2.1 `Normalized Tile TIR`

这一层必须保留：

- tile op
- `BufferLoad / BufferStore`
- address expr
- region / predicate / loop/domain
- loop-carried / dataflow structure

它负责承载算法与访存语义，
但不负责：

- target block placement
- TT builtin family
- sync / ABI / execution realization

### 2.2 `SpatialPlan`

这一层必须把 target-independent 的
virtual spatial/dataflow 语义显式化。

它应当回答：

- 哪些 anchored sub-TIR 构成稳定执行单元
- 单元之间有哪些显式数据流 / carry / reduction / broadcast 关系
- virtual layout / sharding / distribution 关系是什么
- virtual ordering / materialization boundary 是什么

如果这些东西不在这一层显式存在，
下游就只能继续靠名字、attrs、局部 matcher
或 payload bag 去猜。

### 2.3 `TTProgram`

这一层必须承接 TT-specific physical realization：

- block placement
- kernel kind / builtin selection result
- transport / routing / delivery
- sync / completion / ordering
- ABI / runtime args / accessor binding
- execution / launch order / waves

它不能继续承担：

- 恢复 target-independent spatial/dataflow 语义
- 替上游补中间层抽象

### 2.4 `ExecutableSpec`

这一层只做 leaf projection 和执行物化，
不是第二份长期语义表示。

build / codegen / runtime
只能消费这一层的投影结果，
不能回头恢复 planning 语义。

## 3. 为什么当前代码一定会膨胀

一旦缺失中间显式表示层，
系统就只能依赖：

1. loop 形状
2. buffer 读写
3. 零散 attr
4. 局部 matcher

于是自然会长出：

- legacy transition attrs
- helper bridge
- payload bag
- workload-specific lowering residue

这些对象的共同问题不是“名字不好”，
而是它们都在
**替代本该由显式表示层承载的语义**。

## 4. 论文和参考输入真正支持的是什么

这次重设计不是拍脑袋换名词，
而是直接受下面几类结论约束：

- `Dato / Revet / SPADA`
  - virtual dataflow / routing / ordering
    不是 emitter 尾部细节，
    必须先显式化
- `TL / T2S`
  - “算什么”
    和
    “怎么 spatially organize / realize”
    必须拆开
- `MLIR / Alive2 / Abstract Interpretation`
  - 多层 IR 的意义在于
    每层承接不同合法性和优化问题
  - validator 是主链对象
  - 缺证据时必须 fail-closed

对仓库的直接含义是：

- `SpatialPlan`
  必须是稳定的中间显式表示
- `TTProgram`
  必须是稳定的 target realization 表示
- build/codegen/runtime
  不能再补 planning 语义

## 5. 正确的 rewrite 方向

整改方向固定为：

1. **立起 `SpatialPlan`**
   - 从薄兼容层
     改成 virtual spatial/dataflow representation
2. **把 `TTProgram` 收回到 target realization 边界**
   - 停止依赖 fake protocol
3. **把 leaf readers 收回到 `ExecutableSpec`**
   - build / codegen / runtime
     停止读 legacy gate attrs
4. **把 legacy protocol 当债，不当资产**
   - 只允许 shrink / delete
   - 不允许升格

## 6. IR-First 纪律

从这个根因可以直接推出下面几条纪律：

1. 语义只能存在于显式表示层本身
2. analysis 只能是从当前 IR 派生的、
   可失效、可重算的临时结果
3. 如果下游需要的信息当前 IR 没有，
   结论只能是扩 IR / 扩显式对象 /
   扩 validator / 显式 unsupported
4. 不能先把当前后段实现固定成前提，
   再要求上游去适配
5. 后段如果在表示层边界上越权，
   应优先改后段实现，
   不是先给前面补 attr / matcher / facts bag

## 7. 与后续文档的关系

- `task1_spatial_plan_companion.md`
  负责定义 `SpatialPlan` 这层表示的合同
- `task2_ttprogram_companion_cutover.md`
  负责定义 `TTProgram` 这层表示的合同
- `task3_runtime_gate_and_workload_cutover.md`
  负责定义 leaf reader / workload cutover 合同
- `tasks/progress.md`
  负责记录 repo HEAD 当前状态和下一步
