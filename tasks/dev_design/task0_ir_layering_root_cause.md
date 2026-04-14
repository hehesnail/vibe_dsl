# Task 0: Root Cause and Rewrite Direction

## 基本信息

- **文档角色**: 当前 Blackhole 后端根因诊断文档
- **当前状态**: `2026-04-14` 活动设计文档
- **任务链位置**: `Task 0`；解释为什么必须从旧链切到新的
  `TIR -> SpatialPlan -> TTProgram -> ExecutableSpec` 主链
- **定位**: 不是第二份总体设计；只回答
  “旧方案为什么会反复长出历史包袱，以及新的路线为什么必须这么切”
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 根因结论

当前 Blackhole 后端的根因不是“少几个 matcher”或者“少一层 contract”，
而是下面三件事同时发生：

1. **target builtin 选择放得太晚**
   - tile op、layout、真实 `load/store` 关系已经被打散
   - 后段只能从 lowered loop、bridge attr、局部 pattern
     去恢复“原来这是什么算子、怎么搬数据、怎么同步”
2. **把 target planning 建模成了 side contract**
   - `row_*`
   - `broadcast_sources`
   - `index map / access pattern`
   - `buffer_distribution_contract`
   - 以及各种 bridge attr / payload bag
   这些都不是本质对象，只是晚期恢复失败后长出来的补丁
3. **把过渡产物当成了长期架构**
   - `blackhole.work_decomposition`
   - `blackhole.compute_regions`
   - `blackhole.pipeline_stages`
   - `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
   这些可以作为迁移时期的过渡残留，
   但不能再被文档写成长期 owner 链的一部分

一句话：

> **旧路线的问题不是“后段 matcher 不够聪明”，而是 target mapping 本来就不该在那里发生。**

## 2. 第一性原理

对 tile-based spatial hardware，算子最终只会落成三类东西：

1. **访存**
   - 从哪里读
   - 读到哪个 tile / 哪个 L1 / 哪个 CB
   - 怎么写回
   - 是否跨 core / multicast / gather / remote write
2. **计算**
   - 在 tile 上执行什么 compute builtin
   - 需要什么输入 tile / 输出 tile / tile register 语义
3. **通信 / 同步**
   - 数据什么时候可见
   - 谁等谁
   - 何时需要 barrier / semaphore / completion relation

因此，对 Blackhole 的长期分层，必须正对这三类事实：

- `TIR`
  持有算子语义本身
- `SpatialPlan`
  只决定局部闭包、切分和边界
- `TTProgram`
  把访存 / 计算 / 通信 具体化成 target plan
- `ExecutableSpec`
  只冻结并执行

## 3. 从 TileLang GPU 路线得到的直接启发

TileLang GPU 路线的成熟点不在“API 长得像 GPU”，
而在于它把 builtin 选择放在了正确边界：

- tile op 还活着
- layout 还活着
- operand/result region 还活着
- 真实 `load/store` 关系还活着

也就是说，GPU 并不是先把一切打平，
再靠后段把 `gemm/copy/reduce` 猜回来；
而是直接在 tile-op lowering 边界完成 target-aware 选择。

Blackhole 需要吸收的不是 GPU 的 noun，
而是 GPU 的 **mapping boundary**。

## 4. 从 TT-Metal 得到的直接约束

TT-Metal 本地 repo 和官方 API 都说明了同一件事：

- **Compute** 是一组明确的 builtin family
  - `matmul`
  - `eltwise`
  - `reduce`
  - `sfpu`
  - `copy / pack / untilize`
- **Data Movement** 不是一个抽象 pattern，
  而是一组协议组合
  - `TensorAccessorArgs / TensorAccessor`
  - `cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front`
  - `noc_async_read/write/(multicast)`
  - `semaphore`

所以对我们来说：

- `reduce` 就是 `reduce`
- `broadcast` 就是 `broadcast`
- `direct / indirect / paged / sharded`
  也不是独立 semantic class，
  而是地址表达式和 transport protocol 的不同 realization

这意味着：

> **不能再把 data movement 另抽成 `access pattern / index map` 一层，也不能把 compute 另抽成 `row_*` 一层。**

## 5. 旧路线为什么一定会膨胀

一旦 builtin 选择放到太晚的边界，后段就只能做四种事：

1. 看 loop 形状
2. 看 buffer 读写
3. 看零散 attr
4. 看局部 matcher

于是系统自然会长出这些东西：

- `row reduction / row broadcast`
- `grouped_rows / row_state`
- `broadcast_sources`
- `buffer_distribution_contracts`
- `compute_regions / lowering_requirements`
- `PlanTTKernelABI` 里的恢复逻辑

这些对象的问题不只是名字不好，
而是它们都在试图用后段拼出来的局部事实，
替代原本应该在前面就冻结好的 target mapping。

所以旧路线不是“还不够通用”，而是**方向本身就错了**。

## 6. 正确的 owner 边界

长期主链应该只保留下面这些 owner：

### 6.1 `TIR`

只要信息还能由 TIR 本身稳定表达，就不允许复制到 companion：

- tile op
- loop/domain
- predicate
- `BufferLoad / BufferStore`
- address expr
- region / subscript
- tile-op 参数

### 6.2 `SpatialPlan`

`SpatialPlan` 只回答：

- 哪些 anchored sub-TIR 构成一个局部执行闭包
- 闭包之间的稳定 boundary 是什么
- 哪些 frontier 可以切
- 哪些 hint 经过 validate 可以进入 planner

它不负责：

- access pattern
- index map
- TT builtin family
- CB / semaphore / runtime arg

### 6.3 `TTProgram`

`TTProgram` 才是 target realization owner。

它至少需要承接：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

这里最关键的是两类 planning：

- `PlanTTTransport`
  - 从 `BufferLoad / BufferStore + ClosureBoundary + TTBlockPlan`
    选择 `TensorAccessor + CB + NoC + semaphore/mcast`
- `PlanTTCompute`
  - 从 tile op、layout、operand/result region
    选择 compute builtin family

## 7. 新路线下什么必须删除

下面这些都不再是长期设计对象：

- `SemanticProgram`
- `SpatialProgram` 作为独立 execution-bearing IR
- `row_*`
- `broadcast_sources`
- `index map`
- `access pattern`
- `buffer_distribution_contract`
- 任何只为 late matcher 服务的 side contract

下面这些只允许作为**迁移期残留**存在：

- `blackhole.work_decomposition`
- `blackhole.compute_regions`
- `blackhole.pipeline_stages`
- `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`

它们的退出方向已经固定：

- 不扩
- 不升格
- 不再写成长期 owner
- 只允许被真正的 `PlanTTTransport / PlanTTCompute / PlanTTSync / ...`
  替换掉

## 8. 对当前代码状态的重新判断

当前代码并不是已经站在最终架构上，
而是处于一个**过渡实现仍然太重**的状态：

- 外层旧 semantic / projection / seed bridge
  已经删掉不少
- 但 active path 仍然保留
  一批旧路线遗留下来的 analysis facts 和 helper bridge
- 这些残留还没有被真正的
  `PlanTTTransport + PlanTTCompute`
  所替代

所以接下来的工作，不是“在现有 helper 上继续补 case”，
而是：

1. 把 target builtin mapping 边界前移
2. 把 transport / compute / sync 的 owner 拆实
3. 把 side contract 和 late matcher 逐批删除

补充：

- 当前 rewrite 的完成判定
  不能只看 transport / compute cut-in
- 还要同时看：
  1. sync truth
     是否稳定落到
     `PlanTTSync + runtime semantics`
  2. 真源位置
     是否回到
     `TIR / SpatialPlan / TTProgram / ExecutableSpec`
     各自边界
  3. runtime / codegen
     是否已经退回 reader，
     不再补 planning truth

## 9. Rewrite Direction

当前 rewrite 路线固定为：

```text
Normalized Tile TIR
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> PlanTTBlocks
  -> PlanTTTransport
  -> PlanTTCompute
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

执行纪律：

1. 缺信息就补 `TIR/DSL/schema`
2. 缺 analysis 就补更早层的 analysis
3. 不能稳定决定的 variant 就 `unsupported`
4. 不再引入新的 side contract
5. 不再把过渡层写成长期架构

## 10. 结论

这轮重写的核心不是“换名字”，而是：

> **把 target builtin 选择放回它本来该发生的位置，并让 TTProgram 只承接真正的硬件 realization。**

只要这条边界不立住，后面无论再删多少旧名词，
系统还是会重新长出下一批旧名词。
