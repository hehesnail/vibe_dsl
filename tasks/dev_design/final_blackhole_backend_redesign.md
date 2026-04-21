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
就不允许为下游再造一份旁路语义表示。

### 3.2 `SpatialPlan`

`SpatialPlan` 的新语义是：
**virtual spatial/dataflow program**。

它负责回答：

- 哪些 anchored sub-TIR 构成稳定执行单元
- 单元之间有哪些显式数据流 / carry / reduction / broadcast 关系
- virtual layout / sharding / distribution 语义是什么
- virtual phase / ordering / materialization boundary 是什么
- 哪些 hint 经 validate 后进入 planner

长期显式表示对象：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`

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

- block / core placement
- kernel family / role
- transport / routing / delivery
- sync / completion / ordering
- ABI / runtime args / accessor binding
- execution / launch order / waves

长期显式表示对象：

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

如果当前实现里仍保留：

- `TTProgram.payload`
- `buffer_tile_bridge_specs`
- `compute_contract` /
  `gemm_contract` /
  `multi_*_contracts`
- `direct_runtime_unsupported_reasons`

这些也只能被写成
leaf compatibility residue /
admission metadata，
不能回升成
`TTProgram`
的 owner truth。

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
  execution records
- 基于显式 leaf projection
  做 execution backend 选择 /
  admission gate /
  runtime-module materialization

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
  如果 repo HEAD
  里仍保留
  `blackhole.segment_kind`
  之类 source marker，
  它也只能是
  cleanup debt，
  不是 `ExecutableSpec`
  成立的前提
- 当前 direct runtime / codegen 的 admitted support surface
  只属于 leaf execution concern
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
- `TTKernelPlan`
  中的 operand/result binding
- `TTKernelPlan`
  中的 tile register / pack-unpack / accumulation / reduction protocol

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

fail-closed 纪律固定为：

- 缺 evidence 就 reject / unsupported
- 不再用名字匹配、位置假设、临时分支去补语义
- backend-specific unsupported
  只能停在 leaf execution gate，
  不能回流成上游表示层 / legality 约束

## 6. Fake Protocol 去留规则

审计表中列出的
legacy transition attrs /
helper bridge /
payload bag /
planning seed
都不是长期显式语义。

其中必须明确写死：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  如果仍存在，
  也只是 cleanup 期间
  唯一允许存活的窄 temporary handoff，
  不是新的
  `SpatialPlan` /
  `TTProgram` /
  `ExecutableSpec`
  语义层
- `blackhole.lowering_requirements` /
  `tl.blackhole_lowering_requirements_seed` /
  `blackhole.cb_requirements`
  如果仍存在，
  也只是 forced implementation debt，
  不能继续被文档表述成
  `TTProgram`
  的合法输入边界
- `TTProgram.payload`
  中仍存活的 bridge /
  contract /
  admission payload family
  只允许作为
  leaf compatibility debt，
  不能回升成 planning source

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

当前 rewrite 不再围绕“补 fake attr”推进，
而是围绕下面 4 个 closure set 推进：

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
4. **`Legacy Protocol Deletion`**
   - 删除 fake protocol 和 late matcher residue

架构收口仍按
`Task 1 -> Task 2 -> Task 3 -> Legacy Protocol Deletion`
理解显式表示层边界；
cleanup 的架构依赖顺序
固定按 cleanup 文档理解为：

1. **`Cleanup Task 0`**
   - 锁定 exact TT-Metal builtin surface
   - 收正 builtin-selection pass
2. **`Cleanup Task 1`**
   - 把 logical bridge capture
     收成唯一窄 bridge attr
3. **`Cleanup Task 2`**
   - 删除 public / internal legacy analysis bag
   - 同时把
     `SpatialPlan`
     收成单一 direct builder implementation；
     当前
     `BuildSpatialPlanCompanion`
     这个名字如果继续保留，
     也只是历史实现名，
     不是架构契约
4. **`Cleanup Task 3`**
   - 删除
     `blackhole.copy_semantics`
5. **`Cleanup Task 4`**
   - 删除
     `blackhole.segment_kind`
   - 收紧 planner / projection / runtime
     对 kernel kind / segment plan
     的直接表示消费
6. **`Cleanup Task 5`**
   - 做最终 cleanup scan、
     文档同步、
     验证与 memory 沉淀
7. **cleanup 完成后**
   - 才恢复 support surface /
     workload payoff 扩展

这里要和
`tasks/progress.md`
明确分开：

- 上面是 cleanup 的架构依赖顺序
- repo HEAD 当前从哪个切片继续推进、
  当前 blocker 在哪里，
  统一只看 `progress.md`
- `Cleanup Task 0`
  出现在依赖顺序里，
  不等于 repo HEAD
  已按完成口径收口；
  当前只有 selector-forwarding
  局部结果，
  不能当成完成声明

补充：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

都只是
`Task 1: SpatialPlan Representation Cutover`
里的 preparatory substeps，
不再单独充当顶层 roadmap

当前活动设计文档按下面顺序约束实现：

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
5. `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
   - 固定当前 repo HEAD 的 cleanup 总执行顺序
6. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
   到
   `task5.md`
   - 固定每一步 cleanup 的具体收口范围

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
