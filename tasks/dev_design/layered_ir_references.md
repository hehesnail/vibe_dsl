# 分层 IR 设计参考

## 1. 文档角色

这份文档收拢的是 **TileLang Blackhole 分层 IR 设计的研究参考**。

它的作用只有一个：

- 说明当前 `Semantic -> Spatial -> TT Target` 分层方向背后的研究输入和方法论来源

它**不是**当前协议真源，也**不是**状态文档。

当前仓库的真源仍然是：

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/progress.md`

因此，这份文档适合回答的问题是：

- 为什么要把算法语义、空间组织、目标资源分成三层
- 为什么 `Task / Channel / Layout / ProgramPhase / SyncEdge` 要在 target mapping 之前成为一等对象
- 为什么 validator 不能只放在最后一层
- 为什么 `virtual spatial program` 和 `physical target mapping` 不能混在一起

它不适合回答的问题是：

- 当前 schema 字段名是什么
- 当前某个阶段是否完成
- 当前 blocker 是什么
- 当前测试基线是什么

## 2. 使用方式

读这份文档时，要把“参考输入”和“当前协议”严格分开。

正确用法：

- 用它理解为什么当前设计会收成今天这样
- 用它帮助判断某个问题更像 `Phase A`、`Phase B` 还是 `Phase C`
- 用它帮助审视 validator、映射边界、typed object 边界是否站得住

错误用法：

- 因为某篇论文用了某个 noun，就把它直接塞进当前长期 schema
- 因为某篇论文某一层更厚，就要求仓库立刻照搬
- 把论文内容当作比当前设计文档更高的真源

原则很简单：

- 论文提供方向和启发
- 仓库文档定义当前协议和边界

## 3. `Phase A` / `Stateful Semantic IR` 参考

`Phase A` 当前最核心的问题不是“再做一个漂亮的中间表示”，而是：

- 如何把 evidence 收成一个有边界的语义抽象层
- 如何让这个抽象层可验证、可拒绝、可做 refinement
- 如何避免后段继续从压碎后的 IR 反向猜语义

这部分最直接对应 [`stage4_phase_a_formalization_note.md`](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_formalization_note.md) 里的理论轨。

### 3.1 Abstract Interpretation

- **题目**: `Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints`
- **作者**: `Patrick Cousot, Radhia Cousot`
- **年份**: `1977`
- **链接**: `https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml`

对当前设计真正有价值的点：

- `Phase A` 更像一个 **bounded semantic abstraction**，而不是万能语义恢复器
- 抽象层可以有意选择保守、拒绝、失精，而不必承诺 completeness
- validator 的职责是检查抽象是否 sound，而不是强行把所有程序都吃下

这正好对应当前 `Phase A` 的诚实表述：

- 不承诺 arbitrary workload completeness
- 只对 canonical evidence domain 上的抽象 soundness 负责
- 缺证据时允许 `reject / invalidate / defer`

### 3.2 Constructive Galois Connections

- **题目**: `Constructive Galois Connections: Taming the Galois Connection Framework for Mechanized Metatheory`
- **作者**: `David Darais, David Van Horn`
- **年份**: `2016`
- **链接**: `https://icfp16.sigplan.org/details/icfp-2016-papers/23/Constructive-Galois-Connections-Taming-the-Galois-Connection-Framework-for-Mechanize`
- **可读预印本**: `https://arxiv.org/abs/1511.06965`

它对我们重要，不是因为仓库要 mechanize 成 proof assistant 项目，而是因为它强调：

- 抽象/实现边界必须能被明确写出来
- effectful specification 和 extractable implementation 需要区分
- “可计算的东西”和“只用于证明/规格的东西”不能混作一谈

这和当前 `Phase A` 很贴：

- `SemanticProgram` 是稳定抽象
- `StateVersion / StateDef / StateUse / StateJoin` 是 normalized semantic skeleton
- `preserve / typed_rebind / invalidate` 是 companion lifecycle contract

它最直接支持的是：

- 为什么 `Phase A` 要有显式 abstraction boundary
- 为什么 refinement checker 应该是可枚举 obligation family
- 为什么缺失 evidence 时应 fail-closed，而不是继续猜

### 3.3 MLIR

- **题目**: `MLIR: A Compiler Infrastructure for the End of Moore's Law`
- **作者**: `Chris Lattner` 等
- **年份**: `2020`
- **链接**: `https://arxiv.org/abs/2002.11054`

MLIR 对当前设计的价值不是“我们也要长得像 MLIR”，而是：

- 多层 IR 不是负担，而是解决多类问题的必要结构
- 不同层可以承接不同语义、分析、合法性与 lowering 责任
- 类型化的对象和跨层显式接口，远好于把所有 truth 混进一层 attr bag

这支持了我们今天的三层分工：

- `SemanticProgram` 冻结算法真相
- `SpatialProgram` 冻结空间组织真相
- `TTProgram` 冻结 TT 资源与 ABI 真相

### 3.4 Alive2

- **题目**: `Alive2: Bounded Translation Validation for LLVM`
- **作者**: `Nuno P. Lopes` 等
- **年份**: `2021`
- **链接**: `https://www.pldi21.org/poster_pldi.30.html`
- **论文 PDF**: `https://users.cs.utah.edu/~regehr/alive2-pldi21.pdf`

它和当前 `Phase A` / `Phase B` 最像的地方在于：

- 把“优化/lowering 之后是否仍然保持上层语义”这个问题单独 object 化
- translation validation 不依赖“作者感觉这个变换应该没问题”
- bounded / conservative validation 依然很有工程价值

对仓库最直接的启发是：

- `ValidateSemanticRefinement` 不是装饰，它是主链 contract
- 后续 `ValidateSpatialRefinement` / `ValidateTTTargetProgram` 也是同类问题
- validator 不需要先解决所有 completeness 问题，先把 soundness boundary 站稳更重要

## 4. `Phase B` / `Spatial Program IR` 核心参考

`Phase B` 这一层关心的是：

- 如何把冻结后的算法真相组织成 `Task / Channel / Layout / WorkPartition / ProgramPhase / SyncEdge`
- 如何在不泄漏 TT noun 的前提下承接 capability 约束
- 如何把“虚拟空间程序”和“具体目标映射”分开

这一层最接近的参考，不是纯 loop schedule 论文，而是 task/dataflow/spatial programming 这一档。

### 4.1 Dato

- **题目**: `Dato: A Task-Based Programming Model for Dataflow Accelerators`
- **年份**: `2025`
- **链接**: `https://arxiv.org/abs/2509.06794`

对我们最重要的启发：

- `task` 和通信结构本身应该是程序对象，不该让 backend 晚点再恢复
- layout/sharding 不应只是 codegen 副作用
- 虚拟映射和物理映射之间应该有清晰边界

对应到仓库：

- `Task / Channel / Layout` 成为 `Phase B` 一等对象
- `Task` 不默认等于 `TTKernel`
- `SpatialProgram` 先冻结 virtual truth，再交给 `Phase C`

### 4.2 Revet

- **题目**: `Revet: A Language and Compiler for Dataflow Threads`
- **年份**: `2023`
- **链接**: `https://arxiv.org/abs/2302.06124`

它最有价值的地方是：

- 高层执行语义和 backend realization 可以明确分离
- 一个更丰富的上层模型，仍然可以先 lower 成 generic dataflow backend
- 后段不应反向恢复前段语义

对应到仓库：

- `Phase B` 只能消费冻结后的 semantic truth
- `Phase C` 不允许继续发明 task graph / phase truth / update law

### 4.3 SPADA

- **题目**: `SPADA: A Spatial Dataflow Architecture Programming Language`
- **年份**: `2025`
- **链接**: `https://arxiv.org/abs/2511.09447`

它对我们最关键的价值是：

- routing / synchronization / asynchronous dataflow 语义必须显式化
- spatial/dataflow 编译如果没有严格 validation，很容易退化成约定俗成的脆弱系统

对应到仓库：

- `Channel.kind / payload_kind / delivery_kind`
- `ProgramPhase / SyncEdge`
- cross-phase materialization / ordering legality

这些都必须成为 typed contract，而不是靠后段猜。

### 4.4 T2S

- **题目**: `Programmatic Control of a Compiler for Generating High-performance Spatial Hardware`
- **别名**: `T2S`
- **年份**: `2017`
- **链接**: `https://arxiv.org/abs/1711.07606`

T2S 给我们的不是 object model 模板，而是一个很硬的哲学支点：

- “算什么” 和 “如何空间组织/映射” 必须拆开

对应到仓库：

- `Phase A` 管 algorithmic truth
- `Phase B` 管 spatial organization truth

这也是为什么我们没有把 `Task / Channel / Layout` 塞回 `SemanticProgram`。

### 4.5 Spatial

- **题目**: `Spatial: A Language and Compiler for Application Accelerators`
- **年份**: `2018`
- **链接**: `https://pldi18.sigplan.org/event/pldi-2018-papers-spatial-a-language-and-compiler-for-application-accelerators`

它更像一个历史前例：

- accelerator compiler 天然需要不止一层表示
- mapping / memory / pipeline 这些问题不能只在 emitter 末端临时处理

对我们有帮助的点是：

- 支持“accelerator backend 需要 layered IR”这个总判断
- 支持“mapping/resource planning 需要明确 object boundary”这个方向

## 5. `Phase C` / `TT Target IR` 参考

`Phase C` 的问题和 `Phase B` 已经不一样了。
它不再问“虚拟空间程序长什么样”，而是问：

- 这个 virtual program 如何落成 TT contract
- 哪些 kernel、CB、transport、sync、ABI object 必须显式存在
- 谁是 target truth，谁只是 materialization 结果

### 5.1 TL

- **题目**: `TL: Automatic End-to-End Compiler of Tile-Based Languages for Spatial Dataflow Architectures`
- **年份**: `2025`
- **链接**: `https://arxiv.org/abs/2512.22168`

这篇和当前 `Phase C` 的距离最近。

它对我们最有价值的点是：

- 硬件表示应当是显式 compiler object
- 空间映射和 target realization 应该是 compiler-owned，而不是散落的 target 常量 + emitter 经验
- 真正困难的问题不只是单 tile 内 codegen，而是 tile instance 在空间分布式硬件上的组织和映射

对应到仓库：

- `TTHardwareModel`
- `SpatialCapabilityModel`
- 后续 `TTProgram / MaterializeTTExecutableSpec`

### 5.2 SPADA

`SPADA` 在 `Phase C` 仍然重要，因为它提醒我们：

- target-side routing / sync contract 不能退化成“代码能跑就算对”
- validation 应该继续存在于 target layer，而不是只在 spatial layer 结束

因此它既支撑 `Phase B`，也支撑未来的 `ValidateTTTargetProgram` 思路。

### 5.3 Spatial

`Spatial` 对 `Phase C` 的帮助更多是历史层面的：

- accelerator compiler 最终一定会面对 memory / pipeline / resource / execution-plan 这些问题
- 这些问题不能被伪装成“只是 codegen 细节”

## 6. 补充参考

下面这些不是当前设计的直接模板，但对“为什么要多层、为什么 validator 很重要”仍然有帮助。

### 6.1 ScaleHLS

- **题目**: `ScaleHLS: A New Scalable High-Level Synthesis Framework on Multi-Level Intermediate Representation`
- **年份**: `2021`
- **链接**: `https://arxiv.org/abs/2107.11673`

它强化的是一个工程判断：

- 不同优化与合法性问题，通常适合在不同 abstraction level 上解决

这支持了我们不把所有问题重新压回单层 backend IR。

### 6.2 K-CIRCT

- **题目**: `K-CIRCT: A Layered, Composable, and Executable Formal Semantics for CIRCT Hardware IRs`
- **年份**: `2024`
- **链接**: `https://arxiv.org/abs/2404.18756`

它强化的是另一个判断：

- layered IR 不只是方便 lowering
- layered semantics 和可执行验证边界，本身就是系统长期稳定性的来源

## 7. 从这些参考里提炼出来的当前准则

这些论文并不都长得一样，但它们共同支持了几条今天仍然有效的准则。

### 7.1 先冻结意义，再做组织，再做实现

也就是：

- `Phase A` 冻结 algorithmic truth
- `Phase B` 冻结 spatial/dataflow truth
- `Phase C` 冻结 TT target truth

这不是审美偏好，而是问题结构推出来的。

### 7.2 通信和同步不能继续隐式化

如果 `channel / ordering / materialization` 不显式，
后段就只能：

- 靠名字猜
- 靠 kernel 形态猜
- 靠 target object 反推

这正是当前设计明确禁止的事情。

### 7.3 virtual mapping 和 physical mapping 必须分开

如果一个问题必须用 TT noun 才能回答，例如：

- CB purpose
- semaphore kind
- transport family
- ABI plan
- execution plan

那它就不应该留在 `Phase B`。

### 7.4 validator 是主链对象，不是事后补丁

这也是为什么当前设计坚持：

- `ValidateSemanticRefinement`
- `ValidateSpatialProgram`
- 未来 `ValidateTTTargetProgram`

因为 layered IR 真正有价值的前提，就是每层都能说清楚：

- 它承诺了什么
- 它没有承诺什么
- 它和上一层是什么 refinement 关系

## 8. 当前仓库如何使用这些参考

最实用的读法如下。

### 8.1 看 `Phase A` 时

优先看：

- Abstract Interpretation
- Constructive Galois Connections
- MLIR
- Alive2

主要回答：

- 为什么 `SemanticProgram` 是抽象层
- 为什么 refinement validator 合理
- 为什么 evidence 不足时允许 fail-closed

### 8.2 看 `Phase B` 时

优先看：

- Dato
- Revet
- SPADA
- T2S

主要回答：

- 为什么要有 `Task / Channel / Layout / ProgramPhase`
- 为什么 `SpatialProgram` 不是 target attr bag
- 为什么 capability 约束在这层只应裁合法空间，而不是直接写 TT 资源计划

### 8.3 看 `Phase C` 时

优先看：

- TL
- SPADA
- Spatial

主要回答：

- 为什么要有显式 `TTHardwareModel`
- 为什么 `TTProgram` 应是唯一 target truth
- 为什么 `ExecutableSpec` 不应和 `TTProgram` 双真源并存

## 9. 不该怎么用这些参考

不要用这些参考去：

- 绕开当前仓库文档已经写清楚的边界
- 因为某篇论文方便，就把 paper-specific noun 硬塞进长期 schema
- 把 target-specific 语义偷带回 `Phase B`
- 把 algorithmic truth 偷带回 `Phase C`
- 让“论文里这么做”替代“当前协议是否一致”

结论很简单：

- 这些参考值得保留
- 但它们是设计输入，不是协议真源
