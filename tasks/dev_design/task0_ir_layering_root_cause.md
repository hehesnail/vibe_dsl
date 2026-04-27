# Task 0: Root Cause And Rewrite Direction

## 基本信息

- **文档角色**: 当前 Blackhole layered IR 根因诊断文档
- **当前状态**: 活动设计基线
- **任务链位置**: 根因索引 `Task 0`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 本文档定义根因和 rewrite 方向；
  不维护 cleanup 完成状态
- 它不是当前 repo HEAD 状态文档
- 当前实现状态统一以 `tasks/progress.md` 为准
- 这里的 `Task 0`
  是历史根因索引，
  不是 cleanup 主线里的
  `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`

## 1. 根因结论

Blackhole 后端这轮 rewrite 的根因
不是“后段 matcher 不够聪明”，
而是历史上下面三件事同时发生：

1. **compute-side exact builtin 选择放得太晚**
   - tile op、layout、真实 `BufferLoad / BufferStore`
     的语义边界已经被打碎
   - TT-Metal API 粒度的 tile compute semantics
     （matmul / reduce / unary / binary /
     broadcast / copy / pack /
     tilize / untilize 等）
     被 generic scalar lowering
     提前破坏或隐藏，
     迫使后段用 late matcher
     从 scalar loop / local expression
     中恢复 compute intent
2. **`SpatialPlan` 这层显式表示没有真正立起来**
   - 当前中间层过薄，
     无法稳定承接
     virtual spatial/dataflow 语义
3. **下游阶段被迫恢复语义**
   - 于是长出
     legacy transition attrs / helper bridge / payload bag
     这类影子协议

一句话：

> **真正缺的不是更多后段 contract，而是 `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec` 这条显式主链必须成为唯一跨阶段语义载体；否则 exact builtin legality、virtual spatial/dataflow、target realization、leaf materialization 会继续被挤到 side channel 里。**

截至 `2026-04-26`，
repo HEAD 已经把 broad legacy protocol /
public wrapper /
cross-pass bag
从 active chain 中收掉。
根因仍用于解释
当前剩余 support-surface debt：

1. `SpatialPlan`
   已收成 cleanup owner truth，
   但 support-surface 扩展所需的
   logical live-value /
   materialization-boundary
   一等表示仍需补齐
2. planner / `TTProgram`
   已成为 TT-specific realization
   owner truth，
   `tl.blackhole_logical_buffer_tile_bridge_specs`
   这条窄 bridge attr
   已退出 active chain；
   mesh /
   buffer distribution
   schema 已在
   `TTProgram`
   typed fields 中表达
3. build / codegen / runtime
   已站在
   `tl.tt_program -> tl.blackhole_executable`
   的显式投影上，
   contract-family payload /
   fallback
   已删除；
   剩余问题是
   flash-attn
   exact row-reduction
   仍需要正确消费上游
   CB-live source
4. direct runtime
   当前仍只是
   unit mesh /
   replicated buffer /
   copy-GEMM admitted subset，
   它不能继续被当成
   codegen/export
   或 `TTProgram`
   的能力边界；
   TT-Metal 自身的
   `Program / MeshWorkload / MeshBuffer`
   模型已经覆盖
   multi-device /
   sharded /
   fabric
   方向

因此这份 root-cause 文档
必须明确：

- `TTProgram`
  不是根因本体
- leaf residue
  已从 legacy cleanup
  转成 support-surface admission
  问题
- direct runtime
  只是 leaf backend admission，
  不是 target realization
  的 owner truth
- **第一推动错误**
  是 target-independent 的
  spatial/dataflow owner truth
  没有及时收进显式 `SpatialPlan`
  和后续显式对象；
  当前后续工作不能把 leaf debt
  反向写成新的上层协议

## 2. 这件事在分层 IR 上到底意味着什么

当前 repo HEAD 的 active chain
虽然名字很多，
但长期边界仍只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

当前 pass / phase 顺序
只是这四层的现有实现手段，
不是新的 IR 层。

当前 repo HEAD 的真实情况是：

- 显式主链已经存在并成为 active chain
- broad legacy analysis /
  public wrapper /
  cross-pass bag
  已退出
- 剩余问题集中在
  logical live-value
  缺失、
  narrow bridge attr
  和 leaf contract-family
  fallback，
  这些只能继续按 debt 收敛，
  不能写成新的中间层

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
- `CB / semaphore / runtime arg / launch`
- sync / ABI / execution realization

额外约束固定为：

- compute-side exact TT-Metal builtin selection
  的 owner truth
  必须在这一层仍保留
  tile op / layout /
  真实 `BufferLoad / BufferStore`
  语义时建立
- Blackhole target
  的 TT-Metal API 粒度 tile compute semantics
  必须在这一层保留或规范化；
  这覆盖 matmul / reduce / unary /
  binary / broadcast / copy / pack /
  tilize / untilize
  等通用 leaf API 粒度，
  不是 reduce 或 flash-attn 专项例外
- local tile expression
  如果需要拆成多个 TT-Metal leaf API，
  decomposition
  必须基于 IR 结构、类型、region、axis
  在当前层完成；
  不允许在后段按 workload 名 /
  buffer 名 /
  scalar loop 形态恢复语义
- 这里的
  selected exact-builtin compute IR
  只是当前 `Normalized Tile TIR`
  的 checked postcondition /
  admitted subset，
  不是新层
- 如果 exact legality
  只能靠后面的 planner / payload /
  runtime compatibility 证明，
  说明边界已经错了

### 2.2 `SpatialPlan`

这一层必须把 target-independent 的
virtual spatial/dataflow 语义显式化，
并且只能站在显式对象上。

它应当回答：

- 哪些 anchored sub-TIR 构成稳定执行单元
- 单元之间有哪些显式数据流 / carry / reduction / broadcast 关系
- virtual layout / sharding / distribution 关系是什么
- virtual ordering / materialization boundary 是什么

长期对象固定按总体设计文档收在：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`
- `LiveValue`
- `LiveValueEdge`
- `MaterializationBoundary`

如果这些东西不在这一层显式存在，
下游就只能继续靠名字、attrs、局部 matcher
或 payload bag 去猜。

### 2.3 `TTProgram`

这一层必须承接唯一的
TT-specific physical realization：

- `kernel / transport / sync / ABI / execution`
  这些 target slices
- block / core placement
- routing / delivery
- runtime args / accessor binding
- launch order / waves

它消费的是：

- `SpatialPlan`
  的显式对象
- anchored sub-TIR
- 已经建立好的
  compute-side exact builtin selection 结果

它不能继续承担：

- 恢复 target-independent spatial/dataflow 语义
- 替上游补中间层抽象
- 用 bridge bag / helper seed
  反向定义 builtin legality

补充说明：

- `CB / semaphore / runtime arg / launch`
  这类 TT-Metal program-construction 事实
  属于这一层和后续 leaf projection 边界
- 这和 TT-Metal
  显式 `Program / MeshWorkload / CreateKernel /
  CreateCircularBuffer / CreateSemaphore /
  SetRuntimeArgs / LaunchProgram`
  的程序模型一致；
  它们不是 `SpatialPlan`
  或 compute-side legality
  应该承载的语义
- multi-device /
  distributed /
  mesh
  方向同样属于
  `SpatialPlan`
  的 virtual layout /
  distribution
  到 `TTProgram`
  的
  mesh / device-range /
  `MeshBuffer`
  distribution /
  fabric transport
  lowering；
  不能被当前 direct runtime
  的 `create_unit_mesh(0)`
  admitted subset
  反向裁掉

### 2.4 `ExecutableSpec`

这一层只做 leaf projection 和执行物化，
不是第二份长期语义表示。

build / codegen / runtime
只能消费：

- `tl.tt_program`
- `tl.blackhole_executable`
- `ExecutableSpec`
- `artifact.rt_mod`

这条显式链上的结果，
不能回头恢复 planning 语义。

补充：

- direct runtime
  的 unsupported reason
  只说明
  当前 executable
  对该 backend
  未 admission
- codegen/export
  可以拥有不同的
  schema / emitter
  admission gate
- 因此
  “direct runtime 不能跑”
  不能等价为
  “TTProgram 不能表达”
  或
  “TT-Metal codegen
   不能生成”

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

在 cleanup 主线启动时，
这些东西的典型形态已经很明确：

- producer-side analysis /
  public bag residue，
  现已退出 active chain：
  - `blackhole.work_decomposition`
  - `blackhole.compute_regions`
  - `blackhole.pipeline_stages`
  - public `AnalyzeBlackhole*`
    wrapper
- planner / `TTProgram`
  broad forced debt
  已收敛：
  - `blackhole.lowering_requirements`
  - `blackhole.cb_requirements`
  - `tl.blackhole_lowering_requirements_seed`
  - `tl.internal_tt_*`
  - `tl.blackhole_logical_buffer_tile_bridge_specs`
- leaf semantic recovery residue：
  - `blackhole.copy_semantics`
  - `blackhole.segment_kind`
  - `blackhole.resource_plan`
    这些 broad / cross-pass residue
    已退出 active chain；
    `blackhole.segment_kind`
    只允许作为 pass-local mechanics
    并在 leaf 前剥离
  - `buffer_tile_bridge_specs`
    payload / projection / codegen residue
  - `compute_contract`
    /
    `multi_compute_contracts`
    /
    `gemm_contract`
    compatibility fallback
  上述 leaf residue
  也已退出 active chain；
  `blackhole.segment_kind`
  只允许作为
  `lower_blackhole_ops.cc`
  内部 pass-local mechanics，
  不允许到达 final IR /
  leaf reader

这些对象的共同问题不是“名字不好”，
而是它们都在
**替代本该由显式表示层承载的语义**。

更准确地说，
它们不是三类互不相关的历史包袱，
而是同一条根因链的连续症状：

1. `SpatialPlan`
   没把 virtual spatial/dataflow owner truth
   收干净
2. planner / `TTProgram`
   如果继续 carry
   facts bag / seed / bridge residue，
   就是在替上游显式表示补洞
3. leaf
   如果再从 marker / payload / fallback
   恢复更晚阶段才需要的 meaning
   就是在把 projection 反向变成 planning

## 4. repo-local 成熟后端和目标模型真正支持的是什么

这一轮收紧后，
task0 根因结论
已经不需要靠“再造一套 review 说法”来支撑；
repo-local 成熟后端
和 TT-Metal 稳定程序模型
本身就在支持同一条纪律。

### 4.1 repo-local 成熟 backend 的共同模式

仓库里成熟 GPU passes
已经在用同一模式：

- 当前 IR
- pass-local analysis / collector
- 直接 rewrite 成当前层
  或更显式的 target-facing 对象

而不是：

- shared facts bag
- public helper wrapper
- 跨 pass payload carrier

直接例子包括：

- [lower_ldg_stg.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_ldg_stg.cc)
  - 直接从
    `BufferLoad / BufferStore / Ramp / IfThenElse`
    结构判断是否可 lower，
    然后就地改写成显式 `ldg/stg` intrinsics
- [lower_hopper_intrin.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_hopper_intrin.cc)
  - 在同一个 pass 里
    收集 descriptor / mbarrier 需要的局部事实，
    然后直接把它们写回当前函数的显式 host/device setup
- [wgmma_sync_rewriter.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/wgmma_sync_rewriter.cc)
  - 只用当前 TIR 的 buffer access / stmt order
    做局部分析，
    然后立即重写 sync ordering
- [annotate_warp_group_reg_alloc.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/annotate_warp_group_reg_alloc.cc)
  - derived register hint
    只在 pass 内收集和注入，
    没有长成 shared semantic carrier

这些 repo-local 先例共同说明：

- target lowering
  可以大量使用 visitor / matcher / mutator / builder
- 但它们必须直接站在 current IR /
  current object 上工作
- 不能把 analysis 结果升格成长期协议面

### 4.2 TT-Metal 程序模型支持的边界

TT-Metal 的稳定 host-side truth
是显式：

- `Program`
- `MeshWorkload`
- `CreateKernel`
- `CreateCircularBuffer`
- `CreateSemaphore`
- `SetRuntimeArgs`
- `CompileProgram`
- `WriteRuntimeArgsToDevice`
- `ConfigureDeviceWithProgram`
- `LaunchProgram`

这说明：

- exact builtin selection
  必须在更早的 current IR
  仍保有 compute 结构时完成
- program-construction 事实
  应收在 `TTProgram -> ExecutableSpec`
  这条 target/leaf 边界
- runtime API
  不会替编译器恢复
  `SpatialPlan`
  或 current TIR
  本应显式承载的语义

## 5. 正确的 rewrite 方向

整改方向不是“多加几个 pass”，
而是把 owner truth
重新收回到显式表示层。

固定方向如下：

1. **compute-side exact builtin legality**
   - 收成当前 `Normalized Tile TIR`
     的 checked postcondition
   - 不再由旧 planner / helper / seed
     间接拥有
2. **`SpatialPlan`**
   - 收成唯一的
     virtual spatial/dataflow representation
   - 不再允许 broad bridge bag
     代替它
3. **`TTProgram`**
   - 收成唯一的
     target realization representation
   - `CB / semaphore / runtime arg / launch`
     这类事实只允许留在这里
     及其后续 leaf projection
4. **`ExecutableSpec`**
   - 只做 `TTProgram`
     的 leaf projection / materialization
   - build / codegen / runtime
     只能读它
5. **legacy protocol**
   - 统一按
     `wrong now, delete later`
     处理
   - 只允许 shrink / delete，
     不允许因为 repo HEAD
     还依赖它就升格成合法边界
   - 已删除的 bridge attr /
     payload fallback /
     contract-family surface
     不能以 debug helper
     或测试兼容面名义重新引入

## 6. IR-First 纪律

从这个根因可以直接推出下面几条纪律：

1. 语义只能存在于显式表示层本身
2. analysis 只能是从当前 IR 派生的、
   可失效、可重算的临时结果
3. 如果下游需要的信息当前 IR 没有，
   结论只能是扩 IR / 扩显式对象 /
   扩 validator / 显式 unsupported
4. 架构边界只能写成：
   - `Normalized Tile TIR`
   - `SpatialPlan`
   - `TTProgram`
   - `ExecutableSpec`
   这些显式表示层，
   不能写成 pass / file / helper 名字
5. visitor / matcher / mutator / builder
   允许作为 pass-local mechanics
   存在，
   但它们必须直接支撑当前 rewrite，
   不能长成新的 shared layer
6. 不能先把当前后段实现固定成前提，
   再要求上游去适配
7. 后段如果在表示层边界上越权，
   应优先改后段实现，
   不是先给前面补 attr / matcher / facts bag
8. 如果当前实现还依赖错误边界，
   文档必须把它写成
   `wrong now, delete later`，
   不能写成“暂时合理的中间层”
9. exact builtin legality
   必须在进入 `TTProgram`
   之前独立成立；
   `ValidateTTProgram`
   只能复验已经进入
   `TTProgram`
   的显式 realization 结果
10. leaf/runtime 的 admitted support surface
    可以拒绝某个
    `ExecutableSpec -> execution backend`
    组合，
    但不能反向收窄
    上层显式语义和 validator 边界

## 7. 与后续文档的关系

- 本文档只负责：
  - 根因诊断
  - layered IR rewrite 方向
  - IR-first 纪律基线
- cleanup `task0-task5`
  是已完成 broad convergence
  的执行切片，
  不是新的 IR 层；
  当前剩余 narrow leaf debt
  只按 `tasks/progress.md`
  和对应任务设计继续跟踪
- `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
  负责 cleanup 启动时
  compute-side exact builtin legality
  这条 cleanup 切片的 owner truth
- `task1_spatial_plan_companion.md`
  负责定义 `SpatialPlan` 这层表示的合同
- `task2_ttprogram_companion_cutover.md`
  负责定义 `TTProgram` 这层表示的合同
- `task3_runtime_gate_and_workload_cutover.md`
  负责定义 leaf reader / workload cutover 合同
- `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
  负责把 task0-task5
  组织成 cleanup 完成边界和验证索引
- `tasks/progress.md`
  负责记录 repo HEAD 当前状态和下一步
