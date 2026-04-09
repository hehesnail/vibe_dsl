# Spatial/Dataflow Program Model Feature

## 基本信息

- **文档角色**: spatial/dataflow 架构导向的 cross-layer feature 设计
- **当前状态**: `2026-04-09` 活动设计文档
- **定位**: 不替代总体设计；只定义本 feature 的问题边界、owner 链、
  两阶段落地方式与对现有 IR/program model 的影响
- **适用范围**: 面向 spatial/dataflow 类后端；当前以 Blackhole 为第一落地点
- **非目标**: 不把本文档定义成 GPU/CPU 等所有 target 的统一 schedule 框架

## 1. 问题定义

当前 TileLang 的大量用户可写字面量与 sample-level schedule 写法，
主要来自 GPU-first 语境：

- `T.Kernel(...)` grid / threads
- `T.Parallel(...)`
- `alloc_shared / alloc_fragment` 的显式 shape
- `block_M / block_N / block_K / pipeline_stage` 等调优字面量

这些信息在当前主链里经常同时承担三种不同职责：

- 算法语义的真实硬约束
- 可移植的 schedule 提示
- 某个样例或某个 target 的局部实现细节

对 spatial/dataflow 后端来说，这种混写会导致三类问题：

1. 后端被迫把 GPU-first 字面量直接当成 target truth，
   无法根据 SRAM/L1/CB、通信和 placement 重新做 capacity-aware planning
2. `SpatialProgram` 和 `TTProgram` 的 owner boundary 被 case-local shape 穿透，
   迫使 target 侧从 sample 代码和 payload 反推真正的 block / transport / sync contract
3. 一旦需要跨 family 支持更复杂的 partition、communication、pipeline 或 sync，
   就会不断回退成 case-driven 特判

因此本文档定义的 feature 目标不是“再加几个 Blackhole knob”，
而是收正一个面向 spatial/dataflow 架构的 program model：

- 把用户字面量的语义层次分开
- 把 abstract partition/placement/transport intent 与 target concrete plan 分开
- 让 memory/pipeline/sync 成为一等 planning truth
- 让专家控制通过显式 target namespace 暴露，而不是继续借用 GPU-first 字面量

## 2. 设计输入与研究结论

本文档吸收的主要输入不是单篇论文，而是一致的结构性结论：

- `T2S` 把 temporal definition 与 spatial mapping 分离，
  说明 mapping 应是独立、可验证、可组合的一层，而不是散落在 kernel 字面量中
- `Spatial` 把片上内存、流水与 controller 视为一等对象，
  说明 compute 之外的 memory/pipeline truth 不能继续当成 schedule 副产物
- `Allo` 把 compute / memory / communication customization 解耦成可组合 primitive，
  说明 program model 必须允许不同类 intent 分开表达
- `Revet`、`Ripple`、`SAM`、`Dato` 都在强调：
  partition、channel、streaming/async、layout、virtual mapping
  如果不沿 IR 栈显式保留，下游就只能靠恢复语义和 late patching 勉强工作
- `TileScale` 进一步说明：
  计算、内存、通信最好共享统一的层次化视角；
  即便第一阶段只做单设备，也应给 hierarchy 留出显式扩展位

这些工作并不要求我们复刻它们的语言表面，
但共同支持下面这条 owner discipline：

```text
DSL literal semantics
  -> Spatial partition / placement / transport / sync intent
  -> TT concrete block / memory / execution plan
  -> ExecutableSpec materialization
```

## 3. Feature 目标与非目标

### 3.1 目标

1. 为 spatial/dataflow 类后端建立统一的字面量语义分层
2. 让 `SpatialProgram` 成为 abstract spatial intent 的真源，
   而不是让 target 从 case-local staging shape 反推意图
3. 让 `TTProgram` 成为 concrete block / memory / execution planning 的真源
4. 让 SRAM/L1/CB-aware sizing、buffering、transport、sync 成为 typed planning truth
5. 给 expert 用户提供显式 target namespace 的 validated hint API
6. 第一阶段先服务单设备 spatial/dataflow program；
   第二阶段在同一模型下扩展 hierarchy / multi-scale / inter-chip

### 3.2 非目标

- 不把本文档改写成第二份总体设计文档
- 不把 GPU/CPU 执行模型强行塞进同一套统一 API
- 不把物理 core 坐标、具体 CB id、具体 semaphore id 直接暴露为稳定 DSL 表面
- 不要求第一阶段就完成 multi-chip 运行时、collective 库或全层次 cost model

## 4. 字面量语义分层

本文档把用户可写的 schedule 相关字面量分成三类：

### 4.1 `semantic constraint`

真正改变程序可观察语义或合法域的约束。

示例：

- 真实 loop/domain bound
- 真实 buffer extent
- 因果 / layout / state update / aliasing 等硬约束

这类信息不能被后端静默放大、缩小或改写；
若需要变换，必须由显式 rewrite / retile / legality-preserving transform 完成。

### 4.2 `portable hint`

面向 spatial/dataflow family 的可移植调度意图，
不直接等于 target 最终 block / pipeline 结果。

示例：

- 倾向怎样切 parallel axes
- 是否偏好沿 reduction 方向分片
- 是否偏好 producer-consumer reuse
- 是否偏好较少 tail 或较粗 block

它们可以影响 planner，但不拥有最终 concrete size。

### 4.3 `target-specific validated hint`

显式写给某个 target family 的专家控制。
这类 hint 允许更强地影响 planner，但仍然要经过 legality 和 capability 验证。

第一批形式统一为：

- `T.blackhole.schedule(...)`
- `T.blackhole.place(...)`
- `T.blackhole.transport(...)`
- `T.blackhole.memory(...)`
- `T.blackhole.sync(...)`

这些都是 `validated hint`，不是 hard override：

- planner 优先 obey
- 若可合法降级，则 shrink / adjust 并给出诊断
- 只有与 semantic constraint 冲突，或 target 明确不支持时才 fail-fast

## 5. Owner 链与对象边界

### 5.1 DSL 层

DSL 负责表达：

- 算法语义
- 通用 spatial/dataflow intent
- 显式 target-specific expert hints

DSL 不负责拥有：

- 最终 block size
- 最终 CB 页数
- 最终 per-core decomposition
- 最终 route / semaphore / execution packet

### 5.2 `SpatialProgram`

`SpatialProgram` 负责 abstract spatial intent，
不负责 exact target sizing。

长期需要显式承载的类目包括：

- `WorkPartition`
- `Placement`
- `TransportIntent`
- `SyncIntent`
- `ProgramPhase`

其中 `WorkPartition` 的长期职责是表达：

- `parallel_axes`
- `reduction_axes`
- `partition_constraints`
- `tail_policy`
- `overlap_policy`
- `reuse_policy`

它回答的是“哪些切分合法、哪些复用和 closure 必须保留”，
而不是“在这块硬件上最终切多大”。

### 5.3 `TTProgram`

`TTProgram` 负责把 spatial intent 与 `TTHardwareModel` 一起收敛成 concrete target plan。

第一批长期 owner 包括：

- `TTBlockPlan`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTExecutionPlan`

其中新增 `TTBlockPlan`，作为 concrete sizing 真源，长期负责：

- `parallel_block_shape`
- `reduction_slice_shape`
- `subblock_shape`
- `buffering_policy`
- `block_linearization`
- `block_to_core_decomposition`
- 与 block 相关的 tail 处理和 diagnostics

`TTCBPlan` 只承接资源后果，
不再直接从 case-local shared shape 成为 decomposition owner。

### 5.4 `TTCoreGroup.work_packets`

`work_packets` 继续只表示调度结果：

- 哪些 logical block id 分配到哪些 core
- 每个 core 消费哪些 packet / wave

它不能再承担 decomposition owner，
也不应再被迫反推出更早层的 partition / block truth。

### 5.5 `ExecutableSpec`

`ExecutableSpec` 只物化冻结后的 target truth：

- buffer / CB / runtime args
- host transfer / accessor materialization
- direct launch / wave launch / synchronization materialization

它不是第二真源。

## 6. Phase 1: Single-Device Spatial/Dataflow Program Model

第一阶段的目标不是 hierarchy，而是先把单设备 spatial/dataflow program 写扎实。

### 6.1 交付范围

第一阶段必须覆盖：

- 字面量语义分层
- `SpatialProgram` 中的 partition / placement / transport / sync intent
- `TTProgram` 中的 `TTBlockPlan`
- SRAM/L1/CB-aware block and memory planning
- validated hint diagnostics
- 对现有 GPU-first 字面量的 compatibility intake

### 6.2 `TTBlockPlan` 生成算法

`TTBlockPlan` 是 target-side planning pass 的产物，
输入为：

- `SpatialProgram` 中冻结后的 intent
- op-local compute / transport truth
- `TTHardwareModel`

planner 的基本流程固定为：

1. 生成 target-legal candidate space  
   从 abstract intent 推出合法的 parallel block / reduction slice / subblock 候选
2. 做容量裁剪  
   估算 staging、carry、output、intermediate、buffering 成本；
   超出 SRAM/L1/CB 预算的候选直接丢弃
3. 选择最优合法 plan  
   优先最大化 block volume、减少 tail 和 wave，
   再综合 reuse / transport / buffering 偏好
4. 把 chosen plan 投影成 `TTCBPlan` 与 `work_packets`

这条链必须替代当前“上层 case 写了多大 shared shape，下游就照单全收”的模式。

### 6.3 SRAM/L1/CB-aware Planning

单设备第一阶段必须把 memory/pipeline 作为一等 planning truth。

长期至少要显式考虑：

- input staging footprint
- output / carry footprint
- intermediate / dst footprint
- buffering depth
- page size / page count / data format
- transport-local reuse 是否能减少 staging

也就是说，“能不能吃满 SRAM、该不该 double buffer、该不该缩 block”
都不再由 sample 手调决定，而由 target planner 在 typed truth 上决定。

### 6.4 Expert Hint API

第一阶段的 Blackhole expert hint family 定为：

- `T.blackhole.schedule(...)`
- `T.blackhole.place(...)`
- `T.blackhole.transport(...)`
- `T.blackhole.memory(...)`
- `T.blackhole.sync(...)`

其共同语义为：

- 局部作用域，默认作用在当前 `T.Kernel`
- 只提供 target-specific validated hint，不承载算法语义
- planner 必须产出“obey / shrink / reject”的诊断结果

这些 API 是专家通道，不替代 portable hints；
也不允许把 GPU-first 字面量继续当成隐式 Blackhole override。

### 6.5 Compatibility 策略

第一阶段不要求立刻删除现有 GPU-first 写法，
但要明确它们在 spatial/dataflow 路径中的新身份：

- 通用 `block_* / threads / stage` 类字面量默认作为 imported hint intake
- 如果其中某项已经是语义可观察约束，则必须显式标成 semantic constraint
- 未标注的 imported hints 不拥有最终 target size

这样可以在不破坏大量现有样例的情况下，
把 ownership 从 sample-local schedule 字面量迁回 planner。

## 7. Phase 2: Hierarchical / Multi-Scale Extension

第二阶段不推翻第一阶段，
而是在相同 owner discipline 下扩展 hierarchy。

### 7.1 交付目标

`Phase 2` 的目标不是一句“支持 hierarchy”，
而是把 `Phase 1` 的单设备 spatial/dataflow program，
扩成“多级 spatial/dataflow 系统”上的同一套模型。

第一批明确交付包括：

- 表达 chip 内多级 scale 与跨 chip / node 的 scale domain
- 表达 inter-core / inter-die / inter-chip / inter-node 的通信域
- 表达分层 memory、network 与可见性边界
- 表达分层 collective、task specialization 与 remote placement
- 在不推翻 `Phase 1` 对象边界的前提下，增加跨 scale planner 与 materialization

### 7.2 新对象与 schema 扩展

第二阶段需要在第一阶段模型上继续增加下面这些方向：

- hierarchical `Placement` scope
- scale-aware `TransportIntent`
- collectives 与 remote data movement 的 typed intent
- 跨 scale 的 synchronization / phase ownership
- 分层 virtual device / capability model

更具体地说，第一批需要落成的新增 truth 是：

- `SpatialProgram`
  - `ScaleDomain`
  - scope-aware `Placement`
  - remote / collective-aware `TransportIntent`
  - cross-scale `SyncIntent`
- `TTProgram`
  - scale-aware `TTBlockPlan`
  - remote / collective-capable `TTTransportPlan`
  - cross-scale `TTExecutionPlan`
  - hierarchical capability-aware planning 输入

这些对象不要求第一版就把所有 target family 全铺满，
但 schema 必须允许：

- local vs remote movement
- p2p vs collective
- compact vs spread placement
- one-scale vs multi-scale synchronization

但这些新增抽象仍然遵守相同边界：

- `SpatialProgram` 只表达 hierarchy intent 与 legality
- `TTProgram` 决定某个具体 target family 上的 concrete mapping / routing / buffering

### 7.3 planner 变化

相对 `Phase 1`，`Phase 2` 的 planner 需要新增 4 类能力：

1. scale selection
   决定某个 partition / task 是停留在本地 scale，
   还是提升到 cluster / die / chip / node 等更高一级 scale
2. remote transport selection
   决定 local reuse、p2p remote movement、collective
   之间该选哪种 transport family
3. hierarchical placement
   决定哪些 producer-consumer 应共置，
   哪些 partition 应 spread 到更高一级 scale
4. cross-scale synchronization planning
   决定哪些 dependency 是 local completion，
   哪些需要 remote arrival / collective completion

也就是说，`Phase 2` 的 planner 不只是“块更大一点”，
而是开始回答：

- 这段 work 落在哪一级 scale
- 数据在这一层是 local、remote 还是 collective
- 哪些 memory 层是真正可见的
- 哪个 synchronization contract 跨越了 scale boundary

### 7.4 非目标

`Phase 2` 明确不做下面这些事：

- 不要求第一版就做完整 distributed runtime
- 不要求第一版就暴露 MPI 风格 DSL 表面
- 不把物理 topology 细节、具体 route path、具体 link id
  直接暴露成稳定用户 API
- 不推翻 `Phase 1` 的对象和 planner 边界

### 7.5 与第一阶段的关系

第一阶段的所有 object boundary 都应按可扩展方式命名和组织，
避免把 single-device 假设写死进 schema 名字。

例如：

- `reduction_slice_shape` 是合法的长期名字
- `k_block` 不是
- `TransportIntent` / `TTTransportPlan` 应允许从单设备 reuse
  自然扩展到 remote / collective 场景

`Phase 2` 因此不是一套新模型，
而是对 `Phase 1` 的同一套 owner discipline 做层次化扩展。

### 7.6 Show Cases

#### Showcase A: Tensor-Parallel GEMM / AllReduce

这是最典型、也最贴近 `TileScale` 风格的例子。

逻辑程序仍然只是一次 GEMM，
但输出张量被按某个高一级 scale 分片，
局部 GEMM 完成后还需要做一次 cross-scale reduction。

在这套模型里，它会被表达成：

- `SpatialProgram`
  - `WorkPartition` 先定义局部 `M/N/reduction` 切分是否合法
  - `Placement` 决定 GEMM block 是 compact 在本地 scale，
    还是 spread 到更高一级 scale
  - `TransportIntent` 表达输入张量哪部分是 local shard，
    哪部分需要 remote visibility
  - `SyncIntent` 表达局部 GEMM 完成后要进入 collective completion
- `TTProgram`
  - 本地 GEMM 生成 local `TTBlockPlan`
  - reduction realization 选择 `TTTransportPlan`
    是 p2p tree、ring 还是 collective primitive
  - `TTExecutionPlan` 定义 local GEMM phase 与 cross-scale reduce phase 的顺序

这说明 `Phase 2` 不是“再加个 allreduce API”，
而是让 local compute、remote transport、collective sync
进入同一条 owner 链。

#### Showcase B: Sharded Paged Decode / Remote KV Access

第二个例子更贴近我们自己的 family。

设想后续 paged decode / sparse decode 的 KV state
不再都在单设备本地，而是按更高一级 scale 分片。
这时 decode kernel 的关键问题已经不只是 local block size，
而是：

- 当前 token 对应的 page 在本地还是 remote
- remote page 是按 p2p 拉取，还是先做某级缓存 / regroup
- attention/update 完成后哪些状态写回需要跨 scale 可见

在这套模型里，它会被表达成：

- `SpatialProgram`
  - `WorkPartition` 表达 token/page/head 这些轴的合法切分
  - `Placement` 表达 decode worker 与 KV shard 的 affinity
  - `TransportIntent` 表达 page fetch 是 local、remote copy
    还是更高一级 gather
  - `SyncIntent` 表达 page arrival 与 compute phase 的依赖
- `TTProgram`
  - 选择 local staging 还是 remote-aware transport plan
  - 选择 page fetch 与 compute 是否 overlap
  - 选择 local completion 还是 remote-visible completion

这个例子能说明：
`Phase 2` 真正新增的是 scale-aware placement/transport/sync，
不是把 `Phase 1` 的 block planner 简单复制到更大设备上。

## 8. 对现有 IR 与实现栈的影响

### 8.1 对 `Phase B`

`Phase B` 不需要重新打开，
但 `SpatialProgram` 的 object boundary 需要继续按本文档验证和扩展：

- `WorkPartition` 继续上提 typed fields
- `Placement` 从“仅有对象名义”继续走向稳定 consumer
- `TransportIntent` / `SyncIntent` 逐步减少 payload-only truth

### 8.2 对 `Phase C`

`Phase C` 是本文档第一阶段的当前主要落地点。
具体体现为：

- 新增 `TTBlockPlan`
- 让 `TTProgram` 成为 concrete sizing / memory / execution planning 真源
- 把 `TTCBPlan` 和 `work_packets` 降回正确 owner 边界
- 后续 family 扩展都应优先复用同一条 owner 链，
  而不是继续添加 case-local planner 特判

### 8.3 对运行时和 codegen

runtime / codegen 的长期角色不变：

- 只消费冻结后的 target truth
- 不负责恢复 partition / placement / transport / sync 语义
- 只负责物化、执行和诊断

## 9. 验证与推进方式

这份 feature 的实现推进分三类验证：

1. schema / validator 验证  
   确保 literal taxonomy、intent objects、`TTBlockPlan`
   与 diagnostics schema 自洽且可构造
2. planner 验证  
   确保 planner 会根据 `TTHardwareModel`、
   SRAM/L1/CB 预算和 validated hints 选择合法 concrete plan
3. runtime/correctness 验证  
   确保 target truth 的 owner 链最终能支撑 direct runtime、
   wider family 与更复杂 support surface

推进顺序固定为：

1. 先建 object/schema 边界
2. 再建 planner
3. 再让 family 逐步切到这条主链

## 10. 与现有文档的分工

- `final_blackhole_backend_redesign.md`
  继续是唯一总体设计；只保留长期分层架构与阶段判断
- `stage4_phase_b_spatial_ir.md`
  继续负责 `SpatialProgram` 当前阶段边界
- `stage4_phase_c_tt_target_ir.md`
  继续负责 `Phase C` 当前剩余项与完成判定
- 本文档
  负责跨 `SpatialProgram -> TTProgram` 的
  spatial/dataflow program model feature 设计，
  尤其是 literal semantics、planner owner 链、
  memory/pipeline planning 与 expert hint API

本文档不承担 `Phase C` backlog 跟踪，
阶段状态与验证摘要仍分别维护在阶段文档与 `tasks/progress.md`。
