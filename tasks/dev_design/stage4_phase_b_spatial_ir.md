# Stage 4 Phase B: Spatial Program IR

## 基本信息

- **文档角色**: `Phase B` 实施与设计边界文档
- **当前状态**: `2026-04-06` 已完成 compile-path hardening：
  `SpatialProgram / ProgramPhase`、copy/GEMM fast-path、`flash-attn` multi-phase gate、
  representative family gate、`LowerToSpatialProgram -> ValidateSpatialProgram`、
  以及 `LowerBlackholeOps` 的 spatial-only consumer cutover 均已进入主链。
  但当前实现仍偏向 structural scaffold；`Phase B` 的下一阶段重点不是再补对象数量，
  而是把 `SpatialProgram` 从“结构投影 IR”继续收紧成 execution-bearing spatial contract。
- **上游输入**: 冻结后的 `SemanticProgram`
- **下游输出**: 冻结后的 `SpatialProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. Phase B 的职责

`Phase B` 只负责一件事：

- 把已经冻结的 algorithmic truth 组织成稳定的 spatial/dataflow program structure

它回答的问题是：

- 哪些 `Update` 应该组织成哪些 `Task`
- 哪些 `State` 之间要形成哪些 `Channel`
- 哪些 `Layout / WorkPartition / SyncEdge / Placement / ResourceIntent` 必须显式存在
- 多 `T.Kernel` / 多 device member 程序的 `ProgramPhase` 边界在哪里

它不负责：

- 再做一次 semantic recovery
- 发明 TT resource / CB / semaphore / ABI
- 从 raw TIR 或 late builtin 重新猜 `carry / selection / recurrence / source_states`

## 2. Core Design Boundary

### 2.1 核心对象

`Phase B` 的长期 core object set 是：

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

### 2.2 ProgramPhase 的稳定宿主

`ProgramPhase` 的跨函数真相固定挂在：

- `IRModule.global_infos["tl.device_programs"]`

而：

- `PrimFunc.attrs["tl.spatial_program"]`

只保留 member-local spatial truth。

因此：

- 单 `PrimFunc` 程序只是 `member_funcs.size() == 1` 的退化情况
- 任何 cross-function 的 phase order、shared buffer、global sync 都不能由单个 `PrimFunc` 自己发明

### 2.3 小闭集 family

`Phase B` 仍然遵守 small-closed family 设计：

- `Task.kind`
  - `transfer`
  - `compute`
  - `collective`
  - `control`
- `Layout.kind`
  - `regular`
  - `packed`
  - `indexed`
- `WorkPartition.kind`
  - `replicated`
  - `blocked`
  - `indexed`
  - `filtered`
- `Placement.kind`
  - `execution`
  - `communication`
  - `phase_boundary`
- `SyncEdge.kind`
  - `dependency`
  - `barrier`
  - `completion`
- `ResourceIntent.kind`
  - `buffer`
  - `state_residency`
  - `synchronization_support`
  - `phase_boundary_materialization`

更细差异通过 bindings 与 typed traits 表达，不通过 workload noun 扩 schema。

### 2.4 Legality 与 Policy 的边界

`Phase B` 必须明确区分：

- **analysis / legality facts**
  - 哪些 cut 是必须的
  - 哪些 state flow 必须显式化
  - 哪些 phase boundary 必须物化
  - 哪些 layout / partition family 是被语义强制的
- **policy decisions**
  - 合法空间内的 fusion / split 偏好
  - canonical fast-path 选择
  - placement / reuse / sync 粒度偏好

规则是：

- analysis 决定 legality
- policy 只在合法空间内选择

### 2.5 不能回退的约束

`Phase B` 不允许：

- 直接消费 raw fragment attrs 当 semantic truth
- 通过名字匹配恢复 spatial 边界
- 把 `Task:TTKernel = 1:1` 固化成默认心智模型
- 用 TT resource 名词污染 `Task / Channel / Layout / WorkPartition`

### 2.6 参考方法论收口

`Phase B` 后续不再只以“对象有没有建出来”为目标，而明确按参考论文里的共同方法论收口：

- `T2S` 给出的核心纪律是：temporal definition 和 spatial mapping 分离；
  `SpatialProgram` 必须承载 mapping 本身，而不是把 mapping 留给后段恢复
- `Dato` 给出的核心纪律是：task / stream / shard / virtual placement 必须是一等对象；
  `SpatialProgram` 必须能表达 virtual task graph 与 virtual mapping，
  再交给 target-specific physical mapping
- `SPADA` 给出的核心纪律是：routing / async / ordering 不是注释，
  而是 legality object；`SpatialProgram` 必须能对 flow / sync 做 fail-fast legality
- `TL` 给出的核心纪律是：spatialization 必须读取 hardware capability，
  不可能在真空中完成；但这种输入应该是 abstract capability，而不是 TT noun
- `Revet` 给出的核心纪律是：rich program model 应先降成 generic dataflow program，
  再落到 backend；不能把 backend 写成 workload matcher bag

因此，`Phase B` 的正确目标是：

- 构造一个 target-informed 但 non-target-materialized 的 virtual spatial/dataflow program

而不是：

- 只构造一组结构对象，等 `Phase C` 再补空间执行语义

## 3. Semantic To Spatial Contract

`Phase B` 只允许从下列上游真源读取必须保留的算法约束：

- `SemanticProgram`
- internal state/effect graph
- companion lifecycle contract

这也意味着：

- `Phase B` 不直接读取 `tl.semantic_manifest`
- `Phase B` 不直接读取 `blackhole.fragment_regions`
- manifest / fragment evidence 必须先在 `Phase A` 被归约成冻结后的 semantic truth

主要投影关系是：

- `Update` -> 一个或多个 `Task`
- `State` -> `Channel` / `SyncEdge` / `ResourceIntent`
- `Domain` -> `Layout` / `WorkPartition` 的候选空间
- `AccessMap` -> gather / scatter / paged / routed 等空间边界
- `UpdateLaw` -> ordered update / carry / merge / completion requirement

如果 `Phase B` 发现缺失 truth，处理原则只能是：

1. 若该 truth 本质上属于 `Phase A`，回去补 witness/core/validator
2. 若该 truth 本质上属于 spatial organization，本层补 object / legality / policy
3. 不允许直接加 matcher 绕过分层

### 3.1 Spatial IR 必须承载的执行语义

`Phase B` 的目标不是把 semantic truth 换一套对象名存一遍，而是冻结
“如何执行成 spatial/dataflow program” 这件事本身。

这意味着：

- `Task` 必须承载稳定的 execution-unit formation truth：
  哪些 update/state interaction 共同组成一个可执行单元，为什么需要 split/fuse，
  它属于哪个 phase，以及它对下游是 `transfer / compute / collective / control`
  中的哪一类执行角色
- `Channel` 必须承载稳定的 flow truth：
  source/target task、相关 state、是否跨 phase、是否要求 ordered delivery /
  completion / versioned state flow；不能只剩“有一条边”
- `Layout` / `WorkPartition` 必须承载稳定的 domain-realization truth：
  它来源于哪个 semantic domain，是否是 indexed / filtered / grouped / paged /
  blocked 之类的空间展开，是否存在 work-dependent bounds，不能只剩 axes 列表
- `ProgramPhase` / `SyncEdge` 必须承载稳定的 partial-order truth：
  哪些 task/channel 共同构成一个 phase，phase 间为什么需要 barrier / completion /
  carry boundary，不能只剩 phase 名字和显示顺序
- `Placement` / `ResourceIntent` 必须承载 stable spatial obligations：
  execution / communication / phase-boundary 的 placement intent，
  以及 state residency / boundary materialization / pipeline / fragment 这些
  已经不是 semantic、但还没进入 TT resource 的 contract

判断标准只有一个：

- 如果某个 non-TT-specific truth 是 `Phase C` 做合法 mapping 必须知道的，
  那它就必须在 `SpatialProgram` 里显式存在；`Phase C` 不允许自行恢复

### 3.2 抽象硬件能力接口

`SpatialProgram` 不能只看语义，不看机器。
但它看的也不应该是 TT resource noun，而是抽象 hardware capability。

这里引入的设计口径是：

- `Phase B` 读取 `SpatialCapabilityModel`
- `Phase C` 读取 `TTHardwareModel`
- `SpatialCapabilityModel` 是从 concrete target model 导出的抽象能力视图

`SpatialCapabilityModel` 至少要表达：

- topology class / communication neighborhood
- virtual placement domain 的形状与允许关系
- flow capability：point-to-point / multicast / reduce / gather / scatter / carry
- ordering capability：dependency / completion / barrier / async arrival
- residency capability：ephemeral / persistent / transport-backed / replicated
- partition capability：blocked / indexed / filtered / grouped / paged / chunked

它的职责不是决定具体 TT resource id，而是限制和引导：

- 哪些 task formation 是合法且值得保留的
- 哪些 flow/sync family 是可表达的
- 哪些 layout / work partition family 是目标能力允许的
- 哪些 virtual placement / communication shaping 是可行的

### 3.3 需要补进 Spatial IR 的算法职责

参考论文里真正有价值的不是 object 名字，而是 spatial synthesis 算法。
按这个口径，`Phase B` 后续必须补的不是“更多字段”，而是更明确的算法职责：

- task formation algorithm
  - 从 semantic state/update graph 构造稳定 task graph
  - split/fuse 必须有 legality basis 和 capability-aware policy basis
- flow shaping algorithm
  - 从 access/update/state version 关系构造 point-to-point、broadcast、
    gather/scatter、carry、reduction 这些 flow class
- domain realization algorithm
  - 从 `Domain + AccessMap + UpdateLaw` 构造 indexed / filtered / grouped /
    paged / chunked 的 layout / partition contract
- phase and ordering synthesis
  - 从 stateful dependence、cross-update completion、carry/reduction semantics
    构造 partial order，而不是只靠一两个固定 phase 名
- capability-informed legality / policy
  - 用 `SpatialCapabilityModel` 裁掉不合法候选
  - 在合法候选里再做 locality / reuse / communication / synchronization tradeoff

### 3.4 具体算法定义

下面这些算法不是实现建议，而是 `Phase B` 的正式职责定义。
后续代码、validator 和 `Phase C` 边界都按它们来约束。

#### 3.4.1 Task Formation Algorithm

**输入**

- `SemanticProgram.updates`
- `SemanticProgram.states / state_defs / state_uses / state_joins`
- `Domain / AccessMap / UpdateLaw`
- `SpatialCapabilityModel`

**输出**

- 一组稳定的 `Task`
- 每个 `Task` 的 execution role、phase membership 候选、update membership

**算法**

1. 先构造 `update-state` 二部图：
   - update 写哪些 state version
   - update 读哪些 state version
   - 哪些 state join/merge 形成多源汇合
2. 对每个 update 计算 `execution signature`：
   - update law class：`map / reduce / select / recurrence`
   - access class：dense / indexed / indirect / paged / grouped / predicate-filtered
   - state interaction class：stateless / carry / reduction / selection / index-valued
3. 计算 **mandatory cut predicates**。两个 update 之间只要满足任一条件，就不能默认 fuse：
   - 中间存在 state version boundary，且该 version 要求 materialized flow
   - law class 不兼容，必须区分 `transfer / compute / collective / control`
   - access class 对 layout/partition 的要求不兼容
   - `SpatialCapabilityModel` 不允许它们共享 placement / communication domain
   - 会破坏 ordered update / carry / reduction completion 语义
4. 以 update 为初始 seed task，先按 mandatory cut 切开。
5. 仅在以下条件全部满足时，允许把相邻 seed task 融合为同一个 virtual task：
   - phase class 兼容
   - layout/partition family 兼容
   - flow 仍可在 task 内局部实现，不需要显式 channel
   - capability model 允许共享 placement / locality domain
6. 对每个 task 赋 `Task.kind`：
   - 以主导 execution signature 决定，不允许靠 workload 名字决定
   - `map` + data movement dominated -> `transfer`
   - dense arithmetic dominated -> `compute`
   - reduction / merge dominated -> `collective`
   - select / route / recurrence control dominated -> `control`

**不允许**

- 直接按 workload 名字或单个 kernel 形态分 task
- 先假设 `Task:TTKernel = 1:1`，再倒推 task

#### 3.4.2 Flow Shaping Algorithm

**输入**

- `state_defs / state_uses / state_joins`
- `UpdateLaw`
- `AccessMap`
- `SpatialCapabilityModel`

**输出**

- 一组 `Channel`
- 每条 channel 的 flow class、delivery semantics、state/version contract

**算法**

1. 以 `state version` 为主键构造 producer-consumer relation。
2. 对每条 relation 先求 `flow class`：
   - 单 producer -> 单 consumer：`point_to_point`
   - 单 producer -> 多 consumer 同版本：`broadcast`
   - 多 producer / join -> 单 consumer：`gather` 或 `reduce_merge`
   - recurrence / carry edge：`carry`
   - indirect index selected subset：`scatter` / `filtered`
3. 再求 `delivery semantics`：
   - 是否必须 ordered
   - 是否必须 completion-visible
   - 是否必须跨 phase materialize
   - 是否允许 async arrival / buffered transport
4. 如果 flow class 超出 `SpatialCapabilityModel` 能表达的 family，
   直接判非法，不允许偷偷降成 generic tensor flow。
5. 生成 `Channel` 时，显式绑定：
   - source task
   - target task
   - state / version target
   - flow class
   - delivery semantics

**不允许**

- 把所有边都压成同一种 `tensor_flow`
- 丢掉 version / delivery semantics 让 `Phase C` 再猜

#### 3.4.3 Domain Realization Algorithm

**输入**

- `Domain`
- `AccessMap`
- `UpdateLaw`
- `SemanticSupplement` 里的 derived/indexed/paged/grouped/chunk evidence
- `SpatialCapabilityModel`

**输出**

- `Layout`
- `WorkPartition`

**算法**

1. 从 semantic domain 取出 base iteration space。
2. 从 access map / supplement 识别 domain transform：
   - derived index
   - predicate filter
   - indirect gather/scatter
   - grouped / routed remap
   - paged indirection
   - chunked step decomposition
3. 先确定 `layout family`：
   - direct affine / separable access -> `regular`
   - packed contiguous subgroup -> `packed`
   - derived / indirect / paged index -> `indexed`
   - predicate-selected subset -> `filtered`
   - routed/grouped remap -> grouped/filter-index hybrid contract
4. 再确定 `partition family`：
   - separable multi-axis tiling -> `blocked`
   - scalar or replicated read-mostly domain -> `replicated`
   - index-driven subset ownership -> `indexed`
   - predicate- or route-selected subset -> `filtered`
5. 用 `SpatialCapabilityModel` 过滤不合法选择：
   - 不支持该 partition family 就 fail-fast
   - 不支持 required replication / sharding / neighborhood 就 fail-fast
6. 在合法候选中做 policy 选择：
   - 优先 locality preserving
   - 优先减少 cross-neighborhood traffic
   - 优先保留 stateful carry/reduction 所需稳定 ownership

**不允许**

- 只按轴数决定 `blocked / replicated`
- 把 paged / grouped / routed / chunked 全都退化成普通 indexed axes

#### 3.4.4 Phase And Ordering Synthesis Algorithm

**输入**

- `Task`
- `Channel`
- stateful/reduction/carry semantics
- `SpatialCapabilityModel`

**输出**

- `ProgramPhase`
- `SyncEdge`
- `phase_boundary_materialization` requirements

**算法**

1. 从 task graph 构造 must-happen-before relation。
2. 把以下边标成 ordering-critical：
   - carry
   - reduction completion
   - selection/index state handoff
   - explicit phase-boundary materialization
3. 对每条 ordering-critical edge 判断：
   - 是否允许同 phase 局部实现
   - 是否必须跨 phase
   - 是否需要 barrier / completion / async arrival
4. 把可局部闭包的子图合成同一个 phase。
5. 把必须 materialize 的边切成 cross-phase edge，并生成：
   - `SyncEdge`
   - `ResourceIntent(kind=phase_boundary_materialization)`
6. 对 phase condensation graph 做 topological ordering，得到稳定 `ProgramPhase` 序列。

**不允许**

- 只靠固定的 `phase0_compute / phase1_stateful` 模板命名
- 没有 ordering proof 就先生成 phase 再补 sync

#### 3.4.5 Capability-Informed Legality And Policy

**输入**

- 候选 `Task / Channel / Layout / WorkPartition / ProgramPhase`
- `SpatialCapabilityModel`

**输出**

- 合法候选空间
- policy 选中的 canonical `SpatialProgram`

**legality 必须回答**

- 该 flow class 是否被机器支持
- 该 layout / partition family 是否被机器支持
- 该 placement / neighborhood 假设是否被机器支持
- 该 ordering / sync family 是否可表达
- 该 residency / persistence 假设是否成立

**policy 才能回答**

- 在多个合法 task split/fuse 中选哪个
- 在多个合法 partition/layout 中选哪个
- 在多个合法 communication shaping 中选哪个

**不允许**

- policy 先选，再让 validator 兜底
- 因为当前样例“能跑”就把 capability 不支持的结构塞进 Spatial IR

## 4. 当前实施重点

当前 `Phase B` 的实施重点是：

1. 保住已经完成的 compile-path hardening，不回退到 legacy attr 主链
2. 把 `SpatialProgram` 从 structural scaffold 收紧成 execution-bearing contract
3. 让 `Layout / WorkPartition / Channel / SyncEdge / ProgramPhase`
   不只是结构 summary，而是稳定的执行语义载体
4. 保证 `Phase C` 只能做 TT mapping/materialization，不能再发明 spatial structure

## 5. 当前稳定边界

当前代码已经稳定成立的边界只有这些，其他内容不要再从本文件里读成“已完成实现”：

- `SpatialProgram / ProgramPhase / Task / Channel / Layout / WorkPartition /
  Placement / SyncEdge / ResourceIntent` object set 已落地主链
- `LowerToSpatialProgram -> ValidateSpatialProgram` 已位于
  `ValidateSemanticRefinement` 之后、`LowerBlackholeOps` 之前
- `LowerBlackholeOps` 已显式要求 `tl.spatial_program`，
  lowering-requirements 主路径不再接受 legacy-only spatial truth
- module-scope `ProgramPhase` truth 已聚合到 `tl.device_programs`
- representative compile-path family gate 已覆盖：
  `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`
- 第一轮 stronger-contract payload 已落地：
  - `SpatialLayout / WorkPartition.payload.domain_index`
  - `ResourceIntent.payload.target_kind / target_index`
  - `Task.payload.phase_index`
  - `Channel.payload.source_task_index / target_task_index / state_index`
  - `Placement.payload.task_index`
  - `SyncEdge.payload.source_task_index / target_task_index`
  - `ProgramPhase.payload.phase_index / task_indices / channel_indices`

当前不应再从本文件里继续继承的旧心智模型：

- `Phase B` 只是“对象化 summary”
- `Phase B` 还在依赖 `segment_plan / pipeline_stages / fragment_regions` 做主链判断
- `Phase B` 必须停在 compile-path hardening，不再承接更强 contract 设计

## 6. 未完成设计必须收实的部分

### 6.1 `SpatialCapabilityModel`

`SpatialProgram` 要成为有价值的 virtual spatial program，必须先引入
`SpatialCapabilityModel`。这不是 implementation hint，而是本阶段缺失的正式对象。

**宿主**

- module-scope global info：`IRModule.global_infos["tl.spatial_capability_model"]`

**producer**

- target-specific capability lowerer
- 对 Blackhole，初始 producer 从 concrete `TTHardwareModel` 导出抽象能力视图

**consumer**

- `LowerToSpatialProgram`
- `ValidateSpatialProgram`

**最小字段**

- `topology_class`
- `placement_domains`
- `communication_domains`
- `supported_flow_kinds`
- `supported_sync_kinds`
- `supported_layout_kinds`
- `supported_partition_kinds`
- `supported_residency_kinds`
- `replication_capabilities`
- `cross_domain_visibility_capabilities`

**边界**

- 它只能表达抽象能力，不表达 resource id、kernel kind、CB plan、ABI slot
- 它是 `Phase B` legality/policy 输入，不是 `Phase C` materialization 输出

### 6.2 `Task` Contract

当前 `Task` 只有 `kind + update_names + phase_index` 级别的信息，不够。
未完成设计必须补到下面这个强度：

**必须新增的 contract**

- `execution_signature`
  - `law_class`
  - `access_class`
  - `state_interaction_class`
- `formation_basis`
  - 哪些 semantic edge 迫使 split
  - 哪些条件允许 fuse
- `phase_class`
  - phase-local compute / stateful update / routing / combine / carry step
- `placement_constraints`
  - 是否要求 locality / communication adjacency / replicated execution

**算法输出要求**

- `Task` 必须能回答“为什么它是一个 task”，不是只回答“它叫什么”

### 6.3 `Channel` Contract

当前 `Channel` 还偏向“边 + state name”。
未完成设计必须补到下面这个强度：

**必须新增的 contract**

- `flow_kind`
  - `point_to_point`
  - `broadcast`
  - `gather`
  - `reduce_merge`
  - `carry`
  - `scatter`
- `delivery_kind`
  - `ordered`
  - `completion_visible`
  - `buffered_async`
  - `phase_boundary_materialized`
- `state_contract`
  - `state_index`
  - `source_version`
  - `target_version`
- `communication_constraints`
  - same-domain only / neighborhood-limited / cross-domain allowed

**算法输出要求**

- `Channel` 必须能回答“这条 flow 为什么存在、它要求什么可见性和顺序”

### 6.4 `Layout / WorkPartition` Contract

当前 `Layout / WorkPartition` 仍然太像 semantic domain 的外显化。
未完成设计必须补到下面这个强度：

**必须新增的 contract**

- `domain_realization_kind`
  - direct
  - packed
  - indexed
  - filtered
  - grouped
  - paged
  - chunked
- `transform_basis`
  - 来自哪个 `AccessMap`
  - 哪类 remap / filter / indirection / chunking 触发它
- `ownership_basis`
  - blocked / replicated / indexed-owner / filtered-owner
- `capability_requirements`
  - replication needed
  - neighborhood communication needed
  - stable carry ownership needed

**明确禁止**

- 只按轴数决定 `blocked / replicated`
- 把 grouped / paged / routed / chunked 全退化成 `indexed`

### 6.5 `ProgramPhase / SyncEdge` Contract

当前 `ProgramPhase` 和 `SyncEdge` 已有 linkage，但 ordering semantics 仍不够强。
未完成设计必须补到下面这个强度：

**必须新增的 contract**

- `ProgramPhase.phase_class`
  - local_compute
  - stateful_update
  - route_dispatch
  - combine
  - carry_step
- `ProgramPhase.closure_basis`
  - 为什么这些 task 可以在同一 phase 内闭包
- `SyncEdge.ordering_kind`
  - dependency
  - completion
  - barrier
  - async_arrival
- `SyncEdge.visibility_kind`
  - local
  - cross_phase
  - cross_member
- `SyncEdge.materialization_requirement`
  - 是否要求显式 phase-boundary state materialization

**算法输出要求**

- phase 不是显示分组，而是 partial-order condensation 的结果

### 6.6 `ValidateSpatialProgram` 的最终职责

当前 validator 已经不是空壳，但还不够。
未完成设计要把 validator 的职责固定成：

**必须 fail-fast 的错误**

- task formation contract 自相矛盾
- channel flow kind 与 state/version contract 不一致
- layout/work partition 与 domain/access basis 不一致
- phase closure basis 与 sync edge ordering 不一致
- capability requirement 超出 `SpatialCapabilityModel`
- module-scope `tl.device_programs` 与 member-local `tl.spatial_program`
  的 phase truth 不一致

**明确禁止**

- 只做结构串联检查
- 让 `Phase C` 发现 legality 问题后反向修正 `SpatialProgram`

## 7. 当前退出条件

`Phase B` 不再以“compile-path 已打通”作为完成标准。
当前真正的退出条件是：

1. `SpatialCapabilityModel` 落地，并被 builder/validator 主链消费
2. `Task / Channel / Layout / WorkPartition / ProgramPhase / SyncEdge`
   补齐上面列出的 execution-bearing contract
3. `ValidateSpatialProgram` 能对这些 contract 做 capability-informed legality 检查
4. `Phase C` translator 不需要再恢复 non-TT-specific spatial semantics

在这四条未达成前，`Phase B` 的 compile-path 虽然已完成，
但 `SpatialProgram` 还不能视为最终形态的 virtual spatial program。

## 8. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```
