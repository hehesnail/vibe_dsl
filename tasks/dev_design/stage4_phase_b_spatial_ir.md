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

## 5. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## 6. Task 3: Stage 3 - Phase B Spatial Program IR

**Files:**
- Create: `tilelang_repo/src/transform/lower_to_spatial_program.cc`
- Create: `tilelang_repo/src/transform/validate_spatial_program.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/src/transform/common/semantic_program.h`
- Modify: `tilelang_repo/src/transform/common/semantic_program.cc`
- Modify: `tilelang_repo/src/transform/collect_device_programs.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [x] **Step 1: Introduce `SpatialProgram` and `ProgramPhase`**

Required objects:

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

Rules:

- module-scope `ProgramPhase` truth lives in `tl.device_programs`
- member-local truth lives in `PrimFunc.attrs["tl.spatial_program"]`
- simple workload gets canonical fast-path

- [x] **Step 2: Add simple-workload fast-path for copy / GEMM**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'copy or gemm or fast_path' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
```

Expected:

- copy / GEMM 不需要进入重 candidate synthesis
- trivial workload 仍能快速构造 canonical `SpatialProgram`

- [x] **Step 3: Add one non-trivial multi-phase spatial gate**

Recommended first gate: `flash-attn`

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -k 'flash_attention or multi_phase' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- 至少一个 case 证明下游不会退化回 `Task:TTKernel = 1:1`
- `ProgramPhase` / `Channel` / phase-boundary materialization 在结构测试里可见

- [x] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [x] **Step 5: Stage 3 exit gate**

Only proceed when:

- `SpatialProgram` 能消费冻结后的 semantic truth
- simple-workload fast-path 稳定
- 至少一个 non-trivial multi-phase spatial gate 通过

### 6.1 2026-04-06 实施结果

- `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入 Blackhole 主线，位置在
  `ValidateSemanticRefinement` 之后、`LowerBlackholeOps` 之前
- `tl.device_programs` 现在会聚合 `ProgramPhase` truth；对 pre-`SplitHostDevice`
  的单 `PrimFunc` 退化场景，registry/validator 已支持 root-symbol fallback
- copy canonical fast-path：
  单 `transfer` task + 单 phase + 单 channel
- GEMM canonical fast-path：
  `reader / compute / writer` 三 task + 三 channel
- `flash-attn` 首个 non-trivial gate 已通过：
  `phase0_compute(reduce_*) -> phase1_stateful(recur_*)`
- 为保持 `Phase B` 只消费冻结后的 semantic truth，split device `main_kernel`
  路径缺失的 `row_reduction.kind` 已回补到 Phase A evidence，而不是让 `Phase B`
  直接回退消费 `fragment_regions`
- `LowerBlackholeOps` 已开始显式读取 `tl.spatial_program`，并把
  `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
  写入 `blackhole.lowering_requirements`

### 6.2 本轮验证

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

### 6.3 2026-04-06 Hardening Slice: Semantic-Domain-First Spatial Scaffolding

本轮 `Phase B` hardening 已落地一个 truth-source purity 子项：

- `LowerToSpatialProgram` 构造 `Layout / WorkPartition` 时，已改为优先读取
  `SemanticProgram.domains[*].axes / traits`
- `blackhole.work_decomposition` 仍允许作为过渡回退，但不再是 primary truth source

这一步当前覆盖到：

- `layout.axes`
- `work_partition.axes`
- `layout.kind == indexed` 的 `derived_indices` 判定

对应测试：

- 删除 `blackhole.work_decomposition` 后，`SpatialProgram` 仍能从
  `SemanticProgram.domain` 恢复正确的 axes / indexed layout

### 6.4 2026-04-06 Hardening Slice: Validator / Consumer / Family Gate

本轮继续补了三个直接面向 `Phase C` 的 hardening 子项：

- `ValidateSpatialProgram` 不再只看 object 是否“能串起来”，还会校验
  `SpatialLayout / WorkPartition` 与 `SemanticProgram.domain` 的一致性
  - `layout.axes` 必须和 semantic domain axes 一致
  - `work_partition.axes` 必须和 semantic domain axes 一致
  - `layout.kind == indexed` 必须和 semantic domain 的 `derived_indices` trait 对齐
- `LowerBlackholeOps` 的 `work_axes / derived_index_expr_count` 已改为优先从
  `tl.spatial_program` 恢复
  - `blackhole.work_decomposition` 仍保留 compatibility fallback
  - `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
    继续只从 `tl.spatial_program` 恢复
- `Phase B` family gate 已补到 `topk / selection`
  - `SpatialProgram` 现在有 compile-path 测试覆盖 `select + recurrence` workload family

对应测试：

- 伪造 semantic-domain 不一致的 `SpatialProgram` 时，`ValidateSpatialProgram`
  会 fail-fast
- 删除 `blackhole.work_decomposition` 后，`LowerBlackholeOps` 仍能从
  `tl.spatial_program` 导出 `work_axes / derived_index_expr_count`
- `example_topk` 已纳入 `Phase B` transform-level coverage，验证 selection/indexing
  family 不会立刻退化回 workload-specific matcher

### 6.5 2026-04-06 Hardening Slice: GEMM Builder Purity / Fragment Fallback

本轮继续补了两个直接指向 `truth-source purity` 和 `consumer cutover` 的缺口：

- `LowerToSpatialProgram` 的 GEMM fast-path 已不再读取 `blackhole.segment_plan`
  - `reader / compute / writer` task graph 现在直接从
    `SplitBlackholeKernel` 留在 IR body 上的 `blackhole.segment_kind`
    annotation 恢复
  - `segment_plan` 可以继续作为后续 lowering / runtime compatibility attr 存在，
    但不再参与 `SpatialProgram` 构造
- `LowerBlackholeOps` 的 fragment lowering requirements 已不再把
  `blackhole.fragment_regions` 当成唯一来源
  - 当 `fragment_regions` 存在时，仍优先消费这份 lowering-facing compatibility summary
  - 当 `fragment_regions` 缺失时，会退回到
    `SemanticProgram + residual body scan`
    恢复最小 fragment contract：
    `fragment_op_kinds / pointwise_op_kinds / row_reduction_targets /
    row_broadcast_sources / fragment_loop_carried_state`

对应测试：

- 删除 `blackhole.segment_plan` 后，GEMM 仍能恢复 `reader / compute / writer`
  spatial fast-path
- 删除 `blackhole.fragment_regions` 后，`LowerBlackholeOps` 仍能恢复
  `flash-attn` 所需的 fragment lowering requirements

### 6.6 2026-04-06 Hardening Slice: Stronger Phase / Registry Legality

本轮继续把 `ValidateSpatialProgram` 从“最小 semantic-domain legality”往真正的
spatial legality validator 再推进一格：

- `Placement` 现在必须引用已知 task，且 `member_func` 必须和当前
  `SpatialProgram.member_func` 一致
- `SyncEdge` 现在必须引用已知 task
- phase 不再只检查“引用的 channel 名存在”
  - phase 引用的 channel，其 `target_task` 必须属于该 phase
  - multi-phase 程序里，非首 phase 不能失去 channel contract
- `tl.device_programs` 不再只比较聚合出来的 phase 数量
  - 现在会逐项核对 phase signature：`name / task_names / channel_names`

对应测试：

- 伪造一个下游 phase 失去 channel contract 的 `SpatialProgram`，validator 会 fail-fast
- 伪造一个 member-local phase 名和 `tl.device_programs` 聚合 truth 不一致的
  `SpatialProgram`，validator 会 fail-fast

### 6.7 2026-04-06 Hardening Slice: Pipeline Contract Migration

本轮把 `pipeline_stages` 从 lowering-facing legacy attr 往 typed companion contract
再推进一层，目标是让 `LowerBlackholeOps` 不再把
`blackhole.pipeline_stages` 当 primary source：

- `AnalyzeSemanticStructure` 现在会把 `blackhole.pipeline_stages`
  收成 `SemanticProgram.supplements[*]`
  - kind: `pipeline_structure`
  - payload: `pipeline_stages[*].loop_var / num_stages / stage_local_buffers /
    loop_carried_state`
- `ResourceIntent` schema 已补上 typed `payload`
- `LowerToSpatialProgram` 现在会把 semantic supplement 投影成
  `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract)`
- `ValidateSpatialProgram` 现在会校验：
  - pipeline contract 必须真的携带 `pipeline_stages`
  - 每个 stage entry 必须携带 `loop_var / num_stages`
  - 语义侧要求 pipeline contract 时，`SpatialProgram` 不能缺失该 resource intent
- `LowerBlackholeOps` 现在优先从 `tl.spatial_program.resource_intents[*].payload`
  恢复 `pipeline_stage_counts / pipeline_loop_vars`
  - `blackhole.pipeline_stages` 退回 compatibility fallback
  - body `num_stages` annotation 退回最后 fallback

对应测试：

- `SpatialProgram` 必须显式暴露 pipeline contract resource intent
- 删掉 pipeline contract 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.pipeline_stages` 后，`LowerBlackholeOps` 仍能恢复
  `pipeline_stage_counts / pipeline_loop_vars`

### 6.8 2026-04-06 Hardening Slice: Work-Dependent Bound Contract Migration

本轮继续收 `blackhole.work_decomposition` 的 residual truth，但只针对它还在
`LowerBlackholeOps` 里承担唯一价值的那一项：

- `work_dependent_loop_bounds`

具体迁移如下：

- `AnalyzeSemanticStructure` 现在会把
  `blackhole.work_decomposition.work_dependent_loop_bounds`
  收成 `SemanticProgram.supplements[*]`
  - kind: `work_decomposition_structure`
  - payload: `work_dependent_loop_bounds[*]`
- `WorkPartition` schema 已补上 typed `payload`
- `LowerToSpatialProgram` 现在会把这份 truth 投影到
  `SpatialProgram.work_partitions[*].payload.work_dependent_loop_bounds`
- `ValidateSpatialProgram` 现在会校验：
  - semantic domain 带 `work_dependent_bounds` trait 时，
    `SpatialProgram` 不能丢失对应的 `WorkPartition` payload
  - payload 不能是空的
- `LowerBlackholeOps` 现在优先从 `WorkPartition.payload`
  恢复 `work_dependent_loop_bound_count`
  - `blackhole.work_decomposition` 只退回 compatibility fallback

对应测试：

- causal `flash-attn` 的 `SpatialProgram` 必须显式投影
  `work_dependent_loop_bounds`
- 删掉 `WorkPartition` payload 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.work_decomposition` 后，`LowerBlackholeOps` 仍能恢复
  `work_dependent_loop_bound_count`

### 6.9 2026-04-06 Hardening Slice: Fragment Contract Migration

本轮继续收 `blackhole.fragment_regions` 的 lowering-facing residual truth，但仍保持
它在 `Phase A` 里的 compatibility 身份，不把删除 attr 本身当目标。目标是：

- 让 `SpatialProgram` 开始显式持有 `fragment` lowering contract
- 让 `LowerBlackholeOps` 不再把 `blackhole.fragment_regions` 当 primary input

具体迁移如下：

- `AnalyzeSemanticStructure` 现在会把
  `blackhole.fragment_regions` 里的 lowering-facing summary 收成
  `SemanticProgram.supplements[*]`
  - kind: `fragment_lowering_structure`
  - payload:
    `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
    pointwise_op_kinds / fragment_loop_carried_state`
- `LowerToSpatialProgram` 现在会把这份 truth 投影成
  `ResourceIntent(kind=lowering_support, traits+=fragment_contract)`
- `ValidateSpatialProgram` 现在会校验：
  - semantic 侧要求 fragment contract 时，`SpatialProgram` 不能缺失该 resource intent
  - contract payload 必须显式携带 `fragment_op_kinds`
  - `pointwise_chain` / `row_broadcast` 不能丢失其从属 payload
- `LowerBlackholeOps` 现在优先从
  `tl.spatial_program.resource_intents[*].payload`
  恢复 fragment lowering requirements
  - `blackhole.fragment_regions` 退回 compatibility fallback
  - `SemanticProgram + residual body scan` 仍保留作无 attr 时的最后 fallback

对应测试：

- `flash-attn` 的 `SpatialProgram` 必须显式投影 fragment contract resource intent
- 删掉 fragment contract 后，`ValidateSpatialProgram` 会 fail-fast
- 删掉 `blackhole.fragment_regions` 后，`LowerBlackholeOps` 仍能恢复
  `fragment_op_kinds / pointwise_op_kinds`

### 6.10 2026-04-06 Hardening Slice: Chunk Recurrence Family Gate

本轮继续补 wider family gate，但不再新增定制样例，而是直接拿现有
`gdn/example_chunk_o.py` 作为非-attention recurrence family 的 compile-path 代表。

结论是：

- `chunk_o` 当前已经能稳定通过
  `LowerToSpatialProgram -> ValidateSpatialProgram`
- 其 `SpatialProgram` 会显式暴露：
  - multi-phase structure
  - `select + recurrence` task traits
  - `fragment_contract` resource intent

这说明当前 `Phase B` 的 generic spatialization，已经不只覆盖
`flash-attn / topk`，也覆盖至少一类 chunk recurrence family。

对应测试：

- `test_chunk_o_spatial_program_exposes_chunk_recurrence_family_gate`

### 6.11 2026-04-06 Hardening Slice: Routed/Paged Family Entry via Blackhole Resource Canonicalization

在补 `chunk_o` family gate 之后，`routed / paged` family 仍然没有进入 `Phase B`。
本轮继续顺着 compile-path 往前查，结论是：

- 真实 blocker 不是 `MergeSharedMemoryAllocations`
- 真实 blocker 是 `BlackholeDeviceResourceCanonicalization`
  只 canonicalize 了 `blackhole.resource_plan` 里显式列出的 resource
- 对 `grouped / routed / paged` 这类 kernel，block-local `shared` alloc_buffer
  常会留在 `shared.dyn`，没有被收成 `blackhole.cb.*`

因此本轮的修正是：

- 不动 TileLang 公用的 `MergeSharedMemoryAllocations`
- 只在 Blackhole-only 的
  `BlackholeDeviceResourceCanonicalization` 里补 IR-structural fallback
  - `shared* -> blackhole.cb`
  - `local.fragment -> blackhole.acc`
- fallback 依据是 storage scope，本身不依赖 workload 名字、buffer 名字或 case-specific matcher

这轮修正之后，compile-path 上的结果是：

- `grouped_gemm` 现在会把 `A_shared / B_shared`
  稳定 canonicalize 到 `blackhole.cb.*`
- `fusedmoe/example_fusedmoe_tilelang.py` 的 routed path
  现在能稳定进入 `LowerToSpatialProgram -> ValidateSpatialProgram`
- `deepseek_mla/example_mla_decode_paged.py`
  现在也能稳定进入 `LowerToSpatialProgram -> ValidateSpatialProgram`
- 它们的 `SpatialProgram` 都会暴露：
  - multi-phase structure
  - `select + recurrence` task traits
  - `selection_state` resource intent

这说明当前 `Phase B` 的 generic spatialization 已经不只覆盖
`copy / GEMM / flash-attn / topk / chunk recurrence`，也至少覆盖：

- 一个 `routed / grouped dispatch` family
- 一个 `paged / indexed sparse decode` family

对应测试：

- `test_grouped_gemm_resource_canonicalization_rewrites_shared_buffers_to_blackhole_cb`
- `test_fusedmoe_routed_spatial_program_exposes_routed_dispatch_family_gate`
- `test_paged_decode_spatial_program_exposes_paged_indexed_family_gate`

### 6.12 2026-04-06 Hardening Slice: `LowerBlackholeOps` Consumer Hard Cutover

在 family gate 补齐之后，`Phase B` 主链里最危险的残余问题已经不是 coverage，
而是 `LowerBlackholeOps` 的 lowering-requirements 构造仍允许走 legacy-only 输入。

这会制造一个错误心智模型：

- 代码里虽然已经有 `tl.spatial_program`
- 但 consumer 其实仍能在缺失 `SpatialProgram` 时回头读
  `blackhole.work_decomposition / fragment_regions / pipeline_stages`
- 结果测试和局部调试很容易继续绕开 `SemanticProgram -> SpatialProgram` 主链

本轮把这条边界做成 hard cutover：

- `LowerBlackholeOps` 现在显式要求 `tl.spatial_program`
  - 缺失时直接报错
  - 不再回退到 legacy-only 输入
- lowering requirements 里的下列字段现在只允许从 `SpatialProgram` 恢复：
  - `work_axes`
  - `derived_index_expr_count`
  - `work_dependent_loop_bound_count`
  - `spatial_phase_count`
  - `spatial_channel_count`
  - `spatial_phase_boundary_states`
  - `pipeline_stage_counts / pipeline_loop_vars`
  - `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
    pointwise_op_kinds / fragment_loop_carried_state`
- 旧的 residual fallback：
  - `blackhole.work_decomposition`
  - `blackhole.fragment_regions`
  - `blackhole.pipeline_stages`
  - body `num_stages` annotation
  已从 `LowerBlackholeOps` lowering-requirements 主路径删除

这轮同时也把 target/transform 测试统一收回真实主线：

- target tests 不再手搓 `SplitBlackholeKernel -> LowerBlackholeOps`
- 改为走测试侧 helper：
  `SplitBlackholeKernel -> Analyze* -> AnalyzeSemanticStructure ->
  LiftStatefulSemanticIR -> Validate* -> LowerToSpatialProgram ->
  ValidateSpatialProgram -> LowerBlackholeOps`

对应测试：

- `test_lower_blackhole_ops_requires_spatial_program_contract`
- `test_blackhole_copy_pass_attrs`
- `test_blackhole_copy_lowering_prefers_buffer_handles_over_annotation_names`
- `test_blackhole_copy_semantics_survives_flatten_and_vectorize`
- `test_blackhole_gemm_*` 下所有直接调用 `LowerBlackholeOps` 的 target tests
- `test_blackhole_flash_attention_pipeline.py` 的 `_lower_blackhole_ops` helper paths

## 7. Hardening Gates Before Phase C

`Phase B` 的目标不是“已经有一套 `SpatialProgram` 对象”，而是：

- 让 `Spatial Program IR` 真正成为 `SemanticProgram -> TTProgram` 之间唯一可信的
  spatial truth owner

只有满足下面这些 gate，`Phase C` 才允许开始做正式 cutover。

### 7.1 Truth-Source Purity

`LowerToSpatialProgram` 必须完成真源纯化：

- `Phase B` 只能消费冻结后的 `SemanticProgram`
- `blackhole.work_decomposition`
- `blackhole.segment_plan`
- `blackhole.pipeline_stages`
- `blackhole.fragment_regions`

这些 attr 都不能再作为 spatial truth source。

允许的过渡状态只有两种：

1. 这些信息已经被 `Phase A` 归约成 semantic truth
2. 这些信息只作为 lowering compatibility summary 存在，不再参与 `SpatialProgram`
   构造决策

当前代码仍未完全达标：

- `Layout / WorkPartition` 已切到 semantic-domain-first，
  但 `blackhole.work_decomposition` 仍保留 compatibility fallback
- `pipeline_stages` / `fragment_regions` 仍保留 lowering-facing compatibility 角色，
  但都已开始迁入 typed contract，尚未完全删掉 fallback

本轮新增的明确 cutover 设计是：

- `blackhole.pipeline_stages` 不能被“直接删掉”
- `AnalyzeSemanticStructure` 必须先把 pipeline stage truth 收成
  `SemanticProgram.supplements[*]`
- `LowerToSpatialProgram` 再把这份 truth 投影成 `SpatialProgram.resource_intents[*]`
  里的 typed pipeline contract
- 只有在 `LowerBlackholeOps` 能从 `tl.spatial_program` 直接恢复
  `pipeline_stage_counts / pipeline_loop_vars` 之后，
  `blackhole.pipeline_stages` 才能降成纯 compatibility fallback
- `blackhole.fragment_regions` 也不能被“直接删掉”
- `AnalyzeSemanticStructure` 必须先把 fragment lowering truth 收成
  `SemanticProgram.supplements[*]`
- `LowerToSpatialProgram` 再把这份 truth 投影成
  `SpatialProgram.resource_intents[*]` 里的 typed fragment contract
- 只有在 `LowerBlackholeOps` 能从 `tl.spatial_program` 直接恢复
  `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
  pointwise_op_kinds / fragment_loop_carried_state` 之后，
  `blackhole.fragment_regions` 才能继续往纯 compatibility fallback 收敛

### 7.2 Schema Strengthening

当前 `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
已经 object 化，但仍偏向 `name / kind / traits` summary。

进入 `Phase C` 前，至少要补到足以稳定承载下面这些 spatial-owned truth：

- task ownership
- channel payload / source-state / versioned-state flow
- layout / work-partition 的结构化依据
- phase-boundary materialization contract
- cross-member phase ordering
- synchronization semantics

原则是：

- 不能让 `Phase C` 再回头猜 task graph / state flow / phase boundary
- 不能把这些 truth 继续散落在 legacy `blackhole.*` attr 里
- 不能把 `SpatialProgram` 只做成结构化 display object；schema 必须足够强，
  让 `Phase C` 只能消费它、不能重新合成它

当前 schema strengthening 的真正方向不是“继续补更多字段”，而是把下列 execution-bearing
truth 变成稳定 contract：

- task formation basis
- flow / delivery / state-version semantics
- domain remap / filter / index / shard semantics
- phase-local 与 cross-phase partial order
- spatial-owned but non-TT-specific resource obligations

本轮优先补强的 schema 子项是：

- `ResourceIntent` 需要从纯 `name / kind / target / traits` summary
  升到可携带 typed payload 的 contract node
- pipeline legality 相关的 spatial-owned truth
  （`loop_var / num_stages / stage_local_buffers / loop_carried_state`）
  先挂在 `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract)` 上
- fragment legality / lowering 相关的 spatial-owned truth
  （`fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
  pointwise_op_kinds / fragment_loop_carried_state`）
  先挂在 `ResourceIntent(kind=lowering_support, traits+=fragment_contract)` 上
- 这不是最终 schema 终点，但它必须足以支撑
  `LowerBlackholeOps` 不再把 `blackhole.pipeline_stages / fragment_regions`
  当 primary source

本轮已经先落了一层 **index-based linkage contract**：

- `SpatialLayout.payload.domain_index`
  与 `WorkPartition.payload.domain_index`
  现在是 validator 和 consumer 的 primary domain linkage
- `ResourceIntent.payload.target_kind + target_index`
  现在是 semantic-state-targeted contract 的 primary linkage
- `LowerBlackholeOps` 的 phase-boundary state 恢复已经切到
  `semantic_state[target_index]`，不再按 `target_name` 字符串恢复
- `Task / Channel / Placement / SyncEdge / ProgramPhase`
  现在也开始显式携带 linkage payload：
  - `Task.payload.phase_index`
  - `Channel.payload.source_task_index / target_task_index / state_index`
  - `Placement.payload.task_index`
  - `SyncEdge.payload.source_task_index / target_task_index`
  - `ProgramPhase.payload.phase_index / task_indices / channel_indices`
- `ValidateSpatialProgram` 现在会显式要求这些 payload contract 存在，
  不能再只靠 `phase_name / task_name / source_task / target_task / channel_names`
  这些 display 字段把结构“串起来”

下一轮 schema strengthening 的优先级不再是继续塞更多名字字段，而是继续把这层
index-based linkage 扩到更多 object：

- `SpatialLayout / WorkPartition` 必须显式携带 `domain_index`
- `ResourceIntent` 必须显式携带 `target_kind + target_index`
- `Task` 必须显式携带 `phase_index`
- `Channel` 必须显式携带
  `source_task_index / target_task_index / state_index`
- `Placement` 必须显式携带 `task_index`
- `SyncEdge` 必须显式携带 `source_task_index / target_task_index`
- `ProgramPhase` 必须显式携带
  `phase_index / task_indices / channel_indices`

目的不是“把名字删光”，而是把名字降级成 display/identity 字段，让跨层 consumer 和
validator 优先吃显式 linkage contract，而不是靠 `state_name / target_name / task_name`
字符串重新查表。

### 7.3 Legality Must Be Explicit

`ValidateSpatialProgram` 当前已经具备最小 semantic-domain legality gate，但还不等于完整的
spatial legality validator。

进入 `Phase C` 前，validator 至少要能 fail-fast 检出：

- phase order / phase boundary 不一致
- channel source-target-state 绑定不一致
- layout / partition 与 semantic domain 不一致
- multi-phase state materialization 缺失
- module-scope `tl.device_programs` 与 member-local `tl.spatial_program`
  的 cross-function truth 不一致

本轮已新增的显式 contract：

- phase-channel contract 必须真正落到 owning phase
- downstream multi-phase phase 不能没有 channel contract
- module-scope aggregated phase truth 不能只在 phase 数量上“凑巧一致”
- pipeline domain program 不能丢失 pipeline contract resource intent

规则仍然是：

- analysis 决定 legality
- policy 只在合法空间内选择

### 7.4 Lowering Consumer Cutover

`LowerBlackholeOps` 在 `Phase B` 完成前必须收窄成 spatial consumer，而不是继续承担
残余 spatial recovery。

目标状态：

- `LowerBlackholeOps` 只读取 `SpatialProgram` 提供的 lowering contract
- 对 task / channel / layout / sync / phase-boundary 的判断不再回头读取
  `fragment_regions` / `work_decomposition` / `pipeline_stages`

当前仍未完全达标：

- lowering requirements 这条 consumer path 已经 hard cutover：
  `LowerBlackholeOps` 显式要求 `tl.spatial_program`，
  不再接受 legacy-only `work_decomposition / fragment_regions / pipeline_stages`
  输入
- target/transform tests 也已统一到 `SemanticProgram -> SpatialProgram ->
  LowerBlackholeOps` 主线

当前还没完全完成的，不再是这条 consumer path 本身，而是：

- `LowerToSpatialProgram` 仍保留对 `blackhole.work_decomposition`
  的过渡回退
- `ValidateSpatialProgram` 还不是最终形态的完整 legality validator
- `SpatialProgram` schema 还需要继续把 task/channel/placement/phase linkage
  从名字查表收成更强 contract，才能无保留进入 `Phase C`

这类 mixed ownership 在 `Phase C` 之前必须继续收紧。

### 7.5 Family Coverage Gates

当前 `copy / GEMM / flash-attn / topk / chunk recurrence / routed dispatch / paged decode`
的 compile-path coverage，已经说明 `SemanticProgram -> SpatialProgram` 的 object boundary
不会一离开 attention/GEMM 就立刻退化回 matcher bag；但这仍不能替代 truth-source purity /
schema strengthening / consumer cutover。

在进入 `Phase C` 前，`Phase B` 至少需要补齐：

1. 一个 `selection / indexing` family 的 spatial gate
   - 推荐第一项：`topk`
2. 一个非 attention 的更复杂 family gate
   - 推荐从下面三者中至少打一项：
   - `routed / grouped dispatch`
   - `paged / indexed sparse decode`
   - `chunk recurrence / scan`

当前状态：

- `selection / indexing` 已经通过 `topk`
- `chunk recurrence / scan` 已经至少通过一项（`gdn/example_chunk_o.py`）
- `routed / grouped dispatch` 已经至少通过一项
  （`fusedmoe/example_fusedmoe_tilelang.py`）
- `paged / indexed sparse decode` 已经至少通过一项
  （`deepseek_mla/example_mla_decode_paged.py`）

因此 family coverage gate 的最小 compile-path 要求现在已经满足；
`Phase B` 之所以仍不能进入 `Phase C`，剩下卡的不是 family gate，而是：

- truth-source purity 还没完全收干净
- schema strengthening 还没够强
- `ValidateSpatialProgram` 还不是完整 legality validator
- `LowerBlackholeOps` 还没有完全切成只读 spatial contract 的 consumer

要求不是“所有 runtime correctness 一次做完”，而是：

- compile-path 上能证明 `SemanticProgram -> SpatialProgram` 的 object boundary 成立
- spatial object set 不会因新 family 立刻退化回 workload-specific matcher

### 7.6 Module-Scope Program Truth

`tl.device_programs` 不能只停留在“聚合 phase 列表”的最小实现。

进入 `Phase C` 前，它必须稳定承载 cross-function spatial truth，至少包括：

- phase order
- member_func ownership
- cross-member phase-boundary truth

单 `PrimFunc` 程序仍只是退化情况，不应反向成为默认心智模型。

## 7.7 2026-04-06 Hardening Execution Order

`Phase B` 的剩余问题必须按 fixed order 收口，不能按“哪个 attr 看起来更顺手”零散修补。

本轮 hardening 的执行顺序固定为：

1. 先把 `ValidateSpatialProgram` 从结构检查升级到最小 legality gate
   - phase / task / channel / layout / work-partition 的 cross-reference 必须一致
   - multi-phase 程序除了 `phase_boundary_materialization` 外，还必须证明 phase 间真的有
     channel 或 boundary state contract
2. 再把 `LowerBlackholeOps` 收窄成 spatial consumer
   - `spatial_phase_count / spatial_channel_count / spatial_phase_boundary_states`
     只允许从 `tl.spatial_program` 恢复
   - `work_axes / derived_index_count` 应优先从 `SpatialProgram.layouts /
     work_partitions` 恢复，不再把 `blackhole.work_decomposition` 当 primary source
3. 最后扩 family gate
   - 至少补一个非 `copy / GEMM / flash-attn` 的 compile-path spatial case，确认当前 schema
     不会一离开这三类 workload 就退化

这轮 hardening 的明确边界：

- `segment_plan` 已不再参与 `SpatialProgram` builder
- `LowerToSpatialProgram` 不再把 `work_decomposition` 当 builder truth source
- `work_decomposition` / `fragment_regions` / `pipeline_stages` 仍可在更后段保留
  compatibility path
- 但不能让这些 legacy attrs 继续外溢成 validator、generic builder，
  或 `LowerBlackholeOps` 的 primary truth source

另外，generic builder 不能再用 `root_map` 之类的 update name 做协议分支：

- generic path 按 semantic `Update` object 自身建 task
- update name 只作为 IR object identity、调试和打印，不承担语义分流职责

## 8. Current Gap Inventory

基于当前实现状态，`Phase B` 的当前结论应表述为：

1. `SemanticProgram -> SpatialProgram` compile-path cutover 已完成
2. spatial builder / validator / consumer 的 primary truth 已切回 typed companion IR
3. `ValidateSpatialProgram` 当前会把 semantic statefulness 显式投影成最小 legality：
   stateful semantic states 必须对应 `state_residency`，multi-phase program 还必须覆盖
   每个 stateful state 的 `phase_boundary_materialization`
4. representative family gate 已证明当前 object boundary 不会立刻退化回 workload-specific matcher
5. stronger-contract schema 的第一轮已经落地：
   `domain_index` 与 `target_kind / target_index`
   已进入 `SpatialProgram` payload contract，并被 validator / consumer 主链消费
6. stronger-contract schema 的第二轮也已经落地：
   `Task / Channel / Placement / SyncEdge / ProgramPhase`
   已补齐第一批 `*_index / *_indices` linkage payload，
   `ValidateSpatialProgram` 已切成 contract-first
7. 后续更强 schema / legality contract 仍然需要，但它们现在属于 `Phase C`
   translator 驱动的下一轮增强，而不是继续停在 `Phase B` 的 blocker
