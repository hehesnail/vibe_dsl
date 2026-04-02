# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19（创建），2026-04-02（重写并收敛为当前版本）
- **状态**: 当前唯一权威总体设计文档
- **范围**: `tilelang_repo` Blackhole 编译器架构、compiler-internal IR 分层、TT 目标映射、runtime materialization 边界
- **取代**:
  - 已归档的混合 runtime 架构叙事：`tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
  - 已归档的旧单层 `Stateful Tiled IR` 顶层方向

## 1. 问题定义

Blackhole 现在面对的核心问题，已经不是“怎么再打印一个 TT-Metal kernel 字符串”，而是 **复杂前端计算语义和 spatial/dataflow 硬件编程模型之间的结构性表示鸿沟**。

这个鸿沟在仓库里已经不是抽象问题，而是被多类真实样例共同暴露出来：

- `examples/flash_attention/`、`examples/online_softmax/`、`examples/attention_sink/`
  - 暴露的是 `stateful reduction-update / carry / multi-phase` 语义
- `examples/fusedmoe/`、`examples/grouped_gemm/`
  - 暴露的是 `routed / grouped / ragged dispatch` 语义
- `examples/topk/`、`examples/deepseek_v32/topk_selector.py`
  - 暴露的是 `selection / index generation / selected subset` 语义
- `examples/blocksparse_attention/*paged*.py`、`examples/deepseek_mla/example_mla_decode_paged.py`、`examples/flash_decoding/`
  - 暴露的是 `paged / indexed / sparse decode` 语义
- `examples/linear_attention/example_mamba_chunk_state.py`、`examples/kda/`、`examples/gdn/`
  - 暴露的是 `chunked recurrence / scan / cross-phase state` 语义

这些 workload family 虽然长相不同，但它们有共同特征：

1. `PrimFunc / TIR` 在 tile op 被 lower、fragment 被 scalarize 之后，无法稳定保留：
   - state
   - phase
   - relation
   - routed / paged / ragged / indexed domain
   - multi-phase carry/update
2. TT-Metal 和 TT 类 spatial/dataflow 硬件要求显式程序结构：
   - task / kernel role
   - channel / communication edge
   - circular buffer
   - semaphore / multicast / synchronization edge
   - dst/register layout
   - core placement / work distribution
   - compile-time / runtime ABI
3. 当前主链把三类本应分层处理的责任混在了一起：
   - 从被压碎的 TIR 中恢复算法语义
   - 发明空间程序结构
   - 选择 TT-specific 资源与 ABI

`blackhole.acc` 混合语义只是这一结构问题在当前 attention-like consumer 上暴露得最明显的一个症状，不是本设计的目标边界，也不是设计本身的中心。

**设计总论点**：
编译器不能继续试图在一个层里同时解决 semantic recovery、spatial organization 和 TT target planning。下一阶段架构必须是多层 compiler-internal IR：

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

每一层只承接自己的语义真相。下层可以消费上层冻结后的事实，但不能反向猜测上层。

## 2. 设计目标与非目标

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定。
2. 结束 late target-specific semantic guessing。
3. 让 state、phase、relation、layout、task、sync、TT resource planning 在正确层里成为一等对象。
4. 在 target 层保持 TT-first，但让 semantic 层和 spatial 层具备超出 TT 的抽象价值。
5. 让 codegen/runtime 退回 materialization 和 execution，而不是继续承担语义重建。
6. 让这套设计不仅能解释当前 `flash-attn`，还要能承接其他复杂前端计算 family。

### 2.2 非目标

1. 不设计 TT-Metal 专用用户 DSL。
2. 不把 `task / channel / CB / semaphore / runtime_args` 暴露成 Python 前端的一等编程概念。
3. 不把全部复杂度重新塞回一个“super IR”。
4. 不引入第二条正式执行路径；当前正式路径仍是 direct host path。
5. 不为了某一个 consumer 单独固化协议或 matcher。

## 3. 当前硬约束

这次重设计不是 greenfield compiler，而是在现有 Blackhole 主链上重构边界。

1. `BlackholeModule` 进程内 direct host path 仍是唯一正式执行路径。
2. `ExecutableSpec` 仍是 runtime 消费的最终物化产物。
3. Stage 0-3 已经落地的基线不能推倒重来：
   - `ExecutableSpec`
   - `rt_mod_blackhole`
   - `BlackholeModule`
   - copy / GEMM / multi-core direct host path
4. 现有 recovery-oriented analysis pass 仍然是 semantic recovery 的起点：
   - `AnalyzeBlackholeWorkDecomposition`
   - `AnalyzeBlackholeFragmentRegions`
   - `AnalyzeBlackholePipelineStages`
5. `PlanBlackholeCB`、`AssignBlackholeCores`、`rt_mod_blackhole` 在迁移期间仍保留，但它们的长期归属要收回到新的 target/runtime 边界里。

## 4. 权威架构

### 4.1 总流程

```text
TileLang DSL / Python
  -> PrimFunc / TIR
  -> Semantic Recovery
  -> Stateful Semantic IR
  -> Semantic Validation
  -> Spatialization
  -> Spatial Program IR
  -> Spatial Validation
  -> Hardware-Aware Mapping
  -> TT Target IR
  -> Target Validation
  -> MaterializeTTExecutableSpec
  -> Codegen / rt_mod_blackhole / BlackholeModule
```

### 4.2 各层摘要

| 层 | 它回答的问题 | 真源 | 典型产物 |
|----|--------------|------|----------|
| `PrimFunc / TIR` | 用户在标准编译管线里写了什么？ | 通用 TileLang / TVM IR | 规范化 TIR |
| `Stateful Semantic IR` | 这个程序到底在算什么？ | 算法语义 | `SemanticProgram` |
| `Spatial Program IR` | 这个算法应该如何组织成 spatial/dataflow 程序？ | task/channel/layout/sync/work 图 | `SpatialProgram` |
| `TT Target IR` | 这个 spatial program 如何变成合法的 TT-Metal contract？ | TT 资源与 ABI 合约 | `TTProgram` |
| `ExecutableSpec / runtime` | 冻结的 TT contract 如何被 materialize 并执行？ | 目标物化 schema | 可执行 `ExecutableSpec` 与 host 对象 |

### 4.3 设计输入

这套分层设计主要借鉴四类研究方向：

- `T2S`
  - 算法语义与空间映射必须分层
- `Dato`
  - `task / channel / layout` 应该是一等表示
  - `virtual -> physical mapping` 应分两阶段
- `TL`
  - hardware representation 与 mapping 是编译器主问题，不是 codegen 后处理
- `SPADA`
  - routing 与 synchronization correctness 需要显式 validation

这些论文是设计输入，不是协议源；TileLang Blackhole 的协议仍以本仓库 IR 和实现边界为准。

### 4.4 工作负载族覆盖矩阵

当前分层 IR 的设计目标，不是“解释 flash-attn”，而是覆盖下面这些复杂前端计算 family：

| 工作负载族 | 仓库示例 | `Stateful Semantic IR` 必须表达 | `Spatial Program IR` 必须表达 | `TT Target IR` 必须冻结 |
|------------|----------|--------------------------------|-------------------------------|-------------------------|
| Dense tiled compute | `copy`、`GEMM`、`grouped_gemm`、`split-k` | tile domain、普通 tensor state、reduce relation | load/compute/store task、tile layout、split-k partition | reader/compute/writer、CB transport、ABI、core placement |
| Selection / indexing | `examples/topk/`、`deepseek_v32/topk_selector.py` | `index_state`、`selected_from`、`reduce/select` region | select task、index channel、selected-subset partition | index scratch、selector/reduction kernel role、index ABI |
| Routed / grouped / ragged dispatch | `examples/fusedmoe/` | grouped/ragged domain、segment descriptor、expert index、weight relation | route/compute/combine task、grouped layout、expert partition、ragged sync | routed buffer/index buffer、dispatch ABI、core-group mapping |
| Paged / indexed / sparse decode | `blocksparse_attention/*paged*.py`、`deepseek_mla/example_mla_decode_paged.py`、`flash_decoding/` | paged/index-remapped domain、block/page index state、bounded predicate | page-stream task、paged layout、page partition、merge/combine sync | page/index buffer、runtime descriptor、multicore split merge plan |
| Stateful reduction-update | `flash_attention/`、`online_softmax/`、`attention_sink/`、`norm/` | carry state、combine function、cross-phase relation、bound/predicate | update task、carry channel、persistent resource intent、multi-phase sync | dst layout、persistent carry plan、reader/compute/writer contract |
| Chunked recurrence / scan | `linear_attention/example_mamba_chunk_state.py`、`kda/`、`gdn/` | cross-phase state、chunk domain、recurrence relation、decay/update combine | chunk task graph、chunk partition、state carry、phase boundary sync | persistent carry、dst/CB realization、execution order、runtime chunk descriptors |

这张表是总设计的边界声明。后续实现顺序可以从其中某一个 family 起步，但总设计不能被单一 consumer 绑死。

## 5. 各层 IR 设计

### 5.1 `Stateful Semantic IR`

#### 为什么需要这一层

这一层只回答一个问题：**程序的算法语义到底是什么？**

如果没有这一层，下游只能从下面这些东西里倒推语义：

- 被 scalarize 的 `BufferLoad / BufferStore`
- fragment helper 名字
- target builtin
- runtime 侧 heuristics

这不仅会在 attention-like kernel 上制造 `blackhole.acc` 之类的混合语义，也会在 MoE、paged decode、chunk recurrence 这类 workload 上把 `group/domain/index/carry` 信息压碎到后段无法稳定恢复。

#### 设计目标

1. 在做任何 TT 资源决策之前，先冻结算法语义真相。
2. 显式表示 state、phase、relation、domain constraint。
3. 用统一语义词汇覆盖：
   - selection / topk
   - routed / grouped / ragged dispatch
   - paged / indexed sparse access
   - stateful reduction-update
   - recurrence / scan

#### 核心对象

| 对象 | 关键字段 | 含义 |
|------|----------|------|
| `SemanticProgram` | `domains`, `states`, `relations`, `phases`, `regions` | 语义真相容器 |
| `Domain` | `kind`, `iter_vars`, `bound_expr`, `predicate`, `index_remapping`, `segment_descriptor` | 逻辑迭代域 |
| `State` | `kind`, `lifetime`, `shape`, `value_semantics` | 可更新算法状态 |
| `Relation` | `kind`, `source`, `target`, `combine_function`, `mapping_expr` | state/domain/region 之间的语义关系 |
| `Phase` | `kind`, `live_in`, `live_out`, `regions` | 算法阶段边界 |
| `SemanticRegion` | `op_family`, `reads`, `writes`, `domain`, `phase` | 语义上单一职责的一段计算 |

#### 对象设计

**`Domain`**

- `kind`:
  - `dense`
  - `segmented`
  - `routed`
  - `paged`
  - `ragged`
- 必须支持：
  - `bound_expr`
    - 例如 causal bound、chunk 边界、data-dependent range
  - `predicate`
    - 例如 mask、valid row、valid page
  - `index_remapping`
    - 例如 grouped head remap、block/page indirection、logical-to-selected subset mapping
  - `segment_descriptor`
    - 例如 `group_sizes / group_offsets / group_padded_offsets`
    - 它是语义上的 grouped/ragged domain 描述，不是 target block table

**`State`**

- `kind`:
  - `matrix_state`
  - `vector_state`
  - `scalar_state`
  - `index_state`
- `index_state` 专门承接：
  - topk indices
  - block/page indices
  - group/expert ids
  - selected subset descriptors
- `lifetime`:
  - `ephemeral`
  - `carry`
  - `cross_phase`
- 这一层只描述状态角色，不描述 TT backing resource

**`Relation`**

- `kind` 建议至少覆盖：
  - `reduced_from`
  - `applies_to`
  - `indexes`
  - `gathers_from`
  - `scatters_to`
  - `selected_from`
  - `partitioned_by`
  - `carried_across`
- `combine_function` 不能被压成一个过小 enum；它必须能表达：
  - online softmax rescale/update
  - Welford
  - chunk recurrence update
  - weighted combine

**`Phase`**

- 表达 `algorithm phase`
- 不表达 reader/compute/writer
- 不表达 TT pipeline kernel role

**`SemanticRegion`**

- `op_family` 至少要覆盖：
  - `matmul`
  - `reduce`
  - `normalize`
  - `select`
  - `gather`
  - `scatter`
  - `dispatch`
  - `combine`
  - `recurrence`
- `source_anchor`
  - 指向生成该 region 的结构锚点
  - 它是 semantic/TIR 桥接对象，不是算法语义本体

#### 输入

- `PrimFunc / TIR`
- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`
- 只有在 IR 无法稳定恢复语义时，才允许最小 Python annotation

#### 输出

- 冻结后的 `SemanticProgram`

#### 验证职责

1. `state kind / lifetime / shape` 一致性
2. `carried_across` 与 `live_in / live_out` 一致性
3. `bound_expr / predicate / index_remapping / segment_descriptor` 完整性
4. `combine_function` 存在性与连接性
5. 禁止同一个对象同时充当算法 state 和 target scratch

#### 明确不属于这一层的内容

- `reader / compute / writer`
- `task / channel / placement`
- `CB / semaphore / dst offset / core group`
- compile-time ABI
- runtime ABI
- carry 的 TT 实现策略

#### 与 TIR 的桥接对象

`Stateful Semantic IR` 不是悬空对象图。它需要直接持有对 TIR 世界的类型化引用，但不能长期依赖“某棵旧 TIR 子树的地址”。

因此，这一层额外引入两类桥接对象：

| 对象 | 关键字段 | 作用 |
|------|----------|------|
| `TIRAnchor` | `func`, `anchor_id`, `reads`, `writes`, `iter_vars`, `span` | 语义区域对应的结构锚点 |
| `TIRValueBinding` | `func`, `buffer`, `var`, `expr`, `anchor_id` | semantic object 到 TIR 原子的直接绑定 |

设计要求：

1. `State.backing_buffer`、`Domain.iter_vars`、`Domain.predicate`、`Relation.mapping_expr` 等字段，优先直接持有 `Buffer / Var / PrimExpr`。
2. `SemanticRegion` 不长期保存某个 `Stmt*` 或 `Block*` 指针，而是保存 `source_anchor : TIRAnchor`。
3. `TIRAnchor` 是编译器在固定 canonicalization 点上重建出来的结构身份，不是用户可见名字，也不是 case-specific 语义标签。
4. semantic 层与 TIR 的关系必须是显式对象引用，不允许退化成字符串名匹配或位置假设。

#### 语义恢复的具体规则

`SemanticProgram` 不是把 analysis pass 的零散结论直接拼起来，而是要经过一套稳定的恢复流程：

1. **域恢复**
   - 从 loop nest、launch/work decomposition、predicate、indirect index expr 中恢复 `Domain`
   - 产出 `bound_expr / predicate / index_remapping / segment_descriptor`
2. **状态恢复**
   - 从 def-use、buffer scope、lifetime、loop-carried use 恢复 `State`
   - 判定 `ephemeral / carry / cross_phase`
3. **关系恢复**
   - 从读写索引关系、归约模式、indirect access、cross-iteration use 恢复 `Relation`
   - 判定 `reduced_from / indexes / selected_from / partitioned_by / carried_across`
4. **阶段恢复**
   - 从 loop-carried edge、pre/post-loop 使用、显式 pipeline/stage signal 恢复 `Phase`
   - 明确 `live_in / live_out`
5. **区域恢复**
   - 从 `AtomicEffect` 图按通用切分规则聚成 `SemanticRegion`

这五步里，前四步都是在建立“语义事实”；只有最后一步才在事实之上做 region partition。

#### semantic 层哪些来自分析，哪些必须显式提供

默认优先从结构恢复：

- loop/domain/bounds/predicate
- 普通 reduce / gather / scatter / carry
- 基于 indirect expr 的 index remap
- 基于 def-use 的 state lifetime

当以下语义无法仅靠结构稳定恢复时，必须允许前端或早期 IR 提供最小显式 hint：

- route / segment 的高层含义
- selection policy
- combine function 的高层类别
- page/block descriptor 的外部协议语义

规则是：

1. 能从 IR 稳定恢复的，必须分析得到。
2. 不能稳定恢复的，不允许后段猜，必须由更早层显式补齐。

### 5.2 `Spatial Program IR`

#### 为什么需要这一层

这一层回答的是另一个问题：**这个算法应该如何组织成 spatial/dataflow 程序？**

`task / channel / layout / sync` 不是算法真相，但也明显高于 TT-specific 资源。如果跳过这一层，通常只会出现两种坏结果：

1. semantic IR 被执行拓扑污染，不再是 semantic
2. TT target lowering 重新长成一个黑洞，同时发明 task graph 和 TT resource plan

#### 设计目标

1. 显式表示 task graph、channel graph、layout、work partition、sync。
2. 在 target planning 之前，就让 routing/synchronization/layout 成为可验证对象。
3. 保持这一层足够通用，使 TT-specific 细节不上窜。

#### 核心对象

| 对象 | 关键字段 | 含义 |
|------|----------|------|
| `SpatialProgram` | `tasks`, `channels`, `layouts`, `work_partitions`, `placements`, `sync_edges`, `resource_intents` | spatial 程序容器 |
| `Task` | `kind`, `semantic_regions`, `input_ports`, `output_ports`, `execution_scope` | 逻辑执行单元 |
| `Channel` | `producer`, `consumer`, `payload_kind`, `transport_semantics`, `ordering` | task 间数据流边 |
| `Layout` | `layout_kind`, `partition_axes`, `mapping_expr` | 分布式数据组织 |
| `WorkPartition` | `partition_kind`, `partition_expr`, `load_balance_policy` | 逻辑工作划分 |
| `Placement` | `placement_kind`, `constraints` | virtual placement 关系 |
| `SyncEdge` | `kind`, `scope`, `source_task`, `target_task` | 同步要求 |
| `ResourceIntent` | `kind`, `capacity_hint`, `visibility`, `reuse_policy` | 尚未 target bind 的资源需求 |

#### 对象设计

**`Task`**

- `kind` 示例：
  - `load`
  - `compute`
  - `reduce`
  - `select`
  - `route`
  - `exchange`
  - `combine`
  - `store`
- 一个 `Task` 不等于一个 TT kernel

**`Channel`**

- `payload_kind` 示例：
  - `tile`
  - `vector`
  - `scalar`
  - `index`
  - `token_batch`
  - `page`
- `transport_semantics` 示例：
  - `fifo`
  - `multicast`
  - `gather`
  - `scatter`
  - `reduce`

**`Layout`**

- `layout_kind` 示例：
  - `tile`
  - `shard`
  - `grouped`
  - `routed`
  - `paged`
  - `ragged`

**`WorkPartition`**

- `partition_kind` 示例：
  - `row`
  - `tile`
  - `pair`
  - `split_k`
  - `expert`
  - `page`
  - `chunk`
  - `selected_subset`

**`Placement`**

- 只表达 virtual placement
- 示例：
  - `colocated`
  - `adjacent`
  - `row_group`
  - `column_group`

**`SyncEdge`**

- 示例：
  - `producer_consumer`
  - `barrier`
  - `completion`
  - `multicast_ready`

**`ResourceIntent`**

- 示例：
  - `transport_buffer`
  - `scratch`
  - `persistent_carry`
  - `reduction_carrier`
  - `index_buffer`
  - `output`

#### 输入

- 冻结后的 `SemanticProgram`
- target-neutral spatialization policy：
  - task fusion / split policy
  - layout choice policy
  - work partition policy
  - sync construction policy

#### 输出

- 冻结后的 `SpatialProgram`

#### 验证职责

1. task/channel 图闭合
2. producer-consumer、barrier、completion 结构正确
3. work partition 与 layout 一致
4. carry state 穿越 task 边界时的流向正确
5. grouped/ragged/paged route 一致性
6. virtual placement 约束不冲突
7. 明显的 race / deadlock / route inconsistency 可在这一层被提前发现

#### 明确不属于这一层的内容

- `CBIndex`
- `semaphore_id`
- `dst offset`
- `CreateCircularBuffer / SetRuntimeArgs`
- TT physical core identifier

#### Spatialization 是如何从 semantic 层构造出来的

`SpatialProgram` 不是从 raw TIR 直接恢复出来的，而是从冻结后的 `SemanticProgram` 做**空间化投影**。

推荐的构造顺序：

1. **建立 semantic flow graph**
   - 节点：`SemanticRegion`
   - 边：通过 `State / Relation / Phase` 建出的 def-use、carry、fanout、join 关系
2. **找出必须切开的边界**
   - phase boundary
   - carry / cross_phase boundary
   - routed / paged / grouped domain boundary
   - select / combine / dispatch / merge 这类天然 topology boundary
   - 必须单独同步的 producer-consumer boundary
3. **形成候选 task**
   - 将同 phase、同 domain family、同主要执行职责的一组 region 聚合为 `Task`
4. **生成 channel**
   - 所有跨 task 流动的 state 都显式变成 `Channel`
5. **生成 layout / partition**
   - 从 domain 及 access relation 投影出 `Layout` 与 `WorkPartition`
6. **生成 sync**
   - 从 phase edge、carry edge、fanout/fanin、completion requirement 构造 `SyncEdge`
7. **生成 virtual placement**
   - 从 channel topology 和 locality pressure 生成 `Placement`
8. **生成 resource intent**
   - 从 carry / scratch / output / index payload 生成 `ResourceIntent`

因此，`SpatialProgram` 的职责不是“理解算法”，而是把已经冻结的算法语义组织成可执行的空间程序图。

#### Spatial 层的分析事实与 policy 边界

必须由分析确定的事实：

- 哪些 region 之间存在必须的数据依赖
- 哪些 state 必须跨 task 流动
- 哪些边界必须同步
- 哪些 domain/layout/partition 族是语义上被强制的

可以由 spatial policy 决定的内容：

- task 是否进一步 fusion / split
- channel 是否选择更细或更粗粒度
- `row / tile / pair / split_k / expert / page / chunk` 的具体 partition 粒度
- virtual placement 偏好
- resource intent 的 reuse 策略

这条边界必须写死：
**analysis 决定 legality 和 must-have structure；policy 只在合法空间内做选择。**

#### Spatial 层与 semantic/TIR 的交互 contract

`SpatialProgram` 的主要输入是 `SemanticProgram`，不是原始 TIR 子树。

它与上层的显式绑定应该体现在：

- `Task.semantic_regions`
- `Channel.payload_states`
- `Layout.domain_bindings`
- `WorkPartition.domain_bindings`
- `SyncEdge.phase_or_state_bindings`

允许保留的 TIR 级对象只有：

- `PrimExpr`
  - 例如 partition expr、layout expr、route expr
- `GlobalVar`
  - 标识所属 device function

不允许的做法：

- 重新沿 TIR 树匹配 task 边界
- 根据 builtin 名字猜 sync 边
- 根据 buffer 名字猜 layout / partition

#### SpatialProgram 的验证与失效规则

1. `SpatialProgram` 依赖 `SemanticProgram`；semantic 层失效时，spatial 层必须同时失效。
2. `SpatialProgram` 自身一旦生成，后续 target 层不能再修改其 task/channel/layout/sync 结构。
3. 只有当 semantic 结构不变时，才允许在 spatial policy 层重建不同的 `SpatialProgram` 变体。

### 5.3 `TT Target IR`

#### 为什么需要这一层

这一层回答 TT-specific 问题：**这个 spatial program 如何变成合法且稳定的 TT-Metal contract？**

TT 的复杂度不是“最后 codegen 再管一下”：

- kernel role
- circular buffer
- synchronization protocol
- dst/register planning
- core placement
- compile-time/runtime ABI

这些必须先变成 compiler object，再谈 runtime。

#### 设计目标

1. 把 TT-Metal program structure 收成 target contract，而不是散落在 pass side effect 中。
2. 让 `CB / semaphore / dst layout / kernel role / ABI / execution plan` 成为显式对象。
3. 把 `rt_mod_blackhole` 和 runtime 收窄到 materialization。

#### 核心对象

| 对象 | 关键字段 | 含义 |
|------|----------|------|
| `TTProgram` | `kernels`, `core_groups`, `cb_plan`, `semaphore_plan`, `dst_layout_plan`, `abi_plan`, `execution_plan` | TT 合约容器 |
| `TTKernel` | `role`, `task_subset`, `core_group`, `compile_time_args`, `runtime_args` | TT kernel 合约 |
| `TTCoreGroup` | `physical_cores`, `core_type`, `topology_role` | physical execution group |
| `TTCBPlan` | `resource_class`, `capacity`, `producer`, `consumer`, `binding_scope` | 最终 CB 规划 |
| `TTSemaphorePlan` | `kind`, `source_group`, `target_group`, `protocol` | 最终同步规划 |
| `TTDstLayoutPlan` | `state_bindings`, `offset`, `tile_span`, `layout_role` | dst/register residency 规划 |
| `TTABIPlan` | `compile_time_arg_specs`, `runtime_arg_specs`, `accessor_specs`, `launch_specs` | ABI 合约 |
| `TTExecutionPlan` | `work_distribution`, `remote_core_descriptors`, `kernel_order` | 执行与 launch 规划 |

#### 对象设计

**`TTKernel.role`**

- 当前最小稳定角色：
  - `reader`
  - `compute`
  - `writer`
- 未来可扩角色：
  - `relay`
  - `reduction`
  - `dispatcher`

**`TTCBPlan.resource_class`**

- 示例：
  - `transport`
  - `tile_scratch`
  - `vector_scratch`
  - `index_scratch`
  - `persistent_carry`
  - `output`

**`TTSemaphorePlan.kind`**

- `local`
- `remote`
- `multicast`
- `barrier`

**`TTDstLayoutPlan`**

- 绑定的是“长生命周期 compute-local state”到具体 dst/register offset
- 这里不只服务 attention：
  - attention carry
  - chunk recurrence state
  - routed intermediate tile state
  - 其他 compute-local long-lived state

#### 输入

- 冻结后的 `SpatialProgram`
- TT hardware model：
  - topology
  - memory hierarchy
  - NoC / multicast / semaphore capabilities
  - dst/register capacity
  - core kinds 与 placement constraints

#### 输出

- 冻结后的 `TTProgram`
- 可 materialize 的 `ExecutableSpec`

#### 验证职责

1. L1 / CB / dst capacity 合法性
2. semaphore / multicast / routing 合法性
3. core placement 合法性
4. compile-time/runtime ABI 完整性
5. `ExecutableSpec` materialization 所需信息齐全，且无需猜测

#### 明确不属于这一层的内容

- 修改 semantic state / relation / phase
- 修改 spatial task / channel / layout / sync 结构
- runtime 侧补协议或反推语义

#### TT target mapping 是如何从 spatial 层构造出来的

`TTProgram` 不是直接从 TIR 或 runtime schema 拼出来的，而是从 `SpatialProgram + TT hardware model` 做 target-specific mapping。

推荐的构造顺序：

1. **kernel clustering**
   - 把 `Task` 按职责和 TT program model 聚成 `TTKernel`
   - 当前最小稳定切法是 `reader / compute / writer`
2. **core-group mapping**
   - 把 `Placement` 和 `WorkPartition` 落成 `TTCoreGroup`
   - 得到 logical work 到 physical core 的映射
3. **channel/resource -> CB plan**
   - `Channel + ResourceIntent` 落成 `TTCBPlan`
4. **sync -> semaphore plan**
   - `SyncEdge` 落成 `TTSemaphorePlan`
5. **long-lived compute state -> dst plan**
   - carry / reduction carrier / persistent compute-local state 落成 `TTDstLayoutPlan`
6. **program ABI derivation**
   - 从 kernel inputs/outputs/accessors/work distribution 推出 `TTABIPlan`
7. **execution plan derivation**
   - 生成 `kernel_order / remote_core_descriptors / work_distribution`

也就是说，TT 层做的是 **legal target contract synthesis**，不是重新发明空间图。

#### TT 层的分析事实与 policy 边界

必须由 hardware-aware analysis 决定的事实：

- 哪些 mapping 违反 topology / NoC / semaphore / dst / L1 约束
- 哪些 channel 必须 materialize 成 transport buffer
- 哪些 carry 可以 register-resident，哪些必须 round-trip
- 哪些 sync edge 必须落成 semaphore / barrier / multicast protocol

可以由 target policy 决定的内容：

- 具体 kernel clustering 粒度
- CB class / page size / reuse 偏好
- dst layout 的具体 packing 方案
- core placement 在合法区域内的优化选择
- compile-time / runtime arg 的组织细节

同样必须遵守：
**analysis 决定 legality；policy 只在合法空间内选一个实现。**

#### TT 层与 TIR/codegen/runtime 的交互 contract

`TTProgram` 自身不是 codegen 直接消费的 AST。它向下游提供两类结果：

1. **target contract**
   - `ExecutableSpec`
   - `blackhole.segment_plan`
   - `blackhole.runtime_args`
   - `blackhole.cb_configs`
   - `blackhole.core_plan`
2. **executable body 输入**
   - TT-lowered `PrimFunc`
   - 或等价的 builtin emission plan，再统一 materialize 成 `PrimFunc.body`

因此：

- `codegen_blackhole` 继续以 TIR/builtin 为入口
- `rt_mod_blackhole` 继续以 `ExecutableSpec` 为主要入口
- 但二者都不再负责恢复 semantic/spatial 结构

#### TTProgram 的验证与失效规则

1. `TTProgram` 依赖 `SpatialProgram` 与 hardware model；任一失效时必须整体重建。
2. `TTProgram` 一旦通过 `ValidateTTTargetProgram`，下游只能 materialize，不允许再补语义或重构 task 图。
3. materialized `blackhole.*` attrs 只是 `TTProgram` 的投影，不能被 runtime/codegen 回写成新的真源。

### 5.4 IR 承载模型与实现形态

这套分层 IR 不会实现成第二套 textual IR，也不会把 `SemanticProgram / SpatialProgram / TTProgram` 伪装成新的 `PrimFunc`。它们会实现成 **基于 TVM object system 的 companion IR**：

- C++ 侧定义 `ObjectNode / ObjectRef`
- 注册 reflection / printer / structural equal / hash
- 提供 visitor / mutator / verifier
- Python 侧通过 FFI 可见

推荐命名空间：

- `tl.semantic.*`
- `tl.spatial.*`
- `tl.tt.*`

对应的稳定容器：

| 承载位置 | 放什么 | 为什么 |
|----------|--------|--------|
| `PrimFunc.body` | 规范化 TIR 与最终可执行 device body | 继续承接通用计算结构与 codegen 入口 |
| `PrimFunc.attrs["tl.semantic_program"]` | `SemanticProgram` | 每个 device function 的语义真源 |
| `PrimFunc.attrs["tl.spatial_program"]` | `SpatialProgram` | 每个 device function 的空间程序真源 |
| `PrimFunc.attrs["tl.tt_program"]` | `TTProgram` | 每个 device function 的 TT target contract 真源 |
| `IRModule.global_infos` | `hardware_model`、`topology`、全局 mapping policy | 真正跨函数共享的 module-scope 对象 |
| materialized `blackhole.*` attrs | `segment_plan`、`runtime_args`、`cb_configs`、`core_plan` | 仅用于兼容现有 codegen/runtime 消费面 |

这意味着：

1. 不建议把三层 IR 都编码成 `Array<Any> / Map<String, Any>` 风格 attrs。
2. 也不建议把 semantic/spatial/tt 层直接做成 `BaseFunc`；它们不是 executable function。
3. `attrs` 在这里的职责只是“挂载 typed IR object”，不是“编码协议本体”。
4. `IRModule.global_infos` 只放真正 module-scope 的共享对象；每个 kernel 的 companion IR 仍以 `PrimFunc` 为粒度挂载。

### 5.5 `TIR + Companion IR` 混合编译链

这次重设计不是“拿新 IR 替换 TIR”，而是把当前 `TIR-only` 主链改成长期稳定的混合链：

```text
PrimFunc / TIR
  -> Semantic Recovery
  -> attach SemanticProgram
  -> LowerToSpatialProgram
  -> attach SpatialProgram
  -> LowerSpatialProgramToTTTarget
  -> attach TTProgram
  -> MaterializeTTExecutableSpec
  -> TT-lowered PrimFunc + materialized blackhole.* attrs
  -> codegen / rt_mod_blackhole
```

各层职责如下：

1. `TIR`
   - 承接前端通用计算描述
   - 承接 target-lowered executable body
   - 不再单独承担全部高层语义
2. `SemanticProgram / SpatialProgram / TTProgram`
   - 作为 compiler truth object 存在
   - 中段 pass 直接消费这些对象，而不是反复从 raw TIR 猜
3. materialization
   - 只把 `TTProgram` 翻译成现有 runtime/codegen 仍需的 schema
   - `ExecutableSpec` 与 `blackhole.segment_plan/runtime_args/cb_configs/core_plan` 都来自这一层

因此，稳定形态不是 “all-TIR” 也不是 “all-new-IR”，而是：

- 上半程以 TIR 为 carrier
- 中段以 companion IR 为真源
- 下半程再回到 target-lowered TIR 与 runtime schema

### 5.6 Companion IR 与 TIR 的交互 contract

新 IR 要能维护住与 TIR 的关系，前提不是“把信息都塞到 attrs”，而是定义清楚**谁引用谁、谁是单向真源、什么时候失效**。

#### 5.6.1 允许的直接引用

`SemanticProgram` 可以直接持有下列 TIR 原子：

- `Buffer`
- `Var`
- `PrimExpr`
- `GlobalVar`
- `TIRAnchor`

典型绑定：

- `State.backing_buffer -> Buffer`
- `Domain.iter_vars -> Array<Var>`
- `Domain.predicate / bound_expr / index_remapping -> PrimExpr`
- `Relation.mapping_expr -> PrimExpr`
- `SemanticRegion.source_anchor -> TIRAnchor`

`SpatialProgram` 主要消费 `SemanticProgram`，只保留必要的 `PrimExpr` 或 `GlobalVar` 级引用；它不再依赖原始 TIR 子树结构。
`TTProgram` 原则上不再直接依赖 TIR 结构，只消费 `SpatialProgram + hardware model`。

#### 5.6.2 真源规则

1. `LiftToStatefulSemanticIR` 完成之后，TIR 不再是算法语义真源。
2. `LowerToSpatialProgram` 完成之后，task/channel/layout/sync 结构只能从 `SpatialProgram` 读取。
3. `LowerSpatialProgramToTTTarget` 完成之后，TT resource / ABI / execution contract 只能从 `TTProgram` 读取。
4. `codegen_blackhole` 与 `rt_mod_blackhole` 只消费物化后的 target contract，不再回头理解 semantic/spatial 语义。

#### 5.6.3 生命周期与失效规则

1. `LiftToStatefulSemanticIR` 必须运行在固定 canonicalization 点之后。
2. 一旦某个 `PrimFunc` 挂上 `tl.semantic_program`，后续就不能再运行会破坏 semantic anchor 的任意 TIR 改写。
3. 如果后续 pass 必须实质性改动相关 TIR 结构，就必须显式失效并删除：
   - `tl.semantic_program`
   - `tl.spatial_program`
   - `tl.tt_program`
   然后重跑 recovery/lowering。
4. `blackhole.*` materialized attrs 不是上游 IR 的真源；它们可以被整体重建，而不允许被下游 patch 成第二真源。

### 5.7 Anchor 与 Semantic Region 的通用恢复规则

`anchor` 和 `semantic region` 不能靠 workload 特判，也不能靠样例名猜。它们必须来自一套对复杂计算普遍成立的结构恢复规则。

#### 5.7.1 固定恢复入口

semantic recovery 必须在以下条件满足后执行：

- tile op 已经过必要 canonicalization
- 基础 loop / buffer / predicate 结构已经稳定
- 但还没有进入 TT-specific builtin lowering

现有：

- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`

继续保留，但它们后续只作为 signal producers，不再直接定义最终语义。

#### 5.7.2 `AtomicEffect` 作为恢复原子

推荐在 `LiftToStatefulSemanticIR` 之前先建立一个统一的内部恢复表示：`AtomicEffect`。

每个 `AtomicEffect` 代表一个可分析的原子 effect 节点，例如：

- `BufferStore`
- reduction update
- gather/scatter write
- copy / transport effect
- 带 target-independent side effect 的 builtin

每个 `AtomicEffect` 至少分析出：

| 字段 | 含义 |
|------|------|
| `reads` | 读集合 |
| `writes` | 写集合 |
| `domain_signature` | enclosing loops、bounds、predicate、index remap、launch/work vars |
| `effect_kind` | `map / reduce / gather / scatter / select / recurrence / transport` |
| `access_relation` | 读写索引映射 |
| `temporal_role` | `pre_loop / in_loop / post_loop / loop_carried` |
| `scope_class` | `ephemeral / local_state / cross_phase / materialized` |

`TIRAnchor` 绑定的就是这些 `AtomicEffect` 节点，而不是一整棵脆弱的旧 TIR 子树。

#### 5.7.3 `domain_signature` 的通用来源

`domain_signature` 必须从结构事实中恢复，而不是从 op 名推断：

- enclosing loop vars
- loop `min / extent`
- launch axes 与 work decomposition
- predicate
- indirect/index remapping expr
- segment descriptor / grouped-ragged descriptor

这使同一恢复框架可以覆盖：

- dense tiled compute
- routed/grouped/ragged dispatch
- paged/indexed sparse access
- chunk recurrence

#### 5.7.4 `effect_kind` 的通用分类

分类应基于读写与索引关系，而不是特定样例：

- `map`
  - 写目标不回读自身
- `reduce`
  - 写目标以降维/聚合方式回读自身
- `recurrence`
  - 写入值在后续迭代再次被读
- `gather`
  - 读索引来自 index buffer 或 indirect expr
- `scatter`
  - 写索引来自 indirect expr
- `select`
  - 输出表达 selected subset 或 value/index pair
- `transport`
  - 主要语义是位置搬运或 scope crossing

如果某类 effect 无法仅靠结构区分，就要求前端/早期 IR 提供最小显式 hint，而不是由后段猜。

#### 5.7.5 `SemanticRegion` 的切分不变量

`SemanticRegion` 定义为：**在同一 phase 内，满足同一 domain family 与主要语义职责的一组最大连通 effect 节点。**

必须切 region 的情况：

1. `domain_signature` 改变
2. 主要 `effect_kind` 改变
3. 出现 loop-carried 或 cross-phase 边
4. 中间值不再是 region-internal ephemeral
5. 需要独立 spatialization 决策
6. 遇到 target-visible synchronization / transport boundary

可以合并在同一个 region 的前提：

1. 同一 domain family
2. 同一 phase
3. 同一主要 effect kind
4. 中间边都是 local ephemeral
5. 对外接口不变
6. 不需要单独 task/channel/sync 决策

这套切分规则是 workload-neutral 的，不依赖 attention、MoE、paged decode 或 recurrence 的样例名。

## 6. 层间接口与不变量

### 6.1 真源规则

1. 算法语义只存在于 `Stateful Semantic IR`
2. 空间组织只存在于 `Spatial Program IR`
3. TT 资源与 ABI 只存在于 `TT Target IR`
4. `ExecutableSpec` 由 `TT Target IR` 物化，不是第二真源

### 6.2 交接契约

| From | To | 必须交付的契约 | 允许做的决策 | 明确禁止 |
|------|----|----------------|--------------|----------|
| `Semantic Recovery` | `Stateful Semantic IR` | 恢复出的 domain/state/relation/phase 事实 | 对象化与冻结 | 泄漏 TT resource 事实 |
| `Stateful Semantic IR` | `Spatial Program IR` | 冻结后的算法语义 | 构造 task/channel/layout/sync/work | 改变语义含义 |
| `Spatial Program IR` | `TT Target IR` | 冻结后的空间结构 | TT mapping、resource planning、ABI 定义 | 发明新的 task graph 或 semantic combine |
| `TT Target IR` | `ExecutableSpec / runtime` | 冻结后的 TT contract | API materialization 与 launch emission | semantic recovery 或 protocol patching |

### 6.3 验证层

- `ValidateStatefulSemanticIR`
- `ValidateSpatialProgram`
- `ValidateTTTargetProgram`

错误必须在最早拥有该不变量的层暴露出来。

### 6.4 禁止反向推断

下面这些行为是明确禁止的：

1. 用 `CB / dst layout / runtime args` 反推 state semantics
2. 用 TT kernel 名字反推 task graph 结构
3. 让 runtime 补丢失的 synchronization 或 carry strategy
4. 因为 backend 需要 `task / channel / semaphore`，就把它们直接暴露成 Python DSL 表面概念

## 7. 代表性工作负载示例

本节的作用不是定义协议，而是证明三层 IR 不是围着单一 consumer 设计的。

### 7.1 状态化 reduction-update：attention / online softmax 类

代表：

- `examples/flash_attention/`
- `examples/online_softmax/`
- `examples/attention_sink/`

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `q_tile_domain`
  - `kv_chunk_domain`
  - 可选 `bound_expr`（causal）
  - 可选 `index_remapping`（grouped head）
- `State`
  - `acc_o : matrix_state(carry)`
  - `scores_max : vector_state(carry)`
  - `scores_sum : vector_state(carry)`
  - `logsum : scalar_state(cross_phase)` 或 epilogue-derived result
- `Relation`
  - `scores_max reduced_from qk_scores`
  - `scores_sum reduced_from exp_scores`
  - `acc_o / scores_max / scores_sum carried_across kv_chunk_phase`
- `Phase`
  - `kv_chunk_phase`
  - `epilogue_phase`

**`Spatial Program IR` 应表达**

- `Task`
  - `load_q_task`
  - `stream_k_task`
  - `stream_v_task`
  - `attention_update_task`
  - `store_out_task`
- `Channel`
  - `q_tiles`
  - `k_tiles`
  - `v_tiles`
  - `carry_state`
  - `out_tiles`
- `Layout`
  - `row_layout`
  - `kv_chunk_layout`
- `WorkPartition`
  - `row_partition`
  - 可选 `split_partition`

**`TT Target IR` 应表达**

- `reader_qkv`
- `compute_update`
- `writer_out`
- transport CB、persistent carry plan、`TTDstLayoutPlan`、split merge execution plan

### 7.2 Selection / indexing：topk 类

代表：

- `examples/topk/example_topk.py`
- `examples/deepseek_v32/topk_selector.py`

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `token_domain`
  - `candidate_domain`
- `State`
  - `score_state : vector_state`
  - `selected_index : index_state`
  - `selected_value : vector_state`
- `Relation`
  - `selected_index selected_from logits_domain`
  - `selected_value applies_to selected_index`
- `SemanticRegion`
  - `reduce_max`
  - `select`
  - `emit_index`

**`Spatial Program IR` 应表达**

- `Task`
  - `load_logits`
  - `select_topk`
  - `write_indices`
- `Channel`
  - `logits`
  - `selected_index`
  - `selected_value`
- `WorkPartition`
  - `row_partition`
  - `selected_subset`

**`TT Target IR` 应表达**

- index scratch / vector scratch
- selector/reduction kernel role（当前可先映射进 `compute`）
- index/runtime ABI

### 7.3 Routed / grouped / ragged dispatch：MoE 类

代表：

- `examples/fusedmoe/example_fusedmoe_tilelang.py`

该类 workload 的关键不是“多了一个 matmul”，而是：

- `group_sizes / group_offsets / group_padded_offsets`
- `group_idx_for_bx`
- ragged token block
- expert-specific dispatch 与 combine

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `routed_token_domain`
  - `expert_domain`
  - `segment_descriptor = {group_sizes, group_offsets, group_padded_offsets}`
- `State`
  - `expert_index : index_state`
  - `expert_weight : vector_state`
  - `up_logits : matrix_state(ephemeral)`
  - `output_state : matrix_state`
- `Relation`
  - `routed_token_domain partitioned_by expert_index`
  - `output_state applies_to expert_weight`
  - `up_logits gathered_from routed_token_domain`
- `Phase`
  - `gate_up_phase`
  - `down_phase`
  - 可选 `combine_phase`

**`Spatial Program IR` 应表达**

- `Task`
  - `route_tokens`
  - `expert_gate_up`
  - `expert_down`
  - `combine_output`
- `Channel`
  - `token_shard`
  - `expert_index`
  - `expert_weight`
  - `intermediate_tile`
- `Layout`
  - `grouped_layout`
  - `ragged_layout`
  - `expert_layout`
- `WorkPartition`
  - `expert_partition`
  - `ragged_token_partition`

**`TT Target IR` 应表达**

- routed/index buffer
- token tile transport buffer
- expert-kernel ABI
- grouped core placement
- combine writer plan

### 7.4 Paged / indexed sparse decode

代表：

- `examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py`
- `examples/deepseek_mla/example_mla_decode_paged.py`
- `examples/flash_decoding/example_gqa_decode.py`

这一类 workload 的关键是：

- `block_indices`
- `block_table`
- `cache_seqlens`
- logical block -> physical page 的两级映射
- split decode 与 combine

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `query_domain`
  - `selected_block_domain`
  - `paged_kv_domain`
  - `index_remapping` 表达 logical block -> physical page mapping 语义
- `State`
  - `block_index : index_state`
  - `page_index : index_state`
  - `cache_bound : scalar_state`
  - `acc_o / logsum : carry state`
- `Relation`
  - `block_index indexes paged_kv_domain`
  - `acc_o carried_across split_phase`
  - `logsum reduced_from score_tiles`

**`Spatial Program IR` 应表达**

- `Task`
  - `stream_query`
  - `stream_selected_page`
  - `decode_update`
  - `merge_split`
- `Channel`
  - `query_tile`
  - `page_tile`
  - `page_index`
  - `partial_out`
- `Layout`
  - `paged_layout`
  - `selected_block_layout`
- `WorkPartition`
  - `page_partition`
  - `split_partition`

**`TT Target IR` 应表达**

- page/index runtime descriptor
- paged transport buffer
- split merge execution plan
- optional multicore semaphore / completion plan

### 7.5 Chunked recurrence / scan

代表：

- `examples/linear_attention/example_mamba_chunk_state.py`
- `examples/kda/`
- `examples/gdn/`

这一类 workload 的关键不是 attention，而是：

- `chunk domain`
- `cross-phase state`
- recurrence update / decay
- chunk-local compute 与 chunk-writeback

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `chunk_domain`
  - `intra_chunk_step_domain`
- `State`
  - `chunk_state : matrix_state(cross_phase)`
  - `decay_state : vector_state(ephemeral)`
  - `dt_state : vector_state`
- `Relation`
  - `chunk_state carried_across chunk_phase`
  - `decay_state applies_to chunk_state`
  - `chunk_state reduced_from step_inputs`
- `Phase`
  - `chunk_body_phase`
  - `chunk_writeback_phase`

**`Spatial Program IR` 应表达**

- `Task`
  - `load_chunk`
  - `recurrence_step`
  - `write_chunk_state`
- `Channel`
  - `chunk_input`
  - `carry_state`
  - `chunk_output`
- `Layout`
  - `chunk_layout`
- `WorkPartition`
  - `chunk_partition`

**`TT Target IR` 应表达**

- persistent carry realization
- dst/register or CB-round-trip choice
- chunk execution order
- runtime chunk descriptor ABI

## 8. 代码映射与迁移方案

### 8.1 当前组件映射到新架构

| 当前 Pass / 模块 | 新归属 | 长期状态 |
|------------------|--------|----------|
| `AnalyzeBlackholeWorkDecomposition` | semantic recovery 输入生产者 | 保留并泛化 |
| `AnalyzeBlackholeFragmentRegions` | semantic recovery 输入生产者 | 保留并泛化 |
| `AnalyzeBlackholePipelineStages` | semantic recovery 输入生产者 | 保留并收紧职责 |
| `AnalyzeSemanticStructure` | semantic recovery 聚合器；构造 `AtomicEffect` / dependence / anchor | 新增 |
| `LowerBlackholeOps` | 当前混合 legacy layer；后续拆成 spatial lowering + TT target lowering | 收缩，最终不再作为 monolithic 黑洞 |
| `PlanBlackholeCB` | TT target planner 子模块 | 保留但降级 |
| `AssignBlackholeCores` | TT target planner 子模块 | 保留但收窄 |
| `rt_mod_blackhole` | codegen/runtime materialization | 保留并收紧 |
| `ExecutableSpec` | TT target materialization 结果 | 保留 |

### 8.2 目标 Pass 链

```text
SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> AnalyzeSemanticStructure
  -> LiftToStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> LowerToSpatialProgram
  -> ValidateSpatialProgram
  -> LowerSpatialProgramToTTTarget
  -> ValidateTTTargetProgram
  -> MaterializeTTExecutableSpec
  -> rt_mod_blackhole
```

### 8.3 迁移阶段

**Phase A: Semantic IR**

1. 加入 typed companion object：
   - `Domain / State / Relation / Phase / SemanticRegion`
   - `TIRAnchor / TIRValueBinding`
2. 明确 `PrimFunc.attrs["tl.semantic_program"]` 挂载 contract
3. 落 `AnalyzeSemanticStructure`
   - 构造 `AtomicEffect`
   - 构造 dependence / carry / domain signature
   - 生成 anchor 与 semantic region partition
4. 落 `LiftToStatefulSemanticIR`
5. 落 `ValidateStatefulSemanticIR`
6. 要求现有 target lowering 从 semantic 真源消费信息，而不是继续自己猜主要语义

**Phase B: Spatial Program IR**

1. 加入 typed companion object：
   - `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
2. 明确 `PrimFunc.attrs["tl.spatial_program"]` 挂载 contract
3. 落 `LowerToSpatialProgram`
4. 落 `ValidateSpatialProgram`
5. 把 task/channel/layout/sync/work-partition 逻辑从 `LowerBlackholeOps` 中拆出去

**Phase C: TT Target IR**

1. 加入 typed companion object：
   - `TTKernel / TTCoreGroup / TTCBPlan / TTSemaphorePlan / TTDstLayoutPlan / TTABIPlan / TTExecutionPlan`
2. 明确 `PrimFunc.attrs["tl.tt_program"]` 与 `IRModule.global_infos` 中 `hardware_model/topology` 的承载 contract
3. 落 `LowerSpatialProgramToTTTarget`
4. 落 `ValidateTTTargetProgram`
5. 从 `TT Target IR` 物化：
   - `ExecutableSpec`
   - `blackhole.segment_plan`
   - `blackhole.runtime_args`
   - `blackhole.cb_configs`
   - `blackhole.core_plan`
   - TT-lowered executable `PrimFunc`
6. 把 `PlanBlackholeCB`、`AssignBlackholeCores`、`rt_mod_blackhole` 收回到它们应有的 target/runtime 职责

## 9. TT 目标层约束

TT-specific 事实必须明确地落在 target 层，而不是污染 semantic/spatial 层。

### 9.1 TT Program Model 是 program-level structure

TT-Metal 程序天然围绕以下对象建立：

- `reader / compute / writer`
- host 侧 `CreateCircularBuffer / CreateKernel / SetRuntimeArgs`
- per-core runtime arguments

这些事实必须留在 `TT Target IR`，不能上窜到 semantic 或 spatial 层。

### 9.2 Dst / Register Layout 是一等 target decision

长生命周期 compute-local state 可能驻留在 dst/register 空间，例如：

- attention carry
- chunk recurrence state
- routed intermediate tile state

因此：

- `TTDstLayoutPlan` 是必需对象
- carry strategy 可以是 `register-resident` 或 `CB-round-trip`
- 这是一层 target mapping 决策，不是 semantic 真相

### 9.3 CB 与 Semaphore 是 target 资源

CB 和 semaphore planning 不能再被当成 generic buffer lowering 的副产品。

它们是以下结构的显式实现：

- transport channel
- scratch resource
- persistent carry
- synchronization protocol

### 9.4 当前 TT ground truth 参考

当前已落地的 Blackhole/TT 目标实现面，主要还是先从 SDPA/attention 参考中对齐，因此目前继续交叉核对这些文件：

- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`

这只说明“当前最先落地的 target reference surface 在 attention 类程序上”，不意味着上层 IR 只能为 attention 设计。随着 routed/page/chunk workload 的支持推进，target 参考也应扩到对应 TT ground truth。

## 10. 推进顺序与验收标准

### 10.1 推进顺序

1. 先完成总体架构重写，结束 mixed old/new 文档结构
2. 在当前 direct runtime 基线上执行 Phase A
3. 再执行 Phase B，把 task/channel/layout/sync 从 monolithic lowering 中拆出去
4. 最后执行 Phase C，让 TT 资源与 ABI planning 成为一等 target contract

### 10.2 验收标准

只有当下面条件同时成立，才能认为这次重设计真正落地：

1. semantic truth 在 TT target planning 之前冻结
2. spatial structure 在 TT resource / ABI planning 之前冻结
3. runtime/codegen 不再反推缺失语义
4. `PrimFunc + typed companion IR + module global infos + materialized attrs` 的承载边界固定，并在实现中被遵守
5. `anchor / semantic region` 恢复规则以通用结构分析为基础，而不是按样例名或 op 名做 case matcher
6. copy / GEMM compile-path regression gate 保持绿色
7. 设计与实现计划明确覆盖以下 workload family，而不是只覆盖单一 consumer：
   - selection / indexing
   - routed / grouped / ragged dispatch
   - paged / indexed sparse access
   - stateful reduction-update
   - chunked recurrence / scan
8. `flash-attn` 可以是第一批 consumer，但不能继续被当作总架构边界

## 11. 历史文档

下面这些文档只作为历史记录或实现历史参考，不再作为当前实现依据：

- `tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
- `tasks/dev_design/archive/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`

新实现必须以本文档为准。

## 12. 参考论文

下面这些论文影响了本文档中的分层、validation 和 target-mapping 方向。它们是设计输入，不是 TileLang Blackhole 协议的直接真源。

- `Dato: A Task-Based Programming Model for Dataflow Accelerators` (2025)
  https://arxiv.org/abs/2509.06794
  - 主要借鉴：`task / channel / layout` 一等表示，`virtual -> physical` mapping 分层

- `TL: Automatic End-to-End Compiler of Tile-Based Languages for Spatial Dataflow Architectures` (2025)
  https://arxiv.org/abs/2512.22168
  - 主要借鉴：显式 hardware representation 与 compiler-owned spatial mapping

- `SPADA: A Spatial Dataflow Architecture Programming Language` (2025)
  https://arxiv.org/abs/2511.09447
  - 主要借鉴：rigorous dataflow semantics 以及 routing / synchronization validation

- `Revet: A Language and Compiler for Dataflow Threads` (2023/2024)
  https://arxiv.org/abs/2302.06124
  - 主要借鉴：把高层 threaded/dataflow semantics 和 backend realization 分离

- `Programmatic Control of a Compiler for Generating High-performance Spatial Hardware` (`T2S`, 2017)
  https://arxiv.org/abs/1711.07606
  - 主要借鉴：算法语义和空间映射分层

- `Spatial: A Language and Compiler for Application Accelerators` (PLDI 2018)
  https://pldi18.sigplan.org/event/pldi-2018-papers-spatial-a-language-and-compiler-for-application-accelerators
  - 主要借鉴：面向加速器的语言/编译器分层先例
