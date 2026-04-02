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
  - `compound_state`
- `index_state` 专门承接：
  - topk indices
  - block/page indices
  - group/expert ids
  - selected subset descriptors
- `compound_state` 专门承接需要多步复合更新的长生命周期状态：
  - attention carry（rescale + matmul 累加）
  - chunk recurrence state（outer-product + decay 累加）
  - routed intermediate tile state（dispatch + combine）
  - 当同一个 `matrix_state(carry)` 无法区分不同 update 律时，用 `compound_state` + `update_semantics` 显式区分
- `lifetime`:
  - `ephemeral`
  - `carry`
  - `cross_phase`
- `update_semantics`（可选，仅对 `carry` / `cross_phase` 有意义）：
  - 描述该 state 在跨迭代/跨 phase 时的更新律类别
  - 示例：`rescale_accumulate`（attention carry）、`decay_outer_product`（mamba chunk state）、`weighted_combine`（MoE combine）、`iterative_mask_select`（topk）
  - 这不是 target 实现策略，而是算法语义事实
  - 当 `CombineSpec` 挂载到对应 `Relation` 上时，`update_semantics` 与 `CombineSpec.kind` 必须一致
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

当以下语义无法仅靠结构稳定恢复时，必须允许前端或早期 IR 提供最小显式语义补充：

- 某个 state 的语义身份
- 某个 remap / indirect access 的高层语义身份
- 某个跨 phase 依赖是否构成正式 semantic boundary
- 某段 update / selection 骨架的真实语义类别

规则是：

1. 能从 IR 稳定恢复的，必须分析得到。
2. 不能稳定恢复的，不允许后段猜，必须由更早层显式补齐。

#### Recovery Boundary：统一语义系统下的自动恢复与 DSL 补语义

Recovery boundary 不按 workload family 定义，而按统一语义系统定义。
`flash-attn`、`topk`、`fusedmoe`、`paged decode`、`chunk recurrence`
只是 `Domain / State / Relation / Phase / SemanticRegion` 这组对象的不同组合，
不是各自独立的语义入口。

统一语义系统下，后端必须自动恢复的内容包括：

- `Domain`
  - loop/domain/bounds/predicate 的结构骨架
  - dense / ragged / remapped / paged 等逻辑域骨架
- `State`
  - def-use、buffer scope、lifetime、loop-carried use 所暴露的 state 骨架
- `Relation`
  - read / write / reduce / select / remap / carry 的结构关系
- `Phase`
  - loop-carried edge、pre/post-loop use、显式 pipeline/stage signal
- `SemanticRegion`
  - map / contract / reduce / select / update / scatter-gather 等职责切分骨架

这些都属于“程序在算什么”的结构事实。只要现有 IR 原理上能表达，就必须由
semantic recovery 自动完成，不能因为当前实现不足就提前要求 DSL 补充。

只有当编译必需的**语义身份**无法从 IR 唯一决定时，才允许前端 DSL 显式提供。
这里所谓“语义身份”，不是在描述结构本身，而是在裁决结构之外的高层语义归属。
典型场景包括：

- 某个 state 是正式 `carry`，还是 lowering 残留下来的普通 temporary
- 某个 indirect access 是普通 gather/scatter，还是逻辑分页/路由 remap
- 某段 serial-select / iterative-update 骨架是否应提升为正式 semantic object
- 某个跨 phase 依赖是否构成稳定 semantic boundary，而不是局部实现细节

这条边界的含义是：

1. 用户不负责告诉编译器“这里有 reduce / select / remap / carry”这类结构骨架。
2. 用户只在 IR 无法唯一裁决对象身份时，补充最小 semantic fact。
3. DSL 补充的是 `Domain / State / Relation / Phase / SemanticRegion` 的身份信息，
   不是 workload family 名字，也不是 target 细节。

#### DSL 补语义机制

本设计不引入开放式 hint language，也不引入长对象式 annotation。
前端 DSL 只在统一语义系统存在真实表达缺口时，提供最小显式语义补充。

设计原则：

1. DSL 补充的是**语义真相**，不表达 spatial/target 结构
2. DSL 补充默认应当为零；copy / GEMM 以及可稳定恢复的复杂 kernel 不应要求用户标注
3. DSL 补充必须很短、局部、固定形态，不能演化成第二套描述语言
4. DSL 补充在 lowering 前进入 `PrimFunc.attrs["tl.semantic_hints"]`
5. 当结构恢复与 DSL 显式补充冲突时，以 DSL 显式语义为准

第一版不设计通用、开放式的 `hint(key=value, ...)` 接口。更稳的策略是：

- 先把统一语义系统与 recovery 主链落起来
- 先验证 `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
  的主要结构骨架是否都能自动恢复
- 只有在某一类“身份裁决”被证明无法从 IR 稳定恢复时，
  才把那一个语义事实提升为 DSL/IR 的正式表达能力

因此，Phase A 的目标不是“先设计一套丰富的 annotation API”，而是：

1. 先定义统一语义系统
2. 先把自动 recovery 的覆盖面做到最大
3. 再把真正不可恢复的少数语义身份，收成最小 DSL 补语义点

一句话概括本边界：

> 能从 IR 分析出来的，一律后端自动分析。  
> IR 无法表达但编译又必需的语义，一律由前端 DSL 显式提供。

#### 高层 semantic descriptor 族

为了避免 `combine_function`、`selection policy`、`segment/page descriptor` 再次退化成零散 attrs，本设计把它们收成 typed descriptor family。它们属于 semantic 层，因为它们描述的是**算法语义约束**，不是 spatial 或 TT 实现细节。

| Descriptor | 挂载位置 | 关键字段 | 典型 workload |
|------------|----------|----------|---------------|
| `CombineSpec` | `Relation` / `SemanticRegion` | `kind`, `init_expr`, `update_expr`, `finalize_expr`, `algebraic_traits` | online softmax、Welford、weighted combine、chunk update |
| `SelectionSpec` | `SemanticRegion` / `State` | `score_expr`, `rank_kind`, `threshold_policy`, `tie_break_policy`, `output_form` | topk、selector、argmax-like family |
| `SegmentSpec` | `Domain` | `segment_ids`, `segment_sizes`, `segment_offsets`, `logical_to_segment_expr`, `padding_policy` | MoE/grouped/ragged dispatch |
| `PageSpec` | `Domain` / `State` | `page_table`, `page_size`, `logical_to_physical_expr`, `validity_expr`, `sequence_length_expr` | paged decode、block/page sparse access |
| `RecurrenceSpec` | `Relation` / `Phase` | `carry_state`, `step_state`, `decay_expr`, `update_order`, `boundary_policy` | chunk recurrence、scan、state-space family |

约束：

1. 这些 descriptor 是 semantic truth，不得被降成字符串名或匿名 `Map<String, Any>`。
2. `CombineSpec` 不等于 target kernel 模板名；它只表达算法更新律。
3. `SegmentSpec / PageSpec` 表达逻辑分段与逻辑页语义，不等于 TT runtime descriptor。
4. 一个 workload 可以同时携带多个 descriptor；例如 paged topk 或 routed weighted combine。

这些 descriptor 的来源规则如下：

- 默认由结构恢复得到：
  - `CombineSpec` 的基础 reduce/update 骨架
  - `SelectionSpec` 的基础 rank/threshold 结构
  - `SegmentSpec` 的 offsets/size/index remap 骨架
  - `PageSpec` 的 page-table/index-remap/validity 骨架
- 当结构不足以稳定表达高层含义时，允许前端/早期 IR 显式补齐：
  - `SelectionSpec.tie_break_policy`
  - `CombineSpec.algebraic_traits`
  - `SegmentSpec.padding_policy`
  - `PageSpec` 的外部协议含义
  - `RecurrenceSpec.update_order`

#### 统一语义系统的最小闭合集

Phase A 不能从第一天起把全部 descriptor、policy、annotation protocol 一次性做满。
更稳的做法是先冻结一个最小 semantic core：

- `Domain`
- `State`
- `Relation`
- `Phase`
- `SemanticRegion`

这五类对象是一等语义对象；它们共同回答“程序在算什么”。
其余内容都应作为：

- 这些对象的字段
- 这些对象上的 typed refinement
- 或后续阶段的 policy / validation 输入

而不是再并列长出一层平行核心对象。

因此：

1. `CombineSpec / SelectionSpec / SegmentSpec / PageSpec / RecurrenceSpec`
   是 typed semantic refinement，不是另一套 parallel core
2. Phase A1 先落最小 core object model
3. Phase A2 再把 descriptor 家族和更宽 workload support 补齐

#### Semantic -> Spatial 的投影规则

`Stateful Semantic IR` 和 `Spatial Program IR` 之间不能只停留在“概念上分层”，
还需要有明确的投影规则：

- `SemanticRegion`
  - 映射为一个或多个 `Task`
- `Relation`
  - 决定 `Channel` / `SyncEdge` 的存在与类型
- `Phase`
  - 决定 `ProgramPhase`
- `Domain`
  - 约束 `Layout` 与 `WorkPartition` 的合法候选空间
- `State`
  - 决定该值是 local-only、communicated、persistent，还是需要跨 task / phase 保活

跨层禁止项同样必须明确：

1. semantic 层不泄漏 TT resource 事实
2. spatial 层不重新解释 semantic object 的身份
3. target 层不重新发明 task graph 或 semantic combine

这条规则的意义是：Phase B 只能消费冻结后的 `SemanticProgram`，
不能继续从 raw TIR 或晚期 builtin 序列里“再恢复一次语义”。

#### 最小 TT Target Contract

`TTProgram` 也需要一个最小闭合集，避免 target 层无限膨胀。
第一版至少冻结下面这些对象：

- `TTKernel`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`

它们分别回答：

- kernel role / processor / source binding
- CB identity / page / capacity / format
- semaphore identity / visibility / binding
- dst / pack / unpack / data-format / accumulator layout
- compile-time / common-runtime / per-work ABI
- launch / core mapping / execution order

同时，`TTHardwareModel` 第一版至少要约束：

- processor kinds
- CB / L1 resource bounds
- semaphore capabilities
- dst / pack / unpack legality
- matmul + reduction reinit 约束
- data-format / layout reconfiguration 规则

这一步的目标不是一次性覆盖全部 TT-Metal 细节，
而是先把 target contract 的最小真源收进 `TTProgram`。

#### `ExecutableSpec` 的物化边界

`ExecutableSpec` 不能继续充当第二真源；它必须被收回为 `TTProgram` 的物化结果。

因此未来 `ExecutableSpec` 只应该保存：

- runtime buffer materialization
- kernel creation config
- runtime/common-runtime/per-work arg layout
- direct runtime execution metadata

而不应该再新增：

- semantic phase 解释
- semantic relation
- ad-hoc planning fallback
- late semantic recovery hint

这条边界的设计目的，是防止 Phase C 完成后又回到
“语义、空间结构、target contract 都在 `ExecutableSpec` 里重新长一遍”的旧架构。

#### Workload Validation Matrix

总体设计声明覆盖：

- `flash-attn`
- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

这组 workload family 不能只作为叙述例子，而必须转成正式验证矩阵。
每一类都要明确：

1. 验证哪几个 semantic object
2. 哪些语义应自动恢复
3. 哪些一旦恢复不了，就必须提升为 DSL/IR 正式语义
4. 通过标准是什么

当前最小验证职责建议为：

- `flash-attn`
  - 验证 `carry state / reduce relation / phase boundary`
- `topk`
  - 验证 `select relation / index state`
- `fusedmoe`
  - 验证 `remapped domain / grouped state / phase split`
- `paged decode`
  - 验证 `paged/remapped domain / indirect relation`
- `chunk recurrence`
  - 验证 `carry state / update relation / chunk phase`

这样统一语义系统才能从“设计目标”变成“可验证 contract”。

#### 设计收敛顺序

为了避免再次滑回 monolithic lowering，本设计的收敛顺序应固定为：

1. 先冻结最小 semantic core
2. 先把 recovery boundary 立清
3. 再定义 semantic -> spatial 投影规则
4. 再冻结最小 `TTProgram` contract
5. 最后把 `ExecutableSpec` 收回到 materialization boundary

这条顺序本身就是架构约束：

- 不先做 semantic core，就不要设计更宽 DSL 补语义机制
- 不先做 semantic -> spatial mapping，就不要让 target 层继续发明 task/channel
- 不先做 `TTProgram`，就不要继续把 richer target contract 塞进 `ExecutableSpec`

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
| `SpatialProgram` | `program_phases`, `tasks`, `channels`, `layouts`, `work_partitions`, `placements`, `sync_edges`, `resource_intents` | spatial 程序容器 |
| `ProgramPhase` | `phase_id`, `tasks`, `channels`, `sync_edges`, `global_sync_before`, `global_sync_after`, `shared_buffers` | multi-kernel composition 的全局阶段边界 |
| `Task` | `kind`, `semantic_regions`, `input_ports`, `output_ports`, `execution_scope`, `program_phase` | 逻辑执行单元 |
| `Channel` | `producer`, `consumer`, `payload_kind`, `transport_semantics`, `ordering` | task 间数据流边 |
| `Layout` | `layout_kind`, `partition_axes`, `mapping_expr` | 分布式数据组织 |
| `WorkPartition` | `partition_kind`, `partition_expr`, `load_balance_policy` | 逻辑工作划分 |
| `Placement` | `placement_kind`, `constraints` | virtual placement 关系 |
| `SyncEdge` | `kind`, `scope`, `source_task`, `target_task` | 同步要求 |
| `ResourceIntent` | `kind`, `capacity_hint`, `visibility`, `reuse_policy` | 尚未 target bind 的资源需求 |

#### 对象设计

**`ProgramPhase`**

- 表达 multi-kernel composition 中的**全局阶段边界**
- 典型场景：
  - fusedmoe 的两个 `T.Kernel` block（gate+up phase 和 down phase），通过全局 buffer `up_logits` 通信
  - flash_decoding 的 split phase 和 combine phase，通过 `Output_partial` 通信
  - 任何包含多个 `T.Kernel` 的 program
- `global_sync_before / global_sync_after` 表达阶段间的全局同步要求（如 device-level barrier）
- `shared_buffers` 标识跨阶段共享的全局 buffer（如 `up_logits`、`Output_partial`）
- 当 program 只有一个 `T.Kernel` block 时，`program_phases` 退化为单元素
- `ProgramPhase` 不等于算法 `Phase`：
  - 算法 `Phase` 是 semantic IR 中的语义阶段（如 kv_chunk_phase / epilogue_phase）
  - `ProgramPhase` 是 spatial 层的执行组织单元

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
- `program_phase` 指向所属的 `ProgramPhase`

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

#### 空间化候选、策略与代价模型

`SpatialProgram` 是最终冻结的空间程序真源，但在冻结之前，compiler 必须能表达“存在多个合法 spatialization 候选”的事实。否则，`LowerToSpatialProgram` 还是会退化成一个带硬编码启发式的黑盒 pass。

因此，在 spatial 层引入下面三类 planning object：

| 对象 | 作用 | 关键字段 |
|------|------|----------|
| `SpatialCandidate` | 一个合法的空间化候选 | `tasks`, `channels`, `layouts`, `work_partitions`, `placements`, `sync_edges`, `resource_intents`, `decision_trace` |
| `SpatialPolicy` | 在合法空间内做选择的 policy 输入 | `fusion_policy`, `partition_preferences`, `layout_preferences`, `placement_preferences`, `sync_preferences`, `reuse_preferences` |
| `SpatialCostModel` | 对候选进行排序或裁剪 | `transfer_cost`, `fanout_cost`, `sync_cost`, `imbalance_cost`, `locality_score`, `persistent_state_pressure` |

规则：

1. `Analyze / SemanticProgram` 只产出 legality constraints 和 must-have structure。
2. `LowerToSpatialProgram` 可以先生成一个或多个 `SpatialCandidate`。
3. `SelectSpatialCandidate` 依据 `SpatialPolicy + SpatialCostModel` 选出唯一 winner，并冻结成 `SpatialProgram`。
4. 一旦 `SpatialProgram` 冻结，下游 TT 层不得再改变 task/channel/layout/sync 结构。

对真实 workload 的意义：

- MoE 可以在 `expert` 粒度和 `tile` 粒度 route/compute/combine 之间形成多个合法候选。
- paged decode 可以在 `page` 粒度和 `split` 粒度之间选择 merge 组织方式。
- causal attention 可在 `row` 和 `pair` work partition 候选之间比较。
- recurrence family 可在 `carry-local` 与 `carry-channelized` 候选之间比较。

这三类对象中，只有最终选中的 `SpatialProgram` 是长生命周期真源；候选集合与代价明细可以作为 debug artifact 挂接或打印，但不是 runtime/codegen 协议的一部分。

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
| `TTCBPlan` | `resource_class`, `capacity`, `producer`, `consumer`, `binding_scope`, `data_format`, `pack_mode`, `unpack_mode`, `l1_acc_mode` | 最终 CB 规划 |
| `TTSemaphorePlan` | `kind`, `source_group`, `target_group`, `protocol` | 最终同步规划 |
| `TTDstLayoutPlan` | `state_bindings`, `offset`, `tile_span`, `layout_role`, `fpu_sfpu_ordering`, `pack_format`, `accumulator_mode` | dst/register residency 规划 |
| `TTABIPlan` | `compile_time_arg_specs`, `runtime_arg_specs`, `accessor_specs`, `launch_specs` | ABI 合约 |
| `TTExecutionPlan` | `work_distribution`, `remote_core_descriptors`, `kernel_order` | 执行与 launch 规划 |

#### 对象设计

**`TTKernel.role`**

`role` 设计为**可组合的 role set**，而不是单一固定 enum。原因：TT-Metal MoE kernel 的实际复杂度远超 reader/compute/writer 三角色 — 一个 unified kernel 内通过 `constexpr` role flags 区分 30+ 种行为（`is_sender_core`、`is_gate_mm_core`、`is_gated_reduce_core` 等）。

- 基础角色原子（可组合）：
  - `reader`
  - `compute`
  - `writer`
  - `relay`
  - `reduction`
  - `dispatcher`
  - `sender`（multicast 发送方）
  - `receiver`（multicast 接收方）
- 组合规则：
  - 一个 `TTKernel` 可以持有 `role_set : Set<Role>`，例如 `{reader, sender}` 或 `{compute, reduction}`
  - 当前 GEMM/copy 的最小稳定切法仍是 `{reader}` / `{compute}` / `{writer}` 各一个 kernel
  - MoE 等复杂 workload 可以用 unified kernel + `role_set = {reader, compute, writer, sender}` 和 `constexpr` role flags 区分行为
  - 每个 role 对应的 core type mapping（BRISC/NCRISC/TRISC）由 `TTCoreGroup.core_type` 决定，不由 role 隐含
- `role_flags`（可选）：
  - 对于 unified kernel pattern，可携带 `role_flags : Map<String, Bool>` 表达更细粒度的 constexpr role 条件
  - 例如：`{"is_gate_mm_core": true, "is_mcast_grid_core": true}`
  - 这些 flags 直接物化为 TT-Metal kernel 的 named compile-time args

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
- `compute_internal`
  - 表达 compute kernel 内部 FPU/SFPU 之间的 micro-architecture 级同步
  - 典型场景：SDPA 中 FPU matmul 完成后 SFPU reduce_max 读取 dst，需要 `t6_semaphore_wait_on_max` / `t6_semaphore_post` 协调
  - 这不同于 task 级的 local/remote semaphore；它发生在同一个 kernel 内部的两个执行单元之间
  - `protocol` 字段对 `compute_internal` kind 应表达具体的 wait/post 模式（`wait_on_max`、`wait_on_zero`、`post`）

**`TTCBPlan` 新增字段说明**

- `data_format`：CB 的数据精度格式（`bfloat16`、`float32`、`bfp8` 等）
  - TT-Metal SDPA 中 `cb_exp_max_diff` 等精度敏感的 CB 使用 float32，其他用 bfloat16
  - 精度选择直接影响 carry state 的数值稳定性
- `pack_mode`：dst 到 CB 的 pack 模式
  - 示例：`standard`、`float32_to_bfloat16`、`l1_accumulate`
- `unpack_mode`：CB 到 dst 的 unpack 模式
  - 示例：`standard`、`unpack_to_dest`（直接 unpack 到 dst register，跳过 CB 中转）
- `l1_acc_mode`：是否启用 L1 accumulation（`pack_reconfig_l1_acc(true)`）
  - 启用后 pack 操作变为 `+=` 而不是 `=`
  - TT-Metal SDPA 中用于 `update_cur_mm_out` 的增量累加

**`TTDstLayoutPlan`**

- 绑定的是“长生命周期 compute-local state”到具体 dst/register offset
- 这里不只服务 attention：
  - attention carry
  - chunk recurrence state
  - routed intermediate tile state
  - 其他 compute-local long-lived state
- `fpu_sfpu_ordering`：该 dst region 上 FPU/SFPU 操作的合法执行顺序
  - 例如 attention carry：`matmul(FPU) -> commit -> reduce_max(SFPU) -> wait -> bcast_sub(FPU) -> exp(SFPU) -> ...`
  - 这是 `TTComputeModel.fpu_sfpu_semaphore_model` 在具体 state binding 上的实例化
- `pack_format`：从 dst 输出到 CB 时的精度格式
- `accumulator_mode`：是否启用 accumulator mode（`fp32_dest_acc_en`）

#### 输入

- 冻结后的 `SpatialProgram`
- TT hardware model：
  - topology
  - memory hierarchy
  - NoC / multicast / semaphore capabilities
  - dst/register capacity
  - core kinds 与 placement constraints

#### `TTHardwareModel` schema

`TTProgram` 的合法性不能建立在模糊的“知道机器大概长什么样”上。`IRModule.global_infos["tl.tt.hardware_model"]` 必须是一个 typed object，并至少包含下面这些子模型：

| 子模型 | 关键字段 | 作用 |
|--------|----------|------|
| `TTTopologyModel` | `grid_shape`, `core_kinds`, `core_coordinates`, `neighbor_sets`, `placement_regions` | physical core 拓扑与可放置区域 |
| `TTMemoryModel` | `l1_bytes`, `cb_capacity_rules`, `dram_access_kinds`, `accessor_constraints`, `buffer_visibility_rules` | L1/CB/DRAM 资源约束 |
| `TTNoCModel` | `unicast_support`, `multicast_support`, `route_scope`, `remote_access_rules`, `bandwidth_classes` | NoC 与 transport 约束 |
| `TTSyncModel` | `local_semaphore_support`, `remote_semaphore_support`, `barrier_scopes`, `completion_protocols` | sync/signal 合法性边界 |
| `TTDstModel` | `dst_tile_capacity`, `accumulator_modes`, `register_residency_rules`, `dst_alias_rules` | dst/register 相关约束 |
| `TTABILimitModel` | `compile_arg_limits`, `runtime_arg_limits`, `common_runtime_arg_support`, `patchable_arg_rules` | ABI 与 launch 约束 |
| `TTComputeModel` | `fpu_sfpu_independence`, `fpu_operations`, `sfpu_operations`, `fpu_sfpu_semaphore_model`, `matmul_dst_output_rules`, `pack_unpack_modes`, `l1_acc_support`, `data_format_reconfig_rules` | compute unit 约束与能力 |

设计要求：

1. `TTTopologyModel` 必须足以表达 row/column core grouping 与合法邻接关系。
2. `TTMemoryModel` 必须足以决定某类 `Channel/ResourceIntent` 是否能落成指定 `CB`。
3. `TTNoCModel + TTSyncModel` 必须足以决定某条 `SyncEdge` 是否能合法 materialize 成 semaphore/barrier/multicast protocol。
4. `TTDstModel` 必须足以决定长生命周期 compute-local state 能否驻留 dst/register。
5. `TTABILimitModel` 必须足以约束 compile-time/runtime arg 组织，而不是让 materialization 阶段再失败。
6. `TTComputeModel` 必须足以约束 compute kernel 内部的 FPU/SFPU 操作交错与 dst 争用。

`TTComputeModel` 的背景：

TT-Metal compute core 内部有两个独立执行单元：FPU（matmul 等）和 SFPU（exp/reduce/recip 等）。它们共享 dst register bank，需要通过 `t6_semaphore` 协调。这不是 reader-compute-writer 之间的 task 级 sync，而是 **compute kernel 内部的 micro-architecture 级 sync**。

`blackhole.acc` 混合语义问题的根因之一就是 FPU matmul output 和 SFPU reduce 争用同一 dst region，如果 `TTDstLayoutPlan` 不知道 FPU/SFPU 的操作顺序，就无法保证 correctness。

`TTComputeModel` 至少需要表达：

- `fpu_sfpu_independence`：FPU 和 SFPU 可以并行执行，但共享 dst
- `fpu_operations`：`matmul_tiles`、`bcast_mul`、`bcast_sub` 等 FPU 操作集
- `sfpu_operations`：`exp`、`recip`、`reduce_max`、`reduce_sum` 等 SFPU 操作集
- `fpu_sfpu_semaphore_model`：FPU/SFPU 之间的 semaphore wait/post 协议（`wait_on_max`、`wait_on_zero`、`post`）
- `matmul_dst_output_rules`：matmul 输出到 dst 后，哪些 dst region 被占用，何时可以被其他操作安全读取
- `pack_unpack_modes`：CB 到 dst 的 unpack 模式（`UnpackToDestEn` 等）和 dst 到 CB 的 pack 模式（`float32 vs bfloat16`、`l1_acc_mode` 等）
- `l1_acc_support`：`pack_reconfig_l1_acc(true/false)` 的 L1 accumulation 能力
- `data_format_reconfig_rules`：`reconfig_data_format` 在不同操作之间切换精度的规则

这意味着 `hardware_model` 不是“设备名 + 几个 capability flag”，而是 `ValidateTTTargetProgram` 的直接输入。

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

这里必须明确唯一物化路径：

1. `MaterializeTTExecutableSpec` 是唯一允许从 `TTProgram` 生成 executable body 的 pass。
2. 它直接把 `TTProgram` 物化成：
   - materialized `blackhole.*` attrs / `ExecutableSpec`
   - target-lowered `PrimFunc.body`
3. 不保留长期存在的“等价 builtin emission plan”第二路径。
4. `LowerBlackholeOps`、`codegen_blackhole`、`rt_mod_blackhole` 都不得在 materialization 之后继续发明新的 TT builtins、CB 关系或 ABI 字段。

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
2. 一旦某个 `PrimFunc` 挂上 `tl.semantic_program`，post-semantic-lift pass 默认都视为 `unsafe`；只有经过显式审计并声明为 `safe` 的 pass，才允许在不失效 companion IR 的前提下运行。
3. 如果后续 pass 必须实质性改动相关 TIR 结构，就必须显式失效并删除：
   - `tl.semantic_program`
   - `tl.spatial_program`
   - `tl.tt_program`
   然后重跑 recovery/lowering。
4. `blackhole.*` materialized attrs 不是上游 IR 的真源；它们可以被整体重建，而不允许被下游 patch 成第二真源。

**Safe TIR mutations（不触发 companion IR 失效的改动类型）**：

第一版 safe 集必须**刻意收得很小**。原因不是这些改动一定会改变算法语义，而是
`SemanticProgram` 直接持有 `Buffer / Var / PrimExpr / TIRAnchor`，只要改动可能改变这些
原子的身份、canonical form 或 anchor 对应关系，就可能把 companion IR 留成“语义上等价、
引用上失效”的悬挂状态。

因此，第一版只把下面这些改动视为 safe：

- 只改 `PrimFunc.attrs` / `IRModule.global_infos` 的 metadata 更新
- 只增加调试、诊断、打印辅助信息，不改任何被 companion IR 直接引用的 TIR 原子
- 纯分析 pass / verifier pass
- 具备**显式 rebinding 逻辑**的表达式级改写：
  - pass 必须声明自己会重写所有受影响的 `TIRValueBinding`
  - pass 必须重建所有受影响的 `TIRAnchor` 内嵌 `PrimExpr`
  - pass 结束后必须仍能通过 `ValidateStatefulSemanticIR`

以下 TIR mutation 为 "unsafe"，必须触发失效并重建：

- 默认的 constant folding、arithmetic simplification、常量传播
- 默认的 dead code elimination
- 默认的 `PrimExpr` 等价替换，即使它“按语义等价”
- 改变 buffer scope（如 `local.fragment` -> `blackhole.acc`）
- 改变 loop structure（fusion / split / reorder / unroll）
- 增加或删除 `Allocate` / `DeclBuffer`
- 改变 `BufferStore` / `BufferLoad` 的 access pattern
- 改变 launch axes 或 thread_extent 结构
- 改变 block structure（增删 `BlockRealize`）

实现方式：

1. 每个 post-semantic-lift pass 必须声明自己是 `safe` 还是 `unsafe`
2. 未声明者一律按 `unsafe` 处理
3. 只有“identity-preserving”或“rebind-aware”的 audited pass 才能声明 `safe`
4. pipeline builder 依据声明插入 invalidation + re-lift

#### 5.6.4 Materialized Attr Ownership 与 Cutover

为了落实“`TTProgram` 是 target contract 真源、`blackhole.*` attrs 只是物化结果”，
必须把 materialized attrs 的 writer ownership 写死。

| Attr | 当前临时 writer | 稳态唯一 writer | Cutover 规则 |
|------|-----------------|----------------|--------------|
| `blackhole.segment_plan` | `SplitBlackholeKernel`、`LowerBlackholeOps`、`rt_mod_blackhole` 的兼容路径 | `MaterializeTTExecutableSpec` | pre-Phase C 可继续作为兼容产物存在，但 semantic/spatial/TT companion IR pass 不得把它当真源 |
| `blackhole.runtime_args` | `LowerBlackholeOps`、`rt_mod_blackhole` 的兼容聚合路径 | `MaterializeTTExecutableSpec` | 一旦 `TTABIPlan` 已能物化该 schema，所有前置 writer 必须删除 |
| `blackhole.cb_configs` | `PlanBlackholeCB` | `MaterializeTTExecutableSpec` | `PlanBlackholeCB` 后续只能作为 `TTCBPlan` 构造/验证 helper，不再直接写最终 attrs |
| `blackhole.core_plan` | `AssignBlackholeCores` | `MaterializeTTExecutableSpec` | `AssignBlackholeCores` 后续只能作为 `TTExecutionPlan` 构造/验证 helper，不再直接写最终 attrs |

补充规则：

1. `blackhole.gemm_contract`、`blackhole.accessors` 等 legacy helper attrs 也遵守同一原则：
   要么被 `TTProgram` 物化正式接管，要么在迁移完成后删除。
2. pre-Phase C 的 legacy writer 只允许为了保持 current direct-path 基线而存在，
   不允许再向上游 companion IR pass 提供“事实来源”。
3. 一旦某个 attr 已由 `MaterializeTTExecutableSpec` 稳定生成，所有更早阶段的 writer
   对该 attr 都视为架构违规。
4. `codegen_blackhole` 与 `rt_mod_blackhole` 可以做 schema 验证、拆分、只读适配，
   但不得继续合成缺失语义、补齐缺失 contract 或回写新真源。

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
- `SplitBlackholeKernel`（仅保留 pre-semantic canonicalization / temporary signal producer 职责）

继续保留，但它们后续只作为 signal producers，不再直接定义最终语义。

额外约束：

1. `SplitBlackholeKernel` 在 semantic lift 之前最多只允许写入临时 `segment_kind` /
   compatibility attrs，帮助当前 direct-path 维持基线。
2. 这些 pre-semantic `blackhole.segment_plan` / `blackhole.segment_kind` 信息不是
   semantic truth，也不是 spatial truth。
3. `AnalyzeSemanticStructure`、`LowerToSpatialProgram`、`LowerSpatialProgramToTTTarget`
   不得读取这些 compatibility attrs 来裁决语义对象身份或空间结构。

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
  - `acc_o : compound_state(carry, update_semantics=rescale_accumulate)`
  - `scores_max : vector_state(carry)`
  - `scores_sum : vector_state(carry)`
  - `logsum : scalar_state(cross_phase)` 或 epilogue-derived result
- `Relation`
  - `scores_max reduced_from qk_scores`
  - `scores_sum reduced_from exp_scores`
  - `acc_o / scores_max / scores_sum carried_across kv_chunk_phase`
  - attached `CombineSpec(kind=rescale_accumulate, algebraic_traits=[online_stable])`
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
  - `score_state : vector_state(update_semantics=iterative_mask_select)`
  - `selected_index : index_state`
  - `selected_value : vector_state`
- `Relation`
  - `selected_index selected_from logits_domain`
  - `selected_value applies_to selected_index`
  - attached `SelectionSpec(score_expr=reduce_max, rank_kind=iterative, output_form=index_value_pair)`
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
  - attached `SegmentSpec(segment_ids=group_idx_for_bx, segment_sizes=group_sizes, segment_offsets=group_offsets, padding_policy=padded_offsets)`
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

- `ProgramPhase`
  - `gate_up_phase`：对应第一个 `T.Kernel` block（gate + up matmul + activation）
  - `down_phase`：对应第二个 `T.Kernel` block（down matmul + combine）
  - 两个 phase 之间通过 `shared_buffers`（`intermediate_tile`）连接
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
- expert-kernel ABI（composable `role_set = {reader, compute, writer, sender}` + `role_flags` 区分 gate_up / down 行为）
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
  - attached `PageSpec(page_table=block_table, page_size=block_size, logical_to_physical_expr=block_table[bid, floordiv(idx, ratio)], validity_expr=idx < cache_seqlens)`
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

- `ProgramPhase`（仅 split-decode 场景）
  - `decode_phase`：每个 split 独立执行 query + page stream + decode update
  - `combine_phase`：跨 split 合并 partial results
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
  - `chunk_state : compound_state(cross_phase, update_semantics=decay_outer_product)`
  - `decay_state : vector_state(ephemeral)`
  - `dt_state : vector_state`
- `Relation`
  - `chunk_state carried_across chunk_phase`
  - `decay_state applies_to chunk_state`
  - `chunk_state reduced_from step_inputs`
  - attached `RecurrenceSpec(carry_state=chunk_state, decay_expr=exp2(dA_cs_last - dA_cumsum), update_order=chunk_sequential)`
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
| `SplitBlackholeKernel` | pre-semantic canonicalization / temporary signal producer | 收窄；最终不再承担 `blackhole.segment_plan` writer 职责 |
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
SplitBlackholeKernel（仅 canonicalization / temporary signal-only）
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

说明：

- 在 `MaterializeTTExecutableSpec` 落地之前，管线中仍可能存在 legacy `blackhole.*` attrs
  作为 compatibility artifact。
- 这些 attrs 在迁移期只服务现有 runtime/codegen 消费面，不构成 semantic/spatial/TT
  三层 companion IR 的真源。

### 8.3 迁移阶段

**Phase A: Semantic IR**

Phase A 工程量大（~13 个新 TVM object 类型 + analysis + validation），因此细分为两个子阶段：

**Phase A1: 基础 semantic infrastructure + minimal multi-family recovery**

1. 加入核心 typed companion object：
   - `Domain / State / Relation / Phase / SemanticRegion`
   - `TIRAnchor / TIRValueBinding`
   - `CombineSpec`（最先需要，因为它是 attention carry 的关键）
2. 明确 `PrimFunc.attrs["tl.semantic_program"]` 挂载 contract
3. 落 `AnalyzeSemanticStructure`（最小版本）
   - 构造 `AtomicEffect`
   - 构造 dependence / carry / domain signature
   - 生成 anchor 与 semantic region partition
   - 最小覆盖面：dense tiled compute + stateful reduction-update + 至少一个非 attention 的 indexing/remap skeleton（`topk` 优先，`paged decode` 可接受）
4. 落 `LiftToStatefulSemanticIR`（最小版本）
5. 落 `ValidateStatefulSemanticIR`（核心验证规则）
6. 要求现有 target lowering 从 semantic 真源消费信息，而不是继续自己猜主要语义
7. **验收 gate**：
   - `flash-attn` compile-path 在新架构下跑通
   - copy / GEMM zero-regression
   - 至少一个非 attention workload 的 semantic skeleton recovery test 通过（`topk` 优先，`paged decode` 可接受）

**Phase A2: 泛化 recovery + all descriptor types + DSL annotation protocol**

1. 加入剩余 typed descriptor：
   - `SelectionSpec / SegmentSpec / PageSpec / RecurrenceSpec`
2. 实现 DSL semantic annotation protocol（`T.annotate_semantic`）
3. 泛化 `AnalyzeSemanticStructure` 覆盖所有 workload family：
   - selection / indexing recovery
   - grouped / ragged / routed domain recovery
   - paged / indexed sparse access recovery
   - chunk recurrence recovery
4. 对每个 workload family 实现 recovery accuracy test：
   - 构造 `SemanticProgram` 后验证 state/relation/phase 是否符合预期
5. **验收 gate**：所有 workload family 的 recovery accuracy test 通过

**Phase B: Spatial Program IR**

1. 加入 typed companion object：
   - `ProgramPhase / Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
2. 明确 `PrimFunc.attrs["tl.spatial_program"]` 挂载 contract
3. 落 `LowerToSpatialProgram`
   - 初期走 **deterministic heuristic** 路线：`SpatialPolicy` 直接产出唯一 `SpatialProgram`，不做 candidate search
   - `SpatialCandidate` 和 `SpatialCostModel` 作为 debug artifact 挂接，但不走搜索
   - 把 candidate search 推迟到 Phase B 之后的优化阶段
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
