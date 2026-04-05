# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19（创建），2026-04-03（重写并收敛为当前版本），2026-04-05（基于 review 与代码交叉审计修订），2026-04-06（在 `Phase A` 完成后按当前执行状态精简与同步）
- **状态**: 当前唯一权威总体设计文档
- **范围**: `tilelang_repo` Blackhole 编译器架构、compiler-internal IR 分层、TT 目标映射、runtime materialization 边界
- **取代**:
  - 已归档的混合 runtime 架构叙事：`tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
  - 已归档的旧单层 `Stateful Tiled IR` 顶层方向

## 1. 问题定义

Blackhole 现在面对的核心问题，已经不是“怎么再打印一个 TT-Metal kernel 字符串”，而是 **复杂前端计算语义和 spatial/dataflow 硬件编程模型之间的结构性表示鸿沟**。

这个鸿沟在仓库里已经不是抽象问题，而是被多类真实样例共同暴露出来：

- `examples/flash_attention/`、`examples/online_softmax/`、`examples/attention_sink/`
  - 暴露的是 `stateful reduction-update / carry / normalized recurrence` 语义
- `examples/fusedmoe/`、`examples/grouped_gemm/`
  - 暴露的是 `routed / grouped / ragged dispatch` 语义
- `examples/topk/`、`examples/deepseek_v32/topk_selector.py`
  - 暴露的是 `selection / index generation / selected subset` 语义
- `examples/blocksparse_attention/*paged*.py`、`examples/deepseek_mla/example_mla_decode_paged.py`、`examples/flash_decoding/`
  - 暴露的是 `paged / indexed / sparse decode` 语义
- `examples/linear_attention/example_mamba_chunk_state.py`、`examples/kda/`、`examples/gdn/`
  - 暴露的是 `chunked recurrence / scan / cross-step carry state` 语义

这些 workload family 虽然长相不同，但它们有共同特征：

1. `PrimFunc / TIR` 在 tile op 被 lower、fragment 被 scalarize 之后，无法稳定保留：
   - state
   - update law
   - access map
   - routed / paged / ragged / indexed domain
   - ordered-update / carry boundary
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
3. 让 domain、state、update、layout、task、sync、TT resource planning 在正确层里成为一等对象。
4. 在 target 层保持 TT-first，但让 semantic 层和 spatial 层具备超出 TT 的抽象价值。
5. 让 codegen/runtime 退回 materialization 和 execution，而不是继续承担语义重建。
6. 让这套设计不仅能解释当前 `flash-attn`，还要能承接其他复杂前端计算 family。

### 2.1.1 可证明性边界

本设计不承诺下面这个强命题：

- 对任意未来 workload family，固定不变的 `Stateful Semantic IR` vocabulary 都一定足够

这个强命题本身就不该被承诺。对任意程序自动恢复所有非平凡语义性质，会触到不可判定性边界；因此总设计只追求更弱、也更正确的命题：

- 对一个有限的 semantic core，如果某类 workload 的算法语义可以归约到这套 core，
  则 `Stateful Semantic IR` 可以作为该 workload family 的有界抽象域

因此，本设计要求的“通用性”不是“以后永远不需要新增任何概念”，而是：

1. 正式长期 vocabulary 必须保持小闭集
2. 新增 workload 时，优先证明它是否可归约到现有 semantic core
3. 只有当某个新语义轴确实跨 family 复用、且无法归约到现有 core 时，才允许扩 semantic core
4. 如果某类信息本质上属于 task/layout/sync/placement/transport/ABI，就必须进入 `Spatial Program IR` 或 `TT Target IR`，而不是继续塞回 `Stateful Semantic IR`

换句话说，本设计允许证明的是：

- bounded semantic vocabulary
- sound abstraction over a chosen workload class
- evidence-to-core reducibility

本设计不承诺：

- arbitrary future workload completeness
- automatic semantic recovery of every nontrivial program property

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

### 3.1 系统评审结论与优先级

本节收敛 2026-04-05 两轮系统性 design review 的结论。评审范围同时覆盖：

- TileLang 前端 workload family 与通用 lowering
- Blackhole 当前 pass / runtime / materialization 主链
- TT-Metal host API、program factory、device kernel 协议

这些结论不是第二份总体设计；它们是本总设计后续实现时必须遵守的收口约束。

#### P0：已在 Stage 0 / Phase A 收口，后续不得回退

1. **单一真源 cutover 必须变成机器可执行的删除门槛**
   - `MaterializeTTExecutableSpec` 已被定义为唯一稳态 writer，但当前实现里
     `LowerBlackholeOps`、`codegen_blackhole`、`rt_mod_blackhole`、`BlackholeModule`
     仍共同维护 `segment_plan / runtime_args / accessors / buffer-role fallback`
   - 因此必须为每一类 materialized attr 明确：
     - 哪个 companion IR / TT object 是上游真源
     - 何时允许 compatibility writer 继续存在
     - 对应稳态字段齐备后，哪些 legacy writer / reader / fallback 必须删除
   - 否则 layered IR 落地后，旧 attrs 仍会继续长成第二真源

2. **companion IR 的 lift 点、失效规则、rebind contract 必须和真实 lowering 主链对齐**
   - 当前 Blackhole 路径在 semantic 之前已经会经过：
     - `AnnotateBlackholeCopySemantics`
     - `BlackholeDeviceResourceCanonicalization`
     - `LowerOpaqueBlock`
     - `SplitHostDevice`
   - 这些 pass 会改变 `Buffer / Var / PrimExpr` 的 identity 与结构关系；
     当前资源 canonicalization 甚至已经需要 name-based fallback 才能维持逻辑绑定
   - 当前这部分已经在 Stage 0 / Phase A 收口为：
     - 固定 semantic lift canonicalization 点
     - post-lift `preserve / typed_rebind / invalidate` contract
     - typed `TIRValueBinding` rebind API
     - unsafe mutation 后整体删除并重建 `tl.semantic_program / tl.spatial_program / tl.tt_program` 的硬规则

3. **`ProgramPhase` 的稳定宿主固定为 module-scope device program container**
   - 当前主链里 `SplitHostDevice` 发生在 semantic lift 之前，因此 cross-function 的 phase truth
     不能继续依赖“挂在单个 `PrimFunc.attrs` 上”这种假设
   - 第一版直接固定：
     - `ProgramPhase / phase_order / shared_buffers / cross_func_sync` 的稳定宿主是
       `IRModule.global_infos["tl.device_programs"]`
     - 单 `PrimFunc` 程序只是 `member_funcs.size() == 1` 的退化情况
     - `PrimFunc.attrs["tl.spatial_program"]` 只保留 member-local spatial truth
   - 同时必须在 `SplitHostDevice` 之前先建立 device-program registry，
     保留 multi-`T.Kernel` 的 membership / order；split 之后只做 member rebind
   - 否则 Spatial IR 在 fusedmoe / split-decode / 多 kernel 程序上会重新退化成实现约定

4. **TT ABI 中的 common-runtime 必须成为一等对象**
   - 总设计已经多次把 ABI 边界描述成 `compile-time / common-runtime / per-work`
   - 但当前 `TTKernel` / `TTABIPlan` 核心对象里还没有显式的 common-runtime schema
   - 必须补齐：
     - `TTKernel` 是否显式拥有 common-runtime bindings
     - `TTABIPlan` 是否显式拥有 `common_runtime_arg_specs`
     - accessor / work-distribution / shared per-kernel metadata 如何进入 common-runtime 层
   - 否则 `blackhole.common_runtime_args` 很容易继续以 compatibility attr 形式长期滞留

5. **semantic recovery 不能只押注 post-lowering replay，必须允许 early semantic capture**
   - 当前 `phase.py` 中 `AnnotateBlackholeCopySemantics` 已经明确放在 `LowerOpaqueBlock` 之前，
     说明有些语义事实在结构被压碎前更容易、也更可靠地保留下来
   - 因此第一版必须允许在 `LowerTileOp` 或更早的 canonicalization 阶段，用 compiler-internal
     lightweight typed signal 保留：
     - tile op family
     - reduction / selection / recurrence skeleton
     - pipeline stage skeleton
     - multi-`T.Kernel` membership / order
   - 这些 signal 不是 `SemanticProgram`，也不是 `SemanticSupplement`；
     它们只是 `AnalyzeSemanticStructure` 的输入，semantic lift 成功后不再充当真源

#### P1：在 Phase B / C 落地前必须补齐

1. **TT route / transport / protocol 需要一等对象，不能只留在 `remote_core_descriptors` 或 runtime args**
   - TT-Metal 真实程序已经广泛使用：
     - ring / line / tree reduction
     - multicast / fanout / fanin
     - fabric mux / fabric connection
     - sender / receiver / reducer / relay 等 protocol role
   - 仅靠 `TTExecutionPlan.remote_core_descriptors` 与零散 `protocol` 字段不足以承载这些结构
   - 因此应补一个显式的 transport / route / protocol subplan，
     至少覆盖 unicast、multicast、tree、ring/line、fabric mux 这类通信协议骨架

2. **`Placement.kind` 与 `ResourceIntent.kind` 不能继续保持“待补的小闭枚举”状态**
   - 这两个对象当前只有“应保持小闭枚举”的原则，但没有第一版 base family
   - 这是实现期最容易重新长出 noun bag 的空洞字段
   - 处理原则只能二选一：
     - 要么现在就给出第一版 base family
     - 要么在第一版 schema 中先去掉 `kind`，只保留 bindings + typed traits

3. **`TTDstLayoutPlan` 与 `TTComputeSyncPlan` 的 ownership 必须更明确**
   - 二者当前都在描述“同一 compute-local state 在同一 dst region 上如何合法存在”
   - 必须写死单向 ownership：
     - `TTDstLayoutPlan` 只负责 residency / alias / offset / capacity / accumulator mode
     - `TTComputeSyncPlan` 只负责该 residency 前提下的 hazard / ordering requirement 与 protocol satisfaction
   - 否则 attention / recurrence / routed intermediate state 一落地，就会再次出现 dst truth overlap

4. **host logical layout / tilize-untilize / transpose responsibility 需要进入稳态 materialization contract**
   - supporting audit 已经确认这不是优化项，而是 correctness contract
   - 当前 direct runtime 对 GEMM 已显式承担 transpose、tilize、untilize
   - 总设计里还应补清：
     - host logical tensor layout 的 schema 归属
     - host/device layout conversion 的 responsibility
     - 这部分属于 `ExecutableSpec` program-shared metadata、`KernelSpec`，还是 runtime-only helper

#### P2：方向已经证明正确，应保持不回退

1. **`Stateful Semantic IR` 的 semantic core 收敛方向正确**
   - `Domain / State / Update`
   - `AccessMap / UpdateLaw` 作为 `Update` 的 typed 组成部分
   - 不再为 paged / routed / selection / recurrence 并排维护 workload-specific descriptor family

2. **Spatial 层采用 small-closed family + fixed trait axes 的方向正确**
   - `Task / Channel / Layout / WorkPartition` 当前这组 base family 足以覆盖
     attention、selection、routing、paged decode、chunk recurrence 的主要空间骨架
   - `traits` 收成固定轴，而不是自由字符串，是防止 case-by-case 膨胀的关键约束

3. **TT 层采用 small-closed target family + capability-oriented hardware model 的方向正确**
   - `TTKernel.kind + traits`
   - `TTCBPlan / TTSemaphorePlan / TTComputeSyncPlan / TTDstLayoutPlan`
   - `TTHardwareModel` 的 capability-oriented 子模型
   - `ExecutableSpec` 收成 program container + `KernelSpec[]`
   - 这些方向都应保持，不应因当前实现过渡状态回退到 monolithic attr protocol

4. **这套总体设计不是 attention-only 设计**
   - 两轮评审已确认：
     - `flash-attn`
     - `topk`
     - `fusedmoe`
     - paged decode
     - chunk recurrence / scan
   - 都要求同一套 layered IR 边界，只是对 semantic / spatial / TT contract 的投影不同
   - 因此后续实现验证不能再只用 attention family 当唯一 gate

#### 这组优先级对当前执行的直接要求

当前阶段必须继续保持下面这组执行纪律：

1. P0 相关内容已经进入 Stage 0 / Phase A 文档与代码合同，后续实现不得回退
2. P1 仍必须作为 Phase B / Phase C 的 schema gate，而不是“实现时再看”
3. P2 中被判定为“方向已对”的部分，不允许在实现期因为当前样例便利性回退成 case-by-case matcher 或 noun bag schema

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
| `Stateful Semantic IR` | 这个程序在逻辑域上如何更新算法状态？ | 算法语义 | `SemanticProgram` |
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
| Dense tiled compute | `copy`、`GEMM`、`grouped_gemm`、`split-k` | tile domain、tensor state、map/reduce update | load/compute/store task、tile layout、split-k partition | reader/compute/writer、CB transport、ABI、core placement |
| Selection / indexing | `examples/topk/`、`deepseek_v32/topk_selector.py` | index-valued state、selection update、selector axis、masked predicate | select task、index channel、selected-subset partition | index scratch、selector/reduction kernel role、index ABI |
| Routed / grouped / ragged dispatch | `examples/fusedmoe/` | remapped domain、segmented/indirect access map、expert/index state、materialized intermediate update | route/compute/combine task、grouped layout、expert partition、ragged sync | routed buffer/index buffer、dispatch ABI、core-group mapping |
| Paged / indexed / sparse decode | `blocksparse_attention/*paged*.py`、`deepseek_mla/example_mla_decode_paged.py`、`flash_decoding/` | paged/indirect access map、carry state、bounded predicate、merge update | page-stream task、paged layout、page partition、merge/combine sync | page/index buffer、runtime descriptor、multicore split merge plan |
| Stateful reduction-update | `flash_attention/`、`online_softmax/`、`attention_sink/`、`norm/` | carry state、normalized reduction/update law、causal/predicate-bound domain | update task、carry channel、persistent resource intent、ordered-update completion sync | dst layout、persistent carry plan、reader/compute/writer contract |
| Chunked recurrence / scan | `linear_attention/example_mamba_chunk_state.py`、`kda/`、`gdn/` | cross-step state、chunk domain、ordered recurrence update、decay access/update law | chunk task graph、chunk partition、state carry、ordered-update completion sync | persistent carry、dst/CB realization、execution order、runtime chunk descriptors |

这张表是总设计的边界声明。后续实现顺序可以从其中某一个 family 起步，但总设计不能被单一 consumer 绑死。

## 5. 各层 IR 设计

### 5.1 `Stateful Semantic IR`

#### 为什么需要这一层

这一层只回答一个问题：**程序在逻辑域上如何更新算法状态？**

`TIR` 对前端和通用 lowering 来说当然足够强，甚至是图灵完备的；问题不在于它“表达不了”，
而在于它对 Blackhole 中后段来说过于自由。相同的算法语义可以落成很多等价 TIR 形态，
一旦再经历 canonicalization、loop 变形、buffer 化、fragment/builtin lowering，后段就只能从：

- 被 scalarize 的 `BufferLoad / BufferStore`
- fragment helper 名字
- target builtin
- runtime 侧 heuristics

里重新猜“程序到底在算什么”。

这正是 attention-like kernel 上 `blackhole.acc` 混合语义的根因：算法 state 和 target scratch
在晚期 TIR 里已经难以稳定区分。MoE、paged decode、chunk recurrence 的
`domain/index/carry/update-law` 也会在同一位置被压碎。

因此，`Stateful Semantic IR` 不是为了增强语言表达能力，而是为了在 TT 资源决策之前，
先把后端真正关心的**算法不变量**冻结成稳定真源。

#### 设计目标

1. 在做任何 spatial / TT 决策之前，先冻结算法语义真相。
2. 把“逻辑点集”“算法状态”“状态更新”从实现 scratch、task 切分、target contract 中剥离。
3. 用统一语义骨架覆盖：
   - selection / topk
   - routed / grouped / ragged dispatch
   - paged / indexed sparse access
   - stateful reduction-update
   - recurrence / scan

#### 核心对象

`Stateful Semantic IR` 的长期 companion schema 只保留一套很小的 core：

| 对象 | 关键字段 | 含义 |
|------|----------|------|
| `SemanticProgram` | `domains`, `states`, `updates` | 语义真相容器 |
| `Domain` | `axes`, `constraints`, `predicate` | 逻辑迭代域；只表达哪些逻辑点存在 |
| `State` | `state_type`, `lifetime`, `role` | 算法上有意义的值 |
| `Update` | `domain`, `reads`, `writes`, `law`, `guard`, `anchor` | 在某个逻辑域上对 state 进行一次更新 |

`AccessMap` 与 `UpdateLaw` 是 `Update` 的一等组成部分，不再额外长出平行 descriptor 家族。
`SemanticRegion` 保留为 recovery/debug 视图，但不再是 semantic 真源的 core object。

这条边界必须贯穿全文：

1. 第一层 IR 的真源是 `SemanticProgram(Domain / State / Update)`。
2. `AccessMap / UpdateLaw` 作为 `Update` 的组成部分存在，而不是另一套平行核心对象。
3. `TIRAnchor / TIRValueBinding / AtomicEffect / SemanticRegion` 都只是恢复、绑定或投影视图辅助对象，不得反客为主变成新的第一层 schema 核心。

#### Semantic Core 与 Analysis Evidence

为了避免 `Phase A` 退化成无限增长的语义词汇表，`Stateful Semantic IR` 必须明确区分两类东西：

1. **semantic core**
   - 对外成立、长期稳定、允许被后续层依赖的正式语义对象
   - 只包括：
     - `Domain`
     - `State`
     - `Update`
     - `AccessMap`
     - `UpdateLaw`
     - 少量固定 role / law / trait 轴

2. **analysis evidence**
   - 只用于从 TIR 恢复 semantic core 的中间证据
   - 例如局部 selection pairing、arg-reduction target、recurrence edge、source-state 依赖
   - 它们可以存在于 analysis attrs 中，但不能被视为长期 companion IR vocabulary

这一区分是 `Phase A` 可自证通用性的关键：

- semantic core 必须保持小闭集
- analysis evidence 允许随恢复需求变化，但它们不是正式 vocabulary
- 每一个 evidence 都必须能归约到某个已有 core 字段

若某个 evidence 不能归约到下列任一项：

- `State.role`
- `UpdateLaw.kind`
- `UpdateLaw.source_states`
- `AccessMap.traits`
- `Update.bindings`

则说明两种可能：

1. semantic core 少了一个真正基础、跨 family 复用的语义轴
2. 该信息不属于 `Stateful Semantic IR`，而属于 `Spatial Program IR` 或 `TT Target IR`

因此，`Phase A` 的设计纪律不是“尽量少加 evidence”，而是：

- evidence 不是 vocabulary
- evidence 必须可归约
- 不可归约的复杂性必须分流到 Phase B / C

#### 对象设计

**`Domain`**

`Domain` 只表达**逻辑点集**，不默认表达执行顺序。顺序依赖属于 `UpdateLaw`。

```text
Domain {
  axes: [Axis]
  constraints: [BoolExpr]
  predicate: BoolExpr?
}
```

其中：

- `Axis`
  - `name`
  - `extent_expr`
  - `kind`
    - `data`
    - `reduction`
    - `step`
    - `selector`

约束：

1. 不再把 `dense / paged / routed / ragged` 做成 `Domain.kind` workload 枚举。
2. paged、grouped、routed 这类特征应当主要体现在 `AccessMap` 或 `predicate/constraints` 上。
3. `Domain` 默认只回答“哪些逻辑点存在”；`ordered update` 不写在 `Domain` 本体里。

**`State`**

`State` 表达算法上有意义的值，不表达 TT backing resource。

```text
State {
  name
  state_type
  lifetime
  role
  backing_buffer?
}
```

第一版 `state_type` 采用接近经典 IR / MLIR shaped type 的骨架：

- `Scalar<T>`
- `Tensor<T, Shape>`
- `IndexTensor<IndexSpace, Shape>`
- `Tuple<...>`

这样：

- topk indices、block/page indices、expert ids 都落在 `IndexTensor`
- flash-attn carry、MoE intermediate、chunk state 需要结构化表达时，优先用 `Tuple`，
  而不是重新发明 `compound_state`

`lifetime` 建议至少覆盖：

- `input`
- `output`
- `ephemeral`
- `carry`
- `cross_update`

`role` 只描述算法职责，例如：

- `data`
- `index`
- `mask`
- `carry`
- `intermediate`

**`Update`**

`Update` 是 semantic 层唯一的“可执行语义节点”。它表达：
**在某个 `Domain` 上，如何从若干 `State` 读，向若干 `State` 写。**

```text
Update {
  name
  domain: DomainRef
  reads: [StateRead]
  writes: [StateWrite]
  law: UpdateLaw
  guard: BoolExpr?
  anchor: TIRAnchor
}
```

其中：

- `StateRead  = { state, access_map }`
- `StateWrite = { state, access_map, write_mode }`

`write_mode` 第一版收成最小集合：

- `overwrite`
- `accumulate`
- `ordered_update`

**`AccessMap`**

`AccessMap` 是 semantic 层的关键对象之一。它把“普通 direct access”和
“grouped/paged/indirect/scatter”收进统一 schema：

```text
AccessMap {
  domain: DomainRef
  index_inputs: [StateRef]
  coord_exprs: [IndexExpr]
  validity: BoolExpr?
  traits: Set<AccessTrait>
}
```

`AccessTrait` 第一版建议至少覆盖：

- `affine`
- `projected`
- `broadcast`
- `indirect_read`
- `indirect_write`
- `segmented`
- `paged`

这意味着：

- paged decode 不是单独一套 `PageSpec`
- routed/grouped/ragged 不是单独一套 `SegmentSpec`
- 它们主要是 `AccessMap` 的性质

**`UpdateLaw`**

更新律不再拆成一组互不正交的 descriptor，而是收成 `Update` 的 typed law：

- `MapLaw`
- `ReduceLaw`
- `SelectLaw`
- `RecurrenceLaw`

第一版最小字段：

| `UpdateLaw` | 关键字段 | 典型 workload |
|-------------|----------|---------------|
| `MapLaw` | `expr_anchor`, `traits` | pointwise / transform / masked write |
| `ReduceLaw` | `reduced_dims`, `identity_expr`, `combine_anchor`, `finalize_anchor?`, `traits` | sum/max/reduce + finalize |
| `SelectLaw` | `score_anchor`, `selector_axis`, `selector_kind`, `tie_break_policy`, `output_contract` | topk / argmax / threshold select |
| `RecurrenceLaw` | `ordered_dims`, `init_expr?`, `step_anchor`, `merge_anchor?`, `finalize_anchor?`, `boundary_policy` | online softmax / chunk recurrence / ordered carry update |

其中 algebraic / semantic traits 应作为 `UpdateLaw` 的 typed 字段存在，例如：

- `associative`
- `commutative`
- `idempotent`
- `stable_order_required`
- `normalized`

说明：

- `expr_anchor / combine_anchor / finalize_anchor / score_anchor / step_anchor / merge_anchor`
  都是 typed `TIRAnchor`
- 它们不是新的 semantic object，只是把 `UpdateLaw` 对应的 TIR 计算骨架稳定绑定回来
- `MoE` 的 `combine_output_update` 第一版优先规范成：
  - `MapLaw(weighted_scale)` + `ReduceLaw(accumulate)` 的组合
  - 而不是为 weighted combine 单独新增 `UpdateLaw` variant
- 只有当实践证明这类分解无法稳定覆盖某个 workload family，
  才允许重新评估 `UpdateLaw` variant 数量

#### 内部规范化分析图

长期 public schema 保持小，但编译器内部仍应采用近似 `MemorySSA` 的思路做 state versioning。

推荐内部维护：

- `StateVersion`
- `StateDef`
- `StateUse`
- `StateJoin`

它们用于：

- reaching-def / carry 验证
- semantic invalidation / re-lift
- semantic -> spatial lowering

但不要求一开始就把它们全部暴露成长期 companion IR 的一等公开对象。

#### 输入

- `PrimFunc / TIR`
- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`
- 只有在 IR 无法唯一裁决编译必需语义时，才允许最小显式语义补充

#### 早期语义捕获与 preservation channel

仅靠 post-lowering recovery 去“事后回放”所有语义，工程风险过高。当前代码已经给出明确信号：

- `phase.py` 中 `AnnotateBlackholeCopySemantics` 需要在 `LowerOpaqueBlock` 之前运行
- `BlackholeDeviceResourceCanonicalization` 当前仍有 name-based fallback，说明某些结构身份在后段并不稳定

因此第一版 `AnalyzeSemanticStructure` 必须允许消费一个 compiler-internal 的 early semantic capture 通道。
推荐以轻量 typed annotation / `SemanticSeed` 形式附着到 TIR 或相关 attrs 上，但它必须满足：

1. 只承载“保留容易丢失、但早期廉价可知的事实”，例如：
   - tile op family
   - reduction / select / recurrence skeleton
   - `T.Pipelined` stage skeleton
   - routed / paged / indirect remap marker
   - pre-`SplitHostDevice` 的 multi-`T.Kernel` membership / order
2. 这些 seed 不是第二套真源：
   - semantic lift 成功后，真源仍然只有 `SemanticProgram`
   - seed 只作为 `AnalyzeSemanticStructure` 的输入，之后可整体失效或丢弃
3. 它们不是 `SemanticSupplement`：
   - seed 是编译器自动保留的内部信号
   - supplement 是“IR 无法唯一裁决时”的 typed 裁决通道
4. 它们不得携带：
   - workload family 名字
   - spatial/target 结构
   - ad-hoc kernel 角色或 runtime 协议

一句话说：第一版 semantic recovery 采用
**early semantic capture + canonicalized TIR recovery + typed supplement（仅限歧义裁决）**
的混合策略，而不是只押注晚期 replay。

推荐对象设计：

```text
SemanticSeed {
  seed_kind
  owner_anchor_id
  payload
  source_phase
}
```

第一版 `seed_kind` 固定为：

- `op_family`
- `reduction_skeleton`
- `selection_skeleton`
- `recurrence_skeleton`
- `pipeline_stage_skeleton`

`payload` 只允许携带最小 typed 事实，例如：

- `op_family = {copy, gemm, reduce, select, recurrence}`
- `reduction_skeleton = {reduce_kind, reduced_axes}`
- `selection_skeleton = {selector_axis, selector_kind}`
- `recurrence_skeleton = {ordered_axes, boundary_kind}`
- `pipeline_stage_skeleton = {num_stages, rotation_scope}`

第一版 carrying contract：

1. seed 由 `LowerTileOp`、早期 canonicalization pass、或已有 analysis pass 生产
2. semantic lift 之前，pipeline 负责把它们投影到最终 device `PrimFunc` 的：
   - `PrimFunc.attrs["tl.semantic_seeds"]`
3. `AnalyzeSemanticStructure` 同时消费：
   - canonicalized TIR
   - `tl.semantic_seeds`
   - 现有 `AnalyzeBlackhole*` analysis 结果
4. 一旦 `LiftStatefulSemanticIR` 成功：
   - seed 可整体删除
   - 后续层不得再把 seed 当作真源继续读取

#### 输出

- 冻结后的 `SemanticProgram`

#### 验证职责

1. `Domain` 的 axes / constraints / predicate 完整性
2. `State` 的 `state_type / lifetime / role` 一致性
3. `Update.reads / writes / access_map / update_law` 的闭合性
4. ordered update / carry / join 的 reaching-def 一致性
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

因此，这一层额外引入两类桥接对象，但它们都只服务于 semantic core 的恢复、绑定和重绑：

| 对象 | 关键字段 | 作用 |
|------|----------|------|
| `TIRAnchor` | `func`, `anchor_id`, `anchor_kind`, `span`, `root_effect_ids`, `iter_context` | 给 `Update` 或 `UpdateLaw` 的源计算骨架提供稳定结构锚点 |
| `TIRValueBinding` | `owner_kind`, `owner_ref`, `field_kind`, `element_index?`, `value_kind`, `func`, `buffer?`, `var?`, `expr?`, `anchor_id?` | 为 `Domain / State / AccessMap / Update / UpdateLaw` 字段维护 typed TIR 绑定索引与 rebind 入口 |

设计要求：

1. `State.backing_buffer`、`Domain.axes/constraints/predicate`、`AccessMap.coord_exprs/validity`
   等字段，优先直接持有 `Buffer / Var / PrimExpr`；但编译器仍应同步维护对应的 typed `TIRValueBinding` 索引。
2. `Update.anchor` 绑定 update 主体；`UpdateLaw` 中的各类 `*_anchor` 绑定对应的 law payload（如 combine/step/finalize/score）。
3. `TIRAnchor` 只负责回答“这段语义事实来自哪段稳定 TIR 骨架”，不再重复保存字段级 binding；字段级 binding 全部由 `TIRValueBinding` 统一维护。
4. `TIRValueBinding` 必须按 semantic owner 分类，而不是无类型地存一包 `buffer/var/expr`：
   - `owner_kind` 至少覆盖 `domain / state / access_map / update / update_law`
   - `field_kind` 至少覆盖 `state_buffer / domain_axis / domain_constraint / domain_predicate / access_index_input / access_coord / access_validity / update_guard / law_expr`
   - 重复字段必须带 `element_index`
   - `value_kind` 用来区分 `buffer / var / expr`
5. `TIRAnchor` 是编译器在固定 canonicalization 点上重建出来的结构身份，不是用户可见名字，也不是 case-specific 语义标签。
6. semantic 层与 TIR 的关系必须是显式对象引用与 typed binding，不允许退化成字符串名匹配或位置假设。

一句话说，新的 bridge contract 应该是：

- semantic core 自己保存 typed semantic fact
- `TIRAnchor` 保存“这段语义事实来自哪段稳定 TIR 骨架”
- `TIRValueBinding` 保存“semantic core 的哪个字段、哪个元素绑定到哪个 TIR 原子”

这样 companion IR invalidation / safe-rebind 才能真正围绕 `Domain / State / Update / AccessMap / UpdateLaw`
工作，而不是继续沿用旧 schema 时代的松散绑定模型。

#### 语义恢复的具体规则

`SemanticProgram` 不是把 analysis pass 的零散结论直接拼起来，而是要经过一套稳定的恢复流程：

1. **域恢复**
   - 从 loop nest、launch/work decomposition、predicate、indirect index expr 中恢复 `Domain`
   - 建立 `axes / constraints / predicate`
2. **状态恢复**
   - 从 def-use、buffer scope、materialized output、loop-carried use 恢复 `State`
   - 判定 `input / output / ephemeral / carry / cross_update`
3. **更新恢复**
   - 从 `AtomicEffect`、读写索引关系、归约模式、select 骨架、cross-iteration use 恢复 `Update`
   - 恢复 `reads / writes / access_map / update_law`
4. **状态流恢复**
   - 在 `Update` 图上构造内部 `StateVersion / StateJoin`
   - 判定 carry / join / ordered update 的 reaching-def
5. **区域恢复（导出视图）**
   - 以恢复出的 `Update` 图为真源，按通用切分规则导出 `SemanticRegion`
   - `AtomicEffect` 只作为切分诊断与 trace 辅助，不再直接形成 region 真源
   - `SemanticRegion` 仅作为 debug / recovery / spatialization helper，不是 semantic 真源

这五步里，前四步都在建立“语义事实”；最后一步只是从事实导出便于分析的区域视图。

### 5.2 显式语义补充边界

这一节单独回答一个不同于 semantic core 的问题：
**当 automatic recovery 无法唯一裁决编译必需语义时，编译器如何接收最小显式补充？**

它不是 `Stateful Semantic IR` 的另一套 core schema，也不是对外承诺一套公开 DSL annotation API。
它只是 semantic lift 之前的**裁决机制**，用于补齐少数 IR 无法唯一决定、但后续编译又必须知道的语义事实。

#### 5.2.1 自动恢复与显式补充的边界

Recovery boundary 不按 workload family 定义，而按统一语义系统定义。
`flash-attn`、`topk`、`fusedmoe`、`paged decode`、`chunk recurrence`
只是 `Domain / State / Update` 这组对象的不同组合，不是各自独立的语义入口。

统一语义系统下，后端必须自动恢复的内容包括：

- `Domain`
  - loop/domain/bounds/predicate 的结构骨架
- `State`
  - def-use、buffer scope、lifetime、loop-carried use 所暴露的 state 骨架
- `Update`
  - read / write / reduce / select / remap / carry 的结构骨架
- `AccessMap`
  - direct / indirect / segmented / paged / scatter-gather 的访问骨架
- `UpdateLaw`
  - map / reduce / select / recurrence 的更新律骨架

这些都属于“程序在算什么”的结构事实。只要现有 IR 原理上能表达，就必须由
semantic recovery 自动完成，不能因为当前实现不足就提前要求显式补充。

只有当编译必需的**语义裁决**无法从 IR 唯一决定时，才允许显式补充。
这里的“语义裁决”不是在重述结构本身，而是在裁决结构之外的高层语义归属。第一版只允许下列四类事实被补充：

- `State` 身份裁决
  - 例如某个 state 是正式 `carry`，还是 lowering 残留下来的普通 temporary
- `AccessMap` 语义 trait 裁决
  - 例如某个 indirect access 是普通 gather/scatter，还是逻辑分页/路由 remap
- `UpdateLaw` trait 裁决
  - 例如 tie-break / algebraic trait / boundary policy / normalized
- semantic boundary 裁决
  - 例如某个 ordered update 是否构成稳定 semantic boundary，而不是局部实现细节

显式补充**不允许**表达：

- workload family 名字
- 整段算法模板
- `task / channel / layout / sync`
- `CB / semaphore / core / ABI / runtime_args`
- 任何本来就应该由结构恢复自动得出的骨架事实

一句话概括本边界：

> 能从 IR 分析出来的，一律后端自动分析。
> 只有 IR 无法唯一裁决、但编译又必需的少数语义裁决事实，才允许显式补充。

#### 5.2.2 显式补充的承载方式

第一版不承诺公开 `T.annotate_semantic()` 之类的用户 DSL API，也不引入开放式
`hint(key=value, ...)` 语言。更稳的 contract 是先定义一个 compiler-internal 的 typed supplement channel。

推荐对象：

```text
SemanticSupplement {
  supplement_kind
  target_anchor_id
  payload
  source
}
```

其中：

- `supplement_kind`
  - `state_identity`
  - `access_trait`
  - `update_law_trait`
  - `semantic_boundary`
- `target_anchor_id`
  - 指向 pre-lift recovery 阶段构造出的稳定结构锚点
- `payload`
  - 该次裁决所需的最小 typed 事实
- `source`
  - 标识补充来自 frontend lowering、early canonicalization pass，或未来可能存在的 frontend sugar

推荐承载位置：

- `PrimFunc.attrs["tl.semantic_supplement"]`

它的定位是：

1. semantic lift 之前的 compiler-internal 输入通道
2. 只用于协助 `AnalyzeSemanticStructure / LiftStatefulSemanticIR` 完成裁决
3. `LiftStatefulSemanticIR` 成功后不再是真源；真源仍然只有 `SemanticProgram`

这条设计的目的，是把“允许少量显式补充”与“承诺公开 DSL annotation API”拆开。
前者是 Phase A 必需的编译器 contract；后者只有在反复证明用户源代码层必须显式表达时，才值得设计。

#### 5.2.3 补充规则与验证规则

显式补充只能**缩小歧义**，不能**发明新语义层**。因此验证规则必须写死：

1. supplement 只能裁决允许列表中的四类事实，不能越权描述 spatial/target 结构
2. supplement 必须绑定到稳定结构锚点，而不是源码名字、buffer 名字或位置猜测
3. supplement 只能收窄 recovery 的歧义，不能覆盖已经由结构唯一决定的事实
4. 如果 supplement 与结构恢复结果冲突，`ValidateStatefulSemanticIR` 应报错，而不是默默“以 supplement 为准”
5. supplement 默认应为空；copy / GEMM 以及可稳定恢复的复杂 kernel 不应要求显式补充

因此，这一层真正的原则不是“DSL 可以 override recovery”，而是：

> recovery 先给出候选语义；supplement 只在允许的歧义点上做 typed 裁决；
> 一旦两者发生结构性冲突，编译失败，而不是让补充机制变成第二语义系统。

#### 5.2.4 Phase A 的收敛原则

`Phase A` 当前已经按分阶段设计落地，因此这里不再保留 A1/A2 的实施流水，只保留长期收敛原则：

1. semantic core 维持小闭集：
   - `Domain`
   - `State`
   - `Update`
   - `AccessMap`
   - `UpdateLaw`
   - `SemanticSupplement`
2. automatic recovery 优先于显式补充：
   - 能从 IR 与 canonical evidence 恢复的事实，一律由 recovery 完成
   - supplement 只用于裁决 IR 无法唯一决定、但编译必需的少数语义事实
3. semantic 层不再维护 `CombineSpec / SelectionSpec / SegmentSpec / PageSpec / RecurrenceSpec`
   这类平行 descriptor 家族
4. semantic 层当前已落实更宽的 `AccessMap / UpdateLaw` traits 与 typed supplement contract，
   但仍不承诺公开 `T.annotate_semantic()` 之类的开放式 frontend API

#### Semantic -> Spatial 的投影规则

`Stateful Semantic IR` 和 `Spatial Program IR` 之间不能只停留在“概念上分层”，
还需要有明确的投影规则：

- `Update`
  - 映射为一个或多个 `Task` 候选
- `State`
  - 决定 `Channel` / `SyncEdge` 的存在与类型
- `Domain`
  - 约束 `Layout` 与 `WorkPartition` 的合法候选空间
- `AccessMap`
  - 约束 route / gather / scatter / page stream 这类空间边界
- `UpdateLaw`
  - 决定 ordered update、carry、merge、completion requirement 等必须保留的执行约束

跨层禁止项同样必须明确：

1. semantic 层不泄漏 TT resource 事实
2. spatial 层不重新解释 semantic object 的身份
3. target 层不重新发明 task graph 或 semantic update law

这条规则的意义是：Phase B 只能消费冻结后的 `SemanticProgram`，
不能继续从 raw TIR 或晚期 builtin 序列里“再恢复一次语义”。

#### Simple-Workload Canonical Spatial Fast-Path

simple workload 不应该也被迫走重型 candidate synthesis。第一版直接定义一个 canonical fast-path。

触发条件：

1. `SemanticProgram` 只包含一个主要 `Update`
2. `AccessMap` 不含 `indirect_* / paged / segmented` trait
3. `UpdateLaw` 属于 dense `MapLaw` 或 `ReduceLaw`
4. 不存在 carry / merge / ordered-update / cross-phase state
5. `DeviceProgramInfo.member_funcs.size() == 1`

满足条件时，`LowerToSpatialProgram` 直接构造：

1. 单个 `ProgramPhase`
2. 单个主 `Task`
   - 纯 direct copy 映射为 `transfer`
   - 其余 dense map/reduce 映射为 `compute`
3. 读 state -> `input_channels`
4. 写 state -> `output_channels`
5. 一个 canonical `Layout`
   - 直接由 `Domain.axes` 顺序与 `AccessMap.coord_exprs` 的 contiguous/projected 关系导出
6. 一个 canonical `WorkPartition`
   - 直接复用现有 work decomposition analysis 的 launch/grid 事实

重要约束：

- 这里仍然只是在构造 `SpatialProgram`
- 如果后续 TT 层需要 reader/compute/writer clustering，那是 `TT Target IR` 的职责
- fast-path 不允许偷渡 target noun，也不允许把 trivial case 固化成新的 workload-specific kind

#### `SemanticProgram` 在后续层的主要作用

semantic 层不是“恢复完就挂在 attrs 里”的静态记录。它在后续编译阶段至少承担四个明确作用：

1. **validation truth**
   - `ValidateStatefulSemanticIR` 在这一层检查 `Domain / State / Update / AccessMap / UpdateLaw`
     是否自洽，尽早暴露 carry、ordered update、indirect access、state identity 的错误
2. **spatialization input**
   - `LowerToSpatialProgram` 只能从 `SemanticProgram` 读取 must-preserve 的算法约束：
     - 哪些 `Update` 必须切开
     - 哪些 `State` 必须跨 task 流动
     - 哪些 `UpdateLaw` 要求 ordered update / completion / merge
     - 哪些 `AccessMap` 要求 gather / scatter / paged / routed 边界
3. **invalidation cut**
   - semantic lift 之后，TIR pass 的 safe/unsafe、rebind、re-lift 都以 semantic core 为判据，
     不再以 ad-hoc attrs 或 runtime schema 为判据
4. **workload normalization**
   - `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
     最终都先收进同一套 `Domain / State / Update / AccessMap / UpdateLaw`
     语义骨架，再由 spatial / TT 层做各自的组织与 target realization

因此，semantic 层后面的“主要作用”不是直接参与 codegen，而是：
**给后续所有层提供唯一、稳定、可验证、可失效重建的算法语义真源。**

#### 最小 TT Target Contract

`TTProgram` 也需要一个最小闭合集，避免 target 层无限膨胀。
第一版至少冻结下面这些对象：

- `TTKernel`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`

它们分别回答：

- kernel family / specialization / source binding
- CB identity / page / capacity / format
- program-level semaphore / barrier identity / visibility / binding
- compute-kernel internal sync / hazard protocol
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

更具体地说，`ExecutableSpec` 应收成一个 **program container + per-kernel materialized view**：

- program-shared 部分：
  - buffer materialization
  - shared semaphore descriptors
  - shared work distribution / launch metadata
  - host entry invocation metadata
- per-kernel 部分（通过 `KernelSpec` 挂接）：
  - kernel creation config
  - compile-time / common-runtime / per-work ABI
  - accessor descriptors
  - remote-core descriptors
  - kernel-local synchronization bindings

稳态下应满足：

- `KernelSpec` 是 per-kernel ABI 与 materialization config 的稳定所有者
- `ExecutableSpec` 顶层只保留 program-shared metadata 与 `KernelSpec[]`
- 顶层 `runtime_args`、`common_runtime_args`、`accessors` 等 aggregate view 如果继续存在，只能是 compatibility view，不是长期 truth

而不应该再新增：

- semantic `Domain / State / Update` 解释
- semantic `AccessMap / UpdateLaw` 真相
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
  - 验证 `carry state / normalized recurrence law / causal domain`
- `topk`
  - 验证 `selection law / index-valued state`
- `fusedmoe`
  - 验证 `remapped domain / segmented access map / materialized intermediate update`
- `paged decode`
  - 验证 `paged access map / indirect update / merge law`
- `chunk recurrence`
  - 验证 `carry state / ordered recurrence law / chunk domain`

这样统一语义系统才能从“设计目标”变成“可验证 contract”。

#### 设计收敛顺序

为了避免再次滑回 monolithic lowering，本设计的收敛顺序应固定为：

1. 先冻结最小 semantic core
2. 先把 recovery boundary 立清
3. 再定义 semantic -> spatial 投影规则
4. 再冻结最小 `TTProgram` contract
5. 最后把 `ExecutableSpec` 收回到 materialization boundary

这条顺序本身就是架构约束：

- 不先做 semantic core，就不要设计更宽的显式语义补充机制
- 不先做 semantic -> spatial mapping，就不要让 target 层继续发明 task/channel
- 不先做 `TTProgram`，就不要继续把 richer target contract 塞进 `ExecutableSpec`

### 5.3 `Spatial Program IR`

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
| `Task` | `kind`, `traits`, `semantic_updates`, `input_channels`, `output_channels`, `layout_bindings`, `partition_bindings`, `resource_intents`, `execution_scope`, `program_phase` | 逻辑执行单元 |
| `Channel` | `producer`, `consumer`, `payload_states`, `payload_kind`, `transport_semantics`, `traits`, `ordering`, `program_phase` | task 间数据流边 |
| `Layout` | `kind`, `traits`, `domain_bindings`, `mapping_expr` | 分布式数据组织 |
| `WorkPartition` | `kind`, `traits`, `domain_bindings`, `partition_expr`, `load_balance_policy` | 逻辑工作划分 |
| `Placement` | `kind`, `traits`, `target_ref`, `constraints` | virtual placement 关系 |
| `SyncEdge` | `kind`, `traits`, `scope`, `source_task`, `target_task`, `update_or_state_bindings` | 同步要求 |
| `ResourceIntent` | `kind`, `traits`, `attachment_ref`, `payload_states`, `capacity_hint`, `visibility`, `reuse_policy` | 尚未 target bind 的资源需求 |

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
- `ProgramPhase` 不等于 semantic 层的 algorithmic cut / ordered-update boundary：
  - semantic 层表达的是 `UpdateLaw` 所要求保留的顺序、carry、merge 约束
  - `ProgramPhase` 是 spatial 层的执行组织单元
- `ProgramPhase` 是 **强边界**：
  - `Channel` 不允许跨 `ProgramPhase`
  - phase 间数据只能通过 `shared_buffers` 物化
  - producer phase 通过 `global_sync_after` 结束，consumer phase 通过 `global_sync_before` 开始
  - 也就是说，跨 phase communication 不是普通 channel，而是“materialized state + global phase sync”

#### `ProgramPhase` 的宿主与多 `T.Kernel` 程序承载

`ProgramPhase` 不能停留在“概念上存在”但没有稳定宿主的状态。结合当前主链里
`SplitHostDevice` 早于 semantic lift 的事实，第一版直接固定为下面这条承载规则：

1. 所有 cross-function 的 device-program / `ProgramPhase` truth 统一挂在：
   - `IRModule.global_infos["tl.device_programs"]`
2. 推荐字段：
   - `program_id`
   - `member_funcs`
   - `phase_order`
   - `shared_buffers`
   - `cross_func_sync`
3. 单个 `PrimFunc.attrs["tl.spatial_program"]` 只保留该 member function 内部的
   local `Task / Channel / Layout / SyncEdge / WorkPartition` truth；
   任何跨 member function 的 phase-boundary materialization、
   `shared_buffers` ownership、global phase sync 都只能由 module-scope device program object 持有。
4. 单 `PrimFunc` 程序不是第二套宿主模式，而只是：
   - `member_funcs.size() == 1`
   的退化情况
5. 为了让 semantic lift 继续保留在 `SplitHostDevice` 之后，
   pipeline 必须在 `SplitHostDevice` 之前先建立轻量 device-program registry，
   保存 multi-`T.Kernel` 的 membership / order / boundary skeleton；
   split 之后只把 member function 绑定回同一个 `program_id`
6. 不允许同时让：
   - module-scope device program object
   - 多个 member `PrimFunc.attrs["tl.spatial_program"]`
   各自独立发明同一条 phase boundary / shared buffer / global sync 关系。

这条规则的设计目的，是让 `ProgramPhase` 在 fusedmoe、split decode、以及未来更宽的 multi-`T.Kernel`
程序上拥有唯一稳态宿主，而不是让实现期再去判断“当前是不是特殊 case”。

推荐对象设计：

```text
DeviceProgramInfo {
  program_id
  pre_split_anchor_id
  member_funcs: [GlobalVar]
  member_order: [Int]
  phase_order: [PhaseId]
  phase_boundaries: [PhaseBoundaryMaterialization]
}

PhaseBoundaryMaterialization {
  from_phase
  to_phase
  state_bindings
  shared_buffer_ref
  sync_refs
}
```

第一版 pass contract：

1. `CollectDevicePrograms`
   - 在 `SplitHostDevice` 之前运行
   - 从多 `T.Kernel` / 多 device region 结构收集 `pre_split_anchor_id`、member order、program membership
   - 产出 module-scope `IRModule.global_infos["tl.device_programs"]`
2. `SplitHostDevice`
   - 只负责把 composite device program 拆成 member `PrimFunc`
3. `BindSplitDeviceMembers`
   - 把 split 后的 `GlobalVar` 回填到 `DeviceProgramInfo.member_funcs`
   - 不允许在这一步新增 phase truth，只允许做 identity binding
4. `LowerToSpatialProgram`
   - 读取 `DeviceProgramInfo`
   - 填充 `phase_order` 与 `phase_boundaries`
   - 并把 member-local spatial truth 写回各自 `PrimFunc.attrs["tl.spatial_program"]`

**`Task`**

- `kind` 不应成为 workload-specific 名词袋子。更稳的设计是：
  - 第一版固定为 4 个 base family：
    - `transfer`
    - `compute`
    - `collective`
    - `control`
- 只有当新的 family 会改变 legality analysis、candidate generation 或 target mapping 的一级分派逻辑时，
  才允许新增 `Task.kind`
- 一个 `Task` 不等于一个 TT kernel
- `Task` 必须显式绑定：
  - `semantic_updates`
  - `input_channels / output_channels`
  - `layout_bindings / partition_bindings`
  - `resource_intents`
- `program_phase` 指向所属的 `ProgramPhase`

**`Channel`**

- `Channel` 的一级分派不靠 workload noun，而靠 `payload_kind x transport_semantics`
- `payload_kind` 第一版固定为：
  - `data`
  - `index`
  - `state`
  - `control`
- `transport_semantics` 第一版固定为：
  - `point_to_point`
  - `broadcast`
  - `gather`
  - `scatter`
  - `reduce`
- 只有当新的 channel family 无法写成现有 `payload_kind x transport_semantics + traits` 时，
  才允许新增 base family
- `Channel.program_phase` 必须与 producer/consumer task 的 `program_phase` 一致，因此 channel 天然不能跨 phase

**`Layout`**

- `Layout` 只表达“某组 domain/state 在空间程序中的数据组织视图”
- `kind` 第一版固定为：
  - `regular`
  - `packed`
  - `indexed`
- `sharded` 默认不提升为 `Layout.kind`；它通常更像 `WorkPartition + Placement` 的结果
- `domain_bindings` 是 core 字段，不再只在后文 contract 里补充
- 只有当某类 layout 真正改变 legality 与 target realization 的一级分派，才允许新增 `Layout.kind`

**`WorkPartition`**

- `WorkPartition` 只表达“逻辑工作如何被切分”
- `kind` 第一版固定为：
  - `replicated`
  - `blocked`
  - `indexed`
  - `filtered`
- `domain_bindings` 是 core 字段，用来保证 partition 不是脱离 semantic domain 的裸表达式
- 只有当某类 partition 无法写成现有 `kind + traits + domain_bindings` 的组合时，
  才允许新增 `WorkPartition.kind`

**`Placement`**

- 只表达 virtual placement
- `kind` 第一版固定为：
  - `execution`
  - `communication`
  - `phase_boundary`
- `target_ref` 必须显式指出 placement 约束绑定的是 task、channel 还是 phase
- 含义约束：
  - `execution`：task 或 task group 的执行位置 / 邻接 / 同组约束
  - `communication`：channel / route / fanout-fanin 路径的 locality 约束
  - `phase_boundary`：跨 `ProgramPhase` materialization 对共享 buffer / sync 边界的 placement 约束

**`SyncEdge`**

- `kind` 只保留小闭枚举：
  - `dependency`
  - `barrier`
  - `completion`
- `update_or_state_bindings` 必须进入 core schema，而不是只在后文 contract 中出现

**`ResourceIntent`**

- `kind` 第一版固定为：
  - `buffer`
  - `state_residency`
  - `synchronization_support`
  - `phase_boundary_materialization`
- `attachment_ref` 必须显式绑定到 task、channel 或 phase-boundary materialization
- `payload_states` 必须显式指出该 intent 服务的语义对象
- 细粒度的 transport/scratch/index/output/carry 差异不再靠无限增补 `kind`；
  统一通过 typed `traits`、`payload_states` 与后续 target binding 决定

#### Spatial trait system

`traits` 不能是自由字符串集合。更稳的设计是：**每类 spatial object 只允许来自少数固定轴（trait axes）的 typed trait value。**

规则：

1. 每个 trait 必须属于预定义 axis，而不是任意新增标签
2. 同一 object 在同一 axis 上至多取一个值
3. trait 只表达正交修饰，不承担一级 legality/candidate/target 分派
4. semantic truth 必须通过 bindings 表达，不能偷塞进 trait
5. 一旦某个属性会改变一级分派逻辑，就应升级为 `kind`、binding，或 legality fact，而不是继续塞进 trait

第一版固定轴如下。

**`Task.traits`**

- `flow_role`
  - `load`
  - `store`
  - `reduce`
  - `select`
  - `route`
  - `exchange`
  - `combine`
- `ordering`
  - `unordered`
  - `ordered`
  - `completion_required`
- `statefulness`
  - `ephemeral`
  - `persistent`
  - `carrying`

**`Channel.traits`**

- `ordering`
  - `fifo`
  - `ordered`
  - `unordered`
- `topology`
  - `linear`
  - `fanout`
  - `fanin`
  - `multicast`
- `indirection`
  - `direct`
  - `indexed`
  - `routed`
  - `paged`

**`Layout.traits`**

- `distribution`
  - `tiled`
  - `sharded`
  - `grouped`
  - `ragged`
- `indirection`
  - `direct`
  - `indexed`
  - `routed`
  - `paged`

**`WorkPartition.traits`**

- `granularity`
  - `row`
  - `tile`
  - `pair`
  - `chunk`
- `indirection`
  - `direct`
  - `indexed`
  - `expert`
  - `page`
- `selectivity`
  - `dense`
  - `filtered`
  - `selected_subset`

**`Placement.traits`**

- `locality`
  - `colocated`
  - `adjacent`
- `grouping`
  - `row_group`
  - `column_group`

**`SyncEdge.traits`**

- `cause`
  - `producer_consumer`
  - `ordered_update`
  - `carry_completion`
  - `multicast_ready`

**`ResourceIntent.traits`**

- `storage_role`
  - `transport_buffer`
  - `scratch`
  - `index_buffer`
  - `output`
- `lifetime`
  - `ephemeral`
  - `persistent`
  - `phase_boundary`

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

因此，在 spatial 层引入下面四类 planning object：

| 对象 | 作用 | 关键字段 |
|------|------|----------|
| `SpatialLegalityFacts` | analysis 产出的 must-have structure 与 legality constraints | `cut_constraints`, `flow_constraints`, `phase_constraints`, `layout_constraints`, `partition_constraints`, `sync_constraints` |
| `SpatialCandidate` | 一个合法的空间化候选 | `tasks`, `channels`, `layouts`, `work_partitions`, `placements`, `sync_edges`, `resource_intents`, `decision_trace` |
| `SpatialPolicy` | 在合法空间内做选择的 policy 输入 | `fusion_policy`, `partition_preferences`, `layout_preferences`, `placement_preferences`, `sync_preferences`, `reuse_preferences` |
| `SpatialCostModel` | 对候选进行排序或裁剪 | `transfer_cost`, `fanout_cost`, `sync_cost`, `imbalance_cost`, `locality_score`, `persistent_state_pressure` |

`SpatialLegalityFacts` 不应只是“几个摘要列表”。更稳的设计是把 legality 显式对象化成一组 entry family，
使 `AnalyzeSpatialLegality` 的输出可以被 `LowerToSpatialProgram` 机械消费，而不是靠黑盒启发式再理解一遍。

推荐最小 entry family：

```text
CutConstraint {
  updates
  reason
}

FlowConstraint {
  states
  producer_updates
  consumer_updates
  flow_kind
}

PhaseConstraint {
  states_or_updates
  phase_boundary_kind
  materialization_required
}

LayoutConstraint {
  target_ref
  allowed_layout_kinds
  required_traits
}

PartitionConstraint {
  target_ref
  allowed_partition_kinds
  required_traits
}

SyncConstraint {
  source_updates
  target_updates
  required_sync_kind
  required_traits
}
```

对应到 `SpatialLegalityFacts`：

- `cut_constraints : [CutConstraint]`
- `flow_constraints : [FlowConstraint]`
- `phase_constraints : [PhaseConstraint]`
- `layout_constraints : [LayoutConstraint]`
- `partition_constraints : [PartitionConstraint]`
- `sync_constraints : [SyncConstraint]`

这样 legality 输出就从“摘要提示”变成“可直接驱动 candidate 生成的 typed contract”。

规则：

1. `AnalyzeSpatialLegality(SemanticProgram)` 先显式产出 `SpatialLegalityFacts`。
2. `LowerToSpatialProgram` 必须在 `SpatialLegalityFacts` 约束下生成一个或多个 `SpatialCandidate`。
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
   - 节点：`Update`
   - 边：通过 `State` 建出的 def-use、carry、fanout、join 关系
2. **找出必须切开的边界**
   - algorithmic cut / ordered-update boundary
   - carry / cross-update boundary
   - routed / paged / grouped access/domain boundary
   - select / combine / dispatch / merge 这类天然 topology boundary
   - 必须单独同步的 producer-consumer boundary
3. **形成候选 task**
   - 主路径：将同 algorithmic cut、同 domain family、同主要执行职责的一组 `Update`
     聚合为 `Task`
   - 实现可选地先构造导出 `SemanticRegion` 视图做 clustering / debug，但 `Task` 的 truth binding
     仍然必须回到 `semantic_updates`
4. **生成 channel**
   - 所有同一 `ProgramPhase` 内跨 task 流动的 state 都显式变成 `Channel`
5. **生成 layout / partition**
   - 从 domain 及 `AccessMap` 投影出 `Layout` 与 `WorkPartition`
6. **生成 sync**
   - 从 ordered update、carry edge、fanout/fanin、completion requirement 构造 `SyncEdge`
7. **生成 phase-boundary materialization**
   - 所有跨 `ProgramPhase` 的 state 都必须落成 `shared_buffers`
   - 同时生成对应的 `global_sync_after / global_sync_before`
8. **生成 virtual placement**
   - 从 channel topology 和 locality pressure 生成 `Placement`
9. **生成 resource intent**
   - 从 carry / scratch / output / index payload 生成 `ResourceIntent`

因此，`SpatialProgram` 的职责不是“理解算法”，而是把已经冻结的算法语义组织成可执行的空间程序图。

#### Spatial 层的分析事实与 policy 边界

必须由分析确定的事实：

- 哪些 `Update` 必须被切开
- 哪些 state 必须形成显式流动
- 哪些 state 必须跨 `ProgramPhase` 物化
- 哪些边界必须同步
- 哪些 layout / partition family 被语义强制

也就是说，analysis 不只给“有这些约束”，而要给出 typed legality entries。
policy 只能在这些 entries 许可的空间内选择，不能反向修改 legality。

可以由 spatial policy 决定的内容：

- task 是否进一步 fusion / split
- channel 是否选择更细或更粗粒度
- 小闭枚举所允许 family 内的具体 trait 选择与粒度
- virtual placement 偏好
- resource intent 的 reuse 策略

这条边界必须写死：
**analysis 决定 legality 和 must-have structure；policy 只在合法空间内做选择。**

#### Spatial 层与 semantic/TIR 的交互 contract

`SpatialProgram` 的主要输入是 `SemanticProgram`，不是原始 TIR 子树。

它与上层的显式绑定应该体现在：

- `Task.semantic_updates`
- `Channel.payload_states`
- `Layout.domain_bindings`
- `WorkPartition.domain_bindings`
- `SyncEdge.update_or_state_bindings`

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
4. `ProgramPhase` 一旦冻结，后续层不得再把 phase-boundary materialization 回退成普通 channel。

### 5.4 `TT Target IR`

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
| `TTProgram` | `kernels`, `core_groups`, `cb_plan`, `transport_plan`, `semaphore_plan`, `compute_sync_plan`, `dst_layout_plan`, `abi_plan`, `execution_plan` | TT 合约容器 |
| `TTKernel` | `kind`, `traits`, `task_bindings`, `channel_bindings`, `core_group`, `compute_sync_bindings`, `compile_time_args`, `common_runtime_args`, `runtime_args` | TT kernel 合约 |
| `TTCoreGroup` | `placement_bindings`, `partition_bindings`, `physical_cores`, `core_type`, `topology_role` | physical execution group |
| `TTCBPlan` | `resource_class`, `traits`, `channel_bindings`, `resource_bindings`, `capacity`, `binding_scope`, `data_format`, `pack_mode`, `unpack_mode`, `l1_acc_mode` | 最终 CB 规划 |
| `TTTransportPlan` | `kind`, `channel_bindings`, `source_group`, `target_group`, `topology`, `protocol_class`, `descriptor_bindings` | NoC / route / transport 协议规划 |
| `TTSemaphorePlan` | `kind`, `sync_bindings`, `source_group`, `target_group`, `protocol` | program-level semaphore / barrier 规划 |
| `TTComputeSyncPlan` | `kernel_binding`, `state_bindings`, `hazard_scope`, `protocol` | compute kernel 内部微结构同步规划 |
| `TTDstLayoutPlan` | `state_bindings`, `kernel_bindings`, `offset`, `tile_span`, `layout_role`, `fpu_sfpu_ordering`, `pack_format`, `accumulator_mode` | dst/register residency 规划 |
| `TTABIPlan` | `kernel_bindings`, `channel_bindings`, `accessor_bindings`, `work_distribution_bindings`, `compile_time_arg_specs`, `common_runtime_arg_specs`, `runtime_arg_specs`, `accessor_specs`, `launch_specs` | ABI 合约 |
| `TTExecutionPlan` | `kernel_bindings`, `work_distribution_bindings`, `remote_core_descriptors`, `kernel_order` | 执行与 launch 规划 |

#### 对象设计

**`TTKernel.kind` 与 `traits`**

`TTKernel` 也要避免重新长成 target noun bag。推荐和 spatial 层一样，采用**小闭 family + trait axes + explicit bindings**，而不是 `role_set + Map<String, Bool>`。

- `kind` 只保留一级 family：
  - `data_movement`
  - `compute`
  - `collective`
  - `control`
- `traits` 只表达正交 specialization：
  - `io_role`
    - `reader`
    - `writer`
    - `reader_writer`
  - `aggregation`
    - `reduction`
    - `combine`
    - `relay`
  - `distribution`
    - `dispatcher`
    - `sender`
    - `receiver`
  - `specialization`
    - `dedicated`
    - `unified`
- 表达规则：
  - 当前 GEMM/copy 的最小稳定切法仍是三个 kernel：`data_movement{io_role=reader}` / `compute` / `data_movement{io_role=writer}`
  - MoE 等 unified kernel pattern 应表达为 `kind = compute` 或 `collective`，再通过 `traits` 和 typed ABI specialization 区分 gate-up / gate-down / sender 等行为
  - 更细粒度的 kernel specialization 应进入 `TTABIPlan.compile_time_arg_specs` 或 dedicated subplan，不再引入自由字符串 `role_flags`
  - core type mapping（BRISC/NCRISC/TRISC）仍由 `TTCoreGroup.core_type` 决定，不由 `kind` 或 `traits` 隐含

**`TTKernel` / `TTABIPlan` 中的 common-runtime ABI**

TT ABI 第一版必须显式分三层，而不是只在正文里提到、在核心对象里缺席：

1. `compile-time`
   - kernel 创建时固定
   - 例如 specialization define、固定 CB binding、固定 compile arg
2. `common-runtime`
   - launch 时可 patch，但同一 kernel 的所有 core / work item 共享
   - 例如：
     - accessor 的 common runtime payload
     - page table / tensor descriptor 的共享部分
     - shared semaphore id / shared remote descriptor 索引
     - 整个 kernel 共享的 topology / multicast / route descriptor
3. `per-work runtime`
   - 对不同 core、不同 work packet、不同 launch shard 可以变化
   - 例如 tile start/count/stride、per-core role、per-work packet coordinates

因此必须满足：

- `TTKernel` 显式拥有 `common_runtime_args`
- `TTABIPlan` 显式拥有 `common_runtime_arg_specs`
- `MaterializeTTExecutableSpec` 必须把这层稳定物化到 `KernelSpec.common_runtime_args`
- 顶层 `blackhole.common_runtime_args` 只能是 compatibility aggregate view，不能继续承担真源职责

分类规则也必须写死：

- 能在 compile time 固定的，不得降到 common-runtime
- 被所有 core/work 共享、但 launch 时需要 patch 的，必须进入 common-runtime
- 真正随 core/work packet 变化的，才进入 per-work runtime

这条分层的目的，是避免 accessor、shared descriptor、shared route metadata 继续混在
顶层 aggregate attrs 或 per-work runtime args 里。

**`TTCBPlan.resource_class`**

`resource_class` 也收成小闭 family，而不是继续按 case 发散：

- 基础 family：
  - `transport`
  - `scratch`
  - `carry`
  - `output`
- 细化差异通过 `traits` 表达：
  - `payload_shape`
    - `tile`
    - `vector`
    - `index`
  - `lifetime`
    - `ephemeral`
    - `persistent`
  - `visibility`
    - `local`
    - `remote`

**`TTSemaphorePlan.kind`**

- `local`
- `remote`
- `multicast`
- `barrier`

这里必须保持清晰边界：

- `TTSemaphorePlan` 只表达 **program-level** semaphore / barrier / multicast contract
- compute kernel 内部 FPU/SFPU 之间的 micro-architecture 级同步不属于 `TTSemaphorePlan`
- 那部分应单独进入 `TTComputeSyncPlan`

**`TTTransportPlan`**

只靠 `TTExecutionPlan.remote_core_descriptors` 和零散 runtime args，不足以承载 TT-Metal 真实存在的
ring / line / tree / fabric mux / multicast transport 协议。因此第一版必须把 transport/route 提升成一等对象。

- `kind` 第一版固定为：
  - `unicast`
  - `multicast`
  - `tree`
  - `ring`
  - `line`
  - `fabric_mux`
- `channel_bindings` 指出该 transport 服务于哪些 `Channel`
- `source_group / target_group` 指出涉及的 `TTCoreGroup`
- `topology` 表达该 transport 依赖的拓扑角色或 route skeleton
- `protocol_class` 表达 transport 自身的协议类别，而不是散装 runtime arg 名字
- `descriptor_bindings` 负责把该 transport 需要的共享 descriptor 暴露给 `TTABIPlan` / `TTExecutionPlan`

边界规则：

- `TTTransportPlan` 负责 program-level NoC / route / transport 结构
- `TTSemaphorePlan` 负责与之配套的 program-level synchronization object
- `TTExecutionPlan` 负责 launch order、remote descriptor 使用顺序、以及 work distribution
- runtime arg 只承接物化后的 descriptor view，不再反向充当 transport truth

**`TTComputeSyncPlan`**

- 表达 compute kernel 内部执行单元之间的同步 / hazard protocol
- 典型场景：SDPA 中 FPU matmul 完成后，SFPU reduce_max 读取 dst，需要 `wait_on_max` / `post` 这类 protocol 协调
- 它不是 host `CreateSemaphore(...)` 物化的 program object，也不是 task 级 producer-consumer sync
- `protocol` 应表达 protocol class，而不是散装 opcode 名字
- ownership 约束：
  - 它只回答“在既定 dst residency 前提下，需要什么 hazard / ordering protocol”
  - 它不负责决定 state 住在哪个 dst region，也不负责分配 dst offset
  - 那部分只能由 `TTDstLayoutPlan` 决定

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
  - 这是 `TTComputeModel.unit_sync_capabilities + dst_hazard_rules` 在具体 state binding 上的实例化
- `pack_format`：从 dst 输出到 CB 时的精度格式
- `accumulator_mode`：是否启用 accumulator mode（`fp32_dest_acc_en`）
- ownership 约束：
  - 它只负责 residency / alias / offset / tile span / accumulator mode
  - 它不负责证明 hazard 已被满足；hazard 的满足只能由 `TTComputeSyncPlan` 表达
  - 因此 `TTDstLayoutPlan` 与 `TTComputeSyncPlan` 是单向依赖：
    `TTComputeSyncPlan` 依赖既定的 dst residency，但不能反向改写 residency truth

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
| `TTComputeModel` | `execution_units`, `unit_sync_capabilities`, `dst_hazard_rules`, `pack_unpack_capabilities`, `accumulator_capabilities`, `data_format_rules` | compute unit 约束与能力 |

设计要求：

1. `TTTopologyModel` 必须足以表达 row/column core grouping 与合法邻接关系。
2. `TTMemoryModel` 必须足以决定某类 `Channel/ResourceIntent` 是否能落成指定 `CB`。
3. `TTNoCModel + TTSyncModel` 必须足以决定某条 `SyncEdge` 是否能合法 materialize 成 semaphore/barrier/multicast protocol。
4. `TTDstModel` 必须足以决定长生命周期 compute-local state 能否驻留 dst/register。
5. `TTABILimitModel` 必须足以约束 compile-time/runtime arg 组织，而不是让 materialization 阶段再失败。
6. `TTComputeModel` 必须足以约束 compute kernel 内部的 FPU/SFPU 操作交错与 dst 争用。

实现顺序上还必须补一条现实约束：

- Phase C 之前，先把当前散落在实现里的 hardcoded constant 收成一个 minimal `TTHardwareModel` stub
- 第一轮最少要收进：
  - `TTMemoryModel`
    - 例如当前 `PlanBlackholeCB` 里的 `64` 个 CB、`1.5MB` L1 限制、CB ID 分段
  - `TTDstModel`
    - 例如 `fp32_dest_acc_en` / `dst_full_sync_en` 对 dst 容量的约束
  - `TTABILimitModel`
    - 例如 common-runtime / per-core runtime arg 的上限与 patchability
  - `TTComputeModel`
    - 例如 FPU/SFPU / packer hazard 与 accumulator/data-format legality
- `TTTopologyModel / TTNoCModel / TTSyncModel` 第一轮可以先是 skeletal typed object，
  但不能继续把 legality 规则直接散落在 pass 里的字面量上

这样做的目的，不是一次性把硬件模型做满，而是先停止让后续设计继续依赖裸常量和 ad-hoc 判断。

推荐第一版 bootstrap 对象：

```text
TTHardwareModelStub {
  memory: {
    max_cb_count
    max_l1_bytes
    input_cb_id_range
    compute_cb_id_range
    spill_cb_id_range
  }
  dst: {
    fp16_dst_tiles_double_buffered
    fp16_dst_tiles_full_sync
    fp32_dst_tiles_double_buffered
    fp32_dst_tiles_full_sync
  }
  abi: {
    per_core_runtime_arg_limit
    common_runtime_arg_limit
    supports_common_runtime_args
    accessor_common_runtime_support
  }
  compute: {
    execution_units
    supported_sync_protocols
    accumulator_modes
    data_format_rules
  }
}
```

第一版数据来源也写死：

1. `memory` 先直接从当前 `PlanBlackholeCB` 常量与 direct runtime 约束抽取
2. `dst` 先从当前 `ComputeConfig` / TT-Metal 文档里的 `fp32_dest_acc_en`、`dst_full_sync_en` 组合规则抽取
3. `abi` 先从 TT-Metal host/dataflow/compute runtime-arg API 与当前 runtime 限制抽取
4. `compute` 先从 FPU/SFPU/semaphore/dst hazard 相关 kernel API 与现有 SDPA kernel 实现抽取

`TTComputeModel` 的背景：

TT-Metal compute core 内部有两个独立执行单元：FPU（matmul 等）和 SFPU（exp/reduce/recip 等）。它们共享 dst register bank，需要通过 `t6_semaphore` 协调。这不是 reader-compute-writer 之间的 task 级 sync，而是 **compute kernel 内部的 micro-architecture 级 sync**。

`blackhole.acc` 混合语义问题的根因之一就是 FPU matmul output 和 SFPU reduce 争用同一 dst region，如果 `TTDstLayoutPlan` 不知道 FPU/SFPU 的操作顺序，就无法保证 correctness。

`TTComputeModel` 至少需要表达：

- `execution_units`：FPU / SFPU 这类执行单元的类型、独立性和共享资源关系
- `unit_sync_capabilities`：执行单元之间允许的 protocol class，例如 wait/post 或 completion-style coordination
- `dst_hazard_rules`：dst region 被 matmul / sfpu / pack-unpack 占用、读取、复用时的 hazard / alias 规则
- `pack_unpack_capabilities`：CB 到 dst、dst 到 CB 的 pack/unpack family 与合法组合
- `accumulator_capabilities`：`fp32_dest_acc_en`、`l1_acc_mode` 这类累加模式是否支持以及何时合法
- `data_format_rules`：不同 execution unit、CB、dst 之间的数据格式切换规则

这里要避免重新走到“列操作名大全”的方向。
`TTComputeModel` 需要的是 **capability classes + legality rules**，不是一套 workload-specific opcode catalog。

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

- 修改 semantic `Domain / State / Update / AccessMap / UpdateLaw`
- 修改 spatial task / channel / layout / sync 结构
- runtime 侧补协议或反推语义

#### TT target mapping 是如何从 spatial 层构造出来的

`TTProgram` 不是直接从 TIR 或 runtime schema 拼出来的，而是从 `SpatialProgram + TT hardware model` 做 target-specific mapping。

推荐的构造顺序：

1. **kernel clustering**
   - 把 `Task` 按职责和 TT program model 聚成 `TTKernel`
   - 当前最小稳定切法仍是 `reader / compute / writer` 这组三分 specialization
2. **core-group mapping**
   - 把 `Placement` 和 `WorkPartition` 落成 `TTCoreGroup`
   - 得到 logical work 到 physical core 的映射
3. **channel/resource -> CB plan**
   - `Channel + ResourceIntent` 落成 `TTCBPlan`
4. **route/transport synthesis**
   - `Channel + Placement + TTNoCModel` 落成 `TTTransportPlan`
5. **sync -> semaphore plan**
   - `SyncEdge` 落成 `TTSemaphorePlan`
6. **kernel-internal sync synthesis**
   - compute-local hazard / ordering requirement 落成 `TTComputeSyncPlan`
7. **long-lived compute state -> dst plan**
   - carry / reduction carrier / persistent compute-local state 落成 `TTDstLayoutPlan`
8. **program ABI derivation**
   - 从 kernel inputs/outputs/accessors/work distribution 推出 `TTABIPlan`
9. **execution plan derivation**
   - 生成 `kernel_order / remote_core_descriptors / work_distribution`

也就是说，TT 层做的是 **legal target contract synthesis**，不是重新发明空间图。

#### TT 层的分析事实与 policy 边界

必须由 hardware-aware analysis 决定的事实：

- 哪些 mapping 违反 topology / NoC / semaphore / dst / L1 约束
- 哪些 channel 必须 materialize 成 transport buffer
- 哪些 channel 必须落成显式 `TTTransportPlan`
- 哪些 carry 可以 register-resident，哪些必须 round-trip
- 哪些 sync edge 必须落成 semaphore / barrier / multicast protocol

可以由 target policy 决定的内容：

- 具体 kernel clustering 粒度
- CB class / page size / reuse 偏好
- transport family 已固定后的具体 route / descriptor packing 偏好
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
   - `blackhole.common_runtime_args`
   - `blackhole.accessors`
   - `blackhole.cb_configs`
   - `blackhole.semaphore_plan`
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

#### `KernelSpec` / `ExecutableSpec` 的 ABI ownership

为了避免 TT contract 在 materialization 之后又长出第二份真相，Phase C 之后应遵守下面这条 ownership：

- `ExecutableSpec`
  - 只拥有 program-shared materialization metadata
  - 以及 `KernelSpec[]` 这个 per-kernel materialized view 容器
- `KernelSpec`
  - 拥有单 kernel 的 compile-time / common-runtime / per-work ABI
  - 拥有单 kernel 的 accessor / semaphore binding / remote-core descriptor
  - 拥有单 kernel 的 launch / compute config

因此：

- 顶层 `ExecutableSpec.runtime_args/common_runtime_args` 只能作为 legacy aggregate view 过渡存在
- 一旦 `KernelSpec` 已完整承接 per-kernel ABI，顶层 aggregate view 必须降为 compatibility-only，最终删除
- runtime 与 codegen 应优先消费 `KernelSpec` 的 per-kernel schema，而不是反向从 aggregate view 再推导一次

#### 当前仍保留的 compatibility shim

在 `MaterializeTTExecutableSpec` 完全取代 legacy writer 之前，当前实现仍允许存在少数 compatibility shim，但必须在设计上被明确标成过渡层，而不是稳态协议：

- 从 `segment_plan[*]` 聚合顶层 `blackhole.runtime_args/common_runtime_args`
- 从 segment/kernel schema 回写顶层 `blackhole.accessors`
- 用 legacy `blackhole.*` attrs 参与 device-kernel 检测
- runtime 中对 buffer role / work packet 的 positional 或 fallback 解释

这些 shim 的存在只用于兼容当前 direct runtime / codegen 消费面。Phase C cutover 之后，它们必须满足：

1. 不能再生成新的 truth，只能读取或验证已物化结果。
2. 不能再引入 fallback-based protocol patching。
3. 一旦对应的 `KernelSpec / ExecutableSpec` 稳态字段已齐备，就必须删除。

#### host logical layout / tilize-untilize / transpose 的物化归属

host logical tensor layout 不是 runtime helper 私下知道的经验知识，而是 materialization contract 的一部分。
第一版必须明确：

1. host-visible logical layout、device-visible physical layout、以及 upload/readback transform
   属于 **program-shared buffer materialization metadata**
2. 如果某个输入/输出需要：
   - `transpose`
   - `tilize`
   - `untilize`
   - 其他 host/device layout conversion
   则这必须由 `MaterializeTTExecutableSpec` 从 `TTProgram + accessor/buffer materialization truth`
   物化出来，而不是由 runtime 临时猜测
3. runtime helper 可以执行这些转换，但不能决定：
   - 是否需要转
   - 先后顺序
   - 哪个 tensor / buffer binding 需要哪种转换
4. 这类 layout conversion 不得再散落在 `BlackholeModule` 的 ad-hoc kernel-specific 分支里；
   稳态下应成为 `ExecutableSpec` program-shared materialization metadata 的一部分

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
| `PrimFunc.attrs["tl.semantic_seeds"]` | pre-lift `SemanticSeed[]` | compiler-internal semantic capture 输入通道；只服务 semantic lift，不是真源 |
| `PrimFunc.attrs["tl.semantic_program"]` | `SemanticProgram` | 每个 device function 的语义真源 |
| `PrimFunc.attrs["tl.spatial_program"]` | member-local `SpatialProgram` | 每个 device function 的 local spatial truth |
| `PrimFunc.attrs["tl.tt_program"]` | `TTProgram` | 每个 device function 的 TT target contract 真源 |
| `IRModule.global_infos["tl.device_programs"]` | module-scope device program / cross-function `ProgramPhase` truth | multi-`T.Kernel` / multi-device-function 程序的稳定宿主 |
| `IRModule.global_infos` | `hardware_model`、`topology`、全局 mapping policy | 真正跨函数共享的 module-scope 对象 |
| materialized `blackhole.*` attrs | `segment_plan`、`runtime_args`、`common_runtime_args`、`accessors`、`cb_configs`、`semaphore_plan`、`core_plan` | 仅用于兼容现有 codegen/runtime 消费面 |

这意味着：

1. 不建议把三层 IR 都编码成 `Array<Any> / Map<String, Any>` 风格 attrs。
2. 也不建议把 semantic/spatial/tt 层直接做成 `BaseFunc`；它们不是 executable function。
3. `attrs` 在这里的职责只是“挂载 typed IR object”，不是“编码协议本体”。
4. `IRModule.global_infos` 只放真正 module-scope 的共享对象；`tl.device_programs`
   是所有 cross-function phase/program truth 的唯一稳定宿主，单 `PrimFunc` 也只是其中的退化情况。
5. 每个 kernel 的 local companion IR 仍以 `PrimFunc` 为粒度挂载。
6. `tl.semantic_seeds` 只允许在 semantic lift 之前存在；一旦 `SemanticProgram` 建立成功，就不得继续充当真源。

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
   - `ExecutableSpec` 与 `blackhole.segment_plan/runtime_args/common_runtime_args/accessors/cb_configs/semaphore_plan/core_plan` 都来自这一层

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
- `Domain.axes -> Array<Var>`
- `Domain.constraints / predicate -> PrimExpr`
- `AccessMap.coord_exprs / validity -> PrimExpr`
- `Update.guard -> PrimExpr`
- `Update.anchor -> TIRAnchor`

这些直接引用负责让 semantic object 本身保持可读、可验证；对应的字段级 rebind / invalidation
则统一通过 `TIRValueBinding` 索引维护，而不是分散塞回各个对象内部。

`SpatialProgram` 主要消费 `SemanticProgram`，只保留必要的 `PrimExpr` 或 `GlobalVar` 级引用；它不再依赖原始 TIR 子树结构。
`TTProgram` 原则上不再直接依赖 TIR 结构，只消费 `SpatialProgram + hardware model`。

#### 5.6.2 真源规则

1. `LiftStatefulSemanticIR` 完成之后，TIR 不再是算法语义真源。
2. `LowerToSpatialProgram` 完成之后，task/channel/layout/sync 结构只能从 `SpatialProgram` 读取。
3. `LowerSpatialProgramToTTTarget` 完成之后，TT resource / ABI / execution contract 只能从 `TTProgram` 读取。
4. `codegen_blackhole` 与 `rt_mod_blackhole` 只消费物化后的 target contract，不再回头理解 semantic/spatial 语义。

#### 5.6.3 生命周期与失效规则

1. `LiftStatefulSemanticIR` 必须运行在固定 canonicalization 点之后。
2. 一旦某个 `PrimFunc` 挂上 `tl.semantic_program`，companion IR 默认进入 hard-freeze 管理；
   后续 pass 只能显式属于 `preserve`、`typed_rebind` 或 `invalidate` 三类合同之一。
3. 如果后续 pass 必须实质性改动相关 TIR 结构，就必须显式失效并删除：
   - `tl.semantic_structure`
   - `tl.semantic_witnesses`
   - `tl.semantic_program`
   - `tl.spatial_program`
   - `tl.tt_program`
   然后重跑 recovery/lowering。
4. `blackhole.*` materialized attrs 不是上游 IR 的真源；它们可以被整体重建，而不允许被下游 patch 成第二真源。

当前已收口的 semantic companion window 直接固定为：

```text
ProjectSemanticSeeds
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> ValidateSemanticRefinement
  -> LowerToSpatialProgram
```

这条窗口的含义是：

1. `LiftStatefulSemanticIR` 与 `LowerToSpatialProgram` 之间默认不允许插入 TIR mutator
2. 如果 pipeline 中间必须插入新的 unsafe pass，则 builder 必须显式执行：
   - companion invalidation
   - 重新跑 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> ValidateStatefulSemanticIR`
3. semantic companion 生命周期现在由 machine-checkable contract 约束，而不是靠实现约定维持

**Safe TIR mutations（不触发 companion IR 失效的改动类型）**：

当前 safe 集仍然必须刻意收得很小。原因不是这些改动一定会改变算法语义，而是
`SemanticProgram` 直接持有 `Buffer / Var / PrimExpr / TIRAnchor`，并且 `TIRValueBinding`
还维护着字段级 typed binding 索引。只要改动可能改变这些原子的身份、canonical form、
field-to-atom 绑定关系或 anchor 对应关系，就可能把 companion IR 留成“语义上等价、
引用上失效”的悬挂状态。

因此，当前只把下面这些改动视为 `preserve` 级 safe：

- 只改 `PrimFunc.attrs` / `IRModule.global_infos` 的 metadata 更新
- 只增加调试、诊断、打印辅助信息，不改任何被 companion IR 直接引用的 TIR 原子
- 纯分析 pass / verifier pass

以下 TIR mutation 默认仍为 `unsafe`，必须触发失效并重建：

- 默认的 constant folding、arithmetic simplification、常量传播
- 默认的 dead code elimination
- 默认的 `PrimExpr` 等价替换，即使它“按语义等价”
- 改变 buffer scope（如 `local.fragment` -> `blackhole.acc`）
- 改变 loop structure（fusion / split / reorder / unroll）
- 增加或删除 `Allocate` / `DeclBuffer`
- 改变 `BufferStore` / `BufferLoad` 的 access pattern
- 改变 launch axes 或 thread_extent 结构
- 改变 block structure（增删 `BlockRealize`）

已经固定的执行规则是：

1. post-semantic-lift pass 不能再默认被视为 safe
2. audited-safe pass 必须显式声明 `preserve` 或 `typed_rebind`
3. pipeline builder 依据声明插入 invalidation + re-lift，或执行正式 rebind 入口

#### 5.6.3.1 固定 semantic lift canonicalization 点

“固定 canonicalization 点”不能只停留在原则描述上。当前 contract 的关键不是一串易变的 pass 名，
而是下面这些稳定输入条件：

- semantic lift 的输入是 **generic canonicalized device program**
- semantic lift 已经可以稳定读取：
  - `tl.device_programs`
  - `tl.semantic_seeds`
  - `blackhole.work_decomposition`
  - `blackhole.fragment_regions`
  - `blackhole.pipeline_stages`
- 如果程序包含多个 `T.Kernel` / 多 device region，则 cross-function membership / order
  已经先收进 module-scope device program registry，而不是等 semantic/spatial 层再临时发明
- 此时尚未进入：
  - `LowerBlackholeOps`
  - `PlanBlackholeCB`
  - `AssignBlackholeCores`
  - `MaterializeTTExecutableSpec`

如果未来主链继续调整，这条规则仍应保持下面这些等价不变量：

1. copy/resource/device-program 边界已稳定
2. semantic lift 所需的 pre-lift evidence 已固定并可验证
3. companion IR 仍发生在 TT target contract 生成与物化之前

也就是说，semantic lift 的稳定输入不是 raw TIR，也不是已经开始发明 TT contract 的半物化程序，
而是“已完成 generic device canonicalization、但尚未进入 spatial/TT materialization 的 device program”。

#### 5.6.3.2 `TIRValueBinding` / `TIRAnchor` typed rebind contract

这套 rebind contract 现在已经不是未来计划，而是当前 `Phase A` 的正式合同之一。
任何 post-semantic-lift audited-safe pass 如果想保留 companion IR，都必须提供：

1. pass 必须产出完整的 typed rebind 结果：
   - `TIRValueBinding` 从旧原子到新原子的映射
   - `TIRAnchor` 从旧结构锚点到新结构锚点的映射
2. rebind 必须按 object identity / typed field-binding 进行，禁止用：
   - buffer 名字
   - 位置序号
   - “看起来像同一个表达式”的启发式匹配
3. 如果 pass 不能为全部受影响 binding / anchor 产出完整 rebind 结果，
   该 pass 必须标成 `unsafe` 并触发 companion IR 整体失效
4. `ValidateStatefulSemanticIR` 必须把 rebind 后的一致性检查作为 safe pass 的后置条件，
   而不是只检查语义大致等价

当前建议的正式入口是：

- `TypedRebindBlackholeCompanionPrograms`

它的职责不是“修补一部分 attrs”，而是对 structure / witness / semantic core 做受控重绑，
并刷新 freeze contract。

#### 5.6.4 Materialized Attr Ownership 与 Cutover

为了落实“`TTProgram` 是 target contract 真源、`blackhole.*` attrs 只是物化结果”，
必须把 materialized attrs 的 writer ownership 写死。

| Attr | 当前临时 writer | 稳态唯一 writer | Cutover 规则 |
|------|-----------------|----------------|--------------|
| `blackhole.segment_plan` | `SplitBlackholeKernel`、`LowerBlackholeOps`、`rt_mod_blackhole` 的兼容路径 | `MaterializeTTExecutableSpec` | pre-Phase C 可继续作为兼容产物存在，但 semantic/spatial/TT companion IR pass 不得把它当真源 |
| `blackhole.runtime_args` | `LowerBlackholeOps`、`rt_mod_blackhole` 的兼容聚合路径 | `MaterializeTTExecutableSpec` | 一旦 `TTABIPlan` 已能物化该 schema，所有前置 writer 必须删除 |
| `blackhole.common_runtime_args` | `LowerBlackholeOps`、`rt_mod_blackhole` 的兼容聚合路径 | `MaterializeTTExecutableSpec` | 一旦 `TTABIPlan` 已能物化共享 common-runtime ABI，所有前置 writer 必须删除 |
| `blackhole.accessors` | `LowerBlackholeOps`、`rt_mod_blackhole` 的兼容聚合路径 | `MaterializeTTExecutableSpec` | 一旦 `TTABIPlan.accessor_specs` 已能稳定物化该 schema，所有顶层聚合 writer 必须删除 |
| `blackhole.cb_configs` | `PlanBlackholeCB` | `MaterializeTTExecutableSpec` | `PlanBlackholeCB` 后续只能作为 `TTCBPlan` 构造/验证 helper，不再直接写最终 attrs |
| `blackhole.semaphore_plan` | 测试注入 / 过渡 synchronization helper | `MaterializeTTExecutableSpec` | 一旦 `TTSemaphorePlan` 已稳定物化该 schema，所有测试注入或过渡 writer 都降为 validation-only helper |
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

#### 5.6.4.1 compatibility shim 的 deletion gates

compatibility shim 的退出不能靠“大家都知道差不多该删了”。每一类 legacy writer / fallback
都必须满足显式 deletion gate，才能继续往后推进：

1. 上游 `TTProgram` / `TTABIPlan` / `TTExecutionPlan` 已存在与该 shim 对应的稳态字段
2. `MaterializeTTExecutableSpec` 已能从稳态字段完整物化目标 schema
3. `ValidateTTTargetProgram` 或 materialization verifier 已能在缺字段时 fail-fast
4. `codegen_blackhole` / `rt_mod_blackhole` / `BlackholeModule` 已切到只读消费稳态字段或其物化结果，
   不再依赖 fallback 聚合 / positional 推断
5. 对应 compatibility 路径已从测试基线中移除，CI 不再需要它维持行为

一旦 1-5 全部满足：

- 对应 legacy writer / reader / fallback 必须在同一轮 cutover 中删除
- 不允许再保留“以防万一”的双写或双读路径
- 不允许 materialization 之后再从 aggregate attrs 反向推导稳态 schema

### 5.7 Anchor / AtomicEffect / SemanticRegion 的恢复辅助规则

这一节描述的不是 semantic 层的 core schema，而是 semantic recovery 为了稳定构造
`Domain / State / Update / AccessMap / UpdateLaw` 所需要的辅助对象与规则。

`anchor`、`AtomicEffect` 和 `SemanticRegion` 不能靠 workload 特判，也不能靠样例名猜。
它们必须来自一套对复杂计算普遍成立的结构恢复规则。

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

推荐在 `LiftStatefulSemanticIR` 之前先建立一个统一的内部恢复表示：`AtomicEffect`。

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
| `access_signature` | 读写索引映射的恢复签名；后续投影为 `AccessMap` |
| `temporal_role` | `pre_loop / in_loop / post_loop / loop_carried` |
| `scope_class` | `ephemeral / local_state / cross_update / materialized` |

recovery 阶段会先把 `TIRAnchor` 附着到这些 `AtomicEffect` 节点上，随后再把一组相关 anchor
投影到最终的 `Update` 或 `UpdateLaw` payload 上。也就是说：

- `AtomicEffect` 是恢复入口
- `Update / UpdateLaw` 才是 semantic 真源
- `TIRAnchor` 只是两者之间的结构桥

#### 5.7.3 `domain_signature` 的通用来源

`domain_signature` 必须从结构事实中恢复，而不是从 op 名推断：

- enclosing loop vars
- loop `min / extent`
- launch axes 与 work decomposition
- predicate
- indirect/index remapping expr
- segmented/grouped-ragged 的 index/predicate 结构

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

如果某类 effect 无法仅靠结构区分，就要求前端/早期 IR 提供最小显式语义补充，而不是由后段猜。

#### 5.7.5 `SemanticRegion` 的切分不变量

`SemanticRegion` 不再是 effect graph 上的准真源对象，而是**从 `Update` 图导出的非真源视图**。

推荐定义为：

- 在同一 algorithmic cut 内
- 共享相近 domain family
- 共享主要 semantic role / `UpdateLaw` 家族
- 且只通过 local ephemeral state 彼此连接

的一组 `Update` 引用及其导出摘要。

它的作用只有三个：

1. debug / 可视化
2. recovery 诊断
3. spatial clustering 的辅助视图

它**不**承担：

- 算法语义真源
- 独立于 `Update` 的 schema 身份
- `Task` 的最终 truth binding

必须切 region 的情况：

1. 主要 `Domain` family 改变
2. 主要 `UpdateLaw` 家族或 semantic role 改变
3. 出现 carry、cross-update 或 ordered-update 边
4. 中间值不再是 region-internal ephemeral state
5. 需要独立 spatialization 决策
6. 遇到 target-visible synchronization / transport boundary

可以合并在同一个 region 的前提：

1. 同一 domain family
2. 同一 algorithmic cut
3. 同一主要 `UpdateLaw` 家族或 semantic role
4. 中间边都是 local ephemeral state
5. 对外接口不变
6. 不需要单独 task/channel/sync 决策

这套切分规则是 workload-neutral 的，不依赖 attention、MoE、paged decode 或 recurrence 的样例名。
更重要的是：`SemanticRegion` 必须始终能被看成“对一组 `Update` 的投影视图”，而不是另一套并行 schema。

## 6. 层间接口与不变量

### 6.1 真源规则

1. 算法语义只存在于 `Stateful Semantic IR`
2. 空间组织只存在于 `Spatial Program IR`
3. TT 资源与 ABI 只存在于 `TT Target IR`
4. `ExecutableSpec` 由 `TT Target IR` 物化，不是第二真源

### 6.2 交接契约

| From | To | 必须交付的契约 | 允许做的决策 | 明确禁止 |
|------|----|----------------|--------------|----------|
| `Semantic Recovery` | `Stateful Semantic IR` | 恢复出的 `Domain / State / Update / AccessMap / UpdateLaw` 事实 | 对象化与冻结 | 泄漏 TT resource 事实 |
| `Stateful Semantic IR` | `Spatial Program IR` | 冻结后的算法语义 | 构造 task/channel/layout/sync/work | 改变语义含义 |
| `Spatial Program IR` | `TT Target IR` | 冻结后的空间结构 | TT mapping、resource planning、ABI 定义 | 发明新的 task graph 或 semantic update/access law |
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

**注意**：本节中出现的 `Update`、`State`、`Domain` 等对象名字纯属叙述用例，不是 schema 约束。
semantic recovery 不应匹配这些名字，而应基于 structural pattern（读写集、降维结构、index 依赖关系、cross-iteration def-use）来恢复。
如果某个实现中出现了 `if update.name == "qk_scores_update"` 这类逻辑，那就违反了通用性原则。

本节里出现的 `qk_scores_update`、`combine_output_update`、`decode_phase` 等名字都只是叙述用例，
不是 schema key，不是 matcher token，也不是实现期允许硬编码识别的协议字段。
实现必须通过 `Domain / State / Update / AccessMap / UpdateLaw` 结构来恢复和验证，
不能通过匹配这些示例名字来识别 workload。

### 7.1 状态化 reduction-update：attention / online softmax 类

代表：

- `examples/flash_attention/`
- `examples/online_softmax/`
- `examples/attention_sink/`

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `q_tile_domain`
  - `kv_chunk_domain`
  - 可选 `constraints/predicate`（causal）
  - 可选 grouped-head remap 约束
- `State`
  - `scores_stats : Tuple<Scalar, Scalar>`（carry）
  - `acc_o : Tensor`（carry）
  - `logsum : Scalar`（output 或 cross-update derived result）
- `Update`
  - `qk_scores_update`
    - `law = ReduceLaw(max)`
  - `softmax_normalize_update`
    - `law = MapLaw`
  - `online_attention_update`
    - `law = RecurrenceLaw(normalized, stable_order_required)`
    - `reads/writes` 涉及 `acc_o` 与 `scores_stats`

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
  - `selector_axis`
- `State`
  - `logits_state : Tensor`
  - `selected_index : IndexTensor`
  - `selected_value : Tensor`
- `Update`
  - `row_select_update`
    - `law = SelectLaw(topk, selector_axis=k, output_contract=index_value_pair)`
    - 包含 iterative mask/select 的 ordered update 语义

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
  - `constraints/predicate` 覆盖 ragged token block
- `State`
  - `expert_index : IndexTensor`
  - `expert_weight : Tensor`
  - `up_logits : Tensor`（materialized intermediate）
  - `output_state : Tensor`
- `Update`
  - `route_gate_up_update`
    - `reads/writes` 使用 `segmented/indirect AccessMap`
  - `expert_down_update`
    - 消费 `up_logits`
  - `combine_output_update`
    - 对 `output_state` 做 weighted combine

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
- expert-kernel ABI（`TTKernel.kind + traits` 表达 unified kernel family，gate_up / down 等 specialization 进入 typed ABI）
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
  - `constraints/predicate` 覆盖 `cache_seqlens`
- `State`
  - `block_index : IndexTensor`
  - `page_index : IndexTensor`
  - `cache_bound : Scalar`
  - `partial_stats : Tuple<Scalar, Tensor>`（carry / merge payload）
- `Update`
  - `page_lookup_update`
    - `AccessMap.traits = {indirect_read, paged}`
  - `decode_partial_update`
    - `law = RecurrenceLaw(normalized)`
  - `merge_split_update`
    - 合并 partial outputs / partial stats

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
- `cross-update carry state`
- recurrence update / decay
- chunk-local compute 与 chunk-writeback

**`Stateful Semantic IR` 应表达**

- `Domain`
  - `chunk_domain`
  - `intra_chunk_step_domain`
  - `step_axis`
- `State`
  - `chunk_state : Tensor | Tuple`（cross-update carry）
  - `decay_state : Tensor`
  - `dt_state : Tensor`
- `Update`
  - `decay_compute_update`
    - `law = MapLaw`
  - `chunk_state_update`
    - `law = RecurrenceLaw(ordered_dims=[step_axis])`
    - `step_anchor` 对应 outer-product + decay 累加

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
  -> LiftStatefulSemanticIR
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

这一节不再重复分阶段实施 checklist，而只保留当前执行状态与剩余迁移方向。
每个阶段的具体实施细节，统一以下面分文档为准：

- `stage4_stage0_guardrails.md`
- `stage4_phase_a_semantic_ir.md`
- `stage4_phase_b_spatial_ir.md`
- `stage4_phase_c_tt_target_ir.md`

**Stage 0：已完成**

当前已经落实：

- `tl.device_programs`
- `tl.semantic_seeds`
- post-lift hard-freeze / invalidation 基础护栏

**Phase A：已完成**

当前已经落实：

- `SemanticProgram`
- `Domain / State / Update / AccessMap / UpdateLaw / SemanticSupplement`
- semantic witness algebra
- typed vocab / decoder / payload / rule modules
- `ValidateStatefulSemanticIR`
- `ValidateSemanticRefinement`
- internal state/effect graph
- `preserve / typed_rebind / invalidate` lifecycle contract

这意味着 `Phase A` 已经从“实施阶段”转成了：

- `Phase B` 的输入边界
- semantic truth 的稳定真源

**Phase B：当前主实施阶段**

当前剩余主线是把冻结后的 semantic truth 组织成 `SpatialProgram`，重点包括：

- `ProgramPhase / Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
- simple-workload canonical fast-path
- 至少一个 non-trivial multi-phase spatial gate
- 明确 `Phase B` 不能再回头发明 semantic truth

**Phase C：下一阶段**

`Phase B` 之后，继续推进：

- `TTProgram`
- `TTHardwareModel`
- `MaterializeTTExecutableSpec`
- compatibility writer / reader / fallback deletion
- `flash-attn` correctness payoff
- 在新主链下做更宽 family expansion

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

1. 保持总设计、阶段文档和代码实现三者边界一致
2. 当前主线执行 Phase B，把 task/channel/layout/sync 从 monolithic lowering 中拆出去
3. 之后执行 Phase C，让 TT 资源与 ABI planning 成为一等 target contract
4. 最后在新主链下继续做 family expansion，而不是回退去补 semantic matcher

### 10.2 验收标准

只有当下面条件同时成立，才能认为这次重设计真正落地：

1. semantic truth 在 TT target planning 之前冻结
2. spatial structure 在 TT resource / ABI planning 之前冻结
3. runtime/codegen 不再反推缺失语义
4. `PrimFunc + typed companion IR + module global infos + materialized attrs` 的承载边界固定，并在实现中被遵守
5. `Domain / State / Update / AccessMap / UpdateLaw` 已成为第一层真源；`anchor / SemanticRegion`
   只作为通用结构恢复与投影视图 helper，而不是另一套 schema 核心
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
