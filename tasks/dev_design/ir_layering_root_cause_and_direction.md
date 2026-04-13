# IR Layering 根因分析与整改方向

## 基本信息

- **文档角色**: 诊断当前 Blackhole 后端 layered IR 出现的共同 blocker 模式，
  并给出后续整改方向
- **当前状态**: `2026-04-10` 新活动设计文档
- **任务链位置**: `Task 0` 的根因诊断输入；解释为什么必须切到 companion 主链
- **定位**: 不是第二份总体设计；是对
  `final_blackhole_backend_redesign.md` 里 layered IR discipline
  的一次具体落地审计，产出的是“接下来应往哪改”的决策
- **适用范围**:
  - 诊断对象：当前代码基线
    （`Stateful Semantic IR -> Spatial Program IR -> TT Target IR -> ExecutableSpec`）
  - 结论去向：新的长期总设计
    （`Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`）
- **非目标**:
  - 不在本文档里敲定最终的 C++ class 层次
  - 不替代 `spatial_dataflow_program_model.md` 已经圈定的 feature 边界
  - 不重开 `Phase A / Phase B / Phase C` 的 stage 判定

## 1. 当前症状

最近几轮推进（`flash-attn` `Phase C2`、`blackhole.acc` fragment live-form
blocker、fragment buffer flow contract 提升）持续命中同一组症状：

1. **“上游信息不足，下游只能猜”**
   - `AnalyzeSemanticStructure` 一度需要按 op family 名字匹配
     （`gemm_py` 等）来决定协议
   - `LowerBlackholeOps` 持续依赖结构化 pattern recovery
     来重构 row-reduction、row-broadcast、fragment fill/cast、scalar FMA、
     scalar exp2、local→CB slice 等“其实上游就有的算子”
   - `blackhole.acc` 类 fragment 的 live form
     （thread-distributed / materialized / cb-live / dst-live）
     没有 owner-side contract，
     `LowerBlackholeOps + codegen` 就按 logical extent / 默认 lane-0 假设
2. **“硬编码类型越加越多”**
   - `src/transform/common/companion_base.h` 已经累积了 6 个
     stringly-typed 命名空间：
     `schema_key::`（~90 条常量）
     `fragment_flow::`（state / stream / republish / logical_tile /
     write / compute_consume / transport_consume / reference）
     `fragment_materialization::`（`intermediate_fragment_merge` /
     `republished_logical_tile` / `intermediate_buffer` /
     `republished_buffer` / `tile_nfaces_materialization` /
     `fragment_delta` / `consumer_input` / `fragment_add` /
     `direct_write` / `dst_cb_binary_pack` / `tiled_cb_republish`）
     `fragment_live_form::`（`tiled_cb` / `local_fragment`）
     `fragment_layout::`（`linear` / `grouped_rows` / `row_state` /
     `thread_distributed`）
     `spatial_contract::`
   - `FragmentMaterializationInfo` 当前形态：
     ```cpp
     struct FragmentMaterializationInfo {
       Buffer target_buffer;
       ffi::String materialization_kind;
       ffi::String bridge_kind;
       ffi::String value_role;
       ffi::String merge_kind;
       ffi::String execution_protocol;
       ffi::String result_live_form;
     };
     ```
     本质上是“六个 string 的并列袋子”；
     每新增一个 op / 一组执行模式，就要再加枚举值
   - 每一个 `SpatialProgram` 节点（Task / Channel / Placement /
     SyncEdge / ResourceIntent / ProgramPhase）都是
     `{ String kind; Array<String> traits; Map<String, Any> payload; }` 的
     半 stringly-typed 形态，
     `SpatialCapabilityModel` 持有 9 个 `Array<String>`
     的 supported kinds 列表

3. **“每新增一个算子就要补一条 pattern recovery”**
   - `lower_blackhole_ops.cc` 已经 7461 行、持有 17 个 `Match*`
     function；当前 owner 侧再加一条 typed contract，下游几乎总是
     要再加一条相应的 pattern matcher 或一条新 enum
   - 这种扩张模式在 topk / fusedmoe / paged decode / chunk recurrence
     等下一批 family 面前是不可持续的

这些症状出现在不同 pass、不同 phase、不同数据结构上，但根因是同一条。

## 2. 根因定位

### 2.1 根因 A: 信息先被销毁，再被重建

主链里存在**两次明确的信息销毁点**，每次销毁之后都由一套
“seed + recover”机制补回来：

**销毁点 1：Fragment layout 在 `OptimizeForTarget / SplitHostDevice` 被丢弃**

- TileLang 已经有一套成熟的 `Fragment` layout 抽象：
  一个 fragment 是 `(logical shape, distribution layout, storage scope)`
  的可组合 layout function；`ReplicateExtent` / `ThreadExtent` 等
  已经把“每个线程具体持有哪几个元素”写成可计算的形式
- 但当前 Python 主链在 `OptimizeForTarget -> SplitHostDevice` 之后
  就把 `layout_map / tl.Fragment` truth 丢掉了
- 于是新增了一个补救机制：把这批 truth 先投影成
  `tl.fragment_layout_seeds`，
  再由 `AnalyzeBlackholeFragmentRegions` 重新 materialize 成 typed
  `fragment_layout_contracts`（见 `progress.md` 相关记录）
- 但这套 seeds → contract 只覆盖了 logical layout 和 per-lane local extent，
  原本 `Fragment` 就已经能回答的
  “thread-distributed vs materialized / lane 映射 / replicate”等信息，
  在 recover 过程中又丢了一部分
- `blackhole.acc` blocker 就是这个后果：
  `32x32` 逻辑 fragment 在 device-side 只剩 per-lane `8` 的 physical
  extent，下游无法分辨“这是一条 per-lane slice”还是
  “已经 materialized 的线性 logical fragment”

**销毁点 2：tile 算子在 `LowerTileOp` 被拆成 scalar loop，之后靠 `Match*` 反推**

- `row_reduce / row_broadcast / fragment_fill / fragment_cast /
  scalar_fma / exp2_broadcast / local→CB slice` 等算子，
  在进入 Blackhole 专属 pass 之前就已经被 lower 成
  裸 for-loop + scalar load/store 的形态
- 于是 `LowerBlackholeOps` 里出现了 17 个 `Match*` 函数：
  `MatchDirectRowReduction / MatchGroupedRowReduction /
  MatchAllocatedRowReduction / MatchDirectRowBroadcast /
  MatchScalarFmaStore / MatchGroupedScalarFmaLoop /
  MatchExp2RowBroadcastAffine / MatchScalarExp2AffineStore / ...`
- 它们做的本质上都是“把上游原本已经 typed 过的算子再从 loop 结构里
  识别回来”，而且每新增一个 family 都会再加一条
- 这是一种**结构化**而非**名字化**的 semantic recovery，
  但仍然违反 CLAUDE.md 里已经写死的原则：
  “所需信息优先从 IR 分析；缺失就扩 IR/DSL，不要让后段猜”

这两次销毁点共同形成一条对所有后端都有害的模式：

```text
typed truth
  → (销毁)
  → untyped form + seeds/attrs
  → 下游按“可能是什么”重建 typed truth
  → 下游再根据重建结果产生新的 typed contract
```

每一次新增 case，这条链上的 **每一层**都要再打补丁。

### 2.2 根因 B: IR 在建模“名字化的类别”，不在建模“可计算的结构”

当前 IR（尤其是 companion schema 与 `FragmentMaterializationInfo` /
`SpatialProgram` 节点 payload）的扩张方向是：
> “再加一个 enum string，表示又一种情况。”

而不是：
> “用已经存在的结构（layout、storage、distribution、topology、edge）
> 把这种情况**算出来**。”

举一个有代表性的例子：**fragment live form**。

当前 layered IR 想补的字段是：

- `live_form_kind` ∈ {`thread_distributed_fragment`,
  `materialized_local_fragment`, `cb_live_tile`, `dst_live_fragment`}
- `execution_topology_kind`
- `physical_local_extent`

但这三件事其实都不是独立语义：

- 一个 buffer 是不是 `thread_distributed_fragment`，
  等价于问“它的 `Fragment` layout 是否包含 `ThreadRange`
  并且 storage scope 是 register/fragment”
- 一个 buffer 是不是 `cb_live_tile`，
  等价于问“它的 storage 被绑定到 CB 资源 + `logical_tile` granule”
- `physical_local_extent` 等价于
  `Fragment.PerThreadElements()` / `local_shape`

换言之：**live form 是 `(layout, storage, distribution)` triple 的派生结论，
不是一个独立的枚举量**。一旦我们把它变成独立的字符串 enum，就必须：

1. 手动枚举所有物理形态（现在 4 种，后面 topk / paged_decode 会再加 N 种）
2. 每种形态都要各自写 flow contract 条目
3. 下游每次收到 contract 都要再做一次 “enum → 结构”的 inverse mapping

这正是根因 A 的镜像：**当 IR 不承载结构，就必须承载越来越多的名字**。

`materialization_kind / bridge_kind / merge_kind / execution_protocol /
result_live_form` 全部是同一模式：它们不是正交的独立协议，而是
“某种 compute graph 边在某种 buffer storage 上的执行方式”
在同一个结构上的不同切片描述。

### 2.3 两条根因的关系

A 是“信息在时间上被销毁”。B 是“信息在类型系统里本来就没有被表达”。
A 让下游必须 recover，B 让 recover 的结果只能以 enum 形式落回。
它们互相强化：

- 因为算子在 `LowerTileOp` 就被拆散，下游没法问“这是什么 op”
- 没法问“这是什么 op”就只能贴“这是 `intermediate_fragment_merge`”
  这样的 string 标签
- 而 `intermediate_fragment_merge` 是一个**被命名的类别**，
  下一个情况就需要 `republished_logical_tile`，再下一个需要
  `tile_nfaces_materialization`，无限扩张
- 每个新标签都要求 analysis pass、lowering pass、planner、codegen、
  Python runtime 同时认一个新字符串，违反 layered IR 的基本纪律

因此任何只在“再加一个 string kind”层面的修复都会让问题恶化。

## 3. 对当前 blocker 的重新判定

`progress.md` 里旧主链下暴露出来的 blocker，
表面上看是这些局部问题：

- `SpatialProgram` 对跨-op intermediate edge 的 `dataflow contract`
  和 per-buffer `work/access contract` 仍未完整 formalize
- fragment-side `live_form_kind / execution_topology_kind /
  physical_local_extent` 缺 owner-side contract
- `LowerBlackholeOps` 仍要用 `Match*` 把 loop/buffer 形态反推回 typed op

但把这些问题放回根因 A/B 之后，结论已经很清楚：

- 这不是“再补一个 contract”就能收口的问题
- 也不是“把 `Fragment` truth 多传几步”就能治本的问题
- 真正要切换的是 active owner 链本身

也就是说，当前 blocker 的长期解法不再是：

```text
在旧 State/Semantic/Spatial 主链上
继续补 live-form / materialization / matcher 的 typed uplift
```

而应该直接变成：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec
```

这样重判之后：

- `Fragment` / `shared` / `layout_map` / `blackhole.acc`
  不再是要保住的第一层 truth，只是旧前端 hint 或 target realization
- `Task / Channel`
  不再是 primary truth owner，只是 companion 上的 coarse grouping/view
- `SemanticProgram`
  不再是长期 stable IR，而退回成旧主链里的 analysis/recovery 中间物

## 4. 整改方向

### 4.1 三条 invariant（必须同时满足）

1. **No destroy-then-recover**
   - 计算图、依赖、carry/merge、task 边界 truth 一旦建立，
     就必须从 owner 侧 typed 传到真正 materialization 的那一层
   - 允许 compatibility attr/projection，但不能再成为 primary truth
2. **No enum-of-forms for things that are structure**
   - 任何“某种 buffer / edge / task 在某种执行模式下的角色”，
     都先问能否由结构派生
   - 能派生的就不再立独立 `*_kind` 字段
   - 派生不了才扩 typed IR/schema
3. **Analysis reads IR, does not recover it**
   - analysis pass 只能消费 owner 侧已经 typed 暴露的事实
   - 禁止用 `Match*` / 名字 / lowered loop 结构去猜“这其实是什么”

### 4.2 长期结构

新的长期主链必须是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec / BlackholeModule
```

其中：

- `Normalized Tile TIR`
  - owner：语义 body 本身
- `SpatialPlan companion`
  - owner：closure、boundary、validated hints 这些
    TIR 没有对象化、但 planning 必须持久化的 truth
- `TTProgram companion`
  - owner：block/resource/CB/semaphore/core/ABI/execution
- `Task / Channel`
  - 继续存在，但退成 derived execution/materialization view

### 4.3 `SpatialPlan companion` 的最小对象集

第一层 companion 不应重复 TIR 里的 expr/访问语义，
只保留最小 planning object：

- `ExecutionClosure`
- `ClosureBoundary`
- `ValidatedHintSet`

关键纪律：

- `fragment/shared/layout_map`
  不是 ontology
- `blackhole.acc / blackhole.cb`
  不是 ontology
- 它们只允许作为 frontend hint 或 target realization 出现
- index expr、predicate、indirection、tile-op 参数
  继续留在 TIR 里，不在 companion 里重复编码

### 4.4 `TTProgram` 的最小 owner 补齐

第二层 stable IR 的关键整改点不是再补一个 runtime protocol，
而是补齐正确的 target owner：

- `TTBlockPlan`
- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTABIPlan`
- `TTExecutionPlan`

其中 `TTBlockPlan` 是当前缺失的核心：

- `work_packets`
  不能继续挂在 `TTCoreGroup` 上兼职 decomposition owner
- `CB / semaphore / runtime args`
  不能继续从 lowered loop matcher 反推

### 4.5 Pass 介入点与主链替换

新的主链介入点固定在 generic frontend normalization 之后：

```text
BindTarget
  -> AddWrapperForSingleBufStore
  -> LegalizeNegativeIndex
  -> VerifyParallelLoop
  -> InjectAssumes
  -> Simplify
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
```

而不是建在：

- `LayoutReducer / LayoutInference` 之后
- `CollectSemanticManifestSeeds / LowerTileOp` 之后

原因是：

- 到 `Simplify` 为止，tile 级计算、structured loop、region truth 仍然活着
- 再往后就开始进入 GPU realization 与 destroy-then-recover 链

因此当前旧主链里的这整段 recovery/companion 路线，
都不应再是长期 active owner 链：

- `CollectSemanticManifestSeeds`
- `ProjectSemanticManifest`
- `AugmentSemanticManifest`
- `AnalyzeSemanticStructure`
- `LiftStatefulSemanticIR`
- `AnalyzeSpatial*`
- `MaterializeSpatialProgram`
- `LowerBlackholeOps`

## 5. 与现有设计文档的关系

- `final_blackhole_backend_redesign.md`
  - 这里已经收敛为唯一总设计：
    `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
  - 本文档负责说明为什么必须这么切，以及当前旧主链具体错在哪里
- `spatial_dataflow_program_model.md`
  - 已按
    `TIR body / SpatialPlan companion / TTProgram companion`
    边界重写
  - 旧的 `live_form_kind / execution_topology_kind /
    physical_local_extent` 一类提案不再作为独立方向推进
- `stage4_phase_c_tt_target_ir.md`
  - 继续记录当前已落地 `TTProgram` 基线、支持面与 gate
  - 不再承担总体 layering 权威
- `memory/general_dev.md`
  - 本文档与其纪律一致：analysis 不再做名字/结构 recovery，
    所需信息优先从 owner-side typed IR/schema 提供

## 6. 下一步

1. 把 supporting design 文档收正到两层总设计：
   - `spatial_dataflow_program_model.md`
2. 在 `Simplify` 后建立新入口：
   - `AnalyzeSpatialStructureFacts`
   - `BuildSpatialPlanCompanion`
3. 基于 companion 建立 target planning：
   - `PlanTTBlocks`
   - `PlanTTTransport`
   - `PlanTTSync`
   - `PlanTTABI`
   - `PlanTTExecution`
4. 用 `TTBlockPlan` 重新收正 target owner 链：
   - 不再让 `work_packets / CB / semaphore` 兼职上游 owner
5. 在同一轮 cutover 中退场旧 recovery 主链，
   而不是继续沿旧链做 `*_kind`/matcher 增量修补
