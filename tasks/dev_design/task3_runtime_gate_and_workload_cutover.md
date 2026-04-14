# Task 3: 旧 Recovery 链退场、Runtime Gate 与 Workload Cutover

## 基本信息

- **文档角色**: `Task 3` 的 runtime gate、support surface
  与 workload re-enable 设计文档
- **当前状态**: `2026-04-14` 活动设计文档；`Task 2` 已完成，
  `Task 3A` persistent/public 删除批次已完成，当前工作进入 `Task 3B`
- **任务链位置**:
  `Task 2` owner cutover 完成之后，
  负责 runtime/correctness payoff 与 wider family 承接
- **非目标**:
  - 不重新定义 `SpatialPlan` / `TTProgram` 的 owner 边界
  - 不在 runtime/codegen 里补 planning contract
  - 不恢复旧 recovery 主链
  - 不把 non-Blackhole backend 一并收口到同一套 runtime / artifact contract
  - 不把 public Python `transform` API 改名当作本任务目标
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **上游设计输入**:
  - `tasks/dev_design/task1_spatial_plan_companion.md`
  - `tasks/dev_design/task2_ttprogram_companion_cutover.md`

读法说明：

- 本文档只记录当前 `Task 3` 的 gate、cutover 和 workload 承接
- 若文中出现旧 pass / 旧链清理条目，
  默认按**迁移 inventory**理解，
  不是当前长期设计骨架

## 1. 作用域

`Task 3` 只负责三件事：

- 保持 persistent `SemanticProgram / Stateful Semantic IR`
  这一层旧 companion 已经删除，且不再回流
- 在新主链上兑现 runtime / correctness payoff
- 在新主链上重新承接 workload family 与更宽 support surface

这里的“新主链”固定指：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec
  -> runtime / codegen / BlackholeModule
```

`Task 3` 不重新讨论 owner 链本身。
如果 runtime 或某个 workload 仍然缺 truth，
结论只能回到：

- `Task 2` owner cutover 还没完成
- 上游 `TIR/DSL` 或 analysis 还缺 truth
- 当前 variant 必须显式 `unsupported`

补充边界：

- `Task 3` 默认只讨论 `Blackhole` active runtime / codegen path
- non-Blackhole backend 的统一化不进入当前 backlog

## 2. Shared Zero-Regression Baseline

无论 `Task 3` 扩多少支持面，下面这些基线都不能回退：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 不回退
- copy / GEMM / export 当前正式支持面不回退
- Blackhole runtime / direct-runtime 正式 correctness baseline
  统一使用 `bf16`
- 对未支持功能继续 explicit unsupported / fail-fast，
  不回退到 silent fallback 或 late runtime failure

## 3. Runtime Gate

runtime / codegen 的消费纪律固定为：

- 只读 `ExecutableSpec`
- 不读 legacy attrs
- 不补 target planning
- 不从 `work_linear_id`、arg kind、builtin 序列、
  payload bag 恢复语义

当前及后续 gate 一律按同一原则收口：

1. **per-work access truth 缺失**
   - 如果 executable 没有显式 per-work access descriptor，
     runtime 不允许再从 `work_linear_id` 重建 tile access
2. **transport / compute protocol 缺失**
   - 如果 executable 还缺显式
     data movement / compute / materialization
     相关 protocol truth，
     runtime 不允许再从 builtin 序列猜语义
3. **target sync truth 缺失**
   - 一旦 executable 带显式同步对象或绑定，
     但 runtime 还没有对应执行语义，
     必须 fail-fast

这条 gate 的目的不是“先挡住错误 case”，
而是强迫所有新支持面都经由 owner-side typed truth 落地。

## 3.1 Legacy Chain Inventory 与删除顺序

当前仓库中的“旧链路”不是单一 pass，而是 5 类残留叠加：

0. **persistent semantic layer**
   - `LiftStatefulSemanticIR / ValidateStatefulSemanticIR /
     ValidateSemanticRefinement`
   - `tl.semantic_program`
     这层 typed attr
   - 这条链已经从 active path 删除；
     当前约束是不得再把 TIR/SpatialPlan 已有信息
     包装回独立 semantic companion

1. **projection bridge**
   - `tilelang/engine/lower.py::_project_blackhole_host_attrs_to_device`
   - 把 host entry 上的
     `tl.semantic_* / tl.spatial_* / blackhole.resource_plan`
     重新投影到 device kernel
   - 这类桥接属于典型 side-channel；
     只允许收缩，不允许继续扩张
2. **typed seed bridge**
   - 当前 active path 已不再暴露 legacy pass 名字；
     `BuildTTProgram` 内部暂时通过
     `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
     helper chain 直接产出 planning object
   - `BuildTTProgram` 直接把 planner result 聚合成 `TTProgram`，
     不再经由
     `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
     tl.tt_core_groups / tl.tt_program_payload`
     这组中间 attrs
3. **legacy attr synthesis**
   - `SplitBlackholeKernel` 已退回成纯 IR annotation pass，
     不再写 `blackhole.segment_plan`
   - `PlanTTKernelABI` 已不再写
     `blackhole.runtime_args / blackhole.gemm_contract /
     blackhole.compute_contract`
     这组 legacy attr；target truth 直接留在
     `TTKernel / TTABIPlan / TTProgram payload`
4. **matcher / recovery owner residue**
   - `PlanTTKernelABI` 仍承担 target planning 的一部分恢复责任
   - 尤其是 phase-boundary / fragment / runtime arg /
     compute contract 这组 target truth
   - 这说明真正的
     `PlanTTBlocks / PlanTTTransport / PlanTTCompute /
     PlanTTSync / PlanTTABI / PlanTTExecution`
     还没有完全落地为独立 owner pass

删除纪律：

- 不允许直接硬删当前仍在承担 owner 职责的 pass，
  然后把缺口重新塞回 projection / payload / runtime fallback
- 旧链路的删除顺序必须从**最外层 bridge**往里收：
  1. 先删 persistent semantic layer，
     让 active path 直接消费
     `Normalized Tile TIR + SpatialPlan companion +
     Blackhole analysis facts`
  2. 再删 projection bridge
  3. 再删 fragment-layout / semantic seed 这类 side-channel
  4. 再把 typed seed bridge 替换成真正的 `PlanTT*` owner pass
  5. 最后删 legacy attr synthesis 与 matcher/recovery owner residue

当前明确的删除批次：

### Batch 0: 删除 persistent semantic layer

- 删除
  `LiftStatefulSemanticIR / ValidateStatefulSemanticIR /
  ValidateSemanticRefinement`
  这组 pass 的 active-path owner 责任
- `AnalyzeSpatialDomainPlan / AnalyzeSpatialExecutionPlan /
  MaterializeSpatialProgram / ValidateSpatialProgram /
  LowerBlackholeOps / PlanTTKernelABI`
  必须直接从
  `Normalized Tile TIR + SpatialPlan companion +
  blackhole.work_decomposition / blackhole.compute_regions /
  blackhole.pipeline_stages`
  读取所需 truth
- `tl.semantic_*`
  不再作为 active path 上的稳定 attr
- 删除依赖这层 attr 的 wrapper、helper 与过期测试
- 当前状态：
  已完成；当前 active path 已不再保留
  `Stateful Semantic IR`
  这一层，相关 decoder / witness / vocab / refinement /
  state-effect graph 实现也已删除

### Batch A: 删除 projection bridge

- 删除或清空
  `_project_blackhole_host_attrs_to_device`
- 直接禁止新增任何 host->device projection key
- `fragment_layout` / semantic / sync / transport truth
  缺口一律回到：
  `补 analysis / 补 companion schema / explicit unsupported`
- 当前状态：
  已完成；`blackhole_codegen` 不再把 host attr 投影回 device module

### Batch B: 删除 fragment-layout side-channel

- 删除 `tl.fragment_layout_seeds`
  的生产与消费
- `AnalyzeBlackholeComputeRegions`
  只能从 TIR 结构与已冻结 companion contract 取事实
- grouped row / fragment distribution 这类缺口
  不能再落进新的 side contract；
  只能通过
  `补 analysis / 补 TIR-schema / 补 TTTransportPlan/TTABIPlan owner`
  或 `explicit unsupported` 解决
- 当前状态：
  已完成；当前 admitted 基线不再依赖 fragment-layout projection seed
  且 `AnalyzeBlackholeComputeRegions`
  已成为当前 active 文件/测试入口，
  不再保留旧 `FragmentRegions` 文件名和 side-channel helper

### Batch B1: resource canonicalization 不再制造双真源

- `BlackholeDeviceResourceCanonicalization`
  必须在
  `LowerToBlackholeExecutable`
  之后运行；
  先让 `TTProgram / ExecutableSpec` owner 链 materialize companion truth，
  再 canonicalize device-private physical resource class
- canonicalize 之后，
  必须同步改写
  `blackhole.lowering_requirements / blackhole.compute_regions /
  tl.spatial_program / tl.tt_program`
  中的 `scope`
- 不允许出现
  “TIR body 已经是 `blackhole.acc` /
  `blackhole.cb.*`，companion contract 还写 `local` /
  `shared`”
  这种双真源状态
- 当前状态：
  已完成；resource canonicalization 当前已和 companion scope
  rewrite 一起收口

### Batch C: typed seed bridge -> owner pass

- 用真实的
  `PlanTTBlocks / PlanTTTransport / PlanTTCompute /
  PlanTTSync / PlanTTABI / PlanTTExecution`
  取代
  当前内部
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
  helper bridge
  在 owner 链上的兼职责任
- `BuildTTProgram`
  只做 plan object 聚合，
  不再读 seed bridge attr
- 当前状态：
  已完成中间 seed attr 退场；`BuildTTProgram`
  不再物化/读取
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
  tl.tt_core_groups / tl.tt_program_payload`
  这组中间 attrs，最终 Phase C 输出只保留 `tl.tt_program`
  作为 target truth
- 补充：
  canonical `LowerToBlackholeTTProgram`
  已不再显式串
  任何 legacy pass 名字；
  当前剩余残留是 `BuildTTProgram`
  内部仍临时复用
  `PlanTTKernelABI -> PlanTTCBAlloc -> PlanTTCoreGroups`
  这组 helper 直接承担 planning owner 责任
  - public `tilelang.transform` wrapper、
    FFI `tl.transform.*` global registration
    以及测试层
    `lower_blackhole_ops_through_phase_b` /
    typed-seed helper
    已删除，不再允许从公开面重走这条链

### Batch D: 删除 legacy attr synthesis / fallback

- 删除 `blackhole.segment_plan / blackhole.runtime_args /
  blackhole.common_runtime_args / blackhole.cb_configs /
  blackhole.core_plan / blackhole.gemm_contract /
  blackhole.compute_contract`
  的生成、helper fallback 和测试 fallback
- runtime / codegen / tests 全部只读
  `TTProgram / ExecutableSpec`
- 当前状态：
  已完成 active path / helper / test fallback 删除；
  `SplitBlackholeKernel` 不再写 `blackhole.segment_plan`，
  `PlanTTKernelABI` 不再综合
  `blackhole.segment_plan / blackhole.runtime_args /
  blackhole.common_runtime_args / blackhole.cb_configs /
  blackhole.core_plan / blackhole.gemm_contract /
  blackhole.compute_contract`
  这组 compatibility attrs。
  runtime / codegen 与相关回归当前只依赖
  `TTProgram / ExecutableSpec`；
  剩余未完成项已经从 attr synthesis 切换成
  `PlanTT*` owner pass 的真实拆分

每一批删除前都必须满足：

1. 当前 zero-regression baseline 不回退
2. 缺口已经由 typed companion / analysis 承接，或显式 unsupported
3. 不引入新的 compatibility projection
4. 不把删除后的缺口转移给 runtime/codegen

## 4. `flash-attn` 与 runtime payoff

`flash-attn` 在 `Task 3` 里的角色是：

- 作为第一批验证新主链是否真的能承接
  multi-op / multi-work / intermediate dataflow 的 workload

对这条线，完成标准不是“compile-path 通了”，而是：

1. 支持的 `MHA / GQA` subset
   在 admitted support surface 内真实执行
2. runtime 结果与 reference 数值对齐，
   不只是“不挂死”
3. 对仍未支持的 shape / variant
   给出显式 unsupported / fail-fast
4. 不再依赖旧 recovery 主链、
   legacy attrs、名字匹配或 codegen heuristics

当前补充约束：

- `flash-attn` 当前主要缺口
  仍然是 multi-phase data movement / reduction / broadcast
  在新 `PlanTTTransport + PlanTTCompute`
  路线上的完整承接
- 因此前一轮把这类前提当作稳定绿测的 probe case
  已从 canonical TT target probe 文件中移除，
  不再把未 admitted 的前提固化成当前基线

## 5. Wider Family Cutover

`Task 3` 必须在新主链下重新承接下面这些 family：

- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

对每个 family，最低完成标准固定为：

1. 走完整新主链：
   `Normalized Tile TIR -> SpatialPlan companion ->
   TTProgram companion -> ExecutableSpec`
2. target truth 从 `TTProgram / ExecutableSpec` 物化，
   不是 runtime/codegen 侧补洞
3. 至少有一组明确支持的 subset，
   带有对应的 transform / target regression
4. 对未支持部分有显式 unsupported / fail-fast 边界

## 5.1 当前执行优先级

`Task 3` 的推进顺序固定为：

1. **P0: 真实 `PlanTTTransport + PlanTTCompute` cut-in**
   - 用 anchored sub-TIR 上仍保留的
     tile-op / layout / `load/store`
     完成 target builtin mapping
   - 取代 `BuildTTProgram` 内部 helper bridge
2. **P1: runtime gate 收口**
   - 先确保新主链上的 copy / GEMM / export
     仍维持当前 zero-regression baseline
3. **P2: `flash-attn` payoff**
   - 再拿它验证 multi-op / multi-work /
     multi-phase data movement 的 admitted subset
4. **P3: wider family cutover**
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
5. **P4: wider copy / dataflow**
   - 在 admitted subset 内逐步放宽 range / stride / staged-dataflow
6. **P5: wider sync**
   - 只在 `TTSyncPlan + executable binding + runtime semantics`
     三者都稳定后进入 admitted surface

## 5.2 当前旧链清理优先级

在 `Task 3B` 里，当前剩余的旧链清理按下面顺序执行：

1. **P0: 删除 runtime/build 的 legacy single-kernel fallback**
   - `rt_mod_blackhole` 不再接受
     “`TTProgram` 缺 segment truth 时退回单 kernel codegen”
   - `fused_dataflow` 也必须走显式 `TTKernel/TTABIPlan`
     物化后的 segment codegen
2. **P1: 删除 segment ABI 的 top-level runtime-args 回填**
   - 不再使用
     `tt_uses_top_level_runtime_args /
     tt_uses_top_level_common_runtime_args`
     这类过渡标记
   - segment 需要的 runtime/common runtime args
     必须直接作为 segment ABI truth 冻结在
     `TTABIPlan`
3. **P2: 删除 runtime 里的 positional buffer-role fallback**
   - `BlackholeModule` 不再从“最后一个 buffer 是 output”
     这类位置约定恢复输入输出角色
   - 缺显式 buffer role/buffer name schema 时直接 fail-fast
4. **P3: 收紧 `BuildTTProgram` helper bridge**
   - `BuildTTProgram` 继续往纯聚合器收
   - 不再在 helper chain 里保留 top-level 汇总 truth，
     需要的 truth 直接下沉到对应 owner object
5. **P4: 拆 `PlanTTKernelABI` 的 matcher / recovery owner residue**
   - 把仍然混在一起的 block/transport/sync/ABI/execution owner
     责任继续拆开
   - 目标不是系统性换名，而是让每层只持有它自己的 target truth

### 5.2.1 `2026-04-14` 当前状态

`Task 3B` 这 5 项旧链清理已完成：

- **P0**
  - `rt_mod_blackhole` 已删除 legacy single-kernel/fused-dataflow codegen fallback
  - device build 现在硬要求显式 `TTProgram` segment truth
- **P1**
  - `tt_uses_top_level_runtime_args /
    tt_uses_top_level_common_runtime_args`
    已删除
  - `fused_dataflow` copy 的 runtime/common-runtime/per-work truth
    已直接冻结到 segment ABI
- **P2**
  - direct runtime 已删除 positional buffer-role fallback
  - 缺显式 buffer binding/schema 的 executable
    直接在 build/gate 边界 fail-fast
- **P3**
  - segment materialization helper
    已删除 top-level runtime/common-runtime 回填
- **P4**
  - `PlanTTKernelABI` copy classifier
    现在只接受 explicit direction truth
  - `unknown -> fallback` 的恢复分支已去掉

因此当前剩余重点已转到 `P5`：

- 真实 `PlanTTTransport + PlanTTCompute` cut-in
- `flash-attn` correctness payoff
- `Task 3C` wider family / support surface

## 6. Wider Copy / Dataflow / Sync 支持面

支持面扩张也必须服从同一条 owner 纪律。

### 6.1 Copy / Dataflow

后续可以扩张：

- wider copy range / stride / layout subset
- wider staged-copy / multi-stage dataflow subset
- 跨 op intermediate dataflow 的更多 admitted subset

但扩张前提只能是：

- 上游 `TIR/DSL` 已经显式表达所需 truth
- analysis 已能稳定读出
- `TTProgram / ExecutableSpec`
  已把对应 target truth 冻结下来

### 6.2 Synchronization

后续可以扩张：

- 显式 semaphore / compute sync / barrier admitted subset
- 更多 local / remote / multicast synchronization case

但所有同步支持都必须满足：

- `TTSyncPlan` 已成为稳定 target truth
- executable 已能稳定承载对应 binding / ordering
- runtime 对该 sync family 有明确执行语义

在这之前，一律 fail-fast。

## 7. 完成判定

`Task 3` 完成必须同时满足：

1. `flash-attn` admitted subset
   在新主链上真实通过 runtime/correctness gate
2. `topk / fusedmoe / paged decode / chunk recurrence`
   至少各有一组支持 subset
3. wider copy / dataflow / sync 支持面
   通过 `TTProgram / ExecutableSpec` 扩张，
   不靠 fallback
4. 对未支持部分仍保持显式 unsupported / fail-fast
5. 旧 recovery 主链不再作为任何 family 的兜底路线
