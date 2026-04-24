# Task 2: Remove Public And Internal Legacy Analysis Bags

## 0. `2026-04-24` 收口更新

- **状态**: `completed`
- repo HEAD 当前已完成：
  - public
    `AnalyzeBlackhole*`
    wrapper 删除
  - internal
    `AnalyzeBlackhole*Evidence(...)`
    helper / C++ pass / legacy analysis test 删除
  - producer-side logical bridge handoff
    已不再依赖
    `AnalyzeBlackholeComputeRegions`
  - `blackhole.lowering_requirements`
    broad bag
    已退出 active chain
  - `tl.blackhole_lowering_requirements_seed`
    /
    `blackhole.cb_requirements`
    已退出 active chain
  - `BuildTTProgram`
    不再接收
    `tl.internal_tt_*`
    staging bag
- 当前剩余相关 debt
  不属于 task2 broad-analysis 删除：
  - `tl.blackhole_logical_buffer_tile_bridge_specs`
    是唯一窄 bridge attr，
    后续由
    `SpatialPlan`
    logical live-value /
    materialization-boundary
    与 typed leaf materialization schema
    替换
  - `compute_contract`
    /
    `gemm_contract`
    /
    `multi_*_contracts`
    是 task3 /
    leaf contract-family debt

> 下文 `2026-04-17` 的“当前状态 / 当前代码现实”
> 保留为任务开始前快照，不再代表 repo HEAD 当前状态。

## 1. 任务目标

这个 cleanup task 只负责一件事：
把 Blackhole 这几类 legacy analysis 面从 active planning chain 里拿掉：

- public `AnalyzeBlackhole*` wrapper
- internal `*Evidence(...)` helper
- `blackhole_lowering_requirements.cc`
  里仍在承担 planning 语义的 `Map<String, Any>` bag

完成后，任何需要信息的代码只能做两种事：

1. 直接读取当前 `PrimFunc` / `SpatialPlan` / `TTProgram`；
2. 如果结果必须跨边界保留，就直接写进显式表示层或最终 projection。

这个 task 的目标不是“把 legacy attr 从最终输出里藏起来”，
而是把这些 bag 真正从 active semantic path 里删掉。

## 2. 范围

这个 task 允许做的事：

- 删除 `tilelang.transform`
  暴露的 public legacy analysis wrapper
- 删除内部 `AnalyzeBlackhole*Evidence(...)`
  helper entrypoint
- 把现有 consumer 从 evidence bag / lowering bag
  切到当前 IR 或显式表示层
- 把 `blackhole_lowering_requirements.cc`
  从 planning semantic carrier
  收缩成 leaf helper，或者直接删除

这个 task 不负责的事：

- 重新定义新的表示层
- 再发明一个 replacement bag / replacement helper pass
- 用另一套 `Map<String, Any>` 契约替换旧 bag
- 删除
  `TTProgram.payload`
  /
  `tl.blackhole_executable`
  上仍然存活的
  leaf payload / projection residue；
  这属于 task3
- 删除
  `blackhole.segment_kind`
  这类 leaf-side body slicing residue；
  这属于 task4
- 顶层 `Task 2: TTProgram Representation Cutover`
  的全部工作

## 3. 历史状态快照 (`2026-04-17`)

当时 **不算完成**。

repo HEAD 上已经成立的只有一些表面现象：

- 最终 `PrimFunc`
  通常已经不再暴露
  `blackhole.lowering_requirements`
- 多个 pipeline test
  已经会检查这些 legacy attr
  不再出现在最终输出上

但 task2 的主合同还没有成立：

- `tilelang.transform`
  仍然公开导出
  `AnalyzeBlackholeWorkDecomposition`
  `AnalyzeBlackholeComputeRegions`
  `AnalyzeBlackholePipelineStages`
- 对应的 registered pass / evidence helper / 源文件都还在
- `engine/lower.py`
  当前 active Blackhole path
  仍直接调用
  `AnalyzeBlackholeComputeRegions()(LowerToBlackholePhaseB(mod))`
- `blackhole_lowering_requirements.cc`
  仍然是 active semantic carrier：
  它会把 work decomposition /
  compute regions /
  pipeline stages /
  buffer contracts
  聚合成一个 `Map<String, Any>`
- canonical
  `LowerToBlackholeTTProgram`
  主链本身已经不再调用
  public `AnalyzeBlackhole*` wrapper，
  `BuildTTProgram`
  也已经是 staged
  `TTProgram`
  的聚合/清理点，
  不是 task2 的主要问题源
- `lower_blackhole_ops.cc`
  和 `blackhole_lowering_requirements.cc`
  仍然直接消费
  `AnalyzeBlackhole*Evidence(...)`
- runtime / codegen / build
  reader
  现在已经主要读取
  `tl.blackhole_executable`
  /
  `TTProgram.payload`
  投影，
  不再直接读取这些
  public / internal legacy analysis 面
- `test_blackhole_flash_attention_analysis.py`
  仍然是围绕 public wrapper 和 legacy attr 写的，
  还没有迁走

所以当前只能写成：

> 最终 legacy attrs 基本已不落到最终产物

不能写成：

> public / internal legacy analysis bag
> 已经退出 active planning chain

## 4. 当前代码现实

### 4.1 public wrapper 还在，而且 active path 还在用

当前 [tilelang/transform/__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py#L656)
仍然导出：

- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeComputeRegions`
- `AnalyzeBlackholePipelineStages`

而且 `AnalyzeBlackholeComputeRegions`
不只是“历史 API 还没删”。
它现在还在 active Blackhole lowering path 里被直接调用，
见 [engine/lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L369)
、
[testing/python/target/blackhole/common.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/common.py#L97)
和
[test_blackhole_flash_attention_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py#L102)。

但要注意：
[LowerToBlackholeTTProgram](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py#L377)
这条 canonical
`TTProgram`
主链本身已经不再调用这些 public wrapper。

这说明 task2 当前真正要切掉的是：

- `engine/lower.py`
  / target test helper
  这类 compile helper / test scaffolding 依赖
- planner 内部对 evidence helper /
  lowering bag 的复用

而不是去把
`BuildTTProgram`
重新定义一遍。

这意味着：
task2 不能把“先删 public wrapper”
写成第一步，
因为当前主链还没从它身上切走。

### 4.2 internal evidence helper 还在被 active consumer 读取

当前 evidence helper 仍然存在并被直接消费：

- [AnalyzeBlackholeWorkDecompositionEvidence](/root/dev/vibe_dsl/tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc#L192)
- [AnalyzeBlackholeComputeRegionEvidence](/root/dev/vibe_dsl/tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc#L973)
- [AnalyzeBlackholePipelineStageEvidence](/root/dev/vibe_dsl/tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc#L341)

直接 consumer 还能看到：

- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1205)
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1262)
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L109)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1749)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L2045)

所以现在 legacy analysis 不是“只剩 public surface 没删”，
而是内部语义消费还在。

### 4.3 `blackhole.lowering_requirements` attr 虽然不见了，但 lowering bag 还活着，而且仍在驱动 planner

当前很多测试已经证明：

- 最终函数 attrs 里没有
  `blackhole.lowering_requirements`

但这不等于 task2 已经完成一半。

真正的问题在于：
[BuildBlackholeLoweringRequirements](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1294)
现在仍然把多种语义揉进一个
`Map<String, Any>`：

- work decomposition
- phase facts
- pipeline stages
- compute support payload
- bridge specs
- materialization contracts
- flow contracts

然后
[lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1460)
/
[lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L2319)
再把它作为 active planning 输入继续消费。

当前已经能直接看到的用途包括：

- `HasComputeSupportContract`
  用
  `compute_op_kinds`
  决定是否需要 compute segment，
  见 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1452)
- `ValidateComputePipelineLegality`
  用
  `pipeline_stage_counts`
  做 legality gate，
  见 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1419)
- `LoadBufferFlowContracts`
  /
  `BuildBufferMaterializationContractMap`
  /
  `LoadBufferTileBridgeSpecs`
  用这些 bag 字段继续做 planner / payload shaping，
  见 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1500)
  [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1849)
- `StoreLeafExecutableContracts`
  又把
  `buffer_tile_bridge_specs`
  和衍生出的
  `unsupported_compute_ops`
  写回
  `TTProgram.payload`，
  见 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L3958)

而且这个 bag
不只是“有 active reader”这么简单。
repo HEAD 里
[BuildBlackholeLoweringRequirements](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1294)
现在还会继续产出一批
repo 内已经找不到 active consumer
的字段：

- `work_axes`
- `derived_index_expr_count`
- `work_dependent_loop_bound_count`
- `spatial_phase_count`
- `spatial_channel_count`
- `spatial_phase_boundary_buffers`
- `pipeline_loop_vars`

这些字段没有证明
“bag 还有合理边界”。
相反，它们只说明：
这个 broad
`Map<String, Any>`
已经在同时承担
active planner input
和 bag-only residue
两种角色，
所以 task2 的 required end-state
不能是“换个名字继续保留总包”，
而必须是把需要的事实
拆回当前 IR /
当前 `SpatialPlan` /
pass-local helper /
leaf-local projection。

也就是说：
`blackhole.lowering_requirements`
这个 legacy carrier
只是从“最终 attr”
退回成了“internal helper return value”，
并没有真正退出 active planning chain。

### 4.4 `BuildTTProgram` 不是 task2 当前的主 blocker

[BuildTTProgram](/root/dev/vibe_dsl/tilelang_repo/src/transform/build_tt_program.cc#L602)
现在做的事情已经比较接近
task2 需要的角色：

- 读取 staged
  `tl.tt_program`
- 检查
  `TTBlockPlan / TTKernelPlan / TTSyncPlan /
   TTABIPlan / TTExecutionPlan`
  和 compatibility payload 的对齐
- 清理
  `tl.internal_tt_*`
  与
  `tl.blackhole_lowering_requirements_seed`
  这类中间 attrs

对应测试也已经在检查：

- `BuildTTProgram`
  后
  `tl.internal_tt_*`
  不再残留，
  见 [test_blackhole_spatial_ir.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py#L336)
- staged planning 过程中
  `tl.tt_program`
  已经承接 owner truth，
  不再依赖
  `tl.internal_tt_*`
  attr，
  见 [test_blackhole_spatial_ir.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py#L364)

所以 task2 不应该被写成：

- “重做 `BuildTTProgram`”
- “重新引入一层 internal staging attr”

task2 真正要修的是
`BuildTTProgram`
之前那层
analysis/bag
依赖。

### 4.5 runtime / codegen / build reader 已经基本不再读 legacy analysis bag

当前 runtime / codegen / build
reader 的主边界已经是：

- `TTProgram`
  -> `tl.blackhole_executable`
  projection，
  见 [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h#L206)
  和 [materialize_blackhole_executable.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/materialize_blackhole_executable.cc#L17)
- `codegen_blackhole.cc`
  /
  `rt_mod_blackhole.cc`
  读取
  `tl.blackhole_executable`
  而不是
  `AnalyzeBlackhole*`
  或
  `blackhole.lowering_requirements`
  这类 legacy analysis 面，
  见 [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L231)
  和 [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L1703)

但这不代表 leaf-side residue
已经收口。

当前仍然存活的包括：

- `TTProgram.payload`
  /
  executable projection
  里的
  `buffer_tile_bridge_specs`
  和 contract payload，
  这是 task3 范围
- runtime-side
  `blackhole.segment_kind`
  body slicing residue，
  这是 task4 范围

所以 task2 不能把这些后续 debt
误记成
“public/internal analysis bag
还必须保留”。

### 4.6 task2 当前还受 task1 卡住

task1 还没有完成 direct logical bridge capture。
所以当前 active path
仍然依赖
`AnalyzeBlackholeComputeRegions`
做 bridge handoff，
见 [engine/lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L374)。

这意味着：
task2 不能先把 compute-region public wrapper 删掉，
否则会直接打断当前主链。

正确顺序只能是：

1. 先完成 task1，
   让 bridge handoff 不再依赖 compute-region wrapper
2. 再做 task2，
   删除 public/internal legacy analysis 面

### 4.7 还有历史兼容代码在改写这些 legacy attr

当前 [blackhole_device_resource_canonicalization.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc#L763)
仍然会在 attr 存在时改写：

- `blackhole.lowering_requirements`
- `blackhole.compute_regions`
- `blackhole.pipeline_stages`

这不是 active planning 主链，
但说明 legacy analysis 相关 compatibility residue
在代码库里还没有清完。

## 5. 基于审计结果修正后的任务内容

### 5.1 先切掉 active consumer，再删 public wrapper

原文档里把
“先删 public wrapper”
写成了早期步骤。
这个顺序是错的。

正确顺序应该是：

1. 先让 active consumer
   不再依赖这些 wrapper / evidence helper
2. 再删除 public wrapper / registered pass / 源文件

否则当前 Blackhole 主链会直接断掉。

### 5.2 task2 真正要删的是“analysis 作为语义 carrier”的角色

task2 的重点不是：

- 最终 attrs 里有没有 `blackhole.lowering_requirements`

task2 的重点是：

- active planning chain
  是否仍然依赖 broad analysis bag / evidence helper

因此 task2 完成后必须同时成立：

- public wrapper 不再存在
- internal evidence helper 不再存在
- `blackhole_lowering_requirements.cc`
  不再承担 planning semantic carrier

### 5.3 不允许把 `blackhole_lowering_requirements.cc` 留成新的内部总包

只要这个文件还在 active planning chain 里返回
“一个装满各种语义的 `Map<String, Any>`”，
task2 就没完成。

这点不只是当前 Blackhole 路线的偏好，
也是 repo 内成熟后端的共同边界：

- local collector / visitor
- 直接 rewrite 成显式 attr / typed object
- leaf writer / runtime reader

而不是：

- public wrapper
- internal evidence helper
- broad `Map<String, Any>` 总包
- 最后再让 builder / runtime / codegen 去猜

如果这个文件在 task2 后仍然保留，
它只能做：

- 同一个 planner `.cc`
  里的 pass-local utility
- leaf-local projection helper
- leaf-local validation helper

它不能再承担：

- 汇总 work / compute / pipeline / bridge / materialization / flow
  的统一 semantic bag
- 保留一批
  repo 内已经没有 active reader
  的 bag-only residue 字段
- 以 public 头文件 /
  cross-pass helper API
  的形式继续当语义入口

### 5.4 `BuildTTProgram` 已经接近正确角色，task2 不应回退它

task2 文档里要明确：

- `BuildTTProgram`
  当前已经是 staged
  `TTProgram`
  的聚合 / 清理点
- task2
  不需要通过
  `tl.internal_tt_*`
  或新的 bag
  再给它喂一层中间协议

换句话说：
task2 的实现重点
必须落在
`engine/lower.py`
/
`blackhole_lowering_requirements.cc`
/
`PlanTTKernelABI`
这类真正还在吃
analysis bag
的地方。

这里也要明确：
TT-Metal 的稳定 lowering 边界
本来就是显式
program / kernel / circular-buffer /
semaphore / runtime-arg
对象与 API。
它没有提供任何 target-model 理由，
要求我们保留
`work_decomposition`
/
`compute_regions`
/
`pipeline_stages`
/
`lowering_requirements`
这类 broad semantic bag，
更没有理由把
`BuildTTProgram`
写成新的 semantic owner。

### 5.5 task2 / task3 / task4 边界要写清楚

task2 删除的是：

- public `AnalyzeBlackhole*` wrapper
- internal `*Evidence(...)`
  helper
- broad
  `blackhole_lowering_requirements`
  planner bag

task2 不直接删除的是：

- `TTProgram.payload`
  /
  `tl.blackhole_executable`
  里的 leaf payload / projection residue；
  这是 task3
- runtime-side
  `blackhole.segment_kind`
  residue；
  这是 task4

但 task2 也不能因为
task3 / task4
还没做，
就把前面的
analysis bag
继续描述成“必须保留的中间层”。

如果 task2 后
`buffer_tile_bridge_specs`
之类 leaf payload
还暂时存在，
它的来源也只能是：

- 当前 IR
- 当前 `SpatialPlan`
- task1 定义的窄 bridge attr
- 或明确的 `TTProgram` / executable projection writer

而不能继续是：

- public wrapper
- internal evidence helper
- broad lowering bag

### 5.6 task1 / task2 边界要写清楚

task2 文档里必须明确：

- task1 负责把 bridge handoff
  从 compute-region wrapper 上切走
- task2 才负责删除
  `AnalyzeBlackholeComputeRegions`
  这类 public / internal analysis 面

不能把这两层混成一个 task，
也不能拿 task1 未完成
来把 task2 的真正合同写虚。

### 5.7 旧 analysis 测试现在还不是“已经 obsolete”

[test_blackhole_flash_attention_analysis.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py#L1)
当前仍然是在给现有 public wrapper / legacy attr
提供覆盖。

只有在 task2 真正删掉这些 surface 之后，
它才应该被删除或迁移。

所以 task2 文档里不能把它写成
“现在已经是纯废弃测试，只等删除”。
更准确的说法是：

- task2 完成时应删除它，
  并把仍然有价值的断言迁到 canonical path coverage

## 6. 执行切片

1. 先完成 task1 依赖：
   active Blackhole path
   不再依赖
   `AnalyzeBlackholeComputeRegions`
   做 bridge handoff
2. 把 `lower_blackhole_ops.cc`
   和 `blackhole_lowering_requirements.cc`
   从 `AnalyzeBlackhole*Evidence(...)`
   上切走
3. 让 current consumer
   直接读取当前 `PrimFunc` / `SpatialPlan`
   或当前 planner 内的 pass-local helper，
   不再返回新的 broad `Map<String, Any>` bag
4. 收缩或删除
   `blackhole_lowering_requirements.cc`
   的 planning semantic-carrier 角色
5. 保持
   `BuildTTProgram`
   继续作为 staged
   `TTProgram`
   聚合 / 清理点，
   不重新引入
   `tl.internal_tt_*`
   或 replacement staging bag
6. 确认 compile helper /
   target test helper
   不再调用
   `AnalyzeBlackholeComputeRegions`
   这类 public wrapper
7. 删除 public wrapper、
   registered pass、
   internal evidence helper、
   对应源文件和头文件
8. 删除或迁移
   `test_blackhole_flash_attention_analysis.py`
   里的剩余有效断言
9. 清理仍然保留的 legacy analysis compatibility residue

## 7. 相关文件

- `tilelang_repo/tilelang/transform/__init__.py`
- `tilelang_repo/tilelang/engine/lower.py`
- `tilelang_repo/testing/python/target/blackhole/common.py`
- `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- `tilelang_repo/src/transform/common/blackhole_lowering_requirements.h`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.h`
- `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.h`
- `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- `tilelang_repo/src/transform/common/compute_region_analysis.h`
- `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`
- `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

## 8. 验证要求

task2 的验证要证明
“legacy analysis 已经退出 active planning chain”，
不是只证明最终函数 attrs 更干净。

至少要覆盖：

1. `tilelang.transform`
   不再导出
   `AnalyzeBlackhole*`
2. active Blackhole path
   不再调用 public legacy wrapper
3. `lower_blackhole_ops.cc`
   和 `blackhole_lowering_requirements.cc`
   不再直接调用
   `AnalyzeBlackhole*Evidence(...)`
4. active planning chain
   不再依赖 broad `Map<String, Any>` lowering bag
   去做 legality / planner shaping / payload emission
   而且不再残留
   只存在于 bag 内、
   repo 内没有 active consumer
   的计数/摘要字段
5. `BuildTTProgram`
   后仍然不残留
   `tl.internal_tt_*`
   和
   `tl.blackhole_lowering_requirements_seed`
6. runtime / codegen / build
   继续只读
   `tl.blackhole_executable`
   /
   executable projection，
   不回退成 legacy analysis bag reader
7. 最终输出仍然不暴露
   `blackhole.lowering_requirements`
   `blackhole.work_decomposition`
   `blackhole.compute_regions`
   `blackhole.pipeline_stages`
8. legacy analysis tests
   已删除或迁移到 canonical path

## 9. 完成判据

只有下面这些同时成立，
task2 才算完成：

- public `AnalyzeBlackhole*` wrapper 已删除
- internal `AnalyzeBlackhole*Evidence(...)`
  helper 已删除
- active consumer 已切到当前 IR /
  `SpatialPlan` /
  当前 planner 内的 pass-local helper
- `blackhole_lowering_requirements.cc`
  不再承担 planning semantic carrier
- `BuildTTProgram`
  继续只做 staged
  `TTProgram`
  聚合 / 清理，
  没有新的
  `tl.internal_tt_*`
  或 replacement bag
- runtime / codegen / build
  不依赖 legacy analysis bag；
  剩余 payload/projection debt
  仍明确归 task3，
  `blackhole.segment_kind`
  residue
  仍明确归 task4
- legacy analysis 相关测试和 compatibility residue
  已同步清理
- task2 文档和代码口径一致，
  不再把“最终 attrs 更干净”误写成
  “legacy analysis 已退出主链”
