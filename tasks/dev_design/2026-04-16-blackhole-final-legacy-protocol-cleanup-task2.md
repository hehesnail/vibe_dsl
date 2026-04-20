# Task 2: Remove Public And Internal Legacy Analysis Bags

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
- 顶层 `Task 2: TTProgram Representation Cutover`
  的全部工作

## 3. 当前状态 (`2026-04-17`)

当前 **不算完成**。

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
- `lower_blackhole_ops.cc`
  和 `blackhole_lowering_requirements.cc`
  仍然直接消费
  `AnalyzeBlackhole*Evidence(...)`
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
和 [testing/python/target/blackhole/common.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/common.py#L97)。

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

### 4.3 `blackhole.lowering_requirements` attr 虽然不见了，但 lowering bag 还活着

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

然后 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1460)
再把它作为 active planning 输入继续消费。

也就是说：
`blackhole.lowering_requirements`
这个 legacy carrier
只是从“最终 attr”
退回成了“internal helper return value”，
并没有真正退出 active planning chain。

### 4.4 task2 当前还受 task1 卡住

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

### 4.5 还有历史兼容代码在改写这些 legacy attr

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

如果这个文件在 task2 后仍然保留，
它只能做：

- leaf-local projection helper
- leaf-local validation helper
- 极窄的 current-stage local utility

它不能再承担：

- 汇总 work / compute / pipeline / bridge / materialization / flow
  的统一 semantic bag

### 5.4 task1 / task2 边界要写清楚

task2 文档里必须明确：

- task1 负责把 bridge handoff
  从 compute-region wrapper 上切走
- task2 才负责删除
  `AnalyzeBlackholeComputeRegions`
  这类 public / internal analysis 面

不能把这两层混成一个 task，
也不能拿 task1 未完成
来把 task2 的真正合同写虚。

### 5.5 旧 analysis 测试现在还不是“已经 obsolete”

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
   直接读取当前 `PrimFunc` / `SpatialPlan` / `TTProgram`
   或极窄的 local helper，
   不再返回新的 broad `Map<String, Any>` bag
4. 收缩或删除
   `blackhole_lowering_requirements.cc`
   的 planning semantic-carrier 角色
5. 删除 public wrapper、
   registered pass、
   internal evidence helper、
   对应源文件和头文件
6. 删除或迁移
   `test_blackhole_flash_attention_analysis.py`
   里的剩余有效断言
7. 清理仍然保留的 legacy analysis compatibility residue

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
5. 最终输出仍然不暴露
   `blackhole.lowering_requirements`
   `blackhole.work_decomposition`
   `blackhole.compute_regions`
   `blackhole.pipeline_stages`
6. legacy analysis tests
   已删除或迁移到 canonical path

## 9. 完成判据

只有下面这些同时成立，
task2 才算完成：

- public `AnalyzeBlackhole*` wrapper 已删除
- internal `AnalyzeBlackhole*Evidence(...)`
  helper 已删除
- active consumer 已切到当前 IR / 显式表示层
- `blackhole_lowering_requirements.cc`
  不再承担 planning semantic carrier
- legacy analysis 相关测试和 compatibility residue
  已同步清理
- task2 文档和代码口径一致，
  不再把“最终 attrs 更干净”误写成
  “legacy analysis 已退出主链”
