# Task 1: Replace Compute-Region Bags With Direct Logical Bridge Capture

## 0. `2026-04-22` 收口更新

- **状态**: `completed`
- repo HEAD 已成立：
  - `CaptureBlackholeLogicalBridgeSpecs`
    已接入
    `LowerToBlackholePhaseB`
  - `lower.py`
    / target test helper
    已直接读取
    `tl.blackhole_logical_buffer_tile_bridge_specs`
  - `AnalyzeBlackholeComputeRegions`
    / `blackhole.compute_regions`
    已退出 producer-side bridge handoff
- 本 task 剩余相关 debt
  已移动到：
  - cleanup task2：
    public/internal legacy analysis bag 删除
  - cleanup task3：
    `TTProgram.payload`
    / executable projection
    bridge compatibility residue 删除

> 下文 `2026-04-20` 的“当前状态 / 当前代码现实”
> 保留为任务开始前的快照，不再代表 repo HEAD 当前状态。

## 1. 任务目标

这个 cleanup task 只负责一件事：

> **把 leaf-local logical buffer/tile bridge handoff 的 owner truth
> 从 `blackhole.compute_regions`
> / `AnalyzeBlackholeComputeRegionEvidence`
> 切回当前阶段的 direct capture，
> 并把
> `tl.blackhole_logical_buffer_tile_bridge_specs`
> 收成唯一窄 temporary handoff。**

task1 要修正的是
**producer-side owner truth**，
不是一次性做完所有 bridge consumer 的删除。

因此 task1 的正确合同是：

1. source/helper -> optimized device func
   这段 bridge handoff
   不再从 broad compute-region bag 抽取；
2. 当前仍然存在的 downstream carrier
   如果还要临时保留，
   也必须以 direct capture 的结果为来源，
   不能再把 broad bag 当真源；
3. 文档不能把
   `buffer_tile_bridge_specs`
   重新洗白成新的长期协议对象。

## 2. 范围

这个 task 允许做的事：

- 在当前阶段直接 capture
  leaf-local logical bridge handoff
- 让
  `tl.blackhole_logical_buffer_tile_bridge_specs`
  充当 source/helper 和 optimized device func
  之间的唯一窄 temporary carrier
- 把当前 bridge-specific source
  从 `blackhole.compute_regions`
  切到 current-stage capture / narrow attr
- 把 repo HEAD 现存的 payload / projection / codegen
  bridge 路径
  明确写成 **必须删除** 的 downstream compatibility debt

这个 task 不负责的事：

- 删除整个 public / internal compute-region analysis 面
  - 这是 task2
- 删除
  `TTProgram.payload["buffer_tile_bridge_specs"]`
  / executable projection
  / codegen bridge reader
  - 这是 task3 的 leaf reader cutover
- 定义新的长期 bridge representation
- 新增 public analysis wrapper
- 把另一份 `Map<String, Any>` 或 payload
  包装成 replacement protocol

## 3. 当前状态 (`2026-04-20`)

当前 **不算完成**。

repo HEAD 当前的 bridge 链路仍然是：

```text
AnalyzeBlackholeComputeRegionEvidence
  -> AnalyzeBlackholeComputeRegions
  -> blackhole.compute_regions["buffer_tile_bridge_specs"]
  -> lower.py:_extract_logical_bridge_specs()
     / testing helper `_align_blackhole_device_symbol()`
  -> tl.blackhole_logical_buffer_tile_bridge_specs
  -> blackhole_lowering_requirements / lowering seed
  -> TTProgram.payload["buffer_tile_bridge_specs"]
  -> tl.blackhole_executable["buffer_tile_bridge_specs"]
  -> codegen_blackhole.cc
```

这条链路暴露了两个事实：

1. 上游 owner truth
   仍然是 broad compute-region bag
2. 下游 leaf/codegen
   仍然保留 bridge compatibility path

当前代码里已经成立的只有：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  attr key 已存在
- `BuildTTProgram`
  会在中间态结束时 strip
  `tl.blackhole_logical_buffer_tile_bridge_specs`
  和
  `tl.blackhole_lowering_requirements_seed`
- `rt_mod_blackhole.cc`
  并不是
  `buffer_tile_bridge_specs`
  的 reader
- 直接 bridge reader
  当前其实是
  `codegen_blackhole.cc`
  而不是 runtime module

但这些都不代表 task1 已完成。

当前真正的问题仍然是：

- [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L374)
  仍会先构造
  `AnalyzeBlackholeComputeRegions()(LowerToBlackholePhaseB(mod))`
  再对齐 optimized device func
- [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L221)
  还在从
  `blackhole.compute_regions`
  抽 bridge handoff
- [testing/python/target/blackhole/common.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/common.py#L99)
  以及相关 pipeline test
  仍通过 public wrapper
  + `_align_blackhole_device_symbol()`
  走同一条 broad-bag bridge 路径
- repo HEAD 里还没有
  `capture_blackhole_logical_bridge_specs.cc`
  这类 direct capture 实现
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1335)
  和
  [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1460)
  仍继续把 bridge specs
  当作 active consumer 输入
- [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py#L1317)
  与
  [test_blackhole_flash_attention_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py#L974)
  仍显式断言
  `TTProgram.payload["buffer_tile_bridge_specs"]`
  的形状，
  说明 leaf bridge path
  仍是 repo HEAD 下可观察的 compatibility surface

## 4. 当前代码现实

### 4.1 broad compute-region bag 仍然拥有 producer-side owner truth

当前 bridge spec 的构造
仍然发生在 compute-region analysis 里，
不是发生在当前 exact builtin / current-TIR capture 上。

直接证据：

- [buffer_tile_bridge_spec_utils.h](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/buffer_tile_bridge_spec_utils.h#L34)
  从 fragment layout 构造 bridge spec
- [tilelang_repo/tilelang/transform/__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py#L665)
  仍把 public wrapper
  写成
  “emits structured `blackhole.compute_regions`”
- [analyze_blackhole_compute_regions.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc#L176)
  把 bridge specs 写进 region bag
- [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L221)
  再把它复制到
  `tl.blackhole_logical_buffer_tile_bridge_specs`

所以现在 narrow attr
只是 broad bag 的复制品，
还不是 direct capture 的 owner truth。

而且这条
“public structured bag
 -> Python/helper 再读取”
的路径，
在 repo 里也没有成熟 backend 先例。

当前成熟 target pass
的主模式是：

- pass-local collector
- 或 narrow temporary attr
- 然后直接 rewrite / projection

不是把 broad analysis bag
写成公开协议面，
再由后续 helper
重新读取。

### 4.2 `buffer_tile_bridge_specs` 当前同时扮演两种角色

repo HEAD 里，
它不是单一角色。

它同时是：

1. **planning side channel**
   - [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1611)
     会把它装进
     `buffer_tile_bridge_specs_by_buffer_`
   - [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L7099)
     还会在 tiled republish materialization
     缺少
     `logical_row_width`
     / `logical_element_count`
     时拿它做 fallback
2. **leaf compatibility carrier**
   - [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L3958)
     把它写进
     `TTProgram.payload`
   - [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h#L235)
     把它投影到
     `tl.blackhole_executable`
   - [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L669)
     读取它生成 generic bridge code

而且 repo HEAD 里
它已经不只是“代码里还剩一个 reader”：

- [validate_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/validate_tt_program.cc#L106)
  仍把它当成
  `TTProgram.payload`
  的结构要求
- [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py#L1317)
  和
  [test_blackhole_flash_attention_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py#L974)
  仍直接断言 payload 内容

这说明这条 leaf bridge path
当前仍是 repo HEAD 可观察的 compatibility surface，
不能被描述成
“已经退化成纯内部死代码”。

因此 task1 文档不能把它写成
“只剩 leaf debt”
或
“只剩 planning source”
其中任意一种。

### 4.3 当前 builtin/current-TIR surface 还不足以直接删掉 leaf bridge path

repo HEAD 当前的 generic bridge codegen
不仅需要：

- offset
- num_elements
- row_width

还需要：

- `local_shape`
- inverse logical index expressions

这些信息当前仍通过
`BufferTileBridgeBinding`
进入 codegen，
见：

- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L2718)
- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L2872)
- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L3063)

而当前 builtin surface
并没有把这些量完整编码出来，
见：

- [builtin_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/tir/builtin_blackhole.cc#L489)

这意味着：

> **在 current-TIR / builtin surface
> 还没有补齐之前，
> task1 不能把 payload / projection / codegen
> bridge path 的删除
> 写成自己的完成判据。**

### 4.4 任务排序已经被其他权威文档固定

当前其它权威文档对职责划分是明确的：

- [cleanup 总览](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md#L63)
  把 task1 写成
  direct logical bridge capture
- [task2](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md#L136)
  负责删除 public / internal legacy analysis bag
- [task3_runtime_gate_and_workload_cutover.md](/root/dev/vibe_dsl/tasks/dev_design/task3_runtime_gate_and_workload_cutover.md#L19)
  明确把 leaf reader cutover
  放在 `ExecutableSpec / leaf reader` 边界
- [protocol audit](/root/dev/vibe_dsl/tasks/dev_design/blackhole_first_principles_protocol_audit.md#L205)
  也把 internal bridge payload
  的退场写在 leaf reader 清理侧

所以 task1 文档如果把
payload / projection / codegen reader
删除写进自己的 completion contract，
就会和当前权威入口冲突。

## 5. 基于审计结果修正后的任务内容

### 5.1 task1 只切 bridge handoff 的来源和 owner truth

task1 的正确要求是：

1. source/helper -> optimized device func
   之间的 bridge handoff
   不再来自
   `blackhole.compute_regions`
2. repo HEAD 里必须出现
   direct capture 的 current-stage 实现
   - 可以是 dedicated pass
   - 也可以是等价的 current-stage local collector
   - 但不能再是 broad bag copy
3. `tl.blackhole_logical_buffer_tile_bridge_specs`
   只允许作为这段 handoff 的窄 temporary carrier

如果当前阶段还无法稳定恢复所需 handoff，
结论只能是：

- 扩当前阶段表示 / 结构
- 显式 reject / unsupported

不能继续把 broad compute-region bag
留成真源。

### 5.2 `tl.blackhole_logical_buffer_tile_bridge_specs` 只是 cleanup exception

task1 文档还必须把这件事写清楚：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  是当前 cleanup
  允许存在的唯一窄 temporary handoff
- 它不是
  `SpatialPlan`
  / `TTProgram`
  / `ExecutableSpec`
  的长期表示
- 它也不是
  TT-Metal program /
  runtime contract
- 它不是新的 medium-term bridge layer
- 它当前存在的唯一理由，
  是 optimized/helper entry
  仍需要一段 leaf-local handoff

因此它的正确口径只能是：

> **当前允许的窄 exception，
> 不是被 task1 合法化的新协议层。**

一旦 task3
把 leaf bridge payload / projection / codegen reader
切掉，
或者等价的显式 leaf representation
已经承接这段信息，
这个 narrow attr
也必须一起退场。

### 5.3 task1 不负责删掉全部 downstream bridge consumer

task1 改完以后，
repo HEAD 如果仍然保留 downstream bridge carrier，
那也只能被写成
**尚未删除的过渡债务**，
不能被写成“已经合理存在的协议面”。

它们只能按下面口径描述：

- `blackhole_lowering_requirements.cc`
  / `lower_blackhole_ops.cc`
  里的 bridge consumer
  仍属于 active planning residue
  - 它们后续由 task2 继续收口
- `TTProgram.payload`
  / executable projection
  / codegen bridge reader
  仍属于 leaf compatibility debt
  - 它们后续由 task3 删除

这里的关键不是
“先在 task1 里把所有 reader 都实现性删光”，
而是：

> **这些 surviving consumer
> 不能再反向定义 owner truth。**

如果 task1 改完以后，
这些 consumer 仍然以 broad compute-region bag
充当真源，
那 task1 仍然算失败。

### 5.4 payload / projection / codegen 路径只能被写成待删 leaf compatibility debt

审计后的固定口径是：

- 它不是 `SpatialPlan` 语义
- 它不是 `TTProgram` 的长期 planning slice
- 它不是 TT-Metal runtime contract
- 它只是 repo HEAD 当前 leaf/codegen
  还没收口时的 compatibility debt
- 它的删除是强制终态，
  不是“如果以后方便再删”的可选优化
- `ValidateTTProgram`
  现在即使仍校验
  `buffer_tile_bridge_specs`
  payload 结构，
  也只说明这条路径
  仍是 live compatibility surface，
  不能把它合法化成长期边界
- TT-Metal 的 public target model
  也没有
  `buffer_tile_bridge_specs`
  这类稳定对象；
  target-facing contract
  仍然是显式
  `CB` / accessor / runtime-arg /
  launch construction

因此 task1 文档必须同时做到两点：

1. 不把它写成当前应该长期保留的协议对象
2. 也不把它误写成
   “task1 现在就必须删完”

而且这里的直接 reader
在 repo HEAD 下应明确写成：

- `codegen_blackhole.cc`
  是 bridge payload 的直接 leaf consumer
- `rt_mod_blackhole.cc`
  并不是
  `buffer_tile_bridge_specs`
  的直接 reader

这样 task1 文档就不会把
“runtime/codegen leaf debt”
写成一个模糊的大口袋。

### 5.5 后续删除这条 leaf bridge 路径的前提要写清楚

将来要删除
`TTProgram.payload -> executable projection -> codegen`
这条 bridge path，
必须先满足至少一条：

1. current-TIR / builtin surface
   已经能直接表达 codegen 所需 mapping
2. 或者显式 leaf representation
   已经拥有等价字段

在这之前，
不能把 repo HEAD 当前 leaf bridge path
描述成正确长期设计；
也不能把它的删除
提前压到 task1 上。

换句话说：

> **task1 不是在给 leaf bridge path
> 找新的合法身份，
> 而是在把它锁死成后续必须删除的 debt。**

## 6. 执行切片

1. 新增 direct logical bridge capture
   - 直接从当前阶段 capture
     leaf-local bridge handoff
   - 不再从
     `AnalyzeBlackholeComputeRegionEvidence`
     复制
2. 更新 [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py)
   和测试 helper
   - 不再通过
     `blackhole.compute_regions`
     提取 bridge handoff
3. 把当前 bridge-specific source
   从 broad bag
   切到 direct capture / narrow attr
   - 如果某些 consumer
     还要暂时保留，
     也只能消费这份 direct capture 结果
4. 不在 task1 里扩大 scope 去删除：
   - public/internal legacy analysis 面
   - `TTProgram.payload`
   - executable projection
   - codegen bridge reader
5. 同步文档和测试口径
   - 明确 task1 / task2 / task3 的分工
   - 明确 surviving leaf bridge path
     只是待删 debt，
     不是可接受长期边界

## 7. 相关文件

- `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
  - 计划中的 direct capture 文件；
    repo HEAD 当前还不存在
- `tilelang_repo/tilelang/engine/lower.py`
- `tilelang_repo/testing/python/target/blackhole/common.py`
- `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
- `tilelang_repo/src/transform/common/buffer_tile_bridge_spec_utils.h`
- `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/build_tt_program.cc`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md`
- `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

## 8. 验证要求

task1 的验证必须证明：

> **bridge handoff 的 owner truth
> 已经不再来自 broad compute-region bag。**

至少要覆盖：

1. `engine/lower.py`
   / compile helper
   / 测试 helper
   不再从
   `blackhole.compute_regions`
   抽取 bridge handoff
2. repo HEAD 已经有 current-stage 的 direct capture 实现，
   而不是 Python 复制 broad bag
3. `tl.blackhole_logical_buffer_tile_bridge_specs`
   成为唯一窄 temporary handoff
4. 当前仍然保留的 downstream bridge consumer
   不再把 broad compute-region bag
   当真源
5. 文档明确写出：
   - task2 删除 public / internal legacy analysis bag
   - task3 删除 payload / projection / codegen bridge path
   - 这条 leaf bridge path
     只是必须删除的过渡债务，
     不是被 task1 合法化的新边界
   - `tl.blackhole_logical_buffer_tile_bridge_specs`
     只是 cleanup exception，
     不是长期 bridge layer
6. malformed bridge handoff
   仍会在现有 validator / codegen 边界
   fail closed

## 9. 完成判据

只有下面这些同时成立，
task1 才算完成：

- bridge handoff
  已不再从
  `blackhole.compute_regions`
  / `AnalyzeBlackholeComputeRegionEvidence`
  抽取
- repo HEAD 已有 direct current-stage capture，
  不再靠 broad bag copy
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  已成为 source/helper -> optimized device func
  之间唯一窄 temporary carrier
- `tilelang.engine.lower.lower()`
  和测试 helper
  已不再依赖
  `AnalyzeBlackholeComputeRegions`
  / `blackhole.compute_regions`
  去对齐 bridge handoff
- surviving downstream bridge path
  已不再被写成 owner truth
  或长期协议对象
- surviving leaf bridge path
  已被明确锁定为
  后续必须删除的 debt，
  不是 repo HEAD 下被认可的新协议面
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  已被明确写成 cleanup exception，
  不是新的中期协议层
- task1 文档已经和
  cleanup 总览 / task2 / task3 / protocol audit
  保持一致：
  - task1 只切 owner truth
  - task2 再删 legacy analysis bag
  - task3 再删 leaf bridge reader / internal bridge payload
