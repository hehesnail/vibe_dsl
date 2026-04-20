# Task 1: Replace Compute-Region Bags With Direct Logical Bridge Capture

## 1. 任务目标

`Task 1` 只做一件事：

> **切掉 `blackhole.compute_regions` / compute-region evidence
> 对 leaf-local logical buffer/tile correspondence 的 owner truth。**

这里要切掉的不是“某个测试里的 bridge spec 字段名”，
而是 broad compute-region bag
在 active chain 上偷偷承担的跨阶段语义角色。

task1 完成后必须同时满足：

1. optimized/helper 路径需要的 leaf-local logical bridge handoff
   不再从 `blackhole.compute_regions` 提取；
2. 这份 handoff
   在 transform pipeline 内
   只能通过
   `tl.blackhole_logical_buffer_tile_bridge_specs`
   这个 narrow temporary attr 传递；
3. `buffer_tile_bridge_specs`
   不会被文档重新洗白成新的长期协议对象。

## 2. 这份语义到底属于哪一层

这次必须先把层边界讲清楚，
否则 task1 永远会滑回
"把当前实现面修辞合法化"。

### 2.1 它不属于 `SpatialPlan`

这份 bridge 语义
不是 target-independent 的
virtual spatial/dataflow 表示。

它不回答：

- execution unit 是什么
- unit 间 dataflow 是什么
- virtual layout / phase / carry / reduction 是什么

所以它不是 `SpatialPlan`
应该长期承载的对象。

### 2.2 它也不是 `TTProgram` 的长期 planning slice

这份 bridge 语义
也不是 target realization
本身的长期 slice。

它不回答：

- block/core placement
- transport / sync / ABI / execution
- TT kernel family / role

因此它也不应该被包装成
新的 `TTProgram` planning noun。

### 2.3 它本质上是当前 leaf path 的兼容债

repo HEAD 里之所以还有这份 bridge 语义，
是因为当前优化后 device func /
projection / codegen
之间还没有把 logical buffer/tile identity
完整落到 typed leaf representation，
导致 codegen 仍然需要一份
leaf-local logical correspondence。

所以正确口径只能是：

- **task1 只修正它的来源和 owner truth**
- **不授予它长期表示层地位**

### 2.4 长期终态固定为后续 leaf cutover 义务

task1 不负责完成长期终态，
但必须把终态义务写死。

长期只能二选一收口：

1. 默认方案：
   codegen 所需 bridge 语义
   能从 typed leaf representation
   稳定导出；
2. 只有在确实无法稳定导出时：
   才允许把它升级成
   typed leaf field / object。

无论哪种情况，
`buffer_tile_bridge_specs`
都不是应该被永久保留的协议对象名。

## 3. 当前状态 (`2026-04-20`)

当前 **不算完成**。

repo HEAD 已经有的局部现实包括：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  attr key 已定义；
- `blackhole_lowering_requirements.cc`
  已能读取这个 attr；
- `build_tt_program.cc`
  会在 TTProgram finalization 时
  strip 掉这个 temporary attr；
- `lower_blackhole_ops.cc`
  会把 bridge specs
  放进 `TTProgram.payload`；
- `tt_program_projection.h`
  会把 payload 上的
  `buffer_tile_bridge_specs`
  投影到 `tl.blackhole_executable`；
- `codegen_blackhole.cc`
  还会继续消费这份 projection；
- payload 侧已经有正向测试。

但这些都只是
**当前兼容实现面**，
不是 task1 已经完成的证明。

当前真正的问题仍然是：

- [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L221)
  还在从
  `blackhole.compute_regions`
  提取 `buffer_tile_bridge_specs`；
- [common.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/common.py#L97)
  的测试 helper
  也还在走同一路径；
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L109)
  和
  [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1749)
  里仍有 bridge-specific consumer
  直接读
  `AnalyzeBlackholeComputeRegionEvidence(func)`。

所以 repo HEAD 当前只能写成：

> narrow attr 的消费落点已接好

不能写成：

> direct logical bridge capture 已成立

## 4. 当前失败模式

### 4.1 broad bag 仍然拥有 bridge handoff 的来源

当前 active 路径仍是：

1. `LowerToBlackholePhaseB`
2. `AnalyzeBlackholeComputeRegions`
3. 从 `blackhole.compute_regions`
   抽 `buffer_tile_bridge_specs`
4. 再通过 `_align_blackhole_device_symbol()`
   贴回 optimized device func

这意味着：

- narrow attr 不是 producer-side owner truth
- broad bag 才是

这正是 task1 要切掉的东西。

### 4.2 “最终 `PrimFunc` 看不到 `blackhole.compute_regions`” 不是完成证明

最终 active-chain `PrimFunc`
通常不会暴露
`blackhole.compute_regions`。

但这只说明 broad bag
没有被留到最终产物里；
并不说明中途没有拿它做 owner truth。

当前 repo HEAD 的问题
恰恰就是：

- 它仍然在中途承担 owner truth
- 只是最终被丢掉了

### 4.3 payload / projection 只是现状，不是 task1 的背书对象

repo HEAD 当前还存在这条链：

```text
tl.blackhole_logical_buffer_tile_bridge_specs
  -> TTProgram.payload["buffer_tile_bridge_specs"]
  -> tl.blackhole_executable["buffer_tile_bridge_specs"]
  -> codegen
```

task1 必须明确：

- 这是当前 admitted compatibility path
- 不是新的长期设计边界
- 更不是 `bridge spec`
  已经找到长期 owner truth

## 5. Task1 收口后的正确合同

### 5.1 producer-side handoff 只能是 narrow temporary attr

对 transform pipeline 内部的
leaf-local bridge handoff 来说，
唯一允许的 temporary carrier
只能是：

- `tl.blackhole_logical_buffer_tile_bridge_specs`

它只能做一件事：

- 在 source/helper/optimized device func
  之间传递当前仍需要的
  logical buffer/tile correspondence

它不能变成：

- planning representation
- public analysis wrapper
- broad bag 替身
- 新的长期协议对象

### 5.2 broad compute-region bag 不能再作为 bridge 来源

task1 完成后：

- `engine/lower.py`
  不能再从
  `blackhole.compute_regions`
  提取 bridge handoff；
- 测试 helper
  不能再复用这个 broad bag 路径；
- `blackhole_lowering_requirements.cc`
  里 broad bag 合并 bridge spec 的逻辑
  必须退出 bridge-source 角色；
- 凡是只为了 bridge handoff /
  leaf-local logical correspondence
  而读
  `AnalyzeBlackholeComputeRegionEvidence(func)`
  的代码，
  都必须切走。

task1 不是要求
“所有 compute-region evidence 读取归零”，
而是要求：

> **bridge handoff 不再由 broad bag 拥有。**

### 5.3 capture pass 只能是 pass-local mechanics

task1 改完后，
仓库里必须有 dedicated capture pass，
例如：

- `capture_blackhole_logical_bridge_specs.cc`

它的职责固定是：

- 直接遍历当前 `PrimFunc`
- 基于当前 IR 结构、
  buffer/binding/type/attr
  这些显式信息
  捕获 leaf-local logical bridge handoff
- 只写出
  `tl.blackhole_logical_buffer_tile_bridge_specs`

它明确不能：

- 顺便写 broad compute-region bag
- 依赖名字匹配恢复语义
- 变成新的 public analysis object
- 变成共享 helper layer

如果当前 `PrimFunc`
还不足以稳定恢复这份 handoff，
结论只能是：

- 上游显式表示还不够
- 需要补表示 / validator / reject

而不是继续保留
`blackhole.compute_regions`
作为隐性真源。

## 6. task1 / task2 / task3 的责任边界

### 6.1 task1 负责什么

task1 只负责：

- 切掉 broad compute-region bag
  对 bridge handoff 的 owner truth
- 建立 direct logical bridge capture
- 把 narrow attr
  明确收成 producer-side temporary carrier
- 把当前 payload / projection 路径
  明确记成 compatibility debt

### 6.2 task2 负责什么

task2 才负责：

- 删除 public/internal legacy analysis bag
- 删除 mixed compute-region consumer
- 把 compute-region stack
  彻底退成 history / local mechanics

所以 task1 不能假装
“所有 evidence 读取都应在本轮归零”。

### 6.3 task3 负责什么

task3 才负责真正的 leaf 收口：

- 让 build / codegen / runtime
  严格回到 `ExecutableSpec`
  reader discipline
- 删除当前 payload / projection
  上仍残留的 compatibility debt
- 完成 bridge 语义的长期终态
  二选一收口：
  - 默认：从 typed leaf representation
    稳定导出
  - 不可导出时：升级成 typed leaf field / object

所以 task1 必须把这条义务写死，
但不能假装自己已经把 leaf end-state 做完。

## 7. 当前 compatibility debt 的纪律

repo HEAD 当前仍有：

- `TTProgram.payload["buffer_tile_bridge_specs"]`
- `tl.blackhole_executable["buffer_tile_bridge_specs"]`

这两层在 task1 里的正确口径只有一个：

> **当前 admitted compatibility debt**

这意味着：

- runtime `ExecutableSpec`
  不是 bridge spec owner；
- 当前 downstream consumer
  只是 `codegen_blackhole.cc`；
- 不能新增新的 reader
  继续消费这条 payload / projection path；
- 不能把这条路径
  写成长期表示层对象；
- 后续 leaf cutover
  必须把这条路径删除掉。

## 8. 结构约束

task1 虽然不把 `bridge spec`
当长期对象，
但当前 compatibility carrier
仍然必须受显式结构约束。

至少要继续满足 repo HEAD
已经存在的 gate：

- validator 要求每个 spec
  至少包含
  `buffer`
  `shape`
  `local_shape`
- codegen generic bridge
  还要求
  `local_shape`
  是 1-D，
  并要求 inverse logical index
  相关表达满足当前可投影形式

这说明：

- 当前 attr / payload
  仍然不能退化成任意 bag
- 但“有结构约束”
  不等于“它已经成了长期表示对象”

## 9. 执行切片

1. 新增 direct logical bridge capture pass，
   只从当前 `PrimFunc`
   产出
   `tl.blackhole_logical_buffer_tile_bridge_specs`
2. 更新 `engine/lower.py`
   和测试 helper，
   不再通过
   `AnalyzeBlackholeComputeRegions`
   提取 bridge handoff
3. 把 bridge-specific consumer
   从 broad compute-region evidence
   切到：
   - narrow temporary attr
   - 当前 IR 局部恢复
   - 或现有 compatibility path
     的最小必需消费
4. 明确把
   `TTProgram.payload / executable projection`
   上的 bridge 路径
   标成 compatibility debt，
   并把后续删除方向写进文档
5. 保持 task2 / task3 的边界，
   不把 mixed consumer deletion
   和 leaf end-state
   偷偷塞进 task1

## 10. 相关文件

- `tilelang_repo/src/transform/capture_blackhole_logical_bridge_specs.cc`
- `tilelang_repo/tilelang/engine/lower.py`
- `tilelang_repo/tilelang/engine/phase.py`
- `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/common/companion_base.h`
- `tilelang_repo/src/transform/build_tt_program.cc`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/testing/python/target/blackhole/common.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`

## 11. 验证要求

task1 的验证必须证明：

> **bridge handoff 的真源已经切换**

不是只证明
“最后 payload 里还有 bridge specs”。

至少要覆盖：

1. optimized/helper path
   仍能保留正确的
   `buffer_tile_bridge_specs`
2. `engine/lower.py`
   和测试 helper
   都不再依赖
   `AnalyzeBlackholeComputeRegions`
   提取 bridge handoff
3. dedicated capture pass
   只写出
   `tl.blackhole_logical_buffer_tile_bridge_specs`
4. broad compute-region bag
   不再作为 bridge-source
   合并进 lowering path
5. 最终 active-chain `PrimFunc`
   仍不暴露
   `blackhole.compute_regions`
6. `build_tt_program.cc`
   仍会在中间态结束时
   strip 掉 temporary attr
7. `TTProgram.payload`
   和 executable projection
   仍能承接当前 compatibility path
8. malformed bridge payload
   继续在 validator / codegen
   边界 fail closed
9. 文档明确把
   payload / projection
   标成 compatibility debt
10. 文档明确把长期终态
    固定到 task3 的
    typed leaf 收口义务上

## 12. 完成判据

只有下面这些同时成立，
task1 才算完成：

- dedicated direct capture pass 已存在
- bridge handoff 已不再从 `blackhole.compute_regions` 抽取
- broad compute-region bag
  已退出 bridge owner truth
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  已成为 transform pipeline
  内唯一 producer-side temporary carrier
- `TTProgram.payload / executable projection`
  只被定义成当前 compatibility debt，
  不被定义成长期表示边界
- runtime `ExecutableSpec`
  不再被文档或代码当成 bridge owner
- task1 已把长期终态义务
  明确压给后续 leaf cutover：
  默认导出，
  不可导出时才升级成 typed leaf field / object
- task1 文档和代码口径一致，
  不再把“消费落点已接好”
  误写成“capture 已完成”，
  也不再把
  `buffer_tile_bridge_specs`
  描述成应该长期保留的协议对象
