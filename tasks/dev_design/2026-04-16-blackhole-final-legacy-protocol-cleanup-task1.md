# Task 1: Replace Compute-Region Bags With Direct Logical Bridge Capture

## 1. 任务目标

这个 task 只负责一个窄收口：
把 optimized/helper 路径里需要的
logical buffer/tile bridge handoff
收成唯一一个 producer-side leaf-local temporary carrier：

- `tl.blackhole_logical_buffer_tile_bridge_specs`

它的目标不是删除整个 compute-region 分析体系，
而是让 bridge handoff 不再通过
`blackhole.compute_regions`
这类 broad bag 在主链里传递。

这个 task 的 owner-truth 约束还包括：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  只负责 transform pipeline 内部
  source/helper/optimized device func
  之间的 bridge handoff；
- downstream durable carrier
  不是这个临时 attr 本身，
  而是
  `TTProgram.payload["buffer_tile_bridge_specs"]`
  以及后续
  `tl.blackhole_executable["buffer_tile_bridge_specs"]`
  projection；
- runtime `ExecutableSpec`
  不是 bridge spec 的 owner，
  也不直接消费这份语义；
  codegen 才是 downstream bridge-spec consumer。

## 2. 范围

这个 task 允许做的事：

- 新增一个 dedicated capture pass，
  直接从当前 `PrimFunc`
  捕获 logical bridge specs
- 让 optimized/helper 路径只消费这个窄 attr 完成 bridge handoff
- 把 bridge-specific consumer
  从 broad compute-region bag 上切走
- 把 bridge spec 的 downstream carrier
  明确收口到
  `TTProgram.payload`
  和 executable projection，
  而不是让临时 attr
  偷偷长成新的长期协议

这个 task 不负责的事：

- 删除 `AnalyzeBlackholeComputeRegions`
  public wrapper
- 删除 `AnalyzeBlackholeComputeRegionEvidence`
  internal helper
- 删除整个 compute-region / pipeline-stage / work-decomposition 分析栈
- 完成顶层 `Task 1: SpatialPlan Representation Cutover`

这些删除类动作属于后续 cleanup task，
尤其是 `Cleanup Task 2`。

更具体地说：

- `engine/lower.py`
  和测试 helper
  从 `blackhole.compute_regions`
  抽 bridge specs
  属于 task1 必须切掉的桥接旁路；
- `blackhole_lowering_requirements.cc`
  里把 broad compute-region bag
  当作 bridge-spec 来源的逻辑，
  属于 task1 必须切掉的 bridge-source merge；
- compute-op kind /
  reduction / broadcast /
  non-bridge logical-shape recovery /
  physical-buffer binding
  这类 mixed consumer
  仍属于 task2 及后续 cleanup，
  不能在 task1
  里被笼统写成“一并删除”。

## 3. 当前状态 (`2026-04-20`)

当前 **不算完成**。

repo HEAD 上已经有的局部铺垫包括：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  attr key 已定义；
- `blackhole_lowering_requirements.cc`
  已能读取这个 attr；
- `build_tt_program.cc`
  会在 TTProgram finalization 时
  把这个 temporary attr strip 掉；
- `lower_blackhole_ops.cc`
  已会把 bridge specs
  存进 `TTProgram.payload`；
- `tt_program_projection.h`
  已会把
  `TTProgram.payload["buffer_tile_bridge_specs"]`
  投影到
  `tl.blackhole_executable`；
- `codegen_blackhole.cc`
  已从 executable projection
  消费 bridge specs；
- payload 侧已经有 bridge spec 的正向测试。

但 task1 要求的主合同还没有成立：

- 仓库里还没有
  `capture_blackhole_logical_bridge_specs.cc`
  这样的 dedicated capture pass；
- `engine/lower.py`
  仍然先跑
  `AnalyzeBlackholeComputeRegions()(LowerToBlackholePhaseB(mod))`
  生成平行的 `analysis_mod`；
- 之后再从
  `analysis_mod.attrs["blackhole.compute_regions"]`
  里抽
  `buffer_tile_bridge_specs`；
- 然后通过
  `_align_blackhole_device_symbol()`
  把这些 bridge specs 回贴到 optimized device func 上；
- `blackhole_lowering_requirements.cc`
  和 `lower_blackhole_ops.cc`
  里仍有 bridge/buffer recovery
  直接重跑 `AnalyzeBlackholeComputeRegionEvidence(func)`。

所以当前只能写成：

> 窄 bridge attr 已有消费落点

不能写成：

> direct logical bridge capture 已完成

## 4. 当前代码现实

### 4.1 当前 bridge handoff 仍通过 broad bag 旁路提取

当前 [engine/lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L221)
和 [engine/lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py#L369)
里的实际链路是：

1. `LowerAndLegalize`
2. `LowerToBlackholePhaseB`
3. `AnalyzeBlackholeComputeRegions`
4. 从 `blackhole.compute_regions`
   中抽取 `buffer_tile_bridge_specs`
5. 通过 `_align_blackhole_device_symbol()`
   把窄 attr 回贴到 optimized device func

这说明当前真正承载 bridge handoff 数据的来源
仍然是 broad compute-region bag，
不是 direct capture pass。

### 4.2 窄 attr 现在只是 temporary carrier，不是 bridge 语义的完整生命周期描述

当前窄 attr 的落点已经存在：

- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1145)
  会合并这个 attr 上的 bridge specs；
- [build_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/build_tt_program.cc#L37)
  会在中间态结束时 strip 掉它；
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L3964)
  会把 bridge specs
  放进 `TTProgram.payload`；
- [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h#L245)
  会把 payload 上的
  `buffer_tile_bridge_specs`
  投影到 `tl.blackhole_executable`；
- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L669)
  会从 executable projection
  读取 bridge specs。

这说明当前真正完整的表示生命周期是：

```text
temporary producer-side attr
  -> TTProgram.payload["buffer_tile_bridge_specs"]
  -> tl.blackhole_executable["buffer_tile_bridge_specs"]
  -> codegen consumer
```

因此 task1 不能只写
“leaf-local attr 会被 strip 掉”，
还必须写清：

- 被 strip 掉的是 temporary producer-side attr
- 不是 bridge spec 本身的 downstream contract

否则就会把“删临时 attr”
误写成“bridge spec 在 TTProgram 之前就该消失”。

### 4.3 bridge-specific consumer 仍然直接依赖 compute-region evidence

当前只要是为了 bridge specs /
logical shape / leaf-local logical correspondence
而读 broad compute-region evidence 的地方，
都说明 task1 还没有收口。

现在能直接看到的地方包括：

- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L109)
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc#L1029)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1745)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L2045)

这里需要特别区分两类读取：

- **task1 必切**
  - 任何把 broad compute-region bag
    当成 bridge-spec 来源的读取
  - 任何只为了 bridge handoff /
    leaf-local logical correspondence
    而读取 broad evidence 的代码
- **task2 以后再切**
  - compute-op classification
  - pointwise / reduction / broadcast
  - non-bridge logical shape 恢复
  - physical buffer binding
  - 其他仍然混在 compute-region stack
    里的 residual semantics

task1 的目标不是“所有 evidence 读取归零”，
而是“bridge handoff 不再由 broad bag 拥有”。

### 4.4 runtime / codegen 边界不能混写

当前 downstream consumer
并不是一个笼统的
“runtime/codegen”整体。

repo HEAD 的现实是：

- `rt_mod_blackhole.cc`
  并不解析 bridge specs；
- `ExecutableSpec`
  也没有 dedicated bridge-spec field；
- bridge specs 的 downstream consumer
  是 `codegen_blackhole.cc`
  通过 executable projection
  读取后的 codegen 逻辑。

因此 task1 文档里凡是提到
bridge spec downstream contract 的地方，
都必须明确写成：

- `TTProgram.payload`
  / executable projection / codegen

而不能写成：

- runtime 侧也拥有同一份 bridge-spec owner truth

### 4.5 最终输出不带 `blackhole.compute_regions` 不是完成证明

现在最终 active-chain `PrimFunc`
通常不会再暴露
`blackhole.compute_regions`。

这只能说明：

- broad bag 没被留到最终输出

不能说明：

- 主链已经不再依赖它做 bridge handoff

当前真实情况是：
它仍然在旁路分支里被使用，
只是在最终产物里被丢掉了。

## 5. 基于审计结果修正后的任务内容

### 5.1 落 dedicated capture pass

task1 改完之后，
仓库里必须有一个专门的 pass
负责 direct logical bridge capture。

这个 pass 只能做一件事：

- 从当前 `PrimFunc`
  基于当前 IR 结构、
  buffer/binding/type/attr
  这类显式信息，
  捕获 optimized/helper 路径所需的 logical bridge specs
- 只写出
  `tl.blackhole_logical_buffer_tile_bridge_specs`

它不能：

- 顺便写 broad compute-region bag
- 依赖名字匹配恢复语义
- 变成新的 public analysis bag
- 变成 planning representation

如果当前 `PrimFunc`
还不足以稳定恢复 bridge spec，
那说明缺的是上游显式表示，
而不是继续保留
`blackhole.compute_regions`
作为旁路 owner truth。

### 5.2 bridge handoff 不能再从 `blackhole.compute_regions` 抽取

task1 完成后，
`engine/lower.py`
和测试 helper
都不能再从
`blackhole.compute_regions`
里抽
`buffer_tile_bridge_specs`。

如果 source 分支到 optimized 分支的搬运逻辑还暂时保留，
它也只能搬运 dedicated capture pass
写出来的窄 attr。

### 5.3 窄 attr 只负责 producer-side handoff，downstream durable carrier 单独收口

对 bridge handoff 这件具体事情来说，
transform pipeline 内部
唯一允许的 producer-side temporary carrier
只能是：

- `tl.blackhole_logical_buffer_tile_bridge_specs`

不能继续维持：

- broad compute-region bag 里也带一份 bridge specs，
  consumer 再从那边优先读取

与此同时，
downstream durable carrier
必须明确写成：

- `TTProgram.payload["buffer_tile_bridge_specs"]`
- `tl.blackhole_executable["buffer_tile_bridge_specs"]`

task1 不把这两层混成一个概念：

- temporary attr 可以被 strip
- payload / executable projection
  仍然必须保留 bridge-spec 语义
- runtime `ExecutableSpec`
  仍然不拥有 bridge spec

### 5.4 只为 bridge 而存在的 broad bag 读取必须切走

task1 完成后，
凡是“只为了 bridge handoff”
而去读
`AnalyzeBlackholeComputeRegionEvidence(func)`
的代码都必须切走。

这里的边界是：

- 为 bridge handoff 服务的 broad analysis 读取必须消失

不是：

- 整个 compute-region analysis 栈必须在 task1 里一起删除

### 5.5 bridge spec schema 仍然要有显式结构约束

task1 的 capture contract
不能只是“塞一组 map 进去”。

至少要继续满足 repo HEAD
已经存在的 downstream structural gate：

- validator 侧要求每个 spec
  至少有
  `buffer`
  `shape`
  `local_shape`
- codegen 侧对 generic bridge
  还要求
  `local_shape`
  是 1-D，
  并要求 inverse logical index
  相关表达满足当前可投影形式

这意味着：

- task1 可以让 producer-side carrier
  继续用 attr 暂存
- 但不能把 bridge spec
  退化成一个无 schema 的随意 bag

### 5.6 保持 leaf-local temporary attr 的边界

`tl.blackhole_logical_buffer_tile_bridge_specs`
仍然只允许是一个 leaf-local temporary attr：

- 只给 optimized/helper path 做 handoff
- 不是新的 planning representation
- 不是新的 analysis bag
- 在最终 `PrimFunc`
  收口时 strip 掉

但这里要同时明确：

- 被 strip 的只是 temporary attr
- 不是 `TTProgram.payload`
  / executable projection
  上的 downstream bridge-spec carrier

## 6. 执行切片

1. 新增 dedicated capture pass，
   直接从当前 `PrimFunc`
   产出
   `tl.blackhole_logical_buffer_tile_bridge_specs`
2. 更新 `engine/lower.py`
   和测试 helper，
   不再通过
   `AnalyzeBlackholeComputeRegions`
   提取 bridge specs
3. 把 bridge-specific consumer
   从 broad compute-region evidence
   切到窄 attr /
   downstream payload contract /
   当前 IR 局部恢复
4. 保持 task2 的边界：
   broad compute-region 分析体系本身可以暂时存在，
   但不再承担 bridge handoff 主职责
5. 重新校正文档和测试，
   确保
   temporary carrier /
   durable carrier /
   runtime 非 owner
   这三件事都写实

## 7. 相关文件

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

## 8. 验证要求

task1 的验证要证明
“bridge handoff 的来源已经切换”，
不是只证明 payload 里最后还有 bridge specs。

至少要覆盖：

1. optimized/helper path
   仍能保留正确的 `buffer_tile_bridge_specs`
2. `engine/lower.py`
   和测试 helper
   都不再依赖
   `AnalyzeBlackholeComputeRegions`
   提取 bridge specs
3. dedicated capture pass
   只写出
   `tl.blackhole_logical_buffer_tile_bridge_specs`
4. broad compute-region bag
   不再作为 bridge-spec 来源
   被 `blackhole_lowering_requirements.cc`
   合并进 bridge contract
5. 最终 active-chain `PrimFunc`
   仍不暴露
   `blackhole.compute_regions`
6. `build_tt_program.cc`
   仍会在中间态结束时
   strip 掉这个 temporary attr
7. `TTProgram.payload`
   仍保留
   `buffer_tile_bridge_specs`
8. executable projection /
   codegen
   仍能消费 bridge specs
9. malformed bridge spec
   会继续在 validator / codegen
   边界 fail closed

## 9. 完成判据

只有下面这些同时成立，
task1 才算完成：

- dedicated direct capture pass 已存在
- bridge handoff 已不再从 `blackhole.compute_regions` 抽取
- bridge-specific consumer 已从 broad compute-region bag 上切走
- 窄 attr
  已成为 transform pipeline
  内唯一的 producer-side temporary carrier
- `TTProgram.payload`
  / executable projection
  已成为唯一 downstream durable carrier
- runtime `ExecutableSpec`
  不再被文档或代码当成 bridge-spec owner
- task1 文档和代码口径一致，
  不再把“消费落点已接好”误写成“capture 已完成”，
  也不再把 temporary attr
  和 downstream payload/projection
  混成同一个表示层对象
