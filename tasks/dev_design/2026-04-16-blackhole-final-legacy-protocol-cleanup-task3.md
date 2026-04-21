# Task 3: Replace `blackhole.copy_semantics` With Direct Current-Stage Recovery

## 1. 任务目标

这个 cleanup task 只负责一件事：
把 `blackhole.copy_semantics`
从 active chain 里删掉，
让仍然依赖它的 compiler-side consumer
回到各自当前阶段能合法看到的
IR / 显式表示层 / pass-local derived facts。

完成后不再允许：

- `AnnotateBlackholeCopySemantics`
  作为 active prepass
- `blackhole.copy_semantics`
  作为跨 pass 语义 carrier
- consumer 通过读 loop annotation 里的
  `kind` / `direction` / `src_buffer` / `dst_buffer`
  来决定 canonicalization / split / lowering 行为

这里的关键词是
**current-stage recovery**，
不是“再找一个统一的新共享 copy contract”。

## 2. 范围

这个 task 允许做的事：

- 在每个 consumer 所在阶段，
  直接根据当前阶段已存在的
  IR / 显式表示层 / pass-local analysis
  恢复 copy 含义
- 删除 `AnnotateBlackholeCopySemantics`
  prepass、public transform 和实现文件
- 删除围绕 annotation schema 写的过时测试
- 更新 active path 中依赖该 annotation 的 consumer
- 把测试 staging
  调整成和 mainline phase ordering 一致

这个 task 不允许做的事：

- 再引入新的 shared copy-analysis layer
- 再引入新的 copy-direction attr / helper bag / exported matcher
- 把 `blackhole.copy_semantics`
  换个名字继续跨 pass 传
- 把所有 consumer
  粗暴写成“统一从 `SpatialPlan` 读取”
- 用仍然存在的 leaf payload / projection /
  `blackhole.segment_kind`
  残留，反过来合法化
  `blackhole.copy_semantics`

这里要特别注意：

- 不是所有 consumer 都处在同一个表示层
- 不是所有 consumer 都能看到 `SpatialPlan`
- target / codegen / build / runtime
  当前已经不是直接读
  `blackhole.copy_semantics`
  的 reader boundary

所以 task3 的正确合同不是
“统一从 `SpatialPlan.DataflowEdge` 恢复”，
而是：

> 每个 consumer 只能依赖它在当前阶段本来就能合法看到的
> IR / 显式表示层，不能依赖 legacy copy annotation。

## 3. 当前状态 (`2026-04-21`)

当前 **不算完成**。

repo HEAD 上，
`blackhole.copy_semantics`
仍然是 active compiler-side protocol：

- `phase.py`
  在 Blackhole lowering 主链里仍显式运行
  `AnnotateBlackholeCopySemantics()`
- `tilelang.transform`
  仍公开导出
  `AnnotateBlackholeCopySemantics()`
- `BlackholeDeviceResourceCanonicalization`
  仍直接读
  `blackhole.copy_semantics`
- `SplitBlackholeKernel`
  仍直接读
  `blackhole.copy_semantics`
- `lower_blackhole_ops.cc`
  里的 `PlanTTKernelABI`
  仍直接读
  `blackhole.copy_semantics`
- 多个 copy / gemm 测试
  仍直接调用
  `AnnotateBlackholeCopySemantics()`
  或直接校验 annotation schema

但同样必须写清楚：

- target / codegen / build / runtime
  现在并不是
  `blackhole.copy_semantics`
  的直接 consumer
- 它们当前主要站在
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  projection 边界上

所以当前不能写成：

> task3 只剩两个 consumer 还没切掉 annotation

也不能写成：

> task3 的统一方案是把所有 consumer
> 都切到 `SpatialPlan.DataflowEdge`

更不能写成：

> target/runtime reader 还在直接读 copy annotation

## 4. 当前代码现实

### 4.1 producer 写的是 `For.annotations`，不是 `AttrStmt`

当前
[annotate_blackhole_copy_semantics.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/annotate_blackhole_copy_semantics.cc#L281)
实际把
`blackhole.copy_semantics`
写进
`ForNode::annotations`，
不是额外包一层 `AttrStmt`。

这和现有部分 doc/comment
里还写着 “AttrStmt / wrap”
并不一致。

因此 task3 文档不能继续沿用
“删掉某个 `AttrStmt` wrapper”
这种过时表述；
真实 deletion target
就是 loop annotation carrier 本身。

### 4.2 `BlackholeDeviceResourceCanonicalization` 是最早的 active consumer

当前
[phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py#L243)
里，
`BlackholeDeviceResourceCanonicalization`
运行在：

- `AnnotateBlackholeCopySemantics()`
  之后
- `AnnotateDeviceRegions`
  之前
- 更早于
  `AnalyzeSpatialStructureFacts /
   BuildSpatialPlanCompanion /
   ValidateSpatialPlan /
   SplitBlackholeKernel`

而
[blackhole_device_resource_canonicalization.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc#L176)
仍直接从 annotation 里取：

- `dst_buffer`
  来标记 `cb_input`
- `src_buffer`
  来标记 `cb_output`
- `mid_buffer`
  来标记 `intermed`

这里最关键的一点是：
**它根本还看不到 `SpatialPlan`。**

因此 task3 不能把
“统一从 `SpatialPlan.DataflowEdge` 恢复”
写成这个 consumer 的方案。
它只能依赖当前阶段已有的 TIR 结构。

这不是 task3 的局部偏好，
而是 repo 内成熟 lowering
的一般模式：
像
[lower_ptx_async_copy.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_ptx_async_copy.cc#L252)
这种成熟 pass，
也是直接从当前
`BufferLoad / BufferStore`
结构恢复 copy/data-movement 含义，
然后在本地完成 rewrite，
而不是先发布一个跨 pass
copy 语义 carrier。

### 4.3 `SplitBlackholeKernel` 仍是 active consumer，但已经有部分 direct fallback

当前
[split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc#L218)
仍直接读
`blackhole.copy_semantics`
来做 reader / writer 分类。

但 repo HEAD 也已经不是
“完全没有 direct recovery”：

- post-compute writer
  已经有
  `FindWriterOutputBuffer(...)`
  这类直接恢复路径
- pure-copy function
  仍直接保持原样，
  `SplitBlackholeKernel`
  只在有 compute 时介入

所以 task3 的要求不是
“从零开始发明 direct recovery”，
而是：

> 把当前已经存在的 direct structural recovery
> 收成 owner truth，
> 删掉剩余 annotation-based segment classification。

另外，
mainline phase ordering
里它运行时
validated `SpatialPlan`
已经存在，
但 pass 本体目前并不读取它。

这代表两件事：

- task3 可以在需要时使用当前阶段已存在的显式表示层
- 但文档不能把
  “一定要统一切到 `SpatialPlan`”
  写成唯一正确实现

### 4.4 `PlanTTKernelABI` 才是晚期 lowering 侧的真实 consumer

task3 原文档漏掉了第三个 active implementation consumer。

当前
[lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L7699)
里，
`PlanTTKernelABI::VisitStmt_(const ForNode*)`
仍会读
`blackhole.copy_semantics`
来设置：

- `copy_input_buffer_`
- `copy_output_buffer_`
- `copy_input_shape_`
- `copy_output_shape_`
- `copy_intermediate_shape_`
- `needs_copy_runtime_args_`
- `saw_copy_op_`

但这里也要写准：

- `PlanTTKernelABI`
  本体已经有大量 direct lowering logic
  处理 pure copy / staged copy
- `SelectBlackholeTTMetalBuiltins`
  走的是
  `select_compute_builtins_only_`
  路径，
  不是当前 copy annotation
  的主 blocker

也就是说，
task3 在晚期 lowering 侧的真实问题
不是“compute builtin selection 还靠 copy annotation”，
而是：

> copy buffer identity / shape /
> runtime-arg shaping
> 这些 bookkeeping
> 仍然由 annotation 充当 owner truth。

### 4.5 当前测试面仍在固定旧 annotation 协议

当前测试里仍有一整批直接依赖 annotation 的内容：

- [test_blackhole_copy_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py#L1141)
  直接检查 annotation schema
- [test_blackhole_copy_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py#L1191)
  检查 annotation 在 flatten/vectorize 后仍可存活
- [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py#L337)
  等多处显式调用
  `AnnotateBlackholeCopySemantics()`
- [test_blackhole_flash_attention_analysis.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py#L35)
  还把
  `SplitBlackholeKernel()`
  当成可 standalone 调用的分析入口

这说明 task3 的测试修订
不能只盯
`test_blackhole_copy_pipeline.py`。

还要明确：

- gemm 相关测试
  也必须一起切
- 如果 `SplitBlackholeKernel`
  最终需要依赖 mainline ordering
  已经建立的前置表示，
  那么 standalone analysis test
  也必须同步到真实 precondition
- 但如果某些测试
  是明确在验证
  `TTProgram`
  /
  `tl.blackhole_executable`
  这类 leaf projection boundary，
  那么手工重建
  `tl.tt_program`
  或
  `tl.blackhole_executable`
  本身不是 task3 blocker；
  真正必须删除的是
  手工插入
  `AnnotateBlackholeCopySemantics()`
  或直接断言 annotation schema
  的 staging/协议依赖

### 4.6 target-side reader boundary 已经不在 `copy_semantics`

runtime / codegen / build
当前主要读取的是：

- `TTProgram`
- `tl.blackhole_executable`
- `ExecutableSpec`

而不是
`blackhole.copy_semantics`。

这意味着 task3 文档必须同时写清楚两件事：

1. compiler-side annotation carrier
   仍然是 wrong-now boundary，
   必须删除
2. target-side 仍然存在的
   payload / projection /
   segment slicing residue
   不能反过来把 copy annotation
   合法化成“还需要保留的中间层”

其中尤其要分清：

- `buffer_tile_bridge_specs`
  是 projection / payload residue
- `compute_contract`
  /
  `multi_compute_contracts`
  /
  `gemm_contract`
  这组
  ExecutableSpec-side contract family，
  当前在
  [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L857)
  和
  [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc#L229)
  里仍有
  `compute_contract <- gemm_contract`
  fallback /
  compatibility gate；
  这是 task3 的 leaf/runtime compatibility debt，
  不是 copy annotation 的 target-side reader
- `blackhole.segment_kind`
  是 kernel body slicing residue

它们都不是
`blackhole.copy_semantics`
继续活着的理由。

## 5. 基于审计结果修正后的任务内容

### 5.1 consumer 必须按所在阶段各自恢复 copy 含义

task3 的核心不是“统一恢复算法”，
而是“去掉 legacy annotation carrier”。

因此正确要求是：

- 对 `BlackholeDeviceResourceCanonicalization`：
  只能依赖它当前阶段可见的 TIR 结构
- 对 `SplitBlackholeKernel`：
  可以依赖当前 IR；
  如果确实有帮助，
  也可以依赖当前阶段已经存在的显式表示层
  （例如 validated `SpatialPlan`）
- 对 `PlanTTKernelABI`：
  只能依赖当前阶段已有的
  IR / validated `SpatialPlan` /
  pass-local lowering analysis

不能把某个阶段看不到的表示层
硬写成统一依赖。

### 5.2 先切完全部 compiler-side consumer，再删 prepass

正确顺序应该是：

1. 切掉
   `BlackholeDeviceResourceCanonicalization`
2. 切掉
   `SplitBlackholeKernel`
3. 切掉
   `PlanTTKernelABI`
   的 annotation bookkeeping
4. 再删除
   `AnnotateBlackholeCopySemantics`
   public transform / pass /
   源文件 / 主链调用点

否则 annotation 仍然会作为真实语义前置条件残留。

### 5.3 不允许用新的共享 copy contract 替代旧 annotation

task3 完成时，
不能把 `blackhole.copy_semantics`
替换成：

- 新的 copy-direction attr
- 新的 shared `MatchedCopyTransfer`
  exported object
- 新的 `Map<String, Any>` copy bag
- 新的 public copy matcher API

如果多个 consumer 需要近似的结构遍历，
可以各自在文件内写 local helper；
必要时也只能抽非常薄的 traversal utility，
不能再抽成新的语义 carrier。

repo 内成熟先例
也支持这个边界：
`lower_ptx_async_copy`
这类 pass
直接在当前 subtree
做 structural recovery，
并不会把 copy 含义
提升成 exported matcher /
shared bag。

### 5.4 task3 要完成的是 owner-truth cutover，不是复制旧 schema

repo HEAD 里已经存在一部分
direct recovery /
direct lowering logic。

因此 task3 的实现目标不是：

- 复制 annotation schema
- 再做一个新的共享结构体
- 把旧字段平移到另一个 helper

而是：

> 让当前 consumer
> 真正以当前阶段可验证的结构事实
> 作为 owner truth，
> 并删掉 annotation schema
> 对 buffer identity / shape /
> runtime-arg bookkeeping
> 的主导地位。

### 5.5 过时测试要跟着真实 pipeline contract 一起改

task3 完成后，
下面这类测试就不应该继续存在：

- annotation schema 测试
- annotation survives flatten/vectorize 测试
- 手工先跑
  `AnnotateBlackholeCopySemantics()`
  再验证下游行为的测试
- 把 `SplitBlackholeKernel()`
  当成独立于 mainline phase ordering
  的稳定 public analysis surface 的测试

它们要么删除，
要么改写成新的 canonical-path 验证：

- 没有 annotation
  仍能得到正确的 split / resource role /
  copy lowering / runtime arg 结果
- 测试 staging
  与真实 phase precondition 一致
- 明确的 leaf-boundary 测试
  仍可直接检查
  `TTProgram`
  /
  `tl.blackhole_executable`
  /
  `ExecutableSpec`
  合同，
  但不能再通过手工插入
  `AnnotateBlackholeCopySemantics()`
  来得到这些合同

### 5.6 target/codegen/build/runtime 继续保持 projection boundary

task3 删除 copy annotation 时，
不应该顺手把 target-side boundary
重新拉回到 TIR annotation / helper bag。

必须继续保持：

- target/codegen/build/runtime
  读取
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  projection
- 不新增 direct `copy_semantics`
  fallback
- 不为了删 annotation
  重新发明 target-visible copy semantic carrier

这里还要额外写清楚：

- TT-Metal
  的稳定 target truth
  是显式
  kernel / circular-buffer /
  semaphore / runtime-arg /
  launch
  对象与 API；
  copy/data movement
  是 kernel 行为，
  不是 target-side semantic tag
- 因此没有任何 target-model 理由
  要求保留
  compiler-side
  `blackhole.copy_semantics`
- repo HEAD 当前
  `rt_mod_blackhole.cc`
  /
  `blackhole_module.cc`
  上还存在的
  `compute_contract <- gemm_contract`
  fallback，
  只能写成
  task3 的 leaf compatibility debt；
  required end-state
  是 leaf/runtime
  直接消费显式
  `compute_contract`
  /
  `multi_compute_contracts`
  或其它明确 projection，
  而不是继续在 leaf 侧做 legacy contract recovery

## 6. 执行切片

1. 先梳理全部 active compiler-side consumer：
   `BlackholeDeviceResourceCanonicalization`
   `SplitBlackholeKernel`
   `PlanTTKernelABI`
2. 在每个 consumer 内，
   用当前阶段已有
   IR / 显式表示层 / pass-local analysis
   做 direct recovery，
   不再读 `blackhole.copy_semantics`
3. 更新相关测试：
   copy / gemm /
   analysis staging
   不再依赖 annotation schema
4. 更新 `phase.py`
   和 `tilelang/transform/__init__.py`，
   把 prepass 从主链和 public surface 上拿掉
5. 删除
   `annotate_blackhole_copy_semantics.cc`
   及其注册入口
6. 清理
   `SplitBlackholeKernel`
   / `BlackholeDeviceResourceCanonicalization`
   / Python wrapper
   里对 annotation precondition
   的过时注释
7. 验证 target/codegen/build/runtime
   仍然站在 executable projection boundary，
   没有被迫拉回新的 copy side channel

## 7. 相关文件

- `tilelang_repo/src/transform/annotate_blackhole_copy_semantics.cc`
- `tilelang_repo/tilelang/transform/__init__.py`
- `tilelang_repo/tilelang/engine/phase.py`
- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

## 8. 验证要求

task3 的验证要证明：
`blackhole.copy_semantics`
已经退出 active chain，
而不是只证明少数 consumer 不再直接读它。

至少要覆盖：

1. `phase.py`
   不再运行
   `AnnotateBlackholeCopySemantics()`
2. `tilelang.transform`
   不再导出
   `AnnotateBlackholeCopySemantics`
3. `BlackholeDeviceResourceCanonicalization`
   不再读取
   `blackhole.copy_semantics`
4. `SplitBlackholeKernel`
   不再读取
   `blackhole.copy_semantics`
5. `PlanTTKernelABI`
   不再读取
   `blackhole.copy_semantics`
6. `SelectBlackholeTTMetalBuiltins`
   没有因为 task3
   回退成新的 copy-annotation 依赖入口
7. target / codegen / build / runtime
   仍然不直接读取
   `blackhole.copy_semantics`
   或新的等价 copy carrier
8. ExecutableSpec-side
   `compute_contract`
   /
   `multi_compute_contracts`
   /
   `gemm_contract`
   这类 leaf compatibility 路径
   不能反向把
   `blackhole.copy_semantics`
   重新合法化成
   target-side input
9. copy / gemm canonical path
   仍能得到正确的 split、
   resource role、
   copy lowering 和 runtime arg 结果
10. annotation schema /
   survives-flatten /
   手工 prepass staging
   这类旧测试
   已删除或迁移

## 9. 完成判据

只有下面这些同时成立，
task3 才算完成：

- `blackhole.copy_semantics`
  已不再被任何 active implementation consumer 读取
- `AnnotateBlackholeCopySemantics`
  已从主链和 public surface 删除
- 各 consumer 已改成 direct current-stage recovery
- target/codegen/build/runtime
  仍保持
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  reader boundary
- 现存 leaf/runtime compatibility debt
  不再被文档或实现
  反向用来合法化
  `blackhole.copy_semantics`
- 没有新的共享 copy semantic carrier 被引入
- 测试面已经从
  “校验 annotation 协议”
  切到
  “校验 canonical path 结果”
- task3 文档和代码口径一致，
  不再把某个单一 consumer 的切换
  误写成整个 task 完成
