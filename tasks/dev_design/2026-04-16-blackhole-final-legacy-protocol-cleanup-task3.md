# Task 3: Replace `blackhole.copy_semantics` With Direct Current-Stage Recovery

## 1. 任务目标

这个 cleanup task 只负责一件事：
把 `blackhole.copy_semantics`
从 active chain 里删掉，
让各个 consumer
直接根据自己当前阶段能看到的 IR / 显式表示层恢复 copy 含义。

完成后不再允许：

- `AnnotateBlackholeCopySemantics`
  作为前置语义 prepass
- `blackhole.copy_semantics`
  作为跨 pass 语义 carrier
- consumer 通过读 loop annotation 里的
  `direction` / `kind` / `src_buffer` / `dst_buffer`
  来决定 planner / canonicalization / lowering 行为

## 2. 范围

这个 task 允许做的事：

- 在每个 consumer 所在阶段，
  用当前阶段已有信息直接恢复 copy 含义
- 删除 `AnnotateBlackholeCopySemantics`
  prepass 和对应 public transform
- 删除围绕 annotation schema 写的过时测试
- 更新 active path 中依赖该 annotation 的 consumer

这个 task 不允许做的事：

- 再引入新的 shared copy-analysis layer
- 再引入新的 copy-direction attr / helper bag / exported matcher
- 把 `blackhole.copy_semantics`
  换个名字继续跨 pass 传

这里要特别注意：

- 不是所有 consumer 都处在同一个表示层
- 不是所有 consumer 都能看到 `SpatialPlan`

所以 task3 的正确合同不是
“统一从 `SpatialPlan.DataflowEdge` 恢复”，
而是：

> 每个 consumer 只能依赖它在当前阶段本来就能合法看到的
> IR / 显式表示层，不能依赖 legacy copy annotation

## 3. 当前状态 (`2026-04-17`)

当前 **不算完成**。

repo HEAD 上，`blackhole.copy_semantics`
仍然在 active chain 里承担实质语义：

- `phase.py`
  在 Blackhole lowering 主链里仍显式运行
  `AnnotateBlackholeCopySemantics()`
- `SplitBlackholeKernel`
  仍直接读
  `blackhole.copy_semantics`
- `BlackholeDeviceResourceCanonicalization`
  仍直接读
  `blackhole.copy_semantics`
- `lower_blackhole_ops.cc / PlanTTKernelABI`
  仍直接读
  `blackhole.copy_semantics`
- 多个 copy / gemm 测试
  仍直接调用
  `AnnotateBlackholeCopySemantics()`
  或直接校验 annotation schema

所以当前不能写成：

> 只剩下两个 consumer 还没切掉 annotation

也不能写成：

> task3 只需要把 `SplitBlackholeKernel`
> 和 resource canonicalization
> 切到 `SpatialPlan.DataflowEdge`

## 4. 当前代码现实

### 4.1 active prepass 还在主链里

当前 [phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py#L246)
明确仍在 Blackhole lowering 主链里运行：

- `AnnotateBlackholeCopySemantics()`
- `BlackholeDeviceResourceCanonicalization()`

而 [tilelang/transform/__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py#L602)
也仍然公开导出
`AnnotateBlackholeCopySemantics()`。

这说明 task3 不是在清理“已经下线的历史 prepass”，
而是在清理当前仍在使用的 active prepass。

### 4.2 `SplitBlackholeKernel` 仍直接消费 annotation

当前 [split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc#L35)
的分类逻辑仍然是：

- `copy_semantics.direction == "dram_to_cb"` -> `reader`
- `copy_semantics.kind == "fused_staged_copy"` -> `reader`
- `copy_semantics.direction == "cb_to_dram"` -> `writer`

这说明它现在并没有从当前 IR / 当前表示层直接恢复 copy 含义，
而是在消费旧 annotation 协议。

### 4.3 `BlackholeDeviceResourceCanonicalization` 也仍直接消费 annotation

当前 [blackhole_device_resource_canonicalization.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc#L40)
明确写着
“Must run after AnnotateBlackholeCopySemantics”，
并且 [同文件](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc#L172)
直接从 annotation 里抽：

- `dram_to_cb` -> `cb_input_names_`
- `cb_to_dram` -> `cb_output_names_`
- `fused_staged_copy` -> `mid_buffer`

这里最关键的一点是：
这个 pass 运行在 [phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py#L246)
里，时序早于 `AnalyzeSpatialStructureFacts / BuildSpatialPlanCompanion / ValidateSpatialPlan / SplitBlackholeKernel`。

也就是说：
**它根本还看不到 `SpatialPlan`。**

因此原文档里把
“从 `SpatialPlan.DataflowEdge` 恢复”
写成统一方案，
在这个 consumer 上就是错的。

### 4.4 `lower_blackhole_ops.cc` 也是 active consumer

task3 原文档漏掉了第三个 active consumer。

当前 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L7703)
里，
`PlanTTKernelABI::VisitStmt_(const ForNode*)`
仍会读
`blackhole.copy_semantics`
来设置：

- `copy_input_buffer_`
- `copy_output_buffer_`
- 各类 copy shape
- `needs_copy_runtime_args_`
- `saw_copy_op_`

这说明 task3 不只是一个“前段 split/canonicalization 清理任务”，
它还涉及晚期 lowering consumer。

### 4.5 当前测试面也仍围绕 annotation 协议写

当前测试里仍有一整批直接依赖 annotation 的内容：

- [test_blackhole_copy_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py#L1141)
  直接检查 annotation schema
- [test_blackhole_copy_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py#L1191)
  检查 annotation 在 flatten/vectorize 后仍可存活
- [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py#L337)
  等多处显式调用
  `AnnotateBlackholeCopySemantics()`

这说明 task3 原文档只列
`test_blackhole_copy_pipeline.py`
是不够的；
gemm 相关测试也必须一起改。

## 5. 基于审计结果修正后的任务内容

### 5.1 consumer 必须按所在阶段各自恢复 copy 含义

task3 的核心不是“统一恢复算法”，
而是“去掉 legacy annotation carrier”。

因此正确要求是：

- 对 `BlackholeDeviceResourceCanonicalization`：
  只能依赖它当前阶段能看到的 IR 结构
- 对 `SplitBlackholeKernel`：
  可以依赖当前 IR，
  如果确实有帮助，也可以依赖当前阶段已经存在的显式表示层
  （例如 validated `SpatialPlan`）
- 对 `lower_blackhole_ops.cc`：
  只能依赖它当前阶段已有的 IR / 显式表示层 / pass-local state

不能把某个阶段看不到的表示层
硬写成统一依赖。

### 5.2 先切完全部 consumer，再删 prepass

原文档里把 task3 基本写成：

- 改 `SplitBlackholeKernel`
- 改 resource canonicalization
- 删 prepass

这个范围不够。

正确顺序应该是：

1. 切掉 `BlackholeDeviceResourceCanonicalization`
2. 切掉 `SplitBlackholeKernel`
3. 切掉 `lower_blackhole_ops.cc`
4. 再删除 `AnnotateBlackholeCopySemantics`
   public transform / pass / 源文件 / 主链调用点

否则 annotation 仍然会作为真实语义前置条件残留。

### 5.3 不允许用新的共享 copy contract 替代旧 annotation

task3 完成时，
不能把 `blackhole.copy_semantics`
替换成：

- 新的 copy-direction attr
- 新的 shared `MatchedCopyTransfer` exported object
- 新的 `Map<String, Any>` copy bag
- 新的 public copy matcher API

如果多个 consumer 需要近似的结构遍历，
可以各自在文件内写 local helper；
必要时也只能抽非常薄的 traversal utility，
不能再抽成新的语义 carrier。

### 5.4 过时测试要跟着任务目标一起改

task3 完成后，
下面这类测试就不应该继续存在：

- annotation schema 测试
- annotation survives flatten/vectorize 测试
- 手工先跑 `AnnotateBlackholeCopySemantics()`
  再验证下游行为的测试

它们要么删除，
要么改写成新的 canonical-path 验证：

- 没有 annotation 仍能得到正确 split / resource role / copy lowering / runtime arg 结果

## 6. 执行切片

1. 先梳理全部 active consumer：
   `BlackholeDeviceResourceCanonicalization`
   `SplitBlackholeKernel`
   `lower_blackhole_ops.cc`
2. 在每个 consumer 内，
   用当前阶段已有 IR / 显式表示层做 direct recovery，
   不再读 `blackhole.copy_semantics`
3. 更新 `phase.py`
   和 `tilelang/transform/__init__.py`，
   把 prepass 从主链和 public surface 上拿掉
4. 删除
   `annotate_blackhole_copy_semantics.cc`
   及其注册入口
5. 更新 copy / gemm 测试，
   删除针对 annotation schema 的旧测试，
   改成 canonical-path 验证
6. 清理 `SplitBlackholeKernel`
   / `BlackholeDeviceResourceCanonicalization`
   注释里对 annotation precondition 的表述

## 7. 相关文件

- `tilelang_repo/src/transform/annotate_blackhole_copy_semantics.cc`
- `tilelang_repo/tilelang/transform/__init__.py`
- `tilelang_repo/tilelang/engine/phase.py`
- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

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
3. `SplitBlackholeKernel`
   不再读取
   `blackhole.copy_semantics`
4. `BlackholeDeviceResourceCanonicalization`
   不再读取
   `blackhole.copy_semantics`
5. `lower_blackhole_ops.cc`
   不再读取
   `blackhole.copy_semantics`
6. copy / gemm canonical path
   仍能得到正确的 split、resource role、
   copy lowering 和 runtime arg 结果
7. annotation schema / survives-flatten 这类旧测试
   已删除或迁移

## 9. 完成判据

只有下面这些同时成立，
task3 才算完成：

- `blackhole.copy_semantics`
  已不再被任何 active consumer 读取
- `AnnotateBlackholeCopySemantics`
  已从主链和 public surface 删除
- 各 consumer 已改成 direct current-stage recovery
- 没有新的共享 copy semantic carrier 被引入
- 测试面已经从“校验 annotation 协议”
  切到“校验 canonical path 结果”
- task3 文档和代码口径一致，
  不再把某个单一 consumer 的切换
  误写成整个 task 完成
