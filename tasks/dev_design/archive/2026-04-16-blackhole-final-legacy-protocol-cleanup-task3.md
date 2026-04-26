# Task 3: Replace `blackhole.copy_semantics` With Direct Current-Stage Recovery

## 0. `2026-04-26` 收口更新

- **状态**: `completed`
- repo HEAD 当前已完成：
  - `blackhole.copy_semantics`
    已退出 active chain
  - leaf reader /
    codegen /
    runtime
    站在
    `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
    projection boundary
  - bridge payload /
    contract-family fallback
    已删除，
    不能再作为 task3
    或 runtime compatibility debt
    恢复
- 下文的“当前状态 / 当前代码现实”
  保留为任务执行期快照；
  不再代表 repo HEAD 当前状态。

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

## 3. 当前状态 (`2026-04-22`)

当前 **已完成**。

repo HEAD 上，
`blackhole.copy_semantics`
已经退出 active chain：

- `phase.py`
  不再运行
  `AnnotateBlackholeCopySemantics()`
- `tilelang.transform`
  不再公开导出
  `AnnotateBlackholeCopySemantics()`
- `annotate_blackhole_copy_semantics.cc`
  实现文件
  已删除
- `BlackholeDeviceResourceCanonicalization`
  已改成
  基于当前
  `BufferLoad / BufferStore`
  结构
  直接恢复 copy direction
  和 resource role
- `SplitBlackholeKernel`
  不再消费
  `blackhole.copy_semantics`
- `PlanTTKernelABI`
  已改成
  基于当前 TIR /
  logical-shape metadata /
  transport coverage
  的 direct recovery
- copy / gemm 回归
  已切到 mainline phase ordering，
  并显式断言
  旧 public entry
  与 annotation
  不再存在

同时保持不变的是：

- target / codegen / build / runtime
  仍然不是
  `blackhole.copy_semantics`
  的 reader boundary
- 它们继续只站在
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  projection 边界上

## 4. 当前代码现实

### 4.1 deletion target 已经落到真实 carrier 本身

这次删除的不是某个想象中的
`AttrStmt` wrapper，
而是实际的
loop annotation carrier /
public prepass /
Python wrapper /
实现文件 /
测试 helper
整套旧入口。

### 4.2 最早 consumer 现在直接依赖当前 TIR

`BlackholeDeviceResourceCanonicalization`
仍然处在
很早的 lowering 阶段，
也仍然看不到
`SpatialPlan`
作为它的 owner truth。

repo HEAD 的正确形态是：

- 直接从当前
  `BufferLoad / BufferStore`
  结构
  恢复
  DRAM <-> CB
  copy direction
- 在 pass-local 内部
  完成 `cb_input / cb_output`
  分类
- 不再依赖
  任何跨 pass copy annotation

### 4.3 flatten / vectorize 后的 shared shape 也必须从当前 IR 恢复

删除 `blackhole.copy_semantics`
之后，
flattened staged copy
不能再依赖旧 annotation
里携带的 shared/global shape。

repo HEAD 当前做法是：

- logical buffer shape registry
  对同 data identity
  的 alias
  保留更高优先级 /
  更高维度的 logical shape
- 对已经 flatten 成 1-D 的 shared staging buffer，
  绑定 transport var 的静态 extent，
  直接从当前 global access
  的 row/col coverage
  推出 shared matrix shape

这条恢复链
完全站在当前 TIR /
pass-local analysis 上，
没有重新引入
新的 shared copy carrier。

### 4.4 `SplitBlackholeKernel` 已退回历史 phase hook

repo HEAD 的
`SplitBlackholeKernel`
仍保留 phase name
作为主链历史 hook，
但它不再承担
copy annotation consumer /
segment marker producer
角色。

因此 task3 的实际 end-state
不是“换一个统一新 bag”，
而是：

> 每个 compiler-side consumer
> 只能依赖它在当前阶段本来就能合法看到的
> IR / 显式表示层 /
> pass-local derived facts。

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
