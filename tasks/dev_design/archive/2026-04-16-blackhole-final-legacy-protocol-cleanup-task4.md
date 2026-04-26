# Task 4: Replace `blackhole.segment_kind` With Direct Kernel-Kind Construction

## 0. `2026-04-26` 收口更新

- **状态**: `completed`
- repo HEAD 当前口径：
  - `TTKernelPlan.kind`
    与 executable
    `segment_plan.kind`
    是跨阶段 kernel-kind truth
  - runtime /
    codegen /
    leaf reader
    不允许读取
    `blackhole.segment_kind`
  - `blackhole.segment_kind`
    只允许作为
    `lower_blackhole_ops.cc`
    内部 pass-local mechanics，
    final IR /
    `ExecutableSpec`
    /
    leaf reader
    前必须剥离
- 下文的“当前状态 / 当前代码现实”
  保留为任务执行期快照；
  不再代表 repo HEAD 当前状态。

## 1. 任务目标

这个 cleanup task 的正确目标不是
“先把 attr 全删掉再说”，
而是：

先把
`reader / compute / writer / fused_dataflow`
的跨阶段 owner truth
固定到
`TTProgram / ExecutableSpec`
的显式 kernel records 上，
再把剩余的
leaf-local body slicing residue
单独删掉。

也就是说，
task4 必须分清两件事：

1. **跨阶段 kernel-kind truth**
2. **leaf 端按 kernel kind 切 raw body 的 mechanics**

完成后不再允许：

- planner / ABI /
  `TTProgram` projection /
  runtime schema /
  codegen metadata
  通过扫描
  `AttrStmt("blackhole.segment_kind")`
  决定 kernel kind
- 把 `BuildTTProgram`
  当成 kernel-kind classifier
- 把
  `MaterializeBlackholeExecutable`
  或
  `codegen_blackhole.cc`
  写成 attr consumer
- 把 leaf-local slice marker
  再包装回 planning protocol

task4 的最终 end-state
仍然是：

> `blackhole.segment_kind`
> 完全退出 active chain。

但实现顺序必须先切 owner truth，
再删 body slicing residue，
不能把这两件事混成一个模糊的“全删 attr”口号。

## 2. 范围

### 2.1 允许做的事

- 在
  `PlanTTCompute / PlanTTKernelABI`
  内直接构造
  kernel kind / ABI / segment records
- 让
  `TTKernel.kind`、
  `TTKernelPlan.kind`、
  projected executable
  `segment_plan[*].kind`
  成为唯一跨阶段 kernel-kind truth
- 补强 validator，
  让
  `kind / core_type / ABI / compute payload`
  的一致性在
  `ValidateTTProgram`
  处 fail closed
- 为剩余 leaf-local body slicing residue
  写清楚边界、退出条件和验证要求
- 如果在同一任务里补上
  explicit kernel-body truth
  或 leaf-local direct slicer，
  一并删除
  `SegmentBodyExtractor`
  和 upstream marker emission

### 2.2 不允许做的事

- 新增 shared segment-slice analysis
- 新增 replacement attr / helper bag /
  executable side helper layer
- 让 runtime/codegen
  再从名字、pass 顺序或 late matcher
  恢复 kernel kind
- 把 leaf-local structural marker
  重新包装成 planning protocol
- 把 `BuildTTProgram`
  改成新的 segment classifier
- 用 task3 尚未删除的
  `blackhole.copy_semantics`
  依赖，
  合法化 task4 的终态

## 3. 当前状态 (`2026-04-22`)

当前 **已完成**。

repo HEAD 上，
`blackhole.segment_kind`
已经退出 cross-pass active chain：

- `SplitBlackholeKernel`
  不再发出
  source-level
  `blackhole.segment_kind`
  marker
- `TTKernel.kind`
  / projected executable
  `segment_plan[*].kind`
  成为唯一跨阶段
  kernel-kind truth
- `PlanTTKernelABI`
  对 segment-sensitive
  CB/accessor bookkeeping
  只保留 pass-local marker mechanics，
  并在最终 body
  strip 掉
  marker residue
- `SegmentBodyExtractor`
  已改成
  基于
  `segment_plan.kind`
  +
  lowered builtin family
  的 structural slicer，
  不再读取 marker
- copy / gemm 回归
  已显式断言
  最终 lowered body
  不再残留
  `blackhole.segment_kind`

因此 task4 的完成口径现在是：

1. planner / `TTProgram` / executable
   的 owner truth
   已全部站在显式
   `kind`
   字段链上
2. leaf-local body slicing
   也不再依赖
   source marker

## 4. 当前代码现实

### 4.1 显式 kernel-kind truth 已经成为唯一跨阶段边界

repo HEAD 当前稳定链路是：

```text
PlanTTKernelABI staged segment planning
  -> TTKernel.kind
  -> executable segment_plan[*].kind
  -> ExecutableSpec / KernelSpec
```

这里的
`kind`
已经是
planner / projection / leaf reader
共同消费的唯一显式 truth，
不再需要
source-level marker
充当中间 owner。

### 4.2 planner 侧 marker 只允许停在 pass-local mechanics

`PlanTTKernelABI`
内部仍会为了
segment-sensitive
CB depth /
accessor slot /
copy lowering
使用局部 marker mechanics，
但这些 marker：

- 只存在于当前 pass
  内部重写阶段
- 不再作为 cross-pass contract
  被后续 consumer 读取
- 在最终写回
  `PrimFunc.body`
  前被 strip

也就是说，
task4 当前允许存在的
唯一 residue
是 pass-local implementation mechanics，
不是新的长期协议面。

### 4.3 leaf 侧 body slicing 改成 structural extraction

`SegmentBodyExtractor`
现在按：

- `segment_plan.kind`
- lowered builtin family
- reader / compute / writer
  anchor builtin

直接切 per-kernel raw body。

这条路径不再要求：

- `SplitBlackholeKernel`
  先发 source marker
- runtime / build
  再回读 marker
- 新的 helper layer
  在 leaf 侧补 segment semantics

因此 task4 的 end-state
是：

> `segment_kind`
> 只作为显式 executable segment kind
> 保留在投影对象里，
> 不再作为 source-level marker
> 存活在 active path。

## 5. 基于审计结果修正后的任务内容

### 5.1 唯一合法的跨阶段 kind truth

task4 的架构合同固定为：

- `TTKernel.kind`
- `TTKernelPlan.kind`
- projected executable
  `segment_plan[*].kind`

它们是唯一合法的跨阶段 kernel-kind truth。

runtime/build 继续站在这条显式链上时，
`ExecutableSpec / KernelSpec`
只是这条 truth
在 leaf execution boundary
上的投影结果，
不是新的独立语义来源。

### 5.2 planner-side 必须把 `segment_plan_` 改成 direct construction

`PlanTTKernelABI`
的 required end-state
不是“继续读 attr，
然后把结果写进 `TTProgram`”，
而是：

> 用当前 lowering 过程中
> 已经看到的 local segment records
> 直接构造 `segment_plan_`，
> 再由
> `BuildTTKernelAndABISeeds`
> 生成
> `TTKernel / TTABIPlan`。

这里允许使用 pass-local
matcher / visitor / helper，
但这些 helpers
只能留在
`lower_blackhole_ops.cc`
内部，
不能再升级成新的 shared analysis layer。

对应的 repo-local
成熟模式
也只能写成：

- 像
  `SplitHostDevice`
  /
  `KernelInfo`
  /
  `runtime::FunctionInfo`
  /
  `kernel_metadata`
  这样，
  在当前 pass
  收成本地显式 records
- 如果某个 attr
  只是短命 lowering mechanics，
  它必须尽快被冻结到
  显式对象
  或直接删除

不能把
`blackhole.segment_kind`
重写成另一个
shared marker /
shared matcher /
helper bag。

### 5.3 `BuildTTProgram` / projection / codegen 继续保持当前职责

- `BuildTTProgram`
  只聚合 staged TTProgram slices，
  不是 kernel-kind classifier
- `MaterializeBlackholeExecutable`
  只从 `TTProgram`
  投影 executable records，
  不负责 semantic recovery
- `codegen_blackhole.cc`
  只读 executable projection，
  不负责 attr scan
  或 segment re-classification

task4 不应该把这些已经收口的 reader
重新拉回 attr boundary。

### 5.4 leaf-local body slicing residue 必须单独处理

`SegmentBodyExtractor`
当前仍直接读
`blackhole.segment_kind`。

它的 required end-state
只有两种：

1. `TTProgram / ExecutableSpec`
   获得显式 kernel-body truth，
   leaf 端直接消费它
2. leaf 端改成
   direct current-IR local slicer，
   不再依赖跨阶段 marker attr

在真正做到这两者之一以前，
文档必须把
`SegmentBodyExtractor`
明确写成：

- architecturally wrong 的 leaf-local residue
- 当前唯一允许的
  `blackhole.segment_kind`
  reader
- 不得上升回 planning /
  projection / runtime-schema
  的合法边界

这里还必须附带一个
TT-Metal-facing
硬约束：
TT-Metal
只要求最终 kernel source /
kernel class /
runtime args /
program launch
对象闭合。

它不要求
compiler 保留
source-level role marker。

所以这条 residue
没有任何
target-model
合法性加成。

### 5.5 task4 的实现顺序必须分两段

task4 的正确顺序是：

1. **先切 cross-stage owner truth**
   - `PlanTTKernelABI`
     不再用 attr scan /
     marker-threaded
     bookkeeping
     构造 `segment_plan_`
     或驱动
     segment-sensitive
     CB/accessor mechanics
   - `TTKernel.kind`
     / `TTKernelPlan.kind`
     / executable `segment_plan.kind`
     成为唯一合法 truth
2. **再删 leaf-local body slicing residue**
   - 删除
     `SplitBlackholeKernel`
     的 marker emission
   - 删除
     `SegmentBodyExtractor`
   - 或以显式 kernel-body truth /
     leaf-local direct slicer
     取代它们

不能在第 1 段
刚做完时
就把 task4 写成“已经完成”。

### 5.6 task3 前置依赖必须显式保留

如果 task3
还没先把
`SplitBlackholeKernel`
从
`blackhole.copy_semantics`
上切下来，
那么 task4
即使完成了 kind owner-truth cutover，
也不能被表述成“splitter 已完全 IR-first”。

正确写法只能是：

- task4 负责删
  `blackhole.segment_kind`
  的跨阶段 owner truth
- task3 负责删
  `blackhole.copy_semantics`
  对 splitter 的分类依赖

两者都必须完成，
最终设计才成立。

## 6. 执行切片

1. 先补回归，
   锁定
   `TTKernel.kind`
   / `TTKernelPlan.kind`
   / executable `segment_plan.kind`
   是唯一跨阶段 truth
2. 在
   `PlanTTKernelABI`
   内把
   `segment_plan_`
   改成 direct construction，
   不再依赖
   `CollectSegmentKindsFromBody(...)`
   /
   `AnalyzeCBDepthEffect(...)`
   /
   attr-driven
   `current_segment_kind_`
   / accessor /
   copy-lowering
   bookkeeping
3. 保持
   `BuildTTProgram`
   / projection /
   codegen /
   runtime metadata
   继续只读显式 records，
   不回退成 attr consumer
4. 单独隔离
   `rt_mod_blackhole.cc`
   的 body slicing residue，
   把它限制成唯一剩余 reader
   并写明退出条件
5. 当 explicit kernel-body truth
   或 leaf-local direct slicer
   准备好后，
   再删
   `SplitBlackholeKernel`
   marker emission
   和
   `SegmentBodyExtractor`

## 7. 相关文件

- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/transform/split_blackhole_kernel.h`
- `tilelang_repo/tilelang/transform/__init__.py`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/build_tt_program.cc`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/transform/materialize_blackhole_executable.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/testing/python/target/blackhole/common.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`

## 8. 验证要求

task4 的验证必须分成两组：

### 8.1 owner-truth cutover 验证

至少要覆盖：

1. `PlanTTKernelABI`
   不再通过
   `CollectSegmentKindsFromBody(...)`
   /
   `AnalyzeCBDepthEffect(...)`
   /
   `current_segment_kind_`
   这类 marker-driven
   路径
   把 body attr
   当成 `segment_plan_`
   /
   accessor /
   CB depth
   truth 来源
2. `TTKernel.kind`
   / `TTKernelPlan.kind`
   / executable `segment_plan.kind`
   非空且一致
3. `ValidateTTProgram`
   对
   `kind / core_type / compute payload`
   的一致性继续 fail closed
4. `BuildTTProgram`
   不是 segment classifier
5. `MaterializeBlackholeExecutable`
   和 `codegen_blackhole.cc`
   仍不读取
   `blackhole.segment_kind`
6. target/build/runtime metadata
   仍从 executable `segment_plan`
   读取 kind truth
7. Python 回归测试
   继续以
   `TTProgram`
   /
   executable `segment_plan`
   为主断言面，
   不重新引入
   raw marker schema
   测试

### 8.2 full attr deletion 验证

当进入 task4 的第二段时，
还必须覆盖：

1. `SplitBlackholeKernel`
   不再发出
   `blackhole.segment_kind`
2. `rt_mod_blackhole.cc`
   不再有
   `SegmentBodyExtractor`
   或其他 attr reader
3. `rg -n "blackhole\\.segment_kind" tilelang_repo/src`
   不再命中 active-chain consumer
4. 回归测试仍能证明：
   per-kernel body materialization /
   runtime args / codegen metadata
   都不依赖 legacy marker

## 9. 完成判据

只有下面这些同时成立，
task4 才算完成：

- `TTKernel.kind`
  / `TTKernelPlan.kind`
  / executable `segment_plan.kind`
  是唯一跨阶段 kernel-kind truth
- `PlanTTKernelABI`
  不再把
  `blackhole.segment_kind`
  当成 `segment_plan_`
  /
  accessor /
  CB depth
  truth 来源
- `BuildTTProgram`
  不是 segment classifier
- `MaterializeBlackholeExecutable`
  / `codegen_blackhole.cc`
  / runtime metadata reader
  都继续只读显式 projection
- `SplitBlackholeKernel`
  不再发出
  `blackhole.segment_kind`
- `rt_mod_blackhole.cc`
  不再通过 raw TIR body
  按 attr 切 segment body
- 没有新的 shared segment analysis /
  replacement attr /
  helper bag 被引入
- task4 文档和代码口径一致，
  不再把 planner truth
  和 leaf-local body slicing
  混写成同一个边界
