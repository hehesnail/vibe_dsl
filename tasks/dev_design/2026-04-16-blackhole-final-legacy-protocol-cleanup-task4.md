# Task 4: Replace `blackhole.segment_kind` With Direct Kernel-Kind Construction

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

## 3. 当前状态 (`2026-04-21`)

当前 **不算完成**。

repo HEAD 上，
`blackhole.segment_kind`
已经不再是“全链都在直接读”的旧协议，
但也远没有删干净。

当前真实状态是：

- `SplitBlackholeKernel`
  在 validated `SpatialPlan`
  之后仍会发出
  `AttrStmt("blackhole.segment_kind", ...)`
- `PlanTTKernelABI`
  仍通过 body attr scan、
  `current_segment_kind_`
  和 segment-sensitive
  CB/accessor bookkeeping
  把 marker 当成
  `segment_plan_`
  /
  accessor slot /
  CB depth
  的来源
- `TTKernel.kind`
  / `TTKernelPlan.kind`
  / projected executable
  `segment_plan.kind`
  已经是显式字段，
  并且 downstream reader
  大多已经站在这条显式链上
- `BuildTTProgram`
  不是 kernel-kind classifier；
  它只聚合 staged TTProgram slices
- `MaterializeBlackholeExecutable`
  和 `codegen_blackhole.cc`
  已经不是 attr reader
- `rt_mod_blackhole.cc`
  里仍有
  `SegmentBodyExtractor`
  直接按
  `blackhole.segment_kind`
  切 per-kernel body

因此 task4 当前真正的 wrong-now boundary
不是整个 downstream target path，
而是：

1. planner 侧仍拿 attr
   充当 kind truth
2. leaf 侧仍拿 attr
   充当 raw-body slicing marker

## 4. 当前代码现实

### 4.1 显式 kernel-kind truth 已经存在于 `TTProgram / ExecutableSpec`

当前 repo HEAD
已经有完整的显式字段链：

- `TTKernel.kind`
- `TTKernelPlan.kind`
- executable `segment_plan[*].kind`

具体链路是：

```text
PlanTTKernelABI segment_plan_
  -> TTKernel / TTABIPlan
  -> TTKernelPlan
  -> executable segment_plan
  -> ExecutableSpec / KernelSpec
```

这意味着 task4 不是
“从零开始发明 kernel-kind 表示”。

真正的问题是：
**上游 owner truth 还没有彻底从 attr scan 上切下来。**

### 4.2 `PlanTTKernelABI` 才是 planner-side 的真实 wrong-now boundary

当前
[lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1550)
仍会扫描 body 上的
`blackhole.segment_kind`
来收集 segment kinds，
而
[StoreSegmentPlan](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L2717)
仍把这些 attr-scan 结果
建成 `segment_plan_`。

同时，
[VisitStmt_(AttrStmt)](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L7446)
还用
`current_segment_kind_`
在 lowering 过程中追踪活跃 segment。

但 planner-side
对 marker 的依赖
并不只停在
`segment_plan_`
这一层。

[AnalyzeCBDepthEffect](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L644)
/
[UpdateCBRequirementDepthsFromLoweredBody](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L746)
还会递归读取
`blackhole.segment_kind`
来做 segment-sensitive
CB depth 推导；
[WrapSegmentStmtIfNeeded](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1590)
/
[MaybeWrapComputeSegment](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1616)
继续发 marker，
而
[ResolveAccessorSegmentKind](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L4063)
和 copy lowering
则继续用
`current_segment_kind_`
决定 accessor slot /
runtime-arg / CB class
bookkeeping。

所以 task4 在 compiler-side
真正要切的是：

- `CollectSegmentKindsFromBody(...)`
- attr-driven `segment_plan_`
  construction
- `AnalyzeCBDepthEffect(...)`
  这类基于 marker
  的 segment-sensitive
  CB depth 推导
- attr-driven
  `current_segment_kind_`
  /
  accessor /
  copy-lowering
  bookkeeping

而不是去改
`BuildTTProgram`
或已完成 projection cutover
的 downstream reader。

### 4.3 `BuildTTProgram` 不是 classifier，projection/codegen 也不是主要切口

当前
[build_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/build_tt_program.cc#L468)
里的 `PlanTTCompute`
只是调用
`PlanTTKernelABI`，
再把产出的
`TTKernel / TTABIPlan`
写进 staged `TTProgram`。

它不会自己推导 kernel kind。

而：

- [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h#L100)
  直接把 `TTKernel.kind`
  投影成 executable `segment_plan.kind`
- [materialize_blackhole_executable.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/materialize_blackhole_executable.cc#L17)
  只是写 projection
- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L231)
  只读 executable `segment_plan`

这说明 task4
不能再把
`BuildTTProgram`、
`MaterializeBlackholeExecutable`
或
`codegen_blackhole.cc`
写成主要 attr consumer。

repo 内成熟 backend
先例
也不支持继续保留
这种 marker truth：

- [split_host_device.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_host_device.cc#L64)
  直接把 host/device body
  切成独立 device `PrimFunc`
- [lower_device_kernel_launch.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_device_kernel_launch.cc#L41)
  把 per-kernel launch truth
  收成显式
  `KernelInfo`
- CUDA / HIP / CuTeDSL
  runtime builder /
  wrapper
  继续消费
  `runtime::FunctionInfo`
  /
  `kernel_metadata`
  这类显式 records，
  而不是跨 pass marker attr

这些 repo-local
成熟模式都说明：
attr 可以是短命 lowering
mechanics，
但不能继续充当
中期或长期
cross-stage owner truth。

### 4.4 `rt_mod_blackhole.cc` 的 body slicing 是单独的 leaf-local residue

当前
[rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L1750)
里的
`SegmentBodyExtractor`
仍然直接读
`AttrStmt("blackhole.segment_kind")`，
而
[MakeSegmentPrimFunc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L2663)
会据此把原始 device `PrimFunc`
切成 per-kernel `PrimFunc`。

这里要明确：

- 这不是 planner truth
- 这也不是 projection truth
- 它只是当前 build/runtime path
  还没删掉的
  **leaf-local body slicing residue**

同一个
`rt_mod_blackhole.cc`
文件里，
runtime metadata
其余部分
已经主要站在
projected executable
`segment_plan`
和显式
`kernel.kind`
/
`kernel.core_type`
上；
真正还碰
`blackhole.segment_kind`
的地方，
只剩
`SegmentBodyExtractor`
这条 raw-body slicer。

而 TT-Metal
target model
本身也不要求
这种 source-level marker：
它要求的是
host 端 `Program`
里显式 kernel objects、
`CreateKernel`
时选定的
kernel class/config、
以及最终 source /
runtime args。

因此 per-kernel body slicing
在 task4 里
只能被写成
compiler-local deletion problem，
不是 target/runtime
必须保留的协议。

因此 task4 文档必须同时写两件事：

1. 它是 architecturally wrong，
   不能长期存在
2. 在完全删除前，
   它也不能被重新表述成
   “runtime 仍需要的合法中间层”

### 4.5 task3 仍是 task4 的真实前置依赖

当前
`SplitBlackholeKernel`
仍通过
`blackhole.copy_semantics`
做 reader / writer classification。

因此 task4 文档必须写清楚：

- task4 的当前实现
  可以先切
  `blackhole.segment_kind`
  的 owner truth
- 但这**不等于**
  `SplitBlackholeKernel`
  的分类来源已经合法
- task3 仍然负责删除
  `blackhole.copy_semantics`
  依赖；
  task4 不得把它带进最终设计口径

### 4.6 当前测试主面已经站在显式 kind records 上

当前 Python 回归测试里，
已经没有一条主测试面
继续直接断言
`blackhole.segment_kind`
字面 schema。

主要断言
大多落在：

- `TTProgram.kernels[*].kind`
- executable `segment_plan[*].kind`

而不是直接扫
`AttrStmt("blackhole.segment_kind")`。

这说明 task4 的测试重心
也应该继续放在：

- `TTProgram`
- executable projection
- runtime/build metadata
- 显式 `kind`
  records 的 round-trip
  和 leaf-boundary
  行为校验

而不是去“清扫一批
并不存在的 raw-marker schema test”，
或把 body attr
重新拉回成主回归 surface。

允许继续存在的测试
是：
显式 leaf-boundary
重建
`TTProgram`
/
`segment_plan`
后
验证 codegen/runtime
行为的测试。

不允许重新引入的测试
是：

- 直接把
  `blackhole.segment_kind`
  当 public protocol
  断言
- 用 marker attr
  作为 task4 已完成的
  staging 证明
- 用 task3
  手工
  `AnnotateBlackholeCopySemantics()`
  staging
  反向替 task4
  掩盖 splitter
  前置依赖

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
