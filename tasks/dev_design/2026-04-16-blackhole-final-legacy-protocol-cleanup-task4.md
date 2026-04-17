# Task 4: Replace `blackhole.segment_kind` With Direct Kernel-Kind Construction

## 1. 任务目标

这个 cleanup task 的正确目标不是
“先把 attr 全删掉再说”，
而是：

把
`reader / compute / writer / fused_dataflow`
的跨阶段 owner truth
从 `blackhole.segment_kind`
切到
`TTProgram / ExecutableSpec`
的显式 kernel records 上，
并把仍然存在的 leaf body slicing residue
单独识别出来。

完成后，以下行为必须消失：

- planner / ABI / executable projection / runtime arg schema /
  codegen metadata
  通过扫描
  `AttrStmt("blackhole.segment_kind")`
  决定 kernel kind
- 把 `BuildTTProgram`
  当成 kernel-kind 推导点
- 把
  `MaterializeBlackholeExecutable`
  或
  `codegen_blackhole.cc`
  写成 attr consumer

但当前 task4
**不能再默认等同于**：

- 立刻删除所有
  `blackhole.segment_kind`
  structural slice marker
- 立刻删除 runtime/build
  里基于原始 TIR body 的
  segment extraction

因为 repo HEAD 里，
leaf build path 仍在用原始 device `PrimFunc`
重新切出 per-kernel `PrimFunc`。

## 2. 范围

### 2.1 允许做的事

- 在 `PlanTTCompute / PlanTTKernelABI`
  内直接构造
  kernel kind / ABI / segment records
- 让
  `TTKernel.kind`、
  `TTKernelPlan.kind`
  和 projected executable
  `segment_plan[*].kind`
  成为唯一跨阶段 kernel-kind truth
- 为剩余 leaf-local slice residue
  写清楚边界和退出条件
- 增加针对
  `TTProgram / executable projection / runtime schema`
  的回归测试
- 补强 validator，
  让
  kind / core_type / ABI / compute payload
  的一致性在
  `ValidateTTProgram`
  处 fail closed

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

## 3. 当前状态 (`2026-04-17`)

当前 **不算完成**。

repo HEAD 上，
和 `segment_kind`
相关的真实 owner truth / consumer 关系是：

- `LowerToBlackholePhaseB`
  仍在 validated `SpatialPlan`
  之后运行
  `SplitBlackholeKernel`
  ([phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py#L368))
- `SplitBlackholeKernel`
  仍在 Phase B
  把原始 TIR body
  包成
  `AttrStmt("blackhole.segment_kind", ...)`
  ([split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc#L20))
- `PlanTTCompute`
  本身不在
  `build_tt_program.cc`
  里推导 kernel kind；
  它只是调用
  `PlanTTKernelABI`，
  然后把 planner 产出的
  `TTKernel / TTABIPlan`
  写进 `TTProgram`
  ([build_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/build_tt_program.cc#L476))
- 真正当前还在构造
  `segment_plan_ -> TTKernel / TTABIPlan`
  的地方是
  `lower_blackhole_ops.cc / PlanTTKernelABI`：
  - 先扫 body 上的
    `blackhole.segment_kind`
    收集 segment kinds
    ([lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1550))
  - lowering 过程中继续补写
    `blackhole.segment_kind`
    ([lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1590))
  - `StoreSegmentPlan`
    仍把显式 segment records
    建在 attr scan 上
    ([lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L2717))
  - `BuildTTKernelAndABISeeds`
    再由 `segment_plan_`
    生成
    `TTKernel / TTABIPlan`
    ([lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L3293))
- `tt_program_projection.h`
  已经直接从
  `TTKernel + TTABIPlan`
  生成 executable `segment_plan`
  ([tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h#L100))
- `MaterializeBlackholeExecutable`
  已经只是写 projection，
  不是 attr reader
  ([materialize_blackhole_executable.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/materialize_blackhole_executable.cc#L17))
- `codegen_blackhole.cc`
  已经从 executable `segment_plan`
  读 runtime args / metadata，
  不是 attr reader
  ([codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc#L231))
- 真正还在 leaf path
  直接读
  `blackhole.segment_kind`
  的关键 consumer
  是
  `rt_mod_blackhole.cc`：
  - `SegmentBodyExtractor`
    按 attr 切 body
    ([rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L1750))
  - `MakeSegmentPrimFunc`
    用它把原始 device `PrimFunc`
    切成 per-kernel `PrimFunc`
    再发给 codegen
    ([rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc#L2663))

## 4. 系统性审查结论

### 4.1 `Severity: Blocker`

**症状**

原文档把
“kernel kind 的跨阶段 owner truth”
和
“leaf 端 per-kernel body slicing truth”
写成了同一个 cutover，
并在 Step 5/6 里要求
同时删除 emission 和所有 reader。

**根因**

文档把
`blackhole.segment_kind`
视为单一协议，
但 repo HEAD 里它实际上承担了两种不同职责：

1. planner 侧的
   kind truth / ABI 分段输入
2. leaf build 侧的
   TIR body slicing marker

前者应该进入
`TTProgram / ExecutableSpec`。
后者如果还存在，
它的退出条件必须是显式 kernel-body truth
或等价的 leaf-local direct slicer。

**影响**

按原文档实现会出现二选一坏结果：

- 要么 runtime/build
  失去 per-kernel body 提取能力，
  直接破坏 kernel source emission
- 要么临时再发明新的 side channel，
  把问题换个名字继续跨阶段传

**建议修改**

把 task4 改成两层合同：

- **当前 task4 必须完成**
  `kernel kind owner truth -> TTKernel / executable segment_plan`
- **当前 task4 不再默认承诺**
  立刻删除所有 structural slice marker；
  如果 leaf build 仍要切原始 TIR body，
  这个 residue 必须单独标注并给出退出条件

**优先级**

第一优先级

### 4.2 `Severity: High`

**症状**

原文档把 kernel-kind construction
写到
`build_tt_program.cc`
和
`BuildTTProgram`
上。

**根因**

文档在用历史 pass 名字
描述架构边界，
没有按当前显式表示对象
和真实 construction point 写。

**影响**

这会把实现者引到错误切口：
去改 TTProgram 聚合 pass，
却绕过了真正的 owner truth 产生点。

**建议修改**

把 task4 的主实施点改写为：

- `PlanTTCompute / PlanTTKernelABI`
- `segment_plan_`
- `BuildTTKernelAndABISeeds`
- `TTKernel.kind / TTABIPlan`

`BuildTTProgram`
只负责聚合与校验 staged slices，
不是 kernel kind classifier。

**优先级**

高

### 4.3 `Severity: High`

**症状**

原文档把
`materialize_blackhole_executable.cc`
和
`codegen_blackhole.cc`
写成当前主要 attr consumer，
却没有把
`lower_blackhole_ops.cc`
和
`rt_mod_blackhole.cc`
的真实 residue 拆清楚。

**根因**

文档没有按 repo HEAD
重新做 implementation audit。

**影响**

会把工作量投到
已经完成 projection cutover 的地方，
同时掩盖真正还在读 attr 的消费者。

**建议修改**

把文件范围改成：

- **必改**
  `lower_blackhole_ops.cc`
  `validate_tt_program.cc`
  `rt_mod_blackhole.cc`
  测试
- **只在 planner interface 变化时改**
  `build_tt_program.cc`
- **只在 full body-slice deletion 时改**
  `split_blackhole_kernel.cc`
  `tt_program_projection.h`
  以及可能的 leaf materialization helpers

`materialize_blackhole_executable.cc`
和
`codegen_blackhole.cc`
在当前 task4 里
不是主要 reader 切口。

**优先级**

高

### 4.4 `Severity: High`

**症状**

原文档把 task4
写得像是可以独立开始，
但当前 Phase B splitter
仍直接依赖
`blackhole.copy_semantics`
来产生
reader/writer classification
([split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc#L34))。

**根因**

文档没有把 task3 和 task4
的真实依赖关系写出来。

**影响**

如果 task3 没先把
`SplitBlackholeKernel`
切到 direct current-stage recovery，
那么 task4
即使把 `segment_kind`
改成 direct kernel records，
上游仍会带着旧
`copy_semantics`
依赖继续工作。

**建议修改**

把 task4 前置条件写清楚：

- task3 已经让
  `SplitBlackholeKernel`
  不再依赖
  `blackhole.copy_semantics`
- 或者 task4
  在同一实现里把这部分
  local classifier 一并切掉，
  但不能把旧 annotation 依赖
  默认带进 task4 终态

**优先级**

高

### 4.5 `Severity: Medium`

**症状**

原文档没有把
validator / failure mode
写成显式完成合同。

当前 `ValidateTTProgram`
只要求
`kind` 非空和 payload 基本完整
([validate_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/validate_tt_program.cc#L134))，
但没有把
“禁止 planner/backend metadata consumer
回退到 attr scan”
写成 task4 的 completion contract。

**根因**

文档偏重文件编辑步骤，
轻了 representation contract。

**影响**

后续很容易出现：
`TTKernel.kind`
已经有了，
但某个 leaf / runtime helper
又悄悄回头扫 attr。

**建议修改**

completion gate 必须明确写：

- `TTKernel.kind`
  / `TTKernelPlan.kind`
  / executable `segment_plan.kind`
  是唯一跨阶段 kind truth
- no planner / ABI / projection /
  runtime-schema / codegen-metadata
  consumer scans `blackhole.segment_kind`
- 如果 `SegmentBodyExtractor`
  还存在，
  它是唯一允许的 leaf-local residue，
  并且文档必须写明退出条件

**优先级**

中

## 5. 修正后的任务合同

### 5.1 跨阶段 owner truth

Task4 完成后，
跨阶段的 kernel-kind truth
固定只有：

- `TTKernel.kind`
- `TTKernelPlan.kind`
- projected executable
  `segment_plan[*].kind`

kernel kind
必须在
`PlanTTCompute / PlanTTKernelABI`
对当前 IR 的 direct construction
过程中确定，
而不是在
`BuildTTProgram`、
`MaterializeBlackholeExecutable`
或 runtime/codegen leaf
里二次恢复。

### 5.2 叶子侧 body slicing residue

如果 repo HEAD
仍通过原始 device `PrimFunc`
重新切出 per-kernel body，
那么任何剩余的 slice mechanism
都只能被视为
**leaf-local residue**，
不是长期协议。

也就是说：

- 它不能再被
  planner / ABI / projection /
  runtime schema / codegen metadata
  读取
- 它不能再成为
  `TTKernel.kind`
  的来源
- 它必须有明确退出条件：
  1. `TTProgram / executable`
     获得显式 kernel-body truth，或
  2. leaf 端改成
     direct current-IR local slicer，
     不再依赖跨阶段 marker attr

### 5.3 `BuildTTProgram` / projection / codegen 的角色

- `BuildTTProgram`
  只聚合 staged TTProgram slices，
  不负责 segment classification
- `MaterializeBlackholeExecutable`
  只从 `TTProgram`
  投影 executable records，
  不负责 semantic recovery
- `codegen_blackhole.cc`
  只读 executable projection，
  不负责 attr scan
  或 segment re-classification

## 6. 修正后的实施顺序

### 6.1 先写回归，锁定真实 contract

最先补的不是
“Phase B 不再有 `segment_kind` attr”
这种当前还不成立的假合同，
而是下面这三类回归：

1. `TTProgram` kernel records
   持有稳定
   `kind / core_type / ABI`
   owner truth
2. executable `segment_plan`
   直接投影自
   `TTKernel + TTABIPlan`，
   而不是 body attr scan
3. runtime / codegen metadata
   只从 executable projection
   读 segment records

如果当前仍保留
leaf-local body slicing，
测试要把这层 residue
单独写出来，
不要把它和 owner-truth cutover
混成一个断言。

### 6.2 把 kernel kind construction 收回 `PlanTTCompute / PlanTTKernelABI`

主切口应放在：

- `StoreSegmentPlan`
- `BuildTTKernelAndABISeeds`
- 各类
  `segment_kind -> accessor / runtime_arg / compile_time_arg`
  编码路径

目标是：

- 不再通过
  `CollectSegmentKindsFromBody`
  把 body attr scan
  当成 kernel truth 来源
- 改成由当前 lowering 过程中
  已经看到的 local segment records
  直接生成 `segment_plan_`

需要的小型 matcher / visitor
可以留在
`lower_blackhole_ops.cc`
内部，
但不能升级成新的
shared analysis layer。

### 6.3 用 `TTKernel` / executable projection 贯通后段

一旦 `segment_plan_`
来自 direct construction，
就继续沿现有路径：

```text
segment_plan_
  -> TTKernel / TTABIPlan
  -> TTProgram
  -> executable segment_plan
```

让 runtime / codegen metadata
继续只读 projection，
不要再回头读 body attr。

### 6.4 单独处理 leaf body slicing

这里必须显式二选一：

1. **最小可行路径**
   暂时保留
   leaf-local slice residue，
   但把它严格限制在
   `rt_mod_blackhole.cc`
   的 per-kernel body materialization 内；
   task4 completion
   不再把
   “全 repo 无 `segment_kind`”
   当成当前硬 gate
2. **更彻底路径**
   同一任务里补上
   explicit kernel-body truth
   或 leaf-local direct slicer，
   然后删掉
   `SegmentBodyExtractor`
   和 upstream marker emission

默认推荐第一条，
因为它不要求在当前 cleanup task
里同时扩展新的 body-carrying representation。

### 6.5 补强 validator 和 completion gate

至少补上下面这些 gate：

- `TTKernel.kind`
  / `TTKernelPlan.kind`
  非空且与
  `core_type / payload`
  对齐
- compute kernels 的
  `compute_config`
  继续由 validator
  fail closed
- no planner / ABI / projection /
  runtime-schema / codegen-metadata
  consumer scans `blackhole.segment_kind`
- 如果
  `SegmentBodyExtractor`
  仍存在，
  它是唯一允许的 attr reader，
  并且只能用于
  leaf body materialization

## 7. 建议文件范围

### 7.1 必改

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

### 7.2 按接口变化决定是否改

- `tilelang_repo/src/transform/build_tt_program.cc`

### 7.3 只有在同一任务里做 full body-slice deletion 时才需要改

- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/target/tt_program_projection.h`
- 其他 leaf materialization helpers

### 7.4 当前不是主要切口

- `tilelang_repo/src/transform/materialize_blackhole_executable.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`

## 8. 验证

建议最少跑：

```bash
pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k "tt_program or segment"
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k "segment or kernel or runtime_arg"
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "segment or reader or writer"
```

如果同时改了
leaf body slicing，
再加上对应
full-suite / codegen / direct-runtime case。

另外增加 grep cross-check：

```bash
rg -n "blackhole\\.segment_kind" tilelang_repo/src
```

完成判定分两种：

- **owner-truth cutover 完成**
  允许只剩
  leaf-local residue
  命中
- **full attr deletion 完成**
  不允许再有任何 active-chain 命中

## 9. Self-Review

### 9.1 Task4 完成最低合同

- `TTKernel.kind`
  / `TTKernelPlan.kind`
  / executable `segment_plan.kind`
  是唯一跨阶段 kind truth
- `BuildTTProgram`
  不是 segment classifier
- `MaterializeBlackholeExecutable`
  不是 attr reader
- `codegen_blackhole.cc`
  只读 projection
- 文档里明确写出
  leaf body slicing residue
  和退出条件

### 9.2 只有做到下面这条，才算
`blackhole.segment_kind`
完全删除

- 运行时 / build 端
  已经不再通过原始 TIR body
  按
  `AttrStmt("blackhole.segment_kind")`
  切 segment body
