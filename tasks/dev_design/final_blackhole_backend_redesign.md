# TileLang Blackhole 后端最终重构设计

## 基本信息

- **文档ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19
- **状态**: 当前唯一权威总体设计
- **适用范围**: `tilelang_repo` Blackhole 后端、host/device 主链、运行时执行路径、相关阶段设计

## 1. 当前目标

Blackhole 后端当前的正式目标收敛为三点：

1. **后端主产物是 `ExecutableSpec`，不是单个 kernel 字符串**
2. **实现路径必须复用 TileLang / TVM 的正式 PrimFunc/TIR 与 host/device 主链**
3. **正式执行路径必须是 `BlackholeModule` 进程内 direct host path，而不是 external runner**

因此 Blackhole 后端不再以“写出一个能跑的 TT-Metal kernel 字符串”或“`spec.json -> runner` 能跑”为完成标准，而是以：

- DSL / PrimFunc / TIR 主链保持成立
- tile/dataflow/shared/block 语义在正确层级被保留并转换
- split 后 device kernel 产出正式 execution plan / memory plan / kernel ABI
- `ExecutableSpec -> BlackholeModule direct host materialization -> TT-Metal launch`

作为正式目标。

## 2. 当前问题判断

当前代码已经完成了部分基础工作：

- `ExecutableSpec`、`rt_mod_blackhole`、`BlackholeModule` 已存在
- Blackhole 已重新接回 `AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch`
- copy 已开始从 runtime 特化迁回 pass / builtin / codegen 主链
- single-core staged copy 的最小 direct-call 路径已经闭环

但当前最大的结构问题已经收敛为四点：

1. **split 前的语义规划和 split 后的正式 plan 提取还没有彻底分层**
   - 当前仍偏向依赖 `LowerBlackholeOps` 从晚期 staged loop 恢复 copy 语义
2. **execution plan / memory plan 还没有作为显式中间层稳定建模**
   - `blackhole.core_plan` 仍偏摘要
   - `blackhole.cb_configs` 仍偏 MVP allocator
3. **正式执行路径还没有完全切到 `BlackholeModule` direct host path**
   - external runner 仍残留为过渡主路径的一部分
4. **部分中后段通用 pass 还没有接回正式主线**
   - `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等仍会打断当前 copy 识别

因此当前阶段的核心任务已经不是“继续补一个能跑的 copy/gemm bring-up”，而是：

- **先把 Blackhole 重新接回 TileLang / TVM 的正式主链**
- **明确 split 前语义规划 / split 后正式 plan 提取 / host-side materialization 三层**
- **再在这套结构上推进 copy / GEMM**

## 3. 正式架构

### 3.1 总体结构

```text
TileLang DSL
  -> PrimFunc / TIR
  -> LowerAndLegalize
  -> split 前 Blackhole 语义规划
      -> LowerTileOp(Blackhole-aware)
      -> 通用 planning / normalize 子集
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> AnnotateReadOnlyParams
  -> MakePackedAPI
  -> LowerDeviceKernelLaunch
  -> split 后 Blackhole 正式 plan 提取
      -> LowerBlackholeOps
      -> PlanBlackholeCB
      -> AssignBlackholeCores
  -> rt_mod_blackhole
      -> Extract ExecutableSpec
      -> Emit kernel source(s)
      -> Build BlackholeModule(spec)
  -> BlackholeModule
      -> Program / CreateCircularBuffer / CreateKernel
      -> SetRuntimeArgs / ConfigureDeviceWithProgram / LaunchProgram
      -> readback
```

### 3.2 设计原则

1. **以后端执行模型约束 lowering，不以后端打印器约束执行模型**
2. **优先复用现有 PrimFunc/TIR 与 host/device 主链，只在少量 target-aware 边界定制**
3. **host/device 划分与 Packed API 参数语义沿用 TileLang / TVM 正式模型**
4. **split 前做语义规划，split 后做正式 plan 提取，host side 做 materialization**
5. **compile-time attrs/schema 与 runtime args 必须严格分层**
6. **逻辑 block/grid 语义应保留在 TIR/pass 中，不在 runtime 侧重建**
7. **multi-core 主要由 host/runtime materialize，不由 codegen 主导**
8. **`SplitBlackholeKernel`、`blackhole_module_direct.cc`、external runner 都不再是主路径设计前提**

### 3.3 三层分工

#### A. split 前语义规划

职责：

- 保留 copy/gemm/tile/shared/pipeline/block 语义
- 稳定成 Blackhole 后续仍可识别的 TIR

产物：

- `Blackhole-preserving TIR`

不在这一层做的事：

- 不生成最终 `runtime_args`
- 不分配最终 `cb_id`
- 不生成最终 `core_plan`
- 不定义 host-side launch/materialization 细节

#### B. split 后正式 plan 提取

职责：

- 面向 split 后 device kernel
- 提取 runtime ABI、memory plan、execution plan

产物：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.core_plan`

#### C. host-side materialization

职责：

- 直接 materialize TT-Metal host objects
- 按正式 plan 完成 launch/readback

产物：

- 真正执行结果

## 4. 模块边界

### 4.1 `LowerTileOp`

这是 Blackhole 最关键的 split 前 target-aware 接入点。

长期要求：

- 继续在整个 PrimFunc 上工作
- 保留普通控制流、标量计算、索引与边界逻辑
- 对 `tl.copy` / `tl.gemm` 等 tile 语义增加 Blackhole-aware 分支
- 不再把这些语义完全压碎后再让 Blackhole 从晚期 loop 恢复

职责：

- 只负责 split 前语义规划
- 输出 Blackhole-preserving TIR

### 4.2 `LowerBlackholeOps`

职责：

- 消费 split 后 device kernel
- 提取 segment / runtime arg schema / CB requirements
- 产出 `blackhole.segment_plan`、`blackhole.runtime_args`、`blackhole.cb_requirements`

不再承担：

- 从晚期普通 loop 中恢复绝大多数 copy/gemm 语义
- 猜 PrimFunc 参数结构
- 用 runtime 特判兜底 device kernel 结构

### 4.3 `PlanBlackholeCB`

职责：

- 从 `blackhole.cb_requirements` 收敛到正式 `blackhole.cb_configs`
- 形成 per-device-kernel memory plan

至少覆盖：

- `cb_id`
- role
- page size / num pages / format
- 生命周期与复用
- per-core L1 峰值约束

### 4.4 `AssignBlackholeCores`

职责：

- 保留并消费逻辑 grid/block 语义
- 生成 host/runtime 消费的 execution plan

至少覆盖：

- `logical_grid_x/y`
- linearization policy
- `physical_cores`
- `work_packets`

### 4.5 `rt_mod_blackhole`

职责：

- 从 split 后 device kernel 及其 `blackhole.*` attrs 提取 `ExecutableSpec`
- 构造 `BlackholeModule(spec)`

约束：

- 只消费 pass 产出的 device-side 语义和 schema
- 不再长期把“没有 `calling_conv` 的 PrimFunc”视为正式 device kernel 模型
- 不再定义 PrimFunc 参数类别、host/device 边界或 runtime 参数意义

### 4.6 `BlackholeModule`

职责：

- 作为 TVM runtime adapter / host-side 执行载体
- 在进程内 direct materialize TT-Metal host objects
- 完成 launch / readback

约束：

- 不再通过 external runner 作为正式主路径
- 不再长期承担 Packed API 或 host/device 语义定义
- 输入输出 buffer、scalar、dynamic shape 如何映射到 runtime args，必须由 PrimFunc + pass schema 决定
- 不再按位置规则猜 ABI

### 4.7 external runner

状态：

- 降级为 bring-up/debug/protocol-check 工具

约束：

- 不是正式执行路径
- 不是阶段完成标准
- 后续可删除

## 5. 核心协议

### 5.1 `ExecutableSpec`

当前主协议结构保持：

- `CBConfig`
- `CorePlan`
- `KernelArgSpec`
- `KernelSpec`
- `ExecutableSpec`

后续工作重点不是继续改协议名字，而是让这些字段的来源真正回到 split 后 device-side attrs/schema。

允许扩展的方向：

- `CorePlan` 补足 logical grid / work packet 语义
- `KernelSpec` 补足 kernel-level ABI / launch information
- `CBConfig` 补足 memory hierarchy materialization 所需字段

不允许扩展的方向：

- 为 runner 新增脱离 TIR 的平行 ABI
- 在 module/runtime 里定义第二套 host/device 语义

### 5.2 Attr Schema

当前主线只保留：

- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.core_plan`
- `blackhole.segment_plan`
- `blackhole.runtime_args`

旧协议不再扩展。

## 6. 算子策略

### 6.1 Copy

copy 的正式目标是：

- 以 TileLang 原始 `T.copy` 语义为验收对象
- 在 Blackhole 上收敛成 `global -> shared/CB -> global`
- 经历 split 前语义规划、host/device 主链、split 后正式 plan 提取、`BlackholeModule` direct host path 后执行

正式验证至少覆盖：

- `32x32`
- `32x64`
- `64x32`
- 至少一个 `grid > 1` 且 `bx/by` 参与索引的 case
- 至少一个总数据量大于 `1.5MB` 的 large-shape copy case

### 6.2 GEMM

GEMM 的后续集成必须复用与 copy 相同的结构：

- 不再接受 runtime 侧大块 GEMM 特化
- reader / compute / writer 语义必须主要来自 split 前语义规划和 split 后正式 plan 提取

### 6.3 Multi-core

multi-core 的主要实现位置保持不变：

- host/runtime
- 基于 `CorePlan`
- materialize per-core runtime args / work packets

## 7. 分阶段路线

### Stage 0: 协议与执行载体

目标：

- attrs 收口到 `blackhole.*`
- 引入 `ExecutableSpec`
- 改造 `rt_mod_blackhole`
- 引入 `BlackholeModule`

状态：

- **已基本完成**

### Stage 1: single-core copy bring-up

目标：

- 证明最小 copy 路径能执行

状态：

- **已完成**

说明：

- 该阶段的 runner/scratch/固定 schema 只视为 bring-up 过渡路径，不再扩大为正式主线

### Stage 2A: pass 主链接入收正

目标：

- 复用 TileLang / TVM 正式主链
- 建立 split 前语义规划 / split 后 plan 提取 / host-side materialization 三层

### Stage 2B: single-core copy 正式主链

目标：

- copy 由 pass 主导并走 `BlackholeModule` direct host path 完成正式 E2E

### Stage 2C: single-core GEMM 语义集成

目标：

- 用与 copy 相同的结构接入 GEMM

### Stage 2D: single-core true E2E

目标：

- copy 与 GEMM 都由正式主链执行并完成 direct host-device E2E

### Stage 3: multi-core runtime 调度

目标：

- `CorePlan`
- per-core runtime args
- multi-core execution / memory plan materialization

## 8. 正式验收标准

正式阶段完成标准只看：

- TileLang 正式编译产物对外暴露的 host callable
- 通过 `BlackholeModule` 进程内 direct host path 执行
- 由模块内部完成 TT-Metal host materialization / launch / readback
- 与 PyTorch 参考结果一致

以下都不再是正式阶段完成标准：

- `spec.json -> runner`
- external runner 单独执行通过
- 手动按 `"main"` 名称去调用内部符号
