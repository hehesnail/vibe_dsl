# TileLang Blackhole 后端最终重构设计

## 基本信息

- **文档ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19
- **状态**: 当前唯一权威总体设计
- **适用范围**: `tilelang_repo` Blackhole 后端、runner 协议、相关测试与阶段设计

## 1. 当前目标

Blackhole 后端当前的正式目标已经收敛为两点：

1. **后端主产物是 `ExecutableSpec`，不是单个 kernel 字符串**
2. **实现路径必须尽量复用 TileLang / TVM 现有 PrimFunc/TIR 主链，而不是自建一条平行流水线**

因此 Blackhole 后端不再以“写出一个能跑的 TT-Metal kernel 字符串”为完成标准，而是以：

- DSL / PrimFunc / TIR 主链保持成立
- tile 语义在正确层级被保留并转换
- `ExecutableSpec -> BlackholeModule -> runner` 负责执行

作为正式目标。

## 2. 当前问题判断

当前代码已经完成了一部分基础工作：

- `ExecutableSpec`、runner 协议、`BlackholeModule` 外部执行路径已经存在
- Stage 1 single-core copy 的 runner 路径和 direct-call 路径都跑通过
- copy 已开始从 runtime 特化向 pass / builtin / codegen 主链迁移

但当前最大的结构问题已经从“协议没落地”转成了下面三点：

1. **Blackhole 虽已重新接回 host/device 主链，但还没有把全部中后段通用 pass 收回正式主路径**
   - `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等仍会打断当前 copy staged-lowering
2. **execution / memory / launch plan 还没有被显式建模成正式中间层**
   - `rt_mod_blackhole` / `BlackholeModule` / runner 仍残留一部分过渡期猜测逻辑
3. **`LowerBlackholeOps` 的主接入点仍偏晚**
   - 当前很多 copy 语义仍是从 `LowerTileOp` 之后的 staged loop 恢复，而不是在更稳定的 target-aware lowering 边界上保留

因此当前阶段的核心任务已经不是“继续补一个能跑的 copy/gemm”，而是：

- **先把 Blackhole 重新接回 TileLang / TVM 的 TIR 主链**
- **再在这个基础上推进 copy / gemm 的语义集成**

## 3. 正式架构

### 3.1 总体结构

```text
TileLang DSL
  -> PrimFunc / TIR
  -> 通用 legalize / normalize / optimize passes
  -> LowerTileOp(Blackhole-aware)
  -> AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch
  -> LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores
  -> Blackhole execution plan
      -> logical grid / work packets
      -> memory objects / CB configs
      -> kernel ABI / runtime arg schema
  -> BuildTileLangBlackholeWithoutHost
      -> Extract ExecutableSpec
      -> Emit kernel source(s)
      -> Build BlackholeModule(spec)
  -> BlackholeModule
      -> serialize spec + tensor/scalar args
      -> invoke runner
  -> runner
      -> CreateProgram / CB / Kernel / RuntimeArgs
      -> enqueue and read back
```

### 3.2 设计原则

1. **以后端执行模型约束 lowering，不以后端打印器约束执行模型**
2. **优先复用现有 PrimFunc/TIR pass 主链，只在少量 target-aware 边界定制**
3. **host/device 划分与 Packed API 参数语义尽量沿用 TileLang / TVM 现有模型**
4. **compile-time args 与 runtime args 必须严格分层**
5. **逻辑 block/grid 语义应保留在 TIR/pass 中，不在 runtime 侧重建**
6. **host 侧主要 materialize 已分析好的执行计划，不重新发明 memory plan / launch plan**
7. **multi-core 主要由 host/runtime materialize，不由 codegen 主导**
8. **`SplitBlackholeKernel` 和 `blackhole_module_direct.cc` 都不再是主路径设计前提**

### 3.3 Blackhole execution plan

Blackhole 与 CUDA 的关键差异，不在于“是否有 host”，而在于：

- CUDA 的逻辑 block/grid 语义可以较直接映射到硬件 CTA/grid launch
- Blackhole 需要额外一层 `logical block -> physical core/work packet` 映射

因此 Blackhole 主路径里必须显式存在一层 execution plan，用来承接：

- 逻辑 grid / block 语义
- segment / kernel graph
- memory hierarchy plan
- kernel ABI
- core scheduling / work distribution

这层 plan 的主要来源应是 TIR / target-aware passes，而不是 `BlackholeModule` 或 runner 猜测。

哪怕当前只考虑 single-core，也应保持：

- 逻辑 grid 仍来自 `T.Kernel(...)`
- single-core 只是 `physical_core_count = 1`
- 一个物理 core 可以顺序处理多个 logical blocks
- `blockIdx.x/y` 不应在 codegen 中被长期常量化成 `0`

## 4. 模块边界

### 4.1 `LowerTileOp`

这是 Blackhole 最关键的 target-aware 接入点。

长期要求：

- 继续在整个 PrimFunc 上工作
- 保留普通控制流、标量计算、索引与边界逻辑
- 对 `tl.copy` / `tl.gemm` 等 tile 语义增加 Blackhole-aware 分支
- 不再把这些语义完全压碎后再让 Blackhole 从晚期 loop 恢复

### 4.2 `LowerBlackholeOps`

职责：

- 消费 `LowerTileOp(Blackhole-aware)` 之后的 Blackhole-preserving TIR
- 提取 segment / CB requirements / runtime arg schema
- 产出 `blackhole.*` attrs 和中层 builtin

不再承担：

- 从晚期普通 loop 中恢复绝大多数 copy/gemm 语义
- 猜 PrimFunc 参数结构
- 用 runtime 特判兜底 device kernel 结构

### 4.3 `PlanBlackholeCB`

职责保持明确：

- 从 `blackhole.cb_requirements` / segment 信息收敛到 runtime-ready `blackhole.cb_configs`
- 逐步扩展为显式 memory-plan 的 materialization 层，覆盖：
  - CB page size / num_pages / format / role
  - scratch / intermediate L1 对象
  - 生命周期与峰值容量约束

### 4.4 `AssignBlackholeCores`

职责保持明确：

- 保留并消费逻辑 grid/block 语义
- 生成 host/runtime 消费的 core scheduling / work distribution plan
- 不在 codegen 中固化 `blockIdx -> 常量 0`
- 不在 runner 中长期写死单核 `{0, 0}` 执行路径

### 4.5 `rt_mod_blackhole`

职责：

- 从 device-side PrimFunc 和 attrs 提取 `ExecutableSpec`
- 构造 `BlackholeModule(spec)`

约束：

- 只消费 pass 产出的 device-side 语义和 schema
- 不再长期把“没有 `calling_conv` 的 PrimFunc”视为正式 device kernel 模型
- 不再定义 PrimFunc 参数类别、host/device 边界和 runtime 参数意义

### 4.6 `BlackholeModule`

职责：

- 作为 TVM runtime adapter / host-side 执行载体
- 按 spec 序列化 `spec.json + input.bin + output.bin`
- 调用 runner

约束：

- 不再长期承担 Packed API 或 host/device 语义
- 输入输出 buffer、scalar、dynamic shape 参数如何映射到 runtime args，必须由 PrimFunc + pass schema 决定
- 不再按“最后一个 buffer 是 output”这类位置规则长期猜 ABI

### 4.7 runner

职责：

- 读取 `ExecutableSpec`
- materialize TT-Metal host-side对象：
  - `Program`
  - CB / memory objects
  - kernel
  - compile-time args / runtime args
- 执行并回读

runner 只消费协议，不理解 PrimFunc / TIR 语义。

runner 可以是外部进程，但它只能是 execution plan 的执行器，不能再承担：

- 猜 PrimFunc 参数类别
- 猜 input / output / scalar 位置
- 猜逻辑 block 如何映射到 core
- 猜 CB / scratch / launch graph

## 5. 核心协议

### 5.1 `ExecutableSpec`

当前主协议结构保持：

- `CBConfig`
- `CorePlan`
- `KernelArgSpec`
- `KernelSpec`
- `ExecutableSpec`

后续工作重点不是继续改协议名字，而是让这些字段的来源真正回到 pass / device-side schema。

允许扩展的方向是：

- 在 `CorePlan` 中补足 logical grid / work packet 语义
- 在 `KernelSpec` 中补足 kernel-level ABI / launch information
- 在 `CBConfig` 中补足 memory hierarchy materialization 所需字段

不允许继续扩展的方向是：

- 在 runner / module 里新增一套脱离 TIR 的手写固定 ABI

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

copy 的正式目标不再是“simple assignment 能猜成 copy”，而是：

- 以 TileLang 原始 `T.copy` 语义为验收对象
- 在 Blackhole 上收敛成 `global -> shared/CB -> global` 形式
- 经过 TIR 主链、host/device 主链、Blackhole device passes 后形成 `ExecutableSpec`

### 6.2 GEMM

GEMM 的后续集成必须复用与 copy 相同的结构：

- 不再接受 runtime 侧大块 gemm 特化
- reader / compute / writer 语义必须主要来自 pass 产物

### 6.3 Multi-core

multi-core 的主要实现位置保持不变：

- host/runtime
- 基于 `CorePlan`
- materialize per-core runtime args

## 7. 分阶段路线

### Stage 0: 协议与执行载体

目标：

- 统一 attrs 到 `blackhole.*`
- 引入 `ExecutableSpec`
- 改造 `rt_mod_blackhole` / `BlackholeModule`
- runner 切到 `spec.json` 协议

状态：

- **已基本完成**

### Stage 1: single-core copy 执行闭环

目标：

- 在 TT-Sim 上跑通最小 single-core copy
- 验证 `ExecutableSpec -> BlackholeModule -> runner` 主执行路径

状态：

- **已完成**

### Stage 2A: pass 主链接入收正

目标：

- 恢复 Blackhole 对通用 TIR / host-device / Packed API pass 的复用
- 结束当前 `OptimizeForTarget` early return 的长期设计
- 恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch` 或其 Blackhole 分支

状态：

- **当前进行中**

### Stage 2B: single-core copy 语义集成

目标：

- 在已收正的 pass 主链上完成 copy 的 Blackhole-aware lowering
- 让 copy 的 runtime arg / CB / segment / codegen 主要来自 pass 产物

状态：

- **未完成，已有中间态实现**

### Stage 2C: single-core GEMM 语义集成

目标：

- 用与 copy 相同的结构承接 GEMM
- 停止 runtime 侧 gemm 特化路径扩张

状态：

- **未开始**

### Stage 2D: single-core true E2E

目标：

- copy 与 GEMM 都在 TT-Sim 或真实设备上完成 true E2E
- 且其关键执行语义主要来自 pass，不来自 runtime/module 特判

状态：

- **未完成**

### Stage 3: multi-core runtime 调度

目标：

- runner/runtime 消费 `CorePlan`
- 支持 per-core runtime args
- multi-core copy / GEMM 真执行

状态：

- **未开始**

## 8. 验收标准

### Stage 2 之前

- 不再把“能生成 kernel 字符串”视为阶段完成
- 不再把 runtime 特判路径视为正式编译器路径

### Stage 2 完成

必须同时满足：

- Blackhole 已重新接回 TIR / host-device / Packed API 主链
- copy / GEMM 的关键执行语义主要来自 pass 产物
- `ExecutableSpec` 成为 pass 结果的执行载体
- single-core copy / GEMM 完成 true E2E

### Stage 3 完成

- multi-core copy / GEMM true E2E
- per-core runtime args 与 `CorePlan` 正常工作

## 9. 当前配套文档

- `tasks/progress.md`: 当前阶段状态与任务流转
- `tasks/dev_design/stage2_pass_reuse_matrix.md`: pass 复用矩阵与接入边界
- `tasks/dev_design/stage2_single_core_pass_integration.md`: Stage 2 活动阶段设计
- `tasks/dev_design/stage0_executable_spec_attr_alignment.md`: Stage 0 历史设计
- `tasks/dev_design/stage1_single_core_copy_closure.md`: Stage 1 历史设计
