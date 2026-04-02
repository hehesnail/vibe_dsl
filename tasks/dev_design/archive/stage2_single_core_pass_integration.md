# Stage 2 Single-Core Pass Integration 设计

> 支撑设计说明：
> 本文保留 Stage 2 分层与阶段拆分的设计背景；
> 当前主任务请看 `tasks/dev_design/stage2d_gemm_integration.md`，
> 当前状态请看 `tasks/progress.md`。

## 基本定位

- **状态**: Stage 2 支撑设计（仍有效，但当前主任务已转到 `stage2d_gemm_integration.md` / `stage2e_blackhole_device_resource_semantics.md`）
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **前置接入矩阵**: `stage2_pass_reuse_matrix.md`

本文件只描述 Stage 2 的目标、阶段拆分和验收标准，不重复承载总体架构结论。

## 阶段目标

Stage 2 的正式目标已经收紧为：

- **先把 Blackhole 重新接回 TileLang / TVM 的正式 PrimFunc/TIR 与 host/device 主链**
- **再在这条主链上完成 single-core copy 与 GEMM 的语义集成**
- **最后通过 `BlackholeModule` direct host path 完成 true E2E**

因此 Stage 2 不再接受：

- copy 长期停留在 runtime 专用 emitter
- GEMM 继续靠 runtime 侧特化拼语义
- 绕过 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 继续推进
- 把 external runner 或 `spec.json -> runner` 当成正式 E2E 标准

## 阶段拆分

### Stage 2A: pass 主链接入收正

目标：

- 恢复 Blackhole 对通用 TIR / host-device / Packed API 主链的复用
- 建立 split 前语义规划 / split 后正式 plan 提取 / host-side materialization 三层

任务：

- 按 `stage2_pass_reuse_matrix.md` 逐项收正 pass 接入
- 恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
- 收正 `rt_mod_blackhole` / `BlackholeModule` 的边界

完成标准：

- Blackhole 主路径重新回到 TIR / host-device / Packed API 正式主链
- 不再长期依赖“无 `calling_conv` 也可当 device kernel”的路径

### Stage 2B: single-core copy 正式主链

目标：

- 在已收正的主链上完成 copy 的 split 前语义规划与 split 后正式 plan 提取

任务：

- 在 `LowerTileOp` 中保留 copy 的 Blackhole-preserving 语义
- `LowerBlackholeOps` 提取：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
- `PlanBlackholeCB` 生成正式 `blackhole.cb_configs`
- `AssignBlackholeCores` 生成正式 `blackhole.core_plan`
- `BlackholeModule` direct host path 消费这些 plan 执行

完成标准：

- copy 的 runtime args / memory plan / execution plan 主要来自 pass 产物
- Blackhole 正式执行只经过 `BlackholeModule` direct host path

### Stage 2C: single-core GEMM 语义集成

目标：

- 用与 copy 相同的结构接入 GEMM

任务：

- 在 `LowerTileOp` 中保留 GEMM 的 Blackhole-preserving 语义
- `LowerBlackholeOps` 提取 GEMM 的 reader / compute / writer schema
- `PlanBlackholeCB` 与 `AssignBlackholeCores` 为 GEMM 生成同类 plan
- 停止扩展 runtime 侧 GEMM 特化

完成标准：

- GEMM 的关键执行语义主要来自 pass，而不是 runtime/module 特判

### Stage 2D: single-core true E2E

目标：

- copy 与 GEMM 都通过正式 host-device 主路径完成 true E2E

完成标准：

- TileLang 暴露的正式 host callable 可以直接执行编译产物
- `BlackholeModule` 在进程内完成 TT-Metal host materialization / launch / readback
- copy / GEMM 的关键执行语义主要来自 pass 产物

## 当前仍有效的边界

当前允许保留的过渡项：

- 无。legacy runner 路径已删除，当前只保留 direct host path 主线

当前不允许继续扩大的过渡项：

- copy runtime 特化继续扩大为正式主路径
- GEMM 继续复制 copy 阶段的 runtime 特化做法
- `rt_mod_blackhole` 继续承担 kernel 语义恢复、PrimFunc 参数分类和 host/device 语义定义
- external runner 继续作为正式执行路径

## 历史阶段收正目标

本文件形成时，Stage 2B 的具体目标已经收紧为：

- 继续保留 single-core copy true execution
- 把 copy 语义从“只对最小 case 成立”收成“按实际 DSL tile shape / logical block / memory plan 推导”
- 同时收正：
  - logical block 语义
  - single-core work distribution
  - memory plan
  - host launch ABI

相关设计见：

- `stage2_blackhole_logical_block_launch_plan.md`

当时必须优先解决的问题：

- `blockIdx.x/y -> 0` 的 codegen 常量化
- `blackhole.core_plan` 仍是摘要信息
- `PlanBlackholeCB` 仍偏 MVP allocator
- ~~`BlackholeModule` 还没有成为唯一正式执行路径~~ → **已修正（2026-03-20）**：`ExecuteDirect()` 已补全 CB 创建 + runtime args + work-packet 迭代

Phase 3 split-before 语义规划方案确认：

- **方案 A（推荐）**：新增 `AnnotateBlackholeCopySemantics` pass，位于 `LowerTileOp` 之后、`FlattenBuffer` 之前
  - 识别 copy pattern（BufferStore where value is BufferLoad），添加 `blackhole.copy_direction` annotation
  - 不修改 `LowerTileOp` 核心降级逻辑
- 方案 B（备选）：修改 `LowerTileOp` 加 Blackhole 分支，风险较高

通用 pass 回收策略：

- shared-scope buffer 需要豁免 `StorageRewrite` 的合并
- `FlattenBuffer` 的 shared-scope 跳过分配已在 `codegen_blackhole.cc` 中处理
- 每个 pass 单独回收 + 单独测试

## 历史阶段进展

- Blackhole 已恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
- `lower()` 产物已重新分成：
  - host `main`
  - device `main_kernel`
- `target.build.tilelang_blackhole[_without_host]` 已消费 `host_mod + lowered device_mod`
- copy 已开始产出：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
  - `blackhole.cb_configs`
  - `tl.blackhole.read_tile_to_cb / write_tile_from_cb`
- staged copy 的最小 direct execution 已覆盖：
  - `32x32`
  - `32x64`
  - `64x32`

但当前路径仍主要证明：

- minimal copy path 已经 bring-up
- 还不能视为正式 compiler/runtime path 已完成

## 验证方式

### 结构验证

- Blackhole target 恢复正式主链验证：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
- split 后稳定产出：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
  - `blackhole.cb_configs`
  - `blackhole.core_plan`

### copy / GEMM 语义验证

- pass 产出的 attrs / builtin / segment / memory plan / execution plan 能支撑 `ExecutableSpec`
- `ExecutableSpec` 中的 kernels / runtime args / CB / core plan 不再主要来自 runtime 猜测

### 执行验证

正式执行验证只覆盖：

- TileLang 暴露的正式 host callable
- `BlackholeModule` direct host path

copy 必须覆盖：

- `32x32`
- `32x64`
- `64x32`
- 至少一个 `grid > 1` 且 `bx/by` 参与索引的 case
- 至少一个总数据量大于 `1.5MB` 的 large-shape copy case

并补一个负例：

- 构造 per-core memory plan 超出 `1572864` bytes 的 case
- 编译期直接失败

环境不满足时应显式 skip，而不是混成编译链失败。
