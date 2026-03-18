# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态、任务拆分和当前下一步。

## 当前阶段

- **阶段**: Stage 2A pass 主链接入收正
- **日期**: 2026-03-19
- **当前目标**: 先把 Blackhole 重新接回 TileLang / TVM 的 PrimFunc/TIR pass 主链与 host/device 主链，再继续推进 copy / GEMM 的语义集成

## 当前状态判断

- Stage 0 的协议和执行载体已经落地：
  - `ExecutableSpec`
  - `BlackholeModule`
  - `spec.json -> runner`
- Stage 1 single-core copy 执行闭环已完成：
  - runner 路径已在 TT-Sim 上通过
  - Python direct-call 路径已在 TT-Sim 上通过
- copy 已开始从 runtime 特化迁回 pass / builtin / codegen 主链：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_configs`
  - `tl.blackhole.read_tile_to_cb / write_tile_from_cb`
- 但当前最主要的结构问题已经明确：
  - Blackhole 仍在 `OptimizeForTarget` 中 early return
  - 通用 TIR 规范化、`SplitHostDevice`、`MakePackedAPI`、`LowerDeviceKernelLaunch` 仍被旁路
  - `rt_mod_blackhole` / `BlackholeModule` 还在间接承担部分 PrimFunc 参数和 host/device 语义

## 分阶段任务

| 阶段 | 目标 | 状态 | 当前重点 |
|------|------|------|----------|
| Stage 0 | 协议与执行载体 | ✅ 已基本完成 | `ExecutableSpec`、runner 协议、module/runner 主路径已落地 |
| Stage 1 | single-core copy 执行闭环 | ✅ 已完成 | TT-Sim 下 runner 与 direct-call 都已通过 |
| Stage 2A | pass 主链接入收正 | 🔄 进行中 | 恢复通用 TIR / host-device / Packed API pass 主线 |
| Stage 2B | single-core copy 语义集成 | ⏳ 未完成 | 在收正后的主链上完成 copy 的 Blackhole-aware lowering |
| Stage 2C | single-core GEMM 语义集成 | ⏳ 未开始 | 用与 copy 相同的结构接入 GEMM |
| Stage 2D | single-core true E2E | ⏳ 未完成 | copy + GEMM 都由 pass 主导并完成 true E2E |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | `CorePlan`、per-core runtime args、多核执行 |

## Stage 2A 任务拆分

### 任务 1: 恢复 pass 主链

- 恢复 Blackhole 对通用 TIR 规范化 pass 的复用
- 去掉 `OptimizeForTarget` 中对 Blackhole 的长期 early return 设计

### 任务 2: 恢复 host/device 主链

- 恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch` 或其 Blackhole 分支
- 停止长期依赖“没有 `calling_conv` 也可当 device kernel”的路径

### 任务 3: 收正 runtime/module 边界

- `rt_mod_blackhole` 只消费 device-side PrimFunc 和 pass schema
- `BlackholeModule` 不再定义 PrimFunc 参数类别或 host/device 边界

## Stage 2B 任务拆分

- 在已恢复的主链上，给 `LowerTileOp` 增加 Blackhole-aware copy lowering
- 让 `LowerBlackholeOps` 从 Blackhole-preserving TIR 提取：
  - segment
  - CB requirements
  - runtime args
- 让 copy 的 spec/codegen 主要由 pass 产物驱动

## Stage 2C 任务拆分

- 给 `LowerTileOp` 增加 Blackhole-aware GEMM lowering
- 停止为 GEMM 扩展 runtime 侧特化
- 让 GEMM 复用 copy 已建立的 schema / CB / spec 路径

## 当前下一步

1. 按 `stage2_pass_reuse_matrix.md` 收正 Blackhole 对现有 pass 主链的接入。
2. 优先恢复 `SplitHostDevice`、`MakePackedAPI` 与 `LowerDeviceKernelLaunch` 相关路径。
3. 在主链接入恢复后，再继续推进 copy 的 tile-range / dataflow 语义分析。
4. 继续把真执行测试按环境 gate 分层，避免把 TT-Sim 环境问题记成编译链问题。

## 当前活动设计文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage2_pass_reuse_matrix.md`
- `tasks/dev_design/stage2_single_core_pass_integration.md`
