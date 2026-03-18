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
- `blackhole.target_mode` 已从当前主协议移除：
  - pass 不再产出该 attr
  - `ExecutableSpec` / `spec.json` / runner 不再依赖该字段
  - copy fallback 改为按 device-side builtin/schema 判断，而不是模式字符串
- copy true E2E 新增进展：
  - `spec.json -> runner` 已在 TT-Sim 下通过
  - `artifact.codegen_mod["main"](...)` 已在 TT-Sim 下通过
  - `32x32 float16` staged `T.copy(global -> shared -> global)` 输出与 PyTorch 参考一致
- 当前仍然存在的主要结构问题：
  - Blackhole 还没有把全部中后段通用规范化 pass 接回主线
  - `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 仍会打断当前 copy staged-lowering
  - `rt_mod_blackhole` / `BlackholeModule` 的 host/device 边界虽然已经收正，但 GEMM 还没有复用这条新主线
- 当前新增进展：
  - Blackhole `lower()` 主路径已恢复：
    - `AnnotateDeviceRegions`
    - `SplitHostDevice`
    - `MakePackedAPI`
    - `LowerDeviceKernelLaunch`
  - `lower()` 产物已重新分成：
    - host `main`
    - device `main_kernel`
  - `target.build.tilelang_blackhole[_without_host]` 现已消费 `host_mod + lowered device_mod` 的组合模块
  - `BlackholeModule` 对外 entry 的参数签名已重新对齐对应 device kernel，不再错误继承 Packed API 的 4 参低层签名
- 当前仍然保留的阶段限制：
  - 为了保持 Stage 2B copy 的 staged-copy 识别，`FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等会破坏当前 copy 形态的 pass 还没有提前恢复到 `LowerBlackholeOps` 之前

## 分阶段任务

| 阶段 | 目标 | 状态 | 当前重点 |
|------|------|------|----------|
| Stage 0 | 协议与执行载体 | ✅ 已基本完成 | `ExecutableSpec`、runner 协议、module/runner 主路径已落地 |
| Stage 1 | single-core copy 执行闭环 | ✅ 已完成 | TT-Sim 下 runner 与 direct-call 都已通过 |
| Stage 2A | pass 主链接入收正 | 🔄 进行中 | 恢复通用 TIR / host-device / Packed API pass 主线 |
| Stage 2B | single-core copy 语义集成 | 🔄 进行中 | copy 已完成 true E2E，但仍需继续接回剩余通用 pass |
| Stage 2C | single-core GEMM 语义集成 | ⏳ 未开始 | 用与 copy 相同的结构接入 GEMM |
| Stage 2D | single-core true E2E | ⏳ 未完成 | copy + GEMM 都由 pass 主导并完成 true E2E |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | `CorePlan`、per-core runtime args、多核执行 |

## Stage 2A 任务拆分

### 任务 1: 恢复 pass 主链

- 恢复 Blackhole 对通用 TIR 规范化 pass 的复用
- 去掉 `OptimizeForTarget` 中对 Blackhole 的长期 early return 设计
- 当前状态：
  - 已去掉 Blackhole 的长期 early return
  - 已恢复一条“host/device 主链优先、copy 识别安全”的 pass 子链
  - `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 仍待在后续阶段重新接回

### 任务 2: 恢复 host/device 主链

- 恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch` 或其 Blackhole 分支
- 停止长期依赖“没有 `calling_conv` 也可当 device kernel”的路径
- 当前状态：
  - 已恢复上述四个 pass
  - `is_device_call` / Blackhole build 入口已经改为以 split 后的 `calling_conv` 语义为主
  - 仍保留对旧 unsplit device attrs 的保守兼容检测

### 任务 3: 收正 runtime/module 边界

- `rt_mod_blackhole` 只消费 device-side PrimFunc 和 pass schema
- `BlackholeModule` 不再定义 PrimFunc 参数类别或 host/device 边界
- 当前状态：
  - `target.build.tilelang_blackhole[_without_host]` 已不再把 host Packed API PrimFunc 当成 device kernel 做 codegen
  - `ExtractBlackholeFuncInfo()` 已改为：
    - 从 split 后 device kernel 提取 `ExecutableSpec`
    - 用 host entry -> launched kernel 绑定恢复对外 entry 名和用户参数签名

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

1. 让 `LowerBlackholeOps` 直接消费 split 后 device kernel 的更稳定 staged-copy 形态。
2. 在不破坏当前 copy true E2E 的前提下，逐步把 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等 pass 接回 Blackhole 主链。
3. 用 copy 已打通的 `host entry -> device kernel -> spec -> runner` 结构推进 GEMM。
4. 继续把真执行测试按环境 gate 分层，避免把 TT-Sim 环境问题记成编译链问题。

## 当前活动设计文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage2_pass_reuse_matrix.md`
- `tasks/dev_design/stage2_single_core_pass_integration.md`
