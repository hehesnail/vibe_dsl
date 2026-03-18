# TileLang Blackhole 后端实现状态

> 当前唯一设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只保留实现状态摘要，不再维护任何替代性架构说明。

- **日期**: 2026-03-18
- **当前阶段**: Stage 0 协议重构
- **状态**: 新总体设计已定稿，旧主路径已移除

## 当前判断

### 已确认保留

- `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores` 已接入 lowering pipeline
- builtin visitor 框架保留
- 外部 runner 工程边界保留

### 已确认退出主路径

- 单 kernel 字符串作为后端主产物
- `SplitBlackholeKernel` 作为当前前置
- `blackhole_module_direct.cc` 作为主实现方向
- 旧 runner 命令行协议

## 当前重点

1. 引入 `ExecutableSpec`
2. 统一 attrs 到 `blackhole.*`
3. 重构 `rt_mod_blackhole`
4. 重构 `BlackholeModule`
5. 重写 runner 协议
