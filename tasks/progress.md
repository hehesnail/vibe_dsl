# TileLang Blackhole 后端开发进度

> 当前唯一设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态和任务流转，不再承载任何总体架构设计。

## 当前阶段

- **阶段**: Stage 0 协议重构
- **日期**: 2026-03-18
- **目标**: 建立 `ExecutableSpec`、统一 attrs、替换旧 runner 协议

## 当前判断

- pass 已接入 lowering pipeline。
- runtime、codegen、runner 仍未形成真实执行闭环。
- 当前不再把“能生成 kernel 字符串”视为阶段完成。

## 任务状态总览

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| Stage 0 | 统一 attrs 到 `blackhole.*` | ⏳ 未开始 | 替换旧协议 |
| Stage 0 | 引入 `ExecutableSpec` | ⏳ 未开始 | 新核心接口 |
| Stage 0 | 重构 `rt_mod_blackhole` | ⏳ 未开始 | 改为 spec extractor |
| Stage 0 | 重构 `BlackholeModule` | ⏳ 未开始 | 改为 spec serializer |
| Stage 0 | 重写 runner 协议 | ⏳ 未开始 | 改为 `spec.json + input.bin + output.bin` |
| Stage 1 | single-core copy 闭环 | ⏳ 未开始 | true E2E |
| Stage 2 | single-core gemm 闭环 | ⏳ 未开始 | true E2E |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | per-core args |

## 当前下一步

1. 在运行时与 codegen 层引入 `ExecutableSpec`。
2. 清理并统一 attr schema。
3. 让 runner 转为 spec-driven executor。
4. 之后再做 copy 和 gemm 的真实执行闭环。
