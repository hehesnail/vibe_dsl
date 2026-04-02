# Stage 1 Single-Core Copy 闭环设计

> 历史文档说明：
> 当前状态与下一步请看 `tasks/progress.md`；
> 当前总体设计请看 `tasks/dev_design/final_blackhole_backend_redesign.md`。

## 文档定位

- **状态**: 已完成阶段的历史设计
- **当前依据**: `final_blackhole_backend_redesign.md`

本文件只保留 Stage 1 的阶段结论，不再承载当前活动阶段的设计。

说明：

- 文中出现的 `runner` / `spec.json -> runner` 仅用于记录 Stage 1 bring-up 历史
- 当前正式执行路径仍以 `ExecutableSpec -> BlackholeModule` 进程内 direct host path 为准

## Stage 1 目标

Stage 1 的目标是验证当时最小 single-core copy 真执行路径：

- `ExecutableSpec -> BlackholeModule -> runner`
- TT-Sim 下真实执行
- Python direct-call 主调用面收口

## 已完成结论

- `spec.json -> runner` 的 single-core copy 已在 TT-Sim 上通过
- `artifact.codegen_mod["main"](...)` 的 direct-call 路径已在 TT-Sim 上通过
- 最小 copy runtime args schema 已落地
- `BlackholeModule` packed tensor 参数提取问题已修正

## 当前仍然有效的阶段边界

- Stage 1 只证明了执行主路径可行
- Stage 1 不等价于“编译 pass 已打通”
- copy 的 runtime emitter 只能视为 Stage 1 的过渡实现，不能继续作为正式主路径扩张

## 对当前设计的影响

Stage 1 完成后，后续重心必须切到：

- pass 主链接入收正
- copy / GEMM 的语义前移
- 由 pass 主导的 single-core true E2E

详见 `stage2_pass_reuse_matrix.md` 与 `stage2_single_core_pass_integration.md`。
