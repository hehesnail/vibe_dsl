# Stage 0 ExecutableSpec 与 Attr 对齐设计

> 历史文档说明：
> 当前状态与下一步请看 `tasks/progress.md`；
> 当前总体设计请看 `tasks/dev_design/final_blackhole_backend_redesign.md`。

## 文档定位

- **状态**: 已完成阶段的历史设计
- **当前依据**: `final_blackhole_backend_redesign.md`

本文件只保留 Stage 0 的阶段结论，不再承载当前总体架构。

## Stage 0 目标

Stage 0 的目标是把 Blackhole 后端从散乱的 attr / 运行时结构，收口到统一协议：

- 引入 `ExecutableSpec` 骨架
- attrs 统一到 `blackhole.*`
- `BlackholeModule` / runner 切到 `spec.json` 协议

## 已完成结论

- `ExecutableSpec / KernelSpec / CorePlan / KernelArgSpec` 已落地
- `rt_mod_blackhole` 已切到 `blackhole.cb_configs / core_plan / runtime_args / segment_plan` 主路径
- 历史 bring-up 期曾使用 `spec.json + input.bin + output.bin` runner 协议；当前正式执行路径已收敛到 `BlackholeModule` direct host path

## 对当前仍然有效的约束

- 主协议继续围绕 `ExecutableSpec`
- attrs 主线继续使用 `blackhole.*`
- runner 不再回退到旧固定命令行协议

## 对当前已不再需要重复展开的内容

- Stage 0 的具体脚本、构建步骤和最小 JSON 字段细节
- 这些内容已经沉淀到代码、脚本和 `memory/` 中，不再作为当前阶段设计重点
