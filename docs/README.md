# 源代码分析文档

本目录包含各种 AI 编译器和高性能计算框架的源代码分析文档。

说明：

- 本目录主要承载源码分析与背景资料，不维护本仓库当前开发状态
- 当前 Blackhole 任务状态、设计主线与下一步请查看：
  - `tasks/progress.md`
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
- `docs/ttsim_setup.md` 现在只保留 TT-Sim 环境入口规则，不再维护具体 case 的测试命令模板

## 📚 已完成的分析

### [TileLang](./tilelang/) - 高性能 GPU 内核 DSL

TileLang 是一个用于开发高性能 GPU/CPU 内核的领域特定语言，基于 TVM 编译器基础设施构建。

- **文档数量**: 36 篇
- **覆盖范围**: Python DSL、JIT 编译系统、C++ 核心实现、高级功能、示例代码
- **关键特性**: CUDA/ROCm/Metal 多后端支持、Tensor Core 优化、自动调优

[查看详情 →](./tilelang/)

---

### [TT-Metal](./tt_metal/) - Tenstorrent 硬件编程框架

TT-Metal 是 Tenstorrent 公司的低级别硬件编程框架，用于 AI/ML 工作负载。

- **文档数量**: 多篇
- **覆盖范围**: 架构概述、源码分析、技术报告
- **关键特性**: 异构计算、Grayskull/Wormhole 架构支持

[查看详情 →](./tt_metal/)

---

## 📝 文档规范

- 代码引用格式: `file_path:line_number`
- 使用 Markdown 格式编写
- 包含架构图和流程图（Mermaid 语法）
