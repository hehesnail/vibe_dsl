# TT Metalium 文档目录

本文档目录包含 TT Metalium (TT-Metal) 编程框架的完整参考手册及相关资源。

## 主要文档

### 📘 完整参考手册
**TT_Metalium_Complete_Reference.md** (13,529 行)
- 最全面的 API 参考文档
- 包含 9 个主要章节和 4 个附录
- 适合系统学习和查阅

### 📄 分章节文档

| 章节 | 文件 | 行数 | 内容 |
|------|------|------|------|
| 第1章 | section_architecture.md | 459 | 架构概述 |
| 第2章 | section_core_concepts.md | 684 | 核心概念 |
| 第3章 | section_programming_examples.md | 2,347 | 编程示例 |
| 第4章 | section_host_api.md | 1,719 | Host API 参考 |
| 第5章 | section_circular_buffer_api.md | 1,362 | Circular Buffer API |
| 第6章 | section_data_movement_api.md | 1,090 | Data Movement API |
| 第7章 | section_compute_api.md | 2,088 | Compute Kernel API |
| 第8章 | section_performance_optimization.md | 1,754 | 性能优化指南 |
| 第9章 | section_debugging_tools.md | 1,231 | 调试工具详解 |

## 附录

**appendices.md** (728 行)
- 附录 A: API 快速索引
- 附录 B: 常见任务速查表
- 附录 C: 术语表
- 附录 D: 错误代码参考

## 分析与中间文档

### 调研阶段产出
- `official_docs_structure.md` - 官方文档站点结构分析
- `api_reference_scraped.md` - API 参考手册抓取（1,420 行）
- `programming_examples_collection.md` - 编程示例收集（687 行）
- `advanced_topics_collection.md` - 高级主题与优化指南（739 行）

### 源码验证产出
- `github_repo_structure.md` - GitHub 仓库结构分析
- `header_api_extraction.md` - Header 文件 API 提取（1,194 行）
- `examples_source_analysis.md` - 示例代码源码分析

### 差异分析产出
- `doc_comparison_analysis.md` - 现有 vs 官方文档对比
- `source_verification_report.md` - 官方文档 vs 源码验证
- `gap_analysis.md` - 缺失内容识别（708 行）

### 文档规划
- `document_outline.md` - 新文档大纲
- `tasks.md` - 完整任务计划

## 快速开始

### 新用户推荐路径
1. 阅读 **TT_Metalium_Complete_Reference.md** 第1-2章（架构和核心概念）
2. 完成第3章的 Hello World 示例
3. 根据需求查阅第4-7章 API 参考

### 开发者推荐路径
1. 直接查阅 **TT_Metalium_Complete_Reference.md** 附录 A（API 索引）
2. 参考附录 B（任务速查表）快速完成任务
3. 遇到问题查看附录 D（错误代码参考）

## 文档统计

| 类别 | 文件数 | 总行数 |
|------|--------|--------|
| 主要章节 | 9 | 12,734 |
| 完整手册 | 1 | 13,529 |
| 附录 | 1 | 728 |
| 调研文档 | 4 | 3,846 |
| 分析文档 | 6 | ~3,000 |
| **总计** | **20+** | **~23,000** |

## 版本信息

- **文档版本**: v1.0
- **最后更新**: 2026-03-12
- **适用 TT-Metalium 版本**: v0.55+
- **项目地址**: https://github.com/tenstorrent/tt-metal
- **官方文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/

---

*本文档是 TT Metalium API 深度调研与文档完善计划的成果*
