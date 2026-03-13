# TT Metalium 文档目录

本文档目录包含 TT Metalium (TT-Metal) 编程框架的完整参考手册。

## 主要文档

### 📘 TT_Metalium_完整指南_v2.0.md

**统一完整文档** (约14,000 行)

v2.0 版本整合了以下内容的最终文档：
- 原有完整参考手册的全部9个章节
- METALIUM_GUIDE.md 官方指南的独特内容（翻译为中文）
- TT-Metal Tech Reports 技术报告补充

**文档结构**：

```
TT_Metalium_完整指南_v2.0.md
│
├── 前言
│   ├── 执行摘要 (来自 METALIUM_GUIDE)
│   ├── 目标读者
│   └── 学习路径
│
├── 第一部分: 架构与概念
│   ├── 第1章: 架构概述
│   │   ├── 1.1-1.7: 原有架构内容
│   │   └── 1.8: 硬件架构专家视角 (新增)
│   │       ├── GPU 专家视角
│   │       ├── CPU 专家视角
│   │       ├── 缓存层次结构详解
│   │       └── SRAM/缓冲区详解
│   │
│   ├── 第2章: 核心概念
│   │   └── 2.5.6: 原生 Tile 计算详解 (新增)
│   │
│   └── 第3章: 编程示例
│       └── 2.1: 完整的向量加法示例 (新增)
│
├── 第二部分: API 参考
│   ├── 第4章: Host API 参考
│   ├── 第5章: Circular Buffer API
│   ├── 第6章: Data Movement API
│   └── 第7章: Compute Kernel API
│       └── 1.1: Compute API 跨代兼容性 (新增)
│
└── 第三部分: 进阶主题
    ├── 第8章: 性能优化指南
    │   └── 9: Fast Dispatch 详解 (新增)
    └── 第9章: 调试工具详解
```

## v2.0 新增内容

| 章节 | 新增内容 | 来源 |
|------|---------|------|
| **前言** | 执行摘要、目标读者指南 | METALIUM_GUIDE |
| **第1.8节** | 硬件架构专家视角 | METALIUM_GUIDE |
| **第2.5.6节** | 原生 Tile 计算详解 | METALIUM_GUIDE |
| **第3章** | 完整的向量加法示例 | METALIUM_GUIDE |
| **第7.1.1节** | Compute API 跨代兼容性 | METALIUM_GUIDE |
| **第8.9节** | Fast Dispatch vs Slow Dispatch | METALIUM_GUIDE |

## 快速开始

### 新用户推荐路径
1. 阅读 **前言 - 执行摘要** 了解整体架构
2. 学习 **第1-2章** 掌握架构和核心概念
3. 完成 **第3章 - 完整的向量加法示例**
4. 根据需求查阅 **第4-7章** API 参考

### 开发者推荐路径
1. 直接查阅 **第1.8节 - 硬件架构专家视角** 深入理解架构
2. 参考 **第8章 - 性能优化指南** 进行优化
3. 遇到问题查看 **第9章 - 调试工具详解**

## 版本信息

- **文档版本**: v2.0
- **最后更新**: 2026-03-13
- **适用 TT-Metalium 版本**: v0.55+
- **项目地址**: https://github.com/tenstorrent/tt-metal
- **官方文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/

### v2.0 更新说明

本次更新整合了 METALIUM_GUIDE.md 官方指南的独特内容，包括：
- Executive Summary 执行摘要
- GPU/CPU 专家视角的架构对比
- 缓存层次结构详解
- 完整的向量加法编程示例
- Compute API 跨代兼容性说明
- Fast Dispatch 与 Slow Dispatch 详解

---

*本文档是 TT Metalium API 深度调研与文档完善计划的成果 - v2.0 整合版*
