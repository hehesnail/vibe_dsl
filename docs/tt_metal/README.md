# TT Metalium 文档

本文档目录包含 Tenstorrent TT Metalium (TT-Metal) 编程框架的完整技术参考，涵盖使用指南和源码实现两大主题。

---

## 文档结构

```
tt_metal/
├── guide/              # 使用指南与API参考
│   └── TT_Metalium_完整指南_v2.0.md
└── source_analysis/    # 源码深度解析
    └── [13个模块文档]
```

---

## 📘 guide/ - 使用指南

**适合人群**: 应用开发者、算法工程师、初学用户

包含 TT Metalium 的完整使用指南，从架构概念到API参考，从编程示例到性能优化：

| 章节 | 内容 |
|------|------|
| **架构与概念** | 硬件架构概述、核心概念、编程模型 |
| **API参考** | Host API、Circular Buffer、Data Movement、Compute Kernel |
| **进阶主题** | 性能优化、调试工具、Fast/Slow Dispatch |

**快速开始**: 新用户建议按顺序阅读，从架构概述到编程示例，逐步掌握 TT Metalium 编程。

---

## 🔍 source_analysis/ - 源码深度解析

**适合人群**: 框架开发者、性能优化工程师、进阶用户

对 `tt_metal/` 源码目录的深度分析，涵盖13个核心模块的实现细节：

| 模块 | 描述 |
|------|------|
| **api/** | Host API 接口定义与实现 |
| **impl/** | 核心实现层（Allocator、Dispatch、Program等）|
| **llrt/** | 底层运行时（HAL、Firmware、Cluster管理）|
| **hw/** | 硬件抽象层（Firmware、CKernels、寄存器定义）|
| **distributed/** | 分布式执行与Mesh Workload |
| **fabric/** | 芯片间通信与Collective通信 |
| **jit_build/** | JIT编译系统与内核缓存 |
| **tools/** | 性能分析工具与调试工具 |

**使用建议**: 配合源码阅读，深入理解框架内部机制。

---

## 如何选择

| 你的需求 | 推荐阅读 |
|---------|----------|
| 学习如何使用 TT Metalium 编程 | `guide/` |
| 了解框架内部实现机制 | `source_analysis/` |
| 编写高性能计算内核 | 两者结合 |
| 调试性能问题 | `source_analysis/` 重点查看调度与内存分配 |
| 移植模型到 Tenstorrent 硬件 | `guide/` 的API参考 + `source_analysis/` 的示例 |

---

## 资源链接

- **项目地址**: https://github.com/tenstorrent/tt-metal
- **官方文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/

---

*文档最后更新: 2026-03-13*
