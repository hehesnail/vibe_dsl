# TT Metalium API 深度调研与文档完善计划

## 项目目标

对 Tenstorrent TT Metal 底层编程 API 进行全面调研、交叉验证和文档完善，形成一份权威、完整、准确的 API 参考文档。

---

## 阶段一：官方文档爬虫与信息收集

### 任务 1.1: 官方文档站点结构分析 ✅
- **目标**: 梳理官方文档的完整结构
- **行动**:
  - 访问 https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/
  - 记录所有文档页面 URL 和层级关系
  - 识别核心 API 文档、教程、示例分类
- **产出**: 官方文档站点结构图
- **完成状态**: 已完成，产出文件 `docs/tt_metal/official_docs_structure.md`
- **关键发现**:
  - 文档包含 3 大板块: Get Started, TT-Metalium, Resources
  - 识别出 160+ API (Host: ~15, Kernel: 140+)
  - 现有文档仅覆盖约 40% API，SFPU API 尤其缺失 (仅覆盖 20%)
  - 发现新类别: Lab Exercises, Advanced Topics, Tools

### 任务 1.2: API 参考手册抓取
- **目标**: 收集所有 API 参考文档
- **行动**:
  - 抓取 Host API 完整文档
  - 抓取 Device/Kernel API 完整文档
  - 抓取 Python API 文档
  - 抓取 Data Movement API 文档
  - 抓取 Compute Kernel API 文档
- **产出**: API 文档原始集合

### 任务 1.3: 编程示例收集
- **目标**: 收集所有官方示例代码
- **行动**:
  - 抓取编程示例文档
  - 记录 GitHub 示例代码路径
  - 分类整理示例类型（Matmul, Element-wise, Multi-chip等）
- **产出**: 示例代码清单

### 任务 1.4: 高级主题与优化指南收集
- **目标**: 收集进阶内容
- **行动**:
  - 抓取多芯片编程指南
  - 抓取性能优化最佳实践
  - 抓取调试与剖析工具文档
- **产出**: 高级主题文档集合

---

## 阶段二：GitHub 源码验证

### 任务 2.1: GitHub 仓库结构分析
- **目标**: 分析 tt-metal 仓库结构
- **行动**:
  - 浏览 https://github.com/tenstorrent/tt-metal
  - 识别关键目录：tt_metal/, docs/, tests/, programming_examples/
  - 记录头文件组织结构
- **产出**: 仓库结构图

### 任务 2.2: Header 文件 API 提取
- **目标**: 从源码提取完整 API 定义
- **行动**:
  - 提取 host_api.hpp 完整接口
  - 提取 dataflow_api.h 完整接口
  - 提取 compute_kernel_api/ 下所有头文件接口
  - 提取 tt_metal.h 底层接口
- **产出**: 源码级 API 清单

### 任务 2.3: 示例代码源码分析
- **目标**: 分析编程示例源码
- **行动**:
  - 分析 loopback 示例
  - 分析 matmul_single_core 示例
  - 分析 matmul_multi_core 示例
  - 分析 matmul_multichip 示例
  - 查找其他隐藏示例
- **产出**: 示例源码分析报告

---

## 阶段三：交叉验证与差异分析

### 任务 3.1: 现有文档 vs 官方文档对比
- **目标**: 识别现有文档的缺失和错误
- **行动**:
  - 逐章节对比现有文档与官方文档
  - 标记新增 API
  - 标记已弃用 API
  - 记录参数差异
- **产出**: 差异分析表

### 任务 3.2: 官方文档 vs 源码对比
- **目标**: 验证官方文档的准确性
- **行动**:
  - 对比文档 API 与头文件定义
  - 识别文档未覆盖的 API
  - 检查参数类型一致性
- **产出**: 源码验证报告

### 任务 3.3: 缺失内容识别
- **目标**: 找出需要补充的内容
- **行动**:
  - 列出未覆盖的 API 类别
  - 识别缺少的示例
  - 找出性能优化细节缺失
- **产出**: 缺失内容清单

---

## 阶段四：文档重构与完善

### 任务 4.1: 架构部分完善
- **目标**: 更新架构概述
- **行动**:
  - 验证硬件代际信息（Grayskull/Wormhole/Blackhole）
  - 补充 Blackhole 最新特性
  - 完善软件栈层次说明
- **产出**: 架构章节更新稿

### 任务 4.2: Core Concepts 完善
- **目标**: 完善核心概念章节
- **行动**:
  - 验证 Tensix 核心架构细节
  - 补充 Host-Device 内存模型细节
  - 完善 NoC 通信机制说明
- **产出**: 核心概念章节更新稿

### 任务 4.3: Host API 参考完善
- **目标**: 完整的 Host API 文档
- **行动**:
  - 设备管理 API（CreateDevice, CloseDevice等）
  - Buffer 管理 API（CreateBuffer, Buffer类型）
  - Kernel 创建 API（CreateKernel, 配置结构体）
  - 程序执行 API（EnqueueProgram, Finish等）
  - 运行时参数 API（SetRuntimeArgs）
  - Event/Semaphore API
- **产出**: Host API 参考章节

### 任务 4.4: Circular Buffer API 完善
- **目标**: 完整的 CB API 文档
- **行动**:
  - Host 端 CB 创建配置
  - Device 端 CB 操作（reserve/push/wait/pop）
  - CB 高级用法（双缓冲、多生产者等）
- **产出**: CB API 参考章节

### 任务 4.5: Data Movement Kernel API 完善
- **目标**: 完整的数据移动 API
- **行动**:
  - noc_async_read/write 系列
  - noc_async_read/write_barrier
  - noc_multicast 操作
  - DRAM/NOC 地址操作
  - 信号量操作
- **产出**: Data Movement API 参考章节

### 任务 4.6: Compute Kernel API 完善
- **目标**: 完整的计算 API
- **行动**:
  - 矩阵运算 API（matmul_tiles, matmul_block等）
  - 逐元素二元操作（add, sub, mul, div等）
  - 逐元素一元操作（relu, gelu, sigmoid, exp等）
  - 数据格式转换（tilize, untilize）
  - Tile 寄存器操作（acquire/commit/wait/release）
  - 归约操作 API
- **产出**: Compute API 参考章节

### 任务 4.7: Python API 补充
- **目标**: 补充 Python 绑定文档
- **行动**:
  - 调研 Python API 覆盖范围
  - 对比 C++ API 差异
  - 补充 Python 特有功能
- **产出**: Python API 参考章节

### 任务 4.8: 编程示例扩展
- **目标**: 丰富示例库
- **行动**:
  - 保留现有示例（DRAM Loopback, Matmul）
  - 添加 Element-wise 操作示例
  - 添加多芯片通信示例
  - 添加自定义 Kernel 示例
  - 每个示例包含：代码、解释、运行步骤
- **产出**: 编程示例章节

### 任务 4.9: 性能优化指南完善
- **目标**: 详细优化指南
- **行动**:
  - 双缓冲技术详解
  - CB 大小计算最佳实践
  - NoC 带宽优化技巧
  - Math Fidelity 选择指南
  - 多核负载均衡策略
  - 多芯片通信优化
- **产出**: 性能优化章节

### 任务 4.10: 调试工具详解
- **目标**: 完整调试工具文档
- **行动**:
  - Tracy Profiler 详细配置
  - Device Profiler 使用指南
  - Watcher 调试技巧
  - DPRINT 调试方法
  - 常见问题排查
- **产出**: 调试工具章节

---

## 阶段五：文档整合与发布

### 任务 5.1: 文档结构重构
- **目标**: 设计清晰的文档结构
- **行动**:
  - 设计目录层级
  - 组织章节顺序
  - 创建交叉引用
- **产出**: 新文档大纲

### 任务 5.2: 内容整合
- **目标**: 合并所有更新内容
- **行动**:
  - 整合各章节更新稿
  - 统一术语和格式
  - 添加版本信息
- **产出**: 完整文档草稿

### 任务 5.3: 格式规范与美化
- **目标**: 提升文档可读性
- **行动**:
  - 统一代码块格式
  - 优化表格展示
  - 添加目录和锚点
  - 检查链接有效性
- **产出**: 格式化文档

### 任务 5.4: 文档发布
- **目标**: 发布到指定目录
- **行动**:
  - 创建 docs/tt_metal/ 目录结构
  - 写入最终文档
  - 创建索引文件
- **产出**: 发布文档

---

## 任务依赖关系

```
阶段一 (信息收集)
    ├── 任务 1.1 ──┐
    ├── 任务 1.2 ──┼──→ 阶段三 (交叉验证)
    ├── 任务 1.3 ──┤         ↓
    └── 任务 1.4 ──┘    阶段四 (文档重构)
                              ↓
阶段二 (源码验证) ─────────→ 阶段五 (发布)
    ├── 任务 2.1
    ├── 任务 2.2
    └── 任务 2.3
```

---

## 执行策略

1. **并行执行**: 阶段一和阶段二可以并行进行
2. **迭代完善**: 阶段三、四、五可能需要多轮迭代
3. **增量交付**: 每完成一个章节即可进行小版本更新
4. **质量把关**: 每个阶段结束后进行 Review

---

## 成功标准

- [ ] 覆盖 100% 的核心 Host API
- [ ] 覆盖 100% 的 Data Movement API
- [ ] 覆盖 100% 的 Compute Kernel API
- [ ] 至少 10 个完整编程示例
- [ ] 所有 API 包含：函数签名、参数说明、返回值、使用示例
- [ ] 文档结构与官方保持一致但内容更详细
- [ ] 发现并修正现有文档中的错误

---

## 当前状态

- **已有资源**: TT_Metal_Documentation_Summary.md（约 930 行，基础较完整）
- **待调研源**:
  - https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/
  - https://github.com/tenstorrent/tt-metal
- **目标目录**: /root/dev/vibe_dsl/docs/tt_metal/

---

*创建时间: 2026-03-12*
*计划版本: v1.0*
