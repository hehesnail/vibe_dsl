# TileLang Tenstorrent 后端扩展项目

## 项目目标

为 TileLang DSL 添加 Tenstorrent 硬件后端支持，基于 TT-Metal 底层框架实现编译器后端。

## 核心架构

```
TileLang DSL (Python)
       ↓
TIR (TVM IR)
       ↓
TT-Metal CodeGen → TT-Metal Runtime
       ↓
Tenstorrent Hardware (Grayskull/Wormhole)
```

## 开发原则

### 1. 渐进式实现
- 先跑通端到端流程，再优化性能
- 从简单算子（copy/gemm）开始，逐步支持复杂模式
- 保持与现有后端的兼容性

### 2. 复用现有基础设施
- 基于 TVM 的代码生成框架
- 复用 TileLang 的内存管理和调度逻辑
- 对接 TT-Metal 的 Kernel 编译和加载机制

---

## 知识管理体系

### 文件结构

```
memory/
├── bugs.md          # 问题与Bug解决方案
└── general_dev.md   # 通用开发模式与最佳实践

tasks/
└── progress.md      # 开发进度追踪
```

### Agent 工作模式

#### 1. 领取任务
1. **查看进度**: 阅读 `tasks/progress.md`，找到未完成的任务
2. **选择任务**: 按阶段顺序选择最靠前的未完成任务
3. **确认理解**: 确保理解任务目标和预期产出

#### 2. 开始任务前（写代码前）
1. **查阅模式**: 阅读 `memory/general_dev.md`，了解已有的开发模式和最佳实践
2. **查阅历史**: 查看 `memory/bugs.md`，避免重复踩坑
3. **方案规划**: 基于已有模式，规划实现方案（至少考虑 2 种备选）

#### 3. 遇到问题/bug时
1. **查阅历史**: 先查看 `memory/bugs.md`，检查是否遇到过类似问题
2. **尝试解决**: 如未找到，尝试 2-3 种不同方案自主解决
3. **记录问题**: 解决后，将问题描述、根本原因、解决方案写入 `memory/bugs.md`

#### 4. 完成任务后
1. **标记完成**: 在 `tasks/progress.md` 中用删除线或 ✅ 标记任务完成
2. **反思总结**: 回顾任务过程中的关键决策和学到的经验
3. **模式提炼**: 将可复用的模式、最佳实践写入 `memory/general_dev.md`

#### 5. 多方案探索
- 对关键设计决策，至少考虑 2 种实现方案
- 在代码注释或提交信息中记录方案对比
- 将最终决策理由记录到相应文档

---

## 开发流程

### 1. 理解阶段
- 阅读相关文档（`docs/tilelang/`, `docs/tt_metal/`）
- 分析现有后端实现（`codegen_cuda`, `codegen_hip`）
- 确定 TT-Metal 的对应机制

### 2. 实现阶段
- 按模块分解任务
- 每完成一个模块，更新 `tasks/progress.md`
- 遇到问题及时查阅和更新 `memory/bugs.md`

### 3. 验证阶段
- 编写单元测试
- 对比参考实现的输出
- 在真实硬件上验证（如可用）

### 4. 总结阶段
- 更新 `memory/general_dev.md`
- 记录可复用的模式
- 标记遗留问题和下一步计划

---

## 关键文件位置

```
tilelang/src/target/
├── codegen_cuda.h/cc      # CUDA 后端参考
tilelang/src/target/
├── codegen_hip.h/cc       # HIP 后端参考
docs/tilelang/cpp_core/
├── 04_target_backends.md  # 后端架构文档
docs/tt_metal/
├── source_analysis/       # TT-Metal 源码分析
```

## 代码规范

- 遵循 TVM/TileLang 的命名和代码风格
- 关键逻辑添加注释，说明与 TT-Metal 的对应关系
- 保持代码简洁，避免过度工程化
