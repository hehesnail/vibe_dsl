# 任务开发设计文档

本目录存放每个开发任务的详细设计文档。

## 命名规范

```
dev_design/
├── phase0_tilelang_setup.md          # Phase 0: TileLang 环境
├── phase0_tt_metal_build.md          # Phase 0: TT-Metal 编译
├── phase0_tt_sim_build.md            # Phase 0: TT-Sim 编译
├── phase1_codegen_framework.md       # Phase 1: CodeGen 框架
├── phase1_runtime_framework.md       # Phase 1: Runtime 框架
├── phase2_split_kernel.md            # Phase 2: R/C/W 拆分
├── phase2_plan_cb.md                 # Phase 2: CB 规划
├── phase2_assign_cores.md            # Phase 2: Core 分配
├── phase3_gemm.md                    # Phase 3: GEMM 支持
└── phase4_optimization.md            # Phase 4: 性能优化
```

## 文档模板

每个任务设计文档应包含：

```markdown
# 任务名称

## 基本信息

- **任务ID**: phaseX_task_name
- **所属阶段**: Phase X
- **前置任务**: task1, task2
- **负责人**: -
- **状态**: 设计中 / 开发中 / 已完成

## 目标

本任务要实现什么功能。

## 设计概要

### 输入

- 输入数据/接口描述

### 输出

- 输出数据/接口描述

### 核心逻辑

关键算法或处理流程。

## 实现方案

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A | ... | ... | ✅ |
| B | ... | ... | - |

### 详细设计

关键代码结构、接口定义。

## 测试计划

- [ ] 单元测试
- [ ] 集成测试
- [ ] 端到端测试

## 开发记录

### YYYY-MM-DD

- 今日完成：...
- 遇到问题：...
- 解决方案：...

## 经验总结

（任务完成后填写）

- 关键决策及原因
- 可复用的模式
- 踩过的坑
```

## 当前活跃任务

暂无 - 等待 Phase 0 开始
