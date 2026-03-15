# TileLang Tenstorrent 后端扩展项目

## 项目目标

为 TileLang DSL 添加 Tenstorrent 硬件后端支持，基于 TT-Metal 底层框架实现编译器后端。

## 核心架构（Blackhole）

```
TileLang DSL (Python)
       ↓ LowerAndLegalize
TIR (TVM IR)
       ↓ Blackhole Passes
├─ AssignBlackholeCores (14x10 grid)
├─ PlanBlackholeCB (64 CBs, 1.5MB L1)
└─ SplitBlackholeKernel (R/C/W 拆分)
       ↓ CodeGen
TT-Metal C++ (BRISC/TRISC/NCRISC)
       ↓ JIT Build (libtt_metal.so)
RISC-V ELF
       ↓ Runtime
Blackhole Hardware (140 cores)
```

## 开发原则

1. **Blackhole 优先**：专注 14x10 grid, 64 CBs, 1.5MB L1 架构
2. **DSL 兼容**：同一套 TileLang 代码零修改编译到 Blackhole
3. **动态链接**：通过 libtt_metal.so 调用 TT-Metal API，非 Python 包
4. **TT-Sim 验证**：无硬件条件下使用仿真器验证功能

---

## 知识管理体系

### 文件结构

```
memory/                          # 经验与问题总结
├── bugs.md                      # 问题与Bug解决方案
└── general_dev.md               # 通用开发模式与最佳实践

tasks/                           # 任务管理
├── progress.md                  # 开发进度追踪（任务状态看板）
├── arch_design.md               # 总体架构设计（只含架构，不含进度）
└── dev_design/                  # 各任务详细设计文档
    ├── README.md
    ├── phase0_tilelang_setup.md
    ├── phase1_codegen_framework.md
    └── ...
```

### Agent 工作模式

#### 1. 领取任务（自动触发）
当用户说"开始工作"或"领取任务"时：
1. **阅读 CLAUDE.md**: 加载本文件（已自动完成）
2. **查看进度**: 阅读 `tasks/progress.md`，找到"进行中"或"未开始"的任务
3. **学习经验**: 阅读 `memory/general_dev.md` 和 `memory/bugs.md`
4. **选择任务**: 按阶段顺序选择最靠前的未完成任务
5. **确认理解**: 确保理解任务目标和预期产出

#### 2. 开始任务前（写代码前）
1. **查阅架构**: 阅读 `tasks/arch_design.md`，了解总体架构约束
2. **查阅设计**: 检查 `tasks/dev_design/` 是否已有该任务的设计文档
   - 如无：创建新的设计文档（使用 dev_design/README.md 模板）
   - 如有：阅读并理解已有设计
3. **方案规划**: 基于已有模式，规划实现方案（至少考虑 2 种备选）
4. **更新进度**: 在 `tasks/progress.md` 中将任务标记为"进行中"

#### 3. 开发过程中
1. **参考文档**: 使用 `docs/` 下的文档作为技术参考
2. **查阅代码**: 阅读工程源码（tilelang_repo, tt_metal_repo）寻找实现思路
3. **遇到问题/bug时**:
   - 先查看 `memory/bugs.md`，检查是否遇到过类似问题
   - 尝试 2-3 种不同方案自主解决
   - 解决后，将问题描述、根本原因、解决方案写入 `memory/bugs.md`
4. **更新设计**: 如实现与原始设计有偏差，更新 `tasks/dev_design/` 中的设计文档

#### 4. 完成任务后
1. **标记完成**: 在 `tasks/progress.md` 中将任务标记为"已完成"
2. **经验总结**: 将可复用的模式、最佳实践写入 `memory/general_dev.md`
3. **更新设计**: 确保 `tasks/dev_design/` 中的设计文档与最终实现一致
4. **提交推送**: git commit & push 所有变更（包括文档更新）

#### 5. 多方案探索
- 对关键设计决策，至少考虑 2 种实现方案
- 在 `tasks/dev_design/` 的设计文档中记录方案对比
- 将最终决策理由记录到相应文档

---

## 开发流程

### 完整工作流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           开始新会话                                     │
│                    （CLAUDE.md 自动加载）                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 领取任务                                                        │
│ ├── 读取 tasks/progress.md → 找到未完成任务                              │
│ ├── 读取 memory/general_dev.md → 学习经验                               │
│ ├── 读取 memory/bugs.md → 避免踩坑                                      │
│ └── 选择最靠前的未完成任务                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 任务设计（写代码前）                                             │
│ ├── 读取 tasks/arch_design.md → 了解架构约束                             │
│ ├── 检查 tasks/dev_design/ → 是否有该任务设计文档                         │
│ │   ├── 如无：创建新设计文档（使用模板）                                  │
│ │   └── 如有：阅读并理解设计                                             │
│ ├── 方案规划（至少 2 种备选）                                            │
│ └── 更新 tasks/progress.md → 标记"进行中"                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: 开发实现                                                        │
│ ├── 参考 docs/ 技术文档                                                  │
│ ├── 查阅 tilelang_repo / tt_metal_repo 源码                              │
│ ├── 遇到问题：                                                          │
│ │   ├── 查阅 memory/bugs.md                                              │
│ │   ├── 尝试 2-3 种方案解决                                              │
│ │   └── 解决后更新 bugs.md                                               │
│ └── 如有偏差：更新 tasks/dev_design/ 中的设计文档                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: 验证测试                                                        │
│ ├── 编写并运行单元测试                                                   │
│ ├── 集成测试（TT-Sim 验证）                                              │
│ └── 对比参考实现输出                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 任务完成                                                        │
│ ├── 更新 tasks/progress.md → 标记"已完成"                               │
│ ├── 更新 memory/general_dev.md → 记录经验                               │
│ ├── 确保 tasks/dev_design/ 与实际实现一致                                 │
│ └── git commit & push（代码+文档）                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 各阶段详细说明

#### 理解阶段
- 阅读 `tasks/arch_design.md` 了解总体架构
- 阅读 `docs/tilelang/`, `docs/tt_metal/` 技术细节
- 分析现有后端实现（`codegen_cuda`, `codegen_hip`）

#### 设计阶段
- 在 `tasks/dev_design/` 创建任务设计文档
- 记录至少 2 种备选方案及选择理由
- 明确输入输出接口
- 制定测试计划

#### 实现阶段
- 按设计文档实现
- 保持与 `tasks/arch_design.md` 架构一致
- 遇到问题先查 `memory/bugs.md`

#### 验证阶段
- Pass 单元测试（gtest）
- CodeGen 测试（生成代码语法检查）
- Runtime 测试（模块功能）
- 集成测试（TT-Sim 端到端）

#### 总结阶段
- 更新 `tasks/progress.md` 任务状态
- 更新 `memory/general_dev.md` 记录经验
- 更新 `tasks/dev_design/` 确保一致性
- git commit & push

---

## 关键文件位置

### 代码实现（参考/新增）
```
tilelang_repo/src/target/
├── codegen_cuda.h/cc              # CUDA 后端参考模式
├── codegen_blackhole.h/cc         # [新增] Blackhole CodeGen
└── rt_mod_blackhole.cc            # [新增] Blackhole Runtime

tilelang_repo/src/transform/
├── split_blackhole_kernel.cc      # [新增] R/C/W 拆分 Pass
├── plan_blackhole_cb.cc           # [新增] CB 分配 Pass
└── assign_blackhole_cores.cc      # [新增] Core 分配 Pass
```

### 技术文档
```
docs/tilelang/cpp_core/
└── 04_target_backends.md          # TileLang 后端架构
docs/tt_metal/source_analysis/
├── api.md                         # TT-Metal Host API
├── jit_build.md                   # JIT 编译系统
└── hw.md                          # Blackhole 硬件规格
```

### 项目管理
```
tasks/
├── arch_design.md                 # 总体架构设计
├── progress.md                    # 任务状态看板
└── dev_design/                    # 各任务详细设计
memory/
├── general_dev.md                 # 开发经验总结
└── bugs.md                        # 问题与解决方案
```

## 代码规范

- **命名**：遵循 TVM 规范，`CamelCase` 类名，`snake_case` 函数/变量
- **注释**：关键逻辑说明与 TT-Metal API 的对应关系
- **文件**：每个 Pass/CodeGen 独立文件，头文件暴露接口
- **测试**：每段代码配套单元测试（gtest/pytest）
- **提交**：代码+文档同时提交，commit 说明任务阶段
