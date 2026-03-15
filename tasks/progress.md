# TileLang Blackhole 后端开发进度

## 工作流程说明

本文档记录开发任务的状态。每个任务应遵循以下流程：

1. **领取任务**：从"未开始"或"进行中"任务中选择
2. **任务设计**：在 `dev_design/` 下创建任务设计文档
3. **开发实现**：参考 `memory/general_dev.md` 和 `docs/`
4. **问题记录**：遇到问题更新 `memory/bugs.md`
5. **经验总结**：完成后更新 `memory/general_dev.md`
6. **标记完成**：在此文档中更新任务状态
7. **提交推送**：git commit & push

---

## 当前阶段

**阶段**: Phase 0 - 环境准备
**目标**: 准备 TileLang、TT-Metal、TT-Sim 编译环境
**开始日期**: 2026-03-15
**预计完成**: 2026-03-22

---

## 任务状态总览

| 阶段 | 任务 | 状态 | 负责人 | 设计文档 | 备注 |
|------|------|------|--------|----------|------|
| Phase 0 | TileLang 环境准备 | ✅ 已完成 | - | [phase0_tilelang_setup](./dev_design/phase0_tilelang_setup.md) | 基础功能验证通过 |
| Phase 0 | TT-Metal 编译 | ✅ 已完成 | - | [phase0_tt_metal_build](./dev_design/phase0_tt_metal_build.md) | libtt_metal.so 18MB |
| Phase 0 | TT-Sim 编译 | ✅ 已完成 | - | [phase0_tt_sim_build](./dev_design/phase0_tt_sim_build.md) | libttsim_bh.so 178KB |
| Phase 0 | TileLang+Blackhole 配置 | ⏳ 未开始 | - | - | CMake 配置 |
| Phase 1 | CodeGen 框架 | ⏳ 未开始 | - | - | 单核 Copy |
| Phase 1 | Runtime 框架 | ⏳ 未开始 | - | - | 延迟编译 |
| Phase 2 | SplitBlackholeKernel | ⏳ 未开始 | - | - | R/C/W 拆分 |
| Phase 2 | PlanBlackholeCB | ⏳ 未开始 | - | - | CB 分配 |
| Phase 2 | AssignBlackholeCores | ⏳ 未开始 | - | - | 140 核分配 |
| Phase 3 | GEMM 支持 | ⏳ 未开始 | - | - | matmul_tiles |
| Phase 4 | 性能优化 | ⏳ 未开始 | - | - | 自动 tile size |

**状态图例**：
- ⏳ 未开始
- 🔄 进行中
- ✅ 已完成
- 🐛 有阻塞问题

---

## 当前阻塞问题

| 问题 | 优先级 | 状态 | 相关任务 | 备注 |
|------|--------|------|----------|------|
| 无 | - | - | - | Phase 0 准备中 |

---

## 下一步行动

1. 完成 TileLang 基础环境编译
2. 验证基础测试通过
3. 准备 TT-Metal 编译

---

## 已完成任务归档

### 架构设计阶段 (2026-03-13 ~ 2026-03-15)

- ✅ Blackhole 硬件规格调研
- ✅ TT-Metal 工具链调研
- ✅ TileLang 后端模式调研
- ✅ 架构设计文档编写

---

## 参考文档

- [架构设计](./arch_design.md) - 总体架构设计
- [开发设计目录](./dev_design/) - 各任务详细设计
- [开发经验](../memory/general_dev.md) - 最佳实践
- [问题记录](../memory/bugs.md) - Bug 解决方案
