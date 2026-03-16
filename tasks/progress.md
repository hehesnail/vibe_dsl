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

**阶段**: Phase 2 返工 + Phase 3 补全
**目标**: 让 Pipeline 真正连通，实现 Copy kernel E2E
**更新日期**: 2026-03-17

### 2026-03-17 设计审查结论

经全面审查所有文档和代码，发现 Phase 2 的 Transform Pass 全部是 Stub，
lower.py 未接入任何 Blackhole Pass，CodeGen 存在硬编码路径。
详见 [design_review.md](./design_review.md)。

**关键决策变更**：
1. Pass 顺序调整为：`LowerBlackholeOps → PlanBlackholeCB → AssignBlackholeCores`
2. SplitBlackholeKernel 降级为可选（P2 优化项），短期合并 R/C/W 到 BRISC 单核
3. CodeGen 统一走 IR Visitor 路径，删除硬编码 Copy 模板
4. Runtime 分阶段：外部进程（当前）→ C 桥接（中期）→ 直连（长期）

> 📋 **架构详情**: 见 [arch_design.md](./arch_design.md)
> 📋 **设计审查**: 见 [design_review.md](./design_review.md)

---

## 任务状态总览

### 状态图例
- ⏳ 未开始
- 🔄 进行中
- ✅ 已完成
- ⚠️ 框架完成（Stub，需实现）
- ⏸️ 搁置
- 🐛 有阻塞问题

| 阶段 | 任务 | 状态 | 设计文档 | 备注 |
|------|------|------|----------|------|
| Phase 0 | TileLang 环境准备 | ✅ 已完成 | [phase0_tilelang_setup](./dev_design/phase0_tilelang_setup.md) | |
| Phase 0 | TT-Metal 编译 | ✅ 已完成 | [phase0_tt_metal_build](./dev_design/phase0_tt_metal_build.md) | libtt_metal.so 18MB |
| Phase 0 | TileLang+Blackhole 配置 | ✅ 已完成 | [phase0_tilelang_blackhole_config](./dev_design/phase0_tilelang_blackhole_config.md) | |
| Phase 0 | TT-Sim 配置 | ✅ 已完成 | [phase0_tt_sim_build](./dev_design/phase0_tt_sim_build.md) | |
| Phase 1 | CodeGen 框架 | ✅ 已完成 | [phase1_codegen_framework](./dev_design/phase1_codegen_framework.md) | kernel_main 格式 |
| Phase 1 | Runtime 框架 | ✅ 已完成 | [phase1_runtime_framework](./dev_design/phase1_runtime_framework.md) | 外部进程模式 |
| Phase 1 | E2E Copy 手动测试 | ✅ 已完成 | - | 手动编写 Kernel 在 TT-Sim 通过 |
| Phase 2 | LowerBlackholeOps | 🔄 **40%** | [phase3_gemm](./dev_design/phase3_gemm.md) | matmul✅ copy❌ clear❌ |
| Phase 2 | PlanBlackholeCB | ⚠️ **Stub** | [phase2_plan_blackhole_cb](./dev_design/phase2_plan_blackhole_cb.md) | 返回原函数，需实现 |
| Phase 2 | AssignBlackholeCores | 🔄 **60%** | [phase2_assign_blackhole_cores](./dev_design/phase2_assign_blackhole_cores.md) | 逻辑正确，结果未存储到 attrs |
| Phase 2 | SplitBlackholeKernel | ⏸️ **搁置** | [phase2_split_blackhole_kernel](./dev_design/phase2_split_blackhole_kernel.md) | TT-Sim 不支持 NCRISC，降为 P2 优化项 |
| Phase 2 | lower.py 接入 Pass | ⏳ **未开始** | - | Blackhole 分支无 Pass 调用 |
| Phase 2 | CodeGen 统一入口 | ⏳ **未开始** | - | 需删除硬编码，统一走 Visitor |
| Phase 3 | GEMM CodeGen | ✅ 已完成 | [phase3_gemm](./dev_design/phase3_gemm.md) | Builtin visitor 已实现 |
| Phase 3 | GEMM E2E 验证 | ⏳ **未开始** | - | 需 Pipeline 连通后 |
| Phase 4 | 性能优化 | ⏳ 未开始 | - | 三核拆分、自动 tile size |

---

## 下一步行动计划

### P0：让 Pipeline 连通（Copy E2E）

1. **修改 lower.py 接入 Blackhole Pass**
   - 在 `device_codegen()` 的 blackhole 分支中调用 LowerBlackholeOps/PlanCB/AssignCores
2. **实现 LowerBlackholeOps copy 序列**
   - `GenerateCopySequence()` 生成 cb_reserve_back + noc_async_read + cb_push_back
3. **实现 PlanBlackholeCB MVP**
   - 收集 alloc_shared → 分配 CB ID → 验证约束 → 存入 attrs
4. **修复 AssignBlackholeCores**
   - 将计算结果存入 func attrs（当前只计算不存储）
5. **CodeGen 删除硬编码路径**
   - 删除 DetectSimpleCopyKernel / GenerateCopyKernelMain
   - 统一走 GenerateGenericKernelMain + HandleBlackholeBuiltin
6. **Copy kernel E2E 验证**
   - DSL → LowerOps → PlanCB → AssignCores → CodeGen → TT-Sim

### P1：GEMM E2E

7. **集成 matmul + copy 的完整 Pipeline**
8. **Runtime 传递 CB 配置**
9. **GEMM E2E 验证**

---

## 已完成工作归档

### Phase 0: 环境准备 ✅

- TileLang 编译安装（v0.1.8+cu128）
- TT-Metal 编译（libtt_metal.so 18MB）
- TT-Sim 配置（libttsim_bh.so，官方示例通过）
- TileLang + Blackhole CMake 配置

### Phase 1: 基础框架 ✅

- CodeGenBlackhole：kernel_main 入口 + get_arg_val 参数获取 + TT-Metal 头文件
- BlackholeModule：外部进程执行模式 (fork/exec runner)
- tilelang_blackhole_runner：编译成功 (705KB)
- TT-Sim 手动 Copy 测试通过（手动编写 Kernel）

### Phase 3 部分: GEMM CodeGen ✅

- 15 个 TT-Metal builtin 定义和注册
- LowerBlackholeOps matmul 序列生成
- CodeGen Visitor 处理所有 builtin
- C++ 单元测试 3/3 通过

---

## 参考文档

- [架构设计](./arch_design.md) - 总体架构设计
- [设计审查](./design_review.md) - 全面设计审查与改进方案
- [开发设计目录](./dev_design/) - 各任务详细设计
- [开发经验](../memory/general_dev.md) - 最佳实践
- [问题记录](../memory/bugs.md) - Bug 解决方案
