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

**阶段**: Phase 3 - GEMM 支持
**目标**: 实现 matmul_tiles 和完整 GEMM 算子
**开始日期**: 2026-03-16
**预计完成**: 2026-03-25

### 上一阶段总结 (Phase 2)

**状态**: ✅ 已完成
**完成日期**: 2026-03-16
**主要成果**:
- **SplitBlackholeKernel Pass**: 将 unified PrimFunc 拆分为 Reader/Compute/Writer 三个 kernel
- **PlanBlackholeCB Pass**: 规划 64 个 CB 的分配，验证 1.5MB L1 限制
- **AssignBlackholeCores Pass**: 将 T.Kernel grid 映射到 Blackhole 14x10 Tensix core grid
- 单元测试: 20 个测试用例覆盖所有三个 Pass
- 设计文档: 3 份详细设计文档

### Phase 1 总结

**状态**: ✅ 已完成
**完成日期**: 2026-03-16
**主要成果**:
- CodeGen 框架: 支持生成 TT-Metal 风格的 Reader/Writer Kernel
- Runtime 框架: Build 函数和 Device API 实现
- TT-Sim 端到端测试: ✅ 4096 个元素全部正确复制
- 详细报告: [PHASE1_TTSIM_TEST_REPORT.md](../tests/target/PHASE1_TTSIM_TEST_REPORT.md)

---

## 任务状态总览

| 阶段 | 任务 | 状态 | 负责人 | 设计文档 | 备注 |
|------|------|------|--------|----------|------|
| Phase 0 | TileLang 环境准备 | ✅ 已完成 | - | [phase0_tilelang_setup](./dev_design/phase0_tilelang_setup.md) | 基础功能验证通过 |
| Phase 0 | TT-Metal 编译 | ✅ 已完成 | - | [phase0_tt_metal_build](./dev_design/phase0_tt_metal_build.md) | libtt_metal.so 18MB |
| Phase 0 | TT-Sim 配置 | ✅ 已完成 | - | [phase0_tt_sim_build](./dev_design/phase0_tt_sim_build.md) | libttsim_bh.so 182KB, metal_example_add_2_integers_in_riscv 测试通过 |
| Phase 0 | TileLang+Blackhole 配置 | ✅ 已完成 | - | [phase0_tilelang_blackhole_config](./dev_design/phase0_tilelang_blackhole_config.md) | USE_BLACKHOLE, CodeGen 框架 |
| Phase 1 | CodeGen 框架 | ✅ 已完成 | - | [phase1_codegen_framework](./dev_design/phase1_codegen_framework.md) | 单核 Copy, Reader/Writer Kernel |
| Phase 1 | Runtime 框架 | ✅ 已完成 | - | [phase1_runtime_framework](./dev_design/phase1_runtime_framework.md) | Build函数, DeviceAPI, 测试 |
| Phase 1 | E2E Copy 测试 | ✅ 已完成 | - | [PHASE1_TTSIM_TEST_REPORT](../tests/target/PHASE1_TTSIM_TEST_REPORT.md) | 端到端测试, TT-Sim 验证通过, 4096 元素正确 |
| Phase 2 | SplitBlackholeKernel | ✅ 已完成 | - | [phase2_split_blackhole_kernel](./dev_design/phase2_split_blackhole_kernel.md) | R/C/W 拆分 |
| Phase 2 | PlanBlackholeCB | ✅ 已完成 | - | [phase2_plan_blackhole_cb](./dev_design/phase2_plan_blackhole_cb.md) | CB 分配 |
| Phase 2 | AssignBlackholeCores | ✅ 已完成 | - | [phase2_assign_blackhole_cores](./dev_design/phase2_assign_blackhole_cores.md) | 140 核分配 |
| Phase 3 | GEMM 支持 | ✅ 已完成 | - | [phase3_gemm](./dev_design/phase3_gemm.md) | matmul_tiles, Visitor CodeGen, E2E测试 |
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
| 无 | - | - | - | Phase 2 已完成，准备进入 Phase 3 |

---

## 当前阻塞问题

| 问题 | 优先级 | 状态 | 相关任务 | 备注 |
|------|--------|------|----------|------|
| 无 | - | - | - | Phase 3 已完成，准备进入 Phase 4 |

---

## 下一步行动

### Phase 4 计划 (性能优化)

1. **自动 Tile Size 选择**
   - 基于 Blackhole 硬件特性（1.5MB L1, 64 CBs）
   - 实现启发式算法选择最优 tile size

2. **多核并行优化**
   - 实现 140 Tensix cores 的负载均衡
   - 优化 NOC 路由以减少通信延迟

3. **TT-Sim 性能分析**
   - 测量单核 GEMM 峰值性能
   - 测量多核扩展效率
   - 与理论峰值对比分析

---

## 已完成任务归档

### Phase 3: GEMM 支持与 E2E 测试 (2026-03-16)

- ✅ **True End-to-End GEMM Test**
  - 测试文件: `tests/target/test_blackhole_gemm_true_e2e.py`
  - 启动脚本: `tests/target/run_blackhole_e2e.sh`
  - 完整流程: TileLang DSL → TIR → TT-Metal C++ 代码生成
  - 算法验证: 与 PyTorch 参考实现对比，误差在 FP16 精度范围内

- ✅ **Blackhole Target 注册**
  - 文件: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - 注册 `TVM_REGISTER_TARGET_KIND("blackhole", ...)`
  - 添加 `target.build.tilelang_blackhole` 构建函数

- ✅ **CodeGenBlackhole 完善**
  - `VisitStmt_(AttrStmtNode)` - 处理 thread_extent、storage_scope
  - `BindThreadIndex` - 映射 CUDA 线程索引到 Blackhole core 坐标
  - `PrintStorageScope` - 处理 shared.dyn/local/global 内存标记
  - `VisitExpr_(FloorDivNode/FloorModNode)` - 整数运算支持
  - 移除 `codegen_c_host.h` 中 `VisitStmt_` 的 `final` 关键字

- ✅ **TileLang Lower 集成**
  - 文件: `tilelang_repo/tilelang/engine/lower.py`
  - 添加 Blackhole 到 `device_codegen` 和 `device_codegen_without_compile`

### Phase 2: 多核拆分与调度 (2026-03-16)

1. **matmul_tiles 实现**
   - 调研 TT-Metal LLK 的 matmul_tiles API
   - 在 CodeGen 中生成矩阵乘法代码
   - 支持 FP16/BF16 输入，FP32 累加

2. **完整 GEMM 算子**
   - 实现 T.gemm() 的 Blackhole 后端
   - 支持分块矩阵乘法 (blocked GEMM)
   - 双缓冲优化

3. **TT-Sim GEMM 验证**
   - 小尺寸 GEMM 验证 (32x32x32)
   - 中等尺寸 GEMM 验证 (256x256x256)
   - 与参考实现对比结果正确性

4. **性能基准测试**
   - 测量单核 GEMM 性能
   - 测量多核并行 GEMM 性能
   - 与理论峰值对比

---

## 已完成任务归档

### Phase 2: 多核拆分与调度 (2026-03-16)

- ✅ **SplitBlackholeKernel Pass** 实现
  - 头文件: `tilelang_repo/src/transform/split_blackhole_kernel.h`
  - 实现: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
  - 单元测试: `tests/transform/test_split_blackhole_kernel.cc`
  - 功能: 将 unified PrimFunc 拆分为 Reader/Compute/Writer 三个 kernel

- ✅ **PlanBlackholeCB Pass** 实现
  - 头文件: `tilelang_repo/src/transform/plan_blackhole_cb.h`
  - 实现: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
  - 单元测试: `tests/transform/test_plan_blackhole_cb.cc`
  - 功能: 规划 64 个 CB 的分配，确保不超过 1.5MB L1 限制

- ✅ **AssignBlackholeCores Pass** 实现
  - 头文件: `tilelang_repo/src/transform/assign_blackhole_cores.h`
  - 实现: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
  - 单元测试: `tests/transform/test_assign_blackhole_cores.cc`
  - 功能: 将 T.Kernel grid 映射到 Blackhole 14x10 Tensix core grid

### Phase 1: CodeGen 与 Runtime 框架 (2026-03-16)

- ✅ CodeGenBlackhole 实现
  - Reader/Writer Kernel 生成
  - TT-Sim 兼容代码生成 (InterleavedAddrGen)
- ✅ Runtime 框架实现
  - BuildTileLangBlackhole 函数
  - BlackholeDeviceAPI
- ✅ TT-Sim 端到端测试
  - 单核 Copy kernel 验证
  - 4096 个 FP16 元素正确复制
  - 详细报告: [PHASE1_TTSIM_TEST_REPORT](../tests/target/PHASE1_TTSIM_TEST_REPORT.md)

### Phase 0: 环境准备 (2026-03-14 ~ 2026-03-15)

- ✅ TileLang 环境准备
- ✅ TT-Metal 编译 (libtt_metal.so 18MB)
- ✅ TT-Sim 配置 (libttsim_bh.so 182KB)
- ✅ 官方示例验证 (add_2_integers_in_riscv)

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
