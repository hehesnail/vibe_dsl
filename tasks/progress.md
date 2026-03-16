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

**阶段**: Phase 3 - GEMM 支持（代码生成重构中）
**目标**: 实现从 TileLang DSL 到 TT-Sim 执行的完整端到端流程
**开始日期**: 2026-03-16
**当前状态**: 🐛 **存在阻塞问题 - CodeGen 需要重构**

### 关键发现

**问题核心**: 当前 `CodeGenBlackhole` 继承自 `CodeGenCHost`，生成的是**标准C代码**，
不是真正的 **TT-Metal kernel 格式**。

**差异对比**:

| 方面 | 当前生成 | 真正TT-Metal Kernel |
|------|---------|---------------------|
| 函数入口 | `void func(args)` | `void kernel_main()` |
| 参数获取 | 函数参数 | `get_arg_val<uint32_t>(idx)` |
| 全局内存 | 指针访问 | `InterleavedAddrGen` + NOC |
| 共享内存 | 数组分配 | `cb_reserve_back/push_back` |
| 同步 | 无 | `noc_async_read_barrier` |

**影响**: 生成的代码无法在 TT-Sim 上执行，需要完整重构 CodeGen。

---

## 任务状态总览

| 阶段 | 任务 | 状态 | 负责人 | 设计文档 | 备注 |
|------|------|------|--------|----------|------|
| Phase 0 | TileLang 环境准备 | ✅ 已完成 | - | [phase0_tilelang_setup](./dev_design/phase0_tilelang_setup.md) | 基础功能验证通过 |
| Phase 0 | TT-Metal 编译 | ✅ 已完成 | - | [phase0_tt_metal_build](./dev_design/phase0_tt_metal_build.md) | libtt_metal.so 18MB |
| Phase 0 | TT-Sim 配置 | ✅ 已完成 | - | [phase0_tt_sim_build](./dev_design/phase0_tt_sim_build.md) | libttsim_bh.so 182KB |
| Phase 1 | CodeGen 框架 | 🔄 **需重构** | - | [phase1_codegen_framework](./dev_design/phase1_codegen_framework.md) | 生成C代码而非TT-Metal Kernel |
| Phase 1 | Runtime 框架 | ⏳ 未开始 | - | [phase1_runtime_framework](./dev_design/phase1_runtime_framework.md) | 需要实现Python绑定 |
| Phase 1 | E2E Copy 测试 | ✅ 已完成 | - | [PHASE1_TTSIM_TEST_REPORT](../tests/target/PHASE1_TTSIM_TEST_REPORT.md) | 手动编写Kernel验证 |
| Phase 2 | SplitBlackholeKernel | ✅ 已实现 | - | [phase2_split_blackhole_kernel](./dev_design/phase2_split_blackhole_kernel.md) | R/C/W 拆分Pass |
| Phase 2 | PlanBlackholeCB | ✅ 已实现 | - | [phase2_plan_blackhole_cb](./dev_design/phase2_plan_blackhole_cb.md) | CB 分配Pass |
| Phase 2 | PlanBlackholeCB | ✅ 已实现 | - | [phase2_assign_blackhole_cores](./dev_design/phase2_assign_blackhole_cores.md) | Core分配Pass |
| Phase 3 | GEMM 支持 | 🐛 **阻塞** | - | [phase3_gemm](./dev_design/phase3_gemm.md) | CodeGen重构后才能进行 |
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
| CodeGenBlackhole 生成格式错误 | P0 | 🐛 阻塞 | Phase 1/3 | 生成C代码而非TT-Metal Kernel |
| 缺少Runtime Python绑定 | P1 | 🐛 阻塞 | Phase 1 | 无法从Python调用编译执行 |
| 缺少端到端测试流程 | P1 | 🐛 阻塞 | Phase 3 | DSL→TT-Sim完整链路不通 |

### 问题详细说明

#### 1. CodeGenBlackhole 生成格式错误 (P0)

**问题描述**: `CodeGenBlackhole` 继承 `CodeGenCHost`，生成标准C函数格式，
无法在TT-Metal上编译为RISC-V ELF。

**当前生成代码示例**:
```cpp
void matmul_kernel(half* A, half* B, half* C) {
  uint8_t buf_dyn_shmem[4096];  // 错误：应该是CB
  float C_local[1024];
  // ... 普通C代码
}
```

**期望生成代码**:
```cpp
void kernel_main() {
  uint32_t A_addr = get_arg_val<uint32_t>(0);
  uint32_t B_addr = get_arg_val<uint32_t>(1);
  uint32_t C_addr = get_arg_val<uint32_t>(2);

  cb_reserve_back(cb_id, 1);
  uint32_t l1_addr = get_write_ptr(cb_id);
  noc_async_read_tile(...);
  noc_async_read_barrier();
  cb_push_back(cb_id, 1);
  // ...
}
```

**解决方案**: 重写 `CodeGenBlackhole::AddFunction` 和相关visitor方法

**所需工作**:
- 重写函数入口生成逻辑
- 实现参数到 `get_arg_val` 的转换
- 实现 `T.alloc_shared` 到 CB 的转换
- 实现 `T.copy` 到 NOC 异步读写的转换
- 实现循环到 tile 迭代的转换

**预估工作量**: 2-3 天

#### 2. 缺少Runtime Python绑定 (P1)

**问题描述**: 无法从Python直接编译并执行生成的kernel。

**需要实现**:
- Python接口调用TT-Metal JIT编译
- 自动保存kernel文件到TT-Metal目录
- 自动编译并执行
- 结果返回给Python

**参考模式**: TileLang CUDA后端通过`CythonKernelAdapter`调用

#### 3. 缺少端到端测试流程 (P1)

**问题描述**: 即使kernel生成正确，也没有自动化流程将DSL代码转为TT-Sim执行。

**期望流程**:
```python
# Python端
import tilelang
func = matmul_kernel(M, N, K)
artifact = tilelang.lower(func, target="blackhole")
result = artifact.execute(A, B)  # 自动编译→执行→返回结果
```

**实际状态**: 目前需要手动复制代码、编写host程序、CMake编译、shell执行。

---

## 下一步行动计划

### 短期目标 (1-2周)

1. **重构 CodeGenBlackhole** (P0)
   - [ ] 重写 `AddFunction` 生成 `kernel_main` 入口
   - [ ] 实现参数到 `get_arg_val` 的映射
   - [ ] 实现 buffer 到 CB 的转换
   - [ ] 实现 `T.copy` 到 NOC 的转换
   - [ ] 验证生成的kernel可在TT-Sim编译执行

2. **创建端到端测试脚本** (P1)
   - [ ] Python脚本：DSL→保存kernel→调用CMake编译→执行→验证
   - [ ] 对比结果与Numpy参考

### 中期目标 (2-4周)

3. **实现Runtime Python绑定**
   - [ ] 研究TileLang CUDA Runtime实现
   - [ ] 实现Blackhole专用Runtime模块
   - [ ] 实现 `artifact.execute()` 接口

4. **完整GEMM支持**
   - [ ] 验证T.gemm()生成正确kernel
   - [ ] 多核支持
   - [ ] 性能基准测试

---

## 已完成工作归档

### 已实现但需重构的工作

**CodeGenBlackhole 基础** (需重写)
- 文件: `tilelang_repo/src/target/codegen_blackhole.cc/.h`
- 状态: 框架存在，生成逻辑错误
- 问题: 继承 `CodeGenCHost`，生成C函数而非TT-Metal Kernel

**Blackhole Target 注册** (可用)
- 文件: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- 状态: 正常
- 功能: `target.build.tilelang_blackhole` 已注册

**TIR Transform Passes** (可用)
- SplitBlackholeKernel: R/C/W拆分
- PlanBlackholeCB: CB分配规划
- AssignBlackholeCores: Core分配
- 状态: 实现完成，单元测试通过

**Phase 1 TT-Sim Copy测试** (参考用)
- 手动编写的Kernel和Host程序
- 验证了TT-Sim执行流程
- 可作为CodeGen生成目标的参考

### 真正可用的端到端流程

目前唯一可用的完整流程是**手动编写**:

```
tt_metal_repo/tt_metal/programming_examples/tilelang_copy_test/
├── kernels/phase1_copy_kernel.cpp  (手动编写)
├── phase1_ttsim_host.cpp           (手动编写)
└── CMakeLists.txt                  (手动配置)
```

执行: `./build/programming_examples/phase1_tilelang_ttsim` ✅

---

## 参考文档

- [架构设计](./arch_design.md) - 总体架构设计（需更新，与实现有偏差）
- [开发设计目录](./dev_design/) - 各任务详细设计
- [开发经验](../memory/general_dev.md) - 最佳实践
- [问题记录](../memory/bugs.md) - Bug 解决方案
- [TT-Sim GEMM 测试报告](../tests/target/TILELANG_GEMM_TTSIM_TEST.md) - 手动Kernel测试

---

## 经验教训

### 重要发现

1. **CodeGen 继承关系陷阱**
   - 最初认为继承 `CodeGenCHost` 可以复用代码
   - 实际上TT-Metal kernel与普通C函数差异太大，需要独立实现

2. **TT-Metal Kernel 特殊性**
   - 必须使用 `kernel_main` 入口
   - 必须使用 TT-Metal API（CB/NOC）
   - 不能直接使用标准C内存操作

3. **端到端测试复杂度**
   - 不仅是代码生成问题
   - 还涉及文件系统操作、CMake编译、进程调用
   - 需要完整的Runtime支持

### 修正的计划

原计划: DSL → TIR → C++ → 执行
实际: DSL → TIR → **TT-Metal Kernel** → JIT编译 → 执行

需要补充: CodeGen 重构 + Runtime 实现
