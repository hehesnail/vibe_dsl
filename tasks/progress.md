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

**阶段**: Phase 3 - GEMM 支持（Runtime 完善中）
**目标**: 实现从 TileLang DSL 到 TT-Sim 执行的完整端到端流程
**开始日期**: 2026-03-16
**当前状态**: 🔄 **分层验证完成（CodeGen + PyTorch 对比），真正的端到端测试待实现**

### 关键成果

**CodeGenBlackhole 已重写**: 现在生成符合 TT-Metal 格式的 kernel 代码：
- ✅ 函数入口: `void kernel_main()`
- ✅ 参数获取: `get_arg_val<uint32_t>(idx)`
- ✅ 支持 TT-Metal API 头文件包含
- ✅ 生成代码可通过 TileLang Build 流程

**分层验证进展**（⚠️ 注意：真正的端到端测试 = DSL -> TIR -> CodeGen -> Execute -> Compare，目前仅完成前两步）：
- ✅ TileLang DSL -> TIR -> CodeGen 流程验证通过
- ✅ PyTorch 参考结果生成（用于后续对比）
- ⚠️ Copy Kernel（仅生成代码+参考数据，未执行）
- ⚠️ Element-wise Add（仅生成代码+参考数据，未执行）
- 🐛 GEMM（lower pass 问题）

**真正的端到端测试待实现**：
- 🔄 Runtime 执行 kernel 并返回结果
- 🔄 Python 端数值对比（np.allclose）

**差异对比**:

| 方面 | 之前（错误） | 现在（正确） |
|------|-------------|-------------|
| 函数入口 | `void func(args)` | `void kernel_main()` |
| 参数获取 | 函数参数 | `get_arg_val<uint32_t>(idx)` |
| 生成格式 | 标准C代码 | TT-Metal Kernel 格式 |

> 📋 **架构详情**: 见 [arch_design.md](./arch_design.md)

---

## 任务状态总览

| 阶段 | 任务 | 状态 | 负责人 | 设计文档 | 备注 |
|------|------|------|--------|----------|------|
| Phase 0 | TileLang 环境准备 | ✅ 已完成 | - | [phase0_tilelang_setup](./dev_design/phase0_tilelang_setup.md) | 基础功能验证通过 |
| Phase 0 | TT-Metal 编译 | ✅ 已完成 | - | [phase0_tt_metal_build](./dev_design/phase0_tt_metal_build.md) | libtt_metal.so 18MB |
| Phase 0 | TT-Sim 配置 | ✅ 已完成 | - | [phase0_tt_sim_build](./dev_design/phase0_tt_sim_build.md) | libttsim_bh.so 182KB |
| Phase 1 | CodeGen 框架 | ✅ **已完成** | - | [phase1_codegen_framework](./dev_design/phase1_codegen_framework.md) | 生成 `kernel_main` TT-Metal 格式 |
| Phase 1 | Runtime 框架 | ✅ **已完成** | - | [phase1_runtime_framework](./dev_design/phase1_runtime_framework.md) | BlackholeModule 核心实现完成 |
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

## 当前状态

### 已完成

| 组件 | 状态 | 文件 | 说明 |
|------|------|------|------|
| BlackholeModule | ✅ 已实现 | `src/target/blackhole_module.h/cc` | TVM Module 接口，MeshDevice 管理，Kernel 缓存 |
| Build 函数 | ✅ 已实现 | `src/target/rt_mod_blackhole.cc` | 提取函数信息，创建 BlackholeModule |
| CMake 配置 | ✅ 已更新 | `CMakeLists.txt` | 添加 blackhole_module.cc 到构建 |
| 测试脚本 | ✅ 已创建 | `tests/target/test_blackhole_e2e.py` | E2E 测试脚本 |

### 验证结果

| 步骤 | 命令 | 结果 |
|------|------|----------|
| 编译 TileLang | `cmake .. -DUSE_BLACKHOLE=ON && make -j32` | ✅ 成功编译 |
| CodeGen 测试 | `python tests/target/test_blackhole_e2e.py` | ✅ Build 测试通过 |
| CodeGen+参考生成 | `python tests/target/test_blackhole_true_e2e.py` | ✅ 2/3 测试通过（⚠️ 非真正E2E，未执行kernel） |

### 仍存在的问题

| 问题 | 优先级 | 状态 | 相关任务 | 备注 |
|------|--------|------|----------|------|
| TT-Metal 库链接 | P2 | 🔄 进行中 | Phase 3 | TT-Metal 依赖链复杂（fmt, nlohmann_json, tt_stl 等）|
| BlackholeModule 执行 | P2 | ⏳ 待实现 | Phase 3 | 需要完整的 MeshDevice/Program 集成 |
| GEMM lower pass | P2 | 🐛 阻塞 | Phase 3 | TileLang PlanAndUpdateBufferAllocationLocation 失败 |

### 问题详细说明

#### 1. TT-Metal API 集成 (P1)

**问题描述**: `BlackholeModule` 当前是 stub 实现，需要接入实际的 TT-Metal API。

**当前状态**:
- ✅ `BlackholeModuleNode` 类结构完成
- ✅ `GetFunction` 返回 PackedFunc 正常工作
- 🔄 `EnsureDeviceInitialized` 需要接入 `MeshDevice::create()`
- 🔄 `GetOrCompileProgram` 需要接入 TT-Metal JIT 编译

**解决方案**: 分阶段实现：
1. 阶段1: 接入设备初始化和内存分配
2. 阶段2: 接入 Program 和 Kernel 编译
3. 阶段3: 接入执行队列和同步

**预估工作量**: 3-5 天

#### 2. 缺少Runtime Python绑定 (P1)

**问题描述**: 无法从Python直接编译并执行生成的kernel。

**需要实现**:
- Python接口调用TT-Metal JIT编译
- 自动保存kernel文件到TT-Metal目录
- 自动编译并执行
- 结果返回给Python

**参考模式**: TileLang CUDA后端通过`CythonKernelAdapter`调用

#### 3. 真正的端到端测试流程 (P1)

**端到端测试定义**: DSL -> TIR -> CodeGen -> Runtime Execute -> Python Compare

**问题描述**: 当前仅实现 DSL -> CodeGen + PyTorch 参考生成，缺少 Runtime 执行和结果对比。

**期望流程**:
```python
# Python端
import tilelang
func = matmul_kernel(M, N, K)
artifact = tilelang.lower(func, target="blackhole")
result = artifact.execute(A, B)  # 自动编译→执行→返回结果
np.allclose(result, pytorch_ref)  # 验证正确性
```

**当前状态**:
- ✅ DSL -> CodeGen 完成
- ✅ PyTorch 参考生成完成
- 🔄 Runtime 执行 stub 实现（未接入 TT-Metal）
- ⏳ 结果回传 Python 未实现
- ⏳ 数值对比未进行

---

## 下一步行动计划

### 短期目标 (1-2周)

1. **CodeGenBlackhole** ✅ 已完成
   - [x] 重写 `AddFunction` 生成 `kernel_main` 入口
   - [x] 实现参数到 `get_arg_val` 的映射
   - [x] 编译测试通过
   - [ ] 实现 buffer 到 CB 的转换（待完善）
   - [ ] 实现 `T.copy` 到 NOC 的转换（待完善）

2. **TT-Metal Runtime 集成** (P1)
   - [ ] 接入 `MeshDevice` 设备初始化
   - [ ] 接入 `CreateProgram` 和 `CreateKernelFromString`
   - [ ] 接入 `EnqueueProgram` 执行

3. **真正的端到端测试** (P1) ⚠️ 关键缺口
   - [ ] Runtime 执行 kernel 并返回结果到 Python
   - [ ] Python 端数值对比（np.allclose with PyTorch/NumPy 参考）
   - [ ] 自动化测试：DSL → CodeGen → Execute → Verify

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

### 已实现的工作

**CodeGenBlackhole** ✅ 可用
- 文件: `tilelang_repo/src/target/codegen_blackhole.cc/.h`
- 状态: 已实现 `kernel_main` 入口和 `get_arg_val` 参数加载
- 生成格式: TT-Metal Kernel 格式（符合 `dataflow_api.h` 规范）
- 测试: `test_blackhole_e2e.py` 通过

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

### 分层验证流程（⚠️ 非真正端到端）

**2026-03-16 更新**: 实现分层验证前两步（DSL → CodeGen + PyTorch 参考生成）

```
TileLang DSL (Python)
    ↓ tilelang.lower()
TIR (TVM IR)
    ↓ CodeGenBlackhole
TT-Metal Kernel (C++)

[缺口] Runtime Execute → Result → Python Compare
    ↓
PyTorch Reference (仅生成参考，未对比)
```

**测试脚本**: `tests/target/test_blackhole_true_e2e.py`（⚠️ 命名有误，实际非E2E）
- ⚠️ Copy Kernel (FP16): DSL → CodeGen + 参考数据生成（未执行kernel）
- ⚠️ Element-wise Add (FP16): DSL → CodeGen + 参考数据生成（未执行kernel）
- 🐛 GEMM (FP16→FP32): lower pass 失败

**执行命令**:
```bash
python tests/target/test_blackhole_true_e2e.py
```

### 手动编写的参考实现

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

4. **TT-Metal 库依赖链**
   - TT-Metal 依赖 fmt, nlohmann_json, tt_stl, umd 等多个库
   - 头文件路径复杂，需要逐一添加 include 路径
   - 直接链接 TT-Metal 库困难，建议采用外部进程调用方式

5. **分层验证策略**
   - 第一层: CodeGen 正确性（生成代码格式检查）
   - 第二层: PyTorch 参考对比（验证算法正确性）
   - 第三层: TT-Sim/Hardware 执行（完整验证）
   - 前两层可以在无硬件环境下完成

### 修正的计划

原计划: DSL → TIR → C++ → 执行
实际: DSL → TIR → **TT-Metal Kernel** → JIT编译 → 执行

状态更新:
- ✅ CodeGen 重写完成（kernel_main + get_arg_val）
- 🔄 分层验证完成（CodeGen + PyTorch 参考生成），真正的端到端测试待实现
- 🔄 Runtime 实现中（BlackholeModule TT-Metal 集成）
