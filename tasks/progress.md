# TileLang Tenstorrent 后端开发进度

## 当前阶段

**阶段**: 初始规划
**目标**: 建立完整的后端扩展框架
**开始日期**: 2026-03-13

---

## 任务清单

### 阶段1: 基础设施建立
- [x] 项目结构理解
  - [x] TileLang 后端架构分析
  - [x] TT-Metal 编程模型梳理
  - [x] 现有后端（CUDA/HIP）代码分析
- [x] 开发环境建立
  - [x] CLAUDE.md 编写
  - [x] 知识管理文件创建

### 阶段2: TT-Metal 深度调研
- [ ] JIT 编译系统分析
  - [ ] `jit_build/` 模块理解
  - [ ] Kernel 编译流程梳理
- [ ] 内存模型调研
  - [ ] Circular Buffer 机制
  - [ ] DRAM/L1/Register 层次映射
- [ ] 执行模型分析
  - [ ] Reader/Compute/Writer Kernel 分离机制
  - [ ] 与 TileLang unified kernel 的对应方案

### 阶段3: CodeGen 实现
- [ ] 基础框架
  - [ ] `CodeGenTileLangMetal` 类创建
  - [ ] 头文件模板系统
- [ ] 类型系统
  - [ ] 基础数据类型映射
  - [ ] 特殊类型支持（BF16等）
- [ ] 内存 Scope 映射
  - [ ] `shared` → L1 SRAM
  - [ ] `local` → Register
  - [ ] `global` → DRAM

### 阶段4: Runtime 集成
- [ ] 运行时模块实现
- [ ] 与 TT-Metal 的加载机制对接

### 阶段5: 验证与优化
- [ ] 简单算子验证（copy）
- [ ] 复杂算子验证（gemm）
- [ ] 性能优化

---

## 当前阻塞问题

| 问题 | 优先级 | 状态 | 备注 |
|------|--------|------|------|
| 代码生成路径选择 | 高 | 分析中 | 倾向生成 TT-Metal C++ Kernel |
| 内存模型映射 | 高 | 待调研 | 需理解 CB 机制 |
| 执行模型适配 | 中 | 待调研 | unified vs 分离 kernel |

---

## 下一步行动

1. 深入阅读 `docs/tt_metal/source_analysis/jit_build.md`
2. 研究 TT-Metal 的 Kernel 编译接口
3. 设计 `CodeGenTileLangMetal` 初步架构

---

## 参考资料

- [TileLang 后端架构](../docs/tilelang/cpp_core/04_target_backends.md)
- [TT-Metal 源码分析](../docs/tt_metal/source_analysis/)
- [TT-Metal JIT 编译](../docs/tt_metal/source_analysis/jit_build.md)
