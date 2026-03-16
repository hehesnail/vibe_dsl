# Phase 0: TT-Metal 编译

## 基本信息

- **任务ID**: phase0_tt_metal_build
- **所属阶段**: Phase 0
- **前置任务**: phase0_tilelang_setup
- **负责人**: -
- **状态**: 进行中

## 目标

编译 TT-Metal 框架，生成 `libtt_metal.so` 动态链接库，为 TileLang Blackhole 后端提供运行时支持。

## 设计概要

### 输入

- TT-Metal 源代码 (tt_metal_repo/)
- 系统环境 (Ubuntu, CMake, Python 3)
- 已编译的 TileLang 环境

### 输出

- `libtt_metal.so` - TT-Metal 核心动态库
- `libdevice.so` - 设备操作库
- 必要的头文件路径配置

### 核心逻辑

1. 安装 TT-Metal 系统依赖
2. 配置 CMake 构建（启用 Blackhole 支持）
3. 编译共享库
4. 验证库文件生成
5. 配置环境变量

## 实现方案

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A: 使用 build_metal.sh | 官方推荐，配置完整 | 可能包含不需要的组件 | ✅ |
| B: 手动 CMake | 精简，只编译需要部分 | 需要深入了解依赖 | - |
| C: 使用 Python 包 | 简单 | 性能差，不符合架构设计 | - |

选择方案 A，使用官方 build_metal.sh 脚本，确保 Blackhole 支持完整。

### 详细设计

**编译步骤**:

```bash
# 1. 进入源码目录
cd tt_metal_repo

# 2. 安装依赖（如需要）
# sudo ./install_dependencies.sh

# 3. 使用官方脚本编译共享库
./build_metal.sh --build-shared-libs --enable-blackhole

# 4. 验证输出
ls build/lib/libtt_metal.so
ls build/lib/libdevice.so
```

**关键编译选项**:
- `--build-shared-libs`: 生成动态链接库（不是静态库）
- `--enable-blackhole`: 启用 Blackhole 架构支持
- `--release`: Release 模式（优化性能）

**环境变量配置**:

```bash
# 添加到 .bashrc 或环境配置
export TT_METAL_HOME=/root/dev/vibe_dsl/tt_metal_repo
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH
```

## 测试计划

- [ ] TT-Metal 编译通过
- [ ] libtt_metal.so 生成成功
- [ ] 基础头文件路径正确
- [ ] 简单示例可以链接运行

## 开发记录

### 2026-03-15

- **今日完成**:
  - 创建任务设计文档
  - 成功编译 TT-Metal（libtt_metal.so 18MB，libdevice.so 4.6MB）
  - 解决编译依赖问题
- **遇到问题**:
  1. clang-20 未找到 → 创建软链接 clang-20 -> clang
  2. libnuma-dev 缺失 → apt 安装
  3. libhwloc-dev 缺失 → apt 安装
  4. libcapstone-dev 缺失 → apt 安装
  5. RPATH 配置问题 → 使用 CMAKE_BUILD_WITH_INSTALL_RPATH=ON
  6. 运行时库路径问题 → 设置 LD_LIBRARY_PATH
- **解决方案**: 记录在设计文档中
- **下一步**: 提交任务完成

## 经验总结

（任务完成后填写）

