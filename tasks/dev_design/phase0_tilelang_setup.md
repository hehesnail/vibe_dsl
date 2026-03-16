# Phase 0: TileLang 环境准备

## 基本信息

- **任务ID**: phase0_tilelang_setup
- **所属阶段**: Phase 0
- **前置任务**: 无
- **负责人**: -
- **状态**: 已完成

## 目标

编译并安装 TileLang，验证基础功能正常，为后续 Blackhole 后端开发奠定基础。

## 设计概要

### 输入

- TileLang 源代码 (tilelang_repo/)
- 系统环境 (Ubuntu, CMake, Python 3)

### 输出

- 编译好的 TileLang C++ 库
- 安装好的 tilelang Python 包
- 通过基础功能测试

### 核心逻辑

1. 安装系统依赖
2. 配置 CMake 构建
3. 编译 C++ 核心库
4. 安装 Python 包
5. 运行基础测试验证

## 实现方案

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A: 完整编译 (CUDA+HIP) | 功能最全 | 需要 GPU 环境 | ✅ |
| B: 基础编译 (仅 CPU) | 快速、无硬件依赖 | 无法测试 GPU 功能 | - |
| C: 使用 Docker | 环境隔离 | 额外开销 | - |

选择方案 A，因为有 8x A100 GPU 环境，可以完整测试 CUDA 后端作为参考。

### 详细设计

**环境设置（推荐）**:

使用提供的 setup 脚本一键配置：
```bash
./setup_tilelang.sh
```

该脚本会自动完成：
1. 克隆 tilelang_repo（含子模块）
2. 配置上游远程
3. 编译（启用 CUDA）
4. 安装 Python 包
5. 验证安装

**手动编译步骤**:

```bash
# 1. 进入源码目录
cd tilelang_repo

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 配置 CMake（启用 CUDA）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DTILELANG_ENABLE_CUDA=ON

# 4. 编译
make -j$(nproc)

# 5. 安装 Python 包（使用 .pth 方式）
echo "$(pwd)/.." > "$(python3 -c "import site; print(site.getsitepackages()[0])")/tilelang.pth"
```

**验证测试**:

```python
# 基础导入测试
import tilelang
from tilelang import T

# 简单 DSL 测试
@T.prim_func
def simple_add(A: T.Buffer((128,), "float32"),
               B: T.Buffer((128,), "float32"),
               C: T.Buffer((128,), "float32")):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        tid = T.thread_binding(0, 128, thread="threadIdx.x")
        C[tid] = A[tid] + B[tid]
```

## 测试计划

- [ ] TileLang 基础编译通过
- [ ] Python 包安装成功
- [ ] 基础导入测试通过
- [ ] 简单 DSL 测试通过

## 开发记录

### 2026-03-15

- **今日完成**:
  - 创建任务设计文档
  - 成功编译 TileLang C++ 库（libtilelang.so，CUDA 启用）
  - 安装 tilelang Python 包 (v0.1.8+cu128)
  - 通过基础功能测试（DSL + Lowering）
  - 创建 setup_tilelang.sh 脚本（简化环境配置）
  - 更新 .gitignore（排除 tilelang_repo/）
- **遇到问题**:
  - pip install -e 需要 scikit-build-core，已解决
  - DSL 测试需要从文件运行（不能 -c 内联），已解决
  - tilelang_repo 体积太大（1.1GB）不能直接提交，采用 setup 脚本方案
- **下一步**: TT-Metal 编译准备

## 经验总结

### 关键决策
1. **仓库组织**: tilelang_repo 作为独立子目录，不提交到 vibe_dsl（体积太大 1.1GB）
   - 提供 setup_tilelang.sh 脚本一键初始化
   - 在 .gitignore 中排除 tilelang_repo/
   - Blackhole 后端代码开发在 tilelang_repo 中，提交到 fork

2. **Python 包安装**: 使用 .pth 文件而非 pip install -e
   - 避免 scikit-build-core 重新运行 cmake 的问题
   - 直接指向 build 目录，使用已编译的库

3. **CUDA 启用**: 有 8x A100 环境，完整编译 CUDA 后端作为参考

### 可复用模式
```bash
# 快速设置
./setup_tilelang.sh

# 手动编译
cd tilelang_repo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTILELANG_ENABLE_CUDA=ON
make -j$(nproc)
echo "$(pwd)/.." > "$(python3 -c "import site; print(site.getsitepackages()[0])")/tilelang.pth"
```

### 踩过的坑
1. tilelang_repo 体积 1.1GB（含 3rdparty），不能直接提交到 vibe_dsl
2. pip install -e 会重新运行 cmake，可能因环境变量问题失败
3. @T.prim_func 装饰器需要源码文件，不支持 python -c 内联代码
4. 从 tilelang.language 导入 T，而不是 tilelang.T
