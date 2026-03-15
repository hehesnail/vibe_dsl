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
| A: 完整编译 (CUDA+HIP) | 功能最全 | 需要 GPU 环境 | - |
| B: 基础编译 (仅 CPU) | 快速、无硬件依赖 | 无法测试 GPU 功能 | ✅ |
| C: 使用 Docker | 环境隔离 | 额外开销 | - |

选择方案 B，因为 Phase 0 只需要基础功能，Blackhole 后端开发不依赖 CUDA/HIP。

### 详细设计

**编译步骤**:

```bash
# 1. 进入源码目录
cd tilelang_repo

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 配置 CMake (关闭 CUDA/HIP，专注基础功能)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTILELANG_ENABLE_CUDA=OFF \
  -DTILELANG_ENABLE_ROCM=OFF \
  -DTILELANG_ENABLE_METAL=OFF \
  -DTILELANG_ENABLE_PYTHON_BINDINGS=ON

# 4. 编译
make -j$(nproc)

# 5. 安装 Python 包
cd .. && pip install -e .
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
  - 成功编译 TileLang C++ 库（libtilelang.so）
  - 安装 tilelang Python 包 (v0.1.8)
  - 通过基础功能测试（DSL + Lowering）
- **遇到问题**:
  - pip install -e 需要 scikit-build-core，已解决
  - DSL 测试需要从文件运行（不能 -c 内联），已解决
- **下一步**: TT-Metal 编译准备

## 经验总结

### 关键决策
- 使用 `--no-build-isolation` 参数安装，避免 scikit-build-core 环境隔离问题
- DSL 测试代码必须写入文件运行，因为 TileLang 需要获取源码 AST

### 可复用模式
```bash
# TileLang 编译流程
cd tilelang_repo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd .. && pip install -e . --no-build-isolation
```

### 踩过的坑
1. scikit-build-core 未安装时 pip install 会报错
2. @T.prim_func 装饰器需要源码文件，不支持 python -c 内联代码
3. 从 tilelang.language 导入 T，而不是 tilelang.T
