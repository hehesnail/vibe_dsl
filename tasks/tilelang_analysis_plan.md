# TileLang 源代码分析计划

## 项目概述

TileLang 是一个用于开发高性能 GPU/CPU 内核的领域特定语言 (DSL)，基于 TVM 编译器基础设施。

**项目规模:**
- Python 文件: 217 个
- C++ 文件: 200 个 (cc + h)
- 示例文件: 258 个
- 仓库地址: https://github.com/tile-ai/tilelang

## 目标

对 TileLang 项目的所有模块进行详细分析，输出到 `docs/tilelang/` 文件夹下。

## 模块分析规划

### 1. 项目概览与架构 (docs/tilelang/README.md)
**内容:**
- 项目背景与设计哲学
- 整体架构图
- 模块依赖关系
- 编译流程概述
- 支持的硬件后端 (CUDA, ROCm, Metal, WebGPU, Ascend)

**涵盖源码:**
- `README.md`
- `pyproject.toml`
- `CMakeLists.txt`

---

### 2. Python 核心包分析 (docs/tilelang/python_core/)

#### 2.1 包入口与初始化 (01_package_init.md)
**内容:**
- `__init__.py` 导出的 API
- 版本管理 (`libinfo.py`, `dtypes.py`)
- 类型定义 (`_typing.py`)
- FFI 接口 (`_ffi_api.py`, `ir.py`)

**涵盖源码:**
- `tilelang/__init__.py`
- `tilelang/libinfo.py`
- `tilelang/dtypes.py`
- `tilelang/_typing.py`
- `tilelang/_ffi_api.py`
- `tilelang/ir.py`

#### 2.2 环境管理 (02_environment.md)
**内容:**
- 环境变量管理
- TVM 集成
- 配置系统
- 缓存机制

**涵盖源码:**
- `tilelang/env.py`
- `tilelang/cache/`

#### 2.3 DSL 语言核心 (03_language_core/)

##### 03a_kernel_and_frame.md
**内容:**
- 内核定义机制 (`language/kernel.py`)
- 栈帧管理 (`language/frame.py`)
- 装饰器实现 (`language/customize.py`)

**涵盖源码:**
- `tilelang/language/kernel.py` (13KB)
- `tilelang/language/frame.py` (6KB)
- `tilelang/language/customize.py` (3KB)

##### 03b_builtin_ops.md
**内容:**
- 内置操作详解 (`language/builtin.py`)
- 线程/块管理 (blockIdx, threadIdx)
- 内存操作
- 同步原语
- Tensor Core 操作 (WGMMA, MMA)

**涵盖源码:**
- `tilelang/language/builtin.py` (47KB - 核心文件)

##### 03c_memory_ops.md
**内容:**
- 内存分配 (`language/allocate.py`)
- 拷贝操作 (`language/copy_op.py`)
- 填充操作 (`language/fill_op.py`)
- Shared Memory 管理

**涵盖源码:**
- `tilelang/language/allocate.py` (13KB)
- `tilelang/language/copy_op.py` (9KB)
- `tilelang/language/fill_op.py` (3KB)

##### 03d_compute_ops.md
**内容:**
- GEMM 操作 (`language/gemm_op.py`)
- 归约操作 (`language/reduce_op.py`)
- 原子操作 (`language/atomic.py`)
- 数学内联函数 (`language/math_intrinsics.py`)
- 稀疏 GEMM (`language/experimental/gemm_sp.py`)

**涵盖源码:**
- `tilelang/language/gemm_op.py` (13KB)
- `tilelang/language/reduce_op.py` (18KB)
- `tilelang/language/atomic.py` (21KB)
- `tilelang/language/math_intrinsics.py` (10KB)
- `tilelang/language/experimental/gemm_sp.py` (8KB)

##### 03e_control_flow.md
**内容:**
- 循环管理 (`language/loop.py`)
- 条件执行
- 分支与跳转

**涵盖源码:**
- `tilelang/language/loop.py` (10KB)
- `tilelang/language/logical.py` (3KB)

##### 03f_data_types_and_proxy.md
**内容:**
- 数据类型系统 (`language/dtypes.py`)
- 代理对象 (`language/proxy.py`)
- 类型推断

**涵盖源码:**
- `tilelang/language/dtypes.py` (23KB)
- `tilelang/language/proxy.py` (11KB)

##### 03g_ast_and_parser.md
**内容:**
- AST 定义 (`language/ast/`)
- 解析器实现 (`language/parser/`)
- TIR 转换 (`language/tir/`)
- Eager 模式 (`language/eager/`)

**涵盖源码:**
- `tilelang/language/ast/ir.py` (59KB)
- `tilelang/language/parser/`
- `tilelang/language/tir/`
- `tilelang/language/eager/builder.py` (48KB)
- `tilelang/language/eager/ast.py` (26KB)

##### 03h_advanced_features.md
**内容:**
- 集群操作 (`language/cluster.py`)
- Warp 组管理 (`language/warpgroup.py`)
- 快速数学 (`language/fastmath.py`)
- 打印/调试 (`language/print_op.py`)
- 随机数 (`language/random.py`)
- 注解 (`language/annotations.py`)

**涵盖源码:**
- `tilelang/language/cluster.py`
- `tilelang/language/warpgroup.py`
- `tilelang/language/fastmath.py`
- `tilelang/language/print_op.py` (9KB)
- `tilelang/language/random.py`
- `tilelang/language/annotations.py`

---

### 3. JIT 编译系统 (docs/tilelang/jit/)

#### 3.1 JIT 核心 (01_jit_core.md)
**内容:**
- JIT 架构概述
- 内核编译流程 (`jit/kernel.py`)
- 执行后端 (`jit/execution_backend.py`)
- 参数处理 (`jit/param.py`)
- 异常处理 (`jit/exceptions.py`)

**涵盖源码:**
- `tilelang/jit/__init__.py` (21KB)
- `tilelang/jit/kernel.py` (28KB)
- `tilelang/jit/execution_backend.py`
- `tilelang/jit/param.py`
- `tilelang/jit/exceptions.py`

#### 3.2 适配器详解 (02_adapters.md)
**内容:**
- TVM FFI 适配器 (`jit/adapter/tvm_ffi.py`)
- CuTeDSL 适配器 (`jit/adapter/cutedsl/`)
  - CuTeDSL 适配实现
  - 代码生成
  - Wrapper 生成
- NVRTC 适配器 (`jit/adapter/nvrtc/`)
  - NVRTC 编译
  - CUDA 代码生成
- Cython 适配器 (`jit/adapter/cython/`)
- Torch 适配器 (`jit/adapter/torch/`)
- 包装器生成 (`jit/adapter/wrapper.py`)
- 库生成 (`jit/adapter/libgen.py`)

**涵盖源码:**
- `tilelang/jit/adapter/tvm_ffi.py` (13KB)
- `tilelang/jit/adapter/cutedsl/adapter.py` (17KB)
- `tilelang/jit/adapter/cutedsl/wrapper.py` (60KB)
- `tilelang/jit/adapter/nvrtc/adapter.py`
- `tilelang/jit/adapter/nvrtc/wrapper.py` (24KB)
- `tilelang/jit/adapter/cython/adapter.py` (17KB)
- `tilelang/jit/adapter/wrapper.py` (42KB)

---

### 4. 代码变换与优化 (docs/tilelang/transform/)

#### 4.1 Python 层变换 (python_transforms.md)
**内容:**
- Python 层的代码变换 pass
- 优化策略
- 分析工具

**涵盖源码:**
- `tilelang/transform/`

#### 4.2 C++ 层变换 (cpp_transforms.md)
**内容:**
- C++ 层的编译优化
- TVM IR 变换
- 目标代码生成

**涵盖源码:**
- `src/transform/`

---

### 5. C++ 核心源码 (docs/tilelang/cpp_core/)

#### 5.1 IR 定义 (01_ir_definition.md)
**内容:**
- TileLang IR 定义 (`ir.cc`)
- 配置系统 (`config.h`)

**涵盖源码:**
- `src/ir.cc` (15KB)
- `src/config.h`

#### 5.2 操作实现 (02_operations/)

##### 02a_copy_and_memory.md
**内容:**
- 拷贝操作实现 (`op/copy.cc`, `op/copy.h`)
- 内存操作
- Async Copy

**涵盖源码:**
- `src/op/copy.cc` (85KB - 重要)
- `src/op/copy.h` (15KB)

##### 02b_gemm_operations.md
**内容:**
- GEMM 核心实现 (`op/gemm.cc`, `op/gemm.h`)
- Python 绑定 (`op/gemm_py.cc`)
- 稀疏 GEMM (`op/gemm_sp.cc`)

**涵盖源码:**
- `src/op/gemm.cc` (33KB)
- `src/op/gemm.h`
- `src/op/gemm_py.cc` (13KB)
- `src/op/gemm_sp.cc` (14KB)
- `src/op/gemm_sp_py.cc` (11KB)

##### 02c_reduction_and_atomic.md
**内容:**
- 归约操作 (`op/reduce.cc`, `op/reduce.h`)
- 原子操作 (`op/atomic_add.cc`, `op/atomic_reduce.cc`)
- 并行处理 (`op/parallel.cc`)

**涵盖源码:**
- `src/op/reduce.cc` (25KB)
- `src/op/reduce.h`
- `src/op/atomic_add.cc` (24KB)
- `src/op/atomic_reduce.cc` (10KB)
- `src/op/parallel.cc` (31KB)

##### 02d_builtin_and_math.md
**内容:**
- 内置函数 (`op/builtin.cc`, `op/builtin.h`)
- 数学函数 (`op/math.cc`)
- 填充操作 (`op/fill.cc`)
- 逻辑操作 (`op/logical.cc`)
- 区域管理 (`op/region.cc`)

**涵盖源码:**
- `src/op/builtin.cc` (22KB)
- `src/op/builtin.h` (26KB)
- `src/op/math.cc`
- `src/op/fill.cc`
- `src/op/logical.cc`

#### 5.3 布局系统 (03_layout_system.md)
**内容:**
- 布局定义与计算
- Swizzle 操作
- Fragment 布局

**涵盖源码:**
- `src/layout/`
- `tilelang/layout/`

#### 5.4 目标后端 (04_target_backends.md)
**内容:**
- CUDA 后端
- ROCm/HIP 后端
- Metal 后端
- WebGPU 后端
- Ascend 后端支持

**涵盖源码:**
- `src/target/`

#### 5.5 运行时系统 (05_runtime.md)
**内容:**
- 运行时支持
- 内存管理
- 内核启动

**涵盖源码:**
- `src/runtime/`

#### 5.6 模板系统 (06_templates.md)
**内容:**
- CUTLASS/CuTe 模板
- 代码生成模板

**涵盖源码:**
- `src/tl_templates/`

---

### 6. 高级功能模块 (docs/tilelang/advanced/)

#### 6.1 自动微分 (01_autodd.md)
**内容:**
- 自动微分实现 (`autodd.py`)
- 梯度计算
- 反向传播支持

**涵盖源码:**
- `tilelang/autodd.py` (40KB - 重要)

#### 6.2 自动调优 (02_autotuner.md)
**内容:**
- 自动调优器 (`autotuner/`)
- 搜索策略
- 性能分析集成

**涵盖源码:**
- `tilelang/autotuner/`

#### 6.3 性能分析 (03_profiler.md)
**内容:**
- Profiler 实现 (`profiler/`)
- 性能指标收集
- 分析工具

**涵盖源码:**
- `tilelang/profiler/`

#### 6.4 量化支持 (04_quantization.md)
**内容:**
- 量化实现 (`quantize/`)
- 低精度计算

**涵盖源码:**
- `tilelang/quantize/`

#### 6.5 分析工具 (05_analysis.md)
**内容:**
- 代码分析 (`analysis/`)
- 布局分析
- 性能预测

**涵盖源码:**
- `tilelang/analysis/`

#### 6.6 Carver 模块 (06_carver.md)
**内容:**
- Carver 架构分析
- 代码生成
- 优化策略

**涵盖源码:**
- `tilelang/carver/`

#### 6.7 TileOp 模块 (07_tileop.md)
**内容:**
- TileOp 定义
- 操作融合

**涵盖源码:**
- `tilelang/tileop/`

#### 6.8 工具集 (08_tools.md)
**内容:**
- 开发工具 (`tools/`)
- 调试工具

**涵盖源码:**
- `tilelang/tools/`

---

### 7. 示例分析 (docs/tilelang/examples/)

#### 7.1 基础示例 (01_basic_examples.md)
**内容:**
- GEMM 示例 (`examples/gemm/`)
- 基础操作示例

**涵盖源码:**
- `examples/gemm/`

#### 7.2 FlashAttention 示例 (02_flash_attention.md)
**内容:**
- FlashAttention 实现详解
- 前向/反向传播

**涵盖源码:**
- `examples/flash_attention/`

#### 7.3 线性注意力示例 (03_linear_attention.md)
**内容:**
- Linear Attention 实现

**涵盖源码:**
- `examples/linear_attention/`

#### 7.4 DeepSeek MLA 示例 (04_deepseek_mla.md)
**内容:**
- MLA Decoding 实现
- AMD 平台优化

**涵盖源码:**
- `examples/deepseek_mla/`

#### 7.5 量化 GEMM 示例 (05_dequantize_gemm.md)
**内容:**
- 量化矩阵乘法

**涵盖源码:**
- `examples/dequantize_gemm/`

#### 7.6 其他重要示例 (06_other_examples.md)
**内容:**
- Convolution
- Native Sparse Attention
- Mamba
- 注意力 Sink
- 布局分析

**涵盖源码:**
- `examples/convolution/`
- `examples/deepseek_nsa/`
- `examples/mamba2/`
- `examples/attention_sink/`
- `examples/plot_layout/`
- `examples/analyze/`

---

### 8. 测试框架 (docs/tilelang/testing.md)
**内容:**
- 测试架构 (`testing/`)
- 单元测试策略
- 集成测试

**涵盖源码:**
- `tilelang/testing/`
- `testing/`

---

### 9. 基准测试 (docs/tilelang/benchmark.md)
**内容:**
- 基准测试框架 (`benchmark/`)
- 性能对比方法

**涵盖源码:**
- `benchmark/`

---

## 任务执行计划

### Phase 1: 项目概览与核心架构
1. ✅ 克隆仓库，分析项目结构
2. 编写项目概览文档 (README.md)
3. 分析编译流程与架构

### Phase 2: Python DSL 核心
4. 包入口与初始化
5. 环境管理系统
6. 内核定义与框架
7. 内置操作详解
8. 内存操作
9. 计算操作 (GEMM, Reduce, Atomic)
10. 控制流
11. 数据类型与代理
12. AST 与解析器
13. 高级功能

### Phase 3: JIT 编译系统
14. JIT 核心架构
15. TVM FFI 适配器
16. CuTeDSL 适配器
17. NVRTC 适配器
18. Cython 适配器

### Phase 4: C++ 核心实现
19. IR 定义
20. 拷贝与内存操作
21. GEMM 操作
22. 归约与原子操作
23. 内置函数
24. 布局系统
25. 目标后端
26. 运行时系统

### Phase 5: 高级功能
27. 自动微分
28. 自动调优
29. 性能分析
30. 量化支持
31. Carver 模块

### Phase 6: 示例与应用
32. 基础示例分析
33. FlashAttention
34. Linear Attention
35. DeepSeek MLA
36. 其他重要示例

### Phase 7: 文档整合
37. 编写测试框架文档
38. 编写基准测试文档
39. 最终整合与交叉引用

---

## 输出目录结构

```
docs/tilelang/
├── README.md                          # 项目概览
├── 01_architecture_overview.md        # 架构总览
├── 02_compilation_pipeline.md         # 编译流程
├── python_core/
│   ├── 01_package_init.md
│   ├── 02_environment.md
│   ├── 03_language_core/
│   │   ├── 03a_kernel_and_frame.md
│   │   ├── 03b_builtin_ops.md
│   │   ├── 03c_memory_ops.md
│   │   ├── 03d_compute_ops.md
│   │   ├── 03e_control_flow.md
│   │   ├── 03f_data_types_and_proxy.md
│   │   ├── 03g_ast_and_parser.md
│   │   └── 03h_advanced_features.md
│   ├── 04_transform.md
│   └── 05_analysis.md
├── jit/
│   ├── 01_jit_core.md
│   ├── 02_tvm_ffi_adapter.md
│   ├── 03_cutedsl_adapter.md
│   ├── 04_nvrtc_adapter.md
│   └── 05_cython_adapter.md
├── cpp_core/
│   ├── 01_ir_definition.md
│   ├── 02_operations/
│   │   ├── 02a_copy_and_memory.md
│   │   ├── 02b_gemm_operations.md
│   │   ├── 02c_reduction_and_atomic.md
│   │   └── 02d_builtin_and_math.md
│   ├── 03_layout_system.md
│   ├── 04_target_backends.md
│   ├── 05_runtime.md
│   └── 06_templates.md
├── advanced/
│   ├── 01_autodd.md
│   ├── 02_autotuner.md
│   ├── 03_profiler.md
│   ├── 04_quantization.md
│   ├── 05_carver.md
│   ├── 06_tileop.md
│   └── 07_tools.md
├── examples/
│   ├── 01_basic_examples.md
│   ├── 02_flash_attention.md
│   ├── 03_linear_attention.md
│   ├── 04_deepseek_mla.md
│   ├── 05_dequantize_gemm.md
│   └── 06_other_examples.md
├── testing.md
└── benchmark.md
```

---

## 文档风格规范

1. **代码引用**: 使用 `file_path:line_number` 格式
2. **结构**: 每个文档包含：
   - 模块概述
   - 关键类/函数详解
   - 实现逻辑分析
   - 与其他模块的关系
   - 示例代码
3. **图表**: 使用 Mermaid 语法绘制流程图和架构图
4. **代码片段**: 包含关键源码片段并加注释说明

---

## 时间估计

根据项目规模 (217 Python + 200 C++ 文件)，预计完成所有分析需要:
- 每篇文档平均 30-60 分钟
- 约 35-40 篇文档
- 总计约 25-35 小时的分析工作

建议分阶段执行，每完成一个阶段进行验证。
