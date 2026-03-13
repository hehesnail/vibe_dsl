# TileLang 源代码分析

TileLang 是一个用于开发高性能 GPU/CPU 内核的领域特定语言 (DSL)，基于 TVM 编译器基础设施构建。

## 项目概览

- **Python 文件**: ~217 个
- **C++ 文件**: ~200 个
- **示例文件**: ~258 个
- **官方仓库**: https://github.com/tile-ai/tilelang

## 支持的硬件后端

| 后端 | 状态 | 说明 |
|------|------|------|
| NVIDIA CUDA | ✅ | H100 (TMA/WGMMA), A100, V100, RTX 系列 |
| AMD ROCm/HIP | ✅ | MI250/MI300X |
| Apple Metal | ✅ | macOS GPU |
| WebGPU | ✅ | Web 端推理 |
| Huawei Ascend | ✅ | AscendC / NPU IR |

## 核心特性

- **Pythonic DSL**: 使用 Python 语法定义高性能内核
- **自动优化**: 向量化、内存优化、软件流水线
- **多后端**: 统一 DSL 生成 CUDA/HIP/Metal 代码
- **高级功能**: 自动微分、自动调优、性能分析、量化支持

## 文档索引

### Python 核心包
- [01. 包入口与初始化](./python_core/01_package_init.md)
- [03a. Kernel 与 Frame](./python_core/03_language_core/03a_kernel_and_frame.md)
- [03b. 内置操作](./python_core/03_language_core/03b_builtin_ops.md)
- [03c. 内存操作](./python_core/03_language_core/03c_memory_ops.md)
- [03d. 计算操作](./python_core/03_language_core/03d_compute_ops.md)
- [03e. 控制流](./python_core/03_language_core/03e_control_flow.md)
- [03f. 数据类型与代理](./python_core/03_language_core/03f_data_types_and_proxy.md)
- [03g. AST 与解析器](./python_core/03_language_core/03g_ast_and_parser.md)
- [04. 变换系统](./python_core/04_transform.md)

### JIT 编译系统
- [01. JIT 核心](./jit/01_jit_core.md)
- [02. TVM FFI 适配器](./jit/02_tvm_ffi_adapter.md)
- [03. CuTeDSL 适配器](./jit/03_cutedsl_adapter.md)
- [04. NVRTC 适配器](./jit/04_nvrtc_adapter.md)
- [05. Cython 适配器](./jit/05_cython_adapter.md)

### C++ 核心实现
- [01. IR 定义](./cpp_core/01_ir_definition.md)
- [02a. 拷贝与内存操作](./cpp_core/02_operations/02a_copy_and_memory.md)
- [02b. GEMM 操作](./cpp_core/02_operations/02b_gemm_operations.md)
- [02c. 归约与原子操作](./cpp_core/02_operations/02c_reduction_and_atomic.md)
- [02d. 内置函数与数学函数](./cpp_core/02_operations/02d_builtin_and_math.md)
- [03. 布局系统](./cpp_core/03_layout_system.md)
- [04. 目标后端](./cpp_core/04_target_backends.md)
- [05. 运行时系统](./cpp_core/05_runtime.md)
- [06. 模板系统](./cpp_core/06_templates.md)

### 高级功能
- [01. 自动微分](./advanced/01_autodd.md)
- [02. 自动调优](./advanced/02_autotuner.md)
- [03. 性能分析](./advanced/03_profiler.md)
- [04. 量化支持](./advanced/04_quantization.md)
- [05. 分析工具](./advanced/05_analysis.md)
- [06. Carver 模块](./advanced/06_carver.md)

### 示例代码
- [01. 基础示例 (GEMM)](./examples/01_basic_examples.md)
- [02. FlashAttention](./examples/02_flash_attention.md)
- [03. Linear Attention](./examples/03_linear_attention.md)
- [04. DeepSeek MLA](./examples/04_deepseek_mla.md)
- [05. 量化 GEMM](./examples/05_dequantize_gemm.md)
- [06. 其他示例](./examples/06_other_examples.md)

### 其他
- [测试框架](./testing.md)
- [基准测试](./benchmark.md)
- [完成报告](./ANALYSIS_COMPLETE.md)
