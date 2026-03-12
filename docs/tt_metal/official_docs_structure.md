# TT-Metalium 官方文档站点结构分析

> **分析时间**: 2026-03-12
> **文档来源**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/

---

## 1. 文档整体结构

```
TT-Metalium Documentation
├── Get Started (入门)
│   ├── Getting Started
│   └── Install
│
├── TT-Metalium (核心文档)
│   ├── Programming Model (编程模型)
│   ├── Programming Examples (编程示例)
│   ├── Lab Exercises (实验练习)
│   ├── Advanced Topics (高级主题)
│   ├── APIs (API参考)
│   ├── Environment Variables (环境变量)
│   └── Tools (工具)
│
└── Resources (资源)
    ├── Support (支持)
    └── Contributing (贡献指南)
```

---

## 2. 详细导航结构

### 2.1 Get Started 部分

| 页面 | 路径 | 说明 |
|------|------|------|
| Getting Started | `get_started/get_started.html` | 软件栈概述、编程哲学、快速开始 |
| Install | `installing.html` | 安装指南 (二进制/Docker/源码/Anaconda) |

**Getting Started 子章节**:
- Software Stack Overview
- Programming Philosophy
- Installation
- Key Concepts
- Quick Start Guide (Basic/Intermediate/Advanced)
- Resources and Debugging

---

### 2.2 Programming Model (编程模型)

**路径**: `tt_metal/programming_model/index.html`

**内容概要**:
- Host-Device 编程模型
- Kernel 类型详解
- 内存层次结构
- NoC 通信机制

---

### 2.3 Programming Examples (编程示例)

| 示例 | 路径 | 复杂度 |
|------|------|--------|
| DRAM Loopback | `tt_metal/examples/dram_loopback.html` | 入门 |
| Eltwise Binary | `tt_metal/examples/eltwise_binary.html` | 基础 |
| Eltwise SFPU | `tt_metal/examples/eltwise_sfpu.html` | 基础 |
| Matmul (Single Core) | `tt_metal/examples/matmul_single_core.html` | 中级 |
| Matmul (Multi Core) | `tt_metal/examples/matmul_multi_core.html` | 高级 |
| Matmul (Multi Core Optimized) | `tt_metal/examples/matmul_multi_core_optimized.html` | 专家 |
| Custom SFPI Operations | `tt_metal/examples/custom_sfpi.html` | 专家 |

**Matmul Multi Core Optimized 子章节**:
- Data Reuse in `matmul_multicore_reuse`
- Data Multicasting in `matmul_multicore_reuse_mcast`

**Custom SFPI 子章节**:
- Vector addition using SFPI
- Smoothstep using SFPI

---

### 2.4 Lab Exercises (实验练习)

| 实验 | 路径 |
|------|------|
| Lab 1: Single Core Matrix Multiplication | `tt_metal/labs/matmul/lab1/lab1.html` |

**Lab 1 内容**:
- Introduction (矩阵乘法基本算法)
- Row-Major Memory Layout
- Linear Transformation
- Loop Tiling
- TT-Metalium Programming Model
- Debug Facilities (DPRINT, tt-triage, Profiling)
- Matrix Multiplication Implementation Exercise

---

### 2.5 Advanced Topics (高级主题)

| 主题 | 路径 |
|------|------|
| Tiles | `tt_metal/advanced_topics/tiles.html` |
| Memory from kernel developer's perspective | `tt_metal/advanced_topics/memory_for_kernel_developers.html` |
| Compute Engines and Data Flow | `tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html` |
| FP32 Accuracy | `tt_metal/advanced_topics/fp32_accuracy.html` |

**Memory 主题子章节**:
- Data addressing on Tenstorrent processors
- RISC-V Address Space
- DRAM tiles
- Memory access via the NoC
- Tensor Layout
- Memory placement (Lock step, Interleaved, SRAM buffers, Sharded)

---

### 2.6 APIs (API参考) - 核心部分

#### 2.6.1 Host APIs

| 分类 | 路径 | 包含API |
|------|------|---------|
| Program | `host_apis/program/program.html` | CreateProgram |
| Buffers | `host_apis/buffers/buffers.html` | CreateBuffer, CircularBuffers, CreateSemaphore, DeallocateBuffer, AssignGlobalBufferToProgram |
| Device Management | `host_apis/device_management/device_management.html` | CreateDevice, CloseDevice, QueryDevices |
| Kernels | `host_apis/kernels/kernels.html` | CreateKernel, CreateKernelFromString |
| Runtime Arguments | `host_apis/runtime_args/runtime_args.html` | SetRuntimeArgs |
| Profiler | `host_apis/profiler/profiler.html` | ReadMeshDeviceProfilerResults, GetLatestProgramsPerfData, GetAllProgramsPerfData |

#### 2.6.2 Kernel APIs

##### Common APIs

| 分类 | 路径 | API列表 |
|------|------|---------|
| Circular Buffer APIs | `kernel_apis/circular_buffers/circular_buffers.html` | cb_pages_available_at_front, cb_wait_front, cb_pages_reservable_at_back, cb_reserve_back, cb_push_back, cb_pop_front, get_read_ptr, get_write_ptr |
| Kernel Argument APIs | `kernel_apis/kernel_args/kernel_args.html` | get_arg_addr, get_arg_val, get_common_arg_addr, get_common_arg_val, get_compile_time_arg_val |

##### Data Movement APIs (数据移动)

**核心API列表** (共 35+ 个):

| API名称 | 说明 |
|---------|------|
| `noc_async_read` | 异步读取 |
| `noc_async_read_set_state` | 设置读取状态 |
| `noc_async_read_with_state` | 带状态读取 |
| `noc_async_read_one_packet` | 单包读取 |
| `noc_async_read_page` | 页面读取 |
| `noc_async_read_shard` | Shard读取 |
| `noc_async_write` | 异步写入 |
| `noc_async_write_multicast` | 多播写入 |
| `noc_async_write_one_packet` | 单包写入 |
| `noc_async_write_page` | 页面写入 |
| `noc_async_write_shard` | Shard写入 |
| `noc_inline_dw_write` | 内联双字写入 |
| `noc_async_read_barrier` | 读取屏障 |
| `noc_async_write_barrier` | 写入屏障 |
| `noc_async_atomic_barrier` | 原子操作屏障 |
| `noc_async_full_barrier` | 全屏障 |
| `noc_semaphore_set` | 信号量设置 |
| `noc_semaphore_inc` | 信号量增加 |
| `noc_semaphore_wait` | 信号量等待 |
| `get_noc_addr` | 获取NoC地址 |
| `get_noc_addr_from_bank_id` | 从bank ID获取地址 |
| `get_noc_multicast_addr` | 获取多播地址 |

##### Compute APIs (计算)

**Synchronization (同步)**:
- acquire_dst, release_dst, reg_api

**Initialization (初始化)**:
- any_init, binary_init_funcs

**FPU/Matrix Engine (矩阵引擎)**:
- add_tiles, sub_tiles, mul_tiles
- add_tiles_bcast, sub_tiles_bcast, mul_tiles_bcast
- matmul_tiles, matmul_block
- reduce_tile
- transpose_wh_tile
- tilize, untilize

**SFPU/Vector Engine (向量引擎)** - 80+ API:

| 类别 | API数量 | 示例 |
|------|---------|------|
| Basic arithmetic | 16 | add_binary_tile, sqrt_tile, recip_tile |
| Integer operations | 8 | add_int_tile, gcd_tile, remainder_tile |
| Exponential/Log | 7 | exp_tile, log_tile, expm1_tile |
| Comparison/Logical | 19 | unary_gt_tile, max_tile, isinf_tile |
| Bitwise operations | 10 | bitwise_and_tile, left_shift_tile |
| Trigonometric | - | sin_tile, cos_tile, tan_tile |
| Hyperbolic | - | sinh_tile, cosh_tile, tanh_tile |
| Activation functions | - | relu_tile, gelu_tile, sigmoid_tile |
| Rounding | - | floor_tile, ceil_tile, round_tile |
| Miscellaneous | - | sign_tile, clamp_tile, where_tile |

---

### 2.7 Environment Variables (环境变量)

**路径**: `tt_metal/environment_variables/index.html`

---

### 2.8 Tools (工具)

**路径**: `tools/index.html`

包含调试、性能分析工具文档。

---

## 3. API 统计汇总

| API类别 | API数量 | 完整度评估 |
|---------|---------|-----------|
| Host APIs | ~15 | 中 |
| Circular Buffer APIs | 8 | 高 |
| Kernel Argument APIs | 5 | 高 |
| Data Movement APIs | 35+ | 高 |
| Compute APIs (FPU) | 15 | 高 |
| Compute APIs (SFPU) | 80+ | 高 |
| **总计** | **160+** | - |

---

## 4. 与现有文档对比发现

### 4.1 现有文档已覆盖的内容
- [x] 基础架构概念
- [x] 三种Kernel类型
- [x] 基本Host API (CreateDevice, CreateBuffer, CreateKernel)
- [x] 基本CB操作
- [x] 基本Data Movement API
- [x] 基本Compute API (matmul, add, relu等)
- [x] 主要编程示例

### 4.2 现有文档缺失/不完整的内容
- [ ] Host API细节不完整 (缺少Profiler API, Semaphore API)
- [ ] 大量SFPU API未覆盖 (仅覆盖~20%, 实际有80+)
- [ ] Data Movement API不完整 (缺少shard, page, multicast变体)
- [ ] 高级主题细节不足 (Tiles, Memory placement)
- [ ] Lab Exercises内容
- [ ] 环境变量完整列表
- [ ] 工具使用详细指南
- [ ] Python API文档

### 4.3 新增发现的重要API
1. **Data Movement扩展**: read/write shard, page, multicast变体
2. **Semaphore完整套件**: set/inc/wait 及 multicast 版本
3. **Barrier细化**: atomic, full, with_trid 变体
4. **SFPU丰富**: 整数运算、位运算、比较逻辑运算等
5. **Host Profiling API**: ReadMeshDeviceProfilerResults等

---

## 5. 下一步建议

1. **优先补充**: Host Profiler API、完整的Data Movement API
2. **重点扩展**: SFPU API (约60个API待补充)
3. **详细展开**: Advanced Topics (Tiles, Memory)
4. **示例丰富**: Lab Exercises代码和解释
5. **工具文档**: 调试和性能分析工具完整指南

---

*此结构分析为后续文档完善工作提供基础参考。*
