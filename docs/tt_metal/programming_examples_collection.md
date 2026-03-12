# Tenstorrent TT-Metal 官方编程示例收集

**收集日期**: 2026-03-12
**文档版本**: 基于 TT-Metal 最新版本 (v0.59.0+)

---

## 目录

1. [概述](#概述)
2. [示例分类总览](#示例分类总览)
3. [基础示例](#基础示例)
4. [Matmul 相关示例](#matmul-相关示例)
5. [Element-wise 操作示例](#element-wise-操作示例)
6. [自定义 SFPI 示例](#自定义-sfpi-示例)
7. [数据传输与优化示例](#数据传输与优化示例)
8. [构建与运行指南](#构建与运行指南)
9. [关键 API 参考](#关键-api-参考)
10. [参考资源](#参考资源)

---

## 概述

TT-Metal 编程示例位于 `tt_metal/programming_examples/` 目录下，提供从基础到高级的渐进式学习路径。这些示例展示了如何使用 TT-Metalium API 编写在 Tenstorrent Tensix 核心上运行的内核程序。

### 官方文档地址
- 主文档: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/
- 编程示例: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html
- GitHub 仓库: https://github.com/tenstorrent/tt-metal

### 学习路径
```
DRAM Loopback → Eltwise Binary → Eltwise SFPU → Matmul Single Core → Matmul Multi Core → Matmul Multi Core Optimized → Custom SFPI
```

---

## 示例分类总览

| 类别 | 示例数量 | 示例名称 |
|------|---------|---------|
| 基础示例 | 1 | DRAM Loopback |
| Element-wise 操作 | 2 | Eltwise Binary, Eltwise SFPU |
| Matmul 相关 | 4 | Matmul Single Core, Matmul Multi Core, Matmul Multi Core Reuse, Matmul Multi Core Reuse Multicast |
| 自定义 SFPI | 2 | Custom SFPI Add, Custom SFPI Smoothstep |
| **总计** | **9** | |

---

## 基础示例

### 1. DRAM Loopback

**路径**: `tt_metal/programming_examples/loopback/`

**功能描述**:
最基础的示例，演示数据从 DRAM → L1 (SRAM) → DRAM 的完整循环。用于学习设备初始化、缓冲区管理和基本数据移动内核。

**关键概念**:
- 设备初始化 (`CreateDevice`)
- 命令队列 (`CommandQueue`)
- DRAM 和 L1 缓冲区创建
- 数据移动内核基础
- 运行时参数设置

**关键 API 使用**:
```cpp
// 创建设备
Device* device = CreateDevice(device_id);

// 创建缓冲区
Buffer input_buffer = CreateBuffer(InterleavedBufferConfig{...});
Buffer output_buffer = CreateBuffer(InterleavedBufferConfig{...});

// 创建数据移动内核
KernelHandle kernel_id = CreateKernel(
    program,
    "kernels/dataflow/loopback.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
);

// 设置运行时参数
SetRuntimeArgs(program, kernel_id, core, {input_buffer->address(), output_buffer->address(), ...});

// 执行程序
EnqueueProgram(device->command_queue(), program, false);
Finish(device->command_queue());
```

**运行步骤**:
```bash
export TT_METAL_HOME=<path/to/tt-metal>
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_loopback
```

**文件结构**:
```
loopback/
├── kernels/
│   └── dataflow/
│       └── loopback.cpp    # 数据移动内核
├── loopback.cpp            # 主程序
└── CMakeLists.txt
```

---

## Element-wise 操作示例

### 2. Eltwise Binary

**路径**: `tt_metal/programming_examples/eltwise_binary/`

**功能描述**:
演示使用 FPU (矩阵引擎) 执行逐元素加法操作。引入循环缓冲区 (Circular Buffer) 概念，展示数据移动内核与计算内核之间的数据传递。

**关键概念**:
- 循环缓冲区 (Circular Buffers)
- 数据移动内核 (Reader/Writer)
- 计算内核 (Compute Kernel)
- FPU (矩阵引擎) 使用
- 双缓冲区数据流

**关键 API 使用**:
```cpp
// 创建循环缓冲区
CircularBufferConfig cb_in0_config = CircularBufferConfig(cb_size, {{in0_cb_data_format, tt::CBIndex::c_0}});
CBHandle cb_in0 = CreateCircularBuffer(program, core, cb_in0_config);

// 创建计算内核 (在 Tensix 的 MATH 模块上运行)
KernelHandle compute_kernel_id = CreateKernel(
    program,
    "kernels/compute/eltwise_binary.cpp",
    core,
    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, ...}
);
```

**内核代码片段** (计算内核):
```cpp
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    init_tile(cb_in0);
    init_tile(cb_in1);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);  // FPU 加法
        tile_regs_commit();

        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, 1);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_release();
    }
}
```

**运行步骤**:
```bash
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_eltwise_binary
```

---

### 3. Eltwise SFPU

**路径**: `tt_metal/programming_examples/eltwise_sfpu/`

**功能描述**:
演示使用 SFPU (Special Function Processing Unit，特殊函数处理单元/向量引擎) 执行逐元素操作。SFPU 用于向量运算，与 FPU (矩阵引擎) 形成互补。

**关键概念**:
- SFPU (向量引擎) 使用
- 向量化的数学运算 (exp, sqrt, etc.)
- 与 FPU 的区别

**关键 API 使用**:
```cpp
// SFPU 计算内核
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    init_sfpu(cb_in0, cb_out0);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);

        tile_regs_acquire();
        copy_tile(cb_in0, 0, 0);
        exp_tile(0);  // SFPU exp 操作
        tile_regs_commit();

        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, 1);

        cb_pop_front(cb_in0, 1);
        tile_regs_release();
    }
}
```

**运行步骤**:
```bash
./build/programming_examples/metal_example_eltwise_sfpu
```

---

## Matmul 相关示例

### 4. Matmul Single Core

**路径**: `tt_metal/programming_examples/matmul/matmul_single_core/`

**功能描述**:
在单个 Tensix 核心上使用 FPU (矩阵引擎) 执行矩阵乘法。演示分块 (tilized) 矩阵运算、流水线数据移动和计算重叠。

**关键概念**:
- 32x32 分块矩阵运算
- 流水线数据移动
- Reader/Compute/Writer 三阶段流水线
- 循环缓冲区协调

**文件结构**:
```
matmul/matmul_single_core/
├── kernels/
│   ├── dataflow/
│   │   ├── reader_single_core_mm.cpp   # 读取 A 和 B 矩阵
│   │   └── writer_single_core_mm.cpp   # 写入结果矩阵
│   └── compute/
│       └── mm.cpp                       # 矩阵乘法计算
├── matmul_single_core.cpp
└── CMakeLists.txt
```

**关键 API 使用**:
```cpp
// 创建 Reader 内核 (RISCV_1)
auto reader_id = CreateKernel(
    program,
    "kernels/dataflow/reader_single_core_mm.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
);

// 创建 Writer 内核 (RISCV_0)
auto writer_id = CreateKernel(
    program,
    "kernels/dataflow/writer_single_core_mm.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
);

// 创建计算内核
auto compute_id = CreateKernel(
    program,
    "kernels/compute/mm.cpp",
    core,
    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, ...}
);
```

**计算内核关键代码**:
```cpp
void MAIN {
    // ... 初始化代码
    mm_init(cb_in0, cb_in1, cb_out0);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                matmul_tiles(cb_in0, cb_in1, 0, 0, 0, kt == 0);

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            cb_reserve_back(cb_out0, 1);
            pack_tile(0, cb_out0);
            cb_push_back(cb_out0, 1);
            tile_regs_release();
        }
    }
}
```

**运行步骤**:
```bash
./build/programming_examples/metal_example_matmul_single_core
```

---

### 5. Matmul Multi Core

**路径**: `tt_metal/programming_examples/matmul/matmul_multi_core/`

**功能描述**:
将矩阵乘法工作负载分布到多个 Tensix 核心上并行执行。演示 SPMD (单程序多数据) 并行化、工作分区和运行时参数配置。

**关键概念**:
- 多核心网格计算 (`compute_with_storage_grid_size()`)
- 工作分区策略
- 每个核心的运行时参数
- 核心坐标 (`CoreCoord`)

**关键 API 使用**:
```cpp
// 获取设备计算网格大小
auto grid_size = device->compute_with_storage_grid_size();
uint32_t num_cores_x = grid_size.x;
uint32_t num_cores_y = grid_size.y;

// 计算每个核心处理的 tile 数量
uint32_t num_output_tiles_per_core = ...;

// 为每个核心设置运行时参数
for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
    CoreCoord core = {core_idx % num_cores_x, core_idx / num_cores_x};

    SetRuntimeArgs(program, reader_id, core, {
        src0_addr,
        src1_addr,
        Mt, Kt, Nt,
        Mt * Kt,
        Kt * Nt,
        output_tile_start_id,  // 每个核心不同
        num_output_tiles_per_core  // 每个核心不同
    });

    SetRuntimeArgs(program, writer_id, core, {
        dst_addr,
        Mt, Nt,
        Kt * Nt,
        output_tile_start_id,
        num_output_tiles_per_core
    });

    SetRuntimeArgs(program, compute_id, core, {
        num_output_tiles_per_core,
        Kt
    });
}
```

**运行步骤**:
```bash
./build/programming_examples/metal_example_matmul_multi_core
```

---

### 6. Matmul Multi Core Reuse

**路径**: `tt_metal/programming_examples/matmul/matmul_multi_core_reuse/`

**功能描述**:
优化的多核矩阵乘法，通过数据重用减少 DRAM 访问。实现细粒度的块大小控制和中间结果缓存。

**关键概念**:
- 数据重用策略
- 细粒度块大小控制
- 中间循环缓冲区配置
- 步长内核参数

**优化要点**:
- 在 L1 中缓存中间结果
- 减少 DRAM 带宽压力
- 优化数据局部性

**文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core_optimizations/data_reuse.html

---

### 7. Matmul Multi Core Reuse Multicast

**路径**: `tt_metal/programming_examples/matmul/matmul_multi_core_reuse_mcast/`

**功能描述**:
使用多播 (Multicast) 技术进一步优化矩阵乘法。通过 NOC 多播减少 DRAM 读取和 NOC 拥塞。

**关键概念**:
- 多播数据传输
- NOC 优化
- 信号量同步
- 核心网格配置

**关键 API 使用**:
```cpp
// 多播地址生成
uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start, uint32_t noc_y_start,
    uint32_t noc_x_end, uint32_t noc_y_end,
    uint32_t addr
);

// 多播写入
void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests
);

// 信号量多播
void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t num_dests
);
```

**文档**: https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/matmul_multi_core_optimizations/data_mcast.html

---

## 自定义 SFPI 示例

### 8. Vector Addition using SFPI

**路径**: `tt_metal/programming_examples/custom_sfpi_add/`

**功能描述**:
演示如何使用 SFPI (Special Function Processing Interface) 编写自定义向量运算。SFPI 允许直接操作 SFPU 的 SIMD 向量寄存器。

**关键概念**:
- SFPI 编程模型
- SIMD 向量操作 (`vFloat`)
- 目标寄存器 (`dst_reg[]`)
- LLK (Low Level Kernel) API
- 瓦片面 (tile face) 操作

**文件结构**:
```
custom_sfpi_add/
├── kernels/
│   ├── dataflow/
│   │   ├── reader_add.cpp      # 读取输入 tile
│   │   └── writer_add.cpp      # 写入结果 tile
│   └── compute/
│       └── tiles_add.cpp       # SFPI 自定义加法
├── custom_sfpi_add.cpp
└── CMakeLists.txt
```

**自定义 SFPI 实现**:
```cpp
// 低层函数：操作一个 tile 面 (256 个元素)
void add_tile_face(const uint32_t dst_index_in0,
                   const uint32_t dst_index_in1,
                   const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

    // 处理一个面的 256 个元素 (8 次 SIMD 操作)
    for (size_t i = 0; i < 8; i++) {
        vFloat a = dst_reg[in0_base_idx + i];
        vFloat b = dst_reg[in1_base_idx + i];
        dst_reg[out_base_idx + i] = a + b;
    }
}

// 高层 API 函数
void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        add_tile_face, idx_dst0, idx_dst1, idx_out0));
}
```

**运行步骤**:
```bash
./build/programming_examples/metal_example_custom_sfpi_add
```

**文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi_add.html

**警告**: Tenstorrent 不保证用户实现的 SFPI 函数的向后兼容性。调用底层 SFPI 函数的 API 可能会在没有通知的情况下更改。

---

### 9. Smoothstep using SFPI

**路径**: `tt_metal/programming_examples/custom_sfpi_smoothstep/`

**功能描述**:
更高级的 SFPI 示例，演示参数传递和向量谓词 (vector predicates) 的使用。实现 smoothstep 数学函数。

**关键概念**:
- 参数传递到 SFPI 内核
- 向量谓词 (`v_if`, `v_elseif`, `v_endif`)
- 条件向量执行

**向量谓词示例**:
```cpp
vFloat x = dst_reg[base_idx + i];
v_if(x < 0.0f) {
    dst_reg[out_idx] = 0.0f;
} v_elseif(x > 1.0f) {
    dst_reg[out_idx] = 1.0f;
} v_else {
    dst_reg[out_idx] = x * x * (3.0f - 2.0f * x);
} v_endif;
```

**文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi_smoothstep.html

---

## 数据传输与优化示例

### 共享组件

**路径**: `tt_metal/programming_examples/matmul_common/`

**内容**:
Matmul 示例共享的内核代码，包括数据移动和计算内核。

**文件结构**:
```
matmul_common/
└── kernels/
    ├── dataflow/
    │   ├── reader_bmm_tile_layout.cpp
    │   ├── reader_bmm_tile_layout_in0_sender.cpp
    │   ├── reader_bmm_tile_layout_in0_sender_in1_sender.cpp
    │   ├── writer_bmm_tile_layout.cpp
    │   └── writer_bmm_tile_layout_*.cpp
    └── compute/
        └── bmm.cpp
```

---

## 构建与运行指南

### 环境设置
```bash
export TT_METAL_HOME=<path/to/tt-metal>
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
```

### 构建所有示例
```bash
# 使用构建脚本
./build_metal.sh --build-programming-examples

# 或使用 CMake
cmake -DBUILD_PROGRAMMING_EXAMPLES=ON ...
```

### 运行示例
```bash
# 所有可执行文件位于
./build/programming_examples/metal_example_<example_name>

# 例如
./build/programming_examples/metal_example_loopback
./build/programming_examples/metal_example_eltwise_binary
./build/programming_examples/metal_example_matmul_single_core
./build/programming_examples/metal_example_matmul_multi_core
./build/programming_examples/metal_example_custom_sfpi_add
```

---

## 关键 API 参考

### 设备管理
| API | 描述 |
|-----|------|
| `CreateDevice(device_id)` | 创建设备实例 |
| `CloseDevice(device)` | 关闭设备 |
| `device->compute_with_storage_grid_size()` | 获取计算网格大小 |

### 缓冲区管理
| API | 描述 |
|-----|------|
| `CreateBuffer(InterleavedBufferConfig{...})` | 创建 DRAM 缓冲区 |
| `CreateCircularBuffer(program, core, config)` | 创建 L1 循环缓冲区 |
| `get_tile_size(cb_index)` | 获取 tile 大小 |

### 内核创建
| API | 描述 |
|-----|------|
| `CreateKernel(program, file, core, config)` | 创建内核 |
| `DataMovementConfig{processor, noc, compile_args}` | 数据移动内核配置 |
| `ComputeConfig{math_fidelity, ...}` | 计算内核配置 |

### NOC 数据移动 API
| API | 描述 |
|-----|------|
| `noc_async_read_tile(tile_id, accessor, dst_addr)` | 异步读取 tile |
| `noc_async_write_tile(tile_id, accessor, src_addr)` | 异步写入 tile |
| `noc_async_read_barrier()` | 等待读取完成 |
| `noc_async_write_barrier()` | 等待写入完成 |
| `get_noc_addr(x, y, addr)` | 获取 NOC 地址 |
| `noc_async_write_multicast(...)` | 多播写入 |

### 循环缓冲区 API
| API | 描述 |
|-----|------|
| `cb_reserve_back(cb_id, num_tiles)` | 预留写入空间 |
| `cb_push_back(cb_id, num_tiles)` | 推送数据 |
| `cb_wait_front(cb_id, num_tiles)` | 等待数据 |
| `cb_pop_front(cb_id, num_tiles)` | 弹出数据 |
| `get_write_ptr(cb_id)` | 获取写入地址 |
| `get_read_ptr(cb_id)` | 获取读取地址 |

### 计算 API
| API | 描述 |
|-----|------|
| `add_tiles(cb0, cb1, tile0, tile1, dst)` | Tile 加法 |
| `matmul_tiles(cb0, cb1, tile0, tile1, dst, init)` | Tile 矩阵乘法 |
| `exp_tile(dst)` | Tile 指数运算 |
| `copy_tile(cb, src, dst)` | 复制 tile |
| `pack_tile(src, cb)` | 打包 tile 到 CB |
| `tile_regs_acquire()` | 获取 tile 寄存器 |
| `tile_regs_commit()` | 提交 tile 寄存器 |
| `tile_regs_release()` | 释放 tile 寄存器 |

---

## 参考资源

### 官方文档
- [TT-Metalium 主页](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/)
- [编程示例索引](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)
- [DRAM Loopback](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/dram_loopback.html)
- [Matmul Single Core](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html)
- [Matmul Multi Core](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core.html)
- [Matmul Multi Core Optimized](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core_optimizations/data_reuse.html)
- [Eltwise SFPU](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/eltwise_sfpu.html)
- [Custom SFPI Add](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi_add.html)
- [Custom SFPI Smoothstep](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi_smoothstep.html)
- [数据移动 API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/data_movement/data_movement.html)

### GitHub 资源
- [主仓库](https://github.com/tenstorrent/tt-metal)
- [编程示例目录](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

### 实验室练习
- [Lab Exercises](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/index.html)

---

## 总结

本收集文档涵盖了 TT-Metal 官方提供的 9 个核心编程示例，按难度递进组织：

1. **入门**: DRAM Loopback
2. **基础**: Eltwise Binary, Eltwise SFPU
3. **进阶**: Matmul Single Core, Matmul Multi Core
4. **高级**: Matmul Multi Core Reuse, Matmul Multi Core Reuse Multicast
5. **专家**: Custom SFPI Add, Custom SFPI Smoothstep

每个示例都提供了完整的路径、功能描述、关键 API 使用和运行步骤，可作为学习和参考使用。
