# TT Metalium (TT-Metal) 编程框架完整文档总结

> **最后更新**: 2026-03-12
> **项目地址**: https://github.com/tenstorrent/tt-metal
> **官方文档**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/

---

## 目录

1. [架构概述](#1-架构概述)
2. [核心概念](#2-核心概念)
3. [Kernel 类型详解](#3-kernel-类型详解)
4. [API 参考手册](#4-api-参考手册)
5. [编程示例](#5-编程示例)
6. [构建与开发环境](#6-构建与开发环境)
7. [高级主题](#7-高级主题)
8. [性能优化指南](#8-性能优化指南)
9. [调试与剖析工具](#9-调试与剖析工具)

---

## 1. 架构概述

### 1.1 什么是 TT Metalium

TT Metalium 是 Tenstorrent 开源的底层软件开发套件（SDK），提供对 Tenstorrent Tensix 硬件的直接访问能力。它是整个软件栈的基础层，位于高级框架和硬件之间。

**软件栈层次结构**:
```
┌─────────────────────────────────────┐
│  TT-Forge / TT-MLIR (编译器层)       │
├─────────────────────────────────────┤
│  TT-NN (高级 ML 操作库)              │
├─────────────────────────────────────┤
│  TT-Metalium ⬅ (本层)               │
│  - 底层 Kernel 编程                  │
│  - 硬件抽象层                        │
├─────────────────────────────────────┤
│  TT-LLK (底层 Kernel 库)             │
├─────────────────────────────────────┤
│  Tenstorrent 硬件 (Tensix 核心)      │
└─────────────────────────────────────┘
```

### 1.2 支持的硬件代际

| 代际 | 制程 | Tensix 核心数 | 每核心 SRAM | 外部内存 | 状态 |
|------|------|--------------|------------|---------|------|
| **Grayskull** (第1代) | 12nm | 96-120 | 1 MB | 8 GB LPDDR4 | v0.55 后停止支持 |
| **Wormhole** (第2代) | 12nm | 72-128 | 1.5 MB | 12-24 GB GDDR6 | 活跃支持 |
| **Blackhole** (第3代) | 6nm | ~140 | 1.5 MB | 32 GB GDDR6 | 活跃支持 |

**Blackhole 创新特性**:
- 集成 24 个 RISC-V CPU 核心 (SiFive X280)
- 752 个小型 RISC-V 核心用于计算/数据移动
- 12× 400Gbps 以太网 (对比 Wormhole 的 16× 100Gbps)
- 可直接在芯片上运行 Linux 操作系统

### 1.3 核心设计原则

1. **裸机编程模型**: "编程核心，而非线程" - 显式控制，无隐藏抽象
2. **三 Kernel 流水线**: Reader、Compute、Writer Kernel 并发运行
3. **显式数据移动**: 无硬件缓存 - 每个字节移动都由程序员控制
4. **基于 Tile 的计算**: 原生支持 32×32 Tile 操作，自动格式转换

---

## 2. 核心概念

### 2.1 Tensix 核心架构

每个 Tensix 核心包含 **5 个 RISC-V "Baby Core"**:

| 核心 | 名称 | 用途 |
|------|------|------|
| **BRISC** | RISC-V 0 | 数据移动 (Reader) |
| **NCRISC** | RISC-V 1 | 数据移动 (Writer) |
| **TRISC0** | RISC-V 2 | 计算 (Unpack 解包) |
| **TRISC1** | RISC-V 3 | 计算 (Math 数学) |
| **TRISC2** | RISC-V 4 | 计算 (Pack 打包) |

### 2.2 Host-Device 编程模型

| 方面 | Host (CPU) | Device (Tensix) |
|------|-----------|-----------------|
| **处理器** | x86/ARM | Tensix 核心网格 |
| **内存** | Host DRAM | Device DRAM + L1 SRAM |
| **执行** | 编排协调 | 专用 Kernel |
| **地址空间** | 独立 | 独立 |

**关键原则**: Host 和 Device 有不同的地址空间。Host 指针 Device 不可见。

### 2.3 内存层次结构

```
┌─────────────────────────────────────────┐
│           HOST DRAM                     │
└─────────────────┬───────────────────────┘
                  │ PCIe 传输
                  ▼
┌─────────────────────────────────────────┐
│         DEVICE DRAM (GDDR6)             │
│    (12-32 GB, 512 GB/s 带宽)             │
└─────────────────┬───────────────────────┘
                  │ NoC DMA
                  ▼
┌─────────────────────────────────────────┐
│      L1 SRAM (片上, 每核心)              │
│    (~1.5 MB 每 Tensix 核心)              │
│    - Circular Buffer (CB)               │
│    - 不是缓存 - 显式管理                 │
└─────────────────────────────────────────┘
```

**关键设计**: 零硬件缓存。所有数据移动都是显式的。

### 2.4 Network-on-Chip (NoC)

NoC 使用**物理坐标**进行通信:

```cpp
// 地址格式: (x, y, local_addr) 元组
uint64_t noc_addr = get_noc_addr(x, y, addr_on_target_tile);
noc_async_read(noc_addr, ptr_l1_buffer, size);
noc_async_write(ptr_l1_buffer, noc_addr, size);
```

**关键规则**:
- RISC-V 核心只能直接访问私有内存和本地 L1 SRAM
- 访问 DRAM 或其他核心的 SRAM 需要 NoC DMA
- 栈变量不能作为 DMA 源/目标

---

## 3. Kernel 类型详解

TT Metalium 使用**三种协作 Kernel**，它们并发运行:

### 3.1 Data Movement Kernels (数据移动 Kernel)

#### Reader Kernel (BRISC - RISC-V 0)
- 通过 NoC 从 Device DRAM → L1 SRAM 进行 DMA 传输
- 使用 `noc_async_read()` 操作
- 通过 `cb_push_back()` 将数据推入 Circular Buffer

#### Writer Kernel (NCRISC - RISC-V 1)
- 通过 NoC 从 L1 SRAM → Device DRAM 进行 DMA 传输
- 使用 `noc_async_write()` 操作
- 通过 `cb_wait_front()` 等待输出 CB

**关键数据移动 API**:

| 函数 | 描述 |
|------|------|
| `noc_async_read(src, dst, size)` | 非阻塞 DRAM 读取 |
| `noc_async_write(src, dst, size)` | 非阻塞 DRAM 写入 |
| `noc_async_read_barrier()` | 等待读取完成 |
| `noc_async_write_barrier()` | 等待写入完成 |
| `noc_async_read_multicast` | 多播到多个核心 |
| `noc_semaphore_set/wait/inc` | NoC 信号量操作 |

### 3.2 Compute Kernels (计算 Kernel)

在 TRISC 核心（3 个线程）上运行，使用 **Math/Unpack/Pack 流水线**:

**关键计算 API**:

| 函数 | 描述 |
|------|------|
| `matmul_tiles(cb_a, cb_b, ...)` | 矩阵乘法: C += A @ B |
| `add_tiles`, `sub_tiles`, `mul_tiles` | 逐元素操作 |
| `exp_tile`, `log_tile`, `sqrt_tile` | 数学函数 |
| `relu_tile`, `gelu_tile`, `sigmoid_tile` | 激活函数 |
| `tilize` / `untilize` | 数据格式转换 |

**初始化函数**:
```cpp
mm_init(cb_in0, cb_in1, cb_out);           // 矩阵乘法
binary_op_init_common(cb_in0, cb_in1, cb_out);  // 逐元素操作
```

### 3.3 Ethernet Kernels (以太网 Kernel)

**用途**: 处理芯片间以太网通信

**使用场景**:
- 多芯片配置 (QuietBox, Galaxy)
- 扩展到 32+ 芯片
- 2D/3D 环形网格拓扑

**配置**: `EthernetConfig` 结构体用于设置以太网链路

**关键特性**:
- 以太网 RISC-V 核心 (erisc) 处理芯片间通信
- `tt_fabric` 提供集群级通信结构
- 透明多芯片执行

---

## 4. API 参考手册

### 4.1 Host API

#### 设备管理
```cpp
IDevice* CreateDevice(device_id);
CommandQueue& device->command_queue();
CloseDevice(device);
```

#### Buffer 管理
```cpp
// 创建 DRAM Buffer
InterleavedBufferConfig config{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::DRAM
};
auto buffer = CreateBuffer(config);

// 创建 L1 Buffer
config.buffer_type = BufferType::L1;
```

#### Kernel 创建
```cpp
// Data Movement Kernel
auto reader = CreateKernel(
    program,
    "reader.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = compile_time_args
    }
);

// Compute Kernel
auto compute = CreateKernel(
    program,
    "compute.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .compile_args = {Mt, Kt, Nt}
    }
);
```

#### 程序执行
```cpp
// 设置运行时参数
SetRuntimeArgs(program, kernel, core, {arg1, arg2, arg3});

// 入队程序
EnqueueProgram(cq, program, blocking);

// 同步
Finish(cq);
```

### 4.2 Circular Buffer (CB) API

#### Host 端创建
```cpp
CircularBufferConfig cb_config(
    num_pages * page_size,
    {{cb_index, data_format}}
).set_page_size(cb_index, page_size);

auto cb = CreateCircularBuffer(program, core, cb_config);
```

#### Device 端操作

| 函数 | 用途 |
|------|------|
| `cb_reserve_back(cb_id, num_tiles)` | 生产者: 预留空间 |
| `cb_push_back(cb_id, num_tiles)` | 生产者: 提交数据 |
| `cb_wait_front(cb_id, num_tiles)` | 消费者: 等待数据 |
| `cb_pop_front(cb_id, num_tiles)` | 消费者: 释放空间 |
| `get_write_ptr(cb_id)` | 获取 L1 写入地址 |
| `get_read_ptr(cb_id)` | 获取 L1 读取地址 |

### 4.3 Compute Kernel API

#### 矩阵运算
```cpp
#include "compute_kernel_api/matmul.h"

void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
}
```

#### 逐元素操作
```cpp
#include "compute_kernel_api/eltwise_binary.h"

void MAIN {
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
}
```

#### 激活函数
```cpp
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/sigmoid.h"

relu_tile_init();
relu_tile(cb_in, cb_out, 0, 0);
```

### 4.4 Dataflow API

#### DRAM 访问
```cpp
// 获取 DRAM 地址
uint64_t dram_addr = get_dram_noc_addr(tile_id, dram_buffer, bank_base_address);

// 异步读取
noc_async_read(dram_addr, l1_buffer, size);
noc_async_read_barrier();

// 异步写入
noc_async_write(l1_buffer, dram_addr, size);
noc_async_write_barrier();
```

#### 核心间通信
```cpp
// 获取其他核心的 L1 地址
uint64_t noc_addr = get_noc_addr(x, y, local_addr);

// 多播
noc_async_write_multicast(
    src_addr,
    dst_noc_addr_multicast,
    size,
    num_dests
);
```

---

## 5. 编程示例

### 5.1 DRAM Loopback (Hello World)

**目的**: 通过 L1 SRAM 将一个 DRAM Buffer 复制到另一个

**Kernel 代码** (`loopback_dram_copy.cpp`):
```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t l1_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_src = get_arg_val<uint32_t>(1);
    uint32_t dram_dst = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    for(uint32_t i = 0; i < num_tiles; i++) {
        // 读取 tile 到 L1
        noc_async_read_tile(i, in0, l1_addr);
        noc_async_read_barrier();

        // 写入到 DRAM
        noc_async_write_tile(i, out0, l1_addr);
        noc_async_write_barrier();
    }
}
```

### 5.2 单核矩阵乘法

**架构**:
- **Reader Kernel**: 从 DRAM 读取 tiles 到 CBs
- **Compute Kernel**: 使用 FPU 执行 matmul
- **Writer Kernel**: 将结果写回 DRAM

**Reader Kernel**:
```cpp
void kernel_main() {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(mt * Kt + kt, in0, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }
}
```

**Compute Kernel**:
```cpp
#include "compute_kernel_api/matmul.h"

void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
```

### 5.3 多核矩阵乘法

**特性**:
- 将工作分发到多个 Tensix 核心
- 每个核心计算子矩阵
- 使用数据分片实现并行化

**Host 代码**:
```cpp
// 定义核心网格
CoreRange cores({0, 0}, {7, 7});  // 8x8 核心网格

// 计算每个核心的工作负载
uint32_t per_core_M = M / 8;
uint32_t per_core_N = N / 8;

// 为每个核心设置运行时参数
for (uint32_t y = 0; y < 8; y++) {
    for (uint32_t x = 0; x < 8; x++) {
        CoreCoord core(x, y);
        SetRuntimeArgs(program, reader, core, {
            per_core_M, per_core_K, per_core_N,
            y * per_core_M,  // 输入 A 的起始行
            x * per_core_N   // 输入 B 的起始列
        });
    }
}
```

### 5.4 带激活函数的逐元素操作

```cpp
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/relu.h"

void MAIN {
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    relu_tile_init();

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        relu_tile(0, 0);  // 对结果应用 ReLU
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
```

---

## 6. 构建与开发环境

### 6.1 系统依赖

```bash
# 使用安装脚本
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/main/install_dependencies.sh
sudo ./install_dependencies.sh

# 手动安装包
sudo apt install cmake=3.16.3-1ubuntu1 pandoc libtbb-dev libcapstone-dev pkg-config ninja-build
```

### 6.2 硬件驱动

- **TT-KMD**: 内核驱动 (DKMS)
- **TT-Flash**: 固件
- **TT-SMI**: 系统管理接口
- **Huge Pages**: 大页内存设置

### 6.3 构建选项

#### 选项 A: 构建脚本 (推荐)
```bash
./build_metal.sh
./build_metal.sh --build-programming-examples
./build_metal.sh --enable-tracy  # 启用性能分析
```

#### 选项 B: 手动 CMake
```bash
export ARCH_NAME=wormhole_b0  # 或 grayskull, blackhole
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebugInfo
ninja install
```

### 6.4 环境变量

| 变量 | 描述 |
|------|------|
| `ARCH_NAME` | 目标架构: `grayskull`, `wormhole_b0`, `blackhole` |
| `TT_METAL_HOME` | 仓库根目录 |
| `PYTHONPATH` | Python API 路径 |
| `TT_METAL_RUNTIME_ROOT` | 运行时工件位置 |

### 6.5 Python 环境

```bash
./create_venv.sh
source python_env/bin/activate
```

### 6.6 编程示例路径

| 示例 | 路径 |
|------|------|
| DRAM Loopback | `tt_metal/programming_examples/loopback/` |
| 单核 Matmul | `tt_metal/programming_examples/matmul_single_core/` |
| 多核 Matmul | `tt_metal/programming_examples/matmul_multi_core/` |
| 多芯片 | `tt_metal/programming_examples/matmul_multichip/` |

---

## 7. 高级主题

### 7.1 多芯片编程 (Galaxy)

**横向扩展支持**:
- **Galaxy 系统**: 32+ 芯片的 2D/3D 环形拓扑
- **基于以太网**: Blackhole 提供 1 TB/s 横向扩展带宽
- **透明执行**: 相同 Kernel 代码跨芯片运行

**多芯片 API**:
```cpp
// 创建设备网格
DeviceMesh device_mesh(DeviceGrid{2, 4});  // 2x4 网格

// 设置以太网 Kernel
EthernetConfig eth_config;
eth_config.eth_mode = Eth::SENDER;
eth_config.dataflow = EthDataflow::DISABLED;

auto eth_kernel = CreateKernel(
    program,
    "eth_kernel.cpp",
    eth_core,
    eth_config
);
```

**集合通信** (CCL - Collective Communication Library):
```cpp
// AllGather
all_gather(input_tensor, output_tensor, dim, device_mesh);

// ReduceScatter
reduce_scatter(input_tensor, output_tensor, dim, reduce_op, device_mesh);
```

### 7.2 Tile 格式与数据布局

**Tile 布局**:
- 默认 Tile 大小: 32×32 元素
- 数据格式: Float16_b, Bfloat16, Float32, Int32 等
- 内存布局: 行优先，交织存储

**格式转换**:
```cpp
// 将线性数据转换为 Tile 格式
tilize_init_short(cb_in);
tile_regs_acquire();
tilize_block(cb_in, num_tiles, cb_out);
tile_regs_commit();

// 将 Tile 格式转换回线性
untilize_init_short(cb_in);
untilize_block(cb_in, num_tiles, cb_out);
```

### 7.3 Math Fidelity (数学精度)

```cpp
enum class MathFidelity {
    LoFi,      // 最低精度，最高性能
    HiFi2,     // 中等精度
    HiFi3,     // 较高精度
    HiFi4      // 最高精度，较低性能
};

// 在 ComputeConfig 中设置
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    // ...
}
```

---

## 8. 性能优化指南

### 8.1 关键优化技术

1. **双缓冲 (Double Buffering)**: 重叠计算和数据移动
2. **数据复用**: 将常用数据保留在 L1 SRAM
3. **多播**: 高效广播数据到多个核心
4. **分片张量**: 跨核心分发数据实现并行

### 8.2 优化层次

| 层次 | 优化目标 |
|------|---------|
| 单核心 | 优化 CB 大小和 Tile 流 |
| 多核心 | 平衡工作负载分发 |
| 多芯片 | 最小化芯片间通信 |

### 8.3 CB 大小计算

```cpp
// 计算最佳 CB 大小
// 公式: cb_size = num_tiles * tile_size
// 其中 tile_size = 32 * 32 * element_size

// 示例: Float16 (2字节) 的 4-tile 双缓冲
uint32_t tile_size = 32 * 32 * 2;  // 2048 bytes
uint32_t cb_size = 4 * tile_size;   // 8192 bytes (8KB)
```

### 8.4 内存带宽优化

```cpp
// 批量读取以减少 NoC 开销
const uint32_t batch_size = 8;
for (uint32_t b = 0; b < num_tiles; b += batch_size) {
    cb_reserve_back(cb_id, batch_size);
    uint32_t l1_addr = get_write_ptr(cb_id);

    // 批量读取
    for (uint32_t i = 0; i < batch_size; i++) {
        noc_async_read_tile(b + i, dram_buffer, l1_addr + i * tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, batch_size);
}
```

---

## 9. 调试与剖析工具

### 9.1 工具概览

| 工具 | 用途 | 使用方法 |
|------|------|---------|
| **Tracy Profiler** | Host + Device 性能分析 | `TT_METAL_TRACY=1` |
| **Device Profiler** | Kernel 级计时 | `TT_METAL_DEVICE_PROFILER=1` |
| **Watcher** | 调试监控 | `TT_METAL_WATCHER=1` |
| **DPRINT** | Kernel printf 调试 | `TT_METAL_DPRINT_CORES=all` |

**重要**: 这些工具互斥（不能同时运行）

### 9.2 Tracy Profiler

```bash
# 使用 Tracy 构建
./build_metal.sh --enable-tracy

# 运行并分析
python -m tracy script.py

# 启动 Tracy GUI
tracy-profiler
```

**Kernel 标记**:
```cpp
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedN("DataMovementKernel");

    DeviceZoneScopedN("ReadPhase");
    // 读取代码

    DeviceZoneScopedN("WritePhase");
    // 写入代码
}
```

### 9.3 Device Profiler

```cpp
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedN("ComputeKernel");

    // 记录特定区域
    {
        DeviceZoneScopedN("MatmulLoop");
        for (...) {
            // matmul 代码
        }
    }
}
```

### 9.4 Watcher 调试

```bash
# 启用 Watcher
export TT_METAL_WATCHER=1

# 运行程序
./your_program

# Watcher 输出将显示:
# - Kernel 状态
# - NOC 状态
# - 断言和错误
```

### 9.5 DPRINT 调试

```cpp
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from kernel!" << ENDL();
    DPRINT << "Value: " << my_variable << ENDL();

    // 条件打印
    DPRINT << SETW(32) << "Formatted" << ENDL();
}
```

```bash
# 启用 DPRINT
export TT_METAL_DPRINT_CORES=all  # 或特定核心，如 "1,1"

# 查看输出
# 输出将显示在 stderr
```

### 9.6 调试技巧

1. **验证 CB 状态**:
```cpp
// 检查 CB 是否有足够数据
ASSERT(cb_num_tiles_available(cb_id) >= required_tiles);
```

2. **NOC 地址验证**:
```cpp
// 验证 NOC 地址
ASSERT(noc_addr != 0);
ASSERT(noc_addr < NOC_MAX_ADDR);
```

3. **Tile 数量检查**:
```cpp
// 确保 Tile 数量合理
ASSERT(num_tiles > 0 && num_tiles <= MAX_TILES);
```

---

## 10. 参考资源

### 10.1 官方文档链接

| 资源 | URL |
|------|-----|
| **官方文档** | https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/ |
| **入门指南** | https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html |
| **GitHub 仓库** | https://github.com/tenstorrent/tt-metal |
| **Metalium 指南** | https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md |
| **API 参考** | https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/ |
| **编程示例** | https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html |

### 10.2 头文件参考

| 头文件 | 用途 |
|--------|------|
| `tt_metal/host_api.hpp` | Host 端设备管理 |
| `tt_metal/detail/tt_metal.hpp` | 底层设备控制 |
| `compute_kernel_api/matmul.h` | 矩阵乘法 API |
| `compute_kernel_api/eltwise_binary.h` | 二元运算 API |
| `compute_kernel_api/eltwise_unary/*.h` | 一元运算 API |
| `dataflow_api.h` | 数据移动 API |
| `debug/dprint.h` | 调试打印 |
| `tools/profiler/kernel_profiler.hpp` | 性能分析 |

### 10.3 社区与支持

- **GitHub Issues**: https://github.com/tenstorrent/tt-metal/issues
- **Discord**: Tenstorrent 社区
- **论坛**: https://github.com/tenstorrent/tt-metal/discussions

---

## 11. 常见模式与最佳实践

### 11.1 Kernel 设计模式

**模式 1: Pipeline Pattern (流水线模式)**
```
Reader → [CB0] → Compute → [CB1] → Writer
         (L1)              (L1)
```

**模式 2: Producer-Consumer (生产者-消费者)**
```cpp
// Producer (Reader)
cb_reserve_back(cb_id, num_tiles);
// ... 写入数据 ...
cb_push_back(cb_id, num_tiles);

// Consumer (Compute)
cb_wait_front(cb_id, num_tiles);
// ... 读取数据 ...
cb_pop_front(cb_id, num_tiles);
```

**模式 3: Double Buffering (双缓冲)**
```cpp
// 使用两个 CB 交替进行读写
while (tiles_remaining) {
    // 在当前 buffer 计算的同时，读取下一个到另一个 buffer
    cb_reserve_back(cb_ping, num_tiles);
    // 读取到 ping
    cb_push_back(cb_ping, num_tiles);

    cb_wait_front(cb_pong, num_tiles);
    // 处理 pong
    cb_pop_front(cb_pong, num_tiles);

    // 交换 ping/pong
    std::swap(cb_ping, cb_pong);
}
```

### 11.2 错误处理

```cpp
void kernel_main() {
    // 验证参数
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    if (num_tiles == 0) {
        // 错误: 无效参数
        return;
    }

    // 验证 buffer 地址
    uint32_t l1_addr = get_write_ptr(cb_id);
    if (l1_addr == 0) {
        // 错误: 无效地址
        return;
    }

    // 主循环
    for (uint32_t i = 0; i < num_tiles; i++) {
        // 带超时的等待
        uint32_t timeout = 1000000;
        while (cb_num_tiles_available(cb_id) == 0 && timeout-- > 0);
        if (timeout == 0) {
            // 超时错误
            return;
        }
        // ...
    }
}
```

### 11.3 性能检查清单

- [ ] CB 大小是否针对 L1 缓存优化?
- [ ] 是否使用了双缓冲重叠计算和通信?
- [ ] NoC 操作是否批量执行?
- [ ] 是否最小化了 CB 的 push/pop 操作?
- [ ] 多核工作负载是否平衡?
- [ ] 是否使用了多播减少冗余传输?
- [ ] Math Fidelity 是否针对精度要求优化?

---

*本文档总结了 Tenstorrent TT Metalium 框架的核心概念、API 和最佳实践。如需最新信息，请参考官方文档仓库。*
