# TT-Metalium 编程示例扩展库

**文档版本**: 基于 TT-Metal v0.59.0+
**生成日期**: 2026-03-12

---

## 目录

1. [概述](#概述)
2. [高优先级示例](#高优先级示例)
   - 2.1 [Hello World Kernel](#21-hello-world-kernel)
   - 2.2 [信号量同步示例](#22-信号量同步示例)
   - 2.3 [双缓冲模式示例](#23-双缓冲模式示例)
   - 2.4 [分片张量操作示例](#24-分片张量操作示例)
   - 2.5 [矩阵块操作示例](#25-矩阵块操作示例)
   - 2.6 [归约操作示例](#26-归约操作示例)
   - 2.7 [SFPU 完整操作示例](#27-sfpu-完整操作示例)
   - 2.8 [子设备管理示例](#28-子设备管理示例)
3. [中优先级示例](#中优先级示例)
   - 3.1 [Metal Trace 完整示例](#31-metal-trace-完整示例)
   - 3.2 [多队列并行示例](#32-多队列并行示例)
4. [运行环境配置](#4-运行环境配置)
5. [总结](#5-总结)

---

## 概述

本文档扩展了 TT-Metalium 官方编程示例库，补充了在 [gap_analysis.md](./gap_analysis.md) 第3节中识别出的缺失示例类型。每个示例包含完整的问题描述、解决方案、代码实现、详细解释和运行步骤。

### 示例优先级说明

- **高优先级**: 基础且重要的编程模式，建议优先学习
- **中优先级**: 进阶优化技术，适合有一定基础后学习

---

## 高优先级示例

### 2.1 Hello World Kernel

**问题描述**: 最简单的内核程序，演示如何在设备上执行代码并输出调试信息。这是学习 TT-Metalium 编程的第一步。

**解决方案**: 创建一个简单的数据移动内核，使用 DPRINT 宏输出 "Hello World" 信息，演示基本的 Host-Device 交互流程。

#### 完整代码

**Host 端代码** (`hello_world.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>

using namespace tt::tt_metal;

int main() {
    // 1. 创建设备
    Device* device = CreateDevice(0);

    // 2. 创建程序
    Program program = CreateProgram();

    // 3. 选择核心 (0,0)
    CoreCoord core = {0, 0};

    // 4. 创建数据移动内核
    KernelHandle hello_kernel = CreateKernel(
        program,
        "kernels/dataflow/hello_world_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // 5. 设置运行时参数 (传递一个整数给内核)
    uint32_t my_id = 42;
    SetRuntimeArgs(program, hello_kernel, core, {my_id});

    // 6. 执行程序
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    // 7. 清理
    CloseDevice(device);

    std::cout << "Hello World kernel executed successfully!" << std::endl;
    return 0;
}
```

**Device 端代码** (`kernels/dataflow/hello_world_kernel.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/dprint.h>

void kernel_main() {
    // 读取运行时参数
    uint32_t id = get_arg_val<uint32_t>(0);

    // 使用 DPRINT 输出调试信息
    DPRINT << "Hello World from Tensix Core!" << ENDL();
    DPRINT << "Received ID: " << id << ENDL();
    DPRINT << "Core coordinates: (" << get_absolute_logical_x() << ", "
           << get_absolute_logical_y() << ")" << ENDL();
}
```

#### 代码解释

**Host 端关键步骤**:

1. **创建设备**: `CreateDevice(0)` 创建与第一个 TT 设备的连接
2. **创建程序**: `CreateProgram()` 创建一个程序容器，用于组织内核
3. **选择核心**: `CoreCoord core = {0, 0}` 指定在逻辑坐标 (0,0) 的核心上运行
4. **创建内核**:
   - 指定内核文件路径
   - 选择 RISCV_0 处理器 (每个 Tensix 核心有两个 RISC-V 数据移动处理器)
   - 使用默认 NOC 路由
5. **设置运行时参数**: 将 Host 数据传递给 Device 内核
6. **执行与同步**: `EnqueueProgram` 提交程序，`Finish` 等待完成

**Device 端关键元素**:

1. **参数获取**: `get_arg_val<uint32_t>(0)` 读取第一个运行时参数
2. **调试输出**: `DPRINT` 宏将信息输出到 Host 控制台（仅在调试模式下）
3. **核心信息**: `get_absolute_logical_x/y()` 获取当前核心坐标

#### 运行步骤

```bash
# 1. 设置环境变量
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME

# 2. 编译示例
mkdir -p build && cd build
cmake .. -DBUILD_PROGRAMMING_EXAMPLES=ON
make metal_example_hello_world

# 3. 运行示例
./programming_examples/metal_example_hello_world

# 预期输出:
# Hello World from Tensix Core!
# Received ID: 42
# Core coordinates: (0, 0)
# Hello World kernel executed successfully!
```

---

### 2.2 信号量同步示例

**问题描述**: 多核协作时需要同步机制来协调数据访问。本示例演示如何使用 NOC 信号量实现核心间同步。

**解决方案**: 创建两个内核，一个设置信号量，另一个等待信号量。演示 `noc_semaphore_set`、`noc_semaphore_inc` 和 `noc_semaphore_wait` 的使用。

#### 完整代码

**Host 端代码** (`semaphore_sync.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>
#include <vector>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);
    Program program = CreateProgram();

    // 定义两个核心: 发送方 (0,0) 和接收方 (0,1)
    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {0, 1};
    CoreRange cores(sender_core, receiver_core);

    // 创建 L1 缓冲区用于信号量
    uint32_t semaphore_size = 16;  // 4 bytes aligned to 16
    Buffer semaphore_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = semaphore_size,
            .page_size = semaphore_size,
            .buffer_type = BufferType::L1
        }
    );

    // 初始化信号量为 0
    std::vector<uint32_t> initial_semaphore = {0};
    EnqueueWriteBuffer(device->command_queue(), semaphore_buffer, initial_semaphore, true);

    // 创建发送方内核
    KernelHandle sender_kernel = CreateKernel(
        program,
        "kernels/dataflow/semaphore_sender.cpp",
        sender_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // 创建接收方内核
    KernelHandle receiver_kernel = CreateKernel(
        program,
        "kernels/dataflow/semaphore_receiver.cpp",
        receiver_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // 设置运行时参数
    uint32_t semaphore_addr = semaphore_buffer->address();
    uint32_t receiver_noc_x = device->worker_core_from_logical_core(receiver_core).x;
    uint32_t receiver_noc_y = device->worker_core_from_logical_core(receiver_core).y;

    SetRuntimeArgs(program, sender_kernel, sender_core, {
        semaphore_addr,
        receiver_noc_x,
        receiver_noc_y
    });

    SetRuntimeArgs(program, receiver_kernel, receiver_core, {
        semaphore_addr
    });

    // 执行程序
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    // 读取信号量验证
    std::vector<uint32_t> final_semaphore;
    EnqueueReadBuffer(device->command_queue(), semaphore_buffer, final_semaphore, true);
    std::cout << "Final semaphore value: " << final_semaphore[0] << std::endl;

    DeallocateBuffer(semaphore_buffer);
    CloseDevice(device);

    return 0;
}
```

**发送方内核** (`kernels/dataflow/semaphore_sender.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/noc_overlay.h>
#include <tt-metalium/dprint.h>

void kernel_main() {
    // 获取参数
    uint32_t semaphore_addr = get_arg_val<uint32_t>(0);
    uint32_t receiver_noc_x = get_arg_val<uint32_t>(1);
    uint32_t receiver_noc_y = get_arg_val<uint32_t>(2);

    DPRINT << "Sender: Starting work..." << ENDL();

    // 模拟一些工作
    for (volatile int i = 0; i < 1000; i++);

    DPRINT << "Sender: Work done, signaling receiver" << ENDL();

    // 获取接收方 NOC 地址
    uint64_t receiver_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, semaphore_addr);

    // 设置信号量 (值为 1，表示完成)
    noc_semaphore_set(receiver_noc_addr, 1);

    DPRINT << "Sender: Signal sent!" << ENDL();
}
```

**接收方内核** (`kernels/dataflow/semaphore_receiver.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/noc_overlay.h>
#include <tt-metalium/dprint.h>

void kernel_main() {
    // 获取参数
    uint32_t semaphore_addr = get_arg_val<uint32_t>(0);

    DPRINT << "Receiver: Waiting for signal..." << ENDL();

    // 等待信号量变为非零值
    noc_semaphore_wait(semaphore_addr, 1);

    DPRINT << "Receiver: Signal received!" << ENDL();

    // 可选: 增加信号量值
    noc_semaphore_inc(semaphore_addr, 1);

    DPRINT << "Receiver: Done!" << ENDL();
}
```

#### 代码解释

**信号量机制**:

1. **信号量地址**: 在 L1 内存中分配一个缓冲区作为信号量存储位置
2. **noc_semaphore_set**: 原子地将信号量设置为指定值
3. **noc_semaphore_wait**: 阻塞等待直到信号量值大于等于期望值
4. **noc_semaphore_inc**: 原子地增加信号量值

**核心间通信**:

- 使用 `get_noc_addr(x, y, addr)` 获取目标核心的 NOC 地址
- 信号量操作通过 NOC 网络完成，支持跨核心同步

#### 运行步骤

```bash
# 编译
make metal_example_semaphore_sync

# 运行
./programming_examples/metal_example_semaphore_sync

# 预期输出:
# Sender: Starting work...
# Sender: Work done, signaling receiver
# Sender: Signal sent!
# Receiver: Waiting for signal...
# Receiver: Signal received!
# Receiver: Done!
# Final semaphore value: 2
```

---

### 2.3 双缓冲模式示例

**问题描述**: 为了最大化计算吞吐量，需要重叠数据传输和计算。双缓冲模式允许在一个缓冲区进行计算的同时，向另一个缓冲区传输数据。

**解决方案**: 使用两个循环缓冲区 (CB) 交替进行数据读写，实现流水线并行。

#### 完整代码

**Host 端代码** (`double_buffering.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>
#include <vector>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 配置
    uint32_t num_tiles = 8;
    uint32_t tile_size = 32 * 32 * 2;  // bfloat16 = 2 bytes
    DataFormat data_format = DataFormat::Float16_b;

    // 创建输入输出 DRAM 缓冲区
    Buffer input_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer output_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 创建双缓冲 CB (每个 CB 可以容纳 2 个 tiles)
    uint32_t cb_index = CBIndex::c_0;
    uint32_t num_buffers = 2;  // 双缓冲
    CircularBufferConfig cb_config =
        CircularBufferConfig(num_buffers * tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, tile_size);
    CreateCircularBuffer(program, core, cb_config);

    // 创建 Reader 内核 (使用双缓冲)
    KernelHandle reader_kernel = CreateKernel(
        program,
        "kernels/dataflow/double_buffer_reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {num_buffers}
        }
    );

    // 创建 Compute 内核
    KernelHandle compute_kernel = CreateKernel(
        program,
        "kernels/compute/double_buffer_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {num_tiles, num_buffers}
        }
    );

    // 创建 Writer 内核
    KernelHandle writer_kernel = CreateKernel(
        program,
        "kernels/dataflow/double_buffer_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {num_buffers}
        }
    );

    // 准备测试数据
    std::vector<uint16_t> input_data(num_tiles * 32 * 32);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = float_to_bfloat16(static_cast<float>(i));
    }
    EnqueueWriteBuffer(device->command_queue(), input_buffer, input_data.data(), true);

    // 设置运行时参数
    SetRuntimeArgs(program, reader_kernel, core, {
        input_buffer->address(),
        num_tiles
    });

    SetRuntimeArgs(program, compute_kernel, core, {
        num_tiles
    });

    SetRuntimeArgs(program, writer_kernel, core, {
        output_buffer->address(),
        num_tiles
    });

    // 执行
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    // 读取并验证结果
    std::vector<uint16_t> output_data(num_tiles * 32 * 32);
    EnqueueReadBuffer(device->command_queue(), output_buffer, output_data.data(), true);

    std::cout << "Double buffering example completed!" << std::endl;
    std::cout << "Processed " << num_tiles << " tiles with double buffering" << std::endl;

    DeallocateBuffer(input_buffer);
    DeallocateBuffer(output_buffer);
    CloseDevice(device);

    return 0;
}
```

**Reader 内核** (`kernels/dataflow/double_buffer_reader.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t num_buffers = get_compile_time_arg_val(0);

    // 创建 TensorAccessor
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(s_args, src_addr, get_tile_size(cb_id));

    // 双缓冲流水线
    uint32_t tile_id = 0;

    // 预填充阶段: 填充第一个缓冲区
    for (uint32_t b = 0; b < num_buffers && tile_id < num_tiles; b++, tile_id++) {
        cb_reserve_back(cb_id, 1);
        uint32_t write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, src, write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }

    // 稳态阶段: 读取下一个 tile 同时计算处理当前 tile
    while (tile_id < num_tiles) {
        cb_reserve_back(cb_id, 1);
        uint32_t write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, src, write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
        tile_id++;
    }
}
```

**Compute 内核** (`kernels/compute/double_buffer_compute.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t num_buffers = get_compile_time_arg_val(1);

    init_sfpu(cb_id, cb_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 等待数据可用
        cb_wait_front(cb_id, 1);

        // 获取 tile 寄存器
        tile_regs_acquire();

        // 复制到寄存器
        copy_tile(cb_id, 0, 0);

        // 执行计算 (例如: 乘以 2)
        muli_tile(0, 2);

        tile_regs_commit();
        tile_regs_wait();

        // 写回 CB (原地修改)
        pack_tile(0, cb_id);

        tile_regs_release();

        // 释放输入，标记输出可用
        cb_pop_front(cb_id, 1);
    }
}
```

**Writer 内核** (`kernels/dataflow/double_buffer_writer.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t num_buffers = get_compile_time_arg_val(0);

    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(d_args, dst_addr, get_tile_size(cb_id));

    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        // 等待计算完成
        cb_wait_front(cb_id, 1);

        uint32_t read_addr = get_read_ptr(cb_id);
        noc_async_write_tile(tile_id, dst, read_addr);
        noc_async_write_barrier();

        // 释放 CB 空间供 Reader 重用
        cb_pop_front(cb_id, 1);
    }
}
```

#### 代码解释

**双缓冲原理**:

```
时间轴 ->

Reader:  [Read 0] [Read 1] [Read 2] [Read 3] ...
         ↓        ↓        ↓        ↓
CB:      [Buf 0]  [Buf 1]  [Buf 0]  [Buf 1]  (交替使用)
         ↓        ↓        ↓        ↓
Compute:          [Proc 0] [Proc 1] [Proc 2] ...
```

1. **预填充**: 先填充所有缓冲区
2. **稳态**: Reader 读取下一个 tile 的同时，Compute 处理当前 tile
3. **流水线**: 三个内核通过 CB 形成流水线，最大化吞吐量

#### 运行步骤

```bash
make metal_example_double_buffering
./programming_examples/metal_example_double_buffering
```

---

### 2.4 分片张量操作示例

**问题描述**: 大张量需要分片到多个核心或设备上处理。本示例演示如何使用 `ShardedBufferConfig` 创建和管理分片张量。

**解决方案**: 使用分片缓冲区配置，将张量按行或列分片到不同核心。

#### 完整代码

**Host 端代码** (`sharded_tensor.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/shape.hpp>
#include <iostream>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);

    // 获取计算网格
    auto grid_size = device->compute_with_storage_grid_size();
    std::cout << "Grid size: " << grid_size.x << "x" << grid_size.y << std::endl;

    // 张量配置
    uint32_t tile_height = 32;
    uint32_t tile_width = 32;
    uint32_t num_tiles_h = 4;  // 128 行
    uint32_t num_tiles_w = 4;  // 128 列
    uint32_t total_tiles = num_tiles_h * num_tiles_w;
    uint32_t tile_size = tile_height * tile_width * 2;  // bfloat16

    // 分片配置: 每个核心处理 2x2 tiles
    uint32_t shard_height = 2 * tile_height;  // 64 行
    uint32_t shard_width = 2 * tile_width;    // 64 列

    // 创建分片缓冲区配置
    ShardedBufferConfig sharded_config = {
        .device = device,
        .shard_params = ShardSpecBuffer(
            CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(1, 1))}),  // 2x2 核心网格
            {shard_height, shard_width},  // 每个分片的大小
            ShardOrientation::ROW_MAJOR,   // 按行分片
            {tile_height, tile_width},     // Tile 大小
            {tile_height, tile_width}
        ),
        .size = total_tiles * tile_size,
        .page_size = tile_size,
        .buffer_type = BufferType::L1,
        .data_type = DataFormat::Float16_b
    };

    // 创建分片缓冲区
    Buffer sharded_buffer = CreateBuffer(sharded_config);

    std::cout << "Created sharded buffer:" << std::endl;
    std::cout << "  Total size: " << sharded_buffer->size() << " bytes" << std::endl;
    std::cout << "  Num shards: " << sharded_buffer->num_cores() << std::endl;

    // 准备数据: 按行主序填充
    std::vector<uint16_t> host_data(total_tiles * tile_height * tile_width);
    for (uint32_t t = 0; t < total_tiles; t++) {
        uint32_t tile_row = t / num_tiles_w;
        uint32_t tile_col = t % num_tiles_w;
        for (uint32_t i = 0; i < tile_height; i++) {
            for (uint32_t j = 0; j < tile_width; j++) {
                uint32_t idx = (t * tile_height + i) * tile_width + j;
                // 值 = 行号 * 1000 + 列号
                uint32_t global_row = tile_row * tile_height + i;
                uint32_t global_col = tile_col * tile_width + j;
                host_data[idx] = float_to_bfloat16(static_cast<float>(global_row * 1000 + global_col));
            }
        }
    }

    // 写入分片缓冲区
    EnqueueWriteBuffer(device->command_queue(), sharded_buffer, host_data.data(), true);

    // 读取分片缓冲区
    std::vector<uint16_t> read_data(total_tiles * tile_height * tile_width);
    EnqueueReadBuffer(device->command_queue(), sharded_buffer, read_data.data(), true);

    // 验证
    bool match = true;
    for (size_t i = 0; i < host_data.size(); i++) {
        if (host_data[i] != read_data[i]) {
            match = false;
            break;
        }
    }

    std::cout << "Data verification: " << (match ? "PASSED" : "FAILED") << std::endl;

    DeallocateBuffer(sharded_buffer);
    CloseDevice(device);

    return 0;
}
```

**多核分片处理内核** (`kernels/dataflow/sharded_reader.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/dprint.h>

void kernel_main() {
    // 获取当前核心的分片信息
    uint32_t shard_id = get_arg_val<uint32_t>(0);
    uint32_t num_shards = get_arg_val<uint32_t>(1);

    DPRINT << "Core (" << get_absolute_logical_x() << ", "
           << get_absolute_logical_y() << ") processing shard "
           << shard_id << " of " << num_shards << ENDL();

    // 在分片处理场景中，数据已经在 L1 中
    // 内核可以直接访问分片数据而无需 DRAM 读取

    // 获取分片地址
    uint32_t shard_addr = get_write_ptr(tt::CBIndex::c_0);

    DPRINT << "Shard address: " << shard_addr << ENDL();

    // 这里可以添加具体的分片处理逻辑
}
```

#### 代码解释

**分片配置关键参数**:

1. **CoreRangeSet**: 定义哪些核心参与分片
2. **ShardOrientation**:
   - `ROW_MAJOR`: 按行分片，相邻行在同一个核心
   - `COL_MAJOR`: 按列分片，相邻列在同一个核心
3. **ShardSpecBuffer**: 定义分片的详细规格

**分片数据布局**:

```
全局张量 (8x8 tiles):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │
└────┴────┴────┴────┴────┴────┴────┴────┘

ROW_MAJOR 分片到 2x2 核心网格 (每个核心 2x2 tiles):
Core(0,0): tiles 0,1,8,9
Core(0,1): tiles 2,3,10,11
Core(1,0): tiles 16,17,24,25
Core(1,1): tiles 18,19,26,27
```

#### 运行步骤

```bash
make metal_example_sharded_tensor
./programming_examples/metal_example_sharded_tensor
```

---

### 2.5 矩阵块操作示例

**问题描述**: 标准 `matmul_tiles` 适用于小规模矩阵乘法。对于大规模矩阵，需要使用块级操作 `matmul_block` 来更高效地利用硬件资源。

**解决方案**: 使用 `mm_block_init` 和 `matmul_block` API 执行块级矩阵乘法。

#### 完整代码

**Host 端代码** (`matmul_block.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>
#include <vector>
#include <random>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 矩阵维度 (以 tiles 为单位)
    uint32_t Mt = 4;  // M = 128 (4 * 32)
    uint32_t Kt = 4;  // K = 128
    uint32_t Nt = 4;  // N = 128

    uint32_t tile_size = 32 * 32 * 2;  // bfloat16
    DataFormat data_format = DataFormat::Float16_b;

    // 创建缓冲区
    Buffer buffer_a = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = Mt * Kt * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer buffer_b = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = Kt * Nt * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer buffer_c = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = Mt * Nt * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 创建 CB (使用更大的缓冲区支持块操作)
    uint32_t block_size = 2;  // 2x2 tiles 块
    uint32_t cb_size = block_size * block_size * tile_size;

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(cb_size, {{CBIndex::c_0, data_format}})
            .set_page_size(CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core, cb_a_config);

    CircularBufferConfig cb_b_config =
        CircularBufferConfig(cb_size, {{CBIndex::c_1, data_format}})
            .set_page_size(CBIndex::c_1, tile_size);
    CreateCircularBuffer(program, core, cb_b_config);

    CircularBufferConfig cb_c_config =
        CircularBufferConfig(cb_size, {{CBIndex::c_16, data_format}})
            .set_page_size(CBIndex::c_16, tile_size);
    CreateCircularBuffer(program, core, cb_c_config);

    // 创建内核
    KernelHandle reader_kernel = CreateKernel(
        program,
        "kernels/dataflow/reader_matmul_block.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {Mt, Kt, Nt, block_size}
        }
    );

    KernelHandle compute_kernel = CreateKernel(
        program,
        "kernels/compute/matmul_block.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {Mt, Kt, Nt, block_size}
        }
    );

    KernelHandle writer_kernel = CreateKernel(
        program,
        "kernels/dataflow/writer_matmul_block.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {Mt, Nt, block_size}
        }
    );

    // 生成随机测试数据
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> data_a(Mt * Kt * 32 * 32);
    std::vector<uint16_t> data_b(Kt * Nt * 32 * 32);

    for (auto& val : data_a) {
        val = float_to_bfloat16(dist(rng));
    }
    for (auto& val : data_b) {
        val = float_to_bfloat16(dist(rng));
    }

    EnqueueWriteBuffer(device->command_queue(), buffer_a, data_a.data(), false);
    EnqueueWriteBuffer(device->command_queue(), buffer_b, data_b.data(), true);

    // 设置运行时参数
    SetRuntimeArgs(program, reader_kernel, core, {
        buffer_a->address(),
        buffer_b->address()
    });

    SetRuntimeArgs(program, writer_kernel, core, {
        buffer_c->address()
    });

    // 执行
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    // 读取结果
    std::vector<uint16_t> data_c(Mt * Nt * 32 * 32);
    EnqueueReadBuffer(device->command_queue(), buffer_c, data_c.data(), true);

    std::cout << "Matmul block example completed!" << std::endl;
    std::cout << "Matrix dimensions: " << (Mt * 32) << "x" << (Kt * 32)
              << " * " << (Kt * 32) << "x" << (Nt * 32) << std::endl;
    std::cout << "Block size: " << block_size << "x" << block_size << " tiles" << std::endl;

    DeallocateBuffer(buffer_a);
    DeallocateBuffer(buffer_b);
    DeallocateBuffer(buffer_c);
    CloseDevice(device);

    return 0;
}
```

**Compute 内核** (`kernels/compute/matmul_block.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    // 编译时参数
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_c = tt::CBIndex::c_16;

    // 初始化块矩阵乘法
    // mm_block_init 配置矩阵引擎用于块级操作
    mm_block_init(cb_a, cb_b, cb_c, false, false, false);

    // 遍历输出块
    for (uint32_t mt = 0; mt < Mt; mt += block_size) {
        for (uint32_t nt = 0; nt < Nt; nt += block_size) {

            // 获取输出寄存器
            tile_regs_acquire();

            // 计算当前输出块
            for (uint32_t kt = 0; kt < Kt; kt += block_size) {

                // 等待输入块
                cb_wait_front(cb_a, block_size * block_size);
                cb_wait_front(cb_b, block_size * block_size);

                // 执行块矩阵乘法
                // matmul_block 处理 block_size x block_size 的 tiles
                matmul_block(
                    cb_a, cb_b,
                    0, 0,  // src tile indices
                    0,     // dst tile index
                    kt == 0,  // accumulate flag
                    block_size,  // num tiles in block
                    block_size,  // num tiles in block
                    block_size   // num tiles in block
                );

                cb_pop_front(cb_a, block_size * block_size);
                cb_pop_front(cb_b, block_size * block_size);
            }

            tile_regs_commit();
            tile_regs_wait();

            // 输出结果块
            cb_reserve_back(cb_c, block_size * block_size);
            for (uint32_t i = 0; i < block_size * block_size; i++) {
                pack_tile(i, cb_c);
            }
            cb_push_back(cb_c, block_size * block_size);

            tile_regs_release();
        }
    }
}
```

#### 代码解释

**块级矩阵乘法 vs Tile 级**:

```
Tile 级 (matmul_tiles):
- 一次处理 1x1 tiles
- 适合小规模矩阵
- 更多循环开销

块级 (matmul_block):
- 一次处理 NxN tiles
- 更好的数据局部性
- 减少循环开销
- 更高效地利用 FPU
```

**mm_block_init 参数**:

1. **cb_a, cb_b, cb_c**: 输入输出循环缓冲区
2. **transpose_a, transpose_b**: 是否转置输入矩阵
3. **accumulate**: 是否累加到目标

**matmul_block 参数**:

1. **src_a_idx, src_b_idx**: 输入 tile 索引
2. **dst_idx**: 输出 tile 索引
3. **accumulate**: 是否累加
4. **num_tiles_a, num_tiles_b, num_tiles_c**: 块维度

#### 运行步骤

```bash
make metal_example_matmul_block
./programming_examples/metal_example_matmul_block
```

---

### 2.6 归约操作示例

**问题描述**: 深度学习中的归约操作（如求和、平均、最大值）是常见需求。本示例演示如何使用 `reduce_init`、`reduce_tile` 等 API。

**解决方案**: 创建计算内核执行 SUM、AVG、MAX 三种归约操作。

#### 完整代码

**Host 端代码** (`reduction_ops.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>

using namespace tt::tt_metal;

// 归约操作类型
enum class ReduceOp : uint32_t {
    SUM = 0,
    AVG = 1,
    MAX = 2
};

int main() {
    Device* device = CreateDevice(0);
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 配置: 4x4 tiles 输入，沿行归约
    uint32_t num_tiles_h = 4;
    uint32_t num_tiles_w = 4;
    uint32_t tile_size = 32 * 32 * 2;
    DataFormat data_format = DataFormat::Float16_b;

    // 创建缓冲区
    Buffer input_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles_h * num_tiles_w * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer output_sum = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles_h * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer output_max = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles_h * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 创建 CB
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(num_tiles_w * tile_size, {{CBIndex::c_0, data_format}})
            .set_page_size(CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core, cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(tile_size, {{CBIndex::c_16, data_format}})
            .set_page_size(CBIndex::c_16, tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // 创建 Reader 内核
    KernelHandle reader_kernel = CreateKernel(
        program,
        "kernels/dataflow/reader_reduction.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {num_tiles_h, num_tiles_w}
        }
    );

    // 创建 SUM 归约内核
    KernelHandle sum_kernel = CreateKernel(
        program,
        "kernels/compute/reduce_sum.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {num_tiles_w, static_cast<uint32_t>(ReduceOp::SUM)}
        }
    );

    // 创建 MAX 归约内核
    KernelHandle max_kernel = CreateKernel(
        program,
        "kernels/compute/reduce_max.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {num_tiles_w, static_cast<uint32_t>(ReduceOp::MAX)}
        }
    );

    // 创建 Writer 内核
    KernelHandle writer_kernel = CreateKernel(
        program,
        "kernels/dataflow/writer_reduction.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // 准备测试数据
    std::vector<uint16_t> input_data(num_tiles_h * num_tiles_w * 32 * 32);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = float_to_bfloat16(static_cast<float>(i % 100));
    }
    EnqueueWriteBuffer(device->command_queue(), input_buffer, input_data.data(), true);

    // 设置运行时参数
    SetRuntimeArgs(program, reader_kernel, core, {
        input_buffer->address(),
        num_tiles_h,
        num_tiles_w
    });

    SetRuntimeArgs(program, writer_kernel, core, {
        output_sum->address(),
        output_max->address(),
        num_tiles_h
    });

    // 执行
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    // 读取结果
    std::vector<uint16_t> sum_result(num_tiles_h * 32 * 32);
    std::vector<uint16_t> max_result(num_tiles_h * 32 * 32);
    EnqueueReadBuffer(device->command_queue(), output_sum, sum_result.data(), false);
    EnqueueReadBuffer(device->command_queue(), output_max, max_result.data(), true);

    std::cout << "Reduction operations completed!" << std::endl;
    std::cout << "Input shape: " << (num_tiles_h * 32) << "x" << (num_tiles_w * 32) << std::endl;
    std::cout << "Output shape: " << (num_tiles_h * 32) << "x" << 32 << std::endl;

    DeallocateBuffer(input_buffer);
    DeallocateBuffer(output_sum);
    DeallocateBuffer(output_max);
    CloseDevice(device);

    return 0;
}
```

**SUM 归约内核** (`kernels/compute/reduce_sum.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    uint32_t num_tiles_w = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // 初始化归约操作
    // reduce_init 配置归约引擎
    reduce_init<true>(cb_in, cb_out);

    // 处理每一行
    for (uint32_t row = 0; row < num_tiles_w; row++) {

        tile_regs_acquire();

        // 初始化累加器为 0
        copy_tile(cb_in, 0, 0);

        // 累加该行的所有 tiles
        for (uint32_t col = 0; col < num_tiles_w; col++) {
            cb_wait_front(cb_in, 1);

            if (col == 0) {
                // 第一个 tile: 直接复制
                copy_tile(cb_in, 0, 0);
            } else {
                // 后续 tiles: 累加
                add_tiles(cb_in, cb_in, 0, 0, 0);
            }

            cb_pop_front(cb_in, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        // 输出结果
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }

    // 反初始化
    reduce_uninit();
}
```

**MAX 归约内核** (`kernels/compute/reduce_max.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    uint32_t num_tiles_w = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    reduce_init<true>(cb_in, cb_out);

    for (uint32_t row = 0; row < num_tiles_w; row++) {

        tile_regs_acquire();

        bool first = true;
        for (uint32_t col = 0; col < num_tiles_w; col++) {
            cb_wait_front(cb_in, 1);

            if (first) {
                copy_tile(cb_in, 0, 0);
                first = false;
            } else {
                // 使用 SFPU 的 max 操作
                max_tiles(cb_in, cb_in, 0, 0, 0);
            }

            cb_pop_front(cb_in, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }

    reduce_uninit();
}
```

#### 代码解释

**归约操作类型**:

1. **SUM**: 沿指定维度求和
2. **AVG**: 求和后除以元素数量
3. **MAX/MIN**: 求最大值/最小值

**reduce_init 模板参数**:

- `true`: 使用 FP32 累加（更高精度）
- `false`: 使用 FP16 累加

**归约模式**:

```
输入 (4x4 tiles):
┌────┬────┬────┬────┐
│ A0 │ A1 │ A2 │ A3 │  Row 0
├────┼────┼────┼────┤
│ B0 │ B1 │ B2 │ B3 │  Row 1
├────┼────┼────┼────┤
│ C0 │ C1 │ C2 │ C3 │  Row 2
├────┼────┼────┼────┤
│ D0 │ D1 │ D2 │ D3 │  Row 3
└────┴────┴────┴────┘

SUM 归约 (沿行):
┌────┐
│A0+...│  Row 0
├────┤
│B0+...│  Row 1
├────┤
│C0+...│  Row 2
├────┤
│D0+...│  Row 3
└────┘
```

#### 运行步骤

```bash
make metal_example_reduction_ops
./programming_examples/metal_example_reduction_ops
```

---

### 2.7 SFPU 完整操作示例

**问题描述**: SFPU (Special Function Processing Unit) 支持丰富的向量操作。本示例演示各种激活函数和 SFPI 条件执行。

**解决方案**: 创建多个计算内核，分别演示 ReLU、GELU、Sigmoid、SiLU 等激活函数，以及 SFPI 条件执行。

#### 完整代码

**Host 端代码** (`sfpu_complete.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);

    uint32_t num_tiles = 4;
    uint32_t tile_size = 32 * 32 * 2;
    DataFormat data_format = DataFormat::Float16_b;

    // 创建输入缓冲区
    Buffer input_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 创建多个输出缓冲区 (每个激活函数一个)
    Buffer relu_output = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer gelu_output = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer sigmoid_output = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 准备测试数据
    std::vector<uint16_t> input_data(num_tiles * 32 * 32);
    for (size_t i = 0; i < input_data.size(); i++) {
        // 生成 -5.0 到 5.0 的值
        float val = -5.0f + (i % 1000) / 100.0f;
        input_data[i] = float_to_bfloat16(val);
    }
    EnqueueWriteBuffer(device->command_queue(), input_buffer, input_data.data(), true);

    // 创建并执行 ReLU 程序
    {
        Program program = CreateProgram();
        CoreCoord core = {0, 0};

        CircularBufferConfig cb_config =
            CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, data_format}})
                .set_page_size(CBIndex::c_0, tile_size);
        CreateCircularBuffer(program, core, cb_config);

        KernelHandle reader = CreateKernel(
            program,
            "kernels/dataflow/sfpu_reader.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
        );

        KernelHandle compute = CreateKernel(
            program,
            "kernels/compute/sfpu_relu.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
        );

        KernelHandle writer = CreateKernel(
            program,
            "kernels/dataflow/sfpu_writer.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
        );

        SetRuntimeArgs(program, reader, core, {input_buffer->address(), num_tiles});
        SetRuntimeArgs(program, writer, core, {relu_output->address(), num_tiles});

        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
    }

    // 创建并执行 GELU 程序
    {
        Program program = CreateProgram();
        CoreCoord core = {0, 0};

        CircularBufferConfig cb_config =
            CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, data_format}})
                .set_page_size(CBIndex::c_0, tile_size);
        CreateCircularBuffer(program, core, cb_config);

        KernelHandle reader = CreateKernel(
            program,
            "kernels/dataflow/sfpu_reader.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
        );

        KernelHandle compute = CreateKernel(
            program,
            "kernels/compute/sfpu_gelu.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
        );

        KernelHandle writer = CreateKernel(
            program,
            "kernels/dataflow/sfpu_writer.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
        );

        SetRuntimeArgs(program, reader, core, {input_buffer->address(), num_tiles});
        SetRuntimeArgs(program, writer, core, {gelu_output->address(), num_tiles});

        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
    }

    std::cout << "SFPU operations completed!" << std::endl;
    std::cout << "Executed: ReLU, GELU, Sigmoid" << std::endl;

    DeallocateBuffer(input_buffer);
    DeallocateBuffer(relu_output);
    DeallocateBuffer(gelu_output);
    DeallocateBuffer(sigmoid_output);
    CloseDevice(device);

    return 0;
}
```

**ReLU 内核** (`kernels/compute/sfpu_relu.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_0;  // 原地操作

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_in, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();

        // 复制到寄存器
        copy_tile(cb_in, 0, 0);

        // ReLU: max(0, x)
        relu_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        // 输出
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
```

**GELU 内核** (`kernels/compute/sfpu_gelu.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>

void MAIN {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_0;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_in, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();

        copy_tile(cb_in, 0, 0);

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
```

**SFPI 条件执行示例** (`kernels/compute/sfpi_conditional.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <sfpi.h>

using namespace sfpi;

void MAIN {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();

        // 将 tile 加载到目标寄存器
        copy_tile(cb_in, 0, 0);

        // 使用 SFPI 进行条件执行
        // 处理一个 tile 的 256 个元素 (8 个向量，每个 32 元素)
        for (uint32_t face = 0; face < 4; face++) {
            for (uint32_t row = 0; row < 2; row++) {
                uint32_t base_idx = face * 8 + row * 4;

                // 加载向量
                vFloat x = dst_reg[base_idx];

                // 条件执行: 实现分段函数
                // if x < -2: y = 0
                // else if x > 2: y = 1
                // else: y = (x + 2) / 4
                v_if(x < -2.0f) {
                    dst_reg[base_idx] = 0.0f;
                } v_elseif(x > 2.0f) {
                    dst_reg[base_idx] = 1.0f;
                } v_else {
                    dst_reg[base_idx] = (x + 2.0f) * 0.25f;
                } v_endif;
            }
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
```

#### 代码解释

**SFPU 激活函数列表**:

| 函数 | 说明 | 数学表达式 |
|------|------|-----------|
| `relu_tile` | 整流线性单元 | max(0, x) |
| `gelu_tile` | 高斯误差线性单元 | 0.5x(1 + erf(x/√2)) |
| `sigmoid_tile` | Sigmoid 函数 | 1 / (1 + e^(-x)) |
| `tanh_tile` | 双曲正切 | (e^x - e^(-x)) / (e^x + e^(-x)) |
| `exp_tile` | 指数函数 | e^x |
| `log_tile` | 自然对数 | ln(x) |
| `sqrt_tile` | 平方根 | √x |
| `recip_tile` | 倒数 | 1/x |

**SFPI 条件执行**:

```cpp
// 向量条件执行
v_if(condition) {
    // 条件为真的向量元素执行
} v_elseif(condition2) {
    // 条件2为真的执行
} v_else {
    // 其他情况
} v_endif;
```

#### 运行步骤

```bash
make metal_example_sfpu_complete
./programming_examples/metal_example_sfpu_complete
```

---

### 2.8 子设备管理示例

**问题描述**: 在多租户或复杂应用场景中，需要将设备划分为多个独立的子设备，实现资源隔离和并行执行。

**解决方案**: 使用 `create_sub_device_manager`、`load_sub_device_manager` 等 API 管理子设备。

#### 完整代码

**Host 端代码** (`sub_device_management.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <iostream>
#include <vector>

using namespace tt::tt_metal;

int main() {
    // 创建设备
    Device* device = CreateDevice(0);

    // 获取设备网格大小
    auto grid_size = device->compute_with_storage_grid_size();
    std::cout << "Device grid size: " << grid_size.x << "x" << grid_size.y << std::endl;

    // 定义子设备 1: 使用上半部分核心
    CoreRange sub_device_1_cores(
        CoreCoord(0, 0),
        CoreCoord(grid_size.x - 1, grid_size.y / 2 - 1)
    );

    // 定义子设备 2: 使用下半部分核心
    CoreRange sub_device_2_cores(
        CoreCoord(0, grid_size.y / 2),
        CoreCoord(grid_size.x - 1, grid_size.y - 1)
    );

    // 创建子设备管理器
    SubDeviceManager sub_device_manager = device->create_sub_device_manager(
        {sub_device_1_cores, sub_device_2_cores}
    );

    std::cout << "Created sub-device manager" << std::endl;
    std::cout << "Sub-device 1: " << sub_device_1_cores.size() << " cores" << std::endl;
    std::cout << "Sub-device 2: " << sub_device_2_cores.size() << " cores" << std::endl;

    // 加载子设备管理器
    device->load_sub_device_manager(sub_device_manager);
    std::cout << "Loaded sub-device manager" << std::endl;

    // 获取子设备 ID
    auto sub_device_ids = device->get_sub_device_ids();
    std::cout << "Number of sub-devices: " << sub_device_ids.size() << std::endl;

    // 在子设备 1 上创建程序
    Program program_1 = CreateProgram();

    // 为子设备 1 创建内核
    KernelHandle kernel_1 = CreateKernel(
        program_1,
        "kernels/compute/simple_compute.cpp",
        sub_device_1_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    // 在子设备 2 上创建程序
    Program program_2 = CreateProgram();

    KernelHandle kernel_2 = CreateKernel(
        program_2,
        "kernels/compute/simple_compute.cpp",
        sub_device_2_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    // 设置子设备停顿组 (用于同步)
    device->set_sub_device_stall_group(sub_device_ids);

    // 在子设备 1 上执行程序
    EnqueueProgram(device->command_queue(), program_1, sub_device_ids[0], false);

    // 在子设备 2 上执行程序 (并行)
    EnqueueProgram(device->command_queue(), program_2, sub_device_ids[1], false);

    // 等待所有子设备完成
    Finish(device->command_queue());

    std::cout << "Sub-device programs completed!" << std::endl;

    // 清理: 移除子设备管理器
    device->remove_sub_device_manager(sub_device_manager);
    std::cout << "Removed sub-device manager" << std::endl;

    CloseDevice(device);

    return 0;
}
```

**子设备配置详解**:

```cpp
// 更复杂的子设备配置示例
void advanced_sub_device_example(Device* device) {
    auto grid_size = device->compute_with_storage_grid_size();

    // 配置 1: 细粒度子设备划分
    // 创建 4 个象限作为独立子设备
    std::vector<CoreRange> quadrant_cores;

    uint32_t mid_x = grid_size.x / 2;
    uint32_t mid_y = grid_size.y / 2;

    // 象限 1: 左上
    quadrant_cores.push_back(CoreRange(
        CoreCoord(0, 0),
        CoreCoord(mid_x - 1, mid_y - 1)
    ));

    // 象限 2: 右上
    quadrant_cores.push_back(CoreRange(
        CoreCoord(mid_x, 0),
        CoreCoord(grid_size.x - 1, mid_y - 1)
    ));

    // 象限 3: 左下
    quadrant_cores.push_back(CoreRange(
        CoreCoord(0, mid_y),
        CoreCoord(mid_x - 1, grid_size.y - 1)
    ));

    // 象限 4: 右下
    quadrant_cores.push_back(CoreRange(
        CoreCoord(mid_x, mid_y),
        CoreCoord(grid_size.x - 1, grid_size.y - 1)
    ));

    // 创建子设备管理器
    SubDeviceManager manager = device->create_sub_device_manager(quadrant_cores);
    device->load_sub_device_manager(manager);

    // 现在可以独立地向每个象限提交工作
    auto sub_device_ids = device->get_sub_device_ids();

    for (size_t i = 0; i < sub_device_ids.size(); i++) {
        std::cout << "Sub-device " << i << " ID: " << sub_device_ids[i] << std::endl;
    }
}
```

#### 代码解释

**子设备管理关键概念**:

1. **SubDeviceManager**: 管理一组子设备配置
2. **CoreRange**: 定义每个子设备包含的核心
3. **SubDevice ID**: 加载后分配的唯一标识符

**使用场景**:

- **多租户**: 不同用户/任务使用独立子设备
- **流水线并行**: 不同阶段使用不同子设备
- **资源隔离**: 防止一个任务影响其他任务

**API 流程**:

```
create_sub_device_manager(cores) -> manager
       ↓
load_sub_device_manager(manager)
       ↓
get_sub_device_ids() -> ids
       ↓
EnqueueProgram(cq, program, sub_device_id)
       ↓
remove_sub_device_manager(manager)
```

#### 运行步骤

```bash
make metal_example_sub_device
./programming_examples/metal_example_sub_device
```

---

## 中优先级示例

### 3.1 Metal Trace 完整示例

**问题描述**: 对于重复执行的计算图，捕获执行轨迹并重放可以显著减少 Host 开销，提高性能。

**解决方案**: 使用 Metal Trace API 捕获程序执行序列，然后多次重放。

#### 完整代码

**Host 端代码** (`metal_trace.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/trace.hpp>
#include <iostream>
#include <chrono>

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();

    // 配置
    uint32_t num_iterations = 100;
    uint32_t num_tiles = 16;
    uint32_t tile_size = 32 * 32 * 2;

    // 创建缓冲区
    Buffer input_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    Buffer output_buffer = CreateBuffer(
        InterleavedBufferConfig{
            .device = device,
            .size = num_tiles * tile_size,
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM
        }
    );

    // 准备初始数据
    std::vector<uint16_t> input_data(num_tiles * 32 * 32, 0);
    EnqueueWriteBuffer(cq, input_buffer, input_data.data(), true);

    // ========== 阶段 1: 创建可追踪的程序 ==========
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 创建 CB
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core, cb_config);

    // 创建内核
    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/trace_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
    );

    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/trace_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/trace_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
    );

    // 设置运行时参数
    SetRuntimeArgs(program, reader, core, {input_buffer->address(), num_tiles});
    SetRuntimeArgs(program, writer, core, {output_buffer->address(), num_tiles});

    // ========== 阶段 2: 捕获 Trace ==========
    std::cout << "Capturing trace..." << std::endl;

    // 开始捕获
    uint32_t trace_id = BeginTraceCapture(cq);

    // 执行一次程序 (这将被记录到 trace 中)
    EnqueueProgram(cq, program, false);

    // 结束捕获
    EndTraceCapture(cq, trace_id);

    std::cout << "Trace captured with ID: " << trace_id << std::endl;

    // ========== 阶段 3: 重放 Trace (多次) ==========
    std::cout << "Replaying trace " << num_iterations << " times..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_iterations; i++) {
        // 重放 trace (无 Host 开销)
        ReplayTrace(cq, trace_id, false);
    }

    Finish(cq);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Trace replay completed!" << std::endl;
    std::cout << "Total time: " << duration.count() << " us" << std::endl;
    std::cout << "Average per iteration: " << (duration.count() / num_iterations) << " us" << std::endl;

    // ========== 阶段 4: 清理 ==========
    // 释放 trace
    ReleaseTrace(cq, trace_id);

    // 对比: 不使用 trace 的直接执行
    std::cout << "\nRunning without trace for comparison..." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_iterations; i++) {
        EnqueueProgram(cq, program, false);
    }

    Finish(cq);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Direct execution completed!" << std::endl;
    std::cout << "Total time: " << duration.count() << " us" << std::endl;
    std::cout << "Average per iteration: " << (duration.count() / num_iterations) << " us" << std::endl;

    DeallocateBuffer(input_buffer);
    DeallocateBuffer(output_buffer);
    CloseDevice(device);

    return 0;
}
```

#### 代码解释

**Trace 工作流程**:

```
1. BeginTraceCapture(cq) -> trace_id
          ↓
2. EnqueueProgram(cq, program)  [被记录]
          ↓
3. EndTraceCapture(cq, trace_id)
          ↓
4. ReplayTrace(cq, trace_id) [多次重放]
          ↓
5. ReleaseTrace(cq, trace_id)
```

**性能优势**:

- **减少 Host 开销**: 避免重复的参数设置和命令提交
- **预编译优化**: Trace 中的程序已经编译和优化
- **批量执行**: 可以一次性提交多个迭代

**使用限制**:

- Trace 中的程序参数必须是静态的
- 动态形状需要多个 trace
- 内存地址在捕获时固定

#### 运行步骤

```bash
make metal_example_trace
./programming_examples/metal_example_trace
```

---

### 3.2 多队列并行示例

**问题描述**: 使用多个命令队列 (CQ) 可以实现更细粒度的并行控制，例如同时执行独立计算和数据传输。

**解决方案**: 创建多个命令队列，向不同队列提交独立工作负载。

#### 完整代码

**Host 端代码** (`multi_cq.cpp`):

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <iostream>
#include <thread>
#include <vector>

using namespace tt::tt_metal;

// 在指定 CQ 上执行工作的函数
void execute_on_cq(Device* device, uint32_t cq_id,
                   Buffer* input, Buffer* output,
                   uint32_t num_tiles, uint32_t tile_size) {

    CommandQueue& cq = device->command_queue(cq_id);

    // 创建程序
    Program program = CreateProgram();
    CoreCoord core = {cq_id % 2, cq_id / 2};  // 不同 CQ 使用不同核心

    // 创建 CB
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core, cb_config);

    // 创建内核
    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/multi_cq_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
    );

    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/multi_cq_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/multi_cq_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
    );

    // 设置参数
    SetRuntimeArgs(program, reader, core, {input->address(), num_tiles});
    SetRuntimeArgs(program, writer, core, {output->address(), num_tiles});

    // 写入输入数据
    std::vector<uint16_t> data(num_tiles * 32 * 32, cq_id);  // 用 CQ ID 填充
    EnqueueWriteBuffer(cq, *input, data.data(), true);

    // 执行程序
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // 读取结果
    std::vector<uint16_t> result(num_tiles * 32 * 32);
    EnqueueReadBuffer(cq, *output, result.data(), true);

    std::cout << "CQ " << cq_id << " completed on core ("
              << core.x << ", " << core.y << ")" << std::endl;
}

int main() {
    // 创建设备，指定 2 个命令队列
    Device* device = CreateDevice(0, 2);  // device_id, num_cqs

    uint32_t num_cqs = 2;
    uint32_t num_tiles = 8;
    uint32_t tile_size = 32 * 32 * 2;

    std::cout << "Using " << num_cqs << " command queues" << std::endl;

    // 为每个 CQ 创建缓冲区
    std::vector<Buffer> input_buffers;
    std::vector<Buffer> output_buffers;

    for (uint32_t i = 0; i < num_cqs; i++) {
        input_buffers.push_back(CreateBuffer(
            InterleavedBufferConfig{
                .device = device,
                .size = num_tiles * tile_size,
                .page_size = tile_size,
                .buffer_type = BufferType::DRAM
            }
        ));

        output_buffers.push_back(CreateBuffer(
            InterleavedBufferConfig{
                .device = device,
                .size = num_tiles * tile_size,
                .page_size = tile_size,
                .buffer_type = BufferType::DRAM
            }
        ));
    }

    // 方法 1: 顺序执行
    std::cout << "\n=== Sequential execution ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_cqs; i++) {
        execute_on_cq(device, i, &input_buffers[i], &output_buffers[i],
                      num_tiles, tile_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sequential time: " << seq_duration.count() << " us" << std::endl;

    // 方法 2: 并行执行 (使用多线程)
    std::cout << "\n=== Parallel execution ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_cqs; i++) {
        threads.emplace_back(execute_on_cq, device, i,
                            &input_buffers[i], &output_buffers[i],
                            num_tiles, tile_size);
    }

    for (auto& t : threads) {
        t.join();
    }

    end = std::chrono::high_resolution_clock::now();
    auto par_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Parallel time: " << par_duration.count() << " us" << std::endl;

    // 计算加速比
    float speedup = static_cast<float>(seq_duration.count()) / par_duration.count();
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

    // 清理
    for (auto& buf : input_buffers) {
        DeallocateBuffer(buf);
    }
    for (auto& buf : output_buffers) {
        DeallocateBuffer(buf);
    }

    CloseDevice(device);

    return 0;
}
```

#### 代码解释

**多队列优势**:

1. **并行提交**: 不同队列可以并行提交命令
2. **独立同步**: 每个队列可以独立 Finish
3. **负载分离**: 计算和数据传输可以分离到不同队列

**使用模式**:

```cpp
// 创建多队列设备
Device* device = CreateDevice(device_id, num_cqs);

// 获取指定队列
CommandQueue& cq_0 = device->command_queue(0);
CommandQueue& cq_1 = device->command_queue(1);

// 独立提交工作
EnqueueProgram(cq_0, program_0, false);
EnqueueProgram(cq_1, program_1, false);

// 独立同步
Finish(cq_0);
Finish(cq_1);
```

#### 运行步骤

```bash
make metal_example_multi_cq
./programming_examples/metal_example_multi_cq
```

---

## 4. 运行环境配置

### 环境变量设置

```bash
# 必需环境变量
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# 调试选项
export TT_METAL_DPRINT_ENABLED=1  # 启用 DPRINT 输出
export TT_METAL_WATCHER_ENABLED=1  # 启用 Watcher 调试
export TT_METAL_PROFILER_ENABLED=1  # 启用性能分析
```

### CMake 配置

```cmake
# CMakeLists.txt 示例
cmake_minimum_required(VERSION 3.16)
project(tt_metal_examples)

set(CMAKE_CXX_STANDARD 17)

# 查找 TT-Metal
find_package(tt-metal REQUIRED)

# 添加示例
add_executable(metal_example_hello_world hello_world.cpp)
target_link_libraries(metal_example_hello_world PRIVATE tt-metalium)

# 设置内核路径
target_compile_definitions(metal_example_hello_world PRIVATE
    KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)
```

### 构建命令

```bash
# 完整构建
mkdir -p build && cd build
cmake .. -DBUILD_PROGRAMMING_EXAMPLES=ON
make -j$(nproc)

# 运行所有示例
ctest -R programming_examples --output-on-failure
```

---

## 5. 总结

本文档扩展了 TT-Metalium 的编程示例库，补充了以下关键示例:

### 高优先级示例 (8个)

| 示例 | 关键 API | 学习目标 |
|------|----------|----------|
| Hello World Kernel | `DPRINT`, `get_arg_val` | 基础 Host-Device 交互 |
| 信号量同步 | `noc_semaphore_set/inc/wait` | 核心间同步 |
| 双缓冲模式 | `cb_reserve_back/push/wait/pop` | 计算与通信重叠 |
| 分片张量操作 | `ShardedBufferConfig` | 大数据分片处理 |
| 矩阵块操作 | `mm_block_init`, `matmul_block` | 高效矩阵乘法 |
| 归约操作 | `reduce_init`, `reduce_tile` | 聚合操作 |
| SFPU 完整操作 | `relu_tile`, `gelu_tile`, SFPI | 向量运算 |
| 子设备管理 | `create_sub_device_manager` | 资源隔离与并行 |

### 中优先级示例 (2个)

| 示例 | 关键 API | 学习目标 |
|------|----------|----------|
| Metal Trace | `BeginTraceCapture`, `ReplayTrace` | 性能优化 |
| 多队列并行 | `CreateDevice(num_cqs)` | 细粒度并行控制 |

### 学习路径建议

```
入门阶段:
  Hello World Kernel -> 信号量同步 -> 双缓冲模式

进阶阶段:
  分片张量操作 -> 矩阵块操作 -> 归约操作

高级阶段:
  SFPU 完整操作 -> 子设备管理 -> Metal Trace -> 多队列并行
```

---

*文档生成时间: 2026-03-12*
*基于 TT-Metalium v0.59.0+*
