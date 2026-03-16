# programming_examples/ 源码解析

## 1. 模块概述

`programming_examples/` 是 TT-Metalium 框架的官方编程示例集合，位于 `/tmp/tt-metal/tt_metal/programming_examples/`。这些示例展示了如何使用 TT-Metalium API 在 Tenstorrent AI 加速器上进行编程，从简单的 "Hello World" 到复杂的分布式计算。

### 1.1 目录结构

```
programming_examples/
├── CMakeLists.txt
├── README.md
├── hello_world_compute_kernel/       # 计算内核入门
├── hello_world_datamovement_kernel/  # 数据移动内核入门
├── hello_world_datatypes_kernel/     # 数据类型示例
├── add_2_integers_in_compute/        # 计算内核加法
├── add_2_integers_in_riscv/          # RISC-V 内核加法
├── loopback/                         # 数据回环测试
├── eltwise_binary/                   # 二元逐元素运算
├── eltwise_sfpu/                     # SFPU 逐元素运算
├── sfpu_eltwise_chain/               # SFPU 运算链
├── custom_sfpi_add/                  # 自定义 SFPI 加法
├── custom_sfpi_smoothstep/           # 自定义 SFPI smoothstep
├── matmul/                           # 矩阵乘法系列
│   ├── matmul_single_core/           # 单核矩阵乘法
│   ├── matmul_multi_core/            # 多核矩阵乘法
│   ├── matmul_multicore_reuse/       # 多核复用优化
│   └── matmul_multicore_reuse_mcast/ # 多核复用+多播
├── vecadd_multi_core/                # 多核向量加法
├── vecadd_sharding/                  # 分片向量加法
├── pad_multi_core/                   # 多核填充操作
├── shard_data_rm/                    # 行主序分片数据
├── NoC_tile_transfer/                # NoC 片上传输
├── distributed/                      # 分布式编程示例
│   ├── 1_distributed_program_dispatch/   # 分布式程序分发
│   ├── 2_distributed_buffer_rw/          # 分布式缓冲区读写
│   ├── 3_distributed_eltwise_add/        # 分布式逐元素加法
│   └── 4_distributed_trace_and_events/   # 分布式追踪与事件
├── profiler/                         # 性能分析示例
│   ├── test_custom_cycle_count/          # 自定义周期计数
│   ├── test_full_buffer/                 # 完整缓冲区测试
│   ├── test_multi_op/                    # 多操作测试
│   └── test_timestamped_events/          # 带时间戳事件
└── contributed/                      # 社区贡献示例
    ├── vecadd/                       # 向量加法
    └── multicast/                    # 多播示例
```

### 1.2 学习路径

```
入门阶段 ──────────────────────────────────────────────>
    │
    ├── hello_world_compute_kernel       # 了解计算内核基础
    ├── hello_world_datamovement_kernel  # 了解数据移动内核
    ├── add_2_integers_in_riscv          # RISC-V 数据移动内核计算
    └── add_2_integers_in_compute        # 完整数据流：DRAM -> CB -> 计算 -> CB -> DRAM

进阶阶段 ──────────────────────────────────────────────>
    │
    ├── loopback                         # 数据回环：DRAM -> L1 -> DRAM
    ├── eltwise_binary                   # 多瓦片二元运算
    ├── eltwise_sfpu                     # SFPU 数学函数
    └── sfpu_eltwise_chain               # 多阶段 SFPU 运算链

高级阶段 ──────────────────────────────────────────────>
    │
    ├── matmul_single_core               # 单核矩阵乘法
    ├── matmul_multi_core                # 多核并行矩阵乘法
    ├── vecadd_multi_core                # 多核向量加法
    └── pad_multi_core / shard_data_rm   # 数据布局操作

分布式阶段 ────────────────────────────────────────────>
    │
    ├── 1_distributed_program_dispatch   # 多设备程序分发
    ├── 2_distributed_buffer_rw          # 分布式缓冲区管理
    ├── 3_distributed_eltwise_add        # 分布式计算
    └── 4_distributed_trace_and_events   # 高级分布式特性

性能优化 ──────────────────────────────────────────────>
    │
    └── profiler/                        # 性能分析与调试
```

## 2. 示例分类

### 2.1 基础示例

#### 2.1.1 hello_world_compute_kernel

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/hello_world_compute_kernel/`

**功能说明**: 最简单的计算内核示例，演示如何在 Tensix 核心的计算单元上运行代码。

**核心概念**:
- Tensix 核心包含 3 个计算核心：UNPACK、MATH、PACK
- 使用 `CreateKernel` 创建计算内核
- 使用 `DPRINT_MATH/DPRINT_UNPACK/DPRINT_PACK` 进行调试输出

**代码要点**:
```cpp
// 创建计算内核
KernelHandle void_compute_kernel_id = CreateKernel(
    program,
    "kernels/compute/void_compute_kernel.cpp",
    core,
    ComputeConfig{});
```

#### 2.1.2 hello_world_datamovement_kernel

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/hello_world_datamovement_kernel/`

**功能说明**: 演示数据移动内核的使用。每个 Tensix 有 2 个 RISC-V 数据移动核心（BR/NC）。

**核心概念**:
- DataMovementProcessor::RISCV_0 (BR - Bridge)
- DataMovementProcessor::RISCV_1 (NC - Network Controller)
- 两个核心可以并行执行

**代码要点**:
```cpp
// 创建两个数据移动内核，分别在不同的 RISC-V 核心上运行
KernelHandle data_movement_kernel_0 = CreateKernel(
    program, kernel_path, core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

KernelHandle data_movement_kernel_1 = CreateKernel(
    program, kernel_path, core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
```

#### 2.1.3 add_2_integers_in_compute

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/add_2_integers_in_compute/`

**功能说明**: 完整的数据流示例，演示从 DRAM 读取数据、在计算核心执行加法、写回 DRAM 的完整流程。

**架构组件**:
- **Reader Kernel**: 从 DRAM 读取两个输入瓦片到 Circular Buffer
- **Compute Kernel**: 从 CB 读取、执行加法、写回 CB
- **Writer Kernel**: 从 CB 读取结果写回 DRAM

**核心概念**:
- Circular Buffer (CB): 内核间数据传输的管道
- CBIndex::c_0, c_1: 输入缓冲区
- CBIndex::c_16: 输出缓冲区
- 双缓冲机制（2 tiles per CB）实现计算与数据传输重叠

**代码结构**:
```cpp
// 1. 创建 Circular Buffers
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));  // 输入0
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));  // 输入1
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16)); // 输出

// 2. 创建三个内核
KernelHandle binary_reader_kernel_id = CreateKernel(program, reader_path, core, DataMovementConfig{...});
KernelHandle unary_writer_kernel_id = CreateKernel(program, writer_path, core, DataMovementConfig{...});
KernelHandle eltwise_binary_kernel_id = CreateKernel(program, compute_path, core, ComputeConfig{...});
```

#### 2.1.4 add_2_integers_in_riscv

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/add_2_integers_in_riscv/`

**功能说明**: 演示在数据移动核心（RISC-V）上直接执行计算，而非使用专用计算核心。

**适用场景**:
- 简单算术运算
- 数据预处理/后处理
- 控制流逻辑

**注意**: 虽然数据移动核心可以执行计算，但性能远低于专用计算核心。

### 2.2 进阶示例

#### 2.2.1 eltwise_binary

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/eltwise_binary/`

**功能说明**: 处理多个瓦片的二元逐元素运算（如加法）。

**关键特性**:
- 处理 64 个瓦片的数据
- 使用 `TensorAccessorArgs` 进行编译时参数传递
- 支持 MathFidelity 配置（HiFi4 最精确）

**代码要点**:
```cpp
// 编译时参数传递
std::vector<uint32_t> reader_compile_time_args;
TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

// Math Fidelity 设置
ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
```

#### 2.2.2 eltwise_sfpu

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/eltwise_sfpu/`

**功能说明**: 使用 SFPU（Special Function Processing Unit）执行数学函数（如 exp）。

**SFPU 功能**:
- 指数、对数、三角函数
- 激活函数（sigmoid、tanh 等）
- 向量运算

#### 2.2.3 matmul/ 系列

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/matmul/`

**包含子示例**:

| 示例 | 描述 | 核心特性 |
|------|------|----------|
| `matmul_single_core` | 单核矩阵乘法 | 基础实现，理解瓦片化矩阵运算 |
| `matmul_multi_core` | 多核矩阵乘法 | SPMD 并行，工作负载分配 |
| `matmul_multicore_reuse` | 多核复用优化 | 内存复用优化 |
| `matmul_multicore_reuse_mcast` | 多核+多播 | 多播数据传输优化 |

**matmul_single_core 核心概念**:
```cpp
// 瓦片维度计算
uint32_t Mt = M / TILE_HEIGHT;  // M 维度的瓦片数
uint32_t Kt = K / TILE_WIDTH;   // K 维度的瓦片数
uint32_t Nt = N / TILE_WIDTH;   // N 维度的瓦片数

// 数据布局转换（行主序 -> 瓦片化）
src0_vec = tilize_nfaces(src0_vec, M, K);
result_vec = untilize_nfaces(result_vec, M, N);
```

**matmul_multi_core 工作分配**:
```cpp
// 使用 split_work_to_cores 自动分配工作
auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
    split_work_to_cores(core_grid, num_output_tiles_total);
```

### 2.3 分布式示例

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/distributed/`

#### 2.3.1 1_distributed_program_dispatch

**功能**: 演示如何在多设备网格（MeshDevice）上分发程序。

**核心概念**:
```cpp
// 创建 2x4 设备网格
auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

// 创建 MeshWorkload
auto mesh_workload = MeshWorkload();
auto target_devices = MeshCoordinateRange(mesh_device->shape());
mesh_workload.add_program(target_devices, std::move(example_program));

// 分发到所有设备
EnqueueMeshWorkload(cq, mesh_workload, false);
```

#### 2.3.2 2_distributed_buffer_rw

**功能**: 演示分布式缓冲区（MeshBuffer）的读写操作。

**核心概念**:
```cpp
// 创建分片缓冲区配置
auto distributed_buffer_config = ShardedBufferConfig{
    .global_size = distributed_buffer_size_bytes,
    .global_buffer_shape = distributed_buffer_shape,
    .shard_shape = shard_shape
};

// 在 L1 内存中分配分布式缓冲区
auto mesh_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
```

#### 2.3.3 3_distributed_eltwise_add

**功能**: 演示分布式逐元素加法计算。

**核心特性**:
- 数据自动分片到多个设备
- 并行执行加法运算
- 结果收集与验证

#### 2.3.4 4_distributed_trace_and_events

**功能**: 高级分布式编程示例，演示：
- 多 MeshCommandQueue（数据移动和计算分离）
- SubDevice 配置
- MeshTrace 捕获与重放
- MeshEvent 同步

**核心概念**:
```cpp
// 创建带有追踪区域的设备
auto mesh_device = MeshDevice::create(
    MeshDeviceConfig(MeshShape(2, 4)),
    0,              // l1 small size
    16 << 20,       // trace region size (16MB)
    2,              // num MeshCQs
    DispatchCoreType::ETH);

// 创建 SubDevice
SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);

// 捕获追踪
auto trace_id = BeginTraceCapture(mesh_device.get(), workload_cq_id);
EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
mesh_device->end_mesh_trace(workload_cq_id, trace_id);

// 重放追踪
mesh_device->replay_mesh_trace(workload_cq_id, trace_id, false);
```

### 2.4 性能分析示例

**文件路径**: `/tmp/tt-metal/tt_metal/programming_examples/profiler/`

#### 2.4.1 test_custom_cycle_count

**功能**: 演示如何测量内核执行周期数。

**关键 API**:
```cpp
// 读取性能分析结果
ReadMeshDeviceProfilerResults(*mesh_device);
```

#### 2.4.2 test_timestamped_events

**功能**: 演示带时间戳的事件追踪。

## 3. 核心示例详解

### 3.1 add_2_integers_in_compute 完整分析

**文件**: `/tmp/tt-metal/tt_metal/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute.cpp`

#### 3.1.1 主机代码流程

```cpp
// 1. 创建设备
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

// 2. 创建缓冲区配置
distributed::DeviceLocalBufferConfig dram_config{
    .page_size = single_tile_size,
    .buffer_type = tt_metal::BufferType::DRAM};
distributed::ReplicatedBufferConfig distributed_buffer_config{.size = single_tile_size};

// 3. 创建 DRAM 缓冲区
auto src0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
auto src1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
auto dst_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());

// 4. 创建 Circular Buffers（L1 内存中的管道）
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));
CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));

// 5. 创建内核
KernelHandle binary_reader_kernel_id = CreateKernel(program, reader_path, core, DataMovementConfig{...});
KernelHandle unary_writer_kernel_id = CreateKernel(program, writer_path, core, DataMovementConfig{...});
KernelHandle eltwise_binary_kernel_id = CreateKernel(program, compute_path, core, ComputeConfig{...});

// 6. 设置运行时参数
SetRuntimeArgs(program, binary_reader_kernel_id, core, {src0_addr, src1_addr});
SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_addr});

// 7. 执行
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

#### 3.1.2 Reader Kernel 分析

**文件**: `/tmp/tt-metal/tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp`

```cpp
void kernel_main() {
    // 从运行时参数获取 DRAM 地址
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);

    // 定义 CB 索引
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // 获取瓦片大小
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // 创建地址生成器（用于交错内存访问）
    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = in0_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // 读取第一个输入
    cb_reserve_back(cb_in0, 1);                          // 确保 CB 有空间
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);        // 获取 CB 写入地址
    noc_async_read_tile(0, in0, cb_in0_addr);            // 异步读取
    noc_async_read_barrier();                            // 等待读取完成
    cb_push_back(cb_in0, 1);                             // 标记数据就绪

    // 读取第二个输入（类似流程）...
}
```

#### 3.1.3 Compute Kernel 分析

**文件**: `/tmp/tt-metal/tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp`

```cpp
void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // 初始化二元运算
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    // 等待输入就绪
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    // 获取瓦片寄存器
    tile_regs_acquire();

    // 执行加法：从 cb_in0 和 cb_in1 读取，结果写入寄存器 0
    add_tiles(cb_in0, cb_in1, 0, 0, 0);

    // 提交结果
    tile_regs_commit();
    tile_regs_wait();

    // 打包结果到输出 CB
    pack_tile(0, cb_out0);
    tile_regs_release();

    // 释放输入 CB
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_push_back(cb_out0, 1);
}
```

### 3.2 matmul_multi_core 完整分析

**文件**: `/tmp/tt-metal/tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.cpp`

#### 3.2.1 工作分配策略

```cpp
// 获取设备核心网格
auto core_grid = mesh_device->compute_with_storage_grid_size();
auto num_output_tiles_total = (M * N) / TILE_HW;

// 自动分配工作到核心
auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
    split_work_to_cores(core_grid, num_output_tiles_total);
```

#### 3.2.2 多核运行时参数设置

```cpp
uint32_t work_offset = 0;
auto work_groups = {std::make_pair(core_group_1, work_per_core1),
                    std::make_pair(core_group_2, work_per_core2)};

for (const auto& [ranges, work_per_core] : work_groups) {
    for (const auto& range : ranges.ranges()) {
        for (const auto& core : range) {
            // 为每个核心设置不同的运行时参数
            tt_metal::SetRuntimeArgs(program, reader_id, core,
                {src0_addr, src1_addr, Mt, Kt, Nt, work_offset, work_per_core});
            tt_metal::SetRuntimeArgs(program, writer_id, core,
                {dst_addr, work_per_core, work_offset});
            tt_metal::SetRuntimeArgs(program, compute_kernel_id, core,
                {work_per_core, Kt});
            work_offset += work_per_core;
        }
    }
}
```

### 3.3 distributed_eltwise_add 完整分析

**文件**: `/tmp/tt-metal/tt_metal/programming_examples/distributed/3_distributed_eltwise_add/distributed_eltwise_add.cpp`

#### 3.3.1 分布式缓冲区创建

```cpp
// 定义全局缓冲区形状和分片形状
auto shard_shape = Shape2D{32, 32};
auto distributed_buffer_shape = Shape2D{
    shard_shape.height() * mesh_device->num_rows(),
    shard_shape.width() * mesh_device->num_cols()};

// 创建分片缓冲区配置
auto distributed_buffer_config = ShardedBufferConfig{
    .global_size = distributed_buffer_size_bytes,
    .global_buffer_shape = distributed_buffer_shape,
    .shard_shape = shard_shape,
    .shard_orientation = ShardOrientation::ROW_MAJOR};

// 创建设备本地配置
auto local_buffer_config = DeviceLocalBufferConfig{
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::DRAM};

// 创建分布式缓冲区
auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
auto b = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
```

#### 3.3.2 分布式程序创建

```cpp
Program CreateEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles) {
    auto program = CreateProgram();
    auto target_tensix_core = CoreRange(CoreCoord{0, 0});

    // 创建 CB、内核...
    // 使用 TensorAccessorArgs 传递缓冲区访问信息
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*b->get_reference_buffer()).append_to(reader_compile_time_args);

    return program;
}
```

## 4. 编程模式总结

### 4.1 核心编程模型

```
┌─────────────────────────────────────────────────────────────┐
│                        Host Program                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Buffer    │  │   Buffer    │  │       Buffer        │  │
│  │   (Input)   │  │   (Input)   │  │      (Output)       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────▼────────────────▼─────────────────────▼──────────┐  │
│  │                      Program                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │  │
│  │  │    CB0      │  │    CB1      │  │      CB16       │ │  │
│  │  │  (Input 0)  │  │  (Input 1)  │  │    (Output)     │ │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │  │
│  │         │                │                   │          │  │
│  │  ┌──────▼────────────────▼───────────────────▼────────┐ │  │
│  │  │              Compute Kernel (Tensix)                │ │  │
│  │  │     UNPACK ◄─── MATH (FPU/SFPU) ───► PACK          │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 典型代码模板

```cpp
// 1. 设备初始化
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
auto& cq = mesh_device->mesh_command_queue();

// 2. 创建程序
Program program = CreateProgram();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range(mesh_device->shape());

// 3. 定义核心范围
CoreCoord core = {0, 0};  // 或 CoreRange 用于多核

// 4. 创建缓冲区
auto buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

// 5. 创建 Circular Buffers
CreateCircularBuffer(program, core, CircularBufferConfig(size, {{cb_index, data_format}})
    .set_page_size(cb_index, page_size));

// 6. 创建内核
KernelHandle reader = CreateKernel(program, reader_path, core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, ...});
KernelHandle writer = CreateKernel(program, writer_path, core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, ...});
KernelHandle compute = CreateKernel(program, compute_path, core,
    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, ...});

// 7. 设置运行时参数
SetRuntimeArgs(program, reader, core, {src_addr, num_tiles});
SetRuntimeArgs(program, writer, core, {dst_addr, num_tiles});
SetRuntimeArgs(program, compute, core, {num_tiles});

// 8. 写入输入数据
EnqueueWriteMeshBuffer(cq, src_buffer, host_data, false);

// 9. 执行程序
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);

// 10. 读取结果
EnqueueReadMeshBuffer(cq, result_data, dst_buffer, true);

// 11. 清理
mesh_device->close();
```

### 4.3 关键 API 汇总

| 类别 | API | 用途 |
|------|-----|------|
| 设备 | `MeshDevice::create_unit_mesh(id)` | 创建单设备网格 |
| 设备 | `MeshDevice::create(config)` | 创建多设备网格 |
| 缓冲区 | `MeshBuffer::create(config, local_config, device)` | 创建分布式缓冲区 |
| 程序 | `CreateProgram()` | 创建程序 |
| 程序 | `CreateCircularBuffer(program, core, config)` | 创建循环缓冲区 |
| 内核 | `CreateKernel(program, path, core, config)` | 创建内核 |
| 执行 | `SetRuntimeArgs(program, kernel, core, args)` | 设置运行时参数 |
| 执行 | `EnqueueMeshWorkload(cq, workload, blocking)` | 提交工作负载 |
| 数据传输 | `EnqueueWriteMeshBuffer(cq, buffer, data, blocking)` | 写入数据 |
| 数据传输 | `EnqueueReadMeshBuffer(cq, data, buffer, blocking)` | 读取数据 |
| 同步 | `Finish(cq)` | 等待队列完成 |

### 4.4 性能优化要点

1. **双缓冲**: 使用 2+ tiles per CB 实现计算与数据传输重叠
2. **工作分配**: 使用 `split_work_to_cores` 实现负载均衡
3. **编译时参数**: 使用 `compile_args` 而非运行时参数减少开销
4. **Math Fidelity**: 根据精度需求选择合适的 fidelity 级别
5. **追踪捕获**: 使用 MeshTrace 减少重复调度开销
6. **多队列**: 分离数据移动和计算队列实现并行
