# TT-Metalium 编程示例源码深度分析

## 目录

1. [Loopback 示例](#1-loopback-示例)
2. [Matmul Single Core 示例](#2-matmul_single_core-示例)
3. [Matmul Multi Core 示例](#3-matmul_multi_core-示例)
4. [Distributed/Multichip 示例](#4-distributedmultichip-示例)
5. [其他重要示例](#5-其他重要示例)
6. [关键 API 汇总](#6-关键-api-汇总)

---

## 1. Loopback 示例

### 1.1 代码结构

**文件位置**: `tt_metal/programming_examples/loopback/`

```
loopback/
├── loopback.cpp              # Host 端主程序
├── kernels/
│   └── loopback_dram_copy.cpp  # Device 端数据移动内核
└── dram_loopback.md          # 文档
```

### 1.2 功能概述

Loopback 示例演示了最基本的数据传输流程：
- 将数据从 Host DRAM 写入 Device DRAM
- 通过 NoC 将数据从 Device DRAM 读取到 L1 SRAM
- 再从 L1 SRAM 写回 Device DRAM
- 最后将数据读回 Host 进行验证

### 1.3 Host 端代码分析 (`loopback.cpp`)

#### 关键执行流程

```
1. 创建设备网格 (MeshDevice::create_unit_mesh)
2. 获取命令队列 (mesh_command_queue)
3. 配置 Buffer (DeviceLocalBufferConfig)
4. 分配内存 (MeshBuffer::create)
   - L1 Buffer (中间缓存)
   - Input DRAM Buffer (输入数据)
   - Output DRAM Buffer (输出数据)
5. 创建 Program (CreateProgram)
6. 创建 MeshWorkload
7. 创建数据移动内核 (CreateKernel)
8. 生成随机测试数据
9. 写入输入数据 (EnqueueWriteMeshBuffer)
10. 设置运行时参数 (SetRuntimeArgs)
11. 执行程序 (EnqueueMeshWorkload)
12. 等待完成 (Finish)
13. 读取结果 (EnqueueReadMeshBuffer)
14. 验证数据并清理
```

#### 核心代码片段

```cpp
// 1. 创建设备网格
std::shared_ptr<distributed::MeshDevice> mesh_device =
    distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

// 2. 配置 Buffer
distributed::DeviceLocalBufferConfig dram_config{
    .page_size = tile_size_bytes,
    .buffer_type = tt::tt_metal::BufferType::DRAM};
distributed::DeviceLocalBufferConfig l1_config{
    .page_size = tile_size_bytes,
    .buffer_type = tt::tt_metal::BufferType::L1};

// 3. 分配内存
auto l1_buffer = distributed::MeshBuffer::create(
    l1_buffer_config, l1_config, mesh_device.get());
auto input_dram_buffer = distributed::MeshBuffer::create(
    dram_buffer_config, dram_config, mesh_device.get());
auto output_dram_buffer = distributed::MeshBuffer::create(
    dram_buffer_config, dram_config, mesh_device.get());

// 4. 创建 Program 和 Workload
Program program = CreateProgram();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range =
    distributed::MeshCoordinateRange(mesh_device->shape());

// 5. 创建内核
KernelHandle dram_copy_kernel_id = CreateKernel(
    program,
    "loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = dram_copy_compile_time_args});

// 6. 设置运行时参数
SetRuntimeArgs(program, dram_copy_kernel_id, core,
    {l1_buffer->address(), input_dram_buffer->address(),
     output_dram_buffer->address(), num_tiles});

// 7. 执行
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

### 1.4 Device 端代码分析 (`loopback_dram_copy.cpp`)

#### 执行流程

```
1. 从运行时参数获取地址信息
2. 创建 TensorAccessor 用于 DRAM 访问
3. 循环处理每个 tile:
   a. 从 DRAM 异步读取 tile 到 L1
   b. 等待读取完成 (barrier)
   c. 从 L1 异步写入 tile 到 DRAM
   d. 等待写入完成 (barrier)
```

#### 核心代码片段

```cpp
void kernel_main() {
    // 读取运行时参数
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);

    // 创建 TensorAccessor
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);
    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    // 循环处理 tiles
    for (uint32_t i = 0; i < num_tiles; i++) {
        // DRAM -> L1
        noc_async_read_tile(i, in0, l1_buffer_addr);
        noc_async_read_barrier();

        // L1 -> DRAM
        noc_async_write_tile(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }
}
```

### 1.5 关键 API 使用清单

| API | 类型 | 用途 |
|-----|------|------|
| `MeshDevice::create_unit_mesh()` | Host | 创建单设备网格 |
| `mesh_device->mesh_command_queue()` | Host | 获取命令队列 |
| `MeshBuffer::create()` | Host | 分配设备内存 |
| `CreateProgram()` | Host | 创建程序容器 |
| `CreateKernel()` | Host | 创建内核 |
| `SetRuntimeArgs()` | Host | 设置运行时参数 |
| `EnqueueWriteMeshBuffer()` | Host | 写入设备内存 |
| `EnqueueMeshWorkload()` | Host | 执行程序 |
| `Finish()` | Host | 等待完成 |
| `EnqueueReadMeshBuffer()` | Host | 读取设备内存 |
| `get_arg_val<>()` | Device | 获取运行时参数 |
| `TensorAccessor` | Device | DRAM 访问抽象 |
| `noc_async_read_tile()` | Device | 异步读取 tile |
| `noc_async_read_barrier()` | Device | 等待读取完成 |
| `noc_async_write_tile()` | Device | 异步写入 tile |
| `noc_async_write_barrier()` | Device | 等待写入完成 |

---

## 2. Matmul Single Core 示例

### 2.1 代码结构

**文件位置**: `tt_metal/programming_examples/matmul/matmul_single_core/`

```
matmul_single_core/
├── matmul_single_core.cpp           # Host 端主程序
├── matmul_single_core.md            # 文档
└── kernels/
    ├── dataflow/
    │   ├── reader_single_core_mm.cpp   # 读取内核
    │   └── writer_single_core_mm.cpp   # 写入内核
    └── compute/
        └── mm.cpp                      # 计算内核
```

### 2.2 功能概述

单核矩阵乘法演示了：
- 使用 Circular Buffer (CB) 在数据移动内核和计算内核之间传递数据
- 三内核协作：Reader (RISC-V 1) + Compute (FPU) + Writer (RISC-V 0)
- 双缓冲技术实现计算与数据传输重叠
- 分块矩阵乘法 (blocked matmul)

### 2.3 架构流程图

```
┌─────────────────────────────────────────────────────────────┐
│                         Host                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │ Matrix A │  │ Matrix B │  │ Create Program + Kernels │  │
│  │ (MxK)    │  │ (KxN)    │  │ Setup CBs                │  │
│  └────┬─────┘  └────┬─────┘  └───────────┬──────────────┘  │
│       │             │                    │                  │
│       └─────────────┼────────────────────┘                  │
│                     ▼                                       │
│            EnqueueWriteMeshBuffer                           │
│                     │                                       │
│                     ▼                                       │
│            EnqueueMeshWorkload                              │
│                     │                                       │
│                     ▼                                       │
│            EnqueueReadMeshBuffer                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│              Device │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Core {0,0}                         │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │           Reader Kernel (RISC-V 1)             │  │   │
│  │  │  - Read A tiles from DRAM -> CB 0              │  │   │
│  │  │  - Read B tiles from DRAM -> CB 1              │  │   │
│  │  └──────────────┬────────────────────────────────┘  │   │
│  │                 │ CB 0 / CB 1                        │   │
│  │                 ▼                                  │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │          Compute Kernel (FPU)                  │  │   │
│  │  │  - Wait for input tiles in CBs                 │  │   │
│  │  │  - matmul_tiles() accumulation                 │  │   │
│  │  │  - Pack result -> CB 16                        │  │   │
│  │  └──────────────┬────────────────────────────────┘  │   │
│  │                 │ CB 16                              │   │
│  │                 ▼                                  │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │           Writer Kernel (RISC-V 0)             │  │   │
│  │  │  - Read result from CB 16                      │  │   │
│  │  │  - Write to DRAM                               │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Host 端代码分析

#### Circular Buffer 配置

```cpp
// CB 0: 输入 A (双缓冲)
uint32_t src0_cb_index = CBIndex::c_0;
uint32_t num_input_tiles = 2;  // 双缓冲
CircularBufferConfig cb_src0_config =
    CircularBufferConfig(num_input_tiles * single_tile_size,
                        {{src0_cb_index, cb_data_format}})
        .set_page_size(src0_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

// CB 1: 输入 B (双缓冲)
uint32_t src1_cb_index = CBIndex::c_1;
CircularBufferConfig cb_src1_config =
    CircularBufferConfig(num_input_tiles * single_tile_size,
                        {{src1_cb_index, cb_data_format}})
        .set_page_size(src1_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

// CB 16: 输出 C (双缓冲)
uint32_t output_cb_index = tt::CBIndex::c_16;
CircularBufferConfig cb_output_config =
    CircularBufferConfig(num_output_tiles * single_tile_size,
                        {{output_cb_index, cb_data_format}})
        .set_page_size(output_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

#### 内核创建

```cpp
// Reader 内核 (RISC-V 1)
auto reader_id = tt_metal::CreateKernel(
    program,
    "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
    core,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

// Writer 内核 (RISC-V 0)
auto writer_id = tt_metal::CreateKernel(
    program,
    "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
    core,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

// Compute 内核 (FPU)
std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
tt_metal::CreateKernel(
    program,
    "matmul/matmul_single_core/kernels/compute/mm.cpp",
    core,
    tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .compile_args = compute_compile_time_args});
```

### 2.5 Device 端代码分析

#### Reader 内核 (`reader_single_core_mm.cpp`)

```cpp
void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // 创建 TensorAccessor
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, get_tile_size(cb_id_in0));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, get_tile_size(cb_id_in1));

    // 三重循环处理矩阵乘法
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // 读取 A tile (mt, kt)
                uint32_t a_tile_index = mt * Kt + kt;
                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(a_tile_index, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);

                // 读取 B tile (kt, nt)
                uint32_t b_tile_index = kt * Nt + nt;
                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(b_tile_index, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, 1);
            }
        }
    }
}
```

#### Compute 内核 (`mm.cpp`)

```cpp
void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // 初始化矩阵引擎
    mm_init(cb_in0, cb_in1, cb_out);

    // 分块矩阵乘法
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();  // 获取输出寄存器

            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);  // 等待输入 A
                cb_wait_front(cb_in1, 1);  // 等待输入 B

                // 执行矩阵乘法并累加
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                cb_pop_front(cb_in0, 1);  // 释放输入 A
                cb_pop_front(cb_in1, 1);  // 释放输入 B
            }

            tile_regs_commit();  // 提交计算结果
            tile_regs_wait();    // 等待结果就绪

            cb_reserve_back(cb_out, 1);  // 预留输出空间
            pack_tile(0, cb_out);        // 打包结果到 CB
            cb_push_back(cb_out, 1);     // 标记输出可用

            tile_regs_release();  // 释放寄存器
        }
    }
}
```

#### Writer 内核 (`writer_single_core_mm.cpp`)

```cpp
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = 16;
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out0));

    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Nt; ++n) {
            cb_wait_front(cb_id_out0, 1);  // 等待计算结果

            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            noc_async_write_tile(m * Nt + n, s, l1_read_addr);
            noc_async_write_barrier();

            cb_pop_front(cb_id_out0, 1);  // 释放 CB 空间
        }
    }
}
```

### 2.6 CB 使用方式详解

```
┌─────────────────────────────────────────────────────────────┐
│                    Circular Buffer 机制                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Reader Kernel          Compute Kernel        Writer Kernel│
│        │                      │                     │      │
│        │  cb_push_back()     │                     │      │
│        ├────────────────────>│                     │      │
│        │                      │ cb_wait_front()    │      │
│        │                      │<├───────────────────┤      │
│        │                      │                     │      │
│        │                      │ cb_pop_front()     │      │
│        │                      ├────────────────────>│      │
│        │                      │                     │      │
│        │                      │ cb_reserve_back()  │      │
│        │                      │<├───────────────────┤      │
│        │                      │                     │      │
│        │                      │ cb_push_back()     │      │
│        │                      ├────────────────────>│      │
│        │                      │                     │      │
│        │                      │              cb_wait_front()│
│        │                      │<────────────────────┤      │
│                                                             │
│  CB 状态转换:                                               │
│  EMPTY -> RESERVED -> FILLED -> WAITED -> POPPED -> EMPTY  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.7 关键 API 使用清单

| API | 类型 | 用途 |
|-----|------|------|
| `CircularBufferConfig` | Host | CB 配置结构体 |
| `CreateCircularBuffer()` | Host | 创建 CB |
| `mm_init()` | Device | 初始化矩阵引擎 |
| `cb_reserve_back()` | Device | 预留 CB 写入空间 |
| `cb_push_back()` | Device | 标记 CB 数据就绪 |
| `cb_wait_front()` | Device | 等待 CB 数据可用 |
| `cb_pop_front()` | Device | 释放 CB 读取空间 |
| `get_write_ptr()` | Device | 获取 CB 写入地址 |
| `get_read_ptr()` | Device | 获取 CB 读取地址 |
| `tile_regs_acquire()` | Device | 获取计算寄存器 |
| `tile_regs_commit()` | Device | 提交计算结果 |
| `tile_regs_wait()` | Device | 等待计算完成 |
| `tile_regs_release()` | Device | 释放计算寄存器 |
| `matmul_tiles()` | Device | 执行 tile 矩阵乘法 |
| `pack_tile()` | Device | 打包结果到 CB |

---

## 3. Matmul Multi Core 示例

### 3.1 代码结构

**文件位置**: `tt_metal/programming_examples/matmul/matmul_multi_core/`

```
matmul_multi_core/
├── matmul_multi_core.cpp              # Host 端主程序
├── matmul_multi_core.md               # 文档
└── kernels/
    ├── dataflow/
    │   ├── reader_mm_output_tiles_partitioned.cpp  # 分块读取内核
    │   └── writer_unary_interleaved_start_id.cpp   # 分块写入内核
    └── compute/
        └── mm.cpp                                    # 计算内核
```

### 3.2 功能概述

多核矩阵乘法演示了：
- **SPMD (Single Program, Multiple Data)** 并行化策略
- 使用 `split_work_to_cores()` 自动分配负载
- 输出 tile 分区策略：每个核心处理不同的输出 tile 子集
- 运行时参数动态设置每个核心的工作范围

### 3.3 并行策略与负载分配

```
┌─────────────────────────────────────────────────────────────────┐
│                    输出矩阵 C (Mt x Nt)                          │
│                     划分为多个 Tiles                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  ← Core 0        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│  │  7  │  8  │  9  │ 10  │ 11  │ 12  │ 13  │  ← Core 1        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│  │ 14  │ 15  │ 16  │ 17  │ 18  │ 19  │ 20  │  ← Core 2        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│  │ 21  │ 22  │ 23  │ 24  │ 25  │ 26  │ 27  │  ← Core 3        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │
│                                                                 │
│  每个输出 tile 需要读取 A 的一行 tiles 和 B 的一列 tiles        │
│  例如 tile 0 需要: A[0:Mt, 0:Kt] @ B[0:Kt, 0:Nt]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 负载分配代码分析

```cpp
// 获取计算网格大小
auto core_grid = mesh_device->compute_with_storage_grid_size();
auto num_output_tiles_total = (M * N) / TILE_HW;

// 使用 split_work_to_cores 自动分配工作
// 返回:
// - num_cores: 使用的核心数
// - all_cores: 所有使用的核心集合
// - core_group_1: 主核心组 (处理更多 tiles)
// - core_group_2: 次核心组 (处理较少 tiles)
// - work_per_core1: 主组每个核心处理的 tile 数
// - work_per_core2: 次组每个核心处理的 tile 数
auto [num_cores, all_cores, core_group_1, core_group_2,
      work_per_core1, work_per_core2] =
    split_work_to_cores(core_grid, num_output_tiles_total);

// 为每个核心设置运行时参数
uint32_t work_offset = 0;
auto work_groups = {std::make_pair(core_group_1, work_per_core1),
                    std::make_pair(core_group_2, work_per_core2)};

for (const auto& [ranges, work_per_core] : work_groups) {
    for (const auto& range : ranges.ranges()) {
        for (const auto& core : range) {
            // Reader 参数: 包含起始 tile ID 和 tile 数量
            tt_metal::SetRuntimeArgs(
                program, reader_id, core,
                {src0_dram_buffer->address(),
                 src1_dram_buffer->address(),
                 Mt, Kt, Nt,
                 work_offset,        // 起始 tile ID
                 work_per_core});    // 该核心处理的 tile 数

            // Writer 参数
            tt_metal::SetRuntimeArgs(
                program, writer_id, core,
                {dst_dram_buffer->address(),
                 work_per_core,      // 写入 tile 数
                 work_offset});      // 起始 tile ID

            // Compute 参数
            tt_metal::SetRuntimeArgs(
                program, compute_kernel_id, core,
                {work_per_core, Kt});

            work_offset += work_per_core;
        }
    }
}
```

### 3.5 Device 端代码分析

#### Reader 内核 (`reader_mm_output_tiles_partitioned.cpp`)

```cpp
void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(5);  // 起始 tile ID
    uint32_t num_output_tiles = get_arg_val<uint32_t>(6);      // 处理的 tile 数

    // 处理分配给该核心的输出 tiles
    for (uint32_t output_tile = 0; output_tile < num_output_tiles; output_tile++) {
        uint32_t current_tile_id = output_tile_start_id + output_tile;

        // 将线性 tile ID 转换为 2D 坐标
        uint32_t out_row = current_tile_id / Nt;
        uint32_t out_col = current_tile_id % Nt;

        // 读取所有 K tiles 用于该输出位置
        for (uint32_t k = 0; k < Kt; k++) {
            // 读取 A tile (out_row, k)
            uint32_t tile_A = out_row * Kt + k;
            cb_reserve_back(cb_id_in0, 1);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(tile_A, a, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, 1);

            // 读取 B tile (k, out_col)
            uint32_t tile_B = k * Nt + out_col;
            cb_reserve_back(cb_id_in1, 1);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_B, b, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, 1);
        }
    }
}
```

#### Compute 内核 (`mm.cpp`)

```cpp
void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);  // 输出 tile 数
    uint32_t Kt = get_arg_val<uint32_t>(1);                // K 维度 tiles

    mm_init(cb_in0, cb_in1, cb_out);

    // 处理分配给该核心的所有输出 tiles
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
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
```

### 3.6 同步机制

```
┌─────────────────────────────────────────────────────────────────┐
│                      多核同步机制                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  同步级别 1: 单个核心内部                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Reader → CB → Compute → CB → Writer                    │   │
│  │  (通过 CB 的 wait/push/pop 实现同步)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  同步级别 2: Host-Device 之间                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Host: EnqueueMeshWorkload()                            │   │
│  │              ↓                                          │   │
│  │  Device: 所有核心并行执行                                │   │
│  │              ↓                                          │   │
│  │  Host: Finish() 等待所有核心完成                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  注意: 核心之间不需要显式同步，因为它们处理不同的输出 tiles     │
│        没有数据依赖关系                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Distributed/Multichip 示例

### 4.1 代码结构

**文件位置**: `tt_metal/programming_examples/distributed/`

```
distributed/
├── 1_distributed_program_dispatch/      # 基础多设备程序分发
│   ├── distributed_program_dispatch.cpp
│   └── kernels/
│       └── void_kernel.cpp
├── 2_distributed_buffer_rw/             # 分布式 Buffer 读写
│   └── distributed_buffer_rw.cpp
├── 3_distributed_eltwise_add/           # 分布式逐元素加法
│   └── distributed_eltwise_add.cpp
├── 4_distributed_trace_and_events/      # 分布式追踪和事件
├── CMakeLists.txt
└── README.md
```

### 4.2 功能概述

分布式示例演示了：
- **MeshDevice**: 将多个物理设备抽象为统一的逻辑设备网格
- **MeshBuffer**: 跨设备的分布式内存分配
- **MeshWorkload**: 跨设备的程序分发和执行
- **数据分片 (Sharding)**: 将大数据集分片到多个设备

### 4.3 多设备通信架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MeshDevice (2x4 Mesh)                       │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┐              │   │
│  │  │ (0,0)   │ (0,1)   │ (0,2)   │ (0,3)   │  Row 0       │   │
│  │  │ Device0 │ Device1 │ Device2 │ Device3 │              │   │
│  │  ├─────────┼─────────┼─────────┼─────────┤              │   │
│  │  │ (1,0)   │ (1,1)   │ (1,2)   │ (1,3)   │  Row 1       │   │
│  │  │ Device4 │ Device5 │ Device6 │ Device7 │              │   │
│  │  └─────────┴─────────┴─────────┴─────────┘              │   │
│  │                                                          │   │
│  │  • 每个设备有独立的 DRAM 和 L1                           │   │
│  │  • 设备间通过 NoC (Network on Chip) 通信                 │   │
│  │  • Host 通过统一 API 管理所有设备                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 示例 1: 分布式程序分发

```cpp
// 创建 2x4 设备网格
auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
auto& cq = mesh_device->mesh_command_queue();

// 创建程序 (与单设备相同)
auto example_program = CreateProgram();
auto target_tensix_cores = CoreRange{
    CoreCoord{0, 0}, CoreCoord{1, 1}};

auto compute_kernel_id = CreateKernel(
    example_program,
    "kernels/void_kernel.cpp",
    target_tensix_cores,
    ComputeConfig{.compile_args = {}});

SetRuntimeArgs(example_program, compute_kernel_id,
               target_tensix_cores, runtime_args);

// 创建 MeshWorkload 并广播到所有设备
auto mesh_workload = MeshWorkload();
auto target_devices = MeshCoordinateRange(mesh_device->shape());

mesh_workload.add_program(target_devices, std::move(example_program));
EnqueueMeshWorkload(cq, mesh_workload, false);

Finish(cq);
```

### 4.5 示例 2: 分布式 Buffer 读写

```cpp
// 创建 2x4 设备网格
auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
auto& cq = mesh_device->mesh_command_queue();

// 配置分片 Buffer
auto shard_shape = Shape2D{32, 32};
auto distributed_buffer_shape = Shape2D{
    32 * mesh_device->num_rows(),
    32 * mesh_device->num_cols()};

uint32_t tile_size_bytes = tt::tile_size(tt::DataFormat::UInt32);
uint32_t distributed_buffer_size_bytes = 64 * 128 * tile_size_bytes;

// 本地 Buffer 配置
auto local_buffer_config = DeviceLocalBufferConfig{
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::L1};

// 分布式 Buffer 配置
auto distributed_buffer_config = ShardedBufferConfig{
    .global_size = distributed_buffer_size_bytes,
    .global_buffer_shape = distributed_buffer_shape,
    .shard_shape = shard_shape};

// 创建分布式 Buffer (自动分片到所有设备)
auto mesh_buffer = MeshBuffer::create(
    distributed_buffer_config, local_buffer_config, mesh_device.get());

// 写入数据 (自动分发到各设备)
std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(...);
EnqueueWriteMeshBuffer(cq, mesh_buffer, src_data);

// 读取数据 (自动从各设备收集)
std::vector<uint32_t> read_back_data{};
EnqueueReadMeshBuffer(cq, read_back_data, mesh_buffer, true);

// 验证数据
assert(src_data == read_back_data);
```

### 4.6 示例 3: 分布式逐元素加法

```cpp
// 创建分布式 Buffer (分片配置)
auto shard_shape = Shape2D{32, 32};
auto distributed_buffer_shape = Shape2D{
    shard_shape.height() * mesh_device->num_rows(),
    shard_shape.width() * mesh_device->num_cols()};

auto local_buffer_config = DeviceLocalBufferConfig{
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::DRAM};

auto distributed_buffer_config = ShardedBufferConfig{
    .global_size = distributed_buffer_size_bytes,
    .global_buffer_shape = distributed_buffer_shape,
    .shard_shape = shard_shape,
    .shard_orientation = ShardOrientation::ROW_MAJOR};

// 创建输入输出 Buffer
auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
auto b = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

// 准备数据
std::vector<uint32_t> a_data = create_random_vector_of_bfloat16(...);
std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(...);

// 写入数据
EnqueueWriteMeshBuffer(cq, a, a_data, false);
EnqueueWriteMeshBuffer(cq, b, b_data, false);

// 创建程序 (与单设备相同)
auto program = CreateEltwiseAddProgram(a, b, c, tile_size_bytes, num_tiles);

// 广播到所有设备执行
auto mesh_workload = MeshWorkload();
auto device_range = MeshCoordinateRange(mesh_device->shape());
mesh_workload.add_program(device_range, std::move(program));
EnqueueMeshWorkload(cq, mesh_workload, false);

// 读取结果
EnqueueReadMeshBuffer(cq, result_data, c, true);
```

### 4.7 数据分发策略

```
┌─────────────────────────────────────────────────────────────────┐
│                   数据分片策略                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  全局 Buffer (8x8 tiles)                                        │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                     │
│  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │                     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                     │
│  │ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │                     │
│  ├────┼────┼────┼────┼────┼────┼────┼────┤                     │
│  │ ... (分片到 2x2 设备网格)                                  │   │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                     │
│                                                                 │
│  分片结果:                                                      │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   Device (0,0)  │  │   Device (0,1)  │                      │
│  │  ┌────┬────┐   │  │  ┌────┬────┐   │                      │
│  │  │ 0  │ 1  │   │  │  │ 2  │ 3  │   │                      │
│  │  ├────┼────┤   │  │  ├────┼────┤   │                      │
│  │  │ 8  │ 9  │   │  │  │ 10 │ 11 │   │                      │
│  │  └────┴────┘   │  │  └────┴────┘   │                      │
│  └─────────────────┘  └─────────────────┘                      │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   Device (1,0)  │  │   Device (1,1)  │                      │
│  │  ┌────┬────┐   │  │  ┌────┬────┐   │                      │
│  │  │ 4  │ 5  │   │  │  │ 6  │ 7  │   │                      │
│  │  ├────┼────┤   │  │  ├────┼────┤   │                      │
│  │  │ 12 │ 13 │   │  │  │ 14 │ 15 │   │                      │
│  │  └────┴────┘   │  │  └────┴────┘   │                      │
│  └─────────────────┘  └─────────────────┘                      │
│                                                                 │
│  ShardOrientation::ROW_MAJOR: 按行分片                          │
│  ShardOrientation::COL_MAJOR: 按列分片                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.8 关键 API 使用清单

| API | 类型 | 用途 |
|-----|------|------|
| `MeshDevice::create()` | Host | 创建设备网格 |
| `MeshShape(rows, cols)` | Host | 定义网格形状 |
| `MeshCoordinateRange()` | Host | 定义设备坐标范围 |
| `MeshBuffer::create()` | Host | 创建分布式 Buffer |
| `ShardedBufferConfig` | Host | 分片 Buffer 配置 |
| `MeshWorkload` | Host | 分布式工作负载 |
| `add_program()` | Host | 向 workload 添加程序 |
| `EnqueueMeshWorkload()` | Host | 执行分布式工作负载 |
| `EnqueueWriteMeshBuffer()` | Host | 写入分布式 Buffer |
| `EnqueueReadMeshBuffer()` | Host | 读取分布式 Buffer |

---

## 5. 其他重要示例

### 5.1 Eltwise Binary 示例

**位置**: `tt_metal/programming_examples/eltwise_binary/`

演示逐元素二元操作（如加法）的完整流程：
- 使用 3 个 CB (c_0, c_1, c_16) 形成处理流水线
- Reader + Compute + Writer 三内核协作
- 双缓冲实现计算与数据传输重叠

```cpp
// CB 配置 (双缓冲)
CreateCircularBuffer(program, core, CircularBufferConfig(
    tiles_per_cb * tile_size_bytes,
    {{src0_cb_index, tt::DataFormat::Float16_b}})
    .set_page_size(src0_cb_index, tile_size_bytes));
```

### 5.2 Hello World Data Movement 示例

**位置**: `tt_metal/programming_examples/hello_world_datamovement_kernel/`

演示两个 Data Movement 内核同时运行：
- RISCV_0 和 RISCV_1 同时执行
- 使用 DPRINT 输出调试信息

```cpp
// 创建两个 Data Movement 内核
KernelHandle data_movement_kernel_0 = CreateKernel(
    program,
    "kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, ...});

KernelHandle data_movement_kernel_1 = CreateKernel(
    program,
    "kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, ...});
```

### 5.3 VecAdd Multi Core 示例

**位置**: `tt_metal/programming_examples/vecadd_multi_core/`

演示多核向量加法的完整实现：
- 使用 `split_work_to_cores()` 分配工作
- 多核并行处理不同数据段

### 5.4 Tests 目录中的隐藏示例

**位置**: `tt_metal/programming_examples/tests/`

```
tests/
├── mesh_device_test.cpp        # MeshDevice 基础测试
├── out_of_tree_test.cpp        # 外部内核编译测试
└── package_integration_test.cpp # 包集成测试
```

#### Mesh Device Test

```cpp
// 测试创建完整系统网格
auto mesh_shape = distributed::SystemMesh::instance().shape();
auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
std::shared_ptr<distributed::MeshDevice> mesh_device =
    distributed::MeshDevice::create(mesh_device_config);
```

#### Out of Tree Test

演示如何在 TT_METAL_HOME 外部编译和执行内核：
- 动态生成内核文件
- 使用临时目录编译内核

---

## 6. 关键 API 汇总

### 6.1 Host API 分类

#### 设备管理

| API | 功能 |
|-----|------|
| `MeshDevice::create_unit_mesh(device_id)` | 创建单设备网格 |
| `MeshDevice::create(config)` | 创建设备网格 |
| `mesh_device->mesh_command_queue()` | 获取命令队列 |
| `mesh_device->close()` | 关闭设备 |
| `mesh_device->compute_with_storage_grid_size()` | 获取计算网格大小 |

#### Buffer 管理

| API | 功能 |
|-----|------|
| `MeshBuffer::create(config, local_config, device)` | 创建分布式 Buffer |
| `DeviceLocalBufferConfig` | 本地 Buffer 配置 |
| `ReplicatedBufferConfig` | 复制 Buffer 配置 |
| `ShardedBufferConfig` | 分片 Buffer 配置 |
| `buffer->address()` | 获取 Buffer 地址 |

#### Program 管理

| API | 功能 |
|-----|------|
| `CreateProgram()` | 创建程序容器 |
| `CreateKernel(program, file, core, config)` | 创建内核 |
| `SetRuntimeArgs(program, kernel, core, args)` | 设置运行时参数 |
| `DataMovementConfig` | 数据移动内核配置 |
| `ComputeConfig` | 计算内核配置 |

#### Circular Buffer

| API | 功能 |
|-----|------|
| `CreateCircularBuffer(program, core, config)` | 创建 CB |
| `CircularBufferConfig(size, format_spec)` | CB 配置 |
| `set_page_size(cb, size)` | 设置页大小 |
| `CBIndex::c_0` ~ `CBIndex::c_31` | CB 索引 |

#### 执行控制

| API | 功能 |
|-----|------|
| `EnqueueWriteMeshBuffer(cq, buffer, data, blocking)` | 写入 Buffer |
| `EnqueueReadMeshBuffer(cq, data, buffer, blocking)` | 读取 Buffer |
| `MeshWorkload` | 分布式工作负载 |
| `add_program(range, program)` | 添加程序到 workload |
| `EnqueueMeshWorkload(cq, workload, blocking)` | 执行 workload |
| `Finish(cq)` | 等待完成 |

#### 工作分配

| API | 功能 |
|-----|------|
| `split_work_to_cores(grid, num_tiles, row_major)` | 分配工作到核心 |
| `CoreRange(start, end)` | 核心范围 |
| `CoreCoord(x, y)` | 核心坐标 |

### 6.2 Device API 分类

#### 参数获取

| API | 功能 |
|-----|------|
| `get_arg_val<T>(index)` | 获取运行时参数 |
| `get_compile_time_arg_val(index)` | 获取编译时参数 |

#### NoC 数据传输

| API | 功能 |
|-----|------|
| `noc_async_read_tile(tile_id, accessor, dst_addr)` | 异步读取 tile |
| `noc_async_read_barrier()` | 等待读取完成 |
| `noc_async_write_tile(tile_id, accessor, src_addr)` | 异步写入 tile |
| `noc_async_write_barrier()` | 等待写入完成 |
| `noc_async_write_flushed()` | 等待写入请求发送 |
| `TensorAccessor` | DRAM 访问抽象 |
| `TensorAccessorArgs<offset>()` | 编译时参数模板 |

#### Circular Buffer 操作

| API | 功能 |
|-----|------|
| `cb_reserve_back(cb_id, num_tiles)` | 预留写入空间 |
| `cb_push_back(cb_id, num_tiles)` | 标记数据就绪 |
| `cb_wait_front(cb_id, num_tiles)` | 等待数据可用 |
| `cb_pop_front(cb_id, num_tiles)` | 释放读取空间 |
| `get_write_ptr(cb_id)` | 获取写入地址 |
| `get_read_ptr(cb_id)` | 获取读取地址 |
| `get_tile_size(cb_id)` | 获取 tile 大小 |

#### 计算操作

| API | 功能 |
|-----|------|
| `mm_init(cb_in0, cb_in1, cb_out)` | 初始化矩阵引擎 |
| `matmul_tiles(cb_a, cb_b, dst, a_tile, b_tile)` | 矩阵乘法 |
| `add_tiles(cb_a, cb_b, dst, a_tile, b_tile)` | 逐元素加法 |
| `pack_tile(src, cb_dst)` | 打包结果到 CB |
| `tile_regs_acquire()` | 获取计算寄存器 |
| `tile_regs_commit()` | 提交计算结果 |
| `tile_regs_wait()` | 等待计算完成 |
| `tile_regs_release()` | 释放计算寄存器 |

### 6.3 数据类型和常量

| 名称 | 说明 |
|------|------|
| `bfloat16` | Brain Float 16 数据类型 |
| `TILE_HEIGHT = 32` | Tile 高度 |
| `TILE_WIDTH = 32` | Tile 宽度 |
| `TILE_HW = 1024` | Tile 元素总数 |
| `DataFormat::Float16_b` | BFloat16 格式 |
| `DataFormat::Float32` | Float32 格式 |
| `MathFidelity::HiFi4` | 高精度数学模式 |
| `BufferType::DRAM` | DRAM 内存类型 |
| `BufferType::L1` | L1 SRAM 内存类型 |
| `DataMovementProcessor::RISCV_0` | RISC-V 0 处理器 |
| `DataMovementProcessor::RISCV_1` | RISC-V 1 处理器 |
| `NOC::RISCV_0_default` | RISC-V 0 默认 NoC |
| `NOC::RISCV_1_default` | RISC-V 1 默认 NoC |

---

## 附录: 示例代码索引

### 官方编程示例

| 示例名称 | 路径 | 主要演示内容 |
|----------|------|--------------|
| loopback | `programming_examples/loopback/` | 基础数据传输 |
| matmul_single_core | `programming_examples/matmul/matmul_single_core/` | 单核矩阵乘法 |
| matmul_multi_core | `programming_examples/matmul/matmul_multi_core/` | 多核矩阵乘法 |
| matmul_multicore_reuse | `programming_examples/matmul/matmul_multicore_reuse/` | CB 复用优化 |
| matmul_multicore_reuse_mcast | `programming_examples/matmul/matmul_multicore_reuse_mcast/` | 多播优化 |
| eltwise_binary | `programming_examples/eltwise_binary/` | 逐元素二元操作 |
| eltwise_sfpu | `programming_examples/eltwise_sfpu/` | SFPU 逐元素操作 |
| hello_world_datamovement_kernel | `programming_examples/hello_world_datamovement_kernel/` | Data Movement 内核 |
| hello_world_compute_kernel | `programming_examples/hello_world_compute_kernel/` | Compute 内核 |
| hello_world_datatypes_kernel | `programming_examples/hello_world_datatypes_kernel/` | 数据类型 |
| vecadd_multi_core | `programming_examples/vecadd_multi_core/` | 多核向量加法 |
| distributed/1_distributed_program_dispatch | `programming_examples/distributed/1_distributed_program_dispatch/` | 多设备程序分发 |
| distributed/2_distributed_buffer_rw | `programming_examples/distributed/2_distributed_buffer_rw/` | 分布式 Buffer |
| distributed/3_distributed_eltwise_add | `programming_examples/distributed/3_distributed_eltwise_add/` | 分布式计算 |
| distributed/4_distributed_trace_and_events | `programming_examples/distributed/4_distributed_trace_and_events/` | 追踪和事件 |
| add_2_integers_in_compute | `programming_examples/add_2_integers_in_compute/` | 基础计算 |
| add_2_integers_in_riscv | `programming_examples/add_2_integers_in_riscv/` | RISC-V 计算 |
| NoC_tile_transfer | `programming_examples/NoC_tile_transfer/` | NoC 传输 |
| pad_multi_core | `programming_examples/pad_multi_core/` | 多核填充 |
| shard_data_rm | `programming_examples/shard_data_rm/` | 数据分片 |
| vecadd_sharding | `programming_examples/vecadd_sharding/` | 分片向量加法 |
| sfpu_eltwise_chain | `programming_examples/sfpu_eltwise_chain/` | SFPU 操作链 |
| custom_sfpi_add | `programming_examples/custom_sfpi_add/` | 自定义 SFPI |
| custom_sfpi_smoothstep | `programming_examples/custom_sfpi_smoothstep/` | 自定义 SFPI |

### 测试示例

| 示例名称 | 路径 | 主要演示内容 |
|----------|------|--------------|
| mesh_device_test | `programming_examples/tests/mesh_device_test.cpp` | MeshDevice 测试 |
| out_of_tree_test | `programming_examples/tests/out_of_tree_test.cpp` | 外部编译测试 |
| package_integration_test | `programming_examples/tests/package_integration_test.cpp` | 包集成测试 |

---

*文档生成时间: 2026-03-12*
*基于 tt-metal 仓库 main 分支*
