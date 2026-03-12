# 附录

---

## 附录 A: API 快速索引

### A.1 Host API 索引

#### 设备管理 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `CreateDevice` | 第3章 Host API | 创建并初始化设备实例 |
| `CreateDeviceMinimal` | 第3章 Host API | 创建最小化设备实例（用于故障恢复） |
| `CloseDevice` | 第3章 Host API | 关闭设备并释放资源 |
| `GetNumAvailableDevices` | 第3章 Host API | 获取可用设备数量 |
| `GetNumPCIeDevices` | 第3章 Host API | 获取 PCIe 设备数量 |
| `IsGalaxyCluster` | 第3章 Host API | 检测是否为 Galaxy 集群 |
| `ReleaseOwnership` | 第3章 Host API | 释放 MetalContext 所有权 |
| `SetRootDir` | 第3章 Host API | 设置 TT Metal 根目录 |

#### Buffer 管理 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `CreateBuffer` (Interleaved) | 第3章 Host API | 创建交织缓冲区 |
| `CreateBuffer` (Sharded) | 第3章 Host API | 创建分片缓冲区 |
| `DeallocateBuffer` | 第3章 Host API | 释放缓冲区 |
| `AssignGlobalBufferToProgram` | 第3章 Host API | 将全局缓冲区分配给程序 |

#### Kernel 创建 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `CreateKernel` | 第3章 Host API | 从文件创建 Kernel |
| `CreateKernelFromString` | 第3章 Host API | 从源代码字符串创建 Kernel |

#### 程序执行 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `EnqueueProgram` | 第3章 Host API | 将程序入队到命令队列 |
| `Finish` | 第3章 Host API | 等待命令队列完成 |
| `LaunchProgram` | 第3章 Host API | 直接启动程序 |
| `CompileProgram` | 第3章 Host API | 显式编译程序 |
| `WaitProgramDone` | 第3章 Host API | 等待程序执行完成 |

#### 运行时参数 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `SetRuntimeArgs` | 第3章 Host API | 设置 Kernel 运行时参数 |
| `SetCommonRuntimeArgs` | 第3章 Host API | 设置通用运行时参数 |
| `GetRuntimeArgs` | 第3章 Host API | 获取运行时参数 |
| `GetCommonRuntimeArgs` | 第3章 Host API | 获取通用运行时参数 |

#### 子设备管理 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `create_sub_device_manager` | 第3章 Host API | 创建子设备管理器 |
| `load_sub_device_manager` | 第3章 Host API | 加载子设备管理器 |
| `remove_sub_device_manager` | 第3章 Host API | 移除子设备管理器 |
| `set_sub_device_stall_group` | 第3章 Host API | 设置子设备停顿组 |
| `get_sub_device_ids` | 第3章 Host API | 获取子设备 ID 列表 |

#### Event/Semaphore API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `CreateSemaphore` | 第3章 Host API | 创建信号量 |
| `CreateGlobalSemaphore` | 第3章 Host API | 创建全局信号量 |

#### 直接内存访问 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `WriteToDeviceL1` | 第3章 Host API | 直接写入设备 L1 内存 |
| `ReadFromDeviceL1` | 第3章 Host API | 从设备 L1 内存读取 |
| `WriteToDeviceDRAMChannel` | 第3章 Host API | 直接写入 DRAM 通道 |
| `ReadFromDeviceDRAMChannel` | 第3章 Host API | 从 DRAM 通道读取 |

#### Circular Buffer 管理 API (Host)

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `CreateCircularBuffer` | 第5章 CB API | 创建 Circular Buffer |
| `GetCircularBufferConfig` | 第5章 CB API | 获取 CB 配置 |
| `UpdateCircularBufferTotalSize` | 第5章 CB API | 更新 CB 总大小 |
| `UpdateCircularBufferPageSize` | 第5章 CB API | 更新 CB 页大小 |
| `UpdateDynamicCircularBufferAddress` | 第5章 CB API | 更新动态 CB 地址 |
| `UpdateDynamicCircularBufferAddressAndTotalSize` | 第5章 CB API | 同时更新地址和大小 |

---

### A.2 Device API 索引

#### NoC 读写操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_async_read` | 第4章 Data Movement API | 异步读取数据到 L1 |
| `noc_async_write` | 第4章 Data Movement API | 异步写入数据从 L1 |
| `noc_async_read_barrier` | 第4章 Data Movement API | 等待读取完成 |
| `noc_async_write_barrier` | 第4章 Data Movement API | 等待写入完成 |
| `noc_async_full_barrier` | 第4章 Data Movement API | 等待所有 NoC 操作完成 |

#### 单包操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_async_read_one_packet` | 第4章 Data Movement API | 单包异步读取 |
| `noc_async_read_one_packet_set_state` | 第4章 Data Movement API | 设置单包读取状态 |
| `noc_async_read_one_packet_with_state` | 第4章 Data Movement API | 使用状态单包读取 |
| `noc_async_write_one_packet` | 第4章 Data Movement API | 单包异步写入 |
| `noc_async_write_one_packet_set_state` | 第4章 Data Movement API | 设置单包写入状态 |
| `noc_async_write_one_packet_with_state` | 第4章 Data Movement API | 使用状态单包写入 |

#### 多播操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_async_write_multicast` | 第4章 Data Movement API | 多播写入数据 |
| `noc_async_write_multicast_loopback_src` | 第4章 Data Movement API | 多播写入并回环源 |

#### 页面操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_async_read_page` | 第4章 Data Movement API | 基于页面 ID 读取 |
| `noc_async_write_page` | 第4章 Data Movement API | 基于页面 ID 写入 |

#### 分片操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_async_read_shard` | 第4章 Data Movement API | 从分片张量读取 |
| `noc_async_write_shard` | 第4章 Data Movement API | 写入分片张量 |

#### 信号量操作

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `noc_semaphore_set` | 第4章 Data Movement API | 设置信号量值 |
| `noc_semaphore_inc` | 第4章 Data Movement API | 增加信号量值 |
| `noc_semaphore_wait` | 第4章 Data Movement API | 等待信号量值 |
| `noc_semaphore_wait_min` | 第4章 Data Movement API | 等待最小值 |
| `noc_semaphore_set_multicast` | 第4章 Data Movement API | 多播设置信号量 |
| `noc_semaphore_set_multicast_loopback_src` | 第4章 Data Movement API | 多播设置并回环 |
| `noc_semaphore_set_remote` | 第4章 Data Movement API | 远程设置信号量 |
| `get_semaphore` | 第4章 Data Movement API | 获取信号量地址 |

#### 地址函数

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `get_noc_addr` | 第4章 Data Movement API | 获取核心 L1 的 NoC 地址 |
| `get_noc_addr_from_bank_id` | 第4章 Data Movement API | 从 bank ID 获取地址 |
| `get_noc_multicast_addr` | 第4章 Data Movement API | 获取多播地址 |
| `get_dram_noc_addr` | 第4章 Data Movement API | 获取 DRAM NoC 地址 |

#### Circular Buffer 操作 (Device)

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `cb_reserve_back` | 第5章 CB API | 预留 CB 后端空间 |
| `cb_push_back` | 第5章 CB API | 推送数据到 CB |
| `cb_wait_front` | 第5章 CB API | 等待 CB 前端数据 |
| `cb_pop_front` | 第5章 CB API | 弹出 CB 前端数据 |
| `get_write_ptr` | 第5章 CB API | 获取写入地址 |
| `get_read_ptr` | 第5章 CB API | 获取读取地址 |
| `cb_pages_reservable_at_back` | 第5章 CB API | 检查可预留页数 |
| `cb_pages_available_at_front` | 第5章 CB API | 检查可用页数 |

#### Tile 信息查询

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `get_tile_size` | 第4章 Data Movement API | 获取 Tile 大小 |
| `get_tile_hw` | 第4章 Data Movement API | 获取 Tile 高宽 |
| `get_tile_num_faces` | 第4章 Data Movement API | 获取 Tile 面数 |
| `get_dataformat` | 第4章 Data Movement API | 获取数据格式 |

#### 核心坐标与参数访问

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `get_absolute_logical_x/y` | 第4章 Data Movement API | 获取绝对逻辑坐标 |
| `get_relative_logical_x/y` | 第4章 Data Movement API | 获取相对逻辑坐标 |
| `get_arg_val` | 第4章 Data Movement API | 获取运行时参数值 |
| `get_common_arg_val` | 第4章 Data Movement API | 获取通用参数值 |
| `get_arg_addr` | 第4章 Data Movement API | 获取参数地址 |

#### 计算操作 API

| 函数名 | 所在章节 | 简要描述 |
|--------|----------|----------|
| `mm_init` | 第6章 Compute API | 初始化矩阵乘法 |
| `matmul_tiles` | 第6章 Compute API | 执行 Tile 矩阵乘法 |
| `add_tiles` | 第6章 Compute API | Tile 加法 |
| `sub_tiles` | 第6章 Compute API | Tile 减法 |
| `mul_tiles` | 第6章 Compute API | Tile 乘法 |
| `relu_tile` | 第6章 Compute API | ReLU 激活 |
| `sigmoid_tile` | 第6章 Compute API | Sigmoid 激活 |
| `gelu_tile` | 第6章 Compute API | GELU 激活 |
| `exp_tile` | 第6章 Compute API | 指数运算 |
| `log_tile` | 第6章 Compute API | 对数运算 |
| `sqrt_tile` | 第6章 Compute API | 平方根运算 |
| `recip_tile` | 第6章 Compute API | 倒数运算 |
| `pack_tile` | 第6章 Compute API | 打包 Tile 到 CB |
| `tile_regs_acquire` | 第6章 Compute API | 获取 Tile 寄存器 |
| `tile_regs_commit` | 第6章 Compute API | 提交 Tile 寄存器 |
| `tile_regs_wait` | 第6章 Compute API | 等待 Tile 寄存器 |
| `tile_regs_release` | 第6章 Compute API | 释放 Tile 寄存器 |

---

## 附录 B: 常见任务速查表

### B.1 设备初始化流程

```cpp
// 1. 检查可用设备
size_t num_devices = GetNumAvailableDevices();
if (num_devices == 0) {
    throw std::runtime_error("No devices found");
}

// 2. 创建设备实例
IDevice* device = CreateDevice(0);  // 设备 ID 从 0 开始

// 3. 获取命令队列
CommandQueue& cq = device->command_queue();

// 4. 使用设备...

// 5. 关闭设备
CloseDevice(device);
```

**完整配置选项：**
```cpp
IDevice* device = CreateDevice(
    device_id,           // 设备 ID
    num_hw_cqs,          // 硬件命令队列数量 (默认 1)
    l1_small_size,       // L1 小缓冲区大小 (默认 0)
    trace_region_size,   // Trace 区域大小 (默认 0)
    dispatch_core_config,// 分发核心配置
    l1_bank_remap,       // L1 Bank 重映射表
    worker_l1_size       // Worker L1 大小
);
```

---

### B.2 Buffer 创建

#### Interleaved Buffer (交织缓冲区)

```cpp
// DRAM 缓冲区
InterleavedBufferConfig dram_config{
    .device = device,
    .size = 1024 * 1024,        // 1 MB
    .page_size = 2048,          // 2 KB 页
    .buffer_type = BufferType::DRAM
};
auto dram_buffer = CreateBuffer(dram_config);

// L1 缓冲区
InterleavedBufferConfig l1_config{
    .device = device,
    .size = 32768,              // 32 KB
    .page_size = 2048,
    .buffer_type = BufferType::L1
};
auto l1_buffer = CreateBuffer(l1_config);
```

#### Sharded Buffer (分片缓冲区)

```cpp
// 定义分片规格
ShardSpec shard_spec{
    .grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))),
    .shape = {32, 32},
    .orientation = ShardOrientation::ROW_MAJOR
};

ShardSpecBuffer shard_spec_buffer{
    .tensor_shard_spec = shard_spec,
    .page_shape = {32, 32},
    .tensor2d_shape_in_pages = {8, 8}
};

ShardedBufferConfig sharded_config{
    .device = device,
    .size = 1024 * 1024,
    .page_size = 2048,
    .buffer_type = BufferType::L1,
    .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
    .shard_parameters = shard_spec_buffer
};

auto sharded_buffer = CreateBuffer(sharded_config);
```

**布局类型对比：**

| 布局类型 | 适用场景 | 特点 |
|----------|----------|------|
| `INTERLEAVED` | 通用场景 | 数据均匀分布在所有 bank |
| `HEIGHT_SHARDED` | 行并行计算 | 按行分片，每核心处理部分行 |
| `WIDTH_SHARDED` | 列并行计算 | 按列分片，每核心处理部分列 |
| `BLOCK_SHARDED` | 2D 并行计算 | 按块分片，适合矩阵运算 |

---

### B.3 Kernel 创建

#### Data Movement Kernel

```cpp
// Reader Kernel (BRISC - RISCV_0)
auto reader = CreateKernel(
    program,
    "kernels/dataflow/reader.cpp",
    CoreCoord(0, 0),
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = {num_tiles, tile_size}
    }
);

// Writer Kernel (NCRISC - RISCV_1)
auto writer = CreateKernel(
    program,
    "kernels/dataflow/writer.cpp",
    CoreCoord(0, 0),
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = {num_tiles, tile_size}
    }
);
```

#### Compute Kernel

```cpp
auto compute = CreateKernel(
    program,
    "kernels/compute/matmul.cpp",
    CoreCoord(0, 0),
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = {Mt, Kt, Nt}
    }
);
```

**MathFidelity 选择：**

| 精度级别 | 描述 | 适用场景 |
|----------|------|----------|
| `LoFi` | 最低精度，最高性能 | 推理，容忍精度损失 |
| `HiFi2` | 中等精度 | 平衡性能和精度 |
| `HiFi3` | 较高精度 | 训练，需要较好精度 |
| `HiFi4` | 最高精度，较低性能 | 训练，需要最高精度 |

#### Ethernet Kernel

```cpp
EthernetConfig eth_config;
eth_config.eth_mode = Eth::SENDER;  // 或 Eth::RECEIVER
eth_config.processor = DataMovementProcessor::RISCV_0;
eth_config.noc = NOC::RISCV_0_default;

auto eth_kernel = CreateKernel(
    program,
    "kernels/eth/eth_sender.cpp",
    eth_core,
    eth_config
);
```

---

### B.4 Circular Buffer 配置

#### 基本配置

```cpp
// 计算 Tile 大小
uint32_t tile_size = 32 * 32 * 2;  // Float16_b: 2048 bytes

// 创建 CB 配置
CircularBufferConfig cb_config(
    num_tiles * tile_size,                    // 总大小
    {{cb_index, tt::DataFormat::Float16_b}}   // 数据格式
).set_page_size(cb_index, tile_size);

// 创建 CB
CBHandle cb = CreateCircularBuffer(program, core_spec, cb_config);
```

#### 双缓冲配置

```cpp
// 双缓冲：允许重叠计算和通信
uint32_t num_tiles_double_buffer = 2;
uint32_t cb_size = num_tiles_double_buffer * tile_size;

CircularBufferConfig cb_config(
    cb_size,
    {{0, tt::DataFormat::Float16_b}}
).set_page_size(0, tile_size);
```

#### 多 CB 配置

```cpp
// 输入 CB 0 和 1，输出 CB 16
CircularBufferConfig in0_config(
    8192,  // 4 tiles * 2048
    {{0, tt::DataFormat::Float16_b}}
).set_page_size(0, 2048);

CircularBufferConfig in1_config(
    8192,
    {{1, tt::DataFormat::Float16_b}}
).set_page_size(1, 2048);

CircularBufferConfig out_config(
    8192,
    {{16, tt::DataFormat::Float16_b}}
).set_page_size(16, 2048);

CBHandle cb_in0 = CreateCircularBuffer(program, core, in0_config);
CBHandle cb_in1 = CreateCircularBuffer(program, core, in1_config);
CBHandle cb_out = CreateCircularBuffer(program, core, out_config);
```

**数据格式选择：**

| 格式 | 每元素字节 | 适用场景 |
|------|-----------|----------|
| `Float32` | 4 | 高精度计算 |
| `Float16_b` | 2 | 训练（推荐）|
| `Float16` | 2 | 通用计算 |
| `Bfp8_b` | 1 | 推理优化 |
| `Bfp4_b` | 0.5 | 极致性能 |
| `Int32` | 4 | 整数运算 |

---

### B.5 程序执行流程

```cpp
// 1. 创建程序
Program program = CreateProgram();

// 2. 创建 Kernels
auto reader = CreateKernel(program, "reader.cpp", core, reader_config);
auto writer = CreateKernel(program, "writer.cpp", core, writer_config);
auto compute = CreateKernel(program, "compute.cpp", core, compute_config);

// 3. 创建 Circular Buffers
CBHandle cb = CreateCircularBuffer(program, core, cb_config);

// 4. 设置运行时参数
SetRuntimeArgs(program, reader, core, {num_tiles, src_addr, dst_addr});
SetRuntimeArgs(program, writer, core, {num_tiles, dst_addr});
SetRuntimeArgs(program, compute, core, {Mt, Kt, Nt});

// 5. 执行程序
CommandQueue& cq = device->command_queue();
EnqueueProgram(cq, program, false);  // 非阻塞
// ... 可以执行其他操作 ...
Finish(cq);  // 等待完成

// 或者阻塞执行
EnqueueProgram(cq, program, true);  // 阻塞等待完成
```

**多核心参数设置：**

```cpp
// 不同核心不同参数
std::vector<CoreCoord> cores;
std::vector<std::vector<uint32_t>> args;
for (uint32_t y = 0; y < 8; y++) {
    for (uint32_t x = 0; x < 8; x++) {
        cores.push_back(CoreCoord(x, y));
        args.push_back({
            per_core_M, per_core_N, per_core_K,
            y * per_core_M * Kt,
            x * per_core_N
        });
    }
}
SetRuntimeArgs(program, kernel, cores, args);
```

---

### B.6 调试环境变量汇总

#### Tracy Profiler

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `TT_METAL_TRACY` | `1` | 启用 Tracy 分析 |

**使用：**
```bash
export TT_METAL_TRACY=1
./your_application
# 启动 Tracy GUI: tracy-profiler
```

#### Device Profiler

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `TT_METAL_DEVICE_PROFILER` | `1` | 启用设备性能分析 |

**使用：**
```bash
export TT_METAL_DEVICE_PROFILER=1
./your_application
# 结果在 ${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv
```

#### Watcher

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `TT_METAL_WATCHER` | `<seconds>` | 启用 Watcher，设置轮询间隔 |
| `TT_METAL_WATCHER_APPEND` | `1` | 追加到现有日志 |
| `TT_METAL_WATCHER_DUMP_ALL` | `1` | 转储所有状态 |
| `TT_METAL_WATCHER_DISABLE_ASSERT` | `1` | 禁用断言检查 |
| `TT_METAL_WATCHER_DISABLE_PAUSE` | `1` | 禁用暂停功能 |
| `TT_METAL_WATCHER_DISABLE_RING_BUFFER` | `1` | 禁用环形缓冲区 |
| `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE` | `1` | 禁用 NOC 清理 |
| `TT_METAL_WATCHER_DISABLE_WAYPOINT` | `1` | 禁用路点跟踪 |
| `TT_METAL_WATCHER_DISABLE_STACK_USAGE` | `1` | 禁用堆栈使用测量 |

**使用：**
```bash
export TT_METAL_WATCHER=120  # 每 120 秒检查一次
./your_application
```

#### DPRINT (Kernel Debug Print)

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `TT_METAL_DPRINT_CORES` | `all` | 打印所有核心 |
| `TT_METAL_DPRINT_CORES` | `x,y` | 打印指定核心，如 `1,1` |
| `TT_METAL_DPRINT_CORES` | `x0,y0;x1,y1` | 打印多个核心 |

**使用：**
```bash
export TT_METAL_DPRINT_CORES=all
./your_application
```

**重要提示：** Tracy、Device Profiler、DPRINT 和 Watcher 互斥，不能同时启用。

---

## 附录 C: 术语表

### C.1 硬件术语

| 术语 | 英文 | 定义 |
|------|------|------|
| Tensix 核心 | Tensix Core | Tenstorrent 芯片的基本计算单元，包含 5 个 RISC-V 核心和计算引擎 |
| BRISC | BRISC | Tensix 中的数据移动核心 (RISC-V 0)，通常作为 Reader |
| NCRISC | NCRISC | Tensix 中的数据移动核心 (RISC-V 1)，通常作为 Writer |
| TRISC | TRISC | Tensix 中的计算核心组 (RISC-V 2-4)，包含 Unpack/Math/Pack |
| ERISC | ERISC | 以太网核心，用于芯片间通信 |
| NoC | Network-on-Chip | 片上网络，用于核心间和核心与 DRAM 间的数据传输 |
| NOC_0 / NOC_1 | NOC 0/1 | 双 NoC 通道，可并行传输提高带宽 |
| Tile | Tile | 基本数据单位，默认 32×32 元素 |
| Face | Face | Tile 的子单元，通常是 16×16 元素 |
| CB | Circular Buffer | 循环缓冲区，位于 L1 SRAM，用于 Kernel 间数据传递 |
| L1 SRAM | L1 SRAM | Tensix 核心上的高速片上内存 (~1.5 MB) |
| DRAM | DRAM | 设备外部内存 (GDDR6)，容量大但速度较慢 |
| Bank | Bank | DRAM 或 L1 的存储分区单元 |
| Mesh | Mesh | 多芯片拓扑结构，支持 2D/3D 网格配置 |
| Galaxy | Galaxy | Tenstorrent 的大规模多芯片系统 |

### C.2 软件术语

| 术语 | 英文 | 定义 |
|------|------|------|
| Kernel | Kernel | 在 Tensix 核心上运行的程序，分为 Data Movement、Compute、Ethernet 三类 |
| Program | Program | 包含多个 Kernel 和 CB 配置的执行单元 |
| Runtime Args | Runtime Arguments | 运行时传递给 Kernel 的参数 |
| Compile Args | Compile Arguments | 编译时传递给 Kernel 的参数 |
| Command Queue | Command Queue | 主机向设备发送命令的队列 |
| Enqueue | Enqueue | 将操作加入命令队列 |
| Barrier | Barrier | 同步点，等待操作完成 |
| Semaphore | Semaphore | 信号量，用于核心间同步 |
| Dispatch | Dispatch | 命令分发机制，将命令发送到设备核心 |
| Trace | Trace | 记录命令序列用于重放，减少主机开销 |
| Sub-Device | Sub-Device | 设备的逻辑分区，支持多租户隔离 |

### C.3 编程术语

| 术语 | 英文 | 定义 |
|------|------|------|
| Interleaved | Interleaved | 数据交织布局，均匀分布在多个 bank |
| Sharded | Sharded | 分片布局，数据按特定方式分片到不同核心 |
| Height Sharded | Height Sharded | 按高度（行）方向分片 |
| Width Sharded | Width Sharded | 按宽度（列）方向分片 |
| Block Sharded | Block Sharded | 按 2D 块方式分片 |
| Tilize | Tilize | 将线性数据转换为 Tile 格式 |
| Untilize | Untilize | 将 Tile 格式转换为线性数据 |
| Pack | Pack | 将计算结果从寄存器打包到 CB |
| Unpack | Unpack | 从 CB 解包数据到寄存器 |
| Math Fidelity | Math Fidelity | 数学运算精度级别 (LoFi/HiFi2/HiFi3/HiFi4) |
| SFPU | Special Function Unit | 特殊函数单元，执行激活函数等 |
| FPU | Float Point Unit | 浮点运算单元 |
| Data Format | Data Format | 数据类型格式 (Float16_b, Bfp8_b 等) |
| Page | Page | CB 中的数据页，通常包含一个或多个 Tile |
| Multicast | Multicast | 多播操作，同时向多个目标发送数据 |
| Producer | Producer | 数据生产者，向 CB 写入数据 |
| Consumer | Consumer | 数据消费者，从 CB 读取数据 |

---

## 附录 D: 错误代码参考

### D.1 主机端错误代码

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `TT_ASSERT @ ... failed` | 断言失败，参数或状态不符合预期 | 检查输入参数和前置条件 |
| `Device not found` | 设备未连接或驱动问题 | 检查硬件连接、驱动安装、设备权限 |
| `Out of memory` | 内存分配失败 | 减少缓冲区大小、分批处理、检查内存泄漏 |
| `Timeout waiting for kernel` | 内核死锁或无限循环 | 使用 Watcher 检查路点、检查 CB 同步 |
| `NOC address out of range` | 无效 NoC 地址 | 验证坐标计算、检查地址范围 |
| `Buffer allocation failed` | 缓冲区分配失败 | 检查可用内存、减少缓冲区大小 |
| `Invalid core coordinate` | 无效核心坐标 | 验证坐标在有效范围内 |
| `Program compilation failed` | 程序编译失败 | 检查 Kernel 代码语法、检查编译参数 |
| `Command queue full` | 命令队列已满 | 等待队列清空、减少并发命令 |
| `PCIe transfer error` | PCIe 传输错误 | 检查硬件连接、重启设备 |

### D.2 设备端错误代码

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `CB overflow` | Circular Buffer 溢出 | 增加 CB 大小、减少推送页数、检查 push/pop 平衡 |
| `CB underflow` | Circular Buffer 下溢 | 检查 pop 操作前是否有足够数据、检查生产者速度 |
| `Stack overflow` | 堆栈溢出 | 减少局部变量、避免大数组在栈上分配 |
| `NOC transaction error` | NoC 传输错误 | 检查地址对齐、检查传输大小、验证目标地址 |
| `Invalid kernel args` | 内核参数错误 | 验证运行时参数数量和类型 |
| `Invalid CB index` | 无效 CB 索引 | 确保 CB 索引在 0-31 范围内 |
| `Alignment error` | 内存对齐错误 | 确保地址 16 字节对齐 |
| `Watchdog timeout` | 看门狗超时 | 检查无限循环、添加适当同步点 |
| `Invalid data format` | 无效数据格式 | 检查 CB 配置的数据格式是否支持 |
| `Tile size mismatch` | Tile 大小不匹配 | 确保 CB page_size 与实际数据大小一致 |

### D.3 同步错误代码

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `Semaphore timeout` | 信号量等待超时 | 检查信号量初始值、验证递增逻辑、检查死锁 |
| `Barrier mismatch` | 屏障同步失败 | 确保所有核心到达屏障、检查条件分支 |
| `Deadlock detected` | 死锁 | 使用 Watcher 检查路点、检查 CB 操作顺序 |
| `Race condition` | 竞态条件 | 添加适当同步、使用信号量协调 |
| `Sync point error` | 同步点错误 | 检查同步点放置位置、确保成对使用 |

### D.4 调试错误排查流程

```
程序崩溃/异常
    │
    ├── 主机端崩溃？
    │       ├── 是 → 检查 TT_ASSERT 消息
    │       │           ├── 参数错误 → 验证输入
    │       │           ├── 空指针 → 检查初始化
    │       │           └── 其他 → 查看堆栈跟踪
    │       └── 否 → 继续
    │
    ├── 设备端挂起？
    │       ├── 是 → 启用 Watcher
    │       │           ├── 查看路点状态
    │       │           ├── 检查 CB 状态
    │       │           └── 验证 NOC 事务
    │       └── 否 → 继续
    │
    ├── 结果错误？
    │       ├── 是 → 启用 DPRINT
    │       │           ├── 打印中间值
    │       │           ├── 检查 Tile 数据
    │       │           └── 验证计算步骤
    │       └── 否 → 继续
    │
    └── 性能问题？
            ├── 是 → 使用 Profilers
            │           ├── Tracy 分析主机端
            │           ├── Device Profiler 分析内核
            │           └── 识别热点和瓶颈
            └── 否 → 查看日志和文档
```

### D.5 常见 CB 错误排查

| 症状 | 可能原因 | 检查方法 |
|------|----------|----------|
| Kernel 在 `cb_reserve_back` 挂起 | CB 已满，消费者未及时消费 | 检查消费者速度、增加 CB 大小 |
| Kernel 在 `cb_wait_front` 挂起 | CB 为空，生产者未及时生产 | 检查生产者速度、检查信号量同步 |
| 数据损坏 | CB 大小计算错误 | 验证 tile_size * num_tiles = total_size |
| 随机崩溃 | 内存对齐问题 | 确保所有地址 16 字节对齐 |
| 结果不正确 | CB 索引错误 | 验证 CB 索引与配置一致 |

---

*附录版本: 1.0*
*最后更新: 2026-03-12*
*适用于: TT-Metalium v0.55+*
