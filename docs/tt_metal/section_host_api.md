# Host API 参考手册

本文档详细描述 TT-Metalium Host 端 C++ API，用于设备管理、内存管理、Kernel 创建和程序执行控制。

**头文件**: `#include <tt_metal/host_api.hpp>`
**命名空间**: `tt::tt_metal`

---

## 目录

1. [设备管理 API](#1-设备管理-api)
2. [Buffer 管理 API](#2-buffer-管理-api)
3. [Kernel 创建 API](#3-kernel-创建-api)
4. [程序执行 API](#4-程序执行-api)
5. [运行时参数 API](#5-运行时参数-api)
6. [子设备管理 API](#6-子设备管理-api)
7. [Event/Semaphore API](#7-eventsemaphore-api)
8. [直接内存访问 API](#8-直接内存访问-api)
9. [配置结构体详解](#9-配置结构体详解)

---

## 1. 设备管理 API

### 1.1 CreateDevice

创建并初始化一个设备实例。

```cpp
IDevice* CreateDevice(
    ChipId device_id,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device_id` | `ChipId` | 设备 ID（从 0 开始） |
| `num_hw_cqs` | `uint8_t` | 硬件命令队列数量（默认 1） |
| `l1_small_size` | `size_t` | L1 小缓冲区大小（默认 0） |
| `trace_region_size` | `size_t` | Trace 区域大小（默认 0） |
| `dispatch_core_config` | `DispatchCoreConfig` | 分发核心配置 |
| `l1_bank_remap` | `vector<uint32_t>` | L1 Bank 重映射表 |
| `worker_l1_size` | `size_t` | Worker L1 大小 |

**返回值**: 指向设备实例的指针 (`IDevice*`)

**使用示例**:

```cpp
#include <tt_metal/host_api.hpp>

using namespace tt::tt_metal;

// 基本用法
IDevice* device = CreateDevice(0);

// 使用多个命令队列
IDevice* device = CreateDevice(0, 2);

// 完整配置
DispatchCoreConfig dispatch_config;
IDevice* device = CreateDevice(
    0,                          // device_id
    1,                          // num_hw_cqs
    0,                          // l1_small_size
    0,                          // trace_region_size
    dispatch_config,            // dispatch_core_config
    {},                         // l1_bank_remap
    DEFAULT_WORKER_L1_SIZE      // worker_l1_size
);
```

---

### 1.2 CreateDeviceMinimal

创建最小化设备实例，用于故障恢复或连接到异常状态的设备。

```cpp
IDevice* CreateDeviceMinimal(
    ChipId device_id,
    uint8_t num_hw_cqs = 1,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{}
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device_id` | `ChipId` | 设备 ID |
| `num_hw_cqs` | `uint8_t` | 硬件命令队列数量 |
| `dispatch_core_config` | `DispatchCoreConfig` | 分发核心配置 |

**返回值**: 指向设备实例的指针 (`IDevice*`)

**使用场景**:
- 设备处于异常状态需要恢复
- 快速检测设备是否存在
- 不需要完整初始化的场景

**使用示例**:

```cpp
// 尝试连接到可能异常的设备
try {
    IDevice* device = CreateDeviceMinimal(0);
    // 执行诊断操作
    CloseDevice(device);
} catch (const std::exception& e) {
    // 处理设备不可用的情况
}
```

---

### 1.3 CloseDevice

关闭设备并释放资源。

```cpp
bool CloseDevice(IDevice* device);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 要关闭的设备指针 |

**返回值**: 成功返回 `true`，失败返回 `false`

**使用示例**:

```cpp
IDevice* device = CreateDevice(0);
// ... 使用设备 ...
bool success = CloseDevice(device);
if (!success) {
    // 处理关闭失败
}
```

---

### 1.4 GetNumAvailableDevices

获取系统中可用的 Tenstorrent 设备数量。

```cpp
size_t GetNumAvailableDevices();
```

**返回值**: 可用设备数量

**使用示例**:

```cpp
size_t num_devices = GetNumAvailableDevices();
std::cout << "Found " << num_devices << " Tenstorrent device(s)" << std::endl;

// 遍历所有设备
for (size_t i = 0; i < num_devices; i++) {
    IDevice* device = CreateDevice(i);
    // ... 使用设备 ...
    CloseDevice(device);
}
```

---

### 1.5 GetNumPCIeDevices

获取通过 PCIe 连接的 Tenstorrent 设备数量。

```cpp
size_t GetNumPCIeDevices();
```

**返回值**: PCIe 设备数量

**使用示例**:

```cpp
size_t pcie_devices = GetNumPCIeDevices();
size_t total_devices = GetNumAvailableDevices();

std::cout << "PCIe devices: " << pcie_devices << std::endl;
std::cout << "Total devices (including Ethernet): " << total_devices << std::endl;
```

---

### 1.6 IsGalaxyCluster

检测设备是否配置为 Galaxy 集群。

```cpp
bool IsGalaxyCluster();
```

**返回值**: 如果是 Galaxy 集群返回 `true`

**使用示例**:

```cpp
if (IsGalaxyCluster()) {
    // Galaxy 集群特定配置
    std::cout << "Running on Galaxy cluster" << std::endl;
} else {
    // 单机配置
    std::cout << "Running on single machine" << std::endl;
}
```

---

### 1.7 ReleaseOwnership

释放 MetalContext 单例的所有权。

```cpp
void ReleaseOwnership();
```

**使用场景**:
- 多进程环境中释放设备控制
- 清理前释放资源

**使用示例**:

```cpp
// 在程序退出或需要释放设备控制时
ReleaseOwnership();
```

---

### 1.8 SetRootDir

设置 TT Metal 元数据文件的根目录。

```cpp
void SetRootDir(const std::string& root_dir);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `root_dir` | `string` | 根目录路径 |

**使用示例**:

```cpp
SetRootDir("/path/to/tt-metal");
```

---

## 2. Buffer 管理 API

### 2.1 CreateBuffer (Interleaved)

创建交织缓冲区（Interleaved Buffer）。

```cpp
// 基本创建
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config);

// 指定地址创建
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address);

// 指定子设备创建
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `config` | `InterleavedBufferConfig` | 缓冲区配置 |
| `address` | `DeviceAddr` | 指定缓冲区地址（可选） |
| `sub_device_id` | `SubDeviceId` | 子设备 ID（可选） |

**返回值**: 指向 Buffer 的共享指针

**使用示例**:

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

// 指定地址创建（用于特定内存布局）
DeviceAddr fixed_addr = 0x10000;
auto fixed_buffer = CreateBuffer(dram_config, fixed_addr);
```

---

### 2.2 CreateBuffer (Sharded)

创建分片缓冲区（Sharded Buffer）。

```cpp
// 基本创建
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config);

// 指定地址创建
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address);

// 指定子设备创建
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `config` | `ShardedBufferConfig` | 分片缓冲区配置 |
| `address` | `DeviceAddr` | 指定缓冲区地址（可选） |
| `sub_device_id` | `SubDeviceId` | 子设备 ID（可选） |

**使用示例**:

```cpp
// 定义分片规格
ShardSpec shard_spec{
    .grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))),  // 8x8 核心网格
    .shape = {32, 32},  // 每个核心的分片形状
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

---

### 2.3 DeallocateBuffer

释放缓冲区。

```cpp
void DeallocateBuffer(Buffer& buffer);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `buffer` | `Buffer&` | 要释放的缓冲区引用 |

**使用示例**:

```cpp
auto buffer = CreateBuffer(config);
// ... 使用缓冲区 ...
DeallocateBuffer(*buffer);
```

---

### 2.4 AssignGlobalBufferToProgram

将全局缓冲区分配给程序。

```cpp
void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `buffer` | `shared_ptr<Buffer>` | 全局缓冲区 |
| `program` | `Program&` | 目标程序 |

**使用场景**:
- 在多个程序间共享缓冲区
- 管理缓冲区生命周期

**使用示例**:

```cpp
auto global_buffer = CreateBuffer(config);
Program program = CreateProgram();

// 将缓冲区分配给程序
AssignGlobalBufferToProgram(global_buffer, program);

// 现在程序拥有对该缓冲区的引用
```

---

## 3. Kernel 创建 API

### 3.1 CreateKernel

从文件创建 Kernel 并添加到程序。

```cpp
KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `file_name` | `string` | Kernel 源文件路径 |
| `core_spec` | `CoreCoord/CoreRange/CoreRangeSet` | 核心规格 |
| `config` | `DataMovementConfig/ComputeConfig/EthernetConfig` | Kernel 配置 |

**返回值**: Kernel 句柄 (`KernelHandle`)

**使用示例**:

```cpp
Program program = CreateProgram();

// Data Movement Kernel (Reader)
auto reader_kernel = CreateKernel(
    program,
    "tt_metal/kernels/dataflow/reader.cpp",
    CoreCoord(0, 0),
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = {num_tiles, tile_size}
    }
);

// Data Movement Kernel (Writer)
auto writer_kernel = CreateKernel(
    program,
    "tt_metal/kernels/dataflow/writer.cpp",
    CoreCoord(0, 0),
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = {num_tiles, tile_size}
    }
);

// Compute Kernel
auto compute_kernel = CreateKernel(
    program,
    "tt_metal/kernels/compute/matmul.cpp",
    CoreCoord(0, 0),
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = {Mt, Kt, Nt}
    }
);

// 多核心配置
CoreRange core_range(CoreCoord(0, 0), CoreCoord(7, 7));
auto multi_core_kernel = CreateKernel(
    program,
    "kernel.cpp",
    core_range,
    DataMovementConfig{...}
);
```

---

### 3.2 CreateKernelFromString

从源代码字符串创建 Kernel。

```cpp
KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `kernel_src_code` | `string` | Kernel 源代码字符串 |
| `core_spec` | `CoreCoord/CoreRange/CoreRangeSet` | 核心规格 |
| `config` | `DataMovementConfig/ComputeConfig/EthernetConfig` | Kernel 配置 |

**返回值**: Kernel 句柄 (`KernelHandle`)

**使用示例**:

```cpp
const std::string reader_src = R"(
    #include "dataflow_api.h"

    void kernel_main() {
        uint32_t num_tiles = get_arg_val<uint32_t>(0);
        uint32_t src_addr = get_arg_val<uint32_t>(1);

        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_reserve_back(cb_id, 1);
            uint32_t write_addr = get_write_ptr(cb_id);
            noc_async_read(src_addr + i * tile_size, write_addr, tile_size);
            noc_async_read_barrier();
            cb_push_back(cb_id, 1);
        }
    }
)";

auto reader_kernel = CreateKernelFromString(
    program,
    reader_src,
    CoreCoord(0, 0),
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);
```

---

## 4. 程序执行 API

### 4.1 EnqueueProgram

将程序入队到命令队列执行。

```cpp
void EnqueueProgram(
    CommandQueue& cq,
    Program& program,
    bool blocking
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `cq` | `CommandQueue&` | 命令队列 |
| `program` | `Program&` | 要执行的程序 |
| `blocking` | `bool` | 是否阻塞等待完成 |

**使用示例**:

```cpp
// 获取命令队列
CommandQueue& cq = device->command_queue();

// 非阻塞执行
EnqueueProgram(cq, program, false);

// 后续操作...

// 阻塞执行（等待完成）
EnqueueProgram(cq, program, true);

// 或使用 Finish 同步
EnqueueProgram(cq, program, false);
Finish(cq);
```

---

### 4.2 Finish

等待命令队列中的所有操作完成。

```cpp
void Finish(CommandQueue& cq);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `cq` | `CommandQueue&` | 命令队列 |

**使用示例**:

```cpp
CommandQueue& cq = device->command_queue();

EnqueueProgram(cq, program1, false);
EnqueueProgram(cq, program2, false);
EnqueueProgram(cq, program3, false);

// 等待所有程序完成
Finish(cq);

// 现在可以安全读取结果
```

---

### 4.3 LaunchProgram

直接启动程序（不使用命令队列）。

```cpp
// 基本启动
void LaunchProgram(
    IDevice* device,
    Program& program,
    bool wait_until_cores_done = true,
    bool force_slow_dispatch = false
);

// 使用共享指针
void LaunchProgram(
    IDevice* device,
    const std::shared_ptr<Program>& program,
    bool wait_until_cores_done = true,
    bool force_slow_dispatch = false
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `program` | `Program&/shared_ptr` | 要启动的程序 |
| `wait_until_cores_done` | `bool` | 是否等待核心完成 |
| `force_slow_dispatch` | `bool` | 强制使用慢速分发 |

**使用示例**:

```cpp
// 快速启动并等待完成
LaunchProgram(device, program, true);

// 异步启动
LaunchProgram(device, program, false);
// ... 执行其他工作 ...
WaitProgramDone(device, program);
```

---

### 4.4 CompileProgram

显式编译程序中的所有 Kernel。

```cpp
void CompileProgram(
    IDevice* device,
    Program& program,
    bool force_slow_dispatch = false
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `program` | `Program&` | 要编译的程序 |
| `force_slow_dispatch` | `bool` | 强制使用慢速分发 |

**使用示例**:

```cpp
// 预编译程序
CompileProgram(device, program);

// 多次执行时无需重新编译
for (int i = 0; i < num_iterations; i++) {
    LaunchProgram(device, program, true);
}
```

---

### 4.5 WaitProgramDone

等待程序执行完成。

```cpp
void WaitProgramDone(
    IDevice* device,
    Program& program,
    bool read_device_profiler_results = true
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `program` | `Program&` | 要等待的程序 |
| `read_device_profiler_results` | `bool` | 是否读取性能分析结果 |

**使用示例**:

```cpp
LaunchProgram(device, program, false);

// 执行其他工作...

// 等待程序完成
WaitProgramDone(device, program);
```

---

## 5. 运行时参数 API

### 5.1 SetRuntimeArgs

设置 Kernel 的运行时参数。

```cpp
// 单核心
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    stl::Span<const uint32_t> runtime_args
);

// 使用初始化列表
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::initializer_list<uint32_t> runtime_args
);

// 多核心批量设置
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `kernel` | `KernelHandle` | Kernel 句柄 |
| `core_spec` | `CoreCoord/CoreRange/CoreRangeSet/vector` | 核心规格 |
| `runtime_args` | `Span/initializer_list/vector` | 运行时参数 |

**限制**: 每个核心最多 341 个参数

**使用示例**:

```cpp
// 单核心单参数
SetRuntimeArgs(program, kernel, CoreCoord(0, 0), {num_tiles, src_addr, dst_addr});

// 使用初始化列表
SetRuntimeArgs(program, kernel, CoreCoord(0, 0), {1024, 0x1000, 0x2000});

// 多核心相同参数
CoreRange cores(CoreCoord(0, 0), CoreCoord(7, 7));
SetRuntimeArgs(program, kernel, cores, {num_tiles, src_addr, dst_addr});

// 多核心不同参数
std::vector<CoreCoord> cores;
std::vector<std::vector<uint32_t>> args;
for (uint32_t y = 0; y < 8; y++) {
    for (uint32_t x = 0; x < 8; x++) {
        cores.push_back(CoreCoord(x, y));
        args.push_back({
            per_core_M, per_core_N, per_core_K,
            y * per_core_M * Kt,  // A 偏移
            x * per_core_N        // B 偏移
        });
    }
}
SetRuntimeArgs(program, kernel, cores, args);
```

---

### 5.2 SetCommonRuntimeArgs

设置所有核心共享的通用运行时参数。

```cpp
void SetCommonRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    stl::Span<const uint32_t> runtime_args
);

void SetCommonRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    std::initializer_list<uint32_t> runtime_args
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `kernel_id` | `KernelHandle` | Kernel 句柄 |
| `runtime_args` | `Span/initializer_list` | 通用参数 |

**使用示例**:

```cpp
// 设置所有核心共享的参数
SetCommonRuntimeArgs(program, kernel, {num_tiles, tile_size});

// 在 Kernel 中使用 get_common_arg_val 访问
// uint32_t num_tiles = get_common_arg_val<uint32_t>(0);
```

---

### 5.3 GetRuntimeArgs

获取 Kernel 的运行时参数。

```cpp
// 获取单个核心的参数
RuntimeArgsData& GetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const CoreCoord& logical_core
);

// 获取所有核心的参数
std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `kernel_id` | `KernelHandle` | Kernel 句柄 |
| `logical_core` | `CoreCoord` | 逻辑核心坐标 |

**返回值**: 运行时参数数据引用

**使用示例**:

```cpp
// 修改现有参数
auto& args = GetRuntimeArgs(program, kernel, CoreCoord(0, 0));
args[0] = new_num_tiles;  // 修改第一个参数

// 获取所有参数
auto& all_args = GetRuntimeArgs(program, kernel);
for (auto& core_args : all_args) {
    for (auto& arg : core_args) {
        // 处理参数
    }
}
```

---

### 5.4 GetCommonRuntimeArgs

获取通用运行时参数。

```cpp
RuntimeArgsData& GetCommonRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `kernel_id` | `KernelHandle` | Kernel 句柄 |

**返回值**: 通用运行时参数数据引用

**使用示例**:

```cpp
auto& common_args = GetCommonRuntimeArgs(program, kernel);
common_args[0] = new_value;
```

---

## 6. 子设备管理 API

### 6.1 create_sub_device_manager

创建子设备管理器。

```cpp
SubDeviceManagerId create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices,
    DeviceAddr local_l1_size
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `sub_devices` | `Span<SubDevice>` | 子设备配置数组 |
| `local_l1_size` | `DeviceAddr` | 本地 L1 大小 |

**返回值**: 子设备管理器 ID

**使用示例**:

```cpp
// 定义子设备配置
std::vector<SubDevice> sub_devices;

// 子设备 0: 使用核心 (0,0) 到 (3,3)
sub_devices.push_back(SubDevice{
    .cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(3, 3)))
});

// 子设备 1: 使用核心 (4,0) 到 (7,7)
sub_devices.push_back(SubDevice{
    .cores = CoreRangeSet(CoreRange(CoreCoord(4, 0), CoreCoord(7, 7)))
});

// 创建子设备管理器
SubDeviceManagerId manager_id = device->create_sub_device_manager(
    sub_devices,
    1024 * 1024  // 1MB L1
);
```

---

### 6.2 load_sub_device_manager

加载子设备管理器。

```cpp
void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `sub_device_manager_id` | `SubDeviceManagerId` | 子设备管理器 ID |

**使用示例**:

```cpp
// 加载子设备管理器
device->load_sub_device_manager(manager_id);

// 现在可以在特定子设备上执行操作
```

---

### 6.3 remove_sub_device_manager

移除子设备管理器。

```cpp
void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `sub_device_manager_id` | `SubDeviceManagerId` | 子设备管理器 ID |

**使用示例**:

```cpp
// 清理子设备管理器
device->remove_sub_device_manager(manager_id);
```

---

### 6.4 set_sub_device_stall_group

设置子设备停顿组。

```cpp
void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `sub_device_ids` | `Span<SubDeviceId>` | 子设备 ID 数组 |

**使用示例**:

```cpp
// 设置停顿组
std::vector<SubDeviceId> stall_group = {0, 1};
device->set_sub_device_stall_group(stall_group);

// 重置停顿组
device->reset_sub_device_stall_group();
```

---

### 6.5 get_sub_device_ids

获取所有子设备 ID。

```cpp
const std::vector<SubDeviceId>& get_sub_device_ids() const;
```

**返回值**: 子设备 ID 列表

**使用示例**:

```cpp
const auto& sub_device_ids = device->get_sub_device_ids();
for (auto id : sub_device_ids) {
    std::cout << "SubDevice ID: " << (int)id << std::endl;
}
```

---

## 7. Event/Semaphore API

### 7.1 CreateSemaphore

创建信号量。

```cpp
uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `program` | `Program&` | 目标程序 |
| `core_spec` | `CoreRange/CoreRangeSet` | 核心规格 |
| `initial_value` | `uint32_t` | 初始值 |

**返回值**: 信号量 ID

**使用示例**:

```cpp
// 创建信号量用于核心间同步
uint32_t semaphore_id = CreateSemaphore(
    program,
    CoreRange(CoreCoord(0, 0), CoreCoord(7, 7)),
    0  // 初始值
);

// 在 Kernel 中使用信号量
// noc_semaphore_wait(get_semaphore(semaphore_id), expected_value);
// noc_semaphore_set(get_semaphore(semaphore_id), new_value);
```

---

### 7.2 CreateGlobalSemaphore

创建全局信号量。

```cpp
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `cores` | `CoreRangeSet` | 核心集合 |
| `initial_value` | `uint32_t` | 初始值 |
| `buffer_type` | `BufferType` | 缓冲区类型（默认 L1） |

**返回值**: 全局信号量对象

**使用示例**:

```cpp
// 创建全局信号量
GlobalSemaphore global_sem = CreateGlobalSemaphore(
    device,
    CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))),
    0,
    BufferType::L1
);

// 在 Host 端操作信号量
// global_sem.set_value(new_value);
// uint32_t value = global_sem.get_value();
```

---

## 8. 直接内存访问 API

### 8.1 WriteToDeviceL1

直接写入设备 L1 内存。

```cpp
// 使用 span
bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER
);

// 使用 vector
bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `logical_core` | `CoreCoord` | 逻辑核心坐标 |
| `address` | `uint32_t` | L1 目标地址 |
| `host_buffer` | `span/vector` | Host 数据缓冲区 |
| `core_type` | `CoreType` | 核心类型（默认 WORKER） |

**返回值**: 成功返回 `true`

**使用示例**:

```cpp
// 准备数据
std::vector<uint32_t> data = {0x12345678, 0x9ABCDEF0, 0x11223344};

// 写入 L1
bool success = WriteToDeviceL1(
    device,
    CoreCoord(0, 0),
    0x10000,  // L1 地址
    data
);

if (!success) {
    // 处理错误
}
```

---

### 8.2 ReadFromDeviceL1

直接从设备 L1 内存读取。

```cpp
// 使用 span
bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER
);

// 使用 vector
bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `logical_core` | `CoreCoord` | 逻辑核心坐标 |
| `address` | `uint32_t` | L1 源地址 |
| `size` | `uint32_t` | 读取大小（vector 版本） |
| `host_buffer` | `span/vector` | Host 数据缓冲区 |
| `core_type` | `CoreType` | 核心类型 |

**返回值**: 成功返回 `true`

**使用示例**:

```cpp
// 读取数据
std::vector<uint32_t> data;
bool success = ReadFromDeviceL1(
    device,
    CoreCoord(0, 0),
    0x10000,  // L1 地址
    1024,     // 读取 1024 字节
    data
);

// 处理数据
for (auto& val : data) {
    std::cout << "0x" << std::hex << val << std::endl;
}
```

---

### 8.3 WriteToDeviceDRAMChannel

直接写入设备 DRAM 通道。

```cpp
// 使用 span
bool WriteToDeviceDRAMChannel(
    IDevice* device,
    int dram_channel,
    uint32_t address,
    std::span<const uint8_t> host_buffer
);

// 使用 vector
bool WriteToDeviceDRAMChannel(
    IDevice* device,
    int dram_channel,
    uint32_t address,
    std::vector<uint32_t>& host_buffer
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `dram_channel` | `int` | DRAM 通道号 |
| `address` | `uint32_t` | DRAM 目标地址 |
| `host_buffer` | `span/vector` | Host 数据缓冲区 |

**返回值**: 成功返回 `true`

**使用示例**:

```cpp
std::vector<uint32_t> data(1024, 0xABCD);

bool success = WriteToDeviceDRAMChannel(
    device,
    0,          // DRAM 通道 0
    0x10000,    // DRAM 地址
    data
);
```

---

### 8.4 ReadFromDeviceDRAMChannel

直接从设备 DRAM 通道读取。

```cpp
// 使用 span
bool ReadFromDeviceDRAMChannel(
    IDevice* device,
    int dram_channel,
    uint32_t address,
    std::span<uint8_t> host_buffer
);

// 使用 vector
bool ReadFromDeviceDRAMChannel(
    IDevice* device,
    int dram_channel,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer
);
```

**参数说明**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 目标设备 |
| `dram_channel` | `int` | DRAM 通道号 |
| `address` | `uint32_t` | DRAM 源地址 |
| `size` | `uint32_t` | 读取大小（vector 版本） |
| `host_buffer` | `span/vector` | Host 数据缓冲区 |

**返回值**: 成功返回 `true`

**使用示例**:

```cpp
std::vector<uint32_t> data;

bool success = ReadFromDeviceDRAMChannel(
    device,
    0,          // DRAM 通道 0
    0x10000,    // DRAM 地址
    4096,       // 读取 4KB
    data
);
```

---

## 9. 配置结构体详解

### 9.1 DataMovementConfig

数据移动 Kernel 配置结构体。

```cpp
struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    std::vector<uint32_t> compile_args = {};
    std::string defines = "";
};
```

**字段说明**:

| 字段 | 类型 | 描述 |
|------|------|------|
| `processor` | `DataMovementProcessor` | 处理器选择（RISCV_0/RISCV_1） |
| `noc` | `NOC` | NoC 选择 |
| `compile_args` | `vector<uint32_t>` | 编译时参数 |
| `defines` | `string` | 预处理器定义 |

**处理器选择**:

```cpp
enum class DataMovementProcessor : uint8_t {
    RISCV_0 = 0,  // BRISC - 通常用于 Reader
    RISCV_1 = 1   // NCRISC - 通常用于 Writer
};
```

**NoC 选择**:

```cpp
enum NOC : uint8_t {
    NOC_0 = 0,
    NOC_1 = 1,
    RISCV_0_default = NOC_0,
    RISCV_1_default = NOC_1
};
```

---

### 9.2 ComputeConfig

计算 Kernel 配置结构体。

```cpp
struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args = {};
    std::string defines = "";
};
```

**字段说明**:

| 字段 | 类型 | 描述 |
|------|------|------|
| `math_fidelity` | `MathFidelity` | 数学精度 |
| `fp32_dest_acc_en` | `bool` | 启用 FP32 累加 |
| `math_approx_mode` | `bool` | 启用近似模式 |
| `compile_args` | `vector<uint32_t>` | 编译时参数 |
| `defines` | `string` | 预处理器定义 |

**MathFidelity 枚举**:

```cpp
enum class MathFidelity {
    LoFi,   // 最低精度，最高性能
    HiFi2,  // 中等精度
    HiFi3,  // 较高精度
    HiFi4   // 最高精度，较低性能
};
```

---

### 9.3 EthernetConfig

以太网 Kernel 配置结构体。

```cpp
struct EthernetConfig {
    Eth eth_mode;
    DataMovementProcessor processor;
    NOC noc;
    std::vector<uint32_t> compile_args = {};
    std::string defines = "";
};
```

**字段说明**:

| 字段 | 类型 | 描述 |
|------|------|------|
| `eth_mode` | `Eth` | 以太网模式（SENDER/RECEIVER） |
| `processor` | `DataMovementProcessor` | 处理器 |
| `noc` | `NOC` | NoC 选择 |
| `compile_args` | `vector<uint32_t>` | 编译时参数 |
| `defines` | `string` | 预处理器定义 |

---

### 9.4 InterleavedBufferConfig

交织缓冲区配置结构体。

```cpp
struct InterleavedBufferConfig {
    IDevice* device;
    DeviceAddr size;
    DeviceAddr page_size;
    BufferType buffer_type;
};
```

**字段说明**:

| 字段 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 设备指针 |
| `size` | `DeviceAddr` | 缓冲区总大小（字节） |
| `page_size` | `DeviceAddr` | 页大小（字节） |
| `buffer_type` | `BufferType` | 缓冲区类型 |

**BufferType 枚举**:

```cpp
enum class BufferType {
    DRAM,      // DRAM 缓冲区
    L1,        // L1 SRAM 缓冲区
    L1_SMALL,  // L1 小缓冲区
    TRACE      // Trace 缓冲区
};
```

---

### 9.5 ShardedBufferConfig

分片缓冲区配置结构体。

```cpp
struct ShardedBufferConfig {
    IDevice* device{};
    DeviceAddr size{};
    DeviceAddr page_size{};
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};
```

**字段说明**:

| 字段 | 类型 | 描述 |
|------|------|------|
| `device` | `IDevice*` | 设备指针 |
| `size` | `DeviceAddr` | 缓冲区总大小 |
| `page_size` | `DeviceAddr` | 页大小 |
| `buffer_type` | `BufferType` | 缓冲区类型 |
| `buffer_layout` | `TensorMemoryLayout` | 内存布局 |
| `shard_parameters` | `ShardSpecBuffer` | 分片参数 |

**TensorMemoryLayout 枚举**:

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED,     // 交织布局
    HEIGHT_SHARDED,  // 高度分片
    WIDTH_SHARDED,   // 宽度分片
    BLOCK_SHARDED    // 块分片
};
```

---

### 9.6 CircularBufferConfig

循环缓冲区配置结构体。

```cpp
class CircularBufferConfig {
public:
    CircularBufferConfig(
        uint32_t total_size,
        const std::map<uint8_t, tt::DataFormat>& data_formats
    );

    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);
    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);
    CircularBufferConfig& set_tile_dims(uint8_t buffer_index, uint32_t num_tiles);
};
```

**使用示例**:

```cpp
// 创建 CB 配置
CircularBufferConfig cb_config(
    num_pages * page_size,  // 总大小
    {{cb_index, tt::DataFormat::Float16_b}}  // 数据格式
);

cb_config.set_page_size(cb_index, page_size);

// 创建 CB
auto cb = CreateCircularBuffer(program, core_spec, cb_config);
```

---

## 10. 完整示例

### 10.1 基本矩阵乘法程序

```cpp
#include <tt_metal/host_api.hpp>
#include <tt_metal/detail/tt_metal.hpp>

using namespace tt::tt_metal;

int main() {
    // 1. 创建设备
    IDevice* device = CreateDevice(0);

    // 2. 创建程序
    Program program = CreateProgram();

    // 3. 定义核心网格
    CoreRange cores(CoreCoord(0, 0), CoreCoord(7, 7));

    // 4. 创建缓冲区
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = 1024 * 1024,
        .page_size = 2048,
        .buffer_type = BufferType::DRAM
    };
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // 5. 创建 Kernels
    auto reader = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    auto writer = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    auto compute = CreateKernel(
        program,
        "kernels/compute/matmul.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false
        }
    );

    // 6. 创建 Circular Buffers
    CircularBufferConfig cb0_config(
        2 * 2048,
        {{0, tt::DataFormat::Float16_b}}
    ).set_page_size(0, 2048);
    auto cb0 = CreateCircularBuffer(program, cores, cb0_config);

    // 7. 设置运行时参数
    for (uint32_t y = 0; y < 8; y++) {
        for (uint32_t x = 0; x < 8; x++) {
            CoreCoord core(x, y);
            SetRuntimeArgs(program, reader, core, {
                input_buffer->address(),
                1024,  // num_tiles
                x * 128 + y * 1024  // offset
            });
            SetRuntimeArgs(program, writer, core, {
                output_buffer->address(),
                1024
            });
        }
    }

    // 8. 执行程序
    CommandQueue& cq = device->command_queue();
    EnqueueProgram(cq, program, true);

    // 9. 清理
    CloseDevice(device);

    return 0;
}
```

---

## 附录：类型别名

```cpp
using ChipId = int;
using DeviceAddr = uint64_t;
using KernelHandle = uint64_t;
using CBHandle = uintptr_t;
using SubDeviceId = uint8_t;
using SubDeviceManagerId = uint32_t;
```

---

*文档版本: 1.0*
*最后更新: 2026-03-12*
