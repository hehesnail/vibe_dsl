# tools/ 模块源码解析

## 1. 模块概述

TT-Metal的 `tools/` 模块是一个综合性的开发和调试工具集合，为开发者提供了多种用于性能分析、内存调试、固件预编译和系统监控的实用工具。

### 模块职责

- **开发工具支持**：提供内存设置、固件预编译等开发辅助功能
- **性能分析**：主机和设备带宽测试、内核性能分析、NOC事件追踪
- **调试支持**：Watcher系统状态转储、命令队列分析、NOC传输数据记录
- **LightMetal支持**：二进制文件执行和回放

### 工具列表和功能

| 工具 | 类型 | 主要功能 |
|------|------|----------|
| `memset` | 脚本+可执行程序 | 向芯片DRAM或L1内存写入指定值 |
| `dump-consts.py` | Python脚本 | 从ELF文件中提取编译时常量 |
| `lightmetal_runner` | 可执行程序 | 执行LightMetal二进制文件 |
| `mem_bench` | 可执行程序 | 主机和设备内存带宽基准测试 |
| `precompile_fw` | 可执行程序 | 预编译固件二进制文件 |
| `watcher_dump` | 可执行程序 | 转储Watcher调试信息 |
| `profiler` | 头文件库 | 内核性能分析框架 |

---

## 2. 目录结构

```
tt_metal/tools/
├── CMakeLists.txt           # 主CMake配置
├── README.md                # 模块说明文档
├── sources.cmake            # 源文件列表
├── dump-consts.py           # ELF常量提取脚本
├── memset.cpp               # 内存设置C++实现
├── memset.py                # 内存设置Python封装
├── lightmetal_runner/       # LightMetal执行器
│   ├── CMakeLists.txt
│   ├── sources.cmake
│   └── lightmetal_runner.cpp
├── mem_bench/               # 内存带宽测试
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── sources.cmake
│   ├── context.hpp          # 测试上下文定义
│   ├── device_utils.hpp/.cpp # 设备操作工具
│   ├── host_utils.hpp/.cpp   # 主机操作工具
│   ├── work_thread.hpp      # 线程同步工具
│   ├── mem_bench.cpp        # 主测试程序
│   └── kernels/
│       └── mem_bench_kernel.cpp  # 设备端测试内核
├── precompile_fw/           # 固件预编译工具
│   ├── CMakeLists.txt
│   ├── sources.cmake
│   └── precompile_fw.cpp
├── profiler/                # 性能分析器
│   ├── README.md
│   ├── kernel_profiler.hpp  # 内核分析主框架
│   ├── event_metadata.hpp   # NOC事件元数据
│   ├── noc_event_profiler.hpp      # NOC事件分析
│   ├── noc_event_profiler_utils.hpp # NOC事件工具
│   ├── fabric_event_profiler.hpp    # Fabric事件分析
│   ├── noc_debugging_profiler.hpp   # NOC调试分析
│   ├── noc_debugging_metadata.hpp   # NOC调试元数据
│   ├── perf_counters.hpp    # 性能计数器
│   ├── tt_metal_tracy.hpp   # Tracy集成
│   ├── cpp_device_analyses.json     # 设备分析配置
│   └── sync/                # 同步内核
│       ├── sync_kernel.cpp
│       ├── sync_device_kernel_sender.cpp
│       └── sync_device_kernel_receiver.cpp
└── watcher_dump/            # Watcher转储工具
    ├── CMakeLists.txt
    ├── sources.cmake
    └── watcher_dump.cpp
```

---

## 3. 工具详解

### 3.1 LightMetal Runner

#### 用途

LightMetal Runner是一个独立的命令行工具，用于执行由 `LightMetalBeginCapture()` 和 `LightMetalEndCapture()` API生成的LightMetal二进制文件。这些二进制文件包含：

- 主机API调用的序列化表示
- 设备CQ工作负载/追踪
- （未来支持）预编译程序/内核以实现快速部署

#### 实现分析

**源码位置**: `/tmp/tt-metal/tt_metal/tools/lightmetal_runner/lightmetal_runner.cpp`

**核心实现**:

```cpp
int main(int argc, char* argv[]) {
    using namespace tt::tt_metal::experimental::lightmetal;

    // 处理命令行参数
    std::string program_filename = argv[0];
    TT_FATAL(argc == 2, "Invalid number of supplied arguments. Usage: {} <binary_file>", program_filename.c_str());
    std::string binary_filename = argv[1];

    // 读取LightMetal二进制文件并执行
    auto binary = LightMetalBinary::load_from_file(binary_filename);
    LightMetalReplay lm_replay(std::move(binary));

    if (!lm_replay.run()) {
        log_fatal(tt::LogMetalTrace, "Light Metal Binary {} failed to execute or encountered errors.", binary_filename);
        return 1;
    }
    log_info(tt::LogMetalTrace, "Light Metal Binary {} executed successfully", binary_filename);
    return 0;
}
```

**关键组件**:

1. **LightMetalBinary**: 从文件加载二进制数据
2. **LightMetalReplay**: 负责执行二进制文件中的操作序列

**构建配置**:

```cmake
add_executable(lightmetal_runner ${LIGHTMETAL_RUNNER_SOURCES})
target_link_libraries(
    lightmetal_runner
    PRIVATE
        Metalium::Metal
        TT::STL
        FlatBuffers::FlatBuffers
)
```

#### 使用方法

```bash
./build/tools/lightmetal_runner <binary_file>
```

**参数**:
- `binary_file`: 要执行的LightMetal二进制文件路径

**返回值**:
- `0`: 执行成功
- `1`: 执行失败或遇到错误

---

### 3.2 Memory Benchmark (mem_bench)

#### 用途

`mem_bench` 是一个用于测量Tenstorrent设备上主机和设备内存带宽的综合基准测试工具。它可以测试：

- 主机到HugePage的拷贝带宽
- 设备通过PCIe读取/写入HugePage的带宽
- 多线程主机拷贝性能
- 多设备MMIO性能

#### 实现分析

**源码位置**: `/tmp/tt-metal/tt_metal/tools/mem_bench/`

**核心架构**:

1. **Context结构** (`context.hpp`): 定义测试配置和状态

```cpp
struct Context {
    std::map<ChipId, IDevice*> devices;
    L1MemoryMap device_address{};
    uint32_t total_size{0};      // 总传输大小
    uint32_t page_size{0};       // 页面大小
    int threads{0};              // 主机线程数
    int number_reader_kernels{0};   // 读内核数
    int number_writer_kernels{0};   // 写内核数
    bool enable_host_copy_with_kernels{false};
    int iterations{0};
};
```

2. **TestResult结构**: 存储测试结果

```cpp
struct TestResult {
    double host_bytes_processed{0};
    double host_time_elapsed{0};
    double host_wait_for_kernel_time_elapsed{0};
    double total_cores_cycles{0};
    double total_cores_time{0};
    double total_cores_bytes_rd{0};
    double total_cores_bytes_wr{0};
    // ... 单核统计
};
```

**主要测试场景**:

| 测试函数 | 描述 |
|----------|------|
| `mem_bench_page_sizing` | 测试不同页面大小的主机拷贝性能 |
| `mem_bench_copy_multithread` | 多线程主机拷贝带宽测试 |
| `mem_bench_copy_with_active_kernel` | 主机拷贝与设备活动内核并发测试 |
| `mem_bench_copy_active_kernel_different_page` | 不同HugePage上的并发测试 |
| `mem_bench_copy_with_read_and_write_kernel` | 同时读写内核测试 |
| `mem_bench_multi_mmio_devices_reading_same_node` | 同NUMA节点多设备测试 |
| `mem_bench_multi_mmio_devices_reading_different_node` | 跨NUMA节点多设备测试 |

**设备端内核** (`kernels/mem_bench_kernel.cpp`):

内核通过编译时参数配置为读或写模式：

```cpp
// 读内核参数
constexpr uint32_t my_rd_dst_addr = get_compile_time_arg_val(0);
constexpr uint32_t pcie_rd_base = get_compile_time_arg_val(1);
constexpr uint32_t pcie_rd_size = get_compile_time_arg_val(2);
constexpr uint32_t pcie_rd_transfer_size = get_compile_time_arg_val(3);

// 写内核参数
constexpr uint32_t pcie_wr_base = get_compile_time_arg_val(6);
constexpr uint32_t pcie_wr_size = get_compile_time_arg_val(7);
constexpr uint32_t pcie_wr_transfer_size = get_compile_time_arg_val(8);
```

内核使用NOC异步读写API进行PCIe传输：
- `noc_async_read()`: 从主机读取数据
- `noc_async_write()`: 向主机写入数据
- `noc_async_read_barrier()`: 等待读完成
- `noc_async_write_barrier()`: 等待写完成

**线程同步机制** (`work_thread.hpp`):

使用条件变量实现精确的线程同步：

```cpp
template <typename F, typename IntermediateF, typename... Args>
double execute_work_synced_start(int num_threads, F&& work_fn, IntermediateF&& intermediate_fn, Args&&... args) {
    // 使用condition_variable同步所有线程启动
    // 计算最早开始和最晚结束时间
    // 返回总执行时间
}
```

#### 使用方法

```bash
# 基本测试（默认5次迭代）
./build/tools/mem_bench

# 完整测试套件
./build/tools/mem_bench --full

# 指定设备ID
./build/tools/mem_bench --device-id=0

# 输出JSON格式结果
./build/tools/mem_bench --benchmark_format=json

# NUMA系统优化（绑定到节点0）
numactl --cpubind=0 --membind=0 ./build/tools/mem_bench
```

**输出指标**:

| 指标 | 说明 |
|------|------|
| `bytes_per_second` | 主机拷贝到HugePage的聚合带宽 |
| `dev_bw` | 设备核心平均PCIe读/写带宽 |
| `dev_rd_bw` / `dev_wr_bw` | 设备读/写带宽 |
| `kernel_0_bw` | 第一个核心的PCIe带宽 |

---

### 3.3 Precompile Firmware (precompile_fw)

#### 用途

`precompile_fw` 工具用于预编译TT-Metal固件二进制文件，支持多种芯片架构和配置组合。预编译的固件可以加速后续的设备初始化过程。

#### 实现分析

**源码位置**: `/tmp/tt-metal/tt_metal/tools/precompile_fw/precompile_fw.cpp`

**支持的配置**:

```cpp
const auto supported_configs = std::to_array<PrecompileConfig>({
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_eth_dispatch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_fabric_mux.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_eth_dispatch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_fabric_mux.yaml"},
});
```

**配置枚举逻辑**:

1. **架构特定配置**:
   - Wormhole B0: 支持路由固件启用/禁用配置
   - Blackhole: 支持2-ERISC模式，DRAM收割配置

2. **核心描述符解析**:
   ```cpp
   void enumerate_jit_device_configs(
       tt::ARCH arch,
       const std::string& core_descriptor_path,
       const std::function<void(const JitDeviceConfig&)>& callback
   )
   ```

3. **JIT设备配置参数**:
   - HAL实例
   - 架构类型
   - DRAM/L1 bank数量
   - PCIe核心坐标
   - 调度核心类型（Tensix/Ethernet）
   - 调度核心轴（行/列）
   - 硬件CQ数量
   - 路由固件启用状态

**固件构建流程**:

```cpp
void precompile_for_config(
    const tt::tt_metal::JitDeviceConfig& jit_device_config,
    const tt::llrt::RunTimeOptions& rtoptions
) {
    BuildEnvManager build_env_manager(*jit_device_config.hal);
    build_env_manager.add_build_env(0, jit_device_config, rtoptions);
    build_env_manager.build_firmware(0, /*ignore_precompiled=*/true);

    // 复制生成的.elf文件到预编译目录
    auto build_key = dev_build_env.build_key();
    auto precompiled_firmware_dir = rtoptions.get_root_dir() + "tt_metal/pre-compiled/" + std::to_string(build_key) + "/";
    copy_firmware_to_precompiled_dir(firmware_out_path, precompiled_firmware_dir);
}
```

**CMake集成**:

```cmake
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
    COMMAND ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_odr_violation=0 $<TARGET_FILE:precompile_fw>
    COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS precompile_fw
    COMMENT "Pre-compiling firmware binaries"
)

add_custom_target(precompile-fw ALL DEPENDS ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp)
```

#### 使用方法

该工具通常由CMake构建系统自动调用：

```bash
# 手动运行
./build/tools/precompile_fw

# 预编译固件输出目录
./build/tt_metal/pre-compiled/<build_key>/
```

---

### 3.4 Profiler

#### 用途

Profiler是一个综合性的性能分析框架，提供：

- **内核性能分析**: 测量内核执行时间、标记代码区域
- **NOC事件追踪**: 记录NOC读写操作、屏障事件
- **Fabric事件追踪**: 记录Fabric网络通信事件
- **性能计数器**: 访问FPU、PACK、UNPACK等硬件计数器
- **Tracy集成**: 与Tracy分析器集成进行可视化分析

#### 实现分析

**源码位置**: `/tmp/tt-metal/tt_metal/tools/profiler/`

##### 3.4.1 Kernel Profiler (`kernel_profiler.hpp`)

**核心宏定义**:

```cpp
// 基本区域分析
#define DeviceZoneScopedN(name)  // 标记命名代码区域
#define DeviceZoneScopedMainN(name)  // 标记主区域（保证记录）
#define DeviceZoneScopedMainChildN(name)  // 标记子区域

// 时间戳数据
#define DeviceTimestampedData(name, data)  // 记录带时间戳的数据
#define DeviceRecordEvent(event_id)  // 记录事件

// 性能计数器
#define StartPerfCounters()  // 启动性能计数器
#define StopPerfCounters()   // 停止性能计数器
#define RecordPerfCounters() // 使用RAII记录性能计数器
```

**编译时哈希生成**:

```cpp
template <size_t N>
constexpr uint32_t Hash16_CT(const char (&s)[N]) {
    auto res = Hash32_CT(s, N - 1);
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}
```

使用编译时字符串哈希为每个分析区域生成唯一ID。

**缓冲区管理**:

```cpp
constexpr uint32_t PROFILER_L1_VECTOR_SIZE = PROFILER_L1_BUFFER_SIZE / sizeof(uint32_t);
constexpr uint32_t NOC_ALIGNMENT_FACTOR = 4;

// L1内存中的分析数据缓冲区
volatile tt_l1_ptr profiler_msg_buffer_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));
```

**profileScope模板类**:

```cpp
template <uint32_t timer_id, DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() {
        if (bufferHasRoom<dispatch>()) {
            stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = true;
            mark_time_at_index_inlined(wIndex, timer_id);
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    inline __attribute__((always_inline)) ~profileScope() {
        if (start_marked) {
            mark_time_at_index_inlined(wIndex, get_const_id(timer_id, ZONE_END));
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            // ...
        }
    }
};
```

##### 3.4.2 NOC事件分析 (`noc_event_profiler.hpp`)

**支持的事件类型** (来自`event_metadata.hpp`):

```cpp
enum class NocEventType : unsigned char {
    // 读操作
    READ = 1, READ_SET_STATE, READ_SET_TRID, READ_WITH_STATE,
    READ_BARRIER_START, READ_BARRIER_END,

    // 写操作
    WRITE_ = 11, WRITE_SET_TRID, WRITE_WITH_TRID, WRITE_INLINE,
    WRITE_MULTICAST, WRITE_SET_STATE, WRITE_BARRIER_START,
    WRITE_BARRIER_END, WRITE_FLUSH,

    // 屏障和同步
    FULL_BARRIER = 25, ATOMIC_BARRIER,
    SEMAPHORE_INC, SEMAPHORE_WAIT, SEMAPHORE_SET,

    // Fabric事件
    FABRIC_UNICAST_WRITE = 33, FABRIC_UNICAST_INLINE_WRITE,
    FABRIC_MULTICAST_WRITE, FABRIC_UNICAST_SCATTER_WRITE,
    FABRIC_ROUTING_FIELDS_1D, FABRIC_ROUTING_FIELDS_2D,
};
```

**事件记录宏**:

```cpp
#define RECORD_NOC_EVENT_WITH_ADDR(type, local_addr, noc_addr, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT_WITH_ID(type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT(type, posted, noc)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)
```

**NOC地址解码**:

```cpp
FORCE_INLINE std::pair<uint32_t, uint32_t> decode_noc_addr_to_coord(uint64_t noc_addr) {
    return decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS);
}

FORCE_INLINE uint32_t decode_noc_addr_to_local_addr(uint64_t noc_addr) {
    return NOC_LOCAL_ADDR_OFFSET(noc_addr);
}
```

##### 3.4.3 Fabric事件分析 (`fabric_event_profiler.hpp`)

支持记录Fabric通信事件：

```cpp
// 单播写
recordFabricNocEvent(KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_WRITE, ...);

// 多播写
recordFabricNocEventMulticast(KernelProfilerNocEventMetadata::NocEventType::FABRIC_MULTICAST_WRITE, ...);

// 散播写
recordFabricScatterEvent(KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_SCATTER_WRITE, ...);

// 路由字段记录
recordRoutingFields1D(routing_fields);
recordRoutingFields2D(routing_fields, route_buffer);
```

##### 3.4.4 性能计数器 (`perf_counters.hpp`)

**计数器组**:

```cpp
enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1, INSTRN };
enum PerfCounterType : uint8_t {
    SFPU_COUNTER, FPU_COUNTER, MATH_COUNTER
};
```

**性能计数器寄存器操作**:

```cpp
void start_perf_counter() {
    for (auto counter_group : counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg = ...;
            // 设置连续模式并启动计数器
            cntl_reg[0] = 0;
            cntl_reg[1] = counter_select << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
            cntl_reg[2] = PERF_CNT_START_VALUE;
        }
    }
}
```

##### 3.4.5 Tracy集成 (`tt_metal_tracy.hpp`)

提供与Tracy分析器的集成：

```cpp
#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)  // 标记追踪开始
#define TracyTTMetalEndMeshTrace(device_ids, trace_id)    // 标记追踪结束
#define TracyTTMetalReplayMeshTrace(device_ids, trace_id) // 标记追踪回放
#define TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)
```

##### 3.4.6 设备分析配置 (`cpp_device_analyses.json`)

定义了多种设备分析类型：

```json
{
    "type": "PROGRAM_FIRST_TO_LAST_MARKER",
    "dimension": "PROGRAM",
    "results_config": {
        "analysis_name": "DEVICE FW DURATION [ns]"
    },
    "start_config": {
        "marker_type": "ZONE_START",
        "marker_name_keywords": ["-FW"]
    },
    "end_config": {
        "marker_type": "ZONE_END",
        "marker_name_keywords": ["-FW"]
    }
}
```

#### 使用方法

**内核代码中使用**:

```cpp
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceProfilerInit();

    {
        DeviceZoneScopedN("MY_KERNEL_COMPUTE");
        // 要分析的代码
    }

    DeviceTimestampedData("CUSTOM_DATA", my_data_value);

    // 性能计数器
    RecordPerfCounters();
}
```

**编译选项**:

```bash
# 启用内核分析
TT_METAL_PROFILE_KERNEL=1

# 启用NOC事件追踪
TT_METAL_PROFILE_NOC_EVENTS=1

# 启用性能计数器
TT_METAL_PROFILE_PERF_COUNTERS=1
```

---

### 3.5 Watcher Dump

#### 用途

Watcher Dump工具用于从TT-Metal设备转储调试信息，包括：

- **Watcher日志**: 内核执行状态、断言信息
- **命令队列**: Issue Queue和Completion Queue的内容
- **NOC传输数据**: 记录NOC传输的详细信息

该工具特别适用于调试挂起或崩溃的程序，可以附加到已经运行的设备上获取状态信息。

#### 实现分析

**源码位置**: `/tmp/tt-metal/tt_metal/tools/watcher_dump/watcher_dump.cpp`

**核心功能**:

```cpp
void dump_data(
    vector<ChipId>& device_ids,
    bool dump_watcher,      // 转储Watcher日志
    bool dump_cqs,          // 转储命令队列
    bool dump_cqs_raw_data, // 转储原始CQ数据
    bool dump_noc_xfers,    // 转储NOC传输数据
    bool eth_dispatch,      // 使用Ethernet调度
    int num_hw_cqs          // 硬件CQ数量
) {
    // 禁用L1清除以避免覆盖状态
    tt_metal::MetalContext::instance().rtoptions().set_clear_l1(false);

    // 禁用Watcher以避免干扰
    tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(false);

    // 创建最小化设备连接
    IDevice* device = tt::tt_metal::CreateDeviceMinimal(
        id, num_hw_cqs, DispatchCoreConfig{eth_dispatch ? DispatchCoreType::ETH : DispatchCoreType::WORKER});

    // 执行转储
    if (dump_cqs) {
        internal::dump_cqs(cq_file, iq_file, *sysmem_manager, dump_cqs_raw_data);
    }
    if (dump_watcher) {
        MetalContext::instance().watcher_server()->isolated_dump(device_ids);
    }
    if (dump_noc_xfers) {
        DumpNocData(device_ids);
    }
}
```

**命令行参数处理**:

```cpp
// 支持的参数
-h, --help                    // 显示帮助
-d=LIST, --devices=LIST      // 指定设备ID列表（如"0,2,3"或"all"）
-n=INT, --num-hw-cqs=INT      // 硬件CQ数量
-c, --dump-cqs                // 转储命令队列
--dump-cqs-data               // 转储CQ原始数据
-w, --dump-watcher            // 转储Watcher数据
--dump-noc-transfer-data      // 转储NOC传输数据
--eth-dispatch                // 使用Ethernet调度
```

**关键实现细节**:

1. **最小化设备创建**: 使用 `CreateDeviceMinimal` 避免完整的设备初始化，允许附加到已运行的设备

2. **输出目录结构**:
   ```
   generated/watcher/
   ├── command_queue_dump/
   │   ├── device_{id}_completion_q.txt
   │   └── device_{id}_issue_q.txt
   └── watcher.log
   ```

3. **NOC数据传输记录**: 需要程序在编译时定义 `TT_METAL_RECORD_NOC_TRANSFER_DATA`

#### 使用方法

```bash
# 转储所有设备的Watcher日志
./build/tools/watcher_dump --dump-watcher

# 转储特定设备的Watcher和CQ数据
./build/tools/watcher_dump --devices=0,1 --dump-watcher --dump-cqs

# 转储NOC传输数据（需要程序支持）
./build/tools/watcher_dump --dump-noc-transfer-data

# 使用Ethernet调度的设备
./build/tools/watcher_dump --eth-dispatch --num-hw-cqs=2
```

---

## 4. 设计模式与实现技巧

### 4.1 编译时元编程

**常量哈希生成**:

Profiler使用编译时字符串哈希避免运行时开销：

```cpp
#define SrcLocNameToHash(name)                   \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name))); \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));
```

这会在编译时生成16位哈希值，运行时直接使用而无需字符串比较。

### 4.2 RAII资源管理

**作用域分析**:

```cpp
template <uint32_t timer_id, DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() {
        // 构造函数中记录开始时间
    }
    inline __attribute__((always_inline)) ~profileScope() {
        // 析构函数中记录结束时间
    }
};
```

使用RAII确保即使发生异常也能正确记录结束时间。

**性能计数器包装器**:

```cpp
struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;
```

### 4.3 条件编译与零开销抽象

**条件编译宏**:

```cpp
#if defined(PROFILE_KERNEL) && (!defined(DISPATCH_KERNEL) || ...)
    // 分析代码
#else
    #define DeviceZoneScopedN(name)
    #define DeviceTimestampedData(name, data)
    // 空定义，零开销
#endif
```

当分析禁用时，宏展开为空，不产生任何运行时开销。

### 4.4 类型安全的状态管理

**强类型枚举**:

```cpp
enum class NocEventType : unsigned char {
    READ = 1, WRITE_ = 11, FULL_BARRIER = 25, ...
};

enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };
```

### 4.5 内存对齐与打包

**紧凑的数据结构**:

```cpp
struct alignas(uint64_t) KernelProfilerNocEventMetadata {
    union EventData {
        RawEvent raw_event;
        LocalNocEvent local_event;
        FabricNoCEvent fabric_event;
        // ...
    } data{};
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
```

确保元数据结构可以原子地作为64位值处理。

### 4.6 线程同步模式

**屏障同步**:

```cpp
template <typename F, typename IntermediateF, typename... Args>
double execute_work_synced_start(...) {
    std::mutex m;
    std::condition_variable go_cv;
    int threads_ready{0};

    // 所有线程等待条件变量
    std::unique_lock lk{m};
    threads_ready++;
    if (threads_ready == total_threads) {
        go_cv.notify_all();
    }
    go_cv.wait(lk, [&] { return threads_ready == total_threads; });
}
```

使用条件变量实现精确的同步启动，确保所有线程同时开始执行。

### 4.7 配置驱动架构

**YAML配置解析**:

`precompile_fw` 使用YAML文件定义支持的配置：

```cpp
YAML::Node core_descriptor_yaml = YAML::LoadFile(core_descriptor_path);
for (const auto& product : core_descriptor_yaml) {
    for (const auto& axis_config : product.second) {
        for (const auto& config_node : axis_config.second) {
            // 枚举所有配置组合
        }
    }
}
```

### 4.8 设备抽象层

**统一的设备接口**:

```cpp
// 获取HugePage指针
void* get_hugepage(int device_id, uint32_t base_offset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto mmio_device_id = cluster.get_associated_mmio_device(device_id);
    auto channel = cluster.get_assigned_channel_for_device(device_id);
    return (void*)(cluster.host_dma_address(base_offset, mmio_device_id, channel));
}
```

通过MetalContext提供统一的设备访问接口，隐藏底层硬件差异。

---

## 总结

TT-Metal的 `tools/` 模块提供了全面的开发和调试支持：

1. **LightMetal Runner** 支持工作负载的序列化和回放
2. **Memory Benchmark** 提供详细的带宽性能分析
3. **Precompile Firmware** 加速设备初始化
4. **Profiler** 提供内核和NOC级别的详细性能分析
5. **Watcher Dump** 支持运行时状态检查和调试

这些工具共同构成了TT-Metal开发环境的重要组成部分，帮助开发者优化Tenstorrent硬件上的AI工作负载性能。
