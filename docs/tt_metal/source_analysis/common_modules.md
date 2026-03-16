# 通用模块源码解析 (common/, detail/, hostdevcommon/, graph/, logging/)

## 1. common/ - 基础类型和工具

### 1.1 模块概述

`common/` 模块是 TT-Metal 框架的基础工具库，提供了核心数据类型、坐标系统、线程池、内存管理工具等基础设施。这些组件被整个框架广泛使用，是上层功能构建的基石。

### 1.2 核心组件

#### 1.2.1 核心坐标系统 (core_coord.hpp/cpp)

核心坐标系统是 TT-Metal 中最重要的基础类型之一，用于标识和管理 Tenstorrent 芯片上的计算核心。

**关键类型：**
- `RelativeCoreCoord` - 相对坐标，支持负值索引（如从网格末尾开始计数）
- `CoreRange` - 连续的矩形核心区域，由起始和结束坐标定义
- `CoreRangeSet` - 多个不重叠 CoreRange 的集合

**核心功能：**
```cpp
// 坐标转换
CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size);

// CoreRange 操作
bool intersects(const CoreRange& other) const;      // 判断是否相交
std::optional<CoreRange> intersection(const CoreRange& other) const;  // 求交集
std::optional<CoreRange> merge(const CoreRange& cr) const;            // 合并相邻区域
bool contains(const CoreCoord& other) const;        // 判断是否包含

// 网格遍历
CoreIterator begin() const;  // 支持范围for循环遍历区域内所有核心
```

**网格遍历工具函数：**
```cpp
std::vector<CoreCoord> grid_to_cores(uint32_t num_cores, uint32_t grid_size_x,
                                     uint32_t grid_size_y, bool row_wise);
std::vector<CoreCoord> corerange_to_cores(const CoreRangeSet& crs,
                                          std::optional<uint32_t> max_cores, bool row_wise);
```

#### 1.2.2 形状系统 (shape*.cpp)

TT-Metal 使用多层次的形状表示系统来处理张量维度。

**ShapeBase** - 基础形状类：
- 内部使用固定4维存储，支持任意rank的形状
- 提供 `view()` 方法获取原始维度的span视图
- 自动补齐前导1维度

**Shape** - 主要形状类：
- 基于 `tt::stl::SmallVector<uint32_t>` 实现
- 支持rank转换：`to_rank()` 方法
- 提供4D数组转换：`to_array_4D()`
- 计算体积：`volume()` 方法

**Shape2D** - 专用2D形状：
- 用于表示2D空间维度（如height/width）
- 支持标量乘法运算
- 提供到 `std::pair` 和 `std::array` 的转换

**辅助函数：**
```cpp
// 计算步长（strides）
tt::stl::SmallVector<size_t> compute_strides(const Shape& shape);

// 计算扁平化索引
std::size_t compute_flat_indices(Span<const uint32_t> indices, Span<const size_t> strides);
```

#### 1.2.3 核心分配策略 (core_assignment.hpp/cpp)

该模块实现了 DRAM 接口工作核心的最优分配算法，针对 Wormhole 和 Blackhole 架构进行了优化。

**主要功能：**
```cpp
std::vector<CoreCoord> get_optimal_dram_to_physical_worker_assignment(
    ARCH arch,
    const std::vector<CoreCoord>& dram_phy_coords,
    uint32_t full_grid_size_x,
    uint32_t full_grid_size_y,
    std::vector<uint32_t> worker_phy_x,
    std::vector<uint32_t> worker_phy_y);
```

**算法特点：**
- 根据 DRAM 控制器位置放置工作核心
- 处理 harvesting（禁用行/列）情况下的核心重新分配
- Wormhole: 行 harvesting，将禁用行的工作转移到其他行
- Blackhole: 列 harvesting，将禁用列的工作转移到右侧列

#### 1.2.4 线程池 (thread_pool.hpp/cpp)

TT-Metal 实现了 NUMA 感知的线程池，优化多设备场景下的主机性能。

**接口设计：**
```cpp
class ThreadPool {
public:
    virtual void enqueue(std::function<void()>&& f, std::optional<uint32_t> device_idx = std::nullopt) = 0;
    virtual void wait() = 0;
};

// 工厂函数
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads);
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(const std::vector<IDevice*>& physical_devices);
std::shared_ptr<ThreadPool> create_passthrough_thread_pool();
```

**NumaAwareExecutor 实现：**
- 每个工作线程绑定到特定 CPU 核心
- NUMA 感知：线程优先绑定到距离目标设备最近的 NUMA 节点
- 使用环形缓冲区任务队列（TaskQueue）
- 支持线性退避策略减少 CPU 空转

**线程绑定逻辑：**
```cpp
uint32_t get_cpu_core_for_physical_device(uint32_t physical_device_id) {
    // 如果设备均衡分布在所有 NUMA 节点上，按 NUMA 布局分配
    // 否则跨所有 NUMA 节点/CPU 核心均衡分配以减少资源竞争
}
```

#### 1.2.5 多生产者单消费者队列 (multi_producer_single_consumer_queue.hpp)

基于环形缓冲区实现的 lock-free（带互斥锁保护）队列，用于异步任务处理。

**设计特点：**
- 静态分配 8192 个节点的环形缓冲区
- 支持 `push(T&&)` 和 `push(std::shared_ptr<T>)` 两种接口
- 提供迭代器支持范围遍历
- 满队列时自旋等待（stall）

#### 1.2.6 计时器 (scoped_timer.hpp)

RAII 风格的计时器，用于性能分析。

```cpp
template <typename TimeUnit = std::chrono::nanoseconds>
struct ScopedTimer {
    ScopedTimer(std::string name_, bool print_duration_ = true);
    ~ScopedTimer() {
        // 自动计算并打印耗时
        log_info(tt::LogTimer, "{} -- elapsed: {}{}", name, elapsed.count(), time_unit_to_string());
    }
};
```

#### 1.2.7 稳定哈希 (stable_hash.hpp)

FNV-1a 哈希算法的实现，用于需要跨运行保持一致的哈希场景（如缓存路径）。

```cpp
class FNV1a {
    static constexpr uint64_t FNV_PRIME = 0x100000001b3;
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325;

    void update(uint64_t data);
    void update(const std::string& s);
    uint64_t digest() const;
};
```

#### 1.2.8 环境变量解析 (env_lib.hpp)

类型安全的环境变量解析模板函数。

```cpp
template <typename T>
T parse_env(const char* env_name, const T& default_value);

// 支持类型：bool, std::string, int, uint32_t, uint64_t
```

#### 1.2.9 工作分割 (work_split.cpp)

提供将工作负载均匀分配到多个核心的工具函数。

**核心函数：**
```cpp
// 将工作均匀分割到核心网格
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t>
split_work_to_cores(const CoreCoord grid_size, const uint32_t units_to_divide, const bool row_wise);

// 将核心数量转换为 CoreRangeSet
CoreRangeSet num_cores_to_corerangeset(const CoreCoord start_core,
                                       const uint32_t target_num_cores,
                                       const CoreCoord grid_size,
                                       const bool row_wise);

// 在子网格中分配核心
CoreRangeSet num_cores_to_corerangeset_in_subcoregrids(
    const CoreCoord start_core,
    const uint32_t target_num_cores,
    const CoreRangeSet& sub_core_grids,
    const bool row_wise);
```

#### 1.2.10 执行器 (executor.hpp)

基于 Taskflow 库的并行执行框架。

```cpp
using Executor = tf::Executor;
using ExecTask = tf::Task;

// 获取全局执行器实例
Executor& GetExecutor();

// 异步任务提交（带异常传播）
template <class F, class... Args>
auto async(F&& func, Args&&... args);
```

**特点：**
- 自动检测 CPU 核心数（可通过 `TT_METAL_THREADCOUNT` 环境变量覆盖）
- 支持 fork 安全：通过 `pthread_atfork` 在子进程中重新初始化

#### 1.2.11 内存固定 (memory_pin.cpp, host_buffer.cpp)

`MemoryPin` 类实现了引用计数风格的内存生命周期管理。

```cpp
class MemoryPin {
    std::function<void()> inc_;  // 增加引用计数
    std::function<void()> dec_;  // 减少引用计数
};
```

`HostBuffer` 使用 `MemoryPin` 管理主机内存缓冲区，支持：
- 字节视图访问
- 拷贝和移动语义
- 相等性比较

#### 1.2.12 网格坐标 (mesh_coord.cpp)

分布式计算场景下的多维坐标系统。

**MeshShape** - 网格形状：
- 支持任意维度
- 自动计算 strides
- 提供 `mesh_size()` 计算总设备数
- 判断是否为线性拓扑：`is_line_topology()`

**MeshCoordinate** - 网格坐标：
- 支持负索引（Python风格）
- 转换为线性索引：`to_linear_index()`
- 获取邻居坐标：`get_neighbor()`（支持 WRAP/CLAMP/NONE 边界模式）

**MeshCoordinateRange** - 坐标范围：
- 支持普通范围和环绕（wraparound）范围
- 提供迭代器支持
- 交集、包含判断

**MeshCoordinateRangeSet** - 范围集合：
- 自动合并可合并的范围
- 集合减法操作

#### 1.2.13 舍入工具 (tt_rounding.h)

编译期整数舍入工具函数。

```cpp
template <class Integer>
constexpr Integer round_to_power_of_2(Integer x);

template <class T, class U>
constexpr T round_up_to(T x, U multiple);

template <class T, class U>
constexpr T round_up_div(T dividend, U divisor);

template <class Integer>
static constexpr Integer log2_const(Integer x);
```

#### 1.2.14 后端API类型 (tt_backend_api_types.hpp/cpp)

架构枚举和数据格式的字符串转换工具。

```cpp
std::string get_string(ARCH arch);
std::string get_string_lowercase(ARCH arch);
std::string get_alias(ARCH arch);
ARCH get_arch_from_string(const std::string& arch_str);
bool is_integer_format(DataFormat format);
```

### 1.3 关键代码分析

#### CoreRangeSet 的合并算法

`CoreRangeSet::merge` 实现了复杂的矩形合并算法：

1. 将所有范围绘制到二维网格
2. 按行扫描，识别连续的水平段
3. 尝试与上一行的段垂直合并
4. 返回最小化的矩形集合

这个算法确保了 CoreRangeSet 始终保持最简形式，提高后续操作的效率。

#### NumaAwareExecutor 的任务调度

执行器使用原子计数器和条件变量实现高效的任务同步：

```cpp
void enqueue(std::function<void()>&& f) {
    tasks_.push(std::move(f));
    task_counter_.fetch_add(1, std::memory_order_relaxed);
}

std::exception_ptr wait() const {
    int current;
    while ((current = task_counter_.load(std::memory_order_acquire)) > 0) {
        task_counter_.wait(current, std::memory_order_relaxed);
    }
    return stored_exception_;
}
```

工作线程使用线性退避策略减少 CPU 占用：
- 前 100 次尝试：忙等待
- 100-300 次：每次增加 5 微秒睡眠
- 超过 300 次：最大 1ms 睡眠间隔

---

## 2. detail/ - 内部实现细节

### 2.1 模块概述

`detail/` 模块包含 TT-Metal 的内部实现细节，主要提供内存使用报告和调试功能。这些接口通常不直接暴露给最终用户，而是用于内部诊断和优化。

### 2.2 报告功能

#### 2.2.1 内存报告器 (memory_reporter.hpp/cpp)

单例模式的内存使用报告器，生成详细的设备内存使用报告。

**功能特性：**
- 跟踪程序 L1 内存使用
- 生成 CSV 格式的内存使用摘要
- 支持 DRAM、L1、L1_SMALL 三种缓冲区类型的统计

**生成的报告文件：**
- `program_memory_usage_summary.csv` - 程序内存使用摘要
- `program_l1_usage_summary.csv` - L1 内存使用摘要
- `program_detailed_memory_usage.csv` - 详细内存使用报告

**报告内容示例：**
```csv
Program ID, Total Allocatable Size (B), Total Allocated (B), Total Free (B), Largest Free Block (B)
```

**MemoryView 结构：**
```cpp
struct MemoryView {
    uint32_t num_banks;
    size_t total_bytes_per_bank;
    size_t total_bytes_allocated_per_bank;
    size_t total_bytes_free_per_bank;
    size_t largest_contiguous_bytes_free_per_bank;
    std::vector<MemoryBlockTable> block_table;
};
```

#### 2.2.2 报告工具 (report_utils.hpp)

提供报告目录管理功能：

```cpp
inline const std::string& get_reports_dir() {
    static std::string outpath;
    if (outpath.empty()) {
        outpath = MetalContext::instance().rtoptions().get_logs_dir() + "/generated/reports/";
    }
    return outpath;
}
```

### 2.3 关键代码分析

#### 内存统计收集流程

```cpp
void MemoryReporter::flush_program_memory_usage(uint64_t program_id, const IDevice* device) {
    // 1. 延迟初始化报告文件
    if (not this->program_memory_usage_summary_report_.is_open()) {
        this->init_reports();
    }

    // 2. 写入程序ID
    this->program_memory_usage_summary_report_ << program_id;

    // 3. 收集各类缓冲区的统计信息
    populate_reports(device, ...);
}
```

`populate_reports` 函数收集三种缓冲区类型的数据：
1. DRAM - 外部存储器
2. L1 - 核心本地 SRAM
3. L1_SMALL - 小缓冲区 L1 区域

---

## 3. hostdevcommon/ - Host-Device通用定义

### 3.1 模块概述

`hostdevcommon/` 包含主机代码（C++）和设备代码（RISC-V 内核）共享的定义。这些头文件在编译主机代码和设备内核时都会被包含，确保双方对数据结构和常量的理解一致。

### 3.2 API定义

#### 3.2.1 通用值 (common_values.hpp)

定义主机和设备共享的基本常量：

```cpp
constexpr static std::uint32_t INVALID = 0;
constexpr static std::uint32_t VALID = 1;
constexpr static std::size_t DEFAULT_L1_SMALL_SIZE = 0;
constexpr static std::size_t DEFAULT_TRACE_REGION_SIZE = 0;
constexpr static std::size_t DEFAULT_WORKER_L1_SIZE = 0;  // 动态确定
```

#### 3.2.2 调试打印通用定义 (dprint_common.h)

设备调试打印系统的共享定义。

**缓冲区布局：**
```cpp
struct DebugPrintMemLayout {
    struct Aux {
        uint32_t wpos;      // 写位置
        uint32_t rpos;      // 读位置
        uint16_t core_x;    // 核心X坐标
        uint16_t core_y;    // 核心Y坐标
    } aux;
    uint8_t data[DPRINT_BUFFER_SIZE - sizeof(Aux)];
};
```

**支持的打印类型：**
- 基本类型：UINT8/16/32/64, INT8/16/32/64, FLOAT32, BFLOAT16
- 格式化：SETW, SETPRECISION, FIXED, DEFAULTFLOAT
- 进制：HEX, OCT, DEC
- 特殊：TILESLICE（张量切片打印）

**TileSlice 配置：**
```cpp
struct SliceRange {
    uint8_t h0, h1, hs, w0, w1, ws;  // [h0:h1:hs, w0:w1:ws]

    // 预定义切片范围
    static SliceRange hw0_32_16();   // [0:32:16, 0:32:16]
    static SliceRange h0_32_w0();    // [0:32, 0]
    static SliceRange hw041();       // [0:4:1, 0:4:1]
};
```

#### 3.2.3 Fabric 网络通用定义 (fabric_common.h)

TT-Fabric 网络通信系统的共享定义，支持 1D 和 2D 路由。

**基本类型：**
```cpp
using chan_id_t = std::uint8_t;
using routing_plane_id_t = std::uint8_t;

static constexpr std::uint32_t CLIENT_INTERFACE_SIZE = 3280;
static constexpr std::uint32_t PACKET_WORD_SIZE_BYTES = 16;
static constexpr std::uint32_t MAX_MESH_SIZE = 256;
static constexpr std::uint32_t MAX_NUM_MESHES = 1024;
```

**以太网通道方向：**
```cpp
enum eth_chan_directions : std::uint8_t {
    EAST = 0, WEST = 1, NORTH = 2, SOUTH = 3, Z = 4, COUNT = 5
};
```

**压缩路由表：**
```cpp
template <std::uint32_t ArraySize>
struct __attribute__((packed)) direction_table_t {
    static constexpr uint32_t BITS_PER_COMPRESSED_ENTRY = 3;
    std::uint8_t packed_directions[ArraySize * 3 / 8];  // 3 bits per entry
};
```

**2D 压缩路由条目：**
```cpp
struct __attribute__((packed)) compressed_route_2d_t {
    // 位域布局：ns_hops(7) | ew_hops(7) | ns_dir(1) | ew_dir(1) | turn_point(7)
    uint32_t data;

    void set(uint8_t ns_hops, uint8_t ew_hops, uint8_t ns_dir,
             uint8_t ew_dir, uint8_t turn_point);
};
```

**路由编码函数：**
```cpp
// 1D 单播编码
inline void encode_1d_unicast(uint8_t num_hops, uint32_t* buffer, uint32_t num_words);

// 1D 组播编码
inline void encode_1d_multicast(uint8_t start_hop, uint8_t range_hops,
                                uint32_t* buffer, uint32_t num_words);

// 1D 稀疏组播编码（基于hop掩码）
template <typename HopMaskType>
inline void encode_1d_sparse_multicast(HopMaskType hop_mask, uint32_t& buffer);

// 2D 单播编码
inline void encode_2d_unicast(uint8_t ns_hops, uint8_t ew_hops, uint8_t ns_dir,
                              uint8_t ew_dir, uint8_t* buffer,
                              uint32_t max_buffer_size, bool prepend_one_hop);
```

**路由表结构：**
```cpp
struct routing_l1_info_t {
    RouterStateManager state_manager;                    // 32 bytes
    uint16_t my_mesh_id;                                 // 当前网格ID
    uint16_t my_device_id;                               // 当前设备ID
    direction_table_t<MAX_MESH_SIZE> intra_mesh_direction_table;      // 96 bytes
    direction_table_t<MAX_NUM_MESHES> inter_mesh_direction_table;     // 384 bytes
    union {
        intra_mesh_routing_path_t<1, false> routing_path_table_1d;    // 1024 bytes
        intra_mesh_routing_path_t<2, true> routing_path_table_2d;     // 1024 bytes
    };
    std::uint8_t exit_node_table[MAX_NUM_MESHES];        // 1024 bytes
};
// 总计：2576 bytes
```

**路由器命令：**
```cpp
enum class RouterCommand : std::uint32_t {
    RUN = 0,      // 正常转发消息和信用
    PAUSE = 1,    // 暂停处理
    DRAIN = 3,    // 丢弃消息（/dev/null模式）
    RETRAIN = 4   // 链路重训练
};
```

#### 3.2.4 标志模板类 (flags.hpp)

类型安全的枚举标志类模板。

```cpp
template <typename E>
class Flags {
    using Underlying = std::underlying_type_t<E>;

    constexpr Flags operator|(E other) const noexcept;
    constexpr Flags operator|(Flags other) const noexcept;
    constexpr bool test(E single) const noexcept;
    void set(E single, bool value = true) noexcept;
    constexpr Underlying raw() const noexcept;
};
```

#### 3.2.5 内核结构 (kernel_structs.h)

循环缓冲区（CB）索引定义。

**CBIndex 枚举（64个索引）：**
```cpp
enum CBIndex : std::uint8_t {
    c_0 = 0, c_1 = 1, ..., c_63 = 63
};
```

**传统CB枚举（已弃用）：**
```cpp
enum CB : std::uint8_t {
    c_in0 = 0, ..., c_in7 = 7,           // 计算输入
    dataflow0 = 8, ..., dataflow7 = 15,  // 数据流
    c_out0 = 16, ..., c_out7 = 23,       // 计算输出
    c_intermed0 = 24, ..., c_intermed7 = 31  // 中间结果
};
```

**目标模式：**
```cpp
enum DstMode : std::uint8_t {
    Full = 0, Half = 1, Tile = 2, NUM_DST_MODES = 3
};
```

#### 3.2.6 性能分析器通用定义 (profiler_common.h)

设备端性能分析的共享定义。

**缓冲区索引：**
```cpp
enum BufferIndex {
    ID_HH, ID_HL, ID_LH, ID_LL,           // 标识符
    GUARANTEED_MARKER_1_H, GUARANTEED_MARKER_1_L,  // 保证标记
    CUSTOM_MARKERS                        // 自定义标记起始
};
```

**控制缓冲区：**
```cpp
enum ControlBuffer {
    HOST_BUFFER_END_INDEX_BR_ER,          // 主机缓冲区结束索引
    DEVICE_BUFFER_END_INDEX_BR_ER,        // 设备缓冲区结束索引
    FW_RESET_H, FW_RESET_L,               // 固件重置标记
    DRAM_PROFILER_ADDRESS_DEFAULT,        // DRAM分析器地址
    RUN_COUNTER,                          // 运行计数器
    NOC_X, NOC_Y,                         // NOC坐标
    PROFILER_DONE,                        // 分析完成标记
};
```

**数据包类型：**
```cpp
enum PacketTypes {
    ZONE_START, ZONE_END, ZONE_TOTAL,     // 区域计时
    TS_DATA, TS_EVENT, TS_DATA_16B        // 时间戳数据
};
```

**缓冲区大小：**
```cpp
constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 32;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = 128;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
```

#### 3.2.7 张量访问器参数配置 (arg_config.hpp)

张量访问器的编译时配置标志。

```cpp
enum class ArgConfig : uint8_t {
    None = 0,
    Sharded = 1 << 0,           // 分片张量
    IsDram = 1 << 1,            // DRAM存储
    RuntimeRank = 1 << 2,       // 运行时rank
    RuntimeNumBanks = 1 << 3,   // 运行时bank数
    RuntimeTensorShape = 1 << 4,// 运行时张量形状
    RuntimeShardShape = 1 << 5, // 运行时分片形状
    RuntimeBankCoords = 1 << 6, // 运行时bank坐标
    Runtime = RuntimeRank | RuntimeNumBanks | RuntimeTensorShape |
              RuntimeShardShape | RuntimeBankCoords
};
```

### 3.3 关键代码分析

#### 条件编译模式

hostdevcommon 的头文件使用条件编译区分主机和设备代码：

```cpp
#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)  // 主机代码
    // 主机端实现
#else  // 设备代码
    // 设备端实现
#endif
```

这种模式允许同一份头文件在两种编译环境下提供适当的实现。

#### Fabric 路由编码示例

1D 单播路由编码生成 2-bit 每跳的命令序列：

```cpp
inline void encode_1d_unicast(uint8_t num_hops, uint32_t* buffer, uint32_t num_words) {
    // 零初始化缓冲区
    for (uint32_t i = 0; i < num_words; i++) buffer[i] = 0;
    if (num_hops == 0) return;  // 自路由

    // 计算写入位置
    const uint32_t write_hop_index = num_hops - 1;
    const uint32_t write_word_index = write_hop_index / 16;  // 16 hops per word
    const uint32_t write_bit_pos = (write_hop_index % 16) * 2;

    // 生成模式：FORWARD_ONLY (0b10) 直到最后一跳，然后 WRITE_ONLY (0b01)
    const uint32_t forward_mask = (1U << write_bit_pos) - 1;
    const uint32_t write_word_value =
        (FWD_ONLY_FIELD & forward_mask) | (WRITE_ONLY << write_bit_pos);

    // 填充缓冲区
    for (uint32_t i = 0; i < num_words; i++) {
        if (i < write_word_index) buffer[i] = FWD_ONLY_FIELD;
        else if (i == write_word_index) buffer[i] = write_word_value;
    }
}
```

---

## 4. graph/ - 图相关功能

### 4.1 模块概述

`graph/` 模块提供操作图跟踪功能，用于记录和分析 TT-Metal 程序的执行流程。这对于调试、性能分析和可视化非常重要。

### 4.2 关键代码分析

#### GraphTracker 单例

```cpp
class GraphTracker {
    std::vector<std::shared_ptr<IGraphProcessor>> processors;
    std::shared_ptr<IGraphHooks> hook;
    std::unordered_set<const Buffer*> hooked_buffers;
    std::mutex hooked_buffers_mutex;

public:
    static GraphTracker& instance();
    bool is_enabled() const;

    // 处理器管理
    void push_processor(const std::shared_ptr<IGraphProcessor>& new_processor);
    void pop_processor();

    // 跟踪操作
    void track_allocate(const Buffer* buffer);
    void track_deallocate(Buffer* buffer);
    void track_allocate_cb(const CoreRangeSet& core_range_set, uint64_t addr,
                           uint64_t size, bool is_globally_allocated, const IDevice* device);
    void track_program(Program* program, const IDevice* device);

    // Hook操作（拦截并重定向）
    bool hook_allocate(const Buffer* buffer);
    bool hook_deallocate(Buffer* buffer);
    bool hook_write_to_device(const Buffer* buffer);
    bool hook_read_from_device(Buffer* buffer);
    bool hook_program(Program* program);
};
```

#### 跟踪与Hook的区别

- **Track（跟踪）**：记录操作的发生，用于事后分析
- **Hook（钩子）**：拦截操作并可能重定向其行为，用于测试和模拟

#### 使用模式

```cpp
// 启用图跟踪
GraphTracker::instance().push_processor(std::make_shared<MyProcessor>());

// 执行操作（自动被跟踪）
auto buffer = device->allocate_buffer(...);

// 结束跟踪
GraphTracker::instance().pop_processor();
```

---

## 5. logging/ - 日志系统

### 5.1 模块概述

`logging/` 模块提供 TT-Metal 的日志功能，基于 spdlog 库实现，支持多种日志级别和运行时级别调整。

### 5.2 关键代码分析

#### 日志级别定义

```cpp
enum class level {
    trace,     // 最详细的执行跟踪
    debug,     // 调试信息
    info,      // 一般信息
    warn,      // 警告
    error,     // 错误
    critical,  // 严重错误
    off        // 关闭日志
};
```

#### 级别设置

```cpp
void set_level(level lvl) {
    ::tt::LoggerRegistry::instance().set_level(to_spdlog_level(lvl));
}

spdlog::level::level_enum to_spdlog_level(level lvl) {
    switch (lvl) {
        case level::trace: return spdlog::level::trace;
        case level::debug: return spdlog::level::debug;
        // ...
    }
}
```

#### 使用方式

日志系统通过 `tt-logger` 库提供的宏使用：

```cpp
log_info(tt::LogMetal, "Message: {}", value);
log_warning(tt::LogMetal, "Warning: {}", warning);
log_error(tt::LogMetal, "Error: {}", error);
```

---

## 6. 设计模式总结

### 6.1 单例模式

多个模块使用单例模式管理全局状态：

- `GraphTracker::instance()` - 图跟踪器
- `MemoryReporter::inst()` - 内存报告器
- `MetalContext::instance()` - Metal 运行时上下文

### 6.2 RAII 模式

资源管理广泛使用 RAII：

- `ScopedTimer` - 自动计时
- `MemoryPin` - 自动引用计数管理
- `HostBuffer` - 自动内存生命周期

### 6.3 策略模式

线程池实现使用策略模式：

- `ThreadPool` 接口
- `DeviceBoundThreadPool` - NUMA 感知实现
- `PassThroughThreadPool` - 直通实现

### 6.4 模板元编程

多处使用模板实现编译期优化：

- `Flags<E>` - 类型安全的枚举标志
- `parse_env<T>` - 类型安全的环境变量解析
- `round_to_power_of_2<Integer>` - 编译期整数运算

### 6.5 条件编译

hostdevcommon 使用条件编译实现主机/设备代码共享：

```cpp
#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    // 主机代码
#else
    // 设备代码
#endif
```

### 6.6 位域压缩

Fabric 路由系统大量使用位域压缩减少内存占用：

- 3-bit 方向表项（8个方向）
- 2-bit 每跳命令（1D路由）
- 23-bit 2D路由压缩（ns_hops:7, ew_hops:7, ns_dir:1, ew_dir:1, turn_point:7）

### 6.7 迭代器模式

坐标范围类提供标准迭代器接口：

```cpp
for (const auto& coord : core_range) { ... }
for (const auto& coord : mesh_coordinate_range) { ... }
```

### 6.8 工厂模式

线程池创建使用工厂函数：

```cpp
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads);
std::shared_ptr<ThreadPool> create_passthrough_thread_pool();
```

---

## 文件位置汇总

| 模块 | 文件路径 |
|------|----------|
| common/core_coord | `/tmp/tt-metal/tt_metal/common/core_coord.hpp`, `/tmp/tt-metal/tt_metal/common/core_coord.cpp` |
| common/shape | `/tmp/tt-metal/tt_metal/common/shape.cpp`, `/tmp/tt-metal/tt_metal/common/shape2d.cpp`, `/tmp/tt-metal/tt_metal/common/shape_base.cpp` |
| common/core_assignment | `/tmp/tt-metal/tt_metal/common/core_assignment.hpp`, `/tmp/tt-metal/tt_metal/common/core_assignment.cpp` |
| common/thread_pool | `/tmp/tt-metal/tt_metal/common/thread_pool.hpp`, `/tmp/tt-metal/tt_metal/common/thread_pool.cpp` |
| common/queue | `/tmp/tt-metal/tt_metal/common/multi_producer_single_consumer_queue.hpp` |
| common/timer | `/tmp/tt-metal/tt_metal/common/scoped_timer.hpp` |
| common/hash | `/tmp/tt-metal/tt_metal/common/stable_hash.hpp` |
| common/env | `/tmp/tt-metal/tt_metal/common/env_lib.hpp` |
| common/work_split | `/tmp/tt-metal/tt_metal/common/work_split.cpp` |
| common/executor | `/tmp/tt-metal/tt_metal/common/executor.hpp` |
| common/memory | `/tmp/tt-metal/tt_metal/common/memory_pin.cpp`, `/tmp/tt-metal/tt_metal/common/host_buffer.cpp` |
| common/mesh_coord | `/tmp/tt-metal/tt_metal/common/mesh_coord.cpp` |
| common/rounding | `/tmp/tt-metal/tt_metal/common/tt_rounding.h` |
| common/backend_types | `/tmp/tt-metal/tt_metal/common/tt_backend_api_types.hpp`, `/tmp/tt-metal/tt_metal/common/tt_backend_api_types.cpp` |
| detail/memory_reporter | `/tmp/tt-metal/tt_metal/detail/reports/memory_reporter.hpp`, `/tmp/tt-metal/tt_metal/detail/reports/memory_reporter.cpp` |
| detail/report_utils | `/tmp/tt-metal/tt_metal/detail/reports/report_utils.hpp` |
| hostdevcommon/common_values | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` |
| hostdevcommon/dprint | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/dprint_common.h` |
| hostdevcommon/fabric | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` |
| hostdevcommon/flags | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/flags.hpp` |
| hostdevcommon/kernel_structs | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h` |
| hostdevcommon/profiler | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h` |
| hostdevcommon/tensor_accessor | `/tmp/tt-metal/tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp` |
| graph/tracking | `/tmp/tt-metal/tt_metal/graph/graph_tracking.cpp` |
| logging/logging | `/tmp/tt-metal/tt_metal/logging/logging.cpp` |
