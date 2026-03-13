# fabric/ 模块源码解析

## 1. 模块概述

### 1.1 模块职责

Fabric模块是TT-Metal框架的核心通信层，负责实现芯片间(Chip-to-Chip)的高速数据传输。它提供了一个可扩展的、低延迟的互联网络抽象，支持多种拓扑结构（Mesh、Torus、Ring等）。

**核心功能**：
- **路由管理**：基于以太网通道的2D/1D路由，支持虚拟通道(VC0/VC1)分离流量
- **数据移动**：通过ERISC (Ethernet RISC) 数据移动器实现零拷贝传输
- **流控制**：基于credit的流控机制，防止网络拥塞
- **拓扑抽象**：支持物理拓扑发现和逻辑拓扑映射
- **集合通信**：为CCL (Collective Communication Library) 提供底层支持

### 1.2 在系统中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│              CCL (Collective Communication)                 │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Fabric Layer                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │  │
│  │  │  Builder │  │ Control  │  │   Hardware Interface │ │  │
│  │  │  System  │  │  Plane   │  │     (ERISC/MUX)      │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    HAL / Device Layer                       │
├─────────────────────────────────────────────────────────────┤
│              UMD (Unified Metal Driver)                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 与其他模块的交互

- **Control Plane**: 提供路由表生成、拓扑发现和芯片映射
- **HAL (Hardware Abstraction Layer)**: 提供底层硬件访问接口
- **Program/Kernels**: Fabric kernel作为ERISC固件运行
- **Metal Context**: 全局访问点，管理Fabric生命周期

---

## 2. 目录结构

```
tt_metal/fabric/
├── builder/                    # Fabric构建系统
│   ├── connection_registry.*   # 连接注册表（测试/验证）
│   ├── connection_writer_adapter.*  # 连接写入适配器
│   ├── fabric_builder_config.* # 构建配置（通道数、内存区域）
│   ├── fabric_builder_helpers.*# 构建辅助函数
│   ├── fabric_channel_allocator.*   # 通道内存分配器基类
│   ├── fabric_core_placement.* # 核心放置策略
│   ├── fabric_remote_channels_allocator.*  # 远程通道分配
│   ├── fabric_static_sized_channels_allocator.*  # 静态大小通道分配
│   ├── router_connection_mapping.*  # 路由器连接映射
│   └── static_sized_channel_connection_writer_adapter.*
│
├── ccl/                        # 集合通信支持
│   └── ccl_common.*            # CCL通用接口
│
├── hw/inc/                     # 硬件接口头文件
│   ├── api_common.h            # 通用API定义
│   ├── fabric_config.h         # 运行时配置读取
│   ├── fabric_direction_table_interface.h  # 方向表接口
│   ├── fabric_routing_mode.h   # 路由模式定义
│   ├── fabric_routing_path_interface.h     # 路由路径接口
│   └── edm_fabric/             # ERISC数据移动器Fabric实现
│       ├── adapters/           # 适配器模式实现
│       ├── datastructures/     # 数据结构定义
│       ├── telemetry/          # 遥测和性能分析
│       ├── compile_time_arg_tmp.hpp    # 编译时参数模板
│       ├── edm_fabric_flow_control_helpers.hpp  # 流控辅助
│       ├── edm_fabric_utils.hpp        # 工具函数
│       ├── edm_fabric_worker_adapters.hpp       # Worker适配器
│       ├── edm_handshake.hpp           # 握手协议
│       ├── fabric_channel_traits.hpp   # 通道特性
│       ├── fabric_connection_interface.hpp       # 连接接口
│       ├── fabric_connection_manager.hpp         # 连接管理器
│       ├── fabric_edm_packet_header_validate.hpp # 包头验证
│       ├── fabric_edm_packet_transmission.hpp    # 包传输
│       ├── fabric_erisc_datamover_channels.hpp   # ERISC通道
│       ├── fabric_erisc_router_ct_args.hpp       # 编译时参数
│       ├── fabric_erisc_router_speedy_path.hpp   # 快速路径
│       ├── fabric_router_adapter.hpp     # 路由器适配器
│       ├── fabric_router_flow_control.hpp# 路由器流控
│       ├── fabric_static_channels_ct_args.hpp    # 静态通道参数
│       ├── fabric_stream_regs.hpp        # 流寄存器
│       └── ...
│
├── impl/kernels/               # 内核实现
│   └── edm_fabric/             # ERISC Fabric内核
│       └── tt_fabric_mux.cpp   # MUX内核主程序
│
├── config/                     # 配置文件
│   └── channel_trimming_overrides/  # 通道裁剪覆盖
│
├── mesh_graph_descriptors/     # 网格图描述符
│   ├── n150_mesh_graph_descriptor.textproto
│   ├── n300_mesh_graph_descriptor.textproto
│   ├── t3k_mesh_graph_descriptor.textproto
│   ├── tg_mesh_graph_descriptor.textproto
│   └── ... (各种集群配置)
│
├── protobuf/                   # Protobuf定义
│   └── mesh_graph_descriptor.proto  # MGD格式定义
│
├── serialization/              # 序列化支持
│
├── cabling_descriptors/        # 线缆描述符
│
├── debug/                      # 调试工具
│
├── fabric_types.cpp            # Fabric类型实现
├── fabric_context.*            # Fabric上下文（核心配置）
├── fabric_builder.*            # Fabric构建器主类
├── fabric_builder_context.*    # 构建时上下文
├── fabric.cpp                  # 公共API实现
├── fabric_edm_packet_header.hpp # 包头部定义（关键）
├── fabric_host_utils.*         # 主机端工具
├── fabric_init.*               # 初始化
├── fabric_mux_config.cpp       # MUX配置
├── fabric_router_builder.*     # 路由器构建器基类
├── fabric_router_channel_mapping.*  # 通道映射
├── fabric_switch_manager.cpp   # 交换机管理
├── fabric_tensix_builder.*     # Tensix构建器
├── fabric_tensix_builder_impl.*# Tensix实现
├── erisc_datamover_builder.*   # ERISC数据移动器构建器
├── control_plane.cpp           # 控制平面（核心）
├── mesh_graph.cpp              # 网格图实现
├── mesh_graph_descriptor.cpp   # MGD解析
├── routing_table_generator.cpp # 路由表生成
├── topology_mapper.cpp         # 拓扑映射
├── topology_solver.cpp         # 拓扑求解
├── physical_system_discovery.* # 物理系统发现
├── physical_system_descriptor.*# 物理系统描述
├── physical_grouping_descriptor_*.cpp  # 物理分组描述
├── compressed_direction_table.* # 压缩方向表
├── compressed_routing_path.*   # 压缩路由路径
├── channel_trimming_*.cpp      # 通道裁剪
└── ...
```

---

## 3. 核心组件解析

### 3.1 Builder系统

Builder系统负责Fabric的构建时配置，包括通道分配、内存布局、路由器配置等。

#### 3.1.1 FabricBuilder

`FabricBuilder`是构建流程的主控制器，采用分阶段构建模式：

```cpp
class FabricBuilder {
public:
    // 构建阶段（必须按顺序调用）
    void discover_channels();          // 发现活跃以太网通道
    void create_routers();             // 创建路由器构建器
    void connect_routers();            // 连接路由器
    void compile_ancillary_kernels();  // 编译辅助内核（如MUX）
    void create_kernels();             // 创建主ERISC内核
};
```

**构建流程**：
1. **Discover**: 查询Control Plane获取每个方向的活跃以太网通道
2. **Create Routers**: 为每个通道创建`FabricRouterBuilder`（可能是`ComputeMeshRouterBuilder`）
3. **Connect**: 建立路由器间的双向连接，配置虚拟通道(VC)
4. **Compile Ancillary**: 编译Tensix MUX内核（UDM模式下）
5. **Create Kernels**: 生成最终的ERISC路由器内核

#### 3.1.2 通道分配器

Fabric支持两种通道分配策略：

**静态大小通道分配器** (`FabricStaticSizedChannelsAllocator`)：
- 编译时确定缓冲区大小和数量
- 适用于已知工作负载的场景
- 更好的性能（无运行时分配开销）

```cpp
// 通道配置常量（来自fabric_builder_config.hpp）
namespace builder_config {
    static constexpr std::size_t MAX_NUM_VCS = 2;  // VC0和VC1
    static constexpr std::size_t num_sender_channels_1d = 2;
    static constexpr std::size_t num_sender_channels_2d = 8;
    static constexpr std::size_t num_receiver_channels_1d = 1;
    static constexpr std::size_t num_receiver_channels_2d = 2;
}
```

#### 3.1.3 ConnectionRegistry

用于测试和验证的连接注册表：

```cpp
enum class ConnectionType {
    INVALID,
    INTRA_MESH,   // 同Mesh内设备间连接
    MESH_TO_Z,    // Mesh路由器到Z路由器
    Z_TO_MESH,    // Z路由器到Mesh路由器
};

struct RouterConnectionRecord {
    FabricNodeId source_node;
    RoutingDirection source_direction;
    FabricNodeId dest_node;
    RoutingDirection dest_direction;
    ConnectionType connection_type;
    // ...
};
```

### 3.2 CCL (Collective Communication)

CCL模块提供集合通信操作的底层支持：

```cpp
// ccl/ccl_common.hpp
tt::tt_metal::KernelHandle generate_edm_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id);
```

Fabric为CCL提供：
- 可靠的芯片间数据传输
- 多播(Multicast)和单播(Unicast)语义
- 原子操作支持（Atomic Increment）
- 流控制集成

### 3.3 Hardware Interface

#### 3.3.1 包头部格式

`fabric_edm_packet_header.hpp`定义了Fabric包的核心结构：

```cpp
struct PacketHeader {
    NocSendType noc_send_type;           // NOC发送类型
    ChipSendType chip_send_type;         // 芯片发送类型（单播/多播）
    RoutingFields routing_fields;        // 路由字段
    uint16_t payload_size_bytes;         // 负载大小
    uint8_t src_ch_id;                   // 源芯片ID

    union {
        UnicastWriteCommand unicast_write;
        UnicastAtomicIncCommand unicast_seminc;
        UnicastScatterWriteCommand unicast_scatter_write;
        // ... 其他命令类型
    } command_fields;
};
```

**路由字段编码**：
- **1D路由**: 使用hop count，支持最多63跳
- **2D路由**: 使用路由缓冲区，每个字节编码一跳，支持最多67跳

```cpp
struct RoutingFields {
    static constexpr uint32_t HOP_DISTANCE_MASK = 0x3F;  // 6位跳数
    static constexpr uint32_t RANGE_MASK = 0x1C0;        // 多播范围
    uint32_t value;
};
```

#### 3.3.2 Worker到EDM的适配器

`edm_fabric_worker_adapters.hpp`提供了Worker核心与ERISC数据移动器之间的通信接口：

```cpp
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS>
struct WorkerToFabricEdmSenderImpl {
    // 连接管理
    void open();      // 建立连接
    void close();     // 关闭连接

    // 数据传输
    void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size);
    void send_payload_non_blocking_from_address(uint32_t source_address, size_t size_bytes);

    // 流控制
    bool edm_has_space_for_packet() const;
    void wait_for_empty_write_slot() const;
};
```

**流控制协议**：
- 基于读写指针的credit机制
- Worker维护本地写指针(wrptr)
- 从EDM读取远程读指针(rdptr)
- 差值决定是否有空间发送新包

#### 3.3.3 ERISC通道实现

`fabric_erisc_datamover_channels.hpp`实现了ERISC侧的通道逻辑：

```cpp
// 静态大小的发送通道
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class StaticSizedSenderEthChannel : public SenderEthChannelInterface<...> {
public:
    size_t get_buffer_address_impl() const;
    void advance_to_next_cached_buffer_slot_addr_impl();
};

// 静态大小的接收通道
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class StaticSizedEthChannelBuffer : public EthChannelBufferInterface<...> {
public:
    template <typename T>
    volatile T* get_packet_header_impl(const BufferIndex& buffer_index) const;

#if defined(COMPILE_FOR_ERISC)
    bool eth_is_acked_or_completed_impl(const BufferIndex& buffer_index) const;
#endif
};
```

#### 3.3.4 流寄存器分配

`StreamRegAssignments`定义了所有流寄存器的用途：

```cpp
struct StreamRegAssignments {
    // 包发送/确认流ID
    static constexpr uint32_t to_receiver_0_pkts_sent_id = 0;   // VC0以太网接收
    static constexpr uint32_t to_receiver_1_pkts_sent_id = 1;   // VC1以太网接收
    static constexpr uint32_t to_sender_0_pkts_acked_id = 2;    // VC0发送通道0确认
    // ...

    // 接收通道空闲槽位流ID（来自下游的credit）
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_1 = 14;
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_2 = 15;
    // ...

    // 发送通道空闲槽位流ID
    static constexpr uint32_t sender_channel_0_free_slots_stream_id = 22;
    // ...
};
```

### 3.4 Implementation

#### 3.4.1 MUX内核

`impl/kernels/tt_fabric_mux.cpp`是Tensix MUX内核的实现：

```cpp
void kernel_main() {
    // 初始化状态
    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::STARTED;

    // 建立与Fabric路由器的连接
    auto fabric_connection = tt::tt_fabric::FabricMuxToEdmSender::build_from_args<CORE_TYPE>(rt_args_idx);
    fabric_connection.open<use_worker_allocated_credit_address>();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;

    // 主循环：转发数据
    while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
        // 处理全大小通道
        for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
            forward_data<NUM_BUFFERS_FULL_SIZE_CHANNEL>(...);
        }
        // 处理仅头部通道
        for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
            forward_data<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(...);
        }
    }

    fabric_connection.close();
    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}
```

#### 3.4.2 包传输实现

`fabric_edm_packet_transmission.hpp`实现了包的实际传输逻辑：

```cpp
// 单播到本地芯片
void execute_chip_unicast_to_local_chip_impl(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint32_t transaction_id,
    uint8_t rx_channel_id) {

    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE:
            noc_async_write_one_packet_with_trid(...);
            break;
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC:
            noc_semaphore_inc(...);
            break;
        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE:
            // 分散写入：最多4个目标
            break;
        // ...
    }
}
```

---

## 4. 通信流程

### 4.1 Fabric初始化流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  MeshGraph  │────▶│ ControlPlane│────▶│FabricContext│────▶│FabricBuilder│
│   加载      │     │   初始化    │     │   创建      │     │   构建      │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  内核创建   │◀────│ 路由器连接  │◀────│ 路由器创建  │◀────│ 通道发现    │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 4.2 数据包发送流程

```
Worker Core                              ERISC Router                         Remote Router
     │                                        │                                    │
     │  1. 检查credit (edm_has_space)         │                                    │
     │───────────────────────────────────────▶│                                    │
     │                                        │                                    │
     │  2. 写入包到EDM缓冲区                  │                                    │
     │  send_payload_non_blocking_from_address│                                    │
     │───────────────────────────────────────▶│                                    │
     │                                        │  3. 通过以太网发送                 │
     │                                        │───────────────────────────────────▶│
     │                                        │                                    │
     │  4. 接收credit更新                     │  5. 接收ACK                        │
     │◀───────────────────────────────────────│◀───────────────────────────────────│
     │                                        │                                    │
```

### 4.3 流控制细节

**Credit-based Flow Control**：

```cpp
// Worker侧逻辑
template <size_t num_slots = 1>
FORCE_INLINE bool edm_has_space_for_packet() const {
    invalidate_l1_cache();
    auto used_slots = this->buffer_slot_write_counter.counter - *this->edm_buffer_local_free_slots_read_ptr;
    return used_slots < this->num_buffers_per_channel;
}

// EDM侧逻辑：当缓冲区被释放时
FORCE_INLINE void notify_worker_of_free_space() {
    noc_inline_dw_write(
        this->cached_worker_semaphore_address,
        local_read_counter.counter,
        0xf,
        WORKER_HANDSHAKE_NOC);
}
```

### 4.4 路由决策流程

```cpp
// 2D路由：基于方向表
RoutingDirection get_output_direction(const PacketHeader& header) {
    if (header.routing_fields.is_2d_routing()) {
        // 从路由缓冲区读取下一跳
        uint8_t next_hop = routing_buffer[hop_index];
        return decode_direction(next_hop);
    } else {
        // 1D路由：基于跳数计数
        if (header.routing_fields.hop_count == 0) {
            return RoutingDirection::LOCAL;  // 到达目标
        }
        return forwarding_direction;  // 继续转发
    }
}
```

---

## 5. 设计模式与实现技巧

### 5.1 CRTP (Curiously Recurring Template Pattern)

广泛用于硬件接口层，实现零开销抽象：

```cpp
// 基类定义接口
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS, typename DERIVED_T>
class EthChannelBufferInterface {
public:
    FORCE_INLINE size_t get_buffer_address(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->get_buffer_address_impl(buffer_index);
    }
};

// 派生类实现具体逻辑
class StaticSizedEthChannelBuffer : public EthChannelBufferInterface<...> {
    FORCE_INLINE size_t get_buffer_address_impl(const BufferIndex& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }
};
```

### 5.2 编译时参数传递

使用模板和constexpr实现编译时配置：

```cpp
// 编译时参数模板
template <size_t Index>
constexpr size_t get_compile_time_arg_val() {
    return COMPILE_TIME_ARGS[Index];
}

// 使用示例
constexpr size_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val<0>();
constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val<1>();
```

### 5.3 类型安全的强类型ID

```cpp
// 使用StrongType模式
using MeshId = tt::stl::StrongType<uint32_t, struct MeshIdTag>;
using FabricNodeId = tt::stl::StrongType<uint32_t, struct FabricNodeIdTag>;

// 好处：编译时防止混淆不同类型的ID
void process(MeshId mesh_id, ChipId chip_id);  // 类型安全
```

### 5.4 虚拟通道(Virtual Channel)分离

Fabric使用两个虚拟通道隔离流量：

```cpp
// VC0: 主数据路径
// - Worker到Worker通信
// - 同Mesh内路由

// VC1: 跨Mesh流量
// - 不同Mesh间的转发
// - 避免死锁的流量分离

// 发送通道分配（2D模式）
// VC0: 4个发送通道 (Worker + 3个方向)
// VC1: 4个发送通道 (跨Mesh流量)
```

### 5.5 内存布局优化

```cpp
// 通道缓冲区布局（重复N次）
// ┌─────────────────┐ ◄── channel_base_address
// │     Header      │
// ├─────────────────┤
// │                 │
// │    Payload      │
// │                 │
// ├─────────────────┤
// │  Channel Sync   │
// └─────────────────┘

// 预计算缓冲区地址避免运行时乘法
for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
    this->buffer_addresses[i] = channel_base_address + i * max_eth_payload_size_in_bytes;
}
```

### 5.6 延迟隐藏技术

```cpp
// 非阻塞发送 + 批量刷新
void send_payload_flush_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
    // 1. 启动非阻塞写入
    noc_async_write(source_address, dest_address, size_bytes);

    // 2. 更新指针（不等待）
    post_send_payload_increment_pointers();
}

// 稍后统一刷新
void flush_all_writes() {
    noc_async_writes_flushed();
}
```

---

## 6. 源码注释摘录

### 6.1 关于流控制协议

```cpp
/*
 * ### Flow Control Protocol:
 * The flow control protocol is rd/wr ptr based and is implemented as follows (from the worker's perspective):
 * The adapter has a local write pointer (wrptr) which is used to track the next buffer slot to write to. The adapter
 * also has a local memory slot that holds the remote read pointer (rdptr) of the EDM. The adapter uses the difference
 * between these two pointers (where rdptr trails wrptr) to determine if the EDM has space to accept a new packet.
 *
 * As the adapter writes into the EDM, it updates the local wrptr. As the EDM reads from its local L1 channel buffer,
 * it will notify the worker/adapter (here) by updating the worker remote_rdptr to carry the value of the EDM rdptr.
 */
```

### 6.2 关于2D路由头部大小

```cpp
// 2D routing: buffer tiers optimized to maximize capacity per header size
// Header base = 61B, aligned to 16B boundaries
// Strategy: One tier per header size (max capacity) to avoid bloat
//   80B:  61+19=80  (max capacity)
//   96B:  61+35=96  (max capacity)
//   112B: 61+51=112 (max capacity)
//   128B: 61+67=128 (max capacity)
```

### 6.3 关于连接管理器

```cpp
/*
 * FabricConnectionManager
 *
 * Advanced usage API: build_mode
 * Allow the user to opt-in to 3 build modes:
 *
 * BUILD_ONLY: just build the connection manager but don't open a connection
 *
 * BUILD_AND_OPEN_CONNECTION: build the connection manager and open a connection, wait for connection to be fully
 *         open and established before returning
 *
 * BUILD_AND_OPEN_CONNECTION_START_ONLY: build the connection manager and send the connection open request to
 *         fabric but don't wait for the connection readback to complete before returning.
 *         !!! IMPORTANT !!!
 *         User must call open_finish() manually, later, if they use this mode.
 */
```

### 6.4 关于ERISC通道结构

```cpp
/* Ethernet channel structure is as follows (for both sender and receiver):
              &header->  |----------------|\  <-  channel_base_address
                         |    header      | \
             &payload->  |----------------|  \
                         |                |   |- repeated n times
                         |    payload     |  /
                         |                | /
                         |----------------|/
*/
```

### 6.5 关于死锁避免

```cpp
// enable_deadlock_avoidance: When enabled, the EDM will use transaction IDs to ensure
// that writes to the same destination don't block each other, preventing deadlock
// scenarios in certain traffic patterns.
```

---

## 7. 关键文件索引

| 文件路径 | 描述 |
|---------|------|
| `/tmp/tt-metal/tt_metal/fabric/fabric_context.hpp` | Fabric上下文，核心配置管理 |
| `/tmp/tt-metal/tt_metal/fabric/fabric_builder.hpp` | Fabric构建器主类 |
| `/tmp/tt-metal/tt_metal/fabric/control_plane.cpp` | 控制平面实现，路由表生成 |
| `/tmp/tt-metal/tt_metal/fabric/fabric_edm_packet_header.hpp` | 包头部定义 |
| `/tmp/tt-metal/tt_metal/fabric/erisc_datamover_builder.hpp` | ERISC数据移动器构建器 |
| `/tmp/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | Worker适配器 |
| `/tmp/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | ERISC通道实现 |
| `/tmp/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` | 路由器流控 |
| `/tmp/tt-metal/tt_metal/fabric/builder/fabric_builder_config.hpp` | 构建配置 |
| `/tmp/tt-metal/tt_metal/fabric/mesh_graph.cpp` | 网格图实现 |
| `/tmp/tt-metal/tt_metal/fabric/MGD_README.md` | MGD格式文档 |

---

## 8. 总结

TT-Metal Fabric模块是一个精心设计的芯片间通信系统，具有以下特点：

1. **分层架构**：清晰的Builder/Control Plane/Hardware分层
2. **灵活拓扑**：支持Mesh、Torus、Ring等多种拓扑
3. **高性能**：零拷贝传输、基于credit的流控、编译时优化
4. **可扩展**：支持2D路由、虚拟通道、多Mesh系统
5. **类型安全**：广泛使用强类型ID防止错误
6. **硬件感知**：充分利用ERISC和NOC硬件特性

理解Fabric模块对于开发高性能分布式AI工作负载至关重要。
