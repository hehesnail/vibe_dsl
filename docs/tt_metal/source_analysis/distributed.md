# distributed/ 模块源码解析

## 1. 模块概述

### 1.1 模块职责

`distributed/` 模块是 TT-Metal 框架的分布式执行核心，提供以下关键功能：

- **Mesh 设备抽象**：将多个物理设备组织成逻辑网格（Mesh）结构
- **分布式执行**：支持跨多设备的程序编译、加载和执行
- **多主机通信**：基于 MPI 的多主机分布式计算支持
- **Socket 通信**：设备间高速数据传输机制
- **工作负载管理**：MeshWorkload 的编译、分发和执行
- **追踪捕获**：分布式程序执行的追踪和重放

### 1.2 在系统中的位置

```
tt_metal/
├── api/tt-metalium/          # 公共 API 头文件
│   ├── mesh_device.hpp       # MeshDevice 接口
│   ├── mesh_buffer.hpp       # MeshBuffer 接口
│   ├── mesh_command_queue.hpp # 命令队列接口
│   └── distributed.hpp       # 分布式工具函数
├── distributed/              # 本模块实现
│   ├── mesh_device.cpp       # MeshDevice 实现
│   ├── mesh_buffer.cpp       # MeshBuffer 实现
│   ├── fd_mesh_command_queue.cpp  # 快速调度命令队列
│   ├── mesh_workload.cpp     # 分布式工作负载
│   ├── multihost/            # 多主机支持
│   ├── flatbuffer/           # 序列化
│   └── experimental/         # 实验性功能
└── impl/                     # 底层实现
    └── dispatch/             # 调度系统
```

### 1.3 与其他模块的交互

| 模块 | 交互方式 | 用途 |
|------|----------|------|
| `impl/dispatch/` | 调用调度 API | 命令分发到设备 |
| `impl/device/` | 设备管理 | 物理设备控制 |
| `impl/program/` | 程序编译 | 内核编译和加载 |
| `experimental/fabric/` | 控制平面 | 网络拓扑和路由 |
| `llrt/` | 运行时 | 底层运行时支持 |

## 2. 目录结构

```
tt_metal/distributed/
├── CMakeLists.txt                    # 构建配置
├── sources.cmake                     # 源文件列表
│
├── distributed.cpp                   # 分布式工具函数
├── dispatch_context.cpp              # 调度上下文管理
│
# Mesh 核心组件
├── mesh_device.cpp/.hpp              # Mesh 设备实现
├── mesh_device_impl.hpp              # MeshDevice 内部实现
├── mesh_device_view.cpp/.hpp         # 设备视图管理
├── mesh_buffer.cpp                   # Mesh 缓冲区
├── mesh_command_queue_base.cpp/.hpp  # 命令队列基类
├── mesh_event.cpp                    # 事件管理
├── mesh_trace.cpp/.hpp               # 追踪捕获
├── mesh_workload.cpp/.hpp            # 工作负载管理
├── mesh_workload_utils.cpp/.hpp      # 工作负载工具
│
# 命令队列实现
├── fd_mesh_command_queue.cpp/.hpp    # 快速调度 (Fast Dispatch)
├── sd_mesh_command_queue.cpp/.hpp    # 慢速调度 (Slow Dispatch)
├── dummy_mesh_command_queue.cpp/.hpp # 虚拟命令队列
│
# Socket 通信系统
├── mesh_socket.cpp                   # Socket 实现
├── mesh_socket_utils.cpp/.hpp        # Socket 工具
├── mesh_socket_serialization.cpp/.hpp # Socket 序列化
├── h2d_socket.cpp                    # Host-to-Device Socket
├── d2h_socket.cpp                    # Device-to-Host Socket
│
# 系统 Mesh 管理
├── system_mesh.cpp                   # 系统级 Mesh 管理
├── system_mesh_translation_map.cpp/.hpp # 坐标映射
├── distributed_coordinate_translator.cpp/.hpp # 坐标转换
│
# 内存管理
├── pinned_memory.cpp/.hpp            # 固定内存管理
├── distributed_host_buffer.cpp       # 分布式主机缓冲区
│
# 多主机支持
├── multihost/
│   ├── distributed_context.cpp       # 分布式上下文工厂
│   ├── mpi_distributed_context.cpp/.hpp  # MPI 实现
│   └── single_host_context.cpp/.hpp  # 单主机回退
│
# 序列化
├── flatbuffer/
│   └── socket_peer_descriptor.fbs    # Socket 描述符 Schema
│
# 实验性功能
└── experimental/
    └── blitz_decode_pipeline.cpp     # Blitz 解码流水线
```

## 3. 核心组件解析

### 3.1 Multihost Support（多主机支持）

多主机支持通过抽象层实现，允许代码在单主机和多主机环境中无缝运行。

#### 3.1.1 架构设计

```cpp
// 抽象基类定义在 api/tt-metalium/distributed_context.hpp
class DistributedContext {
public:
    // 基础信息
    virtual Rank rank() const = 0;           // 当前进程 rank
    virtual Size size() const = 0;           // 总进程数
    virtual void barrier() const = 0;        // 同步屏障

    // 点对点通信
    virtual void send(Span<byte> buf, Rank dest, Tag tag) = 0;
    virtual void recv(Span<byte> buf, Rank src, Tag tag) = 0;
    virtual RequestPtr isend(Span<byte> buf, Rank dest, Tag tag) = 0;
    virtual RequestPtr irecv(Span<byte> buf, Rank src, Tag tag) = 0;

    // 集合通信
    virtual void broadcast(Span<byte> buf, Rank root) = 0;
    virtual void all_reduce(Span<byte> send, Span<byte> recv, ReduceOp op, DType dtype) = 0;
    virtual void all_gather(Span<byte> send, Span<byte> recv) = 0;
    virtual void reduce_scatter(...) = 0;

    // 通信域管理
    virtual ContextPtr duplicate() = 0;
    virtual ContextPtr split(Color color, Key key) = 0;
    virtual ContextPtr create_sub_context(Span<int> ranks) = 0;
};
```

#### 3.1.2 MPIContext（MPI 实现）

```cpp
// multihost/mpi_distributed_context.hpp
class MPIContext : public DistributedContext {
    MPI_Comm comm_;
    MPI_Group group_;
    int rank_, size_;

public:
    // 工厂方法
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();

    // MPI 特有功能
    bool supports_fault_tolerance() const override;  // ULFM 支持
    void revoke_and_shrink() override;               // 故障恢复
};
```

**关键特性**：
- 使用 `MPI_THREAD_MULTIPLE` 支持多线程
- 支持 ULFM（User Level Failure Mitigation）容错
- 自动 MPI 初始化和清理（通过 `std::atexit`）
- 通信域复制和分割支持

**类型映射**：
```cpp
// 数据类型转换
constexpr MPI_Datatype dtype_to_mpi(DType dt) {
    switch (dt) {
        case DType::INT8: return MPI_INT8_T;
        case DType::FLOAT32: return MPI_FLOAT;
        case DType::FLOAT64: return MPI_DOUBLE;
        // ... 更多类型
    }
}

// 归约操作转换
constexpr MPI_Op reduce_to_mpi(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM: return MPI_SUM;
        case ReduceOp::MAX: return MPI_MAX;
        case ReduceOp::MIN: return MPI_MIN;
        // ... 更多操作
    }
}
```

#### 3.1.3 SingleHostContext（单主机回退）

```cpp
// multihost/single_host_context.hpp
class SingleHostContext : public DistributedContext {
    int rank_ = 0;
    int size_ = 1;

public:
    // 大部分通信操作抛出异常（单主机不支持）
    void send(...) const override {
        TT_THROW("method send is unsupported for single-host distributed contexts.");
    }

    // all_gather 简化为内存拷贝
    void all_gather(Span<byte> send_buf, Span<byte> recv_buf) const override {
        std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
    }

    // barrier 为空操作
    void barrier() const override { return; }
};
```

#### 3.1.4 编译时选择

```cpp
// multihost/distributed_context.cpp
#if defined(OPEN_MPI)
    using ContextImpl = MPIContext;
#else
    using ContextImpl = SingleHostContext;
#endif

void DistributedContext::create(int argc, char** argv) {
    ContextImpl::create(argc, argv);
}
```

### 3.2 Flatbuffer Serialization（序列化）

#### 3.2.1 Schema 定义

```protobuf
// flatbuffer/socket_peer_descriptor.fbs
table SocketPeerDescriptor {
    config: SocketConfig;
    config_buffer_address: uint64;
    data_buffer_address: uint64;
    exchange_tag: uint32;
}

table SocketConfig {
    socket_connections: [SocketConnection];
    socket_mem_config: SocketMemoryConfig;
    sender_rank: uint32;
    receiver_rank: uint32;
    sender_mesh_id: uint32;
    receiver_mesh_id: uint32;
}

table SocketConnection {
    sender_core: MeshCoreCoord;
    receiver_core: MeshCoreCoord;
}

table MeshCoreCoord {
    device_coord: MeshCoordinate;
    core_coord: CoreCoord;
}
```

#### 3.2.2 序列化实现

```cpp
// mesh_socket_serialization.cpp
std::vector<uint8_t> serialize_to_bytes(const SocketPeerDescriptor& socket_peer_desc) {
    flatbuffers::FlatBufferBuilder builder;

    // 构建嵌套结构
    auto connections_vector_fb = to_flatbuffer(builder, socket_config.socket_connection_config);
    auto mem_config_fb = to_flatbuffer(builder, socket_config.socket_mem_config);

    // 创建 SocketConfig
    auto socket_config_fb = distributed::flatbuffer::CreateSocketConfig(
        builder, connections_vector_fb, mem_config_fb, ...);

    // 创建根对象
    auto socket_peer_desc_fb = distributed::flatbuffer::CreateSocketPeerDescriptor(
        builder, socket_config_fb, ...);

    builder.Finish(socket_peer_desc_fb);

    // 提取字节数据
    return std::vector<uint8_t>(
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize());
}

SocketPeerDescriptor deserialize_from_bytes(const std::vector<uint8_t>& data) {
    // 验证缓冲区
    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    TT_FATAL(distributed::flatbuffer::VerifySocketPeerDescriptorBuffer(verifier), ...);

    // 反序列化
    const auto* socket_peer_desc_fb = distributed::flatbuffer::GetSocketPeerDescriptor(data.data());
    // ... 提取字段
}
```

### 3.3 Distributed Execution（分布式执行）

#### 3.3.1 MeshDevice 架构

```cpp
// mesh_device_impl.hpp
class MeshDeviceImpl : public IDevice {
private:
    class ScopedDevices {
        std::vector<MaybeRemote<IDevice*>> devices_;
        std::map<ChipId, IDevice*> opened_local_devices_;
    } scoped_devices_;

    int mesh_id_;
    std::unique_ptr<MeshDeviceView> view_;
    std::shared_ptr<MeshDevice> parent_mesh_;
    std::vector<std::weak_ptr<MeshDevice>> submeshes_;

    // 命令队列
    tt::stl::SmallVector<std::unique_ptr<MeshCommandQueueBase>> mesh_command_queues_;

    // 线程池
    std::shared_ptr<ThreadPool> dispatch_thread_pool_;
    std::shared_ptr<ThreadPool> reader_thread_pool_;

    // 分布式上下文
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;
    std::shared_ptr<distributed::multihost::DistributedContext> active_distributed_context_;

public:
    // 设备管理
    static std::shared_ptr<MeshDevice> create(const MeshDeviceConfig& config, ...);
    std::shared_ptr<MeshDevice> create_submesh(const MeshShape& submesh_shape, ...);
    void reshape(const MeshShape& new_shape);

    // IDevice 接口实现
    tt::ARCH arch() const override;
    CoreCoord grid_size() const override;
    SystemMemoryManager& sysmem_manager() override;
    // ... 更多接口
};
```

#### 3.3.2 MeshBuffer 类型

```cpp
// 缓冲区配置变体
using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

// 复制缓冲区 - 数据在所有设备上相同
struct ReplicatedBufferConfig {
    size_t size;
};

// 分片缓冲区 - 数据分片分布在设备上
struct ShardedBufferConfig {
    Shape2D global_buffer_shape;      // 全局缓冲区形状
    Shape2D shard_shape;              // 分片形状
    ShardOrientation shard_orientation; // ROW_MAJOR 或 COL_MAJOR

    // 计算物理分片形状（处理 0 维度表示复制）
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
};

// MeshBuffer 创建
class MeshBuffer {
    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);

    // 获取特定设备的缓冲区
    Buffer* get_device_buffer(const MeshCoordinate& device_coord) const;
};
```

#### 3.3.3 MeshWorkload 执行流程

```cpp
// mesh_workload.cpp
class MeshWorkloadImpl {
    std::unordered_map<MeshCoordinateRange, Program> programs_;
    uint64_t id;

public:
    // 添加程序到指定设备范围
    void add_program(const MeshCoordinateRange& device_range, Program&& program) {
        // 检查范围重叠
        auto intersection = find_intersection(programs_, device_range);
        TT_FATAL(!intersection, "Program range overlaps with existing range");
        programs_[device_range] = std::move(program);
    }

    // 多阶段编译
    void compile(MeshDevice* mesh_device) {
        // 1. 编译内核二进制
        // 2. 分配和验证 CB
        // 3. 计算 L1 偏移

        if (programs_.size() == 1) {
            // 同质工作负载：主线程编译
            compile_program(programs_.begin()->first, mesh_device);
        } else {
            // 异质工作负载：多线程编译
            for (auto& [range, _] : programs_) {
                mesh_device->enqueue_to_thread_pool([=]() {
                    compile_program(range, mesh_device);
                });
            }
            mesh_device->wait_for_thread_pool();
        }
        finalize_offsets(mesh_device);
    }

    // 加载二进制到设备
    void load_binaries(MeshCommandQueue& mesh_cq) {
        // 分配最大大小的内核缓冲区
        uint32_t max_kernel_bin_buf_size = compute_max_binary_size();

        auto kernel_bin_buf = MeshBuffer::create(
            ReplicatedBufferConfig{.size = max_kernel_bin_buf_size},
            device_local_kernel_bin_buf_config,
            mesh_device);

        // 写入每个子网格的程序二进制
        for (auto& [range, program] : programs_) {
            mesh_cq.enqueue_write_shard_to_sub_grid(
                *kernel_bin_buf_view,
                program.impl().get_program_transfer_info().binary_data.data(),
                range,
                false);
        }
    }
};
```

#### 3.3.4 快速调度命令队列

```cpp
// fd_mesh_command_queue.hpp
class FDMeshCommandQueue final : public MeshCommandQueueBase {
private:
    // 共享状态
    std::shared_ptr<CQSharedState> cq_shared_state_;

    // 工作线程管理
    DispatchArray<uint32_t> expected_num_workers_completed_;
    DispatchArray<WorkerConfigBufferMgr> config_buffer_mgr_;

    // 预取缓存
    std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_;

    // 完成队列读取线程
    std::thread completion_queue_reader_thread_;
    MultiProducerSingleConsumerQueue<MeshCompletionReaderVariant> completion_queue_reads_;

    // 分布式上下文
    std::shared_ptr<distributed::multihost::DistributedContext> active_distributed_context_;

public:
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) override;
    MeshEvent enqueue_record_event(...) override;
    void enqueue_trace(const MeshTraceId& trace_id, bool blocking) override;

    // 追踪捕获
    void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) override;
    void record_end() override;

    // 完成队列读取循环
    void read_completion_queue();
};
```

**工作负载入队流程**：

```cpp
void FDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    auto lock = lock_api_function_();
    in_use_ = true;

    // 1. 确定子设备
    auto sub_device_ids = mesh_workload.impl().determine_sub_device_ids(mesh_device_);
    SubDeviceId sub_device_id = *(sub_device_ids.begin());

    // 2. 计算工作线程数
    uint32_t num_workers = 0;
    bool unicast_go_signals = mesh_workload.impl().runs_on_noc_unicast_only_cores();
    bool mcast_go_signals = mesh_workload.impl().runs_on_noc_multicast_only_cores();

    if (mcast_go_signals) {
        num_workers += mesh_device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }
    if (unicast_go_signals) {
        num_workers += mesh_device_->num_virtual_eth_cores(sub_device_id);
    }

    // 3. 为每个子网格写入程序命令
    for (auto& [sub_grid, program] : mesh_workload.impl().programs_) {
        write_program_cmds_to_subgrid(sub_grid, program_cmd_seq, ...);
    }

    // 4. 向未使用的子网格发送 Go 信号
    write_go_signal_to_unused_sub_grids(...);

    // 5. 更新工作线程计数
    update_expected_num_workers_completed(num_workers, sub_device_id);
}
```

### 3.4 Socket 通信系统

#### 3.4.1 MeshSocket 设计

```cpp
// Socket 端点类型
enum class SocketEndpoint { SENDER, RECEIVER };

// Socket 配置
struct SocketConfig {
    std::vector<SocketConnection> socket_connection_config;
    SocketMemoryConfig socket_mem_config;
    multihost::Rank sender_rank;
    multihost::Rank receiver_rank;
    std::optional<tt_fabric::MeshId> sender_mesh_id;
    std::optional<tt_fabric::MeshId> receiver_mesh_id;
    std::shared_ptr<multihost::DistributedContext> distributed_context;
};

// Socket 连接
struct SocketConnection {
    MeshCoreCoord sender_core;
    MeshCoreCoord receiver_core;
};
```

#### 3.4.2 多主机 Socket 建立流程

```cpp
// mesh_socket.cpp
class MeshSocket {
    void connect_with_peer(const std::shared_ptr<multihost::DistributedContext>& context) {
        // 生成本地端点描述符
        auto local_endpoint_desc = generate_local_endpoint_descriptor(*this, context->id());
        SocketPeerDescriptor remote_endpoint_desc;

        // 约定：
        // - 发送端先发送描述符，然后接收对端描述符
        // - 接收端先接收对端描述符，然后发送本地描述符
        // 这种非对称设计避免死锁
        if (socket_endpoint_type_ == SocketEndpoint::SENDER) {
            forward_descriptor_to_peer(local_endpoint_desc, ...);
            remote_endpoint_desc = receive_and_verify_descriptor_from_peer(local_endpoint_desc, ...);
        } else {
            remote_endpoint_desc = receive_and_verify_descriptor_from_peer(local_endpoint_desc, ...);
            forward_descriptor_to_peer(local_endpoint_desc, ...);
        }

        // 写入 Socket 配置到设备
        write_socket_configs(config_buffer_, local_endpoint_desc, remote_endpoint_desc, ...);

        // 同步屏障
        barrier_across_send_recv_ranks(sender_ranks, recv_ranks, context);
    }
};
```

#### 3.4.3 描述符传输

```cpp
// mesh_socket_utils.cpp
void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context,
    const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table) {

    // 序列化描述符
    auto serialized_desc = serialize_to_bytes(desc);

    // 确定目标 rank
    multihost::Rank global_peer_rank = (socket_endpoint_type == SocketEndpoint::SENDER)
        ? desc.config.receiver_rank
        : desc.config.sender_rank;

    // 转换到当前通信域的 rank
    auto it = rank_translation_table.find(global_peer_rank);
    multihost::Rank peer_rank = it->second;

    // 发送给对端
    context->send(
        tt::stl::Span<std::byte>(serialized_desc.data(), serialized_desc.size()),
        peer_rank,
        desc.exchange_tag);
}
```

### 3.5 实验性功能

#### 3.5.1 Blitz Decode Pipeline

```cpp
// experimental/blitz_decode_pipeline.cpp
namespace tt::tt_metal::experimental::blitz {

// 物理流水线阶段配置
struct PhysicalPipelineStageConfig {
    uint32_t entry_node_tray_id;
    uint32_t exit_node_tray_id;
    uint32_t entry_node_asic_location;
    uint32_t exit_node_asic_location;
};

// 创建 ASIC ID 到 Mesh 坐标的映射
std::unordered_map<AsicID, MeshCoordinate> get_asic_id_to_mesh_coord_map(
    const MeshDevice& mesh_device) {
    // 收集本地映射
    for (const auto& coord : MeshCoordinateRange(mesh_device.shape())) {
        auto fabric_node_id = mesh_device.get_fabric_node_id(coord);
        AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map[asic_id] = coord;
    }

    // 通过广播收集所有主机的映射
    auto& distributed_context = DistributedContext::get_current_world();
    for (auto rank = 0; rank < *(distributed_context->size()); rank++) {
        if (rank == *(distributed_context->rank())) {
            // 广播本地条目数
            distributed_context->broadcast(..., Rank{rank});
            // 广播每个条目
            for (auto& [asic_id, mesh_coord] : asic_id_to_mesh_coord_map) {
                distributed_context->broadcast(...);
            }
        } else {
            // 接收其他主机的条目
            distributed_context->broadcast(...);
        }
    }
}

// 预定义的流水线配置（4 主机和 16 主机）
std::vector<PhysicalPipelineStageConfig> generate_physical_pipeline_config() {
    auto num_procs = *(DistributedContext::get_current_world()->size());
    switch (num_procs) {
        case 4: return { /* 4 主机配置 */ };
        case 16: return { /* 16 主机配置 */ };
    }
}

} // namespace
```

## 4. 数据流向

### 4.1 程序执行数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host Application                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MeshWorkload Creation                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Program 1  │  │  Program 2  │  │  Program N  │             │
│  │ (subgrid A) │  │ (subgrid B) │  │ (subgrid C) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Compile Phase                              │
│  1. Compile Kernel Binaries (per program)                       │
│  2. Allocate Circular Buffers                                   │
│  3. Finalize Dataflow Buffer Configs                            │
│  4. Compute Relative Offsets                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Load Binaries                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         MeshBuffer (Replicated Kernel Binaries)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              FDMeshCommandQueue::enqueue_mesh_workload          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │   │
│  │  │ Dispatch│ │ Dispatch│ │ Dispatch│  (per subgrid)    │   │
│  │  │ Cmds A  │ │ Cmds B  │ │ Cmds C  │                   │   │
│  │  └─────────┘ └─────────┘ └─────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Physical Devices                             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │
│  │Dev 0│ │Dev 1│ │Dev 2│ │Dev 3│ │ ... │ │Dev N│             │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Socket 通信数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sender Host (Rank 0)                        │
│  ┌─────────────┐        ┌─────────────┐                        │
│  │ Config Buffer│◄──────│  MeshSocket │                        │
│  └─────────────┘        └─────────────┘                        │
│         │                         │                            │
│         │  serialize_to_bytes()   │                            │
│         ▼                         │                            │
│  ┌─────────────┐                  │  MPI_Send()                │
│  │ FlatBuffer  │──────────────────┘                            │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ MPI Network
┌─────────────────────────────────────────────────────────────────┐
│                    Receiver Host (Rank 1)                       │
│  ┌─────────────┐        ┌─────────────┐                        │
│  │ Config Buffer│◄──────│  MeshSocket │◄────── MPI_Recv()      │
│  └─────────────┘        └─────────────┘                        │
│         ▲                         │                            │
│         │  write_socket_configs() │                            │
│  ┌─────────────┐                  │                            │
│  │ Data Buffer │◄─────────────────┘                            │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 多主机同步流程

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Host 0  │     │ Host 1  │     │ Host 2  │     │ Host 3  │
│ (Rank 0)│     │ (Rank 1)│     │ (Rank 2)│     │ (Rank 3)│
└────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
     │               │               │               │
     │  MPI_Barrier  │  MPI_Barrier  │  MPI_Barrier  │
     │◄─────────────►│◄─────────────►│◄─────────────►│
     │               │               │               │
     │          Workload Execution    │               │
     │               │               │               │
     │  MPI_AllGather│  MPI_AllGather│  MPI_AllGather│
     │◄─────────────►│◄─────────────►│◄─────────────►│
     │               │               │               │
```

## 5. 设计模式与实现技巧

### 5.1 多态与类型擦除

```cpp
// 使用 variant 实现类型擦除
using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

// 访问者模式处理不同类型
DeviceAddr device_local_size = std::visit(
    tt::stl::overloaded{
        [](const ReplicatedBufferConfig& c) { return c.size; },
        [](const ShardedBufferConfig& config) {
            auto [shard_height, shard_width] = config.physical_shard_shape();
            return config.compute_datum_size_bytes() * shard_height * shard_width;
        }
    },
    mesh_buffer_config);
```

### 5.2 RAII 资源管理

```cpp
// ScopedDevices 自动管理物理设备生命周期
class MeshDeviceImpl::ScopedDevices {
public:
    // 构造函数获取资源
    ScopedDevices(size_t l1_small_size, size_t trace_region_size, ...);

    // 析构函数释放资源
    ~ScopedDevices();

    // 禁止拷贝
    ScopedDevices(const ScopedDevices&) = delete;
    ScopedDevices& operator=(const ScopedDevices&) = delete;
};
```

### 5.3 线程安全设计

```cpp
class MeshDeviceImpl {
    std::mutex api_mutex_;  // 保护状态修改操作

    std::lock_guard<std::mutex> lock_api() {
        return std::lock_guard<std::mutex>(api_mutex_);
    }

    // 无锁数据结构用于高频操作
    std::atomic<uint32_t> num_outstanding_reads_ = 0;
    std::atomic<bool> exit_condition_ = false;
};
```

### 5.4 编译时多态（策略模式）

```cpp
// 通过宏在编译时选择实现
#if defined(OPEN_MPI)
    using ContextImpl = MPIContext;
#else
    using ContextImpl = SingleHostContext;
#endif

// 统一的工厂接口
void DistributedContext::create(int argc, char** argv) {
    ContextImpl::create(argc, argv);
}
```

### 5.5 延迟初始化与缓存

```cpp
class MeshWorkloadImpl {
    // 延迟计算的内核组缓存
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t type_index) {
        if (kernel_groups_.at(type_index).empty()) {
            // 首次访问时计算
            compute_kernel_groups();
        }
        return kernel_groups_.at(type_index);
    }
};
```

### 5.6 异常安全与超时处理

```cpp
template <typename OperationType, typename... Args>
void execute_with_timeout(OperationType&& operation, Args&&... args) {
    const auto timeout = std::chrono::duration<float>(10.0f);
    std::atomic<bool> completed{false};
    std::exception_ptr exception_ptr{nullptr};

    std::thread thread([&]() {
        try {
            operation(std::forward<Args>(args)...);
            completed = true;
        } catch (...) {
            exception_ptr = std::current_exception();
        }
    });

    // 超时检测循环
    auto start = std::chrono::steady_clock::now();
    while (!completed) {
        auto elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout.count()) {
            thread.detach();
            TT_THROW("Operation timed out");
        }
        std::this_thread::yield();
    }

    if (thread.joinable()) thread.join();
    if (exception_ptr) std::rethrow_exception(exception_ptr);
}
```

## 6. 源码注释摘录

### 6.1 分布式上下文创建

```cpp
// mpi_distributed_context.cpp
inline void init_env(int& argc, char**& argv) {
    static std::once_flag mpi_once;

    std::call_once(mpi_once, [&] {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
            TT_THROW("MPI_Init_thread failed");
        }

        // Ensure MPI_Finalize is called when the program exits
        std::atexit([] { MPI_Finalize(); });
    });
}
```

### 6.2 MeshWorkload 编译流程

```cpp
// mesh_workload.cpp
void MeshWorkloadImpl::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    if (programs_.size() == 1) {
        // Compile from main thread for homogeneous workloads
        this->compile_program(programs_.begin()->first, mesh_device);
    } else {
        for (auto& [device_range, _] : programs_) {
            // Multi-Threaded Compile: Useful for heterogeneous MeshWorkloads
            mesh_device->enqueue_to_thread_pool(
                [device_range, mesh_device, this]() {
                    this->compile_program(device_range, mesh_device);
                });
        }
        mesh_device->wait_for_thread_pool();
    }
    finalize_offsets(mesh_device);
}
```

### 6.3 Socket 连接建立约定

```cpp
// mesh_socket.cpp
void MeshSocket::connect_with_peer(const std::shared_ptr<multihost::DistributedContext>& context) {
    // Convention:
    //  - Sender Endpoint sends its descriptor first, then receives the peer's descriptor.
    //  - Receiver Endpoint receives the peer's descriptor first, then sends its own descriptor.
    // Asymmetry ensures that the blocking send/recv do not deadlock.
    if (socket_endpoint_type_ == SocketEndpoint::SENDER) {
        forward_descriptor_to_peer(local_endpoint_desc, ...);
        remote_endpoint_desc = receive_and_verify_descriptor_from_peer(...);
    } else {
        remote_endpoint_desc = receive_and_verify_descriptor_from_peer(...);
        forward_descriptor_to_peer(local_endpoint_desc, ...);
    }
    // ...
}
```

### 6.4 追踪捕获三阶段

```cpp
// mesh_trace.hpp
// MeshTrace capture consists of 3 steps:
// 1. Staging: Workload dispatch commands are recorded inside a host data structure
// and the MeshTraceStagingMetadata holds information for where the trace data/commands
// have been stored. The commands are not ready to be committed to device DRAM in this
// form, hence they are temporarily staged and will be processed downstream.
// 2. Assembly: Create a MeshTrace from the staged commands by moving all dispatch
// commands out of the staging structure, and consolidate them into a single MeshTrace
// that can be written out to DRAM.
// 3. Commit to Mesh: Write assembled trace to DRAM buffer.
```

### 6.5 快速调度初始化检查

```cpp
// dispatch_context.cpp
void DispatchContext::initialize_fast_dispatch(MeshDevice* mesh_device) {
    TT_FATAL(
        !fast_dispatch_enabled_,
        "Fast Dispatch can only be manually enabled when running the workload with Slow Dispatch mode.");
    TT_FATAL(num_fd_inits_ == 0, "Fast Dispatch can only be manually initialized and torn down once.");
    TT_FATAL(
        cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
        "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.");
    // ...
}
```

### 6.6 设备所有权验证

```cpp
// mesh_socket.cpp
void validate_device_ownership(
    multihost::Rank global_sender_rank,
    multihost::Rank global_receiver_rank,
    const SocketConfig& config) {

    bool is_sender = global_distributed_context->rank() == global_sender_rank;
    bool is_receiver = global_distributed_context->rank() == global_receiver_rank;

    if (is_sender || is_receiver) {
        for (const auto& connection : config.socket_connection_config) {
            if (is_sender) {
                TT_FATAL(
                    sender_coord_range.contains(connection.sender_core.device_coord),
                    "Sender core coordinate {} is out of bounds for rank {} on mesh id {}",
                    connection.sender_core.device_coord,
                    *global_sender_rank,
                    *config.sender_mesh_id);
            }
            // ...
        }
    }
}
```

---

*文档生成时间：2026-03-13*
*源码版本：基于 TT-Metal 最新 main 分支*
