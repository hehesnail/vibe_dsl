# impl/ 模块源码解析

## 1. 模块概述

### 1.1 模块职责

`impl/` 模块是 TT-Metal 框架的核心实现层，位于 API 层 (`api/`) 之下、硬件抽象层 (`hw/`) 之上。它实现了以下关键功能：

- **内存管理**：DRAM/L1 内存分配器、Bank 管理、缓冲区分配
- **设备管理**：物理设备抽象、设备生命周期管理、子设备管理
- **调度系统**：命令队列、硬件命令分发、系统内存管理
- **程序管理**：内核编译、程序配置、运行时参数管理
- **内核管理**：数据移动内核、计算内核、以太网内核
- **事件追踪**：事件记录、追踪捕获、性能分析

### 1.2 在系统中的位置

```
┌─────────────────────────────────────────┐
│           TTNN / Applications           │
├─────────────────────────────────────────┤
│              api/ (API层)                │
│    (buffer.hpp, program.hpp, etc.)      │
├─────────────────────────────────────────┤
│  ➤ impl/ (本模块 - 核心实现层)           │
│    allocator/, buffers/, device/        │
│    dispatch/, program/, kernels/        │
├─────────────────────────────────────────┤
│              hw/ (硬件层)                │
│         HAL, UMD, Soc Descriptor        │
└─────────────────────────────────────────┘
```

### 1.3 与其他模块的交互

- **向上 (API层)**：通过 `IDevice`, `Buffer`, `Program` 等接口提供服务
- **向下 (硬件层)**：通过 HAL (Hardware Abstraction Layer) 访问硬件
- **跨模块**：与 `tt_cluster`, `llrt`, `jit_build` 等模块紧密协作

---

## 2. 目录结构

```
tt_metal/impl/
├── allocator/              # 内存分配器实现
│   ├── algorithms/         # 分配算法
│   │   ├── allocator_algorithm.hpp   # 分配器算法基类
│   │   ├── free_list_opt.hpp         # 优化空闲列表算法
│   │   └── free_list_opt.cpp
│   ├── allocator.hpp       # 分配器实现头文件
│   ├── allocator.cpp       # 分配器实现
│   ├── allocator_types.hpp # 分配器类型定义
│   ├── allocator_state.cpp # 分配器状态管理
│   ├── bank_manager.hpp    # Bank 管理器
│   ├── bank_manager.cpp
│   ├── l1_banking_allocator.hpp  # L1 Banking 分配器
│   └── l1_banking_allocator.cpp
│
├── buffers/                # 缓冲区管理
│   ├── buffer.cpp          # Buffer 实现
│   ├── buffer_distribution_spec.cpp  # 缓冲区分布规范
│   ├── buffer_page_mapping.cpp       # 页面映射
│   ├── circular_buffer.hpp # 循环缓冲区头文件
│   ├── circular_buffer.cpp # 循环缓冲区实现
│   ├── circular_buffer_config.cpp    # CB 配置
│   ├── dispatch.hpp        # 缓冲区调度
│   ├── dispatch.cpp
│   ├── global_circular_buffer.cpp    # 全局循环缓冲区
│   ├── global_semaphore.cpp          # 全局信号量
│   ├── semaphore.hpp       # 信号量头文件
│   └── semaphore.cpp       # 信号量实现
│
├── context/                # 上下文管理
│   ├── metal_context.hpp   # Metal 上下文
│   ├── metal_context.cpp   # Metal 上下文实现
│   ├── context_descriptor.hpp        # 上下文描述符
│   └── metal_env*.hpp      # Metal 环境
│
├── data_format/            # 数据格式处理
│   ├── bfloat16.cpp        # BFloat16 格式
│   ├── bfloat4.cpp         # BFloat4 格式
│   ├── bfloat8.cpp         # BFloat8 格式
│   ├── tile.cpp            # Tile 格式
│   └── tilize_utils.cpp    # Tilize 工具
│
├── dataflow_buffer/        # 数据流缓冲区
│   └── dataflow_buffer*.hpp/cpp
│
├── debug/                  # 调试工具
│   ├── dprint_server.hpp   # 打印服务器
│   ├── watcher_server.hpp  # Watcher 服务器
│   └── inspector/          # 检查器工具
│
├── device/                 # 设备管理
│   ├── device_impl.hpp     # 设备实现头文件
│   ├── device.cpp          # 设备实现
│   ├── device_manager.hpp  # 设备管理器
│   ├── device_manager.cpp  # 设备管理器实现
│   ├── dispatch.hpp        # 设备调度
│   ├── mock_device*.hpp/cpp # Mock 设备
│   ├── experimental/       # 实验性功能
│   └── firmware/           # 固件初始化
│       ├── firmware_initializer.hpp
│       ├── risc_firmware_initializer.hpp
│       └── dispatch_kernel_initializer.hpp
│
├── dispatch/               # 调度系统
│   ├── command_queue_common.hpp/cpp
│   ├── device_command.hpp  # 设备命令
│   ├── device_command.cpp  # 设备命令实现
│   ├── device_command_calculator.hpp
│   ├── dispatch_core_manager.hpp/cpp
│   ├── dispatch_mem_map.hpp/cpp
│   ├── dispatch_query_manager.hpp
│   ├── dispatch_settings.hpp
│   ├── hardware_command_queue.hpp  # 硬件命令队列
│   ├── system_memory_manager.hpp   # 系统内存管理器
│   ├── system_memory_manager.cpp
│   ├── topology.hpp        # 调度拓扑
│   ├── topology.cpp        # 拓扑实现
│   ├── worker_config_buffer.hpp
│   ├── ringbuffer_cache.hpp/cpp
│   └── kernel_config/      # 内核配置
│
├── event/                  # 事件管理
│   ├── dispatch.hpp        # 事件调度
│   ├── dispatch.cpp
│   └── event.cpp           # 事件实现
│
├── experimental/           # 实验性功能
│
├── flatbuffer/             # FlatBuffer 支持
│
├── kernels/                # 内核管理
│   ├── kernel.hpp          # 内核头文件
│   ├── kernel.cpp          # 内核实现
│   └── kernel_types.cpp    # 内核类型
│
├── lightmetal/             # LightMetal 支持
│
├── profiler/               # 性能分析器
│
├── program/                # 程序管理
│   ├── dispatch.hpp        # 程序调度接口
│   ├── dispatch.cpp        # 程序调度实现 (159KB)
│   ├── program_impl.hpp    # 程序实现头文件
│   ├── program.cpp         # 程序实现 (93KB)
│   ├── program_command_sequence.hpp
│   ├── program_descriptors.cpp
│   ├── program_device_map.hpp/cpp
│   └── dispatch.hpp        # 调度接口
│
├── sub_device/             # 子设备管理
│   └── sub_device_manager.hpp
│
├── tensor/                 # 张量管理
│   ├── tensor_types.cpp    # 张量类型
│   ├── spec/               # 张量规范
│   └── topology/           # 张量拓扑
│
└── trace/                  # 追踪系统
    ├── dispatch.hpp
    ├── dispatch.cpp
    ├── trace_buffer.hpp    # 追踪缓冲区
    └── trace_node.hpp      # 追踪节点
```

---

## 3. 核心组件解析

### 3.1 allocator/ 内存分配器

#### 3.1.1 架构概述

分配器模块采用分层设计：

```
┌─────────────────────────────────────────┐
│         AllocatorImpl (API)             │
├─────────────────────────────────────────┤
│  DRAM Manager │ L1 Manager │ L1_SMALL   │
│  (BankManager)│(BankManager)│(BankManager)
├─────────────────────────────────────────┤
│      FreeListOpt (分配算法)              │
│   (Size-Segregated + Hash Table)        │
└─────────────────────────────────────────┘
```

#### 3.1.2 Algorithm 基类

**文件**: `/tmp/tt-metal/tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp`

```cpp
// 行 18-86: 分配器算法基类
class Algorithm {
public:
    Algorithm(
        DeviceAddr max_size_bytes,
        DeviceAddr offset_bytes,
        DeviceAddr min_allocation_size,
        DeviceAddr alignment);

    virtual ~Algorithm() = default;

    // 核心分配接口
    virtual std::optional<DeviceAddr> allocate(
        DeviceAddr size_bytes,
        bool bottom_up = true,
        DeviceAddr address_limit = 0) = 0;

    virtual std::optional<DeviceAddr> allocate_at_address(
        DeviceAddr absolute_start_address,
        DeviceAddr size_bytes) = 0;

    virtual void deallocate(DeviceAddr absolute_address) = 0;
    virtual void clear() = 0;

    // 查询接口
    virtual std::vector<std::pair<DeviceAddr, DeviceAddr>> available_addresses(
        DeviceAddr size_bytes) const = 0;
    virtual std::vector<std::pair<DeviceAddr, DeviceAddr>> allocated_addresses() const = 0;
    virtual std::optional<DeviceAddr> get_allocation_size(
        DeviceAddr absolute_address) const = 0;
    virtual Statistics get_statistics() const = 0;
    virtual MemoryBlockTable get_memory_block_table() const = 0;

protected:
    DeviceAddr max_size_bytes_;
    DeviceAddr offset_bytes_;
    DeviceAddr min_allocation_size_;
    DeviceAddr alignment_;
    DeviceAddr shrink_size_ = 0;
    std::optional<DeviceAddr> lowest_occupied_address_;
};
```

#### 3.1.3 FreeListOpt - 优化的空闲列表算法

**文件**: `/tmp/tt-metal/tt_metal/impl/allocator/algorithms/free_list_opt.hpp`

FreeListOpt 是一个高性能的内存分配算法，具有以下优化特性：

**核心优化技术** (行 19-26):
```cpp
// - SoA (Structure of Arrays) 代替链表，提高缓存局部性
// - 大小分级 (Size Segregated) 避免不必要的小块搜索
// - 哈希表存储已分配块，加速释放时的查找
// - 元数据复用，避免重复分配
// - 缓存优化，减少缓存未命中
```

**数据结构** (行 76-102):
```cpp
// SoA 空闲列表组件
std::vector<DeviceAddr> block_address_;      // 块地址
std::vector<DeviceAddr> block_size_;         // 块大小
std::vector<ssize_t> block_prev_block_;      // 前一个块索引
std::vector<ssize_t> block_next_block_;      // 后一个块索引
std::vector<uint8_t> block_is_allocated_;    // 分配状态
std::vector<uint8_t> meta_block_is_allocated_; // 元数据状态

// 空闲元数据块索引（复用）
std::vector<size_t> free_meta_block_indices_;

// 已分配块哈希表（512 桶，每桶初始 10 个槽位）
static constexpr size_t n_alloc_table_buckets = 512;
static constexpr size_t n_alloc_table_init_bucket_size = 10;
std::vector<std::vector<std::pair<DeviceAddr, size_t>>> allocated_block_table_;

// 大小分级的空闲块列表（基于 TLSF 算法思想）
static constexpr size_t size_segregated_base = 1024;  // 基础大小 1KB
const size_t size_segregated_count;
std::vector<std::vector<size_t>> free_blocks_segregated_by_size_;
```

**分配算法** (行 99-176):
```cpp
std::optional<DeviceAddr> FreeListOpt::allocate(
    DeviceAddr size_bytes,
    bool bottom_up,
    DeviceAddr address_limit) {

    DeviceAddr alloc_size = align(std::max(size_bytes, min_allocation_size_));

    // 1. 计算大小分级索引
    ssize_t target_block_index = -1;
    size_t size_segregated_index = get_size_segregated_index(alloc_size);

    // 2. 在大小分级列表中搜索最佳块
    for (size_t i = size_segregated_index; i < free_blocks_segregated_by_size_.size(); i++) {
        auto& free_blocks = free_blocks_segregated_by_size_[i];
        // 根据 bottom_up 选择搜索方向
        ssize_t increment = bottom_up ? 1 : -1;
        for (ssize_t j = bottom_up ? 0 : free_blocks.size() - 1;
             j >= 0 && j < free_blocks.size();
             j += increment) {

            size_t block_index = free_blocks[j];
            if (policy_ == SearchPolicy::BEST) {
                // 最佳适配：找刚好能容纳的块
                if (block_size_[block_index] == alloc_size) {
                    target_block_index = block_index;
                    break;
                }
                if (block_size_[block_index] >= alloc_size &&
                    (target_block_index == -1 ||
                     block_size_[block_index] < block_size_[target_block_index])) {
                    target_block_index = block_index;
                }
            }
        }
    }

    if (target_block_index == -1) {
        return std::nullopt;  // 分配失败
    }

    // 3. 从分级列表中移除
    segregated_list->erase(segregated_list->begin() + segregated_item_index);

    // 4. 在块内分配
    size_t offset = bottom_up ? 0 : block_size_[target_block_index] - alloc_size;
    size_t allocated_block_index = allocate_in_block(target_block_index, alloc_size, offset);

    DeviceAddr start_address = block_address_[allocated_block_index];
    update_lowest_occupied_address(start_address);
    return start_address + offset_bytes_;
}
```

**块内分配** (行 212-256):
```cpp
size_t FreeListOpt::allocate_in_block(
    size_t block_index,
    DeviceAddr alloc_size,
    size_t offset) {

    // 完全匹配，直接标记为已分配
    if (block_size_[block_index] == alloc_size && offset == 0) {
        block_is_allocated_[block_index] = true;
        insert_block_to_alloc_table(block_address_[block_index], block_index);
        return block_index;
    }

    bool left_aligned = offset == 0;
    bool right_aligned = offset + alloc_size == block_size_[block_index];

    // 创建左侧空闲空间（如果不是左对齐）
    if (!left_aligned) {
        size_t free_block_size = offset;
        DeviceAddr free_block_address = block_address_[block_index];
        ssize_t prev_block = block_prev_block_[block_index];

        // 调整原块
        block_size_[block_index] -= offset;
        block_address_[block_index] += offset;

        // 创建新的空闲块
        size_t new_block_index = alloc_meta_block(
            free_block_address, free_block_size, prev_block, block_index, false);

        // 更新链表
        if (prev_block != -1) {
            block_next_block_[prev_block] = new_block_index;
        }
        block_prev_block_[block_index] = new_block_index;
        insert_block_to_segregated_list(new_block_index);
    }

    // 创建右侧空闲空间（如果不是右对齐）
    if (!right_aligned) {
        size_t free_block_size = block_size_[block_index] - alloc_size;
        DeviceAddr free_block_address = block_address_[block_index] + alloc_size;
        ssize_t prev_block = block_index;
        ssize_t next_block = block_next_block_[block_index];

        block_size_[block_index] -= free_block_size;

        size_t new_block_index = alloc_meta_block(
            free_block_address, free_block_size, prev_block, next_block, false);

        if (next_block != -1) {
            block_prev_block_[next_block] = new_block_index;
        }
        block_next_block_[block_index] = new_block_index;
        insert_block_to_segregated_list(new_block_index);
    }

    block_is_allocated_[block_index] = true;
    insert_block_to_alloc_table(block_address_[block_index], block_index);
    return block_index;
}
```

**释放与合并** (行 258-329):
```cpp
void FreeListOpt::deallocate(DeviceAddr absolute_address) {
    DeviceAddr addr = absolute_address - offset_bytes_;

    // 1. 从哈希表查找块索引
    auto block_index_opt = get_and_remove_from_alloc_table(addr);
    if (!block_index_opt.has_value()) {
        return;  // 地址未分配
    }
    size_t block_index = *block_index_opt;
    block_is_allocated_[block_index] = false;

    ssize_t prev_block = block_prev_block_[block_index];
    ssize_t next_block = block_next_block_[block_index];

    // 2. 与前一个空闲块合并
    if (prev_block != -1 && !block_is_allocated_[prev_block]) {
        // 从分级列表移除前一个块
        size_t size_segregated_index = get_size_segregated_index(block_size_[prev_block]);
        std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
        auto it = std::find(segregated_list.begin(), segregated_list.end(), prev_block);
        segregated_list.erase(it);

        // 合并
        block_size_[prev_block] += block_size_[block_index];
        block_next_block_[prev_block] = next_block;
        if (next_block != -1) {
            block_prev_block_[next_block] = prev_block;
        }
        free_meta_block(block_index);
        block_index = prev_block;
    }

    // 3. 与后一个空闲块合并（类似逻辑）
    if (next_block != -1 && !block_is_allocated_[next_block]) {
        // ... 类似前一个块的合并逻辑
    }

    // 4. 更新最低占用地址
    if (addr <= *lowest_occupied_address_) {
        lowest_occupied_address_ = std::nullopt;
        // 重新扫描找最低占用地址
        ssize_t curr_block_index = block_next_block_[block_index];
        while (curr_block_index != -1) {
            if (block_is_allocated_[curr_block_index]) {
                lowest_occupied_address_ = block_address_[curr_block_index];
                break;
            }
            curr_block_index = block_next_block_[curr_block_index];
        }
    }

    // 5. 加入分级列表
    insert_block_to_segregated_list(block_index);
}
```

#### 3.1.4 BankManager - Bank 管理器

**文件**: `/tmp/tt-metal/tt_metal/impl/allocator/bank_manager.hpp`

BankManager 管理多个 Bank 的内存分配，支持分配器依赖关系。

**AllocatorDependencies 结构** (行 50-67):
```cpp
// 描述多个分配器之间的依赖关系
// 使用无向邻接表存储，支持双向依赖
struct AllocatorDependencies {
    using AllocatorID = ttsl::StrongType<uint32_t, struct AllocatorIDTag>;
    using AdjacencyList = ttsl::SmallVector<ttsl::SmallVector<AllocatorID>>;

    // 每个状态在邻接表中的位置对应于该状态的分配器ID
    // 依赖按分配器ID排序
    AdjacencyList dependencies{{}};  // 默认：单个独立分配器

    AllocatorDependencies();
    explicit AllocatorDependencies(
        const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>>& dependencies_map);

    uint32_t num_allocators() const;
    ttsl::SmallVector<AllocatorID> allocator_ids() const;
};
```

**核心接口** (行 69-151):
```cpp
class BankManager {
public:
    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_offsets,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const AllocatorDependencies& dependencies = AllocatorDependencies());

    uint32_t num_banks() const;
    DeviceAddr bank_size() const;
    int64_t bank_offset(uint32_t bank_id) const;

    // 分配/释放缓冲区
    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});

    void deallocate_buffer(
        DeviceAddr address,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});
    void deallocate_all();

    // 状态管理
    void clear();
    std::optional<DeviceAddr> lowest_occupied_address(uint32_t bank_id, ...) const;
    Statistics get_statistics(...) const;
    MemoryBlockTable get_memory_block_table(...) const;

    // 分配器状态提取/应用（用于追踪捕获）
    AllocatorState::BufferTypeState extract_state(...) const;
    AllocatorState::BufferTypeState extract_merged_state() const;
    void apply_state(const AllocatorState::BufferTypeState& state, ...);
    void override_state(const AllocatorState::BufferTypeState& state, ...);

    // 高水位追踪
    void begin_high_water_mark_tracking();
    DeviceAddr end_high_water_mark_tracking();
    DeviceAddr get_high_water_mark() const;
};
```

**计算可用地址** (行 276-382):
```cpp
// 计算考虑依赖分配器占用后的可用地址范围
std::vector<std::pair<DeviceAddr, DeviceAddr>> BankManager::compute_available_addresses(
    AllocatorDependencies::AllocatorID allocator_id,
    DeviceAddr size_per_bank,
    DeviceAddr address_limit) {

    auto* alloc = this->get_allocator_from_id(allocator_id);

    // 1. 获取当前分配器的可用范围
    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_ranges =
        alloc->available_addresses(size_per_bank);

    // 2. 根据 address_limit 裁剪
    if (address_limit != 0) {
        for (auto& r : ranges) {
            r.first = std::max(r.first, address_limit);
        }
        // 移除空范围
        ranges.erase(
            std::remove_if(ranges.begin(), ranges.end(),
                [](const auto& r) { return r.second <= r.first; }),
            ranges.end());
    }
    std::sort(available_ranges.begin(), available_ranges.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // 3. 获取依赖分配器的已分配范围（合并后）
    const auto& allocated_ranges_in_dependent_allocators =
        this->compute_merged_allocated_ranges(allocator_id);

    // 4. 从可用范围中减去已分配范围（双指针扫描）
    auto subtract_ranges = [](const auto& available_ranges, const auto& allocated_ranges) {
        std::vector<std::pair<DeviceAddr, DeviceAddr>> out;
        out.reserve(available_ranges.size());

        size_t j = 0;  // 单调指针扫描所有已分配范围
        for (const auto& available : available_ranges) {
            DeviceAddr available_start = available.first;
            DeviceAddr available_end = available.second;

            // 跳过完全在可用范围左侧的已分配范围
            while (j < allocated_ranges.size() && allocated_ranges[j].second <= available_start) {
                j++;
            }

            DeviceAddr cur_available_start = available_start;
            // 扫描可能与当前可用范围重叠的已分配范围
            while (j < allocated_ranges.size() && allocated_ranges[j].first < available_end) {
                DeviceAddr allocated_start = allocated_ranges[j].first;
                DeviceAddr allocated_end = allocated_ranges[j].second;

                // 情况 A: 已分配范围前有空隙
                if (allocated_start > cur_available_start) {
                    out.emplace_back(cur_available_start, allocated_start);
                }

                // 情况 B: 已分配范围覆盖到可用范围末尾
                if (allocated_end >= available_end) {
                    cur_available_start = available_end;
                    break;
                }

                // 情况 C: 已分配范围在可用范围中间
                cur_available_start = allocated_end;
                j++;
            }

            // 添加剩余尾部
            if (cur_available_start < available_end) {
                out.emplace_back(cur_available_start, available_end);
            }
        }
        return out;
    };

    return subtract_ranges(available_ranges, allocated_ranges_in_dependent_allocators);
}
```

#### 3.1.5 AllocatorImpl - 分配器实现

**文件**: `/tmp/tt-metal/tt_metal/impl/allocator/allocator.hpp`

```cpp
// 行 16-135: 分配器实现类
class AllocatorImpl {
public:
    explicit AllocatorImpl(const AllocatorConfig& alloc_config);
    ~AllocatorImpl();

    // 缓冲区分配/释放
    DeviceAddr allocate_buffer(Buffer* buffer);
    void deallocate_buffer(Buffer* buffer);
    void deallocate_buffers();

    // 查询接口
    std::unordered_set<Buffer*> get_allocated_buffers() const;
    size_t get_num_allocated_buffers() const;
    std::uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;

    // Bank 映射查询
    std::uint32_t get_dram_channel_from_bank_id(std::uint32_t bank_id) const;
    CoreCoord get_logical_core_from_bank_id(std::uint32_t bank_id) const;
    int32_t get_bank_offset(BufferType buffer_type, std::uint32_t bank_id) const;
    const std::vector<std::uint32_t>& get_bank_ids_from_dram_channel(std::uint32_t dram_channel) const;
    const std::vector<std::uint32_t>& get_bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;

    // 配置和统计
    DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const;
    const AllocatorConfig& get_config() const;
    std::uint32_t get_alignment(BufferType buffer_type) const;
    Statistics get_statistics(const BufferType& buffer_type) const;
    MemoryBlockTable get_memory_block_table(const BufferType& buffer_type) const;

    // 高水位追踪（用于追踪捕获）
    void begin_dram_high_water_mark_tracking();
    DeviceAddr end_dram_high_water_mark_tracking();
    DeviceAddr get_dram_high_water_mark() const;

    // 状态管理
    AllocatorState extract_state() const;
    void override_state(const AllocatorState& state);
    void clear();

    // 安全标记（追踪活动时分配不安全）
    void mark_allocations_unsafe();
    void mark_allocations_safe();

private:
    mutable std::mutex mutex_;
    bool allocations_unsafe_ = false;  // 追踪活动时设为 true

    std::unique_ptr<BankManager> dram_manager_;
    std::unique_ptr<BankManager> l1_manager_;
    std::unique_ptr<BankManager> l1_small_manager_;
    std::unique_ptr<BankManager> trace_buffer_manager_;

    // Bank 映射表
    std::unordered_map<std::uint32_t, std::uint32_t> bank_id_to_dram_channel_;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> dram_channel_to_bank_ids_;
    std::unordered_map<std::uint32_t, CoreCoord> bank_id_to_logical_core_;
    std::unordered_map<BufferType, std::unordered_map<CoreCoord, std::vector<std::uint32_t>>>
        logical_core_to_bank_ids_;
    std::unordered_set<Buffer*> allocated_buffers_;

    std::unique_ptr<AllocatorConfig> config_;
    std::unique_ptr<Allocator> view_;  // 外部 API 视图
};
```

---

### 3.2 buffers/ 缓冲区管理

#### 3.2.1 Buffer 类

**文件**: `/tmp/tt-metal/tt_metal/impl/buffers/buffer.cpp`

Buffer 是设备内存的抽象，支持多种内存布局和分片策略。

**Buffer 创建** (行 274-339):
```cpp
Buffer::Buffer(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id,
    const bool owns_data,
    Private) :
    device_(device),
    size_(size),
    buffer_type_(buffer_type),
    buffer_layout_(sharding_args.buffer_layout()),
    bottom_up_(bottom_up.value_or(this->is_dram())),
    sub_device_id_(sub_device_id),
    owns_data_(owns_data),
    page_size_(page_size),
    shard_spec_(sharding_args.shard_spec()),
    buffer_distribution_spec_(sharding_args.buffer_distribution_spec()) {

    TT_FATAL(this->device_ != nullptr, "Device needs to not be null.");

    // 子设备验证
    if (this->sub_device_id_.has_value()) {
        validate_sub_device_id(this->sub_device_id_, this->device_, buffer_type, shard_spec_);
        this->sub_device_manager_id_ = this->device_->get_active_sub_device_manager_id();
        this->allocator_ = device->allocator_impl(*this->sub_device_id_).get();
    } else {
        this->allocator_ = device->allocator_impl().get();
    }

    validate_buffer_parameters(size, page_size, buffer_type, buffer_layout_, shard_spec_,
                               buffer_distribution_spec_);
    unique_id_ = next_unique_id.fetch_add(1);
}

// 工厂方法：创建并分配 Buffer
std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const BufferShardingArgs& sharding_args,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {

    auto buffer = std::make_shared<Buffer>(
        device, size, page_size, buffer_type, sharding_args, bottom_up, sub_device_id,
        true /* owns data */, Private());

    if (buffer->size_ == 0) {
        buffer->allocation_status_ = AllocationStatus::ALLOCATED;
        return buffer;
    }

    buffer->allocate_impl();  // 实际分配内存
    return buffer;
}
```

**页面映射生成** (行 220-272):
```cpp
UncompressedBufferPageMapping generate_buffer_page_mapping(const Buffer& buffer) {
    UncompressedBufferPageMapping buffer_page_mapping;

    if (buffer.size() == 0) {
        return buffer_page_mapping;
    }

    // 使用 BufferDistributionSpec 计算页面映射
    if (buffer.buffer_distribution_spec().has_value()) {
        return buffer.buffer_distribution_spec()->compute_page_mapping();
    }

    uint32_t num_cores = buffer.num_cores().value();
    auto shard_spec = buffer.shard_spec();
    bool row_major = shard_spec.orientation() == ShardOrientation::ROW_MAJOR;

    // 获取所有核心
    buffer_page_mapping.all_cores = corerange_to_cores(shard_spec.grid(), num_cores, row_major);

    // 计算核心到主机页面的映射
    auto [core_host_page_indices, shard_shape] = core_to_host_pages(
        num_dev_pages,
        shard_spec.num_pages(),
        num_cores,
        buffer.buffer_layout(),
        shard_spec.page_shape,
        shard_spec.shape(),
        shard_spec.tensor2d_shape_in_pages);

    // 填充页面映射
    auto shape_in_pages = shard_spec.shape_in_pages();
    for (uint32_t core_index = 0; core_index < core_host_page_indices.size(); core_index++) {
        uint32_t valid_shard_page = 0;
        buffer_page_mapping.core_host_page_indices[core_index].resize(
            shard_spec.num_pages(), UncompressedBufferPageMapping::PADDING);

        for (uint32_t shard_page_x = 0; shard_page_x < shape_in_pages[0]; shard_page_x++) {
            for (uint32_t shard_page_y = 0; shard_page_y < shape_in_pages[1]; shard_page_y++) {
                if (shard_page_x < shard_shape[core_index][0] &&
                    shard_page_y < shard_shape[core_index][1]) {
                    uint32_t host_page = core_host_page_indices[core_index][valid_shard_page];
                    size_t core_page_idx = (shard_page_x * shape_in_pages[1]) + shard_page_y;
                    buffer_page_mapping.core_host_page_indices[core_index][core_page_idx] = host_page;
                    valid_shard_page++;
                }
            }
        }
    }

    return buffer_page_mapping;
}
```

#### 3.2.2 CircularBuffer - 循环缓冲区

**文件**: `/tmp/tt-metal/tt_metal/impl/buffers/circular_buffer.hpp`

CircularBuffer (CB) 是 Tensix 核心上的高速 L1 内存缓冲区，用于内核间数据传递。

```cpp
// 行 13-78: 循环缓冲区实现
class CircularBufferImpl {
public:
    CircularBufferImpl(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    CircularBufferImpl(
        const CoreRangeSet& core_ranges,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);

    CBHandle id() const { return id_; }
    const CoreRangeSet& core_ranges() const { return core_ranges_; }
    const CircularBufferConfig& config() const { return config_; }
    CircularBufferConfig& config() { return config_; }

    // Buffer 索引管理
    const std::unordered_set<uint8_t>& buffer_indices() const { return config_.buffer_indices(); }
    const std::unordered_set<uint8_t>& local_buffer_indices() const { return config_.local_buffer_indices(); }
    const std::unordered_set<uint8_t>& remote_buffer_indices() const { return config_.remote_buffer_indices(); }

    // 属性查询
    uint32_t page_size(uint32_t buffer_index) const;
    bool globally_allocated() const { return this->config_.globally_allocated_address().has_value(); }
    bool is_global_circular_buffer() const { return this->shadow_global_circular_buffer_ != nullptr; }
    uint32_t size() const { return this->config_.total_size(); }
    uint32_t num_pages(uint32_t buffer_index) const;
    DataFormat data_format(uint32_t buffer_index) const;
    const std::optional<Tile>& tile(uint32_t buffer_index) const;

    // 地址管理
    uint32_t address() const;
    void assign_global_address();
    void set_locally_allocated_address(uint32_t address) { this->locally_allocated_address_ = address; }
    DeviceAddr config_address() const;

    // 核心范围检查
    bool is_on_logical_corerange(const CoreRange& logical_cr) const;
    bool is_on_logical_core(const CoreCoord& logical_core) const;

private:
    const uintptr_t id_;
    const CoreRangeSet core_ranges_;
    CircularBufferConfig config_;

    // 地址分配方式
    std::optional<uint32_t> locally_allocated_address_;  // 本地分配
    uint32_t globally_allocated_address_{};              // 全局分配
    DeviceAddr global_circular_buffer_config_address_{};
    const experimental::GlobalCircularBuffer* shadow_global_circular_buffer_ = nullptr;
};
```

#### 3.2.3 Semaphore - 信号量

**文件**: `/tmp/tt-metal/tt_metal/impl/buffers/semaphore.hpp`

```cpp
// 行 17-48: 信号量实现
class Semaphore {
public:
    Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value);
    Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type);

    uint32_t id() const { return id_; }
    uint32_t offset() const;  // 返回 id_ * sizeof(uint32_t)
    CoreRangeSet core_range_set() const { return core_range_set_; }
    CoreType core_type() const { return core_type_; }
    uint32_t initial_value() const { return initial_value_; }
    bool initialized_on_logical_core(const CoreCoord& logical_core) const;

private:
    CoreRangeSet core_range_set_;  // 信号量初始化的核心范围
    uint32_t id_;                  // 信号量 ID (0-15)
    uint32_t initial_value_;       // 初始值
    CoreType core_type_;           // 核心类型
};

constexpr std::uint32_t NUM_SEMAPHORES = 16;  // 每核心最大信号量数
```

---

### 3.3 device/ 设备管理

#### 3.3.1 Device 类

**文件**: `/tmp/tt-metal/tt_metal/impl/device/device_impl.hpp`

Device 类是物理 Tenstorrent 设备的抽象，管理设备生命周期、内存、命令队列等。

```cpp
// 行 31-251: 设备实现类
class Device : public IDevice {
public:
    Device(
        ChipId device_id,
        uint8_t num_hw_cqs,
        std::size_t l1_small_size,
        std::size_t trace_region_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false,
        uint32_t worker_thread_core = 0,
        uint32_t completion_queue_reader_core = 0,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
    ~Device() override;

    // 设备信息
    tt::ARCH arch() const override;
    ChipId id() const override { return id_; }
    ChipId build_id() const override { return id_; }
    uint8_t num_hw_cqs() const override { return num_hw_cqs_; }
    bool is_initialized() const override { return this->initialized_; }

    // 内存查询
    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;
    uint32_t get_clock_rate_mhz() const override;

    // 核心坐标转换
    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;
    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;

    // 以太网核心
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const override;
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const override;
    std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const override;

    // 分配器访问
    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;

    // 命令队列
    SystemMemoryManager& sysmem_manager() override;
    HWCommandQueue& command_queue(std::optional<uint8_t> cq_id = std::nullopt);

    // 初始化/关闭
    bool initialize(...) override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;
    bool close() override;

    // 程序缓存
    void enable_program_cache() override;
    void clear_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override { return program_cache_; }

    // 追踪
    uint32_t get_trace_buffers_size() const override { return trace_buffers_size_; }
    void set_trace_buffers_size(uint32_t size) override { trace_buffers_size_ = size; }

private:
    ChipId id_;
    std::vector<std::vector<ChipId>> tunnels_from_mmio_;
    bool initialized_ = false;

    // 命令队列
    std::vector<std::unique_ptr<Program>> command_queue_programs_;
    bool using_fast_dispatch_ = false;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    uint8_t num_hw_cqs_ = 1;
    std::vector<std::unique_ptr<HWCommandQueue>> command_queues_;

    // 核心集合
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;
    std::vector<CoreCoord> optimal_dram_bank_to_logical_worker_assignment_;

    // Bank 映射
    std::vector<int32_t> dram_bank_offset_map_;
    std::vector<int32_t> l1_bank_offset_map_;
    std::vector<uint16_t> dram_bank_to_noc_xy_;
    std::vector<uint16_t> l1_bank_to_noc_xy_;

    // 缓存和追踪
    program_cache::detail::ProgramCache program_cache_;
    uint32_t trace_buffers_size_ = 0;

    // 分配器
    std::unique_ptr<AllocatorImpl> default_allocator_;

    friend class experimental::DispatchContext;
};
```

#### 3.3.2 DeviceManager - 设备管理器

**文件**: `/tmp/tt-metal/tt_metal/impl/device/device_manager.hpp`

DeviceManager 是单例模式，管理所有活动设备的生命周期。

```cpp
// 行 28-97: 设备管理器
class DeviceManager {
public:
    ~DeviceManager();
    DeviceManager();

    bool is_initialized() const { return is_initialized_; }

    // 初始化
    void initialize(
        const std::vector<ChipId>& device_ids,
        bool init_profiler,
        bool initialize_fabric_and_dispatch_fw,
        const std::shared_ptr<ContextDescriptor>& descriptor);

    // 设备访问
    IDevice* get_active_device(ChipId device_id) const;
    std::vector<IDevice*> get_all_active_devices() const;
    bool close_device(ChipId device_id);
    std::vector<ChipId> get_all_active_device_ids() const;
    bool is_device_active(ChipId id) const;

    // 固件管理
    bool is_dispatch_firmware_active() const;
    void initialize_fabric_and_dispatch_fw();
    void initialize_dispatch_firmware(bool force_recreate_topology);
    void reset_dispatch_topology();

    // 调度核心查询
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) const;

private:
    uint8_t num_hw_cqs_{};
    size_t l1_small_size_{};
    size_t trace_region_size_{};
    size_t worker_l1_size_{};
    std::vector<uint32_t> l1_bank_remap_;
    bool using_fast_dispatch_ = false;
    bool is_initialized_ = false;

    mutable std::mutex lock_;
    std::vector<std::unique_ptr<Device>> devices_;

    std::shared_ptr<ContextDescriptor> descriptor_;
    std::map<InitializerKey, std::unique_ptr<FirmwareInitializer>> initializers_;
    std::unordered_set<InitializerKey> init_done_;

    void open_devices(const std::vector<ChipId>& device_ids);
    void activate_device(ChipId id);
    Device* get_active_device_internal(ChipId device_id) const;
};
```

---

### 3.4 dispatch/ 调度系统

#### 3.4.1 架构概述

调度系统是 TT-Metal 的核心，负责将主机命令转换为设备可执行的指令序列。

```
┌─────────────────────────────────────────────────────────────┐
│                      Host (CPU)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Program    │→ │   Dispatch   │→ │  DeviceCommand   │  │
│  │   (API)      │  │   (Impl)     │  │  (Binary Format) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Command Queue (CQ)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Prefetch   │→ │   Dispatch   │→ │   Worker Cores   │  │
│  │   (Read Cmd) │  │  (Execute)   │  │  (Tensix/Eth)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 3.4.2 DeviceCommand - 设备命令

**文件**: `/tmp/tt-metal/tt_metal/impl/dispatch/device_command.hpp`

DeviceCommand 封装了发送到设备的二进制命令。

```cpp
// 行 28-325: 设备命令模板类
template <bool hugepage_write = false>
class DeviceCommand {
public:
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;
    static constexpr uint32_t LOG2_PROGRAM_PAGE_SIZE = std::bit_width(PROGRAM_PAGE_SIZE) - 1;

    // 等待命令
    void add_dispatch_wait(
        uint32_t flags, uint32_t address, uint32_t stream, uint32_t count, uint8_t dispatcher_type = 0);

    // Prefetch 命令
    void add_prefetch_relay_linear(uint32_t noc_xy_addr, DeviceAddr lengthB, DeviceAddr addr);
    void add_prefetch_relay_paged(
        uint8_t is_dram, uint8_t start_page, uint32_t base_addr,
        uint32_t page_size, uint32_t pages, uint16_t length_adjust = 0);
    void add_prefetch_relay_paged_packed(
        uint32_t length, const std::vector<CQPrefetchRelayPagedPackedSubCmd>& sub_cmds, ...);

    // Dispatch 写命令
    template <bool flush_prefetch = true, bool inline_data = false>
    void add_dispatch_write_linear(
        uint8_t num_mcast_dests, uint32_t noc_xy_addr, DeviceAddr addr,
        DeviceAddr data_sizeB, const void* data = nullptr, uint32_t write_offset_index = 0);

    template <bool inline_data = false>
    void add_dispatch_write_paged(
        bool flush_prefetch, uint8_t is_dram, uint16_t start_page,
        uint32_t base_addr, uint32_t page_size, uint32_t pages, const void* data = nullptr);

    // Go 信号多播
    void add_dispatch_go_signal_mcast(
        uint32_t wait_count, uint32_t go_signal, uint32_t wait_stream,
        uint8_t multicast_go_offset, uint8_t num_unicast_txns,
        uint8_t noc_data_start_index, DispatcherSelect dispatcher_type);

    // Packed 写命令
    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint8_t type, uint16_t num_sub_cmds, uint32_t common_addr,
        uint16_t packed_data_sizeB, uint32_t payload_sizeB,
        const std::vector<PackedSubCmd>& sub_cmds,
        const std::vector<std::pair<const void*, uint32_t>>& data_collection, ...);

    // 终止命令
    void add_dispatch_terminate(DispatcherSelect dispatcher_type = DispatcherSelect::DISPATCH_MASTER);
    void add_prefetch_terminate();

    // 命令序列更新
    void update_cmd_sequence(uint32_t cmd_offsetB, const void* new_data, uint32_t data_sizeB);

private:
    uint32_t cmd_sequence_sizeB = 0;
    void* cmd_region = nullptr;
    uint32_t cmd_write_offsetB = 0;
    uint32_t pcie_alignment;
    uint32_t l1_alignment;
    vector_aligned<uint32_t> cmd_region_vector;
};

// 类型别名
using HugepageDeviceCommand = DeviceCommand<true>;
using HostMemDeviceCommand = DeviceCommand<false>;
```

#### 3.4.3 SystemMemoryManager - 系统内存管理器

**文件**: `/tmp/tt-metal/tt_metal/impl/dispatch/system_memory_manager.hpp`

管理命令队列的系统内存区域，处理主机与设备间的数据传输。

```cpp
// 行 22-117: 系统内存管理器
class SystemMemoryManager {
public:
    SystemMemoryManager(ChipId device_id, uint8_t num_hw_cqs);

    // 事件管理
    uint32_t get_next_event(uint8_t cq_id);
    uint32_t get_last_event(uint8_t cq_id);
    void reset_event_id(uint8_t cq_id);
    void increment_event_id(uint8_t cq_id, uint32_t val);
    void set_last_completed_event(uint8_t cq_id, uint32_t event_id);
    uint32_t get_last_completed_event(uint8_t cq_id);

    // 队列大小管理
    void set_issue_queue_size(uint8_t cq_id, uint32_t issue_queue_size);
    uint32_t get_issue_queue_size(uint8_t cq_id) const;
    uint32_t get_issue_queue_limit(uint8_t cq_id) const;
    uint32_t get_completion_queue_size(uint8_t cq_id) const;
    uint32_t get_completion_queue_limit(uint8_t cq_id) const;

    // 队列指针
    uint32_t get_issue_queue_write_ptr(uint8_t cq_id) const;
    uint32_t get_completion_queue_read_ptr(uint8_t cq_id) const;
    uint32_t get_completion_queue_read_toggle(uint8_t cq_id) const;

    // 命令提交
    void* issue_queue_reserve(uint32_t cmd_size_B, uint8_t cq_id);
    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr);
    void issue_queue_push_back(uint32_t push_size_B, uint8_t cq_id);

    // 完成队列
    uint32_t completion_queue_wait_front(uint8_t cq_id, std::atomic<bool>& exit_condition) const;
    void send_completion_queue_read_ptr(uint8_t cq_id) const;
    void* get_completion_queue_ptr(uint8_t cq_id) const;
    void completion_queue_pop_front(uint32_t num_pages_read, uint8_t cq_id);

    // 获取队列
    void fetch_queue_reserve_back(uint8_t cq_id);
    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher = false);

    // 旁路模式（用于测试）
    void set_bypass_mode(bool enable, bool clear);
    bool get_bypass_mode() const;
    std::vector<uint32_t>& get_bypass_data();

private:
    ChipId device_id = 0;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start = nullptr;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size = 0;

    std::vector<uint32_t> cq_to_event;
    std::vector<uint32_t> cq_to_last_completed_event;
    mutable std::vector<std::mutex> cq_to_event_locks;

    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<umd::Writer> prefetch_q_writers;
    std::vector<umd::Writer> completion_q_writers;

    bool bypass_enable = false;
    std::vector<uint32_t> bypass_buffer;
};
```

#### 3.4.4 HWCommandQueue - 硬件命令队列

**文件**: `/tmp/tt-metal/tt_metal/impl/dispatch/hardware_command_queue.hpp`

```cpp
// 行 30-58: 硬件命令队列
class HWCommandQueue {
public:
    HWCommandQueue(IDevice* device, uint32_t id, NOC noc_index);
    ~HWCommandQueue() = default;

    const CoreCoord& virtual_enqueue_program_dispatch_core() const;

    void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data);

    uint32_t id() const;
    SystemMemoryManager& sysmem_manager();
    IDevice* device();

    void terminate();  // 终止命令队列

private:
    uint32_t id_;
    SystemMemoryManager& manager_;
    IDevice* device_;
    CoreCoord virtual_enqueue_program_dispatch_core_;
};
```

#### 3.4.5 DispatchTopology - 调度拓扑

**文件**: `/tmp/tt-metal/tt_metal/impl/dispatch/topology.hpp`

管理调度内核的拓扑结构，支持多设备、多隧道配置。

```cpp
// 行 46-56: 调度内核节点
struct DispatchKernelNode {
    int id;
    ChipId device_id;                // 内核所在设备
    ChipId servicing_device_id;      // 服务的远程设备（MMIO 内核用）
    uint8_t cq_id;                   // 实现的命令队列
    DispatchWorkerType kernel_type;  // 调度内核类型
    std::vector<int> upstream_ids;   // 上游调度内核
    std::vector<int> downstream_ids; // 下游调度内核
    noc_selection_t noc_selection;   // NOC 选择
    int tunnel_index{-1};            // 隧道索引
};

// 行 58-104: 调度拓扑
class DispatchTopology {
public:
    explicit DispatchTopology(
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        DeviceManager* device_manager,
        const GetControlPlaneFn& get_control_plane = {},
        ...);
    ~DispatchTopology();

    // 填充调度内核
    void populate_fd_kernels(const std::vector<Device*>& devices, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::set<ChipId>& device_ids, uint32_t num_hw_cqs);
    void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes);

    // 命令队列程序
    void populate_cq_static_args(Device* device);
    void create_cq_program(Device* device);
    void compile_cq_programs();
    std::unique_ptr<Program> get_compiled_cq_program(Device* device);
    void configure_dispatch_cores(Device* device);

    // 查询
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(ChipId dev_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(ChipId dev_id) const;

    void reset();

private:
    std::vector<DispatchKernelNode> generate_nodes(const std::set<ChipId>& device_ids,
                                                    uint32_t num_hw_cqs) const;

    const ContextDescriptor& descriptor_;
    dispatch_core_manager& dispatch_core_manager_;
    DeviceManager* device_manager_;

    std::unique_ptr<DispatchMemMap> dispatch_mem_map_[enchantum::to_underlying(CoreType::COUNT)];
    std::vector<FDKernel*> node_id_to_kernel_;
    std::unique_ptr<detail::ProgramCompileGroup> command_queue_compile_group_;

    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> dispatch_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> routing_cores_;
    std::unordered_map<ChipId, std::unordered_set<TerminationInfo>> termination_info_;
};
```

---

### 3.5 program/ 程序管理

#### 3.5.1 ProgramImpl - 程序实现

**文件**: `/tmp/tt-metal/tt_metal/impl/program/program_impl.hpp`

ProgramImpl 是 Program 的内部实现，管理内核、循环缓冲区、信号量等资源。

```cpp
// 行 175-442: 程序实现类
class ProgramImpl : public std::enable_shared_from_this<ProgramImpl> {
public:
    ProgramImpl();
    ~ProgramImpl() noexcept;

    // ID 管理
    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;
    ProgramId get_id() const;
    std::size_t num_kernels() const;

    // 资源访问
    std::span<const std::shared_ptr<CircularBufferImpl>> circular_buffers() const;
    const std::vector<Semaphore>& semaphores() const;
    KernelGroup* kernels_on_core(const CoreCoord& core, uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);

    // 编译和分配
    void compile(IDevice* device, bool force_slow_dispatch = false);
    void allocate_circular_buffers(const IDevice* device);
    void allocate_dataflow_buffers(const IDevice* device);
    void invalidate_circular_buffer_allocation();
    void invalidate_dataflow_buffer_allocation();

    // 最终化
    bool is_finalized() const;
    void set_finalized();
    void finalize_offsets(IDevice* device);
    static uint32_t finalize_program_offsets(
        IDevice* device,
        const KernelsGetter& kernels_getter,
        const KernelGroupsGetter& kernel_groups_getter,
        const SemaphoresGetter& semaphores_getter,
        tt::stl::Span<ProgramImpl*> programs);

    // 缓存
    bool is_cached() const { return this->cached_device_hash_.has_value(); }
    void set_cached(uint64_t device_hash) { this->cached_device_hash_ = device_hash; }
    ProgramBinaryStatus get_program_binary_status(ChipId device_id) const;
    void set_program_binary_status(ChipId device_id, ProgramBinaryStatus status);

    // 追踪
    void generate_trace_dispatch_commands(IDevice* device, bool use_prefetcher_cache);
    std::unordered_map<uint64_t, ProgramCommandSequence>& get_trace_cached_program_command_sequences() noexcept;

    // 添加资源
    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& core_type);
    void add_semaphore(const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    // 验证
    void validate_circular_buffer_region(const IDevice* device);
    void validate_circular_buffer_core_ranges(const IDevice* device);

private:
    // 循环缓冲区分配器
    struct CircularBufferAllocator {
        CircularBufferAllocator(const CoreRange& core_range_) : core_range(core_range_) {}
        CoreRange core_range;
        // 存储循环缓冲区分配的地址范围 [start, end)
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

        uint64_t get_cb_region_end() const {
            return this->l1_regions.empty() ? 0 : this->l1_regions.back().second;
        }
        void mark_address(uint64_t address, uint64_t size, uint64_t base_address);
        void reset_available_addresses() { this->l1_regions.clear(); }
    };

    uint64_t id;
    uint64_t runtime_id{0};
    static std::atomic<uint64_t> program_counter;

    // 内核存储（按可编程核心类型索引）
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_;
    std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_;
    std::vector<std::vector<uint8_t>> core_to_kernel_group_index_table_;

    // 循环缓冲区
    std::vector<std::shared_ptr<CircularBufferImpl>> circular_buffers_;
    std::unordered_map<CBHandle, std::shared_ptr<CircularBufferImpl>> circular_buffer_by_id_;
    std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_cb_indices_;
    std::vector<CircularBufferAllocator> cb_allocators_;

    // 数据流缓冲区
    std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> dataflow_buffers_;

    // 信号量
    std::vector<Semaphore> semaphores_;

    // 程序配置
    std::vector<ProgramConfig> program_configs_;
    std::vector<uint32_t> program_config_sizes_;
    uint32_t kernel_bins_sizeB = 0;

    // 缓存的命令序列
    std::unordered_map<uint64_t, ProgramCommandSequence> cached_program_command_sequences_;
    std::unordered_map<uint64_t, ProgramCommandSequence> trace_cached_program_command_sequences_;

    // 二进制状态
    std::unordered_map<ChipId, ProgramBinaryStatus> binaries_on_device_;
    std::optional<uint64_t> cached_device_hash_;

    bool finalized_{false};
    bool local_circular_buffer_allocation_needed_{false};
    bool local_dataflow_buffer_allocation_needed_{false};

    friend void program_dispatch::assemble_device_commands(...);
    friend HWCommandQueue;
    friend EnqueueProgramCommand;
};
```

#### 3.5.2 KernelGroup - 内核组

**文件**: `/tmp/tt-metal/tt_metal/impl/program/program_impl.hpp` (行 69-94)

```cpp
struct KernelGroup {
    uint32_t programmable_core_type_index{};
    CoreRangeSet core_ranges;
    // kernel_ids 按处理器索引排序
    std::vector<KernelHandle> kernel_ids;
    // RTA/CRTA 布局
    std::vector<uint32_t> rta_sizes;
    std::vector<uint32_t> crta_offsets;
    std::vector<uint32_t> crta_sizes;
    uint32_t total_rta_size{};
    // kernel_text_offsets 按核心内处理器索引
    std::vector<uint32_t> kernel_text_offsets;
    dev_msgs::launch_msg_t launch_msg;
    dev_msgs::go_msg_t go_msg;

    KernelGroup(
        const detail::ProgramImpl& program,
        uint32_t programmable_core_type_index,
        std::vector<KernelHandle> kernel_ids,
        uint64_t local_cb_mask,
        uint32_t min_remote_cb_start_index,
        const CoreRangeSet& new_ranges,
        const dev_msgs::Factory& dev_msgs_factory);

    CoreType get_core_type() const;
};
```

#### 3.5.3 program_dispatch 命名空间

**文件**: `/tmp/tt-metal/tt_metal/impl/program/dispatch.hpp`

```cpp
// 程序调度元数据
struct ProgramDispatchMetadata {
    std::vector<ConfigBufferEntry> kernel_config_addrs;
    uint32_t sync_count{};
    uint32_t stall_first{};
    uint32_t stall_before_program{};

    struct {
        uint32_t mesh_max_program_kernels_sizeB;
        bool is_cached;
        uint32_t offset;
    } prefetcher_cache_info{};
};

// 预期工作线程更新
struct ExpectedNumWorkerUpdates {
    uint32_t previous = 0;  // 更新前的工作线程数
    uint32_t current = 0;   // 更新后的工作线程数
    bool wrapped = false;   // 是否发生回绕
};

// 主要调度函数
namespace program_dispatch {

// RTA/CRTA 偏移配置
uint32_t configure_rta_offsets_for_kernel_groups(...);
uint32_t configure_crta_offsets_for_kernel_groups(...);

// 最终化函数
uint32_t finalize_rt_args(...);
uint32_t finalize_sems(...);
uint32_t finalize_cbs(...);
uint32_t finalize_kernel_bins(...);

// 命令序列操作
void insert_stall_cmds(ProgramCommandSequence& program_command_sequence, SubDeviceId sub_device_id, IDevice* device);
void update_program_dispatch_commands(...);
void write_program_command_sequence(...);

// 追踪
TraceNode create_trace_node(detail::ProgramImpl& program, IDevice* device, bool use_prefetcher_cache);

// 工作线程管理
ExpectedNumWorkerUpdates get_expected_num_workers_completed_updates(uint32_t num_workers, uint32_t num_additional_workers);
void reset_expected_num_workers_completed_on_device(IDevice* device, SubDeviceId sub_device_id,
                                                    uint32_t num_expected_workers, uint8_t cq_id);

}  // namespace program_dispatch
```

---

### 3.6 kernels/ 内核管理

#### 3.6.1 Kernel 基类

**文件**: `/tmp/tt-metal/tt_metal/impl/kernels/kernel.hpp`

```cpp
// 行 111-250: 内核基类
class Kernel : public JitBuildSettings {
public:
    using Config = std::variant<
        DataMovementConfig,
        EthernetConfig,
        ComputeConfig,
        experimental::quasar::QuasarDataMovementConfig,
        experimental::quasar::QuasarComputeConfig>;

    virtual ~Kernel() override = default;

    // 基本信息
    std::string name() const;
    const KernelSource& kernel_source() const { return kernel_src_; }
    const CoreRangeSet& core_range_set() const { return core_range_set_; }
    const std::set<CoreCoord>& logical_cores() const;
    bool is_on_logical_core(const CoreCoord& logical_core) const;

    // 编译时参数
    std::vector<uint32_t> compile_time_args() const { return compile_time_args_; }
    std::unordered_map<std::string, uint32_t> named_compile_time_args() const { return named_compile_time_args_; }

    // 运行时参数
    std::vector<uint32_t>& runtime_args(const CoreCoord& logical_core);
    RuntimeArgsData& runtime_args_data(const CoreCoord& logical_core);
    void set_runtime_args(const CoreCoord& logical_core, stl::Span<const uint32_t> runtime_args);

    std::vector<uint32_t>& common_runtime_args();
    void set_common_runtime_args(stl::Span<const uint32_t> runtime_args);

    // 配置
    virtual bool configure(IDevice* device, const CoreCoord& logical_core,
                          uint32_t base_address, const uint32_t offsets[]) const = 0;
    virtual Config config() const = 0;

    // HAL 信息
    HalProgrammableCoreType get_kernel_programmable_core_type() const { return programmable_core_type_; }
    HalProcessorClassType get_kernel_processor_class() const { return processor_class_; }
    virtual uint32_t get_kernel_processor_type(int index) const = 0;
    CoreType get_kernel_core_type() const;

    // 二进制管理
    const std::vector<const ll_api::memory*>& binaries(uint64_t build_key) const;
    void set_binaries(uint64_t build_key, std::vector<const ll_api::memory*>&& binaries);
    virtual void generate_binaries(IDevice* device, JitBuildOptions& build_options) const = 0;
    virtual void read_binaries(IDevice* device) = 0;

    // 哈希
    uint64_t compute_hash() const;

protected:
    Kernel(
        HalProgrammableCoreType programmable_core_type,
        HalProcessorClassType processor_class,
        const KernelSource& kernel_src,
        const CoreRangeSet& core_range_set,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines,
        const std::unordered_map<std::string, uint32_t>& named_compile_args);

    HalProgrammableCoreType programmable_core_type_;
    HalProcessorClassType processor_class_;

    KernelSource kernel_src_;
    std::string kernel_full_name_;  // 名称 + 哈希
    CoreRangeSet core_range_set_;

    std::vector<uint32_t> compile_time_args_;
    std::unordered_map<std::string, uint32_t> named_compile_time_args_;

    // 运行时参数存储 [核心][处理器][参数]
    std::vector<std::vector<std::vector<uint32_t>>> core_to_runtime_args_;
    std::vector<std::vector<RuntimeArgsData>> core_to_runtime_args_data_;

    uint32_t common_runtime_args_count_{0};
    std::vector<uint32_t> common_runtime_args_;

    std::map<std::string, std::string> defines_;
    std::set<CoreCoord> logical_cores_;

    // 构建密钥 -> 二进制
    std::unordered_map<uint64_t, std::vector<const ll_api::memory*>> binaries_;
};
```

#### 3.6.2 DataMovementKernel - 数据移动内核

```cpp
// 行 252-292: 数据移动内核
class DataMovementKernel : public Kernel {
public:
    DataMovementKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set,
                       const DataMovementConfig& config) :
        Kernel(
            HalProgrammableCoreType::TENSIX,
            HalProcessorClassType::DM,
            kernel_src, cr_set, config.compile_args, config.defines, config.named_compile_args),
        config_(config) {
        TT_FATAL(MetalContext::instance().get_cluster().arch() != ARCH::QUASAR,
                 "DataMovementKernel is not supported on Quasar.");
    }

    uint32_t get_kernel_processor_type(int index) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;
    bool configure(IDevice* device, const CoreCoord& logical_core,
                   uint32_t base_address, const uint32_t offsets[]) const override;
    Config config() const override { return this->config_; }

private:
    const DataMovementConfig config_;
    uint8_t expected_num_binaries() const override;
    std::string config_hash() const override;
};
```

#### 3.6.3 ComputeKernel - 计算内核

```cpp
// 行 332-373: 计算内核
class ComputeKernel : public Kernel {
public:
    ComputeKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set,
                  const ComputeConfig& config) :
        Kernel(
            HalProgrammableCoreType::TENSIX,
            HalProcessorClassType::COMPUTE,
            kernel_src, cr_set, config.compile_args, config.defines, config.named_compile_args),
        config_(config) {
        TT_FATAL(MetalContext::instance().get_cluster().arch() != ARCH::QUASAR,
                 "ComputeKernel is not supported on Quasar.");
    }

    uint32_t get_kernel_processor_type(int index) const override;
    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;
    bool configure(IDevice* device, const CoreCoord& logical_core,
                   uint32_t base_address, const uint32_t offsets[]) const override;
    Config config() const override { return this->config_; }

private:
    const ComputeConfig config_;
    uint8_t expected_num_binaries() const override;
    std::string config_hash() const override;
};
```

#### 3.6.4 EthernetKernel - 以太网内核

```cpp
// 行 294-330: 以太网内核
class EthernetKernel : public Kernel {
public:
    EthernetKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set,
                   const EthernetConfig& config) :
        Kernel(
            config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH
                                          : HalProgrammableCoreType::ACTIVE_ETH,
            HalProcessorClassType::DM,
            kernel_src, cr_set, config.compile_args, config.defines, config.named_compile_args),
        config_(config) {}

    uint32_t get_kernel_processor_type(int index) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;
    bool configure(IDevice* device, const CoreCoord& logical_core,
                   uint32_t base_address, const uint32_t offsets[]) const override;
    Config config() const override { return this->config_; }
    bool is_idle_eth() const;

private:
    const EthernetConfig config_;
    uint8_t expected_num_binaries() const override;
    std::string config_hash() const override;
};
```

---

### 3.7 tensor/ 张量管理

**文件**: `/tmp/tt-metal/tt_metal/impl/tensor/tensor_types.cpp`

张量类型模块定义了张量在设备上的存储格式和布局规范。

```cpp
// 主要类型定义
// - TensorMemoryLayout: 张量内存布局（INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED）
// - ShardSpec: 分片规范
// - BufferDistributionSpec: 缓冲区分布规范
```

---

### 3.8 event/ & trace/ 事件与追踪

#### 3.8.1 Event Dispatch

**文件**: `/tmp/tt-metal/tt_metal/impl/event/dispatch.hpp`

```cpp
// 行 21-29: 事件描述符
struct ReadEventDescriptor {
    uint32_t event_id;
    uint32_t global_offset{0};

    explicit ReadEventDescriptor(uint32_t event) : event_id(event) {}
    void set_global_offset(uint32_t offset) { global_offset = offset; }
    uint32_t get_global_event_id() const { return global_offset + event_id; }
};

// 行 31-56: 事件调度函数
namespace event_dispatch {

// 发出记录事件命令
void issue_record_event_commands(
    IDevice* device,
    ChipId device_id,
    uint32_t event_id,
    uint8_t cq_id,
    uint32_t num_command_queues,
    SystemMemoryManager& manager,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    bool notify_host = true,
    bool clear_count = false);

// 发出等待事件命令
void issue_wait_for_event_commands(
    uint8_t cq_id, uint8_t event_cq_id, SystemMemoryManager& sysmem_manager, uint32_t event_id);

// 从完成队列读取事件
void read_events_from_completion_queue(
    ReadEventDescriptor& event_descriptor,
    ChipId mmio_device_id,
    ChipId device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager);

}  // namespace event_dispatch
```

#### 3.8.2 Trace Buffer

**文件**: `/tmp/tt-metal/tt_metal/impl/trace/trace_buffer.hpp`

```cpp
// 行 23-27: 追踪工作线程描述符
struct TraceWorkerDescriptor {
    uint32_t num_completion_worker_cores = 0;
    uint32_t num_traced_programs_needing_go_signal_multicast = 0;
    uint32_t num_traced_programs_needing_go_signal_unicast = 0;
};

// 行 29-36: 追踪描述符
struct TraceDescriptor {
    // sub_device_id 到描述符的映射
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    // 存储映射键的向量（优化用）
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<uint32_t> data;
};

// 行 38-46: 追踪缓冲区
struct TraceBuffer {
    std::shared_ptr<TraceDescriptor> desc;
    std::shared_ptr<Buffer> buffer;

    TraceBuffer(std::shared_ptr<TraceDescriptor> desc, std::shared_ptr<Buffer> buffer);
    ~TraceBuffer();

    void validate();
};
```

---

## 4. 数据流向

### 4.1 程序执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Host Side                                   │
│                                                                          │
│  1. CreateProgram()                                                      │
│       ↓                                                                  │
│  2. CreateKernel() → Kernel::generate_binaries() → Compile               │
│       ↓                                                                  │
│  3. CreateCircularBuffer() / CreateBuffer()                              │
│       ↓                                                                  │
│  4. Program::finalize_offsets()                                          │
│       ↓                                                                  │
│  5. EnqueueProgram()                                                     │
│       ↓                                                                  │
│  6. program_dispatch::assemble_device_commands()                         │
│       ↓                                                                  │
│  7. DeviceCommand (binary) → SystemMemoryManager::issue_queue_reserve()  │
│       ↓                                                                  │
│  8. issue_queue_push_back() → Write to HugePage                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           Device Side                                    │
│                                                                          │
│  9. Prefetch Kernel reads commands from HugePage                         │
│       ↓                                                                  │
│  10. Dispatch Kernel processes commands                                  │
│       ↓                                                                  │
│  11. Write kernel binaries to L1/DRAM                                    │
│       ↓                                                                  │
│  12. Send GO signal to Worker Cores                                      │
│       ↓                                                                  │
│  13. Worker Cores execute kernels                                        │
│       ↓                                                                  │
│  14. Completion signal → Completion Queue                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 内存分配流程

```
Buffer::create()
    ↓
Buffer::allocate_impl()
    ↓
AllocatorImpl::allocate_buffer(Buffer* buffer)
    ↓
BankManager::allocate_buffer(size, page_size, bottom_up, ...)
    ↓
FreeListOpt::allocate(size_bytes, bottom_up, address_limit)
    ↓
[Search size-segregated list] → [Find best block] → [Split if needed]
    ↓
Return device address
```

### 4.3 追踪捕获流程

```
BeginTraceCapture()
    ↓
[Record commands to TraceDescriptor instead of issuing]
    ↓
EnqueueProgram() → program_dispatch::create_trace_node()
    ↓
EndTraceCapture()
    ↓
TraceBuffer::validate()
    ↓
ExecuteTrace() → Replay recorded commands
```

---

## 5. 设计模式与实现技巧

### 5.1 设计模式

| 模式 | 应用位置 | 说明 |
|------|----------|------|
| **单例模式** | `MetalContext`, `DeviceManager` | 全局唯一实例管理 |
| **工厂模式** | `Buffer::create()`, `Kernel::create()` | 统一对象创建 |
| **策略模式** | `Algorithm` 基类 + `FreeListOpt` | 可替换的分配算法 |
| **观察者模式** | `WatcherServer`, `DPrintServer` | 调试信息监听 |
| **访问者模式** | `JitBuildSettings` | 内核构建配置处理 |
| **PIMPL** | `Program`/`ProgramImpl` | 接口与实现分离 |

### 5.2 实现技巧

#### 5.2.1 SoA (Structure of Arrays)

FreeListOpt 使用 SoA 代替传统的链表结构，提高缓存局部性：

```cpp
// 传统链表（AoS）
struct Block {
    DeviceAddr address;
    DeviceAddr size;
    Block* prev;
    Block* next;
    bool is_allocated;
};
std::list<Block> blocks;  // 内存不连续，缓存不友好

// SoA 优化
std::vector<DeviceAddr> block_address_;
std::vector<DeviceAddr> block_size_;
std::vector<ssize_t> block_prev_block_;
std::vector<ssize_t> block_next_block_;
std::vector<uint8_t> block_is_allocated_;
// 内存连续，缓存友好
```

#### 5.2.2 大小分级 (Size Segregation)

```cpp
// 避免在大块分配时扫描小块
static constexpr size_t size_segregated_base = 1024;
std::vector<std::vector<size_t>> free_blocks_segregated_by_size_;

size_t get_size_segregated_index(DeviceAddr size_bytes) const {
    // 使用 log2 快速计算分级索引
    size_t lg = 0;
    size_t n = size_bytes / size_segregated_base;
    while (n >>= 1) { lg++; }
    return std::min(size_segregated_count - 1, lg);
}
```

#### 5.2.3 哈希表加速释放

```cpp
// O(1) 查找已分配块
static constexpr size_t n_alloc_table_buckets = 512;
std::vector<std::vector<std::pair<DeviceAddr, size_t>>> allocated_block_table_;

size_t hash_device_address(DeviceAddr address) {
    return (address >> 6) % n_alloc_table_buckets;  // 按 64 字节对齐取哈希
}
```

#### 5.2.4 双指针范围减法

BankManager 使用双指针算法高效计算可用地址：

```cpp
// O(|available| + |allocated|) 复杂度
auto subtract_ranges = [](const auto& available_ranges, const auto& allocated_ranges) {
    size_t j = 0;  // 单调指针，不重置
    for (const auto& available : available_ranges) {
        // 跳过左侧不重叠的已分配范围
        while (j < allocated_ranges.size() && allocated_ranges[j].second <= available_start) {
            j++;
        }
        // 处理重叠...
    }
};
```

#### 5.2.5 线程本地存储

```cpp
// 每个线程独立的命令队列 ID 栈
static thread_local CommandQueueIdStack command_queue_id_stack_for_thread_;

// 允许线程临时切换命令队列而不影响其他线程
void push_cq_id(uint8_t cq_id) {
    command_queue_id_stack_for_thread_.push_back(cq_id);
}

void pop_cq_id() {
    command_queue_id_stack_for_thread_.pop_back();
}
```

---

## 6. 源码注释摘录

### 6.1 分配器线程安全

```cpp
// allocator.hpp, 行 15-16
// THREAD SAFETY: Allocator is thread safe.
class AllocatorImpl {
    // ...
    mutable std::mutex mutex_;
};
```

### 6.2 FreeListOpt 优化说明

```cpp
// free_list_opt.hpp, 行 19-26
// Essentially the same free list algorithm as FreeList with BestFit policy,
// but with (IMO absurdly) optimized code. Including:
// - SoA instead of linked list for the free list
// - Size segregated to avoid unnecessary searches of smaller blocks
// - Hash table to store allocated blocks for faster block lookup during deallocation
// - Keeps metadata locality to avoid cache misses
// - Metadata reuse to avoid allocations
```

### 6.3 BankManager 依赖说明

```cpp
// bank_manager.hpp, 行 35-48
/**
 * @brief Describes dependencies between multiple allocators which share the same memory space.
 *
 * Allocator dependencies are bidirectional and are stored as undirected adjacency lists.
 * Eg. If allocator A depends on allocator B, then A and B cannot allocate in regions
 * occupied by the other. It is used by BankManager to differentiate between different
 * allocators and query dependencies between them.
 */
```

### 6.4 分配安全警告

```cpp
// allocator.cpp, 行 95-108
void AllocatorImpl::verify_safe_allocation() const {
    // Inform the user that its unsafe to allocate buffers when a trace is live on device.
    // If the user does this, they are meant to ensure that buffers allocated when a trace
    // is active, have a lifetime that ends before the trace is executed.
    // Print the warning once per device, to ensure that user output is not clobbered.
    thread_local static bool warning_generated = false;
    if (allocations_unsafe_ and not warning_generated) {
        log_warning(tt::LogMetal,
            "Allocating device buffers is unsafe due to the existence of an active trace. "
            "These buffers may be corrupted once a trace is executed.");
        warning_generated = true;
    }
}
```

### 6.5 设备命令常量

```cpp
// device_command.hpp, 行 42-44
static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;
static constexpr uint32_t LOG2_PROGRAM_PAGE_SIZE = std::bit_width(PROGRAM_PAGE_SIZE) - 1;
```

### 6.6 程序编译组

```cpp
// program_impl.hpp, 行 147-172
// Internal class for holding a group of programs for parallel compilation.
class ProgramCompileGroup {
private:
    std::mutex mutex_;
    std::unordered_map<IDevice*, std::unique_ptr<Program>> program_device_map_;

public:
    // Add a program to the compile group. Throws if the program already exists in the group.
    void add_program(IDevice* device, std::unique_ptr<Program> program);

    // Compiles all programs in the group
    void compile_all(bool force_slow_dispatch);

    // Write runtime args for all programs in the group
    void write_runtime_args(bool force_slow_dispatch);
};
```

---

## 7. 关键文件路径汇总

| 组件 | 关键文件 |
|------|----------|
| **分配器** | `/tmp/tt-metal/tt_metal/impl/allocator/allocator.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/allocator/bank_manager.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/allocator/algorithms/free_list_opt.hpp` |
| **缓冲区** | `/tmp/tt-metal/tt_metal/impl/buffers/buffer.cpp` |
| | `/tmp/tt-metal/tt_metal/impl/buffers/circular_buffer.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/buffers/semaphore.hpp` |
| **设备** | `/tmp/tt-metal/tt_metal/impl/device/device_impl.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/device/device_manager.hpp` |
| **调度** | `/tmp/tt-metal/tt_metal/impl/dispatch/device_command.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/dispatch/system_memory_manager.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/dispatch/hardware_command_queue.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/dispatch/topology.hpp` |
| **程序** | `/tmp/tt-metal/tt_metal/impl/program/program_impl.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/program/dispatch.hpp` |
| **内核** | `/tmp/tt-metal/tt_metal/impl/kernels/kernel.hpp` |
| **上下文** | `/tmp/tt-metal/tt_metal/impl/context/metal_context.hpp` |
| **事件/追踪** | `/tmp/tt-metal/tt_metal/impl/event/dispatch.hpp` |
| | `/tmp/tt-metal/tt_metal/impl/trace/trace_buffer.hpp` |

---

*文档生成时间: 2026-03-13*
*基于 TT-Metal 源码版本: main branch*
