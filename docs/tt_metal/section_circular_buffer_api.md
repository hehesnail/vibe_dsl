# Circular Buffer (CB) API 完整参考手册

> **文档版本**: 1.0
> **最后更新**: 2026-03-12
> **适用范围**: TT-Metalium 框架

---

## 目录

1. [概述](#1-概述)
2. [Host 端 CB 创建配置](#2-host-端-cb-创建配置)
3. [Host 端 CB 管理](#3-host-端-cb-管理)
4. [Device 端 CB 操作](#4-device-端-cb-操作)
5. [CB 查询函数](#5-cb-查询函数)
6. [CB 高级用法](#6-cb-高级用法)
7. [常见问题排查](#7-常见问题排查)

---

## 1. 概述

Circular Buffer (CB) 是 TT-Metalium 中用于在 Tensix 核心 L1 SRAM 中存储数据的核心机制。CB 作为生产者-消费者模式的基础组件，在 Reader、Compute 和 Writer Kernel 之间传递数据 Tile。

### 1.1 CB 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                    Circular Buffer (CB)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│   │   Reader     │ ───> │      CB      │ ───> │ Compute  │ │
│   │  (Producer)  │      │   (L1 SRAM)  │      │(Consumer)│ │
│   └──────────────┘      └──────────────┘      └──────────┘ │
│         cb_reserve_back()    cb_wait_front()               │
│         cb_push_back()       cb_pop_front()                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 CB 关键特性

| 特性 | 说明 |
|------|------|
| **位置** | 位于 Tensix 核心 L1 SRAM |
| **大小** | 每核心 ~1.5 MB (Wormhole/Blackhole) |
| **数量** | 最多 32 个 CB (索引 0-31) |
| **对齐** | 16 字节对齐 (L1_ALIGNMENT) |
| **组织** | 按页 (page) 组织，每页包含一个或多个 Tile |

---

## 2. Host 端 CB 创建配置

### 2.1 CreateCircularBuffer

创建 Circular Buffer 并返回句柄。

**函数签名**:
```cpp
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config
);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `program` | `Program&` | 要添加 CB 的程序对象 |
| `core_spec` | `CoreCoord` / `CoreRange` / `CoreRangeSet` | 目标核心或核心范围 |
| `config` | `CircularBufferConfig&` | CB 配置对象 |

**返回值**:
- `CBHandle` - CB 句柄，用于后续引用此 CB

**使用示例**:
```cpp
// 创建程序
Program program;

// 定义核心范围
CoreRange cores({0, 0}, {7, 7});  // 8x8 核心网格

// 配置 CB
CircularBufferConfig cb_config(
    num_pages * page_size,                    // 总大小
    {{cb_index, tt::DataFormat::Float16_b}}   // 数据格式映射
).set_page_size(cb_index, page_size);

// 创建 CB
CBHandle cb_handle = CreateCircularBuffer(program, cores, cb_config);
```

---

### 2.2 CircularBufferConfig 结构体详解

**定义**:
```cpp
class CircularBufferConfig {
public:
    // 构造函数
    CircularBufferConfig(
        uint32_t total_size,
        const std::map<uint8_t, tt::DataFormat>& data_formats
    );

    // 链式配置方法
    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);
    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);
    CircularBufferConfig& set_tile_size(uint8_t buffer_index, uint32_t tile_size);

    // 获取配置
    uint32_t total_size() const;
    uint32_t page_size(uint8_t buffer_index) const;
    tt::DataFormat data_format(uint8_t buffer_index) const;
};
```

**配置参数详解**:

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `total_size` | `uint32_t` | CB 总大小（字节） | 8192 (8KB) |
| `page_size` | `uint32_t` | 每页大小（字节） | 2048 (1 Tile) |
| `data_format` | `tt::DataFormat` | 数据格式 | Float16_b, Bfp8_b |
| `buffer_index` | `uint8_t` | CB 索引 (0-31) | 0, 1, 2... |

**支持的数据格式**:

| 格式 | 描述 | 每元素字节数 |
|------|------|-------------|
| `Float32` | 32位浮点 | 4 |
| `Float16` | 16位浮点 | 2 |
| `Float16_b` | Brain 16位浮点 | 2 |
| `Bfp8_b` | Block floating point 8-bit | 1 |
| `Bfp4_b` | Block floating point 4-bit | 0.5 |
| `Int32` | 32位整数 | 4 |
| `UInt16` | 16位无符号整数 | 2 |
| `UInt8` | 8位无符号整数 | 1 |

**完整配置示例**:
```cpp
// 示例 1: 基本配置
CircularBufferConfig config1(
    8192,  // 8KB 总大小
    {{0, tt::DataFormat::Float16_b}}  // CB 0 使用 Float16_b
).set_page_size(0, 2048);  // 每页 2048 字节 (1 Tile)

// 示例 2: 多 CB 索引配置
CircularBufferConfig config2(
    16384,  // 16KB 总大小
    {
        {0, tt::DataFormat::Float16_b},
        {1, tt::DataFormat::Float16_b}
    }
).set_page_size(0, 2048)
 .set_page_size(1, 2048);

// 示例 3: 使用 Buffer 地址的动态分配
auto buffer = CreateBuffer(config);
CircularBufferConfig config3(
    buffer_size,
    {{0, tt::DataFormat::Bfp8_b}}
).set_globally_allocated_address(*buffer);
```

---

### 2.3 页面大小和总大小配置

**计算公式**:
```cpp
// Tile 大小计算 (32x32 元素)
uint32_t tile_size = 32 * 32 * element_size;

// Float16_b (2字节/元素)
uint32_t tile_size_f16b = 32 * 32 * 2;  // 2048 bytes

// Float32 (4字节/元素)
uint32_t tile_size_f32 = 32 * 32 * 4;   // 4096 bytes

// CB 总大小计算
uint32_t num_tiles = 4;  // 同时容纳 4 个 Tile
uint32_t cb_total_size = num_tiles * tile_size;  // 8192 bytes
```

**双缓冲配置**:
```cpp
// 双缓冲: 同时容纳 2 组 tiles (一组用于计算，一组用于传输)
uint32_t num_tiles_double_buffer = 2 * num_tiles_per_batch;
uint32_t cb_size_double = num_tiles_double_buffer * tile_size;
```

---

### 2.4 数据格式设置

**格式选择指南**:

| 场景 | 推荐格式 | 原因 |
|------|----------|------|
| 训练 (高精度) | Float16_b | 平衡精度和性能 |
| 推理 (高性能) | Bfp8_b | 更高吞吐量 |
| 极致性能 | Bfp4_b | 最大带宽效率 |
| 整数运算 | Int32 | 精确整数计算 |

**格式设置示例**:
```cpp
// 矩阵乘法输入 (高精度)
CircularBufferConfig mm_input_config(
    8192,
    {
        {0, tt::DataFormat::Float16_b},  // 输入 A
        {1, tt::DataFormat::Float16_b}   // 输入 B
    }
).set_page_size(0, 2048)
 .set_page_size(1, 2048);

// 推理优化配置
CircularBufferConfig inference_config(
    4096,
    {{0, tt::DataFormat::Bfp8_b}}
).set_page_size(0, 1024);
```

---

## 3. Host 端 CB 管理

### 3.1 GetCircularBufferConfig

获取已创建 CB 的配置引用。

**函数签名**:
```cpp
const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `program` | `Program&` | 包含 CB 的程序 |
| `cb_handle` | `CBHandle` | CB 句柄 |

**返回值**:
- `const CircularBufferConfig&` - CB 配置的常量引用

**使用示例**:
```cpp
// 创建 CB
CBHandle cb = CreateCircularBuffer(program, core, config);

// 获取配置
const CircularBufferConfig& current_config = GetCircularBufferConfig(program, cb);
uint32_t current_size = current_config.total_size();
```

---

### 3.2 UpdateCircularBufferTotalSize

更新 CB 的总大小（动态调整）。

**函数签名**:
```cpp
void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `program` | `Program&` | 包含 CB 的程序 |
| `cb_handle` | `CBHandle` | CB 句柄 |
| `total_size` | `uint32_t` | 新的总大小（字节） |

**使用场景**:
- 运行时调整 CB 大小以适应不同输入尺寸
- 动态内存管理

**使用示例**:
```cpp
// 初始创建 CB
CircularBufferConfig config(4096, {{0, tt::DataFormat::Float16_b}});
CBHandle cb = CreateCircularBuffer(program, core, config);

// 后续需要更大空间时更新
UpdateCircularBufferTotalSize(program, cb, 8192);

// 注意: 新大小必须兼容原有 page_size 设置
```

---

### 3.3 UpdateCircularBufferPageSize

更新指定 CB 索引的页大小。

**函数签名**:
```cpp
void UpdateCircularBufferPageSize(
    Program& program,
    CBHandle cb_handle,
    uint8_t buffer_index,
    uint32_t page_size
);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `program` | `Program&` | 包含 CB 的程序 |
| `cb_handle` | `CBHandle` | CB 句柄 |
| `buffer_index` | `uint8_t` | CB 索引 (0-31) |
| `page_size` | `uint32_t` | 新的页大小（字节） |

**使用示例**:
```cpp
// 更新 CB 0 的页大小
UpdateCircularBufferPageSize(program, cb_handle, 0, 4096);

// 注意: 新 page_size 必须能整除 total_size
```

---

### 3.4 UpdateDynamicCircularBufferAddress

更新动态 CB 的基地址（用于动态内存分配场景）。

**函数签名**:
```cpp
void UpdateDynamicCircularBufferAddress(
    Program& program,
    CBHandle cb_handle,
    const Buffer& buffer
);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `program` | `Program&` | 包含 CB 的程序 |
| `cb_handle` | `CBHandle` | CB 句柄 |
| `buffer` | `const Buffer&` | 新的 Buffer 对象 |

**使用场景**:
- Metal Trace 重放时切换不同输入 Buffer
- 动态内存池管理
- 多阶段计算中复用 CB 空间

**使用示例**:
```cpp
// 创建动态 CB (使用 Buffer 地址)
auto buffer1 = CreateBuffer(buffer_config);
CircularBufferConfig cb_config(
    buffer1->size(),
    {{0, tt::DataFormat::Float16_b}}
).set_globally_allocated_address(*buffer1);

CBHandle dynamic_cb = CreateCircularBuffer(program, core, cb_config);

// 后续切换到另一个 Buffer
auto buffer2 = CreateBuffer(buffer_config);
UpdateDynamicCircularBufferAddress(program, dynamic_cb, *buffer2);
```

---

### 3.5 UpdateDynamicCircularBufferAddressAndTotalSize

同时更新动态 CB 的地址和总大小。

**函数签名**:
```cpp
void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program,
    CBHandle cb_handle,
    const Buffer& buffer,
    uint32_t total_size
);
```

**使用示例**:
```cpp
// 同时更新地址和大小
auto new_buffer = CreateBuffer(new_buffer_config);
UpdateDynamicCircularBufferAddressAndTotalSize(
    program,
    cb_handle,
    *new_buffer,
    new_buffer->size()
);
```

---

## 4. Device 端 CB 操作

### 4.1 cb_reserve_back

生产者预留 CB 后端空间。

**函数签名**:
```cpp
FORCE_INLINE void cb_reserve_back(int32_t operand, int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 要预留的页数 |

**功能描述**:
- 阻塞等待直到 CB 后端有足够空间容纳指定页数
- 生产者必须在写入数据前调用
- 与 `cb_push_back` 配对使用

**使用示例**:
```cpp
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 预留 1 页空间
        cb_reserve_back(cb_id, 1);

        // 获取写入地址
        uint32_t write_addr = get_write_ptr(cb_id);

        // 写入数据到 L1
        noc_async_read_tile(i, dram_buffer, write_addr);
        noc_async_read_barrier();

        // 提交数据
        cb_push_back(cb_id, 1);
    }
}
```

---

### 4.2 cb_push_back

生产者将数据提交到 CB。

**函数签名**:
```cpp
FORCE_INLINE void cb_push_back(const int32_t operand, const int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 要提交的页数 |

**功能描述**:
- 将预留的数据标记为可用，供消费者读取
- 必须在数据写入完成后调用
- 与 `cb_reserve_back` 配对使用

**使用示例**:
```cpp
// 批量提交示例
cb_reserve_back(cb_id, batch_size);

for (uint32_t i = 0; i < batch_size; i++) {
    uint32_t addr = get_write_ptr(cb_id) + i * page_size;
    noc_async_read_tile(tile_id + i, dram_buffer, addr);
}
noc_async_read_barrier();

// 批量提交
cb_push_back(cb_id, batch_size);
```

---

### 4.3 cb_wait_front

消费者等待 CB 前端数据可用。

**函数签名**:
```cpp
FORCE_INLINE void cb_wait_front(int32_t operand, int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 要等待的页数 |

**功能描述**:
- 阻塞等待直到 CB 前端有足够数据页可用
- 消费者必须在读取数据前调用
- 与 `cb_pop_front` 配对使用

**使用示例**:
```cpp
void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                // 等待输入数据就绪
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // 执行矩阵乘法
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                // 释放输入数据
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            // 输出结果
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
```

---

### 4.4 cb_pop_front

消费者释放 CB 前端数据。

**函数签名**:
```cpp
FORCE_INLINE void cb_pop_front(int32_t operand, int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 要释放的页数 |

**功能描述**:
- 标记已消费的数据页为可重用空间
- 释放的空间可供生产者再次使用
- 与 `cb_wait_front` 配对使用

**使用示例**:
```cpp
// 处理完数据后释放
cb_wait_front(cb_id, 2);  // 等待 2 页

// 处理数据...
process_tiles(cb_id, 2);

// 释放空间
cb_pop_front(cb_id, 2);
```

---

### 4.5 get_write_ptr

获取 CB 的当前写入地址。

**函数签名**:
```cpp
FORCE_INLINE uint32_t get_write_ptr(uint32_t operand);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `uint32_t` | CB 索引 (0-31) |

**返回值**:
- `uint32_t` - L1 内存中的写入地址

**使用示例**:
```cpp
cb_reserve_back(cb_id, 1);
uint32_t l1_write_addr = get_write_ptr(cb_id);

// 使用 NoC 读取数据到 CB
noc_async_read_tile(tile_id, dram_buffer, l1_write_addr);
noc_async_read_barrier();

cb_push_back(cb_id, 1);
```

---

### 4.6 get_read_ptr

获取 CB 的当前读取地址。

**函数签名**:
```cpp
FORCE_INLINE uint32_t get_read_ptr(uint32_t operand);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `uint32_t` | CB 索引 (0-31) |

**返回值**:
- `uint32_t` - L1 内存中的读取地址

**使用示例**:
```cpp
cb_wait_front(cb_id, 1);
uint32_t l1_read_addr = get_read_ptr(cb_id);

// 使用 NoC 写入数据到 DRAM
noc_async_write(l1_read_addr, dram_addr, tile_size);
noc_async_write_barrier();

cb_pop_front(cb_id, 1);
```

---

## 5. CB 查询函数

### 5.1 cb_pages_available_at_front

查询 CB 前端可用的页数。

**函数签名**:
```cpp
FORCE_INLINE bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 查询的页数 |

**返回值**:
- `true` - 前端至少有 `num_pages` 页可用
- `false` - 前端可用页数不足

**使用示例**:
```cpp
// 非阻塞检查
if (cb_pages_available_at_front(cb_id, 2)) {
    // 可以处理 2 页数据
    process_two_pages(cb_id);
    cb_pop_front(cb_id, 2);
} else {
    // 数据不足，执行其他工作或等待
}
```

---

### 5.2 cb_pages_reservable_at_back

查询 CB 后端可预留的页数。

**函数签名**:
```cpp
FORCE_INLINE bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `operand` | `int32_t` | CB 索引 (0-31) |
| `num_pages` | `int32_t` | 查询的页数 |

**返回值**:
- `true` - 后端至少有 `num_pages` 页空间可预留
- `false` - 后端可用空间不足

**使用示例**:
```cpp
// 检查是否有空间写入
if (cb_pages_reservable_at_back(cb_id, batch_size)) {
    // 可以批量写入
    write_batch(cb_id, batch_size);
} else {
    // 空间不足，等待或减小批量
}
```

---

### 5.3 cb_num_tiles_available

查询 CB 中可用的 Tile 数量（Device 端使用）。

**函数签名**:
```cpp
// 在 compute kernel 中通过 tile 计数查询
// 实际实现依赖于 CB 内部状态
```

**使用场景**:
- 调试时验证 CB 状态
- 条件执行逻辑

**使用示例**:
```cpp
// 调试检查
ASSERT(cb_num_tiles_available(cb_id) >= required_tiles);

// 条件处理
while (cb_num_tiles_available(cb_id) > 0) {
    process_one_tile(cb_id);
    cb_pop_front(cb_id, 1);
}
```

---

## 6. CB 高级用法

### 6.1 双缓冲模式详解

双缓冲通过交替使用两个 CB 实现计算和通信重叠。

```
时间线:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ 时间片  │   T0    │   T1    │   T2    │   T3    │
├─────────┼─────────┼─────────┼─────────┼─────────┤
│ CB_A    │ 读取    │ 计算    │ 读取    │ 计算    │
│ CB_B    │ 空闲    │ 读取    │ 计算    │ 读取    │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

**实现示例**:
```cpp
// Host 端配置
CircularBufferConfig cb_config_ping(
    2 * tile_size,  // 容纳 2 个 tiles
    {{0, tt::DataFormat::Float16_b}}
).set_page_size(0, tile_size);

CircularBufferConfig cb_config_pong(
    2 * tile_size,
    {{1, tt::DataFormat::Float16_b}}
).set_page_size(1, tile_size);

CBHandle cb_ping = CreateCircularBuffer(program, core, cb_config_ping);
CBHandle cb_pong = CreateCircularBuffer(program, core, cb_config_pong);

// Device 端 Reader Kernel (双缓冲)
void kernel_main() {
    uint32_t cb_ping = get_arg_val<uint32_t>(0);
    uint32_t cb_pong = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint32_t cb_current = cb_ping;
    uint32_t cb_next = cb_pong;

    // 预填充第一个 buffer
cb_reserve_back(cb_current, 2);
    noc_async_read_tile(0, dram_buffer, get_write_ptr(cb_current));
    noc_async_read_tile(1, dram_buffer, get_write_ptr(cb_current) + tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_current, 2);

    // 主循环
    for (uint32_t i = 2; i < num_tiles; i += 2) {
        // 在下一个 buffer 读取的同时，当前 buffer 被计算使用
        cb_reserve_back(cb_next, 2);
        noc_async_read_tile(i, dram_buffer, get_write_ptr(cb_next));
        noc_async_read_tile(i+1, dram_buffer, get_write_ptr(cb_next) + tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_next, 2);

        // 交换 buffer
        uint32_t temp = cb_current;
        cb_current = cb_next;
        cb_next = temp;
    }
}

// Device 端 Compute Kernel (双缓冲)
void MAIN {
    uint32_t cb_ping = get_arg_val<uint32_t>(0);
    uint32_t cb_pong = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint32_t cb_current = cb_ping;

    for (uint32_t i = 0; i < num_tiles; i += 2) {
        cb_wait_front(cb_current, 2);

        // 处理 2 个 tiles
        tile_regs_acquire();
        // ... 计算逻辑 ...
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 2);
        // ... 打包输出 ...
        cb_push_back(cb_out, 2);
        tile_regs_release();

        cb_pop_front(cb_current, 2);

        // 切换到另一个 buffer
        cb_current = (cb_current == cb_ping) ? cb_pong : cb_ping;
    }
}
```

---

### 6.2 多生产者/消费者模式

多个 Kernel 可以共享同一个 CB，需要谨慎管理同步。

```
场景 1: 单生产者 - 多消费者
┌──────────┐      ┌──────────┐
│ Producer │ ───> │    CB    │
│  (NOC)   │      │  (L1)    │
└──────────┘      └────┬─────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
      ┌────────┐  ┌────────┐  ┌────────┐
      │Compute1│  │Compute2│  │Compute3│
      │(TRISC) │  │(TRISC) │  │(TRISC) │
      └────────┘  └────────┘  └────────┘

场景 2: 多生产者 - 单消费者
┌──────────┐      ┌──────────┐
│Producer1 │ ───> │          │
│  (BRISC) │      │    CB    │ ───> │ Consumer |
├──────────┤      │  (L1)    │      │ (NOC)    │
│Producer2 │ ───> │          │
│  (NCRISC)│      └──────────┘
└──────────┘
```

**实现示例**:
```cpp
// Host 端: 多个 Reader Kernel 写入同一个 CB
Program program;
CoreRange cores({0, 0}, {3, 3});  // 4x4 网格

CircularBufferConfig cb_config(
    16384,
    {{0, tt::DataFormat::Float16_b}}
).set_page_size(0, 2048);

CBHandle shared_cb = CreateCircularBuffer(program, cores, cb_config);

// 每个核心有自己的 Reader，但写入同一个 CB
auto reader = CreateKernel(
    program,
    "reader.cpp",
    cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

// Device 端: 需要额外的同步机制
void kernel_main() {
    uint32_t core_id = get_arg_val<uint32_t>(0);
    uint32_t num_cores = get_arg_val<uint32_t>(1);

    // 使用信号量协调多生产者
    volatile tt_l1_ptr uint32_t* sem = get_semaphore(0);

    for (uint32_t i = core_id; i < total_tiles; i += num_cores) {
        // 预留空间
        cb_reserve_back(cb_id, 1);

        uint32_t write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(i, dram_buffer, write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);

        // 增加信号量计数
        noc_semaphore_inc(get_noc_addr(semaphore_core, sem_addr), 1);
    }
}
```

---

### 6.3 CB 大小计算最佳实践

**计算公式**:
```cpp
// 基础参数
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

// Tile 大小计算
uint32_t get_tile_size(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Float32:   return 32 * 32 * 4;  // 4096
        case tt::DataFormat::Float16:
        case tt::DataFormat::Float16_b: return 32 * 32 * 2;  // 2048
        case tt::DataFormat::Bfp8_b:    return 32 * 32 * 1;  // 1024
        case tt::DataFormat::Bfp4_b:    return 32 * 32 / 2;  // 512
        default: return 2048;
    }
}

// CB 大小计算
struct CBSizeConfig {
    uint32_t tiles_per_batch;      // 每批处理的 tiles
    uint32_t num_buffers;          // 缓冲数量 (1=单缓冲, 2=双缓冲)
    uint32_t extra_space;          // 额外空间 (用于对齐等)
};

uint32_t calculate_cb_size(
    tt::DataFormat format,
    const CBSizeConfig& config
) {
    uint32_t tile_size = get_tile_size(format);
    uint32_t base_size = config.tiles_per_batch * tile_size;
    uint32_t total_size = base_size * config.num_buffers + config.extra_space;

    // 确保 16 字节对齐
    total_size = (total_size + 15) & ~15;

    return total_size;
}

// 使用示例
CBSizeConfig config{
    .tiles_per_batch = 4,
    .num_buffers = 2,        // 双缓冲
    .extra_space = 32        // 额外对齐空间
};

uint32_t cb_size = calculate_cb_size(tt::DataFormat::Float16_b, config);
// 结果: 4 * 2048 * 2 + 32 = 16416 -> 对齐后 16416
```

**内存预算分配**:
```cpp
// Wormhole/Blackhole: 每核心 1.5 MB L1
constexpr uint32_t L1_SIZE_PER_CORE = 1.5 * 1024 * 1024;  // 1572864 bytes

// 典型分配方案
struct L1Budget {
    uint32_t kernel_stack;      // ~64KB
    uint32_t semaphores;        // ~4KB
    uint32_t cb_input0;         // ~256KB
    uint32_t cb_input1;         // ~256KB
    uint32_t cb_output;         // ~256KB
    uint32_t cb_intermediate;   // ~128KB
    uint32_t reserved;          // 剩余空间
};

// 验证总大小
void validate_l1_budget(const L1Budget& budget) {
    uint32_t total = budget.kernel_stack + budget.semaphores +
                     budget.cb_input0 + budget.cb_input1 +
                     budget.cb_output + budget.cb_intermediate;

    if (total > L1_SIZE_PER_CORE) {
        // 错误: 超出 L1 容量
    }
}
```

---

### 6.4 CB 索引分配策略

**推荐索引分配**:
```cpp
// 标准索引分配方案
namespace CBIndex {
    constexpr uint32_t INPUT_0 = 0;
    constexpr uint32_t INPUT_1 = 1;
    constexpr uint32_t INPUT_2 = 2;
    constexpr uint32_t INPUT_3 = 3;
    constexpr uint32_t OUTPUT_0 = 16;
    constexpr uint32_t OUTPUT_1 = 17;
    constexpr uint32_t INTERMEDIATE_0 = 24;
    constexpr uint32_t INTERMEDIATE_1 = 25;
}

// 使用示例
CircularBufferConfig in0_config(
    8192,
    {{CBIndex::INPUT_0, tt::DataFormat::Float16_b}}
).set_page_size(CBIndex::INPUT_0, 2048);

CircularBufferConfig out0_config(
    8192,
    {{CBIndex::OUTPUT_0, tt::DataFormat::Float16_b}}
).set_page_size(CBIndex::OUTPUT_0, 2048);
```

---

## 7. 常见问题排查

### 7.1 CB 溢出/下溢

**症状**: Kernel 挂起或产生错误结果

**原因与解决**:

| 问题 | 原因 | 解决 |
|------|------|------|
| 生产者过快 | CB 满，无法预留空间 | 减小 CB 大小或增加消费者速度 |
| 消费者过快 | CB 空，无法获取数据 | 增加生产者速度或添加延迟 |
| 未配对操作 | reserve/push 或 wait/pop 不匹配 | 确保每个操作都有对应配对 |
| 页数不匹配 | 预留/提交页数不一致 | 检查 num_pages 参数 |

**调试代码**:
```cpp
// 添加调试检查
void kernel_main() {
    #ifdef DEBUG_CB
    DPRINT << "CB State: available=" << cb_num_tiles_available(cb_id) << ENDL();
    #endif

    cb_reserve_back(cb_id, 1);

    #ifdef DEBUG_CB
    DPRINT << "After reserve: write_ptr=" << get_write_ptr(cb_id) << ENDL();
    #endif

    // ... 写入数据 ...

    cb_push_back(cb_id, 1);
}
```

---

### 7.2 内存对齐问题

**症状**: 数据损坏或硬件异常

**检查清单**:
- [ ] CB 总大小是否为 16 字节对齐
- [ ] Page 大小是否与数据格式匹配
- [ ] 地址计算是否正确

**验证代码**:
```cpp
// 对齐检查
static_assert((CB_SIZE & 15) == 0, "CB size must be 16-byte aligned");
static_assert((PAGE_SIZE & 15) == 0, "Page size must be 16-byte aligned");

// 运行时检查
uint32_t write_ptr = get_write_ptr(cb_id);
if (write_ptr & 15) {
    // 错误: 未对齐地址
}
```

---

### 7.3 多核同步问题

**症状**: 随机挂起或数据竞争

**解决方案**:
```cpp
// 使用信号量进行跨核同步
void kernel_main() {
    uint32_t my_core_id = get_arg_val<uint32_t>(0);
    uint32_t num_cores = get_arg_val<uint32_t>(1);

    // 获取信号量地址
    volatile tt_l1_ptr uint32_t* sync_sem = get_semaphore(0);

    // 阶段 1: 所有核心完成读取
    if (my_core_id == 0) {
        // 主核心等待所有从核心
        for (uint32_t i = 1; i < num_cores; i++) {
            noc_semaphore_wait(sync_sem + i, 1);
        }
        // 重置信号量
        for (uint32_t i = 0; i < num_cores; i++) {
            sync_sem[i] = 0;
        }
    } else {
        // 从核心通知主核心
        noc_semaphore_set_remote(
            (uint32_t)sync_sem + my_core_id,
            get_noc_addr(0, 0, (uint32_t)sync_sem + my_core_id)
        );
    }
}
```

---

### 7.4 性能优化检查清单

- [ ] **CB 大小**: 是否足够大以避免频繁同步，但又不会浪费 L1 空间
- [ ] **双缓冲**: 是否使用了双缓冲重叠计算和通信
- [ ] **批量操作**: 是否批量进行 reserve/push 和 wait/pop 操作
- [ ] **数据格式**: 是否选择了合适的数据格式（Bfp8_b vs Float16_b）
- [ ] **Tile 大小**: 是否根据实际数据选择最优 Tile 配置

---

## 附录 A: 完整示例代码

### A.1 矩阵乘法完整示例

```cpp
// ==================== Host 端代码 ====================
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

int main() {
    // 创建设备
    IDevice* device = CreateDevice(0);

    // 创建程序
    Program program;

    // 定义核心
    CoreCoord core(0, 0);

    // 配置参数
    uint32_t tile_size = 32 * 32 * 2;  // Float16_b
    uint32_t num_tiles = 4;
    uint32_t cb_size = num_tiles * tile_size;

    // 创建输入 CBs
    CircularBufferConfig cb_in0_config(
        cb_size,
        {{0, tt::DataFormat::Float16_b}}
    ).set_page_size(0, tile_size);
    CBHandle cb_in0 = CreateCircularBuffer(program, core, cb_in0_config);

    CircularBufferConfig cb_in1_config(
        cb_size,
        {{1, tt::DataFormat::Float16_b}}
    ).set_page_size(1, tile_size);
    CBHandle cb_in1 = CreateCircularBuffer(program, core, cb_in1_config);

    // 创建输出 CB
    CircularBufferConfig cb_out_config(
        cb_size,
        {{16, tt::DataFormat::Float16_b}}
    ).set_page_size(16, tile_size);
    CBHandle cb_out = CreateCircularBuffer(program, core, cb_out_config);

    // 创建 Kernels
    auto reader = CreateKernel(
        program,
        "reader_matmul.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    auto writer = CreateKernel(
        program,
        "writer_matmul.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    auto compute = CreateKernel(
        program,
        "compute_matmul.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false
        }
    );

    // 设置运行时参数
    SetRuntimeArgs(program, reader, core, {Mt, Kt, Nt, /* ... */});
    SetRuntimeArgs(program, writer, core, {Mt, Nt, /* ... */});
    SetRuntimeArgs(program, compute, core, {Mt, Kt, Nt});

    // 执行程序
    CommandQueue& cq = device->command_queue();
    EnqueueProgram(cq, program, false);
    Finish(cq);

    CloseDevice(device);
    return 0;
}

// ==================== Reader Kernel ====================
// reader_matmul.cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t tile_size = 2048;

    // 读取输入 A
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(mt * Kt + kt, in0_dram, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }

    // 读取输入 B
    for (uint32_t nt = 0; nt < Nt; ++nt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_reserve_back(cb_in1, 1);
            uint32_t l1_addr = get_write_ptr(cb_in1);
            noc_async_read_tile(kt * Nt + nt, in1_dram, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}

// ==================== Compute Kernel ====================
// compute_matmul.cpp
#include "compute_kernel_api/matmul.h"

void MAIN {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 16;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
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
}

// ==================== Writer Kernel ====================
// writer_matmul.cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t tile_size = 2048;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_out, 1);

            uint32_t l1_addr = get_read_ptr(cb_out);
            noc_async_write_tile(mt * Nt + nt, out_dram, l1_addr);
            noc_async_write_barrier();

            cb_pop_front(cb_out, 1);
        }
    }
}
```

---

## 附录 B: API 快速参考表

### Host 端 API

| 函数 | 用途 | 复杂度 |
|------|------|--------|
| `CreateCircularBuffer` | 创建 CB | 高 |
| `GetCircularBufferConfig` | 获取配置 | 低 |
| `UpdateCircularBufferTotalSize` | 更新大小 | 中 |
| `UpdateCircularBufferPageSize` | 更新页大小 | 中 |
| `UpdateDynamicCircularBufferAddress` | 更新地址 | 高 |

### Device 端 API

| 函数 | 用途 | 调用者 |
|------|------|--------|
| `cb_reserve_back` | 预留空间 | 生产者 |
| `cb_push_back` | 提交数据 | 生产者 |
| `cb_wait_front` | 等待数据 | 消费者 |
| `cb_pop_front` | 释放空间 | 消费者 |
| `get_write_ptr` | 获取写地址 | 生产者 |
| `get_read_ptr` | 获取读地址 | 消费者 |
| `cb_pages_available_at_front` | 查询可用页 | 消费者 |
| `cb_pages_reservable_at_back` | 查询可预留页 | 生产者 |

---

*文档结束*
