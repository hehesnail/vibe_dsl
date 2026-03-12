# Data Movement Kernel API 参考手册

> **版本**: TT-Metalium v0.55+
> **头文件**: `dataflow_api.h`
> **适用核心**: BRISC (RISC-V 0), NCRISC (RISC-V 1), ERISC (以太网核心)

---

## 目录

1. [核心 NOC 读写操作](#1-核心-noc-读写操作)
2. [单包操作](#2-单包操作)
3. [状态管理函数](#3-状态管理函数)
4. [多播操作](#4-多播操作)
5. [页面操作](#5-页面操作)
6. [分片操作](#6-分片操作)
7. [信号量操作](#7-信号量操作)
8. [屏障函数](#8-屏障函数)
9. [地址函数](#9-地址函数)
10. [核心坐标与参数访问](#10-核心坐标与参数访问)
11. [Circular Buffer 操作](#11-circular-buffer-操作)
12. [Tile 信息查询](#12-tile-信息查询)

---

## 1. 核心 NOC 读写操作

### 1.1 noc_async_read

非阻塞异步读取数据从 NoC 地址到本地 L1 内存。

```cpp
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true>
inline void noc_async_read(
    uint64_t src_noc_addr,      // 源 NoC 地址 (DRAM 或其他核心 L1)
    uint32_t dst_local_l1_addr, // 目标本地 L1 地址
    uint32_t size,              // 读取字节数
    uint8_t noc = noc_index     // NoC 索引 (0 或 1)
);
```

**参数说明**:
- `src_noc_addr`: 源地址，通过 `get_noc_addr()` 或 `get_dram_noc_addr()` 获取
- `dst_local_l1_addr`: 本地 L1 目标地址，通常通过 `get_write_ptr()` 获取
- `size`: 要读取的字节数
- `noc`: 使用的 NoC 网络 (0 或 1)

**返回值**: 无

**使用示例**:
```cpp
void kernel_main() {
    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_buffer = get_write_ptr(cb_id);

    // 从 DRAM 读取 2048 字节到 L1
    noc_async_read(dram_addr, l1_buffer, 2048);
    noc_async_read_barrier();  // 等待读取完成
}
```

---

### 1.2 noc_async_write

非阻塞异步写入数据从本地 L1 内存到 NoC 地址。

```cpp
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true, bool posted = false>
inline void noc_async_write(
    uint32_t src_local_l1_addr, // 源本地 L1 地址
    uint64_t dst_noc_addr,      // 目标 NoC 地址
    uint32_t size,              // 写入字节数
    uint8_t noc = noc_index     // NoC 索引
);
```

**参数说明**:
- `src_local_l1_addr`: 本地 L1 源地址
- `dst_noc_addr`: 目标 NoC 地址
- `size`: 要写入的字节数
- `posted`: 如果为 true，写入不等待确认 (更高性能但无完成保证)

**返回值**: 无

**使用示例**:
```cpp
void kernel_main() {
    uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, dst_addr);
    uint32_t l1_buffer = get_read_ptr(cb_id);

    // 写入 2048 字节到目标核心
    noc_async_write(l1_buffer, dst_noc_addr, 2048);
    noc_async_write_barrier();  // 等待写入完成
}
```

---

### 1.3 noc_async_read_barrier

等待所有挂起的 NoC 读取操作完成。

```cpp
void noc_async_read_barrier(uint8_t noc = noc_index);
```

**参数说明**:
- `noc`: 要等待的 NoC 索引

**使用示例**:
```cpp
// 批量读取多个 tiles
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_read(src_addrs[i], dst_addrs[i], tile_size);
}
noc_async_read_barrier();  // 等待所有读取完成
```

---

### 1.4 noc_async_write_barrier

等待所有挂起的 NoC 写入操作完成。

```cpp
FORCE_INLINE void noc_async_write_barrier(uint8_t noc = noc_index);
```

**使用示例**:
```cpp
// 批量写入
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_write(src_addrs[i], dst_addrs[i], tile_size);
}
noc_async_write_barrier();  // 确保所有写入完成
```

---

### 1.5 noc_async_full_barrier

等待所有挂起的 NoC 操作（读取和写入）完成。

```cpp
FORCE_INLINE void noc_async_full_barrier(uint8_t noc_idx = noc_index);
```

**使用场景**: 在需要确保所有数据传输完成的同步点使用。

**使用示例**:
```cpp
// 读写混合操作后完全同步
noc_async_read(src1, dst1, size1);
noc_async_write(src2, dst2, size2);
noc_async_full_barrier();  // 等待所有操作完成
```

---

## 2. 单包操作

单包操作用于传输小于或等于 NOC 最大突发大小 (NOC_MAX_BURST_SIZE) 的数据，具有更低的延迟。

### 2.1 noc_async_read_one_packet

单包异步读取，适用于小数据传输。

```cpp
template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_one_packet(
    uint64_t src_noc_addr,      // 源 NoC 地址
    uint32_t dst_local_l1_addr, // 目标本地 L1 地址
    uint32_t size,              // 读取字节数 (<= NOC_MAX_BURST_SIZE)
    uint8_t noc = noc_index     // NoC 索引
);
```

**使用示例**:
```cpp
// 读取单个 tile (假设 tile_size <= NOC_MAX_BURST_SIZE)
noc_async_read_one_packet(dram_addr, l1_buffer, tile_size);
noc_async_read_barrier();
```

---

### 2.2 noc_async_read_one_packet_set_state

设置单包读取的状态，用于后续多次读取相同大小的数据。

```cpp
template <bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_set_state(
    uint64_t src_noc_addr,  // 源 NoC 地址
    uint32_t size,          // 数据包大小
    const uint32_t vc = 0,  // 虚拟通道
    uint8_t noc = noc_index // NoC 索引
);
```

**使用场景**: 多次读取相同大小的数据时，预先设置状态可以提高性能。

---

### 2.3 noc_async_read_one_packet_with_state

使用预设状态执行单包读取。

```cpp
template <bool inc_num_issued = true, bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_with_state(
    uint32_t src_local_l1_addr, // 源本地地址
    uint32_t dst_local_l1_addr, // 目标本地地址
    const uint32_t vc = 0,      // 虚拟通道
    uint8_t noc = noc_index     // NoC 索引
);
```

**使用示例**:
```cpp
// 预先设置状态
noc_async_read_one_packet_set_state(base_dram_addr, tile_size);

// 多次使用状态进行读取
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_read_one_packet_with_state(
        base_dram_addr + i * tile_size,
        l1_buffer + i * tile_size
    );
}
noc_async_read_barrier();
```

---

### 2.4 noc_async_write_one_packet

单包异步写入。

```cpp
template <bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet(
    std::uint32_t src_local_l1_addr, // 源本地 L1 地址
    std::uint64_t dst_noc_addr,      // 目标 NoC 地址
    std::uint32_t size,              // 写入字节数
    uint8_t noc = noc_index          // NoC 索引
);
```

---

### 2.5 noc_async_write_one_packet_set_state

设置单包写入状态。

```cpp
FORCE_INLINE void noc_async_write_one_packet_set_state(
    std::uint64_t dst_noc_addr, // 目标 NoC 地址
    std::uint32_t size,         // 数据包大小
    uint8_t noc = noc_index     // NoC 索引
);
```

---

### 2.6 noc_async_write_one_packet_with_state

使用预设状态执行单包写入。

```cpp
FORCE_INLINE void noc_async_write_one_packet_with_state(
    std::uint32_t src_local_l1_addr, // 源本地地址
    std::uint64_t dst_noc_addr,      // 目标 NoC 地址
    uint8_t noc = noc_index          // NoC 索引
);
```

---

## 3. 状态管理函数

状态管理函数用于优化多次相同类型传输的性能。

### 3.1 noc_async_read_set_state

设置异步读取的状态。

```cpp
FORCE_INLINE void noc_async_read_set_state(
    uint64_t src_noc_addr,  // 源 NoC 地址
    uint8_t noc = noc_index // NoC 索引
);
```

**使用场景**: 当从同一源地址多次读取不同大小时使用。

---

### 3.2 noc_async_read_with_state

使用预设状态执行读取。

```cpp
template <bool inc_num_issued = true>
FORCE_INLINE void noc_async_read_with_state(
    uint32_t src_local_l1_addr, // 源本地地址
    uint32_t dst_local_l1_addr, // 目标本地地址
    uint32_t size,              // 读取字节数
    uint8_t noc = noc_index     // NoC 索引
);
```

**使用示例**:
```cpp
// 设置源状态
noc_async_read_set_state(dram_base_addr);

// 多次读取不同大小
noc_async_read_with_state(dram_base_addr, l1_buf1, size1);
noc_async_read_with_state(dram_base_addr + offset, l1_buf2, size2);
noc_async_read_barrier();
```

---

### 3.3 noc_async_write_set_state

设置异步写入的状态。

```cpp
FORCE_INLINE void noc_async_write_set_state(
    uint64_t dst_noc_addr,  // 目标 NoC 地址
    uint8_t noc = noc_index // NoC 索引
);
```

---

### 3.4 noc_async_write_with_state

使用预设状态执行写入。

```cpp
FORCE_INLINE void noc_async_write_with_state(
    uint32_t src_local_l1_addr, // 源本地地址
    uint32_t dst_local_l1_addr, // 目标本地地址
    uint32_t size,              // 写入字节数
    uint8_t noc = noc_index     // NoC 索引
);
```

---

### 3.5 noc_async_read_inc_num_issued

增加已发出的读取操作计数。

```cpp
FORCE_INLINE void noc_async_read_inc_num_issued(
    std::uint32_t num_issued_reads_inc, // 增加的计数
    uint8_t noc = noc_index             // NoC 索引
);
```

**使用场景**: 手动管理屏障计数时使用。

---

## 4. 多播操作

多播操作允许将数据同时发送到多个目标核心。

### 4.1 noc_async_write_multicast

将数据多播到多个目标核心。

```cpp
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr,     // 源本地 L1 地址
    uint64_t dst_noc_addr_multicast, // 多播目标地址 (通过 get_noc_multicast_addr 获取)
    uint32_t size,                   // 写入字节数
    uint32_t num_dests,              // 目标核心数量
    bool linked = false,             // 是否链接到前一个多播
    uint8_t noc = noc_index          // NoC 索引
);
```

**参数说明**:
- `dst_noc_addr_multicast`: 通过 `get_noc_multicast_addr()` 生成的多播地址
- `num_dests`: 接收数据的目标核心数量
- `linked`: 如果为 true，此操作链接到前一个多播操作

**使用示例**:
```cpp
void kernel_main() {
    uint32_t src_l1 = get_read_ptr(cb_id);

    // 多播到 8x8 核心网格
    uint64_t multicast_addr = get_noc_multicast_addr(
        0, 0,           // 起始核心 (x, y)
        7, 7,           // 结束核心 (x, y)
        dst_l1_addr     // 目标 L1 地址
    );

    noc_async_write_multicast(
        src_l1,
        multicast_addr,
        tile_size,
        64              // 8x8 = 64 个核心
    );
    noc_async_write_barrier();
}
```

---

### 4.2 noc_async_write_multicast_loopback_src

多播写入并回环到源核心。

```cpp
inline void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,     // 源本地 L1 地址
    std::uint64_t dst_noc_addr_multicast, // 多播目标地址
    std::uint32_t size,                   // 写入字节数
    std::uint32_t num_dests,              // 目标核心数量 (包含源核心)
    bool linked = false,                  // 是否链接
    uint8_t noc = noc_index               // NoC 索引
);
```

**使用场景**: 当源核心也需要接收数据副本时使用。

---

## 5. 页面操作

页面操作用于基于页面 ID 的内存访问，通常与地址生成器一起使用。

### 5.1 noc_async_read_page

基于页面 ID 的异步读取。

```cpp
template <typename AddrGen, bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,          // 页面 ID
    const AddrGen& addrgen,     // 地址生成器
    uint32_t dst_local_l1_addr, // 目标本地 L1 地址
    uint32_t offset = 0,        // 页面内偏移
    uint8_t noc = noc_index     // NoC 索引
);
```

**使用场景**: 使用 DRAM 或 L1 的地址生成器进行基于页面的访问。

**使用示例**:
```cpp
// 使用 DRAM 地址生成器
InterleavedAddrGen<true> dram_addr_gen;
dram_addr_gen.bank_base_address = dram_base;
dram_addr_gen.page_size = page_size;

// 读取页面 0
noc_async_read_page(0, dram_addr_gen, l1_buffer);
noc_async_read_barrier();
```

---

### 5.2 noc_async_write_page

基于页面 ID 的异步写入。

```cpp
template <typename AddrGen, bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id,          // 页面 ID
    const AddrGen& addrgen,     // 地址生成器
    uint32_t src_local_l1_addr, // 源本地 L1 地址
    uint32_t size = 0,          // 写入大小 (0 = 使用页面大小)
    uint32_t offset = 0,        // 页面内偏移
    uint8_t noc = noc_index     // NoC 索引
);
```

---

## 6. 分片操作

分片操作用于处理分片张量 (sharded tensors)。

### 6.1 noc_async_read_shard

从分片张量读取数据。

```cpp
template <typename DSpec>
FORCE_INLINE void noc_async_read_shard(
    const uint32_t shard_id,        // 分片 ID
    const TensorAccessor<DSpec>& s, // 张量访问器
    std::uint32_t dst_local_l1_addr, // 目标本地 L1 地址
    uint8_t noc = noc_index          // NoC 索引
);
```

**使用场景**: 处理 HEIGHT_SHARDED、WIDTH_SHARDED 或 BLOCK_SHARDED 布局的张量。

---

### 6.2 noc_async_write_shard

写入数据到分片张量。

```cpp
template <typename DSpec, bool posted = false>
FORCE_INLINE void noc_async_write_shard(
    const uint32_t shard_id,        // 分片 ID
    const TensorAccessor<DSpec>& s, // 张量访问器
    std::uint32_t src_local_l1_addr, // 源本地 L1 地址
    uint8_t noc = noc_index          // NoC 索引
);
```

---

## 7. 信号量操作

信号量用于核心间的同步。

### 7.1 noc_semaphore_set

设置本地信号量的值。

```cpp
FORCE_INLINE void noc_semaphore_set(
    volatile tt_l1_ptr uint32_t* sem_addr, // 信号量地址
    uint32_t val                           // 要设置的值
);
```

**使用示例**:
```cpp
uint32_t sem_addr = get_semaphore(0);
noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(sem_addr), 1);
```

---

### 7.2 noc_semaphore_inc

增加本地信号量的值。

```cpp
FORCE_INLINE void noc_semaphore_inc(
    volatile tt_l1_ptr uint32_t* sem_addr, // 信号量地址
    uint32_t val                           // 增加的值
);
```

---

### 7.3 noc_semaphore_wait

等待信号量达到指定值。

```cpp
FORCE_INLINE void noc_semaphore_wait(
    volatile tt_l1_ptr uint32_t* sem_addr, // 信号量地址
    uint32_t val                           // 等待的目标值
);
```

**使用示例**:
```cpp
// 生产者-消费者同步
volatile uint32_t* sem = reinterpret_cast<volatile uint32_t*>(get_semaphore(0));

// 消费者等待数据就绪
noc_semaphore_wait(sem, 1);

// 处理数据...

// 重置信号量
noc_semaphore_set(sem, 0);
```

---

### 7.4 noc_semaphore_wait_min

等待信号量达到或超过最小值。

```cpp
FORCE_INLINE void noc_semaphore_wait_min(
    volatile tt_l1_ptr uint32_t* sem_addr, // 信号量地址
    uint32_t val                           // 最小值
);
```

---

### 7.5 noc_semaphore_set_multicast

多播设置信号量到多个核心。

```cpp
inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr,      // 源信号量地址
    uint64_t dst_noc_addr_multicast,  // 多播目标地址
    uint32_t num_dests,               // 目标核心数量
    bool linked = false,              // 是否链接
    uint8_t noc = noc_index           // NoC 索引
);
```

**使用场景**: 同时向多个核心发送同步信号。

---

### 7.6 noc_semaphore_set_multicast_loopback_src

多播设置信号量并回环到源核心。

```cpp
inline void noc_semaphore_set_multicast_loopback_src(
    uint32_t src_local_l1_addr,      // 源信号量地址
    uint64_t dst_noc_addr_multicast,  // 多播目标地址
    uint32_t num_dests,               // 目标核心数量 (包含源)
    bool linked = false,              // 是否链接
    uint8_t noc = noc_index           // NoC 索引
);
```

---

### 7.7 noc_semaphore_set_remote

远程设置单个核心的信号量。

```cpp
inline void noc_semaphore_set_remote(
    std::uint32_t src_local_l1_addr, // 源本地值地址
    std::uint64_t dst_noc_addr,       // 目标 NoC 地址
    uint8_t noc = noc_index           // NoC 索引
);
```

**使用场景**: 向特定核心发送同步信号。

---

### 7.8 get_semaphore

获取信号量的本地 L1 地址。

```cpp
template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
FORCE_INLINE uint32_t get_semaphore(uint32_t semaphore_id);
```

**使用示例**:
```cpp
uint32_t sem0_addr = get_semaphore(0);
uint32_t sem1_addr = get_semaphore(1);
```

---

## 8. 屏障函数

### 8.1 noc_async_writes_flushed

等待所有写入被刷新到 NoC。

```cpp
FORCE_INLINE void noc_async_writes_flushed(uint8_t noc = noc_index);
```

**使用场景**: 确保写入操作已离开核心，但不一定到达目标。

---

### 8.2 noc_async_posted_writes_flushed

等待所有 posted 写入被刷新。

```cpp
FORCE_INLINE void noc_async_posted_writes_flushed(uint8_t noc = noc_index);
```

---

### 8.3 noc_async_atomic_barrier

等待所有原子操作完成。

```cpp
FORCE_INLINE void noc_async_atomic_barrier(uint8_t noc_idx = noc_index);
```

**使用场景**: 使用原子操作时确保顺序一致性。

---

## 9. 地址函数

### 9.1 get_noc_addr

获取指定核心 L1 地址的 NoC 地址。

```cpp
inline uint64_t get_noc_addr(
    uint32_t x,             // 目标核心 X 坐标
    uint32_t y,             // 目标核心 Y 坐标
    uint32_t local_addr     // 目标核心上的 L1 地址
);
```

**返回值**: 64 位 NoC 地址，可用于 `noc_async_read`/`noc_async_write`

**使用示例**:
```cpp
// 获取核心 (2, 3) 的 L1 地址 0x10000 的 NoC 地址
uint64_t noc_addr = get_noc_addr(2, 3, 0x10000);
noc_async_read(noc_addr, my_l1_buffer, size);
```

---

### 9.2 get_noc_addr_from_bank_id

从 bank ID 获取 NoC 地址。

```cpp
inline uint64_t get_noc_addr_from_bank_id(
    uint32_t bank_id,       // Bank ID
    uint32_t local_addr,    // 本地地址偏移
    bool is_dram            // 是否为 DRAM
);
```

---

### 9.3 get_noc_multicast_addr

获取多播地址。

```cpp
inline uint64_t get_noc_multicast_addr(
    uint32_t x_start,       // 起始 X 坐标
    uint32_t y_start,       // 起始 Y 坐标
    uint32_t x_end,         // 结束 X 坐标
    uint32_t y_end,         // 结束 Y 坐标
    uint32_t local_addr     // 目标 L1 地址
);
```

**返回值**: 64 位多播地址

**使用示例**:
```cpp
// 多播到 4x4 核心网格 (0,0) 到 (3,3)
uint64_t mcast_addr = get_noc_multicast_addr(0, 0, 3, 3, l1_dst_addr);

noc_async_write_multicast(
    src_l1_addr,
    mcast_addr,
    size,
    16  // 4x4 = 16 个核心
);
```

---

### 9.4 get_dram_noc_addr

获取 DRAM 地址的 NoC 表示。

```cpp
// 通常通过地址生成器或 buffer 对象获取
// 示例: noc_async_read_tile 内部使用
```

---

## 10. 核心坐标与参数访问

### 10.1 核心坐标函数

```cpp
// 获取绝对逻辑坐标
inline uint8_t get_absolute_logical_x();
inline uint8_t get_absolute_logical_y();

// 获取相对逻辑坐标 (在核心网格内)
inline uint8_t get_relative_logical_x();
inline uint8_t get_relative_logical_y();

// Quasar 架构特有
inline uint32_t get_num_threads();    // 获取线程数
inline uint32_t get_my_thread_id();   // 获取当前线程 ID
```

---

### 10.2 运行时参数访问

```cpp
// 获取参数地址
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx);

// 获取通用参数地址
template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx);

// 获取参数原始地址
static FORCE_INLINE uintptr_t get_arg_addr(int arg_idx);
static FORCE_INLINE uintptr_t get_common_arg_addr(int arg_idx);
```

**使用示例**:
```cpp
void kernel_main() {
    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t cb_id = get_arg_val<uint32_t>(2);
    // ...
}
```

---

## 11. Circular Buffer 操作

### 11.1 cb_push_back

将数据推入 Circular Buffer (生产者操作)。

```cpp
FORCE_INLINE void cb_push_back(
    const int32_t operand,  // CB ID
    const int32_t num_pages // 推送的页数
);
```

**使用示例**:
```cpp
// 预留空间并写入数据后推送
cb_reserve_back(cb_id, num_tiles);
uint32_t write_addr = get_write_ptr(cb_id);
// ... 写入数据到 write_addr ...
cb_push_back(cb_id, num_tiles);
```

---

### 11.2 cb_pop_front

从 Circular Buffer 弹出数据 (消费者操作)。

```cpp
FORCE_INLINE void cb_pop_front(
    int32_t operand,   // CB ID
    int32_t num_pages  // 弹出的页数
);
```

---

### 11.3 cb_reserve_back

在 CB 后端预留空间。

```cpp
FORCE_INLINE void cb_reserve_back(
    int32_t operand,   // CB ID
    int32_t num_pages  // 预留页数
);
```

---

### 11.4 cb_wait_front

等待 CB 前端有数据。

```cpp
FORCE_INLINE void cb_wait_front(
    int32_t operand,   // CB ID
    int32_t num_pages  // 等待页数
);
```

**使用示例**:
```cpp
// 消费者等待数据
cb_wait_front(cb_id, num_tiles);
uint32_t read_addr = get_read_ptr(cb_id);
// ... 从 read_addr 读取数据 ...
cb_pop_front(cb_id, num_tiles);
```

---

### 11.5 cb_pages_reservable_at_back

检查 CB 后端是否有足够空间可预留。

```cpp
FORCE_INLINE bool cb_pages_reservable_at_back(
    int32_t operand,   // CB ID
    int32_t num_pages  // 需要的页数
);
```

**返回值**: true 如果空间可用

---

### 11.6 cb_pages_available_at_front

检查 CB 前端是否有足够数据。

```cpp
FORCE_INLINE bool cb_pages_available_at_front(
    int32_t operand,   // CB ID
    int32_t num_pages  // 需要的页数
);
```

---

### 11.7 get_write_ptr

获取 CB 的写入地址。

```cpp
FORCE_INLINE uint32_t get_write_ptr(uint32_t operand);
```

**返回值**: CB 当前写入位置的 L1 地址

---

### 11.8 get_read_ptr

获取 CB 的读取地址。

```cpp
FORCE_INLINE uint32_t get_read_ptr(uint32_t operand);
```

**返回值**: CB 当前读取位置的 L1 地址

---

## 12. Tile 信息查询

### 12.1 get_tile_size

获取 Tile 大小。

```cpp
constexpr inline std::int32_t get_tile_size(const std::int32_t operand);
```

---

### 12.2 get_tile_hw

获取 Tile 高宽信息。

```cpp
constexpr inline uint32_t get_tile_hw(const std::int32_t operand);
```

---

### 12.3 get_tile_num_faces

获取 Tile 面数。

```cpp
constexpr inline uint32_t get_tile_num_faces(const std::int32_t operand);
```

---

### 12.4 get_dataformat

获取数据格式。

```cpp
constexpr inline DataFormat get_dataformat(const std::int32_t operand);
```

---

## 附录 A: 常见使用模式

### A.1 Reader Kernel 模式

```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t cb_id = get_arg_val<uint32_t>(2);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);

        noc_async_read(src_addr + i * tile_size, l1_addr, tile_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }
}
```

### A.2 Writer Kernel 模式

```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t cb_id = get_arg_val<uint32_t>(2);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_addr = get_read_ptr(cb_id);

        noc_async_write(l1_addr, dst_addr + i * tile_size, tile_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id, 1);
    }
}
```

### A.3 多播同步模式

```cpp
// 生产者核心
void kernel_main() {
    uint32_t sem_addr = get_semaphore(0);

    // 写入数据到多播目标
    noc_async_write_multicast(src, dst_mcast, size, num_dests);
    noc_async_write_barrier();

    // 通知所有消费者
    noc_semaphore_set_multicast(sem_addr, sem_mcast_addr, num_dests);
}

// 消费者核心
void kernel_main() {
    uint32_t sem_addr = get_semaphore(0);

    // 等待数据到达
    noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(sem_addr), 1);
    noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(sem_addr), 0);

    // 处理数据...
}
```

---

## 附录 B: 常量参考

| 常量 | 描述 | 典型值 |
|------|------|--------|
| `NOC_MAX_BURST_SIZE` | NoC 最大突发传输大小 | 8192 字节 |
| `NOC_UNICAST_WRITE_VC` | 单播写入虚拟通道 | 0 |
| `NOC_MULTICAST_WRITE_VC` | 多播写入虚拟通道 | 1 |
| `L1_ALIGNMENT` | L1 内存对齐要求 | 16 字节 |
| `NUM_CIRCULAR_BUFFERS` | 最大 CB 数量 | 32 |

---

*文档版本: 1.0 | 最后更新: 2026-03-12*
