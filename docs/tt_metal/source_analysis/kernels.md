# kernels/ 模块源码解析

## 1. 模块概述

TT-Metal的 `kernels/` 模块提供了内置的内核模板和示例代码，作为开发自定义内核的起点。该模块位于 `/tmp/tt-metal/tt_metal/kernels/`，包含两类核心内核：

- **计算内核 (Compute Kernels)**: 在Tenstorrent张量核心(TRISC)上执行数学运算
- **数据流内核 (Dataflow Kernels)**: 管理数据在DRAM、L1缓存和计算核心之间的传输

### 在系统中的位置

```
tt_metal/
├── kernels/              # 本模块 - 内核模板
│   ├── compute/          # 计算内核模板
│   └── dataflow/         # 数据流内核模板
├── hw/inc/api/           # 内核API头文件
│   ├── compute/          # 计算API
│   └── dataflow/         # 数据流API
└── impl/                 # 主机端实现
```

内核通过TT-Metal的编译系统编译为RISC-V代码，在Wormhole/Grayskull/Quasar芯片的专用核心上运行。

---

## 2. 目录结构

```
tt_metal/kernels/
├── compute/                          # 计算内核模板
│   ├── blank.cpp                     # 空白计算内核模板
│   ├── eltwise_binary.cpp            # 逐元素二元运算内核
│   └── eltwise_sfpu.cpp              # 逐元素SFPU运算内核
└── dataflow/                         # 数据流内核模板
    ├── blank.cpp                     # 空白数据流内核模板
    ├── reader_binary_diff_lengths.cpp # 双输入读取器（不同长度）
    ├── reader_unary.cpp              # 单输入读取器
    ├── writer_unary.cpp              # 单输出写入器（实验性API）
    └── writer_unary_1.cpp            # 单输出写入器（传统API）
```

---

## 3. 核心组件解析

### 3.1 Compute Kernels (计算内核)

计算内核在TRISC（Tensor RISC）核心上运行，分为三个协作线程：
- **UNPACK**: 从CB解包数据到SRC寄存器
- **MATH**: 在SRC寄存器上执行数学运算，结果写入DST寄存器
- **PACK**: 从DST寄存器打包数据到输出CB

#### 3.1.1 blank.cpp - 空白计算内核模板

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/compute/blank.cpp`

**功能描述**: 最简单的计算内核模板，作为自定义内核的起点。

```cpp
#include "api/compute/blank.h"

void kernel_main() {}
```

**关键分析**:
- 仅包含空的 `kernel_main()` 函数
- 包含 `blank.h` 头文件，该文件根据编译目标定义不同的入口点：
  - `TRISC_MATH`: 定义 `MAIN` 为 `math_main()`
  - `TRISC_PACK`: 定义 `MAIN` 为 `pack_main()`
  - `TRISC_UNPACK`: 定义 `MAIN` 为 `unpack_main()`

**使用场景**: 作为占位内核或开发新内核时的起点。

---

#### 3.1.2 eltwise_binary.cpp - 逐元素二元运算内核

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/compute/eltwise_binary.cpp`

**功能描述**: 执行逐元素二元运算（加、减、乘等），支持多种编译时配置选项。

**关键代码分析**:

```cpp
void kernel_main() {
    // 获取运行时参数
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // 每个核心的块数
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);  // 每个块的大小（tile数）
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);           // 是否累加到DST

    // 定义循环缓冲区索引
    constexpr auto cb_in0 = tt::CBIndex::c_0;    // 输入0
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // 输入1
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // 输出
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // 可选的累加输入

    // 初始化二元运算
    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

    // 主循环：处理每个块
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // 等待输入数据就绪
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        // 获取DST寄存器
        tile_regs_acquire();

        // 执行逐元素运算
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);
        }
        tile_regs_commit();

        // 打包结果到输出CB
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        // 释放输入，推送输出
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

**编译时配置宏**:

| 宏 | 功能 |
|----|------|
| `ELTWISE_OP_TYPE` | 定义二元运算类型（ADD/SUB/MUL等） |
| `ELTWISE_OP` | 实际运算函数（如 `add_tiles`） |
| `ELTWISE_OP_INIT` | 运算初始化函数 |
| `ELTWISE_DEST_REUSE_TYPE` | DST寄存器复用模式 |
| `DST_ACCUM_MODE` | 累加模式启用 |
| `SFPU_OP_CHAIN_0` | SFPU操作链 |
| `PACK_RELU` | 打包时应用ReLU激活 |
| `FULL_INIT` | 完整初始化模式 |

**编程模式**:
1. **参数获取**: 使用 `get_arg_val<T>(index)` 获取运行时参数
2. **CB管理**: 使用 `cb_wait_front()` 等待输入，`cb_reserve_back()` 预留输出空间
3. **DST寄存器**: 使用 `tile_regs_acquire/commit/wait/release` 管理DST生命周期
4. **运算执行**: 在循环中调用 `ELTWISE_OP` 执行逐元素运算
5. **结果打包**: 使用 `pack_tile()` 将结果写入输出CB
6. **CB同步**: 使用 `cb_pop_front()` 释放输入，`cb_push_back()` 推送输出

---

#### 3.1.3 eltwise_sfpu.cpp - 逐元素SFPU运算内核

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/compute/eltwise_sfpu.cpp`

**功能描述**: 执行逐元素SFPU（Special Function Processing Unit）运算，如激活函数、三角函数等。

**关键代码分析**:

```cpp
void kernel_main() {
    // 获取编译时参数（而非运行时参数）
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // 初始化SFPU
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // 逐个tile处理
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // 执行SFPU操作链
            #ifdef SFPU_OP_CHAIN_0
                SFPU_OP_CHAIN_0
            #endif

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);

            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
}
```

**与eltwise_binary.cpp的区别**:

| 特性 | eltwise_binary.cpp | eltwise_sfpu.cpp |
|------|-------------------|------------------|
| 参数获取 | `get_arg_val` (运行时) | `get_compile_time_arg_val` (编译时) |
| 输入数量 | 2个输入CB | 1个输入CB |
| 运算类型 | 二元运算 | 一元SFPU运算 |
| Tile处理 | 批量处理 | 逐个处理 |
| 初始化 | `binary_op_init_common` | `init_sfpu` |

**SFPU_OP_CHAIN_0 示例**:
```cpp
// 典型的SFPU操作链定义
#define SFPU_OP_CHAIN_0 \
    tanh_tile(0);       \
    relu_tile(0);       \
    sqrt_tile(0);
```

---

### 3.2 Dataflow Kernels (数据流内核)

数据流内核在BRISC（Bridge RISC）或NCRISC（NoC RISC）核心上运行，负责：
- 从DRAM读取数据到L1缓存
- 在L1缓存中管理循环缓冲区（Circular Buffer）
- 将计算结果写回DRAM

#### 3.2.1 blank.cpp - 空白数据流内核模板

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/dataflow/blank.cpp`

**功能描述**: 最简单的数据流内核模板。

```cpp
void kernel_main() {}
```

**使用场景**: 作为数据流内核开发的起点。

---

#### 3.2.2 reader_unary.cpp - 单输入读取器

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/dataflow/reader_unary.cpp`

**功能描述**: 从单一源（DRAM）读取数据到L1循环缓冲区。

**关键代码分析**:

```cpp
void kernel_main() {
    // 获取运行时参数
    uint32_t src_addr  = get_arg_val<uint32_t>(0);   // 源地址
    uint32_t bank_id = get_arg_val<uint32_t>(1);     // 银行ID
    uint32_t num_tiles = get_arg_val<uint32_t>(2);   // tile数量

    constexpr uint32_t cb_id_in0 = 0;  // 输入CB索引
    constexpr uint32_t ublock_size_tiles = 1;  // ublock大小（tile数）

    // 计算ublock字节大小
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

    // 主循环：读取每个ublock
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        // 计算NOC地址
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

        // 预留CB空间
        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        // 发起异步读取
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        // 等待读取完成
        noc_async_read_barrier();

        // 推送数据到CB（对计算核心可见）
        cb_push_back(cb_id_in0, ublock_size_tiles);

        // 更新源地址
        src_addr += ublock_size_bytes;
    }
}
```

**关键API**:

| API | 功能 |
|-----|------|
| `get_arg_val<T>(idx)` | 获取运行时参数 |
| `get_noc_addr_from_bank_id<true>(bank_id, addr)` | 从银行ID获取NOC地址 |
| `get_tile_size(cb_id)` | 获取tile大小（字节） |
| `cb_reserve_back(cb_id, num_tiles)` | 预留CB尾部空间 |
| `get_write_ptr(cb_id)` | 获取CB写指针 |
| `noc_async_read(noc_addr, l1_addr, size)` | 发起异步NOC读取 |
| `noc_async_read_barrier()` | 等待所有读取完成 |
| `cb_push_back(cb_id, num_tiles)` | 推送数据到CB |

---

#### 3.2.3 reader_binary_diff_lengths.cpp - 双输入读取器

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp`

**功能描述**: 从两个不同长度的源读取数据到两个CB，支持长度不匹配的情况。

**关键代码分析**:

```cpp
void kernel_main() {
    // 源0参数
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src0_num_tiles  = get_arg_val<uint32_t>(2);

    // 源1参数
    uint32_t src1_addr  = get_arg_val<uint32_t>(3);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(4);
    uint32_t src1_num_tiles  = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // 计算ublock大小
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    // 确定循环次数（取较大值）
    uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;

    // 读取ublocks到CB0/CB1
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        // 条件读取源0
        if (i < src0_num_tiles) {
            uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, ublock_size_tiles);
            src0_addr += ublock_size_bytes_0;
        }

        // 条件读取源1
        if (i < src1_num_tiles) {
            uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);
            cb_reserve_back(cb_id_in1, ublock_size_tiles);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, ublock_size_tiles);
            src1_addr += ublock_size_bytes_1;
        }
    }
}
```

**设计特点**:
- 支持两个输入长度不同的情况
- 使用条件判断 `if (i < srcX_num_tiles)` 处理长度不匹配
- 较长的输入会继续读取，较短的输入停止读取
- 适用于广播（broadcast）场景

---

#### 3.2.4 writer_unary_1.cpp - 单输出写入器（传统API）

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/dataflow/writer_unary_1.cpp`

**功能描述**: 将计算结果从CB写入DRAM，使用传统C风格API。

**关键代码分析**:

```cpp
void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_x = get_arg_val<uint32_t>(1);  // 目标X坐标
    uint32_t dst_y = get_arg_val<uint32_t>(2);  // 目标Y坐标
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;  // 输出CB

    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        // 计算目标NOC地址（使用坐标）
        uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, dst_addr);

        // 等待CB中有数据
        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        // 发起异步写入
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        // 等待写入完成
        noc_async_write_barrier();

        // 释放CB中的数据
        cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
```

**关键API**:

| API | 功能 |
|-----|------|
| `get_noc_addr(x, y, addr)` | 从坐标获取NOC地址 |
| `cb_wait_front(cb_id, num_tiles)` | 等待CB中有足够数据 |
| `get_read_ptr(cb_id)` | 获取CB读指针 |
| `noc_async_write(l1_addr, noc_addr, size)` | 发起异步NOC写入 |
| `noc_async_write_barrier()` | 等待所有写入完成 |
| `cb_pop_front(cb_id, num_tiles)` | 从CB头部弹出数据 |

---

#### 3.2.5 writer_unary.cpp - 单输出写入器（实验性API）

**文件路径**: `/tmp/tt-metal/tt_metal/kernels/dataflow/writer_unary.cpp`

**功能描述**: 使用实验性的C++对象封装API实现相同功能。

**关键代码分析**:

```cpp
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    // 创建实验性API对象
    experimental::CircularBuffer cb(cb_id_out0);
    experimental::Noc noc;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        // 使用对象方法
        cb.wait_front(ublock_size_tiles);

        // 使用现代C++ API进行异步写入
        noc.async_write(
            cb,
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dst_addr});

        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
```

**API对比**:

| 特性 | writer_unary_1.cpp (传统) | writer_unary.cpp (实验性) |
|------|--------------------------|--------------------------|
| 风格 | C风格函数调用 | C++对象封装 |
| CB操作 | `cb_wait_front()` / `cb_pop_front()` | `cb.wait_front()` / `cb.pop_front()` |
| NOC操作 | `noc_async_write()` | `noc.async_write()` |
| 地址生成 | `get_noc_addr()` | `AllocatorBank` 类型标签 |
| 类型安全 | 较低 | 较高（模板类型检查） |

---

## 4. 内核编程模式

### 4.1 计算内核编程模式

```
┌─────────────────────────────────────────────────────────────┐
│                    Compute Kernel Pattern                   │
├─────────────────────────────────────────────────────────────┤
│ 1. 获取参数                                                 │
│    get_arg_val<uint32_t>(idx)                               │
│                                                             │
│ 2. 定义CB索引                                               │
│    constexpr auto cb_in = tt::CBIndex::c_0;                 │
│    constexpr auto cb_out = tt::CBIndex::c_16;               │
│                                                             │
│ 3. 初始化运算                                               │
│    binary_op_init_common(cb_in0, cb_in1, cb_out);           │
│    binary_tiles_init<full_init, OP_TYPE>(cb_in0, cb_in1);   │
│                                                             │
│ 4. 主循环（块级别）                                         │
│    for (block = 0; block < block_cnt; block++) {            │
│                                                             │
│        // 等待输入就绪                                       │
│        cb_wait_front(cb_in, block_size);                    │
│                                                             │
│        // 预留输出空间                                       │
│        cb_reserve_back(cb_out, block_size);                 │
│                                                             │
│        // 获取DST寄存器                                      │
│        tile_regs_acquire();                                 │
│                                                             │
│        // 执行运算（tile级别）                               │
│        for (i = 0; i < block_size; i++) {                   │
│            ELTWISE_OP(cb_in0, cb_in1, i, i, i);             │
│        }                                                    │
│                                                             │
│        tile_regs_commit();                                  │
│                                                             │
│        // 打包结果                                           │
│        tile_regs_wait();                                    │
│        for (i = 0; i < block_size; i++) {                   │
│            pack_tile(i, cb_out);                            │
│        }                                                    │
│        tile_regs_release();                                 │
│                                                             │
│        // 释放输入，推送输出                                 │
│        cb_pop_front(cb_in, block_size);                     │
│        cb_push_back(cb_out, block_size);                    │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 数据流内核编程模式

```
┌─────────────────────────────────────────────────────────────┐
│                   Dataflow Kernel Pattern                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Reader Pattern (DRAM -> CB):                               │
│  ───────────────────────────                                │
│  for (i = 0; i < num_tiles; i += ublock_size) {             │
│      // 计算源地址                                           │
│      noc_addr = get_noc_addr_from_bank_id(bank_id, addr);   │
│                                                             │
│      // 预留CB空间                                           │
│      cb_reserve_back(cb_id, ublock_size);                   │
│      l1_addr = get_write_ptr(cb_id);                        │
│                                                             │
│      // 异步读取                                             │
│      noc_async_read(noc_addr, l1_addr, size);               │
│      noc_async_read_barrier();                              │
│                                                             │
│      // 推送数据                                             │
│      cb_push_back(cb_id, ublock_size);                      │
│      addr += size;                                          │
│  }                                                          │
│                                                             │
│  Writer Pattern (CB -> DRAM):                               │
│  ───────────────────────────                                │
│  for (i = 0; i < num_tiles; i += ublock_size) {             │
│      // 等待数据就绪                                         │
│      cb_wait_front(cb_id, ublock_size);                     │
│      l1_addr = get_read_ptr(cb_id);                         │
│                                                             │
│      // 计算目标地址                                         │
│      noc_addr = get_noc_addr(x, y, addr);                   │
│                                                             │
│      // 异步写入                                             │
│      noc_async_write(l1_addr, noc_addr, size);              │
│      noc_async_write_barrier();                             │
│                                                             │
│      // 释放数据                                             │
│      cb_pop_front(cb_id, ublock_size);                      │
│      addr += size;                                          │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 设计模式与实现技巧

### 5.1 编译时与运行时参数

TT-Metal支持两种参数传递方式：

```cpp
// 运行时参数 - 每个核心可以不同
uint32_t block_cnt = get_arg_val<uint32_t>(0);

// 编译时参数 - 所有核心相同，可优化
uint32_t block_dim = get_compile_time_arg_val(1);
```

**选择原则**:
- 编译时参数：值在编译时已知，可用于模板参数、数组大小等
- 运行时参数：值在运行时确定，如地址、tile数量等

### 5.2 条件编译与模板特化

```cpp
// 根据运算类型选择不同实现
template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init(...) {
    if constexpr (eltwise_binary_type == ELWMUL) {
        // 乘法使用高精度
        MATH((llk_math_eltwise_binary_init<..., MATH_FIDELITY>(...)));
    } else {
        // 加减使用低精度（更快）
        MATH((llk_math_eltwise_binary_init<..., MathFidelity::LoFi>(...)));
    }
}
```

### 5.3 TRISC线程协作

```cpp
// UNPACK线程执行
UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));

// MATH线程执行
MATH((llk_math_eltwise_binary<ELWMUL, ...>(icb0, icb1, idst, true)));

// PACK线程执行
PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
```

### 5.4 异步操作与屏障

```cpp
// 发起异步读取（立即返回）
noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

// 执行其他工作（如果有）...

// 等待读取完成（阻塞）
noc_async_read_barrier();
```

**优化技巧**: 通过批量处理隐藏延迟，在处理当前块时预取下一块。

### 5.5 循环缓冲区同步模式

```cpp
// 生产者（Reader）模式
cb_reserve_back(cb_id, num_tiles);  // 等待有空闲空间
// 写入数据
cb_push_back(cb_id, num_tiles);     // 通知消费者数据就绪

// 消费者（Compute/Writer）模式
cb_wait_front(cb_id, num_tiles);    // 等待有数据
// 读取数据
cb_pop_front(cb_id, num_tiles);     // 释放空间给生产者
```

---

## 6. 源码注释摘录

### 6.1 eltwise_binary.h 中的API文档

```cpp
/**
 * Init function for all binary ops
 * Followed by the specific init required with an opcode (binrary_op_specific_init)
 *
 * | Argument       | Description                                                   | Type     | Valid Range                | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                    | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                    | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16 | True     |
 */
```

### 6.2 tile_move_copy.h 中的copy_tile文档

```cpp
/**
 * Copies a single tile from the specified input CB and writes the result to
 * DST at a specified index. The function will employ unpacker to first unpack into SRC
 * registers and then perform move into DST registers, at a specified index.
 * For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
 * previously called to ensure that at least some number n>0 of tiles are available
 * in the input CB. The CB index 0 then references the first tile in the received section of the CB,
 * up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
 * acquire_dst call. This call is blocking and is only available on the compute
 * engine.
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | True     |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | True     |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | True     |
 */
```

### 6.3 dataflow_api.h 中的坐标获取API

```cpp
/**
 * Returns the absolute logical X coordinate value that this kernel is running on.
 * The absolute coordinate is the one relative to the origin of the physical grid.
 *
 * Return value: X coordinate value.
 */
inline uint8_t get_absolute_logical_x() {
    extern uint8_t my_logical_x_;  // Set in FW
    return my_logical_x_;
}
```

### 6.4 内核启动配置 (blank.h)

```cpp
#ifdef TRISC_MATH
#define MAIN math_main()
#endif

#ifdef TRISC_PACK
#define MAIN pack_main()
#endif

#ifdef TRISC_UNPACK
#define MAIN unpack_main()
#endif
```

---

## 7. 文件清单

| 文件路径 | 类型 | 描述 |
|----------|------|------|
| `/tmp/tt-metal/tt_metal/kernels/compute/blank.cpp` | 模板 | 空白计算内核 |
| `/tmp/tt-metal/tt_metal/kernels/compute/eltwise_binary.cpp` | 示例 | 逐元素二元运算 |
| `/tmp/tt-metal/tt_metal/kernels/compute/eltwise_sfpu.cpp` | 示例 | 逐元素SFPU运算 |
| `/tmp/tt-metal/tt_metal/kernels/dataflow/blank.cpp` | 模板 | 空白数据流内核 |
| `/tmp/tt-metal/tt_metal/kernels/dataflow/reader_unary.cpp` | 示例 | 单输入读取器 |
| `/tmp/tt-metal/tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp` | 示例 | 双输入读取器 |
| `/tmp/tt-metal/tt_metal/kernels/dataflow/writer_unary_1.cpp` | 示例 | 单输出写入器（传统API） |
| `/tmp/tt-metal/tt_metal/kernels/dataflow/writer_unary.cpp` | 示例 | 单输出写入器（实验性API） |

---

## 8. 参考链接

- 相关API头文件:
  - `/tmp/tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h`
  - `/tmp/tt-metal/tt_metal/hw/inc/api/compute/tile_move_copy.h`
  - `/tmp/tt-metal/tt_metal/hw/inc/api/compute/common.h`
  - `/tmp/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h`
  - `/tmp/tt-metal/tt_metal/hw/inc/experimental/circular_buffer.h`
  - `/tmp/tt-metal/tt_metal/hw/inc/experimental/endpoints.h`
