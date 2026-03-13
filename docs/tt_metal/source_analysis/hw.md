# hw/ 模块源码解析

## 1. 模块概述

### 1.1 模块职责

`hw/` 模块是 TT-Metal 框架的底层硬件抽象层，负责：

- **硬件寄存器定义**：定义 Tensix 核心、NOC（Network on Chip）、以太网等的寄存器映射
- **固件代码**：BRISC、NCRISC、TRISC、ERISC 等 RISC-V 处理器的固件实现
- **计算内核 API**：提供 LLK（Low Level Kernel）API 用于计算操作
- **工具链支持**：链接器脚本、启动代码、编译配置

### 1.2 支持的硬件架构

| 架构 | 代号 | 特点 |
|------|------|------|
| Blackhole | tt-bh | 最新架构，1.5MB L1，支持 64 位 CB mask |
| Wormhole B0 | tt-wh | 前代架构，NCRISC 使用 IRAM 约束 |
| Quasar | tt-qsr | 下一代架构，8 个 DM 核心，4 个 TRISC 核心 |

### 1.3 与其他模块的交互

```
hw/ 模块交互图
================

hw/inc/api/         --> 提供给 kernels/ 的数据流和计算 API
hw/firmware/        --> 由 impl/ 加载到设备
hw/toolchain/       --> 被 jit_build/ 用于编译内核
hw/ckernels/        --> 被 kernels/ 调用
```

## 2. 目录结构

```
hw/
├── CMakeLists.txt          # 构建配置，SFPI 工具链下载/编译
├── sources.cmake           # JIT API 头文件列表
├── Makefile-runtime        # 运行时 Makefile
├── ckernels/               # 计算内核实现
│   ├── blackhole/          # Blackhole 架构特定实现
│   ├── quasar/             # Quasar 架构特定实现
│   └── wormhole_b0/        # Wormhole B0 架构特定实现
├── firmware/               # 固件源码
│   └── src/
│       ├── tt-1xx/         # TT-1xx 系列 (Blackhole, Wormhole)
│       └── tt-2xx/         # TT-2xx 系列 (Quasar)
├── inc/                    # 头文件
│   ├── api/                # 公共 API (数据流、计算、调试)
│   ├── experimental/       # 实验性功能
│   ├── hostdev/            # 主机-设备共享定义
│   └── internal/           # 内部实现头文件
└── toolchain/              # 编译工具链配置
    ├── *.ld                # 链接器脚本
    ├── *.S                 # 汇编启动代码
    └── substitutes.cpp     # C++ 运行时替代函数
```

## 3. 核心组件解析

### 3.1 ckernels/ - 计算内核

#### 3.1.1 架构特定目录结构

每个架构目录包含：

```
ckernels/<arch>/metal/
├── common/
│   └── chlkc_list.h        # 计算内核列表和入口点
├── llk_api/                # Low Level Kernel API
│   ├── llk_math_*.h        # 数学运算 API
│   ├── llk_pack_api.h      # Pack 操作 API
│   ├── llk_unpack_*.h      # Unpack 操作 API
│   └── llk_sfpu/           # SFPU (Special Function Unit) 操作
└── llk_io/                 # IO 操作接口
    ├── llk_io.h
    ├── llk_io_pack.h
    └── llk_io_unpack.h
```

#### 3.1.2 chlkc_list.h - 计算内核入口

**文件路径**: `/tmp/tt-metal/tt_metal/hw/ckernels/blackhole/metal/common/chlkc_list.h`

```cpp
// 计算内核入口函数
uint run_kernel() {
#ifdef UCK_CHLKC_MATH
    zeroacc();
    chlkc_math::math_main();    // 数学运算主函数
#endif

#ifdef UCK_CHLKC_PACK
    chlkc_pack::pack_main();    // Pack 操作主函数
#endif

#ifdef UCK_CHLKC_UNPACK
    zerosrc();
    chlkc_unpack::unpack_main(); // Unpack 操作主函数
#endif
    return 0;
}
```

关键宏定义：
- `UCK_CHLKC_MATH`：编译数学内核
- `UCK_CHLKC_PACK`：编译 Pack 内核
- `UCK_CHLKC_UNPACK`：编译 Unpack 内核

#### 3.1.3 LLK Math API

**文件路径**: `/tmp/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_common_api.h`

核心功能：

```cpp
// 等待目标寄存器可用
template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand);

// 等待目标寄存器可用
inline void llk_math_wait_for_dest_available();

// 目标区域完成
template <bool is_fp32_dest_acc_en>
inline void llk_math_dest_section_done();

// 数据格式重新配置
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand);

// 获取/清除特殊值标志
inline std::uint32_t llk_math_get_compute_special_value_flags();
inline void llk_math_clear_compute_special_value_flags();
```

#### 3.1.4 LLK Pack API

**文件路径**: `/tmp/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h`

核心功能：

```cpp
// Pack MOP 配置
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void llk_pack_mop_config(const uint32_t output, std::uint32_t num_tiles = 1);

// Pack 初始化
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void llk_pack_init(const std::uint32_t pack_output = 16, std::uint32_t num_tiles = 1);

// 执行 Pack 操作
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, bool untilize = false>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0);

// Pack untilize 操作
template <std::uint32_t block_ct_dim = 8, ...>
inline void llk_pack_untilize(std::uint32_t block_rt_dim, std::uint32_t output, ...);

// Pack rows 操作
inline void llk_pack_rows_init(const std::uint32_t num_rows);
inline void llk_pack_rows(const std::uint32_t dst_index, const std::uint32_t output, ...);
```

### 3.2 firmware/ - 固件

#### 3.2.1 处理器类型

TT-Metal 支持多种 RISC-V 处理器：

| 处理器 | 角色 | 描述 |
|--------|------|------|
| BRISC | 主控制 | 协调内核执行，管理 NOC |
| NCRISC | 数据移动 | NOC 数据传输，IRAM 约束（Wormhole） |
| TRISC0 | 计算 | Unpack 操作 |
| TRISC1 | 计算 | 数学运算 |
| TRISC2 | 计算 | Pack 操作 |
| ERISC | 以太网 | 以太网数据包处理 |
| AERISC | 主动以太网 | 主动以太网核心 |

#### 3.2.2 BRISC 固件

**文件路径**: `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc`

核心流程：

```cpp
int main() {
    configure_csr();
    WAYPOINT("I");  // 初始化路标

    do_crt1((uint32_t*)MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH);
    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    mailboxes->launch_msg_rd_ptr = 0;
    risc_init();
    device_setup();  // 设备初始化

    deassert_ncrisc_trisc();  // 释放 NCRISC/TRISC
    wait_ncrisc_trisc();      // 等待初始化完成

    while (1) {
        // 等待 GO 信号
        while (go_message_signal != RUN_MSG_GO) {
            invalidate_l1_cache();
        }

        // 执行内核
        uint32_t enables = launch_msg_address->kernel_config.enables;
        run_triscs(enables);

        // 运行 BRISC 内核
        if (enables & (1u << TensixProcessorTypes::DM0)) {
            auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        }

        wait_ncrisc_trisc();

        // 通知调度器完成
        notify_dispatch_core_done(dispatch_addr, noc_index);
    }
}
```

关键函数：

```cpp
// 设备设置
void device_setup() {
    // 设置指令缓冲区
    instrn_buf[0] = core.instrn_buf_base(0);
    instrn_buf[1] = core.instrn_buf_base(1);
    instrn_buf[2] = core.instrn_buf_base(2);

    // 启用时钟门控
    WRITE_REG(RISCV_TDMA_REG_CLK_GATE_EN, 0x3f);

    // 配置 NOC
    noc_set_active_instance(0);
    uint32_t niu_cfg0 = noc_get_cfg_reg(NIU_CFG_0);
    noc_set_cfg_reg(NIU_CFG_0, niu_cfg0 | 0x1);

    // 设置复位地址
    set_deassert_addresses();

    // 清零内存区域
    wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE);

    // 使能 ECC 清理器
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Enable_RMW, 1);

    // 初始化信号量
    core.initialize_tensix_semaphores(instrn_buf[0]);
}
```

#### 3.2.3 NCRISC 固件

**文件路径**: `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc`

特点：
- Wormhole 上 NCRISC 内核受 IRAM 约束（16KB）
- 使用 DMA 从 L1 复制代码到 IRAM

```cpp
#if defined(ARCH_WORMHOLE)
void l1_to_ncrisc_iram_copy(uint32_t src_addr, uint16_t size, uint32_t address_offset = 0) {
    // 使用 tensix DMA 从 L1 复制到 IRAM
    tdma_xmov(TDMA_MOVER0, src_addr, MEM_MOVER_VIEW_IRAM_BASE_ADDR + address_offset, size, XMOV_L1_TO_L0);
}
#endif

int main(int argc, char* argv[]) {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t tt_l1_ptr*)MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    risc_init();
    signal_ncrisc_completion();

    while (1) {
        wait_for_brisc_notification();

        uint32_t kernel_config_base = firmware_config_init(...);

#if defined(ARCH_WORMHOLE)
        // 复制内核到 IRAM
        l1_to_ncrisc_iram_copy(kernel_lma >> 4, launch_msg->kernel_config.ncrisc_kernel_size16, 0);
        l1_to_ncrisc_iram_copy_wait();
#endif

        // 设置 CB 接口
        setup_local_cb_read_write_interfaces<true, true, false, false>(cb_l1_base, 0, local_cb_mask_low);

        // 执行内核
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();

        signal_ncrisc_completion();
    }
}
```

#### 3.2.4 TRISC 固件

**文件路径**: `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc`

特点：
- 三个 TRISC 核心分别运行 Unpack、Math、Pack
- 通过 `COMPILE_FOR_TRISC` 宏区分（0=Unpack, 1=Math, 2=Pack）

```cpp
namespace ckernel {
    const uint8_t thread_id = COMPILE_FOR_TRISC;  // 0, 1, or 2

    volatile tt_l1_ptr uint8_t* const trisc_run =
        &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))
             ->subordinate_sync.map[COMPILE_FOR_TRISC + 1];
}

int main(int argc, char* argv[]) {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t tt_l1_ptr*)PREPROCESSOR_EXPAND(MEM_TRISC, COMPILE_FOR_TRISC, _INIT_LOCAL_L1_BASE_SCRATCH));

    // 初始化 GPRs
    for (int i = 0; i < 64; i++) {
        regfile[i] = 0;
    }

    // 初始化 PRNG 种子
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    cfg[PRNG_SEED_Seed_Val_ADDR32] = 0;

    *trisc_run = RUN_SYNC_MSG_DONE;

    while (1) {
        while (*trisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }

        // 设置 CB 接口
        setup_local_cb_read_write_interfaces<cb_init_read, cb_init_write, cb_init_write, false>(
            cb_l1_base, 0, local_cb_mask_low);

        // 执行内核
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();

        tensix_sync();
        *trisc_run = RUN_SYNC_MSG_DONE;
    }
}
```

### 3.3 inc/ - 硬件抽象层头文件

#### 3.3.1 内存映射定义

**Blackhole 内存映射** (`/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`):

```cpp
// L1 内存
#define MEM_L1_BASE 0x0
#define MEM_L1_SIZE (1536 * 1024)  // 1.5MB

// 本地内存（处理器私有）
#define MEM_LOCAL_BASE 0xFFB00000
#define MEM_BRISC_LOCAL_SIZE (8 * 1024)
#define MEM_NCRISC_LOCAL_SIZE (8 * 1024)
#define MEM_TRISC_LOCAL_SIZE (4 * 1024)

// 固件大小
#define MEM_BRISC_FIRMWARE_SIZE (6 * 1024 + 2560)
#define MEM_NCRISC_FIRMWARE_SIZE 2560
#define MEM_TRISC0_FIRMWARE_SIZE 2560

// 内核大小（Blackhole 无 IRAM 约束）
#define MEM_MAX_KERNEL_SIZE (1497 * 1024)
#define MEM_BRISC_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_NCRISC_KERNEL_SIZE MEM_MAX_KERNEL_SIZE

// Mailbox 区域
#define MEM_MAILBOX_BASE 96
#define MEM_MAILBOX_SIZE 12896

// 系统保留区域结束
#define MEM_MAP_END (MEM_PACKET_HEADER_POOL_BASE + MEM_PACKET_HEADER_POOL_SIZE)
```

**Wormhole 内存映射差异**:

```cpp
#define MEM_L1_SIZE (1464 * 1024)  // 1.43MB

// NCRISC IRAM 约束
#define NCRISC_HAS_IRAM 1
#define MEM_NCRISC_IRAM_BASE 0xFFC00000
#define MEM_NCRISC_IRAM_SIZE (16 * 1024)
#define MEM_NCRISC_KERNEL_SIZE MEM_NCRISC_IRAM_SIZE  // 限制为 16KB
```

**Quasar 内存映射** (`/tmp/tt-metal/tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h`):

```cpp
#define MEM_L1_SIZE (4 * 1024 * 1024)  // 4MB
#define MEM_L1_UNCACHED_BASE (MEM_L1_BASE + MEM_L1_SIZE)  // 上 4MB 绕过缓存

// 8 个 DM 核心
#define NUM_DM_CORES 8
#define NUM_TRISC_CORES 4  // 4 个 TRISC

// 固件/内核基址
#define MEM_DM_FIRMWARE_BASE (MEM_LLK_DEBUG_BASE + MEM_LLK_DEBUG_SIZE)
#define MEM_TRISC0_FIRMWARE_BASE (MEM_DM_FIRMWARE_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM_KERNEL_BASE (MEM_DM_LOCAL_BASE + MEM_DM_LOCAL_SIZE * NUM_DM_CORES)
```

#### 3.3.2 Tensix 寄存器定义

**文件路径**: `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h`

```cpp
// 寄存器文件基址
#define REGFILE_BASE 0xFFE00000  // 64 个 32 位寄存器

// 指令缓冲区
#define INSTRN_BUF_BASE 0xFFE40000
#define INSTRN_BUF_STRIDE 0x00010000

// PC 缓冲区
#define PC_BUF_BASE 0xFFE80000
#define PC1_BUF_BASE 0xFFE90000
#define PC2_BUF_BASE 0xFFEA0000

// Mailbox
#define TENSIX_MAILBOX0_BASE 0xFFEC0000  // Brisc
#define TENSIX_MAILBOX1_BASE 0xFFEC1000  // Trisc0
#define TENSIX_MAILBOX2_BASE 0xFFEC2000  // Trisc1
#define TENSIX_MAILBOX3_BASE 0xFFEC3000  // Trisc2

// 配置寄存器
#define TENSIX_CFG_BASE 0xFFEF0000

// TDMA 寄存器
#define RISCV_TDMA_REGS_START_ADDR 0xFFB11000
#define RISCV_TDMA_REG_XMOV_SRC_ADDR 0xFFB11000
#define RISCV_TDMA_REG_XMOV_DST_ADDR 0xFFB11004
#define RISCV_TDMA_REG_XMOV_SIZE 0xFFB11008
#define RISCV_TDMA_REG_PACKED_SIZE 0xFFB11018

// 调试寄存器
#define RISCV_DEBUG_REGS_START_ADDR 0xFFB12000
#define RISCV_DEBUG_REG_WALL_CLOCK_L (RISCV_DEBUG_REGS_START_ADDR | 0x1F0)
#define RISCV_DEBUG_REG_SOFT_RESET_0 (RISCV_DEBUG_REGS_START_ADDR | 0x1B0)
```

#### 3.3.3 NOC 寄存器定义

**文件路径**: `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`

```cpp
// NOC 寄存器基址
#define NOC_REG_SPACE_START_ADDR 0xFF000000
#define NOC_REGS_START_ADDR 0xFFB20000
#define NOC_CMD_BUF_OFFSET 0x00000800
#define NOC_INSTANCE_OFFSET 0x00010000

// NIU 主接口控制寄存器
#define NOC_TARG_ADDR_LO (NOC_REGS_START_ADDR + 0x0)
#define NOC_TARG_ADDR_MID (NOC_REGS_START_ADDR + 0x4)
#define NOC_TARG_ADDR_HI (NOC_REGS_START_ADDR + 0x8)
#define NOC_RET_ADDR_LO (NOC_REGS_START_ADDR + 0xC)
#define NOC_RET_ADDR_MID (NOC_REGS_START_ADDR + 0x10)
#define NOC_RET_ADDR_HI (NOC_REGS_START_ADDR + 0x14)
#define NOC_PACKET_TAG (NOC_REGS_START_ADDR + 0x18)
#define NOC_CTRL (NOC_REGS_START_ADDR + 0x1C)
#define NOC_AT_LEN_BE (NOC_REGS_START_ADDR + 0x20)
#define NOC_AT_DATA (NOC_REGS_START_ADDR + 0x28)
#define NOC_CMD_CTRL (NOC_REGS_START_ADDR + 0x40)
#define NOC_NODE_ID (NOC_REGS_START_ADDR + 0x44)

// NOC 命令字段
#define NOC_CMD_AT (0x1 << 0)        // 原子操作
#define NOC_CMD_CPY (0x0 << 0)       // 复制
#define NOC_CMD_RD (0x0 << 1)        // 读
#define NOC_CMD_WR (0x1 << 1)        // 写
#define NOC_CMD_WR_BE (0x1 << 2)     // 写字节使能
#define NOC_CMD_WR_INLINE (0x1 << 3) // 内联写
#define NOC_CMD_RESP_MARKED (0x1 << 4)
#define NOC_CMD_BRCST_PACKET (0x1 << 5)  // 广播包
#define NOC_CMD_VC_LINKED (0x1 << 6)
#define NOC_CMD_PATH_RESERVE (0x1 << 8)

// 地址格式
#define NOC_ADDR_LOCAL_BITS 36
#define NOC_ADDR_NODE_ID_BITS 6
#define NOC_XY_ADDR(x, y, addr) \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | ((uint64_t)(addr)))

// 多播地址
#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr) \
    ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) | ...)
```

#### 3.3.4 c_tensix_core 类

**文件路径**: `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h`

```cpp
class c_tensix_core {
public:
    static const bool is_emulated = false;

    // 缓冲区基址
    static vptr_uint instrn_buf_base(uint32_t thread_id);
    static vptr_pc_buf pc_buf_base(uint32_t thread_id);
    static vptr_uint regfile_base();
    static vptr_uint cfg_regs_base(uint state_id = 0);
    static vptr_mailbox mailbox_base(uint32_t thread_id);

    // Tensix 操作
    static void ex_stallwait(vptr_uint instrn_buf, uint wait_res, uint stall_res);
    static void ex_setc16(uint addr, uint val, vptr_uint instrn_buf);
    static void ex_instrn_wrcfg(uint gpr, uint cfg_addr, vptr_uint instrn_buf);
    static void ex_zeroacc(vptr_uint instrn_buf, uint clear_mode = 3, ...);
    static void ex_encc(vptr_uint instrn_buf);
    static void ex_load_const(vptr_uint instrn_buf);
    static void ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer);

    // NOC 操作
    static void noc_copy(uint32_t src_coordinate, uint64_t src_addr, uint32_t dst_coordinate,
                        uint64_t dst_addr, uint32_t size, bool linked, bool posted, ...);
    static void noc_atomic_increment(uint32_t noc_coordinate, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked);
    static void noc_multicast_copy(uint32_t src_coordinate, uint64_t src_addr, uint32_t dst_coordinate,
                                   uint64_t dst_addr, uint32_t multicast_mode, uint32_t size, ...);

    // 流寄存器
    static inline void write_stream_register(uint32_t stream_id, uint32_t index, uint32_t value);
    static inline uint32_t read_stream_register(uint32_t stream_id, uint32_t index);

    // 时钟
    static uint64_t read_wall_clock();
    static uint32_t read_wall_clock_l();
};
```

#### 3.3.5 数据流 API

**文件路径**: `/tmp/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h`

```cpp
// 获取逻辑坐标
inline uint8_t get_absolute_logical_x();
inline uint8_t get_absolute_logical_y();
inline uint8_t get_relative_logical_x();
inline uint8_t get_relative_logical_y();

// 运行时参数访问
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx);

template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx);

static FORCE_INLINE uintptr_t get_arg_addr(int arg_idx);
static FORCE_INLINE uintptr_t get_common_arg_addr(int arg_idx);

// Quasar 特定
inline uint32_t get_num_threads();
inline uint32_t get_my_thread_id();
```

### 3.4 toolchain/ - 工具链

#### 3.4.1 链接器脚本

**主链接器脚本** (`/tmp/tt-metal/tt_metal/hw/toolchain/main.ld`):

```ld
// 处理器选择
#if defined(COMPILE_FOR_BRISC)
  #define DATA_START MEM_LOCAL_BASE
  #define DATA_SIZE MEM_BRISC_LOCAL_SIZE
  #define STACK_MIN_SIZE MEM_BRISC_STACK_MIN_SIZE
  #define TEXT_START MEM_BRISC_FIRMWARE_BASE
#elif defined(COMPILE_FOR_NCRISC)
  #define DATA_START MEM_LOCAL_BASE
  #define DATA_SIZE MEM_NCRISC_LOCAL_SIZE
  #define STACK_MIN_SIZE MEM_NCRISC_STACK_MIN_SIZE
  #define TEXT_START MEM_NCRISC_FIRMWARE_BASE
#elif defined(COMPILE_FOR_TRISC)
  // TRISC 特定选择宏
#endif

ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  text PT_LOAD;
  data PT_LOAD;
}

SECTIONS
{
  // 设备打印字符串表（仅文件）
  .device_print_strings 0x6400000 : {
    __device_print_strings_start = .;
    KEEP(*(.device_print_strings))
    __device_print_strings_end = .;
  }

  // 代码段
  .text TEXT_START : {
    *(.start)
    *(.text .stub .text.*)
  } :text

  // 数据段
  .data DATA_START : {
    __ldm_data_start = .;
    *(.rodata .rodata.*)
    *(.sdata .sdata.*)
    *(.data .data.*)
    __ldm_data_end = .;
  } :data

  // BSS 段
  .bss : {
    __ldm_bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    __ldm_bss_end = .;
  } :data

  // 栈顶
  __stack_top = DATA_START + DATA_SIZE;
}
```

#### 3.4.2 启动代码

**文件路径**: `/tmp/tt-metal/tt_metal/hw/toolchain/crt0.S`

```asm
.section .start,"ax",@progbits
.global _start
.type _start,@function

_start:
    // 初始化线程指针和栈指针
    lui  tp, %hi(__local_base)
    addi tp, tp, %lo(__local_base)
    lui  sp, %hi(__local_stride)
    addi sp, sp, %lo(__local_stride)

#ifndef COMPILE_FOR_TRISC
    // 非 TRISC：根据 hart ID 计算实际地址
    csrr s2, mhartid
    mul  s2, sp, s2
    add  tp, tp, s2
#endif
    add  sp, tp, sp

    // 跳转到高级启动代码
    tail _start1
```

#### 3.4.3 C++ 运行时替代

**文件路径**: `/tmp/tt-metal/tt_metal/hw/toolchain/substitutes.cpp`

```cpp
// 禁用 atexit（无静态析构）
extern "C" int atexit(void (*f)(void)) { return 0; }

// 无限循环 exit
extern "C" void exit(int ec) {
    while (1) {
        asm volatile("" ::: "memory");
    }
}

// 优化的内存清零
extern "C" void wzerorange(uint32_t* start, uint32_t* end) {
    // 手动展开 4 次
    start += 4;
    while (start <= end) {
        start[-4] = start[-3] = start[-2] = start[-1] = 0;
        asm inline("addi %0,%0,4 * %1" : "+r"(start) : "i"(sizeof(*start)));
    }
    // 处理剩余 0-3 个字
    ...
}

// L1 到本地内存复制
void l1_to_local_mem_copy(uint32_t* dst, uint32_t __attribute__((rvtt_l1_ptr))* src, int32_t len) {
    // 每次处理 3 个字，优化加载-存储延迟
    while (len >= 3) {
        auto v0 = src[0], v1 = src[1], v2 = src[2];
        asm inline(...);  // 指针更新
        dst[-3] = v0, dst[-2] = v1, dst[-1] = v2;
    }
    // 处理剩余
    ...
}
```

## 4. 硬件架构支持

### 4.1 架构对比

| 特性 | Wormhole B0 | Blackhole | Quasar |
|------|-------------|-----------|--------|
| L1 大小 | 1.43 MB | 1.5 MB | 4 MB |
| NCRISC 约束 | IRAM (16KB) | 无 | 无 |
| CB mask | 32 位 | 64 位 | 64 位 |
| DM 核心 | 2 (BRISC+NCRISC) | 2 | 8 |
| TRISC 核心 | 3 | 3 | 4 |
| 本地内存 | 4-8KB | 4-8KB | 4-8KB |
| 缓存 | 无 | 无 | 有（可绕过）|

### 4.2 NOC 实现

**文件路径**: `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/blackhole/noc.c`

```c
// NOC 寄存器访问宏
#define NOC_WRITE_REG(addr, val) \
    ((*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + \
                             noc_get_active_instance() * NOC_INSTANCE_OFFSET + (addr)))) = (val))

#define NOC_READ_REG(addr) \
    (*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + \
                            noc_get_active_instance() * NOC_INSTANCE_OFFSET + (addr))))

// NOC 复制操作
void noc_copy(uint32_t src_coordinate, uint64_t src_addr,
              uint32_t dst_coordinate, uint64_t dst_addr,
              uint32_t size, bool linked, bool posted, ...) {
    bool src_local = unicast_addr_local(src_coordinate);
    if (!src_local) posted = true;

    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, src_coordinate);
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, dst_coordinate);
    NOC_WRITE_REG(NOC_AT_LEN_BE, size);
    NOC_WRITE_REG(NOC_CTRL, ...);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);  // 启动传输
}

// 原子操作
void noc_atomic_increment(uint32_t noc_coordinate, uint64_t addr,
                          uint32_t incr, uint32_t wrap, bool linked) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, noc_coordinate);
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT);
    NOC_WRITE_REG(NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | ...);
    NOC_WRITE_REG(NOC_AT_DATA, incr);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}
```

## 5. 设计模式与实现技巧

### 5.1 条件编译架构支持

```cpp
// 使用宏区分架构
#if defined(ARCH_WORMHOLE)
    // Wormhole 特定代码
#elif defined(ARCH_BLACKHOLE)
    // Blackhole 特定代码
#elif defined(ARCH_QUASAR)
    // Quasar 特定代码
#endif
```

### 5.2 模板化数据类型支持

```cpp
// SFPU 操作模板
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu() {
    // 编译时确定精度和迭代次数
}

// Pack 操作模板
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, bool untilize = false>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, ...);
```

### 5.3 内存访问优化

```cpp
// 使用 tt_l1_ptr 属性标记 L1 指针
typedef uint32_t tt_l1_ptr* tt_l1_ptr_uint32_t;

// 使用 volatile 防止编译器优化硬件寄存器访问
volatile tt_reg_ptr uint* reg_base = reinterpret_cast<volatile uint*>(0xFFB10000);

// 手动循环展开
#pragma GCC unroll 0
while (len >= 3) {
    // 处理 3 个元素
}
```

### 5.4 同步原语

```cpp
// 信号量操作
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

// 等待操作
inline void llk_math_wait_for_dest_available() {
    WAYPOINT("MWDW");
    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
    WAYPOINT("MWDD");
}
```

### 5.5 调试支持

```cpp
// Waypoint 宏用于跟踪执行
#define WAYPOINT(code) \
    do { \
        extern volatile uint32_t DEBUG_MAILBOX[]; \
        DEBUG_MAILBOX[0] = (code)[0] | ((code)[1] << 8) | ((code)[2] << 16) | ((code)[3] << 24); \
    } while (0)

// 使用方式
WAYPOINT("I");   // 初始化
WAYPOINT("W");   // 等待
WAYPOINT("R");   // 运行
WAYPOINT("D");   // 完成
```

## 6. 源码注释摘录

### 6.1 内存映射注释

```cpp
// dev_mem_map.h
// Before adding a define here, read the following:
// 1) Any "truly global" address must be specified explicitly here.  Truly
// global addresses are addresses that are referenced on both the host and
// device or between processors
// 2) Memory section sizes must be specified here, these are used in the
// linker scripts
// 3) static/global variables generally should NOT be listed here.  If
// they are global to a processor, declare them in that processor's source
// code, they will get placed in local memory
// 4) L1 data sections are no longer supported as addressing them with XIP
// binaries requires runtime address patching.  Instead of using named
// variables in the L1 data section use a mailbox (or address in the mailbox
// range and initialize explicitly)
```

### 6.2 NOC 地址格式注释

```cpp
// noc_parameters.h
// BH has 64 bit address space but pipegen was not updated to support this so WH scheme of encoding addresses is used
// (36 bits of address followed by coordinates) This means that lo and mid registers need to have the address portion
// while the coordinates go into hi register
```

### 6.3 Wormhole NCRISC IRAM 注释

```cpp
// ncrisc.cc
// The NCRISC behaves badly if it jumps from L1 to IRAM, so instead halt it and then reset it to the IRAM
// address it provides.
// ...
// Ensure branch predictor will only ever predict into L1. Otherwise, the branch predictor may predict an IRAM
// address, which can cause an instruction to be fetched from IRAM while the mover is writing to IRAM, which can
// cause corruption.
```

### 6.4 内联写注释

```cpp
// dev_mem_map.h (Blackhole)
// On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
// time. If one port on the recipient has no back-pressure then the transaction will hang because there is no mechanism
// to allow one memory port to move ahead of another. To workaround this hang, we emulate inline writes on Blackhole by
// writing the value to be written to local L1 first and then issue a noc async write.
```

### 6.5 编译器优化注释

```cpp
// substitutes.cpp
// 1) Make sure the optimizer does not think this is memcpy by
// hiding the pointer bookkeeping in an asm.
// 2) The scheduler doesn't know the above loads have 6 cycle
// latency. We emit the 3 bookkeeping adds as a single block
// in the load shadow before the stores. The optimizer will
// not be able to move these.
```

---

## 附录：关键文件索引

| 类别 | 文件路径 | 描述 |
|------|----------|------|
| 构建 | `/tmp/tt-metal/tt_metal/hw/CMakeLists.txt` | 主构建配置 |
| 构建 | `/tmp/tt-metal/tt_metal/hw/sources.cmake` | JIT API 头文件列表 |
| 固件 | `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc` | BRISC 固件 |
| 固件 | `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc` | NCRISC 固件 |
| 固件 | `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc` | TRISC 固件 |
| 固件 | `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/erisc.cc` | ERISC 固件 |
| 内存映射 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | Blackhole 内存映射 |
| 内存映射 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` | Wormhole 内存映射 |
| 内存映射 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h` | Quasar 内存映射 |
| 寄存器 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h` | Tensix 寄存器 |
| 寄存器 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` | NOC 寄存器 |
| 核心类 | `/tmp/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h` | Tensix 核心类 |
| NOC 实现 | `/tmp/tt-metal/tt_metal/hw/firmware/src/tt-1xx/blackhole/noc.c` | NOC 操作实现 |
| LLK API | `/tmp/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_common_api.h` | 数学 API |
| LLK API | `/tmp/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h` | Pack API |
| 数据流 | `/tmp/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h` | 数据流 API |
| 工具链 | `/tmp/tt-metal/tt_metal/hw/toolchain/main.ld` | 主链接器脚本 |
| 工具链 | `/tmp/tt-metal/tt_metal/hw/toolchain/crt0.S` | 启动代码 |
| 工具链 | `/tmp/tt-metal/tt_metal/hw/toolchain/substitutes.cpp` | C++ 运行时替代 |
