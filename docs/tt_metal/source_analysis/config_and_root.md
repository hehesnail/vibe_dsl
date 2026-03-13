# 配置与根文件源码解析

## 1. 构建配置

### 1.1 CMakeLists.txt

**文件路径**: `/tmp/tt-metal/tt_metal/CMakeLists.txt`

CMakeLists.txt 是 TT-Metal 构建系统的核心配置文件，定义了主库 `tt_metal` 的构建规则：

#### 主要功能

1. **库定义与别名**
   - 创建 `tt_metal` 主库
   - 提供两个别名：`TT::Metalium` 和 `Metalium::Metal`（向后兼容）

2. **FlatBuffers 代码生成**
   - 使用 `GENERATE_FBS_HEADER` 从 `mesh_coordinate.fbs` 生成序列化描述符头文件
   - 生成目标：`metalium_GeneratedHeaders`

3. **JIT API 库**
   - 创建 `jitapi` 接口库
   - 包含计算内核 API、Fabric 硬件头文件和 TT-LLK 头文件
   - 包含核心描述符、SoC 描述符和网格图描述符 YAML/文本文件

4. **依赖链接**
   - **公共依赖**: `umd::device`, `enchantum::enchantum`, `fmt::fmt-header-only`, `TracyClient`, `nlohmann_json::nlohmann_json`
   - **私有依赖**: `Metalium::Metal::Impl`, `llrt`, `fabric`, `FlatBuffers::FlatBuffers`, `Reflect::Reflect`

5. **子目录添加**
   - `logging/`, `hw/`, `hostdevcommon/`, `common/`, `jit_build/`
   - `llrt/`, `tools/`, `impl/`, `detail/`, `distributed/`, `fabric/`

6. **安装配置**
   - 支持 `TT_INSTALL` 选项
   - 导出第三方依赖：`reflect`, `enchantum`, `TracyClient`, `tt-logger`

### 1.2 sources.cmake

**文件路径**: `/tmp/tt-metal/tt_metal/sources.cmake`

定义了 TT-Metal 库的源文件和公共 API 头文件列表：

#### TT_METAL_PUBLIC_API (110+ 个头文件)

包含以下主要类别的 API：

| 类别 | 主要头文件 |
|------|-----------|
| 内存管理 | `allocator.hpp`, `buffer.hpp`, `buffer_types.hpp` |
| 循环缓冲区 | `circular_buffer.hpp`, `circular_buffer_config.hpp`, `circular_buffer_constants.h` |
| 设备管理 | `device.hpp`, `device_types.hpp`, `cluster.hpp` |
| 分布式计算 | `distributed.hpp`, `mesh_device.hpp`, `mesh_buffer.hpp`, `mesh_command_queue.hpp` |
| 数据结构 | `bfloat16.hpp`, `bfloat4.hpp`, `bfloat8.hpp`, `data_types.hpp` |
| HAL 接口 | `hal.hpp`, `hal_types.hpp` |
| Fabric | `fabric.hpp`, `fabric_types.hpp`, `control_plane.hpp` |
| 张量 | `shape.hpp`, `tile.hpp`, `tensor_accessor_args.hpp` |
| LightMetal | `lightmetal_binary.hpp`, `lightmetal_capture_utils.hpp`, `lightmetal_replay.hpp` |
| UDM | `mesh_builder.hpp`, `mesh_kernel.hpp`, `mesh_program.hpp` |

#### TT_METAL_SOURCES

核心源文件：
- `tt_metal.cpp` - 主 API 实现
- `hal.cpp` - HAL 接口实现
- `graph/graph_tracking.cpp` - 图追踪功能

#### JITAPI_FILES

包含运行时编译所需的文件：
- 核心描述符 YAML 文件（Blackhole、Wormhole B0）
- SoC 描述符 YAML 文件
- Fabric 网格图描述符（N150、N300、P100、P150、T3K、Galaxy 等）
- 内核源代码（`cq_dispatch.cpp`, `fabric_erisc_router.cpp` 等）
- 计算和数据流内核模板

---

## 2. 核心源文件

### 2.1 tt_metal.cpp

**文件路径**: `/tmp/tt-metal/tt_metal/tt_metal.cpp`

这是 TT-Metal 的主实现文件，提供了设备管理、内存操作和程序执行的核心功能。

#### 主要命名空间与结构

```cpp
namespace tt::tt_metal {
    // 内部辅助函数和结构
    namespace {
        CoreRangeSet GetCoreRangeSet(...);
        struct DataMovementConfigStatus { ... };
        DataMovementConfigStatus CheckDataMovementConfig(...);
        void ConfigureKernelGroup(...);
        std::optional<uint32_t> get_semaphore_id(...);
    }

    // 详细实现接口
    namespace detail {
        bool WriteToDeviceDRAMChannel(...);
        bool ReadFromDeviceDRAMChannel(...);
        bool WriteToDeviceL1(...);
        bool ReadFromDeviceL1(...);
        IDevice* GetActiveDevice(...);
        std::map<ChipId, IDevice*> CreateDevices(...);
        void LaunchProgram(...);
        bool ConfigureDeviceWithProgram(...);
        void WriteRuntimeArgsToDevice(...);
        void CompileProgram(...);
    }
}
```

#### 核心功能模块

1. **设备生命周期管理**
   - `CreateDevices()` - 创建设备实例，支持多设备配置
   - `CloseDevices()` - 关闭并清理设备资源
   - `ReleaseOwnership()` - 释放 MetalContext 单例

2. **内存操作**
   - `WriteToBuffer()` / `ReadFromBuffer()` - 主机与设备间数据传输
   - `WriteToDeviceSharded()` / `ReadFromDeviceSharded()` - 分片内存操作
   - `WriteToDeviceInterleavedContiguous()` - 交错连续内存写入
   - `ReadShard()` - 读取特定核心分片

3. **程序执行**
   - `LaunchProgram()` - 启动程序执行
   - `WaitProgramDone()` - 等待程序完成
   - `ConfigureDeviceWithProgram()` - 使用程序配置设备

4. **内核配置验证**
   - `CheckDataMovementConfig()` - 验证数据移动配置
   - 检查 NoC 使用冲突（防止死锁）
   - 验证 RISC-V 处理器使用情况

5. **信号量管理**
   - `get_semaphore_id()` - 获取可用信号量 ID
   - 支持最多 `NUM_SEMAPHORES` 个信号量

6. **运行时参数设置**
   - `SetRuntimeArgsImpl()` - 为内核设置运行时参数
   - 支持 `CoreCoord`、`CoreRange`、`CoreRangeSet` 三种规格

### 2.2 hal.cpp

**文件路径**: `/tmp/tt-metal/tt_metal/hal.cpp`

HAL（Hardware Abstraction Layer）接口的简化实现，通过 `MetalContext` 单例访问底层 HAL 功能。

#### 主要函数

```cpp
namespace tt::tt_metal::hal {
    // 架构信息
    tt::ARCH get_arch();
    std::string get_arch_name();

    // 内存大小
    uint32_t get_l1_size();

    // 内存对齐
    uint32_t get_dram_alignment();
    uint32_t get_l1_alignment();
    uint32_t get_pcie_alignment();

    // ERISC L1 内存
    uint32_t get_erisc_l1_unreserved_base();
    uint32_t get_erisc_l1_unreserved_size();

    // Worker L1 内存
    uint32_t get_max_worker_l1_unreserved_size();

    // 特殊浮点值
    float get_eps();
    float get_nan();
    float get_inf();

    // 架构特性
    uint32_t get_arch_num_circular_buffers();
}
```

#### 实现特点

- 所有函数通过 `MetalContext::instance().hal()` 访问底层 HAL
- 使用 `HalProgrammableCoreType` 区分核心类型（TENSIX、ACTIVE_ETH、IDLE_ETH）
- 使用 `HalL1MemAddrType` 区分 L1 内存地址类型（BASE、UNRESERVED、KERNEL_CONFIG、LAUNCH）
- 使用 `HalMemType` 区分内存类型（DRAM、L1、HOST）

---

## 3. 硬件描述符

### 3.1 Core Descriptors

**目录路径**: `/tmp/tt-metal/tt_metal/core_descriptors/`

核心描述符 YAML 文件定义了不同芯片配置的处理器核心布局和调度核心分配。

#### 文件列表

| 文件名 | 描述 |
|--------|------|
| `blackhole_140_arch.yaml` | Blackhole 架构标准配置 |
| `blackhole_140_arch_eth_dispatch.yaml` | Blackhole 以太网调度配置 |
| `blackhole_140_arch_fabric_mux.yaml` | Blackhole Fabric Mux 配置 |
| `wormhole_b0_80_arch.yaml` | Wormhole B0 标准配置 |
| `wormhole_b0_80_arch_eth_dispatch.yaml` | Wormhole B0 以太网调度配置 |
| `wormhole_b0_80_arch_fabric_mux.yaml` | Wormhole B0 Fabric Mux 配置 |

#### Blackhole 核心描述符结构

```yaml
unharvested:    # 未收获配置
  col:          # 列布局
    1:          # 1 个硬件命令队列
      compute_with_storage_grid_range:
        start: [0, 0]
        end: [12, 9]    # 13x10 计算网格
      dispatch_cores: [[-1, 0], [-1, 1], ...]  # 10 个调度核心
      dispatch_core_type: "tensix"
    2:          # 2 个硬件命令队列
      ...

1xharvested:    # 1 列收获配置
  ...

2xharvested:    # 2 列收获配置
  ...
```

#### Wormhole B0 核心描述符结构

```yaml
galaxy:         # Galaxy 系统配置
  row/col:      # 行/列布局选项
    1/2:        # 1 或 2 个硬件命令队列
      compute_with_storage_grid_range:
        start: [0, 0]
        end: [7, 8]     # Galaxy 行布局: 8x9 网格
      tg_compute_with_storage_grid_range:  # TG 模式
      dispatch_cores: [...]
      tg_dispatch_cores: [...]  # TG 扩展调度核心
      dispatch_core_type: "tensix"

nebula_x1:      # Nebula X1 配置
  ...

nebula_x2:      # Nebula X2 配置
  ...
```

#### Fabric Mux 配置

Fabric Mux 配置将调度核心分为两部分：
- `fabric_mux_cores` - Fabric 多路复用器核心
- `dispatch_cores` - 标准调度核心

例如 Blackhole Fabric Mux 配置：
```yaml
fabric_mux_cores: [[0, -1], [1, -1], ..., [7, -1]]  # 8 个核心
dispatch_cores: [[8, -1], [9, -1], ..., [13, -1]]   # 6 个核心
```

### 3.2 SoC Descriptors

**目录路径**: `/tmp/tt-metal/tt_metal/soc_descriptors/`

SoC 描述符 YAML 文件定义了芯片的物理硬件布局，包括网格大小、各类核心的位置、内存配置等。

#### 文件列表

| 文件名 | 描述 |
|--------|------|
| `blackhole_140_arch.yaml` | Blackhole SoC 描述 |
| `wormhole_b0_80_arch.yaml` | Wormhole B0 SoC 描述 |
| `quasar_32_arch.yaml` | Quasar SoC 描述 |

#### Blackhole SoC 描述符

```yaml
grid:
  x_size: 17
  y_size: 12

arc: [8-0]                    # ARC 核心位置
pcie: [2-0, 11-0]             # PCIe 核心位置

dram: [                        # DRAM 通道映射
  [0-0, 0-1, 0-11],           # 通道 0
  [0-2, 0-10, 0-3],           # 通道 1
  ...                         # 共 8 个通道
]

dram_views: [                  # DRAM 视图配置
  {
    channel: 0,
    eth_endpoint: [2, 1],
    worker_endpoint: [2, 1],
    address_offset: 0
  },
  ...
]

dram_view_size: 4278190080    # ~4GB 每视图

eth: [1-1, 16-1, 2-1, ...]    # 14 个以太网核心

functional_workers: [          # 120 个功能工作核心
  1-2, 2-2, 3-2, ...
]

harvested_workers: []         # 收获的核心（空表示无）

router_only: [                # 仅路由核心
  1-0, 3-0, 4-0, ...
]

worker_l1_size: 1572864       # 1.5MB L1 内存
dram_bank_size: 4278190080    # ~4GB DRAM 每 bank
eth_l1_size: 524288           # 512KB 以太网 L1

arch_name: BLACKHOLE

features:                     # 架构特性
  noc:
    translation_id_enabled: True
  unpacker:
    version: 2
    inline_srca_trans_without_srca_trans_instr: True
  math:
    dst_size_alignment: 32768
  packer:
    version: 2
  overlay:
    version: 2
```

#### Wormhole B0 SoC 描述符

```yaml
grid:
  x_size: 10
  y_size: 12

arc: [0-10]                   # ARC 核心
pcie: [0-3]                   # PCIe 核心

dram: [                        # 6 个 DRAM 通道
  [0-0, 0-1, 0-11],
  [0-5, 0-6, 0-7],
  ...
]

dram_views: 12                # 12 个 DRAM 视图
dram_view_size: 1073741824    # 1GB 每视图

eth: [9-0, 1-0, 8-0, ...]     # 16 个以太网核心

functional_workers: 80        # 80 个功能工作核心

worker_l1_size: 1499136       # ~1.4MB L1
dram_bank_size: 2147483648    # 2GB DRAM 每 bank
eth_l1_size: 262144           # 256KB 以太网 L1

arch_name: WORMHOLE_B0
```

#### Quasar SoC 描述符

```yaml
grid:
  x_size: 8
  y_size: 4

arc: []                       # 无 ARC 核心
pcie: []                      # 无 PCIe 核心

dram: [[2-7], [3-7]]          # 2 个 DRAM 通道

functional_workers: 32        # 32 个功能工作核心

worker_l1_size: 1499136
dram_bank_size: 1073741824    # 1GB DRAM
eth_l1_size: 0                # 无以太网 L1

arch_name: QUASAR

features:
  unpacker:
    version: 1                # 版本 1（较旧）
  packer:
    version: 1
  overlay:
    version: 1
```

---

## 4. 第三方依赖

**目录路径**: `/tmp/tt-metal/tt_metal/third_party/`

### 4.1 目录结构

```
third_party/
├── .clang-tidy          # Clang-tidy 配置文件
├── CMakeLists.txt       # 第三方库构建配置
├── tracy/               # Tracy 性能分析器（空目录，外部依赖）
├── tt_llk/              # Tenstorrent 低级内核库（空目录，外部依赖）
└── umd/                 # 统一设备管理器（空目录，外部依赖）
```

### 4.2 CMakeLists.txt

```cmake
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

add_subdirectory(umd)
```

该配置：
- 禁用 Clang-tidy 检查（第三方代码）
- 禁用接口头文件集验证
- 仅添加 UMD 子目录（Tracy 和 TT-LLK 通过其他方式集成）

### 4.3 依赖说明

#### UMD (Unified Device Manager)
- **用途**: 底层硬件设备管理接口
- **集成方式**: `add_subdirectory(umd)`
- **CMake 目标**: `umd::device`

#### TT-LLK (Tenstorrent Low-Level Kernels)
- **用途**: 低级计算内核头文件库
- **集成方式**: 通过 `jitapi` 接口库包含头文件
- **路径模式**:
  - `third_party/tt_llk/common/*.h`
  - `third_party/tt_llk/tt_llk_wormhole_b0/**/*.h`
  - `third_party/tt_llk/tt_llk_blackhole/**/*.h`
  - `third_party/tt_llk/tt_llk_quasar/**/*.h`

#### Tracy
- **用途**: 实时性能分析和帧分析
- **集成方式**: 通过 CMake 目标 `TracyClient`
- **公共 API**: 通过 `tracy/Tracy.hpp` 使用

### 4.4 依赖关系图

```
tt_metal (主库)
├── PUBLIC 依赖
│   ├── umd::device          (设备管理)
│   ├── enchantum::enchantum (枚举工具)
│   ├── fmt::fmt-header-only (格式化)
│   ├── TracyClient          (性能分析)
│   ├── nlohmann_json::nlohmann_json (JSON)
│   ├── TT::Metalium::HostDevCommon (主机设备通用)
│   ├── TT::STL              (标准模板库)
│   └── tt-logger::tt-logger (日志)
│
├── PRIVATE 依赖
│   ├── Metalium::Metal::Impl (内部实现)
│   ├── llrt                 (低级运行时)
│   ├── fabric               (Fabric 网络)
│   ├── jit_build            (JIT 构建)
│   ├── FlatBuffers::FlatBuffers (序列化)
│   └── Reflect::Reflect     (反射)
│
└── jitapi (接口库)
    ├── 计算内核 API
    ├── Fabric 硬件头文件
    └── TT-LLK 头文件
```

---

## 5. 总结

TT-Metal 的配置和根文件构成了整个框架的基础架构：

1. **构建系统**: CMake 配置支持多目标架构（Blackhole、Wormhole、Quasar），通过 YAML 描述符灵活配置硬件资源

2. **核心 API**: `tt_metal.cpp` 提供了设备管理、内存操作、程序执行的完整接口，`hal.cpp` 提供简化的硬件抽象访问

3. **硬件描述**: YAML 描述符将硬件配置与代码分离，支持：
   - 不同收获配置（unharvested、1xharvested、2xharvested）
   - 多种产品形态（Galaxy、Nebula、单芯片）
   - 灵活的调度核心分配（Tensix、Ethernet、Fabric Mux）

4. **第三方集成**: 通过 UMD 访问硬件，TT-LLK 提供计算内核，Tracy 提供性能分析能力
