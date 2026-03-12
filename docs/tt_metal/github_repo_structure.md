# Tenstorrent tt-metal GitHub 仓库结构分析

分析日期: 2026-03-12
仓库地址: https://github.com/tenstorrent/tt-metal

## 1. 顶层目录结构

```
tt-metal/
├── tt_metal/           # 核心 C++ 库代码
├── ttnn/               # TT Neural Network 库 (Python/C++)
├── tt-train/           # 训练框架
├── models/             # 模型实现 (ResNet, BERT, etc.)
├── tests/              # 测试代码
├── docs/               # 文档
├── programming_examples/ # 编程示例 (在 tt_metal/ 下)
├── tools/              # 工具脚本
├── scripts/            # 构建和辅助脚本
├── cmake/              # CMake 配置
├── infra/              # 基础设施配置
├── tech_reports/       # 技术报告
├── third_party/        # 第三方依赖
├── conftest.py         # pytest 配置
├── setup.py            # Python 包安装
├── CMakeLists.txt      # 主 CMake 配置
├── build_metal.sh      # 构建脚本
└── README.md
```

## 2. tt_metal/ 目录详细结构

### 2.1 API 头文件 (tt_metal/api/tt-metalium/)

这是主要的公共 API 目录，包含 127 个头文件：

**核心 API 文件：**
- `host_api.hpp` - 主机端主要 API (43KB，包含设备管理、缓冲区、程序、内核等 API)
- `buffer.hpp` - 缓冲区管理
- `device.hpp` - 设备接口
- `program.hpp` - 程序管理
- `kernel_types.hpp` - 内核类型定义
- `circular_buffer.hpp` - 循环缓冲区
- `event.hpp` - 事件管理
- `mesh_device.hpp` - 网格设备 (多芯片)
- `distributed.hpp` - 分布式计算支持

**数据类型和工具：**
- `bfloat16.hpp`, `bfloat8.hpp`, `bfloat4.hpp` - 低精度浮点类型
- `core_coord.hpp` - 核心坐标系统
- `hal.hpp` - 硬件抽象层接口
- `allocator.hpp` - 内存分配器

**实验性 API (tt_metal/api/tt-metalium/experimental/)：**
- `host_api.hpp` - 实验性主机 API
- `fabric/` - 高速互联结构 (Fabric) 相关 API
- `lightmetal/` - LightMetal 轻量级运行时 API
- `kernel_cache.hpp` - 内核缓存

### 2.2 实现代码 (tt_metal/impl/)

```
impl/
├── allocator/              # 内存分配器实现
│   └── algorithms/         # 分配算法
├── buffers/                # 缓冲区实现
├── context/                # 上下文管理
├── data_format/            # 数据格式处理
├── dataflow_buffer/        # 数据流缓冲区
├── debug/                  # 调试工具
│   └── inspector/          # 检查器工具
├── device/                 # 设备实现
│   ├── experimental/       # 实验性功能
│   └── firmware/           # 固件相关
├── dispatch/               # 调度系统
│   ├── kernel_config/      # 内核配置
│   ├── kernels/            # 调度内核
│   └── util/               # 工具函数
├── event/                  # 事件实现
├── experimental/           # 实验性功能
│   └── udm/                # UDM (Unified Data Movement)
├── flatbuffer/             # FlatBuffer 序列化
├── kernels/                # 内核管理
├── lightmetal/             # LightMetal 实现
│   └── host_api_capture_helpers.cpp
├── profiler/               # 性能分析器
├── program/                # 程序实现
├── sub_device/             # 子设备管理
├── tensor/                 # 张量实现
│   ├── spec/               # 张量规范
│   └── topology/           # 拓扑信息
└── trace/                  # 追踪功能
```

### 2.3 硬件抽象层 (tt_metal/hw/)

```
hw/
├── ckernels/               # 芯片内核库
│   ├── blackhole/          # Blackhole 架构
│   ├── quasar/             # Quasar 架构
│   └── wormhole_b0/        # Wormhole B0 架构
├── firmware/               # 固件代码
│   └── src/
├── inc/                    # 硬件头文件
│   ├── api/                # 公共 API 头文件
│   │   ├── compute/        # 计算内核 API
│   │   ├── dataflow/       # 数据流 API
│   │   ├── debug/          # 调试 API
│   │   ├── numeric/        # 数值类型
│   │   └── tensor/         # 张量 API
│   ├── experimental/       # 实验性头文件
│   ├── hostdev/            # 主机-设备共享定义
│   └── internal/           # 内部实现头文件
│       ├── dataflow/       # 数据流内部 API
│       ├── debug/          # 调试内部 API
│       ├── ethernet/       # 以太网内部 API
│       ├── tt-1xx/         # TT-1xx 系列架构
│       │   ├── blackhole/
│       │   └── wormhole/
│       └── tt-2xx/         # TT-2xx 系列架构
│           └── quasar/
└── toolchain/              # 工具链配置
```

### 2.4 detail/ 目录

```
detail/
└── reports/                # 报告生成
```

注意: `detail/` 目录相对较小，大部分底层实现在 `hw/inc/internal/` 和 `impl/` 中。

### 2.5 其他重要目录

```
tt_metal/
├── common/                 # 通用工具代码
├── core_descriptors/       # 核心描述符配置
├── distributed/            # 分布式计算
│   ├── experimental/
│   ├── flatbuffer/
│   └── multihost/          # 多主机支持
├── fabric/                 # 高速互联结构实现
│   ├── builder/
│   ├── ccl/                # 集合通信库
│   ├── config/
│   ├── debug/
│   ├── hw/                 # Fabric 硬件代码
│   ├── impl/
│   ├── mesh_graph_descriptors/
│   └── serialization/
├── graph/                  # 图跟踪
├── hostdevcommon/          # 主机-设备共享代码
│   └── api/hostdevcommon/
├── jit_build/              # JIT 编译系统
├── kernels/                # 内置内核模板
│   ├── compute/            # 计算内核模板
│   └── dataflow/           # 数据流内核模板
├── llrt/                   # 低级运行时
│   ├── hal/                # HAL 实现
│   │   ├── codegen/
│   │   ├── tt-1xx/
│   │   └── tt-2xx/
│   └── llrt_common/
├── logging/                # 日志系统
├── programming_examples/   # 编程示例 (见下文)
├── soc_descriptors/        # SoC 描述符
├── test/                   # 内部测试
├── third_party/            # 第三方库
└── tools/                  # 开发工具
    ├── profiler/
    └── watcher/
```

## 3. 关键头文件位置

### 3.1 Host API

| 文件 | 路径 | 描述 |
|------|------|------|
| host_api.hpp | `tt_metal/api/tt-metalium/host_api.hpp` | 主主机 API |
| experimental/host_api.hpp | `tt_metal/api/tt-metalium/experimental/host_api.hpp` | 实验性 API |

**host_api.hpp 主要功能模块：**
- 设备管理 (Device management)
- 缓冲区管理 (Buffers)
- 程序创建 (Program creation)
- 内核创建 (Kernel creation)
- 运行时参数 (Runtime arguments)
- 循环缓冲区 (Circular buffers)
- 事件和追踪 (Events and trace)

### 3.2 Dataflow API

| 文件 | 路径 | 描述 |
|------|------|------|
| dataflow_api.h | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | 数据流主 API (132KB) |
| dataflow_api_addrgen.h | `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h` | 地址生成内部 API |
| dataflow_api_common.h | `tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h` | 通用数据流定义 |

### 3.3 Compute Kernel API

| 文件 | 路径 | 描述 |
|------|------|------|
| compute_kernel_api.h | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | 计算内核主 API (53KB) |
| cb_api.h | `tt_metal/hw/inc/api/compute/cb_api.h` | 循环缓冲区 API |
| matmul.h | `tt_metal/hw/inc/api/compute/matmul.h` | 矩阵乘法 API |
| pack.h | `tt_metal/hw/inc/api/compute/pack.h` | Pack 操作 API |
| reduce.h | `tt_metal/hw/inc/api/compute/reduce.h` | 归约操作 API |
| eltwise_binary.h | `tt_metal/hw/inc/api/compute/eltwise_binary.h` | 逐元素二元操作 |
| eltwise_unary/ | `tt_metal/hw/inc/api/compute/eltwise_unary/` | 逐元素一元操作 (74 个文件) |

**eltwise_unary/ 目录包含：**
- 激活函数: `activations.h`, `gelu.h`, `relu.h`, `sigmoid.h`, `softmax.h`
- 数学运算: `exp.h`, `log1p.h`, `recip.h`, `rsqrt.h`, `sqrt.h`
- 三角函数: `trigonometry.h`
- 其他: `clamp.h`, `dropout.h`, `typecast.h`, `where.h`

### 3.4 调试 API

| 文件 | 路径 | 描述 |
|------|------|------|
| assert.h | `tt_metal/hw/inc/api/debug/assert.h` | 断言支持 |
| dprint.h | `tt_metal/hw/inc/api/debug/dprint.h` | 设备打印 |
| ring_buffer.h | `tt_metal/hw/inc/api/debug/ring_buffer.h` | 环形缓冲区调试 |

## 4. docs/ 文档目录结构

```
docs/
└── source/
    ├── common/               # 共享资源
    │   ├── _static/
    │   ├── _templates/
    │   └── images/
    ├── tech_reports/         # 技术报告
    │   └── WH_Galaxy/
    ├── tt-metalium/          # tt-metalium 文档
    │   ├── get_started/
    │   ├── tools/
    │   └── tt_metal/
    │       ├── advanced_topics/
    │       ├── apis/
    │       │   ├── host_apis/     # 主机 API 文档
    │       │   │   ├── buffers/
    │       │   │   ├── device_management/
    │       │   │   ├── kernels/
    │       │   │   ├── profiler/
    │       │   │   ├── program/
    │       │   │   └── runtime_args/
    │       │   └── kernel_apis/   # 内核 API 文档
    │       │       ├── circular_buffers/
    │       │       ├── compute/
    │       │       ├── data_movement/
    │       │       ├── kernel_args/
    │       │       ├── pack_unpack/
    │       │       └── sfpu/
    │       ├── environment_variables/
    │       ├── examples/
    │       ├── labs/
    │       │   └── matmul/
    │       └── programming_model/
    └── ttnn/                 # TTNN 文档
        ├── tools/
        └── ttnn/
            └── tutorials/
```

## 5. tests/ 测试代码结构

```
tests/
├── device_perf_tests/      # 设备性能测试
├── didt/                   # 电流瞬态测试
├── end_to_end_tests/       # 端到端测试
├── nightly/                # 夜间回归测试
│   ├── blackhole/
│   ├── single_card/        # 单卡模型测试
│   │   ├── common_models/
│   │   ├── resnet50/
│   │   ├── bert/
│   │   ├── vit/
│   │   └── ... (30+ 模型)
│   ├── t3000/              # T3000 系统测试
│   └── tg/                 # Galaxy 测试
├── pipeline_reorg/         # 流水线重组测试
├── scale_out/              # 扩展测试
│   └── 4x_bh_quietbox/
├── scripts/                # 测试脚本
│   ├── multihost/
│   ├── single_card/
│   ├── t3000/
│   └── tg/
├── sweep_framework/        # 参数扫描测试框架
│   ├── framework/
│   └── sweeps/             # 各类算子扫描测试
│       ├── ccl/
│       ├── conv2d/
│       ├── eltwise/
│       ├── matmul/
│       └── ...
└── tt_metal/               # tt_metal 核心测试
    ├── distributed/        # 分布式测试
    ├── llrt/               # 低级运行时测试
    ├── microbenchmarks/    # 微基准测试
    │   ├── ethernet/
    │   ├── noc/
    │   └── tensix/
    ├── multihost/          # 多主机测试
    ├── test_utils/         # 测试工具
    ├── tools/              # 工具测试
    ├── tt_fabric/          # Fabric 测试
    └── tt_metal/           # 核心功能测试
        ├── api/            # API 测试
        ├── common/
        ├── context/
        ├── data_movement/  # 数据移动测试
        ├── debug_tools/    # 调试工具测试
        ├── device/         # 设备测试
        ├── dispatch/       # 调度测试
        ├── eth/            # 以太网测试
        ├── hal_codegen/    # HAL 代码生成测试
        ├── jit_build/      # JIT 编译测试
        ├── llk/            # LLK 测试
        ├── noc/            # NoC 测试
        ├── perf_microbenchmark/
        ├── sfpi/           # SFPI 测试
        └── test_kernels/   # 测试内核
```

## 6. programming_examples/ 示例代码

```
tt_metal/programming_examples/
├── add_2_integers_in_compute/      # 计算内核加法示例
│   └── kernels/compute/
├── add_2_integers_in_riscv/        # RISC-V 内核加法示例
│   └── kernels/
├── contributed/                    # 社区贡献示例
│   ├── multicast/
│   └── vecadd/
├── custom_sfpi_add/                # SFPI 自定义加法
├── custom_sfpi_smoothstep/         # SFPI smoothstep 函数
├── distributed/                    # 分布式编程示例
│   ├── 1_distributed_program_dispatch/
│   ├── 2_distributed_buffer_rw/
│   ├── 3_distributed_eltwise_add/
│   └── 4_distributed_trace_and_events/
├── eltwise_binary/                 # 逐元素二元操作
├── eltwise_sfpu/                   # SFPU 逐元素操作
├── hello_world_compute_kernel/     # 计算内核 Hello World
├── hello_world_datamovement_kernel/# 数据移动内核 Hello World
├── hello_world_datatypes_kernel/   # 数据类型示例
├── loopback/                       # 回环测试
├── matmul/                         # 矩阵乘法示例
│   ├── matmul_common/              # 共享代码
│   ├── matmul_single_core/         # 单核版本
│   ├── matmul_multi_core/          # 多核版本
│   ├── matmul_multicore_reuse/     # 多核复用版本
│   └── matmul_multicore_reuse_mcast/ # 多播版本
├── NoC_tile_transfer/              # NoC 数据传输
├── pad_multi_core/                 # 多核填充
├── profiler/                       # 性能分析示例
│   ├── kernel_profiler/
│   ├── profile_host_device_memory/
│   ├── profile_host_dispatch/
│   ├── profile_host_perf_markers/
│   ├── profile_host_tracy/
│   ├── profile_noc_event_trace/
│   ├── profile_noc_metrics/
│   └── profile_tensix_compute/
├── sfpu_eltwise_chain/             # SFPU 操作链
├── shard_data_rm/                  # 行主序分片数据
├── tests/                          # 示例测试
├── vecadd_multi_core/              # 多核向量加法
└── vecadd_sharding/                # 分片向量加法
```

## 7. 模块划分逻辑

### 7.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API (ttnn/)                       │
├─────────────────────────────────────────────────────────────┤
│                    C++ API (api/tt-metalium/)               │
│  - host_api.hpp (设备、缓冲区、程序、内核管理)               │
│  - buffer.hpp, device.hpp, program.hpp                      │
├─────────────────────────────────────────────────────────────┤
│                    Implementation (impl/)                   │
│  - 设备管理、调度、内存分配、张量操作                        │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Abstraction (hw/)               │
│  - Kernel API: compute/, dataflow/                          │
│  - Architecture specific: ckernels/, inc/internal/          │
├─────────────────────────────────────────────────────────────┤
│                    Low Level Runtime (llrt/)                │
│  - HAL, cluster management, firmware interface              │
├─────────────────────────────────────────────────────────────┤
│                    Hardware (Blackhole/Quasar/Wormhole)     │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 关键设计模式

1. **Host API vs Kernel API 分离**
   - Host API: `api/tt-metalium/` - 主机端 C++ API
   - Kernel API: `hw/inc/api/` - 设备端内核 API

2. **架构抽象**
   - `hw/inc/internal/tt-1xx/` - Blackhole, Wormhole 架构
   - `hw/inc/internal/tt-2xx/` - Quasar 架构

3. **分层实现**
   - API 声明在 `api/`
   - 实现在 `impl/`
   - 硬件细节在 `hw/`

4. **多芯片支持**
   - `distributed/` - 分布式计算
   - `fabric/` - 芯片间高速互联
   - `mesh_device.hpp` - 网格设备抽象

## 8. 重要文件路径汇总

| 类别 | 文件路径 | 说明 |
|------|----------|------|
| 主 API | `tt_metal/api/tt-metalium/host_api.hpp` | 主机端主要 API |
| 数据流 API | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | 数据流操作 API |
| 计算 API | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | 计算内核 API |
| 设备接口 | `tt_metal/api/tt-metalium/device.hpp` | 设备抽象 |
| 缓冲区 | `tt_metal/api/tt-metalium/buffer.hpp` | 缓冲区管理 |
| 程序 | `tt_metal/api/tt-metalium/program.hpp` | 程序定义 |
| HAL | `tt_metal/api/tt-metalium/hal.hpp` | 硬件抽象层 |
| 集群 | `tt_metal/api/tt-metalium/cluster.hpp` | 集群管理 |
| 网格设备 | `tt_metal/api/tt-metalium/mesh_device.hpp` | 多芯片设备 |
| 主实现 | `tt_metal/tt_metal.cpp` | 主要实现文件 |
| 构建脚本 | `build_metal.sh` | 主构建脚本 |
| 安装指南 | `INSTALLING.md` | 安装说明 |
| 贡献指南 | `CONTRIBUTING.md` | 贡献者指南 |
| Metalium 指南 | `METALIUM_GUIDE.md` | Metalium 使用指南 |

---

*此文档基于 tt-metal 仓库的代码结构分析生成*
