# test/ 模块源码解析

## 1. 模块概述

TT-Metal 的测试模块位于 `/tmp/tt-metal/tests/tt_metal/tt_metal/`，是一个全面的 C++ 测试框架，基于 Google Test (gtest) 构建，用于验证 TT-Metal 硬件抽象层的各项功能。

### 测试框架

- **框架**: Google Test (gtest)
- **构建系统**: CMake
- **测试类型**: 单元测试、集成测试、性能微基准测试
- **调度模式**: 支持 Fast Dispatch 和 Slow Dispatch 两种模式

### 测试类型

1. **单元测试 (Unit Tests)**: 测试单个组件的功能
2. **集成测试 (Integration Tests)**: 测试多个组件的协同工作
3. **性能微基准测试 (Perf Microbenchmarks)**: 测量特定操作的性能指标
4. **数据移动测试 (Data Movement Tests)**: 验证 NOC (Network on Chip) 数据传输

## 2. 目录结构

```
/tmp/tt-metal/tests/tt_metal/tt_metal/
├── CMakeLists.txt              # 主构建配置
├── sources.cmake               # 遗留测试源文件列表
├── README.md                   # 测试开发指南
├── test_kernels/               # 测试用内核文件
│   ├── compute/               # 计算内核
│   ├── dataflow/              # 数据流内核
│   ├── device_print/          # 设备打印内核
│   ├── misc/                  # 杂项内核
│   └── sfpi/                  # SFPI 内核
├── common/                     # 共享测试夹具
│   ├── device_fixture.hpp     # 设备测试夹具
│   ├── command_queue_fixture.hpp  # 命令队列夹具
│   ├── mesh_dispatch_fixture.hpp  # Mesh 调度夹具
│   ├── multi_device_fixture.hpp   # 多设备夹具
│   └── matmul_test_utils.hpp  # Matmul 测试工具
├── api/                        # API 测试
│   ├── allocator/             # 分配器测试
│   ├── circular_buffer/       # 循环缓冲区测试
│   ├── core_coord/            # 核心坐标测试
│   ├── tensor/                # 张量测试
│   └── test_*.cpp             # 各类 API 测试
├── device/                     # 设备测试
│   ├── test_device.cpp        # 设备功能测试
│   ├── test_device_init_and_teardown.cpp
│   ├── test_device_cluster_api.cpp
│   └── galaxy_fixture.hpp     # Galaxy 测试夹具
├── dispatch/                   # 调度测试
│   ├── dispatch_buffer/       # 缓冲区调度测试
│   ├── dispatch_device/       # 设备调度测试
│   ├── dispatch_event/        # 事件调度测试
│   ├── dispatch_program/      # 程序调度测试
│   ├── dispatch_trace/        # 追踪调度测试
│   ├── dispatch_util/         # 调度工具测试
│   └── sources.cmake          # 分级别的测试源文件
├── integration/                # 集成测试
│   ├── matmul/                # 矩阵乘法测试
│   ├── vecadd/                # 向量加法测试
│   └── test_*.cpp             # 各类集成测试
├── llk/                        # 低级别内核测试
│   ├── test_broadcast.cpp     # 广播测试
│   ├── test_reduce.cpp        # 归约测试
│   ├── test_sfpu_compute.cpp  # SFPU 计算测试
│   └── test_*.cpp             # 其他 LLK 测试
├── debug_tools/                # 调试工具测试
│   ├── device_print/          # 设备打印测试
│   ├── dprint/                # DPrint 测试
│   ├── watcher/               # Watcher 测试
│   └── inspector/             # Inspector 测试
├── data_movement/              # 数据移动测试
│   ├── dram_unary/            # DRAM 一元操作
│   ├── one_to_one/            # 点对点通信
│   ├── one_to_all/            # 一对多通信
│   ├── all_to_all/            # 多对多通信
│   ├── interleaved/           # 交错内存访问
│   └── noc_estimator_tests/   # NOC 估算器测试
├── eth/                        # 以太网测试
│   ├── test_basic_eth.cpp     # 基础以太网测试
│   ├── test_erisc_app_direct_send.cpp
│   └── test_*.cpp             # 其他以太网测试
├── noc/                        # NOC 测试
│   └── test_dynamic_noc.cpp   # 动态 NOC 测试
├── perf_microbenchmark/        # 性能微基准测试
│   ├── dispatch/              # 调度性能测试
│   ├── ethernet/              # 以太网性能测试
│   ├── noc/                   # NOC 性能测试
│   ├── routing/               # 路由性能测试
│   └── tensix/                # Tensix 性能测试
├── jit_build/                  # JIT 构建测试
├── hal_codegen/                # HAL 代码生成测试
├── lightmetal/                 # LightMetal 测试
├── sfpi/                       # SFPI 测试
├── noc_debugging/              # NOC 调试测试
└── context/                    # 上下文测试
```

## 3. 测试分析

### 3.1 单元测试

#### API 测试 (`api/`)

API 测试覆盖 TT-Metal 的核心 API 功能：

- **Allocator 测试**: `test_free_list_opt_allocator.cpp`, `test_l1_banking_allocator.cpp`
- **Circular Buffer 测试**: `test_CircularBuffer_allocation.cpp`, `test_CircularBuffer_creation.cpp`
- **Core Coord 测试**: `test_CoreRange_*.cpp`, `test_CoreRangeSet_*.cpp`
- **Buffer 测试**: `test_dram.cpp`, `test_simple_dram_buffer.cpp`, `test_simple_l1_buffer.cpp`
- **Kernel 测试**: `test_kernel_creation.cpp`, `test_kernel_compile_cache.cpp`
- **NOC 测试**: `test_noc.cpp`
- **Runtime Args 测试**: `test_runtime_args.cpp`

示例代码结构：
```cpp
TEST_F(MeshDeviceSingleCardFixture, Datacopy) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 创建缓冲区
    InterleavedBufferConfig dram_config{...};
    auto src_dram_buffer = CreateBuffer(dram_config);

    // 创建内核
    auto unary_reader_kernel = CreateKernel(program, kernel_path, core, config);

    // 执行并验证
    detail::LaunchProgram(dev, program);
    EXPECT_EQ(src_vec, result_vec);
}
```

#### 设备测试 (`device/`)

设备测试验证设备生命周期管理：

- `test_device_init_and_teardown.cpp`: 设备初始化和清理
- `test_device_cluster_api.cpp`: 集群 API 测试
- `test_device_pool.cpp`: 设备池管理
- `test_galaxy_cluster_api.cpp`: Galaxy 集群 API
- `test_simulator_device.cpp`: 模拟器设备测试

#### 调度测试 (`dispatch/`)

调度测试分为三个级别：

1. **Smoke Tests** (`UNIT_TESTS_DISPATCH_SMOKE_SOURCES`):
   - `test_enqueue_read_write_core.cpp`
   - `test_EnqueueWaitForEvent.cpp`
   - `test_dispatch_stress.cpp`

2. **Basic Tests** (`UNIT_TESTS_DISPATCH_BASIC_SOURCES`):
   - `test_BufferCorePageMapping_Iterator.cpp`
   - `test_dispatch.cpp`
   - `test_dataflow_cb.cpp`

3. **Slow Tests** (`UNIT_TESTS_DISPATCH_SLOW_SOURCES`):
   - `test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp`
   - `test_large_mesh_buffer.cpp`
   - `test_EnqueueProgram.cpp`

### 3.2 集成测试

#### 矩阵乘法测试 (`integration/matmul/`)

- `test_matmul_single_core.cpp`: 单核矩阵乘法
- `test_matmul_multi_core_X_dram.cpp`: 多核多 DRAM 矩阵乘法
- `test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp`: 多播矩阵乘法

#### 其他集成测试

- `test_basic_pipeline.cpp`: 基础流水线测试
- `test_flatten.cpp`: 扁平化操作测试
- `test_sfpu_compute.cpp`: SFPU 计算测试

### 3.3 测试工具

#### 测试夹具 (Fixtures)

位于 `common/` 目录，提供测试基础设施：

**`mesh_dispatch_fixture.hpp`**: 基础 Mesh 调度夹具
```cpp
class MeshDispatchFixture : public ::testing::Test {
protected:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    bool slow_dispatch_{};

    void SetUp() override;
    void TearDown() override;
    void RunProgram(...);
    void WriteBuffer(...);
    void ReadBuffer(...);
};
```

**`device_fixture.hpp`**: 设备测试夹具
```cpp
class MeshDeviceFixture : public MeshDispatchFixture {
    // 慢速调度模式测试
};

class MeshDeviceSingleCardFixture : public MeshDispatchFixture {
    // 单卡测试
};
```

**`command_queue_fixture.hpp`**: 命令队列夹具
```cpp
class UnitMeshCQFixture : public MeshDispatchFixture {
    // 快速调度模式测试
};

class UnitMeshCQSingleCardFixture : virtual public MeshDispatchFixture {
    // 单卡快速调度
};

class UnitMeshCQMultiDeviceFixture : public MeshDispatchFixture {
    // 多设备快速调度
};
```

#### 测试工具函数

位于 `test_utils/` 目录：

- `env_vars.hpp`: 环境变量获取工具
- `test_common.hpp`: 命令行参数解析工具
- `comparison.hpp`: 结果比较工具
- `stimulus.hpp`: 测试数据生成工具
- `packing.hpp`: 数据打包工具
- `bfloat_utils.hpp`: BFloat16 工具

## 4. 测试模式

### 4.1 命名规范

根据 `README.md` 中的规范：

**测试命名**:
- `Tensix*`: 使用 Tensix 核心的测试
- `ActiveEth*`: 使用活动以太网核心的测试
- `IdleEth*`: 使用空闲以太网核心的测试
- `Eth*`: 同时使用活动和空闲以太网核心的测试
- `TensixActiveEth*`, `TensixIdleEth*`: 混合核心类型测试

**文件命名**:
- `*_fixture.hpp`: 包含测试夹具的文件
- `*_test_utils.hpp`: 包含测试工具函数的文件
- `test_*.cpp`: 包含测试用例的文件

### 4.2 测试分类

| 目录 | 测试内容 | 核心类型 |
|------|----------|----------|
| `api/` | API 功能测试 | Tensix |
| `device/` | 设备生命周期 | 混合 |
| `dispatch/` | 调度系统 | Tensix |
| `eth/` | 以太网通信 | Ethernet |
| `llk/` | 低级别内核 | Tensix |
| `debug_tools/` | 调试工具 | Tensix |
| `data_movement/` | 数据移动 | Tensix |
| `integration/` | 端到端场景 | Tensix |
| `perf_microbenchmark/` | 性能基准 | 混合 |

### 4.3 调度模式支持

**Fast Dispatch (快速调度)**:
- 使用 `UnitMeshCQ*` 夹具
- 通过 `MeshDevice::create_unit_meshes()` 创建设备
- 使用 `EnqueueMeshWorkload()` 提交工作负载
- 环境变量: `TT_METAL_SLOW_DISPATCH_MODE` 未设置

**Slow Dispatch (慢速调度)**:
- 使用 `MeshDevice*` 夹具
- 通过 `distributed::MeshDevice::create_unit_meshes()` 创建设备
- 使用 `detail::LaunchProgram()` 启动程序
- 环境变量: `TT_METAL_SLOW_DISPATCH_MODE=1`

### 4.4 构建配置

测试通过 CMake 构建，主要目标：

```cmake
# 遗留单元测试
add_executable(unit_tests_legacy)

# 分类单元测试
add_executable(unit_tests_api)
add_executable(unit_tests_dispatch)
add_executable(unit_tests_device)
add_executable(unit_tests_data_movement)
add_executable(unit_tests_llk)
add_executable(unit_tests_eth)
add_executable(unit_tests_debug_tools)

# 性能微基准测试
add_executable(perf_microbenchmark_tests)

# 验证目标
add_executable(tt-metalium-validation-smoke)
add_executable(tt-metalium-validation-basic)
```

### 4.5 运行测试

```bash
# 构建测试
./build_metal.sh --build-tests

# 运行所有测试
./build/test/tt_metal/unit_tests_api
./build/test/tt_metal/unit_tests_dispatch

# 运行特定测试
./build/test/tt_metal/unit_tests_api --gtest_filter="*DRAM*"

# 慢速调度模式
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_legacy
```

### 4.6 Python 测试支持

项目同时支持 Python 测试框架：

**pytest 配置** (`pytest.ini`):
```ini
[pytest]
timeout = 300
markers =
    post_commit: mark tests to run on post-commit
    slow: marks tests as slow and long
```

**conftest.py**: 提供测试夹具和配置，包括：
- `reset_seeds`: 重置随机种子
- `function_level_defaults`: 函数级默认配置
- `is_ci_env`: CI 环境检测
- `galaxy_type`: Galaxy 集群类型检测

**Sweep Test Framework** (`tests/sweep_framework/`):
- 参数化测试生成
- 结果导出到 PostgreSQL 或 JSON
- 性能分析支持

---

## 参考文件

- `/tmp/tt-metal/tests/tt_metal/tt_metal/README.md`: 测试开发指南
- `/tmp/tt-metal/tests/tt_metal/tt_metal/CMakeLists.txt`: 主构建配置
- `/tmp/tt-metal/tests/tt_metal/tt_metal/sources.cmake`: 遗留测试源文件
- `/tmp/tt-metal/tests/README.md`: 测试框架文档
- `/tmp/tt-metal/pytest.ini`: Python 测试配置
- `/tmp/tt-metal/conftest.py`: Python 测试夹具
