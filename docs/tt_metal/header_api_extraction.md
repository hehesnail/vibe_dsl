# TT-Metal Header API Extraction

This document contains the extracted API definitions from TT-Metal header files.

**Source Repository:** https://github.com/tenstorrent/tt-metal
**Extraction Date:** 2026-03-12

---

## Table of Contents

1. [Host API (host_api.hpp)](#1-host-api-host_apihpp)
2. [TT-Metal Core API (tt_metal.hpp)](#2-tt-metal-core-api-tt_metalhpp)
3. [Data Movement API (dataflow_api.h)](#3-data-movement-api-dataflow_apih)
4. [Compute Kernel API](#4-compute-kernel-api)
5. [Buffer API (buffer.hpp)](#5-buffer-api-bufferhpp)
6. [Device API (device.hpp)](#6-device-api-devicehpp)
7. [Program API (program.hpp)](#7-program-api-programhpp)

---

## 1. Host API (host_api.hpp)

**File Path:** `/tmp/tt-metal/tt_metal/api/tt-metalium/host_api.hpp`

### 1.1 Device Management

#### SetRootDir
```cpp
void SetRootDir(const std::string& root_dir);
```
Sets the root directory for TT Metal meta data files like kernel sources.

#### GetNumAvailableDevices
```cpp
size_t GetNumAvailableDevices();
```
Returns number of Tenstorrent devices that can be targeted.

#### IsGalaxyCluster
```cpp
bool IsGalaxyCluster();
```
Returns whether Tenstorrent devices are in a Galaxy cluster.

#### GetNumPCIeDevices
```cpp
size_t GetNumPCIeDevices();
```
Returns number of Tenstorrent devices connected via PCIe.

#### GetPCIeDeviceID
```cpp
ChipId GetPCIeDeviceID(ChipId device_id);
```

#### CreateDevice
```cpp
IDevice* CreateDevice(
    ChipId device_id,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
```
Instantiates a device object.

#### CreateDeviceMinimal
```cpp
IDevice* CreateDeviceMinimal(
    ChipId device_id,
    uint8_t num_hw_cqs = 1,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{});
```
Instantiates a device with minimal setup for attaching to a device in bad state.

#### CloseDevice
```cpp
bool CloseDevice(IDevice* device);
```
Resets device and closes device.

### 1.2 Program & Kernels

#### CreateProgram
```cpp
Program CreateProgram();
```
Creates a Program object - main container for kernels, circular buffers, and semaphores.

#### CreateKernel
```cpp
KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config);
```
Creates a data movement or compute kernel and adds it to the program.

#### CreateKernelFromString
```cpp
KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config);
```
Creates a kernel from source code string.

### 1.3 Circular Buffers

#### CreateCircularBuffer
```cpp
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config);
```
Creates a Circular Buffer (CB) in L1 memory.

#### GetCircularBufferConfig
```cpp
const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle);
```
Gets reference to config owned by circular buffer.

#### UpdateCircularBufferTotalSize
```cpp
void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size);
```
Updates total size of circular buffer.

#### UpdateCircularBufferPageSize
```cpp
void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size);
```
Updates page size at specified buffer index.

#### UpdateDynamicCircularBufferAddress
```cpp
void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer);
```
Updates address of dynamic circular buffer.

#### UpdateDynamicCircularBufferAddressAndTotalSize
```cpp
void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size);
```

### 1.4 Semaphores

#### CreateSemaphore
```cpp
uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value);
```
Initializes semaphore on all cores within core range.

#### CreateGlobalSemaphore
```cpp
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);
```
Initializes a global semaphore on all cores.

### 1.5 Buffers

#### CreateBuffer (Interleaved)
```cpp
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config);
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address);
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id);
```
Creates pre-allocated interleaved DRAM or L1 buffer.

#### CreateBuffer (Sharded)
```cpp
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config);
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address);
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id);
```
Creates pre-allocated sharded DRAM or L1 buffer.

#### DeallocateBuffer
```cpp
void DeallocateBuffer(Buffer& buffer);
```
Deallocates buffer from device.

#### AssignGlobalBufferToProgram
```cpp
void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program);
```
Gives program ownership of the buffer.

### 1.6 Runtime Arguments

#### SetRuntimeArgs
```cpp
void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    stl::Span<const uint32_t> runtime_args);

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::initializer_list<uint32_t> runtime_args);

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args);
```
Set runtime args for a kernel. Maximum 341 args per core.

#### SetCommonRuntimeArgs
```cpp
void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args);
void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, std::initializer_list<uint32_t> runtime_args);
```
Set common (shared by all cores) runtime args.

#### GetRuntimeArgs
```cpp
RuntimeArgsData& GetRuntimeArgs(const Program& program, KernelHandle kernel_id, const CoreCoord& logical_core);
std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(const Program& program, KernelHandle kernel_id);
```
Get runtime args for a kernel.

#### GetCommonRuntimeArgs
```cpp
RuntimeArgsData& GetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id);
```
Get common runtime args for a kernel.

### 1.7 Profiling & Events

#### ReadMeshDeviceProfilerResults
```cpp
void ReadMeshDeviceProfilerResults(
    distributed::MeshDevice& mesh_device,
    ProfilerReadState state = ProfilerReadState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});
```
Read device side profiler data.

#### EventQuery
```cpp
bool EventQuery(const std::shared_ptr<Event>& event);
```
Host queries an event for completion status.

#### Command Queue Stack Operations
```cpp
void PushCurrentCommandQueueIdForThread(uint8_t cq_id);
uint8_t PopCurrentCommandQueueIdForThread();
uint8_t GetCurrentCommandQueueIdForThread();
```

---

## 2. TT-Metal Core API (tt_metal.hpp)

**File Path:** `/tmp/tt-metal/tt_metal/api/tt-metalium/tt_metal.hpp`

### 2.1 Device Management (detail namespace)

#### CreateDevices
```cpp
std::map<ChipId, IDevice*> CreateDevices(
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const tt_metal::DispatchCoreConfig& dispatch_core_config = tt_metal::DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    bool init_profiler = true,
    bool ignored = false,
    bool initialize_fabric_and_dispatch_fw = true);
```

#### CloseDevices
```cpp
void CloseDevices(const std::map<ChipId, IDevice*>& devices);
```

#### ReleaseOwnership
```cpp
void ReleaseOwnership();
```
Release ownership of MetalContext singleton.

#### GetActiveDevice
```cpp
IDevice* GetActiveDevice(ChipId device_id);
```

### 2.2 Buffer Operations

#### WriteToBuffer
```cpp
void WriteToBuffer(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer);
template <typename DType> void WriteToBuffer(Buffer& buffer, const std::vector<DType>& host_buffer);
template <typename DType> void WriteToBuffer(const std::shared_ptr<Buffer>& buffer, const std::vector<DType>& host_buffer);
```
Copies data from host buffer to device buffer.

#### ReadFromBuffer
```cpp
void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer);
template <typename DType> void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer);
template <typename DType> void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<DType>& host_buffer);
```
Copies data from device buffer to host buffer.

#### ReadShard
```cpp
void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id);
template <typename DType> void ReadShard(Buffer& buffer, std::vector<DType>& host_buffer, const uint32_t& core_id);
```

### 2.3 Program Execution

#### LaunchProgram
```cpp
void LaunchProgram(IDevice* device, Program& program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);
void LaunchProgram(IDevice* device, const std::shared_ptr<Program>& program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);
void WaitProgramDone(IDevice* device, Program& program, bool read_device_profiler_results = true);
```

#### CompileProgram
```cpp
void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);
```
Compiles all kernels within the program.

#### WriteRuntimeArgsToDevice
```cpp
void WriteRuntimeArgsToDevice(IDevice* device, Program& program, bool force_slow_dispatch = false);
```

#### ConfigureDeviceWithProgram
```cpp
bool ConfigureDeviceWithProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);
```

### 2.4 Program ID Encoding

#### EncodePerDeviceProgramID
```cpp
uint32_t EncodePerDeviceProgramID(uint32_t base_program_id, uint32_t device_id, bool is_host_fallback_op = false);
```

#### DecodePerDeviceProgramID
```cpp
DeviceProgramId DecodePerDeviceProgramID(uint32_t device_program_id);
```

### 2.5 Direct Memory Access

#### WriteToDeviceDRAMChannel
```cpp
bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<const uint8_t> host_buffer);
bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer);
```

#### ReadFromDeviceDRAMChannel
```cpp
bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer);
bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer);
```

#### WriteToDeviceL1
```cpp
bool WriteToDeviceL1(IDevice* device, const CoreCoord& logical_core, uint32_t address, std::span<const uint8_t> host_buffer, CoreType core_type = CoreType::WORKER);
bool WriteToDeviceL1(IDevice* device, const CoreCoord& logical_core, uint32_t address, std::vector<uint32_t>& host_buffer, CoreType core_type = CoreType::WORKER);
```

#### ReadFromDeviceL1
```cpp
bool ReadFromDeviceL1(IDevice* device, const CoreCoord& logical_core, uint32_t address, std::span<uint8_t> host_buffer, CoreType core_type = CoreType::WORKER);
bool ReadFromDeviceL1(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer, CoreType core_type = CoreType::WORKER);
```

#### WriteRegToDevice / ReadRegFromDevice
```cpp
bool WriteRegToDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, const uint32_t& regval);
bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval);
```

### 2.6 Platform Info

#### get_platform_architecture_name
```cpp
std::string get_platform_architecture_name();
```

---

## 3. Data Movement API (dataflow_api.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h`

### 3.1 Core Coordinate Functions

```cpp
inline uint8_t get_absolute_logical_x();
inline uint8_t get_absolute_logical_y();
inline uint8_t get_relative_logical_x();
inline uint8_t get_relative_logical_y();
inline uint32_t get_num_threads();        // ARCH_QUASAR only
inline uint32_t get_my_thread_id();       // ARCH_QUASAR only
```

### 3.2 Runtime Argument Access

```cpp
static FORCE_INLINE uintptr_t get_arg_addr(int arg_idx);
static FORCE_INLINE uintptr_t get_common_arg_addr(int arg_idx);
template <typename T> FORCE_INLINE T get_arg_val(int arg_idx);
template <typename T> FORCE_INLINE T get_common_arg_val(int arg_idx);
```

### 3.3 Circular Buffer Operations

```cpp
FORCE_INLINE void cb_push_back(const int32_t operand, const int32_t num_pages);
FORCE_INLINE void cb_pop_front(int32_t operand, int32_t num_pages);
FORCE_INLINE uint32_t get_write_ptr(uint32_t operand);
FORCE_INLINE uint32_t get_read_ptr(uint32_t operand);
FORCE_INLINE bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
FORCE_INLINE void cb_reserve_back(int32_t operand, int32_t num_pages);
FORCE_INLINE bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
FORCE_INLINE void cb_wait_front(int32_t operand, int32_t num_pages);
```

### 3.4 Tile Information

```cpp
constexpr inline std::int32_t get_tile_size(const std::int32_t operand);
constexpr inline uint32_t get_tile_hw(const std::int32_t operand);
constexpr inline uint32_t get_tile_num_faces(const std::int32_t operand);
constexpr inline DataFormat get_dataformat(const std::int32_t operand);
```

### 3.5 NOC Read Operations

```cpp
template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_one_packet(
    uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index);

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true>
inline void noc_async_read(
    uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index);

template <bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_set_state(uint64_t src_noc_addr, uint32_t size, const uint32_t vc = 0, uint8_t noc = noc_index);

template <bool inc_num_issued = true, bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_with_state(
    uint32_t src_local_l1_addr, uint32_t dst_local_l1_addr, const uint32_t vc = 0, uint8_t noc = noc_index);

FORCE_INLINE void noc_async_read_set_state(uint64_t src_noc_addr, uint8_t noc = noc_index);

template <bool inc_num_issued = true>
FORCE_INLINE void noc_async_read_with_state(
    uint32_t src_local_l1_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = noc_index);

FORCE_INLINE void noc_async_read_inc_num_issued(std::uint32_t num_issued_reads_inc, uint8_t noc = noc_index);
```

### 3.6 NOC Write Operations

```cpp
template <bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index);

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true, bool posted = false>
inline void noc_async_write(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t noc = noc_index);

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr_multicast, uint32_t size,
    uint32_t num_dests, bool linked = false, uint8_t noc = noc_index);

inline void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size, std::uint32_t num_dests, bool linked = false, uint8_t noc = noc_index);
```

### 3.7 Page-Based Operations

```cpp
template <typename AddrGen, bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id, const AddrGen& addrgen, uint32_t dst_local_l1_addr, uint32_t offset = 0, uint8_t noc = noc_index);

template <typename AddrGen, bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_page(
    const uint32_t id, const AddrGen& addrgen, uint32_t src_local_l1_addr,
    uint32_t size = 0, uint32_t offset = 0, uint8_t noc = noc_index);
```

### 3.8 Shard Operations

```cpp
template <typename DSpec>
FORCE_INLINE void noc_async_read_shard(
    const uint32_t shard_id, const TensorAccessor<DSpec>& s, std::uint32_t dst_local_l1_addr, uint8_t noc = noc_index);

template <typename DSpec, bool posted = false>
FORCE_INLINE void noc_async_write_shard(
    const uint32_t shard_id, const TensorAccessor<DSpec>& s, std::uint32_t src_local_l1_addr, uint8_t noc = noc_index);
```

### 3.9 Semaphore Operations

```cpp
template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
FORCE_INLINE uint32_t get_semaphore(uint32_t semaphore_id);

inline void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, uint8_t noc = noc_index);

inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr_multicast, uint32_t num_dests,
    bool linked = false, uint8_t noc = noc_index);

inline void noc_semaphore_set_multicast_loopback_src(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr_multicast, uint32_t num_dests,
    bool linked = false, uint8_t noc = noc_index);
```

### 3.10 Barrier Functions

```cpp
void noc_async_read_barrier(uint8_t noc = noc_index);
FORCE_INLINE void noc_async_write_barrier(uint8_t noc = noc_index);
FORCE_INLINE void noc_async_writes_flushed(uint8_t noc = noc_index);
FORCE_INLINE void noc_async_posted_writes_flushed(uint8_t noc = noc_index);
FORCE_INLINE void noc_async_atomic_barrier(uint8_t noc_idx = noc_index);
FORCE_INLINE void noc_async_full_barrier(uint8_t noc_idx = noc_index);
```

### 3.11 Local Semaphore Operations

```cpp
FORCE_INLINE void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val);
FORCE_INLINE void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val);
```

---

## 4. Compute Kernel API

### 4.1 Main Compute Kernel API (compute_kernel_api.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_api.h`

#### Activation Functions
```cpp
template <bool fast_and_approx = false> ALWI void sigmoid_tile_init();
template <int vec_mode = VectorMode::RC, bool fast_and_approx = false> ALWI void sigmoid_tile(uint32_t idst);

template <bool fast_and_approx = false> ALWI void log_tile_init();
template <bool fast_and_approx = false> ALWI void log_tile(uint32_t idst);

template <bool fast_and_approx = false> ALWI void log_with_base_tile_init();
template <bool fast_and_approx = false> ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale);

template <bool fast_and_approx = false> ALWI void tanh_tile_init();
template <bool fast_and_approx = false> ALWI void tanh_tile(uint32_t idst);

ALWI void signbit_tile_init();
ALWI void signbit_tile(uint32_t idst);
ALWI void signbit_tile_int32(uint32_t idst);

ALWI void abs_tile(uint32_t idst);
ALWI void abs_tile_init();
ALWI void abs_tile_int32(uint32_t idst);

ALWI void sign_tile(uint32_t idst);
ALWI void sign_tile_init();

ALWI void square_tile(uint32_t idst);
ALWI void square_tile_init();

ALWI void tiled_prod_tile(uint32_t idst);
ALWI void tiled_prod_tile_init();

ALWI void power_tile_init();
ALWI void power_tile(uint32_t idst, uint32_t param0);

ALWI void power_iterative_tile_init();
ALWI void power_iterative_tile(uint32_t idst, uint32_t param0);

ALWI void exp2_tile(uint32_t idst);
ALWI void exp2_tile_init();

ALWI void heaviside_tile_init();
ALWI void heaviside_tile(uint32_t idst, uint32_t param0);

template <bool approx = false> ALWI void expm1_tile(uint32_t idst);
template <bool approx = false> ALWI void expm1_tile_init();

ALWI void silu_tile(uint32_t idst);
ALWI void silu_tile_init();
```

### 4.2 Matrix Multiplication (matmul.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/compute/matmul.h`

```cpp
// Initialization
ALWI void mm_init(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id,
    const uint32_t transpose = 0, uint32_t call_line = __builtin_LINE());

ALWI void mm_init_short(
    uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0, uint32_t call_line = __builtin_LINE());

ALWI void mm_init_short_with_dt(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t c_in_old_srca, const uint32_t transpose = 0);

ALWI void mm_block_init(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id,
    const uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE());

// Operations
ALWI void matmul_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst);

template <uint32_t num_faces = 4> ALWI void matmul_tiles_math(uint32_t idst);

ALWI void matmul_block(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index,
    uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim,
    uint32_t call_line = __builtin_LINE());

// Dynamic throttling (Blackhole only)
ALWI void matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim);
```

### 4.3 Element-wise Binary Operations (eltwise_binary.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h`

```cpp
// Initialization
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t call_line = __builtin_LINE());

template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE());

ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE());
ALWI void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE());
ALWI void sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = __builtin_LINE());

// Operations
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);

// Dest reuse variants
template <EltwiseBinaryType eltwise_binary_type = ELWADD, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE());

template <EltwiseBinaryType eltwise_binary_type = ELWADD, EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index);
```

### 4.4 Reduce Operations (reduce.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/compute/reduce.h`

```cpp
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb, uint32_t call_line = __builtin_LINE());

template <bool enforce_fp32_accumulation = false>
ALWI void reduce_uninit(uint32_t icb = 0);

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst);

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4);
```

### 4.5 Pack Operations (pack.h)

**File Path:** `/tmp/tt-metal/tt_metal/hw/inc/api/compute/pack.h`

```cpp
template <bool out_of_order_output = false>
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0);

ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles);

template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t new_cb_id);

template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t old_cb_id, const uint32_t new_cb_id);

ALWI void pack_reconfig_l1_acc(const uint32_t l1_acc_en);

ALWI void pack_rows_init(uint32_t num_rows);
ALWI void pack_rows(uint32_t idst, uint32_t ocb, uint32_t output_index = 0);
ALWI void pack_rows_uninit();
```

---

## 5. Buffer API (buffer.hpp)

**File Path:** `/tmp/tt-metal/tt_metal/api/tt-metalium/buffer.hpp`

### 5.1 Structures

```cpp
struct ShardSpec {
    CoreRangeSet grid;
    std::array<uint32_t, 2> shape;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;

    uint32_t num_cores() const;
    uint32_t numel() const;
};

struct ShardSpecBuffer {
    ShardSpec tensor_shard_spec;
    std::array<uint32_t, 2> page_shape{};
    std::array<uint32_t, 2> tensor2d_shape_in_pages{};

    CoreRangeSet grid() const;
    std::array<uint32_t, 2> shape() const;
    ShardOrientation orientation() const;
    std::array<uint32_t, 2> shape_in_pages() const;
    DeviceAddr num_pages() const;
};

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved
    BufferType buffer_type;
};

using InterleavedBufferConfig = BufferConfig;

struct ShardedBufferConfig {
    IDevice* device{};
    DeviceAddr size{};
    DeviceAddr page_size{};
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};
```

### 5.2 Buffer Class

```cpp
class Buffer final : public std::enable_shared_from_this<Buffer> {
public:
    static std::shared_ptr<Buffer> create(
        IDevice* device, DeviceAddr size, DeviceAddr page_size, BufferType buffer_type,
        const BufferShardingArgs& sharding_args = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    static std::shared_ptr<Buffer> create(
        IDevice* device, DeviceAddr address, DeviceAddr size, DeviceAddr page_size,
        BufferType buffer_type, const BufferShardingArgs& sharding_args = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    std::shared_ptr<Buffer> view(const BufferRegion& region);

    IDevice* device() const;
    Allocator* allocator() const;
    DeviceAddr size() const;
    bool is_allocated() const;
    uint32_t address() const;
    DeviceAddr page_size() const;
    void set_page_size(DeviceAddr page_size);
    uint32_t num_pages() const;
    uint32_t num_dev_pages() const;
    BufferType buffer_type() const;
    HalMemType memory_type() const;
    CoreType core_type() const;
    bool is_l1() const;
    bool is_dram() const;
    bool is_trace() const;
    TensorMemoryLayout buffer_layout() const;
    bool bottom_up() const;
    DeviceAddr page_address(DeviceAddr bank_id, DeviceAddr page_index) const;
    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;
    DeviceAddr aligned_size_per_bank() const;

    // Sharded API
    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const;
    bool has_shard_spec() const;
    ShardSpecBuffer shard_spec() const;
    void set_shard_spec(const ShardSpecBuffer& shard_spec);
    std::optional<uint32_t> num_cores() const;
    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    std::shared_ptr<Buffer> root_buffer();
    BufferRegion root_buffer_region() const;
    std::optional<SubDeviceId> sub_device_id() const;
    size_t unique_id() const;
    void mark_as_deallocated();
};
```

### 5.3 Helper Functions

```cpp
bool is_sharded(const TensorMemoryLayout& layout);
UncompressedBufferPageMapping generate_buffer_page_mapping(const Buffer& buffer);

using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint8_t>>,
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void*>;
```

---

## 6. Device API (device.hpp)

**File Path:** `/tmp/tt-metal/tt_metal/api/tt-metalium/device.hpp`

### 6.1 IDevice Interface

```cpp
class IDevice {
public:
    virtual ~IDevice() = default;

    // Architecture & Identity
    virtual tt::ARCH arch() const = 0;
    virtual ChipId id() const = 0;
    virtual ChipId build_id() const = 0;
    virtual uint8_t num_hw_cqs() const = 0;
    virtual bool is_initialized() const = 0;

    // Memory Info
    virtual int num_dram_channels() const = 0;
    virtual uint32_t l1_size_per_core() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;
    virtual int get_clock_rate_mhz() const = 0;

    // Grid Info
    virtual CoreCoord grid_size() const = 0;
    virtual CoreCoord logical_grid_size() const = 0;
    virtual CoreCoord dram_grid_size() const = 0;
    virtual CoreCoord compute_with_storage_grid_size() const = 0;

    // Coordinate Conversions
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;
    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;
    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord, const CoreType& core_type) const = 0;
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) = 0;

    // Worker Cores
    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    // Allocators
    virtual const std::unique_ptr<Allocator>& allocator() const = 0;
    virtual const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const = 0;

    // DRAM
    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;
    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const = 0;

    // L1 Address Tracking
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const = 0;
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const = 0;

    // NOC Encoding
    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    // System Memory
    virtual SystemMemoryManager& sysmem_manager() = 0;

    // Trace Buffers
    virtual uint32_t get_trace_buffers_size() const = 0;
    virtual void set_trace_buffers_size(uint32_t size) = 0;

    // Initialization
    virtual bool initialize(
        uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size,
        size_t worker_l1_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) = 0;
    virtual bool close() = 0;

    // Program Cache
    virtual void enable_program_cache() = 0;
    virtual void clear_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    // Core Type & Memory
    virtual HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const = 0;
    virtual HalMemType get_mem_type_of_core(CoreCoord virtual_core) const = 0;

    // Device Addresses
    uint64_t get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;
    uint64_t get_dev_size(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;

    // NOC Transactions
    virtual bool has_noc_mcast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const = 0;

    // Sub-device Management
    virtual SubDeviceManagerId get_active_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId get_default_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void clear_loaded_sub_device_manager() = 0;
    virtual const std::vector<SubDeviceId>& get_sub_device_ids() const = 0;
    virtual const std::vector<SubDeviceId>& get_sub_device_stall_group() const = 0;
    virtual void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) = 0;
    virtual void reset_sub_device_stall_group() = 0;
    virtual uint32_t num_sub_devices() const = 0;
    virtual uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) = 0;

    // MMIO
    virtual bool is_mmio_capable() const = 0;

    // Mesh Device
    virtual std::shared_ptr<distributed::MeshDevice> get_mesh_device() = 0;

    // Ethernet Cores (Internal)
    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const = 0;
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const = 0;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const = 0;
    virtual std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;
    virtual std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const = 0;
    virtual const std::set<CoreCoord>& ethernet_cores() const = 0;
};
```

---

## 7. Program API (program.hpp)

**File Path:** `/tmp/tt-metal/tt_metal/api/tt-metalium/program.hpp`

### 7.1 Program Class

```cpp
using ProgramId = std::uint64_t;

class Program {
public:
    Program();
    explicit Program(const ProgramDescriptor& descriptor);
    ~Program() noexcept;

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;
    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    // ID related functions
    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;

    // Buffer related functions
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers() const;

    // Internal usage
    detail::ProgramImpl& impl();
    const detail::ProgramImpl& impl() const;
};
```

### 7.2 Kernel Metadata

```cpp
namespace detail {
struct KernelMeta;
std::vector<detail::KernelMeta> collect_kernel_meta(Program const& program, IDevice* device);
}
```

---

## 8. Key Constants and Macros

### 8.1 Common Values

```cpp
// From common_values.hpp (referenced)
constexpr size_t DEFAULT_L1_SMALL_SIZE = ...;
constexpr size_t DEFAULT_TRACE_REGION_SIZE = ...;
constexpr size_t DEFAULT_WORKER_L1_SIZE = ...;
constexpr uint32_t DRAM_UNRESERVED_BASE = ...;
constexpr uint32_t L1_ALIGNMENT = 16;
constexpr uint32_t NUM_CIRCULAR_BUFFERS = 32;
```

### 8.2 NOC Constants

```cpp
constexpr uint32_t NOC_MAX_BURST_SIZE = ...;
constexpr uint32_t NOC_UNICAST_WRITE_VC = ...;
constexpr uint32_t NOC_MULTICAST_WRITE_VC = ...;
```

### 8.3 Compute Kernel Macros

```cpp
#define ALWI inline __attribute__((always_inline))
#define MATH(x) ...  // Active on TRISC_MATH
#define PACK(x) ...  // Active on TRISC_PACK
#define UNPACK(x) ... // Active on TRISC_UNPACK
#define MAIN ...     // Entry point

// Architecture defines
#define ARCH_BLACKHOLE
#define ARCH_QUASAR
#define ARCH_WORMHOLE

// Math fidelity
#define MATH_FIDELITY ...
#define DST_ACCUM_MODE ...
#define APPROX ...
```

---

## 9. Enumerations

### 9.1 Buffer Types

```cpp
enum class BufferType {
    DRAM,
    L1,
    L1_SMALL,
    TRACE
};
```

### 9.2 Tensor Memory Layout

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED
};
```

### 9.3 Shard Orientation

```cpp
enum class ShardOrientation {
    ROW_MAJOR,
    COL_MAJOR
};
```

### 9.4 Core Types

```cpp
enum class CoreType {
    WORKER,
    ETH,
    PCIE,
    ARC
};
```

### 9.5 Reduce Operations

```cpp
enum PoolType {
    SUM,
    AVG,
    MAX
};

enum ReduceDim {
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR
};
```

### 9.6 Binary Operations

```cpp
enum EltwiseBinaryType {
    ELWADD,
    ELWSUB,
    ELWMUL,
    // ...
};

enum class EltwiseBinaryReuseDestType {
    NONE,
    DEST_TO_SRCA,
    DEST_TO_SRCB
};
```

### 9.7 NOC Modes

```cpp
enum NOC : uint8_t {
    NOC_0 = 0,
    NOC_1 = 1
};
```

---

## 10. Type Aliases

```cpp
using ChipId = int;
using DeviceAddr = uint64_t;
using KernelHandle = uint64_t;
using CBHandle = uintptr_t;
using SubDeviceId = uint8_t;
using SubDeviceManagerId = uint32_t;
```

---

## File Locations Summary

| API Category | File Path |
|--------------|-----------|
| Host API | `/tmp/tt-metal/tt_metal/api/tt-metalium/host_api.hpp` |
| Core API | `/tmp/tt-metal/tt_metal/api/tt-metalium/tt_metal.hpp` |
| Device | `/tmp/tt-metal/tt_metal/api/tt-metalium/device.hpp` |
| Buffer | `/tmp/tt-metal/tt_metal/api/tt-metalium/buffer.hpp` |
| Program | `/tmp/tt-metal/tt_metal/api/tt-metalium/program.hpp` |
| Dataflow API | `/tmp/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h` |
| Compute Kernel API | `/tmp/tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| Matmul | `/tmp/tt-metal/tt_metal/hw/inc/api/compute/matmul.h` |
| Eltwise Binary | `/tmp/tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h` |
| Reduce | `/tmp/tt-metal/tt_metal/hw/inc/api/compute/reduce.h` |
| Pack | `/tmp/tt-metal/tt_metal/hw/inc/api/compute/pack.h` |

---

*End of API Extraction Document*
