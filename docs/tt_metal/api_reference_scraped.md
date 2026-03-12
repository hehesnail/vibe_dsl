# TT Metal API Reference Documentation

**Source:** Tenstorrent TT-Metalium & TT-NN Documentation
**Generated:** 2026-03-12
**Documentation URLs:**
- TT-Metalium: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/
- TT-NN: https://docs.tenstorrent.com/tt-metal/latest/ttnn/

---

## Table of Contents

1. [Host API](#1-host-api)
2. [Device/Kernel API](#2-devicekernel-api)
3. [Python API (TT-NN)](#3-python-api-tt-nn)
4. [Data Movement API](#4-data-movement-api)
5. [Compute Kernel API](#5-compute-kernel-api)
6. [Circular Buffer API](#6-circular-buffer-api)
7. [NOC Semaphore API](#7-noc-semaphore-api)

---

## 1. Host API

The Host API provides interfaces for managing device operations from the host CPU, including device initialization, buffer management, command queue operations, and program execution.

### 1.1 Device Management

#### `CreateDevice`
Creates and opens a device for use.

```cpp
IDevice* tt::tt_metal::v0::CreateDevice(
    chip_id_t device_id,                    // Device ID (0 to GetNumAvailableDevices-1)
    const uint8_t num_hw_cqs = 1,           // Number of hardware command queues
    const size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {}
);
```

**Returns:** `IDevice*` - Pointer to device object

**Example:**
```cpp
constexpr int device_id = 0;
IDevice* device = CreateDevice(device_id);
// Or for mesh devices:
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
```

---

#### `CloseDevice`
Closes and cleans up a device.

```cpp
void CloseDevice(IDevice* device);
```

---

#### `QueryDevices`
Queries available devices.

```cpp
std::vector<chip_id_t> QueryDevices();
```

---

### 1.2 Buffer Management

#### `CreateBuffer` (Interleaved)
Creates an interleaved buffer in DRAM or L1 memory.

```cpp
// Version 1: Basic interleaved buffer
std::shared_ptr<Buffer> tt::tt_metal::CreateBuffer(
    const InterleavedBufferConfig& config
);

// Version 2: With specific address
std::shared_ptr<Buffer> tt::tt_metal::CreateBuffer(
    const InterleavedBufferConfig& config,
    DeviceAddr address
);

// Version 3: With sub-device ID
std::shared_ptr<Buffer> tt::tt_metal::CreateBuffer(
    const InterleavedBufferConfig& config,
    SubDeviceId sub_device_id
);
```

**Buffer Configuration Structure:**
```cpp
InterleavedBufferConfig {
    .device = device,           // IDevice pointer
    .size = buffer_size,        // Total buffer size in bytes
    .page_size = page_size,     // Page size for interleaving
    .buffer_type = BufferType::DRAM  // or BufferType::L1
};
```

**Example:**
```cpp
InterleavedBufferConfig config{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::DRAM
};
auto buffer = CreateBuffer(config);
```

---

#### `CreateBuffer` (Sharded)
Creates a sharded buffer distributed across cores.

```cpp
std::shared_ptr<Buffer> tt::tt_metal::CreateBuffer(
    const ShardedBufferConfig& config
);

std::shared_ptr<Buffer> tt::tt_metal::CreateBuffer(
    const ShardedBufferConfig& config,
    DeviceAddr address
);
```

---

#### `DeallocateBuffer`
Deallocates a buffer.

```cpp
void DeallocateBuffer(std::shared_ptr<Buffer> buffer);
```

---

### 1.3 Command Queue Operations

#### `EnqueueWriteBuffer`
Writes data from host to device buffer.

```cpp
void tt::tt_metal::EnqueueWriteBuffer(
    CommandQueue& cq,                    // Command queue reference
    std::shared_ptr<Buffer> buffer,      // Destination buffer
    const void* data,                    // Host data pointer
    bool blocking                        // Block until complete
);
```

**Example:**
```cpp
CommandQueue& cq = device->command_queue();
EnqueueWriteBuffer(cq, input_buffer, host_data.data(), false);
```

---

#### `EnqueueReadBuffer`
Reads data from device buffer to host.

```cpp
void tt::tt_metal::EnqueueReadBuffer(
    CommandQueue& cq,
    std::shared_ptr<Buffer> buffer,
    void* data,
    bool blocking
);
```

---

#### `EnqueueWriteSubBuffer`
Writes to a sub-region of a buffer.

```cpp
void EnqueueWriteSubBuffer(
    CommandQueue& cq,
    std::shared_ptr<Buffer> buffer,
    const void* data,
    bool blocking
);
```

---

#### `EnqueueReadSubBuffer`
Reads from a sub-region of a buffer.

```cpp
void EnqueueReadSubBuffer(
    CommandQueue& cq,
    std::shared_ptr<Buffer> buffer,
    void* data,
    bool blocking
);
```

---

#### `EnqueueProgram`
Enqueues a program for execution.

```cpp
void tt::tt_metal::EnqueueProgram(
    CommandQueue& cq,
    Program& program,
    bool blocking
);
```

---

#### `Finish`
Blocks until all operations in queue complete.

```cpp
void Finish(CommandQueue& cq);
```

---

#### `Synchronize`
Synchronizes device with host.

```cpp
void Synchronize(
    IDevice* device,
    uint8_t cq_id = 0,
    const std::vector<SubDeviceId>& sub_device_ids = {}
);
```

---

### 1.4 Program and Kernel Management

#### `CreateProgram`
Creates a new program object.

```cpp
Program CreateProgram();
```

---

#### `CreateKernel` (Data Movement)
Creates a data movement kernel from source file.

```cpp
KernelHandle tt::tt_metal::CreateKernel(
    Program& program,
    const std::string& kernel_file_path,
    const CoreRange& core_range,
    const DataMovementConfig& config
);
```

**DataMovementConfig Structure:**
```cpp
DataMovementConfig {
    .processor = DataMovementProcessor::RISCV_0,  // or RISCV_1
    .noc = NOC::RISCV_0_default,                   // NOC routing
    .compile_args = {arg1, arg2, ...},             // Compile-time args
    .defines = {}                                   // Preprocessor defines
};
```

**Example:**
```cpp
auto reader_id = tt_metal::CreateKernel(
    program,
    "kernels/dataflow/reader.cpp",
    core,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_args
    });
```

---

#### `CreateKernel` (Compute)
Creates a compute kernel from source file.

```cpp
KernelHandle tt::tt_metal::CreateKernel(
    Program& program,
    const std::string& kernel_file_path,
    const CoreRange& core_range,
    const ComputeConfig& config
);
```

**ComputeConfig Structure:**
```cpp
ComputeConfig {
    .math_fidelity = MathFidelity::HiFi4,
    .math_approx_mode = false,
    .compile_args = {},
    .defines = {}
};
```

**Example:**
```cpp
auto compute_id = tt_metal::CreateKernel(
    program,
    "kernels/compute/mm.cpp",
    core,
    tt_metal::ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .compile_args = compute_compile_time_args
    });
```

---

#### `SetRuntimeArgs`
Sets runtime arguments for a kernel.

```cpp
void SetRuntimeArgs(
    Program& program,
    KernelHandle kernel_id,
    const CoreCoord& core,
    const std::vector<uint32_t>& args
);
```

---

### 1.5 Circular Buffers (Host-side)

#### `CreateCircularBuffer`
Creates a circular buffer for inter-kernel communication.

```cpp
CBHandle CreateCircularBuffer(
    Program& program,
    const CoreRange& core_range,
    const CircularBufferConfig& config
);
```

---

### 1.6 Semaphores

#### `CreateSemaphore`
Creates a semaphore for synchronization.

```cpp
uint32_t CreateSemaphore(IDevice* device, uint32_t initial_value);
```

---

## 2. Device/Kernel API

The Device/Kernel API provides low-level interfaces for kernel programming on Tensix cores. Each Tensix core contains 5 baby RISC-V processors programmed using 3 user kernels.

### 2.1 Tensix Core Architecture

| Kernel Type | Count | Purpose | RISC-V Assignment |
|-------------|-------|---------|-------------------|
| **Data Movement Kernels** | 2 | Asynchronous reads/writes, DRAM/SRAM access | RISCV_0, RISCV_1 |
| **Compute Kernel** | 1 | Matrix/vector operations using FPU/SFPU | Compiled to 3 RISC-V threads |

### 2.2 Kernel Argument APIs

#### `get_arg_val`
Gets a runtime argument value.

```cpp
template<typename T>
T get_arg_val(uint32_t arg_idx);
```

**Example:**
```cpp
uint32_t n_tiles = get_arg_val<uint32_t>(0);
uint32_t src_addr = get_arg_val<uint32_t>(1);
```

---

#### `get_compile_time_arg_val`
Gets a compile-time argument value.

```cpp
template<typename T>
T get_compile_time_arg_val(uint32_t arg_idx);
```

---

## 3. Python API (TT-NN)

The TT-NN (Tenstorrent Neural Network) library provides PyTorch-like operations optimized for Tenstorrent hardware.

### 3.1 Tensor Creation Operations

| Function | Description |
|----------|-------------|
| `ttnn.arange(start, end, step)` | Creates tensor with values from start to end |
| `ttnn.empty(shape, device, dtype)` | Creates uninitialized device tensor |
| `ttnn.empty_like(tensor)` | Creates uninitialized tensor with same shape |
| `ttnn.zeros(shape, device, dtype)` | Creates tensor filled with 0.0 |
| `ttnn.zeros_like(tensor)` | Creates tensor of same shape filled with 0.0 |
| `ttnn.ones(shape, device, dtype)` | Creates tensor filled with 1.0 |
| `ttnn.ones_like(tensor)` | Creates tensor of same shape filled with 1.0 |
| `ttnn.full(shape, fill_value, device, dtype)` | Creates tensor filled with specified value |
| `ttnn.full_like(tensor, fill_value)` | Creates tensor of same shape with specified value |
| `ttnn.as_tensor(torch_tensor, dtype, layout)` | Converts torch.Tensor to ttnn.Tensor |
| `ttnn.from_torch(torch_tensor, dtype, layout)` | Converts torch.Tensor to ttnn.Tensor |

---

### 3.2 Device Operations

| Function | Description |
|----------|-------------|
| `ttnn.to_device(tensor, device, memory_config)` | Copies tensor to MeshDevice |
| `ttnn.from_device(tensor)` | Copies tensor to host |
| `ttnn.to_layout(tensor, layout)` | Converts to ROW_MAJOR_LAYOUT or TILE_LAYOUT |
| `ttnn.to_dtype(tensor, dtype)` | Converts tensor to desired dtype |
| `ttnn.to_memory_config(tensor, memory_config)` | Converts to desired memory configuration |
| `ttnn.copy(input_a, input_b)` | Copies elements from input_a to input_b |
| `ttnn.clone(tensor, memory_config, dtype)` | Clones input tensor |
| `ttnn.move(tensor, memory_config)` | Moves elements to memory with specified layout |
| `ttnn.deallocate(tensor)` | Releases resources for tensor explicitly |

---

### 3.3 Tensor Manipulation

| Function | Description |
|----------|-------------|
| `ttnn.reshape(tensor, shape)` | Reshapes tensor (0-cost view if conditions met) |
| `ttnn.permute(tensor, dims)` | Permutes dimensions according to specified order |
| `ttnn.transpose(tensor, dim0, dim1)` | Swaps two dimensions |
| `ttnn.expand(tensor, shape)` | Expands singleton dimensions to larger size |
| `ttnn.repeat(tensor, shape)` | Repeats input tensor according to shape |
| `ttnn.repeat_interleave(tensor, repeats, dim)` | Repeats elements in given dimension |
| `ttnn.chunk(tensor, chunks, dim)` | Splits tensor into multiple chunks |
| `ttnn.concat(tensors, dim)` | Concatenates input tensors |
| `ttnn.split(tensor, split_size, dim)` | Splits tensor along dimension |
| `ttnn.slice(tensor, starts, ends, steps)` | Returns a sliced tensor |
| `ttnn.pad(tensor, padding, value)` | Returns padded tensor |
| `ttnn.squeeze(tensor, dim)` | Removes dimensions of size 1 |
| `ttnn.unsqueeze(tensor, dim)` | Adds dimension of size 1 |
| `ttnn.gather(tensor, dim, indices)` | Extracts values based on indices |
| `ttnn.tilize(tensor)` | Changes data layout to TILE |
| `ttnn.untilize(tensor)` | Changes data layout to ROW_MAJOR |

---

### 3.4 Pointwise Unary Operations

| Function | Description |
|----------|-------------|
| `ttnn.abs(tensor)` | Element-wise absolute value |
| `ttnn.neg(tensor)` | Element-wise negation |
| `ttnn.sign(tensor)` | Element-wise sign |
| `ttnn.exp(tensor)` | Element-wise exponential |
| `ttnn.exp2(tensor)` | Element-wise base-2 exponential |
| `ttnn.expm1(tensor)` | Element-wise exp(x) - 1 |
| `ttnn.log(tensor)` | Element-wise natural logarithm |
| `ttnn.log2(tensor)` | Element-wise base-2 logarithm |
| `ttnn.log10(tensor)` | Element-wise base-10 logarithm |
| `ttnn.log1p(tensor)` | Element-wise log(1 + x) |
| `ttnn.sqrt(tensor)` | Element-wise square root |
| `ttnn.rsqrt(tensor)` | Element-wise reciprocal square root |
| `ttnn.cbrt(tensor)` | Element-wise cube root |
| `ttnn.sin(tensor)` | Element-wise sine |
| `ttnn.cos(tensor)` | Element-wise cosine |
| `ttnn.tan(tensor)` | Element-wise tangent |
| `ttnn.asin(tensor)` | Element-wise arcsine |
| `ttnn.acos(tensor)` | Element-wise arccosine |
| `ttnn.atan(tensor)` | Element-wise arctangent |
| `ttnn.sinh(tensor)` | Element-wise hyperbolic sine |
| `ttnn.cosh(tensor)` | Element-wise hyperbolic cosine |
| `ttnn.tanh(tensor)` | Element-wise hyperbolic tangent |
| `ttnn.asinh(tensor)` | Element-wise inverse hyperbolic sine |
| `ttnn.acosh(tensor)` | Element-wise inverse hyperbolic cosine |
| `ttnn.atanh(tensor)` | Element-wise inverse hyperbolic tangent |
| `ttnn.relu(tensor)` | ReLU activation |
| `ttnn.relu6(tensor)` | ReLU6 activation |
| `ttnn.leaky_relu(tensor, negative_slope)` | Leaky ReLU activation |
| `ttnn.prelu(tensor, weight)` | PReLU activation |
| `ttnn.elu(tensor, alpha)` | ELU activation |
| `ttnn.selu(tensor)` | SELU activation |
| `ttnn.celu(tensor, alpha)` | CELU activation |
| `ttnn.sigmoid(tensor)` | Sigmoid activation |
| `ttnn.silu(tensor)` | SiLU/Swish activation |
| `ttnn.mish(tensor)` | Mish activation |
| `ttnn.gelu(tensor)` | GELU activation |
| `ttnn.hardswish(tensor)` | Hard Swish activation |
| `ttnn.hardsigmoid(tensor)` | Hard Sigmoid activation |
| `ttnn.hardtanh(tensor, min_val, max_val)` | Hard Tanh activation |
| `ttnn.floor(tensor)` | Element-wise floor |
| `ttnn.ceil(tensor)` | Element-wise ceiling |
| `ttnn.round(tensor)` | Element-wise round |
| `ttnn.trunc(tensor)` | Element-wise truncate |
| `ttnn.isfinite(tensor)` | Element-wise finite check |
| `ttnn.isinf(tensor)` | Element-wise infinity check |
| `ttnn.isnan(tensor)` | Element-wise NaN check |
| `ttnn.reciprocal(tensor)` | Element-wise reciprocal |
| `ttnn.square(tensor)` | Element-wise square |

---

### 3.5 Pointwise Binary Operations

| Function | Description |
|----------|-------------|
| `ttnn.add(input_a, input_b)` | Element-wise addition |
| `ttnn.sub(input_a, input_b)` | Element-wise subtraction |
| `ttnn.mul(input_a, input_b)` | Element-wise multiplication |
| `ttnn.div(input_a, input_b)` | Element-wise division |
| `ttnn.pow(input_a, exponent)` | Element-wise power |
| `ttnn.atan2(input_a, input_b)` | Element-wise arctangent of input_a/input_b |
| `ttnn.eq(input_a, input_b)` | Element-wise equal comparison |
| `ttnn.ne(input_a, input_b)` | Element-wise not-equal comparison |
| `ttnn.lt(input_a, input_b)` | Element-wise less-than comparison |
| `ttnn.le(input_a, input_b)` | Element-wise less-or-equal comparison |
| `ttnn.gt(input_a, input_b)` | Element-wise greater-than comparison |
| `ttnn.ge(input_a, input_b)` | Element-wise greater-or-equal comparison |
| `ttnn.logical_and(input_a, input_b)` | Element-wise logical AND |
| `ttnn.logical_or(input_a, input_b)` | Element-wise logical OR |
| `ttnn.logical_xor(input_a, input_b)` | Element-wise logical XOR |
| `ttnn.logical_not(tensor)` | Element-wise logical NOT |
| `ttnn.bitwise_and(input_a, input_b)` | Element-wise bitwise AND |
| `ttnn.bitwise_or(input_a, input_b)` | Element-wise bitwise OR |
| `ttnn.bitwise_xor(input_a, input_b)` | Element-wise bitwise XOR |
| `ttnn.bitwise_not(tensor)` | Element-wise bitwise NOT |
| `ttnn.bitwise_left_shift(tensor, shift)` | Element-wise left shift |
| `ttnn.bitwise_right_shift(tensor, shift)` | Element-wise right shift |
| `ttnn.minimum(input_a, input_b)` | Element-wise minimum |
| `ttnn.maximum(input_a, input_b)` | Element-wise maximum |
| `ttnn.outer(input_a, input_b)` | Outer product |
| `ttnn.scatter(input, dim, index, src)` | Scatter operation |

---

### 3.6 Pointwise Ternary Operations

| Function | Description |
|----------|-------------|
| `ttnn.addcdiv(input, tensor1, tensor2, value)` | input + value * tensor1 / tensor2 |
| `ttnn.addcmul(input, tensor1, tensor2, value)` | input + value * tensor1 * tensor2 |
| `ttnn.lerp(input, end, weight)` | Linear interpolation |
| `ttnn.mac(input, tensor1, tensor2)` | Multiply-accumulate |
| `ttnn.where(condition, input, other)` | Conditional selection |

---

### 3.7 Matrix Operations

| Function | Description |
|----------|-------------|
| `ttnn.matmul(input_a, input_b)` | Matrix multiplication |
| `ttnn.linear(input, weight, bias)` | Linear transformation (matmul + bias) |
| `ttnn.bmm(input_a, input_b)` | Batch matrix multiplication |

---

### 3.8 Reduction Operations

| Function | Description |
|----------|-------------|
| `ttnn.sum(tensor, dim, keepdim)` | Sum reduction |
| `ttnn.mean(tensor, dim, keepdim)` | Mean reduction |
| `ttnn.max(tensor, dim, keepdim)` | Max reduction |
| `ttnn.min(tensor, dim, keepdim)` | Min reduction |
| `ttnn.argmax(tensor, dim, keepdim)` | Index of maximum value |
| `ttnn.argmin(tensor, dim, keepdim)` | Index of minimum value |
| `ttnn.std(tensor, dim, keepdim)` | Standard deviation |
| `ttnn.var(tensor, dim, keepdim)` | Variance |

---

### 3.9 Normalization Operations

| Function | Description |
|----------|-------------|
| `ttnn.layer_norm(tensor, weight, bias, epsilon)` | Layer normalization |
| `ttnn.batch_norm(tensor, weight, bias, running_mean, running_var, training, momentum, epsilon)` | Batch normalization |
| `ttnn.group_norm(tensor, num_groups, weight, bias, epsilon)` | Group normalization |
| `ttnn.rms_norm(tensor, weight, epsilon)` | RMS normalization |
| `ttnn.normalize_global(tensor, eps)` | Global normalization |
| `ttnn.normalize_hw(tensor, eps)` | Height/width normalization |

---

### 3.10 Convolution Operations

| Function | Description |
|----------|-------------|
| `ttnn.conv1d(input, weight, bias, stride, padding, dilation, groups)` | 1D convolution |
| `ttnn.conv2d(input, weight, bias, stride, padding, dilation, groups)` | 2D convolution |
| `ttnn.conv_transpose2d(input, weight, bias, stride, padding, output_padding, dilation, groups)` | 2D transposed convolution |
| `ttnn.prepare_conv_weights(weights, input_memory_config, weights_memory_config, bias_memory_config, device)` | Preprocesses weights for Conv2D |
| `ttnn.prepare_conv_bias(bias, input_memory_config, weights_memory_config, bias_memory_config, device)` | Preprocesses bias for Conv2D |

---

### 3.11 Pooling Operations

| Function | Description |
|----------|-------------|
| `ttnn.max_pool2d(input, kernel_size, stride, padding, dilation, return_indices)` | 2D max pooling |
| `ttnn.avg_pool2d(input, kernel_size, stride, padding)` | 2D average pooling |
| `ttnn.global_avg_pool2d(input)` | Global 2D adaptive average pooling |
| `ttnn.upsample(input, scale_factor, mode)` | Upsamples multi-channel 2D data |

---

### 3.12 Transformer Operations

| Function | Description |
|----------|-------------|
| `ttnn.transformer.split_query_key_value_and_split_heads(input, num_heads)` | Splits QKV for attention |
| `ttnn.transformer.concatenate_heads(input)` | Concatenates attention heads |
| `ttnn.transformer.attention_softmax(input, mask)` | Attention softmax with optional mask |
| `ttnn.transformer.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)` | Causal scaled dot product attention |
| `ttnn.transformer.scaled_dot_product_attention_decode(query, key, value, attn_mask)` | Decode-optimized attention |
| `ttnn.experimental.rotary_embedding(input, cos_cached, sin_cached, token_idx)` | Applies rotary embeddings |

---

### 3.13 Collective Communication (CCL) Operations

| Function | Description |
|----------|-------------|
| `ttnn.all_gather(tensor, dim, num_links, memory_config)` | All-gather across devices |
| `ttnn.reduce_scatter(tensor, dim, math_op, num_links, memory_config)` | Reduce-scatter across devices |
| `ttnn.experimental.all_reduce(tensor, math_op, num_links, memory_config)` | All-reduce across devices |

---

### 3.14 Embedding Operations

| Function | Description |
|----------|-------------|
| `ttnn.embedding(input, weight)` | Embedding lookup |
| `ttnn.embedding_bw(input, weight, grad_output)` | Embedding backward pass |

---

### 3.15 Data Movement / Sharding Operations

| Function | Description |
|----------|-------------|
| `ttnn.interleaved_to_sharded(tensor, sharded_memory_config)` | Converts interleaved to sharded memory |
| `ttnn.sharded_to_interleaved(tensor, memory_config)` | Converts sharded to interleaved memory |
| `ttnn.fold(tensor, stride_h, stride_w)` | Fold TT Tensor |
| `ttnn.untilize_with_halo_v2(tensor, padding_config, ncores_nhw)` | Untilize with halo for sharding |

---

## 4. Data Movement API

The Data Movement API provides functions for transferring data between DRAM, L1 SRAM, and between Tensix cores via the NOC (Network-on-Chip).

### 4.1 NOC Asynchronous Read

#### `noc_async_read`
Initiates an asynchronous read from a source node to local L1 memory.

```cpp
template<uint32_t max_page_size=NOC_MAX_BURST_SIZE+1, bool enable_noc_tracing=true>
inline void noc_async_read(
    uint64_t src_noc_addr,           // Source NOC address (x,y)+address
    uint32_t dst_local_l1_addr,      // Destination in local L1 (0..1MB)
    uint32_t size,                   // Transfer size in bytes (0..1MB)
    uint8_t noc=noc_index,           // Which NOC to use (0 or 1)
    uint32_t read_req_vc=NOC_UNICAST_WRITE_VC  // Virtual channel
);
```

**Parameters:**
| Parameter | Description | Type | Valid Range |
|-----------|-------------|------|-------------|
| `src_noc_addr` | NOC address encoding via `get_noc_addr()` | `uint64_t` | Valid NOC coordinates |
| `dst_local_l1_addr` | Destination in local L1 memory | `uint32_t` | 0..1MB |
| `size` | Data transfer size | `uint32_t` | 0..1MB |
| `noc` | NOC to use | `uint8_t` | 0 or 1 |
| `max_page_size` | Max transaction size (template) | `uint32_t` | Configurable |

---

### 4.2 NOC Asynchronous Write

#### `noc_async_write`
Initiates an asynchronous write from local L1 to a destination node.

```cpp
template<uint32_t max_page_size=NOC_MAX_BURST_SIZE+1>
inline void noc_async_write(
    uint32_t src_local_l1_addr,      // Source in local L1
    uint64_t dst_noc_addr,           // Destination NOC address
    uint32_t size,                   // Transfer size in bytes
    uint8_t noc=noc_index            // Which NOC to use
);
```

---

#### `noc_async_write_multicast`
Initiates an asynchronous multicast write to multiple destinations.

```cpp
void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,
    bool linked,
    uint8_t noc = noc_index
);
```

---

### 4.3 NOC Addressing

#### `get_noc_addr`
Encodes NOC coordinates and address into a 64-bit NOC address.

```cpp
uint64_t get_noc_addr(
    uint32_t x,                      // X coordinate
    uint32_t y,                      // Y coordinate
    uint32_t addr                    // Local address
);
```

**Example:**
```cpp
uint64_t src_noc_addr = get_noc_addr(src_x, src_y, src_addr);
noc_async_read(src_noc_addr, dst_l1_addr, size);
```

---

#### `get_noc_addr_from_bank_id`
Gets NOC address from bank ID and offset.

```cpp
uint64_t get_noc_addr_from_bank_id(
    uint32_t bank_id,
    uint32_t offset
);
```

---

#### `get_noc_multicast_addr`
Gets multicast NOC address for a range of cores.

```cpp
uint64_t get_noc_multicast_addr(
    uint32_t x_start,
    uint32_t y_start,
    uint32_t x_end,
    uint32_t y_end,
    uint32_t addr
);
```

---

### 4.4 Synchronization Barriers

#### `noc_async_read_barrier`
Waits for all read operations to complete.

```cpp
void noc_async_read_barrier();
```

---

#### `noc_async_write_barrier`
Waits for all write operations to complete.

```cpp
void noc_async_write_barrier();
```

---

#### `noc_async_full_barrier`
Waits for all NOC operations (reads and writes) to complete.

```cpp
void noc_async_full_barrier();
```

---

### 4.5 Typical Data Movement Pattern

**Reader Kernel:**
```cpp
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);
        uint64_t src_noc_addr = get_noc_addr(src_x, src_y, src_addr + i * tile_size);
        noc_async_read(src_noc_addr, l1_write_addr, tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
```

**Writer Kernel:**
```cpp
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);
        uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, dst_addr + i * tile_size);
        noc_async_write(l1_read_addr, dst_noc_addr, tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
```

---

## 5. Compute Kernel API

The Compute Kernel API provides functions for performing matrix and vector operations on the Tensix compute engine.

### 5.1 Tile Register Management

#### `tile_regs_acquire`
Acquires Dst registers for unpacker and math core; initializes them to zero.

```cpp
void tile_regs_acquire();
```

**Note:** `acquire_dst()` is deprecated; use `tile_regs_acquire()` instead.

---

#### `tile_regs_commit`
Signals math core is done; transfers ownership to packer.

```cpp
void tile_regs_commit();
```

---

#### `tile_regs_wait`
Waits for packer to be ready to access Dst registers.

```cpp
void tile_regs_wait();
```

---

#### `tile_regs_release`
Releases Dst registers for next iteration.

```cpp
void tile_regs_release();
```

**Note:** `release_dst()` is deprecated; use `tile_regs_release()` instead.

---

### 5.2 Tile Movement Operations

#### `copy_tile`
Copies a single tile from input CB to DST register.

```cpp
void ckernel::copy_tile(
    uint32_t in_cb_id,           // Source circular buffer ID (0-31)
    uint32_t in_tile_index,      // Tile index in input CB
    uint32_t dst_tile_index      // Destination tile index in DST REG
);
```

**Requirements:**
- Must call `cb_wait_front(n)` first
- DST register buffer must be in acquired state via `tile_regs_acquire()`
- Only available on compute engine
- Blocking call

---

#### `pack_tile`
Packs tile from DST register to output circular buffer.

```cpp
void pack_tile(
    uint32_t src_dst_idx,        // Source index in DST register
    uint32_t out_cb_id           // Output circular buffer ID
);
```

---

### 5.3 Matrix Operations

#### `matmul_tiles`
Performs tile-sized matrix multiplication C = A x B and accumulates to DST.

```cpp
void ckernel::matmul_tiles(
    uint32_t in0_cb_id,          // First input CB (0-31)
    uint32_t in1_cb_id,          // Second input CB (0-31)
    uint32_t in0_tile_index,     // Tile index in first CB
    uint32_t in1_tile_index,     // Tile index in second CB
    uint32_t dst_tile_index      // Destination tile index in DST REG
);
```

**Notes:**
- Blocking call - only available on compute engine
- Requires `tile_regs_acquire()` called first
- Use `mm_init(cb_in0, cb_in1, cb_out)` before calling
- Accumulates to DST (DST += C)

---

### 5.4 Arithmetic Operations

#### `add_tiles`
Element-wise addition of two tiles.

```cpp
void add_tiles(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dst_tile_index
);
```

---

#### `sub_tiles`
Element-wise subtraction of two tiles.

```cpp
void sub_tiles(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dst_tile_index
);
```

---

#### `mul_tiles`
Element-wise multiplication of two tiles.

```cpp
void mul_tiles(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dst_tile_index
);
```

---

### 5.5 Broadcast Operations

| Function | Description |
|----------|-------------|
| `add_tiles_bcast` | Addition with broadcasting |
| `add_tiles_bcast_rows` | Addition broadcasting rows |
| `add_tiles_bcast_cols` | Addition broadcasting columns |
| `sub_tiles_bcast` | Subtraction with broadcasting |
| `sub_tiles_bcast_rows` | Subtraction broadcasting rows |
| `sub_tiles_bcast_cols` | Subtraction broadcasting columns |
| `mul_tiles_bcast` | Multiplication with broadcasting |
| `mul_tiles_bcast_rows` | Multiplication broadcasting rows |
| `mul_tiles_bcast_cols` | Multiplication broadcasting columns |

---

### 5.6 SFPU Operations

SFPU (Special Function Processing Unit) operations for vector math:

| Function | Description |
|----------|-------------|
| `exp_tile_init()` / `exp_tile()` | Exponential |
| `log_tile_init()` / `log_tile()` | Natural logarithm |
| `sqrt_tile_init()` / `sqrt_tile()` | Square root |
| `recip_tile_init()` / `recip_tile()` | Reciprocal |
| `sin_tile_init()` / `sin_tile()` | Sine |
| `cos_tile_init()` / `cos_tile()` | Cosine |
| `tanh_tile_init()` / `tanh_tile()` | Hyperbolic tangent |
| `gelu_tile_init()` / `gelu_tile()` | GELU activation |
| `relu_tile_init()` / `relu_tile()` | ReLU activation |
| `sigmoid_tile_init()` / `sigmoid_tile()` | Sigmoid activation |

---

### 5.7 Typical Compute Kernel Pattern

```cpp
#include "compute_kernel_api.h"

void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Initialize operations
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait for input data
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Acquire Dst registers
        tile_regs_acquire();

        // Copy tiles to DST
        copy_tile(cb_in0, 0, 0);
        copy_tile(cb_in1, 0, 1);

        // Perform computation
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

        // Commit results
        tile_regs_commit();

        // Reserve output space
        cb_reserve_back(cb_out, 1);

        // Wait for packer and pack results
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        // Push output and consume inputs
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
```

---

## 6. Circular Buffer API

Circular buffers act as limited capacity double-ended queues with a single producer pushing tiles to the back and a single consumer popping tiles off the front.

### 6.1 Core Circular Buffer Functions

#### `cb_wait_front`
Blocks until at least `num_tiles` are available at the front of the circular buffer.

```cpp
void cb_wait_front(uint32_t cb_id, uint32_t num_tiles);
```

**Parameters:**
| Parameter | Description | Range |
|-----------|-------------|-------|
| `cb_id` | Circular buffer ID | 0-31 |
| `num_tiles` | Number of tiles to wait for | 1 to CB size |

---

#### `cb_reserve_back`
Reserves space for `num_tiles` at the back of the circular buffer.

```cpp
void cb_reserve_back(uint32_t cb_id, uint32_t num_tiles);
```

---

#### `cb_push_back`
Pushes `num_tiles` to the back of the circular buffer (makes them available to consumer).

```cpp
void cb_push_back(uint32_t cb_id, uint32_t num_tiles);
```

---

#### `cb_pop_front`
Pops/consumes `num_tiles` from the front of the circular buffer.

```cpp
void cb_pop_front(uint32_t cb_id, uint32_t num_tiles);
```

---

### 6.2 Pointer Access Functions

#### `get_read_ptr`
Returns the SRAM address for reading from the front of the circular buffer.

```cpp
uint32_t get_read_ptr(uint32_t cb_id);
```

**Returns:** L1 memory address for reading

---

#### `get_write_ptr`
Returns the SRAM address for writing to the back of the circular buffer.

```cpp
uint32_t get_write_ptr(uint32_t cb_id);
```

**Returns:** L1 memory address for writing

---

### 6.3 Query Functions

#### `cb_pages_available_at_front`
Returns the number of pages available to read from the front.

```cpp
uint32_t cb_pages_available_at_front(uint32_t cb_id);
```

---

#### `cb_pages_reservable_at_back`
Returns the number of pages that can be reserved at the back.

```cpp
uint32_t cb_pages_reservable_at_back(uint32_t cb_id);
```

---

## 7. NOC Semaphore API

NOC semaphores provide synchronization primitives for coordinating between Tensix cores.

### 7.1 Semaphore Operations

#### `noc_semaphore_set`
Sets the value of a local L1 memory semaphore.

```cpp
void noc_semaphore_set(
    volatile uint32_t* sem_addr,     // Semaphore address in local L1
    uint32_t val                     // Value to set
);
```

**Parameters:**
| Parameter | Description | Type | Valid Range |
|-----------|-------------|------|-------------|
| `sem_addr` | Semaphore address in local L1 | `uint32_t*` | 0..1MB |
| `val` | Value to set | `uint32_t` | Any uint32_t |

---

#### `noc_semaphore_inc`
Initiates an atomic increment of a remote Tensix core's L1 memory address.

```cpp
template<bool posted=false>
void noc_semaphore_inc(
    uint64_t addr,                   // Destination (x,y)+address
    uint32_t incr,                   // Value to increment by
    uint8_t noc_id=noc_index,        // Which NOC to use
    uint8_t vc=NOC_UNICAST_WRITE_VC  // Virtual channel
);
```

**Notes:**
- Performs atomic increment with 32-bit wraparound
- Used for synchronization between cores

---

#### `noc_semaphore_wait`
Blocking call that waits until a local L1 semaphore value equals a target value.

```cpp
void noc_semaphore_wait(
    volatile uint32_t* sem_addr,     // Semaphore address
    uint32_t val                     // Target value to wait for
);
```

---

#### `noc_semaphore_set_remote`
Asynchronously writes 4 bytes to set a remote semaphore.

```cpp
void noc_semaphore_set_remote(
    uint32_t val,
    uint8_t noc,
    uint64_t addr
);
```

---

#### `noc_semaphore_set_multicast_loopback_src`
Sets semaphore across multiple destination cores.

```cpp
void noc_semaphore_set_multicast_loopback_src(
    uint32_t val,
    uint8_t noc,
    uint64_t addr,
    uint32_t num_dests
);
```

---

### 7.2 Typical Semaphore Usage Pattern

**Producer Core:**
```cpp
// Signal completion to consumer
uint64_t consumer_sem_addr = get_noc_addr(consumer_x, consumer_y, sem_addr);
noc_semaphore_inc(consumer_sem_addr, 1);
```

**Consumer Core:**
```cpp
// Wait for producer signal
volatile uint32_t* local_sem = (volatile uint32_t*)sem_addr;
noc_semaphore_wait(local_sem, expected_value);
// Reset for next iteration
noc_semaphore_set(local_sem, 0);
```

---

## Appendix: Complete Working Example

Here's a complete example showing how all APIs work together:

### Host Code
```cpp
#include <tt_metal/host_api.hpp>

int main() {
    // 1. Device initialization
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();

    // 2. Create program
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 3. Create buffers
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = buffer_size,
        .page_size = tile_size,
        .buffer_type = BufferType::DRAM
    };
    auto src_buffer = CreateBuffer(dram_config);
    auto dst_buffer = CreateBuffer(dram_config);

    // 4. Create circular buffers
    CircularBufferConfig cb_config = CircularBufferConfig(
        num_tiles * tile_size,
        {{tt::DataFormat::Float16_b, tt::Tile{32, 32}}}
    );
    auto cb = CreateCircularBuffer(program, core, cb_config);

    // 5. Create kernels
    auto reader_id = CreateKernel(
        program,
        "kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    auto writer_id = CreateKernel(
        program,
        "kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    auto compute_id = CreateKernel(
        program,
        "kernels/compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    // 6. Set runtime args
    SetRuntimeArgs(program, reader_id, core, {src_addr, num_tiles});
    SetRuntimeArgs(program, writer_id, core, {dst_addr, num_tiles});
    SetRuntimeArgs(program, compute_id, core, {num_tiles});

    // 7. Execute
    EnqueueWriteBuffer(cq, src_buffer, host_data.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_buffer, output.data(), true);

    // 8. Cleanup
    CloseDevice(device);
    return 0;
}
```

### Reader Kernel (RISCV_1)
```cpp
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_addr = get_write_ptr(cb_in0);
        uint64_t src_noc = get_noc_addr(src_x, src_y, src_addr);
        noc_async_read(src_noc, l1_addr, tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
```

### Compute Kernel
```cpp
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();
        copy_tile(cb_in0, 0, 0);
        // ... computation ...
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
    }
}
```

### Writer Kernel (RISCV_0)
```cpp
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        uint64_t dst_noc = get_noc_addr(dst_x, dst_y, dst_addr);
        noc_async_write(l1_addr, dst_noc, tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
```

---

## References

- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
- [Host APIs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis.html)
- [Kernel APIs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis.html)
- [Data Movement APIs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/data_movement/data_movement.html)
- [Compute APIs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/compute.html)
- [Circular Buffer APIs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html)
- [TT-NN API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html)
