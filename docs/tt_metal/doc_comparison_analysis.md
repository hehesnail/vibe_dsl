# TT Metalium 文档对比分析报告

**分析日期**: 2026-03-12
**对比文档**:
1. 现有文档: `/root/dev/vibe_dsl/TT_Metal_Documentation_Summary.md` (约930行)
2. 官方文档: `/root/dev/vibe_dsl/docs/tt_metal/api_reference_scraped.md`

---

## 1. 执行摘要

通过对比现有文档与官方文档，发现以下关键差异：

- **官方文档新增 API**: 约 120+ 个函数/操作
- **现有文档缺失内容**: Python TT-NN API 几乎完全缺失
- **参数差异**: 部分 C++ API 参数定义不完整
- **弃用提示**: 发现若干 API 弃用说明

---

## 2. 逐章节对比分析

### 2.1 Host API 对比

#### 设备管理

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `CreateDevice` | 已覆盖 (简化版) | 已覆盖 (完整版) | 现有文档缺少完整参数列表: `num_hw_cqs`, `l1_small_size`, `trace_region_size`, `dispatch_core_config`, `l1_bank_remap` | 高 |
| `CloseDevice` | 已覆盖 | 已覆盖 | 一致 | - |
| `QueryDevices` | **缺失** | 新增 | 官方文档新增 API，用于查询可用设备 | 中 |
| `device->command_queue()` | 已覆盖 | 已覆盖 | 一致 | - |

#### Buffer 管理

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `CreateBuffer` (Interleaved) | 已覆盖 (基础版) | 已覆盖 (3个重载) | 现有文档缺少带 `address` 和 `sub_device_id` 参数的重载版本 | 中 |
| `CreateBuffer` (Sharded) | **缺失** | 新增 | 官方文档新增分片 Buffer 创建 API | 高 |
| `DeallocateBuffer` | **缺失** | 新增 | 官方文档新增 Buffer 释放 API | 中 |

#### 命令队列操作

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `EnqueueWriteBuffer` | **缺失** | 新增 | 官方文档新增 Host 到 Device 数据传输 API | 高 |
| `EnqueueReadBuffer` | **缺失** | 新增 | 官方文档新增 Device 到 Host 数据传输 API | 高 |
| `EnqueueWriteSubBuffer` | **缺失** | 新增 | 官方文档新增子 Buffer 写入 API | 中 |
| `EnqueueReadSubBuffer` | **缺失** | 新增 | 官方文档新增子 Buffer 读取 API | 中 |
| `EnqueueProgram` | 已覆盖 (简化版) | 已覆盖 (完整版) | 参数一致 | - |
| `Finish` | 已覆盖 | 已覆盖 | 一致 | - |
| `Synchronize` | **缺失** | 新增 | 官方文档新增设备同步 API，支持 sub-device | 中 |

#### 程序与 Kernel 管理

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `CreateProgram` | **缺失** | 新增 | 官方文档明确列出此 API | 低 |
| `CreateKernel` (Data Movement) | 已覆盖 | 已覆盖 | 官方文档显示 `DataMovementConfig` 还包含 `.defines` 字段 | 低 |
| `CreateKernel` (Compute) | 已覆盖 | 已覆盖 | 官方文档显示 `ComputeConfig` 还包含 `.math_approx_mode` 和 `.defines` 字段 | 低 |
| `SetRuntimeArgs` | 已覆盖 | 已覆盖 | 一致 | - |

#### Circular Buffer (Host 端)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `CreateCircularBuffer` | 已覆盖 | 已覆盖 | 官方文档参数更完整，显示 `CoreRange` 支持 | 低 |

#### 信号量

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `CreateSemaphore` | **缺失** | 新增 | 官方文档新增 Host 端信号量创建 API | 中 |

---

### 2.2 Device/Kernel API 对比

#### Kernel 参数 API

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `get_arg_val` | 已覆盖 | 已覆盖 | 一致 | - |
| `get_compile_time_arg_val` | **缺失** | 新增 | 官方文档新增编译时参数获取 API | 中 |

---

### 2.3 Python API (TT-NN) 对比

**整体评估**: 现有文档几乎完全缺失 TT-NN Python API 的详细覆盖。

#### 张量创建操作 (3.1节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.arange` | **缺失** | 新增 | 创建序列张量 | 中 |
| `ttnn.empty` / `empty_like` | **缺失** | 新增 | 创建未初始化张量 | 中 |
| `ttnn.zeros` / `zeros_like` | **缺失** | 新增 | 创建零张量 | 中 |
| `ttnn.ones` / `ones_like` | **缺失** | 新增 | 创建一张量 | 中 |
| `ttnn.full` / `full_like` | **缺失** | 新增 | 创建填充张量 | 中 |
| `ttnn.as_tensor` / `from_torch` | **缺失** | 新增 | PyTorch 张量转换 | 高 |

#### 设备操作 (3.2节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.to_device` | **缺失** | 新增 | 张量上传到设备 | 高 |
| `ttnn.from_device` | **缺失** | 新增 | 张量下载到主机 | 高 |
| `ttnn.to_layout` | **缺失** | 新增 | 布局转换 (ROW_MAJOR/TILE) | 高 |
| `ttnn.to_dtype` | **缺失** | 新增 | 数据类型转换 | 中 |
| `ttnn.to_memory_config` | **缺失** | 新增 | 内存配置转换 | 中 |
| `ttnn.copy` / `clone` / `move` | **缺失** | 新增 | 张量复制/移动 | 中 |
| `ttnn.deallocate` | **缺失** | 新增 | 显式释放张量 | 中 |

#### 张量操作 (3.3节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.reshape` | **缺失** | 新增 | 重塑张量形状 | 高 |
| `ttnn.permute` | **缺失** | 新增 | 维度置换 | 高 |
| `ttnn.transpose` | **缺失** | 新增 | 转置 | 高 |
| `ttnn.expand` / `repeat` | **缺失** | 新增 | 扩展/重复张量 | 中 |
| `ttnn.chunk` / `concat` / `split` | **缺失** | 新增 | 分割/连接 | 中 |
| `ttnn.slice` / `pad` | **缺失** | 新增 | 切片/填充 | 中 |
| `ttnn.squeeze` / `unsqueeze` | **缺失** | 新增 | 维度增删 | 高 |
| `ttnn.gather` | **缺失** | 新增 | 索引收集 | 中 |
| `ttnn.tilize` / `untilize` | 已覆盖 (C++) | 新增 (Python) | Python API 版本 | 中 |

#### 逐元素一元操作 (3.4节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.abs` / `neg` / `sign` | **缺失** | 新增 | 基础数学运算 | 中 |
| `ttnn.exp` / `exp2` / `expm1` | **缺失** | 新增 | 指数运算 | 中 |
| `ttnn.log` / `log2` / `log10` / `log1p` | **缺失** | 新增 | 对数运算 | 中 |
| `ttnn.sqrt` / `rsqrt` / `cbrt` | **缺失** | 新增 | 根运算 | 中 |
| `ttnn.sin` / `cos` / `tan` | **缺失** | 新增 | 三角函数 | 中 |
| `ttnn.asin` / `acos` / `atan` | **缺失** | 新增 | 反三角函数 | 低 |
| `ttnn.sinh` / `cosh` / `tanh` | **缺失** | 新增 | 双曲函数 | 低 |
| `ttnn.asinh` / `acosh` / `atanh` | **缺失** | 新增 | 反双曲函数 | 低 |
| `ttnn.relu` / `relu6` | 已覆盖 (C++) | 新增 (Python) | Python API 版本 | 高 |
| `ttnn.leaky_relu` / `prelu` / `elu` | **缺失** | 新增 | ReLU 变体 | 中 |
| `ttnn.selu` / `celu` | **缺失** | 新增 | SELU/CELU | 低 |
| `ttnn.sigmoid` / `silu` / `mish` | 部分覆盖 | 新增 | Python API 版本 | 中 |
| `ttnn.gelu` | 已覆盖 (C++) | 新增 (Python) | Python API 版本 | 高 |
| `ttnn.hardswish` / `hardsigmoid` / `hardtanh` | **缺失** | 新增 | Hard 激活函数 | 低 |
| `ttnn.floor` / `ceil` / `round` / `trunc` | **缺失** | 新增 | 取整运算 | 中 |
| `ttnn.isfinite` / `isinf` / `isnan` | **缺失** | 新增 | 数值检查 | 低 |
| `ttnn.reciprocal` / `square` | **缺失** | 新增 | 倒数/平方 | 中 |

#### 逐元素二元操作 (3.5节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.add` / `sub` / `mul` / `div` | 已覆盖 (C++) | 新增 (Python) | Python API 版本 | 高 |
| `ttnn.pow` / `atan2` | **缺失** | 新增 | 幂/反正切2 | 中 |
| `ttnn.eq` / `ne` / `lt` / `le` / `gt` / `ge` | **缺失** | 新增 | 比较运算 | 高 |
| `ttnn.logical_and` / `or` / `xor` / `not` | **缺失** | 新增 | 逻辑运算 | 中 |
| `ttnn.bitwise_and` / `or` / `xor` / `not` | **缺失** | 新增 | 位运算 | 中 |
| `ttnn.bitwise_left_shift` / `right_shift` | **缺失** | 新增 | 位移运算 | 低 |
| `ttnn.minimum` / `maximum` | **缺失** | 新增 | 最值运算 | 中 |
| `ttnn.outer` | **缺失** | 新增 | 外积 | 低 |
| `ttnn.scatter` | **缺失** | 新增 | 分散操作 | 低 |

#### 逐元素三元操作 (3.6节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.addcdiv` / `addcmul` | **缺失** | 新增 | 复合运算 | 低 |
| `ttnn.lerp` | **缺失** | 新增 | 线性插值 | 低 |
| `ttnn.mac` | **缺失** | 新增 | 乘累加 | 低 |
| `ttnn.where` | **缺失** | 新增 | 条件选择 | 中 |

#### 矩阵操作 (3.7节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.matmul` | 已覆盖 (C++) | 新增 (Python) | Python API 版本 | 高 |
| `ttnn.linear` | **缺失** | 新增 | 线性变换 | 高 |
| `ttnn.bmm` | **缺失** | 新增 | 批量矩阵乘法 | 高 |

#### 归约操作 (3.8节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.sum` / `mean` / `max` / `min` | **缺失** | 新增 | 基础归约 | 高 |
| `ttnn.argmax` / `argmin` | **缺失** | 新增 | 极值索引 | 中 |
| `ttnn.std` / `var` | **缺失** | 新增 | 标准差/方差 | 中 |

#### 归一化操作 (3.9节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.layer_norm` | **缺失** | 新增 | 层归一化 | 高 |
| `ttnn.batch_norm` | **缺失** | 新增 | 批归一化 | 高 |
| `ttnn.group_norm` | **缺失** | 新增 | 组归一化 | 高 |
| `ttnn.rms_norm` | **缺失** | 新增 | RMS 归一化 | 中 |
| `ttnn.normalize_global` / `normalize_hw` | **缺失** | 新增 | 全局/HW 归一化 | 低 |

#### 卷积操作 (3.10节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.conv1d` | **缺失** | 新增 | 1D 卷积 | 高 |
| `ttnn.conv2d` | **缺失** | 新增 | 2D 卷积 | 高 |
| `ttnn.conv_transpose2d` | **缺失** | 新增 | 2D 转置卷积 | 中 |
| `ttnn.prepare_conv_weights` / `prepare_conv_bias` | **缺失** | 新增 | 卷积权重预处理 | 中 |

#### 池化操作 (3.11节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.max_pool2d` | **缺失** | 新增 | 2D 最大池化 | 高 |
| `ttnn.avg_pool2d` | **缺失** | 新增 | 2D 平均池化 | 高 |
| `ttnn.global_avg_pool2d` | **缺失** | 新增 | 全局平均池化 | 中 |
| `ttnn.upsample` | **缺失** | 新增 | 上采样 | 中 |

#### Transformer 操作 (3.12节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.transformer.split_query_key_value_and_split_heads` | **缺失** | 新增 | QKV 分割 | 高 |
| `ttnn.transformer.concatenate_heads` | **缺失** | 新增 | 头连接 | 高 |
| `ttnn.transformer.attention_softmax` | **缺失** | 新增 | Attention Softmax | 高 |
| `ttnn.transformer.scaled_dot_product_attention` | **缺失** | 新增 | SDPA | 高 |
| `ttnn.transformer.scaled_dot_product_attention_decode` | **缺失** | 新增 | Decode 优化 SDPA | 高 |
| `ttnn.experimental.rotary_embedding` | **缺失** | 新增 | 旋转位置编码 | 高 |

#### 集合通信 (CCL) 操作 (3.13节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.all_gather` | 已覆盖 (简化版) | 已覆盖 (完整版) | 官方文档参数更完整 | 高 |
| `ttnn.reduce_scatter` | 已覆盖 (简化版) | 已覆盖 (完整版) | 官方文档参数更完整 | 高 |
| `ttnn.experimental.all_reduce` | **缺失** | 新增 | All-Reduce 操作 | 高 |

#### 嵌入操作 (3.14节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.embedding` | **缺失** | 新增 | 嵌入查找 | 高 |
| `ttnn.embedding_bw` | **缺失** | 新增 | 嵌入反向传播 | 中 |

#### 数据移动/分片操作 (3.15节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `ttnn.interleaved_to_sharded` | **缺失** | 新增 | 交错到分片 | 高 |
| `ttnn.sharded_to_interleaved` | **缺失** | 新增 | 分片到交错 | 高 |
| `ttnn.fold` | **缺失** | 新增 | Fold 操作 | 中 |
| `ttnn.untilize_with_halo_v2` | **缺失** | 新增 | 带 Halo 的 Untilize | 中 |

---

### 2.4 Data Movement API 对比

#### NOC 异步读取

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `noc_async_read` | 已覆盖 (简化版) | 已覆盖 (完整版) | 现有文档缺少模板参数 `max_page_size`, `enable_noc_tracing` 和参数 `read_req_vc` | 中 |

#### NOC 异步写入

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `noc_async_write` | 已覆盖 (简化版) | 已覆盖 (完整版) | 现有文档缺少模板参数 `max_page_size` | 低 |
| `noc_async_write_multicast` | 已覆盖 (简化版) | 已覆盖 (完整版) | 官方文档显示额外参数 `linked` 和 `noc` | 中 |

#### NOC 地址

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `get_noc_addr` | 已覆盖 | 已覆盖 | 一致 | - |
| `get_noc_addr_from_bank_id` | **缺失** | 新增 | 从 bank ID 获取 NOC 地址 | 中 |
| `get_noc_multicast_addr` | **缺失** | 新增 | 获取多播 NOC 地址 | 中 |

#### 同步屏障

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `noc_async_read_barrier` | 已覆盖 | 已覆盖 | 一致 | - |
| `noc_async_write_barrier` | 已覆盖 | 已覆盖 | 一致 | - |
| `noc_async_full_barrier` | **缺失** | 新增 | 等待所有 NOC 操作完成 | 中 |

---

### 2.5 Compute Kernel API 对比

#### Tile 寄存器管理

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `tile_regs_acquire` | 已覆盖 | 已覆盖 | 官方文档提示 `acquire_dst()` 已弃用 | 高 |
| `tile_regs_commit` | 已覆盖 | 已覆盖 | 一致 | - |
| `tile_regs_wait` | 已覆盖 | 已覆盖 | 一致 | - |
| `tile_regs_release` | 已覆盖 | 已覆盖 | 官方文档提示 `release_dst()` 已弃用 | 高 |

#### Tile 移动操作

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `copy_tile` | **缺失** | 新增 | 从输入 CB 复制 tile 到 DST 寄存器 | 高 |
| `pack_tile` | 已覆盖 | 已覆盖 | 官方文档参数说明更详细 | 低 |

#### 矩阵操作

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `matmul_tiles` | 已覆盖 | 已覆盖 | 官方文档说明 DST 是累加模式 (DST += C) | 中 |

#### 算术操作

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `add_tiles` | 已覆盖 | 已覆盖 | 一致 | - |
| `sub_tiles` | 已覆盖 | 已覆盖 | 一致 | - |
| `mul_tiles` | 已覆盖 | 已覆盖 | 一致 | - |

#### 广播操作 (5.5节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `add_tiles_bcast` / `add_tiles_bcast_rows` / `add_tiles_bcast_cols` | **缺失** | 新增 | 加法广播操作 | 中 |
| `sub_tiles_bcast` / `sub_tiles_bcast_rows` / `sub_tiles_bcast_cols` | **缺失** | 新增 | 减法广播操作 | 中 |
| `mul_tiles_bcast` / `mul_tiles_bcast_rows` / `mul_tiles_bcast_cols` | **缺失** | 新增 | 乘法广播操作 | 中 |

#### SFPU 操作 (5.6节)

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `exp_tile` / `exp_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |
| `log_tile` / `log_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |
| `sqrt_tile` / `sqrt_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |
| `recip_tile` / `recip_tile_init` | **缺失** | 新增 | 倒数操作 | 中 |
| `sin_tile` / `sin_tile_init` | **缺失** | 新增 | 正弦 | 低 |
| `cos_tile` / `cos_tile_init` | **缺失** | 新增 | 余弦 | 低 |
| `tanh_tile` / `tanh_tile_init` | **缺失** | 新增 | 双曲正切 | 中 |
| `gelu_tile` / `gelu_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |
| `relu_tile` / `relu_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |
| `sigmoid_tile` / `sigmoid_tile_init` | 已覆盖 | 已覆盖 | 一致 | - |

---

### 2.6 Circular Buffer API 对比

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `cb_wait_front` | 已覆盖 | 已覆盖 | 官方文档参数范围说明更详细 | 低 |
| `cb_reserve_back` | 已覆盖 | 已覆盖 | 一致 | - |
| `cb_push_back` | 已覆盖 | 已覆盖 | 一致 | - |
| `cb_pop_front` | 已覆盖 | 已覆盖 | 一致 | - |
| `get_read_ptr` | 已覆盖 | 已覆盖 | 一致 | - |
| `get_write_ptr` | 已覆盖 | 已覆盖 | 一致 | - |
| `cb_pages_available_at_front` | **缺失** | 新增 | 查询前端可用页数 | 中 |
| `cb_pages_reservable_at_back` | **缺失** | 新增 | 查询后端可预留页数 | 中 |

---

### 2.7 NOC Semaphore API 对比

| API 名称 | 现有文档状态 | 官方文档状态 | 差异描述 | 优先级 |
|----------|-------------|-------------|----------|--------|
| `noc_semaphore_set` | 已覆盖 (提及) | 已覆盖 (完整版) | 现有文档仅提及，官方文档完整定义 | 中 |
| `noc_semaphore_inc` | 已覆盖 (提及) | 已覆盖 (完整版) | 官方文档显示模板参数 `posted` 和更多参数 | 中 |
| `noc_semaphore_wait` | 已覆盖 (提及) | 已覆盖 (完整版) | 现有文档仅提及，官方文档完整定义 | 中 |
| `noc_semaphore_set_remote` | **缺失** | 新增 | 异步设置远程信号量 | 中 |
| `noc_semaphore_set_multicast_loopback_src` | **缺失** | 新增 | 多播设置信号量 | 中 |

---

## 3. 差异分类汇总

### 3.1 新增 API (官方有，现有文档无)

**高优先级 (核心功能)**:
1. `EnqueueWriteBuffer` / `EnqueueReadBuffer` - Host-Device 数据传输
2. `ttnn.from_torch` / `ttnn.to_device` - PyTorch 互操作
3. `ttnn.reshape` / `ttnn.permute` / `ttnn.transpose` - 张量操作
4. `ttnn.matmul` / `ttnn.linear` / `ttnn.bmm` - 矩阵运算 (Python)
5. `ttnn.conv1d` / `ttnn.conv2d` - 卷积操作
6. `ttnn.layer_norm` / `ttnn.batch_norm` / `ttnn.group_norm` - 归一化
7. `ttnn.embedding` - 嵌入层
8. `ttnn.transformer.*` - Transformer 操作 (SDPA, RoPE 等)
9. `ttnn.all_reduce` - 集合通信
10. `copy_tile` - Tile 复制

**中优先级 (重要功能)**:
1. `QueryDevices` - 设备查询
2. `CreateBuffer` (Sharded) - 分片 Buffer
3. `DeallocateBuffer` - Buffer 释放
4. `EnqueueWriteSubBuffer` / `EnqueueReadSubBuffer` - 子 Buffer 操作
5. `Synchronize` - 设备同步
6. `CreateSemaphore` - 信号量创建
7. `get_compile_time_arg_val` - 编译时参数
8. `noc_async_full_barrier` - 完整屏障
9. `get_noc_addr_from_bank_id` / `get_noc_multicast_addr` - NOC 地址
10. `noc_semaphore_set_remote` / `noc_semaphore_set_multicast_loopback_src` - 信号量操作
11. `cb_pages_available_at_front` / `cb_pages_reservable_at_back` - CB 查询
12. `recip_tile` / `sin_tile` / `cos_tile` / `tanh_tile` - SFPU 操作
13. `*_tiles_bcast*` - 广播操作
14. `ttnn.interleaved_to_sharded` / `ttnn.sharded_to_interleaved` - 分片转换

**低优先级 (扩展功能)**:
1. 大量 TT-NN 逐元素操作 (三角函数、双曲函数、位运算等)
2. `ttnn.fold` / `ttnn.untilize_with_halo_v2` - 特殊操作

### 3.2 参数差异 (函数签名不同)

| API | 现有文档 | 官方文档 | 建议 |
|-----|---------|---------|------|
| `CreateDevice` | `CreateDevice(device_id)` | 5个参数，含可选配置 | 更新为完整签名 |
| `CreateBuffer` | 基础配置 | 3个重载版本 | 添加重载说明 |
| `noc_async_read` | 基础参数 | 含模板参数和 VC 参数 | 添加高级参数 |
| `noc_async_write_multicast` | 4个参数 | 6个参数 | 添加 `linked` 和 `noc` 参数 |
| `DataMovementConfig` | 3个字段 | 4个字段 (含 `.defines`) | 添加 `.defines` 字段 |
| `ComputeConfig` | 2个字段 | 4个字段 | 添加 `.math_approx_mode` 和 `.defines` |

### 3.3 弃用 API 标记

| 旧 API | 新 API | 来源 |
|--------|--------|------|
| `acquire_dst()` | `tile_regs_acquire()` | 官方文档 |
| `release_dst()` | `tile_regs_release()` | 官方文档 |

---

## 4. 文档覆盖差异

### 4.1 现有文档覆盖较好的部分

1. **架构概述**: 硬件代际、软件栈层次
2. **核心概念**: Tensix 核心架构、内存层次、NoC
3. **Kernel 类型**: Reader/Compute/Writer 协作模型
4. **基础 C++ API**: Host API、CB API、Dataflow API 基础函数
5. **编程示例**: DRAM Loopback、单核/多核 Matmul
6. **构建指南**: 依赖、环境变量、示例路径
7. **调试工具**: Tracy、Watcher、DPRINT

### 4.2 现有文档缺失/不足的部分

1. **TT-NN Python API**: 几乎完全缺失 (约 100+ 函数)
2. **完整 C++ API 签名**: 大量函数参数不完整
3. **Sharded Buffer**: 分片内存管理
4. **子 Buffer 操作**: EnqueueWriteSubBuffer 等
5. **信号量完整 API**: Host 和 Device 端完整操作
6. **广播操作**: Compute Kernel 广播函数
7. **更多 SFPU 操作**: 三角函数、双曲函数等

---

## 5. 建议更新优先级

### 5.1 高优先级 (必须补充)

1. **添加 TT-NN Python API 章节**
   - 张量创建、设备操作、张量操作
   - 矩阵运算、卷积、归一化
   - Transformer 操作

2. **完善 Host API**
   - `EnqueueWriteBuffer` / `EnqueueReadBuffer`
   - `CreateBuffer` 重载和 Sharded 版本
   - `Synchronize` API

3. **补充 Compute Kernel API**
   - `copy_tile`
   - 广播操作
   - 弃用说明 (`acquire_dst` -> `tile_regs_acquire`)

### 5.2 中优先级 (建议补充)

1. **信号量完整文档**
2. **CB 查询函数**
3. **更多 SFPU 操作**
4. **NOC 地址函数**

### 5.3 低优先级 (可选补充)

1. **完整 TT-NN 逐元素操作参考**
2. **高级分片操作**
3. **特殊操作 (fold, halo 等)**

---

## 6. 总结

现有文档提供了良好的 TT Metalium 框架入门指南，涵盖了核心概念、基础 API 和编程示例。但与官方文档相比，存在以下主要差距：

1. **Python API 缺失**: TT-NN 库是现代 ML 工作流的主要接口，现有文档几乎未覆盖
2. **API 完整性**: 大量 C++ API 参数定义不完整，缺少重载版本
3. **新功能**: 分片 Buffer、子 Buffer 操作、更多 SFPU 函数等新功能未文档化
4. **弃用信息**: 缺少 API 弃用和迁移指南

建议优先补充 TT-NN Python API 文档，然后完善 C++ API 的完整签名和参数说明。

---

*本报告由自动化文档对比工具生成，供参考使用。*
