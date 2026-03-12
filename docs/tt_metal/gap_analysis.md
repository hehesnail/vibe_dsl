# TT-Metalium 文档缺失内容分析报告

**生成日期**: 2026-03-12
**分析范围**: 基于 TT_Metal_Documentation_Summary.md、api_reference_scraped.md、header_api_extraction.md、programming_examples_collection.md 和 advanced_topics_collection.md

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [API 缺失分析](#2-api-缺失分析)
3. [示例缺失分析](#3-示例缺失分析)
4. [性能优化细节缺失](#4-性能优化细节缺失)
5. [其他缺失内容](#5-其他缺失内容)
6. [优先级排序与建议](#6-优先级排序与建议)

---

## 1. 执行摘要

通过对现有文档的全面分析，识别出以下关键缺失领域：

| 类别 | 缺失数量 | 优先级 |
|------|----------|--------|
| Host API | 15+ 函数 | 高 |
| Data Movement API | 20+ 函数 | 高 |
| Compute Kernel API (SFPU) | 30+ 操作 | 高 |
| Circular Buffer API | 5+ 函数 | 中 |
| Python API | 40+ 函数/操作 | 高 |
| 编程示例 | 12+ 类型 | 中-高 |
| 性能优化 | 8+ 主题 | 高 |
| 错误处理/调试 | 完整缺失 | 高 |

---

## 2. API 缺失分析

### 2.1 Host API 缺失项

#### 2.1.1 设备管理 (高优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `CreateDeviceMinimal()` | 最小化设备创建（故障恢复） | 未记录 |
| `GetNumPCIeDevices()` | 获取 PCIe 设备数量 | 未记录 |
| `IsGalaxyCluster()` | 检测 Galaxy 集群 | 未记录 |
| `GetPCIeDeviceID()` | 获取 PCIe 设备 ID | 未记录 |
| `ReleaseOwnership()` | 释放 MetalContext 所有权 | 未记录 |
| `SetRootDir()` | 设置元数据根目录 | 未记录 |

#### 2.1.2 缓冲区管理 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `CreateBuffer(address)` | 指定地址创建缓冲区 | 部分记录 |
| `CreateBuffer(sub_device_id)` | 子设备缓冲区创建 | 未记录 |
| `DeallocateBuffer()` | 缓冲区释放 | 未记录 |
| `AssignGlobalBufferToProgram()` | 全局缓冲区分配 | 未记录 |
| `Buffer::view()` | 缓冲区视图 | 未记录 |

#### 2.1.3 运行时参数 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `SetCommonRuntimeArgs()` | 设置通用运行时参数 | 未记录 |
| `GetRuntimeArgs()` | 获取运行时参数 | 未记录 |
| `GetCommonRuntimeArgs()` | 获取通用运行时参数 | 未记录 |

#### 2.1.4 子设备管理 (高优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `create_sub_device_manager()` | 创建子设备管理器 | 未记录 |
| `remove_sub_device_manager()` | 移除子设备管理器 | 未记录 |
| `load_sub_device_manager()` | 加载子设备管理器 | 未记录 |
| `set_sub_device_stall_group()` | 设置子设备停顿组 | 未记录 |
| `get_sub_device_ids()` | 获取子设备 ID 列表 | 未记录 |

#### 2.1.5 程序执行控制 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `LaunchProgram()` | 直接启动程序 | 未记录 |
| `CompileProgram()` | 显式编译程序 | 未记录 |
| `WaitProgramDone()` | 等待程序完成 | 未记录 |
| `ConfigureDeviceWithProgram()` | 使用程序配置设备 | 未记录 |

#### 2.1.6 直接内存访问 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `WriteToDeviceDRAMChannel()` | 直接写入 DRAM 通道 | 未记录 |
| `ReadFromDeviceDRAMChannel()` | 直接从 DRAM 通道读取 | 未记录 |
| `WriteToDeviceL1()` | 直接写入 L1 | 未记录 |
| `ReadFromDeviceL1()` | 直接从 L1 读取 | 未记录 |
| `WriteRegToDevice()` / `ReadRegFromDevice()` | 寄存器访问 | 未记录 |

---

### 2.2 Data Movement API 缺失项

#### 2.2.1 NOC 读取操作 (高优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_async_read_one_packet()` | 单包异步读取 | 未记录 |
| `noc_async_read_one_packet_set_state()` | 设置单包读取状态 | 未记录 |
| `noc_async_read_one_packet_with_state()` | 使用状态单包读取 | 未记录 |
| `noc_async_read_set_state()` | 设置读取状态 | 未记录 |
| `noc_async_read_with_state()` | 使用状态读取 | 未记录 |
| `noc_async_read_inc_num_issued()` | 增加读取计数 | 未记录 |

#### 2.2.2 NOC 写入操作 (高优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_async_write_one_packet()` | 单包异步写入 | 未记录 |
| `noc_async_write_one_packet_set_state()` | 设置单包写入状态 | 未记录 |
| `noc_async_write_one_packet_with_state()` | 使用状态单包写入 | 未记录 |
| `noc_async_write_set_state()` | 设置写入状态 | 未记录 |
| `noc_async_write_with_state()` | 使用状态写入 | 未记录 |
| `noc_async_write_multicast_loopback_src()` | 多播回环写入 | 未记录 |

#### 2.2.3 页面操作 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_async_read_page()` | 页面异步读取 | 未记录 |
| `noc_async_write_page()` | 页面异步写入 | 未记录 |

#### 2.2.4 分片操作 (高优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_async_read_shard()` | 分片异步读取 | 未记录 |
| `noc_async_write_shard()` | 分片异步写入 | 未记录 |

#### 2.2.5 信号量多播 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_semaphore_set_multicast_loopback_src()` | 信号量多播回环 | 未记录 |
| `noc_semaphore_set_remote()` | 远程信号量设置 | 未记录 |

#### 2.2.6 其他屏障 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `noc_async_writes_flushed()` | 等待写入刷新 | 未记录 |
| `noc_async_posted_writes_flushed()` | 等待 posted 写入刷新 | 未记录 |
| `noc_async_atomic_barrier()` | 原子操作屏障 | 未记录 |

---

### 2.3 Compute Kernel API 缺失项 (SFPU 重点)

#### 2.3.1 数学函数 (高优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 以任意底数对数 | `log_with_base_tile()` | 未记录 |
| 双曲正切 | `tanh_tile()` | 未记录 |
| 符号位 | `signbit_tile()` | 未记录 |
| 符号函数 | `sign_tile()` | 未记录 |
| 平方 | `square_tile()` | 未记录 |
| 平铺乘积 | `tiled_prod_tile()` | 未记录 |
| 幂运算 | `power_tile()` | 未记录 |
| 迭代幂运算 | `power_iterative_tile()` | 未记录 |
| 以 2 为底的指数 | `exp2_tile()` | 未记录 |
| Heaviside 阶跃 | `heaviside_tile()` | 未记录 |
| exp(x)-1 | `expm1_tile()` | 未记录 |
| SiLU/Swish | `silu_tile()` | 未记录 |

#### 2.3.2 矩阵乘法变体 (高优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 短初始化 | `mm_init_short()` | 未记录 |
| 带数据类型短初始化 | `mm_init_short_with_dt()` | 未记录 |
| 块矩阵初始化 | `mm_block_init()` | 未记录 |
| 块矩阵乘法 | `matmul_block()` | 未记录 |
| 动态节流块乘法 | `matmul_block_math_dynamic_throttle()` | 未记录 |

#### 2.3.3 二元操作变体 (中优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 通用初始化 | `binary_op_init_common()` | 已记录但需更多细节 |
| 模板初始化 | `binary_tiles_init()` | 未记录 |
| 乘法初始化 | `mul_tiles_init()` | 未记录 |
| 加法初始化 | `add_tiles_init()` | 未记录 |
| 减法初始化 | `sub_tiles_init()` | 未记录 |
| 目标重用初始化 | `binary_dest_reuse_tiles_init()` | 未记录 |
| 目标重用操作 | `binary_dest_reuse_tiles()` | 未记录 |

#### 2.3.4 归约操作 (高优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 归约初始化 | `reduce_init()` | 未记录 |
| 归约反初始化 | `reduce_uninit()` | 未记录 |
| 归约 Tile | `reduce_tile()` | 未记录 |
| 归约数学 | `reduce_tile_math()` | 未记录 |

#### 2.3.5 打包操作 (中优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 打包 Tile 块 | `pack_tile_block()` | 未记录 |
| 重新配置数据格式 | `pack_reconfig_data_format()` | 未记录 |
| 重新配置 L1 累加 | `pack_reconfig_l1_acc()` | 未记录 |
| 打包行初始化 | `pack_rows_init()` | 未记录 |
| 打包行 | `pack_rows()` | 未记录 |
| 打包行反初始化 | `pack_rows_uninit()` | 未记录 |

#### 2.3.6 其他激活函数 (中优先级)

| 缺失操作 | 函数签名 | 当前状态 |
|----------|----------|----------|
| 绝对值 | `abs_tile()` | 未记录 |
| 绝对值 (Int32) | `abs_tile_int32()` | 未记录 |

---

### 2.4 Circular Buffer API 缺失项

#### 2.4.1 查询函数 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `cb_pages_available_at_front()` | 查询前端可用页数 | 未记录 |
| `cb_pages_reservable_at_back()` | 查询后端可预留页数 | 未记录 |
| `cb_num_tiles_available()` | 查询可用 Tile 数 | 未记录 |

#### 2.4.2 Host 端 CB 管理 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `GetCircularBufferConfig()` | 获取 CB 配置 | 未记录 |
| `UpdateCircularBufferTotalSize()` | 更新 CB 总大小 | 未记录 |
| `UpdateCircularBufferPageSize()` | 更新 CB 页大小 | 未记录 |
| `UpdateDynamicCircularBufferAddress()` | 更新动态 CB 地址 | 未记录 |
| `UpdateDynamicCircularBufferAddressAndTotalSize()` | 同时更新地址和大小 | 未记录 |

#### 2.4.3 Tile 信息查询 (中优先级)

| 缺失函数 | 用途 | 当前状态 |
|----------|------|----------|
| `get_tile_hw()` | 获取 Tile 高宽 | 未记录 |
| `get_tile_num_faces()` | 获取 Tile 面数 | 未记录 |
| `get_dataformat()` | 获取数据格式 | 未记录 |

---

### 2.5 Python API (TT-NN) 缺失项

#### 2.5.1 张量创建 (低优先级)

- `ttnn.arange()` - 范围张量创建
- `ttnn.empty()` / `ttnn.empty_like()` - 未初始化张量
- `ttnn.full()` / `ttnn.full_like()` - 填充值张量
- `ttnn.as_tensor()` - 转换张量

#### 2.5.2 张量操作 (中优先级)

- `ttnn.move()` - 移动张量
- `ttnn.clone()` - 克隆张量
- `ttnn.copy()` - 复制张量
- `ttnn.deallocate()` - 显式释放

#### 2.5.3 形状操作 (中优先级)

- `ttnn.expand()` - 扩展维度
- `ttnn.repeat()` - 重复张量
- `ttnn.repeat_interleave()` - 重复元素
- `ttnn.chunk()` - 分块
- `ttnn.split()` - 分割
- `ttnn.squeeze()` / `ttnn.unsqueeze()` - 维度压缩/扩展
- `ttnn.gather()` - 收集操作

#### 2.5.4 一元操作 (部分已记录，需补充)

| 操作 | 状态 |
|------|------|
| `exp2`, `expm1` | 未记录 |
| `log2`, `log10`, `log1p` | 未记录 |
| `cbrt` | 未记录 |
| `asin`, `acos`, `atan` | 未记录 |
| `sinh`, `cosh`, `tanh` | 部分记录 |
| `asinh`, `acosh`, `atanh` | 未记录 |
| `relu6`, `leaky_relu`, `prelu` | 未记录 |
| `elu`, `selu`, `celu` | 未记录 |
| `silu`, `mish` | 未记录 |
| `hardswish`, `hardsigmoid`, `hardtanh` | 未记录 |
| `floor`, `ceil`, `round`, `trunc` | 未记录 |
| `isfinite`, `isinf`, `isnan` | 未记录 |
| `reciprocal`, `square` | 未记录 |

#### 2.5.5 二元操作 (部分已记录)

| 操作 | 状态 |
|------|------|
| `pow` | 未记录 |
| `atan2` | 未记录 |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | 部分记录 |
| `logical_and`, `logical_or`, `logical_xor`, `logical_not` | 未记录 |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not` | 未记录 |
| `bitwise_left_shift`, `bitwise_right_shift` | 未记录 |
| `minimum`, `maximum` | 未记录 |
| `outer` | 未记录 |
| `scatter` | 未记录 |

#### 2.5.6 三元操作 (中优先级)

- `ttnn.addcdiv()` - 加除融合
- `ttnn.addcmul()` - 加乘融合
- `ttnn.lerp()` - 线性插值
- `ttnn.mac()` - 乘累加
- `ttnn.where()` - 条件选择

#### 2.5.7 归约操作 (部分已记录)

- `ttnn.argmax()` / `ttnn.argmin()` - 未记录
- `ttnn.std()` / `ttnn.var()` - 未记录

#### 2.5.8 归一化操作 (中优先级)

- `ttnn.batch_norm()` - 批归一化
- `ttnn.group_norm()` - 组归一化
- `ttnn.normalize_global()` - 全局归一化
- `ttnn.normalize_hw()` - 高宽归一化

#### 2.5.9 卷积操作 (高优先级)

- `ttnn.conv1d()` - 1D 卷积
- `ttnn.conv_transpose2d()` - 转置卷积
- `ttnn.prepare_conv_weights()` - 权重预处理
- `ttnn.prepare_conv_bias()` - 偏置预处理

#### 2.5.10 池化操作 (中优先级)

- `ttnn.max_pool2d()` - 最大池化
- `ttnn.avg_pool2d()` - 平均池化
- `ttnn.global_avg_pool2d()` - 全局平均池化
- `ttnn.upsample()` - 上采样

#### 2.5.11 Transformer 操作 (高优先级)

- `ttnn.transformer.split_query_key_value_and_split_heads()` - QKV 分割
- `ttnn.transformer.concatenate_heads()` - 头拼接
- `ttnn.transformer.attention_softmax()` - 注意力 Softmax
- `ttnn.transformer.scaled_dot_product_attention()` - 缩放点积注意力
- `ttnn.transformer.scaled_dot_product_attention_decode()` - 解码优化注意力
- `ttnn.experimental.rotary_embedding()` - 旋转位置编码

#### 2.5.12 集合通信 (高优先级)

- `ttnn.all_gather()` - 全收集
- `ttnn.reduce_scatter()` - 归约分散
- `ttnn.experimental.all_reduce()` - 全归约

#### 2.5.13 嵌入操作 (中优先级)

- `ttnn.embedding()` - 嵌入查找
- `ttnn.embedding_bw()` - 嵌入反向传播

#### 2.5.14 分片操作 (高优先级)

- `ttnn.interleaved_to_sharded()` - 交织转分片
- `ttnn.sharded_to_interleaved()` - 分片转交织
- `ttnn.fold()` - 折叠操作
- `ttnn.untilize_with_halo_v2()` - 带 halo 的 untilize

---

## 3. 示例缺失分析

### 3.1 基础示例 (高优先级)

| 示例类型 | 描述 | 优先级 |
|----------|------|--------|
| **Hello World Kernel** | 最简单的内核打印 | 高 |
| **L1 Buffer 操作** | 纯 L1 内存操作示例 | 高 |
| **信号量同步** | 核心间信号量同步 | 高 |
| **多核同步** | 多核协作模式 | 高 |

### 3.2 中级示例 (中优先级)

| 示例类型 | 描述 | 优先级 |
|----------|------|--------|
| **分片张量操作** | Sharded tensor 示例 | 高 |
| **双缓冲模式** | Double buffering 实现 | 高 |
| **数据重用优化** | 更详细的数据重用示例 | 中 |
| **动态形状处理** | 运行时形状变化 | 中 |
| **多队列并行** | Multi-CQ 示例 | 中 |

### 3.3 高级示例 (中-高优先级)

| 示例类型 | 描述 | 优先级 |
|----------|------|--------|
| **Metal Trace 捕获重放** | 完整的 Trace 示例 | 高 |
| **子设备管理** | Sub-device 创建和使用 | 高 |
| **以太网内核** | Ethernet kernel 编程 | 高 |
| **多芯片 AllGather** | CCL 操作示例 | 高 |
| **自定义数据格式** | Bfp8_b, Bfp4_b 使用 | 中 |
| **动态 CB 配置** | 运行时 CB 调整 | 中 |
| **性能剖析集成** | 与 Profiler 集成的完整示例 | 中 |

### 3.4 SFPU/自定义内核示例 (高优先级)

| 示例类型 | 描述 | 优先级 |
|----------|------|--------|
| **完整 SFPU 操作集** | 所有 SFPU 操作示例 | 高 |
| **SFPI 条件执行** | v_if/v_else 模式 | 高 |
| **SFPI 循环优化** | 向量循环模式 | 中 |
| **自定义归约** | Reduce 操作示例 | 中 |
| **矩阵块操作** | Matmul block 示例 | 高 |

---

## 4. 性能优化细节缺失

### 4.1 内存优化 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **L1 内存布局** | 详细的 L1 内存映射和布局策略 | 高 |
| **Bank 冲突避免** | DRAM bank 访问模式优化 | 高 |
| **CB 大小计算** | 不同场景下的 CB 大小选择指南 | 高 |
| **内存池管理** | 内存分配器内部机制 | 中 |
| **碎片整理** | 内存碎片问题及解决 | 中 |

### 4.2 计算优化 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **Math Fidelity 选择指南** | 不同精度模式的详细对比 | 高 |
| **FP32 累加模式** | fp32_dest_acc_en 使用场景 | 高 |
| **近似模式权衡** | fast_and_approx 参数影响 | 高 |
| **FPU vs SFPU 选择** | 何时使用矩阵引擎 vs 向量引擎 | 高 |
| **Tile 面操作优化** | Face-level 优化技巧 | 中 |

### 4.3 数据传输优化 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **NOC 路由优化** | NOC_0 vs NOC_1 选择 | 高 |
| **多播策略** | 何时使用多播及最佳实践 | 高 |
| **批量传输** | 批量读写优化技巧 | 高 |
| **异步流水线** | 计算与通信重叠深度优化 | 高 |
| **NoC 拥塞避免** | 拥塞检测和缓解策略 | 中 |

### 4.4 并行化策略 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **核心网格划分** | 不同形状工作负载的最佳划分 | 高 |
| **负载均衡** | 动态负载均衡技术 | 高 |
| **同步开销最小化** | 减少同步等待的技巧 | 高 |
| **子设备并行** | Sub-device 并行执行模式 | 高 |

### 4.5 Metal Trace 深度优化 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **Trace 捕获最佳实践** | 完整的捕获/重放工作流 | 高 |
| **动态形状处理** | Trace 与动态形状的兼容 | 高 |
| **Trace 缓冲区管理** | 缓冲区大小和分配策略 | 中 |
| **多 Trace 切换** | 多个 Trace 的管理 | 中 |

### 4.6 多芯片优化 (高优先级)

| 主题 | 缺失内容 | 优先级 |
|------|----------|--------|
| **芯片间拓扑感知** | 拓扑感知的通信优化 | 高 |
| **CCL 算法选择** | 不同 CCL 操作的算法选择 | 高 |
| **以太网带宽优化** | 最大化以太网利用率 | 高 |
| **Galaxy 特定优化** | 32+ 芯片系统的优化 | 中 |

---

## 5. 其他缺失内容

### 5.1 错误处理文档 (高优先级)

| 主题 | 缺失内容 | 说明 |
|------|----------|------|
| **常见错误代码** | 错误代码参考手册 | 完整缺失 |
| **调试技巧** | 系统性调试方法论 | 部分缺失 |
| **超时处理** | 内核超时检测和处理 | 未记录 |
| **内存越界检测** | 检测和调试内存访问错误 | 部分缺失 |
| **CB 溢出处理** | Circular Buffer 溢出处理 | 未记录 |
| **NOC 错误恢复** | NOC 传输错误处理 | 未记录 |

### 5.2 调试技巧 (高优先级)

| 主题 | 缺失内容 | 说明 |
|------|----------|------|
| **DPRINT 高级用法** | 条件打印、格式化输出 | 部分缺失 |
| **Watcher 深度使用** | 高级 Watcher 功能 | 部分缺失 |
| **GDB 调试内核** | 使用 GDB 调试设备代码 | 未记录 |
| **性能瓶颈分析** | 系统性性能分析方法 | 部分缺失 |
| **内存泄漏检测** | 检测内存泄漏的方法 | 未记录 |
| **死锁调试** | 检测和解决死锁 | 未记录 |

### 5.3 最佳实践 (高优先级)

| 主题 | 缺失内容 | 说明 |
|------|----------|------|
| **Kernel 设计模式** | 常见设计模式目录 | 部分缺失 |
| **代码组织** | 大型项目代码结构建议 | 未记录 |
| **测试策略** | 内核测试方法论 | 未记录 |
| **CI/CD 集成** | 持续集成最佳实践 | 未记录 |
| **版本兼容性** | 跨版本兼容性指南 | 未记录 |
| **迁移指南** | 从其他平台迁移指南 | 未记录 |

### 5.4 架构特定文档 (中优先级)

| 主题 | 缺失内容 | 说明 |
|------|----------|------|
| **Wormhole 特定优化** | Wormhole 架构优化 | 部分缺失 |
| **Blackhole 特定优化** | Blackhole 架构优化 | 部分缺失 |
| **Grayskull 废弃迁移** | 从 Grayskull 迁移 | 未记录 |
| **硬件特性对比** | 各代硬件特性详细对比 | 部分缺失 |

### 5.5 工具和集成 (中优先级)

| 主题 | 缺失内容 | 说明 |
|------|----------|------|
| **CMake 集成** | 完整的 CMake 集成指南 | 部分缺失 |
| **Bazel 支持** | Bazel 构建支持 | 未记录 |
| **IDE 配置** | VSCode/CLion 配置 | 未记录 |
| **Docker 使用** | 容器化开发环境 | 未记录 |
| **Kubernetes 部署** | K8s 部署指南 | 未记录 |

---

## 6. 优先级排序与建议

### 6.1 最高优先级 (立即补充)

1. **Compute Kernel API - SFPU 操作**
   - 30+ 个 SFPU 函数需要完整文档
   - 包括初始化函数和使用示例

2. **错误处理文档**
   - 常见错误代码手册
   - 调试方法论
   - 问题排查流程

3. **Data Movement API - 高级 NOC 操作**
   - 单包操作函数
   - 状态管理函数
   - 分片操作函数

4. **Python API - Transformer/CCL 操作**
   - 注意力机制操作
   - 集合通信操作
   - 卷积操作

### 6.2 高优先级 (短期补充)

1. **Host API - 子设备管理**
   - 完整的子设备 API
   - 使用场景和示例

2. **性能优化深度指南**
   - Math Fidelity 选择
   - 内存布局优化
   - NOC 路由优化

3. **编程示例 - 高级场景**
   - Metal Trace 完整示例
   - 多芯片 CCL 示例
   - 子设备管理示例

4. **调试技巧完整文档**
   - 系统性调试方法
   - 工具组合使用
   - 性能分析深度指南

### 6.3 中优先级 (中期补充)

1. **Circular Buffer 高级 API**
   - 查询函数
   - 动态配置

2. **Python API - 完整覆盖**
   - 所有未记录的操作
   - 参数详细说明

3. **架构特定优化**
   - Wormhole/Blackhole 特定指南

4. **工具和集成**
   - IDE 配置
   - 构建系统集成

### 6.4 建议补充顺序

```
第1阶段 (2-3周):
├── SFPU API 完整文档
├── 错误处理指南
├── 调试技巧手册
└── 常见问题 FAQ

第2阶段 (3-4周):
├── Data Movement 高级 API
├── Host 子设备 API
├── 性能优化深度指南
└── Metal Trace 完整示例

第3阶段 (4-6周):
├── Python API 完整覆盖
├── 编程示例扩展
├── 架构特定优化
└── 工具和集成指南
```

---

## 附录 A: 缺失 API 汇总表

### A.1 Host API (15个)

| # | 函数名 | 优先级 |
|---|--------|--------|
| 1 | `CreateDeviceMinimal` | 高 |
| 2 | `GetNumPCIeDevices` | 中 |
| 3 | `IsGalaxyCluster` | 中 |
| 4 | `GetPCIeDeviceID` | 中 |
| 5 | `ReleaseOwnership` | 中 |
| 6 | `SetRootDir` | 低 |
| 7 | `create_sub_device_manager` | 高 |
| 8 | `remove_sub_device_manager` | 高 |
| 9 | `load_sub_device_manager` | 高 |
| 10 | `set_sub_device_stall_group` | 高 |
| 11 | `LaunchProgram` | 中 |
| 12 | `CompileProgram` | 中 |
| 13 | `WriteToDeviceL1` | 中 |
| 14 | `ReadFromDeviceL1` | 中 |
| 15 | `SetCommonRuntimeArgs` | 中 |

### A.2 Data Movement API (20个)

| # | 函数名 | 优先级 |
|---|--------|--------|
| 1 | `noc_async_read_one_packet` | 高 |
| 2 | `noc_async_read_one_packet_set_state` | 高 |
| 3 | `noc_async_read_one_packet_with_state` | 高 |
| 4 | `noc_async_read_set_state` | 高 |
| 5 | `noc_async_read_with_state` | 高 |
| 6 | `noc_async_write_one_packet` | 高 |
| 7 | `noc_async_write_one_packet_set_state` | 高 |
| 8 | `noc_async_write_one_packet_with_state` | 高 |
| 9 | `noc_async_write_set_state` | 高 |
| 10 | `noc_async_write_with_state` | 高 |
| 11 | `noc_async_write_multicast_loopback_src` | 中 |
| 12 | `noc_async_read_page` | 中 |
| 13 | `noc_async_write_page` | 中 |
| 14 | `noc_async_read_shard` | 高 |
| 15 | `noc_async_write_shard` | 高 |
| 16 | `noc_semaphore_set_multicast_loopback_src` | 中 |
| 17 | `noc_semaphore_set_remote` | 中 |
| 18 | `noc_async_writes_flushed` | 低 |
| 19 | `noc_async_posted_writes_flushed` | 低 |
| 20 | `noc_async_atomic_barrier` | 低 |

### A.3 Compute Kernel API - SFPU (30个)

| # | 函数名 | 类别 | 优先级 |
|---|--------|------|--------|
| 1 | `log_with_base_tile` | 数学 | 高 |
| 2 | `tanh_tile` | 数学 | 高 |
| 3 | `signbit_tile` | 数学 | 中 |
| 4 | `sign_tile` | 数学 | 中 |
| 5 | `square_tile` | 数学 | 中 |
| 6 | `power_tile` | 数学 | 高 |
| 7 | `power_iterative_tile` | 数学 | 中 |
| 8 | `exp2_tile` | 数学 | 中 |
| 9 | `heaviside_tile` | 数学 | 低 |
| 10 | `expm1_tile` | 数学 | 中 |
| 11 | `silu_tile` | 激活 | 高 |
| 12 | `mm_init_short` | 矩阵 | 高 |
| 13 | `mm_init_short_with_dt` | 矩阵 | 高 |
| 14 | `mm_block_init` | 矩阵 | 高 |
| 15 | `matmul_block` | 矩阵 | 高 |
| 16 | `matmul_block_math_dynamic_throttle` | 矩阵 | 中 |
| 17 | `binary_tiles_init` | 二元 | 中 |
| 18 | `mul_tiles_init` | 二元 | 中 |
| 19 | `add_tiles_init` | 二元 | 中 |
| 20 | `sub_tiles_init` | 二元 | 中 |
| 21 | `binary_dest_reuse_tiles_init` | 二元 | 低 |
| 22 | `binary_dest_reuse_tiles` | 二元 | 低 |
| 23 | `reduce_init` | 归约 | 高 |
| 24 | `reduce_uninit` | 归约 | 高 |
| 25 | `reduce_tile` | 归约 | 高 |
| 26 | `reduce_tile_math` | 归约 | 高 |
| 27 | `pack_tile_block` | 打包 | 中 |
| 28 | `pack_reconfig_data_format` | 打包 | 中 |
| 29 | `pack_reconfig_l1_acc` | 打包 | 中 |
| 30 | `pack_rows` | 打包 | 中 |

---

*报告结束*
