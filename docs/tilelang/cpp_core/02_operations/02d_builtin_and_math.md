# TileLang C++ 核心：内置函数与数学运算模块

本文档详细分析 TileLang 项目中内置函数(Builtin)和数学运算的 C++ 核心实现。

## 1. 模块概述

内置函数模块 (`src/op/builtin.cc` 和 `src/op/builtin.h`) 定义了 TileLang 支持的所有底层原语操作，包括：

- **快速数学函数**: 指数、对数、三角函数等
- **高精度 IEEE 运算**: 符合 IEEE 标准的算术运算
- **内存访问原语**: TMA、异步拷贝、全局内存访问等
- **同步原语**: 屏障、同步操作
- **张量核心操作**: WGMMA、MMA、WMMA 等
- **原子操作**: 原子加、原子最大/最小等
- **随机数生成**: RNG 相关操作
- **线程/线程束查询**: 线程索引、线程束索引等

## 2. 配置选项 (`src/op/builtin.cc:19-48`)

TileLang 提供丰富的编译时配置选项：

```cpp
// 调试和开发选项
TVM_REGISTER_PASS_CONFIG_OPTION(kDebugMergeSharedMemoryAllocations, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableTMALower, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSafeMemoryLegalize, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableWarpSpecialized, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableThreadStorageSync, Bool);

// 性能优化选项
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAggressiveSharedMemoryMerge, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableVectorize256, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAsyncCopy, Bool);

// 代码生成选项
TVM_REGISTER_PASS_CONFIG_OPTION(kPtxasRegisterUsageLevel, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnablePTXASVerboseOutput, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kConfigIndexBitwidth, Integer);

// 调试输出选项
TVM_REGISTER_PASS_CONFIG_OPTION(kASTPrintEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableDumpIR, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDumpIRDir, ffi::Array<ffi::String>);
```

### 2.1 配置常量定义 (`src/op/builtin.h:44-145`)

```cpp
static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kEnableFastMath = "tl.enable_fast_math";
static constexpr const char *kDisableFastMath = "tl.disable_fast_math";
// ... 更多配置
```

## 3. 快速数学函数

### 3.1 指数和对数函数 (`src/op/builtin.cc:67-80`)

```cpp
// 快速指数函数
TIR_DEFINE_TL_BUILTIN(__exp).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 以10为底的指数
TIR_DEFINE_TL_BUILTIN(__exp10).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 自然对数
TIR_DEFINE_TL_BUILTIN(__log).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 以2为底的对数
TIR_DEFINE_TL_BUILTIN(__log2).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 以10为底的对数
TIR_DEFINE_TL_BUILTIN(__log10).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

### 3.2 三角函数 (`src/op/builtin.cc:82-89`)

```cpp
// 正切
TIR_DEFINE_TL_BUILTIN(__tan).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 余弦
TIR_DEFINE_TL_BUILTIN(__cos).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 正弦
TIR_DEFINE_TL_BUILTIN(__sin).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

### 3.3 头文件声明 (`src/op/builtin.h:174-190`)

```cpp
// 快速数学相关操作
// __exp(x) - 快速指数
TVM_DLL const Op &__exp();
// __exp10(x) - 快速以10为底指数
TVM_DLL const Op &__exp10();
// __log(x) - 快速自然对数
TVM_DLL const Op &__log();
// __log2(x) - 快速以2为底对数
TVM_DLL const Op &__log2();
// __log10(x) - 快速以10为底对数
TVM_DLL const Op &__log10();
// __tan(x) - 快速正切
TVM_DLL const Op &__tan();
// __cos(x) - 快速余弦
TVM_DLL const Op &__cos();
// __sin(x) - 快速正弦
TVM_DLL const Op &__sin();
```

## 4. 高精度 IEEE 运算

### 4.1 基本算术运算 (`src/op/builtin.cc:92-118`)

```cpp
// IEEE 标准加法
TIR_DEFINE_TL_BUILTIN(ieee_add).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// IEEE 标准减法
TIR_DEFINE_TL_BUILTIN(ieee_sub).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// IEEE 标准乘法
TIR_DEFINE_TL_BUILTIN(ieee_mul).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// IEEE 标准融合乘加
TIR_DEFINE_TL_BUILTIN(ieee_fmaf).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// IEEE 标准倒数
TIR_DEFINE_TL_BUILTIN(ieee_frcp).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// IEEE 标准平方根
TIR_DEFINE_TL_BUILTIN(ieee_fsqrt)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// IEEE 标准倒数平方根
TIR_DEFINE_TL_BUILTIN(ieee_frsqrt)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// IEEE 标准除法
TIR_DEFINE_TL_BUILTIN(ieee_fdiv).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));
```

### 4.2 函数签名说明 (`src/op/builtin.h:192-208`)

| 函数 | 参数 | 说明 |
|-----|------|------|
| `ieee_add` | (x, y, rounding_mode) | IEEE 标准加法 |
| `ieee_sub` | (x, y, rounding_mode) | IEEE 标准减法 |
| `ieee_mul` | (x, y, rounding_mode) | IEEE 标准乘法 |
| `ieee_fmaf` | (x, y, z, rounding_mode) | IEEE 标准融合乘加 |
| `ieee_frcp` | (x, rounding_mode) | IEEE 标准倒数 |
| `ieee_fsqrt` | (x, rounding_mode) | IEEE 标准平方根 |
| `ieee_frsqrt` | (x) | IEEE 标准倒数平方根 |
| `ieee_fdiv` | (x, y, rounding_mode) | IEEE 标准除法 |

### 4.3 打包 FP32x2 运算 (`src/op/builtin.cc:121-129`)

```cpp
// 打包 FP32x2 加法 (PTX .f32x2)
TIR_DEFINE_TL_BUILTIN(fadd2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// 打包 FP32x2 乘法
TIR_DEFINE_TL_BUILTIN(fmul2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// 打包 FP32x2 融合乘加
TIR_DEFINE_TL_BUILTIN(fma2).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));
```

## 5. 内存访问原语

### 5.1 指针访问元数据 (`src/op/builtin.cc:61-64`)

```cpp
// 指针访问元数据操作 (前端专用，后续降级)
TIR_DEFINE_TL_BUILTIN(access_ptr)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
```

参数说明 (`src/op/builtin.h:155-172`):
```cpp
/*!
 * \brief TileLang intrinsic for carrying pointer access metadata in frontend.
 *
 * access_ptr(base_load, extent, rw_mask)
 * - base_load: BufferLoad，索引表示基元素地址
 * - extent: 1D 范围（元素数）
 * - rw_mask: 1=读, 2=写, 3=读写
 */
TVM_DLL const Op &access_ptr();
```

### 5.2 TMA (Tensor Memory Accelerator) 操作

#### TMA 描述符创建 (`src/op/builtin.cc:146-155`)

```cpp
// TMA 描述符创建（用于分块加载）
TIR_DEFINE_TL_BUILTIN(create_tma_descriptor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// TMA Im2Col 描述符创建（用于图像到列加载）
TIR_DEFINE_TL_BUILTIN(create_tma_im2col_descriptor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
```

#### TMA 加载和存储 (`src/op/builtin.cc:161-170`)

```cpp
// TMA 加载：全局张量描述符 -> 共享内存
TIR_DEFINE_TL_BUILTIN(tma_load).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// TMA Im2Col 加载
TIR_DEFINE_TL_BUILTIN(tma_load_im2col)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TMA 存储：共享内存 -> 全局张量描述符
TIR_DEFINE_TL_BUILTIN(tma_store).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

#### TMA 同步 (`src/op/builtin.cc:252-260`)

```cpp
// TMA 存储到达信号
TIR_DEFINE_TL_BUILTIN(tma_store_arrive)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TMA 存储等待完成
TIR_DEFINE_TL_BUILTIN(tma_store_wait)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 5.3 异步拷贝操作 (`src/op/builtin.cc:237-250`)

```cpp
// PTX 异步拷贝屏障（使用 cp.async.mbarrier.arrive.noinc）
TIR_DEFINE_TL_BUILTIN(ptx_cp_async_barrier_noinc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TileLang PTX 异步拷贝：全局内存 -> 共享内存
TIR_DEFINE_TL_BUILTIN(ptx_cp_async)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 异步操作共享内存屏障
TIR_DEFINE_TL_BUILTIN(fence_proxy_async)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 5.4 全局内存访问 (`src/op/builtin.cc:499-542`)

```cpp
// __ldg - CUDA 只读缓存加载
TIR_DEFINE_TL_BUILTIN(__ldg).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg32 - 32位全局内存加载
TIR_DEFINE_TL_BUILTIN(ldg32).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg64 - 64位全局内存加载
TIR_DEFINE_TL_BUILTIN(ldg64).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg128 - 128位全局内存加载
TIR_DEFINE_TL_BUILTIN(ldg128).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg256 - 256位全局内存加载 (CUDA 12.9+)
TIR_DEFINE_TL_BUILTIN(ldg256).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// stg32 - 32位全局内存存储
TIR_DEFINE_TL_BUILTIN(stg32).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg64 - 64位全局内存存储
TIR_DEFINE_TL_BUILTIN(stg64).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg128 - 128位全局内存存储
TIR_DEFINE_TL_BUILTIN(stg128).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg256 - 256位全局内存存储 (CUDA 12.9+)
TIR_DEFINE_TL_BUILTIN(stg256).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

## 6. 同步原语

### 6.1 屏障操作 (`src/op/builtin.cc:141-145`)

```cpp
// 创建 mbarrier 列表
TIR_DEFINE_TL_BUILTIN(create_list_of_mbarrier)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 通过 barrier_id 获取编译器注入的 mbarrier
TIR_DEFINE_TL_BUILTIN(get_mbarrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
```

### 6.2 PTX 屏障操作 (`src/op/builtin.cc:172-191`)

```cpp
// PTX 屏障初始化 fence
TIR_DEFINE_TL_BUILTIN(ptx_fence_barrier_init)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 集群屏障到达
TIR_DEFINE_TL_BUILTIN(ptx_arrive_cluster_barrier)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// mbarrier 等待（带奇偶位）
TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// mbarrier 期望事务
TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 6.3 集群同步 (`src/op/builtin.cc:319-346`)

```cpp
// 集群屏障到达（宽松排序）
TIR_DEFINE_TL_BUILTIN(cluster_arrive_relaxed)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 集群屏障到达
TIR_DEFINE_TL_BUILTIN(cluster_arrive)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 集群屏障等待
TIR_DEFINE_TL_BUILTIN(cluster_wait)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 集群同步（到达 + 等待）
TIR_DEFINE_TL_BUILTIN(cluster_sync)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 获取集群中的块排名
TIR_DEFINE_TL_BUILTIN(block_rank_in_cluster)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
```

### 6.4 网格和线程束同步 (`src/op/builtin.cc:344-348`)

```cpp
// 网格同步
TIR_DEFINE_TL_BUILTIN(sync_grid).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 线程束同步
TIR_DEFINE_TL_BUILTIN(sync_warp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

## 7. 张量核心操作

### 7.1 WGMMA (Warp Group Matrix Multiply Accumulate) (`src/op/builtin.cc:192-211`)

```cpp
// PTX WGMMA SS (Shared-Shared) 指令
TIR_DEFINE_TL_BUILTIN(ptx_wgmma_ss)
    .set_num_inputs(15)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// PTX WGMMA RS (Register-Shared) 指令
TIR_DEFINE_TL_BUILTIN(ptx_wgmma_rs)
    .set_num_inputs(15)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

WGMMA 参数说明 (`src/op/builtin.h:321-340`):
```cpp
/*!
 * \brief tvm intrinsic for ptx tensor core wgmma instructions.
 *
 *  void ptx_wgmma_ss(StringImm accum_dtype, StringImm wgmma_prefix,
 *                    bool a_is_k_major, bool b_is_k_major,
 *                    StringImm a_dtype_abbrv, StringImm b_dtype_abbrv,
 *                    StringImm accum_dtype_abbrv,
 *                    Var A_descriptor, PrimExpr A_offset,
 *                    Var B_descriptor, Var B_offset,
 *                    Var C_data, Var C_offset,
 *                    bool scale_out, bool scale_in_a, bool scale_in_b);
 */
```

### 7.2 TCGEN05 MMA (Blackwell) (`src/op/builtin.cc:202-211`)

```cpp
// TCGEN05 MMA SS (Shared-Shared) 指令
TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ss)
    .set_num_inputs(14)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TCGEN05 MMA TS (Tensor-Shared) 指令
TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ts)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// TCGEN05 MMA 屏障到达
TIR_DEFINE_TL_BUILTIN(tcgen05_mma_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 7.3 张量内存操作 (`src/op/builtin.cc:212-221`)

```cpp
// 初始化张量内存
TIR_DEFINE_TL_BUILTIN(ptx_init_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 释放张量内存
TIR_DEFINE_TL_BUILTIN(ptx_deallocate_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 7.4 SM70 MMA 和 LDMATRIX (`src/op/builtin.cc:222-235`)

```cpp
// PTX MMA SM70 指令
TIR_DEFINE_TL_BUILTIN(ptx_mma_sm70)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// PTX LDMATRIX 指令
TIR_DEFINE_TL_BUILTIN(ptx_ldmatrix)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// PTX STMATRIX 指令
TIR_DEFINE_TL_BUILTIN(ptx_stmatrix)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 7.5 AMD 矩阵核心 (`src/op/builtin.cc:371-387`)

```cpp
// AMD MFMA 指令
TIR_DEFINE_TL_BUILTIN(tvm_mfma).set_num_inputs(12).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// AMD MFMA 存储
TIR_DEFINE_TL_BUILTIN(tvm_mfma_store)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// AMD RDNA WMMA 指令
TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma)
    .set_num_inputs(12)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// AMD RDNA WMMA 存储
TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma_store)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 8. 线程束组操作 (`src/op/builtin.cc:261-290`)

```cpp
// 设置最大寄存器数
TIR_DEFINE_TL_BUILTIN(set_max_nreg)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 不设置最大寄存器数
TIR_DEFINE_TL_BUILTIN(no_set_max_nreg)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束组到达
TIR_DEFINE_TL_BUILTIN(warpgroup_arrive)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束组提交批次
TIR_DEFINE_TL_BUILTIN(warpgroup_commit_batch)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束组等待
TIR_DEFINE_TL_BUILTIN(warpgroup_wait)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束组屏障操作数
TIR_DEFINE_TL_BUILTIN(warpgroup_fence_operand)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 等待 WGMMA 完成
TIR_DEFINE_TL_BUILTIN(wait_wgmma)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 9. 线程/线程束查询 (`src/op/builtin.cc:291-310`)

```cpp
// 获取 lane 索引
TIR_DEFINE_TL_BUILTIN(get_lane_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// 获取 warp 索引（同步）
TIR_DEFINE_TL_BUILTIN(get_warp_idx_sync)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// 获取 warp 索引（不同步）
TIR_DEFINE_TL_BUILTIN(get_warp_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// 获取 warp group 索引
TIR_DEFINE_TL_BUILTIN(get_warp_group_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
```

## 10. 原子操作 (`src/op/builtin.cc:409-457`)

### 10.1 原子加法

```cpp
// 元素级原子加法
TIR_DEFINE_TL_BUILTIN(atomic_add_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 元素级原子加法（带返回值）
TIR_DEFINE_TL_BUILTIN(atomic_add_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 向量化原子加法 x2
TIR_DEFINE_TL_BUILTIN(atomic_addx2_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 向量化原子加法 x4
TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 10.2 原子加载/存储

```cpp
// 原子加载
TIR_DEFINE_TL_BUILTIN(atomic_load_elem_op)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 原子存储
TIR_DEFINE_TL_BUILTIN(atomic_store_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 10.3 原子最大/最小

```cpp
// 元素级原子最大值
TIR_DEFINE_TL_BUILTIN(atomic_max_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 元素级原子最大值（带返回值）
TIR_DEFINE_TL_BUILTIN(atomic_max_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 元素级原子最小值
TIR_DEFINE_TL_BUILTIN(atomic_min_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 元素级原子最小值（带返回值）
TIR_DEFINE_TL_BUILTIN(atomic_min_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 11. 随机数生成 (`src/op/builtin.cc:130-140`)

```cpp
// RNG 初始化
TIR_DEFINE_TL_BUILTIN(rng_init).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// RNG 随机数生成
TIR_DEFINE_TL_BUILTIN(rng_rand).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// RNG 浮点随机数
TIR_DEFINE_TL_BUILTIN(rng_rand_float)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 12. 线程束归约 (`src/op/builtin.cc:474-498`)

```cpp
// 线程束归约求和
TIR_DEFINE_TL_BUILTIN(warp_reduce_sum)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束归约最大值
TIR_DEFINE_TL_BUILTIN(warp_reduce_max)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束归约最小值
TIR_DEFINE_TL_BUILTIN(warp_reduce_min)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束归约按位与
TIR_DEFINE_TL_BUILTIN(warp_reduce_bitand)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 线程束归约按位或
TIR_DEFINE_TL_BUILTIN(warp_reduce_bitor)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 13. 其他内置函数

### 13.1 打包操作 (`src/op/builtin.cc:316-318`)

```cpp
// 打包两个 b16 值为 b32
TIR_DEFINE_TL_BUILTIN(pack_b16).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));
```

### 13.2 程序依赖 (`src/op/builtin.cc:350-362`)

```cpp
// 程序化依赖触发
TIR_DEFINE_TL_BUILTIN(pdl_trigger)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 程序化依赖同步
TIR_DEFINE_TL_BUILTIN(pdl_sync).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// 循环中断
TIR_DEFINE_TL_BUILTIN(loop_break)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 13.3 GEMM 操作 (`src/op/builtin.cc:363-369`)

```cpp
// TileLang GEMM
TIR_DEFINE_TL_BUILTIN(tl_gemm).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// TileLang 稀疏 GEMM
TIR_DEFINE_TL_BUILTIN(tl_gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 13.4 Shuffle 和描述符 (`src/op/builtin.cc:389-407`)

```cpp
// Shuffle elect
TIR_DEFINE_TL_BUILTIN(tl_shuffle_elect)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// 初始化 WGMMA 描述符
TIR_DEFINE_TL_BUILTIN(initialize_wgmma_descriptor)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 初始化 TCGEN05 描述符
TIR_DEFINE_TL_BUILTIN(initialize_tcgen05_descriptor)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 增加描述符偏移
TIR_DEFINE_TL_BUILTIN(increase_descriptor_offset)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 13.5 设备断言 (`src/op/builtin.cc:459-467`)

```cpp
// 设备断言
TIR_DEFINE_TL_BUILTIN(device_assert)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// 设备断言（带消息）
TIR_DEFINE_TL_BUILTIN(device_assert_with_msg)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 14. 属性命名空间 (`src/op/builtin.h:24-42`)

```cpp
namespace attr {
static constexpr const char *kSafeValueMap = "safe_value_map";
static constexpr const char *kWarpSpecializationScope =
    "kWarpSpecializationScope";
static constexpr const char *kCustomWarpSpecialization =
    "kCustomWarpSpecialization";

// 控制 PTX 异步拷贝重写的循环注解键
static constexpr const char *kLoopPreferAsync = "parallel_prefer_async";

// 控制是否省略异步提交/等待的循环注解键
static constexpr const char *kParallelAsyncWithoutAsyncCommitWait =
    "parallel_async_without_async_commit_wait";

static constexpr const char *kLocalVarInit = "tl.local_var_init";

// 携带不应标记为 restrict 的 handle Var 列表的函数级属性
static constexpr const char *kNonRestrictParams = "tl.non_restrict_params";
} // namespace attr
```

## 15. 宏定义模式

### 15.1 TIR_DEFINE_TL_BUILTIN 宏 (`src/op/builtin.cc:52-58`)

```cpp
#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)
```

这个宏用于：
1. 定义一个返回 `Op` 引用的函数
2. 注册操作到 TVM 操作表
3. 设置脚本打印名称属性

### 15.2 使用示例

```cpp
TIR_DEFINE_TL_BUILTIN(__exp)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

展开后等价于：

```cpp
const Op &__exp() {
  static const Op &op = Op::Get("tl.__exp");
  return op;
}
TVM_REGISTER_OP("tl.__exp")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "__exp")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

## 16. 效果类型 (CallEffectKind)

TileLang 使用 `CallEffectKind` 标记操作的效果类型：

| 类型 | 值 | 说明 |
|-----|---|------|
| `kPure` | 0 | 纯函数，无副作用，相同输入总是产生相同输出 |
| `kOpaque` | 1 | 不透明操作，可能有副作用或无法分析 |

快速数学函数和 IEEE 运算标记为 `kOpaque`，因为它们通常映射到内联汇编或内在函数。
内存访问和同步操作标记为 `kOpaque`。

## 17. 文件引用汇总

| 文件 | 行数 | 主要功能 |
|-----|------|---------|
| `src/op/builtin.h` | 946 | 内置函数声明和配置常量 |
| `src/op/builtin.cc` | 546 | 内置函数注册和实现 |

## 18. 函数分类统计

| 类别 | 函数数量 | 主要函数 |
|-----|---------|---------|
| 快速数学 | 8 | `__exp`, `__log`, `__sin`, `__cos`, etc. |
| IEEE 运算 | 11 | `ieee_add`, `ieee_mul`, `ieee_fmaf`, etc. |
| 内存访问 | 15 | `access_ptr`, `tma_load`, `ldg32`, etc. |
| 同步原语 | 12 | `sync_warp`, `cluster_sync`, `mbarrier_wait`, etc. |
| 张量核心 | 12 | `ptx_wgmma_ss`, `ptx_tcgen05_mma_ss`, etc. |
| 原子操作 | 10 | `atomic_add_elem_op`, `atomic_max_elem_op`, etc. |
| 线程查询 | 4 | `get_lane_idx`, `get_warp_idx`, etc. |
| 其他 | 10 | `rng_init`, `pack_b16`, `loop_break`, etc. |

总计约 80+ 个内置函数。
