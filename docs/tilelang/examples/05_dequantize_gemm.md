# TileLang 量化 GEMM 示例详解

## 概述

`examples/dequantize_gemm/` 目录包含 TileLang 中量化矩阵乘法（Quantized GEMM）的实现示例。这些示例展示了如何在 GPU 上高效地进行低精度量化矩阵运算，包括 FP4、INT4、W4A8 等多种量化格式。

## 文件结构

```
examples/dequantize_gemm/
├── README.md                           # 示例说明文档
├── dequantize_utils.py                 # 反量化工具函数
├── example_dequant_gemm_fp4_hopper.py  # FP4 量化 GEMM (Hopper架构)
├── example_dequant_gemm_bf16_fp4_hopper.py  # BF16+FP4 混合精度
├── example_dequant_gemm_bf16_mxfp4_hopper.py # BF16+MXFP4 混合精度
├── example_dequant_gemm_fine_grained.py # 细粒度量化 GEMM
├── example_dequant_gemm_w4a8.py        # W4A8 (4bit权重8bit激活)
├── example_dequant_gemv_fp16xint4.py   # FP16xINT4 GEMV
├── example_dequant_groupedgemm_bf16_mxfp4_hopper.py # Grouped GEMM
├── regression_example_dequantize_gemm.py # 回归测试
└── test_example_dequantize_gemm.py     # 单元测试
```

## 核心概念

### 1. 量化格式

#### FP4 (4-bit Floating Point)

FP4 格式使用 4 位表示浮点数，格式为 `s1e2m1`（1位符号、2位指数、1位尾数）：

```python
# examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py:10-28
def _tir_u8_to_f4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.float16
    assert val.dtype == T.uint8
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + ExponentialBias(f16, f4) = e_f4 + (2^4 - 2^1) = e_f4 + 14
    # s1e2m1 格式
    mask = tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)  # 符号位
    e_f4 = (f4 & tir.const(6, T.uint16)) >> tir.const(1, T.uint16)  # 指数
    e_f16 = e_f4 + tir.const(14, T.uint16)  # 转换到 FP16 指数
    m_f4 = f4 & tir.const(1, T.uint16)  # 尾数
    m_f16 = m_f4
    val_f16 = tir.reinterpret(
        T.float16, ((e_f16 | (s << tir.const(5, T.uint16))) << tir.const(10, T.uint16) | m_f16 << tir.const(9, T.uint16)).astype(T.uint16)
    )
    return val_f16
```

#### INT4 (4-bit Integer)

INT4 格式使用 4 位表示有符号整数：

```python
# examples/dequantize_gemm/example_dequant_gemm_w4a8.py:10-21
def _tir_u8_to_i4_to_i8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.int8
    assert val.dtype == T.uint8

    mask = tir.const((1 << nbit) - 1, T.uint8)
    i4 = (val >> (pos.astype(T.uint8) * tir.const(nbit, T.uint8))) & mask

    i8_shifted = tir.reinterpret(T.int8, i4 << tir.const(4, T.uint8))
    i8 = i8_shifted >> tir.const(4, T.int8)  # 符号扩展
    return i8
```

### 2. 存储布局

量化权重通常将多个元素打包到一个字节中存储：

```python
# examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py:60-66
num_elems_per_byte = 8 // num_bits  # 每个字节存储的元素数
storage_dtype = T.uint8
B_shape = (N, K // num_elems_per_byte)  # 压缩后的形状
B_shared_shape = (block_N, block_K // num_elems_per_byte)
B_dequantize_shared_shape = (block_N, block_K)
```

## 主要示例详解

### 1. FP4 Hopper GEMM

`example_dequant_gemm_fp4_hopper.py` 实现了基于 NVIDIA Hopper 架构的 FP4 量化 GEMM：

```python
# examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py:124-222
def matmul(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits=4, tune=False):
    @tilelang.jit(out_idx=[2])
    def kernel_func(block_M, block_N, block_K, num_stages, threads, split=1):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = T.uint8
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout(
                    {
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    }
                )

                T.clear(Ct_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    # 反量化操作
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_f4_to_f16(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, Ct_shared)
                T.copy(Ct_shared, Ct[bx * block_N : (bx + 1) * block_N, by * block_M : (by + 1) * block_M])
```

**关键特性**：
- 使用 `T.Pipelined` 实现多级流水线并行
- 通过 `make_swizzled_layout` 优化共享内存访问模式
- 支持 Split-K 算法以提高并行度

### 2. W4A8 量化 GEMM

`example_dequant_gemm_w4a8.py` 实现了 4-bit 权重和 8-bit 激活的混合精度 GEMM：

```python
# examples/dequantize_gemm/example_dequant_gemm_w4a8.py:95-163
def matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits=4, tune=False):
    @tilelang.jit(out_idx=[2])
    def kernel_func(block_M, block_N, block_K, num_stages, threads):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = T.uint8
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_local_shape = (block_N, block_K)

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout(
                    {
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    }
                )

                T.clear(Ct_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, Ct_shared)
                T.copy(Ct_shared, Ct[bx * block_N : (bx + 1) * block_N, by * block_M : (by + 1) * block_M])
```

### 3. FP16xINT4 GEMV

`example_dequant_gemv_fp16xint4.py` 实现了矩阵-向量乘法（GEMV）的量化版本：

```python
# examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py:11-155
@tilelang.jit
def dequantize_gemv(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_bits: int = 4,
    storage_dtype: T.dtype = T.int8,
    source_format: str = "uint",
    n_partition: int = 4,
    reduce_thread: int = 32,
    fast_decoding: bool = False,
    trans_A: bool = False,
    trans_B: bool = True,
    group_size: int = -1,
    with_scaling: bool = False,
) -> Callable[..., Any]:
    # 计算微内核大小
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    block_K = reduce_thread * micro_size_k

    @T.prim_func
    def main(
        A: T.Tensor[A_shape, in_dtype],
        B: T.Tensor[B_shape, storage_dtype],
        C: T.Tensor[C_shape, out_dtype],
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            M,
            threads=(reduce_thread, n_partition),
        ) as (bx, by):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(import_source)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed) + kr * micro_size_k_compressed + v,
                    ]

                if fast_decoding:
                    # 使用 LOP3 指令进行快速解码
                    T.call_extern(
                        func_name,
                        T.access_ptr(B_quant_local, "r"),
                        T.access_ptr(B_dequantize_local, "w"),
                        dtype=in_dtype,
                    )
                else:
                    for ki in T.serial(micro_size_k):
                        B_dequantize_local[ki] = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                            num_bits, B_quant_local[ki // num_elems_per_byte], ki % num_elems_per_byte, in_dtype
                        )

                if use_dp4a:
                    # 使用 DP4A 指令加速
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_dequantize_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki] * B_dequantize_local[ki]

            # 线程间规约
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]
```

**关键特性**：
- 使用 `T.thread_binding` 显式绑定线程索引
- 支持 `fast_decoding` 使用 LOP3 指令加速反量化
- 支持 `dp4a` 指令进行 8-bit 整数点积运算

### 4. 细粒度量化 GEMM

`example_dequant_gemm_fine_grained.py` 展示了更细粒度的量化控制，支持 Ladder 布局转换：

```python
# examples/dequantize_gemm/example_dequant_gemm_fine_grained.py:134-155
def tl_matmul_with_ladder_weight_only_transform_block_reduce_int4(
    M, N, K, in_dtype, out_dtype, accum_dtype, transform_b
):
    # 使用 MMA (Matrix Multiply Accumulate) 指令
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        reduce_k=reduce_k,
        transform_kind_b=transform_b,
        num_elems_per_byte=num_elems_per_byte,
    )

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads, prelude=decode_i4_to_f16) as (bx, by):
            # ... 内存分配 ...
            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # 使用 MMA emitter 进行高效的矩阵乘法
                mma_emitter.ldmatrix_a(A_local, A_shared, ki, rk=rk)
                mma_emitter.ldmatrix_b(B_local, B_shared, ki, rk=rk)
                # 反量化
                T.call_extern("handle", "decode_i4u_to_f16", ...)
                mma_emitter.mma(A_local, B_dequantize_local, C_local)
```

## 工具函数

### 反量化工具

`dequantize_utils.py` 提供了 PyTorch 实现的参考反量化函数：

```python
# examples/dequantize_gemm/dequantize_utils.py:4-54
def torch_convert_bit_twiddling(tensor):
    """
    将 uint8 张量转换为 bf16 张量。
    每个输出元素通过组合两个字节并提取 bf16 模式的 16 位来生成。
    """
    assert tensor.dim() == 2 and tensor.dtype == torch.uint8
    N, K = tensor.shape
    assert K % 2 == 0, "Number of columns must be even"

    # 将 uint8 值对组合成 uint32 以在 CUDA 上进行安全的位操作
    val0 = tensor[:, 0::2].to(torch.int32)
    val1 = tensor[:, 1::2].to(torch.int32)
    val_concat = (val0 << 8) | val1  # (N, K//2), uint32

    # 位操作解码
    mask = 0b1000000111000000
    mask1 = 0b1000000000000000
    mask2 = 0b0000000110000000
    mask3 = 0b0000000001000000

    res0 = val_concat_expanded & mask
    res1 = (val_concat_expanded << 3) & mask
    res2 = (val_concat_expanded << 6) & mask
    res3 = ((val_concat_expanded << 1) & mask1) | ((val_concat_expanded >> 3) & mask2) | ((val_concat_expanded >> 7) & mask3)

    bf16 = torch.where(pos == 0, res0, torch.where(pos == 1, res1, torch.where(pos == 2, res2, res3)))
    bf16_uint16 = (bf16 & 0xFFFF).to(torch.uint16)
    bf16_bf16 = bf16_uint16.view(torch.bfloat16)
    bf16_new = bf16_bf16 * (2.0**126)  # 指数缩放
    return bf16_new
```

## 性能调优

### 自动调优配置

```python
# examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py:111-121
def get_configs():
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [128, 256]
    num_stages = [1, 2]
    threads = [128, 256]
    splits = [1]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads, splits))

    configs = [{"block_M": c[0], "block_N": c[1], "block_K": c[2], "num_stages": c[3], "threads": c[4], "split": c[5]} for c in _configs]
    return configs
```

### 使用 Roller 优化搜索空间

TileLang 支持使用 BitBLAS Roller 来推导优化的搜索空间：

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA

carve_template = MatmulTemplate(
    M=M, N=N, K=K,
    in_dtype=T.float16,
    out_dtype=T.float16,
    accum_dtype=T.float32,
).with_arch(arch)

roller_hints = carve_template.recommend_hints(topk=10)
```

## 测试与验证

### 单元测试

```python
# examples/dequantize_gemm/test_example_dequantize_gemm.py
def test_dequantize_gemm():
    # 测试 FP4 转换
    test_fp4_fp16_convert_close()
    # 测试 GEMM 正确性
    main(m=256, n=256, k=256, tune=False)
```

### 性能回归测试

```python
# examples/dequantize_gemm/regression_example_dequantize_gemm.py
def regression_example_dequantize_gemm():
    return run_regression_perf(m=4096, n=4096, k=4096)
```

## 最佳实践

1. **内存布局优化**：使用 `make_swizzled_layout` 避免共享内存 bank 冲突
2. **流水线并行**：使用 `T.Pipelined` 隐藏内存访问延迟
3. **量化格式选择**：根据硬件支持选择合适的量化格式（FP4/INT4/INT8）
4. **快速解码**：在支持的硬件上使用 LOP3 指令加速反量化
5. **自动调优**：使用 TileLang 的自动调优器搜索最优配置

## 参考

- [BitBLAS 项目](https://github.com/microsoft/BitBLAS) - 更复杂的量化 GEMM 实现
- `testing/python/kernel/test_tilelang_dequantize_gemm.py` - 更多测试用例
