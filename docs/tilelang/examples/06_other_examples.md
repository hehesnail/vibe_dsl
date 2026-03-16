# TileLang 其他示例详解

## 概述

本文档介绍 TileLang 项目中除基础 GEMM 和 FlashAttention 之外的其他重要示例，包括 DeepSeek NSA、Mamba2、Attention Sink、卷积运算、布局可视化和分析工具等。

---

## 1. DeepSeek Native Sparse Attention (NSA)

### 1.1 概述

`examples/deepseek_nsa/` 实现了 DeepSeek 提出的原生稀疏注意力机制（Native Sparse Attention, NSA）。NSA 是一种硬件友好的稀疏注意力架构，通过块级稀疏性选择和压缩来加速长序列处理。

### 1.2 文件结构

```
examples/deepseek_nsa/
├── example_tilelang_nsa_fwd.py      # TileLang 前向实现
├── example_tilelang_nsa_fwd_varlen.py # 变长序列前向
├── example_tilelang_nsa_bwd.py      # TileLang 反向传播
├── example_tilelang_nsa_decode.py   # 解码阶段优化
├── example_triton_nsa_fwd.py        # Triton 参考实现
├── example_triton_nsa_fwd_varlen.py # Triton 变长实现
├── example_triton_nsa_bwd.py        # Triton 反向实现
├── reference.py                     # PyTorch 参考实现
└── benchmark/                       # 性能基准测试
```

### 1.3 核心算法

NSA 的核心思想是通过块选择机制只计算重要的注意力块：

```python
# examples/deepseek_nsa/example_tilelang_nsa_fwd.py:46-118
@T.prim_func
def native_sparse_attention(
    Q: T.Tensor(q_shape, dtype),
    K: T.Tensor(kv_shape, dtype),
    V: T.Tensor(kv_shape, dtype),
    BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
    Output: T.Tensor(q_shape, dtype),
):
    with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([G, BK], dtype)
        K_shared = T.alloc_shared([BS, BK], dtype)
        V_shared = T.alloc_shared([BS, BV], dtype)
        O_shared = T.alloc_shared([G, BV], dtype)

        acc_s = T.alloc_fragment([G, BS], accum_dtype)
        acc_s_cast = T.alloc_fragment([G, BS], dtype)
        acc_o = T.alloc_fragment([G, BV], accum_dtype)
        scores_max = T.alloc_fragment([G], accum_dtype)
        scores_max_prev = T.alloc_fragment([G], accum_dtype)
        scores_scale = T.alloc_fragment([G], accum_dtype)
        scores_sum = T.alloc_fragment([G], accum_dtype)
        logsum = T.alloc_fragment([G], accum_dtype)

        i_t, i_v, i_bh = bx, by, bz
        i_b, i_h = i_bh // head_kv, i_bh % head_kv

        NS = S
        T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))

        # 遍历选中的块索引
        for i in T.Pipelined(NS, num_stages=num_stages):
            i_s = BlockIndices[i_b, i_t, i_h, i] * BS
            if i_s <= i_t and i_s >= 0:  # 只处理有效的块
                T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Softmax 计算
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                for i in T.Parallel(G):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(G):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                # 累加注意力输出
                for i, j in T.Parallel(G, BV):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        for i, j in T.Parallel(G, BV):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, O_shared)
        T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])
```

### 1.4 解码优化

`example_tilelang_nsa_decode.py` 针对解码阶段（seq_len=1）进行了特殊优化：

```python
# examples/deepseek_nsa/example_tilelang_nsa_decode.py:21-129
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def native_sparse_attention(
    batch, heads, seq_len, dim, scale=None, block_size=64, groups=1, selected_blocks=16
):
    # 修改形状以支持推理（q 的 seq_len=1）
    q_shape = [batch, 1, heads, dim]  # seq_len 改为 1
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, 1, head_kv, selected_blocks]
```

---

## 2. Mamba2 状态空间模型

### 2.1 概述

`benchmark/mamba2/` 包含 Mamba2 模型的 Chunk Scan 操作实现。Mamba2 是状态空间模型（SSM）的优化版本，使用硬件感知的算法设计。

### 2.2 核心实现

```python
# benchmark/mamba2/benchmark_mamba_chunk_scan.py:183-340
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def chunk_scan_fwd(
    batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
    block_M=64, block_N=64, block_K=64, block_Dstate=128, num_stages=2, threads=128
):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),
        D: T.Tensor((nheads), dtype),
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),
    ):
        with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=threads) as (
            bz, bx, by
        ):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype)
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype)
            dA_cs_m_shared = T.alloc_shared((block_M), dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype)
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 计算状态衰减和输出
            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)

            # 计算 prev_state 贡献
            T.copy(C[...], C_shared)
            T.copy(prev_states[...], prev_state_shared)
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            # 计算 chunk 内贡献
            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(cb[...], cb_shared)
                T.copy(cb_shared, cb_local)
                # 应用衰减
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.gemm(cb_local, x_shared, acc_o)

            # 添加残差连接
            D_local[0] = D[bz]
            T.copy(x[...], x_residual_shared)
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[...])
```

---

## 3. Attention Sink

### 3.1 概述

`examples/attention_sink/` 实现了 Attention Sink 机制，这是一种在流式 LLM 中保持注意力稳定性的技术。通过在注意力计算中引入"sink"项，可以在使用滑动窗口注意力的同时保持模型性能。

### 3.2 算法原理

Attention Sink 在标准 FlashAttention 的基础上添加了一个 sink 项：

```python
# examples/attention_sink/example_mha_sink_fwd_bhsd.py:52-131
@T.prim_func
def main(
    Q: T.Tensor(q_shape, dtype),
    K: T.Tensor(kv_shape, dtype),
    V: T.Tensor(kv_shape, dtype),
    Output: T.Tensor(q_shape, dtype),
    Sinks: T.Tensor([heads], dtype),
):
    with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
        # ... 内存分配 ...
        sinks = T.alloc_fragment([block_M], dtype)

        T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        for i in T.Parallel(block_M):
            sinks[i] = Sinks[by]

        # 滑动窗口注意力计算
        start = T.max(0, (bx * block_M + past_len - window_size) // block_N) if window_size is not None else 0
        end = T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))

        for k in T.Pipelined(start, end, num_stages=num_stages):
            T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
            # 应用因果掩码和滑动窗口掩码
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                if window_size is not None:
                    acc_s[i, j] = T.if_then_else(q_idx >= k_idx and q_idx < k_idx + window_size, 0, -T.infinity(acc_s.dtype))
                else:
                    acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            # Softmax 计算
            # ...

        # 关键修改：添加 sink 项到 softmax 分母
        for i in T.Parallel(block_M):
            logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, O_shared)
        T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
```

### 3.3 性能基准

根据 README 中的数据，在 H800 上的性能表现：

| SEQ_LEN | headdim | Triton TFLOPs | TileLang TFLOPs | Speedup |
|---------|---------|---------------|-----------------|---------|
| 2048    | 64      | 232.98        | **281.89**      | 1.21x   |
| 2048    | 128     | 321.55        | **417.98**      | 1.30x   |
| 4096    | 64      | 280.70        | **349.47**      | 1.25x   |
| 4096    | 128     | 369.61        | **497.13**      | 1.35x   |
| 8192    | 64      | 299.04        | **385.56**      | 1.29x   |
| 8192    | 128     | 399.39        | **507.93**      | 1.27x   |

---

## 4. 卷积运算

### 4.1 概述

`examples/convolution/` 展示了如何使用 TileLang 实现高效的 2D 卷积运算，包括 Im2Col 转换和自动调优。

### 4.2 核心实现

```python
# examples/convolution/example_convolution.py:27-69
@tilelang.jit(out_idx=[2])
def convolution(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = T.float16
    accum_dtype = T.float32
    is_hopper = check_hopper()

    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                if is_hopper:
                    # Hopper 架构使用专门的 Im2Col 指令
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    # 通用 Im2Col 实现
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                        access_w = m % OW * S + k // C % KW * D - P
                        in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
                        data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])
```

### 4.3 自动调优版本

```python
# examples/convolution/example_convolution_autotune.py
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[2])
def convolution_autotune(N, C, H, W, F, K, S, D, P, block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
    # ... 自动调优配置
```

---

## 5. 布局可视化

### 5.1 概述

`examples/plot_layout/` 展示了如何使用 TileLang 的布局可视化工具来理解和调试内存布局。

### 5.2 MMA 布局可视化

```python
# examples/plot_layout/fragment_mma_load_a.py:8-98
def make_mma_load_base_layout(dtype: T.dtype = T.float16, matrix: Literal["A", "B"] = "A", transposed: bool = False) -> T.Fragment:
    """
    创建 MMA (Matrix Multiply Accumulate) 结果存储到片段缓冲区的布局函数。
    """
    from tilelang.intrinsics.mma_layout import (
        shared_16x8_to_mma_32x4_layout_sr_a,
        shared_16x16_to_mma_32x8_layout_sr_a,
        shared_16x32_to_mma_32x16_layout_sr_a,
    )

    assert matrix in ["A", "B"], "matrix should be either A or B"
    dtype_bits = DataType(dtype).bits

    # s 表示空间轴，r 表示归约轴
    # sr 表示两个维度是空间+归约
    transform_func_sr_a = None
    if dtype_bits == 32:
        transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
    elif dtype_bits == 16:
        transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
    elif dtype_bits == 8:
        transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a

    is_sr_conditions = [False]
    is_sr_conditions.append(matrix == "A" and not transposed)
    is_sr_conditions.append(matrix == "B" and transposed)
    is_sr_axis_order = any(is_sr_conditions)

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(dtype)
    transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)

    inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

    def forward_thread(i: int, j: int) -> int:
        lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
        return lane_id

    def forward_index(i: int, j: int) -> int:
        _, local_id = inverse_mma_load_layout.map_indices([i, j])
        return local_id

    base_fragment = T.Fragment(
        [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
        forward_thread_fn=forward_thread,
        forward_index_fn=forward_index,
    )
    return base_fragment


# 创建 16x16 矩阵布局用于 ldmatrix 操作
base_layout = make_mma_load_base_layout(dtype=T.float16, matrix="A", transposed=False)
print(base_layout)
plot_layout(base_layout, name="base_layout")

# 创建 warp 级布局 32x16
warp_layout = base_layout.repeat([block_rows, 1], repeat_on_thread=True).replicate(block_cols)
plot_layout(warp_layout, name="warp_layout")

# 创建 block 级布局 128x32
block_layout = warp_layout.repeat([warp_rows, chunk], repeat_on_thread=False, lower_dim_first=False)
plot_layout(block_layout, name="block_layout")
```

### 5.3 布局转换示例

```python
# examples/plot_layout/layout_transform.py
import tilelang.language as T
from tilelang.tools import plot_layout

# 示例 1: 简单的 2D 转置 (4x4)
transpose_layout = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(transpose_layout, name="transpose_4x4")

# 示例 2: 3D -> 2D reshape + 转置
reshape_layout = T.Layout([2, 4, 8], lambda i, j, k: (k, i * 4 + j))
plot_layout(reshape_layout, name="reshape_3d_to_2d")

# 示例 3: 交错布局
interleave = T.Layout([8, 4], lambda i, j: (i % 4 * 2 + i // 4, j))
plot_layout(interleave, name="interleave_8x4")
```

---

## 6. 性能分析工具

### 6.1 概述

`examples/analyze/` 提供了 TVM IR 性能分析工具，可以计算 FLOPs、内存带宽利用率和执行时间估计。

### 6.2 分析器使用

```python
# examples/analyze/example_gemm_analyze.py
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
import torch

M = N = K = 1024

def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None, enable_rasteration=None):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return matmul

cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
result = Analyzer.analysis(kernel(128, 128, 32, 3, 128, True), cuda_device)

print(f"Analyzed FLOPs: {result.total_flops}")
print(f"Expected FLOPs: {2 * M * N * K}")
```

### 6.3 分析结果

```python
# AnalysisResult 数据结构
@dataclass(frozen=True)
class AnalysisResult:
    total_flops: int          # 总浮点运算次数
    total_global_bytes: int   # 全局内存流量（字节）
    estimated_time: float     # 预测执行时间（秒）
    tflops: float             # 达到的 TFLOPS
    bandwidth_GBps: float     # 内存带宽利用率
```

### 6.4 卷积分析示例

```python
# examples/analyze/example_conv_analyze.py
def kernel(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads):
    @T.prim_func
    def conv(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        # ... 卷积实现 ...
    return conv

cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
result = Analyzer.analysis(kernel(64, 256, 512, 512, 512, 3, 1, 1, 1, 64, 128, 32, 3, 256), cuda_device)
print(f"Analyzed FLOPs: {result.total_flops}")
```

---

## 7. 总结

这些示例展示了 TileLang 在不同深度学习算子上的应用能力：

1. **DeepSeek NSA**：展示了如何处理稀疏注意力模式
2. **Mamba2**：展示了状态空间模型的优化实现
3. **Attention Sink**：展示了注意力机制的变体优化
4. **卷积运算**：展示了传统 CV 算子的实现
5. **布局可视化**：展示了内存布局的调试工具
6. **性能分析**：展示了编译时性能预测能力

每个示例都遵循 TileLang 的核心设计原则：
- 使用 Python DSL 描述计算
- 通过 JIT 编译生成高效 CUDA 代码
- 支持自动调优优化性能
- 提供与 PyTorch 的无缝集成
