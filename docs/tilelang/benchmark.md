# TileLang 基准测试框架详解

## 概述

TileLang 的基准测试框架位于 `benchmark/` 目录，提供了系统化的性能评估工具。该框架支持矩阵乘法、Mamba2 Chunk Scan、FP8 运算等多种算子的性能基准测试，并与自动调优功能紧密集成。

## 文件结构

```
benchmark/
├── matmul/                          # 矩阵乘法基准测试
│   ├── README.md                    # 性能结果文档
│   ├── benchmark_matmul.py          # 基础 MatMul 基准
│   ├── benchmark_matmul_intrinsic.py # 使用 Intrinsic 的 MatMul
│   └── benchmark_matmul_sp.py       # 稀疏 MatMul 基准
├── mamba2/                          # Mamba2 状态空间模型基准
│   ├── README.md                    # 说明文档
│   ├── benchmark_mamba_chunk_scan.py # Chunk Scan 基准
│   └── mamba_benchmark_result.png   # 性能结果图表
├── matmul_fp8/                      # FP8 矩阵乘法基准
└── blocksparse_attention/           # 块稀疏注意力基准
```

## 矩阵乘法基准测试

### 1. 基础 MatMul 基准 (`benchmark_matmul.py`)

#### 1.1 核心功能

```python
# benchmark/matmul/benchmark_matmul.py:116-213
@autotune(
    configs=get_configs,
    warmup=3,
    rep=20,
)
@jit(
    out_idx=[2],
)
def matmul(
    M, N, K, with_roller,
    block_M=None, block_N=None, block_K=None,
    num_stages=None, thread_num=None, policy=None, enable_rasteration=None
):
    """
    创建自动调优的矩阵乘法核函数

    矩阵形状：
      - A: (M, K)
      - B: (N, K)
      - C: (M, N)

    Returns
    -------
    (best_latency, best_config, ref_latency)
        best_latency : float - 找到的最佳延迟
        best_config : dict - 产生最佳延迟的参数配置
        ref_latency : float - 参考程序的基线延迟
    """
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
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
                T.gemm(
                    A_shared, B_shared, C_local,
                    transpose_B=True, policy=policy
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
```

#### 1.2 配置生成

```python
# benchmark/matmul/benchmark_matmul.py:33-105
def get_configs(args, kwargs):
    """
    生成用于调优的配置字典列表

    Parameters
    ----------
    with_roller : bool
        是否启用 BitBLAS roller 推导搜索空间

    Returns
    -------
    list of dict
        每个配置字典包含各种块大小、流水线阶段、线程数等参数
    """
    M, N, K, with_roller = args[:4]

    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA, CDNA
        from tilelang.carver.roller.rasterization import NoRasterization
        import torch

        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        topk = 10

        carve_template = MatmulTemplate(
            M=M, N=N, K=K,
            in_dtype=T.float16, out_dtype=T.float16, accum_dtype=T.float32
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        roller_hints = carve_template.recommend_hints(topk=topk)

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            config["policy"] = T.GemmWarpPolicy.from_warp_partition(block_rows, block_cols)
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        # 默认网格搜索配置
        iter_params = dict(
            block_M=[64, 128, 256],
            block_N=[64, 128, 256],
            block_K=[32, 64],
            num_stages=[0, 1, 2, 3],
            thread_num=[128, 256],
            policy=[T.GemmWarpPolicy.Square],
            enable_rasteration=[True, False],
        )
        return [{k: v for k, v in zip(iter_params, values)}
                for values in itertools.product(*iter_params.values())]
    return configs
```

#### 1.3 运行基准测试

```python
# benchmark/matmul/benchmark_matmul.py:216-248
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--with_roller", action="store_true",
                        help="Whether to enable BitBLAS roller for search space")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    with_roller = args.with_roller

    total_flops = 2 * M * N * K
    best_result = matmul(M, N, K, with_roller)
    best_latency = best_result.latency
    best_config = best_result.config
    ref_latency = best_result.ref_latency

    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")

    if ref_latency is not None:
        print(f"Reference TFlops: {total_flops / ref_latency * 1e-9:.3f}")
```

### 1.4 性能结果

根据 `benchmark/matmul/README.md`，在 NVIDIA H800 SXM 上的性能表现：

| K     | Latency (s) | Throughput (TFLOPs) |
|-------|-------------|---------------------|
| 256   | 0.089056    | 386                 |
| 512   | 0.132064    | 520                 |
| 1024  | 0.218816    | 628                 |
| 2048  | 0.390112    | 705                 |
| 4096  | 0.746752    | 736                 |
| 8192  | 1.449888    | 758                 |
| 16384 | 2.871168    | 766                 |

## Mamba2 Chunk Scan 基准测试

### 2.1 概述

`benchmark_mamba_chunk_scan.py` 实现了 Mamba2 模型的核心算子 Chunk Scan 的性能基准测试，并与 Triton 和 Helion 实现进行对比。

### 2.2 参考实现

```python
# benchmark/mamba2/benchmark_mamba_chunk_scan.py:24-69
def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    """
    参数说明：
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) 或 (nheads,)
    返回：
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size

    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)

    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp", scores_decay.to(x.dtype), dt.to(x.dtype),
        rearrange(x, "b (c s) h p -> b c s h p", c=nchunks)
    )
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = (
        torch.einsum("bclhn,bchpn->bclhp", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks), prev_states.to(C.dtype))
        * state_decay_out
    )
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out
```

### 2.3 Triton 对比实现

```python
# benchmark/mamba2/benchmark_mamba_chunk_scan.py:72-74
def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states, D):
    out, _ = _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D)
    return out
```

### 2.4 TileLang 实现

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
    block_M=64, block_N=64, block_K=64, block_Dstate=128,
    num_stages=2, threads=128
):
    """TileLang Chunk Scan 前向实现"""
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
        with T.Kernel(
            nheads,
            T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
            batch * nchunks,
            threads=threads
        ) as (bz, bx, by):
            # 内存分配
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

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout({
                cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared),
            })

            T.no_set_max_nreg()

            # 计算状态衰减
            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)

            # 计算 prev_state 贡献
            T.copy(
                C[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                  bz // (nheads // ngroups), 0:block_Dstate],
                C_shared
            )
            T.copy(
                prev_states[batch_idx, chunk_idx, bz, n_idx * block_N : (n_idx + 1) * block_N, 0:block_Dstate],
                prev_state_shared
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            # 计算 chunk 内贡献
            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    cb[batch_idx, chunk_idx, bz // (nheads // ngroups),
                       m_idx * block_M : (m_idx + 1) * block_M, k * block_K : (k + 1) * block_K],
                    cb_shared
                )
                T.copy(cb_shared, cb_local)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0)
                T.copy(
                    x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                      bz, n_idx * block_N : (n_idx + 1) * block_N],
                    x_shared
                )
                T.gemm(cb_local, x_shared, acc_o)

            # 添加残差连接
            D_local[0] = D[bz]
            T.copy(
                x[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                  bz, n_idx * block_N : (n_idx + 1) * block_N],
                x_residual_shared
            )
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                       bz, n_idx * block_N : (n_idx + 1) * block_N]
            )

    return main
```

### 2.5 运行基准测试

```python
# benchmark/mamba2/benchmark_mamba_chunk_scan.py:343-389
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=80, help="heads")
    parser.add_argument("--groups", type=int, default=1, help="groups")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--chunk_size", type=int, default=256, help="chunk size")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument("--dstate", type=int, default=128, help="dstate")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()

    batch, heads, groups, seq_len, chunk_size, dim, dstate = (
        args.batch, args.heads, args.groups, args.seq_len, args.chunk_size, args.dim, args.dstate
    )
    nchunks = math.ceil(seq_len / chunk_size)
    total_flops = 2 * batch * seq_len * chunk_size * heads * dim * 0.5 + 2 * batch * seq_len * heads * dim * dstate

    print("Benchmarking TileLang...")
    kernel = chunk_scan_fwd(batch, seq_len, chunk_size, groups, heads, dim, dstate)
    best_latency = kernel.latency
    best_config = kernel.config
    ref_latency = kernel.ref_latency
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")

    # 准备输入数据
    cb = torch.randn(batch, nchunks, groups, chunk_size, chunk_size).half().cuda()
    x = torch.randn(batch, seq_len, heads, dim).half().cuda()
    dt = torch.randn(batch, heads, nchunks, chunk_size).half().cuda()
    dA_cumsum = torch.randn(batch, heads, nchunks, chunk_size).half().cuda()
    C = torch.randn(batch, seq_len, groups, dstate).half().cuda()
    states = torch.randn(batch, nchunks, heads, dim, dstate).half().cuda()
    D = torch.randn(heads).half().cuda()

    print("Benchmarking Triton...")
    triton_latency = do_bench(
        lambda: chunk_scan_triton(cb, x, dt, dA_cumsum, C, states, D),
        _n_warmup=10, _n_repeat=10
    )
    print(f"Triton TFlops: {total_flops / triton_latency * 1e-9}")

    print("Benchmarking Helion...")
    chunk_scan_helion(cb, x, dt, dA_cumsum, C, states, D)
```

## 使用 Roller 优化搜索空间

TileLang 的基准测试框架支持使用 BitBLAS Roller 来推导优化的搜索空间：

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA, CDNA
from tilelang.carver.roller.rasterization import NoRasterization

# 创建架构对象
arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")

# 创建模板
carve_template = MatmulTemplate(
    M=M, N=N, K=K,
    in_dtype=T.float16,
    out_dtype=T.float16,
    accum_dtype=T.float32
).with_arch(arch)

# 获取 Roller 推荐的配置提示
roller_hints = carve_template.recommend_hints(topk=10)

# 转换为 TileLang 配置
for hint in roller_hints:
    config = {}
    block_m, block_n = hint.block
    warp_m, warp_n = hint.warp
    block_rows, block_cols = block_m // warp_m, block_n // warp_n
    config["block_M"] = block_m
    config["block_N"] = block_n
    config["block_K"] = hint.rstep[0]
    config["num_stages"] = hint.pipeline_stage
    config["thread_num"] = block_rows * block_cols * 32
    config["policy"] = T.GemmWarpPolicy.from_warp_partition(block_rows, block_cols)
    config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
```

## 性能分析指标

### 1. 计算吞吐量 (TFLOPS)

```python
total_flops = 2 * M * N * K  # 对于 GEMM
tflops = total_flops / latency * 1e-9
```

### 2. 内存带宽利用率

```python
# 计算内存访问量
total_bytes = (M * K + N * K + M * N) * dtype_size
bandwidth_gbps = total_bytes / latency / 1e9
```

### 3. 与参考实现对比

```python
# 获取 TileLang 性能
best_result = matmul(M, N, K, with_roller)
tilelang_latency = best_result.latency

# 获取参考性能（如 PyTorch/cuBLAS）
ref_latency = best_result.ref_latency

# 计算加速比
speedup = ref_latency / tilelang_latency
```

## 最佳实践

### 1. 自动调优配置

```python
def get_configs():
    """生成调优配置"""
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[32, 64, 128],
        num_stages=[0, 1, 2, 3, 4],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values))
            for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[...])
def kernel(...):
    pass
```

### 2. 基准测试脚本结构

```python
import argparse
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.profiler import do_bench

def ref_program(...):
    """参考实现（如 PyTorch）"""
    pass

def get_configs():
    """生成调优配置"""
    pass

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[...])
def kernel(...):
    """TileLang 实现"""
    pass

def main():
    """主函数：解析参数、运行基准测试、输出结果"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    # ... 其他参数
    args = parser.parse_args()

    # 运行基准测试
    result = kernel(...)

    # 计算性能指标
    total_flops = ...
    tflops = total_flops / result.latency * 1e-9

    print(f"Best latency: {result.latency}")
    print(f"Best TFLOPS: {tflops}")
    print(f"Best config: {result.config}")

if __name__ == "__main__":
    main()
```

### 3. 使用 CUPTI 后端进行精确计时

```python
from tilelang.profiler import do_bench

# 使用 CUPTI 后端获取更精确的 GPU 计时
latency = do_bench(lambda: kernel(...), backend="cupti")
```

## 运行基准测试

### 1. 基础 MatMul

```bash
cd benchmark/matmul
python benchmark_matmul.py --m 8192 --n 8192 --k 8192

# 使用 Roller 优化搜索空间
python benchmark_matmul.py --m 8192 --n 8192 --k 8192 --with_roller
```

### 2. Mamba2 Chunk Scan

```bash
cd benchmark/mamba2
python benchmark_mamba_chunk_scan.py --batch 8 --heads 80 --seq_len 4096 --chunk_size 256
```

### 3. 性能回归测试

```bash
# 运行所有示例的回归测试
python -m tilelang.testing.regression

# 以 JSON 格式输出
TL_PERF_REGRESSION_FORMAT=json python examples/gemm/regression_example_gemm.py
```

## 参考

- [BitBLAS Roller](https://github.com/microsoft/BitBLAS) - 自动搜索空间推导
- `examples/` 目录中的各种示例实现
- `testing/python/kernel/` 中的核心算子测试
