# TileLang FlashAttention 示例详解

本文档详细分析 TileLang 项目中的 FlashAttention 示例代码，涵盖前向传播、反向传播、变长序列以及分组查询注意力（GQA）等实现。

## 目录

1. [FlashAttention 基础概念](#flashattention-基础概念)
2. [MHA 前向传播](#mha-前向传播)
3. [MHA 反向传播](#mha-反向传播)
4. [变长序列支持](#变长序列支持)
5. [分组查询注意力 GQA](#分组查询注意力-gqa)
6. [性能优化技巧总结](#性能优化技巧总结)

---

## FlashAttention 基础概念

### 算法概述

FlashAttention 是一种 IO-aware 的精确注意力算法，通过分块计算（tiling）和重计算（recomputation）来减少 HBM（高带宽内存）访问，从而加速注意力计算。

### 核心公式

标准注意力计算：
```
S = Q * K^T / sqrt(d)
P = softmax(S)
O = P * V
```

FlashAttention 使用在线 softmax 技术，分块计算并维护运行时的最大值和累加和。

### TileLang 优势

TileLang 通过以下抽象简化 FlashAttention 实现：
- `T.alloc_shared`: 在共享内存中分配 Q、K、V 块
- `T.alloc_fragment`: 在寄存器中分配中间结果
- `T.Pipelined`: 自动流水线重叠数据加载与计算
- `T.gemm`: 高效的矩阵乘法原语

---

## MHA 前向传播

### 文件位置
- `/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_mha_fwd_bshd.py` (BSHD 布局)
- `/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py` (BHSD 布局)

### 示例概述

这两个文件分别实现了两种数据布局的 Multi-Head Attention 前向传播：
- **BSHD**: Batch-Sequence-Head-Dimension
- **BHSD**: Batch-Head-Sequence-Dimension

### 关键代码详解

#### 1. 内核定义与缩放因子

```python
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch, heads, seq_len, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
```

**关键点：**
- `out_idx=[3]`: 输出是第 4 个参数（Output）
- `TL_ENABLE_FAST_MATH: True`: 启用快速数学优化
- `scale` 包含 `log2(e)` 因子，用于 `exp2` 计算而非标准 `exp`

#### 2. 内存分配

```python
with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
    Q_shared = T.alloc_shared([block_M, dim], dtype)
    K_shared = T.alloc_shared([block_N, dim], dtype)
    V_shared = T.alloc_shared([block_N, dim], dtype)
    O_shared = T.alloc_shared([block_M, dim], dtype)
    acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
    acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
    acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
    scores_max = T.alloc_fragment([block_M], accum_dtype)
    scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
    scores_scale = T.alloc_fragment([block_M], accum_dtype)
    scores_sum = T.alloc_fragment([block_M], accum_dtype)
    logsum = T.alloc_fragment([block_M], accum_dtype)
```

**关键点：**
- `Q_shared`: 存储 Q 的一个块（seq 维度分块）
- `K_shared`, `V_shared`: 存储 K、V 的块（在循环中迭代）
- `acc_s`: 注意力分数矩阵 S = Q*K^T
- `scores_max`, `logsum`: 在线 softmax 所需的统计量

#### 3. 因果掩码处理

```python
loop_range = (
    T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
    if is_causal
    else T.ceildiv(seq_len, block_N)
)

for k in T.Pipelined(loop_range, num_stages=num_stages):
    T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
    if is_causal:
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.if_then_else(
                bx * block_M + i >= k * block_N + j,
                0,
                -T.infinity(acc_s.dtype)
            )
    else:
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.if_then_else(
                k * block_N + j >= seq_len,
                -T.infinity(acc_s.dtype),
                0
            )
```

**关键点：**
- 因果掩码只计算上三角部分（包括对角线）
- `loop_range` 在因果模式下减少迭代次数
- `T.if_then_else` 用于条件赋值

#### 4. 在线 Softmax 计算

```python
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

# 计算缩放因子
for i in T.Parallel(block_M):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

# 缩放之前的输出
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] *= scores_scale[i]

# 计算 softmax 分数
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

T.reduce_sum(acc_s, scores_sum, dim=1)
for i in T.Parallel(block_M):
    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
```

**关键点：**
- `T.reduce_max`: 计算每行的最大值
- `T.reduce_sum`: 计算每行的和
- 使用 `exp2` 而非 `exp` 以获得更好性能
- `scores_scale` 用于调整之前计算的输出块

#### 5. 输出计算

```python
T.copy(acc_s, acc_s_cast)
T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

# 循环结束后归一化
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]
T.copy(acc_o, O_shared)
T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
```

---

## MHA 反向传播

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_mha_bwd_bshd.py`

### 示例概述

反向传播实现包含三个核心内核：
1. **flashattn_fwd**: 前向传播（保存 lse 用于反向）
2. **flashattn_bwd_preprocess**: 预处理，计算 Delta
3. **flashattn_bwd**: 核心反向传播计算
4. **flashattn_bwd_postprocess**: 后处理，重新排列 dQ

### 关键代码详解

#### 1. 前向传播（保存中间结果）

```python
@tilelang.jit(out_idx=[3, 4])
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    # ... 同前向传播，但额外输出 lse
    for i in T.Parallel(block_M):
        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
    T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
```

**关键点：**
- `out_idx=[3, 4]`: 输出 Output 和 lse（log-sum-exp）
- lse 用于反向传播中的梯度计算

#### 2. 预处理内核

```python
@tilelang.jit(out_idx=[2])
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])
```

**关键点：**
- Delta = sum(O * dO, dim=-1)，用于反向传播中的梯度计算
- 分块计算以节省寄存器

#### 3. 反向传播核心

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (bx, by, bz):
            # 分配共享内存和寄存器...

            # 加载 K, V 块
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                # 计算 qkT = K * Q^T
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # 应用 softmax
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                # 计算 dS
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # 计算 dV
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                # 计算 dK
                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                # 计算 dQ（使用 atomic_add）
                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
```

**关键点：**
- 反向传播需要计算 dQ、dK、dV 三个梯度
- dQ 使用 `T.atomic_add` 因为多个块可能写入同一位置
- 因果掩码在反向时条件相反（`by * block_M + i <= k * block_N + j`）

#### 4. dQ 布局优化

```python
def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])

@tilelang.jit(out_idx=[1])
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(dQ[bz, bx * blk : (bx + 1) * blk, by, :], dQ_out[bz, bx * blk : (bx + 1) * blk, by, :])
```

**关键点：**
- dQ 使用特殊布局以匹配 GEMM fragment 的 8x8 结构
- `T.annotate_layout` 注解自定义内存布局
- 后处理将 dQ 从优化布局转回标准布局

---

## 变长序列支持

### 文件位置
`/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_mha_fwd_varlen.py`

### 示例概述

变长序列（Variable Length）支持处理批次中不同样本具有不同序列长度的情况，通过 `cu_seqlens`（累积序列长度）数组来索引数据。

### 关键代码详解

#### 1. 数据布局

```python
q_shape = [UQ, heads, dim]  # UQ = 所有样本的 query 总数（去填充后）
k_shape = [UKV, heads, dim]  # UKV = 所有样本的 key/value 总数
```

#### 2. 累积序列长度索引

```python
batch_idx = bz
head_idx = by

q_start_idx = cu_seqlens_q[batch_idx]
kv_start_idx = cu_seqlens_k[batch_idx]
q_end_idx = cu_seqlens_q[batch_idx + 1]
kv_end_idx = cu_seqlens_k[batch_idx + 1]

q_current_seqlen = q_end_idx - q_start_idx
kv_current_seqlen = kv_end_idx - kv_start_idx
```

#### 3. 数据加载与边界处理

```python
T.copy(
    Q_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :],
    Q_shared
)
```

#### 4. 因果掩码（考虑长度差异）

```python
offset = kv_current_seqlen - q_current_seqlen  # 始终右对齐

loop_range = (
    T.min(T.ceildiv(offset + (bx + 1) * block_M, block_N), T.ceildiv(kv_current_seqlen, block_N))
    if is_causal
    else T.ceildiv(kv_current_seqlen, block_N)
)

if is_causal:
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.if_then_else(
            (bx * block_M + i + offset < k * block_N + j)
            or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
            -1e9,
            0,
        )
```

**关键点：**
- `offset` 处理 query 和 key/value 长度不一致的情况
- 边界检查处理超出实际序列长度的位置

---

## 分组查询注意力 GQA

### 文件位置
- `/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_gqa_fwd_bshd.py`
- `/root/dev/vibe_dsl/tilelang/examples/flash_attention/example_gqa_bwd.py`

### 示例概述

分组查询注意力（Grouped Query Attention, GQA）是一种内存优化技术，多个 query head 共享相同的 key/value head。

### 关键代码详解

#### 1. 形状定义

```python
def flashattn(batch, heads, seq_len, dim, is_causal, groups=1, block_M=64, block_N=64):
    head_kv = heads // groups  # key/value head 数量
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
```

#### 2. KV 加载（使用组索引）

```python
for k in T.Pipelined(loop_range, num_stages=num_stages):
    # by // groups: 根据 query head 索引计算对应的 kv head 索引
    T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
    # ...
    T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
```

#### 3. GQA 反向传播的特殊处理

```python
# 反向传播中，dK 和 dV 需要累加来自多个 query head 的梯度

# 方法 1: 使用 atomic_add
T.atomic_add(dV[bz, by * block_M : (by + 1) * block_M, bx // groups, :], dv_shared)
T.atomic_add(dK[bz, by * block_M : (by + 1) * block_M, bx // groups, :], dk_shared)

# 方法 2: 使用 split（避免原子操作）
dk_shape = [groups, batch, seq_len, head_kv, dim]  # 额外维度存储每个组的梯度
dv_shape = [groups, batch, seq_len, head_kv, dim]
# 内核结束后在 CPU/GPU 上求和
```

---

## 性能优化技巧总结

### 1. 内存优化

| 技术 | 描述 | 代码示例 |
|------|------|----------|
| 共享内存缓存 | 缓存 Q、K、V 块 | `T.alloc_shared([block_M, dim], dtype)` |
| 寄存器累加 | 中间结果存寄存器 | `T.alloc_fragment([block_M, block_N], accum_dtype)` |
| 流水线加载 | 重叠数据拷贝与计算 | `T.Pipelined(loop_range, num_stages=2)` |

### 2. 计算优化

| 技术 | 描述 | 代码示例 |
|------|------|----------|
| 在线 Softmax | 避免存储完整注意力矩阵 | `T.reduce_max`, `T.reduce_sum` |
| exp2 替代 exp | 使用更快的指令 | `T.exp2(x * scale)` |
| GEMM Warp 策略 | 控制 warp 级并行 | `policy=T.GemmWarpPolicy.FullRow` |

### 3. 因果掩码优化

| 技术 | 描述 | 代码示例 |
|------|------|----------|
| 循环范围裁剪 | 减少因果模式下的迭代次数 | `loop_range = T.ceildiv((bx + 1) * block_M, block_N)` |
| 条件初始化 | 条件赋值而非掩码乘法 | `T.if_then_else(condition, 0, -inf)` |

### 4. 反向传播优化

| 技术 | 描述 | 代码示例 |
|------|------|----------|
| 原子操作 | 处理 dQ 的并行写入 | `T.atomic_add(dQ[...], dq[i, j])` |
| 自定义布局 | 优化 atomic add 性能 | `T.annotate_layout({dQ: make_dq_layout(dQ)})` |
| Split 策略 | 避免原子操作，事后求和 | `dk_shape = [groups, batch, seq_len, head_kv, dim]` |

### 5. 配置调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| block_M | 64, 128 | Query 序列分块大小 |
| block_N | 64, 128 | Key/Value 序列分块大小 |
| num_stages | 1-3 | 流水线深度 |
| threads | 128, 256 | 每块线程数 |

### 6. 代码文件索引

| 文件 | 主要特性 | 关键行号 |
|------|----------|----------|
| `example_mha_fwd_bshd.py` | BSHD 布局前向 | 23, 36, 59, 67, 89 |
| `example_mha_fwd_bhsd.py` | BHSD 布局前向 | 23, 40, 65, 75, 97 |
| `example_mha_fwd_varlen.py` | 变长序列 | 27, 47, 64, 82, 106 |
| `example_mha_bwd_bshd.py` | 反向传播 | 15, 89, 123, 150, 174 |
| `example_gqa_fwd_bshd.py` | GQA 前向 | 68, 83, 107, 133 |
| `example_gqa_bwd.py` | GQA 反向 | 14, 84, 156, 178, 248 |

---

## 参考

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
