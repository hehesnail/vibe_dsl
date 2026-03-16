# TileLang Linear Attention 示例详解

本文档详细分析 TileLang 项目中 Linear Attention 相关的示例代码，包括 Linear Attention 前向/反向传播、Retention 机制以及 Mamba2 的 Chunk State 和 Chunk Scan 操作。

## 目录

1. [示例概述](#示例概述)
2. [Linear Attention 前向传播](#linear-attention-前向传播)
3. [Linear Attention 反向传播](#linear-attention-反向传播)
4. [Retention 机制](#retention-机制)
5. [Mamba2 Chunk State](#mamba2-chunk-state)
6. [Mamba2 Chunk Scan](#mamba2-chunk-scan)
7. [性能优化技巧](#性能优化技巧)

---

## 示例概述

Linear Attention 是一类替代传统 Softmax Attention 的机制，通过使用核技巧将注意力计算的复杂度从 O(N^2) 降低到 O(N)。TileLang 提供了多个 Linear Attention 变体的实现：

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `example_linear_attn_fwd.py` | Linear Attention 前向传播 | Chunk-based 计算，累积 KV 状态 |
| `example_linear_attn_bwd.py` | Linear Attention 反向传播 | 分块反向计算，支持梯度累积 |
| `example_retention_fwd.py` | Retention 前向传播 | 指数衰减机制，因果掩码 |
| `example_mamba_chunk_state.py` | Mamba2 Chunk State | 状态压缩，分组头机制 |
| `example_mamba_chunk_scan.py` | Mamba2 Chunk Scan | 块间扫描，因果卷积 |

---

## Linear Attention 前向传播

### 算法原理

Linear Attention 的核心思想是将传统的 Softmax Attention:

```
O = softmax(Q @ K^T / sqrt(d)) @ V
```

转换为线性形式:

```
O = (Q @ K^T) @ V = Q @ (K^T @ V)
```

通过先计算 `K^T @ V` (KV 状态)，可以将计算复杂度从 O(N^2) 降低到 O(N)。

### 关键代码详解

**文件**: `/root/dev/vibe_dsl/tilelang/examples/linear_attention/example_linear_attn_fwd.py`

#### 1. 核函数定义与配置

```python
@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def tl_fused_chunk_fwd_kernel(
    B, S, H, DK, DV,
    dtype: T.dtype = T.float16,
    scale: float = None,
) -> torch.Tensor:
```

**关键配置说明** (`example_linear_attn_fwd.py:12-17`):
- `out_idx=[4]`: 指定第 5 个参数 (O) 为输出张量
- `TL_DISABLE_TMA_LOWER`: 禁用 TMA (Tensor Memory Accelerator) 降低优化
- `TL_DISABLE_WARP_SPECIALIZED`: 禁用 Warp 特化，使用标准执行模式

#### 2. 分块参数设置

```python
chunk_size = 64
BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
NK = tilelang.cdiv(DK, BK)
NV = tilelang.cdiv(DV, BV)
NT = tilelang.cdiv(S, chunk_size)
```

**分块策略** (`example_linear_attn_fwd.py:32-37`):
- `chunk_size=64`: 序列维度分块大小
- `BK=BV=64`: Key 和 Value 的特征维度分块
- `NK, NV, NT`: 各维度的分块数量

#### 3. Kernel 启动配置

```python
with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
    i_b = i_bh // H
    i_h = i_bh % H
```

**并行策略** (`example_linear_attn_fwd.py:47-49`):
- 3D Grid: `(NV, NK, B * H)`
- 每个线程块处理一个 (batch, head) 对的一个 Value 分块
- `i_v`: Value 维度索引
- `i_k`: Key 维度索引
- `i_bh`: batch 和 head 的合并索引

#### 4. 共享内存与寄存器分配

```python
q = T.alloc_shared([chunk_size, BK], dtype)
k = T.alloc_shared([chunk_size, BK], dtype)
v = T.alloc_shared([chunk_size, BV], dtype)
h = T.alloc_fragment([BK, BV], accum_dtype)
h_shared = T.alloc_shared([BK, BV], dtype)
s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
o = T.alloc_fragment([chunk_size, BV], accum_dtype)
```

**内存层次** (`example_linear_attn_fwd.py:51-58`):
- `alloc_shared`: 分配共享内存，用于线程块内数据共享
- `alloc_fragment`: 分配寄存器片段，用于累加计算
- `h`: 累积的 KV 状态 (BK x BV)
- `s`: 当前块的注意力分数

#### 5. Swizzle 优化

```python
T.use_swizzle(10)
```

**作用** (`example_linear_attn_fwd.py:61`):
- 启用线程块 Swizzling，优化 L2 缓存命中率
- `panel_size=10`: Swizzle 面板大小

#### 6. 核心计算循环

```python
for i in T.Pipelined(0, NT):
    # 加载 Q, K, V 到共享内存
    for row, col in T.Parallel(chunk_size, BK):
        q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
    T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
    T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)

    # 计算注意力分数
    T.gemm(q, k, s, clear_accum=True, transpose_B=True)
    for row, col in T.Parallel(chunk_size, chunk_size):
        s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

    # 计算输出
    T.gemm(s_shared, v, o, clear_accum=True)
    T.copy(h, h_shared)
    T.gemm(k, v, h, transpose_A=True)
    T.gemm(q, h_shared, o)

    # 原子累加输出
    T.copy(o, o_shared)
    T.atomic_add(O[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], o_shared)
```

**计算流程** (`example_linear_attn_fwd.py:65-80`):

1. **数据加载**: 使用 `T.Parallel` 并行加载 Q，使用 `T.copy` 加载 K, V
2. **Intra-chunk 注意力**: `q @ k^T`，应用因果掩码 (`row >= col`)
3. **Inter-chunk 注意力**: 使用累积的 KV 状态 `h` 计算跨块注意力
4. **状态更新**: `h += k^T @ v`，更新累积 KV 状态
5. **原子累加**: 使用 `T.atomic_add` 合并各分块结果

#### 7. 输出最终状态

```python
T.copy(h, final_state[i_b, i_h, i_k * BK : (i_k + 1) * BK, i_v * BV : (i_v + 1) * BV])
```

**用途** (`example_linear_attn_fwd.py:83`):
- 输出最终的累积 KV 状态，可用于增量推理

---

## Linear Attention 反向传播

### 算法原理

Linear Attention 的反向传播需要计算 Q, K, V 的梯度。由于 Linear Attention 的线性特性，梯度计算可以分解为：

1. **dQ 计算**: 基于当前块和累积状态
2. **dK, dV 计算**: 反向遍历序列，累积梯度

### 关键代码详解

**文件**: `/root/dev/vibe_dsl/tilelang/examples/linear_attention/example_linear_attn_bwd.py`

#### 1. 梯度缓冲区分配

```python
ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
ds_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
dq_shared = T.alloc_shared([chunk_size, BK], dtype)
dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
dk_shared = T.alloc_shared([chunk_size, BK], dtype)
dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
dv_shared = T.alloc_shared([chunk_size, BV], dtype)
```

**内存管理** (`example_linear_attn_bwd.py:52-59`):
- 为 ds, dq, dk, dv 分配寄存器和共享内存
- 使用共享内存作为中间缓冲区，减少寄存器压力

#### 2. dQ 计算 (正向遍历)

```python
# Calculate dQ
for i in T.Pipelined(0, NT):
    T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
    T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)
    T.copy(dO[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do)

    T.gemm(do, v, ds, transpose_B=True, clear_accum=True)
    for row, col in T.Parallel(chunk_size, chunk_size):
        ds_shared[row, col] = T.if_then_else(row >= col, ds[row, col], 0)

    T.gemm(ds_shared, k, dq, clear_accum=True)
    T.copy(h, h_shared)
    T.gemm(do, h_shared, dq)
    T.gemm(v, k, h, transpose_A=True)
    for row, col in T.Parallel(chunk_size, BK):
        dq[row, col] *= scale
    T.copy(dq, dq_shared)
    T.atomic_add(dQ[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], dq_shared)
```

**计算逻辑** (`example_linear_attn_bwd.py:75-91`):

1. **加载数据**: K, V, dO (输出梯度)
2. **计算 ds**: `dO @ V^T`，应用因果掩码
3. **Intra-chunk dQ**: `ds @ K`
4. **Inter-chunk dQ**: `dO @ h^T`，使用累积 KV 状态
5. **更新状态**: `h += V^T @ K`
6. **原子累加**: 将 dQ 累加到全局缓冲区

#### 3. dK, dV 计算 (反向遍历)

```python
# Calculate dK, dV (reversely)
for i in T.Pipelined(1, NT + 1):
    start = NT - i
    # ... 加载数据 ...

    # Calculate dk
    T.gemm(v, do, ds, transpose_B=True, clear_accum=True)
    for row, col in T.Parallel(chunk_size, chunk_size):
        ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
    T.gemm(ds_shared, q, dk, clear_accum=True)
    T.copy(dh, dh_shared)
    T.gemm(v, dh_shared, dk, transpose_B=True)

    # Calculate dv
    T.gemm(k, q, ds, transpose_B=True, clear_accum=True)
    for row, col in T.Parallel(chunk_size, chunk_size):
        ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
    T.gemm(ds_shared, do, dv, clear_accum=True)
    T.gemm(k, dh_shared, dv)

    # Update dh
    T.gemm(q, do, dh, transpose_A=True)

    # 原子累加 dk, dv
    T.atomic_add(dK[...], dk_shared)
    T.atomic_add(dV[...], dv_shared)
```

**反向计算** (`example_linear_attn_bwd.py:94-123`):

1. **反向遍历**: 从序列末尾向前遍历
2. **计算 dk**: 结合当前块注意力和累积梯度状态 dh
3. **计算 dv**: 类似逻辑，使用转置的注意力分数
4. **更新 dh**: `dh += q^T @ dO`，累积梯度状态

---

## Retention 机制

Retention 是一种带有指数衰减的线性注意力变体，由 RetNet 提出。

### 算法原理

Retention 在标准线性注意力的基础上增加了位置相关的指数衰减：

```
Retention(X) = (Q @ K^T * D) @ V
```

其中 D 是衰减矩阵: `D[i,j] = gamma^(i-j) if i >= j else 0`

### 关键代码详解

**文件**: `/root/dev/vibe_dsl/tilelang/examples/linear_attention/example_retention_fwd.py`

#### 1. 衰减系数计算

```python
log_decay = T.alloc_var(T.float32)
log_decay = T.log2(1 - T.exp2(-5.0 - 1.0 * i_h))  # Head-specific log decay
```

**衰减机制** (`example_retention_fwd.py:40-41`):
- 每个头有不同的衰减率
- 基于头索引 `i_h` 计算对数衰减系数

#### 2. 带衰减的注意力计算

```python
T.gemm(q, k, s, clear_accum=True, transpose_B=True)
for row, col in T.Parallel(chunk_size, chunk_size):
    s_shared[row, col] = T.if_then_else(row >= col, s[row, col] * T.exp2((row - col) * log_decay), 0)
```

**衰减应用** (`example_retention_fwd.py:61-63`):
- 计算 Q @ K^T
- 应用因果掩码和指数衰减: `exp2((row - col) * log_decay)`
- 位置越远，衰减越大

#### 3. 跨块衰减处理

```python
T.gemm(q, h_shared, o, clear_accum=True)
for row, col in T.Parallel(chunk_size, BV):
    o[row, col] = T.exp2((row + 1) * log_decay) * o[row, col]
```

**状态衰减** (`example_retention_fwd.py:66-68`):
- 累积状态 h 在跨块传播时需要衰减
- 每行的衰减因子为 `exp2((row + 1) * log_decay)`

#### 4. V 的衰减更新

```python
for row, col in T.Parallel(chunk_size, BV):
    v[row, col] = v[row, col] * T.exp2((chunk_size - row - 1) * log_decay)
for row, col in T.Parallel(BK, BV):
    h[row, col] = T.exp2(chunk_size * log_decay) * h[row, col]
```

**V 和 H 的衰减** (`example_retention_fwd.py:72-74`):
- V 需要反向衰减以匹配位置编码
- H 整体衰减以传递到下一个块

---

## Mamba2 Chunk State

Mamba2 是状态空间模型 (SSM) 的优化版本，使用分块计算来提高效率。

### 算法原理

Chunk State 计算将序列分成多个块，在每个块内计算状态累积：

```
state = sum(B * decay * dt * x)
```

其中:
- B: 输入相关的状态矩阵
- decay: 基于累积和的状态衰减
- dt: 时间步长
- x: 输入

### 关键代码详解

**文件**: `/root/dev/vibe_dsl/tilelang/examples/linear_attention/example_mamba_chunk_state.py`

#### 1. 自动调优配置

```python
def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[32, 64, 128],
        block_K=[32, 64],
        num_stages=[1, 2, 3, 4, 5]
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[4])
def chunk_state_fwd(...):
```

**自动调优** (`example_mamba_chunk_state.py:48-54`):
- 定义多种分块配置
- 使用 `@autotune` 自动搜索最优配置

#### 2. Kernel 启动与索引计算

```python
with T.Kernel(
    nheads,
    T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N),
    batch * nchunks,
    threads=threads
) as (bz, bx, by):
    batch_idx = by % batch
    chunk_idx = by // batch
    m_idx = bx // T.ceildiv(dstate, block_N)
    n_idx = bx % T.ceildiv(dstate, block_N)
```

**3D Grid 映射** (`example_mamba_chunk_state.py:71-90`):
- X 维度: heads
- Y 维度: headdim x dstate 的 2D 分块
- Z 维度: batch x nchunks

#### 3. 衰减计算

```python
dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
T.clear(acc_o)
for k in T.Pipelined(loop_range, num_stages=num_stages):
    T.copy(dA_cumsum[...], dA_cumsum_shared)
    T.copy(dt[...], dt_shared)
    T.copy(dA_cumsum_shared, dA_cumsum_local)
    T.copy(dt_shared, dt_local)
    for i in T.Parallel(block_K):
        scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
```

**衰减逻辑** (`example_mamba_chunk_state.py:94-111`):
- `dA_cumsum`: 累积的状态转移矩阵
- `scale = exp2(dA_last - dA_current) * dt`: 计算每个位置的衰减因子
- `p = 1.44269504`: log2(e)，用于指数转换

#### 4. 状态计算

```python
T.copy(x_shared, x_local)
for i, j in T.Parallel(block_M, block_K):
    xt_local[i, j] = x_local[j, i] * scale[j]
T.copy(B[...], B_shared)
T.gemm(xt_local, B_shared, acc_o)
```

**核心计算** (`example_mamba_chunk_state.py:112-124`):
- 转置并缩放输入 x
- 与 B 矩阵相乘，累积状态

---

## Mamba2 Chunk Scan

Chunk Scan 是 Mamba2 的另一个核心操作，负责在块之间扫描和传播状态。

### 算法原理

Chunk Scan 结合了两个部分的贡献：
1. **Intra-chunk**: 块内的局部卷积计算
2. **Inter-chunk**: 前一个块的状态传递

### 关键代码详解

**文件**: `/root/dev/vibe_dsl/tilelang/examples/linear_attention/example_mamba_chunk_scan.py`

#### 1. 复杂分块配置

```python
def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[32, 64],
        block_K=[64, 128, 256],
        block_Dstate=[128],
        num_stages=[1, 2, 3, 4, 5]
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]
```

**配置** (`example_mamba_chunk_scan.py:65-67`):
- 支持更大的分块大小 (最大 256)
- `block_Dstate`: 状态维度分块

#### 2. 动态共享内存

```python
x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
```

**动态内存** (`example_mamba_chunk_scan.py:122`):
- 使用 `scope="shared.dyn"` 分配动态共享内存
- 在运行时分配合适的大小

#### 3. 前序状态贡献

```python
for i in T.Parallel(block_M):
    scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
T.copy(C[...], C_shared)
T.copy(prev_states[...], prev_state_shared)
T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
for i, j in T.Parallel(block_M, block_N):
    acc_o[i, j] *= scale_m_local[i]
```

**状态传播** (`example_mamba_chunk_scan.py:147-165`):
- 加载前一个块的状态 `prev_states`
- 与 C 矩阵相乘，并应用衰减

#### 4. Intra-chunk 卷积

```python
for k in T.Pipelined(loop_range, num_stages=num_stages):
    T.copy(cb[...], cb_shared)
    T.copy(cb_shared, cb_local)
    T.copy(dA_cumsum[...], dA_cs_k_shared)
    T.copy(dA_cumsum_shared, dA_cs_k_local)
    for i, j in T.Parallel(block_M, block_K):
        cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
    T.copy(dt[...], dt_shared)
    T.copy(dt_shared, dt_local)
    for i, j in T.Parallel(block_M, block_K):
        cb_local[i, j] *= dt_local[j]
    for i, j in T.Parallel(block_M, block_K):
        cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0)
    T.copy(x[...], x_shared)
    T.gemm(cb_local, x_shared, acc_o)
```

**局部卷积** (`example_mamba_chunk_scan.py:169-200`):
- 加载局部卷积矩阵 cb
- 应用衰减: `exp2(dA_m - dA_k) * dt`
- 应用因果掩码
- 与输入 x 相乘累加

#### 4. 残差连接

```python
D_local[0] = D[bz]
T.copy(x[...], x_residual_shared)
T.copy(x_residual_shared, x_residual_local)
for i, j in T.Parallel(block_M, block_N):
    acc_o[i, j] += x_residual_local[i, j] * D_local[0]
```

**Skip Connection** (`example_mamba_chunk_scan.py:202-214`):
- 加载参数 D (类似 Transformer 中的门控)
- 添加输入的残差连接

---

## 性能优化技巧

### 1. 分块策略

```python
chunk_size = 64  # 序列分块
BK = BV = 64     # 特征分块
```

**建议**:
- 序列分块通常为 64 或 128
- 特征分块需平衡寄存器使用和并行度
- 确保维度可被分块大小整除

### 2. Pipeline 优化

```python
for k in T.Pipelined(loop_range, num_stages=num_stages):
    # 循环体
```

**作用** (`example_linear_attn_fwd.py:65`):
- 重叠内存访问和计算
- `num_stages`: 流水线阶段数，通常为 2-4

### 3. Swizzle 优化

```python
T.use_swizzle(10)
```

**效果** (`example_linear_attn_fwd.py:61`):
- 提高 L2 缓存命中率
- 减少内存访问冲突

### 4. 布局注解

```python
T.annotate_layout({
    x_shared: tilelang.layout.make_swizzled_layout(x_shared),
})
```

**用途** (`example_mamba_chunk_state.py:92`):
- 避免共享内存 Bank 冲突
- 优化数据布局以提高访问效率

### 5. 原子操作

```python
T.atomic_add(O[...], o_shared)
```

**场景** (`example_linear_attn_fwd.py:80`):
- 多线程块写入同一位置时使用
- 确保数据一致性

### 6. Warp 策略

```python
T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
```

**策略选择** (`example_mla_decode.py:67`):
- `FullCol`: 每 Warp 计算一列
- `FullRow`: 每 Warp 计算一行
- 根据矩阵形状选择合适的策略

### 7. 寄存器控制

```python
T.set_max_nreg(240, 1)  # 消费者 Warp
T.set_max_nreg(80, 0)   # 生产者 Warp
```

**Warp 特化** (`example_mla_decode_ws.py:93`):
- 控制每个 Warp 的寄存器使用
- 消费者使用更多寄存器进行计算
- 生产者使用较少寄存器进行数据传输

### 8. 快速数学

```python
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
```

**效果** (`example_mamba_chunk_scan.py:71-75`):
- 启用快速但精度稍低的数学运算
- 适用于对精度要求不高的场景

---

## 总结

TileLang 的 Linear Attention 示例展示了如何高效实现现代注意力机制：

1. **分块计算**: 将长序列分成小块，降低内存压力
2. **状态累积**: 利用线性注意力的特性，累积 KV 状态
3. **反向传播**: 支持高效的梯度计算
4. **扩展机制**: Retention 和 Mamba2 展示了如何添加衰减和状态空间
5. **性能优化**: Swizzle、Pipeline、Warp 特化等技巧的综合应用

这些技术不仅适用于 Linear Attention，也可应用于其他需要高效序列建模的场景。
