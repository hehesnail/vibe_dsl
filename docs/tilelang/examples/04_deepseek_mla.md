# TileLang DeepSeek MLA 示例详解

本文档详细分析 TileLang 项目中 DeepSeek MLA (Multi-Head Latent Attention) 的示例代码，包括基础解码、分页解码、Warp 特化优化以及 AMD 平台适配版本。

## 目录

1. [示例概述](#示例概述)
2. [MLA 算法原理](#mla-算法原理)
3. [基础 MLA 解码](#基础-mla-解码)
4. [分页 MLA 解码](#分页-mla-解码)
5. [Warp 特化优化版本](#warp-特化优化版本)
6. [AMD 平台优化](#amd-平台优化)
7. [性能优化技巧](#性能优化技巧)
8. [基准测试](#基准测试)

---

## 示例概述

DeepSeek 的 MLA (Multi-Head Latent Attention) 是一种新型注意力机制，以其硬件效率和推理速度提升而闻名。TileLang 提供了多个 MLA 实现版本：

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `example_mla_decode.py` | 基础 MLA 解码 | 标准实现，Split-KV 支持 |
| `example_mla_decode_paged.py` | 分页 MLA 解码 | 支持 PagedAttention 的 KV Cache |
| `example_mla_decode_ws.py` | Warp 特化版本 | Hopper 架构优化，显式屏障同步 |
| `example_mla_decode_persistent.py` | 持久化内核 | 持久化线程块调度 |
| `amd/benchmark_mla_decode_amd_tilelang.py` | AMD 优化版本 | ROCm 平台适配 |

---

## MLA 算法原理

### 核心概念

MLA 通过低秩压缩减少 KV Cache 的内存占用：

```
# 传统 MHA
Q: [batch, heads, dim]
K: [batch, seqlen, heads, dim]
V: [batch, seqlen, heads, dim]

# MLA (压缩表示)
Q: [batch, heads, dim] = [Q_nope, Q_pe]  # 512 + 64 = 576
KV: [batch, seqlen, 1, dim]  # 共享的 Key-Value
K_pe: [batch, seqlen, 1, pe_dim]  # 位置编码部分
```

### 注意力计算

```
scores = Q_nope @ KV^T + Q_pe @ K_pe^T
attention = softmax(scores / sqrt(dim + pe_dim))
output = attention @ KV
```

### 优化挑战

MLA 的主要挑战在于大的头维度：
- Query/Key 维度: 576 (512 + 64)
- Value 维度: 512

这导致累积缓冲区 `acc_o` 过大，容易造成寄存器溢出。

---

## 基础 MLA 解码

**文件**: `/root/dev/vibe_dsl/tilelang/examples/deepseek_mla/example_mla_decode.py`

### 1. 核函数定义

```python
@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(
    batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
    block_N, block_H, num_split, softmax_scale
):
```

**配置说明** (`example_mla_decode.py:10-15`):
- `out_idx=[6]`: 指定 Output 为输出张量
- `TL_ENABLE_FAST_MATH`: 启用快速数学运算优化

### 2. 张量布局

```python
Q: T.Tensor([batch, heads, dim], dtype)          # Query (无位置编码部分)
Q_pe: T.Tensor([batch, heads, pe_dim], dtype)    # Query (位置编码部分)
KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype)   # 共享 KV
K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype)  # Key 位置编码
glse: T.Tensor([batch, heads, num_split], dtype)              # Log-Sum-Exp
Output_partial: T.Tensor([batch, heads, num_split, dim], dtype)  # 部分输出
Output: T.Tensor([batch, heads, dim], dtype)     # 最终输出
```

**输入输出** (`example_mla_decode.py:25-32`):
- Q 被分为 Q_nope (512d) 和 Q_pe (64d) 两部分
- KV 在 heads 维度共享 (kv_head_num=1)
- 支持 Split-KV 的部分输出和合并

### 3. Split-KV 计算内核

```python
with T.Kernel(
    batch,
    heads // min(block_H, kv_group_num),
    num_split,
    threads=256
) as (bid, hid, bz):
```

**Grid 配置** (`example_mla_decode.py:35`):
- X: batch 维度
- Y: heads 分块维度
- Z: num_split (KV 分割数)
- 256 线程 = 2 warpgroups (Hopper)

### 4. 共享内存分配

```python
Q_shared = T.alloc_shared([block_H, dim], dtype)
S_shared = T.alloc_shared([block_H, block_N], dtype)
Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
KV_shared = T.alloc_shared([block_N, dim], dtype)
K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
O_shared = T.alloc_shared([block_H, dim], dtype)
```

**内存布局** (`example_mla_decode.py:36-41`):
- Q/KV 使用大共享内存缓冲区
- S_shared 存储注意力分数
- 总共享内存: (64*512 + 64*64 + 64*512 + 64*64) * 2 bytes ≈ 128KB

### 5. 寄存器缓冲区

```python
acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
scores_max = T.alloc_fragment([block_H], accum_dtype)
scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
scores_scale = T.alloc_fragment([block_H], accum_dtype)
scores_sum = T.alloc_fragment([block_H], accum_dtype)
logsum = T.alloc_fragment([block_H], accum_dtype)
```

**寄存器使用** (`example_mla_decode.py:42-49`):
- `acc_o`: [64, 512] float32 = 128KB 寄存器 (关键瓶颈)
- 使用 FP32 累加保证数值稳定性

### 6. 核心计算循环

```python
loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
for k in T.Pipelined(loop_range, num_stages=2):
    kv_start = (seqlen_kv // num_split) * bz + k * block_N
    kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
    T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
    T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)

    T.clear(acc_s)
    T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
    T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
```

**计算流程** (`example_mla_decode.py:61-68`):
1. **加载 KV**: 从全局内存加载当前 KV 块
2. **Q @ K**: 两部分矩阵乘法 (Q_nope @ KV + Q_pe @ K_pe)
3. `FullCol` 策略: 每 warpgroup 计算一列

### 7. Softmax 在线计算

```python
T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(block_H):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
for i in T.Parallel(block_H):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
for i, j in T.Parallel(block_H, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
T.reduce_sum(acc_s, scores_sum, dim=1)
```

**Softmax** (`example_mla_decode.py:69-78`):
- 在线计算避免存储完整注意力矩阵
- 使用 log-sum-exp 技巧保证数值稳定性
- `scale = softmax_scale * log2(e)` 用于 exp2 优化

### 8. 输出更新

```python
T.copy(acc_s, S_shared)
T.copy(S_shared, acc_s_cast)
for i in T.Parallel(block_H):
    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
for i, j in T.Parallel(block_H, dim):
    acc_o[i, j] *= scores_scale[i]
T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
```

**累加逻辑** (`example_mla_decode.py:79-85`):
- 缩放并累加输出
- 更新 logsum 用于后续归一化

### 9. 最终归一化与输出

```python
for i, j in T.Parallel(block_H, dim):
    acc_o[i, j] /= logsum[i]
for i in T.Parallel(block_H):
    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
T.copy(logsum, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz])
T.copy(acc_o, O_shared)
T.copy(O_shared, Output_partial[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, :])
```

**输出处理** (`example_mla_decode.py:86-92`):
- 使用 logsum 归一化输出
- 存储 LSE (Log-Sum-Exp) 用于合并

### 10. Split-KV 合并

```python
with T.Kernel(heads, batch, threads=128) as (hid, bz):
    po_local = T.alloc_fragment([dim], dtype)
    o_accum_local = T.alloc_fragment([dim], accum_dtype)
    lse_local_split = T.alloc_var(accum_dtype)
    lse_logsum_local = T.alloc_var(accum_dtype)
    lse_max_local = T.alloc_var(accum_dtype)
    scale_local = T.alloc_var(accum_dtype)

    T.clear(lse_logsum_local)
    T.clear(o_accum_local)
    lse_max_local = -T.infinity(accum_dtype)

    # 找到最大 LSE
    for k in T.serial(num_split):
        lse_max_local = T.max(lse_max_local, glse[bz, hid, k])

    # 计算归一化因子
    for k in T.Pipelined(num_split, num_stages=1):
        lse_local_split = glse[bz, hid, k]
        lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
    lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local

    # 合并输出
    for k in T.serial(num_split):
        for i in T.Parallel(dim):
            po_local[i] = Output_partial[bz, hid, k, i]
        lse_local_split = glse[bz, hid, k]
        scale_local = T.exp2(lse_local_split - lse_logsum_local)
        for i in T.Parallel(dim):
            o_accum_local[i] += po_local[i] * scale_local

    for i in T.Parallel(dim):
        Output[bz, hid, i] = o_accum_local[i]
```

**合并逻辑** (`example_mla_decode.py:95-120`):
- 基于 LSE 的加权合并
- 使用 log-sum-exp 技巧避免数值溢出
- 128 线程足够处理合并任务

---

## 分页 MLA 解码

**文件**: `/root/dev/vibe_dsl/tilelang/examples/deepseek_mla/example_mla_decode_paged.py`

### 1. 分页 KV Cache 支持

```python
@T.prim_func
def main_split(
    Q: T.Tensor([batch, h_q, dv], dtype),
    Q_pe: T.Tensor([batch, h_q, dpe], dtype),
    KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
    K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
    block_table: T.Tensor([batch, max_seqlen_pad // block_size], T.int32),
    cache_seqlens: T.Tensor([batch], T.int32),
    glse: T.Tensor([batch, h_q, num_split], dtype),
    Output_partial: T.Tensor([batch, h_q, num_split, dv], dtype),
    Output: T.Tensor([batch, h_q, dv], dtype),
):
```

**分页张量** (`example_mla_decode_paged.py:28-38`):
- `block_table`: 块表，映射逻辑位置到物理块
- `cache_seqlens`: 每个序列的实际长度
- KV 存储为扁平化缓冲区

### 2. 块表查找

```python
total_blocks = T.ceildiv(cache_seqlens[bx], block_N)
blocks_per_split = T.floordiv(total_blocks, num_split)
remaining_blocks = T.floormod(total_blocks, num_split)
loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N

for k in T.Pipelined(loop_range, num_stages=2):
    kv_start = block_table[bx, (start + k * block_N) // block_size] * block_size + (k * block_N) % block_size
    T.copy(KV[kv_start : kv_start + block_N, cur_kv_head, :], KV_shared)
```

**地址计算** (`example_mla_decode_paged.py:65-73`):
- 通过 `block_table` 查找物理地址
- 支持不均匀分割处理变长序列

### 3. 掩码处理

```python
for i, j in T.Parallel(block_H, block_N):
    acc_s[i, j] = T.if_then_else(
        start + k * block_N + j >= cache_seqlens[bx],
        -T.infinity(accum_dtype),
        acc_s[i, j]
    )
```

**变长掩码** (`example_mla_decode_paged.py:80-81`):
- 根据 `cache_seqlens` 应用填充掩码
- 超出实际长度的位置设为 -inf

### 4. 反向遍历优化

```python
for kr in T.Pipelined(loop_range, num_stages=2):
    k = loop_range - 1 - kr  # 反向遍历
    kv_start = block_table[bx, (k * block_N) // block_size] * block_size + (k * block_N) % block_size
```

**反向遍历** (`example_mla_decode_paged.py:170-172`):
- 在某些场景下反向遍历可以提高缓存效率
- 最后一轮迭代应用掩码

---

## Warp 特化优化版本

**文件**: `/root/dev/vibe_dsl/tilelang/examples/deepseek_mla/example_mla_decode_ws.py`

### 1. 编译标志

```python
@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)
```

**编译优化** (`example_mla_decode_ws.py:10-26`):
- O3 优化级别
- 启用半精度运算
- PTXAS 详细输出和寄存器使用级别 10

### 2. 分块共享内存

```python
Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
```

**双缓冲分块** (`example_mla_decode_ws.py:48-56`):
- 将 512 维分成左右各 256 维
- 双缓冲设计 (0 和 1) 支持流水线

### 3. Warp 特化屏障

```python
bar_q = T.alloc_barrier(arrive_count=384)
bar_k_0_ready = T.alloc_barrier(arrive_count=128)
bar_k_1_ready = T.alloc_barrier(arrive_count=128)
bar_k_0_free = T.alloc_barrier(arrive_count=256)
bar_k_1_free = T.alloc_barrier(arrive_count=256)
bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)
```

**屏障分配** (`example_mla_decode_ws.py:73-79`):
- 384 线程 = 3 warpgroups (1 生产者 + 2 消费者)
- 显式屏障用于生产者-消费者同步

### 4. 线程角色分配

```python
tx = T.get_thread_binding()

if tx < 128:
    # 消费者 0: 计算左半部分和 Softmax
    T.set_max_nreg(240, 1)
    # ...
elif tx >= 128 and tx < 256:
    # 消费者 1: 计算右半部分
    T.set_max_nreg(168, 1)
    # ...
elif tx >= 256:
    # 生产者: 加载 KV
    T.set_max_nreg(80, 0)
    # ...
```

**角色划分** (`example_mla_decode_ws.py:84-212`):
- 消费者 0 (0-127): 计算左半部分，控制 Softmax
- 消费者 1 (128-255): 计算右半部分
- 生产者 (256-383): 使用 cp.async 加载数据

### 5. 显式异步拷贝

```python
for r in T.serial(4):
    kv_indices = (seqlen_kv // num_split) * bz + (i_i * 2) * block_N + r * 16 + (tx - 256) // 8
    for u in T.serial(4):
        T.ptx_cp_async(
            T.access_ptr(KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8], "w", 8),
            T.access_ptr(KV[bid, kv_indices, cur_kv_head, 64 * u + (tx - 256) % 8 * 8], "r", 8),
            16,
        )
T.cp_async_barrier_noinc(bar_k_0_ready[0])
```

**异步加载** (`example_mla_decode_ws.py:216-234`):
- 使用 `cp.async` 指令异步加载数据
- 16 字节对齐访问
- 显式屏障同步

### 6. WGMMA 异步矩阵乘法

```python
T.clear(acc_s)
T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)
T.wait_wgmma(0)
```

**WGMMA** (`example_mla_decode_ws.py:103-108`):
- `wg_wait=-1`: 异步启动所有 GEMM
- `wait_wgmma(0)`: 等待所有 WGMMA 完成
- 最大化指令级并行

---

## AMD 平台优化

**文件**: `/root/dev/vibe_dsl/tilelang/examples/deepseek_mla/amd/benchmark_mla_decode_amd_tilelang.py`

### 1. 自动调优配置

```python
def get_configs():
    BLOCK_N = [16, 32, 64, 128]
    BLOCK_H = [16, 32, 64, 128]
    num_split = [1, 2, 4, 8, 16, 32]
    threads = [128, 256]
    _configs = list(itertools.product(BLOCK_N, BLOCK_H, num_split, threads))
    return [
        {
            "block_N": c[0],
            "block_H": c[1],
            "num_split": c[2],
            "threads": c[3],
        }
        for c in _configs
    ]

@tilelang.autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashmla_decode(...):
```

**自动调优** (`benchmark_mla_decode_amd_tilelang.py:9-36`):
- 广泛的配置搜索空间
- 自动选择最优分块参数

### 2. ROCm 适配

```python
T.gemm(Q_local, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
T.gemm(Q_pe_local, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

**策略调整** (`benchmark_mla_decode_amd_tilelang.py:85-86`):
- 使用 `FullRow` 策略而非 `FullCol`
- 更好地适配 AMD GPU 的 Wave 架构

### 3. Pipeline 配置

```python
for k in T.Pipelined(loop_range, num_stages=0):
```

**零阶段流水线** (`benchmark_mla_decode_amd_tilelang.py:79`):
- AMD 架构使用不同的流水线策略
- `num_stages=0` 禁用软件流水线

---

## 性能优化技巧

### 1. 分块大小选择

```python
BLOCK_N = 64   # KV 序列分块
BLOCK_H = 64   # Head 分块
```

**建议**:
- `BLOCK_N`: 64 是较好的平衡点
- `BLOCK_H`: 不超过 kv_group_num
- 确保 dim % BLOCK == 0

### 2. Split-KV 策略

```python
num_split = 1   # 小 batch
num_split = 4   # 大 batch，低并行度
```

**使用场景**:
- batch 小或 seqlen 短时使用 num_split=1
- 需要增加并行度时增大 num_split

### 3. Swizzle 优化

```python
T.use_swizzle(10)
```

**效果**:
- 提高 L2 缓存命中率
- 减少线程块间的内存冲突

### 4. 快速数学

```python
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
```

**权衡**:
- 提高 5-10% 性能
- 轻微精度损失 (通常可接受)

### 5. Warp 策略选择

```python
# NVIDIA Hopper
policy=T.GemmWarpPolicy.FullCol

# AMD GPU
policy=T.GemmWarpPolicy.FullRow
```

**选择依据**:
- 根据 GPU 架构选择
- FullCol 适合 Tensor Core 布局
- FullRow 适合 AMD Wave 架构

### 6. 流水线深度

```python
for k in T.Pipelined(loop_range, num_stages=2):
```

**调优**:
- 通常 2-3 阶段效果最佳
- 过多阶段增加共享内存压力

### 7. 寄存器控制

```python
T.set_max_nreg(240, 1)  # 消费者
T.set_max_nreg(80, 0)   # 生产者
```

**分配策略**:
- 消费者需要更多寄存器进行计算
- 生产者只需少量寄存器进行数据传输
- 避免寄存器溢出到本地内存

---

## 基准测试

**文件**: `/root/dev/vibe_dsl/tilelang/examples/deepseek_mla/benchmark_mla.py`

### 1. 测试配置

```python
shape_configs = [
    {
        "b": batch,
        "s_q": 1,
        "cache_seqlens": torch.tensor([seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device="cuda"),
        "h_q": head,
        "h_kv": 1,
        "d": 512 + 64,
        "dv": 512,
        "causal": True,
        "dtype": torch.float16,
    }
    for batch in [128]
    for seqlen in [1024, 2048, 4096, 8192, 16384, 32768]
    for head in [128]
]
```

**配置** (`benchmark_mla.py:541-556`):
- batch=128
- seqlen 从 1024 到 32768
- 128 heads，GQA (kv_head=1)

### 2. 性能对比

根据 README 中的基准测试结果：

| 实现 | 相对性能 | 代码行数 |
|------|----------|----------|
| FlashMLA (CUTLASS) | 1.0x (baseline) | ~2000+ |
| TileLang | ~0.95-1.0x | ~80 |
| FlashInfer | ~0.8x | - |
| Triton | ~0.7x | ~300 |

### 3. 关键优化点总结

1. **Layout Inference**: TileLang 自动推导最优缓冲区布局
2. **Split-KV**: 提高小 batch 场景的并行度
3. **Warp 特化**: Hopper 架构的显式优化
4. **Swizzle**: 提高缓存命中率
5. **Pipeline**: 重叠计算和内存访问

---

## 总结

TileLang 的 DeepSeek MLA 实现展示了如何高效实现现代 LLM 推理优化：

1. **低秩压缩**: 利用 MLA 的 KV 共享特性减少内存占用
2. **分页支持**: 适配 vLLM 等推理框架的 KV Cache 管理
3. **架构优化**: Warp 特化和显式屏障同步
4. **跨平台**: NVIDIA 和 AMD GPU 的适配
5. **易用性**: 约 80 行代码实现接近 CUTLASS 的性能

这些技术为构建高性能 LLM 推理系统提供了重要参考。
