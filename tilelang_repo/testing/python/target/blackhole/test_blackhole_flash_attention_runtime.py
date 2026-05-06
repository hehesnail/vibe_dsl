import sys
import types
import re
from pathlib import Path

import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import assert_tensors_close_or_dump, check_blackhole_direct_execution_requirements


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_gqa_fwd_bshd as gqa_example
import example_mha_fwd_bshd as mha_example


BLACKHOLE_FLASH_ATTENTION_DTYPE_EXPR = "T.bfloat16"
BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE = torch.bfloat16
MULTI_PAGE_EXACT_CB_REPUBLISH_REASON = "multi-page exact CB-republish live-form"
MULTI_BLOCK_EXACT_CB_REPUBLISH_REASON = "multi-block exact CB-republish"
LOOP_CARRIED_EXACT_CB_PACR_REASON = (
    "loop-carried exact-CB backedge direct runtime is gated: TT-Sim reports "
    "tensix_execute_pacr: count=1 for the admitted compute-side pack path"
)

def _load_flash_attention_module_with_dtype(module_path, dtype_expr=BLACKHOLE_FLASH_ATTENTION_DTYPE_EXPR):
    source = Path(module_path).read_text()
    source = source.replace("dtype = T.float16", f"dtype = {dtype_expr}", 1)
    mutated = types.ModuleType(f"{Path(module_path).stem}_{dtype_expr.replace('.', '_')}")
    mutated.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), mutated.__dict__)
    return mutated


blackhole_gqa_example = _load_flash_attention_module_with_dtype(gqa_example.__file__)
blackhole_mha_example = _load_flash_attention_module_with_dtype(mha_example.__file__)


def paged_gqa_decode_kernel(
    *,
    batch=2,
    heads=4,
    groups=4,
    pages_per_sequence=2,
    total_pages=4,
    block_M=32,
    block_N=32,
    dim=32,
):
    """Ordinary TIR paged GQA decode tile for the first T9.2 slice."""
    assert pages_per_sequence == 2
    assert heads == groups
    dtype = T.bfloat16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        Q: T.Tensor((batch, block_M, heads, dim), dtype),
        KCache: T.Tensor((total_pages * block_N, dim), dtype),
        VCache: T.Tensor((total_pages * block_N, dim), dtype),
        PageTable: T.Tensor((batch, pages_per_sequence), "int32"),
        CacheSeqLens: T.Tensor((batch,), "int32"),
        Output: T.Tensor((batch, block_M, heads, dim), dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K0_shared = T.alloc_shared((block_N, dim), dtype)
            V0_shared = T.alloc_shared((block_N, dim), dtype)
            K1_shared = T.alloc_shared((block_N, dim), dtype)
            V1_shared = T.alloc_shared((block_N, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q[bx, 0:block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            cache_len = CacheSeqLens[bx]

            page0_k = PageTable[bx, 0]
            for i, j in T.Parallel(block_N, dim):
                K0_shared[i, j] = T.if_then_else(
                    i < cache_len,
                    KCache[page0_k * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K0_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    j < cache_len,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            page0_v = PageTable[bx, 0]
            for i, j in T.Parallel(block_N, dim):
                V0_shared[i, j] = T.if_then_else(
                    i < cache_len,
                    VCache[page0_v * block_N + i, j],
                    0,
                )
            T.gemm(acc_s_cast, V0_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            page1_k = PageTable[bx, 1]
            for i, j in T.Parallel(block_N, dim):
                K1_shared[i, j] = T.if_then_else(
                    block_N + i < cache_len,
                    KCache[page1_k * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K1_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    block_N + j < cache_len,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            page1_v = PageTable[bx, 1]
            for i, j in T.Parallel(block_N, dim):
                V1_shared[i, j] = T.if_then_else(
                    block_N + i < cache_len,
                    VCache[page1_v * block_N + i, j],
                    0,
                )
            T.gemm(acc_s_cast, V1_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, 0:block_M, by, :])

    return main


def _paged_gqa_decode_reference(q, k_cache, v_cache, page_table, cache_seq_lens):
    batch, _, heads, dim = q.shape
    page_rows = 32
    out = torch.empty_like(q)
    scale = dim ** -0.5
    for seq in range(batch):
        pages = [
            k_cache[int(page_table[seq, page]) * page_rows : (int(page_table[seq, page]) + 1) * page_rows]
            for page in range(page_table.shape[1])
        ]
        values = [
            v_cache[int(page_table[seq, page]) * page_rows : (int(page_table[seq, page]) + 1) * page_rows]
            for page in range(page_table.shape[1])
        ]
        cache_len = int(cache_seq_lens[seq])
        k_seq = torch.cat(pages, dim=0)[:cache_len]
        v_seq = torch.cat(values, dim=0)[:cache_len]
        for head in range(heads):
            scores = torch.matmul(q[seq, :, head, :].float(), k_seq.float().T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[seq, :, head, :] = torch.matmul(probs, v_seq.float()).to(q.dtype)
    return out


def sparse_ragged_gqa_decode_kernel(
    *,
    batch=2,
    heads=4,
    groups=4,
    sparse_blocks=2,
    total_blocks=4,
    block_M=32,
    block_N=32,
    dim=32,
):
    """Ordinary TIR sparse/ragged GQA decode tile for the first T9.4 slice."""
    assert sparse_blocks == 2
    assert heads == groups
    dtype = T.bfloat16
    accum_dtype = T.float32
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        Q: T.Tensor((batch, block_M, heads, dim), dtype),
        KBlocks: T.Tensor((total_blocks * block_N, dim), dtype),
        VBlocks: T.Tensor((total_blocks * block_N, dim), dtype),
        BlockIndices: T.Tensor((batch, sparse_blocks), "int32"),
        ValidRows: T.Tensor((batch, sparse_blocks), "int32"),
        Output: T.Tensor((batch, block_M, heads, dim), dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K0_shared = T.alloc_shared((block_N, dim), dtype)
            V0_shared = T.alloc_shared((block_N, dim), dtype)
            K1_shared = T.alloc_shared((block_N, dim), dtype)
            V1_shared = T.alloc_shared((block_N, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q[bx, 0:block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            block0 = BlockIndices[bx, 0]
            valid0 = ValidRows[bx, 0]
            for i, j in T.Parallel(block_N, dim):
                K0_shared[i, j] = T.if_then_else(
                    i < valid0,
                    KBlocks[block0 * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K0_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    j < valid0,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            for i, j in T.Parallel(block_N, dim):
                V0_shared[i, j] = T.if_then_else(
                    i < valid0,
                    VBlocks[block0 * block_N + i, j],
                    0,
                )
            T.gemm(acc_s_cast, V0_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            block1 = BlockIndices[bx, 1]
            valid1 = ValidRows[bx, 1]
            for i, j in T.Parallel(block_N, dim):
                K1_shared[i, j] = T.if_then_else(
                    i < valid1,
                    KBlocks[block1 * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K1_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    j < valid1,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            for i, j in T.Parallel(block_N, dim):
                V1_shared[i, j] = T.if_then_else(
                    i < valid1,
                    VBlocks[block1 * block_N + i, j],
                    0,
                )
            T.gemm(acc_s_cast, V1_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, 0:block_M, by, :])

    return main


def _sparse_ragged_gqa_decode_reference(q, k_blocks, v_blocks, block_indices, valid_rows):
    batch, _, heads, dim = q.shape
    block_rows = 32
    out = torch.empty_like(q)
    scale = dim ** -0.5
    for seq in range(batch):
        k_parts = []
        v_parts = []
        for slot in range(block_indices.shape[1]):
            block = int(block_indices[seq, slot])
            rows = int(valid_rows[seq, slot])
            k_parts.append(k_blocks[block * block_rows : block * block_rows + rows])
            v_parts.append(v_blocks[block * block_rows : block * block_rows + rows])
        k_seq = torch.cat(k_parts, dim=0)
        v_seq = torch.cat(v_parts, dim=0)
        for head in range(heads):
            scores = torch.matmul(q[seq, :, head, :].float(), k_seq.float().T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[seq, :, head, :] = torch.matmul(probs, v_seq.float()).to(q.dtype)
    return out


def paged_mla_decode_kernel(
    *,
    batch=2,
    heads=4,
    pages_per_sequence=2,
    total_pages=4,
    block_M=32,
    block_N=32,
    dv=32,
    dpe=32,
):
    """Ordinary TIR paged MLA decode tile for the first T9.3 slice."""
    assert pages_per_sequence == 2
    dtype = T.bfloat16
    accum_dtype = T.float32
    scale = (1.0 / (dv + dpe)) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        QNope: T.Tensor((batch, block_M, heads, dv), dtype),
        QPe: T.Tensor((batch, block_M, heads, dpe), dtype),
        KVLatentCache: T.Tensor((total_pages * block_N, dv), dtype),
        KPeCache: T.Tensor((total_pages * block_N, dpe), dtype),
        PageTable: T.Tensor((batch, pages_per_sequence), "int32"),
        CacheSeqLens: T.Tensor((batch,), "int32"),
        Output: T.Tensor((batch, block_M, heads, dv), dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            QNope_shared = T.alloc_shared((block_M, dv), dtype)
            QPe_shared = T.alloc_shared((block_M, dpe), dtype)
            KV0_shared = T.alloc_shared((block_N, dv), dtype)
            KPe0_shared = T.alloc_shared((block_N, dpe), dtype)
            KV1_shared = T.alloc_shared((block_N, dv), dtype)
            KPe1_shared = T.alloc_shared((block_N, dpe), dtype)
            O_shared = T.alloc_shared((block_M, dv), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, dv), accum_dtype)
            scores_max = T.alloc_fragment((block_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
            scores_scale = T.alloc_fragment((block_M,), accum_dtype)
            scores_sum = T.alloc_fragment((block_M,), accum_dtype)
            logsum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(QNope[bx, 0:block_M, by, :], QNope_shared)
            T.copy(QPe[bx, 0:block_M, by, :], QPe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            cache_len = CacheSeqLens[bx]

            page0 = PageTable[bx, 0]
            for i, j in T.Parallel(block_N, dv):
                KV0_shared[i, j] = T.if_then_else(
                    i < cache_len,
                    KVLatentCache[page0 * block_N + i, j],
                    0,
                )
            for i, j in T.Parallel(block_N, dpe):
                KPe0_shared[i, j] = T.if_then_else(
                    i < cache_len,
                    KPeCache[page0 * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                QNope_shared,
                KV0_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.gemm(
                QPe_shared,
                KPe0_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    j < cache_len,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dv):
                acc_o[i, j] *= scores_scale[i]
            T.gemm(acc_s_cast, KV0_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            page1 = PageTable[bx, 1]
            for i, j in T.Parallel(block_N, dv):
                KV1_shared[i, j] = T.if_then_else(
                    block_N + i < cache_len,
                    KVLatentCache[page1 * block_N + i, j],
                    0,
                )
            for i, j in T.Parallel(block_N, dpe):
                KPe1_shared[i, j] = T.if_then_else(
                    block_N + i < cache_len,
                    KPeCache[page1 * block_N + i, j],
                    0,
                )
            T.fill(acc_s, 0)
            T.gemm(
                QNope_shared,
                KV1_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.gemm(
                QPe_shared,
                KPe1_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    block_N + j < cache_len,
                    acc_s[i, j],
                    -T.infinity(acc_s.dtype),
                )
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, dv):
                acc_o[i, j] *= scores_scale[i]
            T.gemm(acc_s_cast, KV1_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dv):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, 0:block_M, by, :])

    return main


def _paged_mla_decode_reference(q_nope, q_pe, kv_latent, k_pe, page_table, cache_seq_lens):
    batch, _, heads, dv = q_nope.shape
    dpe = q_pe.shape[-1]
    page_rows = 32
    out = torch.empty_like(q_nope)
    scale = (dv + dpe) ** -0.5
    for seq in range(batch):
        latent_pages = [
            kv_latent[
                int(page_table[seq, page]) * page_rows : (int(page_table[seq, page]) + 1)
                * page_rows
            ]
            for page in range(page_table.shape[1])
        ]
        pe_pages = [
            k_pe[
                int(page_table[seq, page]) * page_rows : (int(page_table[seq, page]) + 1)
                * page_rows
            ]
            for page in range(page_table.shape[1])
        ]
        cache_len = int(cache_seq_lens[seq])
        kv_seq = torch.cat(latent_pages, dim=0)[:cache_len]
        k_pe_seq = torch.cat(pe_pages, dim=0)[:cache_len]
        for head in range(heads):
            scores = (
                torch.matmul(q_nope[seq, :, head, :].float(), kv_seq.float().T)
                + torch.matmul(q_pe[seq, :, head, :].float(), k_pe_seq.float().T)
            ) * scale
            probs = torch.softmax(scores, dim=-1)
            out[seq, :, head, :] = torch.matmul(probs, kv_seq.float()).to(q_nope.dtype)
    return out


def paged_mla_dual_score_kernel(
    *,
    batch=1,
    heads=1,
    total_pages=3,
    block_M=32,
    block_N=32,
    dv=32,
    dpe=32,
):
    """Ordinary TIR score-only MLA page tile for isolating T9.3 score lowering."""
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        QNope: T.Tensor((batch, block_M, heads, dv), dtype),
        QPe: T.Tensor((batch, block_M, heads, dpe), dtype),
        KVLatentCache: T.Tensor((total_pages * block_N, dv), dtype),
        KPeCache: T.Tensor((total_pages * block_N, dpe), dtype),
        PageTable: T.Tensor((batch, 1), "int32"),
        Output: T.Tensor((batch, block_M, heads, block_N), accum_dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            QNope_shared = T.alloc_shared((block_M, dv), dtype)
            QPe_shared = T.alloc_shared((block_M, dpe), dtype)
            KV_shared = T.alloc_shared((block_N, dv), dtype)
            KPe_shared = T.alloc_shared((block_N, dpe), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.copy(QNope[bx, 0:block_M, by, :], QNope_shared)
            T.copy(QPe[bx, 0:block_M, by, :], QPe_shared)
            page = PageTable[bx, 0]
            for i, j in T.Parallel(block_N, dv):
                KV_shared[i, j] = KVLatentCache[page * block_N + i, j]
            for i, j in T.Parallel(block_N, dpe):
                KPe_shared[i, j] = KPeCache[page * block_N + i, j]

            T.fill(acc_s, 0)
            T.gemm(
                QNope_shared,
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.gemm(
                QPe_shared,
                KPe_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(acc_s, Output[bx, 0:block_M, by, :])

    return main


def paged_qk_gemm_kernel(
    *,
    batch=2,
    heads=4,
    pages_per_sequence=1,
    page_column=0,
    total_pages=4,
    block_M=32,
    block_N=32,
    dim=32,
):
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor((batch, block_M, heads, dim), dtype),
        KCache: T.Tensor((total_pages * block_N, dim), dtype),
        PageTable: T.Tensor((batch, pages_per_sequence), "int32"),
        Output: T.Tensor((batch, block_M, heads, block_N), accum_dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.copy(Q[bx, 0:block_M, by, :], Q_shared)
            page = PageTable[bx, page_column]
            for i, j in T.Parallel(block_N, dim):
                K_shared[i, j] = KCache[page * block_N + i, j]
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(acc_s, Output[bx, 0:block_M, by, :])

    return main


def paged_av_gemm_kernel(
    *,
    batch=2,
    heads=4,
    pages_per_sequence=1,
    page_column=0,
    total_pages=4,
    block_M=32,
    block_N=32,
    dim=32,
):
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((batch, block_M, heads, block_N), dtype),
        VCache: T.Tensor((total_pages * block_N, dim), dtype),
        PageTable: T.Tensor((batch, pages_per_sequence), "int32"),
        Output: T.Tensor((batch, block_M, heads, dim), accum_dtype),
    ):
        with T.Kernel(batch, heads, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            V_shared = T.alloc_shared((block_N, dim), dtype)
            acc_o = T.alloc_fragment((block_M, dim), accum_dtype)

            T.copy(A[bx, 0:block_M, by, :], A_shared)
            page = PageTable[bx, page_column]
            for i, j in T.Parallel(block_N, dim):
                V_shared[i, j] = VCache[page * block_N + i, j]
            T.fill(acc_o, 0)
            T.gemm(A_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc_o, Output[bx, 0:block_M, by, :])

    return main


def seq_qk_gemm_kernel(
    *,
    batch=1,
    heads=4,
    seq_len=64,
    block_M=32,
    block_N=32,
    dim=32,
):
    assert batch == 1
    assert seq_len % block_M == 0
    dtype = T.bfloat16
    accum_dtype = T.float32
    seq_tiles = seq_len // block_M

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),
        K: T.Tensor((batch, seq_len, heads, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads, block_N), accum_dtype),
    ):
        with T.Kernel(seq_tiles, heads, threads=128) as (bx, by):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.copy(Q[0, bx * block_M : bx * block_M + block_M, by, :], Q_shared)
            T.copy(K[0, bx * block_N : bx * block_N + block_N, by, :], K_shared)
            T.fill(acc_s, 0)
            T.gemm(
                Q_shared,
                K_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(acc_s, Output[0, bx * block_M : bx * block_M + block_M, by, :])

    return main


def _lower_blackhole_flash_attention_metadata(kernel):
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)
    return artifact, artifact.codegen_mod.get_function_metadata("main")


def _run_blackhole_flash_attention(kernel, *inputs):
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    reasons = metadata.get("direct_runtime_unsupported_reasons", [])
    if reasons:
        pytest.skip(
            "Blackhole flash-attention direct runtime is not yet supported for this kernel: "
            + ", ".join(str(reason) for reason in reasons)
        )
    artifact.codegen_mod["main"](*inputs)


def _has_multi_page_republish_event(metadata):
    return any(
        str(config["flow_class"]) == "republish"
        and (
            int(config.get("publish_pages_per_event", 0)) > 1
            or int(config.get("consume_pages_per_event", 0)) > 1
        )
        for config in metadata["cb_configs"]
    )


def _extract_c_for_loop_body(source, header):
    start = source.find(header)
    assert start >= 0, f"missing C loop header: {header}"
    open_brace = source.find("{", start)
    assert open_brace >= 0, f"missing C loop open brace after: {header}"
    depth = 0
    for pos in range(open_brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : pos], source[pos + 1 :]
    raise AssertionError(f"missing C loop close brace after: {header}")


def _split_optional_c_for_loop_body(source, header):
    if header not in source:
        return "", source
    return _extract_c_for_loop_body(source, header)


def _assert_compute_cb_reserves_fit_visible_capacity(metadata, compute_source):
    cb_capacity = {
        int(config["cb_id"]): int(config["num_pages"])
        for config in metadata["cb_configs"]
    }
    visible_front_pages = {}
    over_reserve_sites = []
    for event in re.finditer(
        r"\bcb_(reserve_back|push_back|pop_front)\((\d+),\s*(\d+)\);",
        compute_source,
    ):
        kind, cb_id_text, pages_text = event.groups()
        cb_id = int(cb_id_text)
        pages = int(pages_text)
        if kind == "reserve_back":
            front_pages = visible_front_pages.get(cb_id, 0)
            capacity = cb_capacity.get(cb_id)
            if capacity is not None and front_pages + pages > capacity:
                line = compute_source.count("\n", 0, event.start()) + 1
                over_reserve_sites.append(
                    f"line {line}: cb{cb_id} front={front_pages} "
                    f"reserve={pages} capacity={capacity}"
                )
        elif kind == "push_back":
            visible_front_pages[cb_id] = visible_front_pages.get(cb_id, 0) + pages
        elif kind == "pop_front":
            visible_front_pages[cb_id] = max(0, visible_front_pages.get(cb_id, 0) - pages)

    assert over_reserve_sites == []


def _assert_compute_cb_waits_only_visible_pages(metadata, compute_source):
    input_cbs = {
        int(config["cb_id"])
        for config in metadata["cb_configs"]
        if str(config["role"]) == "input"
    }
    visible_front_pages = {}
    under_wait_sites = []
    for event in re.finditer(
        r"\bcb_(wait_front|push_back|pop_front)\((\d+),\s*(\d+)\);",
        compute_source,
    ):
        kind, cb_id_text, pages_text = event.groups()
        cb_id = int(cb_id_text)
        pages = int(pages_text)
        if kind == "wait_front":
            if cb_id not in input_cbs and visible_front_pages.get(cb_id, 0) < pages:
                line = compute_source.count("\n", 0, event.start()) + 1
                under_wait_sites.append(
                    f"line {line}: cb{cb_id} front={visible_front_pages.get(cb_id, 0)} "
                    f"wait={pages}"
                )
        elif kind == "push_back":
            visible_front_pages[cb_id] = visible_front_pages.get(cb_id, 0) + pages
        elif kind == "pop_front":
            visible_front_pages[cb_id] = max(0, visible_front_pages.get(cb_id, 0) - pages)

    assert under_wait_sites == []


def _assert_t7_seq64_mha_exact_cb_partial_combine_contract(metadata):
    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == []
    assert not any(MULTI_PAGE_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)
    assert not any(MULTI_BLOCK_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)

    cb_by_name = {str(config["name"]): config for config in metadata["cb_configs"]}
    acc_s_cb = cb_by_name["acc_s"]
    assert str(acc_s_cb["data_format"]) == "Float16_b"
    assert int(acc_s_cb["page_size"]) == 2048
    for cb_name in ("K_shared", "V_shared", "acc_s_cast"):
        cb = cb_by_name[cb_name]
        assert int(cb["num_pages"]) == 2
        assert int(cb["publish_pages_per_event"]) == 1
        assert int(cb["consume_pages_per_event"]) == 1

    materialization_plans = {
        str(plan["target_buffer"]): plan for plan in metadata["materialization_plans"]
    }
    acc_s_cast_plan = materialization_plans["acc_s_cast"]
    assert str(acc_s_cast_plan["source_live_form"]) == "live_form_acc_s"
    assert str(acc_s_cast_plan["materialization_protocol"]) == "cb_republish"
    assert str(acc_s_cast_plan["publication_protocol"]) == "tilize_cast_fragment_slice"
    assert str(acc_s_cast_plan["produced_live_form"]) == "live_form_acc_s_cast"

    live_form_plans = {
        str(plan["name"]): plan for plan in metadata["live_form_plans"]
    }
    assert str(live_form_plans["live_form_acc_s"]["physical_form"]) == "thread_distributed_slice"
    assert str(live_form_plans["live_form_acc_s_cast"]["physical_form"]) == "cb_materialized_tile"
    assert (
        str(live_form_plans["live_form_acc_s_cast"]["ownership_kind"])
        == "materialized_cb_pages_single_event"
    )

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
        )
    )
    assert "tilelang_add_fragment(dst, src, num_elements);" not in compute_source
    assert "tilelang_get_cb_write_ptr_bytes" not in compute_source
    assert "get_tile_address(0)" not in compute_source
    assert "add_tiles_init(" in compute_source
    assert "add_tiles(" in compute_source
    q_cb = int(cb_by_name["Q_shared"]["cb_id"])
    k_cb = int(cb_by_name["K_shared"]["cb_id"])
    v_cb = int(cb_by_name["V_shared"]["cb_id"])
    acc_o_cb = int(cb_by_name["acc_o"]["cb_id"])
    assert f"matmul_tiles({q_cb}, {k_cb}, 0, 0, 0);" in compute_source
    assert f"matmul_tiles({q_cb}, {k_cb}, 0, 1, 0);" in compute_source
    assert re.search(rf"matmul_tiles\(\d+, {v_cb}, 0, 0, 0\);", compute_source)
    assert re.search(rf"matmul_tiles\(\d+, {v_cb}, 0, 1, 0\);", compute_source)
    assert f"add_tiles_init({acc_o_cb}, " not in compute_source
    serial_loop_body, after_serial_loop = _split_optional_c_for_loop_body(
        compute_source, "for (int32_t tx = 0; tx < 128; ++tx)"
    )
    assert "matmul_tiles(" not in serial_loop_body
    assert "reduce_tile<" not in serial_loop_body
    assert "pack_tile(" not in serial_loop_body
    for cb_id, pop_pages in ((q_cb, 1), (k_cb, 2), (v_cb, 2)):
        assert f"cb_pop_front({cb_id}," not in serial_loop_body
        assert f"cb_pop_front({cb_id}, {pop_pages});" in after_serial_loop

    merge_pairs = re.findall(r"add_tiles_init\((\d+), (\d+)\);", compute_source)
    assert merge_pairs
    merge_window_pattern = re.compile(
        r"add_tiles_init\((\d+), (\d+)\);.*?add_tiles\(\1, \2, 0, 0, 0\);.*?"
        r"pack_tile\(0, (\d+)(?:, \d+)?\);",
        re.DOTALL,
    )
    merge_windows = list(merge_window_pattern.finditer(compute_source))
    assert merge_windows
    assert all("tile_regs_commit()" in window.group(0) for window in merge_windows)
    assert all("tile_regs_wait()" in window.group(0) for window in merge_windows)

    merge_cb_ids = {cb_id for pair in merge_pairs for cb_id in pair}
    merge_output_cb_ids = {window.group(3) for window in merge_windows}
    for cb_id in merge_cb_ids:
        assert re.search(rf"cb_wait_front\({cb_id},\s*\d+\);", compute_source)
    for cb_id in merge_output_cb_ids:
        assert re.search(rf"cb_reserve_back\({cb_id},\s*\d+\);", compute_source)
        assert re.search(rf"cb_push_back\({cb_id},\s*\d+\);", compute_source)
    assert any(f"cb_pop_front({cb_id}, 1);" in compute_source for cb_id in merge_cb_ids)
    _assert_compute_cb_reserves_fit_visible_capacity(metadata, compute_source)
    _assert_compute_cb_waits_only_visible_pages(metadata, compute_source)


def test_blackhole_flash_attention_runtime_gate_is_queryable():
    can_run, msg = check_blackhole_direct_execution_requirements()
    assert isinstance(can_run, bool)
    assert isinstance(msg, str)


def test_blackhole_flash_attention_single_work_item_metadata_drops_contract_family():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in metadata


def test_blackhole_flash_attention_single_work_item_runtime_metadata_admits_typed_materialization():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert list(metadata["tvm_arg_names"]) == ["Q", "K", "V", "Output"]
    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)
    assert "compute_epilogue_ops" not in metadata
    materialization_plans = {
        str(plan["target_buffer"]): plan for plan in metadata["materialization_plans"]
    }
    assert str(materialization_plans["acc_s_cast"]["materialization_protocol"]) == "cb_republish"
    assert (
        str(materialization_plans["acc_s_cast"]["publication_protocol"])
        == "tilize_cast_fragment_slice"
    )


def test_blackhole_flash_attention_small_bf16_compute_source_uses_non_mailbox_publication():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    assert not any(
        "thread-distributed cb_republish materialization" in str(reason)
        for reason in metadata.get("direct_runtime_unsupported_reasons", [])
    )

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    assert "tilelang_get_cb_write_ptr_bytes" not in compute_source
    assert "tilelang_cb_write_ptr_bytes_direct" not in compute_source
    assert "get_local_cb_interface" not in compute_source
    assert "mailbox_write" not in compute_source
    assert "mailbox_read" not in compute_source

    cb_configs = {str(config["name"]): config for config in metadata["cb_configs"]}
    reduce_scalers = [
        config for name, config in cb_configs.items() if "exact_const_tile_reduce_scaler" in name
    ]
    assert reduce_scalers
    assert all(str(config["data_format"]) == "Float16_b" for config in reduce_scalers)

    pack_cb_ids = [
        int(cb_id)
        for cb_id in re.findall(
            r"\b(?:pack_reconfig_data_format(?:<true>)?|pack_tile)\([^;\n]*?(\d+)",
            compute_source,
        )
    ]
    assert pack_cb_ids
    assert max(pack_cb_ids) <= 31


def test_blackhole_flash_attention_small_bf16_exact_cb_reuse_releases_before_reserve():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    _assert_compute_cb_reserves_fit_visible_capacity(metadata, compute_source)


def test_blackhole_flash_attention_small_bf16_exact_cb_reuse_waits_only_live_pages():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    _assert_compute_cb_waits_only_visible_pages(metadata, compute_source)


def test_blackhole_flash_attention_small_bf16_prunes_dead_constant_fragment_fills():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    dead_fill_vars = []
    for fill in re.finditer(
        r"tilelang_fill_fragment\(dst,\s*[^;]+;\s*\}\);",
        compute_source,
    ):
        prefix = compute_source[max(0, fill.start() - 180) : fill.start()]
        dst_match = re.search(r"reinterpret_cast<[^>]+>\((\w+)\)", prefix)
        if dst_match is None:
            continue
        dst_var = dst_match.group(1)
        if not re.search(rf"\b{re.escape(dst_var)}\b", compute_source[fill.end() :]):
            dead_fill_vars.append(dst_var)

    assert dead_fill_vars == []


def test_blackhole_flash_attention_small_bf16_compute_leafs_are_not_thread_serialized():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])

    assert "for (int32_t tx = 0; tx < 128; ++tx)" not in compute_source


def test_blackhole_flash_attention_first_row_reduction_consumes_matmul_live_form():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    first_reduce = re.search(
        r"reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>\((\d+),",
        compute_source,
    )
    assert first_reduce is not None
    reduce_src_cb = int(first_reduce.group(1))
    cb_configs = {int(config["cb_id"]): config for config in metadata["cb_configs"]}
    reduce_src_config = cb_configs[reduce_src_cb]

    assert str(reduce_src_config["flow_class"]) == "stream"
    assert "reduce_src" not in str(reduce_src_config["name"])

    first_reduce_offset = first_reduce.start()
    source_reserve_offset = compute_source.rfind(
        f"cb_reserve_back({reduce_src_cb},", 0, first_reduce_offset
    )
    source_push_offset = compute_source.rfind(
        f"cb_push_back({reduce_src_cb},", 0, first_reduce_offset
    )
    source_matmul_offset = compute_source.rfind("matmul_tiles(", 0, source_push_offset)
    assert source_reserve_offset >= 0
    assert source_push_offset > source_reserve_offset
    assert source_matmul_offset >= 0
    assert "fill_tile_bitcast" not in compute_source[source_reserve_offset:source_push_offset]


def test_blackhole_flash_attention_final_publish_consumes_normalized_live_form():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
        )
    )
    writer_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "writer" and str(kernel["core_type"]) == "ncrisc"
        )
    )
    writer_wait = re.search(r"cb_wait_front\((\d+), 1\);", writer_source)
    assert writer_wait is not None
    output_cb = int(writer_wait.group(1))

    final_publish_matches = list(
        re.finditer(
            rf"cb_wait_front\((?P<src>\d+), 1\);\s*"
            rf"cb_reserve_back\({output_cb}, 1\);\s*"
            r"tile_regs_acquire\(\);\s*"
            r"reconfig_data_format\((?P=src), (?P=src)\);\s*"
            r"copy_tile_to_dst_init_short(?:_with_dt)?\((?:\d+,\s*)?(?P=src)\);\s*"
            r"copy_tile\((?P=src), 0, 0\);\s*"
            r"tile_regs_commit\(\);\s*"
            r"tile_regs_wait\(\);\s*"
            rf"pack_reconfig_data_format(?:<true>)?\({output_cb}\);\s*"
            rf"pack_tile\(0, {output_cb}, 0\);\s*"
            r"tile_regs_release\(\);\s*"
            r"(?P<source_lifetime>(?:cb_pop_front\(\d+, 1\);\s*)*)"
            rf"cb_push_back\({output_cb}, 1\);",
            compute_source,
        )
    )
    final_publish = final_publish_matches[-1] if final_publish_matches else None
    assert final_publish is not None
    copied_cb = int(final_publish.group("src"))
    assert f"cb_pop_front({copied_cb}, 1);" in final_publish.group("source_lifetime")

    cb_configs = {int(config["cb_id"]): config for config in metadata["cb_configs"]}
    source_config = cb_configs[copied_cb]
    assert copied_cb != output_cb
    assert str(source_config["data_format"]) == "Float16_b"


def test_blackhole_flash_attention_row_reduction_init_uses_rewritten_output_cb():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    compute_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_source = str(compute_kernel["source_code"])

    reduce_windows = re.findall(
        r"reduce_init<[^>]+>\(\d+, \d+, (\d+)\);"
        r".*?pack_reconfig_data_format(?:<true>)?\((\d+)\);"
        r"\npack_tile\(0, (\d+), 0\);",
        compute_source,
        flags=re.DOTALL,
    )
    assert reduce_windows
    assert all(int(init_cb) == int(pack_cb) == int(pack_tile_cb)
               for init_cb, pack_cb, pack_tile_cb in reduce_windows)


def test_blackhole_flash_attention_reader_reserves_each_input_cb_before_read():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        1,
        32,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reader_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "reader" and str(kernel["core_type"]) == "brisc"
    )
    reader_source = str(reader_kernel["source_code"])
    read_windows = re.findall(
        r"cb_reserve_back\((\d+), 1\);"
        r"\n\{[^{}]*get_write_ptr\((\d+)\).*?read_tile\(tile_index, src_gen, cb_l1_addr\);"
        r".*?\};\ncb_push_back\((\d+), 1\);",
        reader_source,
        flags=re.DOTALL,
    )
    assert len(read_windows) == 3
    assert all(int(reserve_cb) == int(write_cb) == int(push_cb)
               for reserve_cb, write_cb, push_cb in read_windows)


@pytest.mark.parametrize(
    ("kernel",),
    [
        (
            blackhole_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                32,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
        ),
        (
            blackhole_gqa_example.flashattn.jit_impl.get_tir(
                1,
                4,
                32,
                32,
                False,
                groups=4,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
        ),
        (
            blackhole_gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                128,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
        ),
        (
            blackhole_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
        ),
        (
            blackhole_gqa_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                groups=4,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
        ),
    ],
)
def test_blackhole_flash_attention_multi_work_item_metadata_exposes_explicit_per_work_access_bindings(
    kernel,
):
    _, metadata = _lower_blackhole_flash_attention_metadata(
        kernel
    )

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any("missing explicit per-work access binding" in reason for reason in reasons)
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)
    if _has_multi_page_republish_event(metadata):
        assert any(MULTI_PAGE_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)
    else:
        assert not any(MULTI_PAGE_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)

    reader_specs = [
        spec
        for kernel in metadata["kernels"]
        if kernel["kind"] == "reader"
        for spec in kernel["per_work_arg_specs"]
    ]
    writer_specs = [
        spec
        for kernel in metadata["kernels"]
        if kernel["kind"] == "writer"
        for spec in kernel["per_work_arg_specs"]
    ]
    assert reader_specs
    assert writer_specs
    assert all(str(spec["arg_kind"]) for spec in reader_specs + writer_specs)
    assert all(str(spec["value_source"]) for spec in reader_specs + writer_specs)
    assert all(str(spec["arg_identity"]) for spec in reader_specs + writer_specs)

    reader_start_sources = {
        str(spec["value_source"])
        for spec in reader_specs
        if str(spec["arg_kind"]) in {"a_tile_start_id", "b_tile_start_id"}
    }
    assert reader_start_sources & {
        "logical_block_y",
        "logical_block_yx_linear",
        "work_linear_id",
    }
    assert any(
        str(spec["arg_kind"]) in {"a_tile_start_id", "b_tile_start_id", "output_tile_start_id"}
        and str(spec["value_source"])
        in {"logical_block_y", "logical_block_yx_linear", "work_linear_id"}
        for spec in reader_specs + writer_specs
    )


def test_blackhole_flash_attention_seq64_bf16_metadata_admits_multi_block_direct_runtime_contract():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        4,
        64,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any(MULTI_PAGE_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)
    assert not any(MULTI_BLOCK_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)

    cb_by_name = {str(config["name"]): config for config in metadata["cb_configs"]}
    for cb_name in ("K_shared", "V_shared", "acc_s_cast"):
        cb = cb_by_name[cb_name]
        assert int(cb["num_pages"]) == 2
        assert int(cb["publish_pages_per_event"]) == 1
        assert int(cb["consume_pages_per_event"]) == 1

    materialization_plans = {
        str(plan["target_buffer"]): plan for plan in metadata["materialization_plans"]
    }
    assert str(materialization_plans["acc_s_cast"]["materialization_protocol"]) == "cb_republish"
    assert (
        str(materialization_plans["acc_s_cast"]["publication_protocol"])
        == "tilize_cast_fragment_slice"
    )

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
        )
    )
    cb_format_by_id = {
        int(config["cb_id"]): str(config["data_format"]) for config in metadata["cb_configs"]
    }
    pack_reconfig_cb_ids = [
        int(cb_id)
        for cb_id in re.findall(
            r"pack_reconfig_data_format(?:<true>)?\((\d+)\);",
            compute_source,
        )
    ]
    assert any(
        cb_format_by_id[cb_id] == "Float16_b"
        for cb_id in pack_reconfig_cb_ids
    )


def test_blackhole_flash_attention_runtime_metadata_preserves_buffer_abi_order():
    _, metadata = _lower_blackhole_flash_attention_metadata(
        blackhole_mha_example.flashattn.jit_impl.get_tir(
            1,
            32,
            128,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )

    buffer_abi_order = [
        arg["buffer"]
        for arg in metadata["runtime_args"]
        if arg["kind"] in {"input_buffer_addr32", "input_buffer_addr", "output_buffer_addr32", "output_buffer_addr"}
    ]
    assert buffer_abi_order == ["Q", "K", "V", "Output"]


def test_blackhole_flash_attention_mha_bf16_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 32
    dim = 32
    is_causal = False
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole MHA bf16 flash-attention forward mismatch",
    )


def test_blackhole_flash_attention_gqa_bf16_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 32
    dim = 32
    is_causal = False
    groups = 4
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    head_kv = heads // groups
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, head_kv, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, head_kv, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_gqa_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        groups=groups,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_gqa_example.ref_program(q, k, v, is_causal=is_causal, groups=groups).to(
        dtype=out.dtype
    )
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole GQA bf16 flash-attention forward mismatch",
    )


def test_blackhole_flash_attention_seq64_mha_bf16_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 64
    dim = 32
    is_causal = False
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole seq64 MHA bf16 flash-attention forward mismatch",
    )


def test_blackhole_t7_seq64_mha_bf16_exact_cb_partial_combine_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 64
    dim = 32
    is_causal = False
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    _assert_t7_seq64_mha_exact_cb_partial_combine_contract(metadata)
    artifact.codegen_mod["main"](q, k, v, out)

    ref = blackhole_mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message=(
            "Blackhole T7 seq64 MHA bf16 exact-CB partial-combine direct runtime mismatch"
        ),
    )


@pytest.mark.parametrize("seq_len", [128, 256, 512])
def test_blackhole_flash_attention_extended_seq_metadata_carries_loop_carried_exact_cb(seq_len):
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        1,
        4,
        seq_len,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == [LOOP_CARRIED_EXACT_CB_PACR_REASON]

    virtual_values = list(metadata.get("exact_cb_virtual_values", []))
    intervals = list(metadata.get("exact_cb_live_intervals", []))
    allocations = list(metadata.get("exact_cb_allocations", []))
    releases = list(metadata.get("exact_cb_release_events", []))

    acc_o_values = [
        value
        for value in virtual_values
        if str(value["logical_value"]) == "acc_o"
        and str(value["event_lifetime_kind"]) == "loop_carried"
    ]
    assert acc_o_values
    acc_o_names = {str(value["name"]) for value in acc_o_values}

    assert any(
        str(interval["virtual_value"]) in acc_o_names
        and bool(interval["loop_carried"])
        and bool(interval["live_in"])
        and bool(interval["live_out"])
        for interval in intervals
    )
    assert any(str(allocation["virtual_value"]) in acc_o_names for allocation in allocations)
    assert any(
        str(release["allocation"]) == str(allocation["name"])
        for allocation in allocations
        if str(allocation["virtual_value"]) in acc_o_names
        for release in releases
    )


@pytest.mark.parametrize("seq_len", [128, 256, 512])
def test_blackhole_flash_attention_extended_seq_mha_bf16_forward_direct_runtime(seq_len):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    dim = 32
    is_causal = False
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message=(
            f"Blackhole seq{seq_len} MHA bf16 flash-attention forward mismatch"
        ),
    )


def test_blackhole_flash_attention_seq64_gqa_bf16_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 64
    dim = 32
    is_causal = False
    groups = 4
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    head_kv = heads // groups
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, head_kv, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, head_kv, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_gqa_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        groups=groups,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_gqa_example.ref_program(q, k, v, is_causal=is_causal, groups=groups).to(
        dtype=out.dtype
    )
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole seq64 GQA bf16 flash-attention forward mismatch",
    )


def test_blackhole_t9_paged_gqa_decode_projects_page_table_and_cache_len_bindings():
    kernel = paged_gqa_decode_kernel()
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == []
    assert list(metadata["tvm_arg_names"]) == [
        "Q",
        "KCache",
        "VCache",
        "PageTable",
        "CacheSeqLens",
        "Output",
    ]

    reader = next(kernel for kernel in metadata["kernels"] if str(kernel["kind"]) == "reader")
    reader_source = str(reader["source_code"])
    assert "PageTable" not in reader_source
    assert "CacheSeqLens" not in reader_source
    assert "get_arg_val<uint32_t>" in reader_source

    page_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "PageTable" in str(spec.get("value_expr", ""))
    ]
    assert len(page_specs) >= 2
    assert {str(spec.get("buffer", "")) for spec in page_specs} == {"KCache", "VCache"}
    assert all("index_buffer" not in spec for spec in page_specs)

    valid_row_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "CacheSeqLens" in str(spec.get("value_expr", ""))
    ]
    assert valid_row_specs
    assert all("index_buffer" not in spec for spec in valid_row_specs)

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute"
        )
    )
    writer_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "writer"
        )
    )
    assert "add_tiles_init(" in compute_source
    assert "add_tiles(" in compute_source
    assert "tilelang_add_fragment(dst, src, num_elements);" not in compute_source
    assert "tilelang_cb_write_ptr_bytes_direct" not in compute_source

    guard_mask_cb_ids = {
        int(config["cb_id"])
        for config in metadata["cb_configs"]
        if str(config["role"]) == "input"
        and "_guard_mask_" in str(config["name"])
    }
    assert guard_mask_cb_ids
    for mask_cb_id in guard_mask_cb_ids:
        mask_apply = re.search(
            rf"binary_op_init_common\((\d+),\s*{mask_cb_id},\s*(\d+)\);"
            rf"(?P<body>.*?)"
            rf"reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>\((\d+),",
            compute_source,
            re.S,
        )
        assert mask_apply, f"missing guard mask apply followed by row max for CB {mask_cb_id}"
        guard_out_cb = int(mask_apply.group(2))
        reduce_src_cb = int(mask_apply.group(4))
        assert reduce_src_cb == guard_out_cb
        assert f"cb_pop_front({guard_out_cb}, 1);" not in mask_apply.group("body")

    serial_loop_body, after_serial_loop = _split_optional_c_for_loop_body(
        compute_source, "for (int32_t tx = 0; tx < 128; ++tx)"
    )
    reader_input_names = {"Q_shared", "K0_shared", "V0_shared", "K1_shared", "V1_shared"}
    reader_input_cbs = [
        config
        for config in metadata["cb_configs"]
        if str(config["role"]) == "input"
        and str(config["name"]) in reader_input_names
    ]
    assert {str(config["name"]) for config in reader_input_cbs} == reader_input_names
    for config in reader_input_cbs:
        cb_id = int(config["cb_id"])
        assert f"cb_pop_front({cb_id}," not in serial_loop_body
        assert f"cb_pop_front({cb_id}, 1);" in after_serial_loop

    writer_output_cb_ids = {
        int(match)
        for match in re.findall(r"get_read_ptr\((\d+)\)", writer_source)
    }
    assert writer_output_cb_ids
    writer_wait_cb_ids = {
        int(match)
        for match in re.findall(r"cb_wait_front\((\d+),\s*1\);", writer_source)
    }
    writer_pop_cb_ids = {
        int(match)
        for match in re.findall(r"cb_pop_front\((\d+),\s*1\);", writer_source)
    }
    assert writer_wait_cb_ids == writer_output_cb_ids
    assert writer_pop_cb_ids == writer_output_cb_ids
    for cb_id in writer_output_cb_ids:
        assert f"cb_reserve_back({cb_id}," not in serial_loop_body
        assert f"cb_push_back({cb_id}," not in serial_loop_body
        assert f"cb_reserve_back({cb_id}, 1);" in after_serial_loop
        assert f"cb_push_back({cb_id}, 1);" in after_serial_loop


def test_blackhole_t9_sparse_ragged_gqa_decode_projects_block_and_valid_row_bindings():
    kernel = sparse_ragged_gqa_decode_kernel()
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == []
    assert list(metadata["tvm_arg_names"]) == [
        "Q",
        "KBlocks",
        "VBlocks",
        "BlockIndices",
        "ValidRows",
        "Output",
    ]

    reader = next(kernel for kernel in metadata["kernels"] if str(kernel["kind"]) == "reader")
    reader_source = str(reader["source_code"])
    assert "BlockIndices" not in reader_source
    assert "ValidRows" not in reader_source
    assert "get_arg_val<uint32_t>" in reader_source

    block_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "BlockIndices" in str(spec.get("value_expr", ""))
    ]
    assert len(block_specs) >= 4
    assert {str(spec.get("buffer", "")) for spec in block_specs} == {"KBlocks", "VBlocks"}
    assert all("index_buffer" not in spec for spec in block_specs)

    valid_row_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "ValidRows" in str(spec.get("value_expr", ""))
    ]
    assert valid_row_specs
    assert all("index_buffer" not in spec for spec in valid_row_specs)

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute"
        )
    )
    assert "add_tiles_init(" in compute_source
    assert "add_tiles(" in compute_source
    assert "tilelang_add_fragment(dst, src, num_elements);" not in compute_source

    reader_input_names = {"Q_shared", "K0_shared", "V0_shared", "K1_shared", "V1_shared"}
    reader_input_configs = [
        config
        for config in metadata["cb_configs"]
        if str(config["role"]) == "input"
        and str(config["name"]) in reader_input_names
    ]
    assert {str(config["name"]) for config in reader_input_configs} == reader_input_names


def test_blackhole_t9_paged_mla_decode_projects_latent_and_pe_page_bindings():
    kernel = paged_mla_decode_kernel()
    _, metadata = _lower_blackhole_flash_attention_metadata(kernel)

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == []
    assert list(metadata["tvm_arg_names"]) == [
        "QNope",
        "QPe",
        "KVLatentCache",
        "KPeCache",
        "PageTable",
        "CacheSeqLens",
        "Output",
    ]

    reader = next(kernel for kernel in metadata["kernels"] if str(kernel["kind"]) == "reader")
    reader_source = str(reader["source_code"])
    assert "PageTable" not in reader_source
    assert "CacheSeqLens" not in reader_source
    assert "get_arg_val<uint32_t>" in reader_source

    page_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "PageTable" in str(spec.get("value_expr", ""))
    ]
    assert len(page_specs) >= 4
    page_buffers = {str(spec.get("buffer", "")) for spec in page_specs}
    assert {"KVLatentCache", "KPeCache"} <= page_buffers
    assert all("index_buffer" not in spec for spec in page_specs)

    valid_row_specs = [
        spec
        for spec in reader["per_work_arg_specs"]
        if str(spec.get("value_source", "")) == "value_expr"
        and "CacheSeqLens" in str(spec.get("value_expr", ""))
    ]
    assert valid_row_specs
    assert all("index_buffer" not in spec for spec in valid_row_specs)

    reader_input_names = {
        "QNope_shared",
        "QPe_shared",
        "KV0_shared",
        "KPe0_shared",
        "KV1_shared",
        "KPe1_shared",
    }
    reader_input_configs = [
        config
        for config in metadata["cb_configs"]
        if str(config["role"]) == "input"
        and str(config["name"]) in reader_input_names
    ]
    assert {str(config["name"]) for config in reader_input_configs} == reader_input_names

    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in metadata["kernels"]
            if str(kernel["kind"]) == "compute"
        )
    )
    assert compute_source.count("matmul_tiles(") >= 6
    assert "add_tiles_init(" in compute_source
    assert "add_tiles(" in compute_source
    assert "tilelang_cb_write_ptr_bytes_direct" not in compute_source


def test_blackhole_t9_page_addressed_qk_gemm_b_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    total_pages = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(1)
    q = torch.randn(
        batch,
        block_M,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2], [1]], dtype=torch.int32)
    out = torch.zeros(
        batch,
        block_M,
        heads,
        block_N,
        dtype=torch.float32,
    )
    ref = torch.empty_like(out)
    for seq in range(batch):
        page = int(page_table[seq, 0])
        k_page = k_cache[page * block_N : (page + 1) * block_N, :]
        for head in range(heads):
            ref[seq, :, head, :] = torch.matmul(q[seq, :, head, :].float(), k_page.float().T)

    kernel = paged_qk_gemm_kernel(
        batch=batch,
        heads=heads,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q, k_cache, page_table, out)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole page-addressed QK GEMM B direct runtime mismatch",
    )


def test_blackhole_t9_page_addressed_qk_gemm_b_page1_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    pages_per_sequence = 2
    page_column = 1
    total_pages = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(11)
    q = torch.randn(
        batch,
        block_M,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    out = torch.zeros(batch, block_M, heads, block_N, dtype=torch.float32)
    ref = torch.empty_like(out)
    for seq in range(batch):
        page = int(page_table[seq, page_column])
        k_page = k_cache[page * block_N : (page + 1) * block_N, :]
        for head in range(heads):
            ref[seq, :, head, :] = torch.matmul(q[seq, :, head, :].float(), k_page.float().T)

    kernel = paged_qk_gemm_kernel(
        batch=batch,
        heads=heads,
        pages_per_sequence=pages_per_sequence,
        page_column=page_column,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q, k_cache, page_table, out)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole page1-addressed QK GEMM B direct runtime mismatch",
    )


def test_blackhole_t9_page_addressed_av_gemm_b_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    total_pages = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(2)
    a = torch.randn(
        batch,
        block_M,
        heads,
        block_N,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    v_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2], [1]], dtype=torch.int32)
    out = torch.zeros(batch, block_M, heads, dim, dtype=torch.float32)
    ref = torch.empty_like(out)
    for seq in range(batch):
        page = int(page_table[seq, 0])
        v_page = v_cache[page * block_N : (page + 1) * block_N, :]
        for head in range(heads):
            ref[seq, :, head, :] = torch.matmul(a[seq, :, head, :].float(), v_page.float())

    kernel = paged_av_gemm_kernel(
        batch=batch,
        heads=heads,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](a, v_cache, page_table, out)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole page-addressed AV GEMM B direct runtime mismatch",
    )


def test_blackhole_t9_page_addressed_av_gemm_b_page1_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    pages_per_sequence = 2
    page_column = 1
    total_pages = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(12)
    a = torch.randn(
        batch,
        block_M,
        heads,
        block_N,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    v_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    out = torch.zeros(batch, block_M, heads, dim, dtype=torch.float32)
    ref = torch.empty_like(out)
    for seq in range(batch):
        page = int(page_table[seq, page_column])
        v_page = v_cache[page * block_N : (page + 1) * block_N, :]
        for head in range(heads):
            ref[seq, :, head, :] = torch.matmul(a[seq, :, head, :].float(), v_page.float())

    kernel = paged_av_gemm_kernel(
        batch=batch,
        heads=heads,
        pages_per_sequence=pages_per_sequence,
        page_column=page_column,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](a, v_cache, page_table, out)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole page1-indexed AV GEMM B direct runtime mismatch",
    )


def test_blackhole_seq64_qk_gemm_direct_runtime_layout():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 4
    seq_len = 64
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(5)
    q = torch.randn(
        batch,
        seq_len,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k = torch.randn(
        batch,
        seq_len,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    out = torch.zeros(batch, seq_len, heads, block_N, dtype=torch.float32)
    ref = torch.empty_like(out)
    for tile in range(seq_len // block_M):
        rows = slice(tile * block_M, (tile + 1) * block_M)
        cols = slice(tile * block_N, (tile + 1) * block_N)
        for head in range(heads):
            ref[0, rows, head, :] = torch.matmul(
                q[0, rows, head, :].float(),
                k[0, cols, head, :].float().T,
            )

    kernel = seq_qk_gemm_kernel(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q, k, out)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole seq64 QK GEMM layout direct runtime mismatch",
    )


def test_blackhole_t9_paged_gqa_decode_bf16_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    pages_per_sequence = 2
    total_pages = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(0)
    q = torch.randn(
        batch,
        block_M,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    v_cache = torch.randn(
        total_pages * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    cache_seq_lens = torch.tensor([45, 64], dtype=torch.int32)
    out = torch.zeros_like(q)

    kernel = paged_gqa_decode_kernel(
        batch=batch,
        heads=heads,
        groups=heads,
        pages_per_sequence=pages_per_sequence,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q, k_cache, v_cache, page_table, cache_seq_lens, out)

    ref = _paged_gqa_decode_reference(q, k_cache, v_cache, page_table, cache_seq_lens)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole T9 paged GQA decode bf16 direct runtime mismatch",
    )


def test_blackhole_t9_sparse_ragged_gqa_decode_bf16_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    sparse_blocks = 2
    total_blocks = 4
    block_M = 32
    block_N = 32
    dim = 32

    torch.manual_seed(34)
    q = torch.randn(
        batch,
        block_M,
        heads,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_blocks = torch.randn(
        total_blocks * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    v_blocks = torch.randn(
        total_blocks * block_N,
        dim,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    block_indices = torch.tensor([[3, 0], [2, 1]], dtype=torch.int32)
    valid_rows = torch.tensor([[19, 32], [32, 11]], dtype=torch.int32)
    out = torch.zeros_like(q)

    kernel = sparse_ragged_gqa_decode_kernel(
        batch=batch,
        heads=heads,
        groups=heads,
        sparse_blocks=sparse_blocks,
        total_blocks=total_blocks,
        block_M=block_M,
        block_N=block_N,
        dim=dim,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q, k_blocks, v_blocks, block_indices, valid_rows, out)

    ref = _sparse_ragged_gqa_decode_reference(
        q, k_blocks, v_blocks, block_indices, valid_rows
    )
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole T9 sparse/ragged GQA decode bf16 direct runtime mismatch",
    )


def test_blackhole_t9_paged_mla_dual_score_bf16_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 1
    total_pages = 3
    block_M = 32
    block_N = 32
    dv = 32
    dpe = 32

    torch.manual_seed(21)
    q_nope = torch.randn(
        batch,
        block_M,
        heads,
        dv,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    q_pe = torch.randn(
        batch,
        block_M,
        heads,
        dpe,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    kv_latent = torch.randn(
        total_pages * block_N,
        dv,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_pe = torch.randn(
        total_pages * block_N,
        dpe,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2]], dtype=torch.int32)
    out = torch.zeros(
        batch,
        block_M,
        heads,
        block_N,
        dtype=torch.float32,
    )

    kernel = paged_mla_dual_score_kernel(
        batch=batch,
        heads=heads,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dv=dv,
        dpe=dpe,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q_nope, q_pe, kv_latent, k_pe, page_table, out)

    page = int(page_table[0, 0])
    ref = (
        torch.matmul(
            q_nope[0, :, 0, :].float(),
            kv_latent[page * block_N : (page + 1) * block_N].float().T,
        )
        + torch.matmul(
            q_pe[0, :, 0, :].float(),
            k_pe[page * block_N : (page + 1) * block_N].float().T,
        )
    )[None, :, None, :]
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Blackhole T9 paged MLA dual-score bf16 direct runtime mismatch",
    )


def test_blackhole_t9_paged_mla_decode_bf16_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    heads = 4
    pages_per_sequence = 2
    total_pages = 4
    block_M = 32
    block_N = 32
    dv = 32
    dpe = 32

    torch.manual_seed(13)
    q_nope = torch.randn(
        batch,
        block_M,
        heads,
        dv,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    q_pe = torch.randn(
        batch,
        block_M,
        heads,
        dpe,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    kv_latent = torch.randn(
        total_pages * block_N,
        dv,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    k_pe = torch.randn(
        total_pages * block_N,
        dpe,
        dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE,
    )
    page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    cache_seq_lens = torch.tensor([45, 64], dtype=torch.int32)
    out = torch.zeros_like(q_nope)

    kernel = paged_mla_decode_kernel(
        batch=batch,
        heads=heads,
        pages_per_sequence=pages_per_sequence,
        total_pages=total_pages,
        block_M=block_M,
        block_N=block_N,
        dv=dv,
        dpe=dpe,
    )
    artifact, metadata = _lower_blackhole_flash_attention_metadata(kernel)
    assert [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](q_nope, q_pe, kv_latent, k_pe, page_table, cache_seq_lens, out)

    ref = _paged_mla_decode_reference(q_nope, q_pe, kv_latent, k_pe, page_table, cache_seq_lens)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole T9 paged MLA decode bf16 direct runtime mismatch",
    )


def test_blackhole_flash_attention_small_bf16_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 1
    seq_len = 32
    dim = 32
    is_causal = False
    block_M = 32
    block_N = 32
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    k = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    v = torch.randn(batch, seq_len, heads, dim, dtype=BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE)
    out = torch.zeros_like(q)

    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    _run_blackhole_flash_attention(kernel, q, k, v, out)

    ref = blackhole_mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=5e-2,
        rtol=5e-2,
        failure_message="Blackhole small bf16 flash-attention forward mismatch",
    )
