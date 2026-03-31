import sys
from pathlib import Path

import pytest
import torch

from tilelang.engine.lower import lower
from tvm.target import Target

from .common import assert_tensors_close_or_dump, check_blackhole_direct_execution_requirements


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_gqa_fwd_bshd as gqa_example
import example_mha_fwd_bshd as mha_example


def _run_blackhole_flash_attention(kernel, *inputs):
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)
    artifact.codegen_mod["main"](*inputs)


def test_blackhole_flash_attention_runtime_gate_is_queryable():
    can_run, msg = check_blackhole_direct_execution_requirements()
    assert isinstance(can_run, bool)
    assert isinstance(msg, str)


def test_blackhole_flash_attention_mha_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 32
    seq_len = 128
    dim = 128
    is_causal = False
    block_M = 128
    block_N = 128
    num_stages = 1
    threads = 128

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
    v = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
    out = torch.zeros_like(q)

    kernel = mha_example.flashattn.jit_impl.get_tir(
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

    ref = mha_example.ref_program(q, k, v, is_causal=is_causal).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
        failure_message="Blackhole MHA flash-attention forward mismatch",
    )


def test_blackhole_flash_attention_gqa_forward_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 1
    heads = 16
    seq_len = 128
    dim = 128
    is_causal = False
    groups = 16
    block_M = 64
    block_N = 64
    num_stages = 2
    threads = 128

    head_kv = heads // groups
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
    k = torch.randn(batch, seq_len, head_kv, dim, dtype=torch.float16)
    v = torch.randn(batch, seq_len, head_kv, dim, dtype=torch.float16)
    out = torch.zeros_like(q)

    kernel = gqa_example.flashattn.jit_impl.get_tir(
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

    ref = gqa_example.ref_program(q, k, v, is_causal=is_causal, groups=groups).to(dtype=out.dtype)
    assert_tensors_close_or_dump(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
        failure_message="Blackhole GQA flash-attention forward mismatch",
    )
