import sys
import types
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


BLACKHOLE_FLASH_ATTENTION_DTYPE_EXPR = "T.bfloat16"
BLACKHOLE_FLASH_ATTENTION_TORCH_DTYPE = torch.bfloat16


def _load_flash_attention_module_with_dtype(module_path, dtype_expr=BLACKHOLE_FLASH_ATTENTION_DTYPE_EXPR):
    source = Path(module_path).read_text()
    source = source.replace("dtype = T.float16", f"dtype = {dtype_expr}", 1)
    mutated = types.ModuleType(f"{Path(module_path).stem}_{dtype_expr.replace('.', '_')}")
    mutated.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), mutated.__dict__)
    return mutated


blackhole_gqa_example = _load_flash_attention_module_with_dtype(gqa_example.__file__)
blackhole_mha_example = _load_flash_attention_module_with_dtype(mha_example.__file__)


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


def test_blackhole_flash_attention_single_work_item_runtime_metadata_uses_typed_materialization_gate():
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
    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert any("thread-distributed cb_republish materialization" in reason for reason in reasons)
    assert "compute_epilogue_ops" not in metadata
    materialization_plans = {
        str(plan["target_buffer"]): plan for plan in metadata["materialization_plans"]
    }
    assert str(materialization_plans["acc_s_cast"]["materialization_protocol"]) == "cb_republish"
    assert str(materialization_plans["acc_s_cast"]["publication_protocol"]) == "mailbox_write_ptr"


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
def test_blackhole_flash_attention_multi_work_item_metadata_exposes_explicit_per_work_access_descriptors(
    kernel,
):
    _, metadata = _lower_blackhole_flash_attention_metadata(
        kernel
    )

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any("missing explicit per-work access descriptor" in reason for reason in reasons)
    assert any("thread-distributed cb_republish materialization" in reason for reason in reasons)

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
    assert all(str(spec["value_kind"]) for spec in reader_specs + writer_specs)

    reader_start_specs = {
        str(spec["arg_kind"]): str(spec["value_kind"])
        for spec in reader_specs
        if str(spec["arg_kind"]).endswith("_tile_start_id")
    }
    assert "a_tile_start_id" in reader_start_specs
    assert reader_start_specs["a_tile_start_id"] in {"logical_block_y", "current_work_linear_id"}
    assert any(
        str(spec["arg_kind"]) == "output_tile_start_id"
        and str(spec["value_kind"]) == "current_work_linear_id"
        for spec in reader_specs + writer_specs
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
