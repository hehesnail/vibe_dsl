import sys
import types
import re
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
MULTI_PAGE_EXACT_CB_REPUBLISH_REASON = "multi-page exact CB-republish live-form"


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


def _has_multi_page_republish_input_cb(metadata):
    return any(
        str(config["role"]) == "input"
        and str(config["flow_class"]) == "republish"
        and int(config["num_pages"]) > 1
        for config in metadata["cb_configs"]
    )


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
        == "cast_fragment_slice_to_tiled_cb"
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
        for cb_id in re.findall(r"\b(?:pack_reconfig_data_format|pack_tile)\([^;\n]*?(\d+)", compute_source)
    ]
    assert pack_cb_ids
    assert max(pack_cb_ids) <= 31


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

    final_publish_matches = list(re.finditer(
        rf"cb_wait_front\((\d+), 1\);"
        rf"\ncb_reserve_back\({output_cb}, 1\);"
        rf".*?copy_tile_to_dst_init_short(?:_with_dt)?\([^)]*?(\d+)\);"
        rf"\ncopy_tile\((\d+), 0, 0\);"
        rf".*?pack_tile\(0, {output_cb}, 0\);",
        compute_source,
        flags=re.DOTALL,
    ))
    final_publish = final_publish_matches[-1] if final_publish_matches else None
    assert final_publish is not None
    waited_cb, init_cb, copied_cb = map(int, final_publish.groups())
    assert waited_cb == init_cb == copied_cb

    cb_configs = {int(config["cb_id"]): config for config in metadata["cb_configs"]}
    source_config = cb_configs[copied_cb]
    assert "row_bcast_out" in str(source_config["name"])


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
        r".*?pack_reconfig_data_format\((\d+)\);"
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
def test_blackhole_flash_attention_multi_work_item_metadata_exposes_explicit_per_work_access_descriptors(
    kernel,
):
    _, metadata = _lower_blackhole_flash_attention_metadata(
        kernel
    )

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any("missing explicit per-work access descriptor" in reason for reason in reasons)
    if _has_multi_page_republish_input_cb(metadata):
        assert any(MULTI_PAGE_EXACT_CB_REPUBLISH_REASON in reason for reason in reasons)
    else:
        assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)
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
    assert all(str(spec["descriptor_kind"]) for spec in reader_specs + writer_specs)
    assert all(str(spec["value_source"]) for spec in reader_specs + writer_specs)
    assert all(str(spec["arg_identity"]) for spec in reader_specs + writer_specs)

    reader_start_sources = {
        str(spec["value_source"])
        for spec in reader_specs
        if str(spec["descriptor_kind"]) == "tile_start"
    }
    assert reader_start_sources & {"logical_block_y", "work_linear_id"}
    assert any(
        str(spec["descriptor_kind"]) == "tile_start"
        and str(spec["value_source"]) == "work_linear_id"
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
