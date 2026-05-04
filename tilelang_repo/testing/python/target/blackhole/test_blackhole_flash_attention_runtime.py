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
MULTI_BLOCK_EXACT_CB_REPUBLISH_REASON = "multi-block exact CB-republish"


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


def _has_multi_page_republish_event(metadata):
    return any(
        str(config["flow_class"]) == "republish"
        and (
            int(config.get("publish_pages_per_event", 0)) > 1
            or int(config.get("consume_pages_per_event", 0)) > 1
        )
        for config in metadata["cb_configs"]
    )


def _assert_t7_seq64_mha_exact_cb_partial_combine_contract(metadata):
    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert reasons == []
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
def test_blackhole_flash_attention_multi_work_item_metadata_exposes_explicit_per_work_access_descriptors(
    kernel,
):
    _, metadata = _lower_blackhole_flash_attention_metadata(
        kernel
    )

    reasons = [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]
    assert not any("missing explicit per-work access descriptor" in reason for reason in reasons)
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
