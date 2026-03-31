import sys
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang.engine.phase import LowerAndLegalize
from tilelang.engine.phase import OptimizeForTarget
from tilelang import tvm
from tvm.target import Target

from .common import check_blackhole_codegen_requirements


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example
import example_gqa_fwd_bshd as gqa_example


def _lower_flash_attention_through_blackhole_ops(*, is_causal=False):
    target = Target("blackhole")
    mod = tvm.IRModule(
        {
            "main": mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                is_causal,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            )
        }
    )
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    return mod["main"]


def _run_flash_attention_lower_blackhole_ops(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    return tilelang.transform.LowerBlackholeOps()(mod)


def _run_flash_attention_lower_blackhole_ops_after_optimize(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    return tilelang.transform.LowerBlackholeOps()(mod)


def test_flash_attention_forward_lower_blackhole_ops_emits_generic_lowering_requirements():
    lowered = _lower_flash_attention_through_blackhole_ops()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs

    lowering_requirements = attrs["blackhole.lowering_requirements"]
    assert list(lowering_requirements["work_axes"]) == ["bx", "by", "bz"]
    assert {
        "gemm",
        "pointwise_chain",
    }.issubset(set(lowering_requirements["fragment_op_kinds"]))
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])
    assert "row_broadcast_sources" not in lowering_requirements
    assert {"exp2", "mul", "div"}.issubset(
        set(lowering_requirements["pointwise_op_kinds"])
    )
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "add" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_scores_max_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.scalar_max" in script
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_lower_blackhole_ops_lowers_row_reductions_to_builtins():
    lowered = _lower_flash_attention_through_blackhole_ops()
    script = lowered.script()

    assert "tl.blackhole.reduce_row" in script
    assert "\"max\"" in script
    assert "\"sum\"" in script


def test_flash_attention_forward_optimized_path_lowers_row_reductions_to_builtins():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.reduce_row" in script
    assert "row_reduction" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_acc_o_row_broadcast_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.mul_row_bcast" in script
    assert "tl.blackhole.div_row_bcast" in script


def test_flash_attention_forward_optimized_path_lowers_logsum_scalar_fma():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.scalar_fma" in script


def test_flash_attention_forward_optimized_path_lowers_scores_exp2_affine_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.exp2_row_bcast_affine" in script
    assert "tl.blackhole.scalar_exp2_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_fills():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.fill_fragment" in script
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_casts():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.cast_fragment_slice" in script
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_local_to_cb_staging():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.write_local_slice_to_cb" in script
    assert "O_shared_1[tx" not in script


def test_flash_attention_gqa_optimized_path_lowers_grouped_row_broadcasts():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        gqa_example,
        1,
        16,
        1024,
        128,
        False,
        groups=16,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.mul_grouped_row_bcast" in script
    assert "tl.blackhole.div_grouped_row_bcast" in script
    assert "tl.blackhole.exp2_grouped_row_bcast_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_gqa_reader_runtime_args_cover_all_accessor_buffers():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        gqa_example,
        1,
        16,
        1024,
        128,
        False,
        groups=16,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )["main"]

    reader_segments = [seg for seg in lowered.attrs["blackhole.segment_plan"] if seg["kind"] == "reader"]
    assert len(reader_segments) == 1
    reader = reader_segments[0]

    accessor_buffers = [acc["buffer"] for acc in reader["accessors"]]
    runtime_arg_buffers = [
        arg["buffer"]
        for arg in reader["runtime_args"]
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert accessor_buffers == ["Q", "K", "V"]
    assert runtime_arg_buffers == ["Q", "K", "V"]


def test_flash_attention_gqa_top_level_runtime_args_aggregate_segment_buffers():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        gqa_example,
        1,
        16,
        1024,
        128,
        False,
        groups=16,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )["main"]

    assert "blackhole.runtime_args" in lowered.attrs

    runtime_arg_buffers = [
        arg["buffer"]
        for arg in lowered.attrs["blackhole.runtime_args"]
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert runtime_arg_buffers[:3] == ["Q", "K", "V"]


def test_flash_attention_forward_lowers_mha_pipeline_end_to_end():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_lowers_gqa_pipeline_for_supported_stage_count():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                1024,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_rejects_unsupported_pipeline_stage_count():
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole fragment pipeline legality: unsupported stage count 4",
    ):
        target = Target("blackhole")
        with target:
            lower(
                gqa_example.flashattn.jit_impl.get_tir(
                    1,
                    16,
                    1024,
                    128,
                    False,
                    groups=16,
                    block_M=64,
                    block_N=64,
                    num_stages=4,
                    threads=128,
                ),
                target=target,
            )
