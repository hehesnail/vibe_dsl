import sys
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang.engine.phase import LowerAndLegalize
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


def test_flash_attention_forward_lower_blackhole_ops_emits_generic_lowering_requirements():
    lowered = _lower_flash_attention_through_blackhole_ops()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs

    lowering_requirements = attrs["blackhole.lowering_requirements"]
    assert list(lowering_requirements["work_axes"]) == ["bx", "by", "bz"]
    assert {
        "gemm",
        "row_reduction",
        "pointwise_chain",
        "row_broadcast",
    }.issubset(set(lowering_requirements["fragment_op_kinds"]))
    assert list(lowering_requirements["row_reduction_targets"]) == [
        "scores_max",
        "scores_sum",
    ]
    assert list(lowering_requirements["row_broadcast_sources"]) == [
        "scores_max",
        "scores_scale",
        "logsum",
    ]
    assert {"fill", "exp2", "mul", "div", "max", "add", "cast"}.issubset(
        set(lowering_requirements["pointwise_op_kinds"])
    )
    assert list(lowering_requirements["pipeline_stage_counts"]) == [1]
    assert list(lowering_requirements["pipeline_loop_vars"]) == ["k"]


def test_flash_attention_forward_rejects_unlowered_fragment_subset():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with pytest.raises(tvm.TVMError) as excinfo:
        with target:
            lower(
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
    message = str(excinfo.value)
    assert "Blackhole fragment compute subset lowering is not implemented" in message
    assert "pointwise_chain" not in message
    assert "row_reduction" in message
    assert "row_broadcast" in message
    assert "fill" in message
    assert "mul" in message


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
