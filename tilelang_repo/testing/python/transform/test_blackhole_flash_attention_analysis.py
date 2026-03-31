import sys
from pathlib import Path

import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from tvm.target import Target


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_gqa_fwd_bshd as gqa_example
import example_mha_fwd_bshd as mha_example


def _lower_to_blackhole_legalized_prim_func(prim_func):
    mod = tvm.IRModule({"main": prim_func})
    target = Target("blackhole")
    with target:
        mod = LowerAndLegalize(mod, target)
    return mod["main"]


def _lower_flash_attention_example(example_module, *args, **kwargs):
    return _lower_to_blackhole_legalized_prim_func(
        example_module.flashattn.jit_impl.get_tir(*args, **kwargs)
    )


def _analyze_blackhole_work_decomposition(prim_func):
    mod = tvm.IRModule({"main": prim_func})
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    return mod["main"]


def _analyze_blackhole_fragment_regions(prim_func):
    mod = tvm.IRModule({"main": prim_func})
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    return mod["main"]


def _analyze_blackhole_pipeline_stages(prim_func):
    mod = tvm.IRModule({"main": prim_func})
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    return mod["main"]


def test_mha_forward_exposes_work_decomposition_attrs():
    lowered = _analyze_blackhole_work_decomposition(
        _lower_flash_attention_example(
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
        )
    )
    work_info = lowered.attrs["blackhole.work_decomposition"]
    assert list(work_info["axes"]) == ["bx", "by", "bz"]

    derived_index_exprs = work_info["derived_index_exprs"]
    assert len(derived_index_exprs) > 0
    assert any(
        "expr" in entry
        and isinstance(entry["expr"], tvm.tir.PrimExpr)
        and not isinstance(entry["expr"], str)
        and str(entry["expr"]) == "bx * 128"
        for entry in derived_index_exprs
    )

    causal_lowered = _analyze_blackhole_work_decomposition(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            True,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    causal_work_info = causal_lowered.attrs["blackhole.work_decomposition"]
    loop_bounds = causal_work_info["work_dependent_loop_bounds"]
    assert len(loop_bounds) > 0
    k_loop_bounds = [bound for bound in loop_bounds if str(bound["loop_var"]) == "k"]
    assert len(k_loop_bounds) > 0
    k_loop_bound = k_loop_bounds[0]
    assert set(k_loop_bound.keys()) >= {"loop_var", "min", "extent"}
    assert isinstance(k_loop_bound["min"], tvm.tir.PrimExpr)
    assert isinstance(k_loop_bound["extent"], tvm.tir.PrimExpr)
    assert not isinstance(k_loop_bound["min"], str)
    assert not isinstance(k_loop_bound["extent"], str)
    assert str(k_loop_bound["min"]) == "0"
    assert str(k_loop_bound["extent"]) == "bx + 1"


def test_gqa_forward_exposes_fragment_region_attrs():
    lowered = _analyze_blackhole_fragment_regions(
        _lower_flash_attention_example(
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
        )
    )
    regions = lowered.attrs["blackhole.fragment_regions"]
    assert len(regions) == 1

    region = regions[0]
    fragment_buffer_names = {entry["name"] for entry in region["fragment_buffers"]}
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(fragment_buffer_names)

    assert {
        "gemm",
        "row_reduction",
        "row_broadcast",
        "pointwise_chain",
    }.issubset(set(region["ops"]))

    row_reduction_targets = {entry["target"] for entry in region["row_reductions"]}
    assert {"scores_max", "scores_sum"}.issubset(row_reduction_targets)

    row_broadcast_sources = {entry["source"] for entry in region["row_broadcasts"]}
    assert {"scores_max", "scores_scale", "logsum"}.issubset(row_broadcast_sources)

    loop_carried_state = {entry["name"] for entry in region["loop_carried_state"]}
    assert {"scores_max", "logsum", "acc_o"}.issubset(loop_carried_state)


def test_gqa_forward_wider_pipeline_still_exposes_row_broadcast_roles():
    lowered = _analyze_blackhole_fragment_regions(
        _lower_flash_attention_example(
            gqa_example,
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
        )
    )
    regions = lowered.attrs["blackhole.fragment_regions"]
    assert len(regions) == 1

    region = regions[0]
    assert "row_broadcast" in set(region["ops"])
    row_broadcast_sources = {entry["source"] for entry in region["row_broadcasts"]}
    assert {"scores_max", "scores_scale", "logsum"}.issubset(row_broadcast_sources)


def test_mha_forward_exposes_fragment_region_roles():
    lowered = _analyze_blackhole_fragment_regions(
        _lower_flash_attention_example(
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
        )
    )
    regions = lowered.attrs["blackhole.fragment_regions"]
    assert len(regions) == 1

    region = regions[0]
    fragment_buffer_names = {entry["name"] for entry in region["fragment_buffers"]}
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(fragment_buffer_names)

    assert {
        "gemm",
        "row_reduction",
        "row_broadcast",
        "pointwise_chain",
    }.issubset(set(region["ops"]))

    row_reduction_targets = {entry["target"] for entry in region["row_reductions"]}
    assert {"scores_max", "scores_sum"}.issubset(row_reduction_targets)

    row_broadcast_sources = {entry["source"] for entry in region["row_broadcasts"]}
    assert {"scores_max", "scores_scale", "logsum"}.issubset(row_broadcast_sources)

    loop_carried_state = {entry["name"] for entry in region["loop_carried_state"]}
    assert {"scores_max", "logsum", "acc_o"}.issubset(loop_carried_state)


def test_forward_pipeline_exposes_stage_attrs():
    lowered = _analyze_blackhole_pipeline_stages(
        _lower_flash_attention_example(
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
        )
    )
    stages = lowered.attrs["blackhole.pipeline_stages"]
    assert len(stages) == 1

    stage = stages[0]
    assert stage["num_stages"] == 2

    stage_local_buffers = {entry["name"] for entry in stage["stage_local_buffers"]}
    assert {"K_shared", "V_shared"}.issubset(stage_local_buffers)

    loop_carried_state = {entry["name"] for entry in stage["loop_carried_state"]}
    assert {"acc_o", "scores_max", "logsum"}.issubset(loop_carried_state)
    assert stage["loop_var"] == "k"
