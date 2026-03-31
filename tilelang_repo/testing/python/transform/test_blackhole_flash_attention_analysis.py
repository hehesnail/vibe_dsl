import sys
from pathlib import Path

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


def test_mha_forward_exposes_work_decomposition_attrs():
    lowered = _lower_flash_attention_example(
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
    assert lowered.attrs.get("blackhole.work_decomposition") is not None


def test_gqa_forward_exposes_fragment_region_attrs():
    lowered = _lower_flash_attention_example(
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
    assert lowered.attrs.get("blackhole.fragment_regions") is not None


def test_forward_pipeline_exposes_stage_attrs():
    lowered = _lower_flash_attention_example(
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
    assert lowered.attrs.get("blackhole.pipeline_stages") is not None
