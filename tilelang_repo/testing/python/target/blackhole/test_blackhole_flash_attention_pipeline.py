import sys
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang import tvm
from tvm.target import Target

from .common import check_blackhole_codegen_requirements


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example


def test_flash_attention_forward_rejects_unlowered_fragment_subset():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole fragment compute subset lowering is not implemented",
    ):
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
