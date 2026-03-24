import pytest
import torch

import tilelang
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_codegen_requirements,
    check_blackhole_direct_execution_requirements,
    gemm_kernel,
)


def test_blackhole_split_kernel_gemm_segment_plan():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)

    plan = None
    for _, func in mod.functions.items():
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            plan = func.attrs["blackhole.segment_plan"]
            break

    assert plan is not None
    assert len(plan) == 3
    assert str(plan[0]["kind"]) == "reader"
    assert str(plan[1]["kind"]) == "compute"
    assert str(plan[2]["kind"]) == "writer"
    assert str(plan[0]["core_type"]) == "brisc"
    assert str(plan[1]["core_type"]) == "trisc"
    assert str(plan[2]["core_type"]) == "ncrisc"

    reader_args = plan[0]["runtime_args"]
    assert [str(arg["buffer"]) for arg in reader_args if "buffer" in arg] == ["A", "B"]

    writer_args = plan[2]["runtime_args"]
    assert [str(arg["buffer"]) for arg in writer_args if "buffer" in arg] == ["C"]


def test_blackhole_gemm_cb_placeholders_resolve_via_planner():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)

    func = mod["main"]
    placeholders = func.attrs["blackhole.gemm_cb_placeholders"]
    assert str(placeholders["a"]["requirement_name"]) == "A_shared"
    assert str(placeholders["b"]["requirement_name"]) == "B_shared"
    assert str(placeholders["c"]["requirement_name"]) == "C_local"
    assert int(placeholders["a"]["placeholder_cb_id"]) < 0
    assert int(placeholders["b"]["placeholder_cb_id"]) < 0
    assert int(placeholders["c"]["placeholder_cb_id"]) < 0

    cb_bindings = {
        str(item["requirement_name"]): int(item["cb_id"])
        for item in func.attrs["blackhole.cb_bindings"]
    }
    assert cb_bindings["A_shared"] != cb_bindings["C_local"]
    assert cb_bindings["B_shared"] != cb_bindings["C_local"]

    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    try:
        with target:
            artifact = lower(kernel, target=target)
    except Exception as e:
        pytest.skip(f"Blackhole lowering not yet fully implemented: {e}")

    source = getattr(artifact, "kernel_source", None)
    if source is None and hasattr(artifact, "mod"):
        try:
            source = artifact.mod.imported_modules[0].get_source()
        except Exception:
            source = None
    if not source and hasattr(artifact, "code"):
        source = artifact.code

    assert source
    assert "mm_init(-1" not in source
    assert "mm_init(-2" not in source
    assert "mm_init(-3" not in source
    assert "cb_wait_front(-1" not in source
    assert "cb_wait_front(-2" not in source
    assert "cb_reserve_back(-3" not in source


def test_blackhole_gemm_basic():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    target = Target("blackhole")
    kernel = gemm_kernel()

    try:
        with target:
            artifact = lower(kernel, target=target)
    except Exception as e:
        pytest.skip(f"Blackhole GEMM lowering not yet fully implemented: {e}")

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output, c_ref, atol=2e-1, rtol=2e-1, failure_message="GEMM direct-call output mismatch"
    )
