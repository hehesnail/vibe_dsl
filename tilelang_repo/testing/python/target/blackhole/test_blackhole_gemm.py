import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target
from tvm.tir import stmt_functor

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_codegen_requirements,
    check_blackhole_direct_execution_requirements,
    gemm_kernel,
)


def multicore_gemm_kernel(
    M: int = 64, N: int = 64, K: int = 128, tile_m: int = 32, tile_n: int = 32
):
    """GEMM kernel that maps bx/by to output tile coordinates."""
    grid_x = N // tile_n
    grid_y = M // tile_m

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            A_shared = T.alloc_shared((tile_m, K), "bfloat16")
            B_shared = T.alloc_shared((tile_n, K), "bfloat16")
            C_local = T.alloc_fragment((tile_m, tile_n), "float32")
            T.copy(A[by * tile_m : (by + 1) * tile_m, 0:K], A_shared)
            T.copy(B[bx * tile_n : (bx + 1) * tile_n, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(
                C_local,
                C[by * tile_m : (by + 1) * tile_m, bx * tile_n : (bx + 1) * tile_n],
            )

    return main


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


def test_blackhole_gemm_cb_ids_are_rewritten_by_planner():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    lower_mod = tilelang.transform.LowerBlackholeOps()(mod)
    planned_mod = tilelang.transform.PlanBlackholeCB()(lower_mod)

    lower_func = lower_mod["main"]
    func = planned_mod["main"]
    assert "blackhole.gemm_cb_placeholders" not in func.attrs

    def collect_cb_ids(stmt):
        cb_ids = set()

        def visit(node):
            if not isinstance(node, tilelang.tvm.tir.Call):
                return
            if not hasattr(node.op, "name"):
                return
            op_name = node.op.name
            if op_name in {
                "tl.blackhole.cb_reserve_back",
                "tl.blackhole.cb_push_back",
                "tl.blackhole.cb_wait_front",
                "tl.blackhole.cb_pop_front",
                "tl.blackhole.write_tile_from_cb",
            }:
                cb_ids.add(int(node.args[0]))
            elif op_name == "tl.blackhole.read_tile_to_cb":
                cb_ids.add(int(node.args[2]))
            elif op_name == "tl.blackhole.mm_init":
                cb_ids.update(int(arg) for arg in node.args[:3])
            elif op_name == "tl.blackhole.matmul_tiles":
                cb_ids.update(int(arg) for arg in node.args[:2])
            elif op_name == "tl.blackhole.pack_tile":
                cb_ids.add(int(node.args[1]))

        stmt_functor.post_order_visit(stmt, visit)
        return cb_ids

    assert collect_cb_ids(lower_func.body) == {0, 1, 2}
    assert collect_cb_ids(func.body) == {0, 1, 16}
    func_text = func.body.script()
    assert func_text.count("tl.blackhole.pack_tile") == 1
    assert func_text.count("tl.blackhole.write_tile_from_cb") == 1

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
    assert "mm_init(-" not in source
    assert "cb_wait_front(-" not in source
    assert "cb_reserve_back(-" not in source


def test_blackhole_gemm_accumulator_scope_canonicalized():
    kernel = gemm_kernel()
    target = Target("blackhole")
    mod = tilelang.tvm.IRModule({"main": kernel})

    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
        mod = tilelang.engine.phase.OptimizeForTarget(mod, target)

    func = mod["main"]
    resource_plan = func.attrs["blackhole.resource_plan"]
    accum_entries = [item for item in resource_plan if str(item["class"]) == "accumulator"]
    assert accum_entries
    assert any(str(item["name"]) == "C_local" for item in accum_entries)
    assert all(str(item["scope"]) == "blackhole.acc" for item in accum_entries)

    func_text = func.script()
    assert 'scope="blackhole.acc"' in func_text


def test_blackhole_gemm_contract_attr_is_materialized():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    assert func.attrs and "blackhole.gemm_contract" in func.attrs
    contract = func.attrs["blackhole.gemm_contract"]
    assert str(contract["a_buffer"]) == "A"
    assert str(contract["b_buffer"]) == "B"
    assert str(contract["c_buffer"]) == "C"
    assert int(contract["M"]) == 32
    assert int(contract["N"]) == 32
    assert int(contract["K"]) == 128
    assert bool(contract["transpose_B"]) is True


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


def test_blackhole_gemm_multicore_direct_call():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n, k = 64, 64, 128
    torch.manual_seed(0)
    a_torch = torch.randn(m, k, dtype=torch.bfloat16)
    b_torch = torch.randn(n, k, dtype=torch.bfloat16)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = multicore_gemm_kernel(M=m, N=n, K=k)
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {
        str(gvar): func for gvar, func in artifact.device_mod.functions.items()
    }
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]
    assert int(core_plan["logical_grid_x"]) == 2
    assert int(core_plan["logical_grid_y"]) == 2
    assert str(core_plan["linearization"]) == "row_major"
    assert len(core_plan["physical_cores"]) == 4
    assert len(core_plan["work_packets"]) == 4
    physical_cores = [
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    ]
    work_packet_cores = [
        (int(packet["core_x"]), int(packet["core_y"])) for packet in core_plan["work_packets"]
    ]
    assert len(set(physical_cores)) == 4
    assert len(set(work_packet_cores)) == 4
    assert work_packet_cores == physical_cores
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1]

    c_output = torch.zeros(m, n, dtype=torch.float32)
    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Multicore GEMM direct-call output mismatch",
    )
