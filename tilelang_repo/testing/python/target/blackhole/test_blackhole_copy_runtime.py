import pytest
import torch

from tilelang.engine.lower import lower
from tilelang.engine.lower import merge_ir_modules
from tvm.target import Target
from tilelang import tvm
from tvm import tir
from tvm.tir import stmt_functor

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
    grid_indexed_staged_copy_kernel,
    staged_copy_kernel,
    staged_stick_copy_kernel,
)


def _rebuild_direct_runtime_module_with_runtime_args(artifact, runtime_args):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.runtime_args" in func.attrs:
            func = func.with_attr("blackhole.runtime_args", runtime_args)
        rewritten[gvar] = func
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    target = Target("blackhole")
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_direct_runtime_module_with_body_and_attrs(
    artifact, *, body_mutator=None, semaphore_plan=None, runtime_args=None
):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            if body_mutator is not None:
                func = func.with_body(body_mutator(func.body))
            if semaphore_plan is not None:
                func = func.with_attr("blackhole.semaphore_plan", semaphore_plan)
            if runtime_args is not None:
                func = func.with_attr("blackhole.runtime_args", runtime_args)
        rewritten[gvar] = func
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    target = Target("blackhole")
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_direct_runtime_module_with_core_plan(artifact, core_plan):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.core_plan" in func.attrs:
            func = func.with_attr("blackhole.core_plan", core_plan)
        rewritten[gvar] = func
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    target = Target("blackhole")
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _inject_worker_semaphore_handshake(remote_core_x, remote_core_y):
    def mutate(original_body):
        def postorder(node):
            if not isinstance(node, tir.SeqStmt) or len(node.seq) != 6:
                return node
            try:
                read_call = node.seq[1].value
                write_call = node.seq[4].value
            except AttributeError:
                return node
            if read_call.op.name != "tl.blackhole.read_tile_to_cb":
                return node
            if write_call.op.name != "tl.blackhole.write_tile_from_cb":
                return node

            bx = read_call.args[1]
            consumer_wait_semaphore_addr = tir.Var("copy_sem_addr_consumer_wait", "uint32")
            producer_remote_semaphore_addr = tir.Var("copy_sem_addr_consumer_remote", "uint32")
            remote_noc_x = tir.call_intrin(
                "uint32",
                tir.op.Op.get("tl.blackhole.runtime_arg_u32"),
                tir.StringImm("remote_noc_x"),
            )
            remote_noc_y = tir.call_intrin(
                "uint32",
                tir.op.Op.get("tl.blackhole.runtime_arg_u32"),
                tir.StringImm("remote_noc_y"),
            )
            consumer_semaphore_id = tir.call_intrin(
                "uint32", tir.op.Op.get("tl.blackhole.get_semaphore"), tir.IntImm("uint32", 0)
            )
            semaphore_wait = tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_wait"),
                    consumer_wait_semaphore_addr,
                    tir.IntImm("uint32", 1),
                )
            )
            semaphore_inc_remote = tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_inc_remote"),
                    producer_remote_semaphore_addr,
                    remote_noc_x,
                    remote_noc_y,
                    tir.IntImm("uint32", 1),
                )
            )
            consumer_wait = tir.IfThenElse(
                bx == tir.IntImm("int32", 1),
                tir.LetStmt(consumer_wait_semaphore_addr, consumer_semaphore_id, semaphore_wait),
                tir.Evaluate(0),
            )
            producer_set = tir.IfThenElse(
                bx == tir.IntImm("int32", 0),
                tir.LetStmt(producer_remote_semaphore_addr, consumer_semaphore_id, semaphore_inc_remote),
                tir.Evaluate(0),
            )
            return tir.SeqStmt([consumer_wait, node, producer_set])

        return stmt_functor.ir_transform(original_body, None, postorder, ["tir.SeqStmt"])

    return mutate


def test_blackhole_module_direct_call():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=m // 32, tile_cols=n // 32)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output, b_ref, atol=1e-3, rtol=1e-3, failure_message="Direct-call output mismatch"
    )


def test_blackhole_module_direct_call_rectangular_tiles():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 64
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1, tile_m=32, tile_n=64)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Rectangular direct-call output mismatch",
    )


def test_blackhole_module_direct_call_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, :16] = a_torch[:, :16]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=32, tile_n=16, global_n=32, dtype="float32")
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_tall_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, :16] = a_torch[:, :16]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=64, tile_n=16, global_n=32, dtype="float32")
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Tall stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_offset_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 48
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, 16:32] = a_torch[:, 16:32]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=48, dtype="float32", src_col=16, dst_col=16
    )
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Offset stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 2, 3
    m, n = grid_y * 32, grid_x * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]
    assert int(core_plan["logical_grid_x"]) == grid_x
    assert int(core_plan["logical_grid_y"]) == grid_y
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Grid-indexed direct-call output mismatch",
    )


def test_blackhole_module_direct_call_grid_indexed_copy_worker_semaphore_handshake():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 2, 1
    m, n = grid_y * 32, grid_x * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]
    producer_core = core_plan["physical_cores"][0]
    consumer_core = core_plan["physical_cores"][1]
    assert int(producer_core["core_x"]) == 1
    assert int(producer_core["core_y"]) == 2
    assert int(consumer_core["core_x"]) == 2
    assert int(consumer_core["core_y"]) == 2

    runtime_args = list(device_main.attrs["blackhole.runtime_args"])
    runtime_args.extend(
        [
            {
                "name": "remote_noc_x",
                "kind": "logical_core_noc_x",
                "dtype": "uint32",
                "core_x": int(consumer_core["core_x"]),
                "core_y": int(consumer_core["core_y"]),
            },
            {
                "name": "remote_noc_y",
                "kind": "logical_core_noc_y",
                "dtype": "uint32",
                "core_x": int(consumer_core["core_x"]),
                "core_y": int(consumer_core["core_y"]),
            },
        ]
    )

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 2, "core_y": 2},
                }
            ],
        }
        ,
    ]
    mutated_mod = _rebuild_direct_runtime_module_with_body_and_attrs(
        artifact,
        body_mutator=_inject_worker_semaphore_handshake(
            consumer_core["core_x"], consumer_core["core_y"]
        ),
        semaphore_plan=semaphore_plan,
        runtime_args=runtime_args,
    )

    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Worker semaphore producer-consumer copy mismatch",
    )


def test_blackhole_module_direct_call_rejects_oversubscribed_multi_core_launch():
    kernel = grid_indexed_staged_copy_kernel(grid_x=15, grid_y=10)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]
    assert int(core_plan["logical_grid_x"]) == 15
    assert int(core_plan["logical_grid_y"]) == 10
    assert int(device_main.attrs["blackhole.work_per_core"]) == 2

    m, n = 10 * 32, 15 * 32
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(Exception, match="oversubscribed direct launch is not supported"):
        artifact.codegen_mod["main"](a_torch, b_output)


def test_blackhole_module_direct_call_rejects_empty_work_packets_at_build_time():
    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = dict(device_main.attrs["blackhole.core_plan"])
    assert list(core_plan["work_packets"])
    core_plan["work_packets"] = []

    with pytest.raises(tvm.error.InternalError, match="core_plan.work_packets|planner/runtime"):
        _rebuild_direct_runtime_module_with_core_plan(artifact, core_plan)


def test_blackhole_large_shape_copy_keeps_per_core_l1_small():
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    assert int(device_main.attrs["blackhole.total_l1_bytes"]) == 4096
    cb_configs = device_main.attrs["blackhole.cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 2
    assert int(cb_configs[0]["total_size_bytes"]) == 4096
    assert int(cb_configs[0]["lifetime_begin"]) == 0
    assert int(cb_configs[0]["lifetime_end"]) == 0


def test_blackhole_module_direct_call_large_shape_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 25 * 32, 32 * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Large-shape direct-call output mismatch",
    )


def test_blackhole_module_direct_call_rejects_unsupported_richer_copy_schema():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    with target:
        artifact = lower(kernel, target=target)

    unsupported_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "identity": "input_buffer_addr32:A", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "identity": "output_buffer_addr32:B", "dtype": "uint32", "buffer": "B"},
        {"name": "work_linear_id", "kind": "work_linear_id", "identity": "work_linear_id", "dtype": "uint32"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "identity": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "identity": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "identity": "a_tile_stride", "dtype": "uint32"},
        {"name": "b_tile_start_id", "kind": "b_tile_start_id", "identity": "b_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "identity": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "identity": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "identity": "output_tile_stride", "dtype": "uint32"},
    ]
    mutated_mod = _rebuild_direct_runtime_module_with_runtime_args(artifact, unsupported_runtime_args)

    a_torch = torch.randn(32, 32, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    with pytest.raises(Exception, match="b_tile_start_id|unsupported richer schema"):
        mutated_mod["main"](a_torch, b_output)
