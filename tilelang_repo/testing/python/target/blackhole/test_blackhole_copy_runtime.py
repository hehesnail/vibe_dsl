import pytest
import torch

from tilelang.engine.lower import lower
from tilelang.engine.lower import merge_ir_modules
from tvm.target import Target
from tilelang import tvm

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
    grid_indexed_staged_copy_kernel,
    staged_copy_kernel,
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
        {"name": "A_addr", "kind": "input_buffer_addr32", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "dtype": "uint32", "buffer": "B"},
        {"name": "work_linear_id", "kind": "work_linear_id", "dtype": "uint32"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "dtype": "uint32"},
        {"name": "b_tile_start_id", "kind": "b_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "dtype": "uint32"},
    ]
    mutated_mod = _rebuild_direct_runtime_module_with_runtime_args(artifact, unsupported_runtime_args)

    a_torch = torch.randn(32, 32, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    with pytest.raises(Exception, match="b_tile_start_id|unsupported richer schema"):
        mutated_mod["main"](a_torch, b_output)
