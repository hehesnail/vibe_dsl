import re

import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import assert_tensors_close_or_dump, check_blackhole_direct_execution_requirements
from .test_blackhole_copy_pipeline import _extract_blackhole_executable_spec


def _lower_blackhole(kernel):
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _kernel_source(executable_spec, kind):
    return str(
        next(
            kernel["source_code"]
            for kernel in executable_spec["kernels"]
            if str(kernel["kind"]) == kind
        )
    )


def t9_chunk_scan_kernel(*, batch=2, num_chunks=3, tile_m=32, tile_n=32):
    """Ordinary TIR chunk scan for the first T9.5 admitted slice."""
    assert num_chunks == 3
    dtype = T.bfloat16

    @T.prim_func
    def main(
        StateIn: T.Tensor((batch, tile_m, tile_n), dtype),
        X: T.Tensor((batch, num_chunks, tile_m, tile_n), dtype),
        Output: T.Tensor((batch, num_chunks, tile_m, tile_n), dtype),
        StateOut: T.Tensor((batch, tile_m, tile_n), dtype),
    ):
        with T.Kernel(batch, threads=128) as bx:
            state_shared = T.alloc_shared((tile_m, tile_n), dtype)
            x_shared = T.alloc_shared((tile_m, tile_n), dtype)
            state = T.alloc_fragment((tile_m, tile_n), dtype)
            x_local = T.alloc_fragment((tile_m, tile_n), dtype)

            T.copy(StateIn[bx, 0:tile_m, 0:tile_n], state_shared)
            T.copy(state_shared, state)
            for chunk in T.serial(num_chunks):
                T.copy(X[bx, chunk, 0:tile_m, 0:tile_n], x_shared)
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(tile_m, tile_n):
                    state[i, j] = state[i, j] + x_local[i, j]
                T.copy(state, Output[bx, chunk, 0:tile_m, 0:tile_n])
            T.copy(state, StateOut[bx, 0:tile_m, 0:tile_n])

    return main


def _chunk_scan_inputs(batch, num_chunks, tile_m, tile_n):
    values = torch.arange(
        batch * (num_chunks + 1) * tile_m * tile_n,
        dtype=torch.float32,
    ).reshape(batch, num_chunks + 1, tile_m, tile_n)
    state_in = ((values[:, 0].remainder(113) - 37) / 97.0).to(torch.bfloat16)
    x = ((values[:, 1:].remainder(127) - 51) / 89.0).to(torch.bfloat16)
    expected_chunks = []
    state = state_in.float()
    for chunk in range(num_chunks):
        state = state + x[:, chunk].float()
        expected_chunks.append(state.to(torch.bfloat16))
    return state_in, x, torch.stack(expected_chunks, dim=1), state.to(torch.bfloat16)


def test_blackhole_t9_chunk_scan_projects_loop_carried_state_lifecycle():
    kernel = t9_chunk_scan_kernel()
    artifact = _lower_blackhole(kernel)
    executable_spec = _extract_blackhole_executable_spec(artifact)

    assert "chunk_scan_plans" not in executable_spec
    assert "scan_plans" not in executable_spec
    assert "recurrence_plans" not in executable_spec
    assert [str(reason) for reason in executable_spec.get("direct_runtime_unsupported_reasons", [])] == []

    operation_names = {
        str(op["operation_name"])
        for kernel_spec in executable_spec["kernels"]
        for op in kernel_spec.get("compute_ops", [])
    }
    assert "add_tiles" in operation_names
    assert "scan" not in operation_names
    assert "chunk_scan" not in operation_names

    virtual_values = list(executable_spec.get("exact_cb_virtual_values", []))
    intervals = list(executable_spec.get("exact_cb_live_intervals", []))
    allocations = list(executable_spec.get("exact_cb_allocations", []))
    releases = list(executable_spec.get("exact_cb_release_events", []))

    state_values = [
        value
        for value in virtual_values
        if str(value["logical_value"]) == "state"
        and str(value["event_lifetime_kind"]) == "loop_carried"
    ]
    assert state_values
    state_names = {str(value["name"]) for value in state_values}
    assert any(
        str(interval["virtual_value"]) in state_names
        and bool(interval["loop_carried"])
        and bool(interval["live_in"])
        and bool(interval["live_out"])
        for interval in intervals
    )
    assert any(str(allocation["virtual_value"]) in state_names for allocation in allocations)
    assert any(
        str(release["allocation"]) == str(allocation["name"])
        for allocation in allocations
        if str(allocation["virtual_value"]) in state_names
        for release in releases
    )

    state_cb_ids = {
        int(allocation["physical_cb_id"])
        for allocation in allocations
        if str(allocation["virtual_value"]) in state_names
    }
    assert state_cb_ids
    writer_source = _kernel_source(executable_spec, "writer")
    output_loop = re.search(
        r"for\s*\([^;]*chunk\s*=\s*0;\s*chunk\s*<\s*3;\s*\+\+chunk\)\s*\{(?P<body>.*?)\n\s*\}",
        writer_source,
        re.DOTALL,
    )
    assert output_loop
    output_loop_body = output_loop.group("body")
    assert not any(
        f"cb_wait_front({state_cb_id}, 1);" in output_loop_body
        or f"cb_pop_front({state_cb_id}, 1);" in output_loop_body
        for state_cb_id in state_cb_ids
    )


def test_blackhole_t9_chunk_scan_bf16_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    batch = 2
    num_chunks = 3
    tile_m = 32
    tile_n = 32
    state_in, x, expected_output, expected_state = _chunk_scan_inputs(
        batch, num_chunks, tile_m, tile_n
    )
    output = torch.zeros_like(expected_output)
    state_out = torch.zeros_like(expected_state)

    kernel = t9_chunk_scan_kernel(
        batch=batch,
        num_chunks=num_chunks,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    artifact = _lower_blackhole(kernel)
    executable_spec = _extract_blackhole_executable_spec(artifact)
    assert [str(reason) for reason in executable_spec.get("direct_runtime_unsupported_reasons", [])] == []

    artifact.codegen_mod["main"](state_in, x, output, state_out)
    assert_tensors_close_or_dump(
        output,
        expected_output,
        atol=8e-2,
        rtol=8e-2,
        failure_message="Blackhole T9.5 chunk scan per-chunk output mismatch",
    )
    assert_tensors_close_or_dump(
        state_out,
        expected_state,
        atol=8e-2,
        rtol=8e-2,
        failure_message="Blackhole T9.5 chunk scan final state mismatch",
    )
