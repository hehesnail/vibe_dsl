import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
    extract_blackhole_work_per_core,
)
from .test_blackhole_copy_pipeline import _extract_blackhole_executable_spec


def _lower_blackhole(kernel):
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _require_device_main(artifact):
    for func in artifact.device_mod.functions.values():
        if getattr(func, "attrs", None) and "tl.tt_program" in func.attrs:
            return func
    pytest.fail("Expected artifact device module to carry tl.tt_program")


def _bf16_pattern(m, n, *, scale=41.0, bias=0.0, positive=False):
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    values = (values.remainder(127) - 63) / scale + bias
    if positive:
        values = values.abs() + bias
    return values.to(torch.bfloat16)


def _strategy_layout(strategy):
    return {
        "height": "HEIGHT_SHARDED",
        "width": "WIDTH_SHARDED",
        "block": "BLOCK_SHARDED",
    }[strategy]


def _memory_configs_by_subject(executable_spec):
    return {
        str(plan["subject"]): plan
        for plan in executable_spec["tensor_memory_config_plans"]
    }


def _distributions_by_buffer(executable_spec):
    return {
        str(plan["buffer"]): plan
        for plan in executable_spec["buffer_distribution_plans"]
    }


def _reshard_edges(executable_spec):
    return {
        (str(plan["source_value"]), str(plan["target_value"])): plan
        for plan in executable_spec["reshard_plans"]
    }


def _compute_operation_names(executable_spec):
    return {
        str(op["operation_name"])
        for kernel in executable_spec["kernels"]
        for op in kernel.get("compute_ops", [])
    }


def _assert_t3_compute_contract(
    executable_spec,
    *,
    sources,
    residents,
    strategy,
    source_region_shape,
    expected_ops,
):
    memory_configs = _memory_configs_by_subject(executable_spec)
    distributions = _distributions_by_buffer(executable_spec)
    reshard_edges = _reshard_edges(executable_spec)

    for source, resident in zip(sources, residents):
        assert str(memory_configs[source]["memory_layout"]) == "INTERLEAVED"
        assert str(memory_configs[source]["buffer_type"]) == "DRAM"
        assert str(memory_configs[resident]["memory_layout"]) == _strategy_layout(strategy)
        assert str(memory_configs[resident]["buffer_type"]) == "L1"
        assert str(memory_configs[resident]["source_buffer"]) == source
        assert str(memory_configs[resident]["shard_distribution_strategy"]) == strategy

        distribution = distributions[resident]
        assert str(distribution["distribution_kind"]) == "sharded"
        assert str(distribution["memory_space"]) == "L1"
        assert str(distribution["sharding_strategy"]) == strategy
        assert str(distribution["source_buffer"]) == source
        assert str(distribution["source_region_kind"]) == "per_work_tile"
        assert tuple(int(dim) for dim in distribution["source_region_shape"]) == (
            source_region_shape
        )
        assert str(distribution["logical_index_mapping"]) == "work_packet_row_major"
        assert str(distribution["core_local_address_mapping"]) == "l1_shard_linear"

        reshard = reshard_edges[(source, resident)]
        assert str(reshard["conversion_kind"]) == "interleaved_to_sharded"
        assert str(reshard["materialization_protocol"]) == "staged_copy"
        assert str(reshard["admission_status"]) == "admitted"
        assert str(reshard["unsupported_reason"]) == ""

    assert expected_ops <= _compute_operation_names(executable_spec)
    assert not executable_spec.get("direct_runtime_unsupported_reasons", [])


def _sharded_l1_config(strategy, grid_x, grid_y, tile_m, tile_n):
    return T.sharded_l1(
        strategy=strategy,
        grid=T.CoreGrid(x=max(1, grid_x), y=max(1, grid_y)),
        shard_shape=(tile_m, tile_n),
        orientation="row_major",
        allow_reshard=True,
    )


def _t3_elementwise_chain_kernel(*, grid_x, grid_y, strategy, tile_m=32, tile_n=32):
    m = grid_y * tile_m
    n = grid_x * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "bfloat16"),
        B: T.Tensor((m, n), "bfloat16"),
        C: T.Tensor((m, n), "bfloat16"),
        D: T.Tensor((m, n), "bfloat16"),
        E: T.Tensor((m, n), "bfloat16"),
        O: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            a_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            b_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            c_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            d_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            e_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            acc = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            rhs_b = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            rhs_c = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            rhs_d = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            rhs_e = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            sharded_l1 = _sharded_l1_config(strategy, grid_x, grid_y, tile_m, tile_n)
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    B: T.interleaved_dram(),
                    C: T.interleaved_dram(),
                    D: T.interleaved_dram(),
                    E: T.interleaved_dram(),
                    a_tile: sharded_l1,
                    b_tile: sharded_l1,
                    c_tile: sharded_l1,
                    d_tile: sharded_l1,
                    e_tile: sharded_l1,
                    O: T.interleaved_dram(),
                }
            )
            row = by * tile_m
            col = bx * tile_n
            T.copy(A[row, col], a_tile)
            T.copy(B[row, col], b_tile)
            T.copy(C[row, col], c_tile)
            T.copy(D[row, col], d_tile)
            T.copy(E[row, col], e_tile)
            T.copy(a_tile, acc)
            T.copy(b_tile, rhs_b)
            T.copy(c_tile, rhs_c)
            T.copy(d_tile, rhs_d)
            T.copy(e_tile, rhs_e)
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] + rhs_b[i, j]
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] - rhs_c[i, j]
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] * rhs_d[i, j]
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] / rhs_e[i, j]
            T.copy(acc, O[row, col])

    return main


def _t3_elementwise_reduce_kernel(*, grid_x, grid_y, strategy, tile_m=32, tile_n=32):
    m = grid_y * tile_m
    n = grid_x * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "bfloat16"),
        B: T.Tensor((m, n), "bfloat16"),
        O: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            a_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            b_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            acc = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            rhs = T.alloc_fragment((tile_m, tile_n), "bfloat16")
            row_sum = T.alloc_fragment((tile_m,), "bfloat16")
            sharded_l1 = _sharded_l1_config(strategy, grid_x, grid_y, tile_m, tile_n)
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    B: T.interleaved_dram(),
                    a_tile: sharded_l1,
                    b_tile: sharded_l1,
                    O: T.interleaved_dram(),
                }
            )
            row = by * tile_m
            col = bx * tile_n
            T.copy(A[row, col], a_tile)
            T.copy(B[row, col], b_tile)
            T.copy(a_tile, acc)
            T.copy(b_tile, rhs)
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] + rhs[i, j]
            T.reduce_sum(acc, row_sum, dim=1, clear=True)
            for i, j in T.Parallel(tile_m, tile_n):
                acc[i, j] = acc[i, j] / row_sum[i]
            T.copy(acc, O[row, col])

    return main


ELEMENTWISE_CHAIN_CASES = [
    ("height_256x32", 1, 8, "height", 32, 32),
    ("width_32x256", 8, 1, "width", 32, 32),
    ("block_rect_128x512", 8, 4, "block", 32, 64),
    ("block_large_1024x1024", 32, 32, "block", 32, 32),
]


@pytest.mark.parametrize(
    "case_name,grid_x,grid_y,strategy,tile_m,tile_n",
    ELEMENTWISE_CHAIN_CASES,
)
def test_blackhole_t3_sharded_elementwise_chain_bf16_direct_runtime(
    case_name, grid_x, grid_y, strategy, tile_m, tile_n
):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m = grid_y * tile_m
    n = grid_x * tile_n
    a = _bf16_pattern(m, n, scale=59.0)
    b = _bf16_pattern(m, n, scale=73.0, bias=0.25)
    c = _bf16_pattern(m, n, scale=67.0, bias=-0.125)
    d = _bf16_pattern(m, n, scale=113.0, bias=0.75, positive=True)
    e = _bf16_pattern(m, n, scale=131.0, bias=1.25, positive=True)
    out = torch.zeros_like(a)
    expected = ((((a.float() + b.float()) - c.float()) * d.float()) / e.float()).to(
        torch.bfloat16
    )

    artifact = _lower_blackhole(
        _t3_elementwise_chain_kernel(
            grid_x=grid_x,
            grid_y=grid_y,
            strategy=strategy,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    )
    if case_name == "block_large_1024x1024":
        assert int(extract_blackhole_work_per_core(_require_device_main(artifact))) > 1
    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_t3_compute_contract(
        executable_spec,
        sources=["A", "B", "C", "D", "E"],
        residents=["a_tile", "b_tile", "c_tile", "d_tile", "e_tile"],
        strategy=strategy,
        source_region_shape=(tile_m, tile_n),
        expected_ops={"add_tiles", "sub_tiles", "mul_tiles", "recip_tile"},
    )

    artifact.codegen_mod["main"](a, b, c, d, e, out)
    assert_tensors_close_or_dump(
        out,
        expected,
        atol=8e-2,
        rtol=8e-2,
        failure_message=f"T3 {case_name} elementwise chain mismatch",
    )


REDUCE_MIXED_CASES = [
    ("height_reduce_256x32", 1, 8, "height", 32, 32),
    ("width_reduce_32x256", 8, 1, "width", 32, 32),
    ("block_reduce_256x512", 16, 8, "block", 32, 32),
]


@pytest.mark.parametrize(
    "case_name,grid_x,grid_y,strategy,tile_m,tile_n",
    REDUCE_MIXED_CASES,
)
def test_blackhole_t3_sharded_elementwise_reduce_mix_bf16_direct_runtime(
    case_name, grid_x, grid_y, strategy, tile_m, tile_n
):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m = grid_y * tile_m
    n = grid_x * tile_n
    a = _bf16_pattern(m, n, scale=89.0, bias=0.5, positive=True)
    b = _bf16_pattern(m, n, scale=97.0, bias=0.25, positive=True)
    out = torch.zeros_like(a)
    expected = torch.zeros_like(a)
    combined = a.float() + b.float()
    for by in range(grid_y):
        row = by * tile_m
        for bx in range(grid_x):
            col = bx * tile_n
            tile = combined[row : row + tile_m, col : col + tile_n]
            expected[row : row + tile_m, col : col + tile_n] = (
                tile / tile.sum(dim=1, keepdim=True)
            ).to(torch.bfloat16)

    artifact = _lower_blackhole(
        _t3_elementwise_reduce_kernel(
            grid_x=grid_x,
            grid_y=grid_y,
            strategy=strategy,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    )
    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_t3_compute_contract(
        executable_spec,
        sources=["A", "B"],
        residents=["a_tile", "b_tile"],
        strategy=strategy,
        source_region_shape=(tile_m, tile_n),
        expected_ops={"add_tiles", "reduce_tile", "recip_tile", "mul_tiles_bcast_cols"},
    )

    artifact.codegen_mod["main"](a, b, out)
    assert_tensors_close_or_dump(
        out,
        expected,
        atol=8e-2,
        rtol=8e-2,
        failure_message=f"T3 {case_name} elementwise/reduce mix mismatch",
    )
