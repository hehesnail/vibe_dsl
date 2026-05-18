from math import prod

import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import assert_tensors_close_or_dump, check_blackhole_direct_execution_requirements


TILE_SIZE = 32
PARTICIPANT_COUNT = 2


def _lower_blackhole(kernel):
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _require_blackhole_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")


def _bf16_tensor(shape, offset):
    values = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    return (values.remainder(31) + offset).to(torch.bfloat16)


def _l1_tile_config(tile_m=TILE_SIZE, tile_n=TILE_SIZE):
    return T.sharded_l1(
        strategy="block",
        grid=T.CoreGrid(x=8, y=8),
        shard_shape=(tile_m, tile_n),
        orientation="row_major",
    )


def t10_single_card_all_gather_kernel(tile_rows=2, tile_cols=2):
    rows = tile_rows * TILE_SIZE
    shard_cols = tile_cols * TILE_SIZE
    output_cols = shard_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A0: T.Tensor((rows, shard_cols), "bfloat16"),
        A1: T.Tensor((rows, shard_cols), "bfloat16"),
        O: T.Tensor((PARTICIPANT_COUNT, rows, output_cols), "bfloat16"),
    ):
        with T.Kernel(tile_cols, tile_rows, PARTICIPANT_COUNT, threads=128) as (bx, by, bp):
            a0_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a1_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            l1_tile = _l1_tile_config()
            T.annotate_memory_config(
                {
                    A0: T.interleaved_dram(),
                    A1: T.interleaved_dram(),
                    a0_tile: l1_tile,
                    a1_tile: l1_tile,
                    O: T.interleaved_dram(),
                }
            )
            row = by * TILE_SIZE
            col = bx * TILE_SIZE
            T.copy(A0[row, col], a0_tile)
            T.copy(A1[row, col], a1_tile)
            T.copy(a0_tile, O[bp, row, col])
            T.copy(a1_tile, O[bp, row, col + shard_cols])

    return main


def t10_single_card_reduce_scatter_kernel(tile_rows=2, tile_cols=2):
    rows = tile_rows * TILE_SIZE
    output_cols = tile_cols * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A0: T.Tensor((rows, input_cols), "bfloat16"),
        A1: T.Tensor((rows, input_cols), "bfloat16"),
        O: T.Tensor((PARTICIPANT_COUNT, rows, output_cols), "bfloat16"),
    ):
        with T.Kernel(tile_cols, tile_rows, PARTICIPANT_COUNT, threads=128) as (bx, by, bp):
            a_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            b_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            acc = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            rhs = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            l1_tile = _l1_tile_config()
            T.annotate_memory_config(
                {
                    A0: T.interleaved_dram(),
                    A1: T.interleaved_dram(),
                    a_tile: l1_tile,
                    b_tile: l1_tile,
                    O: T.interleaved_dram(),
                }
            )
            row = by * TILE_SIZE
            col = bx * TILE_SIZE
            source_col = bp * output_cols + col
            T.copy(A0[row, source_col], a_tile)
            T.copy(A1[row, source_col], b_tile)
            T.copy(a_tile, acc)
            T.copy(b_tile, rhs)
            for i, j in T.Parallel(TILE_SIZE, TILE_SIZE):
                acc[i, j] = acc[i, j] + rhs[i, j]
            T.copy(acc, O[bp, row, col])

    return main


def t10_single_card_all_to_all_kernel(tile_rows=4, tile_cols=2):
    rows = tile_rows * TILE_SIZE
    output_cols = tile_cols * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A: T.Tensor((rows, input_cols), "bfloat16"),
        O: T.Tensor((PARTICIPANT_COUNT, rows, output_cols), "bfloat16"),
    ):
        with T.Kernel(tile_cols, tile_rows, PARTICIPANT_COUNT, threads=128) as (bx, by, bp):
            dest_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            l1_tile = _l1_tile_config()
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    dest_tile: l1_tile,
                    O: T.interleaved_dram(),
                }
            )
            row = by * TILE_SIZE
            col = bx * TILE_SIZE
            source_col = bp * output_cols + col
            T.copy(A[row, source_col], dest_tile)
            T.copy(dest_tile, O[bp, row, col])

    return main


def test_blackhole_t10_single_card_all_gather_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    rows = 2 * TILE_SIZE
    shard_cols = 2 * TILE_SIZE
    output_cols = shard_cols * PARTICIPANT_COUNT
    a0 = _bf16_tensor((rows, shard_cols), offset=3)
    a1 = _bf16_tensor((rows, shard_cols), offset=19)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    reference = torch.stack(
        [torch.cat([a0, a1], dim=1) for _ in range(PARTICIPANT_COUNT)],
        dim=0,
    )

    artifact = _lower_blackhole(t10_single_card_all_gather_kernel())
    artifact.codegen_mod["main"](a0, a1, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=0,
        rtol=0,
        failure_message="single-card all-gather participant outputs mismatch",
    )


def test_blackhole_t10_single_card_reduce_scatter_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    rows = 2 * TILE_SIZE
    output_cols = 2 * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT
    a0 = _bf16_tensor((rows, input_cols), offset=5)
    a1 = _bf16_tensor((rows, input_cols), offset=29)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    reduced = (a0.to(torch.float32) + a1.to(torch.float32)).to(torch.bfloat16)
    reference = torch.stack(
        [reduced[:, :output_cols], reduced[:, output_cols:]],
        dim=0,
    )

    artifact = _lower_blackhole(t10_single_card_reduce_scatter_kernel())
    artifact.codegen_mod["main"](a0, a1, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=1e-3,
        rtol=1e-3,
        failure_message="single-card reduce-scatter participant outputs mismatch",
    )


def test_blackhole_t10_single_card_all_to_all_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    rows = 4 * TILE_SIZE
    output_cols = 2 * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT
    a = _bf16_tensor((rows, input_cols), offset=37)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    reference = torch.stack([a[:, :output_cols], a[:, output_cols:]], dim=0)

    artifact = _lower_blackhole(t10_single_card_all_to_all_kernel())
    artifact.codegen_mod["main"](a, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=0,
        rtol=0,
        failure_message="single-card all-to-all participant outputs mismatch",
    )
