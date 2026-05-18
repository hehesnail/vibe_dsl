from math import prod

import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import assert_tensors_close_or_dump, check_blackhole_direct_execution_requirements


TILE_SIZE = 32
PARTICIPANT_COUNT = 4
TILE_ROWS = 8
TILE_COLS = 8


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


def t10_single_card_all_gather_kernel(tile_rows=TILE_ROWS, tile_cols=TILE_COLS):
    rows = tile_rows * TILE_SIZE
    shard_cols = tile_cols * TILE_SIZE
    output_cols = shard_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A0: T.Tensor((rows, shard_cols), "bfloat16"),
        A1: T.Tensor((rows, shard_cols), "bfloat16"),
        A2: T.Tensor((rows, shard_cols), "bfloat16"),
        A3: T.Tensor((rows, shard_cols), "bfloat16"),
        O: T.Tensor((PARTICIPANT_COUNT, rows, output_cols), "bfloat16"),
    ):
        with T.Kernel(tile_cols, tile_rows, PARTICIPANT_COUNT, threads=128) as (bx, by, bp):
            a0_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a1_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a2_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a3_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            l1_tile = _l1_tile_config()
            T.annotate_memory_config(
                {
                    A0: T.interleaved_dram(),
                    A1: T.interleaved_dram(),
                    A2: T.interleaved_dram(),
                    A3: T.interleaved_dram(),
                    a0_tile: l1_tile,
                    a1_tile: l1_tile,
                    a2_tile: l1_tile,
                    a3_tile: l1_tile,
                    O: T.interleaved_dram(),
                }
            )
            row = by * TILE_SIZE
            col = bx * TILE_SIZE
            T.copy(A0[row, col], a0_tile)
            T.copy(A1[row, col], a1_tile)
            T.copy(A2[row, col], a2_tile)
            T.copy(A3[row, col], a3_tile)
            T.copy(a0_tile, O[bp, row, col])
            T.copy(a1_tile, O[bp, row, col + shard_cols])
            T.copy(a2_tile, O[bp, row, col + shard_cols * 2])
            T.copy(a3_tile, O[bp, row, col + shard_cols * 3])

    return main


def t10_single_card_reduce_scatter_kernel(tile_rows=TILE_ROWS, tile_cols=TILE_COLS):
    rows = tile_rows * TILE_SIZE
    output_cols = tile_cols * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A0: T.Tensor((rows, input_cols), "bfloat16"),
        A1: T.Tensor((rows, input_cols), "bfloat16"),
        A2: T.Tensor((rows, input_cols), "bfloat16"),
        A3: T.Tensor((rows, input_cols), "bfloat16"),
        O: T.Tensor((PARTICIPANT_COUNT, rows, output_cols), "bfloat16"),
    ):
        with T.Kernel(tile_cols, tile_rows, PARTICIPANT_COUNT, threads=128) as (bx, by, bp):
            a0_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a1_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a2_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            a3_tile = T.alloc_shared((TILE_SIZE, TILE_SIZE), "bfloat16")
            acc = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            rhs1 = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            rhs2 = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            rhs3 = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "bfloat16")
            l1_tile = _l1_tile_config()
            T.annotate_memory_config(
                {
                    A0: T.interleaved_dram(),
                    A1: T.interleaved_dram(),
                    A2: T.interleaved_dram(),
                    A3: T.interleaved_dram(),
                    a0_tile: l1_tile,
                    a1_tile: l1_tile,
                    a2_tile: l1_tile,
                    a3_tile: l1_tile,
                    O: T.interleaved_dram(),
                }
            )
            row = by * TILE_SIZE
            col = bx * TILE_SIZE
            source_col = bp * output_cols + col
            T.copy(A0[row, source_col], a0_tile)
            T.copy(A1[row, source_col], a1_tile)
            T.copy(A2[row, source_col], a2_tile)
            T.copy(A3[row, source_col], a3_tile)
            T.copy(a0_tile, acc)
            T.copy(a1_tile, rhs1)
            T.copy(a2_tile, rhs2)
            T.copy(a3_tile, rhs3)
            for i, j in T.Parallel(TILE_SIZE, TILE_SIZE):
                acc[i, j] = acc[i, j] + rhs1[i, j]
            for i, j in T.Parallel(TILE_SIZE, TILE_SIZE):
                acc[i, j] = acc[i, j] + rhs2[i, j]
            for i, j in T.Parallel(TILE_SIZE, TILE_SIZE):
                acc[i, j] = acc[i, j] + rhs3[i, j]
            T.copy(acc, O[bp, row, col])

    return main


def t10_single_card_all_to_all_kernel(tile_rows=TILE_ROWS, tile_cols=TILE_COLS):
    source_tile_rows = tile_rows // PARTICIPANT_COUNT
    source_rows = source_tile_rows * TILE_SIZE
    rows = source_rows * PARTICIPANT_COUNT
    output_cols = tile_cols * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT

    @T.prim_func
    def main(
        A: T.Tensor((PARTICIPANT_COUNT, source_rows, input_cols), "bfloat16"),
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
            source_participant = by // source_tile_rows
            source_row = (by % source_tile_rows) * TILE_SIZE
            source_col = bp * output_cols + col
            T.copy(A[source_participant, source_row, source_col], dest_tile)
            T.copy(dest_tile, O[bp, row, col])

    return main


def test_blackhole_t10_single_card_all_gather_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    rows = TILE_ROWS * TILE_SIZE
    shard_cols = TILE_COLS * TILE_SIZE
    output_cols = shard_cols * PARTICIPANT_COUNT
    a0 = _bf16_tensor((rows, shard_cols), offset=3)
    a1 = _bf16_tensor((rows, shard_cols), offset=19)
    a2 = _bf16_tensor((rows, shard_cols), offset=35)
    a3 = _bf16_tensor((rows, shard_cols), offset=51)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    reference = torch.stack(
        [torch.cat([a0, a1, a2, a3], dim=1) for _ in range(PARTICIPANT_COUNT)],
        dim=0,
    )

    artifact = _lower_blackhole(t10_single_card_all_gather_kernel())
    artifact.codegen_mod["main"](a0, a1, a2, a3, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=0,
        rtol=0,
        failure_message="single-card all-gather participant outputs mismatch",
    )


def test_blackhole_t10_single_card_reduce_scatter_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    rows = TILE_ROWS * TILE_SIZE
    output_cols = TILE_COLS * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT
    a0 = _bf16_tensor((rows, input_cols), offset=5)
    a1 = _bf16_tensor((rows, input_cols), offset=17)
    a2 = _bf16_tensor((rows, input_cols), offset=29)
    a3 = _bf16_tensor((rows, input_cols), offset=41)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    reduced = (
        a0.to(torch.float32)
        + a1.to(torch.float32)
        + a2.to(torch.float32)
        + a3.to(torch.float32)
    ).to(torch.bfloat16)
    reference = torch.stack(
        [
            reduced[:, participant * output_cols : (participant + 1) * output_cols]
            for participant in range(PARTICIPANT_COUNT)
        ],
        dim=0,
    )

    artifact = _lower_blackhole(t10_single_card_reduce_scatter_kernel())
    artifact.codegen_mod["main"](a0, a1, a2, a3, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=1e-3,
        rtol=1e-3,
        failure_message="single-card reduce-scatter participant outputs mismatch",
    )


def test_blackhole_t10_single_card_all_to_all_multitile_runtime_correctness():
    _require_blackhole_direct_runtime()

    source_rows = (TILE_ROWS // PARTICIPANT_COUNT) * TILE_SIZE
    rows = TILE_ROWS * TILE_SIZE
    output_cols = TILE_COLS * TILE_SIZE
    input_cols = output_cols * PARTICIPANT_COUNT
    a = _bf16_tensor((PARTICIPANT_COUNT, source_rows, input_cols), offset=37)
    output = torch.zeros((PARTICIPANT_COUNT, rows, output_cols), dtype=torch.bfloat16)
    full_rows = torch.cat([a[participant] for participant in range(PARTICIPANT_COUNT)], dim=0)
    reference = torch.stack(
        [
            full_rows[:, participant * output_cols : (participant + 1) * output_cols]
            for participant in range(PARTICIPANT_COUNT)
        ],
        dim=0,
    )

    artifact = _lower_blackhole(t10_single_card_all_to_all_kernel())
    artifact.codegen_mod["main"](a, output)

    assert_tensors_close_or_dump(
        output,
        reference,
        atol=0,
        rtol=0,
        failure_message="single-card all-to-all participant outputs mismatch",
    )
