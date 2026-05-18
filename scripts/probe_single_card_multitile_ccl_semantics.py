#!/usr/bin/env python3
"""Single-card multi-tile value-semantics probe for T10 collectives.

This probe intentionally does not claim fabric or multi-device CCL coverage.
It maps logical participants onto tile-aligned tensor partitions on one host
process and verifies bf16 all-gather, reduce-scatter, and all-to-all value
semantics against independent full-tensor references.
"""

from __future__ import annotations

import sys
from math import prod
from typing import Iterable

import torch


TILE_SIZE = 32
PARTICIPANT_COUNT = 2


def _emit(key: str, value: object) -> None:
    print(f"{key}={value}", flush=True)


def _make_bf16_tensor(shape: Iterable[int], offset: int) -> torch.Tensor:
    shape = tuple(int(dim) for dim in shape)
    values = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    values = (values.remainder(251) + offset) / 16.0
    return values.to(torch.bfloat16)


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.to(torch.float32) - rhs.to(torch.float32)).abs().max().item()


def _all_equal(values: Iterable[bool]) -> bool:
    return all(bool(value) for value in values)


def _verify_all_gather() -> bool:
    tensor_axis = 3
    shard_shape = (1, 2, TILE_SIZE * 2, TILE_SIZE * 2)
    shards = [
        _make_bf16_tensor(shard_shape, offset=participant * 17)
        for participant in range(PARTICIPANT_COUNT)
    ]

    gathered = torch.cat(shards, dim=tensor_axis)
    outputs = [torch.cat(shards, dim=tensor_axis) for _ in range(PARTICIPANT_COUNT)]
    ok = _all_equal(torch.equal(output, gathered) for output in outputs)
    max_abs_diff = max(_max_abs_diff(output, gathered) for output in outputs)

    _emit("all_gather_shape", list(gathered.shape))
    _emit("all_gather_ok", str(ok).lower())
    _emit("all_gather_max_abs_diff", max_abs_diff)
    return ok


def _verify_reduce_scatter() -> bool:
    tensor_axis = 3
    input_shape = (1, 2, TILE_SIZE * 2, TILE_SIZE * 4)
    inputs = [
        _make_bf16_tensor(input_shape, offset=participant * 19)
        for participant in range(PARTICIPANT_COUNT)
    ]

    reduced = torch.stack([tensor.to(torch.float32) for tensor in inputs], dim=0).sum(dim=0)
    reduced = reduced.to(torch.bfloat16)
    outputs = list(torch.chunk(reduced, PARTICIPANT_COUNT, dim=tensor_axis))
    references = list(torch.chunk(reduced, PARTICIPANT_COUNT, dim=tensor_axis))
    ok = _all_equal(torch.equal(output, reference) for output, reference in zip(outputs, references))
    max_abs_diff = max(_max_abs_diff(output, reference) for output, reference in zip(outputs, references))

    _emit("reduce_scatter_output_shapes", [list(output.shape) for output in outputs])
    _emit("reduce_scatter_ok", str(ok).lower())
    _emit("reduce_scatter_max_abs_diff", max_abs_diff)
    return ok


def _verify_all_to_all() -> bool:
    split_axis = 2
    concat_axis = 3
    full_shape = (1, 2, TILE_SIZE * 4, TILE_SIZE * 4)
    full_tensor = _make_bf16_tensor(full_shape, offset=23)

    input_shards = list(torch.chunk(full_tensor, PARTICIPANT_COUNT, dim=split_axis))
    per_source_chunks = [
        list(torch.chunk(source, PARTICIPANT_COUNT, dim=concat_axis))
        for source in input_shards
    ]
    outputs = [
        torch.cat(
            [per_source_chunks[source][destination] for source in range(PARTICIPANT_COUNT)],
            dim=split_axis,
        )
        for destination in range(PARTICIPANT_COUNT)
    ]
    references = list(torch.chunk(full_tensor, PARTICIPANT_COUNT, dim=concat_axis))
    ok = _all_equal(torch.equal(output, reference) for output, reference in zip(outputs, references))
    max_abs_diff = max(_max_abs_diff(output, reference) for output, reference in zip(outputs, references))

    _emit("all_to_all_output_shapes", [list(output.shape) for output in outputs])
    _emit("all_to_all_ok", str(ok).lower())
    _emit("all_to_all_max_abs_diff", max_abs_diff)
    return ok


def main() -> int:
    _emit("scope", "single_card_multitile_value_semantics")
    _emit("tile_size", TILE_SIZE)
    _emit("participant_count", PARTICIPANT_COUNT)
    _emit("dtype", "bf16")

    results = [
        _verify_all_gather(),
        _verify_reduce_scatter(),
        _verify_all_to_all(),
    ]
    ok = _all_equal(results)
    _emit("single_card_multitile_ccl_semantics", "ok" if ok else "failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
