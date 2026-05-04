import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
)
from .test_blackhole_copy_pipeline import _extract_blackhole_executable_spec


VALUE_INDEX_SELECTION_MARKER = (
    "Existing TIR value/index row selection lowered as one typed compute scan."
)


def existing_tir_value_index_selection_kernel(
    *,
    M=320,
    N=128,
    k=6,
    blk_m=64,
    dtype=T.float32,
    index_dtype=T.int32,
):
    @T.prim_func
    def main(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, k], dtype),
        topk_indices: T.Tensor([M, k], index_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
            max_val = T.alloc_fragment([blk_m], dtype=dtype)
            expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
            max_idx = T.alloc_fragment([blk_m], T.int32)

            T.copy(logits[bx * blk_m, 0], logits_frag)

            for selection_rank in T.serial(k):
                T.fill(expand_max_idx, -1)
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):
                    expand_max_idx[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j],
                        j,
                        expand_max_idx[i, j],
                    )

                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):
                    logits_frag[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j],
                        -10000.0,
                        logits_frag[i, j],
                    )

                for i in T.Parallel(blk_m):
                    topk_gates[bx * blk_m + i, selection_rank] = max_val[i]
                    topk_indices[bx * blk_m + i, selection_rank] = max_idx[i]

    return main


def _lower_blackhole(kernel):
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _direct_runtime_unsupported_reasons(artifact):
    metadata = artifact.codegen_mod.get_function_metadata("main")
    return [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]


def _kernel_source(executable_spec, kind):
    return str(
        next(
            kernel["source_code"]
            for kernel in executable_spec["kernels"]
            if str(kernel["kind"]) == kind
        )
    )


def test_blackhole_existing_tir_value_index_selection_projects_contracts():
    artifact = _lower_blackhole(existing_tir_value_index_selection_kernel())
    executable_spec = _extract_blackhole_executable_spec(artifact)

    assert "selection_plans" not in executable_spec

    runtime_buffers = {
        str(arg["buffer"]): str(arg["kind"])
        for arg in executable_spec["runtime_args"]
        if str(arg["kind"])
        in {
            "input_buffer_addr",
            "input_buffer_addr32",
            "output_buffer_addr",
            "output_buffer_addr32",
        }
    }
    assert runtime_buffers["logits"].startswith("input_buffer_addr")
    assert runtime_buffers["topk_gates"].startswith("output_buffer_addr")
    assert runtime_buffers["topk_indices"].startswith("output_buffer_addr")

    reader_source = _kernel_source(executable_spec, "reader")
    compute_source = _kernel_source(executable_spec, "compute")
    assert "logits_addr" in reader_source
    assert "TensorAccessorArgs<0>()" in reader_source
    assert "noc_async_read_tile" in reader_source
    assert "Argument 0: logits" not in compute_source
    assert "((float*)logits)" not in compute_source
    assert "logits[" not in compute_source
    assert VALUE_INDEX_SELECTION_MARKER in compute_source
    assert "reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>" not in compute_source
    assert "tl.blackhole.topk" not in compute_source
    assert "TTSelectionPlan" not in compute_source
    assert (
        "tilelang_fill_tiled_cb_slice_nfaces<int32_t>(dst, static_cast<uint32_t>(0), "
        "static_cast<uint32_t>(8192)"
    ) not in compute_source

    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert reasons == []


def _make_unique_topk_logits(M, N, torch_dtype):
    pattern = ((torch.arange(N, dtype=torch.float32) * 37) % N) / float(N)
    return pattern.unsqueeze(0).repeat(M, 1).to(torch_dtype)


def _run_direct_topk_case(*, M, N, k, blk_m, tir_dtype, torch_dtype, atol, rtol):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = existing_tir_value_index_selection_kernel(
        M=M,
        N=N,
        k=k,
        blk_m=blk_m,
        dtype=tir_dtype,
    )
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)

    assert _direct_runtime_unsupported_reasons(artifact) == []

    logits = _make_unique_topk_logits(M, N, torch_dtype)
    topk_gates = torch.full((M, k), -777.0, dtype=torch_dtype)
    topk_indices = torch.full((M, k), -999, dtype=torch.int32)

    artifact.codegen_mod.get_function("main")(logits, topk_gates, topk_indices)

    expected_gates, expected_indices = torch.topk(logits.to(torch.float32), k, dim=1)
    expected_gates = expected_gates.to(torch_dtype)
    expected_indices = expected_indices.to(torch.int32)

    assert_tensors_close_or_dump(
        topk_gates.to(torch.float32),
        expected_gates.to(torch.float32),
        atol=atol,
        rtol=rtol,
        failure_message=f"topk values mismatch M={M} N={N} k={k} dtype={torch_dtype}",
    )
    assert torch.equal(topk_indices, expected_indices)


def test_blackhole_existing_tir_value_index_selection_direct_runtime_fp32_single_work():
    _run_direct_topk_case(
        M=64,
        N=128,
        k=6,
        blk_m=64,
        tir_dtype=T.float32,
        torch_dtype=torch.float32,
        atol=1e-5,
        rtol=1e-5,
    )


def test_blackhole_existing_tir_value_index_selection_direct_runtime_fp32_multi_work():
    _run_direct_topk_case(
        M=320,
        N=128,
        k=6,
        blk_m=64,
        tir_dtype=T.float32,
        torch_dtype=torch.float32,
        atol=1e-5,
        rtol=1e-5,
    )


def test_blackhole_existing_tir_value_index_selection_direct_runtime_bf16_values_int32_indices():
    _run_direct_topk_case(
        M=128,
        N=128,
        k=6,
        blk_m=64,
        tir_dtype="bfloat16",
        torch_dtype=torch.bfloat16,
        atol=1e-2,
        rtol=1e-2,
    )
