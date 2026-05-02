import re

import pytest
import torch

from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
    fragment_fill_cast_publish_kernel,
)
from .test_blackhole_copy_pipeline import _extract_blackhole_executable_spec


M = 32
N = 32
STANDALONE_REDUCE_SIM_REASON = "standalone reduce_tile leaf direct runtime is gated"
STANDALONE_REDUCE_SIM_REASON_DETAILS = (
    "tensix_execute_pacr count=1",
    "vector-output materialization is not yet admitted",
)
STANDALONE_FILL_TYPECAST_SIM_REASON = (
    "standalone fill/typecast publish direct runtime is gated"
)


def _lower_blackhole(kernel):
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _direct_runtime_unsupported_reasons(artifact):
    metadata = artifact.codegen_mod.get_function_metadata("main")
    return [str(reason) for reason in metadata.get("direct_runtime_unsupported_reasons", [])]


def _compute_operation_names(executable_spec):
    operations = []
    for kernel in executable_spec["kernels"]:
        for op in kernel.get("compute_ops", []):
            operations.append(str(op["operation_name"]))
    return operations


def _compute_kernel_source(executable_spec):
    return str(
        next(
            kernel["source_code"]
            for kernel in executable_spec["kernels"]
            if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
        )
    )


def binary_leaf_kernel(operation):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "bfloat16"),
        B: T.Tensor((M, N), "bfloat16"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, N), "bfloat16")
            B_shared = T.alloc_shared((M, N), "bfloat16")
            A_local = T.alloc_fragment((M, N), "bfloat16")
            B_local = T.alloc_fragment((M, N), "bfloat16")
            C_local = T.alloc_fragment((M, N), "bfloat16")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(M, N):
                if operation == "add":
                    C_local[i, j] = A_local[i, j] + B_local[i, j]
                elif operation == "mul":
                    C_local[i, j] = A_local[i, j] * B_local[i, j]
                else:
                    C_local[i, j] = T.max(A_local[i, j], B_local[i, j])
            T.copy(C_local, C)

    return main


def broadcast_cols_leaf_kernel(operation):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "bfloat16"),
        B: T.Tensor((M,), "bfloat16"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, N), "bfloat16")
            B_shared = T.alloc_shared((M,), "bfloat16")
            A_local = T.alloc_fragment((M, N), "bfloat16")
            B_local = T.alloc_fragment((M,), "bfloat16")
            C_local = T.alloc_fragment((M, N), "bfloat16")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(M, N):
                if operation == "add":
                    C_local[i, j] = A_local[i, j] + B_local[i]
                else:
                    C_local[i, j] = A_local[i, j] * B_local[i]
            T.copy(C_local, C)

    return main


def unary_exp2_leaf_kernel():
    @T.prim_func
    def main(A: T.Tensor((M, N), "bfloat16"), C: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, N), "bfloat16")
            A_local = T.alloc_fragment((M, N), "bfloat16")
            C_local = T.alloc_fragment((M, N), "bfloat16")
            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(M, N):
                C_local[i, j] = T.exp2(A_local[i, j])
            T.copy(C_local, C)

    return main


def reduction_sum_leaf_kernel():
    @T.prim_func
    def main(A: T.Tensor((M, N), "bfloat16"), C: T.Tensor((M,), "bfloat16")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((M, N), "bfloat16")
            A_local = T.alloc_fragment((M, N), "bfloat16")
            C_local = T.alloc_fragment((M,), "bfloat16")
            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            T.reduce_sum(A_local, C_local, dim=1, clear=True)
            T.copy(C_local, C)

    return main


LEAF_CONTRACT_CASES = [
    ("binary_add", binary_leaf_kernel("add"), {"add_tiles"}),
    ("binary_mul", binary_leaf_kernel("mul"), {"mul_tiles"}),
    ("binary_max", binary_leaf_kernel("max"), {"binary_max_tile"}),
    (
        "broadcast_add_cols",
        broadcast_cols_leaf_kernel("add"),
        {"add_tiles_bcast_cols"},
    ),
    (
        "broadcast_mul_cols",
        broadcast_cols_leaf_kernel("mul"),
        {"mul_tiles_bcast_cols"},
    ),
    ("unary_exp2", unary_exp2_leaf_kernel(), {"exp2_tile"}),
    ("reduction_sum", reduction_sum_leaf_kernel(), {"reduce_tile"}),
    ("typecast_publish", fragment_fill_cast_publish_kernel(), {"fill_tile", "typecast_tile"}),
]


@pytest.mark.parametrize("case_name,kernel,expected_ops", LEAF_CONTRACT_CASES)
def test_blackhole_standalone_leaf_compute_projects_typed_runtime_contracts(
    case_name, kernel, expected_ops
):
    artifact = _lower_blackhole(kernel)
    executable_spec = _extract_blackhole_executable_spec(artifact)

    assert "compute_contract" not in executable_spec
    assert "multi_compute_contracts" not in executable_spec
    reasons = _direct_runtime_unsupported_reasons(artifact)
    if case_name == "reduction_sum":
        reduce_reasons = [
            reason for reason in reasons if STANDALONE_REDUCE_SIM_REASON in reason
        ]
        assert reduce_reasons
        assert all(
            detail in reduce_reasons[0] for detail in STANDALONE_REDUCE_SIM_REASON_DETAILS
        )
    elif case_name == "typecast_publish":
        assert any(STANDALONE_FILL_TYPECAST_SIM_REASON in reason for reason in reasons)
    else:
        assert not reasons, case_name
    assert expected_ops <= set(_compute_operation_names(executable_spec))


def test_blackhole_standalone_reduce_packs_before_reduce_uninit():
    artifact = _lower_blackhole(reduction_sum_leaf_kernel())
    executable_spec = _extract_blackhole_executable_spec(artifact)
    compute_source = _compute_kernel_source(executable_spec)

    reduce_window = re.search(
        r"reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>\(\d+, \d+, (?P<out_cb>\d+)\);"
        r"(?P<body>.*?)"
        r"cb_push_back\((?P=out_cb), 1\);",
        compute_source,
        flags=re.DOTALL,
    )
    assert reduce_window is not None

    out_cb = int(reduce_window.group("out_cb"))
    body = reduce_window.group("body")
    reduce_tile_offset = body.find("reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>")
    pack_offset = body.find(f"pack_tile(0, {out_cb}, 0);")
    uninit_offset = body.find("reduce_uninit<false>();")

    assert reduce_tile_offset >= 0
    assert pack_offset > reduce_tile_offset
    assert uninit_offset > pack_offset


@pytest.mark.parametrize(
    "case_name,kernel,input_builder,expected_builder",
    [
        (
            "binary_add",
            binary_leaf_kernel("add"),
            lambda: (
                torch.randn(M, N, dtype=torch.bfloat16),
                torch.randn(M, N, dtype=torch.bfloat16),
            ),
            lambda a, b: (a + b).to(torch.bfloat16),
        ),
        (
            "binary_mul",
            binary_leaf_kernel("mul"),
            lambda: (
                torch.randn(M, N, dtype=torch.bfloat16),
                torch.randn(M, N, dtype=torch.bfloat16),
            ),
            lambda a, b: (a * b).to(torch.bfloat16),
        ),
        (
            "binary_max",
            binary_leaf_kernel("max"),
            lambda: (
                torch.randn(M, N, dtype=torch.bfloat16),
                torch.randn(M, N, dtype=torch.bfloat16),
            ),
            lambda a, b: torch.maximum(a, b).to(torch.bfloat16),
        ),
        (
            "broadcast_add_cols",
            broadcast_cols_leaf_kernel("add"),
            lambda: (
                torch.randn(M, N, dtype=torch.bfloat16),
                torch.randn(M, dtype=torch.bfloat16),
            ),
            lambda a, b: (a + b.view(M, 1)).to(torch.bfloat16),
        ),
        (
            "broadcast_mul_cols",
            broadcast_cols_leaf_kernel("mul"),
            lambda: (
                torch.randn(M, N, dtype=torch.bfloat16),
                torch.randn(M, dtype=torch.bfloat16),
            ),
            lambda a, b: (a * b.view(M, 1)).to(torch.bfloat16),
        ),
        (
            "unary_exp2",
            unary_exp2_leaf_kernel(),
            lambda: (torch.randn(M, N, dtype=torch.bfloat16) * 0.125,),
            lambda a: torch.exp2(a.float()).to(torch.bfloat16),
        ),
        (
            "reduction_sum",
            reduction_sum_leaf_kernel(),
            lambda: (torch.randn(M, N, dtype=torch.bfloat16),),
            lambda a: torch.sum(a, dim=1).to(torch.bfloat16),
        ),
        (
            "typecast_publish",
            fragment_fill_cast_publish_kernel(),
            lambda: (),
            lambda: torch.full((M, N), 3.5, dtype=torch.bfloat16),
        ),
    ],
)
def test_blackhole_standalone_leaf_compute_bf16_direct_runtime(
    case_name, kernel, input_builder, expected_builder
):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    inputs = input_builder()
    expected = expected_builder(*inputs)
    output = torch.zeros_like(expected)

    artifact = _lower_blackhole(kernel)
    reasons = _direct_runtime_unsupported_reasons(artifact)
    if reasons:
        pytest.skip(f"{case_name} direct runtime gated: {reasons}")

    artifact.codegen_mod["main"](*inputs, output)
    assert_tensors_close_or_dump(
        output,
        expected,
        atol=2e-2,
        rtol=2e-2,
        failure_message=f"{case_name} standalone leaf direct-runtime mismatch",
    )
