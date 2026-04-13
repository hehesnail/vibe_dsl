import os
import re
import sys
import types
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang.engine.phase import LowerAndLegalize
from tilelang.engine.phase import OptimizeForTarget
from tilelang import tvm
from tvm.target import Target

from .common import (
    check_blackhole_codegen_requirements,
    lower_blackhole_to_tt_target,
    rebuild_tt_kernel,
    rebuild_tt_program,
    require_tt_kernel,
    require_tt_program,
    tt_abi_for_kernel,
)
from .test_blackhole_copy_pipeline import (
    _rebuild_codegen_module_with_tt_program,
    _extract_blackhole_executable_spec,
    _require_blackhole_kernel,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example
import example_gqa_fwd_bshd as gqa_example


def _lower_flash_attention_to_tt_target(*, is_causal=False):
    target = Target("blackhole")
    mod = tvm.IRModule(
        {
            "main": mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                is_causal,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            )
        }
    )
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = lower_blackhole_to_tt_target(mod)
    return mod["main"]


def _run_flash_attention_tt_target(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
    return lower_blackhole_to_tt_target(mod)


def _run_flash_attention_tt_target_after_optimize(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    return lower_blackhole_to_tt_target(mod)


def _load_flash_attention_module_with_dtype(module_path, dtype_expr):
    source = Path(module_path).read_text()
    source = source.replace("dtype = T.float16", f"dtype = {dtype_expr}", 1)
    mutated = types.ModuleType(f"{Path(module_path).stem}_{dtype_expr.replace('.', '_')}")
    mutated.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), mutated.__dict__)
    return mutated


def test_flash_attention_forward_tt_target_emits_generic_lowering_requirements():
    lowered = _lower_flash_attention_to_tt_target()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs

    lowering_requirements = attrs["blackhole.lowering_requirements"]
    assert list(lowering_requirements["work_axes"]) == ["bx", "by", "bz"]
    assert {
        "gemm",
        "pointwise_chain",
    }.issubset(set(lowering_requirements["compute_op_kinds"]))
    assert "row_broadcast" not in set(lowering_requirements["compute_op_kinds"])
    assert "row_broadcast_sources" not in lowering_requirements
    assert {"exp2", "mul", "div"}.issubset(
        set(lowering_requirements["pointwise_op_kinds"])
    )
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "add" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_scores_max_updates():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.scalar_max" in script
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_tt_target_lowers_row_reductions_to_builtins():
    lowered = _lower_flash_attention_to_tt_target()
    script = lowered.script()

    assert "tl.blackhole.reduce_row" in script
    assert "\"max\"" in script
    assert "\"sum\"" in script


def test_flash_attention_forward_optimized_path_lowers_row_reductions_to_builtins():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.reduce_row" in script
    assert "row_reduction" not in set(lowering_requirements["compute_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_acc_o_row_broadcast_updates():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.mul_row_bcast" in script
    assert "tl.blackhole.div_grouped_row_bcast" in script


def test_flash_attention_forward_optimized_path_lowers_logsum_scalar_fma():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.scalar_fma" in script


def test_flash_attention_forward_optimized_path_lowers_scores_exp2_affine_updates():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.exp2_grouped_row_bcast_affine" in script
    assert "tl.blackhole.scalar_exp2_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["compute_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_fills():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.fill_fragment" in script
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_casts():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.cast_fragment_slice" in script
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_local_to_cb_staging():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.write_local_fragment_slice_to_tiled_cb" in script
    assert "O_shared_1[tx" not in script


def test_flash_attention_forward_runtime_shape_lowers_local_to_cb_with_thread_offset():
    lowered = _run_flash_attention_tt_target_after_optimize(
        mha_example,
        1,
        32,
        128,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.write_local_fragment_slice_to_tiled_cb" in script
    assert "tx * 128" in script


def test_flash_attention_gqa_reader_runtime_args_cover_all_accessor_buffers():
    lowered = _run_flash_attention_tt_target_after_optimize(
        gqa_example,
        1,
        16,
        1024,
        128,
        False,
        groups=16,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )["main"]

    tt_program = require_tt_program(lowered)
    reader_abi = tt_abi_for_kernel(tt_program, require_tt_kernel(tt_program, kind="reader", core_type="brisc"))

    accessor_buffers = [acc["buffer"] for acc in reader_abi.accessors]
    runtime_arg_buffers = [
        arg["buffer"]
        for arg in reader_abi.runtime_args
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert len(accessor_buffers) == 3
    assert runtime_arg_buffers == accessor_buffers


def test_flash_attention_gqa_top_level_runtime_args_aggregate_segment_buffers():
    lowered = _run_flash_attention_tt_target_after_optimize(
        gqa_example,
        1,
        16,
        1024,
        128,
        False,
        groups=16,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )["main"]

    tt_program = require_tt_program(lowered)
    seen_runtime_arg_identities = set()
    top_level_runtime_arg_buffers = []
    for abi in tt_program.abi_plans:
        for arg in abi.runtime_args:
            if arg["kind"] != "input_buffer_addr32":
                continue
            identity = str(arg["identity"]) if "identity" in arg else ""
            dedupe_key = f"{identity}:{str(arg['kind'])}"
            if dedupe_key in seen_runtime_arg_identities:
                continue
            seen_runtime_arg_identities.add(dedupe_key)
            top_level_runtime_arg_buffers.append(arg["buffer"])

    reader_abi = tt_abi_for_kernel(tt_program, require_tt_kernel(tt_program, kind="reader", core_type="brisc"))
    reader_runtime_arg_buffers = [
        arg["buffer"] for arg in reader_abi.runtime_args if arg["kind"] == "input_buffer_addr32"
    ]

    assert len(reader_runtime_arg_buffers) == 3
    assert top_level_runtime_arg_buffers[: len(reader_runtime_arg_buffers)] == reader_runtime_arg_buffers


def test_flash_attention_forward_lowers_mha_pipeline_end_to_end():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_pipeline_omits_legacy_semantic_attrs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    attrs = artifact.device_mod["main_kernel"].attrs
    assert attrs.get("tl.semantic_seeds") is None
    assert attrs.get("tl.semantic_manifest_seeds") is None
    assert attrs.get("tl.semantic_manifest") is None
    assert attrs.get("tl.semantic_structure") is None
    assert attrs.get("tl.semantic_witnesses") is None


def test_flash_attention_forward_pipeline_attaches_multi_phase_spatial_program():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    device_func = artifact.device_mod["main_kernel"]
    program = device_func.attrs["tl.spatial_program"]
    lowering_requirements = device_func.attrs["blackhole.lowering_requirements"]
    assert len(program.phases) >= 2
    assert len(program.channels) >= 1
    assert "phase_boundary_materialization" in {str(intent.kind) for intent in program.resource_intents}
    assert int(lowering_requirements["spatial_phase_count"]) >= 2
    assert int(lowering_requirements["spatial_channel_count"]) >= 1
    assert "scores_max" in set(lowering_requirements["spatial_phase_boundary_buffers"])


def test_flash_attention_forward_lowers_gqa_pipeline_for_supported_stage_count():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                1024,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_compute_cb_ids_stay_in_compute_window():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute_kernel = _require_blackhole_kernel(spec["kernels"], kind="compute", core_type="trisc")
    compute_source = str(compute_kernel["source_code"])

    compute_cb_ids = {
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if int(cb["cb_id"]) >= 16 and f"({int(cb['cb_id'])}" in compute_source
    }

    assert compute_cb_ids
    assert compute_cb_ids == {cb_id for cb_id in compute_cb_ids if 16 <= cb_id <= 31}


def test_flash_attention_mha_reader_accessors_have_distinct_compile_time_slots():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader_kernels = [kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader"]
    assert len(reader_kernels) == 1
    reader = reader_kernels[0]

    accessors = list(reader["accessors"])
    runtime_arg_buffers = [
        str(arg["buffer"])
        for arg in reader["runtime_args"]
        if str(arg["kind"]) == "input_buffer_addr32"
    ]
    assert len(runtime_arg_buffers) == 3

    input_accessors = [
        accessor for accessor in accessors if str(accessor["buffer"]) in runtime_arg_buffers
    ]
    assert len(input_accessors) == 3

    offsets = sorted(int(accessor["compile_time_arg_offset"]) for accessor in input_accessors)
    counts = {int(accessor["compile_time_arg_count"]) for accessor in input_accessors}
    assert counts == {2}
    assert offsets == [0, 2, 4]


def test_flash_attention_gqa_executable_spec_materializes_all_reader_inputs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                128,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader_kernels = [kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader"]
    assert len(reader_kernels) == 1
    reader = reader_kernels[0]

    runtime_arg_buffers = {
        str(arg["buffer"])
        for arg in reader["runtime_args"]
        if str(arg["kind"]) == "input_buffer_addr32"
    }
    accessor_buffers = {str(accessor["buffer"]) for accessor in reader["accessors"]}
    materialized_buffers = {str(entry["buffer"]) for entry in spec["buffer_materializations"]}

    assert len(runtime_arg_buffers) == 3
    assert accessor_buffers == runtime_arg_buffers
    assert runtime_arg_buffers.issubset(materialized_buffers)


@pytest.mark.parametrize(
    ("example_module", "args", "kwargs"),
    [
        (
            mha_example,
            (1, 32, 128, 128, False),
            {"block_M": 128, "block_N": 128, "num_stages": 1, "threads": 128},
        ),
        (
            gqa_example,
            (1, 16, 128, 128, False),
            {
                "groups": 16,
                "block_M": 64,
                "block_N": 64,
                "num_stages": 2,
                "threads": 128,
            },
        ),
    ],
)
def test_flash_attention_tile_transport_metadata_preserves_tensor_page_size(
    example_module, args, kwargs
):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(example_module.flashattn.jit_impl.get_tir(*args, **kwargs), target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    buffer_materializations = {
        str(entry["buffer"]): int(entry["transport_page_size"])
        for entry in spec["buffer_materializations"]
    }
    assert buffer_materializations == {
        "Q": 2048,
        "K": 2048,
        "V": 2048,
        "Output": 2048,
    }

    reader = _require_blackhole_kernel(spec["kernels"], kind="reader")
    writer = _require_blackhole_kernel(spec["kernels"], kind="writer")
    for accessor in reader["accessors"]:
        assert int(accessor["transport_page_size"]) == 2048
    for accessor in writer["accessors"]:
        assert int(accessor["transport_page_size"]) == 2048


def test_flash_attention_executable_spec_keeps_tile_transport_page_sizes_consistent():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")
    output_accessor = next(
        accessor for accessor in writer["accessors"] if str(accessor["buffer"]) == "Output"
    )
    output_materialization = next(
        entry for entry in spec["buffer_materializations"] if str(entry["buffer"]) == "Output"
    )

    assert int(output_accessor["transport_page_size"]) == 2048
    assert int(output_materialization["transport_page_size"]) == 2048


@pytest.mark.parametrize(
    ("example_module", "args", "kwargs", "required_axis_orders"),
    [
        (
            mha_example,
            (1, 32, 128, 128, False),
            {"block_M": 128, "block_N": 128, "num_stages": 1, "threads": 128},
            {
                "Q": [0, 2, 1, 3],
                "K": [0, 2, 1, 3],
                "V": [0, 2, 1, 3],
                "Output": [0, 2, 1, 3],
            },
        ),
        (
            gqa_example,
            (1, 16, 128, 128, False),
            {
                "groups": 16,
                "block_M": 64,
                "block_N": 64,
                "num_stages": 2,
                "threads": 128,
            },
            {
                "Q": [0, 2, 1, 3],
                "Output": [0, 2, 1, 3],
            },
        ),
    ],
)
def test_flash_attention_buffer_materialization_emits_explicit_host_axis_order(
    example_module, args, kwargs, required_axis_orders
):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(example_module.flashattn.jit_impl.get_tir(*args, **kwargs), target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    actual_axis_orders = {
        str(entry["buffer"]): [int(axis) for axis in entry["host_axis_order"]]
        for entry in spec["buffer_materializations"]
    }
    for buffer, axis_order in required_axis_orders.items():
        assert actual_axis_orders[buffer] == axis_order


def test_flash_attention_bf16_variant_keeps_shared_bridge_cbs_in_bfloat16():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    cb_formats = {str(cb["name"]): str(cb["data_format"]) for cb in spec["cb_configs"]}
    assert cb_formats["Q_shared"] == "Float16_b"
    assert cb_formats["K_shared"] == "Float16_b"
    assert cb_formats["acc_s_cast"] == "Float16_b"
    assert cb_formats["V_shared"] == "Float16_b"
    assert cb_formats["O_shared"] == "Float16_b"


@pytest.mark.parametrize(
    ("example_module", "args", "kwargs"),
    [
        (
            mha_example,
            (1, 32, 128, 128, False),
            {"block_M": 128, "block_N": 128, "num_stages": 1, "threads": 128},
        ),
        (
            gqa_example,
            (1, 16, 128, 128, False),
            {
                "groups": 16,
                "block_M": 64,
                "block_N": 64,
                "num_stages": 2,
                "threads": 128,
            },
        ),
    ],
)
def test_flash_attention_executable_spec_materializes_multi_gemm_contracts_and_reports_contract_gates(
    example_module, args, kwargs
):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(example_module.flashattn.jit_impl.get_tir(*args, **kwargs), target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    reasons = [str(reason) for reason in spec.get("direct_runtime_unsupported_reasons", [])]
    assert not any("missing explicit per-work access descriptor" in reason for reason in reasons)
    assert not reasons

    multi_gemm_contracts = spec.get("multi_gemm_contracts", [])
    multi_compute_contracts = spec.get("multi_compute_contracts", [])
    fragment_merge_ops = [
        op for op in spec.get("compute_epilogue_ops", []) if str(op.get("kind", "")) == "merge_fragment_tiles"
    ]
    projected_contracts = [op["buffer_materialization_contract"] for op in fragment_merge_ops]
    assert len(multi_gemm_contracts) == 2
    assert len(multi_compute_contracts) == 2
    assert len(fragment_merge_ops) == 2
    assert all(str(contract.get("kind", "")) == "gemm" for contract in multi_compute_contracts)
    assert {str(contract["target_buffer"]) for contract in projected_contracts} == {"acc_s", "acc_o"}
    assert all(str(contract["kind"]) == "intermediate_accumulator_merge" for contract in projected_contracts)
    assert all(str(contract["materialization_kind"]) == "intermediate_buffer" for contract in projected_contracts)
    assert all(
        str(contract["bridge_kind"]) == "tile_nfaces_materialization"
        for contract in projected_contracts
    )
    assert all(str(contract["value_role"]) == "accumulator_delta" for contract in projected_contracts)
    assert all(str(contract["merge_kind"]) == "accumulator_add" for contract in projected_contracts)
    assert all(str(contract["execution_protocol"]) == "dst_cb_binary_pack" for contract in projected_contracts)
    reader_kernel = _require_blackhole_kernel(spec["kernels"], kind="reader", core_type="brisc")
    writer_kernel = _require_blackhole_kernel(spec["kernels"], kind="writer", core_type="ncrisc")
    reader_per_work = {
        spec["arg_kind"]: spec["value_kind"] for spec in reader_kernel["per_work_arg_specs"]
    }
    writer_per_work = {
        spec["arg_kind"]: spec["value_kind"] for spec in writer_kernel["per_work_arg_specs"]
    }
    assert reader_per_work["a_tile_start_id"] == "logical_block_y"
    assert reader_per_work["b_tile_start_id"] == "logical_block_x"
    assert reader_per_work["num_k_tiles"] == "gemm_num_k_tiles"
    assert writer_per_work["output_tile_start_id"] == "current_work_linear_id"
    for contract in [*multi_gemm_contracts, *multi_compute_contracts]:
        for field in ("a_buffer", "b_buffer", "c_buffer", "M", "N", "K"):
            assert field in contract


def test_flash_attention_executable_spec_exposes_compute_epilogue_ops():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    epilogue_ops = {
        (
            str(op["kind"]),
            str(op.get("reduce_kind", "")),
        )
        for op in spec.get("compute_epilogue_ops", [])
    }

    assert {
        ("scalar_max", ""),
        ("reduce_row", "max"),
        ("reduce_row", "sum"),
        ("exp2_grouped_row_bcast_affine", ""),
        ("scalar_exp2_affine", ""),
        ("cast_fragment_slice", ""),
    }.issubset(epilogue_ops)
    assert {
        ("mul_grouped_row_bcast", ""),
        ("div_grouped_row_bcast", ""),
        ("scalar_fma", ""),
        ("merge_fragment_tiles", ""),
        ("cast_fragment_slice", ""),
        ("write_local_fragment_slice_to_tiled_cb", ""),
    }.issubset(epilogue_ops)


def test_flash_attention_spatial_program_exposes_buffer_materialization_contracts():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    program = artifact.device_mod["main_kernel"].attrs["tl.spatial_program"]
    materialization_intent = next(
        intent
        for intent in program.resource_intents
        if "buffer_materialization_support" in {str(trait) for trait in intent.traits}
    )
    materialization_contracts = list(materialization_intent.payload["buffer_materialization_contracts"])
    by_buffer = {str(contract["target_buffer"]): contract for contract in materialization_contracts}
    merge_contracts = [by_buffer[name] for name in ("acc_s", "acc_o")]

    assert {"acc_s", "acc_o"}.issubset(by_buffer)
    assert all(str(contract["kind"]) == "intermediate_accumulator_merge" for contract in merge_contracts)
    assert all(
        str(contract["materialization_kind"]) == "intermediate_buffer"
        for contract in merge_contracts
    )
    assert all(
        str(contract["bridge_kind"]) == "tile_nfaces_materialization"
        for contract in merge_contracts
    )
    assert all(str(contract["value_role"]) == "accumulator_delta" for contract in merge_contracts)
    assert all(str(contract["merge_kind"]) == "accumulator_add" for contract in merge_contracts)
    assert all(
        str(contract["execution_protocol"]) == "dst_cb_binary_pack"
        for contract in merge_contracts
    )
    assert all(str(contract["result_live_form"]) == "tiled_cb" for contract in merge_contracts)
    assert all(str(contract["scope"]) == "blackhole.acc" for contract in merge_contracts)
    republish_contract = by_buffer["acc_s_cast"]
    assert str(republish_contract["kind"]) == "republished_logical_tile"
    assert str(republish_contract["materialization_kind"]) == "republished_buffer"
    assert str(republish_contract["bridge_kind"]) == "tile_nfaces_materialization"
    assert str(republish_contract["value_role"]) == "consumer_input"
    assert str(republish_contract["merge_kind"]) == "direct_write"
    assert str(republish_contract["execution_protocol"]) == "tiled_cb_republish"
    assert str(republish_contract["result_live_form"]) == "tiled_cb"
    assert str(republish_contract["source_buffer"]) == "acc_s"
    assert int(republish_contract["logical_row_width"]) == 128
    assert str(republish_contract["scope"]) == "blackhole.acc"


def test_flash_attention_spatial_program_exposes_buffer_flow_contracts():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    program = artifact.device_mod["main_kernel"].attrs["tl.spatial_program"]
    flow_intent = next(
        intent
        for intent in program.resource_intents
        if "buffer_flow_support" in {str(trait) for trait in intent.traits}
    )
    flow_contracts = list(flow_intent.payload["buffer_flow_contracts"])
    by_buffer = {str(contract["buffer"]): contract for contract in flow_contracts}

    assert "acc_s_cast" in by_buffer
    assert str(by_buffer["acc_s_cast"]["flow_class"]) == "republish"
    assert int(by_buffer["acc_s_cast"]["publish_granule"]) == 1
    assert int(by_buffer["acc_s_cast"]["consume_granule"]) == 1
    assert str(by_buffer["acc_s_cast"]["granule_kind"]) == "logical_tile"

    event_kinds = [str(event["kind"]) for event in by_buffer["acc_s_cast"]["events"]]
    assert event_kinds.count("write") >= 2
    assert "compute_consume" in event_kinds


def test_flash_attention_spatial_program_exposes_buffer_distribution_contracts():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    program = artifact.device_mod["main_kernel"].attrs["tl.spatial_program"]
    distribution_intent = next(
        intent
        for intent in program.resource_intents
        if "buffer_distribution_support" in {str(trait) for trait in intent.traits}
    )
    distribution_contracts = list(distribution_intent.payload["buffer_distribution_contracts"])
    by_buffer = {str(contract["buffer"]): contract for contract in distribution_contracts}

    assert {"acc_s", "acc_o", "scores_max", "scores_sum", "logsum", "scores_scale"}.issubset(
        by_buffer
    )

    for name in ("acc_s", "acc_o"):
        contract = by_buffer[name]
        assert str(contract["scope"]) == "blackhole.acc"
        assert str(contract["distribution_kind"]) == "grouped_rows"
        assert str(contract["storage_topology_kind"]) == "linear"
        assert tuple(int(dim) for dim in contract["shape"]) == (128,)

    for name in ("scores_max", "scores_sum", "logsum", "scores_scale"):
        contract = by_buffer[name]
        assert str(contract["scope"]) == "blackhole.acc"
        assert str(contract["distribution_kind"]) == "row_state"
        assert str(contract["storage_topology_kind"]) == "linear"
        assert tuple(int(dim) for dim in contract["shape"]) == (1,)


def test_flash_attention_segment_kernels_prefer_explicit_tile_descriptors_over_work_id():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")

    reader_tile_index_lines = [
        line for line in str(reader["source_code"]).splitlines() if "tile_index =" in line
    ]
    writer_tile_index_lines = [
        line for line in str(writer["source_code"]).splitlines() if "tile_index =" in line
    ]

    assert reader_tile_index_lines
    assert writer_tile_index_lines
    assert any(
        "a_tile_start_id" in line or "b_tile_start_id" in line for line in reader_tile_index_lines
    )
    assert any("output_tile_start_id" in line for line in writer_tile_index_lines)
    assert not any("work_linear_id" in line for line in reader_tile_index_lines)
    assert not any("work_linear_id" in line for line in writer_tile_index_lines)


def test_flash_attention_segment_writer_block_indices_follow_per_work_value_kind():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                128,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
            target=target,
        )

    def mutate(tt_program):
        writer = require_tt_kernel(tt_program, kind="writer", core_type="ncrisc")
        rebuilt_kernels = []
        for kernel in tt_program.kernels:
            payload = dict(kernel.payload)
            if str(kernel.name) == str(writer.name):
                updated_specs = []
                for spec in payload["per_work_arg_specs"]:
                    spec = dict(spec)
                    if str(spec.get("arg_kind", "")) == "output_tile_start_id":
                        spec["value_kind"] = "logical_block_x"
                    updated_specs.append(spec)
                payload["per_work_arg_specs"] = updated_specs
            rebuilt_kernels.append(rebuild_tt_kernel(kernel, payload=payload))
        return rebuild_tt_program(tt_program, kernels=rebuilt_kernels)

    rebuilt = _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)
    spec = rebuilt.get_function_metadata("main")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")
    writer_tile_index_lines = [
        line for line in str(writer["source_code"]).splitlines() if "tile_index =" in line
    ]

    assert writer_tile_index_lines
    assert any("output_tile_start_id" in line for line in writer_tile_index_lines)
    assert not any("%" in line for line in writer_tile_index_lines)
    assert not any(" / " in line for line in writer_tile_index_lines)


def test_flash_attention_segment_reader_tile_transport_matches_compute_input_contract():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader")
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")

    reader_source = str(reader["source_code"])
    compute_source = str(compute["source_code"])

    def count_reader_reads(cb_id: int) -> int:
        return len(re.findall(rf"get_write_ptr\({cb_id}\)", reader_source))

    def count_compute_waited_tiles(cb_id: int) -> int:
        return sum(
            int(tile_count)
            for tile_count in re.findall(rf"cb_wait_front\({cb_id},\s*(\d+)\);", compute_source)
        )

    for cb_id in (0, 1, 3):
        assert count_reader_reads(cb_id) == count_compute_waited_tiles(cb_id)


def test_flash_attention_segment_kernels_keep_buffer_runtime_args_role_local():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader")
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")

    reader_buffer_kinds = {str(arg["kind"]) for arg in reader["runtime_args"] if "buffer" in arg}
    compute_buffer_kinds = {str(arg["kind"]) for arg in compute["runtime_args"] if "buffer" in arg}
    writer_buffer_kinds = {str(arg["kind"]) for arg in writer["runtime_args"] if "buffer" in arg}

    assert reader_buffer_kinds == {"input_buffer_addr32"}
    assert compute_buffer_kinds == set()
    assert writer_buffer_kinds == {"output_buffer_addr32"}


def test_flash_attention_segment_kernels_do_not_leak_compute_resources_into_writer():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")

    compute_source = str(compute["source_code"])
    writer_source = str(writer["source_code"])

    # Fragment state must appear in compute kernel (as CB-backed pointers or array refs)
    assert "acc_o" in compute_source
    assert "scores_max" in compute_source
    assert "acc_s_cast" in compute_source

    # Fragment state must NOT leak into writer kernel
    assert "acc_o" not in writer_source
    assert "scores_max" not in writer_source
    assert "acc_s_cast" not in writer_source
    assert "/* blackhole managed resource */ half" not in writer_source


def test_flash_attention_compute_source_does_not_materialize_fragment_arrays():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "/* blackhole managed resource */ float acc_o[" not in compute_source
    assert "/* blackhole managed resource */ float acc_s[" not in compute_source
    assert "/* blackhole managed resource */ float scores_max[" not in compute_source
    assert "/* blackhole managed resource */ float logsum[" not in compute_source
    assert "/* blackhole managed resource */ float scores_scale[" not in compute_source
    assert "/* blackhole managed resource */ float scores_sum[" not in compute_source
    assert "/* blackhole managed resource */ half acc_s_cast[" not in compute_source
    assert "float* acc_o = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* acc_s = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_max = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* logsum = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_scale = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_sum = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "half* acc_s_cast = reinterpret_cast<half*>(get_local_cb_interface(" not in compute_source
    assert " = exp2f(" not in compute_source
    assert "std::exp2" not in compute_source


def test_flash_attention_compute_serializes_thread_row_offset_in_local_to_cb_materialization():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    dst_offset_lines = [
        line for line in compute_source.splitlines() if "dst_offset_elements =" in line
    ]
    tx_loop_lines = [
        line for line in compute_source.splitlines() if "for (int32_t tx = 0;" in line
    ]

    assert dst_offset_lines
    assert tx_loop_lines
    assert any(re.search(r"dst_offset_elements = .*tx.*128", line) for line in dst_offset_lines)


def test_flash_attention_compute_source_derives_grouped_reduce_rows_from_num_elements():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "const uint32_t num_elements = 16384;" in compute_source
    assert "const uint32_t row_width = 128;" in compute_source
    assert "const uint32_t num_rows = num_elements / row_width;" in compute_source
    assert "const uint32_t num_rows = 16384;" not in compute_source


def test_flash_attention_compute_source_fills_full_matrix_fragments():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_o); const uint32_t num_elements = 16384;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_s); const uint32_t num_elements = 16384;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(logsum); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(scores_max); const uint32_t num_elements = 128;"
        in compute_source
    )


def test_flash_attention_compute_source_materializes_full_acc_s_matrix_into_tiled_cb_before_second_matmul():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert (
        "dst_bits[tiled_index] = tilelang_float_to_half_bits(static_cast<float>(src[src_offset_elements + i]));"
        in compute_source
    )
    assert "const uint32_t num_elements = 16384;" in compute_source


def test_flash_attention_compute_source_preserves_thread_row_offset_in_cast_fragment_slice():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert (
        "const uint32_t src_offset = ((tx * 128) + (i_1 * 8));" in compute_source
    )


def test_flash_attention_compute_source_publishes_acc_s_cast_cb_before_second_matmul():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    acc_s_cb_id = cb_by_name["acc_s_cast"]
    write_ptr = f"tilelang_get_cb_write_ptr_bytes({acc_s_cb_id})"
    cast_pos = compute_source.find(write_ptr)

    reserve_match = re.search(rf"cb_reserve_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    assert reserve_match is not None
    acc_s_tiles = reserve_match.group(1)

    publish_match = re.search(rf"cb_push_back\({acc_s_cb_id}, {acc_s_tiles}\);", compute_source)
    second_mm_match = re.search(rf"mm_init\({acc_s_cb_id}, \d+, \d+\);", compute_source)
    wait_match = re.search(rf"cb_wait_front\({acc_s_cb_id},\s*{acc_s_tiles}\);", compute_source)

    assert cast_pos != -1
    assert publish_match is not None
    assert second_mm_match is not None
    assert wait_match is not None
    assert cast_pos < publish_match.start() < second_mm_match.start()


def test_flash_attention_small_bf16_compute_source_keeps_acc_s_cast_cb_pages_consistent():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                1,
                32,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    acc_s_cb_id = cb_by_name["acc_s_cast"]

    reserve_match = re.search(rf"cb_reserve_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    push_match = re.search(rf"cb_push_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    second_mm_match = re.search(rf"mm_init\({acc_s_cb_id}, \d+, \d+\);", compute_source)

    assert reserve_match is not None
    assert push_match is not None
    assert second_mm_match is not None
    assert reserve_match.group(1) == push_match.group(1)
    assert push_match.start() < second_mm_match.start()


def test_flash_attention_small_bf16_compute_source_uses_explicit_fp32_to_bf16_bitcasts():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                1,
                32,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    output_cb_id = next(
        int(cb["cb_id"]) for cb in spec["cb_configs"] if str(cb["name"]) == "O_shared"
    )

    assert "tilelang_float_to_bfloat_bits" in compute_source
    assert (
        "dst_bits[tiled_index] = tilelang_float_to_bfloat_bits(static_cast<float>(src[src_offset_elements + i]));"
        in compute_source
    )
    assert (
        "MATH({ uint16_t* dst_bits = reinterpret_cast<uint16_t*>(O_shared_local_cast);"
        in compute_source
    )
    assert (
        "{ const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(O_shared_local_cast);"
        in compute_source
    )
    assert f"tilelang_get_cb_write_ptr_bytes({output_cb_id})" in compute_source
    assert f"get_local_cb_interface({output_cb_id}).fifo_wr_ptr" not in compute_source
    assert (
        f"PACK({{ uint16_t* dst_bits = reinterpret_cast<uint16_t*>((get_local_cb_interface({output_cb_id}).fifo_wr_ptr << 4)"
        not in compute_source
    )


def test_flash_attention_small_bf16_compute_source_math_scopes_raw_fragment_helpers():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                1,
                32,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "MATH({ float* dst = reinterpret_cast<float*>(acc_s);" in compute_source
    assert "MATH({ const float* src = reinterpret_cast<const float*>(acc_s);" in compute_source
    assert "MATH({ float* dst = reinterpret_cast<float*>(scores_scale);" in compute_source
    assert "MATH({ float* dst = reinterpret_cast<float*>(acc_o);" in compute_source
    assert "tilelang_div_grouped_row_bcast(dst, scalar, num_elements, row_width); })" in compute_source


def test_flash_attention_small_bf16_compute_source_emits_debug_waypoints_when_requested(monkeypatch):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    monkeypatch.setenv("TILELANG_BLACKHOLE_DEBUG_WAYPOINTS", "1")
    try:
        with target:
            artifact = lower(
                bf16_mha_example.flashattn.jit_impl.get_tir(
                    1,
                    1,
                    32,
                    32,
                    False,
                    block_M=32,
                    block_N=32,
                    num_stages=1,
                    threads=128,
                ),
                target=target,
            )
    finally:
        os.environ.pop("TILELANG_BLACKHOLE_DEBUG_WAYPOINTS", None)

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert 'WAYPOINT("QKAD")' in compute_source
    assert 'WAYPOINT("MXPV")' in compute_source
    assert 'WAYPOINT("MCLR")' in compute_source
    assert 'WAYPOINT("RMAX")' in compute_source
    assert 'WAYPOINT("SMAX")' in compute_source
    assert 'WAYPOINT("SEXP")' in compute_source
    assert 'WAYPOINT("AEXP")' in compute_source
    assert 'WAYPOINT("RSUM")' in compute_source
    assert 'WAYPOINT("LFMA")' in compute_source
    assert 'WAYPOINT("ACST")' in compute_source
    assert 'WAYPOINT("CAST")' in compute_source
    assert 'WAYPOINT("QVAD")' in compute_source
    assert 'WAYPOINT("NORM")' in compute_source
    assert 'WAYPOINT("OCST")' in compute_source
    assert 'WAYPOINT("OWRT")' in compute_source
    assert 'WAYPOINT("OPUB")' in compute_source


def test_flash_attention_seq64_bf16_compute_source_retains_q_cb_until_last_k_step():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    first_mm = re.search(r"mm_init\((\d+), (\d+), (\d+)\);", compute_source)
    assert first_mm is not None
    q_cb_id = first_mm.group(1)
    k_cb_id = first_mm.group(2)

    q_wait = f"cb_wait_front({q_cb_id}, 1);"
    q_pop = f"cb_pop_front({q_cb_id}, 1);"
    k_wait = f"cb_wait_front({k_cb_id}, 1);"
    k_pop = f"cb_pop_front({k_cb_id}, 1);"

    assert compute_source.count(q_wait) == 2
    assert compute_source.count(q_pop) == 1
    assert compute_source.count(k_wait) == 2
    assert compute_source.count(k_pop) == 2

    first_q_wait_pos = compute_source.find(q_wait)
    second_q_wait_pos = compute_source.find(q_wait, first_q_wait_pos + len(q_wait))
    q_pop_pos = compute_source.find(q_pop)
    assert first_q_wait_pos != -1
    assert second_q_wait_pos != -1
    assert q_pop_pos != -1
    assert second_q_wait_pos < q_pop_pos


def test_flash_attention_seq64_bf16_compute_source_reacquires_acc_s_cast_pages_between_k_steps():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    acc_s_cast_cb = cb_by_name["acc_s_cast"]
    reserve = f"cb_reserve_back({acc_s_cast_cb}, 1);"
    push = f"cb_push_back({acc_s_cast_cb}, 1);"
    pop = f"cb_pop_front({acc_s_cast_cb}, 1);"

    assert compute_source.count(reserve) == 2
    assert compute_source.count(push) == 2
    assert compute_source.count(pop) == 2

    first_reserve_pos = compute_source.find(reserve)
    first_push_pos = compute_source.find(push)
    first_pop_pos = compute_source.find(pop)
    second_reserve_pos = compute_source.find(reserve, first_reserve_pos + len(reserve))

    assert first_reserve_pos != -1
    assert first_push_pos != -1
    assert first_pop_pos != -1
    assert second_reserve_pos != -1
    assert first_reserve_pos < first_push_pos < first_pop_pos < second_reserve_pos


def test_flash_attention_seq64_bf16_compute_source_refreshes_acc_s_cast_write_ptr_after_rereserve():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    acc_s_cast_cb = cb_by_name["acc_s_cast"]
    refresh = f"tilelang_get_cb_write_ptr_bytes({acc_s_cast_cb})"

    reserve_matches = list(
        re.finditer(rf"cb_reserve_back\({acc_s_cast_cb}, (\d+)\);", compute_source)
    )
    assert len(reserve_matches) >= 2
    assert reserve_matches[-1].group(1) == "1"

    second_reserve_pos = reserve_matches[-1].start()
    refresh_pos = compute_source.find(refresh, second_reserve_pos)

    assert second_reserve_pos != -1
    assert refresh_pos != -1
    assert second_reserve_pos < refresh_pos


def test_flash_attention_seq64_bf16_cb_plan_allocates_two_pages_for_staged_kv_inputs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    def extract_cb_pages(example_module, *args, **kwargs):
        bf16_example = _load_flash_attention_module_with_dtype(
            example_module.__file__, "T.bfloat16"
        )
        target = Target("blackhole")
        with target:
            artifact = lower(
                bf16_example.flashattn.jit_impl.get_tir(*args, **kwargs),
                target=target,
            )
        spec = _extract_blackhole_executable_spec(artifact)
        return {str(cb["name"]): int(cb["num_pages"]) for cb in spec["cb_configs"]}

    mha_cb_pages = extract_cb_pages(
        mha_example,
        1,
        4,
        64,
        32,
        False,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    assert mha_cb_pages["Q_shared"] == 1
    assert mha_cb_pages["K_shared"] == 2
    assert mha_cb_pages["V_shared"] == 2

    gqa_cb_pages = extract_cb_pages(
        gqa_example,
        1,
        4,
        64,
        32,
        False,
        groups=4,
        block_M=32,
        block_N=32,
        num_stages=1,
        threads=128,
    )
    assert gqa_cb_pages["Q_shared"] == 1
    assert gqa_cb_pages["K_shared"] == 2
    assert gqa_cb_pages["V_shared"] == 2


def test_flash_attention_seq64_bf16_cb_plan_double_buffers_republished_acc_s_cast_input():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    cb_by_name = {str(cb["name"]): cb for cb in spec["cb_configs"]}

    assert int(cb_by_name["acc_s_cast"]["num_pages"]) == 2
    assert str(cb_by_name["acc_s_cast"]["flow_class"]) == "republish"
    assert int(cb_by_name["acc_s_cast"]["publish_pages_per_event"]) == 1
    assert int(cb_by_name["acc_s_cast"]["consume_pages_per_event"]) == 1


def test_flash_attention_seq64_bf16_compute_source_stages_actual_output_pages():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")
    compute_source = str(compute["source_code"])
    writer_source = str(writer["source_code"])

    writer_wait = re.search(r"cb_wait_front\((\d+), 1\);", writer_source)
    assert writer_wait is not None
    output_cb_id = writer_wait.group(1)

    assert f"cb_reserve_back({output_cb_id}, 1);" in compute_source
    assert f"cb_push_back({output_cb_id}, 1);" in compute_source
    assert f"cb_reserve_back({output_cb_id}, 2);" not in compute_source
    assert f"cb_push_back({output_cb_id}, 2);" not in compute_source


def test_flash_attention_seq64_bf16_compute_source_does_not_republish_loop_carried_acc_states():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    scores_max_prev_cb = cb_by_name["scores_max_prev"]
    assert compute_source.count(f"cb_reserve_back({scores_max_prev_cb}, 2);") == 1
    assert f"cb_push_back({scores_max_prev_cb}, 2);" not in compute_source

    for name in ("acc_s", "acc_o"):
        cb_id = cb_by_name[name]
        assert compute_source.count(f"cb_reserve_back({cb_id}, 1);") == 1
        assert f"cb_push_back({cb_id}, 1);" not in compute_source
        assert f"pack_tile(0, {cb_id}, 0);" not in compute_source
        assert f"tilelang_get_cb_write_ptr_bytes({cb_id})" in compute_source


def test_flash_attention_seq64_bf16_compute_source_accumulates_clear_accum_false_gemm_via_tiled_merge_protocol():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                4,
                64,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    assert "tilelang_add_fragment(dst, src, num_elements);" not in compute_source
    assert "add_tiles_init(" in compute_source
    assert "add_tiles(" in compute_source

    merge_pairs = re.findall(r"add_tiles_init\((\d+), (\d+)\);", compute_source)
    assert merge_pairs
    assert all(reload_cb != scratch_cb for reload_cb, scratch_cb in merge_pairs)
    reload_cb, scratch_cb = merge_pairs[0]

    merge_window_pattern = re.compile(
        r"add_tiles_init\(\d+, \d+\);.*?add_tiles\(\d+, \d+, 0, 0, 0\);.*?pack_tile\(0, \d+\);",
        re.DOTALL,
    )
    merge_windows = merge_window_pattern.findall(compute_source)
    assert merge_windows
    assert all("tile_regs_commit()" in window for window in merge_windows)
    assert all("tile_regs_wait()" in window for window in merge_windows)

    assert f"cb_reserve_back({reload_cb}, 1);" in compute_source
    assert f"cb_push_back({reload_cb}, 1);" in compute_source
    assert f"cb_wait_front({reload_cb}, 1);" in compute_source
    assert f"cb_pop_front({reload_cb}, 1);" in compute_source
    assert f"cb_wait_front({scratch_cb}, 1);" in compute_source
    assert f"cb_pop_front({scratch_cb}, 1);" in compute_source
    assert f"pack_tile(0, {reload_cb});" in compute_source or f"pack_tile(0, {reload_cb}, 0);" in compute_source
    assert "get_tile_address(0)" in compute_source


def test_flash_attention_small_bf16_metadata_marks_k_materialization_as_transposed():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    bf16_mha_example = _load_flash_attention_module_with_dtype(
        mha_example.__file__, "T.bfloat16"
    )
    target = Target("blackhole")
    with target:
        artifact = lower(
            bf16_mha_example.flashattn.jit_impl.get_tir(
                1,
                1,
                32,
                32,
                False,
                block_M=32,
                block_N=32,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    by_buffer = {str(item["buffer"]): item for item in spec["buffer_materializations"]}
    assert int(by_buffer["K"].get("transpose_2d", 0)) == 1
    assert int(by_buffer["Q"].get("transpose_2d", 0)) == 0
    assert int(by_buffer["V"].get("transpose_2d", 0)) == 0
    assert int(by_buffer["Output"].get("transpose_2d", 0)) == 0

    reader = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader")
    reader_accessors = {str(item["buffer"]): item for item in reader["accessors"]}
    assert int(reader_accessors["K"].get("transpose_2d", 0)) == 1
    assert int(reader_accessors["Q"].get("transpose_2d", 0)) == 0
    assert int(reader_accessors["V"].get("transpose_2d", 0)) == 0


def test_flash_attention_compute_source_does_not_rereserve_blackhole_acc_gemm_outputs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "tile_regs_wait();\ncb_reserve_back(16, 1);\npack_tile(0, 16);" not in compute_source
    assert "tile_regs_wait();\ncb_reserve_back(17, 1);\npack_tile(0, 17);" not in compute_source


def test_flash_attention_compute_source_hoists_output_cb_staging_around_thread_row_loop():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}

    output_cb_id = cb_by_name["O_shared"]
    reserve_pos = compute_source.find(f"cb_reserve_back({output_cb_id}, 16);")
    tx_loop_pos = compute_source.find("for (int32_t tx = 0; tx < 128; ++tx)")
    push_pos = compute_source.find(f"cb_push_back({output_cb_id}, 16);")

    assert reserve_pos != -1
    assert tx_loop_pos != -1
    assert push_pos != -1
    assert reserve_pos < tx_loop_pos < push_pos
    assert "cb_reserve_back(0, 1);" not in compute_source


def test_flash_attention_compute_source_hoists_thread_invariant_matmul_pipeline_outside_thread_row_loop():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    tx_loop_pos = compute_source.find("for (int32_t tx = 0; tx < 128; ++tx)")
    first_mm_match = re.search(r"mm_init\(0, 1, \d+\);", compute_source)
    second_mm_match = re.search(r"mm_init\(2, 3, \d+\);", compute_source)
    acc_s_publish_match = re.search(r"cb_push_back\(2,\s*\d+\);", compute_source)
    acc_s_publish_pos = -1 if acc_s_publish_match is None else acc_s_publish_match.start()
    first_mm_pos = -1 if first_mm_match is None else first_mm_match.start()
    second_mm_pos = -1 if second_mm_match is None else second_mm_match.start()

    assert tx_loop_pos != -1
    assert first_mm_pos != -1
    assert second_mm_pos != -1
    assert acc_s_publish_pos != -1
    assert first_mm_pos < tx_loop_pos
    assert second_mm_pos < tx_loop_pos
    assert acc_s_publish_pos < tx_loop_pos
    assert re.search(r"mm_init\(0, 1, \d+\);", compute_source[tx_loop_pos:]) is None
    assert re.search(r"mm_init\(2, 3, \d+\);", compute_source[tx_loop_pos:]) is None
    assert re.search(r"cb_wait_front\(0,\s*\d+\);", compute_source[tx_loop_pos:]) is None
    assert re.search(r"cb_wait_front\(2,\s*\d+\);", compute_source[tx_loop_pos:]) is None
    assert re.search(r"cb_push_back\(2,\s*\d+\);", compute_source[tx_loop_pos:]) is None


def test_flash_attention_compute_source_keeps_multitile_blackhole_acc_cb_layout_for_multiphase_mha():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}
    acc_s_cb = cb_by_name["acc_s"]
    acc_o_cb = cb_by_name["acc_o"]

    assert f"cb_reserve_back({acc_s_cb}, 16);" in compute_source
    assert f"cb_reserve_back({acc_o_cb}, 16);" in compute_source
    assert f"cb_push_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_push_back({acc_o_cb}, 16);" not in compute_source
    assert f"cb_reserve_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_reserve_back({acc_o_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_o_cb}, 1);" not in compute_source


def test_flash_attention_forward_rejects_unsupported_pipeline_stage_count():
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole compute pipeline legality: unsupported stage count 4",
    ):
        target = Target("blackhole")
        with target:
            lower(
                gqa_example.flashattn.jit_impl.get_tir(
                    1,
                    16,
                    1024,
                    128,
                    False,
                    groups=16,
                    block_M=64,
                    block_N=64,
                    num_stages=4,
                    threads=128,
                ),
                target=target,
            )
