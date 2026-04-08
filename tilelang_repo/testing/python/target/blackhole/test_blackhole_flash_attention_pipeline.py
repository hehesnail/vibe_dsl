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
    lower_blackhole_ops_through_phase_b,
    lower_blackhole_to_tt_target,
    require_tt_kernel,
    require_tt_program,
    tt_abi_for_kernel,
)
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
    _require_blackhole_kernel,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example
import example_gqa_fwd_bshd as gqa_example


def _lower_flash_attention_through_blackhole_ops(*, is_causal=False):
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
    mod = lower_blackhole_ops_through_phase_b(mod)
    return mod["main"]


def _run_flash_attention_lower_blackhole_ops(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
    return lower_blackhole_ops_through_phase_b(mod)


def _run_flash_attention_lower_blackhole_ops_after_optimize(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    return lower_blackhole_ops_through_phase_b(mod)


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


def test_flash_attention_forward_lower_blackhole_ops_emits_generic_lowering_requirements():
    lowered = _lower_flash_attention_through_blackhole_ops()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs

    lowering_requirements = attrs["blackhole.lowering_requirements"]
    assert list(lowering_requirements["work_axes"]) == ["bx", "by", "bz"]
    assert {
        "gemm",
        "pointwise_chain",
    }.issubset(set(lowering_requirements["fragment_op_kinds"]))
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])
    assert "row_broadcast_sources" not in lowering_requirements
    assert {"exp2", "mul", "div"}.issubset(
        set(lowering_requirements["pointwise_op_kinds"])
    )
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "add" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_scores_max_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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


def test_flash_attention_forward_lower_blackhole_ops_lowers_row_reductions_to_builtins():
    lowered = _lower_flash_attention_through_blackhole_ops()
    script = lowered.script()

    assert "tl.blackhole.reduce_row" in script
    assert "\"max\"" in script
    assert "\"sum\"" in script


def test_flash_attention_forward_optimized_path_lowers_row_reductions_to_builtins():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    assert "row_reduction" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_acc_o_row_broadcast_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    assert "tl.blackhole.div_row_bcast" in script


def test_flash_attention_forward_optimized_path_lowers_logsum_scalar_fma():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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

    assert "tl.blackhole.exp2_row_bcast_affine" in script
    assert "tl.blackhole.scalar_exp2_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_fills():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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

    assert "tl.blackhole.write_local_slice_to_cb" in script
    assert "O_shared_1[tx" not in script


def test_flash_attention_forward_runtime_shape_lowers_local_to_cb_with_thread_offset():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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

    assert "tl.blackhole.write_local_slice_to_cb" in script
    assert "tx * 128" in script


def test_flash_attention_gqa_optimized_path_lowers_grouped_row_broadcasts():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.mul_grouped_row_bcast" in script
    assert "tl.blackhole.div_grouped_row_bcast" in script
    assert "tl.blackhole.exp2_grouped_row_bcast_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


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


def test_flash_attention_forward_pipeline_lifts_semantic_roles_without_workload_specific_schema():
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

    program = artifact.device_mod["main_kernel"].attrs["tl.semantic_program"]
    state_roles = {str(state.role) for state in program.states}
    law_kinds = {str(update.law.kind) for update in program.updates}

    assert {"carry", "reduction_accumulator", "transient"}.issubset(state_roles)
    assert "recurrence" in law_kinds


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
    assert set(lowering_requirements["spatial_phase_boundary_states"]) == {"scores_max"}


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
def test_flash_attention_executable_spec_materializes_multi_gemm_contracts_without_runtime_blocker(
    example_module, args, kwargs
):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(example_module.flashattn.jit_impl.get_tir(*args, **kwargs), target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    assert [str(reason) for reason in spec.get("direct_runtime_unsupported_reasons", [])] == []

    multi_gemm_contracts = spec.get("multi_gemm_contracts", [])
    multi_compute_contracts = spec.get("multi_compute_contracts", [])
    assert len(multi_gemm_contracts) == 2
    assert len(multi_compute_contracts) == 2
    assert all(str(contract.get("kind", "")) == "gemm" for contract in multi_compute_contracts)
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
        ("exp2_row_bcast_affine", ""),
        ("scalar_exp2_affine", ""),
        ("cast_fragment_slice", ""),
    }.issubset(epilogue_ops)
    assert {
        ("mul_row_bcast", ""),
        ("div_row_bcast", ""),
        ("scalar_fma", ""),
        ("cast_fragment_slice", ""),
        ("write_local_slice_to_cb", ""),
    }.issubset(epilogue_ops)


def test_flash_attention_segment_kernels_preserve_runtime_work_index_in_tile_transport():
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
    assert any("work_linear_id" in line for line in reader_tile_index_lines)
    assert any("work_linear_id" in line for line in writer_tile_index_lines)


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
    assert "const uint32_t num_rows = row_width == 0 ? 0 : (num_elements / row_width);" in compute_source
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
        "{ float* dst = reinterpret_cast<float*>(acc_o); const uint32_t num_elements = 16384;"
        in compute_source
    )
    assert (
        "{ float* dst = reinterpret_cast<float*>(acc_s); const uint32_t num_elements = 16384;"
        in compute_source
    )
    assert (
        "{ float* dst = reinterpret_cast<float*>(logsum); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert (
        "{ float* dst = reinterpret_cast<float*>(scores_max); const uint32_t num_elements = 128;"
        in compute_source
    )


def test_flash_attention_compute_source_casts_full_acc_s_matrix_before_second_matmul():
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
        "{ uint16_t* dst_bits = reinterpret_cast<uint16_t*>(acc_s_cast);"
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

    acc_s_cb_match = re.search(
        r"half\* acc_s_cast = reinterpret_cast<half\*>\(tilelang_get_cb_write_ptr_bytes\((\d+)\)\);",
        compute_source,
    )
    cast_pos = compute_source.find(
        "{ half* dst = reinterpret_cast<half*>(acc_s_cast);"
    )
    assert acc_s_cb_match is not None
    acc_s_cb_id = acc_s_cb_match.group(1)

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

    acc_s_cb_match = re.search(
        r"__bf16\* acc_s_cast = reinterpret_cast<__bf16\*>\(tilelang_get_cb_write_ptr_bytes\((\d+)\)\);",
        compute_source,
    )
    assert acc_s_cb_match is not None
    acc_s_cb_id = acc_s_cb_match.group(1)

    reserve_match = re.search(rf"cb_reserve_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    push_match = re.search(rf"cb_push_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    second_mm_match = re.search(rf"mm_init\({acc_s_cb_id}, \d+, \d+\);", compute_source)

    assert reserve_match is not None
    assert push_match is not None
    assert second_mm_match is not None
    assert reserve_match.group(1) == push_match.group(1)
    assert push_match.start() < second_mm_match.start()


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

    reserve_pos = compute_source.find("cb_reserve_back(23, 16);")
    tx_loop_pos = compute_source.find("for (int32_t tx = 0; tx < 128; ++tx)")
    push_pos = compute_source.find("cb_push_back(23, 16);")

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
    first_mm_pos = compute_source.find("mm_init(0, 1, 16);")
    second_mm_pos = compute_source.find("mm_init(2, 3, 17);")
    acc_s_publish_match = re.search(r"cb_push_back\(2,\s*\d+\);", compute_source)
    acc_s_publish_pos = -1 if acc_s_publish_match is None else acc_s_publish_match.start()

    assert tx_loop_pos != -1
    assert first_mm_pos != -1
    assert second_mm_pos != -1
    assert acc_s_publish_pos != -1
    assert first_mm_pos < tx_loop_pos
    assert second_mm_pos < tx_loop_pos
    assert acc_s_publish_pos < tx_loop_pos
    assert compute_source.find("mm_init(0, 1, 16);", tx_loop_pos) == -1
    assert compute_source.find("mm_init(2, 3, 17);", tx_loop_pos) == -1
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

    assert "cb_reserve_back(16, 16);" in compute_source
    assert "cb_reserve_back(17, 16);" in compute_source
    assert "cb_push_back(16, 16);" in compute_source
    assert "cb_push_back(17, 16);" in compute_source
    assert "cb_reserve_back(16, 1);" not in compute_source
    assert "cb_reserve_back(17, 1);" not in compute_source
    assert "cb_push_back(16, 1);" not in compute_source
    assert "cb_push_back(17, 1);" not in compute_source


def test_flash_attention_forward_rejects_unsupported_pipeline_stage_count():
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole fragment pipeline legality: unsupported stage count 4",
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
