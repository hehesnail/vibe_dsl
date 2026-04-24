import os
import re
import sys
import types
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import _align_blackhole_device_symbol, lower
from tilelang.engine.phase import LowerAndLegalize
from tilelang.engine.phase import LowerToBlackholePhaseB
from tilelang.engine.phase import LowerToBlackholeTTProgram
from tilelang.engine.phase import OptimizeForTarget
from tilelang import tvm
from tvm.target import Target

from .common import (
    check_blackhole_codegen_requirements,
    lower_blackhole_to_tt_target,
    rebuild_tt_abi_plan,
    rebuild_tt_kernel,
    rebuild_tt_program,
    require_tt_kernel,
    require_tt_program,
    tt_abi_for_kernel,
)
from .test_blackhole_copy_pipeline import (
    _collect_blackhole_builtin_names,
    _rebuild_codegen_module_with_tt_program,
    _extract_blackhole_executable_spec,
    _require_blackhole_kernel,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example
import example_gqa_fwd_bshd as gqa_example


HELPER_COMPOSITE_BLACKHOLE_BUILTINS = (
    "reduce_row",
    "mul_row_bcast",
    "mul_grouped_row_bcast",
    "div_row_bcast",
    "div_grouped_row_bcast",
    "exp2_row_bcast_affine",
    "exp2_grouped_row_bcast_affine",
    "scalar_max",
    "scalar_exp2_affine",
    "copy_tile_from_cb",
)

LEGACY_LOCAL_BLACKHOLE_BUILTINS = (
    "binary_max_tile_local",
    "reduce_rows_local",
    "mul_tiles_bcast_rows_local",
    "div_tiles_bcast_rows_local",
    "exp_tiles_bcast_rows_affine_local",
    "exp_tile_affine_local",
)


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
        phase_b_mod = LowerToBlackholePhaseB(mod)
        mod = OptimizeForTarget(mod, target)
    mod = _align_blackhole_device_symbol(phase_b_mod, mod)
    return LowerToBlackholeTTProgram(mod)


def _load_flash_attention_module_with_dtype(module_path, dtype_expr):
    source = Path(module_path).read_text()
    source = source.replace("dtype = T.float16", f"dtype = {dtype_expr}", 1)
    mutated = types.ModuleType(f"{Path(module_path).stem}_{dtype_expr.replace('.', '_')}")
    mutated.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), mutated.__dict__)
    return mutated


def _tt_program_payload(func):
    return dict(require_tt_program(func).payload)


def _assert_no_executable_contract_family(spec):
    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in spec


def _strip_tt_program_accessor_host_axis_order(tt_program):
    def strip_accessors(accessors):
        stripped = []
        for accessor in accessors:
            item = dict(accessor)
            item.pop("host_axis_order", None)
            stripped.append(item)
        return stripped

    def strip_compile_time_arg_specs(specs):
        stripped = []
        for spec in specs:
            item = dict(spec)
            item.pop("host_axis_order", None)
            stripped.append(item)
        return stripped

    rebuilt_abi_plans = list(tt_program.abi_plans)
    rebuilt_kernels = []
    for kernel in tt_program.kernels:
        abi_plan_index = int(kernel.abi_plan_index)
        abi = tt_program.abi_plans[abi_plan_index]
        accessors = strip_accessors(list(abi.accessors))
        compile_time_arg_specs = strip_compile_time_arg_specs(
            list(abi.compile_time_arg_specs)
        )
        payload = dict(kernel.payload)
        if "accessors" in payload:
            payload["accessors"] = strip_accessors(list(payload["accessors"]))
        if "compile_time_arg_specs" in payload:
            payload["compile_time_arg_specs"] = strip_compile_time_arg_specs(
                list(payload["compile_time_arg_specs"])
            )
        rebuilt_abi_plans[abi_plan_index] = rebuild_tt_abi_plan(
            abi, accessors=accessors, compile_time_arg_specs=compile_time_arg_specs
        )
        rebuilt_kernels.append(rebuild_tt_kernel(kernel, payload=payload))

    return rebuild_tt_program(
        tt_program, kernels=rebuilt_kernels, abi_plans=rebuilt_abi_plans
    )


def _blackhole_builtin_suffixes(func):
    return {
        name.split("tl.blackhole.", 1)[1]
        for name in _collect_blackhole_builtin_names(func)
    }


def _buffer_tile_bridge_specs_by_buffer(func):
    payload = _tt_program_payload(func)
    return {
        str(spec["buffer"]): dict(spec)
        for spec in payload.get("buffer_tile_bridge_specs", [])
    }


def test_flash_attention_forward_tt_target_emits_tt_program_payload():
    lowered = _lower_flash_attention_to_tt_target()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs
    assert "blackhole.lowering_requirements" not in attrs

    payload = _tt_program_payload(lowered)
    bridge_buffers = {
        str(spec["buffer"]) for spec in payload["buffer_tile_bridge_specs"]
    }
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert len(payload["multi_compute_contracts"]) == 2
    assert all(
        str(contract["kind"]) == "gemm"
        for contract in payload["multi_compute_contracts"]
    )
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(bridge_buffers)
    assert {
        "binary_max_tile_init",
        "binary_max_tile",
        "reduce_init",
        "reduce_tile",
        "reduce_uninit",
        "mul_tiles_init",
        "mul_tiles",
        "add_tiles_init",
        "add_tiles",
        "mul_bcast_rows_init_short",
        "mul_tiles_bcast_rows",
        "add_bcast_rows_init_short",
        "add_tiles_bcast_rows",
        "exp2_tile_init",
        "exp2_tile",
        "pack_tile",
        "cast_fragment_slice",
        "tilize_local_fragment_slice",
        "tilize_cast_fragment_slice",
        "untilize_cb_front_tile_fragment",
    }.issubset(builtin_names)
    assert not any(name in LEGACY_LOCAL_BLACKHOLE_BUILTINS for name in builtin_names)


def test_flash_attention_tt_program_projects_two_typed_gemm_compute_ops():
    lowered = _lower_flash_attention_to_tt_target()
    tt_program = require_tt_program(lowered)

    gemm_ops = [
        op for op in tt_program.compute_op_plans if str(op.kind) == "gemm"
    ]
    assert len(gemm_ops) == 2
    operand_sets = []
    for op in gemm_ops:
        assert str(op.kernel_name) == "compute"
        assert int(op.kernel_plan_index) >= 0
        assert tuple(str(axis) for axis in op.problem_shape_axes) == ("M", "N", "K")
        assert all(int(dim) > 0 for dim in op.problem_shape)
        assert all(int(dim) > 0 for dim in op.tile_shape)
        operands = {str(binding.role): binding for binding in op.operand_bindings}
        assert {"a", "b", "c"} == set(operands)
        operand_sets.append(
            tuple(str(operands[role].buffer) for role in ("a", "b", "c"))
        )

    assert ("Q_shared", "K_shared", "acc_s") in operand_sets
    assert ("acc_s_cast", "V_shared", "acc_o") in operand_sets

    mod = tvm.IRModule({"main": lowered})
    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    assert len(executable["compute_op_plans"]) == 2
    compute_segments = [
        segment
        for segment in executable["segment_plan"]
        if str(segment["kind"]) == "compute"
    ]
    assert len(compute_segments) == 1
    assert len(compute_segments[0]["compute_ops"]) == 2


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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "binary_max_tile" in builtin_names
    assert "binary_max_tile_local" not in builtin_names


def test_flash_attention_forward_tt_target_lowers_reductions_to_builtins():
    lowered = _lower_flash_attention_to_tt_target()
    builtin_names = _blackhole_builtin_suffixes(lowered)
    assert "reduce_init" in builtin_names
    assert "reduce_tile" in builtin_names
    assert "reduce_uninit" in builtin_names
    assert "reduce_rows_local" not in builtin_names


def test_flash_attention_forward_optimized_path_lowers_reductions_to_builtins():
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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "reduce_init" in builtin_names
    assert "reduce_tile" in builtin_names
    assert "reduce_uninit" in builtin_names
    assert "reduce_rows_local" not in builtin_names


def test_flash_attention_forward_optimized_path_lowers_acc_o_broadcast_updates():
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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "mul_tiles_bcast_rows" in builtin_names
    assert "recip_tile_init" in builtin_names
    assert "recip_tile" in builtin_names
    assert "mul_tiles_bcast_rows_local" not in builtin_names
    assert "div_tiles_bcast_rows_local" not in builtin_names


def test_flash_attention_forward_optimized_path_lowers_logsum_to_exact_tile_binary_ops():
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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "mul_tiles_init" in builtin_names
    assert "mul_tiles" in builtin_names
    assert "add_tiles_init" in builtin_names
    assert "add_tiles" in builtin_names
    assert "scalar_fma" not in builtin_names


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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "exp2_tile" in builtin_names
    assert "exp2_tile_init" in builtin_names
    assert "exp_tiles_bcast_rows_affine_local" not in builtin_names
    assert "exp_tile_affine_local" not in builtin_names


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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "fill_fragment" in builtin_names
    assert "compute_epilogue_ops" not in _tt_program_payload(lowered)


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
    builtin_names = _blackhole_builtin_suffixes(lowered)

    assert "cast_fragment_slice" in builtin_names
    assert "tilize_cast_fragment_slice" in builtin_names


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
    builtin_names = _blackhole_builtin_suffixes(lowered)
    script = lowered.script()

    assert "tilize_local_fragment_slice" in builtin_names
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
    builtin_names = _blackhole_builtin_suffixes(lowered)
    script = lowered.script()

    assert "tilize_local_fragment_slice" in builtin_names
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


def test_flash_attention_gqa_aggregated_runtime_args_cover_segment_buffers():
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


def test_flash_attention_forward_pipeline_keeps_plan_and_tt_program_only():
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
    plan = device_func.attrs["tl.spatial_plan"]
    payload = _tt_program_payload(device_func)
    bridge_buffers = {
        str(spec["buffer"]) for spec in payload["buffer_tile_bridge_specs"]
    }
    assert device_func.attrs.get("tl.spatial_program") is None
    assert device_func.attrs.get("blackhole.lowering_requirements") is None
    assert len(plan.execution_units) >= 2
    assert len(plan.dataflow_edges) >= 1
    assert len(plan.phase_plans) >= 1
    assert "scores_max" in bridge_buffers


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


def test_flash_attention_build_rejects_missing_explicit_host_axis_order():
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

    with pytest.raises(
        tvm.TVMError,
        match="Blackhole buffer materialization requires explicit host_axis_order",
    ):
        _rebuild_codegen_module_with_tt_program(
            artifact,
            tt_program_mutator=_strip_tt_program_accessor_host_axis_order,
        )


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
def test_flash_attention_executable_spec_drops_contract_family_and_reports_contract_gates(
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
    assert any("thread-distributed cb_republish materialization" in reason for reason in reasons)

    _assert_no_executable_contract_family(spec)
    reader_kernel = _require_blackhole_kernel(spec["kernels"], kind="reader", core_type="brisc")
    writer_kernel = _require_blackhole_kernel(spec["kernels"], kind="writer", core_type="ncrisc")
    reader_per_work = {
        (str(spec["descriptor_kind"]), str(spec["value_source"]))
        for spec in reader_kernel["per_work_arg_specs"]
    }
    writer_per_work = {
        (str(spec["descriptor_kind"]), str(spec["value_source"]))
        for spec in writer_kernel["per_work_arg_specs"]
    }
    assert ("tile_start", "logical_block_y") in reader_per_work
    assert ("tile_start", "logical_block_x") in reader_per_work
    assert ("k_tile_count", "compute_op_num_k_tiles") in reader_per_work
    assert ("tile_start", "work_linear_id") in writer_per_work


def test_flash_attention_executable_spec_drops_contract_family():
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
    _assert_no_executable_contract_family(spec)


def test_flash_attention_no_compute_epilogue_payload():
    lowered = _lower_flash_attention_to_tt_target()
    payload = _tt_program_payload(lowered)
    spec = lower(
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
        target=Target("blackhole"),
    )
    executable_spec = _extract_blackhole_executable_spec(spec)
    builtin_names = _collect_blackhole_builtin_names(lowered)

    assert "compute_epilogue_ops" not in payload
    assert "compute_epilogue_ops" not in executable_spec
    for builtin_name in HELPER_COMPOSITE_BLACKHOLE_BUILTINS:
        assert f"tl.blackhole.{builtin_name}" not in builtin_names


def test_flash_attention_tt_program_payload_keeps_buffer_tile_bridge_specs_for_internal_buffers():
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

    device_func = artifact.device_mod["main_kernel"]
    payload = _tt_program_payload(device_func)
    bridge_specs = _buffer_tile_bridge_specs_by_buffer(device_func)

    assert "buffer_materialization_contracts" not in payload
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(bridge_specs)
    for name in ("acc_s", "acc_s_cast", "acc_o"):
        spec = bridge_specs[name]
        assert str(spec["scope"]) == "local"
        assert tuple(int(dim) for dim in spec["shape"]) == (128, 128)
        assert tuple(int(dim) for dim in spec["local_shape"]) == (128,)
        assert int(spec["thread_extent"]) == 128
        assert int(spec["replicate_extent"]) == 1
        assert len(spec["inverse_logical_index_exprs"]) == 3
    for name in ("scores_max", "scores_max_prev", "scores_scale", "scores_sum", "logsum"):
        spec = bridge_specs[name]
        assert str(spec["scope"]) == "local"
        assert tuple(int(dim) for dim in spec["shape"]) == (128,)
        assert tuple(int(dim) for dim in spec["local_shape"]) == (1,)
        assert int(spec["thread_extent"]) == 128
        assert int(spec["replicate_extent"]) == 1
        assert len(spec["inverse_logical_index_exprs"]) == 2


def test_flash_attention_tt_program_payload_keeps_republished_buffer_bridge_specs():
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

    device_func = artifact.device_mod["main_kernel"]
    payload = _tt_program_payload(device_func)
    bridge_specs = _buffer_tile_bridge_specs_by_buffer(device_func)

    assert "buffer_flow_contracts" not in payload
    assert {"acc_s", "acc_s_cast", "acc_o"}.issubset(bridge_specs)

    acc_s_spec = bridge_specs["acc_s"]
    acc_s_cast_spec = bridge_specs["acc_s_cast"]
    assert str(acc_s_cast_spec["scope"]) == "local"
    assert tuple(int(dim) for dim in acc_s_cast_spec["shape"]) == (32, 32)
    assert tuple(int(dim) for dim in acc_s_cast_spec["local_shape"]) == (8,)
    assert int(acc_s_cast_spec["thread_extent"]) == 128
    assert int(acc_s_cast_spec["replicate_extent"]) == 1
    assert tuple(int(dim) for dim in acc_s_spec["shape"]) == tuple(
        int(dim) for dim in acc_s_cast_spec["shape"]
    )
    assert tuple(int(dim) for dim in acc_s_spec["local_shape"]) == tuple(
        int(dim) for dim in acc_s_cast_spec["local_shape"]
    )


def test_flash_attention_tt_program_payload_keeps_bridge_specs_without_distribution_contracts():
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

    payload = _tt_program_payload(artifact.device_mod["main_kernel"])
    assert "buffer_distribution_contracts" not in payload
    assert "buffer_tile_bridge_specs" in payload
    bridge_buffers = {
        str(spec["buffer"]) for spec in payload["buffer_tile_bridge_specs"]
    }
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(bridge_buffers)


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


def test_flash_attention_segment_writer_block_indices_follow_per_work_value_source():
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
                        spec["value_source"] = "logical_block_x"
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

    reader_buffer_kinds = {
        str(arg["kind"])
        for arg in reader.get("runtime_args", [])
        if "kind" in arg and "buffer_addr" in str(arg["kind"])
    }
    compute_buffer_kinds = {
        str(arg["kind"])
        for arg in compute.get("runtime_args", [])
        if "kind" in arg and "buffer_addr" in str(arg["kind"])
    }
    writer_buffer_kinds = {
        str(arg["kind"])
        for arg in writer.get("runtime_args", [])
        if "kind" in arg and "buffer_addr" in str(arg["kind"])
    }

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
    assert "reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>" in compute_source
    assert "reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>" in compute_source
    assert "reduce_uninit<true>()" in compute_source
    assert "tilelang_reduce_grouped_row_max" not in compute_source
    assert "const uint32_t num_rows = num_elements / row_width;" not in compute_source


def test_flash_attention_compute_source_fills_thread_distributed_fragment_slices():
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

    assert "/* scope: blackhole.acc */ float acc_o[128];" in compute_source
    assert "/* scope: blackhole.acc */ float acc_s[128];" in compute_source
    assert "/* scope: blackhole.acc */ float logsum[1];" in compute_source
    assert "/* scope: blackhole.acc */ float scores_max[1];" in compute_source
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_o); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_s); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(logsum); const uint32_t num_elements = 1;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(scores_max); const uint32_t num_elements = 1;"
        in compute_source
    )
    assert "const uint32_t num_elements = 16384; const uint32_t row_width = 128;" in compute_source


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

    assert re.search(
        r"dst_bits\[tiled_index\] = tilelang_float_to_half_bits"
        r"\(static_cast<float>\(src\[src_offset_elements \+ [A-Za-z0-9_]+\]\)\);",
        compute_source,
    )
    assert "const uint32_t num_elements = 16384;" in compute_source


def test_flash_attention_compute_source_keeps_thread_row_offset_shape_in_cast_fragment_slice():
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

    assert re.search(
        r"const uint32_t src_offset = \(\(tx \* 128\) \+ "
        r"\([A-Za-z0-9_]+ \* 8\)\);",
        compute_source,
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

    publish_match = re.search(rf"cb_push_back\({acc_s_cb_id}, \d+\);", compute_source)
    second_mm_match = re.search(rf"mm_init\({acc_s_cb_id}, \d+, \d+\);", compute_source)
    wait_match = re.search(rf"cb_wait_front\({acc_s_cb_id},\s*\d+\);", compute_source)
    second_mm_issue_match = re.search(rf"matmul_tiles\({acc_s_cb_id}, \d+,", compute_source)

    assert cast_pos != -1
    assert reserve_match is not None
    assert publish_match is not None
    assert second_mm_match is not None
    assert wait_match is not None
    assert second_mm_issue_match is not None
    assert reserve_match.start() < cast_pos < publish_match.start() < second_mm_match.start()
    assert second_mm_match.start() < wait_match.start() < second_mm_issue_match.start()


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
    assert re.search(
        r"dst_bits\[tiled_index\] = tilelang_float_to_bfloat_bits"
        r"\(static_cast<float>\(src\[src_offset_elements \+ [A-Za-z0-9_]+\]\)\);",
        compute_source,
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
    assert "reinterpret_cast<const uint32_t*>(acc_s)" in compute_source
    assert "reinterpret_cast<const uint32_t*>(scores_scale)" in compute_source
    assert "reinterpret_cast<const uint32_t*>(logsum)" in compute_source
    assert "MATH({ float* dst = reinterpret_cast<float*>(acc_o);" in compute_source
    assert "recip_tile_init();" in compute_source
    assert "recip_tile(0);" in compute_source
    assert "mul_bcast_rows_init_short" in compute_source
    assert "mul_tiles_bcast_rows" in compute_source
    assert "tilelang_div_grouped_row_bcast" not in compute_source


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

    assert "WAYPOINT(" in compute_source
    assert 'WAYPOINT("CAST")' in compute_source
    for tag in ("MXPV", "MCLR", "OCST", "QKAD", "QVAD", "ACST"):
        assert f'WAYPOINT("{tag}")' not in compute_source


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

    assert compute_source.count(reserve) >= 3
    assert compute_source.count(push) == 2
    assert compute_source.count(pop) == 2

    first_reserve_pos = compute_source.find(reserve)
    first_push_pos = compute_source.find(push)
    first_pop_pos = compute_source.find(pop)
    second_reserve_pos = compute_source.find(reserve, first_pop_pos + len(pop))

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
    cb_by_name = {str(cb["name"]): cb for cb in spec["cb_configs"]}
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")
    compute_source = str(compute["source_code"])
    writer_source = str(writer["source_code"])
    output_cb = cb_by_name["O_shared"]
    output_cb_id = int(output_cb["cb_id"])

    assert str(output_cb["flow_class"]) == "republish"
    assert int(output_cb["publish_pages_per_event"]) == 1
    assert int(output_cb["consume_pages_per_event"]) == 1
    assert f"cb_reserve_back({output_cb_id}, 1);" in compute_source
    assert f"cb_push_back({output_cb_id}, 1);" in compute_source
    assert f"cb_reserve_back({output_cb_id}, 2);" not in compute_source
    assert f"cb_push_back({output_cb_id}, 2);" not in compute_source
    assert f"cb_pop_front({output_cb_id}, 1);" in writer_source
    assert f"cb_pop_front({output_cb_id}, 2);" not in writer_source


def test_flash_attention_seq64_bf16_compute_source_keeps_acc_s_private_and_acc_o_single_tile_state_publication():
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
    acc_s_cb = cb_by_name["acc_s"]
    acc_o_cb = cb_by_name["acc_o"]
    assert f"cb_push_back({scores_max_prev_cb}, 2);" not in compute_source

    assert f"cb_push_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 16);" not in compute_source
    assert f"pack_tile(0, {acc_s_cb}, 0);" not in compute_source
    assert f"pack_tile(0, {acc_s_cb});" not in compute_source

    assert f"cb_reserve_back({acc_o_cb}, 1);" in compute_source
    assert f"cb_push_back({acc_o_cb}, 1);" in compute_source
    assert f"pack_tile(0, {acc_o_cb});" in compute_source
    assert f"cb_reserve_back({acc_o_cb}, 16);" not in compute_source
    assert f"cb_push_back({acc_o_cb}, 16);" not in compute_source


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

    merge_window_pattern = re.compile(
        r"add_tiles_init\(\d+, \d+\);.*?add_tiles\(\d+, \d+, 0, 0, 0\);.*?pack_tile\(0, \d+\);",
        re.DOTALL,
    )
    merge_windows = merge_window_pattern.findall(compute_source)
    assert merge_windows
    assert all("tile_regs_commit()" in window for window in merge_windows)
    assert all("tile_regs_wait()" in window for window in merge_windows)

    merge_cb_ids = {cb_id for pair in merge_pairs for cb_id in pair}
    for cb_id in merge_cb_ids:
        assert f"cb_reserve_back({cb_id}, 1);" in compute_source
        assert f"cb_push_back({cb_id}, 1);" in compute_source
        assert f"cb_wait_front({cb_id}, 1);" in compute_source
        assert f"cb_pop_front({cb_id}, 1);" in compute_source
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


def test_flash_attention_compute_source_keeps_multiphase_acc_cb_layout_for_mha():
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
    reload_cb = next(
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["name"]).startswith("acc_s_fragment_merge_reload")
    )
    live_form_cb = next(
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["name"]).startswith("acc_s_fragment_merge_live_form")
    )

    assert f"cb_reserve_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_reserve_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_reserve_back({acc_o_cb}, 16);" in compute_source
    assert f"cb_push_back({acc_o_cb}, 16);" in compute_source
    assert f"cb_reserve_back({acc_o_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_o_cb}, 1);" not in compute_source
    assert reload_cb not in (acc_s_cb, acc_o_cb, live_form_cb)
    assert f"cb_reserve_back({live_form_cb}, 16);" in compute_source
    assert f"cb_push_back({live_form_cb}, 16);" in compute_source
    assert "float acc_s[" in compute_source
    assert "float acc_o[" in compute_source


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
