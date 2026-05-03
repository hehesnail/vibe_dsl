import os
import re
import sys
import types
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import _align_blackhole_device_symbol, get_device_call, lower
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
    tt_accessor_specs_to_list,
    tt_compile_time_arg_specs_to_list,
    tt_per_work_arg_specs_to_list,
    tt_runtime_arg_specs_to_list,
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


BLACKHOLE_FLASH_ATTN_LEAF_COMPUTE_OPS = {
    "binary_max_tile",
    "reduce_tile",
    "mul_tiles",
    "add_tiles",
    "mul_tiles_bcast_cols",
    "add_tiles_bcast_cols",
    "exp2_tile",
    "copy_tile",
    "pack_tile",
    "typecast_tile",
    "fill_tile",
    "recip_tile",
}


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


def _run_mha_device_module_to_selected_builtins(*args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": mha_example.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
        phase_b_mod = LowerToBlackholePhaseB(mod)
        mod = OptimizeForTarget(mod, target)
    device_mod = tvm.tir.transform.Filter(get_device_call(False))(mod)
    device_mod = _align_blackhole_device_symbol(phase_b_mod, device_mod)
    for transform in (
        tilelang.transform.BuildSpatialPlan(),
        tilelang.transform.ValidateSpatialPlan(),
        tilelang.transform.SplitBlackholeKernel(),
        tilelang.transform.PlanTTBlocks(),
        tilelang.transform.SelectBlackholeTTMetalBuiltins(),
    ):
        device_mod = transform(device_mod)
    return device_mod


def _has_allocate_for_buffer(func, buffer_name):
    found = False

    def visit(node):
        nonlocal found
        if found:
            return
        if isinstance(node, tvm.tir.Allocate) and str(node.buffer_var.name) == buffer_name:
            found = True

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return found


def _missing_allocates_for_buffers(func, buffer_names):
    return {
        name
        for name in buffer_names
        if not _has_allocate_for_buffer(func, name)
    }


def _cb_logical_names(cb_config):
    names = [str(cb_config["name"])]
    names.extend(str(name) for name in cb_config.get("requirement_names", []))
    return names


def _find_cb_id_by_logical_prefix(spec, prefix):
    return next(
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if any(name.startswith(prefix) for name in _cb_logical_names(cb))
    )


def _assert_cb_queue_events_do_not_underflow_or_exceed_capacity(spec, kernel_kinds):
    capacity = {int(config["cb_id"]): int(config["num_pages"]) for config in spec["cb_configs"]}
    front = {cb_id: 0 for cb_id in capacity}
    reserved = {cb_id: 0 for cb_id in capacity}
    kernels_by_kind = {
        str(kernel["kind"]): str(kernel["source_code"])
        for kernel in spec["kernels"]
    }
    event_pattern = re.compile(
        r"cb_(reserve_back|push_back|wait_front|pop_front)\((\d+),\s*(\d+)\);"
    )

    for label in kernel_kinds:
        source = kernels_by_kind[label]
        for event in event_pattern.finditer(source):
            op, cb_text, pages_text = event.groups()
            cb_id = int(cb_text)
            pages = int(pages_text)
            if cb_id not in capacity:
                continue
            if op == "reserve_back":
                assert front[cb_id] + reserved[cb_id] + pages <= capacity[cb_id], (
                    f"{label}: cb{cb_id} reserve would exceed capacity "
                    f"{capacity[cb_id]} at source offset {event.start()}"
                )
                reserved[cb_id] += pages
            elif op == "push_back":
                assert reserved[cb_id] >= pages, (
                    f"{label}: cb{cb_id} push has only {reserved[cb_id]} reserved pages "
                    f"at source offset {event.start()}"
                )
                reserved[cb_id] -= pages
                front[cb_id] += pages
            elif op == "wait_front":
                assert front[cb_id] >= pages, (
                    f"{label}: cb{cb_id} wait has only {front[cb_id]} front pages "
                    f"at source offset {event.start()}"
                )
            else:
                assert front[cb_id] >= pages, (
                    f"{label}: cb{cb_id} pop has only {front[cb_id]} front pages "
                    f"at source offset {event.start()}"
                )
                front[cb_id] -= pages


def _assert_compute_source_does_not_read_dynamic_tiles_from_single_page_cbs(spec):
    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in spec["kernels"]
            if str(kernel["kind"]) == "compute"
        )
    )
    single_page_cbs = {
        int(config["cb_id"])
        for config in spec["cb_configs"]
        if int(config["num_pages"]) == 1
    }
    assert single_page_cbs

    def assert_static_zero_tile(cb_id, tile_expr, call_text):
        assert str(tile_expr).strip() == "0", (
            f"compute source reads dynamic/nonzero tile {tile_expr!r} from "
            f"single-page cb{cb_id}: {call_text}"
        )

    for match in re.finditer(r"copy_tile\((\d+),\s*([^,\)]+),\s*\d+\);", compute_source):
        cb_id = int(match.group(1))
        if cb_id in single_page_cbs:
            assert_static_zero_tile(cb_id, match.group(2), match.group(0))

    tile_binary_pattern = re.compile(
        r"(?:add_tiles|sub_tiles|mul_tiles|"
        r"mul_tiles_bcast<BroadcastType::COL>|add_tiles_bcast_cols)"
        r"\((\d+),\s*(\d+),\s*([^,\)]+),\s*([^,\)]+),\s*\d+\);"
    )
    for match in tile_binary_pattern.finditer(compute_source):
        lhs_cb = int(match.group(1))
        rhs_cb = int(match.group(2))
        if lhs_cb in single_page_cbs:
            assert_static_zero_tile(lhs_cb, match.group(3), match.group(0))
        if rhs_cb in single_page_cbs:
            assert_static_zero_tile(rhs_cb, match.group(4), match.group(0))


def _load_flash_attention_module_with_dtype(module_path, dtype_expr):
    source = Path(module_path).read_text()
    source = source.replace("dtype = T.float16", f"dtype = {dtype_expr}", 1)
    mutated = types.ModuleType(f"{Path(module_path).stem}_{dtype_expr.replace('.', '_')}")
    mutated.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), mutated.__dict__)
    return mutated


def _assert_no_executable_contract_family(spec):
    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in spec


def _collect_executable_compute_ops(executable_spec):
    if "segment_plan" in executable_spec:
        return [
            op
            for segment in executable_spec["segment_plan"]
            for op in segment.get("compute_ops", [])
        ]
    return [
        op
        for kernel in executable_spec.get("kernels", [])
        for op in kernel.get("compute_ops", [])
    ]


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
    for kernel in tt_program.kernels:
        abi_plan_index = int(kernel.abi_plan_index)
        abi = tt_program.abi_plans[abi_plan_index]
        accessors = strip_accessors(tt_accessor_specs_to_list(abi.accessors))
        compile_time_arg_specs = strip_compile_time_arg_specs(
            tt_compile_time_arg_specs_to_list(abi.compile_time_arg_specs)
        )
        rebuilt_abi_plans[abi_plan_index] = rebuild_tt_abi_plan(
            abi, accessors=accessors, compile_time_arg_specs=compile_time_arg_specs
        )

    return rebuild_tt_program(tt_program, abi_plans=rebuilt_abi_plans)


def _blackhole_builtin_suffixes(func):
    return {
        name.split("tl.blackhole.", 1)[1]
        for name in _collect_blackhole_builtin_names(func)
    }


def _typed_layout_plans_by_buffer(func):
    return {
        str(plan.buffer): plan
        for plan in require_tt_program(func).buffer_distribution_plans
        if len(plan.logical_shape) > 0
    }


def test_flash_attention_forward_tt_target_emits_typed_tt_program_without_payload():
    lowered = _lower_flash_attention_to_tt_target()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs
    assert "blackhole.lowering_requirements" not in attrs

    tt_program = require_tt_program(lowered)
    layout_buffers = set(_typed_layout_plans_by_buffer(lowered))
    builtin_names = _blackhole_builtin_suffixes(lowered)
    gemm_ops = [
        op for op in tt_program.compute_op_plans if str(op.kind) == "gemm"
    ]

    assert not hasattr(tt_program, "payload")
    assert len(gemm_ops) == 2
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(layout_buffers)
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
        "mul_bcast_cols_init_short",
        "mul_tiles_bcast_cols",
        "add_bcast_cols_init_short",
        "add_tiles_bcast_cols",
        "exp2_tile_init",
        "exp2_tile",
        "pack_tile",
        "pack_fill_fragment_to_tiled_cb",
        "tilize_cast_fragment_slice",
    }.issubset(builtin_names)
    assert "cast_fragment_slice" not in builtin_names


def test_flash_attention_tt_program_projects_two_typed_gemm_compute_ops():
    lowered = _lower_flash_attention_to_tt_target()
    tt_program = require_tt_program(lowered)
    assert all(not hasattr(kernel, "payload") for kernel in tt_program.kernels)

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
    executable_gemm_ops = [
        op for op in executable["compute_op_plans"] if str(op["kind"]) == "gemm"
    ]
    assert len(executable_gemm_ops) == 2
    compute_segments = [
        segment
        for segment in executable["segment_plan"]
        if str(segment["kind"]) == "compute"
    ]
    assert len(compute_segments) == 1
    segment_gemm_ops = [
        op for op in compute_segments[0]["compute_ops"] if str(op["kind"]) == "gemm"
    ]
    assert len(segment_gemm_ops) == 2


def test_flash_attention_tt_program_projects_non_gemm_exact_compute_ops():
    lowered = _lower_flash_attention_to_tt_target()
    tt_program = require_tt_program(lowered)

    non_gemm_ops = [
        op for op in tt_program.compute_op_plans if str(op.kind) != "gemm"
    ]
    operation_names = {str(op.operation_name) for op in non_gemm_ops}
    assert operation_names <= BLACKHOLE_FLASH_ATTN_LEAF_COMPUTE_OPS
    assert {
        "binary_max_tile",
        "reduce_tile",
        "mul_tiles",
        "add_tiles",
        "mul_tiles_bcast_cols",
        "add_tiles_bcast_cols",
        "exp2_tile",
    }.issubset(operation_names)

    expected_kind_by_operation = {
        "binary_max_tile": "binary",
        "reduce_tile": "reduce",
        "mul_tiles": "binary",
        "add_tiles": "binary",
        "mul_tiles_bcast_cols": "binary",
        "add_tiles_bcast_cols": "binary",
        "exp2_tile": "unary",
    }
    for op in non_gemm_ops:
        assert str(op.kernel_name) == "compute"
        assert int(op.kernel_plan_index) >= 0
        assert bool(op.enabled)
        if str(op.operation_name) in expected_kind_by_operation:
            assert str(op.kind) == expected_kind_by_operation[str(op.operation_name)]
        roles = {str(binding.role) for binding in op.operand_bindings}
        if str(op.kind) == "binary":
            assert {"lhs", "rhs", "output"}.issubset(roles)
        elif str(op.kind) == "unary":
            assert {"input", "output"}.issubset(roles)
        elif str(op.kind) == "reduce":
            assert {"input", "scaler", "output"}.issubset(roles)

    mod = tvm.IRModule({"main": lowered})
    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    compute_segments = [
        segment
        for segment in executable["segment_plan"]
        if str(segment["kind"]) == "compute"
    ]
    assert len(compute_segments) == 1
    projected_operation_names = {
        str(op["operation_name"]) for op in compute_segments[0]["compute_ops"]
    }
    assert operation_names.issubset(projected_operation_names)


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

    assert "mul_tiles_bcast_cols" in builtin_names
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


def test_flash_attention_forward_optimized_path_lowers_exp2_to_leaf_tile_ops():
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
    assert not hasattr(require_tt_program(lowered), "payload")


def test_flash_attention_forward_optimized_path_uses_tilize_cast_publication():
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

    assert "cast_fragment_slice" not in builtin_names
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

    assert "tilize_cast_fragment_slice" in builtin_names
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

    assert "tilize_cast_fragment_slice" in builtin_names
    assert "O_shared_1[tx" not in script


def test_flash_attention_gqa_reader_runtime_args_cover_all_accessor_buffers():
    lowered = _run_flash_attention_tt_target_after_optimize(
        gqa_example,
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
    )["main"]

    tt_program = require_tt_program(lowered)
    reader_abi = tt_abi_for_kernel(tt_program, require_tt_kernel(tt_program, kind="reader", core_type="brisc"))

    reader_accessors = tt_accessor_specs_to_list(reader_abi.accessors)
    reader_runtime_args = tt_runtime_arg_specs_to_list(reader_abi.runtime_args)
    accessor_buffers = [acc["buffer"] for acc in reader_accessors]
    runtime_arg_buffers = [
        arg["buffer"]
        for arg in reader_runtime_args
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert len(accessor_buffers) == 3
    assert runtime_arg_buffers == accessor_buffers


def test_flash_attention_gqa_aggregated_runtime_args_cover_segment_buffers():
    lowered = _run_flash_attention_tt_target_after_optimize(
        gqa_example,
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
    )["main"]

    tt_program = require_tt_program(lowered)
    seen_runtime_arg_identities = set()
    top_level_runtime_arg_buffers = []
    for abi in tt_program.abi_plans:
        for arg in tt_runtime_arg_specs_to_list(abi.runtime_args):
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
        arg["buffer"]
        for arg in tt_runtime_arg_specs_to_list(reader_abi.runtime_args)
        if arg["kind"] == "input_buffer_addr32"
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


def test_plan_tt_compute_preserves_acc_allocator_when_redundant_fill_is_pruned():
    selected = _run_mha_device_module_to_selected_builtins(
        1,
        32,
        256,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )

    expected_compute_resources = {
        "acc_o",
        "acc_s",
        "logsum",
        "scores_max",
        "scores_scale",
    }
    func = next(iter(selected.functions.values()))
    assert not _missing_allocates_for_buffers(func, expected_compute_resources)

    planned = tilelang.transform.PlanTTCompute()(selected)
    func = next(iter(planned.functions.values()))

    assert not _missing_allocates_for_buffers(func, {"acc_o", "acc_s"})
    assert _missing_allocates_for_buffers(
        func, {"logsum", "scores_max", "scores_scale"}
    ) == {"logsum", "scores_max", "scores_scale"}


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
    layout_buffers = set(_typed_layout_plans_by_buffer(device_func))
    assert device_func.attrs.get("tl.spatial_program") is None
    assert device_func.attrs.get("blackhole.lowering_requirements") is None
    assert len(plan.execution_units) >= 2
    assert len(plan.dataflow_edges) >= 1
    assert len(plan.phase_plans) >= 1
    assert "scores_max" in layout_buffers


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
    compute_kernel = _require_blackhole_kernel(spec["kernels"], kind="compute", core_type="trisc")
    compute_source = str(compute_kernel["source_code"])

    pack_cb_ids = {
        int(cb_id)
        for cb_id in re.findall(
            r"\bpack_reconfig_data_format(?:<true>)?\((\d+)\)",
            compute_source,
        )
    }

    assert pack_cb_ids
    assert pack_cb_ids == {cb_id for cb_id in pack_cb_ids if 0 <= cb_id <= 31}


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
def test_flash_attention_executable_spec_drops_contract_family_and_reports_typed_materialization_gate(
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
    assert any(
        "thread-distributed cb_republish materialization" in reason
        or "multi-page exact CB-republish live-form" in reason
        for reason in reasons
    )

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


def test_flash_attention_has_no_tt_program_payload_or_executable_epilogue_contract():
    lowered = _lower_flash_attention_to_tt_target()
    tt_program = require_tt_program(lowered)
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
    tt_operation_names = {str(op.operation_name) for op in tt_program.compute_op_plans}
    executable_operation_names = {
        str(op["operation_name"])
        for op in _collect_executable_compute_ops(executable_spec)
    }

    assert not hasattr(tt_program, "payload")
    assert "compute_epilogue_ops" not in executable_spec
    assert tt_operation_names <= BLACKHOLE_FLASH_ATTN_LEAF_COMPUTE_OPS | {"matmul_tiles"}
    assert executable_operation_names <= BLACKHOLE_FLASH_ATTN_LEAF_COMPUTE_OPS | {"matmul_tiles"}
    assert "tl.blackhole.reduce_tile" in builtin_names
    assert "tl.blackhole.exp2_tile" in builtin_names


def test_flash_attention_tt_program_keeps_typed_layout_specs_for_internal_buffers():
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
    tt_program = require_tt_program(device_func)
    layout_plans = _typed_layout_plans_by_buffer(device_func)

    assert not hasattr(tt_program, "payload")
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(layout_plans)
    for name in ("acc_s", "acc_s_cast", "acc_o"):
        plan = layout_plans[name]
        assert str(plan.memory_space) == "L1"
        assert tuple(int(dim) for dim in plan.logical_shape) == (128, 128)
        assert tuple(int(dim) for dim in plan.local_shape) == (128,)
        assert int(plan.thread_extent) == 128
        assert int(plan.replicate_extent) == 1
        assert len(plan.inverse_logical_index_exprs) == 3
    for name in ("scores_max", "scores_max_prev", "scores_scale", "scores_sum", "logsum"):
        plan = layout_plans[name]
        assert str(plan.memory_space) == "L1"
        assert tuple(int(dim) for dim in plan.logical_shape) == (128,)
        assert tuple(int(dim) for dim in plan.local_shape) == (1,)
        assert int(plan.thread_extent) == 128
        assert int(plan.replicate_extent) == 1
        assert len(plan.inverse_logical_index_exprs) == 2


def test_flash_attention_tt_program_keeps_typed_layout_specs_for_republished_buffers():
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
    tt_program = require_tt_program(device_func)
    layout_plans = _typed_layout_plans_by_buffer(device_func)

    assert not hasattr(tt_program, "payload")
    assert {"acc_s", "acc_s_cast", "acc_o"}.issubset(layout_plans)

    acc_s_plan = layout_plans["acc_s"]
    acc_s_cast_plan = layout_plans["acc_s_cast"]
    assert str(acc_s_cast_plan.memory_space) == "L1"
    assert tuple(int(dim) for dim in acc_s_cast_plan.logical_shape) == (32, 32)
    assert tuple(int(dim) for dim in acc_s_cast_plan.local_shape) == (8,)
    assert int(acc_s_cast_plan.thread_extent) == 128
    assert int(acc_s_cast_plan.replicate_extent) == 1
    assert tuple(int(dim) for dim in acc_s_plan.logical_shape) == tuple(
        int(dim) for dim in acc_s_cast_plan.logical_shape
    )
    assert tuple(int(dim) for dim in acc_s_plan.local_shape) == tuple(
        int(dim) for dim in acc_s_cast_plan.local_shape
    )


def test_flash_attention_tt_program_keeps_typed_layout_specs_without_distribution_contracts():
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
    assert not hasattr(require_tt_program(device_func), "payload")
    layout_buffers = set(_typed_layout_plans_by_buffer(device_func))
    assert {
        "acc_s",
        "acc_s_cast",
        "acc_o",
        "scores_max",
        "scores_max_prev",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(layout_buffers)


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
            per_work_arg_specs = tt_per_work_arg_specs_to_list(kernel.per_work_arg_specs)
            if str(kernel.name) == str(writer.name):
                updated_specs = []
                for spec in per_work_arg_specs:
                    spec = dict(spec)
                    if str(spec.get("arg_kind", "")) == "output_tile_start_id":
                        spec["value_source"] = "logical_block_x"
                    updated_specs.append(spec)
                per_work_arg_specs = updated_specs
            rebuilt_kernels.append(
                rebuild_tt_kernel(kernel, per_work_arg_specs=per_work_arg_specs)
            )
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

    def count_compute_reserved_tiles(cb_id: int) -> int:
        return sum(
            int(tile_count)
            for tile_count in re.findall(rf"cb_reserve_back\({cb_id},\s*(\d+)\);", compute_source)
        )

    def count_compute_pushed_tiles(cb_id: int) -> int:
        return sum(
            int(tile_count)
            for tile_count in re.findall(rf"cb_push_back\({cb_id},\s*(\d+)\);", compute_source)
        )

    reader_backed_input_cbs = [
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["role"]) == "input" and int(cb["initial_reserve_pages"]) == 0
    ]
    assert reader_backed_input_cbs
    for cb_id in reader_backed_input_cbs:
        assert count_reader_reads(cb_id) == count_compute_waited_tiles(cb_id)

    compute_published_input_cbs = [
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["role"]) == "input" and int(cb["initial_reserve_pages"]) > 0
    ]
    for cb_id in compute_published_input_cbs:
        waited_tiles = count_compute_waited_tiles(cb_id)
        assert count_reader_reads(cb_id) == 0
        assert waited_tiles > 0
        assert count_compute_reserved_tiles(cb_id) == waited_tiles
        assert count_compute_pushed_tiles(cb_id) == waited_tiles


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
    compute_op_buffers = {
        str(binding["buffer"])
        for op in compute.get("compute_ops", [])
        for binding in op.get("operand_bindings", [])
        if "buffer" in binding
    }

    # Fragment state must belong to the compute kernel even when some values are
    # fully CB-backed and no longer need named local arrays in generated source.
    assert "acc_o" in compute_source
    assert {"acc_o", "scores_max", "acc_s_cast"}.issubset(compute_op_buffers)

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
    assert not tx_loop_lines
    assert any("dst_offset_elements = 0" in line for line in dst_offset_lines)
    assert "thread_idx_x = 0" in compute_source


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
    assert "reduce_uninit<false>()" in compute_source
    assert "tilelang_reduce_grouped_row_max" not in compute_source
    assert "const uint32_t num_rows = num_elements / row_width;" not in compute_source


def test_flash_attention_small_compute_source_prunes_dead_acc_fragment_fills():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
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

    assert "MATH({ float* dst = reinterpret_cast<float*>(" not in compute_source
    assert "; tilelang_fill_fragment(dst, num_elements, value);" not in compute_source
    assert "/* scope: blackhole.acc */ float " not in compute_source
    assert "reinterpret_cast<float*>(acc_o)" not in compute_source
    assert "reinterpret_cast<float*>(acc_s)" not in compute_source
    assert "reinterpret_cast<float*>(logsum)" not in compute_source
    assert "fill_tile(0, static_cast<float>(-inff))" in compute_source


def test_flash_attention_small_compute_source_publishes_gemm_output_before_compute_reuse():
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
    acc_o_cb = _find_cb_id_by_logical_prefix(spec, "acc_o")

    reserve = f"cb_reserve_back({acc_o_cb}, 1);"
    push = f"cb_push_back({acc_o_cb}, 1);"
    wait = f"cb_wait_front({acc_o_cb}, 1);"
    pack_candidates = (
        f"pack_tile(0, {acc_o_cb});",
        f"pack_tile(0, {acc_o_cb}, 0);",
    )
    pack = next((candidate for candidate in pack_candidates if candidate in compute_source), None)

    assert reserve in compute_source
    assert pack is not None
    assert push in compute_source
    assert wait in compute_source
    assert compute_source.index(reserve) < compute_source.index(pack)
    assert compute_source.index(pack) < compute_source.index(push)
    assert compute_source.index(push) < compute_source.index(wait)


def test_flash_attention_small_compute_source_respects_cb_capacity_on_reuse():
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
    _assert_cb_queue_events_do_not_underflow_or_exceed_capacity(
        spec, ("reader", "compute")
    )


def test_flash_attention_compute_source_uses_cb_backed_row_state_fills():
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
    assert "/* scope: blackhole.acc */ float logsum[1];" not in compute_source
    assert "/* scope: blackhole.acc */ float scores_max[1];" not in compute_source
    assert "/* scope: blackhole.acc */ float scores_scale" not in compute_source
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_o); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert (
        "MATH({ float* dst = reinterpret_cast<float*>(acc_s); const uint32_t num_elements = 128;"
        in compute_source
    )
    assert "MATH({ float* dst = reinterpret_cast<float*>(logsum);" not in compute_source
    assert "fill_tile(0, static_cast<float>(-inff))" in compute_source
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
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}
    acc_s_cb_id = cb_by_name["acc_s_cast"]

    assert re.search(
        r"dst_bits\[tiled_index\] = tilelang_float_to_(?:half|bfloat)_bits"
        r"\(static_cast<float>\(src\[src_offset_elements \+ [A-Za-z0-9_]+\]\)\);",
        compute_source,
    )
    assert "const uint32_t num_elements = 16384;" in compute_source
    assert re.search(rf"cb_reserve_back\({acc_s_cb_id},\s*16\);", compute_source)
    assert re.search(rf"pack_tile\(0,\s*{acc_s_cb_id},\s*15\);", compute_source)
    assert re.search(rf"cb_push_back\({acc_s_cb_id},\s*16\);", compute_source)


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

    assert "const uint32_t src_offset_elements = 0;" in compute_source
    assert "const uint32_t row_width = 128;" in compute_source
    assert "const uint32_t thread_idx_x = 0;" in compute_source


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
    reserve_match = re.search(rf"cb_reserve_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    assert reserve_match is not None
    acc_s_tiles = reserve_match.group(1)

    pack_match = re.search(rf"pack_tile\(0, {acc_s_cb_id}, \d+\);", compute_source)
    publish_match = re.search(rf"cb_push_back\({acc_s_cb_id}, (\d+)\);", compute_source)
    second_mm_match = re.search(rf"mm_init\({acc_s_cb_id}, \d+, \d+\);", compute_source)
    wait_match = re.search(rf"cb_wait_front\({acc_s_cb_id},\s*(\d+)\);", compute_source)
    second_mm_issue_match = re.search(rf"matmul_tiles\({acc_s_cb_id}, \d+,", compute_source)

    assert pack_match is not None
    assert publish_match is not None
    assert second_mm_match is not None
    assert wait_match is not None
    assert second_mm_issue_match is not None
    assert acc_s_tiles == publish_match.group(1) == wait_match.group(1)
    assert reserve_match.start() < pack_match.start() < publish_match.start()
    assert publish_match.start() < second_mm_match.start()
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


def test_flash_attention_small_bf16_compute_source_publishes_output_via_typed_pack_path():
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

    assert "O_shared_local_cast" not in compute_source
    publish_match = re.search(
        rf"cb_reserve_back\({output_cb_id}, 1\);\s*"
        r"tile_regs_acquire\(\);\s*"
        r"copy_tile_to_dst_init_short\((\d+)\);\s*"
        r"copy_tile\((\d+), 0, 0\);\s*"
        r"tile_regs_commit\(\);\s*"
        r"tile_regs_wait\(\);\s*"
        rf"pack_reconfig_data_format(?:<true>)?\({output_cb_id}\);\s*"
        rf"pack_tile\(0, {output_cb_id}, 0\);\s*"
        r"tile_regs_release\(\);\s*"
        r"(?P<source_lifetime>(?:cb_pop_front\(\d+, 1\);\s*)*)"
        rf"cb_push_back\({output_cb_id}, 1\);",
        compute_source,
    )
    assert publish_match is not None
    assert publish_match.group(1) == publish_match.group(2)
    source_cb_id = publish_match.group(2)
    assert f"cb_pop_front({source_cb_id}, 1);" in publish_match.group("source_lifetime")
    assert f"tilelang_cb_write_ptr_bytes_direct({output_cb_id})" not in compute_source
    assert f"tilelang_get_cb_write_ptr_bytes({output_cb_id})" not in compute_source
    assert f"get_local_cb_interface({output_cb_id}).fifo_wr_ptr" not in compute_source
    assert (
        f"PACK({{ uint16_t* dst_bits = reinterpret_cast<uint16_t*>((get_local_cb_interface({output_cb_id}).fifo_wr_ptr << 4)"
        not in compute_source
    )


def test_flash_attention_small_bf16_compute_source_uses_typed_exact_helpers():
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

    assert "MATH({ float* dst = reinterpret_cast<float*>(acc_s);" not in compute_source
    assert "MATH({ float* dst = reinterpret_cast<float*>(acc_o);" not in compute_source
    assert "fill_tile(0, static_cast<float>(0.000000e+00f))" in compute_source
    assert "fill_tile(0, static_cast<float>(-inff))" in compute_source
    assert "reinterpret_cast<const uint32_t*>(acc_s)" not in compute_source
    assert "reinterpret_cast<const uint32_t*>(scores_scale)" not in compute_source
    assert "reinterpret_cast<const uint32_t*>(logsum)" not in compute_source
    assert "reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>" in compute_source
    assert "reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>" in compute_source
    assert "binary_max_tile" in compute_source
    assert "recip_tile_init();" in compute_source
    assert "recip_tile(0);" in compute_source
    assert "mul_bcast_cols_init_short" in compute_source
    assert "mul_tiles_bcast<BroadcastType::COL>" in compute_source
    assert "BroadcastType::ROW" not in compute_source


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
    assert 'WAYPOINT("FILL")' in compute_source
    assert 'WAYPOINT("CAST")' not in compute_source
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

    assert compute_source.count(reserve) == 2
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


def test_flash_attention_seq64_bf16_republish_consumes_fresh_acc_s_live_cb():
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

    acc_s_cb = cb_by_name["acc_s"]
    acc_s_cast_cb = cb_by_name["acc_s_cast"]
    cast_reserve = f"cb_reserve_back({acc_s_cast_cb}, 1);"
    cast_push = f"cb_push_back({acc_s_cast_cb}, 1);"

    first_cast_reserve_pos = compute_source.find(cast_reserve)
    assert first_cast_reserve_pos != -1
    first_cast_push_pos = compute_source.find(cast_push, first_cast_reserve_pos)
    assert first_cast_push_pos != -1
    republish_match = re.search(
        rf"cb_wait_front\((\d+), 1\);\s*"
        rf"cb_reserve_back\({acc_s_cast_cb}, 1\);\s*"
        rf"tile_regs_acquire\(\);\s*"
        rf"copy_tile_to_dst_init_short\(\1\);\s*"
        rf"copy_tile\(\1, 0, 0\);",
        compute_source[max(0, first_cast_reserve_pos - 256) : first_cast_push_pos],
    )

    assert republish_match is not None
    fresh_acc_s_cb = int(republish_match.group(1))
    assert fresh_acc_s_cb != acc_s_cb
    source_pop = f"cb_pop_front({fresh_acc_s_cb}, 1);"
    source_pop_pos = compute_source.find(source_pop, first_cast_reserve_pos)
    assert first_cast_reserve_pos < source_pop_pos < first_cast_push_pos


def test_flash_attention_seq64_bf16_compute_source_packs_acc_s_cast_after_each_rereserve():
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
    push = f"cb_push_back({acc_s_cast_cb}, 1);"
    pack_reconfig_pattern = rf"pack_reconfig_data_format(?:<true>)?\({acc_s_cast_cb}\);"
    pack_tile = f"pack_tile(0, {acc_s_cast_cb}, 0);"

    assert f"tilelang_get_cb_write_ptr_bytes({acc_s_cast_cb})" not in compute_source

    reserve_matches = list(
        re.finditer(rf"cb_reserve_back\({acc_s_cast_cb}, (\d+)\);", compute_source)
    )
    assert len(reserve_matches) == 2
    assert reserve_matches[-1].group(1) == "1"

    for reserve_match in reserve_matches:
        reserve_pos = reserve_match.start()
        pack_reconfig_match = re.search(pack_reconfig_pattern, compute_source[reserve_pos:])
        pack_reconfig_pos = (
            reserve_pos + pack_reconfig_match.start() if pack_reconfig_match else -1
        )
        pack_tile_pos = compute_source.find(pack_tile, reserve_pos)
        push_pos = compute_source.find(push, reserve_pos)
        assert pack_reconfig_pos != -1
        assert pack_tile_pos != -1
        assert push_pos != -1
        assert reserve_pos < pack_reconfig_pos < pack_tile_pos < push_pos


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


def test_flash_attention_seq64_bf16_compute_source_keeps_acc_s_and_acc_o_single_tile_state_publication():
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

    scores_max_prev_cb = cb_by_name.get("scores_max_prev")
    acc_s_cb = cb_by_name["acc_s"]
    acc_o_publication_cbs = [
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["name"]) == "acc_o"
        or str(cb["name"]).startswith("acc_o_fragment_merge_live_form")
    ]
    if scores_max_prev_cb is not None:
        assert f"cb_push_back({scores_max_prev_cb}, 2);" not in compute_source

    assert f"cb_reserve_back({acc_s_cb}, 1);" in compute_source
    assert f"cb_push_back({acc_s_cb}, 1);" in compute_source
    assert (
        f"pack_tile(0, {acc_s_cb});" in compute_source
        or f"pack_tile(0, {acc_s_cb}, 0);" in compute_source
    )
    assert f"cb_reserve_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 16);" not in compute_source

    def publishes_single_acc_o_page(cb_id):
        return (
            f"cb_reserve_back({cb_id}, 1);" in compute_source
            and f"cb_push_back({cb_id}, 1);" in compute_source
            and (
                f"pack_tile(0, {cb_id});" in compute_source
                or f"pack_tile(0, {cb_id}, 0);" in compute_source
            )
        )

    assert any(publishes_single_acc_o_page(cb_id) for cb_id in acc_o_publication_cbs)
    for cb_id in acc_o_publication_cbs:
        assert f"cb_reserve_back({cb_id}, 16);" not in compute_source
        assert f"cb_push_back({cb_id}, 16);" not in compute_source


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
        r"add_tiles_init\((\d+), (\d+)\);.*?add_tiles\(\1, \2, 0, 0, 0\);.*?"
        r"pack_tile\(0, (\d+)(?:, \d+)?\);",
        re.DOTALL,
    )
    merge_windows = list(merge_window_pattern.finditer(compute_source))
    assert merge_windows
    assert all("tile_regs_commit()" in window.group(0) for window in merge_windows)
    assert all("tile_regs_wait()" in window.group(0) for window in merge_windows)

    merge_cb_ids = {cb_id for pair in merge_pairs for cb_id in pair}
    merge_output_cb_ids = {window.group(3) for window in merge_windows}
    for cb_id in merge_cb_ids:
        assert re.search(rf"cb_wait_front\({cb_id},\s*\d+\);", compute_source)
    for cb_id in merge_output_cb_ids:
        assert re.search(rf"cb_reserve_back\({cb_id},\s*\d+\);", compute_source)
        assert re.search(rf"cb_push_back\({cb_id},\s*\d+\);", compute_source)
    assert any(f"cb_pop_front({cb_id}, 1);" in compute_source for cb_id in merge_cb_ids)
    assert "tilelang_add_fragment(dst, src, num_elements);" not in compute_source
    assert "tilelang_get_cb_write_ptr_bytes" not in compute_source
    assert "get_tile_address(0)" not in compute_source


def test_flash_attention_seq64_bf16_compute_source_keeps_cb_events_queue_consistent():
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
    _assert_cb_queue_events_do_not_underflow_or_exceed_capacity(
        spec, ("reader", "compute")
    )


def test_flash_attention_seq64_bf16_pv_merge_consumes_scaled_acc_o_live_form():
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

    scaled_acc_o_then_pv_merge = re.compile(
        r"binary_op_init_common\((?P<acc_o>\d+), (?P<scale>\d+), (?P<scaled>\d+)\);"
        r".*?mul_tiles_bcast<BroadcastType::COL>\((?P=acc_o), (?P=scale), 0, 0, 0\);"
        r".*?cb_push_back\((?P=scaled), 1\);"
        r".*?add_tiles_init\((?P=scaled), (?P<partials>\d+)\);"
        r".*?add_tiles\((?P=scaled), (?P=partials), 0, 0, 0\);",
        re.DOTALL,
    )

    assert scaled_acc_o_then_pv_merge.search(compute_source) is not None


def test_flash_attention_seq64_bf16_compute_source_uses_static_tile_zero_for_single_page_cb_inputs():
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
    _assert_compute_source_does_not_read_dynamic_tiles_from_single_page_cbs(spec)


def test_flash_attention_seq64_bf16_compute_source_releases_qk_scores_before_next_scores_publish():
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
    compute_source = str(
        next(
            kernel["source_code"]
            for kernel in spec["kernels"]
            if str(kernel["kind"]) == "compute"
        )
    )
    cb_by_name = {str(cb["name"]): int(cb["cb_id"]) for cb in spec["cb_configs"]}
    acc_s_cb = cb_by_name["acc_s"]

    scores_reserves = [
        match.start()
        for match in re.finditer(rf"cb_reserve_back\({acc_s_cb},\s*1\);", compute_source)
    ]
    assert len(scores_reserves) >= 2
    first_scores_scale_consume = compute_source.find(f"mul_tiles({acc_s_cb},")
    assert first_scores_scale_consume != -1
    release_before_second_scores_publish = compute_source.find(
        f"cb_pop_front({acc_s_cb}, 1);",
        first_scores_scale_consume,
        scores_reserves[1],
    )
    assert release_before_second_scores_publish != -1


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


def test_flash_attention_compute_source_hoists_output_cb_staging_as_single_window():
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
    pack_match = re.search(
        rf"pack_reconfig_data_format(?:<true>)?\({output_cb_id}\);",
        compute_source[reserve_pos:],
    )
    pack_pos = reserve_pos + pack_match.start() if pack_match else -1
    push_pos = compute_source.find(f"cb_push_back({output_cb_id}, 16);")

    assert reserve_pos != -1
    assert pack_pos != -1
    assert push_pos != -1
    assert reserve_pos < pack_pos < push_pos
    assert "for (int32_t tx = 0; tx < 128; ++tx)" not in compute_source
    assert f"cb_reserve_back({output_cb_id}, 1);" not in compute_source
    assert "cb_reserve_back(0, 1);" not in compute_source


def test_flash_attention_compute_source_emits_thread_invariant_matmul_pipeline_without_thread_row_loop():
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

    q_cb = cb_by_name["Q_shared"]
    k_cb = cb_by_name["K_shared"]
    v_cb = cb_by_name["V_shared"]
    acc_s_cast_cb = cb_by_name["acc_s_cast"]

    tx_loop_pos = compute_source.find("for (int32_t tx = 0; tx < 128; ++tx)")
    first_mm_match = re.search(rf"mm_init\({q_cb}, {k_cb}, \d+\);", compute_source)
    second_mm_match = re.search(
        rf"mm_init\({acc_s_cast_cb}, {v_cb}, \d+\);", compute_source
    )
    acc_s_publish_match = re.search(
        rf"cb_push_back\({acc_s_cast_cb},\s*\d+\);", compute_source
    )
    acc_s_publish_pos = -1 if acc_s_publish_match is None else acc_s_publish_match.start()
    first_mm_pos = -1 if first_mm_match is None else first_mm_match.start()
    second_mm_pos = -1 if second_mm_match is None else second_mm_match.start()

    assert tx_loop_pos == -1
    assert first_mm_pos != -1
    assert second_mm_pos != -1
    assert acc_s_publish_pos != -1
    assert first_mm_pos < acc_s_publish_pos < second_mm_pos


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
    reload_cbs = [
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if str(cb["name"]).startswith("acc_s_fragment_merge_reload")
    ]
    live_form_cb = _find_cb_id_by_logical_prefix(spec, "acc_s_fragment_merge_live_form")
    acc_o_live_form_cb = _find_cb_id_by_logical_prefix(spec, "acc_o_fragment_merge_live_form")

    assert f"cb_reserve_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 16);" not in compute_source
    assert f"cb_reserve_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_s_cb}, 1);" not in compute_source
    assert f"cb_reserve_back({acc_o_live_form_cb}, 16);" in compute_source
    assert f"cb_push_back({acc_o_live_form_cb}, 16);" in compute_source
    assert f"cb_reserve_back({acc_o_live_form_cb}, 1);" not in compute_source
    assert f"cb_push_back({acc_o_live_form_cb}, 1);" not in compute_source
    for reload_cb in reload_cbs:
        assert reload_cb not in (acc_s_cb, acc_o_live_form_cb, live_form_cb)
    assert live_form_cb != acc_s_cb
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
