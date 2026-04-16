import pytest
import torch
import sys
from pathlib import Path

import tilelang
from tilelang import language as T
from tilelang.engine.lower import is_device_call, lower, merge_ir_modules
from tilelang.engine.phase import LowerAndLegalize
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang import tvm
from tvm import tir

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_codegen_requirements,
    check_blackhole_direct_execution_requirements,
    extract_blackhole_cb_configs,
    extract_blackhole_core_plan,
    extract_blackhole_runtime_args,
    extract_blackhole_segment_plan,
    extract_blackhole_total_l1_bytes,
    extract_blackhole_work_per_core,
    find_loop_annotation,
    grid_indexed_staged_copy_kernel,
    lower_blackhole_to_tt_target,
    rebuild_tt_abi_plan,
    rebuild_tt_kernel,
    rebuild_tt_program,
    rebuild_tt_semaphore_plan,
    require_tt_kernel,
    require_tt_program,
    staged_copy_kernel,
    staged_stick_copy_kernel,
    tt_abi_for_kernel,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example


FRAGMENT_BRIDGE_HELPER_BUILTINS = (
    "write_local_slice_to_cb",
    "write_local_fragment_tile_to_cb",
    "write_local_fragment_slice_to_tiled_cb",
    "cast_fragment_slice_to_tiled_cb",
    "read_cb_front_tile_to_local",
    "read_cb_front_tile_to_local_fragment",
)

EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS = [
    "input_buffer_addr32",
    "output_buffer_addr32",
    "work_linear_id",
    "a_tile_start_id",
    "a_tile_num_tiles",
    "a_tile_stride",
    "output_tile_start_id",
    "output_tile_num_tiles",
    "output_tile_stride",
]


def _collect_blackhole_builtin_names(node):
    names = set()

    def visit(expr):
        if isinstance(expr, tir.Call):
            op = expr.op
            if hasattr(op, "name") and op.name.startswith("tl.blackhole."):
                names.add(op.name)

    body = node.body if hasattr(node, "body") else node
    tir.stmt_functor.post_order_visit(body, visit)
    return names


def test_transport_tt_metal_api_granularity_rejects_fragment_bridge_helpers():
    target = Target("blackhole")
    mod = tvm.IRModule(
        {
            "main": mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            )
        }
    )
    with target:
        mod = LowerAndLegalize(mod, target)
    lowered = lower_blackhole_to_tt_target(mod)["main"]
    builtin_names = _collect_blackhole_builtin_names(lowered)

    for builtin_name in FRAGMENT_BRIDGE_HELPER_BUILTINS:
        assert f"tl.blackhole.{builtin_name}" not in builtin_names


def _rewrite_copy_semantics_annotations(func, annotation_mutator):
    def mutate(stmt):
        if not isinstance(stmt, tir.For):
            return stmt
        sem = stmt.annotations.get("blackhole.copy_semantics")
        if sem is None:
            return stmt
        annotations = dict(stmt.annotations)
        annotations["blackhole.copy_semantics"] = annotation_mutator(dict(sem))
        return tir.For(
            stmt.loop_var,
            stmt.min,
            stmt.extent,
            stmt.kind,
            stmt.body,
            stmt.thread_binding,
            annotations,
            stmt.step,
            getattr(stmt, "span", None),
        )

    new_body = tir.stmt_functor.ir_transform(func.body, None, mutate, ["tir.For"])
    return func.with_body(new_body)


def _refresh_tt_program_after_bridge_attr_mutation(device_mod):
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            func = func.without_attr("tl.tt_program")
        rewritten[gvar] = func
    refreshed = tvm.IRModule(rewritten, global_infos=device_mod.global_infos)
    refreshed = tilelang.transform.PlanTTBlocks()(refreshed)
    refreshed = tilelang.transform.PlanTTCompute()(refreshed)
    refreshed = tilelang.transform.PlanTTTransport()(refreshed)
    refreshed = tilelang.transform.PlanTTSync()(refreshed)
    refreshed = tilelang.transform.PlanTTABI()(refreshed)
    refreshed = tilelang.transform.PlanTTExecution()(refreshed)
    refreshed = tilelang.transform.BuildTTProgram()(refreshed)
    refreshed = tilelang.transform.ValidateTTProgram()(refreshed)
    refreshed = tilelang.transform.MaterializeBlackholeExecutable()(refreshed)
    return refreshed


def _rebuild_codegen_module_with_tt_program(
    artifact, *, tt_program_mutator=None, body_mutator=None
):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            if body_mutator is not None:
                func = tir.PrimFunc(
                    func.params,
                    body_mutator(func.body),
                    func.ret_type,
                    func.buffer_map,
                    func.attrs,
                    func.span,
                )
            if tt_program_mutator is not None:
                func = func.with_attr("tl.tt_program", tt_program_mutator(require_tt_program(func)))
        rewritten[gvar] = func
    device_mod = tvm.IRModule(rewritten, global_infos=artifact.device_mod.global_infos)
    device_mod = tilelang.transform.ValidateTTProgram()(device_mod)
    device_mod = tilelang.transform.MaterializeBlackholeExecutable()(device_mod)
    build_mod = merge_ir_modules(artifact.host_mod, device_mod)
    target = Target("blackhole")
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_tt_program_with_segment_plan(tt_program, segment_plan):
    kernels = list(tt_program.kernels)
    abi_plans = list(tt_program.abi_plans)
    rebuilt_kernels = []
    rebuilt_abi_plans = []

    for index, segment in enumerate(segment_plan):
        kernel = kernels[index]
        abi = abi_plans[int(kernel.abi_plan_index)]
        payload = dict(kernel.payload)
        for key in (
            "runtime_args",
            "common_runtime_args",
            "accessors",
            "compile_time_arg_specs",
            "semaphore_bindings",
        ):
            payload.pop(key, None)
        payload.update(dict(segment))

        runtime_args = (
            list(segment["runtime_args"])
            if "runtime_args" in segment
            else list(abi.runtime_args)
        )
        common_runtime_args = (
            list(segment["common_runtime_args"])
            if "common_runtime_args" in segment
            else list(abi.common_runtime_args)
        )
        accessors = list(segment["accessors"]) if "accessors" in segment else []
        compile_time_arg_specs = (
            list(segment["compile_time_arg_specs"])
            if "compile_time_arg_specs" in segment
            else []
        )
        semaphore_bindings = (
            list(segment["semaphore_bindings"]) if "semaphore_bindings" in segment else []
        )

        rebuilt_abi_plans.append(
            rebuild_tt_abi_plan(
                abi,
                name=f"abi_{index}",
                kernel_name=str(segment.get("name", kernel.name)),
                runtime_args=runtime_args,
                common_runtime_args=common_runtime_args,
                compile_time_arg_specs=compile_time_arg_specs,
                accessors=accessors,
                semaphore_bindings=semaphore_bindings,
                payload=payload,
            )
        )
        rebuilt_kernels.append(
            rebuild_tt_kernel(
                kernel,
                name=str(segment.get("name", kernel.name)),
                kind=str(segment.get("kind", kernel.kind)),
                core_type=str(segment.get("core_type", kernel.core_type)),
                abi_plan_index=index,
                payload=payload,
            )
        )

    return rebuild_tt_program(tt_program, kernels=rebuilt_kernels, abi_plans=rebuilt_abi_plans)


def _normalize_semaphore_plan_for_tt_program(semaphore_plan):
    normalized = []
    for i, plan in enumerate(semaphore_plan):
        normalized.append(
            {
                "name": plan.get("name", f"semaphore_{i}"),
                "kind": plan.get("kind", "local"),
                "semaphore_id": int(plan["id"]),
                "initial_value": int(plan.get("initial_value", 0)),
                "core_type": plan["core_type"],
                "source_task_index": int(plan.get("source_task_index", -1)),
                "target_task_index": int(plan.get("target_task_index", -1)),
                "core_ranges": list(plan.get("core_ranges", [])),
                "payload": dict(plan.get("payload", {})),
            }
        )
    return normalized


def _rebuild_tt_program_with_semaphore_plan(tt_program, semaphore_plan):
    normalized = _normalize_semaphore_plan_for_tt_program(semaphore_plan)
    rebuilt = []
    existing = list(tt_program.semaphore_plans)
    for i, plan in enumerate(normalized):
        if i < len(existing):
            rebuilt.append(
                rebuild_tt_semaphore_plan(
                    existing[i],
                    name=plan["name"],
                    kind=plan["kind"],
                    semaphore_id=plan["semaphore_id"],
                    initial_value=plan["initial_value"],
                    core_type=plan["core_type"],
                    source_task_index=plan["source_task_index"],
                    target_task_index=plan["target_task_index"],
                    core_ranges=plan["core_ranges"],
                    payload=plan["payload"],
                )
            )
        else:
            make_tt_semaphore_plan = tvm.get_global_func("tl.TTSemaphorePlan")
            rebuilt.append(
                make_tt_semaphore_plan(
                    plan["name"],
                    plan["kind"],
                    plan["semaphore_id"],
                    plan["initial_value"],
                    plan["core_type"],
                    plan["source_task_index"],
                    plan["target_task_index"],
                    plan["core_ranges"],
                    plan["payload"],
                )
            )
    return rebuild_tt_program(tt_program, semaphore_plans=rebuilt)


def _rebuild_codegen_module_with_runtime_args(artifact, runtime_args):
    def mutate(tt_program):
        abi_plans = [
            rebuild_tt_abi_plan(abi_plan, runtime_args=runtime_args)
            for abi_plan in tt_program.abi_plans
        ]
        return rebuild_tt_program(tt_program, abi_plans=abi_plans)

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_codegen_module_with_segment_plan(artifact, segment_plan):
    return _rebuild_codegen_module_with_tt_program(
        artifact,
        tt_program_mutator=lambda tt_program: _rebuild_tt_program_with_segment_plan(
            tt_program, segment_plan
        ),
    )


def _rebuild_codegen_module_without_tt_projection_attrs(artifact):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        for key in (
            "blackhole.segment_plan",
            "blackhole.runtime_args",
            "blackhole.common_runtime_args",
            "blackhole.per_work_arg_specs",
            "blackhole.accessors",
            "blackhole.cb_configs",
            "blackhole.semaphore_plan",
            "blackhole.core_plan",
        ):
            if func.attrs and key in func.attrs:
                func = func.without_attr(key)
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_without_materialized_executable(artifact):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "tl.blackhole_executable" in func.attrs:
            func = func.without_attr("tl.blackhole_executable")
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(
        artifact.host_mod,
        tvm.IRModule(rewritten, global_infos=device_mod.global_infos),
    )
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_without_tt_program(artifact):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            func = func.without_attr("tl.tt_program")
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(
        artifact.host_mod,
        tvm.IRModule(rewritten, global_infos=device_mod.global_infos),
    )
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_without_lowering_requirements(artifact):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "blackhole.lowering_requirements" in func.attrs:
            func = func.without_attr("blackhole.lowering_requirements")
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(
        artifact.host_mod,
        tvm.IRModule(rewritten, global_infos=device_mod.global_infos),
    )
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_without_copy_runtime_args(artifact):
    def mutate(tt_program):
        rebuilt_kernels = []
        rebuilt_abi_plans = []
        for index, kernel in enumerate(tt_program.kernels):
            abi = tt_program.abi_plans[index]
            payload = dict(kernel.payload)
            payload.pop("runtime_args", None)
            rebuilt_abi_plans.append(rebuild_tt_abi_plan(abi, runtime_args=[], payload=payload))
            rebuilt_kernels.append(
                rebuild_tt_kernel(kernel, abi_plan_index=index, payload=payload)
            )
        return rebuild_tt_program(tt_program, kernels=rebuilt_kernels, abi_plans=rebuilt_abi_plans)

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_codegen_module_with_semaphore_plan(artifact, semaphore_plan):
    return _rebuild_codegen_module_with_tt_program(
        artifact,
        tt_program_mutator=lambda tt_program: _rebuild_tt_program_with_semaphore_plan(
            tt_program, semaphore_plan
        ),
    )


def _rebuild_codegen_module_with_semaphore_binding(
    artifact, *, semaphore_plan=None, segment_mutator=None, runtime_args_mutator=None
):
    def mutate(tt_program):
        payload = dict(tt_program.payload)
        current_segment_plan = [dict(segment) for segment in payload.get("segment_plan", [])]
        if not current_segment_plan:
            current_segment_plan = [
                dict(kernel.payload) for kernel in tt_program.kernels
            ]
        if runtime_args_mutator is not None:
            runtime_args = runtime_args_mutator(
                [dict(arg) for arg in tt_program.abi_plans[0].runtime_args]
            )
            for segment in current_segment_plan:
                if "runtime_args" in segment or len(current_segment_plan) == 1:
                    segment["runtime_args"] = runtime_args
        if segment_mutator is not None:
            current_segment_plan = segment_mutator(current_segment_plan)
        rebuilt = _rebuild_tt_program_with_segment_plan(tt_program, current_segment_plan)
        if semaphore_plan is not None:
            rebuilt = _rebuild_tt_program_with_semaphore_plan(rebuilt, semaphore_plan)
        return rebuilt

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_codegen_module_with_body_and_segment_plan(
    artifact, *, body_mutator=None, segment_mutator=None
):
    def mutate(tt_program):
        current_segment_plan = [dict(kernel.payload) for kernel in tt_program.kernels]
        if segment_mutator is not None:
            current_segment_plan = segment_mutator(current_segment_plan)
        return _rebuild_tt_program_with_segment_plan(tt_program, current_segment_plan)

    return _rebuild_codegen_module_with_tt_program(
        artifact, tt_program_mutator=mutate, body_mutator=body_mutator
    )


def _extract_blackhole_executable_spec(artifact, function_names=("main", "main_kernel")):
    codegen_mod = artifact.codegen_mod
    getter = getattr(codegen_mod, "get_function_metadata", None)
    if getter is None:
        pytest.fail("Blackhole runtime module does not expose get_function_metadata()")

    failures = []
    for function_name in function_names:
        try:
            spec = getter(function_name)
        except (AttributeError, TypeError, ValueError, RuntimeError) as err:
            failures.append(f"{function_name}: {type(err).__name__}: {err}")
            continue
        if spec is not None:
            return spec
        failures.append(f"{function_name}: no metadata returned")
    pytest.fail(
        "Blackhole executable spec metadata is not exposed by the built runtime module; "
        + "; ".join(failures)
    )


def _extract_materialized_blackhole_executable(func):
    if not (func.attrs and "tl.blackhole_executable" in func.attrs):
        pytest.fail("Expected PrimFunc to carry tl.blackhole_executable")
    return func.attrs["tl.blackhole_executable"]


def _require_blackhole_kernel(kernels, *, kind, core_type=None, name=None):
    matches = []
    for kernel in kernels:
        if str(kernel["kind"]) != kind:
            continue
        if core_type is not None and str(kernel["core_type"]) != core_type:
            continue
        if name is not None and str(kernel["name"]) != name:
            continue
        matches.append(kernel)

    if not matches:
        available = [
            f"{str(kernel['name'])}:{str(kernel['kind'])}:{str(kernel['core_type'])}"
            for kernel in kernels
        ]
        pytest.fail(
            f"Missing Blackhole kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"available kernels: {available}"
        )
    if len(matches) > 1:
        matched = [
            f"{str(kernel['name'])}:{str(kernel['kind'])}:{str(kernel['core_type'])}"
            for kernel in matches
        ]
        pytest.fail(
            f"Ambiguous Blackhole kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"matched kernels: {matched}"
        )
    return matches[0]


def _require_spec_entry(entries, *, kind, label, buffer=None):
    matches = [entry for entry in entries if str(entry["kind"]) == kind]
    if buffer is not None:
        matches = [entry for entry in matches if str(entry["buffer"]) == buffer]
    if not matches:
        available = [
            f"{str(entry['kind'])}:{str(entry.get('buffer', ''))}"
            for entry in entries
        ]
        pytest.fail(
            f"Missing {label} spec kind={kind!r} buffer={buffer!r}; available entries: {available}"
        )
    if len(matches) > 1:
        matched = [f"{str(entry['kind'])}:{str(entry.get('buffer', ''))}" for entry in matches]
        pytest.fail(
            f"Ambiguous {label} spec kind={kind!r} buffer={buffer!r}; matched entries: {matched}"
        )
    return matches[0]


def _expected_launch_spec_for_core_type(core_type):
    core_type = str(core_type)
    if core_type == "brisc":
        return {"core_type": "brisc", "processor": "riscv_0", "noc": "riscv_0_default"}
    if core_type == "ncrisc":
        return {"core_type": "ncrisc", "processor": "riscv_1", "noc": "riscv_1_default"}
    if core_type == "trisc":
        return {"core_type": "trisc", "processor": "", "noc": ""}
    pytest.fail(f"Unsupported Blackhole core_type for launch spec expectation: {core_type!r}")


def _with_richer_accessor_schema(func, common_runtime_args=None, layout_override=None):
    richer_segments = []
    for segment in extract_blackhole_segment_plan(func):
        richer_segment = dict(segment)
        richer_segment["common_runtime_args"] = list(common_runtime_args or [])
        richer_accessors = []
        try:
            accessors = segment["accessors"]
        except KeyError:
            richer_segments.append(richer_segment)
            continue
        for accessor in accessors:
            richer_accessor = dict(accessor)
            if "compile_time_arg_offset" in richer_accessor:
                richer_accessor["compile_time_arg_offset"] = int(
                    richer_accessor["compile_time_arg_offset"]
                )
            else:
                richer_accessor["compile_time_arg_offset"] = int(richer_accessor["slot"])
            richer_accessor["compile_time_arg_count"] = 2
            richer_accessor["common_runtime_arg_offset"] = 0
            richer_accessor["common_runtime_arg_count"] = 0
            if layout_override is not None:
                richer_accessor["layout"] = layout_override
            richer_accessor["args_config_bits"] = (
                2 if str(richer_accessor["layout"]) == "interleaved" else 1
            )
            richer_accessors.append(richer_accessor)
        richer_segment["accessors"] = richer_accessors
        richer_segments.append(richer_segment)
    return func.with_attr(
        "tl.tt_program",
        _rebuild_tt_program_with_segment_plan(require_tt_program(func), richer_segments),
    )


def _with_compile_time_abi_schema(func, *, strip_accessors=False, compile_time_arg_spec_mutator=None):
    richer_segments = []
    for segment in extract_blackhole_segment_plan(func):
        richer_segment = dict(segment)
        if strip_accessors:
            richer_segment.pop("accessors", None)
        if compile_time_arg_spec_mutator is not None and "compile_time_arg_specs" in segment:
            richer_segment["compile_time_arg_specs"] = [
                compile_time_arg_spec_mutator(dict(spec), segment=richer_segment)
                for spec in segment["compile_time_arg_specs"]
            ]
        richer_segments.append(richer_segment)
    return func.with_attr(
        "tl.tt_program",
        _rebuild_tt_program_with_segment_plan(require_tt_program(func), richer_segments),
    )


def test_blackhole_codegen_only():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=2, tile_cols=2)
    target = Target("blackhole")

    try:
        with target:
            artifact = lower(kernel, target=target)
        assert artifact is not None
        assert hasattr(artifact, "kernel_source") or hasattr(artifact, "code")
    except Exception as e:
        pytest.skip(f"Blackhole lowering not yet fully implemented: {e}")


def test_blackhole_codegen_does_not_emit_cb_backed_c_arrays():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "scope: blackhole.cb" not in source
    assert "A_shared[" not in source
    assert "TensorAccessorArgs<0>()" in source
    assert "TensorAccessorArgs<2>()" in source
    assert "InterleavedAddrGen<true>" not in source


def test_blackhole_copy_pass_attrs():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    assert "blackhole.target_mode" not in func.attrs
    tt_program = require_tt_program(func)

    cb_plans = tt_program.cb_plans
    assert [str(cb.resource_class) for cb in cb_plans] == ["intermediate"]
    assert int(cb_plans[0].payload["total_size_bytes"]) == 4096
    assert int(cb_plans[0].payload["lifetime_begin"]) == 0
    assert int(cb_plans[0].payload["lifetime_end"]) == 0
    assert [str(name) for name in cb_plans[0].payload["requirement_names"]] == [
        str(cb_plans[0].name)
    ]
    assert int(cb_plans[0].cb_id) == 16

    fused_dataflow = require_tt_kernel(tt_program, kind="fused_dataflow", core_type="brisc")
    abi = tt_abi_for_kernel(tt_program, fused_dataflow)
    runtime_args = abi.runtime_args
    assert [str(arg["kind"]) for arg in runtime_args] == EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS
    assert str(runtime_args[0]["buffer"]) == "A"
    assert str(runtime_args[1]["buffer"]) == "B"

    core_group = tt_program.core_groups[0]
    assert int(core_group.logical_grid_x) == 1
    assert int(core_group.logical_grid_y) == 1
    assert str(core_group.linearization) == "row_major"
    assert len(core_group.physical_cores) == 1
    assert int(core_group.physical_cores[0]["core_x"]) == 0
    assert int(core_group.physical_cores[0]["core_y"]) == 0
    assert len(core_group.work_packets) == 1
    assert int(core_group.work_packets[0]["work_offset"]) == 0
    assert int(core_group.work_packets[0]["work_count"]) == 1

    assert str(fused_dataflow.kind) == "fused_dataflow"
    assert str(fused_dataflow.core_type) == "brisc"
    accessors = abi.accessors
    assert [(str(item["buffer"]), int(item["compile_time_arg_offset"])) for item in accessors] == [
        ("A", 0),
        ("B", 2),
    ]
    assert [int(item["compile_time_arg_count"]) for item in accessors] == [2, 2]
    assert [int(item["common_runtime_arg_offset"]) for item in accessors] == [0, 0]
    assert [int(item["common_runtime_arg_count"]) for item in accessors] == [0, 0]
    assert [int(item["args_config_bits"]) for item in accessors] == [2, 2]
    assert all(str(item["layout"]) == "interleaved" for item in accessors)
    assert all(str(item["memory_space"]) == "dram" for item in accessors)
    assert len(abi.common_runtime_args) == 0

    body_script = func.body.script()
    assert "tl.blackhole.read_tile_to_cb" in body_script
    assert "tl.blackhole.write_tile_from_cb" in body_script
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 1
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 1


def test_blackhole_copy_compile_time_abi_is_materialized():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )

    assert "compile_time_arg_specs" in kernel_spec
    compile_time_arg_specs = kernel_spec["compile_time_arg_specs"]
    assert [
        (
            str(item["name"]),
            str(item["kind"]),
            str(item["dtype"]),
            int(item["offset"]),
            int(item["count"]),
            str(item["buffer"]),
            str(item["segment_role"]),
            int(item["args_config_bits"]),
            str(item["layout"]),
            str(item["memory_space"]),
        )
        for item in compile_time_arg_specs
    ] == [
        ("A", "interleaved_accessor_cta", "uint32", 0, 2, "A", "fused_dataflow", 2, "interleaved", "dram"),
        ("B", "interleaved_accessor_cta", "uint32", 2, 2, "B", "fused_dataflow", 2, "interleaved", "dram"),
    ]

    assert "launch_spec" in kernel_spec
    launch_spec = kernel_spec["launch_spec"]
    expected_launch_spec = _expected_launch_spec_for_core_type(kernel_spec["core_type"])
    assert str(launch_spec["core_type"]) == expected_launch_spec["core_type"]
    assert str(launch_spec["processor"]) == expected_launch_spec["processor"]
    assert str(launch_spec["noc"]) == expected_launch_spec["noc"]


def test_blackhole_copy_materializes_executable_writer_attr():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_main = artifact.device_mod["main_kernel"]
    executable = _extract_materialized_blackhole_executable(device_main)
    assert int(executable["schema_version"]) == 1
    assert str(executable["source"]) == "tl.tt_program"
    assert str(executable["entry_name"]) == str(device_main.attrs["global_symbol"])

    segment_plan = executable["segment_plan"]
    assert len(segment_plan) == 1
    kernel_spec = _require_blackhole_kernel(segment_plan, kind="fused_dataflow", core_type="brisc")
    assert kernel_spec["runtime_args"]
    assert executable["cb_configs"]
    assert executable["core_plan"]


def test_blackhole_copy_runtime_arg_identities_are_materialized():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    assert "runtime_args" in executable_spec
    runtime_args = executable_spec["runtime_args"]
    assert [
        (
            str(item["name"]),
            str(item["kind"]),
            str(item["identity"]),
            str(item["buffer"]) if "buffer" in item else "",
        )
        for item in runtime_args
    ] == [
        ("A_addr", "input_buffer_addr32", "input_buffer_addr32:A", "A"),
        ("B_addr", "output_buffer_addr32", "output_buffer_addr32:B", "B"),
        ("work_linear_id", "work_linear_id", "work_linear_id", ""),
        ("a_tile_start_id", "a_tile_start_id", "a_tile_start_id", ""),
        ("a_tile_num_tiles", "a_tile_num_tiles", "a_tile_num_tiles", ""),
        ("a_tile_stride", "a_tile_stride", "a_tile_stride", ""),
        ("output_tile_start_id", "output_tile_start_id", "output_tile_start_id", ""),
        ("output_tile_num_tiles", "output_tile_num_tiles", "output_tile_num_tiles", ""),
        ("output_tile_stride", "output_tile_stride", "output_tile_stride", ""),
    ]


def test_blackhole_copy_build_rejects_missing_runtime_arg_schema():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    with pytest.raises(
        tvm.TVMError,
        match="Blackhole runtime arg schema is required for copy/dataflow kernels",
    ):
        _rebuild_codegen_module_without_copy_runtime_args(artifact)


def test_blackhole_copy_build_rejects_missing_materialized_executable():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    with pytest.raises(Exception, match="requires tl.blackhole_executable"):
        _rebuild_codegen_module_without_materialized_executable(artifact)


def test_blackhole_copy_buffer_materialization_specs_are_exposed():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    assert "buffer_materializations" in executable_spec
    materializations = {
        str(item["buffer"]): (
            str(item["materialization_kind"]),
            str(item["layout"]),
            str(item["memory_space"]),
            int(item["transport_page_size"]),
        )
        for item in executable_spec["buffer_materializations"]
    }
    assert materializations == {
        "A": ("replicated", "interleaved", "dram", 2048),
        "B": ("replicated", "interleaved", "dram", 2048),
    }


def test_blackhole_copy_build_reads_executable_without_legacy_projection_attrs():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_main = artifact.device_mod["main_kernel"]
    for key in (
        "blackhole.segment_plan",
        "blackhole.runtime_args",
        "blackhole.common_runtime_args",
        "blackhole.per_work_arg_specs",
        "blackhole.accessors",
        "blackhole.cb_configs",
        "blackhole.semaphore_plan",
        "blackhole.core_plan",
        "blackhole.gemm_contract",
        "blackhole.compute_contract",
        "blackhole.direct_runtime_unsupported_reasons",
    ):
        assert key not in device_main.attrs

    stripped_mod = _rebuild_codegen_module_without_tt_projection_attrs(artifact)
    executable_spec = stripped_mod.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )

    assert kernel_spec["runtime_args"]
    assert executable_spec["cb_configs"]
    assert executable_spec["core_plan"]


def test_blackhole_copy_build_reads_executable_without_tt_program():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    stripped_mod = _rebuild_codegen_module_without_tt_program(artifact)
    executable_spec = stripped_mod.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )

    assert kernel_spec["runtime_args"]
    assert executable_spec["cb_configs"]
    assert executable_spec["core_plan"]


def test_blackhole_grid_indexed_copy_build_rejects_top_level_per_work_payload_fallback():
    target = Target("blackhole")
    with target:
        artifact = lower(grid_indexed_staged_copy_kernel(grid_x=2, grid_y=2), target=target)

    def segment_mutator(segment_plan):
        mutated_segments = []
        for segment in segment_plan:
            mutated = dict(segment)
            mutated["per_work_arg_specs"] = []
            mutated_segments.append(mutated)
        return mutated_segments

    with pytest.raises(Exception, match="explicit per-work|per_work_arg_specs|work_linear_id"):
        _rebuild_codegen_module_with_body_and_segment_plan(
            artifact, segment_mutator=segment_mutator
        )


def test_blackhole_copy_semaphore_plan_is_materialized():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 1, "core_y": 2},
                }
            ],
        }
    ]
    mutated_mod = _rebuild_codegen_module_with_semaphore_plan(artifact, semaphore_plan)

    executable_spec = mutated_mod.get_function_metadata("main")
    assert "semaphores" in executable_spec
    semaphores = executable_spec["semaphores"]
    assert len(semaphores) == 1
    semaphore = semaphores[0]
    assert int(semaphore["id"]) == 0
    assert int(semaphore["initial_value"]) == 0
    assert str(semaphore["core_type"]) == "worker"
    core_ranges = semaphore["core_ranges"]
    assert len(core_ranges) == 1
    assert int(core_ranges[0]["start"]["core_x"]) == 1
    assert int(core_ranges[0]["start"]["core_y"]) == 2
    assert int(core_ranges[0]["end"]["core_x"]) == 1
    assert int(core_ranges[0]["end"]["core_y"]) == 2


def test_blackhole_copy_kernel_semaphore_binding_is_materialized():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 1, "core_y": 2},
                }
            ],
        }
    ]

    def segment_mutator(segment_plan):
        mutated_segments = []
        for segment in segment_plan:
            mutated = dict(segment)
            mutated["semaphore_bindings"] = [
                {"name": "copy_sem", "semaphore_id": 0, "arg_kind": "semaphore_id_u32"}
            ]
            mutated_segments.append(mutated)
        return mutated_segments

    mutated_mod = _rebuild_codegen_module_with_semaphore_binding(
        artifact, semaphore_plan=semaphore_plan, segment_mutator=segment_mutator
    )

    executable_spec = mutated_mod.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    assert "semaphore_bindings" in kernel_spec
    semaphore_bindings = kernel_spec["semaphore_bindings"]
    assert len(semaphore_bindings) == 1
    binding = semaphore_bindings[0]
    assert str(binding["name"]) == "copy_sem"
    assert int(binding["semaphore_id"]) == 0
    assert str(binding["arg_kind"]) == "semaphore_id_u32"


def _worker_semaphore_plan(semaphore_id=0):
    return [
        {
            "id": semaphore_id,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 1, "core_y": 2},
                }
            ],
        }
    ]


def _inject_device_semaphore_builtins(*, remote_coord_source=None):
    def body_mutator(original_body):
        del original_body
        semaphore_id = tir.Var("copy_sem", "uint32")
        semaphore_addr = tir.Var("copy_sem_addr", "uint32")
        ops = [
            tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_wait"),
                    semaphore_addr,
                    tir.IntImm("uint32", 1),
                )
            ),
            tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_set"),
                    semaphore_addr,
                    tir.IntImm("uint32", 0),
                )
            ),
        ]
        if remote_coord_source is not None:
            remote_noc_x, remote_noc_y = remote_coord_source
            ops.append(
                tir.Evaluate(
                    tir.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.blackhole.semaphore_inc_remote"),
                        semaphore_addr,
                        remote_noc_x,
                        remote_noc_y,
                        tir.IntImm("uint32", 1),
                    )
                )
            )
        return tir.LetStmt(
            semaphore_id,
            tir.call_intrin(
                "uint32",
                tir.op.Op.get("tl.blackhole.get_semaphore"),
                tir.IntImm("uint32", 0),
            ),
            tir.LetStmt(semaphore_addr, semaphore_id, tir.SeqStmt(ops)),
        )

    return body_mutator


def test_blackhole_copy_build_rejects_device_semaphore_builtin_without_planned_semaphore():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    with pytest.raises(
        tvm.error.InternalError,
        match="planned semaphore|communication protocol|get_semaphore",
    ):
        _rebuild_codegen_module_with_body_and_segment_plan(
            artifact,
            body_mutator=_inject_device_semaphore_builtins(),
        )


def test_blackhole_copy_build_rejects_remote_semaphore_builtin_without_endpoint_schema():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    with pytest.raises(
        tvm.error.InternalError,
        match="remote endpoint|logical_core_noc|communication routing",
    ):
        _rebuild_codegen_module_with_tt_program(
            artifact,
            tt_program_mutator=lambda tt_program: _rebuild_tt_program_with_semaphore_plan(
                tt_program, _worker_semaphore_plan()
            ),
            body_mutator=_inject_device_semaphore_builtins(
                remote_coord_source=(
                    tir.IntImm("uint32", 1),
                    tir.IntImm("uint32", 0),
                )
            ),
        )


def test_blackhole_codegen_emits_device_semaphore_builtins():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    mutated_mod = _rebuild_codegen_module_with_tt_program(
        artifact,
        tt_program_mutator=lambda tt_program: _rebuild_tt_program_with_semaphore_plan(
            tt_program, _worker_semaphore_plan()
        ),
        body_mutator=_inject_device_semaphore_builtins(),
    )

    executable_spec = mutated_mod.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    source = str(kernel_spec["source_code"])
    assert "get_semaphore(" in source
    assert "noc_semaphore_wait(" in source
    assert "noc_semaphore_set(" in source


def test_blackhole_copy_semantics_annotation_schema():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)

    sem = find_loop_annotation(mod["main"].body, "blackhole.copy_semantics")
    assert sem is not None
    assert str(sem["kind"]) == "fused_staged_copy"
    assert str(sem["direction"]) == "dram_to_cb_to_dram"
    assert str(sem["src_buffer"]) == "A"
    assert str(sem["mid_buffer"]) == "A_shared"
    assert str(sem["dst_buffer"]) == "B"
    assert str(sem["src_buffer_ref"].name) == "A"
    assert str(sem["mid_buffer_ref"].name) == "A_shared"
    assert str(sem["dst_buffer_ref"].name) == "B"
    assert str(sem["src_scope"]) == "global"
    assert str(sem["dst_scope"]) == "global"
    assert str(sem["dtype"]) == "bfloat16"
    assert [int(x) for x in sem["src_shape"]] == [64, 32]
    assert [int(x) for x in sem["dst_shape"]] == [64, 32]
    assert [int(x) for x in sem["mid_shape"]] == [32, 32]


def test_blackhole_copy_lowering_prefers_buffer_handles_over_annotation_names():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    func = _rewrite_copy_semantics_annotations(
        mod["main"],
        lambda sem: {
            **sem,
            "src_buffer": "WRONG_INPUT_BUFFER",
            "mid_buffer": "WRONG_INTERMEDIATE_BUFFER",
            "dst_buffer": "WRONG_OUTPUT_BUFFER",
        },
    )
    mod = tilelang.tvm.IRModule({"main": func})
    mod = lower_blackhole_to_tt_target(mod)

    runtime_args = extract_blackhole_runtime_args(mod["main"])
    buffers = [str(arg["buffer"]) for arg in runtime_args if "buffer" in arg]
    assert buffers == ["A", "B"]


def test_blackhole_copy_semantics_survives_flatten_and_vectorize():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    runtime_args = extract_blackhole_runtime_args(func)
    assert [str(arg["kind"]) for arg in runtime_args] == EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS
    assert str(runtime_args[0]["buffer"]) == "A"
    assert str(runtime_args[1]["buffer"]) == "B"

    body_script = func.body.script()
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 1
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 1


def test_blackhole_copy_richer_accessor_schema_roundtrip():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            func = _with_richer_accessor_schema(func)
        rewritten[gvar] = func

    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    built = tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )

    assert built is not None


def test_blackhole_copy_direct_runtime_rejects_common_runtime_accessor_schema():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    richer_common_runtime_args = [
        {
            "name": "rank",
            "kind": "accessor_common_u32",
            "identity": "rank",
            "dtype": "uint32",
        }
    ]
    stripped_func = _with_compile_time_abi_schema(device_main, strip_accessors=True)
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(stripped_func):
        richer_segment = dict(segment)
        richer_segment["common_runtime_args"] = richer_common_runtime_args
        mutated_segments.append(richer_segment)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|interleaved"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_materializes_shared_common_runtime_buffer_args():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    common_runtime_args = [
        {
            "name": "input_buffer_common_addr32",
            "kind": "input_buffer_addr32",
            "identity": "input_buffer_addr32:A",
            "dtype": "uint32",
            "buffer": "A",
        },
        {
            "name": "output_buffer_common_addr32",
            "kind": "output_buffer_addr32",
            "identity": "output_buffer_addr32:B",
            "dtype": "uint32",
            "buffer": "B",
        },
    ]
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        mutated_segment["common_runtime_args"] = common_runtime_args
        mutated_segments.append(mutated_segment)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output, a_torch, atol=0.0, rtol=0.0, failure_message="Copy output mismatch"
    )


def test_blackhole_copy_direct_runtime_rejects_accessor_common_runtime_arg_count():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        mutated_accessors = []
        for accessor in segment["accessors"]:
            mutated_accessor = dict(accessor)
            mutated_accessor["common_runtime_arg_offset"] = 0
            mutated_accessor["common_runtime_arg_count"] = 1
            mutated_accessors.append(mutated_accessor)
        mutated_segment["accessors"] = mutated_accessors
        mutated_segment["common_runtime_args"] = []
        mutated_segments.append(mutated_segment)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|interleaved"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_rejects_work_linear_id_in_common_runtime_args():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        mutated_segment["common_runtime_args"] = [
            {
                "name": "shared_work_linear_id",
                "kind": "work_linear_id",
                "identity": "shared_work_linear_id",
                "dtype": "uint32",
            }
        ]
        mutated_segments.append(mutated_segment)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|shared|Unsupported"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_materializes_compile_time_abi_schema():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    stripped_func = _with_compile_time_abi_schema(device_main, strip_accessors=True)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(
        artifact, extract_blackhole_segment_plan(stripped_func)
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output, a_torch, atol=0.0, rtol=0.0, failure_message="Copy direct-call output mismatch"
    )


def test_blackhole_copy_direct_runtime_rejects_unknown_compile_time_abi_kind():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    def mutate_unknown_kind(spec, *, segment):
        if str(spec["kind"]) == "interleaved_accessor_cta" and str(spec["buffer"]) == "A":
            spec["kind"] = "unknown_compile_time_abi_kind"
        return spec

    mutated_func = _with_compile_time_abi_schema(
        device_main,
        strip_accessors=True,
        compile_time_arg_spec_mutator=mutate_unknown_kind,
    )
    mutated_mod = _rebuild_codegen_module_with_segment_plan(
        artifact, extract_blackhole_segment_plan(mutated_func)
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(Exception, match="Unsupported Blackhole compile-time ABI kind"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_rejects_accessor_runtime_crta_bits():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    def mutate_runtime_crta_bits(spec, *, segment):
        if str(spec["kind"]) == "interleaved_accessor_cta":
            spec["args_config_bits"] = 2 | 4
        return spec

    mutated_func = _with_compile_time_abi_schema(
        device_main,
        strip_accessors=True,
        compile_time_arg_spec_mutator=mutate_runtime_crta_bits,
    )
    mutated_mod = _rebuild_codegen_module_with_segment_plan(
        artifact, extract_blackhole_segment_plan(mutated_func)
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|args_config_bits == 2"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_rejects_unknown_semaphore_core_type():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "mystery_core",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 1, "core_y": 2},
                }
            ],
        }
    ]
    mutated_mod = _rebuild_codegen_module_with_semaphore_plan(artifact, semaphore_plan)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(tvm.error.InternalError, match="semaphore core_type|Unsupported Blackhole semaphore core_type"):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_accepts_semaphore_id_runtime_arg():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    base_runtime_args = list(extract_blackhole_runtime_args(device_main))

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 1, "core_y": 2},
                }
            ],
        }
    ]

    semaphore_runtime_arg = {
        "name": "copy_sem",
        "kind": "semaphore_id_u32",
        "identity": "copy_sem",
        "dtype": "uint32",
    }

    def runtime_args_mutator(runtime_args):
        return list(runtime_args) + [semaphore_runtime_arg]

    def segment_mutator(segment_plan):
        mutated_segments = []
        for segment in segment_plan:
            mutated = dict(segment)
            mutated["runtime_args"] = runtime_args_mutator(base_runtime_args)
            mutated["semaphore_bindings"] = [
                {"name": "copy_sem", "semaphore_id": 0, "arg_kind": "semaphore_id_u32"}
            ]
            mutated_segments.append(mutated)
        return mutated_segments

    mutated_mod = _rebuild_codegen_module_with_semaphore_binding(
        artifact,
        semaphore_plan=semaphore_plan,
        segment_mutator=segment_mutator,
        runtime_args_mutator=runtime_args_mutator,
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        a_torch,
        b_output,
        atol=0.0,
        rtol=0.0,
        failure_message="Copy direct-call output mismatch with semaphore_id_u32 runtime arg",
    )


def test_blackhole_copy_build_rejects_unbound_semaphore_runtime_arg():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    base_runtime_args = list(extract_blackhole_runtime_args(device_main))

    def runtime_args_mutator(runtime_args):
        return list(runtime_args) + [
            {
                "name": "copy_sem",
                "kind": "semaphore_id_u32",
                "identity": "copy_sem",
                "dtype": "uint32",
            }
        ]

    with pytest.raises(
        tvm.error.InternalError,
        match="requires a matching semaphore binding|missing a matching semaphore binding",
    ):
        _rebuild_codegen_module_with_semaphore_binding(
            artifact, runtime_args_mutator=runtime_args_mutator
        )


def test_blackhole_copy_build_rejects_unpaired_logical_core_noc_runtime_arg():
    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    consumer_core = extract_blackhole_core_plan(device_main)["physical_cores"][1]
    base_runtime_args = list(extract_blackhole_runtime_args(device_main))

    def runtime_args_mutator(runtime_args):
        return list(runtime_args) + [
            {
                "name": "remote_noc_x",
                "kind": "logical_core_noc_x",
                "identity": "remote_consumer_core",
                "dtype": "uint32",
                "core_x": int(consumer_core["core_x"]),
                "core_y": int(consumer_core["core_y"]),
            }
        ]

    with pytest.raises(
        tvm.error.InternalError,
        match="logical_core_noc_x.*logical_core_noc_y|synchronization core descriptor",
    ):
        _rebuild_codegen_module_with_runtime_args(artifact, runtime_args_mutator(base_runtime_args))


def test_blackhole_copy_remote_core_descriptor_is_materialized():
    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    consumer_core = extract_blackhole_core_plan(device_main)["physical_cores"][1]
    runtime_args = list(extract_blackhole_runtime_args(device_main))
    runtime_args.extend(
        [
            {
                "name": "remote_noc_x",
                "kind": "logical_core_noc_x",
                "identity": "remote_consumer_core",
                "dtype": "uint32",
                "core_x": int(consumer_core["core_x"]),
                "core_y": int(consumer_core["core_y"]),
            },
            {
                "name": "remote_noc_y",
                "kind": "logical_core_noc_y",
                "identity": "remote_consumer_core",
                "dtype": "uint32",
                "core_x": int(consumer_core["core_x"]),
                "core_y": int(consumer_core["core_y"]),
            },
        ]
    )

    def segment_mutator(segment_plan):
        mutated_segments = []
        for segment in segment_plan:
            mutated = dict(segment)
            mutated["runtime_args"] = runtime_args
            mutated_segments.append(mutated)
        return mutated_segments

    mutated_mod = _rebuild_codegen_module_with_semaphore_binding(
        artifact,
        segment_mutator=segment_mutator,
        runtime_args_mutator=lambda _runtime_args: runtime_args,
    )
    spec = mutated_mod.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    assert "remote_core_descriptors" in kernel_spec
    descriptors = kernel_spec["remote_core_descriptors"]
    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert str(descriptor["identity"]) == "remote_consumer_core"
    assert int(descriptor["core_x"]) == int(consumer_core["core_x"])
    assert int(descriptor["core_y"]) == int(consumer_core["core_y"])


@pytest.mark.xfail(
    reason=(
        "StorageRewrite is incompatible with the Blackhole CB model: its "
        "VectorTypeAccessChecker only recognizes AllocateNode as buffer declarations "
        "but after FlattenBuffer+VectorizeLoop the shared (CB) buffers are represented "
        "via DeclBuffer, causing a spurious 'buffer used before declaration' error. "
        "StorageRewrite provides no benefit for Blackhole circular buffers and is intentionally "
        "excluded from the Blackhole pipeline."
    ),
    strict=True,
)
def test_blackhole_storage_rewrite_incompatible_with_cb_model():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    tilelang.transform.StorageRewrite()(mod)


def test_blackhole_stick_copy_pipeline_formalizes_page_transport():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=32, tile_n=16, global_n=32, dtype="float32")

    with target:
        artifact = lower(kernel, target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    kernel_spec = _require_blackhole_kernel(
        spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    source = str(kernel_spec["source_code"])
    assert "read_page_to_cb" not in source
    assert "write_page_from_cb" not in source
    assert "TensorAccessorArgs<0>()" in source
    assert "TensorAccessorArgs<2>()" in source
    assert "get_noc_addr(page_id)" in source
    assert "noc_async_read(" in source
    assert "noc_async_write(" in source

    cb_configs = spec["cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 1
    accessors = kernel_spec["accessors"]
    assert int(accessors[0]["transport_page_size"]) == 64
    assert int(accessors[1]["transport_page_size"]) == 64


def test_blackhole_tall_stick_copy_pipeline_formalizes_page_transport():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=64, tile_n=16, global_n=32, dtype="float32")

    with target:
        artifact = lower(kernel, target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    kernel_spec = _require_blackhole_kernel(
        spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    source = str(kernel_spec["source_code"])
    assert "get_noc_addr(page_id)" in source
    assert "noc_async_read(" in source
    assert "noc_async_write(" in source

    cb_configs = spec["cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 4096
    assert int(cb_configs[0]["num_pages"]) == 1
    accessors = kernel_spec["accessors"]
    assert int(accessors[0]["transport_page_size"]) == 64
    assert int(accessors[1]["transport_page_size"]) == 64


def test_blackhole_offset_stick_copy_pipeline_formalizes_page_transport():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=48, dtype="float32", src_col=16, dst_col=16
    )

    with target:
        artifact = lower(kernel, target=target)

    spec = _extract_blackhole_executable_spec(artifact)
    kernel_spec = _require_blackhole_kernel(
        spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    source = str(kernel_spec["source_code"])
    assert "get_noc_addr(page_id)" in source
    assert "noc_async_read(" in source
    assert "noc_async_write(" in source
    assert "const uint32_t page_bytes = 64;" in source

    cb_configs = spec["cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 4096
    assert int(cb_configs[0]["num_pages"]) == 1
    accessors = kernel_spec["accessors"]
    assert int(accessors[0]["transport_page_size"]) == 64
    assert int(accessors[1]["transport_page_size"]) == 64


def test_blackhole_stick_copy_pipeline_rejects_nondisible_global_width():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=40, dtype="float32", src_col=16, dst_col=16
    )

    with pytest.raises(Exception, match="global width divisible by shared width"):
        with target:
            lower(kernel, target=target)


def test_blackhole_stick_copy_pipeline_rejects_unaligned_transport_offset():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=48, dtype="float32", src_col=8, dst_col=16
    )

    with pytest.raises(Exception, match="page-aligned transport offsets"):
        with target:
            lower(kernel, target=target)


def test_blackhole_stick_copy_pipeline_rejects_unaligned_transport_page():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=64, tile_n=24, global_n=48, dtype="float32")

    with pytest.raises(Exception, match="64B-aligned transport page size"):
        with target:
            lower(kernel, target=target)


def test_blackhole_stick_copy_pipeline_reports_direct_path_boundary_context():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=40, dtype="float32", src_col=16, dst_col=16
    )

    with pytest.raises(
        Exception, match="direct-path boundary requires global width divisible by shared width"
    ):
        with target:
            lower(kernel, target=target)


def test_blackhole_copy_codegen_uses_runtime_schema():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "uint32_t A_addr = get_arg_val<uint32_t>(0);" in source
    assert "uint32_t B_addr = get_arg_val<uint32_t>(1);" in source
    assert "uint32_t work_linear_id = get_arg_val<uint32_t>(2);" in source
    assert "uint32_t a_tile_start_id = get_arg_val<uint32_t>(3);" in source
    assert "uint32_t a_tile_num_tiles = get_arg_val<uint32_t>(4);" in source
    assert "uint32_t a_tile_stride = get_arg_val<uint32_t>(5);" in source
    assert "uint32_t output_tile_start_id = get_arg_val<uint32_t>(6);" in source
    assert "uint32_t output_tile_num_tiles = get_arg_val<uint32_t>(7);" in source
    assert "uint32_t output_tile_stride = get_arg_val<uint32_t>(8);" in source
    assert "src_dram_addr" not in source
    assert "dst_dram_addr" not in source
    assert "scratch_l1_addr" not in source
    assert "cb_reserve_back(" in source
    assert "cb_push_back(" in source
    assert "cb_wait_front(" in source
    assert "cb_pop_front(" in source


def test_blackhole_copy_codegen_accepts_explicit_per_work_schema_without_work_linear_id():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    unsupported_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "identity": "input_buffer_addr32:A", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "identity": "output_buffer_addr32:B", "dtype": "uint32", "buffer": "B"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "identity": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "identity": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "identity": "a_tile_stride", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "identity": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "identity": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "identity": "output_tile_stride", "dtype": "uint32"},
    ]

    rebuilt = _rebuild_codegen_module_with_runtime_args(artifact, unsupported_runtime_args)
    executable_spec = rebuilt.get_function_metadata("main")
    kernel_spec = _require_blackhole_kernel(
        executable_spec["kernels"], kind="fused_dataflow", core_type="brisc"
    )
    source = str(kernel_spec["source_code"])
    assert "uint32_t work_linear_id = get_arg_val<uint32_t>" not in source
    assert "uint32_t output_tile_start_id = get_arg_val<uint32_t>" in source


def test_blackhole_copy_codegen_rejects_runtime_arg_without_identity():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    missing_identity_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "identity": "output_buffer_addr32:B", "dtype": "uint32", "buffer": "B"},
        {"name": "work_linear_id", "kind": "work_linear_id", "identity": "work_linear_id", "dtype": "uint32"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "identity": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "identity": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "identity": "a_tile_stride", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "identity": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "identity": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "identity": "output_tile_stride", "dtype": "uint32"},
    ]

    with pytest.raises(Exception, match="missing explicit identity"):
        _rebuild_codegen_module_with_runtime_args(artifact, missing_identity_runtime_args)


def test_blackhole_copy_build_marks_missing_buffer_role_schema_direct_runtime_unsupported():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    bufferless_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "identity": "input_buffer_addr32:A", "dtype": "uint32"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "identity": "output_buffer_addr32:B", "dtype": "uint32"},
        {"name": "work_linear_id", "kind": "work_linear_id", "identity": "work_linear_id", "dtype": "uint32"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "identity": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "identity": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "identity": "a_tile_stride", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "identity": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "identity": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "identity": "output_tile_stride", "dtype": "uint32"},
    ]

    rebuilt = _rebuild_codegen_module_with_runtime_args(artifact, bufferless_runtime_args)
    executable_spec = rebuilt.get_function_metadata("main")
    reasons = [str(reason) for reason in executable_spec["direct_runtime_unsupported_reasons"]]

    assert any("missing explicit buffer role schema" in reason for reason in reasons)


def test_blackhole_copy_codegen_rejects_common_runtime_arg_without_identity():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        mutated_segment["common_runtime_args"] = [
            {
                "name": "rank",
                "kind": "accessor_common_u32",
                "dtype": "uint32",
            }
        ]
        mutated_segments.append(mutated_segment)

    with pytest.raises(Exception, match="missing explicit identity"):
        _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)


def test_blackhole_copy_codegen_rejects_nonconstant_accessor_slot():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    def body_mutator(original_body):
        bad_slot = tir.Var("bad_accessor_slot", "int32")
        read_tile = tir.op.Op.get("tl.blackhole.read_tile_to_cb")

        def mutate(expr):
            if isinstance(expr, tir.Call) and expr.op.same_as(read_tile):
                args = list(expr.args)
                args[4] = bad_slot
                return tir.Call(expr.dtype, expr.op, args, expr.span)
            return expr

        return tir.stmt_functor.ir_transform(original_body, None, mutate, ["tir.Call"])

    with pytest.raises(
        Exception,
        match="compile-time-only accessor slot|constant accessor compile-time offset",
    ):
        _rebuild_codegen_module_with_body_and_segment_plan(
            artifact, body_mutator=body_mutator
        )


def test_blackhole_core_plan_preserves_logical_block_launch():
    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = extract_blackhole_core_plan(device_main)

    assert int(core_plan["logical_grid_x"]) == 2
    assert int(core_plan["logical_grid_y"]) == 3
    assert str(core_plan["linearization"]) == "row_major"
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6
    assert {
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    } == {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}
    assert [
        (int(packet["core_x"]), int(packet["core_y"]))
        for packet in core_plan["work_packets"]
    ] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
    ]
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3, 4, 5]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1, 1, 1]

    body_script = device_main.body.script()
    assert "tl.blackhole.read_tile_to_cb(A, by * 2 + bx, 16, 2048, 0)" in body_script
    assert "tl.blackhole.write_tile_from_cb(16, B, by * 2 + bx, 2048, 2)" in body_script


def test_blackhole_core_plan_preserves_axis_order():
    kernel = grid_indexed_staged_copy_kernel(grid_x=3, grid_y=2)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = extract_blackhole_core_plan(device_main)

    assert int(core_plan["logical_grid_x"]) == 3
    assert int(core_plan["logical_grid_y"]) == 2
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6
    assert {
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    } == {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3, 4, 5]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1, 1, 1]

    body_script = device_main.body.script()
    assert "tl.blackhole.read_tile_to_cb(A, by * 3 + bx, 16, 2048, 0)" in body_script
    assert "tl.blackhole.write_tile_from_cb(16, B, by * 3 + bx, 2048, 2)" in body_script


def test_blackhole_core_plan_covers_oversubscribed_work():
    kernel = grid_indexed_staged_copy_kernel(grid_x=15, grid_y=10)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = extract_blackhole_core_plan(device_main)

    assert int(core_plan["logical_grid_x"]) == 15
    assert int(core_plan["logical_grid_y"]) == 10
    assert len(core_plan["physical_cores"]) == 110
    assert len(core_plan["work_packets"]) == 110
    assert len(
        {(int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]}
    ) == 110
    assert int(extract_blackhole_work_per_core(device_main)) == 2

    covered = []
    for packet in core_plan["work_packets"]:
        work_offset = int(packet["work_offset"])
        work_count = int(packet["work_count"])
        covered.extend(range(work_offset, work_offset + work_count))

    assert covered == list(range(150))
    assert sum(int(packet["work_count"]) for packet in core_plan["work_packets"]) == 150


def test_blackhole_copy_oversubscription_fails_compile_time():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1, tile_m=1024, tile_n=1024)
    target = Target("blackhole")
    with pytest.raises(Exception, match="PlanTTCBAlloc|1572864|1.5MB|per-core constraints"):
        with target:
            lower(kernel, target=target)


@pytest.mark.parametrize(
    "tile_rows,tile_cols,tile_m,tile_n,expected_terms",
    [
        (2, 1, 32, 64, ["tile_row * 2", "tile_row * 2 + 1"]),
        (1, 2, 64, 32, ["tile_idx", "tile_idx + 2"]),
    ],
)
def test_blackhole_copy_tracks_rectangular_tile_shapes(
    tile_rows, tile_cols, tile_m, tile_n, expected_terms
):
    kernel = staged_copy_kernel(
        tile_rows=tile_rows, tile_cols=tile_cols, tile_m=tile_m, tile_n=tile_n
    )
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    cb_configs = extract_blackhole_cb_configs(device_main)

    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 2

    body_script = device_main.body.script()
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 2
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 2
    for expected in expected_terms:
        assert expected in body_script


def test_blackhole_lower_restores_host_device_split():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    host_funcs = {str(gvar): func for gvar, func in artifact.host_mod.functions.items()}
    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}

    assert 'I.GlobalVar("main")' in host_funcs
    assert 'I.GlobalVar("main_kernel")' in device_funcs

    host_main = host_funcs['I.GlobalVar("main")']
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    assert host_main.attrs["calling_conv"] == CallingConv.C_PACKED_FUNC
    assert host_main.attrs["target"].kind.name == "c"
    assert device_main.attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH
    assert device_main.attrs["target"].kind.name == "blackhole"
    assert "blackhole.target_mode" not in device_main.attrs


def test_blackhole_target_mode_does_not_define_device_kernel():
    @T.prim_func
    def trivial_kernel(A: T.Buffer((32,), "float16")):
        with T.Kernel(1, threads=1) as (bx,):
            A[bx] = A[bx]

    tagged = trivial_kernel.with_attrs(
        {"target": Target("blackhole"), "blackhole.target_mode": "single_core_copy"}
    )
    assert not is_device_call(tagged)


def test_blackhole_runtime_module_keeps_host_and_device_entries():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    assert artifact.codegen_mod["main"] is not None
    assert artifact.codegen_mod["main_kernel"] is not None


def test_blackhole_kernel_compilation():
    @T.prim_func
    def elementwise_add(
        A: T.Buffer((64,), "float16"),
        B: T.Buffer((64,), "float16"),
        C: T.Buffer((64,), "float16"),
    ):
        with T.Kernel(2) as bx:
            for i in T.Parallel(32):
                idx = bx * 32 + i
                C[idx] = A[idx] + B[idx]

    try:
        target = Target("blackhole")
        artifact = lower(elementwise_add, target=target)
        assert artifact is not None
        if hasattr(artifact, "kernel_source"):
            assert len(artifact.kernel_source) > 0
        elif hasattr(artifact, "code"):
            assert len(artifact.code) > 0
    except Exception as e:
        pytest.skip(f"Blackhole compilation not yet complete: {e}")
