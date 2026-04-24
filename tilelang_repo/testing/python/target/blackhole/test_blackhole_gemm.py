import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import lower, merge_ir_modules
from tilelang import tvm
from tvm.target import Target

from .common import (
    assert_tensors_close_or_dump,
    prepare_blackhole_phase_b_module,
    check_blackhole_codegen_requirements,
    check_blackhole_direct_execution_requirements,
    contains_attr_stmt_key,
    extract_blackhole_compute_contract,
    extract_blackhole_core_plan,
    extract_blackhole_segment_plan,
    gemm_kernel,
    gemm_kernel_with_compute_config_extras,
    gemm_kernel_with_compute_abi,
    fragment_fill_cast_publish_kernel,
    gemm_kernel_with_post_merge_cast_consumer,
    gemm_kernel_with_mbar,
    gemm_kernel_with_policy,
    gemm_kernel_with_transpose_flags,
    lower_blackhole_to_tt_target,
    rebuild_tt_core_group,
    rebuild_tt_kernel,
    rebuild_tt_program,
    require_tt_kernel,
    require_tt_program,
    tt_abi_for_kernel,
)
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
    _expected_launch_spec_for_core_type,
    _rebuild_codegen_module_without_lowering_requirements,
    _rebuild_codegen_module_with_tt_program,
    _rebuild_tt_program_with_segment_plan,
    _refresh_tt_program_after_bridge_attr_mutation,
    _with_compile_time_abi_schema,
    _require_blackhole_kernel,
    _require_spec_entry,
)


def _with_richer_accessor_schema(func, common_runtime_args=None):
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
            richer_accessor["args_config_bits"] = 2 if str(richer_accessor["layout"]) == "interleaved" else 1
            richer_accessors.append(richer_accessor)
        richer_segment["accessors"] = richer_accessors
        richer_segments.append(richer_segment)
    return func.with_attr(
        "tl.tt_program",
        _rebuild_tt_program_with_segment_plan(require_tt_program(func), richer_segments),
    )


def _with_sharded_accessor_schema(func):
    richer_segments = []
    for segment in extract_blackhole_segment_plan(func):
        richer_segment = dict(segment)
        richer_segment["common_runtime_args"] = list(segment["common_runtime_args"])
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
            richer_accessor["layout"] = "sharded"
            richer_accessor["args_config_bits"] = 1
            richer_accessors.append(richer_accessor)
        richer_segment["accessors"] = richer_accessors
        richer_segments.append(richer_segment)
    return func.with_attr(
        "tl.tt_program",
        _rebuild_tt_program_with_segment_plan(require_tt_program(func), richer_segments),
    )


def _with_mutated_segment_plan(func, segment_mutator):
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(func):
        mutated_segments.append(segment_mutator(dict(segment)))
    return func.with_attr(
        "tl.tt_program",
        _rebuild_tt_program_with_segment_plan(require_tt_program(func), mutated_segments),
    )


def _rebuild_codegen_module_with_segment_plan(artifact, segment_plan):
    return _rebuild_codegen_module_with_tt_program(
        artifact,
        tt_program_mutator=lambda tt_program: _rebuild_tt_program_with_segment_plan(
            tt_program, segment_plan
        ),
    )


def _assert_no_contract_family(executable_spec):
    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in executable_spec


def _require_compute_kernel_spec(executable_spec):
    return _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )


def _require_compute_config(executable_spec):
    return dict(_require_compute_kernel_spec(executable_spec)["compute_config"])


def _require_gemm_compute_op(executable_spec):
    compute = _require_compute_kernel_spec(executable_spec)
    return dict(next(item for item in compute["compute_ops"] if str(item["kind"]) == "gemm"))


def _rebuild_codegen_module_with_compute_overrides(artifact, compute_overrides):
    def mutate(tt_program):
        payload = dict(tt_program.payload)
        rebuilt_kernels = []
        for kernel in tt_program.kernels:
            kernel_payload = dict(kernel.payload)
            if str(kernel.kind) == "compute" or str(kernel.core_type) == "trisc":
                compute_config = dict(kernel_payload["compute_config"])
                for key in (
                    "math_fidelity",
                    "fp32_dest_acc_en",
                    "dst_full_sync_en",
                    "math_approx_mode",
                    "unpack_to_dest_mode",
                    "bfp8_pack_precise",
                    "defines",
                    "named_compile_args",
                    "clear_accum",
                    "k_pack",
                    "wg_wait",
                    "policy_type",
                    "policy_name",
                ):
                    if key in compute_overrides:
                        value = compute_overrides[key]
                        compute_config[key] = list(value) if isinstance(value, list) else value
                kernel_payload["compute_config"] = compute_config
                if "compute_ops" in kernel_payload:
                    compute_ops = []
                    for compute_op in kernel_payload["compute_ops"]:
                        compute_op = dict(compute_op)
                        if str(compute_op.get("kind", "")) == "gemm":
                            for key in (
                                "has_mbarrier",
                                "mbarrier_buffer",
                                "mbarrier_scope",
                                "mbarrier_index_exprs",
                            ):
                                if key in compute_overrides:
                                    value = compute_overrides[key]
                                    compute_op[key] = list(value) if isinstance(value, list) else value
                        compute_ops.append(compute_op)
                    kernel_payload["compute_ops"] = compute_ops
            rebuilt_kernels.append(rebuild_tt_kernel(kernel, payload=kernel_payload))
        return rebuild_tt_program(tt_program, kernels=rebuilt_kernels, payload=payload)

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_codegen_module_without_contract_family_payload(artifact):
    def mutate(tt_program):
        payload = dict(tt_program.payload)
        for key in (
            "gemm_contract",
            "compute_contract",
            "multi_gemm_contracts",
            "multi_compute_contracts",
            "compute_epilogue_ops",
        ):
            payload.pop(key, None)
        return rebuild_tt_program(tt_program, payload=payload)

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_codegen_module_without_legacy_contract_attrs(artifact):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "blackhole.gemm_contract" in func.attrs:
            func = func.without_attr("blackhole.gemm_contract")
        if func.attrs and "blackhole.compute_contract" in func.attrs:
            func = func.without_attr("blackhole.compute_contract")
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(
        artifact.host_mod,
        tvm.IRModule(rewritten, global_infos=device_mod.global_infos),
    )
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_with_core_plan(artifact, core_plan):
    def mutate(tt_program):
        core_group = tt_program.core_groups[0]
        rebuilt_core_group = rebuild_tt_core_group(
            core_group,
            logical_grid_x=int(core_plan["logical_grid_x"]),
            logical_grid_y=int(core_plan["logical_grid_y"]),
            linearization=str(core_plan["linearization"]),
            physical_cores=list(core_plan["physical_cores"]),
            work_packets=list(core_plan["work_packets"]),
            payload=dict(core_plan),
        )
        return rebuild_tt_program(tt_program, core_groups=[rebuilt_core_group])

    return _rebuild_codegen_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _direct_runtime_unsupported_reasons(artifact):
    metadata = artifact.codegen_mod.get_function_metadata("main")
    return [
        str(reason)
        for reason in metadata.get("direct_runtime_unsupported_reasons", [])
    ]


def multicore_gemm_kernel(
    M: int = 64, N: int = 64, K: int = 128, tile_m: int = 32, tile_n: int = 32
):
    """GEMM kernel that maps bx/by to output tile coordinates."""
    grid_x = N // tile_n
    grid_y = M // tile_m

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            A_shared = T.alloc_shared((tile_m, K), "bfloat16")
            B_shared = T.alloc_shared((tile_n, K), "bfloat16")
            C_local = T.alloc_fragment((tile_m, tile_n), "float32")
            T.copy(A[by * tile_m : (by + 1) * tile_m, 0:K], A_shared)
            T.copy(B[bx * tile_n : (bx + 1) * tile_n, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(
                C_local,
                C[by * tile_m : (by + 1) * tile_m, bx * tile_n : (bx + 1) * tile_n],
            )

    return main


def test_blackhole_gemm_pipeline_uses_spatial_plan_without_spatial_program():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_func = artifact.device_mod["main_kernel"]
    plan = device_func.attrs["tl.spatial_plan"]
    tt_program = require_tt_program(device_func)
    compute_contract = dict(tt_program.payload["compute_contract"])
    assert device_func.attrs.get("tl.spatial_program") is None
    assert device_func.attrs.get("blackhole.lowering_requirements") is None
    assert {"ingress", "compute", "egress"}.issubset(
        {str(unit.unit_role) for unit in plan.execution_units}
    )
    assert str(compute_contract["kind"]) == "gemm"


def test_blackhole_gemm_arg_identity_drives_cross_segment_dedupe():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    def _with_identity(arg):
        arg = dict(arg)
        if "identity" not in arg:
            if "buffer" in arg:
                arg["identity"] = f"{arg['kind']}:{arg['buffer']}"
            else:
                arg["identity"] = str(arg["kind"])
        return arg

    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        runtime_args = [_with_identity(arg) for arg in segment["runtime_args"]]
        if str(segment["kind"]) in {"reader", "writer"} and not any(
            str(arg["kind"]) == "work_linear_id" for arg in runtime_args
        ):
            runtime_args.append(
                {
                    "name": "work_linear_id",
                    "kind": "work_linear_id",
                    "identity": f"{segment['kind']}_work_linear_id",
                    "dtype": "uint32",
                }
            )
        for arg in runtime_args:
            if str(arg["kind"]) == "work_linear_id":
                arg["identity"] = f"{segment['kind']}_work_linear_id"
        mutated_segment["runtime_args"] = runtime_args
        mutated_segment["common_runtime_args"] = [
            {
                "name": "shared_rank",
                "kind": "accessor_common_u32",
                "identity": f"{segment['kind']}_shared_rank",
                "dtype": "uint32",
            }
        ]
        mutated_segments.append(mutated_segment)

    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)
    executable_spec = mutated_mod.get_function_metadata("main")

    work_linear_ids = [
        str(item["identity"])
        for item in executable_spec["runtime_args"]
        if str(item["kind"]) == "work_linear_id"
    ]
    assert work_linear_ids == ["reader_work_linear_id", "writer_work_linear_id"]

    common_runtime_arg_ids = [
        str(item["identity"]) for item in executable_spec["common_runtime_args"]
    ]
    assert common_runtime_arg_ids == [
        "reader_shared_rank",
        "compute_shared_rank",
        "writer_shared_rank",
    ]


def test_blackhole_gemm_accumulator_scope_canonicalized():
    kernel = gemm_kernel()
    target = Target("blackhole")
    mod = tilelang.tvm.IRModule({"main": kernel})

    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
        mod = tilelang.engine.phase.OptimizeForTarget(mod, target)

    func = mod["main"]
    assert func.attrs.get("blackhole.resource_plan") is None
    func_text = func.script()
    assert 'scope="blackhole.acc"' in func_text


def test_blackhole_gemm_contract_attr_is_materialized():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    tt_program = require_tt_program(func)
    contract = tt_program.payload["gemm_contract"]
    assert str(contract["a_buffer"]) == "A"
    assert str(contract["b_buffer"]) == "B"
    assert str(contract["c_buffer"]) == "C"
    assert int(contract["M"]) == 32
    assert int(contract["N"]) == 32
    assert int(contract["K"]) == 128
    assert bool(contract["transpose_B"]) is True
    assert str(contract["a_tensor_dtype"]) == "Float16_b"
    assert str(contract["b_tensor_dtype"]) == "Float16_b"
    assert str(contract["c_tensor_dtype"]) == "Float32"
    assert str(contract["a_cb_dtype"]) == "Float16_b"
    assert str(contract["b_cb_dtype"]) == "Float16_b"
    assert str(contract["c_cb_dtype"]) == "Float32"
    assert str(contract["accumulator_dtype"]) == "Float32"

    reader_abi = tt_abi_for_kernel(tt_program, require_tt_kernel(tt_program, kind="reader", core_type="brisc"))
    writer_abi = tt_abi_for_kernel(tt_program, require_tt_kernel(tt_program, kind="writer", core_type="ncrisc"))
    assert [(str(item["buffer"]), int(item["compile_time_arg_offset"])) for item in reader_abi.accessors] == [
        ("A", 0),
        ("B", 2),
    ]
    assert [int(item["compile_time_arg_count"]) for item in reader_abi.accessors] == [2, 2]
    assert [int(item["common_runtime_arg_offset"]) for item in reader_abi.accessors] == [0, 0]
    assert [int(item["common_runtime_arg_count"]) for item in reader_abi.accessors] == [0, 0]
    assert [int(item["args_config_bits"]) for item in reader_abi.accessors] == [2, 2]
    assert [(str(item["buffer"]), int(item["compile_time_arg_offset"])) for item in writer_abi.accessors] == [
        ("C", 0)
    ]
    assert [int(item["compile_time_arg_count"]) for item in writer_abi.accessors] == [2]
    assert [int(item["common_runtime_arg_offset"]) for item in writer_abi.accessors] == [0]
    assert [int(item["common_runtime_arg_count"]) for item in writer_abi.accessors] == [0]
    assert [int(item["args_config_bits"]) for item in writer_abi.accessors] == [2]
    assert all(str(item["layout"]) == "interleaved" for item in reader_abi.accessors)
    assert all(str(item["memory_space"]) == "dram" for item in reader_abi.accessors)
    assert len(reader_abi.common_runtime_args) == 0
    assert len(writer_abi.common_runtime_args) == 0


def test_blackhole_gemm_segment_plan_is_not_backed_by_segment_markers():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    assert not contains_attr_stmt_key(func.body, "blackhole.segment_kind")
    segment_plan = extract_blackhole_segment_plan(func)
    assert [str(segment["kind"]) for segment in segment_plan] == [
        "reader",
        "compute",
        "writer",
    ]


def test_blackhole_compute_contract_attr_is_materialized():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    contract = require_tt_program(mod["main"]).payload["compute_contract"]
    assert str(contract["kind"]) == "gemm"
    assert bool(contract["enabled"]) is True
    assert str(contract["a_buffer"]) == "A"
    assert str(contract["b_buffer"]) == "B"
    assert str(contract["c_buffer"]) == "C"
    assert int(contract["M"]) == 32
    assert int(contract["N"]) == 32
    assert int(contract["K"]) == 128
    assert int(contract["Mt"]) == 1
    assert int(contract["Nt"]) == 1
    assert int(contract["Kt"]) == 4
    assert bool(contract["transpose_A"]) is False
    assert bool(contract["transpose_B"]) is True
    assert str(contract["a_tensor_dtype"]) == "Float16_b"
    assert str(contract["b_tensor_dtype"]) == "Float16_b"
    assert str(contract["c_tensor_dtype"]) == "Float32"
    assert str(contract["a_cb_dtype"]) == "Float16_b"
    assert str(contract["b_cb_dtype"]) == "Float16_b"
    assert str(contract["c_cb_dtype"]) == "Float32"
    assert str(contract["accumulator_dtype"]) == "Float32"
    assert int(contract["block_m_tiles"]) == 1
    assert int(contract["block_n_tiles"]) == 1
    assert int(contract["block_k_tiles"]) == 4
    assert int(contract["subblock_m_tiles"]) == 1
    assert int(contract["subblock_n_tiles"]) == 1
    assert str(contract["math_fidelity"]) == "HiFi4"
    assert bool(contract["fp32_dest_acc_en"]) is True
    assert bool(contract["math_approx_mode"]) is False
    assert [str(item) for item in contract["unpack_to_dest_mode"]] == []
    assert bool(contract["clear_accum"]) is True
    assert int(contract["k_pack"]) == 1
    assert int(contract["wg_wait"]) == 0
    assert int(contract["policy_type"]) == 0
    assert str(contract["policy_name"]) == "Square"


def test_blackhole_fresh_fragment_gemm_does_not_materialize_accumulator_merge_contract():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    payload = dict(require_tt_program(mod["main"]).payload)
    assert "buffer_materialization_contracts" not in payload


def test_blackhole_precleared_fragment_gemm_does_not_materialize_accumulator_merge_contract():
    kernel = gemm_kernel_with_compute_abi(
        clear_accum=False, k_pack=1, wg_wait=0, preclear_output_fragment=True
    )
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    payload = dict(require_tt_program(mod["main"]).payload)
    assert "buffer_materialization_contracts" not in payload


def test_blackhole_compute_contract_attr_materializes_nondefault_compute_abi():
    kernel = gemm_kernel_with_compute_abi()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    contract = extract_blackhole_compute_contract(func)
    assert bool(contract["clear_accum"]) is True
    assert int(contract["k_pack"]) == 2
    assert int(contract["wg_wait"]) == 3


def test_blackhole_compute_contract_attr_materializes_richer_compute_config_extras():
    kernel = gemm_kernel_with_compute_config_extras()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    contract = extract_blackhole_compute_contract(func)
    assert bool(contract["dst_full_sync_en"]) is True
    assert bool(contract["bfp8_pack_precise"]) is True
    assert [(str(item["name"]), str(item["value"])) for item in contract["defines"]] == [
        ("BLACKHOLE_ACC_MODE", "fp32"),
        ("BLACKHOLE_TEST_DEFINE", "1"),
    ] or [(str(item["name"]), str(item["value"])) for item in contract["defines"]] == [
        ("BLACKHOLE_TEST_DEFINE", "1"),
        ("BLACKHOLE_ACC_MODE", "fp32"),
    ]
    assert sorted(
        (str(item["name"]), int(item["value"])) for item in contract["named_compile_args"]
    ) == [("c_0", 0), ("c_1", 1), ("c_16", 16)]


def test_blackhole_compute_segment_compute_config_follows_compute_contract():
    kernel = gemm_kernel_with_compute_abi()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func = mod["main"]
    contract = extract_blackhole_compute_contract(func)
    plan = extract_blackhole_segment_plan(func)
    compute = next(segment for segment in plan if str(segment["kind"]) == "compute")
    compute_config = compute["compute_config"]

    assert str(compute_config["math_fidelity"]) == str(contract["math_fidelity"])
    assert bool(compute_config["fp32_dest_acc_en"]) is bool(contract["fp32_dest_acc_en"])
    assert bool(compute_config["math_approx_mode"]) is bool(contract["math_approx_mode"])
    assert [str(item) for item in compute_config["unpack_to_dest_mode"]] == [
        str(item) for item in contract["unpack_to_dest_mode"]
    ]
    assert bool(compute_config["clear_accum"]) is bool(contract["clear_accum"])
    assert int(compute_config["k_pack"]) == int(contract["k_pack"])
    assert int(compute_config["wg_wait"]) == int(contract["wg_wait"])
    assert int(compute_config["policy_type"]) == int(contract["policy_type"])
    assert str(compute_config["policy_name"]) == str(contract["policy_name"])


def test_blackhole_compute_contract_attr_materializes_nondefault_policy():
    kernel = gemm_kernel_with_policy()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    contract = extract_blackhole_compute_contract(mod["main"])
    assert int(contract["policy_type"]) == 1
    assert str(contract["policy_name"]) == "FullRow"


def test_blackhole_compute_contract_attr_materializes_mbar_binding():
    kernel = gemm_kernel_with_mbar()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    contract = extract_blackhole_compute_contract(mod["main"])
    assert bool(contract["has_mbarrier"]) is True
    assert str(contract["mbarrier_buffer"]) == "mbar"
    assert str(contract["mbarrier_scope"]) == "shared.barrier"
    assert [str(item) for item in contract["mbarrier_index_exprs"]] == ["0"]


def test_blackhole_gemm_compile_time_abi_is_materialized():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    kernels = executable_spec["kernels"]
    reader = _require_blackhole_kernel(kernels, kind="reader", core_type="brisc")
    compute = _require_blackhole_kernel(kernels, kind="compute", core_type="trisc")
    writer = _require_blackhole_kernel(kernels, kind="writer", core_type="ncrisc")

    assert "compile_time_arg_specs" in reader
    reader_compile_time_arg_specs = reader["compile_time_arg_specs"]
    reader_a = _require_spec_entry(
        reader_compile_time_arg_specs,
        kind="interleaved_accessor_cta",
        label="reader compile-time",
        buffer="A",
    )
    reader_b = _require_spec_entry(
        reader_compile_time_arg_specs,
        kind="interleaved_accessor_cta",
        label="reader compile-time",
        buffer="B",
    )
    assert str(reader_a["name"]) == "A"
    assert str(reader_a["dtype"]) == "uint32"
    assert int(reader_a["offset"]) == 0
    assert int(reader_a["count"]) == 2
    assert str(reader_a["buffer"]) == "A"
    assert str(reader_a["segment_role"]) == "reader"
    assert int(reader_a["args_config_bits"]) == 2
    assert str(reader_a["layout"]) == "interleaved"
    assert str(reader_a["memory_space"]) == "dram"
    assert str(reader_b["name"]) == "B"
    assert str(reader_b["dtype"]) == "uint32"
    assert int(reader_b["offset"]) == 2
    assert int(reader_b["count"]) == 2
    assert str(reader_b["buffer"]) == "B"
    assert str(reader_b["segment_role"]) == "reader"
    assert int(reader_b["args_config_bits"]) == 2
    assert str(reader_b["layout"]) == "interleaved"
    assert str(reader_b["memory_space"]) == "dram"

    assert "launch_spec" in reader
    reader_launch_spec = reader["launch_spec"]
    expected_reader_launch_spec = _expected_launch_spec_for_core_type(reader["core_type"])
    assert str(reader_launch_spec["core_type"]) == expected_reader_launch_spec["core_type"]
    assert str(reader_launch_spec["processor"]) == expected_reader_launch_spec["processor"]
    assert str(reader_launch_spec["noc"]) == expected_reader_launch_spec["noc"]

    assert "compile_time_arg_specs" in compute
    compute_compile_time_arg_specs = compute["compile_time_arg_specs"]
    gemm_shape = _require_spec_entry(
        compute_compile_time_arg_specs, kind="gemm_shape", label="compute compile-time"
    )
    gemm_transpose_flags = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_transpose_flags",
        label="compute compile-time",
    )
    gemm_block_shape = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_block_shape",
        label="compute compile-time",
    )
    gemm_subblock_shape = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_subblock_shape",
        label="compute compile-time",
    )
    gemm_clear_accum = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_clear_accum",
        label="compute compile-time",
    )
    gemm_k_pack = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_k_pack",
        label="compute compile-time",
    )
    gemm_wg_wait = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_wg_wait",
        label="compute compile-time",
    )
    gemm_policy = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_policy",
        label="compute compile-time",
    )
    assert str(gemm_shape["name"]) == "gemm_shape"
    assert str(gemm_shape["dtype"]) == "uint32"
    assert int(gemm_shape["offset"]) == 0
    assert int(gemm_shape["count"]) == 3
    assert str(gemm_shape["segment_role"]) == "compute"
    assert [int(value) for value in gemm_shape["values"]] == [1, 4, 1]
    assert str(gemm_transpose_flags["name"]) == "gemm_transpose_flags"
    assert str(gemm_transpose_flags["dtype"]) == "uint32"
    assert int(gemm_transpose_flags["offset"]) == 3
    assert int(gemm_transpose_flags["count"]) == 2
    assert str(gemm_transpose_flags["segment_role"]) == "compute"
    assert [int(value) for value in gemm_transpose_flags["values"]] == [0, 1]
    assert [int(value) for value in gemm_block_shape["values"]] == [1, 1, 4]
    assert [int(value) for value in gemm_subblock_shape["values"]] == [1, 1]
    assert [int(value) for value in gemm_clear_accum["values"]] == [1]
    assert [int(value) for value in gemm_k_pack["values"]] == [1]
    assert [int(value) for value in gemm_wg_wait["values"]] == [0]
    assert [int(value) for value in gemm_policy["values"]] == [0]

    assert "launch_spec" in compute
    compute_launch_spec = compute["launch_spec"]
    expected_compute_launch_spec = _expected_launch_spec_for_core_type(compute["core_type"])
    assert str(compute_launch_spec["core_type"]) == expected_compute_launch_spec["core_type"]
    assert str(compute_launch_spec["processor"]) == expected_compute_launch_spec["processor"]
    assert str(compute_launch_spec["noc"]) == expected_compute_launch_spec["noc"]
    assert "compute_config" in compute
    compute_config = compute["compute_config"]
    assert str(compute_config["math_fidelity"]) == "HiFi4"
    assert bool(compute_config["fp32_dest_acc_en"]) is True
    assert bool(compute_config["math_approx_mode"]) is False
    assert [str(item) for item in compute_config["unpack_to_dest_mode"]] == []

    assert "compile_time_arg_specs" in writer
    writer_compile_time_arg_specs = writer["compile_time_arg_specs"]
    writer_c = _require_spec_entry(
        writer_compile_time_arg_specs,
        kind="interleaved_accessor_cta",
        label="writer compile-time",
        buffer="C",
    )
    assert str(writer_c["name"]) == "C"
    assert str(writer_c["dtype"]) == "uint32"
    assert int(writer_c["offset"]) == 0
    assert int(writer_c["count"]) == 2
    assert str(writer_c["buffer"]) == "C"
    assert str(writer_c["segment_role"]) == "writer"
    assert int(writer_c["args_config_bits"]) == 2
    assert str(writer_c["layout"]) == "interleaved"
    assert str(writer_c["memory_space"]) == "dram"

    assert "launch_spec" in writer
    writer_launch_spec = writer["launch_spec"]
    expected_writer_launch_spec = _expected_launch_spec_for_core_type(writer["core_type"])
    assert str(writer_launch_spec["core_type"]) == expected_writer_launch_spec["core_type"]
    assert str(writer_launch_spec["processor"]) == expected_writer_launch_spec["processor"]
    assert str(writer_launch_spec["noc"]) == expected_writer_launch_spec["noc"]

    _assert_no_contract_family(executable_spec)
    gemm_op = next(item for item in compute["compute_ops"] if str(item["kind"]) == "gemm")
    compute_config = compute["compute_config"]
    assert str(gemm_op["kind"]) == "gemm"
    assert bool(gemm_op["enabled"]) is True
    assert int(gemm_op["M"]) == 32
    assert int(gemm_op["N"]) == 32
    assert int(gemm_op["K"]) == 128
    assert int(gemm_op["Mt"]) == 1
    assert int(gemm_op["Nt"]) == 1
    assert int(gemm_op["Kt"]) == 4
    assert bool(gemm_op["transpose_A"]) is False
    assert bool(gemm_op["transpose_B"]) is True
    assert int(gemm_op["block_m_tiles"]) == 1
    assert int(gemm_op["block_n_tiles"]) == 1
    assert int(gemm_op["block_k_tiles"]) == 4
    assert int(gemm_op["subblock_m_tiles"]) == 1
    assert int(gemm_op["subblock_n_tiles"]) == 1
    assert str(compute_config["math_fidelity"]) == "HiFi4"
    assert bool(compute_config["fp32_dest_acc_en"]) is True
    assert bool(compute_config["math_approx_mode"]) is False
    assert [str(item) for item in compute_config["unpack_to_dest_mode"]] == []


def test_blackhole_gemm_buffer_materialization_specs_are_exposed():
    kernel = gemm_kernel()
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
        "C": ("replicated", "interleaved", "dram", 4096),
    }


def test_blackhole_fragment_fill_cast_publish_exposes_typed_live_form_owner_truth():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    mod = tilelang.tvm.IRModule({"main": kernel})
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = lower_blackhole_to_tt_target(mod)

    tt_program = require_tt_program(mod["main"])
    live_forms = {str(plan.logical_value): plan for plan in tt_program.live_form_plans}
    assert {"C_local", "D_local"}.issubset(live_forms)
    assert str(live_forms["C_local"].physical_form) == "thread_distributed_slice"
    assert str(live_forms["D_local"].physical_form) == "cb_materialized_tile"
    assert str(live_forms["D_local"].execution_topology) == "thread_distributed"
    assert int(live_forms["D_local"].logical_element_count) == 1024
    assert int(live_forms["D_local"].physical_local_extent) == 8

    materializations = {
        str(plan.target_buffer): plan for plan in tt_program.materialization_plans
    }
    assert "D_local" in materializations
    d_local = materializations["D_local"]
    assert str(d_local.materialization_protocol) == "cb_republish"
    assert str(d_local.produced_live_form) == str(live_forms["D_local"].name)
    assert [int(index) for index in d_local.required_cb_plan_indices]

    bindings = {
        str(plan.consumer_op_kind): plan for plan in tt_program.consumer_binding_plans
    }
    assert "cast_fragment_slice" in bindings
    cast_binding = bindings["cast_fragment_slice"]
    assert str(cast_binding.source_live_form) == str(live_forms["C_local"].name)
    assert bool(cast_binding.accepts_distributed_slice) is True
    assert bool(cast_binding.requires_full_logical_tile) is False


def test_blackhole_fragment_fill_cast_publish_tt_plans_reference_spatial_live_boundaries():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    mod = tilelang.tvm.IRModule({"main": kernel})
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = lower_blackhole_to_tt_target(mod)

    main = mod["main"]
    spatial_plan = main.attrs["tl.spatial_plan"]
    tt_program = require_tt_program(main)

    live_value_by_subject = {
        str(live_value.subject): (index, live_value)
        for index, live_value in enumerate(spatial_plan.live_values)
    }
    boundary_by_source_live_value = {
        str(boundary.source_live_value): (index, boundary)
        for index, boundary in enumerate(spatial_plan.materialization_boundaries)
    }
    live_edge_by_name = {
        str(edge.name): (index, edge)
        for index, edge in enumerate(spatial_plan.live_value_edges)
    }

    c_local_index, c_local_live_value = live_value_by_subject["C_local"]
    c_local_boundary_index, c_local_boundary = boundary_by_source_live_value[
        str(c_local_live_value.name)
    ]
    c_local_live_edge_index, c_local_live_edge = live_edge_by_name[
        str(c_local_boundary.live_value_edge)
    ]

    live_forms = {str(plan.logical_value): plan for plan in tt_program.live_form_plans}
    c_local_live_form = live_forms["C_local"]
    assert str(c_local_live_form.spatial_live_value) == str(c_local_live_value.name)
    assert int(c_local_live_form.spatial_live_value_index) == c_local_index

    materializations = {
        str(plan.target_buffer): plan for plan in tt_program.materialization_plans
    }
    d_local_materialization = materializations["D_local"]
    assert str(d_local_materialization.materialization_boundary) == str(c_local_boundary.name)
    assert int(d_local_materialization.materialization_boundary_index) == c_local_boundary_index

    bindings = {
        str(plan.consumer_op_kind): plan for plan in tt_program.consumer_binding_plans
    }
    cast_binding = bindings["cast_fragment_slice"]
    assert str(cast_binding.live_value_edge) == str(c_local_live_edge.name)
    assert int(cast_binding.live_value_edge_index) == c_local_live_edge_index


def test_blackhole_fragment_fill_cast_publish_projects_leaf_materialization_plans():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    live_forms = {
        str(plan["logical_value"]): plan for plan in executable_spec["live_form_plans"]
    }
    assert str(live_forms["C_local"]["spatial_live_value"]).startswith("live_")
    assert int(live_forms["C_local"]["spatial_live_value_index"]) >= 0

    materializations = {
        str(plan["target_buffer"]): plan
        for plan in executable_spec["materialization_plans"]
    }
    assert "D_local" in materializations
    d_local = materializations["D_local"]
    assert str(d_local["materialization_protocol"]) == "cb_republish"
    assert str(d_local["publication_protocol"]) == "pack_thread_direct_store"
    assert str(d_local["source_live_form"]) == "live_form_C_local"
    assert str(d_local["produced_live_form"]) == "live_form_D_local"
    assert str(d_local["materialization_boundary"]).startswith("materialization_")
    assert int(d_local["materialization_boundary_index"]) >= 0
    consumer_bindings = {
        str(plan["consumer_op_kind"]): plan
        for plan in executable_spec["consumer_binding_plans"]
    }
    assert str(consumer_bindings["cast_fragment_slice"]["live_value_edge"]).startswith(
        "live_edge_"
    )
    assert int(consumer_bindings["cast_fragment_slice"]["live_value_edge_index"]) >= 0

    output_materialization = next(
        item for item in executable_spec["buffer_materializations"] if str(item["buffer"]) == "D"
    )
    assert str(output_materialization["live_form_kind"]) == "cb_materialized_tile"
    assert str(output_materialization["execution_topology_kind"]) == "thread_distributed"
    assert int(output_materialization["physical_local_extent"]) == 8
    assert int(output_materialization["logical_element_count"]) == 1024
    assert str(output_materialization["producer_kernel"]) == "compute"
    assert str(output_materialization["materialization_protocol"]) == "cb_republish"
    assert str(output_materialization["publication_protocol"]) == "pack_thread_direct_store"


def test_blackhole_fragment_fill_cast_publish_admits_non_mailbox_cb_republish():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    compute_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    assert "tilelang_get_cb_write_ptr_bytes(17)" not in compute_source
    assert "tilelang_pack_fill_bfloat16_tiled_cb(17" in compute_source


def test_blackhole_gemm_kernel_compute_config_follows_compute_contract_in_spec():
    kernel = gemm_kernel_with_compute_abi()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in executable_spec
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_config = compute["compute_config"]

    assert str(compute_config["math_fidelity"]) == "HiFi4"
    assert bool(compute_config["fp32_dest_acc_en"]) is True
    assert bool(compute_config["math_approx_mode"]) is False
    assert [str(item) for item in compute_config["unpack_to_dest_mode"]] == []
    assert bool(compute_config["clear_accum"]) is True
    assert int(compute_config["k_pack"]) == 2
    assert int(compute_config["wg_wait"]) == 3
    assert int(compute_config["policy_type"]) == 0
    assert str(compute_config["policy_name"]) == "Square"


def test_blackhole_gemm_kernel_projects_typed_compute_ops_schema():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    assert "gemm_compute" not in compute
    assert "compute_ops" in compute
    compute_ops = [dict(item) for item in compute["compute_ops"]]
    gemm_op = next(item for item in compute_ops if str(item["kind"]) == "gemm")

    assert bool(gemm_op["enabled"]) is True
    assert int(gemm_op["M"]) == 32
    assert int(gemm_op["N"]) == 32
    assert int(gemm_op["K"]) == 128
    assert int(gemm_op["Mt"]) == 1
    assert int(gemm_op["Nt"]) == 1
    assert int(gemm_op["Kt"]) == 4
    assert str(gemm_op["a_buffer"]) == "A"
    assert str(gemm_op["b_buffer"]) == "B"
    assert str(gemm_op["c_buffer"]) == "C"
    assert bool(gemm_op["transpose_A"]) is False
    assert bool(gemm_op["transpose_B"]) is True
    assert str(gemm_op["a_tensor_dtype"]) == "Float16_b"
    assert str(gemm_op["b_tensor_dtype"]) == "Float16_b"
    assert str(gemm_op["c_tensor_dtype"]) == "Float32"


def test_blackhole_gemm_compute_ops_carry_typed_operand_bindings():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    gemm_op = _require_gemm_compute_op(executable_spec)

    operand_bindings = {
        str(binding["role"]): dict(binding)
        for binding in gemm_op["operand_bindings"]
    }
    assert {role: str(binding["buffer"]) for role, binding in operand_bindings.items()} == {
        "a": "A_shared",
        "b": "B_shared",
        "c": "C_local",
    }
    assert {role: str(binding["host_buffer"]) for role, binding in operand_bindings.items()} == {
        "a": "A",
        "b": "B",
        "c": "C",
    }
    assert str(gemm_op["a_buffer"]) == str(operand_bindings["a"]["host_buffer"])
    assert str(gemm_op["b_buffer"]) == str(operand_bindings["b"]["host_buffer"])
    assert str(gemm_op["c_buffer"]) == str(operand_bindings["c"]["host_buffer"])


def test_blackhole_gemm_spec_survives_without_legacy_contract_attrs():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    stripped_mod = _rebuild_codegen_module_without_legacy_contract_attrs(artifact)
    executable_spec = stripped_mod.get_function_metadata("main")

    for legacy_key in (
        "gemm_contract",
        "compute_contract",
        "multi_gemm_contracts",
        "multi_compute_contracts",
        "compute_epilogue_ops",
    ):
        assert legacy_key not in executable_spec
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    gemm_op = next(item for item in compute["compute_ops"] if str(item["kind"]) == "gemm")
    assert bool(gemm_op["enabled"]) is True
    assert int(gemm_op["M"]) == 32
    assert int(gemm_op["N"]) == 32
    assert int(gemm_op["K"]) == 128


def test_blackhole_gemm_spec_survives_without_contract_family_payload():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    stripped_mod = _rebuild_codegen_module_without_contract_family_payload(artifact)
    executable_spec = stripped_mod.get_function_metadata("main")
    _assert_no_contract_family(executable_spec)

    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    assert "compute_ops" in compute
    gemm_op = next(item for item in compute["compute_ops"] if str(item["kind"]) == "gemm")
    assert bool(gemm_op["enabled"]) is True
    assert int(gemm_op["K"]) == 128


def test_blackhole_gemm_direct_runtime_uses_typed_compute_ops_without_contract_family():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    stripped_mod = _rebuild_codegen_module_without_contract_family_payload(artifact)

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    stripped_mod["main"](a_torch, b_torch, c_output)
    expected = torch.matmul(a_torch.to(torch.float32), b_torch.to(torch.float32).T)
    assert_tensors_close_or_dump(
        c_output,
        expected,
        atol=1e-2,
        rtol=1e-2,
        failure_message="gemm_typed_compute_without_contract_family mismatch",
    )


def test_blackhole_gemm_compile_time_abi_materializes_nondefault_compute_abi():
    kernel = gemm_kernel_with_compute_abi()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)
    assert bool(compute_config["clear_accum"]) is True
    assert int(compute_config["k_pack"]) == 2
    assert int(compute_config["wg_wait"]) == 3

    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_compile_time_arg_specs = compute["compile_time_arg_specs"]
    gemm_clear_accum = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_clear_accum",
        label="compute compile-time",
    )
    gemm_k_pack = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_k_pack",
        label="compute compile-time",
    )
    gemm_wg_wait = _require_spec_entry(
        compute_compile_time_arg_specs,
        kind="gemm_wg_wait",
        label="compute compile-time",
    )
    assert [int(value) for value in gemm_clear_accum["values"]] == [1]
    assert [int(value) for value in gemm_k_pack["values"]] == [2]
    assert [int(value) for value in gemm_wg_wait["values"]] == [3]


def test_blackhole_gemm_compute_config_materializes_extended_precision_flags():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    mutated_config = _require_compute_config(executable_spec)
    mutated_config["dst_full_sync_en"] = True
    mutated_config["bfp8_pack_precise"] = True
    mutated_mod = _rebuild_codegen_module_with_compute_overrides(artifact, mutated_config)

    executable_spec = mutated_mod.get_function_metadata("main")
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)

    assert bool(compute_config["dst_full_sync_en"]) is True
    assert bool(compute_config["bfp8_pack_precise"]) is True


def test_blackhole_gemm_compute_config_materializes_defines_and_named_compile_args():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    mutated_config = _require_compute_config(executable_spec)
    mutated_config["defines"] = [
        {"name": "BLACKHOLE_TEST_DEFINE", "value": "1"},
        {"name": "BLACKHOLE_ACC_MODE", "value": "fp32"},
    ]
    mutated_config["named_compile_args"] = [
        {"name": "c_0", "value": 0},
        {"name": "c_1", "value": 1},
        {"name": "c_16", "value": 16},
    ]
    mutated_mod = _rebuild_codegen_module_with_compute_overrides(artifact, mutated_config)

    executable_spec = mutated_mod.get_function_metadata("main")
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)

    assert [(str(item["name"]), str(item["value"])) for item in compute_config["defines"]] == [
        ("BLACKHOLE_TEST_DEFINE", "1"),
        ("BLACKHOLE_ACC_MODE", "fp32"),
    ]
    assert [
        (str(item["name"]), int(item["value"])) for item in compute_config["named_compile_args"]
    ] == [("c_0", 0), ("c_1", 1), ("c_16", 16)]


def test_blackhole_gemm_spec_materializes_dsl_produced_compute_config_extras():
    kernel = gemm_kernel_with_compute_config_extras()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)

    assert bool(compute_config["dst_full_sync_en"]) is True
    assert bool(compute_config["bfp8_pack_precise"]) is True
    assert sorted((str(item["name"]), str(item["value"])) for item in compute_config["defines"]) == [
        ("BLACKHOLE_ACC_MODE", "fp32"),
        ("BLACKHOLE_TEST_DEFINE", "1"),
    ]
    assert sorted(
        (str(item["name"]), int(item["value"])) for item in compute_config["named_compile_args"]
    ) == [("c_0", 0), ("c_1", 1), ("c_16", 16)]


def test_blackhole_gemm_compile_time_abi_materializes_nondefault_policy():
    kernel = gemm_kernel_with_policy()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)
    assert int(compute_config["policy_type"]) == 1
    assert str(compute_config["policy_name"]) == "FullRow"

    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    gemm_policy = _require_spec_entry(
        compute["compile_time_arg_specs"], kind="gemm_policy", label="compute compile-time"
    )
    assert [int(value) for value in gemm_policy["values"]] == [1]


def test_blackhole_gemm_compile_time_abi_materializes_mbar_binding():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    mutated_mod = _rebuild_codegen_module_with_compute_overrides(
        artifact,
        {
            "has_mbarrier": True,
            "mbarrier_buffer": "mbar",
            "mbarrier_scope": "shared.barrier",
            "mbarrier_index_exprs": ["0"],
        },
    )

    executable_spec = mutated_mod.get_function_metadata("main")
    _assert_no_contract_family(executable_spec)
    gemm_op = _require_gemm_compute_op(executable_spec)
    assert bool(gemm_op["has_mbarrier"]) is True
    assert str(gemm_op["mbarrier_buffer"]) == "mbar"
    assert str(gemm_op["mbarrier_scope"]) == "shared.barrier"
    assert [str(item) for item in gemm_op["mbarrier_index_exprs"]] == ["0"]

    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    assert all(str(spec["kind"]) != "gemm_mbarrier" for spec in compute["compile_time_arg_specs"])


def test_blackhole_gemm_compile_time_abi_rejects_misaligned_shapes():
    kernel = gemm_kernel(M=48, N=32, K=128)
    target = Target("blackhole")

    with pytest.raises(Exception, match="aligned to 32"):
        with target:
            lower(kernel, target=target)


def test_blackhole_gemm_direct_runtime_rejects_sharded_accessor_schema():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    def mutate_sharded_compile_time_spec(spec, *, segment):
        if str(spec["kind"]) == "interleaved_accessor_cta":
            spec["layout"] = "sharded"
            spec["memory_space"] = "dram"
        return spec

    richer_func = _with_compile_time_abi_schema(
        device_main,
        strip_accessors=True,
        compile_time_arg_spec_mutator=mutate_sharded_compile_time_spec,
    )
    mutated_mod = _rebuild_codegen_module_with_segment_plan(
        artifact, extract_blackhole_segment_plan(richer_func)
    )

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="interleaved"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_direct_runtime_rejects_accessor_common_runtime_arg_count():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        if "accessors" in segment:
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

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|interleaved"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_direct_runtime_rejects_accessor_runtime_crta_bits():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
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

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="common runtime args|args_config_bits == 2"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_direct_runtime_rejects_mismatched_launch_spec_core_type():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    mutated_segments = []
    for segment in extract_blackhole_segment_plan(device_main):
        mutated_segment = dict(segment)
        if str(mutated_segment["kind"]) == "reader":
            launch_spec = dict(mutated_segment["launch_spec"])
            launch_spec["core_type"] = "trisc"
            mutated_segment["launch_spec"] = launch_spec
        mutated_segments.append(mutated_segment)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(artifact, mutated_segments)

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="launch_spec.core_type mismatch"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_direct_runtime_rejects_unknown_math_fidelity():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    mutated_config = _require_compute_config(executable_spec)
    mutated_config["math_fidelity"] = "UltraFi9"
    mutated_mod = _rebuild_codegen_module_with_compute_overrides(artifact, mutated_config)

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="math_fidelity"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_direct_runtime_rejects_mbarrier_compute_contract():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    mutated_mod = _rebuild_codegen_module_with_compute_overrides(
        artifact,
        {
            "has_mbarrier": True,
            "mbarrier_buffer": "mbar",
            "mbarrier_scope": "shared.barrier",
            "mbarrier_index_exprs": ["0"],
        },
    )

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="mbarrier"):
        mutated_mod["main"](a_torch, b_torch, c_output)


def test_blackhole_gemm_reader_binds_tensor_accessor_to_buffer_addrs():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    reader_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="reader", core_type="brisc"
    )
    source = str(reader_kernel["source_code"])

    assert "TensorAccessor(src_accessor_args, A_addr, tile_bytes)" in source
    assert "TensorAccessor(src_accessor_args, B_addr, tile_bytes)" in source
    assert "TensorAccessor(src_accessor_args, a_tile_stride, tile_bytes)" not in source
    assert "TensorAccessor(src_accessor_args, b_tile_stride, tile_bytes)" not in source


def test_blackhole_gemm_direct_runtime_materializes_compile_time_abi_schema():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    stripped_func = _with_compile_time_abi_schema(device_main, strip_accessors=True)
    mutated_mod = _rebuild_codegen_module_with_segment_plan(
        artifact, extract_blackhole_segment_plan(stripped_func)
    )

    mutated_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output, c_ref, atol=2e-1, rtol=2e-1, failure_message="GEMM direct-call output mismatch"
    )


def test_blackhole_gemm_direct_runtime_preserves_richer_compute_config_correctness():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = gemm_kernel_with_compute_config_extras()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)
    assert bool(compute_config["dst_full_sync_en"]) is True
    assert bool(compute_config["bfp8_pack_precise"]) is True
    assert sorted((str(item["name"]), str(item["value"])) for item in compute_config["defines"]) == [
        ("BLACKHOLE_ACC_MODE", "fp32"),
        ("BLACKHOLE_TEST_DEFINE", "1"),
    ]
    assert sorted(
        (str(item["name"]), int(item["value"])) for item in compute_config["named_compile_args"]
    ) == [("c_0", 0), ("c_1", 1), ("c_16", 16)]

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="GEMM richer compute-config direct-call output mismatch",
    )


def test_blackhole_precleared_fragment_gemm_canonicalizes_to_clear_accum_true():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = gemm_kernel_with_compute_abi(
        clear_accum=False, k_pack=1, wg_wait=0, preclear_output_fragment=True
    )
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)
    assert bool(compute_config["clear_accum"]) is True
    assert int(compute_config["k_pack"]) == 1
    assert int(compute_config["wg_wait"]) == 0

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Precleared GEMM canonicalized clear-accum direct-call output mismatch",
    )


def test_blackhole_gemm_post_merge_cast_consumer_keeps_buffer_tile_bridge_specs():
    kernel = gemm_kernel_with_post_merge_cast_consumer()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_func = artifact.device_mod["main_kernel"]
    payload = dict(require_tt_program(device_func).payload)
    bridge_specs = list(payload["buffer_tile_bridge_specs"])
    by_buffer = {str(spec["buffer"]): spec for spec in bridge_specs}

    assert "compute_epilogue_ops" not in payload
    assert {"C_local", "D_local"}.issubset(by_buffer)
    for name in ("C_local", "D_local"):
        spec = by_buffer[name]
        assert str(spec["scope"]) == "local"
        assert tuple(int(dim) for dim in spec["shape"]) == (32, 32)
        assert tuple(int(dim) for dim in spec["local_shape"]) == (8,)
        assert int(spec["thread_extent"]) == 128
        assert int(spec["replicate_extent"]) == 1
        assert len(spec["inverse_logical_index_exprs"]) == 3


def test_blackhole_gemm_post_merge_cast_consumer_uses_pack_tile_materialization():
    kernel = gemm_kernel_with_post_merge_cast_consumer()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    materializations = {
        str(plan["target_buffer"]): plan
        for plan in executable_spec["materialization_plans"]
    }
    assert "D_local" in materializations
    d_local = materializations["D_local"]
    assert str(d_local["materialization_protocol"]) == "cb_republish"
    assert str(d_local["publication_protocol"]) == "pack_tile"
    assert str(d_local["source_live_form"]) == "live_form_C_local"
    assert str(d_local["produced_live_form"]) == "live_form_D_local"

    d_local_configs = [
        cfg for cfg in executable_spec["cb_configs"] if str(cfg["name"]) == "D_local"
    ]
    assert len(d_local_configs) == 1
    d_local_cb_id = int(d_local_configs[0]["cb_id"])

    compute_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_source = str(compute_kernel["source_code"])
    assert f"pack_tile(0, {d_local_cb_id});" in compute_source
    assert f"tilelang_get_cb_write_ptr_bytes({d_local_cb_id})" not in compute_source
    assert "reinterpret_cast<uint32_t*>(tilelang_get_cb_write_ptr_bytes(" not in compute_source
    assert "reinterpret_cast<uint16_t*>(tilelang_get_cb_write_ptr_bytes(" not in compute_source


def test_blackhole_gemm_post_merge_cast_consumer_without_zero_preclear_keeps_materialization_gate():
    kernel = gemm_kernel_with_post_merge_cast_consumer(preclear_output_fragment=False)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert any("thread-distributed cb_republish materialization" in reason for reason in reasons)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    materializations = {
        str(plan["target_buffer"]): plan
        for plan in executable_spec["materialization_plans"]
    }
    assert str(materializations["D_local"]["publication_protocol"]) != "pack_tile"


def test_blackhole_gemm_direct_runtime_preserves_clear_accum_false_fragment_for_cast_consumer():
    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    d_output = torch.zeros(32, 32, dtype=torch.bfloat16)
    d_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1)).to(torch.bfloat16)

    kernel = gemm_kernel_with_post_merge_cast_consumer()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    compute_config = _require_compute_config(executable_spec)
    assert bool(compute_config["clear_accum"]) is False
    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact.codegen_mod["main"](a_torch, b_torch, d_output)
    assert_tensors_close_or_dump(
        d_output,
        d_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message=(
            "GEMM clear-accum-false preserve-for-cast-consumer direct-call output mismatch"
        ),
    )


def test_blackhole_fragment_fill_cast_publish_runtime():
    d_output = torch.zeros(32, 32, dtype=torch.bfloat16)
    d_ref = torch.full((32, 32), 3.5, dtype=torch.bfloat16)

    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    reasons = _direct_runtime_unsupported_reasons(artifact)
    assert not any("thread-distributed cb_republish materialization" in reason for reason in reasons)
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    artifact.codegen_mod["main"](d_output)
    assert_tensors_close_or_dump(
        d_output,
        d_ref,
        atol=0,
        rtol=0,
        failure_message=(
            "Fragment fill->cast->publish direct-call output mismatch"
        ),
    )


def test_blackhole_fragment_fill_cast_publish_exposes_buffer_tile_bridge_specs():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    mod = tilelang.tvm.IRModule({"main": kernel})
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = lower_blackhole_to_tt_target(mod)

    payload = dict(require_tt_program(mod["main"]).payload)
    bridge_specs = list(payload["buffer_tile_bridge_specs"])
    by_buffer = {str(spec["buffer"]): spec for spec in bridge_specs}

    assert {"C_local", "D_local"}.issubset(by_buffer)
    for name in ("C_local", "D_local"):
        spec = by_buffer[name]
        assert str(spec["scope"]) == "local"
        assert tuple(int(dim) for dim in spec["shape"]) == (32, 32)
        assert tuple(int(dim) for dim in spec["local_shape"]) == (8,)
        assert int(spec["thread_extent"]) == 128
        assert int(spec["replicate_extent"]) == 1
        assert len(spec["inverse_logical_index_exprs"]) == 3


def test_blackhole_fragment_fill_cast_publish_build_reads_executable_without_lowering_requirements():
    kernel = fragment_fill_cast_publish_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_main = artifact.device_mod["main_kernel"]
    assert "blackhole.lowering_requirements" not in device_main.attrs

    rebuilt = _rebuild_codegen_module_without_lowering_requirements(artifact)
    executable_spec = rebuilt.get_function_metadata("main")
    compute_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )

    assert compute_kernel["source_code"]
    assert executable_spec["cb_configs"]


def test_blackhole_gemm_direct_runtime_supports_transpose_a_compute_contract():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(128, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float().transpose(0, 1), b_torch.float().transpose(0, 1))

    kernel = gemm_kernel_with_transpose_flags(transpose_A=True, transpose_B=True)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_no_contract_family(executable_spec)
    gemm_op = _require_gemm_compute_op(executable_spec)
    assert bool(gemm_op["transpose_A"]) is True
    assert bool(gemm_op["transpose_B"]) is True
    assert int(gemm_op["Mt"]) == 1
    assert int(gemm_op["Nt"]) == 1
    assert int(gemm_op["Kt"]) == 4

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="GEMM transpose-A direct-call output mismatch",
    )


def test_blackhole_multicore_gemm_lowering_respects_transposed_b_layout():
    kernel = multicore_gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = lower_blackhole_to_tt_target(mod)

    func_text = mod["main"].script()
    assert func_text.count("tl.blackhole.write_tile_from_cb") == 1
    assert "T.tl.blackhole.read_tile_to_cb(B.data, bx, 1, 2048, 2)" in func_text
    assert "T.tl.blackhole.read_tile_to_cb(B.data, bx + 2, 1, 2048, 2)" in func_text
    assert "T.tl.blackhole.read_tile_to_cb(B.data, bx + 4, 1, 2048, 2)" in func_text
    assert "T.tl.blackhole.read_tile_to_cb(B.data, bx + 6, 1, 2048, 2)" in func_text


def test_blackhole_gemm_richer_accessor_schema_roundtrip():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            richer_common_runtime_args = [
                {
                    "name": "rank",
                    "kind": "accessor_common_u32",
                    "identity": "rank",
                    "dtype": "uint32",
                }
            ]
            func = _with_richer_accessor_schema(func, richer_common_runtime_args)
        rewritten[gvar] = func

    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    built = tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )

    assert built is not None


def test_blackhole_gemm_reuses_single_cb_requirement_for_accumulator_output():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    c_local_configs = [cfg for cfg in executable_spec["cb_configs"] if str(cfg["name"]) == "C_local"]
    assert len(c_local_configs) == 1

    c_local_cb_id = int(c_local_configs[0]["cb_id"])
    compute_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    writer_kernel = _require_blackhole_kernel(
        executable_spec["kernels"], kind="writer", core_type="ncrisc"
    )

    compute_source = str(compute_kernel["source_code"])
    writer_source = str(writer_kernel["source_code"])
    assert f"pack_tile(0, {c_local_cb_id});" in compute_source
    assert f"cb_push_back({c_local_cb_id}, 1);" in compute_source
    assert f"cb_wait_front({c_local_cb_id}, 1);" in writer_source
    assert f"cb_pop_front({c_local_cb_id}, 1);" in writer_source


def test_blackhole_gemm_basic():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    torch.manual_seed(0)
    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    target = Target("blackhole")
    kernel = gemm_kernel()

    try:
        with target:
            artifact = lower(kernel, target=target)
    except Exception as e:
        pytest.skip(f"Blackhole GEMM lowering not yet fully implemented: {e}")

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output, c_ref, atol=2e-1, rtol=2e-1, failure_message="GEMM direct-call output mismatch"
    )


def test_blackhole_gemm_multicore_direct_call():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n, k = 64, 64, 128
    torch.manual_seed(0)
    a_torch = torch.randn(m, k, dtype=torch.bfloat16)
    b_torch = torch.randn(n, k, dtype=torch.bfloat16)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = multicore_gemm_kernel(M=m, N=n, K=k)
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {
        str(gvar): func for gvar, func in artifact.device_mod.functions.items()
    }
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = extract_blackhole_core_plan(device_main)
    assert int(core_plan["logical_grid_x"]) == 2
    assert int(core_plan["logical_grid_y"]) == 2
    assert str(core_plan["linearization"]) == "row_major"
    assert len(core_plan["physical_cores"]) == 4
    assert len(core_plan["work_packets"]) == 4
    physical_cores = [
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    ]
    work_packet_cores = [
        (int(packet["core_x"]), int(packet["core_y"])) for packet in core_plan["work_packets"]
    ]
    assert len(set(physical_cores)) == 4
    assert len(set(work_packet_cores)) == 4
    assert work_packet_cores == physical_cores
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1]

    c_output = torch.zeros(m, n, dtype=torch.float32)
    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Multicore GEMM direct-call output mismatch",
    )


def test_blackhole_gemm_multicore_direct_call_supports_oversubscribed_work_packets():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n, k = 352, 352, 128
    torch.manual_seed(0)
    a_torch = torch.randn(m, k, dtype=torch.bfloat16)
    b_torch = torch.randn(n, k, dtype=torch.bfloat16)
    c_ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1))

    kernel = multicore_gemm_kernel(M=m, N=n, K=k)
    target = Target("blackhole")
    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {
        str(gvar): func for gvar, func in artifact.device_mod.functions.items()
    }
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = extract_blackhole_core_plan(device_main)
    assert int(core_plan["logical_grid_x"]) == 11
    assert int(core_plan["logical_grid_y"]) == 11
    assert len(core_plan["physical_cores"]) == 110
    assert len(core_plan["work_packets"]) == 110
    assert sum(int(packet["work_count"]) for packet in core_plan["work_packets"]) == 121
    assert max(int(packet["work_count"]) for packet in core_plan["work_packets"]) == 2

    c_output = torch.zeros(m, n, dtype=torch.float32)
    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="Oversubscribed multicore GEMM direct-call output mismatch",
    )


def test_blackhole_gemm_rejects_empty_work_packets_at_build_time():
    target = Target("blackhole")
    kernel = multicore_gemm_kernel()

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = dict(extract_blackhole_core_plan(device_main))
    assert list(core_plan["work_packets"])
    core_plan["work_packets"] = []

    with pytest.raises(
        tvm.error.InternalError,
        match="core_plan.work_packets|planner/runtime|TTCoreGroup requires work_packets",
    ):
        _rebuild_codegen_module_with_core_plan(artifact, core_plan)
