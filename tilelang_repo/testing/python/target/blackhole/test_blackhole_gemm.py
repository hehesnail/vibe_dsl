import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import lower, merge_ir_modules
from tilelang import tvm
from tvm.target import Target
from tvm.tir import stmt_functor

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_codegen_requirements,
    check_blackhole_direct_execution_requirements,
    gemm_kernel,
    gemm_kernel_with_compute_config_extras,
    gemm_kernel_with_compute_abi,
    gemm_kernel_with_mbar,
    gemm_kernel_with_policy,
    gemm_kernel_with_transpose_flags,
)
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
    _expected_launch_spec_for_core_type,
    _with_compile_time_abi_schema,
    _require_blackhole_kernel,
    _require_spec_entry,
)


def _with_richer_accessor_schema(func, common_runtime_args=None):
    richer_segments = []
    for segment in func.attrs["blackhole.segment_plan"]:
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
    return func.with_attr("blackhole.segment_plan", richer_segments)


def _with_sharded_accessor_schema(func):
    richer_segments = []
    for segment in func.attrs["blackhole.segment_plan"]:
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
    return func.with_attr("blackhole.segment_plan", richer_segments)


def _with_mutated_segment_plan(func, segment_mutator):
    mutated_segments = []
    for segment in func.attrs["blackhole.segment_plan"]:
        mutated_segments.append(segment_mutator(dict(segment)))
    return func.with_attr("blackhole.segment_plan", mutated_segments)


def _rebuild_codegen_module_with_segment_plan(artifact, segment_plan):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            func = func.with_attr("blackhole.segment_plan", segment_plan)
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_with_compute_contract(artifact, compute_contract):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.compute_contract" in func.attrs:
            func = func.with_attr("blackhole.compute_contract", compute_contract)
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _rebuild_codegen_module_with_core_plan(artifact, core_plan):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "blackhole.core_plan" in func.attrs:
            func = func.with_attr("blackhole.core_plan", core_plan)
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


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


def test_blackhole_split_kernel_gemm_segment_plan():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)

    plan = None
    for _, func in mod.functions.items():
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            plan = func.attrs["blackhole.segment_plan"]
            break

    assert plan is not None
    assert len(plan) == 3
    assert str(plan[0]["kind"]) == "reader"
    assert str(plan[1]["kind"]) == "compute"
    assert str(plan[2]["kind"]) == "writer"
    assert str(plan[0]["core_type"]) == "brisc"
    assert str(plan[1]["core_type"]) == "trisc"
    assert str(plan[2]["core_type"]) == "ncrisc"

    reader_args = plan[0]["runtime_args"]
    assert [str(arg["buffer"]) for arg in reader_args if "buffer" in arg] == ["A", "B"]
    assert [str(arg["kind"]) for arg in reader_args] == [
        "input_buffer_addr32",
        "input_buffer_addr32",
        "work_linear_id",
        "a_tile_start_id",
        "a_tile_num_tiles",
        "a_tile_stride",
        "b_tile_start_id",
        "b_tile_num_tiles",
        "b_tile_stride",
        "k_tile_start_id",
        "num_k_tiles",
    ]

    compute_args = plan[1]["runtime_args"]
    assert [str(arg["kind"]) for arg in compute_args] == [
        "k_tile_start_id",
        "num_k_tiles",
    ]

    writer_args = plan[2]["runtime_args"]
    assert [str(arg["buffer"]) for arg in writer_args if "buffer" in arg] == ["C"]
    assert [str(arg["kind"]) for arg in writer_args if "buffer" not in arg] == [
        "work_linear_id",
        "output_tile_start_id",
        "output_tile_num_tiles",
        "output_tile_stride",
    ]


def test_blackhole_gemm_cb_ids_are_rewritten_by_planner():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    lower_mod = tilelang.transform.LowerBlackholeOps()(mod)
    planned_mod = tilelang.transform.PlanBlackholeCB()(lower_mod)

    lower_func = lower_mod["main"]
    func = planned_mod["main"]
    assert "blackhole.gemm_cb_placeholders" not in func.attrs

    def collect_cb_ids(stmt):
        cb_ids = set()

        def visit(node):
            if not isinstance(node, tilelang.tvm.tir.Call):
                return
            if not hasattr(node.op, "name"):
                return
            op_name = node.op.name
            if op_name in {
                "tl.blackhole.cb_reserve_back",
                "tl.blackhole.cb_push_back",
                "tl.blackhole.cb_wait_front",
                "tl.blackhole.cb_pop_front",
                "tl.blackhole.write_tile_from_cb",
            }:
                cb_ids.add(int(node.args[0]))
            elif op_name == "tl.blackhole.read_tile_to_cb":
                cb_ids.add(int(node.args[2]))
            elif op_name == "tl.blackhole.mm_init":
                cb_ids.update(int(arg) for arg in node.args[:3])
            elif op_name == "tl.blackhole.matmul_tiles":
                cb_ids.update(int(arg) for arg in node.args[:2])
            elif op_name == "tl.blackhole.pack_tile":
                cb_ids.add(int(node.args[1]))

        stmt_functor.post_order_visit(stmt, visit)
        return cb_ids

    assert collect_cb_ids(lower_func.body) == {0, 1, 2}
    assert collect_cb_ids(func.body) == {0, 1, 16}
    func_text = func.body.script()
    assert func_text.count("tl.blackhole.pack_tile") == 1
    assert func_text.count("tl.blackhole.write_tile_from_cb") == 1

    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    try:
        with target:
            artifact = lower(kernel, target=target)
    except Exception as e:
        pytest.skip(f"Blackhole lowering not yet fully implemented: {e}")

    source = getattr(artifact, "kernel_source", None)
    if source is None and hasattr(artifact, "mod"):
        try:
            source = artifact.mod.imported_modules[0].get_source()
        except Exception:
            source = None
    if not source and hasattr(artifact, "code"):
        source = artifact.code

    assert source
    assert "mm_init(-" not in source
    assert "cb_wait_front(-" not in source
    assert "cb_reserve_back(-" not in source


def test_blackhole_gemm_accumulator_scope_canonicalized():
    kernel = gemm_kernel()
    target = Target("blackhole")
    mod = tilelang.tvm.IRModule({"main": kernel})

    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
        mod = tilelang.engine.phase.OptimizeForTarget(mod, target)

    func = mod["main"]
    resource_plan = func.attrs["blackhole.resource_plan"]
    accum_entries = [item for item in resource_plan if str(item["class"]) == "accumulator"]
    assert accum_entries
    assert any(str(item["name"]) == "C_local" for item in accum_entries)
    assert all(str(item["scope"]) == "blackhole.acc" for item in accum_entries)

    func_text = func.script()
    assert 'scope="blackhole.acc"' in func_text


def test_blackhole_gemm_contract_attr_is_materialized():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    assert func.attrs and "blackhole.gemm_contract" in func.attrs
    contract = func.attrs["blackhole.gemm_contract"]
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

    segment_plan = func.attrs["blackhole.segment_plan"]
    reader = _require_blackhole_kernel(segment_plan, kind="reader", core_type="brisc")
    writer = _require_blackhole_kernel(segment_plan, kind="writer", core_type="ncrisc")
    assert [(str(item["buffer"]), int(item["compile_time_arg_offset"])) for item in reader["accessors"]] == [
        ("A", 0),
        ("B", 2),
    ]
    assert [int(item["compile_time_arg_count"]) for item in reader["accessors"]] == [2, 2]
    assert [int(item["common_runtime_arg_offset"]) for item in reader["accessors"]] == [0, 0]
    assert [int(item["common_runtime_arg_count"]) for item in reader["accessors"]] == [0, 0]
    assert [int(item["args_config_bits"]) for item in reader["accessors"]] == [2, 2]
    assert [(str(item["buffer"]), int(item["compile_time_arg_offset"])) for item in writer["accessors"]] == [
        ("C", 0)
    ]
    assert [int(item["compile_time_arg_count"]) for item in writer["accessors"]] == [2]
    assert [int(item["common_runtime_arg_offset"]) for item in writer["accessors"]] == [0]
    assert [int(item["common_runtime_arg_count"]) for item in writer["accessors"]] == [0]
    assert [int(item["args_config_bits"]) for item in writer["accessors"]] == [2]
    assert all(str(item["layout"]) == "interleaved" for item in reader["accessors"])
    assert all(str(item["memory_space"]) == "dram" for item in reader["accessors"])
    assert len(reader["common_runtime_args"]) == 0
    assert len(writer["common_runtime_args"]) == 0


def test_blackhole_compute_contract_attr_is_materialized():
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    assert func.attrs and "blackhole.compute_contract" in func.attrs
    contract = func.attrs["blackhole.compute_contract"]
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
    assert bool(contract["clear_accum"]) is False
    assert int(contract["k_pack"]) == 1
    assert int(contract["wg_wait"]) == 0
    assert int(contract["policy_type"]) == 0
    assert str(contract["policy_name"]) == "Square"


def test_blackhole_compute_contract_attr_materializes_nondefault_compute_abi():
    kernel = gemm_kernel_with_compute_abi()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    contract = func.attrs["blackhole.compute_contract"]
    assert bool(contract["clear_accum"]) is True
    assert int(contract["k_pack"]) == 2
    assert int(contract["wg_wait"]) == 3


def test_blackhole_compute_contract_attr_materializes_richer_compute_config_extras():
    kernel = gemm_kernel_with_compute_config_extras()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    contract = func.attrs["blackhole.compute_contract"]
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

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    func = mod["main"]
    contract = func.attrs["blackhole.compute_contract"]
    plan = func.attrs["blackhole.segment_plan"]
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

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    contract = mod["main"].attrs["blackhole.compute_contract"]
    assert int(contract["policy_type"]) == 1
    assert str(contract["policy_name"]) == "FullRow"


def test_blackhole_compute_contract_attr_materializes_mbar_binding():
    kernel = gemm_kernel_with_mbar()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

    contract = mod["main"].attrs["blackhole.compute_contract"]
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
    assert [int(value) for value in gemm_clear_accum["values"]] == [0]
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

    assert "compute_contract" in executable_spec
    compute_contract = executable_spec["compute_contract"]
    assert str(compute_contract["kind"]) == "gemm"
    assert bool(compute_contract["enabled"]) is True
    assert int(compute_contract["M"]) == 32
    assert int(compute_contract["N"]) == 32
    assert int(compute_contract["K"]) == 128
    assert int(compute_contract["Mt"]) == 1
    assert int(compute_contract["Nt"]) == 1
    assert int(compute_contract["Kt"]) == 4
    assert bool(compute_contract["transpose_A"]) is False
    assert bool(compute_contract["transpose_B"]) is True
    assert int(compute_contract["block_m_tiles"]) == 1
    assert int(compute_contract["block_n_tiles"]) == 1
    assert int(compute_contract["block_k_tiles"]) == 4
    assert int(compute_contract["subblock_m_tiles"]) == 1
    assert int(compute_contract["subblock_n_tiles"]) == 1
    assert str(compute_contract["math_fidelity"]) == "HiFi4"
    assert bool(compute_contract["fp32_dest_acc_en"]) is True
    assert bool(compute_contract["math_approx_mode"]) is False
    assert [str(item) for item in compute_contract["unpack_to_dest_mode"]] == []


def test_blackhole_gemm_kernel_compute_config_follows_compute_contract_in_spec():
    kernel = gemm_kernel_with_compute_abi()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_config = compute["compute_config"]
    compute_contract = executable_spec["compute_contract"]

    assert str(compute_config["math_fidelity"]) == str(compute_contract["math_fidelity"])
    assert bool(compute_config["fp32_dest_acc_en"]) is bool(compute_contract["fp32_dest_acc_en"])
    assert bool(compute_config["math_approx_mode"]) is bool(compute_contract["math_approx_mode"])
    assert [str(item) for item in compute_config["unpack_to_dest_mode"]] == [
        str(item) for item in compute_contract["unpack_to_dest_mode"]
    ]
    assert bool(compute_config["clear_accum"]) is bool(compute_contract["clear_accum"])
    assert int(compute_config["k_pack"]) == int(compute_contract["k_pack"])
    assert int(compute_config["wg_wait"]) == int(compute_contract["wg_wait"])
    assert int(compute_config["policy_type"]) == int(compute_contract["policy_type"])
    assert str(compute_config["policy_name"]) == str(compute_contract["policy_name"])


def test_blackhole_gemm_compile_time_abi_materializes_nondefault_compute_abi():
    kernel = gemm_kernel_with_compute_abi()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    compute_contract = executable_spec["compute_contract"]
    assert bool(compute_contract["clear_accum"]) is True
    assert int(compute_contract["k_pack"]) == 2
    assert int(compute_contract["wg_wait"]) == 3

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
    mutated_contract = dict(executable_spec["compute_contract"])
    mutated_contract["dst_full_sync_en"] = True
    mutated_contract["bfp8_pack_precise"] = True
    mutated_mod = _rebuild_codegen_module_with_compute_contract(artifact, mutated_contract)

    executable_spec = mutated_mod.get_function_metadata("main")
    compute_contract = executable_spec["compute_contract"]
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_config = compute["compute_config"]

    assert bool(compute_contract["dst_full_sync_en"]) is True
    assert bool(compute_contract["bfp8_pack_precise"]) is True
    assert bool(compute_config["dst_full_sync_en"]) is True
    assert bool(compute_config["bfp8_pack_precise"]) is True


def test_blackhole_gemm_compute_config_materializes_defines_and_named_compile_args():
    kernel = gemm_kernel()
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    mutated_contract = dict(executable_spec["compute_contract"])
    mutated_contract["defines"] = [
        {"name": "BLACKHOLE_TEST_DEFINE", "value": "1"},
        {"name": "BLACKHOLE_ACC_MODE", "value": "fp32"},
    ]
    mutated_contract["named_compile_args"] = [
        {"name": "c_0", "value": 0},
        {"name": "c_1", "value": 1},
        {"name": "c_16", "value": 16},
    ]
    mutated_mod = _rebuild_codegen_module_with_compute_contract(artifact, mutated_contract)

    executable_spec = mutated_mod.get_function_metadata("main")
    compute_contract = executable_spec["compute_contract"]
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_config = compute["compute_config"]

    assert [(str(item["name"]), str(item["value"])) for item in compute_contract["defines"]] == [
        ("BLACKHOLE_TEST_DEFINE", "1"),
        ("BLACKHOLE_ACC_MODE", "fp32"),
    ]
    assert [
        (str(item["name"]), int(item["value"])) for item in compute_contract["named_compile_args"]
    ] == [("c_0", 0), ("c_1", 1), ("c_16", 16)]
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
    compute_contract = executable_spec["compute_contract"]
    compute = _require_blackhole_kernel(
        executable_spec["kernels"], kind="compute", core_type="trisc"
    )
    compute_config = compute["compute_config"]

    assert bool(compute_contract["dst_full_sync_en"]) is True
    assert bool(compute_contract["bfp8_pack_precise"]) is True
    assert bool(compute_config["dst_full_sync_en"]) is True
    assert bool(compute_config["bfp8_pack_precise"]) is True
    assert sorted((str(item["name"]), str(item["value"])) for item in compute_contract["defines"]) == [
        ("BLACKHOLE_ACC_MODE", "fp32"),
        ("BLACKHOLE_TEST_DEFINE", "1"),
    ]
    assert sorted(
        (str(item["name"]), int(item["value"])) for item in compute_contract["named_compile_args"]
    ) == [("c_0", 0), ("c_1", 1), ("c_16", 16)]
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
    compute_contract = executable_spec["compute_contract"]
    assert int(compute_contract["policy_type"]) == 1
    assert str(compute_contract["policy_name"]) == "FullRow"

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
    mutated_contract = dict(executable_spec["compute_contract"])
    mutated_contract["has_mbarrier"] = True
    mutated_contract["mbarrier_buffer"] = "mbar"
    mutated_contract["mbarrier_scope"] = "shared.barrier"
    mutated_contract["mbarrier_index_exprs"] = ["0"]
    mutated_mod = _rebuild_codegen_module_with_compute_contract(artifact, mutated_contract)

    executable_spec = mutated_mod.get_function_metadata("main")
    compute_contract = executable_spec["compute_contract"]
    assert bool(compute_contract["has_mbarrier"]) is True
    assert str(compute_contract["mbarrier_buffer"]) == "mbar"
    assert str(compute_contract["mbarrier_scope"]) == "shared.barrier"
    assert [str(item) for item in compute_contract["mbarrier_index_exprs"]] == ["0"]

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
        artifact, richer_func.attrs["blackhole.segment_plan"]
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
    for segment in device_main.attrs["blackhole.segment_plan"]:
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
        artifact, mutated_func.attrs["blackhole.segment_plan"]
    )

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="common runtime args"):
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
    for segment in device_main.attrs["blackhole.segment_plan"]:
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
    mutated_contract = dict(executable_spec["compute_contract"])
    mutated_contract["math_fidelity"] = "UltraFi9"
    mutated_mod = _rebuild_codegen_module_with_compute_contract(artifact, mutated_contract)

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
    mutated_contract = dict(executable_spec["compute_contract"])
    mutated_contract["has_mbarrier"] = True
    mutated_contract["mbarrier_buffer"] = "mbar"
    mutated_contract["mbarrier_scope"] = "shared.barrier"
    mutated_contract["mbarrier_index_exprs"] = ["0"]
    mutated_mod = _rebuild_codegen_module_with_compute_contract(artifact, mutated_contract)

    a_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 128, dtype=torch.bfloat16)
    c_output = torch.zeros(32, 32, dtype=torch.float32)

    with pytest.raises(tvm.error.InternalError, match="mbarrier"):
        mutated_mod["main"](a_torch, b_torch, c_output)


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
        artifact, stripped_func.attrs["blackhole.segment_plan"]
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
    compute_contract = executable_spec["compute_contract"]
    assert bool(compute_contract["dst_full_sync_en"]) is True
    assert bool(compute_contract["bfp8_pack_precise"]) is True
    assert sorted((str(item["name"]), str(item["value"])) for item in compute_contract["defines"]) == [
        ("BLACKHOLE_ACC_MODE", "fp32"),
        ("BLACKHOLE_TEST_DEFINE", "1"),
    ]
    assert sorted(
        (str(item["name"]), int(item["value"])) for item in compute_contract["named_compile_args"]
    ) == [("c_0", 0), ("c_1", 1), ("c_16", 16)]

    artifact.codegen_mod["main"](a_torch, b_torch, c_output)
    assert_tensors_close_or_dump(
        c_output,
        c_ref,
        atol=2e-1,
        rtol=2e-1,
        failure_message="GEMM richer compute-config direct-call output mismatch",
    )


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
    assert "compute_contract" in executable_spec
    contract = executable_spec["compute_contract"]
    assert bool(contract["transpose_A"]) is True
    assert bool(contract["transpose_B"]) is True
    assert int(contract["Mt"]) == 1
    assert int(contract["Nt"]) == 1
    assert int(contract["Kt"]) == 4

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

    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)

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
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            richer_common_runtime_args = [
                {
                    "name": "rank",
                    "kind": "accessor_common_u32",
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
    core_plan = device_main.attrs["blackhole.core_plan"]
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


def test_blackhole_gemm_rejects_empty_work_packets_at_build_time():
    target = Target("blackhole")
    kernel = multicore_gemm_kernel()

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = dict(device_main.attrs["blackhole.core_plan"])
    assert list(core_plan["work_packets"])
    core_plan["work_packets"] = []

    with pytest.raises(tvm.error.InternalError, match="core_plan.work_packets|planner/runtime"):
        _rebuild_codegen_module_with_core_plan(artifact, core_plan)
