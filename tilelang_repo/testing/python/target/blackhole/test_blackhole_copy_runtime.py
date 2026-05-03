import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import lower
from tilelang.engine.lower import merge_ir_modules
from tvm.target import Target
from tilelang import tvm
from tvm import tir
from tvm.tir import stmt_functor

from .common import (
    assert_tensors_close_or_dump,
    check_blackhole_direct_execution_requirements,
    extract_blackhole_cb_configs,
    extract_blackhole_total_l1_bytes,
    extract_blackhole_work_per_core,
    grid_indexed_staged_copy_kernel,
    rebuild_tt_buffer_distribution_plan,
    rebuild_tt_abi_plan,
    rebuild_tt_kernel,
    rebuild_tt_core_group,
    rebuild_tt_program,
    rebuild_tt_reshard_plan,
    rebuild_tt_semaphore_plan,
    require_tt_program,
    staged_copy_kernel,
    staged_stick_copy_kernel,
    tt_per_work_arg_specs_to_list,
    tt_runtime_arg_specs_to_list,
)
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
    _external_sharded_l1_copy_kernel,
)


def _rebuild_direct_runtime_module_with_tt_program(
    artifact, *, tt_program_mutator=None, body_mutator=None
):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            if body_mutator is not None:
                func = func.with_body(body_mutator(func.body))
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


def _rebuild_direct_runtime_module_with_runtime_args(artifact, runtime_args):
    def mutate(tt_program):
        abi_plans = [
            rebuild_tt_abi_plan(abi_plan, runtime_args=runtime_args)
            for abi_plan in tt_program.abi_plans
        ]
        return rebuild_tt_program(tt_program, abi_plans=abi_plans)

    return _rebuild_direct_runtime_module_with_tt_program(artifact, tt_program_mutator=mutate)


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
            }
        )
    return normalized


def _rebuild_semaphore_plans(tt_program, semaphore_plan):
    make_tt_semaphore_plan = tvm.get_global_func("tl.TTSemaphorePlan")
    normalized = _normalize_semaphore_plan_for_tt_program(semaphore_plan)
    existing = list(tt_program.semaphore_plans)
    rebuilt = []
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
                )
            )
        else:
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
                )
            )
    return rebuilt


def _rebuild_direct_runtime_module_with_body_and_attrs(
    artifact, *, body_mutator=None, semaphore_plan=None, runtime_args=None
):
    def mutate(tt_program):
        kwargs = {}
        if semaphore_plan is not None:
            kwargs["semaphore_plans"] = _rebuild_semaphore_plans(tt_program, semaphore_plan)
        if runtime_args is not None:
            kwargs["abi_plans"] = [
                rebuild_tt_abi_plan(abi_plan, runtime_args=runtime_args)
                for abi_plan in tt_program.abi_plans
            ]
        return rebuild_tt_program(tt_program, **kwargs)

    return _rebuild_direct_runtime_module_with_tt_program(
        artifact, tt_program_mutator=mutate, body_mutator=body_mutator
    )


def _rebuild_direct_runtime_module_with_core_plan(artifact, core_plan):
    def mutate(tt_program):
        core_groups = list(tt_program.core_groups)
        if not core_groups:
            pytest.fail("Expected TTProgram to carry a TTCoreGroup")
        core_groups[0] = rebuild_tt_core_group(
            core_groups[0],
            logical_grid_x=int(core_plan["logical_grid_x"]),
            logical_grid_y=int(core_plan["logical_grid_y"]),
            linearization=str(core_plan["linearization"]),
            physical_cores=list(core_plan["physical_cores"]),
            work_packets=list(core_plan["work_packets"]),
        )
        return rebuild_tt_program(tt_program, core_groups=core_groups)

    return _rebuild_direct_runtime_module_with_tt_program(artifact, tt_program_mutator=mutate)


def _rebuild_direct_runtime_module_with_executable_mutator(artifact, executable_mutator):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.blackhole_executable" in func.attrs:
            executable = {
                str(key): value for key, value in func.attrs["tl.blackhole_executable"].items()
            }
            func = func.with_attr("tl.blackhole_executable", executable_mutator(executable))
        rewritten[gvar] = func
    device_mod = tvm.IRModule(rewritten, global_infos=artifact.device_mod.global_infos)
    build_mod = merge_ir_modules(artifact.host_mod, device_mod)
    target = Target("blackhole")
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def _extract_tt_program_core_plan(device_main):
    tt_program = require_tt_program(device_main)
    if not tt_program.core_groups:
        pytest.fail("Expected TTProgram to carry a TTCoreGroup")
    core_group = tt_program.core_groups[0]
    return {
        "logical_grid_x": int(core_group.logical_grid_x),
        "logical_grid_y": int(core_group.logical_grid_y),
        "linearization": str(core_group.linearization),
        "physical_cores": list(core_group.physical_cores),
        "work_packets": list(core_group.work_packets),
    }


def _extract_tt_program_runtime_args(device_main):
    tt_program = require_tt_program(device_main)
    if not tt_program.abi_plans:
        pytest.fail("Expected TTProgram to carry a TTABIPlan")
    return tt_runtime_arg_specs_to_list(tt_program.abi_plans[0].runtime_args)


def _require_device_main(artifact):
    for func in artifact.device_mod.functions.values():
        if getattr(func, "attrs", None) and "tl.tt_program" in func.attrs:
            return func
    pytest.fail("Expected artifact device module to carry tl.tt_program")


def _bf16_matrix(m, n):
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    values = (values.remainder(251) - 125) / 17
    return values.to(torch.bfloat16)


def _explicit_user_reshard_copy_kernel(
    *,
    grid_x: int,
    grid_y: int,
    tile_m: int = 32,
    tile_n: int = 32,
):
    m = grid_y * tile_m
    n = grid_x * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "bfloat16"),
        B: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            resident_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    resident_tile: T.sharded_l1(
                        strategy="block",
                        grid=T.CoreGrid(x=8, y=8),
                        shard_shape=(tile_m, tile_n),
                        orientation="row_major",
                        allow_reshard=True,
                    ),
                    B: T.interleaved_dram(),
                }
            )
            T.copy(A[by * tile_m, bx * tile_n], resident_tile)
            T.copy(resident_tile, B[by * tile_m, bx * tile_n])

    return main


def _explicit_multi_reshard_copy_kernel(
    *,
    grid_x: int,
    grid_y: int,
    tile_m: int = 32,
    tile_n: int = 32,
):
    m = grid_y * tile_m
    n = grid_x * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "bfloat16"),
        C: T.Tensor((m, n), "bfloat16"),
        B: T.Tensor((m, n), "bfloat16"),
        D: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            resident_a = T.alloc_shared((tile_m, tile_n), "bfloat16")
            resident_c = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    C: T.interleaved_dram(),
                    resident_a: T.sharded_l1(
                        strategy="block",
                        grid=T.CoreGrid(x=8, y=8),
                        shard_shape=(tile_m, tile_n),
                        orientation="row_major",
                        allow_reshard=True,
                    ),
                    resident_c: T.sharded_l1(
                        strategy="block",
                        grid=T.CoreGrid(x=8, y=8),
                        shard_shape=(tile_m, tile_n),
                        orientation="row_major",
                        allow_reshard=True,
                    ),
                    B: T.interleaved_dram(),
                    D: T.interleaved_dram(),
                }
            )
            T.copy(A[by * tile_m, bx * tile_n], resident_a)
            T.copy(resident_a, B[by * tile_m, bx * tile_n])
            T.copy(C[by * tile_m, bx * tile_n], resident_c)
            T.copy(resident_c, D[by * tile_m, bx * tile_n])

    return main


def _offset_grid_indexed_reshard_copy_kernel(
    *,
    grid_x: int,
    grid_y: int,
    src_tile_row: int,
    src_tile_col: int,
    dst_tile_row: int,
    dst_tile_col: int,
    tile_m: int = 32,
    tile_n: int = 32,
):
    input_m = (src_tile_row + grid_y) * tile_m
    input_n = (src_tile_col + grid_x) * tile_n
    output_m = (dst_tile_row + grid_y) * tile_m
    output_n = (dst_tile_col + grid_x) * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((input_m, input_n), "bfloat16"),
        B: T.Tensor((output_m, output_n), "bfloat16"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            resident_tile = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_memory_config(
                {
                    A: T.interleaved_dram(),
                    resident_tile: T.sharded_l1(
                        strategy="block",
                        grid=T.CoreGrid(x=8, y=8),
                        shard_shape=(tile_m, tile_n),
                        orientation="row_major",
                        allow_reshard=True,
                    ),
                    B: T.interleaved_dram(),
                }
            )
            T.copy(
                A[(src_tile_row + by) * tile_m, (src_tile_col + bx) * tile_n],
                resident_tile,
            )
            T.copy(
                resident_tile,
                B[(dst_tile_row + by) * tile_m, (dst_tile_col + bx) * tile_n],
            )

    return main


def _t3_memory_configs_by_subject(executable_spec):
    return {
        str(plan["subject"]): plan
        for plan in executable_spec["tensor_memory_config_plans"]
    }


def _t3_reshard_plans_by_edge(executable_spec):
    return {
        (str(plan["source_value"]), str(plan["target_value"])): plan
        for plan in executable_spec["reshard_plans"]
    }


def _t3_distributions_by_buffer(executable_spec):
    return {
        str(plan["buffer"]): plan
        for plan in executable_spec["buffer_distribution_plans"]
    }


def _assert_t3_interleaved_to_sharded_contract(
    executable_spec,
    *,
    source: str,
    target: str,
    source_region_shape=(32, 32),
):
    memory_configs = _t3_memory_configs_by_subject(executable_spec)
    reshard_plans = _t3_reshard_plans_by_edge(executable_spec)
    distributions = _t3_distributions_by_buffer(executable_spec)

    assert str(memory_configs[source]["memory_layout"]) == "INTERLEAVED"
    assert str(memory_configs[source]["buffer_type"]) == "DRAM"
    assert str(memory_configs[target]["memory_layout"]) == "BLOCK_SHARDED"
    assert str(memory_configs[target]["buffer_type"]) == "L1"
    assert str(memory_configs[target]["source_buffer"]) == source

    sharded_distribution = distributions[target]
    assert str(sharded_distribution["distribution_kind"]) == "sharded"
    assert str(sharded_distribution["memory_space"]) == "L1"
    assert str(sharded_distribution["sharding_strategy"]) == "block"
    assert str(sharded_distribution["shard_orientation"]) == "row_major"
    assert str(sharded_distribution["source_buffer"]) == source
    assert str(sharded_distribution["source_region_kind"]) == "per_work_tile"
    assert tuple(int(dim) for dim in sharded_distribution["source_region_shape"]) == (
        source_region_shape
    )
    assert str(sharded_distribution["logical_index_mapping"]) == "work_packet_row_major"
    assert str(sharded_distribution["core_local_address_mapping"]) == "l1_shard_linear"

    reshard = reshard_plans[(source, target)]
    assert str(reshard["source_memory_config_plan"]) == str(memory_configs[source]["name"])
    assert str(reshard["target_memory_config_plan"]) == str(memory_configs[target]["name"])
    assert str(reshard["conversion_kind"]) == "interleaved_to_sharded"
    assert str(reshard["source_region_kind"]) == "per_work_tile"
    assert tuple(int(dim) for dim in reshard["source_region_shape"]) == source_region_shape
    assert str(reshard["materialization_protocol"]) == "staged_copy"
    assert str(reshard["scheduling_kind"]) == "runtime"
    assert str(reshard["inserted_by"]) == "planner"
    assert str(reshard["admission_status"]) == "admitted"
    assert str(reshard["unsupported_reason"]) == ""
    assert not executable_spec.get("direct_runtime_unsupported_reasons", [])


def _inject_worker_semaphore_handshake(remote_core_x, remote_core_y):
    def mutate(original_body):
        def postorder(node):
            if not isinstance(node, tir.SeqStmt) or len(node.seq) != 6:
                return node
            try:
                read_call = node.seq[1].value
                write_call = node.seq[4].value
            except AttributeError:
                return node
            if read_call.op.name != "tl.blackhole.read_tile_to_cb":
                return node
            if write_call.op.name != "tl.blackhole.write_tile_from_cb":
                return node

            bx = read_call.args[1]
            consumer_wait_semaphore_addr = tir.Var("copy_sem_addr_consumer_wait", "uint32")
            producer_remote_semaphore_addr = tir.Var("copy_sem_addr_consumer_remote", "uint32")
            remote_noc_x = tir.call_intrin(
                "uint32",
                tir.op.Op.get("tl.blackhole.runtime_arg_u32"),
                tir.StringImm("remote_noc_x"),
            )
            remote_noc_y = tir.call_intrin(
                "uint32",
                tir.op.Op.get("tl.blackhole.runtime_arg_u32"),
                tir.StringImm("remote_noc_y"),
            )
            consumer_semaphore_id = tir.call_intrin(
                "uint32", tir.op.Op.get("tl.blackhole.get_semaphore"), tir.IntImm("uint32", 0)
            )
            semaphore_wait = tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_wait"),
                    consumer_wait_semaphore_addr,
                    tir.IntImm("uint32", 1),
                )
            )
            semaphore_inc_remote = tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.blackhole.semaphore_inc_remote"),
                    producer_remote_semaphore_addr,
                    remote_noc_x,
                    remote_noc_y,
                    tir.IntImm("uint32", 1),
                )
            )
            consumer_wait = tir.IfThenElse(
                bx == tir.IntImm("int32", 1),
                tir.LetStmt(consumer_wait_semaphore_addr, consumer_semaphore_id, semaphore_wait),
                tir.Evaluate(0),
            )
            producer_set = tir.IfThenElse(
                bx == tir.IntImm("int32", 0),
                tir.LetStmt(producer_remote_semaphore_addr, consumer_semaphore_id, semaphore_inc_remote),
                tir.Evaluate(0),
            )
            return tir.SeqStmt([consumer_wait, node, producer_set])

        return stmt_functor.ir_transform(original_body, None, postorder, ["tir.SeqStmt"])

    return mutate


def test_blackhole_module_direct_call():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=m // 32, tile_cols=n // 32)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output, b_ref, atol=1e-3, rtol=1e-3, failure_message="Direct-call output mismatch"
    )


def test_blackhole_module_direct_call_rectangular_tiles():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 64
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1, tile_m=32, tile_n=64)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Rectangular direct-call output mismatch",
    )


def test_blackhole_t4_external_sharded_l1_accessor_direct_runtime_bf16():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 64
    a_torch = _bf16_matrix(m, n)
    b_output = torch.zeros_like(a_torch)

    target = Target("blackhole")
    kernel = _external_sharded_l1_copy_kernel(grid_x=2, grid_y=1)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=1e-3,
        rtol=1e-3,
        failure_message="External sharded L1 accessor direct-runtime output mismatch",
    )


def test_blackhole_t4_direct_runtime_rejects_sharded_accessor_missing_distribution_metadata():
    target = Target("blackhole")
    kernel = _external_sharded_l1_copy_kernel(grid_x=2, grid_y=1)
    with target:
        artifact = lower(kernel, target=target)

    def drop_shard_shape(executable):
        distributions = []
        for item in executable["buffer_distribution_plans"]:
            plan = {str(key): value for key, value in item.items()}
            if str(plan.get("buffer", "")) == "A":
                plan.pop("shard_shape", None)
            distributions.append(plan)
        executable["buffer_distribution_plans"] = distributions
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="sharded.*requires shard_shape|lacks sharded L1",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, drop_shard_shape
        )


def test_blackhole_module_direct_call_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, :16] = a_torch[:, :16]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=32, tile_n=16, global_n=32, dtype="float32")
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_tall_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, :16] = a_torch[:, :16]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(tile_m=64, tile_n=16, global_n=32, dtype="float32")
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Tall stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_offset_stick_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 64, 48
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, 16:32] = a_torch[:, 16:32]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=64, tile_n=16, global_n=48, dtype="float32", src_col=16, dst_col=16
    )
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Offset stick direct-call output mismatch",
    )


def test_blackhole_module_direct_call_grid_indexed_copy_multicore_launch():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 2, 3
    m, n = grid_y * 32, grid_x * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    core_plan = executable_spec["core_plan"]
    assert int(core_plan["logical_grid_x"]) == grid_x
    assert int(core_plan["logical_grid_y"]) == grid_y
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6
    distribution_by_buffer = {
        str(plan["buffer"]): plan
        for plan in executable_spec["buffer_distribution_plans"]
    }
    sharded_l1 = distribution_by_buffer["A_shared"]
    assert str(sharded_l1["distribution_kind"]) == "sharded"
    assert str(sharded_l1["memory_space"]) == "L1"
    assert str(sharded_l1["sharding_strategy"]) == "block"
    assert str(sharded_l1["shard_orientation"]) == "row_major"
    assert str(sharded_l1["source_buffer"]) == "A"
    assert str(sharded_l1["source_region_kind"]) == "per_work_tile"
    assert tuple(int(dim) for dim in sharded_l1["source_region_shape"]) == (32, 32)
    assert str(sharded_l1["logical_index_mapping"]) == "work_packet_row_major"
    assert str(sharded_l1["core_local_address_mapping"]) == "l1_shard_linear"
    per_work_arg_specs = {
        (spec["buffer"], spec["descriptor_kind"]): spec["value_source"]
        for spec in executable_spec["per_work_arg_specs"]
    }
    assert per_work_arg_specs[("A", "tile_start")] == "work_linear_id"
    assert per_work_arg_specs[("B", "tile_start")] == "work_linear_id"
    kernel_per_work_arg_specs = {
        (spec["buffer"], spec["descriptor_kind"]): spec["value_source"]
        for spec in executable_spec["kernels"][0]["per_work_arg_specs"]
    }
    assert kernel_per_work_arg_specs == per_work_arg_specs

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Grid-indexed direct-call output mismatch",
    )


@pytest.mark.parametrize(
    "grid_x,grid_y",
    [
        (32, 32),
        (128, 64),
        (64, 128),
        (128, 128),
    ],
)
def test_blackhole_t3_large_shape_explicit_reshard_direct_runtime(grid_x, grid_y):
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    tile_m, tile_n = 32, 32
    m, n = grid_y * tile_m, grid_x * tile_n
    a_torch = _bf16_matrix(m, n)
    b_output = torch.zeros_like(a_torch)

    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    device_main = _require_device_main(artifact)
    assert int(extract_blackhole_work_per_core(device_main)) > 1
    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="A",
        target="resident_tile",
        source_region_shape=(tile_m, tile_n),
    )

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=0.0,
        rtol=0.0,
        failure_message=f"T3 large-shape reshard copy mismatch for {m}x{n}",
    )


def test_blackhole_t3_explicit_user_placement_reshard_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 5, 4
    m, n = grid_y * 32, grid_x * 32
    a_torch = _bf16_matrix(m, n)
    b_output = torch.zeros_like(a_torch)

    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    assert "A_shared" not in _t3_memory_configs_by_subject(executable_spec)
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="A",
        target="resident_tile",
    )

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=0.0,
        rtol=0.0,
        failure_message="T3 explicit user-placement reshard copy mismatch",
    )


def test_blackhole_t3_multiple_projected_reshards_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 4, 3
    m, n = grid_y * 32, grid_x * 32
    a_torch = _bf16_matrix(m, n)
    c_torch = (_bf16_matrix(m, n).float() * -0.5).to(torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    d_output = torch.zeros_like(c_torch)

    target = Target("blackhole")
    kernel = _explicit_multi_reshard_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="A",
        target="resident_a",
    )
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="C",
        target="resident_c",
    )
    assert len(_t3_reshard_plans_by_edge(executable_spec)) == 2

    artifact.codegen_mod["main"](a_torch, c_torch, b_output, d_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=0.0,
        rtol=0.0,
        failure_message="T3 multi-reshard A output mismatch",
    )
    assert_tensors_close_or_dump(
        d_output,
        c_torch,
        atol=0.0,
        rtol=0.0,
        failure_message="T3 multi-reshard C output mismatch",
    )


def test_blackhole_t3_large_offset_subregion_reshard_direct_runtime():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 64, 64
    src_tile_row, src_tile_col = 64, 64
    dst_tile_row, dst_tile_col = 32, 16
    tile_m, tile_n = 32, 32
    input_m = (src_tile_row + grid_y) * tile_m
    input_n = (src_tile_col + grid_x) * tile_n
    output_m = (dst_tile_row + grid_y) * tile_m
    output_n = (dst_tile_col + grid_x) * tile_n

    a_torch = _bf16_matrix(input_m, input_n)
    b_output = torch.zeros((output_m, output_n), dtype=torch.bfloat16)
    b_ref = torch.zeros_like(b_output)
    src_row = src_tile_row * tile_m
    src_col = src_tile_col * tile_n
    dst_row = dst_tile_row * tile_m
    dst_col = dst_tile_col * tile_n
    region_m = grid_y * tile_m
    region_n = grid_x * tile_n
    b_ref[dst_row : dst_row + region_m, dst_col : dst_col + region_n] = (
        a_torch[src_row : src_row + region_m, src_col : src_col + region_n]
    )

    target = Target("blackhole")
    kernel = _offset_grid_indexed_reshard_copy_kernel(
        grid_x=grid_x,
        grid_y=grid_y,
        src_tile_row=src_tile_row,
        src_tile_col=src_tile_col,
        dst_tile_row=dst_tile_row,
        dst_tile_col=dst_tile_col,
    )
    with target:
        artifact = lower(kernel, target=target)

    device_main = _require_device_main(artifact)
    assert int(extract_blackhole_work_per_core(device_main)) > 1
    executable_spec = _extract_blackhole_executable_spec(artifact)
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="A",
        target="resident_tile",
    )

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=0.0,
        rtol=0.0,
        failure_message="T3 large offset/subregion reshard copy mismatch",
    )


def test_blackhole_t3_serialized_module_preserves_reshard_runtime_contract():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 3, 2
    m, n = grid_y * 32, grid_x * 32
    a_torch = _bf16_matrix(m, n)
    b_output = torch.zeros_like(a_torch)

    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    serialized = artifact.codegen_mod.save_to_bytes()
    assert len(serialized) > 0
    loader = tvm.get_global_func("ffi.Module.load_from_bytes.blackhole")
    loaded = loader(serialized)
    executable_spec = loaded.get_function_metadata("main")
    _assert_t3_interleaved_to_sharded_contract(
        executable_spec,
        source="A",
        target="resident_tile",
    )

    loaded["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=0.0,
        rtol=0.0,
        failure_message="T3 serialized module reshard copy mismatch",
    )


def test_blackhole_module_direct_call_page_indexed_copy_consumes_address_contract():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.float32)
    b_output = torch.zeros_like(a_torch)
    b_ref = torch.zeros_like(a_torch)
    b_ref[:, 0:16] = a_torch[:, 0:16]

    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=32,
        tile_n=16,
        global_n=32,
        dtype="float32",
        src_col=0,
        dst_col=0,
    )
    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    kernel_spec = executable_spec["kernels"][0]
    accessors = {str(accessor["buffer"]): accessor for accessor in kernel_spec["accessors"]}
    compile_specs = {
        str(spec["buffer"]): spec
        for spec in kernel_spec["compile_time_arg_specs"]
        if "buffer" in spec and "accessor_cta" in str(spec["kind"])
    }
    assert str(compile_specs["A"]["kind"]) == "page_indexed_accessor_cta"
    assert str(compile_specs["B"]["kind"]) == "page_indexed_accessor_cta"
    assert str(accessors["A"]["layout"]) == "page_indexed"
    assert str(accessors["B"]["layout"]) == "page_indexed"
    assert int(accessors["A"]["transport_page_size"]) == 64
    assert int(accessors["B"]["transport_page_size"]) == 64
    distribution_by_buffer = {
        str(plan["buffer"]): plan
        for plan in executable_spec["buffer_distribution_plans"]
    }
    assert str(distribution_by_buffer["A"]["logical_index_mapping"]) == "interleaved_page_index"
    assert str(distribution_by_buffer["B"]["logical_index_mapping"]) == "interleaved_page_index"
    assert str(distribution_by_buffer["A"]["layout"]) == "page_indexed"
    assert str(distribution_by_buffer["B"]["layout"]) == "page_indexed"
    assert int(distribution_by_buffer["A"]["page_size_bytes"]) == 64
    assert int(distribution_by_buffer["B"]["page_size_bytes"]) == 64

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-5,
        rtol=1e-5,
        failure_message="Page-indexed copy direct-call output mismatch",
    )


def test_blackhole_t4_direct_runtime_rejects_page_indexed_accessor_missing_page_metadata():
    target = Target("blackhole")
    kernel = staged_stick_copy_kernel(
        tile_m=32,
        tile_n=16,
        global_n=32,
        dtype="float32",
        src_col=0,
        dst_col=0,
    )
    with target:
        artifact = lower(kernel, target=target)

    def drop_page_size_bytes(executable):
        plans = []
        for item in executable["buffer_distribution_plans"]:
            plan = {str(key): value for key, value in item.items()}
            if str(plan.get("buffer", "")) == "A":
                plan.pop("page_size_bytes", None)
            plans.append(plan)
        executable["buffer_distribution_plans"] = plans
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="buffer_distribution_plans.*A.*page_size_bytes",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, drop_page_size_bytes
        )


def test_blackhole_module_direct_call_grid_indexed_copy_worker_semaphore_handshake():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 2, 1
    m, n = grid_y * 32, grid_x * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=grid_x, grid_y=grid_y)
    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    core_plan = executable_spec["core_plan"]
    producer_core = core_plan["physical_cores"][0]
    consumer_core = core_plan["physical_cores"][1]
    assert int(producer_core["core_x"]) == 0
    assert int(producer_core["core_y"]) == 0
    assert int(consumer_core["core_x"]) == 1
    assert int(consumer_core["core_y"]) == 0

    runtime_args = list(executable_spec["runtime_args"])
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

    semaphore_plan = [
        {
            "id": 0,
            "initial_value": 0,
            "core_type": "worker",
            "core_ranges": [
                {
                    "start": {"core_x": 1, "core_y": 2},
                    "end": {"core_x": 2, "core_y": 2},
                }
            ],
        }
        ,
    ]
    mutated_mod = _rebuild_direct_runtime_module_with_body_and_attrs(
        artifact,
        body_mutator=_inject_worker_semaphore_handshake(
            consumer_core["core_x"], consumer_core["core_y"]
        ),
        semaphore_plan=semaphore_plan,
        runtime_args=runtime_args,
    )

    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Worker semaphore producer-consumer copy mismatch",
    )


def test_blackhole_module_direct_call_accepts_oversubscribed_multi_core_launch():
    kernel = grid_indexed_staged_copy_kernel(grid_x=15, grid_y=10)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = _extract_tt_program_core_plan(device_main)
    assert int(core_plan["logical_grid_x"]) == 15
    assert int(core_plan["logical_grid_y"]) == 10
    assert int(extract_blackhole_work_per_core(device_main)) == 2

    m, n = 10 * 32, 15 * 32
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=0.0,
        rtol=0.0,
        failure_message="Oversubscribed multi-core copy output mismatch",
    )


def test_blackhole_module_direct_call_rejects_oversubscribed_communication_contract():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = grid_indexed_staged_copy_kernel(grid_x=15, grid_y=10)
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
    mutated_mod = _rebuild_direct_runtime_module_with_body_and_attrs(
        artifact,
        semaphore_plan=semaphore_plan,
    )

    m, n = 10 * 32, 15 * 32
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)

    with pytest.raises(
        tvm.error.InternalError,
        match="oversubscribed launch|no explicit semaphore or remote-core synchronization contract",
    ):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_module_direct_call_rejects_empty_work_packets_at_build_time():
    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)

    with target:
        artifact = lower(kernel, target=target)

    executable_spec = _extract_blackhole_executable_spec(artifact)
    core_plan = dict(executable_spec["core_plan"])
    assert list(core_plan["work_packets"])
    core_plan["work_packets"] = []

    with pytest.raises(
        tvm.error.InternalError,
        match="core_plan.work_packets|planner/runtime|TTCoreGroup requires work_packets",
    ):
        _rebuild_direct_runtime_module_with_core_plan(artifact, core_plan)


def test_blackhole_build_rejects_incomplete_executable_buffer_distribution_contract():
    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=2)

    with target:
        artifact = lower(kernel, target=target)

    def drop_sharded_source_buffer(executable):
        plans = []
        for item in executable["buffer_distribution_plans"]:
            plan = {str(key): value for key, value in item.items()}
            if str(plan.get("distribution_kind", "")) == "sharded":
                plan.pop("source_buffer", None)
            plans.append(plan)
        executable["buffer_distribution_plans"] = plans
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="buffer_distribution_plans|source_buffer",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, drop_sharded_source_buffer
        )


def test_blackhole_t3_direct_runtime_rejects_missing_projected_reshard_record():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=2, grid_y=2)
    with target:
        artifact = lower(kernel, target=target)

    def drop_reshard_plans(executable):
        executable["reshard_plans"] = []
        return executable

    mutated_mod = _rebuild_direct_runtime_module_with_executable_mutator(
        artifact, drop_reshard_plans
    )

    a_torch = _bf16_matrix(64, 64)
    b_output = torch.zeros_like(a_torch)
    with pytest.raises(
        tvm.error.InternalError,
        match="missing projected reshard conversion",
    ):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_t3_direct_runtime_rejects_unsupported_projected_reshard():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=2, grid_y=2)
    with target:
        artifact = lower(kernel, target=target)

    def mutate(tt_program):
        reshard_plans = list(tt_program.reshard_plans)
        assert len(reshard_plans) == 1
        reshard_plans[0] = rebuild_tt_reshard_plan(
            reshard_plans[0],
            conversion_kind="reshard",
            admission_status="unsupported",
            unsupported_reason="forced unsupported reshard for runtime execution gate",
        )
        return rebuild_tt_program(tt_program, reshard_plans=reshard_plans)

    mutated_mod = _rebuild_direct_runtime_module_with_tt_program(
        artifact, tt_program_mutator=mutate
    )

    a_torch = _bf16_matrix(64, 64)
    b_output = torch.zeros_like(a_torch)
    with pytest.raises(
        tvm.error.InternalError,
        match="forced unsupported reshard for runtime execution gate",
    ):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_t3_build_rejects_reshard_source_region_mismatch():
    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=2, grid_y=2)
    with target:
        artifact = lower(kernel, target=target)

    def mutate_reshard_source_region(executable):
        plans = []
        for item in executable["reshard_plans"]:
            plan = {str(key): value for key, value in item.items()}
            if str(plan.get("target_value", "")) == "resident_tile":
                plan["source_region_shape"] = [16, 32]
            plans.append(plan)
        executable["reshard_plans"] = plans
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="source_region_shape.*buffer distribution",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, mutate_reshard_source_region
        )


def test_blackhole_t3_build_rejects_tensor_memory_config_distribution_mismatch():
    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=2, grid_y=2)
    with target:
        artifact = lower(kernel, target=target)

    def mutate_tensor_memory_config(executable):
        plans = []
        for item in executable["tensor_memory_config_plans"]:
            plan = {str(key): value for key, value in item.items()}
            if str(plan.get("subject", "")) == "resident_tile":
                plan["memory_layout"] = "HEIGHT_SHARDED"
                plan["shard_distribution_strategy"] = "height"
            plans.append(plan)
        executable["tensor_memory_config_plans"] = plans
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="tensor memory config.*buffer distribution",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, mutate_tensor_memory_config
        )


def test_blackhole_t3_build_rejects_missing_tensor_memory_config_records():
    target = Target("blackhole")
    kernel = _explicit_user_reshard_copy_kernel(grid_x=2, grid_y=2)
    with target:
        artifact = lower(kernel, target=target)

    def drop_tensor_memory_configs(executable):
        executable["tensor_memory_config_plans"] = []
        return executable

    with pytest.raises(
        tvm.error.InternalError,
        match="tensor memory config|source_memory_config_plan_index",
    ):
        _rebuild_direct_runtime_module_with_executable_mutator(
            artifact, drop_tensor_memory_configs
        )


def test_blackhole_direct_runtime_rejects_unadmitted_buffer_distribution_kind():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    with target:
        artifact = lower(kernel, target=target)

    def mutate(tt_program):
        distributions = []
        for plan in tt_program.buffer_distribution_plans:
            if str(plan.buffer) == "A":
                distributions.append(
                    rebuild_tt_buffer_distribution_plan(
                        plan,
                        distribution_kind="replicated",
                        logical_index_mapping="none",
                    )
                )
            else:
                distributions.append(plan)
        return rebuild_tt_program(tt_program, buffer_distribution_plans=distributions)

    mutated_mod = _rebuild_direct_runtime_module_with_tt_program(
        artifact, tt_program_mutator=mutate
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    with pytest.raises(
        tvm.error.InternalError,
        match="buffer distribution|replicated",
    ):
        mutated_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_rejects_noncompact_input_tensor_layout():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    with target:
        artifact = lower(kernel, target=target)

    base = torch.randn(32, 64, dtype=torch.bfloat16)
    a_torch = base[:, ::2]
    b_output = torch.zeros(32, 32, dtype=torch.bfloat16)
    assert not a_torch.is_contiguous()

    with pytest.raises(tvm.error.InternalError, match="compact row-major"):
        artifact.codegen_mod["main"](a_torch, b_output)


def test_blackhole_copy_direct_runtime_rejects_noncompact_output_tensor_layout():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    with target:
        artifact = lower(kernel, target=target)

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    base = torch.zeros(32, 64, dtype=torch.bfloat16)
    b_output = base[:, ::2]
    assert not b_output.is_contiguous()

    with pytest.raises(tvm.error.InternalError, match="compact row-major"):
        artifact.codegen_mod["main"](a_torch, b_output)


def test_blackhole_large_shape_copy_keeps_per_core_l1_small():
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    assert int(extract_blackhole_total_l1_bytes(device_main)) == 4096
    executable_spec = _extract_blackhole_executable_spec(artifact)
    cb_configs = executable_spec["cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 2
    planner_cb_configs = extract_blackhole_cb_configs(device_main)
    assert int(planner_cb_configs[0]["total_size_bytes"]) == 4096
    assert int(planner_cb_configs[0]["lifetime_begin"]) == 0
    assert int(planner_cb_configs[0]["lifetime_end"]) >= int(planner_cb_configs[0]["lifetime_begin"])


def test_blackhole_module_direct_call_large_shape_copy():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    m, n = 25 * 32, 32 * 32
    torch.manual_seed(42)
    a_torch = torch.randn(m, n, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)
    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        b_ref,
        atol=1e-3,
        rtol=1e-3,
        failure_message="Large-shape direct-call output mismatch",
    )


def test_blackhole_module_direct_call_accepts_richer_copy_schema_with_explicit_per_work_spec():
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    with target:
        artifact = lower(kernel, target=target)

    richer_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "identity": "input_buffer_addr32:A", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "identity": "output_buffer_addr32:B", "dtype": "uint32", "buffer": "B"},
        {"name": "work_linear_id", "kind": "work_linear_id", "identity": "work_linear_id", "dtype": "uint32"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "identity": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "identity": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "identity": "a_tile_stride", "dtype": "uint32"},
        {"name": "b_tile_start_id", "kind": "b_tile_start_id", "identity": "b_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "identity": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "identity": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "identity": "output_tile_stride", "dtype": "uint32"},
    ]

    def mutate(tt_program):
        abi_plans = [
            rebuild_tt_abi_plan(abi_plan, runtime_args=richer_runtime_args)
            for abi_plan in tt_program.abi_plans
        ]
        rebuilt_kernels = []
        for kernel in tt_program.kernels:
            per_work_arg_specs = tt_per_work_arg_specs_to_list(kernel.per_work_arg_specs)
            per_work_arg_specs.append(
                {
                    "arg_kind": "b_tile_start_id",
                    "arg_identity": "b_tile_start_id",
                    "descriptor_kind": "tile_start",
                    "value_source": "logical_block_x",
                }
            )
            rebuilt_kernels.append(
                rebuild_tt_kernel(kernel, per_work_arg_specs=per_work_arg_specs)
            )
        return rebuild_tt_program(tt_program, abi_plans=abi_plans, kernels=rebuilt_kernels)

    mutated_mod = _rebuild_direct_runtime_module_with_tt_program(
        artifact, tt_program_mutator=mutate
    )

    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_output = torch.zeros_like(a_torch)
    mutated_mod["main"](a_torch, b_output)
    assert_tensors_close_or_dump(
        b_output,
        a_torch,
        atol=0.0,
        rtol=0.0,
        failure_message="Richer copy-schema direct-call output mismatch",
    )
