import pytest
import torch

import tilelang
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
    rebuild_tt_abi_plan,
    rebuild_tt_kernel,
    rebuild_tt_core_group,
    rebuild_tt_program,
    rebuild_tt_semaphore_plan,
    require_tt_program,
    staged_copy_kernel,
    staged_stick_copy_kernel,
)
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
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
                "payload": dict(plan.get("payload", {})),
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
                    payload=plan["payload"],
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
                    plan["payload"],
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
            payload=dict(core_groups[0].payload),
        )
        return rebuild_tt_program(tt_program, core_groups=core_groups)

    return _rebuild_direct_runtime_module_with_tt_program(artifact, tt_program_mutator=mutate)


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
    return list(tt_program.abi_plans[0].runtime_args)


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
    per_work_arg_specs = {
        spec["arg_kind"]: spec["value_kind"]
        for spec in executable_spec["per_work_arg_specs"]
    }
    assert per_work_arg_specs["a_tile_start_id"] == "current_work_linear_id"
    assert per_work_arg_specs["output_tile_start_id"] == "current_work_linear_id"
    kernel_per_work_arg_specs = {
        spec["arg_kind"]: spec["value_kind"]
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
    assert int(planner_cb_configs[0]["lifetime_end"]) == 0


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
            payload = dict(kernel.payload)
            per_work_arg_specs = list(payload["per_work_arg_specs"])
            per_work_arg_specs.append(
                {
                    "arg_kind": "b_tile_start_id",
                    "arg_identity": "b_tile_start_id",
                    "value_kind": "logical_block_x",
                }
            )
            payload["per_work_arg_specs"] = per_work_arg_specs
            rebuilt_kernels.append(rebuild_tt_kernel(kernel, payload=payload))
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
