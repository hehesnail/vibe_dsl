from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang import tvm
from tvm import tir
from tvm.target import Target

from .common import rebuild_tt_kernel, rebuild_tt_program, require_tt_program
from .test_blackhole_copy_pipeline import _rebuild_codegen_module_with_tt_program
from .test_blackhole_flash_attention_runtime import blackhole_mha_example
from .test_blackhole_t3_compute_runtime import (
    _lower_blackhole,
    _t3_elementwise_chain_kernel,
)


def _seq64_mha_artifact():
    kernel = blackhole_mha_example.flashattn.jit_impl.get_tir(
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
    target = Target("blackhole")
    with target:
        return lower(kernel, target=target)


def _validate_mutated_tt_program(artifact, tt_program_mutator):
    rewritten = {}
    for gvar, func in artifact.device_mod.functions.items():
        if func.attrs and "tl.tt_program" in func.attrs:
            func = func.with_attr("tl.tt_program", tt_program_mutator(require_tt_program(func)))
        rewritten[gvar] = func
    device_mod = tvm.IRModule(rewritten, global_infos=artifact.device_mod.global_infos)
    return tilelang.transform.ValidateTTProgram()(device_mod)


def _rebuild_tt_cb_plan(plan, *, data_format=None, requirement_indices=None):
    make_tt_cb_plan = tilelang.tvm.get_global_func("tl.TTCBPlan")
    return make_tt_cb_plan(
        str(plan.name),
        int(plan.cb_id),
        str(plan.resource_class),
        int(plan.num_pages),
        int(plan.page_size_bytes),
        str(plan.data_format) if data_format is None else data_format,
        int(plan.initial_reserve_pages),
        str(plan.flow_class),
        int(plan.publish_pages_per_event),
        int(plan.consume_pages_per_event),
        int(plan.lifetime_begin),
        int(plan.lifetime_end),
        list(plan.requirement_indices) if requirement_indices is None else requirement_indices,
    )


def _rebuild_exact_cb_allocation(plan, *, release_reason=None):
    make_allocation = tilelang.tvm.get_global_func("tl.TTExactCBAllocation")
    return make_allocation(
        str(plan.name),
        str(plan.virtual_value),
        int(plan.virtual_value_index),
        str(plan.cb_plan),
        int(plan.cb_plan_index),
        int(plan.physical_cb_id),
        int(plan.page_count),
        int(plan.release_program_point),
        str(plan.release_reason) if release_reason is None else release_reason,
    )


def _rebuild_exact_cb_release_event(event, *, reason=None):
    make_release = tilelang.tvm.get_global_func("tl.TTExactCBReleaseEvent")
    return make_release(
        str(event.name),
        str(event.allocation),
        int(event.allocation_index),
        str(event.cb_plan),
        int(event.cb_plan_index),
        int(event.program_point),
        int(event.page_count),
        str(event.reason) if reason is None else reason,
    )


def _rebuild_exact_cb_use_event(event, *, virtual_value=None, virtual_value_index=None):
    make_use_event = tilelang.tvm.get_global_func("tl.TTExactCBUseEvent")
    return make_use_event(
        str(event.name),
        str(event.virtual_value) if virtual_value is None else virtual_value,
        int(event.virtual_value_index) if virtual_value_index is None else virtual_value_index,
        str(event.consumer_kernel),
        str(event.consumer_event),
        str(event.operand_role),
        int(event.program_point),
        bool(event.requires_full_logical_tile),
        str(event.borrow_kind),
    )


def _rebuild_exact_cb_live_interval(interval, *, end_point=None):
    make_interval = tilelang.tvm.get_global_func("tl.TTExactCBLiveInterval")
    return make_interval(
        str(interval.name),
        str(interval.virtual_value),
        int(interval.virtual_value_index),
        int(interval.begin_point),
        int(interval.end_point) if end_point is None else end_point,
        bool(interval.live_in),
        bool(interval.live_out),
        bool(interval.loop_carried),
        str(interval.interference_class),
    )


def _append_compute_cb_event(tt_program, op_name, cb_id, pages):
    kernels = []
    for kernel in tt_program.kernels:
        body = getattr(kernel, "body", None)
        if str(kernel.kind) == "compute" and str(kernel.core_type) == "trisc":
            event = tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get(f"tl.blackhole.{op_name}"),
                    tir.IntImm("uint32", cb_id),
                    tir.IntImm("uint32", pages),
                )
            )
            body = tir.SeqStmt([body, event])
        kernels.append(rebuild_tt_kernel(kernel, body=body))
    return rebuild_tt_program(tt_program, kernels=kernels)


def _metadata_from_artifact(artifact):
    rebuilt = _rebuild_codegen_module_with_tt_program(artifact)
    return rebuilt.get_function_metadata("main")


@pytest.fixture(scope="module")
def seq64_artifact():
    return _seq64_mha_artifact()


def test_kernel_specs_carry_structured_cb_queue_events(seq64_artifact):
    metadata = _metadata_from_artifact(seq64_artifact)
    kernels = metadata["kernels"]
    assert kernels

    cb_ids = {int(config["cb_id"]) for config in metadata["cb_configs"]}
    compute_kernel = next(
        kernel
        for kernel in kernels
        if str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc"
    )
    compute_events = list(compute_kernel.get("queue_events", []))
    assert compute_events
    assert {str(event["kind"]) for event in compute_events} >= {
        "reserve_back",
        "push_back",
        "wait_front",
        "pop_front",
    }
    assert all(int(event["cb_id"]) in cb_ids for event in compute_events)
    assert all(int(event["pages"]) > 0 for event in compute_events)

    external_pushes = [
        event
        for kernel in kernels
        if not (str(kernel["kind"]) == "compute" and str(kernel["core_type"]) == "trisc")
        for event in kernel.get("queue_events", [])
        if str(event["kind"]) == "push_back"
    ]
    assert external_pushes


def test_structured_writer_queue_events_consume_all_output_pages():
    artifact = _lower_blackhole(
        _t3_elementwise_chain_kernel(
            grid_x=8,
            grid_y=4,
            strategy="block",
            tile_m=32,
            tile_n=64,
        )
    )
    metadata = _metadata_from_artifact(artifact)
    output_cb_ids = {
        int(config["cb_id"])
        for config in metadata["cb_configs"]
        if str(config["role"]) == "output"
    }
    assert output_cb_ids

    writer_kernel = next(
        kernel
        for kernel in metadata["kernels"]
        if str(kernel["kind"]) == "writer" and str(kernel["core_type"]) == "ncrisc"
    )
    wait_pages_by_cb = {cb_id: 0 for cb_id in output_cb_ids}
    pop_pages_by_cb = {cb_id: 0 for cb_id in output_cb_ids}
    for event in writer_kernel.get("queue_events", []):
        cb_id = int(event["cb_id"])
        if cb_id not in output_cb_ids:
            continue
        if str(event["kind"]) == "wait_front":
            wait_pages_by_cb[cb_id] += int(event["pages"])
        elif str(event["kind"]) == "pop_front":
            pop_pages_by_cb[cb_id] += int(event["pages"])

    assert any(wait_pages_by_cb.values())
    assert pop_pages_by_cb == wait_pages_by_cb


def test_runtime_does_not_recover_queue_events_from_kernel_body():
    source = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "target"
        / "rt_mod_blackhole.cc"
    ).read_text(encoding="utf-8")

    assert "MatchCBQueueEventCall" not in source
    assert "ExtractCBQueueEvents" not in source
    assert "BuildCBRequirementIndexRemap" not in source


def test_typed_tile_cb_verifier_rejects_duplicate_requirement_owner(seq64_artifact):
    def duplicate_requirement_owner(tt_program):
        cb_plans = list(tt_program.cb_plans)
        assert len(cb_plans) >= 2
        requirement = int(list(cb_plans[0].requirement_indices)[0])
        cb_plans[1] = _rebuild_tt_cb_plan(
            cb_plans[1],
            requirement_indices=list(cb_plans[1].requirement_indices) + [requirement],
        )
        return rebuild_tt_program(tt_program, cb_plans=cb_plans)

    with pytest.raises(tvm.TVMError, match="CB requirement index .* owned by multiple TTCBPlan"):
        _validate_mutated_tt_program(seq64_artifact, duplicate_requirement_owner)


def test_typed_tile_cb_verifier_rejects_exact_cb_data_format_mismatch(seq64_artifact):
    def corrupt_data_format(tt_program):
        allocations = list(tt_program.exact_cb_allocations)
        virtual_values = list(tt_program.exact_cb_virtual_values)
        cb_plans = list(tt_program.cb_plans)
        assert allocations
        allocation = allocations[0]
        virtual_value = virtual_values[int(allocation.virtual_value_index)]
        wrong_format = "Float32" if str(virtual_value.data_format) != "Float32" else "Float16_b"
        cb_index = int(allocation.cb_plan_index)
        cb_plans[cb_index] = _rebuild_tt_cb_plan(cb_plans[cb_index], data_format=wrong_format)
        return rebuild_tt_program(tt_program, cb_plans=cb_plans)

    with pytest.raises(tvm.TVMError, match="exact-CB allocation data_format"):
        _validate_mutated_tt_program(seq64_artifact, corrupt_data_format)


def test_typed_tile_cb_verifier_rejects_unknown_exact_cb_release_reason(seq64_artifact):
    def corrupt_release_reason(tt_program):
        allocations = list(tt_program.exact_cb_allocations)
        releases = list(tt_program.exact_cb_release_events)
        assert releases
        release_index = 0
        allocation_index = int(releases[release_index].allocation_index)
        allocations[allocation_index] = _rebuild_exact_cb_allocation(
            allocations[allocation_index], release_reason="source_local_guess"
        )
        releases[release_index] = _rebuild_exact_cb_release_event(
            releases[release_index], reason="source_local_guess"
        )
        return rebuild_tt_program(
            tt_program,
            exact_cb_allocations=allocations,
            exact_cb_release_events=releases,
        )

    with pytest.raises(tvm.TVMError, match="exact-CB release reason"):
        _validate_mutated_tt_program(seq64_artifact, corrupt_release_reason)


def test_typed_tile_cb_verifier_rejects_stale_exact_cb_producer(seq64_artifact):
    def bind_stale_producer(tt_program):
        virtual_values = list(tt_program.exact_cb_virtual_values)
        uses = list(tt_program.exact_cb_use_events)
        intervals = list(tt_program.exact_cb_live_intervals)
        interval_by_index = {
            int(interval.virtual_value_index): (interval_index, interval)
            for interval_index, interval in enumerate(intervals)
        }
        by_logical = {}
        for index, value in enumerate(virtual_values):
            by_logical.setdefault(str(value.logical_value), []).append((index, value))
        for use_index, event in enumerate(uses):
            selected_index = int(event.virtual_value_index)
            selected_begin = int(interval_by_index[selected_index][1].begin_point)
            candidates = [
                (index, value)
                for index, value in by_logical[str(virtual_values[selected_index].logical_value)]
                if index != selected_index
                and int(interval_by_index[index][1].begin_point) < selected_begin
                and int(interval_by_index[index][1].begin_point) <= int(event.program_point)
            ]
            if not candidates:
                continue
            stale_index, stale_value = max(
                candidates, key=lambda item: int(interval_by_index[item[0]][1].begin_point)
            )
            uses[use_index] = _rebuild_exact_cb_use_event(
                event,
                virtual_value=str(stale_value.name),
                virtual_value_index=stale_index,
            )
            stale_interval_index, stale_interval = interval_by_index[stale_index]
            intervals[stale_interval_index] = _rebuild_exact_cb_live_interval(
                stale_interval,
                end_point=max(
                    int(stale_interval.end_point),
                    int(event.program_point),
                ),
            )
            return rebuild_tt_program(
                tt_program,
                exact_cb_use_events=uses,
                exact_cb_live_intervals=intervals,
            )
        pytest.fail("Expected at least one exact-CB logical value with multiple producers")

    with pytest.raises(tvm.TVMError, match="latest exact-CB producer"):
        _validate_mutated_tt_program(seq64_artifact, bind_stale_producer)


@pytest.mark.parametrize("op_name", ["cb_wait_front", "cb_pop_front", "cb_reserve_back"])
def test_typed_tile_cb_verifier_rejects_invalid_compute_queue_event(seq64_artifact, op_name):
    def corrupt_compute_queue(tt_program):
        cb_plan = next(
            plan
            for plan in tt_program.cb_plans
            if str(plan.resource_class) != "input"
        )
        return _append_compute_cb_event(
            tt_program,
            op_name,
            int(cb_plan.cb_id),
            int(cb_plan.num_pages) + 1,
        )

    with pytest.raises(tvm.TVMError, match="physical CB queue"):
        _rebuild_codegen_module_with_tt_program(
            seq64_artifact,
            tt_program_mutator=corrupt_compute_queue,
        )
