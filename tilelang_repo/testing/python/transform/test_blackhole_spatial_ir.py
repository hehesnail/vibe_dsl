import sys
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tvm.target import Target

THIS_DIR = Path(__file__).resolve().parent
BLACKHOLE_TARGET_TEST_DIR = THIS_DIR.parent / "target" / "blackhole"
TOPK_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "topk"
GDN_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "gdn"
FUSEDMOE_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "fusedmoe"
DEEPSEEK_MLA_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "deepseek_mla"
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(TOPK_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(TOPK_EXAMPLE_DIR))
if str(GDN_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(GDN_EXAMPLE_DIR))
if str(FUSEDMOE_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(FUSEDMOE_EXAMPLE_DIR))
if str(DEEPSEEK_MLA_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(DEEPSEEK_MLA_EXAMPLE_DIR))

from common import gemm_kernel, grid_indexed_staged_copy_kernel, staged_copy_kernel
import example_topk
import example_chunk_o
import example_fusedmoe_tilelang
import example_mla_decode_paged
from test_blackhole_flash_attention_analysis import _lower_flash_attention_example, mha_example


def _prepare_blackhole_phase_b_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        if not (mod["main"].attrs and mod["main"].attrs.get("target") is not None):
            mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    mod = tilelang.transform.LowerDeviceStorageAccessInfo()(mod)
    mod = tilelang.transform.AugmentSemanticManifest()(mod)
    mod = tilelang.transform.LowerIntrin()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tilelang.transform.HoistBroadcastValues()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    mod = tilelang.transform.AnalyzeSemanticStructure()(mod)
    mod = tilelang.transform.LiftStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateSemanticRefinement()(mod)
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    return mod


def _strip_attr(mod, attr_name: str):
    func = mod["main"].without_attr(attr_name)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _drop_existing_spatial_program(mod):
    func = mod["main"].without_attr("tl.spatial_program")
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _replace_spatial_program(mod, program):
    func = mod["main"].with_attr("tl.spatial_program", program)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _replace_semantic_program(mod, program):
    func = mod["main"].with_attr("tl.semantic_program", program)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _replace_layout_payload(layout, payload):
    make_layout = tvm.get_global_func("tl.SpatialLayout")
    return make_layout(
        layout.name,
        layout.kind,
        layout.target_name,
        layout.axes,
        layout.traits,
        payload,
        layout.anchors,
    )


def _replace_partition_payload(partition, payload):
    make_partition = tvm.get_global_func("tl.WorkPartition")
    return make_partition(
        partition.name,
        partition.kind,
        partition.target_name,
        partition.axes,
        partition.traits,
        payload,
        partition.anchors,
    )


def _make_broadcast_fanout_semantic_program():
    make_domain = tvm.get_global_func("tl.Domain")
    make_state = tvm.get_global_func("tl.State")
    make_access_map = tvm.get_global_func("tl.AccessMap")
    make_update_law = tvm.get_global_func("tl.UpdateLaw")
    make_update = tvm.get_global_func("tl.Update")
    make_state_version = tvm.get_global_func("tl.StateVersion")
    make_state_def = tvm.get_global_func("tl.StateDef")
    make_state_use = tvm.get_global_func("tl.StateUse")
    make_program = tvm.get_global_func("tl.SemanticProgram")

    domain = make_domain("fanout_domain", ["bx"], [], [])
    shared_state = make_state("shared_value", "transient", "local", [])
    left_state = make_state("left_value", "transient", "local", [])
    right_state = make_state("right_value", "transient", "local", [])
    scaffold = make_update("root_map", "", make_update_law("map", "", [], []), [], [])
    producer = make_update(
        "produce_shared",
        "shared_value",
        make_update_law("map", "shared_value", [], []),
        [],
        [],
    )
    shared_access = [make_access_map("source_state", [], [])]
    consumer_left = make_update(
        "consume_left",
        "left_value",
        make_update_law("map", "left_value", ["shared_value"], shared_access),
        [],
        [],
    )
    consumer_right = make_update(
        "consume_right",
        "right_value",
        make_update_law("map", "right_value", ["shared_value"], shared_access),
        [],
        [],
    )
    produced_version = make_state_version(
        "produce_shared_out",
        "shared_value",
        "produce_shared",
        "update_result",
        [],
        [],
    )
    produced_def = make_state_def(
        "def_produce_shared",
        "shared_value",
        "produce_shared_out",
        "produce_shared",
        "update_result",
        [],
    )
    left_version = make_state_version(
        "consume_left_out",
        "left_value",
        "consume_left",
        "update_result",
        [],
        [],
    )
    right_version = make_state_version(
        "consume_right_out",
        "right_value",
        "consume_right",
        "update_result",
        [],
        [],
    )
    left_def = make_state_def(
        "def_consume_left",
        "left_value",
        "consume_left_out",
        "consume_left",
        "update_result",
        [],
    )
    right_def = make_state_def(
        "def_consume_right",
        "right_value",
        "consume_right_out",
        "consume_right",
        "update_result",
        [],
    )
    left_use = make_state_use(
        "use_consume_left",
        "consume_left",
        "shared_value",
        "produce_shared_out",
        "source_state",
        [],
    )
    right_use = make_state_use(
        "use_consume_right",
        "consume_right",
        "shared_value",
        "produce_shared_out",
        "source_state",
        [],
    )
    return make_program(
        [domain],
        [shared_state, left_state, right_state],
        [scaffold, producer, consumer_left, consumer_right],
        [],
        [],
        [],
        [produced_version, left_version, right_version],
        [produced_def, left_def, right_def],
        [left_use, right_use],
        [],
    )


def _make_multi_domain_semantic_program():
    make_domain = tvm.get_global_func("tl.Domain")
    make_update_law = tvm.get_global_func("tl.UpdateLaw")
    make_update = tvm.get_global_func("tl.Update")
    make_program = tvm.get_global_func("tl.SemanticProgram")

    base_domain = make_domain("base_domain", ["bx"], [], [])
    indexed_domain = make_domain("indexed_domain", ["tx"], ["derived_indices"], [])
    update = make_update(
        "multi_domain_compute",
        "",
        make_update_law("map", "", [], []),
        [],
        [],
    )

    return make_program(
        [base_domain, indexed_domain],
        [],
        [update],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )


def _make_routed_multi_domain_semantic_program():
    make_domain = tvm.get_global_func("tl.Domain")
    make_state = tvm.get_global_func("tl.State")
    make_access_map = tvm.get_global_func("tl.AccessMap")
    make_update_law = tvm.get_global_func("tl.UpdateLaw")
    make_update = tvm.get_global_func("tl.Update")
    make_program = tvm.get_global_func("tl.SemanticProgram")

    base_domain = make_domain("base_domain", ["bx"], [], [])
    routed_domain = make_domain("routed_domain", ["tx"], ["derived_indices"], [])
    selection_state = make_state("selection_gate", "selection_state", "local", [])
    routed_state = make_state("routed_value", "transient", "local", [])
    routed_access = [
        make_access_map(
            "source_state",
            [tvm.tir.Var("tx", "int32")],
            ["selected", "indexed"],
        )
    ]
    route_output = make_update(
        "route_output",
        "routed_value",
        make_update_law("map", "routed_value", ["selection_gate"], routed_access),
        [],
        [],
    )

    return make_program(
        [base_domain, routed_domain],
        [selection_state, routed_state],
        [route_output],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )


def _make_transitive_phase_chain_semantic_program():
    make_domain = tvm.get_global_func("tl.Domain")
    make_state = tvm.get_global_func("tl.State")
    make_access_map = tvm.get_global_func("tl.AccessMap")
    make_update_law = tvm.get_global_func("tl.UpdateLaw")
    make_update = tvm.get_global_func("tl.Update")
    make_state_version = tvm.get_global_func("tl.StateVersion")
    make_state_def = tvm.get_global_func("tl.StateDef")
    make_state_use = tvm.get_global_func("tl.StateUse")
    make_program = tvm.get_global_func("tl.SemanticProgram")

    domain = make_domain("phase_chain_domain", ["bx"], [], [])
    carry_a = make_state("carry_a", "carry", "local", [])
    carry_b = make_state("carry_b", "carry", "local", [])
    carry_c = make_state("carry_c", "carry", "local", [])
    source_access = [make_access_map("source_state", [tvm.tir.Var("bx", "int32")], [])]
    seed_a = make_update(
        "seed_a",
        "carry_a",
        make_update_law("recurrence", "carry_a", [], []),
        [],
        [],
    )
    step_b = make_update(
        "step_b",
        "carry_b",
        make_update_law("recurrence", "carry_b", ["carry_a"], source_access),
        [],
        [],
    )
    step_c = make_update(
        "step_c",
        "carry_c",
        make_update_law("recurrence", "carry_c", ["carry_b"], source_access),
        [],
        [],
    )
    seed_a_out = make_state_version("seed_a_out", "carry_a", "seed_a", "update_result", [], [])
    step_b_out = make_state_version("step_b_out", "carry_b", "step_b", "update_result", [], [])
    step_c_out = make_state_version("step_c_out", "carry_c", "step_c", "update_result", [], [])
    seed_a_def = make_state_def("def_seed_a", "carry_a", "seed_a_out", "seed_a", "update_result", [])
    step_b_def = make_state_def("def_step_b", "carry_b", "step_b_out", "step_b", "update_result", [])
    step_c_def = make_state_def("def_step_c", "carry_c", "step_c_out", "step_c", "update_result", [])
    use_step_c = make_state_use("use_step_c", "step_c", "carry_b", "step_b_out", "carried_state", [])
    use_step_b = make_state_use("use_step_b", "step_b", "carry_a", "seed_a_out", "carried_state", [])

    return make_program(
        [domain],
        [carry_a, carry_b, carry_c],
        [seed_a, step_b, step_c],
        [],
        [],
        [],
        [seed_a_out, step_b_out, step_c_out],
        [seed_a_def, step_b_def, step_c_def],
        [use_step_c, use_step_b],
        [],
    )


def _make_zero_state_gemm_semantic_program():
    make_domain = tvm.get_global_func("tl.Domain")
    make_update_law = tvm.get_global_func("tl.UpdateLaw")
    make_update = tvm.get_global_func("tl.Update")
    make_program = tvm.get_global_func("tl.SemanticProgram")

    domain = make_domain("gemm_domain", ["bx"], [], [])
    update = make_update(
        "gemm_compute",
        "",
        make_update_law("map", "", [], []),
        [],
        [],
    )

    return make_program(
        [domain],
        [],
        [update],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )


def _expected_phase_boundary_state_indices(program, semantic_program):
    stateful_state_indices = {
        str(state.name): i
        for i, state in enumerate(semantic_program.states)
        if str(state.role) in {"carry", "reduction_accumulator", "selection_state", "index_state"}
    }
    return sorted(
        {
            int(channel.payload["state_index"])
            for channel in program.channels
            if str(channel.payload.get("delivery_kind", "")) == "phase_boundary_materialized"
            and "state_index" in channel.payload
            and str(channel.state_name) in stateful_state_indices
        }
    )


def _make_spatial_program_like(program, **overrides):
    make_program = tvm.get_global_func("tl.SpatialProgram")
    return make_program(
        overrides.get("member_func", program.member_func),
        overrides.get("phases", program.phases),
        overrides.get("tasks", program.tasks),
        overrides.get("channels", program.channels),
        overrides.get("layouts", program.layouts),
        overrides.get("work_partitions", program.work_partitions),
        overrides.get("placements", program.placements),
        overrides.get("sync_edges", program.sync_edges),
        overrides.get("resource_intents", program.resource_intents),
        overrides.get("anchors", program.anchors),
    )


def _strip_attr_from_all_functions(mod, attr_name: str):
    rewritten = {}
    for gvar, func in mod.functions.items():
        if func.attrs and func.attrs.get(attr_name) is not None:
            func = func.without_attr(attr_name)
        rewritten[gvar] = func
    return tvm.IRModule(rewritten, global_infos=mod.global_infos)


def test_spatial_passes_are_registered():
    assert hasattr(tilelang.transform, "LowerToSpatialProgram")
    assert hasattr(tilelang.transform, "ValidateSpatialProgram")


def test_lower_to_spatial_program_publishes_spatial_capability_model_snapshot():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))

    capability_models = mod.global_infos["tl.spatial_capability_model"]
    assert len(capability_models) == 1
    capability = capability_models[0]

    assert str(capability.arch_name) == "BLACKHOLE"
    assert str(capability.topology_class) == "grid"
    assert str(capability.placement_domain) == "logical_worker_grid"
    assert int(capability.logical_worker_grid_x) == 11
    assert int(capability.logical_worker_grid_y) == 10
    assert int(capability.worker_l1_size) > 0
    assert int(capability.dram_view_size) > 0
    assert int(capability.functional_worker_count) > 0
    assert "point_to_point" in {str(kind) for kind in capability.supported_flow_kinds}
    assert "broadcast" in {str(kind) for kind in capability.supported_flow_kinds}
    assert "carry" in {str(kind) for kind in capability.supported_flow_kinds}
    assert "completion" in {str(kind) for kind in capability.supported_sync_kinds}
    assert "phase_boundary_materialization" in {
        str(kind) for kind in capability.supported_resource_intent_kinds
    }


def test_copy_spatial_program_uses_single_transfer_fast_path():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) == 1
    assert [str(task.name) for task in program.tasks] == ["copy"]
    assert [str(task.kind) for task in program.tasks] == ["transfer"]
    assert len(program.channels) == 1
    assert str(program.channels[0].kind) == "point_to_point"
    assert str(program.channels[0].payload["payload_kind"]) == "tensor"
    assert str(program.channels[0].payload["delivery_kind"]) == "buffered_async"


def test_gemm_spatial_program_uses_reader_compute_writer_fast_path():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    assert [str(task.name) for task in program.tasks] == ["reader", "compute", "writer"]
    assert [str(task.kind) for task in program.tasks] == ["transfer", "compute", "transfer"]
    assert len(program.phases) == 1
    assert len(program.channels) >= 2

    placements = {str(placement.task_name): placement for placement in program.placements}
    assert {task: str(placement.payload["affinity_kind"]) for task, placement in placements.items()} == {
        "reader": "ingress",
        "compute": "compute",
        "writer": "egress",
    }
    assert all(
        {"brisc", "trisc", "ncrisc"}.isdisjoint({str(trait) for trait in placement.traits})
        for placement in program.placements
    )

    channels = {str(channel.name): channel for channel in program.channels}
    assert str(channels["a_tiles"].kind) == "point_to_point"
    assert str(channels["a_tiles"].payload["payload_kind"]) == "tensor"
    assert str(channels["a_tiles"].payload["delivery_kind"]) == "buffered_async"
    assert str(channels["c_tiles"].kind) == "point_to_point"
    assert str(channels["c_tiles"].payload["payload_kind"]) == "state_version"
    assert str(channels["c_tiles"].payload["delivery_kind"]) == "completion_visible"


def test_flash_attention_spatial_program_exposes_multi_phase_channels():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    main = mod["main"]
    program = main.attrs["tl.spatial_program"]
    registry = mod.global_infos["tl.device_programs"]

    assert len(program.phases) >= 2
    assert len(program.channels) >= 1
    assert "phase_boundary_materialization" in {str(intent.kind) for intent in program.resource_intents}
    assert len(registry) == 1
    assert len(registry[0].phases) >= 2
    assert "phase_boundary_materialized" in {
        str(channel.payload["delivery_kind"]) for channel in program.channels
    }
    assert {"carry", "reduce_merge", "gather", "scatter"} & {
        str(channel.kind) for channel in program.channels
    }


def test_flash_attention_spatial_program_projects_pipeline_contract_resource_intent():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=2,
            threads=128,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    pipeline_contracts = [
        intent
        for intent in program.resource_intents
        if str(intent.kind) == "synchronization_support"
        and "pipeline_contract" in {str(trait) for trait in intent.traits}
    ]

    assert len(pipeline_contracts) == 1
    stage_records = pipeline_contracts[0].payload["pipeline_stages"]
    assert [int(stage["num_stages"]) for stage in stage_records] == [2, 2]
    assert [str(stage["loop_var"]) for stage in stage_records] == ["k", "k"]


def test_flash_attention_spatial_program_projects_fragment_contract_resource_intent():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    fragment_contracts = [
        intent
        for intent in program.resource_intents
        if str(intent.kind) == "lowering_support"
        and "fragment_contract" in {str(trait) for trait in intent.traits}
    ]

    assert len(fragment_contracts) == 1
    payload = fragment_contracts[0].payload
    assert "pointwise_chain" in {str(item) for item in payload["fragment_op_kinds"]}
    assert {"mul", "div"} <= {str(item) for item in payload["pointwise_op_kinds"]}
    assert len(payload["row_reduction_targets"]) > 0


def test_flash_attention_spatial_program_projects_work_dependent_bounds_into_work_partition_payload():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            True,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    partition = mod["main"].attrs["tl.spatial_program"].work_partitions[0]

    loop_bounds = partition.payload["work_dependent_loop_bounds"]
    assert len(loop_bounds) > 0
    assert any(str(bound["loop_var"]) == "k" for bound in loop_bounds)


def test_topk_spatial_program_exposes_selection_and_recurrence_family_gate():
    mod = _prepare_blackhole_phase_b_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) >= 2
    assert {"select", "recurrence"} <= {str(trait) for task in program.tasks for trait in task.traits}
    assert {"selection_state", "index_state"} <= {
        str(intent.traits[0]) for intent in program.resource_intents if len(intent.traits) > 0
    }
    assert len(program.channels) >= 3


def test_topk_spatial_program_tasks_project_execution_role_and_formation_basis():
    mod = _prepare_blackhole_phase_b_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert all("execution_role" in task.payload for task in program.tasks)
    assert all("formation_basis" in task.payload for task in program.tasks)
    assert all(str(task.payload["execution_role"]).strip() for task in program.tasks)
    assert all(str(task.payload["formation_basis"]).strip() for task in program.tasks)


def test_flash_attention_phase_contract_projects_closure_basis_and_graph_derived_truth():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) >= 2
    assert len(program.sync_edges) >= 1
    assert all("closure_basis" in phase.payload for phase in program.phases)
    assert all("ordering_kind" in edge.payload for edge in program.sync_edges)
    task_phase_index_by_task = {
        int(task_index): int(task.payload["phase_index"])
        for task_index, task in enumerate(program.tasks)
    }
    phase_task_indices = {
        int(phase.payload["phase_index"]): {int(task_index) for task_index in phase.payload["task_indices"]}
        for phase in program.phases
    }
    assert all(
        int(task_index) in phase_task_indices[phase_index]
        for task_index, phase_index in task_phase_index_by_task.items()
    )
    assert any(
        task_phase_index_by_task[int(edge.payload["source_task_index"])]
        != task_phase_index_by_task[int(edge.payload["target_task_index"])]
        for edge in program.sync_edges
    )


def test_topk_phase_order_projects_ordering_basis_contract():
    mod = _prepare_blackhole_phase_b_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    ordering_kinds = {str(edge.payload["ordering_kind"]) for edge in program.sync_edges}
    closure_basis = {str(phase.payload["closure_basis"]) for phase in program.phases}

    assert {"selection_index_handoff", "reduction_completion"} & ordering_kinds
    assert any("ordering_basis=" in basis for basis in closure_basis)


def test_chunk_o_spatial_program_exposes_chunk_recurrence_family_gate():
    mod = _prepare_blackhole_phase_b_module(
        example_chunk_o.tilelang_chunk_fwd_o.get_tir(
            B=1,
            S=64,
            H=1,
            DK=16,
            DV=16,
            input_dtype="float16",
            output_dtype="float16",
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=64,
            scale=1.0,
            use_g=True,
            block_S=64,
            block_DK=16,
            block_DV=16,
            threads=128,
            num_stages=1,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) >= 2
    assert {"select", "recurrence"} <= {
        str(trait) for task in program.tasks for trait in task.traits
    }
    assert len(program.channels) >= 1
    assert any(
        str(intent.kind) == "lowering_support"
        and "fragment_contract" in {str(trait) for trait in intent.traits}
        for intent in program.resource_intents
    )


def test_chunk_o_channels_project_source_version_and_target_version_contract():
    mod = _prepare_blackhole_phase_b_module(
        example_chunk_o.tilelang_chunk_fwd_o.get_tir(
            B=1,
            S=64,
            H=1,
            DK=16,
            DV=16,
            input_dtype="float16",
            output_dtype="float16",
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=64,
            scale=1.0,
            use_g=True,
            block_S=64,
            block_DK=16,
            block_DV=16,
            threads=128,
            num_stages=1,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert any(
        "source_version" in channel.payload and "target_version" in channel.payload
        for channel in program.channels
    )


def test_chunk_o_join_consumers_preserve_join_version_and_dependency_multiplicity():
    mod = _prepare_blackhole_phase_b_module(
        example_chunk_o.tilelang_chunk_fwd_o.get_tir(
            B=1,
            S=64,
            H=1,
            DK=16,
            DV=16,
            input_dtype="float16",
            output_dtype="float16",
            accum_dtype="float32",
            gate_dtype="float32",
            chunk_size=64,
            scale=1.0,
            use_g=True,
            block_S=64,
            block_DK=16,
            block_DV=16,
            threads=128,
            num_stages=1,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]
    semantic_program = mod["main"].attrs["tl.semantic_program"]

    join_output_versions = {str(join.output_version) for join in semantic_program.state_joins}
    expected_join_use_kinds = {}
    for use in semantic_program.state_uses:
        version_name = str(use.version_name)
        if version_name not in join_output_versions:
            continue
        key = (str(use.consumer_update), str(use.state_name), version_name)
        expected_join_use_kinds.setdefault(key, set()).add(str(use.kind))

    actual_channels_by_key = {}
    for channel in program.channels:
        source_version = str(channel.payload.get("source_version", ""))
        key = (str(channel.target_task), str(channel.state_name), source_version)
        actual_channels_by_key.setdefault(key, []).append(str(channel.name))

    assert any(len(use_kinds) > 1 for use_kinds in expected_join_use_kinds.values())
    for key, use_kinds in expected_join_use_kinds.items():
        assert len(actual_channels_by_key.get(key, [])) >= len(use_kinds)


def test_spatial_program_projects_broadcast_flow_for_multi_consumer_transient_state():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), _make_broadcast_fanout_semantic_program())
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    broadcast_channels = [channel for channel in program.channels if str(channel.kind) == "broadcast"]

    assert len(broadcast_channels) == 2
    assert {str(channel.target_task) for channel in broadcast_channels} == {
        "consume_left",
        "consume_right",
    }
    assert all(str(channel.source_task) == "produce_shared" for channel in broadcast_channels)


def test_fusedmoe_routed_spatial_program_exposes_routed_dispatch_family_gate():
    mod = _prepare_blackhole_phase_b_module(
        example_fusedmoe_tilelang.moe_forward_tilelang_routed.get_tir(
            64,
            32,
            4,
            "float16",
            64,
            4,
            block_token=32,
            block_dhidden=32,
            block_dexpert=32,
            threads=128,
            num_stages=1,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) >= 2
    assert {"select", "recurrence"} <= {
        str(trait) for task in program.tasks for trait in task.traits
    }
    assert len(program.channels) >= 1
    assert any(
        str(intent.kind) == "state_residency"
        and "selection_state" in {str(trait) for trait in intent.traits}
        for intent in program.resource_intents
    )


def test_paged_decode_spatial_program_exposes_paged_indexed_family_gate():
    mod = _prepare_blackhole_phase_b_module(
        example_mla_decode_paged.mla_decode_tilelang.get_tir(
            1,
            1,
            1,
            64,
            16,
            4,
            16,
            1,
            2,
            16,
            None,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) >= 2
    assert {"select", "recurrence"} <= {
        str(trait) for task in program.tasks for trait in task.traits
    }
    assert len(program.channels) >= 1
    assert any(
        str(intent.kind) == "state_residency"
        and "selection_state" in {str(trait) for trait in intent.traits}
        for intent in program.resource_intents
    )


def test_paged_decode_spatial_program_projects_domain_transform_kind_contract():
    mod = _prepare_blackhole_phase_b_module(
        example_mla_decode_paged.mla_decode_tilelang.get_tir(1, 1, 1, 64, 16, 4, 16, 1, 2, 16, None)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    assert any(
        str(layout.payload.get("domain_index")) == str(partition.payload.get("domain_index"))
        and str(layout.payload.get("domain_transform_kind")) == "paged"
        and str(partition.payload.get("partition_family")) == "paged"
        for layout in program.layouts
        for partition in program.work_partitions
    )


def test_spatial_program_materializes_layout_and_partition_for_each_semantic_domain():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), _make_multi_domain_semantic_program())
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    layout_by_domain_index = {int(layout.payload["domain_index"]): layout for layout in program.layouts}
    partition_by_domain_index = {
        int(partition.payload["domain_index"]): partition for partition in program.work_partitions
    }

    assert set(layout_by_domain_index) == {0, 1}
    assert set(partition_by_domain_index) == {0, 1}
    assert str(layout_by_domain_index[0].kind) == "regular"
    assert str(layout_by_domain_index[0].payload["domain_transform_kind"]) == "identity"
    assert str(layout_by_domain_index[1].kind) == "indexed"
    assert str(layout_by_domain_index[1].payload["domain_transform_kind"]) == "derived"
    assert str(partition_by_domain_index[0].kind) == "replicated"
    assert str(partition_by_domain_index[0].payload["partition_family"]) == "regular"
    assert str(partition_by_domain_index[1].kind) == "indexed"
    assert str(partition_by_domain_index[1].payload["partition_family"]) == "derived"


def test_spatial_program_derives_domain_contracts_per_semantic_domain():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), _make_routed_multi_domain_semantic_program())
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    layout_by_domain_index = {int(layout.payload["domain_index"]): layout for layout in program.layouts}
    partition_by_domain_index = {
        int(partition.payload["domain_index"]): partition for partition in program.work_partitions
    }

    assert str(layout_by_domain_index[0].payload["domain_transform_kind"]) == "identity"
    assert str(partition_by_domain_index[0].payload["partition_family"]) == "regular"
    assert str(layout_by_domain_index[1].payload["domain_transform_kind"]) == "routed"
    assert str(partition_by_domain_index[1].payload["partition_family"]) == "routed"


def test_spatial_program_layout_axes_come_from_semantic_program_not_work_decomposition():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=2, tile_cols=3))
    semantic_program = mod["main"].attrs["tl.semantic_program"]
    expected_axes = [str(axis) for axis in semantic_program.domains[0].axes]

    mod = _strip_attr(mod, "blackhole.work_decomposition")
    mod = _drop_existing_spatial_program(mod)
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    assert [str(axis) for axis in program.layouts[0].axes] == expected_axes
    assert [str(axis) for axis in program.work_partitions[0].axes] == expected_axes


def test_spatial_program_projects_domain_index_contracts_into_layout_and_partition():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=2, tile_cols=3))
    program = mod["main"].attrs["tl.spatial_program"]

    assert int(program.layouts[0].payload["domain_index"]) == 0
    assert int(program.work_partitions[0].payload["domain_index"]) == 0


def test_spatial_program_projects_task_phase_channel_and_placement_index_contracts():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    task = program.tasks[0]
    channel = program.channels[0]
    phase = program.phases[0]
    placement = program.placements[0]

    assert int(task.payload["phase_index"]) == 0
    assert int(channel.payload["source_task_index"]) == 0
    assert int(channel.payload["target_task_index"]) == 0
    assert [int(item) for item in phase.payload["task_indices"]] == [0]
    assert [int(item) for item in phase.payload["channel_indices"]] == [0]
    assert int(placement.payload["task_index"]) == 0


def test_spatial_program_projects_state_and_sync_index_contracts():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    c_tiles = next(channel for channel in program.channels if str(channel.name) == "c_tiles")
    compute_to_writer = next(edge for edge in program.sync_edges if str(edge.name) == "compute_to_writer")

    assert int(c_tiles.payload["state_index"]) == 0
    assert int(compute_to_writer.payload["source_task_index"]) == 1
    assert int(compute_to_writer.payload["target_task_index"]) == 2


def test_spatial_program_layout_kind_comes_from_semantic_domain_traits_not_work_decomposition():
    mod = _prepare_blackhole_phase_b_module(grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3))
    semantic_program = mod["main"].attrs["tl.semantic_program"]
    assert "derived_indices" in {str(trait) for trait in semantic_program.domains[0].traits}

    mod = _strip_attr(mod, "blackhole.work_decomposition")
    mod = _drop_existing_spatial_program(mod)
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    assert str(program.layouts[0].kind) == "indexed"
    assert str(program.work_partitions[0].kind) == "indexed"


def test_gemm_spatial_program_uses_segment_kind_ir_not_segment_plan():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _strip_attr(mod, "blackhole.segment_plan")
    mod = _drop_existing_spatial_program(mod)
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    assert [str(task.name) for task in program.tasks] == ["reader", "compute", "writer"]
    assert [str(task.kind) for task in program.tasks] == ["transfer", "compute", "transfer"]
    assert len(program.channels) >= 2


def test_spatial_program_builder_does_not_branch_on_root_map_update_name():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    semantic_program = mod["main"].attrs["tl.semantic_program"]
    scaffold_index = next(
        i
        for i, update in enumerate(semantic_program.updates)
        if str(update.law.kind) == "map"
        and str(update.state_name) == ""
        and str(update.law.target_state) == ""
        and len(update.law.source_states) == 0
    )

    make_update = tvm.get_global_func("tl.Update")
    make_program = tvm.get_global_func("tl.SemanticProgram")
    rebuilt_updates = list(semantic_program.updates)
    scaffold_update = rebuilt_updates[scaffold_index]
    rebuilt_updates[scaffold_index] = make_update(
        "renamed_scaffold_map",
        scaffold_update.state_name,
        scaffold_update.law,
        scaffold_update.anchors,
        scaffold_update.bindings,
    )
    rebuilt_program = make_program(
        list(semantic_program.domains),
        list(semantic_program.states),
        rebuilt_updates,
        list(semantic_program.supplements),
        list(semantic_program.seeds),
        list(semantic_program.anchors),
        list(semantic_program.state_versions),
        list(semantic_program.state_defs),
        list(semantic_program.state_uses),
        list(semantic_program.state_joins),
    )

    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), rebuilt_program)
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    rebuilt_task_names = [str(task.name) for task in mod["main"].attrs["tl.spatial_program"].tasks]

    assert "renamed_scaffold_map" in rebuilt_task_names
    assert "root_map" not in rebuilt_task_names


def test_validate_spatial_program_rejects_layout_axes_mismatch_with_semantic_domain():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=2, tile_cols=3))
    program = mod["main"].attrs["tl.spatial_program"]

    make_layout = tvm.get_global_func("tl.SpatialLayout")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    bad_layout = make_layout(
        program.layouts[0].name,
        program.layouts[0].kind,
        program.layouts[0].target_name,
        ["bogus_axis"],
        program.layouts[0].traits,
        program.layouts[0].payload,
        program.layouts[0].anchors,
    )
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        [bad_layout],
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="layout axes.*semantic domain"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_unknown_task_kind():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_task = tvm.get_global_func("tl.Task")
    rebuilt_tasks = [
        make_task(
            task.name,
            "bogus_task_kind",
            task.phase_name,
            task.update_names,
            task.traits,
            task.payload,
            task.anchors,
        )
        if i == 0
        else task
        for i, task in enumerate(program.tasks)
    ]
    mod = _replace_spatial_program(mod, _make_spatial_program_like(program, tasks=rebuilt_tasks))

    with pytest.raises(Exception, match="unknown task kind"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_task_without_execution_role_contract():
    mod = _prepare_blackhole_phase_b_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_task = tvm.get_global_func("tl.Task")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_tasks = [
        make_task(
            task.name,
            task.kind,
            task.phase_name,
            task.update_names,
            task.traits,
            {key: value for key, value in task.payload.items() if key != "execution_role"},
            task.anchors,
        )
        if i == 0
        else task
        for i, task in enumerate(program.tasks)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        rebuilt_tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="execution_role contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_task_without_formation_basis_contract():
    mod = _prepare_blackhole_phase_b_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_task = tvm.get_global_func("tl.Task")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_tasks = [
        make_task(
            task.name,
            task.kind,
            task.phase_name,
            task.update_names,
            task.traits,
            {key: value for key, value in task.payload.items() if key != "formation_basis"},
            task.anchors,
        )
        if i == 0
        else task
        for i, task in enumerate(program.tasks)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        rebuilt_tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="formation_basis contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_phase_without_closure_basis_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_phase = tvm.get_global_func("tl.ProgramPhase")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_phases = [
        make_phase(
            phase.name,
            phase.task_names,
            phase.channel_names,
            phase.traits,
            {key: value for key, value in phase.payload.items() if key != "closure_basis"},
            phase.anchors,
        )
        if i == 0
        else phase
        for i, phase in enumerate(program.phases)
    ]
    bad_program = make_program(
        program.member_func,
        rebuilt_phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="closure_basis contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_state_version_channel_with_semantic_linkage_mismatch():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_channel = tvm.get_global_func("tl.Channel")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_channels = [
        make_channel(
            channel.name,
            channel.kind,
            channel.source_task,
            channel.target_task,
            channel.state_name,
            channel.traits,
            {**dict(channel.payload), "source_version": "bogus_version"},
            channel.anchors,
        )
        if "state_index" in channel.payload
        else channel
        for channel in program.channels
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        rebuilt_channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="source_version inconsistent|unresolved source_version"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_layout_without_domain_transform_kind_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_layout = tvm.get_global_func("tl.SpatialLayout")
    rebuilt_layouts = [
        make_layout(
            layout.name,
            layout.kind,
            layout.target_name,
            layout.axes,
            layout.traits,
            {key: value for key, value in layout.payload.items() if key != "domain_transform_kind"},
            layout.anchors,
        )
        if i == 0
        else layout
        for i, layout in enumerate(program.layouts)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, layouts=rebuilt_layouts)
    )

    with pytest.raises(Exception, match="domain_transform_kind contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_missing_semantic_domain_layout_coverage():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), _make_multi_domain_semantic_program())
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    mod = _replace_spatial_program(
        mod,
        _make_spatial_program_like(program, layouts=[program.layouts[0]]),
    )

    with pytest.raises(Exception, match="layout contract to cover every semantic domain index"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_partition_without_partition_family_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_partition = tvm.get_global_func("tl.WorkPartition")
    rebuilt_partitions = [
        make_partition(
            partition.name,
            partition.kind,
            partition.target_name,
            partition.axes,
            partition.traits,
            {key: value for key, value in partition.payload.items() if key != "partition_family"},
            partition.anchors,
        )
        if i == 0
        else partition
        for i, partition in enumerate(program.work_partitions)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, work_partitions=rebuilt_partitions)
    )

    with pytest.raises(Exception, match="partition_family contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_domain_transform_kind_misaligned_with_semantic_truth():
    mod = _prepare_blackhole_phase_b_module(
        example_mla_decode_paged.mla_decode_tilelang.get_tir(1, 1, 1, 64, 16, 4, 16, 1, 2, 16, None)
    )
    program = mod["main"].attrs["tl.spatial_program"]

    rebuilt_layouts = [
        _replace_layout_payload(
            layout,
            {
                **dict(layout.payload),
                "domain_transform_kind": "derived",
            },
        )
        for layout in program.layouts
    ]
    rebuilt_partitions = [
        _replace_partition_payload(
            partition,
            {
                **dict(partition.payload),
                "partition_family": "derived",
            },
        )
        for partition in program.work_partitions
    ]
    mod = _replace_spatial_program(
        mod,
        _make_spatial_program_like(
            program,
            layouts=rebuilt_layouts,
            work_partitions=rebuilt_partitions,
        ),
    )

    with pytest.raises(Exception, match="domain_transform_kind inconsistent with semantic truth"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_unknown_channel_kind():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_channel = tvm.get_global_func("tl.Channel")
    rebuilt_channels = [
        make_channel(
            channel.name,
            "bogus_channel_kind",
            channel.source_task,
            channel.target_task,
            channel.state_name,
            channel.traits,
            channel.payload,
            channel.anchors,
        )
        if i == 0
        else channel
        for i, channel in enumerate(program.channels)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, channels=rebuilt_channels)
    )

    with pytest.raises(Exception, match="unknown channel kind"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_unknown_layout_and_partition_kinds():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=2, tile_cols=3))
    program = mod["main"].attrs["tl.spatial_program"]

    make_layout = tvm.get_global_func("tl.SpatialLayout")
    make_partition = tvm.get_global_func("tl.WorkPartition")
    rebuilt_layouts = [
        make_layout(
            layout.name,
            "bogus_layout_kind",
            layout.target_name,
            layout.axes,
            layout.traits,
            layout.payload,
            layout.anchors,
        )
        if i == 0
        else layout
        for i, layout in enumerate(program.layouts)
    ]
    rebuilt_partitions = [
        make_partition(
            partition.name,
            "bogus_partition_kind",
            partition.target_name,
            partition.axes,
            partition.traits,
            partition.payload,
            partition.anchors,
        )
        if i == 0
        else partition
        for i, partition in enumerate(program.work_partitions)
    ]

    bad_layout_mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, layouts=rebuilt_layouts)
    )
    with pytest.raises(Exception, match="unknown layout kind"):
        tilelang.transform.ValidateSpatialProgram()(bad_layout_mod)

    bad_partition_mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, work_partitions=rebuilt_partitions)
    )
    with pytest.raises(Exception, match="unknown work partition kind"):
        tilelang.transform.ValidateSpatialProgram()(bad_partition_mod)


def test_validate_spatial_program_rejects_unknown_placement_sync_and_resource_intent_kinds():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_placement = tvm.get_global_func("tl.Placement")
    make_sync_edge = tvm.get_global_func("tl.SyncEdge")
    make_resource_intent = tvm.get_global_func("tl.ResourceIntent")
    rebuilt_placements = [
        make_placement(
            placement.name,
            "bogus_placement_kind",
            placement.task_name,
            placement.member_func,
            placement.traits,
            placement.payload,
            placement.anchors,
        )
        if i == 0
        else placement
        for i, placement in enumerate(program.placements)
    ]
    rebuilt_edges = [
        make_sync_edge(
            edge.name,
            "bogus_sync_kind",
            edge.source,
            edge.target,
            edge.traits,
            edge.payload,
            edge.anchors,
        )
        if i == 0
        else edge
        for i, edge in enumerate(program.sync_edges)
    ]
    rebuilt_intents = [
        make_resource_intent(
            intent.name,
            "bogus_resource_intent_kind",
            intent.target_name,
            intent.traits,
            intent.payload,
            intent.anchors,
        )
        if i == 0
        else intent
        for i, intent in enumerate(program.resource_intents)
    ]

    bad_placement_mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, placements=rebuilt_placements)
    )
    with pytest.raises(Exception, match="unknown placement kind"):
        tilelang.transform.ValidateSpatialProgram()(bad_placement_mod)

    bad_sync_mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, sync_edges=rebuilt_edges)
    )
    with pytest.raises(Exception, match="unknown sync edge kind"):
        tilelang.transform.ValidateSpatialProgram()(bad_sync_mod)

    bad_intent_mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, resource_intents=rebuilt_intents)
    )
    with pytest.raises(Exception, match="unknown resource intent kind"):
        tilelang.transform.ValidateSpatialProgram()(bad_intent_mod)


def test_validate_spatial_program_rejects_multi_phase_program_without_channel_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_phase = tvm.get_global_func("tl.ProgramPhase")
    rebuilt_phases = []
    for i, phase in enumerate(program.phases):
        rebuilt_phases.append(
            make_phase(
                phase.name,
                phase.task_names,
                [] if i == 1 else phase.channel_names,
                phase.traits,
                (
                    {
                        "phase_index": phase.payload["phase_index"],
                        "task_indices": phase.payload["task_indices"],
                        "channel_indices": [],
                        "closure_basis": phase.payload["closure_basis"],
                    }
                    if i == 1
                    else phase.payload
                ),
                phase.anchors,
            )
        )
    make_program = tvm.get_global_func("tl.SpatialProgram")
    bad_program = make_program(
        program.member_func,
        rebuilt_phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="downstream multi-phase programs to reference at least one channel"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_insufficient_phase_boundary_materialization():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    semantic_program = mod["main"].attrs["tl.semantic_program"]

    program = mod["main"].attrs["tl.spatial_program"]
    expected_state_indices = set(_expected_phase_boundary_state_indices(program, semantic_program))
    assert len(expected_state_indices) > 0
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_intents = []
    for intent in program.resource_intents:
        if str(intent.kind) == "phase_boundary_materialization":
            continue
        rebuilt_intents.append(intent)

    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        rebuilt_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="phase-boundary resource intents|phase-boundary intents to cover every cross-phase materialized state index"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_task_without_phase_index_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_task = tvm.get_global_func("tl.Task")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_tasks = [
        make_task(
            task.name,
            task.kind,
            task.phase_name,
            task.update_names,
            task.traits,
            {key: value for key, value in task.payload.items() if key != "phase_index"},
            task.anchors,
        )
        if i == 0
        else task
        for i, task in enumerate(program.tasks)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        rebuilt_tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="tasks to carry phase_index contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_phase_without_task_index_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_phase = tvm.get_global_func("tl.ProgramPhase")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_phases = [
        make_phase(
            phase.name,
            phase.task_names,
            phase.channel_names,
            phase.traits,
            {"phase_index": phase.payload["phase_index"]},
            phase.anchors,
        )
        if i == 0
        else phase
        for i, phase in enumerate(program.phases)
    ]
    bad_program = make_program(
        program.member_func,
        rebuilt_phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="program phases to carry task_indices contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_channel_without_task_index_contract():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_channel = tvm.get_global_func("tl.Channel")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_channels = [
        make_channel(
            channel.name,
            channel.kind,
            channel.source_task,
            channel.target_task,
            channel.state_name,
            channel.traits,
            {},
            channel.anchors,
        )
        if i == 0
        else channel
        for i, channel in enumerate(program.channels)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        rebuilt_channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="channels to carry source_task_index/target_task_index contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_channel_without_payload_and_delivery_contract():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_channel = tvm.get_global_func("tl.Channel")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_channels = [
        make_channel(
            channel.name,
            channel.kind,
            channel.source_task,
            channel.target_task,
            channel.state_name,
            channel.traits,
            {
                key: value
                for key, value in channel.payload.items()
                if key not in {"payload_kind", "delivery_kind"}
            },
            channel.anchors,
        )
        if i == 0
        else channel
        for i, channel in enumerate(program.channels)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        rebuilt_channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="channels to carry payload_kind/delivery_kind contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_placement_without_task_index_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_placement = tvm.get_global_func("tl.Placement")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_placements = [
        make_placement(
            placement.name,
            placement.kind,
            placement.task_name,
            placement.member_func,
            placement.traits,
            {},
            placement.anchors,
        )
        if i == 0
        else placement
        for i, placement in enumerate(program.placements)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        rebuilt_placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="placements to carry task_index contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_placement_without_obligation_kind_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_placement = tvm.get_global_func("tl.Placement")
    rebuilt_placements = [
        make_placement(
            placement.name,
            placement.kind,
            placement.task_name,
            placement.member_func,
            placement.traits,
            {key: value for key, value in placement.payload.items() if key != "obligation_kind"},
            placement.anchors,
        )
        if i == 0
        else placement
        for i, placement in enumerate(program.placements)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, placements=rebuilt_placements)
    )

    with pytest.raises(Exception, match="obligation_kind contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_sync_edge_without_task_index_contract():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    make_sync_edge = tvm.get_global_func("tl.SyncEdge")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_edges = [
        make_sync_edge(
            edge.name,
            edge.kind,
            edge.source,
            edge.target,
            edge.traits,
            {},
            edge.anchors,
        )
        if i == 0
        else edge
        for i, edge in enumerate(program.sync_edges)
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        rebuilt_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="sync edges to carry source_task_index/target_task_index contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_sync_edge_without_ordering_kind_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_sync_edge = tvm.get_global_func("tl.SyncEdge")
    rebuilt_edges = [
        make_sync_edge(
            edge.name,
            edge.kind,
            edge.source,
            edge.target,
            edge.traits,
            {key: value for key, value in edge.payload.items() if key != "ordering_kind"},
            edge.anchors,
        )
        if i == 0
        else edge
        for i, edge in enumerate(program.sync_edges)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, sync_edges=rebuilt_edges)
    )

    with pytest.raises(Exception, match="ordering_kind contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_sync_edge_without_materialization_kind_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_sync_edge = tvm.get_global_func("tl.SyncEdge")
    rebuilt_edges = [
        make_sync_edge(
            edge.name,
            edge.kind,
            edge.source,
            edge.target,
            edge.traits,
            {key: value for key, value in edge.payload.items() if key != "materialization_kind"},
            edge.anchors,
        )
        if i == 0
        else edge
        for i, edge in enumerate(program.sync_edges)
    ]
    mod = _replace_spatial_program(
        mod, _make_spatial_program_like(program, sync_edges=rebuilt_edges)
    )

    with pytest.raises(Exception, match="materialization_kind contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_multi_phase_program_without_sync_edge_coverage():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]
    mod = _replace_spatial_program(mod, _make_spatial_program_like(program, sync_edges=[]))

    with pytest.raises(Exception, match="cross-phase channel coverage to materialize sync_edge contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_spatial_program_phase_boundary_intents_project_state_index_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]
    phase_boundary_intents = [
        intent for intent in program.resource_intents if str(intent.kind) == "phase_boundary_materialization"
    ]
    semantic_program = mod["main"].attrs["tl.semantic_program"]
    expected_state_indices = _expected_phase_boundary_state_indices(program, semantic_program)

    assert len(phase_boundary_intents) > 0
    assert all(str(intent.payload["target_kind"]) == "semantic_state" for intent in phase_boundary_intents)
    assert sorted(int(intent.payload["target_index"]) for intent in phase_boundary_intents) == expected_state_indices


def test_spatial_program_phase_synthesis_propagates_transitive_ordering_chain():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _replace_semantic_program(
        _drop_existing_spatial_program(mod), _make_transitive_phase_chain_semantic_program()
    )
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    phase_index_by_task = {str(task.name): int(task.payload["phase_index"]) for task in program.tasks}
    assert phase_index_by_task["seed_a"] < phase_index_by_task["step_b"] < phase_index_by_task["step_c"]


def test_gemm_fast_path_zero_state_program_omits_state_residency_intent():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _replace_semantic_program(_drop_existing_spatial_program(mod), _make_zero_state_gemm_semantic_program())
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    program = mod["main"].attrs["tl.spatial_program"]

    assert [str(task.name) for task in program.tasks] == ["reader", "compute", "writer"]
    assert all(str(intent.kind) != "state_residency" for intent in program.resource_intents)


def test_validate_spatial_program_rejects_pipeline_program_without_pipeline_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=2,
            threads=128,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_program = tvm.get_global_func("tl.SpatialProgram")
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        [
            intent
            for intent in program.resource_intents
            if "pipeline_contract" not in {str(trait) for trait in intent.traits}
        ],
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="pipeline programs to materialize at least one pipeline contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_fragment_program_without_fragment_contract():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_program = tvm.get_global_func("tl.SpatialProgram")
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        [
            intent
            for intent in program.resource_intents
            if "fragment_contract" not in {str(trait) for trait in intent.traits}
        ],
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="fragment programs to materialize at least one fragment contract"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_work_dependent_domain_without_partition_payload():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            True,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_partition = tvm.get_global_func("tl.WorkPartition")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_partitions = [
        make_partition(
            partition.name,
            partition.kind,
            partition.target_name,
            partition.axes,
            partition.traits,
            {
                "domain_index": partition.payload["domain_index"],
                "partition_family": partition.payload["partition_family"],
            },
            partition.anchors,
        )
        for partition in program.work_partitions
    ]
    bad_program = make_program(
        program.member_func,
        program.phases,
        program.tasks,
        program.channels,
        program.layouts,
        rebuilt_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="work-dependent domains to materialize work partition payload"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_registry_phase_signature_mismatch():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_phase = tvm.get_global_func("tl.ProgramPhase")
    make_task = tvm.get_global_func("tl.Task")
    make_program = tvm.get_global_func("tl.SpatialProgram")

    original_phase_name = str(program.phases[0].name)
    renamed_phase_name = original_phase_name + "_renamed"
    rebuilt_phases = []
    for phase in program.phases:
        rebuilt_phases.append(
            make_phase(
                renamed_phase_name if str(phase.name) == original_phase_name else phase.name,
                phase.task_names,
                phase.channel_names,
                phase.traits,
                phase.payload,
                phase.anchors,
            )
        )
    rebuilt_tasks = []
    for task in program.tasks:
        rebuilt_tasks.append(
            make_task(
                task.name,
                task.kind,
                renamed_phase_name if str(task.phase_name) == original_phase_name else task.phase_name,
                task.update_names,
                task.traits,
                task.payload,
                task.anchors,
            )
        )
    bad_program = make_program(
        program.member_func,
        rebuilt_phases,
        rebuilt_tasks,
        program.channels,
        program.layouts,
        program.work_partitions,
        program.placements,
        program.sync_edges,
        program.resource_intents,
        program.anchors,
    )
    mod = _replace_spatial_program(mod, bad_program)

    with pytest.raises(Exception, match="aggregated ProgramPhase truth to match member-local phase signatures"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_validate_spatial_program_rejects_registry_phase_closure_basis_mismatch():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_phase = tvm.get_global_func("tl.ProgramPhase")
    rebuilt_phases = [
        make_phase(
            phase.name,
            phase.task_names,
            phase.channel_names,
            phase.traits,
            (
                {
                    **dict(phase.payload),
                    "closure_basis": str(phase.payload["closure_basis"]) + "|mutated",
                }
                if i == 0
                else phase.payload
            ),
            phase.anchors,
        )
        for i, phase in enumerate(program.phases)
    ]
    mod = _replace_spatial_program(mod, _make_spatial_program_like(program, phases=rebuilt_phases))

    with pytest.raises(Exception, match="aggregated ProgramPhase truth to match member-local phase signatures"):
        tilelang.transform.ValidateSpatialProgram()(mod)


def test_lower_blackhole_ops_uses_spatial_program_work_axes_without_work_decomposition():
    mod = _prepare_blackhole_phase_b_module(grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3))
    mod = _strip_attr(mod, "blackhole.work_decomposition")
    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert list(lowering_requirements["work_axes"]) == ["bx", "by"]
    assert int(lowering_requirements["derived_index_expr_count"]) == 1


def test_lower_blackhole_ops_requires_spatial_program_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _strip_attr_from_all_functions(mod, "tl.spatial_program")

    with pytest.raises(Exception, match="requires tl.spatial_program"):
        tilelang.transform.LowerBlackholeOps()(mod)


def test_lower_blackhole_ops_recovers_fragment_requirements_without_fragment_regions():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    mod = _strip_attr(mod, "blackhole.fragment_regions")
    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert {"pointwise_chain", "gemm"} <= {
        str(item) for item in lowering_requirements["fragment_op_kinds"]
    }
    assert {"mul", "div"} <= {str(item) for item in lowering_requirements["pointwise_op_kinds"]}
    assert len(lowering_requirements["fragment_loop_carried_state"]) > 0


def test_lower_blackhole_ops_prefers_spatial_fragment_contract_over_fragment_regions():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    program = mod["main"].attrs["tl.spatial_program"]

    make_resource_intent = tvm.get_global_func("tl.ResourceIntent")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_intents = []
    for intent in program.resource_intents:
        if "fragment_contract" in {str(trait) for trait in intent.traits}:
            payload = dict(intent.payload)
            payload["pointwise_op_kinds"] = ["mul"]
            rebuilt_intents.append(
                make_resource_intent(
                    intent.name,
                    intent.kind,
                    intent.target_name,
                    intent.traits,
                    payload,
                    intent.anchors,
                )
            )
        else:
            rebuilt_intents.append(intent)
    mod = _replace_spatial_program(
        mod,
        make_program(
            program.member_func,
            program.phases,
            program.tasks,
            program.channels,
            program.layouts,
            program.work_partitions,
            program.placements,
            program.sync_edges,
            rebuilt_intents,
            program.anchors,
        ),
    )

    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert {str(item) for item in lowering_requirements["pointwise_op_kinds"]} == {"mul"}


def test_lower_blackhole_ops_uses_phase_boundary_state_index_contract_not_target_name():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
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
        )
    )
    semantic_program = mod["main"].attrs["tl.semantic_program"]
    program = mod["main"].attrs["tl.spatial_program"]
    expected_state_names = [
        str(semantic_program.states[state_index].name)
        for state_index in _expected_phase_boundary_state_indices(program, semantic_program)
    ]

    make_resource_intent = tvm.get_global_func("tl.ResourceIntent")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_intents = []
    for intent in program.resource_intents:
        if str(intent.kind) == "phase_boundary_materialization":
            payload = dict(intent.payload)
            rebuilt_intents.append(
                make_resource_intent(
                    intent.name,
                    intent.kind,
                    f"bogus_state_{int(payload['target_index'])}",
                    intent.traits,
                    payload,
                    intent.anchors,
                )
            )
        else:
            rebuilt_intents.append(intent)
    mod = _replace_spatial_program(
        mod,
        make_program(
            program.member_func,
            program.phases,
            program.tasks,
            program.channels,
            program.layouts,
            program.work_partitions,
            program.placements,
            program.sync_edges,
            rebuilt_intents,
            program.anchors,
        ),
    )

    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert [str(item) for item in lowering_requirements["spatial_phase_boundary_states"]] == expected_state_names


def test_lower_blackhole_ops_recovers_pipeline_requirements_without_pipeline_attr():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=2,
            threads=128,
        )
    )
    mod = _strip_attr(mod, "blackhole.pipeline_stages")
    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert [int(item) for item in lowering_requirements["pipeline_stage_counts"]] == [2, 2]
    assert [str(item) for item in lowering_requirements["pipeline_loop_vars"]] == ["k"]


def test_lower_blackhole_ops_recovers_work_dependent_loop_bound_count_without_work_decomposition():
    mod = _prepare_blackhole_phase_b_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            True,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    mod = _strip_attr(mod, "blackhole.work_decomposition")
    lowered = tilelang.transform.LowerBlackholeOps()(mod)["main"]
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert int(lowering_requirements["work_dependent_loop_bound_count"]) > 0
