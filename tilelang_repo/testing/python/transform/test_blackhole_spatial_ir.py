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


def test_copy_spatial_program_uses_single_transfer_fast_path():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    assert len(program.phases) == 1
    assert [str(task.name) for task in program.tasks] == ["copy"]
    assert [str(task.kind) for task in program.tasks] == ["transfer"]
    assert len(program.channels) == 1


def test_gemm_spatial_program_uses_reader_compute_writer_fast_path():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    program = mod["main"].attrs["tl.spatial_program"]

    assert [str(task.name) for task in program.tasks] == ["reader", "compute", "writer"]
    assert [str(task.kind) for task in program.tasks] == ["transfer", "compute", "transfer"]
    assert len(program.phases) == 1
    assert len(program.channels) >= 2


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
    assert str(program.work_partitions[0].kind) == "blocked"


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
    stateful_count = sum(
        1
        for state in semantic_program.states
        if str(state.role)
        in {"carry", "reduction_accumulator", "selection_state", "index_state"}
    )
    assert stateful_count > 1

    program = mod["main"].attrs["tl.spatial_program"]
    make_program = tvm.get_global_func("tl.SpatialProgram")
    seen_boundary_intent = False
    rebuilt_intents = []
    for intent in program.resource_intents:
        if str(intent.kind) == "phase_boundary_materialization":
            if seen_boundary_intent:
                continue
            seen_boundary_intent = True
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

    with pytest.raises(Exception, match="phase-boundary intent for each stateful semantic state"):
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
            {},
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
    expected_state_indices = [
        i
        for i, state in enumerate(semantic_program.states)
        if str(state.role) in {"carry", "reduction_accumulator", "selection_state", "index_state"}
    ]

    assert len(phase_boundary_intents) > 0
    assert all(str(intent.payload["target_kind"]) == "semantic_state" for intent in phase_boundary_intents)
    assert sorted(int(intent.payload["target_index"]) for intent in phase_boundary_intents) == expected_state_indices


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
            {"domain_index": partition.payload["domain_index"]},
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
    expected_state_names = [
        str(state.name)
        for state in semantic_program.states
        if str(state.role) in {"carry", "reduction_accumulator", "selection_state", "index_state"}
    ]
    program = mod["main"].attrs["tl.spatial_program"]

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
