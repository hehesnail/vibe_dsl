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
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(TOPK_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(TOPK_EXAMPLE_DIR))

from common import gemm_kernel, grid_indexed_staged_copy_kernel, staged_copy_kernel
import example_topk
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
            {},
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

    assert "pointwise_chain" in {str(item) for item in lowering_requirements["fragment_op_kinds"]}
    assert {"mul", "div"} <= {str(item) for item in lowering_requirements["pointwise_op_kinds"]}


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
