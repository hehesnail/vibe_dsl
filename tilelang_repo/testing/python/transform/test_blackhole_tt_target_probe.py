import sys
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tvm.target import Target

THIS_DIR = Path(__file__).resolve().parent
BLACKHOLE_TARGET_TEST_DIR = THIS_DIR.parent / "target" / "blackhole"
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import gemm_kernel, staged_copy_kernel
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
    mod = tilelang.transform.AnalyzeSpatialDomainPlan()(mod)
    mod = tilelang.transform.AnalyzeSpatialExecutionPlan()(mod)
    mod = tilelang.transform.MaterializeSpatialProgram()(mod)
    mod = tilelang.transform.ValidateSpatialProgram()(mod)
    return mod


def _replace_spatial_program(mod, program):
    func = mod["main"].with_attr("tl.spatial_program", program)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def test_tt_target_probe_pass_is_registered():
    assert hasattr(tilelang.transform, "LowerSpatialProgramToTTTargetProbe")


def test_tt_target_probe_accepts_copy_and_publishes_hardware_snapshot():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    probed = tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)

    hardware_models = probed.global_infos["tl.tt_hardware_model"]
    assert len(hardware_models) == 1
    hardware = hardware_models[0]
    assert str(hardware.arch_name) == "BLACKHOLE"
    assert int(hardware.logical_worker_grid_x) == 11
    assert int(hardware.logical_worker_grid_y) == 10
    assert int(hardware.worker_l1_size) > 0


def test_tt_target_probe_accepts_gemm_fast_path():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_accepts_multi_phase_flash_attention_subset():
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
    tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_channel_delivery_contract():
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
            {key: value for key, value in channel.payload.items() if key != "delivery_kind"},
            channel.anchors,
        )
        if i == 0
        else channel
        for i, channel in enumerate(program.channels)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: channel .* delivery_kind"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_out_of_bounds_channel_task_index():
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
            {**dict(channel.payload), "source_task_index": len(program.tasks)},
            channel.anchors,
        )
        if i == 0
        else channel
        for i, channel in enumerate(program.channels)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: channel .* source_task_index out of bounds"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_state_version_source_version_contract():
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
            {key: value for key, value in channel.payload.items() if key != "source_version"},
            channel.anchors,
        )
        if "state_index" in channel.payload
        else channel
        for channel in program.channels
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: channel .* source_version"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_phase_closure_contract():
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
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: phase .* closure_basis"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_sync_edge_ordering_contract():
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
    make_program = tvm.get_global_func("tl.SpatialProgram")
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
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: sync edge .* ordering_kind"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_out_of_bounds_sync_edge_task_index():
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
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_edges = [
        make_sync_edge(
            edge.name,
            edge.kind,
            edge.source,
            edge.target,
            edge.traits,
            {**dict(edge.payload), "target_task_index": len(program.tasks)},
            edge.anchors,
        )
        if i == 0
        else edge
        for i, edge in enumerate(program.sync_edges)
    ]
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
            rebuilt_edges,
            program.resource_intents,
            program.anchors,
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: sync edge .* target_task_index out of bounds"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_cross_phase_sync_edge_coverage():
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
            [],
            program.resource_intents,
            program.anchors,
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: cross-phase channel coverage requires sync edge"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_tt_leaking_placement_affinity():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
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
            {
                **dict(placement.payload),
                "affinity_kind": "trisc",
            },
            placement.anchors,
        )
        if str(placement.task_name) == "compute"
        else placement
        for placement in program.placements
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="TT-leaking placement affinity"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_placement_domain_contract():
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
            {
                key: value
                for key, value in placement.payload.items()
                if key != "placement_domain"
            },
            placement.anchors,
        )
        if i == 0
        else placement
        for i, placement in enumerate(program.placements)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: placement .* placement_domain"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_layout_domain_index_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_layout = tvm.get_global_func("tl.SpatialLayout")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_layouts = [
        make_layout(
            layout.name,
            layout.kind,
            layout.target_name,
            layout.axes,
            layout.traits,
            {key: value for key, value in layout.payload.items() if key != "domain_index"},
            layout.anchors,
        )
        if i == 0
        else layout
        for i, layout in enumerate(program.layouts)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
            program.member_func,
            program.phases,
            program.tasks,
            program.channels,
            rebuilt_layouts,
            program.work_partitions,
            program.placements,
            program.sync_edges,
            program.resource_intents,
            program.anchors,
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: layout .* domain_index"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_layout_domain_transform_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    program = mod["main"].attrs["tl.spatial_program"]

    make_layout = tvm.get_global_func("tl.SpatialLayout")
    make_program = tvm.get_global_func("tl.SpatialProgram")
    rebuilt_layouts = [
        make_layout(
            layout.name,
            layout.kind,
            layout.target_name,
            layout.axes,
            layout.traits,
            {
                key: value
                for key, value in layout.payload.items()
                if key != "domain_transform_kind"
            },
            layout.anchors,
        )
        if i == 0
        else layout
        for i, layout in enumerate(program.layouts)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
            program.member_func,
            program.phases,
            program.tasks,
            program.channels,
            rebuilt_layouts,
            program.work_partitions,
            program.placements,
            program.sync_edges,
            program.resource_intents,
            program.anchors,
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: layout .* domain_transform_kind"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_partition_family_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
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
                key: value
                for key, value in partition.payload.items()
                if key != "partition_family"
            },
            partition.anchors,
        )
        if i == 0
        else partition
        for i, partition in enumerate(program.work_partitions)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: work partition .* partition_family"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)


def test_tt_target_probe_rejects_missing_partition_domain_index_contract():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
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
            {key: value for key, value in partition.payload.items() if key != "domain_index"},
            partition.anchors,
        )
        if i == 0
        else partition
        for i, partition in enumerate(program.work_partitions)
    ]
    mod = _replace_spatial_program(
        mod,
        make_program(
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
        ),
    )

    with pytest.raises(Exception, match="missing spatial contract: work partition .* domain_index"):
        tilelang.transform.LowerSpatialProgramToTTTargetProbe()(mod)
