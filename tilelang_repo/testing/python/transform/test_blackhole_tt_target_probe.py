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
    mod = tilelang.transform.LowerToSpatialProgram()(mod)
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
