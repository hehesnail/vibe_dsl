import sys
from pathlib import Path

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

from common import gemm_kernel, grid_indexed_staged_copy_kernel, staged_copy_kernel
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
    mod = tvm.IRModule({"main": func})
    if "tl.device_programs" in mod.global_infos:
        mod = mod.with_attr("tl.device_programs", mod.global_infos["tl.device_programs"])
    return mod


def _drop_existing_spatial_program(mod):
    func = mod["main"].without_attr("tl.spatial_program")
    mod = tvm.IRModule({"main": func})
    if "tl.device_programs" in mod.global_infos:
        mod = mod.with_attr("tl.device_programs", mod.global_infos["tl.device_programs"])
    return mod


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
