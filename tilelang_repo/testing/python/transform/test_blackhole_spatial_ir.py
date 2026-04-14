import sys
from pathlib import Path

import tilelang
from tilelang import tvm
from tilelang.engine.phase import (
    LowerAndLegalize,
    LowerToBlackholePhaseB,
    OptimizeForTarget,
)
from tvm.target import Target

THIS_DIR = Path(__file__).resolve().parent
BLACKHOLE_TARGET_TEST_DIR = THIS_DIR.parent / "target" / "blackhole"
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import gemm_kernel, staged_copy_kernel
from test_blackhole_flash_attention_analysis import mha_example


def _prepare_blackhole_phase_b_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        if not (mod["main"].attrs and mod["main"].attrs.get("target") is not None):
            mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    mod = tilelang.transform.LowerDeviceStorageAccessInfo()(mod)
    mod = tilelang.transform.LowerIntrin()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tilelang.transform.HoistBroadcastValues()(mod)
    return LowerToBlackholePhaseB(mod)


def _drop_legacy_spatial_attrs(mod):
    func = mod["main"]
    for attr_name in (
        "tl.spatial_domain_plan",
        "tl.spatial_execution_plan",
        "tl.spatial_program",
    ):
        if func.attrs and func.attrs.get(attr_name) is not None:
            func = func.without_attr(attr_name)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def test_spatial_pass_surface_exposes_only_structure_and_plan_companions():
    assert hasattr(tilelang.transform, "AnalyzeSpatialStructureFacts")
    assert hasattr(tilelang.transform, "BuildSpatialPlanCompanion")
    assert not hasattr(tilelang.transform, "AnalyzeSpatialDomainPlan")
    assert not hasattr(tilelang.transform, "AnalyzeSpatialExecutionPlan")
    assert not hasattr(tilelang.transform, "MaterializeSpatialProgram")
    assert not hasattr(tilelang.transform, "LowerToSpatialProgram")
    assert not hasattr(tilelang.transform, "ValidateSpatialProgram")


def test_task1_copy_spatial_plan_emits_flow_boundary_from_tir():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]

    assert main.attrs.get("tl.spatial_structure_facts") is not None
    assert main.attrs.get("tl.spatial_domain_plan") is None
    assert main.attrs.get("tl.spatial_execution_plan") is None
    assert main.attrs.get("tl.spatial_program") is None
    assert mod.global_infos.get("tl.spatial_capability_model") is None

    closures = {str(closure.name): closure for closure in plan.closures}
    assert len(closures) == 2
    assert {str(closure.execution_role) for closure in closures.values()} == {
        "ingress",
        "egress",
    }

    flow_boundaries = [boundary for boundary in plan.boundaries if str(boundary.kind) == "flow"]
    assert len(flow_boundaries) == 1
    boundary = flow_boundaries[0]
    assert str(boundary.source_closure) in closures
    assert str(boundary.target_closure) in closures
    assert str(boundary.subject) == "A_shared"


def test_task1_gemm_spatial_plan_emits_compute_closure():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]

    roles = {str(closure.execution_role) for closure in plan.closures}
    assert "compute" in roles
    assert "ingress" in roles
    assert "egress" in roles
    assert any(str(boundary.kind) == "flow" for boundary in plan.boundaries)


def test_phase_b_pipeline_keeps_blackhole_analysis_attrs_without_spatial_program():
    mod = _prepare_blackhole_phase_b_module(
        mha_example.flashattn.jit_impl.get_tir(
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

    assert main.attrs.get("tl.spatial_plan") is not None
    assert main.attrs.get("blackhole.work_decomposition") is not None
    assert main.attrs.get("blackhole.compute_regions") is not None
    assert main.attrs.get("tl.spatial_domain_plan") is None
    assert main.attrs.get("tl.spatial_execution_plan") is None
    assert main.attrs.get("tl.spatial_program") is None


def test_build_tt_program_consumes_plan_and_analysis_attrs_without_spatial_program():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_legacy_spatial_attrs(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)

    tt_program = mod["main"].attrs["tl.tt_program"]
    assert tt_program is not None
    assert not tt_program.transport_plans or all(
        int(plan.source_task_index) >= 0 and int(plan.target_task_index) >= 0
        for plan in tt_program.transport_plans
    )
