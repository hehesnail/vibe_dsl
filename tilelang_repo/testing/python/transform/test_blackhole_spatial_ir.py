import sys
from pathlib import Path

import pytest
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


def _drop_validated_spatial_plan_stamp(mod):
    func = mod["main"]
    if func.attrs and func.attrs.get("tl.spatial_plan_validated") is not None:
        func = func.without_attr("tl.spatial_plan_validated")
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _rebuild_dataflow_edge(
    edge,
    *,
    name=None,
    kind=None,
    producer_unit=None,
    consumer_unit=None,
    producer_unit_index=None,
    consumer_unit_index=None,
    subject=None,
    crosses_phase=None,
    traits=None,
    anchors=None,
):
    make_dataflow_edge = tvm.get_global_func("tl.DataflowEdge")
    return make_dataflow_edge(
        str(edge.name) if name is None else name,
        str(edge.kind) if kind is None else kind,
        str(edge.producer_unit) if producer_unit is None else producer_unit,
        str(edge.consumer_unit) if consumer_unit is None else consumer_unit,
        int(edge.producer_unit_index) if producer_unit_index is None else producer_unit_index,
        int(edge.consumer_unit_index) if consumer_unit_index is None else consumer_unit_index,
        str(edge.subject) if subject is None else subject,
        bool(edge.crosses_phase) if crosses_phase is None else crosses_phase,
        list(edge.traits) if traits is None else traits,
        list(edge.anchors) if anchors is None else anchors,
    )


def _rebuild_spatial_plan(
    plan,
    *,
    execution_units=None,
    dataflow_edges=None,
    layout_specs=None,
    phase_plans=None,
    validated_hints=None,
    closures=None,
    boundaries=None,
    anchors=None,
):
    make_spatial_plan = tvm.get_global_func("tl.SpatialPlan")
    return make_spatial_plan(
        str(plan.member_func),
        list(plan.execution_units) if execution_units is None else execution_units,
        list(plan.dataflow_edges) if dataflow_edges is None else dataflow_edges,
        list(plan.layout_specs) if layout_specs is None else layout_specs,
        list(plan.phase_plans) if phase_plans is None else phase_plans,
        plan.validated_hints if validated_hints is None else validated_hints,
        list(plan.closures) if closures is None else closures,
        list(plan.boundaries) if boundaries is None else boundaries,
        list(plan.anchors) if anchors is None else anchors,
    )


def test_spatial_pass_surface_exposes_only_structure_and_plan_companions():
    assert hasattr(tilelang.transform, "AnalyzeSpatialStructureFacts")
    assert hasattr(tilelang.transform, "BuildSpatialPlanCompanion")
    assert hasattr(tilelang.transform, "ValidateSpatialPlan")
    assert hasattr(tilelang.transform, "PlanTTBlocks")
    assert hasattr(tilelang.transform, "PlanTTCompute")
    assert hasattr(tilelang.transform, "PlanTTTransport")
    assert hasattr(tilelang.transform, "PlanTTSync")
    assert hasattr(tilelang.transform, "PlanTTABI")
    assert hasattr(tilelang.transform, "PlanTTExecution")
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
    assert main.attrs.get("tl.spatial_plan_validated")
    assert main.attrs.get("tl.spatial_domain_plan") is None
    assert main.attrs.get("tl.spatial_execution_plan") is None
    assert main.attrs.get("tl.spatial_program") is None
    assert mod.global_infos.get("tl.spatial_capability_model") is None

    execution_units = {str(unit.name): unit for unit in plan.execution_units}
    assert len(execution_units) == 2
    assert {str(unit.unit_role) for unit in execution_units.values()} == {
        "ingress",
        "egress",
    }

    flow_edges = [edge for edge in plan.dataflow_edges if str(edge.kind) == "flow"]
    assert len(flow_edges) == 1
    flow_edge = flow_edges[0]
    assert str(flow_edge.producer_unit) in execution_units
    assert str(flow_edge.consumer_unit) in execution_units
    assert str(flow_edge.subject) == "A_shared"
    assert bool(flow_edge.crosses_phase)

    layout_subjects = {str(layout.subject) for layout in plan.layout_specs}
    assert "A" in layout_subjects
    assert "A_shared" in layout_subjects

    assert len(plan.phase_plans) == 2
    assert {int(phase.phase_index) for phase in plan.phase_plans} == {0, 1}
    assert {
        str(subject)
        for phase in plan.phase_plans
        for subject in phase.boundary_subjects
    } == {"A_shared"}

    assert len(plan.closures) == len(plan.execution_units)
    assert len(plan.boundaries) == len(plan.dataflow_edges)


def test_task1_gemm_spatial_plan_emits_compute_closure():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]

    roles = {str(unit.unit_role) for unit in plan.execution_units}
    assert "compute" in roles
    assert "ingress" in roles
    assert "egress" in roles
    assert any(str(edge.kind) == "flow" for edge in plan.dataflow_edges)
    assert len(plan.layout_specs) >= 3
    assert len(plan.phase_plans) >= 1


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
    assert main.attrs.get("tl.spatial_plan_validated")
    assert main.attrs.get("blackhole.work_decomposition") is not None
    assert main.attrs.get("blackhole.compute_regions") is not None
    assert len(main.attrs["tl.spatial_plan"].phase_plans) >= 1
    assert main.attrs.get("tl.spatial_domain_plan") is None
    assert main.attrs.get("tl.spatial_execution_plan") is None
    assert main.attrs.get("tl.spatial_program") is None


def test_build_tt_program_consumes_plan_and_analysis_attrs_without_spatial_program():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_legacy_spatial_attrs(mod)
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)

    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert tt_program is not None
    assert len(tt_program.block_plans) == len(tt_program.core_groups)
    assert len(tt_program.kernel_plans) == len(tt_program.kernels)
    assert len(tt_program.sync_plans) == len(tt_program.compute_sync_plans)
    assert main.attrs.get("tl.internal_tt_block_plans") is None
    assert main.attrs.get("tl.internal_tt_kernel_plans") is None
    assert main.attrs.get("tl.internal_tt_sync_plans") is None
    assert main.attrs.get("tl.internal_tt_execution_plans") is None
    assert not tt_program.transport_plans or all(
        int(plan.source_task_index) >= 0 and int(plan.target_task_index) >= 0
        for plan in tt_program.transport_plans
    )


def test_task1_validate_spatial_plan_rejects_incomplete_dataflow_edge():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_edge = _rebuild_dataflow_edge(plan.dataflow_edges[0], subject="")
    invalid_plan = _rebuild_spatial_plan(
        plan,
        dataflow_edges=[invalid_edge, *list(plan.dataflow_edges[1:])],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr("tl.spatial_plan_validated")
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="DataflowEdge.*subject"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_task1_tt_planning_requires_validated_spatial_plan():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_validated_spatial_plan_stamp(mod)

    with pytest.raises(Exception, match="validated SpatialPlan"):
        tilelang.transform.PlanTTBlocks()(mod)


def test_build_tt_program_requires_explicit_tt_owner_plan_attrs():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_legacy_spatial_attrs(mod)
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)

    with pytest.raises(Exception, match="tl.internal_tt_sync_plans|PlanTTSync"):
        tilelang.transform.BuildTTProgram()(mod)
