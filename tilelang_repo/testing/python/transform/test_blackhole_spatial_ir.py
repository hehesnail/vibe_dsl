import subprocess
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
REPO_ROOT = THIS_DIR.parents[3]
BLACKHOLE_TARGET_TEST_DIR = THIS_DIR.parent / "target" / "blackhole"
EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "flash_attention"
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

import example_gqa_fwd_bshd as gqa_example
import example_mha_fwd_bshd as mha_example
from common import (
    fragment_fill_cast_publish_kernel,
    gemm_kernel,
    lower_blackhole_to_tt_target,
    staged_copy_kernel,
)


HELPER_COMPOSITE_BLACKHOLE_BUILTINS = (
    "reduce_row",
    "mul_row_bcast",
    "mul_grouped_row_bcast",
    "div_row_bcast",
    "div_grouped_row_bcast",
    "exp2_row_bcast_affine",
    "exp2_grouped_row_bcast_affine",
    "scalar_max",
    "scalar_exp2_affine",
    "copy_tile_from_cb",
)

LEGACY_LOCAL_BLACKHOLE_BUILTINS = (
    "binary_max_tile_local",
    "reduce_rows_local",
    "mul_tiles_bcast_rows_local",
    "div_tiles_bcast_rows_local",
    "exp_tiles_bcast_rows_affine_local",
    "exp_tile_affine_local",
)


def _collect_blackhole_builtin_names(node):
    names = set()

    def visit(expr):
        if isinstance(expr, tvm.tir.Call):
            op = expr.op
            if hasattr(op, "name") and op.name.startswith("tl.blackhole."):
                names.add(op.name)

    body = node.body if hasattr(node, "body") else node
    tvm.tir.stmt_functor.post_order_visit(body, visit)
    return names


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


def _prepare_blackhole_tt_program_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        if not (mod["main"].attrs and mod["main"].attrs.get("target") is not None):
            mod = LowerAndLegalize(mod, target)
        mod = lower_blackhole_to_tt_target(mod)
    return mod


def _prepare_blackhole_builtin_selection_module(prim_func):
    mod = _prepare_blackhole_phase_b_module(prim_func)
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    return mod


def _source_tree_rg(pattern, *paths):
    result = subprocess.run(
        ["rg", "-n", pattern, *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


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


def _rebuild_live_value_edge(
    edge,
    *,
    name=None,
    source_live_value=None,
    source_live_value_index=None,
    dataflow_edge=None,
    dataflow_edge_index=None,
    producer_unit=None,
    consumer_unit=None,
    producer_unit_index=None,
    consumer_unit_index=None,
    relation_kind=None,
    requires_full_logical_value=None,
    accepts_distributed_slice=None,
    anchors=None,
):
    make_live_value_edge = tvm.get_global_func("tl.LiveValueEdge")
    return make_live_value_edge(
        str(edge.name) if name is None else name,
        str(edge.source_live_value)
        if source_live_value is None
        else source_live_value,
        int(edge.source_live_value_index)
        if source_live_value_index is None
        else source_live_value_index,
        str(edge.dataflow_edge) if dataflow_edge is None else dataflow_edge,
        int(edge.dataflow_edge_index)
        if dataflow_edge_index is None
        else dataflow_edge_index,
        str(edge.producer_unit) if producer_unit is None else producer_unit,
        str(edge.consumer_unit) if consumer_unit is None else consumer_unit,
        int(edge.producer_unit_index)
        if producer_unit_index is None
        else producer_unit_index,
        int(edge.consumer_unit_index)
        if consumer_unit_index is None
        else consumer_unit_index,
        str(edge.relation_kind) if relation_kind is None else relation_kind,
        bool(edge.requires_full_logical_value)
        if requires_full_logical_value is None
        else requires_full_logical_value,
        bool(edge.accepts_distributed_slice)
        if accepts_distributed_slice is None
        else accepts_distributed_slice,
        list(edge.anchors) if anchors is None else anchors,
    )


def _rebuild_materialization_boundary(
    boundary,
    *,
    name=None,
    source_live_value=None,
    source_live_value_index=None,
    target_live_value=None,
    target_live_value_index=None,
    live_value_edge=None,
    live_value_edge_index=None,
    required_visibility=None,
    logical_coverage=None,
    phase_relation=None,
    anchors=None,
):
    make_materialization_boundary = tvm.get_global_func("tl.MaterializationBoundary")
    return make_materialization_boundary(
        str(boundary.name) if name is None else name,
        str(boundary.source_live_value)
        if source_live_value is None
        else source_live_value,
        int(boundary.source_live_value_index)
        if source_live_value_index is None
        else source_live_value_index,
        str(boundary.target_live_value)
        if target_live_value is None
        else target_live_value,
        int(boundary.target_live_value_index)
        if target_live_value_index is None
        else target_live_value_index,
        str(boundary.live_value_edge) if live_value_edge is None else live_value_edge,
        int(boundary.live_value_edge_index)
        if live_value_edge_index is None
        else live_value_edge_index,
        str(boundary.required_visibility)
        if required_visibility is None
        else required_visibility,
        str(boundary.logical_coverage)
        if logical_coverage is None
        else logical_coverage,
        str(boundary.phase_relation) if phase_relation is None else phase_relation,
        list(boundary.anchors) if anchors is None else anchors,
    )


def _rebuild_spatial_plan(
    plan,
    *,
    execution_units=None,
    dataflow_edges=None,
    layout_specs=None,
    phase_plans=None,
    live_values=None,
    live_value_edges=None,
    materialization_boundaries=None,
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
        list(plan.live_values) if live_values is None else live_values,
        list(plan.live_value_edges)
        if live_value_edges is None
        else live_value_edges,
        list(plan.materialization_boundaries)
        if materialization_boundaries is None
        else materialization_boundaries,
        plan.validated_hints if validated_hints is None else validated_hints,
        list(plan.closures) if closures is None else closures,
        list(plan.boundaries) if boundaries is None else boundaries,
        list(plan.anchors) if anchors is None else anchors,
    )


def _rebuild_tt_program(
    program,
    *,
    mesh_plans=None,
    buffer_distribution_plans=None,
    compute_op_plans=None,
    cb_plans=None,
    live_form_plans=None,
    materialization_plans=None,
    consumer_binding_plans=None,
    payload=None,
):
    make_tt_program = tvm.get_global_func("tl.TTProgram")
    return make_tt_program(
        program.entry_name,
        program.member_func,
        list(program.mesh_plans) if mesh_plans is None else mesh_plans,
        list(program.buffer_distribution_plans)
        if buffer_distribution_plans is None
        else buffer_distribution_plans,
        list(program.block_plans),
        list(program.kernel_plans),
        list(program.compute_op_plans) if compute_op_plans is None else compute_op_plans,
        list(program.transport_plans),
        list(program.sync_plans),
        list(program.abi_plans),
        list(program.execution_plans),
        list(program.kernels),
        list(program.core_groups),
        list(program.cb_plans) if cb_plans is None else cb_plans,
        list(program.semaphore_plans),
        list(program.compute_sync_plans),
        list(program.dst_layout_plans),
        list(program.live_form_plans) if live_form_plans is None else live_form_plans,
        list(program.materialization_plans)
        if materialization_plans is None
        else materialization_plans,
        list(program.consumer_binding_plans)
        if consumer_binding_plans is None
        else consumer_binding_plans,
        program.payload if payload is None else payload,
    )


def test_spatial_pass_surface_exposes_only_direct_spatial_plan_builder():
    assert hasattr(tilelang.transform, "BuildSpatialPlan")
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
    assert not hasattr(tilelang.transform, "AnalyzeSpatialStructureFacts")
    assert not hasattr(tilelang.transform, "BuildSpatialPlanCompanion")
    assert not hasattr(tilelang.transform, "AnalyzeBlackholeWorkDecomposition")
    assert not hasattr(tilelang.transform, "AnalyzeBlackholeComputeRegions")
    assert not hasattr(tilelang.transform, "AnalyzeBlackholePipelineStages")


def test_task5_source_tree_has_no_internal_tt_or_resource_plan_definition_surface():
    hits = _source_tree_rg(
        r"blackhole\.resource_plan|tl\.internal_tt_",
        REPO_ROOT / "tilelang_repo/src",
        REPO_ROOT / "tilelang_repo/tilelang",
    )
    assert hits == []


def test_task1_copy_spatial_plan_emits_flow_boundary_from_tir():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]

    assert main.attrs.get("tl.spatial_structure_facts") is None
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


def test_task1_spatial_plan_exposes_logical_live_value_boundaries():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    plan = mod["main"].attrs["tl.spatial_plan"]

    flow_edges = [edge for edge in plan.dataflow_edges if str(edge.kind) == "flow"]
    assert len(flow_edges) == 1
    flow_edge = flow_edges[0]

    live_values = {str(value.name): value for value in plan.live_values}
    live_edges = {str(edge.dataflow_edge): edge for edge in plan.live_value_edges}
    boundaries = {
        str(boundary.live_value_edge): boundary
        for boundary in plan.materialization_boundaries
    }

    assert str(flow_edge.name) in live_edges
    live_edge = live_edges[str(flow_edge.name)]
    assert str(live_edge.relation_kind) == "flow"
    assert bool(live_edge.requires_full_logical_value)
    assert str(live_edge.source_live_value) in live_values

    live_value = live_values[str(live_edge.source_live_value)]
    assert str(live_value.subject) == "A_shared"
    assert str(live_value.value_role) == "fragment"
    assert str(live_value.producer_unit) == str(flow_edge.producer_unit)
    assert int(live_value.producer_unit_index) == int(flow_edge.producer_unit_index)
    assert tuple(int(dim) for dim in live_value.logical_shape)
    assert str(live_value.dtype)

    assert str(live_edge.name) in boundaries
    boundary = boundaries[str(live_edge.name)]
    assert str(boundary.source_live_value) == str(live_value.name)
    assert str(boundary.required_visibility) == "next_phase"
    assert str(boundary.logical_coverage) == "full_logical_value"
    assert str(boundary.phase_relation) == "cross_phase"


def test_task1_spatial_plan_materialization_boundary_names_target_live_value():
    mod = _prepare_blackhole_phase_b_module(fragment_fill_cast_publish_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]

    live_values = {str(value.name): value for value in plan.live_values}
    live_value_by_subject = {str(value.subject): value for value in plan.live_values}
    assert {"C_local", "D_local"}.issubset(live_value_by_subject)

    c_live = live_value_by_subject["C_local"]
    d_live = live_value_by_subject["D_local"]
    c_to_d_boundaries = [
        boundary
        for boundary in plan.materialization_boundaries
        if str(boundary.source_live_value) == str(c_live.name)
        and str(boundary.target_live_value) == str(d_live.name)
    ]
    assert len(c_to_d_boundaries) == 1
    boundary = c_to_d_boundaries[0]
    assert int(boundary.source_live_value_index) == list(plan.live_values).index(c_live)
    assert int(boundary.target_live_value_index) == list(plan.live_values).index(d_live)
    assert str(boundary.required_visibility) == "same_unit"
    assert str(boundary.logical_coverage) == "distributed_slice"
    assert str(boundary.phase_relation) == "same_phase"

    live_edge = plan.live_value_edges[int(boundary.live_value_edge_index)]
    assert str(live_edge.name) == str(boundary.live_value_edge)
    assert str(live_edge.source_live_value) in live_values
    assert str(live_edge.relation_kind) == "materialize"
    assert bool(live_edge.accepts_distributed_slice) is True
    assert bool(live_edge.requires_full_logical_value) is False


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


def test_phase_b_pipeline_exposes_only_spatial_plan_without_legacy_analysis_attrs():
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
    assert len(main.attrs["tl.spatial_plan"].phase_plans) >= 1
    assert main.attrs.get("tl.spatial_domain_plan") is None
    assert main.attrs.get("tl.spatial_execution_plan") is None
    assert main.attrs.get("tl.spatial_program") is None
    assert main.attrs.get("blackhole.work_decomposition") is None
    assert main.attrs.get("blackhole.compute_regions") is None
    assert main.attrs.get("blackhole.pipeline_stages") is None


def test_flash_attention_spatial_plan_keeps_local_state_and_shared_layouts():
    mod = _prepare_blackhole_phase_b_module(
        gqa_example.flashattn.jit_impl.get_tir(
            1,
            16,
            1024,
            128,
            False,
            groups=16,
            block_M=64,
            block_N=64,
            num_stages=4,
            threads=128,
        )
    )
    plan = mod["main"].attrs["tl.spatial_plan"]

    compute_units = [unit for unit in plan.execution_units if str(unit.unit_role) == "compute"]
    assert len(compute_units) >= 1
    read_buffers = {
        str(buffer) for unit in compute_units for buffer in unit.read_buffers
    }
    write_buffers = {
        str(buffer) for unit in compute_units for buffer in unit.write_buffers
    }
    carry_subjects = {
        str(edge.subject) for edge in plan.dataflow_edges if str(edge.kind) == "carry"
    }
    layout_scopes = {str(layout.subject): str(layout.scope) for layout in plan.layout_specs}

    assert {"K_shared", "V_shared", "acc_o", "scores_max", "logsum"}.issubset(read_buffers)
    assert {"V_shared", "acc_o", "scores_max", "logsum"}.issubset(write_buffers)
    assert {
        "K_shared",
        "V_shared",
        "acc_o",
        "scores_max",
        "scores_scale",
        "scores_sum",
        "logsum",
    }.issubset(carry_subjects)
    assert layout_scopes["K_shared"].startswith("blackhole.cb")
    assert layout_scopes["V_shared"].startswith("blackhole.cb")
    assert layout_scopes["Output"] == "global"
    assert layout_scopes["acc_o"] == "blackhole.acc"
    assert layout_scopes["scores_max"] == "blackhole.acc"
    assert layout_scopes["logsum"] == "blackhole.acc"


def test_phase_b_pipeline_captures_direct_logical_bridge_specs():
    mod = tvm.IRModule({"main": fragment_fill_cast_publish_kernel().with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = LowerToBlackholePhaseB(mod)
    main = mod["main"]

    assert main.attrs.get("blackhole.compute_regions") is None
    specs = list(main.attrs["tl.blackhole_logical_buffer_tile_bridge_specs"])
    by_buffer = {str(spec["buffer"]): spec for spec in specs}

    assert {"C_local", "D_local"}.issubset(by_buffer)
    for name in ("C_local", "D_local"):
        spec = by_buffer[name]
        assert str(spec["scope"]) == "local"
        assert tuple(int(dim) for dim in spec["shape"]) == (32, 32)
        assert tuple(int(dim) for dim in spec["local_shape"]) == (8,)
        assert int(spec["thread_extent"]) == 128
        assert int(spec["replicate_extent"]) == 1
        assert len(spec["inverse_logical_index_exprs"]) == 3


def test_tt_metal_api_granularity_rejects_helper_composite_builtins():
    mod = _prepare_blackhole_tt_program_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            32,
            128,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    builtin_names = _collect_blackhole_builtin_names(main)
    payload = dict(main.attrs["tl.tt_program"].payload)

    for builtin_name in HELPER_COMPOSITE_BLACKHOLE_BUILTINS:
        assert f"tl.blackhole.{builtin_name}" not in builtin_names
    assert "compute_epilogue_ops" not in payload


def test_tt_metal_builtin_selector_lowers_compute_idioms_before_plan_tt_compute():
    mod = _prepare_blackhole_builtin_selection_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            32,
            128,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    builtin_suffixes = {
        name.split("tl.blackhole.", 1)[1]
        for name in _collect_blackhole_builtin_names(main)
    }

    assert main.attrs.get("tl.blackhole_tt_metal_builtin_selection")
    assert {
        "binary_max_tile_init",
        "binary_max_tile",
        "reduce_init",
        "reduce_tile",
        "reduce_uninit",
        "mul_tiles_init",
        "mul_tiles",
        "add_tiles_init",
        "add_tiles",
        "mul_bcast_rows_init_short",
        "mul_tiles_bcast_rows",
        "add_bcast_rows_init_short",
        "add_tiles_bcast_rows",
        "exp2_tile_init",
        "exp2_tile",
        "pack_tile",
    }.issubset(builtin_suffixes)
    assert not any(
        name.split("tl.blackhole.", 1)[1] in HELPER_COMPOSITE_BLACKHOLE_BUILTINS
        for name in _collect_blackhole_builtin_names(main)
    )
    assert not any(name in LEGACY_LOCAL_BLACKHOLE_BUILTINS for name in builtin_suffixes)


def test_tt_builtin_selection_stages_cb_plans_without_legacy_attr_handoff():
    mod = _prepare_blackhole_builtin_selection_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            32,
            128,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]

    assert main.attrs.get("blackhole.cb_requirements") is None
    assert main.attrs.get("tl.blackhole_lowering_requirements_seed") is None
    assert len(tt_program.cb_plans) > 0
    assert all(
        int(cb_plan.cb_id) == index for index, cb_plan in enumerate(tt_program.cb_plans)
    )
    assert all(
        str(cb_plan.flow_class) in {"state", "stream", "republish"}
        for cb_plan in tt_program.cb_plans
    )
    assert all(int(cb_plan.page_size_bytes) > 0 for cb_plan in tt_program.cb_plans)
    assert all("requirement_index" not in dict(cb_plan.payload) for cb_plan in tt_program.cb_plans)


def test_plan_tt_transport_consumes_staged_cb_plans_without_legacy_cb_attr():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)

    func = mod["main"]
    assert func.attrs.get("tl.tt_program") is not None
    if func.attrs and func.attrs.get("blackhole.cb_requirements") is not None:
        func = func.without_attr("blackhole.cb_requirements")
    if func.attrs and func.attrs.get("tl.blackhole_lowering_requirements_seed") is not None:
        func = func.without_attr("tl.blackhole_lowering_requirements_seed")
    mod = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    mod = tilelang.transform.PlanTTTransport()(mod)
    main = mod["main"]

    assert main.attrs.get("blackhole.cb_requirements") is None
    assert main.attrs["tl.tt_program"].cb_plans


def test_build_tt_program_consumes_plan_and_analysis_attrs_without_spatial_program():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_legacy_spatial_attrs(mod)
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
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


def test_build_tt_program_exposes_mesh_and_buffer_distribution_plans():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    spatial_plan = mod["main"].attrs["tl.spatial_plan"]
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)

    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert len(tt_program.mesh_plans) == 1
    mesh_plan = tt_program.mesh_plans[0]
    assert str(mesh_plan.name) == "unit_mesh"
    assert str(mesh_plan.mesh_kind) == "unit_mesh"
    assert tuple(int(dim) for dim in mesh_plan.mesh_shape) == (1, 1)
    assert tuple(int(dim) for dim in mesh_plan.device_range_shape) == (1, 1)

    spatial_subjects = {str(layout.subject) for layout in spatial_plan.layout_specs}
    distributions = {
        str(plan.buffer): plan for plan in tt_program.buffer_distribution_plans
    }
    assert spatial_subjects.issubset(distributions)
    assert all(str(plan.mesh_plan) == "unit_mesh" for plan in distributions.values())
    assert all(int(plan.mesh_plan_index) == 0 for plan in distributions.values())
    assert {str(plan.distribution_kind) for plan in distributions.values()} == {"replicated"}
    assert str(distributions["A"].memory_space) == "DRAM"
    assert str(distributions["A_shared"].memory_space) == "L1"

    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    assert "mesh_plans" in executable
    assert "buffer_distribution_plans" in executable


def test_build_tt_program_exposes_typed_compute_op_plans():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)

    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert len(tt_program.compute_op_plans) == 1
    compute_op = tt_program.compute_op_plans[0]
    assert str(compute_op.kind) == "gemm"
    assert str(compute_op.kernel_name) == "compute"
    assert int(compute_op.kernel_plan_index) == next(
        index
        for index, plan in enumerate(tt_program.kernel_plans)
        if str(plan.name) == "compute"
    )
    assert tuple(str(axis) for axis in compute_op.problem_shape_axes) == ("M", "N", "K")
    assert tuple(int(dim) for dim in compute_op.problem_shape) == (32, 32, 128)
    assert tuple(int(dim) for dim in compute_op.tile_shape) == (1, 1, 4)
    assert tuple(int(dim) for dim in compute_op.block_shape) == (1, 1, 4)
    operands = {str(binding.role): binding for binding in compute_op.operand_bindings}
    assert {str(role) for role in operands} == {"a", "b", "c"}
    assert str(operands["a"].buffer) == "A_shared"
    assert str(operands["a"].host_buffer) == "A"
    assert str(operands["a"].tensor_dtype) == "Float16_b"
    assert str(operands["b"].buffer) == "B_shared"
    assert str(operands["b"].host_buffer) == "B"
    assert str(operands["b"].transform_kind) == "transpose"
    assert str(operands["c"].buffer) == "C_local"
    assert str(operands["c"].host_buffer) == "C"
    assert str(compute_op.accumulator_dtype) == "Float32"

    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    assert "compute_op_plans" in executable
    compute_segments = [
        segment
        for segment in executable["segment_plan"]
        if str(segment["kind"]) == "compute"
    ]
    assert len(compute_segments) == 1
    assert str(compute_segments[0]["compute_ops"][0]["kind"]) == "gemm"


def test_validate_tt_program_rejects_unresolved_unsupported_compute_ops():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    payload = dict(tt_program.payload)
    payload["unsupported_compute_ops"] = ["tl.blackhole.unsupported"]
    invalid_program = _rebuild_tt_program(tt_program, payload=payload)
    invalid_main = main.with_attr("tl.tt_program", invalid_program)
    invalid_mod = tvm.IRModule({"main": invalid_main}, global_infos=mod.global_infos)

    with pytest.raises(tvm.error.InternalError, match="unsupported_compute_ops remain"):
        tilelang.transform.ValidateTTProgram()(invalid_mod)


def test_tt_planning_stages_tt_program_without_internal_bridge_attrs():
    mod = _prepare_blackhole_phase_b_module(gemm_kernel())
    mod = _drop_legacy_spatial_attrs(mod)

    mod = tilelang.transform.PlanTTBlocks()(mod)
    main = mod["main"]
    assert main.attrs.get("tl.tt_program") is not None
    assert main.attrs.get("tl.internal_tt_block_plans") is None
    assert main.attrs.get("tl.internal_tt_core_groups") is None

    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert len(tt_program.block_plans) >= 1
    assert len(tt_program.core_groups) >= 1
    assert len(tt_program.kernel_plans) >= 1
    assert len(tt_program.kernels) >= 1
    assert len(tt_program.abi_plans) >= 1
    assert main.attrs.get("tl.internal_tt_kernel_plans") is None
    assert main.attrs.get("tl.internal_tt_kernels") is None
    assert main.attrs.get("tl.internal_tt_abi_plan_seeds") is None
    assert main.attrs.get("tl.internal_tt_program_payload") is None


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


def test_task1_validate_spatial_plan_rejects_live_value_edge_without_source_value():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_edge = _rebuild_live_value_edge(
        plan.live_value_edges[0],
        source_live_value="missing_live_value",
        source_live_value_index=-1,
    )
    invalid_plan = _rebuild_spatial_plan(
        plan,
        live_value_edges=[invalid_edge, *list(plan.live_value_edges[1:])],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr(
        "tl.spatial_plan_validated"
    )
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="LiveValueEdge.*source_live_value"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_task1_validate_spatial_plan_rejects_materialization_without_target_value():
    mod = _prepare_blackhole_phase_b_module(fragment_fill_cast_publish_kernel())
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_boundary = _rebuild_materialization_boundary(
        plan.materialization_boundaries[0],
        target_live_value="missing_target_live_value",
        target_live_value_index=-1,
    )
    invalid_plan = _rebuild_spatial_plan(
        plan,
        materialization_boundaries=[
            invalid_boundary,
            *list(plan.materialization_boundaries[1:]),
        ],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr(
        "tl.spatial_plan_validated"
    )
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="MaterializationBoundary.*target_live_value"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_task1_validate_spatial_plan_rejects_missing_live_value_schema():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_plan = _rebuild_spatial_plan(
        plan,
        live_values=[],
        live_value_edges=[],
        materialization_boundaries=[],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr(
        "tl.spatial_plan_validated"
    )
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="SpatialPlan live_values"):
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
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)

    with pytest.raises(Exception, match="sync_plans|PlanTTSync|TTProgram"):
        tilelang.transform.BuildTTProgram()(mod)
