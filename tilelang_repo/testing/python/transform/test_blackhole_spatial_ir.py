import re
import subprocess
import sys
from pathlib import Path

import pytest
import tilelang
from tilelang import language as T
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


BLACKHOLE_TILE_COMPUTE_LEAF_OPS = {
    "fill_tile",
    "copy_tile",
    "binary_max_tile",
    "add_tiles",
    "mul_tiles",
    "mul_tiles_bcast_cols",
    "add_tiles_bcast_cols",
    "exp2_tile",
    "typecast_tile",
    "reduce_tile",
    "pack_tile",
    "recip_tile",
}

BLACKHOLE_TILE_COMPUTE_PATTERN_OPS = BLACKHOLE_TILE_COMPUTE_LEAF_OPS | {
    "matmul_tiles",
}


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


def _collect_call_op_names(node):
    names = set()

    def visit(expr):
        if isinstance(expr, tvm.tir.Call):
            op = expr.op
            if hasattr(op, "name"):
                names.add(op.name)

    body = node.body if hasattr(node, "body") else node
    tvm.tir.stmt_functor.post_order_visit(body, visit)
    return names


def _collect_blackhole_tile_compute_operations(node):
    operations = set()

    def visit(expr):
        if isinstance(expr, tvm.tir.Call):
            op = expr.op
            if hasattr(op, "name") and op.name == "tl.tileop.blackhole_compute":
                if expr.args and isinstance(expr.args[0], tvm.tir.StringImm):
                    operations.add(str(expr.args[0].value))

    body = node.body if hasattr(node, "body") else node
    tvm.tir.stmt_functor.post_order_visit(body, visit)
    return operations


def _row_reduce_sum_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32,), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "bfloat16")
            A_local = T.alloc_fragment((32, 32), "float32")
            B_local = T.alloc_fragment((32,), "float32")

            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            T.reduce_sum(A_local, B_local, dim=1, clear=True)
            T.copy(B_local, B)

    return main


def _lower_blackhole_frontend(prim_func):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    with target:
        return LowerAndLegalize(mod, target)


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


def _source_emitter_hook_names():
    source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc"
    ).read_text()
    hook_table = source.split("GetTileComputeSourceEmitterHooks()", 1)[1]
    hook_table = hook_table.split("return hooks;", 1)[0]
    enum_hooks = set(
        re.findall(
            r"\{\s*BlackholeTileComputeSourceEmitterKind::k([A-Za-z0-9_]+)\s*,"
            r"\s*&PlanTTKernelABI::Emit",
            hook_table,
        )
    )
    if enum_hooks:
        def _snake(name):
            return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()

        return {_snake(name) for name in enum_hooks}
    return set(re.findall(r'\{\s*"([^"]+)"\s*,\s*&PlanTTKernelABI::Emit', hook_table))


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


def _replace_spatial_plan(mod, spatial_plan):
    func = mod["main"].with_attr("tl.spatial_plan", spatial_plan)
    return tvm.IRModule({"main": func}, global_infos=mod.global_infos)


def _prepare_blackhole_phase_b_without_target_opt_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        mod = LowerAndLegalize(mod, target)
    return LowerToBlackholePhaseB(mod)


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


def _rebuild_access_region(
    region,
    *,
    name=None,
    subject=None,
    unit_name=None,
    unit_index=None,
    access_kind=None,
    value_kind=None,
    logical_rank=None,
    loop_vars=None,
    index_exprs=None,
    lower_bounds=None,
    extents=None,
    strides=None,
    coverage_kind=None,
    predicate_kind=None,
    anchors=None,
):
    make_access_region = tvm.get_global_func("tl.AccessRegion")
    return make_access_region(
        str(region.name) if name is None else name,
        str(region.subject) if subject is None else subject,
        str(region.unit_name) if unit_name is None else unit_name,
        int(region.unit_index) if unit_index is None else unit_index,
        str(region.access_kind) if access_kind is None else access_kind,
        str(region.value_kind) if value_kind is None else value_kind,
        int(region.logical_rank) if logical_rank is None else logical_rank,
        list(region.loop_vars) if loop_vars is None else loop_vars,
        list(region.index_exprs) if index_exprs is None else index_exprs,
        list(region.lower_bounds) if lower_bounds is None else lower_bounds,
        list(region.extents) if extents is None else extents,
        list(region.strides) if strides is None else strides,
        str(region.coverage_kind) if coverage_kind is None else coverage_kind,
        str(region.predicate_kind) if predicate_kind is None else predicate_kind,
        list(region.anchors) if anchors is None else anchors,
    )


def _rebuild_dependence_component(
    component,
    *,
    name=None,
    component_kind=None,
    unit_indices=None,
    edge_indices=None,
    subjects=None,
    anchors=None,
):
    make_dependence_component = tvm.get_global_func("tl.DependenceComponent")
    return make_dependence_component(
        str(component.name) if name is None else name,
        str(component.component_kind) if component_kind is None else component_kind,
        list(component.unit_indices) if unit_indices is None else unit_indices,
        list(component.edge_indices) if edge_indices is None else edge_indices,
        list(component.subjects) if subjects is None else subjects,
        list(component.anchors) if anchors is None else anchors,
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
    use_kind=None,
    consumer_access_region_index=None,
    source_version_index=None,
    target_version_index=None,
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
        str(edge.use_kind) if use_kind is None else use_kind,
        int(edge.consumer_access_region_index)
        if consumer_access_region_index is None
        else consumer_access_region_index,
        int(edge.source_version_index)
        if source_version_index is None
        else source_version_index,
        int(edge.target_version_index)
        if target_version_index is None
        else target_version_index,
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
    source_access_region_index=None,
    target_access_region_index=None,
    event_lifetime_kind=None,
    min_publish_pages=None,
    max_consume_pages=None,
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
        int(boundary.source_access_region_index)
        if source_access_region_index is None
        else source_access_region_index,
        int(boundary.target_access_region_index)
        if target_access_region_index is None
        else target_access_region_index,
        str(boundary.event_lifetime_kind)
        if event_lifetime_kind is None
        else event_lifetime_kind,
        int(boundary.min_publish_pages)
        if min_publish_pages is None
        else min_publish_pages,
        int(boundary.max_consume_pages)
        if max_consume_pages is None
        else max_consume_pages,
        list(boundary.anchors) if anchors is None else anchors,
    )


def _rebuild_spatial_plan(
    plan,
    *,
    execution_units=None,
    access_regions=None,
    dataflow_edges=None,
    dependence_components=None,
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
        list(plan.access_regions) if access_regions is None else access_regions,
        list(plan.dataflow_edges) if dataflow_edges is None else dataflow_edges,
        list(plan.dependence_components)
        if dependence_components is None
        else dependence_components,
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
    kernels=None,
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
        list(program.kernels) if kernels is None else kernels,
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
    )


def _rebuild_tt_compute_op_plan(
    plan,
    *,
    kind=None,
    operation_name=None,
    operand_bindings=None,
):
    make_tt_compute_op = tvm.get_global_func("tl.TTComputeOpPlan")
    return make_tt_compute_op(
        str(plan.name),
        str(plan.kernel_name),
        int(plan.kernel_plan_index),
        str(plan.kind) if kind is None else kind,
        str(plan.operation_name) if operation_name is None else operation_name,
        bool(plan.enabled),
        list(plan.operand_bindings) if operand_bindings is None else operand_bindings,
        list(plan.problem_shape_axes),
        list(plan.problem_shape),
        list(plan.tile_shape),
        list(plan.block_shape),
        list(plan.subblock_shape),
        str(plan.accumulator_dtype),
        str(plan.mbarrier_buffer),
        str(plan.mbarrier_scope),
        list(plan.mbarrier_index_exprs),
    )


def _rebuild_tt_materialization_plan(
    plan,
    *,
    materialization_boundary=None,
    materialization_boundary_index=None,
):
    make_tt_materialization_plan = tvm.get_global_func("tl.TTMaterializationPlan")
    return make_tt_materialization_plan(
        str(plan.name),
        str(plan.source_live_form),
        str(plan.materialization_boundary)
        if materialization_boundary is None
        else materialization_boundary,
        int(plan.materialization_boundary_index)
        if materialization_boundary_index is None
        else materialization_boundary_index,
        str(plan.target_buffer),
        str(plan.host_buffer),
        str(plan.target_kernel),
        str(plan.bridge_kind),
        str(plan.materialization_kind),
        str(plan.materialization_protocol),
        str(plan.publication_protocol),
        list(plan.required_cb_plan_indices),
        list(plan.required_sync_plan_indices),
        str(plan.produced_live_form),
    )


def _rebuild_tt_consumer_binding_plan(
    plan,
    *,
    live_value_edge=None,
    live_value_edge_index=None,
):
    make_tt_consumer_binding_plan = tvm.get_global_func("tl.TTConsumerBindingPlan")
    return make_tt_consumer_binding_plan(
        str(plan.name),
        str(plan.consumer_kernel),
        str(plan.consumer_op_kind),
        str(plan.source_live_form),
        str(plan.live_value_edge) if live_value_edge is None else live_value_edge,
        int(plan.live_value_edge_index)
        if live_value_edge_index is None
        else live_value_edge_index,
        bool(plan.accepts_distributed_slice),
        bool(plan.requires_full_logical_tile),
        int(plan.abi_plan_index),
        str(plan.target_buffer),
        str(plan.materialization_plan),
    )


def _assert_no_tt_plan_payload_surface(tt_program):
    plan_groups = (
        tt_program.mesh_plans,
        tt_program.buffer_distribution_plans,
        tt_program.block_plans,
        tt_program.kernel_plans,
        tt_program.cb_plans,
        tt_program.transport_plans,
        tt_program.sync_plans,
        tt_program.abi_plans,
        tt_program.execution_plans,
        tt_program.semaphore_plans,
        tt_program.dst_layout_plans,
        tt_program.live_form_plans,
        tt_program.materialization_plans,
        tt_program.consumer_binding_plans,
    )
    for plans in plan_groups:
        assert all(not hasattr(plan, "payload") for plan in plans)


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


def test_blackhole_lowering_support_facts_have_no_contract_map_surface():
    hits = _source_tree_rg(
        r"buffer_materialization_contracts|buffer_flow_contracts|"
        r"BuildBufferMaterializationContractMap|FindBufferMaterializationContract",
        REPO_ROOT / "tilelang_repo/src/transform/common/blackhole_lowering_requirements.h",
        REPO_ROOT / "tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.h",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.cc",
    )
    assert hits == []


def test_blackhole_compute_op_planning_has_no_map_seed_contract_surface():
    hits = _source_tree_rg(
        r"compute_op_seeds_|BuildGemmComputeOpSeed|"
        r"BuildTTComputeOpPlanFromContract|BuildComputeOperandBindingPlanFromContract",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.h",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.cc",
    )
    assert hits == []


def test_tile_compute_pattern_table_covers_current_leaf_operation_names():
    pattern_table = tvm.get_global_func("tl.BlackholeTileComputePatternTable")()
    operation_names = {str(pattern["operation_name"]) for pattern in pattern_table}
    source_emitters = {
        str(pattern["source_emitter"])
        for pattern in pattern_table
        if str(pattern["source_emitter"])
    }
    composite_names = {
        "softmax",
        "exp2_affine",
        "row_broadcast_exp2_affine",
    }
    standalone_explicit_source_ops = {
        "fill_tile",
        "copy_tile",
        "typecast_tile",
        "binary_max_tile",
        "add_tiles",
        "mul_tiles",
        "mul_tiles_bcast_cols",
        "exp2_tile",
        "reduce_tile",
    }

    assert BLACKHOLE_TILE_COMPUTE_PATTERN_OPS <= operation_names
    assert operation_names.isdisjoint(composite_names)
    assert all(str(pattern["selected_output"]) == "tt_compute_op_plan" for pattern in pattern_table)
    assert {
        str(pattern["operation_name"])
        for pattern in pattern_table
        if str(pattern["source_emitter"])
    } == standalone_explicit_source_ops
    assert "none" not in source_emitters


def test_tile_compute_pattern_schema_uses_typed_enums_and_optional_emitters():
    header = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.h"
    ).read_text()

    assert "enum class BlackholeTileComputeOperation" in header
    assert "enum class BlackholeTileComputeOperandRole" in header
    assert "enum class BlackholeTileComputeSourceEmitterKind" in header
    assert "std::optional<BlackholeTileComputeSourceEmitterKind> source_emitter" in header
    assert "std::vector<BlackholeTileComputeCallOperand> blackhole_compute_operands" in header
    assert "std::vector<BlackholeTileComputeCallOperand> generic_tile_op_operands" in header
    assert "std::string source_emitter;" not in header
    assert "std::vector<std::string> operand_roles;" not in header


def test_tile_compute_pattern_strings_use_compact_lookup_tables():
    source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.cc"
    ).read_text()

    assert "EnumStringEntry" in source
    assert "FindEnumName" in source
    assert "switch (kind)" not in source
    assert "switch (operation)" not in source
    assert "switch (role)" not in source
    assert "switch (form)" not in source
    assert "switch (side_effect_class)" not in source
    assert "switch (source_emitter)" not in source
    assert "Args(" not in source


def test_tile_compute_dag_builder_uses_pattern_operand_layout():
    source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_dag.cc"
    ).read_text()

    assert "AddPatternOperandEdges" in source
    assert "blackhole_compute_operands" in source
    assert "generic_tile_op_operands" in source
    assert "operation == blackhole_tile_compute_schema::" not in source


def test_tile_compute_read_only_dag_diagnostic_represents_explicit_reduce_and_gemm():
    build_dag = tvm.get_global_func("tl.BuildBlackholeTileComputeDAGDiagnostic")
    reduce_mod = _prepare_blackhole_phase_b_module(_row_reduce_sum_kernel())
    gemm_mod = _prepare_blackhole_phase_b_module(gemm_kernel())

    reduce_diag = build_dag(reduce_mod["main"])
    gemm_diag = build_dag(gemm_mod["main"])

    reduce_nodes = {str(node["op_name"]) for node in reduce_diag["nodes"]}
    reduce_roles = {str(edge["value_role"]) for edge in reduce_diag["edges"]}
    gemm_nodes = {str(node["op_name"]) for node in gemm_diag["nodes"]}
    gemm_roles = {str(edge["value_role"]) for edge in gemm_diag["edges"]}

    assert "reduce_tile" in reduce_nodes
    assert {"input", "output"} <= reduce_roles
    assert "matmul_tiles" in gemm_nodes
    assert {"a", "b", "c"} <= gemm_roles
    assert all(str(node["token_output"]) for node in reduce_diag["nodes"])


def test_tile_compute_read_only_dag_diagnostic_represents_flash_attention_leaf_ops():
    build_dag = tvm.get_global_func("tl.BuildBlackholeTileComputeDAGDiagnostic")
    mod = _lower_blackhole_frontend(
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
    diag = build_dag(mod["main"])
    node_names = {str(node["op_name"]) for node in diag["nodes"]}
    edge_roles = {str(edge["value_role"]) for edge in diag["edges"]}

    assert {
        "fill_tile",
        "copy_tile",
        "binary_max_tile",
        "mul_tiles",
        "mul_tiles_bcast_cols",
        "exp2_tile",
        "typecast_tile",
    } <= node_names
    assert {"lhs", "rhs", "output"} <= edge_roles


def test_tile_compute_covering_rejects_composite_operation_name_before_projection():
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
    tt_program = main.attrs["tl.tt_program"]
    assert tt_program.compute_op_plans

    compute_op_plans = list(tt_program.compute_op_plans)
    compute_op_plans[0] = _rebuild_tt_compute_op_plan(
        compute_op_plans[0],
        operation_name="softmax",
    )
    invalid_program = _rebuild_tt_program(
        tt_program,
        compute_op_plans=compute_op_plans,
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="TileCompute covering rejected"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_tile_compute_covering_selects_patterns_for_current_leaf_ops():
    select_covering = tvm.get_global_func("tl.SelectBlackholeTileComputeCoveringDiagnostic")
    pattern_table = tvm.get_global_func("tl.BlackholeTileComputePatternTable")()
    for pattern in pattern_table:
        operation_name = str(pattern["operation_name"])
        decision = select_covering(operation_name)
        assert str(decision["selection_kind"]) == "selected_pattern"
        assert str(decision["operation_name"]) == operation_name
        assert str(decision["result_kind"]) == str(pattern["result_kind"])
        assert str(decision["source_emitter"]) == str(pattern["source_emitter"])
        assert str(decision["selected_output"]) == "tt_compute_op_plan"


def test_tile_compute_dag_covering_selects_patterns_in_dependence_order():
    select_dag_covering = tvm.get_global_func(
        "tl.SelectBlackholeTileComputeDAGCoveringDiagnostic"
    )
    mod = _lower_blackhole_frontend(
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

    diagnostic = select_dag_covering(mod["main"])
    selected_patterns = list(diagnostic["selected_patterns"])
    selected_operations = {str(pattern["operation_name"]) for pattern in selected_patterns}

    assert str(diagnostic["selection_kind"]) == "local_dag_dp"
    assert str(diagnostic["selection_status"]) == "selected"
    assert str(diagnostic["selection_order"]) == "dependence_order"
    assert selected_patterns
    assert int(diagnostic["total_cost"]) >= len(selected_patterns)
    assert str(diagnostic["stale_fallback_policy"]) == "reject"
    assert {
        "fill_tile",
        "copy_tile",
        "binary_max_tile",
        "mul_tiles",
        "mul_tiles_bcast_cols",
        "exp2_tile",
        "typecast_tile",
        "matmul_tiles",
    } <= selected_operations
    assert all(int(pattern["node_id"]) >= 0 for pattern in selected_patterns)
    assert all(str(pattern["pattern_name"]) for pattern in selected_patterns)
    assert all(
        str(pattern["selected_output"]) == "tt_compute_op_plan"
        for pattern in selected_patterns
    )


def test_tile_compute_dag_covering_reports_fanout_materialization_policy():
    select_dag_covering = tvm.get_global_func(
        "tl.SelectBlackholeTileComputeDAGCoveringDiagnostic"
    )
    mod = _lower_blackhole_frontend(
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

    diagnostic = select_dag_covering(mod["main"])
    fanout_decisions = list(diagnostic["fanout_decisions"])
    materialization_decisions = list(diagnostic["materialization_decisions"])

    assert fanout_decisions
    assert materialization_decisions
    assert all(int(decision["producer_node"]) >= 0 for decision in fanout_decisions)
    assert all(int(decision["use_count"]) >= 2 for decision in fanout_decisions)
    assert {
        "materialize_before_cross_event_use",
        "share_live_value",
    } >= {str(decision["policy"]) for decision in fanout_decisions}
    assert any(
        str(decision["policy"]) == "live_form_solver_required_for_cross_event_use"
        for decision in materialization_decisions
    )
    assert all(str(decision["evidence"]) for decision in fanout_decisions)


def test_tile_compute_production_path_uses_covering_selection():
    source_dispatch_hits = _source_tree_rg(
        r"SelectBlackholeTileComputeCovering|EmitCoveredBlackholeTileCompute",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc",
    )
    plan_recording_hits = _source_tree_rg(
        r"SelectBlackholeTileComputeCovering",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_abi.cc",
    )
    assert source_dispatch_hits
    assert plan_recording_hits


def test_tile_compute_production_path_does_not_persist_dag_covering_cache():
    tile_compute_source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc"
    ).read_text()
    planner_source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.cc"
    ).read_text()
    planner_header = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.h"
    ).read_text()

    assert "LoadTileComputeDAGCovering" not in planner_source
    assert "LoadTileComputeDAGCovering" not in tile_compute_source
    assert "TileComputeDAGCoveringDecisionForStmt" not in planner_header
    assert "CurrentTileComputeDAGCoveringDecision" not in planner_header
    assert "tile_compute_dag_covering_decisions_" not in planner_header
    assert "active_tile_compute_dag_covering_decision_" not in planner_header


def test_tile_compute_covering_header_does_not_expose_dag_covering_as_production_api():
    covering_header = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_covering.h"
    ).read_text()

    assert "struct BlackholeTileComputeDAGCovering" not in covering_header
    assert "SelectBlackholeTileComputeDAGCovering(" not in covering_header


def test_tile_compute_explicit_source_path_uses_leaf_covering_without_dag_cache():
    tile_compute_source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc"
    ).read_text()
    assert "SelectBlackholeTileComputeCovering(operation)" in tile_compute_source
    assert "EmitCoveredBlackholeTileCompute(op, covering)" in tile_compute_source
    assert "active_tile_compute_dag_covering_decision_" not in tile_compute_source


def test_tile_compute_gemm_plan_construction_uses_leaf_covering_decision():
    abi_source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_abi.cc"
    ).read_text()
    assert (
        "const BlackholeTileComputeCoveringDecision covering" in abi_source
        and "BuildTTComputeOpPlanFromFact(" in abi_source
        and "SelectBlackholeTileComputeCovering(\"matmul_tiles\")" in abi_source
    )


def test_tile_compute_covered_source_path_has_no_operation_name_dispatch_chain():
    legacy_dispatch_hits = _source_tree_rg(
        r"if \(operation == blackhole_tile_compute_schema::",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc",
    )
    assert legacy_dispatch_hits == []


def test_tile_compute_binary_source_emission_has_no_operation_name_builtin_selection():
    legacy_binary_selection_hits = _source_tree_rg(
        r"operation_name == blackhole_tile_compute_schema::",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc",
    )
    assert legacy_binary_selection_hits == []


def test_tile_compute_source_emitter_hooks_cover_pattern_table():
    pattern_table = tvm.get_global_func("tl.BlackholeTileComputePatternTable")()
    pattern_emitters = {
        str(pattern["source_emitter"])
        for pattern in pattern_table
        if str(pattern["source_emitter"])
    }
    hook_emitters = _source_emitter_hook_names()

    assert pattern_emitters <= hook_emitters
    assert "none" not in hook_emitters


def test_tile_compute_covered_source_dispatch_has_no_inline_emitter_table():
    legacy_dispatch_hits = _source_tree_rg(
        r"using SourceEmitter|std::vector<std::pair<std::string, SourceEmitter>>|"
        r"std::find_if\(",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc",
    )
    assert legacy_dispatch_hits == []


def test_tile_compute_reduce_source_path_uses_covering_dispatch():
    legacy_reduce_dispatch_hits = _source_tree_rg(
        r"GenerateRowReductionSequence\(explicit_reduce_match\)",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.cc",
    )
    covered_dispatch_hits = _source_tree_rg(
        r"LowerExplicitTileComputeCall\(call\)",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.cc",
    )

    assert legacy_reduce_dispatch_hits == []
    assert covered_dispatch_hits


def test_blackhole_frontend_preserves_reduce_tileop_before_tt_selection():
    mod = _lower_blackhole_frontend(_row_reduce_sum_kernel())
    op_names = _collect_call_op_names(mod["main"])

    assert "tl.tileop.reduce" in op_names
    assert "tl.blackhole.reduce_tile" not in op_names
    assert "tir.call_extern" not in op_names


def test_blackhole_frontend_normalizes_flash_attention_leaf_tile_compute_before_tt_selection():
    mod = _lower_blackhole_frontend(
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
    operations = _collect_blackhole_tile_compute_operations(mod["main"])

    assert {
        "fill_tile",
        "copy_tile",
        "binary_max_tile",
        "add_tiles",
        "mul_tiles",
        "mul_tiles_bcast_cols",
        "exp2_tile",
        "typecast_tile",
    }.issubset(operations)


def test_blackhole_frontend_tile_compute_normalization_uses_leaf_operations():
    mod = _lower_blackhole_frontend(
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
    operations = _collect_blackhole_tile_compute_operations(mod["main"])

    assert operations
    assert operations <= BLACKHOLE_TILE_COMPUTE_LEAF_OPS


def test_lower_tile_op_has_single_blackhole_tile_compute_normalizer_surface():
    source = (REPO_ROOT / "tilelang_repo/src/transform/lower_tile_op.cc").read_text()

    helper_defs = re.findall(r"\bStmt\s+MakeBlackholeTileComputeCall\s*\(", source)
    store_defs = re.findall(
        r"\bStmt\s+TryNormalize(?:BlackholeTileCompute)?Store\s*\(", source
    )
    loop_defs = re.findall(
        r"\bStmt\s+TryNormalize(?:BlackholeTileCompute)?Loop\s*\(", source
    )

    assert helper_defs == ["Stmt MakeBlackholeTileComputeCall("]
    assert len(store_defs) == 1
    assert len(loop_defs) == 1


def test_spatial_plan_records_preserved_reduce_as_compute_producer():
    mod = _prepare_blackhole_phase_b_module(_row_reduce_sum_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]
    compute_units = [unit for unit in plan.execution_units if str(unit.unit_role) == "compute"]
    compute_reads = {str(buffer) for unit in compute_units for buffer in unit.read_buffers}
    compute_writes = {str(buffer) for unit in compute_units for buffer in unit.write_buffers}
    dataflow_subjects = {str(edge.subject) for edge in plan.dataflow_edges}

    assert "A_local" in compute_reads
    assert "B_local" in compute_writes
    assert "B_local" in dataflow_subjects


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


def test_algorithmic_access_region_covers_copy_unit_reads_and_writes():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    plan = mod["main"].attrs["tl.spatial_plan"]

    regions = list(plan.access_regions)
    unit_access_count = sum(
        len(unit.read_buffers) + len(unit.write_buffers)
        for unit in plan.execution_units
    )

    assert len(regions) == unit_access_count
    assert {str(region.access_kind) for region in regions} == {"read", "write"}
    assert all(int(region.unit_index) >= 0 for region in regions)
    assert all(str(region.unit_name) for region in regions)
    assert all(str(region.subject) for region in regions)
    assert all(int(region.logical_rank) >= 0 for region in regions)
    assert all(str(region.coverage_kind) in {"full", "scalar"} for region in regions)
    assert all(str(region.predicate_kind) == "unconditional" for region in regions)

    by_unit_subject_kind = {
        (str(region.unit_name), str(region.subject), str(region.access_kind))
        for region in regions
    }
    for unit in plan.execution_units:
        for subject in unit.read_buffers:
            assert (str(unit.name), str(subject), "read") in by_unit_subject_kind
        for subject in unit.write_buffers:
            assert (str(unit.name), str(subject), "write") in by_unit_subject_kind


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


def test_algorithmic_live_values_carry_ssa_version_and_boundary_evidence():
    mod = _prepare_blackhole_phase_b_module(fragment_fill_cast_publish_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]

    live_value_by_subject = {str(value.subject): value for value in plan.live_values}
    assert {"C_local", "D_local"}.issubset(live_value_by_subject)
    c_live = live_value_by_subject["C_local"]
    d_live = live_value_by_subject["D_local"]

    assert int(c_live.version_index) >= 0
    assert int(d_live.version_index) >= 0
    assert str(c_live.definition_kind) == "compute_write"
    assert str(d_live.definition_kind) == "compute_write"
    assert int(c_live.defining_access_region_index) >= 0
    assert int(d_live.defining_access_region_index) >= 0

    boundary = next(
        boundary
        for boundary in plan.materialization_boundaries
        if str(boundary.source_live_value) == str(c_live.name)
        and str(boundary.target_live_value) == str(d_live.name)
    )
    live_edge = plan.live_value_edges[int(boundary.live_value_edge_index)]

    assert str(live_edge.use_kind) == "materialization_consume"
    assert int(live_edge.source_version_index) == int(c_live.version_index)
    assert int(live_edge.target_version_index) == int(d_live.version_index)
    assert int(live_edge.consumer_access_region_index) >= 0
    assert int(boundary.source_access_region_index) == int(
        c_live.defining_access_region_index
    )
    assert int(boundary.target_access_region_index) == int(
        d_live.defining_access_region_index
    )
    assert str(boundary.event_lifetime_kind) == "single_event"
    assert int(boundary.min_publish_pages) == 1
    assert int(boundary.max_consume_pages) == 1


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


def test_algorithmic_spatial_plan_records_carry_dependence_components():
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

    carry_components = [
        component
        for component in plan.dependence_components
        if str(component.component_kind) == "carry_cycle"
    ]
    component_subjects = {
        str(subject) for component in carry_components for subject in component.subjects
    }
    component_edge_kinds = {
        str(plan.dataflow_edges[int(edge_index)].kind)
        for component in carry_components
        for edge_index in component.edge_indices
    }

    assert carry_components
    assert {"acc_o", "scores_max", "logsum"}.issubset(component_subjects)
    assert "carry" in component_edge_kinds


def test_phase_b_pipeline_records_live_values_without_logical_bridge_attr():
    mod = tvm.IRModule({"main": fragment_fill_cast_publish_kernel().with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = LowerToBlackholePhaseB(mod)
    main = mod["main"]

    assert main.attrs.get("blackhole.compute_regions") is None
    assert main.attrs.get("tl.blackhole_logical_buffer_tile_bridge_specs") is None
    spatial_plan = main.attrs["tl.spatial_plan"]
    layout_specs = {str(spec.subject): spec for spec in spatial_plan.layout_specs}

    assert {"C_local", "D_local"}.issubset(layout_specs)
    for name in ("C_local", "D_local"):
        spec = layout_specs[name]
        assert tuple(int(dim) for dim in spec.logical_shape) == (32, 32)
        assert tuple(int(dim) for dim in spec.local_shape) == (8,)
        assert int(spec.thread_extent) == 128
        assert int(spec.replicate_extent) == 1
        assert len(spec.inverse_logical_index_exprs) == 3


def test_tt_metal_api_granularity_uses_typed_leaf_plans():
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
    tt_program = main.attrs["tl.tt_program"]

    assert not hasattr(tt_program, "payload")
    assert "tl.blackhole.reduce_tile" in builtin_names
    assert "tl.blackhole.exp2_tile" in builtin_names
    assert {
        str(op.operation_name)
        for op in tt_program.compute_op_plans
        if str(op.kind) != "gemm"
    } <= BLACKHOLE_TILE_COMPUTE_LEAF_OPS


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
        "mul_bcast_cols_init_short",
        "mul_tiles_bcast_cols",
        "add_bcast_cols_init_short",
        "add_tiles_bcast_cols",
        "exp2_tile_init",
        "exp2_tile",
        "pack_tile",
    }.issubset(builtin_suffixes)


def test_tt_metal_builtin_selector_lowers_preserved_reduce_tileop():
    mod = _prepare_blackhole_builtin_selection_module(_row_reduce_sum_kernel())
    main = mod["main"]
    op_names = _collect_call_op_names(main)
    reduce_ops = [
        op
        for op in main.attrs["tl.tt_program"].compute_op_plans
        if str(op.operation_name) == "reduce_tile"
    ]

    assert "tl.tileop.reduce" not in op_names
    assert {
        "tl.blackhole.reduce_init",
        "tl.blackhole.reduce_tile",
        "tl.blackhole.reduce_uninit",
    }.issubset(op_names)
    assert reduce_ops
    assert all(str(op.kind) == "reduce" for op in reduce_ops)


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
    assert all(not hasattr(cb_plan, "payload") for cb_plan in tt_program.cb_plans)


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
    _assert_no_tt_plan_payload_surface(tt_program)
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
    assert not hasattr(tt_program, "payload")
    assert all(not hasattr(kernel, "payload") for kernel in tt_program.kernels)
    assert all(not hasattr(core_group, "payload") for core_group in tt_program.core_groups)
    assert len(tt_program.compute_op_plans) == 1
    compute_op = tt_program.compute_op_plans[0]
    assert not hasattr(compute_op, "payload")
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
    assert all(not hasattr(binding, "payload") for binding in operands.values())
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


def test_tt_program_has_no_unresolved_unsupported_compute_payload():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    tt_program = mod["main"].attrs["tl.tt_program"]

    assert not hasattr(tt_program, "payload")


def test_tt_kernel_does_not_expose_payload_compute_ops_surface():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    tt_program = mod["main"].attrs["tl.tt_program"]

    assert all(not hasattr(kernel, "payload") for kernel in tt_program.kernels)
    assert all(
        not hasattr(compute_op, "payload") for compute_op in tt_program.compute_op_plans
    )
    assert all(
        not hasattr(binding, "payload")
        for compute_op in tt_program.compute_op_plans
        for binding in compute_op.operand_bindings
    )


def test_tt_kernel_public_schema_has_no_map_any_leaf_fields():
    hits = _source_tree_rg(
        r"ffi::Map<ffi::String, ffi::Any>\s+(launch_spec|compute_config)|"
        r"ffi::Array<ffi::Any>\s+per_work_arg_specs",
        "tilelang_repo/src/transform/common/tt_target_program.h",
        "tilelang_repo/src/transform/common/tt_target_program.cc",
    )
    assert hits == []


def test_tt_abi_plan_public_schema_has_no_array_any_leaf_fields():
    hits = _source_tree_rg(
        r"ffi::Array<ffi::Any>\s+"
        r"(runtime_args|common_runtime_args|compile_time_arg_specs|accessors|semaphore_bindings)",
        "tilelang_repo/src/transform/common/tt_target_program.h",
        "tilelang_repo/src/transform/common/tt_target_program.cc",
    )
    assert hits == []


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


def test_algorithmic_validate_spatial_plan_rejects_missing_access_region():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_plan = _rebuild_spatial_plan(plan, access_regions=[])
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr("tl.spatial_plan_validated")
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="SpatialPlan access_regions"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_algorithmic_validate_spatial_plan_rejects_invalid_dependence_component_edge():
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
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_component = _rebuild_dependence_component(
        plan.dependence_components[0],
        edge_indices=[-1],
    )
    invalid_plan = _rebuild_spatial_plan(
        plan,
        dependence_components=[
            invalid_component,
            *list(plan.dependence_components[1:]),
        ],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr("tl.spatial_plan_validated")
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="DependenceComponent.*edge_index"):
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


def test_algorithmic_validate_spatial_plan_rejects_live_value_edge_version_mismatch():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    main = mod["main"]
    plan = main.attrs["tl.spatial_plan"]
    invalid_edge = _rebuild_live_value_edge(
        plan.live_value_edges[0],
        source_version_index=-1,
    )
    invalid_plan = _rebuild_spatial_plan(
        plan,
        live_value_edges=[invalid_edge, *list(plan.live_value_edges[1:])],
    )
    func = main.with_attr("tl.spatial_plan", invalid_plan).without_attr(
        "tl.spatial_plan_validated"
    )
    broken = tvm.IRModule({"main": func}, global_infos=mod.global_infos)

    with pytest.raises(Exception, match="LiveValueEdge.*source_version_index"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_algorithmic_validate_spatial_plan_rejects_distributed_slice_without_access_region():
    mod = _prepare_blackhole_phase_b_module(fragment_fill_cast_publish_kernel())
    plan = mod["main"].attrs["tl.spatial_plan"]
    materialize_edge_index = next(
        index
        for index, edge in enumerate(plan.live_value_edges)
        if str(edge.relation_kind) == "materialize"
    )
    invalid_edge = _rebuild_live_value_edge(
        plan.live_value_edges[materialize_edge_index],
        consumer_access_region_index=-1,
        requires_full_logical_value=False,
        accepts_distributed_slice=True,
    )
    live_value_edges = list(plan.live_value_edges)
    live_value_edges[materialize_edge_index] = invalid_edge
    invalid_plan = _rebuild_spatial_plan(plan, live_value_edges=live_value_edges)
    broken = _drop_validated_spatial_plan_stamp(
        _replace_spatial_plan(mod, invalid_plan)
    )

    with pytest.raises(Exception, match="distributed slice.*AccessRegion"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_algorithmic_validate_spatial_plan_rejects_loop_carried_boundary_without_component():
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
    assert any(
        str(boundary.event_lifetime_kind) == "loop_carried"
        for boundary in plan.materialization_boundaries
    )
    invalid_plan = _rebuild_spatial_plan(plan, dependence_components=[])
    broken = _drop_validated_spatial_plan_stamp(
        _replace_spatial_plan(mod, invalid_plan)
    )

    with pytest.raises(Exception, match="loop_carried.*DependenceComponent"):
        tilelang.transform.ValidateSpatialPlan()(broken)


def test_algorithmic_live_form_solver_uses_boundary_coverage_for_consumer_binding():
    mod = _prepare_blackhole_phase_b_without_target_opt_module(
        fragment_fill_cast_publish_kernel()
    )
    plan = mod["main"].attrs["tl.spatial_plan"]
    boundaries = list(plan.materialization_boundaries)
    boundary_index = next(
        index
        for index, boundary in enumerate(boundaries)
        if str(boundary.logical_coverage) == "distributed_slice"
    )
    boundaries[boundary_index] = _rebuild_materialization_boundary(
        boundaries[boundary_index],
        logical_coverage="full_logical_value",
    )
    mod = _drop_validated_spatial_plan_stamp(
        _replace_spatial_plan(
            mod, _rebuild_spatial_plan(plan, materialization_boundaries=boundaries)
        )
    )
    mod = tilelang.transform.ValidateSpatialPlan()(mod)
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)

    tt_program = mod["main"].attrs["tl.tt_program"]
    binding = next(
        plan
        for plan in tt_program.consumer_binding_plans
        if str(plan.consumer_op_kind) == "cast_fragment_slice"
    )
    assert bool(binding.accepts_distributed_slice) is False
    assert bool(binding.requires_full_logical_tile) is True


def test_algorithmic_live_form_solver_owns_tt_live_form_decision_literals():
    hits = _source_tree_rg(
        r"thread_distributed_slice|cb_materialized_tile|"
        r"producer_thread_lane|materialized_cb_pages",
        "tilelang_repo/src/transform/lower_blackhole_state.cc",
    )
    assert hits == []


def test_algorithmic_plan_tt_kernel_abi_uses_boundary_indices_not_subject_live_value_maps():
    hits = _source_tree_rg(
        r"spatial_live_value_by_subject_|FindSpatialLiveValueRef|version_by_subject",
        "tilelang_repo/src/transform/lower_blackhole_state.cc",
        "tilelang_repo/src/transform/lower_blackhole_ops.h",
    )
    assert hits == []


def test_algorithmic_live_form_solver_has_worklist_lattice_surface():
    hits = _source_tree_rg(
        r"TTLiveFormLatticeKind|TTLiveFormWorkItem|ApplyBoundaryTransfer|JoinLiveFormState",
        "tilelang_repo/src/transform/common/tt_live_form_solver.h",
        "tilelang_repo/src/transform/common/tt_live_form_solver.cc",
    )
    assert len(hits) >= 4


def test_executable_projection_rejects_materialization_without_boundary_index():
    mod = _prepare_blackhole_tt_program_module(fragment_fill_cast_publish_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert tt_program.materialization_plans

    materialization_plans = list(tt_program.materialization_plans)
    materialization_plans[0] = _rebuild_tt_materialization_plan(
        materialization_plans[0],
        materialization_boundary_index=-1,
    )
    invalid_program = _rebuild_tt_program(
        tt_program,
        materialization_plans=materialization_plans,
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="materialization_boundary_index"):
        tilelang.transform.MaterializeBlackholeExecutable()(broken)


def test_executable_projection_rejects_consumer_binding_without_live_edge_index():
    mod = _prepare_blackhole_tt_program_module(fragment_fill_cast_publish_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert tt_program.consumer_binding_plans

    consumer_binding_plans = list(tt_program.consumer_binding_plans)
    consumer_binding_plans[0] = _rebuild_tt_consumer_binding_plan(
        consumer_binding_plans[0],
        live_value_edge_index=-1,
    )
    invalid_program = _rebuild_tt_program(
        tt_program,
        consumer_binding_plans=consumer_binding_plans,
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="live_value_edge_index"):
        tilelang.transform.MaterializeBlackholeExecutable()(broken)


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
