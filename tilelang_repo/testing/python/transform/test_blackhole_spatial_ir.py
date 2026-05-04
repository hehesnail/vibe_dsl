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
    grid_indexed_staged_copy_kernel,
    lower_blackhole_to_tt_target,
    staged_copy_kernel,
)


BLACKHOLE_TILE_COMPUTE_LEAF_OPS = {
    "fill_tile",
    "copy_tile",
    "binary_max_tile",
    "add_tiles",
    "sub_tiles",
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


def _memory_config_annotation_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "bfloat16"),
        C: T.Tensor((64, 64), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((64, 64), "bfloat16")
            T.annotate_memory_config(
                {
                    A: T.sharded_dram(
                        strategy="width",
                        grid=T.CoreGrid(x=2, y=1),
                        shard_shape=(64, 32),
                        orientation="row_major",
                        allow_reshard=False,
                    ),
                    C: T.interleaved_dram(),
                }
            )
            T.copy(A, A_shared)
            T.copy(A_shared, C)

    return main


def _elementwise_add_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((32, 32), "bfloat16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "bfloat16")
            B_shared = T.alloc_shared((32, 32), "bfloat16")
            A_local = T.alloc_fragment((32, 32), "bfloat16")
            B_local = T.alloc_fragment((32, 32), "bfloat16")
            C_local = T.alloc_fragment((32, 32), "bfloat16")

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(32, 32):
                C_local[i, j] = A_local[i, j] + B_local[i, j]
            T.copy(C_local, C)

    return main


def _three_input_add_residue_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((32, 32), "bfloat16"),
        D: T.Tensor((32, 32), "bfloat16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "bfloat16")
            B_shared = T.alloc_shared((32, 32), "bfloat16")
            C_shared = T.alloc_shared((32, 32), "bfloat16")
            A_local = T.alloc_fragment((32, 32), "bfloat16")
            B_local = T.alloc_fragment((32, 32), "bfloat16")
            C_local = T.alloc_fragment((32, 32), "bfloat16")
            D_local = T.alloc_fragment((32, 32), "bfloat16")

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(C, C_shared)
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            T.copy(C_shared, C_local)
            for i, j in T.Parallel(32, 32):
                D_local[i, j] = A_local[i, j] + B_local[i, j] + C_local[i, j]
            T.copy(D_local, D)

    return main


def _lower_blackhole_frontend(prim_func):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    with target:
        return LowerAndLegalize(mod, target)


def _prepare_blackhole_layout_inferred_module(prim_func):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    with target:
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutReducer()(mod)
        mod = tilelang.transform.LayoutInference()(mod)
    return mod


def _prepare_blackhole_tt_program_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        if not (mod["main"].attrs and mod["main"].attrs.get("target") is not None):
            mod = LowerAndLegalize(mod, target)
        mod = lower_blackhole_to_tt_target(mod)
    return mod


def _with_test_hardware_model(
    mod,
    *,
    logical_worker_grid_x,
    logical_worker_grid_y,
    functional_worker_count,
    dram_view_count=8,
    worker_l1_size=1572864,
    dram_view_size=4278190080,
    max_cb_count=64,
    l1_allocation_alignment_bytes=32,
):
    make_hardware_model = tvm.get_global_func("tl.TTHardwareModel")
    hardware_model = make_hardware_model(
        "BLACKHOLE_TEST",
        "",
        logical_worker_grid_x,
        logical_worker_grid_y,
        functional_worker_count,
        0,
        dram_view_count,
        worker_l1_size,
        dram_view_size,
        max_cb_count,
        l1_allocation_alignment_bytes,
        True,
        2,
        2,
        2,
    )
    return tvm.IRModule({"main": mod["main"]}, global_infos={
        "tl.tt_hardware_model": [hardware_model]
    })


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


def _source_emitter_schema_names():
    pattern_table = tvm.get_global_func("tl.BlackholeTileComputePatternTable")()
    return {
        str(pattern["source_emitter"])
        for pattern in pattern_table
        if str(pattern["source_emitter"])
        and str(pattern["source_emitter_category"]) != "none"
    }


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
    tensor_placement_intents=None,
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
        list(plan.tensor_placement_intents)
        if tensor_placement_intents is None
        else tensor_placement_intents,
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
    tensor_memory_config_plans=None,
    op_sharding_contracts=None,
    placement_resolution_plans=None,
    reshard_plans=None,
    compute_op_plans=None,
    cb_plans=None,
    live_form_plans=None,
    materialization_plans=None,
    consumer_binding_plans=None,
    exact_cb_virtual_values=None,
    exact_cb_use_events=None,
    exact_cb_live_intervals=None,
    exact_cb_allocations=None,
    exact_cb_release_events=None,
    resource_demands=None,
    resource_pressure_reports=None,
    core_groups=None,
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
        list(program.tensor_memory_config_plans)
        if tensor_memory_config_plans is None
        else tensor_memory_config_plans,
        list(program.op_sharding_contracts)
        if op_sharding_contracts is None
        else op_sharding_contracts,
        list(program.placement_resolution_plans)
        if placement_resolution_plans is None
        else placement_resolution_plans,
        list(program.reshard_plans) if reshard_plans is None else reshard_plans,
        list(program.block_plans),
        list(program.kernel_plans),
        list(program.compute_op_plans) if compute_op_plans is None else compute_op_plans,
        list(program.transport_plans),
        list(program.sync_plans),
        list(program.abi_plans),
        list(program.execution_plans),
        list(program.kernels) if kernels is None else kernels,
        list(program.core_groups) if core_groups is None else core_groups,
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
        list(program.exact_cb_virtual_values)
        if exact_cb_virtual_values is None
        else exact_cb_virtual_values,
        list(program.exact_cb_use_events)
        if exact_cb_use_events is None
        else exact_cb_use_events,
        list(program.exact_cb_live_intervals)
        if exact_cb_live_intervals is None
        else exact_cb_live_intervals,
        list(program.exact_cb_allocations)
        if exact_cb_allocations is None
        else exact_cb_allocations,
        list(program.exact_cb_release_events)
        if exact_cb_release_events is None
        else exact_cb_release_events,
        list(program.resource_demands)
        if resource_demands is None
        else resource_demands,
        list(program.resource_pressure_reports)
        if resource_pressure_reports is None
        else resource_pressure_reports,
    )


def _rebuild_tt_core_group(
    core_group,
    *,
    logical_grid_x=None,
    logical_grid_y=None,
    linearization=None,
    physical_cores=None,
    work_packets=None,
):
    make_tt_core_group = tvm.get_global_func("tl.TTCoreGroup")
    return make_tt_core_group(
        str(core_group.name),
        int(core_group.logical_grid_x) if logical_grid_x is None else logical_grid_x,
        int(core_group.logical_grid_y) if logical_grid_y is None else logical_grid_y,
        str(core_group.linearization) if linearization is None else linearization,
        list(core_group.physical_cores) if physical_cores is None else physical_cores,
        list(core_group.work_packets) if work_packets is None else work_packets,
    )


def _rebuild_tt_buffer_distribution_plan(
    plan,
    *,
    distribution_kind=None,
    layout=None,
    memory_space=None,
    page_size_bytes=None,
    shard_shape=None,
    shard_grid_shape=None,
    sharding_strategy=None,
    shard_orientation=None,
    source_buffer=None,
    source_region_kind=None,
    source_region_shape=None,
    logical_index_mapping=None,
    core_local_address_mapping=None,
    host_visibility=None,
    attached_core_group=None,
    attached_core_group_index=None,
):
    make_plan = tvm.get_global_func("tl.TTBufferDistributionPlan")
    return make_plan(
        str(plan.name),
        str(plan.buffer),
        str(plan.mesh_plan),
        int(plan.mesh_plan_index),
        str(plan.distribution_kind)
        if distribution_kind is None
        else distribution_kind,
        str(plan.layout) if layout is None else layout,
        str(plan.memory_space) if memory_space is None else memory_space,
        int(plan.page_size_bytes) if page_size_bytes is None else page_size_bytes,
        list(plan.shard_shape) if shard_shape is None else shard_shape,
        list(plan.shard_grid_shape)
        if shard_grid_shape is None
        else shard_grid_shape,
        str(plan.sharding_strategy)
        if sharding_strategy is None
        else sharding_strategy,
        str(plan.shard_orientation) if shard_orientation is None else shard_orientation,
        str(plan.source_buffer) if source_buffer is None else source_buffer,
        str(plan.source_region_kind)
        if source_region_kind is None
        else source_region_kind,
        list(plan.source_region_shape)
        if source_region_shape is None
        else source_region_shape,
        str(plan.logical_index_mapping)
        if logical_index_mapping is None
        else logical_index_mapping,
        str(plan.core_local_address_mapping)
        if core_local_address_mapping is None
        else core_local_address_mapping,
        str(plan.host_visibility) if host_visibility is None else host_visibility,
        str(plan.attached_core_group)
        if attached_core_group is None
        else attached_core_group,
        int(plan.attached_core_group_index)
        if attached_core_group_index is None
        else attached_core_group_index,
        list(plan.logical_shape),
        list(plan.local_shape),
        plan.thread_extent,
        plan.replicate_extent,
        list(plan.inverse_logical_index_vars),
        list(plan.inverse_logical_index_exprs),
        str(plan.spatial_layout),
        str(plan.spatial_distribution_kind),
        str(plan.abi_layout),
        str(plan.abi_memory_space),
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
        int(plan.tile_compute_dag_node_id),
        str(plan.tile_compute_source_emitter),
        str(plan.tile_compute_materialization_policy),
        int(plan.tile_compute_fanout_use_count),
        str(plan.tile_compute_fanout_policy),
    )


def _rebuild_tt_op_sharding_contract(
    contract,
    *,
    memory_config_plan=None,
    memory_config_plan_index=None,
    accepted_memory_layouts=None,
    accepted_buffer_types=None,
    accepted_sharding_strategies=None,
    required_shard_orientation=None,
    output_policy=None,
    can_produce_output_placement=None,
):
    make_contract = tvm.get_global_func("tl.TTOpShardingContract")
    return make_contract(
        str(contract.name),
        str(contract.compute_op_plan),
        int(contract.compute_op_plan_index),
        str(contract.operation_name),
        str(contract.op_kind),
        str(contract.operand_role),
        str(contract.operand_buffer),
        str(contract.operand_host_buffer),
        str(contract.memory_config_plan)
        if memory_config_plan is None
        else memory_config_plan,
        int(contract.memory_config_plan_index)
        if memory_config_plan_index is None
        else memory_config_plan_index,
        list(contract.accepted_memory_layouts)
        if accepted_memory_layouts is None
        else accepted_memory_layouts,
        list(contract.accepted_buffer_types)
        if accepted_buffer_types is None
        else accepted_buffer_types,
        list(contract.accepted_sharding_strategies)
        if accepted_sharding_strategies is None
        else accepted_sharding_strategies,
        str(contract.required_shard_orientation)
        if required_shard_orientation is None
        else required_shard_orientation,
        str(contract.output_policy) if output_policy is None else output_policy,
        bool(contract.may_request_input_conversion),
        bool(contract.can_produce_output_placement)
        if can_produce_output_placement is None
        else can_produce_output_placement,
        bool(contract.direct_external_write_allowed),
        str(contract.reject_reason),
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
    source_live_form=None,
    live_value_edge=None,
    live_value_edge_index=None,
    accepts_distributed_slice=None,
    requires_full_logical_tile=None,
):
    make_tt_consumer_binding_plan = tvm.get_global_func("tl.TTConsumerBindingPlan")
    return make_tt_consumer_binding_plan(
        str(plan.name),
        str(plan.consumer_kernel),
        str(plan.consumer_op_kind),
        str(plan.source_live_form) if source_live_form is None else source_live_form,
        str(plan.live_value_edge) if live_value_edge is None else live_value_edge,
        int(plan.live_value_edge_index)
        if live_value_edge_index is None
        else live_value_edge_index,
        bool(plan.accepts_distributed_slice)
        if accepts_distributed_slice is None
        else accepts_distributed_slice,
        bool(plan.requires_full_logical_tile)
        if requires_full_logical_tile is None
        else requires_full_logical_tile,
        int(plan.abi_plan_index),
        str(plan.target_buffer),
        str(plan.materialization_plan),
    )


def _make_exact_cb_lifecycle_records(program):
    make_virtual_value = tvm.get_global_func("tl.TTExactCBVirtualValue")
    make_use_event = tvm.get_global_func("tl.TTExactCBUseEvent")
    make_interval = tvm.get_global_func("tl.TTExactCBLiveInterval")
    make_allocation = tvm.get_global_func("tl.TTExactCBAllocation")
    make_release = tvm.get_global_func("tl.TTExactCBReleaseEvent")
    live_form_index, live_form = next(
        (index, plan)
        for index, plan in enumerate(program.live_form_plans)
        if str(plan.physical_form) == "cb_materialized_tile"
    )
    cb_plan_index, cb_plan = next(
        (index, plan)
        for index, plan in enumerate(program.cb_plans)
        if str(plan.flow_class) == "state" and int(plan.num_pages) > 0
    )

    virtual_value = make_virtual_value(
        f"{live_form.logical_value}.loop_exit.value",
        str(live_form.logical_value),
        str(live_form.name),
        live_form_index,
        str(live_form.producer_kernel),
        "loop_exit_value",
        "loop_carried",
        "loop_exit",
        int(cb_plan.num_pages),
        int(cb_plan.page_size_bytes),
        str(cb_plan.data_format),
    )
    use_event = make_use_event(
        f"{live_form.logical_value}.final_consume",
        str(virtual_value.name),
        0,
        "compute",
        "mul_tiles_bcast_cols",
        "lhs",
        42,
        True,
        "borrow",
    )
    interval = make_interval(
        f"{live_form.logical_value}.loop_exit.interval",
        str(virtual_value.name),
        0,
        17,
        42,
        True,
        True,
        True,
        "intermediate_exact_cb",
    )
    allocation = make_allocation(
        f"{live_form.logical_value}.loop_exit.alloc",
        str(virtual_value.name),
        0,
        str(cb_plan.name),
        cb_plan_index,
        int(cb_plan.cb_id),
        1,
        43,
        "last_use",
    )
    release = make_release(
        f"{live_form.logical_value}.loop_exit.release",
        str(allocation.name),
        0,
        str(cb_plan.name),
        cb_plan_index,
        43,
        1,
        "last_use",
    )
    return virtual_value, use_event, interval, allocation, release


def _rebuild_tt_reshard_plan(
    plan,
    *,
    target_memory_config_plan_index=None,
):
    make_tt_reshard_plan = tvm.get_global_func("tl.TTReshardPlan")
    return make_tt_reshard_plan(
        str(plan.name),
        str(plan.source_value),
        str(plan.target_value),
        str(plan.source_memory_config_plan),
        int(plan.source_memory_config_plan_index),
        str(plan.target_memory_config_plan),
        int(plan.target_memory_config_plan_index)
        if target_memory_config_plan_index is None
        else target_memory_config_plan_index,
        str(plan.conversion_kind),
        str(plan.source_region_kind),
        list(plan.source_region_shape),
        str(plan.materialization_plan),
        int(plan.materialization_plan_index),
        str(plan.materialization_protocol),
        list(plan.required_cb_plan_indices),
        list(plan.required_sync_plan_indices),
        str(plan.scheduling_kind),
        str(plan.inserted_by),
        str(plan.admission_status),
        str(plan.unsupported_reason),
    )


def _rebuild_tt_resource_pressure_report(
    report,
    *,
    tile_compute_unsupported_reasons=None,
    unsupported_reasons=None,
    required_materializations=None,
    per_core_cb_id_pressure=None,
    per_core_cb_l1_bytes=None,
    per_core_l1_buffer_bytes=None,
    max_simultaneous_l1_bytes=None,
    cb_id_limit=None,
    worker_l1_budget_bytes=None,
    l1_alignment_bytes=None,
    per_core_cb_l1_aligned_bytes=None,
    l1_alignment_waste_bytes=None,
):
    make_report = tvm.get_global_func("tl.TTResourcePressureReport")
    return make_report(
        str(report.name),
        str(report.kernel_name),
        str(report.core_group),
        int(report.core_group_index),
        list(report.tile_compute_unsupported_reasons)
        if tile_compute_unsupported_reasons is None
        else tile_compute_unsupported_reasons,
        list(report.required_materializations)
        if required_materializations is None
        else required_materializations,
        int(report.per_core_cb_id_pressure)
        if per_core_cb_id_pressure is None
        else per_core_cb_id_pressure,
        int(report.per_core_cb_l1_bytes)
        if per_core_cb_l1_bytes is None
        else per_core_cb_l1_bytes,
        int(report.per_core_l1_buffer_bytes)
        if per_core_l1_buffer_bytes is None
        else per_core_l1_buffer_bytes,
        int(report.max_simultaneous_l1_bytes)
        if max_simultaneous_l1_bytes is None
        else max_simultaneous_l1_bytes,
        int(report.cb_id_limit) if cb_id_limit is None else cb_id_limit,
        int(report.worker_l1_budget_bytes)
        if worker_l1_budget_bytes is None
        else worker_l1_budget_bytes,
        int(report.l1_alignment_bytes)
        if l1_alignment_bytes is None
        else l1_alignment_bytes,
        int(report.per_core_cb_l1_aligned_bytes)
        if per_core_cb_l1_aligned_bytes is None
        else per_core_cb_l1_aligned_bytes,
        int(report.l1_alignment_waste_bytes)
        if l1_alignment_waste_bytes is None
        else l1_alignment_waste_bytes,
        str(report.core_grid_requirement),
        str(report.dram_view_requirement),
        list(report.unsupported_reasons)
        if unsupported_reasons is None
        else unsupported_reasons,
    )


def _assert_no_tt_plan_payload_surface(tt_program):
    plan_groups = (
        tt_program.mesh_plans,
        tt_program.buffer_distribution_plans,
        tt_program.tensor_memory_config_plans,
        tt_program.op_sharding_contracts,
        tt_program.placement_resolution_plans,
        tt_program.reshard_plans,
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
        tt_program.resource_demands,
        tt_program.resource_pressure_reports,
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


def test_modern_cpp_audit_blackhole_scalar_and_tensor_boundaries_fail_closed():
    module_source = (
        REPO_ROOT / "tilelang_repo/src/target/blackhole_module.cc"
    ).read_text()
    codegen_source = (
        REPO_ROOT / "tilelang_repo/src/target/codegen_blackhole.cc"
    ).read_text()

    assert "*reinterpret_cast<uint32_t*>(&f)" not in module_source
    assert "reinterpret_cast<\" << dtype << \"*>(&\" << param_name" not in codegen_source
    assert "RequireCompactRowMajorLayout" in module_source
    assert 'RequireCompactRowMajorLayout(tensor, "input transfer"' in module_source
    assert 'RequireCompactRowMajorLayout(binding.tensor, "output transfer"' in module_source
    assert "unsupported scalar argument" in module_source


def test_modern_cpp_audit_blackhole_runtime_leaf_readers_are_not_fail_open():
    source = (REPO_ROOT / "tilelang_repo/src/target/rt_mod_blackhole.cc").read_text()

    assert "value_or(ffi::Map<ffi::String, ffi::Any>())" not in source
    assert "plan.physical_cores.push_back(PhysicalCore{});" not in source
    assert "RequireMap" in source
    assert "RequireExecutableArrayField" in source


def test_modern_cpp_audit_blackhole_serialization_contract_is_real():
    header = (REPO_ROOT / "tilelang_repo/src/target/blackhole_module.h").read_text()
    source = (REPO_ROOT / "tilelang_repo/src/target/blackhole_module.cc").read_text()

    assert "opaque imported runtime modules" in header
    assert "kBinarySerializable | ffi::Module::kRunnable" in header
    assert "tilelang.blackhole.module.v1" in source
    assert "WriteExecutableSpecMap(stream, fmap_)" in source
    assert "ReadExecutableSpecMap(stream)" in source
    assert "ffi.Module.load_from_bytes.blackhole" in source
    assert "BlackholeModule SaveToBytes not implemented" not in source
    assert "BlackholeModule WriteToFile not implemented" in source
    assert 'return ffi::Bytes("")' not in source


def test_modern_cpp_audit_blackhole_resource_canonicalizer_has_no_var_name_fallback():
    source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc"
    ).read_text()

    assert "name_to_new_var_" not in source
    assert "Name-based fallback" not in source
    assert "VisitExpr_(const VarNode* op)" in source


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


def test_exact_cb_release_source_does_not_keep_local_last_use_fallback():
    source_paths = [
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.h",
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_exact_cb.cc",
    ]
    hits = _source_tree_rg(
        r"ShouldReleaseBorrowedExactInputAfterUse|"
        r"should_release\s*=|"
        r"RecordExactCBUseAndMaybeRelease\([^;]+should_release",
        *source_paths,
    )
    assert hits == []


def test_exact_cb_materialization_pop_requires_typed_release_event():
    source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_exact_cb.cc"
    ).read_text()

    assert "blackhole_cb_pop_front(), {IntImm32(cb_value.cb_id)" not in source


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
        "sub_tiles",
        "mul_tiles",
        "mul_tiles_bcast_cols",
        "add_tiles_bcast_cols",
        "exp2_tile",
        "recip_tile",
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
    assert "ConsumeTileComputeDAGLoweringDecision(operation)" in tile_compute_source
    assert "BlackholeTileComputeSourceProjection::Emit(this, op, covering)" in tile_compute_source
    assert "SelectBlackholeTileComputeCovering(operation)" not in tile_compute_source
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


def test_tile_compute_dag_feeds_typed_resource_pressure_report():
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
    tt_program = mod["main"].attrs["tl.tt_program"]

    assert tt_program.resource_demands
    assert tt_program.resource_pressure_reports
    assert all(not hasattr(plan, "payload") for plan in tt_program.resource_demands)
    assert all(
        not hasattr(plan, "payload") for plan in tt_program.resource_pressure_reports
    )

    demand = tt_program.resource_demands[0]
    fanout_demands = list(demand.tile_compute_fanout_demands)
    materialization_demands = list(demand.tile_compute_materialization_demands)
    assert fanout_demands
    assert materialization_demands
    assert all(int(demand.use_count) >= 2 for demand in fanout_demands)
    assert {
        "share_live_value",
        "materialize_before_cross_event_use",
    } >= {str(demand.policy) for demand in fanout_demands}

    report = tt_program.resource_pressure_reports[0]
    assert report.required_materializations
    assert not list(report.tile_compute_unsupported_reasons)
    assert int(report.per_core_cb_id_pressure) == len(tt_program.cb_plans)
    assert int(report.per_core_cb_l1_bytes) == sum(
        int(plan.num_pages) * int(plan.page_size_bytes)
        for plan in tt_program.cb_plans
    )


def test_tile_compute_dag_decisions_drive_typed_compute_lower_plan():
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
    tt_program = mod["main"].attrs["tl.tt_program"]
    assert tt_program.compute_op_plans

    dag_compute_plans = [
        plan
        for plan in tt_program.compute_op_plans
        if int(plan.tile_compute_dag_node_id) >= 0
    ]
    assert dag_compute_plans
    assert all(str(plan.tile_compute_source_emitter) for plan in dag_compute_plans)
    assert {
        "none",
        "materialization_boundary_required_when_cross_phase",
        "live_form_solver_required_for_cross_event_use",
    } >= {str(plan.tile_compute_materialization_policy) for plan in dag_compute_plans}

    materialization_demand_nodes = {
        int(demand.node_id)
        for resource_demand in tt_program.resource_demands
        for demand in resource_demand.tile_compute_materialization_demands
    }
    materialized_compute_nodes = {
        int(plan.tile_compute_dag_node_id)
        for plan in dag_compute_plans
        if str(plan.tile_compute_materialization_policy) != "none"
    }
    assert materialized_compute_nodes
    assert materialization_demand_nodes
    assert materialized_compute_nodes <= materialization_demand_nodes


def test_executable_projection_carries_dag_driven_compute_lower_plan():
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
    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]

    compute_plans = list(executable["compute_op_plans"])
    dag_compute_plans = [
        plan for plan in compute_plans if int(plan["tile_compute_dag_node_id"]) >= 0
    ]
    assert dag_compute_plans
    assert all(str(plan["tile_compute_source_emitter"]) for plan in dag_compute_plans)
    assert all("tile_compute_materialization_policy" in plan for plan in dag_compute_plans)


def test_validate_tt_program_consumes_typed_resource_pressure_report():
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
    assert tt_program.resource_pressure_reports

    broken_program = _rebuild_tt_program(
        tt_program,
        resource_pressure_reports=[],
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", broken_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="ResourcePressureReport"):
        tilelang.transform.ValidateTTProgram()(broken)

    orphan_report_program = _rebuild_tt_program(
        tt_program,
        resource_demands=[],
    )
    orphan_report = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", orphan_report_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="ResourcePressureReport requires matching"):
        tilelang.transform.ValidateTTProgram()(orphan_report)

    report = tt_program.resource_pressure_reports[0]
    rejected_report = _rebuild_tt_resource_pressure_report(
        report,
        tile_compute_unsupported_reasons=[
            "forced tile-compute resource-pressure rejection"
        ],
        unsupported_reasons=["forced tile-compute resource-pressure rejection"],
    )
    rejected_program = _rebuild_tt_program(
        tt_program,
        resource_pressure_reports=[rejected_report],
    )
    rejected = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", rejected_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="ResourcePressureReport unsupported"):
        tilelang.transform.ValidateTTProgram()(rejected)


def test_validate_tt_program_consumes_exact_cb_lifecycle_records():
    mod = _prepare_blackhole_tt_program_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            4,
            64,
            32,
            False,
            block_M=32,
            block_N=32,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    lifecycle_records = _make_exact_cb_lifecycle_records(tt_program)

    program_with_lifecycle = _rebuild_tt_program(
        tt_program,
        exact_cb_virtual_values=[lifecycle_records[0]],
        exact_cb_use_events=[lifecycle_records[1]],
        exact_cb_live_intervals=[lifecycle_records[2]],
        exact_cb_allocations=[lifecycle_records[3]],
        exact_cb_release_events=[lifecycle_records[4]],
    )
    valid = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", program_with_lifecycle)},
        global_infos=mod.global_infos,
    )
    tilelang.transform.ValidateTTProgram()(valid)
    projected = tilelang.transform.MaterializeBlackholeExecutable()(valid)
    executable = projected["main"].attrs["tl.blackhole_executable"]
    assert "exact_cb_virtual_values" in executable
    assert "exact_cb_use_events" in executable
    assert "exact_cb_live_intervals" in executable
    assert "exact_cb_allocations" in executable
    assert "exact_cb_release_events" in executable
    assert str(executable["exact_cb_release_events"][0]["reason"]) == "last_use"

    broken_release = _make_exact_cb_lifecycle_records(tt_program)[4]
    make_release = tvm.get_global_func("tl.TTExactCBReleaseEvent")
    premature_release = make_release(
        str(broken_release.name),
        str(broken_release.allocation),
        int(broken_release.allocation_index),
        str(broken_release.cb_plan),
        int(broken_release.cb_plan_index),
        41,
        int(broken_release.page_count),
        str(broken_release.reason),
    )
    broken_program = _rebuild_tt_program(
        tt_program,
        exact_cb_virtual_values=[lifecycle_records[0]],
        exact_cb_use_events=[lifecycle_records[1]],
        exact_cb_live_intervals=[lifecycle_records[2]],
        exact_cb_allocations=[lifecycle_records[3]],
        exact_cb_release_events=[premature_release],
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", broken_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="TTExactCBReleaseEvent.*last use"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_validate_tt_program_rejects_loop_carried_exact_cb_without_exit_evidence():
    mod = _prepare_blackhole_tt_program_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            4,
            64,
            32,
            False,
            block_M=32,
            block_N=32,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    virtual_value, use_event, interval, allocation, release = (
        _make_exact_cb_lifecycle_records(tt_program)
    )
    make_interval = tvm.get_global_func("tl.TTExactCBLiveInterval")
    missing_exit_interval = make_interval(
        str(interval.name),
        str(interval.virtual_value),
        int(interval.virtual_value_index),
        int(interval.begin_point),
        int(interval.end_point),
        True,
        False,
        True,
        str(interval.interference_class),
    )
    broken_program = _rebuild_tt_program(
        tt_program,
        exact_cb_virtual_values=[virtual_value],
        exact_cb_use_events=[use_event],
        exact_cb_live_intervals=[missing_exit_interval],
        exact_cb_allocations=[allocation],
        exact_cb_release_events=[release],
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", broken_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="loop-carried.*live.*out"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_validate_tt_program_rejects_interfering_exact_cb_intervals_sharing_cb():
    mod = _prepare_blackhole_tt_program_module(
        mha_example.flashattn.jit_impl.get_tir(
            1,
            4,
            64,
            32,
            False,
            block_M=32,
            block_N=32,
            num_stages=1,
            threads=128,
        )
    )
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    virtual_value, use_event, interval, allocation, release = (
        _make_exact_cb_lifecycle_records(tt_program)
    )
    make_virtual_value = tvm.get_global_func("tl.TTExactCBVirtualValue")
    make_use_event = tvm.get_global_func("tl.TTExactCBUseEvent")
    make_interval = tvm.get_global_func("tl.TTExactCBLiveInterval")
    make_allocation = tvm.get_global_func("tl.TTExactCBAllocation")
    make_release = tvm.get_global_func("tl.TTExactCBReleaseEvent")

    other_value = make_virtual_value(
        f"{virtual_value.name}.other",
        str(virtual_value.logical_value),
        str(virtual_value.live_form),
        int(virtual_value.live_form_index),
        str(virtual_value.producer_kernel),
        "other_loop_exit_value",
        str(virtual_value.event_lifetime_kind),
        str(virtual_value.loop_role),
        int(virtual_value.num_pages),
        int(virtual_value.page_size_bytes),
        str(virtual_value.data_format),
    )
    other_use = make_use_event(
        f"{other_value.name}.consume",
        str(other_value.name),
        1,
        str(use_event.consumer_kernel),
        str(use_event.consumer_event),
        str(use_event.operand_role),
        44,
        bool(use_event.requires_full_logical_tile),
        str(use_event.borrow_kind),
    )
    other_interval = make_interval(
        f"{other_value.name}.interval",
        str(other_value.name),
        1,
        18,
        44,
        bool(interval.live_in),
        bool(interval.live_out),
        bool(interval.loop_carried),
        str(interval.interference_class),
    )
    other_allocation = make_allocation(
        f"{other_value.name}.alloc",
        str(other_value.name),
        1,
        str(allocation.cb_plan),
        int(allocation.cb_plan_index),
        int(allocation.physical_cb_id),
        int(allocation.page_count),
        45,
        str(allocation.release_reason),
    )
    other_release = make_release(
        f"{other_value.name}.release",
        str(other_allocation.name),
        1,
        str(release.cb_plan),
        int(release.cb_plan_index),
        45,
        int(release.page_count),
        str(release.reason),
    )
    broken_program = _rebuild_tt_program(
        tt_program,
        exact_cb_virtual_values=[virtual_value, other_value],
        exact_cb_use_events=[use_event, other_use],
        exact_cb_live_intervals=[interval, other_interval],
        exact_cb_allocations=[allocation, other_allocation],
        exact_cb_release_events=[release, other_release],
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", broken_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="interfering.*physical CB"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_executable_projection_carries_resource_pressure_report():
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
    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]

    assert "resource_pressure_reports" in executable
    reports = list(executable["resource_pressure_reports"])
    assert reports
    assert reports[0]["required_materializations"]
    assert not list(reports[0]["tile_compute_unsupported_reasons"])
    assert int(reports[0]["cb_id_limit"]) > 0
    assert int(reports[0]["worker_l1_budget_bytes"]) > 0
    assert int(reports[0]["l1_alignment_bytes"]) > 0
    assert int(reports[0]["max_simultaneous_l1_bytes"]) == (
        int(reports[0]["per_core_cb_l1_aligned_bytes"])
        + int(reports[0]["per_core_l1_buffer_bytes"])
    )


def test_resource_pressure_report_carries_hardware_cb_l1_admission_facts():
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
    tt_program = mod["main"].attrs["tl.tt_program"]
    report = tt_program.resource_pressure_reports[0]
    hardware_model = mod.global_infos["tl.tt_hardware_model"][0]

    assert int(hardware_model.max_cb_count) == 64

    raw_cb_l1 = sum(
        int(plan.num_pages) * int(plan.page_size_bytes)
        for plan in tt_program.cb_plans
    )
    assert int(report.cb_id_limit) == int(hardware_model.max_cb_count)
    assert int(report.worker_l1_budget_bytes) == int(hardware_model.worker_l1_size)
    assert int(report.l1_alignment_bytes) == int(
        hardware_model.l1_allocation_alignment_bytes
    )
    assert int(report.per_core_cb_l1_bytes) == raw_cb_l1
    assert int(report.per_core_cb_l1_aligned_bytes) >= raw_cb_l1
    assert int(report.l1_alignment_waste_bytes) == (
        int(report.per_core_cb_l1_aligned_bytes) - raw_cb_l1
    )
    assert int(report.max_simultaneous_l1_bytes) == (
        int(report.per_core_cb_l1_aligned_bytes)
        + int(report.per_core_l1_buffer_bytes)
    )
    assert int(report.per_core_cb_id_pressure) <= int(report.cb_id_limit)
    assert int(report.max_simultaneous_l1_bytes) <= int(
        report.worker_l1_budget_bytes
    )


def test_plan_tt_blocks_uses_hardware_model_for_core_group_capacity():
    mod = _prepare_blackhole_phase_b_module(grid_indexed_staged_copy_kernel(3, 3))
    mod = _with_test_hardware_model(
        mod,
        logical_worker_grid_x=2,
        logical_worker_grid_y=2,
        functional_worker_count=4,
    )

    mod = tilelang.transform.PlanTTBlocks()(mod)
    tt_program = mod["main"].attrs["tl.tt_program"]
    core_group = tt_program.core_groups[0]

    assert int(core_group.logical_grid_x) == 3
    assert int(core_group.logical_grid_y) == 3
    assert len(core_group.physical_cores) == 4
    assert len(core_group.work_packets) == 4
    assert {
        (int(core["core_x"]), int(core["core_y"]))
        for core in core_group.physical_cores
    } == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert sum(int(packet["work_count"]) for packet in core_group.work_packets) == 9


def test_validate_tt_program_rejects_core_group_outside_hardware_grid():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(1, 1))
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    mod = _with_test_hardware_model(
        mod,
        logical_worker_grid_x=1,
        logical_worker_grid_y=1,
        functional_worker_count=1,
    )

    core_group = tt_program.core_groups[0]
    invalid_core = {"core_x": 1, "core_y": 0}
    invalid_packet = {"core_x": 1, "core_y": 0, "work_offset": 0, "work_count": 1}
    invalid_program = _rebuild_tt_program(
        tt_program,
        core_groups=[
            _rebuild_tt_core_group(
                core_group,
                physical_cores=[invalid_core],
                work_packets=[invalid_packet],
            )
        ],
    )
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="TTCoreGroup.*hardware"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_validate_tt_program_rejects_cb_id_pressure_over_hardware_limit():
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
    report = tt_program.resource_pressure_reports[0]
    rejected_report = _rebuild_tt_resource_pressure_report(
        report,
        per_core_cb_id_pressure=int(report.cb_id_limit) + 1,
    )
    rejected_program = _rebuild_tt_program(
        tt_program,
        resource_pressure_reports=[rejected_report],
    )
    rejected = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", rejected_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="CB id pressure exceeds hardware limit"):
        tilelang.transform.ValidateTTProgram()(rejected)


def test_validate_tt_program_rejects_l1_pressure_over_worker_budget():
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
    report = tt_program.resource_pressure_reports[0]
    rejected_report = _rebuild_tt_resource_pressure_report(
        report,
        max_simultaneous_l1_bytes=int(report.worker_l1_budget_bytes) + 1,
    )
    rejected_program = _rebuild_tt_program(
        tt_program,
        resource_pressure_reports=[rejected_report],
    )
    rejected = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", rejected_program)},
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="L1 pressure exceeds worker budget"):
        tilelang.transform.ValidateTTProgram()(rejected)


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


def test_tile_compute_source_emission_schema_covers_pattern_table():
    pattern_table = tvm.get_global_func("tl.BlackholeTileComputePatternTable")()
    pattern_emitters = {
        str(pattern["source_emitter"])
        for pattern in pattern_table
        if str(pattern["source_emitter"])
    }
    schema_emitters = _source_emitter_schema_names()

    assert pattern_emitters <= schema_emitters
    assert "none" not in schema_emitters
    for pattern in pattern_table:
        category = str(pattern["source_emitter_category"])
        if category in {"binary", "broadcast_cols_binary", "unary"}:
            assert str(pattern["source_init_builtin"]).startswith("tl.blackhole.")
            assert str(pattern["source_tile_builtin"]).startswith("tl.blackhole.")


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


def test_blackhole_tile_compute_normalizes_before_lower_tile_op():
    mod = _prepare_blackhole_layout_inferred_module(_elementwise_add_kernel())
    normalized = tilelang.transform.NormalizeBlackholeTileCompute()(mod)
    normalized_ops = _collect_blackhole_tile_compute_operations(normalized["main"])

    assert "add_tiles" in normalized_ops

    with Target("blackhole"):
        lowered = tilelang.transform.LowerTileOp()(normalized)
    lowered_ops = _collect_blackhole_tile_compute_operations(lowered["main"])

    assert "add_tiles" in lowered_ops


def test_blackhole_frontend_rejects_compute_residue_after_normalization():
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole tile compute normalization left scalar compute residue",
    ):
        _lower_blackhole_frontend(_three_input_add_residue_kernel())


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


def test_spatial_plan_derives_layout_specs_for_preserved_flash_leaf_compute():
    mod = _prepare_blackhole_phase_b_module(
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
    plan = mod["main"].attrs["tl.spatial_plan"]
    layouts = {str(layout.subject): layout for layout in plan.layout_specs}

    for name in ("acc_s", "acc_s_cast", "acc_o"):
        layout = layouts[name]
        assert tuple(int(dim) for dim in layout.logical_shape) == (128, 128)
        assert tuple(int(dim) for dim in layout.local_shape) == (128,)
        assert int(layout.thread_extent) == 128
        assert int(layout.replicate_extent) == 1
        assert len(layout.inverse_logical_index_exprs) == 3

    for name in ("scores_max", "scores_max_prev", "scores_scale", "scores_sum", "logsum"):
        layout = layouts[name]
        assert tuple(int(dim) for dim in layout.logical_shape) == (128,)
        assert tuple(int(dim) for dim in layout.local_shape) == (1,)
        assert int(layout.thread_extent) == 128
        assert int(layout.replicate_extent) == 1
        assert len(layout.inverse_logical_index_exprs) == 2


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


def test_blackhole_frontend_tile_compute_normalization_rejects_composite_payloads():
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
    calls = []

    def visit(expr):
        if isinstance(expr, tvm.tir.Call):
            op = expr.op
            if hasattr(op, "name") and op.name == "tl.tileop.blackhole_compute":
                if expr.args and isinstance(expr.args[0], tvm.tir.StringImm):
                    calls.append(expr)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert calls
    for call in calls:
        operation = str(call.args[0].value)
        if operation in {"exp2_tile", "mul_tiles_bcast_cols"}:
            assert not isinstance(call.args[1], tvm.tir.StringImm)


def test_blackhole_frontend_decomposes_row_division_to_recip_leaf():
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

    assert "recip_tile" in operations


def test_lower_tile_op_has_single_blackhole_tile_compute_normalizer_surface():
    lower_tile_source = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_tile_op.cc"
    ).read_text()
    normalizer_source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_normalizer.cc"
    ).read_text()
    phase_source = (REPO_ROOT / "tilelang_repo/tilelang/engine/phase.py").read_text()

    normalize_index = phase_source.index("NormalizeBlackholeTileCompute()(mod)")
    validate_index = phase_source.index("ValidateBlackholeTileComputeNormalized()(mod)")
    lower_tile_index = phase_source.index("LowerTileOp()(mod)")

    assert "MakeBlackholeTileComputeCall(" not in lower_tile_source
    assert "blackhole_tile_compute_normalizer.h" not in lower_tile_source
    assert "NormalizeBlackholeTileComputeLoop(" not in lower_tile_source
    assert "TryNormalizeBlackholeTileComputeStore(" not in lower_tile_source
    assert "TryNormalizeBlackholeTileComputeLoop(" not in lower_tile_source
    assert normalize_index < validate_index < lower_tile_index
    assert phase_source.count("NormalizeBlackholeTileCompute()(mod)") == 1
    assert phase_source.count("ValidateBlackholeTileComputeNormalized()(mod)") == 1
    assert "class TileComputeIRBuilder" in normalizer_source
    assert "NormalizeBlackholeTileComputeStore(" in normalizer_source
    assert "struct TileComputeRewriteRule" not in normalizer_source
    assert "GetBlackholeTileComputeRewriteRules" not in normalizer_source
    assert "for (const TileComputeRewriteRule& rule" not in normalizer_source
    assert "TryNormalizeBlackholeTileComputeStore(" not in normalizer_source

    helper_defs = re.findall(
        r"\bStmt\s+MakeBlackholeTileComputeCall\s*\(", normalizer_source
    )
    store_defs = re.findall(
        r"\bStmt\s+TryNormalize(?:BlackholeTileCompute)?Store\s*\(",
        normalizer_source,
    )
    loop_defs = re.findall(
        r"\bStmt\s+NormalizeBlackholeTileComputeLoop\s*\(",
        normalizer_source,
    )

    assert helper_defs == ["Stmt MakeBlackholeTileComputeCall("]
    assert store_defs == []
    assert len(loop_defs) == 1
    assert "TryNormalizeBlackholeTileComputeLoop(" not in normalizer_source


def test_tile_compute_source_projection_is_not_declared_on_plan_tt_kernel_abi():
    planner_header = (
        REPO_ROOT / "tilelang_repo/src/transform/lower_blackhole_ops.h"
    ).read_text()
    pattern_header = (
        REPO_ROOT
        / "tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.h"
    ).read_text()
    tile_compute_source = (
        REPO_ROOT
        / "tilelang_repo/src/transform/lower_blackhole_tile_compute.cc"
    ).read_text()

    assert "TileComputeSourceEmitterHook" not in planner_header
    assert "EmitFillFragmentTileComputeSource" not in planner_header
    assert "EmitRecipTileComputeSource" not in planner_header
    assert "BlackholeTileComputeSourceEmitterCategory" in pattern_header
    assert "source_init_builtin" in pattern_header
    assert "source_tile_builtin" in pattern_header
    assert "BlackholeTileComputeSourceProjection::Emit" in tile_compute_source
    assert "struct Hook" not in tile_compute_source
    assert "BlackholeTileComputeSourceProjection::Hooks" not in tile_compute_source
    assert "EmitAddTiles(" not in tile_compute_source
    assert "EmitMulTiles(" not in tile_compute_source
    assert "EmitExp2(" not in tile_compute_source
    assert "EmitRecip(" not in tile_compute_source
    assert "EmitBinary(" in tile_compute_source
    assert "EmitBroadcastColsBinary(" in tile_compute_source
    assert "EmitUnary(" in tile_compute_source


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


def test_memory_config_annotation_lowers_to_spatial_tensor_placement_intent():
    mod = _prepare_blackhole_phase_b_module(_memory_config_annotation_kernel())
    mod = tilelang.transform.BuildSpatialPlan()(mod)
    spatial_plan = mod["main"].attrs["tl.spatial_plan"]

    intents = {str(intent.subject): intent for intent in spatial_plan.tensor_placement_intents}

    assert str(intents["A"].source) == "user"
    assert str(intents["A"].dsl_origin) == "memory_config_map"
    assert str(intents["A"].memory_space_class) == "DRAM"
    assert str(intents["A"].strategy_class) == "width_sharded"
    assert tuple(int(dim) for dim in intents["A"].shard_grid_shape) == (1, 2)
    assert tuple(int(dim) for dim in intents["A"].shard_shape) == (64, 32)
    assert str(intents["A"].shard_orientation) == "row_major"
    assert not bool(intents["A"].allow_reshard)
    assert bool(intents["A"].hard_requirement)

    assert str(intents["C"].source) == "user"
    assert str(intents["C"].strategy_class) == "interleaved"
    assert str(intents["C"].memory_space_class) == "DRAM"


def test_unannotated_global_buffers_get_explicit_interleaved_dram_intent():
    mod = _prepare_blackhole_phase_b_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = tilelang.transform.BuildSpatialPlan()(mod)
    spatial_plan = mod["main"].attrs["tl.spatial_plan"]

    intents = {str(intent.subject): intent for intent in spatial_plan.tensor_placement_intents}
    assert str(intents["A"].source) == "derived_default"
    assert str(intents["A"].strategy_class) == "interleaved"
    assert str(intents["A"].memory_space_class) == "DRAM"
    assert str(intents["B"].source) == "derived_default"
    assert str(intents["B"].strategy_class) == "interleaved"
    assert str(intents["B"].memory_space_class) == "DRAM"
    assert "A_shared" not in intents


def test_memory_config_python_surface_rejects_invalid_strategy_and_orientation():
    with pytest.raises(ValueError, match="strategy"):
        T.sharded_dram(
            strategy="row_major",
            grid=T.CoreGrid(x=2, y=1),
            shard_shape=(64, 32),
        )

    with pytest.raises(ValueError, match="orientation"):
        T.sharded_l1(
            strategy="height",
            grid=T.CoreGrid(x=1, y=2),
            shard_shape=(32, 64),
            orientation="block",
        )


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
    assert str(distributions["A"].distribution_kind) == "interleaved"
    assert str(distributions["A_shared"].distribution_kind) == "sharded"
    assert str(distributions["B"].distribution_kind) == "interleaved"
    assert str(distributions["A"].memory_space) == "DRAM"
    assert str(distributions["A_shared"].memory_space) == "L1"

    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    assert "mesh_plans" in executable
    assert "buffer_distribution_plans" in executable


def test_plan_tt_abi_uses_hardware_backed_buffer_distribution():
    mod = _prepare_blackhole_phase_b_module(grid_indexed_staged_copy_kernel(3, 3))
    mod = _with_test_hardware_model(
        mod,
        logical_worker_grid_x=2,
        logical_worker_grid_y=2,
        functional_worker_count=4,
        dram_view_count=8,
        worker_l1_size=1572864,
        l1_allocation_alignment_bytes=32,
    )
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)
    mod = tilelang.transform.ValidateTTProgram()(mod)

    tt_program = mod["main"].attrs["tl.tt_program"]
    distributions = {
        str(plan.buffer): plan for plan in tt_program.buffer_distribution_plans
    }
    input_plan = distributions["A"]
    output_plan = distributions["B"]
    l1_plan = distributions["A_shared"]

    assert str(input_plan.distribution_kind) == "interleaved"
    assert str(output_plan.distribution_kind) == "interleaved"
    assert str(input_plan.layout) == "interleaved"
    assert str(input_plan.memory_space) == "DRAM"
    assert int(input_plan.page_size_bytes) == 2048
    assert not list(input_plan.shard_shape)
    assert str(input_plan.attached_core_group) == ""
    assert int(input_plan.attached_core_group_index) == -1

    assert str(l1_plan.distribution_kind) == "sharded"
    assert str(l1_plan.memory_space) == "L1"
    assert int(l1_plan.page_size_bytes) > 0
    assert tuple(int(dim) for dim in l1_plan.shard_grid_shape) == (2, 2)
    assert str(l1_plan.sharding_strategy) == "block"
    assert tuple(int(dim) for dim in l1_plan.shard_shape) == (32, 32)
    assert tuple(int(dim) for dim in l1_plan.source_region_shape) == (32, 32)
    assert str(l1_plan.shard_orientation) == "row_major"
    assert str(l1_plan.source_buffer) == "A"
    assert str(l1_plan.source_region_kind) == "per_work_tile"
    assert str(l1_plan.logical_index_mapping) == "work_packet_row_major"
    assert str(l1_plan.core_local_address_mapping) == "l1_shard_linear"
    assert str(l1_plan.attached_core_group) == "main_core_group"
    assert int(l1_plan.attached_core_group_index) == 0
    assert str(l1_plan.host_visibility) == "device_local"

    report = tt_program.resource_pressure_reports[0]
    if str(l1_plan.layout) == "circular_buffer":
        assert int(report.per_core_cb_l1_bytes) >= int(l1_plan.page_size_bytes)
        assert int(report.per_core_l1_buffer_bytes) == 0
    else:
        assert int(report.per_core_l1_buffer_bytes) >= int(l1_plan.page_size_bytes)
    assert int(report.max_simultaneous_l1_bytes) == (
        int(report.per_core_cb_l1_aligned_bytes)
        + int(report.per_core_l1_buffer_bytes)
    )

    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]
    executable_distributions = {
        str(plan["buffer"]): plan for plan in executable["buffer_distribution_plans"]
    }
    assert str(executable_distributions["A"]["distribution_kind"]) == "interleaved"
    assert str(executable_distributions["A_shared"]["distribution_kind"]) == "sharded"
    assert tuple(int(dim) for dim in executable_distributions["A_shared"]["shard_grid_shape"]) == (2, 2)
    assert tuple(int(dim) for dim in executable_distributions["A_shared"]["shard_shape"]) == (32, 32)
    assert str(executable_distributions["A_shared"]["sharding_strategy"]) == "block"
    assert str(executable_distributions["A_shared"]["shard_orientation"]) == "row_major"
    assert str(executable_distributions["A_shared"]["source_buffer"]) == "A"
    assert str(executable_distributions["A_shared"]["source_region_kind"]) == "per_work_tile"
    assert str(executable_distributions["A_shared"]["attached_core_group"]) == "main_core_group"


def test_build_tt_program_projects_tensor_memory_config_plans_for_current_placements():
    mod = _prepare_blackhole_phase_b_module(grid_indexed_staged_copy_kernel(3, 3))
    mod = _with_test_hardware_model(
        mod,
        logical_worker_grid_x=2,
        logical_worker_grid_y=2,
        functional_worker_count=4,
        dram_view_count=8,
        worker_l1_size=1572864,
        l1_allocation_alignment_bytes=32,
    )
    mod = tilelang.transform.PlanTTBlocks()(mod)
    mod = tilelang.transform.SelectBlackholeTTMetalBuiltins()(mod)
    mod = tilelang.transform.PlanTTCompute()(mod)
    mod = tilelang.transform.PlanTTTransport()(mod)
    mod = tilelang.transform.PlanTTSync()(mod)
    mod = tilelang.transform.PlanTTABI()(mod)
    mod = tilelang.transform.PlanTTExecution()(mod)
    mod = tilelang.transform.BuildTTProgram()(mod)
    mod = tilelang.transform.ValidateTTProgram()(mod)

    tt_program = mod["main"].attrs["tl.tt_program"]
    distributions = {
        str(plan.buffer): plan for plan in tt_program.buffer_distribution_plans
    }
    memory_configs = {
        str(plan.subject): plan for plan in tt_program.tensor_memory_config_plans
    }

    assert set(distributions).issubset(memory_configs)
    assert str(memory_configs["A"].memory_layout) == "INTERLEAVED"
    assert str(memory_configs["A"].buffer_type) == "DRAM"
    assert str(memory_configs["A"].origin) == "derived_default"
    assert str(memory_configs["B"].memory_layout) == "INTERLEAVED"
    assert str(memory_configs["B"].buffer_type) == "DRAM"

    shared_config = memory_configs["A_shared"]
    shared_distribution = distributions["A_shared"]
    assert str(shared_config.memory_layout) == "BLOCK_SHARDED"
    assert str(shared_config.buffer_type) == "L1"
    assert str(shared_config.origin) == "materialization_requirement"
    assert tuple(int(dim) for dim in shared_config.shard_shape) == tuple(
        int(dim) for dim in shared_distribution.shard_shape
    )
    assert tuple(int(dim) for dim in shared_config.shard_grid_shape) == tuple(
        int(dim) for dim in shared_distribution.shard_grid_shape
    )
    assert str(shared_config.shard_orientation) == str(
        shared_distribution.shard_orientation
    )


def test_validate_tt_program_rejects_invalid_buffer_distribution_placement():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(3, 3))
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    distributions = list(tt_program.buffer_distribution_plans)
    l1_index = next(
        index
        for index, plan in enumerate(distributions)
        if str(plan.memory_space) == "L1"
    )
    distributions[l1_index] = _rebuild_tt_buffer_distribution_plan(
        distributions[l1_index],
        attached_core_group="",
        attached_core_group_index=-1,
    )
    invalid_program = _rebuild_tt_program(
        tt_program,
        buffer_distribution_plans=distributions,
    )
    invalid = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="attached_core_group"):
        tilelang.transform.ValidateTTProgram()(invalid)


def test_validate_tt_program_rejects_incomplete_sharded_address_abi():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(3, 3))
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    distributions = list(tt_program.buffer_distribution_plans)
    l1_index = next(
        index
        for index, plan in enumerate(distributions)
        if str(plan.memory_space) == "L1"
    )

    distributions_without_grid = list(distributions)
    distributions_without_grid[l1_index] = _rebuild_tt_buffer_distribution_plan(
        distributions_without_grid[l1_index],
        shard_grid_shape=[],
    )
    invalid_grid = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(
                    tt_program,
                    buffer_distribution_plans=distributions_without_grid,
                ),
            )
        },
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="shard_grid_shape"):
        tilelang.transform.ValidateTTProgram()(invalid_grid)

    distributions_without_source = list(distributions)
    distributions_without_source[l1_index] = _rebuild_tt_buffer_distribution_plan(
        distributions_without_source[l1_index],
        source_buffer="",
    )
    invalid_source = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(
                    tt_program,
                    buffer_distribution_plans=distributions_without_source,
                ),
            )
        },
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="source_buffer"):
        tilelang.transform.ValidateTTProgram()(invalid_source)

    distributions_with_bad_strategy = list(distributions)
    distributions_with_bad_strategy[l1_index] = _rebuild_tt_buffer_distribution_plan(
        distributions_with_bad_strategy[l1_index],
        sharding_strategy="row_major",
    )
    invalid_strategy = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(
                    tt_program,
                    buffer_distribution_plans=distributions_with_bad_strategy,
                ),
            )
        },
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="sharding_strategy"):
        tilelang.transform.ValidateTTProgram()(invalid_strategy)

    distributions_with_bad_orientation = list(distributions)
    distributions_with_bad_orientation[l1_index] = _rebuild_tt_buffer_distribution_plan(
        distributions_with_bad_orientation[l1_index],
        shard_orientation="block",
    )
    invalid_orientation = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(
                    tt_program,
                    buffer_distribution_plans=distributions_with_bad_orientation,
                ),
            )
        },
        global_infos=mod.global_infos,
    )
    with pytest.raises(Exception, match="shard_orientation"):
        tilelang.transform.ValidateTTProgram()(invalid_orientation)


def test_validate_tt_program_rejects_dram_buffer_distribution_over_hardware_view():
    mod = _prepare_blackhole_tt_program_module(staged_copy_kernel(tile_rows=2, tile_cols=1))
    mod = _with_test_hardware_model(
        mod,
        logical_worker_grid_x=8,
        logical_worker_grid_y=8,
        functional_worker_count=64,
        dram_view_size=1024,
    )
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    distributions = list(tt_program.buffer_distribution_plans)
    dram_index = next(
        index
        for index, plan in enumerate(distributions)
        if str(plan.memory_space) == "DRAM"
    )
    distributions[dram_index] = _rebuild_tt_buffer_distribution_plan(
        distributions[dram_index],
        page_size_bytes=2048,
    )
    invalid_program = _rebuild_tt_program(
        tt_program,
        buffer_distribution_plans=distributions,
    )
    invalid = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="DRAM view"):
        tilelang.transform.ValidateTTProgram()(invalid)


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


def test_build_tt_program_projects_op_sharding_contracts_for_compute_operands():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    tt_program = mod["main"].attrs["tl.tt_program"]

    compute_op = tt_program.compute_op_plans[0]
    memory_configs = {
        str(plan.subject): (index, plan)
        for index, plan in enumerate(tt_program.tensor_memory_config_plans)
    }
    contracts = {
        str(contract.operand_role): contract
        for contract in tt_program.op_sharding_contracts
    }

    assert set(contracts) == {"a", "b", "c"}
    for role, contract in contracts.items():
        binding = next(
            binding
            for binding in compute_op.operand_bindings
            if str(binding.role) == role
        )
        memory_config_index, memory_config = memory_configs[str(binding.buffer)]
        assert str(contract.compute_op_plan) == str(compute_op.name)
        assert int(contract.compute_op_plan_index) == 0
        assert str(contract.operation_name) == "matmul_tiles"
        assert str(contract.op_kind) == "gemm"
        assert str(contract.operand_buffer) == str(binding.buffer)
        assert str(contract.operand_host_buffer) == str(binding.host_buffer)
        assert str(contract.memory_config_plan) == str(memory_config.name)
        assert int(contract.memory_config_plan_index) == memory_config_index
        assert {str(layout) for layout in contract.accepted_memory_layouts} == {
            str(memory_config.memory_layout)
        }
        assert {str(buffer_type) for buffer_type in contract.accepted_buffer_types} == {
            str(memory_config.buffer_type)
        }
        assert {str(strategy) for strategy in contract.accepted_sharding_strategies} == {
            str(memory_config.shard_distribution_strategy)
        }
        assert str(contract.required_shard_orientation) == str(
            memory_config.shard_orientation
        )
        assert bool(contract.may_request_input_conversion) is False
        assert str(contract.reject_reason) == ""

    assert str(contracts["c"].output_policy) == "produces_operand_placement"
    assert bool(contracts["c"].can_produce_output_placement) is True
    assert str(contracts["a"].output_policy) == "not_output"
    assert bool(contracts["a"].can_produce_output_placement) is False


def test_validate_tt_program_rejects_op_sharding_contract_memory_config_mismatch():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    contracts = list(tt_program.op_sharding_contracts)
    contract_index = next(
        index
        for index, contract in enumerate(contracts)
        if str(contract.operand_role) == "a"
    )
    mismatched_index = next(
        index
        for index, plan in enumerate(tt_program.tensor_memory_config_plans)
        if str(plan.subject) != str(contracts[contract_index].operand_buffer)
    )
    mismatched_config = tt_program.tensor_memory_config_plans[mismatched_index]
    contracts[contract_index] = _rebuild_tt_op_sharding_contract(
        contracts[contract_index],
        memory_config_plan=str(mismatched_config.name),
        memory_config_plan_index=mismatched_index,
    )
    broken = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(tt_program, op_sharding_contracts=contracts),
            )
        },
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="memory config subject"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_build_tt_program_projects_placement_resolution_for_op_contracts():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    tt_program = mod["main"].attrs["tl.tt_program"]
    contracts = {str(contract.name): contract for contract in tt_program.op_sharding_contracts}
    resolutions = {
        str(plan.op_sharding_contract): plan
        for plan in tt_program.placement_resolution_plans
    }

    assert set(contracts).issubset(resolutions)
    for contract_name, contract in contracts.items():
        resolution = resolutions[contract_name]
        assert str(resolution.consumer_op_plan) == str(contract.compute_op_plan)
        assert int(resolution.consumer_op_plan_index) == int(
            contract.compute_op_plan_index
        )
        assert str(resolution.consumer_operand_role) == str(contract.operand_role)
        assert str(resolution.selected_memory_config_plan) == str(
            contract.memory_config_plan
        )
        assert int(resolution.selected_memory_config_plan_index) == int(
            contract.memory_config_plan_index
        )
        assert str(resolution.resolution_kind) == "selected_existing"
        assert bool(resolution.conversion_required) is False
        assert str(resolution.conversion_plan) == ""
        assert str(resolution.conflict_reason) == ""


def test_validate_tt_program_rejects_op_contract_placement_conflict():
    mod = _prepare_blackhole_tt_program_module(gemm_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    contracts = list(tt_program.op_sharding_contracts)
    contract_index = next(
        index
        for index, contract in enumerate(contracts)
        if str(contract.operand_role) == "a"
    )
    contracts[contract_index] = _rebuild_tt_op_sharding_contract(
        contracts[contract_index],
        accepted_memory_layouts=["INTERLEAVED"],
    )
    broken = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(tt_program, op_sharding_contracts=contracts),
            )
        },
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="placement conflict.*operand.*a"):
        tilelang.transform.ValidateTTProgram()(broken)


def test_build_tt_program_projects_reshard_plan_for_staged_copy_materialization():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(3, 3))
    tt_program = mod["main"].attrs["tl.tt_program"]
    memory_configs = {
        str(plan.subject): (index, plan)
        for index, plan in enumerate(tt_program.tensor_memory_config_plans)
    }
    reshard = next(
        plan
        for plan in tt_program.reshard_plans
        if str(plan.source_value) == "A" and str(plan.target_value) == "A_shared"
    )

    source_index, source_config = memory_configs["A"]
    target_index, target_config = memory_configs["A_shared"]
    assert str(reshard.conversion_kind) == "interleaved_to_sharded"
    assert str(reshard.source_memory_config_plan) == str(source_config.name)
    assert int(reshard.source_memory_config_plan_index) == source_index
    assert str(reshard.target_memory_config_plan) == str(target_config.name)
    assert int(reshard.target_memory_config_plan_index) == target_index
    assert str(reshard.materialization_protocol) != ""
    assert str(reshard.scheduling_kind) == "runtime"
    assert str(reshard.inserted_by) == "planner"
    assert str(reshard.admission_status) == "admitted"
    assert str(reshard.unsupported_reason) == ""


def test_executable_projection_projects_tensor_memory_config_and_reshard_records():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(3, 3))
    mod = tilelang.transform.MaterializeBlackholeExecutable()(mod)
    executable = mod["main"].attrs["tl.blackhole_executable"]

    assert "tensor_memory_config_plans" in executable
    assert "reshard_plans" in executable
    memory_configs = {
        str(plan["subject"]): plan for plan in executable["tensor_memory_config_plans"]
    }
    reshard_plans = {
        (str(plan["source_value"]), str(plan["target_value"])): plan
        for plan in executable["reshard_plans"]
    }

    assert str(memory_configs["A"]["memory_layout"]) == "INTERLEAVED"
    assert str(memory_configs["A"]["buffer_type"]) == "DRAM"
    assert str(memory_configs["A_shared"]["memory_layout"]) == "BLOCK_SHARDED"
    assert str(memory_configs["A_shared"]["buffer_type"]) == "L1"
    assert str(memory_configs["A_shared"]["source_buffer"]) == "A"

    reshard = reshard_plans[("A", "A_shared")]
    assert str(reshard["conversion_kind"]) == "interleaved_to_sharded"
    assert str(reshard["source_memory_config_plan"]) == str(memory_configs["A"]["name"])
    assert str(reshard["target_memory_config_plan"]) == str(
        memory_configs["A_shared"]["name"]
    )
    assert str(reshard["source_region_kind"]) == "per_work_tile"
    assert tuple(int(dim) for dim in reshard["source_region_shape"]) == (32, 32)
    assert str(reshard["materialization_protocol"]) == "staged_copy"
    assert str(reshard["scheduling_kind"]) == "runtime"
    assert str(reshard["inserted_by"]) == "planner"
    assert str(reshard["admission_status"]) == "admitted"
    assert str(reshard["unsupported_reason"]) == ""


def test_executable_projection_rejects_reshard_without_target_memory_config_index():
    mod = _prepare_blackhole_tt_program_module(grid_indexed_staged_copy_kernel(3, 3))
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    assert tt_program.reshard_plans

    reshard_plans = list(tt_program.reshard_plans)
    reshard_plans[0] = _rebuild_tt_reshard_plan(
        reshard_plans[0],
        target_memory_config_plan_index=-1,
    )
    invalid_program = _rebuild_tt_program(tt_program, reshard_plans=reshard_plans)
    broken = tvm.IRModule(
        {"main": main.with_attr("tl.tt_program", invalid_program)},
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="target_memory_config_plan_index"):
        tilelang.transform.MaterializeBlackholeExecutable()(broken)


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


def test_validate_tt_program_rejects_full_tile_consumer_bound_to_slice_live_form():
    mod = _prepare_blackhole_tt_program_module(fragment_fill_cast_publish_kernel())
    main = mod["main"]
    tt_program = main.attrs["tl.tt_program"]
    live_form_by_name = {str(plan.name): plan for plan in tt_program.live_form_plans}
    binding_index, binding = next(
        (index, plan)
        for index, plan in enumerate(tt_program.consumer_binding_plans)
        if bool(plan.accepts_distributed_slice)
        and not bool(plan.requires_full_logical_tile)
        and str(live_form_by_name[str(plan.source_live_form)].physical_form)
        == "thread_distributed_slice"
    )
    consumer_binding_plans = list(tt_program.consumer_binding_plans)
    consumer_binding_plans[binding_index] = _rebuild_tt_consumer_binding_plan(
        binding,
        accepts_distributed_slice=False,
        requires_full_logical_tile=True,
    )
    broken = tvm.IRModule(
        {
            "main": main.with_attr(
                "tl.tt_program",
                _rebuild_tt_program(
                    tt_program, consumer_binding_plans=consumer_binding_plans
                ),
            )
        },
        global_infos=mod.global_infos,
    )

    with pytest.raises(Exception, match="full logical tile.*distributed slice"):
        tilelang.transform.ValidateTTProgram()(broken)


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
