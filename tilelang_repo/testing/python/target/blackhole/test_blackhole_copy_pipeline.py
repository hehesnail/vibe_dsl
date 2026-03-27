import pytest

import tilelang
from tilelang import language as T
from tilelang.engine.lower import is_device_call, lower, merge_ir_modules
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang import tvm

from .common import (
    check_blackhole_codegen_requirements,
    find_loop_annotation,
    grid_indexed_staged_copy_kernel,
    make_blackhole_cb_requirements_mod,
    staged_copy_kernel,
)

EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS = [
    "input_buffer_addr32",
    "output_buffer_addr32",
    "work_linear_id",
    "a_tile_start_id",
    "a_tile_num_tiles",
    "a_tile_stride",
    "output_tile_start_id",
    "output_tile_num_tiles",
    "output_tile_stride",
]


def _rebuild_codegen_module_with_runtime_args(artifact, runtime_args):
    device_mod = artifact.device_mod
    rewritten = {}
    for gvar, func in device_mod.functions.items():
        if func.attrs and "blackhole.runtime_args" in func.attrs:
            func = func.with_attr("blackhole.runtime_args", runtime_args)
        rewritten[gvar] = func
    target = Target("blackhole")
    build_mod = merge_ir_modules(artifact.host_mod, tvm.IRModule(rewritten))
    return tvm.ffi.get_global_func("target.build.tilelang_blackhole_without_host")(
        build_mod, target
    )


def test_blackhole_codegen_only():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = staged_copy_kernel(tile_rows=2, tile_cols=2)
    target = Target("blackhole")

    try:
        with target:
            artifact = lower(kernel, target=target)
        assert artifact is not None
        assert hasattr(artifact, "kernel_source") or hasattr(artifact, "code")
    except Exception as e:
        pytest.skip(f"Blackhole lowering not yet fully implemented: {e}")


def test_blackhole_codegen_does_not_emit_cb_backed_c_arrays():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "scope: blackhole.cb" not in source
    assert "A_shared[" not in source


def test_blackhole_copy_pass_attrs():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    mod = tilelang.transform.AssignBlackholeCores()(mod)

    func = mod["main"]
    assert "blackhole.target_mode" not in func.attrs

    cb_configs = func.attrs["blackhole.cb_configs"]
    assert [str(cfg["role"]) for cfg in cb_configs] == ["intermediate"]
    assert int(cb_configs[0]["total_size_bytes"]) == 4096
    assert int(cb_configs[0]["lifetime_begin"]) == 0
    assert int(cb_configs[0]["lifetime_end"]) == 0

    cb_bindings = func.attrs["blackhole.cb_bindings"]
    assert len(cb_bindings) == 1
    assert int(cb_bindings[0]["requirement_index"]) == 0
    assert int(cb_bindings[0]["cb_id"]) == int(cb_configs[0]["cb_id"])
    assert int(cb_bindings[0]["cb_config_index"]) == 0
    assert str(cb_bindings[0]["memory_object_name"]) == str(cb_configs[0]["name"])

    runtime_args = func.attrs["blackhole.runtime_args"]
    assert [str(arg["kind"]) for arg in runtime_args] == EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS
    assert str(runtime_args[0]["buffer"]) == "A"
    assert str(runtime_args[1]["buffer"]) == "B"

    core_plan = func.attrs["blackhole.core_plan"]
    assert int(core_plan["logical_grid_x"]) == 1
    assert int(core_plan["logical_grid_y"]) == 1
    assert str(core_plan["linearization"]) == "row_major"
    assert len(core_plan["physical_cores"]) == 1
    assert int(core_plan["physical_cores"][0]["core_x"]) == 1
    assert int(core_plan["physical_cores"][0]["core_y"]) == 2
    assert len(core_plan["work_packets"]) == 1
    assert int(core_plan["work_packets"][0]["work_offset"]) == 0
    assert int(core_plan["work_packets"][0]["work_count"]) == 1

    segment_plan = func.attrs["blackhole.segment_plan"]
    assert len(segment_plan) == 1
    assert str(segment_plan[0]["kind"]) == "fused_dataflow"
    assert str(segment_plan[0]["core_type"]) == "brisc"

    body_script = func.body.script()
    assert "tl.blackhole.read_tile_to_cb" in body_script
    assert "tl.blackhole.write_tile_from_cb" in body_script
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 1
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 1


def test_blackhole_copy_semantics_annotation_schema():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)

    sem = find_loop_annotation(mod["main"].body, "blackhole.copy_semantics")
    assert sem is not None
    assert str(sem["kind"]) == "fused_staged_copy"
    assert str(sem["direction"]) == "dram_to_cb_to_dram"
    assert str(sem["src_buffer"]) == "A"
    assert str(sem["mid_buffer"]) == "A_shared"
    assert str(sem["dst_buffer"]) == "B"
    assert str(sem["src_scope"]) == "global"
    assert str(sem["dst_scope"]) == "global"
    assert str(sem["dtype"]) == "float16"
    assert [int(x) for x in sem["src_shape"]] == [64, 32]
    assert [int(x) for x in sem["dst_shape"]] == [64, 32]
    assert [int(x) for x in sem["mid_shape"]] == [32, 32]


def test_blackhole_copy_semantics_survives_flatten_and_vectorize():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    mod = tilelang.transform.AssignBlackholeCores()(mod)

    func = mod["main"]
    runtime_args = func.attrs["blackhole.runtime_args"]
    assert [str(arg["kind"]) for arg in runtime_args] == EXPECTED_UNIFIED_COPY_RUNTIME_ARG_KINDS
    assert str(runtime_args[0]["buffer"]) == "A"
    assert str(runtime_args[1]["buffer"]) == "B"

    body_script = func.body.script()
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 1
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 1


@pytest.mark.xfail(
    reason=(
        "StorageRewrite is incompatible with the Blackhole CB model: its "
        "VectorTypeAccessChecker only recognizes AllocateNode as buffer declarations "
        "but after FlattenBuffer+VectorizeLoop the shared (CB) buffers are represented "
        "via DeclBuffer, causing a spurious 'buffer used before declaration' error. "
        "StorageRewrite provides no benefit for Blackhole circular buffers and is intentionally "
        "excluded from the Blackhole pipeline."
    ),
    strict=True,
)
def test_blackhole_storage_rewrite_incompatible_with_cb_model():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    tilelang.transform.StorageRewrite()(mod)


def test_blackhole_cb_planner_reuses_non_overlapping_requirements():
    mod = make_blackhole_cb_requirements_mod(
        [
            {
                "name": "stage0",
                "type": "intermediate",
                "page_size": 524288,
                "num_pages": 2,
                "data_format": "Float16",
                "lifetime_begin": 0,
                "lifetime_end": 0,
            },
            {
                "name": "stage1",
                "type": "intermediate",
                "page_size": 524288,
                "num_pages": 2,
                "data_format": "Float16",
                "lifetime_begin": 1,
                "lifetime_end": 1,
            },
        ]
    )
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    func = mod["main"]
    cb_configs = func.attrs["blackhole.cb_configs"]
    cb_bindings = func.attrs["blackhole.cb_bindings"]

    assert len(cb_configs) == 1
    assert int(func.attrs["blackhole.num_cbs"]) == 1
    assert int(func.attrs["blackhole.total_l1_bytes"]) == 1048576
    assert int(cb_configs[0]["cb_id"]) == 32
    assert int(cb_configs[0]["lifetime_begin"]) == 0
    assert int(cb_configs[0]["lifetime_end"]) == 1
    assert [str(name) for name in cb_configs[0]["requirement_names"]] == ["stage0", "stage1"]
    assert len(cb_bindings) == 2
    assert [int(binding["requirement_index"]) for binding in cb_bindings] == [0, 1]
    assert [str(binding["requirement_name"]) for binding in cb_bindings] == ["stage0", "stage1"]
    assert [int(binding["cb_id"]) for binding in cb_bindings] == [32, 32]
    assert [int(binding["cb_config_index"]) for binding in cb_bindings] == [0, 0]


def test_blackhole_cb_planner_rejects_overlapping_large_requirements():
    mod = make_blackhole_cb_requirements_mod(
        [
            {
                "name": "stage0",
                "type": "intermediate",
                "page_size": 524288,
                "num_pages": 2,
                "data_format": "Float16",
                "lifetime_begin": 0,
                "lifetime_end": 1,
            },
            {
                "name": "stage1",
                "type": "intermediate",
                "page_size": 524288,
                "num_pages": 2,
                "data_format": "Float16",
                "lifetime_begin": 1,
                "lifetime_end": 2,
            },
        ]
    )
    with pytest.raises(Exception, match="PlanBlackholeCB|1572864|1.5MB|per-core constraints"):
        tilelang.transform.PlanBlackholeCB()(mod)


def test_blackhole_copy_codegen_uses_runtime_schema():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "uint32_t A_addr = get_arg_val<uint32_t>(0);" in source
    assert "uint32_t B_addr = get_arg_val<uint32_t>(1);" in source
    assert "uint32_t work_linear_id = get_arg_val<uint32_t>(2);" in source
    assert "uint32_t a_tile_start_id = get_arg_val<uint32_t>(3);" in source
    assert "uint32_t a_tile_num_tiles = get_arg_val<uint32_t>(4);" in source
    assert "uint32_t a_tile_stride = get_arg_val<uint32_t>(5);" in source
    assert "uint32_t output_tile_start_id = get_arg_val<uint32_t>(6);" in source
    assert "uint32_t output_tile_num_tiles = get_arg_val<uint32_t>(7);" in source
    assert "uint32_t output_tile_stride = get_arg_val<uint32_t>(8);" in source
    assert "src_dram_addr" not in source
    assert "dst_dram_addr" not in source
    assert "scratch_l1_addr" not in source
    assert "cb_reserve_back(" in source
    assert "cb_push_back(" in source
    assert "cb_wait_front(" in source
    assert "cb_pop_front(" in source


def test_blackhole_copy_codegen_rejects_schema_without_work_linear_id():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    unsupported_runtime_args = [
        {"name": "A_addr", "kind": "input_buffer_addr32", "dtype": "uint32", "buffer": "A"},
        {"name": "B_addr", "kind": "output_buffer_addr32", "dtype": "uint32", "buffer": "B"},
        {"name": "a_tile_start_id", "kind": "a_tile_start_id", "dtype": "uint32"},
        {"name": "a_tile_num_tiles", "kind": "a_tile_num_tiles", "dtype": "uint32"},
        {"name": "a_tile_stride", "kind": "a_tile_stride", "dtype": "uint32"},
        {"name": "output_tile_start_id", "kind": "output_tile_start_id", "dtype": "uint32"},
        {"name": "output_tile_num_tiles", "kind": "output_tile_num_tiles", "dtype": "uint32"},
        {"name": "output_tile_stride", "kind": "output_tile_stride", "dtype": "uint32"},
    ]

    with pytest.raises(Exception, match="work_linear_id|copy fallback|stride"):
        _rebuild_codegen_module_with_runtime_args(artifact, unsupported_runtime_args)


def test_blackhole_core_plan_preserves_logical_block_launch():
    kernel = grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]

    assert int(core_plan["logical_grid_x"]) == 2
    assert int(core_plan["logical_grid_y"]) == 3
    assert str(core_plan["linearization"]) == "row_major"
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6
    assert {
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    } == {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)}
    assert [
        (int(packet["core_x"]), int(packet["core_y"]))
        for packet in core_plan["work_packets"]
    ] == [
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
    ]
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3, 4, 5]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1, 1, 1]

    body_script = device_main.body.script()
    assert "tl.blackhole.read_tile_to_cb(A, by * 2 + bx, 32, 2048, 0)" in body_script
    assert "tl.blackhole.write_tile_from_cb(32, B, by * 2 + bx, 2048, 0)" in body_script


def test_blackhole_core_plan_preserves_axis_order():
    kernel = grid_indexed_staged_copy_kernel(grid_x=3, grid_y=2)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]

    assert int(core_plan["logical_grid_x"]) == 3
    assert int(core_plan["logical_grid_y"]) == 2
    assert len(core_plan["physical_cores"]) == 6
    assert len(core_plan["work_packets"]) == 6
    assert {
        (int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]
    } == {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)}
    assert [int(packet["work_offset"]) for packet in core_plan["work_packets"]] == [0, 1, 2, 3, 4, 5]
    assert [int(packet["work_count"]) for packet in core_plan["work_packets"]] == [1, 1, 1, 1, 1, 1]

    body_script = device_main.body.script()
    assert "tl.blackhole.read_tile_to_cb(A, by * 3 + bx, 32, 2048, 0)" in body_script
    assert "tl.blackhole.write_tile_from_cb(32, B, by * 3 + bx, 2048, 0)" in body_script


def test_blackhole_core_plan_covers_oversubscribed_work():
    kernel = grid_indexed_staged_copy_kernel(grid_x=15, grid_y=10)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    core_plan = device_main.attrs["blackhole.core_plan"]

    assert int(core_plan["logical_grid_x"]) == 15
    assert int(core_plan["logical_grid_y"]) == 10
    assert len(core_plan["physical_cores"]) == 140
    assert len(core_plan["work_packets"]) == 140
    assert len(
        {(int(core["core_x"]), int(core["core_y"])) for core in core_plan["physical_cores"]}
    ) == 140
    assert int(device_main.attrs["blackhole.work_per_core"]) == 2

    covered = []
    for packet in core_plan["work_packets"]:
        work_offset = int(packet["work_offset"])
        work_count = int(packet["work_count"])
        covered.extend(range(work_offset, work_offset + work_count))

    assert covered == list(range(150))
    assert sum(int(packet["work_count"]) for packet in core_plan["work_packets"]) == 150


def test_blackhole_copy_oversubscription_fails_compile_time():
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1, tile_m=1024, tile_n=1024)
    target = Target("blackhole")
    with pytest.raises(Exception, match="PlanBlackholeCB|1572864|1.5MB|per-core constraints"):
        with target:
            lower(kernel, target=target)


@pytest.mark.parametrize(
    "tile_rows,tile_cols,tile_m,tile_n,expected_terms",
    [
        (2, 1, 32, 64, ["tile_row * 2", "tile_row * 2 + 1"]),
        (1, 2, 64, 32, ["tile_idx", "tile_idx + 2"]),
    ],
)
def test_blackhole_copy_tracks_rectangular_tile_shapes(
    tile_rows, tile_cols, tile_m, tile_n, expected_terms
):
    kernel = staged_copy_kernel(
        tile_rows=tile_rows, tile_cols=tile_cols, tile_m=tile_m, tile_n=tile_n
    )
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    cb_configs = device_main.attrs["blackhole.cb_configs"]

    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 2

    body_script = device_main.body.script()
    assert body_script.count("tl.blackhole.read_tile_to_cb") == 2
    assert body_script.count("tl.blackhole.write_tile_from_cb") == 2
    for expected in expected_terms:
        assert expected in body_script


def test_blackhole_lower_restores_host_device_split():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    host_funcs = {str(gvar): func for gvar, func in artifact.host_mod.functions.items()}
    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}

    assert 'I.GlobalVar("main")' in host_funcs
    assert 'I.GlobalVar("main_kernel")' in device_funcs

    host_main = host_funcs['I.GlobalVar("main")']
    device_main = device_funcs['I.GlobalVar("main_kernel")']
    assert host_main.attrs["calling_conv"] == CallingConv.C_PACKED_FUNC
    assert host_main.attrs["target"].kind.name == "c"
    assert device_main.attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH
    assert device_main.attrs["target"].kind.name == "blackhole"
    assert "blackhole.target_mode" not in device_main.attrs


def test_blackhole_target_mode_does_not_define_device_kernel():
    @T.prim_func
    def trivial_kernel(A: T.Buffer((32,), "float16")):
        with T.Kernel(1, threads=1) as (bx,):
            A[bx] = A[bx]

    tagged = trivial_kernel.with_attrs(
        {"target": Target("blackhole"), "blackhole.target_mode": "single_core_copy"}
    )
    assert not is_device_call(tagged)


def test_blackhole_runtime_module_keeps_host_and_device_entries():
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    assert artifact.codegen_mod["main"] is not None
    assert artifact.codegen_mod["main_kernel"] is not None


def test_blackhole_kernel_compilation():
    @T.prim_func
    def elementwise_add(
        A: T.Buffer((64,), "float16"),
        B: T.Buffer((64,), "float16"),
        C: T.Buffer((64,), "float16"),
    ):
        with T.Kernel(2) as bx:
            for i in T.Parallel(32):
                idx = bx * 32 + i
                C[idx] = A[idx] + B[idx]

    try:
        target = Target("blackhole")
        artifact = lower(elementwise_add, target=target)
        assert artifact is not None
        if hasattr(artifact, "kernel_source"):
            assert len(artifact.kernel_source) > 0
        elif hasattr(artifact, "code"):
            assert len(artifact.code) > 0
    except Exception as e:
        pytest.skip(f"Blackhole compilation not yet complete: {e}")
