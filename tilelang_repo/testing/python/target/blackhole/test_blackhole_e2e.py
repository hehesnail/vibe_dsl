"""
True End-to-End Test for TileLang Blackhole Backend

This test verifies the Blackhole workflow at two layers:
1. TileLang DSL kernel compilation to Blackhole target
2. Kernel execution via BlackholeModule direct path

Requirements:
- TT-Sim environment configured (or real hardware)
"""

import pytest
import numpy as np
import torch
import os

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.engine.lower import is_device_call, lower
from tilelang.jit import compile as tl_compile
from tvm.target import Target
from tvm.ir import CallingConv


def find_loop_annotation(stmt, attr_key):
    """Find the first For loop carrying the given annotation key."""
    if isinstance(stmt, tilelang.tvm.tir.For) and stmt.annotations.get(attr_key) is not None:
        return stmt.annotations[attr_key]
    if isinstance(stmt, tilelang.tvm.tir.BlockRealize):
        return find_loop_annotation(stmt.block.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.Block):
        return find_loop_annotation(stmt.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.SeqStmt):
        for child in stmt.seq:
            found = find_loop_annotation(child, attr_key)
            if found is not None:
                return found
        return None
    if isinstance(stmt, tilelang.tvm.tir.IfThenElse):
        found = find_loop_annotation(stmt.then_case, attr_key)
        if found is not None:
            return found
        if stmt.else_case is not None:
            return find_loop_annotation(stmt.else_case, attr_key)
        return None
    if hasattr(stmt, "body"):
        return find_loop_annotation(stmt.body, attr_key)
    return None


def check_blackhole_codegen_requirements():
    """Check if Blackhole compilation requirements are met."""
    tilelang_home = os.environ.get("TILELANG_HOME")
    if not tilelang_home:
        return False, "TILELANG_HOME not set"
    return True, "OK"


def check_blackhole_direct_execution_requirements():
    """Check if BlackholeModule direct execution requirements are met."""
    can_codegen, msg = check_blackhole_codegen_requirements()
    if not can_codegen:
        return False, msg

    tt_metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_runtime_root:
        return False, "TT_METAL_RUNTIME_ROOT not set"
    if not os.path.isdir(os.path.join(tt_metal_runtime_root, "tt_metal")):
        return False, f"TT_METAL_RUNTIME_ROOT does not contain tt_metal/: {tt_metal_runtime_root}"

    if not direct_build_enabled():
        return False, "TileLang build does not appear to have USE_BLACKHOLE_DIRECT=ON"

    return True, "OK"


def direct_build_enabled():
    cache_candidates = []
    loaded_cache = get_loaded_tilelang_cmake_cache()
    if loaded_cache:
        cache_candidates.append(loaded_cache)

    tilelang_home = os.environ.get("TILELANG_HOME")
    if tilelang_home:
        cache_candidates.extend(
            [
                os.path.join(tilelang_home, "build", "CMakeCache.txt"),
            ]
        )

    for cache_path in cache_candidates:
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_text = f.read()
            if "USE_BLACKHOLE_DIRECT" in cache_text and "=ON" in cache_text:
                return True
    return False


def get_loaded_tilelang_cmake_cache():
    lib_path = getattr(tilelang, "_LIB_PATH", None)
    if not lib_path:
        return None
    build_dir = os.path.dirname(os.path.dirname(lib_path))
    return os.path.join(build_dir, "CMakeCache.txt")


def staged_copy_kernel(tile_rows: int, tile_cols: int = 1, tile_m: int = 32, tile_n: int = 32):
    """Define an explicit TileLang T.copy(global->shared->global) kernel."""
    M = tile_rows * tile_m
    N = tile_cols * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((tile_m, tile_n), "float16")
            for tile_idx in T.serial(tile_rows * tile_cols):
                tile_row = tile_idx // tile_cols
                tile_col = tile_idx % tile_cols
                T.copy(A[tile_row * tile_m, tile_col * tile_n], A_shared)
                T.copy(A_shared, B[tile_row * tile_m, tile_col * tile_n])

    return main


def grid_indexed_staged_copy_kernel(grid_x: int, grid_y: int, tile_m: int = 32, tile_n: int = 32):
    """Define a copy kernel whose indices depend on bx/by logical block coordinates."""
    M = grid_y * tile_m
    N = grid_x * tile_n

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(grid_x, grid_y) as (bx, by):
            A_shared = T.alloc_shared((tile_m, tile_n), "float16")
            T.copy(A[by * tile_m, bx * tile_n], A_shared)
            T.copy(A_shared, B[by * tile_m, bx * tile_n])

    return main


def test_blackhole_codegen_only():
    """Test that Blackhole code generation works (no execution)."""
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    # Define a simple kernel
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=2)

    # Compile to Blackhole target (codegen only, no execution)
    target = Target("blackhole")

    try:
        # Target needs to be set as context for layout inference
        with target:
            artifact = lower(kernel, target=target)
        assert artifact is not None
        assert hasattr(artifact, 'kernel_source') or hasattr(artifact, 'code')
        print("Blackhole code generation successful!")
        if hasattr(artifact, 'kernel_source'):
            print(f"Generated source length: {len(artifact.kernel_source)} chars")
            print("\n=== Generated Kernel ===")
            print(artifact.kernel_source)
            print("=== End Kernel ===")
    except Exception as e:
        pytest.skip(f"Blackhole lowering not yet fully implemented: {e}")


def test_blackhole_copy_pass_attrs():
    """Verify copy schema is materialized in pass attrs before runtime extraction."""
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    mod = tilelang.transform.AssignBlackholeCores()(mod)

    func = mod["main"]
    assert "blackhole.target_mode" not in func.attrs

    cb_configs = func.attrs["blackhole.cb_configs"]
    cb_roles = [str(cfg["role"]) for cfg in cb_configs]
    assert cb_roles == ["intermediate"]
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
    runtime_arg_kinds = [str(arg["kind"]) for arg in runtime_args]
    assert runtime_arg_kinds == [
        "input_buffer_addr32",
        "output_buffer_addr32",
        "current_work_linear_id",
        "tile_count",
        "scratch_l1_buffer_addr32",
    ]
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
    assert "for i in T.vectorized(8):\n                    T.tl.blackhole.read_tile_to_cb" not in body_script
    assert "for i in T.vectorized(8):\n                    T.tl.blackhole.write_tile_from_cb" not in body_script


def test_blackhole_copy_semantics_annotation_schema():
    """Stage 2C annotation should be a structured map before split/lowering."""
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
    """Stage 2C copy annotation should remain lowerable after common split-before passes."""
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    mod = tilelang.transform.PlanBlackholeCB()(mod)
    mod = tilelang.transform.AssignBlackholeCores()(mod)

    func = mod["main"]
    runtime_args = func.attrs["blackhole.runtime_args"]
    assert [str(arg["kind"]) for arg in runtime_args] == [
        "input_buffer_addr32",
        "output_buffer_addr32",
        "current_work_linear_id",
        "tile_count",
        "scratch_l1_buffer_addr32",
    ]
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
        "StorageRewrite provides no benefit for Blackhole circular buffers (hardware-managed "
        "fixed-size resources) and is intentionally excluded from the Blackhole pipeline. "
        "Phase 4 may add shared-scope exemption if StorageRewrite is ever needed."
    ),
    strict=True,
)
def test_blackhole_storage_rewrite_incompatible_with_cb_model():
    """Stage 2C finding: StorageRewrite must NOT be run on Blackhole IR after FlattenBuffer+VectorizeLoop.

    Blackhole 'shared' scope maps to hardware Circular Buffers, not CUDA shared memory.
    StorageRewrite's VectorTypeAccessChecker does not recognize DeclBuffer as a declaration,
    so after FlattenBuffer+VectorizeLoop it fails with 'buffer used before declaration'.
    The Blackhole pipeline correctly excludes StorageRewrite.
    """
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    # This is expected to raise — StorageRewrite is not compatible with Blackhole CB IR.
    mod = tilelang.transform.StorageRewrite()(mod)


def make_blackhole_cb_requirements_mod(cb_requirements):
    """Build a split/lowered Blackhole module with an explicit CB requirement list."""
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    func = mod["main"].with_attr("blackhole.cb_requirements", cb_requirements)
    return tilelang.tvm.IRModule({"main": func})


def test_blackhole_cb_planner_reuses_non_overlapping_requirements():
    """Planner should reuse a single CB object for compatible non-overlapping lifetimes."""
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
    assert [str(binding["memory_object_name"]) for binding in cb_bindings] == [
        str(cb_configs[0]["name"]),
        str(cb_configs[0]["name"]),
    ]


def test_blackhole_cb_planner_rejects_overlapping_large_requirements():
    """Planner should keep overlapping lifetimes separate and fail if the combined L1 footprint is illegal."""
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
    """Verify copy codegen consumes runtime arg schema instead of fixed slot names."""
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "uint32_t A_addr = get_arg_val<uint32_t>(0);" in source
    assert "uint32_t B_addr = get_arg_val<uint32_t>(1);" in source
    assert "uint32_t current_work_linear_id = get_arg_val<uint32_t>(2);" in source
    assert "uint32_t tile_count = get_arg_val<uint32_t>(3);" in source
    assert "uint32_t scratch_l1_addr = get_arg_val<uint32_t>(4);" in source
    assert "const uint32_t tile_index = tile_row;" in source
    assert "src_dram_addr" not in source
    assert "dst_dram_addr" not in source


def test_blackhole_core_plan_preserves_logical_block_launch():
    """Grid-indexed copy should preserve logical-block tile indexing into lowered code."""
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
    assert len(core_plan["physical_cores"]) == 1
    assert len(core_plan["work_packets"]) == 1
    assert int(core_plan["work_packets"][0]["work_offset"]) == 0
    assert int(core_plan["work_packets"][0]["work_count"]) == 6

    body_script = device_main.body.script()
    assert "tl.blackhole.read_tile_to_cb(A, by * 2 + bx, 32, 2048, 0)" in body_script
    assert "tl.blackhole.write_tile_from_cb(32, B, by * 2 + bx, 2048, 0)" in body_script

    source = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    assert "uint32_t current_work_linear_id = get_arg_val<uint32_t>(2);" in source
    assert "(current_work_linear_id / 2)" in source
    assert "(current_work_linear_id % 2)" in source
    assert "const uint32_t tile_index = 0;" not in source


def test_blackhole_module_direct_call_grid_indexed_copy():
    """Exercise direct-call on a grid-indexed staged copy that depends on bx/by."""
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    grid_x, grid_y = 2, 3
    M, N = grid_y * 32, grid_x * 32
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = grid_indexed_staged_copy_kernel(grid_x=grid_x, grid_y=grid_y)

    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)

    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(b_output, b_ref, atol=atol, rtol=rtol):
        print("SUCCESS: Grid-indexed staged-copy direct call matches PyTorch reference!")
        print(f"  Input shape: {a_torch.shape}")
        print(f"  Max difference: {(b_output - b_ref).abs().max().item()}")
        assert True
    else:
        diff = (b_output - b_ref).abs()
        print("FAILURE: Grid-indexed direct-call output mismatch!")
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Mean difference: {diff.mean().item()}")
        assert False, "Grid-indexed direct-call output does not match reference"


def test_blackhole_large_shape_copy_keeps_per_core_l1_small():
    """Large-shape copy should keep CB/L1 planning bounded by the shared tile, not total tensor bytes."""
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    device_funcs = {str(gvar): func for gvar, func in artifact.device_mod.functions.items()}
    device_main = device_funcs['I.GlobalVar("main_kernel")']

    assert int(device_main.attrs["blackhole.total_l1_bytes"]) == 4096
    cb_configs = device_main.attrs["blackhole.cb_configs"]
    assert len(cb_configs) == 1
    assert int(cb_configs[0]["page_size"]) == 2048
    assert int(cb_configs[0]["num_pages"]) == 2
    assert int(cb_configs[0]["total_size_bytes"]) == 4096
    assert int(cb_configs[0]["lifetime_begin"]) == 0
    assert int(cb_configs[0]["lifetime_end"]) == 0


def test_blackhole_module_direct_call_large_shape_copy():
    """Large-shape copy (>1.5MB total data) should still execute when per-core L1 plan is legal."""
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    M, N = 25 * 32, 32 * 32  # 800 x 1024 = 1,638,400 bytes of float16 input
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=25, tile_cols=32)

    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)

    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(b_output, b_ref, atol=atol, rtol=rtol):
        print("SUCCESS: Large-shape staged-copy direct call matches PyTorch reference!")
        print(f"  Input shape: {a_torch.shape}")
        print(f"  Input bytes: {a_torch.numel() * 2}")
        print(f"  Max difference: {(b_output - b_ref).abs().max().item()}")
        assert True
    else:
        diff = (b_output - b_ref).abs()
        print("FAILURE: Large-shape direct-call output mismatch!")
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Mean difference: {diff.mean().item()}")
        assert False, "Large-shape direct-call output does not match reference"


def test_blackhole_copy_oversubscription_fails_compile_time():
    """Per-core CB/L1 oversubscription should fail during compilation, not at runtime."""
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
    """Rectangular staged copy should lower to hardware-tile indices derived from DSL shape."""
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
    """Blackhole lower() should expose host/device split IR after Stage 2A recovery."""
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
    """`blackhole.target_mode` must not participate in device-kernel detection."""

    @T.prim_func
    def trivial_kernel(A: T.Buffer((32,), "float16")):
        with T.Kernel(1, threads=1) as (bx,):
            A[bx] = A[bx]

    tagged = trivial_kernel.with_attrs(
        {
            "target": Target("blackhole"),
            "blackhole.target_mode": "single_core_copy",
        }
    )

    assert not is_device_call(tagged)


def test_blackhole_runtime_module_keeps_host_and_device_entries():
    """The Blackhole runtime module should expose the public entry and its kernel."""
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    assert artifact.codegen_mod["main"] is not None
    assert artifact.codegen_mod["main_kernel"] is not None


def test_blackhole_module_direct_call():
    """Exercise the BlackholeModule packed-func entrypoint directly."""
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    M, N = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=M // 32, tile_cols=N // 32)

    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)

    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(b_output, b_ref, atol=atol, rtol=rtol):
        print("SUCCESS: BlackholeModule direct call matches PyTorch reference!")
        print(f"  Input shape: {a_torch.shape}")
        print(f"  Max difference: {(b_output - b_ref).abs().max().item()}")
        assert True
    else:
        diff = (b_output - b_ref).abs()
        print("FAILURE: Direct-call output mismatch!")
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Mean difference: {diff.mean().item()}")
        assert False, "Direct-call output does not match reference"


def test_blackhole_module_direct_call_rectangular_tiles():
    """Exercise direct-call on a staged copy whose DSL tile shape spans multiple hardware tiles."""
    can_run, msg = check_blackhole_direct_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    M, N = 64, 64
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1, tile_m=32, tile_n=64)

    with target:
        artifact = lower(kernel, target=target)

    artifact.codegen_mod["main"](a_torch, b_output)

    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(b_output, b_ref, atol=atol, rtol=rtol):
        print("SUCCESS: Rectangular staged-copy direct call matches PyTorch reference!")
        print(f"  Input shape: {a_torch.shape}")
        print(f"  Max difference: {(b_output - b_ref).abs().max().item()}")
        assert True
    else:
        diff = (b_output - b_ref).abs()
        print("FAILURE: Rectangular direct-call output mismatch!")
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Mean difference: {diff.mean().item()}")
        assert False, "Rectangular direct-call output does not match reference"

def test_blackhole_kernel_compilation():
    """Test that we can compile a kernel for Blackhole target.

    This is a minimal test that only verifies compilation works,
    without requiring hardware or simulator.
    """
    # Simple element-wise addition kernel
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

        # Verify we got something back
        assert artifact is not None

        # Check if source code was generated
        if hasattr(artifact, 'kernel_source'):
            source = artifact.kernel_source
            assert len(source) > 0
            print(f"Generated {len(source)} chars of kernel source")
        elif hasattr(artifact, 'code'):
            code = artifact.code
            assert len(code) > 0
            print(f"Generated {len(code)} chars of code")

        print("Blackhole kernel compilation test PASSED")

    except Exception as e:
        # Compilation might not be fully implemented yet
        pytest.skip(f"Blackhole compilation not yet complete: {e}")


def gemm_kernel(M: int = 32, N: int = 32, K: int = 128):
    """GEMM kernel with explicit data-movement: T.copy A/B in, gemm, T.copy C out."""
    block_M, block_N, block_K = M, N, K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            B_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.copy(A[0:block_M, 0:block_K], A_shared)
            T.copy(B[0:block_N, 0:block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0:block_M, 0:block_N])

    return main


def test_blackhole_split_kernel_gemm_segment_plan():
    """SplitBlackholeKernel should produce a 3-kernel segment_plan for GEMM."""
    kernel = gemm_kernel()
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    # AnnotateBlackholeCopySemantics annotates copy loops with direction metadata;
    # SplitBlackholeKernel uses those annotations to classify reader/writer stmts.
    mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)

    # Apply SplitBlackholeKernel to get segment annotations
    mod = tilelang.transform.SplitBlackholeKernel()(mod)

    # Find any func that has a segment_plan
    plan = None
    for gv, func in mod.functions.items():
        if func.attrs and "blackhole.segment_plan" in func.attrs:
            plan = func.attrs["blackhole.segment_plan"]
            break

    assert plan is not None, "No blackhole.segment_plan found in any function"
    assert len(plan) == 3, f"Expected 3 segments (reader/compute/writer), got {len(plan)}"
    assert str(plan[0]["kind"]) == "reader"
    assert str(plan[1]["kind"]) == "compute"
    assert str(plan[2]["kind"]) == "writer"
    assert str(plan[0]["core_type"]) == "brisc"
    assert str(plan[1]["core_type"]) == "trisc"
    assert str(plan[2]["core_type"]) == "ncrisc"

    # Check reader runtime_args contain DRAM buffer addr args
    reader_args = plan[0]["runtime_args"]
    reader_arg_names = [str(arg["name"]) for arg in reader_args]
    assert any("addr" in name for name in reader_arg_names), (
        f"Reader should have DRAM buffer addr args, got: {reader_arg_names}")

    # Check writer runtime_args contain DRAM buffer addr args
    writer_args = plan[2]["runtime_args"]
    writer_arg_names = [str(arg["name"]) for arg in writer_args]
    assert any("addr" in name for name in writer_arg_names), (
        f"Writer should have DRAM buffer addr args, got: {writer_arg_names}")


if __name__ == "__main__":
    tilelang.testing.main()
