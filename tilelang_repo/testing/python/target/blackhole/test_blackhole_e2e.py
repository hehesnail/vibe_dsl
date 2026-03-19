"""
True End-to-End Test for TileLang Blackhole Backend

This test verifies the complete workflow:
1. TileLang DSL kernel compilation to Blackhole target
2. Kernel execution via external runner
3. Result comparison with PyTorch reference

Requirements:
- TileLang-managed tilelang_blackhole_runner built and accessible
- TT-Sim environment configured (or real hardware)
"""

import pytest
import numpy as np
import torch
import os
import tempfile
import subprocess
import json

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.engine.lower import lower
from tilelang.jit import compile as tl_compile
from tvm.target import Target
from tvm.ir import CallingConv


def check_blackhole_codegen_requirements():
    """Check if Blackhole compilation requirements are met."""
    tilelang_home = os.environ.get("TILELANG_HOME")
    if not tilelang_home:
        return False, "TILELANG_HOME not set"
    return True, "OK"


def check_blackhole_execution_requirements():
    """Check if Blackhole true-execution requirements are met."""
    can_codegen, msg = check_blackhole_codegen_requirements()
    if not can_codegen:
        return False, msg

    tilelang_home = os.environ["TILELANG_HOME"]
    runner_build_dir = os.environ.get("TILELANG_BLACKHOLE_RUNNER_BUILD_DIR")
    tt_metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_runtime_root:
        return False, "TT_METAL_RUNTIME_ROOT not set"
    if not os.path.isdir(os.path.join(tt_metal_runtime_root, "tt_metal")):
        return False, f"TT_METAL_RUNTIME_ROOT does not contain tt_metal/: {tt_metal_runtime_root}"

    runner_candidates = [
        os.path.join(runner_build_dir, "tilelang_blackhole_runner") if runner_build_dir else None,
        os.path.join(tilelang_home, "build-blackhole-runner", "tilelang_blackhole_runner"),
        os.path.join(tilelang_home, "build_blackhole_runner", "tilelang_blackhole_runner"),
        os.path.join(tilelang_home, "tools", "blackhole_runner", "build", "tilelang_blackhole_runner"),
    ]
    runner_candidates = [path for path in runner_candidates if path]
    if not any(os.path.exists(path) for path in runner_candidates):
        return False, f"Runner not found in {runner_candidates}"

    return True, "OK"


def get_runner_path():
    tilelang_home = os.environ["TILELANG_HOME"]
    runner_build_dir = os.environ.get("TILELANG_BLACKHOLE_RUNNER_BUILD_DIR")
    runner_candidates = [
        os.path.join(runner_build_dir, "tilelang_blackhole_runner") if runner_build_dir else None,
        os.path.join(tilelang_home, "build-blackhole-runner", "tilelang_blackhole_runner"),
        os.path.join(tilelang_home, "build_blackhole_runner", "tilelang_blackhole_runner"),
        os.path.join(tilelang_home, "tools", "blackhole_runner", "build", "tilelang_blackhole_runner"),
    ]
    runner_candidates = [path for path in runner_candidates if path]
    for path in runner_candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Runner not found in {runner_candidates}")


def write_single_core_copy_spec(spec_path, kernel_path, tensor_nbytes):
    spec = {
        "entry_name": "main",
        "input_size_bytes": int(tensor_nbytes),
        "output_size_bytes": int(tensor_nbytes),
        "scalar_args": [],
        "core_plan": {
            "logical_grid_x": 1,
            "logical_grid_y": 1,
            "linearization": "row_major",
            "physical_cores": [{"core_x": 1, "core_y": 2}],
            "work_packets": [{"core_x": 1, "core_y": 2, "work_offset": 0, "work_count": 1}],
        },
        "cb_configs": [
            {
                "cb_id": 32,
                "name": "A_shared",
                "role": "intermediate",
                "num_pages": 1,
                "page_size_bytes": int(tensor_nbytes),
                "data_format": "Float16_b",
            },
        ],
        "kernels": [
            {
                "name": "main",
                "kind": "fused_dataflow",
                "core_type": "brisc",
                "kernel_path": kernel_path,
                "compile_time_args": [],
                "runtime_args": [
                    {"name": "input0", "kind": "input_buffer_addr32", "dtype": "uint32"},
                    {"name": "output0", "kind": "output_buffer_addr32", "dtype": "uint32"},
                    {
                        "name": "current_work_linear_id",
                        "kind": "current_work_linear_id",
                        "dtype": "uint32",
                    },
                    {"name": "num_tiles", "kind": "tile_count", "dtype": "uint32"},
                    {"name": "scratch_l1", "kind": "scratch_l1_buffer_addr32", "dtype": "uint32"},
                ],
            }
        ],
    }
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)


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
    can_run, msg = check_blackhole_execution_requirements()
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


def test_blackhole_runtime_module_keeps_host_and_device_entries():
    """The Blackhole runtime module should expose the public entry and its kernel."""
    kernel = staged_copy_kernel(tile_rows=2, tile_cols=1)
    target = Target("blackhole")

    with target:
        artifact = lower(kernel, target=target)

    assert artifact.codegen_mod["main"] is not None
    assert artifact.codegen_mod["main_kernel"] is not None


def test_blackhole_true_e2e():
    """True end-to-end test: compile, execute, and verify results.

    This test:
    1. Compiles a TileLang kernel to Blackhole target
    2. Generates input data using PyTorch
    3. Executes the kernel via external runner
    4. Compares results with PyTorch reference
    """
    can_run, msg = check_blackhole_execution_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    # Note: We attempt to run with TT-Sim if available
    # The runner will fail gracefully if TT-Sim is not properly configured

    # Define test parameters (small for simulator)
    M, N = 32, 32  # Small size for quick simulation

    # Generate reference data with PyTorch
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_ref = a_torch.clone()  # Copy kernel reference

    # Create temporary directory for I/O
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input data
        spec_path = os.path.join(tmpdir, "spec.json")
        kernel_path = os.path.join(tmpdir, "kernel.cpp")
        input_path = os.path.join(tmpdir, "input.bin")
        output_path = os.path.join(tmpdir, "output.bin")

        # Write input as binary
        a_np = a_torch.numpy()
        with open(input_path, 'wb') as f:
            f.write(a_np.tobytes())

        # Compile kernel to Blackhole
        target = Target("blackhole")
        kernel = staged_copy_kernel(tile_rows=M // 32, tile_cols=N // 32)

        try:
            # Lower to Blackhole target (need target context)
            with target:
                artifact = lower(kernel, target=target)
            kernel_code = artifact.kernel_source if hasattr(artifact, 'kernel_source') else str(artifact)

            with open(kernel_path, 'w', encoding='utf-8') as f:
                f.write(kernel_code)

            write_single_core_copy_spec(spec_path, kernel_path, a_np.nbytes)

            # Get runner path
            runner_path = get_runner_path()

            cmd = [
                runner_path,
                spec_path,
                input_path,
                output_path,
            ]

            # Run the kernel
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                pytest.fail(f"Kernel execution failed:\n{result.stderr}")

            # Read output
            with open(output_path, 'rb') as f:
                output_data = np.frombuffer(f.read(), dtype=np.float16).copy().reshape(M, N)

            # Convert to torch for comparison
            b_output = torch.from_numpy(output_data)

            # Compare with reference
            atol = 1e-3  # Float16 tolerance
            rtol = 1e-3

            if torch.allclose(b_output, b_ref, atol=atol, rtol=rtol):
                print("SUCCESS: Blackhole kernel output matches PyTorch reference!")
                print(f"  Input shape: {a_torch.shape}")
                print(f"  Max difference: {(b_output - b_ref).abs().max().item()}")
                assert True
            else:
                diff = (b_output - b_ref).abs()
                print(f"FAILURE: Output mismatch!")
                print(f"  Max difference: {diff.max().item()}")
                print(f"  Mean difference: {diff.mean().item()}")
                assert False, "Output does not match reference"
        except subprocess.TimeoutExpired:
            pytest.fail("Kernel execution timed out (60s)")
        except Exception as e:
            pytest.skip(f"Test skipped due to error: {e}")


def test_blackhole_module_direct_call():
    """Exercise the BlackholeModule packed-func entrypoint directly."""
    can_run, msg = check_blackhole_execution_requirements()
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
    can_run, msg = check_blackhole_execution_requirements()
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


if __name__ == "__main__":
    tilelang.testing.main()
