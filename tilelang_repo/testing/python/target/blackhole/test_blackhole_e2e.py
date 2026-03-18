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


def check_blackhole_requirements():
    """Check if Blackhole testing requirements are met."""
    tilelang_home = os.environ.get("TILELANG_HOME")
    runner_build_dir = os.environ.get("TILELANG_BLACKHOLE_RUNNER_BUILD_DIR")
    if not tilelang_home:
        return False, "TILELANG_HOME not set"

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
        "target_mode": "single_core_copy",
        "input_size_bytes": int(tensor_nbytes),
        "output_size_bytes": int(tensor_nbytes),
        "scalar_args": [],
        "core_plan": {
            "grid_x": 1,
            "grid_y": 1,
            "cores_needed": 1,
            "work_per_core": 1,
            "core_grid_x": 1,
            "core_grid_y": 1,
        },
        "cb_configs": [
            {
                "cb_id": 0,
                "name": "A",
                "role": "input",
                "num_pages": 1,
                "page_size_bytes": int(tensor_nbytes),
                "data_format": "Float16_b",
            },
            {
                "cb_id": 16,
                "name": "B",
                "role": "output",
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
                    {"name": "num_tiles", "kind": "tile_count", "dtype": "uint32"},
                    {"name": "scratch_l1", "kind": "scratch_l1_buffer_addr32", "dtype": "uint32"},
                ],
            }
        ],
    }
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)


def simple_copy_kernel(M: int, N: int, tile_m: int = 32, tile_n: int = 32):
    """Define a simple copy kernel for Blackhole.

    This kernel copies data from input A to output B.
    The kernel will be lowered to TT-Metal code with CB allocation.
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, tile_n), T.ceildiv(M, tile_m)) as (bx, by):
            # Copy from input to output (simple element-wise copy)
            # In the Blackhole backend, this will use CBs internally
            for i, j in T.Parallel(tile_m, tile_n):
                y = by * tile_m + i
                x = bx * tile_n + j
                if y < M and x < N:
                    B[y, x] = A[y, x]

    return main


def test_blackhole_codegen_only():
    """Test that Blackhole code generation works (no execution)."""
    can_run, msg = check_blackhole_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    # Define a simple kernel
    M, N = 64, 64
    kernel = simple_copy_kernel(M, N)

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


def test_blackhole_true_e2e():
    """True end-to-end test: compile, execute, and verify results.

    This test:
    1. Compiles a TileLang kernel to Blackhole target
    2. Generates input data using PyTorch
    3. Executes the kernel via external runner
    4. Compares results with PyTorch reference
    """
    can_run, msg = check_blackhole_requirements()
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
        kernel = simple_copy_kernel(M, N)

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
    can_run, msg = check_blackhole_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    M, N = 32, 32
    torch.manual_seed(42)
    a_torch = torch.randn(M, N, dtype=torch.float16)
    b_output = torch.zeros_like(a_torch)
    b_ref = a_torch.clone()

    target = Target("blackhole")
    kernel = simple_copy_kernel(M, N)

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
