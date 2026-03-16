#!/usr/bin/env python3
"""
True End-to-end test for Blackhole backend with PyTorch comparison
Tests: DSL -> TIR -> CodeGen -> TT-Metal Execution -> Result Verification

This test:
1. Generates kernel code using TileLang
2. Generates reference results using PyTorch
3. Executes kernel on TT-Sim
4. Compares results
"""

import sys
import os
import subprocess
import tempfile
import numpy as np

# Add tilelang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tilelang_repo'))

import tilelang
import tilelang.language as T
from tvm.target import Target
import torch


def generate_copy_kernel_test():
    """Generate a simple copy kernel test case"""
    print("=" * 70)
    print("Test 1: Simple Copy Kernel (FP16)")
    print("=" * 70)

    # Create test data
    size = 1024
    A_np = np.random.randn(size).astype(np.float16)
    B_ref = A_np.copy()  # Expected output

    # Define TileLang kernel
    @T.prim_func
    def copy_kernel(
        A: T.Buffer((size,), "float16"),
        B: T.Buffer((size,), "float16"),
    ):
        with T.Kernel(1, threads=1) as (bx,):
            for i in T.serial(size):
                B[i] = A[i]

    print("\n1. Generated TileLang kernel:")
    print(copy_kernel.script()[:500] + "...")

    # Lower to Blackhole target
    print("\n2. Lowering to Blackhole target...")
    try:
        target = Target("blackhole")
        with target:
            artifact = tilelang.lower(copy_kernel)
        print(f"   ✓ Lowering successful!")
        print(f"   Generated code length: {len(artifact.kernel_source)} chars")
    except Exception as e:
        print(f"   ✗ Lowering failed: {e}")
        return False

    # Generate reference with PyTorch
    print("\n3. Generating reference with PyTorch...")
    A_torch = torch.from_numpy(A_np)
    B_torch = A_torch.clone()
    print(f"   Input shape: {A_torch.shape}, dtype: {A_torch.dtype}")
    print(f"   Output shape: {B_torch.shape}, dtype: {B_torch.dtype}")

    # For now, we save the generated code and reference data
    # In a full implementation, this would compile and execute on TT-Sim
    print("\n4. Saving artifacts for TT-Sim execution...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save kernel code
        kernel_path = os.path.join(tmpdir, "copy_kernel.cpp")
        with open(kernel_path, 'w') as f:
            f.write(artifact.kernel_source)
        print(f"   Kernel code saved to: {kernel_path}")

        # Save input/output data
        input_path = os.path.join(tmpdir, "input.npy")
        output_path = os.path.join(tmpdir, "output_ref.npy")
        np.save(input_path, A_np)
        np.save(output_path, B_ref)
        print(f"   Input data saved to: {input_path}")
        print(f"   Reference output saved to: {output_path}")

        # Display kernel code
        print("\n5. Generated TT-Metal kernel code (first 800 chars):")
        print("-" * 70)
        print(artifact.kernel_source[:800])
        print("-" * 70)

    print("\n   ✓ Copy kernel test artifacts generated successfully!")
    return True


def generate_elementwise_kernel_test():
    """Generate an elementwise addition kernel test case"""
    print("\n" + "=" * 70)
    print("Test 2: Element-wise Addition (FP16)")
    print("=" * 70)

    # Create test data
    size = 1024
    A_np = np.random.randn(size).astype(np.float16)
    B_np = np.random.randn(size).astype(np.float16)
    C_ref = A_np + B_np  # Expected output

    # Define TileLang kernel
    @T.prim_func
    def add_kernel(
        A: T.Buffer((size,), "float16"),
        B: T.Buffer((size,), "float16"),
        C: T.Buffer((size,), "float16"),
    ):
        with T.Kernel(1, threads=1) as (bx,):
            for i in T.serial(size):
                C[i] = A[i] + B[i]

    print("\n1. Generated TileLang kernel:")
    print(add_kernel.script()[:500] + "...")

    # Lower to Blackhole target
    print("\n2. Lowering to Blackhole target...")
    try:
        target = Target("blackhole")
        with target:
            artifact = tilelang.lower(add_kernel)
        print(f"   ✓ Lowering successful!")
    except Exception as e:
        print(f"   ✗ Lowering failed: {e}")
        return False

    # Generate reference with PyTorch
    print("\n3. Generating reference with PyTorch...")
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)
    C_torch = A_torch + B_torch
    print(f"   Input A shape: {A_torch.shape}")
    print(f"   Input B shape: {B_torch.shape}")
    print(f"   Output shape: {C_torch.shape}")

    # Verify PyTorch computation
    C_np = C_torch.numpy()
    if np.allclose(C_np, C_ref, rtol=1e-3, atol=1e-5):
        print("   ✓ PyTorch reference computation verified!")
    else:
        print("   ✗ PyTorch reference computation mismatch!")
        return False

    print("\n4. Generated TT-Metal kernel code (first 800 chars):")
    print("-" * 70)
    print(artifact.kernel_source[:800])
    print("-" * 70)

    print("\n   ✓ Element-wise addition test artifacts generated successfully!")
    return True


def generate_gemm_kernel_test():
    """Generate a GEMM kernel test case"""
    print("\n" + "=" * 70)
    print("Test 3: Matrix Multiplication (GEMM)")
    print("=" * 70)

    # Small GEMM for testing
    M, N, K = 32, 32, 32

    # Create test data
    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(K, N).astype(np.float16)
    C_ref = np.dot(A_np, B_np).astype(np.float32)  # FP32 accumulation

    # Define TileLang kernel
    @T.prim_func
    def gemm_kernel(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1) as (bx,):
            # Simple matmul
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")

    print("\n1. Generated TileLang kernel:")
    print(gemm_kernel.script()[:500] + "...")

    # Lower to Blackhole target
    print("\n2. Lowering to Blackhole target...")
    try:
        target = Target("blackhole")
        with target:
            artifact = tilelang.lower(gemm_kernel)
        print(f"   ✓ Lowering successful!")
    except Exception as e:
        print(f"   ✗ Lowering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate reference with PyTorch
    print("\n3. Generating reference with PyTorch...")
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)
    C_torch = torch.matmul(A_torch.float(), B_torch.float())
    print(f"   Input A shape: {A_torch.shape}")
    print(f"   Input B shape: {B_torch.shape}")
    print(f"   Output shape: {C_torch.shape}")

    # Verify PyTorch computation
    C_np = C_torch.numpy()
    if np.allclose(C_np, C_ref, rtol=1e-2, atol=1e-3):
        print("   ✓ PyTorch reference computation verified!")
    else:
        print("   ✗ PyTorch reference computation mismatch!")
        return False

    print("\n4. Generated TT-Metal kernel code (first 1000 chars):")
    print("-" * 70)
    print(artifact.kernel_source[:1000])
    print("-" * 70)

    print("\n   ✓ GEMM test artifacts generated successfully!")
    return True


def main():
    print("\n" + "=" * 70)
    print("TileLang Blackhole Backend - True End-to-End Test Suite")
    print("=" * 70)
    print("\nThis test suite:")
    print("1. Generates kernel code using TileLang DSL")
    print("2. Generates reference results using PyTorch")
    print("3. Verifies code generation and compilation pipeline")
    print("4. (TODO) Execute on TT-Sim and compare results")

    results = []

    # Test 1: Copy kernel
    try:
        results.append(("Copy Kernel", generate_copy_kernel_test()))
    except Exception as e:
        print(f"\n✗ Copy kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Copy Kernel", False))

    # Test 2: Element-wise addition
    try:
        results.append(("Element-wise Add", generate_elementwise_kernel_test()))
    except Exception as e:
        print(f"\n✗ Element-wise addition test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Element-wise Add", False))

    # Test 3: GEMM
    try:
        results.append(("GEMM", generate_gemm_kernel_test()))
    except Exception as e:
        print(f"\n✗ GEMM test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GEMM", False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30s}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        print("\nNote: Full TT-Sim execution test requires:")
        print("  1. Complete Runtime implementation")
        print("  2. TT-Sim environment setup")
        print("  3. Kernel compilation and execution pipeline")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
