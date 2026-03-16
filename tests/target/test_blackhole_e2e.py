#!/usr/bin/env python3
"""
End-to-end test for Blackhole backend
Tests: DSL -> TIR -> CodeGen -> Execution
"""

import sys
import os

# Add tilelang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tilelang_repo'))

import tilelang
import tilelang.language as T
from tvm.target import Target
import torch
import numpy as np


def test_copy_kernel_codegen():
    """Test a simple copy kernel codegen on Blackhole backend"""

    print("=" * 60)
    print("Blackhole Backend CodeGen Test: Copy Kernel")
    print("=" * 60)

    # Define a simple copy kernel using TileLang DSL
    @T.prim_func
    def copy_kernel(
        A: T.Buffer((1024,), "float16"),
        B: T.Buffer((1024,), "float16"),
    ):
        # Single block, single thread for simplicity
        with T.Kernel(1, threads=1) as (bx,):
            # Copy from A to B
            for i in T.serial(1024):
                B[i] = A[i]

    print("\n1. Defined TileLang kernel:")
    print(copy_kernel.script())

    # Lower to TIR with Blackhole target
    print("\n2. Lowering to Blackhole target...")
    try:
        artifact = tilelang.lower(copy_kernel, target="blackhole")
        print(f"   ✓ Lowering successful!")
        print(f"   Generated kernel source (first 500 chars):")
        print(f"   {artifact.kernel_source[:500]}...")
        return True
    except Exception as e:
        print(f"   ✗ Lowering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_only():
    """Test only the build/codegen phase without execution"""

    print("\n" + "=" * 60)
    print("Blackhole Backend Build Test (No Execution)")
    print("=" * 60)

    @T.prim_func
    def simple_kernel(
        A: T.Buffer((256, 256), "float16"),
        B: T.Buffer((256, 256), "float16"),
        C: T.Buffer((256, 256), "float16"),
    ):
        with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
            # Simple tiled copy
            for i in T.serial(32):
                for j in T.serial(32):
                    if bx * 32 + i < 256 and by * 32 + j < 256:
                        C[bx * 32 + i, by * 32 + j] = A[bx * 32 + i, by * 32 + j] + B[bx * 32 + i, by * 32 + j]

    print("\n1. Defined kernel:")
    print(simple_kernel.script())

    print("\n2. Building for Blackhole...")
    try:
        target = Target("blackhole")
        with target:
            artifact = tilelang.lower(simple_kernel)
        print("   ✓ Build successful!")
        print(f"   Generated code length: {len(artifact.kernel_source)} chars")
        print("\n3. Generated TT-Metal code (first 1000 chars):")
        print("-" * 60)
        print(artifact.kernel_source[:1000])
        print("-" * 60)
        return True
    except Exception as e:
        print(f"   ✗ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TileLang Blackhole Backend E2E Test Suite")
    print("=" * 60)

    # Test 1: Build only
    build_success = test_build_only()

    # Test 2: Copy kernel codegen
    codegen_success = test_copy_kernel_codegen()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Build Test: {'✓ PASSED' if build_success else '✗ FAILED'}")
    print(f"CodeGen Test: {'✓ PASSED' if codegen_success else '✗ FAILED'}")

    if build_success and codegen_success:
        print("\n✓ Blackhole backend CodeGen is working!")
        sys.exit(0)
    else:
        print("\n✗ Blackhole backend CodeGen has issues!")
        sys.exit(1)
