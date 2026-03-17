"""
Analyze IR structure to find the infinite recursion cause
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.lower import lower
from tvm.target import Target

# Test 1: Simple for loop (works)
@T.prim_func
def simple_for(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

# Test 2: With T.Kernel (may cause issues)
@T.prim_func
def kernel_with_grid(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    with T.Kernel(1) as bx:
        for i in range(32):
            B[bx * 32 + i] = A[bx * 32 + i]

# Test 3: With T.Parallel (may cause issues)
@T.prim_func
def kernel_with_parallel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in T.Parallel(32):
        B[i] = A[i]

print("=== Test 1: Simple for loop ===")
try:
    target = Target('cuda')  # Use CUDA for initial lowering to see IR
    with target:
        artifact = lower(simple_for, target=target)
    print("SUCCESS")
    # Print TIR
    if hasattr(artifact, 'device_mod'):
        print("TIR:")
        print(str(artifact.device_mod)[:2000])
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test 2: With T.Kernel ===")
try:
    target = Target('cuda')
    with target:
        artifact = lower(kernel_with_grid, target=target)
    print("SUCCESS")
    if hasattr(artifact, 'device_mod'):
        print("TIR:")
        print(str(artifact.device_mod)[:2000])
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test 3: With T.Parallel ===")
try:
    target = Target('cuda')
    with target:
        artifact = lower(kernel_with_parallel, target=target)
    print("SUCCESS")
    if hasattr(artifact, 'device_mod'):
        print("TIR:")
        print(str(artifact.device_mod)[:2000])
except Exception as e:
    print(f"FAILED: {e}")
