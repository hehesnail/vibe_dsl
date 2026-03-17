"""
Test current Blackhole backend status
"""
import sys
import os

sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower
from tvm.target import Target

@T.prim_func
def test_kernel(A: T.Buffer((32, 32), 'float16'), B: T.Buffer((32, 32), 'float16')):
    with T.Kernel(1, 1) as (bx, by):
        for i, j in T.Parallel(32, 32):
            B[i, j] = A[i, j]

print("Testing Blackhole backend...")
target = Target('blackhole')

print(f"Target: {target}")
print(f"Target kind: {target.kind.name}")

try:
    with target:
        artifact = lower(test_kernel, target=target)
    print('Lower successful!')
    print(f'Artifact type: {type(artifact)}')

    if hasattr(artifact, 'kernel_source'):
        print(f"Kernel source length: {len(artifact.kernel_source)} chars")
        print("\n=== First 500 chars of generated kernel ===")
        print(artifact.kernel_source[:500])
    elif hasattr(artifact, 'code'):
        print(f"Code length: {len(artifact.code)} chars")
        print("\n=== First 500 chars ===")
        print(artifact.code[:500])
    else:
        print(f"Artifact attributes: {dir(artifact)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
