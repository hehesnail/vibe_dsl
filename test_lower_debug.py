"""
Debug Blackhole lower process
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.lower import lower
from tvm.target import Target

print('Step 1: Creating PrimFunc...')

@T.prim_func
def test_kernel(A: T.Buffer((32, 32), 'float16'), B: T.Buffer((32, 32), 'float16')):
    with T.Kernel(1, 1) as (bx, by):
        for i, j in T.Parallel(32, 32):
            B[i, j] = A[i, j]

print(f'PrimFunc created: {test_kernel}')

print('\nStep 2: Creating target...')
target = Target('blackhole')
print(f'Target: {target}')

print('\nStep 3: Calling lower with target context...')
try:
    with target:
        artifact = lower(test_kernel, target=target)
    print(f'Lower completed!')
    print(f'Artifact type: {type(artifact)}')

    if hasattr(artifact, 'kernel_source'):
        print(f"Kernel source length: {len(artifact.kernel_source)} chars")
        print("\n=== First 1000 chars of generated kernel ===")
        print(artifact.kernel_source[:1000])
    elif hasattr(artifact, 'code'):
        print(f"Code length: {len(artifact.code)} chars")
        print("\n=== First 1000 chars ===")
        print(artifact.code[:1000])
    elif hasattr(artifact, 'device_mod'):
        print(f"Device mod: {artifact.device_mod}")
        print(f"Type: {type(artifact.device_mod)}")
    else:
        print(f"Artifact attributes: {[x for x in dir(artifact) if not x.startswith('_')]}")

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
