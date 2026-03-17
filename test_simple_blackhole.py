"""
Simple Blackhole test without T.Parallel
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower
from tvm.target import Target

print('Step 1: Creating a simple kernel without Parallel...')

@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print(f'PrimFunc created')

print('\nStep 2: Creating target...')
target = Target('blackhole')
print(f'Target: {target}')

print('\nStep 3: Calling lower with target context...')
try:
    with target:
        artifact = lower(simple_kernel, target=target)
    print(f'Lower completed!')
    print(f'Artifact type: {type(artifact)}')

    if hasattr(artifact, 'kernel_source'):
        print(f"Kernel source length: {len(artifact.kernel_source)} chars")
        print("\n=== Generated kernel ===")
        print(artifact.kernel_source[:2000])
    elif hasattr(artifact, 'code'):
        print(f"Code length: {len(artifact.code)} chars")
        print("\n=== Generated code ===")
        print(artifact.code[:2000])
    elif hasattr(artifact, 'device_mod'):
        print(f"Device mod:\n{artifact.device_mod}")
    else:
        attrs = [x for x in dir(artifact) if not x.startswith('_')]
        print(f"Artifact attributes: {attrs}")

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
