"""
Print full kernel source
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.lower import lower

@T.prim_func
def copy_with_shared(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
    A_shared = T.alloc_shared((256,), 'float16')
    for i in range(256):
        A_shared[i] = A[i]
    for i in range(256):
        B[i] = A_shared[i]

target = Target('blackhole')

with target:
    artifact = lower(copy_with_shared, target=target, enable_device_compile=False)

print('=' * 60)
print('Full Kernel Source')
print('=' * 60)
print(artifact.kernel_source)
print('=' * 60)
print(f'Kernel source length: {len(artifact.kernel_source)} bytes')
