"""
Check generated kernel source for Blackhole
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

print("=" * 60)
print("Checking Generated Kernel Source")
print("=" * 60)

target = Target('blackhole')

with target:
    artifact = lower(copy_with_shared, target=target, enable_device_compile=False)

print(f"\nKernel source length: {len(artifact.kernel_source)} bytes")
print(f"\nFull kernel source:")
print("=" * 60)
print(artifact.kernel_source)
print("=" * 60)

# Check for key components
source = artifact.kernel_source
print("\nComponent checks:")
print(f"  - Contains 'kernel_main': {'kernel_main' in source}")
print(f"  - Contains 'cb_reserve_back': {'cb_reserve_back' in source}")
print(f"  - Contains 'cb_push_back': {'cb_push_back' in source}")
print(f"  - Contains 'noc_async_read': {'noc_async_read' in source}")
print(f"  - Contains 'dataflow_api.h': {'dataflow_api.h' in source}")
print(f"  - Contains 'compute_kernel_api': {'compute_kernel_api' in source}")
