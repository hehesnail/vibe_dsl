"""
Debug LowerBlackholeOps - check host_mod
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.lower import lower

# Simple kernel that works
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print("Step 1: Create Blackhole target...")
target = Target('blackhole')

print("\nStep 2: Lower with enable_device_compile=False...")
try:
    with target:
        artifact = lower(simple_kernel, target=target, enable_device_compile=False)
    print(f"Lower succeeded!")

    # Check all attributes
    attrs = [x for x in dir(artifact) if not x.startswith('_')]
    print(f"\nArtifact attributes: {attrs}")

    for attr in attrs:
        val = getattr(artifact, attr)
        if not callable(val):
            if isinstance(val, str):
                print(f"  {attr}: str = {val[:200] if len(val) > 200 else val}")
            else:
                print(f"  {attr}: {type(val)} = {val}")

    # Check host_mod
    if hasattr(artifact, 'host_mod') and artifact.host_mod is not None:
        print(f"\n\nHost mod:")
        print(artifact.host_mod)

        print(f"\nNumber of functions in host_mod: {len(artifact.host_mod.functions)}")
        for gvar, func in artifact.host_mod.functions.items():
            print(f"\nHost Function: {gvar}")
            print(f"Type: {type(func)}")
            if hasattr(func, 'body'):
                print(f"Body snippet: {str(func.body)[:500]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
