"""
Analyze Blackhole lower IR structure
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.lower import lower

# Simple kernel that previously worked
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print("Creating target...")
target = Target('blackhole')
print(f"Target: {target}")

print("\nLowering (without Blackhole passes)...")
# First lower without target to see initial IR
try:
    # Use cuda target to get base IR
    base_target = Target('cuda')
    with base_target:
        artifact = lower(simple_kernel, target=base_target)
    print(f"Base lower succeeded")
    if hasattr(artifact, 'device_mod'):
        print("\nBase TIR:")
        print(str(artifact.device_mod))
except Exception as e:
    print(f"Base lower failed: {e}")

print("\n" + "="*60)
print("Lowering with Blackhole target...")
try:
    with target:
        artifact = lower(simple_kernel, target=target)
    print(f"Blackhole lower succeeded")

    if hasattr(artifact, 'device_mod'):
        print("\nBlackhole TIR:")
        print(artifact.device_mod)

    # Inspect artifact
    attrs = [x for x in dir(artifact) if not x.startswith('_')]
    print(f"\nArtifact attributes: {attrs}")

    for attr in attrs:
        val = getattr(artifact, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)} = {val if not isinstance(val, str) or len(val) < 100 else val[:100] + '...'}")

except Exception as e:
    print(f"Blackhole lower failed: {e}")
    import traceback
    traceback.print_exc()
