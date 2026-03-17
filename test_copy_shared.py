"""
Test Copy Kernel with shared buffer allocation
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import is_device_call, is_cpu_device_backend, get_device_call
from tvm import tir
import tilelang.transform

print("=" * 60)
print("Test: Copy with alloc_shared buffer")
print("=" * 60)

@T.prim_func
def copy_with_shared(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
    # Allocate shared buffer (should have scope='shared')
    A_shared = T.alloc_shared((256,), 'float16')

    # Copy A (global) -> A_shared (shared): DRAM -> CB
    for i in range(256):
        A_shared[i] = A[i]

    # Copy A_shared (shared) -> B (global): CB -> DRAM
    for i in range(256):
        B[i] = A_shared[i]

# Check buffer scopes
mod = tvm.IRModule({"copy_with_shared": copy_with_shared})
target = Target('blackhole')

print("\n1. After LowerAndLegalize:")
mod = LowerAndLegalize(mod, target)

for gvar, func in mod.functions.items():
    print(f"\n   Function: {gvar}")
    print("   Buffer map:")
    for var, buf in func.buffer_map.items():
        print(f"      {buf.name}: scope='{buf.scope()}'")

print("\n2. After OptimizeForTarget:")
mod = OptimizeForTarget(mod, target)

_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
device_mod = tir.transform.Filter(_is_device_call)(mod)

print(f"   Device functions: {list(device_mod.functions.keys())}")

print("\n3. Running Blackhole passes...")
device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
device_mod = tilelang.transform.LowerIntrin()(device_mod)
device_mod = tir.transform.Simplify()(device_mod)
device_mod = tilelang.transform.LowerOpaqueBlock()(device_mod)  # Lower BlockRealize
device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

print("\n   Before LowerBlackholeOps:")
for gvar, func in device_mod.functions.items():
    print(f"   Body:\n{str(func.body)[:500]}")

print("\n4. Running LowerBlackholeOps...")
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)

print("\n   After LowerBlackholeOps:")
for gvar, func in device_mod.functions.items():
    body_str = str(func.body)
    print(f"   Body:\n{body_str[:800]}")

    # Check for builtin calls
    if "blackhole" in body_str.lower():
        print("\n   ✓ Contains 'blackhole' builtin calls")
    if "cb_" in body_str.lower():
        print("   ✓ Contains 'cb_' calls")
    if "noc_" in body_str.lower():
        print("   ✓ Contains 'noc_' calls")

print("\n5. Full lowering test:")
from tilelang.engine.lower import lower

try:
    with target:
        artifact = lower(copy_with_shared, target=target, enable_device_compile=False)

    print(f"   ✓ Full lowering succeeded!")
    print(f"\n   Kernel source (first 800 chars):")
    if artifact.kernel_source:
        print(artifact.kernel_source[:800])

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
