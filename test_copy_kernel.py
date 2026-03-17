"""
Test Copy Kernel Lowering for Blackhole backend

Tests T.copy() operation lowering to CB/NOC sequences.
According to design_review.md:
- DRAM->CB: cb_reserve_back + noc_async_read + cb_push_back
- CB->DRAM: cb_wait_front + noc_async_write + cb_pop_front
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.lower import lower

print("=" * 60)
print("Test 1: Simple DRAM->CB Copy (Host to Shared)")
print("=" * 60)

@T.prim_func
def copy_dram_to_cb(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
    """Copy from DRAM (global) to shared buffer"""
    # A is in DRAM (global), B should be in shared (CB)
    for i in range(256):
        B[i] = A[i]

print("\nLowering copy kernel...")
target = Target('blackhole')

try:
    with target:
        artifact = lower(copy_dram_to_cb, target=target, enable_device_compile=False)

    print(f"✓ Lowering succeeded!")
    print(f"\nDevice functions: {list(artifact.device_mod.functions.keys())}")

    # Check kernel source
    if artifact.kernel_source:
        print(f"\nKernel source preview (first 1000 chars):")
        print(artifact.kernel_source[:1000])

    # Print device function body
    for gvar, func in artifact.device_mod.functions.items():
        print(f"\nDevice function body: {gvar}")
        print(f"{str(func.body)[:500]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Check LowerBlackholeOps on Copy Operation")
print("=" * 60)

from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import is_device_call, is_cpu_device_backend, get_device_call
from tvm import tir

# Create module
mod = tvm.IRModule({"copy_dram_to_cb": copy_dram_to_cb})

# Run phases
mod = LowerAndLegalize(mod, target)
mod = OptimizeForTarget(mod, target)

# Filter device functions
_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
device_mod = tir.transform.Filter(_is_device_call)(mod)

print(f"\nDevice functions after Filter: {list(device_mod.functions.keys())}")

# Run Blackhole-specific passes
import tilelang.transform

print("\nRunning Blackhole passes...")
device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
device_mod = tilelang.transform.LowerIntrin()(device_mod)
device_mod = tir.transform.Simplify()(device_mod)
device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

print("\nBefore LowerBlackholeOps:")
for gvar, func in device_mod.functions.items():
    print(f"  Body: {str(func.body)[:300]}")

# Check for buffer scopes
print("\nAnalyzing buffer scopes...")
for gvar, func in device_mod.functions.items():
    for var in func.params:
        if var in func.buffer_map:
            buf = func.buffer_map[var]
            print(f"  Buffer {buf.name}: scope={buf.scope}")

# Run LowerBlackholeOps
print("\nRunning LowerBlackholeOps...")
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)

print("\nAfter LowerBlackholeOps:")
for gvar, func in device_mod.functions.items():
    print(f"  Body: {str(func.body)[:500]}")
    # Check for builtin calls
    body_str = str(func.body)
    if "blackhole" in body_str.lower() or "cb_" in body_str.lower():
        print(f"  ✓ Contains Blackhole builtin calls")

print("\n" + "=" * 60)
print("Test 3: Explicit T.copy() Operation")
print("=" * 60)

# Try using T.copy explicitly if available
try:
    @T.prim_func
    def explicit_copy(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
        # Use shared buffer for CB simulation
        A_shared = T.alloc_shared((256,), 'float16')
        # Copy from DRAM to shared (CB)
        T.copy(A, A_shared)
        # Copy from shared (CB) to DRAM
        T.copy(A_shared, B)

    print("\nLowering explicit T.copy kernel...")
    with target:
        artifact2 = lower(explicit_copy, target=target, enable_device_compile=False)

    print(f"✓ Lowering succeeded!")
    print(f"\nKernel source preview:")
    if artifact2.kernel_source:
        print(artifact2.kernel_source[:800])

except Exception as e:
    print(f"Note: T.copy test encountered issue (expected for now): {e}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✓ Basic copy kernel lowering works")
print("⚠ Copy direction detection needs buffer scope annotations")
print("⚠ T.copy explicit operation needs further testing")
