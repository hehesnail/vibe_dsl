"""
Debug Blackhole passes
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import is_device_call

@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

from tvm import IRModule
mod = IRModule({"simple_kernel": simple_kernel})

print("Step 1: Create Blackhole target...")
target = Target('blackhole')

print("\nStep 2: LowerAndLegalize...")
mod = LowerAndLegalize(mod, target)

print("\nStep 3: OptimizeForTarget...")
mod = OptimizeForTarget(mod, target)

print("\nStep 4: Filter...")
from tvm import tir
from tilelang.engine.lower import is_cpu_device_backend, get_device_call

_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
device_mod = tir.transform.Filter(_is_device_call)(mod)

print("\nStep 5: Common passes...")
import tilelang.transform
device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
device_mod = tilelang.transform.LowerIntrin()(device_mod)
device_mod = tir.transform.Simplify()(device_mod)
device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

print("\nStep 6: LowerBlackholeOps...")
try:
    device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)
    print("  Success!")
    print(f"  Functions: {list(device_mod.functions.keys())}")
    for gvar, func in device_mod.functions.items():
        print(f"    {gvar}: body={str(func.body)[:200]}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDONE!")
