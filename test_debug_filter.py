"""
Debug Filter result
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import is_device_call, get_host_call, get_device_call, is_cpu_device_backend
from tvm import tir

# Simple kernel
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

from tvm import IRModule
mod = IRModule({"simple_kernel": simple_kernel})

print("Step 1: Create targets...")
target = Target('blackhole')
target_host = Target('c')
print(f"Target: {target}")
print(f"Target host: {target_host}")

print("\nStep 2: LowerAndLegalize...")
mod = LowerAndLegalize(mod, target)
print(f"Functions: {list(mod.functions.keys())}")
for gvar, func in mod.functions.items():
    print(f"  {gvar}: is_device={is_device_call(func)}")

print("\nStep 3: OptimizeForTarget...")
mod = OptimizeForTarget(mod, target)
print(f"Functions: {list(mod.functions.keys())}")
for gvar, func in mod.functions.items():
    print(f"  {gvar}: is_device={is_device_call(func)}")

print("\nStep 4: Filter...")
is_device_c = is_cpu_device_backend(target)
print(f"is_device_c: {is_device_c}")

_is_host_call = get_host_call(is_device_c=is_device_c)
_is_device_call = get_device_call(is_device_c=is_device_c)

host_mod = tir.transform.Filter(_is_host_call)(mod)
device_mod = tir.transform.Filter(_is_device_call)(mod)

print(f"Host functions: {list(host_mod.functions.keys())}")
print(f"Device functions: {list(device_mod.functions.keys())}")

for gvar, func in host_mod.functions.items():
    print(f"  Host {gvar}: target={func.attrs.get('target')}")

for gvar, func in device_mod.functions.items():
    print(f"  Device {gvar}: target={func.attrs.get('target')}")
