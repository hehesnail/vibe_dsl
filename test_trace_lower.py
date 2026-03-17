"""
Trace through lower() function step by step
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.engine.lower import is_device_call, is_cpu_device_backend, get_host_call, get_device_call

@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

from tvm import IRModule
mod = IRModule({'simple_kernel': simple_kernel})

print("Step 1: Setup targets...")
target = Target('blackhole')
target_host = "c"

print(f"  target: {target}")
print(f"  target.kind.name: {target.kind.name}")
print(f"  is_cpu_device_backend: {is_cpu_device_backend(target)}")

# This is what lower() does with targets
target_host = tvm.target.Target.canon_target(target_host)
target = tvm.target.Target(target, target_host)

print(f"\n  After canon:")
print(f"  target: {target}")
print(f"  target_host: {target_host}")

_is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

print(f"\nStep 2: PreLowerSemanticCheck...")
PreLowerSemanticCheck(mod)

print(f"\nStep 3: LowerAndLegalize...")
mod = LowerAndLegalize(mod, target)
for gvar, func in mod.functions.items():
    print(f"  {gvar}: target={func.attrs.get('target')}, is_device={is_device_call(func)}")

print(f"\nStep 4: OptimizeForTarget...")
mod = OptimizeForTarget(mod, target)
for gvar, func in mod.functions.items():
    print(f"  {gvar}: target={func.attrs.get('target')}, is_device={is_device_call(func)}")
    print(f"    is_host_call result: {_is_host_call(func)}")
    print(f"    is_device_call result: {_is_device_call(func)}")

print(f"\nStep 5: Filter...")
from tvm import tir
host_mod = tir.transform.Filter(_is_host_call)(mod)
device_mod = tir.transform.Filter(_is_device_call)(mod)

print(f"  Host functions: {list(host_mod.functions.keys())}")
for gvar, func in host_mod.functions.items():
    print(f"    {gvar}: target={func.attrs.get('target')}")

print(f"  Device functions: {list(device_mod.functions.keys())}")
for gvar, func in device_mod.functions.items():
    print(f"    {gvar}: target={func.attrs.get('target')}")
