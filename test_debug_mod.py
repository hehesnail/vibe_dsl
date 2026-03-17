"""
Debug mod before Filter
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import is_device_call

# Simple kernel
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print("Step 1: Create module...")
from tvm import IRModule
mod = IRModule({"simple_kernel": simple_kernel})
print(f"Mod type: {type(mod)}")
print(f"Number of functions: {len(mod.functions)}")

print("\nStep 2: Create targets...")
target = Target('blackhole')
target_host = Target('c')
print(f"Target: {target}")
print(f"Target host: {target_host}")

print("\nStep 3: LowerAndLegalize...")
mod_lowered = LowerAndLegalize(mod, target)
print(f"Lowered mod type: {type(mod_lowered)}")
print(f"Number of functions: {len(mod_lowered.functions)}")

for gvar, func in mod_lowered.functions.items():
    print(f"\nFunction: {gvar}")
    print(f"Func attrs: {func.attrs}")
    if hasattr(func.attrs, 'dict'):
        print(f"Func attr dict: {func.attrs.dict}")
    is_dev = is_device_call(func)
    print(f"Is device call: {is_dev}")

print("\nStep 4: OptimizeForTarget...")
mod_optimized = OptimizeForTarget(mod_lowered, target)
print(f"Optimized mod type: {type(mod_optimized)}")
print(f"Number of functions: {len(mod_optimized.functions)}")

for gvar, func in mod_optimized.functions.items():
    print(f"\nFunction: {gvar}")
    print(f"Func attrs: {func.attrs}")
    if hasattr(func.attrs, 'dict'):
        print(f"Func attr dict: {func.attrs.dict}")
    is_dev = is_device_call(func)
    print(f"Is device call: {is_dev}")
