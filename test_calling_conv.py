"""
Check calling convention of device functions
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

@T.prim_func
def copy_with_shared(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
    A_shared = T.alloc_shared((256,), 'float16')
    for i in range(256):
        A_shared[i] = A[i]
    for i in range(256):
        B[i] = A_shared[i]

mod = tvm.IRModule({'copy_with_shared': copy_with_shared})
target = Target('blackhole')
mod = LowerAndLegalize(mod, target)
mod = OptimizeForTarget(mod, target)

print("All functions after OptimizeForTarget:")
for gvar, func in mod.functions.items():
    print(f"  {gvar}:")
    calling_conv = func.attrs.get('calling_conv') if func.attrs else None
    print(f"    calling_conv: {calling_conv}")
    global_symbol = func.attrs.get('global_symbol') if func.attrs else None
    print(f"    global_symbol: {global_symbol}")

_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
device_mod = tir.transform.Filter(_is_device_call)(mod)

print('\nDevice functions after Filter:')
for gvar, func in device_mod.functions.items():
    print(f"  {gvar}:")
    calling_conv = func.attrs.get('calling_conv') if func.attrs else None
    print(f"    calling_conv: {calling_conv}")
