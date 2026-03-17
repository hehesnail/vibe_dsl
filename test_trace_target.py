"""
Trace target through phases
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
mod = IRModule({'simple_kernel': simple_kernel})

target = Target('blackhole')

mod = LowerAndLegalize(mod, target)
print('After LowerAndLegalize:')
for gvar, func in mod.functions.items():
    print(f'  Function: {gvar}')
    print(f'  Target: {func.attrs.get("target")}')
    print(f'  Global symbol: {func.attrs.get("global_symbol")}')

mod = OptimizeForTarget(mod, target)
print('\nAfter OptimizeForTarget:')
for gvar, func in mod.functions.items():
    print(f'  Function: {gvar}')
    print(f'  Target: {func.attrs.get("target")}')
    print(f'  Is device: {is_device_call(func)}')
    print(f'  Body type: {type(func.body)}')
    # Try to see what's in the body
    body_str = str(func.body)
    print(f'  Body preview: {body_str[:500]}')
