"""
Investigate buffer scope for Copy Kernel
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.phase import LowerAndLegalize

@T.prim_func
def copy_kernel(A: T.Buffer((256,), 'float16'), B: T.Buffer((256,), 'float16')):
    for i in range(256):
        B[i] = A[i]

print("=" * 60)
print("Investigating Buffer Scope")
print("=" * 60)

mod = tvm.IRModule({"copy_kernel": copy_kernel})
target = Target('blackhole')

print("\n1. Original kernel buffer map:")
for gvar, func in mod.functions.items():
    for var, buf in func.buffer_map.items():
        var_name = str(var).split('.')[-1] if hasattr(var, '__str__') else str(var)
        scope_str = buf.scope()
        print(f"   {var_name} -> {buf.name}: scope='{scope_str}'")

print("\n2. After LowerAndLegalize:")
mod = LowerAndLegalize(mod, target)
for gvar, func in mod.functions.items():
    print(f"\n   Function: {gvar}")
    print(f"   Buffer map:")
    for var, buf in func.buffer_map.items():
        var_name = str(var).split('.')[-1] if hasattr(var, '__str__') else str(var)
        print(f"      {var_name} -> {buf.name}")
        print(f"      - scope: '{buf.scope()}'")
        print(f"      - dtype: {buf.dtype}")
        print(f"      - shape: {buf.shape}")
    print(f"\n   Body:\n   {str(func.body)[:300]}")

print("\n3. Checking for AttrStmt with storage scope...")
for gvar, func in mod.functions.items():
    # Look for AttrStmt in body
    def find_attr_stmt(stmt, depth=0):
        if hasattr(stmt, 'attr_key'):
            print(f"   {'  ' * depth}AttrStmt: {stmt.attr_key} = {stmt.node}")
        for child in stmt.__dict__.values():
            if hasattr(child, '__iter__') and not isinstance(child, str):
                for c in child:
                    if hasattr(c, '__class__'):
                        find_attr_stmt(c, depth+1)
            elif hasattr(child, '__class__'):
                find_attr_stmt(child, depth+1)

    find_attr_stmt(func.body)

print("\n4. Expected behavior:")
print("   - Input buffer (A) should have scope='' (global/DRAM)")
print("   - Output buffer (B) should have scope='shared' (CB)")
print("   - Copy direction: DRAM->CB when src=global, dst=shared")
