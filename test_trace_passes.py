"""
Trace each pass in OptimizeForTarget
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.lower import is_device_call

@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

from tvm import IRModule
mod = IRModule({'simple_kernel': simple_kernel})

target = Target('blackhole')

# From LowerAndLegalize - need BindTarget first!
mod = tvm.tir.transform.BindTarget(target)(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.LowerTileOp()(mod)
mod = tilelang.transform.LowerL2Persistent()(mod)
mod = tilelang.transform.DecoupleTypeCast()(mod)
mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
mod = tilelang.transform.LowerAccessPtr()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.HoistNonRestrictParams()(mod)

print("After LowerAndLegalize passes:")
for gvar, func in mod.functions.items():
    print(f"  {gvar}: target={func.attrs.get('target')}")

# Now trace each pass in OptimizeForTarget
passes = [
    ("LowerSharedBarrier", lambda m: tilelang.transform.LowerSharedBarrier()(m)),
    ("LowerSharedTmem", lambda m: tilelang.transform.LowerSharedTmem()(m)),
    ("IfStmtBinding", lambda m: tilelang.transform.IfStmtBinding()(m)),
    ("PlanAndUpdateBufferAllocationLocation", lambda m: tilelang.transform.PlanAndUpdateBufferAllocationLocation()(m)),
    ("PipelinePlanning", lambda m: tilelang.transform.PipelinePlanning()(m)),
    ("InjectSoftwarePipeline", lambda m: tilelang.transform.InjectSoftwarePipeline()(m)),
    ("LowerOpaqueBlock", lambda m: tilelang.transform.LowerOpaqueBlock()(m)),
    ("Simplify", lambda m: tilelang.transform.Simplify()(m)),
    ("OptimizeCPAsyncSync", lambda m: tilelang.transform.OptimizeCPAsyncSync()(m)),
    ("Simplify2", lambda m: tilelang.transform.Simplify()(m)),
    ("NarrowDataType", lambda m: tvm.tir.transform.NarrowDataType(32)(m)),
    ("FlattenBuffer", lambda m: tilelang.transform.FlattenBuffer()(m)),
    ("ConfigIndexBitwidth", lambda m: tilelang.transform.ConfigIndexBitwidth()(m)),
    ("Simplify3", lambda m: tvm.tir.transform.Simplify()(m)),
    ("RemoveNoOp", lambda m: tvm.tir.transform.RemoveNoOp()(m)),
    ("HoistIfThenElse", lambda m: tvm.tir.transform.HoistIfThenElse()(m)),
    ("VerifyMemory", lambda m: tvm.tir.transform.VerifyMemory()(m)),
    ("AnnotateEntryFunc", lambda m: tvm.tir.transform.AnnotateEntryFunc()(m)),
    ("InferFragment", lambda m: tilelang.transform.InferFragment()(m)),
    ("LowerThreadAllreduce", lambda m: tilelang.transform.LowerThreadAllreduce()(m)),
    ("LowerLDGSTG", lambda m: tilelang.transform.LowerLDGSTG()(m)),
    ("LowerHopperIntrin", lambda m: tilelang.transform.LowerHopperIntrin()(m)),
    ("AnnotateDeviceRegions", lambda m: tilelang.transform.AnnotateDeviceRegions()(m)),
]

print("\nTracing passes:")
for name, pass_fn in passes:
    try:
        mod = pass_fn(mod)
        for gvar, func in mod.functions.items():
            target_val = func.attrs.get('target')
            if target_val:
                print(f"  {name}: target={target_val.kind.name}")
            else:
                print(f"  {name}: no target")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\nAfter initial passes, before remaining passes:")
for gvar, func in mod.functions.items():
    print(f"  {gvar}: target={func.attrs.get('target')}, is_device={is_device_call(func)}")

# Continue with remaining passes
remaining_passes = [
    ("AnnotateReadOnlyParams", lambda m: tilelang.transform.AnnotateReadOnlyParams()(m)),
    ("MergeSharedMemoryAllocations", lambda m: tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=False)(m)),
    ("ThreadSync_shared", lambda m: tilelang.transform.ThreadSync("shared")(m)),
    ("ThreadSync_shared_dyn", lambda m: tilelang.transform.ThreadSync("shared.dyn")(m)),
    ("MergeIfStmt", lambda m: tilelang.transform.MergeIfStmt()(m)),
    ("MakePackedAPI", lambda m: tilelang.transform.MakePackedAPI()(m)),
    ("Simplify_final", lambda m: tilelang.transform.Simplify()(m)),
    ("LowerDeviceKernelLaunch", lambda m: tilelang.transform.LowerDeviceKernelLaunch()(m)),
    ("PersistThreadblock", lambda m: tilelang.transform.PersistThreadblock()(m)),
]

print("\nTracing remaining passes:")
for name, pass_fn in remaining_passes:
    try:
        mod = pass_fn(mod)
        for gvar, func in mod.functions.items():
            target_val = func.attrs.get('target')
            if target_val:
                print(f"  {name}: target={target_val.kind.name}")
            else:
                print(f"  {name}: no target")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\nAfter all passes:")
for gvar, func in mod.functions.items():
    print(f"  {gvar}: target={func.attrs.get('target')}, is_device={is_device_call(func)}")
