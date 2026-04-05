import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm.target import Target


@T.prim_func
def _stage0_seed_kernel(
    a: T.Buffer((16,), "float32"),
    b: T.Buffer((16,), "float32"),
):
    with T.Kernel(1, threads=32):
        for i in T.serial(16):
            b[i] = a[i]


def _prepare_blackhole_stage0_module():
    mod = tvm.IRModule({"main": _stage0_seed_kernel.with_attr("global_symbol", "main")})
    mod = tvm.tir.transform.BindTarget(Target("blackhole"))(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    return mod


def test_device_program_registry_is_collected_before_split_host_device():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.CollectDevicePrograms()(mod)

    registry = mod.global_infos["tl.device_programs"]
    assert len(registry) == 1
    assert registry[0].root_symbol == "main"
    assert list(registry[0].member_funcs) == ["main_kernel"]


def test_semantic_seeds_are_projected_before_semantic_lift():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.ProjectSemanticSeeds()(mod)

    seeds = mod["main"].attrs["tl.semantic_seeds"]
    assert list(seeds["device_kernel_regions"]) == ["main_kernel"]
    assert list(seeds["capture_kinds"]) == ["device_program_membership"]
    freeze = mod["main"].attrs["tl.semantic_hard_freeze"]
    assert str(freeze["unsafe_mutation_policy"]) == "invalidate_companion_programs"


def test_hard_freeze_invalidates_companion_programs_after_unsafe_mutation():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.ProjectSemanticSeeds()(mod)
    main = mod["main"].with_attr("tl.semantic_program", {"frozen": True})
    main = main.with_attr("tl.spatial_program", {"frozen": True})
    main = main.with_attr("tl.tt_program", {"frozen": True})
    mod.update_func(mod.get_global_var("main"), main)

    mod = tilelang.transform.InvalidateBlackholeCompanionPrograms("unit_test_unsafe_mutation")(mod)

    attrs = mod["main"].attrs
    assert "tl.semantic_program" not in attrs
    assert "tl.spatial_program" not in attrs
    assert "tl.tt_program" not in attrs
    assert str(attrs["tl.companion_invalidation_reason"]) == "unit_test_unsafe_mutation"
