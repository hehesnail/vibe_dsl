import os

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import _align_blackhole_device_symbol
from tilelang.engine.phase import LowerToBlackholePhaseB


def find_loop_annotation(stmt, attr_key):
    """Find the first For loop carrying the given annotation key."""
    if isinstance(stmt, tilelang.tvm.tir.For) and stmt.annotations.get(attr_key) is not None:
        return stmt.annotations[attr_key]
    if isinstance(stmt, tilelang.tvm.tir.BlockRealize):
        return find_loop_annotation(stmt.block.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.Block):
        return find_loop_annotation(stmt.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.SeqStmt):
        for child in stmt.seq:
            found = find_loop_annotation(child, attr_key)
            if found is not None:
                return found
        return None
    if isinstance(stmt, tilelang.tvm.tir.IfThenElse):
        found = find_loop_annotation(stmt.then_case, attr_key)
        if found is not None:
            return found
        if stmt.else_case is not None:
            return find_loop_annotation(stmt.else_case, attr_key)
        return None
    if hasattr(stmt, "body"):
        return find_loop_annotation(stmt.body, attr_key)
    return None


def contains_attr_stmt_key(stmt, attr_key):
    """Return whether a TIR stmt tree still carries the given AttrStmt key."""
    found = False

    def visit(node):
        nonlocal found
        if found:
            return
        if isinstance(node, tilelang.tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            found = True

    tilelang.tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def check_blackhole_codegen_requirements():
    """Check if Blackhole compilation requirements are met."""
    tilelang_home = os.environ.get("TILELANG_HOME")
    if not tilelang_home:
        return False, "TILELANG_HOME not set"
    return True, "OK"


def get_loaded_tilelang_cmake_cache():
    lib_path = getattr(tilelang, "_LIB_PATH", None)
    if not lib_path:
        return None
    build_dir = os.path.dirname(os.path.dirname(lib_path))
    return os.path.join(build_dir, "CMakeCache.txt")


def direct_build_enabled():
    cache_candidates = []
    loaded_cache = get_loaded_tilelang_cmake_cache()
    if loaded_cache:
        cache_candidates.append(loaded_cache)

    tilelang_home = os.environ.get("TILELANG_HOME")
    if tilelang_home:
        cache_candidates.append(os.path.join(tilelang_home, "build", "CMakeCache.txt"))

    for cache_path in cache_candidates:
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_text = f.read()
            if "USE_BLACKHOLE_DIRECT" in cache_text and "=ON" in cache_text:
                return True
    return False


def check_blackhole_direct_execution_requirements():
    """Check if BlackholeModule direct execution requirements are met."""
    can_codegen, msg = check_blackhole_codegen_requirements()
    if not can_codegen:
        return False, msg

    tt_metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_runtime_root:
        return False, "TT_METAL_RUNTIME_ROOT not set"
    if not os.path.isdir(os.path.join(tt_metal_runtime_root, "tt_metal")):
        return False, f"TT_METAL_RUNTIME_ROOT does not contain tt_metal/: {tt_metal_runtime_root}"

    if not direct_build_enabled():
        return False, "TileLang build does not appear to have USE_BLACKHOLE_DIRECT=ON"

    return True, "OK"


def prepare_blackhole_phase_b_module(mod):
    """Run the Blackhole Phase B mainline up to SpatialPlan plus lowering facts."""
    return tilelang.engine.phase.LowerToBlackholePhaseB(mod)


def lower_blackhole_to_tt_target(mod):
    """Lower a Blackhole module through the validated TTProgram target contract."""
    source_mod = LowerToBlackholePhaseB(mod)
    mod = _align_blackhole_device_symbol(source_mod, mod)
    return tilelang.engine.phase.LowerToBlackholeTTProgram(mod)


def require_tt_program(func):
    """Return the materialized TTProgram attached to a PrimFunc."""
    if not (func.attrs and "tl.tt_program" in func.attrs):
        pytest.fail("Expected PrimFunc to carry tl.tt_program")
    return func.attrs["tl.tt_program"]


def _try_plain_dict(value):
    try:
        return dict(value)
    except (TypeError, ValueError):
        return None


def tt_launch_spec_to_dict(launch_spec):
    if launch_spec is None:
        return {}
    plain = _try_plain_dict(launch_spec)
    if plain is not None:
        return {
            "core_type": str(plain.get("core_type", "")),
            "processor": str(plain.get("processor", "")),
            "noc": str(plain.get("noc", "")),
        }
    if not hasattr(launch_spec, "core_type") or not str(launch_spec.core_type):
        return {}
    return {
        "core_type": str(launch_spec.core_type),
        "processor": str(launch_spec.processor),
        "noc": str(launch_spec.noc),
    }


def tt_compute_config_to_dict(compute_config):
    if compute_config is None:
        return {}
    plain = _try_plain_dict(compute_config)
    if plain is not None:
        return dict(plain)
    if not hasattr(compute_config, "math_fidelity") or not str(compute_config.math_fidelity):
        return {}
    return {
        "math_fidelity": str(compute_config.math_fidelity),
        "fp32_dest_acc_en": bool(compute_config.fp32_dest_acc_en),
        "dst_full_sync_en": bool(compute_config.dst_full_sync_en),
        "math_approx_mode": bool(compute_config.math_approx_mode),
        "unpack_to_dest_mode": [str(item) for item in compute_config.unpack_to_dest_mode],
        "bfp8_pack_precise": bool(compute_config.bfp8_pack_precise),
        "defines": [
            {"name": str(item.name), "value": str(item.value)}
            for item in compute_config.defines
        ],
        "named_compile_args": [
            {"name": str(item.name), "value": int(item.value)}
            for item in compute_config.named_compile_args
        ],
        "clear_accum": bool(compute_config.clear_accum),
        "k_pack": int(compute_config.k_pack),
        "wg_wait": int(compute_config.wg_wait),
        "policy_type": int(compute_config.policy_type),
        "policy_name": str(compute_config.policy_name),
    }


def tt_per_work_arg_specs_to_list(per_work_arg_specs):
    encoded = []
    for spec in per_work_arg_specs or []:
        plain = _try_plain_dict(spec)
        if plain is not None:
            encoded.append(dict(plain))
            continue
        item = {
            "arg_kind": str(spec.arg_kind),
            "arg_identity": str(spec.arg_identity),
            "descriptor_kind": str(spec.descriptor_kind),
            "value_source": str(spec.value_source),
        }
        if str(spec.buffer):
            item["buffer"] = str(spec.buffer)
        if str(spec.value_source) == "constant":
            item["constant_value"] = int(spec.constant_value)
        encoded.append(item)
    return encoded


def make_tt_launch_spec(launch_spec):
    spec = tt_launch_spec_to_dict(launch_spec)
    make = tilelang.tvm.get_global_func("tl.TTKernelLaunchSpec")
    return make(
        spec.get("core_type", ""),
        spec.get("processor", ""),
        spec.get("noc", ""),
    )


def make_tt_compute_config(compute_config):
    spec = tt_compute_config_to_dict(compute_config)
    make_define = tilelang.tvm.get_global_func("tl.TTKernelDefine")
    make_named_arg = tilelang.tvm.get_global_func("tl.TTKernelNamedCompileArg")
    make = tilelang.tvm.get_global_func("tl.TTKernelComputeConfig")
    return make(
        spec.get("math_fidelity", ""),
        bool(spec.get("fp32_dest_acc_en", False)),
        bool(spec.get("dst_full_sync_en", False)),
        bool(spec.get("math_approx_mode", False)),
        [str(item) for item in spec.get("unpack_to_dest_mode", [])],
        bool(spec.get("bfp8_pack_precise", False)),
        [
            make_define(str(item["name"]), str(item["value"]))
            for item in spec.get("defines", [])
        ],
        [
            make_named_arg(str(item["name"]), int(item["value"]))
            for item in spec.get("named_compile_args", [])
        ],
        bool(spec.get("clear_accum", False)),
        int(spec.get("k_pack", 1)),
        int(spec.get("wg_wait", 0)),
        int(spec.get("policy_type", 0)),
        str(spec.get("policy_name", "")),
    )


def make_tt_per_work_arg_specs(per_work_arg_specs):
    make = tilelang.tvm.get_global_func("tl.TTPerWorkArgSpec")
    return [
        make(
            str(item.get("arg_kind", "")),
            str(item.get("arg_identity", "")),
            str(item.get("buffer", "")),
            str(item.get("descriptor_kind", "")),
            str(item.get("value_source", "")),
            int(item.get("constant_value", 0)),
        )
        for item in tt_per_work_arg_specs_to_list(per_work_arg_specs)
    ]


def tt_runtime_arg_specs_to_list(runtime_args):
    encoded = []
    for spec in runtime_args or []:
        plain = _try_plain_dict(spec)
        if plain is not None:
            encoded.append(dict(plain))
            continue
        item = {
            "name": str(spec.name),
            "kind": str(spec.kind),
            "dtype": str(spec.dtype),
        }
        if str(spec.buffer):
            item["buffer"] = str(spec.buffer)
        if str(spec.identity):
            item["identity"] = str(spec.identity)
        if int(spec.core_x) >= 0:
            item["core_x"] = int(spec.core_x)
        if int(spec.core_y) >= 0:
            item["core_y"] = int(spec.core_y)
        encoded.append(item)
    return encoded


def tt_compile_time_arg_specs_to_list(compile_time_arg_specs):
    encoded = []
    for spec in compile_time_arg_specs or []:
        plain = _try_plain_dict(spec)
        if plain is not None:
            encoded.append(dict(plain))
            continue
        item = {
            "name": str(spec.name),
            "kind": str(spec.kind),
            "dtype": str(spec.dtype),
            "offset": int(spec.offset),
            "count": int(spec.count),
        }
        if str(spec.buffer):
            item["buffer"] = str(spec.buffer)
        if str(spec.segment_role):
            item["segment_role"] = str(spec.segment_role)
        if list(spec.values):
            item["values"] = [int(value) for value in spec.values]
        if int(spec.args_config_bits) != 0:
            item["args_config_bits"] = int(spec.args_config_bits)
        if int(spec.transport_page_size) > 0:
            item["transport_page_size"] = int(spec.transport_page_size)
        if str(spec.layout):
            item["layout"] = str(spec.layout)
        if str(spec.memory_space):
            item["memory_space"] = str(spec.memory_space)
        if list(spec.host_axis_order):
            item["host_axis_order"] = [int(axis) for axis in spec.host_axis_order]
        if bool(spec.transpose_2d):
            item["transpose_2d"] = True
        encoded.append(item)
    return encoded


def tt_accessor_specs_to_list(accessors):
    encoded = []
    for spec in accessors or []:
        plain = _try_plain_dict(spec)
        if plain is not None:
            encoded.append(dict(plain))
            continue
        item = {
            "buffer": str(spec.buffer),
            "compile_time_arg_offset": int(spec.compile_time_arg_offset),
            "compile_time_arg_count": int(spec.compile_time_arg_count),
            "common_runtime_arg_offset": int(spec.common_runtime_arg_offset),
            "common_runtime_arg_count": int(spec.common_runtime_arg_count),
            "args_config_bits": int(spec.args_config_bits),
            "layout": str(spec.layout),
            "memory_space": str(spec.memory_space),
        }
        if int(spec.transport_page_size) > 0:
            item["transport_page_size"] = int(spec.transport_page_size)
        if list(spec.host_axis_order):
            item["host_axis_order"] = [int(axis) for axis in spec.host_axis_order]
        if bool(spec.transpose_2d):
            item["transpose_2d"] = True
        encoded.append(item)
    return encoded


def tt_semaphore_binding_specs_to_list(semaphore_bindings):
    encoded = []
    for spec in semaphore_bindings or []:
        plain = _try_plain_dict(spec)
        if plain is not None:
            encoded.append(dict(plain))
            continue
        encoded.append(
            {
                "name": str(spec.name),
                "semaphore_id": int(spec.semaphore_id),
                "arg_kind": str(spec.arg_kind),
            }
        )
    return encoded


def make_tt_runtime_arg_specs(runtime_args):
    make = tilelang.tvm.get_global_func("tl.TTRuntimeArgSpec")
    return [
        make(
            str(item.get("name", "")),
            str(item.get("kind", "")),
            str(item.get("dtype", "")),
            str(item.get("buffer", "")),
            str(item.get("identity", "")),
            int(item.get("core_x", -1)),
            int(item.get("core_y", -1)),
        )
        for item in tt_runtime_arg_specs_to_list(runtime_args)
    ]


def make_tt_compile_time_arg_specs(compile_time_arg_specs):
    make = tilelang.tvm.get_global_func("tl.TTCompileTimeArgSpec")
    return [
        make(
            str(item.get("name", "")),
            str(item.get("kind", "")),
            str(item.get("dtype", "")),
            int(item.get("offset", 0)),
            int(item.get("count", 0)),
            str(item.get("buffer", "")),
            str(item.get("segment_role", "")),
            [int(value) for value in item.get("values", [])],
            int(item.get("args_config_bits", 0)),
            int(item.get("transport_page_size", 0)),
            str(item.get("layout", "")),
            str(item.get("memory_space", "")),
            [int(axis) for axis in item.get("host_axis_order", [])],
            bool(item.get("transpose_2d", False)),
        )
        for item in tt_compile_time_arg_specs_to_list(compile_time_arg_specs)
    ]


def make_tt_accessor_specs(accessors):
    make = tilelang.tvm.get_global_func("tl.TTAccessorSpec")
    return [
        make(
            str(item.get("buffer", "")),
            int(item.get("compile_time_arg_offset", 0)),
            int(item.get("compile_time_arg_count", 0)),
            int(item.get("common_runtime_arg_offset", 0)),
            int(item.get("common_runtime_arg_count", 0)),
            int(item.get("args_config_bits", 0)),
            int(item.get("transport_page_size", 0)),
            str(item.get("layout", "")),
            str(item.get("memory_space", "")),
            [int(axis) for axis in item.get("host_axis_order", [])],
            bool(item.get("transpose_2d", False)),
        )
        for item in tt_accessor_specs_to_list(accessors)
    ]


def make_tt_semaphore_binding_specs(semaphore_bindings):
    make = tilelang.tvm.get_global_func("tl.TTSemaphoreBindingSpec")
    return [
        make(
            str(item.get("name", "")),
            int(item.get("semaphore_id", 0)),
            str(item.get("arg_kind", "")),
        )
        for item in tt_semaphore_binding_specs_to_list(semaphore_bindings)
    ]


def extract_tt_program_segments(func):
    """Extract segment-like kernel/ABI records from TTProgram for regression tests."""
    tt_program = require_tt_program(func)
    abi_plans = list(tt_program.abi_plans)
    segments = []
    for kernel in tt_program.kernels:
        payload = {
            "name": str(kernel.name),
            "kind": str(kernel.kind),
            "core_type": str(kernel.core_type),
        }
        launch_spec = tt_launch_spec_to_dict(getattr(kernel, "launch_spec", None))
        if launch_spec:
            payload["launch_spec"] = launch_spec
        compute_config = tt_compute_config_to_dict(getattr(kernel, "compute_config", None))
        if compute_config:
            payload["compute_config"] = compute_config
        per_work_arg_specs = tt_per_work_arg_specs_to_list(
            getattr(kernel, "per_work_arg_specs", [])
        )
        if per_work_arg_specs:
            payload["per_work_arg_specs"] = per_work_arg_specs
        abi_plan_index = int(kernel.abi_plan_index)
        if 0 <= abi_plan_index < len(abi_plans):
            abi = abi_plans[abi_plan_index]
            payload.setdefault("runtime_args", tt_runtime_arg_specs_to_list(abi.runtime_args))
            payload.setdefault(
                "common_runtime_args",
                tt_runtime_arg_specs_to_list(abi.common_runtime_args),
            )
            payload.setdefault(
                "compile_time_arg_specs",
                tt_compile_time_arg_specs_to_list(abi.compile_time_arg_specs),
            )
            payload.setdefault("accessors", tt_accessor_specs_to_list(abi.accessors))
            payload.setdefault(
                "semaphore_bindings",
                tt_semaphore_binding_specs_to_list(abi.semaphore_bindings),
            )
        compute_ops = [
            encode_tt_compute_op_plan(plan)
            for plan in tt_program.compute_op_plans
            if str(plan.kernel_name) == str(kernel.name)
        ]
        if compute_ops:
            payload["compute_ops"] = compute_ops
        segments.append(payload)
    return segments


def extract_blackhole_segment_plan(func):
    """Return segment-like target truth from TTProgram."""
    return extract_tt_program_segments(func)


def extract_blackhole_runtime_args(func, *, kind=None, core_type=None, common=False):
    """Return TTProgram ABI runtime args."""
    tt_program = require_tt_program(func)
    if kind is not None or core_type is not None:
        kernel = require_tt_kernel(tt_program, kind=kind, core_type=core_type)
        abi = tt_abi_for_kernel(tt_program, kernel)
        return tt_runtime_arg_specs_to_list(
            abi.common_runtime_args if common else abi.runtime_args
        )

    aggregated = []
    seen = set()
    for abi in tt_program.abi_plans:
        args = tt_runtime_arg_specs_to_list(
            abi.common_runtime_args if common else abi.runtime_args
        )
        for arg in args:
            identity = str(arg["identity"]) if "identity" in arg else None
            key = identity or (str(arg["name"]) if "name" in arg else repr(dict(arg)))
            if key in seen:
                continue
            seen.add(key)
            aggregated.append(arg)
    return aggregated


def extract_blackhole_core_plan(func):
    """Return TTProgram core-group truth."""
    tt_program = require_tt_program(func)
    if not tt_program.core_groups:
        pytest.fail("Expected TTProgram to carry a TTCoreGroup")
    core_group = tt_program.core_groups[0]
    return {
        "logical_grid_x": int(core_group.logical_grid_x),
        "logical_grid_y": int(core_group.logical_grid_y),
        "linearization": str(core_group.linearization),
        "physical_cores": list(core_group.physical_cores),
        "work_packets": list(core_group.work_packets),
    }


def extract_blackhole_work_per_core(func):
    """Return the max work-count assigned to any core."""
    core_plan = extract_blackhole_core_plan(func)
    return max((int(packet["work_count"]) for packet in core_plan["work_packets"]), default=0)


def extract_blackhole_cb_configs(func):
    """Return TTProgram CB-plan fields."""
    cb_plans = list(require_tt_program(func).cb_plans)
    configs = []
    for cb_plan in cb_plans:
        config = {
            "cb_id": int(cb_plan.cb_id),
            "name": str(cb_plan.name),
            "role": str(cb_plan.resource_class),
            "num_pages": int(cb_plan.num_pages),
            "page_size": int(cb_plan.page_size_bytes),
            "total_size_bytes": int(cb_plan.num_pages) * int(cb_plan.page_size_bytes),
            "lifetime_begin": int(cb_plan.lifetime_begin),
            "lifetime_end": int(cb_plan.lifetime_end),
            "data_format": str(cb_plan.data_format),
            "requirement_names": [str(name) for name in cb_plan.requirement_names],
            "requirement_indices": [int(index) for index in cb_plan.requirement_indices],
        }
        configs.append(config)
    return configs


def extract_blackhole_total_l1_bytes(func):
    """Return total L1 bytes consumed by CB plans."""
    return sum(int(config["total_size_bytes"]) for config in extract_blackhole_cb_configs(func))


def encode_tt_compute_op_plan(plan):
    operands = {str(binding.role): binding for binding in plan.operand_bindings}
    item = {
        "name": str(plan.name),
        "kernel_name": str(plan.kernel_name),
        "kernel_plan_index": int(plan.kernel_plan_index),
        "enabled": bool(plan.enabled),
        "kind": str(plan.kind),
        "operand_bindings": [
            {
                "role": str(binding.role),
                "buffer": str(binding.buffer),
                "host_buffer": str(binding.host_buffer),
                "tensor_dtype": str(binding.tensor_dtype),
                "cb_dtype": str(binding.cb_dtype),
                "transform_kind": str(binding.transform_kind),
            }
            for binding in plan.operand_bindings
        ],
    }
    axes = [str(axis) for axis in plan.problem_shape_axes]
    shape = [int(dim) for dim in plan.problem_shape]
    if axes and len(axes) == len(shape):
        item.update({axis: value for axis, value in zip(axes, shape)})
    for key, index in (("Mt", 0), ("Nt", 1), ("Kt", 2)):
        if index < len(plan.tile_shape):
            item[key] = int(plan.tile_shape[index])
    for key, index in (("block_m_tiles", 0), ("block_n_tiles", 1), ("block_k_tiles", 2)):
        if index < len(plan.block_shape):
            item[key] = int(plan.block_shape[index])
    for key, index in (("subblock_m_tiles", 0), ("subblock_n_tiles", 1)):
        if index < len(plan.subblock_shape):
            item[key] = int(plan.subblock_shape[index])
    for role, prefix in (("a", "a"), ("b", "b"), ("c", "c")):
        if role not in operands:
            continue
        binding = operands[role]
        item[f"{prefix}_buffer"] = str(binding.host_buffer)
        if str(binding.tensor_dtype):
            item[f"{prefix}_tensor_dtype"] = str(binding.tensor_dtype)
        if str(binding.cb_dtype):
            item[f"{prefix}_cb_dtype"] = str(binding.cb_dtype)
        if role in {"a", "b"}:
            item[f"transpose_{role.upper()}"] = str(binding.transform_kind) == "transpose"
    if str(plan.accumulator_dtype):
        item["accumulator_dtype"] = str(plan.accumulator_dtype)
    item["has_mbarrier"] = bool(str(plan.mbarrier_buffer))
    if str(plan.mbarrier_buffer):
        item["mbarrier_buffer"] = str(plan.mbarrier_buffer)
    if str(plan.mbarrier_scope):
        item["mbarrier_scope"] = str(plan.mbarrier_scope)
    if list(plan.mbarrier_index_exprs):
        item["mbarrier_index_exprs"] = [str(expr) for expr in plan.mbarrier_index_exprs]
    return item


def require_gemm_compute_op(func, *, index=0):
    """Return a typed GEMM TTComputeOpPlan."""
    plans = [plan for plan in require_tt_program(func).compute_op_plans if str(plan.kind) == "gemm"]
    if len(plans) <= index:
        pytest.fail(f"Expected GEMM TTComputeOpPlan at index {index}, found {len(plans)}")
    return plans[index]


def rebuild_tt_kernel(
    kernel,
    *,
    name=None,
    kind=None,
    core_type=None,
    abi_plan_index=None,
    launch_spec=None,
    compute_config=None,
    per_work_arg_specs=None,
):
    """Rebuild a TTKernel with optional field overrides."""
    make_tt_kernel = tilelang.tvm.get_global_func("tl.TTKernel")
    return make_tt_kernel(
        str(kernel.name) if name is None else name,
        str(kernel.kind) if kind is None else kind,
        str(kernel.core_type) if core_type is None else core_type,
        int(kernel.abi_plan_index) if abi_plan_index is None else abi_plan_index,
        make_tt_launch_spec(kernel.launch_spec if launch_spec is None else launch_spec),
        make_tt_compute_config(kernel.compute_config if compute_config is None else compute_config),
        make_tt_per_work_arg_specs(
            kernel.per_work_arg_specs
            if per_work_arg_specs is None
            else per_work_arg_specs
        ),
    )


def rebuild_tt_core_group(
    core_group,
    *,
    name=None,
    logical_grid_x=None,
    logical_grid_y=None,
    linearization=None,
    physical_cores=None,
    work_packets=None,
):
    """Rebuild a TTCoreGroup with optional field overrides."""
    make_tt_core_group = tilelang.tvm.get_global_func("tl.TTCoreGroup")
    return make_tt_core_group(
        str(core_group.name) if name is None else name,
        int(core_group.logical_grid_x) if logical_grid_x is None else logical_grid_x,
        int(core_group.logical_grid_y) if logical_grid_y is None else logical_grid_y,
        str(core_group.linearization) if linearization is None else linearization,
        list(core_group.physical_cores) if physical_cores is None else physical_cores,
        list(core_group.work_packets) if work_packets is None else work_packets,
    )


def rebuild_tt_kernel_plan(
    kernel_plan,
    *,
    name=None,
    kind=None,
    core_type=None,
    block_plan_index=None,
    abi_plan_index=None,
):
    """Rebuild a TTKernelPlan with optional field overrides."""
    make_tt_kernel_plan = tilelang.tvm.get_global_func("tl.TTKernelPlan")
    return make_tt_kernel_plan(
        str(kernel_plan.name) if name is None else name,
        str(kernel_plan.kind) if kind is None else kind,
        str(kernel_plan.core_type) if core_type is None else core_type,
        int(kernel_plan.block_plan_index) if block_plan_index is None else block_plan_index,
        int(kernel_plan.abi_plan_index) if abi_plan_index is None else abi_plan_index,
    )


def rebuild_tt_compute_operand_binding_plan(
    binding,
    *,
    role=None,
    buffer=None,
    host_buffer=None,
    tensor_dtype=None,
    cb_dtype=None,
    transform_kind=None,
):
    """Rebuild a TTComputeOperandBindingPlan with optional field overrides."""
    make_binding = tilelang.tvm.get_global_func("tl.TTComputeOperandBindingPlan")
    return make_binding(
        str(binding.role) if role is None else role,
        str(binding.buffer) if buffer is None else buffer,
        str(binding.host_buffer) if host_buffer is None else host_buffer,
        str(binding.tensor_dtype) if tensor_dtype is None else tensor_dtype,
        str(binding.cb_dtype) if cb_dtype is None else cb_dtype,
        str(binding.transform_kind) if transform_kind is None else transform_kind,
    )


def rebuild_tt_compute_op_plan(
    compute_op,
    *,
    name=None,
    kernel_name=None,
    kernel_plan_index=None,
    kind=None,
    enabled=None,
    operand_bindings=None,
    problem_shape_axes=None,
    problem_shape=None,
    tile_shape=None,
    block_shape=None,
    subblock_shape=None,
    accumulator_dtype=None,
    mbarrier_buffer=None,
    mbarrier_scope=None,
    mbarrier_index_exprs=None,
):
    """Rebuild a TTComputeOpPlan with optional field overrides."""
    make_compute_op = tilelang.tvm.get_global_func("tl.TTComputeOpPlan")
    return make_compute_op(
        str(compute_op.name) if name is None else name,
        str(compute_op.kernel_name) if kernel_name is None else kernel_name,
        int(compute_op.kernel_plan_index)
        if kernel_plan_index is None
        else kernel_plan_index,
        str(compute_op.kind) if kind is None else kind,
        bool(compute_op.enabled) if enabled is None else enabled,
        list(compute_op.operand_bindings)
        if operand_bindings is None
        else operand_bindings,
        list(compute_op.problem_shape_axes)
        if problem_shape_axes is None
        else problem_shape_axes,
        list(compute_op.problem_shape) if problem_shape is None else problem_shape,
        list(compute_op.tile_shape) if tile_shape is None else tile_shape,
        list(compute_op.block_shape) if block_shape is None else block_shape,
        list(compute_op.subblock_shape)
        if subblock_shape is None
        else subblock_shape,
        str(compute_op.accumulator_dtype)
        if accumulator_dtype is None
        else accumulator_dtype,
        str(compute_op.mbarrier_buffer)
        if mbarrier_buffer is None
        else mbarrier_buffer,
        str(compute_op.mbarrier_scope) if mbarrier_scope is None else mbarrier_scope,
        list(compute_op.mbarrier_index_exprs)
        if mbarrier_index_exprs is None
        else mbarrier_index_exprs,
    )


def rebuild_tt_semaphore_plan(
    semaphore_plan,
    *,
    name=None,
    kind=None,
    semaphore_id=None,
    initial_value=None,
    core_type=None,
    source_task_index=None,
    target_task_index=None,
    core_ranges=None,
):
    """Rebuild a TTSemaphorePlan with optional field overrides."""
    make_tt_semaphore_plan = tilelang.tvm.get_global_func("tl.TTSemaphorePlan")
    return make_tt_semaphore_plan(
        str(semaphore_plan.name) if name is None else name,
        str(semaphore_plan.kind) if kind is None else kind,
        int(semaphore_plan.semaphore_id) if semaphore_id is None else semaphore_id,
        int(semaphore_plan.initial_value) if initial_value is None else initial_value,
        str(semaphore_plan.core_type) if core_type is None else core_type,
        int(semaphore_plan.source_task_index) if source_task_index is None else source_task_index,
        int(semaphore_plan.target_task_index) if target_task_index is None else target_task_index,
        list(semaphore_plan.core_ranges) if core_ranges is None else core_ranges,
    )


def rebuild_tt_abi_plan(
    abi_plan,
    *,
    name=None,
    kernel_name=None,
    runtime_args=None,
    common_runtime_args=None,
    compile_time_arg_specs=None,
    accessors=None,
    semaphore_bindings=None,
):
    """Rebuild a TTABIPlan with optional field overrides."""
    make_tt_abi_plan = tilelang.tvm.get_global_func("tl.TTABIPlan")
    return make_tt_abi_plan(
        str(abi_plan.name) if name is None else name,
        str(abi_plan.kernel_name) if kernel_name is None else kernel_name,
        make_tt_runtime_arg_specs(
            abi_plan.runtime_args if runtime_args is None else runtime_args
        ),
        make_tt_runtime_arg_specs(
            abi_plan.common_runtime_args
            if common_runtime_args is None
            else common_runtime_args
        ),
        make_tt_compile_time_arg_specs(
            abi_plan.compile_time_arg_specs
            if compile_time_arg_specs is None
            else compile_time_arg_specs
        ),
        make_tt_accessor_specs(abi_plan.accessors if accessors is None else accessors),
        make_tt_semaphore_binding_specs(
            abi_plan.semaphore_bindings
            if semaphore_bindings is None
            else semaphore_bindings
        ),
    )


def rebuild_tt_program(
    program,
    *,
    entry_name=None,
    member_func=None,
    mesh_plans=None,
    buffer_distribution_plans=None,
    compute_op_plans=None,
    block_plans=None,
    kernel_plans=None,
    sync_plans=None,
    kernels=None,
    core_groups=None,
    cb_plans=None,
    transport_plans=None,
    semaphore_plans=None,
    compute_sync_plans=None,
    dst_layout_plans=None,
    live_form_plans=None,
    materialization_plans=None,
    consumer_binding_plans=None,
    abi_plans=None,
    execution_plans=None,
):
    """Rebuild a TTProgram with optional field overrides."""
    make_tt_program = tilelang.tvm.get_global_func("tl.TTProgram")
    if kernels is not None and kernel_plans is None and len(program.kernel_plans) == len(kernels):
        kernel_plans = [
            rebuild_tt_kernel_plan(
                program.kernel_plans[index],
                name=str(kernel.name),
                kind=str(kernel.kind),
                core_type=str(kernel.core_type),
                abi_plan_index=int(kernel.abi_plan_index),
            )
            for index, kernel in enumerate(kernels)
        ]
    return make_tt_program(
        str(program.entry_name) if entry_name is None else entry_name,
        str(program.member_func) if member_func is None else member_func,
        list(program.mesh_plans) if mesh_plans is None else mesh_plans,
        list(program.buffer_distribution_plans)
        if buffer_distribution_plans is None
        else buffer_distribution_plans,
        list(program.block_plans) if block_plans is None else block_plans,
        list(program.kernel_plans) if kernel_plans is None else kernel_plans,
        list(program.compute_op_plans) if compute_op_plans is None else compute_op_plans,
        list(program.transport_plans) if transport_plans is None else transport_plans,
        list(program.sync_plans) if sync_plans is None else sync_plans,
        list(program.abi_plans) if abi_plans is None else abi_plans,
        list(program.execution_plans) if execution_plans is None else execution_plans,
        list(program.kernels) if kernels is None else kernels,
        list(program.core_groups) if core_groups is None else core_groups,
        list(program.cb_plans) if cb_plans is None else cb_plans,
        list(program.semaphore_plans) if semaphore_plans is None else semaphore_plans,
        list(program.compute_sync_plans) if compute_sync_plans is None else compute_sync_plans,
        list(program.dst_layout_plans) if dst_layout_plans is None else dst_layout_plans,
        list(program.live_form_plans) if live_form_plans is None else live_form_plans,
        list(program.materialization_plans)
        if materialization_plans is None
        else materialization_plans,
        list(program.consumer_binding_plans)
        if consumer_binding_plans is None
        else consumer_binding_plans,
    )


def require_tt_kernel(tt_program, *, kind, core_type=None, name=None):
    """Require a unique TTProgram kernel matching the requested target role."""
    matches = []
    for kernel in tt_program.kernels:
        if str(kernel.kind) != kind:
            continue
        if core_type is not None and str(kernel.core_type) != core_type:
            continue
        if name is not None and str(kernel.name) != name:
            continue
        matches.append(kernel)

    if not matches:
        available = [
            f"{str(kernel.name)}:{str(kernel.kind)}:{str(kernel.core_type)}"
            for kernel in tt_program.kernels
        ]
        pytest.fail(
            f"Missing TTProgram kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"available kernels: {available}"
        )
    if len(matches) > 1:
        matched = [f"{str(kernel.name)}:{str(kernel.kind)}:{str(kernel.core_type)}" for kernel in matches]
        pytest.fail(
            f"Ambiguous TTProgram kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"matched kernels: {matched}"
        )
    return matches[0]


def tt_abi_for_kernel(tt_program, kernel):
    """Return the ABI plan referenced by a TTProgram kernel."""
    abi_plan_index = int(kernel.abi_plan_index)
    assert abi_plan_index >= 0
    return tt_program.abi_plans[abi_plan_index]


def staged_copy_kernel(
    tile_rows: int,
    tile_cols: int = 1,
    tile_m: int = 32,
    tile_n: int = 32,
    dtype: str = "bfloat16",
):
    """Define an explicit TileLang T.copy(global->shared->global) kernel."""
    m = tile_rows * tile_m
    n = tile_cols * tile_n

    if dtype == "bfloat16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "bfloat16"),
            B: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "bfloat16")
                for tile_idx in T.serial(tile_rows * tile_cols):
                    tile_row = tile_idx // tile_cols
                    tile_col = tile_idx % tile_cols
                    T.copy(A[tile_row * tile_m, tile_col * tile_n], A_shared)
                    T.copy(A_shared, B[tile_row * tile_m, tile_col * tile_n])

        return main

    if dtype == "float16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float16"),
            B: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "float16")
                for tile_idx in T.serial(tile_rows * tile_cols):
                    tile_row = tile_idx // tile_cols
                    tile_col = tile_idx % tile_cols
                    T.copy(A[tile_row * tile_m, tile_col * tile_n], A_shared)
                    T.copy(A_shared, B[tile_row * tile_m, tile_col * tile_n])

        return main

    raise ValueError(f"Unsupported staged_copy_kernel dtype: {dtype}")


def grid_indexed_staged_copy_kernel(
    grid_x: int,
    grid_y: int,
    tile_m: int = 32,
    tile_n: int = 32,
    dtype: str = "bfloat16",
):
    """Define a copy kernel whose indices depend on bx/by logical block coordinates."""
    m = grid_y * tile_m
    n = grid_x * tile_n

    if dtype == "bfloat16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "bfloat16"),
            B: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(grid_x, grid_y) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "bfloat16")
                T.copy(A[by * tile_m, bx * tile_n], A_shared)
                T.copy(A_shared, B[by * tile_m, bx * tile_n])

        return main

    if dtype == "float16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float16"),
            B: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(grid_x, grid_y) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "float16")
                T.copy(A[by * tile_m, bx * tile_n], A_shared)
                T.copy(A_shared, B[by * tile_m, bx * tile_n])

        return main

    raise ValueError(f"Unsupported grid_indexed_staged_copy_kernel dtype: {dtype}")


def staged_stick_copy_kernel(
    tile_m: int = 32,
    tile_n: int = 16,
    global_n: int = 32,
    dtype: str = "float32",
    src_col: int = 0,
    dst_col: int = 0,
):
    """Define a minimal interleaved stick-style copy with non-32-aligned width."""
    assert tile_m % 32 == 0
    assert 0 <= src_col <= global_n - tile_n
    assert 0 <= dst_col <= global_n - tile_n

    @T.prim_func
    def main(
        A: T.Tensor((tile_m, global_n), dtype),
        B: T.Tensor((tile_m, global_n), dtype),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((tile_m, tile_n), dtype)
            T.copy(A[0:tile_m, src_col : src_col + tile_n], A_shared)
            T.copy(A_shared, B[0:tile_m, dst_col : dst_col + tile_n])

    return main


def gemm_kernel(M: int = 32, N: int = 32, K: int = 128):
    """GEMM kernel with explicit data-movement: T.copy A/B in, gemm, T.copy C out."""
    block_M, block_N, block_K = M, N, K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            B_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.copy(A[0:block_M, 0:block_K], A_shared)
            T.copy(B[0:block_N, 0:block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0:block_M, 0:block_N])

    return main


def gemm_kernel_with_transpose_flags(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    transpose_A: bool = False,
    transpose_B: bool = True,
):
    """GEMM kernel with configurable transpose flags."""

    a_shape = (K, M) if transpose_A else (M, K)
    b_shape = (N, K) if transpose_B else (K, N)
    @T.prim_func
    def main(
        A: T.Tensor(a_shape, "bfloat16"),
        B: T.Tensor(b_shape, "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared(a_shape, "bfloat16")
            B_shared = T.alloc_shared(b_shape, "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0 : a_shape[0], 0 : a_shape[1]], A_shared)
            T.copy(B[0 : b_shape[0], 0 : b_shape[1]], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_compute_abi(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    clear_accum: bool = True,
    k_pack: int = 2,
    wg_wait: int = 3,
    preclear_output_fragment: bool = False,
):
    """GEMM kernel with non-default compute ABI knobs."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            if preclear_output_fragment:
                T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=clear_accum,
                k_pack=k_pack,
                wg_wait=wg_wait,
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_post_merge_cast_consumer(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    preclear_output_fragment: bool = True,
):
    """GEMM kernel whose accumulated fragment stays live for a later fragment-cast consumer."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        D: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            D_local = T.alloc_fragment((M, N), "bfloat16")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            if preclear_output_fragment:
                T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=clear_accum,
                k_pack=k_pack,
                wg_wait=wg_wait,
            )
            for i, j in T.Parallel(M, N):
                D_local[i, j] = T.cast(C_local[i, j], "bfloat16")
            T.copy(D_local, D[0:M, 0:N])

    return main


def fragment_fill_cast_publish_kernel(
    M: int = 32,
    N: int = 32,
    *,
    fill_value: float = 3.5,
):
    """Kernel that isolates local-fragment -> tiled-CB cast publication."""

    @T.prim_func
    def main(D: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(1, 1) as (bx, by):
            C_local = T.alloc_fragment((M, N), "float32")
            D_local = T.alloc_fragment((M, N), "bfloat16")
            for i, j in T.Parallel(M, N):
                C_local[i, j] = T.float32(fill_value)
            for i, j in T.Parallel(M, N):
                D_local[i, j] = T.cast(C_local[i, j], "bfloat16")
            T.copy(D_local, D[0:M, 0:N])

    return main


def gemm_kernel_with_policy(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    policy=T.GemmWarpPolicy.FullRow,
):
    """GEMM kernel with non-default warp policy."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_mbar(
    M: int = 32,
    N: int = 32,
    K: int = 128,
):
    """GEMM kernel with an explicit barrier binding passed as mbar."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            mbar = T.alloc_barrier(128)
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True, mbar=mbar)
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_compute_config_extras(
    M: int = 32,
    N: int = 32,
    K: int = 128,
):
    """GEMM kernel with richer TT-Metal compute config extras on the DSL surface."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                dst_full_sync_en=True,
                bfp8_pack_precise=True,
                defines={
                    "BLACKHOLE_TEST_DEFINE": "1",
                    "BLACKHOLE_ACC_MODE": "fp32",
                },
                named_compile_args={
                    "c_0": 0,
                    "c_1": 1,
                    "c_16": 16,
                },
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def assert_tensors_close_or_dump(actual, expected, atol, rtol, failure_message):
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    pytest.fail(
        f"{failure_message}; max diff={diff.max().item()}, mean diff={diff.mean().item()}"
    )
