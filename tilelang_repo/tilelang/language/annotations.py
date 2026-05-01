"""Annotation helpers exposed on the TileLang language surface."""

from typing import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.script.parser.tir import attr, block_attr
from tvm.tir import FloatImm

__all__ = [
    "CoreGrid",
    "ShardSpec",
    "NDShardSpec",
    "MemoryConfig",
    "interleaved_dram",
    "sharded_dram",
    "sharded_l1",
    "use_swizzle",
    "annotate_layout",
    "annotate_memory_config",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
]

_SHARD_STRATEGIES = {
    "height": "height_sharded",
    "width": "width_sharded",
    "block": "block_sharded",
}
_MEMORY_LAYOUTS = {"interleaved", *set(_SHARD_STRATEGIES.values()), "nd_sharded"}
_BUFFER_TYPES = {"dram", "l1"}
_ORIENTATIONS = {"row_major", "col_major"}


class CoreGrid:
    """Target-independent user grid hint for tensor memory placement."""

    def __init__(self, *, x: int, y: int):
        if int(x) <= 0 or int(y) <= 0:
            raise ValueError("CoreGrid x and y must be positive")
        self.x = int(x)
        self.y = int(y)

    def to_tir_attr(self):
        return {"shape": [self.y, self.x]}


class ShardSpec:
    """User-facing shard spec mirroring TTNN's 2D ShardSpec shape."""

    def __init__(self, *, grid: CoreGrid, shape, orientation: str = "row_major"):
        if not isinstance(grid, CoreGrid):
            raise TypeError("ShardSpec grid must be a CoreGrid")
        if orientation not in _ORIENTATIONS:
            raise ValueError(f"Invalid shard orientation: {orientation}")
        shard_shape = [int(dim) for dim in shape]
        if not shard_shape or any(dim <= 0 for dim in shard_shape):
            raise ValueError("ShardSpec shape must contain positive dimensions")
        self.grid = grid
        self.shape = shard_shape
        self.orientation = orientation

    def to_tir_attr(self):
        return {
            "kind": "shard_spec",
            "grid_shape": self.grid.to_tir_attr()["shape"],
            "shape": self.shape,
            "orientation": self.orientation,
        }


class NDShardSpec:
    """Placeholder-compatible N-D shard spec surface for later T3 slices."""

    def __init__(
        self,
        *,
        grid: CoreGrid,
        shard_shape,
        orientation: str = "row_major",
        distribution_strategy: str = "block",
    ):
        if distribution_strategy not in _SHARD_STRATEGIES:
            raise ValueError(f"Invalid ND shard distribution strategy: {distribution_strategy}")
        self.shard_spec = ShardSpec(grid=grid, shape=shard_shape, orientation=orientation)
        self.distribution_strategy = distribution_strategy

    def to_tir_attr(self):
        attr = self.shard_spec.to_tir_attr()
        attr["kind"] = "nd_shard_spec"
        attr["distribution_strategy"] = self.distribution_strategy
        return attr


class MemoryConfig:
    """User-facing tensor memory placement request."""

    def __init__(
        self,
        *,
        memory_layout: str,
        buffer_type: str,
        shard=None,
        allow_reshard: bool = True,
    ):
        if memory_layout not in _MEMORY_LAYOUTS:
            raise ValueError(f"Invalid memory_layout: {memory_layout}")
        if buffer_type not in _BUFFER_TYPES:
            raise ValueError(f"Invalid buffer_type: {buffer_type}")
        if memory_layout != "interleaved" and shard is None:
            raise ValueError("Sharded MemoryConfig requires a shard spec")
        if shard is not None and not hasattr(shard, "to_tir_attr"):
            raise TypeError("MemoryConfig shard must be a ShardSpec or NDShardSpec")
        self.memory_layout = memory_layout
        self.buffer_type = buffer_type
        self.shard = shard
        self.allow_reshard = bool(allow_reshard)

    def to_tir_attr(self):
        attr = {
            "kind": "memory_config",
            "memory_layout": self.memory_layout,
            "buffer_type": self.buffer_type,
            "allow_reshard": self.allow_reshard,
        }
        if self.shard is not None:
            attr["shard"] = self.shard.to_tir_attr()
        return attr


def interleaved_dram(*, allow_reshard: bool = True):
    return MemoryConfig(
        memory_layout="interleaved",
        buffer_type="dram",
        allow_reshard=allow_reshard,
    )


def _sharded_config(
    *,
    strategy: str,
    buffer_type: str,
    grid: CoreGrid,
    shard_shape,
    orientation: str = "row_major",
    allow_reshard: bool = True,
):
    if strategy not in _SHARD_STRATEGIES:
        raise ValueError(f"Invalid sharding strategy: {strategy}")
    return MemoryConfig(
        memory_layout=_SHARD_STRATEGIES[strategy],
        buffer_type=buffer_type,
        shard=ShardSpec(grid=grid, shape=shard_shape, orientation=orientation),
        allow_reshard=allow_reshard,
    )


def sharded_dram(
    *,
    strategy: str,
    grid: CoreGrid,
    shard_shape,
    orientation: str = "row_major",
    allow_reshard: bool = True,
):
    return _sharded_config(
        strategy=strategy,
        buffer_type="dram",
        grid=grid,
        shard_shape=shard_shape,
        orientation=orientation,
        allow_reshard=allow_reshard,
    )


def sharded_l1(
    *,
    strategy: str,
    grid: CoreGrid,
    shard_shape,
    orientation: str = "row_major",
    allow_reshard: bool = True,
):
    return _sharded_config(
        strategy=strategy,
        buffer_type="l1",
        grid=grid,
        shard_shape=shard_shape,
        orientation=orientation,
        allow_reshard=allow_reshard,
    )


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


def annotate_memory_config(memory_config_map: dict):
    """Annotate tensor memory placement requests for the current block."""
    _memory_config_map = {}
    for buffer, memory_config in memory_config_map.items():
        if not isinstance(memory_config, MemoryConfig):
            raise TypeError(
                "annotate_memory_config values must be MemoryConfig objects"
            )
        try:
            data_var = buffer.data
        except Exception as e:
            raise TypeError(
                f"annotate_memory_config expects Buffer keys, got {type(buffer)}"
            ) from e
        _memory_config_map[data_var] = memory_config.to_tir_attr()
    return block_attr({"tl.memory_config_map": _memory_config_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return block_attr({"tl.non_restrict_params": data_vars})
