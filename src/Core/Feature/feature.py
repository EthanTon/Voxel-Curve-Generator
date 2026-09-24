from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..Field.block_field import BlockField
from ..modes import ElevationMode, PlacementMode
from ..coordinates import BlockPos
from ..block.states import BlockState
from ..Domain import Domain
from ...Terraform.surface_mask import SurfaceMaskSpec


@runtime_checkable
class Feature(Protocol):
    # DEFORMABLE blocks follow their own column's elevation; RIGID blocks
    # take the y_offset of the point they were placed at.

    blocks: BlockField
    placement_mode: PlacementMode
    domain: Domain
    offset: BlockPos
    elevation_mode: ElevationMode


# An offset of None becomes the zero offset.
_FEATURE_DEFAULTS: dict[str, object] = {
    "offset": None,
    "elevation_mode": ElevationMode.DEFORMABLE,
}


def _normalize_offset(self) -> None:
    offset = self.offset
    self.offset = BlockPos(0, 0, 0) if offset is None else BlockPos.coerce(offset)


def feature_class(cls: type) -> type:
    # Makes `cls` a dataclass with the Feature fields ahead of its own, which
    # therefore need defaults. Anything the class declares itself is kept.
    own = dict(cls.__dict__.get("__annotations__", {}))
    annotations = {}
    for name, annotation in Feature.__annotations__.items():
        if name in own:
            continue
        if name in _FEATURE_DEFAULTS and name not in cls.__dict__:
            default = _FEATURE_DEFAULTS[name]
            setattr(cls, name, default)
            if default is None:
                annotation = f"{annotation} | None"
        annotations[name] = annotation
    cls.__annotations__ = {**annotations, **own}
    if "__post_init__" not in cls.__dict__:
        cls.__post_init__ = _normalize_offset
    return dataclass(eq=False, slots=True)(cls)


@feature_class
class PointFeature:
    # Blocks stamped at each point of its path.
    surface_mask: SurfaceMaskSpec | None = None


@feature_class
class ProfileFeature:
    # A cross-section profile swept along its path.
    surface_mask: SurfaceMaskSpec | None = None


@feature_class
class ContinuousFeature:
    # One run of blocks, such as a wire.
    surface_mask: SurfaceMaskSpec | None = None

    def add(self, block_position: BlockPos, block_state: BlockState) -> None:
        self.blocks.set_block(position=block_position, block=block_state)
