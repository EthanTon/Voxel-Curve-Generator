from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..Field.block_field import BlockField
from ..modes import ElevationMode, PlacementMode
from ..coordinates import BlockPos
from ..block.states import BlockState
from ..Domain import Domain


@runtime_checkable
class Feature(Protocol):
    # What every feature provides. Implementations match this by shape rather
    # than by subclassing, so each one is free to implement its members its
    # own way.
    #
    # `elevation_mode`: DEFORMABLE blocks take the elevation of their own
    # column. RIGID blocks ignore it and take the y_offset of the feature path
    # point they were built from.

    blocks: BlockField
    placement_mode: PlacementMode
    domain: Domain
    offset: BlockPos
    elevation_mode: ElevationMode


# Defaults for the Feature fields that have one. `offset` accepts None, which
# becomes the zero offset.
_FEATURE_DEFAULTS: dict[str, object] = {
    "offset": None,
    "elevation_mode": ElevationMode.DEFORMABLE,
}


def _normalize_offset(self) -> None:
    offset = self.offset
    self.offset = BlockPos(0, 0, 0) if offset is None else BlockPos.coerce(offset)


def feature_class(cls: type) -> type:
    # Make `cls` a dataclass with every Feature field, taken from the Feature
    # interface and placed ahead of its own fields. A class keeps any field,
    # default, or __post_init__ it declares itself. Its own fields follow the
    # defaulted Feature fields, so they need defaults too.
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
    # A feature whose blocks sit at each point on its path.
    pass


@feature_class
class ProfileFeature:
    # The old physical cross section: `blocks` is the profile that is swept
    # along this feature's path. Building the sweep is handled later.
    pass


@feature_class
class ContinuousFeature:
    # Features like wires and tracks.

    def add(self, block_position: BlockPos, block_state: BlockState) -> None:
        self.blocks.set_block(position=block_position, block=block_state)
