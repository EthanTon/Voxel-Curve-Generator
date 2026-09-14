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


def _offset(offset: BlockPos | None) -> BlockPos:
    return BlockPos(0, 0, 0) if offset is None else BlockPos.coerce(offset)


@dataclass(eq=False, slots=True)
class PointFeature:
    blocks: BlockField
    placement_mode: PlacementMode
    domain: Domain
    offset: BlockPos | None = None
    elevation_mode: ElevationMode = ElevationMode.DEFORMABLE

    def __post_init__(self) -> None:
        self.offset = _offset(self.offset)


@dataclass(eq=False, slots=True)
class ProfileFeature:
    # The old physical cross section: `blocks` is the profile that is swept
    # along this feature's path. Building the sweep is handled later.

    blocks: BlockField
    placement_mode: PlacementMode
    domain: Domain
    offset: BlockPos | None = None
    elevation_mode: ElevationMode = ElevationMode.DEFORMABLE

    def __post_init__(self) -> None:
        self.offset = _offset(self.offset)


@dataclass(eq=False, slots=True)
class ContinuousFeature:
    # Features like wires and tracks.

    blocks: BlockField
    placement_mode: PlacementMode
    domain: Domain
    offset: BlockPos | None = None
    elevation_mode: ElevationMode = ElevationMode.DEFORMABLE

    def __post_init__(self) -> None:
        self.offset = _offset(self.offset)

    def add(self, block_position: BlockPos, block_state: BlockState) -> None:
        self.blocks.set_block(position=block_position, block=block_state)
