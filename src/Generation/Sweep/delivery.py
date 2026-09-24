from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from ...Core.CrossSection.annotations import FeatureAnnotation
from ...Core.CrossSection.cross_section import OrientedSection, SectionOffset
from ...Core.Path.path import PathPoint
from ...Core.Path.path_frame import PathFrame, frames_for_index
from ...Core.World.block_world import ColumnMask
from ...Core.coordinates import BlockPos


class Delivery(NamedTuple):
    # One annotation's positions at one path point: world x and z, y as height
    # above the path, with the layout that placed them.
    annotation: FeatureAnnotation
    positions: tuple[BlockPos, ...]
    layout: int = 0
    mirrored: bool = False


def deliver_offset(point: PathPoint, offset: SectionOffset) -> Delivery:
    x, z = point.position.x, point.position.z
    return Delivery(
        offset.annotation,
        tuple(BlockPos(x + o.x, o.y, z + o.z) for o in offset.offsets),
        offset.layout,
        offset.mirrored,
    )


def deliver(
    point: PathPoint,
    section: OrientedSection,
    mask: ColumnMask | None = None,
) -> tuple[Delivery, ...]:
    # Every annotation at one path point, dropping positions outside `mask`.
    deliveries = (deliver_offset(point, offset) for offset in section)
    if mask is None:
        return tuple(deliveries)
    return tuple(
        Delivery(
            delivery.annotation,
            tuple(p for p in delivery.positions if mask.contains(p.x, p.z)),
            delivery.layout,
            delivery.mirrored,
        )
        for delivery in deliveries
    )


@dataclass(frozen=True, slots=True)
class FrameCover:
    # The columns of any of several frames.
    frames: tuple[PathFrame, ...]
    _columns: dict[tuple[float, float], bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))

    def contains(self, x: float, z: float) -> bool:
        column = (x, z)
        cached = self._columns.get(column)
        if cached is None:
            cached = any(frame.contains(x, z) for frame in self.frames)
            self._columns[column] = cached
        return cached


def cover(frames: Sequence[PathFrame]) -> ColumnMask:
    return frames[0] if len(frames) == 1 else FrameCover(tuple(frames))


@dataclass(frozen=True, slots=True)
class _Nowhere:
    def contains(self, x: float, z: float) -> bool:
        return False


NOWHERE = _Nowhere()


def nearby_frames(frames: Sequence[PathFrame], reach: int) -> Callable[[object], ColumnMask]:
    # A fitting may land only in frames within `reach` points of its own, so
    # a frame from the far side of a loop cannot take it.
    covers: dict[tuple[int, ...], ColumnMask] = {}

    def clip(point) -> ColumnMask:
        reached = frames_for_index(frames, point.path_index, reach, reach)
        key = tuple(frame.index for frame in reached)
        mask = covers.get(key)
        if mask is None:
            mask = covers[key] = cover(reached) if reached else NOWHERE
        return mask

    return clip
