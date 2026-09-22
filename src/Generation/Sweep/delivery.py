from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from ...Core.CrossSection.annotations import FeatureAnnotation
from ...Core.CrossSection.cross_section import OrientedSection, SectionOffset
from ...Core.Path.path import PathPoint
from ...Core.Path.path_frame import PathFrame
from ...Core.World.block_world import ColumnMask
from ...Core.coordinates import BlockPos


class Delivery(NamedTuple):
    # The positions one oriented annotation delivers at one path point, in the
    # order they belong on the feature's path. x and z are world columns; y is
    # the local height above the path, since elevation is applied when the
    # feature is placed.
    annotation: FeatureAnnotation
    positions: tuple[BlockPos, ...]
    # The one of the eight layouts these positions were placed in, and
    # whether that layout was the mirrored one.
    layout: int = 0
    mirrored: bool = False


def deliver_offset(point: PathPoint, offset: SectionOffset) -> Delivery:
    # Translate one oriented annotation to one path point.
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
    # Every annotation of an oriented section at one path point. With a mask,
    # positions in columns outside it are dropped, keeping the order of the
    # rest; an annotation left with no positions still delivers, empty.
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
    # The columns of any of several frames: the mask for a point delivered
    # into every frame its lead reaches.
    frames: tuple[PathFrame, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))

    def contains(self, x: float, z: float) -> bool:
        return any(frame.contains(x, z) for frame in self.frames)


def cover(frames: Sequence[PathFrame]) -> ColumnMask:
    # A single frame is already a mask; several need a cover.
    return frames[0] if len(frames) == 1 else FrameCover(tuple(frames))
