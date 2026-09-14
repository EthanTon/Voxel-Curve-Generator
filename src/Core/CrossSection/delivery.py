from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ..coordinates import BlockPos
from .annotations import FeatureAnnotation
from .cross_section import OrientedSection, SectionOffset

if TYPE_CHECKING:
    from ..Path.path import PathPoint


class Delivery(NamedTuple):
    # The positions one oriented annotation delivers at one path point, in the
    # order they belong on the feature's path. x and z are world columns; y is
    # the local height above the path, since elevation is applied when the
    # feature is placed.
    annotation: FeatureAnnotation
    positions: tuple[BlockPos, ...]


def deliver_offset(point: PathPoint, offset: SectionOffset) -> Delivery:
    # Translate one oriented annotation to one path point.
    x, z = point.position.x, point.position.z
    return Delivery(
        offset.annotation,
        tuple(BlockPos(x + o.x, o.y, z + o.z) for o in offset.offsets),
    )


def deliver(point: PathPoint, section: OrientedSection) -> tuple[Delivery, ...]:
    return tuple(deliver_offset(point, offset) for offset in section)
