from __future__ import annotations

from collections.abc import Iterable

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.turns import Turn, heading
from ...Core.Feature.feature import Feature
from ...Core.Path.path import Path
from ..Sweep.schedule import lead_for
from .boundaries import LateralReach


def lateral_reach(
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> LateralReach:
    # How far the sections extend either side of the path, at `turn` or at the
    # worst of every turn. Each delivered position is projected onto the right
    # of travel, then widened by its feature's blocks.
    #
    # Oriented positions are used rather than the authored ones because a
    # diagonal layout shifts its two quarter layouts, which can move positions
    # further out than they were authored.
    turns = range(0, 360, 45) if turn is None else (turn,)
    left = right = 0.0
    for section in sections:
        for each in turns:
            forward = heading(each)
            right_x, right_z = -forward.z, forward.x
            for offset in section.orient(each):
                low, high = feature_lateral_span(offset.annotation.feature)
                for position in offset.offsets:
                    lateral = position.x * right_x + position.z * right_z
                    right = max(right, lateral + high)
                    left = max(left, -(lateral + low))
    return LateralReach(float(left), float(right))


def feature_lateral_span(feature: Feature) -> tuple[int, int]:
    # The local x range, lowest first, a feature's blocks cover around the
    # position it is built at, including its offset.
    #
    # Blocks are measured along local x, the same lateral axis annotations are
    # authored in. How blocks turn at a diagonal is decided when features are
    # placed, so they are not rotated here.
    shift = feature.offset.x
    bounds = feature.blocks.bounds
    if bounds is None:
        return shift, shift
    return bounds.minimum.x + shift, bounds.maximum.x + shift


def longitudinal_reach(
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> float:
    # How far the sections extend along travel from the point they sit at,
    # ahead or behind, at `turn` or at the worst of every turn. Each delivered
    # position is projected onto the heading, then widened by its feature's
    # blocks.
    #
    # CrossSection.reach measures the positions alone. This widens them the way
    # lateral_reach does, since a feature paints blocks around the position it
    # is built at, and those reach further along the path than the position.
    #
    # The widening is taken as a magnitude either way round rather than signed:
    # blocks are measured in local axes, where forward is -z, and how they turn
    # at a diagonal is decided when features are placed, not here.
    turns = range(0, 360, 45) if turn is None else (turn,)
    furthest = 0.0
    for section in sections:
        for each in turns:
            direction = heading(each)
            for offset in section.orient(each):
                low, high = feature_longitudinal_span(offset.annotation.feature)
                for position in offset.offsets:
                    along = position.x * direction.x + position.z * direction.z
                    furthest = max(furthest, abs(along + high), abs(along + low))
    return float(furthest)


def feature_longitudinal_span(feature: Feature) -> tuple[int, int]:
    # The local z range, lowest first, a feature's blocks cover around the
    # position it is built at, including its offset. The counterpart of
    # feature_lateral_span on the depth axis.
    shift = feature.offset.z
    bounds = feature.blocks.bounds
    if bounds is None:
        return shift, shift
    return bounds.minimum.z + shift, bounds.maximum.z + shift


def required_lead(
    path: Path,
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> int:
    #Determines the amount of lead required for starts and and end to ensure a boundary feature
    sections = tuple(sections)
    return lead_for(
        path,
        max(
            longitudinal_reach(sections, turn),
            lateral_reach(sections, turn).half_width,
        ),
    )
