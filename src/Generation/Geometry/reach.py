from __future__ import annotations

from collections.abc import Iterable

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.turns import Turn, heading
from ...Core.Feature.feature import Feature
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
