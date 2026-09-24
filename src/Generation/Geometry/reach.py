from __future__ import annotations

from collections.abc import Iterable, Iterator

from ...Core.CrossSection.cross_section import CrossSection
from ...Core.CrossSection.turns import Turn, heading
from ...Core.Feature.feature import Feature
from ...Core.Path.path import Path
from ...Core.coordinates import BlockPos, Vec2
from ..Sweep.schedule import lead_for
from .boundaries import LateralReach


def lateral_reach(
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> LateralReach:
    # How far the sections' oriented positions, widened by their features'
    # blocks, reach left and right of travel at `turn` or at every turn.
    left = right = 0.0
    for forward, feature, position in _oriented(sections, turn):
        low, high = feature_lateral_span(feature)
        lateral = position.x * -forward.z + position.z * forward.x
        right = max(right, lateral + high)
        left = max(left, -(lateral + low))
    return LateralReach(float(left), float(right))


def longitudinal_reach(
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> float:
    # The same along travel, ahead or behind. Block extents are unsigned
    # since features turn only when placed.
    furthest = 0.0
    for forward, feature, position in _oriented(sections, turn):
        low, high = feature_longitudinal_span(feature)
        along = position.x * forward.x + position.z * forward.z
        furthest = max(furthest, abs(along + high), abs(along + low))
    return float(furthest)


def _oriented(
    sections: Iterable[CrossSection], turn: Turn | None
) -> Iterator[tuple[Vec2, Feature, BlockPos]]:
    # (heading, feature, position) for every position of every layout.
    turns = range(0, 360, 45) if turn is None else (turn,)
    for section in sections:
        for each in turns:
            forward = heading(each)
            for offset in (o for laid in section.layouts(each) for o in laid):
                for position in offset.offsets:
                    yield forward, offset.annotation.feature, position


def feature_lateral_span(feature: Feature) -> tuple[int, int]:
    # The local x range a feature's blocks cover, offset included.
    return _feature_span(feature, "x")


def feature_longitudinal_span(feature: Feature) -> tuple[int, int]:
    # The local z range a feature's blocks cover, offset included.
    return _feature_span(feature, "z")


def _feature_span(feature: Feature, axis: str) -> tuple[int, int]:
    shift = getattr(feature.offset, axis)
    bounds = feature.blocks.bounds
    if bounds is None:
        return shift, shift
    return getattr(bounds.minimum, axis) + shift, getattr(bounds.maximum, axis) + shift


def required_lead(
    path: Path,
    sections: Iterable[CrossSection],
    turn: Turn | None = None,
) -> int:
    # Padding points needed at each end for the sections' reach.
    sections = tuple(sections)
    return lead_for(
        path,
        max(
            longitudinal_reach(sections, turn),
            lateral_reach(sections, turn).half_width,
        ),
    )
