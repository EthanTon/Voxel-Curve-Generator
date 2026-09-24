from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import hypot

from ..CrossSection.turns import (
    CLOCKWISE,
    Rotation,
    Turn,
    angle_between,
    normalize,
)
from ..coordinates import BlockPos, Vec3


@dataclass(frozen=True, slots=True)
class FeaturePathPoint:
    # The path point it was delivered at, and its index in this feature path.
    path_index: int
    point_index: int
    # World x and z; y is height above the path.
    position: BlockPos
    # The path's gradient at `path_index`.
    gradient: Vec3
    turn: Turn
    direction: Rotation
    # The layout that placed this position, which differs from `turn` where
    # two layouts overlap at a seam. None if no layout placed it.
    layout: Turn | None = None
    mirrored: bool = False
    # Elevation at this point's column; rigid features use it for all blocks.
    y_offset: int = 0

    @property
    def laid_at(self) -> Turn:
        if self.layout is not None:
            return self.layout
        return normalize(self.turn * self.direction)


# Whether a point may be appended, given the path before it.
FeaturePathRule = Callable[["FeaturePath", FeaturePathPoint], bool]


def _has_position(points: Iterable[FeaturePathPoint], point: FeaturePathPoint) -> bool:
    return any(kept.position == point.position for kept in points)


def not_in_previous_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    return not _has_position(path.previous_delivery, point)


def not_in_current_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    return not _has_position(path.current_delivery, point)


DEFAULT_RULES: tuple[FeaturePathRule, ...] = (
    not_in_previous_delivery,
    not_in_current_delivery,
)


def either(*rules: FeaturePathRule) -> FeaturePathRule:
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        return any(each(path, point) for each in rules)

    return rule


def apart(spacing: float) -> FeaturePathRule:
    # Keeps a point `spacing` blocks in plan from the last kept.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not len(path):
            return True
        last = path[-1].position
        return hypot(
            point.position.x - last.x, point.position.z - last.z
        ) >= spacing

    return rule


def every(count: int) -> FeaturePathRule:
    # Keeps a point `count` path points after the last kept.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not len(path):
            return True
        return point.path_index - path[-1].path_index >= count

    return rule


def turning(angle: float) -> FeaturePathRule:
    # Keeps a point once the heading has turned `angle` degrees since the
    # last kept, which bounds how far a wire strays from the track.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not len(path):
            return True
        return abs(angle_between(path[-1].gradient, point.gradient)) >= angle

    return rule


def one_per_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    # Keeps only the first position each path point delivers.
    return not path.current_delivery


def ending_at(position: BlockPos) -> FeaturePathRule:
    # Keeps the run's final position regardless of spacing.
    position = BlockPos.coerce(position)

    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        return point.position == position

    return rule


def spacing_rules(
    spacing: float, angle: float, by_index: bool = False
) -> tuple[FeaturePathRule, ...]:
    # Keeps points far enough along or far enough round from the last kept.
    # Every point by index means no rule, so a join keeps all its positions.
    if by_index and 0 < spacing <= 1 and angle <= 0:
        return ()
    rules = []
    if spacing > 0:
        rules.append(every(int(spacing)) if by_index else apart(spacing))
    if angle > 0:
        rules.append(turning(angle))
    if not rules:
        return ()
    return (either(*rules),)


class FeaturePath:
    # An ordered run of delivered points, filtered by rules. Points sharing a
    # path_index form one delivery.

    def __init__(self, rules: Iterable[FeaturePathRule] = DEFAULT_RULES) -> None:
        self.rules = tuple(rules)
        self._path: list[FeaturePathPoint] = []
        self._previous: list[FeaturePathPoint] = []
        self._current: list[FeaturePathPoint] = []
        self._delivery_index: int | None = None

    @property
    def points(self) -> tuple[FeaturePathPoint, ...]:
        return tuple(self._path)

    @property
    def previous_delivery(self) -> tuple[FeaturePathPoint, ...]:
        return tuple(self._previous)

    @property
    def current_delivery(self) -> tuple[FeaturePathPoint, ...]:
        return tuple(self._current)

    def add(
        self,
        position: BlockPos | Sequence[int],
        gradient: Vec3 | Sequence[float],
        turn: Turn,
        path_index: int,
        y_offset: int,
        direction: Rotation = CLOCKWISE,
        layout: Turn | None = None,
        mirrored: bool = False,
    ) -> FeaturePathPoint | None:
        if path_index != self._delivery_index:
            if self._current:
                self._previous = self._current
                self._current = []
            self._delivery_index = path_index

        point = FeaturePathPoint(
            path_index=path_index,
            point_index=len(self._path),
            position=BlockPos.coerce(position),
            gradient=Vec3.coerce(gradient),
            turn=turn,
            direction=direction,
            y_offset=y_offset,
            layout=layout,
            mirrored=mirrored,
        )
        if not all(rule(self, point) for rule in self.rules):
            return None
        self._path.append(point)
        self._current.append(point)
        return point

    def add_point(
        self,
        point: FeaturePathPoint,
        *,
        position: BlockPos | None = None,
        y_offset: int | None = None,
    ) -> FeaturePathPoint | None:
        # Adds a point from another path, optionally moved.
        return self.add(
            point.position if position is None else position,
            point.gradient,
            point.turn,
            point.path_index,
            point.y_offset if y_offset is None else y_offset,
            point.direction,
            point.layout,
            point.mirrored,
        )

    def __iter__(self) -> Iterator[FeaturePathPoint]:
        return iter(self._path)

    def __len__(self) -> int:
        return len(self._path)

    def __getitem__(self, index: int) -> FeaturePathPoint:
        return self._path[index]
