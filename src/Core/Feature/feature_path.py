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
    # Index of the path point this point was delivered at.
    path_index: int
    # Index of this point within its feature path.
    point_index: int
    # World x and z, with y as the local height above the path.
    position: BlockPos
    # Gradient of the path point, not of the feature path.
    gradient: Vec3
    # The turn its section was laid out at based on the path.
    turn: Turn
    # Which way the path was turning there. A turn alone does not say: the
    # same heading is reached by turning either way round, and the section is
    # laid out mirrored when it was reached counter-clockwise.
    direction: Rotation
    # The one of the eight layouts a section placed this position in. Where
    # a point is laid out more than once, to cover the seam between a run of
    # one layout and a run of another, it is not the turn the path point is
    # at. None where nothing placed it in a layout at all.
    layout: Turn | None = None
    # Whether that layout is the mirrored one. The quarters alternate, so the
    # layout alone does not say which way depth runs.
    mirrored: bool = False
    # Elevation at this point's own column. Features that ignore per-block
    # elevation use it as the y offset for everything built from this point.
    y_offset: int = 0

    @property
    def laid_at(self) -> Turn:
        # The layout that placed this position, or the point's own turn where
        # nothing placed it in one -- a lane paints onto the path.
        if self.layout is not None:
            return self.layout
        return normalize(self.turn * self.direction)


# Decides whether a point may be appended to a path. Rules see the path as it
# is before the point is added.
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
    # A point any of these rules would keep. Rules are otherwise all required,
    # so this is how a point earns its place two different ways.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        return any(each(path, point) for each in rules)

    return rule


def apart(spacing: float) -> FeaturePathRule:
    # Keep a point once it is `spacing` blocks from the last one kept,
    # measured in plan so a climb does not shorten the gap.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not path.points:
            return True
        last = path.points[-1].position
        return hypot(
            point.position.x - last.x, point.position.z - last.z
        ) >= spacing

    return rule


def every(count: int) -> FeaturePathRule:
    # Keep a point once `count` path points have passed since the last one
    # kept. Counted in points rather than blocks, for a run whose points are
    # not a block apart.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not path.points:
            return True
        return point.path_index - path.points[-1].path_index >= count

    return rule


def turning(angle: float) -> FeaturePathRule:
    # Keep a point once the path's heading there is `angle` degrees off its
    # heading at the last point kept.
    #
    # This is what holds a wire over the track it spans: the line between two
    # poles is straight, so bounding how far the track turns between them
    # bounds how far the wire strays from it. Measured between the headings
    # rather than along the line between the poles, which on block positions
    # is quantised to eighths of a turn and says little over a short span.
    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        if not path.points:
            return True
        return abs(angle_between(path.points[-1].gradient, point.gradient)) >= angle

    return rule


def one_per_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    # Keep the first position a path point delivered and drop the rest.
    #
    # A swept profile delivers more than one position at a point wherever it
    # would otherwise leave a gap, which is what keeps a deck solid round a
    # bend. Something painted along that run inherits the filling and comes
    # out four-connected. Thinned to one position a point, it steps
    # diagonally instead.
    return not path.current_delivery


def ending_at(position: BlockPos) -> FeaturePathRule:
    # Always keep the point a run ends on. A wire is anchored at both ends,
    # and so is whatever holds it up; without this a run whose length is not a
    # whole number of spacings stops short of its own end.
    #
    # Matched on the position rather than the path point it came from: one
    # path point delivers several positions, and only the last is the end.
    position = BlockPos.coerce(position)

    def rule(path: FeaturePath, point: FeaturePathPoint) -> bool:
        return point.position == position

    return rule


def spacing_rules(
    spacing: float, angle: float, by_index: bool = False
) -> tuple[FeaturePathRule, ...]:
    # Where something placed at intervals along a path sits: far enough
    # along it, or far enough round from the last one. Either on its own is
    # the whole rule; neither keeps every point.
    rules = []
    if spacing > 0:
        rules.append(every(int(spacing)) if by_index else apart(spacing))
    if angle > 0:
        rules.append(turning(angle))
    if not rules:
        return ()
    return (either(*rules),)


class FeaturePath:
    # An ordered run of delivered points. Like a path, it may cross or loop
    # back over itself; only the rules decide what is kept.
    #
    # Points added with the same path_index form one delivery. A new
    # path_index starts the next delivery, and the last delivery that kept any
    # points becomes the previous one.

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

    # Append a point to the path. Returns None when a rule rejects it.
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

    def __iter__(self) -> Iterator[FeaturePathPoint]:
        return iter(self._path)

    def __len__(self) -> int:
        return len(self._path)

    def __getitem__(self, index: int) -> FeaturePathPoint:
        return self._path[index]
