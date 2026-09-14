from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

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
    # Elevation at this point's own column. Features that ignore per-block
    # elevation use it as the y offset for everything built from this point.
    y_offset: int


# Decides whether a point may be appended to a path. Rules see the path as it
# is before the point is added.
FeaturePathRule = Callable[["FeaturePath", FeaturePathPoint], bool]


def not_in_previous_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    return all(kept.position != point.position for kept in path.previous_delivery)


def not_in_current_delivery(path: FeaturePath, point: FeaturePathPoint) -> bool:
    return all(kept.position != point.position for kept in path.current_delivery)


DEFAULT_RULES: tuple[FeaturePathRule, ...] = (
    not_in_previous_delivery,
    not_in_current_delivery,
)


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
        path_index: int,
        y_offset: int,
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
            y_offset=y_offset,
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
