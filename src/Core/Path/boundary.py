from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from ..coordinates import Vec2


@dataclass(frozen=True, slots=True)
class Boundary:
    # A vertical plane through `origin`, facing along the horizontal `normal`.

    origin: Vec2
    normal: Vec2
    path_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", Vec2.coerce(self.origin))
        normal = Vec2.coerce(self.normal)
        if normal.x == 0.0 and normal.z == 0.0:
            raise ValueError("Boundary normal must be non-zero")
        object.__setattr__(self, "normal", normal)

    @classmethod
    def from_direction(
        cls,
        origin: Vec2 | tuple[float, float],
        direction: Vec2 | tuple[float, float],
        path_index: int | None = None,
    ) -> Boundary:
        direction = Vec2.coerce(direction)
        length = hypot(direction.x, direction.z)
        if length == 0.0:
            raise ValueError("Cannot build a boundary from a zero direction")
        return cls(
            origin=origin,
            normal=Vec2(direction.x / length, direction.z / length),
            path_index=path_index,
        )

    def side(self, x: float, z: float) -> float:
        origin = self.origin
        normal = self.normal
        return (x - origin.x) * normal.x + (z - origin.z) * normal.z

    def after(self, x: float, z: float) -> bool:
        # The plane itself counts as after.
        return self.side(x, z) >= 0.0

    def is_parallel_to(self, other: Boundary, tolerance: float = 1e-9) -> bool:
        cross = self.normal.x * other.normal.z - self.normal.z * other.normal.x
        return abs(cross) <= tolerance


def spans_nothing(start: Boundary, end: Boundary) -> bool:
    # Planes facing the same way with `end` at or behind `start` enclose no
    # column; planes facing opposite ways always enclose some.
    if not start.is_parallel_to(end):
        return False
    if start.normal.x * end.normal.x + start.normal.z * end.normal.z < 0.0:
        return False
    return end.side(start.origin.x, start.origin.z) >= 0.0


@dataclass(frozen=True, slots=True)
class PathBounds:
    # The columns at or after `start` and before `end`.

    start: Boundary
    end: Boundary

    def contains(self, x: float, z: float) -> bool:
        return self.start.after(x, z) and not self.end.after(x, z)

    @property
    def is_degenerate(self) -> bool:
        return spans_nothing(self.start, self.end)
