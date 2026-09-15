from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from ..coordinates import Vec2


@dataclass(frozen=True, slots=True)
class Boundary:
    # A vertical plane through `origin`, facing along the horizontal `normal`.
    #
    # Only the geometry lives here. How a boundary is oriented relative to a
    # path -- orthogonal to travel or otherwise -- is decided in Generation.

    origin: Vec2
    normal: Vec2

    # The path point this boundary was placed at, when it came from one.
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
        # Build a boundary facing a (possibly unnormalised) horizontal direction.
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
        # Signed placement of a column relative to the plane.
        origin = self.origin
        normal = self.normal
        return (x - origin.x) * normal.x + (z - origin.z) * normal.z

    def after(self, x: float, z: float) -> bool:
        # Half-open convention: the plane itself belongs to the side after it.
        return self.side(x, z) >= 0.0

    def is_parallel_to(self, other: Boundary, tolerance: float = 1e-9) -> bool:
        cross = self.normal.x * other.normal.z - self.normal.z * other.normal.x
        return abs(cross) <= tolerance


def spans_nothing(start: Boundary, end: Boundary) -> bool:
    # True when no column can be at-or-after `start` and before `end`: the
    # planes are parallel and the end sits at-or-behind the start.
    if not start.is_parallel_to(end):
        return False
    return end.side(start.origin.x, start.origin.z) >= 0.0


@dataclass(frozen=True, slots=True)
class PathBounds:
    # The columns a path owns: at-or-after `start` and strictly before `end`,
    # the same half-open rule as every other boundary.

    start: Boundary
    end: Boundary

    def contains(self, x: float, z: float) -> bool:
        return self.start.after(x, z) and not self.end.after(x, z)

    @property
    def is_degenerate(self) -> bool:
        return spans_nothing(self.start, self.end)
