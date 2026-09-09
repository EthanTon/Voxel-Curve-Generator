from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from ..coordinates import Vec2
from .path import PathPoint


@dataclass(frozen=True, slots=True)
class OrthogonalBoundary:
    path_index: int
    origin: Vec2

    # Horizontal path tangent. This is the boundary-plane normal.
    normal: Vec2

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", Vec2.coerce(self.origin))
        normal = Vec2.coerce(self.normal)
        if normal.x == 0.0 and normal.z == 0.0:
            raise ValueError("OrthogonalBoundary normal must be non-zero")
        object.__setattr__(self, "normal", normal)

    @classmethod
    def from_tangent(
        cls,
        path_index: int,
        origin: Vec2 | tuple[float, float],
        tangent: Vec2 | tuple[float, float],
    ) -> OrthogonalBoundary:
        # Build a boundary from a (possibly unnormalised) horizontal tangent.
        tangent = Vec2.coerce(tangent)
        length = hypot(tangent.x, tangent.z)
        if length == 0.0:
            raise ValueError("Cannot build a boundary from a zero tangent")
        return cls(
            path_index=path_index,
            origin=origin,
            normal=Vec2(tangent.x / length, tangent.z / length),
        )

    @classmethod
    def from_path_point(cls, path_index: int, point: PathPoint) -> OrthogonalBoundary:
        # The boundary at a path point is perpendicular to travel in plan view,
        # so its normal is the horizontal projection of that point's gradient.
        # The vertical component of the gradient plays no part here -- it is the
        # grade, and it belongs to the frame's elevation, not to the plane.
        if point.is_vertical:
            raise ValueError(
                f"Path point at {point.position} has no horizontal heading; "
                "a boundary plane cannot be oriented from it"
            )
        return cls.from_tangent(
            path_index,
            Vec2(point.position.x, point.position.z),
            point.heading,
        )

    def side(self, x: float, z: float) -> float:
        # Signed placement of a column relative to the plane.
        origin = self.origin
        normal = self.normal
        return (x - origin.x) * normal.x + (z - origin.z) * normal.z

    def after(self, x: float, z: float) -> bool:
        # Half-open convention: the plane itself belongs to the frame after it.
        return self.side(x, z) >= 0.0

    def is_parallel_to(self, other: OrthogonalBoundary, tolerance: float = 1e-9) -> bool:
        cross = self.normal.x * other.normal.z - self.normal.z * other.normal.x
        return abs(cross) <= tolerance


@dataclass(frozen=True, slots=True)
class PathFrame:
    # A run of path points held at a single integer elevation.
    #
    # Frames tile the path half-open: a column belongs to this frame when it is
    # at-or-after `start_boundary` and strictly before `end_boundary`. Adjacent
    # frames therefore neither overlap nor leave a gap. A `None` boundary means
    # unbounded on that side, which is how the first and last frames terminate.

    index: int
    path_start: int
    path_stop: int

    start_boundary: OrthogonalBoundary | None
    end_boundary: OrthogonalBoundary | None

    y_offset: int

    def __post_init__(self) -> None:
        if self.path_stop < self.path_start:
            raise ValueError(
                f"Frame {self.index}: path_stop ({self.path_stop}) precedes "
                f"path_start ({self.path_start})"
            )

    def contains(self, x: float, z: float) -> bool:
        if self.start_boundary is not None and not self.start_boundary.after(x, z):
            return False
        if self.end_boundary is not None and self.end_boundary.after(x, z):
            return False
        return True

    @property
    def is_open_ended(self) -> bool:
        return self.start_boundary is None or self.end_boundary is None

    @property
    def is_degenerate(self) -> bool:
        # A frame no column can ever occupy: the two planes coincide, or the
        # end plane sits at-or-behind the start plane along the shared normal.
        start = self.start_boundary
        end = self.end_boundary
        if start is None or end is None:
            return False
        if not start.is_parallel_to(end):
            return False
        return end.side(start.origin.x, start.origin.z) >= 0.0
