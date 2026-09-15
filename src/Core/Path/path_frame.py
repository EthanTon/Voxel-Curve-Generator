from __future__ import annotations

from dataclasses import dataclass

from .boundary import OrthogonalBoundary


@dataclass(frozen=True, slots=True)
class PathFrame:
    # A run of path points held at a single integer elevation.
    #
    # Both ends are orthogonal boundaries, and every boundary behaves the same
    # way whether it sits between two elevations or at either end of the path:
    # a column belongs to this frame when it is at-or-after `start_boundary`
    # and strictly before `end_boundary`. Adjacent frames share the boundary
    # between them, so they neither overlap nor leave a gap; the first frame
    # starts at the path's start boundary and the last ends at its end boundary.
    #
    # A frame knows only its own boundaries and level. The elevation on the
    # far side of either boundary belongs to the neighbouring frame, not here.

    index: int
    path_start: int
    path_stop: int

    start_boundary: OrthogonalBoundary
    end_boundary: OrthogonalBoundary

    y_offset: int

    def __post_init__(self) -> None:
        if self.path_stop < self.path_start:
            raise ValueError(
                f"Frame {self.index}: path_stop ({self.path_stop}) precedes "
                f"path_start ({self.path_start})"
            )

    def contains(self, x: float, z: float) -> bool:
        return self.start_boundary.after(x, z) and not self.end_boundary.after(x, z)

    @property
    def is_degenerate(self) -> bool:
        # A frame no column can ever occupy: the two planes coincide, or the
        # end plane sits at-or-behind the start plane along the shared normal.
        start = self.start_boundary
        end = self.end_boundary
        if not start.is_parallel_to(end):
            return False
        return end.side(start.origin.x, start.origin.z) >= 0.0
