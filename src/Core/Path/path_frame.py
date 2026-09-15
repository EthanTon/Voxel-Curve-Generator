from __future__ import annotations

from dataclasses import dataclass

from .boundary import Boundary, spans_nothing


@dataclass(frozen=True, slots=True)
class PathFrame:
    # A run of path points held at a single integer elevation.
    #
    # Both ends are boundaries, and every boundary behaves the same way whether
    # it sits between two elevations or at either end of the path: a column
    # belongs to this frame when it is at-or-after `start_boundary` and
    # strictly before `end_boundary`. Adjacent frames share the boundary
    # between them, so they neither overlap nor leave a gap.
    #
    # A frame knows only its own boundaries and level. The elevation on the
    # far side of either boundary belongs to the neighbouring frame, and how
    # the boundaries relate to the path is checked in Generation.

    index: int
    path_start: int
    path_stop: int

    start_boundary: Boundary
    end_boundary: Boundary

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
        # A frame no column can ever occupy.
        return spans_nothing(self.start_boundary, self.end_boundary)
