from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .boundary import Boundary, spans_nothing


@dataclass(frozen=True, slots=True)
class PathFrame:
    # A run of path points at one elevation. It owns the columns at or after
    # `start_boundary` and before `end_boundary`, so neighbours tile exactly.

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


def frames_for_index(
    frames: Sequence[PathFrame],
    index: int,
    lead_in: int = 0,
    lead_out: int = 0,
) -> tuple[PathFrame, ...]:
    # Frames whose points, widened by `lead_in` before and `lead_out` after,
    # include `index`. A point on a shared boundary is in both frames.
    if not frames:
        raise ValueError("No frames to place a path point in")
    if lead_in < 0 or lead_out < 0:
        raise ValueError(
            f"Leads must be non-negative, got lead_in={lead_in}, lead_out={lead_out}"
        )
    return tuple(
        frame
        for frame in frames
        if frame.path_start - lead_in <= index <= frame.path_stop + lead_out
    )
