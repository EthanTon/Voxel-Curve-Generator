from __future__ import annotations

from dataclasses import dataclass

from ...Core.CrossSection.turns import (
    DIAGONAL_TOLERANCE,
    SectionTurn,
    heading,
    snap,
    turn_from,
    turn_of,
    turns_along,
)
from ...Core.Path.path import Path
from ...Core.coordinates import Vec2
from .boundaries import orthogonal


@dataclass(frozen=True, slots=True)
class SectionOrientation:
    # One decision, per path point, for both the turn a section is laid out
    # at and the direction the boundary there faces. Keeping them together
    # is what lets seams follow the section's rows.
    #
    # With `snap_boundaries` off, boundaries stay orthogonal to the true
    # heading. With it on, they face the heading the section was laid out
    # for, so a clip never cuts a layout's rows at a slant.

    tolerance: float = DIAGONAL_TOLERANCE
    snap_boundaries: bool = False

    def turns(self, path: Path) -> tuple[SectionTurn, ...]:
        # Every point's turn in one pass. Which way a point turned is decided
        # by the point before it, so a path is walked rather than indexed.
        return tuple(
            turns_along(
                (point.gradient for point in path.path_points),
                tolerance=self.tolerance,
            )
        )

    def turn(
        self, path: Path, index: int, previous: SectionTurn = SectionTurn()
    ) -> SectionTurn:
        # One point's turn, for a caller already walking the path and holding
        # the turn before it, which carries the angle the path has turned by.
        # The first point has nothing before it to turn from, so it starts one.
        points = path.path_points
        if not index:
            return turn_from(points[0].gradient, previous.direction, self.tolerance)
        return turn_of(
            points[index].gradient, points[index - 1].gradient, previous, self.tolerance
        )

    def __call__(self, path: Path, index: int) -> Vec2:
        # Usable anywhere Geometry takes an Orientation. A boundary only faces
        # a heading, which is the same whichever way the path turned to reach
        # it, so this needs no history.
        if self.snap_boundaries:
            return heading(snap(path.path_points[index].gradient, self.tolerance))
        return orthogonal(path, index)
