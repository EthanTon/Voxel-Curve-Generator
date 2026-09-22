from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Iterable, Iterator, Sequence

from ...Core.CrossSection.turns import (
    JOIN_TOLERANCE,
    is_quarter,
    neighbors,
    SectionTurn,
    heading,
    normalize,
    snap,
    turn_from,
    turn_of,
    turns_along,
)
from ...Core.Path.path import Path
from ...Core.coordinates import Vec2
from .boundaries import orthogonal


def once_at_diagonals(walked: Iterable[SectionTurn]) -> tuple[SectionTurn, ...]:
    # Every diagonal laid out once, at the point where it begins, with the
    # quarter it hands over to from there on.
    #
    # A diagonal is a join, not a stretch of path. The section stamps it as
    # one piece reaching half a section either side of itself, so a run of
    # points sitting near the diagonal wants one stamp between them, not one
    # each: stamping every point in the run smears the join along the path
    # and lays several times the deck. How wide that run is, and so how near
    # the diagonal the join lands, is the path's `diagonal_degrees`.
    turns = list(walked)
    for start, stop in _diagonal_runs(turns):
        for index in range(start + 1, stop):
            turns[index] = _quartered(turns[index], ahead=True)
    for index in range(1, len(turns)):
        if _changes_quarter(turns[index - 1], turns[index]):
            turns[index] = _joint(turns[index - 1], turns[index])
    return tuple(turns)


def _changes_quarter(previous: SectionTurn, here: SectionTurn) -> bool:
    # A path that swings past a diagonal between one point and the next has
    # no point near enough to it to hold the join, and goes straight from one
    # quarter to another. The join belongs at the change.
    return (
        is_quarter(previous.absolute)
        and is_quarter(here.absolute)
        and previous.absolute != here.absolute
    )


def _joint(previous: SectionTurn, here: SectionTurn) -> SectionTurn:
    # The diagonal between the quarter a point came from and the one it
    # reached, laid out at the point that reached it.
    onward = 45 if abs(here.angle) > abs(previous.angle) else -45
    return SectionTurn(normalize(previous.turn + onward), here.direction, here.angle)


def _diagonal_runs(turns: Sequence[SectionTurn]) -> Iterator[tuple[int, int]]:
    # Every stretch of consecutive points laid out at one diagonal.
    start = None
    for index, turn in enumerate(turns):
        diagonal = not is_quarter(turn.absolute)
        if diagonal and start is None:
            start = index
        elif start is not None and (
            not diagonal or turn.absolute != turns[start].absolute
        ):
            yield start, index
            start = index if diagonal else None
    if start is not None:
        yield start, len(turns)


def _quartered(turn: SectionTurn, ahead: bool) -> SectionTurn:
    # The same point put on a quarter instead: the one it is heading for past
    # the joint, and the one it came from before it.
    lower, upper = neighbors(turn.turn)
    leaving = upper if ahead else lower
    return SectionTurn(leaving, turn.direction, turn.angle)


@dataclass(frozen=True, slots=True)
class SectionOrientation:
    # One decision, per path point, for both the turn a section is laid out
    # at and the direction the boundary there faces. Keeping them together
    # is what lets seams follow the section's rows.
    #
    # With `snap_boundaries` off, boundaries stay orthogonal to the true
    # heading. With it on, they face the heading the section was laid out
    # for, so a clip never cuts a layout's rows at a slant.

    tolerance: float = JOIN_TOLERANCE
    snap_boundaries: bool = False

    def turns(self, path: Path) -> tuple[SectionTurn, ...]:
        # Every point's turn in one pass. Which way a point turned is decided
        # by the point before it, so a path is walked rather than indexed.
        return once_at_diagonals(
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
