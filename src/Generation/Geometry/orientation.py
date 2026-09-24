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
    # A diagonal is a single join piece, so each run of diagonal points keeps
    # it only at its first point and takes the next quarter after. A path
    # that jumps straight between quarters gets the join at the change.
    turns = list(walked)
    for start, stop in _diagonal_runs(turns):
        for index in range(start + 1, stop):
            turns[index] = _quartered(turns[index], ahead=True)
    for index in range(1, len(turns)):
        if _changes_quarter(turns[index - 1], turns[index]):
            turns[index] = _joint(turns[index - 1], turns[index])
    return tuple(turns)


def _changes_quarter(previous: SectionTurn, here: SectionTurn) -> bool:
    return (
        is_quarter(previous.absolute)
        and is_quarter(here.absolute)
        and previous.absolute != here.absolute
    )


def _joint(previous: SectionTurn, here: SectionTurn) -> SectionTurn:
    # The diagonal between the two quarters, laid at the later point.
    onward = 45 if abs(here.angle) > abs(previous.angle) else -45
    return SectionTurn(normalize(previous.turn + onward), here.direction, here.angle)


def _diagonal_runs(turns: Sequence[SectionTurn]) -> Iterator[tuple[int, int]]:
    # Half-open index ranges of consecutive points at one diagonal.
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
    # The quarter after the diagonal if `ahead`, else the one before.
    lower, upper = neighbors(turn.turn)
    leaving = upper if ahead else lower
    return SectionTurn(leaving, turn.direction, turn.angle)


@dataclass(frozen=True, slots=True)
class SectionOrientation:
    # The layout turn at each path point and the direction its boundary
    # faces: the true heading, or the snapped layout heading when
    # `snap_boundaries` is set.

    tolerance: float = JOIN_TOLERANCE
    snap_boundaries: bool = False

    def turns(self, path: Path) -> tuple[SectionTurn, ...]:
        return once_at_diagonals(
            turns_along(
                (point.gradient for point in path.path_points),
                tolerance=self.tolerance,
            )
        )

    def turn(
        self, path: Path, index: int, previous: SectionTurn = SectionTurn()
    ) -> SectionTurn:
        # One step of a walk, given the previous point's turn.
        points = path.path_points
        if not index:
            return turn_from(points[0].gradient, previous.direction, self.tolerance)
        return turn_of(
            points[index].gradient, points[index - 1].gradient, previous, self.tolerance
        )

    def __call__(self, path: Path, index: int) -> Vec2:
        # An Orientation.
        if self.snap_boundaries:
            return heading(snap(path.path_points[index].gradient, self.tolerance))
        return orthogonal(path, index)
