from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Iterable, Iterator, Sequence

from ...Core.CrossSection.turns import (
    JOIN_TOLERANCE,
    is_quarter,
    neighbors,
    SectionTurn,
    Turn,
    heading,
    normalize,
    snap,
    tolerance_for,
    turn_at,
    turn_from,
    turn_of,
    turns_along,
)
from ...Core.Path.path import Path
from ...Core.coordinates import Vec2
from .boundaries import orthogonal

# Headings this near a diagonal lie on it, whatever float noise the walk adds.
ON_DIAGONAL = tolerance_for(1e-6)


def once_at_diagonals(walked: Iterable[SectionTurn]) -> tuple[SectionTurn, ...]:
    # A diagonal is a single join piece, so each run of diagonal points keeps
    # it only at its first point and takes the next quarter after. A path
    # that jumps straight between quarters gets the join at the change.
    turns = list(walked)
    for start, stop in _diagonal_runs(turns):
        if start == 0:
            # A path setting off near a diagonal heads to one side of it
            # until it first crosses; that stretch is laid on its side.
            start = _sided(turns, start, stop)
            if start == stop:
                continue
        # The sign of the accumulated world heading does not say which way
        # this bend turns (east -> north still has a positive total angle).
        # Choose the join's arms from the local change through the diagonal.
        before, after = max(0, start - 1), min(len(turns) - 1, stop)
        beyond = (
            turns[stop].absolute if stop < len(turns)
            else _side_of(turns, range(stop - 1, start - 1, -1))
        )
        if is_quarter(turns[before].absolute) and turns[before].absolute == beyond:
            # Touching the diagonal and returning to the same quarter, or
            # ending beside it, is not a handover. Advancing its later
            # samples would invent a second quarter and let corner search
            # relayout distant rows.
            for index in range(start, stop):
                turns[index] = _laid_in(turns[before].absolute, turns[index])
            continue
        change = turns[after].angle - turns[before].angle
        if abs(change) > 1e-9:
            direction = 1 if change > 0 else -1
            for index in range(start, stop):
                turn = turns[index]
                turns[index] = SectionTurn(
                    normalize(turn.absolute * direction), direction, turn.angle
                )
        for index in range(start + 1, stop):
            turns[index] = _quartered(turns[index], ahead=True)
    for index in range(1, len(turns)):
        if _changes_quarter(turns[index - 1], turns[index]):
            turns[index] = _joint(turns[index - 1], turns[index])
    return tuple(turns)


def _side(turn: SectionTurn) -> Turn | None:
    # The quarter on the side of its diagonal a point heads, None on it.
    nearest = turn_at(turn.angle, ON_DIAGONAL).absolute
    return nearest if is_quarter(nearest) else None


def _side_of(turns: Sequence[SectionTurn], indices: Iterable[int]) -> Turn | None:
    return next((side for index in indices if (side := _side(turns[index])) is not None), None)


def _laid_in(quarter: Turn, turn: SectionTurn) -> SectionTurn:
    return SectionTurn(normalize(quarter * turn.direction), turn.direction, turn.angle)


def _sided(turns: list[SectionTurn], start: int, stop: int) -> int:
    # Lays the run's points in the quarter beside its diagonal up to the
    # first that heads across it, and returns that point (or `stop`).
    side = _side_of(turns, range(start, stop))
    if side is None:
        return start
    index = start
    while index < stop and _side(turns[index]) in (None, side):
        turns[index] = _laid_in(side, turns[index])
        index += 1
    return index


def _changes_quarter(previous: SectionTurn, here: SectionTurn) -> bool:
    return (
        is_quarter(previous.absolute)
        and is_quarter(here.absolute)
        and previous.absolute != here.absolute
    )


def _joint(previous: SectionTurn, here: SectionTurn) -> SectionTurn:
    # The diagonal between the two quarters, laid at the later point.
    change = here.angle - previous.angle
    direction = 1 if change >= 0 else -1
    absolute = normalize(previous.absolute + 45 * direction)
    return SectionTurn(normalize(absolute * direction), direction, here.angle)


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
