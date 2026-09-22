from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType

from ..modes import StepMode

# Given an annotation's local column (x relative to the origin), the positions
# it delivers, as blocks forward of where it sits: the first is its own
# position. The last also sets how far it moves every point beyond it on its
# side.
StepRule = Callable[[int], tuple[int, ...]]


def step_direction(column: int) -> int:
    # Forward (1) on the left and on the origin's column, backward (-1) on the
    # right.
    return -1 if column > 0 else 1


def chebyshev_step(column: int) -> tuple[int, ...]:
    return (0,)


def manhattan_step(column: int) -> tuple[int, ...]:
    return (0, step_direction(column))


STEP_RULES: Mapping[StepMode, StepRule] = MappingProxyType(
    {
        StepMode.CHEBYSHEV: chebyshev_step,
        StepMode.MANHATTAN: manhattan_step,
    }
)


def column_pushes(steps: Iterable[tuple[int, tuple[int, ...]]]) -> dict[int, int]:
    # How far each column moves the points beyond it, from (column, steps)
    # pairs. Points sharing a column push together, by the largest step among
    # them, so a column moves the points beyond it once.
    pushes: dict[int, int] = {}
    for column, offsets in steps:
        push = offsets[-1]
        if abs(push) > abs(pushes.get(column, 0)):
            pushes[column] = push
    return pushes


def lead(column: int, pushes: Mapping[int, int]) -> int:
    # How far a point at `column` is moved by the columns between it and the
    # origin on its side.
    #
    # Everything beyond a stepping column moves with it, whether it steps
    # itself or not. That is what makes the row a staircase rather than a
    # straight line with two bumps in it: the stepping column is the corner
    # joining the segment inboard of it to the segment outboard.
    return sum(
        push
        for pushing, push in pushes.items()
        if (column - pushing) * step_direction(pushing) < 0
    )
