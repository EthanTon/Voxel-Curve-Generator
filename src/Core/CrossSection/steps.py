from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType

from ..modes import StepMode

# Local column -> the blocks forward it delivers at, its own position first.
# The last also sets how far it pushes every column outboard of it.
StepRule = Callable[[int], tuple[int, ...]]


def step_direction(column: int) -> int:
    # Forward on the left and centre, backward on the right.
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
    # Each column's push: its largest last step, from (column, steps) pairs.
    pushes: dict[int, int] = {}
    for column, offsets in steps:
        push = offsets[-1]
        if abs(push) > abs(pushes.get(column, 0)):
            pushes[column] = push
    return pushes


def lead(column: int, pushes: Mapping[int, int]) -> int:
    # The total push from the columns inboard of `column` on its side, which
    # turns a row with stepping columns into a staircase.
    return sum(
        push
        for pushing, push in pushes.items()
        if (column - pushing) * step_direction(pushing) < 0
    )
