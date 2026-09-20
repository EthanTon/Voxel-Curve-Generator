from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..Core.coordinates import BlockPos
from .grid import Field, Grid


OPEN = 1 << 20


@dataclass(frozen=True, slots=True)
class Clearance:
    floor: Field
    ceiling: Field

    @classmethod
    def none(cls, grid: Grid) -> Clearance:
        return cls(grid.filled(OPEN), grid.filled(-OPEN))

    def holds(self, row: int, cell: int, y: int) -> bool:
        return self.floor[row, cell] <= y <= self.ceiling[row, cell]

    def merged(self, other: Clearance) -> Clearance:
        return Clearance(
            np.minimum(self.floor, other.floor),
            np.maximum(self.ceiling, other.ceiling),
        )

    def limit(self, designed: Field) -> Field:
        return np.minimum(designed, self.floor - 1)

    def carved(self, solid: Iterable[BlockPos], grid: Grid) -> frozenset[BlockPos]:
        return frozenset(
            position
            for position in solid
            if grid.holds(position.x, position.z)
            and self.holds(*grid.at(position.x, position.z), position.y)
        )


def above(grid: Grid, columns: Iterable[BlockPos], headroom: int) -> Clearance:
    floor = grid.filled(OPEN)
    ceiling = grid.filled(-OPEN)
    for position in columns:
        if not grid.holds(position.x, position.z):
            continue
        at = grid.at(position.x, position.z)
        floor[at] = min(floor[at], position.y + 1)
        ceiling[at] = max(ceiling[at], position.y + headroom)
    return Clearance(floor, ceiling)


def open_sky(grid: Grid, columns: Iterable[BlockPos]) -> Clearance:
    return above(grid, columns, OPEN)
