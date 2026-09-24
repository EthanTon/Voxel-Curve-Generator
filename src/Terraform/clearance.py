from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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
        return self.floor.item(row, cell) <= y <= self.ceiling.item(row, cell)

    def merged(self, other: Clearance) -> Clearance:
        return Clearance(
            np.minimum(self.floor, other.floor),
            np.maximum(self.ceiling, other.ceiling),
        )

    def limit(self, designed: Field) -> Field:
        return np.minimum(designed, self.floor - 1)

    def limit_surface(self, natural: Field, designed: Field) -> Field:
        # A buried clearance is a bore, not a cut from the open surface.
        # Its blocks are removed separately by the surveyed intrusion set.
        exposed = self.ceiling >= natural
        return np.where(exposed, self.limit(designed), designed)

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


def enclosing(
    grid: Grid,
    envelope: Mapping[tuple[int, int], Sequence[int]],
    headroom: Callable[[int, int], int],
    footing: Mapping[tuple[int, int], int] | None = None,
) -> Clearance:
    # Per column: every gap between the build's blocks (above the track's
    # `footing`, if given) plus `headroom` over the top block. Ground in the
    # gaps is cleared at any depth, e.g. a tunnel bore.
    floor = grid.filled(OPEN)
    ceiling = grid.filled(-OPEN)
    for (x, z), heights in envelope.items():
        if not grid.holds(x, z) or not heights:
            continue
        levels = sorted(set(heights))
        base = None if footing is None else footing.get((x, z))
        gaps = [
            (below + 1, above - 1)
            for below, above in zip(levels, levels[1:])
            if above - below > 1 and (base is None or below >= base)
        ]
        room = headroom(x, z)
        low = gaps[0][0] if gaps else levels[-1] + 1
        high = max(
            gaps[-1][1] if gaps else -OPEN,
            levels[-1] + room if room > 0 else -OPEN,
        )
        if high < low:
            continue
        at = grid.at(x, z)
        floor[at] = low
        ceiling[at] = high
    return Clearance(floor, ceiling)
