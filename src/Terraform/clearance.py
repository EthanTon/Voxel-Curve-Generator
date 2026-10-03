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
    running: Mapping[tuple[int, int], int] | None = None,
    running_headroom: int = 0,
) -> Clearance:
    # A footing is the bottom of the foundation, not the running surface.
    # Gaps below the latter are soil contact, not rooms to excavate. With
    # no running survey, retain the footing-based envelope behavior.
    floor = grid.filled(OPEN)
    ceiling = grid.filled(-OPEN)
    for (x, z), heights in envelope.items():
        if not grid.holds(x, z) or not heights:
            continue
        levels = sorted(set(heights))
        base = None if footing is None else footing.get((x, z))
        running_level = None if running is None else running.get((x, z))
        gaps = [
            (below + 1, above - 1)
            for below, above in zip(levels, levels[1:])
            if above - below > 1 and (base is None or below >= base)
            and (running_level is None or above - 1 > running_level)
        ]
        room = headroom(x, z)
        low = gaps[0][0] if gaps else levels[-1] + 1
        # Both enclosed rooms and open beds start at their own authored
        # floor, which can sit below the rails. A raised footing reserves
        # missing rail support; wholly buried gaps stay foundation contact.
        if running_level is not None and not gaps and base is not None:
            low = max(low, base + 1)
        high = max(
            gaps[-1][1] if gaps else -OPEN,
            levels[-1] + room if room > 0 else -OPEN,
        )
        if high < low:
            continue
        at = grid.at(x, z)
        floor[at] = low
        ceiling[at] = high
    # Headroom over the highest fitting is separate from the usable space
    # above a rail. A bridge/tunnel can disable the former without allowing
    # soil into the latter. Stop at the first authored overhead block so
    # the roof and the hill over it remain intact.
    if running is not None and running_headroom > 0:
        for (x, z), level in running.items():
            if not grid.holds(x, z):
                continue
            heights = envelope.get((x, z), ())
            overhead = [y for y in heights if y > level]
            bed = max((y for y in heights if y <= level), default=level)
            base = None if footing is None else footing.get((x, z))
            low = max(bed, base if base is not None else bed) + 1
            high = min(level + running_headroom, min(overhead) - 1 if overhead else OPEN)
            if high >= low:
                at = grid.at(x, z)
                floor[at] = min(floor[at], low)
                ceiling[at] = max(ceiling[at], high)
    return Clearance(floor, ceiling)
