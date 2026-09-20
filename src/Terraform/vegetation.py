from __future__ import annotations

from collections.abc import Set

from ..Core.coordinates import BlockPos
from .components import components
from .grid import Grid


def clearing(
    growth: Set[BlockPos],
    grid: Grid,
    within: Set[tuple[int, int]],
) -> frozenset[BlockPos]:
    return frozenset(
        position
        for clump in components(growth)
        if any(
            grid.holds(each.x, each.z) and grid.at(each.x, each.z) in within
            for each in clump
        )
        for position in clump
    )
