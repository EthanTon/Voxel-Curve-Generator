from __future__ import annotations

from collections.abc import Callable, Set

from ..Core.coordinates import BlockPos
from .components import components


def clearing(
    growth: Set[BlockPos],
    in_the_way: Callable[[BlockPos], bool],
    trees: Set[BlockPos] = frozenset(),
) -> frozenset[BlockPos]:
    # A touched tree is cleared whole so nothing is left floating; other
    # growth only where it is itself in the way.
    whole = frozenset(
        position
        for clump in components(trees)
        if any(in_the_way(each) for each in clump)
        for position in clump
    )
    loose = frozenset(
        position for position in growth if position not in trees and in_the_way(position)
    )
    return whole | loose
