from __future__ import annotations

from collections import deque
from collections.abc import Callable, Set

from ..Core.coordinates import BlockPos
from .components import components


# Minecraft branches and crowns also join across block edges and corners.
# Keep this connectivity local to trees; terrain support still uses faces.
TREE_STEPS = tuple(
    BlockPos(x, y, z)
    for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
    if x or y or z
)


def clearing(
    growth: Set[BlockPos],
    in_the_way: Callable[[BlockPos], bool],
    trees: Set[BlockPos] = frozenset(),
    trunks: Set[BlockPos] = frozenset(),
) -> frozenset[BlockPos]:
    # Trim obstructing foliage locally. Only an obstructed trunk removes a
    # whole tree: touching crowns must never turn this into forest clearing.
    direct = {position for position in growth if in_the_way(position)}
    groups = tuple(components(trunks, TREE_STEPS))
    removed = [any(position in direct for position in trunk) for trunk in groups]
    if not any(removed):
        return frozenset(direct)

    # Spread from every trunk through its crown. At equal distances a live
    # tree owns shared leaves, so its canopy survives beside a removed tree.
    distance = {}
    owner = {}
    pending = deque()
    for gone in (False, True):
        for trunk, remove in zip(groups, removed):
            if remove != gone:
                continue
            for position in sorted(trunk):
                distance[position] = 0
                owner[position] = remove
                pending.append(position)
    while pending:
        position = pending.popleft()
        for step in TREE_STEPS:
            neighbour = position + step
            if neighbour in trees and neighbour not in distance:
                distance[neighbour] = distance[position] + 1
                owner[neighbour] = owner[position]
                pending.append(neighbour)
    direct.update(position for position, remove in owner.items() if remove)
    return frozenset(direct)
