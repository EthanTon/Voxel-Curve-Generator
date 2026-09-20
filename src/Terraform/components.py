from __future__ import annotations

from collections.abc import Iterable, Iterator, Set

from ..Core.coordinates import BlockPos


FACES = (
    BlockPos(1, 0, 0),
    BlockPos(-1, 0, 0),
    BlockPos(0, 1, 0),
    BlockPos(0, -1, 0),
    BlockPos(0, 0, 1),
    BlockPos(0, 0, -1),
)


def reaching(within: Set[BlockPos], sources: Iterable[BlockPos]) -> frozenset[BlockPos]:
    found = {each for each in sources if each in within}
    pending = list(found)
    while pending:
        position = pending.pop()
        for step in FACES:
            neighbour = position + step
            if neighbour in within and neighbour not in found:
                found.add(neighbour)
                pending.append(neighbour)
    return frozenset(found)


def components(cells: Set[BlockPos]) -> Iterator[frozenset[BlockPos]]:
    remaining = set(cells)
    while remaining:
        found = reaching(remaining, [next(iter(remaining))])
        remaining -= found
        yield found


def touching(cells: Set[BlockPos], anchors: Set[BlockPos]) -> frozenset[BlockPos]:
    return frozenset(
        position
        for component in components(cells)
        for position in component
        if any(each + step in anchors for each in component for step in FACES)
    )


def detached(cells: Set[BlockPos], anchors: Set[BlockPos]) -> frozenset[BlockPos]:
    return frozenset(cells) - touching(cells, anchors)
