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


def reaching(
    within: Set[BlockPos], sources: Iterable[BlockPos],
    steps: Iterable[BlockPos] = FACES,
) -> frozenset[BlockPos]:
    steps = tuple(steps)
    found = {each for each in sources if each in within}
    pending = list(found)
    while pending:
        position = pending.pop()
        for step in steps:
            neighbour = position + step
            if neighbour in within and neighbour not in found:
                found.add(neighbour)
                pending.append(neighbour)
    return frozenset(found)


def components(
    cells: Set[BlockPos], steps: Iterable[BlockPos] = FACES,
) -> Iterator[frozenset[BlockPos]]:
    steps = tuple(steps)
    remaining = set(cells)
    while remaining:
        found = reaching(remaining, [next(iter(remaining))], steps)
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
