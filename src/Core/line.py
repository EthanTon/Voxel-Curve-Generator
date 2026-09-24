from __future__ import annotations

from collections.abc import Iterator, Sequence

from .coordinates import BlockPos
from .modes import LineMode


def line(
    start: BlockPos | Sequence[int],
    end: BlockPos | Sequence[int],
    mode: LineMode = LineMode.CHEBYSHEV,
) -> tuple[BlockPos, ...]:
    # The blocks from `start` to `end` inclusive (see LineMode). Drawn from
    # the lesser end so both directions give the same blocks.
    start, end = BlockPos.coerce(start), BlockPos.coerce(end)
    if end < start:
        return line(end, start, mode)[::-1]
    diagonal = tuple(_walk(start, end))
    if mode == LineMode.CHEBYSHEV:
        return diagonal
    if mode == LineMode.MANHATTAN:
        return tuple(_face_connected(diagonal, split_vertical=True))
    if mode == LineMode.RAIL:
        return tuple(_face_connected(diagonal))
    raise ValueError(f"Unknown line mode: {mode}")


def _walk(start: BlockPos, end: BlockPos) -> Iterator[BlockPos]:
    # 3D Bresenham driven by the longest axis.
    deltas = [end[axis] - start[axis] for axis in range(3)]
    steps = [_sign(delta) for delta in deltas]
    extents = [abs(delta) for delta in deltas]
    driver = extents.index(max(extents))
    length = extents[driver]

    yield start
    if length == 0:
        return

    position = list(start)
    error = [2 * extents[axis] - length for axis in range(3)]
    for _ in range(length):
        for axis in range(3):
            if axis == driver:
                continue
            if error[axis] > 0:
                position[axis] += steps[axis]
                error[axis] -= 2 * length
            error[axis] += 2 * extents[axis]
        position[driver] += steps[driver]
        yield BlockPos(*position)


def _face_connected(
    diagonal: Sequence[BlockPos], *, split_vertical: bool = False
) -> Iterator[BlockPos]:
    # Splits each diagonal step into moves along x then z (a fixed order, so
    # the staircase leans one way). A climb rides the last horizontal move,
    # as a rail slope needs, unless `split_vertical`.
    if not diagonal:
        return
    yield diagonal[0]
    for before, after in zip(diagonal, diagonal[1:]):
        position = list(before)
        across = [axis for axis in (0, 2) if after[axis] != position[axis]]
        for axis in across[:-1]:
            position[axis] = after[axis]
            yield BlockPos(*position)
        if across:
            position[across[-1]] = after[across[-1]]
        if split_vertical and across and after.y != position[1]:
            yield BlockPos(*position)
        position[1] = after[1]
        yield BlockPos(*position)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
