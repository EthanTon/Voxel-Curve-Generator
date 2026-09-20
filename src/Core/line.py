from __future__ import annotations

from collections.abc import Iterator, Sequence

from .coordinates import BlockPos
from .modes import StepMode


def line(
    start: BlockPos | Sequence[int],
    end: BlockPos | Sequence[int],
    mode: StepMode = StepMode.CHEBYSHEV,
) -> tuple[BlockPos, ...]:
    # A straight run of blocks from one position to another, both included.
    #
    # CHEBYSHEV steps diagonally: one step per block of the longest axis, so
    # the run is as short as it can be and blocks meet at edges and corners.
    # MANHATTAN steps on one axis at a time, so every block shares a face with
    # the next -- what a wire needs to stay connected.
    #
    # The ends are put in a fixed order first. Bresenham breaks its ties on
    # the side it started from, so without that the same two ends would give
    # different blocks depending on which way the run was drawn.
    start, end = BlockPos.coerce(start), BlockPos.coerce(end)
    if end < start:
        return line(end, start, mode)[::-1]
    diagonal = tuple(_walk(start, end))
    if mode == StepMode.CHEBYSHEV:
        return diagonal
    if mode == StepMode.MANHATTAN:
        return tuple(_face_connected(diagonal))
    raise ValueError(f"Unknown step mode: {mode}")


def _walk(start: BlockPos, end: BlockPos) -> Iterator[BlockPos]:
    # Bresenham in three dimensions, driven by the longest axis: each step
    # advances it by one and the other two whenever their error has built up
    # enough to owe a block.
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


def _face_connected(diagonal: Sequence[BlockPos]) -> Iterator[BlockPos]:
    # The same run with every diagonal step broken into single-axis moves, so
    # consecutive blocks always share a face. The longest axis moves first,
    # which keeps the run against the line it is drawn along.
    if not diagonal:
        return
    yield diagonal[0]
    for before, after in zip(diagonal, diagonal[1:]):
        moves = [axis for axis in range(3) if after[axis] != before[axis]]
        moves.sort(key=lambda axis: -abs(after[axis] - diagonal[0][axis]))
        position = list(before)
        for axis in moves:
            position[axis] = after[axis]
            yield BlockPos(*position)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
