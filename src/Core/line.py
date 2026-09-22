from __future__ import annotations

from collections.abc import Iterator, Sequence

from .coordinates import BlockPos
from .modes import LineMode


def line(
    start: BlockPos | Sequence[int],
    end: BlockPos | Sequence[int],
    mode: LineMode = LineMode.CHEBYSHEV,
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
    if mode == LineMode.CHEBYSHEV:
        return diagonal
    if mode == LineMode.MANHATTAN:
        return tuple(_face_connected(diagonal))
    raise ValueError(f"Unknown line mode: {mode}")


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
    # consecutive blocks share a face -- the stepped Bresenham the reference
    # draws a connected line with, which on a diagonal step emits the block
    # one along the first axis before the block that moved both.
    #
    # In plan only. A step that climbs keeps the climb on its last move
    # across, so that move is one along and one up: that is a slope, and it
    # is the only way a rail can climb. Split off as a move of its own the
    # climb puts a block in the corner instead, which a rail cannot express
    # -- it comes out flat and cuts into whatever it runs over.
    #
    # The two axes across go in their own order, x then z, rather than the
    # longer first. A fixed order makes the staircase lean the same way the
    # whole length of a run; picking per step by which axis has travelled
    # furthest makes it change sides partway along.
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
        position[1] = after[1]
        yield BlockPos(*position)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
