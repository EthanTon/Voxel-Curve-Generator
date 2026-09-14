from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from ..coordinates import Vec2, Vec3

# Degrees clockwise from the path's initial heading, in steps of 45. Every
# cross section starts at a turn of 0, where it is used as authored: local x is
# lateral, and forward is local -z.
Turn = int

# How close a unit heading must be to a unit diagonal, as the distance between
# them, to turn 45 degrees off a quarter. Anything further snaps to the
# nearest quarter. 0.02 is a little over one degree.
DIAGONAL_TOLERANCE = 0.02

# Each quarter turn rotates the previous layout 90 degrees clockwise about the
# origin, then mirrors it across its long edge. The long edge starts on the x
# axis and alternates with every turn. Rows are (xx, xz, zx, zz):
#   world x = xx * x + xz * z
#   world z = zx * x + zz * z
_QUARTER_LAYOUTS: dict[Turn, tuple[int, int, int, int]] = {
    0: (1, 0, 0, 1),
    90: (0, 1, 1, 0),
    180: (-1, 0, 0, -1),
    270: (0, -1, -1, 0),
}

_FORWARD: dict[Turn, tuple[int, int]] = {
    0: (0, -1),
    90: (1, 0),
    180: (0, 1),
    270: (-1, 0),
}

_DIAGONAL = sqrt(0.5)
_DIAGONALS: dict[Turn, Vec2] = {
    45: Vec2(_DIAGONAL, -_DIAGONAL),
    135: Vec2(_DIAGONAL, _DIAGONAL),
    225: Vec2(-_DIAGONAL, _DIAGONAL),
    315: Vec2(-_DIAGONAL, -_DIAGONAL),
}


def normalize(turn: Turn) -> Turn:
    if turn % 45 != 0:
        raise ValueError(f"Turn {turn} is not a multiple of 45 degrees")
    return turn % 360


def is_quarter(turn: Turn) -> bool:
    return turn % 90 == 0


def neighbors(diagonal: Turn) -> tuple[Turn, Turn]:
    # The quarters on either side of a diagonal, lower first.
    return diagonal - 45, (diagonal + 45) % 360


def place(quarter: Turn, x: int, z: int) -> tuple[int, int]:
    # A local column's world offset in a quarter layout.
    xx, xz, zx, zz = _QUARTER_LAYOUTS[quarter]
    return xx * x + xz * z, zx * x + zz * z


def forward(quarter: Turn) -> tuple[int, int]:
    # One block forward, in world axes, in a quarter layout.
    return _FORWARD[quarter]


def turn_of(
    gradient: Vec3 | Sequence[float],
    initial_gradient: Vec3 | Sequence[float],
    tolerance: float = DIAGONAL_TOLERANCE,
) -> Turn:
    # The turn from the initial gradient's heading to this gradient's heading.
    heading = Vec3.coerce(gradient).horizontal.unit()
    initial = Vec3.coerce(initial_gradient).horizontal.unit()
    # Express the heading in the section's frame, where the initial heading is
    # forward (0, -1) and its right is (1, 0).
    right = Vec2(-initial.z, initial.x)
    relative = Vec2(heading.dot(right), -heading.dot(initial))
    for turn, diagonal in _DIAGONALS.items():
        if relative.is_close(diagonal, tolerance):
            return turn
    return max(_FORWARD, key=lambda turn: relative.dot(_FORWARD[turn]))
