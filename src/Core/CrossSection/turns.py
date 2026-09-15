from __future__ import annotations

from ..coordinates import Vec2, quarter_turn

# The world direction of travel, in degrees clockwise from west, in steps of
# 45: 0 is west (-x), 90 north (-z), 180 east (+x) and 270 south (+z). Every
# cross section is authored for a turn of 0: local x is lateral, positive to
# the right of travel, and forward is local -z.
Turn = int

# Forward at each quarter: west (-1, 0) turned clockwise.
_FORWARD: dict[Turn, tuple[int, int]] = {
    quarter: quarter_turn(-1, 0, quarter // 90) for quarter in (0, 90, 180, 270)
}


def _quarter_layout(quarter: Turn) -> tuple[int, int, int, int]:
    # Each quarter turn rotates the previous layout 90 degrees clockwise about
    # the origin, then mirrors it across its long edge. Local x always lands to
    # the right of travel; local depth follows travel at even quarters and is
    # mirrored at odd ones.
    fx, fz = _FORWARD[quarter]
    rx, rz = quarter_turn(fx, fz)
    depth = -1 if quarter // 90 % 2 == 0 else 1
    return rx, depth * fx, rz, depth * fz


# Rows are (xx, xz, zx, zz):
#   world x = xx * x + xz * z
#   world z = zx * x + zz * z
_QUARTER_LAYOUTS: dict[Turn, tuple[int, int, int, int]] = {
    quarter: _quarter_layout(quarter) for quarter in _FORWARD
}

# The unit heading of every turn, in world axes: a quarter's forward, or the
# heading halfway between the quarters either side of a diagonal.
_HEADINGS: dict[Turn, Vec2] = {
    **{quarter: Vec2(*step) for quarter, step in _FORWARD.items()},
    **{
        quarter + 45: Vec2(
            _FORWARD[quarter][0] + _FORWARD[(quarter + 90) % 360][0],
            _FORWARD[quarter][1] + _FORWARD[(quarter + 90) % 360][1],
        ).unit()
        for quarter in _FORWARD
    },
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


def heading(turn: Turn) -> Vec2:
    # The unit direction of travel, in world axes, that a turn lays out for.
    return _HEADINGS[normalize(turn)]
