from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from math import asin, atan2, degrees, radians, sin
from typing import NamedTuple

from ..coordinates import Vec2, Vec3, quarter_turn

# Direction of travel in degrees clockwise from west, in steps of 45: 0 west
# (-x), 90 north (-z), 180 east (+x), 270 south (+z). Sections are authored
# at 0 with local x to the right of travel and forward along local -z.
Turn = int

# The side a path turned toward; counter-clockwise lays a turn out mirrored.
Rotation = int
CLOCKWISE: Rotation = 1
COUNTER_CLOCKWISE: Rotation = -1

# Headings closer than this to one line are straight on (or reversed).
STRAIGHT_TOLERANCE = 1e-9

def tolerance_for(degrees_off: float) -> float:
    # An angle as the chord between two unit headings that far apart.
    return 2 * sin(radians(degrees_off) / 2)


# A heading within this of a diagonal lays out diagonally: half the step
# between layouts, so the nearest of the eight is always used.
DIAGONAL_DEGREES = 22.5
DIAGONAL_TOLERANCE = tolerance_for(DIAGONAL_DEGREES)

# How near a diagonal a heading must be for a walk to lay a join there. Kept
# narrow so the join lands close to the true diagonal.
JOIN_DEGREES = 2.0
JOIN_TOLERANCE = tolerance_for(JOIN_DEGREES)

# Forward at each quarter: west (-1, 0) turned clockwise.
_FORWARD: dict[Turn, tuple[int, int]] = {
    quarter: quarter_turn(-1, 0, quarter // 90) for quarter in (0, 90, 180, 270)
}


def _quarter_layout(quarter: Turn) -> tuple[int, int, int, int]:
    # The authored section turned clockwise to `quarter`. Mirroring is
    # applied separately (see `orient_annotation`).
    fx, fz = _FORWARD[quarter]
    rx, rz = quarter_turn(fx, fz)
    return rx, -fx, rz, -fz


# Rows are (xx, xz, zx, zz):
#   world x = xx * x + xz * z
#   world z = zx * x + zz * z
_QUARTER_LAYOUTS: dict[Turn, tuple[int, int, int, int]] = {
    quarter: _quarter_layout(quarter) for quarter in _FORWARD
}

# Unit heading of every turn; a diagonal is halfway between its quarters.
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
    # The quarters either side of a diagonal, lower first.
    return diagonal - 45, (diagonal + 45) % 360


def place(quarter: Turn, x: int, z: int) -> tuple[int, int]:
    # A local column's world offset in a quarter layout.
    xx, xz, zx, zz = _QUARTER_LAYOUTS[quarter]
    return xx * x + xz * z, zx * x + zz * z


def quarter_of(turn: Turn) -> Turn:
    # A diagonal lays out in the quarter below it.
    turn = normalize(turn)
    return turn if is_quarter(turn) else neighbors(turn)[0]


def forward(quarter: Turn) -> tuple[int, int]:
    return _FORWARD[quarter]


def step_along(turn: Turn) -> tuple[int, int]:
    # One block forward in world axes; a diagonal steps along both.
    turn = normalize(turn)
    if is_quarter(turn):
        return _FORWARD[turn]
    lower, upper = neighbors(turn)
    return (
        _FORWARD[lower][0] + _FORWARD[upper][0],
        _FORWARD[lower][1] + _FORWARD[upper][1],
    )


def place_along(turn: Turn, x: int, z: int) -> tuple[int, int]:
    # A local column's world offset. On a diagonal each column steps one
    # block along both axes, so a row stays as many blocks as it is wide.
    turn = normalize(turn)
    if is_quarter(turn):
        return place(turn, x, z)
    forward_x, forward_z = step_along(turn)
    right_x, right_z = -forward_z, forward_x
    return right_x * x - forward_x * z, right_z * x - forward_z * z


def heading(turn: Turn) -> Vec2:
    return _HEADINGS[normalize(turn)]


def travel(turn: Turn, direction: Rotation = CLOCKWISE) -> Vec2:
    return heading(normalize(direction * turn))


def _horizontal(vector: Vec2 | Vec3 | Sequence[float]) -> Vec2:
    values = tuple(vector)
    flat = Vec2.coerce(values) if len(values) == 2 else Vec3.coerce(values).horizontal
    return flat.unit()


class SectionTurn(NamedTuple):
    # The layout at one path point, and the total angle turned to reach it.
    # West to north is (90, CLOCKWISE); west to south (90, COUNTER_CLOCKWISE).

    turn: Turn = 0
    direction: Rotation = CLOCKWISE
    angle: float = 0.0

    @property
    def absolute(self) -> Turn:
        return normalize(self.turn * self.direction)

    @property
    def travel(self) -> Vec2:
        return travel(self.turn, self.direction)


def snap(
    gradient: Vec3 | Sequence[float],
    tolerance: float = DIAGONAL_TOLERANCE,
) -> Turn:
    # A diagonal when within `tolerance` of one, else the nearest quarter.
    direction = Vec3.coerce(gradient).horizontal.unit()
    for turn in range(45, 360, 90):
        if direction.is_close(heading(turn), tolerance):
            return turn
    return max(_FORWARD, key=lambda turn: direction.dot(heading(turn)))


def angle_between(
    previous: Vec2 | Vec3 | Sequence[float],
    gradient: Vec2 | Vec3 | Sequence[float],
) -> float:
    # Signed degrees from one heading to the next; a reversal is +180.
    before, after = _horizontal(previous), _horizontal(gradient)
    return degrees(atan2(before.cross(after), before.dot(after)))


def sense(
    previous: Vec2 | Vec3 | Sequence[float],
    gradient: Vec2 | Vec3 | Sequence[float],
    previous_direction: Rotation = CLOCKWISE,
) -> Rotation:
    # The sign of the turn between two headings; along one line it keeps
    # `previous_direction`.
    angle = angle_between(previous, gradient)
    if min(abs(angle), abs(180 - abs(angle))) <= STRAIGHT_TOLERANCE:
        return previous_direction
    return CLOCKWISE if angle > 0 else COUNTER_CLOCKWISE


def _within(tolerance: float) -> float:
    # The inverse of `tolerance_for`.
    return degrees(2 * asin(min(1.0, tolerance / 2)))


def turn_at(angle: float, tolerance: float = DIAGONAL_TOLERANCE) -> SectionTurn:
    # The layout for a total turned angle: the nearest quarter unless within
    # `tolerance` of a diagonal. The angle is kept for the walk to add to.
    step = round(angle / 45)
    if step % 2 and abs(angle - 45 * step) <= _within(tolerance):
        turn = 45 * step
    else:
        turn = 90 * round(angle / 90)
    direction = CLOCKWISE if turn >= 0 else COUNTER_CLOCKWISE
    return SectionTurn(normalize(abs(turn)), direction, angle)


def turn_from(
    gradient: Vec3 | Sequence[float],
    initial: Rotation = CLOCKWISE,
    tolerance: float = DIAGONAL_TOLERANCE,
) -> SectionTurn:
    # Laid out at the snapped turn, but carrying the heading's own angle so
    # later bends are not measured from the snap.
    snapped = snap(gradient, tolerance)
    return turn_at(initial * snapped + angle_between(heading(snapped), gradient), tolerance)


def turn_of(
    gradient: Vec3 | Sequence[float],
    previous_gradient: Vec3 | Sequence[float],
    previous: SectionTurn = SectionTurn(),
    tolerance: float = DIAGONAL_TOLERANCE,
) -> SectionTurn:
    # Accumulates the raw angle so slight bends add up across points.
    return turn_at(
        previous.angle + angle_between(previous_gradient, gradient), tolerance
    )


def turns_along(
    gradients: Iterable[Vec3 | Sequence[float]],
    initial: Rotation = CLOCKWISE,
    tolerance: float = DIAGONAL_TOLERANCE,
) -> Iterator[SectionTurn]:
    # The layout at every point, carrying the accumulated angle along.
    walked: SectionTurn | None = None
    previous_gradient: Vec3 | Sequence[float] | None = None
    for gradient in gradients:
        walked = (
            turn_from(gradient, initial, tolerance)
            if walked is None or previous_gradient is None
            else turn_of(gradient, previous_gradient, walked, tolerance)
        )
        yield walked
        previous_gradient = gradient
