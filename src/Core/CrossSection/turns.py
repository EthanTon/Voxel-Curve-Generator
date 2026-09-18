from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from math import asin, atan2, degrees
from typing import NamedTuple

from ..coordinates import Vec2, Vec3, quarter_turn

# The world direction of travel, in degrees clockwise from west, in steps of
# 45: 0 is west (-x), 90 north (-z), 180 east (+x) and 270 south (+z). Every
# cross section is authored for a turn of 0: local x is lateral, positive to
# the right of travel, and forward is local -z.
Turn = int

# Which way the path was turning when it reached a turn: the sign of the cross
# product of the heading before it with the heading at it. A section is
# authored for a positive one; a negative one lays the same turn out mirrored
# about the authored heading, travelling the other way round.
Rotation = int
CLOCKWISE: Rotation = 1
COUNTER_CLOCKWISE: Rotation = -1

# An angle no larger than this is two headings along the same line, not a
# turn: the path either ran straight on or doubled back.
STRAIGHT_TOLERANCE = 1e-9

# How close a unit heading must be to a unit diagonal, as the distance between
# them, to lay out 45 degrees off a quarter. Anything further snaps to the
# nearest quarter. 0.02 is a little over one degree.
DIAGONAL_TOLERANCE = 0.02

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


def travel(turn: Turn, direction: Rotation = CLOCKWISE) -> Vec2:
    # The unit direction of travel a turn lays out for, counted `direction`.
    # The same turn counted counter-clockwise travels the other way round,
    # which is what mirrors its layout about the authored heading.
    return heading(normalize(direction * turn))


def _horizontal(vector: Vec2 | Vec3 | Sequence[float]) -> Vec2:
    # Either a heading or a gradient, flattened and sized to one.
    values = tuple(vector)
    flat = Vec2.coerce(values) if len(values) == 2 else Vec3.coerce(values).horizontal
    return flat.unit()


class SectionTurn(NamedTuple):
    # What a section is laid out with at one path point, and the angle the
    # path had turned by to get there. The angle is what a walk carries from
    # point to point; the turn and the direction are that angle snapped to a
    # layout, which is what `CrossSection.orient` takes. West to north is
    # (90, CLOCKWISE) and west to south is (90, COUNTER_CLOCKWISE) -- the same
    # turn, counted the other way, travelling the other way.

    turn: Turn = 0
    direction: Rotation = CLOCKWISE
    angle: float = 0.0

    @property
    def absolute(self) -> Turn:
        # The world direction of travel, in degrees clockwise from west.
        return normalize(self.turn * self.direction)

    @property
    def travel(self) -> Vec2:
        return travel(self.turn, self.direction)


def snap(
    gradient: Vec3 | Sequence[float],
    tolerance: float = DIAGONAL_TOLERANCE,
) -> Turn:
    # The world direction a gradient points along, as a turn clockwise from
    # west: a diagonal only when the heading is within `tolerance` of one, and
    # the nearest quarter otherwise.
    direction = Vec3.coerce(gradient).horizontal.unit()
    for turn in range(45, 360, 90):
        if direction.is_close(heading(turn), tolerance):
            return turn
    return max(_FORWARD, key=lambda turn: direction.dot(heading(turn)))


def angle_between(
    previous: Vec2 | Vec3 | Sequence[float],
    gradient: Vec2 | Vec3 | Sequence[float],
) -> float:
    # The angle in degrees from one heading to the next, signed the way a turn
    # is: the cross product is its sine and the dot product its cosine, so the
    # two together are the whole rotation rather than just its side. A
    # reversal has no side and comes back as half a turn clockwise.
    before, after = _horizontal(previous), _horizontal(gradient)
    return degrees(atan2(before.cross(after), before.dot(after)))


def sense(
    previous: Vec2 | Vec3 | Sequence[float],
    gradient: Vec2 | Vec3 | Sequence[float],
    previous_direction: Rotation = CLOCKWISE,
) -> Rotation:
    # Which way a path turned between two headings, on their own: the sign of
    # the angle between them. Headings along one line turn by nothing at all,
    # whether the path ran straight on or doubled back, and keep the direction
    # it was already turning. A walk takes its direction from the angle it has
    # carried instead, which is what `turns_along` does.
    angle = angle_between(previous, gradient)
    if min(abs(angle), abs(180 - abs(angle))) <= STRAIGHT_TOLERANCE:
        return previous_direction
    return CLOCKWISE if angle > 0 else COUNTER_CLOCKWISE


def _within(tolerance: float) -> float:
    # The vector tolerance as the angle it stands for: the distance between two
    # unit headings is the chord of the angle between them.
    return degrees(2 * asin(min(1.0, tolerance / 2)))


def turn_at(angle: float, tolerance: float = DIAGONAL_TOLERANCE) -> SectionTurn:
    # An angle a path has turned by, as the layout it lays its section out at:
    # the nearest quarter unless the angle is within `tolerance` of a diagonal,
    # which is the bias `snap` takes on a single heading. The angle is kept as
    # it is, so a walk goes on adding to it.
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
    # Where a walk starts. A heading on its own has not turned from anything,
    # so it takes the turn it snaps to, counted `initial`.
    return turn_at(float(initial * snap(gradient, tolerance)), tolerance)


def turn_of(
    gradient: Vec3 | Sequence[float],
    previous_gradient: Vec3 | Sequence[float],
    previous: SectionTurn = SectionTurn(),
    tolerance: float = DIAGONAL_TOLERANCE,
) -> SectionTurn:
    # One step of a walk: the angle carried so far, plus the angle turned by
    # between the two gradients. The gradients are taken as they are rather
    # than as the turns they snap to, so a bend too slight to be a turn of its
    # own is still carried and adds up with the ones after it.
    return turn_at(
        previous.angle + angle_between(previous_gradient, gradient), tolerance
    )


def turns_along(
    gradients: Iterable[Vec3 | Sequence[float]],
    initial: Rotation = CLOCKWISE,
    tolerance: float = DIAGONAL_TOLERANCE,
) -> Iterator[SectionTurn]:
    # The turn at every point of a path. Carrying the angle from point to
    # point rather than deciding a direction at each one is what makes the
    # trajectory decide the layout: a run that holds its heading adds nothing
    # and keeps the turn it had, and a wobble inside a curve moves the angle a
    # little instead of turning the section round.
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
