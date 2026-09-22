from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from math import asin, atan2, degrees, radians, sin
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

def tolerance_for(degrees_off: float) -> float:
    # A tolerance is a distance between two unit headings, which is the chord
    # of the angle between them. This writes one as the angle it stands for.
    return 2 * sin(radians(degrees_off) / 2)


# How close a unit heading must be to a unit diagonal, as the distance between
# them, to lay out 45 degrees off a quarter. Anything further snaps to the
# nearest quarter.
#
# Half of the 45-degree step from one layout to the next, so a diagonal is
# taken whenever it is the nearest of the eight and no sooner. A smaller
# tolerance does not make the layout more careful, it makes it wrong: a path
# a few degrees off a diagonal is laid out at a quarter instead, 45 degrees
# from where it is going, and a curve spends almost all of its length there.
DIAGONAL_DEGREES = 22.5
DIAGONAL_TOLERANCE = tolerance_for(DIAGONAL_DEGREES)

# How near a diagonal a heading has to be for a walk to lay the join there.
#
# This is not the same question as which of the eight a heading is nearest.
# A diagonal is stamped once, where the run of headings near it begins, and
# from there on the quarter it hands over to is used -- so the wider the
# band, the earlier the handover and the further round the quarter is from
# where the path is actually going. A narrow band puts the join within a
# degree or two of the diagonal, which is where it belongs.
#
# It only has to be wide enough that a point lands inside it. A path that
# swings past a diagonal between two points has its join laid at the change
# instead, so the band being too narrow costs nothing.
JOIN_DEGREES = 2.0
JOIN_TOLERANCE = tolerance_for(JOIN_DEGREES)

# Forward at each quarter: west (-1, 0) turned clockwise.
_FORWARD: dict[Turn, tuple[int, int]] = {
    quarter: quarter_turn(-1, 0, quarter // 90) for quarter in (0, 90, 180, 270)
}


def _quarter_layout(quarter: Turn) -> tuple[int, int, int, int]:
    # The section turned that far clockwise about the origin, and nothing
    # else. Local x lands to the right of travel and local depth follows
    # travel.
    #
    # The mirror across the long edge is not here. Every layout carries one,
    # but the arm a join takes up carries the opposite one, and a matrix
    # fixed per quarter cannot be both -- so the mirror is a sign passed to
    # `orient_annotation`, which turns local depth and everything measured
    # along travel over together. Half of it here and half of it there is
    # what left the odd quarters with a mirrored profile and an unmirrored
    # staircase.
    fx, fz = _FORWARD[quarter]
    rx, rz = quarter_turn(fx, fz)
    return rx, -fx, rz, -fz


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


def quarter_of(turn: Turn) -> Turn:
    # The quarter a turn lays out in. A position placed by a section records
    # the arm of a join it landed in, so it is already a quarter; a diagonal
    # that reached here unplaced takes the quarter below it, which is what
    # its blocks take too.
    turn = normalize(turn)
    return turn if is_quarter(turn) else neighbors(turn)[0]


def forward(quarter: Turn) -> tuple[int, int]:
    # One block forward, in world axes, in a quarter layout.
    return _FORWARD[quarter]


def step_along(turn: Turn) -> tuple[int, int]:
    # One block forward in world axes, for a diagonal as well as a quarter.
    # A diagonal steps one on both axes, which is the sum of the quarters
    # either side of it.
    turn = normalize(turn)
    if is_quarter(turn):
        return _FORWARD[turn]
    lower, upper = neighbors(turn)
    return (
        _FORWARD[lower][0] + _FORWARD[upper][0],
        _FORWARD[lower][1] + _FORWARD[upper][1],
    )


def place_along(turn: Turn, x: int, z: int) -> tuple[int, int]:
    # A local column's world offset in any layout, quarter or diagonal.
    #
    # A quarter has a layout of its own. A diagonal lays the same row out
    # along its own axes: local x runs to the right of travel, one block on
    # each axis per column, and local z runs back along travel. An eleven
    # wide row is then eleven blocks laid corner to corner, which is what it
    # is at a quarter too -- not the two arms of an L, which is what joining
    # the quarters either side of it produces and is nearly twice the deck.
    turn = normalize(turn)
    if is_quarter(turn):
        return place(turn, x, z)
    forward_x, forward_z = step_along(turn)
    right_x, right_z = -forward_z, forward_x
    return right_x * x - forward_x * z, right_z * x - forward_z * z


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
