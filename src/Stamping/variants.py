from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, degrees
from typing import TypeVar

from ..Core.coordinates import Vec2, Vec3

# Direction of travel in degrees clockwise from west, as Core's Turn counts
# it: 0 west (-x), 90 north (-z), 180 east (+x), 270 south (+z).
#
# Not limited to multiples of 45. A Turn is, because a diagonal section is
# composed from the quarter layouts either side of it. An authored variant is
# never composed.
Heading = float

# The heading variants are authored along, before any quarter turn.
WEST = Vec2(-1.0, 0.0)

# How far apart authored variants sit.
DEFAULT_RESOLUTION = 45.0

Variant = TypeVar("Variant")


def _horizontal(gradient: Vec2 | Vec3 | Sequence[float]) -> Vec2:
    values = tuple(gradient)
    flat = Vec2.coerce(values) if len(values) == 2 else Vec3.coerce(values).horizontal
    if flat.x == 0.0 and flat.z == 0.0:
        raise ValueError("Cannot take a heading from a zero gradient")
    return flat.unit()


def heading_of(gradient: Vec2 | Vec3 | Sequence[float]) -> Heading:
    # A gradient as the heading it travels, measured off west. Vertical
    # component ignored.
    flat = _horizontal(gradient)
    return degrees(atan2(WEST.cross(flat), WEST.dot(flat))) % 360.0


def steps_per_quarter(resolution: float = DEFAULT_RESOLUTION) -> int:
    # Variants are carried round by whole quarter turns, so a quarter must be
    # a whole number of steps or the authored headings do not line up with
    # the ones they turn to.
    if resolution <= 0.0:
        raise ValueError("Resolution must be positive")
    steps = 90.0 / resolution
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"Resolution {resolution} does not divide a quarter turn")
    return round(steps)


def step_of(heading: Heading, resolution: float = DEFAULT_RESOLUTION) -> int:
    # A heading as its index in the ring of steps. Whole steps rather than
    # degrees, so 67.5 never drifts off itself.
    return round(heading / resolution) % (steps_per_quarter(resolution) * 4)


def snap(
    gradient: Vec2 | Vec3 | Sequence[float],
    resolution: float = DEFAULT_RESOLUTION,
) -> Heading:
    # The authored heading a gradient is stamped at. Stamping's own snap;
    # Core's only answers in multiples of 45.
    return step_of(heading_of(gradient), resolution) * resolution


def authored_turn(
    heading: Heading, resolution: float = DEFAULT_RESOLUTION
) -> tuple[Heading, int]:
    # The variant a heading is stamped from, and the quarter turns to get
    # there. A heading is stamped from wherever it sits in its own quadrant.
    per_quarter = steps_per_quarter(resolution)
    quarter_turns, step = divmod(step_of(heading, resolution), per_quarter)
    return step * resolution, quarter_turns


def authored_headings(resolution: float = DEFAULT_RESOLUTION) -> tuple[Heading, ...]:
    # Every heading a set must be authored for: 0 and 45 at a resolution of
    # 45; 0, 22.5 and 45 at 22.5.
    #
    # The first octant, not the first quadrant. Rotation carries a variant
    # into the other three quadrants, and a mirror carries it across the
    # diagonal to the other half of its own -- 67.5 is 22.5 mirrored, and
    # authoring it as well is authoring the same shape twice.
    return tuple(
        step * resolution for step in range(mirrored_step(resolution) + 1)
    )


def mirrored_step(resolution: float = DEFAULT_RESOLUTION) -> int:
    # The step a quadrant's variants are mirrored about: the diagonal. Steps
    # past it are the ones below it seen the other way round.
    return steps_per_quarter(resolution) // 2


def authored_for(step: int, resolution: float = DEFAULT_RESOLUTION) -> tuple[int, bool]:
    # The variant a step within a quadrant is stamped from, and whether it is
    # mirrored to get there. Steps up to the diagonal are authored; the rest
    # are their reflections, as far past it as the authored one is short of
    # it.
    per_quarter = steps_per_quarter(resolution)
    if step <= mirrored_step(resolution):
        return step, False
    return per_quarter - step, True


def by_authored(
    variants: Mapping[Heading, Variant],
    resolution: float = DEFAULT_RESOLUTION,
) -> dict[int, Variant]:
    # Authored variants indexed by step within a quarter, one required each.
    # A heading past the first quadrant is refused: it is another one turned
    # round, and one past the diagonal is refused too, since it is another
    # one mirrored. Either way authoring both leaves two answers for one
    # shape.
    per_quarter = steps_per_quarter(resolution)
    found: dict[int, Variant] = {}
    for heading, variant in variants.items():
        step = step_of(heading, resolution)
        if step >= per_quarter:
            raise ValueError(
                f"Heading {heading} is not a variant: it is "
                f"{authored_turn(heading, resolution)[0]} turned round"
            )
        if step > mirrored_step(resolution):
            raise ValueError(
                f"Heading {heading} is not a variant: it is "
                f"{(per_quarter - step) * resolution} mirrored"
            )
        found[step] = variant
    missing = [
        heading
        for step, heading in enumerate(authored_headings(resolution))
        if step not in found
    ]
    if missing:
        raise ValueError(f"No variant authored for headings {missing}")
    return found
