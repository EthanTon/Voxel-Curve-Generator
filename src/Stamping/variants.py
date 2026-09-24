from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from math import atan2, degrees
from typing import TypeVar

from ..Core.coordinates import Vec2, Vec3

# Degrees clockwise from west, like Turn but not limited to multiples of 45.
Heading = float

WEST = Vec2(-1.0, 0.0)

DEFAULT_RESOLUTION = 45.0

Variant = TypeVar("Variant")


def _horizontal(gradient: Vec2 | Vec3 | Sequence[float]) -> Vec2:
    values = tuple(gradient)
    flat = Vec2.coerce(values) if len(values) == 2 else Vec3.coerce(values).horizontal
    if flat.x == 0.0 and flat.z == 0.0:
        raise ValueError("Cannot take a heading from a zero gradient")
    return flat.unit()


def heading_of(gradient: Vec2 | Vec3 | Sequence[float]) -> Heading:
    flat = _horizontal(gradient)
    return degrees(atan2(WEST.cross(flat), WEST.dot(flat))) % 360.0


def steps_per_quarter(resolution: float = DEFAULT_RESOLUTION) -> int:
    # Variants turn by quarters, so a quarter must be whole steps.
    if resolution <= 0.0:
        raise ValueError("Resolution must be positive")
    steps = 90.0 / resolution
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"Resolution {resolution} does not divide a quarter turn")
    return round(steps)


def step_of(heading: Heading, resolution: float = DEFAULT_RESOLUTION) -> int:
    return round(heading / resolution) % (steps_per_quarter(resolution) * 4)


def snap(
    gradient: Vec2 | Vec3 | Sequence[float],
    resolution: float = DEFAULT_RESOLUTION,
) -> Heading:
    # The nearest heading at `resolution` (Core's snap only does 45s).
    return step_of(heading_of(gradient), resolution) * resolution


def authored_turn(
    heading: Heading, resolution: float = DEFAULT_RESOLUTION
) -> tuple[Heading, int]:
    # The first-quadrant heading and the quarter turns that reach `heading`.
    per_quarter = steps_per_quarter(resolution)
    quarter_turns, step = divmod(step_of(heading, resolution), per_quarter)
    return step * resolution, quarter_turns


def authored_headings(resolution: float = DEFAULT_RESOLUTION) -> tuple[Heading, ...]:
    # The first octant: the rest are turned or mirrored from it (0, 22.5, 45
    # at a resolution of 22.5).
    return tuple(
        step * resolution for step in range(mirrored_step(resolution) + 1)
    )


def mirrorable_headings(resolution: float = DEFAULT_RESOLUTION) -> tuple[Heading, ...]:
    # Headings past the diagonal, mirrored by default but authorable for
    # structures that differ left and right of travel.
    return tuple(
        step * resolution
        for step in range(mirrored_step(resolution) + 1, steps_per_quarter(resolution))
    )


def mirrored_step(resolution: float = DEFAULT_RESOLUTION) -> int:
    # The diagonal step that variants mirror about.
    return steps_per_quarter(resolution) // 2


def authored_for(
    step: int,
    resolution: float = DEFAULT_RESOLUTION,
    authored: Collection[int] = (),
) -> tuple[int, bool]:
    # The authored step a quadrant step comes from, and whether mirrored.
    per_quarter = steps_per_quarter(resolution)
    if step <= mirrored_step(resolution) or step in authored:
        return step, False
    return per_quarter - step, True


def by_authored(
    variants: Mapping[Heading, Variant],
    resolution: float = DEFAULT_RESOLUTION,
) -> dict[int, Variant]:
    # Variants keyed by step within the first quadrant; the first octant is
    # required.
    per_quarter = steps_per_quarter(resolution)
    found: dict[int, Variant] = {}
    for heading, variant in variants.items():
        step = step_of(heading, resolution)
        if step >= per_quarter:
            raise ValueError(
                f"Heading {heading} is not a variant: it is "
                f"{authored_turn(heading, resolution)[0]} turned round"
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
