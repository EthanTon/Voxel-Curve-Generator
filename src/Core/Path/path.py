from collections.abc import Sequence
from dataclasses import astuple, dataclass
from math import dist, hypot

from ..coordinates import BlockPos, Vec2, Vec3

@dataclass(frozen=True, slots=True)
class PathPoint:
    position: BlockPos
    gradient: Vec3

    @property
    def heading(self) -> Vec2:
        # Horizontal projection of the gradient: the direction of travel in
        # plan view.
        return Vec2(self.gradient.x, self.gradient.z)

    @property
    def heading_magnitude(self) -> float:
        return hypot(self.gradient.x, self.gradient.z)

    @property
    def grade(self) -> float:
        # Rise over horizontal run. Undefined where the gradient is vertical.
        run = self.heading_magnitude
        if run == 0.0:
            raise ValueError(
                f"PathPoint at {self.position} has a purely vertical gradient; "
                "grade is undefined"
            )
        return self.gradient.y / run

    @property
    def is_vertical(self) -> bool:
        return self.heading_magnitude == 0.0

class Path:
    # Points along a curve, optionally padded at either end.
    #
    # The first `lead_in` and last `lead_out` points lie before and after the
    # path proper. They take part in anything computed along the path --
    # geometry, elevation -- so the boundaries at the path's ends are held as
    # cleanly as any boundary in the middle. They are not part of
    # the path: whatever they produce past those boundaries is dropped.
    def __init__(
        self,
        points: Sequence[BlockPos | Sequence[int]],
        gradients: Sequence[Vec3 | Sequence[float]],
        *,
        lead_in: int = 0,
        lead_out: int = 0,
    ) -> None:
        if len(points) != len(gradients):
            raise ValueError(
                "Points and gradients must have the same length. "
                "Ensure that each point has an associated gradient."
            )
        if lead_in < 0 or lead_out < 0:
            raise ValueError("lead_in and lead_out must be non-negative")
        if (lead_in or lead_out) and lead_in + lead_out >= len(points):
            raise ValueError(
                f"Padding of {lead_in} + {lead_out} points leaves none of the "
                f"{len(points)} points on the path"
            )

        self.path_points: list[PathPoint] = [
            PathPoint(
                position=BlockPos.coerce(p),
                gradient=Vec3.coerce(g),
            )
            for p, g in zip(points, gradients)
        ]
        self.lead_in = lead_in
        self.lead_out = lead_out

    @property
    def first_index(self) -> int:
        # Index of the first point on the path proper.
        return self.lead_in

    @property
    def last_index(self) -> int:
        # Index of the last point on the path proper (inclusive).
        return len(self.path_points) - 1 - self.lead_out

    @property
    def bounded_points(self) -> list[PathPoint]:
        return self.path_points[self.first_index : self.last_index + 1]

    def is_padding(self, index: int) -> bool:
        return index < self.first_index or index > self.last_index

    def length(self) -> float:
        # Length of the path proper; padding is not measured.
        points = self.bounded_points
        if len(points) <= 1:
            return 0.0

        distance = 0.0
        for i in range(1, len(points)):
            prev = points[i - 1].position
            curr = points[i].position

            distance += dist(astuple(prev), astuple(curr))

        return distance
