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
        # plan view, and the normal of the boundary plane at this point.
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
    def __init__(
        self,
        points: Sequence[BlockPos | Sequence[int]],
        gradients: Sequence[Vec3 | Sequence[float]],
    ) -> None:
        if len(points) != len(gradients):
            raise ValueError(
                "Points and gradients must have the same length. "
                "Ensure that each point has an associated gradient."
            )

        self.path_points: list[PathPoint] = [
            PathPoint(
                position=BlockPos.coerce(p),
                gradient=Vec3.coerce(g),
            )
            for p, g in zip(points, gradients)
        ]
        
        
    def length(self) -> float:
        if len(self.path_points) <= 1:
            return 0.0

        distance = 0.0
        for i in range(1, len(self.path_points)):
            prev = self.path_points[i - 1].position
            curr = self.path_points[i].position
            
            distance += dist(astuple(prev), astuple(curr))

        return distance
