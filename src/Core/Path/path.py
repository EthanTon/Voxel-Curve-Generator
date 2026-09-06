from collections.abc import Sequence
from dataclasses import astuple, dataclass
from math import dist

from ..coordinates import BlockPos, Vec3

@dataclass(frozen=True, slots=True)
class PathPoint:
    position: BlockPos
    gradient: Vec3

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
