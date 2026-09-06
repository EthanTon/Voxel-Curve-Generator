from dataclasses import dataclass
from math import dist

from ..coordinates import BlockPos

@dataclass(frozen=True, slots=True)
class FeaturePathPoint:
    stamp_index: int
    point_index: int
    position: BlockPos
    distance: float


class FeaturePath:
    def __init__(self, name):
        self.name = name
        self._points = []

    @property
    def points(self):
        return tuple(self._points)

    def append(self, positions: list[BlockPos]):
        positions = tuple(BlockPos.coerce(position) for position in positions)
        
        if not positions:
            raise ValueError("Position must include at least one point")
        
        for position in positions:
            if self._points:
                if self._points[-1].position != position:
                    self._points.append(
                        Feat
                    )
        
