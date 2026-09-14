from dataclasses import dataclass
from math import dist

from abc import ABC, abstractmethod

from typing import Protocol

from ..coordinates import BlockPos, Vec3

@dataclass(frozen=True, slots=True)
class FeaturePathPoint:
    stamp_index: int
    point_index: int
    position: BlockPos
    gradient: Vec3
    
    def __eq__(self, point: FeaturePathPoint = None) -> bool:
        if point is not None and point is FeaturePathPoint:
            if self.position == point.position and self.gradient == point.gradient:
                return True
        return False


class FeaturePath(Protocol):
    _path: list[FeaturePathPoint]
    
    # Append function that adds point to path
    def add(point: FeaturePathPoint) -> FeaturePathPoint:
        ...
    
    # Rule used to determine whether to add a point to the path
    def _rule(point: FeaturePathPoint) -> bool:
        ...