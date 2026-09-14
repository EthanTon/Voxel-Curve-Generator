from dataclasses import dataclass
from ..coordinates import BlockPos

@dataclass(frozen=True, slots=True)
class GroupAnnotation:
    name: str
    cells: frozenset[BlockPos]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cells",
            frozenset(BlockPos.coerce(cell) for cell in self.cells),
        )


@dataclass(frozen=True, slots=True)
class PointAnnotation:
    name: str
    position: BlockPos

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", BlockPos.coerce(self.position))


@dataclass(frozen=True, slots=True)
class FeaturePathAnnotation:
    name: str
    point: str
    group: str

class Annotation:
    def __init__(self, position: BlockPos):
        self._position = BlockPos.coerce(position)
    
