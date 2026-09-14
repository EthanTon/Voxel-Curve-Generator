from enum import Enum

from ..Field.block_field import BlockField
from ..coordinates import BlockPos
from .annotations import PointAnnotation

LocalPoint = BlockPos

class StepMode(Enum):
    # Manhattan / face-connected
    MANHATTAN = "manhattan"
    
    # Chebyshev / diagonal allowed
    CHEBYSHEV = "chebyshev"

class SectionStep:
    at: LocalPoint
    mode: StepMode

class CrossSection:
    field: BlockField
    origin: LocalPoint
    steps: tuple[SectionStep, ...]
    annotations: tuple[PointAnnotation, ...]
    
