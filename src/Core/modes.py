from enum import Enum

class PlacementMode(Enum):
    OVERWRITE = 1
    KEEP_EXISTING = 2
    REPLACE_AIR_ONLY = 3
    REPLACE_MATCHING = 4
    TERRAIN_MASK = 5

class TerrainMaskMode(Enum):
    NONE = 0
    ABOVE = 1
    BELOW = 2
    SURFACE = 3
    FOLLOW_SURFACE = 4


class ElevationMode(Enum):
    DEFORMABLE = 1
    RIGID = 2
    INTERPOLATED = 3

class StepMode(Enum):
    # Manhattan / face-connected
    MANHATTAN = "manhattan"
    
    # Chebyshev / diagonal allowed
    CHEBYSHEV = "chebyshev"