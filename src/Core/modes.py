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
