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
    # How a cross-section point steps along the path. MANHATTAN also delivers
    # a block along travel and pushes columns outboard of it, keeping its run
    # four-connected. See CrossSection/steps.py.
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class LineMode(Enum):
    # How a straight run is drawn. CHEBYSHEV steps diagonally in 3D;
    # MANHATTAN is face-connected in 3D (iron bars); RAIL is face-connected
    # in plan with climbs on horizontal moves, as a rail slope needs.
    CHEBYSHEV = "chebyshev"
    MANHATTAN = "manhattan"
    RAIL = "rail"
