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
    # How a point of a cross section reaches the next point along the path.
    #
    # CHEBYSHEV delivers it where it sits and nowhere else. MANHATTAN also
    # delivers it a block along travel and pushes everything outboard of it
    # by the same, so the run it makes down the path is four-connected and
    # the row it sits in becomes a staircase cornered on it. See
    # CrossSection/steps.py.
    #
    # This is not LineMode. It is about one point of a section reaching the
    # next point of the path, and it moves the points around it; LineMode is
    # about two positions already decided and what goes between them.
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class LineMode(Enum):
    # How a straight run between two positions is drawn.
    #
    # CHEBYSHEV steps diagonally, one block per block of the longest axis, so
    # the run is as short as it can be. MANHATTAN steps one axis at a time,
    # so every block shares a face with the next. See Core/line.py.
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"