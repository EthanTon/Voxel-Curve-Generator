from .modes import PlacementMode, TerrainMaskMode

DEFAULT_BLOCK = "minecraft:air"
DEFAULT_PLACEMENT_MODE = PlacementMode.OVERWRITE
DEFAULT_TERRAIN_MASK_MODE = TerrainMaskMode.NONE

CARDINAL_DIRECTIONS = (
    "north",
    "east",
    "south",
    "west",
)

# Which cardinal a mirror swaps for which, by the world axis it flips. A
# mirror across the plane x = 0 turns east into west and leaves north and
# south where they are; across z = 0, the other way round.
MIRRORED_DIRECTIONS = {
    "x": {"east": "west", "west": "east"},
    "z": {"north": "south", "south": "north"},
}

# A rail's shape under the same two mirrors, for the shapes a mirror moves.
MIRRORED_RAIL_SHAPES = {
    "x": {
        "ascending_east": "ascending_west",
        "ascending_west": "ascending_east",
        "north_east": "north_west",
        "north_west": "north_east",
        "south_east": "south_west",
        "south_west": "south_east",
    },
    "z": {
        "ascending_north": "ascending_south",
        "ascending_south": "ascending_north",
        "north_east": "south_east",
        "south_east": "north_east",
        "north_west": "south_west",
        "south_west": "north_west",
    },
}

RAIL_ROTATION = {
    "north_south": "east_west",
    "east_west": "north_south",

    "ascending_north": "ascending_east",
    "ascending_east": "ascending_south",
    "ascending_south": "ascending_west",
    "ascending_west": "ascending_north",

    "north_east": "south_east",
    "south_east": "south_west",
    "south_west": "north_west",
    "north_west": "north_east",
}

ROTATABLE_PROPERTIES = {
    "axis",
    "facing",
    "rotation",
    "orientation",
}
