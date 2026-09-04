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
