"""Block names for worlds saved before 1.13, by numeric id and data value.

Covers the vanilla 1.12 blocks with the data values that change what a block
is (stone kinds, wood kinds, colours); a variant this table does not name
reads as the id's first name, and an id it does not know at all as
`minecraft:legacy_<id>`, which classifies as a structure.
"""
from __future__ import annotations

from functools import cache

from ..Core.block.states import BlockState

COLOURS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
           "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")
WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak")


def _coloured(suffix: str) -> tuple[str, ...]:
    return tuple(f"{colour}_{suffix}" for colour in COLOURS)


def _wooden(suffix: str) -> tuple[str, ...]:
    return tuple(f"{wood}_{suffix}" for wood in WOODS)


# id: one name, or names by data value (the data value's low bits, see KIND).
NAMES: dict[int, str | tuple[str, ...]] = {
    0: "air",
    1: ("stone", "granite", "polished_granite", "diorite", "polished_diorite", "andesite",
        "polished_andesite"),
    2: "grass_block",
    3: ("dirt", "coarse_dirt", "podzol"),
    4: "cobblestone",
    5: _wooden("planks"),
    6: _wooden("sapling"),
    7: "bedrock",
    8: "water", 9: "water", 10: "lava", 11: "lava",
    12: ("sand", "red_sand"),
    13: "gravel", 14: "gold_ore", 15: "iron_ore", 16: "coal_ore",
    17: ("oak_log", "spruce_log", "birch_log", "jungle_log"),
    18: ("oak_leaves", "spruce_leaves", "birch_leaves", "jungle_leaves"),
    19: ("sponge", "wet_sponge"),
    20: "glass", 21: "lapis_ore", 22: "lapis_block", 23: "dispenser",
    24: ("sandstone", "chiseled_sandstone", "cut_sandstone"),
    25: "note_block", 26: "red_bed", 27: "powered_rail", 28: "detector_rail",
    29: "sticky_piston", 30: "cobweb",
    31: ("dead_bush", "grass", "fern"),
    32: "dead_bush", 33: "piston", 34: "piston_head",
    35: _coloured("wool"),
    36: "moving_piston", 37: "dandelion",
    38: ("poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip",
         "white_tulip", "pink_tulip", "oxeye_daisy"),
    39: "brown_mushroom", 40: "red_mushroom", 41: "gold_block", 42: "iron_block",
    43: ("smooth_stone_slab", "sandstone_slab", "petrified_oak_slab", "cobblestone_slab",
         "brick_slab", "stone_brick_slab", "nether_brick_slab", "quartz_slab"),
    44: ("smooth_stone_slab", "sandstone_slab", "petrified_oak_slab", "cobblestone_slab",
         "brick_slab", "stone_brick_slab", "nether_brick_slab", "quartz_slab"),
    45: "bricks", 46: "tnt", 47: "bookshelf", 48: "mossy_cobblestone", 49: "obsidian",
    50: "torch", 51: "fire", 52: "spawner", 53: "oak_stairs", 54: "chest",
    55: "redstone_wire", 56: "diamond_ore", 57: "diamond_block", 58: "crafting_table",
    59: "wheat", 60: "farmland", 61: "furnace", 62: "furnace", 63: "oak_sign",
    64: "oak_door", 65: "ladder", 66: "rail", 67: "cobblestone_stairs", 68: "oak_wall_sign",
    69: "lever", 70: "stone_pressure_plate", 71: "iron_door", 72: "oak_pressure_plate",
    73: "redstone_ore", 74: "redstone_ore", 75: "redstone_torch", 76: "redstone_torch",
    77: "stone_button", 78: "snow", 79: "ice", 80: "snow_block", 81: "cactus", 82: "clay",
    83: "sugar_cane", 84: "jukebox", 85: "oak_fence", 86: "pumpkin", 87: "netherrack",
    88: "soul_sand", 89: "glowstone", 90: "nether_portal", 91: "jack_o_lantern", 92: "cake",
    93: "repeater", 94: "repeater",
    95: _coloured("stained_glass"),
    96: "oak_trapdoor",
    97: ("infested_stone", "infested_cobblestone", "infested_stone_bricks",
         "infested_mossy_stone_bricks", "infested_cracked_stone_bricks",
         "infested_chiseled_stone_bricks"),
    98: ("stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks"),
    99: "brown_mushroom_block", 100: "red_mushroom_block", 101: "iron_bars",
    102: "glass_pane", 103: "melon", 104: "pumpkin_stem", 105: "melon_stem", 106: "vine",
    107: "oak_fence_gate", 108: "brick_stairs", 109: "stone_brick_stairs", 110: "mycelium",
    111: "lily_pad", 112: "nether_bricks", 113: "nether_brick_fence",
    114: "nether_brick_stairs", 115: "nether_wart", 116: "enchanting_table",
    117: "brewing_stand", 118: "cauldron", 119: "end_portal", 120: "end_portal_frame",
    121: "end_stone", 122: "dragon_egg", 123: "redstone_lamp", 124: "redstone_lamp",
    125: _wooden("slab"), 126: _wooden("slab"),
    127: "cocoa", 128: "sandstone_stairs", 129: "emerald_ore", 130: "ender_chest",
    131: "tripwire_hook", 132: "tripwire", 133: "emerald_block", 134: "spruce_stairs",
    135: "birch_stairs", 136: "jungle_stairs", 137: "command_block", 138: "beacon",
    139: ("cobblestone_wall", "mossy_cobblestone_wall"),
    140: "flower_pot", 141: "carrots", 142: "potatoes", 143: "oak_button",
    144: "skeleton_skull", 145: "anvil", 146: "trapped_chest",
    147: "light_weighted_pressure_plate", 148: "heavy_weighted_pressure_plate",
    149: "comparator", 150: "comparator", 151: "daylight_detector", 152: "redstone_block",
    153: "nether_quartz_ore", 154: "hopper",
    155: ("quartz_block", "chiseled_quartz_block", "quartz_pillar"),
    156: "quartz_stairs", 157: "activator_rail", 158: "dropper",
    159: _coloured("terracotta"),
    160: _coloured("stained_glass_pane"),
    161: ("acacia_leaves", "dark_oak_leaves"),
    162: ("acacia_log", "dark_oak_log"),
    163: "acacia_stairs", 164: "dark_oak_stairs", 165: "slime_block", 166: "barrier",
    167: "iron_trapdoor",
    168: ("prismarine", "prismarine_bricks", "dark_prismarine"),
    169: "sea_lantern", 170: "hay_block",
    171: _coloured("carpet"),
    172: "terracotta", 173: "coal_block", 174: "packed_ice",
    175: ("sunflower", "lilac", "tall_grass", "large_fern", "rose_bush", "peony"),
    176: "white_banner", 177: "white_wall_banner", 178: "daylight_detector",
    179: ("red_sandstone", "chiseled_red_sandstone", "cut_red_sandstone"),
    180: "red_sandstone_stairs", 181: "red_sandstone_slab", 182: "red_sandstone_slab",
    183: "spruce_fence_gate", 184: "birch_fence_gate", 185: "jungle_fence_gate",
    186: "dark_oak_fence_gate", 187: "acacia_fence_gate",
    188: "spruce_fence", 189: "birch_fence", 190: "jungle_fence", 191: "dark_oak_fence",
    192: "acacia_fence", 193: "spruce_door", 194: "birch_door", 195: "jungle_door",
    196: "acacia_door", 197: "dark_oak_door", 198: "end_rod", 199: "chorus_plant",
    200: "chorus_flower", 201: "purpur_block", 202: "purpur_pillar", 203: "purpur_stairs",
    204: "purpur_slab", 205: "purpur_slab", 206: "end_stone_bricks", 207: "beetroots",
    208: "dirt_path", 209: "end_gateway", 210: "repeating_command_block",
    211: "chain_command_block", 212: "frosted_ice", 213: "magma_block",
    214: "nether_wart_block", 215: "red_nether_bricks", 216: "bone_block",
    217: "structure_void", 218: "observer",
    **{219 + index: f"{colour}_shulker_box" for index, colour in enumerate(COLOURS)},
    **{235 + index: f"{colour}_glazed_terracotta" for index, colour in enumerate(COLOURS)},
    251: _coloured("concrete"),
    252: _coloured("concrete_powder"),
    255: "structure_block",
}
# How many low bits of the data value pick the kind, where the rest is
# orientation, growth or half.
KIND = {6: 3, 17: 2, 18: 2, 43: 3, 44: 3, 125: 3, 126: 3, 161: 2, 162: 2, 175: 3}
AXES = ("y", "x", "z")


@cache
def state(block: int, data: int) -> BlockState:
    """The modern block state for a numeric id and data value."""
    names = NAMES.get(block)
    if names is None:
        return BlockState.create(f"minecraft:legacy_{block}")
    if isinstance(names, str):
        name = names
    else:
        kind = data & ((1 << KIND[block]) - 1) if block in KIND else data
        if block == 155:
            kind = min(kind, 2)
        name = names[kind] if kind < len(names) else names[0]
    if block in (99, 100) and data in (10, 15):
        name = "mushroom_stem"
    properties: dict[str, str] = {}
    if block in (17, 162):
        axis = (data >> 2) & 3
        if axis == 3:
            name = name.replace("_log", "_wood")
        else:
            properties["axis"] = AXES[axis]
    elif block in (8, 9, 10, 11):
        properties["level"] = str(data & 15)
    elif block == 78:
        properties["layers"] = str((data & 7) + 1)
    elif block in (44, 126, 182, 205):
        properties["type"] = "top" if data & 8 else "bottom"
    elif block in (43, 125, 181, 204):
        properties["type"] = "double"
    elif block == 175 and data & 8:
        # The upper half carries no kind; it reads as tall grass.
        name, properties = "tall_grass", {"half": "upper"}
    elif block == 175:
        properties["half"] = "lower"
    return BlockState.create(f"minecraft:{name}", properties)
