from __future__ import annotations

from enum import Enum

from ..Core.block.states import BlockState


class BlockClass(Enum):
    AIR = "air"
    GROUND = "ground"
    VEGETATION = "vegetation"
    FLUID = "fluid"
    STRUCTURE = "structure"


AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})
FLUID_NAMES = frozenset({"minecraft:water", "minecraft:lava", "minecraft:bubble_column"})

VEGETATION_SUFFIXES = (
    "_log", "_wood", "_leaves", "_sapling", "_mushroom", "_roots", "_fungus",
    "_bush", "_flower", "_propagule", "_sprouts", "_vines",
)
VEGETATION_NAMES = frozenset(
    "minecraft:" + each
    for each in (
        "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "vine",
        "snow", "sugar_cane", "cactus", "bamboo", "lily_pad", "moss_carpet",
        "sweet_berry_bush", "cocoa", "mushroom_stem", "nether_wart", "seagrass",
        "tall_seagrass", "kelp", "kelp_plant", "hanging_roots", "glow_lichen",
        "dandelion", "poppy", "sunflower", "lilac", "rose_bush", "peony",
    )
)

GROUND_SUFFIXES = (
    "_terracotta", "_concrete_powder", "_sand", "_sandstone", "_ice",
)
GROUND_NAMES = frozenset(
    "minecraft:" + each
    for each in (
        "stone", "granite", "diorite", "andesite", "deepslate", "tuff", "calcite",
        "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "grass_block",
        "gravel", "sand", "clay", "mud", "sandstone", "netherrack", "end_stone",
        "basalt", "blackstone", "magma_block", "soul_sand", "soul_soil", "bedrock",
        "packed_ice", "blue_ice", "snow_block", "powder_snow", "moss_block",
        "cobblestone", "mossy_cobblestone", "obsidian", "ice", "dripstone_block",
    )
)


def classify(state: BlockState) -> BlockClass:
    name = state.name
    if name in AIR_NAMES:
        return BlockClass.AIR
    if name in FLUID_NAMES:
        return BlockClass.FLUID
    if name in VEGETATION_NAMES or name.endswith(VEGETATION_SUFFIXES):
        return BlockClass.VEGETATION
    if name in GROUND_NAMES or name.endswith(GROUND_SUFFIXES):
        return BlockClass.GROUND
    if name.endswith(("_ore", "_block")) and "minecraft:" in name:
        return BlockClass.GROUND
    return BlockClass.STRUCTURE


def is_ground(state: BlockState) -> bool:
    return classify(state) is BlockClass.GROUND


def is_vegetation(state: BlockState) -> bool:
    return classify(state) is BlockClass.VEGETATION
