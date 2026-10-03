from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from functools import cache

from ..Core.block.states import BlockState
from ..Core.constants import AIR_NAMES


class BlockClass(Enum):
    AIR = "air"
    GROUND = "ground"
    VEGETATION = "vegetation"
    FLUID = "fluid"
    STRUCTURE = "structure"


FLUID_NAMES = frozenset({"minecraft:water", "minecraft:lava", "minecraft:bubble_column"})

VEGETATION_SUFFIXES = (
    "_log", "_wood", "_leaves", "_sapling", "_mushroom", "_roots", "_fungus",
    "_bush", "_flower", "_propagule", "_sprouts", "_vines",
)
VEGETATION_NAMES = frozenset(
    "minecraft:" + each
    for each in (
        # "grass" is the plant's name before 1.20.3.
        "short_grass", "grass", "tall_grass", "fern", "large_fern", "dead_bush", "vine",
        "snow", "sugar_cane", "cactus", "bamboo", "lily_pad", "moss_carpet",
        "sweet_berry_bush", "cocoa", "mushroom_stem", "nether_wart", "seagrass",
        "tall_seagrass", "kelp", "kelp_plant", "hanging_roots", "glow_lichen",
        "dandelion", "poppy", "sunflower", "lilac", "rose_bush", "peony",
        "allium", "azure_bluet", "blue_orchid", "cornflower",
        "lily_of_the_valley", "oxeye_daisy", "wither_rose", "torchflower",
        "red_tulip", "orange_tulip", "white_tulip", "pink_tulip",
        "open_eyeblossom", "closed_eyeblossom", "wildflowers",
        "bush", "firefly_bush", "leaf_litter", "short_dry_grass", "tall_dry_grass",
        "cactus_flower", "pink_petals", "azalea", "flowering_azalea",
        "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "spore_blossom",
        "pitcher_plant", "pale_moss_carpet", "pale_hanging_moss", "sea_pickle",
        "cave_vines_plant", "pumpkin", "melon", "bee_nest",
    )
)

# Growth that belongs to trees; touching crowns do not share a trunk.
TREE_SUFFIXES = ("_log", "_wood", "_leaves")
TREE_NAMES = frozenset(
    "minecraft:" + each
    for each in (
        "vine", "cocoa", "mushroom_stem", "mangrove_roots", "mangrove_propagule",
        "bee_nest",
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
        "terracotta", "suspicious_gravel", "smooth_basalt", "budding_amethyst",
        "crimson_nylium", "warped_nylium", "sculk",
    )
)
# Natural ground in other namespaces' blocks (mods, data packs) is told by
# its name: one of these words, and none of a building block's.
GROUND_WORDS = (
    "dirt", "soil", "grass_block", "sand", "gravel", "clay", "mud", "silt", "loam", "peat",
    "stone", "rock", "slate", "granite", "diorite", "andesite", "marble", "limestone",
    "basalt", "shale", "chalk", "regolith", "permafrost", "snow", "ice", "ore",
    "terracotta", "tuff", "calcite",
)
BUILT_WORDS = (
    "brick", "slab", "stair", "wall", "fence", "door", "button", "plate", "tile", "polished",
    "chiseled", "cut_", "smooth", "pillar", "lamp", "glass", "pane", "carpet", "sign", "bars",
    "lantern", "torch", "chest", "table", "furnace", "rail",
)
# Blocks a build names as ground besides these, by name or glob pattern.
_GROUND: tuple[str, ...] = ()


def classify(state: BlockState) -> BlockClass:
    return _classify_name(state.name, _GROUND)


@contextmanager
def ground_blocks(patterns: Iterable[str]) -> Iterator[None]:
    """Count these blocks (names or glob patterns) as ground while inside."""
    global _GROUND
    held = _GROUND
    _GROUND = tuple(patterns)
    try:
        yield
    finally:
        _GROUND = held


@cache
def _classify_name(name: str, ground: tuple[str, ...] = ()) -> BlockClass:
    if name in AIR_NAMES:
        return BlockClass.AIR
    if name in FLUID_NAMES:
        return BlockClass.FLUID
    if ground and _matches(ground, name):
        return BlockClass.GROUND
    if name in VEGETATION_NAMES or name.endswith(VEGETATION_SUFFIXES):
        return BlockClass.VEGETATION
    if name in GROUND_NAMES or name.endswith(GROUND_SUFFIXES):
        return BlockClass.GROUND
    namespace, _, path = name.rpartition(":")
    if namespace == "minecraft":
        if name.endswith(("_ore", "_block")) or path.startswith("infested_"):
            return BlockClass.GROUND
    elif any(word in path for word in GROUND_WORDS) and not any(word in path for word in BUILT_WORDS):
        return BlockClass.GROUND
    return BlockClass.STRUCTURE


@dataclass(frozen=True, slots=True)
class BlockNames:
    # Block names, or glob patterns of them such as "minecraft:*sand".
    patterns: tuple[str, ...]

    def __call__(self, name: str) -> bool:
        return _matches(self.patterns, name)


@cache
def _matches(patterns: tuple[str, ...], name: str) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


# Ground that is soil rather than rock: it slopes, where anything else the
# land is made of -- stone, sandstone, terracotta, a custom hill's own
# blocks -- can stand as a face. Other namespaces' soils go by their names.
DEFAULT_LOOSE = BlockNames(tuple(
    "minecraft:" + each
    for each in (
        "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "grass_block",
        "moss_block", "pale_moss_block", "mud", "muddy_mangrove_roots", "clay",
        "dirt_path", "grass_path", "farmland", "gravel", "suspicious_gravel", "snow_block",
        "powder_snow", "soul_soil", "*sand", "*_concrete_powder",
    )
) + ("*:*dirt*", "*:*soil*", "*:*mud", "*:*clay", "*:*silt*", "*:*loam*", "*:*peat*",
     "*:*gravel*", "*:*sand"))
# Blocks that only ever top the ground: under water a fill shows what lies
# beneath them instead.
DEFAULT_TOPS = BlockNames(tuple(
    "minecraft:" + each
    for each in (
        "grass_block", "podzol", "mycelium", "moss_block", "pale_moss_block",
        "snow_block", "dirt_path", "grass_path", "farmland",
    )
))


def is_loose(name: str) -> bool:
    return DEFAULT_LOOSE(name)


def is_ground(state: BlockState) -> bool:
    return classify(state) is BlockClass.GROUND


def is_vegetation(state: BlockState) -> bool:
    return classify(state) is BlockClass.VEGETATION


def is_trunk(state: BlockState) -> bool:
    return state.name.endswith(("_log", "_wood")) or state.name in {
        "minecraft:mushroom_stem", "minecraft:mangrove_roots",
    }


def is_tree(state: BlockState) -> bool:
    return state.name in TREE_NAMES or state.name.endswith(TREE_SUFFIXES)
