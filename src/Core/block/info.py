from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag, auto


class RotationRule(IntFlag):
    """Independent transformations that may apply to a block state."""

    NONE = 0
    AXIS = 1 << 0
    FACING = 1 << 1
    CARDINAL = 1 << 2
    ROTATION_16 = 1 << 3
    RAIL_SHAPE = 1 << 4
    ORIENTATION = 1 << 5


class NeighborRule(Enum):
    """The kind of state that must be derived from neighboring blocks."""

    NONE = 0
    FENCE = auto()
    PANE = auto()
    WALL = auto()
    FENCE_GATE = auto()
    STAIR = auto()
    RAIL = auto()
    REDSTONE = auto()
    TRIPWIRE = auto()
    CHORUS = auto()
    MULTIFACE = auto()
    VINE = auto()
    FIRE = auto()
    MOSS_CARPET = auto()
    SCAFFOLDING = auto()
    SHELF = auto()
    POINTED_DRIPSTONE = auto()
    LEAVES = auto()
    CHEST = auto()
    STEM = auto()
    HANGING = auto()
    SUPPORT = auto()
    MULTIBLOCK = auto()
    PROPAGATED = auto()
    HUGE_MUSHROOM = auto()
    SNOWY = auto()
    GROWING_PLANT = auto()
    HANGING_MOSS = auto()
    CAMPFIRE = auto()
    TRIPWIRE_HOOK = auto()


class SolidFace(IntFlag):
    """Faces that neighboring blocks may attach to."""

    NONE = 0
    DOWN = 1 << 0
    UP = 1 << 1
    NORTH = 1 << 2
    EAST = 1 << 3
    SOUTH = 1 << 4
    WEST = 1 << 5
    ALL = DOWN | UP | NORTH | EAST | SOUTH | WEST


@dataclass(frozen=True, slots=True)
class BlockInfo:
    """Static generation behavior shared by variants of a block."""

    rotation: RotationRule = RotationRule.NONE
    neighbor_rule: NeighborRule = NeighborRule.NONE
    solid_faces: SolidFace = SolidFace.NONE
    replaceable: bool = False
    behavior: str = "block"
    connection_exception: bool = False
    flammable: bool = False
    tags: frozenset[str] = frozenset()
    center_faces: SolidFace = SolidFace.NONE
    rigid_faces: SolidFace = SolidFace.NONE
    signal_source: bool = False
    wall_cover: int = 0


DEFAULT_BLOCK_INFO = BlockInfo()
