from block_field import BlockField
from modes import PlacementMode, TerrainMaskMode
from constants import DEFAULT_BLOCK, DEFAULT_PLACEMENT_MODE, DEFAULT_TERRAIN_MASK_MODE

from dataclasses import dataclass

@dataclass
class BlockPlacement:
    field: BlockField
    offset: tuple[int, int, int]
    placement_mode: PlacementMode
    terrain_mask_mode: TerrainMaskMode
    replace: str
    


class BlockWorld:
    def __init__(self):
        self.instances: list[BlockPlacement] = []

    def add_field(
        self,
        field: BlockField,
        offset: tuple[int, int, int] = (0, 0, 0),
        placement_mode: PlacementMode = DEFAULT_PLACEMENT_MODE,
        terrain_mask_mode: TerrainMaskMode = DEFAULT_TERRAIN_MASK_MODE,
        replace: str = DEFAULT_BLOCK,
    ) -> None:
        self.instances.append(
            BlockPlacement(
                field=field,
                offset=offset,
                placement_mode=placement_mode,
                terrain_mask_mode=terrain_mask_mode,
                replace=replace,
            )
        )
    