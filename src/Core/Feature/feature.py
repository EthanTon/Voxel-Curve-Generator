from typing import Protocol

from ..Field.block_field import BlockField
from ..modes import PlacementMode
from ..coordinates import BlockPos
from ..block.states import BlockState


class Feature(Protocol):
    @property
    def blocks(self) -> BlockField: ...

    @property
    def placement_mode(self) -> PlacementMode: ...

    @property
    def domain_id(self) -> int: ...

    @property
    def offset(self) -> BlockPos: ...

class Feature:
    def __init__(
        self,
        blocks: BlockField,
        placement_mode: PlacementMode,
        domain_id: int,
        offset: BlockPos | None = None,
    ) -> None:
        self.blocks = blocks
        self.placement_mode = placement_mode
        self.domain_id = domain_id
        self.offset = BlockPos(0, 0, 0) if offset is None else offset

class PointFeature(Feature):
    pass

class ContinuousFeature(Feature):
    # Features like wires and tracks.

    def add(self, block_position: BlockPos, block_state: BlockState) -> None:
        self.blocks.set_block(position=block_position, block=block_state)