from __future__ import annotations

from Core.block.states import BlockState
from Core.block.palette import BlockPalette, BlockId

BlockPos = tuple[int, int, int]

class BlockField:
    def __init__(
        self,
        palette: BlockPalette | None = None,
    ):
        self._field: dict[BlockPos, BlockId] = {}
        self._palette = palette or BlockPalette()

    @property
    def palette(self) -> BlockPalette:
        return self._palette

    #Set a block at a given position
    def set_block(
        self,
        position: BlockPos,
        block: BlockState,
    ) -> None:
        block_id = self._palette.add(block)
        self._field[position] = block_id

    # Set BlockId at a given location
    def set_block_id(
        self,
        position: BlockPos,
        block_id: BlockId,
    ) -> None:
        # Validate that the ID exists.
        self._palette.get_state(block_id)

        self._field[position] = block_id

    # Get the BlockState of a given position
    def get_block(
        self,
        position: BlockPos,
    ) -> BlockState | None:
        block_id = self._field.get(position)

        if block_id is None:
            return None

        return self._palette.get_state(block_id)

    # Get BlockId from a given location
    def get_block_id(
        self,
        position: BlockPos,
    ) -> BlockId | None:
        return self._field.get(position)

    # Remove a block from the field
    def remove_block(
        self,
        position: BlockPos,
    ) -> None:
        self._field.pop(position, None)

    # Check if there exist a block at given location
    def has_block(
        self,
        position: BlockPos,
    ) -> bool:
        return position in self._field

    def __contains__(
        self,
        position: BlockPos,
    ) -> bool:
        return position in self._field

    def __len__(self) -> int:
        return len(self._field)
