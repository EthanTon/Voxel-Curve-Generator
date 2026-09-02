from __future__ import annotations
from dataclasses import dataclass

from constants import DEFAULT_BLOCK
from block_states import BlockState

@dataclass
class BlockField:
    def __init__(self):
        self.field: dict[tuple[int, int, int], int] = {}

        self._palette: dict[BlockState, int] = {
            BlockState(DEFAULT_BLOCK): 0
        }

        self._palette_lookup: dict[int, BlockState] = {
            0: BlockState(DEFAULT_BLOCK)
        }

        self._max_palette_number: int = 0

    def add_block(self, block: BlockState) -> int:
        if block not in self._palette:
            self._max_palette_number += 1

            palette_number = self._max_palette_number

            self._palette[block] = palette_number
            self._palette_lookup[palette_number] = block

        return self._palette[block]

    def add_blocks(self, blocks: list[BlockState]) -> None:
        for block in blocks:
            self.add_block(block)
    