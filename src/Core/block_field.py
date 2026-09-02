from __future__ import annotations
from dataclasses import dataclass

from constants import DEFAULT_BLOCK

@dataclass
class BlockField:
    def __init__(self):
        self.field: dict[tuple[int, int, int], int] = {}

        self._palette: dict[str, int] = {
            DEFAULT_BLOCK: 0
        }

        self._palette_lookup: dict[int, str] = {
            0: DEFAULT_BLOCK
        }

        self._max_palette_number: int = 0

    def add_block(self, block: str) -> int:
        if block not in self._palette:
            self._max_palette_number += 1

            palette_number = self._max_palette_number

            self._palette[block] = palette_number
            self._palette_lookup[palette_number] = block

        return self._palette[block]

    def add_blocks(self, blocks: list[str]) -> None:
        for block in blocks:
            self.add_block(block)
    