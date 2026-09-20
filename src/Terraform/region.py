from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..Core.block.states import BlockState
from ..Format.anvil import (
    CELLS_PER_SECTION,
    decode_block_states,
    read_region_chunk,
    read_region_index,
    region_of,
)
from ..Format.region import SECTION_SIZE


HELD_CHUNKS = 16
HELD_DEPTH = 32
HELD_ENTRIES = 8192

ChunkKey = tuple[int, int]
ColumnKey = tuple[int, int]
Stack = list[tuple[int, BlockState]]


@dataclass(slots=True)
class RegionTerrain:
    directory: Path
    chunks: int = HELD_CHUNKS
    depth: int = HELD_DEPTH
    entries: int = HELD_ENTRIES
    _index: OrderedDict[ChunkKey, tuple[Path, int, int] | None] = field(
        default_factory=OrderedDict, init=False
    )
    _held: OrderedDict[ChunkKey, dict[ColumnKey, Stack]] = field(
        default_factory=OrderedDict, init=False
    )

    @property
    def held_chunks(self) -> int:
        return len(self._held)

    @property
    def held_entries(self) -> int:
        return len(self._index)

    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        return self._entry(chunk_x, chunk_z) is not None

    def column(self, x: int, z: int) -> Sequence[tuple[int, BlockState]]:
        return self._columns(x >> 4, z >> 4).get((x, z), ())

    def _columns(self, chunk_x: int, chunk_z: int) -> dict[ColumnKey, Stack]:
        key = (chunk_x, chunk_z)
        held = self._held.get(key)
        if held is None:
            held = self._decode(key)
            self._held[key] = held
            while len(self._held) > self.chunks:
                self._held.popitem(last=False)
        else:
            self._held.move_to_end(key)
        return held

    def _decode(self, key: ChunkKey) -> dict[ColumnKey, Stack]:
        entry = self._entry(*key)
        if entry is None:
            return {}
        chunk = read_region_chunk(*entry)
        gathered: dict[ColumnKey, Stack] = {}
        for section in chunk.get("sections", ()):
            if "block_states" not in section:
                continue
            states, indices = decode_block_states(section["block_states"])
            base = section["Y"].py_int * SECTION_SIZE
            for index in range(CELLS_PER_SECTION):
                state = states[indices[index]]
                if state.name == "minecraft:air":
                    continue
                column = (
                    key[0] * SECTION_SIZE + (index & 0xF),
                    key[1] * SECTION_SIZE + ((index >> 4) & 0xF),
                )
                gathered.setdefault(column, []).append((base + (index >> 8), state))
        return {column: self._topmost(stack) for column, stack in gathered.items()}

    def _topmost(self, stack: Stack) -> Stack:
        stack.sort()
        floor = stack[-1][0] - self.depth
        kept = [each for each in stack if each[0] >= floor]
        return kept

    def _entry(self, chunk_x: int, chunk_z: int) -> tuple[Path, int, int] | None:
        key = (chunk_x, chunk_z)
        if key in self._index:
            self._index.move_to_end(key)
            return self._index[key]
        self._read_index(chunk_x, chunk_z)
        return self._index.get(key)

    def _read_index(self, chunk_x: int, chunk_z: int) -> None:
        position = region_of(chunk_x, chunk_z)
        path = self.directory / position.file_name
        found = read_region_index(path, position) if path.exists() else {}
        for key, (offset, sectors) in found.items():
            self._index[key] = (path, offset, sectors)
        self._index.setdefault((chunk_x, chunk_z), None)
        while len(self._index) > self.entries:
            self._index.popitem(last=False)
