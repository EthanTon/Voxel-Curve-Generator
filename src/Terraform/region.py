from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..Core.block.states import BlockState
from ..Format.anvil import CELLS_PER_SECTION, RegionFile, decode_block_states, region_of
from ..Format.region import SECTION_SIZE


@dataclass(slots=True)
class RegionTerrain:
    directory: Path
    _files: dict[tuple[int, int], RegionFile | None] = field(
        default_factory=dict, init=False
    )
    _loaded: set[tuple[int, int]] = field(default_factory=set, init=False)
    _columns: dict[tuple[int, int], list[tuple[int, BlockState]]] = field(
        default_factory=dict, init=False
    )

    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        held = self._region(chunk_x, chunk_z)
        return held is not None and (chunk_x, chunk_z) in held.chunks

    def column(self, x: int, z: int) -> Sequence[tuple[int, BlockState]]:
        chunk = (x >> 4, z >> 4)
        if chunk not in self._loaded:
            self._read(*chunk)
        return self._columns.get((x, z), ())

    def _region(self, chunk_x: int, chunk_z: int) -> RegionFile | None:
        position = region_of(chunk_x, chunk_z)
        key = (position.x, position.z)
        if key not in self._files:
            path = self.directory / position.file_name
            self._files[key] = RegionFile.load(path, position) if path.exists() else None
        return self._files[key]

    def _read(self, chunk_x: int, chunk_z: int) -> None:
        self._loaded.add((chunk_x, chunk_z))
        held = self._region(chunk_x, chunk_z)
        chunk = None if held is None else held.chunks.get((chunk_x, chunk_z))
        if chunk is None:
            return
        gathered: dict[tuple[int, int], list[tuple[int, BlockState]]] = {}
        for section in chunk.get("sections", ()):
            if "block_states" not in section:
                continue
            states, indices = decode_block_states(section["block_states"])
            base = section["Y"].py_int * SECTION_SIZE
            for index in range(CELLS_PER_SECTION):
                state = states[indices[index]]
                if state.name == "minecraft:air":
                    continue
                key = (
                    chunk_x * SECTION_SIZE + (index & 0xF),
                    chunk_z * SECTION_SIZE + ((index >> 4) & 0xF),
                )
                gathered.setdefault(key, []).append((base + (index >> 8), state))
        for key, stack in gathered.items():
            stack.sort()
            self._columns[key] = stack
