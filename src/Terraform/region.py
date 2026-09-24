from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..Core.block.states import BlockState
from ..Format.anvil import (
    CELLS_PER_SECTION,
    read_region_chunk,
    read_region_index,
    region_of,
)
from ..Format.codec import unpack_long_array
from ..Format.nbt import block_state_from_tag
from ..Format.region import SECTION_SIZE


HELD_CHUNKS = 16
HELD_DEPTH = 32
HELD_ENTRIES = 8192

ChunkKey = tuple[int, int]
ColumnKey = tuple[int, int]
Stack = list[tuple[int, BlockState]]
SectionBlocks = tuple[list[BlockState], np.ndarray]


@dataclass(slots=True)
class RegionTerrain:
    # The top `depth` levels of each column of a saved world, a chunk at a time.
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
        # Every non-air cell within `depth` of its column's top, as stacks
        # sorted bottom to top.
        entry = self._entry(*key)
        if entry is None:
            return {}
        layers = []
        for section_y, palette, indices in _sections(read_region_chunk(*entry)):
            solid = np.array([each["Name"].py_str != "minecraft:air" for each in palette])
            if solid.any():
                layers.append((section_y * SECTION_SIZE, palette, indices.reshape(16, 16, 16), solid))
        if not layers:
            return {}
        layers.sort(key=lambda layer: layer[0])
        occupied = np.concatenate([solid[indices] for _, _, indices, solid in layers])
        levels = np.concatenate([np.arange(base, base + SECTION_SIZE) for base, _, _, _ in layers])
        filled = occupied.any(axis=0)
        top = np.where(filled, levels[occupied.shape[0] - 1 - np.argmax(occupied[::-1], axis=0)], 0)
        keep = occupied & (levels[:, None, None] >= (top - self.depth)[None])
        level, z, x = np.nonzero(keep)
        if not level.size:
            return {}

        order = np.lexsort((level, x, z))
        level, z, x = level[order], z[order], x[order]
        palette = [tag for _, tags, _, _ in layers for tag in tags]
        offsets = np.cumsum([0] + [len(tags) for _, tags, _, _ in layers[:-1]])
        ids = np.concatenate([indices for _, _, indices, _ in layers])[level, z, x]
        ids = (ids + offsets[level // SECTION_SIZE]).tolist()
        states = {each: block_state_from_tag(palette[each]) for each in set(ids)}
        heights = levels[level].tolist()
        found = [states[each] for each in ids]
        xs, zs = x.tolist(), z.tolist()
        starts = np.flatnonzero(
            np.concatenate(([True], (z[1:] != z[:-1]) | (x[1:] != x[:-1])))
        ).tolist()
        base_x, base_z = key[0] * SECTION_SIZE, key[1] * SECTION_SIZE
        return {
            (base_x + xs[begin], base_z + zs[begin]): list(zip(heights[begin:end], found[begin:end]))
            for begin, end in zip(starts, starts[1:] + [len(ids)])
        }

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


def _sections(chunk) -> Iterator[tuple[int, list, np.ndarray]]:
    # (section y, palette tags, palette indices) of each section with blocks.
    for section in chunk.get("sections", ()):
        if "block_states" not in section:
            continue
        tag = section["block_states"]
        palette = list(tag["palette"])
        data = tag.get("data")
        if data is None or len(palette) == 1:
            indices = np.zeros(CELLS_PER_SECTION, dtype=np.int32)
        else:
            bits = max(4, (len(palette) - 1).bit_length())
            indices = unpack_long_array(
                data.np_array, bits, CELLS_PER_SECTION, straddle=False
            ).astype(np.int32)
        yield section["Y"].py_int, palette, indices


@dataclass(slots=True)
class WorldBlocks:
    # Every block of a saved world, a section at a time, for copying the
    # ground into an export.
    directory: Path
    chunks: int = 64
    _index: dict[tuple[int, int], dict] = field(default_factory=dict, init=False)
    _held: OrderedDict[ChunkKey, dict[int, SectionBlocks]] = field(
        default_factory=OrderedDict, init=False
    )

    def section(self, chunk_x: int, section_y: int, chunk_z: int) -> SectionBlocks | None:
        return self._sections(chunk_x, chunk_z).get(section_y)

    def _sections(self, chunk_x: int, chunk_z: int) -> dict[int, SectionBlocks]:
        key = (chunk_x, chunk_z)
        held = self._held.get(key)
        if held is not None:
            self._held.move_to_end(key)
            return held
        held = self._decode(key)
        self._held[key] = held
        while len(self._held) > self.chunks:
            self._held.popitem(last=False)
        return held

    def _decode(self, key: ChunkKey) -> dict[int, SectionBlocks]:
        position = region_of(*key)
        entries = self._index.get((position.x, position.z))
        if entries is None:
            path = self.directory / position.file_name
            entries = (
                {chunk: (path, *place) for chunk, place in read_region_index(path, position).items()}
                if path.exists() else {}
            )
            self._index[(position.x, position.z)] = entries
        entry = entries.get(key)
        if entry is None:
            return {}
        return {
            section_y: ([block_state_from_tag(each) for each in palette], indices)
            for section_y, palette, indices in _sections(read_region_chunk(*entry))
        }
