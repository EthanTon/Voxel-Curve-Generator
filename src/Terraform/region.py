from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..Core.block.states import BlockState
from ..Format.anvil import (
    CELLS_PER_SECTION,
    read_region_index,
    read_saved_chunk,
    region_of,
)
from ..Format import legacy
from ..Format.codec import unpack_long_array
from ..Format.nbt import block_state_from_tag, block_state_tag
from ..Format.region import SECTION_SIZE


HELD_CHUNKS = 16
HELD_DEPTH = 64
HELD_ENTRIES = 8192

ChunkKey = tuple[int, int]
ColumnKey = tuple[int, int]
Stack = list[tuple[int, BlockState]]
SectionBlocks = tuple[list[BlockState], np.ndarray]


# Generation steps after which a chunk's ground is all there, across the
# names versions since 1.13 have given them.
FINISHED = frozenset((
    "full", "minecraft:full", "features", "light", "spawn", "heightmaps", "decorated",
    "lighted", "mobs_spawned", "finalized", "fullchunk", "postprocessed",
))


def _complete(chunk) -> bool:
    # Before 1.18 a chunk's data sits under Level; before 1.13 a populated
    # flag stands in for its status. Synthetic sources may omit both.
    level = chunk.get("Level", chunk)
    status = level.get("Status")
    if status is not None:
        return status.py_str in FINISHED
    populated = level.get("TerrainPopulated")
    return populated is None or populated.py_int != 0


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
    saved_chunks: dict[ChunkKey, object] | None = field(default=None, init=False)
    _available: dict[ChunkKey, bool] = field(default_factory=dict, init=False)
    _read_to: dict[ColumnKey, int] = field(default_factory=dict, init=False)

    def include_depths(self, columns: Mapping[ColumnKey, int]) -> None:
        # Deep bores and footings need blocks below the usual surface cache.
        # Extend only those columns, keeping the surrounding survey shallow.
        self._read_to = dict(columns)
        self._held.clear()

    def capture(self) -> None:
        # The survey and the export must see the same saved chunk, even if
        # Minecraft saves again while the build is running.
        self.saved_chunks = {}
        self._held.clear()
        self._available.clear()

    @property
    def held_chunks(self) -> int:
        return len(self._held)

    @property
    def held_entries(self) -> int:
        return len(self._index)

    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        key = (chunk_x, chunk_z)
        if key not in self._available:
            entry = self._entry(*key)
            chunk = read_saved_chunk(entry[0], key) if entry is not None else None
            self._available[key] = chunk is not None and _complete(chunk)
            if self.saved_chunks is not None and self._available[key]:
                self.saved_chunks[key] = chunk
        return self._available[key]

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
        # Non-air cells in the surface cache or a requested deeper bore,
        # as stacks sorted bottom to top.
        if not self.generated(*key):
            return {}
        entry = self._entry(*key)
        if entry is None:
            return {}
        layers = []
        chunk = None if self.saved_chunks is None else self.saved_chunks.get(key)
        if chunk is None:
            chunk = read_saved_chunk(entry[0], key)
            if chunk is None:
                raise ValueError(f"Saved chunk {key} disappeared during the terrain survey; save the world and retry")
            if self.saved_chunks is not None:
                self.saved_chunks[key] = chunk
        for section_y, palette, indices in _sections(chunk):
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
        bottom = top - self.depth
        base_x, base_z = key[0] * SECTION_SIZE, key[1] * SECTION_SIZE
        for z in range(SECTION_SIZE):
            for x in range(SECTION_SIZE):
                required = self._read_to.get((base_x + x, base_z + z))
                if required is not None:
                    bottom[z, x] = min(bottom[z, x], required)
        keep = occupied & (levels[:, None, None] >= bottom[None])
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


# The first data version (1.16's 20w17a) whose packed entries never span two longs.
UNSPLIT = 2529


def _sections(chunk) -> Iterator[tuple[int, list, np.ndarray]]:
    # (section y, palette tags, palette indices) of each section with blocks,
    # from 1.18's sections, 1.13 to 1.17's Level sections or the numeric
    # blocks of a world saved before 1.13.
    version = chunk.get("DataVersion")
    version = 0 if version is None else version.py_int
    level = chunk.get("Level", chunk)
    for section in level.get("sections", level.get("Sections", ())):
        if "block_states" in section:
            tag = section["block_states"]
            palette, data, straddle = list(tag["palette"]), tag.get("data"), False
        elif "Palette" in section:
            palette, data, straddle = list(section["Palette"]), section.get("BlockStates"), version < UNSPLIT
        elif "Blocks" in section:
            palette, indices = _numbered(section)
            yield section["Y"].py_int, palette, indices
            continue
        else:
            continue
        if data is None or len(palette) == 1:
            indices = np.zeros(CELLS_PER_SECTION, dtype=np.int32)
        else:
            bits = max(4, (len(palette) - 1).bit_length())
            indices = unpack_long_array(
                data.np_array, bits, CELLS_PER_SECTION, straddle=straddle
            ).astype(np.int32)
        yield section["Y"].py_int, palette, indices


def _numbered(section) -> tuple[list, np.ndarray]:
    # A pre-1.13 section: block ids (with Add's high nibble) and data values,
    # read as a palette of modern states.
    blocks = np.asarray(section["Blocks"].np_array).astype(np.uint8).astype(np.int32)
    data = _nibbles(section.get("Data"))
    add = _nibbles(section.get("Add"))
    keys = (blocks | (add << 8)) * 16 + data
    unique, indices = np.unique(keys, return_inverse=True)
    palette = [block_state_tag(legacy.state(int(key) // 16, int(key) % 16)) for key in unique]
    return palette, indices.astype(np.int32)


def _nibbles(tag) -> np.ndarray:
    if tag is None:
        return np.zeros(CELLS_PER_SECTION, dtype=np.int32)
    packed = np.asarray(tag.np_array).astype(np.uint8)
    out = np.empty(packed.size * 2, dtype=np.int32)
    out[0::2] = packed & 15
    out[1::2] = packed >> 4
    return out


@dataclass(slots=True)
class WorldBlocks:
    # Every block of a saved world, a section at a time, for copying the
    # ground into an export.
    directory: Path
    chunks: int = 64
    snapshot: Mapping[ChunkKey, object] | None = None
    _index: dict[tuple[int, int], dict] = field(default_factory=dict, init=False)
    _held: OrderedDict[ChunkKey, dict[int, SectionBlocks]] = field(
        default_factory=OrderedDict, init=False
    )
    _available: dict[ChunkKey, bool] = field(default_factory=dict, init=False)

    def section(self, chunk_x: int, section_y: int, chunk_z: int) -> SectionBlocks | None:
        return self._sections(chunk_x, chunk_z).get(section_y)

    def generated(self, chunk_x: int, chunk_z: int) -> bool:
        key = (chunk_x, chunk_z)
        if key not in self._available:
            chunk = None if self.snapshot is None else self.snapshot.get(key)
            if chunk is None:
                entry = self._entry(key)
                chunk = read_saved_chunk(entry[0], key) if entry is not None else None
            self._available[key] = chunk is not None and _complete(chunk)
        return self._available[key]

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

    def _entry(self, key: ChunkKey):
        position = region_of(*key)
        entries = self._index.get((position.x, position.z))
        if entries is None:
            path = self.directory / position.file_name
            entries = (
                {chunk: (path, *place) for chunk, place in read_region_index(path, position).items()}
                if path.exists() else {}
            )
            self._index[(position.x, position.z)] = entries
        return entries.get(key)

    def _decode(self, key: ChunkKey) -> dict[int, SectionBlocks]:
        chunk = None if self.snapshot is None else self.snapshot.get(key)
        if chunk is None:
            entry = self._entry(key)
            if entry is None:
                return {}
            chunk = read_saved_chunk(entry[0], key)
            if chunk is None:
                raise ValueError(f"Saved chunk {key} disappeared while copying terrain; save the world and retry")
        if not _complete(chunk):
            return {}
        return {
            section_y: ([block_state_from_tag(each) for each in palette], indices)
            for section_y, palette, indices in _sections(chunk)
        }
