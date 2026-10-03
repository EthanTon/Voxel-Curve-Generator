from __future__ import annotations

import struct
import time
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..Core.block.states import BlockState
from . import nbt
from .codec import bits_for, pack_long_array, unpack_longs
from .region import SECTION_SIZE, BlockRegion

SECTOR = 4096
REGION_CHUNKS = 32
HEADER_SECTORS = 2
CELLS_PER_SECTION = SECTION_SIZE**3
# At least 4 bits per entry; since 1.16 entries never span two longs.
MIN_BITS = 4
STRADDLE = False
GZIP, ZLIB, RAW = 1, 2, 3
DEFAULT_BIOME = "minecraft:plains"
DEFAULT_STATUS = "minecraft:full"


@dataclass(frozen=True, slots=True)
class RegionPos:
    # One .mca file: 32x32 chunk columns, unbounded in y.
    x: int
    z: int

    @property
    def file_name(self) -> str:
        return f"r.{self.x}.{self.z}.mca"


def region_of(chunk_x: int, chunk_z: int) -> RegionPos:
    return RegionPos(chunk_x >> 5, chunk_z >> 5)


def header_index(chunk_x: int, chunk_z: int) -> int:
    return (chunk_x & 31) + (chunk_z & 31) * REGION_CHUNKS


@dataclass(slots=True)
class ChunkColumn:
    # Sections sharing an (x, z), keyed by (possibly negative) section y.
    chunk_x: int
    chunk_z: int
    sections: dict[int, BlockRegion] = field(default_factory=dict)

    @property
    def position(self) -> RegionPos:
        return region_of(self.chunk_x, self.chunk_z)

    def iter_sections(self) -> Iterator[tuple[int, BlockRegion]]:
        for section_y in sorted(self.sections):
            yield section_y, self.sections[section_y]


def chunk_columns(region: BlockRegion) -> dict[tuple[int, int], ChunkColumn]:
    columns: dict[tuple[int, int], ChunkColumn] = {}
    for chunk, section in region.iter_sections():
        column = columns.setdefault((chunk.x, chunk.z), ChunkColumn(chunk.x, chunk.z))
        column.sections[chunk.y] = section
    return columns


def encode_block_states(
    states: Sequence[BlockState],
    indices: Sequence[int],
) -> nbt.CompoundTag:
    palette = nbt.ListTag([nbt.block_state_tag(state) for state in states])
    if len(states) == 1:
        # A section of one state carries no data array.
        return nbt.CompoundTag({"palette": palette})
    bits = bits_for(len(states), minimum=MIN_BITS)
    return nbt.CompoundTag(
        {
            "palette": palette,
            "data": nbt.long_array(pack_long_array(indices, bits, straddle=STRADDLE)),
        }
    )


def decode_block_states(tag: nbt.CompoundTag) -> tuple[list[BlockState], list[int]]:
    states = [nbt.block_state_from_tag(entry) for entry in tag["palette"]]
    data = tag.get("data")
    if data is None or len(states) == 1:
        return states, [0] * CELLS_PER_SECTION
    indices = unpack_longs(
        data.np_array,
        bits_for(len(states), minimum=MIN_BITS),
        CELLS_PER_SECTION,
        straddle=STRADDLE,
    )
    return states, indices


def build_section(
    section_y: int,
    section: BlockRegion,
    *,
    biome: str = DEFAULT_BIOME,
) -> nbt.CompoundTag:
    states = [section.palette.get_state(index) for index in range(len(section.palette))]
    return nbt.CompoundTag(
        {
            "Y": nbt.ByteTag(section_y),
            "block_states": encode_block_states(states, section.cell_array()),
            "biomes": nbt.CompoundTag({"palette": nbt.ListTag([nbt.StringTag(biome)])}),
        }
    )


def build_chunk(
    column: ChunkColumn,
    *,
    data_version: int,
    biome: str = DEFAULT_BIOME,
    status: str = DEFAULT_STATUS,
) -> nbt.CompoundTag:
    # isLightOn=0 and empty Heightmaps make the game recompute both.
    return nbt.CompoundTag(
        {
            "DataVersion": nbt.IntTag(data_version),
            "xPos": nbt.IntTag(column.chunk_x),
            "yPos": nbt.IntTag(min(column.sections, default=0)),
            "zPos": nbt.IntTag(column.chunk_z),
            "Status": nbt.StringTag(status),
            "LastUpdate": nbt.LongTag(0),
            "InhabitedTime": nbt.LongTag(0),
            "sections": nbt.ListTag(
                [
                    build_section(section_y, section, biome=biome)
                    for section_y, section in column.iter_sections()
                ]
            ),
            "block_entities": nbt.ListTag([]),
            "block_ticks": nbt.ListTag([]),
            "fluid_ticks": nbt.ListTag([]),
            "Heightmaps": nbt.CompoundTag({}),
            "isLightOn": nbt.ByteTag(0),
        }
    )


def overlay_chunk(
    chunk: nbt.CompoundTag,
    column: ChunkColumn,
    *,
    biome: str = DEFAULT_BIOME,
) -> nbt.CompoundTag:
    # Replaces only the cells the region occupies.
    existing = {section["Y"].py_int: section for section in chunk["sections"]}
    for section_y, section in column.iter_sections():
        current = existing.get(section_y)
        if current is None:
            existing[section_y] = build_section(section_y, section, biome=biome)
            continue
        states, indices = decode_block_states(current["block_states"])
        lookup = {state: index for index, state in enumerate(states)}
        for index, flag in enumerate(section.occupancy):
            if not flag:
                continue
            state = section.palette.get_state(section.cells[index])
            if state not in lookup:
                lookup[state] = len(states)
                states.append(state)
            indices[index] = lookup[state]
        current["block_states"] = encode_block_states(states, indices)
    chunk["sections"] = nbt.ListTag([existing[key] for key in sorted(existing)])
    chunk["yPos"] = nbt.IntTag(min(existing, default=0))
    chunk["Heightmaps"] = nbt.CompoundTag({})
    chunk["isLightOn"] = nbt.ByteTag(0)
    return chunk


class RegionFile:
    # A .mca file: chunk column -> chunk NBT, sectors laid out on save.

    __slots__ = ("position", "chunks", "timestamps")

    def __init__(self, position: RegionPos) -> None:
        self.position = position
        self.chunks: dict[tuple[int, int], nbt.CompoundTag] = {}
        self.timestamps: dict[tuple[int, int], int] = {}

    @classmethod
    def load(cls, path: Path | str, position: RegionPos | None = None) -> RegionFile:
        path = Path(path)
        if position is None:
            _, region_x, region_z, _ = path.name.split(".")
            position = RegionPos(int(region_x), int(region_z))
        raw = path.read_bytes()
        out = cls(position)
        if len(raw) < HEADER_SECTORS * SECTOR:
            return out
        for index in range(REGION_CHUNKS * REGION_CHUNKS):
            offset = int.from_bytes(raw[index * 4 : index * 4 + 3], "big")
            sectors = raw[index * 4 + 3]
            if offset == 0 or sectors == 0:
                continue
            start = offset * SECTOR
            length = int.from_bytes(raw[start : start + 4], "big")
            payload = _decompress(raw[start + 5 : start + 4 + length], raw[start + 4])
            key = (
                position.x * REGION_CHUNKS + index % REGION_CHUNKS,
                position.z * REGION_CHUNKS + index // REGION_CHUNKS,
            )
            out.chunks[key] = nbt.loads(payload).compound
            out.timestamps[key] = int.from_bytes(
                raw[SECTOR + index * 4 : SECTOR + index * 4 + 4], "big"
            )
        return out

    def save(self, path: Path | str, *, timestamp: int = 0) -> Path:
        locations = bytearray(SECTOR)
        timestamps = bytearray(SECTOR)
        body = bytearray()
        cursor = HEADER_SECTORS
        for key, chunk in sorted(self.chunks.items()):
            payload = zlib.compress(nbt.dumps(chunk, "", compressed=False))
            framed = struct.pack(">I", len(payload) + 1) + bytes([ZLIB]) + payload
            sectors = (len(framed) + SECTOR - 1) // SECTOR
            if sectors > 255:
                raise ValueError(f"chunk {key} needs {sectors} sectors; .mcc is not supported")
            body += framed.ljust(sectors * SECTOR, b"\x00")
            index = header_index(*key)
            locations[index * 4 : index * 4 + 3] = cursor.to_bytes(3, "big")
            locations[index * 4 + 3] = sectors
            stamp = self.timestamps.get(key, timestamp)
            timestamps[index * 4 : index * 4 + 4] = stamp.to_bytes(4, "big")
            cursor += sectors
        path = Path(path)
        path.write_bytes(bytes(locations) + bytes(timestamps) + bytes(body))
        return path


def read_region_index(
    path: Path | str,
    position: RegionPos | None = None,
) -> dict[tuple[int, int], tuple[int, int]]:
    # Chunk locations from the 4 KiB header alone.
    path = Path(path)
    if position is None:
        _, region_x, region_z, _ = path.name.split(".")
        position = RegionPos(int(region_x), int(region_z))
    with path.open("rb") as source:
        header = source.read(SECTOR)
    if len(header) < SECTOR:
        return {}
    found: dict[tuple[int, int], tuple[int, int]] = {}
    for index in range(REGION_CHUNKS * REGION_CHUNKS):
        offset = int.from_bytes(header[index * 4 : index * 4 + 3], "big")
        sectors = header[index * 4 + 3]
        if offset == 0 or sectors == 0:
            continue
        found[
            (
                position.x * REGION_CHUNKS + index % REGION_CHUNKS,
                position.z * REGION_CHUNKS + index // REGION_CHUNKS,
            )
        ] = (offset, sectors)
    return found


def read_region_chunk(path: Path | str, offset: int, sectors: int) -> nbt.CompoundTag:
    with Path(path).open("rb") as source:
        source.seek(offset * SECTOR)
        framed = source.read(sectors * SECTOR)
    length = int.from_bytes(framed[:4], "big")
    return nbt.loads(_decompress(framed[5 : 4 + length], framed[4])).compound


# How often, and how far apart, a chunk caught half written is read again.
SAVED_ATTEMPTS = 20
SAVED_PAUSE = 0.25


def read_saved_chunk(path: Path | str, chunk: tuple[int, int]) -> nbt.CompoundTag | None:
    # A chunk of a save the game may be writing as it is read. Its place is
    # taken afresh from the header each try, since a saved chunk can move,
    # and one caught half written is read again. None if it isn't there.
    path = Path(path)
    slot = (chunk[0] % REGION_CHUNKS + (chunk[1] % REGION_CHUNKS) * REGION_CHUNKS) * 4
    failure: Exception | None = None
    for attempt in range(SAVED_ATTEMPTS):
        if attempt:
            time.sleep(SAVED_PAUSE)
        try:
            with path.open("rb") as source:
                source.seek(slot)
                entry = source.read(4)
                if len(entry) < 4:
                    return None
                offset = int.from_bytes(entry[:3], "big")
                sectors = entry[3]
                if offset == 0 or sectors == 0:
                    return None
                source.seek(offset * SECTOR)
                framed = source.read(sectors * SECTOR)
            length = int.from_bytes(framed[:4], "big")
            if length < 1 or 4 + length > len(framed):
                raise ValueError("the chunk runs past what was read")
            return nbt.loads(_decompress(framed[5 : 4 + length], framed[4])).compound
        except Exception as error:  # noqa: BLE001 -- any half-written chunk
            failure = error
    raise ValueError(
        f"chunk {chunk} in {path} could not be read after {SAVED_ATTEMPTS} tries: "
        "the game may be saving the world. Close the world, or copy the save, "
        "and build again."
    ) from failure


def write_region_files(
    region: BlockRegion,
    directory: Path | str,
    *,
    data_version: int,
    merge: bool = True,
    biome: str = DEFAULT_BIOME,
    status: str = DEFAULT_STATUS,
) -> list[Path]:
    # merge overlays onto .mca files already present. Without it every cell
    # the region does not occupy becomes air.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[RegionPos, list[ChunkColumn]] = {}
    for column in chunk_columns(region).values():
        grouped.setdefault(column.position, []).append(column)

    written = []
    for position in sorted(grouped, key=lambda entry: (entry.x, entry.z)):
        path = directory / position.file_name
        region_file = (
            RegionFile.load(path, position)
            if merge and path.exists()
            else RegionFile(position)
        )
        for column in grouped[position]:
            key = (column.chunk_x, column.chunk_z)
            existing = region_file.chunks.get(key)
            region_file.chunks[key] = (
                build_chunk(column, data_version=data_version, biome=biome, status=status)
                if existing is None
                else overlay_chunk(existing, column, biome=biome)
            )
        written.append(region_file.save(path))
    return written


def _decompress(payload: bytes, compression: int) -> bytes:
    if compression == ZLIB:
        return zlib.decompress(payload)
    if compression == GZIP:
        return zlib.decompress(payload, 16 + zlib.MAX_WBITS)
    if compression == RAW:
        return payload
    raise ValueError(f"unknown chunk compression {compression}")
