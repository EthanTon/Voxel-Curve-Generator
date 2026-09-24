from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from ..Core.coordinates import BlockPos
from . import nbt
from .codec import bits_for, pack_long_array, unpack_long_array
from .region import BlockRegion, SplitBudget, enclosing_bounds, split_by_budget

LITEMATIC_VERSION = 6
LITEMATIC_SUB_VERSION = 1
LITEMATIC_SUFFIX = ".litematic"
# At least 2 bits per entry, which may span two longs (pre-1.16 packing).
MIN_BITS = 2
STRADDLE = True


def build_litematic(
    regions: Sequence[BlockRegion],
    *,
    data_version: int,
    name: str,
    author: str = "",
    description: str = "",
    origin: Sequence[int] | None = None,
    created: int | None = None,
) -> nbt.CompoundTag:
    if not regions:
        raise ValueError("a litematic holds at least one region")
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"region names collide in one file: {sorted(names)}")

    enclosing = enclosing_bounds(regions)
    reference = enclosing.minimum if origin is None else BlockPos.coerce(origin)
    stamp = int(time.time() * 1000) if created is None else created
    return nbt.CompoundTag(
        {
            "Version": nbt.IntTag(LITEMATIC_VERSION),
            "SubVersion": nbt.IntTag(LITEMATIC_SUB_VERSION),
            "MinecraftDataVersion": nbt.IntTag(data_version),
            "Metadata": nbt.CompoundTag(
                {
                    "Name": nbt.StringTag(name),
                    "Author": nbt.StringTag(author),
                    "Description": nbt.StringTag(description),
                    "RegionCount": nbt.IntTag(len(regions)),
                    "TotalVolume": nbt.IntTag(sum(r.volume for r in regions)),
                    "TotalBlocks": nbt.IntTag(sum(r.block_count for r in regions)),
                    "TimeCreated": nbt.LongTag(stamp),
                    "TimeModified": nbt.LongTag(stamp),
                    "EnclosingSize": _vector(
                        enclosing.maximum - enclosing.minimum + (1, 1, 1)
                    ),
                }
            ),
            "Regions": nbt.CompoundTag(
                {region.name: _region_tag(region, reference) for region in regions}
            ),
        }
    )


def write_litematic(
    regions: Sequence[BlockRegion],
    path: Path | str,
    *,
    data_version: int,
    name: str | None = None,
    author: str = "",
    description: str = "",
    origin: Sequence[int] | None = None,
    created: int | None = None,
) -> Path:
    path = Path(path)
    root = build_litematic(
        regions,
        data_version=data_version,
        name=name or path.stem,
        author=author,
        description=description,
        origin=origin,
        created=created,
    )
    return nbt.write(path, root)


def write_split_litematic(
    region: BlockRegion,
    path: Path | str,
    budget: SplitBudget,
    *,
    data_version: int,
    author: str = "",
    description: str = "",
) -> tuple[Path, list[BlockRegion]]:
    # The pieces as named regions of one file.
    pieces = split_by_budget(region, budget)
    if not pieces:
        raise ValueError(f"{region.name} has nothing to write")
    written = write_litematic(
        pieces,
        path,
        data_version=data_version,
        name=region.name,
        author=author,
        description=description,
        origin=region.origin if budget.extent is None else budget.extent.minimum,
    )
    return written, pieces


def load_litematic(path: Path | str) -> list[BlockRegion]:
    regions = []
    for name, entry_tag in nbt.read(path).compound["Regions"].items():
        position = _read_vector(entry_tag["Position"])
        size = _read_vector(entry_tag["Size"])
        states = [nbt.block_state_from_tag(tag) for tag in entry_tag["BlockStatePalette"]]
        indices = unpack_long_array(
            entry_tag["BlockStates"].np_array,
            bits_for(len(states), minimum=MIN_BITS),
            size.x * size.y * size.z,
            straddle=STRADDLE,
        )
        region = BlockRegion(name, position, size)
        region.fill(indices, states)
        for entry in entry_tag.get("Entities", nbt.ListTag([])):
            region.add_entity(nbt.entity_from_tag(entry, position))
        regions.append(region)
    return regions


def _region_tag(region: BlockRegion, reference: BlockPos) -> nbt.CompoundTag:
    # Always a positive size from the minimum corner.
    bits = bits_for(len(region.palette), minimum=MIN_BITS)
    return nbt.CompoundTag(
        {
            "Position": _vector(region.origin - reference),
            "Size": _vector(region.size),
            "BlockStatePalette": nbt.ListTag(
                [
                    nbt.block_state_tag(region.palette.get_state(index))
                    for index in range(len(region.palette))
                ]
            ),
            "BlockStates": nbt.long_array(
                pack_long_array(region.cell_array(), bits, straddle=STRADDLE)
            ),
            "TileEntities": nbt.ListTag([]),
            "Entities": nbt.ListTag(
                [
                    nbt.entity_tag(entity, region.origin, nested=False)
                    for entity in region.entities
                ]
            ),
            "PendingBlockTicks": nbt.ListTag([]),
            "PendingFluidTicks": nbt.ListTag([]),
        }
    )


def _vector(position: BlockPos) -> nbt.CompoundTag:
    return nbt.CompoundTag(
        {
            "x": nbt.IntTag(position.x),
            "y": nbt.IntTag(position.y),
            "z": nbt.IntTag(position.z),
        }
    )


def _read_vector(tag: nbt.CompoundTag) -> BlockPos:
    return BlockPos(tag["x"].py_int, tag["y"].py_int, tag["z"].py_int)
