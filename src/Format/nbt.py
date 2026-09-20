from __future__ import annotations

from pathlib import Path

import numpy as np

from ..Core.block.states import BlockState
from .region import RegionEntity

# amulet-nbt moved from `amulet_nbt` (2.x, 4.x) to `amulet.nbt` (5.x). Every
# name used here is common to both, so either line can be pinned.
try:  # pragma: no cover
    import amulet_nbt as _nbt
except ModuleNotFoundError:  # pragma: no cover
    from amulet import nbt as _nbt

ByteTag = _nbt.ByteTag
ShortTag = _nbt.ShortTag
IntTag = _nbt.IntTag
LongTag = _nbt.LongTag
DoubleTag = _nbt.DoubleTag
StringTag = _nbt.StringTag
ListTag = _nbt.ListTag
CompoundTag = _nbt.CompoundTag


def byte_array(data: bytes | bytearray) -> _nbt.ByteArrayTag:
    # NBT byte arrays are signed; the payloads built here are unsigned.
    return _nbt.ByteArrayTag(np.frombuffer(bytes(data), dtype=np.int8))


def int_array(values) -> _nbt.IntArrayTag:
    return _nbt.IntArrayTag(np.asarray(list(values), dtype=np.int32))


def long_array(values) -> _nbt.LongArrayTag:
    return _nbt.LongArrayTag(np.asarray(list(values), dtype=np.int64))


def block_state_tag(state: BlockState) -> CompoundTag:
    # The {Name, Properties} form anvil sections and litematica regions use.
    entry = {"Name": StringTag(state.name)}
    if state.properties:
        entry["Properties"] = CompoundTag(
            {name: StringTag(value) for name, value in state.properties}
        )
    return CompoundTag(entry)


def block_state_from_tag(tag: CompoundTag) -> BlockState:
    properties = tag.get("Properties")
    return BlockState.create(
        tag["Name"].py_str,
        {} if properties is None else {name: value.py_str for name, value in properties.items()},
    )


def entity_tag(entity: RegionEntity, origin, *, nested: bool) -> CompoundTag:
    # Pos is relative to the region's minimum corner in both .schem and
    # .litematic. v3 nests the entity NBT under Data; v2 and litematica keep
    # it inline.
    body = CompoundTag(entity.data) if entity.data is not None else CompoundTag({})
    if entity.tags:
        body["Tags"] = ListTag([StringTag(tag) for tag in entity.tags])
    entry = {
        "id": StringTag(entity.id),
        "Id": StringTag(entity.id),
        "Pos": ListTag(
            [DoubleTag(entity.pos[axis] - origin[axis]) for axis in range(3)]
        ),
    }
    if nested:
        entry["Data"] = body
        return CompoundTag(entry)
    return CompoundTag({**entry, **dict(body.items())})


def entity_from_tag(tag: CompoundTag, origin) -> RegionEntity:
    body = tag["Data"] if "Data" in tag else tag
    tags = body.get("Tags")
    identifier = tag.get("id") or tag.get("Id")
    skip = {"id", "Id", "Pos", "Data", "Tags"}
    return RegionEntity(
        identifier.py_str,
        tuple(tag["Pos"][axis].py_float + origin[axis] for axis in range(3)),
        () if tags is None else tuple(entry.py_str for entry in tags),
        CompoundTag({k: v for k, v in body.items() if k not in skip}),
    )


def dumps(tag: CompoundTag, name: str = "", *, compressed: bool = True) -> bytes:
    return _nbt.NamedTag(tag, name).save_to(compressed=compressed)


def loads(data: bytes) -> _nbt.NamedTag:
    reader = getattr(_nbt, "read_nbt", None) or _nbt.load
    return reader(data)


def write(path: Path | str, tag: CompoundTag, *, name: str = "") -> Path:
    # The directory is made rather than demanded: a build names where its
    # output goes, and having to create that by hand first is a failure with
    # nothing to say for itself.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps(tag, name))
    return path


def read(path: Path | str) -> _nbt.NamedTag:
    return loads(Path(path).read_bytes())
