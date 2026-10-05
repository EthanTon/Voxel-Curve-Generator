from __future__ import annotations

from pathlib import Path

import numpy as np

from ..Core.block.states import BlockState
from ..Core.block.vanilla import vanilla_catalog
from .region import RegionEntity

# amulet-nbt is `amulet_nbt` before 5.x and `amulet.nbt` from 5.x.
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

BYTE_ARRAY, INT_ARRAY, LONG_ARRAY = 7, 11, 12


def byte_array(data: bytes | bytearray) -> _nbt.ByteArrayTag:
    return _array_tag(BYTE_ARRAY, bytes(data), len(data))


def int_array(values) -> _nbt.IntArrayTag:
    values = np.asarray(list(values), dtype=np.int32)
    return _array_tag(INT_ARRAY, values.astype(">i4").tobytes(), values.size)


def long_array(values) -> _nbt.LongArrayTag:
    values = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=np.int64)
    return _array_tag(LONG_ARRAY, values.astype(">i8").tobytes(), values.size)


def _array_tag(tag_id: int, payload: bytes, count: int):
    # Parsed from binary: the tag constructors copy element by element.
    header = bytes((tag_id, 0, 0)) + count.to_bytes(4, "big", signed=True)
    return loads(header + payload, compressed=False).tag


def block_state_tag(state: BlockState) -> CompoundTag:
    # The legacy {Name, Properties} form used by schematic palettes.
    entry = {"Name": StringTag(state.name)}
    if state.properties:
        entry["Properties"] = CompoundTag(
            {name: StringTag(value) for name, value in state.properties}
        )
    return CompoundTag(entry)


def block_state_from_tag(tag: CompoundTag | StringTag) -> BlockState:
    # Mixed NBT lists wrap non-compound entries under an empty name.
    if isinstance(tag, CompoundTag) and len(tag) == 1 and "" in tag:
        tag = tag[""]
    if isinstance(tag, StringTag):
        name, properties = tag.py_str, None
    elif "Name" in tag:
        # Preserve the older saved-world and schematic representation.
        properties = tag.get("Properties")
        return BlockState.create(
            tag["Name"].py_str,
            {} if properties is None else {key: value.py_str for key, value in properties.items()},
        )
    else:
        name, properties = tag["id"].py_str, tag.get("properties")

    # 26.3's string form represents the block's default state, including
    # properties such as a log's axis. New compounds may override defaults.
    if ":" not in name:
        name = f"minecraft:{name}"
    catalog = vanilla_catalog()
    defaults = dict(catalog.default_state(name).properties) if name in catalog else {}
    if properties is not None:
        defaults.update({key: value.py_str for key, value in properties.items()})
    return BlockState.create(name, defaults)


def entity_tag(entity: RegionEntity, origin, *, nested: bool) -> CompoundTag:
    # Pos is relative to the region's minimum corner. Schematic v3 nests the
    # entity data under Data; v2 and litematica keep it inline.
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


def loads(data: bytes, *, compressed: bool = True) -> _nbt.NamedTag:
    reader = getattr(_nbt, "read_nbt", None) or _nbt.load
    return reader(data, compressed=compressed)


def write(path: Path | str, tag: CompoundTag, *, name: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps(tag, name))
    return path


def read(path: Path | str) -> _nbt.NamedTag:
    return loads(Path(path).read_bytes())
