from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..Core.Field.block_field import BlockField
from ..Core.block.palette import BlockPalette
from ..Core.block.states import BlockState
from ..Core.coordinates import BlockPos
from . import nbt
from .codec import decode_varints_array, encode_varints
from .region import AIR, BlockRegion, SplitBudget, split_by_budget

# Width, Height and Length are shorts.
SCHEM_AXIS_MAX = 32767
SCHEM_SUFFIX = ".schem"
SUPPORTED_VERSIONS = (2, 3)


@dataclass(frozen=True, slots=True)
class SchematicPiece:
    region: BlockRegion
    index: int
    # Relative to the build origin, written as Offset.
    offset: BlockPos
    path: Path | None = None


def state_key(state: BlockState) -> str:
    if not state.properties:
        return state.name
    properties = ",".join(f"{name}={value}" for name, value in state.properties)
    return f"{state.name}[{properties}]"


def parse_state_key(key: str) -> BlockState:
    if not key.endswith("]") or "[" not in key:
        return BlockState.create(key)
    name, _, rest = key.partition("[")
    properties = {}
    for pair in rest[:-1].split(","):
        if pair:
            property_name, _, value = pair.partition("=")
            properties[property_name] = value
    return BlockState.create(name, properties)


def build_schematic(
    region: BlockRegion,
    *,
    data_version: int,
    version: int = 2,
    offset: Sequence[int] = BlockPos(0, 0, 0),
    metadata: Mapping[str, str] | None = None,
    paste_origin: Sequence[int] | None = None,
) -> tuple[str, nbt.CompoundTag]:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported schematic version {version}")
    _require_axis_limits(region)

    palette = nbt.CompoundTag(
        {
            state_key(region.palette.get_state(index)): nbt.IntTag(index)
            for index in range(len(region.palette))
        }
    )
    data = nbt.byte_array(encode_varints(region.cells))
    metadata_tags = {
        "Name": nbt.StringTag(region.name),
        **{key: nbt.StringTag(value) for key, value in (metadata or {}).items()},
    }
    if paste_origin is not None:
        # WorldEdit recovers the paste origin from WEOffset.
        relative_corner = BlockPos.coerce(offset) - BlockPos.coerce(paste_origin)
        for axis, value in zip("XYZ", relative_corner):
            metadata_tags[f"WEOffset{axis}"] = nbt.IntTag(value)
    body = {
        "Version": nbt.IntTag(version),
        "DataVersion": nbt.IntTag(data_version),
        "Metadata": nbt.CompoundTag(metadata_tags),
        "Width": nbt.ShortTag(region.size.x),
        "Height": nbt.ShortTag(region.size.y),
        "Length": nbt.ShortTag(region.size.z),
        "Offset": nbt.int_array(BlockPos.coerce(offset).as_tuple()),
    }
    entities = nbt.ListTag(
        [
            nbt.entity_tag(entity, region.origin, nested=version == 3)
            for entity in region.entities
        ]
    )
    if version == 2:
        body["PaletteMax"] = nbt.IntTag(len(region.palette))
        body["Palette"] = palette
        body["BlockData"] = data
        body["BlockEntities"] = nbt.ListTag([])
        body["Entities"] = entities
        return "Schematic", nbt.CompoundTag(body)
    body["Blocks"] = nbt.CompoundTag(
        {"Palette": palette, "Data": data, "BlockEntities": nbt.ListTag([])}
    )
    body["Entities"] = entities
    return "", nbt.CompoundTag({"Schematic": nbt.CompoundTag(body)})


def write_schematic(
    region: BlockRegion,
    path: Path | str,
    *,
    data_version: int,
    version: int = 2,
    offset: Sequence[int] = BlockPos(0, 0, 0),
    metadata: Mapping[str, str] | None = None,
    paste_origin: Sequence[int] | None = None,
) -> Path:
    name, root = build_schematic(
        region,
        data_version=data_version,
        version=version,
        offset=offset,
        metadata=metadata,
        paste_origin=paste_origin,
    )
    return nbt.write(path, root, name=name)


def write_split(
    region: BlockRegion,
    directory: Path | str,
    budget: SplitBudget,
    *,
    data_version: int,
    version: int = 2,
    origin: Sequence[int] | None = None,
    manifest: bool = True,
) -> list[SchematicPiece]:
    budget = budget.capped(SCHEM_AXIS_MAX)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        reference = BlockPos.coerce(origin)
    elif budget.extent is not None:
        reference = budget.extent.minimum
    else:
        reference = region.origin

    pieces = []
    for index, piece in enumerate(split_by_budget(region, budget)):
        offset = piece.origin - reference
        path = directory / f"{piece.name}{SCHEM_SUFFIX}"
        write_schematic(
            piece,
            path,
            data_version=data_version,
            version=version,
            offset=offset,
            metadata={"Part": f"{index}", "Of": region.name},
        )
        pieces.append(SchematicPiece(piece, index, offset, path))
    if manifest:
        write_manifest(pieces, directory / f"{region.name}.toml", region.name, reference)
    return pieces


def write_manifest(
    pieces: Sequence[SchematicPiece],
    path: Path | str,
    name: str,
    origin: Sequence[int],
) -> Path:
    lines = [
        f'name = "{name}"',
        f"origin = {list(BlockPos.coerce(origin).as_tuple())}",
        f"piece_count = {len(pieces)}",
        f"total_blocks = {sum(piece.region.block_count for piece in pieces)}",
    ]
    for piece in pieces:
        lines += [
            "",
            "[[piece]]",
            f"index = {piece.index}",
            f'name = "{piece.region.name}"',
            f'file = "{piece.path.name if piece.path else ""}"',
            f"origin = {list(piece.region.origin.as_tuple())}",
            f"offset = {list(piece.offset.as_tuple())}",
            f"size = {list(piece.region.size.as_tuple())}",
            f"blocks = {piece.region.block_count}",
            f"volume = {piece.region.volume}",
        ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_schematic(path: Path | str, *, name: str | None = None) -> tuple[BlockRegion, BlockPos]:
    # The region's origin is the file's Offset.
    root = nbt.read(path).compound
    body = root["Schematic"] if "Schematic" in root else root
    size = BlockPos(body["Width"].py_int, body["Height"].py_int, body["Length"].py_int)
    offset = BlockPos(*(int(value) for value in body["Offset"].np_array))
    blocks = body.get("Blocks")
    if blocks is None:
        palette_tag, data_tag = body["Palette"], body["BlockData"]
    else:
        palette_tag, data_tag = blocks["Palette"], blocks["Data"]
    states = {tag.py_int: parse_state_key(key) for key, tag in palette_tag.items()}
    indices = decode_varints_array(data_tag.np_array.tobytes())
    if indices.size != size.x * size.y * size.z:
        raise ValueError("block data does not match the declared volume")
    region = BlockRegion(
        name or body["Metadata"].get("Name", nbt.StringTag(Path(path).stem)).py_str,
        offset,
        size,
    )
    region.fill(indices, states)
    for entry in body.get("Entities", nbt.ListTag([])):
        region.add_entity(nbt.entity_from_tag(entry, offset))
    return region, offset


def load_blocks(path: Path | str, palette: BlockPalette | None = None) -> BlockField:
    # The schematic's non-air blocks at their schematic positions.
    region, _ = load_schematic(path)
    blocks = BlockField(palette)
    for position, state in region.iter_occupied():
        if state != AIR:
            blocks.set_block(position, state)
    return blocks


def _require_axis_limits(region: BlockRegion) -> None:
    for axis, extent in zip("xyz", region.size.as_tuple()):
        if extent > SCHEM_AXIS_MAX:
            raise ValueError(
                f"{region.name} spans {extent} on {axis}; a schematic axis "
                f"cannot exceed {SCHEM_AXIS_MAX}"
            )
