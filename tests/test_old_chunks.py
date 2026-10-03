"""Worlds saved by older games: 1.13-1.17 Level sections and pre-1.13 numeric blocks."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from src.Core.block.states import BlockState
from src.Format import legacy, nbt
from src.Format.anvil import RegionFile, RegionPos
from src.Format.codec import pack_long_array
from src.Terraform.classify import BlockClass, _classify_name, classify, ground_blocks
from src.Terraform.region import RegionTerrain, WorldBlocks, _sections

CELLS = 4096


def level_chunk(palette, indices, version, straddle, status="full", y=4):
    bits = max(4, (len(palette) - 1).bit_length())
    section = nbt.CompoundTag({
        "Y": nbt.ByteTag(y),
        "Palette": nbt.ListTag([nbt.block_state_tag(BlockState.create(name)) for name in palette]),
        "BlockStates": nbt.long_array(pack_long_array(indices, bits, straddle=straddle)),
    })
    level = nbt.CompoundTag({"xPos": nbt.IntTag(0), "zPos": nbt.IntTag(0),
                             "Status": nbt.StringTag(status), "Sections": nbt.ListTag([section])})
    return nbt.CompoundTag({"DataVersion": nbt.IntTag(version), "Level": level})


def numbered_chunk(ids, data, add=None, y=4, populated=1):
    nibbles = (data[0::2] | (data[1::2] << 4)).astype(np.uint8)
    section = {"Y": nbt.ByteTag(y), "Blocks": nbt.byte_array((ids & 255).astype(np.uint8).tobytes()),
               "Data": nbt.byte_array(nibbles.tobytes())}
    if add is not None:
        packed = (add[0::2] | (add[1::2] << 4)).astype(np.uint8)
        section["Add"] = nbt.byte_array(packed.tobytes())
    level = nbt.CompoundTag({"xPos": nbt.IntTag(0), "zPos": nbt.IntTag(0),
                             "TerrainPopulated": nbt.ByteTag(populated),
                             "Sections": nbt.ListTag([nbt.CompoundTag(section)])})
    return nbt.CompoundTag({"DataVersion": nbt.IntTag(1343), "Level": level})


def names(chunk):
    (y, palette, indices), = list(_sections(chunk))
    return y, np.array([tag["Name"].py_str for tag in palette])[indices]


@pytest.mark.parametrize("version, straddle", [(2586, False), (1976, True)])
def test_level_sections_read_whether_or_not_entries_span_longs(version, straddle):
    # 33 states take 6 bits, which split across longs before 1.16.
    palette = ["minecraft:air"] + [f"minecraft:{colour}_wool" for colour in legacy.COLOURS] * 2
    palette = list(dict.fromkeys(palette)) + [f"minecraft:{colour}_concrete" for colour in legacy.COLOURS]
    indices = np.arange(CELLS) % len(palette)
    y, found = names(level_chunk(palette, indices, version, straddle))
    assert y == 4
    assert list(found[:40]) == [palette[i] for i in indices[:40]]
    assert np.array_equal(found, np.array(palette)[indices])


def test_numbered_blocks_read_as_modern_states():
    ids = np.zeros(CELLS, dtype=np.int32)
    data = np.zeros(CELLS, dtype=np.int32)
    ids[0], ids[1], data[1], ids[2], data[2] = 2, 1, 5, 3, 1
    ids[3], data[3] = 17, 4  # an oak log lying along x
    ids[4], data[4] = 159, 14
    ids[5], data[5] = 31, 1  # tall grass, the plant
    y, found = names(numbered_chunk(ids, data))
    assert list(found[:6]) == ["minecraft:grass_block", "minecraft:andesite", "minecraft:coarse_dirt",
                               "minecraft:oak_log", "minecraft:red_terracotta", "minecraft:grass"]
    assert found[6] == "minecraft:air"
    assert legacy.state(17, 4).properties == (("axis", "x"),)


def test_a_block_id_over_255_reads_its_high_nibble_from_add():
    ids = np.full(CELLS, 1, dtype=np.int32)
    data = np.zeros(CELLS, dtype=np.int32)
    add = np.zeros(CELLS, dtype=np.int32)
    add[7] = 1
    _, found = names(numbered_chunk(ids, data, add))
    assert found[7] == "minecraft:legacy_257"
    assert classify(BlockState.create(found[7])) is BlockClass.STRUCTURE
    assert found[8] == "minecraft:stone"


@pytest.mark.parametrize("chunk, ready", [
    (lambda: level_chunk(["minecraft:stone"], np.zeros(CELLS, int), 2586, False, "full"), True),
    (lambda: level_chunk(["minecraft:stone"], np.zeros(CELLS, int), 1631, True, "postprocessed"), True),
    (lambda: level_chunk(["minecraft:stone"], np.zeros(CELLS, int), 2586, False, "carvers"), False),
    (lambda: numbered_chunk(np.ones(CELLS, int), np.zeros(CELLS, int), populated=1), True),
    (lambda: numbered_chunk(np.ones(CELLS, int), np.zeros(CELLS, int), populated=0), False),
])
def test_old_chunks_are_surveyed_once_generated(chunk, ready):
    with TemporaryDirectory() as folder:
        region = RegionFile(RegionPos(0, 0))
        region.chunks[(0, 0)] = chunk()
        region.save(Path(folder) / "r.0.0.mca")
        terrain, blocks = RegionTerrain(Path(folder)), WorldBlocks(Path(folder))
        assert terrain.generated(0, 0) is ready
        assert blocks.generated(0, 0) is ready
        if ready:
            column = terrain.column(3, 5)
            assert column and column[-1][0] == 4 * 16 + 15
            assert column[-1][1].name == "minecraft:stone"
            states, indices = blocks.section(0, 4, 0)
            assert states[indices[0]].name == "minecraft:stone"


def test_old_and_custom_ground_is_ground():
    for name in ("minecraft:terracotta", "minecraft:infested_stone", "minecraft:smooth_basalt",
                 "biomesoplenty:orange_sand", "create:limestone", "terra:dark_soil"):
        assert _classify_name(name) is BlockClass.GROUND, name
    for name in ("minecraft:grass", "minecraft:short_grass"):
        assert _classify_name(name) is BlockClass.VEGETATION, name
    for name in ("create:polished_cut_limestone", "terra:oak_table", "minecraft:stone_bricks"):
        assert _classify_name(name) is BlockClass.STRUCTURE, name


def test_a_build_names_its_own_ground():
    odd = BlockState.create("terra:mossy_lump")
    assert classify(odd) is BlockClass.STRUCTURE
    with ground_blocks(("terra:*",)):
        assert classify(odd) is BlockClass.GROUND
    assert classify(odd) is BlockClass.STRUCTURE
