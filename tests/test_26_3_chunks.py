import numpy as np
import pytest

from src.Core.block.states import BlockState
from src.Format import nbt
from src.Format.anvil import (
    CELLS_PER_SECTION, RegionFile, RegionPos, decode_block_states, write_region_files,
)
from src.Format.codec import pack_long_array
from src.Format.region import AIR, BlockRegion
from src.Terraform.region import RegionTerrain, WorldBlocks


STONE = BlockState.create("minecraft:stone")
LOG_Y = BlockState.create("minecraft:oak_log", {"axis": "y"})
LOG_X = BlockState.create("minecraft:oak_log", {"axis": "x"})


def section(palette, indices, y=4):
    tag = nbt.CompoundTag({"palette": nbt.ListTag(palette)})
    if len(palette) > 1:
        bits = max(4, (len(palette) - 1).bit_length())
        tag["data"] = nbt.long_array(pack_long_array(indices, bits, straddle=False))
    return nbt.CompoundTag({"Y": nbt.ByteTag(y), "block_states": tag})


def save_chunk(directory, sections):
    region = RegionFile(RegionPos(0, 0))
    region.chunks[(0, 0)] = nbt.CompoundTag({
        "DataVersion": nbt.IntTag(5023),
        "Status": nbt.StringTag("minecraft:full"),
        "sections": nbt.ListTag(sections),
    })
    return region.save(directory / "r.0.0.mca")


@pytest.mark.parametrize("palette, expected", [
    ([nbt.StringTag("minecraft:air"), nbt.StringTag("minecraft:stone"),
      nbt.StringTag("minecraft:oak_log")], [AIR, STONE, LOG_Y]),
    ([nbt.CompoundTag({"": nbt.StringTag("minecraft:air")}),
      nbt.CompoundTag({"": nbt.StringTag("minecraft:oak_log")}),
      nbt.CompoundTag({"id": nbt.StringTag("minecraft:oak_log"),
                       "properties": nbt.CompoundTag({"axis": nbt.StringTag("x")})})],
     [AIR, LOG_Y, LOG_X]),
    ([nbt.CompoundTag({"id": nbt.StringTag("minecraft:air")}),
      nbt.CompoundTag({"id": nbt.StringTag("minecraft:stone")}),
      nbt.CompoundTag({"id": nbt.StringTag("minecraft:oak_log"),
                       "properties": nbt.CompoundTag({"axis": nbt.StringTag("x")})})],
     [AIR, STONE, LOG_X]),
])
def test_26_3_palettes_support_terrain_survey_and_copying(tmp_path, palette, expected):
    indices = np.zeros(CELLS_PER_SECTION, dtype=np.int32)
    indices[:2] = [1, 2]
    save_chunk(tmp_path, [section(palette, indices)])

    terrain = RegionTerrain(tmp_path)
    assert terrain.column(0, 0) == [(64, expected[1])]
    assert terrain.column(1, 0) == [(64, expected[2])]
    assert terrain.column(2, 0) == ()

    states, copied_indices = WorldBlocks(tmp_path).section(0, 4, 0)
    assert states == expected
    assert np.array_equal(copied_indices, indices)
    saved = RegionFile.load(tmp_path / "r.0.0.mca").chunks[(0, 0)]
    decoded_states, decoded_indices = decode_block_states(saved["sections"][0]["block_states"])
    assert decoded_states == expected
    assert decoded_indices == indices.tolist()


@pytest.mark.parametrize("name, expected", [("minecraft:air", AIR), ("minecraft:oak_log", LOG_Y)])
def test_single_string_palette_needs_no_packed_data(tmp_path, name, expected):
    save_chunk(tmp_path, [section([nbt.StringTag(name)], None)])
    column = RegionTerrain(tmp_path).column(0, 0)
    assert column == (() if expected == AIR else [(y, expected) for y in range(64, 80)])
    states, indices = WorldBlocks(tmp_path).section(0, 4, 0)
    assert states == [expected]
    assert not indices.any()


def test_old_and_new_sections_can_share_a_chunk(tmp_path):
    indices = np.zeros(CELLS_PER_SECTION, dtype=np.int32)
    indices[0] = 1
    old = section([nbt.block_state_tag(AIR), nbt.block_state_tag(STONE)], indices)
    new = section([nbt.StringTag("minecraft:air"), nbt.StringTag("minecraft:oak_log")], indices, y=5)
    save_chunk(tmp_path, [old, new])
    assert RegionTerrain(tmp_path).column(0, 0) == [(64, STONE), (80, LOG_Y)]
    blocks = WorldBlocks(tmp_path)
    assert blocks.section(0, 4, 0)[0] == [AIR, STONE]
    assert blocks.section(0, 5, 0)[0] == [AIR, LOG_Y]


def test_merging_into_a_26_3_chunk_preserves_default_block_properties(tmp_path):
    indices = np.zeros(CELLS_PER_SECTION, dtype=np.int32)
    indices[0] = 1
    save_chunk(tmp_path, [section([nbt.StringTag("minecraft:air"),
                                   nbt.StringTag("minecraft:oak_log")], indices)])
    build = BlockRegion("build", (1, 64, 0), (1, 1, 1))
    build.set_state((1, 64, 0), STONE)
    write_region_files(build, tmp_path, data_version=5023, merge=True)
    blocks = WorldBlocks(tmp_path)
    states, indices = blocks.section(0, 4, 0)
    assert states[indices[0]] == LOG_Y
    assert states[indices[1]] == STONE
    assert states[indices[2]] == AIR
