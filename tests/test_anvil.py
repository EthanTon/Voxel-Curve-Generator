import tempfile
import unittest
from pathlib import Path

from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format import nbt
from src.Format.anvil import (
    CELLS_PER_SECTION,
    HEADER_SECTORS,
    SECTOR,
    ChunkColumn,
    RegionFile,
    RegionPos,
    build_chunk,
    build_section,
    chunk_columns,
    decode_block_states,
    encode_block_states,
    header_index,
    region_of,
    write_region_files,
)
from src.Format.region import AIR, BlockRegion

DATA_VERSION = 4189
STONE = BlockState.create("minecraft:stone")
DIRT = BlockState.create("minecraft:dirt")
RAIL = BlockState.create("minecraft:rail", {"shape": "north_south"})


class AddressingTests(unittest.TestCase):
    def test_a_region_covers_thirty_two_chunks_each_way(self) -> None:
        self.assertEqual(region_of(0, 0), RegionPos(0, 0))
        self.assertEqual(region_of(31, 31), RegionPos(0, 0))
        self.assertEqual(region_of(32, 0), RegionPos(1, 0))
        self.assertEqual(RegionPos(1, -2).file_name, "r.1.-2.mca")

    def test_negative_chunks_floor_into_the_region_below(self) -> None:
        # A shift, not a division: chunk -1 belongs to region -1, not 0.
        self.assertEqual(region_of(-1, -1), RegionPos(-1, -1))
        self.assertEqual(region_of(-32, -32), RegionPos(-1, -1))
        self.assertEqual(region_of(-33, 0), RegionPos(-2, 0))

    def test_the_header_index_wraps_within_a_region(self) -> None:
        self.assertEqual(header_index(0, 0), 0)
        self.assertEqual(header_index(1, 0), 1)
        self.assertEqual(header_index(0, 1), 32)
        self.assertEqual(header_index(-1, -1), 31 + 31 * 32)
        self.assertEqual(header_index(33, 33), 1 + 32)


class SectionTests(unittest.TestCase):
    def test_a_single_state_section_carries_no_data_array(self) -> None:
        tag = encode_block_states([AIR], [0] * CELLS_PER_SECTION)

        self.assertNotIn("data", tag)
        states, indices = decode_block_states(tag)
        self.assertEqual(states, [AIR])
        self.assertEqual(len(indices), CELLS_PER_SECTION)

    def test_block_states_round_trip_through_the_padded_packing(self) -> None:
        states = [AIR, STONE, DIRT]
        indices = [index % 3 for index in range(CELLS_PER_SECTION)]

        tag = encode_block_states(states, indices)

        self.assertEqual(decode_block_states(tag), (states, indices))
        # Three states still pay four bits: Anvil's floor.
        self.assertEqual(len(tag["data"].np_array), CELLS_PER_SECTION // 16)

    def test_a_section_declares_its_y_and_a_biome(self) -> None:
        section = BlockRegion("s", (0, -64, 0), (16, 16, 16))
        section.set_state((0, -64, 0), STONE)

        tag = build_section(-4, section)

        self.assertEqual(tag["Y"].py_int, -4)
        self.assertIn("palette", tag["biomes"])
        states, indices = decode_block_states(tag["block_states"])
        self.assertEqual(states, [AIR, STONE])
        self.assertEqual(indices[0], 1)


class ChunkTests(unittest.TestCase):
    def test_columns_gather_the_sections_that_share_an_x_and_z(self) -> None:
        region = BlockRegion("build", (0, 0, 0), (1, 40, 1))
        region.set_state((0, 0, 0), STONE)
        region.set_state((0, 33, 0), STONE)

        columns = chunk_columns(region)

        self.assertEqual(list(columns), [(0, 0)])
        self.assertEqual(sorted(columns[(0, 0)].sections), [0, 2])

    def test_a_chunk_reports_its_lowest_section(self) -> None:
        region = BlockRegion("build", (0, -64, 0), (1, 80, 1))
        region.set_state((0, -64, 0), STONE)
        region.set_state((0, 5, 0), STONE)

        column = chunk_columns(region)[(0, 0)]
        chunk = build_chunk(column, data_version=DATA_VERSION)

        self.assertEqual(chunk["yPos"].py_int, -4)
        self.assertEqual([tag["Y"].py_int for tag in chunk["sections"]], [-4, 0])
        self.assertEqual(chunk["DataVersion"].py_int, DATA_VERSION)
        self.assertEqual(chunk["isLightOn"].py_int, 0)


class RegionFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def build(self, name="build", origin=(0, 0, 0), blocks=None):
        blocks = blocks or {(0, 0, 0): STONE}
        low = [min(position[axis] for position in blocks) for axis in range(3)]
        high = [max(position[axis] for position in blocks) for axis in range(3)]
        region = BlockRegion(
            name,
            BlockPos(*low),
            BlockPos(*(high[axis] - low[axis] + 1 for axis in range(3))),
        )
        for position, state in blocks.items():
            region.set_state(position, state)
        return region

    def test_a_written_file_starts_with_two_header_sectors(self) -> None:
        region = self.build()

        paths = write_region_files(region, self.directory, data_version=DATA_VERSION)

        self.assertEqual([path.name for path in paths], ["r.0.0.mca"])
        raw = paths[0].read_bytes()
        self.assertEqual(len(raw) % SECTOR, 0)
        self.assertGreaterEqual(len(raw), (HEADER_SECTORS + 1) * SECTOR)
        offset = int.from_bytes(raw[:3], "big")
        self.assertEqual(offset, HEADER_SECTORS)
        self.assertEqual(raw[3], 1)

    def test_a_written_file_reads_back_as_the_same_chunks(self) -> None:
        region = self.build(blocks={(0, 0, 0): STONE, (17, 4, 33): RAIL})

        paths = write_region_files(region, self.directory, data_version=DATA_VERSION)
        loaded = RegionFile.load(paths[0])

        self.assertEqual(sorted(loaded.chunks), [(0, 0), (1, 2)])
        states, indices = decode_block_states(
            loaded.chunks[(1, 2)]["sections"][0]["block_states"]
        )
        self.assertIn(RAIL, states)
        self.assertEqual(indices[(4 * 16 + 1) * 16 + 1], states.index(RAIL))

    def test_blocks_spread_across_region_boundaries_land_in_separate_files(self) -> None:
        region = self.build(blocks={(0, 0, 0): STONE, (512, 0, 0): STONE, (-1, 0, 0): STONE})

        paths = write_region_files(region, self.directory, data_version=DATA_VERSION)

        self.assertEqual(
            sorted(path.name for path in paths),
            ["r.-1.0.mca", "r.0.0.mca", "r.1.0.mca"],
        )

    def test_a_merge_leaves_cells_the_region_does_not_occupy_alone(self) -> None:
        # A paste must not blank the terrain around it.
        first = self.build("terrain", blocks={(0, 0, 0): DIRT, (1, 0, 0): DIRT})
        write_region_files(first, self.directory, data_version=DATA_VERSION)

        second = self.build("build", blocks={(1, 0, 0): RAIL})
        paths = write_region_files(second, self.directory, data_version=DATA_VERSION, merge=True)

        loaded = RegionFile.load(paths[0])
        states, indices = decode_block_states(
            loaded.chunks[(0, 0)]["sections"][0]["block_states"]
        )
        self.assertEqual(states[indices[0]], DIRT)
        self.assertEqual(states[indices[1]], RAIL)

    def test_merging_off_replaces_the_chunk_outright(self) -> None:
        first = self.build("terrain", blocks={(0, 0, 0): DIRT, (1, 0, 0): DIRT})
        write_region_files(first, self.directory, data_version=DATA_VERSION)

        second = self.build("build", blocks={(1, 0, 0): RAIL})
        paths = write_region_files(second, self.directory, data_version=DATA_VERSION, merge=False)

        loaded = RegionFile.load(paths[0])
        states, indices = decode_block_states(
            loaded.chunks[(0, 0)]["sections"][0]["block_states"]
        )
        self.assertEqual(states[indices[0]], AIR)
        self.assertEqual(states[indices[1]], RAIL)

    def test_a_merge_adds_a_section_the_existing_chunk_lacks(self) -> None:
        first = self.build("terrain", blocks={(0, 0, 0): DIRT})
        write_region_files(first, self.directory, data_version=DATA_VERSION)

        second = self.build("build", blocks={(0, 40, 0): RAIL})
        paths = write_region_files(second, self.directory, data_version=DATA_VERSION, merge=True)

        chunk = RegionFile.load(paths[0]).chunks[(0, 0)]
        self.assertEqual([tag["Y"].py_int for tag in chunk["sections"]], [0, 2])

    def test_a_merge_marks_light_and_heightmaps_stale(self) -> None:
        first = self.build("terrain", blocks={(0, 0, 0): DIRT})
        write_region_files(first, self.directory, data_version=DATA_VERSION)
        second = self.build("build", blocks={(1, 0, 0): RAIL})

        paths = write_region_files(second, self.directory, data_version=DATA_VERSION, merge=True)

        chunk = RegionFile.load(paths[0]).chunks[(0, 0)]
        self.assertEqual(chunk["isLightOn"].py_int, 0)
        self.assertEqual(len(chunk["Heightmaps"]), 0)

    def test_explicit_air_overwrites_on_merge(self) -> None:
        # Clearing terrain is a real instruction.
        first = self.build("terrain", blocks={(0, 0, 0): DIRT, (1, 0, 0): DIRT})
        write_region_files(first, self.directory, data_version=DATA_VERSION)

        second = self.build("clear", blocks={(1, 0, 0): AIR})
        paths = write_region_files(second, self.directory, data_version=DATA_VERSION, merge=True)

        loaded = RegionFile.load(paths[0])
        states, indices = decode_block_states(
            loaded.chunks[(0, 0)]["sections"][0]["block_states"]
        )
        self.assertEqual(states[indices[0]], DIRT)
        self.assertEqual(states[indices[1]], AIR)

    def test_loading_a_file_infers_its_position_from_the_name(self) -> None:
        region = self.build(blocks={(-1, 0, -1): STONE})
        paths = write_region_files(region, self.directory, data_version=DATA_VERSION)

        loaded = RegionFile.load(paths[0])

        self.assertEqual(loaded.position, RegionPos(-1, -1))
        self.assertEqual(sorted(loaded.chunks), [(-1, -1)])


if __name__ == "__main__":
    unittest.main()
