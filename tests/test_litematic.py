import tempfile
import unittest
from pathlib import Path

from src.Core.Field.block_field import BlockBounds
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format import nbt
from src.Format.codec import bits_for, unpack_longs
from src.Format.litematic import (
    LITEMATIC_VERSION,
    MIN_BITS,
    build_litematic,
    load_litematic,
    write_litematic,
    write_split_litematic,
)
from src.Format.region import AIR, Axis, BlockRegion, RegionEntity, SplitBudget

DATA_VERSION = 4189
STONE = BlockState.create("minecraft:stone")
RAIL = BlockState.create("minecraft:rail", {"shape": "north_south"})


def region_of(name, origin, size, blocks):
    region = BlockRegion(name, origin, size)
    for position, state in blocks.items():
        region.set_state(position, state)
    return region


class LitematicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def sample(self):
        return region_of(
            "curve",
            (10, 64, -5),
            (3, 2, 2),
            {(10, 64, -5): STONE, (12, 65, -4): RAIL, (11, 64, -5): AIR},
        )

    def test_the_header_names_the_version_and_the_game(self) -> None:
        root = build_litematic([self.sample()], data_version=DATA_VERSION, name="curve")

        self.assertEqual(root["Version"].py_int, LITEMATIC_VERSION)
        self.assertEqual(root["MinecraftDataVersion"].py_int, DATA_VERSION)
        self.assertEqual(root["Metadata"]["RegionCount"].py_int, 1)
        self.assertEqual(root["Metadata"]["TotalBlocks"].py_int, 2)
        self.assertEqual(root["Metadata"]["TotalVolume"].py_int, 12)

    def test_enclosing_size_spans_every_region(self) -> None:
        left = region_of("left", (0, 0, 0), (2, 1, 1), {(0, 0, 0): STONE})
        right = region_of("right", (10, 0, 0), (2, 1, 1), {(10, 0, 0): STONE})

        root = build_litematic([left, right], data_version=DATA_VERSION, name="both")

        size = root["Metadata"]["EnclosingSize"]
        self.assertEqual(
            (size["x"].py_int, size["y"].py_int, size["z"].py_int), (12, 1, 1)
        )

    def test_region_positions_are_relative_to_the_schematic_origin(self) -> None:
        left = region_of("left", (100, 64, 0), (1, 1, 1), {(100, 64, 0): STONE})
        right = region_of("right", (104, 68, 3), (1, 1, 1), {(104, 68, 3): STONE})

        root = build_litematic([left, right], data_version=DATA_VERSION, name="both")

        positions = {
            name: (entry["Position"]["x"].py_int, entry["Position"]["y"].py_int, entry["Position"]["z"].py_int)
            for name, entry in root["Regions"].items()
        }
        self.assertEqual(positions, {"left": (0, 0, 0), "right": (4, 4, 3)})

    def test_sizes_are_always_positive(self) -> None:
        root = build_litematic([self.sample()], data_version=DATA_VERSION, name="curve")

        size = root["Regions"]["curve"]["Size"]
        self.assertEqual((size["x"].py_int, size["y"].py_int, size["z"].py_int), (3, 2, 2))

    def test_the_palette_starts_at_air(self) -> None:
        root = build_litematic([self.sample()], data_version=DATA_VERSION, name="curve")

        palette = root["Regions"]["curve"]["BlockStatePalette"]
        self.assertEqual(nbt.block_state_from_tag(palette[0]), AIR)

    def test_block_states_use_the_spanning_packing(self) -> None:
        region = self.sample()
        root = build_litematic([region], data_version=DATA_VERSION, name="curve")

        entry = root["Regions"]["curve"]
        bits = bits_for(len(region.palette), minimum=MIN_BITS)
        self.assertEqual(bits, 2)
        self.assertEqual(
            unpack_longs(entry["BlockStates"].np_array, bits, region.volume, straddle=True),
            list(region.iter_cells()),
        )

    def test_writing_makes_the_directory_it_was_pointed_at(self) -> None:
        # A build names where its output goes; the directory is made rather
        # than demanded of whoever runs it.
        path = write_litematic(
            [self.sample()],
            self.directory / "out" / "nested" / "curve.litematic",
            data_version=DATA_VERSION,
        )

        self.assertTrue(path.exists())
        self.assertEqual(len(load_litematic(path)), 1)

    def test_round_trip_preserves_names_positions_and_cells(self) -> None:
        region = self.sample()
        path = write_litematic([region], self.directory / "curve.litematic", data_version=DATA_VERSION)

        loaded = load_litematic(path)

        self.assertEqual(len(loaded), 1)
        restored = loaded[0]
        self.assertEqual(restored.name, "curve")
        self.assertEqual(restored.origin, BlockPos(0, 0, 0))
        self.assertEqual(restored.size, region.size)
        self.assertEqual(
            [restored.palette.get_state(value) for value in restored.iter_cells()],
            [region.palette.get_state(value) for value in region.iter_cells()],
        )

    def test_a_split_build_becomes_named_regions_in_one_file(self) -> None:
        region = BlockRegion("curve", (0, 64, 0), (12, 1, 1))
        for x in range(12):
            region.set_state((x, 64, 0), STONE)

        path, pieces = write_split_litematic(
            region,
            self.directory / "curve.litematic",
            SplitBudget(max_blocks=4, axis=Axis.X),
            data_version=DATA_VERSION,
        )

        self.assertEqual(len(pieces), 3)
        self.assertTrue(path.exists())
        loaded = load_litematic(path)
        self.assertEqual(
            sorted(piece.name for piece in loaded),
            ["curve_000", "curve_001", "curve_002"],
        )
        by_name = {piece.name: piece for piece in loaded}
        self.assertEqual(by_name["curve_001"].origin, BlockPos(4, 0, 0))
        self.assertEqual(sum(piece.block_count for piece in loaded), 12)

    def test_entities_round_trip_per_region(self) -> None:
        region = self.sample()
        region.add_entity(
            RegionEntity(
                "minecraft:block_display",
                (11.5, 64.0, -4.5),
                ("vcg.annotation", "vcg.name-left_rail"),
            )
        )
        path = write_litematic(
            [region], self.directory / "curve.litematic", data_version=DATA_VERSION
        )

        loaded = load_litematic(path)[0]

        self.assertEqual(len(loaded.entities), 1)
        self.assertEqual(loaded.entities[0].tags, ("vcg.annotation", "vcg.name-left_rail"))
        self.assertEqual(loaded.entities[0].block_pos, loaded.origin + (1, 0, 0))

    def test_colliding_region_names_are_refused(self) -> None:
        region = self.sample()

        with self.assertRaises(ValueError) as caught:
            build_litematic([region, region], data_version=DATA_VERSION, name="curve")

        self.assertIn("collide", str(caught.exception))

    def test_a_file_needs_at_least_one_region(self) -> None:
        with self.assertRaises(ValueError):
            build_litematic([], data_version=DATA_VERSION, name="empty")


if __name__ == "__main__":
    unittest.main()
