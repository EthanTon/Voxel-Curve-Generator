import tempfile
import tomllib
import unittest
from pathlib import Path

from src.Core.Field.block_field import BlockBounds
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format import nbt
from src.Format.codec import decode_varints
from src.Format.region import AIR, Axis, BlockRegion, RegionEntity, SplitBudget
from src.Format.schematics import (
    SCHEM_AXIS_MAX,
    build_schematic,
    load_schematic,
    parse_state_key,
    state_key,
    write_schematic,
    write_split,
)

DATA_VERSION = 4189
STONE = BlockState.create("minecraft:stone")
RAIL = BlockState.create("minecraft:rail", {"shape": "north_south", "waterlogged": "false"})
SLAB = BlockState.create("minecraft:stone_slab", {"type": "top"})


def sample_region(name="build"):
    region = BlockRegion(name, (5, 64, -3), (3, 2, 2))
    region.set_state((5, 64, -3), STONE)
    region.set_state((7, 64, -3), RAIL)
    region.set_state((6, 65, -2), SLAB)
    region.set_state((5, 65, -2), AIR)
    return region


class SchematicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def test_writing_makes_the_directory_it_was_pointed_at(self) -> None:
        path = write_schematic(
            sample_region(),
            self.directory / "schems" / "nested" / "build.schem",
            data_version=DATA_VERSION,
        )

        self.assertTrue(path.exists())

    def test_palette_keys_carry_properties_in_sorted_order(self) -> None:
        state = BlockState.create("minecraft:rail", {"waterlogged": "false", "shape": "north_south"})

        self.assertEqual(
            state_key(state),
            "minecraft:rail[shape=north_south,waterlogged=false]",
        )
        self.assertEqual(state_key(STONE), "minecraft:stone")
        self.assertEqual(parse_state_key(state_key(state)), state)
        self.assertEqual(parse_state_key("minecraft:stone"), STONE)

    def test_block_data_is_a_varint_stream_in_cell_order(self) -> None:
        region = sample_region()

        _, root = build_schematic(region, data_version=DATA_VERSION)

        data = root["BlockData"].np_array.tobytes()
        self.assertEqual(decode_varints(data), list(region.iter_cells()))
        self.assertEqual(root["Width"].py_int, 3)
        self.assertEqual(root["Height"].py_int, 2)
        self.assertEqual(root["Length"].py_int, 2)

    def test_a_palette_past_a_byte_still_indexes_correctly(self) -> None:
        region = BlockRegion("wide", (0, 0, 0), (200, 1, 1))
        for x in range(200):
            region.set_state((x, 0, 0), BlockState.create(f"mod:block_{x}"))

        path = write_schematic(region, self.directory / "wide.schem", data_version=DATA_VERSION)
        loaded, _ = load_schematic(path)

        self.assertEqual(len(loaded.palette), 201)
        self.assertEqual(loaded.state_at((199, 0, 0)).name, "mod:block_199")

    def test_version_two_keeps_the_body_at_the_root(self) -> None:
        region = sample_region()
        path = write_schematic(region, self.directory / "v2.schem", data_version=DATA_VERSION, version=2)

        named = nbt.read(path)
        root = named.compound

        self.assertEqual(named.name, "Schematic")
        self.assertEqual(root["Version"].py_int, 2)
        self.assertEqual(root["PaletteMax"].py_int, len(region.palette))

    def test_version_three_nests_the_body_and_the_blocks(self) -> None:
        region = sample_region()
        path = write_schematic(region, self.directory / "v3.schem", data_version=DATA_VERSION, version=3)

        named = nbt.read(path)
        root = named.compound

        self.assertEqual(named.name, "")
        self.assertEqual(root["Schematic"]["Version"].py_int, 3)
        self.assertEqual(root["Schematic"]["DataVersion"].py_int, DATA_VERSION)
        self.assertIn("Palette", root["Schematic"]["Blocks"])

    def test_an_unsupported_version_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_schematic(sample_region(), data_version=DATA_VERSION, version=1)

    def test_round_trip_preserves_every_cell(self) -> None:
        region = sample_region()

        for version in (2, 3):
            with self.subTest(version=version):
                path = write_schematic(
                    region,
                    self.directory / f"round_{version}.schem",
                    data_version=DATA_VERSION,
                    version=version,
                )
                loaded, offset = load_schematic(path)

                self.assertEqual(offset, BlockPos(0, 0, 0))
                self.assertEqual(loaded.size, region.size)
                self.assertEqual(
                    [state_key(loaded.palette.get_state(value)) for value in loaded.iter_cells()],
                    [state_key(region.palette.get_state(value)) for value in region.iter_cells()],
                )

    def test_entities_round_trip_with_their_tags(self) -> None:
        for version in (2, 3):
            with self.subTest(version=version):
                region = sample_region()
                region.add_entity(
                    RegionEntity(
                        "minecraft:block_display",
                        (5.5, 64.25, -3.0),
                        ("vcg.annotation", "vcg.name-left_rail"),
                        nbt.CompoundTag({"billboard": nbt.StringTag("center")}),
                    )
                )
                path = write_schematic(
                    region,
                    self.directory / f"entities_{version}.schem",
                    data_version=DATA_VERSION,
                    version=version,
                )
                loaded, _ = load_schematic(path)

                entity = loaded.entities[0]
                self.assertEqual(entity.id, "minecraft:block_display")
                self.assertEqual(entity.tags, ("vcg.annotation", "vcg.name-left_rail"))
                self.assertEqual(entity.data["billboard"].py_str, "center")

    def test_entity_positions_are_written_relative_to_the_corner(self) -> None:
        region = sample_region()
        region.add_entity(RegionEntity("minecraft:block_display", (6.5, 64.0, -3.0), ("vcg.anchor",)))

        _, root = build_schematic(region, data_version=DATA_VERSION)

        self.assertEqual(
            [tag.py_float for tag in root["Entities"][0]["Pos"]], [1.5, 0.0, 0.0]
        )

    def test_an_axis_over_the_short_limit_is_refused(self) -> None:
        region = BlockRegion("long", (0, 0, 0), (1, 1, SCHEM_AXIS_MAX + 1))

        with self.assertRaises(ValueError) as caught:
            build_schematic(region, data_version=DATA_VERSION)

        self.assertIn("32767", str(caught.exception))

    def test_split_offsets_reassemble_the_build(self) -> None:
        positions = [(x, x % 4, x * 2) for x in range(30)]
        region = BlockRegion("curve", (0, 0, 0), (30, 4, 60))
        for position in positions:
            region.set_state(position, RAIL)

        pieces = write_split(
            region,
            self.directory,
            SplitBudget(max_blocks=7, axis=Axis.X),
            data_version=DATA_VERSION,
        )

        self.assertGreater(len(pieces), 1)
        rebuilt = {}
        for piece in pieces:
            loaded, offset = load_schematic(piece.path)
            self.assertEqual(offset, piece.offset)
            for position, state in loaded.translated(region.origin).iter_occupied():
                if state != AIR:
                    rebuilt[position] = state
        self.assertEqual(rebuilt, {BlockPos(*position): RAIL for position in positions})

    def test_offsets_are_relative_to_a_shared_reference(self) -> None:
        region = BlockRegion("curve", (100, 64, -50), (10, 1, 1))
        for x in range(10):
            region.set_state((100 + x, 64, -50), STONE)

        pieces = write_split(
            region,
            self.directory,
            SplitBudget(max_blocks=4, axis=Axis.X),
            data_version=DATA_VERSION,
        )

        self.assertEqual([tuple(piece.offset) for piece in pieces], [(0, 0, 0), (4, 0, 0), (8, 0, 0)])

    def test_the_extent_origin_is_the_shared_reference(self) -> None:
        extent = BlockBounds((0, 64, 0), (3, 64, 9))
        region = BlockRegion("build", (1, 64, 1), (2, 1, 8))
        region.set_state((1, 64, 1), STONE)
        region.set_state((2, 64, 8), STONE)

        pieces = write_split(
            region,
            self.directory,
            SplitBudget(max_blocks=20, count_air=True, axis=Axis.Z, extent=extent),
            data_version=DATA_VERSION,
        )

        self.assertEqual([tuple(piece.offset) for piece in pieces], [(0, 0, 0), (0, 0, 5)])
        self.assertEqual([tuple(piece.region.size) for piece in pieces], [(4, 1, 5), (4, 1, 5)])

    def test_split_enforces_the_format_axis_limit(self) -> None:
        region = BlockRegion("long", (0, 0, 0), (1, 1, 40000))
        region.set_state((0, 0, 0), STONE)
        region.set_state((0, 0, 39999), STONE)

        pieces = write_split(
            region,
            self.directory,
            SplitBudget(max_blocks=10**9, axis=Axis.Z),
            data_version=DATA_VERSION,
            manifest=False,
        )

        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self.assertLessEqual(piece.region.size.z, SCHEM_AXIS_MAX)

    def test_manifest_lists_pieces_in_paste_order(self) -> None:
        region = BlockRegion("curve", (10, 64, 0), (10, 1, 1))
        for x in range(10):
            region.set_state((10 + x, 64, 0), STONE)

        pieces = write_split(
            region,
            self.directory,
            SplitBudget(max_blocks=4, axis=Axis.X),
            data_version=DATA_VERSION,
        )
        manifest = tomllib.loads((self.directory / "curve.toml").read_text())

        self.assertEqual(manifest["origin"], [10, 64, 0])
        self.assertEqual(manifest["piece_count"], 3)
        self.assertEqual(manifest["total_blocks"], 10)
        self.assertEqual([entry["index"] for entry in manifest["piece"]], [0, 1, 2])
        self.assertEqual([entry["offset"] for entry in manifest["piece"]], [[0, 0, 0], [4, 0, 0], [8, 0, 0]])
        self.assertEqual([entry["blocks"] for entry in manifest["piece"]], [4, 4, 2])
        for entry, piece in zip(manifest["piece"], pieces):
            self.assertTrue((self.directory / entry["file"]).exists())
            self.assertEqual(entry["file"], piece.path.name)

    def test_air_is_ignored_in_the_count_unless_asked_for(self) -> None:
        region = BlockRegion("sparse", (0, 0, 0), (6, 1, 1))
        for x in range(6):
            region.set_state((x, 0, 0), AIR if x % 2 else STONE)

        counted = write_split(
            region,
            self.directory / "blocks",
            SplitBudget(max_blocks=3, axis=Axis.X),
            data_version=DATA_VERSION,
            manifest=False,
        )
        volume = write_split(
            region,
            self.directory / "volume",
            SplitBudget(max_blocks=3, count_air=True, axis=Axis.X),
            data_version=DATA_VERSION,
            manifest=False,
        )

        self.assertEqual([piece.region.block_count for piece in counted], [3])
        self.assertEqual([piece.region.volume for piece in volume], [3, 3])


if __name__ == "__main__":
    unittest.main()
