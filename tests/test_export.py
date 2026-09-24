import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from src.Assembly.export import _boxes, write_pieces
from src.Core.Field.block_field import BlockField
from src.Core.World.block_world import BlockWorld
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Format import nbt
from src.Format.schematics import load_schematic
from src.Parser.spec import SplitSpec
from src.Terraform.clearance import OPEN, Clearance
from src.Terraform.grid import Grid


GRASS = BlockState.create("minecraft:grass_block")
STONE = BlockState.create("minecraft:stone")
BRICKS = BlockState.create("minecraft:stone_bricks")
AIR = "minecraft:air"


def saved_world(directory, size=16, height=64, depth=8):
    from src.Format.anvil import write_region_files
    from src.Format.region import BlockRegion

    region = BlockRegion(
        "world", BlockPos(0, height - depth, 0), BlockPos(size, depth + 1, size)
    )
    for x in range(size):
        for z in range(size):
            for y in range(height - depth, height + 1):
                region.set_state(BlockPos(x, y, z), GRASS if y == height else STONE)
    directory.mkdir(parents=True, exist_ok=True)
    write_region_files(region, directory, data_version=3700, merge=False)


def spec_for(tmp_path, terrain=True, mode="absolute"):
    return SimpleNamespace(
        output=SimpleNamespace(
            split=SplitSpec(tmp_path / "pieces", max_span=64, terrain=terrain),
            name="test",
            mode=mode,
            data_version=3700,
        ),
        terraform=SimpleNamespace(world=tmp_path),
        path=SimpleNamespace(points=[BlockPos(2, 65, 2)]),
    )


def bricks_at(position):
    registry = BlockRegistry()
    world = BlockWorld(registry)
    field = BlockField(registry.palette)
    field.set_block(BlockPos(0, 0, 0), BRICKS)
    world.add_field(field, offset=position)
    return world


def terraforming_with(cut=(), bore=None):
    # `bore` is a column and the lowest and highest level cleared in it.
    grid = Grid(ColumnPos(0, 0), 16, 16)
    floor, ceiling = grid.filled(OPEN), grid.filled(-OPEN)
    if bore is not None:
        (x, z), low, high = bore
        floor[grid.at(x, z)], ceiling[grid.at(x, z)] = low, high
    return SimpleNamespace(
        cut=frozenset(cut), grid=grid, clearance=Clearance(floor, ceiling)
    )


def test_a_piece_with_terrain_is_the_world_as_the_build_leaves_it(tmp_path):
    saved_world(tmp_path / "region")
    pieces = write_pieces(
        bricks_at(BlockPos(4, 66, 4)),
        spec_for(tmp_path),
        terraforming_with(cut=[BlockPos(8, 64, 8)], bore=((6, 6), 63, 64)),
    )
    assert len(pieces) == 1
    region, origin = load_schematic(pieces[0].path)
    assert origin == BlockPos(4, 64, 4)

    def at(x, y, z):
        return region.state_at(BlockPos(x, y, z)).name

    assert at(4, 66, 4) == "minecraft:stone_bricks"   # what the build lays
    assert at(8, 64, 8) == AIR                        # what it cuts
    assert at(6, 64, 6) == AIR                        # what its clearance takes
    assert at(5, 64, 5) == "minecraft:grass_block"    # and the world's own
    assert at(5, 65, 5) == AIR


def test_an_absolute_piece_pastes_back_where_it_came_from(tmp_path):
    saved_world(tmp_path / "region")
    pieces = write_pieces(
        bricks_at(BlockPos(4, 66, 4)), spec_for(tmp_path), terraforming_with()
    )
    body = nbt.read(pieces[0].path).compound
    body = body["Schematic"] if "Schematic" in body else body
    metadata = body["Metadata"]
    corner = tuple(int(value) for value in body["Offset"].np_array)
    offset = tuple(metadata[f"WEOffset{axis}"].py_int for axis in "XYZ")
    # WorldEdit's origin is the corner less its offset: the first path point.
    assert corner == (4, 66, 4)
    assert tuple(np.subtract(corner, offset)) == (2, 65, 2)


def test_the_manifest_says_how_many_pieces_and_how_to_paste_them(tmp_path):
    saved_world(tmp_path / "region")
    write_pieces(
        bricks_at(BlockPos(4, 66, 4)),
        spec_for(tmp_path),
        terraforming_with(cut=[BlockPos(8, 64, 8)]),
    )
    written = (tmp_path / "pieces" / "test.toml").read_text()
    manifest = tomllib.loads(written)
    assert manifest["terrain"] is True
    assert manifest["piece_count"] == len(manifest["piece"]) == 1
    assert manifest["paste_origin"] == [2, 65, 2]
    assert "//paste -o" in written and "-a" not in written


def test_terrain_without_a_world_to_copy_it_from_is_refused(tmp_path):
    spec = spec_for(tmp_path)
    spec.terraform = None
    with pytest.raises(ValueError, match="terrain"):
        write_pieces(bricks_at(BlockPos(4, 66, 4)), spec, None)


def test_pieces_tile_the_build_and_layer_a_tall_tile():
    positions = np.array([[0, 0, 0], [10, 100, 10], [70, 5, 3]])
    boxes = list(_boxes(positions, 64, 11 * 11 * 50))
    assert [(low.as_tuple(), high.as_tuple()) for low, high in boxes] == [
        ((0, 0, 0), (10, 49, 10)),
        ((0, 50, 0), (10, 99, 10)),
        ((0, 100, 0), (10, 100, 10)),
        ((70, 5, 3), (70, 5, 3)),
    ]


def test_a_rerun_leaves_none_of_the_last_runs_pieces_behind(tmp_path):
    saved_world(tmp_path / "region")
    pieces = tmp_path / "pieces"
    pieces.mkdir()
    (pieces / "test_007.schem").write_bytes(b"old")
    (pieces / "other_007.schem").write_bytes(b"not ours")
    write_pieces(bricks_at(BlockPos(4, 66, 4)), spec_for(tmp_path), terraforming_with())
    assert sorted(each.name for each in pieces.glob("*.schem")) == [
        "other_007.schem", "test_000.schem",
    ]
