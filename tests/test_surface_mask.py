from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest

from src.Core.Field.block_field import BlockField
from src.Core.World.block_world import BlockWorld
from src.Core.block.palette import BlockPalette
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Parser import parse
from src.Assembly.assemble import build as build_assembly
from src.Format.region import BlockRegion
from src.Format.schematics import write_schematic
from src.Terraform.grid import Grid
from src.Terraform.layers import Ground, Layer, Level
from src.Terraform.plan import Intensity, Plan, Stretch, read as read_plan
from src.Terraform.surface_mask import SurfaceMaskSpec, SurfaceReference, read_surface_mask
from src.Terraform.terraform import _join_fixed_edges, _surface_limits
from src.Terraform.terrain import FieldTerrain


def sample_terrain():
    blocks = BlockField(BlockPalette())
    for x in (0, 1):
        blocks.set_block(BlockPos(x, 10, 0), BlockState.create("minecraft:grass_block"))
    for y, name in ((11, "snow"), (12, "water"), (13, "red_carpet"),
                    (14, "oak_leaves")):
        blocks.set_block(BlockPos(0, y, 0), BlockState.create(f"minecraft:{name}"))
    return FieldTerrain(blocks)


def test_surface_types_are_opted_in_separately():
    surface = SurfaceReference(sample_terrain())
    assert surface.height(0, 0, SurfaceMaskSpec("below")) == 10
    for switch, height in (("include_snow", 11), ("include_water", 12),
                           ("include_carpet", 13), ("include_foliage", 14)):
        spec = SurfaceMaskSpec("below", **{switch: True})
        assert surface.height(0, 0, spec) == height


def test_retaining_wall_includes_nearby_water_without_raising_to_nearby_ground():
    blocks = BlockField(BlockPalette())
    blocks.set_block(BlockPos(0, 68, 0), BlockState.create("minecraft:grass_block"))
    blocks.set_block(BlockPos(2, 72, 0), BlockState.create("minecraft:water"))
    blocks.set_block(BlockPos(1, 90, 0), BlockState.create("minecraft:stone"))
    blocks.set_block(BlockPos(4, 80, 0), BlockState.create("minecraft:water"))
    reference = SurfaceReference(FieldTerrain(blocks))
    original = SurfaceMaskSpec("below", include_water=True)
    retained = read_surface_mask({"side": "below", "include_water": True, "water_reach": 3})
    assert reference.height(0, 0, original) == 68
    assert reference.height(0, 0, retained) == 72
    assert reference.mask(retained)(BlockPos(0, 72, 0))
    assert not reference.mask(retained)(BlockPos(0, 73, 0))
    assert reference.height(0, 0, SurfaceMaskSpec("below", water_reach=3)) == 68


def test_nearby_water_mask_refreshes_when_design_clears_the_pool():
    blocks = BlockField(BlockPalette())
    blocks.set_block(BlockPos(0, 68, 0), BlockState.create("minecraft:stone"))
    pool = BlockPos(2, 72, 0)
    blocks.set_block(pool, BlockState.create("minecraft:water"))
    reference = SurfaceReference(FieldTerrain(blocks))
    spec = SurfaceMaskSpec("below", reference="designed", include_water=True, water_reach=3)
    grid = Grid(ColumnPos(0, 0), 1, 1)
    shaped = SimpleNamespace(grid=grid, natural=grid.filled(68), designed=grid.filled(68), cut=frozenset())
    reference.update(shaped)
    assert reference.height(0, 0, spec) == 72
    shaped.cut = frozenset({pool})
    reference.update(shaped)
    assert reference.height(0, 0, spec) == 68


@pytest.mark.parametrize("reach", [-1, True, 1.5, "3"])
def test_water_mask_rejects_invalid_neighborhood_reach(reach):
    with pytest.raises(ValueError, match="water_reach"):
        read_surface_mask({"side": "below", "water_reach": reach})


def test_offset_and_side_clip_world_positions_and_refresh_with_design():
    registry = BlockRegistry()
    field = BlockField(registry.palette)
    for y in range(8, 16):
        field.set_block(BlockPos(0, y, 0), BlockState.create("minecraft:stone"))
    surface = SurfaceReference(sample_terrain())
    world = BlockWorld(registry)
    original = surface.mask(SurfaceMaskSpec("below", offset=-1))
    world.add_field(field, surface_mask=original)
    assert {p.y for p in world.iter_source_positions()} == {8, 9}

    designed_world = BlockWorld(registry)
    designed = surface.mask(SurfaceMaskSpec("above", offset=1, reference="designed"))
    designed_world.add_field(field, surface_mask=designed)
    assert {p.y for p in designed_world.iter_source_positions()} == set(range(8, 16))
    grid = Grid(ColumnPos(0, 0), 1, 1)
    surface.update(SimpleNamespace(
        grid=grid, natural=np.array([[10.0]]), designed=np.array([[12.0]]), cut=frozenset(),
    ))
    designed_world.refresh_surface_masks()
    assert {p.y for p in designed_world.iter_source_positions()} == {13, 14, 15}


def test_surface_mask_reads_the_column_after_a_feature_turns():
    terrain_blocks = BlockField(BlockPalette())
    terrain_blocks.set_block(BlockPos(0, 9, 1),
                             BlockState.create("minecraft:grass_block"))
    registry = BlockRegistry()
    feature = BlockField(registry.palette)
    for y in (10, 11):
        feature.set_block(BlockPos(1, y, 0), BlockState.create("minecraft:stone"))
    world = BlockWorld(registry)
    world.add_field(
        feature, quarter_turns=1,
        surface_mask=SurfaceReference(FieldTerrain(terrain_blocks)).mask(
            SurfaceMaskSpec("below", offset=1)
        ),
    )
    assert set(world.iter_source_positions()) == {BlockPos(0, 10, 1)}


def test_terraform_span_limits_fill_and_cut_without_reversing_them():
    terrain = sample_terrain()
    grid = Grid(ColumnPos(0, 0), 2, 1)
    natural = np.array([[10.0, 10.0]])
    proposed = np.array([[15.0, 5.0]])
    zero = np.zeros((1, 2))
    ground = Ground(grid, natural, zero, zero, zero, np.array([[0, 1]]))
    stretches = (
        Stretch(0, 0, Intensity.SHAPE, "embankment", 0, 0,
                surface_mask=SurfaceMaskSpec("below", offset=2)),
        Stretch(1, 1, Intensity.SHAPE, "cutting", 0, 0,
                surface_mask=SurfaceMaskSpec("above", offset=-2)),
    )
    result = _surface_limits(Plan(), terrain, grid, ground, stretches, natural, proposed)
    assert result.tolist() == [[12.0, 7.0]]


def test_designed_terraform_limit_uses_proposed_surface_as_reference():
    terrain = sample_terrain()
    grid = Grid(ColumnPos(1, 0), 1, 1)
    natural = np.array([[10.0]])
    proposed = np.array([[15.0]])
    zero = np.zeros((1, 1))
    ground = Ground(grid, natural, zero, zero, zero, zero)
    limit = SurfaceMaskSpec("below", offset=-2, reference="designed")
    result = _surface_limits(Plan(surface_mask=limit), terrain, grid, ground, (),
                             natural, proposed)
    assert result.tolist() == [[13.0]]


def test_a_fitting_footing_can_have_its_own_surface_limit():
    plan = read_plan({"follows": [{
        "fitting": "mast", "surface_mask": {"side": "below", "offset": 2}
    }]})
    assert plan.follows[0].surface_mask == SurfaceMaskSpec("below", 2)
    grid = Grid(ColumnPos(0, 0), 2, 1)
    natural = np.array([[10.0, 10.0]])
    proposed = np.array([[15.0, 15.0]])
    zero = np.zeros((1, 2))
    follower_ground = Ground(grid, natural, np.array([[0.0, 3.0]]),
                             zero, zero, zero)
    follower = Layer(Level(), mask=np.ones((1, 2)), reach=2,
                     ground=follower_ground,
                     surface_mask=plan.follows[0].surface_mask)
    result = _surface_limits(plan, sample_terrain(), grid, follower_ground,
                             (), natural, proposed, (follower,))
    assert result.tolist() == [[12.0, 15.0]]


def test_fill_tapers_to_a_protected_shore_without_changing_the_water_or_track():
    grid = Grid(ColumnPos(0, 0), 11, 1)
    natural = np.full((1, 11), 60.0)
    designed = natural.copy()
    designed[0, 1:10] = 70.0
    protected = np.zeros((1, 11), dtype=bool)
    protected[0, 0] = True
    writable = np.ones((1, 11))
    writable[0, 0] = 0.0
    distance = np.abs(np.arange(11)[None, :] - 8).astype(float)
    ground = Ground(grid, natural, distance, designed, designed, distance)
    repaired = _join_fixed_edges(
        designed, natural, np.ones((1, 11), dtype=bool), protected,
        writable, ground, 12.0,
    )
    assert repaired[0, 0] == 60.0
    assert repaired[0, 8] == 70.0
    assert repaired[0, 1] < 63.0
    assert np.diff(repaired[0, :9]).max() <= 2.0


def test_unchanged_bank_still_anchors_the_edge_when_footing_mask_covers_it():
    grid = Grid(ColumnPos(0, 0), 11, 1)
    natural = np.full((1, 11), 60.0)
    proposed = natural.copy()
    proposed[0, 1:10] = 70.0
    proposed[0, 1] = 65.0
    distance = np.abs(np.arange(11)[None, :] - 8).astype(float)
    distance[0, 1] = 2.0  # a surveyed footing can cover the bank shoulder
    seed = np.full((1, 11), 70.0)
    ground = Ground(grid, natural, distance, seed, seed, distance)
    repaired = _join_fixed_edges(
        proposed, natural, np.ones((1, 11), dtype=bool),
        np.zeros((1, 11), dtype=bool), np.ones((1, 11)), ground, 12.0,
    )
    assert repaired[0, 0] == 60.0
    assert repaired[0, 1] <= 62.0
    assert repaired[0, 8] == 70.0


def test_parser_accepts_feature_surface_mask_and_rejects_bad_options(tmp_path):
    data = {
        "path": {"points": [[0, 10, 0], [1, 10, 0]],
                 "gradients": [[1, 0, 0], [1, 0, 0]]},
        "output": {"path": "out.schem"},
        "surface": {"world": "world"},
        "domain": [{"name": "rail"}],
        "section": [{"name": "deck", "domain": "rail", "schematic": "deck.schem",
                     "surface_mask": {"side": "below", "offset": -2,
                                      "include_water": True}}],
    }
    spec = parse(data, tmp_path)
    assert spec.surface_world == tmp_path / "world"
    data["surface"]["dimension"] = "minecraft:nova_proxima"
    assert parse(data, tmp_path).surface_dimension == "minecraft:nova_proxima"
    assert spec.section("deck").surface_mask == SurfaceMaskSpec(
        "below", -2, include_water=True,
    )
    with pytest.raises(ValueError, match="side"):
        read_surface_mask({"offset": 2})
    with pytest.raises(ValueError, match="unknown"):
        read_surface_mask({"side": "above", "noise": 1})


def test_build_clips_a_mast_to_the_world_surface():
    def schematic(path, positions, name):
        blocks = BlockField(BlockPalette())
        for position in positions:
            blocks.set_block(position, BlockState.create(name))
        region = BlockRegion.from_field(blocks, name=path.stem)
        write_schematic(region, path, data_version=3700, offset=region.origin)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        schematic(root / "ground.schem", [
            BlockPos(x, 9, z) for x in range(-3, 6) for z in range(-3, 4)
        ], "minecraft:grass_block")
        schematic(root / "deck.schem", [BlockPos(0, 0, 0)], "minecraft:stone")
        schematic(root / "mast.schem", [
            BlockPos(0, y, 0) for y in range(4)
        ], "minecraft:gold_block")
        data = {
            "path": {"points": [[x, 10, 0] for x in range(3)],
                     "gradients": [[1, 0, 0]] * 3},
            "output": {"path": "out.schem"},
            "surface": {"world": "ground.schem"},
            "marker": {"start": [0, 0], "end": [2, 0]},
            "domain": [{"name": "bed"}, {"name": "support"}],
            "section": [{"name": "deck", "domain": "bed", "schematic": "deck.schem",
                         "width": 1, "depth": 1}],
            "assignment": [{"from": "start", "to": "end", "sections": ["deck"]}],
            "structure": [{"name": "mast", "domain": "support", "sections": ["deck"],
                           "headings": {"0.0": "mast.schem"},
                           "offset": [1, 0, 0],
                           "surface_mask": {"side": "below", "offset": 1}}],
        }
        world = build_assembly(parse(data, root)).world
        gold = [position for position in world.iter_source_positions()
                if world.get_block(position).name == "minecraft:gold_block"]
        assert gold
        assert {position.y for position in gold} == {10}

        (root / "terraform.toml").write_text(
            '[terraform]\nworld = "ground.schem"\ndomain = "terrain"\n'
            'default = "shape"\nlook = "minimal"\nmargin = 8\nreach = 8\n'
        )
        data["terraform"] = "terraform.toml"
        data["domain"].append({"name": "terrain"})
        data["path"]["points"] = [[x, 13, 0] for x in range(3)]
        schematic(root / "mast.schem", [
            BlockPos(0, y, 0) for y in range(-4, 4)
        ], "minecraft:gold_block")
        data["structure"][0]["surface_mask"] = {
            "side": "below", "reference": "designed",
        }
        built = build_assembly(parse(data, root))
        assert built.terraforming is not None
        gold = [position for position in built.world.iter_source_positions()
                if built.world.get_block(position).name == "minecraft:gold_block"]
        assert gold
        assert max(position.y for position in gold) < 16
        assert max(position.y for position in gold) > 9
