import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from src.Assembly.export import _boxes, _changed_cells, write, write_pieces
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


def test_a_cut_cannot_yield_back_to_the_saved_block():
    position = BlockPos(4, 75, 4)
    world = SimpleNamespace(
        registry=SimpleNamespace(get_state=lambda _: BlockState.create(AIR)),
        iter_resolved_chunks=lambda _: ((None, {position: 1}),),
        source_at=lambda _: None,
        yields=lambda _: True,
    )
    states, positions, codes, soft = _changed_cells(
        world, SimpleNamespace(cut=frozenset({position}))
    )
    assert states[codes[0]].name == AIR
    assert tuple(positions[0]) == tuple(position)
    assert not soft[0]


def test_a_cut_under_water_fills_with_it():
    grid = Grid(ColumnPos(0, 0), 8, 8)
    world = SimpleNamespace(iter_resolved_chunks=lambda _: ())
    kelp, above = BlockPos(4, 58, 4), BlockPos(4, 63, 4)
    terraforming = SimpleNamespace(
        cut=frozenset({kelp, above}), grid=grid,
        water=grid.filled(61.0), natural=grid.filled(46.0),
    )
    states, positions, codes, _ = _changed_cells(world, terraforming)
    found = {tuple(position): states[code].name for position, code in zip(positions, codes)}
    assert found[tuple(kelp)] == "minecraft:water"
    assert found[tuple(above)] == AIR


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
        terraform=SimpleNamespace(world=tmp_path, terrain_world=tmp_path,
                                  dimension=None, waterlog=False),
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


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("state,capable", (
    (BlockState.create("minecraft:stone_slab", {"type": "bottom"}), True),
    (BlockState.create("minecraft:rail", {"shape": "east_west"}), True),
    (BlockState.create("minecraft:stone_brick_wall"), True),
    (BlockState.create("minecraft:stone_slab", {"type": "double"}), False),
    (BRICKS, False),
))
def test_submerged_build_blocks_waterlog_only_when_enabled_and_capable(tmp_path, enabled, state, capable):
    saved_world(tmp_path / "region", height=60)
    spec = spec_for(tmp_path)
    spec.terraform.waterlog = enabled
    shaped = terraforming_with()
    shaped.natural = shaped.grid.filled(60)
    shaped.designed = shaped.grid.filled(60)
    shaped.water = shaped.grid.filled(64)
    world = bricks_at(BlockPos(4, 62, 4))
    field = BlockField(world.registry.palette)
    field.set_block(BlockPos(0, 0, 0), state)
    world.add_field(field, offset=BlockPos(4, 62, 4))
    world.add_field(field, offset=BlockPos(4, 65, 4))
    piece, = write_pieces(world, spec, shaped)
    region, _ = load_schematic(piece.path)
    assert region.state_at(BlockPos(4, 62, 4)).get("waterlogged") == (
        "true" if enabled and capable else state.get("waterlogged")
    )
    assert region.state_at(BlockPos(4, 65, 4)).get("waterlogged") == state.get("waterlogged")


def test_a_piece_with_terrain_is_the_world_as_the_build_leaves_it(tmp_path):
    saved_world(tmp_path / "region")
    pieces = write_pieces(
        bricks_at(BlockPos(4, 66, 4)),
        spec_for(tmp_path),
        terraforming_with(cut=[BlockPos(8, 64, 8), BlockPos(6, 64, 6)], bore=((6, 6), 63, 64)),
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


def test_exported_tunnel_clears_terrain_below_rail_height_between_the_lanes(tmp_path):
    from src.Terraform import Alignment, Plan, Sculpt, Span, terraform
    from src.Terraform.region import RegionTerrain

    saved_world(tmp_path / "region")
    registry = BlockRegistry()
    world = BlockWorld(registry)
    shell = BlockField(registry.palette)
    envelope = {}
    for x in range(8, 12):
        for z in (7, 8, 9):
            levels = {7: [59, 60, 62, 63, 69], 8: [59, 60, 69], 9: [59, 60, 61, 69]}[z]
            envelope[x, z] = levels
            for y in levels:
                state = (BlockState.create("minecraft:rail", {"shape": "east_west"})
                         if z == 7 and y == 63 else BRICKS)
                shell.set_block(BlockPos(x, y, z), state)
    world.add_field(shell)
    alignment = Alignment.of([BlockPos(x, 59, z) for x in range(8, 12) for z in (7, 8, 9)],
                             [x - 8 for x in range(8, 12) for z in (7, 8, 9)])
    shaped = terraform(
        Plan(margin=2, sculpt=Sculpt(enabled=True),
             spans=(Span((8, 7), (11, 7), profile="tunnel", headroom=0),)),
        RegionTerrain(tmp_path / "region"), alignment, registry.palette, envelope=envelope,
        running={(x, z): 63 for x in range(8, 12) for z in (7, 8, 9)},
    )
    pieces = write_pieces(world, spec_for(tmp_path), shaped)
    regions = [load_schematic(piece.path)[0] for piece in pieces]

    def exported(x, y, z):
        return next(region.state_at(BlockPos(x, y, z)) for region in regions
                    if region.origin.x <= x < region.origin.x + region.size.x
                    and region.origin.y <= y < region.origin.y + region.size.y
                    and region.origin.z <= z < region.origin.z + region.size.z)

    for x in range(8, 12):
        for z, floor in ((7, 63), (8, 60), (9, 61)):
            for y in range(floor + 1, 69):
                assert exported(x, y, z).name == AIR
            assert exported(x, 69, z) == BRICKS
        assert exported(x, 60, 8) == BRICKS
        assert exported(x, 61, 7) == STONE  # soil contact below the rail foundation


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("profile", ["tunnel", "retained_cutting", "bridge"])
@pytest.mark.parametrize("native", [
    BlockState.create("minecraft:water"), BlockState.create("minecraft:lava"),
    BRICKS, BlockState.create("minecraft:oak_leaves"),
])
def test_track_clearance_overrides_protection_and_fluids_only_inside_the_bed(tmp_path, enabled, profile, native):
    from src.Format.anvil import write_region_files
    from src.Format.region import BlockRegion
    from src.Terraform import Alignment, Plan, Span, TerrainMask, terraform
    from src.Terraform.region import RegionTerrain

    saved = BlockRegion("source", (0, 56, 0), (16, 17, 16))
    for x in range(16):
        for z in range(16):
            for y in range(56, 71):
                saved.set_state(BlockPos(x, y, z), GRASS if y == 70 else STONE)
    for z in (7, 8, 12):
        # An existing road above the bore protects the entire native column.
        saved.set_state(BlockPos(8, 72, z), BlockState.create("minecraft:gray_concrete"))
        for y in range(65, 69):
            saved.set_state(BlockPos(8, y, z), native)
    for y in range(61, 69):
        saved.set_state(BlockPos(8, y, 7), BlockState.create("minecraft:water"))
        saved.set_state(BlockPos(8, y, 12), BlockState.create("minecraft:water"))
    for y in (69, 70):
        saved.set_state(BlockPos(8, y, 12), BlockState.create(AIR))
    write_region_files(saved, tmp_path / "region", data_version=4671, merge=False)

    registry = BlockRegistry()
    world = BlockWorld(registry)
    shell = BlockField(registry.palette)
    roof = [69] if profile == "tunnel" else []
    envelope = {(8, 7): [59, 60, 62, 63, *roof], (8, 8): [59, 60, *roof]}
    rail = BlockState.create("minecraft:rail", {"shape": "east_west", "waterlogged": "true"})
    for (x, z), levels in envelope.items():
        for y in levels:
            shell.set_block(BlockPos(x, y, z), rail if z == 7 and y == 63 else BRICKS)
    world.add_field(shell)
    outside = BlockPos(8, 65, 12)
    slab = BlockField(registry.palette)
    slab.set_block(outside, BlockState.create("minecraft:stone_slab", {"type": "bottom"}))
    world.add_field(slab)
    shaped = terraform(
        Plan(margin=5, clear_track=enabled,
             terrain_mask=TerrainMask(preserve_water=True, preserve_structures=True),
             spans=(Span((8, 7), (8, 7), profile=profile, headroom=0),)),
        RegionTerrain(tmp_path / "region"),
        Alignment.of([BlockPos(8, 59, z) for z in (7, 8)], [0, 0]),
        registry.palette, envelope=envelope, running={(8, z): 63 for z in (7, 8)},
    )
    spec = spec_for(tmp_path)
    spec.terraform.waterlog = True
    regions = [load_schematic(piece.path)[0] for piece in write_pieces(world, spec, shaped)]

    def at(x, y, z):
        position = BlockPos(x, y, z)
        return next(region.state_at(position) for region in regions if region.bounds.contains(position))

    if enabled:
        for z, floor in ((7, 63), (8, 60)):
            for y in range(floor + 1, 69):
                assert at(8, y, z).name == AIR
        assert at(8, 63, 7).get("waterlogged") == "false"
    else:
        assert at(8, 64, 8) == STONE
        assert at(8, 65, 8) == native
        assert at(8, 63, 7).get("waterlogged") == "true"
    assert at(8, 60, 8) == BRICKS
    if roof:
        assert at(8, 69, 8) == BRICKS
    assert at(8, 72, 8).name == "minecraft:gray_concrete"
    assert at(8, 66, 12).name == "minecraft:water"
    assert at(*outside).get("waterlogged") == "true"


@pytest.mark.parametrize("clear_track", [False, True])
@pytest.mark.parametrize("waterlog", [False, True])
@pytest.mark.parametrize("authored_wet", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("profile", ["tunnel", "retained_cutting"])
def test_clear_track_keeps_roof_and_fittings_dry_outside_empty_clearance(
    tmp_path, clear_track, waterlog, authored_wet, split, profile,
):
    from src.Format.anvil import write_region_files
    from src.Format.region import BlockRegion
    from src.Terraform import Alignment, Plan, Span, terraform
    from src.Terraform.region import RegionTerrain

    saved = BlockRegion("river", (0, 56, 0), (16, 9, 16))
    for x in range(16):
        for z in range(16):
            for y in range(56, 65):
                saved.set_state(BlockPos(x, y, z), STONE if y <= 58 else BlockState.create("minecraft:water"))
    write_region_files(saved, tmp_path / "region", data_version=4671, merge=False)
    registry = BlockRegistry()
    world = BlockWorld(registry)
    shell = BlockField(registry.palette)
    roof, wire, outside = BlockPos(8, 62, 8), BlockPos(9, 62, 8), BlockPos(8, 62, 12)
    wet = "true" if authored_wet else "false"
    shell.set_block(BlockPos(8, 60, 8), BRICKS)
    shell.set_block(roof, BlockState.create("minecraft:pale_oak_stairs", {"waterlogged": wet}))
    shell.set_block(wire, BlockState.create("minecraft:iron_bars", {"waterlogged": wet}))
    shell.set_block(outside, BlockState.create("minecraft:stone_slab", {"type": "bottom"}))
    world.add_field(shell)
    shaped = terraform(
        Plan(margin=5, clear_track=clear_track,
             spans=(Span((8, 8), (8, 8), profile=profile, headroom=0 if profile == "tunnel" else 12),)),
        RegionTerrain(tmp_path / "region"), Alignment.of([BlockPos(8, 60, 8)]),
        registry.palette, envelope={(8, 8): [60, 62], (9, 8): [62]}, running={(8, 8): 60},
    )
    # A tunnel roof clips the bore; an open cut also drains above its wires.
    if shaped.track_clearance is not None:
        assert shaped.track_clearance.ceiling[shaped.grid.at(8, 8)] == (61 if profile == "tunnel" else 74)
    spec = spec_for(tmp_path)
    spec.terraform.waterlog = waterlog
    if split:
        regions = [load_schematic(piece.path)[0] for piece in write_pieces(world, spec, shaped)]
    else:
        spec.output.split = None
        spec.output.path = tmp_path / "main.schem"
        write(world, spec, shaped)
        regions = [load_schematic(spec.output.path)[0]]

    def at(position):
        return next(region.state_at(position) for region in regions if region.bounds.contains(position))

    for position in (roof, wire):
        assert at(position).get("waterlogged") == ("false" if clear_track else "true" if waterlog else wet)
    assert at(outside).get("waterlogged") == ("true" if waterlog else None)
    if split:
        assert at(BlockPos(8, 63, 12)).name == "minecraft:water"
        if clear_track and profile != "tunnel":
            assert at(BlockPos(8, 63, 8)).name == AIR


@pytest.mark.parametrize('native_height', [56, 66])
def test_exported_bridge_to_trench_clears_the_whole_bed_on_a_grade(tmp_path, native_height):
    from src.Terraform import Alignment, Plan, Sculpt, Span, terraform
    from src.Terraform.region import RegionTerrain

    saved_world(tmp_path / "region", height=native_height)
    registry = BlockRegistry()
    world = BlockWorld(registry)
    bed = BlockField(registry.palette)
    envelope, blocks, stations, running = {}, [], [], {}
    for x in range(8, 12):
        rise = int(x >= 10)
        for z in (7, 8, 9, 10):
            levels = {7: [59, 60, 62, 63], 8: [59, 60], 9: [59, 60, 61],
                      10: list(range(59, 69))}[z]
            envelope[x, z] = [y + rise for y in levels]
            running[x, z] = 63 + rise
            for y in levels:
                position = BlockPos(x, y + rise, z)
                state = (BlockState.create("minecraft:rail", {"shape": "east_west"})
                         if z == 7 and y == 63 else BRICKS)
                bed.set_block(position, state)
                blocks.append(position)
                stations.append(x - 8)
    world.add_field(bed)
    shaped = terraform(
        Plan(margin=2, sculpt=Sculpt(enabled=True), spans=(
            Span((8, 7), (9, 7), profile="bridge", headroom=4, transition=0),
            Span((10, 7), (11, 7), profile="retained_cutting", transition=0),
        )),
        RegionTerrain(tmp_path / "region"), Alignment.of(blocks, stations),
        registry.palette, envelope=envelope, running=running,
    )
    # Match assembly: earthwork fill is applied before exporting the pieces.
    from src.Core.modes import PlacementMode
    world.add_field(shaped.fill, placement_mode=PlacementMode.REPLACE_AIR_ONLY,
                    earthworks=True)
    regions = [load_schematic(piece.path)[0]
               for piece in write_pieces(world, spec_for(tmp_path), shaped)]

    def exported(x, y, z):
        return next(region.state_at(BlockPos(x, y, z)) for region in regions
                    if region.origin.x <= x < region.origin.x + region.size.x
                    and region.origin.y <= y < region.origin.y + region.size.y
                    and region.origin.z <= z < region.origin.z + region.size.z)

    for x in range(8, 12):
        rise = int(x >= 10)
        for z, floor in ((7, 63), (8, 60), (9, 61)):
            for y in range(floor + rise + 1, 67 + rise):
                assert exported(x, y, z).name == AIR
        assert exported(x, 60 + rise, 8) == BRICKS
        assert exported(x, 68 + rise, 10) == BRICKS  # retaining wall
        if native_height == 66:
            assert exported(x, 61 + rise, 7) == STONE  # sleeper contact


def test_a_piece_copies_terrain_from_the_selected_dimension(tmp_path):
    saved_world(tmp_path / "region", height=60)
    saved_world(tmp_path / "dimensions" / "minecraft" / "nova_proxima" / "region")
    spec = spec_for(tmp_path)
    spec.terraform.dimension = "minecraft:nova_proxima"
    pieces = write_pieces(
        bricks_at(BlockPos(4, 66, 4)),
        spec,
        terraforming_with(cut=[BlockPos(8, 64, 8)]),
    )
    region, _ = load_schematic(pieces[0].path)
    assert region.state_at(BlockPos(5, 64, 5)) == GRASS


def test_bridge_pieces_include_a_continuous_strip_and_the_ground_below(tmp_path):
    saved_world(tmp_path / "region", size=32)
    spec = spec_for(tmp_path)
    spec.output.split = SplitSpec(tmp_path / "pieces", max_span=16, terrain=True, terrain_depth=4)
    grid = Grid(ColumnPos(0, 0), 32, 32)
    coverage = grid.filled(False, bool)
    coverage[4:9, 2:30] = True
    terraforming = SimpleNamespace(
        cut=frozenset(), grid=grid, clearance=Clearance.none(grid),
        coverage=coverage, natural=grid.filled(64), terrain_top=grid.filled(64),
    )
    pieces = write_pieces(bricks_at(BlockPos(15, 75, 6)), spec, terraforming)
    regions = [load_schematic(piece.path)[0] for piece in pieces]
    for x in range(2, 30):
        for z in range(4, 9):
            column = [r for r in regions if r.origin.x <= x < r.origin.x + r.size.x
                      and r.origin.z <= z < r.origin.z + r.size.z]
            assert len(column) == 1
            assert column[0].state_at(BlockPos(x, 64, z)) == GRASS
            assert column[0].state_at(BlockPos(x, 60, z)) == STONE


def test_copying_terrain_does_not_clear_preserved_water_from_build_clearance(tmp_path):
    from src.Format.anvil import write_region_files
    from src.Format.region import BlockRegion
    river = BlockRegion("river", BlockPos(0, 60, 0), BlockPos(16, 5, 16))
    for x in range(16):
        for z in range(16):
            river.set_state(BlockPos(x, 60, z), STONE)
            for y in range(61, 65):
                river.set_state(BlockPos(x, y, z), BlockState.create("minecraft:water"))
    write_region_files(river, tmp_path / "region", data_version=3700, merge=False)
    shaped = terraforming_with(bore=((6, 6), 61, 75))
    shaped.coverage = shaped.grid.filled(True, bool)
    shaped.natural = shaped.grid.filled(60)
    shaped.terrain_top = shaped.grid.filled(64)
    pieces = write_pieces(bricks_at(BlockPos(6, 75, 6)), spec_for(tmp_path), shaped)
    region, _ = load_schematic(pieces[0].path)
    assert all(region.state_at(BlockPos(6, y, 6)).name == "minecraft:water" for y in range(61, 65))


def test_a_missing_saved_chunk_is_refused_instead_of_exported_as_air(tmp_path):
    saved_world(tmp_path / "region")
    previous = tmp_path / "pieces" / "test_0.schem"
    previous.parent.mkdir()
    previous.write_bytes(b"previous complete export")
    with pytest.raises(ValueError, match=r"saved chunk \(1, 0\) is missing"):
        write_pieces(bricks_at(BlockPos(20, 75, 6)), spec_for(tmp_path), terraforming_with())
    assert previous.read_bytes() == b"previous complete export"


def test_the_main_schematic_also_carries_terrain_and_explicit_cuts(tmp_path):
    saved_world(tmp_path / "region")
    spec = spec_for(tmp_path)
    spec.output.path = tmp_path / "main.schem"
    shaped = terraforming_with(cut=[BlockPos(8, 64, 8)])
    shaped.coverage = shaped.grid.filled(True, bool)
    shaped.natural = shaped.grid.filled(64)
    shaped.terrain_top = shaped.grid.filled(64)
    write(bricks_at(BlockPos(4, 75, 4)), spec, shaped)
    region, _ = load_schematic(spec.output.path)
    assert region.state_at(BlockPos(5, 64, 5)) == GRASS
    assert region.state_at(BlockPos(5, 60, 5)) == STONE
    assert region.state_at(BlockPos(8, 64, 8)).name == AIR
    assert region.state_at(BlockPos(4, 75, 4)) == BRICKS


def test_an_export_uses_the_surveyed_chunks_even_if_the_save_changes(tmp_path):
    from src.Terraform import Alignment, Plan, terraform
    from src.Terraform.region import RegionTerrain
    saved_world(tmp_path / "region", height=64)
    shaped = terraform(Plan(margin=2), RegionTerrain(tmp_path / "region"), Alignment.of([BlockPos(4, 75, 4)]))
    assert shaped.natural[shaped.grid.at(4, 4)] == 64
    # A later save lowers the terrain; the export still carries the terrain
    # its earthworks were calculated from.
    saved_world(tmp_path / "region", height=55)
    pieces = write_pieces(bricks_at(BlockPos(4, 75, 4)), spec_for(tmp_path), shaped)
    region, _ = load_schematic(pieces[0].path)
    assert region.state_at(BlockPos(5, 64, 5)) == GRASS
    assert region.state_at(BlockPos(5, 60, 5)) == STONE


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
    (pieces / "test_12.schem").write_bytes(b"old")
    (pieces / "other_007.schem").write_bytes(b"not ours")
    (pieces / "test_notes.schem").write_bytes(b"not a piece")
    write_pieces(bricks_at(BlockPos(4, 66, 4)), spec_for(tmp_path), terraforming_with())
    assert sorted(each.name for each in pieces.glob("*.schem")) == [
        "other_007.schem", "test_0.schem", "test_notes.schem",
    ]
