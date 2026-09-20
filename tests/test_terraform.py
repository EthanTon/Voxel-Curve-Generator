import numpy as np
import pytest

from src.Core.Field.block_field import BlockField
from src.Core.block.palette import BlockPalette
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Terraform import (
    Alignment,
    Blend,
    FieldTerrain,
    Follows,
    Grid,
    Ground,
    Intensity,
    Layer,
    Look,
    Plan,
    Slope,
    Span,
    Strata,
    terraform,
)
from src.Terraform import kernels
from src.Terraform.clearance import Clearance, above
from src.Terraform.components import components, reaching
from src.Terraform.distance import nearest
from src.Terraform.layers import Level, Mound, shape
from src.Terraform.plan import Keepout, load as read_plan
from src.Terraform.surroundings import measure
from src.Terraform.terrain import ABSENT, EmptyTerrain, read_surface


STONE = BlockState.create("minecraft:stone")
GRASS = BlockState.create("minecraft:grass_block")
OAK = BlockState.create("minecraft:oak_log")
LEAVES = BlockState.create("minecraft:oak_leaves")


def ground_at(height):
    palette = BlockPalette()
    blocks = BlockField(palette)
    for x in range(-40, 60):
        for z in range(-40, 60):
            top = height(x, z)
            for y in range(top - 6, top + 1):
                blocks.set_block(BlockPos(x, y, z), GRASS if y == top else STONE)
    return FieldTerrain(blocks), palette


def straight_track(level=70, length=20, half_width=2):
    blocks = []
    stations = []
    for step in range(length):
        for z in range(-half_width, half_width + 1):
            blocks.append(BlockPos(step, level, z))
            stations.append(step)
    return Alignment.of(blocks, stations)


def test_box_of_a_constant_field_is_that_constant():
    field = np.full((9, 9), 4.0)
    assert kernels.box(field, 3) == pytest.approx(field)


def test_gaussian_matches_a_reference_kernel():
    field = np.zeros((41, 41))
    field[20, 20] = 1.0
    spread = kernels.gaussian(field, 4.0)
    axis = np.arange(41) - 20
    reference = np.exp(-(axis**2) / 32.0)
    reference = np.outer(reference, reference)
    reference /= reference.sum()
    assert np.abs(spread - reference).max() < 5e-4


def test_smooth_max_is_maximum_without_softness():
    first = np.array([1.0, 5.0, 3.0])
    second = np.array([4.0, 2.0, 3.0])
    assert kernels.smooth_max(first, second, 0.0) == pytest.approx([4.0, 5.0, 3.0])


def test_smooth_max_rounds_the_join():
    joined = kernels.smooth_max(np.array([3.0]), np.array([3.0]), 2.0)
    assert joined[0] == pytest.approx(3.5)


def test_local_minimum_and_maximum_bound_the_field():
    field = np.random.default_rng(0).normal(size=(12, 12))
    assert (kernels.local_minimum(field, 2) <= field + 1e-9).all()
    assert (kernels.local_maximum(field, 2) >= field - 1e-9).all()


def test_nearest_measures_distance_and_carries_a_field():
    seeds = np.zeros((9, 9), dtype=bool)
    seeds[4, 4] = True
    marks = np.zeros((9, 9))
    marks[4, 4] = 7.0
    found = nearest(seeds, (marks,), 8.0)
    assert found.distance[4, 4] == 0.0
    assert found.distance[4, 0] == pytest.approx(4.0)
    assert found.distance[0, 0] == pytest.approx(4 * 1.41421356)
    assert (found[0] == 7.0).all()


def test_nearest_stops_at_the_reach():
    seeds = np.zeros((20, 20), dtype=bool)
    seeds[0, 0] = True
    assert nearest(seeds, (), 5.0).distance.max() == pytest.approx(5.0)


def test_components_separate_disjoint_clumps():
    left = {BlockPos(0, 0, 0), BlockPos(1, 0, 0)}
    right = {BlockPos(8, 0, 0)}
    assert sorted(components(left | right), key=len) == [frozenset(right), frozenset(left)]


def test_reaching_follows_faces_only():
    cells = {BlockPos(0, 0, 0), BlockPos(1, 1, 0)}
    assert reaching(cells, [BlockPos(0, 0, 0)]) == frozenset({BlockPos(0, 0, 0)})


def test_grid_round_trips_a_column():
    grid = Grid(ColumnPos(-5, 3), 10, 10)
    row, cell = grid.at(0, 8)
    assert grid.column(row, cell) == ColumnPos(0, 8)


def test_shape_without_layers_keeps_the_ground():
    grid = Grid(ColumnPos(0, 0), 8, 8)
    natural = grid.filled(64.0)
    ground = Ground(
        grid, natural, grid.filled(3.0), grid.filled(70.0), grid.filled(70.0),
        grid.filled(0.0),
    )
    assert shape(ground, []) == pytest.approx(natural)


def test_a_raising_slope_never_lowers_the_ground():
    grid = Grid(ColumnPos(0, 0), 16, 16)
    natural = grid.filled(64.0)
    ground = Ground(
        grid, natural, np.tile(np.arange(16.0), (16, 1)), grid.filled(70.0),
        grid.filled(70.0), grid.filled(0.0),
    )
    shaped = shape(ground, [Layer(Slope(2.0), Blend.RAISE)])
    assert (shaped >= natural - 1e-9).all()


def test_a_lowering_slope_never_raises_the_ground():
    grid = Grid(ColumnPos(0, 0), 16, 16)
    natural = grid.filled(80.0)
    ground = Ground(
        grid, natural, np.tile(np.arange(16.0), (16, 1)), grid.filled(64.0),
        grid.filled(64.0), grid.filled(0.0),
    )
    shaped = shape(ground, [Layer(Slope(2.0, rising=True), Blend.LOWER)])
    assert (shaped <= natural + 1e-9).all()


def test_a_masked_layer_leaves_unmasked_columns_alone():
    grid = Grid(ColumnPos(0, 0), 12, 12)
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, grid.filled(1.0), grid.filled(70.0), grid.filled(70.0),
        grid.filled(0.0),
    )
    mask = grid.filled(0.0)
    mask[:, :6] = 1.0
    shaped = shape(ground, [Layer(Level(), Blend.SET, mask=mask)])
    assert shaped[:, 6:] == pytest.approx(natural[:, 6:])
    assert shaped[:, :6] == pytest.approx(grid.filled(70.0)[:, :6])


def test_layers_apply_in_order():
    grid = Grid(ColumnPos(0, 0), 6, 6)
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, grid.filled(0.0), grid.filled(70.0), grid.filled(70.0),
        grid.filled(0.0),
    )
    stack = [
        Layer(Level(), Blend.SET, order=0),
        Layer(Mound(4.0, 8.0), Blend.RAISE, order=1),
    ]
    assert shape(ground, stack).max() == pytest.approx(74.0, abs=0.5)


def test_measure_reads_a_flat_plain_as_smooth():
    height = np.full((30, 30), 64, dtype=np.int32)
    found = measure(height, np.ones((30, 30), dtype=bool))
    assert found.roughness == pytest.approx(0.0, abs=1e-9)
    assert found.fill_ratio == pytest.approx(3.0)


def test_measure_reads_a_steep_slope_as_steep():
    height = np.tile(np.arange(30) * 2, (30, 1)).astype(np.int32)
    found = measure(height, np.ones((30, 30), dtype=bool))
    assert found.repose > 1.0
    assert found.fill_ratio == pytest.approx(1.5)


def test_an_ungenerated_world_is_never_terraformed():
    alignment = straight_track()
    result = terraform(
        Plan(default=Intensity.SHAPE, margin=8, reach=12.0),
        EmptyTerrain(),
        alignment,
        BlockPalette(),
    )
    assert len(result.fill) == 0
    assert result.cut == frozenset()
    assert result.report.ungenerated == result.grid.width * result.grid.depth


def test_an_embankment_fills_a_valley_and_leaves_the_far_field_alone():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70)
    result = terraform(
        Plan(look=Look.ENGINEERED, default=Intensity.SHAPE, margin=30, reach=24.0),
        terrain,
        alignment,
        palette,
    )
    assert len(result.fill) > 0
    assert result.cut == frozenset()
    far = result.grid.at(-25, -25)
    assert result.relief[far] == pytest.approx(0.0)


def test_a_cutting_carves_a_hill_and_never_fills():
    terrain, palette = ground_at(lambda x, z: 80)
    alignment = straight_track(level=64)
    result = terraform(
        Plan(look=Look.ENGINEERED, default=Intensity.SHAPE, margin=30, reach=24.0),
        terrain,
        alignment,
        palette,
    )
    assert len(result.cut) > 0
    assert len(result.fill) == 0


def test_terraforming_never_enters_the_clearance():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70)
    clearance = above(
        Grid.around(alignment.columns, 30), alignment.crest_blocks(), 8
    )
    result = terraform(
        Plan(default=Intensity.SHAPE, margin=30, reach=24.0),
        terrain,
        alignment,
        palette,
        clearance=clearance,
    )
    for position, _ in result.fill.iter_block_ids():
        row, cell = result.grid.at(position.x, position.z)
        assert not clearance.holds(row, cell, position.y)


def test_a_keepout_suppresses_its_box():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70)
    plan = Plan(
        default=Intensity.SHAPE,
        margin=24,
        reach=20.0,
        keepouts=(Keepout(ColumnPos(4, 4), ColumnPos(12, 12), "village"),),
    )
    result = terraform(plan, terrain, alignment, palette)
    for position, _ in result.fill.iter_block_ids():
        assert not (4 <= position.x <= 12 and 4 <= position.z <= 12)


def test_a_span_scopes_shaping_to_its_stations():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70, length=40)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(9, 0), profile="embankment"),),
    )
    result = terraform(plan, terrain, alignment, palette)
    assert len(result.fill) > 0
    for position, _ in result.fill.iter_block_ids():
        assert position.x < 30


def test_no_terraforming_is_the_default():
    terrain, palette = ground_at(lambda x, z: 60)
    result = terraform(
        Plan(margin=20, reach=16.0), terrain, straight_track(level=70), palette
    )
    assert len(result.fill) == 0


def test_engineered_slopes_hold_their_batter():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70, length=30)
    plan = Plan(
        look=Look.ENGINEERED,
        default=Intensity.SHAPE,
        margin=30,
        reach=24.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="embankment"),),
    )
    result = terraform(plan, terrain, alignment, palette)
    row = result.grid.at(15, 0)[0]
    profile = [
        result.designed[row, result.grid.at(15, z)[1]]
        for z in range(3, 12)
    ]
    steps = np.diff(profile)
    assert (steps <= 0).all()
    assert steps.min() > -1.2


def test_trees_in_the_corridor_are_cleared_whole():
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = terrain.blocks
    trunk = [BlockPos(6, y, 8) for y in range(61, 66)]
    canopy = [
        BlockPos(6 + dx, 66, 8 + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for position in trunk:
        blocks.set_block(position, OAK)
    for position in canopy:
        blocks.set_block(position, LEAVES)
    grown = FieldTerrain(blocks)
    plan = Plan(default=Intensity.SHAPE, margin=24, reach=20.0)
    result = terraform(plan, grown, straight_track(level=70), palette)
    assert set(trunk) <= result.cut
    assert set(canopy) <= result.cut


def test_foliage_can_be_left_alone():
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = terrain.blocks
    trunk = BlockPos(6, 61, 8)
    blocks.set_block(trunk, OAK)
    grown = FieldTerrain(blocks)
    plan = Plan(
        default=Intensity.SHAPE,
        margin=24,
        reach=20.0,
        foliage=__import__(
            "src.Terraform.plan", fromlist=["Foliage"]
        ).Foliage(clear=False),
    )
    result = terraform(plan, grown, straight_track(level=70), palette)
    assert trunk not in result.cut


def test_read_surface_skips_ungenerated_chunks():
    grid = Grid(ColumnPos(0, 0), 4, 4)
    surface = read_surface(EmptyTerrain(), grid)
    assert not surface.available.any()
    assert (surface.height == ABSENT).all()


def test_strata_layer_by_depth():
    strata = Strata(GRASS, BlockState.create("minecraft:dirt"), STONE, depth=2)
    assert strata.at(0) == GRASS
    assert strata.at(2) == BlockState.create("minecraft:dirt")
    assert strata.at(5) == STONE


def test_a_plan_reads_from_its_own_file(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text(
        """
[terraform]
look = "engineered"
domain = "earth"
default = "shape"
reach = 18.0

[terraform.strata]
surface = "minecraft:podzol"
depth = 2

[[span]]
from = [0, 0]
to = [9, 0]
profile = "embankment"

[[keep_out]]
box = [[4, 4], [8, 8]]
reason = "orchard"

[profile.apron]
action = "slope_down"
ratio = 2.5
"""
    )
    plan = read_plan(written)
    assert plan.look is Look.ENGINEERED
    assert plan.domain == "earth"
    assert plan.strata.surface == BlockState.create("minecraft:podzol")
    assert plan.strata.depth == 2
    assert plan.spans[0].profile == "embankment"
    assert plan.keepouts[0].reason == "orchard"
    assert plan.profile("apron").steps[0].ratio == 2.5
    assert plan.profile("bridge").steps[0].blend is Blend.RESTORE


def test_stretches_cover_everything_when_no_span_is_written():
    plan = Plan(default=Intensity.CLEAR)
    stretched = plan.stretches(lambda where: where, 20)
    assert stretched[0].start == 0
    assert stretched[0].stop == 19
    assert stretched[0].intensity is Intensity.CLEAR


def test_an_unknown_profile_names_the_known_ones():
    with pytest.raises(KeyError, match="embankment"):
        Plan().profile("viaduct")


def test_terraforming_is_stable_across_runs():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70)
    plan = Plan(default=Intensity.SHAPE, margin=24, reach=20.0)
    first = terraform(plan, terrain, alignment, BlockPalette())
    second = terraform(plan, terrain, alignment, BlockPalette())
    assert first.designed == pytest.approx(second.designed)
    assert first.cut == second.cut


def piers(columns, base, top):
    return [
        BlockPos(x, y, z)
        for x in columns
        for z in (-1, 0, 1)
        for y in range(base, top + 1)
    ]


def test_a_follows_layer_levels_a_pad_at_each_placement():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=74, length=30)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
        follows=(Follows(fitting="pier", profile="pad", reach=4.0),),
    )
    result = terraform(
        plan, terrain, alignment, palette, fittings={"pier": piers((10, 20), 57, 73)}
    )
    for x in (10, 20):
        row, cell = result.grid.at(x, 0)
        assert result.designed[row, cell] == pytest.approx(57.0, abs=0.5)
    away = result.grid.at(15, 0)
    assert result.relief[away] == pytest.approx(0.0)


def test_a_bridge_span_leaves_the_valley_alone_without_fittings():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=74, length=30)
    plan = Plan(
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
    )
    result = terraform(plan, terrain, alignment, palette)
    assert len(result.fill) == 0
    assert result.cut == frozenset()


def test_a_follows_layer_needs_its_fitting_to_be_placed():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=74, length=30)
    plan = Plan(
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
        follows=(Follows(fitting="pier", profile="footing"),),
    )
    result = terraform(plan, terrain, alignment, palette, fittings={"pier": []})
    assert result.report.followed == ()
    assert len(result.fill) == 0


def test_a_footing_aprons_out_from_its_pad():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=74, length=30)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
        follows=(Follows(fitting="pier", profile="footing", reach=8.0),),
    )
    result = terraform(
        plan, terrain, alignment, palette, fittings={"pier": piers((14,), 63, 73)}
    )
    assert result.designed[result.grid.at(14, 0)] == pytest.approx(63.0, abs=0.5)
    nearby = result.designed[result.grid.at(14, 6)]
    assert 60.0 < nearby < 63.0
    assert result.relief[result.grid.at(14, 14)] == pytest.approx(0.0)


def test_follows_layers_order_after_spans():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70, length=30)
    shaped = Plan(
        look=Look.ENGINEERED,
        default=Intensity.SHAPE,
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="embankment"),),
        follows=(Follows(fitting="hut", profile="pad", reach=3.0, order=20),),
    )
    result = terraform(
        shaped, terrain, alignment, palette, fittings={"hut": piers((15,), 66, 68)}
    )
    row, cell = result.grid.at(15, 0)
    assert result.designed[row, cell] == pytest.approx(66.0, abs=0.5)


def test_a_plan_reads_follows_from_its_file(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text(
        """
[terraform]
default = "shape"

[[follows]]
fitting = "pillar"
profile = "footing"
reach = 9.0
order = 30
"""
    )
    plan = read_plan(written)
    assert plan.follows[0].fitting == "pillar"
    assert plan.follows[0].profile == "footing"
    assert plan.follows[0].reach == 9.0
    assert plan.follows[0].order == 30


def test_a_follows_without_a_fitting_is_refused(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text("[terraform]\n\n[[follows]]\nprofile = \"footing\"\n")
    with pytest.raises(ValueError, match="fitting"):
        read_plan(written)


def test_a_hard_feather_holds_full_weight_to_its_reach():
    grid = Grid(ColumnPos(0, 0), 16, 1)
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, np.arange(16.0)[None, :], grid.filled(70.0),
        grid.filled(70.0), grid.filled(0.0),
    )
    shaped = shape(
        ground, [Layer(Level(), Blend.SET, reach=5.0, feather=0.0)]
    )
    assert shaped[0, 4] == pytest.approx(70.0)
    assert shaped[0, 6] == pytest.approx(60.0)


def test_blocks_of_reads_a_placement_footprint():
    from src.Assembly.assemble import blocks_of
    from src.Core.World.block_world import BlockWorld
    from src.Core.block.registry import BlockRegistry

    registry = BlockRegistry()
    world = BlockWorld(registry)
    field = BlockField(registry.palette)
    for y in range(3):
        field.set_block(BlockPos(0, y, 0), STONE)
    placement = world.add_field(field, offset=BlockPos(7, 60, -4))
    assert set(blocks_of(world, [placement])) == {
        BlockPos(7, 60 + y, -4) for y in range(3)
    }


def test_an_anchor_takes_its_pad_level_from_the_lowest_block():
    anchor = Alignment.of(piers((5,), 61, 70))
    assert anchor.footing[(5, 0)] == 61
    assert anchor.crest[(5, 0)] == 70
