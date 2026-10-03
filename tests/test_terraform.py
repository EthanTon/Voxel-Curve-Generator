from dataclasses import replace

import numpy as np
import pytest

from src.Core.Field.block_field import BlockField
from src.Core.block.palette import BlockPalette
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, ColumnPos
from src.Terraform import (
    Alignment,
    Blend,
    CrossSection,
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
    TerrainMask,
    terraform,
)
from src.Terraform import kernels
from src.Terraform.clearance import Clearance, above
from src.Terraform.classify import BlockClass, classify, is_tree
from src.Terraform.components import components, reaching
from src.Terraform.distance import nearest, nearest_euclidean
from src.Terraform.layers import AdaptiveSmooth, GaussianPad, Level, Mound, Smooth, shape
from src.Terraform.materialize import build as build_earthworks
from src.Terraform.plan import Keepout, load as read_plan
from src.Terraform.surroundings import Surroundings, measure
from src.Terraform.terraform import _continuous_keeps, _despeckle, _ease_preserved_bridge_ground, _followers, _generator, _ground, _layers, _taper_edit_edges
from src.Terraform.terrain import ABSENT, EmptyTerrain, load_terrain, survey
from sandbox.terraform_test.generate_route import Station as RouteStation, classify as classify_route, settle as settle_route


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


def test_a_linear_two_to_one_bank_reaches_ground_through_the_shore():
    from src.Terraform.plan import Brush, Embankment
    terrain, palette = ground_at(lambda x, z: 60)
    for x in range(-40, 60):
        for z in range(8, 60):
            for y in range(61, 65):
                terrain.blocks.set_block(BlockPos(x, y, z), BlockState.create("minecraft:water"))
    plan = Plan(
        look=Look.ENGINEERED, default=Intensity.SHAPE, margin=32, reach=32,
        brush=Brush(passes=0), embankment=Embankment(ratio=2, shape="linear"),
    )
    result = terraform(plan, FieldTerrain(terrain.blocks), straight_track(), palette)
    bank = np.array([result.designed[result.grid.at(10, z)] for z in range(2, 23)])
    expected = np.maximum(60, 70 - np.arange(21) / 2)
    assert np.floor(bank + 0.5) == pytest.approx(np.floor(expected + 0.5))
    assert result.fill.get_block(BlockPos(10, 67, 8)) is not None
    assert result.relief[result.grid.at(10, 23)] == 0


def test_a_bridge_approach_can_continue_its_bank_to_the_underwater_bed():
    from src.Terraform.plan import Brush, Embankment
    terrain, palette = ground_at(lambda x, z: 60)
    for x in range(15, 40):
        for z in range(-12, 13):
            for y in range(61, 65):
                terrain.blocks.set_block(BlockPos(x, y, z), BlockState.create("minecraft:water"))
    plan = Plan(
        look=Look.ENGINEERED, margin=32, reach=32, brush=Brush(passes=0),
        embankment=Embankment(ratio=2, shape="linear", water_spill=True),
        spans=(Span(start=(0, 0), stop=(14, 0), profile="embankment"),
               Span(start=(14, 0), stop=(29, 0), profile="bridge")),
    )
    result = terraform(plan, FieldTerrain(terrain.blocks), straight_track(length=30), palette)
    assert np.floor(result.designed[result.grid.at(16, 0)] + 0.5) == 69
    assert result.fill.get_block(BlockPos(16, 69, 0)) is not None
    assert result.relief[result.grid.at(38, 0)] == 0


@pytest.mark.parametrize("shape", ("linear", "natural"))
@pytest.mark.parametrize("rotated", (False, True))
def test_trench_earthworks_stop_at_the_walls_even_when_banks_bleed(shape, rotated):
    from src.Terraform.plan import Embankment, Slopes
    terrain, palette = ground_at(lambda x, z: 60)
    blocks, stations = [], []
    for x in range(40):
        for z in range(-2, 3):
            blocks.append(BlockPos(x, 70, z))
            stations.append(x)
        if 10 <= x <= 24:
            for z in (-2, 2):
                blocks.append(BlockPos(x, 79, z))
                stations.append(x)
    alignment = Alignment.of(blocks, stations)
    if rotated:
        alignment = Alignment.of([BlockPos(p.z, p.y, p.x) for p in blocks], stations)
    def column(x, z):
        return (z, x) if rotated else (x, z)
    plan = Plan(
        look=Look.NATURAL, bleed=True, margin=16, reach=16,
        embankment=Embankment(ratio=2, shape=shape),
        slopes=Slopes(repair_passes=32, cone_passes=32, max_drop=2),
        spans=(Span(start=column(0, 0), stop=column(9, 0), profile="bridge"),
               Span(start=column(10, 0), stop=column(24, 0), profile="retained_cutting"),
               Span(start=column(25, 0), stop=column(39, 0), profile="bridge")),
    )
    result = terraform(plan, terrain, alignment, palette)
    assert result.designed[result.grid.at(*column(17, 4))] > 60
    for x in (7, 9, 25, 28):
        cx, cz = column(x, 4)
        assert result.designed[result.grid.at(cx, cz)] == 60
        assert result.fill.get_block(BlockPos(cx, 61, cz)) is None


def test_an_ordinary_bank_still_bleeds_beyond_its_section():
    from src.Terraform.plan import Embankment
    terrain, palette = ground_at(lambda x, z: 60)
    plan = Plan(
        bleed=True, margin=16, reach=16,
        embankment=Embankment(ratio=2, shape="natural"),
        spans=(Span(start=(0, 0), stop=(14, 0), profile="embankment"),
               Span(start=(15, 0), stop=(29, 0), profile="bridge")),
    )
    result = terraform(plan, terrain, straight_track(length=30), palette)
    assert result.designed[result.grid.at(18, 4)] > 60


def test_preserved_bridge_pillars_keep_the_original_ground_in_their_buffer():
    from src.Terraform.plan import Embankment
    terrain, palette = ground_at(lambda x, z: 60)
    plan = Plan(
        default=Intensity.SHAPE, bleed=True, margin=16, reach=16,
        embankment=Embankment(shape="natural"),
        preserve_fittings=("pier",), fitting_buffer=3,
    )
    result = terraform(plan, terrain, straight_track(), palette,
                       fittings={"pier": [BlockPos(10, 61, 8)]})
    assert result.relief[result.grid.at(5, 8)] > 0
    for x in range(7, 14):
        assert result.designed[result.grid.at(x, 8)] == 60
        assert result.fill.get_block(BlockPos(x, 61, 8)) is None


def test_a_pier_buffer_does_not_remove_support_beneath_the_departing_bank():
    terrain, palette = ground_at(lambda x, z: 60)
    plan = Plan(
        bleed=True, margin=16, reach=16,
        preserve_fittings=("pier",), fitting_buffer=4,
        spans=(Span(start=(0, 0), stop=(14, 0), profile="embankment"),
               Span(start=(15, 0), stop=(29, 0), profile="bridge")),
    )
    result = terraform(plan, terrain, straight_track(length=30), palette,
                       fittings={"pier": [BlockPos(16, 61, 0)]})
    assert result.fill.get_block(BlockPos(14, 69, 0)) is not None
    assert result.relief[result.grid.at(16, 4)] == 0


def test_a_tall_bank_gets_room_to_reach_ground_without_widening_mountain_cuts():
    from src.Terraform.plan import Brush, Embankment
    terrain, palette = ground_at(lambda x, z: 85 if z < -32 else 60)
    plan = Plan(
        look=Look.ENGINEERED, default=Intensity.SHAPE, margin=16, reach=16,
        brush=Brush(passes=0), embankment=Embankment(ratio=2, shape="linear", max_reach=48),
    )
    result = terraform(plan, terrain, straight_track(level=80), palette)
    bank = np.floor([result.designed[result.grid.at(10, z)] + 0.5 for z in range(2, 43)])
    assert bank == pytest.approx(np.floor(np.maximum(60, 80 - np.arange(41) / 2) + 0.5))
    assert result.designed[result.grid.at(10, -35)] == 85
    assert result.coverage[result.grid.at(10, 35)]


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


def test_disk_averages_over_a_circle():
    field = np.zeros((41, 41))
    field[20, 20] = 1.0
    spread = kernels.disk(field, 4.0, "constant")
    assert spread.sum() == pytest.approx(1.0)
    assert spread[20, 24] == pytest.approx(spread[24, 20])
    assert spread[20, 20] == pytest.approx(spread[20, 23])
    assert spread[20, 26] == pytest.approx(0.0, abs=1e-12)
    assert spread[23, 23] < spread[20, 24]
    assert kernels.disk(np.full((9, 11), 4.0), 2.5) == pytest.approx(np.full((9, 11), 4.0))
    assert kernels.disk(field, 0.5) is field


def test_a_square_opening_takes_down_ridges_narrower_than_it():
    field = np.full((9, 12), 60.0)
    field[:, 3] = 64.0
    field[:, 7:9] = 64.0
    opened = kernels.square_opening(field, 2)
    assert opened[4, 3] == 60.0
    assert list(opened[4, 7:9]) == [64.0, 64.0]
    assert np.all(kernels.square_opening(field, 3) == 60.0)
    ramp = np.tile(np.arange(12.0), (9, 1))
    assert np.array_equal(kernels.square_opening(ramp, 3), ramp)


def test_a_square_closing_fills_grooves_narrower_than_it():
    field = np.full((9, 12), 60.0)
    field[:, 3] = 57.0
    field[:, 7:9] = 57.0
    closed = kernels.square_closing(field, 2)
    assert closed[4, 3] == 60.0
    assert list(closed[4, 7:9]) == [57.0, 57.0]
    assert np.all(kernels.square_closing(field, 3) == 60.0)
    ramp = np.tile(np.arange(12.0), (9, 1))
    assert np.array_equal(kernels.square_closing(ramp, 3), ramp)


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


def test_exact_nearest_makes_discrete_circles():
    seeds = np.zeros((21, 21), dtype=bool)
    seeds[10, 10] = True
    found = nearest_euclidean(seeds, (), 9.0)
    assert found.distance[13, 14] == pytest.approx(5.0)
    assert found.distance[10, 15] == pytest.approx(5.0)
    assert found.distance[15, 11] == pytest.approx(np.sqrt(26.0))
    assert found.distance[10, 20] == pytest.approx(9.0)


def test_variable_radius_disks_keep_the_larger_footing_round():
    seeds = np.zeros((45, 45), dtype=bool)
    seeds[22, 10] = seeds[22, 34] = True
    radii = np.zeros_like(seeds, dtype=float)
    radii[22, 10], radii[22, 34] = 18.0, 6.0
    found = nearest_euclidean(seeds, (radii,), 18.0, radii=radii)
    assert found.distance[22, 25] == pytest.approx(15.0)
    assert found[0][22, 25] == pytest.approx(18.0)
    assert found.distance[22, 34] == pytest.approx(0.0)


def test_smoothing_strength_passes_and_local_size():
    grid = Grid(ColumnPos(0, 0), 41, 41)
    impulse = grid.filled(0.0)
    impulse[20, 20] = 10.0
    ground = Ground(grid, grid.filled(0.0), grid.filled(0.0),
                    grid.filled(0.0), grid.filled(0.0), grid.filled(0.0))
    assert Smooth(2.0, strength=0.0, passes=3)(impulse, ground) == pytest.approx(impulse)
    one = Smooth(2.0, strength=0.8, passes=1)(impulse, ground)
    many = Smooth(2.0, strength=0.8, passes=3)(impulse, ground)
    adaptive = AdaptiveSmooth(1.0, 1.0, 1, 0.75, 3.5, 1.0)(impulse, ground)
    assert many[20, 20] < one[20, 20] < impulse[20, 20]
    assert adaptive[20, 20] < Smooth(1.0)(impulse, ground)[20, 20]


def test_the_average_kernel_matches_the_gaussian_spread():
    grid = Grid(ColumnPos(0, 0), 41, 41)
    impulse = grid.filled(0.0)
    impulse[20, 20] = 1.0
    ground = Ground(grid, grid.filled(0.0), grid.filled(0.0),
                    grid.filled(0.0), grid.filled(0.0), grid.filled(0.0))
    axis = (np.arange(41) - 20) ** 2

    def variance(smooth):
        return float((smooth(impulse, ground).sum(axis=0) * axis).sum())

    gaussian = variance(Smooth(2.0))
    average = variance(Smooth(2.0, kernel="average"))
    both = variance(Smooth(2.0, kernel="both"))
    assert average == pytest.approx(gaussian, rel=0.3)
    assert both == pytest.approx(gaussian + average, rel=0.05)
    quiet = AdaptiveSmooth(2.0, kernel="average").apply(
        impulse, ground, grid.filled(0.0), "constant"
    )
    assert quiet == pytest.approx(Smooth(2.0, kernel="average")(impulse, ground))


def test_adaptive_smoothing_does_not_use_a_distant_maximum_scale():
    grid = Grid(ColumnPos(0, 0), 81, 81)
    impulse = grid.filled(0.0)
    impulse[40, 40] = 10.0
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, grid.filled(0.0), natural, natural, grid.filled(0.0),
    )
    adaptive = AdaptiveSmooth(2.0, 1.0, 1, 1.0, 20.0, 1.0)

    quiet = adaptive.apply(impulse, ground, grid.filled(0.0), "constant")
    active = adaptive.apply(impulse, ground, grid.filled(6.0), "constant")

    assert quiet == pytest.approx(Smooth(2.0)(impulse, ground))
    assert active == pytest.approx(Smooth(4.0)(impulse, ground))


def test_tall_uneven_fill_uses_wider_embankment_ratio():
    grid = Grid(ColumnPos(0, 0), 9, 9)
    natural = grid.filled(60.0)
    natural[4, 4] = 58.0
    distance = grid.filled(8.0)
    ground = Ground(grid, natural, distance, grid.filled(70.0),
                    grid.filled(70.0), grid.filled(0.0))
    fixed = Slope(1.5)(ground, natural)
    adaptive = Slope(1.5, height_response=0.06, terrain_response=0.2)(ground, natural)
    assert adaptive[4, 4] > fixed[4, 4]
    assert adaptive[0, 0] > fixed[0, 0]


def test_mixed_cross_section_cuts_the_hill_and_fills_the_opposite_side():
    terrain, palette = ground_at(lambda x, z: 80 if z < 0 else 60)
    plan = Plan(
        look=Look.ENGINEERED, margin=24, reach=18.0,
        spans=(Span((0, 0), (19, 0), cross_section="hillside"),),
        cross_sections={"hillside": CrossSection("hillside", bridge_threshold=20.0)},
    )
    result = terraform(plan, terrain, straight_track(level=70), palette)
    assert result.relief[result.grid.at(10, -8)] < -1
    assert result.relief[result.grid.at(10, 8)] > 1


def test_a_mixed_sections_walled_cut_side_holds_the_ground_at_its_wall_top():
    terrain, palette = ground_at(lambda x, z: 90 if z < 0 else 60)
    blocks, stations = [], []
    for x in range(20):
        for z in range(-2, 3):
            blocks.append(BlockPos(x, 70, z))
            stations.append(x)
        for y in range(70, 76):
            blocks.append(BlockPos(x, y, -3))
            stations.append(x)
    plan = Plan(
        margin=24, reach=18.0,
        spans=(Span((0, 0), (19, 0), cross_section="hillside"),),
        cross_sections={"hillside": CrossSection(
            "hillside", left="cutting", right="bridge", cut_base="crest")},
    )
    result = terraform(plan, terrain, Alignment.of(blocks, stations), palette)
    assert 74 <= result.designed[result.grid.at(10, -4)] <= 76


def test_a_railing_on_a_wall_is_left_out_of_its_crest():
    wall = [BlockPos(0, y, 0) for y in range(10)]
    slabs = [BlockPos(1, y, 0) for y in (0, 1)]
    alignment = Alignment.of(
        wall + slabs, [0] * 12,
        decoration=[y >= 7 for y in range(10)] + [True, True],
    )
    assert alignment.crest[(0, 0)] == 6
    assert alignment.crest[(1, 0)] == 1
    assert alignment.footing[(0, 0)] == 0


def test_mixed_sides_split_at_the_path_not_the_middle_of_an_uneven_section():
    from dataclasses import replace
    from src.Terraform.cross_section import axes
    # 3 columns left of a path along x at z = 0, and 7 to its right.
    blocks = [BlockPos(x, 70, z) for x in range(10) for z in range(-3, 8)]
    alignment = Alignment.of(blocks, [block.x for block in blocks])
    assert axes(alignment).z[5] == pytest.approx(2.0)
    centred = replace(alignment, centres={x: (x, 0) for x in range(10)})
    assert axes(centred).z[5] == 0


def test_mixed_cross_section_can_leave_one_side_open_for_a_bridge():
    terrain, palette = ground_at(lambda x, z: 80 if z < 0 else 60)
    plan = Plan(
        look=Look.ENGINEERED, margin=24, reach=18.0,
        spans=(Span((0, 0), (19, 0), cross_section="hillside"),),
        cross_sections={
            "hillside": CrossSection("hillside", left="cutting", right="bridge")
        },
    )
    result = terraform(plan, terrain, straight_track(level=70), palette)
    assert result.relief[result.grid.at(10, -8)] < -1
    assert result.relief[result.grid.at(10, 8)] == pytest.approx(0.0)


def test_route_selector_picks_a_thin_bridge_low_and_a_thick_one_high():
    class Valley:
        def at(self, x, z):
            return (60, False) if x < 40 else (40, False)

    points = [RouteStation((x, 70, 0), (1.0, 0.0, 0.0), float(x))
              for x in range(80)]
    kinds = classify_route(points, Valley())
    assert kinds[10] == "bridge_thin"
    assert kinds[70] == "bridge_thick"


def test_short_water_bridge_at_route_end_is_not_folded_into_embankment():
    kinds = settle_route(["embankment"] * 30 + ["bridge"] * 3)
    assert kinds[-1] == "bridge"
    assert kinds[-5] == "bridge"
    assert kinds[0] == "embankment"


def test_terrain_mask_keeps_water_and_existing_structures_untouched():
    terrain, palette = ground_at(lambda x, z: 60)
    for y in range(61, 65):
        terrain.blocks.set_block(BlockPos(10, y, 8), BlockState.create("minecraft:water"))
    terrain.blocks.set_block(BlockPos(12, 61, -8), BlockState.create("minecraft:stone_bricks"))
    masked = FieldTerrain(terrain.blocks)
    plan = Plan(
        look=Look.ENGINEERED, default=Intensity.SHAPE, margin=24, reach=18.0,
        terrain_mask=TerrainMask(
            preserve_water=True, water_buffer=1, preserve_structures=True
        ),
    )
    result = terraform(plan, masked, straight_track(level=70), palette)
    for x, z in ((10, 8), (12, -8)):
        assert result.relief[result.grid.at(x, z)] == pytest.approx(0.0)
        assert not any(p.x == x and p.z == z for p in result.cut)
        assert not any(p.x == x and p.z == z for p, _ in result.fill.iter_block_ids())
    # The dry shore beside the water may carry a bank down to it, but no cut.
    assert result.relief[result.grid.at(10, 7)] >= 0.0
    assert not any(p.x == 10 and p.z == 7 for p in result.cut)


@pytest.mark.parametrize("bank_height", [60, 68])
def test_water_bank_slopes_to_the_water_surface_without_editing_the_river(bank_height):
    terrain, palette = ground_at(lambda x, z: 60 if z >= 10 else bank_height)
    for x in range(-40, 60):
        for z in range(10, 60):
            for y in range(61, 65):
                terrain.blocks.set_block(BlockPos(x, y, z), BlockState.create("minecraft:water"))
    terrain.blocks.set_block(BlockPos(20, bank_height + 1, 9), BlockState.create("minecraft:stone_bricks"))
    plan = Plan(
        look=Look.MINIMAL, default=Intensity.SHAPE, margin=16, reach=8.0,
        terrain_mask=TerrainMask(
            preserve_water=True, water_buffer=1, preserve_structures=True, slope_to_water=True,
        ),
    )
    alignment = straight_track(level=70)
    result = terraform(plan, FieldTerrain(terrain.blocks), alignment, palette)
    bank = np.array([result.designed[result.grid.at(10, z)] for z in range(2, 10)])
    assert np.floor(bank[-1] + 0.5) == 64  # The water surface, not its bed at 60.
    assert np.all(np.diff(np.floor(bank + 0.5)) <= 0)
    assert np.max(np.abs(np.diff(np.floor(bank + 0.5)))) <= 1
    for z in range(10, 19):
        assert result.relief[result.grid.at(10, z)] == 0
    assert result.relief[result.grid.at(20, 9)] == 0
    # The taper also meets protected high ground without cutting a cliff
    # immediately beside it just to reach the water.
    assert np.floor(result.designed[result.grid.at(19, 9)] + 0.5) >= bank_height - 1
    assert all(result.designed[result.grid.at(x, z)] == height for (x, z), height in alignment.meets.items())
    edited = [(p.x, p.z) for p in result.cut] + [(p.x, p.z) for p, _ in result.fill.iter_block_ids()]
    assert not any(z >= 10 for x, z in edited)
    assert not any((x, z) == (20, 9) for x, z in edited)
    distance = nearest(result.grid.mark(alignment.columns), (), plan.margin + 1.0).distance
    assert np.array_equal(result.designed[distance > plan.margin], result.natural[distance > plan.margin])


def test_buried_tunnel_clearance_removes_bore_without_opening_surface():
    terrain, palette = ground_at(lambda x, z: 90)
    alignment = straight_track(level=70)
    grid = Grid.around(alignment.columns, 16)
    floor = grid.filled(1 << 20)
    ceiling = grid.filled(-(1 << 20))
    floor[grid.at(10, 0)] = 84
    ceiling[grid.at(10, 0)] = 88
    result = terraform(
        Plan(default=Intensity.NONE, look=Look.ENGINEERED, margin=16),
        terrain, alignment, palette, Clearance(floor, ceiling),
    )
    assert result.designed[result.grid.at(10, 0)] == 90
    assert BlockPos(10, 88, 0) in result.cut
    assert BlockPos(10, 89, 0) not in result.cut


def test_wildflowers_are_growth_rather_than_protected_structures():
    for name in (
        "azure_bluet", "allium", "cornflower", "pink_tulip", "bush",
        "pink_petals", "azalea", "small_dripleaf", "pumpkin", "bee_nest",
    ):
        assert classify(BlockState.create(f"minecraft:{name}")) is BlockClass.VEGETATION
    assert is_tree(BlockState.create("minecraft:bee_nest"))


def test_a_shallow_cut_restores_the_local_surface():
    grid = Grid(ColumnPos(0, 0), 1, 1)
    cut = build_earthworks(
        grid, grid.filled(70.0), grid.filled(69.0), Strata(GRASS, STONE, STONE), BlockPalette(),
    )
    assert cut.fill.get_block(BlockPos(0, 69, 0)) == GRASS


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


def test_margin_preserves_a_mountain_inside_the_alignment_bounding_box():
    terrain, palette = ground_at(lambda x, z: 80 if abs(z - 15) < 2 else 60)
    blocks = [BlockPos(x, 70, z) for x in range(20) for z in (0, 30)]
    alignment = Alignment.of(blocks, [x for x in range(20) for _ in (0, 30)])
    result = terraform(
        Plan(default=Intensity.SHAPE, margin=8, reach=24), terrain, alignment, palette,
    )
    distance = nearest(result.grid.mark(alignment.columns), (), 9).distance
    outside = distance > 8
    np.testing.assert_array_equal(result.designed[outside], result.natural[outside])
    assert result.designed[result.grid.at(10, 15)] == 80
    assert result.designed[result.grid.at(10, 2)] > 60
    assert all(distance[result.grid.at(p.x, p.z)] <= 8 for p, _ in result.fill.iter_block_ids())
    assert all(distance[result.grid.at(p.x, p.z)] <= 8 for p in result.cut)


def test_a_short_range_tapers_inside_its_boundary():
    terrain, palette = ground_at(lambda x, z: 64)
    blocks = [BlockPos(x, 70, z) for x in range(20) for z in (0, 30)]
    alignment = Alignment.of(blocks, [x for x in range(20) for _ in (0, 30)])
    result = terraform(
        Plan(default=Intensity.SHAPE, margin=8, reach=24), terrain, alignment, palette,
    )
    heights = np.floor(result.designed[:, result.grid.at(10, 0)[1]] + 0.5)
    assert np.max(np.abs(np.diff(heights))) <= 1
    assert result.designed[result.grid.at(10, 9)] == 64


def test_a_tall_bleeding_bank_tapers_against_its_unchanged_boundary():
    from src.Terraform.plan import Embankment, Slopes
    terrain, palette = ground_at(lambda x, z: 60)
    result = terraform(Plan(default=Intensity.SHAPE, bleed=True, margin=16, reach=16,
                           embankment=Embankment(ratio=2.2, shape="natural"),
                           slopes=Slopes(max_drop=2)),
                       terrain, straight_track(level=86), palette)
    heights = np.floor(result.designed + .5)
    for axis in (0, 1):
        assert np.max(np.abs(np.diff(heights, axis=axis))) <= 2


@pytest.mark.parametrize("profile", ("at_grade", "retained_cutting"))
@pytest.mark.parametrize("lower_bed", (False, True))
def test_rail_terraforming_supports_protruding_rails_without_filling_bridges(profile, lower_bed):
    from types import SimpleNamespace
    from src.Assembly.earthworks import reshape
    from src.Core.Domain import DomainRegistry
    from src.Core.Path.path import Path
    from src.Core.World.block_world import BlockWorld
    from src.Core.block.registry import BlockRegistry

    registry = BlockRegistry()
    domains = DomainRegistry()
    bed = domains.create("bed")
    rails = domains.create("rail")
    domains.create("terrain")
    world = BlockWorld(registry)
    domains.bind(world)
    bed_field = BlockField(registry.palette)
    rail_field = BlockField(registry.palette)
    for x in range(20):
        bed_field.set_block(BlockPos(x, 70, 0), STONE)
        if lower_bed:
            bed_field.set_block(BlockPos(x, 65, 3), STONE)
        rail_field.set_block(BlockPos(x, 71, 3), BlockState.create("minecraft:rail"))
    world.add_field(bed_field, connectivity_domain=bed.domain_id)
    world.add_field(rail_field, connectivity_domain=rails.domain_id)
    path = Path([(x, 70, 0) for x in range(20)], [(1, 0, 0)] * 20)
    terrain, _ = ground_at(lambda x, z: 60)
    plan = Plan(
        default=Intensity.SHAPE, margin=8, reach=8, include_rails=True,
        spans=(Span(start=(0, 0), stop=(19, 0), profile=profile),),
    )
    spec = SimpleNamespace(terraform=plan, sections=(SimpleNamespace(domain="bed"),))
    result = reshape(spec, world, registry, domains, path, {}, apply=False, terrain=terrain)
    assert result.designed[result.grid.at(10, 3)] == 70
    assert result.fill.get_block(BlockPos(10, 70, 3)) is not None
    assert result.fill.get_block(BlockPos(10, 71, 3)) is None

    spec.terraform = replace(plan, spans=(Span(start=(0, 0), stop=(19, 0), profile="bridge"),))
    bridge = reshape(spec, world, registry, domains, path, {}, apply=False, terrain=terrain)
    assert bridge.designed[bridge.grid.at(10, 3)] == 60
    assert len(bridge.fill) == 0


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
    assert (result.designed <= result.natural).all()
    assert all(position.y < result.natural[result.grid.at(position.x, position.z)]
               for position, _ in result.fill.iter_block_ids())


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


def test_raising_a_surface_over_a_buried_bore_keeps_it_empty():
    grid = Grid.around([ColumnPos(0, 0)], 0)
    clearance = Clearance(grid.filled(71), grid.filled(76))
    works = build_earthworks(grid, grid.filled(69), grid.filled(79),
                            Plan().strata, BlockPalette(), clearance=clearance)
    assert {p.y for p, _ in works.fill.iter_block_ids()} == {70, 77, 78, 79}


def test_a_rock_cut_exposes_saved_stone_without_painting_grass_ledges():
    grid = Grid.around([ColumnPos(0, 0)], 0)
    works = build_earthworks(grid, grid.filled(80), grid.filled(74),
                            Plan().strata, BlockPalette(), rock_surface=grid.filled(77))
    assert works.fill.get_block(BlockPos(0, 74, 0)) is None
    assert BlockPos(0, 75, 0) in works.cut
    soil = build_earthworks(grid, grid.filled(80), grid.filled(79),
                           Plan().strata, BlockPalette(), rock_surface=grid.filled(77))
    assert soil.fill.get_block(BlockPos(0, 79, 0)) == GRASS


def test_zero_section_headroom_cannot_leave_soil_over_a_laid_rail():
    terrain, palette = ground_at(lambda x, z: 79)
    alignment = straight_track(level=74)
    envelope = {(x, z): [74, 75] for x in range(20) for z in range(-2, 3)}
    envelope[10, 0] += [83]
    running = {(x, z): 75 for x in range(20) for z in range(-2, 3)}
    result = terraform(Plan(headroom=12, spans=(
        Span((0, 0), (19, 0), profile="bridge", headroom=0),
    )), terrain, alignment, palette, envelope=envelope, running=running)
    assert result.clearance.floor[result.grid.at(10, 0)] == 76
    assert result.clearance.ceiling[result.grid.at(10, 0)] == 82
    assert result.clearance.ceiling[result.grid.at(9, 0)] == 87
    for x in (9, 10):
        for y in (76, 77, 78, 79):
            assert BlockPos(x, y, 0) in result.cut
            assert result.fill.get_block(BlockPos(x, y, 0)) is None
    assert BlockPos(10, 83, 0) not in result.cut
    assert result.fill.get_block(BlockPos(9, 74, 0)) is None


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


def test_a_survey_skips_ungenerated_chunks():
    grid = Grid(ColumnPos(0, 0), 4, 4)
    found = survey(EmptyTerrain(), grid)
    assert not found.available.any()
    assert (found.height == ABSENT).all()
    assert found.growth == frozenset()
    assert found.intruding == frozenset()


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


def test_adaptive_earthwork_controls_are_read_from_the_file(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text("""
[terraform.smoothing]
mode = "adaptive"
size = 1.75
min_size = 0.8
max_size = 4.0
strength = 0.7
passes = 3
terrain_response = 1.2

[terraform.embankment]
adaptive = true
height_response = 0.1
terrain_response = 0.3
max_ratio = 5.0

[[follows]]
fitting = "mast"
reach_mode = "adaptive"
reach_min = 7.0
reach_max = 24.0
""")
    plan = read_plan(written)
    assert plan.smoothing.mode == "adaptive"
    assert plan.smoothing.kernel == "gaussian"
    assert plan.smoothing.passes == 3
    assert plan.smoothing.strength == pytest.approx(0.7)
    assert plan.embankment.adaptive
    assert plan.embankment.max_ratio == pytest.approx(5.0)
    assert plan.follows[0].reach_mode == "adaptive"
    assert plan.follows[0].reach_max == pytest.approx(24.0)


def test_the_cut_ratio_and_feather_are_read_and_checked(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text("[terraform]\nfeather = 10.0\n\n[terraform.cutting]\nratio = 2.0\n")
    plan = read_plan(written)
    assert plan.cutting.ratio == pytest.approx(2.0)
    assert plan.feather == pytest.approx(10.0)
    written.write_text("[terraform.cutting]\nratio = 0\n")
    with pytest.raises(ValueError, match="cutting.ratio"):
        read_plan(written)


def test_the_cut_ratio_sets_every_cut_slope_without_its_own():
    from src.Terraform import Cutting
    from src.Terraform.plan import Action, Step

    surroundings = Surroundings(0.0, 4.0, 0.0, 0.5)
    plan = Plan(cutting=Cutting(ratio=2.5))
    assert _generator(Step(Action.SLOPE_UP), surroundings, plan=plan).ratio == 2.5
    assert _generator(Step(Action.SLOPE_UP, ratio=1.0), surroundings, plan=plan).ratio == 1.0
    assert _generator(Step(Action.SLOPE_UP), surroundings, plan=Plan()).ratio == surroundings.cut_ratio


def test_the_feather_reaches_every_stretch_layer():
    terrain, _ = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70)
    grid = Grid.around(alignment.columns, 10)
    found = _ground(grid, grid.filled(60.0), alignment, grid.mark(alignment.columns), 20.0)
    plan = Plan(default=Intensity.SHAPE, reach=20.0, feather=8.0)
    stretches = plan.stretches(alignment.locate, alignment.length)
    stack = _layers(
        plan, Surroundings(0.0, 4.0, 0.0, 0.5), stretches, [grid.filled(1.0)], grid,
        found, alignment, grid.filled(60.0), grid.filled(float(ABSENT)), grid.filled(True, dtype=bool),
    )
    assert stack and all(layer.feather == 8.0 for layer in stack)


def test_a_cut_takes_the_lands_surface_and_leaves_its_face_as_it_was():
    dirt = BlockState.create("minecraft:dirt")
    grid = Grid(ColumnPos(0, 0), 3, 1)
    natural = grid.filled(80.0)
    strata = Strata(GRASS, dirt, STONE, depth=2)
    for levels in ([70.0, 71.0, 72.0], [70.0, 74.0, 78.0]):
        designed = grid.filled(0.0)
        designed[0] = levels
        cut = build_earthworks(grid, natural, designed, strata, BlockPalette())
        assert cut.fill.get_block(BlockPos(1, int(levels[1]), 0)) == GRASS
        assert cut.fill.get_block(BlockPos(1, int(levels[1]) - 1, 0)) is None


def test_the_smoothing_kernel_is_read_and_checked(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text('[terraform.smoothing]\nkernel = "average"\n')
    assert read_plan(written).smoothing.kernel == "average"
    written.write_text('[terraform.smoothing]\nkernel = "median"\n')
    with pytest.raises(ValueError, match="smoothing.kernel"):
        read_plan(written)


def test_despeckle_drops_lone_columns_and_keeps_edges():
    natural = np.full((9, 9), 70.0)
    designed = natural.copy()
    designed[:, :4] = 68.0
    designed[6, 7] = 69.0
    designed[2, 4] = 68.0
    designed[4, 3] = 70.0
    known = np.ones_like(natural, dtype=bool)
    everywhere = np.ones_like(known)
    cleaned = _despeckle(designed, natural, known, everywhere, everywhere)
    expected = natural.copy()
    expected[:, :4] = 68.0
    assert cleaned == pytest.approx(expected)


def test_despeckle_only_raises_where_it_may_not_lower():
    natural = np.full((5, 5), 70.0)
    designed = natural.copy()
    designed[2, 2] = 71.0
    designed[1, 1] = 69.0
    known = np.ones_like(natural, dtype=bool)
    cleaned = _despeckle(designed, natural, known, known, ~known)
    assert cleaned[2, 2] == 71.0
    assert cleaned[1, 1] == 70.0


def test_terrain_mask_and_named_cross_section_are_read_from_the_file(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text("""
[terraform]
clear_track = true

[terraform.terrain_mask]
preserve_water = true
water_buffer = 2
preserve_structures = true

[cross_section.hillside]
left = "cutting"
right = "bridge"
cut_base = "crest"

[[span]]
from = [0, 0]
to = [19, 0]
cross_section = "hillside"
""")
    plan = read_plan(written)
    assert plan.clear_track
    assert plan.terrain_mask.water_buffer == 2
    assert plan.terrain_mask.preserve_structures
    assert plan.cross_section("hillside").right == "bridge"
    assert plan.spans[0].cross_section == "hillside"


def test_each_pole_gets_a_reach_from_its_own_ground():
    grid = Grid(ColumnPos(0, 0), 61, 21)
    natural = grid.filled(60.0)
    natural[:, 30:] = 54.0
    anchor = Alignment.of((BlockPos(10, 70, 10), BlockPos(45, 70, 10)))
    plan = Plan(follows=(Follows("mast", reach_mode="adaptive"),))
    layers, covered = _followers(
        plan, Surroundings(0.0, 4.0, 0.0, 0.5), grid, natural,
        {"mast": anchor}, grid.filled(1.0),
    )
    radii = layers[0].ground.reach
    assert radii is not None
    assert radii[10, 45] > radii[10, 10]
    assert covered[10, 10] == covered[10, 45] == 1.0


def test_stretches_cover_everything_when_no_span_is_written():
    plan = Plan(default=Intensity.CLEAR)
    stretched = plan.stretches(lambda where: where, 20)
    assert stretched[0].start == 0
    assert stretched[0].stop == 19
    assert stretched[0].intensity is Intensity.CLEAR


def test_a_span_can_follow_every_assignment_of_a_named_section(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text('[[span]]\nsection = "bank"\nprofile = "embankment"\n')
    plan = read_plan(written)
    stretches = plan.stretches(
        lambda where: where, 30, {"bank": ((0, 9), (20, 29))}
    )
    assert [(each.start, each.stop) for each in stretches] == [(0, 9), (20, 29)]
    assert all(each.profile == "embankment" for each in stretches)

    terrain, palette = ground_at(lambda x, z: 60)
    shaped = terraform(
        Plan(
            margin=24, reach=10.0,
            spans=(Span(section="bank", profile="embankment", transition=0),),
        ),
        terrain, straight_track(level=70, length=30), palette,
        section_spans={"bank": ((0, 9), (20, 29))},
    )
    assert shaped.relief[shaped.grid.at(5, 5)] > 0
    assert shaped.relief[shaped.grid.at(15, 5)] == 0
    assert shaped.relief[shaped.grid.at(25, 5)] > 0


def test_a_section_span_requires_an_assigned_section_and_no_markers():
    with pytest.raises(ValueError, match="either section or from/to"):
        Span(start=(0, 0), stop=(9, 0), section="bank")
    with pytest.raises(ValueError, match="section or both from and to"):
        Span(start=(0, 0))
    with pytest.raises(ValueError, match="no assigned path intervals"):
        Plan(spans=(Span(section="missing"),)).stretches(
            lambda where: where, 10, {"bank": ((0, 9),)}
        )
    assert Plan(spans=(Span(section="unused"),)).stretches(
        lambda where: where, 10, {"unused": ()},
    ) == ()


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


def test_a_bank_fills_under_its_own_track_on_a_dry_shore():
    from src.Terraform.plan import TerrainMask
    terrain, palette = ground_at(lambda x, z: 60)
    for x in range(-40, 60):
        terrain.blocks.set_block(BlockPos(x, 61, 3), BlockState.create("minecraft:water"))
    plan = Plan(
        margin=20, reach=16.0,
        terrain_mask=TerrainMask(preserve_water=True, water_buffer=1),
        spans=(Span(start=(0, 0), stop=(19, 0), profile="embankment"),),
    )
    result = terraform(plan, FieldTerrain(terrain.blocks), straight_track(level=70), palette)
    assert result.designed[result.grid.at(10, 2)] >= 69
    assert result.designed[result.grid.at(10, 3)] == result.natural[result.grid.at(10, 3)]


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


def test_a_natural_bridge_trims_a_crown_reaching_into_its_deck():
    from src.Terraform.plan import Bridge
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = terrain.blocks
    trunk = [BlockPos(10, y, 6) for y in range(61, 70)]
    crown = [BlockPos(10, y, z) for y in range(70, 73) for z in range(3, 10)]
    for position in trunk:
        blocks.set_block(position, OAK)
    for position in crown:
        blocks.set_block(position, LEAVES)
    plan = Plan(
        margin=20,
        reach=16.0,
        bridge=Bridge(natural_only=True),
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
    )
    result = terraform(plan, FieldTerrain(blocks), straight_track(level=70, length=30), palette)
    assert {BlockPos(10, y, 3) for y in range(70, 73)} <= result.cut
    assert not set(trunk) & result.cut
    assert BlockPos(10, 71, 8) not in result.cut


def test_a_natural_bridge_takes_the_ground_down_under_its_deck():
    from src.Terraform.plan import Bridge
    terrain, palette = ground_at(lambda x, z: 74 if z <= 0 else 60)
    plan = Plan(
        margin=20,
        reach=16.0,
        bridge=Bridge(natural_only=True),
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
    )
    result = terraform(plan, terrain, straight_track(level=70, length=30), palette)
    assert result.designed[result.grid.at(10, -1)] <= 69
    assert result.designed[result.grid.at(10, -8)] == pytest.approx(74.0)


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


def test_a_poles_footing_has_circular_contours_without_a_pyramid():
    from src.Terraform.plan import Action

    assert Plan().profile("footing").steps[0].action is Action.CIRCULAR
    terrain, palette = ground_at(lambda x, z: 60)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=20,
        reach=16.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="bridge"),),
        follows=(Follows(fitting="mast", profile="footing", reach=8.0),),
    )
    result = terraform(
        plan, terrain, straight_track(level=74, length=30), palette,
        fittings={"mast": [BlockPos(14, 66, 0)]},
    )
    levels = [result.designed[result.grid.at(14, z)] for z in range(9)]
    assert levels[0] == pytest.approx(66.0)
    assert all(earlier > later for earlier, later in zip(levels[:7], levels[1:7]))
    assert all(earlier >= later for earlier, later in zip(levels, levels[1:]))
    assert levels[-1] == pytest.approx(60.0)
    assert result.designed[result.grid.at(16, 0)] == pytest.approx(levels[2])
    assert result.designed[result.grid.at(17, 4)] == pytest.approx(levels[5])
    assert result.designed[result.grid.at(19, 1)] < levels[5]


def test_adaptive_gaussian_footing_stays_finite_outside_its_seeded_reach():
    grid = Grid(ColumnPos(0, 0), 9, 1)
    natural = grid.filled(60.0)
    distance = np.abs(np.arange(9)[None, :] - 4).astype(float)
    local_reach = np.where(distance <= 3, 3.0, 0.0)
    ground = Ground(grid, natural, distance, grid.filled(68.0),
                    grid.filled(68.0), grid.filled(0.0), local_reach)
    result = shape(ground, (Layer(GaussianPad(2.0), blend=Blend.SET, reach=3.0),))
    assert np.isfinite(result).all()
    assert result[0, 0] == pytest.approx(60.0)


def test_land_mast_footing_stops_at_bridge_deck_but_keeps_its_own_pad():
    terrain, palette = ground_at(lambda x, z: 60)
    plan = Plan(
        look=Look.ENGINEERED, margin=20, reach=16.0,
        spans=(
            Span(start=(0, 0), stop=(14, 0), profile="bridge"),
            Span(start=(14, 0), stop=(29, 0), profile="tunnel",
                 intensity=Intensity.NONE),
        ),
        follows=(Follows(fitting="mast", profile="footing_circular", reach=12.0),),
    )
    result = terraform(
        plan, terrain, straight_track(level=70, length=30), palette,
        fittings={"mast": [BlockPos(16, 70, 8)]},
    )
    # The apron is cut off at the deck, and only graded under it a block a
    # column down from the pad.
    assert result.report.steep == 0
    assert result.designed[result.grid.at(13, 2)] <= 70.0 - 6.0
    assert result.designed[result.grid.at(8, 0)] == 60.0
    assert result.designed[result.grid.at(16, 8)] == pytest.approx(70.0)


def test_each_named_footing_can_be_selected_from_the_plan_file(tmp_path):
    terrain, palette = ground_at(lambda x, z: 60)
    levels = {}
    for name in ("footing_flat_slope", "footing_gaussian", "footing_circular"):
        written = tmp_path / f"{name}.toml"
        written.write_text(
            '[terraform]\nlook = "engineered"\nmargin = 20\nreach = 16\n'
            '\n[[span]]\nfrom = [0, 0]\nto = [29, 0]\nprofile = "bridge"\n'
            f'\n[[follows]]\nfitting = "mast"\nprofile = "{name}"\nreach = 8\n'
        )
        plan = read_plan(written)
        assert plan.follows[0].profile == name
        result = terraform(
            plan, terrain, straight_track(level=74, length=30), palette,
            fittings={"mast": [BlockPos(14, 66, 0)]},
        )
        assert result.designed[result.grid.at(14, 0)] == pytest.approx(66.0, abs=0.5)
        assert result.designed[result.grid.at(14, 8)] == pytest.approx(60.0)
        levels[name] = result.designed[result.grid.at(14, 2)]
    assert len({round(value, 3) for value in levels.values()}) == 3


@pytest.mark.parametrize("action", ["gaussian", "circular"])
def test_a_custom_radial_profile_defaults_to_setting_its_shape(tmp_path, action):
    from src.Terraform.plan import Action

    written = tmp_path / "terraform.toml"
    written.write_text(f'[profile.soft_footing]\naction = "{action}"\nwidth = 2.5\n')
    step = read_plan(written).profile("soft_footing").steps[0]
    assert step.action is Action(action)
    assert step.blend is Blend.SET
    assert step.width == pytest.approx(2.5)


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
    from src.Assembly.earthworks import blocks_of
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


def test_a_fittings_feet_are_the_columns_it_stands_on():
    # A mast's arm hangs over the track: a footing levelled under it built a
    # wall of fill up to the arm.
    from src.Assembly.earthworks import feet_of
    from src.Core.World.block_world import BlockWorld
    from src.Core.block.registry import BlockRegistry

    registry = BlockRegistry()
    world = BlockWorld(registry)
    field = BlockField(registry.palette)
    for y in range(11):
        field.set_block(BlockPos(0, y, 0), STONE)
    for across in range(1, 6):
        field.set_block(BlockPos(-across, 10, 0), STONE)
    placement = world.add_field(field, offset=BlockPos(7, 60, -4))
    assert set(feet_of(world, [placement])) == {
        BlockPos(7, 60 + y, -4) for y in range(11)
    }


def test_a_column_takes_its_station_from_the_frame_that_owns_it():
    # At a handover on a curve a column the outgoing section laid can be
    # nearest the incoming section's first point.
    from types import SimpleNamespace

    from src.Assembly.earthworks import _within_frame

    frame = SimpleNamespace(path_start=10, path_stop=20)
    owned = SimpleNamespace(frame_at=lambda x, z: frame)
    path = SimpleNamespace(last_index=100)
    at = BlockPos(0, 0, 0)
    assert _within_frame(20, at, path, owned) == 19
    assert _within_frame(4, at, path, owned) == 10
    assert _within_frame(15, at, path, owned) == 15
    assert _within_frame(20, at, SimpleNamespace(last_index=20), owned) == 20
    assert _within_frame(7, at, path, None) == 7


def test_an_anchor_takes_its_pad_level_from_the_lowest_block():
    anchor = Alignment.of(piers((5,), 61, 70))
    assert anchor.footing[(5, 0)] == 61
    assert anchor.crest[(5, 0)] == 70


def test_a_world_that_is_not_there_says_so(tmp_path):
    with pytest.raises(ValueError, match="not there"):
        load_terrain(tmp_path / "saves" / "region")


def test_a_world_directory_without_regions_says_so(tmp_path):
    (tmp_path / "saves").mkdir()
    with pytest.raises(ValueError, match="no region files"):
        load_terrain(tmp_path / "saves")


def test_a_save_folder_descends_into_its_region_directory(tmp_path):
    region = tmp_path / "saves" / "region"
    region.mkdir(parents=True)
    (region / "r.0.0.mca").write_bytes(b"")
    loaded = load_terrain(tmp_path / "saves")
    assert loaded.directory == region


def test_a_save_selects_named_and_vanilla_dimensions(tmp_path):
    from src.Terraform.terrain import region_directory

    save = tmp_path / "save"
    for folder, dimension in (
        ("region", "minecraft:overworld"),
        ("DIM-1/region", "minecraft:the_nether"),
        ("DIM1/region", "minecraft:the_end"),
        ("dimensions/minecraft/nova_proxima/region", "minecraft:nova_proxima"),
        ("dimensions/example/moon/region", "example:moon"),
    ):
        region = save / folder
        region.mkdir(parents=True)
        (region / "r.0.0.mca").write_bytes(b"")
        assert region_directory(save, dimension) == region
        assert load_terrain(save, dimension).directory == region


def test_a_missing_or_unsafe_dimension_is_rejected(tmp_path):
    save = tmp_path / "save"
    (save / "region").mkdir(parents=True)
    (save / "region" / "r.0.0.mca").write_bytes(b"")
    with pytest.raises(ValueError, match="no region files"):
        load_terrain(save, "minecraft:missing")
    with pytest.raises(ValueError, match="invalid dimension"):
        load_terrain(save, "example:../region")


def test_an_omitted_world_terraforms_nothing():
    assert isinstance(load_terrain(None), EmptyTerrain)


def test_a_world_path_resolves_against_its_own_file(tmp_path):
    (tmp_path / "terraform").mkdir()
    region = tmp_path / "saves" / "region"
    region.mkdir(parents=True)
    (region / "r.0.0.mca").write_bytes(b"")
    written = tmp_path / "terraform" / "mainline.toml"
    written.write_text('[terraform]\nworld = "../saves/region"\n')
    assert read_plan(written).world.resolve() == region.resolve()


def saved_world(directory, columns, height=64, depth=40):
    from src.Format.anvil import write_region_files
    from src.Format.region import BlockRegion

    least = BlockPos(min(x for x, _ in columns), height - depth, min(z for _, z in columns))
    most = BlockPos(max(x for x, _ in columns), height, max(z for _, z in columns))
    region = BlockRegion("world", least, most - least + BlockPos(1, 1, 1))
    for x, z in columns:
        for y in range(height - depth, height + 1):
            region.set_state(BlockPos(x, y, z), GRASS if y == height else STONE)
    directory.mkdir(parents=True, exist_ok=True)
    write_region_files(region, directory, data_version=3700, merge=False)
    return directory


def test_region_terrain_reads_a_saved_world(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(32) for z in range(32)])
    terrain = RegionTerrain(saved)
    assert terrain.generated(0, 0)
    assert not terrain.generated(40, 40)
    stack = terrain.column(5, 5)
    assert stack[-1][1] == GRASS
    assert stack[-1][0] == 64


def test_region_terrain_evicts_chunks_beyond_its_capacity(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(64) for z in range(64)])
    terrain = RegionTerrain(saved, chunks=2)
    for x in range(0, 64, 16):
        for z in range(0, 64, 16):
            terrain.column(x, z)
    assert terrain.held_chunks == 2


def test_an_evicted_chunk_reads_back_the_same(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(48) for z in range(48)])
    terrain = RegionTerrain(saved, chunks=1)
    first = list(terrain.column(3, 3))
    for x in range(16, 48, 16):
        terrain.column(x, 0)
    assert terrain.held_chunks == 1
    assert list(terrain.column(3, 3)) == first


def test_region_terrain_keeps_only_the_top_levels(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(16) for z in range(16)])
    terrain = RegionTerrain(saved, depth=4)
    stack = terrain.column(2, 2)
    assert len(stack) == 5
    assert stack[0][0] == 60
    assert stack[-1][0] == 64


def test_a_deep_bore_extends_only_its_saved_terrain_columns(tmp_path):
    from src.Terraform.region import RegionTerrain
    saved = saved_world(tmp_path / "region", [(2, 2), (3, 2)], height=180, depth=110)
    terrain = RegionTerrain(saved)
    assert terrain.column(2, 2)[0][0] == 116
    terrain.include_depths({(2, 2): 90})
    assert terrain.column(2, 2)[0][0] == 90
    assert terrain.column(3, 2)[0][0] == 116
    plan = Plan(default=Intensity.NONE, margin=2, headroom=0)
    alignment = Alignment.of([BlockPos(2, 100, 2), BlockPos(2, 110, 2)])
    result = terraform(plan, terrain, alignment, envelope={(2, 2): [100, 110]})
    assert {BlockPos(2, y, 2) for y in range(101, 110)} <= result.cut
    assert BlockPos(2, 111, 2) not in result.cut


def test_presence_is_answered_without_decoding(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(16) for z in range(16)])
    terrain = RegionTerrain(saved)
    assert terrain.generated(0, 0)
    assert terrain.held_chunks == 0


def test_a_missing_chunk_is_remembered_as_missing(tmp_path):
    from src.Terraform.region import RegionTerrain

    saved = saved_world(tmp_path / "region", [(x, z) for x in range(16) for z in range(16)])
    terrain = RegionTerrain(saved)
    assert not terrain.generated(20, 20)
    before = terrain.held_entries
    assert not terrain.generated(20, 20)
    assert terrain.held_entries == before


def test_by_chunk_visits_every_column_once():
    grid = Grid(ColumnPos(-3, 5), 21, 19)
    walked = list(grid.by_chunk(8))
    assert len(walked) == grid.width * grid.depth
    assert {(row, cell) for row, cell, _ in walked} == {
        (row, cell) for row in range(grid.depth) for cell in range(grid.width)
    }
    for row, cell, column in walked:
        assert grid.at(column.x, column.z) == (row, cell)


def test_by_chunk_finishes_a_chunk_before_the_next():
    grid = Grid(ColumnPos(0, 0), 32, 32)
    seen = []
    for _, _, column in grid.by_chunk(16):
        chunk = (column.x // 16, column.z // 16)
        if not seen or seen[-1] != chunk:
            seen.append(chunk)
    assert len(seen) == len(set(seen))


# --- the natural look, stretch transitions and growth --------------------------


def test_a_layers_filters_work_on_its_own_change_alone():
    from src.Terraform.layers import Smooth

    grid = Grid(ColumnPos(0, 0), 16, 16)
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, grid.filled(0.0), grid.filled(70.0), grid.filled(70.0),
        grid.filled(0.0),
    )
    left = grid.filled(0.0)
    left[:, :4] = 1.0
    right = grid.filled(0.0)
    right[:, 12:] = 1.0
    stack = [
        Layer(Level(), Blend.SET, mask=left, order=0),
        Layer(Level(), Blend.SET, mask=right, order=1, filters=(Smooth(1.0),)),
    ]
    shaped = shape(ground, stack)
    assert shaped[:, :3] == pytest.approx(grid.filled(70.0)[:, :3])


def test_the_look_settles_the_earthworks_but_holds_the_formation():
    from src.Terraform.layers import Smooth, settle

    grid = Grid(ColumnPos(0, 0), 16, 16)
    natural = grid.filled(60.0)
    ground = Ground(
        grid, natural, grid.filled(0.0), grid.filled(70.0), grid.filled(70.0),
        grid.filled(0.0),
    )
    designed = grid.filled(60.0)
    designed[:, 6:10] = 70.0
    hold = grid.filled(0.0)
    hold[:, 7:9] = 1.0
    settled = settle(ground, designed, (Smooth(2.0),), hold)
    assert settled[:, 7:9] == pytest.approx(designed[:, 7:9])
    assert settled[8, 6] < 70.0
    assert settled[8, 3] > 60.0


def test_a_deck_step_ramps_out_instead_of_cutting_a_line():
    blocks, stations = [], []
    for step in range(40):
        for z in range(-2, 3):
            blocks.append(BlockPos(step, 70 if step < 20 else 71, z))
            stations.append(step)
    alignment = Alignment.of(blocks, stations)
    grid = Grid.around(alignment.columns, 16)
    found = _ground(grid, grid.filled(60.0), alignment, grid.mark(alignment.columns), 16.0)
    on_track = found.seed_height[grid.at(0, 0)[0], grid.at(19, 0)[1]:grid.at(21, 0)[1] + 1]
    assert on_track == pytest.approx([70.0, 71.0, 71.0])
    row = grid.at(0, 12)[0]
    far = found.seed_height[row, grid.at(10, 0)[1]:grid.at(30, 0)[1]]
    assert far[0] == pytest.approx(70.0) and far[-1] == pytest.approx(71.0)
    assert np.abs(np.diff(far)).max() < 0.25


def test_a_stretch_running_out_still_keeps_the_clearance():
    terrain, palette = ground_at(lambda x, z: 75)
    alignment = straight_track(level=70, length=40)
    clearance = above(Grid.around(alignment.columns, 20), alignment.crest_blocks(), 8)
    plan = Plan(
        margin=20, reach=16.0,
        spans=(Span(ColumnPos(0, 0), ColumnPos(15, 0), profile="at_grade"),),
    )
    result = terraform(plan, terrain, alignment, palette, clearance=clearance)
    height = np.rint(result.designed)
    moved = height != result.natural
    assert moved[result.grid.at(22, 0)]
    assert (height[moved] < clearance.floor[moved]).all()


def test_settling_leaves_untouched_ground_exactly_as_it_was():
    from src.Terraform.layers import Smooth, settle

    grid = Grid(ColumnPos(0, 0), 20, 20)
    natural = 60.0 + (np.indices(grid.shape).sum(axis=0) % 3)
    ground = Ground(grid, natural, grid.filled(0.0), grid.filled(70.0),
                    grid.filled(70.0), grid.filled(0.0))
    designed = natural.copy()
    designed[8:12, 8:12] += 3.0
    settled = settle(ground, designed, (Smooth(1.0),), grid.filled(0.0))
    assert settled[:3, :3] == pytest.approx(natural[:3, :3])


def test_the_grain_never_takes_or_adds_more_than_half_an_earthwork():
    from src.Terraform.layers import Roughen

    grid = Grid(ColumnPos(0, 0), 32, 32)
    ground = Ground(
        grid, grid.filled(60.0), grid.filled(0.0), grid.filled(70.0),
        grid.filled(70.0), grid.filled(0.0),
    )
    relief = np.tile(np.linspace(-3.0, 3.0, 32), (32, 1))
    rough = Roughen(amplitude=4.0, scale=2.0)(relief, ground)
    assert (np.abs(rough - relief) <= np.abs(relief) / 2 + 1e-9).all()
    assert (np.sign(rough) == np.sign(relief)).all()
    assert not np.allclose(rough, relief)


def test_natural_grain_stays_below_one_block_on_tall_earthworks():
    from src.Terraform.layers import Roughen

    grid = Grid(ColumnPos(0, 0), 32, 32)
    ground = Ground(
        grid, grid.filled(60.0), grid.filled(0.0), grid.filled(70.0),
        grid.filled(70.0), grid.filled(0.0),
    )
    relief = grid.filled(12.0)
    textured = Roughen(amplitude=8.0, scale=2.0)(relief, ground)
    assert np.max(np.abs(textured - relief)) <= 0.65 + 1e-9


def test_natural_settling_smooths_the_edited_surface_itself():
    from src.Terraform.layers import Smooth, settle

    grid = Grid(ColumnPos(0, 0), 24, 24)
    natural = 60.0 + 2.0 * (np.indices(grid.shape).sum(axis=0) % 2)
    ground = Ground(
        grid, natural, grid.filled(0.0), grid.filled(70.0),
        grid.filled(70.0), grid.filled(0.0),
    )
    designed = natural.copy()
    designed[4:20, 4:20] += 4.0
    settled = settle(ground, designed, (Smooth(1.5),), grid.filled(0.0))
    assert np.ptp(settled[9:15, 9:15]) < 0.5
    assert settled[0, 0] == pytest.approx(natural[0, 0], abs=0.01)


def test_under_water_a_fill_takes_what_lies_beneath_the_lands_top():
    dirt = BlockState.create("minecraft:dirt")
    water = BlockState.create("minecraft:water")
    grid = Grid(ColumnPos(0, 0), 1, 1)
    strata = Strata(GRASS, dirt, STONE, depth=2)
    drowned = build_earthworks(
        grid, grid.filled(60), grid.filled(63), strata, BlockPalette(), None, grid.filled(64),
    )
    assert [drowned.fill.get_block(BlockPos(0, y, 0)) for y in range(61, 64)] == [dirt, dirt, dirt]
    lowered = build_earthworks(
        grid, grid.filled(60), grid.filled(58), strata, BlockPalette(), None, grid.filled(64),
    )
    assert lowered.cut == frozenset()
    assert lowered.fill.get_block(BlockPos(0, 59, 0)) == water
    protected = build_earthworks(
        grid, grid.filled(60), grid.filled(58), strata, BlockPalette(),
        None, grid.filled(64), Clearance(grid.filled(59), grid.filled(64)),
    )
    assert BlockPos(0, 59, 0) in protected.cut
    assert protected.fill.get_block(BlockPos(0, 59, 0)) is None
    dry = build_earthworks(grid, grid.filled(60), grid.filled(65), strata, BlockPalette())
    assert [dry.fill.get_block(BlockPos(0, y, 0)) for y in range(65, 60, -1)] == [GRASS, dirt, dirt, STONE, STONE]


def test_survey_records_water_above_the_riverbed():
    palette = BlockPalette()
    blocks = BlockField(palette)
    blocks.set_block(BlockPos(0, 60, 0), BlockState.create("minecraft:sand"))
    for y in range(61, 65):
        blocks.set_block(BlockPos(0, y, 0), BlockState.create("minecraft:water"))
    found = survey(FieldTerrain(blocks), Grid(ColumnPos(0, 0), 1, 1))
    assert found.height[0, 0] == 60
    assert found.water[0, 0] == 64
    cleared = survey(
        FieldTerrain(blocks), Grid(ColumnPos(0, 0), 1, 1),
        lambda row, cell, y: y == 64,
    )
    assert BlockPos(0, 64, 0) in cleared.intruding


def test_a_natural_embankment_of_many_spans_carries_its_deck_throughout():
    # Ten stretches used to settle the look ten times over, and every
    # handover between two stretches stood on three quarters of the fill.
    terrain, palette = ground_at(lambda x, z: 60 + (3 * x + 5 * z) % 4)
    alignment = straight_track(level=70, length=40)
    ends = (0, 10, 20, 30, 39)
    plan = Plan(
        margin=24,
        reach=16.0,
        spans=tuple(
            Span(start=(a, 0), stop=(b, 0), profile="embankment")
            for a, b in zip(ends, ends[1:])
        ),
    )
    result = terraform(plan, terrain, alignment, palette)
    for x in range(40):
        for z in (-2, 0, 2):
            assert np.rint(result.designed[result.grid.at(x, z)]) == 70


def test_an_embankment_runs_full_to_a_bridge_and_spills_under_its_first_span():
    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70, length=60)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=24,
        reach=16.0,
        spans=(
            Span(start=(0, 0), stop=(29, 0), profile="embankment"),
            Span(start=(29, 0), stop=(59, 0), profile="bridge"),
        ),
    )
    result = terraform(plan, terrain, alignment, palette)

    def level(x):
        return result.designed[result.grid.at(x, 0)]

    # Ten blocks high at a block per two columns, the spill runs twenty
    # columns under the bridge.
    assert level(28) == pytest.approx(70.0, abs=0.5)
    assert 60.5 < level(32) < 69.5
    assert level(45) == pytest.approx(62.0)
    assert level(52) == pytest.approx(60.0)


def test_embankment_runout_does_not_fill_a_bridge_channel():
    terrain, palette = ground_at(lambda x, z: 60)
    water = BlockState.create("minecraft:water")
    for x in range(30, 51):
        for z in range(-12, 13):
            for y in range(61, 65):
                terrain.blocks.set_block(BlockPos(x, y, z), water)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=24,
        reach=16.0,
        spans=(
            Span(start=(0, 0), stop=(29, 0), profile="embankment"),
            Span(start=(29, 0), stop=(59, 0), profile="bridge"),
        ),
    )
    result = terraform(plan, FieldTerrain(terrain.blocks), straight_track(level=70, length=60), palette)
    assert result.designed[result.grid.at(28, 0)] == pytest.approx(70.0, abs=0.5)
    assert result.designed[result.grid.at(32, 0)] == pytest.approx(60.0)
    assert result.fill.get_block(BlockPos(32, 61, 0)) is None


def test_bridge_section_handover_does_not_restart_terrain_transitions():
    terrain, palette = ground_at(lambda x, z: 60 + (x + 2 * z) % 3)
    alignment = straight_track(level=70, length=60)
    plan = Plan(
        look=Look.MINIMAL,
        margin=24,
        reach=16.0,
        spans=(
            Span(start=(0, 0), stop=(29, 0), profile="embankment"),
            Span(start=(29, 0), stop=(59, 0), profile="bridge"),
        ),
    )
    continuous = terraform(plan, terrain, alignment, palette)
    split = terraform(replace(plan, spans=(
        plan.spans[0],
        Span(start=(29, 0), stop=(38, 0), profile="bridge"),
        Span(start=(38, 0), stop=(59, 0), profile="bridge"),
    )), terrain, alignment, palette)
    # Switching bridge schematics must not introduce a new terrain edge,
    # change the approach spill or open a gap in the protected bridge area.
    np.testing.assert_array_equal(split.designed, continuous.designed)
    assert split.cut == continuous.cut


def test_embankment_underpinning_stops_at_the_incoming_bridge():
    from types import SimpleNamespace
    from src.Terraform.materialize import Earthworks
    from src.Terraform.terraform import _underpin

    terrain, palette = ground_at(lambda x, z: 60)
    alignment = straight_track(level=70, length=40)
    grid = Grid.around(alignment.columns, 4)
    plan = Plan(spans=(
        Span(start=(0, 0), stop=(20, 0), profile="embankment"),
        Span(start=(20, 0), stop=(39, 0), profile="bridge"),
    ))
    earthworks = Earthworks(BlockField(palette), frozenset())
    layering = SimpleNamespace(layers=np.empty((1,)), at=lambda *args: STONE)
    _underpin(plan, terrain, alignment, plan.stretches(alignment.locate, 40),
              earthworks, grid, layering)
    assert earthworks.fill.get_block(BlockPos(19, 69, 0)) == STONE
    # The shared marker belongs to the bridge: only the designed approach
    # spill may enter it, never a wall of support fill down from its deck.
    assert earthworks.fill.get_block(BlockPos(20, 69, 0)) is None


@pytest.mark.parametrize("changes", (
    {"start": (40, 0)},  # A gap is still an end of a bridge.
    {"headroom": 0},
    {"transition": 12},
    {"order": 1},
    {"intensity": Intensity.NONE},
    {"cross_section": "open"},
))
def test_distinct_bridge_terrain_rules_keep_their_boundary(changes):
    plan = Plan(spans=(
        Span(start=(0, 0), stop=(29, 0), profile="bridge"),
        replace(Span(start=(29, 0), stop=(59, 0), profile="bridge"), **changes),
    ))
    stretches = plan.stretches(lambda where: where[0], 60)
    assert _continuous_keeps(plan, stretches) == stretches


def test_a_retained_cutting_is_cut_back_from_the_top_of_its_walls():
    terrain, palette = ground_at(lambda x, z: 90)
    blocks, stations = [], []
    for x in range(30):
        for z in range(-3, 4):
            top = 76 if abs(z) == 3 else 70
            for y in range(70, top + 1):
                blocks.append(BlockPos(x, y, z))
                stations.append(x)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=16,
        reach=12.0,
        spans=(Span(start=(0, 0), stop=(29, 0), profile="retained_cutting"),),
    )
    result = terraform(plan, terrain, Alignment.of(blocks, stations), palette)
    assert 76.0 <= result.designed[result.grid.at(15, 4)] < 80.0
    assert result.designed[result.grid.at(15, 0)] == pytest.approx(70.0, abs=0.5)


@pytest.mark.parametrize("bleed", (False, True))
def test_a_tunnel_clears_its_bore_and_keeps_the_mountain_over_its_roof(bleed):
    terrain, palette = ground_at(lambda x, z: 95)
    alignment = straight_track(level=70, length=30)
    envelope = {(x, z): [70, 80] for x in range(30) for z in range(-2, 3)}
    plan = Plan(
        margin=16,
        reach=12.0,
        headroom=12,
        bleed=bleed,
        spans=(
            Span(
                start=(0, 0), stop=(29, 0), intensity=Intensity.NONE,
                profile="tunnel", headroom=0,
            ),
        ),
    )
    result = terraform(plan, terrain, alignment, palette, envelope=envelope)
    row, cell = result.grid.at(15, 0)
    assert result.designed[row, cell] == pytest.approx(95.0)
    assert (result.clearance.floor[row, cell], result.clearance.ceiling[row, cell]) == (71, 79)
    assert not any(position.y > 80 for position in result.cut)


def test_a_bleeding_bank_near_a_tunnel_does_not_remove_its_roof():
    from src.Terraform.plan import Smoothing
    terrain, palette = ground_at(lambda x, z: 95)
    for x in range(10, 30):
        for z in range(-2, 3):
            for y in range(71, 89):
                terrain.blocks.set_block(BlockPos(x, y, z), STONE)
    terrain = FieldTerrain(terrain.blocks)
    alignment = straight_track(level=70, length=30)
    envelope = {(x, z): [70] if x < 10 else [70, 80]
                for x in range(30) for z in range(-2, 3)}
    plan = Plan(
        bleed=True, margin=16, reach=16,
        # This regression checks a held mountain without portal grading.
        smoothing=Smoothing(transition_reach=0),
        spans=(Span(start=(0, 0), stop=(9, 0), profile="embankment"),
               Span(start=(10, 0), stop=(29, 0), profile="tunnel",
                    intensity=Intensity.NONE, headroom=0)),
    )
    result = terraform(plan, terrain, alignment, palette, envelope=envelope)
    assert result.designed[result.grid.at(15, 0)] == 95
    assert BlockPos(15, 75, 0) in result.cut
    assert BlockPos(15, 85, 0) not in result.cut


def test_the_first_point_out_of_a_tunnel_takes_the_open_tracks_headroom():
    terrain, palette = ground_at(lambda x, z: 95)
    alignment = straight_track(level=70, length=30)
    envelope = {
        (x, z): [70, 80] if x < 15 else [70]
        for x in range(30) for z in range(-2, 3)
    }
    plan = Plan(
        margin=16,
        reach=12.0,
        headroom=12,
        spans=(
            Span(
                start=(0, 0), stop=(15, 0), intensity=Intensity.NONE,
                profile="tunnel", headroom=0,
            ),
            Span(start=(15, 0), stop=(29, 0), intensity=Intensity.NONE),
        ),
    )
    result = terraform(plan, terrain, alignment, palette, envelope=envelope)
    row, cell = result.grid.at(15, 0)
    assert (result.clearance.floor[row, cell], result.clearance.ceiling[row, cell]) == (71, 82)


def test_the_space_a_build_encloses_is_cleared_but_not_what_holds_up_its_deck():
    from src.Terraform.clearance import OPEN, enclosing

    grid = Grid(ColumnPos(0, 0), 4, 1)
    envelope = {
        (0, 0): [70, 80],
        (1, 0): [70],
        (2, 0): [60, 61, 62, 64],
        (3, 0): [70, 71, 72],
    }
    footing = {(0, 0): 70, (1, 0): 70, (2, 0): 64}
    cleared = enclosing(
        grid, envelope, lambda x, z: 0 if x in (0, 3) else 5, footing
    )
    assert (cleared.floor[0, 0], cleared.ceiling[0, 0]) == (71, 79)
    assert (cleared.floor[0, 1], cleared.ceiling[0, 1]) == (71, 75)
    assert (cleared.floor[0, 2], cleared.ceiling[0, 2]) == (65, 69)
    assert cleared.floor[0, 3] == OPEN


TUFT = BlockState.create("minecraft:short_grass")


def meadow(terrain):
    blocks = terrain.blocks
    for x in range(-30, 50):
        for z in range(-30, 40):
            blocks.set_block(BlockPos(x, 61, z), TUFT)
    return blocks


def test_a_meadow_is_cleared_only_where_the_ground_changes():
    terrain, palette = ground_at(lambda x, z: 60)
    grown = FieldTerrain(meadow(terrain))
    plan = Plan(
        look=Look.ENGINEERED,
        margin=24,
        reach=10.0,
        spans=(Span(start=(0, 0), stop=(19, 0), profile="embankment"),),
    )
    # A six-block bank runs twelve columns out at a block per two.
    result = terraform(plan, grown, straight_track(level=66), palette)
    assert BlockPos(10, 61, 0) in result.cut
    assert BlockPos(10, 61, 22) not in result.cut


def test_growth_deep_under_a_clearance_is_left_alone():
    terrain, palette = ground_at(lambda x, z: 60)
    grown = FieldTerrain(meadow(terrain))
    plan = Plan(
        margin=24,
        reach=10.0,
        spans=(Span(start=(0, 0), stop=(19, 0), profile="bridge"),),
    )
    result = terraform(plan, grown, straight_track(level=74), palette)
    assert BlockPos(10, 61, 0) not in result.cut


def test_raising_a_bank_leaves_growth_in_a_cave_below_it():
    terrain, palette = ground_at(lambda x, z: 60)
    lichen = BlockPos(10, 55, 0)
    terrain.blocks.set_block(lichen, BlockState.create("minecraft:glow_lichen"))
    result = terraform(Plan(default=Intensity.SHAPE, margin=8, reach=8),
                       terrain, straight_track(level=66), palette)
    assert result.designed[result.grid.at(10, 0)] == 66
    assert lichen not in result.cut


def test_a_protected_shore_keeps_its_soil_but_clears_bridge_headroom():
    from src.Terraform.plan import TerrainMask
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = terrain.blocks
    for x in range(20):
        blocks.set_block(BlockPos(x, 61, 2), BlockState.create("minecraft:water"))
    blocks.set_block(BlockPos(10, 75, 0), LEAVES)
    blocks.set_block(BlockPos(10, 61, 0), TUFT)
    result = terraform(Plan(
        margin=8, reach=8, terrain_mask=TerrainMask(preserve_water=True, water_buffer=3),
        spans=(Span(start=(0, 0), stop=(19, 0), profile="bridge"),),
    ), FieldTerrain(blocks), straight_track(level=74), palette)
    assert result.designed[result.grid.at(10, 0)] == 60
    assert BlockPos(10, 75, 0) in result.cut
    assert BlockPos(10, 61, 0) not in result.cut


def test_a_shore_buffer_does_not_leave_rock_inside_a_bridge_bore():
    terrain, palette = ground_at(lambda x, z: 120)
    blocks = terrain.blocks
    for x in range(20):
        blocks.set_block(BlockPos(x, 121, 2), BlockState.create("minecraft:water"))
    blocks.set_block(BlockPos(10, 76, 0), STONE)
    result = terraform(Plan(
        margin=8, reach=8, terrain_mask=TerrainMask(preserve_water=True, water_buffer=3),
        spans=(Span(start=(0, 0), stop=(19, 0), profile="bridge"),),
    ), FieldTerrain(blocks), straight_track(level=74), palette)
    assert result.designed[result.grid.at(10, 0)] == 120
    assert BlockPos(10, 76, 0) in result.cut


def test_a_pier_buffer_does_not_preserve_shore_foliage_in_bridge_headroom():
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = terrain.blocks
    for x in range(20):
        blocks.set_block(BlockPos(x, 61, 2), BlockState.create("minecraft:water"))
    leaf = BlockPos(10, 75, 0)
    blocks.set_block(leaf, LEAVES)
    result = terraform(Plan(
        margin=8, reach=8, preserve_fittings=("pier",), fitting_buffer=3,
        terrain_mask=TerrainMask(preserve_water=True, water_buffer=3),
        spans=(Span(start=(0, 0), stop=(19, 0), profile="bridge"),),
    ), FieldTerrain(blocks), straight_track(level=74), palette,
        fittings={"pier": [BlockPos(10, 60, 0)]})
    assert result.designed[result.grid.at(10, 0)] == 60
    assert leaf in result.cut


def test_edit_edges_taper_back_toward_fixed_ground_without_extending_the_footprint():
    natural = np.array([[41., 50., 50., 89., 88., 90., 60.]])
    designed = np.array([[67., 50., 50., 89., 85., 90., 60.]])
    movable = np.array([[True, False, False, False, True, False, False]])
    eased = _taper_edit_edges(designed, natural, movable, 2, 16)
    assert eased[0, 0] == 52
    assert eased[0, 4] == 88
    assert eased[0, 1] == designed[0, 1]
    assert eased[0, 5] == designed[0, 5]


def test_a_cut_is_carried_back_from_a_fixed_bore_in_its_existing_footprint():
    natural = np.array([[123., 122.]])
    designed = np.array([[120., 113.]])
    eased = _taper_edit_edges(designed, natural, np.array([[True, False]]), 2, 8)
    assert eased.tolist() == [[115., 113.]]


def test_successive_cuts_carry_the_lower_bore_outward_without_oscillating():
    natural = np.array([[124., 124., 124.]])
    designed = np.array([[122., 120., 113.]])
    eased = _taper_edit_edges(designed, natural, np.array([[True, True, False]]), 2, 8)
    assert eased.tolist() == [[117., 115., 113.]]


def test_a_shallow_cut_beside_unchanged_mountain_is_restored_in_place():
    natural = np.array([[139., 139., 135., 134.]])
    designed = np.array([[139., 136., 132., 130.]])
    movable = np.array([[False, True, True, True]])
    eased = _taper_edit_edges(designed, natural, movable, 2, 16)
    assert eased[0, 0] == 139
    assert eased[0, 1] >= 137
    assert np.max(np.abs(np.diff(eased[0]))) <= 2


def test_bridge_fill_eases_into_preserved_ground_without_touching_the_track():
    natural = np.full((1, 4), 93.)
    designed = np.array([[99., 99., 99., 93.]])
    protected = np.array([[False, False, False, True]])
    distance = np.array([[0., 1., 2., 3.]])
    eased = _ease_preserved_bridge_ground(
        designed, natural, np.ones((1, 4), dtype=bool),
        np.full((1, 4), 50.), protected,
        np.zeros((1, 4), dtype=bool), np.ones((1, 4), dtype=bool),
        distance, 2, 4,
    )
    assert eased.tolist() == [[99., 97., 95., 93.]]


def test_nothing_that_grows_in_a_keepout_is_cleared():
    terrain, palette = ground_at(lambda x, z: 60)
    blocks = meadow(terrain)
    trunk = [BlockPos(13, y, 6) for y in range(61, 66)]
    crown = [BlockPos(x, 66, 6) for x in (12, 13, 14)]
    for position in trunk:
        blocks.set_block(position, OAK)
    for position in crown:
        blocks.set_block(position, LEAVES)
    plan = Plan(
        look=Look.ENGINEERED,
        margin=24,
        reach=12.0,
        spans=(Span(start=(0, 0), stop=(19, 0), profile="embankment"),),
        keepouts=(Keepout(ColumnPos(4, 4), ColumnPos(12, 12), "orchard"),),
    )
    result = terraform(plan, FieldTerrain(blocks), straight_track(level=70), palette)
    assert not any(4 <= p.x <= 12 and 4 <= p.z <= 12 for p in result.cut)
    assert set(trunk) <= result.cut


def test_a_plan_reads_a_spans_headroom_and_where_a_step_is_measured_from(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text(
        """
[[span]]
from = [0, 0]
to = [9, 0]
mode = "none"
profile = "tunnel"
headroom = 0

[profile.behind_walls]
action = "slope_up"
blend = "lower"
from = "crest"
"""
    )
    plan = read_plan(written)
    assert plan.spans[0].headroom == 0
    assert plan.profile("behind_walls").steps[0].base == "crest"
    assert plan.profile("retained_cutting").steps[0].base == "crest"


def test_a_step_measured_from_anywhere_else_is_refused(tmp_path):
    written = tmp_path / "terraform.toml"
    written.write_text('[profile.odd]\naction = "level"\nfrom = "roof"\n')
    with pytest.raises(ValueError, match="footing or crest"):
        read_plan(written)


def test_an_approach_cutting_wraps_back_into_the_hill_beside_a_portal():
    terrain, palette = ground_at(lambda x, z: 90 if x >= 10 else 84)
    blocks = [BlockPos(x, y, z) for x in range(30) for z in range(-2, 3) for y in (70, 76)]
    alignment = Alignment.of(blocks, [block.x for block in blocks])
    plan = Plan(
        margin=24, reach=12,
        spans=(Span(start=(0, 0), stop=(9, 0), profile="cutting"),
               Span(start=(10, 0), stop=(29, 0), profile="tunnel", intensity=Intensity.NONE)),
    )
    result = terraform(plan, terrain, alignment, palette)
    beside = result.designed[result.grid.at(11, 5)]
    assert 77 <= beside < 90
    # Over the lining it is graded too, keeping a block over the roof.
    assert 77 <= result.designed[result.grid.at(11, 0)] < 90
    assert result.designed[result.grid.at(28, 5)] == 90
    assert result.designed[result.grid.at(28, 0)] == 90


def test_a_bleeding_bank_keeps_the_hillside_beside_a_buried_tunnel():
    from src.Terraform.plan import Smoothing
    terrain, palette = ground_at(lambda x, z: 88 if x >= 10 else 60)
    alignment = straight_track(level=70, length=30)
    plan = Plan(
        bleed=True, margin=24, reach=20,
        smoothing=Smoothing(transition_reach=0),
        spans=(Span(start=(0, 0), stop=(9, 0), profile="embankment"),
               Span(start=(10, 0), stop=(29, 0), profile="tunnel", intensity=Intensity.NONE)),
    )
    result = terraform(plan, terrain, alignment, palette)
    for x in range(14, 29):
        for z in range(-18, 19):
            at = result.grid.at(x, z)
            assert result.designed[at] == result.natural[at], (x, z)
    # The approach bank still shapes its own exposed formation.
    assert result.designed[result.grid.at(4, 0)] > 60
