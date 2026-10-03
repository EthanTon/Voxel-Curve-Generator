"""Contour continuity, exposure ownership and bounded three-dimensional repair."""
import numpy as np
import pytest

from src.Core.Field.block_field import BlockField
from src.Core.block.palette import BlockPalette
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Terraform import FieldTerrain, Grid
from src.Terraform import kernels, smoothing
from src.Terraform.clearance import Clearance
from src.Terraform.faces import repair
from src.Terraform.plan import Smoothing, read


def test_main_slope_repair_uses_local_rock_grades_for_cuts_and_soil_grades_for_fill():
    from dataclasses import replace
    from src.Terraform.slopes import SHAPED, limits

    track = np.zeros((5, 15), dtype=bool)
    track[:, 0] = True
    role = np.full(track.shape, SHAPED)
    allowed = np.full(track.shape, 1.)
    allowed[:, :6] = 3.

    def repair(natural, formation):
        made = limits(natural, track, role, np.where(track, formation, 0),
                      np.zeros_like(track), np.zeros_like(track),
                      np.full(track.shape, np.inf), 0, 1, 1, 0,
                      repair_passes=40, ridge_width=0)
        made = replace(made, conform=False, cut_drop=allowed)
        return made.enforce(np.where(track, formation, natural), natural)[0]

    cut = repair(np.full(track.shape, 90.), 70)
    assert list(cut[2, :11]) == [70, 73, 76, 79, 82, 85, 86, 87, 88, 89, 90]
    bank = repair(np.full(track.shape, 60.), 70)
    assert list(bank[2, :11]) == list(range(70, 59, -1))


def test_material_aware_retreat_keeps_rock_steeper_than_soil_without_tall_fins():
    natural = np.full((9, 13), 90.)
    reference = natural.copy()
    reference[:, 0] = 70
    eligible = np.ones_like(reference, dtype=bool)
    eligible[:, 0] = False
    allowed = np.full(reference.shape, 1.)
    allowed[:, :6] = 3.
    result = smoothing.retreat(reference, natural, np.ones_like(eligible), eligible,
                               np.full(reference.shape, -np.inf), allowed, 30)
    assert np.all(np.diff(result, axis=1) <= allowed[:, 1:])
    assert result[4, 5] == 85
    assert result[4, 9] == 89
    assert result[4, 10] == 90
    assert np.array_equal(result[:, 0], reference[:, 0])
    assert np.all(result <= reference)


def test_running_clearance_covers_between_rails_and_stops_at_walls_and_roofs():
    from src.Assembly.earthworks import running_of
    from src.Core.World.block_world import BlockWorld
    from src.Core.block.registry import BlockRegistry
    from src.Terraform import Alignment
    from src.Terraform.clearance import enclosing, OPEN

    registry = BlockRegistry.for_vanilla_26_2()
    world = BlockWorld(registry)
    field = BlockField(registry.palette)
    for z in range(4):
        field.set_block(BlockPos(0, 74, z), BlockState.create('minecraft:stone'))
    field.set_block(BlockPos(0, 75, 0), BlockState.create('minecraft:rail'))
    field.set_block(BlockPos(0, 83, 2), BlockState.create('minecraft:white_concrete'))
    field.set_block(BlockPos(0, 76, 3), BlockState.create('minecraft:stone_bricks'))
    world.add_field(field)
    alignment = Alignment.of([BlockPos(0, 74, z) for z in range(4)])
    running = running_of(world, alignment)
    assert running == {(0, z): 75 for z in range(4)}
    grid = Grid.around(alignment.columns, 0)
    clearance = enclosing(grid, {(0, 0): [74, 75], (0, 1): [74],
                                 (0, 2): [74, 83], (0, 3): [74, 75, 76]},
                          lambda x, z: 0, running=running, running_headroom=12)
    assert clearance.holds(*grid.at(0, 1), 76)
    assert clearance.holds(*grid.at(0, 2), 82)
    assert not clearance.holds(*grid.at(0, 2), 83)
    assert clearance.floor[grid.at(0, 3)] == OPEN


def test_foundation_gaps_preserve_soil_while_bore_and_bridge_headroom_clear():
    from src.Core.coordinates import ColumnPos
    from src.Terraform.clearance import enclosing

    grid = Grid(ColumnPos(0, 0), 3, 1)
    envelope = {(0, 0): [70, 71, 75, 76],  # foundation/sleeper gap
                (1, 0): [70, 71, 75, 76, 86],  # tunnel roof
                (2, 0): [70, 71, 75, 76, 85]}  # bridge fitting
    clearance = enclosing(grid, envelope, lambda x, z: 4 if x == 2 else 0,
                          {(x, 0): 70 for x in range(3)},
                          {(x, 0): 76 for x in range(3)}, 12)
    native = [BlockPos(x, y, 0) for x in range(3) for y in range(70, 90)]
    cut = clearance.carved(native, grid)
    assert all(BlockPos(x, y, 0) not in cut for x in range(3) for y in range(70, 77))
    assert all(BlockPos(1, y, 0) in cut for y in range(77, 86))
    assert BlockPos(1, 86, 0) not in cut  # roof remains
    assert all(BlockPos(2, y, 0) in cut for y in range(86, 90))


def test_tunnel_clearance_starts_at_each_columns_floor_below_the_rails():
    from src.Terraform.clearance import enclosing

    grid = Grid((0, 0), 5, 1)
    envelope = {
        (0, 0): [71, 72, 73, 74, 80, 81, 82],  # rail column
        (1, 0): [71, 72, 73, 81, 82],          # aisle
        (2, 0): [71, 72, 82],                  # lower floor between lanes
        (3, 0): [71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82],  # wall recess
        (4, 0): [71, 72, 73, 74, 75, 76],     # foundation gap, no cavity
    }
    clearance = enclosing(grid, envelope, lambda x, z: 0,
                          {(x, 0): 71 for x in range(5)},
                          {(x, 0): 74 for x in range(5)}, 12)
    assert clearance.floor[0, :4].tolist() == [75, 74, 73, 76]
    for x in range(4):
        levels = envelope[x, 0]
        for below, above in zip(levels, levels[1:]):
            if above - below > 1 and above - 1 > 74:
                assert all(clearance.holds(0, x, y) for y in range(below + 1, above))
        assert not clearance.holds(0, x, levels[-1])  # authored roof
    assert not any(clearance.holds(0, 4, y) for y in range(71, 77))


@pytest.mark.parametrize('running_headroom', [0, 12])
def test_open_bed_clearance_starts_above_each_floor_and_keeps_rail_support(running_headroom):
    from src.Terraform.clearance import enclosing

    grid = Grid((0, 0), 6, 1)
    envelope = {
        (0, 0): [71, 72, 73, 74],  # rail
        (1, 0): [71, 72, 73],      # lower aisle
        (2, 0): [71, 72],          # lower bed between lanes
        (3, 0): list(range(71, 77)),  # retaining wall
        (4, 0): [70, 71, 73, 74],  # soil contact under the sleeper
        (5, 0): [71, 72, 74],      # protruding rail with missing support
    }
    footing = {(x, 0): 71 for x in range(6)}
    footing[5, 0] = 74  # the reserved rail support must not become clearance
    clearance = enclosing(grid, envelope, lambda x, z: 4,
                          footing, {(x, 0): 74 for x in range(6)}, running_headroom)

    assert clearance.floor[0, :3].tolist() == [75, 74, 73]
    assert clearance.holds(0, 1, 74)
    assert clearance.holds(0, 2, 73)
    assert clearance.holds(0, 2, 74)
    assert not any(clearance.holds(0, 3, y) for y in range(71, 77))
    assert not clearance.holds(0, 4, 72)
    assert not clearance.holds(0, 5, 73)


def test_cliff_survey_recognizes_steep_staircases_without_promoting_rolling_banks():
    from src.Terraform.terraform import _natural_cliffs

    columns = np.arange(17)
    steep = np.broadcast_to(70. + columns, (7, 17)).copy()
    known = np.ones_like(steep, dtype=bool)
    assert _natural_cliffs(steep, known).all()
    gentle = np.broadcast_to(70. + columns // 2, steep.shape)
    assert not _natural_cliffs(gentle, known).any()
    known[:, ::4] = False
    assert not _natural_cliffs(steep, known).any()


def test_steep_native_rock_remains_local_without_a_transition_brush():
    from tests.test_terraform import ground_at, straight_track
    from src.Terraform import terraform, Plan, Look, Intensity
    from src.Terraform.plan import Embankment, Slopes, Cutting

    terrain, palette = ground_at(lambda x, z: 80 + max(z, 0))
    result = terraform(Plan(
        look=Look.NATURAL, default=Intensity.SHAPE, margin=32, reach=32,
        embankment=Embankment(shape='natural'), cutting=Cutting(rock=True, rock_drop=4),
        slopes=Slopes(max_drop=1, repair_passes=64, cone_passes=64),
        smoothing=Smoothing(transition_reach=0)),
        terrain, straight_track(level=70), palette)
    assert result.designed[result.grid.at(10, 2)] == 70
    # The required cut ends locally; the existing rock hillside is not
    # forced into the soil grade across the remainder of the corridor.
    for z in range(8, 30):
        at = result.grid.at(10, z)
        assert result.designed[at] == result.natural[at]
    assert not any(p.x == 10 and p.z >= 8 for p in result.cut)


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_rock_ledge_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match='cutting.rock_drop'):
        read({'terraform': {'cutting': {'rock_drop': value}}})


def test_fractional_gaussian_sizes_change_continuously():
    impulse = np.zeros((17, 17))
    impulse[8, 8] = 1
    a = kernels.gaussian(impulse, .99)
    b = kernels.gaussian(impulse, 1.0)
    c = kernels.gaussian(impulse, 1.01)
    assert a[8, 8] > b[8, 8] > c[8, 8]
    assert np.max(abs(a - c)) < .01
    assert b.sum() == pytest.approx(1)


def test_compact_kernel_is_radial_normalized_and_has_no_distant_tail():
    impulse = np.zeros((21, 21))
    impulse[10, 10] = 1
    result = kernels.wendland(impulse, 5, "constant")
    assert result.sum() == pytest.approx(1)
    assert result[10, 15] == 0
    assert result[13, 14] == 0
    assert result[10, 12] == result[12, 10]
    known = np.ones((11, 11), bool)
    known[:, :4] = False
    values = np.where(known, 70., -32768)
    smooth = kernels.normalized(values, known, 2, "wendland")
    assert smooth[known] == pytest.approx(np.full(known.sum(), 70.))


def test_contour_diffusion_cannot_jump_a_mask_and_preserves_a_native_cliff():
    values = np.zeros((9, 13))
    values[:, :5] = 10
    domain = np.ones(values.shape, bool)
    domain[:, 5] = False
    assert kernels.contour(values, np.zeros_like(values), domain, 30)[:, 6:].max() == 0
    guide = np.zeros_like(values)
    guide[:, :5] = 40
    diffused = kernels.contour(values, guide, np.ones_like(domain), 20, 2)
    assert diffused[4, 4] == pytest.approx(10)
    assert diffused[4, 5] < 1e-6


def test_unchanged_column_with_new_long_face_is_not_a_preserved_cliff():
    natural = np.full((9, 9), 80.)
    natural[:, 5:] = 78
    designed = natural.copy()
    designed[:, 5] = 60
    known = np.ones_like(natural, bool)
    assert smoothing.new_exposure(designed, natural, known)[4, 4]
    assert not smoothing.preserved_faces(designed, natural, known)[4, 4]
    assert smoothing.preserved_faces(natural, natural, known)[4, 4]


def test_retreat_removes_a_cut_created_fin_without_erasing_native_cliff_or_keepout():
    natural = np.full((15, 15), 80.)
    natural[:, :3] = 60  # A genuine native cliff.
    designed = natural.copy()
    designed[6:9, 6:9] = 65
    editable = np.ones(natural.shape, bool)
    editable[:, :4] = False
    editable[7, 10] = False
    result = smoothing.retreat(designed, natural, np.ones_like(editable), editable,
                               np.full(natural.shape, 66), 2, 64)
    assert result[7, 9] <= 68
    assert result[7, 10] == 80
    assert np.array_equal(result[:, :4], natural[:, :4])
    assert np.all(result <= designed)
    assert np.all(result[editable] >= np.minimum(designed[editable], 66))


def test_contact_displacement_keeps_native_rolling_contours_and_obeys_masks():
    rows, cells = np.indices((31, 31))
    natural = 60. + 1.5 * np.sin(rows / 5)
    footprint = cells <= 2
    eligible = np.ones(footprint.shape, bool)
    eligible[:, 25:] = False
    heights = np.where(footprint, natural + 8, natural)
    reference = heights.copy()
    result = smoothing.meet(reference, natural, footprint, eligible, heights,
                            1, 64, Smoothing(), 22)
    assert result[15, 5] > natural[15, 5]
    assert result[15, 23] == natural[15, 23]
    assert np.array_equal(result[:, 25:], natural[:, 25:])
    # Connected contour diffusion can soften a seam by a fraction of a
    # block; it must still carry the hillside's actual relief.
    assert result[8, 8] - result[15, 8] == pytest.approx(natural[8, 8] - natural[15, 8], abs=.001)


def test_portal_contour_cut_has_finite_support_and_preserves_unequal_native_relief():
    rows, cells = np.indices((31, 41))
    natural = 90. + 2 * np.sin(rows / 6)
    sources = cells == 2
    targets = natural - 12
    result = smoothing.contour_cut(natural, sources, targets, np.ones_like(sources), 20, 2)
    assert result[sources] == pytest.approx(targets[sources])
    assert np.array_equal(result[:, 14:], natural[:, 14:])
    assert np.all(result <= natural)
    # Native relief survives; this is not an absolute-height pyramid.
    assert result[5, 7] - result[19, 7] == pytest.approx(natural[5, 7] - natural[19, 7], abs=.001)
    increments = np.diff(result[15, 2:15])
    assert np.ptp(increments) > 1


@pytest.mark.parametrize('rising', [False, True])
def test_main_natural_slopes_carry_native_ridges_instead_of_flattening_them(rising):
    from src.Terraform.layers import Ground, Slope
    rows, cells = np.indices((19, 41))
    native = np.full(rows.shape, 90. if rising else 60.)
    relief = native.copy()
    relief[6:13, 10:15] += 5
    distance = cells.astype(float)
    base = np.full(rows.shape, 74.)
    def shaped(land):
        ground = Ground(Grid((0, 0), 41, 19), land, distance, base, base, rows.astype(float))
        return Slope(1., rising=rising, natural=True, reach=30, shoulder=2)(ground, land)
    plain, ridge = shaped(native), shaped(relief)
    assert ridge[9, 12] - plain[9, 12] == pytest.approx(5)
    assert np.array_equal(ridge[:, :3], base[:, :3])
    assert np.array_equal(ridge[:, 33:], relief[:, 33:])


def test_portal_contour_cut_cannot_jump_water_or_a_diagonal_mask_corner():
    natural = np.full((17, 19), 90.)
    sources = np.zeros_like(natural, bool)
    sources[8, 3] = True
    domain = np.ones_like(sources)
    domain[:, 7] = False
    result = smoothing.contour_cut(natural, sources, natural - 20, domain, 18, 2)
    assert np.array_equal(result[:, 7:], natural[:, 7:])
    domain[:] = False
    domain[8, 3] = domain[9, 4] = True
    result = smoothing.contour_cut(natural, sources, natural - 20, domain, 18, 2)
    assert result[9, 4] == natural[9, 4]


def test_natural_portal_replans_approach_and_roof_together_without_a_section_seam():
    from src.Terraform.layers import Ground
    from src.Terraform.plan import Plan, Embankment
    from src.Terraform.terraform import _portal_grading
    rows, cells = np.indices((31, 41))
    natural = 102. + 2 * np.sin(rows / 6)
    tunnel = cells >= 20
    reference = np.where(tunnel, natural, 78.)
    distance = np.maximum(abs(rows - 15) - 2, 0).astype(float)
    ground = Ground(Grid((0, 0), 41, 31), natural, distance,
                    np.full(rows.shape, 73.), np.full(rows.shape, 82.), cells.astype(float))
    result = _portal_grading(
        Plan(reach=24, margin=24, embankment=Embankment(shape='natural')),
        ground, np.ones(41, bool), tunnel, natural, reference,
        corridor=np.ones(rows.shape, bool), contact_drop=np.full(rows.shape, 3.))
    assert result[3, 19] > reference[3, 19] + 15  # less excavation beside the approach
    assert abs(result[3, 19] - result[3, 20]) < 4  # no restored section wall
    assert np.array_equal(result[:, 37:], natural[:, 37:])
    assert np.all(result[tunnel & (distance <= 0)] >= 83)
    bed = (distance <= 0) & ~tunnel
    assert np.array_equal(result[bed], reference[bed])
    assert np.all(result <= natural)


def soil_fixture():
    from src.Terraform.materialize import Strata
    from src.Terraform.subsurface import Layering
    palette = BlockPalette()
    blocks = BlockField(palette)
    stone = BlockState.create('minecraft:stone')
    podzol = BlockState.create('minecraft:podzol')
    dirt = BlockState.create('minecraft:dirt')
    grid = Grid((0, 0), 15, 15)
    for x in range(15):
        for z in range(15):
            for y in range(4, 21):
                blocks.set_block(BlockPos(x, y, z), stone)
    layers = np.zeros((4, 15, 15), dtype=int)
    layers[1:] = 1
    return (FieldTerrain(blocks), grid, BlockField(palette),
            Strata(podzol, dirt, stone), Layering(layers, (podzol, dirt)))


def test_gentle_cut_recovers_environment_soil_without_filling_caves_or_clearance():
    from src.Terraform.materialize import finish_surfaces
    terrain, grid, fill, strata, layering = soil_fixture()
    cut = frozenset(BlockPos(x, y, z) for x in range(15) for z in range(15)
                    for y in range(13, 21)) | {BlockPos(7, 11, 7)}
    floor = grid.filled(999)
    ceiling = grid.filled(-999)
    floor[grid.at(8, 8)], ceiling[grid.at(8, 8)] = 10, 12
    finish_surfaces(terrain, grid, grid.filled(20.), grid.filled(12.),
                    grid.filled(True, dtype=bool), fill, cut, strata, layering,
                    grid.filled(-32768), Clearance(floor, ceiling))
    assert fill.get_block(BlockPos(6, 12, 6)).name == 'minecraft:podzol'
    assert fill.get_block(BlockPos(6, 11, 6)).name == 'minecraft:dirt'
    assert fill.get_block(BlockPos(7, 12, 7)) is None  # unsupported saved/cut void
    assert fill.get_block(BlockPos(8, 12, 8)) is None  # rail/bore clearance
    assert BlockPos(6, 13, 6) in cut


def test_steep_rock_faces_do_not_acquire_grass_stripes():
    from src.Terraform.materialize import finish_surfaces
    terrain, grid, fill, strata, layering = soil_fixture()
    # A sharp face beside the bench remains rock rather than being painted.
    steep = np.full((15, 15), 18.)
    steep[:, :7] = 4
    rock_fill = BlockField(fill.palette)
    cliff_cut = frozenset(BlockPos(x, y, z) for x in range(7) for z in range(15)
                          for y in range(5, 21))
    finish_surfaces(terrain, grid, grid.filled(20.), steep,
                    grid.filled(True, dtype=bool), rock_fill, cliff_cut, strata, layering,
                    grid.filled(-32768), Clearance(grid.filled(999), grid.filled(-999)))
    assert rock_fill.get_block(BlockPos(6, 4, 7)) is None
    assert rock_fill.get_block(BlockPos(7, 18, 7)) is None


def test_new_cut_face_releases_rock_trim_floor_but_keeps_required_roof_cover():
    natural = np.full((9, 15), 106.)
    reference = natural.copy()
    reference[:, 6] = 99
    reference[:, 7:] = 87
    known = np.ones_like(reference, bool)
    rock_floor = np.full_like(reference, -np.inf)
    rock_floor[:, 6] = 99
    cover = np.full_like(reference, -np.inf)
    cover[:, 6] = 83
    floor = smoothing.contact_floor(reference, natural, known, rock_floor, cover)
    result = smoothing.retreat(reference, natural, known, known, floor, 3, 64)
    assert result[4, 6] == 90
    assert np.all(result[:, 6] >= 83)
    assert np.all(result <= reference)
    assert np.all(np.diff(result[4]) >= -3)


def test_native_cliff_retains_rock_floor_and_contact_cover():
    natural = np.full((9, 15), 87.)
    natural[:, :7] = 106
    known = np.ones_like(natural, bool)
    rock_floor = np.full_like(natural, 99.)
    cover = np.full_like(natural, 83.)
    floor = smoothing.contact_floor(natural, natural, known, rock_floor, cover)
    assert np.array_equal(floor, rock_floor)
    result = smoothing.retreat(natural, natural, known, known, floor, 3, 64)
    assert np.array_equal(result, natural)


def test_contact_cover_still_stops_repair_when_approach_is_too_low():
    natural = np.full((9, 15), 106.)
    reference = natural.copy()
    reference[:, 6] = 99
    reference[:, 7:] = 70
    known = np.ones_like(reference, bool)
    floor = smoothing.contact_floor(reference, natural, known,
                                     np.full_like(reference, 99.),
                                     np.full_like(reference, 83.))
    result = smoothing.retreat(reference, natural, known, known, floor, 3, 64)
    assert result[4, 6] == 83
    assert result[4, 7] == 70


def test_contact_override_controls_optional_blending_but_keeps_required_support():
    natural = np.full((21, 31), 60.)
    footprint = np.indices(natural.shape)[1] <= 2
    heights = np.where(footprint, 68., natural)
    known = np.ones_like(footprint)
    options = Smoothing(behavior="grow")
    grown = smoothing.meet(heights, natural, footprint, known, heights, 1, 64, options, 24)
    overridden = smoothing.meet(heights, natural, footprint, known, heights, 1, 64, options, 24,
                                np.full(natural.shape, smoothing.Mode.MELT))
    bounds = smoothing.meet(heights, natural, footprint, known, heights, 1, 64,
                            Smoothing(contact_blend=0), 24)
    assert grown[10, 6] > overridden[10, 6]
    assert np.array_equal(overridden, bounds)


def fixture():
    palette = BlockPalette()
    blocks = BlockField(palette)
    stone = BlockState.create("minecraft:stone")
    grid = Grid((0, 0), 9, 9)
    for x in range(9):
        for z in range(9):
            for y in range(5, 21):
                blocks.set_block(BlockPos(x, y, z), stone)
    natural = grid.filled(20.)
    designed = natural.copy()
    cut = set()
    for x, z in ((3, 4), (5, 4), (4, 3), (4, 5)):
        designed[grid.at(x, z)] = 10
        cut.update(BlockPos(x, y, z) for y in range(11, 21))
    eligible = grid.filled(False, dtype=bool)
    eligible[grid.at(4, 4)] = True
    return blocks, palette, grid, natural, designed, cut, eligible


@pytest.mark.parametrize("mode", list(smoothing.Mode))
def test_voxel_repair_respects_direction_and_a_single_stable_budget(mode):
    blocks, palette, grid, natural, designed, cut, eligible = fixture()
    options = Smoothing(behavior=mode.name.lower(), face_iterations=4, stable_change=1)
    eroded, added = repair(FieldTerrain(blocks), grid, natural, designed,
                           grid.filled(True, dtype=bool), eligible, BlockField(palette),
                           cut, Clearance.none(grid), grid.filled(mode), options)
    assert not added
    if mode == smoothing.Mode.GROW:
        assert not eroded
    elif mode == smoothing.Mode.MELT:
        assert {BlockPos(4, y, 4) for y in range(11, 21)} <= eroded
    else:
        assert len(eroded) == 1
    assert all(p.x == 4 and p.z == 4 for p in eroded)


def test_voxel_repair_keeps_original_caves_and_blocked_columns():
    blocks, palette, grid, natural, designed, cut, eligible = fixture()
    # Keep-out excludes the remnant; grow must not invent fill in a saved cave.
    empty = grid.filled(False, dtype=bool)
    args = (FieldTerrain(blocks), grid, natural, designed, grid.filled(True, dtype=bool),
            empty, BlockField(palette), cut, Clearance.none(grid), grid.filled(smoothing.Mode.MELT), Smoothing())
    assert repair(*args) == (frozenset(), {})
    cave = BlockPos(4, 12, 4)
    blocks.remove_block(cave)
    eroded, added = repair(FieldTerrain(blocks), grid, natural, designed,
                           grid.filled(True, dtype=bool), eligible, BlockField(palette),
                           cut, Clearance.none(grid), grid.filled(smoothing.Mode.GROW), Smoothing(behavior="grow"))
    assert cave not in added
    assert not eroded


def test_auto_face_repair_detects_sideways_cuts_without_changed_column_tops():
    blocks, palette, grid, natural, designed, cut, eligible = fixture()
    designed[:] = natural  # Roofs remain; the cut still opens sides underneath.
    cut = {p for p in cut if p.y < 20}
    eroded, _ = repair(FieldTerrain(blocks), grid, natural, designed,
                      grid.filled(True, dtype=bool), eligible, BlockField(palette),
                      cut, Clearance.none(grid), grid.filled(smoothing.Mode.STABLE), Smoothing())
    assert len(eroded) > 1
    assert BlockPos(4, 12, 4) in eroded
    assert BlockPos(4, 20, 4) not in eroded


def test_new_controls_are_configurable_and_disable_independently():
    options = read({"terraform": {"smoothing": {"kernel": "wendland", "face_iterations": 0,
                   "contour_strength": 0, "contact_blend": .5, "face_min_exposed": 4}}}).smoothing
    assert options.kernel == "wendland"
    assert options.face_iterations == options.contour_strength == 0
    assert options.contact_blend == .5
    assert options.face_min_exposed == 4


@pytest.mark.parametrize("setting", [{"face_iterations": -1}, {"face_depth": True},
    {"face_min_exposed": 7}, {"face_min_exposed": 1}, {"contour_strength": 1.1},
    {"contact_blend": -.1}, {"contour_scale": 0}, {"contour_scale": True},
    {"contour_scale": float("nan")}, {"contour_scale": float("inf")}])
def test_invalid_controls_are_rejected(setting):
    with pytest.raises(ValueError):
        read({"terraform": {"smoothing": setting}})


@pytest.mark.parametrize('name,bank', [
    ('embankment', {-9, -8, -7, 7, 8, 9}),
    ('embankment_trench_left', {7, 8, 9}),
    ('embankment_trench_right', {-9, -8, -7}),
])
def test_saved_bank_masks_yield_outer_decorations_but_keep_bed_and_trench(name, bank):
    from pathlib import Path
    import tomllib
    from src.Parser.sections import read_section
    from src.Assembly.structure import build_section
    from src.Core.block.registry import BlockRegistry
    from src.Core.Domain import DomainRegistry
    root = Path('sandbox/bezier_test')
    data = tomllib.loads((root / 'build.toml').read_text())
    spec = read_section(next(s for s in data['section'] if s['name'] == name), {}, root)
    registry = BlockRegistry.for_vanilla_26_2()
    domains = DomainRegistry()
    domains.create('bed')
    section = build_section(spec, registry, domains)
    covered_names = set()
    protected_bed = 0
    for annotation in section.points:
        feature = annotation.feature
        for position, block_id in feature.blocks.iter_block_ids():
            material = feature.blocks.palette.get_state(block_id).name
            covered = material in feature.yielding
            if abs(annotation.position.x) <= 6:
                assert not covered
                protected_bed += 1
            if annotation.position.x not in bank:
                assert not covered  # trench wall, railing and footing stay structural
            elif covered:
                covered_names.add(material)
    assert protected_bed > 20
    assert {'minecraft:andesite_wall', 'minecraft:iron_bars'} <= covered_names

    # Inspect both composition paths without exporting: copying saved ground
    # and laying new earthworks must obey the same lateral bed mask.
    from types import SimpleNamespace
    from src.Core.World.block_world import BlockWorld
    from src.Core.modes import PlacementMode
    from src.Assembly.export import _changed_cells, _build_piece
    world = BlockWorld(registry)
    for annotation in section.points:
        world.add_field(annotation.feature.blocks,
                        offset=(annotation.position.x + 16, 80, 0),
                        yielding=annotation.feature.yielding)
    # The bank begins at the gray-wool bed, not at the replaceable outer
    # foundations or railings. A retained side still meets its solid wall.
    from src.Assembly.earthworks import alignment_of
    from src.Core.Path.path import Path
    path = Path([(16, 80, 0)], [(0., 0., 1.)])
    alignment = alignment_of(world, path, formation=lambda x, z, station: 79)
    assert set(alignment.footing) == {(16 + x, 0) for x in range(-9, 10) if x not in bank}
    assert all(alignment.meets[16 + x, 0] == 79 for x in range(-6, 7))
    states, positions, codes, soft = _changed_cells(world, None)
    source = SimpleNamespace(generated=lambda x, z: True,
                             section=lambda x, y, z: (
                                 [BlockState.create('minecraft:dirt')], np.zeros(4096, dtype=int)))
    composed, _, _, _ = _build_piece('inspection', BlockPos(7, 75, 0),
                                     BlockPos(25, 89, 0), states, positions, codes,
                                     soft, source, None)
    soil = BlockField(registry.palette)
    for position, code, cover in zip(positions, codes, soft):
        p = BlockPos(*(int(v) for v in position))
        expected = BlockState.create('minecraft:dirt') if cover else states[code]
        assert composed.state_at(p) == expected
        soil.set_block(p, BlockState.create('minecraft:dirt'))
    world.add_field(soil, placement_mode=PlacementMode.REPLACE_AIR_ONLY, earthworks=True)
    for position, code, cover in zip(positions, codes, soft):
        p = BlockPos(*(int(v) for v in position))
        assert world.get_block(p, resolved=False) == (
            BlockState.create('minecraft:dirt') if cover else states[code])


def test_rock_under_gentle_land_does_not_pin_an_exposed_stone_ring():
    from src.Terraform.slopes import SHAPED, limits
    from src.Terraform.terraform import _natural_cliffs
    class Rock:
        def rock_tops(self):
            return natural - 2
        def face(self, rows, cells, low, high):
            return np.ones(len(rows), dtype=bool)
    natural = np.full((9, 31), 90.)
    track = np.zeros_like(natural, bool)
    track[:, 0] = True
    def made(area):
        return limits(natural, track, np.full(track.shape, SHAPED),
                      np.full(track.shape, 74.), np.zeros_like(track),
                      np.zeros_like(track), np.full(track.shape, np.inf),
                      0, 1, 1, 64, rock=Rock(), rock_area=area)
    assert not _natural_cliffs(natural, np.ones_like(track)).any()
    gentle = made(_natural_cliffs(natural, np.ones_like(track)))
    assert gentle.cut[4, 1] < 76
    assert np.all(np.isneginf(gentle.stone))
    rocky = made(np.ones_like(track))
    assert rocky.cut[4, 1] >= 88  # explicit cliff rock still stands


@pytest.mark.parametrize('delta', [-8., 8.])
def test_contact_blend_is_connected_for_both_cut_and_bank(delta):
    natural = np.full((17, 25), 80.)
    footprint = np.zeros_like(natural, bool)
    footprint[:, 2] = True
    eligible = np.ones_like(footprint)
    eligible[:, 7] = False
    reference = np.where(footprint, natural + delta, natural)
    result = smoothing.meet(reference, natural, footprint, eligible, reference,
                            1, 64, Smoothing(), 20)
    assert abs(result[8, 4] - natural[8, 4]) > 4
    assert np.array_equal(result[:, 7:], natural[:, 7:])


@pytest.mark.parametrize('retained', [False, True])
def test_gentle_stone_hill_sweeps_to_bed_or_trench_crest_with_one_block_steps(retained):
    from tests.test_terraform import ground_at
    from src.Terraform import terraform, Plan, Look, Alignment, Intensity, Span
    from src.Terraform.plan import Embankment, Slopes, Cutting
    terrain, palette = ground_at(lambda x, z: 80)
    blocks, stations = [], []
    for x in range(30):
        for z in range(-3, 4):
            top = 76 if retained and abs(z) == 3 else 70
            for y in range(70, top + 1):
                blocks.append(BlockPos(x, y, z))
                stations.append(x)
    result = terraform(Plan(
        look=Look.NATURAL, default=Intensity.SHAPE, margin=32, reach=32,
        embankment=Embankment(shape='natural'), cutting=Cutting(rock=True, rock_drop=4),
        slopes=Slopes(max_drop=1, repair_passes=64, cone_passes=64),
        spans=(Span(start=(0, 0), stop=(29, 0),
                    profile='retained_cutting' if retained else 'embankment'),)),
        terrain, Alignment.of(blocks, stations), palette)
    from src.Terraform.slopes import level
    section = np.array([result.designed[result.grid.at(15, z)] for z in range(3, 36)])
    assert section[0] == (76 if retained else 70)
    assert section[-1] == 80
    assert np.max(np.abs(np.diff(level(section)))) <= 1
    assert result.designed[result.grid.at(15, 0)] == 70
    assert not any(result.clearance.holds(*result.grid.at(p.x, p.z), p.y)
                   for p, _ in result.fill.iter_block_ids())


@pytest.mark.parametrize('options', [
    {'sides': 'yes'}, {'reach': 0}, {'reach': True}, {'max_change': -1},
    {'max_change': 2.5}, {'min_cover': 0}, {'min_cover': True}, {'unexpected': 1},
])
def test_tunnel_contact_options_reject_invalid_values(options):
    with pytest.raises(ValueError, match='tunnel'):
        read({'terraform': {'tunnel': options}})


def test_tunnel_sides_are_opt_in_and_configurable():
    from src.Terraform import Tunnel
    assert not read({}).tunnel.sides
    plan = read({'terraform': {'tunnel': {'sides': True, 'reach': 9,
                                         'max_change': 3, 'min_cover': 2}}})
    assert plan.tunnel == Tunnel(sides=True, reach=9, max_change=3, min_cover=2)


def test_tunnel_side_contacts_grow_and_melt_independently_without_moving_the_body():
    from src.Terraform import Plan, Tunnel
    from src.Terraform.terraform import _tunnel_sides
    from src.Terraform.smoothing import Context
    shape = (35, 35)
    body = np.zeros(shape, bool)
    body[:, 15:20] = True
    natural = np.where(np.indices(shape)[1] < 17, 68., 83.)
    context = Context(np.full(shape, -1, dtype=np.int8), np.zeros(shape))
    plan = Plan(margin=16, reach=16, tunnel=Tunnel(sides=True, reach=8))
    result, area = _tunnel_sides(plan, natural, natural.copy(), body,
                                np.full(shape, 80.), np.full(shape, 70.),
                                np.ones(shape, bool), np.ones(shape, bool), context,
                                np.ones(shape))
    assert result[17, 14] > natural[17, 14]  # exposed low side gets support
    assert result[17, 20] < natural[17, 20]  # thin cover on high side is trimmed
    assert np.array_equal(result[body], natural[body])
    assert not area[body].any()
    assert np.max(abs(result - natural)) <= 4
    assert np.array_equal(result[:, :7], natural[:, :7])
    assert np.array_equal(result[:, 28:], natural[:, 28:])


def test_deep_tunnel_sides_do_not_seed_a_mountain_wide_cut():
    from src.Terraform import Plan, Tunnel
    from src.Terraform.terraform import _tunnel_sides
    from src.Terraform.smoothing import Context
    shape = (25, 25)
    body = np.zeros(shape, bool)
    body[:, 10:15] = True
    natural = 100. + np.indices(shape)[1] * 2
    context = Context(np.full(shape, -1, dtype=np.int8), np.ones(shape))
    result, area = _tunnel_sides(Plan(tunnel=Tunnel(sides=True)), natural, natural.copy(),
                                body, np.full(shape, 80.), np.full(shape, 70.),
                                np.ones(shape, bool), np.ones(shape, bool), context,
                                np.full(shape, 3.))
    assert np.array_equal(result, natural)
    assert not area.any()


@pytest.mark.parametrize('behavior', ['grow', 'stable', 'melt'])
def test_tunnel_side_contacts_obey_material_direction_and_connected_masks(behavior):
    from src.Terraform import Plan, Tunnel
    from src.Terraform.terraform import _tunnel_sides
    from src.Terraform.smoothing import Context, Mode
    shape = (25, 35)
    body = np.zeros(shape, bool)
    body[:, 15:20] = True
    natural = np.where(np.indices(shape)[1] < 17, 68., 84.)
    eligible = np.ones(shape, bool)
    eligible[:, 22] = False  # water or keep-out barrier
    domain = np.ones(shape, bool)
    domain[20:] = False  # another route arm/held section
    context = Context(np.full(shape, Mode[behavior.upper()], dtype=np.int8), np.zeros(shape))
    plan = Plan(tunnel=Tunnel(sides=True), smoothing=Smoothing(stable_change=.5))
    result, _ = _tunnel_sides(plan, natural, natural.copy(), body,
                            np.full(shape, 80.), np.full(shape, 70.),
                            eligible, domain, context, np.full(shape, 3.))
    assert np.array_equal(result[:, 22:], natural[:, 22:])
    assert np.array_equal(result[20:], natural[20:])
    if behavior == 'grow':
        assert np.all(result >= natural)
        assert result[10, 14] > natural[10, 14]
    elif behavior == 'melt':
        assert np.all(result <= natural)
        assert result[10, 20] < natural[10, 20]
    else:
        assert np.max(abs(result - natural)) <= .5


def test_enabled_tunnel_contacts_preserve_bore_bed_roof_cover_and_deep_hillside():
    from tests.test_terraform import ground_at, straight_track, STONE
    from src.Terraform import terraform, Plan, Tunnel, Span, Intensity
    from src.Terraform.plan import Embankment
    terrain, palette = ground_at(lambda x, z: 81 if x < 15 else 100)
    # A solid source down to the rail bed makes cavity checks meaningful.
    for x in range(30):
        for z in range(-3, 4):
            for y in range(68, 101 if x >= 15 else 82):
                terrain.blocks.set_block(BlockPos(x, y, z), STONE)
    alignment = straight_track(level=70, length=30)
    envelope = {(x, z): [70, 71, 80] for x in range(30) for z in range(-2, 3)}
    result = terraform(Plan(
        tunnel=Tunnel(sides=True, min_cover=3), margin=16, reach=12,
        embankment=Embankment(shape='natural'),
        spans=(Span(start=(0, 0), stop=(29, 0), profile='tunnel',
                    intensity=Intensity.SHAPE, headroom=0),)),
        terrain, alignment, palette, envelope=envelope,
        running={(x, z): 71 for x in range(30) for z in range(-2, 3)})
    assert result.designed[result.grid.at(5, 0)] >= 83
    assert result.fill.get_block(BlockPos(5, 83, 0)) is not None
    assert BlockPos(5, 75, 0) in result.cut
    assert all(BlockPos(5, y, 0) not in result.cut for y in (68, 69, 70, 71, 80, 81))
    assert result.designed[result.grid.at(25, 0)] == 100
    assert result.designed[result.grid.at(25, 6)] == 100
    assert not any(p.x == 25 and abs(p.z) > 2 for p in result.cut)


def test_tunnel_side_voxel_floor_preserves_lower_ground_under_an_unchanged_top():
    blocks, palette, grid, natural, designed, cut, eligible = fixture()
    designed[:] = natural
    cut = {p for p in cut if p.y < 20}
    eroded, added = repair(FieldTerrain(blocks), grid, natural, designed,
                           grid.filled(True, dtype=bool), eligible, BlockField(palette),
                           cut, Clearance.none(grid), grid.filled(smoothing.Mode.MELT),
                           Smoothing(behavior='melt', face_iterations=4),
                           cut_floor=grid.filled(16.))
    assert BlockPos(4, 17, 4) in eroded
    assert BlockPos(4, 12, 4) not in eroded
    assert BlockPos(4, 20, 4) not in eroded
    assert all(p.y > 16 for p in eroded)
    assert not added


def test_a_held_tunnel_span_does_not_enable_side_edits_or_extra_cover():
    from tests.test_terraform import ground_at, straight_track
    from src.Terraform import terraform, Plan, Tunnel, Span, Intensity
    terrain, palette = ground_at(lambda x, z: 81)
    result = terraform(Plan(
        tunnel=Tunnel(sides=True, min_cover=3), margin=12, reach=12,
        spans=(Span(start=(0, 0), stop=(19, 0), profile='tunnel',
                    intensity=Intensity.NONE, headroom=0),)),
        terrain, straight_track(), palette,
        envelope={(x, z): [70, 80] for x in range(20) for z in range(-2, 3)})
    assert np.array_equal(result.designed, result.natural)
    assert result.report.tunnel_side_columns == 0
    assert len(result.fill) == 0


def test_voxel_repair_does_not_leave_a_cap_floating_over_eroded_sides():
    blocks, palette, grid, natural, designed, cut, eligible = fixture()
    # The centre's top is a natural knob: two neighbours stand a block lower.
    for x, z in ((3, 4), (4, 3)):
        blocks.remove_block(BlockPos(x, 20, z))
        natural[grid.at(x, z)] = 19
    cut = set()
    for x, z in ((3, 4), (5, 4), (4, 3), (4, 5)):
        designed[grid.at(x, z)] = 16
        cut.update(BlockPos(x, y, z) for y in range(17, int(natural[grid.at(x, z)]) + 1))
    designed[grid.at(4, 4)] = 20
    eroded, _ = repair(FieldTerrain(blocks), grid, natural, designed,
                       grid.filled(True, dtype=bool), eligible, BlockField(palette),
                       cut, Clearance.none(grid), grid.filled(smoothing.Mode.MELT),
                       Smoothing(behavior='melt', face_iterations=4), max_drop=1)
    centre = {p.y for p in eroded if p.x == 4 and p.z == 4}
    assert centre  # the remnant's sides are eroded
    survivors = [y for y in range(5, 21) if y not in centre]
    assert survivors == list(range(5, max(survivors) + 1))  # nothing left over a gap
