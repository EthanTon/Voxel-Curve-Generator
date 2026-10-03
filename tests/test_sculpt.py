"""Shaping the ground the works change: cliffs, gentle grades, banks, spills and skins."""
import numpy as np
import pytest

from src.Core.Field.block_field import BlockField
from src.Core.block.palette import BlockPalette
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Terraform import Grid, sculpt
from src.Terraform.plan import Sculpt, read

SHAPE = (21, 61)
MIDDLE = SHAPE[1] // 2


def rates(rock=3.0, soil=0.5, bank=0.4, shape=SHAPE):
    return sculpt.Rates(np.full(shape, rock), np.full(shape, soil), np.full(shape, bank))


def trench(natural, bed, width=2):
    # A bed of fixed columns down the middle, every other column free.
    fixed = np.zeros(natural.shape, bool)
    fixed[:, MIDDLE - width:MIDDLE + width + 1] = True
    heights = np.where(fixed, bed, natural)
    return fixed, heights, ~fixed


def design(natural, fixed, heights, free, rock_top, rate=None, cap=None, floor=None,
           fill_only=None, hold=None):
    return sculpt.design(natural, fixed, heights, free, rock_top, rate or rates(), cap, floor,
                         fill_only, hold)


def steps(heights, axis=1):
    return np.abs(np.diff(np.floor(heights + 0.5), axis=axis))


def test_a_cut_reaching_rock_stands_as_a_compact_cliff():
    natural = np.full(SHAPE, 100.0)
    fixed, heights, free = trench(natural, 80.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 96.0)).heights
    row = found[10]
    # Twenty blocks up within a handful of columns, soil cap and all.
    beside = row[MIDDLE + 3:]
    assert beside[0] >= 82 and np.argmax(beside >= 100) <= 7
    assert np.all(row <= natural[10]) and np.all(row >= 80)


def test_a_shallow_cut_grades_back_gently_through_the_soil():
    natural = np.full(SHAPE, 100.0)
    fixed, heights, free = trench(natural, 97.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 80.0)).heights
    row = found[10, MIDDLE + 2:]
    assert np.all(np.diff(row) <= 0.5 + 1e-9)
    assert np.all(steps(found)[free[:, 1:] & free[:, :-1]] <= 1)
    assert row[-1] == 100 and np.count_nonzero(row < 99.5) >= 5


def test_a_cut_into_rising_ground_runs_out_instead_of_shaving_the_hillside():
    columns = np.arange(SHAPE[1], dtype=float)
    natural = np.tile(100.0 + 0.5 * np.maximum(columns - MIDDLE, 0), (SHAPE[0], 1))
    fixed, heights, free = trench(natural, 97.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, -np.inf)).heights
    # The hill rises as fast as the soil may: the cut still meets it.
    assert np.array_equal(found[10, -10:], natural[10, -10:])


def test_a_bank_falls_gently_from_the_formation_and_meets_the_land():
    natural = np.full(SHAPE, 70.0)
    fixed, heights, free = trench(natural, 76.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 60.0)).heights
    row = found[10, MIDDLE + 2:]
    assert np.all(np.diff(row) <= 0 + 1e-9) and np.all(-np.diff(row) <= 0.4 + 1e-9)
    assert row[-1] == 70 and np.all(found >= natural)


def test_a_bank_stops_at_a_natural_cliff_rather_than_filling_the_ravine_below():
    natural = np.full(SHAPE, 70.0)
    natural[:, MIDDLE + 6:MIDDLE + 12] = 50.0
    fixed, heights, free = trench(natural, 76.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 40.0)).heights
    assert np.array_equal(found[:, MIDDLE + 6:MIDDLE + 12], natural[:, MIDDLE + 6:MIDDLE + 12])


def test_a_bank_does_not_pass_under_a_deck_that_leaves_it_no_room():
    natural = np.full(SHAPE, 70.0)
    fixed, heights, free = trench(natural, 76.0)
    cap = np.full(SHAPE, np.inf)
    cap[:, MIDDLE + 3:MIDDLE + 5] = 70.0
    found = design(natural, fixed, heights, free, np.full(SHAPE, 40.0), cap=cap).heights
    assert np.all(found[:, MIDDLE + 3:MIDDLE + 5] <= 70)
    assert np.array_equal(found[:, MIDDLE + 5:], natural[:, MIDDLE + 5:])


def test_held_ground_bounds_a_bank_beside_it():
    natural = np.full(SHAPE, 70.0)
    fixed, heights, free = trench(natural, 76.0)
    # A pier's kept ground, held at the land's height, four columns out.
    fixed[:, MIDDLE + 7] = True
    free[:, MIDDLE + 7] = False
    found = design(natural, fixed, heights, free, np.full(SHAPE, 40.0)).heights
    assert np.all(found[:, MIDDLE + 6] <= 70.4 + 1e-9)


def banked(shape=SHAPE, land=70.0, top=76.0):
    # A raised band down the middle, every other column free.
    natural = np.full(shape, land)
    fixed, heights, free = trench(natural, top)
    return natural, fixed, heights, free


def test_a_bank_spills_over_ground_kept_beside_a_pier_but_never_cuts_it():
    natural, fixed, heights, free = banked()
    kept = np.zeros(SHAPE, bool)
    kept[:, MIDDLE + 3:MIDDLE + 8] = True
    found = design(natural, fixed, heights, free, np.full(SHAPE, 50.0), fill_only=kept).heights
    assert np.all(found[:, MIDDLE + 3] > 74)
    # Beside a cut the kept ground stays where the rest is graded down.
    natural = np.full(SHAPE, 70.0)
    fixed, heights, free = trench(natural, 64.0)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 50.0), fill_only=kept).heights
    assert np.all(found[kept] == 70) and np.all(found[:, MIDDLE - 3] < 70)


def test_a_bank_tapers_out_at_a_deep_natural_drop_it_reaches():
    natural, fixed, heights, free = banked()
    rim = MIDDLE + 14
    natural[:, rim + 1:] = 40.0
    heights = np.where(fixed, heights, natural)
    found = design(natural, fixed, heights, free, np.full(SHAPE, 30.0)).heights
    assert np.all(found[:, rim] == 70) and np.all(found[:, rim + 1:] == 40)
    assert np.all(found[:, MIDDLE + 6] > 71)
    # A ledge inside the ravine the bank never reaches holds nothing.
    ledged = natural.copy()
    ledged[:, rim + 6:rim + 9] = 55.0
    again = design(ledged, fixed, np.where(fixed, heights, ledged), free, np.full(SHAPE, 30.0)).heights
    assert np.array_equal(again[:, :rim + 1], found[:, :rim + 1])


def test_a_capped_column_holds_its_neighbours_at_its_hold():
    natural = np.full(SHAPE, 79.0)
    free = np.ones(SHAPE, bool)
    fixed = ~free
    cap = np.full(SHAPE, np.inf)
    cap[:, MIDDLE] = 73.0
    pulled = design(natural, fixed, natural, free, np.full(SHAPE, 60.0), cap=cap).heights
    held = design(natural, fixed, natural, free, np.full(SHAPE, 60.0), cap=cap,
                  hold=np.where(np.isfinite(cap), 79.0, cap)).heights
    assert np.all(pulled[:, MIDDLE] == 73) and np.all(held[:, MIDDLE] == 73)
    assert np.all(pulled[:, MIDDLE + 1] < 79)
    assert np.all(held[:, MIDDLE + 1] == 79)


def test_texture_never_raises_ground_past_the_land():
    natural = np.full(SHAPE, 90.0)
    natural[:, :MIDDLE] = 70.0
    fixed, heights, free = trench(natural, 70.0)
    shaped = design(natural, fixed, heights, free, np.full(SHAPE, 60.0))
    textured = sculpt.texture(shaped.heights, natural, fixed, free, heights, np.full(SHAPE, 60.0),
                              np.full(SHAPE, 95.0), Grid((0, 0), SHAPE[1], SHAPE[0]), 1, (0, 0), 4.0)
    assert np.all(textured[free] <= natural[free] + 1.5 + 1e-9)
    assert np.all(textured[free & (shaped.heights < natural)] <= natural[free & (shaped.heights < natural)])


def test_narrow_ridges_and_grooves_are_levelled_inside_the_region_only():
    heights = np.full((9, 9), 80.0)
    heights[4, :] = 84.0
    heights[:, 2] = 76.0
    region = np.ones(heights.shape, bool)
    region[:, 7:] = False
    levelled = sculpt.level_fins(heights, region)
    assert np.all(levelled[region] == 80)
    assert np.array_equal(levelled[:, 7:], heights[:, 7:])


def test_bank_finishing_rounds_contours_without_moving_the_mound_or_its_toe():
    rows, cells = np.indices((81, 121))
    base = np.full(rows.shape, 60.)
    mound = np.maximum(0, 8 * np.exp(-((cells - 60) / 28) ** 2 - ((rows - 40) / 21) ** 2) - 1)
    heights = base + mound
    heights += np.where(mound > 2, .4 * np.sin(cells * 1.7) * np.sin(rows * 1.3), 0)
    result = sculpt.smooth_banks(heights, base, np.ones(rows.shape, bool), 2.5, 3)
    middle = (mound[1:-1, 1:-1] > 2) & (mound[1:-1, 1:-1] < 5)

    def roughness(surface):
        curvature = (surface[2:, 1:-1] + surface[:-2, 1:-1] + surface[1:-1, 2:]
                     + surface[1:-1, :-2] - 4 * surface[1:-1, 1:-1])
        return np.mean(curvature[middle] ** 2)

    assert roughness(result) < .1 * roughness(heights)
    assert np.unravel_index(np.argmax(result), result.shape) == (40, 60)
    assert abs(np.sum(result - base) / np.sum(heights - base) - 1) < .01
    assert np.array_equal(result > base + .5, heights > base + .5)
    assert np.array_equal(result[heights <= base + .5], heights[heights <= base + .5])
    assert np.all(result >= base)
    assert np.max(np.abs(result - heights)) <= 1
    assert np.array_equal(result, sculpt.smooth_banks(heights, base, np.ones(rows.shape, bool), 2.5, 3))


@pytest.mark.parametrize("along", (False, True))
def test_bank_finishing_holds_cuts_cliffs_contacts_and_masked_banks(along):
    base = np.full((31, 51), 70.)
    heights = base.copy()
    heights[:, :10] = 62 + .4 * np.sin(np.arange(31)[:, None])  # rock cut
    heights[:, 20:] = 76
    heights[10:20, 26:32] += .8
    free = np.ones(base.shape, bool)
    free[:, 37:40] = False  # fixed contact/keep-out separating two banks
    stations = np.indices(base.shape)[0].astype(float) if along else None
    result = sculpt.smooth_banks(heights, base, free, stations=stations)
    assert np.array_equal(result[:, :22], heights[:, :22])
    assert np.array_equal(result[~free], heights[~free])
    assert np.array_equal(result[:, 40:], heights[:, 40:])
    assert not np.array_equal(result[:, 24:35], heights[:, 24:35])
    # Changing the other bank cannot influence this one across the held strip.
    other = heights.copy()
    other[:, 40:] += .4 * np.sin(np.arange(31)[:, None])
    again = sculpt.smooth_banks(other, base, free, stations=stations)
    assert np.array_equal(result[:, :37], again[:, :37])


@pytest.mark.parametrize("wavelength,radius,along", [(10, 3, False), (20, 6, True)])
def test_bank_finishing_smooths_rounded_terrace_ripples(wavelength, radius, along):
    rows, cells = np.indices((101, 81))
    base = np.full(rows.shape, 60.)
    heights = base + np.maximum(0, 8 - .3 * np.abs(cells - 40))
    heights += np.where(heights > base + 1, .75 * np.sin(2 * np.pi * rows / wavelength), 0)
    free = np.abs(cells - 40) > 2
    result = sculpt.smooth_banks(heights, base, free, radius, 8,
                                stations=rows.astype(float) if along else None)

    def terrace(surface):
        return np.sum(np.floor(surface[10:-10, 43:] + .5) >= 65, axis=1)

    before, after = terrace(heights), terrace(result)
    assert np.std(after) < .4 * np.std(before)
    assert np.mean(np.diff(after, n=2) ** 2) < .65 * np.mean(np.diff(before, n=2) ** 2)
    assert abs(np.sum(result - base) / np.sum(heights - base) - 1) < (.02 if along else .01)
    assert np.array_equal(result[~free], heights[~free])


def test_narrow_bank_fill_smooths_without_crossing_its_footprint():
    rows, cells = np.indices((61, 21))
    base = np.full(rows.shape, 60.)
    bank = (cells >= 10) & (cells <= 11)
    heights = np.where(bank, 61 + .3 * np.sin(2 * np.pi * rows / 10), base)
    result = sculpt.smooth_banks(heights, base, np.ones(base.shape, bool), 3, 8)
    assert np.std(result[10:-10, 10]) < .4 * np.std(heights[10:-10, 10])
    assert np.array_equal(result[~bank], base[~bank])
    assert np.array_equal(result > base + .5, bank)


def test_route_directed_bank_finishing_rotates_with_the_path():
    rows, cells = np.indices((61, 81))
    base = np.full(rows.shape, 60.)
    heights = base + np.maximum(0, 8 - .3 * np.abs(cells - 40))
    heights += np.where(heights > base + 2, .6 * np.sin(2 * np.pi * rows / 15), 0)
    free = np.abs(cells - 40) > 2
    stations = rows.astype(float)
    result = sculpt.smooth_banks(heights, base, free, 4, 4, stations=stations)
    rotated = sculpt.smooth_banks(np.rot90(heights), np.rot90(base), np.rot90(free),
                                 4, 4, stations=np.rot90(stations))
    assert np.allclose(rotated, np.rot90(result), rtol=0, atol=1e-10)
    assert np.array_equal(result[~free], heights[~free])
    assert np.max(np.abs(result - heights)) <= 1


@pytest.mark.parametrize("along", (False, True))
def test_bank_finishing_cannot_add_block_steps_or_spend_its_budget_per_round(along):
    rows, cells = np.indices((35, 45))
    base = np.full(rows.shape, 60.)
    heights = base + np.maximum(0, 6 - .35 * np.abs(cells - 22))
    heights += np.where(heights > 62, .3 * np.sin(rows * 1.7), 0)
    stations = rows.astype(float) if along else None
    result = sculpt.smooth_banks(heights, base, np.ones(rows.shape, bool), 2, 30, .35,
                                stations=stations)
    assert np.max(np.abs(result - heights)) <= .35 + 1e-9
    assert np.any(np.floor(result + .5) != np.floor(heights + .5))
    for axis in (0, 1):
        before = np.abs(np.diff(np.floor(heights + .5), axis=axis))
        after = np.abs(np.diff(np.floor(result + .5), axis=axis))
        assert np.all(after <= np.maximum(1, before))
    for radius, passes, most in ((0, 3, 1), (2, 0, 1), (2, 3, 0)):
        assert np.array_equal(heights, sculpt.smooth_banks(heights, base, np.ones(rows.shape, bool),
                                                       radius, passes, most))


@pytest.mark.parametrize("behavior", ("auto", "grow", "melt"))
def test_sculpt_bank_finish_obeys_explicit_direction_and_keeps_structure(behavior):
    from dataclasses import replace
    from src.Terraform import Ground, Plan
    from src.Terraform.clearance import Clearance
    from src.Terraform.smoothing import Context, Mode
    from src.Terraform.terraform import _sculpted

    grid = Grid((0, 0), 61, 41)
    rows, cells = np.indices(grid.shape)
    base = grid.filled(60.)
    distance = np.maximum(np.abs(cells - 30) - 2, 0)
    heights = np.where(distance <= 3, 69 + .7 * np.sin(rows / 3), base)
    ground = Ground(grid, base, distance, heights, heights, grid.filled(0))
    masks = grid.filled(False, dtype=bool)
    plan = Plan(shoulder=3, sculpt=Sculpt(enabled=True, adapt=False))
    options = replace(plan.sculpt, bank_passes=0)
    code = -1 if behavior == "auto" else Mode[behavior.upper()]
    policy = Context(grid.filled(code, dtype=np.int8), grid.filled(0))

    def run(selected):
        return _sculpted(selected, grid, ground, base, heights, ~masks, ~masks,
                         masks, masks, base, masks, grid.filled(-np.inf), Clearance.none(grid),
                         heights, None, masks, grid.filled(-np.inf), policy)[0]

    reference = run(replace(plan, sculpt=options))
    result = run(plan)
    assert not np.array_equal(result, reference)
    assert np.array_equal(result[distance <= 3], reference[distance <= 3])
    assert np.max(np.abs(result - reference)) <= plan.smoothing.stable_change
    assert np.all(result >= base)
    if behavior == "grow":
        assert np.all(result >= reference)
    elif behavior == "melt":
        assert np.all(result <= reference)
    else:
        assert np.any(result < reference) and np.any(result > reference)


def test_riser_cap_raises_only_toward_the_shaped_surface():
    heights = np.array([[80.0, 70.0, 80.0]])
    capped = sculpt.cap_risers(heights, np.array([[80.0, 74.0, 80.0]]), np.ones(heights.shape, bool), 4.0)
    assert capped[0, 1] == 74
    capped = sculpt.cap_risers(heights, np.array([[80.0, 79.0, 80.0]]), np.ones(heights.shape, bool), 4.0)
    assert capped[0, 1] == 76


def test_rates_follow_the_land_within_their_ranges():
    grid = Grid((0, 0), 40, 40)
    gentle = sculpt.rates_for(grid, 3, (2.5, 4.0), (0.25, 0.5), (0.25, 0.45),
                              (np.full((40, 40), 0.25), np.full((40, 40), 2.0)))
    steep = sculpt.rates_for(grid, 3, (2.5, 4.0), (0.25, 0.5), (0.25, 0.45),
                             (np.full((40, 40), 1.0), np.full((40, 40), 6.0)))
    for each in (gentle, steep):
        assert each.rock.min() >= 2.5 and each.rock.max() <= 4.0
        assert each.soil.min() >= 0.25 and each.soil.max() <= 0.5
    assert steep.rock.mean() > gentle.rock.mean() and steep.soil.mean() > gentle.soil.mean()


def test_native_steepness_reads_the_hillsides_round_each_column():
    rows, cells = np.indices((64, 64))
    natural = np.where(cells < 32, 60.0 + 0.25 * cells, 60.0 + 2.0 * cells)
    gentle, steep = sculpt.native_steepness(natural, np.ones(natural.shape, bool), (0.5, 0.97),
                                            tile=8, reach=1)
    assert gentle[32, 4] <= 0.5 and steep[32, 60] >= 1.5


@pytest.mark.parametrize("data, message", [
    ({"enabled": "yes"}, "sculpt.enabled"),
    ({"rock_rise": [3, 1]}, "rock_rise"),
    ({"soil_rise": [0, 1]}, "soil_rise"),
    ({"bank_rise": [0.5]}, "bank_rise"),
    ({"ledges": [1.5, 3]}, "ledges"),
    ({"adapt": 1}, "adapt"),
    ({"bank_radius": 0}, "bank_radius"),
    ({"bank_radius": float("nan")}, "bank_radius"),
    ({"bank_radius": float("inf")}, "bank_radius"),
    ({"bank_radius": True}, "bank_radius"),
    ({"bank_passes": -1}, "bank_passes"),
    ({"bank_passes": 1.5}, "bank_passes"),
    ({"bank_passes": True}, "bank_passes"),
    ({"slope": 1}, "unknown sculpt option"),
])
def test_invalid_sculpt_options_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        read({"terraform": {"sculpt": data}})


def test_sculpt_is_opt_in_and_configurable():
    assert not read({}).sculpt.enabled
    found = read({"terraform": {"sculpt": {
        "enabled": True, "rock_rise": [2, 3], "soil_rise": [0.3, 0.6], "bank_rise": [0.2, 0.4],
        "ledges": [0, 0], "adapt": False}}}).sculpt
    assert found == Sculpt(True, (2.0, 3.0), (0.3, 0.6), (0.2, 0.4), (0, 0), False)
    assert Sculpt().enabled is False
    finish = read({"terraform": {"sculpt": {"bank_radius": 4.5, "bank_passes": 0}}}).sculpt
    assert finish.bank_radius == 4.5 and finish.bank_passes == 0


def test_cut_does_not_paint_a_floating_top_into_a_saved_cave():
    from src.Terraform.materialize import Strata, build
    grid = Grid((0, 0), 2, 1)
    natural = np.array([[82.0, 82.0]])
    designed = np.array([[80.0, 80.0]])
    layers = np.zeros((16, 1, 2), dtype=np.int16)
    layers[2, 0, 1] = -1  # air at y = 80 under the second column's shelf
    grass = BlockState.create("minecraft:grass_block")
    strata = Strata(grass, BlockState.create("minecraft:dirt"), BlockState.create("minecraft:stone"))
    made = build(grid, natural, designed, strata, BlockPalette(), subsurface=layers)
    assert made.fill.get_block(BlockPos(0, 80, 0)) == grass
    assert made.fill.get_block(BlockPos(1, 80, 0)) is None
    assert {BlockPos(1, 81, 0), BlockPos(1, 82, 0)} <= made.cut


def test_a_cut_rock_face_stays_bare_stone_but_a_wide_ledge_takes_the_top():
    from src.Terraform.clearance import Clearance
    from src.Terraform.materialize import Strata, finish_surfaces
    from src.Terraform.terrain import FieldTerrain
    stone, grass = BlockState.create("minecraft:stone"), BlockState.create("minecraft:grass_block")
    palette = BlockPalette()
    blocks = BlockField(palette)
    for x in range(6):
        for y in range(60, 91):
            blocks.set_block(BlockPos(x, y, 0), grass if y == 90 else stone)
    terrain = FieldTerrain(blocks)
    grid = Grid((0, 0), 6, 1)
    natural = np.full((1, 6), 90.0)
    designed = np.array([[80.0, 80.0, 80.0, 83.0, 86.0, 89.0]])
    cut = frozenset(BlockPos(x, y, 0) for x in range(6) for y in range(int(designed[0, x]) + 1, 91))
    fill = BlockField(palette)
    strata = Strata(grass, BlockState.create("minecraft:dirt"), stone)
    skin = np.ones((1, 6), bool)
    finish_surfaces(terrain, grid, natural, designed, skin, fill, cut, strata, None,
                    np.full((1, 6), -32768), Clearance.none(grid), skin=skin,
                    rock_surface=np.full((1, 6), 88.0))
    # The two-wide floor at 80 takes grass and nothing under it; the risers stay stone.
    assert fill.get_block(BlockPos(1, 80, 0)) == grass and fill.get_block(BlockPos(1, 79, 0)) is None
    assert fill.get_block(BlockPos(3, 83, 0)) is None and fill.get_block(BlockPos(4, 86, 0)) is None


def test_a_shelf_left_touching_nothing_by_a_cut_is_released():
    from src.Terraform.faces import floating_remnants
    from src.Terraform.terrain import FieldTerrain
    stone = BlockState.create("minecraft:stone")
    palette = BlockPalette()
    blocks = BlockField(palette)
    for x in range(4):
        for y in range(60, 66):
            blocks.set_block(BlockPos(x, y, 0), stone)
        # A native shelf two blocks thick over a hollow at 66-67.
        for y in (68, 69):
            blocks.set_block(BlockPos(x, y, 0), stone)
    terrain = FieldTerrain(blocks)
    grid = Grid((0, 0), 4, 1)
    # The cut takes the shelf over the first and last columns, leaving the
    # middle two hanging together.
    cut = frozenset(BlockPos(x, y, 0) for x in (0, 3) for y in (68, 69))
    found = floating_remnants(terrain, grid, np.ones((1, 4), bool), BlockField(palette), cut)
    assert found == {BlockPos(x, y, 0) for x in (1, 2) for y in (68, 69)}
    # The same shelf still joined to land past the area stays.
    area = np.array([[True, True, True, False]])
    cut = frozenset(BlockPos(0, y, 0) for y in (68, 69))
    assert not floating_remnants(terrain, grid, area, BlockField(palette), cut)


def test_terraform_sculpt_runs_a_bank_round_a_pier_at_the_bridge_end():
    from dataclasses import replace
    from tests.test_terraform import ground_at, straight_track
    from src.Terraform import terraform, Plan, Span, Intensity
    from src.Terraform.plan import Bridge, Embankment
    terrain, palette = ground_at(lambda x, z: 60)
    pier = [BlockPos(x, y, z) for x in (15, 16) for z in (-1, 0, 1) for y in range(61, 70)]
    plan = Plan(
        bleed=True, margin=16, reach=16, embankment=Embankment(shape="natural"),
        bridge=Bridge(natural_only=True), preserve_fittings=("pier",), fitting_buffer=4,
        spans=(Span(start=(0, 0), stop=(14, 0), profile="embankment", intensity=Intensity.SHAPE),
               Span(start=(15, 0), stop=(29, 0), profile="bridge", intensity=Intensity.SHAPE)),
    )

    def build(enabled):
        return terraform(replace(plan, sculpt=replace(plan.sculpt, enabled=enabled)), terrain,
                         straight_track(length=30), palette, fittings={"pier": pier})

    held, sculpted = build(False), build(True)
    beside = [sculpted.grid.at(x, z) for x in (15, 16, 17) for z in (3, 4)]
    # Without the stage the buffer stays a pit; with it the bank runs round the pier.
    assert all(held.relief[at] == 0 for at in beside)
    assert all(sculpted.relief[at] > 0 for at in beside)
    buffer = [sculpted.grid.at(x, z) for x in range(12, 21) for z in range(-6, 7)]
    assert all(sculpted.relief[at] >= 0 for at in buffer)
    for position, _ in sculpted.fill.iter_block_ids():
        assert not sculpted.clearance.holds(*sculpted.grid.at(position.x, position.z), position.y)


def sculpted_once(natural, designed, *, distance=None, kept=None, water=None, bridges=None,
                  bridge_ground=None, seed=None, clearance=None, tops=None, editable=None):
    from src.Terraform import Ground, Plan
    from src.Terraform.clearance import Clearance
    from src.Terraform.terraform import _sculpted
    grid = Grid((0, 0), natural.shape[1], natural.shape[0])
    none = grid.filled(False, dtype=bool)
    distance = grid.filled(50.0) if distance is None else distance
    seed = designed if seed is None else seed
    ground = Ground(grid, natural, distance, seed, seed, grid.filled(0))
    plan = Plan(sculpt=Sculpt(enabled=True, adapt=False))
    return _sculpted(plan, grid, ground, natural, designed, ~none,
                     ~none if editable is None else editable, none if kept is None else kept,
                     none if bridges is None else bridges,
                     natural if bridge_ground is None else bridge_ground, none, grid.filled(-np.inf),
                     clearance or Clearance.none(grid), designed if tops is None else tops, water, none,
                     grid.filled(-np.inf))[0]


def test_native_grooves_beside_held_water_are_left_alone():
    natural = np.full((21, 21), 64.0)
    pond = np.zeros(natural.shape, bool)
    pond[8:13, :5] = True
    natural[pond] = 60.0
    natural[:, 6] = 62.0
    water = np.where(pond, 64.0, -np.inf)
    result = sculpted_once(natural, natural.copy(), water=water, editable=~pond)
    assert np.array_equal(result[~pond], natural[~pond])


def test_zero_shoulder_starts_a_bank_at_the_bed_and_keeps_the_cut_band():
    natural = np.full((21, 61), 70.0)
    distance = np.maximum(np.abs(np.indices(natural.shape)[1] - 30) - 2, 0)
    band = distance <= 3
    raised = sculpted_once(natural, np.where(band, 76.0, natural), distance=distance)
    lowered = sculpted_once(natural, np.where(band, 64.0, natural), distance=distance)
    assert np.all(raised[distance == 3] < 76)
    assert np.all(raised[distance == 0] == 76)
    assert np.all(lowered[band] == 64)


def test_bank_finishing_does_not_fill_a_native_ledge_inside_a_ravine():
    natural = np.full((21, 21), 71.0)
    natural[:, 10:] = 35.0
    natural[:, 10] = 44.0
    shaped = natural.copy()
    shaped[:, :10] = 73.0
    fixed = np.zeros(natural.shape, bool)
    fixed[:, :8] = True
    result = sculpt.texture(
        shaped, natural, fixed, ~fixed, shaped, np.full(natural.shape, -np.inf),
        np.full(natural.shape, -np.inf), Grid((0, 0), 21, 21), 1, (0, 0), 4.0,
    )
    assert np.array_equal(result[:, 10:], natural[:, 10:])


def test_track_through_ground_kept_beside_a_pier_keeps_its_cut():
    natural = np.full((21, 31), 80.0)
    distance = np.maximum(np.abs(np.indices(natural.shape)[0] - 10.0) - 1.0, 0.0)
    track = distance <= 0
    kept = np.zeros(natural.shape, bool)
    kept[:, 12:19] = True
    result = sculpted_once(natural, np.where(track, 70.0, natural), distance=distance, kept=kept)
    assert np.all(result[track] == 70)
    assert np.all(result[kept & ~track] >= 80)


def test_ground_kept_under_a_deck_comes_down_to_the_deck():
    from src.Terraform.clearance import Clearance
    natural = np.full((21, 31), 80.0)
    distance = np.maximum(np.abs(np.indices(natural.shape)[0] - 10.0) - 1.0, 0.0)
    track = distance <= 0
    bridges = np.zeros(natural.shape, bool)
    bridges[:, 15:] = True
    kept = np.zeros(natural.shape, bool)
    kept[:, 13:20] = True
    seed = np.full(natural.shape, 72.0)
    deck = bridges & track
    clearance = Clearance(np.where(deck, 75.0, 1e9), np.where(deck, 80.0, -1e9))
    designed = np.where(track, np.where(bridges, 76.0, 72.0), natural)
    result = sculpted_once(natural, designed, distance=distance, kept=kept, bridges=bridges,
                           bridge_ground=np.where(deck & ~kept, 71.0, natural), seed=seed,
                           clearance=clearance, tops=np.where(deck, 72.0, -np.inf))
    assert np.all(result[deck] <= 71)
    assert np.all(result[kept & ~track] >= 80)


def test_terraform_sculpt_excavates_less_and_keeps_bed_bore_and_roof():
    from dataclasses import replace
    from tests.test_terraform import straight_track, STONE, GRASS
    from src.Terraform import FieldTerrain, terraform, Plan, Tunnel, Span, Intensity
    from src.Terraform.plan import Embankment

    def build(sculpted):
        palette = BlockPalette()
        blocks = BlockField(palette)
        for x in range(-30, 70):
            for z in range(-30, 30):
                top = 95 if x < 20 else max(72, 95 - (x - 20) * 2)
                for y in range(64, top + 1):
                    blocks.set_block(BlockPos(x, y, z), GRASS if y == top else STONE)
        terrain = FieldTerrain(blocks)
        alignment = straight_track(level=70, length=40)
        envelope = {(x, z): ([70, 71, 80] if x < 20 else [70, 71])
                    for x in range(40) for z in range(-2, 3)}
        plan = Plan(
            tunnel=Tunnel(min_cover=2), margin=24, reach=24,
            embankment=Embankment(shape='natural'),
            spans=(Span(start=(0, 0), stop=(19, 0), profile='tunnel',
                        intensity=Intensity.SHAPE, headroom=0),
                   Span(start=(20, 0), stop=(39, 0), profile='cutting',
                        intensity=Intensity.SHAPE)))
        plan = replace(plan, sculpt=replace(plan.sculpt, enabled=sculpted))
        return terraform(plan, terrain, alignment, palette, envelope=envelope,
                         running={(x, z): 71 for x in range(40) for z in range(-2, 3)})

    legacy, sculpted = build(False), build(True)
    assert len(sculpted.cut) <= len(legacy.cut)
    for x in (5, 30):
        assert all(BlockPos(x, y, 0) not in sculpted.cut for y in (68, 69, 70))
    assert BlockPos(5, 75, 0) in sculpted.cut  # the bore is still carved
    assert sculpted.designed[sculpted.grid.at(5, 0)] >= 82  # roof cover over the lining
    for position, _ in sculpted.fill.iter_block_ids():
        assert not sculpted.clearance.holds(*sculpted.grid.at(position.x, position.z), position.y)


@pytest.mark.parametrize("bank_passes", (0, 4))
def test_sculpted_tunnel_clears_its_full_width_and_height_on_a_grade(bank_passes):
    from src.Terraform import Alignment, FieldTerrain, Intensity, Plan, Span, Tunnel, terraform
    from src.Terraform.plan import Embankment

    stone = BlockState.create("minecraft:stone")
    palette = BlockPalette()
    blocks = BlockField(palette)
    for x in range(-8, 20):
        for z in range(-10, 11):
            for y in range(58, 61 if x < 4 else 96):
                blocks.set_block(BlockPos(x, y, z), stone)
    terrain = FieldTerrain(blocks)
    bed = lambda x: 70 + x // 4
    alignment = Alignment.of([BlockPos(x, bed(x), z) for x in range(12) for z in range(-3, 4)],
                             [x for x in range(12) for z in range(-3, 4)])
    envelope = {}
    for x in range(12):
        for z in range(-3, 4):
            levels = [bed(x)]
            if x >= 4:
                levels += (list(range(bed(x) + 1, bed(x) + 9)) if abs(z) == 3
                           else [bed(x) + 8])
            if abs(z) == 1:
                levels += [bed(x) + 1, bed(x) + 6]  # rails and overhead wire
            envelope[x, z] = levels
    plan = Plan(
        margin=8, reach=8, embankment=Embankment(shape="natural"),
        sculpt=Sculpt(enabled=True, bank_passes=bank_passes),
        tunnel=Tunnel(sides=True, min_cover=2),
        spans=(Span((0, 0), (3, 0), profile="embankment", intensity=Intensity.SHAPE),
               Span((4, 0), (11, 0), profile="tunnel", intensity=Intensity.SHAPE, headroom=0)),
    )
    result = terraform(plan, terrain, alignment, palette, envelope=envelope,
                       running={(x, z): bed(x) + 1 for x in range(12) for z in range(-3, 4)})
    for x in range(4, 12):
        for z in range(-2, 3):
            # Check the entire cavity, including columns between the rails,
            # both portals and the upper gaps around the overhead wire.
            for y in range(bed(x) + 1, bed(x) + 8):
                if y in envelope[x, z]:
                    continue
                position = BlockPos(x, y, z)
                assert position in result.cut
                assert result.fill.get_block(position) is None
            assert BlockPos(x, bed(x), z) not in result.cut
            assert BlockPos(x, bed(x) + 8, z) not in result.cut
            assert result.designed[result.grid.at(x, z)] >= bed(x) + 10
    assert not any(result.clearance.holds(*result.grid.at(p.x, p.z), p.y)
                   for p, _ in result.fill.iter_block_ids())
