"""Material direction, deterministic local decisions and section meeting rules."""
from dataclasses import replace

import numpy as np
import pytest

from src.Core.coordinates import BlockPos, ColumnPos
from src.Terraform import Alignment, FieldTerrain, Ground, Grid, Intensity, Plan, Span, terraform
from src.Terraform.plan import Bridge, Keepout, Smoothing, SmoothingTransition, read
from src.Terraform.smoothing import Mode, brush, choose, context, guard_steps
from src.Terraform.terrain import ABSENT
from tests.test_terraform import ground_at, straight_track


@pytest.mark.parametrize("mode", list(Mode))
def test_material_direction_and_stable_budget_survive_many_passes(mode):
    surface = np.full((13, 13), 70.0)
    surface[3:6, 3:6] = 64.0
    surface[7:10, 7:10] = 76.0
    options = Smoothing(stable_change=1)
    result = brush(surface, np.ones_like(surface), np.full(surface.shape, mode), options, 1.5, 20)
    if mode == Mode.GROW:
        assert np.all(result >= surface)
        assert result[4, 4] > surface[4, 4]
    elif mode == Mode.MELT:
        assert np.all(result <= surface)
        assert result[8, 8] < surface[8, 8]
    else:
        assert np.max(np.abs(result - surface)) <= 1
        assert result[4, 4] > surface[4, 4]
        assert result[8, 8] < surface[8, 8]
    assert np.array_equal(result, brush(surface, np.ones_like(surface),
                                       np.full(surface.shape, mode), options, 1.5, 20))


def test_unknown_and_pinned_columns_do_not_pull_the_brush_down():
    surface = np.full((9, 9), 70.0)
    surface[:, :3] = ABSENT
    surface[4, 6] = 73
    area = np.ones_like(surface)
    area[4, 6] = 0
    result = brush(surface, area, np.zeros(surface.shape), Smoothing(), 2, 5)
    assert np.array_equal(result[:, :3], surface[:, :3])
    assert np.min(result[:, 3:]) >= 70
    assert result[4, 6] == 73


def test_auto_decisions_are_local_to_the_surface_being_designed():
    natural = np.array([[60., 70., 80.]])
    planned = np.full(natural.shape, 70.)
    assert choose(planned, natural).tolist() == [[Mode.GROW, Mode.STABLE, Mode.MELT]]
    assert choose(planned, np.array([[69., 70., 71.]])).tolist() == [[Mode.STABLE] * 3]


def meeting_context(options=Smoothing(), names=("land", "span")):
    alignment = straight_track(length=30)
    grid = Grid.around(alignment.columns, 16)
    natural = grid.filled(70.)
    xs = np.arange(grid.width)[None, :] + grid.origin.x
    station = np.broadcast_to(np.clip(xs, 0, 29), grid.shape).astype(float)
    zs = np.arange(grid.depth)[:, None] + grid.origin.z
    distance = np.broadcast_to(np.maximum(np.abs(zs) - 2, 0), grid.shape)
    ground = Ground(grid, natural, distance, natural, natural, station)
    stretches = (Span((0, 0), (14, 0), profile=names[0]),
                 Span((15, 0), (29, 0), profile=names[1]))
    plan = Plan(smoothing=options, spans=stretches)
    return grid, context(plan, ground, alignment, plan.stretches(alignment.locate, alignment.length))


def test_universal_transition_detection_uses_geometry_not_preset_names():
    grid, first = meeting_context(names=("embankment", "bridge"))
    _, renamed = meeting_context(names=("my_bank", "my_deck"))
    assert np.array_equal(first.transition_weight, renamed.transition_weight)
    assert first.transition_weight[grid.at(15, 5)] > 0
    assert first.transition_weight[grid.at(0, 5)] == 0
    assert first.transition_weight[grid.at(15, 16)] == 0
    assert np.array_equal(first.overrides, renamed.overrides)


def test_transition_overrides_have_precedence_and_follow_travel_sides():
    options = Smoothing(behavior="grow", profiles={"span": "stable"},
                        transitions=(SmoothingTransition("land", "span", "melt", side="left"),))
    grid, resolved = meeting_context(options)
    modes = resolved.modes(grid.filled(70), grid.filled(70), options)
    assert modes[grid.at(15, -5)] == Mode.MELT
    assert modes[grid.at(15, 5)] == Mode.STABLE
    assert modes[grid.at(0, -5)] == Mode.GROW
    assert modes[grid.at(29, -5)] == Mode.STABLE


def test_transition_overrides_apply_in_reverse_and_can_be_directional():
    rule = SmoothingTransition("span", "land", "melt", reach=6)
    grid, resolved = meeting_context(Smoothing(transitions=(rule,)))
    assert resolved.overrides[grid.at(15, 4)] == Mode.MELT
    _, directed = meeting_context(Smoothing(transitions=(replace(rule, bidirectional=False),)))
    assert np.all(directed.overrides == -1)


def test_smaller_default_reach_does_not_truncate_a_rule_and_zero_disables_transitions():
    options = Smoothing(transition_reach=2, transitions=(SmoothingTransition("land", "span", "melt", 12),))
    grid, resolved = meeting_context(options)
    assert resolved.overrides[grid.at(20, 5)] == Mode.MELT
    _, off = meeting_context(Smoothing(transition_reach=0))
    assert not off.transition_weight.any()


def test_an_explicit_rule_can_shorten_or_disable_one_transition():
    options = Smoothing(transitions=(SmoothingTransition("land", "span", "auto", 4),))
    grid, resolved = meeting_context(options)
    assert resolved.transition_weight[grid.at(15, 5)] > 0
    assert resolved.transition_weight[grid.at(15, 8)] == 0
    _, off = meeting_context(Smoothing(transitions=(SmoothingTransition("land", "span", "auto", 0),)))
    assert not off.transition_weight.any()


def test_a_brush_cannot_create_a_new_step_at_an_immovable_edge():
    reference = np.full((7, 7), 70.)
    candidate = reference.copy()
    candidate[2:5, 2:5] = 72
    result = guard_steps(reference, candidate, np.ones(reference.shape, dtype=bool), 1)
    assert max(np.max(np.abs(np.diff(result, axis=axis))) for axis in (0, 1)) <= 1
    # A pre-existing natural cliff is preserved rather than flattened away.
    reference[:, :3] = 80
    assert np.array_equal(guard_steps(reference, reference.copy(), np.ones(reference.shape, bool), 1), reference)


def test_a_section_meeting_does_not_brush_intact_contours_or_rock_outcrops():
    from src.Terraform.clearance import Clearance
    from src.Terraform.terraform import _smooth_transitions
    grid, resolved = meeting_context()
    natural = grid.filled(70.)
    natural[grid.at(15, 7)] = 74  # An existing outcrop within the meeting disk.
    natural[grid.at(15, -7)] = 66
    ground = Ground(grid, natural, grid.filled(5), natural, natural, grid.filled(15))
    masks = grid.filled(False, dtype=bool)
    result = _smooth_transitions(Plan(), ground, natural.copy(), natural,
                                 ~masks, resolved, choose(natural, natural), masks,
                                 masks, masks, Clearance.none(grid))
    assert np.array_equal(result, natural)


def test_a_meeting_feathers_a_cut_locally_and_preserves_nearby_native_rock():
    from src.Terraform.clearance import Clearance
    from src.Terraform.terraform import _smooth_transitions
    grid, resolved = meeting_context()
    natural = grid.filled(70.)
    natural[grid.at(15, 9)] = 74
    designed = natural.copy()
    for x in range(13, 18):
        for z in range(2, 5):
            designed[grid.at(x, z)] = 68
    ground = Ground(grid, natural, grid.filled(5), natural, natural, grid.filled(15))
    masks = grid.filled(False, dtype=bool)
    result = _smooth_transitions(Plan(), ground, designed, natural,
                                 ~masks, resolved, choose(designed, natural), masks,
                                 masks, masks, Clearance.none(grid))
    assert not np.array_equal(result, designed)
    assert result[grid.at(15, 9)] == 74
    assert result[grid.at(15, 8)] == 70
    assert result[grid.at(15, -7)] == 70
    assert np.max(np.abs(result - designed)) <= 1


def test_terraform_defaults_choose_different_modes_on_opposite_sides_and_repeat_exactly():
    terrain, palette = ground_at(lambda x, z: 79 if z < 0 else 60)
    alignment = straight_track()
    plan = Plan(default=Intensity.SHAPE, headroom=0, margin=16, reach=16)
    first = terraform(plan, terrain, alignment, palette)
    second = terraform(plan, terrain, alignment, palette)
    assert first.smoothing_modes[first.grid.at(10, -5)] == Mode.MELT
    assert first.smoothing_modes[first.grid.at(10, 5)] == Mode.GROW
    assert np.array_equal(first.designed, second.designed)
    assert np.array_equal(first.smoothing_modes, second.smoothing_modes)


def test_bridge_smoothing_keeps_native_faces_but_still_smooths_small_bumps():
    from src.Terraform.terraform import _bridge_ground
    natural = np.full((21, 21), 60.)
    natural[:, :5] = 70  # A pre-existing cliff at the bank.
    natural[10, 14] = 61  # A small bump on the valley floor.
    known = np.ones(natural.shape, dtype=bool)
    result = _bridge_ground(Plan(), natural, known)
    assert np.array_equal(result[:, 4:6], natural[:, 4:6])
    assert 60 <= result[10, 14] < 61
    assert np.max(np.abs(result - natural)) <= 1
    altered = _bridge_ground(Plan(smoothing=Smoothing(preserve_cliffs=False)), natural, known)
    assert altered[10, 4] < 70
    assert altered[10, 5] > 60


def test_a_cliff_portal_keeps_its_native_faces_beside_the_required_bore():
    terrain, palette = ground_at(lambda x, z: 90 if x >= 10 else 60)
    blocks = [BlockPos(x, y, z) for x in range(30) for z in range(-2, 3) for y in (70, 76)]
    alignment = Alignment.of(blocks, [p.x for p in blocks])
    plan = Plan(margin=24, reach=12, bridge=Bridge(natural_only=True), spans=(
        Span((0, 0), (9, 0), profile="bridge"),
        Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE),
    ))
    result = terraform(plan, terrain, alignment, palette)
    for z in (-5, 5):
        assert result.designed[result.grid.at(9, z)] == 60
        assert result.designed[result.grid.at(10, z)] == 90
        assert not any(p.x == 10 and p.z == z for p in result.cut)
    # The bore and the mouth still meet the route and keep roof cover.
    assert 77 <= result.designed[result.grid.at(11, 0)] < 90


def test_a_bridge_to_tunnel_meeting_grades_the_portal_and_keeps_the_valley_and_roof():
    terrain, palette = ground_at(lambda x, z: 90 if x >= 10 else 60)
    blocks = [BlockPos(x, y, z) for x in range(30) for z in range(-2, 3) for y in (70, 76)]
    alignment = Alignment.of(blocks, [p.x for p in blocks])
    plan = Plan(margin=24, reach=12, bridge=Bridge(natural_only=True),
                spans=(Span((0, 0), (9, 0), profile="bridge"),
                       Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE)))
    result = terraform(plan, terrain, alignment, palette)
    assert 77 <= result.designed[result.grid.at(11, 0)] < 90
    assert result.smoothing_modes[result.grid.at(11, 0)] == Mode.MELT
    assert result.designed[result.grid.at(28, 0)] == 90
    assert result.designed[result.grid.at(4, 0)] <= 60
    assert result.designed[result.grid.at(4, 12)] == 60


def test_visible_tunnel_roof_edges_do_not_get_artificial_ground_cover():
    terrain, palette = ground_at(lambda x, z: 74)
    alignment = straight_track(length=30)
    envelope = {(x, z): [70, 76] if x >= 10 else [70]
                for x in range(30) for z in range(-2, 3)}
    plan = Plan(margin=20, reach=12, headroom=0, bridge=Bridge(natural_only=True), spans=(
        Span((0, 0), (9, 0), profile="bridge"),
        Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE, headroom=0),
    ))
    result = terraform(plan, terrain, alignment, palette, envelope=envelope)
    # Material replacements on the cut surface are still allowed; there
    # must be no added terrain above the original surface to hide the roof.
    assert all(p.y <= 74 for p, _ in result.fill.iter_block_ids())
    assert result.designed.max() <= 74
    for x in range(10, 30):
        assert result.designed[result.grid.at(x, 0)] <= 74
        assert BlockPos(x, 76, 0) not in result.cut


def test_portal_smoothing_cannot_enter_a_keepout():
    terrain, palette = ground_at(lambda x, z: 90 if x >= 10 else 60)
    alignment = straight_track(length=30)
    plan = Plan(margin=20, reach=12, bridge=Bridge(natural_only=True),
                keepouts=(Keepout(ColumnPos(10, -8), ColumnPos(14, -3)),),
                spans=(Span((0, 0), (9, 0), profile="bridge"),
                       Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE)))
    result = terraform(plan, terrain, alignment, palette)
    for x in range(10, 15):
        for z in range(-8, -2):
            assert result.designed[result.grid.at(x, z)] == 90
            assert not any(p.x == x and p.z == z for p in result.cut)


def test_land_intersecting_a_bridge_end_melts_back_to_its_underside():
    terrain, palette = ground_at(lambda x, z: 75)
    plan = Plan(margin=20, reach=20, bridge=Bridge(natural_only=True), spans=(
        Span((0, 0), (14, 0), profile="cutting"),
        Span((15, 0), (29, 0), profile="bridge"),
    ))
    result = terraform(plan, terrain, straight_track(length=30), palette)
    # The saved hillside used to remain six blocks above the bridge's
    # carved footprint, producing a vertical edge directly beside its deck.
    assert result.designed[result.grid.at(16, 3)] <= 70
    assert result.smoothing_modes[result.grid.at(16, 3)] == Mode.MELT
    line = np.array([result.designed[result.grid.at(16, z)] for z in range(2, 12)])
    assert np.max(np.diff(np.floor(line + .5))) <= 1
    assert result.designed[result.grid.at(16, 20)] == 75


def test_an_explicit_tunnel_roof_is_kept_even_with_a_bed_only_alignment():
    terrain, palette = ground_at(lambda x, z: 95)
    alignment = straight_track(length=30)
    envelope = {(x, z): [70, 80] if x >= 10 else [70]
                for x in range(30) for z in range(-2, 3)}
    plan = Plan(margin=24, reach=16, spans=(
        Span((0, 0), (9, 0), profile="cutting"),
        Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE, headroom=0),
    ))
    result = terraform(plan, terrain, alignment, palette, envelope=envelope)
    for x in range(10, 30):
        assert result.designed[result.grid.at(x, 0)] >= 81
        assert BlockPos(x, 80, 0) not in result.cut


def test_fitting_protection_keeps_the_foot_but_releases_new_soil_faces_over_its_head():
    terrain, palette = ground_at(lambda x, z: 95)
    alignment = straight_track(length=30)
    envelope = {(x, z): [70, 80] if x < 10 else [70]
                for x in range(30) for z in range(-2, 3)}
    plan = Plan(margin=20, reach=16, headroom=0, preserve_fittings=("pier",),
                fitting_buffer=3, bridge=Bridge(natural_only=True), spans=(
                    Span((0, 0), (9, 0), profile="tunnel", intensity=Intensity.NONE, headroom=0),
                    Span((10, 0), (29, 0), profile="bridge"),
                ))
    result = terraform(plan, terrain, alignment, palette, envelope=envelope,
                       fittings={"pier": [BlockPos(14, 65, 0)], "pole": [BlockPos(8, 79, 0)]})
    assert 71 <= result.designed[result.grid.at(14, 0)] < 95
    assert 81 <= result.designed[result.grid.at(8, 0)] < 95
    assert BlockPos(14, 65, 0) not in result.cut
    assert BlockPos(8, 80, 0) not in result.cut


def test_final_slope_repair_cannot_restore_bank_fill_in_a_preserved_bridge_valley():
    from src.Terraform.plan import Embankment
    terrain, palette = ground_at(lambda x, z: 60 + (x % 3 == 0))
    plan = Plan(bleed=True, margin=20, reach=20, bridge=Bridge(natural_only=True),
                embankment=Embankment(shape="natural"), spans=(
                    Span((0, 0), (14, 0), profile="embankment"),
                    Span((15, 0), (29, 0), profile="bridge"),
                ))
    result = terraform(plan, terrain, straight_track(length=30), palette)
    for x in range(16, 29):
        for z in range(-6, 7):
            at = result.grid.at(x, z)
            assert result.designed[at] <= result.natural[at] + plan.bridge.max_change
            assert not any(p.x == x and p.z == z and p.y > result.natural[at] + 1
                           for p, _ in result.fill.iter_block_ids())


def test_water_stays_protected_at_a_bridge_tunnel_interface():
    from src.Core.block.states import BlockState
    from src.Terraform.plan import TerrainMask
    terrain, palette = ground_at(lambda x, z: 90 if x >= 10 else 60)
    water = BlockState.create("minecraft:water")
    for x in range(5, 15):
        for z in range(4, 7):
            # A pond high on the tunnel side also protects portal grading.
            top = 90 if x >= 10 else 60
            terrain.blocks.set_block(BlockPos(x, top + 1, z), water)
    terrain = FieldTerrain(terrain.blocks)
    plan = Plan(margin=20, reach=16, bridge=Bridge(natural_only=True),
                terrain_mask=TerrainMask(preserve_water=True), spans=(
                    Span((0, 0), (9, 0), profile="bridge"),
                    Span((10, 0), (29, 0), profile="tunnel", intensity=Intensity.NONE),
                ))
    result = terraform(plan, terrain, straight_track(length=30), palette)
    for x in range(5, 15):
        for z in range(4, 7):
            assert result.designed[result.grid.at(x, z)] == result.natural[result.grid.at(x, z)]


def test_tall_meetings_get_more_room_and_adaptation_can_be_disabled():
    alignment = straight_track(length=30)
    grid = Grid.around(alignment.columns, 30)
    xs = np.arange(grid.width)[None, :] + grid.origin.x
    station = np.broadcast_to(np.clip(xs, 0, 29), grid.shape).astype(float)
    ground = Ground(grid, grid.filled(95), grid.filled(0), grid.filled(70), grid.filled(80), station)
    spans = (Span((0, 0), (14, 0), profile="bridge"), Span((15, 0), (29, 0), profile="tunnel"))
    plan = Plan(margin=30, reach=30, spans=spans)
    automatic = context(plan, ground, alignment, plan.stretches(alignment.locate, 30))
    fixed = replace(plan, smoothing=Smoothing(adaptive_transitions=False))
    limited = context(fixed, ground, alignment, fixed.stretches(alignment.locate, 30))
    assert automatic.transition_weight[grid.at(15, 20)] > 0
    assert limited.transition_weight[grid.at(15, 20)] == 0
    assert automatic.transition_weight[grid.at(15, 32)] == 0


def test_config_reads_modes_limits_profiles_and_side_specific_transitions():
    options = read({"terraform": {"smoothing": {
        "behavior": "auto", "stable_change": .75, "decision_threshold": 2,
        "transition_reach": 16, "profiles": {"cutting": "melt"},
        "transitions": [{"from": "tunnel", "to": "bridge", "behavior": "stable",
                         "reach": 8, "side": "right", "bidirectional": False}],
    }}}).smoothing
    assert options.stable_change == .75
    assert options.profiles["cutting"] == "melt"
    assert options.transitions == (SmoothingTransition("tunnel", "bridge", "stable", 8, "right", False),)
    assert options.preserve_cliffs
    assert not read({"terraform": {"smoothing": {"preserve_cliffs": False}}}).smoothing.preserve_cliffs


@pytest.mark.parametrize("bad", [
    {"behavior": "random"}, {"stable_change": -1}, {"stable_change": float("nan")},
    {"decision_threshold": float("inf")}, {"transition_reach": -1},
    {"profiles": {"bridge": "random"}}, {"transitions": [{"from": "tunnel"}]},
    {"transitions": [{"from": "tunnel", "to": "bridge", "side": "up"}]},
    {"transitions": [{"from": "tunnel", "to": "bridge", "bidirectional": "yes"}]},
    {"stable_change": True}, {"adaptive_transitions": "yes"}, {"preserve_cliffs": "yes"},
    {"profiles": []}, {"transitions": {}}, {"profiles": {"typo": "melt"}},
    {"transitions": [{"from": "typo", "to": "bridge"}]},
])
def test_invalid_directional_smoothing_options_are_rejected(bad):
    with pytest.raises(ValueError):
        read({"terraform": {"smoothing": bad}})
